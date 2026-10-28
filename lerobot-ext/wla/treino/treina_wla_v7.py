#!/usr/bin/env python
"""Treino v7 do WLA (28/10): MODELO NOVO a partir do UnifoLM-WLA-1.0-Base, TUDO descongelado (com o projetor de
estado, que do v4 ao v6 ficou congelado sem querer). Em cima do treina_wla_v2.py (validação, balanceamento, peso por
parte, co-treino de perguntas), mais:

  - lr POR MÓDULO sem conflito: o build_param_lr_groups da Unitree não descarta um parâmetro que já entrou num grupo
    (dois grupos com o mesmo parâmetro = erro do otimizador); aqui o 1º grupo que pega o parâmetro fica com ele;
  - FRASE TROCADA (trainer.v2.lingua_contraste): numa parte dos passos o mesmo lote passa de novo com a frase de
    outra tarefa, com o MESMO ruído/tempo do fluxo (mesma semente); a perda com a frase certa tem que ficar menor que
    com a trocada por uma margem RELATIVA: max(0, margem * L_certa + L_certa - L_trocada). Ignorar o texto passa a
    custar; com o texto já sendo usado (L_trocada bem maior), a perda some;
  - AUMENTO DE IMAGEM e CÂMERAS DESLIGADAS (trainer.v2.aumento_imagem / desliga_camera -> wla_v2.Balanceado);
  - TESTE CURTO (trainer.teste_curto N): para em N passos (NaN? memória?) antes do treino longo.

    cd ~/DEV/unifolm-wla-lora && accelerate launch ... treina_wla_v7.py --config_yaml .../prometheus_v7.yaml
"""
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import treina_wla_v2 as V2  # noqa: E402  (instala validação + checkpointing do VLM)
import wla_v2  # noqa: E402

C, T = V2.C, V2.C.T


# ── lr por módulo sem parâmetro repetido ────────────────────────────────────────────────────────────────────────
def build_param_lr_groups(model, cfg):
    lr_cfg = cfg.trainer.learning_rate
    usados, grupos = set(), []
    for nome, lr in lr_cfg.items():
        if nome == "base":
            continue
        mod = model
        try:
            for a in nome.split("."):
                mod = getattr(mod, a)
        except AttributeError:
            raise SystemExit(f"learning_rate: módulo {nome!r} não existe no modelo")
        ps = [p for p in mod.parameters() if p.requires_grad and id(p) not in usados]
        if ps:
            grupos.append({"params": ps, "lr": float(lr), "name": nome})
            usados.update(id(p) for p in ps)
            print(f"📐 lr {float(lr):.1e} -> {nome} ({sum(p.numel() for p in ps) / 1e6:.1f} M)", flush=True)
    resto = [p for p in model.parameters() if p.requires_grad and id(p) not in usados]
    if resto:
        grupos.append({"params": resto, "lr": float(lr_cfg.get("base", 1e-4)), "name": "base"})
        print(f"📐 lr {float(lr_cfg.get('base', 1e-4)):.1e} -> resto ({sum(p.numel() for p in resto) / 1e6:.1f} M)",
              flush=True)
    return grupos


T.build_param_lr_groups = build_param_lr_groups
C.TT.build_param_lr_groups = build_param_lr_groups


# ── passo de treino: ação + perguntas + FRASE TROCADA ───────────────────────────────────────────────────────────
def _rivais_da_config(cfg):
    fr = OmegaConf.to_container(cfg.trainer.v2.get("frases", {}) or {}, resolve=True)
    todas = sorted({f for v in fr.values() for f in (v if isinstance(v, list) else [v])} | set(fr))
    riv = wla_v2.rivais(todas)   # caneca <-> maçã (a mesma mesa: só a frase diz qual pegar)
    # o COADOR troca com PEGAR A CANECA (mesmo objeto, outra ação): foi a confusão no robô (28/10: com a caneca na
    # mão ele voltava a "pegar")
    pega = [f for f in todas if wla_v2.tarefa(f) == "caneca"]
    for f in todas:
        if wla_v2.tarefa(f) == "coador" and pega:
            riv[f] = max(pega, key=lambda g: len(set(f.lower().split()) & set(g.lower().split())))
    return riv


def _passo(self, batch_vla, batch_vlm=None):
    cfg = self.config
    v2 = cfg.trainer.get("v2", {}) or {}
    lc = v2.get("lingua_contraste", None) or {}
    co = getattr(cfg.trainer, "cotreino", None) or {}
    n = self.completed_steps
    curto = int(cfg.trainer.get("teste_curto", 0) or 0)
    if curto and n >= curto:
        print(f"\n🧪 TESTE CURTO: {curto} passos sem NaN — parando aqui (lance de novo sem trainer.teste_curto)",
              flush=True)
        self.accelerator.wait_for_everyone()
        sys.exit(0)
    usa_l = bool(lc) and n >= int(lc.get("comeca_em", 1000)) and random.random() < float(lc.get("prob", 0.5))
    if not hasattr(self, "_rivais"):
        self._rivais = _rivais_da_config(cfg)
    if co.get("jsonl") and not hasattr(self, "_perguntas"):
        self._perguntas = C.Perguntas(co["jsonl"], int(co.get("lote", 4)), tuple(cfg.trainer.get("cotreino_imagem",
                                                                                               [336, 448])))
    dev = self.accelerator.device
    semente = random.randrange(2 ** 31)
    r = {}
    with self.accelerator.accumulate(self.model):
        self.optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.random.fork_rng(devices=[dev]):
                torch.manual_seed(semente)
                out = self.model.forward(batch_vla)
            perda = out["action_loss"]
            total = perda
            if usa_l:
                bt = dict(batch_vla)
                bt["task"] = [self._rivais.get(t, t) for t in batch_vla["task"]]
                with torch.random.fork_rng(devices=[dev]):
                    torch.manual_seed(semente)   # o mesmo ruído e tempo do fluxo: só a frase muda
                    trocada = self.model.forward(bt)["action_loss"]
                m = float(lc.get("margem", 0.2))
                l_ling = torch.relu(m * perda.detach() + perda - trocada)
                total = total + float(lc.get("peso", 1.0)) * l_ling
                r.update({"lingua_loss": l_ling.item(), "lingua_razao_treino": (trocada / perda.clamp(min=1e-6)).item()})
            if co.get("jsonl"):
                iface = self.accelerator.unwrap_model(self.model).qwen_vl_interface
                imgs, qs, rs = self._perguntas.proximo()
                ent = iface.build_qwenvl_inputs(images=imgs, instructions=qs, solutions=rs)
                ent.pop("assistant_mask", None)
                vlm = iface(**ent).loss
                total = total + float(co.get("peso", 0.1)) * vlm
                r["vqa_loss"] = vlm.item()
        if not math.isfinite(total.item()):
            raise SystemExit(f"❌ perda não finita no passo {n}: ação {perda.item()} (NaN) — pare e revise os lr")
        self.accelerator.backward(total)
        if cfg.trainer.gradient_clipping is not None:
            self.accelerator.clip_grad_norm_(self.model.parameters(), cfg.trainer.gradient_clipping)
        self.optimizer.step()
        if self.accelerator.sync_gradients:
            self.lr_scheduler.step()
    r["action_dit_loss"] = perda.item()
    r.update({k: out[k].item() for k in ("flow_action_loss", "lm_ce_loss") if k in out})
    return r


T.VLATrainer._train_step = _passo


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_yaml", required=True)
    args, resto = ap.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config_yaml), OmegaConf.from_dotlist(T.normalize_dotlist_args(resto)))
    v2 = cfg.trainer.get("v2", {}) or {}
    if v2.get("split"):
        wla_v2.instala_split(v2["split"], "treino")
    cont = (lambda x: OmegaConf.to_container(x, resolve=True) if x else None)   # noqa: E731
    wla_v2.instala_balanceamento(float(v2.get("balanceia", 1.0)), cont(v2.get("frases", None)), None,
                                 float(v2.get("texto_vazio", 0.0)), None,
                                 cont(v2.get("aumento_imagem", None)), cont(v2.get("desliga_camera", None)))
    wla_v2.instala_pesos_acao(cont(v2.get("pesos", None)) or {})
    cfg = T.apply_config_compat(cfg)
    cfg.config_yaml = args.config_yaml
    T.main(cfg)
