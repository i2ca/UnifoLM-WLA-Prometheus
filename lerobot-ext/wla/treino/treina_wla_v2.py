#!/usr/bin/env python
"""Treino v2 do WLA (21/10): do ZERO a partir do UnifoLM-WLA-1.0-Base, TUDO descongelado, como o pré-treino da Unitree
(unifolm_wla_pretrain_multisource.yaml: freeze_modules "", VLM com lr 10x menor), mais:

  - VALIDAÇÃO de verdade: 5 episódios por tarefa fora do treino (split.json do descreve_datasets.py); episódios
    suspeitos (tentativas que falharam, pega no 1º segundo...) fora dos dois;
  - TAREFAS balanceadas (wla_v2.instala_balanceamento);
  - PESO NA AÇÃO: perda por parte (mão direita 3, dedos direitos 2, pernas/base ~0) — wla_v2.instala_pesos_acao;
  - CO-TREINO das perguntas do ER-1 (opcional, trainer.cotreino) — do treina_cotreino_wla.py.

Config (além do treino normal):
    trainer.v2.split: /data/mrwlker/wla_v2/descricao/split.json
    trainer.v2.balanceia: 1.0                 (0 = proporcional aos quadros, como antes)
    trainer.v2.pesos: {right_xyz_rotvec: 3.0, ...}   (o que faltar vem de wla_v2.PESOS_PADRAO)

    cd ~/DEV/unifolm-wla-lora && accelerate launch ... lerobot-ext/wla/treino/treina_wla_v2.py \\
        --config_yaml unifolm_wla/config/training/prometheus_v2.yaml ...
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))

import treina_cotreino_wla as C  # noqa: E402  (LoRA regex, partir_de, co-treino de perguntas)
import wla_v2  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

def distratores_da_config(dz):
    """v5: {banco: recortes.pt, prob, max} -> o banco carregado (vai junto para os leitores)."""
    if not dz:
        return None
    import torch
    return {"banco": torch.load(dz["banco"], weights_only=False)["img"], "prob": float(dz.get("prob", 0.7)),
            "max": int(dz.get("max", 3))}


_init_ck = C.T.VLATrainer.init_checkpoint_and_lora


def init_checkpoint_and_lora(self):
    """Com o VLM inteiro treinando, as ativações dele não cabem nos 80 GB (lote 4 estourou): recalcula no backward."""
    _init_ck(self)
    if (self.config.trainer.get("v2", {}) or {}).get("checkpoint_vlm", True):
        self.model.qwen_vl_interface.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.qwen_vl_interface.model.config.use_cache = False
        print("♻️  gradient checkpointing no VLM", flush=True)


C.T.VLATrainer.init_checkpoint_and_lora = init_checkpoint_and_lora

# ── VALIDAÇÃO DENTRO DO TREINO (21/10): a cada trainer.v2.val_intervalo passos (e no passo 0 = o modelo base), o
# processo 0 roda o modelo que está treinando nos episódios de VALIDAÇÃO (lotes fixos) e manda val/* ao wandb:
# erro da mão direita em cm, dedos, fecha no tempo e a troca de frase (lingua_%). Sem outra GPU nem outra cópia.
_log = C.T.VLATrainer._log_metrics


def _log_metrics(self, metrics):
    import torch.distributed as dist
    v2 = self.config.trainer.get("v2", {}) or {}
    iv, s = int(v2.get("val_intervalo", 500)), self.completed_steps
    if iv > 0 and v2.get("split") and s % iv == 0 and getattr(self, "_val_feito", -1) != s:
        self._val_feito = s
        if dist.get_rank() == 0:
            import time
            t0 = time.time()
            m = self.accelerator.unwrap_model(self.model)
            if not hasattr(self, "_val_lotes"):
                fr = v2.get("frases", None)
                fr = OmegaConf.to_container(fr, resolve=True) if fr else None
                fixas = {k: (v[0] if isinstance(v, list) else v) for k, v in (fr or {}).items()}
                self._val_lotes = wla_v2.lotes_validacao(self.config.datasets.vla_data.data_config_path,
                                                         int(v2.get("val_lotes", 24)), int(v2.get("val_lote", 4)), fixas,
                                                         v2.get("val_cache"))
            era = m.training
            m.eval()
            try:
                r = wla_v2.metricas_validacao(m, self._val_lotes, bool(v2.get("val_troca_frase", True)))
            finally:
                m.train(era)
            metrics.update({f"val/{k}": v for k, v in r.items()})
            print(f"🧪 validação passo {s} ({time.time() - t0:.0f} s): " +
                  " | ".join(f"{k} {v:.2f}" for k, v in r.items() if "/" not in k), flush=True)
    _log(self, metrics)


C.T.VLATrainer._log_metrics = _log_metrics

if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_yaml", required=True)
    args, resto = ap.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config_yaml), OmegaConf.from_dotlist(C.T.normalize_dotlist_args(resto)))
    v2 = cfg.trainer.get("v2", {}) or {}
    if v2.get("split"):
        wla_v2.instala_split(v2["split"], "treino")
    frases = v2.get("frases", None)
    mk = v2.get("marcador", None)   # v4: {alvos: alvos.jsonl do ER-1, episodios: episodios.jsonl}
    wla_v2.instala_balanceamento(float(v2.get("balanceia", 1.0)),
                                 OmegaConf.to_container(frases, resolve=True) if frases else None,
                                 wla_v2.Marcadores(mk["alvos"], mk["episodios"]) if mk else None,
                                 float(v2.get("texto_vazio", 0.0)),
                                 distratores_da_config(v2.get("distratores", None)))
    pesos = v2.get("pesos", None)
    wla_v2.instala_pesos_acao(OmegaConf.to_container(pesos, resolve=True) if pesos else {})
    cfg = C.T.apply_config_compat(cfg)
    cfg.config_yaml = args.config_yaml
    C.T.main(cfg)
