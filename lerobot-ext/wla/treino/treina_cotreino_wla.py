#!/usr/bin/env python
"""Treino do WLA com o VLM ACORDADO e co-treino de perguntas (03/10) — por cima do train_unifolm_wla.py oficial.

Por quê: no LoRA só do DiT (VLM congelado) o modelo decidia pela IMAGEM/POSIÇÃO e não pela FRASE ("pegar caneca"
com a mão perto da maçã ia na maçã). Aqui, sem mexer no código da Unitree (monkeypatch):

  1. LoRA TAMBÉM no VLM — só nas camadas de LINGUAGEM (q/k/v/o), a visão fica intacta. O grupo de lr do
     qwen_vl_interface já é 10x menor que o do DiT (build_param_lr_groups). Config:
         trainer.freeze_modules: ""                      (o peft congela a base sozinho; o resto é congelado aqui)
         trainer.lora.qwen_vl_interface: {enabled: true, r: 16, lora_alpha: 32, lora_dropout: 0.05,
                                          target_modules: ".*language_model.*\\.(q_proj|k_proj|v_proj|o_proj)"}
     (string = regex do peft; o apply_lora_adapters oficial só aceitava lista.)

  2. CO-TREINO de perguntas do ER-1 (tarefa concluída? segurando? no lugar?) nas NOSSAS imagens: a cada passo,
     além da perda da ação, um lote de perguntas vai ao VLM e a perda de linguagem entra com peso:
         trainer.cotreino.jsonl: /data/mrwlker/er1_lora/dados/treino.jsonl   (do gera_dataset_er1.py)
         trainer.cotreino.peso: 0.1   trainer.cotreino.lote: 4
     O VLM é obrigado a reconhecer "caneca branca" x "maçã vermelha" e o estado da mão pelo NOME.

  3. Partir de um treino anterior (LoRA do DiT já afinado) com o LoRA do VLM novo:
         trainer.partir_de: <.../checkpoints/steps_N_model.safetensors>   (passos recomeçam em 0)

    cd ~/DEV/unifolm-wla-lora && accelerate launch ... lerobot-ext/wla/treino/treina_cotreino_wla.py \\
        --config_yaml unifolm_wla/config/training/lora_prometheus_dex3_cotreino.yaml ...
"""
import json
import os
import random
import sys

sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
from PIL import Image  # noqa: E402

import unifolm_wla.training.train_unifolm_wla as T  # noqa: E402
from unifolm_wla.training.trainer_utils import trainer_tools as TT  # noqa: E402


# ── 1. LoRA com target_modules regex (string) ───────────────────────────────────────────────────────────
def apply_lora_adapters(model, lora_cfg):
    if not lora_cfg or not lora_cfg.get("enabled", False):
        return model
    from peft import LoraConfig, inject_adapter_in_model
    alvos = {"qwen_vl_interface": lambda: model.qwen_vl_interface.model, "action_model": lambda: model.action_model.model}
    for nome, pega in alvos.items():
        sub = lora_cfg.get(nome, None)
        if not sub or not sub.get("enabled", False):
            continue
        tm = sub.get("target_modules", [])
        tm = tm if isinstance(tm, str) else list(tm)
        cfg = LoraConfig(r=sub.get("r", 16), lora_alpha=sub.get("lora_alpha", 32),
                         lora_dropout=sub.get("lora_dropout", 0.05), target_modules=tm, bias=sub.get("bias", "none"))
        inject_adapter_in_model(cfg, pega())
        n = sum(p.numel() for k, p in pega().named_parameters() if "lora_" in k)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"🧩 LoRA em `{nome}`: r={cfg.r}, alvos={tm}, {n / 1e6:.1f} M parâmetros", flush=True)
    return model


TT.TrainerUtils.apply_lora_adapters = staticmethod(apply_lora_adapters)

# ── 3. partir de um checkpoint com LoRA já injetado + congelar a base do VLM ────────────────────────────
_orig_init = T.VLATrainer.init_checkpoint_and_lora


def init_checkpoint_and_lora(self):
    partir = getattr(self.config.trainer, "partir_de", None)
    if partir and not getattr(self.config.trainer, "is_resume", False):
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # ORDEM (04/10): o checkpoint tem o DiT com LoRA (chaves renomeadas: to_q.base_layer...) e o VLM SEM LoRA
        # (q_proj.weight). Injetar os dois antes de carregar renomeava também o VLM e os pesos dele NÃO entravam
        # (perda das perguntas 7,0 em vez de ~0,1). Então: LoRA só no DiT -> carrega -> LoRA no VLM.
        from omegaconf import OmegaConf
        lora = OmegaConf.to_container(self.config.trainer.lora, resolve=True)
        so = lambda nome: {"enabled": True, nome: lora.get(nome)}   # noqa: E731
        self.model = self.apply_lora_adapters(self.model, so("action_model"))
        self.model = self.load_pretrained_backbones(self.model, partir, reload_modules=None)
        self.model = self.apply_lora_adapters(self.model, so("qwen_vl_interface"))
        self.completed_steps, self.resume_from_checkpoint = 0, partir
        print(f"▶ partindo de {partir} (LoRA do DiT afinado; LoRA do VLM novo)", flush=True)
    else:
        _orig_init(self)
    # o VLM só treina o LoRA (o projetor de estado e o resto ficam como estão)
    n = 0
    for k, p in self.model.qwen_vl_interface.named_parameters():
        if "lora_" not in k and p.requires_grad:
            p.requires_grad = False
            n += 1
    treina = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    print(f"VLM: {n} tensores da base congelados | treináveis no total: {treina / 1e6:.1f} M", flush=True)


T.VLATrainer.init_checkpoint_and_lora = init_checkpoint_and_lora


# ── 2. co-treino de perguntas ───────────────────────────────────────────────────────────────────────────
class Perguntas:
    def __init__(self, caminho, lote, tamanho):
        self.linhas = [json.loads(x) for x in open(caminho)]
        self.lote, self.tamanho = lote, tamanho          # tamanho = (H, W) das imagens do treino
        self.ordem, self.i = [], 0
        self.rnd = random.Random(1234 + (torch.distributed.get_rank() if torch.distributed.is_initialized() else 0))

    def proximo(self):
        if self.i + self.lote > len(self.ordem):
            self.ordem = list(range(len(self.linhas)))
            self.rnd.shuffle(self.ordem)
            self.i = 0
        ls = [self.linhas[j] for j in self.ordem[self.i:self.i + self.lote]]
        self.i += self.lote
        h, w = self.tamanho
        imgs = [[Image.open(p).convert("RGB").resize((w, h)) for p in l["imagens"]] for l in ls]
        return imgs, [l["pergunta"] for l in ls], [l["resposta"] for l in ls]


_orig_step = T.VLATrainer._train_step


def _train_step(self, batch_vla, batch_vlm=None):
    co = getattr(self.config.trainer, "cotreino", None)
    if not co or not co.get("jsonl"):
        return _orig_step(self, batch_vla, batch_vlm)
    if not hasattr(self, "_perguntas"):
        tam = tuple(getattr(self.config.trainer, "cotreino_imagem", [336, 448]))
        self._perguntas = Perguntas(co["jsonl"], int(co.get("lote", 4)), tam)
    peso = float(co.get("peso", 0.1))
    iface = self.accelerator.unwrap_model(self.model).qwen_vl_interface
    with self.accelerator.accumulate(self.model):
        self.optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model.forward(batch_vla)
            action_loss = out["action_loss"]
            imgs, qs, rs = self._perguntas.proximo()
            ent = iface.build_qwenvl_inputs(images=imgs, instructions=qs, solutions=rs)
            ent.pop("assistant_mask", None)
            vlm_loss = iface(**ent).loss
            total = action_loss + peso * vlm_loss
        self.accelerator.backward(total)
        if self.config.trainer.gradient_clipping is not None:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
        self.optimizer.step()
        if self.accelerator.sync_gradients:
            self.lr_scheduler.step()
    r = {"action_dit_loss": action_loss.item(), "vqa_loss": vlm_loss.item()}
    r.update({k: out[k].item() for k in ("flow_action_loss", "lm_ce_loss") if k in out})
    return r


T.VLATrainer._train_step = _train_step

if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_yaml", required=True)
    args, resto = ap.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config_yaml),
                          OmegaConf.from_dotlist(T.normalize_dotlist_args(resto)))
    cfg = T.apply_config_compat(cfg)
    cfg.config_yaml = args.config_yaml
    T.main(cfg)
