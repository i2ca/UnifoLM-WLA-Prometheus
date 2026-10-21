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
    wla_v2.instala_balanceamento(float(v2.get("balanceia", 1.0)),
                                 OmegaConf.to_container(frases, resolve=True) if frases else None)
    pesos = v2.get("pesos", None)
    wla_v2.instala_pesos_acao(OmegaConf.to_container(pesos, resolve=True) if pesos else {})
    cfg = C.T.apply_config_compat(cfg)
    cfg.config_yaml = args.config_yaml
    C.T.main(cfg)
