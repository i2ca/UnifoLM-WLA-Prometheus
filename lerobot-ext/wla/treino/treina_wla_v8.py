#!/usr/bin/env python
"""WLA v8 (31/10): LoRA em cima do v7 (passo 20000) com os nossos 3 datasets + 7 de TERCEIROS (G1+Dex3, HuggingFace,
converte_terceiros_wla.py). Tudo do v7 (aumento de imagem, câmeras desligadas, troca de frase, pesos, validação nos
MESMOS episódios — cache do v7), mais:

  - LoRA nas camadas de LINGUAGEM do VLM (r 64) + DiT inteiro + PROJETOR DE ESTADO treinando (o v4 congelava o
    projetor junto com o VLM: foi a causa do "ignora o estado"; aqui ele é descongelado de propósito);
  - mistura por ENTRADA (wla_v2.FONTES): os nossos repetidos até ~metade da época; dentro de cada entrada de terceiros
    equilíbrio pela raiz (alfa 0,5: o dataset de 8 mil quadros não repete 30x ao lado do de 260 mil);
  - frase RIVAL da troca de frase (trainer.v2.rivais_extra): cada frase -> lista de rivais, sorteada a cada passo
    (pera <-> pêssego/tomate/limão, polvo rosa <-> azul; tarefas sozinhas -> uma tarefa de OUTRO dataset).

    cd ~/DEV/unifolm-wla-lora && accelerate launch ... lerobot-ext/wla/treino/treina_wla_v8.py \\
        --config_yaml unifolm_wla/config/training/prometheus_v8.yaml
"""
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))

from omegaconf import OmegaConf  # noqa: E402

import treina_wla_v7 as V7  # noqa: E402  (lr por módulo, passo com a troca de frase, validação, LoRA/partir_de)
import wla_v2  # noqa: E402

T = V7.T


class Rivais(dict):
    """frase -> [rivais]; get sorteia um (o _passo do v7 chama self._rivais.get(t, t))."""

    def get(self, k, padrao=None):
        v = super().get(k)
        if not v:
            return padrao
        return random.choice(v) if isinstance(v, list) else v


_rivais_v7 = V7._rivais_da_config


def _rivais_da_config(cfg):
    base = _rivais_v7(cfg)   # as nossas (caneca <-> maçã, coador <-> pegar caneca)
    r = Rivais({k: [v] for k, v in base.items()})
    extra = OmegaConf.to_container(cfg.trainer.v2.get("rivais_extra", {}) or {}, resolve=True)
    for k, v in extra.items():
        r[k] = list(v) if isinstance(v, list) else [v]
    print(f"🔀 frases com rival: {len(r)} ({len(extra)} de terceiros)", flush=True)
    return r


V7._rivais_da_config = _rivais_da_config

# o projetor de estado TREINA junto com o LoRA do VLM (o init do co-treino congela tudo do VLM que não é LoRA)
_init_v7 = T.VLATrainer.init_checkpoint_and_lora


def init_checkpoint_and_lora(self):
    _init_v7(self)
    n = 0
    for k, p in self.model.qwen_vl_interface.named_parameters():
        if "robot_state_projector" in k and not p.requires_grad:
            p.requires_grad = True
            n += p.numel()
    print(f"🦾 projetor de estado descongelado: {n / 1e6:.2f} M parâmetros", flush=True)


T.VLATrainer.init_checkpoint_and_lora = init_checkpoint_and_lora


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_yaml", required=True)
    args, resto = ap.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config_yaml), OmegaConf.from_dotlist(T.normalize_dotlist_args(resto)))
    v2 = cfg.trainer.get("v2", {}) or {}
    cont = (lambda x: OmegaConf.to_container(x, resolve=True) if x else None)   # noqa: E731
    wla_v2.FONTES.update(cont(v2.get("fontes", None)) or {})
    if v2.get("split"):
        wla_v2.instala_split(v2["split"], "treino")
    wla_v2.instala_balanceamento(float(v2.get("balanceia", 1.0)), cont(v2.get("frases", None)), None,
                                 float(v2.get("texto_vazio", 0.0)), None,
                                 cont(v2.get("aumento_imagem", None)), cont(v2.get("desliga_camera", None)))
    wla_v2.instala_pesos_acao(cont(v2.get("pesos", None)) or {})
    cfg = T.apply_config_compat(cfg)
    cfg.config_yaml = args.config_yaml
    T.main(cfg)
