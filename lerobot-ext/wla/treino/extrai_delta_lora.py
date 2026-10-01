#!/usr/bin/env python
"""Extrai de um checkpoint afinado (LoRA) SÓ o que mudou em relação ao UnifoLM-WLA-1.0-Base.

O `*_adapter.safetensors` que o treino salva tem só os tensores `lora_*`. Mas com o LoRA no DiT
(`action_model.model`), o RESTO do especialista de ação — codificadores de estado/ação, decodificador,
embeddings — continua treinável e muda também: o log mostra 24 M parâmetros treinados. Aqui entram:
  - todo tensor `lora_*`;
  - todo tensor cujo valor difere do base (nomes do LoRA "x.base_layer.weight" comparados com "x.weight").
Resultado: dezenas de MB em vez dos 12,5 GB do checkpoint completo — dá para levar pela VPN.

    python extrai_delta_lora.py <base>/checkpoints/model.safetensors <run>/checkpoints/steps_8000_model.safetensors \\
        <run>/checkpoints/steps_8000_delta.safetensors
"""
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

base_arq, fino_arq, saida = sys.argv[1:4]
base = safe_open(base_arq, "pt")
chaves_base = set(base.keys())
fino = safe_open(fino_arq, "pt")
delta, iguais, sem_par = {}, 0, []
for k in fino.keys():
    t = fino.get_tensor(k)
    if "lora_" in k:
        delta[k] = t
        continue
    kb = k.replace(".base_layer.", ".")
    if kb not in chaves_base:
        sem_par.append(k)
        delta[k] = t
        continue
    tb = base.get_tensor(kb)
    if t.shape != tb.shape or not torch.equal(t, tb.to(t.dtype)):
        delta[k] = t
    else:
        iguais += 1
n = sum(v.numel() for v in delta.values())
save_file(delta, saida)
print(f"delta: {len(delta)} tensores, {n / 1e6:.2f} M parâmetros ({sum(v.numel() * v.element_size() for v in delta.values()) / 1e6:.1f} MB)"
      f" | iguais ao base: {iguais} | sem par no base: {len(sem_par)} {sem_par[:3]}")
por_modulo = {}
for k, v in delta.items():
    m = ".".join(k.split(".")[:3]) if "lora_" not in k else "LoRA"
    por_modulo[m] = por_modulo.get(m, 0) + v.numel()
for m, c in sorted(por_modulo.items(), key=lambda x: -x[1])[:12]:
    print(f"  {m:55s} {c / 1e6:7.2f} M")
