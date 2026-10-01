#!/usr/bin/env python3
"""Testa SÓ o botão de pânico no Jetson do G1 — sem DDS, sem ponte, nada se mexe.

Usa a MESMA classe BotaoPanico da ponte v3: GPIO6 (PCC.03) vira entrada PRIMEIRO, depois
GPIO4 (PI.04) vai a 1. Imprime o valor lido e cada transição.

  0 = circuito fechado (botão solto)     -> normal   (GPIO4 em 0)
  1 = circuito aberto (botão apertado OU cabo solto; pull-up) -> PÂNICO

    python teste_botao_panico.py --segundos 30

Confira as duas coisas: apertar dá 0, soltar dá 1; e com um fio desconectado dá 0 (se der 1
ou ficar trocando, a entrada está flutuando: ponha um resistor de pull-down de 10 kΩ do
GPIO6 ao GND).
"""
import argparse
import importlib.util
import sys
import time
import types
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--segundos", type=float, default=30)
ap.add_argument("--ponte", default=str(Path(__file__).resolve().parent / "dex3_g1_server_v3_panico.py"))
ap.add_argument("--com-som", action="store_true", help="a cada aperto, fala o alerta e acende o LED (AudioClient)")
ap.add_argument("--volume", type=int, default=100)
ap.add_argument("--volume-bip", type=int, default=50)
a = ap.parse_args()

# Carrega só a classe BotaoPanico da ponte, sem importar o SDK da Unitree.
fonte = Path(a.ponte).read_text()
ini = fonte.index("class BotaoPanico:")
fim = fonte.index("def _limita(")
mod = types.ModuleType("botao")
exec("import ctypes, os, threading, time\n" + fonte[ini:fim], mod.__dict__)

alerta = None
if a.com_som:
    # Só o cliente de ÁUDIO no DDS: nenhum tópico de motor é aberto.
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(0)
    alerta = mod.AlertaSonoro(str(Path(a.ponte).resolve().parent), a.volume, a.volume_bip)
b = mod.BotaoPanico()
print("GPIO configurado: entrada PCC.03 (GPIO6), saída PI.04 (GPIO4) = 0. Aperte e solte o botão.", flush=True)
ultimo = None
t0 = time.time()
contagem = {0: 0, 1: 0}
try:
    while time.time() - t0 < a.segundos:
        v = b.le()
        contagem[v] += 1
        if v != ultimo:
            print(f"{time.time() - t0:6.2f} s  ->  {v}  ({'NORMAL' if v == 0 else 'PÂNICO'})", flush=True)
            if alerta is not None and ultimo is not None:
                (alerta.bloqueado if v == 1 else alerta.liberado)()
                (alerta.bip_liga if v == 1 else alerta.bip_desliga)()
            ultimo = v
        time.sleep(0.01)
finally:
    b.fecha()
print(f"leituras: 0 (normal) = {contagem[0]}, 1 (pânico) = {contagem[1]}")
