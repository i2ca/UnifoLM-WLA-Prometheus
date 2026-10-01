#!/usr/bin/env python3
"""Diagnóstico do botão NC entre GPIO4 (PI.04) e GPIO6 (PCC.03): com o GPIO4 em 0 e depois
em 1, o que o GPIO6 lê com o botão fechado/aberto? Sem DDS, sem ponte — nada se mexe.

    python diagnostico_botao.py --saida 0 --segundos 20
"""
import argparse
import ctypes
import time

ap = argparse.ArgumentParser()
ap.add_argument("--saida", type=int, default=0, help="valor do GPIO4 (0 ou 1)")
ap.add_argument("--segundos", type=float, default=20)
a = ap.parse_args()

L = ctypes.CDLL("libgpiod.so.2", use_errno=True)
L.gpiod_line_find.restype = ctypes.c_void_p
L.gpiod_line_find.argtypes = [ctypes.c_char_p]
for f in ("gpiod_line_request_input",):
    getattr(L, f).argtypes = [ctypes.c_void_p, ctypes.c_char_p]
L.gpiod_line_request_output.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
L.gpiod_line_get_value.argtypes = [ctypes.c_void_p]
L.gpiod_line_release.argtypes = [ctypes.c_void_p]
L.gpiod_line_close_chip.argtypes = [ctypes.c_void_p]

ent = L.gpiod_line_find(b"PCC.03")
sai = L.gpiod_line_find(b"PI.04")
assert ent and sai, "linhas não encontradas"
assert L.gpiod_line_request_input(ent, b"diag-botao") == 0, "entrada ocupada (a ponte está rodando?)"
assert L.gpiod_line_request_output(sai, b"diag-botao", a.saida) == 0, "saída ocupada"
print(f"GPIO4 = {a.saida}. Lendo GPIO6 por {a.segundos:.0f} s — aperte e solte o botão.", flush=True)
t0, ult, cont = time.time(), None, {0: 0, 1: 0}
try:
    while time.time() - t0 < a.segundos:
        v = L.gpiod_line_get_value(ent)
        cont[v] = cont.get(v, 0) + 1
        if v != ult:
            print(f"  {time.time() - t0:5.1f} s  GPIO6 = {v}", flush=True)
            ult = v
        time.sleep(0.005)
finally:
    for lin in (sai, ent):
        L.gpiod_line_release(lin)
        L.gpiod_line_close_chip(lin)
print(f"  contagem: {cont}")
