#!/usr/bin/env python
"""Teste SÓ DE LEITURA (29/10): a conta da gravidade da cintura (fk_g1.gravidade_cintura) bate com o torque que os
motores da cintura fazem de verdade? Lê do painel do robô (:8095/estado.json) a pose, a IMU e o tau_est, com o robô
parado em pé, e compara. Bate em sinal e tamanho (roll/pitch) -> dá para mandar esse tau como compensação.
Nada é enviado ao robô.

    python testa_gravidade_cintura.py [--painel http://192.168.123.164:8095] [--s 6]
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fk_g1 import FK  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--painel", default="http://192.168.123.164:8095")
ap.add_argument("--s", type=float, default=6.0)
a = ap.parse_args()
fk = FK()
extra = {l: (0.15, (0.05, 0.0, 0.04)) for l in ("left", "right")}   # câmera no punho (o mesmo do executor)
nomes = {12: "yaw", 13: "roll", 14: "pitch"}
calc, med, braco_c, braco_m = [], [], [], []
t0 = time.time()
while time.time() - t0 < a.s:
    e = json.loads(urllib.request.urlopen(f"{a.painel}/estado.json", timeout=3).read())
    js = e["corpo"]["juntas"]
    q = np.array([j["q"] for j in js[:29]])
    tau_m = np.array([j["tau"] for j in js[:29]])
    r, p, _ = np.radians(e["corpo"]["imu"]["rpy_graus"])
    cr, sr, cp, sp = np.cos(r), np.sin(r), np.cos(p), np.sin(p)
    g = np.array([[cp, sp * sr, sp * cr], [0, cr, -sr], [-sp, cp * sr, cp * cr]]).T @ np.array([0, 0, -9.81])
    tc = fk.gravidade_cintura(q, g, extra)
    calc.append([tc[i] for i in (12, 13, 14)])
    med.append(tau_m[[12, 13, 14]])
    tb = fk.gravidade(q, g, extra)
    braco_c.append(tb[[15, 18, 22, 25]])
    braco_m.append(tau_m[[15, 18, 22, 25]])
    time.sleep(0.4)
c, m = np.mean(calc, 0), np.mean(med, 0)
print(f"{len(calc)} amostras | IMU: {e['corpo']['imu']['rpy_graus'][0]:+.1f}° de lado, {e['corpo']['imu']['rpy_graus'][1]:+.1f}° frente")
print("CINTURA        calculado   medido (tau_est)")
for k, i in enumerate((12, 13, 14)):
    print(f"  {nomes[i]:6s}      {c[k]:+7.2f} Nm   {m[k]:+7.2f} Nm   {'mesmo sinal' if np.sign(c[k]) == np.sign(m[k]) else 'SINAL OPOSTO'}")
bc, bm = np.mean(braco_c, 0), np.mean(braco_m, 0)
print("referência (braços, conta já validada em 02/10):")
for k, nome in enumerate(("ombro pitch E", "cotovelo E", "ombro pitch D", "cotovelo D")):
    print(f"  {nome:14s} {bc[k]:+7.2f} Nm   {bm[k]:+7.2f} Nm")
