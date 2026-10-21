#!/usr/bin/env python
"""Captura para a calibração da câmera da cabeça (21/10): SÓ LÊ (câmera + juntas), não manda nada ao robô.
Grava <n>_cabeca.jpg + <n>.npz (q_medido) no formato das rodadas do executor, para o `servo_er1.py calibra`.
Mexa o braço direito (mão ABERTA, palma para a câmera) por várias posições enquanto roda.

    python captura_calibra.py --segundos 90 --saida ~/wla_real_runs/calibra_$(date +%H%M%S)
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import zmq

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "robo_g1"))
import sombra_wla as sw  # noqa: E402
from teste_cotovelo import Estado  # noqa: E402


class _SemCabine:
    def publica_quadro(self, *a, **k):
        pass


ap = argparse.ArgumentParser()
ap.add_argument("--robo", default="192.168.123.164")
ap.add_argument("--segundos", type=float, default=90)
ap.add_argument("--intervalo", type=float, default=0.5)
ap.add_argument("--saida", required=True)
a = ap.parse_args()
out = Path(a.saida).expanduser()
out.mkdir(parents=True, exist_ok=True)
ctx = zmq.Context.instance()
est = Estado(ctx, a.robo)
cam = sw.Camera(a.robo, 5555, _SemCabine(), "head_stereo_left")
t0 = time.time()
while (est.low is None or cam.rgb is None) and time.time() - t0 < 8:
    time.sleep(0.05)
if est.low is None or cam.rgb is None:
    sys.exit(f"sem lowstate={est.low is None} câmera={cam.rgb is None}")
print(f"capturando {a.segundos:.0f} s em {out}", flush=True)
t0, n = time.time(), 0
while time.time() - t0 < a.segundos:
    if time.time() - cam.t < 0.3:
        q = np.array([est.q(i) for i in range(29)])
        cv2.imwrite(str(out / f"{n:04d}_cabeca.jpg"), cam.rgb[:, :, ::-1])
        np.savez(out / f"{n:04d}.npz", q_medido=q)
        n += 1
        if n % 10 == 0:
            print(f"  {n} quadros ({time.time() - t0:.0f} s)", flush=True)
    time.sleep(a.intervalo)
print(f"pronto: {n} quadros", flush=True)
