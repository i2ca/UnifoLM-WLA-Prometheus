#!/usr/bin/env python
"""Abre e fecha as duas garras Dex1 do sim e imprime comandado vs. medido ao longo do tempo.

Serve para conferir a correção do atrito (robots/unitree.py, friction 200 -> 0.5) e a
leitura do estado (gripper_state.py usa índices fixos [31, 29]): cada garra tem que
seguir o próprio comando, em menos de ~0,5 s.

    python wla/testa_garra_dex1.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ponte_wla_isaac import Comando, EstadoRobo  # noqa: E402
from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: E402

ChannelFactoryInitialize(1)
est, cmd = EstadoRobo(), Comando(80.0, 2.0)
while not est.pronto():
    time.sleep(0.1)

# (esquerda, direita): abre as duas, fecha só a direita, fecha só a esquerda, abre as duas
roteiro = [(5.0, 5.0), (5.0, 0.5), (0.5, 0.5), (0.5, 5.0), (5.0, 5.0)]
for e, d in roteiro:
    t0 = time.time()
    amostras = []
    while time.time() - t0 < 2.0:
        cmd.garra("left", e)
        cmd.garra("right", d)
        amostras.append((time.time() - t0, est.garra["left"], est.garra["right"]))
        time.sleep(1 / 30)
    a = np.array(amostras)
    em_05 = a[np.argmin(np.abs(a[:, 0] - 0.5))]
    print(f"cmd E {e:.1f} D {d:.1f} | medido em 0,5 s: E {em_05[1]:.2f} D {em_05[2]:.2f} | "
          f"em 2 s: E {a[-1, 1]:.2f} D {a[-1, 2]:.2f}", flush=True)
