#!/usr/bin/env python
"""Descobre o referencial de `left/right_wrist_eef_9d` do embodiment REAL_G1 do GR00T N1.7.

O código do N1.7 não define essa pose. Comparamos as estatísticas do modelo base
(statistics.json -> real_g1_relative_eef_relative_joints -> state) com a FK de dados reais de
G1+Dex3 da NVIDIA (PhysicalAI-Robotics-GR00T-Teleop-G1 / g1-pick-apple), para cada candidato
de base x ponta x convenção do rot6d. O candidato certo tem média ~igual e desvio ~igual.

    python ajusta_eef_n17.py --stats /tmp/n17_stats.json --dados ~/wla_dados/nvidia_g1_teleop/g1-pick-apple \
        --urdf ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/assets/g1/g1_body29_hand14.urdf
"""
import argparse
import glob

import numpy as np
import pandas as pd
import pinocchio as pin
import json

ap = argparse.ArgumentParser()
ap.add_argument("--stats", required=True)
ap.add_argument("--dados", required=True)
ap.add_argument("--urdf", required=True)
a = ap.parse_args()

alvo = json.load(open(a.stats))["real_g1_relative_eef_relative_joints"]["state"]
info = json.load(open(f"{a.dados}/meta/info.json"))
nomes = info["features"]["observation.state"]["names"]
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{a.dados}/data/chunk-000/*.parquet"))])
Q = np.stack(df["observation.state"].values)[::10]  # 1 a cada 10 quadros basta

model = pin.buildModelFromUrdf(a.urdf)
data = model.createData()
idx = [(model.joints[model.getJointId(n)].idx_q, j) for j, n in enumerate(nomes) if model.existJointName(n)]


def rot6d(R, conv):
    return np.concatenate([R[:, 0], R[:, 1]]) if conv == "colunas" else np.concatenate([R[0, :], R[1, :]])


resultados = []
for base in ["pelvis", "waist_yaw_link", "torso_link"]:
    for ponta in ["wrist_yaw_link", "hand_palm_link"]:
        for conv in ["colunas", "linhas"]:
            nota, det = 0.0, {}
            for lado in ["left", "right"]:
                fb, ft = model.getFrameId(base), model.getFrameId(f"{lado}_{ponta}")
                V = []
                for q in Q:
                    qq = pin.neutral(model)
                    for iq, j in idx:
                        qq[iq] = q[j]
                    pin.framesForwardKinematics(model, data, qq)
                    T = data.oMf[fb].actInv(data.oMf[ft])
                    V.append(np.concatenate([T.translation, rot6d(T.rotation, conv)]))
                V = np.array(V)
                s = alvo[f"{lado}_wrist_eef_9d"]
                mu, sd = np.array(s["mean"]), np.array(s["std"])
                z = np.abs(V.mean(0) - mu) / np.maximum(sd, 1e-3)
                nota += z.mean()
                det[lado] = (V.mean(0)[:3].round(3), mu[:3].round(3), z[:3].round(2), z[3:].mean().round(2))
            resultados.append((nota, base, ponta, conv, det))

for nota, base, ponta, conv, det in sorted(resultados, key=lambda r: r[0])[:6]:
    print(f"nota {nota:6.2f} | base={base:15s} ponta={ponta:15s} rot6d={conv}")
    for lado, (m, mu, zp, zr) in det.items():
        print(f"     {lado:5s} FK média xyz {m}  alvo {mu}  |z| xyz {zp}  |z| rot6d médio {zr}")
