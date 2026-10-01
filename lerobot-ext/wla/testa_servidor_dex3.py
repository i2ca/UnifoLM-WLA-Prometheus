#!/usr/bin/env python
"""
Teste de fumaça do servidor WLA Dex3 (servidor_wla_dex3.py) SEM robô: manda 3 imagens do nosso dataset da
maçã e o estado da pose de partida (--pose maca), e mostra o que o modelo responde.

    cd ~/DEV/unifolm-wla && ~/miniforge3/envs/wla/bin/python \\
        ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/testa_servidor_dex3.py --imgs ~/wla_testes/imgs_maca --n 5
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, os.path.join(os.getcwd(), "model_server"))
from fk_g1 import FK  # noqa: E402
from tools import msgpack_numpy  # noqa: E402
from websockets.sync.client import connect  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8601")
    ap.add_argument("--imgs", required=True, help="pasta com cabeca.jpg, punho_esq.jpg, punho_dir.jpg (640x480)")
    ap.add_argument("--tarefa", default="Pick up the apple and place it on the black X.")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--pose", default="maca", help="pose de partida do estado (maca, gravacao, inicial...)")
    a = ap.parse_args()

    pose = json.load(open(AQUI / "pose_partida_dex1.json"))
    s = "_" + a.pose
    q = np.array(pose["pernas_maca"] + pose.get("cintura" + s, pose["cintura"]) + pose["left" + s] + pose["right" + s])
    fk = FK()
    ee = {l: fk.ee9(q, l) for l in ("left", "right")}
    im = {n: cv2.imread(str(Path(a.imgs) / f"{n}.jpg")) for n in ("cabeca", "punho_esq", "punho_dir")}
    for n, v in im.items():
        if v is None:
            sys.exit(f"faltou {n}.jpg em {a.imgs}")
    obs = {"observation.images.cam_left_high": im["cabeca"],
           "observation.images.cam_left_wrist": im["punho_esq"],
           "observation.images.cam_right_wrist": im["punho_dir"],
           "observation.state.left_ee_6d": ee["left"], "observation.state.right_ee_6d": ee["right"],
           "observation.state.left_fig6d": np.zeros(6, np.float32),
           "observation.state.right_fig6d": np.zeros(6, np.float32),
           "observation.state.lower_body": q[:15].astype(np.float32), "instruction": a.tarefa}

    ws = connect(a.url, max_size=None, compression=None, open_timeout=10)
    meta = msgpack_numpy.unpackb(ws.recv())
    print(f"metadata: env={meta.get('env')} maos={meta.get('maos')} unnorm={meta.get('default_unnorm_key')} "
          f"chunk={meta.get('action_chunk_size')}")
    packer = msgpack_numpy.Packer()
    for i in range(a.n):
        t0 = time.perf_counter()
        ws.send(packer.pack({"type": "get_action", "obs": obs}))
        r = ws.recv()
        ms = (time.perf_counter() - t0) * 1e3
        if isinstance(r, str):
            sys.exit(f"erro do servidor: {r[:800]}")
        acao = {k: np.asarray(v) for k, v in msgpack_numpy.unpackb(r).items()}
        print(f"\n[{i}] {ms:.0f} ms | chaves: {sorted(acao)}")
        for l in ("left", "right"):
            traj = acao[f"action.{l}_ee_rpy"][0]
            d = (traj[-1, :3] - ee[l][:3]) * 100
            print(f"  mão {l:5s}: agora {np.round(ee[l][:3] * 100, 1)} cm -> fim do trecho {np.round(traj[-1, :3] * 100, 1)} cm"
                  f"  (Δ {np.round(d, 1)} cm)")
            f6 = acao[f"action.{l}_fig6d"][0]
            print(f"             dedos fig6d início {np.round(f6[0], 2)} fim {np.round(f6[-1], 2)}")
        if "action.base_command" in acao:
            print(f"  base (vx, vy, vyaw, altura) fim: {np.round(acao['action.base_command'][0, -1], 3)}")


if __name__ == "__main__":
    main()
