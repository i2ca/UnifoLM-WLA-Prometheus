#!/usr/bin/env python
"""
Mede o FOV e a pose da câmera da cabeça — a do dataset do WLA e a nossa (ZED) — sem alvo de calibração.

Pontos 3D conhecidos: a posição de cada mão/garra no referencial da pelvis (no dataset vem gravada,
observation.state.*_ee_pose_gripper_base; nas nossas rodadas, a FK em ee_medido_*). Pontos 2D: o
UnifoLM-ER-1 aponta a mesma mão na imagem da cabeça. Com N pares 3D<->2D, para cada focal candidata
roda-se um PnP (principal no centro, sem distorção) e fica a focal de menor erro de reprojeção.
Sai: focal (px), FOV horizontal/vertical, posição da câmera na pelvis e inclinação para baixo.

    cd ~/DEV/unifolm-wla && ~/miniforge3/envs/wla/bin/python \\
        ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/calibra_camera_er1.py --dataset --rodada ~/wla_real_runs/20260929_184346
"""
import argparse
import glob
import json
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ER1 = "playground/Pretrained_models/UnifoLM-ER-1"
DS = Path.home() / "wla_dados/UnifoLM_G1_Dex1_Dataset/G1_Dex1_Put_Fruit_On_Plate"
PERGUNTA = {"dex1": "the center of the {lado} robot gripper, between its two fingers",
            "dex3": "the center of the palm of the {lado} robot hand"}


class ER1Ponto:
    def __init__(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(ER1)
        self.m = AutoModelForImageTextToText.from_pretrained(ER1, dtype=torch.bfloat16, device_map="cuda").eval()

    def aponta(self, rgb, oque):
        txt = (f"Point to {oque}. The robot's left hand is on the LEFT side of the image. "
               'Answer only JSON: [{"point_2d": [x, y], "label": "..."}] with coordinates in 0-1000.')
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(rgb)},
                                             {"type": "text", "text": txt}]}]
        e = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                          return_tensors="pt").to(self.m.device)
        with torch.inference_mode():
            s = self.m.generate(**e, max_new_tokens=60, do_sample=False)
        t = self.proc.decode(s[0, e["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", t)
        if not m:
            return None
        h, w = rgb.shape[:2]
        return np.array([float(m.group(1)) / 1000 * w, float(m.group(2)) / 1000 * h])


def ajusta(P3, P2, w, h):
    """Focal que minimiza a reprojeção (PnP para cada candidata)."""
    melhor = None
    for f in np.arange(150, 1200, 5.0):
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        ok, rv, tv = cv2.solvePnP(P3, P2, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(P3, rv, tv, K, None)
        err = np.linalg.norm(proj[:, 0] - P2, axis=1)
        if melhor is None or np.median(err) < melhor[0]:
            melhor = (float(np.median(err)), f, rv, tv, err)
    e, f, rv, tv, err = melhor
    R, _ = cv2.Rodrigues(rv)
    cam = (-R.T @ tv).ravel()                     # posição da câmera na pelvis
    eixo = R.T @ np.array([0, 0, 1.0])            # para onde a câmera olha (pelvis)
    inclina = np.degrees(np.arcsin(-eixo[2]))     # graus para baixo
    return {"focal_px": f, "fov_h_graus": round(float(np.degrees(2 * np.arctan(w / 2 / f))), 1),
            "fov_v_graus": round(float(np.degrees(2 * np.arctan(h / 2 / f))), 1),
            "camera_na_pelvis_cm": np.round(cam * 100, 1).tolist(), "inclinacao_para_baixo_graus": round(inclina, 1),
            "erro_reprojecao_mediana_px": round(e, 1), "pares": int(len(P2))}


def pares_dataset(er1, n_quadros):
    import pandas as pd
    d = pd.read_parquet(sorted(glob.glob(str(DS / "data/chunk-000/*.parquet")))[0])
    d = d[d.episode_index.isin(sorted(d.episode_index.unique())[:3])]
    video = sorted(glob.glob(str(DS / "videos/observation.images.head_stereo_left/chunk-000/*.mp4")))[0]
    eps = pd.read_parquet(sorted(glob.glob(str(DS / "meta/episodes/chunk-000/*.parquet")))[0])
    col = [c for c in eps.columns if "head_stereo_left" in c and "from_timestamp" in c][0]
    P3, P2 = [], []
    linhas = d.iloc[np.linspace(0, len(d) - 1, n_quadros).astype(int)]
    with tempfile.TemporaryDirectory() as tmp:
        for _, r in linhas.iterrows():
            t0 = float(eps.loc[eps.episode_index == r.episode_index, col].iloc[0])
            ts = t0 + float(r.timestamp)
            arq = f"{tmp}/q.jpg"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{ts:.3f}", "-i", video, "-frames:v", "1", arq],
                           check=True)
            rgb = cv2.imread(arq)[:, :, ::-1].copy()
            for lado, col3 in (("left", "observation.state.left_ee_pose_gripper_base"),
                               ("right", "observation.state.right_ee_pose_gripper_base")):
                p = er1.aponta(rgb, PERGUNTA["dex1"].format(lado=lado))
                if p is not None:
                    P3.append(np.asarray(r[col3])[:3])
                    P2.append(p)
    return np.array(P3, np.float64), np.array(P2, np.float64), rgb.shape[1], rgb.shape[0]


def pares_rodada(er1, pasta, n_quadros):
    arqs = sorted(Path(pasta).glob("*_cabeca.jpg"))
    arqs = [arqs[i] for i in np.linspace(0, len(arqs) - 1, min(n_quadros, len(arqs))).astype(int)]
    P3, P2 = [], []
    for a in arqs:
        rgb = cv2.imread(str(a))[:, :, ::-1].copy()
        z = np.load(str(a).replace("_cabeca.jpg", ".npz"))
        for lado in ("left", "right"):
            p = er1.aponta(rgb, PERGUNTA["dex3"].format(lado=lado))
            if p is not None:
                P3.append(z[f"ee_medido_{lado}"][:3])
                P2.append(p)
    return np.array(P3, np.float64), np.array(P2, np.float64), rgb.shape[1], rgb.shape[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="store_true")
    ap.add_argument("--rodada", default=None)
    ap.add_argument("--quadros", type=int, default=24)
    a = ap.parse_args()
    er1 = ER1Ponto()
    res = {}
    if a.dataset:
        res["dataset_unitree"] = ajusta(*pares_dataset(er1, a.quadros))
        print("DATASET (câmera deles):", json.dumps(res["dataset_unitree"], ensure_ascii=False), flush=True)
    if a.rodada:
        res["nossa_zed"] = ajusta(*pares_rodada(er1, a.rodada, a.quadros))
        print("NOSSA ZED:", json.dumps(res["nossa_zed"], ensure_ascii=False), flush=True)
    Path(Path.home() / "calibra_camera_er1.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
