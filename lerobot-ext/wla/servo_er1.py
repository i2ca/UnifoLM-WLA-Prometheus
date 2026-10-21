#!/usr/bin/env python
"""Servo visual com o ER-1 (21/10): onde está o objeto na mesa, em centímetros, no referencial da pelvis.

O WLA chega a poucos centímetros da caneca/maçã e hesita (fecha ao lado, abre, tenta de novo). Aqui:
  1. o ER-1 APONTA o objeto na imagem da câmera da cabeça (pixel);
  2. a câmera está CALIBRADA (focal + pose no referencial do TRONCO — ela gira com a cintura): o pixel vira um
     raio, e o raio cruza o plano da pegada (z da mão ao pegar, ~5 cm na pelvis) -> posição 3D do objeto;
  3. o executor compara com a mão (FK) e corrige os últimos centímetros (roda_wla_real.py --servo).

Calibração (uma vez, ou quando mexer na câmera): o ER-1 aponta o centro da palma da mão direita em quadros
gravados pelo executor; a FK dá a mesma mão em 3D; PnP com focal varrida. Sai ~/servo_camera.json.

    cd ~/DEV/unifolm-wla && ~/miniforge3/envs/wla/bin/python ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/servo_er1.py \\
        calibra --rodadas ~/wla_real_runs/20261021_133430 ~/wla_real_runs/20261008_133441
    ... servo_er1.py testa --rodada ~/wla_real_runs/20261021_133430 --objeto "white mug" --quadros 59,64
"""
import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fk_g1 import FK, _eixo  # noqa: E402

ARQ_CAMERA = Path.home() / "servo_camera.json"
PALMA = "the center of the palm of the robot's right hand (the hand on the RIGHT side of the image)"


# ── geometria ──────────────────────────────────────────────────────────────────────────────────────────
def torso_na_pelvis(fk, q29):
    """Pose do torso (filho do waist_pitch_joint) na pelvis, com a cintura medida."""
    R, p = np.eye(3), np.zeros(3)
    for j in fk.cadeia["right"]:
        p = p + R @ j["p"]
        R = R @ j["R"]
        if j["tipo"] in ("revolute", "continuous") and j["nome"] in fk.idx:
            R = R @ _eixo(j["eixo"], q29[fk.idx[j["nome"]]])
        if j["nome"] == "waist_pitch_joint":
            return R, p
    raise RuntimeError("waist_pitch_joint não está na cadeia")


class Camera:
    """Câmera pinhole (centro da imagem, sem distorção) presa ao tronco: K, R_ct, t_ct (câmera <- tronco)."""

    def __init__(self, cfg):
        self.f, self.w, self.h = cfg["focal_px"], cfg["largura"], cfg["altura"]
        self.R_ct = np.array(cfg["R_camera_tronco"])
        self.t_ct = np.array(cfg["t_camera_tronco"])

    @classmethod
    def carrega(cls, arq=ARQ_CAMERA):
        return cls(json.loads(Path(arq).read_text()))

    def raio_na_pelvis(self, px, R_tp, p_tp):
        """Pixel -> (origem, direção) do raio no referencial da pelvis."""
        d_cam = np.array([(px[0] - self.w / 2) / self.f, (px[1] - self.h / 2) / self.f, 1.0])
        R_tc, o_tronco = self.R_ct.T, -self.R_ct.T @ self.t_ct          # câmera -> tronco
        o = p_tp + R_tp @ o_tronco
        d = R_tp @ (R_tc @ d_cam)
        return o, d / np.linalg.norm(d)

    def ponto_no_plano(self, px, R_tp, p_tp, z):
        """Onde o raio do pixel cruza o plano horizontal z (pelvis). None se não cruza à frente."""
        o, d = self.raio_na_pelvis(px, R_tp, p_tp)
        if abs(d[2]) < 1e-6:
            return None
        s = (z - o[2]) / d[2]
        return o + s * d if s > 0 else None

    def projeta(self, P_pelvis, R_tp, p_tp):
        P_t = R_tp.T @ (np.asarray(P_pelvis) - p_tp)
        P_c = self.R_ct @ P_t + self.t_ct
        return np.array([self.f * P_c[0] / P_c[2] + self.w / 2, self.f * P_c[1] / P_c[2] + self.h / 2])


# ── ER-1 apontando ─────────────────────────────────────────────────────────────────────────────────────
def le_ponto(texto, w, h):
    m = re.search(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", texto or "")
    return None if not m else np.array([float(m.group(1)) / 1000 * w, float(m.group(2)) / 1000 * h])


def pergunta_aponta(oque):
    return (f"Point to {oque}. Answer only JSON: [{{\"point_2d\": [x, y], \"label\": \"...\"}}] "
            "with coordinates in 0-1000.")


class ER1Local:
    """ER-1 carregado aqui (para imagens gravadas). Ao vivo, o executor usa o servidor :8098 (aponta_ao_vivo)."""

    def __init__(self, caminho="playground/Pretrained_models/UnifoLM-ER-1"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(caminho)
        self.m = AutoModelForImageTextToText.from_pretrained(caminho, dtype=torch.bfloat16, device_map="cuda").eval()

    def aponta(self, rgb, oque):
        from PIL import Image
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(rgb)},
                                             {"type": "text", "text": pergunta_aponta(oque)}]}]
        e = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                          return_tensors="pt").to(self.m.device)
        with self.torch.inference_mode():
            s = self.m.generate(**e, max_new_tokens=60, do_sample=False)
        return le_ponto(self.proc.decode(s[0, e["input_ids"].shape[1]:], skip_special_tokens=True),
                        rgb.shape[1], rgb.shape[0])


def aponta_ao_vivo(url, oque, w=640, h=480, timeout=10):
    """Pelo servidor do ER-1 (o quadro mais novo da cabeça). Devolve (pixel ou None, ms)."""
    u = f"{url}/pergunta?" + urllib.parse.urlencode({"q": pergunta_aponta(oque), "livre": 1, "max": 60, "lora": 0,
                                                     "cam": "head_stereo_left"})
    r = json.loads(urllib.request.urlopen(u, timeout=timeout).read())
    if r.get("erro"):
        raise RuntimeError(r["erro"])
    return le_ponto(r.get("cru"), w, h), r.get("ms")


# ── calibração ─────────────────────────────────────────────────────────────────────────────────────────
def calibra(rodadas, n_por_rodada, er1, fk):
    P3, P2, w, h = [], [], 640, 480
    for rod in rodadas:
        arqs = sorted(Path(rod).glob("*_cabeca.jpg"))
        for a in [arqs[i] for i in np.linspace(0, len(arqs) - 1, min(n_por_rodada, len(arqs))).astype(int)]:
            rgb = cv2.imread(str(a))[:, :, ::-1].copy()
            h, w = rgb.shape[:2]
            z = np.load(str(a).replace("_cabeca.jpg", ".npz"))
            q = z["q_medido"]
            R_tp, p_tp = torso_na_pelvis(fk, q)
            mao = fk.pose(q, "right")[1]
            px = er1.aponta(rgb, PALMA)
            if px is None:
                continue
            P3.append(R_tp.T @ (mao - p_tp))       # a mão no referencial do TRONCO
            P2.append(px)
            print(f"  {a.parent.name}/{a.name}: mão {np.round(mao * 100, 1)} cm -> pixel {np.round(px)}", flush=True)
    P3, P2 = np.array(P3, np.float64), np.array(P2, np.float64)
    melhor = None
    for f in np.arange(200, 900, 2.0):
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
    return {"focal_px": float(f), "largura": int(w), "altura": int(h),
            "R_camera_tronco": R.tolist(), "t_camera_tronco": tv.ravel().tolist(),
            "camera_no_tronco_cm": np.round((-R.T @ tv).ravel() * 100, 1).tolist(),
            "fov_h_graus": round(float(np.degrees(2 * np.arctan(w / 2 / f))), 1),
            "erro_reprojecao_mediana_px": round(e, 1), "erro_p90_px": round(float(np.percentile(err, 90)), 1),
            "pares": int(len(P2))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibra")
    c.add_argument("--rodadas", nargs="+", required=True)
    c.add_argument("--quadros", type=int, default=30, help="quadros por rodada")
    t = sub.add_parser("testa", help="aponta o objeto em quadros gravados e compara com a mão (FK)")
    t.add_argument("--rodada", required=True)
    t.add_argument("--objeto", default="white mug")
    t.add_argument("--quadros", required=True, help="números dos quadros, ex.: 59,64")
    t.add_argument("--z", type=float, default=0.05, help="m: plano da pegada na pelvis")
    a = ap.parse_args()
    fk = FK()
    er1 = ER1Local()
    if a.cmd == "calibra":
        cfg = calibra(a.rodadas, a.quadros, er1, fk)
        ARQ_CAMERA.write_text(json.dumps(cfg, indent=1))
        print(json.dumps({k: v for k, v in cfg.items() if not k.startswith(("R_", "t_"))}, ensure_ascii=False))
        print(f"✅ {ARQ_CAMERA}")
    else:
        cam = Camera.carrega()
        for k in [int(x) for x in a.quadros.split(",")]:
            base = Path(a.rodada) / f"{k:04d}"
            rgb = cv2.imread(f"{base}_cabeca.jpg")[:, :, ::-1].copy()
            z = np.load(f"{base}.npz")
            q = z["q_medido"]
            R_tp, p_tp = torso_na_pelvis(fk, q)
            mao = fk.pose(q, "right")[1]
            px_o = er1.aponta(rgb, f"the {a.objeto}")
            px_m = er1.aponta(rgb, PALMA)
            obj = cam.ponto_no_plano(px_o, R_tp, p_tp, a.z) if px_o is not None else None
            print(f"#{k}: objeto px {px_o} -> {None if obj is None else np.round(obj * 100, 1)} cm | "
                  f"mão FK {np.round(mao * 100, 1)} cm (projeção {np.round(cam.projeta(mao, R_tp, p_tp))}, "
                  f"ER-1 aponta {px_m}) | objeto - mão = "
                  f"{None if obj is None else np.round((obj - mao)[:2] * 100, 1)} cm", flush=True)


if __name__ == "__main__":
    main()
