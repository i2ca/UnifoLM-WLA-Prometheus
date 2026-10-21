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
PALMAS = {"right": PALMA,
          "left": "the center of the palm of the robot's left hand (the hand on the LEFT side of the image)"}


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
            for lado in ("right", "left"):        # as DUAS mãos: pontos espalhados = PnP bem condicionado
                mao = fk.pose(q, lado)[1]
                px = er1.aponta(rgb, PALMAS[lado])
                if px is None:
                    continue
                P3.append(R_tp.T @ (mao - p_tp))       # a mão no referencial do TRONCO
                P2.append(px)
                print(f"  {a.parent.name}/{a.name} {lado}: mão {np.round(mao * 100, 1)} cm -> pixel {np.round(px)}",
                      flush=True)
    P3, P2 = np.array(P3, np.float64), np.array(P2, np.float64)
    np.savez(ARQ_CAMERA.with_suffix(".pares.npz"), P3=P3, P2=P2, w=w, h=h)   # refaz o ajuste sem o ER-1
    return ajusta(P3, P2, w, h)


# Câmera da cabeça do G1 (d435_joint no URDF): no tronco, 5,8 cm à frente, 1,8 cm à esquerda, 43 cm acima,
# inclinada 0,83 rad (~48°) para baixo. A ZED fica no mesmo lugar: é o ponto de partida e a âncora do ajuste.
CAM_URDF_POS = np.array([0.0576, 0.0175, 0.4299])
CAM_URDF_PITCH = 0.8308


def ajusta(P3, P2, w, h, ancora_cm=3.0):
    """Ajuste robusto (soft-L1) de rotação + posição + focal, com a posição ANCORADA na do URDF (desvio
    custa como ~ancora_cm de erro). Só o PnP livre escorregava a câmera para 48 cm à frente."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as Rot
    Ry = Rot.from_euler("y", CAM_URDF_PITCH).as_matrix()
    M = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)     # óptico (x dir, y baixo, z frente) no link
    R_tc0 = Ry @ M                                                # câmera -> tronco
    x0 = np.concatenate([Rot.from_matrix(R_tc0).as_rotvec(), CAM_URDF_POS, [370.0]])

    def residuos(x):
        R_tc = Rot.from_rotvec(x[:3]).as_matrix()
        pos, f = x[3:6], x[6]
        Pc = (P3 - pos) @ R_tc                                    # tronco -> câmera
        uv = np.stack([f * Pc[:, 0] / Pc[:, 2] + w / 2, f * Pc[:, 1] / Pc[:, 2] + h / 2], 1)
        r = (uv - P2).ravel()
        ancora = (pos - CAM_URDF_POS) / (ancora_cm / 100) * 10.0   # 1 ancora_cm ~ 10 px
        return np.concatenate([r, ancora])

    sol = least_squares(residuos, x0, loss="soft_l1", f_scale=15.0)
    R_tc = Rot.from_rotvec(sol.x[:3]).as_matrix()
    pos, f = sol.x[3:6], float(sol.x[6])
    Pc = (P3 - pos) @ R_tc
    err = np.linalg.norm(np.stack([f * Pc[:, 0] / Pc[:, 2] + w / 2, f * Pc[:, 1] / Pc[:, 2] + h / 2], 1) - P2, axis=1)
    bons = err < 25
    eixo = R_tc @ np.array([0, 0, 1.0])
    R_ct = R_tc.T
    return {"focal_px": f, "largura": int(w), "altura": int(h),
            "R_camera_tronco": R_ct.tolist(), "t_camera_tronco": (-R_ct @ pos).tolist(),
            "camera_no_tronco_cm": np.round(pos * 100, 1).tolist(),
            "inclinacao_para_baixo_graus": round(float(np.degrees(np.arcsin(-eixo[2]))), 1),
            "fov_h_graus": round(float(np.degrees(2 * np.arctan(w / 2 / f))), 1),
            "erro_reprojecao_mediana_px": round(float(np.median(err[bons])), 1),
            "erro_p90_px": round(float(np.percentile(err[bons], 90)), 1),
            "pares": int(len(P2)), "pares_bons": int(bons.sum())}


def _pnp_livre(P3, P2, w, h):
    # RANSAC: o ER-1 às vezes aponta a coisa errada (outra mão, o pulso). Focal de 200 a 800 px (FOV 44-116°).
    melhor = None
    for f in np.arange(200, 800, 2.0):
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        ok, rv, tv, inl = cv2.solvePnPRansac(P3, P2, K, None, reprojectionError=20.0, iterationsCount=300,
                                             flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < 8:
            continue
        inl = inl.ravel()
        ok, rv, tv = cv2.solvePnP(P3[inl], P2[inl], K, None, rv, tv, True, cv2.SOLVEPNP_ITERATIVE)
        proj, _ = cv2.projectPoints(P3, rv, tv, K, None)
        err = np.linalg.norm(proj[:, 0] - P2, axis=1)
        nota = (-len(inl), float(np.median(err[inl])))           # mais pontos bons, depois menor erro
        if melhor is None or nota < melhor[0]:
            melhor = (nota, f, rv, tv, err, inl)
    (_, e), f, rv, tv, err, inl = melhor[0], melhor[1], melhor[2], melhor[3], melhor[4], melhor[5]
    e = float(np.median(err[inl]))
    R, _ = cv2.Rodrigues(rv)
    return {"focal_px": float(f), "largura": int(w), "altura": int(h),
            "R_camera_tronco": R.tolist(), "t_camera_tronco": tv.ravel().tolist(),
            "camera_no_tronco_cm": np.round((-R.T @ tv).ravel() * 100, 1).tolist(),
            "fov_h_graus": round(float(np.degrees(2 * np.arctan(w / 2 / f))), 1),
            "erro_reprojecao_mediana_px": round(e, 1), "erro_p90_px": round(float(np.percentile(err[inl], 90)), 1),
            "pares": int(len(P2)), "pares_bons": int(len(inl))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibra")
    c.add_argument("--rodadas", nargs="+", required=True)
    c.add_argument("--quadros", type=int, default=30, help="quadros por rodada")
    r = sub.add_parser("reajusta", help="refaz o ajuste com os pares salvos (sem o ER-1)")
    r.add_argument("--ancora-cm", type=float, default=3.0)
    t = sub.add_parser("testa", help="aponta o objeto em quadros gravados e compara com a mão (FK)")
    t.add_argument("--rodada", required=True)
    t.add_argument("--objeto", default="white mug")
    t.add_argument("--quadros", required=True, help="números dos quadros, ex.: 59,64")
    t.add_argument("--z", type=float, default=0.05, help="m: plano da pegada na pelvis")
    a = ap.parse_args()
    fk = FK()
    if a.cmd == "reajusta":
        d = np.load(ARQ_CAMERA.with_suffix(".pares.npz"))
        cfg = ajusta(d["P3"], d["P2"], int(d["w"]), int(d["h"]), a.ancora_cm)
        ARQ_CAMERA.write_text(json.dumps(cfg, indent=1))
        print(json.dumps({k: v for k, v in cfg.items() if not k.startswith(("R_", "t_"))}, ensure_ascii=False))
        return
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
