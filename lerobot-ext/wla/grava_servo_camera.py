#!/usr/bin/env python
"""~/servo_camera.json A PARTIR DA CALIBRAÇÃO LIDAR + ZED (03/11).

calibra_lidar_camera.py acha a pose da câmera RETIFICADA do olho esquerdo no torso; o servo_camera.json (cabine do
WLA, ER-1 apontando, vídeos) usa a imagem ORIGINAL do head_stereo_left (640x480, com as faixas pretas) num pinhole
sem distorção. Aqui: R_orig = R_ret @ R1 (a retificação gira a câmera), f e (cx, cy) do olho esquerdo do SN19436
na grade 640x480. Mede quanto o pinhole sem distorção erra contra o modelo completo (com distorção) e faz backup.

    ~/miniforge3/envs/wla/bin/python lerobot-ext/wla/grava_servo_camera.py
"""
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI / "zed"))
from estereo_zed import EstereoZED  # noqa: E402


def main():
    cal = json.loads((Path.home() / "mapas/calib_lidar_camera.json").read_text())
    c = cal["camera_retificada"]
    z = EstereoZED()
    R_tr, o = np.array(c["R_torso_camera"]), np.array(c["o_no_torso"])   # retificada -> torso
    R_to = R_tr @ z.R1                                                 # original -> torso
    R_ct = R_to.T
    K = z.K1
    f, cx, cy = float((K[0, 0] + K[1, 1]) / 2), float(K[0, 2]), float(K[1, 2] + z.y0)
    # erro do pinhole sem distorção (o que o servo usa) contra o modelo completo, numa grade de pontos à frente
    uu, vv = np.meshgrid(np.linspace(40, 600, 15), np.linspace(z.y0 + 20, z.y0 + z.h - 20, 9))
    px = np.stack([uu.ravel(), vv.ravel() - z.y0], 1).astype(np.float32)
    raio = cv2.undistortPoints(px[:, None], K, z.d1)[:, 0]            # normalizado, câmera original
    P = np.c_[raio, np.ones(len(raio))] * 0.7
    u_pin = np.c_[f * P[:, 0] / P[:, 2] + cx, f * P[:, 1] / P[:, 2] + cy]
    erro = np.linalg.norm(u_pin - np.c_[uu.ravel(), vv.ravel()], axis=1)
    eixo = R_to @ np.array([0, 0, 1.0])
    # o mesmo modelo, agora COM a distorção: confere que fecha (deve dar ~0 px)
    k1, k2, p1, p2, k3 = z.d1
    x, y = P[:, 0] / P[:, 2], P[:, 1] / P[:, 2]
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd, yd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    erro_d = np.linalg.norm(np.c_[K[0, 0] * xd + cx, K[1, 1] * yd + cy] - np.c_[uu.ravel(), vv.ravel()], axis=1)
    novo = {"focal_px": f, "largura": 640, "altura": 480, "cx": cx, "cy": cy, "distorcao": [float(v) for v in z.d1],
            "R_camera_tronco": R_ct.tolist(), "t_camera_tronco": (-R_ct @ o).tolist(),
            "camera_no_tronco_cm": np.round(o * 100, 1).tolist(),
            "inclinacao_para_baixo_graus": round(float(np.degrees(np.arcsin(-eixo[2]))), 1),
            "fov_h_graus": round(float(np.degrees(2 * np.arctan(320 / f))), 1),
            "erro_com_distorcao_px": {"mediana": round(float(np.median(erro_d)), 2), "max": round(float(erro_d.max()), 2)},
            "erro_pinhole_sem_distorcao_px": {"mediana": round(float(np.median(erro)), 2), "p90": round(float(np.percentile(erro, 90)), 2),
                                             "max": round(float(erro.max()), 2)},
            "origem": f"calibra_lidar_camera.py {cal['quando']} (lidar nivelado pelo IMU; ICP ZED->lidar; "
                      f"disparidade {c.get('delta_disparidade_px', 0):+.2f} px)",
            "gravado": time.strftime("%Y-%m-%d %H:%M")}
    arq = Path.home() / "servo_camera.json"
    if arq.exists():
        bak = arq.with_name(f"servo_camera.json.bak-{time.strftime('%Y%m%d-%H%M')}")
        shutil.copy2(arq, bak)
        velho = json.loads(arq.read_text())
        print(f"backup: {bak}  (antes: {velho.get('camera_no_tronco_cm')} cm, {velho.get('inclinacao_para_baixo_graus')}°, "
              f"f {velho.get('focal_px'):.1f})")
    arq.write_text(json.dumps(novo, indent=1))
    print(f"✅ {arq}: câmera {novo['camera_no_tronco_cm']} cm no torso, {novo['inclinacao_para_baixo_graus']}° p/ baixo, "
          f"f {f:.1f} px, centro ({cx:.1f}, {cy:.1f}); com a distorção o modelo erra {novo['erro_com_distorcao_px']} px "
          f"(sem ela: {novo['erro_pinhole_sem_distorcao_px']} px)")


if __name__ == "__main__":
    main()
