#!/usr/bin/env python
"""Profundidade pelo estéreo da ZED da cabeça (27/10), com a CALIBRAÇÃO DE FÁBRICA (SN19436.conf, calib.stereolabs.com).

As imagens chegam do cameras_wla_server já transformadas, e a calibração tem que seguir o mesmo caminho:
  1. modo VGA: cada olho 672x376 (intrínsecos [*_CAM_VGA]);
  2. --zed-invertida: o quadro lado a lado inteiro gira 180°  -> cada olho gira 180° (cx' = W-1-cx, cy' = H-1-cy)
     e os olhos TROCAM: o head_stereo_left que chega é a lente física DIREITA (e vice-versa);
  3. enquadramento "largo": cada olho reduzido a 640 de largura (x 640/672) e centralizado em 640x480 com faixas pretas.

    from estereo_zed import EstereoZED
    z = EstereoZED("SN19436.conf")
    Zm = z.profundidade(esq_bgr, dir_bgr)      # metros, NaN onde não deu (640x480, mesma grade do head_stereo_left)
"""
import configparser
from pathlib import Path

import cv2
import numpy as np

W0, H0 = 672, 376          # um olho em VGA


class EstereoZED:
    def __init__(self, conf=Path(__file__).with_name("SN19436.conf"), invertida=True, largura=640, altura=480,
                 trocar=None):
        c = configparser.ConfigParser()
        c.read(conf)
        K, dist = {}, {}
        for lado in ("LEFT", "RIGHT"):
            s = c[f"{lado}_CAM_VGA"]
            fx, fy, cx, cy = (float(s[k]) for k in ("fx", "fy", "cx", "cy"))
            if invertida:   # olho girado 180°
                cx, cy = W0 - 1 - cx, H0 - 1 - cy
            K[lado] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            dist[lado] = np.array([float(s[k]) for k in ("k1", "k2", "p1", "p2", "k3")])
        st = c["STEREO"]
        B = float(st["Baseline"]) / 1000
        rv = np.array([float(st["RX_VGA"]), float(st["CV_VGA"]), float(st["RZ_VGA"])])
        R, _ = cv2.Rodrigues(rv)                  # direita <- esquerda (lentes físicas)
        T = np.array([-B, float(st.get("TY", 0)) / 1000, float(st.get("TZ", 0)) / 1000])
        trocar = invertida if trocar is None else trocar
        if invertida:
            Rz = np.diag([-1.0, -1.0, 1.0])
            R, T = Rz @ R @ Rz.T, Rz @ T
        if trocar:   # o "esquerdo" que chega é a lente física direita
            K1, d1, K2, d2 = K["RIGHT"], dist["RIGHT"], K["LEFT"], dist["LEFT"]
            R, T = R.T, -R.T @ T
            T = -T if T[0] > 0 else T             # cam2 à direita da cam1 (T[0] < 0 no padrão do OpenCV)
        else:
            K1, d1, K2, d2 = K["LEFT"], dist["LEFT"], K["RIGHT"], dist["RIGHT"]
        # enquadramento "largo": escala e faixa preta
        s = largura / W0
        h = int(round(H0 * s))
        self.y0 = (altura - h) // 2
        self.h, self.w, self.H = h, largura, altura
        S = np.diag([s, s, 1.0])
        K1, K2 = S @ K1, S @ K2
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, d1, K2, d2, (largura, h), R, T, alpha=0)
        self.m1 = cv2.initUndistortRectifyMap(K1, d1, R1, P1, (largura, h), cv2.CV_32FC1)
        self.m2 = cv2.initUndistortRectifyMap(K2, d2, R2, P2, (largura, h), cv2.CV_32FC1)
        self.f, self.B = float(P1[0, 0]), abs(float(P2[0, 3] / P2[0, 0]))
        # de volta para a imagem ORIGINAL do olho esquerdo (para devolver a profundidade na grade que o modelo vê)
        self.K1, self.d1, self.R1, self.P1 = K1, d1, R1, P1
        self.sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=96, blockSize=5, P1=8 * 3 * 25,
                                          P2=32 * 3 * 25, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                                          disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def retifica(self, esq, dir_):
        a = esq[self.y0:self.y0 + self.h]
        b = dir_[self.y0:self.y0 + self.h]
        return (cv2.remap(a, *self.m1, cv2.INTER_LINEAR), cv2.remap(b, *self.m2, cv2.INTER_LINEAR))

    def profundidade(self, esq, dir_, retificada=False):
        """Metros, na grade RETIFICADA do olho esquerdo (retificada=True) ou na original 640x480."""
        a, b = self.retifica(esq, dir_)
        cl = cv2.createCLAHE(3.0, (8, 8))
        ga, gb = (cl.apply(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)) for x in (a, b))
        disp = self.sgbm.compute(ga, gb).astype(np.float32) / 16
        Z = np.where(disp > 1, self.f * self.B / np.maximum(disp, 1e-3), np.nan).astype(np.float32)
        if retificada:
            return Z
        # retificada -> original: para cada pixel original, onde ele cai na retificada
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        pts = cv2.undistortPoints(np.stack([xx.ravel(), yy.ravel()], 1)[:, None], self.K1, self.d1, R=self.R1, P=self.P1)
        mx, my = pts[:, 0, 0].reshape(self.h, self.w), pts[:, 0, 1].reshape(self.h, self.w)
        Zo = cv2.remap(Z, mx, my, cv2.INTER_NEAREST, borderValue=np.nan)
        out = np.full((self.H, self.w), np.nan, np.float32)
        out[self.y0:self.y0 + self.h] = Zo
        return out
