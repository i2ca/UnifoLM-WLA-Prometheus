#!/usr/bin/env python
"""VISUALIZAÇÃO da cabine do robô (03/11): o que o WLA PLANEJA e o que o robô FAZ, para a página :8090 desenhar
(câmera com o plano projetado de verdade, mundo 3D, cintura, dedos). O executor chama:

  viz.plano(trecho, q29)          a cada trecho novo: os 30 passos da mão (pelvis e pixel na câmera da cabeça),
                                  dedos e yaw da cintura planejados
  viz.vivo(tc, q29, ...)          a cada passo do laço (limitado a ~5 Hz aqui dentro): mãos medidas (pelvis e
                                  pixel), rastro, cintura medida x pedida, dedos, temperaturas, passo do trecho

A projeção usa a calibração da ZED (~/servo_camera.json, servo_er1.Camera) e a pose do tronco pelas juntas MEDIDAS
(a câmera é presa no tronco: o yaw da cintura gira a vista). Sem calibração: só o 3D.
"""
import time
from collections import deque
from pathlib import Path

import numpy as np


# Dex3 na ordem do SDK -> nomes do URDF (g1_body29_hand14)
MAO_URDF = {"left": ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"],
            "right": ["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]}


def _l(v, n=3):
    return [round(float(x), n) for x in np.asarray(v).ravel()]


class VizCabine:
    def __init__(self, cabine, fk, camera_json=str(Path.home() / "servo_camera.json"), periodo=0.2):
        self.cab, self.fk, self.periodo = cabine, fk, periodo
        self.cam, self.torso = None, None
        try:
            from servo_er1 import Camera, torso_na_pelvis
            self.cam, self.torso = Camera.carrega(camera_json), torso_na_pelvis
        except Exception as e:  # noqa: BLE001
            print(f"   (viz: sem calibração da câmera ({e}) — a cabine mostra só o 3D)", flush=True)
        self.rastro = {l: deque(maxlen=150) for l in ("left", "right")}
        self.t = 0.0

    def _px(self, P, q):
        if self.cam is None:
            return None
        R_tp, p_tp = self.torso(self.fk, q)
        out = []
        for p in np.atleast_2d(P):
            u = self.cam.projeta(p, R_tp, p_tp)
            out.append(_l(u, 1) if np.all(np.isfinite(u)) else None)
        return out

    def plano(self, tr, q):
        try:
            acao = tr["acao"]
            d = {"t": time.time(), "n": tr.get("n"), "ms": round(float(tr.get("ms", 0))), "frase": tr.get("frase", ""),
                 "img": {"w": getattr(self.cam, "w", 640), "h": getattr(self.cam, "h", 480)}, "maos": {}}
            for l in ("left", "right"):
                P = np.asarray(acao[f"action.{l}_ee_rpy"])[0][:, :3]
                f6 = np.asarray(acao.get(f"action.{l}_fig6d", np.zeros((1, len(P), 6))))[0].mean(-1)
                d["maos"][l] = {"xyz": [_l(p) for p in P], "px": self._px(P, q), "dedos": _l(f6, 2)}
            if "action.lower_body" in acao:
                d["yaw"] = _l(np.asarray(acao["action.lower_body"])[0][:, 12])
            self.cab.publica_extra("plano", d)
        except Exception as e:  # noqa: BLE001  (a visualização nunca derruba o laço)
            self.cab.publica_extra("plano", {"erro": repr(e)[:200]})

    def vivo(self, tc, q, yaw_alvo, fech_mao, passo=None, temps=None, dedos_med=None, maos_q=None):
        if tc - self.t < self.periodo:
            return
        self.t = tc
        try:
            d = {"t": time.time(), "passo": passo, "yaw_medido": round(float(q[12]), 4),
                 "yaw_alvo": round(float(yaw_alvo), 4), "maos": {}}
            for l in ("left", "right"):
                p = self.fk.pose(q, l)[1]
                self.rastro[l].append(_l(p))
                d["maos"][l] = {"xyz": _l(p), "px": (self._px(p, q) or [None])[0], "rastro": list(self.rastro[l]),
                                "fecha_pedido": None if fech_mao.get(l) is None else round(float(fech_mao[l]), 2),
                                "fecha_medido": None if not dedos_med or dedos_med.get(l) is None
                                else round(float(dedos_med[l]), 2)}
            # 03/11: TODAS as juntas pelo nome do URDF, para o modelo 3D do G1 na página
            from fk_g1 import NOMES
            j = {n: round(float(v), 4) for n, v in zip(NOMES, np.asarray(q)[:29])}
            for l, qs in (maos_q or {}).items():
                if qs is not None:
                    j.update({f"{l}_hand_{n}_joint": round(float(v), 4) for n, v in zip(MAO_URDF[l], qs)})
            d["juntas"] = j
            if self.cam is not None:   # 03/11: a pose da câmera na pelvis (a página reprojeta a profundidade em 3D)
                R_tp, p_tp = self.torso(self.fk, q)
                R_cp = R_tp @ self.cam.R_ct.T
                d["camera"] = {"R": [_l(r, 5) for r in R_cp], "t": _l(p_tp - R_cp @ self.cam.t_ct, 4),
                               "f": round(float(self.cam.f), 2), "w": self.cam.w, "h": self.cam.h,
                               "cx": round(float(self.cam.cx), 2), "cy": round(float(self.cam.cy), 2)}
            if temps is not None:
                d["temps"] = temps() if callable(temps) else temps
            self.cab.publica_extra("vivo", d)
        except Exception as e:  # noqa: BLE001
            self.cab.publica_extra("vivo", {"erro": repr(e)[:200]})
