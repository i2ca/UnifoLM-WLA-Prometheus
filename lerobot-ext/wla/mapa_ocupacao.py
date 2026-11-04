#!/usr/bin/env python
"""MAPA DE OCUPAÇÃO 3D pelo lidar da cabeça (Livox MID-360), 03/11. Roda na PGX; SÓ LÊ o robô.

  lidar   tcp://<robô>:5558 (robo_g1/lidar_servidor.py, no init)  -> pontos no referencial do lidar
  juntas  tcp://<robô>:6001 (lowstate da ponte)                     -> FK: lidar -> torso -> pelvis (a cintura gira)

Cada lote (~0,1 s) vira log-odds numa grade de cubos de 1 cm (03/11: era 3 cm) na área de trabalho em volta do robô, no referencial da PELVIS (o mesmo da
cabine e do WLA; o robô parado no lugar):
  o cubo onde o raio BATE ganha  l_occ (+0,85)
  os cubos que o raio ATRAVESSA  l_free (-0,4)     -> o que sai do lugar (uma pessoa, a caneca) some sozinho
  preso em [-2, +3,5]                              -> nunca fica "certo demais" para desaprender
Antes, os pontos do PRÓPRIO CORPO (braços pela FK, tronco/cabeça) saem, senão o braço vira obstáculo.

03/11 — com a CALIBRAÇÃO (calibra_lidar_camera.py -> ~/mapas/calib_lidar_camera.json):
  - o lidar é nivelado pela gravidade (IMU da pelve) e perde a FAIXA DE BAIXO do campo (< -43°), onde nascem pontos
    falsos ao longo dos raios (a "rampa" na frente do robô);
  - o ESTÉREO DA ZED entra no mesmo mapa (~4 Hz, pose e disparidade da calibração), com peso menor: ele vê a parte
    PERTO da mesa (o lidar é cego abaixo de 52°) e os raios dele limpam o que o lidar inventa.
  - o IMU DO LIDAR (lidar_servidor.py :5559) contra o IMU da PELVE: os dois dizem onde é "para cima". A diferença
    na subida é a referência; o que MUDAR depois (folga da cintura, suporte da cabeça — o que as juntas não medem)
    vira uma correção viva da cabeça (lidar e ZED juntos), limitada a 3°.

Serve em http://0.0.0.0:8603/mapa.json (as cabines :8090 e :8092 desenham): cubos ocupados, um pouco da nuvem crua,
as juntas (o modelo 3D do G1), a FOLGA (distância ao obstáculo mais próximo) das mãos e cotovelos e a MESA achada
(altura do tampo e a borda). GET /limpa zera o mapa. O mapa e a mesa são salvos em ~/mapas/ a cada 30 s e voltam
na próxima subida (--carrega 1).

    cd ~/DEV/UnifoLM-WLA-Prometheus && ~/miniforge3/envs/wla/bin/python -u lerobot-ext/wla/mapa_ocupacao.py
"""
import argparse
import base64
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import zmq

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from fk_g1 import NOMES, T_GARRA, FK, _eixo, _rpy  # noqa: E402

# mid360_joint do URDF (no torso_link) + o lidar está DE CABEÇA PARA BAIXO no G1 (medido 03/11: o chão vem em z=+1,2)
P_LIDAR = np.array([0.0002835, 0.00003, 0.41618])
R_URDF = _rpy(0.0, 0.04014257279586953, 0.0)
VIRADAS = {"rx": np.diag([1.0, -1.0, -1.0]), "ry": np.diag([-1.0, 1.0, -1.0])}

L_OCC, L_FREE, L_MIN, L_MAX, L_OCUPADO = 0.85, -0.4, -2.0, 3.5, 0.85   # 0,85 = p 0,7
Z_OCC, Z_FREE = 0.32, -0.3     # 03/11: o estéreo pesa menos que o lidar: 3 acertos para marcar (2 deixavam os
                               # pontinhos falsos da mesa branca sem textura virarem cubos)
ELEV_MIN = -43.0               # graus: abaixo disso (borda do campo do MID-360) o lidar inventa pontos
MAX_ENVIO = 150000   # cubos por /mapa.json (os mais certos primeiro)


class Robo:
    """lowstate (6001) numa thread: o último q29."""

    def __init__(self, ctx, ip):
        self.q, self.t, self.rpy = None, 0.0, None
        s = ctx.socket(zmq.SUB)
        s.setsockopt_string(zmq.SUBSCRIBE, "")
        s.setsockopt(zmq.RCVHWM, 2)
        s.connect(f"tcp://{ip}:6001")
        threading.Thread(target=self._laco, args=(s,), daemon=True).start()

    def _laco(self, s):
        while True:
            try:
                m = json.loads(s.recv().decode("utf-8"))
                m = m.get("data", m)
                self.q = np.array([float(m["motor_state"][i]["q"]) for i in range(29)])
                if m.get("imu_state"):
                    self.rpy = np.array(m["imu_state"]["rpy"], float)
                self.t = time.time()
            except Exception:  # noqa: BLE001
                time.sleep(0.01)

    def q29(self):
        vivo = self.q is not None and time.time() - self.t < 1.0
        return (self.q if vivo else np.zeros(29)), vivo


class ImuLidar:
    """03/11: o IMU do MID-360 (porta 5559): "para cima" no referencial do lidar (média lenta do acelerômetro, só com
    a cabeça parada) e quanto ela está girando."""

    def __init__(self, ctx, ip, porta=5559):
        self.cima, self.giro, self.t = None, 0.0, 0.0
        s = ctx.socket(zmq.SUB)
        s.setsockopt_string(zmq.SUBSCRIBE, "")
        s.connect(f"tcp://{ip}:{porta}")
        threading.Thread(target=self._laco, args=(s,), daemon=True, name="imu_lidar").start()

    def _laco(self, s):
        while True:
            try:
                m = json.loads(s.recv())
                acc, self.giro, self.t = np.array(m["acc"]), float(np.linalg.norm(m["gyro"])), time.time()
                if self.giro < 0.05 and 0.8 < np.linalg.norm(acc) < 1.2:   # parada: o acelerômetro mede a gravidade
                    u = acc / np.linalg.norm(acc)
                    v = u if self.cima is None else 0.97 * self.cima + 0.03 * u
                    self.cima = v / np.linalg.norm(v)
            except Exception:  # noqa: BLE001
                time.sleep(0.05)

    def vivo(self):
        return self.cima is not None and time.time() - self.t < 1.0


def gira_entre(a, b):
    from scipy.spatial.transform import Rotation
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.clip(a @ b, -1, 1))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    return Rotation.from_rotvec(v / np.linalg.norm(v) * np.arccos(c)).as_matrix()


def angulo(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def cadeia_pts(fk, q, lado):
    """Pontos do braço na pelvis: ombro, cotovelo, punho, centro da mão; e a pose do torso."""
    R, p = np.eye(3), np.zeros(3)
    pts, torso = [], None
    for j in fk.cadeia[lado]:
        p = p + R @ j["p"]
        R = R @ j["R"]
        if j["tipo"] in ("revolute", "continuous") and j["nome"] in fk.idx:
            R = R @ _eixo(j["eixo"], q[fk.idx[j["nome"]]])
        if j["nome"] == "waist_pitch_joint":
            torso = (R.copy(), p.copy())
        if any(k in j["nome"] for k in ("shoulder_pitch", "elbow", "wrist_roll")):
            pts.append(p.copy())
    pts.append(p + R @ T_GARRA[lado])
    return np.array(pts), torso


def poses_links(fk, q):
    """03/11: a pose (R, p) de TODOS os links do URDF na pelve, com as juntas medidas (dedos em 0)."""
    poses, pilha = {"pelvis": (np.eye(3), np.zeros(3))}, ["pelvis"]
    while pilha:
        pai = pilha.pop()
        Rp, pp = poses[pai]
        for filho in fk.filhos.get(pai, []):
            j = fk.junta_do_filho[filho]
            R, p_ = Rp @ j["R"], pp + Rp @ j["p"]
            if j["tipo"] in ("revolute", "continuous") and j["nome"] in fk.idx:
                R = R @ _eixo(j["eixo"], q[fk.idx[j["nome"]]])
            poses[filho] = (R, p_)
            pilha.append(filho)
    return poses


def _colisao_def():
    """O modelo de COLISÃO do G1 (03/11): cápsulas entre links vizinhos e esferas nos motores (raios em m, medidos
    a olho no robô e no URDF) — braços com o ombro inteiro, mãos Dex3 com os dedos, pernas, pés, quadril, tronco."""
    segs, esferas = [], []
    for l in ("left", "right"):
        braco = [f"{l}_shoulder_pitch_link", f"{l}_shoulder_roll_link", f"{l}_shoulder_yaw_link", f"{l}_elbow_link",
                 f"{l}_wrist_roll_link", f"{l}_wrist_pitch_link", f"{l}_wrist_yaw_link", f"{l}_hand_palm_link"]
        for a, b, r in zip(braco, braco[1:], (0.075, 0.075, 0.062, 0.055, 0.05, 0.05, 0.06)):
            segs.append((a, (0, 0, 0), b, (0, 0, 0), r))
        esferas += [(f"{l}_shoulder_pitch_link", (0, 0, 0), 0.10), (f"{l}_shoulder_roll_link", (0, 0, 0), 0.12),
                    (f"{l}_elbow_link", (0, 0, 0), 0.07),
                    (f"{l}_hand_palm_link", (0, 0, 0), 0.07)]
        perna = [f"{l}_hip_pitch_link", f"{l}_hip_roll_link", f"{l}_hip_yaw_link", f"{l}_knee_link", f"{l}_ankle_pitch_link"]
        for a, b, r in zip(perna, perna[1:], (0.08, 0.08, 0.085, 0.07)):
            segs.append((a, (0, 0, 0), b, (0, 0, 0), r))
        esferas += [(f"{l}_knee_link", (0, 0, 0), 0.08)]
        segs.append((f"{l}_ankle_roll_link", (-0.06, 0, -0.03), f"{l}_ankle_roll_link", (0.13, 0, -0.03), 0.055))   # o pé
    segs += [("pelvis", (0, 0, -0.05), "torso_link", (0, 0, 0.05), 0.13),      # quadril + cintura
             ("torso_link", (0.0, 0, 0.05), "torso_link", (0.0, 0, 0.32), 0.15),  # o tronco
             ("torso_link", (0.02, 0, 0.36), "torso_link", (0.04, 0, 0.50), 0.11)]  # pescoço + cabeça (+ lidar e ZED)
    return segs, esferas


COLISAO_SEGS, COLISAO_ESFERAS = _colisao_def()
COLISAO_FOLGA = 0.02


def corpo_colisao(fk, q, folga=COLISAO_FOLGA):
    """As cápsulas do corpo na pelve: array (K, 7) = a(3), b(3), raio (esferas: a = b). A mão vai até o centro da
    garra (fk.pose), onde ficam os dedos."""
    P = poses_links(fk, q)
    ponto = lambda link, off: P[link][1] + P[link][0] @ np.asarray(off, float)   # noqa: E731
    out = [(*ponto(a, oa), *ponto(b, ob), r + folga) for a, oa, b, ob, r in COLISAO_SEGS if a in P and b in P]
    out += [(*ponto(l, o), *ponto(l, o), r + folga) for l, o, r in COLISAO_ESFERAS if l in P]
    for lado in ("left", "right"):     # a mão Dex3 com os dedos: da palma ao centro da garra, e uma esfera lá
        m = fk.pose(q, lado)[1]
        pal = ponto(f"{lado}_hand_palm_link", (0, 0, 0))
        out += [(*pal, *m, 0.07 + folga), (*m, *m, 0.11 + folga)]
    return np.array(out, float)


def dentro_do_corpo(P, C):
    """Máscara dos pontos P (N, 3) dentro de alguma cápsula de C (K, 7)."""
    dentro = np.zeros(len(P), bool)
    if not len(C) or not len(P):
        return dentro
    lo = np.minimum(C[:, :3], C[:, 3:6]).min(0) - C[:, 6].max()   # só testa o que está na caixa do corpo
    hi = np.maximum(C[:, :3], C[:, 3:6]).max(0) + C[:, 6].max()
    k = np.nonzero(np.all((P > lo) & (P < hi), axis=1))[0]
    Pk, d = P[k], np.zeros(len(k), bool)
    for c in C:
        d |= dist_segmento(Pk, c[:3], c[3:6]) < c[6]
    dentro[k] = d
    return dentro


def dist_segmento(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


class Mapa:
    def __init__(self, a):
        self.a = a
        self.vox = a.vox
        # 03/11: com 1 cm, só a ÁREA DE TRABALHO (x -0,3..1,4 m, |y| < 0,9 m, z -0,8..+0,7 m da pelve): 4,6 M cubos
        self.origem = np.array([-0.3, -0.9, -0.8])
        self.dims = np.ceil(np.array([1.7, 1.8, 1.5]) / self.vox).astype(int)
        self.L = np.zeros(self.dims, np.float32)
        self.H = np.zeros(self.dims, np.uint16)   # 03/11: quantas vezes cada cubo foi ATINGIDO (o refino grosso -> fino)
        self.trava = threading.Lock()
        self.dist, self.mesa, self.amostra, self.folga = None, None, np.zeros((0, 3)), {}
        self.deduzidos = np.zeros((0, 3), int)
        self.taxa, self.n_lotes, self.virada = 0.0, 0, a.virada
        self.C_torso, self.cam = np.eye(3), None   # 03/11: correção do lidar e a ZED (calibra_lidar_camera.py)
        self.taxa_zed, self.zed_t = 0.0, 0.0
        self.corr, self.o_lidar = np.eye(3), np.zeros(3)   # 03/11: a correção viva da cabeça (IMU do lidar)
        self.imu = {"lidar_vivo": False}
        self.zed_amostra = None   # 03/11: (t, xyz mm int16, rgb uint8) para /vivo.json
        self.colisao = np.zeros((0, 7))   # 03/11: as cápsulas do corpo (para a página desenhar)
        arq_cal = Path(a.calib).expanduser()
        if arq_cal.exists():
            c = json.loads(arq_cal.read_text())
            self.C_torso, self.cam = np.array(c["lidar_correcao_torso"]), c["camera_retificada"]
            print(f"📐 calibração {c['quando']}: lidar {c['lidar_inclinacao_corrigida_graus']}° nivelado; ZED no torso "
                  f"{c['camera_no_torso_cm_depois']} cm, {c.get('camera_inclinacao_graus_depois')}° p/ baixo, "
                  f"disparidade {self.cam.get('delta_disparidade_px', 0):+.2f} px", flush=True)
        self.pasta = Path(a.pasta).expanduser()
        if a.carrega and (self.pasta / "ocupacao.npz").exists():
            z = np.load(self.pasta / "ocupacao.npz")
            if tuple(z["L"].shape) == tuple(self.dims) and float(z["vox"]) == self.vox:
                self.L = z["L"].astype(np.float32)
                print(f"🗺  mapa salvo carregado ({int((self.L > L_OCUPADO).sum())} cubos ocupados)", flush=True)
            if (self.pasta / "mesa.json").exists():
                self.mesa = json.load(open(self.pasta / "mesa.json"))

    # ── lidar -> pelvis ──
    def lidar_na_pelvis(self, torso, virada):
        R_tp, p_tp = torso
        R = R_tp @ self.C_torso @ R_URDF @ VIRADAS[virada]
        return R, p_tp + R_tp @ P_LIDAR

    def filtra_corpo(self, P, q, fk, o):
        braco = {l: cadeia_pts(fk, q, l)[0] for l in ("left", "right")}
        R_tp, p_tp = cadeia_pts(fk, q, "right")[1]
        fora = np.linalg.norm(P - o, axis=1) > 0.15        # zona cega do MID-360
        T = (P - p_tp) @ R_tp                                # no torso: tronco + cabeça numa caixa
        fora &= ~((T[:, 0] > -0.18) & (T[:, 0] < 0.16) & (np.abs(T[:, 1]) < 0.2) & (T[:, 2] > -0.3) & (T[:, 2] < 0.62))
        # 03/11: o MODELO DE COLISÃO do corpo inteiro (ombros, mãos com dedos, pernas, pés) — antes só os braços
        self.colisao = corpo_colisao(fk, q)
        fora[fora] = ~dentro_do_corpo(P[fora], self.colisao)
        return P[fora], braco

    def limpa_corpo(self, braco, folga=0.02):
        """03/11: os cubos DENTRO dos braços (pela FK) viram livres a cada ciclo — senão o braço que se mexe deixa um
        rastro (o lidar/ZED viram o braço antes do filtro pegar, ou com 1-2 cm de erro de calibração)."""
        for pts in braco.values():
            for i in range(len(pts)):
                a = pts[i]
                b = pts[i + 1] if i + 1 < len(pts) else pts[i]
                r = (0.11 if i == len(pts) - 2 else 0.12 if i == len(pts) - 1 else 0.075) + folga
                lo = np.floor((np.minimum(a, b) - r - self.origem) / self.vox).astype(int)
                hi = np.ceil((np.maximum(a, b) + r - self.origem) / self.vox).astype(int)
                lo, hi = np.clip(lo, 0, self.dims), np.clip(hi, 0, self.dims)
                if np.any(hi <= lo):
                    continue
                g = np.stack(np.meshgrid(*[np.arange(l, h) for l, h in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
                c = self.origem + (g + 0.5) * self.vox
                dentro = g[dist_segmento(c, a, b) < r]
                if len(dentro):
                    with self.trava:
                        self.L[tuple(dentro.T)] = np.minimum(self.L[tuple(dentro.T)], 0.0)

    def _idx(self, P):
        i = np.floor((P - self.origem) / self.vox).astype(int)
        ok = np.all((i >= 0) & (i < self.dims), axis=1)
        return i, ok

    def integra(self, P, o, l_occ=L_OCC, l_free=L_FREE, raios=1500):
        rng = np.random.default_rng()
        i, ok = self._idx(P)
        occ = np.unique(np.ravel_multi_index(i[ok].T, self.dims)) if ok.any() else np.zeros(0, int)
        # raios (no máx. 1500 por lote, até 2,2 m) para o espaço LIVRE, uma amostra a cada ~0,8 cubo
        R = P if len(P) <= raios else P[rng.choice(len(P), raios, replace=False)]
        d = R - o
        Ln = np.linalg.norm(d, axis=1)
        esc = np.minimum(1.0, 2.2 / np.maximum(Ln, 1e-6))
        K = int(np.ceil(2.2 / (0.8 * self.vox)))
        t = np.linspace(0.0, 1.0, K, endpoint=False)[None, :, None]
        fim = (esc * np.maximum(0.0, 1.0 - self.vox / np.maximum(Ln, 1e-6)))[:, None, None]
        S = (o + d[:, None, :] * t * fim).reshape(-1, 3)
        j, okj = self._idx(S)
        free = np.unique(np.ravel_multi_index(j[okj].T, self.dims)) if okj.any() else np.zeros(0, int)
        free = np.setdiff1d(free, occ, assume_unique=True)
        Lf = self.L.reshape(-1)
        with self.trava:
            Lf[free] = np.maximum(L_MIN, Lf[free] + l_free)
            Lf[occ] = np.minimum(L_MAX, Lf[occ] + l_occ)
            Hf = self.H.reshape(-1)
            Hf[occ] = np.minimum(60000, Hf[occ].astype(np.int32) + 1).astype(np.uint16)

    # ── distâncias e mesa (1 Hz) ──
    def analisa(self, braco):
        from scipy.ndimage import distance_transform_edt
        with self.trava:
            ocup = self.L > L_OCUPADO
        self.mesa = self.acha_mesa(ocup) or self.mesa
        self.deduzidos = self.completa_tampo(ocup)
        if len(self.deduzidos):   # a folga conta o tampo deduzido também (a mesa não tem furo)
            ocup = ocup.copy()
            ocup[tuple(self.deduzidos.T)] = True
        if ocup.any():
            self.dist = distance_transform_edt(~ocup) * self.vox
        folga = {}
        if self.dist is not None:
            for l, pts in braco.items():
                for nome, p in (("cotovelo", pts[1]), ("punho", pts[2]), ("mao", pts[-1])):
                    i, ok = self._idx(p[None])
                    folga[f"{nome}_{'esq' if l == 'left' else 'dir'}"] = (
                        round(float(self.dist[tuple(i[0])]), 3) if ok[0] else None)
        self.folga = folga

    def completa_tampo(self, ocup):
        """03/11: o TAMPO DEDUZIDO. Atrás dos objetos (coador, canecas) nenhum raio chega e o tampo fica desconhecido
        (o "buraco"). Dentro do contorno CONVEXO do tampo visto, cada coluna sem tampo na altura da mesa (+-1 cubo)
        ganha um cubo deduzido — não entra no log-odds (não é medida), só no desenho e na folga."""
        from scipy.spatial import Delaunay
        if not self.mesa:
            return np.zeros((0, 3), int)
        kz = int(np.floor((self.mesa["altura_pelvis_m"] - self.origem[2]) / self.vox))
        if not 1 <= kz < self.dims[2] - 1:
            return np.zeros((0, 3), int)
        camada = ocup[:, :, kz - 1:kz + 2].any(2)                      # o tampo visto, de cima
        from scipy.ndimage import label
        rot, n = label(camada)
        if n == 0:
            return np.zeros((0, 3), int)
        tam = np.bincount(rot.ravel())[1:]
        maior = rot == (1 + int(np.argmax(tam)))                       # a mesa = o maior pedaço da camada
        ij = np.argwhere(maior)
        if len(ij) < 100:
            return np.zeros((0, 3), int)
        try:
            casco = Delaunay(ij.astype(float))
        except Exception:  # noqa: BLE001
            return np.zeros((0, 3), int)
        i0, j0 = ij.min(0)
        i1, j1 = ij.max(0)
        gi, gj = np.mgrid[i0:i1 + 1, j0:j1 + 1]
        cand = np.c_[gi.ravel(), gj.ravel()]
        dentro = casco.find_simplex(cand.astype(float)) >= 0
        cand = cand[dentro & ~camada[cand[:, 0], cand[:, 1]]]
        return np.c_[cand, np.full(len(cand), kz)].astype(int)

    def acha_mesa(self, ocup):
        """O tampo: a camada de altura mais cheia à frente (x 0,15-1,2 m, |y| < 0,6 m, z -0,5..+0,3 da pelvis)."""
        I = np.argwhere(ocup)
        if len(I) < 50:
            return None
        P = self.origem + (I + 0.5) * self.vox
        m = (P[:, 0] > 0.15) & (P[:, 0] < 1.2) & (np.abs(P[:, 1]) < 0.6) & (P[:, 2] > -0.5) & (P[:, 2] < 0.3)
        if m.sum() < 40:
            return None
        h, e = np.histogram(P[m, 2], bins=np.arange(-0.5, 0.31, self.vox))
        k = int(np.argmax(h))
        if h[k] < 40:
            return None
        z = float(e[k] + self.vox / 2)
        camada = m & (np.abs(P[:, 2] - z) <= self.vox)
        Pc = P[camada]
        frente = Pc[np.abs(Pc[:, 1]) < 0.3]
        return {"t": time.time(), "altura_pelvis_m": round(z, 3), "cubos": int(h[k]),
                "borda_x_m": round(float(np.percentile(frente[:, 0], 3)), 3) if len(frente) > 10 else None,
                "y_min_m": round(float(np.percentile(Pc[:, 1], 2)), 3), "y_max_m": round(float(np.percentile(Pc[:, 1], 98)), 3),
                "x_max_m": round(float(np.percentile(Pc[:, 0], 98)), 3)}

    def salva(self):
        self.pasta.mkdir(parents=True, exist_ok=True)
        with self.trava:
            L = self.L.copy()
        np.savez_compressed(self.pasta / "ocupacao.npz", L=L, vox=self.vox, origem=self.origem, t=time.time())
        if self.mesa:
            json.dump(self.mesa, open(self.pasta / "mesa.json", "w"), indent=1)

    def limpa(self):
        with self.trava:
            self.L[:] = 0
            self.H[:] = 0
        self.dist, self.mesa, self.deduzidos = None, None, np.zeros((0, 3), int)

    REFINO = ((8, 40), (4, 16), (2, 6))   # (lado do bloco em cubos, leituras para dividir) — o resto fica em 1 cm

    def multi(self, ocup):
        """03/11: o mapa GROSSO -> FINO (como o OctoMap dos drones): cada região começa num bloco de 8 cm e só se divide
        em 4, 2 e 1 cm quando já tem leituras suficientes. Devolve (i, j, k, lado) — o canto do bloco em cubos de 1 cm."""
        with self.trava:
            H = self.H.astype(np.int32)
        f0 = self.REFINO[0][0]
        pad = [(0, (-n) % f0) for n in self.dims]
        O, H = np.pad(ocup, pad), np.pad(H * ocup, pad)
        saida, pais = [], None      # pais: máscara (no nível anterior) dos blocos que se dividiram
        for f, minimo in list(self.REFINO) + [(1, 0)]:
            D = [n // f for n in O.shape]
            Of = O.reshape(D[0], f, D[1], f, D[2], f).any((1, 3, 5))
            Hf = H.reshape(D[0], f, D[1], f, D[2], f).sum((1, 3, 5))
            vivo = Of if pais is None else Of & np.repeat(np.repeat(np.repeat(pais, 2, 0), 2, 1), 2, 2)[:D[0], :D[1], :D[2]]
            fica = vivo & (Hf < minimo) if f > 1 else vivo
            c = np.argwhere(fica)
            saida.append(np.c_[c * f, np.full(len(c), f)])
            pais = vivo & ~fica
        return np.concatenate(saida).astype(int) if saida else np.zeros((0, 4), int)

    def json(self, juntas, vivo, lidar_vivo):
        with self.trava:
            L = self.L.copy()
        ocup = L > L_OCUPADO
        # 03/11: para o DESENHO, só cubos com >= 2 vizinhos ocupados (os soltos são ruído do estéreo)
        from scipy.ndimage import convolve
        viz = convolve(ocup.astype(np.uint8), np.ones((3, 3, 3), np.uint8), mode="constant")
        I = np.argwhere(ocup & (viz >= 3))
        n_total = int(ocup.sum())
        M = self.multi(ocup & (viz >= 3))
        multi_b64 = base64.b64encode(M.astype(np.uint8).tobytes()).decode()
        niveis = {int(f): int((M[:, 3] == f).sum()) for f in (8, 4, 2, 1)}
        if len(I) > MAX_ENVIO:
            I = I[np.argsort(-L[tuple(I.T)])[:MAX_ENVIO]]
        # 03/11: os índices em uint8 (cada eixo < 256) e base64 — com 1 cm são dezenas de milhares de cubos
        b64 = base64.b64encode(I.astype(np.uint8).tobytes()).decode()
        ded = base64.b64encode(self.deduzidos.astype(np.uint8).tobytes()).decode()
        return {"pronto": True, "t": time.time(), "vox": self.vox, "origem": self.origem.round(3).tolist(),
                "dims": self.dims.tolist(), "ocupados_b64": b64, "n_ocupados": int(n_total), "n_enviados": int(len(I)),
                "deduzidos_b64": ded, "n_deduzidos": int(len(self.deduzidos)),
                "multi_b64": multi_b64, "niveis": niveis,
                "nuvem": np.round(self.amostra, 3).ravel().tolist(), "juntas": juntas, "robo_vivo": vivo,
                "lidar_vivo": lidar_vivo, "pontos_s": round(self.taxa), "folga": self.folga, "mesa": self.mesa,
                "virada": self.virada, "zed_vivo": time.time() - self.zed_t < 2.0, "zed_hz": round(self.taxa_zed, 1),
                "calibrado": self.cam is not None, "imu": self.imu}


def laco_zed(a, mapa, robo, fk, periodo=0.25):
    """03/11: o estéreo da ZED no mapa — profundidade retificada (estereo_zed.py), a disparidade e a pose da
    calibração, a cada ~0,25 s; raios a partir da câmera (peso menor que o lidar)."""
    sys.path.insert(0, str(AQUI / "zed"))
    sys.path.insert(0, str(AQUI.parent))
    from estereo_zed import EstereoZED
    from robot.Scripts_Prometheus_int.sim.sensor_utils import ImageUtils, SensorClient
    z, c = EstereoZED(), mapa.cam
    fx, cx, cy, delta = c["fx"], c["cx"], c["cy"], c.get("delta_disparidade_px", 0.0)
    fB = fx * z.B
    R_tc, o_c = np.array(c["R_torso_camera"]), np.array(c["o_no_torso"])
    cli = SensorClient()
    cli.start_client(server_ip=a.robo, port=5555)
    n, t0, ult = 0, time.time(), 0.0
    while True:
        try:
            msg = cli.receive_message()
            if time.time() - ult < periodo:
                continue
            ims = msg.get("images") or {}
            if "head_stereo_left" not in ims or "head_stereo_right" not in ims:
                continue
            ult = time.time()
            q, vivo = robo.q29()
            esq, dir_ = ImageUtils.decode_image(ims["head_stereo_left"]), ImageUtils.decode_image(ims["head_stereo_right"])
            Z = z.profundidade(esq, dir_, retificada=True)
            cor = z.retifica(esq, dir_)[0][::3, ::3]                   # RGB na mesma grade retificada
            Z = Z[::3, ::3]
            v, u = np.mgrid[0:Z.shape[0], 0:Z.shape[1]] * 3
            Z = fB * Z / (fB + delta * Z)                              # a disparidade da calibração
            ok = np.isfinite(Z) & (Z > 0.2) & (Z < 1.2)
            Fc = np.c_[(u[ok] - cx) / fx * Z[ok], (v[ok] - cy) / fx * Z[ok], Z[ok]]
            R_tp, p_tp = cadeia_pts(fk, q, "right")[1]
            o = p_tp + R_tp @ o_c
            W = Fc @ (R_tp @ R_tc).T + o
            ol = mapa.o_lidar                                          # a correção viva da cabeça (em torno do lidar)
            W, o = (W - ol) @ mapa.corr.T + ol, mapa.corr @ (o - ol) + ol
            C = cor[ok]
            m_ = (W[:, 2] > -0.85) & (W[:, 2] < 0.9)                  # para a página: a nuvem com cor, sem o chão
            if m_.sum():
                k_ = np.random.default_rng().choice(int(m_.sum()), min(4000, int(m_.sum())), replace=False)
                mapa.zed_amostra = (time.time(), np.round(W[m_][k_] * 1000).astype(np.int16), C[m_][k_].astype(np.uint8))
            W, _ = mapa.filtra_corpo(W, q, fk, o)
            mapa.integra(W, o, Z_OCC, Z_FREE, raios=2500)
            mapa.zed_t, n = time.time(), n + 1
            if mapa.zed_t - t0 > 4:
                mapa.taxa_zed, n, t0 = n / (mapa.zed_t - t0), 0, mapa.zed_t
        except Exception as e:  # noqa: BLE001
            print(f"   (zed: {e!r})", flush=True)
            time.sleep(1)


def escolhe_virada(fk, q, P, mapa):
    """Qual das duas viradas põe a nuvem nas mãos (pela FK)? Conta pontos a < 6 cm do centro de cada mão."""
    torso = cadeia_pts(fk, q, "right")[1]
    melhor = {}
    for v in VIRADAS:
        R, o = mapa.lidar_na_pelvis(torso, v)
        W = P[:, :3] @ R.T + o
        melhor[v] = sum(int((np.linalg.norm(W - cadeia_pts(fk, q, l)[0][-1], axis=1) < 0.06).sum()) for l in ("left", "right"))
    return max(melhor, key=melhor.get), melhor


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robo", default="192.168.123.164")
    ap.add_argument("--porta", type=int, default=8603)
    ap.add_argument("--vox", type=float, default=0.01, help="lado do cubo (m); a grade tem que ter < 256 por eixo")
    ap.add_argument("--virada", default="auto", choices=("auto", *VIRADAS),
                    help="como o lidar está de cabeça para baixo; auto = acha pelas mãos")
    ap.add_argument("--pasta", default="~/mapas")
    ap.add_argument("--carrega", type=int, default=1)
    ap.add_argument("--calib", default="~/mapas/calib_lidar_camera.json", help="calibra_lidar_camera.py")
    ap.add_argument("--zed", type=int, default=1, help="1 = o estéreo da ZED entra no mapa (precisa da calibração)")
    a = ap.parse_args()
    ctx = zmq.Context.instance()
    fk, robo, mapa = FK(), Robo(ctx, a.robo), Mapa(a)
    imu_l = ImuLidar(ctx, a.robo)
    ref = None   # a diferença lidar x pelve na subida (vício do IMU + montagem): só a MUDANÇA depois vira correção
    if np.any(mapa.dims > 255):
        sys.exit(f"grade {mapa.dims.tolist()} passa de 255 por eixo: aumente --vox")
    if a.virada == "auto":
        mapa.virada = "rx"
    if mapa.cam is not None:   # calibrado: a virada já é conhecida (a calibração foi feita com "rx")
        a.virada = "rx"
        if a.zed:
            threading.Thread(target=laco_zed, args=(a, mapa, robo, fk), daemon=True, name="zed").start()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.setsockopt(zmq.RCVHWM, 4)
    sub.setsockopt(zmq.RCVTIMEO, 2000)
    sub.connect(f"tcp://{a.robo}:5558")
    estado = {"juntas": {}, "vivo": False, "lidar_t": 0.0}
    _sobe_http(a.porta, mapa, estado)
    print(f"🗺  mapa de ocupação em http://0.0.0.0:{a.porta}/mapa.json (lidar {a.robo}:5558, juntas :6001)", flush=True)
    t_an = t_salva = t_taxa = time.time()
    n_taxa, decidida, votos = 0, a.virada != "auto", {v: 0 for v in VIRADAS}
    analisando = threading.Event()
    while True:
        try:
            cab, b = sub.recv_multipart()
        except zmq.Again:
            continue
        P = np.frombuffer(b, np.float32).reshape(-1, 4)
        q, vivo = robo.q29()
        estado.update(juntas={n: round(float(v), 4) for n, v in zip(NOMES, q)}, vivo=vivo, lidar_t=time.time())
        if not decidida and vivo:
            v, c = escolhe_virada(fk, q, P, mapa)
            votos[v] += 1
            if sum(votos.values()) >= 20:
                mapa.virada, decidida = max(votos, key=votos.get), True
                print(f"   lidar de cabeça para baixo: virada '{mapa.virada}' (votos {votos})", flush=True)
                mapa.limpa()
        torso = cadeia_pts(fk, q, "right")[1]
        R, o = mapa.lidar_na_pelvis(torso, mapa.virada)
        if imu_l.vivo() and robo.rpy is not None:
            from scipy.spatial.transform import Rotation as Rt
            cima_pelve = Rt.from_euler("xy", robo.rpy[:2]).as_matrix().T @ np.array([0, 0, 1.0])
            D = gira_entre(R @ imu_l.cima, cima_pelve)                    # "para cima" do lidar (na pelve) -> o da pelve
            if ref is None and imu_l.giro < 0.03:
                ref = D
                print(f"   IMU do lidar x IMU da pelve: {angulo(D):.2f}° de diferença na subida (referência)", flush=True)
            if ref is not None:
                alvo = D @ ref.T                                          # o que mudou desde a subida
                if angulo(alvo) > 3.0:
                    alvo = np.eye(3)                                      # grande demais: não confia (cabeça mexendo?)
                rv = 0.9 * Rt.from_matrix(mapa.corr).as_rotvec() + 0.1 * Rt.from_matrix(alvo).as_rotvec()
                mapa.corr = Rt.from_rotvec(rv).as_matrix()
            mapa.imu = {"lidar_vivo": True, "lidar_x_pelve_graus": round(angulo(D), 2),
                        "ajuste_vivo_graus": round(angulo(mapa.corr), 2), "giro_cabeca_rad_s": round(imu_l.giro, 3)}
        else:
            mapa.imu = {"lidar_vivo": False}
        R, mapa.o_lidar = mapa.corr @ R, o
        if mapa.cam is not None:   # sem a faixa de baixo do campo (lidar de pé: y e z trocam de sinal)
            up = P[:, :3] * np.float32([1, -1, -1])
            P = P[np.degrees(np.arctan2(up[:, 2], np.hypot(up[:, 0], up[:, 1]))) > ELEV_MIN]
        W = P[:, :3] @ R.T + o
        W, braco = mapa.filtra_corpo(W, q, fk, o)
        if decidida or not vivo:
            mapa.integra(W, o)
        n_lote = getattr(mapa, "n_lote", 0) + 1
        mapa.n_lote = n_lote
        if vivo and n_lote % 3 == 0:
            mapa.limpa_corpo(braco)
        k = min(len(W), 2000)
        mapa.amostra = W[np.random.default_rng().choice(len(W), k, replace=False)] if k else np.zeros((0, 3))
        n_taxa += len(P)
        agora = time.time()
        if agora - t_taxa > 2:
            mapa.taxa, n_taxa, t_taxa = n_taxa / (agora - t_taxa), 0, agora
        if agora - t_an > 2 and not analisando.is_set():   # a EDT de milhões de cubos fora do laço do lidar
            analisando.set()
            threading.Thread(target=lambda b=braco: (mapa.analisa(b), analisando.clear()), daemon=True).start()
            t_an = agora
        if agora - t_salva > 30:
            mapa.salva()
            t_salva = agora


def _sobe_http(porta, mapa, estado):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path.startswith("/vivo.json"):   # 03/11: leve (a página pede a ~4 Hz): juntas + a ZED com cor
                z = mapa.zed_amostra
                d = {"t": time.time(), "juntas": estado["juntas"], "robo_vivo": estado["vivo"],
                     "colisao": np.round(mapa.colisao, 3).tolist(),
                     "zed": None if z is None or time.time() - z[0] > 2 else
                     {"t": z[0], "mm_b64": base64.b64encode(z[1].tobytes()).decode(), "rgb_b64": base64.b64encode(z[2].tobytes()).decode()}}
            elif self.path.startswith("/limpa"):
                mapa.limpa()
                d = {"ok": True}
            elif self.path.startswith("/mapa.json"):
                d = mapa.json(estado["juntas"], estado["vivo"], time.time() - estado["lidar_t"] < 1.0)
            else:
                self.send_response(404)
                self.end_headers()
                return
            b = json.dumps(d, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

    srv = ThreadingHTTPServer(("0.0.0.0", porta), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True, name="mapa_http").start()


if __name__ == "__main__":
    main()
