#!/usr/bin/env python
"""MAPA 3D COM O NVBLOX (03/11) — troca o miolo do mapa_ocupacao.py pelo nvblox da NVIDIA (TSDF + ESDF na GPU).

Roda no ambiente SEPARADO do nvblox (~/nvblox_teste/env: Python 3.12 + torch 2.9.1 cu130; ver sobe_mapa_nvblox.sh).
Tudo o que estava em volta continua igual e vem do mapa_ocupacao.py: a pose (juntas + IMU da pelve + IMU do lidar),
a calibração lidar+ZED (~/mapas/calib_lidar_camera.json), o filtro do corpo, a mesa, o tampo deduzido, o grosso->fino
e o HTTP (:8603 /mapa.json, /vivo.json, /limpa) — as cabines não mudam.

O que muda:
  - o LIDAR vira uma imagem de distância (azimute x elevação, o alcance em cada pixel; ±52° porque o modelo de lidar do
    nvblox é simétrico e o MID-360 de cabeça para baixo vê -52..+7°) a cada ~0,3 s, sem os pontos do corpo;
  - a ZED entra como câmera de profundidade (a retificada, com a disparidade da calibração) + a COR, com o corpo
    mascarado;
  - o nvblox faz a média das medidas em TSDF (superfícies lisas, sem os pontinhos do estéreo) e a ESDF dá a distância
    de qualquer ponto ao obstáculo mais próximo — a FOLGA dos braços sai direto dela, descontando o raio de cada peça;
  - /mapa.json ganha a MALHA colorida da superfície ("malha": vértices em mm, triângulos, cor).

    ~/nvblox_teste/env/bin/python -u lerobot-ext/wla/mapa_nvblox.py        (ou: bash lerobot-ext/wla/sobe_mapa_nvblox.sh)
"""
import argparse
import base64
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import zmq

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI / "zed"))
sys.path.insert(0, str(AQUI.parent))
import mapa_ocupacao as mo  # noqa: E402
from fk_g1 import NOMES, FK  # noqa: E402
from nvblox_torch.mapper import Mapper, QueryType  # noqa: E402
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType  # noqa: E402
from nvblox_torch.mapper_params import MapperParams  # noqa: E402
from nvblox_torch.sensor import Sensor  # noqa: E402

FOV_LIDAR = np.radians(104.0)      # ±52°
N_AZ, N_EL = 1440, 104             # 0,25° x 1° (o nvblox quer altura e largura múltiplas de 8)
LIDAR_JANELA = 0.3                 # s de pontos por imagem (a varredura do MID-360 não se repete: precisa juntar)
MAX_TRI = 200000                   # triângulos por /mapa.json
ZED_MAX = 0.9                      # m (ver laco_zed)
RAIO = {"cotovelo": 0.075, "punho": 0.06, "mao": 0.10}   # para a folga: distância da SUPERFÍCIE da peça


class MapaNv(mo.Mapa):
    def __init__(self, a):
        a.carrega = 0                              # o mapa antigo (log-odds) não serve aqui
        super().__init__(a)
        self.gpu = threading.Lock()
        # 03/11: UM MAPA POR SENSOR (0 = lidar, 1 = ZED): num mapa só, a ZED (11 cm errada a ~1 m) e o lidar
        # viam o mesmo chão em alturas diferentes, o TSDF fazia a média das duas e a superfície sumia
        # 03/11: só vira SUPERFÍCIE o que o TSDF juntou com peso (o padrão 1e-4 fazia de 1 leitura ruidosa um bloco)
        par = MapperParams()
        mi = par.get_mesh_integrator_params()
        mi.mesh_integrator_min_weight = a.peso_min
        par.set_mesh_integrator_params(mi)
        self.nv = Mapper(voxel_sizes_m=[a.vox, a.vox],
                         integrator_types=[ProjectiveIntegratorType.TSDF, ProjectiveIntegratorType.TSDF],
                         mapper_parameters=par)
        self.ID_LIDAR, self.ID_ZED = 0, 1
        # 03/11: ERRO DO nvblox_torch (py_sensor.cu fromLidar): ele repassa (vertical_fov, min_range) para o C++
        # Lidar(num_az, num_el, min_valid_range_m, vertical_fov_rad) — TROCADOS. Por isso a ordem "ao contrário" aqui.
        self.sensor_lidar = Sensor.from_lidar(N_AZ, N_EL, 0.15, float(FOV_LIDAR))
        self.rpe = FOV_LIDAR / (N_EL - 1)
        self.rpa = 2 * np.pi / N_AZ
        self.polar0 = np.pi / 2 - (FOV_LIDAR / 2 + self.rpe / 2)
        self.az0 = -np.pi - self.rpa / 2
        self.malha = None                          # (v_mm int16, tri uint32, cor uint8)
        self.ms = {}

    # ── entradas ──
    def integra_lidar(self, U, T_wU):
        """U: pontos (N, 3) no referencial do lidar DE PÉ (x frente, z cima), já sem o corpo."""
        if len(U) < 100:
            return
        r = np.linalg.norm(U, axis=1)
        v = np.floor((np.arccos(np.clip(U[:, 2] / np.maximum(r, 1e-6), -1, 1)) - self.polar0) / self.rpe).astype(int)
        u = np.floor((np.arctan2(U[:, 1], U[:, 0]) - self.az0) / self.rpa).astype(int) % N_AZ
        ok = (v >= 0) & (v < N_EL) & (r > 0.15)
        img = np.full(N_EL * N_AZ, np.inf, np.float32)
        np.minimum.at(img, v[ok] * N_AZ + u[ok], r[ok].astype(np.float32))   # o mais perto em cada pixel
        img[~np.isfinite(img)] = 0.0                                          # 0 = sem medida (não limpa nada)
        t0 = time.time()
        with self.gpu:
            self.nv.add_depth_frame(torch.from_numpy(img.reshape(N_EL, N_AZ)).cuda(),
                                    torch.from_numpy(T_wU.astype(np.float32)), self.sensor_lidar, mapper_id=self.ID_LIDAR)
        self.ms["lidar"] = round((time.time() - t0) * 1000, 1)

    def integra_zed(self, Z, rgb, T_wc, sensor):
        t0 = time.time()
        with self.gpu:
            pose = torch.from_numpy(T_wc.astype(np.float32))
            self.nv.add_depth_frame(torch.from_numpy(Z.astype(np.float32)).cuda(), pose, sensor, mapper_id=self.ID_ZED)
            self.nv.add_color_frame(torch.from_numpy(np.ascontiguousarray(rgb)).cuda(), pose, sensor, mapper_id=self.ID_ZED)
        self.ms["zed"] = round((time.time() - t0) * 1000, 1)

    # ── saídas ──
    def analisa(self, braco):
        """A malha (superfície do TSDF), a grade de 1 cm para a mesa/grosso->fino e a folga pela ESDF."""
        t0 = time.time()
        Vs, Ts, Cs, base = [], [], [], 0
        with self.gpu:
            self.nv.update_color_mesh()
            for mid in (self.ID_LIDAR, self.ID_ZED):   # as duas malhas juntas (o lidar em cinza, a ZED com cor)
                m = self.nv.get_color_mesh(mid)
                V_, T_ = m.vertices().cpu().numpy(), m.triangles().cpu().numpy()
                C_ = m.vertex_colors().cpu().numpy() if mid == self.ID_ZED else np.full((len(V_), 3), 150, np.uint8)
                if len(V_):
                    Vs.append(V_); Ts.append(T_ + base); Cs.append(C_); base += len(V_)
        V = np.concatenate(Vs) if Vs else np.zeros((0, 3), np.float32)
        T = np.concatenate(Ts) if Ts else np.zeros((0, 3), np.int64)
        C = np.concatenate(Cs) if Cs else np.zeros((0, 3), np.uint8)
        # FOLGA: distância de cada peça do braço à superfície mais próxima (os vértices das duas malhas), menos o raio.
        # (a ESDF do nvblox não responde dentro do corpo: lá nunca é "visto", o corpo é mascarado)
        folga = {}
        if len(V):
            from scipy.spatial import cKDTree
            arv = cKDTree(V)
            for l, p in braco.items():
                lado = "esq" if l == "left" else "dir"
                for nome, k in (("cotovelo", 1), ("punho", 2), ("mao", -1)):
                    folga[f"{nome}_{lado}"] = round(max(0.0, float(arv.query(p[k])[0]) - RAIO[nome]), 3)
        self.folga = folga
        # só a área de trabalho (a mesma grade do mapa antigo)
        dentro = np.all((V >= self.origem) & (V < self.origem + self.dims * self.vox), axis=1)
        if len(T):
            T = T[dentro[T].all(1)]
        if len(T) > MAX_TRI:
            T = T[np.random.default_rng().choice(len(T), MAX_TRI, replace=False)]
        usados = np.unique(T.ravel()) if len(T) else np.zeros(0, int)
        nova = np.full(len(V), -1, np.int64)
        nova[usados] = np.arange(len(usados))
        self.malha = (np.round(V[usados] * 1000).astype(np.int16), nova[T].astype(np.uint32),
                      C[usados].astype(np.uint8))
        # grade de 1 cm pelos vértices: alimenta a mesa, o tampo deduzido e o grosso->fino do resto do código
        i, ok = self._idx(V[dentro])
        ocup = np.zeros(self.dims, bool)
        H = np.zeros(self.dims, np.uint16)
        if ok.any():
            ii = i[ok]
            ocup[tuple(ii.T)] = True
            np.add.at(H, tuple(ii.T), 1)
        with self.trava:
            self.L = np.where(ocup, mo.L_MAX, 0).astype(np.float32)
            self.H = np.minimum(H.astype(np.int32) * 4, 60000).astype(np.uint16)
        self.mesa = self.acha_mesa(ocup) or self.mesa
        self.deduzidos = self.completa_tampo(ocup)
        self.ms["analise"] = round((time.time() - t0) * 1000, 1)

    def limpa(self):
        super().limpa()
        with self.gpu:
            self.nv.clear()
        self.malha = None

    def salva(self):
        pass   # o mapa é refeito em segundos; não guarda o log-odds antigo

    def json(self, juntas, vivo, lidar_vivo):
        d = super().json(juntas, vivo, lidar_vivo)
        m = self.malha
        if m is not None:
            d["malha"] = {"v_mm_b64": base64.b64encode(m[0].tobytes()).decode(),
                          "tri_b64": base64.b64encode(m[1].tobytes()).decode(),
                          "cor_b64": base64.b64encode(m[2].tobytes()).decode(),
                          "n_v": int(len(m[0])), "n_tri": int(len(m[1]))}
        d["motor"] = "nvblox (lidar + ZED em mapas separados)" if self.zed_t else "nvblox (só o lidar)"
        d["ms"] = self.ms
        return d


def laco_zed(a, mapa, robo, fk, periodo=0.25):
    """A ZED no nvblox: profundidade retificada (com a disparidade da calibração) + cor, o corpo mascarado."""
    from estereo_zed import EstereoZED
    from robot.Scripts_Prometheus_int.sim.sensor_utils import ImageUtils, SensorClient
    z, c = EstereoZED(), mapa.cam
    fx, cx, cy, delta = c["fx"], c["cx"], c["cy"], c.get("delta_disparidade_px", 0.0)
    fB = fx * z.B
    R_tc, o_c = np.array(c["R_torso_camera"]), np.array(c["o_no_torso"])
    cli = SensorClient()
    cli.start_client(server_ip=a.robo, port=5555)
    sensor = None
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
            cor = z.retifica(esq, dir_)[0]
            # o nvblox quer a altura múltipla de 8 (sphere tracer): corta igual em cima e embaixo e acerta o centro
            corte = Z.shape[0] % 8
            c0 = corte // 2
            Z, cor = Z[c0:Z.shape[0] - (corte - c0)], cor[c0:cor.shape[0] - (corte - c0)]
            cy_ = cy - c0
            Hh, Ww = Z.shape
            if sensor is None:
                sensor = Sensor.from_camera(fx, fx, cx, cy_, Ww, Hh)
                print(f"   ZED no nvblox: {Ww}x{Hh}, f {fx:.1f}, centro ({cx:.1f}, {cy_:.1f})", flush=True)
            Z = fB * Z / (fB + delta * Z)
            # 03/11: só até ZED_MAX: a calibração (disparidade + inclinação) foi feita na mesa, 0,5-0,9 m; a 1-1,2 m o
            # chão saía 11 cm alto. Mais longe fica com o lidar.
            Z[~np.isfinite(Z) | (Z < 0.2) | (Z > ZED_MAX)] = 0.0
            R_tp, p_tp = mo.cadeia_pts(fk, q, "right")[1]
            ol = mapa.o_lidar
            R_pc = mapa.corr @ (R_tp @ R_tc)
            o = mapa.corr @ (p_tp + R_tp @ o_c - ol) + ol
            # o corpo: os pixels cujo ponto cai dentro dos braços/tronco ficam sem medida
            vv, uu = np.nonzero(Z > 0)
            if len(vv):
                Zv = Z[vv, uu]
                W = np.c_[(uu - cx) / fx * Zv, (vv - cy_) / fx * Zv, Zv] @ R_pc.T + o
                _, braco = mapa.filtra_corpo(W[:1], q, fk, o)
                fora = np.ones(len(W), bool)
                Rt_, pt_ = R_tp, p_tp
                Tt = (W - pt_) @ Rt_
                fora &= ~((Tt[:, 0] > -0.18) & (Tt[:, 0] < 0.16) & (np.abs(Tt[:, 1]) < 0.2) & (Tt[:, 2] > -0.3) & (Tt[:, 2] < 0.62))
                for pts in braco.values():
                    for i in range(len(pts) - 1):
                        fora &= mo.dist_segmento(W, pts[i], pts[i + 1]) > (0.11 if i == len(pts) - 2 else 0.075)
                    fora &= np.linalg.norm(W - pts[-1], axis=1) > 0.12
                Z[vv[~fora], uu[~fora]] = 0.0
                # a nuvem colorida para a página (/vivo.json), como antes
                k = np.nonzero(fora)[0]
                if len(k):
                    k = np.random.default_rng().choice(k, min(4000, len(k)), replace=False)
                    mapa.zed_amostra = (time.time(), np.round(W[k] * 1000).astype(np.int16), cor[vv[k], uu[k]].astype(np.uint8))
            T_wc = np.eye(4)
            T_wc[:3, :3], T_wc[:3, 3] = R_pc, o
            mapa.integra_zed(Z, cor, T_wc, sensor)
            mapa.zed_t, n = time.time(), n + 1
            if mapa.zed_t - t0 > 4:
                mapa.taxa_zed, n, t0 = n / (mapa.zed_t - t0), 0, mapa.zed_t
        except Exception as e:  # noqa: BLE001
            print(f"   (zed: {e!r})", flush=True)
            time.sleep(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robo", default="192.168.123.164")
    ap.add_argument("--porta", type=int, default=8603)
    ap.add_argument("--vox", type=float, default=0.01)
    ap.add_argument("--pasta", default="~/mapas")
    ap.add_argument("--calib", default="~/mapas/calib_lidar_camera.json")
    ap.add_argument("--zed", type=int, default=0,
                    help="03/11: 0 = SÓ O LIDAR (padrão: o estéreo da ZED errava ~11 cm a 1 m e inventava pontos na mesa "
                         "branca); 1 = a ZED também, num mapa separado, até 0,9 m")
    ap.add_argument("--lidar", type=int, default=1, help="0 = só a ZED (para depurar)")
    ap.add_argument("--peso-min", type=float, default=2.0, help="peso mínimo do TSDF para virar malha (máx. 5)")
    a = ap.parse_args()
    a.virada, a.carrega = "rx", 0
    ctx = zmq.Context.instance()
    fk, robo, mapa = FK(), mo.Robo(ctx, a.robo), MapaNv(a)
    if mapa.cam is None:
        sys.exit("sem a calibração (~/mapas/calib_lidar_camera.json): rode calibra_lidar_camera.py")
    imu_l = mo.ImuLidar(ctx, a.robo)
    ref = None
    if a.zed:
        threading.Thread(target=laco_zed, args=(a, mapa, robo, fk), daemon=True, name="zed").start()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.setsockopt(zmq.RCVHWM, 4)
    sub.setsockopt(zmq.RCVTIMEO, 2000)
    sub.connect(f"tcp://{a.robo}:5558")
    estado = {"juntas": {}, "vivo": False, "lidar_t": 0.0}
    mo._sobe_http(a.porta, mapa, estado)
    print(f"🗺  mapa NVBLOX em http://0.0.0.0:{a.porta}/mapa.json (GPU {torch.cuda.get_device_name(0)})", flush=True)
    t_an = t_taxa = time.time()
    n_taxa, junta, junta_t0 = 0, [], time.time()
    analisando = threading.Event()
    from scipy.spatial.transform import Rotation as Rt
    while True:
        try:
            _, b = sub.recv_multipart()
        except zmq.Again:
            continue
        P = np.frombuffer(b, np.float32).reshape(-1, 4)
        q, vivo = robo.q29()
        estado.update(juntas={n: round(float(v), 4) for n, v in zip(NOMES, q)}, vivo=vivo, lidar_t=time.time())
        torso = mo.cadeia_pts(fk, q, "right")[1]
        R, o = mapa.lidar_na_pelvis(torso, "rx")
        if imu_l.vivo() and robo.rpy is not None:   # a correção viva da cabeça (a mesma do mapa_ocupacao.py)
            cima_pelve = Rt.from_euler("xy", robo.rpy[:2]).as_matrix().T @ np.array([0, 0, 1.0])
            D = mo.gira_entre(R @ imu_l.cima, cima_pelve)
            if ref is None and imu_l.giro < 0.03:
                ref = D
                print(f"   IMU do lidar x IMU da pelve: {mo.angulo(D):.2f}° na subida (referência)", flush=True)
            if ref is not None:
                alvo = D @ ref.T
                if mo.angulo(alvo) > 3.0:
                    alvo = np.eye(3)
                rv = 0.9 * Rt.from_matrix(mapa.corr).as_rotvec() + 0.1 * Rt.from_matrix(alvo).as_rotvec()
                mapa.corr = Rt.from_rotvec(rv).as_matrix()
            mapa.imu = {"lidar_vivo": True, "lidar_x_pelve_graus": round(mo.angulo(D), 2),
                        "ajuste_vivo_graus": round(mo.angulo(mapa.corr), 2), "giro_cabeca_rad_s": round(imu_l.giro, 3)}
        else:
            mapa.imu = {"lidar_vivo": False}
        R, mapa.o_lidar = mapa.corr @ R, o
        up = P[:, :3] * np.float32([1, -1, -1])
        # sem a "rampa": os pontos falsos estavam na faixa de baixo do campo E perto (0,45-0,63 m); o chão nessa
        # faixa fica a ~1,6 m e precisa entrar
        elev = np.degrees(np.arctan2(up[:, 2], np.hypot(up[:, 0], up[:, 1])))
        P = P[(elev > mo.ELEV_MIN) | (np.linalg.norm(P[:, :3], axis=1) > 0.8)]
        W = P[:, :3] @ R.T + o
        Wf, braco = mapa.filtra_corpo(W, q, fk, o)
        k = min(len(Wf), 2000)
        mapa.amostra = Wf[np.random.default_rng().choice(len(Wf), k, replace=False)] if k else np.zeros((0, 3))
        # para o nvblox: no referencial do lidar DE PÉ, juntando ~0,3 s
        junta.append(Wf)                         # na pelve; vira o referencial do lidar com a pose do fim da janela
        if time.time() - junta_t0 >= LIDAR_JANELA and not a.lidar:
            junta, junta_t0 = [], time.time()
        if time.time() - junta_t0 >= LIDAR_JANELA:
            R_wU = R @ np.diag([1.0, -1.0, -1.0])
            T_wU = np.eye(4)
            T_wU[:3, :3], T_wU[:3, 3] = R_wU, o
            mapa.integra_lidar((np.concatenate(junta) - o) @ R_wU, T_wU)   # U = R_wU^T (W - o)
            junta, junta_t0 = [], time.time()
        n_taxa += len(P)
        agora = time.time()
        if agora - t_taxa > 2:
            mapa.taxa, n_taxa, t_taxa = n_taxa / (agora - t_taxa), 0, agora
        if agora - t_an > 1.0 and not analisando.is_set():
            analisando.set()
            threading.Thread(target=lambda b_=braco: (mapa.analisa(b_), analisando.clear()), daemon=True).start()
            t_an = agora


if __name__ == "__main__":
    main()
