#!/usr/bin/env python
"""CALIBRAÇÃO LIDAR + ZED (03/11). Roda na PGX com o robô PARADO na frente da mesa; só lê o robô.

  1. NIVELA O LIDAR: o chão que o lidar vê (plano por RANSAC) tem que ser perpendicular à gravidade que o IMU da
     pelve mede. A diferença (roll/pitch pequenos) vira uma correção da montagem do lidar no torso.
  2. ALINHA A ZED AO LIDAR: a nuvem do estéreo (grade RETIFICADA do olho esquerdo, pinhole exato do SN19436.conf)
     vai para a pelve com a pose atual da câmera e é encaixada na nuvem do lidar (já nivelada) por ICP
     ponto-a-plano robusto. A correção vira a nova pose da câmera retificada no torso.

Salva em ~/mapas/calib_lidar_camera.json (o mapa_ocupacao.py lê). NÃO mexe no ~/servo_camera.json.

    cd ~/DEV/UnifoLM-WLA-Prometheus && ~/miniforge3/envs/wla/bin/python lerobot-ext/wla/calibra_lidar_camera.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import zmq
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI / "zed"))
sys.path.insert(0, str(AQUI.parent))
import mapa_ocupacao as mo  # noqa: E402


def gira_entre(a, b):
    """Menor rotação que leva o vetor unitário a em b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    return Rotation.from_rotvec(v / np.linalg.norm(v) * np.arccos(np.clip(c, -1, 1))).as_matrix()


def plano_ransac(P, it=400, tol=0.015, rng=np.random.default_rng(0)):
    melhor, nm = None, 0
    for _ in range(it):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-6:
            continue
        n /= np.linalg.norm(n)
        k = int((np.abs((P - a) @ n) < tol).sum())
        if k > nm:
            melhor, nm = (n, a), k
    n, a = melhor
    dentro = P[np.abs((P - a) @ n) < tol]
    c = dentro.mean(0)
    n = np.linalg.svd(dentro - c, full_matrices=False)[2][-1]   # refina por PCA (sem a matriz N x N!)
    return (n if n[2] > 0 else -n), c, len(dentro)


def normais(P, k=16):
    arv = cKDTree(P)
    _, idx = arv.query(P, k)
    Q = P[idx] - P[idx].mean(1, keepdims=True)
    C = np.einsum("nki,nkj->nij", Q, Q)
    w, v = np.linalg.eigh(C)
    return v[:, :, 0], w[:, 0] / np.maximum(w.sum(1), 1e-12), arv


def icp_ponto_plano(F, alvo, n_alvo, arv, iters=50, d0=0.08, d1=0.015, peso=None, centro=None,
                    sig_res=0.005, sig_rot=np.radians(3.0), sig_t=0.03, Zc=None, fB=None, sig_delta=4.0):
    """Acha a correção que leva F ao alvo: X = R k(delta) (F - c) + c + t, com o giro em torno da câmera c.
    03/11: k = fB / (fB + delta Zc) é o DESLOCAMENTO DE DISPARIDADE do estéreo (delta px; longe o erro cresce com Z²);
    sem Zc/fB, delta fica 0 (rígido). Ponto-a-plano, Huber, busca que encolhe de d0 a d1 e PRIOR em cada parâmetro
    (o prior vale "todos os pontos a 1 sigma" quando a correção chega em sig_*)."""
    c = np.zeros(3) if centro is None else np.asarray(centro, float)
    R, t, delta = np.eye(3), np.zeros(3), 0.0
    peso = np.ones(len(F)) if peso is None else peso
    usa_delta = Zc is not None and fB is not None
    Pr = np.diag([1 / sig_rot ** 2] * 3 + [1 / sig_t ** 2] * 3 + [1 / sig_delta ** 2 if usa_delta else 1e12])

    def aplica(R, t, delta):
        k = fB / (fB + delta * Zc) if usa_delta else 1.0
        return ((F - c) * np.atleast_1d(k)[:, None] if usa_delta else (F - c)) @ R.T + c + t

    for i in range(iters):
        dmax = d0 + (d1 - d0) * min(1.0, i / (iters * 0.6))
        X = aplica(R, t, delta)
        dist, j = arv.query(X, distance_upper_bound=dmax)
        ok = np.isfinite(dist)
        if ok.sum() < 200:
            break
        x, y, n, w = X[ok], alvo[j[ok]], n_alvo[j[ok]], peso[ok]
        r = ((x - y) * n).sum(1)
        w = w * np.minimum(1.0, (0.5 * dmax) / np.maximum(np.abs(r), 1e-9))   # Huber
        w = w / sig_res ** 2
        dd = (-(Zc[ok]) / (fB + delta * Zc[ok]))[:, None] * (x - c - t) if usa_delta else np.zeros_like(x)
        A = np.c_[np.cross(x - c, n), n, (dd * n).sum(1)]
        atual = np.r_[Rotation.from_matrix(R).as_rotvec(), t, delta]
        Pe = Pr * (w.sum() * sig_res ** 2)
        H = (A * w[:, None]).T @ A + Pe
        g = (A * w[:, None]).T @ (-r) - Pe @ atual
        dx = np.linalg.solve(H, g)
        R, t, delta = Rotation.from_rotvec(dx[:3]).as_matrix() @ R, t + dx[3:6], delta + dx[6]
        if np.linalg.norm(dx) < 1e-7:
            break
    X = aplica(R, t, delta)
    dist, j = arv.query(X, distance_upper_bound=d1)
    ok = np.isfinite(dist)
    res = np.abs(((X[ok] - alvo[j[ok]]) * n_alvo[j[ok]]).sum(1))
    return R, t, float(delta), res, ok.mean(), X


def residuo(F, alvo, n_alvo, arv, d=0.03):
    dist, j = arv.query(F, distance_upper_bound=d)
    ok = np.isfinite(dist)
    return np.abs(((F[ok] - alvo[j[ok]]) * n_alvo[j[ok]]).sum(1)), ok.mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robo", default="192.168.123.164")
    ap.add_argument("--segundos", type=float, default=4.0, help="quanto lidar acumular")
    ap.add_argument("--quadros", type=int, default=6, help="pares do estéreo (a média tira ruído)")
    ap.add_argument("--saida", default="~/mapas/calib_lidar_camera.json")
    ap.add_argument("--camera-medida", default="0.20,0,0,40",
                    help="03/11, medido no robô: a ZED fica DX,DY,DZ (m) à frente/esquerda/acima do LIDAR, inclinada "
                         "INCL graus para baixo (e de cabeça para baixo, o estéreo já desvira). Vazio = servo_camera.json")
    a = ap.parse_args()
    from estereo_zed import EstereoZED
    from robot.Scripts_Prometheus_int.sim.sensor_utils import ImageUtils, SensorClient
    from servo_er1 import Camera

    ctx = zmq.Context.instance()
    fk, robo = mo.FK(), mo.Robo(ctx, a.robo)
    imu = {}
    s_low = ctx.socket(zmq.SUB)
    s_low.setsockopt_string(zmq.SUBSCRIBE, "")
    s_low.connect(f"tcp://{a.robo}:6001")
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.connect(f"tcp://{a.robo}:5558")
    time.sleep(0.6)
    m = json.loads(s_low.recv().decode())
    imu = m.get("data", m)["imu_state"]
    q, vivo = robo.q29()
    if not vivo:
        sys.exit("sem juntas do robô (6001)")
    print(f"IMU da pelve: roll {np.degrees(imu['rpy'][0]):+.2f}°  pitch {np.degrees(imu['rpy'][1]):+.2f}°", flush=True)

    # ── lidar acumulado (robô parado) ──
    Ps, t0 = [], time.time()
    while time.time() - t0 < a.segundos:
        _, b = sub.recv_multipart()
        Ps.append(np.frombuffer(b, np.float32).reshape(-1, 4))
    P = np.concatenate(Ps)
    args_mapa = argparse.Namespace(vox=0.01, virada="rx", pasta="/tmp/calib", carrega=0)
    mapa = mo.Mapa(args_mapa)
    torso = mo.cadeia_pts(fk, q, "right")[1]
    R_tp, p_tp = torso
    R_lp, o = mapa.lidar_na_pelvis(torso, "rx")
    print(f"lidar: {len(P)} pontos em {a.segundos:.0f} s", flush=True)

    # ── 1. nivelamento pelo chão ──
    L = P[:, :3] @ R_lp.T + o
    chao = L[(L[:, 2] < -0.5) & (L[:, 2] > -1.1) & (np.hypot(L[:, 0], L[:, 1]) > 0.5) & (np.hypot(L[:, 0], L[:, 1]) < 3.0)]
    n_chao, c_chao, k = plano_ransac(chao)
    roll, pitch = imu["rpy"][0], imu["rpy"][1]
    R_wp = Rotation.from_euler("xy", [roll, pitch]).as_matrix()      # pelve no mundo (sem o yaw)
    cima_na_pelvis = R_wp.T @ np.array([0, 0, 1.0])
    C = gira_entre(n_chao, cima_na_pelvis)                            # corrige a orientação do lidar
    ang = np.degrees(np.arccos(np.clip(n_chao @ cima_na_pelvis, -1, 1)))
    L2 = (L - o) @ C.T + o
    altura = float(-(C @ (c_chao - o) + o) @ cima_na_pelvis)   # da pelve ao chão, na vertical
    print(f"1) chão: {k} pontos no plano; o lidar estava {ang:.2f}° torto em relação à gravidade -> corrigido. "
          f"Pelve a {abs(altura):.3f} m do chão", flush=True)
    # a correção fica no referencial do TORSO (vale para qualquer giro da cintura)
    C_torso = R_tp.T @ C @ R_tp

    # ── 2. estéreo ──
    z = EstereoZED()
    cli = SensorClient()
    cli.start_client(server_ip=a.robo, port=5555)
    Zs = []
    while len(Zs) < a.quadros:
        msg = cli.receive_message()
        ims = msg.get("images") or {}
        if "head_stereo_left" not in ims or "head_stereo_right" not in ims:
            continue
        e, d = ImageUtils.decode_image(ims["head_stereo_left"]), ImageUtils.decode_image(ims["head_stereo_right"])
        Zs.append(z.profundidade(e, d, retificada=True))
    Zs = np.stack(Zs)
    Zm = np.nanmedian(Zs, 0)
    Zm[np.nanstd(Zs, 0) > 0.02 * np.nan_to_num(Zm, nan=1) + 0.005] = np.nan     # só o que ficou firme nos quadros
    fx, cx, cy = z.P1[0, 0], z.P1[0, 2], z.P1[1, 2]
    vv, uu = np.mgrid[0:Zm.shape[0]:2, 0:Zm.shape[1]:2]
    Zg = Zm[::2, ::2]
    ok = np.isfinite(Zg) & (Zg > 0.25) & (Zg < 2.0)
    Fc = np.c_[(uu[ok] - cx) / fx * Zg[ok], (vv[ok] - cy) / fx * Zg[ok], Zg[ok]]   # na câmera retificada
    cam = Camera.carrega(str(Path.home() / "servo_camera.json"))
    # pose inicial: a do servo_camera.json (câmera original) com a rotação da retificação (R1: original -> retificada)
    R_ct0 = z.R1 @ cam.R_ct
    t_ct0 = z.R1 @ cam.t_ct
    R_tc0, o_c0 = R_ct0.T, -R_ct0.T @ t_ct0                          # câmera -> torso
    if a.camera_medida:   # a medida do robô (o servo_camera.json estava 15 cm atrás e 8 cm de lado)
        dx, dy, dz, incl = (float(v) for v in a.camera_medida.split(","))
        e = np.radians(incl)
        zc = np.array([np.cos(e), 0, -np.sin(e)])                     # eixo óptico: frente e para baixo
        xc = np.array([0, -1.0, 0])                                   # direita da imagem = -y do robô
        R_tc0 = np.c_[xc, np.cross(zc, xc), zc]
        o_c0 = mo.P_LIDAR + np.array([dx, dy, dz])
        print(f"   pose inicial pela medida: {np.round(o_c0 * 100, 1).tolist()} cm no torso, {incl:.0f}° para baixo", flush=True)
    Fp = (Fc @ R_tc0.T + o_c0) @ R_tp.T + p_tp                         # na pelve
    print(f"2) estéreo: {len(Fc)} pontos firmes em {a.quadros} quadros (0,25-2 m)", flush=True)

    # alvo: o lidar nivelado, sem a faixa de baixo do campo (os pontos falsos da "rampa") e sem o corpo
    up = P[:, :3] * np.array([1, -1, -1])
    elev = np.degrees(np.arctan2(up[:, 2], np.hypot(up[:, 0], up[:, 1])))
    A = L2[elev > -43]
    A, _ = mapa.filtra_corpo(A, q, fk, o)
    A = A[(A[:, 0] > 0.1) & (A[:, 0] < 2.5) & (np.abs(A[:, 1]) < 1.5) & (A[:, 2] > -0.9)]
    vox = np.unique(np.floor(A / 0.008).astype(np.int64), axis=0, return_index=True)[1]   # 8 mm, sem repetição
    A = A[vox]
    nA, curv, arv = normais(A)
    print(f"   alvo do lidar: {len(A)} pontos (8 mm)", flush=True)
    peso = 1.0 / np.maximum(Fc[:, 2], 0.3) ** 2                        # o estéreo erra ~Z²
    uv = np.c_[uu[ok], vv[ok]]

    fB = fx * z.B   # px·m: Z = fB / disparidade

    def corrige(Ze, delta):
        """03/11: DESLOCAMENTO DE DISPARIDADE (px): Z = fB / (fB/Ze + delta) — longe o erro cresce, perto quase nada."""
        return fB / (fB / Ze + delta)

    def escala_por_raios(R_pc, o_p, delta=0.0):
        """Compara, pixel a pixel, a profundidade do LIDAR (projetado na câmera) com a do estéreo. Devolve o
        deslocamento de disparidade que melhor casa as duas (robusto), quantos raios e o erro mediano."""
        Ac = (A - o_p) @ R_pc                                         # lidar na câmera retificada
        Ac = Ac[Ac[:, 2] > 0.2]
        u = np.round((fx * Ac[:, 0] / Ac[:, 2] + cx) / 2).astype(int)
        v = np.round((fx * Ac[:, 1] / Ac[:, 2] + cy) / 2).astype(int)
        H2, W2 = Zg.shape
        m = (u >= 0) & (u < W2) & (v >= 0) & (v < H2)
        zb = np.full((H2, W2), np.inf)
        np.minimum.at(zb, (v[m], u[m]), Ac[m, 2])                     # o lidar mais perto em cada pixel
        zl = zb[uv[:, 1] // 2, uv[:, 0] // 2]
        Ze = Fc[:, 2]
        k = np.isfinite(zl) & (np.abs(zl / Ze - 1) < 0.5)
        if k.sum() < 100:
            return delta, int(k.sum()), None
        # delta = fB/Zl - fB/Ze, por pixel; a mediana é robusta aos pixels de borda
        d = np.median(fB / zl[k] - fB / Ze[k])
        erro = np.median(np.abs(corrige(Ze[k], d) - zl[k]))
        return float(d), int(k.sum()), float(erro)

    R_pc, o_p = R_tp @ R_tc0, R_tp @ o_c0 + p_tp                        # câmera retificada na pelve
    r0, f0 = residuo(Fp, A, nA, arv)
    # 03/11: BUSCA GLOBAL (o erro é grosso: 15-20 cm longe) — inclinação x disparidade x ajuste de posição, nota =
    # distância SIMÉTRICA truncada (zed->lidar e lidar->zed) na região que os dois veem (a mesa e os objetos)
    regiao = lambda X: (X[:, 0] > 0.45) & (X[:, 0] < 0.95) & (np.abs(X[:, 1]) < 0.45) & (X[:, 2] > -0.12) & (X[:, 2] < 0.3)  # noqa: E731
    rng = np.random.default_rng(1)
    sub = rng.choice(len(Fc), min(6000, len(Fc)), replace=False)
    Al = A[regiao(A)]
    Al = Al[rng.choice(len(Al), min(6000, len(Al)), replace=False)]
    arv_l = cKDTree(Al)
    eixo_lado = R_tp @ np.array([0, 1.0, 0])                          # inclinar = girar em torno do y do torso

    def nota(incl_extra, delta, dx, dz):
        Rg = Rotation.from_rotvec(eixo_lado * np.radians(incl_extra)).as_matrix()
        F = Fc[sub] * (fB / (fB + delta * Fc[sub, 2]))[:, None]
        X = F @ (Rg @ R_pc).T + o_p + R_tp @ np.array([dx, 0, dz])
        Xr = X[regiao(X)]
        if len(Xr) < 300:
            return 9.0
        d1 = np.minimum(arv_l.query(Xr)[0], 0.05).mean()
        d2 = np.minimum(cKDTree(Xr).query(Al)[0], 0.05).mean()
        return d1 + d2

    melhor = (nota(0, 0, 0, 0), 0.0, 0.0, 0.0, 0.0)
    print(f"   busca: nota na medida {melhor[0] * 100:.2f} cm", flush=True)
    for incl in np.arange(-6, 10.1, 1.0):
        for delta in np.arange(-4, 10.1, 1.0):
            v = nota(incl, delta, 0, 0)
            if v < melhor[0]:
                melhor = (v, incl, delta, 0.0, 0.0)
    v0, incl, delta, _, _ = melhor
    for dx in np.arange(-0.04, 0.041, 0.02):          # refino fino em volta do melhor
        for dz in np.arange(-0.04, 0.041, 0.02):
            for di in (-0.5, 0, 0.5):
                for dd in (-0.5, 0, 0.5):
                    v = nota(incl + di, delta + dd, dx, dz)
                    if v < melhor[0]:
                        melhor = (v, incl + di, delta + dd, dx, dz)
    v, incl, delta0, dx, dz = melhor
    print(f"   busca: melhor nota {v * 100:.2f} cm com +{incl:.1f}° de inclinação, disparidade {delta0:+.1f} px, "
          f"desloc. x {dx * 100:+.0f} cm z {dz * 100:+.0f} cm", flush=True)
    Rg = Rotation.from_rotvec(eixo_lado * np.radians(incl)).as_matrix()
    R_pc, o_p = Rg @ R_pc, o_p + R_tp @ np.array([dx, 0, dz])
    k0 = fB / (fB + delta0 * Fc[:, 2])
    Fc = Fc * k0[:, None]                                             # a disparidade da busca fica "dentro" do Fc
    Fp = Fc @ R_pc.T + o_p
    sig_t = 0.02 if a.camera_medida else 0.06
    R, t, delta, r1, f1, Fp_d = icp_ponto_plano(Fp, A, nA, arv, peso=peso, centro=o_p, sig_t=sig_t,
                                               sig_rot=np.radians(4.0), Zc=Fc[:, 2], fB=fB)
    R_pc, o_p = R @ R_pc, o_p + t                                      # o giro é em torno da câmera
    # disparidade total: a da busca e o resto do ICP (sobre o Fc já corrigido: 1/Z soma)
    delta = delta0 + delta
    esc = delta
    r1, f1 = residuo(Fp_d, A, nA, arv)
    R_tc, o_c = R_tp.T @ R_pc, R_tp.T @ (o_p - p_tp)                   # câmera retificada no torso
    d_raios = escala_por_raios(R_pc, o_p, delta)
    print(f"   ICP: erro mediano {np.median(r0) * 1000:.1f} -> {np.median(r1) * 1000:.1f} mm (encaixados {f0:.0%} -> {f1:.0%}); "
          f"correção {np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec())):.2f}° e {np.linalg.norm(t) * 100:.1f} cm; "
          f"disparidade {delta:+.2f} px", flush=True)
    out = {"t": time.time(), "quando": time.strftime("%Y-%m-%d %H:%M"),
           "lidar_correcao_torso": C_torso.round(6).tolist(), "lidar_inclinacao_corrigida_graus": round(float(ang), 3),
           "pelve_altura_chao_m": round(abs(altura), 3),
           "camera_retificada": {"R_torso_camera": R_tc.round(6).tolist(), "o_no_torso": o_c.round(5).tolist(),
                                 "fx": float(fx), "cx": float(cx), "cy": float(cy), "w": int(Zm.shape[1]), "h": int(Zm.shape[0]),
                                 "delta_disparidade_px": round(delta, 3), "fB_px_m": round(float(fB), 4)},
           "icp": {"erro_mediano_mm_antes": round(float(np.median(r0)) * 1000, 2),
                   "erro_mediano_mm_depois": round(float(np.median(r1)) * 1000, 2),
                   "encaixe_antes": round(float(f0), 3), "encaixe_depois": round(float(f1), 3),
                   "correcao_graus": round(float(np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec()))), 3),
                   "correcao_cm": round(float(np.linalg.norm(t) * 100), 2)},
           "camera_no_torso_cm_antes": (o_c0 * 100).round(1).tolist(), "camera_no_torso_cm_depois": (o_c * 100).round(1).tolist(),
           "camera_inclinacao_graus_depois": round(float(np.degrees(np.arcsin(-R_tc[2, 2]))), 2),
           "camera_medida": a.camera_medida}
    arq = Path(a.saida).expanduser()
    arq.parent.mkdir(parents=True, exist_ok=True)
    arq.write_text(json.dumps(out, indent=1))
    # nuvens para conferir
    np.savez_compressed(arq.with_suffix(".npz"), lidar=A.astype(np.float32), zed_antes=Fp.astype(np.float32),
                        zed_depois=Fp_d.astype(np.float32))
    print(f"✅ salvo em {arq}  (câmera no torso {out['camera_no_torso_cm_antes']} -> {out['camera_no_torso_cm_depois']} cm, "
          f"inclinação {out['camera_inclinacao_graus_depois']}°)")


if __name__ == "__main__":
    main()
