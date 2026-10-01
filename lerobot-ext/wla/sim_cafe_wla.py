#!/usr/bin/env python
"""
G1 + Dex3 na CENA DO CAFÉ (MuJoCo) com uma MAÇÃ e o X PRETO — fingindo ser o robô real
=========================================================================================
Publica EXATAMENTE as mesmas portas ZMQ que o robô real publica (ponte v3 + cameras_wla_server),
para o `roda_wla_real.py` rodar SEM MUDANÇA contra o simulador, com o modelo na Athena:

    5555  PUB   câmeras  {"images": {head_stereo_left/right, wrist_left/right: JPEG base64 (BGR)}}
    6000  PULL  lowcmd   (rt/arm_sdk: braços + cintura, q/kp/kd — com os limites da ponte v3:
                          alvo anda no máx. --vel-braco rad/s, kp cortado em --kp-braco / --kp-punho)
    6001  PUB   lowstate (35 motores, q/dq/tau_est/temperature + imu)
    6002  PUB   estado das mãos Dex3 (ordem do SDK, "side")
    6003  PULL  comando das mãos Dex3 (ordem do SDK; alvo anda no máx. --vel-mao rad/s)
    6004  PUB   {"robot_mode": "loco"}
    6006  PULL  {"panico": true} | {"rearmar": true}      6007  PUB   estado do pânico (gpio "ok")

A cena e as câmeras estão em XML, para editar vendo ao vivo no VS Code (extensão MuJoCo Viewer):
  unitree-g1-mujoco/assets/scene_cafe_wla.xml         a cena do café com a mesa BAIXADA para a altura do
        nosso dataset (tampo 11 cm abaixo da pelvis), cor de papelão, a MAÇÃ (7,6 cm, 150 g, dá para pegar)
        e o X preto; keyframe "maca" = a pose de partida do dataset
  unitree-g1-mujoco/assets/g1_29dof_with_hand_wla.xml  o G1 com a pelvis PRESA (guindaste), a ZED da cabeça e
        uma D435 em cada punho — procure "EDITE AQUI" (pos = onde fica, zaxis = para onde a lente aponta)
A ZED sai como o cameras_wla_server --zed-enquadramento largo (olho 672x376 reduzido a 640 + faixas pretas).

    # notebook (env prometheus-vla)
    python sim_cafe_wla.py                          # abre a janela do MuJoCo e publica as portas
    python sim_cafe_wla.py --foto /tmp/sim.jpg      # só salva as 4 câmeras na pose inicial e sai
    # outro terminal: túnel para o servidor na Athena e o executor, como no robô real
    ssh -N -L 8601:localhost:8601 <usuario>@<athena>
    python roda_wla_real.py --robo 127.0.0.1 --servidor ws://127.0.0.1:8601 --pose maca \\
        --tarefa "Pick up the apple and place it on the black X." --voz nenhuma
"""
import argparse
import base64
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import zmq

AQUI = Path(__file__).resolve().parent
CENA = AQUI.parents[1] / "unitree-g1-mujoco/assets/scene_cafe_wla.xml"
from fk_g1 import NOMES  # noqa: E402

POSE = json.load(open(AQUI / "pose_partida_dex1.json"))
Q_MACA = np.array(POSE["pernas_maca"] + POSE["cintura_maca"] + POSE["left_maca"] + POSE["right_maca"])

# Dex3 na ordem do SDK -> nome da junta no MJCF
MAO_SDK = {"left": ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"],
           "right": ["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]}
PUNHOS = [19, 20, 21, 26, 27, 28]
N_MOTORES = 35
W, H = 640, 480
PELVIS_Z = 0.793   # a pelvis fica presa aqui (g1_29dof_with_hand_wla.xml)


def largo_4x3(img):
    """Igual ao cameras_wla_server.largo_4x3: olho inteiro reduzido a 640 de largura + faixas pretas."""
    ih, iw = img.shape[:2]
    nh = int(round(ih * W / iw))
    img = cv2.resize(img, (W, nh), interpolation=cv2.INTER_AREA)
    topo = (H - nh) // 2
    return cv2.copyMakeBorder(img, topo, H - nh - topo, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def monta_cena(a):
    """Carrega scene_cafe_wla.xml (mesa, maçã, X, ZED e D435 dos punhos ficam TODOS no XML — edite lá, de
    preferência com a extensão MuJoCo Viewer do VS Code) e só põe a maçã e o X onde os argumentos mandam."""
    spec = mujoco.MjSpec.from_file(str(CENA))
    tampo = spec.body("x_preto").pos[2] - 0.0006
    spec.body("maca").pos = [a.maca[0], a.maca[1], tampo + 0.036]
    spec.body("x_preto").pos = [a.x[0], a.x[1], tampo + 0.0006]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    return spec.compile()


class Robo:
    """Estado + controle do G1 simulado, com as regras da ponte v3 (pânico, limites de velocidade e kp)."""

    def __init__(self, m, d, a):
        self.m, self.d, self.a = m, d, a
        self.lock = threading.Lock()
        self.qadr = np.array([m.joint(n).qposadr[0] for n in NOMES])
        self.vadr = np.array([m.joint(n).dofadr[0] for n in NOMES])
        self.act = np.array([m.actuator(n.replace("_joint", "")).id for n in NOMES])
        self.mao = {l: {k: [m.joint(f"{l}_hand_{n}_joint") for n in MAO_SDK[l]] for k in ("j",)} for l in MAO_SDK}
        self.mao_q = {l: np.array([j.qposadr[0] for j in self.mao[l]["j"]]) for l in MAO_SDK}
        self.mao_v = {l: np.array([j.dofadr[0] for j in self.mao[l]["j"]]) for l in MAO_SDK}
        self.mao_act = {l: np.array([m.actuator(f"{l}_hand_{n}").id for n in MAO_SDK[l]]) for l in MAO_SDK}
        d.qpos[self.qadr] = Q_MACA
        mujoco.mj_forward(m, d)
        # alvo (29), kp, kd — começa segurando a pose de partida do dataset
        self.alvo = Q_MACA.copy()
        self.alvo_cmd = Q_MACA.copy()
        self.kp = np.full(29, 60.0)
        self.kd = np.full(29, 2.0)
        self.kp[12:15], self.kd[12:15] = 150.0, 5.0
        self.mao_alvo = {l: np.zeros(7) for l in MAO_SDK}
        self.mao_alvo_cmd = {l: np.zeros(7) for l in MAO_SDK}
        self.panico, self.motivo = False, ""
        self.n_cmd = 0

    # ── comandos (threads ZMQ) ──
    def lowcmd(self, msg):
        mc = msg.get("data", msg).get("motor_cmd", [])
        with self.lock:
            if self.panico:
                return
            for i in range(12, 29):
                c = mc[i] if i < len(mc) else None
                if c and c.get("mode", 0) == 1:
                    self.alvo_cmd[i] = float(c["q"])
                    teto = 1e9 if i < 15 else (self.a.kp_punho if i in PUNHOS else self.a.kp_braco)
                    self.kp[i], self.kd[i] = min(float(c["kp"]), teto), float(c["kd"])
            self.n_cmd += 1

    def handcmd(self, msg):
        lado = "left" if "left" in msg.get("topic", "") else "right"
        mc = msg["data"]["motor_cmd"]
        with self.lock:
            if not self.panico:
                self.mao_alvo_cmd[lado] = np.array([float(c["q"]) for c in mc[:7]])

    def rearme(self, msg):
        with self.lock:
            if msg.get("panico"):
                self.panico, self.motivo = True, "software"
                self.alvo_cmd = self.d.qpos[self.qadr].copy()
                self.alvo = self.alvo_cmd.copy()
                for l in MAO_SDK:
                    self.mao_alvo_cmd[l] = self.d.qpos[self.mao_q[l]].copy()
                print("🛑 [sim] pânico: tudo congelado", flush=True)
            elif msg.get("rearmar"):
                self.panico, self.motivo = False, ""
                print("🔓 [sim] rearmado", flush=True)

    # ── física ──
    def passo(self):
        m, d, a = self.m, self.d, self.a
        dt = m.opt.timestep
        with self.lock:
            lim = a.vel_braco * dt
            self.alvo[12:] += np.clip(self.alvo_cmd[12:] - self.alvo[12:], -lim, lim)
            self.alvo[:12] = Q_MACA[:12]
            for l in MAO_SDK:
                self.mao_alvo[l] += np.clip(self.mao_alvo_cmd[l] - self.mao_alvo[l], -a.vel_mao * dt, a.vel_mao * dt)
            q, v = d.qpos[self.qadr], d.qvel[self.vadr]
            kp = self.kp.copy()
            kp[:12] = 300.0
            tau = kp * (self.alvo - q) - self.kd * v
            if a.comp_gravidade:
                tau[12:] += a.comp_gravidade * d.qfrc_bias[self.vadr[12:]]
            d.ctrl[self.act] = tau
            for l in MAO_SDK:
                d.ctrl[self.mao_act[l]] = a.kp_mao * (self.mao_alvo[l] - d.qpos[self.mao_q[l]]) - 0.05 * d.qvel[self.mao_v[l]]
        mujoco.mj_step(m, d)

    # ── mensagens de estado ──
    def lowstate(self):
        d = self.d
        q, v = d.qpos[self.qadr], d.qvel[self.vadr]
        tau = d.actuator_force[self.act]
        ms = [{"q": float(q[i]), "dq": float(v[i]), "tau_est": float(tau[i]), "temperature": [35, 35], "mode": 1}
              for i in range(29)] + [{"q": 0.0, "dq": 0.0, "tau_est": 0.0, "temperature": [0, 0], "mode": 0}] * 6
        return {"topic": "rt/lowstate", "data": {"mode_pr": 0, "mode_machine": 5, "motor_state": ms, "imu_state": {
            "quaternion": [1.0, 0, 0, 0], "gyroscope": [0.0] * 3, "accelerometer": [0, 0, 9.81], "rpy": [0.0] * 3,
            "temperature": 35.0}}}

    def maostate(self, lado):
        q, v = self.d.qpos[self.mao_q[lado]], self.d.qvel[self.mao_v[lado]]
        return {"topic": f"rt/dex3/{lado}/state", "data": {"side": lado, "motor_state": [
            {"q": float(q[k]), "dq": float(v[k]), "tau_est": 0.0} for k in range(7)], "press_sensor_state": []}}


def servidor_zmq(robo, parar):
    ctx = zmq.Context.instance()

    def pub(porta):
        s = ctx.socket(zmq.PUB)
        s.setsockopt(zmq.SNDHWM, 5)
        s.bind(f"tcp://0.0.0.0:{porta}")
        return s

    low, maos, modo, panico = pub(6001), pub(6002), pub(6004), pub(6007)
    puxa = {}
    for porta, fn in ((6000, robo.lowcmd), (6003, robo.handcmd), (6006, robo.rearme)):
        s = ctx.socket(zmq.PULL)
        s.bind(f"tcp://0.0.0.0:{porta}")
        puxa[s] = fn
    poller = zmq.Poller()
    for s in puxa:
        poller.register(s, zmq.POLLIN)

    def recebe():
        while not parar.is_set():
            for s, _ in poller.poll(50):
                try:
                    puxa[s](json.loads(s.recv(zmq.NOBLOCK).decode("utf-8")))
                except Exception as e:  # noqa: BLE001
                    print(f"[sim] comando inválido: {e}", flush=True)

    threading.Thread(target=recebe, daemon=True).start()
    t_modo = 0.0
    while not parar.is_set():
        with robo.lock:
            ls = robo.lowstate()
            ms = [robo.maostate(l) for l in ("left", "right")]
            pan = {"panico": robo.panico, "motivo": robo.motivo, "gpio": "ok", "botao_ok": True,
                   "botao_solto_s": 0.0, "descartados": 0, "vel_braco": robo.a.vel_braco,
                   "kp_braco": robo.a.kp_braco, "kp_punho": robo.a.kp_punho, "t": time.time(), "sim": True}
        low.send(json.dumps(ls).encode("utf-8"))
        for x in ms:
            maos.send(json.dumps(x).encode("utf-8"))
        panico.send(json.dumps(pan).encode("utf-8"))
        if time.time() - t_modo > 0.5:
            modo.send(json.dumps({"robot_mode": "loco"}).encode("utf-8"))
            t_modo = time.time()
        time.sleep(0.01)


class Cameras:
    CAMS = {"head_stereo_left": (672, 376), "head_stereo_right": (672, 376),
            "wrist_left": (W, H), "wrist_right": (W, H)}

    def __init__(self, m):
        self.r = {k: mujoco.Renderer(m, height=h, width=w) for k, (w, h) in
                  {"zed": (672, 376), "d435": (W, H)}.items()}

    def renderiza(self, d):
        out = {}
        for nome in self.CAMS:
            r = self.r["zed" if nome.startswith("head") else "d435"]
            r.update_scene(d, camera=nome)
            rgb = r.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            out[nome] = largo_4x3(bgr) if nome.startswith("head") else bgr
        return out


def publica_cameras(fila, parar):
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.PUB)
    s.setsockopt(zmq.SNDHWM, 2)
    s.bind("tcp://0.0.0.0:5555")
    while not parar.is_set():
        try:
            imgs, t = fila.get(timeout=0.5)
        except queue.Empty:
            continue
        env = {k: base64.b64encode(cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode("ascii")
               for k, v in imgs.items()}
        s.send_multipart([json.dumps({"images": env, "timestamps": {k: t for k in env},
                                      "servidor": {"versao": "sim_cafe_wla", "config": "MuJoCo"}}).encode("utf-8")])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maca", type=float, nargs=2, default=[0.40, -0.10], help="x y da maçã (m, frame da pelvis)")
    ap.add_argument("--x", type=float, nargs=2, default=[0.33, 0.10], help="x y do X preto")
    ap.add_argument("--aleatorio", type=float, default=0.0, help="sorteia maçã e X ± este tanto (m)")
    ap.add_argument("--vel-braco", type=float, default=0.3, help="como a ponte v3 do robô")
    ap.add_argument("--kp-braco", type=float, default=40.0, help="como o init (KP_BRACO)")
    ap.add_argument("--kp-punho", type=float, default=20.0, help="como o init (KP_PUNHO)")
    ap.add_argument("--vel-mao", type=float, default=5.0, help="como o init (VEL_MAO)")
    ap.add_argument("--kp-mao", type=float, default=2.0, help="kp dos dedos no simulador (N·m/rad)")
    ap.add_argument("--comp-gravidade", type=float, default=0.0,
                    help="fração da gravidade compensada nos braços (0 = só PD, cede como o real)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--foto", default=None, help="salva as 4 câmeras (e uma vista de fora) e sai")
    ap.add_argument("--sem-janela", action="store_true")
    a = ap.parse_args()
    if a.aleatorio:
        rng = np.random.default_rng()
        a.maca = list(np.array(a.maca) + rng.uniform(-a.aleatorio, a.aleatorio, 2))
        a.x = list(np.array(a.x) + rng.uniform(-a.aleatorio, a.aleatorio, 2))

    m = monta_cena(a)
    d = mujoco.MjData(m)
    robo = Robo(m, d, a)
    for _ in range(int(1.0 / m.opt.timestep)):     # 1 s para a maçã assentar e os braços pararem
        robo.passo()
    cams = Cameras(m)

    if a.foto:
        imgs = cams.renderiza(d)
        r = mujoco.Renderer(m, height=H, width=W)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.35, 0, PELVIS_Z]
        cam.distance, cam.azimuth, cam.elevation = 1.6, 200, -25
        r.update_scene(d, camera=cam)
        fora = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
        grade = np.vstack([np.hstack([imgs["head_stereo_left"], imgs["wrist_left"], imgs["wrist_right"]]),
                           np.hstack([fora, imgs["head_stereo_right"], np.zeros_like(fora)])])
        cv2.imwrite(a.foto, grade)
        print("salvo", a.foto)
        return

    parar = threading.Event()
    threading.Thread(target=servidor_zmq, args=(robo, parar), daemon=True).start()
    fila = queue.Queue(maxsize=1)
    threading.Thread(target=publica_cameras, args=(fila, parar), daemon=True).start()
    print(f"🍎 sim do café: maçã em {np.round(a.maca, 2)}, X em {np.round(a.x, 2)} | portas 5555, 6000-6007 "
          f"em 127.0.0.1 | ponte: {a.vel_braco} rad/s, kp {a.kp_braco}/{a.kp_punho}", flush=True)

    viewer = None
    if not a.sem_janela:
        viewer = mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False)
    t_sim0, t_real0, t_cam, t_log = d.time, time.time(), 0.0, time.time()
    try:
        while viewer is None or viewer.is_running():
            alvo_t = t_sim0 + (time.time() - t_real0)
            while d.time < alvo_t:
                robo.passo()
            if d.time - t_cam >= 1.0 / a.fps:
                t_cam = d.time
                imgs = cams.renderiza(d)
                try:
                    fila.put_nowait((imgs, time.time()))
                except queue.Full:
                    pass
            if viewer is not None:
                viewer.sync()
            if time.time() - t_log > 5:
                t_log = time.time()
                pm = d.body("maca").xpos
                print(f"[sim] t={d.time:6.1f}s | comandos {robo.n_cmd} | pânico {robo.panico} | maçã "
                      f"({pm[0]:.2f}, {pm[1]:.2f}, {pm[2] - PELVIS_Z:+.2f}) m", flush=True)
            time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    finally:
        parar.set()
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
