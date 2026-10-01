#!/usr/bin/env python
"""Ponte G1 + Dex1 no Isaac Lab (unitree_sim_isaaclab) <-> servidor UnifoLM-WLA (PGX).

O servidor é o oficial, sem modificação:

    # na PGX
    python -m model_server.action_server_wbc_msgpack_unitree \
        --ckpt_path playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors \
        --port 8600 --unnorm_key UnifoLM_G1_Dex1

    # neste PC, com o sim rodando (sim_main.py ... --enable_dex1_dds --robot_type g129)
    python wla/ponte_wla_isaac.py --servidor ws://192.168.123.52:8600 --instrucao "pick up the red block"

O que a ponte faz a cada chunk:
  1. lê rt/lowstate (29 juntas), rt/dex1/{left,right}/state e as 3 câmeras (ZMQ, JPEG BGR);
  2. FK (URDF g1_body29_hand14) -> pose da garra no referencial da PELVIS. A convenção
     foi medida no dataset G1_Dex1_Put_Fruit_On_Plate (ajusta_ee.py): base = pelvis,
     rotação = a do *_wrist_yaw_link, ponto = wrist_yaw_link + t_garra no eixo x do punho.
     Resíduo ~1 mm e 0,0000 rad nos dois braços;
  3. manda a observação no protocolo do servidor (xyz + rot6d, garra, lower_body);
  4. recebe 30 poses absolutas (xyz + rpy 'xyz', pelvis) + garra, e executa a 30 Hz:
     IK por mínimos quadrados amortecidos em cada braço (7 juntas), publica rt/lowcmd
     (com CRC) e rt/dex1/*/cmd.

Cintura, pernas e base NÃO são comandadas: nas tarefas "-Joint" do sim só as 14 juntas
dos braços são aplicadas (action_provider_dds usa motor_cmd[15:29]).
"""
import argparse
import csv
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pinocchio as pin
import zmq
from scipy.spatial.transform import Rotation
from websockets.sync.client import connect

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_, unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

AQUI = Path(__file__).resolve().parent
URDF = AQUI.parent / "assets/g1/g1_body29_hand14.urdf"
# msgpack_numpy do próprio repositório do WLA: o mesmo empacotador do servidor.
sys.path.insert(0, str(Path.home() / "DEV/unifolm-wla/model_server"))
from tools import msgpack_numpy  # noqa: E402

N_MOTORES = 29
BRACO_E = list(range(15, 22))
BRACO_D = list(range(22, 29))
# Deslocamento da garra no referencial do wrist_yaw_link, ajustado nos dados (ajusta_ee.py).
T_GARRA = {"left": np.array([0.1081, 0.0037, 0.0]), "right": np.array([0.1097, -0.0010, 0.0])}
# Pose de partida do episódio 0 de G1_Dex1_Put_Fruit_On_Plate (xyz + rpy 'xyz', pelvis).
POSE_INICIAL = {
    "left": [0.354, 0.181, 0.185, -0.064, 0.162, -0.749],
    "right": [0.290, -0.224, 0.207, -0.061, 0.218, 0.499],
}
# Média de observation.state.lower_body nos episódios 0-2 do mesmo dataset: o robô real
# fica de pé com os joelhos dobrados; nas tarefas "-Joint" do sim as pernas ficam retas.
PERNAS_DADOS = np.array([-0.393, -0.019, -0.012, 0.676, -0.300, -0.009, -0.426, -0.049, 0.019, 0.649,
                         -0.256, 0.080, -0.014, -0.015, 0.174], np.float32)
NOMES_URDF = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


# ── cinemática ──────────────────────────────────────────────────────────────

class Cinematica:
    """FK/IK do G1 com a pelvis fixa na origem (sem free-flyer), como o referencial do WLA."""

    def __init__(self):
        self.model = pin.buildModelFromUrdf(str(URDF))
        self.data = self.model.createData()
        # índice em q de cada motor Unitree (0..28)
        self.iq = np.array([self.model.joints[self.model.getJointId(n)].idx_q for n in NOMES_URDF])
        self.lo = self.model.lowerPositionLimit[self.iq]
        self.hi = self.model.upperPositionLimit[self.iq]
        self.frames = {}
        for lado in ("left", "right"):
            f = self.model.getFrameId(f"{lado}_wrist_yaw_link")
            fr = self.model.frames[f]
            place = fr.placement * pin.SE3(np.eye(3), T_GARRA[lado])
            self.frames[lado] = self.model.addFrame(
                pin.Frame(f"{lado}_garra_wla", fr.parentJoint, f, place, pin.FrameType.OP_FRAME))
        self.data = self.model.createData()

    def _q(self, q29):
        q = pin.neutral(self.model)
        q[self.iq] = q29
        return q

    def fk(self, q29):
        pin.framesForwardKinematics(self.model, self.data, self._q(q29))
        return {lado: self.data.oMf[f].copy() for lado, f in self.frames.items()}

    def ik(self, q29, alvos, iters=40, amort=1e-3, passo=1.0, peso_rot=0.5):
        """alvos: {lado: pin.SE3}. Mexe só nas juntas do braço de cada lado."""
        q29 = q29.copy()
        for lado, alvo in alvos.items():
            idx = BRACO_E if lado == "left" else BRACO_D
            cols = self.iq[idx] if self.model.nv == self.model.nq else None
            f = self.frames[lado]
            W = np.diag([1, 1, 1, peso_rot, peso_rot, peso_rot])
            for _ in range(iters):
                q = self._q(q29)
                pin.framesForwardKinematics(self.model, self.data, q)
                atual = self.data.oMf[f]
                err = pin.log6(atual.actInv(alvo)).vector  # no referencial LOCAL da garra
                if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3:
                    break
                J = pin.computeFrameJacobian(self.model, self.data, q, f, pin.LOCAL)[:, cols]
                Jw, ew = W @ J, W @ err
                dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + amort * np.eye(6), ew)
                q29[idx] = np.clip(q29[idx] + passo * dq, self.lo[idx], self.hi[idx])
        return q29


def se3_para_ee9(M):
    R = M.rotation
    return np.concatenate([M.translation, R[:, 0], R[:, 1]]).astype(np.float32)


def xyzrpy_para_se3(v):
    return pin.SE3(Rotation.from_euler("xyz", v[3:6]).as_matrix(), np.asarray(v[:3], dtype=float))


# ── entradas do sim ─────────────────────────────────────────────────────────

class Cameras:
    """Assina as câmeras do teleimager do sim (cada mensagem ZMQ = um JPEG BGR)."""

    def __init__(self, host, portas):
        self.ctx = zmq.Context.instance()
        self.ultimo = {}
        self.t = {}
        for nome, porta in portas.items():
            threading.Thread(target=self._loop, args=(nome, host, porta), daemon=True).start()

    def _loop(self, nome, host, porta):
        s = self.ctx.socket(zmq.SUB)
        s.setsockopt(zmq.RCVHWM, 1)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt_string(zmq.SUBSCRIBE, "")
        s.connect(f"tcp://{host}:{porta}")
        while True:
            buf = s.recv()
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                self.ultimo[nome] = img
                self.t[nome] = time.time()


class EstadoRobo:
    def __init__(self):
        self.low = None
        self.garra = {"left": None, "right": None}
        ChannelSubscriber("rt/lowstate", LowState_).Init(self._low, 10)
        ChannelSubscriber("rt/dex1/left/state", MotorStates_).Init(lambda m: self._garra("left", m), 10)
        ChannelSubscriber("rt/dex1/right/state", MotorStates_).Init(lambda m: self._garra("right", m), 10)

    def _low(self, m):
        self.low = m

    def _garra(self, lado, m):
        self.garra[lado] = float(m.states[0].q)

    def q29(self):
        return np.array([self.low.motor_state[i].q for i in range(N_MOTORES)], dtype=np.float64)

    def pronto(self):
        return self.low is not None and all(v is not None for v in self.garra.values())


class Comando:
    def __init__(self, kp, kd):
        self.crc = CRC()
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.pub_garra = {l: ChannelPublisher(f"rt/dex1/{l}/cmd", MotorCmds_) for l in ("left", "right")}
        for p in self.pub_garra.values():
            p.Init()
        self.kp, self.kd = kp, kd

    def braco(self, q29, mode_machine):
        msg = unitree_hg_msg_dds__LowCmd_()
        msg.mode_pr = 0
        msg.mode_machine = mode_machine
        for i in range(N_MOTORES):
            c = msg.motor_cmd[i]
            c.mode = 1
            c.q = float(q29[i])
            c.dq = 0.0
            c.tau = 0.0
            c.kp = self.kp
            c.kd = self.kd
        msg.crc = self.crc.Crc(msg)
        self.pub.Write(msg)

    def garra(self, lado, valor):
        msg = MotorCmds_([unitree_go_msg_dds__MotorCmd_()])
        msg.cmds[0].q = float(valor)
        msg.cmds[0].kp = 5.0
        msg.cmds[0].kd = 0.05
        self.pub_garra[lado].Write(msg)


# ── laço principal ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--servidor", default="ws://192.168.123.52:8600")
    ap.add_argument("--instrucao", default="pick up the object")
    ap.add_argument("--dominio", type=int, default=1, help="domínio DDS do sim")
    ap.add_argument("--iface", default=None, help="interface de rede DDS (ex. lo)")
    ap.add_argument("--host_cam", default="127.0.0.1")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--passos", type=int, default=30, help="ações executadas por chunk (<=30)")
    ap.add_argument("--chunks", type=int, default=100)
    ap.add_argument("--max_dq", type=float, default=0.08, help="rad por passo, trava de segurança")
    ap.add_argument("--kp", type=float, default=80.0)
    ap.add_argument("--kd", type=float, default=2.0)
    ap.add_argument("--saida", default=str(AQUI / "execucoes" / time.strftime("%Y%m%d_%H%M%S")))
    ap.add_argument("--salvar_imagens", action="store_true")
    ap.add_argument("--salvar_obs", action="store_true",
                    help="grava a observação de cada chunk (obs_XXXX.npz) para o mapa de atenção")
    ap.add_argument("--pose_inicial", action="store_true",
                    help="antes da política, leva as mãos à pose de partida dos dados e abre as garras")
    ap.add_argument("--garra_inicial", type=float, default=4.5)
    ap.add_argument("--ki", type=float, default=0.0,
                    help="ganho integral por passo no espaço das juntas: compensa o braço do sim que "
                         "cede à gravidade (fica ~2 cm abaixo do comando). 0 desliga; 0.1 é um bom começo")
    ap.add_argument("--max_integral", type=float, default=0.3, help="rad, limite do termo integral")
    ap.add_argument("--pernas_dados", action="store_true",
                    help="manda lower_body = postura média dos dados em vez das pernas retas do sim")
    a = ap.parse_args()

    ChannelFactoryInitialize(a.dominio, a.iface) if a.iface else ChannelFactoryInitialize(a.dominio)
    cin = Cinematica()
    est = EstadoRobo()
    cmd = Comando(a.kp, a.kd)
    cams = Cameras(a.host_cam, {"head": 55555, "left": 55556, "right": 55557})

    print("esperando rt/lowstate, dex1 e câmeras ...", flush=True)
    t0 = time.time()
    while not (est.pronto() and len(cams.ultimo) == 3):
        if time.time() - t0 > 30:
            print(f"  lowstate={est.low is not None} garras={est.garra} cams={list(cams.ultimo)}")
            t0 = time.time()
        time.sleep(0.1)
    integral = np.zeros(N_MOTORES)

    def enviar(q_desejado):
        """Publica q_desejado nos braços (+ termo integral contra a queda por gravidade do sim)."""
        q_med = est.q29()
        q = q_desejado.copy()
        q[:15] = q_med[:15]  # cintura e pernas: o sim não aplica; manda o que ele mede
        if a.ki > 0:
            integral[15:] = np.clip(integral[15:] + a.ki * (q[15:] - q_med[15:]), -a.max_integral, a.max_integral)
        cmd.braco(q + integral, est.low.mode_machine)
        return q

    if a.pose_inicial:
        q_ini = est.q29()
        q_alvo = cin.ik(q_ini, {l: xyzrpy_para_se3(POSE_INICIAL[l]) for l in ("left", "right")}, iters=200)
        n = int(3.0 * a.hz)
        for k in range(n + int(1.0 * a.hz)):
            s = min(1.0, (k + 1) / n)
            enviar(q_ini + s * (q_alvo - q_ini))
            cmd.garra("left", a.garra_inicial)
            cmd.garra("right", a.garra_inicial)
            time.sleep(1.0 / a.hz)
        p = cin.fk(est.q29())
        print(f"pose inicial: E {p['left'].translation.round(3)} D {p['right'].translation.round(3)} | "
              f"garra medida {est.garra['left']:.2f}/{est.garra['right']:.2f} (comandada {a.garra_inicial})",
              flush=True)

    print("ok. conectando em", a.servidor, flush=True)

    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)
    log = csv.writer(open(saida / "passos.csv", "w", newline=""))
    log.writerow(["chunk", "passo", "t"] + [f"alvo_{l}_{c}" for l in ("E", "D") for c in "xyz"]
                 + [f"med_{l}_{c}" for l in ("E", "D") for c in "xyz"]
                 + ["erro_ik_E_mm", "erro_ik_D_mm", "garra_cmd_E", "garra_cmd_D", "garra_med_E", "garra_med_D"])

    packer = msgpack_numpy.Packer()
    with connect(a.servidor, max_size=None, compression=None, open_timeout=30) as ws:
        meta = msgpack_numpy.unpackb(ws.recv())
        print("servidor:", {k: meta[k] for k in ("default_unnorm_key", "action_chunk_size")}, flush=True)
        # comando contínuo entre chunks: parte do último comando, não da pose medida (que cede)
        q_cmd = q_alvo.copy() if a.pose_inicial else est.q29()
        for ch in range(a.chunks):
            q = est.q29()
            pose = cin.fk(q)
            obs = {
                "observation.images.cam_left_high": cams.ultimo["head"],
                "observation.images.cam_left_wrist": cams.ultimo["left"],
                "observation.images.cam_right_wrist": cams.ultimo["right"],
                "observation.state.left_ee_6d": se3_para_ee9(pose["left"]),
                "observation.state.right_ee_6d": se3_para_ee9(pose["right"]),
                "observation.state.left_gripper": np.array([est.garra["left"]], np.float32),
                "observation.state.right_gripper": np.array([est.garra["right"]], np.float32),
                "observation.state.lower_body": PERNAS_DADOS if a.pernas_dados else q[0:15].astype(np.float32),
                "instruction": a.instrucao,
            }
            if a.salvar_imagens:
                cv2.imwrite(str(saida / f"c{ch:04d}_head.jpg"), cams.ultimo["head"])
            if a.salvar_obs:
                np.savez_compressed(saida / f"obs_{ch:04d}.npz",
                                    **{k.replace(".", "__"): np.asarray(v) for k, v in obs.items()})
            t_inf = time.time()
            ws.send(packer.pack({"type": "get_action", "obs": obs}))
            resp = ws.recv()
            if isinstance(resp, str):
                raise RuntimeError("erro no servidor:\n" + resp)
            act = msgpack_numpy.unpackb(resp)
            t_inf = time.time() - t_inf
            alvo_e = act["action.left_ee_rpy"][0]
            alvo_d = act["action.right_ee_rpy"][0]
            g_e = act["action.left_gripper"][0, :, 0]
            g_d = act["action.right_gripper"][0, :, 0]
            print(f"chunk {ch:3d} | inferência {t_inf*1000:4.0f} ms | E {pose['left'].translation.round(3)} -> "
                  f"{alvo_e[a.passos-1, :3].round(3)} | garra {g_e[0]:.2f}/{g_d[0]:.2f}", flush=True)

            for k in range(min(a.passos, len(alvo_e))):
                t_passo = time.time()
                alvos = {"left": xyzrpy_para_se3(alvo_e[k]), "right": xyzrpy_para_se3(alvo_d[k])}
                q_ik = cin.ik(q_cmd, alvos)
                dq = np.clip(q_ik - q_cmd, -a.max_dq, a.max_dq)
                q_cmd = enviar(q_cmd + dq)
                cmd.garra("left", g_e[k])
                cmd.garra("right", g_d[k])
                fk_cmd = cin.fk(q_cmd)
                med = cin.fk(est.q29())
                erro = [np.linalg.norm(fk_cmd[l].translation - alvos[l].translation) * 1000 for l in ("left", "right")]
                log.writerow([ch, k, f"{time.time():.3f}"]
                             + list(alvo_e[k, :3].round(4)) + list(alvo_d[k, :3].round(4))
                             + list(med["left"].translation.round(4)) + list(med["right"].translation.round(4))
                             + [round(erro[0], 1), round(erro[1], 1), round(float(g_e[k]), 3), round(float(g_d[k]), 3),
                                round(est.garra["left"], 3), round(est.garra["right"], 3)])
                time.sleep(max(0.0, 1.0 / a.hz - (time.time() - t_passo)))


if __name__ == "__main__":
    main()
