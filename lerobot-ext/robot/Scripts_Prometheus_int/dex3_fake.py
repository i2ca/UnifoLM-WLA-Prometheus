#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║           MOCK G1 ROBOT — Simulador de Hardware Unitree          ║
╠══════════════════════════════════════════════════════════════════╣
║  Substitui o dex3_g1_server_v2.py para testes sem robô real.    ║
║  Sobe todos os sockets ZMQ nas mesmas portas do servidor real    ║
║  e publica estado sintético que o LeRobot consegue consumir.     ║
╠══════════════════════════════════════════════════════════════════╣
║  USO:                                                            ║
║    python mock_g1_robot.py              # simula Debug Mode      ║
║    python mock_g1_robot.py --loco       # simula Loco/WBC Mode   ║
║    python mock_g1_robot.py --loco -v    # + log de comandos      ║
╠══════════════════════════════════════════════════════════════════╣
║  PORTAS (idênticas ao servidor real):                            ║
║    6000  PULL  lowcmd      (LeRobot → mock)                      ║
║    6001  PUB   lowstate    (mock → LeRobot)  @ 500Hz             ║
║    6002  PUB   handstate   (mock → LeRobot)  @ 200Hz             ║
║    6003  PULL  handcmd     (LeRobot → mock)                      ║
║    6004  PUB   robot_mode  (mock → LeRobot)  @ 2Hz               ║
╚══════════════════════════════════════════════════════════════════╝

Dependências: apenas zmq e numpy (sem unitree_sdk2py)
    pip install pyzmq numpy
"""

import argparse
import base64
import json
import math
import signal
import threading
import time

import numpy as np
import zmq

# ─────────────────────────── Portas (igual ao servidor real) ──────────────────
LOWCMD_PORT    = 6000
LOWSTATE_PORT  = 6001
HANDSTATE_PORT = 6002
HANDCMD_PORT   = 6003
STATUS_PORT    = 6004

# ─────────────────────────── Constantes do Robô ───────────────────────────────
NUM_BODY_MOTORS = 35
NUM_HAND_MOTORS = 7

# Índices 15-28 = juntas dos braços no array de 35 motores
ARM_INDICES = list(range(15, 29))


# ──────────────────────────────── Estado Global ────────────────────────────────
class RobotState:
    def __init__(self):
        self.body_q    = np.zeros(NUM_BODY_MOTORS, dtype=np.float64)
        self.body_dq   = np.zeros(NUM_BODY_MOTORS, dtype=np.float64)
        self.body_temp = np.full(NUM_BODY_MOTORS, 35.0, dtype=np.float64)

        self.left_hand_q  = np.zeros(NUM_HAND_MOTORS, dtype=np.float64)
        self.right_hand_q = np.zeros(NUM_HAND_MOTORS, dtype=np.float64)

        self.quaternion    = [1.0, 0.0, 0.0, 0.0]
        self.gyroscope     = [0.0, 0.0, 0.0]
        self.accelerometer = [0.0, 0.0, 9.81]
        self.rpy           = [0.0, 0.0, 0.0]
        self.imu_temp      = 35.0
        self.mode_machine  = 0

        self._lock = threading.Lock()

    def apply_body_cmd(self, motor_cmds: list) -> None:
        with self._lock:
            for i, mc in enumerate(motor_cmds):
                if i >= NUM_BODY_MOTORS:
                    break
                target_q = mc.get("q", 0.0)
                kp       = mc.get("kp", 0.0)
                if kp > 0:
                    self.body_q[i] += 0.20 * (target_q - self.body_q[i])
                    self.body_dq[i] = (target_q - self.body_q[i]) * 0.001

    def apply_hand_cmd(self, side: str, motor_cmds: list) -> None:
        with self._lock:
            arr = self.left_hand_q if side == "left" else self.right_hand_q
            for i, mc in enumerate(motor_cmds):
                if i >= NUM_HAND_MOTORS:
                    break
                target_q = mc.get("q", 0.0)
                kp       = mc.get("kp", 0.0)
                if kp > 0:
                    arr[i] += 0.25 * (target_q - arr[i])

    def add_imu_noise(self, t: float) -> None:
        with self._lock:
            self.rpy[0] = 0.002 * math.sin(t * 0.3)
            self.rpy[1] = 0.002 * math.sin(t * 0.2)
            self.quaternion = [1.0, self.rpy[0] / 2, self.rpy[1] / 2, 0.0]

    def to_lowstate_dict(self) -> dict:
        with self._lock:
            motor_states = [
                {
                    "q":           float(self.body_q[i]),
                    "dq":          float(self.body_dq[i]),
                    "tau_est":     0.0,
                    "temperature": float(self.body_temp[i]),
                }
                for i in range(NUM_BODY_MOTORS)
            ]
            return {
                "motor_state": motor_states,
                "imu_state": {
                    "quaternion":    [float(x) for x in self.quaternion],
                    "gyroscope":     [float(x) for x in self.gyroscope],
                    "accelerometer": [float(x) for x in self.accelerometer],
                    "rpy":           [float(x) for x in self.rpy],
                    "temperature":   float(self.imu_temp),
                },
                # wireless_remote: 40 bytes zerados em base64
                "wireless_remote": base64.b64encode(bytes(40)).decode("ascii"),
                "mode_machine":    int(self.mode_machine),
            }

    def to_handstate_dict(self, side: str) -> dict:
        """
        Formato IDÊNTICO ao handstate_to_dict() do servidor real.
        O ChannelSubscriber.Read() do LeRobot deserializa o campo 'data'
        e acessa .motor_state[joint_id].q diretamente.
        """
        with self._lock:
            arr = self.left_hand_q if side == "left" else self.right_hand_q

            # O servidor real publica NUM_HAND_MOTORS=7 entradas, indexadas
            # pelos joint_ids do Dex3 (0-6). O Read() acessa motor_state[joint_id].q
            motor_states = [
                {"q": float(arr[i]), "dq": 0.0, "tau_est": 0.0}
                for i in range(NUM_HAND_MOTORS)
            ]

            # 3 grupos de sensores × 11 pontos = 33 valores de pressão por mão
            press_sensors = [
                {"pressure": [0.0] * 11, "temperature": [35.0] * 11}
                for _ in range(3)
            ]

            return {
                "side":               side,
                "motor_state":        motor_states,
                "press_sensor_state": press_sensors,
            }


# ──────────────────────────────── Threads ─────────────────────────────────────

def lowstate_publish_loop(sock: zmq.Socket, state: RobotState, shutdown: threading.Event):
    """Publica lowstate a ~500Hz — mesmo rate do servidor real."""
    t0 = time.time()
    while not shutdown.is_set():
        t = time.time() - t0
        state.add_imu_noise(t)
        payload = json.dumps({
            "topic": "rt/lowstate",
            "data":  state.to_lowstate_dict(),
        }).encode()
        try:
            sock.send(payload, zmq.NOBLOCK)
        except (zmq.Again, zmq.error.ContextTerminated):
            pass
        time.sleep(0.002)


def handstate_publish_loop(sock: zmq.Socket, state: RobotState, shutdown: threading.Event):
    """
    Publica handstate das duas mãos a ~200Hz no socket da porta 6002.

    IMPORTANTE: publica left e right em SEQUÊNCIA RÁPIDA no mesmo socket PUB.
    O ChannelSubscriber do SDK filtra por 'topic' — left pega 'rt/dex3/left/state'
    e right pega 'rt/dex3/right/state'. A frequência alta (5ms) garante que o
    timeout de 3s do connect() receba os dois antes de expirar.
    """
    while not shutdown.is_set():
        for side, topic in (
            ("left",  "rt/dex3/left/state"),
            ("right", "rt/dex3/right/state"),
        ):
            payload = json.dumps({
                "topic": topic,
                "data":  state.to_handstate_dict(side),
            }).encode()
            try:
                sock.send(payload, zmq.NOBLOCK)
            except (zmq.Again, zmq.error.ContextTerminated):
                pass
        time.sleep(0.005)   # 200Hz — bem mais rápido que o timeout de 3s


def status_publish_loop(sock: zmq.Socket, active_mode: str, shutdown: threading.Event):
    """Publica o modo ativo a cada 0.5s — idêntico ao servidor real."""
    payload = json.dumps({"robot_mode": active_mode}).encode()
    while not shutdown.is_set():
        try:
            sock.send(payload, zmq.NOBLOCK)
        except (zmq.Again, zmq.error.ContextTerminated):
            pass
        time.sleep(0.5)


def lowcmd_receive_loop(
    sock: zmq.Socket, state: RobotState,
    shutdown: threading.Event, verbose: bool, active_mode: str,
):
    """Recebe e valida comandos de corpo do LeRobot."""
    expected_topic = "rt/arm_sdk" if active_mode == "loco" else "rt/lowcmd"
    cmd_count = 0
    last_log  = time.time()

    while not shutdown.is_set():
        try:
            payload = sock.recv(zmq.NOBLOCK)
        except zmq.Again:
            time.sleep(0.001)
            continue
        except zmq.ContextTerminated:
            break

        try:
            msg = json.loads(payload.decode())
        except Exception:
            continue

        topic     = msg.get("topic", "")
        motor_cmd = msg.get("data", {}).get("motor_cmd", [])
        cmd_count += 1

        if topic and topic != expected_topic:
            print(
                f"\n[MOCK ⚠️ ] Tópico inesperado: '{topic}' "
                f"(esperado '{expected_topic}' para modo '{active_mode}')"
            )

        state.apply_body_cmd(motor_cmd)

        now = time.time()
        if verbose or (now - last_log >= 2.0):
            arm_qs = [float(motor_cmd[i].get("q", 0.0)) for i in ARM_INDICES if i < len(motor_cmd)]
            arm_str = " ".join(f"{v:+.3f}" for v in arm_qs[:7])
            print(
                f"[MOCK 📨 ] cmd #{cmd_count:>5} | tópico={topic!r:20s} "
                f"| braço_esq_q: [{arm_str}]"
            )
            last_log = now


def handcmd_receive_loop(
    sock: zmq.Socket, state: RobotState,
    shutdown: threading.Event, verbose: bool,
):
    """Recebe comandos das mãos, aplica ao estado e confirma no log."""
    hand_cmd_count = {"left": 0, "right": 0}

    while not shutdown.is_set():
        try:
            payload = sock.recv(zmq.NOBLOCK)
        except zmq.Again:
            time.sleep(0.001)
            continue
        except zmq.ContextTerminated:
            break

        try:
            msg = json.loads(payload.decode())
        except Exception:
            continue

        topic     = msg.get("topic", "")
        motor_cmd = msg.get("data", {}).get("motor_cmd", [])
        side      = "left" if "left" in topic else "right"

        state.apply_hand_cmd(side, motor_cmd)
        hand_cmd_count[side] += 1

        if verbose:
            qs = [motor_cmd[i].get("q", 0.0) for i in range(min(7, len(motor_cmd)))]
            qs_str = " ".join(f"{v:+.3f}" for v in qs)
            print(f"[MOCK 🖐  ] mão={side:5s} #{hand_cmd_count[side]:>4} | dedos_q: [{qs_str}]")
        elif hand_cmd_count[side] % 100 == 0:
            # Log a cada 100 comandos de mão mesmo sem --verbose
            with state._lock:
                arr = state.left_hand_q if side == "left" else state.right_hand_q
                qs_str = " ".join(f"{v:+.3f}" for v in arr)
            print(f"[MOCK 🖐  ] mão={side:5s} #{hand_cmd_count[side]:>4} | q atual: [{qs_str}]")


def stats_monitor_loop(state: RobotState, active_mode: str, shutdown: threading.Event):
    """Painel de status a cada 5 segundos."""
    t0 = time.time()
    while not shutdown.is_set():
        time.sleep(5.0)
        if shutdown.is_set():
            break
        elapsed = time.time() - t0
        with state._lock:
            arm_q = state.body_q[ARM_INDICES].copy()
            lh_q  = state.left_hand_q.copy()
            rh_q  = state.right_hand_q.copy()
            rpy   = list(state.rpy)

        print("\n" + "─" * 62)
        print(f"  ⏱  Tempo rodando : {elapsed:>6.1f}s   Modo: {active_mode.upper()}")
        print(f"  🦾  Braço esq (q) : {' '.join(f'{v:+.3f}' for v in arm_q[:7])}")
        print(f"  🦾  Braço dir (q) : {' '.join(f'{v:+.3f}' for v in arm_q[7:])}")
        print(f"  🖐   Mão esq  (q) : {' '.join(f'{v:+.3f}' for v in lh_q)}")
        print(f"  🖐   Mão dir  (q) : {' '.join(f'{v:+.3f}' for v in rh_q)}")
        print(f"  🧭  IMU rpy       : r={rpy[0]:+.4f}  p={rpy[1]:+.4f}  y={rpy[2]:+.4f}")
        print("─" * 62)


# ──────────────────────────────── Main ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Mock G1 Robot — simula o hardware Unitree via ZMQ"
    )
    parser.add_argument("--loco", action="store_true", default=False,
                        help="Simula Loco/WBC Mode (rt/arm_sdk). Sem flag = Debug (rt/lowcmd).")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="Imprime cada comando recebido.")
    args = parser.parse_args()

    active_mode = "loco" if args.loco else "debug"

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              MOCK G1 ROBOT — Simulador de Hardware           ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Modo    : {'HIGH LEVEL (Loco/WBC) 🏃' if args.loco else 'LOW LEVEL (Debug)  🛑':^38s}║")
    print(f"║  Verbose : {'SIM' if args.verbose else 'NÃO':^38s}║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Portas ZMQ:                                                 ║")
    print("║    6000 PULL  lowcmd     (LeRobot → mock)                    ║")
    print("║    6001 PUB   lowstate   (mock → LeRobot)  @ 500Hz           ║")
    print("║    6002 PUB   handstate  (mock → LeRobot)  @ 200Hz           ║")
    print("║    6003 PULL  handcmd    (LeRobot → mock)                    ║")
    print("║    6004 PUB   status     (mock → LeRobot)  @ 2Hz             ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("  Aguardando LeRobot conectar... (Ctrl+C para sair)")
    print()

    state = RobotState()
    state.mode_machine = 5 if args.loco else 0

    ctx = zmq.Context.instance()

    lowcmd_sock    = ctx.socket(zmq.PULL)
    lowstate_sock  = ctx.socket(zmq.PUB)
    handstate_sock = ctx.socket(zmq.PUB)
    handcmd_sock   = ctx.socket(zmq.PULL)
    status_sock    = ctx.socket(zmq.PUB)

    # Buffer generoso para não perder mensagens no burst inicial do connect()
    for s in (lowstate_sock, handstate_sock, status_sock):
        s.setsockopt(zmq.SNDHWM, 100)

    lowcmd_sock.bind(f"tcp://0.0.0.0:{LOWCMD_PORT}")
    lowstate_sock.bind(f"tcp://0.0.0.0:{LOWSTATE_PORT}")
    handstate_sock.bind(f"tcp://0.0.0.0:{HANDSTATE_PORT}")
    handcmd_sock.bind(f"tcp://0.0.0.0:{HANDCMD_PORT}")
    status_sock.bind(f"tcp://0.0.0.0:{STATUS_PORT}")

    print(f"[MOCK] ✅ Todos os sockets ZMQ prontos.")
    print(f"[MOCK] 📡 Publicando modo '{active_mode}' na porta {STATUS_PORT}...")
    print()

    shutdown = threading.Event()

    threads = [
        threading.Thread(target=lowstate_publish_loop,
                         args=(lowstate_sock, state, shutdown),
                         name="lowstate_pub", daemon=True),
        threading.Thread(target=handstate_publish_loop,
                         args=(handstate_sock, state, shutdown),
                         name="handstate_pub", daemon=True),
        threading.Thread(target=status_publish_loop,
                         args=(status_sock, active_mode, shutdown),
                         name="status_pub", daemon=True),
        threading.Thread(target=lowcmd_receive_loop,
                         args=(lowcmd_sock, state, shutdown, args.verbose, active_mode),
                         name="lowcmd_recv", daemon=True),
        threading.Thread(target=handcmd_receive_loop,
                         args=(handcmd_sock, state, shutdown, args.verbose),
                         name="handcmd_recv", daemon=True),
        threading.Thread(target=stats_monitor_loop,
                         args=(state, active_mode, shutdown),
                         name="stats_monitor", daemon=True),
    ]

    for t in threads:
        t.start()

    def _stop(sig, frame):
        print("\n\n[MOCK] 🛑 Encerrando mock robot...")
        shutdown.set()

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        while not shutdown.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown.set()

    for t in threads:
        t.join(timeout=2.0)
    ctx.term()
    print("[MOCK] ✅ Finalizado.")


if __name__ == "__main__":
    main()