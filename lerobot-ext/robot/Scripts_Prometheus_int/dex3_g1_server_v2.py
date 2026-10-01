#!/usr/bin/env python3

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DDS-to-ZMQ bridge server for Unitree G1 robot with Dex3 hands.

Startup:
    python dex3_g1_server_v2.py            # High Level / Loco (rt/arm_sdk) — PADRÃO
    python dex3_g1_server_v2.py --debug    # Low Level (rt/lowcmd) — robô SUSPENSO

O padrão é LOCO desde 22/09/2026. Antes era debug, e o loco dependia de alguém
lembrar do `--loco` — que vinha do `USE_LOCO` do `init_prometheus-vla.sh`. Essa
variável foi trocada para `false` no robô em 28/08 sem ninguém perceber, e o
servidor passou a matar a IA na subida. Com o robô em pé, isso é o robô no chão.
O modo perigoso agora é o que exige ser pedido.

`--loco` continua aceito (não faz nada) para não quebrar quem ainda o passa.

O modo é decidido UMA VEZ no startup.
O servidor NÃO troca de modo durante a operação — isso evita quedas acidentais.

Portas ZMQ:
    6000  PULL  lowcmd  (PC → robô)
    6001  PUB   lowstate (robô → PC)
    6002  PUB   handstate (robô → PC)
    6003  PULL  handcmd (PC → robô)
    6004  PUB   robot_mode status (robô → PC)  ← NOVO
"""

import argparse
import base64
import json
import threading
import time
from typing import Any

import zmq
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient


class MotionSwitcher:
    def __init__(self):
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

    def current_mode_name(self) -> str:
        """Retorna o nome do modo atual ('ai', '' para debug, etc.)."""
        try:
            _, result = self.msc.CheckMode()
            return result.get("name", "")
        except Exception:
            return ""

    def enter_debug_mode(self) -> bool:
        """Mata a IA e entra em Low Level. Retorna True se bem-sucedido."""
        try:
            print("[MotionSwitcher] Matando a IA e entrando em Debug Mode (Low Level)...")
            while self.current_mode_name():
                self.msc.ReleaseMode()
                time.sleep(0.5)
            print("[MotionSwitcher] ✅ Debug Mode ativo.")
            return True
        except Exception as e:
            print(f"[MotionSwitcher] ❌ Falha ao entrar em Debug Mode: {e}")
            return False

    def enter_loco_mode(self) -> bool:
        """Ativa a IA (WBC/Loco). Retorna True se bem-sucedido."""
        try:
            print("[MotionSwitcher] Ativando IA (Loco/WBC)...")
            self.msc.SelectMode(nameOrAlias="ai")
            time.sleep(1.0)  # Aguarda a IA estabilizar
            name = self.current_mode_name()
            if name:
                print(f"[MotionSwitcher] ✅ Loco Mode ativo (modo='{name}').")
                return True
            else:
                print("[MotionSwitcher] ⚠️  SelectMode retornou mas CheckMode ainda vazio.")
                return False
        except Exception as e:
            print(f"[MotionSwitcher] ❌ Falha ao entrar em Loco Mode: {e}")
            return False


kTopicLowCommand_Debug  = "rt/lowcmd"
kTopicLowCommand_Motion = "rt/arm_sdk"
kTopicLowState          = "rt/lowstate"
kTopicDex3LeftCommand   = "rt/dex3/left/cmd"
kTopicDex3RightCommand  = "rt/dex3/right/cmd"
kTopicDex3LeftState     = "rt/dex3/left/state"
kTopicDex3RightState    = "rt/dex3/right/state"

LOWCMD_PORT    = 6000
LOWSTATE_PORT  = 6001
HANDSTATE_PORT = 6002
HANDCMD_PORT   = 6003
STATUS_PORT    = 6004   # ← Novo: publica o modo atual para o LeRobot
LOCOCMD_PORT   = 6005   # ← Novo: velocidade de locomoção (PC → robô), só em --loco

# Se o PC parar de mandar velocidade por mais que isto, o robô para sozinho.
# É a rede caindo, o teleop travando ou o operador tirando o óculos: em todos
# esses casos um G1 andando sem ninguém no comando é o pior desfecho possível.
LOCO_WATCHDOG_S = 0.4

NUM_MOTORS      = 35
NUM_HAND_MOTORS = 7

def lowstate_to_dict(msg: hg_LowState) -> dict[str, Any]:
    motor_states = []
    for i in range(NUM_MOTORS):
        temp = msg.motor_state[i].temperature
        avg_temp = float(sum(temp) / len(temp)) if isinstance(temp, list) else float(temp)
        motor_states.append({
            "q": float(msg.motor_state[i].q),
            "dq": float(msg.motor_state[i].dq),
            "tau_est": float(msg.motor_state[i].tau_est),
            "temperature": avg_temp,
        })
    return {
        "motor_state": motor_states,
        "imu_state": {
            "quaternion": [float(x) for x in msg.imu_state.quaternion],
            "gyroscope": [float(x) for x in msg.imu_state.gyroscope],
            "accelerometer": [float(x) for x in msg.imu_state.accelerometer],
            "rpy": [float(x) for x in msg.imu_state.rpy],
            "temperature": float(msg.imu_state.temperature),
        },
        "wireless_remote": base64.b64encode(bytes(msg.wireless_remote)).decode("ascii"),
        "mode_machine": int(msg.mode_machine),
    }

def handstate_to_dict(msg: HandState_, side: str) -> dict[str, Any]:
    motor_states = []
    for i in range(NUM_HAND_MOTORS):
        motor_states.append({
            "q": float(msg.motor_state[i].q),
            "dq": float(msg.motor_state[i].dq),
            "tau_est": float(msg.motor_state[i].tau_est),
        })

    press_sensors = []
    if hasattr(msg, 'press_sensor_state'):
        for p in msg.press_sensor_state:
            press_sensors.append({
                "pressure": list(p.pressure),
                "temperature": list(p.temperature)
            })

    return {
        "side": side,
        "motor_state": motor_states,
        "press_sensor_state": press_sensors,
    }


def dict_to_lowcmd(data: dict[str, Any]) -> hg_LowCmd:
    cmd = unitree_hg_msg_dds__LowCmd_()
    cmd.mode_pr = data.get("mode_pr", 0)
    cmd.mode_machine = data.get("mode_machine", 0)

    # Conversão Pura: Sem hacks de perna, sem mode_pr forçado.
    for i, motor_data in enumerate(data.get("motor_cmd", [])):
        cmd.motor_cmd[i].mode = motor_data.get("mode", 0)
        cmd.motor_cmd[i].q = motor_data.get("q", 0.0)
        cmd.motor_cmd[i].dq = motor_data.get("dq", 0.0)
        cmd.motor_cmd[i].kp = motor_data.get("kp", 0.0)
        cmd.motor_cmd[i].kd = motor_data.get("kd", 0.0)
        cmd.motor_cmd[i].tau = motor_data.get("tau", 0.0)

    return cmd


def dict_to_handcmd(data: dict[str, Any]) -> HandCmd_:
    cmd = unitree_hg_msg_dds__HandCmd_()
    for i, motor_data in enumerate(data.get("motor_cmd", [])):
        cmd.motor_cmd[i].mode = motor_data.get("mode", 0)
        cmd.motor_cmd[i].q = motor_data.get("q", 0.0)
        cmd.motor_cmd[i].dq = motor_data.get("dq", 0.0)
        cmd.motor_cmd[i].kp = motor_data.get("kp", 0.0)
        cmd.motor_cmd[i].kd = motor_data.get("kd", 0.0)
        cmd.motor_cmd[i].tau = motor_data.get("tau", 0.0)
    return cmd


def state_forward_loop(lowstate_sub, lowstate_sock, state_period, shutdown_event):
    last_state_time = 0.0
    while not shutdown_event.is_set():
        msg = lowstate_sub.Read()
        if msg is None: continue
        now = time.time()
        if now - last_state_time >= state_period:
            state_dict = lowstate_to_dict(msg)
            payload = json.dumps({"topic": kTopicLowState, "data": state_dict}).encode("utf-8")
            try: lowstate_sock.send(payload, zmq.NOBLOCK)
            except (zmq.Again, zmq.error.ContextTerminated): pass
            last_state_time = now

def handstate_forward_loop(left_sub, right_sub, handstate_sock, state_period, shutdown_event):
    last_left_time = 0.0
    last_right_time = 0.0
    while not shutdown_event.is_set():
        now = time.time()
        msg_left = left_sub.Read()
        if msg_left is not None and (now - last_left_time >= state_period):
            state_dict = handstate_to_dict(msg_left, "left")
            payload = json.dumps({"topic": kTopicDex3LeftState, "data": state_dict}).encode("utf-8")
            try: handstate_sock.send(payload, zmq.NOBLOCK)
            except (zmq.Again, zmq.error.ContextTerminated): pass
            last_left_time = now
        
        msg_right = right_sub.Read()
        if msg_right is not None and (now - last_right_time >= state_period):
            state_dict = handstate_to_dict(msg_right, "right")
            payload = json.dumps({"topic": kTopicDex3RightState, "data": state_dict}).encode("utf-8")
            try: handstate_sock.send(payload, zmq.NOBLOCK)
            except (zmq.Again, zmq.error.ContextTerminated): pass
            last_right_time = now
        time.sleep(0.001)

def status_broadcast_loop(status_sock, active_mode: str, shutdown_event):
    """
    Publica o modo ativo a cada 0.5s na porta 6004.
    O LeRobot lê isso no connect() para saber para qual tópico mandar comandos.
    Mensagem: {"robot_mode": "loco"} ou {"robot_mode": "debug"}
    """
    payload = json.dumps({"robot_mode": active_mode}).encode("utf-8")
    while not shutdown_event.is_set():
        try:
            status_sock.send(payload, zmq.NOBLOCK)
        except (zmq.Again, zmq.error.ContextTerminated):
            pass
        time.sleep(0.5)


def cmd_forward_loop(lowcmd_sock, lowcmd_pub, crc, shutdown_event):
    """
    Loop de encaminhamento de comandos de corpo.
    O tópico de destino (lowcmd_pub) já foi escolhido no startup — sem troca de modo aqui.
    """
    while not shutdown_event.is_set():
        try:
            payload = lowcmd_sock.recv()
        except zmq.ContextTerminated:
            break

        msg_dict = json.loads(payload.decode("utf-8"))
        cmd = dict_to_lowcmd(msg_dict.get("data", {}))
        cmd.crc = crc.Crc(cmd)
        lowcmd_pub.Write(cmd)

def handcmd_forward_loop(handcmd_sock, left_pub, right_pub, shutdown_event):
    while not shutdown_event.is_set():
        try: payload = handcmd_sock.recv(zmq.NOBLOCK)
        except zmq.Again:
            time.sleep(0.001)
            continue
        except zmq.ContextTerminated: break
        
        msg_dict = json.loads(payload.decode("utf-8"))
        cmd = dict_to_handcmd(msg_dict.get("data", {}))
        topic = msg_dict.get("topic", "")
        if topic == kTopicDex3LeftCommand: left_pub.Write(cmd)
        elif topic == kTopicDex3RightCommand: right_pub.Write(cmd)


def loco_cmd_loop(loco_sock, loco_client, shutdown_event):
    """Recebe velocidade do PC e chama o LocoClient AQUI, no robô.

    O DDS fica todo deste lado de propósito: do PC só sai um JSON por ZMQ,
    pela mesma ponte que já carrega lowcmd e handcmd. Chamar LocoClient de
    fora exigiria abrir o domínio DDS do robô para a rede — mais superfície,
    e um segundo participante DDS competindo pelo mesmo tópico.

    Mensagens aceitas:
        {"vx": float, "vy": float, "vyaw": float}
        {"damp": true}                              → parada macia
    """
    ultimo_cmd = 0.0
    andando = False

    while not shutdown_event.is_set():
        try:
            payload = loco_sock.recv(zmq.NOBLOCK)
        except zmq.Again:
            # Sem mensagem: é aqui que o watchdog trabalha.
            if andando and (time.time() - ultimo_cmd) > LOCO_WATCHDOG_S:
                try:
                    loco_client.Move(0.0, 0.0, 0.0, continous_move=False)
                except Exception as e:
                    print(f"[Loco] Falha ao parar no watchdog: {e}")
                andando = False
                print("[Loco] ⏱️  Watchdog: sem comando do PC, robô parado.")
            time.sleep(0.005)
            continue
        except zmq.ContextTerminated:
            break

        try:
            msg = json.loads(payload.decode("utf-8"))
        except Exception:
            continue

        ultimo_cmd = time.time()

        if msg.get("damp"):
            try:
                loco_client.Damp()
                print("[Loco] 🛑 Damping por pedido do PC.")
            except Exception as e:
                print(f"[Loco] Falha no Damp: {e}")
            andando = False
            continue

        vx = float(msg.get("vx", 0.0))
        vy = float(msg.get("vy", 0.0))
        vyaw = float(msg.get("vyaw", 0.0))

        # Teto também aqui, não só no PC: o servidor não deve confiar que o
        # cliente respeitou o próprio limite.
        vx = max(-0.5, min(0.5, vx))
        vy = max(-0.5, min(0.5, vy))
        vyaw = max(-0.5, min(0.5, vyaw))

        try:
            loco_client.Move(vx, vy, vyaw, continous_move=False)
        except Exception as e:
            print(f"[Loco] Falha ao mover: {e}")
            continue

        andando = not (vx == 0.0 and vy == 0.0 and vyaw == 0.0)


def main():
    parser = argparse.ArgumentParser(description="G1 ZMQ Bridge Server")
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument(
        "--debug",
        action="store_true",
        help="Low Level (Debug Mode, rt/lowcmd): MATA a IA da Unitree. Só com o robô "
             "suspenso ou apoiado — em pé, ele cai.",
    )
    modo.add_argument(
        "--loco",
        action="store_true",
        help="High Level (Loco/WBC). Já é o padrão; aceito só por compatibilidade.",
    )
    args = parser.parse_args()

    use_loco = not args.debug
    active_mode = "loco" if use_loco else "debug"

    print("=========================================================")
    print(f"🚀 G1 ZMQ Bridge - Modo: {'HIGH LEVEL (Loco/WBC) 🏃' if use_loco else 'LOW LEVEL (Debug) 🛑'}")
    print("=========================================================")

    ChannelFactoryInitialize(0)

    # ----------------------------------------------------------------
    # SWITCH DE MODO: acontece UMA VEZ, antes de qualquer loop
    # ----------------------------------------------------------------
    ms = MotionSwitcher()
    current_hw_mode = ms.current_mode_name()
    print(f"[Startup] Modo atual do robô: '{current_hw_mode or 'debug/vazio'}'")

    if use_loco:
        if not current_hw_mode:
            # Robô está em debug — precisa ativar a IA
            ok = ms.enter_loco_mode()
            if not ok:
                print("[Startup] ⚠️  Não foi possível ativar Loco Mode. Verifique o robô e tente novamente.")
                return
        else:
            print(f"[Startup] ✅ Robô já está em Loco Mode ('{current_hw_mode}'). Nenhuma troca necessária.")
        lowcmd_topic = kTopicLowCommand_Motion
    else:
        if current_hw_mode:
            # Robô está em IA — precisa matar para assumir controle bruto
            ok = ms.enter_debug_mode()
            if not ok:
                print("[Startup] ⚠️  Não foi possível entrar em Debug Mode. Verifique o robô e tente novamente.")
                return
        else:
            print("[Startup] ✅ Robô já está em Debug Mode. Nenhuma troca necessária.")
        lowcmd_topic = kTopicLowCommand_Debug

    print(f"[Startup] Tópico de comando do corpo: {lowcmd_topic}")
    print(f"[Startup] Publicando modo '{active_mode}' na porta {STATUS_PORT} para o LeRobot.")

    crc = CRC()

    # Publicador único — tópico fixado pelo modo escolhido no startup
    lowcmd_pub = ChannelPublisher(lowcmd_topic, hg_LowCmd)
    lowcmd_pub.Init()

    lowstate_sub = ChannelSubscriber(kTopicLowState, hg_LowState)
    lowstate_sub.Init()

    left_hand_cmd_pub = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
    left_hand_cmd_pub.Init()
    right_hand_cmd_pub = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
    right_hand_cmd_pub.Init()

    left_hand_state_sub = ChannelSubscriber(kTopicDex3LeftState, HandState_)
    left_hand_state_sub.Init()
    right_hand_state_sub = ChannelSubscriber(kTopicDex3RightState, HandState_)
    right_hand_state_sub.Init()

    ctx = zmq.Context.instance()

    lowcmd_sock = ctx.socket(zmq.PULL)
    lowcmd_sock.bind(f"tcp://0.0.0.0:{LOWCMD_PORT}")

    lowstate_sock = ctx.socket(zmq.PUB)
    lowstate_sock.bind(f"tcp://0.0.0.0:{LOWSTATE_PORT}")

    handstate_sock = ctx.socket(zmq.PUB)
    handstate_sock.bind(f"tcp://0.0.0.0:{HANDSTATE_PORT}")

    handcmd_sock = ctx.socket(zmq.PULL)
    handcmd_sock.bind(f"tcp://0.0.0.0:{HANDCMD_PORT}")

    # ← NOVO: socket de status — o LeRobot lê aqui para saber o modo ativo
    status_sock = ctx.socket(zmq.PUB)
    status_sock.bind(f"tcp://0.0.0.0:{STATUS_PORT}")

    # ← NOVO: velocidade de locomoção. Só existe em --loco: sem o WBC ativo
    # o LocoClient não tem o que comandar, e abrir a porta daria a falsa
    # impressão de que o analógico do VR faria alguma coisa.
    loco_sock = None
    loco_client = None
    if use_loco:
        try:
            from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
            loco_client = LocoClient()
            loco_client.SetTimeout(0.0001)
            loco_client.Init()
            loco_sock = ctx.socket(zmq.PULL)
            loco_sock.bind(f"tcp://0.0.0.0:{LOCOCMD_PORT}")
            print(f"[Startup] 🏃 Locomoção habilitada — escutando velocidade na porta {LOCOCMD_PORT}.")
        except Exception as e:
            loco_client = None
            loco_sock = None
            print(f"[Startup] ⚠️  Locomoção DESABILITADA (LocoClient falhou: {e}). O resto segue normal.")

    shutdown_event = threading.Event()

    t_state = threading.Thread(
        target=state_forward_loop,
        args=(lowstate_sub, lowstate_sock, 0.002, shutdown_event),
        daemon=True,
    )
    t_handstate = threading.Thread(
        target=handstate_forward_loop,
        args=(left_hand_state_sub, right_hand_state_sub, handstate_sock, 0.002, shutdown_event),
        daemon=True,
    )
    t_handcmd = threading.Thread(
        target=handcmd_forward_loop,
        args=(handcmd_sock, left_hand_cmd_pub, right_hand_cmd_pub, shutdown_event),
        daemon=True,
    )
    # ← NOVO: thread que fica anunciando o modo ativo
    t_status = threading.Thread(
        target=status_broadcast_loop,
        args=(status_sock, active_mode, shutdown_event),
        daemon=True,
    )

    t_loco = None
    if loco_sock is not None:
        t_loco = threading.Thread(
            target=loco_cmd_loop,
            args=(loco_sock, loco_client, shutdown_event),
            daemon=True,
        )

    t_state.start()
    t_handstate.start()
    t_handcmd.start()
    t_status.start()
    if t_loco is not None:
        t_loco.start()

    print(f"\n[INFO] Servidor ZMQ escutando na porta {LOWCMD_PORT} para comandos de corpo...")
    print(f"[INFO] LeRobot pode conectar. Modo '{active_mode}' será informado na porta {STATUS_PORT}.")

    try:
        cmd_forward_loop(lowcmd_sock, lowcmd_pub, crc, shutdown_event)
    except KeyboardInterrupt:
        print("\nDesligando a bridge...")
    finally:
        # Parar os pés ANTES de derrubar os sockets: se a bridge cair com uma
        # velocidade pendente, o robô continua andando sem ninguém escutando.
        if loco_client is not None:
            try:
                loco_client.Move(0.0, 0.0, 0.0, continous_move=False)
            except Exception:
                pass
        shutdown_event.set()
        ctx.term()
        t_state.join(timeout=2.0)
        t_handstate.join(timeout=2.0)
        t_handcmd.join(timeout=2.0)
        t_status.join(timeout=2.0)
        if t_loco is not None:
            t_loco.join(timeout=2.0)
        print("Finalizado.")


if __name__ == "__main__":
    main()