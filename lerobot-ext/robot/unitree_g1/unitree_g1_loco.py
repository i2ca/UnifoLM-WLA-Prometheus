#!/usr/bin/env python

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

import json
import logging
import struct
import threading
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.envs.factory import make_env
from lerobot.processor import RobotAction, RobotObservation
from .g1_utils import (
    G1_29_JointIndex,
    G1_29_JointArmIndex,
    G1_29_JointArmWaistIndex,
    G1_WAIST_LOCKED_JOINTS,
)

from lerobot.robots.robot import Robot
from .config_unitree_g1 import UnitreeG1Config

logger = logging.getLogger(__name__)

# DDS topic names follow Unitree SDK naming conventions
# ruff: noqa: N816
kTopicLowCommand_Debug  = "rt/lowcmd"
kTopicLowCommand_Motion = "rt/arm_sdk"  # High Level / Loco (WBC)
kTopicLowState          = "rt/lowstate"

# Porta ZMQ onde o servidor anuncia o modo ativo ("loco" ou "debug")
_STATUS_PORT = 6004

# Índice do campo de peso do arm_sdk (kNotUsedJoint na SDK oficial da Unitree)
# motor_cmd[29].q = 1.0 ativa o controle de braço via rt/arm_sdk
_ARM_SDK_WEIGHT_IDX = 29


@dataclass
class MotorState:
    q: float | None = None  # position
    dq: float | None = None  # velocity
    tau_est: float | None = None  # estimated torque
    temperature: float | None = None  # motor temperature


@dataclass
class IMUState:
    quaternion: np.ndarray | None = None  # [w, x, y, z]
    gyroscope: np.ndarray | None = None  # [x, y, z] angular velocity (rad/s)
    accelerometer: np.ndarray | None = None  # [x, y, z] linear acceleration (m/s²)
    rpy: np.ndarray | None = None  # [roll, pitch, yaw] (rad)
    temperature: float | None = None  # IMU temperature


# g1 observation class
@dataclass
class G1_29_LowState:  # noqa: N801
    motor_state: list[MotorState] = field(default_factory=lambda: [MotorState() for _ in G1_29_JointIndex])
    imu_state: IMUState = field(default_factory=IMUState)
    wireless_remote: Any = None  # Raw wireless remote data
    mode_machine: int = 0  # Robot mode


class UnitreeG1(Robot):
    config_class = UnitreeG1Config
    name = "unitree_g1_ext"

    # unitree remote controller
    class RemoteController:
        def __init__(self):
            self.lx = 0
            self.ly = 0
            self.rx = 0
            self.ry = 0
            self.button = [0] * 16

        def set(self, data):
            # wireless_remote
            keys = struct.unpack("H", data[2:4])[0]
            for i in range(16):
                self.button[i] = (keys & (1 << i)) >> i
            self.lx = struct.unpack("f", data[4:8])[0]
            self.rx = struct.unpack("f", data[8:12])[0]
            self.ry = struct.unpack("f", data[12:16])[0]
            self.ly = struct.unpack("f", data[20:24])[0]

    def __init__(self, config: UnitreeG1Config):
        super().__init__(config)

        logger.info("Initialize UnitreeG1...")

        self.config = config
        self.control_dt = config.control_dt

        # Initialize cameras config (ZMQ-based) - actual connection in connect()
        self._cameras = make_cameras_from_configs(config.cameras)

        # Último quadro bom de cada câmera e desde quando ela está muda.
        # Ver `_read_camera`: é o que segura a teleoperação de pé quando a rede
        # engasga por alguns quadros.
        self._cam_ultimo_quadro: dict[str, np.ndarray] = {}
        self._cam_mudo_desde: dict[str, float] = {}
        self._cam_ultimo_aviso: dict[str, float] = {}

        # Channel classes will be imported in connect() to avoid circular imports
        self._ChannelFactoryInitialize = None
        self._ChannelPublisher = None
        self._ChannelSubscriber = None

        # Initialize state variables
        self.sim_env = None
        self._env_wrapper = None
        self._lowstate = None
        self._shutdown_event = threading.Event()
        self.subscribe_thread = None
        self.remote_controller = self.RemoteController()

        self.last_action_q = {}
        self.smoothing_alpha = 0.1  # Ajuste entre 0.05 (muito suave) e 0.3 (mais responsivo)

        # Modo ativo: "loco" (rt/arm_sdk) ou "debug" (rt/lowcmd)
        # Inicializado em connect() após handshake com o servidor
        self._active_mode = "debug"

    def _query_server_mode(self, robot_ip: str, timeout: float = 5.0) -> str:
        """
        Conecta na porta 6004 do servidor ZMQ e lê o modo ativo ("loco" ou "debug").
        Aguarda até `timeout` segundos. Se não receber resposta, assume "debug" por segurança.
        """
        import zmq
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.connect(f"tcp://{robot_ip}:{_STATUS_PORT}")
        try:
            payload = sock.recv()
            msg = json.loads(payload.decode("utf-8"))
            mode = msg.get("robot_mode", "debug")
            logger.info(f"[UnitreeG1] Servidor confirmou modo: '{mode}'")
            return mode
        except zmq.Again:
            logger.warning(
                f"[UnitreeG1] Timeout ao consultar o servidor ({timeout}s). "
                "Assumindo 'debug' por segurança. Verifique se dex3_g1_server_v2.py está rodando."
            )
            return "debug"
        except Exception as e:
            logger.warning(f"[UnitreeG1] Erro ao consultar modo do servidor: {e}. Assumindo 'debug'.")
            return "debug"
        finally:
            sock.close()

    def _subscribe_motor_state(self):  # polls robot state @ 250Hz
        while not self._shutdown_event.is_set():
            start_time = time.time()

            # Step simulation if in simulation mode
            if self.config.is_simulation and self.sim_env is not None:
                self.sim_env.step()

            # Timeout explícito para esta thread nunca poder parar de vez. `Read()` sem
            # argumento vira `take_one()` no cyclonedds, que espera indefinidamente
            # (`timeout or duration(weeks=99999)`) — e em simulação quem PRODUZ o lowstate
            # é o `sim_env.step()` logo acima, na mesma thread. O laço só gira porque cada
            # step publica uma amostra antes do Read consumi-la; se uma se perder, o Read
            # ficaria esperando um estado que ninguém mais publicaria.
            #
            # Nunca vi isso acontecer — em 60 s de teleoperação em simulação não houve um
            # único timeout aqui. É proteção barata contra um travamento que o desenho
            # permite, não conserto de bug observado.
            #
            # 100 ms é folgado para os dois lados: em simulação o step acabou de publicar,
            # e no robô real o lowstate vem a centenas de hertz. Quando estoura, o
            # unitree_sdk2py devolve None — caso já tratado logo abaixo.
            msg = self.lowstate_subscriber.Read(timeout=0.1)
            if msg is not None:
                lowstate = G1_29_LowState()

                # Capture motor states using jointindex
                for id in G1_29_JointIndex:
                    lowstate.motor_state[id].q = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                    lowstate.motor_state[id].tau_est = msg.motor_state[id].tau_est
                    lowstate.motor_state[id].temperature = msg.motor_state[id].temperature

                # Capture IMU state
                lowstate.imu_state.quaternion = list(msg.imu_state.quaternion)
                lowstate.imu_state.gyroscope = list(msg.imu_state.gyroscope)
                lowstate.imu_state.accelerometer = list(msg.imu_state.accelerometer)
                lowstate.imu_state.rpy = list(msg.imu_state.rpy)
                lowstate.imu_state.temperature = msg.imu_state.temperature

                # Capture wireless remote data
                lowstate.wireless_remote = msg.wireless_remote

                # Capture mode_machine
                lowstate.mode_machine = msg.mode_machine

                self._lowstate = lowstate

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))  # maintain constant control dt
            time.sleep(sleep_time)

    @property
    def body_joint_index(self):
        """
        Enum das juntas de corpo que entram no vetor de ação/estado.

        Fonte única da verdade desta classe. ATENÇÃO: existe um `body_joint_index`
        gêmeo em `unitree_g1.py` — as duas `UnitreeG1` são classes PARALELAS
        (nenhuma herda da outra), e o `UnitreeG1Dex3` herda desta aqui. Mudança
        numa precisa ser espelhada na outra.

          full_body                     → 29 juntas (tudo)
          upper_body/high_level         → 14 braços
          upper_body/high_level + waist → 14 braços + yaw do tronco (dim 14)
        """
        if self.config.control_mode not in ("upper_body", "high_level"):
            return G1_29_JointIndex
        if getattr(self.config, "use_waist_yaw", False):
            return G1_29_JointArmWaistIndex
        return G1_29_JointArmIndex

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Define action space based on control mode (ver `body_joint_index`)."""
        return {f"{motor.name}.q": float for motor in self.body_joint_index}

    def calibrate(self) -> None:  # robot is already calibrated
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:  # connect to DDS
        # Import channel classes and message types based on mode
        # (deferred imports to avoid circular import in unitree_sdk2py)
        #
        # DDS de verdade SÓ na simulação LOCAL. Na remota o MuJoCo está em outra
        # máquina e quem fala DDS lá é a ponte `run_sim_remote.py`, que expõe
        # ZMQ nas portas 6000-6003 — o mesmo protocolo do servidor do robô real.
        # Com o DDS aqui, `ChannelFactoryInitialize(0, "<ip>")` passava o IP
        # como nome de interface ao CycloneDDS e morria com
        # "does not match an available interface".
        if self.config.is_simulation and not self.config.remote_sim_ip:
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
                LowCmd_ as hg_LowCmd,
                LowState_ as hg_LowState,
            )
            from unitree_sdk2py.utils.crc import CRC
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )
            LowCmdMsg = unitree_hg_msg_dds__LowCmd_
        else:
            # Use SDK-free wrappers for ZMQ mode to avoid circular import
            from .unitree_sdk2_socket import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
                LowCmdMsg,
                CRC,
                hg_LowCmd,
                hg_LowState,
            )

        self._ChannelFactoryInitialize = ChannelFactoryInitialize
        self._ChannelPublisher = ChannelPublisher
        self._ChannelSubscriber = ChannelSubscriber

        # Initialize DDS channel and simulation environment
        if self.config.is_simulation and not self.config.remote_sim_ip:
            # ── Simulação LOCAL (comportamento original) ──────────────────
            import sys, os
            local_sim_path = os.path.expanduser("../unitree-g1-mujoco")
            if local_sim_path not in sys.path:
                sys.path.append(local_sim_path)
            from env import make_env as make_local_env
            lista_de_cameras = list(self.config.cameras.keys())
            if "head_camera_depth" not in lista_de_cameras:
                lista_de_cameras.append("head_camera_depth")
            self.sim_env = make_local_env(cameras=lista_de_cameras)
            time.sleep(3.0)
            # Simulação local não tem servidor ZMQ — usa o modo do config diretamente
            active_mode = "loco" if self.config.control_mode == "high_level" else "debug"
            logger.info(f"[UnitreeG1] Simulação local: modo derivado do config = '{active_mode}'")

        elif self.config.remote_sim_ip:
            # ── Simulação REMOTA: MuJoCo roda no PC do Miguel ─────────────
            # Não sobe nada localmente: os sockets ZMQ apontam para a ponte
            # `run_sim_remote.py` no PC remoto (portas 6000-6003).
            self.config.robot_ip = self.config.remote_sim_ip
            self._ChannelFactoryInitialize(0, self.config.remote_sim_ip)
            print(f"🌐 Modo simulação remota: conectando ao MuJoCo em {self.config.remote_sim_ip}")
            # A ponte não publica o modo na 6004 (é coisa do servidor do robô
            # real); consultar só custaria dois timeouts de 5 s. Mesma regra da
            # simulação local.
            active_mode = "loco" if self.config.control_mode == "high_level" else "debug"
            logger.info(f"[UnitreeG1] Simulação remota: modo derivado do config = '{active_mode}'")

        else:
            # ── Robô real ─────────────────────────────────────────────────
            # Passa robot_ip para que os sockets ZMQ apontem para o robô.
            # O handshake abaixo busca o modo ativo na porta 6004.
            self._ChannelFactoryInitialize(0, self.config.robot_ip)

        # ----------------------------------------------------------------
        # HANDSHAKE DE MODO (robô real e simulação remota):
        # Lê da porta 6004 qual modo o servidor está usando e escolhe
        # o tópico de comando correto. Não requer nenhum switch aqui.
        # ----------------------------------------------------------------
        if not self.config.is_simulation:
            active_mode = self._query_server_mode(self.config.robot_ip)
            logger.info(f"[UnitreeG1] Modo reportado pelo servidor: '{active_mode}'")

        # Salva o modo ativo — usado no send_action para manter o peso do arm_sdk
        self._active_mode = active_mode

        # Escolhe o tópico de comando baseado no modo
        if active_mode == "loco":
            lowcmd_topic = kTopicLowCommand_Motion
            logger.info("[UnitreeG1] → Usando rt/arm_sdk (High Level / WBC ativo)")
        else:
            lowcmd_topic = kTopicLowCommand_Debug
            logger.info("[UnitreeG1] → Usando rt/lowcmd (Low Level / Debug Mode)")

        # Initialize direct motor control interface usando tópico escolhido pelo handshake
        self.lowcmd_publisher = self._ChannelPublisher(lowcmd_topic, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = self._ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()

        # Start subscribe thread to read robot state
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.start()

        # Connect cameras
        for cam in self._cameras.values():
            if not cam.is_connected:
                cam.connect()

        logger.info(f"Connected {len(self._cameras)} camera(s).")

        # ----------------------------------------------------------------
        # Cria o objeto de mensagem AQUI — tudo que usa self.msg vem depois
        # ----------------------------------------------------------------
        self.crc = CRC()
        self.msg = LowCmdMsg()
        self.msg.mode_pr = 0
        print(f"[DEBUG] motor_cmd size: {len(self.msg.motor_cmd)}")

        # Wait for first state message to arrive
        lowstate = None
        while lowstate is None:
            lowstate = self._lowstate
            if lowstate is None:
                time.sleep(0.01)
            logger.warning("[UnitreeG1] Waiting for robot state...")
        logger.warning("[UnitreeG1] Connected to robot.")
        self.msg.mode_machine = lowstate.mode_machine

        # Initialize all motors with unified kp/kd from config
        self.kp = np.array(self.config.kp, dtype=np.float32)
        self.kd = np.array(self.config.kd, dtype=np.float32)

        # Configura os motores baseado no modo de controle.
        # No modo loco: WBC cuida de tudo — só controlamos os braços (idx 15..28).
        # Cintura (12,13,14) e pernas (0..11) ficam com mode=0 para o WBC gerenciar.
        ARM_IDX_MIN = G1_29_JointArmIndex.kLeftShoulderPitch   # 15
        ARM_IDX_MAX = G1_29_JointArmIndex.kRightWristYaw        # 28

        # ── Cintura no modo loco ──────────────────────────────────────────
        # Ceder as juntas 12-14 ao WBC (mode=0/kp=0/kd=0) deixa a cintura MOLE:
        # observado no robô real, o tronco balança sozinho enquanto o G1 anda,
        # porque na prática nenhum controlador assume essas juntas.
        #
        # Com use_waist_yaw=True:
        #   • roll (13) e pitch (14) → travados com ganho duro em posição neutra
        #   • yaw (12)               → comandado como as juntas de braço
        #
        # As pernas (0-11) continuam integralmente com o WBC — ali ele de fato
        # atua, e disputar seria brigar com o equilíbrio.
        use_waist_yaw = getattr(self.config, "use_waist_yaw", False)
        commanded = {m.value for m in self.body_joint_index}

        for id in G1_29_JointIndex:
            is_arm = ARM_IDX_MIN <= id.value <= ARM_IDX_MAX  # índices 15..28

            if use_waist_yaw and id.value in G1_WAIST_LOCKED_JOINTS:
                # Ganho vem da array `kp`/`kd` (grupo "waist_lock" em _GAINS),
                # como todas as outras juntas. Um valor fixo mais fraco aqui foi
                # a causa do tronco tombar para a frente.
                self.msg.motor_cmd[id.value].mode = 1
                self.msg.motor_cmd[id.value].kp = self.kp[id.value]
                self.msg.motor_cmd[id.value].kd = self.kd[id.value]
                # cintura_reta (01/10): alvo 0 = coluna RETA, com o kp/kd do grupo "waist_lock".
                # Sem ela: a posição MEDIDA a cada ciclo (só amortece; forçar 0 com kp baixo brigava com o
                # equilíbrio do WBC e o roll/pitch esquentava, 30/09) — a coluna ficava onde caísse.
                self.msg.motor_cmd[id.value].q = (0.0 if getattr(self.config, "cintura_reta", False)
                                                  else float(lowstate.motor_state[id.value].q))
                self.msg.motor_cmd[id.value].dq = 0.0
                self.msg.motor_cmd[id.value].tau = 0.0
                continue

            if not is_arm and id.value in commanded:
                # Yaw do tronco: comandado igual aos braços.
                self.msg.motor_cmd[id.value].mode = 1
                self.msg.motor_cmd[id.value].kp = self.kp[id.value]
                self.msg.motor_cmd[id.value].kd = self.kd[id.value]
                self.msg.motor_cmd[id.value].q = lowstate.motor_state[id.value].q
                self.msg.motor_cmd[id.value].dq = 0.0
                self.msg.motor_cmd[id.value].tau = 0.0
                continue

            if not is_arm and self.config.control_mode in ("upper_body", "high_level"):
                # Pernas: cede ao WBC completamente (mode=0 = sem torque daqui)
                self.msg.motor_cmd[id.value].mode = 0
                self.msg.motor_cmd[id.value].kp = 0.0
                self.msg.motor_cmd[id.value].kd = 0.0
                self.msg.motor_cmd[id.value].q = 0.0
                self.msg.motor_cmd[id.value].dq = 0.0
                self.msg.motor_cmd[id.value].tau = 0.0
            else:
                # Braços: controlados pelo modelo de ML
                self.msg.motor_cmd[id.value].mode = 1
                self.msg.motor_cmd[id.value].kp = self.kp[id.value]
                self.msg.motor_cmd[id.value].kd = self.kd[id.value]
                self.msg.motor_cmd[id.value].q = lowstate.motor_state[id.value].q
                self.msg.motor_cmd[id.value].dq = 0.0
                self.msg.motor_cmd[id.value].tau = 0.0

        # ----------------------------------------------------------------
        # Ativa o arm_sdk: motor_cmd[29].q=1 (kNotUsedJoint) sinaliza ao WBC
        # que este cliente controla os braços via rt/arm_sdk.
        # mode_pr=1 = controle por posição.
        # ----------------------------------------------------------------
        if active_mode == "loco":
            self.msg.mode_pr = 1
            self.msg.motor_cmd[_ARM_SDK_WEIGHT_IDX].q = 1.0
            logger.info(f"[UnitreeG1] arm_sdk ativado: mode_pr=1, motor_cmd[{_ARM_SDK_WEIGHT_IDX}].q=1.0")

    def reset(self, default_positions: list[float] | None = None, **kwargs):
        # Reseta o corpo (braços) – isso já existe na classe pai
        super().reset(default_positions=default_positions, **kwargs)
        # Reseta as mãos
        self.reset_hands(default_positions)

    def disconnect(self):
        # Signal thread to stop and unblock any waits
        self._shutdown_event.set()

        # Wait for subscribe thread to finish
        if self.subscribe_thread is not None:
            self.subscribe_thread.join(timeout=2.0)
            if self.subscribe_thread.is_alive():
                logger.warning("Subscribe thread did not stop cleanly")

        # Close simulation environment
        if self.config.is_simulation and self.sim_env is not None:
            try:
                # Force-kill the image publish subprocess first to avoid long waits
                if hasattr(self.sim_env, "simulator") and hasattr(self.sim_env.simulator, "sim_env"):
                    sim_env_inner = self.sim_env.simulator.sim_env
                    if hasattr(sim_env_inner, "image_publish_process"):
                        proc = sim_env_inner.image_publish_process
                        if proc.process and proc.process.is_alive():
                            logger.info("Force-terminating image publish subprocess...")
                            proc.stop_event.set()
                            proc.process.terminate()
                            proc.process.join(timeout=1)
                            if proc.process.is_alive():
                                proc.process.kill()
                self.sim_env.close()
            except Exception as e:
                logger.warning(f"Error closing sim_env: {e}")
            self.sim_env = None
            self._env_wrapper = None

        # Disconnect cameras
        for cam in self._cameras.values():
            cam.disconnect()

    def get_observation(self) -> RobotObservation:
        lowstate = self._lowstate
        if lowstate is None:
            return {}

        obs = {}

        # Select joints based on control mode
        joint_index = self.body_joint_index

        # Motors - q, dq, tau for controlled joints
        for motor in joint_index:
            name = motor.name
            idx = motor.value
            obs[f"{name}.q"] = lowstate.motor_state[idx].q
            obs[f"{name}.dq"] = lowstate.motor_state[idx].dq
            obs[f"{name}.tau"] = lowstate.motor_state[idx].tau_est

        # IMU - gyroscope
        if lowstate.imu_state.gyroscope:
            obs["imu.gyro.x"] = lowstate.imu_state.gyroscope[0]
            obs["imu.gyro.y"] = lowstate.imu_state.gyroscope[1]
            obs["imu.gyro.z"] = lowstate.imu_state.gyroscope[2]

        # IMU - accelerometer
        if lowstate.imu_state.accelerometer:
            obs["imu.accel.x"] = lowstate.imu_state.accelerometer[0]
            obs["imu.accel.y"] = lowstate.imu_state.accelerometer[1]
            obs["imu.accel.z"] = lowstate.imu_state.accelerometer[2]

        # IMU - quaternion
        if lowstate.imu_state.quaternion:
            obs["imu.quat.w"] = lowstate.imu_state.quaternion[0]
            obs["imu.quat.x"] = lowstate.imu_state.quaternion[1]
            obs["imu.quat.y"] = lowstate.imu_state.quaternion[2]
            obs["imu.quat.z"] = lowstate.imu_state.quaternion[3]

        # IMU - rpy
        if lowstate.imu_state.rpy:
            obs["imu.rpy.roll"] = lowstate.imu_state.rpy[0]
            obs["imu.rpy.pitch"] = lowstate.imu_state.rpy[1]
            obs["imu.rpy.yaw"] = lowstate.imu_state.rpy[2]

        # Controller - parse wireless_remote and add to obs
        if lowstate.wireless_remote and len(lowstate.wireless_remote) >= 24:
            self.remote_controller.set(lowstate.wireless_remote)
        obs["remote.buttons"] = self.remote_controller.button.copy()
        obs["remote.lx"] = self.remote_controller.lx
        obs["remote.ly"] = self.remote_controller.ly
        obs["remote.rx"] = self.remote_controller.rx
        obs["remote.ry"] = self.remote_controller.ry

        # Cameras - read images from ZMQ cameras
        for cam_name, cam in self._cameras.items():
            obs[cam_name] = self._read_camera(cam_name, cam)

        return obs

    def _read_camera(self, nome: str, cam) -> np.ndarray:
        """Lê um quadro da câmera SEM deixar a rede derrubar a teleoperação.

        O caminho antigo era `cam.async_read()` cru: qualquer engasgo de rede
        que atrasasse o quadro além do timeout levantava TimeoutError, que
        subia pelo `get_observation` e matava o processo inteiro no meio da
        sessão. Um soluço de Wi-Fi de 200 ms custava a demonstração toda.

        Agora a espera é curta (`camera_read_timeout_ms`) e o fracasso é
        barato: reaproveita o último quadro bom e segue. Só se a câmera ficar
        muda por mais de `camera_grace_s` — servidor caído, não engasgo — o
        erro sobe, porque aí seguir seria gravar dataset com imagem parada.
        """
        try:
            frame = cam.async_read(timeout_ms=self.config.camera_read_timeout_ms)
        except Exception as e:
            anterior = self._cam_ultimo_quadro.get(nome)
            if anterior is None:
                # Nunca chegou um quadro sequer: não é engasgo, é a câmera no
                # ar errado (IP, porta, servidor não subiu). Falhar aqui é o
                # certo — e acontece no início, não no meio da demo.
                raise

            agora = time.monotonic()
            desde = self._cam_mudo_desde.setdefault(nome, agora)
            mudo_ha = agora - desde

            if mudo_ha > self.config.camera_grace_s:
                raise TimeoutError(
                    f"Câmera '{nome}' sem nenhum quadro novo há {mudo_ha:.1f}s "
                    f"(limite: {self.config.camera_grace_s}s). Não é engasgo de "
                    f"rede — verifique o servidor de imagem no robô."
                ) from e

            # Aviso no máximo 1x por segundo: em 30 Hz isto floodaria o log.
            if agora - self._cam_ultimo_aviso.get(nome, 0.0) > 1.0:
                self._cam_ultimo_aviso[nome] = agora
                logger.warning(
                    f"Câmera '{nome}': sem quadro novo há {mudo_ha:.2f}s "
                    f"(rede engasgada) — reusando o último quadro."
                )
            # Devolve uma cópia: o array guardado é a nossa reserva e não pode sair
            # daqui para as mãos de quem escreve nele.
            return anterior.copy()

        # Guardar uma CÓPIA, não a referência. O `frame` que sai daqui é o mesmo objeto
        # que o consumidor recebe — e a camada ZMQ do lerobot também entrega o array do
        # `latest_frames` sem copiar. Se alguém desenhar um HUD em cima da observação,
        # sem esta cópia a reserva seria corrompida junto, e o estrago só apareceria no
        # próximo engasgo de rede: imagem com lixo, no pior momento possível, sem erro
        # nenhum no log.
        #
        # O custo é um memcpy por quadro por câmera (~1,2 MB a 848x480x3): irrisório
        # perto do decode que acabou de acontecer, e pago de propósito.
        self._cam_ultimo_quadro[nome] = frame.copy()
        self._cam_mudo_desde.pop(nome, None)
        return frame

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def is_connected(self) -> bool:
        return self._lowstate is not None

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features based on control mode (ver `body_joint_index`)."""
        return {f"{motor.name}.q": float for motor in self.body_joint_index}

    @property
    def cameras(self) -> dict:
        return self._cameras

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Define os tensores de imagem: RGB com 3 canais, profundidade com 1.

        O número de canais aqui não é papelada: é ele que decide como o LeRobot
        0.6.1 GRAVA a câmera. Com 1 canal o `hw_to_dataset_features` marca
        `info["is_depth_map"] = True` sozinho, e a câmera passa a ser gravada
        como mapa de profundidade — quadros em TIFF sem perda e vídeo pelo
        `DepthEncoderConfig` (HEVC gray12le, quantização log de 12 bits), com o
        inteiro lido como MILÍMETRO. Com 3 canais ela vira vídeo RGB comum e a
        medida se perde no h264.

        Quem precisa casar com isto é o servidor de imagem: o
        `full_realsenser_server.py` publica a profundidade crua em uint16 (mm),
        1 canal. Trocar um lado sem o outro quebra a gravação.
        """
        features = {}
        for cam in self.cameras:
            canais = 1 if cam.endswith("depth") else 3
            features[cam] = (self.config.cameras[cam].height, self.config.cameras[cam].width, canais)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    def send_action(self, action: RobotAction) -> RobotAction:
        # Inclui o yaw do tronco quando use_waist_yaw=True (ver body_joint_index)
        joint_index = self.body_joint_index

        max_delta = 0.1  # Radianos por ciclo. (~5 rad/s a 250Hz). Ajuste conforme necessário.
        waist_limit = getattr(self.config, "waist_yaw_limit", 1.0)

        for motor in joint_index:
            key = f"{motor.name}.q"
            if key in action:
                target_q = action[key]

                # Curso do tronco limitado: bater no fim de curso durante uma
                # demonstração estraga o episódio e castiga o motor.
                if motor.name == "kWaistYaw":
                    target_q = float(np.clip(target_q, -waist_limit, waist_limit))

                # Inicializa se for a primeira vez.
                # Pega a posição ATUAL do robô para evitar um tranco no primeiro frame.
                if key not in self.last_action_q:
                    if self._lowstate is not None:
                        self.last_action_q[key] = self._lowstate.motor_state[motor.value].q
                    else:
                        self.last_action_q[key] = target_q

                # 1. FILTRO: Suaviza a transição (Low-pass)
                smoothed_q = (1 - self.smoothing_alpha) * self.last_action_q[key] + self.smoothing_alpha * target_q

                # 2. LIMITADOR: Garante que a variação não ultrapasse o max_delta
                delta = smoothed_q - self.last_action_q[key]
                delta_clipped = np.clip(delta, -max_delta, max_delta)
                final_q = float(self.last_action_q[key] + delta_clipped)

                # 3. ATUALIZA ESTADO E COMANDO
                self.last_action_q[key] = final_q
                self.msg.motor_cmd[motor.value].q = final_q

                # Restante dos parâmetros...
                self.msg.motor_cmd[motor.value].qd = 0  # Velocidade desejada zero
                self.msg.motor_cmd[motor.value].kp = self.kp[motor.value]
                self.msg.motor_cmd[motor.value].kd = self.kd[motor.value]
                self.msg.motor_cmd[motor.value].tau = 0

        # ----------------------------------------------------------------
        # Reenvia o peso do arm_sdk a cada frame — mantém o WBC obedecendo.
        # motor_cmd[29].q=1 e mode_pr=1 precisam estar presentes em todo frame.
        # ----------------------------------------------------------------
        if self._active_mode == "loco":
            self.msg.mode_pr = 1
            self.msg.motor_cmd[_ARM_SDK_WEIGHT_IDX].q = 1.0

        self.msg.crc = self.crc.Crc(self.msg)
        self.lowcmd_publisher.Write(self.msg)
        return action

    def get_gravity_orientation(self, quaternion):  # get gravity orientation from quaternion
        """Get gravity orientation from quaternion."""
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)
        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
        return gravity_orientation

    def reset(
        self,
        control_dt: float | None = None,
        default_positions: list[float] | None = None,
    ) -> None:  # move robot to default position
        if control_dt is None:
            control_dt = self.config.control_dt
        if default_positions is None:
            default_positions = np.array(self.config.default_positions, dtype=np.float32)

        # SELECIONA OS MOTORES BASEADO NO MODO (Igual fizemos no send_action)
        joint_index = self.body_joint_index

        if self.config.is_simulation and self.sim_env is not None:
            self.sim_env.reset()

            for motor in joint_index:
                self.msg.motor_cmd[motor.value].q = default_positions[motor.value]
                self.msg.motor_cmd[motor.value].qd = 0
                self.msg.motor_cmd[motor.value].kp = self.kp[motor.value]
                self.msg.motor_cmd[motor.value].kd = self.kd[motor.value]
                self.msg.motor_cmd[motor.value].tau = 0
            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)
        else:
            total_time = 3.0
            num_steps = int(total_time / control_dt)

            # get current state
            obs = self.get_observation()

            # record current positions
            init_dof_pos = np.zeros(29, dtype=np.float32)
            for motor in joint_index:
                init_dof_pos[motor.value] = obs[f"{motor.name}.q"]

            # Interpolate to default position
            for step in range(num_steps):
                start_time = time.time()

                alpha = step / num_steps
                action_dict = {}
                for motor in joint_index:
                    target_pos = default_positions[motor.value]
                    interp_pos = init_dof_pos[motor.value] * (1 - alpha) + target_pos * alpha
                    action_dict[f"{motor.name}.q"] = float(interp_pos)

                self.send_action(action_dict)

                # Maintain constant control rate
                elapsed = time.time() - start_time
                sleep_time = max(0, control_dt - elapsed)
                time.sleep(sleep_time)

        logger.info("Reached default position")