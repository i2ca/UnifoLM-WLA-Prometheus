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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from lerobot.robots.config import RobotConfig

_GAINS: dict[str, dict[str, list[float]]] = {
    "left_leg": {
        "kp": [0, 0, 0, 0, 0, 0],
        "kd": [2, 2, 2, 4, 2, 2],
    },  # pitch, roll, yaw, knee, ankle_pitch, ankle_roll
    "right_leg": {"kp": [0, 0, 0, 0, 0, 0], "kd": [2, 2, 2, 4, 2, 2]},
    # A cintura é dividida em duas partes porque elas têm papéis diferentes:
    #   waist_yaw  → junta COMANDADA (entra no vetor de ação com use_waist_yaw)
    #   waist_lock → roll e pitch, TRAVADOS em posição neutra
    # Separadas, dá para endurecer a trava sem alterar a resposta do yaw.
    # A ordem no vetor plano continua sendo yaw(12), roll(13), pitch(14).
    # Cintura: 300/6 fazia o roll/pitch ESQUENTAR na teleoperação em loco (30/09) — travados em 0
    # brigavam com o equilíbrio do WBC. Agora 150/5 (o exemplo arm_sdk da Unitree usa 60) e a trava
    # segura a posição MEDIDA ao conectar, não 0 (ver unitree_g1_loco.py).
    "waist_yaw": {"kp": [150], "kd": [5]},          # 12
    # 01/10: trava mais dura (300/8) para a coluna ficar RETA com cintura_reta: true (alvo 0). Vigie a
    # temperatura do roll/pitch da cintura no painel — com 300/6 ela esquentava (use o guindaste).
    "waist_lock": {"kp": [300, 300], "kd": [8, 8]},  # 13 roll, 14 pitch
    # ⚠️ TEMPORÁRIO — BRAÇO ESQUERDO DESLIGADO (defeito de hardware).
    # kp=0 e kd=0 com mode=1: o motor recebe o alvo mas não gera torque nenhum,
    # então o braço fica COMPLETAMENTE MOLE e cai por gravidade assim que o
    # robô é energizado. Apoie ou amarre o braço esquerdo antes de ligar.
    # O teleop continua calculando e GRAVANDO as 7 juntas esquerdas no dataset,
    # só que elas não são executadas — episódios gravados assim têm a metade
    # esquerda do vetor de ação inútil. Restaure os valores abaixo quando o
    # braço voltar do conserto:
    #   "left_arm":   {"kp": [80, 80, 80, 80], "kd": [3, 3, 3, 0.3]}
    #   "left_wrist": {"kp": [40, 40, 40],     "kd": [1.5, 1.5, 1.5]}
    "left_arm": {"kp": [80, 80, 80, 80], "kd": [3, 3, 3, 3]},  # shoulder_pitch/roll/yaw, elbow
    "left_wrist": {"kp": [40, 40, 40], "kd": [1.5, 1.5, 1.5]},  # roll, pitch, yaw
    "right_arm": {"kp": [80, 80, 80, 80], "kd": [3, 3, 3, 0.3]},
    "right_wrist": {"kp": [40, 40, 40], "kd": [1.5, 1.5, 1.5]},
    "other": {"kp": [80, 80, 80, 80, 80, 80], "kd": [3, 3, 3, 3, 3, 3]},
}


# Total de motores no vetor plano de ganhos (29 do corpo + 6 de reserva usados
# pelo protocolo, incluindo o peso do arm_sdk no índice 29).
NUM_GAIN_SLOTS = 35


def _build_gains() -> tuple[list[float], list[float]]:
    """Achata os grupos de `_GAINS` no vetor indexado por motor.

    A ordem dos grupos no dicionário É a ordem dos índices de motor — Python
    preserva ordem de inserção desde a 3.7. Mexer na ordem aqui desloca todos os
    ganhos silenciosamente, por isso a checagem de tamanho abaixo.
    """
    kp = [v for g in _GAINS.values() for v in g["kp"]]
    kd = [v for g in _GAINS.values() for v in g["kd"]]
    if len(kp) != NUM_GAIN_SLOTS or len(kd) != NUM_GAIN_SLOTS:
        raise ValueError(
            f"_GAINS produz {len(kp)} kp e {len(kd)} kd, esperado {NUM_GAIN_SLOTS}. "
            "Alguma parte do corpo ficou com contagem errada."
        )
    return kp, kd


_DEFAULT_KP, _DEFAULT_KD = _build_gains()


@RobotConfig.register_subclass("unitree_g1_ext")
@dataclass
class UnitreeG1Config(RobotConfig):
    kp: list[float] = field(default_factory=lambda: _DEFAULT_KP.copy())
    kd: list[float] = field(default_factory=lambda: _DEFAULT_KD.copy())

    # Default joint positions
    default_positions: list[float] = field(default_factory=lambda: [0.0] * 29)

    # Control loop timestep
    control_dt: float = 1.0 / 250.0  # 250Hz

    # Control mode: "full_body" (all 29 joints) or "upper_body" (14 arm joints only)
    control_mode: str = "upper_body"

    # Inclui o yaw do tronco (kWaistYaw, motor 12) no espaço de ação.
    #
    # Só afeta os modos "upper_body" e "high_level" — em "full_body" as 29 juntas
    # já estão presentes. Com True:
    #   • o vetor de corpo passa de 14 para 15 dims (yaw no fim, dim 14)
    #   • roll e pitch da cintura (13, 14) são TRAVADOS em posição neutra
    #   • o yaw fica livre para receber comando do operador ou da política
    #
    # Sem travar roll/pitch a cintura fica mole: com mode=0/kp=0/kd=0 nenhum
    # controlador assume essas duas juntas e o tronco balança sozinho.
    #
    # ATENÇÃO: mudar isto muda o schema do dataset (28 → 29 dims com as mãos).
    # Datasets gravados com valores diferentes não podem ser juntados.
    use_waist_yaw: bool = False

    # Os ganhos da trava de cintura NÃO são campos separados: saem de `kp`/`kd`
    # nos índices 13 e 14, via o grupo "waist_lock" de _GAINS. Para endurecer ou
    # amolecer a trava, edite lá — é o único lugar onde ganho é definido.

    # Limite de curso do yaw do tronco, em radianos. O G1 aceita mais que isso,
    # mas bater no fim de curso durante uma demo estraga o episódio.
    waist_yaw_limit: float = 1.0

    # Coluna RETA: com use_waist_yaw, a trava de roll/pitch segura 0 (com o kp/kd do "waist_lock") em vez
    # de só amortecer na posição medida — sem isto o tronco ficava inclinado onde o peso dos braços deixava.
    cintura_reta: bool = False

    # Launch mujoco simulation
    is_simulation: bool = False

    remote_sim_ip: str = ""

    # Socket config for ZMQ bridge
    robot_ip: str = "192.168.123.164"  # default G1 IP
    #robot_ip: str = "10.9.8.73"  # default G1 IP
    #robot_ip: str = "127.0.0.1"  # default G1 IP


    # Cameras (ZMQ-based remote cameras)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # ── Tolerância a engasgo de rede na leitura das câmeras ───────────────
    # Um soluço de Wi-Fi de alguns centenas de ms fazia `cam.async_read()`
    # estourar TimeoutError e MATAR a teleoperação inteira. Agora a leitura
    # só espera `camera_read_timeout_ms` por um quadro NOVO; se não vier,
    # reaproveita o último quadro bom e segue o laço. O robô continua
    # obedecendo — o que congela é a imagem, por alguns quadros.
    #
    # Esperar muito aqui é pior do que reaproveitar: cada ms deste timeout é
    # um ms de laço de controle parado. Mantenha na ordem do período do FPS
    # da câmera (30 fps → 33 ms), com folga.
    camera_read_timeout_ms: int = 300

    # Aí sim: se a câmera ficar MUDA por mais que isto — sem um único quadro
    # novo —, não é engasgo, é servidor caído. Só nesse caso o erro sobe e
    # derruba a sessão, para não gravar dataset com imagem congelada.
    camera_grace_s: float = 10.0
