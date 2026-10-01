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

"""
Unitree G1 Robot Constants and Joint Definitions.

This module provides a single source of truth for:
- Body joint indices (G1_29_JointIndex)
- Arm joint indices (G1_29_JointArmIndex)  
- Dex3 hand joint indices and limits
- DDS topic names
- Joint name mappings
"""

from enum import IntEnum
import numpy as np

# ruff: noqa: N801, N815

# ==============================================================================
# Body Constants
# ==============================================================================

NUM_MOTORS = 35


class G1_29_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28


class G1_29_JointArmWaistIndex(IntEnum):
    """Braços + yaw do tronco — espaço de ação quando `use_waist_yaw=True`.

    `kWaistYaw` é declarado POR ÚLTIMO de propósito. O IntEnum do Python itera na
    ordem de DEFINIÇÃO (não de valor), e é essa iteração que monta o vetor de
    ação/estado. Com o yaw no fim, as 14 dimensões de braço mantêm exatamente os
    índices 0..13 que a documentação de `loss_per_dim` já usa; o yaw vira a
    dimensão 14 do bloco do corpo, e as mãos passam a ocupar 15..28.

    Roll e pitch da cintura NÃO entram: eles ficam travados em posição neutra
    (ver `unitree_g1_loco.py`). Só o yaw é comandado — é o grau de liberdade que
    o operador usa para girar o tronco e alcançar objetos fora do alcance frontal.
    """

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

    # Tronco — último, para não deslocar os índices dos braços
    kWaistYaw = 12


# Juntas da cintura que ficam TRAVADAS em posição quando o yaw é liberado.
# Sem isso a cintura fica mole: com mode=0/kp=0/kd=0 nenhum controlador assume o
# roll e o pitch, e o tronco balança sozinho enquanto o robô anda.
G1_WAIST_LOCKED_JOINTS = (13, 14)  # kWaistRoll, kWaistPitch


class G1_29_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28


# ==============================================================================
# Dex3 Hand Constants
# ==============================================================================

# DDS Topic names for Dex3 hand communication
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"

# Number of motors per hand
Dex3_Num_Motors = 7

# ==============================================================================
# Dex3 Tactile & Temperature Sensor Constants
# ==============================================================================

class Dex3_1_Left_PressureTemperatureSensors(IntEnum):
    kLeftThumbBase = 0
    kLeftThumbTip = 1
    kLeftMiddleBase = 2
    kLeftMiddleTip = 3
    kLeftIndexBase = 4
    kLeftIndexTip = 5
    kLeftPalm0 = 6
    kLeftPalm1 = 7
    kLeftPalm2 = 8

class Dex3_1_Right_PressureTemperatureSensors(IntEnum):
    kRightThumbBase = 0
    kRightThumbTip = 1
    kRightMiddleBase = 2
    kRightMiddleTip = 3
    kRightIndexBase = 4
    kRightIndexTip = 5
    kRightPalm0 = 6
    kRightPalm1 = 7
    kRightPalm2 = 8

sensor_index = {
    Dex3_1_Left_PressureTemperatureSensors.kLeftThumbBase: [0, 2, 9, 11],
    Dex3_1_Left_PressureTemperatureSensors.kLeftThumbTip: [3, 6, 8],
    Dex3_1_Left_PressureTemperatureSensors.kLeftMiddleBase: [0, 2, 9, 11],
    Dex3_1_Left_PressureTemperatureSensors.kLeftMiddleTip: [3, 6, 8],
    Dex3_1_Left_PressureTemperatureSensors.kLeftIndexBase: [0, 2, 9, 11],
    Dex3_1_Left_PressureTemperatureSensors.kLeftIndexTip: [3, 6, 8],
    Dex3_1_Left_PressureTemperatureSensors.kLeftPalm0: [0, 2, 9, 11],
    Dex3_1_Left_PressureTemperatureSensors.kLeftPalm1: [0, 2, 9, 11],
    Dex3_1_Left_PressureTemperatureSensors.kLeftPalm2: [0, 2, 9, 11],
    Dex3_1_Right_PressureTemperatureSensors.kRightThumbBase: [0, 2, 9, 11],
    Dex3_1_Right_PressureTemperatureSensors.kRightThumbTip: [3, 6, 8],
    Dex3_1_Right_PressureTemperatureSensors.kRightMiddleBase: [0, 2, 9, 11],
    Dex3_1_Right_PressureTemperatureSensors.kRightMiddleTip: [3, 6, 8],
    Dex3_1_Right_PressureTemperatureSensors.kRightIndexBase: [0, 2, 9, 11],
    Dex3_1_Right_PressureTemperatureSensors.kRightIndexTip: [3, 6, 8],
    Dex3_1_Right_PressureTemperatureSensors.kRightPalm0: [0, 2, 9, 11],
    Dex3_1_Right_PressureTemperatureSensors.kRightPalm1: [0, 2, 9, 11],
    Dex3_1_Right_PressureTemperatureSensors.kRightPalm2: [0, 2, 9, 11],
}

class Dex3_1_Left_JointIndex(IntEnum):
    """Left Dex3-1 hand joint indices (matches DDS message structure)."""
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6


class Dex3_1_Right_JointIndex(IntEnum):
    """Right Dex3-1 hand joint indices (matches DDS message structure).
    
    Note: Right hand has different finger order than left (index before middle).
    """
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandIndex0 = 3
    kRightHandIndex1 = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6


# Joint position limits (radians)
DEX3_LEFT_LOWER_LIMITS = np.array([-1.047, -0.724, 0.0, -1.57, -1.74, -1.57, -1.74], dtype=np.float32)
DEX3_LEFT_UPPER_LIMITS = np.array([1.047, 0.920, 1.74, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
DEX3_RIGHT_LOWER_LIMITS = np.array([-1.047, -0.920, -1.74, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
DEX3_RIGHT_UPPER_LIMITS = np.array([1.047, 0.724, 0.0, 1.57, 1.74, 1.57, 1.74], dtype=np.float32)


# URDF-compatible joint names for LeRobot interface
LEFT_HAND_JOINT_NAMES = [
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
]

# Note: Right hand has different order (thumb, INDEX, middle) to match Dex3_1_Right_JointIndex
RIGHT_HAND_JOINT_NAMES = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]

# Arm joint names (for IK and teleoperation)
LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]

RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

ARM_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
HAND_JOINT_NAMES = LEFT_HAND_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES

