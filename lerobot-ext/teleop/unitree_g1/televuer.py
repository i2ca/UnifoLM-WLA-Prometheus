import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator
from .televuer_utils import (
    TeleVuer,
    safe_mat_update,
    fast_mat_inv,
    safe_rot_update,
    CONST_HEAD_POSE,
    CONST_LEFT_ARM_POSE,
    CONST_RIGHT_ARM_POSE,
    CONST_HAND_ROT,
    T_ROBOT_OPENXR,
    T_OPENXR_ROBOT,
    T_TO_UNITREE_HUMANOID_LEFT_ARM,
    T_TO_UNITREE_HUMANOID_RIGHT_ARM,
    T_TO_UNITREE_HAND,
    R_ROBOT_OPENXR,
    R_OPENXR_ROBOT
)
from lerobot.processor import RobotAction
from lerobot.teleoperators.config import TeleoperatorConfig
from dataclasses import dataclass

# Import do seu solver IK e dos nomes exatos do robô
from .utils.g1_arm_ik import G1_29_ArmIK
from robot.unitree_g1.g1_utils import LEFT_HAND_JOINT_NAMES, RIGHT_HAND_JOINT_NAMES

logger = logging.getLogger(__name__)

# ==================== CONSTANTES DAS JUNTAS DO G1 ====================
# O SEGREDO ESTAVA AQUI: Usar a nomenclatura "kLeft..." do Unitree SDK
ARM_JOINT_NAMES = [
    "kLeftShoulderPitch.q", "kLeftShoulderRoll.q", "kLeftShoulderYaw.q",
    "kLeftElbow.q", "kLeftWristRoll.q", "kLeftWristPitch.q", "kLeftWristYaw.q",
    "kRightShoulderPitch.q", "kRightShoulderRoll.q", "kRightShoulderYaw.q",
    "kRightElbow.q", "kRightWristRoll.q", "kRightWristPitch.q", "kRightWristYaw.q"
]

# Pega os nomes exatos das mãos importados do utilitário do G1
LEFT_HAND_NAMES = [f"{name}.q" for name in LEFT_HAND_JOINT_NAMES]
RIGHT_HAND_NAMES = [f"{name}.q" for name in RIGHT_HAND_JOINT_NAMES]

# Alvos cinemáticos para fechar a mão (igual ao seu teclado)
LEFT_HAND_CLOSED_TARGETS =  [0.0,  1.5,  1.5, -1.5, -1.5, -1.5, -1.5]
RIGHT_HAND_CLOSED_TARGETS = [0.0, -1.5, -1.5,  1.5,  1.5,  1.5,  1.5]
# =====================================================================

@TeleoperatorConfig.register_subclass("televuer")
@dataclass
class VuerTeleopConfig(TeleoperatorConfig):
    binocular: bool = False
    use_hand_tracking: bool = True
    img_shape: tuple = (480, 640, 3) 
    img_shm_name: str | None = None
    left_img_shm_name: str | None = None
    right_img_shm_name: str | None = None
    # certificado autoassinado do VR em lerobot-ext/ (gere com: openssl req -x509 -nodes -days 365
    # -newkey rsa:2048 -keyout key.pem -out cert.pem -subj "/CN=prometheus") — ver wiki Teleop-e-Gravacao
    cert_file: str | None = str(Path(__file__).resolve().parents[2] / "cert.pem")
    key_file: str | None = str(Path(__file__).resolve().parents[2] / "key.pem")
    ngrok: bool = False
    webrtc: bool = False
    webrtc_offer_url: str | None = None

class VuerTeleop(Teleoperator):
    config_class = VuerTeleopConfig
    name = "televuer"

    def __init__(self, config: VuerTeleopConfig):
        super().__init__(config)
        self.config = config
        self.tvuer: TeleVuer | None = None
        self._is_connected = False

        # Inicia o tradutor (IK Solver)
        self.ik_solver = G1_29_ArmIK(visualization=False)
        
        self.last_valid_head_pose = CONST_HEAD_POSE.copy()
        self.last_valid_left_arm_pose = CONST_LEFT_ARM_POSE.copy()
        self.last_valid_right_arm_pose = CONST_RIGHT_ARM_POSE.copy()
        self.last_valid_left_hand_rot = CONST_HAND_ROT.copy()
        self.last_valid_right_hand_rot = CONST_HAND_ROT.copy()

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            return

        self.tvuer = TeleVuer(
            binocular=self.config.binocular,
            use_hand_tracking=self.config.use_hand_tracking,
            img_shape=self.config.img_shape,
            img_shm_name=self.config.img_shm_name,
            left_img_shm_name=self.config.left_img_shm_name,
            right_img_shm_name=self.config.right_img_shm_name,
            cert_file=self.config.cert_file,
            key_file=self.config.key_file,
            ngrok=self.config.ngrok,
            webrtc=self.config.webrtc,
            webrtc_offer_url=self.config.webrtc_offer_url,
        )
        self._is_connected = True
        
        if calibrate and not self.is_calibrated:
            self.calibrate()

    def disconnect(self) -> None:
        if self.tvuer and self.tvuer.process:
            self.tvuer.process.terminate()
            self.tvuer.process.join(timeout=5.0)
            if self.tvuer.process.is_alive():
                self.tvuer.process.kill()
                self.tvuer.process.join(timeout=1.0)
        
        if self.tvuer:
            try:
                if hasattr(self.tvuer, 'img_shm') and self.tvuer.img_shm:
                    self.tvuer.img_shm.close()
            except Exception:
                pass
        
        self.tvuer = None
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass
    
    @property
    def action_features(self) -> dict:
        features = {}
        # Prepara o LeRobot para receber as 14 juntas + 14 dedos
        for name in ARM_JOINT_NAMES + LEFT_HAND_NAMES + RIGHT_HAND_NAMES:
            features[name] = float
        return features

    @property
    def feedback_features(self) -> dict:
        return {}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def get_action(self) -> RobotAction:
        if not self.tvuer:
             raise ConnectionError("Teleoperator is not connected.")
        
        # 1. Pega os dados brutos e acha as matrizes espaciais (XYZ)
        Bxr_world_head, head_pose_is_valid = safe_mat_update(self.last_valid_head_pose, self.tvuer.head_pose)
        if head_pose_is_valid:
             self.last_valid_head_pose = Bxr_world_head
        Brobot_world_head = T_ROBOT_OPENXR @ Bxr_world_head @ T_OPENXR_ROBOT

        # Calcula a Pose do Braço Esquerdo
        left_IPxr_Bxr_world_arm, left_arm_is_valid = safe_mat_update(self.last_valid_left_arm_pose, self.tvuer.left_arm_pose)
        if left_arm_is_valid: self.last_valid_left_arm_pose = left_IPxr_Bxr_world_arm
        left_IPxr_Brobot_world_arm = T_ROBOT_OPENXR @ left_IPxr_Bxr_world_arm @ T_OPENXR_ROBOT
        left_IPunitree_Brobot_world_arm = left_IPxr_Brobot_world_arm @ (T_TO_UNITREE_HUMANOID_LEFT_ARM if left_arm_is_valid else np.eye(4))
        left_IPunitree_Brobot_head_arm = left_IPunitree_Brobot_world_arm.copy()
        left_IPunitree_Brobot_head_arm[0:3, 3] -= Brobot_world_head[0:3, 3]
        left_pose = left_IPunitree_Brobot_head_arm.copy()
        left_pose[0, 3] += 0.15
        left_pose[2, 3] += 0.45

        # Calcula a Pose do Braço Direito
        right_IPxr_Bxr_world_arm, right_arm_is_valid = safe_mat_update(self.last_valid_right_arm_pose, self.tvuer.right_arm_pose)
        if right_arm_is_valid: self.last_valid_right_arm_pose = right_IPxr_Bxr_world_arm
        right_IPxr_Brobot_world_arm = T_ROBOT_OPENXR @ right_IPxr_Bxr_world_arm @ T_OPENXR_ROBOT
        right_IPunitree_Brobot_world_arm = right_IPxr_Brobot_world_arm @ (T_TO_UNITREE_HUMANOID_RIGHT_ARM if right_arm_is_valid else np.eye(4))
        right_IPunitree_Brobot_head_arm = right_IPunitree_Brobot_world_arm.copy()
        right_IPunitree_Brobot_head_arm[0:3, 3] -= Brobot_world_head[0:3, 3]
        right_pose = right_IPunitree_Brobot_head_arm.copy()
        right_pose[0, 3] += 0.15
        right_pose[2, 3] += 0.45

        # =========================================================================
        # CONVERSÃO IK: De matriz XYZ para Ângulos de Motor (.q)
        # =========================================================================
        sol_q, _ = self.ik_solver.solve_ik(left_pose, right_pose)
        
        robot_action = {}

        # 1. Popula os ângulos dos 14 motores do braço com os nomes corretos do SDK
        for i, name in enumerate(ARM_JOINT_NAMES):
            robot_action[name] = float(sol_q[i])

        # 2. Lógica das Mãos (Pinch / Trigger)
        if self.config.use_hand_tracking:
            left_grasp = min(max(self.tvuer.left_hand_pinch_value, 0.0), 1.0)
            right_grasp = min(max(self.tvuer.right_hand_pinch_value, 0.0), 1.0)
        else:
            left_grasp = min(max(self.tvuer.left_controller_trigger_value, 0.0), 1.0)
            right_grasp = min(max(self.tvuer.right_controller_trigger_value, 0.0), 1.0)

        # 3. Multiplica o Grasp (0 a 1) pelos alvos de fechamento dos dedos Dex3
        for i, name in enumerate(LEFT_HAND_NAMES):
            robot_action[name] = float(left_grasp * LEFT_HAND_CLOSED_TARGETS[i])
            
        for i, name in enumerate(RIGHT_HAND_NAMES):
            robot_action[name] = float(right_grasp * RIGHT_HAND_CLOSED_TARGETS[i])

        #print(f"🤖 ENVIADO PRO MUJOCO | Ombro L: {robot_action['kLeftShoulderPitch.q']:.2f}, Cotovelo L: {robot_action['kLeftElbow.q']:.2f}")

        return robot_action