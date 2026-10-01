"""
G1 Arm Inverse Kinematics using Pinocchio and CasADi.
Ported from prometheus/src/xr_teleoperate/teleop/robot_control/robot_arm_ik.py

This module provides optimization-based IK solving for the G1 humanoid robot's
14-DOF dual arm system (7 joints per arm).
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import casadi
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin

from .weighted_moving_filter import WeightedMovingFilter

logger = logging.getLogger(__name__)

# Asset paths relative to this file
ASSETS_DIR = Path(__file__).parent.parent / "assets"


class G1_29_ArmIK:
    """
    Inverse Kinematics solver for G1 robot with 29-DOF body + 14-DOF hands.
    
    Uses Pinocchio for robot kinematics and CasADi/IPOPT for optimization.
    Solves for 14 arm joint positions given target wrist poses.
    """
    
    def __init__(self, visualization: bool = False, travar_punho_dir: bool = False,
                 postura_ref: Optional[np.ndarray] = None):
        """
        Initialize the IK solver.

        Args:
            visualization: Enable Meshcat visualization (optional)
            travar_punho_dir: prende as três juntas do punho DIREITO
                (`right_wrist_roll/pitch/yaw`, índices 11-13 do vetor reduzido)
                nos valores passados a cada `solve_ik`. Padrão False — o teleop
                e o robô real continuam com o punho livre, como sempre.

                Para que serve: com o punho livre o solver o gira à vontade
                para atingir a pose, e na tela a mão fica rodando durante o
                movimento. Travado, a posição é resolvida só com as quatro
                juntas de ombro/cotovelo e a mão mantém a orientação.

                O custo é direto: sobram 4 graus de liberdade para 3 de
                posição mais a orientação desejada, então poses que antes eram
                alcançáveis deixam de ser. Medir o aproveitamento depois de
                ligar isto não é opcional.
            postura_ref: postura (14 juntas) em torno da qual REGULARIZAR.
                Padrão None = zeros, que é o comportamento de sempre.

                Isto não é detalhe de ajuste fino. O termo de regularização
                era `sumsqr(var_q)`, ou seja, puxava toda junta para ZERO — e
                cotovelo em zero é BRAÇO ESTICADO. Era essa a razão de o
                cotovelo sair reto e o movimento ficar com cara de alcance
                forçado, não a busca no espaço nulo. Regularizando em torno de
                uma postura com o cotovelo dobrado, o solver passa a preferir
                a mesma família de poses que um braço humano escolhe.
        """
        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        self.visualization = visualization

        # Load G1 URDF
        urdf_path = ASSETS_DIR / "g1" / "g1_body29_hand14.urdf"
        mesh_dir = ASSETS_DIR / "g1"
        
        if not urdf_path.exists():
            raise FileNotFoundError(
                f"G1 URDF not found at {urdf_path}. "
                "Ensure assets are copied from prometheus/src/xr_teleoperate/assets/"
            )
        
        self.robot = pin.RobotWrapper.BuildFromURDF(
            str(urdf_path), str(mesh_dir)
        )

        # Joints to lock (legs, waist, fingers)
        self.joints_to_lock = [
            # Legs
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            # Waist
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            # Left hand fingers
            "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint", "left_hand_middle_1_joint",
            "left_hand_index_0_joint", "left_hand_index_1_joint",
            # Right hand fingers
            "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
            "right_hand_index_0_joint", "right_hand_index_1_joint",
            "right_hand_middle_0_joint", "right_hand_middle_1_joint",
        ]

        # Build reduced robot (only arm joints)
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.joints_to_lock,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        # Add end-effector frames
        self.reduced_robot.model.addFrame(
            pin.Frame(
                'L_ee',
                self.reduced_robot.model.getJointId('left_wrist_yaw_joint'),
                pin.SE3(np.eye(3), np.array([0.05, 0, 0])),
                pin.FrameType.OP_FRAME
            )
        )
        self.reduced_robot.model.addFrame(
            pin.Frame(
                'R_ee',
                self.reduced_robot.model.getJointId('right_wrist_yaw_joint'),
                pin.SE3(np.eye(3), np.array([0.05, 0, 0])),
                pin.FrameType.OP_FRAME
            )
        )

        # Create CasADi model
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get end-effector frame IDs
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")

        # Define error functions
        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3]
                )
            ],
        )
        
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T)
                )
            ],
        )

        # Setup optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)

        # ── Trava do punho direito ─────────────────────────────────────
        # Restrição de IGUALDADE, e não peso na função de custo: peso alto
        # ainda deixa o solver ceder quando o alvo fica difícil, e o punho
        # voltaria a girar justamente nos casos em que se quer que ele não
        # gire. Como parâmetro, o valor pode mudar entre chamadas sem
        # reconstruir o problema.
        self.travar_punho_dir = travar_punho_dir
        if travar_punho_dir:
            self.par_punho_dir = self.opti.parameter(3)
            self.opti.subject_to(self.var_q[11:14] == self.par_punho_dir)

        # Cost terms
        self.translational_cost = casadi.sumsqr(
            self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        self.rotation_cost = casadi.sumsqr(
            self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        # Regularização em torno de `postura_ref` (zeros por padrão). Ver o
        # argumento no docstring: com zeros, o cotovelo é empurrado para reto.
        self.postura_ref = (np.zeros(self.reduced_robot.model.nq)
                            if postura_ref is None
                            else np.asarray(postura_ref, dtype=float))
        self.regularization_cost = casadi.sumsqr(self.var_q - self.postura_ref)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Joint limits
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit
        ))

        # Minimize combined cost
        self.opti.minimize(
            50 * self.translational_cost + 
            self.rotation_cost + 
            0.02 * self.regularization_cost + 
            0.1 * self.smooth_cost
        )

        # Solver options
        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 50,
                'tol': 1e-6
            },
            'print_time': False,
            'calc_lam_p': False
        }
        self.opti.solver("ipopt", opts)

        # State
        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(
            np.array([0.4, 0.3, 0.2, 0.1]), 
            self.reduced_robot.model.nq
        )
        self.vis = None

        logger.info(f"G1_29_ArmIK initialized with {self.reduced_robot.model.nq} DOF")

    def scale_arms(
        self, 
        human_left_pose: np.ndarray, 
        human_right_pose: np.ndarray,
        human_arm_length: float = 0.60,
        robot_arm_length: float = 0.75
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale wrist poses from human arm length to robot arm length.
        
        Args:
            human_left_pose: 4x4 left wrist pose from human
            human_right_pose: 4x4 right wrist pose from human
            human_arm_length: Human arm length in meters
            robot_arm_length: Robot arm length in meters
            
        Returns:
            Tuple of scaled (left_pose, right_pose)
        """
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def solve_ik(
        self,
        left_wrist: np.ndarray,
        right_wrist: np.ndarray,
        current_arm_q: Optional[np.ndarray] = None,
        current_arm_dq: Optional[np.ndarray] = None,
        punho_dir: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve IK for given target wrist poses.
        
        Args:
            left_wrist: 4x4 homogeneous transformation for left wrist
            right_wrist: 4x4 homogeneous transformation for right wrist
            current_arm_q: Current arm joint positions (optional, for warm start)
            current_arm_dq: Current arm joint velocities (optional)
            
        Returns:
            Tuple of (joint_positions, feedforward_torques)
        """
        if current_arm_q is not None:
            self.init_data = current_arm_q
        self.opti.set_initial(self.var_q, self.init_data)

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data)

        # Valor em que as juntas do punho direito ficam presas nesta chamada.
        # Sem `punho_dir`, usa o que o `init_data` já traz — o que congela o
        # punho onde ele estava, em vez de puxá-lo para um valor arbitrário.
        if self.travar_punho_dir:
            alvo_punho = (self.init_data[11:14] if punho_dir is None
                          else np.asarray(punho_dir, dtype=float)[:3])
            self.opti.set_value(self.par_punho_dir, alvo_punho)

        try:
            sol = self.opti.solve()
            sol_q = self.opti.value(self.var_q)
            
            # Apply smoothing filter
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_arm_dq is not None:
                v = current_arm_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            # Compute feedforward torques
            sol_tauff = pin.rnea(
                self.reduced_robot.model, 
                self.reduced_robot.data, 
                sol_q, v, 
                np.zeros(self.reduced_robot.model.nv)
            )

            return sol_q, sol_tauff

        except Exception as e:
            logger.error(f"IK convergence error: {e}")

            # Return debug solution
            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_arm_dq is not None:
                v = current_arm_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(
                self.reduced_robot.model, 
                self.reduced_robot.data, 
                sol_q, v, 
                np.zeros(self.reduced_robot.model.nv)
            )

            logger.error(
                f"sol_q: {sol_q}\n"
                f"motorstate: {current_arm_q}\n"
                f"left_pose:\n{left_wrist}\n"
                f"right_pose:\n{right_wrist}"
            )

            # Fall back to current position
            if current_arm_q is not None:
                return current_arm_q, np.zeros(self.reduced_robot.model.nv)
            return sol_q, sol_tauff

    def reset(self) -> None:
        """Reset the IK solver state."""
        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter.reset()
