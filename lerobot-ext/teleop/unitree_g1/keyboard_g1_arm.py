import logging
import os as _os
import pygame
from dataclasses import dataclass
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.processor import RobotAction

logger = logging.getLogger(__name__)

# O registro e a classe de config combinam com o nome do arquivo python (keyboard_g1_arm.py)
@TeleoperatorConfig.register_subclass("keyboard_g1_arm")
@dataclass
class KeyboardG1ArmConfig(TeleoperatorConfig):
    speed: float = 0.02  # Sensibilidade
    fps: int = 60
    is_simulation: bool = False

    # ── Yaw do tronco (teclas , e .) ──────────────────────────────────────
    # Mais lento que os braços de propósito: o tronco carrega os dois braços e
    # a cabeça, então um passo grande vira um solavanco que borra as câmeras e
    # estraga o episódio.
    waist_speed: float = 0.01
    # Mesmo limite de curso do lado do robô (UnitreeG1Config.waist_yaw_limit).
    # Duplicado aqui de propósito: o teleop não deve NEM GERAR um alvo que o
    # robô vai recortar, senão o dataset guarda um comando que nunca foi obedecido.
    waist_yaw_limit: float = 1.0

    # ── Modo de controle ──────────────────────────────────────────────────
    #   "joint" → cada tecla move UMA junta. Direto, previsível, mas pensar em
    #             ângulo de ombro para chegar num ponto é pouco natural.
    #   "ik"    → cada mão é uma pose 6D (posição + orientação). Você move o
    #             ponto no espaço e a IK do Pinocchio resolve os 7 ângulos do
    #             braço. É como o VR funciona, então as demos gravadas nos dois
    #             ficam parecidas.
    #
    # A IK trava as juntas de cintura e pernas, então o yaw do tronco continua
    # sendo comandado direto pelas teclas , e . nos dois modos.
    mode: str = "joint"

    # Passos da pose 6D no modo "ik".
    ik_lin_speed: float = 0.005   # metros por frame
    ik_ang_speed: float = 0.02    # radianos por frame

# A classe principal não tem a palavra Config
class KeyboardG1Arm(Teleoperator):
    config_class = KeyboardG1ArmConfig
    name = "keyboard_g1_arm" 

    def __init__(self, config: KeyboardG1ArmConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self.screen = None
        
        # Ajuste para usar a sua pasta externa de robô (removendo os 3 pontos)
        from robot.unitree_g1.g1_utils import G1_29_JointIndex, LEFT_HAND_JOINT_NAMES, RIGHT_HAND_JOINT_NAMES
        
        self._left_hand_names = LEFT_HAND_JOINT_NAMES
        self._right_hand_names = RIGHT_HAND_JOINT_NAMES

        # Dicionários que guardam a posição atual de cada motor
        self.body_joints = {f"{motor.name}.q": 0.0 for motor in G1_29_JointIndex}
        self.hand_joints = {f"{name}.q": 0.0 for name in self._left_hand_names + self._right_hand_names}

        # Variáveis para armazenar o estado de fechamento (de 0.0 a 1.0)
        self.left_grasp_state = 0.0
        self.right_grasp_state = 0.0

        # Vetores Cinemáticos para Fechar as Mãos (Mapeados do ROS2)
        # O valor `1.5` ou `-1.5` representa o ângulo exato para dobrar cada falange.
        # Índices: [Polegar Giro, Polegar Base, Polegar Ponta, Indicador, Médio, Anelar, Mínimo]
        self.LEFT_HAND_CLOSED_TARGETS =  [0.0,  1.5,  1.5, -1.5, -1.5, -1.5, -1.5]
        self.RIGHT_HAND_CLOSED_TARGETS = [0.0, -1.5, -1.5,  1.5,  1.5,  1.5,  1.5]

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            return

        # ── Conflito SDL × GLFW no Wayland ────────────────────────────────
        # Numa sessão Wayland o SDL escolhe x11/XWayland por padrão, enquanto o
        # visualizador do MuJoCo (GLFW) usa Wayland nativo. Os dois disputam EGL
        # e a janela do MuJoCo morre no nascimento:
        #
        #   GLFWError: EGL: Failed to make context current
        #   ERROR: could not create window
        #
        # Forçar o SDL para o mesmo backend do GLFW resolve — os dois passam a
        # usar Wayland e coexistem. Medido: com isto, janela do MuJoCo abre e o
        # render offscreen vai a ~1100 fps na NVIDIA.
        #
        # Só age em sessão Wayland e só se você não tiver definido nada.
        if _os.environ.get("WAYLAND_DISPLAY") and not _os.environ.get("SDL_VIDEODRIVER"):
            _os.environ["SDL_VIDEODRIVER"] = "wayland"
            logger.info("[KeyboardG1Arm] Sessão Wayland: SDL_VIDEODRIVER=wayland (evita conflito com o GLFW do MuJoCo).")

        pygame.init()

        # O Pygame precisa de uma janela (display) para capturar inputs do teclado.
        # Criamos uma janela pequena e damos um nome a ela.
        self.screen = pygame.display.set_mode((420, 300))
        pygame.display.set_caption("Controle G1 - clique aqui para focar")
        self._font = pygame.font.SysFont("monospace", 15)
        self._font_big = pygame.font.SysFont("monospace", 19, bold=True)

        # Realiza a janela ANTES do loop começar.
        # Sem isto o SDL só materializa a superfície no primeiro evento — e o
        # sintoma era o robô não assumir a posição inicial até você clicar em
        # cima da janela. Com o pump+flip aqui, o primeiro get_action() já sai
        # com a janela viva e o comando inicial vai para o robô na hora.
        pygame.event.pump()
        self.screen.fill((18, 18, 22))
        pygame.display.flip()

        if self.config.mode == "ik":
            self._setup_ik()
        
        logger.info("Controle por teclado ativado. Mantenha a janela do Pygame em foco.")
        if self.config.mode == "ik":
            print(
                "\n"
                "┌─ TECLADO G1 · MODO IK (pose 6D por mão) ─────────────────┐\n"
                "│ MÃO ESQ     W S  frente/trás    A D  lados               │\n"
                "│             R F  cima/baixo     LSHIFT+ = GIRA           │\n"
                "│ MÃO DIR     I K  frente/trás    J L  lados               │\n"
                "│             Y H  cima/baixo     RSHIFT+ = GIRA           │\n"
                "│ MÃOS        Z fecha esquerda    M fecha direita          │\n"
                "│ TRONCO      , esquerda          . direita                │\n"
                "└──────────────────────────────────────────────────────────┘\n"
                f"  passo: {self.config.ik_lin_speed} m e {self.config.ik_ang_speed} rad por frame\n"
                "  A IK (Pinocchio) resolve os 7 ângulos de cada braço.\n"
            )
        else:
            print(
                "\n"
                "┌─ TECLADO G1 · MODO JUNTA ────────────────────────────────┐\n"
                "│ BRAÇO ESQ   W A S D  ombro   (LSHIFT = punho)            │\n"
                "│             R F      cotovelo    Q E   rotação           │\n"
                "│ BRAÇO DIR   I J K L  ombro   (RSHIFT = punho)            │\n"
                "│             Y H      cotovelo    U O   rotação           │\n"
                "│ MÃOS        Z esquerda           M direita               │\n"
                "│ TRONCO      , gira p/ esquerda   . gira p/ direita       │\n"
                "└──────────────────────────────────────────────────────────┘\n"
                f"  yaw do tronco: passo {self.config.waist_speed} rad, "
                f"limite ±{self.config.waist_yaw_limit} rad\n"
            )
        print("  (o tronco só se move de fato com use_waist_yaw=True no robô)\n")
        self._is_connected = True

    # ══════════════════════════════════════════════════════════════════════════
    # MODO IK — cada mão é uma pose 6D, o Pinocchio resolve os ângulos
    # ══════════════════════════════════════════════════════════════════════════

    # Ordem em que a IK devolve os 14 ângulos: 7 do braço esquerdo, 7 do direito,
    # na mesma ordem do URDF — que é exatamente a de G1_29_JointArmIndex.
    #
    # DERIVADO DO ENUM, não escrito à mão. O enum tem casing inconsistente
    # (`kLeftWristyaw` com y minúsculo, `kRightWristYaw` com Y maiúsculo), e
    # `teleop/unitree_g1/televuer.py` hardcoda `kLeftWristYaw.q` — uma chave que
    # NÃO existe no robô. Como o `send_action` procura por `f"{motor.name}.q"`,
    # esse punho simplesmente nunca é comandado no teleop por VR. Derivando do
    # enum, o bug não se repete aqui e some sozinho se o enum for corrigido.
    @property
    def ARM_JOINT_NAMES(self):  # noqa: N802
        from robot.unitree_g1.g1_utils import G1_29_JointArmIndex

        return [f"{m.name}.q" for m in G1_29_JointArmIndex]

    def _setup_ik(self) -> None:
        """Carrega o solver e ancora as poses iniciais na cinemática direta."""
        import numpy as np
        import pinocchio as pin
        from teleop.robot_control.robot_arm_ik import G1_29_ArmIK

        logger.info("[KeyboardG1Arm] Carregando IK (Pinocchio)...")
        self._ik = G1_29_ArmIK()

        # Pose inicial = onde as mãos REALMENTE estão com as juntas em zero.
        # Chutar um ponto no espaço faria a IK puxar os braços com tudo no
        # primeiro frame — justamente o solavanco que se quer evitar.
        model = self._ik.reduced_robot.model
        data = self._ik.reduced_robot.data
        q0 = np.zeros(model.nq)
        pin.forwardKinematics(model, data, q0)
        pin.updateFramePlacements(model, data)
        self._pose_l = np.array(data.oMf[self._ik.L_hand_id].homogeneous, dtype=float)
        self._pose_r = np.array(data.oMf[self._ik.R_hand_id].homogeneous, dtype=float)
        self._q_ref = q0.copy()

        logger.info(
            f"[KeyboardG1Arm] IK pronta. Mão ESQ em {self._pose_l[:3, 3].round(3)}, "
            f"DIR em {self._pose_r[:3, 3].round(3)} (metros)."
        )

    @staticmethod
    def _girar(pose, eixo, ang):
        """Rotaciona a pose em torno de um eixo do próprio efetuador."""
        import numpy as np
        import pinocchio as pin

        v = np.zeros(3)
        v[eixo] = ang
        pose[:3, :3] = pose[:3, :3] @ pin.exp3(v)
        return pose

    def _update_ik(self, keys) -> None:
        """Move as poses 6D com o teclado e escreve os ângulos resolvidos."""
        import numpy as np

        lin = self.config.ik_lin_speed
        ang = self.config.ik_ang_speed

        # ── Mão ESQUERDA ──────────────────────────────────────────────────
        #   W/S = frente/trás (x)   A/D = lados (y)   R/F = cima/baixo (z)
        #   com LSHIFT: as mesmas teclas GIRAM em vez de transladar
        lx = (keys[pygame.K_w] - keys[pygame.K_s])
        ly = (keys[pygame.K_a] - keys[pygame.K_d])
        lz = (keys[pygame.K_r] - keys[pygame.K_f])
        if keys[pygame.K_LSHIFT]:
            if lx: self._pose_l = self._girar(self._pose_l, 0, lx * ang)
            if ly: self._pose_l = self._girar(self._pose_l, 1, ly * ang)
            if lz: self._pose_l = self._girar(self._pose_l, 2, lz * ang)
        else:
            self._pose_l[:3, 3] += np.array([lx, ly, lz], dtype=float) * lin

        # ── Mão DIREITA ───────────────────────────────────────────────────
        #   I/K = frente/trás   J/L = lados   Y/H = cima/baixo   (RSHIFT gira)
        rx = (keys[pygame.K_i] - keys[pygame.K_k])
        ry = (keys[pygame.K_j] - keys[pygame.K_l])
        rz = (keys[pygame.K_y] - keys[pygame.K_h])
        if keys[pygame.K_RSHIFT]:
            if rx: self._pose_r = self._girar(self._pose_r, 0, rx * ang)
            if ry: self._pose_r = self._girar(self._pose_r, 1, ry * ang)
            if rz: self._pose_r = self._girar(self._pose_r, 2, rz * ang)
        else:
            self._pose_r[:3, 3] += np.array([rx, ry, rz], dtype=float) * lin

        # ── Resolve ───────────────────────────────────────────────────────
        # `self._q_ref` entra como chute inicial: além de convergir mais rápido,
        # mantém a solução perto da anterior e evita o braço "pular" entre duas
        # configurações igualmente válidas para a mesma pose.
        try:
            sol_q, _ = self._ik.solve_ik(self._pose_l, self._pose_r, self._q_ref)
        except Exception as e:
            # Pose fora do alcance: mantém os ângulos atuais em vez de travar o
            # loop de teleoperação inteiro.
            self._ik_erro = str(e)[:40]
            return

        self._ik_erro = None
        self._q_ref = np.asarray(sol_q, dtype=float)
        for i, nome in enumerate(self.ARM_JOINT_NAMES):
            self.body_joints[nome] = float(sol_q[i])

    def disconnect(self) -> None:
        if self._is_connected:
            pygame.quit()
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
        for key in self.body_joints.keys():
            features[key] = float
        for key in self.hand_joints.keys():
            features[key] = float
        return features

    @property
    def feedback_features(self) -> dict:
        return {}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def get_action(self) -> RobotAction:
        if not self._is_connected:
             raise ConnectionError("Teleoperador por Teclado não está conectado.")
        
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        # ================= MAPEAMENTO DO BRAÇO ESQUERDO =================
        # Analógico Esquerdo (W/A/S/D)
        ls_x = keys[pygame.K_d] - keys[pygame.K_a]
        ls_y = keys[pygame.K_s] - keys[pygame.K_w]
        
        # D-Pad (Cotovelo: R/F | Rotação: Q/E)
        dpad_x = keys[pygame.K_e] - keys[pygame.K_q]
        dpad_y = keys[pygame.K_f] - keys[pygame.K_r]
        
        # Modificador de Pulso Esquerdo (LSHIFT)
        lb = keys[pygame.K_LSHIFT]
        
        # Gatilho Mão Esquerda (Z)
        left_hand_close = keys[pygame.K_z]

        # ================= MAPEAMENTO DO BRAÇO DIREITO =================
        # Analógico Direito (I/J/K/L)
        rs_x = keys[pygame.K_l] - keys[pygame.K_j]
        rs_y = keys[pygame.K_k] - keys[pygame.K_i]
        
        # Botões Y/A/X/B (Cotovelo: U/J | Rotação: O/H -> Adaptado para Y/H e O/U)
        btn_y = keys[pygame.K_y] # Cotovelo -
        btn_a = keys[pygame.K_h] # Cotovelo +
        btn_x = keys[pygame.K_u] # Rotação -
        btn_b = keys[pygame.K_o] # Rotação +
        
        # Modificador de Pulso Direito (RSHIFT)
        rb = keys[pygame.K_RSHIFT]
        
        # Gatilho Mão Direita (M)
        right_hand_close = keys[pygame.K_m]

        # ================= MODO IK (pose 6D por mão) =================
        # No modo "ik" quem escreve os 14 ângulos de braço é o Pinocchio.
        # Zeramos os deltas de junta para o bloco abaixo virar no-op, em vez de
        # duplicar o código de mãos/tronco em dois caminhos.
        if self.config.mode == "ik":
            self._update_ik(keys)
            ls_x = ls_y = dpad_x = dpad_y = 0
            rs_x = rs_y = 0
            btn_y = btn_a = btn_x = btn_b = 0

        # ================= CONTROLE DO BRAÇO ESQUERDO =================
        if lb:
            # Se LSHIFT está pressionado, W/A/S/D controla o Pulso
            self.body_joints["kLeftWristPitch.q"] += ls_y * self.config.speed
            self.body_joints["kLeftWristRoll.q"]  += ls_x * self.config.speed
        else:
            # Padrão: controla o Ombro
            self.body_joints["kLeftShoulderPitch.q"] += ls_y * self.config.speed
            self.body_joints["kLeftShoulderRoll.q"]  += ls_x * self.config.speed
        
        # Cotovelo e Rotação (Yaw)
        self.body_joints["kLeftElbow.q"]       += dpad_y * self.config.speed
        self.body_joints["kLeftShoulderYaw.q"] += dpad_x * self.config.speed

        # ================= CONTROLE DO BRAÇO DIREITO =================
        if rb:
            # Se RSHIFT está pressionado, I/J/K/L controla o Pulso
            self.body_joints["kRightWristPitch.q"] += rs_y * self.config.speed
            self.body_joints["kRightWristRoll.q"]  += rs_x * self.config.speed
        else:
            # Padrão: controla o Ombro
            self.body_joints["kRightShoulderPitch.q"] += rs_y * self.config.speed
            self.body_joints["kRightShoulderRoll.q"]  += rs_x * self.config.speed
        
        # Cotovelo e Rotação (Yaw) Direito
        if btn_y: self.body_joints["kRightElbow.q"] -= self.config.speed
        if btn_a: self.body_joints["kRightElbow.q"] += self.config.speed
        if btn_x: self.body_joints["kRightShoulderYaw.q"] -= self.config.speed
        if btn_b: self.body_joints["kRightShoulderYaw.q"] += self.config.speed

        # ================= YAW DO TRONCO ( , e . ) =================
        # Substitui o analógico direito do VR enquanto ele não está disponível.
        # Integra para POSIÇÃO, não velocidade: o dataset guarda posições de
        # junta e o modelo prevê posições — mapear a tecla direto para velocidade
        # gravaria um alvo que não corresponde ao que foi executado.
        #
        # SINAL: kWaistYaw positivo gira o tronco para a ESQUERDA (convenção do
        # URDF, confirmada no robô real). Por isso a vírgula é que soma — estava
        # invertido e as teclas faziam o oposto do que o mapa acima promete.
        waist_dir = keys[pygame.K_COMMA] - keys[pygame.K_PERIOD]
        if waist_dir:
            alvo = self.body_joints["kWaistYaw.q"] + waist_dir * self.config.waist_speed
            lim = self.config.waist_yaw_limit
            self.body_joints["kWaistYaw.q"] = max(-lim, min(lim, alvo))

        # ================= LÓGICA DAS MÃOS (Dex3) =================
        # Velocidade com que a mão abre e fecha
        hand_speed = self.config.speed * 4  
        
        # Mão Esquerda: Calcula o fator (0.0 até 1.0)
        if left_hand_close:
            self.left_grasp_state = min(self.left_grasp_state + hand_speed, 1.0)
        else:
            self.left_grasp_state = max(self.left_grasp_state - hand_speed, 0.0)
            
        # Mão Direita: Calcula o fator (0.0 até 1.0)
        if right_hand_close:
            self.right_grasp_state = min(self.right_grasp_state + hand_speed, 1.0)
        else:
            self.right_grasp_state = max(self.right_grasp_state - hand_speed, 0.0)

        # Aplica o fator multiplicado pelo vetor cinemático real
        for i, name in enumerate(self._left_hand_names):
            self.hand_joints[f"{name}.q"] = self.left_grasp_state * self.LEFT_HAND_CLOSED_TARGETS[i]

        for i, name in enumerate(self._right_hand_names):
            self.hand_joints[f"{name}.q"] = self.right_grasp_state * self.RIGHT_HAND_CLOSED_TARGETS[i]

        # Concatena e envia
        action_data = {**self.body_joints, **self.hand_joints}

        self._draw_status(keys)

        return action_data

    # ── Retorno visual ────────────────────────────────────────────────────────
    # A janela era branca e não dizia nada. Sem retorno não dá para distinguir
    # "o teclado não captura" de "captura, mas o robô não se move" — e as duas
    # causas exigem consertos completamente diferentes.
    #
    # O item mais importante aqui é o FOCO: no Wayland, se a janela não estiver
    # em foco o SDL não entrega tecla nenhuma, e o sintoma é idêntico ao de um
    # teleop quebrado.
    _WATCHED = None  # preenchido na primeira chamada

    def _draw_status(self, keys) -> None:
        if self.screen is None:
            return
        import pygame as _pg

        if self._WATCHED is None:
            self._WATCHED = [
                ("W", _pg.K_w), ("A", _pg.K_a), ("S", _pg.K_s), ("D", _pg.K_d),
                ("I", _pg.K_i), ("J", _pg.K_j), ("K", _pg.K_k), ("L", _pg.K_l),
                ("Z", _pg.K_z), ("M", _pg.K_m), (",", _pg.K_COMMA), (".", _pg.K_PERIOD),
            ]

        focado = _pg.key.get_focused()
        self.screen.fill((18, 18, 22))

        if focado:
            rotulo = "MODO IK (pose 6D)" if self.config.mode == "ik" else "MODO JUNTA"
            self.screen.blit(self._font_big.render(rotulo, True, (90, 220, 120)), (14, 12))
            if getattr(self, "_ik_erro", None):
                self.screen.blit(
                    self._font.render("IK nao convergiu — fora de alcance", True, (240, 170, 90)),
                    (14, 34),
                )
        else:
            _pg.draw.rect(self.screen, (120, 30, 30), (0, 0, 420, 44))
            self.screen.blit(
                self._font_big.render("SEM FOCO — clique aqui", True, (255, 220, 220)), (14, 12)
            )

        # Teclas seguradas agora
        pressionadas = [nome for nome, cod in self._WATCHED if keys[cod]]
        self.screen.blit(
            self._font.render(f"teclas: {' '.join(pressionadas) if pressionadas else '—'}",
                              True, (230, 230, 230)), (14, 58)
        )

        # O que mostrar depende do modo: em IK o número útil é a POSIÇÃO da mão,
        # não o ângulo do ombro — é nela que você pensa ao mirar um objeto.
        if self.config.mode == "ik" and getattr(self, "_pose_l", None) is not None:
            pl, pr = self._pose_l[:3, 3], self._pose_r[:3, 3]
            linhas = [
                ("tronco  yaw   ", self.body_joints["kWaistYaw.q"], self.config.waist_yaw_limit),
                (f"mão E  x{pl[0]:+.2f} y{pl[1]:+.2f}", pl[2], 1.0),
                (f"mão D  x{pr[0]:+.2f} y{pr[1]:+.2f}", pr[2], 1.0),
            ]
        else:
            linhas = [
                ("tronco  kWaistYaw", self.body_joints["kWaistYaw.q"], self.config.waist_yaw_limit),
                ("ombro E pitch    ", self.body_joints["kLeftShoulderPitch.q"], 3.0),
                ("ombro D pitch    ", self.body_joints["kRightShoulderPitch.q"], 3.0),
            ]
        y = 92
        for rotulo, valor, lim in linhas:
            self.screen.blit(
                self._font.render(f"{rotulo} {valor:+.3f}", True, (200, 200, 210)), (14, y)
            )
            # barrinha proporcional, para ver o movimento sem ler número
            _pg.draw.rect(self.screen, (45, 45, 55), (250, y + 4, 150, 9))
            frac = max(-1.0, min(1.0, valor / lim if lim else 0.0))
            largura = int(abs(frac) * 74)
            x0 = 325 if frac >= 0 else 325 - largura
            _pg.draw.rect(self.screen, (90, 170, 230), (x0, y + 4, max(1, largura), 9))
            _pg.draw.rect(self.screen, (110, 110, 120), (325, y + 2, 1, 13))
            y += 30

        if self.config.mode == "ik":
            rodape = ["WASD/RF move mão E   IKJL/YH move mão D",
                      "SHIFT+ gira   Z/M mãos   , . tronco"]
        else:
            rodape = ["WASD ombro E   IJKL ombro D   SHIFT punho",
                      "RF/YH cotovelo   Z/M mãos   , . tronco"]
        for i, txt in enumerate(rodape):
            self.screen.blit(self._font.render(txt, True, (150, 150, 160)), (14, y + 6 + i * 22))
        _pg.display.flip()