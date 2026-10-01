import zmq
import json
import time
import cv2
import threading
import logging
import numpy as np
from dataclasses import dataclass
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.processor import RobotAction

# Imports do ecossistema Unitree (ajuste os caminhos conforme sua pasta)
from televuer import TeleVuerWrapper
from .utils.sensor_utils import SensorClient, ImageUtils
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType

logger = logging.getLogger(__name__)

# Deslocamento CABEÇA → CINTURA que o televuer já soma no alvo do pulso
# (tv_wrapper.get_tele_data, "coordinate origin offset"). Tem que sair antes de
# girar o alvo e voltar depois: é uma constante do CORPO DO ROBÔ, não do
# operador, então não pode girar junto com a cabeça. Se mudar lá, mude aqui.
HEAD_TO_WAIST_OFFSET = np.array([0.15, 0.0, 0.45])

@TeleoperatorConfig.register_subclass("xr_g1_arm")
@dataclass
class XRG1ArmConfig(TeleoperatorConfig):
    img_server_ip: str = "127.0.0.1"
    #img_server_ip: str = "127.0.0.1"
    is_simulation: bool = True
    input_mode: str = "hand"       # 'hand' ou 'controller'
    display_mode: str = "immersive" # 'immersive', 'ego', 'pass-through'

    # Quadros por segundo ENVIADOS ao headset — não confundir com o FPS do laço
    # de controle, que continua o mesmo. É só a taxa do vídeo que sai por
    # websocket.
    #
    # O laço de render do televuer (main_image_*_zmq*) empurra um JPEG por
    # iteração SEM CONTRAPRESSÃO: `session.upsert` só empilha na fila de uplink,
    # sem olhar se o cliente está dando conta. Se o Quest não drenar na
    # velocidade em que a gente enfileira, quem enche é o buffer de escrita do
    # SSL no servidor e a sessão morre poucos segundos depois de conectar. No
    # log isso aparece como `protocol.resume_writing() failed`, seguido do
    # websocket caindo e de `AssertionError('Websocket session is missing.')`
    # — essa asserção é CONSEQUÊNCIA da queda, não a causa.
    #
    # Baixar para 15 corta o tráfego pela metade e costuma resolver em Wi-Fi
    # ruim, ao custo de vídeo mais picotado no headset.
    display_fps: float = 30.0
    ee_type: str = "dex3"
    zmq: bool = True
    webrtc: bool = False

    # ── Certificado HTTPS do Vuer ─────────────────────────────────────────
    # WebXR só liga em página segura, então o Vuer precisa de cert/chave. Se
    # ficarem em None, procuramos cert.pem/key.pem na raiz do lerobot-ext e só
    # então caímos na busca própria do televuer (env XR_TELEOP_CERT/KEY →
    # ~/.config/xr_teleoperate/ → raiz do pacote).
    cert_file: str | None = None
    key_file: str | None = None

    # Porta do servidor Vuer. 8012 é o default do próprio Vuer; mudar aqui não
    # muda a porta do servidor, só a URL que imprimimos no terminal.
    vuer_port: int = 8012

    # IP DESTE PC que o headset deve procurar. Vazio = o da rota padrão.
    # Não use o IP da rota até o robô: com o robô no cabo (192.168.123.x) essa
    # rota devolve o IP do cabo, e o headset, que está no Wi-Fi, não alcança.
    vuer_host_ip: str = ""

    # ── Dashboard HTTP ────────────────────────────────────────────────────
    # Abra http://<ip-deste-pc>:8080 em qualquer máquina da rede para ver o
    # que está indo para o headset, o estado da teleoperação e um QR code com
    # o link do VR para escanear no Quest.
    dashboard: bool = True
    dashboard_port: int = 8080

    # ── Locomoção pelo analógico ESQUERDO ─────────────────────────────────
    # Y = frente/trás, X = gira o corpo inteiro (pivota andando).
    #
    # O DDS fica TODO no robô: daqui só sai um JSON por ZMQ para a porta 6005
    # da ponte (dex3_g1_server_v2.py), e é lá dentro que o LocoClient é
    # chamado. O robô também tem um watchdog — se estes comandos pararem de
    # chegar por 0.4 s, ele para sozinho.
    #
    # ATENÇÃO — isto NÃO entra no vetor de ação e portanto NÃO é gravado no
    # dataset. É comando de locomoção (velocidade), não posição de junta: o
    # schema continua em 29 dims e os datasets já gravados seguem compatíveis.
    # A consequência é que a política nunca aprende a andar, e pior, ela vê a
    # cena girar/transladar sem nada no vetor de ação explicando o motivo. Se
    # for andar durante uma gravação, ande ENTRE episódios, não no meio de um.
    enable_locomotion: bool = False
    # Porta ZMQ do canal de locomoção no robô (LOCOCMD_PORT do
    # dex3_g1_server_v2.py). Só existe quando o servidor roda com --loco.
    loco_port: int = 6005
    # IP da ponte ZMQ. Vazio = usa o img_server_ip, que na prática é o mesmo
    # robô. Existe separado só para o caso de a ponte e o servidor de imagem
    # estarem em máquinas diferentes.
    loco_ip: str = ""
    # Teto do Unitree (issue #135 do xr_teleoperate). Subir daqui é pedir queda.
    loco_max_speed: float = 0.3   # m/s, frente/trás
    loco_max_yaw: float = 0.3     # rad/s, giro do corpo
    # Zona morta maior que a do tronco: aqui um encosto sem querer faz o robô
    # ANDAR, não só girar a cintura.
    loco_deadzone: float = 0.2

    # ── Painel da câmera da cabeça ────────────────────────────────────────
    # O que pesa na visão é o ÂNGULO que o painel ocupa: size/distance. No modo
    # "ego" o padrão do televuer era 0.75 a 2 m (≈21° de altura); 0.5 a 2 m
    # (≈14°) deixa a janela num tamanho de monitor, com sobra para enxergar o
    # ambiente real em volta e para o painel de pulso não brigar por espaço.
    head_cam_size: float | None = 0.5
    head_cam_distance: float | None = 2.0

    # HUD (travado/livre, gravação, tronco) desenhado sobre a visão da cabeça.
    show_hud: bool = True

    # ── Câmera de pulso dentro do headset ─────────────────────────────────
    # Painel extra que ANCORA NA MÃO do operador: você olha para a sua própria
    # mão e vê o que a câmera do pulso do robô está vendo, em vez de ter mais
    # uma janela fixa disputando espaço com a visão da cabeça.
    # Vem de um servidor ZMQ próprio (right_arm_realsense_server.py, porta
    # 5556), separado do feed da cabeça — se esse servidor não estiver rodando,
    # o painel simplesmente não aparece.
    show_wrist_cam: bool = False
    wrist_img_port: int = 5556
    wrist_cam_name: str = "right_wrist_camera"
    # Mão que o painel segue. A câmera está no pulso DIREITO do robô, então o
    # natural é seguir a mão direita do operador.
    wrist_cam_side: str = "right"
    # Altura do painel em metros e deslocamento em relação à mão, no
    # referencial da cabeça. O painel fica ABAIXO da mão (-y): acima dela ele
    # cai bem em cima da janela da câmera da cabeça, que é centralizada.
    # Pequeno de propósito — é um espelho retrovisor, não uma segunda tela.
    # Clique do analógico direito liga e desliga.
    wrist_cam_size: float = 0.10
    wrist_cam_offset: tuple = (0.0, -0.14, 0.0)

    # ── Yaw do tronco pelo analógico DIREITO (eixo X) ─────────────────────
    # Espelha as teclas , e . do keyboard_g1_arm: mesmo passo, mesmo limite,
    # e igual a lá só se move com o robô DESTRAVADO (botão X do controle
    # esquerdo). Mais lento que os braços de propósito — o tronco carrega os
    # dois braços e a cabeça, então passo grande vira solavanco que borra as
    # câmeras e estraga o episódio.
    waist_speed: float = 0.01
    # Mesmo limite de curso do lado do robô (UnitreeG1Config.waist_yaw_limit).
    # Duplicado aqui de propósito: o teleop não deve NEM GERAR um alvo que o
    # robô vai recortar, senão o dataset guarda um comando que nunca foi obedecido.
    waist_yaw_limit: float = 1.0
    # Zona morta do analógico. Controle de VR raramente descansa em (0,0) e sem
    # isso o tronco fica derivando sozinho o episódio inteiro.
    waist_deadzone: float = 0.15
    # true → inverte o sentido. O sinal de kWaistYaw depende da convenção do
    # URDF; se no headset o tronco girar para o lado contrário do analógico,
    # basta virar esta flag no YAML em vez de mexer no código.
    waist_invert: bool = False

    # ── Âncora de guinada do alvo dos braços ──────────────────────────────
    # O televuer entrega o pulso como (mão − cabeça) com a ORIENTAÇÃO DO
    # MUNDO (tv_wrapper.get_tele_data, "translation adjustment only"). Como a
    # sua mão vive ~40 cm à frente do peito, esse vetor gira junto com você:
    # vire 180° e o alvo, que apontava para a frente do robô, passa a apontar
    # para trás. Ligado, descontamos a guinada da cabeça antes do IK, então
    # "mão à minha frente" vira "mão à frente do robô" para qualquer direção
    # que você esteja olhando.
    head_yaw_lock: bool = True
    # Constante de tempo (s) com que a âncora persegue a cabeça. É o que separa
    # olhada rápida de giro de corpo: a âncora quase não anda enquanto você
    # espia o lado e volta, mas acompanha por inteiro um giro sustentado.
    # Compromisso real, medido (mão a 40 cm do peito):
    #   tau=0.8 → olhada de 0.3 s a 90° arrasta o braço 19 cm; giro de 180°
    #             resolvido em 3 s.
    #   tau=2.0 → mesma olhada arrasta 9 cm; giro de 180° ainda tem 40° de
    #             erro aos 3 s, 9° aos 6 s.
    #   tau=5.0 → olhada arrasta 4 cm, mas o giro de 180° leva ~15 s.
    # Suba se o que te incomoda é o braço andar quando você olha para o lado;
    # desça se o que incomoda é ele demorar a se reorientar quando você gira.
    # 0 = trava rígida na cabeça: todo movimento de pescoço arrasta o braço.
    head_yaw_lock_tau: float = 2.0

    # ── Visão ESTÉREO no headset (30/09) ──────────────────────────────────
    # true → o óculos recebe os DOIS olhos da ZED (cameras_wla_server.py no robô:
    # head_stereo_left / head_stereo_right, 640x480 cada), um em cada olho: visão em 3D.
    # false → uma tela só, com `head_cam_keys[0]` (ou head_camera, se existir).
    stereo_head: bool = False
    head_cam_keys: tuple = ("head_stereo_left", "head_stereo_right")

    # ── TRONCO SEGUE A CABEÇA (30/09) ─────────────────────────────────────
    # true → a guinada (yaw) da SUA cabeça no VR gira o yaw da cintura do robô
    # (kWaistYaw), relativa a para onde você olhava ao destravar. Substitui o tronco
    # pelo analógico direito. A referência dos braços fica FIXA nessa direção inicial
    # e o giro do tronco é descontado do alvo (os braços seguem o tronco, como os seus).
    torso_follows_head: bool = False
    torso_head_gain: float = 1.0      # 1.0 = o tronco gira o mesmo ângulo que a cabeça
    torso_max_speed: float = 0.8      # rad/s — o tronco nunca gira mais rápido que isso
    torso_head_deadzone: float = 0.05  # rad — olhadinhas menores que isso não giram o tronco

    # ── ANALÓGICOS como no teleop ORIGINAL da Unitree (30/09) ─────────────
    # "original": ESQUERDO anda (Y = frente/trás, X = para os lados), DIREITO (X) gira o corpo.
    # "antigo":   ESQUERDO anda (Y) e gira (X); DIREITO (X) gira o tronco.
    loco_style: str = "antigo"
    loco_max_lateral: float = 0.2     # m/s, andar de lado (só no estilo "original")
    # true → o comando de andar vai no vetor de ação (base.vx, base.vy, base.vyaw). Precisa do
    # `gravar_locomocao: true` no robô, senão o dataset não tem essas colunas.
    loco_no_dataset: bool = False
    altura_base: float = 0.74          # base.height gravado (não comandamos altura: valor nominal do WBC)

class XRG1Arm(Teleoperator):
    config_class = XRG1ArmConfig
    name = "xr_g1_arm"

    def __init__(self, config: XRG1ArmConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False

        # Só cai no IP padrão da Unitree se o YAML não tiver informado outro —
        # senão isso pisava em cima de qualquer img_server_ip customizado
        # (ex.: robô numa rede diferente de 192.168.123.x) toda vez que
        # is_simulation=False, que é o que o launcher sempre passa fora do --sim.
        if not self.config.is_simulation and self.config.img_server_ip == "127.0.0.1":
            self.config.img_server_ip = "192.168.123.164"
        
        # Carrega os nomes das juntas do seu robô no LeRobot
        from robot.unitree_g1.g1_utils import G1_29_JointIndex, LEFT_HAND_JOINT_NAMES, RIGHT_HAND_JOINT_NAMES
        self._left_hand_names = LEFT_HAND_JOINT_NAMES
        self._right_hand_names = RIGHT_HAND_JOINT_NAMES
        
        self.body_joints = {f"{motor.name}.q": 0.0 for motor in G1_29_JointIndex}
        self.hand_joints = {f"{name}.q": 0.0 for name in self._left_hand_names + self._right_hand_names}

        # Inicializa o Wrapper do VR e o Solver IK
        self.tv_wrapper = None
        self.arm_ik = None
        self.hand_retargeter = None

        # Estado atual do robô (necessário como "semente" para o cálculo do IK)
        self.current_arm_q = np.zeros(14)
        self.current_arm_dq = np.zeros(14)

        # NOVAS VARIÁVEIS DE SEGURANÇA
        self.vr_started = False
        self.start_time = None
        self.countdown_done = False
        
        self.controller_enabled = False
        self.last_x_state = False
        self.last_y_state = False

        # Painel de pulso: visível por padrão, alternado pelo clique do
        # analógico direito. Espelhado aqui para o HUD saber o estado sem
        # precisar ler o Value compartilhado do outro processo a cada quadro.
        self._wrist_visible = True
        self.last_rstick_click = False

        # Âncora de guinada dos braços. None = re-ancorar no próximo quadro
        # válido (primeiro quadro, ou volta de um destravamento).
        self._yaw_ref = None
        self._yaw_last_t = None

        # Locomoção
        self.loco = None
        self.last_stick_click_pair = False
        self._loco_moving = False
        self._loco_vx = 0.0
        self._loco_vyaw = 0.0

        # Dashboard HTTP
        self.dashboard = None
        self._diag_rede = {}

    def _endpoints_zmq(self) -> dict[str, tuple[str, int]]:
        """Tráfego que DEVE ir pelo cabo: imagens e ponte do robô."""
        eps = {"câmera da cabeça": (self.config.img_server_ip, 5555)}
        if self.config.show_wrist_cam:
            eps["câmera de pulso"] = (self.config.img_server_ip, self.config.wrist_img_port)
        if self.config.enable_locomotion:
            eps["locomoção"] = ((self.config.loco_ip or self.config.img_server_ip), self.config.loco_port)
        return eps

    def _checar_rede(self) -> dict:
        """Confere a divisão Wi-Fi (VR) x cabo (ZMQ) e reclama quando não bate."""
        from .dashboard import diagnostico_rede

        diag = diagnostico_rede(self._endpoints_zmq(), self._vuer_host_ip())

        print("\n   ── REDE ────────────────────────────────────────────────────────")
        print(f"   {'tráfego':<20} {'destino':<22} {'sai por':<16} estado")
        print(f"   {'VR / dashboard':<20} {'(este PC)':<22} {self._vuer_host_ip():<16} —")
        for l in diag["linhas"]:
            estado = "responde" if l["ok"] else "SEM RESPOSTA"
            print(f"   {l['nome']:<20} {l['host'] + ':' + str(l['port']):<22} {l['origem']:<16} {estado}")
        for aviso in diag["avisos"]:
            print(f"\n   ⚠️  {aviso}")
        print()
        return diag

    def _connect_dashboard(self) -> None:
        """Sobe o dashboard HTTP. Falhar aqui não pode derrubar a teleoperação."""
        if not self.config.dashboard:
            return
        try:
            from .dashboard import TeleopDashboard, listar_ips_locais

            self.dashboard = TeleopDashboard(
                port=self.config.dashboard_port,
                vr_url=self._vuer_url(),
                info={
                    "robot_ip": (self.config.loco_ip or self.config.img_server_ip),
                    "img_server_ip": self.config.img_server_ip,
                    "wrist_cam": self.config.show_wrist_cam,
                    "wrist_port": self.config.wrist_img_port,
                    "locomocao": self.config.enable_locomotion,
                    "loco_ip": (self.config.loco_ip or self.config.img_server_ip),
                    "loco_port": self.config.loco_port,
                    "vuer_port": self.config.vuer_port,
                    "vuer_host": self._vuer_host_ip(),
                    "modo_entrada": self.config.input_mode,
                    "modo_display": self.config.display_mode,
                    "tronco_limite": self.config.waist_yaw_limit,
                    "ips": listar_ips_locais(),
                    "aviso": self._aviso_hardware(),
                    "rede": getattr(self, "_diag_rede", {}).get("linhas", []),
                    "avisos_rede": getattr(self, "_diag_rede", {}).get("avisos", []),
                },
            )
            self.dashboard.start()
        except Exception as e:
            self.dashboard = None
            logger.error(f"Dashboard DESATIVADO: {e}")

    def _aviso_hardware(self) -> str:
        """Avisa na tela se alguma junta está com ganho zerado (braço desligado).

        Vale a pena checar de verdade em vez de confiar na memória: gravar uma
        sessão inteira sem perceber que metade do robô não obedece é caro.
        """
        try:
            from robot.unitree_g1.config_unitree_g1 import _GAINS
            mortos = [
                nome for nome, g in _GAINS.items()
                if nome.startswith(("left_arm", "left_wrist", "right_arm", "right_wrist"))
                and not any(g["kp"]) and not any(g["kd"])
            ]
            if mortos:
                return (
                    f"Ganhos zerados em: {', '.join(mortos)}. Essas juntas ficam MOLES e não "
                    "obedecem, mas continuam sendo gravadas no dataset."
                )
        except Exception:
            pass
        return ""

    def _atualiza_dashboard(self) -> None:
        """Espelha no dashboard o mesmo estado que o HUD mostra no headset."""
        if self.dashboard is None:
            return
        import sys
        main_mod = sys.modules.get("__main__")
        restante = 0
        if self.vr_started and not self.countdown_done and self.start_time is not None:
            restante = max(0, 3 - int(time.time() - self.start_time))
        self.dashboard.atualizar_estado(
            destravado=self.controller_enabled,
            gravando=getattr(main_mod, "global_events", None) is not None,
            pausado=bool(getattr(main_mod, "robot_paused", False)),
            andando=self._loco_moving,
            vx=self._loco_vx,
            vyaw=self._loco_vyaw,
            painel_pulso=self._wrist_visible,
            tronco=self.body_joints.get("kWaistYaw.q", 0.0),
            estabilizando=restante,
        )

    def _connect_loco(self) -> None:
        """Abre o socket ZMQ que leva velocidade até a ponte do robô.

        Nenhum DDS deste lado: quem fala com o LocoClient é o
        dex3_g1_server_v2.py, dentro do robô. Aqui é só um PUSH para a porta
        6005 — o mesmo padrão que a ponte já usa para lowcmd e handcmd.

        Falha aqui NUNCA derruba a teleoperação: sem locomoção você ainda tem
        braços, mãos e tronco.
        """
        if not self.config.enable_locomotion:
            # Avisa alto: durante a gravação o analógico esquerdo fica inerte de
            # propósito, e sem esta linha isso é indistinguível de controle
            # quebrado — já custou depuração antes.
            logger.info(
                "Locomoção DESLIGADA por config (enable_locomotion: false) — "
                "analógico esquerdo inerte. Posicione o robô ANTES, com o "
                "config/teleop/teleop_vr_real_loco.yaml."
            )
            return
        try:
            ctx = zmq.Context.instance()
            self.loco = ctx.socket(zmq.PUSH)
            # Sem fila: velocidade velha não interessa. Se a rede engasgar,
            # o certo é descartar e mandar a próxima, não acumular comandos
            # antigos que o robô executaria fora de hora.
            self.loco.setsockopt(zmq.SNDHWM, 1)
            self.loco.setsockopt(zmq.LINGER, 0)
            self.loco.connect(f"tcp://{(self.config.loco_ip or self.config.img_server_ip)}:{self.config.loco_port}")
            logger.info(
                f"Locomoção ATIVA (analógico esquerdo) → "
                f"{(self.config.loco_ip or self.config.img_server_ip)}:{self.config.loco_port} — teto "
                f"{self.config.loco_max_speed} m/s e {self.config.loco_max_yaw} rad/s."
            )
        except Exception as e:
            self.loco = None
            logger.error(f"Locomoção DESATIVADA: falhou ao abrir o socket ({e})")

    def _loco_send(self, msg: dict) -> bool:
        """Manda um comando de locomoção. NOBLOCK: nunca segura o laço de controle."""
        if self.loco is None:
            return False
        try:
            self.loco.send_string(json.dumps(msg), zmq.NOBLOCK)
            return True
        except zmq.Again:
            # Fila cheia — o robô está atrás. O watchdog dele cobre o resto.
            return False
        except Exception as e:
            logger.error(f"Falha ao enviar comando de locomoção: {e}")
            return False

    def _loco_stop(self) -> None:
        """Zera a velocidade. Idempotente e silencioso — é chamado em travamento."""
        if self.loco is None or not self._loco_moving:
            return
        self._loco_send({"vx": 0.0, "vy": 0.0, "vyaw": 0.0})
        self._loco_moving = False
        self._loco_vx = self._loco_vyaw = 0.0
        self._loco_vy = 0.0

    def _update_locomotion(self, tele_data) -> None:
        """Analógico esquerdo → LocoClient.Move(vx, 0, vyaw).

        Y do analógico é para TRÁS quando positivo (ver a convenção em
        TeleData.thumbstickValue), daí os sinais negativos.

        Só anda com o robô destravado: o botão X é o mesmo freio dos braços, e
        um robô que continua andando depois de "travar" seria uma armadilha.
        """
        if self.loco is None or self.config.input_mode != "controller":
            return

        stick = getattr(tele_data, "left_ctrl_thumbstickValue", None)
        if stick is None or len(stick) < 2:
            return

        dz = self.config.loco_deadzone

        def eixo(v: float) -> float:
            v = float(np.clip(v, -1.0, 1.0))
            if abs(v) <= dz:
                return 0.0
            return np.sign(v) * (abs(v) - dz) / max(1e-6, 1.0 - dz)

        vx = -eixo(stick[1]) * self.config.loco_max_speed   # frente/trás
        vy = 0.0
        if self.config.loco_style == "original":
            # teleop original da Unitree: esquerdo X = anda para os lados, DIREITO X = gira
            vy = -eixo(stick[0]) * self.config.loco_max_lateral
            dir_ = getattr(tele_data, "right_ctrl_thumbstickValue", None)
            vyaw = -eixo(dir_[0]) * self.config.loco_max_yaw if dir_ is not None and len(dir_) >= 1 else 0.0
        else:
            vyaw = -eixo(stick[0]) * self.config.loco_max_yaw   # gira o corpo

        self._loco_vx, self._loco_vyaw = vx, vyaw
        self._loco_vy = vy

        if vx == 0.0 and vy == 0.0 and vyaw == 0.0:
            self._loco_stop()
            return

        # Reenviado a cada quadro de propósito: é assim que o watchdog do robô
        # sabe que ainda tem alguém no comando.
        if self._loco_send({"vx": vx, "vy": vy, "vyaw": vyaw}):
            self._loco_moving = True

    def _resolve_certs(self) -> tuple[str | None, str | None]:
        """cert.pem/key.pem para o Vuer servir HTTPS (WebXR não roda em HTTP)."""
        cert, key = self.config.cert_file, self.config.key_file
        if cert and key:
            return cert, key
        from pathlib import Path
        raiz = Path(__file__).resolve().parent.parent  # .../lerobot-ext
        c, k = raiz / "cert.pem", raiz / "key.pem"
        if c.exists() and k.exists():
            return str(c), str(k)
        # Deixa o televuer tentar: env XR_TELEOP_CERT/KEY, ~/.config/xr_teleoperate/, raiz do pacote.
        return cert, key

    def _vuer_host_ip(self) -> str:
        """IP desta máquina que o headset deve procurar.

        É o da ROTA PADRÃO, não o da rota até o robô. Com o robô no cabo
        (192.168.123.x), a rota até ele devolve o IP do cabo — uma rede
        ponto a ponto que o headset, no Wi-Fi, não alcança.
        """
        if self.config.vuer_host_ip:
            return self.config.vuer_host_ip
        from .dashboard import listar_ips_locais
        ips = listar_ips_locais()
        return ips[0] if ips else "127.0.0.1"

    def _vuer_url(self) -> str:
        """URL para abrir no navegador do headset.

        O `?ws=` NÃO é opcional: servido por HTTPS, o cliente do Vuer monta o
        websocket como `wss://<host>` SEM porta, o que cai em 443 e nunca
        conecta. E a linha "Visit: https://vuer.ai?..." que o Vuer imprime é a
        do cliente hospedado, que tentaria falar com o localhost DO HEADSET.
        """
        ip = self._vuer_host_ip()
        porta = self.config.vuer_port
        return f"https://{ip}:{porta}?ws=wss://{ip}:{porta}"

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            return

        logger.info("Iniciando conexão com o Vuer VR...")

        cert_file, key_file = self._resolve_certs()

        # 1. Inicia o servidor WebRTC / WebSocket do Vuer
        self.tv_wrapper = TeleVuerWrapper(
            use_hand_tracking=(self.config.input_mode == "hand"),
            binocular=self.config.stereo_head,
            # estéreo: os dois olhos da ZED lado a lado, 640x480 cada (sem reescalar para 16:9)
            img_shape=(480, 1280) if self.config.stereo_head else (720, 1280),
            display_mode=self.config.display_mode,
            display_fps=self.config.display_fps,
            zmq=self.config.zmq,
            webrtc=self.config.webrtc,
            webrtc_url=f"https://{self.config.img_server_ip}:60000/offer",
            wrist_cam=self.config.show_wrist_cam and self.config.zmq,
            wrist_cam_shape=(224, 224, 3),
            wrist_cam_side=self.config.wrist_cam_side,
            wrist_cam_size=self.config.wrist_cam_size,
            wrist_cam_offset=self.config.wrist_cam_offset,
            head_cam_size=self.config.head_cam_size,
            head_cam_distance=self.config.head_cam_distance,
            cert_file=cert_file,
            key_file=key_file,
        )
        
        # 2. Inicia o Solver de Cinemática Inversa dos braços
        logger.info("Carregando URDF e IK do Braço G1_29...")
        self.arm_ik = G1_29_ArmIK()

        # 3. Inicia o Retargeting Das Mãos
        if self.config.ee_type == "dex3":
            logger.info("Iniciando algoritmo de Retargeting para Dex3...")
            self.hand_retargeter = HandRetargeting(HandType.UNITREE_DEX3)

        self._is_connected = True

        # 4. Mapa da rede, locomoção (opcional) e dashboard HTTP
        self._diag_rede = self._checar_rede()
        self._connect_loco()
        self._connect_dashboard()

        # NOVA PARTE: Usando o SensorClient que já funciona!
        if self.config.zmq:
            logger.info(f"Conectando ao feed de vídeo ZMQ via SensorClient em {self.config.img_server_ip}:5555...")
            self.sensor_client = SensorClient()
            self.sensor_client.start_client(server_ip=self.config.img_server_ip, port=5555)
            
            # Inicia uma thread em background para receber as imagens
            self.video_thread = threading.Thread(target=self._receive_video_feed, daemon=True)
            self.video_thread.start()

            # Feed da câmera de pulso: outro servidor, outra porta, outra thread.
            # Separado de propósito — se a RealSense do pulso não estiver
            # publicando, o feed da cabeça continua intacto.
            if self.config.show_wrist_cam:
                logger.info(
                    f"Conectando ao feed da câmera de pulso em "
                    f"{self.config.img_server_ip}:{self.config.wrist_img_port}..."
                )
                self.wrist_sensor_client = SensorClient()
                self.wrist_sensor_client.start_client(
                    server_ip=self.config.img_server_ip, port=self.config.wrist_img_port
                )
                self.wrist_video_thread = threading.Thread(target=self._receive_wrist_feed, daemon=True)
                self.wrist_video_thread.start()

        url = self._vuer_url()
        painel = (
            f"\n   📊 DASHBOARD (qualquer navegador da rede, com QR code do link acima):\n"
            f"\n      http://{self._vuer_host_ip()}:{self.config.dashboard_port}\n"
            if self.dashboard is not None else ""
        )
        print(
            "\n"
            "   ╔══════════════════════════════════════════════════════════════════╗\n"
            "   ║  ABRA ESTE ENDEREÇO NO NAVEGADOR DO HEADSET:                     ║\n"
            "   ╚══════════════════════════════════════════════════════════════════╝\n"
            f"\n      {url}\n"
            f"{painel}\n"
            "   • IGNORE a linha 'Visit: https://vuer.ai?...' que o Vuer imprime —\n"
            "     aquele é o cliente hospedado e ele procura o servidor no localhost\n"
            "     DO HEADSET, não neste PC.\n"
            "   • O certificado é autoassinado: o navegador vai reclamar. Aceite\n"
            "     ('Avançado' → 'Continuar mesmo assim') ou o WebXR não abre.\n"
            "   • Depois toque em 'Enter VR' na página. O robô só começa a se mexer\n"
            "     quando você destrava no botão X do controle esquerdo.\n"
        )
        logger.info("VR Teleoperator Conectado!")

    def _receive_video_feed(self):
        while self._is_connected:
            try:
                # Recebe a mensagem estruturada usando o client oficial do seu sim
                data = self.sensor_client.receive_message()
                
                if not data:
                    time.sleep(0.005)
                    continue

                imagens = data.get("images") or {}

                # ── ESTÉREO: um olho da ZED em cada olho do headset ──
                if self.config.stereo_head:
                    ke, kd = self.config.head_cam_keys[0], self.config.head_cam_keys[1]
                    if ke in imagens and kd in imagens:
                        olhos = []
                        for k in (ke, kd):
                            im = imagens[k]
                            im = ImageUtils.decode_image(im) if isinstance(im, str) else im
                            im = cv2.resize(im, (640, 480)) if im.shape[:2] != (480, 640) else im.copy()
                            if self.config.show_hud:
                                self._draw_hud(im)
                            olhos.append(im)
                        if self.dashboard is not None:
                            self.dashboard.head.publicar(olhos[0])
                        self.tv_wrapper.render_to_xr(cv2.cvtColor(np.hstack(olhos), cv2.COLOR_BGR2RGB))
                    continue

                # Extrai a imagem do dicionário: head_camera (RealSense antiga) ou, se ela não
                # existir, o olho esquerdo da ZED (cameras_wla_server.py). Antes caía na "primeira
                # chave" do dicionário — que agora pode ser "servidor" (status), não uma imagem.
                img_data = None
                for cam_name in ("head_camera", self.config.head_cam_keys[0]):
                    if cam_name in imagens:
                        img_data = imagens[cam_name]
                        break

                # Se achou a imagem, decodifica usando o ImageUtils
                if img_data is not None:
                    if isinstance(img_data, str):
                        img = ImageUtils.decode_image(img_data)
                    else:
                        img = img_data  # Assume que já é numpy array

                    if img is not None and isinstance(img, np.ndarray):
                        # REDIMENSIONAMENTO DE SEGURANÇA: Garante HD exato pro Vuer
                        img = cv2.resize(img, (1280, 720))

                        # HUD desenhado ANTES da conversão de canal, e com cores
                        # em RGB: esta conversão e a que o televuer faz por dentro
                        # se cancelam, então o que sai daqui é o que aparece no
                        # headset. Desenhar depois deixaria vermelho e azul
                        # trocados só nos elementos do HUD.
                        if self.config.show_hud:
                            self._draw_hud(img)

                        # Dashboard recebe o MESMO quadro do headset, HUD e
                        # tudo — é esse o ponto: ver de fora o que o operador vê.
                        if self.dashboard is not None:
                            self.dashboard.head.publicar(img)

                        # Converte de BGR para RGB
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # Envia para o VR
                        self.tv_wrapper.render_to_xr(img_rgb)

            except Exception as e:
                logger.error(f"Erro no feed de vídeo ZMQ: {e}")
                time.sleep(0.01) # Pausa curta para não floodar o log em caso de erro contínuo

    # ── HUD ───────────────────────────────────────────────────────────────
    # Cores em RGB (ver o comentário no _receive_video_feed).
    _HUD_VERDE    = (60, 220, 120)
    _HUD_VERMELHO = (240, 70, 70)
    _HUD_AMARELO  = (250, 200, 60)
    _HUD_CINZA    = (150, 150, 160)
    _HUD_BRANCO   = (245, 245, 245)

    def _hud_badge(self, img, x, y, texto, cor, preenchido=False):
        """Desenha uma pílula com texto e devolve o x onde a próxima começa."""
        fonte, escala, espessura = cv2.FONT_HERSHEY_DUPLEX, 0.8, 1
        (tw, th), _ = cv2.getTextSize(texto, fonte, escala, espessura)
        pad_x, alt = 16, 46
        larg = tw + 2 * pad_x
        p1, p2 = (x, y), (x + larg, y + alt)

        # Fundo semitransparente: o operador precisa ver a cena ATRÁS do HUD.
        overlay = img.copy()
        cv2.rectangle(overlay, p1, p2, cor if preenchido else (20, 22, 28), -1)
        cv2.addWeighted(overlay, 0.75 if preenchido else 0.55, img, 0.25 if preenchido else 0.45, 0, img)
        cv2.rectangle(img, p1, p2, cor, 2)

        cv2.putText(img, texto, (x + pad_x, y + alt - 15), fonte, escala,
                    (15, 15, 20) if preenchido else cor, espessura, cv2.LINE_AA)
        return x + larg + 10

    def _draw_hud(self, img) -> None:
        """Desenha o estado da teleoperação por cima da visão da cabeça.

        Só afeta o que o operador vê no headset. As imagens que entram no
        dataset vêm das câmeras do robô pelo `get_observation`, num caminho
        totalmente separado deste — então o HUD não contamina a gravação.
        """
        h, w = img.shape[:2]
        x, y = 24, 20

        # 1) Travado / livre — é o que decide se o robô obedece ao VR.
        if self.controller_enabled:
            x = self._hud_badge(img, x, y, "LIVRE", self._HUD_VERDE, preenchido=True)
        else:
            x = self._hud_badge(img, x, y, "TRAVADO  [X]", self._HUD_VERMELHO, preenchido=True)

        # 2) Gravação. `global_events` só existe quando o script de gravação
        #    está rodando; na teleoperação solta não há episódio nenhum.
        import sys
        main_mod = sys.modules.get("__main__")
        gravando = getattr(main_mod, "global_events", None) is not None
        pausado = bool(getattr(main_mod, "robot_paused", False))
        if gravando:
            if pausado:
                x = self._hud_badge(img, x, y, "REC  PAUSA", self._HUD_AMARELO)
            else:
                # Pisca em ~1 Hz para não virar parte do cenário.
                aceso = int(time.time() * 2) % 2 == 0
                x = self._hud_badge(img, x, y, "REC", self._HUD_VERMELHO, preenchido=aceso)
        else:
            x = self._hud_badge(img, x, y, "SEM GRAVACAO", self._HUD_CINZA)

        # 3) Andando: o operador precisa saber que os pés estão se movendo,
        #    porque a janela da cabeça mostra a cena passando de qualquer jeito.
        if self._loco_moving:
            x = self._hud_badge(img, x, y, "ANDANDO", self._HUD_AMARELO, preenchido=True)

        # 4) Painel de pulso ligado/desligado (clique do analógico direito).
        if self.config.show_wrist_cam and not self._wrist_visible:
            x = self._hud_badge(img, x, y, "PULSO OFF", self._HUD_CINZA)

        # 5) Yaw do tronco: número e barrinha de curso, canto superior direito.
        yaw = self.body_joints.get("kWaistYaw.q", 0.0)
        lim = max(1e-6, self.config.waist_yaw_limit)
        bw, bh = 230, 10
        bx, by = w - bw - 24, y + 22
        cv2.putText(img, f"TRONCO {yaw:+.2f} rad", (bx, by - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, self._HUD_BRANCO, 1, cv2.LINE_AA)
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), self._HUD_CINZA, 1)
        cv2.line(img, (bx + bw // 2, by - 3), (bx + bw // 2, by + bh + 3), self._HUD_CINZA, 1)
        px = int(bx + bw / 2 + (yaw / lim) * (bw / 2))
        px = max(bx + 3, min(bx + bw - 3, px))
        cv2.circle(img, (px, by + bh // 2), 7, self._HUD_VERDE, -1)

        # 6) Contagem regressiva de estabilização, no centro.
        if self.vr_started and not self.countdown_done and self.start_time is not None:
            restante = max(0, 3 - int(time.time() - self.start_time))
            texto = f"ESTABILIZANDO {restante}"
            fonte, escala, espessura = cv2.FONT_HERSHEY_DUPLEX, 1.8, 3
            (tw, th), _ = cv2.getTextSize(texto, fonte, escala, espessura)
            cx, cy = (w - tw) // 2, h // 2
            overlay = img.copy()
            cv2.rectangle(overlay, (cx - 26, cy - th - 20), (cx + tw + 26, cy + 22), (20, 22, 28), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            cv2.putText(img, texto, (cx, cy), fonte, escala, self._HUD_AMARELO, espessura, cv2.LINE_AA)

    def _receive_wrist_feed(self):
        """Recebe a câmera de pulso e joga no painel que segue a mão no VR.

        NÃO converte canal, diferente do feed da cabeça: o
        right_arm_realsense_server.py já publica RGB e o `render_wrist_to_xr`
        entrega direto ao encoder JPEG do Vuer, que é PIL e espera RGB. Uma
        conversão a mais aqui deixaria a imagem com vermelho e azul trocados.
        """
        while self._is_connected:
            try:
                data = self.wrist_sensor_client.receive_message()
                if not data:
                    time.sleep(0.005)
                    continue

                images = data.get("images", data)
                img_data = images.get(self.config.wrist_cam_name)
                if img_data is None:
                    # Qualquer câmera que esse servidor esteja publicando serve.
                    keys = [k for k in images.keys() if k not in ("timestamps", "images")]
                    if not keys:
                        continue
                    img_data = images[keys[0]]

                img = ImageUtils.decode_image(img_data) if isinstance(img_data, str) else img_data
                if img is not None and isinstance(img, np.ndarray):
                    self.tv_wrapper.render_wrist_to_xr(img)
                    if self.dashboard is not None:
                        self.dashboard.wrist.publicar(img)

            except Exception as e:
                logger.error(f"Erro no feed da câmera de pulso: {e}")
                time.sleep(0.01)

    def disconnect(self) -> None:
        if self._is_connected:
            # Antes de qualquer outra coisa: parar os pés. Sair da teleoperação
            # com uma velocidade pendente deixaria o robô andando sozinho.
            self._loco_stop()
            self._is_connected = False
            
            # Aguarda a thread de vídeo encerrar
            if hasattr(self, 'video_thread') and self.video_thread.is_alive():
                self.video_thread.join(timeout=1.0)

            if hasattr(self, 'wrist_video_thread') and self.wrist_video_thread.is_alive():
                self.wrist_video_thread.join(timeout=1.0)

            if self.tv_wrapper:
                self.tv_wrapper.close()

            # Fecha o client corretamente
            if hasattr(self, 'sensor_client'):
                self.sensor_client.stop_client()

            if hasattr(self, 'wrist_sensor_client'):
                self.wrist_sensor_client.stop_client()

            if self.dashboard is not None:
                self.dashboard.stop()
                self.dashboard = None

            if self.loco is not None:
                try:
                    self.loco.close()
                except Exception:
                    pass
                self.loco = None

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
        if self.config.loco_no_dataset:
            for k in ("base.vx", "base.vy", "base.vyaw", "base.height"):
                features[k] = float
        return features

    @property
    def feedback_features(self) -> dict:
        return {"q": np.ndarray}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if "q" in feedback:
            self.current_arm_q = feedback["q"][:14] 

    def _trigger_record_event(self, action_type):
        """Injeta comandos diretamente no script de gravação (se ele estiver rodando)"""
        import sys
        import threading
        import time
        import os
        
        # Verifica se estamos rodando dentro do script principal (gravador)
        if "__main__" in sys.modules:
            main_mod = sys.modules["__main__"]
            events = getattr(main_mod, "global_events", None)
            
            if action_type == "save" and events is not None:
                print("\n   🎮 [CONTROLE VR] Ação: SALVANDO e preparando o próximo... ✅")
                events["exit_early"] = True
                def auto_skip():
                    time.sleep(1.0)
                    if events: events["exit_early"] = True
                threading.Thread(target=auto_skip, daemon=True).start()
                
            elif action_type == "discard" and events is not None:
                print("\n   🎮 [CONTROLE VR] Ação: DESCARTANDO lixo e recomeçando... ❌")
                events["rerecord_episode"] = True
                def auto_restart():
                    time.sleep(0.5)
                    if events: events["exit_early"] = True
                    time.sleep(0.5)
                    if events: events["exit_early"] = True
                threading.Thread(target=auto_restart, daemon=True).start()

            elif action_type == "toggle_pause":
                # Inverte o estado local da teleoperação
                self.controller_enabled = not self.controller_enabled
                
                # Se estiver rodando no gravador, sincroniza a variável global dele também
                if hasattr(main_mod, "robot_paused"):
                    main_mod.robot_paused = not self.controller_enabled
                
                estado = "DESTRAVADO ▶️" if self.controller_enabled else "CONGELADO 🧊"
                print(f"\n   🎮 [CONTROLE VR] Ação: Robô {estado}")

            elif action_type == "exit":
                print("\n   🎮 [CONTROLE VR] Ação: ENCERRANDO o sistema... 🛑")
                # Se estiver no gravador, manda o sinal de parada global
                if events is not None:
                    events["stop_recording"] = True
                    events["exit_early"] = True
                else:
                    # Se estiver só testando a teleoperação isolada, força o fechamento
                    self.disconnect()
                    os._exit(0)
        else:
            # Modo teleoperação normal - ignora silenciosamente
            pass

    @staticmethod
    def _head_yaw(head_pose) -> float | None:
        """Guinada da cabeça, em radianos, na convenção do robô (x frente, y esquerda, z cima).

        Positivo = olhando para a esquerda. Retorna None quando a guinada não
        existe: olhando reto para cima ou para baixo o eixo frontal fica
        vertical e o atan2 vira ruído puro. Nesse caso caímos no eixo lateral,
        que continua horizontal — e só desistimos se os dois estiverem de pé.
        """
        R = np.asarray(head_pose)[0:3, 0:3]
        frente, lado = R[:, 0], R[:, 1]
        if np.hypot(frente[0], frente[1]) >= np.hypot(lado[0], lado[1]):
            if np.hypot(frente[0], frente[1]) < 1e-3:
                return None
            return float(np.arctan2(frente[1], frente[0]))
        if np.hypot(lado[0], lado[1]) < 1e-3:
            return None
        return float(np.arctan2(-lado[0], lado[1]))

    def _apply_head_yaw_lock(self, tele_data) -> None:
        """Desconta a guinada da cabeça do alvo dos braços, no lugar (in place).

        Sem isso o alvo é (mão − cabeça) com a orientação do MUNDO: gire o corpo
        e o vetor gira junto, até apontar para trás do robô a 180°. Descontando,
        o que vale é a mão RELATIVA a para onde você está virado.

        A âncora persegue a cabeça com constante de tempo `head_yaw_lock_tau`,
        e é isso que separa os dois casos que parecem o mesmo: olhada rápida de
        pescoço não move a âncora (braço fica quieto), giro de corpo sustentado
        move (braço continua à frente do robô).
        """
        if not self.config.head_yaw_lock:
            return

        yaw = self._head_yaw(tele_data.head_pose)
        agora = time.monotonic()

        if yaw is None:
            # Cabeça na vertical: sem guinada confiável, seguimos com a âncora
            # anterior. Nada de re-ancorar aqui — isso daria um pulo no braço
            # justamente quando o operador olhou para o chão.
            yaw = self._yaw_ref
            if yaw is None:
                self._yaw_last_t = agora
                return

        if self._yaw_ref is None:
            # Primeiro quadro, ou volta de um travamento: ancora onde a cabeça
            # está AGORA, sem suavizar, para não arrastar o braço até a âncora
            # velha.
            self._yaw_ref = yaw
        elif self.config.torso_follows_head:
            # Tronco segue a cabeça: a âncora fica FIXA na direção do destravamento — quem
            # absorve o giro é o tronco (ver abaixo), não a âncora.
            pass
        else:
            tau = float(self.config.head_yaw_lock_tau)
            dt = 0.0 if self._yaw_last_t is None else min(max(agora - self._yaw_last_t, 0.0), 0.5)
            erro = (yaw - self._yaw_ref + np.pi) % (2 * np.pi) - np.pi
            alfa = 1.0 if tau <= 1e-3 else 1.0 - np.exp(-dt / tau)
            self._yaw_ref += alfa * erro

        self._yaw_last_t = agora

        c, s = np.cos(-self._yaw_ref), np.sin(-self._yaw_ref)
        Rz_inv = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        # Tronco girado de θ: o IK dos braços trabalha no referencial do TRONCO (cintura a 0
        # no modelo), então o alvo (no referencial fixo da âncora) é girado de -θ em volta do
        # eixo da cintura. Sem isso o giro do tronco seria aplicado duas vezes nos braços.
        if self.config.torso_follows_head:
            th = float(self.body_joints.get("kWaistYaw.q", 0.0))
            ct, st = np.cos(-th), np.sin(-th)
            Rz_t = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
        else:
            Rz_t = np.eye(3)

        for lado in ("left_wrist_pose", "right_wrist_pose"):
            pose = np.array(getattr(tele_data, lado), dtype=float)
            pose[0:3, 3] = Rz_inv @ (pose[0:3, 3] - HEAD_TO_WAIST_OFFSET) + HEAD_TO_WAIST_OFFSET
            pose[0:3, 0:3] = Rz_inv @ pose[0:3, 0:3]
            pose[0:3, 3] = Rz_t @ pose[0:3, 3]
            pose[0:3, 0:3] = Rz_t @ pose[0:3, 0:3]
            setattr(tele_data, lado, pose)

    def _acao_base(self) -> dict:
        """Comando de andar no vetor de ação (só com loco_no_dataset): o que o analógico mandou neste
        quadro, 0 parado ou travado. É o que diz no dataset se o robô andou."""
        if not self.config.loco_no_dataset:
            return {}
        andando = getattr(self, "_loco_moving", False)
        return {"base.vx": float(self._loco_vx) if andando else 0.0,
                "base.vy": float(getattr(self, "_loco_vy", 0.0)) if andando else 0.0,
                "base.vyaw": float(self._loco_vyaw) if andando else 0.0,
                "base.height": float(self.config.altura_base)}

    def _update_waist_from_head(self, tele_data) -> None:
        """Yaw da CABEÇA (relativo à âncora do destravamento) → alvo de kWaistYaw.

        Posição, não velocidade (o dataset guarda posição de junta). Zona morta para as
        olhadinhas, limite de curso `waist_yaw_limit` e de velocidade `torso_max_speed`.
        Positivo = olhando para a ESQUERDA = kWaistYaw positivo (convenção do URDF).
        """
        agora = time.monotonic()
        dt = 0.0 if getattr(self, "_torso_t", None) is None else min(max(agora - self._torso_t, 0.0), 0.1)
        self._torso_t = agora
        if self._yaw_ref is None:
            return
        yaw = self._head_yaw(tele_data.head_pose)
        if yaw is None:
            return
        d = (yaw - self._yaw_ref + np.pi) % (2 * np.pi) - np.pi
        dz = self.config.torso_head_deadzone
        d = 0.0 if abs(d) <= dz else np.sign(d) * (abs(d) - dz)
        if self.config.waist_invert:
            d = -d
        lim = self.config.waist_yaw_limit
        alvo = float(np.clip(self.config.torso_head_gain * d, -lim, lim))
        atual = float(self.body_joints.get("kWaistYaw.q", 0.0))
        passo = self.config.torso_max_speed * dt
        self.body_joints["kWaistYaw.q"] = float(atual + np.clip(alvo - atual, -passo, passo))

    def _update_waist_yaw(self, tele_data) -> None:
        """Integra o analógico direito (eixo X) na posição-alvo de kWaistYaw.

        Integra para POSIÇÃO, não velocidade — mesma decisão do teclado: o
        dataset guarda posições de junta e o modelo prevê posições, então
        mandar o valor cru do analógico gravaria um alvo que não corresponde
        ao que o robô executou.

        Isto gira A JUNTA do tronco (motor 12) com os pés parados, e não o
        robô inteiro pelo controlador de locomoção — o `Move(vx, vy, omega)`
        ficaria fora do vetor de ação e a política nunca aprenderia.
        """
        if self.config.torso_follows_head:
            self._update_waist_from_head(tele_data)
            return
        # No estilo "original" o analógico direito gira o CORPO (locomoção), não o tronco.
        if self.config.loco_style == "original":
            return

        # Hand tracking não tem analógico — nada a fazer.
        if self.config.input_mode != "controller":
            return

        stick = getattr(tele_data, "right_ctrl_thumbstickValue", None)
        if stick is None or len(stick) < 1:
            return

        x = float(np.clip(stick[0], -1.0, 1.0))
        dz = self.config.waist_deadzone
        if abs(x) <= dz:
            return

        # Reescala a partir da BORDA da zona morta: sem isso o tronco saltaria
        # de parado para dz*waist_speed no instante em que o dedo sai do centro.
        amount = (abs(x) - dz) / max(1e-6, 1.0 - dz)
        # Sinal negativo: kWaistYaw positivo gira o tronco para a ESQUERDA
        # (convenção do URDF, confirmada no robô real). Analógico empurrado para
        # a direita tem que virar passo NEGATIVO, senão o tronco vai para o lado
        # oposto ao do polegar. Mesma correção feita nas teclas , e . do teclado.
        step = -np.sign(x) * amount * self.config.waist_speed
        if self.config.waist_invert:
            step = -step

        lim = self.config.waist_yaw_limit
        self.body_joints["kWaistYaw.q"] = float(
            np.clip(self.body_joints["kWaistYaw.q"] + step, -lim, lim)
        )

    def get_action(self) -> RobotAction:
        if not self._is_connected:
             raise ConnectionError("XR Teleoperator não está conectado.")
        

        # 1. Pega os dados do Headset VR
        tele_data = self.tv_wrapper.get_tele_data()

        # =========================
        # CONTROLE ESQUERDO (X = Pause/Play | Y = Encerrar)
        # =========================
        x_pressed = getattr(tele_data, "left_ctrl_aButton", False) # Botão X físico
        y_pressed = getattr(tele_data, "left_ctrl_bButton", False) # Botão Y físico

        # Detecta clique no X (Pause/Play)
        if x_pressed and not self.last_x_state:
            self._trigger_record_event("toggle_pause")

        # Detecta clique no Y (Sair / Encerrar)
        if y_pressed and not self.last_y_state:
            self._trigger_record_event("exit")

        self.last_x_state = x_pressed
        self.last_y_state = y_pressed

        # =========================
        # CONTROLE DE GRAVAÇÃO (A = Salvar / B = Descartar)
        # =========================
        a_pressed_right = getattr(tele_data, "right_ctrl_aButton", False)
        b_pressed_right = getattr(tele_data, "right_ctrl_bButton", False)

        # Inicializa as variáveis de estado de borda se não existirem
        if not hasattr(self, "last_a_right_state"): self.last_a_right_state = False
        if not hasattr(self, "last_b_right_state"): self.last_b_right_state = False

        # Verifica clique no A (Salvar)
        if a_pressed_right and not self.last_a_right_state:
            self._trigger_record_event("save")

        # Verifica clique no B (Descartar)
        if b_pressed_right and not self.last_b_right_state:
            self._trigger_record_event("discard")

        self.last_a_right_state = a_pressed_right
        self.last_b_right_state = b_pressed_right

        # =========================
        # CLIQUES DOS ANALÓGICOS
        # =========================
        # OS DOIS juntos = damping (parada de emergência macia), o mesmo gesto
        # do teleop_hand_and_arm.py original. Tem prioridade sobre o toggle do
        # painel: no meio de uma emergência ninguém quer trocar de câmera.
        lstick_click = getattr(tele_data, "left_ctrl_thumbstick", False)
        rstick_click = getattr(tele_data, "right_ctrl_thumbstick", False)
        ambos = bool(lstick_click and rstick_click)

        if ambos and not self.last_stick_click_pair:
            self.controller_enabled = False
            self._loco_stop()
            self._loco_send({"damp": True})
            print("\n   🛑 [CONTROLE VR] DAMPING — robô mole e TRAVADO. Destrave no X quando estiver seguro.")
        elif rstick_click and not self.last_rstick_click and self.config.show_wrist_cam:
            # Painel de pulso: fica FORA do bloqueio de propósito — esconder o
            # painel é ajuste de visão, não comando para o robô, e é justamente
            # com o robô travado que dá vontade de limpar a tela.
            self._wrist_visible = not self._wrist_visible
            self.tv_wrapper.set_wrist_cam_visible(self._wrist_visible)
            print(f"\n   🎮 [CONTROLE VR] Painel de pulso: {'LIGADO' if self._wrist_visible else 'DESLIGADO'}")

        self.last_rstick_click = rstick_click
        self.last_stick_click_pair = ambos

        # Espelha o estado no dashboard. Fica ANTES do bloqueio: com o robô
        # travado o dashboard tem que continuar vivo, mostrando justamente que
        # está travado.
        self._atualiza_dashboard()

        # =========================
        # 🚨 BLOQUEIO TOTAL AQUI
        # =========================
        if not self.controller_enabled:
            # Travar o robô tem que parar os PÉS também, senão ele continua
            # andando depois de você achar que congelou tudo.
            self._loco_stop()
            # Solta a âncora: travado é exatamente quando o operador se vira
            # para pegar café. Re-ancorar no destravamento evita o braço correr
            # atrás de uma guinada de minutos atrás.
            self._yaw_ref = None
            return {**self.body_joints, **self.hand_joints, **self._acao_base()}

        # --- LÓGICA DE SEGURANÇA CORRIGIDA ---
        
        # Ignoramos a cabeça (head_pose) pois a biblioteca cria uma "falsa" ao carregar a página.
        # A forma 100% garantida é verificar se as mãos/controles estão sendo rastreados,
        # o que só acontece APÓS você clicar em "Enter VR" no óculos.
        if self.config.input_mode == "hand":
            has_right_hand = np.any(tele_data.right_hand_pos != 0.0)
            has_left_hand = np.any(tele_data.left_hand_pos != 0.0)
            active_session = has_right_hand or has_left_hand
        else:
            active_session = self.controller_enabled

        if not active_session:
            if self.vr_started:
                logger.warning("Rastreamento de mãos não detectado (VR inativo). Mantendo posição.")
                self.vr_started = False
                self.countdown_done = False

            # Mesmo motivo do bloqueio acima: sem rastreamento, a âncora velha
            # não vale nada quando o VR voltar.
            self._yaw_ref = None
            # Retorna a posição atual para manter o robô imóvel
            return {**self.body_joints, **self.hand_joints, **self._acao_base()}

        # Inicia contagem se detectou as mãos no VR
        if active_session and not self.vr_started:
            logger.info(">>> MÃOS DETECTADAS NO VR! AGUARDANDO 3 SEGUNDOS PARA INICIAR...")
            self.vr_started = True
            self.start_time = time.time()

        if self.vr_started and not self.countdown_done:
            elapsed = time.time() - self.start_time
            if elapsed < 3.0:
                if int(elapsed * 10) % 10 == 0: 
                    print(f"--- ESTABILIZANDO ROBÔ: {5 - int(elapsed)}s ---", end='\r')
                return {**self.body_joints, **self.hand_joints, **self._acao_base()}
            else:
                logger.info("\n>>> SISTEMA LIBERADO! MOVIMENTANDO G1...")
                self.countdown_done = True
        
        # --- FIM DA LÓGICA DE SEGURANÇA ---

        # 1.5 Yaw do tronco pelo analógico direito (eixo X).
        # Fica DEPOIS do bloqueio e da contagem regressiva de propósito: com o
        # robô travado ou ainda estabilizando, o tronco tem que ficar onde está,
        # igual aos braços.
        self._update_waist_yaw(tele_data)

        # 1.6 Locomoção pelo analógico esquerdo. Mesmo lugar do tronco, pelo
        # mesmo motivo: travado ou estabilizando, os pés ficam parados.
        self._update_locomotion(tele_data)

        # 1.7 Âncora de guinada. Antes do IK e depois do bloqueio: com o robô
        # travado o alvo nem é lido, então também não faz sentido mover a
        # âncora — ela é re-ancorada no destravamento.
        self._apply_head_yaw_lock(tele_data)

        # 2. Calcula IK dos Braços (retorna 14 ângulos)
        sol_q, _ = self.arm_ik.solve_ik(
            tele_data.left_wrist_pose, 
            tele_data.right_wrist_pose, 
            self.current_arm_q, 
            self.current_arm_dq
        )

        # Mapeia os 14 ângulos para o dicionário do LeRobot
        # Esquerdo (índices 0 a 6)
        self.body_joints["kLeftShoulderPitch.q"] = sol_q[0]
        self.body_joints["kLeftShoulderRoll.q"]  = sol_q[1]
        self.body_joints["kLeftShoulderYaw.q"]   = sol_q[2]
        self.body_joints["kLeftElbow.q"]         = sol_q[3]
        self.body_joints["kLeftWristRoll.q"]     = sol_q[4]
        self.body_joints["kLeftWristPitch.q"]    = sol_q[5]
        self.body_joints["kLeftWristYaw.q"]      = sol_q[6]

        # Direito (índices 7 a 13)
        self.body_joints["kRightShoulderPitch.q"] = sol_q[7]
        self.body_joints["kRightShoulderRoll.q"]  = sol_q[8]
        self.body_joints["kRightShoulderYaw.q"]   = sol_q[9]
        self.body_joints["kRightElbow.q"]         = sol_q[10]
        self.body_joints["kRightWristRoll.q"]     = sol_q[11]
        self.body_joints["kRightWristPitch.q"]    = sol_q[12]
        self.body_joints["kRightWristYaw.q"]      = sol_q[13]

        # 3. Calcula o Retargeting das Mãos (Dedos)
        # 3. Calcula o Retargeting das Mãos (Dedos)
        if self.config.ee_type == "dex3":
            
            # =========================================================
            # MODO 1: RASTREAMENTO PELAS MÃOS (HAND TRACKING)
            # =========================================================
            if self.config.input_mode == "hand":
                # CORREÇÃO 2: Formatação e cálculo correto dos vetores das mãos (Conforme Dex3_1_Controller)
                left_hand_data = tele_data.left_hand_pos.reshape(25, 3)
                right_hand_data = tele_data.right_hand_pos.reshape(25, 3)
                
                # Só calcula se a mão foi detectada no frame atual
                if not np.all(right_hand_data == 0.0) and not np.all(left_hand_data[4] == np.array([-1.13, 0.3, 0.15])):
                    
                    ref_left_value = left_hand_data[self.hand_retargeter.left_indices[1,:]] - left_hand_data[self.hand_retargeter.left_indices[0,:]]
                    ref_right_value = right_hand_data[self.hand_retargeter.right_indices[1,:]] - right_hand_data[self.hand_retargeter.right_indices[0,:]]

                    left_hand_q = self.hand_retargeter.left_retargeting.retarget(ref_left_value)[self.hand_retargeter.left_dex_retargeting_to_hardware]
                    right_hand_q = self.hand_retargeter.right_retargeting.retarget(ref_right_value)[self.hand_retargeter.right_dex_retargeting_to_hardware]

                    # --- AJUSTE DE PINÇA (OFFSET FIXO PARA TOQUE LEVE) ---
                    OFFSET_ESQUEDA = 0.17 
                    OFFSET_DIREITA = 0.0

                    left_hand_q[5] -= OFFSET_ESQUEDA
                    left_hand_q[6] -= OFFSET_ESQUEDA
                    right_hand_q[3] += OFFSET_DIREITA
                    right_hand_q[4] += OFFSET_DIREITA
                    
                    # NOVO: DETECÇÃO REAL DE PUNHO
                    dist_medio_esq = np.linalg.norm(left_hand_data[14] - left_hand_data[0])
                    dist_medio_dir = np.linalg.norm(right_hand_data[14] - right_hand_data[0])
                    
                    punho_esq = np.clip((0.15 - dist_medio_esq) / 0.09, 0.0, 1.0)
                    punho_dir = np.clip((0.15 - dist_medio_dir) / 0.09, 0.0, 1.0)
                    
                    FORCA_PUNHO = 0.8 
                    
                    left_hand_q[3] -= (FORCA_PUNHO * punho_esq)
                    left_hand_q[4] -= (FORCA_PUNHO * punho_esq)
                    left_hand_q[5] -= (FORCA_PUNHO * punho_esq) 
                    left_hand_q[6] -= (FORCA_PUNHO * punho_esq)
                    
                    right_hand_q[5] += (FORCA_PUNHO * punho_dir)
                    right_hand_q[6] += (FORCA_PUNHO * punho_dir)
                    right_hand_q[3] += (FORCA_PUNHO * punho_dir) 
                    right_hand_q[4] += (FORCA_PUNHO * punho_dir)

                    for i, name in enumerate(self._left_hand_names):
                        self.hand_joints[f"{name}.q"] = left_hand_q[i]

                    for i, name in enumerate(self._right_hand_names):
                        self.hand_joints[f"{name}.q"] = right_hand_q[i]

            # =========================================================
            # MODO 2: RASTREAMENTO POR CONTROLES (VR CONTROLLERS)
            # =========================================================
            elif self.config.input_mode == "controller":
                
                # --- HACK DE MEMÓRIA: INJEÇÃO DE IMPEDÂNCIA (KP/KD) ---
                if not hasattr(self, "kp_hacked"):
                    import gc
                    for obj in gc.get_objects():
                        if type(obj).__name__ == "UnitreeG1Dex3":
                            
                            NOVO_KP = 0.3  # Padrão era 0.8 (Trator). 0.3 deixa como Mola.
                            NOVO_KD = 0.1  # Amortecimento suave
                            KP_BASE_POLEGAR = 0.8  # <--- Mantemos forte para conseguir voltar!
                            
                            if hasattr(obj, "_left_hand_msg") and obj._left_hand_msg is not None:
                                for i in range(7):
                                    # Aplica força total apenas no motor 0 (base do polegar)
                                    obj._left_hand_msg.motor_cmd[i].kp = KP_BASE_POLEGAR if i == 0 else NOVO_KP
                                    obj._left_hand_msg.motor_cmd[i].kd = NOVO_KD
                                    
                            if hasattr(obj, "_right_hand_msg") and obj._right_hand_msg is not None:
                                for i in range(7):
                                    obj._right_hand_msg.motor_cmd[i].kp = KP_BASE_POLEGAR if i == 0 else NOVO_KP
                                    obj._right_hand_msg.motor_cmd[i].kd = NOVO_KD
                                    
                            self.kp_hacked = True
                            print(f"\n   🪽 [HACK] Kp ajustado! Dedos em {NOVO_KP}, mas base do polegar em {KP_BASE_POLEGAR} para conseguir retornar.")
                            break

                # --- LEITURA DOS GATILHOS COM DEADZONE ---
                left_trigger = np.clip((10.0 - tele_data.left_ctrl_triggerValue) / 10.0, 0.0, 1.0)
                right_trigger = np.clip((10.0 - tele_data.right_ctrl_triggerValue) / 10.0, 0.0, 1.0)

                # FORÇA O RETORNO AO ZERO: Se o gatilho for solto (mesmo com folga no controle), zera o valor.
                if left_trigger < 0.05: left_trigger = 0.0
                if right_trigger < 0.05: right_trigger = 0.0

                left_squeeze = np.clip(tele_data.left_ctrl_squeezeValue, 0.0, 1.0)
                right_squeeze = np.clip(tele_data.right_ctrl_squeezeValue, 0.0, 1.0)

                # =========================
                # LÓGICA DE MOVIMENTO
                # =========================
                left_hand_q = np.zeros(7)
                right_hand_q = np.zeros(7)

                LEFT_TARGET = np.array([0.0,  1.5,  1.5, -1.5, -1.5, -1.5, -1.5])
                RIGHT_TARGET = np.array([0.0, -1.5, -1.5,  1.5,  1.5,  1.5,  1.5])

                # Grip completo
                left_hand_q  = left_squeeze  * LEFT_TARGET
                right_hand_q = right_squeeze * RIGHT_TARGET

                # =========================
                # PINÇA E ROTAÇÃO (AJUSTE FINO)
                # =========================
                PINCH_FORCE = 2.0
                PINCH_OFFSET = 0.2
                PINCH_OFFSET2 = 0.1

                LEFT_INDEX_ID  = 5   
                RIGHT_INDEX_ID = 5
   

                # Aplica movimento do indicador
                left_hand_q[5]   += -PINCH_FORCE * left_trigger
                right_hand_q[5] +=  PINCH_FORCE * right_trigger

                # Offset fixo
                left_hand_q[5]   += -PINCH_OFFSET * left_trigger
                right_hand_q[5] +=  PINCH_OFFSET * right_trigger

                # Aplica movimento do indicador
                left_hand_q[6]   += -PINCH_FORCE * left_trigger
                right_hand_q[6] +=  PINCH_FORCE * right_trigger

                # Offset fixo
                left_hand_q[6]   += -PINCH_OFFSET2 * left_trigger
                right_hand_q[6] +=  PINCH_OFFSET2 * right_trigger    

                # ROTAÇÃO DO POLEGAR
                # Nota: Inverti o sinal da mão direita para +0.5, pois mãos costumam ser espelhadas.
                # Se a mão direita passar a girar para o lado errado, pode voltar para -0.5.
                left_hand_q[0]  += -0.5 * left_trigger
                right_hand_q[0] +=  -0.5 * right_trigger 

                # CURVATURA EXTRA
                #left_hand_q[5]   += -0.5 * left_trigger
                #right_hand_q[5] +=  -0.5 * right_trigger 

                # Polegar acompanha pinça
                left_hand_q[1] += 0.8 * left_trigger
                left_hand_q[2] += 0.8 * left_trigger

                right_hand_q[1] -= 0.8 * right_trigger 
                right_hand_q[2] -= 0.8 * right_trigger

                # =========================
                # APLICAÇÃO FINAL
                # =========================
                for i, name in enumerate(self._left_hand_names):
                    self.hand_joints[f"{name}.q"] = left_hand_q[i]

                for i, name in enumerate(self._right_hand_names):
                    self.hand_joints[f"{name}.q"] = right_hand_q[i]

        # Concatena os dicionários e retorna a Ação Final
        action_data = {**self.body_joints, **self.hand_joints, **self._acao_base()}
        return action_data