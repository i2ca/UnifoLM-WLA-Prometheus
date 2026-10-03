#!/usr/bin/env python
"""
UnifoLM-WLA-1.0 NO G1 REAL — BRAÇOS + MÃOS DEX3, CLIENTE DO SERVIDOR OFICIAL, COM O COGUMELO DE PÂNICO
========================================================================================
Câmera da cabeça + juntas reais -> WLA-1.0 (VLM -> DiT, na PGX, em processo) -> pose das duas
"garras" (xyz + rpy, pelvis) -> IK (fk_g1.py) -> juntas dos BRAÇOS pela ponte v3.

Segurança, em camadas:
  0. POSE INICIAL: antes da rede, os braços vão DEVAGAR (--tempo-pose s) até a pose de partida
     do episódio 0 do dataset Dex1 do WLA (mãos à frente, na altura da mesa), resolvida por IK;
  1. BRAÇOS e MÃOS: a garra do WLA (Dex1, 0-5,5) vira abrir/fechar as Dex3 (a ponte limita os
     dedos a 1 rad/s e kp 1; --sem-maos desliga). Pernas e base do modelo são IGNORADAS; da cintura,
     só o YAW do modelo é seguido (--cintura yaw, até ±--cintura-yaw-max rad; roll/pitch seguros onde estão,
     como na teleoperação, que só comandava o yaw); --cintura parada segura tudo;
  2. ESCALA: `--escala` (padrão 1,0 = 100%) do deslocamento que o modelo pede (posição e rotação);
  3. CAIXA: cada junta fica a no máximo `--caixa` rad (padrão 0,6 ≈ 34°) da pose de partida;
  4. a ponte v3 ainda limita 0,3 rad/s e corta o kp (20 ombro/cotovelo, 10 punho);
     o COGUMELO congela tudo na hora (LED vermelho + bipe);
  5. pânico por software se o estado parar de chegar, se o botão "parar" da cabine for
     apertado, em qualquer erro, e no fim.

Painel: a cabine http://<pgx>:8090 (câmera, trajetória prevista x executada, botão parar).

    cd ~/DEV/unifolm-wla && ~/miniforge3/envs/wla/bin/python \\
        ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/roda_wla_real.py --ensaio      # não envia nada
    cd ~/DEV/unifolm-wla && ~/miniforge3/envs/wla/bin/python \\
        ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/roda_wla_real.py --segundos 30
"""
import argparse
import csv
import json
import re
import threading
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import zmq
from scipy.spatial.transform import Rotation

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "robo_g1"))   # teste_cotovelo (ponte/arm_sdk); no prometheus-vla era groot_n17/
import sombra_wla as sw  # noqa: E402  (carrega o servidor oficial do WLA e a cabine)
from fk_g1 import BRACO, FK, expvec, rotvec  # noqa: E402
from teste_cotovelo import BRACOS, CINTURA, Estado, modo_da_ponte, monta_cmd  # noqa: E402

# Pose de partida em JUNTAS: mediana do 1º quadro dos 490 episódios de G1_Dex1_Put_Fruit_On_Plate
# (observation.state.left_arm/right_arm). Uma IK até a pose da mão do dataset, partindo dos braços
# caídos, caía num ramo torcido (ombro no limite, 2,6 rad de diferença) — por isso junta, não IK.
def _temp(m):
    t = m.get("temperature", 0.0)
    return float(max(t) if isinstance(t, (list, tuple)) else t)


POSE_JUNTAS = json.load(open(AQUI / "pose_partida_dex1.json"))


# Dex3, ordem do SDK. Esq: polegar 0-2, médio 0-1, indicador 0-1; dir: polegar 0-2, indicador 0-1, médio 0-1.
# FECHADA = 80% da mão em repouso medida no robô (que é a mão toda fechada); ABERTA = 0.
MAO_FECHADA = {"left": 0.8 * np.array([0.0, 0.95, 1.57, -1.56, -1.71, -1.52, -1.78]),
               "right": 0.8 * np.array([0.0, -1.0, -1.65, 1.54, 1.72, 1.54, 1.69])}
TOPICO_MAO = {"left": "rt/dex3/left/cmd", "right": "rt/dex3/right/cmd"}


def cmd_mao(lado, garra_wla, kp=1.0, kd=0.2):
    """Garra Dex1 do WLA (0 fechada ... 5,5 aberta) -> as 7 juntas da Dex3, abrindo/fechando junto."""
    c = float(np.clip(1.0 - garra_wla / 5.5, 0.0, 1.0))
    q = c * MAO_FECHADA[lado]
    mc = [{"mode": (k & 0x0F) | (0x01 << 4), "q": float(q[k]), "dq": 0.0, "kp": kp, "kd": kd, "tau": 0.0}
          for k in range(7)]
    return {"topic": TOPICO_MAO[lado], "data": {"motor_cmd": mc}}, c


# ── Dex3 como fig6d (modelo afinado com o nosso dataset; mesma conta do converte_dataset_dex3_wla.py) ──
# ordem fig6d: [polegar1, polegar2, indicador0, indicador1, médio0, médio1], 0 aberto .. 1 no limite.
# A rotação do polegar (polegar0) não está no fig6d: nos dados gravados ela foi sempre 0.
FECHA_DEX3 = {"left": np.array([0.0, 0.92, 1.74, -1.57, -1.74, -1.57, -1.74]),
              "right": np.array([0.0, -0.92, -1.74, 1.57, 1.74, 1.57, 1.74])}
ORDEM_FIG6D = {"left": [1, 2, 5, 6, 3, 4], "right": [1, 2, 3, 4, 5, 6]}


def dex3_para_fig6d(q7, lado):
    idx = ORDEM_FIG6D[lado]
    return np.clip(np.asarray(q7, float)[idx] / FECHA_DEX3[lado][idx], 0.0, 1.0).astype(np.float32)


def fig6d_para_dex3(f6, lado):
    q = np.zeros(7)
    q[ORDEM_FIG6D[lado]] = np.clip(np.asarray(f6, float), 0.0, 1.0) * FECHA_DEX3[lado][ORDEM_FIG6D[lado]]
    return q


def cmd_mao_q(lado, q7, kp=1.0, kd=0.2):
    mc = [{"mode": (k & 0x0F) | (0x01 << 4), "q": float(q7[k]), "dq": 0.0, "kp": kp, "kd": kd, "tau": 0.0}
          for k in range(7)]
    return {"topic": TOPICO_MAO[lado], "data": {"motor_cmd": mc}}


class ClienteWLA:
    """Cliente do SERVIDOR OFICIAL do WLA (model_server/action_server_wbc_msgpack_unitree.py,
    websocket + msgpack), que fica carregado na PGX (sobe_wla_servidor.sh). Mesma observação e
    mesma ação do servidor em processo — só que o executor sobe na hora."""

    def __init__(self, url):
        from websockets.sync.client import connect
        from tools import msgpack_numpy   # model_server/tools (já no sys.path pelo sombra_wla)
        self.mp = msgpack_numpy
        self.packer = msgpack_numpy.Packer()
        self.ws = connect(url, max_size=None, compression=None, open_timeout=10)
        self.meta = msgpack_numpy.unpackb(self.ws.recv())

    def acao(self, obs):
        t0 = time.perf_counter()
        self.ws.send(self.packer.pack({"type": "get_action", "obs": obs}))
        resp = self.ws.recv()
        ms = (time.perf_counter() - t0) * 1e3
        if isinstance(resp, str):
            raise RuntimeError(f"servidor do WLA: {resp[:500]}")
        return None, {k: np.asarray(v) for k, v in self.mp.unpackb(resp).items()}, ms, None, None


class Falador:
    """O robô fala o que vai fazer. Manda o texto ao painel do robô (:8095/fala), que toca no alto-falante
    do G1: voz interna do G1 (TtsMaker) ou, com --voz piper, a voz dos alertas (Piper en_US-lessac,
    gerada aqui na PGX). Não repete a mesma frase e nunca atrasa o controle (thread + fila)."""

    PIPER = Path.home() / "wla_testes/tts"

    def __init__(self, robo, voz):
        import queue
        import threading
        self.url = f"http://{robo}:8095/fala"
        self.voz, self.ultima, self.cache = voz, None, {}
        self.ao_falar = None   # (texto, origem) -> histórico da cabine
        self.fila = queue.Queue(maxsize=2)
        threading.Thread(target=self._laco, daemon=True, name="fala").start()

    def diz(self, texto, origem="executor"):
        if texto:
            self.ultima = texto
            try:
                self.fila.put_nowait((texto, origem))
            except Exception:
                pass

    def _pcm(self, texto):
        import subprocess
        import tempfile
        if texto not in self.cache:
            with tempfile.TemporaryDirectory() as d:
                subprocess.run([str(self.PIPER / ".venv/bin/piper"), "-m", str(self.PIPER / "en_US-lessac-medium.onnx"),
                                "-f", f"{d}/a.wav"], input=texto.encode(), check=True, capture_output=True)
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", f"{d}/a.wav", "-ar", "16000", "-ac", "1",
                                "-f", "s16le", f"{d}/a.pcm"], check=True)
                self.cache[texto] = open(f"{d}/a.pcm", "rb").read()
        return self.cache[texto]

    def _laco(self):
        import urllib.parse
        import urllib.request
        while True:
            texto, origem = self.fila.get()
            try:
                corpo = self._pcm(texto) if self.voz == "piper" else b""
                req = urllib.request.Request(f"{self.url}?texto={urllib.parse.quote(texto)}", data=corpo, method="POST")
                urllib.request.urlopen(req, timeout=3).read()
                print(f"   🗣  {texto}", flush=True)
                if self.ao_falar is not None:
                    self.ao_falar(texto, origem)
            except Exception as e:  # noqa: BLE001
                print(f"   (fala falhou: {e})", flush=True)


def R_de(rpy):
    return Rotation.from_euler("xyz", rpy).as_matrix()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robo", default="192.168.123.164")
    ap.add_argument("--porta", type=int, default=8090, help="porta da cabine")
    ap.add_argument("--segundos", type=float, default=30, help="0 = sem limite (modo cabine: termina com Ctrl+C ou parar)")
    ap.add_argument("--escala", type=float, default=1.0, help="fração do movimento proposto que é executada")
    ap.add_argument("--caixa", type=float, default=0.0,
                    help="caixa POR JUNTA em volta da pose de partida (rad; 0 = desligada, só os limites do URDF). "
                         "Travava o braço esquerdo em 4 juntas quando a IA pedia para descer (29/09)")
    ap.add_argument("--caixa-mao", type=float, nargs=4, default=[0.25, 0.40, 0.50, 0.15],
                    metavar=("FRENTE_TRAS", "LADOS", "BAIXO", "CIMA"),
                    help="caixa em volta da MÃO na pose de partida, em metros: ±frente/trás, ±lados, baixo, cima. "
                         "02/10: era 0,30 para baixo e para os lados — na maçã (mesa baixa) a mão direita parava em "
                         "z 6 cm com o modelo pedindo 2-3 cm, e em y -20 (limite -18)")
    ap.add_argument("--narra", default="http://127.0.0.1:8098",
                    help="ER-1 (er1_pergunta.py): a cada --narra-s o robô diz o que está fazendo, olhando a câmera da "
                         "cabeça com a tarefa atual ('' desliga)")
    ap.add_argument("--narra-s", type=float, default=8.0, help="segundos entre as narrações do ER-1")
    ap.add_argument("--narra-cams", default="wrist_right",
                    help="câmeras que o ER-1 vê para dizer o objeto mais perto da mão direita")
    ap.add_argument("--narra-repete-s", type=float, default=60.0,
                    help="a mesma frase do ER-1 (reaching = reach) só é dita de novo depois disto")
    ap.add_argument("--peso-rot", type=float, default=0.15,
                    help="peso da orientação da mão na IK (posição = 1). Era 0,5: segurar a inclinação travava a descida")
    ap.add_argument("--mao-mao-min", type=float, default=0.08, help="distância lateral mínima entre as mãos (m)")
    ap.add_argument("--juntas-travadas", type=int, nargs="*", default=[],
                    help="juntas com defeito, fixas na posição medida: a IK resolve sem elas e o integrador não "
                         "insiste (ex.: 20 = pitch do punho esquerdo, que estava travado em 29/09)")
    ap.add_argument("--voz", choices=["g1", "piper", "nenhuma"], default="g1",
                    help="o robô fala o que vai fazer: g1 = voz interna do G1; piper = voz dos alertas; nenhuma")
    ap.add_argument("--braco-esquerdo", choices=["parado", "ativo"], default="ativo",
                    help="parado = o braço esquerdo fica na pose inicial (só a MÃO abre/fecha) — o punho esquerdo do "
                         "Prometheus está travado e, sem ele, a IK levava o braço para cima e para trás (29/09)")
    ap.add_argument("--gravidade", type=float, default=1.0,
                    help="compensação da GRAVIDADE dos braços (fator do torque do modelo do G1, 0 desliga; 02/10, como "
                         "o gravity_compensation do LeRobot). A ponte só deixa passar com TAU_MAX_* > 0 no init")
    ap.add_argument("--massa-punho", type=float, default=0.15,
                    help="kg no punho fora do URDF (D435 + suporte), para a compensação da gravidade")
    ap.add_argument("--zona-morta-cm", type=float, nargs=2, default=[0.0, 0.0], metavar=("ESQ", "DIR"),
                    help="cm: se o modelo pede a mão a menos disto do ponto SEGURADO, ela fica parada (0 desliga). "
                         "02/10 (rodada 131051): o modelo pede a mão esquerda ~1 cm abaixo da MEDIDA a cada consulta; "
                         "ancorado na medida isso somava e o braço descia 35 cm em 25 s, sem o integrador agir")
    ap.add_argument("--kp-tronco", type=float, nargs=2, default=[300.0, 8.0], metavar=("KP", "KD"),
                    help="roll e pitch da cintura (alvo 0 = coluna reta, como na gravação). 02/10: com 150/5 o pitch "
                         "ficava 0,08-0,16 rad à frente e o roll/pitch esquentou até a cintura ceder; a gravação usou 300/8")
    ap.add_argument("--temp-tronco", type=float, default=0.0,
                    help="°C: acima disto no roll/pitch da cintura SEGURA a posição (cabine 'parar'); 0 = não segura, "
                         "só avisa a partir de 60 °C (padrão, 02/10)")
    ap.add_argument("--ki", type=float, default=1.5,
                    help="integrador por junta (1/s): corrige o braço que cede sob o peso até a junta MEDIDA chegar "
                         "onde a IK mandou (0 desliga)")
    ap.add_argument("--peso-repouso", type=float, default=0.005,
                    help="mola da IK puxando o braço para a postura de partida (0 desliga). 02/10: com 0, nos alvos reais "
                         "da rodada 20261002_105107 a IK do braço direito saltava até 0,47 rad entre passos e trocava de "
                         "ramo (ombro roll -1,05 -> +0,54); com 0,005 o salto cai para ~0,27 rad com ~1 cm de erro na mão. "
                         "Valores maiores seguram mais a postura, mas a mão erra mais o alvo (0,02: ~3 cm)")
    ap.add_argument("--servidor", default="ws://127.0.0.1:8600",
                    help="servidor oficial do WLA já carregado (sobe_wla_servidor.sh); 'processo' = carrega aqui")
    ap.add_argument("--sem-pose-inicial", action="store_true", help="começa da pose atual (fora da distribuição)")
    ap.add_argument("--pose", choices=["inicial", "elevada", "dataset", "maca", "gravacao", "gravacao_aberta", "coador"], default="inicial",
                    help="inicial = coluna reta e mãos afastadas (~49 cm, 30/09); elevada = a do dataset com o "
                         "ombro 0,35 rad mais alto e tronco 10° à frente; dataset = a mediana do início dos episódios Dex1; maca = a do início dos 50 episódios do nosso dataset da maçã (use com o modelo afinado); gravacao = braços abertos fora da cena (a do botão do painel, 01/10 — use com modelos treinados com episódios que começam nela)")
    ap.add_argument("--cintura-reta", action="store_true",
                    help="não inclina o tronco (padrão: pitch da cintura 0,18 rad, como no dataset)")
    ap.add_argument("--gravar", default=str(Path.home() / "wla_real_runs"),
                    help="pasta onde cada rodada grava imagens + estado + resposta da IA por consulta ('' desliga)")
    ap.add_argument("--tempo-pose", type=float, default=10.0)
    ap.add_argument("--tempo-volta", type=float, default=6.0,
                    help="s: botão 'posição inicial' da cabine — volta devagar para a pose de partida e segura")
    ap.add_argument("--hz", type=float, default=20.0,
                    help="taxa de execução das ações (o dataset é 30 Hz; a 20 Hz um trecho de 30 passos dura 1,5 s, "
                         "mais que a demora da rede, e o movimento não para entre consultas)")
    ap.add_argument("--acoes-por-chunk", type=int, default=30, help="até que passo do trecho executar (máx. 30)")
    ap.add_argument("--passo-max-cm", type=float, default=1.0,
                    help="quanto a mão pode andar por passo de controle (1 cm a 20 Hz = 20 cm/s); corta saltos")
    ap.add_argument("--passo-max-rad", type=float, default=0.03, help="quanto cada junta pode mudar por passo")
    ap.add_argument("--tarefa", default="Pick up the apple.",
                    help='frase para o WLA (a do treino do dataset de fruta era "Pick up the fruit and place it on '
                         'the plate.")')
    ap.add_argument("--cabeca", choices=["cor", "estereo", "zed", "ir"], default="estereo",
                    help="imagem da cabeça para o modelo: estereo/zed/ir = head_stereo_left (ZED com o "
                         "cameras_wla_server.py, ou IR da D435i com o realsense_estereo_server.py); "
                         "cor = head_camera (D435i colorida)")
    ap.add_argument("--sem-punhos", action="store_true",
                    help="não manda as câmeras dos punhos (wrist_left/wrist_right) ao modelo")
    ap.add_argument("--ensaio", action="store_true", help="roda tudo, mas NÃO envia nada ao robô")
    ap.add_argument("--sem-maos", action="store_true", help="não comanda as Dex3 (só braços)")
    ap.add_argument("--z-min", type=float, default=None,
                    help="cm (z da mão na pelvis): PISO — a mão nunca é mandada abaixo disto. Use quando a mesa estiver "
                         "mais alta que no dataset (ex.: maçã gravada na caixa a ~-3 cm, mesa da caneca ~10 cm mais alta: "
                         "--z-min 5)")
    ap.add_argument("--segura-mao-dir", type=float, default=0.0,
                    help="0..1: antes de ligar o modelo, espera você pôr o objeto na mão direita e fecha até este "
                         "fechamento (tarefas que começam segurando: --pose coador --segura-mao-dir 0.9)")
    ap.add_argument("--assenta", type=float, default=3.0,
                    help="s: antes de seguir o modelo, segura a pose de partida com o integrador corrigindo o peso do "
                         "braço (0 desliga)")
    ap.add_argument("--dedos-estado", choices=["zero", "medido"], default="zero",
                    help="o que vai ao modelo como estado dos dedos: zero = como nos datasets gravados até 02/10 (as "
                         "juntas da Dex3 saíram 0 por um defeito da gravação); medido = a posição real (modelos "
                         "treinados com datasets gravados depois da correção)")
    ap.add_argument("--atalhos", nargs="*", default=[
        "pegar caneca=Pick up the white mug.",
        "caneca → coador=Place the white mug under the coffee strainer.",
        "maçã → X=Pick up the apple and place it on the black X.",
        "pegar maçã=Pick up the apple."],
                    help="botões de tarefa na cabine, 'rótulo=frase'")
    ap.add_argument("--mistura", type=float, default=6,
                    help="passos (a --hz) de transição suave entre um trecho e o próximo; 0 desliga")
    ap.add_argument("--cintura", choices=["yaw", "parada"], default="yaw",
                    help="yaw = gira o tronco como o modelo pede (o dataset gravou o tronco seguindo a cabeça); "
                         "parada = segura a cintura onde está")
    ap.add_argument("--cintura-yaw-max", type=float, default=0.6, help="rad: limite do giro do tronco (0,6 ≈ 34°; 02/10: era 0,35, e no dataset da maçã o tronco gira até 0,66)")
    ap.add_argument("--kp-mao", type=float, default=0.5, help="kp dos dedos Dex3 (modelo fig6d)")
    ap.add_argument("--kd-mao", type=float, default=0.1, help="kd dos dedos Dex3 (modelo fig6d)")
    ap.add_argument("--folga-mao", type=float, default=0.25,
                    help="rad: o alvo do dedo fica no máx. isto da posição medida — limita a força ao segurar "
                         "(≈ kp-mao x folga-mao por junta)")
    ap.add_argument("--fecha-max", type=float, default=0.85, help="fração do fechamento total que o modelo pode pedir")
    ap.add_argument("--ckpt_path", default="playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors")
    ap.add_argument("--log", default=str(Path.home() / f"wla_real_{time.strftime('%Y%m%d_%H%M%S')}.csv"))
    a = ap.parse_args()
    # WARNING: o servidor oficial tem um logging.info com 2 "%s" e 1 argumento, que quebra em INFO.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    if a.escala > 1.0 or a.caixa > 1.0 or max(a.caixa_mao) > 0.6:
        sys.exit("escala > 1,0, caixa > 1,0 rad ou caixa da mão > 0,6 m: recusado")

    ctx = zmq.Context.instance()
    if modo_da_ponte(ctx, a.robo) != "loco":
        sys.exit("❌ a ponte não está em 'loco' (6004). Abortando sem enviar nada.")
    est, maos = Estado(ctx, a.robo), sw.Maos(ctx, a.robo)
    cabine = sw.Cabine()
    sw.sobe(cabine, a.porta)
    cabine.define_tarefa(a.tarefa)
    cabine.atalhos = [tuple(x.split("=", 1)) for x in a.atalhos]
    # GPU: o WLA e o ER-1 (narração/tradução) rodam UM DE CADA VEZ (02/10: juntos, a consulta do WLA ia a 1,5-1,7 s e o
    # trecho chegava tarde demais e era descartado; em fila, o WLA só começa depois e observa a cena já atualizada)
    gpu = threading.Lock()
    if a.narra:   # botão PT-BR do histórico de falas: o ER-1 traduz (já desde o começo, 02/10)
        def _traduz(texto):
            import urllib.parse
            import urllib.request
            url = f"{a.narra}/traduz?" + urllib.parse.urlencode({"t": texto})
            with gpu:
                return json.loads(urllib.request.urlopen(url, timeout=30).read())["pt"]
        cabine.tradutor = _traduz
    cabine.publica_estado({"modo": "carregando o WLA-1.0 ..."})
    cam = sw.Camera(a.robo, 5555, cabine, "head_camera" if a.cabeca == "cor" else "head_stereo_left")
    # Punhos: como no treino (cabeça + punho esq. + punho dir.). Entram só se estiverem chegando.
    punhos = {} if a.sem_punhos else {"cam_left_wrist": sw.Camera(a.robo, 5555, cabine, "wrist_left"),
                                      "cam_right_wrist": sw.Camera(a.robo, 5555, cabine, "wrist_right")}
    fk = FK()
    t0 = time.time()
    dex3 = False   # modelo Dex3 (fig6d)? só um servidor externo que anuncie fig6d no metadata liga isto
    if a.servidor == "processo":
        S = sw.Sombra(argparse.Namespace(ckpt_path=a.ckpt_path, instruction=a.tarefa, unnorm_key="UnifoLM_G1_Dex1",
                                         use_bf16=True, image_size=[320, 448], debug_save_dir=None))
        print(f"WLA-1.0 carregado aqui em {time.time() - t0:.0f} s", flush=True)
    else:
        try:
            S = ClienteWLA(a.servidor)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"❌ servidor do WLA não respondeu em {a.servidor} ({e}). Suba com sobe_wla_servidor.sh "
                     "(≈80 s para carregar) e rode de novo. Nada foi enviado ao robô.")
        dex3 = "observation.state.left_fig6d" in (S.meta.get("data_keys") or [])
        print(f"🖐  mãos: {'Dex3 pelos DEDOS (fig6d)' if dex3 else 'garra Dex1 (abre/fecha a mão inteira)'}", flush=True)
        print(f"🔌 conectado ao servidor do WLA {a.servidor} (normalização {S.meta.get('default_unnorm_key')}, "
              f"chunk {S.meta.get('action_chunk_size')})", flush=True)
    t0 = time.time()
    while (est.low is None or est.panico is None or cam.rgb is None or not maos.vivo()) and time.time() - t0 < 8:
        time.sleep(0.05)
    faltando = [n for n, v in (("lowstate", est.low), ("pânico", est.panico), ("câmera", cam.rgb),
                               ("mãos", maos.vivo() or None)) if v is None]
    if faltando:
        sys.exit(f"❌ sem {faltando} em 8 s. Abortando sem enviar nada.")
    if est.panico["gpio"] != "ok":
        sys.exit("❌ o GPIO do botão não está ok na ponte. Abortando.")
    time.sleep(1.0)
    for k, c in punhos.items():
        print(f"📷 {c.nome}: " + (f"chega {c.origem[0]}x{c.origem[1]} -> modelo ({k})" if c.rgb is not None
                                   else "NÃO está chegando — o modelo recebe só a cabeça"), flush=True)
    print(f"📷 {cam.nome}: chega {cam.origem[0]}x{cam.origem[1]}, vai ao modelo 640x480"
          f"{' (RECORTADA de 16:9: suba o realsense_estereo_server.py)' if cam.origem != (640, 480) else ''}",
          flush=True)
    print(f"✅ ponte loco | pânico {est.panico['panico']} ({est.panico['motivo'] or 'livre'}) | câmera ok | "
          f"cabine http://192.168.123.165:{a.porta}/", flush=True)

    rearme = ctx.socket(zmq.PUSH)
    rearme.setsockopt(zmq.LINGER, 500)
    rearme.connect(f"tcp://{a.robo}:6006")
    cmd_sock = ctx.socket(zmq.PUSH)
    cmd_sock.setsockopt(zmq.LINGER, 0)
    cmd_sock.setsockopt(zmq.SNDHWM, 2)
    cmd_sock.connect(f"tcp://{a.robo}:6000")
    mao_sock = ctx.socket(zmq.PUSH)
    mao_sock.setsockopt(zmq.LINGER, 0)
    mao_sock.setsockopt(zmq.SNDHWM, 2)
    mao_sock.connect(f"tcp://{a.robo}:6003")

    def panico_software(motivo):
        print(f"\n🛑 pânico por software: {motivo}", flush=True)
        if not a.ensaio:
            rearme.send(json.dumps({"panico": True}).encode("utf-8"))
            time.sleep(0.3)

    def q29():
        return np.array([est.q(i) for i in range(29)])

    # ── pose inicial por IK ──
    inicio = q29()
    alvo = {i: float(inicio[i]) for i in list(CINTURA) + list(BRACOS)}
    q_ini = inicio.copy()
    if not a.sem_pose_inicial:
        suf = {"inicial": "_inicial", "elevada": "_elevada", "dataset": "", "maca": "_maca", "gravacao": "_gravacao", "gravacao_aberta": "_gravacao_aberta", "coador": "_coador"}[a.pose]
        q_ini[BRACO["left"]] = POSE_JUNTAS["left" + suf]
        q_ini[BRACO["right"]] = POSE_JUNTAS["right" + suf]
        if not a.cintura_reta:
            # tronco como no dataset: pitch 0,18 rad (10° à frente) — o modelo sempre viu isso no estado
            # e a câmera da cabeça deles olhava 58° para baixo por causa disso (state_d435)
            q_ini[list(CINTURA)] = POSE_JUNTAS.get("cintura" + suf, POSE_JUNTAS["cintura"])
        for lado in ("left", "right"):
            p = fk.pose(q_ini, lado)[1]
            print(f"   pose inicial {lado}: mão em {np.round(p * 100, 1)} cm da pelvis (x frente, y esq, z cima)",
                  flush=True)
        maior = max(abs(q_ini[i] - inicio[i]) for i in list(BRACOS) + list(CINTURA))
        print(f"   cintura: {np.round(inicio[list(CINTURA)], 3)} -> {np.round(q_ini[list(CINTURA)], 3)} (yaw, roll, pitch)",
              flush=True)
        print(f"   pose inicial do dataset: até {maior:.2f} rad de diferença; vai em {a.tempo_pose:.0f} s", flush=True)

    if est.panico["panico"] and not a.ensaio:
        input("🔒 Ponte bloqueada. Solte o cogumelo e aperte Enter para rearmar (ou rearme no painel :8095)... ")
        if est.panico["panico"]:
            rearme.send(json.dumps({"rearmar": True}).encode("utf-8"))
            t1 = time.time()
            while est.panico["panico"] and time.time() - t1 < 3:
                time.sleep(0.05)
        if est.panico["panico"]:
            sys.exit(f"❌ rearme recusado: {est.panico['motivo']}")
        print("🔓 rearmado.", flush=True)

    print(f"▶ {a.segundos:.0f} s | escala {a.escala:.0%} | caixa ±{a.caixa} rad | braços"
          f"{' (mãos desligadas)' if a.sem_maos else ' + mãos Dex3 (abre/fecha pela garra do WLA)'} | "
          f"{a.acoes_por_chunk}/30 ações por consulta a {a.hz:.0f} Hz | tronco: "
          f"{'yaw do modelo (±%.2f rad)' % a.cintura_yaw_max if a.cintura == 'yaw' else 'parado'}"
          f"{'  [ENSAIO: nada é enviado]' if a.ensaio else ''}", flush=True)
    input("   Fruta (e prato) na frente do robô? Mão no cogumelo. Enter para começar... ")

    # COMPENSAÇÃO DA GRAVIDADE (02/10): torque que segura o peso do braço na pose COMANDADA, com a gravidade
    # vinda da IMU da pelvis. Conferido no robô parado: ombros/cotovelos a ~10% do tau_est medido.
    extra_punho = {l: (a.massa_punho, (0.05, 0.0, 0.04)) for l in ("left", "right")}

    def tau_gravidade():
        if a.gravidade <= 0:
            return None
        q = q29()
        for i in BRACOS:
            q[i] = alvo[i]
        r, p_ = est.low["imu_state"]["rpy"][:2]
        cr, sr, cp, sp = np.cos(r), np.sin(r), np.cos(p_), np.sin(p_)
        g = np.array([[cp, sp * sr, sp * cr], [0, cr, -sr], [-sp, cp * sr, cp * cr]]).T @ np.array([0, 0, -9.81])
        tau = a.gravidade * fk.gravidade(q, g, extra_punho)
        for i in a.juntas_travadas:   # junta travada (punho esquerdo danificado) não faz força
            tau[i] = 0.0
        return tau

    def envia():
        if not a.ensaio:
            cmd_sock.send(json.dumps(monta_cmd(alvo, est, 80.0, 3.0, 40.0, 1.5, *a.kp_tronco,
                                               tau=tau_gravidade())).encode("utf-8"))

    fech_mao = {"left": None, "right": None}

    def envia_maos(garras_wla):
        for l, g in garras_wla.items():
            msg, fech_mao[l] = cmd_mao(l, g)
            if not a.ensaio and not a.sem_maos:
                mao_sock.send(json.dumps(msg).encode("utf-8"))

    def envia_maos_fig6d(f6s):
        """Modelo Dex3: fig6d (0..1 por dedo) -> as 7 juntas; fechamento = média dos 6.

        FORÇA LIMITADA (01/10): com a maçã na mão o dedo nunca chega ao alvo "fechado", e com o alvo no limite
        o motor empurrava com tudo até a proteção da Dex3 desligar o servo (só reiniciando o robô). Agora:
          - o fechamento máximo é --fecha-max do limite;
          - o alvo de cada junta fica a no máximo --folga-mao rad da posição MEDIDA: encostou no objeto, a
            força para em ≈ kp x folga (padrão 0,5 x 0,25 ≈ 0,12 N·m por junta), em vez de crescer;
          - kp dos dedos --kp-mao (a ponte ainda corta em 1,0)."""
        for l, f6 in f6s.items():
            fech_mao[l] = float(np.mean(f6))
            q_alvo = fig6d_para_dex3(np.clip(np.asarray(f6, float), 0.0, 1.0) * a.fecha_max, l)
            qm = maos.q.get(l)
            if qm is not None and len(qm) == 7:
                qm = np.asarray(qm, float)
                q_alvo = np.clip(q_alvo, qm - a.folga_mao, qm + a.folga_mao)
            if not a.ensaio and not a.sem_maos:
                mao_sock.send(json.dumps(cmd_mao_q(l, q_alvo, kp=a.kp_mao, kd=a.kd_mao)).encode("utf-8"))

    def seguro(onde):
        """Falso se for para parar: pânico na ponte, estado velho ou botão parar da cabine."""
        if est.panico["panico"] and not a.ensaio:
            print(f"\n🛑 pânico na ponte ({est.panico['motivo']}) {onde}. A ponte segura o braço.", flush=True)
            return False
        if time.time() - est.t_low > 0.2:
            panico_software(f"lowstate parou de chegar ({onde})")
            return False
        if cabine.parada_pedida():
            # 02/10: PARAR da cabine = o robô MANTÉM a posição em que está e espera outra tarefa (não é pânico;
            # a emergência é o cogumelo). Limpa a tarefa — o laço segura a pose comandada atual.
            cabine.define_tarefa("")
            print(f"\n   ⏸ parar pedido na cabine ({onde}): mantendo a posição — mande outra tarefa para continuar",
                  flush=True)
        return True

    # ── 0. pose inicial, devagar ──
    ja_na_pose = not a.sem_pose_inicial and max(abs(est.q(i) - q_ini[i]) for i in list(BRACOS) + list(CINTURA)) < 0.12
    if ja_na_pose:
        for i in list(BRACOS) + list(CINTURA):
            alvo[i] = float(q_ini[i])
        print("   já está na pose inicial (botão do painel :8095?) — começa direto.", flush=True)
    if not a.sem_pose_inicial and not ja_na_pose:
        print("   indo para a pose inicial...", flush=True)
        t_p = time.time()
        while time.time() - t_p < a.tempo_pose + 2.0:
            tc = time.time()
            if not seguro("na pose inicial"):
                return
            u = min(1.0, (tc - t_p) / a.tempo_pose)
            s = 0.5 - 0.5 * np.cos(np.pi * u)
            for i in list(BRACOS) + list(CINTURA):
                alvo[i] = float(inicio[i] + s * (q_ini[i] - inicio[i]))
            envia()
            time.sleep(max(0.0, 1.0 / 50 - (time.time() - tc)))
        erro = max(abs(est.q(i) - q_ini[i]) for i in BRACOS)
        print(f"   pose inicial atingida (maior erro medido {erro:.2f} rad).", flush=True)
    partida = dict(alvo)

    # ── OBJETO NA MÃO antes de começar (02/10): tarefas que começam SEGURANDO algo (caneca -> coador). ──
    # O braço fica na pose; você põe o objeto na mão direita, Enter, e a mão fecha com a força limitada
    # (--kp-mao/--folga-mao) até --segura-mao-dir do fechamento — como no início dos episódios do dataset.
    if a.segura_mao_dir > 0 and not a.ensaio:
        import threading as _th
        _para = _th.Event()
        _fecha = [0.0]

        def _segura():
            while not _para.is_set():
                envia()
                envia_maos_fig6d({"left": np.zeros(6), "right": np.full(6, _fecha[0])})
                time.sleep(0.02)

        _th.Thread(target=_segura, daemon=True).start()
        input("   🤲 Ponha o objeto na MÃO DIREITA do robô e aperte Enter para ela fechar... ")
        for u in np.linspace(0, 1, 40):          # fecha em ~2 s
            _fecha[0] = float(u * a.segura_mao_dir)
            time.sleep(0.05)
        time.sleep(1.0)
        q_m = maos.q.get("right")
        print(f"   🤲 mão direita fechada (fig6d medido {np.round(dex3_para_fig6d(q_m, 'right'), 2) if q_m is not None else '?'})",
              flush=True)
        input("   Objeto firme? Enter para ligar o modelo... ")
        _para.set()
        time.sleep(0.05)

    arq = open(a.log, "w", newline="")
    log = csv.writer(arq)
    log.writerow(["t", "consulta", "ms", "lado", "ee_medido", "ee_comandado", "ee_modelo_fim", "ee_alvo_fim",
                  "erro_ik_mm", "juntas_cortadas_pela_caixa", "q_comandado", "q_medido", "passo_inicial"])

    # ── consulta à rede EM PARALELO com o movimento ──
    # A rede leva ~0,67 s. Antes o braço ficava parado esperando cada resposta. Agora uma thread
    # consulta sem parar e o laço de controle executa sempre o trecho MAIS NOVO, no passo que
    # corresponde a "agora" (passo = (agora - instante da observação) x hz). A --hz 20 um trecho
    # de 30 passos dura 1,5 s, mais que o intervalo entre respostas: o movimento não para.
    pare = threading.Event()
    novo_trecho, erro_consulta = [None], [None]
    trava = threading.Lock()

    def monta_obs():
        q = q29()
        ee = {l: fk.ee9(q, l) for l in ("left", "right")}
        garras = {l: maos.garra(l) for l in ("left", "right")}
        frase = cabine.tarefa()[0] or a.tarefa
        obs = {"observation.images.cam_left_high": sw.cv2.cvtColor(cam.rgb, sw.cv2.COLOR_RGB2BGR),
               "observation.state.left_ee_6d": ee["left"], "observation.state.right_ee_6d": ee["right"],
               "observation.state.left_gripper": np.array([garras["left"]], np.float32),
               "observation.state.right_gripper": np.array([garras["right"]], np.float32),
               "observation.state.lower_body": q[:15].astype(np.float32), "instruction": frase}
        if dex3:
            for l in ("left", "right"):
                # --dedos-estado zero: os datasets até 02/10 gravaram as juntas da Dex3 como 0 (defeito no
                # unitree_sdk2_socket) — os modelos treinados com eles nunca viram outro valor de dedo no estado.
                obs[f"observation.state.{l}_fig6d"] = (np.zeros(6, np.float32) if a.dedos_estado == "zero"
                                                       else dex3_para_fig6d(maos.q.get(l, np.zeros(7)), l))
        # Os DOIS punhos ou nenhum: o servidor oficial rotula as imagens pela ORDEM
        # (WBC_IMAGE_ROLES[:n]); só o direito seria rotulado como cam_wrist_left.
        if punhos and all(c.rgb is not None and time.time() - c.t < 0.5 for c in punhos.values()):
            for k, c in punhos.items():
                obs[f"observation.images.{k}"] = sw.cv2.cvtColor(c.rgb, sw.cv2.COLOR_RGB2BGR)
        return obs, q, ee, garras, frase

    grav = None
    if a.gravar:
        from gravador_wla import Gravador
        grav = Gravador(a.gravar, vars(a))
        print(f"💾 gravando cada consulta em {grav.pasta}", flush=True)

    def laco_consulta():
        n = 0
        while not pare.is_set():
            try:
                frase_cab, seq = cabine.tarefa()
                if not (frase_cab or a.tarefa):      # sem tarefa: o robô SEGURA e a rede não é consultada
                    time.sleep(0.1)
                    continue
                with gpu:
                    t_obs = time.time()
                    obs, q, ee, garras, frase = monta_obs()
                    # Âncora = pose COMANDADA no instante da observação (o braço cede sob o peso; ancorar no
                    # medido acumulava a queda). O modelo viu a MEDIDA: só o DESLOCAMENTO dele é aplicado.
                    q_cmd = q.copy()
                    for i in BRACOS:
                        q_cmd[i] = alvo[i]
                    _, acao, ms, _, _ = S.acao(obs)
                n += 1
                trecho = {"n": n, "acao": acao, "ms": ms, "t_obs": t_obs, "q": q, "ee": ee, "garras": garras,
                          "frase": frase, "seq": seq, "base": {l: fk.pose(q_cmd, l) for l in ("left", "right")},
                          "medida": {l: fk.pose(q, l) for l in ("left", "right")}}
                with trava:
                    novo_trecho[0] = trecho
                if grav is not None:
                    grav.consulta(trecho, obs, q_cmd, est.panico, fk)
            except Exception as e:  # noqa: BLE001
                erro_consulta[0] = e
                return

    threading.Thread(target=laco_consulta, daemon=True, name="consulta").start()
    fala = Falador(a.robo, a.voz) if a.voz != "nenhuma" else None
    if fala is not None:
        fala.ao_falar = cabine.registra_fala   # histórico na cabine (02/10)
    # Fala SEM poluir: cada tarefa é anunciada uma vez; "pegando/soltando" só depois de a mão ficar
    # fechada (ou aberta) por >= 1 s, e cada mão espera 6 s entre avisos — a garra do WLA abre e fecha
    # rápido, e antes cada tremida virava uma frase.
    anunciadas = set()
    fechada = {"left": False, "right": False}
    desde = {"left": None, "right": None}
    ultimo_aviso = {"left": 0.0, "right": 0.0}

    # NARRAÇÃO pelo ER-1 (02/10): no lugar de "Grasping/Releasing", a cada --narra-s o ER-1 olha a câmera da
    # cabeça e diz numa frase curta o que o robô está fazendo na tarefa atual (thread própria: nunca atrasa o controle).
    def laco_narra():
        """NARRAÇÃO SEM INVENTAR (02/10): a frase sai dos SENSORES — para onde a mão andou (juntas), dedos abertos,
        fechando ou segurando algo (dedos mandados fechar mas parados antes = objeto na mão) — e o ER-1 só diz o
        NOME do objeto mais perto da mão direita (câmera do punho; resposta de poucas palavras, ~0,3 s). A descrição
        livre do ER-1 inventava ("I have picked up the white mug" com a mão parada e aberta). Repetir é permitido."""
        import urllib.parse
        import urllib.request
        falhou = 0.0
        antes = {"frase": None, "p": None}

        def movimento(d):
            partes = []
            for k, (pos, neg) in enumerate((("forward", "back"), ("left", "right"), ("up", "down"))):
                if abs(d[k]) >= 0.02:
                    partes.append(f"{pos if d[k] > 0 else neg} {abs(d[k]) * 100:.0f} cm")
            return ("moving my right hand " + ", ".join(partes)) if partes else "keeping my right hand still"

        def dedos(l):
            c = fech_mao.get(l)
            qm = maos.q.get(l)
            if c is None:
                return "fingers unknown"
            medido = float(np.mean(dex3_para_fig6d(np.asarray(qm, float), l))) if qm is not None and len(qm) == 7 else None
            if c > 0.6 and medido is not None and medido < c * a.fecha_max - 0.2:
                return "holding something in my right hand"      # mandou fechar e os dedos pararam no objeto
            if c > 0.6:
                return "fingers closed, nothing in my hand"
            if c > 0.3:
                return "closing my fingers"
            return "fingers open"

        while not pare.is_set():
            time.sleep(a.narra_s)
            frase = cabine.tarefa()[0] or a.tarefa
            if not frase:
                continue
            p = fk.pose(q29(), "right")[1]
            if antes["frase"] != frase:
                antes.update(frase=frase, p=p)
            texto = f"{movimento(p - antes['p']).capitalize()}, {dedos('right')}"
            antes["p"] = p
            if time.time() - falhou > 60:
                try:
                    q = ("What object is closest to the robot's right hand in these images? "
                         "Answer with only the object name, at most 3 words, or 'nothing'.")
                    u = f"{a.narra}/pergunta?" + urllib.parse.urlencode({"q": q, "livre": 1, "max": 8,
                                                                         "cam": a.narra_cams})
                    with gpu:
                        r = json.loads(urllib.request.urlopen(u, timeout=10).read())
                    if r.get("erro"):
                        raise RuntimeError(r["erro"])
                    obj = re.sub(r"[^a-zA-Z ]", "", (r.get("cru") or "")).strip().lower()
                    if obj and obj != "nothing" and len(obj.split()) <= 4:
                        texto += f", near the {obj.removeprefix('the ').removeprefix('a ')}"
                except Exception as e:  # noqa: BLE001
                    falhou = time.time()
                    print(f"\n⚠️  ER-1 indisponível ({a.narra}): {e} — narro só pelos sensores por 60 s", flush=True)
            if (cabine.tarefa()[0] or a.tarefa) != frase:
                continue
            texto += "."
            print(f"\n🗣️  narração: {texto}", flush=True)
            fala.diz(texto, "sensores+ER-1")

    if fala is not None and a.narra:
        threading.Thread(target=laco_narra, daemon=True, name="narra").start()

    def narra(frase, fech):
        if fala is None:
            return
        if frase and frase not in anunciadas:
            anunciadas.add(frase)
            fala.diz("I will " + frase[0].lower() + frase[1:])
    dt = 1.0 / a.hz
    atual, q_ik, usados = None, None, 0
    anterior, t_troca = None, 0.0
    lados_ativos = ("right",) if a.braco_esquerdo == "parado" else ("left", "right")
    if a.braco_esquerdo == "parado":
        print("   braço ESQUERDO parado na pose inicial (só a mão esquerda abre/fecha)", flush=True)
    # Integrador por junta: comando = q desejada (IK) + integral do erro (desejada - medida). Só acumula
    # com o alvo parado (senão confunde o atraso do movimento com o peso). Junta que não sai do lugar por
    # 3 s (batente mecânico — o pitch do punho esquerdo ficou cravado em 0,28) deixa de ser integrada.
    integ = {i: 0.0 for i in BRACOS}
    q_des = {i: float(alvo[i]) for i in BRACOS}
    q_des_ant = dict(q_des)
    presa = {i: 0.0 for i in BRACOS}
    desistiu = set(a.juntas_travadas)
    qm_ant = [None]

    def aplica_integrador():
        qm = q29()
        vel = np.zeros(29) if qm_ant[0] is None else np.abs(qm - qm_ant[0]) / dt
        qm_ant[0] = qm
        for i in BRACOS:
            e = q_des[i] - qm[i]
            # só acumula com o ALVO e o BRAÇO parados: durante o caminho o erro é atraso, não peso
            parado = abs(q_des[i] - q_des_ant[i]) < 0.01 and vel[i] < 0.1
            if a.ki > 0 and parado and i not in desistiu:
                integ[i] = float(np.clip(integ[i] + a.ki * e * dt, -0.3, 0.3))
                presa[i] = presa[i] + dt if (abs(e) > 0.2 and abs(integ[i]) >= 0.29) else 0.0
                if presa[i] > 3.0:
                    desistiu.add(i)
                    integ[i] = 0.0
                    print(f"\n⚠️  junta {i} não sai do lugar (comando {q_des[i]:+.2f}, medida {qm[i]:+.2f}): parei de "
                          "forçar. Batente mecânico? (suporte de câmera, cabo)", flush=True)
            q_des_ant[i] = q_des[i]
            alvo[i] = float(q_des[i] + integ[i])
    # A CAIXA vira limite DENTRO da IK: cortar junta por junta DEPOIS da IK fazia as juntas livres
    # (o punho) compensarem as cortadas — o "jab" do braço esquerdo em 29/09 (45 cortes por trecho).
    lo_caixa, hi_caixa = np.full(29, -np.inf), np.full(29, np.inf)
    if a.caixa > 0:
        for i in BRACOS:
            lo_caixa[i], hi_caixa[i] = partida[i] - a.caixa, partida[i] + a.caixa
    q_agora = q29()
    for i in a.juntas_travadas:
        lo_caixa[i] = hi_caixa[i] = float(q_agora[i])   # a IK não conta com ela
        partida[i] = float(q_agora[i])
        alvo[i] = q_des[i] = q_des_ant[i] = float(q_agora[i])
    if a.juntas_travadas:
        print(f"   juntas travadas (fixas na posição medida): {a.juntas_travadas} = "
              f"{[round(float(q_agora[i]), 3) for i in a.juntas_travadas]} rad", flush=True)
    # Caixa da MÃO (cartesiana, pelvis) em volta de onde cada mão está na pose de partida.
    q_part = q29()
    for i in BRACOS:
        q_part[i] = partida[i]
    mao_part = {l: fk.pose(q_part, l)[1] for l in ("left", "right")}
    fx, fy, fb, fc = a.caixa_mao
    caixa_lo = {l: mao_part[l] + np.array([-fx, -fy, -fb]) for l in mao_part}
    caixa_hi = {l: mao_part[l] + np.array([fx, fy, fc]) for l in mao_part}
    # Compensação do PESO do braço: medida UMA vez, parado na pose inicial, e FIXA (±6 cm).
    # Antes era (comandado - medido) a cada consulta: quando o comando andava mais rápido que a ponte
    # deixa, o ATRASO do braço entrava como "peso", empurrava o alvo mais longe, e o atraso crescia —
    # a mão esquerda foi parar em y = -9 cm, no rosto (29/09, 18:43).
    time.sleep(0.5)
    q_med0 = q29()
    comp = {l: np.clip(mao_part[l] - fk.pose(q_med0, l)[1], -0.06, 0.06) for l in mao_part}
    if a.ki > 0:
        comp = {l: np.zeros(3) for l in mao_part}   # o integrador por junta faz a compensação do peso
    print(f"   compensação do peso (fixa): esq {np.round(comp['left'] * 100, 1)} cm, "
          f"dir {np.round(comp['right'] * 100, 1)} cm", flush=True)
    print(f"   caixa da mão: ±{fx * 100:.0f} cm frente/trás, ±{fy * 100:.0f} cm lados, {fb * 100:.0f} cm para baixo, "
          f"{fc * 100:.0f} cm para cima | mãos a ≥ {a.mao_mao_min * 100:.0f} cm uma da outra | peso da orientação "
          f"{a.peso_rot}", flush=True)
    stats = {}
    volta = None   # botão 'posição inicial' da cabine
    t_temp = [0.0, 0.0]
    segurado = {"left": None, "right": None}   # (posição, ponto ee_rpy) que a zona morta segura
    try:
        avisou_assenta = [False]
        t_ini = time.time()
        while a.segundos <= 0 or time.time() - t_ini < a.segundos + a.assenta:
            tc = time.time()
            if not seguro("no laço de controle"):
                return
            if erro_consulta[0] is not None:
                raise RuntimeError(f"consulta à rede falhou: {erro_consulta[0]}")
            # ASSENTAR (02/10): nos primeiros --assenta s o braço só segura a pose de partida e o integrador corrige
            # o peso (com a caneca na mão o braço cedia ~10 cm sob o kp 40 da ponte; na teleoperação era 80). Os
            # trechos que chegam nesse tempo são descartados — o modelo começa com a mão onde o dataset começa.
            if tc - t_ini < a.assenta:
                with trava:
                    novo_trecho[0] = None
                if not avisou_assenta[0]:
                    print(f"   ⏳ assentando o braço {a.assenta:.0f} s (integrador corrigindo o peso) antes de seguir o modelo",
                          flush=True)
                    avisou_assenta[0] = True
                aplica_integrador()
                envia()
                time.sleep(max(0.0, dt - (time.time() - tc)))
                continue
            if avisou_assenta[0] is True:
                ee_d = fk.pose(q29(), "right")[1]
                print(f"   ✅ assentado: mão direita em {np.round(ee_d * 100, 1)} cm (alvo da partida "
                      f"{np.round(mao_part['right'] * 100, 1)} cm) — seguindo o modelo", flush=True)
                avisou_assenta[0] = "feito"
            with trava:
                chegou, novo_trecho[0] = novo_trecho[0], None
            frase_agora, seq_agora = cabine.tarefa()
            if not (frase_agora or a.tarefa):
                chegou = None
                if atual is not None:
                    print("   ⏸ segurando (sem tarefa) — mande uma tarefa na cabine para continuar", flush=True)
                    atual, anterior = None, None
            elif chegou is not None and chegou.get("seq") != seq_agora:
                chegou = None                      # trecho calculado com a frase anterior
            if atual is not None and atual.get("seq") != seq_agora:
                print(f"   ▶ nova tarefa: {frase_agora!r}", flush=True)
                atual, anterior = None, None
            # BOTÃO "POSIÇÃO INICIAL" da cabine (02/10): sem tarefa, os braços (e o yaw) voltam devagar à pose de partida
            if cabine.consome_inicial():
                cabine.define_tarefa("")
                frase_agora, atual, anterior, chegou = "", None, None, None
                volta = {"t0": tc, "de": {i: q_des[i] for i in BRACOS}, "yaw": alvo[12]}
                print(f"\n   ↩ voltando à posição inicial ({a.tempo_volta:.0f} s)", flush=True)
            if volta is not None:
                if frase_agora:                    # chegou tarefa nova no meio: ela manda
                    volta = None
                else:
                    u = min(1.0, (tc - volta["t0"]) / a.tempo_volta)
                    sv = 0.5 - 0.5 * np.cos(np.pi * u)
                    for i in BRACOS:
                        q_des[i] = float(volta["de"][i] + sv * (partida[i] - volta["de"][i]))
                    alvo[12] = float(volta["yaw"] + sv * (partida[12] - volta["yaw"]))
                    if u >= 1.0:
                        volta = None
                        print("   ✅ na posição inicial — mande uma tarefa na cabine", flush=True)
            if chegou is not None:
                if atual is not None:
                    _registra(atual, stats, log, t_ini, alvo, q29(), cabine, a, est, partida, fech_mao, usados)
                anterior, t_troca = atual, tc   # suavização: o trecho que sai ainda conta nos primeiros passos
                atual, usados = chegou, 0
                stats = {"cortes": {"left": 0, "right": 0}, "erros": {"left": 0.0, "right": 0.0}, "fins": {},
                         "passo_inicial": None}
                q_ik = q29()
                for i in BRACOS:
                    q_ik[i] = q_des[i]
            if atual is not None:
                acao = atual["acao"]
                T = min(a.acoes_por_chunk, acao["action.left_ee_rpy"].shape[1])
                k = int((tc - atual["t_obs"]) * a.hz)
                # SUAVIZAÇÃO entre trechos (02/10): cada consulta parte de ruído novo e, com o modelo pouco treinado,
                # trechos seguidos discordam — a mão "pulava" na troca. Nos primeiros --mistura passos do trecho
                # novo, a posição da mão e os dedos vão do trecho antigo (mesmo instante) ao novo, linearmente.
                peso_novo, k_ant = 1.0, None
                if a.mistura > 0 and anterior is not None:
                    passos_desde = (tc - t_troca) * a.hz
                    k_ant = int((tc - anterior["t_obs"]) * a.hz)
                    if passos_desde < a.mistura and k_ant < anterior["acao"]["action.left_ee_rpy"].shape[1]:
                        peso_novo = max(0.0, passos_desde / a.mistura)
                    else:
                        k_ant = None

                def ponto(chave, l):
                    v = np.array(acao[f"action.{l}_{chave}"][0][k], dtype=float)
                    if k_ant is None or peso_novo >= 1.0:
                        return v
                    v_ant = np.asarray(anterior["acao"][f"action.{l}_{chave}"][0][k_ant], dtype=float)
                    if chave == "ee_rpy":   # mistura só a posição; a orientação vem do trecho novo
                        v[:3] = (1 - peso_novo) * v_ant[:3] + peso_novo * v[:3]
                    else:
                        v = (1 - peso_novo) * v_ant + peso_novo * v
                    return v
                if k < T:
                    if stats["passo_inicial"] is None:
                        stats["passo_inicial"] = k
                    # TRONCO: o yaw da cintura que o modelo pede (action.lower_body[12]), limitado e devagar
                    # (no máx. --passo-max-rad por passo; a ponte ainda limita a 0,3 rad/s). A IK usa o yaw MEDIDO:
                    # o alvo da mão é na pelvis, então girar o tronco não muda para onde a mão vai.
                    if a.cintura == "yaw" and "action.lower_body" in acao:
                        yaw = float(np.clip(acao["action.lower_body"][0][k, 12], -a.cintura_yaw_max, a.cintura_yaw_max))
                        alvo[12] = float(alvo[12] + np.clip(yaw - alvo[12], -a.passo_max_rad, a.passo_max_rad))
                        stats["yaw"] = yaw
                    q_ik[12] = float(est.q(12))
                    alvos_p, vs = {}, {}
                    for l in ("left", "right"):
                        R0, p0 = atual["base"][l]
                        Rm, pm = atual["medida"][l]
                        v = ponto("ee_rpy", l)
                        # ZONA MORTA: pedido a menos de --zona-morta-cm do ponto segurado = ficar onde está
                        # (alvo parado -> o integrador corrige o peso); só um pedido maior move a mão
                        zm = a.zona_morta_cm[0 if l == "left" else 1] / 100
                        if zm > 0:
                            if segurado[l] is not None and np.linalg.norm(v[:3] - segurado[l][0]) < zm:
                                v = segurado[l][1]
                            else:
                                segurado[l] = (v[:3].copy(), v.copy())
                        vs[l] = v
                        # alvo = onde a IA quer a mão (absoluto, pelvis) + compensação FIXA do peso;
                        # com escala < 1, só essa fração do caminho a partir da mão medida
                        desejo = pm + a.escala * (v[:3] - pm) + comp[l]
                        alvos_p[l] = np.clip(desejo, caixa_lo[l], caixa_hi[l])
                        if a.z_min is not None:   # piso: a mão não desce abaixo da mesa (z na pelvis)
                            alvos_p[l][2] = max(alvos_p[l][2], a.z_min / 100)
                    # as mãos não se cruzam nem se encostam: esquerda sempre >= mao_mao_min à esquerda da direita
                    if "left" not in lados_ativos:
                        alvos_p["left"] = atual["medida"]["left"][1].copy()   # parado: vale onde ele está
                    folga = alvos_p["left"][1] - alvos_p["right"][1] - a.mao_mao_min
                    if folga < 0:
                        if "left" in lados_ativos:
                            alvos_p["left"][1] -= folga / 2
                            alvos_p["right"][1] += folga / 2
                        else:
                            alvos_p["right"][1] += folga
                    for l in lados_ativos:
                        R0, p0 = atual["base"][l]
                        Rm, pm = atual["medida"][l]
                        v = vs[l]
                        p_alvo = alvos_p[l]
                        R_alvo = Rm @ expvec(a.escala * rotvec(Rm.T @ R_de(v[3:6])))   # absoluto, a partir do MEDIDO
                        if segurado[l] is not None and v is segurado[l][1]:
                            R_alvo = R_de(v[3:6])   # segurando: a orientação também não anda com a medida
                        # a mão comandada não salta: no máximo --passo-max-cm por passo
                        _, p_cmd, _ = fk.pose_jac(q_ik, l)
                        dp = p_alvo - p_cmd
                        lim = a.passo_max_cm / 100
                        if np.linalg.norm(dp) > lim:
                            p_alvo = p_cmd + dp * (lim / np.linalg.norm(dp))
                            stats["cortes"][l] += 1
                        q_ant = q_ik.copy()
                        q_ik, ep, _ = fk.ik(q_ik, l, R_alvo, p_alvo, iters=15, lo=lo_caixa, hi=hi_caixa,
                                            peso_rot=a.peso_rot, q_repouso=q_part, peso_repouso=a.peso_repouso)
                        stats["erros"][l] = max(stats["erros"][l], ep)
                        stats["fins"][l] = (v[:3], p_alvo, v[:3] - pm)
                        # limite por passo: ENCOLHE o movimento inteiro (mesma direção), não junta a junta
                        idx = BRACO[l]
                        dq = q_ik[idx] - q_ant[idx]
                        s = min(1.0, a.passo_max_rad / (np.abs(dq).max() + 1e-9))
                        q_ik[idx] = q_ant[idx] + s * dq
                        # a caixa vale para a MÃO RESULTANTE, não só para o alvo: se a IK deixou a mão
                        # mais de 2 cm fora, o passo é descartado (o braço esquerdo foi a z = 54 cm, fora dela)
                        p_res = fk.pose(q_ik, l)[1]
                        if np.any(p_res < caixa_lo[l] - 0.02) or np.any(p_res > caixa_hi[l] + 0.02):
                            q_ik[idx] = q_ant[idx]
                            stats["cortes"][l] += 1
                        for i in idx:
                            q_des[i] = float(q_ik[i])
                    if "action.left_fig6d" in acao:
                        envia_maos_fig6d({l: ponto("fig6d", l) for l in ("left", "right")})
                    else:
                        envia_maos({l: float(acao[f"action.{l}_gripper"][0][k, 0]) for l in ("left", "right")})
                    narra(atual["frase"], fech_mao)
                    usados += 1
            aplica_integrador()
            # TEMPERATURA da cintura (02/10: o roll/pitch esquentou e a cintura cedeu no meio da tarefa)
            if tc - t_temp[0] > 2.0:
                t_temp[0] = tc
                temps = [_temp(est.low["motor_state"][i]) for i in CINTURA[1:]]
                if a.temp_tronco > 0 and max(temps) >= a.temp_tronco and cabine.tarefa()[0]:
                    print(f"\n🌡️  cintura roll/pitch a {temps} °C (limite {a.temp_tronco:.0f}): SEGURANDO a posição. "
                          "Deixe esfriar; mande a tarefa de novo na cabine.", flush=True)
                    cabine.define_tarefa("")
                elif max(temps) >= (a.temp_tronco - 8 if a.temp_tronco > 0 else 60.0) and tc - t_temp[1] > 30:
                    t_temp[1] = tc
                    print(f"\n⚠️  cintura roll/pitch esquentando: {temps} °C", flush=True)
            envia()   # sempre: sem trecho novo, segura o último alvo (o arm_sdk nunca fica mudo)
            time.sleep(max(0.0, dt - (time.time() - tc)))
        if atual is not None:
            _registra(atual, stats, log, t_ini, alvo, q29(), cabine, a, est, partida, fech_mao, usados)
        print(f"\n✅ fim | {atual['n'] if atual else 0} consultas | log {a.log}", flush=True)
    except KeyboardInterrupt:
        print("\n⏹ cancelado pelo operador.", flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"\n❌ erro: {type(e).__name__}: {e}", flush=True)
    finally:
        pare.set()
        arq.close()
        if grav is not None:
            grav.fecha()
            print(f"💾 gravação: {grav.pasta}  (relatório: python relatorio_wla_real.py {grav.pasta})", flush=True)
        panico_software("fim do teste do WLA — a ponte segura o braço parado")


def _registra(tr, st, log, t_ini, alvo, qm, cabine, a, est, partida, fech_mao, usados):
    """Painel e log de um trecho, quando ele é substituído pelo próximo."""
    fins, ee, base = st["fins"], tr["ee"], tr["base"]
    if not fins:
        print(f"   consulta {tr['n']:3d} | {tr['ms']:4.0f} ms | chegou tarde demais (nenhum passo usado)", flush=True)
        return
    cabine.publica_quadro("wla_trajetoria", sw.desenha_trajetoria(ee, tr["acao"], tr["garras"]))
    info = {"modo": "ENSAIO (nada enviado)" if a.ensaio else "REAL — braços + mãos",
            "consulta": tr["n"], "acao_ms": round(tr["ms"]), "tarefa_no_modelo": tr["frase"], "escala": a.escala,
            "caixa_rad": a.caixa, "hz": a.hz, "passo_inicial": st["passo_inicial"], "passos_usados": usados,
            "trava_do_robo": est.panico,
            "delta_pedido_pelo_modelo_cm": {l: np.round(fins[l][2] * 100, 1).tolist() for l in fins},
            "delta_executado_cm": {l: np.round((fins[l][1] - base[l][1]) * 100, 1).tolist() for l in fins},
            "queda_medido_vs_comandado_cm": {l: round(float(np.linalg.norm(ee[l][:3] - base[l][1])) * 100, 1)
                                             for l in fins},
            "erro_ik_mm": {l: round(st["erros"][l] * 1000, 1) for l in st["erros"]},
            "passos_limitados": st["cortes"],
            "maos_fechamento_0a1": {l: round(v, 2) for l, v in fech_mao.items() if v is not None},
            "maior_desvio_da_partida_rad": round(max(abs(alvo[i] - partida[i]) for i in BRACOS), 3)}
    cabine.publica_estado(info)
    for l in fins:
        log.writerow([round(time.time() - t_ini, 2), tr["n"], round(tr["ms"]), l, np.round(ee[l][:3], 4).tolist(),
                      np.round(base[l][1], 4).tolist(), np.round(fins[l][0], 4).tolist(),
                      np.round(fins[l][1], 4).tolist(), round(st["erros"][l] * 1000, 1), st["cortes"][l],
                      [round(alvo[i], 3) for i in BRACO[l]], [round(float(qm[i]), 3) for i in BRACO[l]],
                      st["passo_inicial"]])
    if tr["n"] % 5 == 1:
        print(f"   consulta {tr['n']:3d} | {tr['ms']:4.0f} ms | passos {st['passo_inicial']}..+{usados} | modelo Δ dir "
              f"{info['delta_pedido_pelo_modelo_cm'].get('right')} cm esq {info['delta_pedido_pelo_modelo_cm'].get('left')}"
              f" cm | até {info['maior_desvio_da_partida_rad']:.2f} rad da partida | mãos fecham "
              f"{info['maos_fechamento_0a1']}"
              f"{' | tronco yaw pedido %+.2f medido %+.2f rad' % (st['yaw'], qm[12]) if 'yaw' in st else ''}", flush=True)


if __name__ == "__main__":
    main()
    # Sai sem desmontar CUDA/threads do DDS-ZMQ: o encerramento normal dava "Segmentation fault" DEPOIS
    # do pânico por software (inofensivo, mas assusta).
    sys.stdout.flush()
    os._exit(0)
