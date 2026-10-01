#!/usr/bin/env python
"""
CÂMERAS NO FORMATO DO DATASET DO WLA — cabeça estéreo (ZED) + punhos (RealSense), 640x480, 30 fps
===============================================================================================
Uma mensagem por quadro na porta 5555 (mesmo formato ZMQ/JSON/JPEG-base64 do
`full_realsenser_server.py`, array RGB codificado como lá), com as chaves do dataset:

  head_stereo_left   ZED (Stereolabs, UVC): olho ESQUERDO, cru (sem retificar), recorte central
  head_stereo_right  4:3 de 1280x720 -> 640x480. É a câmera de cabeça do WLA (config nova:
                     `head_stereo_left` cru, não o `_rec`).
  wrist_left         RealSense do punho esquerdo, cor 640x480
  wrist_right        RealSense do punho direito (141722078588), cor 640x480
  head_camera        a D435i da cabeça, cor 640x480 (quem já lê a head_camera segue igual)

Cada câmera roda na sua thread e é opcional: a que não estiver ligada (ou cair) só some da
mensagem, e volta sozinha quando reconectar. O publicador manda o quadro mais novo de cada uma.

    python cameras_wla_server.py                       # tudo que achar
    python cameras_wla_server.py --sem-zed --sem-punhos
    python cameras_wla_server.py --zed-invertida --sem-cabeca-realsense   # como está montado no Prometheus

ZED DE CABEÇA PARA BAIXO (--zed-invertida): girar o quadro lado a lado INTEIRO em 180° antes de
separar resolve as duas coisas de uma vez — cada olho fica de pé, e as metades trocam de lugar,
então a lente que ficou à direita do robô volta a ser o head_stereo_left.

Cada mensagem leva também "servidor": versão, início, configuração e estado de cada câmera
(o painel de diagnóstico :8095 mostra, para conferir que a versão nova subiu).
"""
import argparse
import glob
import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import zmq

sys.path.insert(0, str(Path(__file__).parent))
from sim.sensor_utils import ImageUtils, SensorServer  # noqa: E402

W, H, FPS = 640, 480, 30
VERSAO = "2026-09-30: punho direito invertível (--punho-dir-invertido) + ZED largo/VGA/descarte"
# Modos da ZED (UVC, olhos lado a lado). Em 2560x720@30 ~25% dos quadros chegavam CORROMPIDOS (buffer
# incompleto: a imagem "escorrega" e mistura dois quadros) mesmo com a ZED sozinha no USB; em VGA ~6%.
# O WLA reduz tudo para 448x320, então VGA (672x376 por olho) não perde nada que ele use.
MODOS_ZED = {"vga": (1344, 376), "hd": (2560, 720)}
SERIAL_CABECA = "327122071538"
SERIAL_PUNHO_DIR = "141722078588"
SERIAL_PUNHO_ESQ = "138422074380"


def largo_4x3(img):
    """Olho INTEIRO (16:9, ~90° de FOV horizontal na ZED 1) reduzido para 640 de largura e completado
    com faixas pretas em cima/embaixo até 480. A câmera da cabeça do dataset tem ~100° (focal ≈ 280 px
    em 640, medido projetando as garras com a pose gravada em state_d435); o recorte central 4:3 deixava
    a ZED com ~73° — 1,5x de zoom em relação ao que o modelo viu no treino."""
    ih, iw = img.shape[:2]
    nh = int(round(ih * W / iw))
    img = cv2.resize(img, (W, nh), interpolation=cv2.INTER_AREA)
    topo = (H - nh) // 2
    return cv2.copyMakeBorder(img, topo, H - nh - topo, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def para_4x3(img):
    ih, iw = img.shape[:2]
    nw = int(round(ih * 4 / 3))
    if nw < iw:
        img = img[:, (iw - nw) // 2:(iw - nw) // 2 + nw]
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA) if img.shape[:2] != (H, W) else img


class Fonte:
    def __init__(self, nomes):
        self.nomes = nomes           # chaves que esta fonte publica
        self.quadros = {}            # chave -> RGB (H, W, 3)
        self.t = 0.0
        self.n = 0
        self.estado = "iniciando"
        self.lock = threading.Lock()

    def poe(self, **quadros):
        with self.lock:
            self.quadros = quadros
            self.t = time.time()
            self.n += 1

    def pega(self):
        with self.lock:
            return dict(self.quadros), self.t


class RealSense(Fonte):
    def __init__(self, nome, serial, invertida=False):
        super().__init__([nome])
        self.nome, self.serial, self.invertida = nome, serial, invertida
        threading.Thread(target=self._laco, daemon=True, name=nome).start()

    def _laco(self):
        while True:
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(self.serial)
            cfg.enable_stream(rs.stream.color, W, H, rs.format.rgb8, FPS)
            try:
                prof = pipe.start(cfg)
                for s in prof.get_device().query_sensors():
                    if s.supports(rs.option.auto_exposure_priority):
                        s.set_option(rs.option.auto_exposure_priority, 0)   # mantém 30 fps
                self.estado = "ok"
                print(f"[{self.nome}] RealSense {self.serial} ok ({W}x{H}@{FPS})", flush=True)
                while True:
                    f = pipe.wait_for_frames(timeout_ms=2000).get_color_frame()
                    if f:
                        img = np.asanyarray(f.get_data())
                        # montada de cabeça para baixo no suporte: gira 180°
                        img = cv2.rotate(img, cv2.ROTATE_180) if self.invertida else img.copy()
                        self.poe(**{self.nome: img})
            except Exception as e:  # noqa: BLE001
                if self.estado != f"erro: {e}":
                    print(f"[{self.nome}] RealSense {self.serial}: {e} — tento de novo em 3 s", flush=True)
                self.estado = f"erro: {e}"
                try:
                    pipe.stop()
                except Exception:
                    pass
                time.sleep(3)


def acha_zed():
    for p in sorted(glob.glob("/sys/class/video4linux/video*")):
        try:
            if open(os.path.join(p, "name")).read().strip().startswith("ZED") and \
                    int(open(os.path.join(p, "index")).read()) == 0:
                return "/dev/" + os.path.basename(p)
        except Exception:
            pass
    return None


def degrau(img):
    """Razão entre o maior salto de intensidade entre duas LINHAS vizinhas e a mediana dos saltos.
    Quadro corrompido (buffer UVC incompleto) tem uma costura horizontal: ~14; quadro normal: ~3."""
    # reduz só a LARGURA: reduzir a altura apaga a costura
    g = cv2.cvtColor(cv2.resize(img, (img.shape[1] // 4, img.shape[0]), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_RGB2GRAY).astype(np.int16)
    rd = np.abs(np.diff(g, axis=0)).mean(axis=1)
    return float(rd.max() / (np.median(rd) + 1e-3))


class Zed(Fonte):
    """ZED como webcam UVC: olho esquerdo | olho direito lado a lado, YUYV."""

    def __init__(self, invertida=False, modo="vga", enquadramento="4x3"):
        super().__init__(["head_stereo_left", "head_stereo_right"])
        self.enquadra = largo_4x3 if enquadramento == "largo" else para_4x3
        self.invertida = invertida
        self.w, self.h = MODOS_ZED[modo]
        self.descartados = 0
        threading.Thread(target=self._laco, daemon=True, name="zed").start()

    def _laco(self):
        while True:
            dev = acha_zed()
            if dev is None:
                if self.estado != "sem ZED":
                    print("[zed] nenhuma ZED no USB — procuro de novo a cada 3 s", flush=True)
                self.estado = "sem ZED"
                time.sleep(3)
                continue
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
            cap.set(cv2.CAP_PROP_FPS, 30)
            ok, f = cap.read()
            if not ok or f.shape[1] != self.w:
                print(f"[zed] {dev} não entregou {self.w}x{self.h} ({None if not ok else f.shape}) — tento de novo",
                      flush=True)
                cap.release()
                time.sleep(3)
                continue
            self.estado = "ok"
            print(f"[zed] {dev} ok: {self.w}x{self.h} -> dois olhos 640x480 (recorte 4:3, cru)", flush=True)
            falhas, seguidos, bom = 0, 0, None
            meio = self.w // 2
            while falhas < 30:
                ok, f = cap.read()
                if not ok:
                    falhas += 1
                    continue
                falhas = 0
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                # Descarta quadro corrompido: costura horizontal forte E diferente do último quadro bom.
                # No máximo 5 seguidos, para uma mudança de cena de verdade nunca congelar a imagem.
                if bom is not None and seguidos < 5 and degrau(rgb) > 6.0 and \
                        np.abs(rgb[::8, ::8].astype(np.int16) - bom[::8, ::8]).mean() > 8:
                    self.descartados += 1
                    seguidos += 1
                    continue
                seguidos, bom = 0, rgb
                if self.invertida:
                    rgb = cv2.rotate(rgb, cv2.ROTATE_180)   # olhos de pé E trocados de lado
                self.poe(head_stereo_left=self.enquadra(rgb[:, :meio]), head_stereo_right=self.enquadra(rgb[:, meio:]))
            print("[zed] parou de entregar quadros — reabrindo", flush=True)
            self.estado = "reabrindo"
            cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--porta", type=int, default=5555)
    ap.add_argument("--sem-zed", action="store_true")
    ap.add_argument("--sem-punhos", action="store_true")
    ap.add_argument("--sem-cabeca-realsense", action="store_true")
    ap.add_argument("--zed-modo", choices=list(MODOS_ZED), default="vga",
                    help="vga = 1344x376 (padrão, poucos quadros corrompidos); hd = 2560x720")
    ap.add_argument("--zed-enquadramento", choices=["4x3", "largo"], default="4x3",
                    help="4x3 = recorte central (~73°); largo = olho inteiro (~90°) com faixas pretas (mais perto "
                         "dos ~100° da câmera do dataset)")
    ap.add_argument("--zed-invertida", action="store_true",
                    help="ZED montada de cabeça para baixo: gira 180° e troca os olhos")
    ap.add_argument("--punho-esq", default=SERIAL_PUNHO_ESQ)
    ap.add_argument("--punho-dir", default=SERIAL_PUNHO_DIR)
    ap.add_argument("--punho-dir-invertido", action="store_true",
                    help="câmera do punho direito montada de cabeça para baixo: gira 180°")
    ap.add_argument("--punho-esq-invertido", action="store_true")
    a = ap.parse_args()

    fontes = []
    if not a.sem_cabeca_realsense:
        fontes.append(RealSense("head_camera", SERIAL_CABECA))
    if not a.sem_zed:
        fontes.append(Zed(a.zed_invertida, a.zed_modo, a.zed_enquadramento))
    if not a.sem_punhos:
        fontes += [RealSense("wrist_left", a.punho_esq, a.punho_esq_invertido),
                   RealSense("wrist_right", a.punho_dir, a.punho_dir_invertido)]

    inicio = time.time()
    config = {"punho_dir_invertido": a.punho_dir_invertido, "punho_esq_invertido": a.punho_esq_invertido,
              "zed_modo": a.zed_modo, "zed_enquadramento": a.zed_enquadramento, "zed_invertida": a.zed_invertida, "zed": not a.sem_zed, "punhos": not a.sem_punhos,
              "cabeca_realsense": not a.sem_cabeca_realsense, "resolucao": f"{W}x{H}@{FPS}"}
    print(f"[câmeras WLA] versão {VERSAO} | {config}", flush=True)
    server = SensorServer()
    server.start_server(port=a.porta)

    def envia_quieto(dados):
        """O send_message do SensorServer imprime a cada 100 mensagens (a cada 3 s aqui): poluía o init."""
        try:
            server.socket.send_multipart([json.dumps(dados).encode("utf-8")], flags=zmq.NOBLOCK)
        except zmq.Again:
            pass
    print(f"[câmeras WLA] publicando na {a.porta} a {FPS} Hz: "
          f"{', '.join(n for f in fontes for n in f.nomes)}", flush=True)
    periodo, ult_log = 1.0 / FPS, time.time()
    enviados = {}
    while True:
        t0 = time.time()
        imagens, tempos = {}, {}
        for f in fontes:
            q, t = f.pega()
            if q and t0 - t < 1.0:          # quadro velho (câmera caiu) não vai
                for k, v in q.items():
                    imagens[k] = ImageUtils.encode_image(v)
                    tempos[k] = t
                    enviados[k] = enviados.get(k, 0) + 1
        status = {"versao": VERSAO, "iniciado": inicio, "config": config,
                  "cameras": {type(f).__name__ + ":" + "+".join(f.nomes):
                              f.estado + (f" ({f.descartados} quadros corrompidos descartados)"
                                          if isinstance(f, Zed) else "") for f in fontes}}
        envia_quieto({"images": imagens, "timestamps": tempos, "servidor": status})
        if t0 - ult_log > 60:
            print("[câmeras WLA] último minuto: " + ", ".join(f"{k} {v / (t0 - ult_log):.0f} fps"
                                                           for k, v in sorted(enviados.items())), flush=True)
            enviados, ult_log = {}, t0
        time.sleep(max(0.0, periodo - (time.time() - t0)))


if __name__ == "__main__":
    main()
