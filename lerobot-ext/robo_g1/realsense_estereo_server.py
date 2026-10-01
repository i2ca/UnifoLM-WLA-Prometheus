#!/usr/bin/env python
"""
RealSense D435i da cabeça no formato da câmera de cabeça do WLA: 640x480 (4:3), 30 fps
=======================================================================================
Publica na mesma porta e no mesmo formato do `full_realsenser_server.py` (ZMQ 5555, JSON
com JPEG em base64, array RGB codificado como no servidor antigo), então quem já lê a
`head_camera` continua funcionando. Três imagens por mensagem:

  head_camera        a COLORIDA em 640x480 (a antiga era 848x480, 16:9; o WLA espreme
                     tudo para 448x320, então 16:9 chegava deformado — o dataset é 4:3)
  head_stereo_left   o par estéreo de verdade da D435i: as câmeras INFRAVERMELHAS 1 e 2,
  head_stereo_right  640x480, em tons de cinza (replicado em 3 canais), com o projetor de
                     pontos DESLIGADO (senão a cena aparece pontilhada). Campo de visão
                     mais largo que o da colorida, mais perto da câmera grande angular do G1
                     que gravou o dataset — mas SEM COR.

Sem profundidade neste modo (o projetor desligado piora a profundidade de qualquer jeito).

    python realsense_estereo_server.py            # no robô, no lugar do full_realsenser_server.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, str(Path(__file__).parent))
from sim.sensor_utils import ImageUtils, SensorServer  # noqa: E402

SERIAL = "327122071538"   # D435i da cabeça do Prometheus
W, H, FPS = 640, 480, 30


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(SERIAL)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, FPS)
    config.enable_stream(rs.stream.infrared, 2, W, H, rs.format.y8, FPS)
    profile = pipeline.start(config)
    dev = profile.get_device()
    estereo = dev.first_depth_sensor()
    if estereo.supports(rs.option.emitter_enabled):
        estereo.set_option(rs.option.emitter_enabled, 0)   # sem pontilhado no IR
    for s in dev.query_sensors():
        if s.supports(rs.option.auto_exposure_priority):
            s.set_option(rs.option.auto_exposure_priority, 0)   # mantém 30 fps
    print(f"[RealSense estéreo] cor + IR1 + IR2 em {W}x{H} @ {FPS} fps, projetor desligado", flush=True)

    server = SensorServer()
    server.start_server(port=5555)
    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue
            cor, ir1, ir2 = frames.get_color_frame(), frames.get_infrared_frame(1), frames.get_infrared_frame(2)
            if not cor or not ir1 or not ir2:
                continue
            agora = time.time()
            rgb = cv2.cvtColor(np.asanyarray(cor.get_data()), cv2.COLOR_BGR2RGB)
            e = np.repeat(np.asanyarray(ir1.get_data())[:, :, None], 3, axis=2)
            d = np.repeat(np.asanyarray(ir2.get_data())[:, :, None], 3, axis=2)
            server.send_message({
                "images": {"head_camera": ImageUtils.encode_image(rgb),
                           "head_stereo_left": ImageUtils.encode_image(e),
                           "head_stereo_right": ImageUtils.encode_image(d)},
                "timestamps": {"head_camera": agora, "head_stereo_left": agora, "head_stereo_right": agora},
            })
    except KeyboardInterrupt:
        pass
    finally:
        if estereo.supports(rs.option.emitter_enabled):
            estereo.set_option(rs.option.emitter_enabled, 1)   # devolve o projetor para os outros usos
        pipeline.stop()
        server.stop_server()


if __name__ == "__main__":
    main()
