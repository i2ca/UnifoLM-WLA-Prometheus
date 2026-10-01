import pyrealsense2 as rs
import numpy as np
import cv2
import zmq
import time
import sys
from pathlib import Path

# Garante que o Python ache o utils do seu projeto
sys.path.insert(0, str(Path(__file__).parent))
from sim.sensor_utils import SensorServer, ImageUtils

def start_real_robot_cameras():
    # ==========================================================
    # CONFIGURAÇÕES DE RESOLUÇÃO
    # ==========================================================
    HEAD_WIDTH, HEAD_HEIGHT = 848, 480
    DEPTH_WIDTH, DEPTH_HEIGHT = 848, 480 
    FPS = 30

    # ==========================================================
    # 1. INICIALIZA A INTEL REALSENSE D435i
    # ==========================================================
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device('327122071538') # Serial do Prometheus

    config.enable_stream(rs.stream.color, HEAD_WIDTH, HEAD_HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)

    try:
        profile = pipeline.start(config)
        
        # ==========================================================
        # 🚀 O PULO DO GATO: FORÇAR 30 FPS (DESATIVAR PRIORIDADE DE EXPOSIÇÃO)
        # ==========================================================
        # Pegamos o sensor de cor (geralmente índice 1)
        color_sensor = profile.get_device().query_sensors()[1]
        if color_sensor.supports(rs.option.auto_exposure_priority):
            # 0 desativa a prioridade, forçando a câmera a manter os 30 FPS constantes
            color_sensor.set_option(rs.option.auto_exposure_priority, 0)
        
        # Fator para converter as unidades cruas do stream z16 em MILÍMETROS.
        # A D435i costuma vir com depth_scale=0.001 (1 unidade = 1 mm), mas isso é
        # calibração de fábrica, não garantia: quem grava depende do número certo,
        # porque o LeRobot lê inteiro como milímetro.
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        depth_units_to_mm = depth_scale * 1000.0

        print(f"[RealSense D435i] Iniciada com sucesso a {FPS} FPS fixos!")
        print(f"[RealSense D435i] depth_scale={depth_scale} → 1 unidade = {depth_units_to_mm:.4f} mm")
    except Exception as e:
        print(f"[Erro RealSense] {e}")
        return

    server = SensorServer()
    server.start_server(port=5555)

    align_to = rs.stream.color
    align = rs.align(align_to)

    print("[ZMQ] Servidor de Visão ativo na porta 5555. Aguardando LeRobot...")

    try:
        # Loop otimizado
        while True:
            # Reduzimos o timeout para 1000ms para o script não ficar "preso"
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # --- TRATAMENTO RGB ---
            img_bgr = np.asanyarray(color_frame.get_data())
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # --- TRATAMENTO DEPTH (formato nativo do LeRobot 0.6.1) ---
            # Profundidade sai CRUA: uint16 em milímetros, 1 canal, sem clip e sem
            # escalar para 8 bits. Era esse achatamento que jogava a medida fora —
            # 2000 mm espremidos em 256 níveis dão 7,8 mm por degrau, e depois o
            # JPEG ainda borrava as bordas do que sobrou.
            #
            # A 0.6.1 sabe guardar profundidade de verdade: declarada com 1 canal,
            # ela vai para o `DepthEncoderConfig` (HEVC gray12le lossless), que
            # quantiza em log com 4096 níveis — ~1 mm de resolução a meio metro,
            # que é a resolução do próprio sensor. Inteiro é lido como milímetro
            # (`infer_depth_unit`), então a unidade aqui não é detalhe.
            depth_raw = np.asanyarray(depth_frame.get_data())
            if abs(depth_units_to_mm - 1.0) > 1e-6:
                depth_mm = (depth_raw.astype(np.float32) * depth_units_to_mm).astype(np.uint16)
            else:
                depth_mm = depth_raw

            # --- ENVIO ---
            descritor_depth, buffer_depth = ImageUtils.encode_raw(depth_mm, part=1)
            current_time = time.time()
            message = {
                "images": {
                    "head_camera": ImageUtils.encode_image(img_rgb),
                    # Fora do JSON: 16 bits não sobrevivem a JPEG nem a base64 sem
                    # inchar. Vai crua, em quadro binário próprio da mensagem ZMQ
                    # (índice 1; o 0 é sempre o JSON) — ver ImageUtils.encode_raw.
                    "head_camera_depth": descritor_depth,
                },
                "timestamps": {
                    "head_camera": current_time,
                    "head_camera_depth": current_time,
                }
            }
            server.send_message(message, parts=[buffer_depth])

    except KeyboardInterrupt:
        print("\nEncerrando transmissão...")
    finally:
        pipeline.stop()
        server.stop_server()

if __name__ == "__main__":
    start_real_robot_cameras()