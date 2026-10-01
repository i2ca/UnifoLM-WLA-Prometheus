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

def start_right_wrist_camera():
    # ==========================================================
    # CONFIGURAÇÕES DE RESOLUÇÃO
    # ==========================================================
    # Menor modo NATIVO da D435 (224x224 não existe como stream da câmera).
    # Capturamos aqui e entregamos 224x224 quadrado — ver OUT_SIZE abaixo.
    WIDTH, HEIGHT = 424, 240
    FPS = 30

    # Tamanho publicado. 224 é exatamente o que a torre visual do OpenVLA
    # consome: publicar maior só gastaria banda e disco, porque o
    # `_preprocess_images` redimensiona tudo para 224x224 antes do modelo.
    #
    # O corte é CENTRAL e quadrado antes do resize. Sem isso, um quadro 424x240
    # (16:9) seria espremido para quadrado e a cena apareceria deformada — e
    # deformada de um jeito que NÃO acontece no MuJoCo, onde a câmera já é
    # quadrada. Treinar com uma geometria e inferir com outra é erro silencioso.
    OUT_SIZE = 224

    # ==========================================================
    # 1. INICIALIZA A INTEL REALSENSE D435 (ENTIDADE DE STREAM)
    # ==========================================================
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device('141722078588') # Serial do pulso direito

    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    try:
        profile = pipeline.start(config)
        
        # Desativa a prioridade de exposição automática para manter os 30 FPS constantes
        for sensor in profile.get_device().query_sensors():
            if sensor.supports(rs.option.auto_exposure_priority):
                sensor.set_option(rs.option.auto_exposure_priority, 0)
                break
        
        print(f"[RealSense D435] Iniciada com sucesso a {FPS} FPS fixos!")
    except Exception as e:
        print(f"[Erro RealSense] {e}")
        return

    # ==========================================================
    # 2. PUBLICA VIA ZMQ PARA CONSUMO DO MODELO VLA
    # ==========================================================
    server = SensorServer()
    server.start_server(port=5556)

    print("[ZMQ] Servidor de Visão ativo na porta 5556. Aguardando LeRobot...")

    try:
        # Loop otimizado
        while True:
            # Reduzimos o timeout para 1000ms para o script não ficar "preso"
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue

            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # --- TRATAMENTO RGB ---
            img_bgr = np.asanyarray(color_frame.get_data())
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Corte central quadrado e depois resize — preserva a geometria.
            h, w = img_rgb.shape[:2]
            lado = min(h, w)
            y0, x0 = (h - lado) // 2, (w - lado) // 2
            img_rgb = cv2.resize(
                img_rgb[y0:y0 + lado, x0:x0 + lado],
                (OUT_SIZE, OUT_SIZE),
                interpolation=cv2.INTER_AREA,
            )

            # --- ENVIO VIA ZMQ ---
            current_time = time.time()
            message = {
                "images": {
                    "right_wrist_camera": ImageUtils.encode_image(img_rgb),
                },
                "timestamps": {
                    "right_wrist_camera": current_time,
                }
            }
            server.send_message(message)

    except KeyboardInterrupt:
        print("\nEncerrando transmissão...")
    finally:
        pipeline.stop()
        server.stop_server()

if __name__ == "__main__":
    start_right_wrist_camera()
