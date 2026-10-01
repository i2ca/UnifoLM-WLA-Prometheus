import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sim.sensor_utils import SensorServer, ImageUtils


def start_realsense_zmq(port: int, serial: str | None, fps: int, enable_depth: bool) -> None:
    pipeline = rs.pipeline()
    cfg = rs.config()

    if serial:
        cfg.enable_device(serial)

    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
    if enable_depth:
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)

    try:
        profile = pipeline.start(cfg)
        if enable_depth:
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()  # m/unit (D435i: ~0.001)
            print(f"[RealSense D435i] Camera iniciada. depth_scale={depth_scale}")
        else:
            depth_scale = None
            print("[RealSense D435i] Camera iniciada em RGB-only.")
    except Exception as e:
        print(f"[Erro] {e}")
        return

    server = SensorServer()
    server.start_server(port=port)
    mode = "RGB + depth" if enable_depth else "RGB-only"
    print(f"[ZMQ] Servidor ativo na porta {port} ({mode})")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            rgb = np.asanyarray(color_frame.get_data())
            if enable_depth:
                frames = rs.align(rs.stream.color).process(frames)
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue

                # Profundidade crua: uint16 em MILÍMETROS, 1 canal. É o formato
                # nativo de profundidade do LeRobot 0.6.1 — ver a nota longa no
                # full_realsenser_server.py, que é o servidor de cabeça em uso.
                depth_raw = np.asanyarray(depth_frame.get_data())
                depth_units_to_mm = depth_scale * 1000.0
                if abs(depth_units_to_mm - 1.0) > 1e-6:
                    depth_mm = (depth_raw.astype(np.float32) * depth_units_to_mm).astype(np.uint16)
                else:
                    depth_mm = depth_raw

                descritor_depth, buffer_depth = ImageUtils.encode_raw(depth_mm, part=1)
                message = {
                    "images": {
                        "head_camera": ImageUtils.encode_image(rgb),
                        # Crua, em quadro binário próprio (índice 1).
                        "head_camera_depth": descritor_depth,
                    },
                    "timestamps": {
                        "head_camera": time.time(),
                        "head_camera_depth": time.time(),
                    },
                }
                depth_parts = [buffer_depth]
            else:
                message = {
                    "images": {
                        "head_camera": ImageUtils.encode_image(rgb)
                    },
                    "timestamps": {
                        "head_camera": time.time()
                    }
                }
                depth_parts = None

            server.send_message(message, parts=depth_parts)

    except KeyboardInterrupt:
        print("Encerrando...")
    finally:
        pipeline.stop()
        server.stop_server()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start RealSense ZMQ server for the Unitree G1 Prometheus camera.")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ server port")
    parser.add_argument("--serial", default=None, help="Realsense serial number if multiple devices are connected")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS")
    parser.add_argument("--no-depth", action="store_true", help="Publish only RGB and disable depth.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_realsense_zmq(port=args.port, serial=args.serial, fps=args.fps, enable_depth=not args.no_depth)
