#!/usr/bin/env python
"""
Imprime os intrínsecos reais da RealSense, no formato do YAML de treino.

Rode isto **no robô** (onde a câmera está plugada), não na atena.

    python Scripts_Prometheus_int/print_camera_intrinsics.py

Por que isso importa
────────────────────
`depth_to_pointcloud` projeta cada pixel de profundidade no espaço 3D usando o
modelo pinhole:

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

Se `cx`/`cy` não forem o ponto principal REAL da câmera, a nuvem de pontos sai
cisalhada: o objeto aparece deslocado em X/Y proporcionalmente à distância. O
encoder 3D aprende a compensar isso no treino, mas o erro volta assim que a cena
muda — é exatamente o tipo de coisa que faz o modelo funcionar no dataset e
falhar no robô.

O `full_realsenser_server.py` grava a 848×480 com `align_to = rs.stream.color`,
então os intrínsecos corretos são os do **stream de COR** nessa resolução — não
os do stream de profundidade.
"""

import pyrealsense2 as rs

WIDTH, HEIGHT, FPS = 848, 480, 30
SERIAL = "327122071538"  # Prometheus — mesmo serial do full_realsenser_server.py


def main() -> None:
    pipeline = rs.pipeline()
    config = rs.config()
    try:
        config.enable_device(SERIAL)
    except Exception:
        print(f"[aviso] serial {SERIAL} não encontrado — usando a primeira câmera disponível.")

    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    profile = pipeline.start(config)
    try:
        for stream in (rs.stream.color, rs.stream.depth):
            intr = profile.get_stream(stream).as_video_stream_profile().get_intrinsics()
            name = "COLOR" if stream == rs.stream.color else "DEPTH"
            print(f"\n─── {name} {intr.width}×{intr.height} ───")
            print(f"  fx={intr.fx:.2f}  fy={intr.fy:.2f}  cx={intr.ppx:.2f}  cy={intr.ppy:.2f}")
            print(f"  modelo={intr.model}  coeffs={[round(c, 5) for c in intr.coeffs]}")

        color = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        depth_sensor = profile.get_device().first_depth_sensor()

        print("\n" + "=" * 62)
        print("Cole no YAML de treino (o depth é alinhado à COR, então use estes):")
        print("=" * 62)
        print("  camera_intrinsics:")
        print(f"    fx: {color.fx:.1f}")
        print(f"    fy: {color.fy:.1f}")
        print(f"    cx: {color.ppx:.1f}")
        print(f"    cy: {color.ppy:.1f}")
        print("=" * 62)
        print(f"\ndepth_scale do sensor: {depth_sensor.get_depth_scale()} m/unidade")
        print(
            "Confira que o clip de 2000 mm do full_realsenser_server.py bate com o "
            "fator 2.0 usado em depth_to_pointcloud."
        )
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
