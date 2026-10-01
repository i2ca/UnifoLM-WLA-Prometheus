#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║         MOCK G1 CAMERAS — Simulador da Intel RealSense D435i     ║
╠══════════════════════════════════════════════════════════════════╣
║  Substitui o servidor de câmera real (start_real_robot_cameras)  ║
║  Publica frames sintéticos RGB + Depth na porta 5555,            ║
║  no mesmo formato que o LeRobot espera via ZMQ.                  ║
╠══════════════════════════════════════════════════════════════════╣
║  USO:                                                            ║
║    python mock_g1_cameras.py                    # padrão 30 FPS  ║
║    python mock_g1_cameras.py --fps 15           # mais leve      ║
║    python mock_g1_cameras.py --scene checkers   # cena alternativa║
║    python mock_g1_cameras.py --show             # janela OpenCV  ║
╠══════════════════════════════════════════════════════════════════╣
║  Cenas disponíveis (--scene):                                    ║
║    gradient   Gradiente colorido animado (padrão)                ║
║    checkers   Tabuleiro preto e branco animado                   ║
║    noise      Ruído aleatório (stress test do codec)             ║
║    solid      Frame sólido simples (mais leve)                   ║
╠══════════════════════════════════════════════════════════════════╣
║  Porta: 5555  (idêntica ao servidor real)                        ║
║  Formato: RGB em JSON+base64 JPEG; profundidade uint16 (mm) crua ║
║  em quadro binário — igual ao SensorServer real                  ║
╚══════════════════════════════════════════════════════════════════╝

Dependências: pyzmq numpy opencv-python
    pip install pyzmq numpy opencv-python
"""

import argparse
import base64
import json
import math
import signal
import sys
import time

import cv2
import numpy as np
import zmq

# ─────────────────────────── Configuração ─────────────────────────────────────
WIDTH  = 640
HEIGHT = 480

JPEG_QUALITY = 85  # qualidade da compressão — igual ao servidor real


# ─────────────────────────── Geração de Frames ────────────────────────────────

def make_rgb_gradient(t: float) -> np.ndarray:
    """
    Frame RGB animado: gradiente que gira com o tempo.
    Parece um ambiente colorido real — fácil de ver se está chegando.
    """
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    # Dois gradientes senoidais que se movem em direções diferentes
    xs = np.linspace(0, 2 * math.pi, WIDTH)
    ys = np.linspace(0, 2 * math.pi, HEIGHT)
    xg, yg = np.meshgrid(xs, ys)

    r = ((np.sin(xg + t * 0.8)        + 1) * 127).astype(np.uint8)
    g = ((np.sin(yg + t * 0.5 + 1.0)  + 1) * 127).astype(np.uint8)
    b = ((np.sin(xg + yg + t * 0.3)   + 1) * 127).astype(np.uint8)

    img[:, :, 0] = r  # R
    img[:, :, 1] = g  # G
    img[:, :, 2] = b  # B

    # Adiciona timestamp legível no canto superior esquerdo
    ts = time.strftime("%H:%M:%S")
    cv2.putText(img, f"MOCK RGB  {ts}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def make_rgb_checkers(t: float) -> np.ndarray:
    """Tabuleiro animado — fácil de detectar artefatos de compressão."""
    cell = 40
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    phase = int(t * 2) % 2  # pisca a cada 0.5s
    for y in range(0, HEIGHT, cell):
        for x in range(0, WIDTH, cell):
            if (x // cell + y // cell + phase) % 2 == 0:
                img[y:y+cell, x:x+cell] = [220, 220, 220]
            else:
                img[y:y+cell, x:x+cell] = [40, 40, 40]
    cv2.putText(img, f"MOCK RGB  {time.strftime('%H:%M:%S')}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2, cv2.LINE_AA)
    return img


def make_rgb_noise(t: float) -> np.ndarray:
    """Ruído aleatório — stress test máximo do codec."""
    img = np.random.randint(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    cv2.putText(img, f"MOCK NOISE {time.strftime('%H:%M:%S')}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    return img


def make_rgb_solid(t: float) -> np.ndarray:
    """Frame sólido simples — mínimo de CPU."""
    hue = int(t * 20) % 180
    hsv = np.full((HEIGHT, WIDTH, 3), [hue, 180, 200], dtype=np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    cv2.putText(img, f"MOCK SOLID {time.strftime('%H:%M:%S')}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


RGB_SCENES = {
    "gradient": make_rgb_gradient,
    "checkers": make_rgb_checkers,
    "noise":    make_rgb_noise,
    "solid":    make_rgb_solid,
}


def make_depth_frame(t: float) -> np.ndarray:
    """
    Gera um frame de profundidade sintético no mesmo formato do servidor real:
    uint16 em MILÍMETROS, 1 canal, cru (sem clip, sem escalar para 8 bits).

    Simula um objeto se aproximando e afastando em loop — útil para testar
    se o LeRobot lê a profundidade corretamente.
    """
    # Gradiente de profundidade: fundo a 1500mm, objeto central pulsando 300–800mm
    depth_mm = np.full((HEIGHT, WIDTH), 1500, dtype=np.uint16)

    # Objeto circular no centro pulsando
    cx, cy = WIDTH // 2, HEIGHT // 2
    radius = 80
    pulse = 300 + int(250 * (0.5 + 0.5 * math.sin(t * 1.2)))  # 300–800mm

    ys, xs = np.ogrid[:HEIGHT, :WIDTH]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
    depth_mm[mask] = pulse

    # Ruído leve
    noise = np.random.randint(-10, 10, (HEIGHT, WIDTH), dtype=np.int16)
    depth_mm = np.clip(depth_mm.astype(np.int32) + noise, 0, 2000).astype(np.uint16)

    # Anotação escrita nos próprios milímetros (o texto fica a 300 mm da câmera),
    # para dar para conferir no dataset gravado que o valor chegou inteiro.
    cv2.putText(depth_mm, f"MOCK DEPTH  obj={pulse}mm  {time.strftime('%H:%M:%S')}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 300, 1, cv2.LINE_AA)

    return depth_mm


# ─────────────────────────── Codificação ──────────────────────────────────────

def encode_depth_raw(depth_mm: np.ndarray) -> tuple:
    """Buffer cru uint16 + descritor — mesmo formato do ImageUtils.encode_raw() real."""
    buf = np.ascontiguousarray(depth_mm)
    return {"part": 1, "dtype": buf.dtype.str, "shape": list(buf.shape)}, buf.tobytes()


def encode_image(img_rgb: np.ndarray) -> str:
    """
    Codifica um frame RGB em JPEG e retorna string base64.
    Mesmo formato do ImageUtils.encode_image() do servidor real.
    """
    # O servidor real usa RGB, OpenCV usa BGR internamente
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Falha ao codificar JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ─────────────────────────── Main ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Mock G1 Cameras — simula a RealSense D435i via ZMQ"
    )
    parser.add_argument("--fps",   type=int, default=30,
                        help="Frames por segundo alvo (padrão: 30)")
    parser.add_argument("--scene", choices=list(RGB_SCENES), default="gradient",
                        help="Cena sintética RGB (padrão: gradient)")
    parser.add_argument("--show",  action="store_true", default=False,
                        help="Abre janela OpenCV para ver os frames localmente")
    parser.add_argument("--port",  type=int, default=5555,
                        help="Porta ZMQ (padrão: 5555)")
    args = parser.parse_args()

    frame_dt   = 1.0 / args.fps
    rgb_fn     = RGB_SCENES[args.scene]

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         MOCK G1 CAMERAS — Simulador RealSense D435i          ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Cena    : {args.scene:<50s}║")
    print(f"║  FPS alvo: {args.fps:<50d}║")
    print(f"║  Porta   : {args.port:<50d}║")
    print(f"║  Janela  : {'SIM (feche com Q)' if args.show else 'NÃO':<50s}║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Publicando:                                                 ║")
    print("║    head_camera        640×480 RGB JPEG                       ║")
    print("║    head_camera_depth  640×480 Depth→Gray RGB JPEG            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("  Aguardando LeRobot conectar... (Ctrl+C para sair)")
    print()

    # ── ZMQ ──────────────────────────────────────────────────────────────────
    ctx  = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    # Buffer: guarda no máximo 2 frames para não acumular lag
    sock.setsockopt(zmq.SNDHWM, 2)
    sock.bind(f"tcp://0.0.0.0:{args.port}")

    # Pequena pausa para o subscriber ter tempo de conectar antes do primeiro frame
    time.sleep(0.5)
    print(f"[MOCK CAM] ✅ Socket ZMQ PUB pronto na porta {args.port}")

    # ── Contadores para log ───────────────────────────────────────────────────
    frame_count = 0
    last_log    = time.time()
    t0          = time.time()

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    running = [True]

    def _stop(sig, frame):
        print("\n[MOCK CAM] 🛑 Encerrando...")
        running[0] = False

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    # ── Loop principal ────────────────────────────────────────────────────────
    while running[0]:
        loop_start = time.time()
        t          = loop_start - t0

        # Gera frames
        rgb_frame   = rgb_fn(t)
        depth_frame = make_depth_frame(t)

        # Codifica
        try:
            rgb_b64 = encode_image(rgb_frame)
            descritor_depth, depth_part = encode_depth_raw(depth_frame)
        except Exception as e:
            print(f"[MOCK CAM] ⚠️  Erro ao codificar frame: {e}")
            time.sleep(frame_dt)
            continue

        # Monta mensagem — mesmo formato do SensorServer real
        current_time = time.time()
        message = {
            "images": {
                "head_camera":       rgb_b64,
                # Profundidade não cabe no JSON: vai crua (uint16), em quadro
                # binário próprio da mensagem ZMQ (o índice 0 é sempre o JSON).
                "head_camera_depth": descritor_depth,
            },
            "timestamps": {
                "head_camera":       current_time,
                "head_camera_depth": current_time,
            },
        }

        # Publica
        payload = json.dumps(message).encode()
        try:
            sock.send_multipart([payload, depth_part], zmq.NOBLOCK)
        except zmq.Again:
            pass  # Subscriber mais lento que o publisher — descarta frame

        frame_count += 1

        # Log periódico a cada 5s
        now = time.time()
        if now - last_log >= 5.0:
            real_fps  = frame_count / (now - t0)
            rgb_kb    = len(base64.b64decode(rgb_b64)) / 1024
            depth_kb  = len(depth_part) / 1024
            total_kbs = (rgb_kb + depth_kb) * real_fps / 1024
            print(
                f"[MOCK CAM] 📷 frame #{frame_count:>5} | "
                f"fps_real={real_fps:.1f} | "
                f"rgb={rgb_kb:.1f}KB  depth={depth_kb:.1f}KB | "
                f"banda≈{total_kbs:.2f} MB/s"
            )
            last_log = now

        # Janela de preview local
        if args.show:
            # Coloca RGB e Depth lado a lado
            # Profundidade agora é uint16 de 1 canal: normaliza só para caber na
            # janela de preview, o que vai para a rede continua sendo o cru.
            depth_vis = cv2.cvtColor(
                cv2.convertScaleAbs(depth_frame, alpha=255.0 / 2000.0), cv2.COLOR_GRAY2RGB
            )
            preview = np.hstack([rgb_frame, depth_vis])
            cv2.imshow("MOCK G1 Cameras  (Q para sair)", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Mantém o FPS alvo
        elapsed    = time.time() - loop_start
        sleep_time = max(0.0, frame_dt - elapsed)
        time.sleep(sleep_time)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if args.show:
        cv2.destroyAllWindows()
    ctx.term()
    print(f"[MOCK CAM] ✅ Finalizado após {frame_count} frames.")


if __name__ == "__main__":
    main()