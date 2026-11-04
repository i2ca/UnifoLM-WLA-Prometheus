#!/usr/bin/env python
"""SERVIDOR DO LIDAR da cabeça (Livox MID-360) no Jetson do G1 (03/11).

Liga o lidar pelo Livox-SDK2 (ctypes na liblivox_lidar_sdk_shared.so, o mesmo do ~/Livox-SDK2) e republica a nuvem
por ZMQ, no mesmo jeito das câmeras (cameras_wla_server.py na 5555):

    tcp://<robô>:5558   PUB, a cada ~0,1 s um envio multipart [cabeçalho JSON, pontos]
        cabeçalho  {"t": time.time() do Jetson no fim do lote, "t0": do início, "n": pontos, "sn": ..., "pacotes": ...}
        pontos     float32 (n, 4): x, y, z em METROS no referencial do lidar (mid360_link do URDF) e refletividade

    tcp://<robô>:5559   PUB, 03/11: o IMU do lidar (ICM40609 dentro do MID-360), média a cada ~20 ms, JSON
        {"t": ..., "n": amostras, "acc": [x, y, z] em g, "gyro": [x, y, z] rad/s}   (referencial do lidar)

O lidar só recebe o destino (host_ip) e o modo de trabalho quando o SDK sobe — sem este servidor ele fica mudo.
Nada aqui mexe no robô.
"""
import argparse
import ctypes
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import zmq

CABECALHO = 36                         # LivoxLidarEthernetPacket até data[] (pack 1)
ALTO = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("r", "u1"), ("tag", "u1")])   # data_type 1, mm
BAIXO = np.dtype([("x", "<i2"), ("y", "<i2"), ("z", "<i2"), ("r", "u1"), ("tag", "u1")])  # data_type 2, cm
CB = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_uint8, ctypes.c_void_p, ctypes.c_void_p)
IMU = np.dtype([("gx", "<f4"), ("gy", "<f4"), ("gz", "<f4"), ("ax", "<f4"), ("ay", "<f4"), ("az", "<f4")])


def config(host_ip, base):
    c = json.load(open(base))
    for h in c["MID360"]["host_net_info"]:
        h["host_ip"] = host_ip
    out = Path.home() / ".cache" / "lidar_mid360_g1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(c, open(out, "w"), indent=2)
    return str(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--porta", type=int, default=5558)
    ap.add_argument("--host-ip", default="192.168.123.164", help="IP do Jetson na rede do robô")
    ap.add_argument("--periodo", type=float, default=0.1, help="s entre envios")
    ap.add_argument("--porta-imu", type=int, default=5559)
    ap.add_argument("--lib", default="/usr/local/lib/liblivox_lidar_sdk_shared.so")
    ap.add_argument("--config", default=str(Path.home() / "Livox-SDK2/samples/livox_lidar_quick_start/mid360_config.json"))
    ap.add_argument("--sdk-log", type=int, default=0, help="1 = deixa as mensagens do Livox-SDK2")
    a = ap.parse_args()

    sdk = ctypes.CDLL(a.lib)
    sdk.LivoxLidarSdkInit.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
    sdk.LivoxLidarSdkInit.restype = ctypes.c_bool
    sdk.LivoxLidarSdkStart.restype = ctypes.c_bool
    sdk.SetLivoxLidarPointCloudCallBack.argtypes = [CB, ctypes.c_void_p]
    sdk.SetLivoxLidarImuDataCallback.argtypes = [CB, ctypes.c_void_p]
    sdk.EnableLivoxLidarImuData.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]

    trava, lote, est = threading.Lock(), [], {"pacotes": 0, "t0": time.time()}

    imu_lote, imu_trava = [], threading.Lock()

    def chegou_imu(_handle, _dev, ptr, _cli):
        try:
            cab = ctypes.string_at(ptr, CABECALHO)
            dots = max(1, int.from_bytes(cab[5:7], "little"))
            p = np.frombuffer(ctypes.string_at(ptr + CABECALHO, dots * IMU.itemsize), IMU)
            with imu_trava:
                imu_lote.append(np.stack([p["ax"], p["ay"], p["az"], p["gx"], p["gy"], p["gz"]], 1))
        except Exception:  # noqa: BLE001
            pass

    def chegou(_handle, _dev, ptr, _cli):
        est["handle"] = _handle
        try:
            cab = ctypes.string_at(ptr, CABECALHO)
            dots, tipo = int.from_bytes(cab[5:7], "little"), cab[10]
            dt = ALTO if tipo == 1 else BAIXO if tipo == 2 else None
            if dt is None or dots == 0:
                return
            p = np.frombuffer(ctypes.string_at(ptr + CABECALHO, dots * dt.itemsize), dt)
            esc = 0.001 if tipo == 1 else 0.01
            xyz = np.stack([p["x"] * esc, p["y"] * esc, p["z"] * esc, p["r"].astype(np.float32)], 1).astype(np.float32)
            xyz = xyz[(p["x"] != 0) | (p["y"] != 0) | (p["z"] != 0)]
            with trava:
                lote.append(xyz)
                est["pacotes"] += 1
        except Exception:  # noqa: BLE001  (o callback do SDK nunca pode levantar)
            pass

    if not a.sdk_log:   # 03/11: o SDK escreve "Handle detection data" 2x/s para sempre no stdout: vai para /dev/null
        sys.stdout = os.fdopen(os.dup(1), "w", buffering=1)
        os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
    cb, cb_imu = CB(chegou), CB(chegou_imu)   # referências vivas: o SDK guarda só o ponteiro
    if not sdk.LivoxLidarSdkInit(config(a.host_ip, a.config).encode(), b"", None):
        raise SystemExit("Livox-SDK2 não iniciou (config/IP?)")
    sdk.SetLivoxLidarPointCloudCallBack(cb, None)
    sdk.SetLivoxLidarImuDataCallback(cb_imu, None)
    sdk.LivoxLidarSdkStart()

    pub = zmq.Context().socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 5)
    pub.bind(f"tcp://0.0.0.0:{a.porta}")
    pub_imu = zmq.Context.instance().socket(zmq.PUB)
    pub_imu.setsockopt(zmq.SNDHWM, 20)
    pub_imu.bind(f"tcp://0.0.0.0:{a.porta_imu}")

    def laco_imu():
        pediu, n_tot, t0 = False, 0, time.time()
        while True:
            time.sleep(0.02)
            with imu_trava:
                pedacos, imu_lote[:] = list(imu_lote), []
            if pedacos:
                X = np.concatenate(pedacos)
                n_tot += len(X)
                m = X.mean(0)
                pub_imu.send_string(json.dumps({"t": time.time(), "n": int(len(X)), "acc": [round(float(v), 5) for v in m[:3]],
                                                "gyro": [round(float(v), 5) for v in m[3:]]}))
            elif not pediu and time.time() - t0 > 3 and "handle" in est:   # sem IMU: pede ao lidar
                sdk.EnableLivoxLidarImuData(est["handle"], None, None)
                print("   (IMU do lidar: pedi para ligar)", flush=True)
                pediu = True
            if time.time() - t0 > 30:
                print(f"   IMU do lidar: {n_tot / (time.time() - t0):.0f} amostras/s", flush=True)
                n_tot, t0 = 0, time.time()

    threading.Thread(target=laco_imu, daemon=True, name="imu").start()
    print(f"📡 lidar MID-360 -> tcp://0.0.0.0:{a.porta} (host {a.host_ip}); IMU -> :{a.porta_imu}", flush=True)
    ultimo_aviso, total = time.time(), 0
    try:
        while True:
            time.sleep(a.periodo)
            with trava:
                pedacos, lote[:] = list(lote), []
                t0, est["t0"] = est["t0"], time.time()
                npac = est["pacotes"]
            if not pedacos:
                if time.time() - ultimo_aviso > 5:
                    print("   (lidar: sem pontos há 5 s)", flush=True)
                    ultimo_aviso = time.time()
                continue
            P = np.concatenate(pedacos)
            total += len(P)
            cab = {"t": time.time(), "t0": t0, "n": int(len(P)), "pacotes": npac}
            pub.send_multipart([json.dumps(cab).encode(), P.tobytes()], copy=False)
            if time.time() - ultimo_aviso > 30:
                print(f"   lidar: {total / 30:.0f} pontos/s", flush=True)
                ultimo_aviso, total = time.time(), 0
    except KeyboardInterrupt:
        pass
    finally:
        sdk.LivoxLidarSdkUninit()


if __name__ == "__main__":
    main()
