#!/usr/bin/env python
"""
PAINEL DE DIAGNÓSTICO DO G1 — roda NO ROBÔ, junto com a ponte (init_prometheus-vla.sh)
=======================================================================================
http://192.168.123.164:8095/

Mostra tudo o que o robô publica, direto do DDS (não depende da ponte estar viva):
  - as 29 juntas do corpo: posição, velocidade, torque estimado, as DUAS temperaturas do
    motor, tensão e o código de erro do motor;
  - as duas Dex3: juntas (com temperatura), sensores de pressão, tensão e erro;
  - IMU, bateria (BMS: carga, tensão, corrente, células, temperaturas, ciclos), placa-mãe;
  - o Jetson: temperaturas, carga e memória; e quais servidores nossos estão rodando;
  - a TRAVA do botão de pânico (porta 6007 da ponte v3), com botões de rearmar e travar;
  - TODAS as câmeras das portas 5555 (cabeça, ZED estéreo, punhos) e 5556, se estiverem no ar.

O que ele MANDA ao robô: nada pelo DDS (só assina). Pela ponte, só os pedidos de rearme e de
pânico da porta 6006, e só quando alguém aperta o botão na página. O rearme continua exigindo
o cogumelo solto há >= 1 s — quem decide é a ponte, não esta página.

    python painel_diagnostico.py            # porta 8095
"""
import argparse
import dataclasses
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import zmq

CORPO = ["quadril pitch E", "quadril roll E", "quadril yaw E", "joelho E", "tornozelo pitch E",
         "tornozelo roll E", "quadril pitch D", "quadril roll D", "quadril yaw D", "joelho D",
         "tornozelo pitch D", "tornozelo roll D", "cintura yaw", "cintura roll", "cintura pitch",
         "ombro pitch E", "ombro roll E", "ombro yaw E", "cotovelo E", "punho roll E", "punho pitch E",
         "punho yaw E", "ombro pitch D", "ombro roll D", "ombro yaw D", "cotovelo D", "punho roll D",
         "punho pitch D", "punho yaw D"]
MAO = {"left": ["polegar 0", "polegar 1", "polegar 2", "médio 0", "médio 1", "indicador 0", "indicador 1"],
       "right": ["polegar 0", "polegar 1", "polegar 2", "indicador 0", "indicador 1", "médio 0", "médio 1"]}
PROCESSOS = {"ponte v3 (pânico)": "dex3_g1_server_v3_panico.py", "ponte v2": "dex3_g1_server_v2.py",
             "câmeras WLA (ZED + punhos)": "cameras_wla_server.py",
             "câmera cabeça": "full_realsenser_server.py", "câmera pulso dir": "right_arm_realsense_server.py"}


def _lista(x):
    return [float(v) for v in x] if isinstance(x, (list, tuple)) else [float(x)]


class Dados:
    def __init__(self):
        self.msg = {}
        self.t = {}
        self.n = {}
        self.lock = threading.Lock()

    def poe(self, nome, m):
        with self.lock:
            self.msg[nome] = m
            self.t[nome] = time.time()
            self.n[nome] = self.n.get(nome, 0) + 1

    def pega(self, nome):
        with self.lock:
            return self.msg.get(nome), time.time() - self.t.get(nome, 0.0)


D = Dados()
POSE = None


# ── DDS: só assinatura ─────────────────────────────────────────────────────
def sobe_dds(interface):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import BmsState_, HandState_, LowState_, MainBoardState_
    if interface:
        ChannelFactoryInitialize(0, interface)
    else:
        ChannelFactoryInitialize(0)
    subs = []
    for nome, topico, tipo in (("low", "rt/lowstate", LowState_), ("left", "rt/dex3/left/state", HandState_),
                               ("right", "rt/dex3/right/state", HandState_), ("bms", "rt/lf/bmsstate", BmsState_),
                               ("placa", "rt/lf/mainboardstate", MainBoardState_)):
        s = ChannelSubscriber(topico, tipo)
        s.Init(lambda m, n=nome: D.poe(n, m), 10)
        subs.append(s)
    return subs


# ── ZMQ: estado da ponte e câmeras ─────────────────────────────────────────
def assina_json(ctx, porta, nome):
    s = ctx.socket(zmq.SUB)
    s.setsockopt_string(zmq.SUBSCRIBE, "")
    s.setsockopt(zmq.RCVHWM, 2)
    s.connect(f"tcp://127.0.0.1:{porta}")
    while True:
        try:
            D.poe(nome, json.loads(s.recv().decode("utf-8")))
        except Exception:
            time.sleep(0.05)


class Camera:
    """Guarda o JPEG cru (base64) mais novo de CADA imagem da mensagem; só decodifica quem
    alguém está olhando."""

    def __init__(self, ctx, porta):
        self.b64, self.t, self.fps, self.tam = {}, {}, 0.0, {}
        self.servidor = None
        self.cache = {}
        s = ctx.socket(zmq.SUB)
        s.setsockopt_string(zmq.SUBSCRIBE, "")
        s.setsockopt(zmq.RCVHWM, 2)
        s.connect(f"tcp://127.0.0.1:{porta}")
        threading.Thread(target=self._laco, args=(s,), daemon=True).start()

    def _laco(self, s):
        n, t0 = 0, time.time()
        while True:
            try:
                fr = s.recv_multipart()
                msg = json.loads(fr[0].decode("utf-8"))
                imgs = msg.get("images") or {}
                if msg.get("servidor"):
                    self.servidor = msg["servidor"]
                agora = time.time()
                for k, v in imgs.items():
                    if isinstance(v, str) and "depth" not in k:
                        self.b64[k], self.t[k] = v, agora
                n += 1
                if time.time() - t0 >= 2.0:
                    self.fps, n, t0 = n / (time.time() - t0), 0, time.time()
            except Exception:
                time.sleep(0.05)

    def jpeg(self, nome):
        b64 = self.b64.get(nome)
        if b64 is None:
            return None
        if self.cache.get(nome, (None,))[0] is b64:
            return self.cache[nome][1]
        import base64
        arr = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return None
        # O servidor da câmera codifica o array RGB como se fosse BGR: troca de volta aqui.
        self.tam[nome] = arr.shape[1::-1]
        if arr.shape[1] > 640:
            arr = cv2.resize(arr, (640, int(arr.shape[0] * 640 / arr.shape[1])), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", arr[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 75])
        self.cache[nome] = (b64, buf.tobytes() if ok else None)
        return self.cache[nome][1]


# ── Jetson ─────────────────────────────────────────────────────────────────
def jetson():
    temps = {}
    for z in sorted(os.listdir("/sys/class/thermal")):
        if not z.startswith("thermal_zone"):
            continue
        try:
            nome = open(f"/sys/class/thermal/{z}/type").read().strip().replace("-therm", "")
            temps[nome] = int(open(f"/sys/class/thermal/{z}/temp").read()) / 1000.0
        except Exception:
            pass
    mem = {}
    for linha in open("/proc/meminfo"):
        k, v = linha.split(":")
        mem[k] = int(v.split()[0])
    try:
        ps = subprocess.run(["ps", "-eo", "pid,etimes,pcpu,args"], capture_output=True, text=True, timeout=2).stdout
    except Exception:
        ps = ""
    procs = {}
    for rotulo, arq in PROCESSOS.items():
        achou = [ln.split(None, 3) for ln in ps.splitlines() if arq in ln and "python" in ln]
        procs[rotulo] = ({"pid": int(achou[0][0]), "ha_s": int(achou[0][1]), "cpu": float(achou[0][2])}
                         if achou else None)
    return {"temperaturas": temps, "carga": os.getloadavg(), "nucleos": os.cpu_count(),
            "mem_usada_mb": (mem["MemTotal"] - mem["MemAvailable"]) // 1024, "mem_total_mb": mem["MemTotal"] // 1024,
            "processos": procs}


# ── resumo ─────────────────────────────────────────────────────────────────
def resumo(cams):
    r = {"agora": time.time(), "fontes": {}}
    for nome in ("low", "left", "right", "bms", "placa", "panico", "modo"):
        _, idade = D.pega(nome)
        r["fontes"][nome] = round(idade, 2) if idade < 1e6 else None

    low, _ = D.pega("low")
    if low is not None:
        j = []
        for i in range(29):
            m = low.motor_state[i]
            j.append({"i": i, "nome": CORPO[i], "q": m.q, "dq": m.dq, "tau": m.tau_est,
                      "temp": _lista(m.temperature), "vol": float(m.vol), "erro": int(m.motorstate),
                      "sensor": _lista(m.sensor)})
        imu = low.imu_state
        r["corpo"] = {"juntas": j, "tick": int(low.tick), "mode_pr": int(low.mode_pr),
                      "mode_machine": int(low.mode_machine),
                      "imu": {"rpy_graus": [float(np.degrees(v)) for v in imu.rpy], "giro": _lista(imu.gyroscope),
                              "acel": _lista(imu.accelerometer), "quat": _lista(imu.quaternion),
                              "temp": float(imu.temperature)}}
    for lado in ("left", "right"):
        h, _ = D.pega(lado)
        if h is None:
            continue
        r[f"mao_{lado}"] = {
            "juntas": [{"nome": MAO[lado][k], "q": h.motor_state[k].q, "dq": h.motor_state[k].dq,
                        "tau": h.motor_state[k].tau_est, "temp": _lista(h.motor_state[k].temperature),
                        "vol": float(h.motor_state[k].vol), "erro": int(h.motor_state[k].motorstate)}
                       for k in range(7)],
            "pressao": [{"pressao": _lista(p.pressure), "temp": _lista(p.temperature)} for p in h.press_sensor_state],
            "power_v": float(h.power_v), "power_a": float(h.power_a), "system_v": float(h.system_v),
            "device_v": float(h.device_v), "erro": _lista(h.error)}
    b, _ = D.pega("bms")
    if b is not None:
        cel = [c for c in b.cell_vol if c]
        r["bateria"] = {"soc": int(b.soc), "soh": int(b.soh), "tensao_v": b.bmsvoltage[0] / 1000.0,
                        "corrente_a": b.current / 1000.0, "ciclos": int(b.cycle),
                        "celulas_mv": [min(cel), max(cel)] if cel else None,
                        "temps": [t for t in b.temperature if t], "estado": list(b.bmsstate)}
    p, _ = D.pega("placa")
    if p is not None:
        r["placa"] = {k: list(getattr(p, k)) for k in ("fan_state", "temperature", "value", "state")}
    pan, idade = D.pega("panico")
    r["panico"] = pan if pan is not None and idade < 1.0 else None
    modo, idade = D.pega("modo")
    r["modo"] = (modo or {}).get("robot_mode") if idade < 2.0 else None
    r["cameras"] = {f"{n}.{k}": {"fps": round(c.fps, 1), "idade": round(time.time() - t, 2),
                                 "nome": k, "tam": c.tam.get(k)}
                    for n, c in cams.items() for k, t in list(c.t.items())}
    r["servidor_cameras"] = {n: c.servidor for n, c in cams.items() if c.servidor}
    for n, c in cams.items():
        if not c.t:
            r["cameras"][n] = {"fps": 0.0, "idade": None, "nome": None, "tam": None}
    r["jetson"] = jetson()
    r["pose_inicial"] = POSE.estado if POSE is not None else None
    r["ultima_fala"] = VOZ.ultima if VOZ is not None else None
    return r


# ── botão "Assumir pose inicial" ────────────────────────────────────────────
POSE_ARQ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_partida_dex1.json")
BRACOS, PUNHOS, CINTURA = list(range(15, 29)), {19, 20, 21, 26, 27, 28}, (12, 13, 14)


class PoseInicial:
    """Leva os braços DEVAGAR à pose de partida do WLA pela ponte v3 (porta 6000, mesmo comando do
    LeRobot: arm_sdk), abre as Dex3 (6003) e, no fim, aciona o pânico por software: a ponte fica
    SEGURANDO os braços na pose. Só começa com a ponte livre, em loco, com o estado chegando; para
    na hora se o cogumelo for apertado."""

    def __init__(self, ctx):
        self.estado = "parada"
        self.rodando = False
        self.cmd = ctx.socket(zmq.PUSH)
        self.cmd.setsockopt(zmq.LINGER, 0)
        self.cmd.setsockopt(zmq.SNDHWM, 2)
        self.cmd.connect("tcp://127.0.0.1:6000")
        self.mao = ctx.socket(zmq.PUSH)
        self.mao.setsockopt(zmq.LINGER, 0)
        self.mao.connect("tcp://127.0.0.1:6003")
        self.pedido = ctx.socket(zmq.PUSH)
        self.pedido.setsockopt(zmq.LINGER, 500)
        self.pedido.connect("tcp://127.0.0.1:6006")

    def comeca(self, tipo):
        if self.rodando:
            return "já está indo para a pose"
        try:
            p = json.load(open(POSE_ARQ))
            suf = {"inicial": "_inicial", "elevada": "_elevada", "gravacao": "_gravacao", "descanso": "_descanso"}.get(tipo, "")
            alvo = dict(zip(BRACOS, list(p["left" + suf]) + list(p["right" + suf])))
            # inicial/gravação: coluna reta; elevada/dataset: tronco 10° à frente, como no dataset
            alvo.update(dict(zip(CINTURA, p.get("cintura" + suf, p.get("cintura", [0.0] * 3)))))
        except Exception as e:  # noqa: BLE001
            return f"sem a pose ({POSE_ARQ}): {e}"
        pan, idade_p = D.pega("panico")
        modo, idade_m = D.pega("modo")
        low, idade_l = D.pega("low")
        if pan is None or idade_p > 1.0:
            return "ponte v3 sem sinal"
        if pan.get("panico"):
            return "a ponte está TRAVADA: rearme primeiro"
        if not modo or idade_m > 2.0 or modo.get("robot_mode") != "loco":
            return "a ponte não está em loco"
        if low is None or idade_l > 0.3:
            return "sem lowstate"
        self.rodando = True
        threading.Thread(target=self._vai, args=(alvo, tipo), daemon=True).start()
        return "indo"

    def _envia(self, q, mm):
        mc = [{"mode": 0, "q": 0.0, "dq": 0.0, "kp": 0.0, "kd": 0.0, "tau": 0.0} for _ in range(35)]
        for i in CINTURA:
            mc[i] = {"mode": 1, "q": q[i], "dq": 0.0, "kp": 300.0, "kd": 3.0, "tau": 0.0}   # 29/10: 300/3 = o kp_high do xr_teleoperate da Unitree (kd 8 fazia o motor "tremer" a corrente)
        # 29/10: roll de volta ao 100/4 de 28/10 (300/3 e 150/3 esquentaram); a ponte soma a gravidade da cintura
        mc[CINTURA[1]]["kp"], mc[CINTURA[1]]["kd"] = 100.0, 4.0
        for i in BRACOS:
            kp, kd = (40.0, 1.5) if i in PUNHOS else (80.0, 3.0)   # a ponte corta para os tetos dela
            if i in JUNTAS_SOLTAS:
                kp, kd = 0.0, 0.0   # 27/10: pitch do punho esquerdo com defeito: solto (pulava e esquentava)
            mc[i] = {"mode": 1, "q": q[i], "dq": 0.0, "kp": kp, "kd": kd, "tau": 0.0}
        mc[29]["q"] = 1.0   # arm_sdk: o WBC obedece nos braços
        self.cmd.send(json.dumps({"topic": "rt/arm_sdk", "data": {"mode_pr": 1, "mode_machine": mm,
                                                                  "motor_cmd": mc}}).encode(), zmq.NOBLOCK)

    # mão FECHADA (punho) = 80% da mão em repouso medida no robô, ordem do SDK — a mesma do executor do WLA
    PUNHO = {"left": [0.0, 0.76, 1.256, -1.248, -1.368, -1.216, -1.424],
             "right": [0.0, -0.8, -1.32, 1.232, 1.376, 1.232, 1.352]}

    def _abre_maos(self, fecha=False):
        for lado in ("left", "right"):
            alvo = self.PUNHO[lado] if fecha else [0.0] * 7
            mc = [{"mode": (k & 0x0F) | (0x01 << 4), "q": alvo[k], "dq": 0.0, "kp": 1.0, "kd": 0.2, "tau": 0.0}
                  for k in range(7)]
            self.mao.send(json.dumps({"topic": f"rt/dex3/{lado}/cmd", "data": {"motor_cmd": mc}}).encode(),
                          zmq.NOBLOCK)

    @staticmethod
    def _perto_do_descanso(q0):
        """Os dois braços perto da pose de descanso (ombro pitch/roll e cotovelo a menos de 0,3 rad)?"""
        try:
            p = json.load(open(POSE_ARQ))
            d = dict(zip(BRACOS, list(p["left_descanso"]) + list(p["right_descanso"])))
        except Exception:  # noqa: BLE001
            return False
        return max(abs(q0[i] - d[i]) for i in (15, 16, 18, 22, 23, 25)) < 0.3

    def _vai(self, alvo, tipo):
        try:
            low, _ = D.pega("low")
            q0 = {i: float(low.motor_state[i].q) for i in list(CINTURA) + BRACOS}
            mm = int(low.mode_machine)
            maior = max(abs(alvo[i] - q0[i]) for i in alvo)
            dur = max(4.0, maior / 0.25)
            print(f"[diagnóstico] pose inicial ({tipo}): até {maior:.2f} rad, em {dur:.0f} s", flush=True)
            q = dict(q0)
            # 03/11: SAINDO DO DESCANSO (braços baixados, mão fechada junto da coxa): abrir a mão ali batia os dedos na
            # quina da perna. Então: 1) AFASTA de mão fechada (ombro para fora e um pouco para a frente), 2) ABRE a mão
            # parada longe da perna, 3) só então vai para a pose. Vale para qualquer pose que não seja o próprio descanso.
            if tipo != "descanso" and self._perto_do_descanso(q0):
                afasta = dict(q0)
                for i, d in ((15, -AFASTA_PITCH), (16, +AFASTA_ROLL), (22, -AFASTA_PITCH), (23, -AFASTA_ROLL)):
                    afasta[i] = q0[i] + d                 # pitch negativo = para a frente; roll para FORA do corpo
                print("[diagnóstico] saindo do descanso: afasta as mãos das pernas de mão fechada, abre, e só então sobe", flush=True)
                for fase, (de, para, dur_f, mao_fechada) in enumerate(((q0, afasta, 1.5, True), (afasta, afasta, 1.0, False))):
                    t0 = time.time()
                    while time.time() - t0 < dur_f:
                        pan, idade_p = D.pega("panico")
                        _, idade_l = D.pega("low")
                        if pan is None or idade_p > 1.0 or pan.get("panico") or idade_l > 0.3:
                            self.estado = "INTERROMPIDA (pânico, ponte ou estado sumiu) — a ponte segura o braço"
                            return
                        u = min(1.0, (time.time() - t0) / dur_f)
                        s_ = 0.5 - 0.5 * np.cos(np.pi * u)
                        for i in alvo:
                            q[i] = de[i] + s_ * (para[i] - de[i])
                        self._envia(q, mm)
                        self._abre_maos(fecha=mao_fechada)
                        self.estado = "afastando as mãos das pernas" if fase == 0 else "abrindo as mãos (longe das pernas)"
                        time.sleep(0.02)
                q0 = dict(afasta)                          # a pose sai daqui
                maior = max(abs(alvo[i] - q0[i]) for i in alvo)
                dur = max(4.0, maior / 0.25)
            # pose de DESCANSO (06/10): fecha o PUNHO primeiro, com o braço parado, e vai de mão fechada
            fecha = tipo == "descanso"
            t_mao = 1.5 if fecha else 0.0
            t0 = time.time()
            while fecha and time.time() - t0 < t_mao:
                self._envia(q, mm)
                self._abre_maos(fecha=True)
                self.estado = "fechando o punho"
                time.sleep(0.02)
            t0 = time.time()
            while time.time() - t0 < dur + 1.5:
                pan, idade_p = D.pega("panico")
                _, idade_l = D.pega("low")
                if pan is None or idade_p > 1.0 or pan.get("panico") or idade_l > 0.3:
                    self.estado = "INTERROMPIDA (pânico, ponte ou estado sumiu) — a ponte segura o braço"
                    print(f"[diagnóstico] pose inicial {self.estado}", flush=True)
                    return
                u = min(1.0, (time.time() - t0) / dur)
                s = 0.5 - 0.5 * np.cos(np.pi * u)
                for i in alvo:
                    q[i] = q0[i] + s * (alvo[i] - q0[i])
                self._envia(q, mm)
                self._abre_maos(fecha=fecha)
                self.estado = f"indo para a pose {tipo}: {100 * u:.0f}%"
                time.sleep(0.02)
            # CINTURA RETA (27/10): kp 300 cede sob o peso dos braços (~17 Nm na pose de gravação = ~3° para a
            # frente; no descanso ~1,7 Nm, ~0,3°). Integrador no roll/pitch até o MEDIDO chegar ao alvo (±0,12 rad),
            # e só então trava a ponte (ela segue segurando o último comando, já corrigido).
            # 27/10: DESLIGADO (ENDIREITA_CINTURA_S = 0): com a ponte travada segurando o alvo corrigido, o roll/pitch
            # ficou brigando com o equilíbrio da Unitree e ESQUENTOU. Volta a 300/8 no alvo 0 (cede ~3° com braço esticado).
            corr = {i: 0.0 for i in CINTURA[1:]}
            t0 = time.time()
            while time.time() - t0 < ENDIREITA_CINTURA_S:
                pan, idade_p = D.pega("panico")
                low, idade_l = D.pega("low")
                if pan is None or idade_p > 1.0 or pan.get("panico") or idade_l > 0.3:
                    self.estado = "INTERROMPIDA (pânico, ponte ou estado sumiu) — a ponte segura o braço"
                    return
                for i in CINTURA[1:]:
                    e = alvo[i] - float(low.motor_state[i].q)
                    corr[i] = float(np.clip(corr[i] + 1.5 * e * 0.02, -0.12, 0.12))
                    q[i] = alvo[i] + corr[i]
                self._envia(q, mm)
                self._abre_maos(fecha=fecha)
                self.estado = f"endireitando a cintura (roll/pitch +{corr[CINTURA[1]]:+.3f}/{corr[CINTURA[2]]:+.3f} rad)"
                time.sleep(0.02)
            self.pedido.send(json.dumps({"panico": True}).encode())
            low, _ = D.pega("low")
            erro = max(abs(float(low.motor_state[i].q) - alvo[i]) for i in alvo)
            self.estado = (f"na pose {tipo} (erro {erro:.2f} rad) — ponte TRAVADA segurando os braços; "
                           "o executor do WLA pede o rearme e começa daqui")
            print(f"[diagnóstico] {self.estado}", flush=True)
        except Exception as e:  # noqa: BLE001
            self.estado = f"erro: {e}"
        finally:
            self.rodando = False


JUNTAS_SOLTAS = {20}         # punho esquerdo (pitch) com defeito: sem força
AFASTA_ROLL = 0.20          # 03/11: rad, ombro para FORA ao sair do descanso (antes de abrir a mão)
AFASTA_PITCH = 0.20         # rad, ombro para a FRENTE, idem
ENDIREITA_CINTURA_S = 0.0   # s de integrador na cintura antes de travar a ponte (0 = não)


# ── fala: o executor do WLA manda PCM (16 kHz, mono, s16le) e o alto-falante do G1 toca ──────
def panico_de_verdade():
    """28/10: a ponte sobe em pânico até o rearme ("início: rearme pendente"). Esse BLOQUEIO DA PARTIDA (botão OK)
    não é alarme: a voz e a luz da conversa podem funcionar. Pânico de verdade = botão apertado, falha ou disparo."""
    pan, _ = D.pega("panico")
    if pan is None or not pan.get("panico"):
        return False
    return not (pan.get("botao_ok") and str(pan.get("motivo", "")).startswith("início"))


def bloqueio_da_partida():
    pan, _ = D.pega("panico")
    return pan is not None and pan.get("panico") and not panico_de_verdade()


class Voz:
    """POST /fala com o áudio cru. Toca pelo AudioClient (o mesmo do alerta de pânico da ponte), numa
    thread com fila de 3: fala velha é descartada. Com a ponte em pânico, não fala (o alarme é dela)."""

    def __init__(self, volume=100, microfone=True):
        import queue
        self.queue = queue
        self.fila = queue.Queue(maxsize=3)
        self.volume = volume
        self.cliente = None
        self.ultima = ""
        self.trava = threading.Lock()   # o cliente RPC é usado pela fala, pelo LED e pelo microfone
        self.mic = None                 # 28/10: resultado da última ativação do microfone
        self.led_atual = None
        threading.Thread(target=self._laco, daemon=True, name="voz").start()
        if microfone:
            threading.Thread(target=self._mantem_mic, daemon=True, name="microfone").start()

    def _cli(self):
        if self.cliente is None:
            from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
            c = AudioClient()
            c.SetTimeout(3.0)
            c.Init()
            c._RegistApi(1008, 0)   # modo do serviço de voz (não está no AudioClient do SDK)
            c.SetVolume(self.volume)
            self.cliente = c
        return self.cliente

    def _mantem_mic(self):
        """28/10: MICROFONE do G1 ligado (voice API 1008 {"mode": 1}): o arranjo de 4 microfones manda PCM 16 kHz
        mono por multicast UDP 239.168.123.161:5555 (a PGX escuta e reconhece a fala). Sem isso o microfone fica
        mudo. Confere a cada 60 s (API 1007) e só religa se o app da Unitree tiver mudado o modo."""
        while True:
            try:
                with self.trava:
                    cli = self._cli()
                    cli._RegistApi(1007, 0)
                    # SÓ liga se não estiver ligado: cada 1008 faz o assistente da Unitree anunciar "Wake up mode" e
                    # acender a luz azul (28/10: religava a cada 60 s)
                    c7, d7 = cli._Call(1007, json.dumps({}))
                    ja = c7 == 0 and json.loads(d7 or "{}").get("mode") == 1
                    code = 0 if ja else cli._Call(1008, json.dumps({"mode": 1}))[0]
                if self.mic != code:
                    print(f"[diagnóstico] 🎤 microfone {'LIGADO (multicast 239.168.123.161:5555)' if code == 0 else f'falhou: código {code}'}",
                          flush=True)
                self.mic = code
            except Exception as e:  # noqa: BLE001
                print(f"[diagnóstico] 🎤 microfone falhou: {e}", flush=True)
                self.mic = str(e)
            time.sleep(60)

    def led(self, r, g, b):
        """28/10: luz do rosto (voice API 1010). Pânico de verdade: a luz é da ponte (recusa). Bloqueio da partida:
        muda, mas "apagada" vira o amarelo do rearme pendente."""
        if panico_de_verdade():
            return False, "ponte em pânico: a luz mostra a trava"
        if bloqueio_da_partida() and (r, g, b) == (0, 0, 0):
            r, g, b = 255, 140, 0   # "apagada" = o AMARELO do rearme pendente (o aviso não some)
        with self.trava:
            code = self._cli().LedControl(int(r), int(g), int(b))
        self.led_atual = (int(r), int(g), int(b))
        return code == 0, f"código {code}"

    def pede(self, pcm, texto):
        try:
            self.fila.put_nowait((pcm, texto))
            return True
        except self.queue.Full:
            return False

    def _laco(self):
        while True:
            pcm, texto = self.fila.get()
            if panico_de_verdade() and "pânico" not in texto.lower():   # o bloqueio da partida não cala a voz
                continue
            try:
                with self.trava:
                    self._cli()
                if not pcm:   # sem áudio: voz interna do G1 (TtsMaker; speaker 1 = inglês)
                    with self.trava:
                        self.cliente.TtsMaker(texto, 1)
                    self.ultima = texto
                    print(f"[diagnóstico] 🗣  (G1) {texto}", flush=True)
                    time.sleep(0.5 + 0.08 * len(texto))
                    continue
                sid = str(time.time_ns())
                for i in range(0, len(pcm), 32000):   # 1 s por pedaço
                    with self.trava:
                        self.cliente.PlayStream("prometheus_fala", sid, list(pcm[i:i + 32000]))
                    time.sleep(min(1.0, (len(pcm) - i) / 32000.0))
                with self.trava:
                    self.cliente.PlayStop("prometheus_fala")
                self.ultima = texto
                print(f"[diagnóstico] 🗣  {texto}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[diagnóstico] fala falhou: {e}", flush=True)


VOZ = None


PAGINA = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "painel_diagnostico.html"), "rb").read() \
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "painel_diagnostico.html")) else b""


class Lidar:
    """03/11: a nuvem crua do lidar da cabeça (lidar_servidor.py, porta 5558), os últimos ~0,4 s, para a vista ao vivo.
    O MID-360 está de cabeça para baixo no G1: aqui y e z são trocados de sinal (fica de pé: x frente, y esquerda, z cima)."""

    def __init__(self, ctx, porta=5558, janela=0.4):
        self.lotes, self.trava, self.janela, self.taxa, self.t = [], threading.Lock(), janela, 0.0, 0.0
        s = ctx.socket(zmq.SUB)
        s.setsockopt_string(zmq.SUBSCRIBE, "")
        s.setsockopt(zmq.RCVHWM, 4)
        s.connect(f"tcp://127.0.0.1:{porta}")
        threading.Thread(target=self._laco, args=(s,), daemon=True, name="lidar").start()

    def _laco(self, s):
        n, t0 = 0, time.time()
        while True:
            try:
                _, b = s.recv_multipart()
                P = np.frombuffer(b, np.float32).reshape(-1, 4)[:, :3] * np.float32([1, -1, -1])
                agora = time.time()
                with self.trava:
                    self.lotes = [x for x in self.lotes if agora - x[0] < self.janela] + [(agora, P)]
                self.t, n = agora, n + len(P)
                if agora - t0 > 2:
                    self.taxa, n, t0 = n / (agora - t0), 0, agora
            except Exception:  # noqa: BLE001
                time.sleep(0.05)

    def json(self, maximo=7000):
        with self.trava:
            P = np.concatenate([x[1] for x in self.lotes]) if self.lotes else np.zeros((0, 3), np.float32)
        if len(P) > maximo:
            P = P[np.random.default_rng().choice(len(P), maximo, replace=False)]
        vivo = time.time() - self.t < 1.0
        return {"vivo": vivo, "pontos_s": round(self.taxa) if vivo else 0, "n": int(len(P)),
                "cm": np.clip(np.round(P * 100), -32000, 32000).astype(int).ravel().tolist()}


# 03/11: TESTE DE PRESSÃO da Dex3 — fechar/abrir a mão pela página (pela ponte, porta 6003, como o executor) e gravar
# a pressão das 9 almofadas x 12 canais (+ juntas e torque dos dedos) para ver o momento em que a caneca apoia na mesa.
# Dex3, ordem do SDK (a mesma do roda_wla_real.py): 80% da mão em repouso medida = toda fechada.
MAO_FECHADA = {"left": 0.8 * np.array([0.0, 0.95, 1.57, -1.56, -1.71, -1.52, -1.78]),
               "right": 0.8 * np.array([0.0, -1.0, -1.65, 1.54, 1.72, 1.54, 1.69])}
TOPICO_MAO = {"left": "rt/dex3/left/cmd", "right": "rt/dex3/right/cmd"}
VALIDO = 50000   # canais com 30000 = sem sensor; o resto ~101 kPa (pressão do ar) + o toque


class TestePressao:
    def __init__(self, ctx):
        self.sock = ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.SNDHWM, 2)
        self.sock.connect("tcp://127.0.0.1:6003")
        self.alvo = {}            # lado -> fechamento 0..1 (mandado a 20 Hz enquanto estiver aqui)
        self.zero = {}            # lado -> (9, 12) pressão de repouso
        self.grav = None          # {"fim": t, "linhas": [...], "arq": ...}
        self.ultimo_arq = None
        threading.Thread(target=self._laco, daemon=True, name="teste_pressao").start()

    @staticmethod
    def pressao(lado):
        h, _ = D.pega(lado)
        if h is None:
            return None, None
        P = np.array([list(p.pressure) for p in h.press_sensor_state], float)
        return P, h

    def mao(self, lado, fecha):
        if fecha is None:
            self.alvo.pop(lado, None)
        else:
            self.alvo[lado] = float(np.clip(fecha, 0.0, 1.0))

    zerado_a_mao = False

    def zera(self):
        self.zerado_a_mao = True
        for lado in ("left", "right"):
            P, _ = self.pressao(lado)
            if P is not None:
                self.zero[lado] = P.copy()

    def grava(self, segundos):
        if not self.zero:
            self.zera()
        pasta = os.path.expanduser("~/testes_pressao")
        os.makedirs(pasta, exist_ok=True)
        self.grav = {"fim": time.time() + segundos, "linhas": [], "arq": os.path.join(pasta, time.strftime("%Y%m%d_%H%M%S") + ".npz")}

    def _laco(self):
        while True:
            t0 = time.time()
            for lado, c in list(self.alvo.items()):
                q = c * MAO_FECHADA[lado]
                mc = [{"mode": (k & 0x0F) | (0x01 << 4), "q": float(q[k]), "dq": 0.0, "kp": 1.0, "kd": 0.2, "tau": 0.0} for k in range(7)]
                try:
                    self.sock.send(json.dumps({"topic": TOPICO_MAO[lado], "data": {"motor_cmd": mc}}).encode(), zmq.NOBLOCK)
                except zmq.Again:
                    pass
            g = self.grav
            if g is not None:
                lin = {"t": time.time()}
                for lado in ("left", "right"):
                    P, h = self.pressao(lado)
                    if P is not None:
                        lin[f"p_{lado}"] = P
                        lin[f"q_{lado}"] = [h.motor_state[k].q for k in range(7)]
                        lin[f"tau_{lado}"] = [h.motor_state[k].tau_est for k in range(7)]
                g["linhas"].append(lin)
                if time.time() >= g["fim"]:
                    self._salva(g)
                    self.grav = None
            time.sleep(max(0.0, 0.02 - (time.time() - t0)))   # 50 Hz

    def _salva(self, g):
        L = g["linhas"]
        out = {"t": np.array([x["t"] for x in L])}
        for k in ("p_left", "p_right", "q_left", "q_right", "tau_left", "tau_right"):
            if all(k in x for x in L):
                out[k] = np.array([x[k] for x in L], float)
        for lado, z in self.zero.items():
            out[f"zero_{lado}"] = z
        np.savez_compressed(g["arq"], **out)
        self.ultimo_arq = g["arq"]
        print(f"[diagnóstico] pressão gravada: {g['arq']} ({len(L)} amostras)", flush=True)

    def resumo(self):
        r = {"segurando": self.alvo, "gravando": None if self.grav is None else round(self.grav["fim"] - time.time(), 1),
             "ultimo_arq": self.ultimo_arq, "zerado": self.zerado_a_mao, "maos": {}}
        for lado in ("left", "right"):
            P, _ = self.pressao(lado)
            if P is None:
                continue
            if lado not in self.zero:   # 03/11: sem "zerar", a referência é a 1ª leitura (antes dava sempre 0)
                self.zero[lado] = P.copy()
            z = self.zero[lado]
            ok = (P > VALIDO) & (z > VALIDO)
            d = np.where(ok, P - z, 0.0) / 1000.0                      # kPa acima do repouso
            cru = P[P > VALIDO]
            r["maos"][lado] = {"almofadas_kpa": np.round(d.sum(1), 2).tolist(), "total_kpa": round(float(d.sum()), 2),
                               "max_kpa": round(float(d.max()), 2), "canais": int(ok.sum()),
                               "cru_max_kpa": round(float(cru.max()) / 1000, 2) if len(cru) else None}
        return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--porta", type=int, default=8095)
    ap.add_argument("--interface", default="", help="interface do DDS (vazio = a mesma da ponte)")
    a = ap.parse_args()

    subs = sobe_dds(a.interface)  # noqa: F841  (guarda as assinaturas vivas)
    ctx = zmq.Context.instance()
    threading.Thread(target=assina_json, args=(ctx, 6007, "panico"), daemon=True).start()
    threading.Thread(target=assina_json, args=(ctx, 6004, "modo"), daemon=True).start()
    cams = {"p5555": Camera(ctx, 5555), "p5556": Camera(ctx, 5556)}
    lidar = Lidar(ctx)
    pedido = ctx.socket(zmq.PUSH)
    pedido.setsockopt(zmq.LINGER, 0)
    pedido.setsockopt(zmq.SNDHWM, 1)
    pedido.connect("tcp://127.0.0.1:6006")
    global POSE, VOZ
    POSE = PoseInicial(ctx)
    teste = TestePressao(ctx)
    VOZ = Voz()
    cache = {"t": 0.0, "json": b"{}"}
    trava_cache = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def _manda(self, corpo, tipo, codigo=200):
            self.send_response(codigo)
            self.send_header("Content-Type", tipo)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(corpo)))
            self.end_headers()
            try:
                self.wfile.write(corpo)
            except (BrokenPipeError, ConnectionResetError):
                pass   # o navegador fechou a conexão (trocou de aba, recarregou): nada a fazer

        def do_GET(self):
            caminho = self.path.split("?")[0]
            if caminho == "/":
                return self._manda(PAGINA, "text/html; charset=utf-8")
            if caminho == "/estado.json":
                with trava_cache:   # vários navegadores abertos não multiplicam o custo
                    if time.time() - cache["t"] > 0.15:
                        cache["json"] = json.dumps(resumo(cams)).encode()
                        cache["t"] = time.time()
                    corpo = cache["json"]
                return self._manda(corpo, "application/json")
            if caminho.startswith("/static/"):   # 03/11: three.js local (a rede do robô não tem internet)
                arq = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", os.path.basename(caminho))
                if not os.path.isfile(arq):
                    return self._manda(b"nao achei", "text/plain", 404)
                return self._manda(open(arq, "rb").read(), "text/javascript")
            if caminho == "/pressao.json":   # 03/11: o teste de pressão da Dex3
                return self._manda(json.dumps(teste.resumo()).encode(), "application/json")
            if caminho == "/lidar.json":   # 03/11: a nuvem crua do lidar, ao vivo
                return self._manda(json.dumps(lidar.json()).encode(), "application/json")
            if caminho.startswith("/cam/") and caminho.endswith(".jpg"):
                porta, _, nome = caminho[5:-4].partition(".")
                c = cams.get(porta)
                jpg = c.jpeg(nome) if c and nome else None
                if jpg is None:
                    return self._manda(b"sem imagem", "text/plain", 404)
                return self._manda(jpg, "image/jpeg")
            self._manda(b"nao achei", "text/plain", 404)

        def do_POST(self):
            if self.path == "/rearmar":
                pedido.send(json.dumps({"rearmar": True}).encode(), zmq.NOBLOCK)
                print("[diagnóstico] pedido de REARME pela página", flush=True)
            elif self.path.startswith("/fala"):
                import urllib.parse
                n = int(self.headers.get("Content-Length", 0))
                pcm = self.rfile.read(n)
                texto = urllib.parse.unquote(self.path.partition("texto=")[2]) or "?"
                ok = VOZ.pede(pcm, texto)
                return self._manda(json.dumps({"ok": ok}).encode(), "application/json")
            elif self.path.startswith("/pose_inicial"):
                tipo = next((x for x in ("dataset", "elevada", "gravacao", "descanso", "inicial") if x in self.path), "inicial")
                msg = POSE.comeca(tipo)
                print(f"[diagnóstico] botão pose inicial ({tipo}): {msg}", flush=True)
                return self._manda(json.dumps({"ok": msg == "indo", "msg": msg}).encode(), "application/json")
            elif self.path.startswith(("/mao", "/zera_pressao", "/grava_pressao")):   # 03/11: teste de pressão
                import urllib.parse
                q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                if self.path.startswith("/mao"):
                    lado = q.get("lado", ["right"])[0]
                    f = q.get("fecha", [""])[0]
                    if lado in MAO_FECHADA:
                        teste.mao(lado, None if f in ("", "solta") else float(f))
                        print(f"[diagnóstico] mão {lado}: {'solta (para de mandar)' if f in ('', 'solta') else f'fecha {f}'}", flush=True)
                elif self.path.startswith("/zera_pressao"):
                    teste.zera()
                else:
                    teste.grava(min(60.0, float(q.get("s", ["20"])[0])))
                return self._manda(json.dumps(teste.resumo()).encode(), "application/json")
            elif self.path.startswith("/led"):   # 28/10: /led?r=0&g=120&b=255 — a luz do rosto
                import urllib.parse
                q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                try:
                    ok, msg = VOZ.led(*(max(0, min(255, int(q.get(k, ["0"])[0]))) for k in ("r", "g", "b")))
                except Exception as e:  # noqa: BLE001
                    ok, msg = False, str(e)
                return self._manda(json.dumps({"ok": ok, "msg": msg}).encode(), "application/json")
            elif self.path == "/panico":
                pedido.send(json.dumps({"panico": True}).encode(), zmq.NOBLOCK)
                print("[diagnóstico] PÂNICO pela página", flush=True)
            else:
                return self._manda(b"nao achei", "text/plain", 404)
            self._manda(b'{"ok":true}', "application/json")

    srv = ThreadingHTTPServer(("0.0.0.0", a.porta), H)
    srv.daemon_threads = True
    print(f"[diagnóstico] http://0.0.0.0:{a.porta}/  (só lê o DDS; rearme/pânico só pelos botões)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
