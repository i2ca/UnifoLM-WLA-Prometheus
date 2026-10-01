#!/usr/bin/env python
"""
TESTE DO COTOVELO — um movimento curto e lento, pela ponte v3 com botão de pânico
==================================================================================
Fala direto com a `dex3_g1_server_v3_panico.py` no robô (ZMQ), no MESMO formato de comando
da classe `UnitreeG1Dex3` (arm_sdk: mode_pr=1, motor_cmd[29].q=1). Só o COTOVELO DIREITO se
move: +delta rad num perfil suave, pausa, e volta. Todo o resto do braço e a cintura ficam
comandados na pose MEDIDA no início. A ponte ainda limita a velocidade (0,3 rad/s) e corta o
kp (20 no ombro/cotovelo, 10 no punho).

Antes de mandar qualquer comando ele confere: modo `loco` na 6004, estado chegando na 6001,
estado do pânico na 6007. Aborta (e aciona o pânico por software) se o pânico disparar, se o
estado parar de chegar por > 0,2 s, ou em qualquer erro. No fim também aciona o pânico por
software: a ponte fica segurando o braço parado.

    ~/miniforge3/envs/prometheus-vla/bin/python robo_g1/teste_cotovelo.py            # delta +0,20 rad
    ~/miniforge3/envs/prometheus-vla/bin/python robo_g1/teste_cotovelo.py --simulado # não envia nada
"""
import argparse
import csv
import json
import math
import sys
import threading
import time
from pathlib import Path

import zmq

ROBO = "192.168.123.164"
COTOVELO_DIR = 25
BRACOS = range(15, 29)
PUNHOS = {19, 20, 21, 26, 27, 28}
CINTURA = (12, 13, 14)
PESO_ARM_SDK = 29
N_MOTORES = 35


class Estado:
    """Assina lowstate (6001) e pânico (6007) em threads; guarda o último de cada."""

    def __init__(self, ctx, robo):
        self.low = None
        self.t_low = 0.0
        self.panico = None
        self.t_panico = 0.0
        for porta, alvo in ((6001, self._low), (6007, self._pan)):
            s = ctx.socket(zmq.SUB)
            s.setsockopt_string(zmq.SUBSCRIBE, "")
            s.setsockopt(zmq.RCVHWM, 2)
            s.connect(f"tcp://{robo}:{porta}")
            threading.Thread(target=self._laco, args=(s, alvo), daemon=True).start()

    def _laco(self, s, alvo):
        while True:
            try:
                alvo(json.loads(s.recv().decode("utf-8")))
            except Exception:
                time.sleep(0.01)

    def _low(self, m):
        self.low = m.get("data", m)
        self.t_low = time.time()

    def _pan(self, m):
        self.panico = m
        self.t_panico = time.time()

    def q(self, i):
        return float(self.low["motor_state"][i]["q"])


def modo_da_ponte(ctx, robo, timeout=3.0):
    s = ctx.socket(zmq.SUB)
    s.setsockopt_string(zmq.SUBSCRIBE, "")
    s.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
    s.connect(f"tcp://{robo}:6004")
    try:
        return json.loads(s.recv().decode("utf-8")).get("robot_mode")
    except zmq.Again:
        return None
    finally:
        s.close()


def monta_cmd(alvo_q: dict, estado: Estado, kp_braco, kd_braco, kp_punho, kd_punho):
    """Comando de 35 motores igual ao da classe UnitreeG1Dex3 em modo loco."""
    mc = [{"mode": 0, "q": 0.0, "dq": 0.0, "kp": 0.0, "kd": 0.0, "tau": 0.0} for _ in range(N_MOTORES)]
    for i in CINTURA:   # cintura segura onde está (a classe trava roll/pitch com kp 300)
        mc[i] = {"mode": 1, "q": alvo_q[i], "dq": 0.0, "kp": 150.0, "kd": 5.0, "tau": 0.0}   # 300/6 esquentava
    for i in BRACOS:
        kp, kd = (kp_punho, kd_punho) if i in PUNHOS else (kp_braco, kd_braco)
        mc[i] = {"mode": 1, "q": alvo_q[i], "dq": 0.0, "kp": kp, "kd": kd, "tau": 0.0}
    mc[PESO_ARM_SDK]["q"] = 1.0   # arm_sdk: o WBC obedece a este cliente nos braços
    return {"topic": "rt/arm_sdk",
            "data": {"mode_pr": 1, "mode_machine": int(estado.low.get("mode_machine", 0)), "motor_cmd": mc}}


def perfil(t, dur):
    """0 -> 1 suave (cosseno), sem tranco no começo nem no fim."""
    if t <= 0:
        return 0.0
    if t >= dur:
        return 1.0
    return 0.5 - 0.5 * math.cos(math.pi * t / dur)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robo", default=ROBO)
    ap.add_argument("--delta", type=float, default=0.20, help="rad no cotovelo direito (máx 0,35)")
    ap.add_argument("--ida", type=float, default=4.0, help="segundos para ir (e para voltar)")
    ap.add_argument("--pausa", type=float, default=2.0)
    ap.add_argument("--segura", type=float, default=3.0, help="segundos segurando a pose antes de mover")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--kp-braco", type=float, default=80.0, help="a ponte corta para 20")
    ap.add_argument("--kd-braco", type=float, default=3.0)
    ap.add_argument("--kp-punho", type=float, default=40.0, help="a ponte corta para 10")
    ap.add_argument("--kd-punho", type=float, default=1.5)
    ap.add_argument("--simulado", action="store_true", help="faz tudo MENOS enviar comando/rearme")
    ap.add_argument("--log", default=str(Path.home() / f"teste_cotovelo_{time.strftime('%Y%m%d_%H%M%S')}.csv"))
    a = ap.parse_args()
    if abs(a.delta) > 0.35:
        sys.exit("--delta acima de 0,35 rad: recusado neste teste")

    ctx = zmq.Context.instance()
    modo = modo_da_ponte(ctx, a.robo)
    if modo != "loco":
        sys.exit(f"❌ a ponte não informou modo 'loco' na 6004 (veio {modo!r}). Abortando sem enviar nada.")
    est = Estado(ctx, a.robo)
    t0 = time.time()
    while (est.low is None or est.panico is None) and time.time() - t0 < 5:
        time.sleep(0.05)
    if est.low is None:
        sys.exit("❌ sem lowstate na 6001 em 5 s. Abortando sem enviar nada.")
    if est.panico is None:
        sys.exit("❌ sem estado do pânico na 6007: esta não é a ponte v3? Abortando sem enviar nada.")
    print(f"✅ ponte v3 em loco | pânico: {est.panico['panico']} ({est.panico['motivo'] or 'livre'}) | "
          f"GPIO: {est.panico['gpio']} | cotovelo dir medido: {est.q(COTOVELO_DIR):+.3f} rad", flush=True)
    if est.panico["gpio"] != "ok":
        sys.exit("❌ o GPIO do botão não está ok na ponte. Abortando.")

    rearme = ctx.socket(zmq.PUSH)
    rearme.setsockopt(zmq.LINGER, 500)
    rearme.connect(f"tcp://{a.robo}:6006")
    cmd_sock = ctx.socket(zmq.PUSH)
    cmd_sock.setsockopt(zmq.LINGER, 0)
    cmd_sock.setsockopt(zmq.SNDHWM, 2)
    cmd_sock.connect(f"tcp://{a.robo}:6000")

    def panico_software(motivo):
        print(f"\n🛑 acionando o pânico por software: {motivo}", flush=True)
        if not a.simulado:
            rearme.send(json.dumps({"panico": True}).encode("utf-8"))
            time.sleep(0.3)

    if est.panico["panico"]:
        input("🔒 A ponte está BLOQUEADA. Solte o cogumelo, confira o robô livre e aperte Enter para rearmar... ")
        if not a.simulado:
            rearme.send(json.dumps({"rearmar": True}).encode("utf-8"))
            t1 = time.time()
            while est.panico["panico"] and time.time() - t1 < 3:
                time.sleep(0.05)
            if est.panico["panico"]:
                sys.exit(f"❌ rearme recusado: {est.panico['motivo']} (o botão está solto há >= 1 s?)")
        print("🔓 rearmado.", flush=True)

    q0 = {i: est.q(i) for i in list(CINTURA) + list(BRACOS)}
    alvo = dict(q0)
    fases = [("segura", a.segura, 0.0, 0.0), ("ida", a.ida, 0.0, a.delta), ("pausa", a.pausa, a.delta, a.delta),
             ("volta", a.ida, a.delta, 0.0), ("final", 2.0, 0.0, 0.0)]
    print(f"▶ plano: segura {a.segura:.0f} s · cotovelo dir {q0[COTOVELO_DIR]:+.3f} -> "
          f"{q0[COTOVELO_DIR] + a.delta:+.3f} em {a.ida:.0f} s · pausa {a.pausa:.0f} s · volta {a.ida:.0f} s"
          f"{'  [SIMULADO: nada é enviado]' if a.simulado else ''}", flush=True)
    input("   Mão no cogumelo. Enter para começar (Ctrl+C cancela)... ")

    arq = open(a.log, "w", newline="")
    log = csv.writer(arq)
    log.writerow(["t", "fase", "alvo_cotovelo", "medido_cotovelo", "max_desvio_outras_juntas", "panico"])
    dt = 1.0 / a.hz
    pior_outras = 0.0
    try:
        t_ini = time.time()
        for nome, dur, de, para in fases:
            tf = time.time()
            print(f"   fase {nome} ({dur:.0f} s)", flush=True)
            while time.time() - tf < dur:
                tc = time.time()
                if est.panico and est.panico["panico"] and not a.simulado:
                    print(f"\n🛑 pânico na ponte ({est.panico['motivo']}): parei de enviar. A ponte segura o braço.",
                          flush=True)
                    return
                if tc - est.t_low > 0.2:
                    panico_software("lowstate parou de chegar")
                    return
                alvo[COTOVELO_DIR] = q0[COTOVELO_DIR] + de + (para - de) * perfil(tc - tf, dur)
                if not a.simulado:
                    cmd_sock.send(json.dumps(monta_cmd(alvo, est, a.kp_braco, a.kd_braco,
                                                       a.kp_punho, a.kd_punho)).encode("utf-8"))
                outras = max(abs(est.q(i) - q0[i]) for i in BRACOS if i != COTOVELO_DIR)
                pior_outras = max(pior_outras, outras)
                log.writerow([round(tc - t_ini, 3), nome, round(alvo[COTOVELO_DIR], 4),
                              round(est.q(COTOVELO_DIR), 4), round(outras, 4), est.panico["panico"]])
                time.sleep(max(0.0, dt - (time.time() - tc)))
        erro_final = est.q(COTOVELO_DIR) - q0[COTOVELO_DIR]
        print(f"\n✅ teste concluído | cotovelo voltou a {erro_final:+.3f} rad da pose inicial | "
              f"maior desvio das OUTRAS juntas do braço: {pior_outras:.3f} rad | log: {a.log}", flush=True)
    except KeyboardInterrupt:
        print("\n⏹ cancelado pelo operador.", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"\n❌ erro: {type(e).__name__}: {e}", flush=True)
    finally:
        arq.close()
        panico_software("fim do teste — a ponte segura o braço parado")


if __name__ == "__main__":
    main()
