#!/usr/bin/env python
"""Teste SEGURO (27/10): o roll/pitch da cintura pode ficar com o controle da Unitree (kp 0 no arm_sdk)?

Com 300/8 em alvo 0 a cintura esquentou (roll/pitch ~58 °C e subindo). Parado, sem o nosso comando, o controle da
Unitree segura o tronco reto sem esforço. Dúvida: com o arm_sdk ligado (peso 1), kp 0 no roll/pitch devolve essas
juntas para a Unitree, ou as deixa MOLES? Este teste descobre sem arriscar:

  1. segura braços, yaw e cintura onde estão (300/8), 2 s;
  2. baixa o kp/kd do roll/pitch de 300/8 até 0 em --rampa s, olhando o ângulo medido;
  3. se o roll ou o pitch sair mais de --limite rad do começo -> VOLTA NA HORA para 300/8 e para (= ficaria mole);
  4. se chegar a 0 com o tronco parado, segura --segura s em kp 0 medindo ângulo e torque (= a Unitree assumiu);
  5. volta para 300/8 e solta o arm_sdk aos poucos.

    cd ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/robo_g1 && python teste_cintura_solta.py    # na PGX, ponte em loco
MÃO NO BOTÃO DE PÂNICO. Robô de pé, parado, na pose de descanso ou de gravação.
"""
import argparse
import json
import sys
import time

import numpy as np
import zmq

from teste_cotovelo import BRACOS, CINTURA, Estado, modo_da_ponte, monta_cmd

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--robo", default="192.168.123.164")
ap.add_argument("--rampa", type=float, default=6.0)
ap.add_argument("--segura", type=float, default=8.0)
ap.add_argument("--limite", type=float, default=0.05, help="rad (~3°) que o roll/pitch pode sair antes de abortar")
a = ap.parse_args()

ctx = zmq.Context.instance()
if modo_da_ponte(ctx, a.robo) != "loco":
    sys.exit("a ponte não está em loco")
est = Estado(ctx, a.robo)
t0 = time.time()
while (est.low is None or est.panico is None) and time.time() - t0 < 6:
    time.sleep(0.05)
if est.low is None or est.panico is None or est.panico.get("panico"):
    sys.exit("sem lowstate/pânico, ou a ponte está travada (rearme no painel)")
cmd = ctx.socket(zmq.PUSH)
cmd.setsockopt(zmq.LINGER, 0)
cmd.connect(f"tcp://{a.robo}:6000")
rearme = ctx.socket(zmq.PUSH)
rearme.connect(f"tcp://{a.robo}:6006")
rearme.send(json.dumps({"rearmar": True}).encode())
time.sleep(0.5)

q0 = {i: est.q(i) for i in list(CINTURA) + list(BRACOS)}
RP = list(CINTURA[1:])
ini = {i: est.q(i) for i in RP}


def envia(kp_rp, kd_rp):
    c = monta_cmd(q0, est, 80.0, 3.0, 40.0, 1.5, kp_rp, kd_rp, kp_yaw=300.0, kd_yaw=8.0)
    cmd.send(json.dumps(c).encode())


def temp(i):
    t = est.low["motor_state"][i].get("temperature", 0)
    return max(t) if isinstance(t, (list, tuple)) else t


def mostra(rot, kp):
    ms = est.low["motor_state"]
    print(f"{rot:10s} kp {kp:5.1f} | roll {est.q(13):+.3f} pitch {est.q(14):+.3f} rad | tau roll "
          f"{ms[13].get('tau_est', 0):+5.1f} pitch {ms[14].get('tau_est', 0):+5.1f} Nm | {temp(13)}/{temp(14)} °C",
          flush=True)


def seguro():
    return est.panico is not None and not est.panico.get("panico")


abortou = False
try:
    t = time.time()
    while time.time() - t < 2.0:
        envia(300.0, 8.0)
        time.sleep(0.02)
    mostra("início", 300)
    t, ult = time.time(), 0
    while time.time() - t < a.rampa:
        s = 1 - (time.time() - t) / a.rampa
        kp, kd = 300.0 * s, 8.0 * s
        envia(kp, kd)
        desvio = max(abs(est.q(i) - ini[i]) for i in RP)
        if desvio > a.limite or not seguro():
            print(f"\n🛑 ABORTADO com kp {kp:.0f}: o tronco saiu {desvio:.3f} rad -> as juntas ficariam MOLES "
                  "(o arm_sdk não devolve o roll/pitch para a Unitree). Voltando para 300/8.", flush=True)
            abortou = True
            break
        if time.time() - ult > 0.5:
            ult = time.time()
            mostra("rampa", kp)
        time.sleep(0.02)
    if not abortou:
        print("\n✅ kp 0 no roll/pitch e o tronco ficou no lugar: segurando para medir...", flush=True)
        t, ult = time.time(), 0
        while time.time() - t < a.segura:
            envia(0.0, 0.0)
            desvio = max(abs(est.q(i) - ini[i]) for i in RP)
            if desvio > a.limite or not seguro():
                print(f"\n🛑 o tronco saiu {desvio:.3f} rad segurando com kp 0 — voltando para 300/8", flush=True)
                abortou = True
                break
            if time.time() - ult > 1.0:
                ult = time.time()
                mostra("kp 0", 0)
            time.sleep(0.02)
finally:
    t = time.time()
    while time.time() - t < 1.5:   # volta para 300/8
        envia(300.0, 8.0)
        time.sleep(0.02)
    mostra("fim", 300)
    rearme.send(json.dumps({"panico": True}).encode())   # a ponte trava e segura a pose (como o painel faz)
    print("\nRESULTADO:", "NÃO dá para soltar (fica mole) — fica com 300/8 ou mais macio" if abortou else
          "DÁ para soltar: com kp 0 a Unitree segura o roll/pitch -> aplicar no executor (--cintura-rp solta)",
          flush=True)
