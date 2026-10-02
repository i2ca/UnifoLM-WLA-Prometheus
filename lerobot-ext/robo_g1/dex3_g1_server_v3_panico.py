#!/usr/bin/env python3

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DDS-to-ZMQ bridge server for Unitree G1 robot with Dex3 hands.

VERSÃO v3 — BOTÃO DE PÂNICO + BRAÇOS LENTOS E LEVES (gerada de dex3_g1_server_v2.py
por gera_ponte_v3_panico.py). Ver a docstring do gerador para o comportamento.
    python dex3_g1_server_v3_panico.py                     # loco, pânico no GPIO4/GPIO6
    python dex3_g1_server_v3_panico.py --vel-braco 0.2     # ainda mais lento

Startup:
    python dex3_g1_server_v2.py            # High Level / Loco (rt/arm_sdk) — PADRÃO
    python dex3_g1_server_v2.py --debug    # Low Level (rt/lowcmd) — robô SUSPENSO

O padrão é LOCO desde 22/09/2026. Antes era debug, e o loco dependia de alguém
lembrar do `--loco` — que vinha do `USE_LOCO` do `init_prometheus-vla.sh`. Essa
variável foi trocada para `false` no robô em 28/08 sem ninguém perceber, e o
servidor passou a matar a IA na subida. Com o robô em pé, isso é o robô no chão.
O modo perigoso agora é o que exige ser pedido.

`--loco` continua aceito (não faz nada) para não quebrar quem ainda o passa.

O modo é decidido UMA VEZ no startup.
O servidor NÃO troca de modo durante a operação — isso evita quedas acidentais.

Portas ZMQ:
    6000  PULL  lowcmd  (PC → robô)
    6001  PUB   lowstate (robô → PC)
    6002  PUB   handstate (robô → PC)
    6003  PULL  handcmd (PC → robô)
    6004  PUB   robot_mode status (robô → PC)  ← NOVO
"""

import argparse
import base64
import json
import threading
import time
from typing import Any

import zmq
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
import copy
import ctypes
import os

# ═══════════════════════════════════════════════════════════════════════
# PÂNICO + LENTO/LEVE (v3)
# ═══════════════════════════════════════════════════════════════════════
REARME_PORT = 6006   # PULL  PC -> robô: {"rearmar": true} | {"panico": true}
PANICO_PORT = 6007   # PUB   robô -> PC: estado do pânico
IDX_BRACOS = list(range(15, 29))
IDX_PUNHOS = {19, 20, 21, 26, 27, 28}
IDX_CINTURA_YAW = 12
IDX_LIMITADOS = [IDX_CINTURA_YAW] + IDX_BRACOS


class BotaoPanico:
    """Botão NC entre uma saída em 0 (GPIO4 = PI.04) e uma entrada com pull-up (GPIO6 =
    PCC.03), pela libgpiod 1.4 do sistema via ctypes (o python `gpiod` não existe no robô).
    `normal()` é True com o NC fechado (entrada em 0); aberto/apertado/cabo solto dá 1.

    ORDEM OBRIGATÓRIA: a entrada é configurada ANTES de a saída ser dirigida. Ao contrário,
    se a entrada ainda estivesse como saída em 1, o botão fechado ligaria 0 contra 1."""

    def __init__(self, entrada="PCC.03", saida="PI.04"):
        self.lib = ctypes.CDLL("libgpiod.so.2", use_errno=True)
        L = self.lib
        L.gpiod_line_find.restype = ctypes.c_void_p
        L.gpiod_line_find.argtypes = [ctypes.c_char_p]
        L.gpiod_line_request_input.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.gpiod_line_request_output.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        L.gpiod_line_get_value.argtypes = [ctypes.c_void_p]
        L.gpiod_line_release.argtypes = [ctypes.c_void_p]
        L.gpiod_line_close_chip.argtypes = [ctypes.c_void_p]
        self.nomes = (entrada, saida)
        self.lin_in = L.gpiod_line_find(entrada.encode())
        self.lin_out = L.gpiod_line_find(saida.encode())
        if not self.lin_in or not self.lin_out:
            raise OSError(f"linha GPIO não encontrada: entrada {entrada}={bool(self.lin_in)} "
                          f"saída {saida}={bool(self.lin_out)}")
        if L.gpiod_line_request_input(self.lin_in, b"panico-g1") != 0:
            raise OSError(ctypes.get_errno(), f"não consegui pedir {entrada} como entrada")
        if L.gpiod_line_request_output(self.lin_out, b"panico-g1", 0) != 0:
            raise OSError(ctypes.get_errno(), f"não consegui pedir {saida} como saída em 0")

    def le(self) -> int:
        v = self.lib.gpiod_line_get_value(self.lin_in)
        if v < 0:
            raise OSError(ctypes.get_errno(), "falha ao ler o GPIO do botão")
        return v

    def normal(self) -> bool:
        return self.le() == 0

    def fecha(self):
        for lin in (self.lin_out, self.lin_in):   # solta a saída primeiro
            try:
                self.lib.gpiod_line_release(lin)
                self.lib.gpiod_line_close_chip(lin)
            except Exception:
                pass


class AlertaSonoro:
    """Fala no alto-falante do G1 e muda o LED (AudioClient do SDK). Roda na PRÓPRIA thread,
    por fila: o áudio nunca atrasa o congelamento do braço, e se ele falhar a trava segue igual.
    Os .wav (16 kHz, mono, 16 bit) ficam ao lado da ponte: pose_bloqueada.wav, pose_liberada.wav."""

    def __init__(self, pasta, volume=100, volume_bip=50):
        import queue
        import wave
        import numpy as np
        self.queue = queue
        self.fila = queue.Queue(maxsize=4)
        # Bipe de emergência: 3 pulsos de 1 kHz de 0,12 s em 1 s, amplitude = volume_bip%.
        # Toca em laço ENQUANTO o botão estiver apertado (bip_liga/bip_desliga).
        amostras = np.arange(16000) / 16000.0
        pulso = ((amostras % 0.33) < 0.12).astype(np.float32)
        tom = np.sin(2 * np.pi * 1000.0 * amostras) * pulso * (max(0, min(100, volume_bip)) / 100.0)
        self.bip_pcm = (tom * 32767).astype("<i2").tobytes()
        self.bip_ativo = False
        self.pcm = {}
        for nome in ("pose_bloqueada", "pose_liberada"):
            try:
                with wave.open(os.path.join(pasta, nome + ".wav")) as w:
                    assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
                    self.pcm[nome] = w.readframes(w.getnframes())
            except Exception as e:
                print(f"[Alerta] sem {nome}.wav ({e}) — só LED", flush=True)
        self.volume = volume
        self.cliente = None
        threading.Thread(target=self._laco, daemon=True).start()

    def bloqueado(self):
        self._pede("bloqueado")

    def bip_liga(self):
        self.bip_ativo = True

    def bip_desliga(self):
        self.bip_ativo = False

    def liberado(self):
        self._pede("liberado")

    def aguardando(self):
        """Botão solto, mas a trava continua: LED AMARELO até o rearme pedido pelo PC/painel."""
        self._pede("aguardando")

    def _pede(self, o_que):
        try:
            self.fila.put_nowait(o_que)
        except Exception:
            pass

    def _cliente(self):
        if self.cliente is None:
            from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
            c = AudioClient()
            c.SetTimeout(3.0)
            c.Init()
            c.SetVolume(self.volume)
            self.cliente = c
        return self.cliente

    def _toca(self, nome, pcm=None):
        pcm = pcm if pcm is not None else self.pcm.get(nome)
        if not pcm:
            return
        c = self._cliente()
        sid = str(time.time_ns())
        passo = 32000   # 1 s de áudio por pedaço
        for i in range(0, len(pcm), passo):
            c.PlayStream("prometheus_panico", sid, list(pcm[i:i + passo]))
            time.sleep(min(1.0, (len(pcm) - i) / 32000.0))
        c.PlayStop("prometheus_panico")

    def _fala(self, texto, wav):
        """Voz interna do G1 (TtsMaker, speaker 1 = inglês), a mesma das falas do executor do WLA.
        Se o TtsMaker devolver erro, toca o .wav gravado."""
        try:
            if self._cliente().TtsMaker(texto, 1) == 0:
                time.sleep(0.5 + 0.08 * len(texto))
                return
        except Exception as e:
            print(f"[Alerta] TtsMaker falhou ({e}) — uso o .wav", flush=True)
        self._toca(wav)

    def _laco(self):
        while True:
            try:
                o_que = self.fila.get(timeout=0.1)
            except self.queue.Empty:
                if self.bip_ativo:
                    try:
                        self._toca("bip", self.bip_pcm)
                    except Exception as e:
                        print(f"[Alerta] bipe falhou ({e})", flush=True)
                        time.sleep(1.0)
                continue
            try:
                c = self._cliente()
                if o_que == "bloqueado":
                    c.LedControl(255, 0, 0)
                    self._fala("Warning. Pose locked.", "pose_bloqueada")
                elif o_que == "aguardando":
                    c.LedControl(255, 140, 0)
                else:
                    c.LedControl(0, 255, 0)
                    self._fala("Pose unlocked.", "pose_liberada")
                    time.sleep(2.0)
                    c.LedControl(0, 0, 0)
            except Exception as e:
                print(f"[Alerta] falhou ({e}) — a trava continua valendo", flush=True)


def _limita(prev, alvo, passo):
    return prev + max(-passo, min(passo, alvo - prev))


class Trava:
    """Filtra TODO comando de braço/mão que passa pela ponte. Começa EM PÂNICO: nada de
    braço é repassado até o botão estar solto há >= 1 s e o PC pedir o rearme."""

    def __init__(self, args):
        self.a = args
        self.lock = threading.Lock()
        self.panico = True
        self.motivo = "início: rearme pendente (botão solto >= 1 s + pedido do PC)"
        self.gpio = "desativado" if args.sem_botao else "iniciando"
        self.botao = None
        self.botao_ok = bool(args.sem_botao)
        self.desde_ok = time.time() if args.sem_botao else None
        self.pedido_rearme = False
        self.lowstate = None
        self.maos = {}
        self.ultimo_corpo = None
        self.ultimo_mao = {}
        self.q_env = {}
        self.t_env = None
        self.q_mao_env = {}
        self.t_mao_env = {}
        self.congelado = None
        self.mao_congelada = {}
        self.rodando = True
        self.n_descartados = 0
        self.alerta = None

    # ── estado vindo do robô ──
    def novo_lowstate(self, msg):
        self.lowstate = msg

    def nova_mao(self, lado, msg):
        self.maos[lado] = msg

    def _q_medido(self, i):
        return None if self.lowstate is None else float(self.lowstate.motor_state[i].q)

    # ── transições ──
    def dispara(self, motivo):
        with self.lock:
            if self.panico:
                return False
            self.panico = True
            self.motivo = motivo
            self.congelado = {}
            for i in IDX_LIMITADOS:
                q = self._q_medido(i)
                self.congelado[i] = q if q is not None else self.q_env.get(i)
            self.mao_congelada = {}
            for lado, msg in self.maos.items():
                self.mao_congelada[lado] = [float(msg.motor_state[k].q) for k in range(NUM_HAND_MOTORS)]
        print(f"\n🛑 PÂNICO: {motivo} — braços congelados na pose medida; comandos do PC descartados.", flush=True)
        if self.alerta is not None:
            self.alerta.bloqueado()
        return True

    def _rearma(self):
        with self.lock:
            if self.congelado:
                for i, q in self.congelado.items():
                    if q is not None:
                        self.q_env[i] = q
            self.t_env = None
            for lado, qs in self.mao_congelada.items():
                self.q_mao_env[lado] = list(qs)
                self.t_mao_env[lado] = None
            self.panico = False
            self.motivo = ""
            self.congelado = None
            self.mao_congelada = {}
        print("\n✅ Rearmado: comandos do PC voltam a passar (lentos e leves).", flush=True)
        if self.alerta is not None:
            self.alerta.liberado()

    # ── filtros (chamados pelos laços de repasse) ──
    def filtra_corpo(self, data):
        with self.lock:
            if self.panico:
                self.n_descartados += 1
                return None
            agora = time.time()
            dt = 0.0 if self.t_env is None else min(agora - self.t_env, 0.05)
            data = copy.deepcopy(data)
            mc = data.get("motor_cmd", [])
            for i in IDX_LIMITADOS:
                if i >= len(mc) or int(mc[i].get("mode", 0)) != 1:
                    continue
                alvo = float(mc[i].get("q", 0.0))
                prev = self.q_env.get(i)
                if prev is None:
                    prev = self._q_medido(i)
                    prev = alvo if prev is None else prev
                q = _limita(prev, alvo, self.a.vel_braco * dt)
                mc[i]["q"] = q
                mc[i]["dq"] = 0.0
                # tau = compensação da GRAVIDADE do braço (02/10, como o gravity_compensation do LeRobot),
                # limitada a --tau-max-braco / --tau-max-punho (0 = zera, como antes); cintura: sempre 0
                tmax = ((self.a.tau_max_punho if i in IDX_PUNHOS else self.a.tau_max_braco)
                        if i in IDX_BRACOS else 0.0)
                mc[i]["tau"] = max(-tmax, min(tmax, float(mc[i].get("tau", 0.0))))
                self.q_env[i] = q
                if i in IDX_BRACOS:
                    teto = self.a.kp_punho if i in IDX_PUNHOS else self.a.kp_braco
                    mc[i]["kp"] = min(float(mc[i].get("kp", 0.0)), teto)
            self.t_env = agora
            self.ultimo_corpo = data
            return data

    def filtra_mao(self, lado, data):
        with self.lock:
            if self.panico:
                self.n_descartados += 1
                return None
            agora = time.time()
            t0 = self.t_mao_env.get(lado)
            dt = 0.0 if t0 is None else min(agora - t0, 0.05)
            data = copy.deepcopy(data)
            mc = data.get("motor_cmd", [])
            prevs = self.q_mao_env.get(lado)
            if prevs is None:
                msg = self.maos.get(lado)
                prevs = [float(msg.motor_state[k].q) if msg is not None else float(mc[k].get("q", 0.0))
                         for k in range(len(mc))]
            novos = []
            for k, m in enumerate(mc):
                q = _limita(prevs[k] if k < len(prevs) else float(m.get("q", 0.0)), float(m.get("q", 0.0)),
                            self.a.vel_mao * dt)
                m["q"] = q
                m["dq"] = 0.0
                m["tau"] = 0.0
                m["kp"] = min(float(m.get("kp", 0.0)), self.a.kp_mao)
                novos.append(q)
            self.q_mao_env[lado] = novos
            self.t_mao_env[lado] = agora
            self.ultimo_mao[lado] = data
            return data

    # ── threads ──
    def inicia(self, ctx, lowcmd_pub, crc, left_pub, right_pub):
        self.pubs = {"corpo": lowcmd_pub, "left": left_pub, "right": right_pub}
        if not self.a.sem_som:
            self.alerta = AlertaSonoro(os.path.dirname(os.path.abspath(__file__)), self.a.volume,
                                       self.a.volume_bip)
        self.crc = crc
        if not self.a.sem_botao:
            try:
                self.botao = BotaoPanico(self.a.gpio_entrada, self.a.gpio_saida)
                self.gpio = "ok"
                print(f"[Pânico] ✅ Botão: entrada {self.a.gpio_entrada} (GPIO6), saída {self.a.gpio_saida} "
                      "(GPIO4) em 0. Normal = entrada em 0 (NC fechado).", flush=True)
            except Exception as e:
                self.gpio = f"indisponivel: {e}"
                self.motivo = f"GPIO indisponível ({e}) — braço bloqueado"
                print(f"[Pânico] ❌ {self.motivo}", flush=True)
        else:
            print("[Pânico] ⚠️⚠️  --sem-botao: SEM BOTÃO DE PÂNICO. Só para bancada.", flush=True)
        self.sock_rearme = ctx.socket(zmq.PULL)
        self.sock_rearme.bind(f"tcp://0.0.0.0:{REARME_PORT}")
        self.sock_status = ctx.socket(zmq.PUB)
        self.sock_status.bind(f"tcp://0.0.0.0:{PANICO_PORT}")
        self.threads = [threading.Thread(target=f, daemon=True)
                        for f in (self._laco_botao, self._laco_segura, self._laco_status, self._laco_rearme)]
        for th in self.threads:
            th.start()

    def para(self):
        self.rodando = False
        for th in getattr(self, "threads", []):
            th.join(timeout=1.0)
        if self.botao is not None:
            self.botao.fecha()

    def _laco_botao(self):
        zeros = 0
        while self.rodando:
            agora = time.time()
            if self.botao is not None:
                try:
                    ok = self.botao.normal()
                except Exception:
                    ok = False   # erro de leitura conta como pânico
                if ok:
                    zeros = 0
                    if not self.botao_ok:
                        self.botao_ok, self.desde_ok = True, agora
                        if self.alerta is not None:
                            self.alerta.bip_desliga()   # soltou: para o bipe (o bloqueio continua)
                            if self.panico:
                                self.alerta.aguardando()   # vermelho -> amarelo: falta o rearme
                else:
                    zeros += 1
                    self.botao_ok, self.desde_ok = False, None
                    if zeros == 2:   # transição: botão acabou de abrir o circuito
                        mudou = self.dispara("botão de pânico apertado (ou cabo aberto)")
                        if self.alerta is not None:
                            if not mudou:
                                self.alerta.bloqueado()
                            self.alerta.bip_liga()      # bipa enquanto estiver apertado
                    elif zeros > 2:
                        self.dispara("botão de pânico apertado (ou cabo aberto)")
            elif not self.a.sem_botao:
                self.botao_ok = False   # GPIO indisponível: nunca rearma
            if self.pedido_rearme:
                self.pedido_rearme = False
                solto = self.botao_ok and self.desde_ok is not None and agora - self.desde_ok >= 1.0
                if not self.panico:
                    pass
                elif solto:
                    self._rearma()
                else:
                    print("[Pânico] rearme RECUSADO: botão não está solto há 1 s (ou GPIO indisponível).",
                          flush=True)
            time.sleep(0.002)

    def _laco_segura(self):
        """Em pânico, segura a pose congelada publicando no lugar do PC (100 Hz). Só se o PC
        já tinha assumido o braço; senão o controle da Unitree continua com ele e nada muda."""
        while self.rodando:
            with self.lock:
                corpo = mao = None
                if self.panico and self.ultimo_corpo is not None and self.congelado:
                    corpo = copy.deepcopy(self.ultimo_corpo)
                    mc = corpo.get("motor_cmd", [])
                    for i, q in self.congelado.items():
                        if q is not None and i < len(mc) and int(mc[i].get("mode", 0)) == 1:
                            mc[i]["q"], mc[i]["dq"], mc[i]["tau"] = q, 0.0, 0.0
                if self.panico:
                    mao = {}
                    for lado, qs in self.mao_congelada.items():
                        base = self.ultimo_mao.get(lado)
                        if base is None:
                            continue
                        d = copy.deepcopy(base)
                        for k, m in enumerate(d.get("motor_cmd", [])):
                            if k < len(qs):
                                m["q"], m["dq"], m["tau"] = qs[k], 0.0, 0.0
                        mao[lado] = d
            if corpo is not None:
                cmd = dict_to_lowcmd(corpo)
                cmd.crc = self.crc.Crc(cmd)
                self.pubs["corpo"].Write(cmd)
            for lado, d in (mao or {}).items():
                self.pubs[lado].Write(dict_to_handcmd(d))
            time.sleep(0.01)

    def _laco_status(self):
        while self.rodando:
            ok = self.desde_ok
            est = {"panico": self.panico, "motivo": self.motivo, "gpio": self.gpio,
                   "botao_ok": self.botao_ok, "botao_solto_s": 0.0 if ok is None else round(time.time() - ok, 2),
                   "descartados": self.n_descartados, "vel_braco": self.a.vel_braco,
                   "kp_braco": self.a.kp_braco, "kp_punho": self.a.kp_punho, "t": time.time()}
            try:
                self.sock_status.send(json.dumps(est).encode("utf-8"), zmq.NOBLOCK)
            except (zmq.Again, zmq.error.ContextTerminated):
                pass
            time.sleep(0.05)

    def _laco_rearme(self):
        while self.rodando:
            try:
                raw = self.sock_rearme.recv(zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.01)
                continue
            except zmq.error.ContextTerminated:
                break
            try:
                msg = json.loads(raw.decode("utf-8"))
            except Exception:
                continue
            if msg.get("panico"):
                self.dispara("pedido do PC")
            elif msg.get("rearmar"):
                self.pedido_rearme = True


TRAVA = None


class MotionSwitcher:
    def __init__(self):
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

    def current_mode_name(self) -> str:
        """Retorna o nome do modo atual ('ai', '' para debug, etc.)."""
        try:
            _, result = self.msc.CheckMode()
            return result.get("name", "")
        except Exception:
            return ""

    def enter_debug_mode(self) -> bool:
        """Mata a IA e entra em Low Level. Retorna True se bem-sucedido."""
        try:
            print("[MotionSwitcher] Matando a IA e entrando em Debug Mode (Low Level)...")
            while self.current_mode_name():
                self.msc.ReleaseMode()
                time.sleep(0.5)
            print("[MotionSwitcher] ✅ Debug Mode ativo.")
            return True
        except Exception as e:
            print(f"[MotionSwitcher] ❌ Falha ao entrar em Debug Mode: {e}")
            return False

    def enter_loco_mode(self) -> bool:
        """Ativa a IA (WBC/Loco). Retorna True se bem-sucedido."""
        try:
            print("[MotionSwitcher] Ativando IA (Loco/WBC)...")
            self.msc.SelectMode(nameOrAlias="ai")
            time.sleep(1.0)  # Aguarda a IA estabilizar
            name = self.current_mode_name()
            if name:
                print(f"[MotionSwitcher] ✅ Loco Mode ativo (modo='{name}').")
                return True
            else:
                print("[MotionSwitcher] ⚠️  SelectMode retornou mas CheckMode ainda vazio.")
                return False
        except Exception as e:
            print(f"[MotionSwitcher] ❌ Falha ao entrar em Loco Mode: {e}")
            return False


kTopicLowCommand_Debug  = "rt/lowcmd"
kTopicLowCommand_Motion = "rt/arm_sdk"
kTopicLowState          = "rt/lowstate"
kTopicDex3LeftCommand   = "rt/dex3/left/cmd"
kTopicDex3RightCommand  = "rt/dex3/right/cmd"
kTopicDex3LeftState     = "rt/dex3/left/state"
kTopicDex3RightState    = "rt/dex3/right/state"

LOWCMD_PORT    = 6000
LOWSTATE_PORT  = 6001
HANDSTATE_PORT = 6002
HANDCMD_PORT   = 6003
STATUS_PORT    = 6004   # ← Novo: publica o modo atual para o LeRobot
LOCOCMD_PORT   = 6005   # ← Novo: velocidade de locomoção (PC → robô), só em --loco

# Se o PC parar de mandar velocidade por mais que isto, o robô para sozinho.
# É a rede caindo, o teleop travando ou o operador tirando o óculos: em todos
# esses casos um G1 andando sem ninguém no comando é o pior desfecho possível.
LOCO_WATCHDOG_S = 0.4

NUM_MOTORS      = 35
NUM_HAND_MOTORS = 7

def lowstate_to_dict(msg: hg_LowState) -> dict[str, Any]:
    motor_states = []
    for i in range(NUM_MOTORS):
        temp = msg.motor_state[i].temperature
        avg_temp = float(sum(temp) / len(temp)) if isinstance(temp, list) else float(temp)
        motor_states.append({
            "q": float(msg.motor_state[i].q),
            "dq": float(msg.motor_state[i].dq),
            "tau_est": float(msg.motor_state[i].tau_est),
            "temperature": avg_temp,
        })
    return {
        "motor_state": motor_states,
        "imu_state": {
            "quaternion": [float(x) for x in msg.imu_state.quaternion],
            "gyroscope": [float(x) for x in msg.imu_state.gyroscope],
            "accelerometer": [float(x) for x in msg.imu_state.accelerometer],
            "rpy": [float(x) for x in msg.imu_state.rpy],
            "temperature": float(msg.imu_state.temperature),
        },
        "wireless_remote": base64.b64encode(bytes(msg.wireless_remote)).decode("ascii"),
        "mode_machine": int(msg.mode_machine),
    }

def handstate_to_dict(msg: HandState_, side: str) -> dict[str, Any]:
    motor_states = []
    for i in range(NUM_HAND_MOTORS):
        motor_states.append({
            "q": float(msg.motor_state[i].q),
            "dq": float(msg.motor_state[i].dq),
            "tau_est": float(msg.motor_state[i].tau_est),
        })

    press_sensors = []
    if hasattr(msg, 'press_sensor_state'):
        for p in msg.press_sensor_state:
            press_sensors.append({
                "pressure": list(p.pressure),
                "temperature": list(p.temperature)
            })

    return {
        "side": side,
        "motor_state": motor_states,
        "press_sensor_state": press_sensors,
    }


def dict_to_lowcmd(data: dict[str, Any]) -> hg_LowCmd:
    cmd = unitree_hg_msg_dds__LowCmd_()
    cmd.mode_pr = data.get("mode_pr", 0)
    cmd.mode_machine = data.get("mode_machine", 0)

    # Conversão Pura: Sem hacks de perna, sem mode_pr forçado.
    for i, motor_data in enumerate(data.get("motor_cmd", [])):
        cmd.motor_cmd[i].mode = motor_data.get("mode", 0)
        cmd.motor_cmd[i].q = motor_data.get("q", 0.0)
        cmd.motor_cmd[i].dq = motor_data.get("dq", 0.0)
        cmd.motor_cmd[i].kp = motor_data.get("kp", 0.0)
        cmd.motor_cmd[i].kd = motor_data.get("kd", 0.0)
        cmd.motor_cmd[i].tau = motor_data.get("tau", 0.0)

    return cmd


def dict_to_handcmd(data: dict[str, Any]) -> HandCmd_:
    cmd = unitree_hg_msg_dds__HandCmd_()
    for i, motor_data in enumerate(data.get("motor_cmd", [])):
        cmd.motor_cmd[i].mode = motor_data.get("mode", 0)
        cmd.motor_cmd[i].q = motor_data.get("q", 0.0)
        cmd.motor_cmd[i].dq = motor_data.get("dq", 0.0)
        cmd.motor_cmd[i].kp = motor_data.get("kp", 0.0)
        cmd.motor_cmd[i].kd = motor_data.get("kd", 0.0)
        cmd.motor_cmd[i].tau = motor_data.get("tau", 0.0)
    return cmd


def state_forward_loop(lowstate_sub, lowstate_sock, state_period, shutdown_event):
    last_state_time = 0.0
    while not shutdown_event.is_set():
        msg = lowstate_sub.Read()
        if msg is None: continue
        TRAVA.novo_lowstate(msg)   # v3: pose medida, para congelar no pânico
        now = time.time()
        if now - last_state_time >= state_period:
            state_dict = lowstate_to_dict(msg)
            payload = json.dumps({"topic": kTopicLowState, "data": state_dict}).encode("utf-8")
            try: lowstate_sock.send(payload, zmq.NOBLOCK)
            except (zmq.Again, zmq.error.ContextTerminated): pass
            last_state_time = now

def handstate_forward_loop(left_sub, right_sub, handstate_sock, state_period, shutdown_event):
    last_left_time = 0.0
    last_right_time = 0.0
    while not shutdown_event.is_set():
        now = time.time()
        msg_left = left_sub.Read()
        if msg_left is not None: TRAVA.nova_mao("left", msg_left)
        if msg_left is not None and (now - last_left_time >= state_period):
            state_dict = handstate_to_dict(msg_left, "left")
            payload = json.dumps({"topic": kTopicDex3LeftState, "data": state_dict}).encode("utf-8")
            try: handstate_sock.send(payload, zmq.NOBLOCK)
            except (zmq.Again, zmq.error.ContextTerminated): pass
            last_left_time = now
        
        msg_right = right_sub.Read()
        if msg_right is not None: TRAVA.nova_mao("right", msg_right)
        if msg_right is not None and (now - last_right_time >= state_period):
            state_dict = handstate_to_dict(msg_right, "right")
            payload = json.dumps({"topic": kTopicDex3RightState, "data": state_dict}).encode("utf-8")
            try: handstate_sock.send(payload, zmq.NOBLOCK)
            except (zmq.Again, zmq.error.ContextTerminated): pass
            last_right_time = now
        time.sleep(0.001)

def status_broadcast_loop(status_sock, active_mode: str, shutdown_event):
    """
    Publica o modo ativo a cada 0.5s na porta 6004.
    O LeRobot lê isso no connect() para saber para qual tópico mandar comandos.
    Mensagem: {"robot_mode": "loco"} ou {"robot_mode": "debug"}
    """
    payload = json.dumps({"robot_mode": active_mode}).encode("utf-8")
    while not shutdown_event.is_set():
        try:
            status_sock.send(payload, zmq.NOBLOCK)
        except (zmq.Again, zmq.error.ContextTerminated):
            pass
        time.sleep(0.5)


def cmd_forward_loop(lowcmd_sock, lowcmd_pub, crc, shutdown_event):
    """
    Loop de encaminhamento de comandos de corpo.
    O tópico de destino (lowcmd_pub) já foi escolhido no startup — sem troca de modo aqui.
    """
    while not shutdown_event.is_set():
        try:
            payload = lowcmd_sock.recv()
        except zmq.ContextTerminated:
            break

        msg_dict = json.loads(payload.decode("utf-8"))
        data = TRAVA.filtra_corpo(msg_dict.get("data", {}))   # v3: pânico + lento/leve
        if data is None:
            continue
        cmd = dict_to_lowcmd(data)
        cmd.crc = crc.Crc(cmd)
        lowcmd_pub.Write(cmd)

def handcmd_forward_loop(handcmd_sock, left_pub, right_pub, shutdown_event):
    while not shutdown_event.is_set():
        try: payload = handcmd_sock.recv(zmq.NOBLOCK)
        except zmq.Again:
            time.sleep(0.001)
            continue
        except zmq.ContextTerminated: break
        
        msg_dict = json.loads(payload.decode("utf-8"))
        topic = msg_dict.get("topic", "")
        lado = "left" if topic == kTopicDex3LeftCommand else "right" if topic == kTopicDex3RightCommand else None
        if lado is None:
            continue
        data = TRAVA.filtra_mao(lado, msg_dict.get("data", {}))   # v3: pânico + lento/leve
        if data is None:
            continue
        cmd = dict_to_handcmd(data)
        if topic == kTopicDex3LeftCommand: left_pub.Write(cmd)
        elif topic == kTopicDex3RightCommand: right_pub.Write(cmd)


def loco_cmd_loop(loco_sock, loco_client, shutdown_event):
    """Recebe velocidade do PC e chama o LocoClient AQUI, no robô.

    O DDS fica todo deste lado de propósito: do PC só sai um JSON por ZMQ,
    pela mesma ponte que já carrega lowcmd e handcmd. Chamar LocoClient de
    fora exigiria abrir o domínio DDS do robô para a rede — mais superfície,
    e um segundo participante DDS competindo pelo mesmo tópico.

    Mensagens aceitas:
        {"vx": float, "vy": float, "vyaw": float}
        {"damp": true}                              → parada macia
    """
    ultimo_cmd = 0.0
    andando = False

    while not shutdown_event.is_set():
        try:
            payload = loco_sock.recv(zmq.NOBLOCK)
        except zmq.Again:
            # Sem mensagem: é aqui que o watchdog trabalha.
            if andando and (time.time() - ultimo_cmd) > LOCO_WATCHDOG_S:
                try:
                    loco_client.Move(0.0, 0.0, 0.0, continous_move=False)
                except Exception as e:
                    print(f"[Loco] Falha ao parar no watchdog: {e}")
                andando = False
                print("[Loco] ⏱️  Watchdog: sem comando do PC, robô parado.")
            time.sleep(0.005)
            continue
        except zmq.ContextTerminated:
            break

        try:
            msg = json.loads(payload.decode("utf-8"))
        except Exception:
            continue

        ultimo_cmd = time.time()

        if msg.get("damp"):
            try:
                loco_client.Damp()
                print("[Loco] 🛑 Damping por pedido do PC.")
            except Exception as e:
                print(f"[Loco] Falha no Damp: {e}")
            andando = False
            continue

        vx = float(msg.get("vx", 0.0))
        vy = float(msg.get("vy", 0.0))
        vyaw = float(msg.get("vyaw", 0.0))

        # Teto também aqui, não só no PC: o servidor não deve confiar que o
        # cliente respeitou o próprio limite.
        vx = max(-0.5, min(0.5, vx))
        vy = max(-0.5, min(0.5, vy))
        vyaw = max(-0.5, min(0.5, vyaw))

        try:
            loco_client.Move(vx, vy, vyaw, continous_move=False)
        except Exception as e:
            print(f"[Loco] Falha ao mover: {e}")
            continue

        andando = not (vx == 0.0 and vy == 0.0 and vyaw == 0.0)


def main():
    parser = argparse.ArgumentParser(description="G1 ZMQ Bridge Server")
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument(
        "--debug",
        action="store_true",
        help="Low Level (Debug Mode, rt/lowcmd): MATA a IA da Unitree. Só com o robô "
             "suspenso ou apoiado — em pé, ele cai.",
    )
    modo.add_argument(
        "--loco",
        action="store_true",
        help="High Level (Loco/WBC). Já é o padrão; aceito só por compatibilidade.",
    )
    parser.add_argument("--vel-braco", type=float, default=0.3, help="rad/s máx. do alvo de braço e yaw da cintura")
    parser.add_argument("--kp-braco", type=float, default=20.0, help="teto de kp: ombros e cotovelos (padrão da classe: 80)")
    parser.add_argument("--kp-punho", type=float, default=10.0, help="teto de kp: punhos (padrão da classe: 40)")
    parser.add_argument("--tau-max-braco", type=float, default=0.0,
                        help="Nm: teto do tau (compensação da gravidade) de ombro/cotovelo vindo do PC; 0 = zera")
    parser.add_argument("--tau-max-punho", type=float, default=0.0, help="Nm: idem para os punhos")
    parser.add_argument("--vel-mao", type=float, default=1.0, help="rad/s máx. dos dedos Dex3")
    parser.add_argument("--kp-mao", type=float, default=1.0, help="teto de kp dos dedos Dex3")
    parser.add_argument("--gpio-entrada", default="PCC.03", help="GPIO6 da Unitree (pino 130 do NX)")
    parser.add_argument("--gpio-saida", default="PI.04", help="GPIO4 da Unitree (pino 234 do NX)")
    parser.add_argument("--sem-botao", action="store_true", help="SEM pânico físico — só para bancada")
    parser.add_argument("--volume", type=int, default=100, help="volume do alerta no alto-falante (0-100)")
    parser.add_argument("--volume-bip", type=int, default=50,
                        help="volume do bipe de emergência enquanto o botão está apertado (0-100)")
    parser.add_argument("--sem-som", action="store_true", help="não fala nem acende LED no pânico")
    args = parser.parse_args()
    global TRAVA
    TRAVA = Trava(args)

    use_loco = not args.debug
    active_mode = "loco" if use_loco else "debug"

    print("=========================================================")
    print(f"🚀 G1 ZMQ Bridge - Modo: {'HIGH LEVEL (Loco/WBC) 🏃' if use_loco else 'LOW LEVEL (Debug) 🛑'}")
    print("=========================================================")

    ChannelFactoryInitialize(0)

    # ----------------------------------------------------------------
    # SWITCH DE MODO: acontece UMA VEZ, antes de qualquer loop
    # ----------------------------------------------------------------
    ms = MotionSwitcher()
    current_hw_mode = ms.current_mode_name()
    print(f"[Startup] Modo atual do robô: '{current_hw_mode or 'debug/vazio'}'")

    if use_loco:
        if not current_hw_mode:
            # Robô está em debug — precisa ativar a IA
            ok = ms.enter_loco_mode()
            if not ok:
                print("[Startup] ⚠️  Não foi possível ativar Loco Mode. Verifique o robô e tente novamente.")
                return
        else:
            print(f"[Startup] ✅ Robô já está em Loco Mode ('{current_hw_mode}'). Nenhuma troca necessária.")
        lowcmd_topic = kTopicLowCommand_Motion
    else:
        if current_hw_mode:
            # Robô está em IA — precisa matar para assumir controle bruto
            ok = ms.enter_debug_mode()
            if not ok:
                print("[Startup] ⚠️  Não foi possível entrar em Debug Mode. Verifique o robô e tente novamente.")
                return
        else:
            print("[Startup] ✅ Robô já está em Debug Mode. Nenhuma troca necessária.")
        lowcmd_topic = kTopicLowCommand_Debug

    print(f"[Startup] Tópico de comando do corpo: {lowcmd_topic}")
    print(f"[Startup] Publicando modo '{active_mode}' na porta {STATUS_PORT} para o LeRobot.")

    crc = CRC()

    # Publicador único — tópico fixado pelo modo escolhido no startup
    lowcmd_pub = ChannelPublisher(lowcmd_topic, hg_LowCmd)
    lowcmd_pub.Init()

    lowstate_sub = ChannelSubscriber(kTopicLowState, hg_LowState)
    lowstate_sub.Init()

    left_hand_cmd_pub = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
    left_hand_cmd_pub.Init()
    right_hand_cmd_pub = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
    right_hand_cmd_pub.Init()

    left_hand_state_sub = ChannelSubscriber(kTopicDex3LeftState, HandState_)
    left_hand_state_sub.Init()
    right_hand_state_sub = ChannelSubscriber(kTopicDex3RightState, HandState_)
    right_hand_state_sub.Init()

    ctx = zmq.Context.instance()

    lowcmd_sock = ctx.socket(zmq.PULL)
    lowcmd_sock.bind(f"tcp://0.0.0.0:{LOWCMD_PORT}")

    lowstate_sock = ctx.socket(zmq.PUB)
    lowstate_sock.bind(f"tcp://0.0.0.0:{LOWSTATE_PORT}")

    handstate_sock = ctx.socket(zmq.PUB)
    handstate_sock.bind(f"tcp://0.0.0.0:{HANDSTATE_PORT}")

    handcmd_sock = ctx.socket(zmq.PULL)
    handcmd_sock.bind(f"tcp://0.0.0.0:{HANDCMD_PORT}")

    # ← NOVO: socket de status — o LeRobot lê aqui para saber o modo ativo
    status_sock = ctx.socket(zmq.PUB)
    status_sock.bind(f"tcp://0.0.0.0:{STATUS_PORT}")

    # ← NOVO: velocidade de locomoção. Só existe em --loco: sem o WBC ativo
    # o LocoClient não tem o que comandar, e abrir a porta daria a falsa
    # impressão de que o analógico do VR faria alguma coisa.
    loco_sock = None
    loco_client = None
    if use_loco:
        try:
            from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
            loco_client = LocoClient()
            loco_client.SetTimeout(0.0001)
            loco_client.Init()
            loco_sock = ctx.socket(zmq.PULL)
            loco_sock.bind(f"tcp://0.0.0.0:{LOCOCMD_PORT}")
            print(f"[Startup] 🏃 Locomoção habilitada — escutando velocidade na porta {LOCOCMD_PORT}.")
        except Exception as e:
            loco_client = None
            loco_sock = None
            print(f"[Startup] ⚠️  Locomoção DESABILITADA (LocoClient falhou: {e}). O resto segue normal.")

    TRAVA.inicia(ctx, lowcmd_pub, crc, left_hand_cmd_pub, right_hand_cmd_pub)   # v3

    shutdown_event = threading.Event()

    t_state = threading.Thread(
        target=state_forward_loop,
        args=(lowstate_sub, lowstate_sock, 0.002, shutdown_event),
        daemon=True,
    )
    t_handstate = threading.Thread(
        target=handstate_forward_loop,
        args=(left_hand_state_sub, right_hand_state_sub, handstate_sock, 0.002, shutdown_event),
        daemon=True,
    )
    t_handcmd = threading.Thread(
        target=handcmd_forward_loop,
        args=(handcmd_sock, left_hand_cmd_pub, right_hand_cmd_pub, shutdown_event),
        daemon=True,
    )
    # ← NOVO: thread que fica anunciando o modo ativo
    t_status = threading.Thread(
        target=status_broadcast_loop,
        args=(status_sock, active_mode, shutdown_event),
        daemon=True,
    )

    t_loco = None
    if loco_sock is not None:
        t_loco = threading.Thread(
            target=loco_cmd_loop,
            args=(loco_sock, loco_client, shutdown_event),
            daemon=True,
        )

    t_state.start()
    t_handstate.start()
    t_handcmd.start()
    t_status.start()
    if t_loco is not None:
        t_loco.start()

    print(f"\n[INFO] Servidor ZMQ escutando na porta {LOWCMD_PORT} para comandos de corpo...")
    print(f"[INFO] LeRobot pode conectar. Modo '{active_mode}' será informado na porta {STATUS_PORT}.")

    try:
        cmd_forward_loop(lowcmd_sock, lowcmd_pub, crc, shutdown_event)
    except KeyboardInterrupt:
        print("\nDesligando a bridge...")
    finally:
        # Parar os pés ANTES de derrubar os sockets: se a bridge cair com uma
        # velocidade pendente, o robô continua andando sem ninguém escutando.
        if loco_client is not None:
            try:
                loco_client.Move(0.0, 0.0, 0.0, continous_move=False)
            except Exception:
                pass
        shutdown_event.set()
        TRAVA.para()   # v3: solta os GPIOs (a saída primeiro)
        ctx.term()
        t_state.join(timeout=2.0)
        t_handstate.join(timeout=2.0)
        t_handcmd.join(timeout=2.0)
        t_status.join(timeout=2.0)
        if t_loco is not None:
            t_loco.join(timeout=2.0)
        print("Finalizado.")


if __name__ == "__main__":
    main()