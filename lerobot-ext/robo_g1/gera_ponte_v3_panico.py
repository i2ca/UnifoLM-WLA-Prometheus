#!/usr/bin/env python3
"""Gera `dex3_g1_server_v3_panico.py` a partir da ponte v2 (sem alterar a v2).

    python gera_ponte_v3_panico.py dex3_g1_server_v2.py dex3_g1_server_v3_panico.py

O que a v3 acrescenta (tudo DENTRO do robô, o último passo antes do DDS):

  PÂNICO — botão cogumelo NC entre GPIO4 (PI.04, saída em 0) e GPIO6 (PCC.03, entrada com
    pull-up interno do Jetson). Normal (NC fechado): GPIO6 lê 0. Botão apertado OU cabo
    aberto: o pull-up leva o GPIO6 a 1 -> PÂNICO travado. (Medido em 28/09: com o GPIO4 em 1
    o GPIO6 lia 1 nos dois estados, por causa do pull-up — por isso a saída é 0.)
    Em pânico a ponte DESCARTA os comandos de braço/mão do PC e publica ela mesma,
    a 100 Hz, a pose MEDIDA no instante do aperto (mantendo o peso do arm_sdk e a
    cintura como estavam). Só rearma com o botão solto há >= 1 s E um pedido
    explícito do PC (porta 6006). Sem GPIO utilizável, a ponte fica em pânico.
    LED do peito: VERMELHO = botão apertado (bipe); AMARELO = botão solto mas ainda
    travado, esperando o rearme (painel de diagnóstico, porta 8095, ou PC); VERDE 2 s =
    rearmado. A ponte sobe AMARELA.

  LENTO E LEVE — nos braços (15-28) e no yaw da cintura (12):
    o alvo anda no máximo --vel-braco rad/s a partir da pose medida (o primeiro
    comando não salta), e o kp do braço é cortado em --kp-braco / --kp-punho.
    Nas mãos Dex3, --vel-mao e --kp-mao.

  Portas novas: 6006 PULL (PC -> robô: {"rearmar": true} ou {"panico": true})
                6007 PUB  (robô -> PC: estado do pânico, 20 Hz)
"""
import sys

origem, destino = sys.argv[1], sys.argv[2]
t = open(origem, encoding="utf-8").read()


def troca(antigo, novo, n=1):
    global t
    achados = t.count(antigo)
    if achados != n:
        sys.exit(f"ERRO: trecho aparece {achados}x (esperado {n}):\n{antigo[:300]}")
    t = t.replace(antigo, novo)


# ── cabeçalho ─────────────────────────────────────────────────────────────
troca('DDS-to-ZMQ bridge server for Unitree G1 robot with Dex3 hands.\n',
      'DDS-to-ZMQ bridge server for Unitree G1 robot with Dex3 hands.\n\n'
      'VERSÃO v3 — BOTÃO DE PÂNICO + BRAÇOS LENTOS E LEVES (gerada de dex3_g1_server_v2.py\n'
      'por gera_ponte_v3_panico.py). Ver a docstring do gerador para o comportamento.\n'
      '    python dex3_g1_server_v3_panico.py                     # loco, pânico no GPIO4/GPIO6\n'
      '    python dex3_g1_server_v3_panico.py --vel-braco 0.2     # ainda mais lento\n')

# ── módulo do pânico ──────────────────────────────────────────────────────
troca('from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient\n',
      '''from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
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
        print(f"\\n🛑 PÂNICO: {motivo} — braços congelados na pose medida; comandos do PC descartados.", flush=True)
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
        print("\\n✅ Rearmado: comandos do PC voltam a passar (lentos e leves).", flush=True)
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
''')

# ── estado para a trava ───────────────────────────────────────────────────
troca('''        msg = lowstate_sub.Read()
        if msg is None: continue
''', '''        msg = lowstate_sub.Read()
        if msg is None: continue
        TRAVA.novo_lowstate(msg)   # v3: pose medida, para congelar no pânico
''')
troca('''        msg_left = left_sub.Read()
''', '''        msg_left = left_sub.Read()
        if msg_left is not None: TRAVA.nova_mao("left", msg_left)
''')
troca('''        msg_right = right_sub.Read()
''', '''        msg_right = right_sub.Read()
        if msg_right is not None: TRAVA.nova_mao("right", msg_right)
''')

# ── filtros nos laços de repasse ──────────────────────────────────────────
troca('''        msg_dict = json.loads(payload.decode("utf-8"))
        cmd = dict_to_lowcmd(msg_dict.get("data", {}))
        cmd.crc = crc.Crc(cmd)
        lowcmd_pub.Write(cmd)
''', '''        msg_dict = json.loads(payload.decode("utf-8"))
        data = TRAVA.filtra_corpo(msg_dict.get("data", {}))   # v3: pânico + lento/leve
        if data is None:
            continue
        cmd = dict_to_lowcmd(data)
        cmd.crc = crc.Crc(cmd)
        lowcmd_pub.Write(cmd)
''')
troca('''        msg_dict = json.loads(payload.decode("utf-8"))
        cmd = dict_to_handcmd(msg_dict.get("data", {}))
        topic = msg_dict.get("topic", "")
''', '''        msg_dict = json.loads(payload.decode("utf-8"))
        topic = msg_dict.get("topic", "")
        lado = "left" if topic == kTopicDex3LeftCommand else "right" if topic == kTopicDex3RightCommand else None
        if lado is None:
            continue
        data = TRAVA.filtra_mao(lado, msg_dict.get("data", {}))   # v3: pânico + lento/leve
        if data is None:
            continue
        cmd = dict_to_handcmd(data)
''')

# ── argumentos e partida ──────────────────────────────────────────────────
troca('''    args = parser.parse_args()
''', '''    parser.add_argument("--vel-braco", type=float, default=0.3, help="rad/s máx. do alvo de braço e yaw da cintura")
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
''')
troca('''    shutdown_event = threading.Event()
''', '''    TRAVA.inicia(ctx, lowcmd_pub, crc, left_hand_cmd_pub, right_hand_cmd_pub)   # v3

    shutdown_event = threading.Event()
''')
troca('''        shutdown_event.set()
        ctx.term()
''', '''        shutdown_event.set()
        TRAVA.para()   # v3: solta os GPIOs (a saída primeiro)
        ctx.term()
''')

open(destino, "w", encoding="utf-8").write(t)
print(f"ok -> {destino}")
