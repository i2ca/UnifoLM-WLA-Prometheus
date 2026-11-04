#!/usr/bin/env python
"""CABINE DO ER-1 (27/10): conversar com o ER-1 vendo a câmera do robô, e deixar ele PLANEJAR e EXECUTAR um objetivo
com as tarefas que o WLA sabe fazer — uma de cada vez, checando "terminou?" antes de passar para a próxima.

  conversa  pergunta livre ao ER-1 (com a imagem da cabeça, adaptador desligado), resposta em português
  objetivo  o ER-1 olha a mesa e escolhe a SEQUÊNCIA de tarefas (só as da lista HABILIDADES); o orquestrador manda
            cada uma para a cabine do robô (POST /tarefa), pergunta "terminou?" ao ER-1 (LoRA, a cada --checa-s) e só
            avança com dois "sim" seguidos. Não terminou em --tempo-max s: volta à pose inicial e tenta de novo uma vez;
            falhou de novo: para e avisa.
  parar     manda a cabine do robô parar (POST /parar) e cancela o objetivo
  falar     o robô fala o texto (voz do G1, pelo painel :8095/fala) e o texto aparece no chat

TRAVAS (para nada ir ao robô por acidente):
  1. o objetivo só vira PLANO (nada vai ao robô); executar exige CONFIRMAR o plano, e o plano expira em 60 s;
  2. a 2ª tentativa de uma etapa também pede CONFIRMAR (60 s; sem confirmar, para);
  3. confirmar só funciona com o robô LIBERADO (chave na página), que volta a TRAVADO sozinho no fim de cada
     objetivo, no PARAR e após 10 min sem uso;
  3. antes de cada etapa confere a cabine do robô: no ar, sem parada pedida e sem outra tarefa rodando;
  4. só as frases da lista HABILIDADES vão ao robô, no máximo 4 etapas por objetivo; Enter no campo só CONVERSA.

Precisa no ar: o ER-1 (sobe_er1_pergunta.sh, :8098) e a cabine do robô com o WLA (sobe_cabine_wla.sh, :8090).
Rode a cabine do robô COM TERMINOU=1: quem pergunta "terminou?" é o executor (revezando a GPU com o WLA); aqui só
se lê o resultado no /estado.json da cabine (perguntar daqui atrasava o WLA e a pega falhava, 27/10).

    python ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/cabine_er1.py      # e abra http://<pgx>:8092
"""
import argparse
import codecs
import json
import queue
import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CABINE_DIR = Path(__file__).resolve().parent.parent / "pontes" / "cabine"   # 03/11: three.js e o URDF (mundo 3D)


# o que o WLA atual sabe fazer (frases do treino; o servidor traduz para as descritivas)
HABILIDADES = [
    "Pick up the white mug.",
    "Place the white mug under the coffee strainer.",
    "Pick up the apple.",
    "Pick up the apple and place it on the black X.",
]
# 28/10: palavras que o ER-1 usa na LISTA do que vê -> o objeto conhecido
OBJETOS_LISTA = {   # SEM cor no nome: a cor quem diz é o ER-1 (28/10: "caneca branca" fazia ele chamar a amarela de branca)
    "chaleira": ["kettle", "teapot"],
    "coador de café": ["dripper", "coffee filter", "filter cone", "strainer", "funnel", "pour.over"],
    "caneca": ["mug", "cup"],
    "pote de café": ["jar", "canister"],
    "tampa do pote": ["lid"],
    "concha de café": ["scoop", "spoon", "ladle"],
    "X preto na mesa": ["x mark", "black x", "cross", r"\bx\b"],
    "maçã": ["apple"],
}
PT = {
    "Pick up the white mug.": "pegar a caneca branca",
    "Place the white mug under the coffee strainer.": "colocar a caneca embaixo do coador",
    "Pick up the apple.": "pegar a maçã",
    "Pick up the apple and place it on the black X.": "pegar a maçã e colocar no X",
}


class Orquestrador:
    def __init__(self, a):
        self.a = a
        self.msgs, self.trava = [], threading.Lock()
        self.rodando, self.cancela = False, threading.Event()
        self.plano, self.passo = [], -1
        self.pendente, self.pendente_t = None, 0.0     # plano esperando CONFIRMAR (expira em 60 s)
        self.historico = []                            # (pergunta, resposta) — memória da conversa
        self.seq_enviado = None                        # número da última tarefa que mandamos à cabine do robô
        self.retry = None                              # (frase, t) esperando CONFIRMAR para tentar de novo
        self.retry_ok = threading.Event()
        self.liberado, self.liberado_t = False, 0.0    # chave "robô liberado" (trava sozinha)
        # 28/10: VOZ — "Prometheus, ..." pelo microfone do robô (ouvido_g1.py) e a luz do rosto pelo painel
        self.led_base = (0, 0, 0)                      # a cor "de descanso" (muda por voz: "muda a luz para azul")
        # 29/10: EXECUTAR POR VOZ — só com a chave "🎤 voz pode executar" (trava sozinha em 10 min); o plano falado
        # espera um "sim" falado (sem precisar de "Prometheus") por --sim-s segundos
        self.voz_executa, self.voz_executa_t = False, 0.0
        self.aguarda_sim_ate = 0.0
        # 30/10: MEMÓRIA (memoria_er1.py: o que ele aprende, buscado a cada pergunta) e VOZ NO NAVEGADOR (a última fala
        # fica em /voz.wav; a página toca quando o número muda — funciona com o robô desligado)
        from memoria_er1 import Memoria
        self.mem = Memoria(a.memoria_db)
        self.audio_n, self.audio_wav, self.audios = 0, b"", {}
        self.voz = a.tts_voz
        # 03/11: ONDE ele fala. "auto" = responde por onde foi perguntado (digitou na página -> só na página; falou
        # com o robô -> só no alto-falante dele). "robo" | "pagina" | "ambos" fixam. Antes saía nos dois sempre.
        self.saida_voz, self.origem = a.saida_voz, "pagina"
        # 30/10: VOZ EM STREAMING — as frases da resposta entram nesta fila assim que o ER-1 as termina
        self.fila_fala = queue.Queue()
        threading.Thread(target=self._fala_fila, daemon=True).start()
        self.ouvido = None
        if getattr(a, "ouvido", 0):
            from ouvido_g1 import Ouvido
            self.ouvido = Ouvido(self.voz_pergunta, self.voz_estado, ip_local=a.ip_local,
                                 log=lambda t: print(t, flush=True))
            self.ouvido.ligado = bool(a.ouvir_ao_iniciar)   # 29/10: começa DESLIGADO; liga na chave 🎤 da página

    # ── registro (a página lê por /estado) ──
    def diz(self, quem, texto, **extra):
        with self.trava:
            self.msgs.append({"t": time.strftime("%H:%M:%S"), "quem": quem, "texto": texto, **extra})
            self.msgs = self.msgs[-300:]
        print(f"[{quem}] {texto}", flush=True)

    # ── ER-1 ──
    def er1(self, q, livre=True, max_tok=200, lora=0, cam="head_stereo_left", timeout=60):
        u = f"{self.a.er1}/pergunta?" + urllib.parse.urlencode(
            {"q": q, "livre": int(livre), "max": max_tok, "lora": lora, "cam": cam})
        r = json.loads(urllib.request.urlopen(u, timeout=timeout).read())
        if r.get("erro"):
            raise RuntimeError(r["erro"])
        return (r.get("cru") or r.get("resposta") or "").strip(), r.get("ms")

    def _fala_fila(self):
        while True:
            f = self.fila_fala.get()
            self.fala(f, "ER-1", registra=False)

    FIM_FRASE = re.compile(r"[.!?…](?=\s|$)")
    PARA_EM = re.compile(r"\n\s*(Usuário|Prometheus)\s*:")

    def er1_stream(self, q, max_tok=200, cam="", ao_frase=None, timeout=60):
        """30/10: a resposta livre do ER-1 em pedaços (/pergunta_stream); cada frase completa (>= 20 caracteres) vai
        para ao_frase na hora — a voz começa antes de ele terminar de pensar."""
        u = f"{self.a.er1}/pergunta_stream?" + urllib.parse.urlencode({"q": q, "max": max_tok, "cam": cam})
        t0 = time.time()
        dec = codecs.getincrementaldecoder("utf-8")(errors="ignore")
        txt, ptr = "", 0
        with urllib.request.urlopen(u, timeout=timeout) as resp:
            while True:
                b = resp.read1(512)
                if not b:
                    break
                txt += dec.decode(b)
                if txt.lstrip().startswith('{"erro"'):
                    continue
                m = self.PARA_EM.search(txt)   # ele às vezes continua a conversa sozinho: para de falar ali
                fim = m.start() if m else len(txt)
                if ao_frase and ptr == 0:   # 1ª fala: pode começar numa vírgula, se já tem corpo (voz sai antes)
                    v = re.search(r"[,;:](?=\s)", txt[45:fim])
                    if v and not self.FIM_FRASE.search(txt, 0, 45 + v.end()):
                        ao_frase(txt[:45 + v.end()])
                        ptr = 45 + v.end()
                while ao_frase:
                    f = self.FIM_FRASE.search(txt, ptr, fim)
                    if not f:
                        break
                    if f.end() - ptr >= 20:
                        ao_frase(txt[ptr:f.end()])
                        ptr = f.end()
                    else:
                        nxt = self.FIM_FRASE.search(txt, f.end(), fim)
                        if not nxt:
                            break
                        ao_frase(txt[ptr:nxt.end()])
                        ptr = nxt.end()
                if m:
                    txt = txt[:m.start()]
                    break
        txt += dec.decode(b"", final=True) if not self.PARA_EM.search(txt) else ""
        if txt.lstrip().startswith('{"erro"'):
            raise RuntimeError(json.loads(txt)["erro"])
        if ao_frase and txt[ptr:].strip():
            ao_frase(txt[ptr:])
        return txt.strip(), round((time.time() - t0) * 1000)

    def terminou(self, frase):
        q = f'The robot\'s task is: "{frase}" Has the robot completed this task? Answer only yes or no.'
        txt, _ = self.er1(q, max_tok=4, lora=1, timeout=20)
        return txt.lower().startswith("yes")

    # ── cabine do robô ──
    def cabine(self, caminho, corpo=None):
        dados = json.dumps(corpo or {}).encode()
        req = urllib.request.Request(f"{self.a.cabine}{caminho}", data=dados, method="POST",
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=10).read())
        if caminho == "/tarefa" and "tarefa_seq" in r:
            self.seq_enviado = r["tarefa_seq"]
        return r

    def cabine_reiniciou(self):
        """True se a cabine do robô foi reiniciada depois do nosso último envio (o número da tarefa voltou para trás)."""
        try:
            e = json.loads(urllib.request.urlopen(f"{self.a.cabine}/estado.json", timeout=5).read())
        except Exception:  # noqa: BLE001
            return True   # não responde: trata como reiniciada (não manda nada)
        return self.seq_enviado is not None and int(e.get("tarefa_seq", 0)) < self.seq_enviado

    # ── voz do robô ──
    # 28/10: como se FALA no Brasil (o Piper soletrava as siglas); só na voz — na tela continua a sigla
    PRONUNCIA = [(re.compile(r"\bUFES\b", re.IGNORECASE), "Ufes"), (re.compile(r"\bLCAD\b", re.IGNORECASE), "éle cádi"),
                 (re.compile(r"\bER-1\b"), "é érre um"), (re.compile(r"\bWLA\b"), "dáblio éle á"),
                 # 30/10: unidades e símbolos por extenso (a voz lia "°C" e "ml" letra por letra)
                 (re.compile(r"\s*°\s*C\b"), " graus"), (re.compile(r"(\d)\s*ml\b"), r"\1 mililitros"),
                 (re.compile(r"(\d)\s*g\b"), r"\1 gramas"), (re.compile(r"(\d)\s*kg\b"), r"\1 quilos"),
                 (re.compile(r"(\d)\s*km\b", re.IGNORECASE), r"\1 quilômetros"), (re.compile(r"(\d)\s*%"), r"\1 por cento"),
                 (re.compile(r"[*_#`•→]+"), " "), (re.compile(r"\s*·\s*"), ", ")]

    def _wav(self, texto):
        """Voz em português -> WAV (o navegador toca este). 30/10: Kokoro (voz_kokoro.py, natural); se não responder,
        o Piper (pt_BR-faber, robótico) para não ficar mudo."""
        for padrao, fala in self.PRONUNCIA:
            texto = padrao.sub(fala, texto)
        if self.a.tts_url and self.voz != "faber":
            try:
                u = f"{self.a.tts_url}/fala?" + urllib.parse.urlencode({"texto": texto, "voz": self.voz,
                                                                        "vel": self.a.tts_vel})
                return urllib.request.urlopen(u, timeout=15).read()
            except Exception as e:  # noqa: BLE001
                print(f"[voz] Kokoro não respondeu ({e}); usando o Piper", flush=True)
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            # 30/10: Faber mais FLUIDA (fonemas um pouco mais curtos, mais variação, menos pausa entre frases)
            subprocess.run([f"{self.a.piper}/.venv/bin/piper", "-m", f"{self.a.piper}/{self.a.voz}", "-f", f"{d}/a.wav",
                            "--length-scale", "0.92", "--noise-scale", "0.75", "--noise-w", "0.95",
                            "--sentence-silence", "0.1"], input=texto.encode(), check=True, capture_output=True)
            return open(f"{d}/a.wav", "rb").read()

    def _pcm(self, wav):
        """WAV -> PCM 16 kHz mono. A voz interna do G1 só fala inglês/chinês; o painel do robô toca o PCM pelo mesmo
        caminho do alarme de pânico."""
        import subprocess
        return subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-ar", "16000", "-ac", "1", "-f", "s16le",
                               "pipe:1"], input=wav, check=True, capture_output=True).stdout

    # ── luz do rosto e voz ──
    CORES = {"vermelh": ((255, 0, 0), "vermelha"), "verde": ((0, 255, 0), "verde"), "azul": ((0, 80, 255), "azul"),
             "amarel": ((255, 200, 0), "amarela"), "laranja": ((255, 110, 0), "laranja"),
             "rox": ((150, 0, 255), "roxa"), "lilás": ((200, 120, 255), "lilás"), "rosa": ((255, 60, 150), "rosa"),
             "branc": ((255, 255, 255), "branca"), "ciano": ((0, 220, 255), "ciano"),
             "turquesa": ((0, 220, 180), "turquesa")}
    PARA_VOZ = re.compile(r"^\W*(para|pare|parar|stop|chega|espera)\b", re.IGNORECASE)
    SIM_VOZ = re.compile(r"^\W*(sim|pode|pode sim|pode começar|isso|confirmo|confirma|vai|claro|ok|beleza)\b",
                         re.IGNORECASE)
    NAO_VOZ = re.compile(r"^\W*(n[aã]o|cancela|deixa|nada)\b", re.IGNORECASE)
    LUZ = re.compile(r"\b(luz|led|cor|olhos?|rosto)\b", re.IGNORECASE)

    def led(self, rgb):
        if not self.robo_ligado():
            return
        try:
            r, g, b = rgb
            json.loads(urllib.request.urlopen(urllib.request.Request(
                f"{self.a.painel}/led?r={r}&g={g}&b={b}", data=b"", method="POST"), timeout=3).read())
        except Exception:  # noqa: BLE001
            pass

    def voz_estado(self, e):
        """O ouvido mudou de estado. Só "acordou" (ouviu 'Prometheus' sozinho) acende — barulho não pisca a luz."""
        if self.a.led_voz and e == "acordou":
            self.origem = "robo"
            self.led((0, 120, 255))
            self.fala("Sim?", "voz", registra=False)
        elif self.a.led_voz and e == "parado" and not self.rodando:
            self.led(self.led_base)   # o assistente da Unitree acende azul quando ouve voz: volta para a nossa cor

    def voz_pergunta(self, texto):
        """Ouviu "Prometheus, <texto>": pergunta de luz é resolvida aqui; o resto vai para a conversa (uma ORDEM vira
        PLANO e precisa de CONFIRMAR na tela, como digitada — a voz não libera nada)."""
        self.origem = "robo"
        t = texto.lower().strip(" .,!?")
        if self.PARA_VOZ.match(t):   # PARAR pela voz: na hora, sem confirmar
            self.diz("você", f"🎤 {texto}")
            self.parar()
            self.aguarda_sim_ate = 0.0
            self.fala("Parei.", "voz")
            return
        if time.time() < self.aguarda_sim_ate:
            if self.SIM_VOZ.match(t):
                self.aguarda_sim_ate = 0.0
                self.diz("você", f"🎤 {texto}")
                self.confirma(voz=True)
                return
            if self.NAO_VOZ.match(t):
                self.aguarda_sim_ate, self.pendente = 0.0, None
                self.diz("você", f"🎤 {texto}")
                self.fala("Tudo bem, não vou fazer.", "voz")
                return
        if self.a.led_voz:
            self.led((150, 0, 255))   # pensando
        if self.LUZ.search(t):
            if re.search(r"\b(apag|deslig)", t):
                self.led_base = (0, 0, 0)
                self.diz("você", f"🎤 {texto}")
                self.fala("Pronto, apaguei a luz.", "voz")
                return
            for chave, (rgb, nome) in self.CORES.items():
                if chave in t:
                    self.led_base = rgb
                    self.diz("você", f"🎤 {texto}")
                    self.led(rgb)
                    self.fala(f"Pronto, luz {nome}.", "voz")
                    return
        self.conversa(texto, voz=True)
        if self.a.led_voz and not self.rodando:
            self.led(self.led_base)

    def fala(self, texto, origem="ER-1", registra=True, no_robo=None):
        """O robô fala em português (pelo painel do robô). registra=False: o texto já está no chat (resposta do ER-1
        marcada com 🔊) — não repete a mesma frase duas vezes."""
        try:
            wav = self._wav(texto)
            r, na_pagina = self.destinos()
            if no_robo is None:
                no_robo = r
            else:   # 03/11: "Falar" da página = o robô fala (se ligado), sem repetir aqui
                no_robo, na_pagina = no_robo and self.robo_ligado(), not self.robo_ligado()
            if self.a.voz_navegador and na_pagina:   # 30/10: a página toca (com o robô ligado ou não), em ordem
                self.audio_wav, self.audio_n = wav, self.audio_n + 1
                self.audios[self.audio_n] = wav
                for k in [k for k in self.audios if k < self.audio_n - 30]:
                    del self.audios[k]
            if no_robo:
                pcm = self._pcm(wav)
                dur = len(pcm) / 32000.0
                if self.ouvido is not None:   # surdo enquanto fala (senão ele se ouve), + a fila do painel
                    self.ouvido.surdo_ate(time.time() + dur + 1.2)
                if self.a.led_voz:
                    self.led((0, 255, 60))
                    threading.Timer(dur + 0.6, lambda: self.led(self.led_base)).start()
                u = f"{self.a.painel}/fala?texto={urllib.parse.quote(texto)}"
                urllib.request.urlopen(urllib.request.Request(u, data=pcm, method="POST"), timeout=10).read()
            if registra:
                self.diz("robô 🔊", texto, origem=origem)
        except Exception as e:  # noqa: BLE001
            self.diz("sistema", f"o robô não conseguiu falar ({e}): {texto}")

    def destinos(self):
        """03/11: (no robô?, na página?) para esta fala. Robô desligado: sempre a página."""
        ligado = self.robo_ligado()
        s = self.saida_voz
        if s == "auto":
            s = "robo" if self.origem == "robo" else "pagina"
        return ligado and s in ("robo", "ambos"), s in ("pagina", "ambos") or not ligado

    def robo_ligado(self):
        """30/10: o corpo está ligado? (o painel do robô responde). Guarda 4 s; desligado = só o cérebro na PGX."""
        agora = time.time()
        if agora - getattr(self, "_robo_t", 0) > 4:
            try:
                urllib.request.urlopen(f"{self.a.painel}/estado.json", timeout=1.0).read()
                self._robo = True
            except Exception:  # noqa: BLE001
                self._robo = False
            self._robo_t = agora
        return self._robo

    def wla_online(self):
        """A cabine do robô (executor do WLA, sobe_cabine_wla.sh) está no ar? Guarda 3 s (a página pergunta muito)."""
        agora = time.time()
        if agora - getattr(self, "_wla_t", 0) > 3:
            try:
                urllib.request.urlopen(f"{self.a.cabine}/estado.json", timeout=1).read()
                self._wla = True
            except Exception:  # noqa: BLE001
                self._wla = False
            self._wla_t = agora
        return self._wla

    # ── travas ──
    def libera(self, sim):
        self.liberado, self.liberado_t = bool(sim), time.time()
        self.diz("sistema", "🔓 robô LIBERADO para executar planos confirmados" if sim else "🔒 robô TRAVADO")

    def libera_voz(self, sim):
        self.voz_executa, self.voz_executa_t = bool(sim), time.time()
        self.diz("sistema", "🎤🔓 a VOZ pode executar: diga 'Prometheus, <tarefa>' e confirme com 'sim'" if sim
                 else "🎤🔒 a voz NÃO executa mais (só conversa)")

    def voz_liberada(self):
        if self.voz_executa and time.time() - self.voz_executa_t > 600:
            self.libera_voz(False)
        return self.voz_executa

    def esta_liberado(self):
        if self.liberado and time.time() - self.liberado_t > 600:
            self.libera(False)
        return self.liberado

    def cabine_pronta(self):
        """No ar, sem parada pedida e sem outra tarefa rodando."""
        try:
            e = json.loads(urllib.request.urlopen(f"{self.a.cabine}/estado.json", timeout=5).read())
        except Exception as ex:  # noqa: BLE001
            return f"a cabine do robô não responde ({ex})"
        if e.get("parada_pedida"):
            return "a cabine do robô está com PARADA pedida"
        if (e.get("tarefa") or "").strip():
            return f"o robô já está fazendo outra tarefa: \"{e['tarefa']}\""
        return None

    def objetos_presentes(self):
        """28/10: o ER-1 LISTA o que vê (texto livre) e a cabine procura os objetos conhecidos na lista. A pergunta
        sim/não ("Is there a kettle on the table?") respondia SIM até olhando para o chão vazio."""
        listas = []
        for q in ("List every object you can see in the image WITH ITS COLOR, comma separated (e.g. 'yellow cup'). "
                  "Do not guess.",
                  "List the objects on the table, comma separated. If there is no table, say: no table."):
            try:
                r, _ = self.er1(q, max_tok=80, lora=0, timeout=30)
                listas.append(r.lower())
            except Exception:  # noqa: BLE001
                pass
        # sem repetição ("black wire, black wire, ..." enchia o prompt)
        listas = [", ".join(dict.fromkeys(i.strip() for i in l.split(",") if i.strip())) for l in listas]
        texto = " | ".join(listas)
        vistos, ausentes = [], []
        for nome, chaves in OBJETOS_LISTA.items():
            (vistos if any(re.search(rf"\b{c}", texto) for c in chaves) else ausentes).append(nome)
        print(f"[ER-1] vê: {texto} -> confirmei: " + (", ".join(vistos) or "nada da lista"), flush=True)
        self.visto_agora = texto
        return vistos, ausentes

    # ── conversa ──
    COMANDO = re.compile(r"^\s*(por favor\s+)?(coloc|peg|lev|faz|faç|tir|abr|fech|bot|põe|poe|guard|mov|ponh)\w*",
                         re.IGNORECASE)

    def traduz(self, texto):
        """Resposta em inglês -> português (o /traduz do servidor do ER-1)."""
        if not re.search(r"\b(the|is|are|and|to|it|on|of|with)\b", texto.lower()):
            return texto
        try:
            u = f"{self.a.er1}/traduz?" + urllib.parse.urlencode({"t": texto})
            return json.loads(urllib.request.urlopen(u, timeout=30).read()).get("pt") or texto
        except Exception:  # noqa: BLE001
            return texto

    # 29/10: o robô sabe de si — relógio sempre; o resto do estado (painel do robô) quando perguntam dele
    STATUS = re.compile(r"\bhora|horas|\bdia\b|\bdata\b|bateria|carga|energia|temperatura|quente|calor|esquent|"
                        r"status|estado|como (voc[eê]|vc|tu) (est[aá]|t[aá])|tudo bem|inclin|sa[uú]de|motor|junta|"
                        r"c[aâ]mera|trava|p[aâ]nico|cansad|sente", re.IGNORECASE)
    DIAS = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]

    def relogio(self):
        t = time.localtime()
        periodo = "manhã" if t.tm_hour < 12 else "tarde" if t.tm_hour < 18 else "noite"
        return f"Agora é {self.DIAS[t.tm_wday]}, {time.strftime('%d/%m/%Y')}, {time.strftime('%H:%M')} ({periodo})."

    def status_robo(self):
        """Resumo em português do estado do robô (painel :8095/estado.json)."""
        try:
            e = json.loads(urllib.request.urlopen(f"{self.a.painel}/estado.json", timeout=3).read())
        except Exception as ex:  # noqa: BLE001
            return f"Não consegui ler o meu estado agora ({ex})."
        partes = []
        b = e.get("bateria") or {}
        if b.get("soc") is not None:
            corr = b.get("corrente_a") or 0
            partes.append(f"bateria {b['soc']}% ({b.get('tensao_v', 0):.1f} V, {'descarregando' if corr < 0 else 'carregando'}"
                          f", saúde {b.get('soh', '?')}%, {b.get('ciclos', '?')} ciclos)")
        js = (e.get("corpo") or {}).get("juntas") or []
        grupos = {"pernas": range(0, 12), "cintura": range(12, 15), "braço esquerdo": range(15, 22),
                  "braço direito": range(22, 29)}
        temps = []
        for nome, idx in grupos.items():
            ts = [(max(js[i].get("temp") or [0]), js[i].get("nome", str(i))) for i in idx if i < len(js)]
            if ts:
                tmax, quem = max(ts)
                temps.append(f"{nome} até {tmax:.0f} °C ({quem})")
        if temps:
            partes.append("motores: " + "; ".join(temps))
        mt = [max(j.get("temp") or [0]) for lado in ("mao_left", "mao_right") for j in (e.get(lado) or {}).get("juntas", [])]
        if mt:
            partes.append(f"mãos até {max(mt):.0f} °C")
        jt = ((e.get("jetson") or {}).get("temperaturas") or {}).get("CPU")
        if jt:
            partes.append(f"computador interno {jt:.0f} °C")
        rpy = ((e.get("corpo") or {}).get("imu") or {}).get("rpy_graus")
        if rpy:
            partes.append(f"inclinação do corpo: {rpy[0]:+.1f}° de lado, {rpy[1]:+.1f}° para frente/trás")
        pan = e.get("panico") or {}
        if pan.get("panico"):
            partes.append("braços TRAVADOS pela segurança (" + (pan.get("motivo") or "pânico") + ")")
        else:
            partes.append("trava de segurança liberada")
        if e.get("modo"):
            partes.append(f"modo de locomoção: {e['modo']}")
        cams = ((e.get("cameras") or {}).get("p5555") or {})
        ok = [n for n, c in cams.items() if isinstance(c, dict) and (c.get("fps") or 0) > 5]
        if cams:
            partes.append(f"câmeras funcionando: {len(ok)} de {len(cams)}")
        partes.append("modelo de manipulação (WLA) " + ("ligado" if self.wla_online() else "desligado"))
        return "Seu estado agora: " + "; ".join(partes) + "."

    VISAO = re.compile(r"vend|\bv[eê]\b|v[eê]s\b|olh|mesa|objeto|\bcor\b|cores|\btem\b|m[aã]o|frente|onde|qual|"
                       r"caneca|copo|ma[çc][aã]|coador|chaleira|pote|isso|aquilo|segur|enxerg|cena|c[aâ]mera", re.IGNORECASE)
    PERGUNTA_TAREFA = re.compile(r"(sabe|consegue|pode|capaz).{0,25}(fazer|pegar|ajudar)|o que (você|vc) faz|"
                                 r"tarefa|habilidade|ajud", re.IGNORECASE)

    def sem_repeticao(self, r):
        """Tira as frases da resposta que já apareceram (quase iguais) nas respostas anteriores da conversa."""
        def norma(f):
            return re.sub(r"[^a-zà-ú ]", "", f.lower()).strip()
        ditas = {norma(f) for _, ant, *_ in self.historico for f in re.split(r"(?<=[.!?])\s+", ant) if norma(f)}
        frases = re.split(r"(?<=[.!?])\s+", r)
        novas = [f for f in frases if norma(f) not in ditas]
        return " ".join(novas).strip() or frases[0]

    # 30/10: ENSINAR e CORRIGIR conversando (vai para a memória e para o dataset do LoRA do ER-1)
    APRENDE = re.compile(r"^\W*(aprend[ae]|lembr[ae](-se)?|guard[ae]|anot[ae]|memoriz[ae]|grav[ae])\b(\s+(de|que|isso|isto|"
                         r"esta|essa|aqui))?\s*[:,-]?\s*(?P<fato>.{6,})", re.IGNORECASE)
    CORRIGE = re.compile(r"^\W*(errad[oa]|n[aã]o\W+[eé]\s+isso|t[aá] errad[oa]|est[aá] errad[oa])?\W*(o\s+)?(certo|"
                         r"correto)\s+(seria|[eé])\s*[:,-]?\s*(?P<certo>.{3,})", re.IGNORECASE)
    SEGUE = re.compile(r"\b(me conta|conta mais|mais sobre|melhor|explica|continua|detalh|e (ele|ela|isso|eles)\b|"
                       r"por que|porque|como assim)", re.IGNORECASE)
    SEGURA = re.compile(r"segur|pegou|pegando|na (sua )?m[aã]o|\bm[aã]os?\b|carreg", re.IGNORECASE)
    CENA = re.compile(r"vend|\bv[eê]\b|v[eê]s\b|olh|enxerg|c[aâ]mera|na sua frente|na mesa|cena|cor d", re.IGNORECASE)

    def aprende(self, fato, via="texto"):
        i = self.mem.adiciona(fato[:60], fato, "fato", f"conversa:{via}")
        self.diz("sistema", f"📚 guardei na memória (#{i}): {fato}", fino=True)
        return "Anotado! Vou lembrar disso."

    def corrige(self, certo, id_conversa=None):
        if id_conversa is None and self.historico:
            id_conversa = self.historico[-1][2] if len(self.historico[-1]) > 2 else None
        pergunta = self.mem.pergunta_de(id_conversa) if id_conversa else None
        if not pergunta:
            return self.aprende(certo)
        i = self.mem.corrige(id_conversa, pergunta, certo)
        self.diz("sistema", f"📚 correção guardada (#{i}) — '{pergunta}' → {certo}", fino=True)
        return "Entendi, obrigado pela correção! Da próxima vez eu respondo certo."

    def conversa(self, texto, voz=False):
        self.origem = "robo" if voz else "pagina"
        if voz:
            self.diz("sistema", "🎤 ouvi pela voz do robô", fino=True)
        ligado = self.robo_ligado()
        m_ap, m_co = self.APRENDE.match(texto), self.CORRIGE.match(texto)
        if m_ap or m_co:
            self.diz("você", texto)
            r = self.aprende(m_ap.group("fato").strip(), "voz" if voz else "texto") if m_ap else \
                self.corrige(m_co.group("certo").strip())
            self.diz("ER-1", r)
            if self.a.fala_respostas:
                self.fala(r, "ER-1", registra=False)
            return
        if self.COMANDO.match(texto) and not ligado:
            self.diz("você", texto)
            r = "Agora o meu corpo está desligado, então não consigo fazer isso. Quando me ligarem, eu faço!"
            self.diz("ER-1", r)
            if self.a.fala_respostas:
                self.fala(r, "ER-1", registra=False)
            return
        if self.COMANDO.match(texto):   # ordem ("coloca o copo no coador"): vira PLANO (nada vai ao robô sem CONFIRMAR)
            print("[sistema] ordem no chat: vira plano (nada vai ao robô sem CONFIRMAR)", flush=True)
            return self.objetivo(texto, voz=voz)
        self.diz("você", texto)
        try:
            # 28/10: prompt CURTO com EXEMPLOS (o ER-1 é pequeno: com listas e regras longas ele respondia "Bom dia!"
            # a "Boa tarde" e não descrevia a cena). A cena só entra quando a pergunta é sobre ela.
            cena, cam = "", ""   # 30/10: imagem só quando a pergunta é sobre a cena (mais rápido)
            if not ligado:
                cena = ("O seu corpo de robô está DESLIGADO agora: só o seu cérebro, no computador, está ligado. Você "
                        "não enxerga nada (câmeras desligadas), não mexe os braços e está conversando pelo computador. "
                        "Se perguntarem o que você vê ou faz, diga isso com naturalidade.\n")
            elif self.CENA.search(texto) or (self.VISAO.search(texto) and not self.mem.busca(texto)):
                cam = "head_stereo_left"
                vistos, _ = self.objetos_presentes()
                cena = (f"Agora a câmera da sua cabeça mostra (lista em inglês, com as cores): "
                        f"{getattr(self, 'visto_agora', '') or 'nada reconhecível'}. Fale só desses objetos e dessas "
                        "cores; não invente outros.\n")
            if self.SEGURA.search(texto) and (not ligado or not self.wla_online()):
                cena += ("Agora você NÃO está segurando nada: os seus braços estão parados. Se uma pessoa segura algo "
                         "na sua frente, quem segura é a pessoa (mão de pele), não você.\n")
            # 30/10: 2 trechos de até 700 caracteres (com 3 x 900 o 1º token passava de 0,2 para ~1 s)
            # 03/11: pergunta de CONTINUAÇÃO ("me conta melhor", "e ele?") busca junto com a pergunta anterior
            from memoria_er1 import palavras
            consulta = texto
            if self.historico and (len(palavras(texto)) <= 2 or self.SEGUE.search(texto)):
                resto = re.sub(r"\b(pfvr|por favor|pf|mais|melhor|sobre|isso|aí|ai)\b", " ", self.SEGUE.sub(" ", texto),
                               flags=re.IGNORECASE)
                consulta = f"{self.historico[-1][0]} {resto}"   # o pedido de "mais" não entra na busca
            trechos = self.mem.busca(consulta, k=2)
            saber = ("O que você sabe sobre isso (use se ajudar; não leia tudo):\n" + "".join(
                f"- {x['titulo']}: {x['texto'][:700]}\n" for x in trechos)) if trechos else ""
            longa = any(x["tipo"] == "receita" for x in trechos)
            tarefas = (f"As únicas tarefas que você sabe fazer com as mãos: {'; '.join(PT[h] for h in HABILIDADES)}.\n"
                       if self.PERGUNTA_TAREFA.search(texto) else "")
            conversa = "".join(f"Usuário: {u}\nPrometheus: {r_}\n" for u, r_, *_ in self.historico[-self.a.memoria:])
            estado = (self.status_robo() + " Fale do seu estado só no que foi perguntado, em palavras simples "
                      "(sem ler a lista inteira).\n") if self.STATUS.search(texto) and ligado else ""
            q = ("Você é o Prometheus, um robô humanoide criado pela equipe do LCAD (Laboratório de Computação de Alto "
                 "Desempenho) da UFES. Você fala português do Brasil, sempre em primeira pessoa, como um assistente de voz "
                 "simpático e natural: frases curtas, tom de conversa, sem listas nem símbolos, e responde exatamente "
                 "ao que perguntaram. Se a resposta não estiver no que você sabe, diga que não sabe; nunca invente datas, números "
                 "ou nomes.\n" + self.relogio() + "\n" + estado + cena +
                 tarefas + saber +
                 "Exemplos de conversa:\n"
                 "Usuário: Boa tarde!\nPrometheus: Boa tarde! Tudo bem com você?\n"
                 "Usuário: Quem é você?\nPrometheus: Eu sou o Prometheus, um robô humanoide criado pela equipe do "
                 "LCAD, da UFES.\n"
                 "Usuário: O que você está vendo?\nPrometheus: Estou vendo um monitor preto e uma cadeira na minha "
                 "frente.\n"
                 "Usuário: Obrigado!\nPrometheus: De nada! Precisando, é só chamar.\n\n"
                 + conversa + f"Usuário: {texto}\nPrometheus:")
            max_tok = max(self.a.max_tokens, 400) if longa else self.a.max_tokens
            if self.a.fala_respostas and self.a.stream:
                def norma(f):
                    return re.sub(r"[^a-zà-ú ]", "", f.lower()).strip()
                ditas = {norma(f) for _, ant, *_ in self.historico for f in re.split(r"(?<=[.!?])\s+", ant) if norma(f)}

                def ao_frase(f):
                    f = re.sub(r"^\s*(rob[oô]|robot|ER-1|prometheus)\s*:\s*", "", f.strip(), flags=re.IGNORECASE)
                    if f and norma(f) not in ditas:
                        self.fila_fala.put(f)
                r, ms = self.er1_stream(q, max_tok, cam, ao_frase)
            else:
                r, ms = self.er1(q, max_tok=max_tok, cam=cam)
                r = self.traduz(r)
            r = re.sub(r"^\s*(rob[oô]|robot|ER-1|prometheus)\s*:\s*", "", r, flags=re.IGNORECASE).strip()   # tira o rótulo
            r = re.split(r"\n\s*(Usuário|Prometheus)\s*:", r)[0].strip()   # ele às vezes continua a conversa sozinho
            r = self.sem_repeticao(r)
            idc = self.mem.registra(texto, r, "voz" if voz else "texto", ligado, trechos)
            self.historico.append((texto, r, idc))
            self.historico = self.historico[-20:]
            fala = bool(self.a.fala_respostas)
            self.diz("ER-1" + (" 🔊" if fala else ""), r, ms=ms, id=idc,
                     fontes=[x["titulo"] for x in trechos])
            if fala and not self.a.stream:   # com stream, a voz já foi frase a frase
                self.fala(r, "ER-1", registra=False)
        except Exception as e:  # noqa: BLE001
            self.diz("sistema", f"ER-1 indisponível: {e}")

    # ── objetivo: planejar + executar ──
    # 29/10: a frase pede UMA coisa ("pegar a caneca", "coloca no coador") -> só essa tarefa, sem o ER-1 (ele montava
    # o objetivo inteiro: pegar E colocar no coador). Pede duas ("pega a caneca e coloca no coador") -> o ER-1 planeja.
    TAREFA_VOZ = [
        (re.compile(r"coador|filtro|coar", re.IGNORECASE), "Place the white mug under the coffee strainer."),
        (re.compile(r"ma[çc][aã]|fruta", re.IGNORECASE), "Pick up the apple."),
        (re.compile(r"\bpeg\w*.*\b(caneca|copo|colo|x[ií]cara)|\b(caneca|copo|colo|x[ií]cara)\b(?!.*coador)",
                    re.IGNORECASE), "Pick up the white mug."),
    ]

    def tarefa_unica(self, texto):
        """A tarefa, se a frase pede exatamente UMA; senão None (o ER-1 planeja)."""
        achou = [h for padrao, h in self.TAREFA_VOZ if padrao.search(texto)]
        pega_e_coloca = re.search(r"\bpeg\w*", texto, re.IGNORECASE) and re.search(r"coador|filtro", texto,
                                                                                   re.IGNORECASE)
        if pega_e_coloca or len(achou) != 1:
            return None
        return achou[0]

    def objetivo(self, texto, voz=False):
        if self.rodando:
            self.diz("sistema", "já estou executando um objetivo — aperte PARAR primeiro")
            if voz:
                self.fala("Já estou fazendo uma tarefa. Diga para, se quiser que eu pare.", "plano")
            return
        if not self.robo_ligado():   # 30/10: sem corpo não há o que planejar (nem câmera para olhar a mesa)
            self.diz("você", f"🎯 objetivo: {texto}")
            self.diz("sistema", "o robô está DESLIGADO — planejar e executar precisam do corpo ligado (o init)")
            return
        self.diz("você", f"🎯 objetivo: {texto}")
        unica = self.tarefa_unica(texto)
        if unica:
            return self._pede_confirmacao([unica], voz)
        lista = "\n".join(f"{i}. {h}" for i, h in enumerate(HABILIDADES, 1))
        if voz:   # uma ordem falada = UMA tarefa
            lista += "\n\nThe user gave ONE spoken order: choose EXACTLY ONE skill."
        q = (f"You control a humanoid robot that can ONLY perform these skills, one at a time:\n{lista}\n\n"
             f'Goal (from the user, in Portuguese): "{texto}"\n'
             "Look at the image. Which skills, in which order, achieve the goal? Answer first with a JSON list of "
             "skill numbers, like [1, 2], and then one short sentence in Brazilian Portuguese explaining the plan.")
        try:
            r, ms = self.er1(q, max_tok=120)
        except Exception as e:  # noqa: BLE001
            self.diz("sistema", f"ER-1 indisponível: {e}")
            return
        m = re.search(r"\[([\d,\s]+)\]", r)
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))] if m else []
        plano = [HABILIDADES[n - 1] for n in nums if 1 <= n <= len(HABILIDADES)]
        explica = (r[m.end():] if m else r).strip(" .\n`")
        self.diz("ER-1", f"plano: {' → '.join(PT[p] for p in plano) or '(nenhum)'}" + (f" — {explica}" if explica else ""),
                 ms=ms)
        if not plano:
            self.diz("sistema", "o ER-1 não montou um plano com as tarefas conhecidas")
            return
        if len(plano) > 4:
            self.diz("sistema", f"plano com {len(plano)} etapas: o limite é 4 — não vou executar")
            return
        if voz:
            plano = plano[:1]
        return self._pede_confirmacao(plano, voz)

    def _pede_confirmacao(self, plano, voz=False):
        self.pendente, self.pendente_t = plano, time.time()
        self.diz("sistema", f"tarefa: {' → '.join(PT[p] for p in plano)}" + (" (uma ordem falada = uma tarefa)" if voz else ""))
        self.diz("sistema", "⚠️ nada foi mandado ao robô ainda: " + ("diga 'sim' (ou aperte CONFIRMAR) em até "
                            f"{self.a.sim_s:.0f} s" if voz and (self.voz_liberada() or self.esta_liberado()) else
                            "com o robô LIBERADO, aperte CONFIRMAR em até 60 s para executar."), pendente=True)
        # o robô fala o plano ANTES de fazer qualquer coisa e pede a confirmação
        etapas = ", depois ".join(PT[p] for p in plano)
        if voz and (self.voz_liberada() or self.esta_liberado()):
            # 29/10: confirmação FALADA: o ouvido aceita a resposta sem "Prometheus" por --sim-s
            self.fala(f"Meu plano é {etapas}. Posso começar? Diga sim.", "plano")
            self.aguarda_sim_ate = time.time() + self.a.sim_s
            if self.ouvido is not None:
                self.ouvido._acordado_ate = self.aguarda_sim_ate
        else:
            self.fala(f"Meu plano é {etapas}. Posso começar? Confirme na tela.", "plano")

    def confirma(self, voz=False):
        liberado = self.esta_liberado() or (voz and self.voz_liberada())
        if self.retry is not None:   # confirmação da 2ª tentativa
            if not self.esta_liberado():
                self.diz("sistema", "🔒 o robô está TRAVADO: ligue a chave 'robô liberado' para tentar de novo")
                return
            self.retry_ok.set()
            return
        plano = self.pendente
        if not plano or time.time() - self.pendente_t > 60:
            self.pendente = None
            self.diz("sistema", "não há plano para confirmar (ou passou de 60 s) — peça o objetivo de novo")
            return
        if not liberado:
            self.diz("sistema", "🔒 o robô está TRAVADO: ligue a chave 'robô liberado' (ou '🎤 voz pode executar') "
                                "antes de confirmar")
            if voz:
                self.fala("Não posso começar: a execução por voz está travada na cabine.", "plano")
            return
        if self.rodando:
            self.diz("sistema", "já estou executando — aperte PARAR primeiro")
            return
        motivo = self.cabine_pronta()
        if motivo:
            self.diz("sistema", f"não vou executar: {motivo}")
            return
        try:
            ativo = json.loads(urllib.request.urlopen(f"{self.a.cabine}/estado.json", timeout=5).read()).get("terminou_ativo")
        except Exception:  # noqa: BLE001
            ativo = False
        if not ativo:
            self.diz("sistema", "não vou executar: a cabine do robô está SEM o 'terminou?' do ER-1. Suba ela com "
                                "TERMINOU=1 (quem checa é o executor, sem atrasar o WLA)")
            if voz:
                self.fala("Não posso começar: o executor está sem a verificação de término.", "plano")
            return
        self.pendente = None
        self.plano, self.passo, self.rodando = plano, -1, True
        self.cancela.clear()
        self.diz("você", "✔ CONFIRMAR plano" + (" (pela voz)" if voz else ""))
        self.fala("Confirmado. Começando.", "plano")
        threading.Thread(target=self._executa, daemon=True).start()

    def _espera(self, s):
        return self.cancela.wait(s)   # True = cancelado

    def _executa(self):
        try:
            for i, frase in enumerate(self.plano):
                self.passo = i
                for tentativa in (1, 2):
                    if self.cancela.is_set():
                        return
                    if self.cabine_reiniciou():
                        self.diz("orquestrador", "🛑 a cabine do robô foi REINICIADA (ou não responde): objetivo cancelado, "
                                                 "não mando mais nada")
                        return
                    if not self.liberado:
                        self.diz("orquestrador", "🛑 o robô foi TRAVADO: objetivo cancelado")
                        return
                    if i > 0 or tentativa > 1:   # na 1ª a checagem já foi feita no CONFIRMAR
                        motivo = self.cabine_pronta() if tentativa == 1 else None
                        if motivo and "outra tarefa" not in motivo:
                            self.diz("orquestrador", f"🛑 parei antes da etapa {i + 1}: {motivo}")
                            return
                    self.diz("orquestrador", f"▶ etapa {i + 1}/{len(self.plano)}: {PT[frase]} "
                                             f"(tentativa {tentativa}) — mandando ao WLA")
                    self.cabine("/tarefa", {"texto": frase})
                    t0, sims, falhas = time.time(), 0, 0
                    while time.time() - t0 < self.a.tempo_max:
                        if self._espera(2.0):
                            return
                        # 27/10: quem pergunta "terminou?" é o EXECUTOR (TERMINOU=1), revezando a GPU com o WLA; aqui só
                        # se lê o resultado na cabine do robô. Perguntar daqui atrasava o WLA (consultas de até 1,6 s) e
                        # a pega falhava.
                        try:
                            e = json.loads(urllib.request.urlopen(f"{self.a.cabine}/estado.json", timeout=5).read())
                        except Exception as ex:  # noqa: BLE001
                            falhas += 1
                            if falhas >= 3:   # 27/10: antes ficava repetindo até o tempo máximo
                                self.diz("orquestrador", f"🛑 a cabine do robô caiu ({ex}): objetivo cancelado, "
                                                         "não mando mais nada")
                                self.fala("A cabine do robô caiu. Parei.", "plano")
                                return
                            continue
                        falhas = 0
                        if self.seq_enviado is not None and int(e.get("tarefa_seq", 0)) < self.seq_enviado:
                            self.diz("orquestrador", "🛑 a cabine do robô foi REINICIADA: objetivo cancelado, não mando "
                                                     "mais nada")
                            return
                        if (e.get("tarefa") or "").strip() != frase or e.get("parada_pedida"):
                            self.diz("orquestrador", "🛑 a tarefa do robô foi mudada ou parada fora daqui: objetivo "
                                                     "cancelado, não mando mais nada")
                            return
                        c = e.get("concluida") or {}
                        if c.get("frase") == frase and c.get("t", 0) > t0:
                            sims = 2
                            self.diz("ER-1", f"terminou \"{PT[frase]}\": sim (2 confirmações no executor)", fino=True)
                            break
                    if sims >= 2:
                        self.diz("orquestrador", f"✅ etapa {i + 1} concluída: {PT[frase]}")
                        self.fala(f"Etapa {i + 1} concluída: {PT[frase]}.")
                        break
                    if tentativa == 1:
                        # 27/10: a 2ª tentativa só com CONFIRMAR de novo (antes reenviava sozinha)
                        self.cabine("/tarefa", {"texto": ""})
                        self.cabine("/inicial")
                        self.retry, self.retry_ok = (frase, time.time()), threading.Event()
                        self.diz("orquestrador", f"⏱️ {self.a.tempo_max:.0f} s sem concluir \"{PT[frase]}\" — voltei à "
                                                 "pose inicial. Tentar de novo? Aperte CONFIRMAR em até 60 s "
                                                 "(sem confirmar, eu paro).", pendente=True)
                        self.fala(f"Não consegui {PT[frase]}. Quer que eu tente de novo? Confirme na tela.", "plano")
                        t1 = time.time()
                        while time.time() - t1 < 60 and not self.retry_ok.is_set():
                            if self._espera(0.5):
                                self.retry = None
                                return
                        ok = self.retry_ok.is_set()
                        self.retry = None
                        if not ok:
                            self.diz("orquestrador", "⏹ sem confirmação para tentar de novo — parei")
                            self.fala("Sem confirmação. Parei.", "plano")
                            return
                        self.diz("você", "✔ CONFIRMAR nova tentativa")
                else:
                    self.diz("orquestrador", f"❌ não consegui \"{PT[frase]}\" em duas tentativas — parei")
                    self.cabine("/tarefa", {"texto": ""})
                    self.fala("Não consegui terminar a tarefa. Parei.")
                    return
            self.diz("orquestrador", "🏁 objetivo concluído!")
            self.cabine("/tarefa", {"texto": ""})
            self.fala("Objetivo concluído.")
        except Exception as e:  # noqa: BLE001
            self.diz("sistema", f"erro na execução: {e}")
        finally:
            self.rodando, self.passo = False, -1
            if self.liberado:
                self.libera(False)   # trava sozinho no fim de cada objetivo

    def reset(self):
        """Novo contexto: CANCELA o objetivo em andamento (27/10: ele continuava e reenviou a tarefa sozinho),
        esquece a conversa e o plano pendente, limpa o chat e trava o robô."""
        estava = self.rodando
        self.cancela.set()
        if estava:
            try:
                self.cabine("/tarefa", {"texto": ""})
            except Exception:  # noqa: BLE001
                pass
        if self.liberado:
            self.liberado = False
        with self.trava:
            self.msgs = []
        self.historico, self.pendente = [], None
        self.diz("sistema", "🧹 contexto apagado" + (" e objetivo CANCELADO" if estava else "") + " — robô travado")

    def parar(self):
        self.cancela.set()
        self.pendente = None
        if self.liberado:
            self.libera(False)
        try:
            self.cabine("/tarefa", {"texto": ""})
            self.cabine("/parar")
        except Exception as e:  # noqa: BLE001
            self.diz("sistema", f"cabine do robô não respondeu ao parar: {e}")
        self.diz("você", "⏹ PARAR")


PAGINA = """<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Cabine do Prometheus</title>
<style>
:root{--fundo:#0e1014;--cartao:#171a21;--cartao2:#1e222b;--borda:#2a2f3a;--texto:#e7e9ee;--fraco:#8a90a0;
--acento:#4f9dff;--ok:#34c38f;--aviso:#e0a020;--erro:#ff5d5d;--voce:#22406e;--robo:#262a33;--sis:#2a2224}
*{box-sizing:border-box}html,body{height:100%}html{color-scheme:dark}
*{scrollbar-color:#3a3f4b transparent;scrollbar-width:thin}
body{margin:0;background:var(--fundo);color:var(--texto);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:10px 16px;border-bottom:1px solid var(--borda)}
header h1{font-size:17px;margin:0 8px 0 0;font-weight:650}
.chip{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;background:var(--cartao2);
  border:1px solid var(--borda);font-size:12.5px;color:var(--fraco);white-space:nowrap}
.chip i{width:8px;height:8px;border-radius:50%;background:var(--fraco);display:inline-block}
.chip.on i{background:var(--ok)}.chip.off i{background:var(--erro)}.chip.on{color:var(--texto)}
main{display:grid;grid-template-columns:minmax(300px,0.9fr) minmax(0,1.3fr);gap:14px;padding:14px 16px;
  height:calc(100vh - 54px);min-height:0}
@media(max-width:980px){main{grid-template-columns:1fr;height:auto}#chat{max-height:60vh}.col,.col:last-child{overflow:visible}}
.col{display:flex;flex-direction:column;gap:14px;min-height:0;overflow:auto}
.cartao{background:var(--cartao);border:1px solid var(--borda);border-radius:12px;padding:12px;min-width:0}
.cartao h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--fraco);margin:0 0 10px;font-weight:600}
.camwrap{position:relative;aspect-ratio:4/3;background:#000;border-radius:9px;overflow:hidden}
.camwrap img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
#camoff{position:absolute;inset:0;display:none;flex-direction:column;align-items:center;justify-content:center;gap:6px;
  text-align:center;padding:16px;color:var(--fraco);background:linear-gradient(160deg,#151922,#0b0d11)}
#camoff b{color:var(--texto);font-size:16px}
.habs{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}.habs span{font-size:12.5px;padding:3px 9px;border-radius:7px;
  background:var(--cartao2);border:1px solid var(--borda);color:var(--fraco)}
input,select,textarea{font:inherit;color:var(--texto);background:#0c0e12;border:1px solid var(--borda);border-radius:9px;
  padding:9px 11px;width:100%}
input:focus,select:focus,textarea:focus{outline:2px solid #4f9dff55;border-color:var(--acento)}
textarea{resize:vertical;min-height:80px}
button{font:inherit;padding:9px 14px;border:1px solid transparent;border-radius:9px;background:var(--cartao2);
  color:var(--texto);cursor:pointer;white-space:nowrap}
button:hover{filter:brightness(1.15)}button:disabled{opacity:.35;cursor:not-allowed;filter:none}
.prim{background:var(--acento)}.verde{background:var(--ok);color:#06210f}.conf{background:var(--aviso);color:#241700}
.parar{background:var(--erro);font-weight:700;min-width:110px}.fant{background:transparent;border-color:var(--borda)}
.chat{display:flex;flex-direction:column;flex:1 1 auto}.col>.cartao{flex:none}.col>.chat{flex:1 1 auto;min-height:380px}
#chat{flex:1 1 0;overflow:auto;display:flex;flex-direction:column;gap:8px;padding:4px 4px 8px;min-height:180px}
.m{max-width:85%;padding:8px 11px;border-radius:12px;background:var(--robo);align-self:flex-start;word-wrap:break-word}
.m.voce{align-self:flex-end;background:var(--voce)}.m.sistema{background:var(--sis);align-self:center;max-width:95%;
  font-size:13px;color:#d9c4c4}.m.orquestrador{background:#1d3226}.m.robô{background:#2b2550}.m.fino{opacity:.7}
.meta{font-size:11px;color:var(--fraco);margin-bottom:2px}.txt{white-space:pre-wrap}
.pe{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-top:6px}
.fontes{font-size:11.5px;color:var(--fraco)}.corr{padding:2px 8px;font-size:12px;background:transparent;
  border-color:var(--borda);color:var(--fraco);margin-left:auto}
.compor{display:flex;flex-direction:column;gap:8px;border-top:1px solid var(--borda);padding-top:10px;margin-top:4px}
.botoes{display:flex;gap:8px;flex-wrap:wrap}.botoes .prim{flex:1}
.linha{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.grupo{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
@media(max-width:600px){.grupo{grid-template-columns:1fr}}
.sub{background:var(--cartao2);border:1px solid var(--borda);border-radius:10px;padding:10px;display:flex;
  flex-direction:column;gap:9px}.sub h3{margin:0;font-size:12px;color:var(--fraco);font-weight:600;text-transform:uppercase;
  letter-spacing:.05em}
.tg{display:flex;align-items:center;gap:8px;cursor:pointer;user-select:none}
.tg input{appearance:none;width:34px;height:20px;border-radius:999px;background:#3a3f4b;position:relative;padding:0;
  border:0;flex:none;cursor:pointer;transition:.15s}
.tg input::after{content:"";position:absolute;top:3px;left:3px;width:14px;height:14px;border-radius:50%;background:#fff;
  transition:.15s}.tg input:checked{background:var(--ok)}.tg input:checked::after{left:17px}
.tg input:disabled{opacity:.4}
input[type=color]{width:40px;height:28px;padding:2px;border-radius:7px}
.t{font-size:12px;color:var(--fraco)}
details summary{cursor:pointer;list-style:none;display:flex;align-items:center;justify-content:space-between;gap:8px}
details summary::-webkit-details-marker{display:none}details[open] summary{margin-bottom:10px}
#mem .campos{display:flex;flex-direction:column;gap:8px}.dois{display:grid;grid-template-columns:1fr 130px;gap:8px}
.arq{display:inline-flex;align-items:center;gap:6px;padding:9px 14px;border-radius:9px;border:1px dashed var(--borda);
  color:var(--fraco);cursor:pointer}.arq input{display:none}
.it{display:flex;justify-content:space-between;align-items:center;gap:8px;font-size:13px;padding:5px 0;
  border-bottom:1px solid var(--borda)}.it button{padding:2px 8px;font-size:12px}
a{color:var(--acento)}
</style></head><body>
<header><h1>🤖 Prometheus</h1>
<span class="chip" id="c_robo"><i></i><span>robô</span></span>
<span class="chip" id="c_wla"><i></i><span>WLA</span></span>
<span class="chip" id="c_ouv"><i></i><span>ouvido</span></span>
<span class="chip" id="estado">parado</span>
<span style="flex:1"></span>
<label class="tg" title="a voz dele toca aqui no navegador (clique uma vez na página para o navegador deixar tocar)">
<input type="checkbox" id="vozweb" checked><span>🔊 voz aqui</span></label>
<select id="saida" title="onde ele fala: auto = responde por onde foi perguntado (digitou aqui -> aqui; falou com o robô -> no robô)" style="width:auto;padding:4px 8px;font-size:13px">
<option value="auto">fala: auto</option><option value="pagina">fala: só aqui</option><option value="robo">fala: só no robô</option><option value="ambos">fala: nos dois</option></select>
<select id="vozsel" title="voz do Prometheus" style="width:auto;padding:4px 8px;font-size:13px">
<option value="pm_alex">voz Alex</option><option value="faber">voz Faber (fluida)</option><option value="pm_santa">voz Santa</option><option value="pf_dora">voz Dora</option></select></header>
<main>
<div class="col">
  <section class="cartao"><h2>Câmera da cabeça</h2>
    <div class="camwrap"><img id="cam" alt=""><div id="camoff"><b>🔌 Corpo do robô desligado</b>
      <span>O cérebro (ER-1) continua ligado: dá para conversar, ensinar e corrigir.<br>A voz sai aqui no navegador.</span></div></div>
    <div class="habs" id="hab"></div></section>
  <section class="cartao"><details id="m3d" open><summary><h2 style="margin:0;white-space:nowrap">🧊 Mundo 3D · mapa do lidar</h2>
    <span class="t" style="text-align:right">arraste p/ girar · roda p/ zoom · <a href="#" id="m3dlimpa" style="color:inherit">limpar mapa</a></span></summary>
    <div id="mundo3d" style="margin-top:8px;border-radius:8px;overflow:hidden"></div>
    <div id="m3dinfo" class="t" style="margin-top:6px"></div></details></section>
  <section class="cartao"><details id="mem" open><summary><h2 style="margin:0;white-space:nowrap">📚 Memória</h2>
    <span id="memresumo" class="t" style="text-align:right"></span></summary>
    <div class="campos">
      <div class="t">Ele consulta isto a cada pergunta. No chat: <b>"aprenda: …"</b> guarda um fato; <b>"o certo é …"</b>
        corrige a última resposta (ou ✏️ em qualquer resposta).</div>
      <div class="dois"><input type="text" id="memtit" placeholder="título (ex.: Café da vó)">
        <select id="memtipo"><option value="fato">fato</option><option value="receita">receita</option>
        <option value="documento">documento</option></select></div>
      <textarea id="memtxt" placeholder="o que ele deve saber"></textarea>
      <div class="linha"><button type="button" class="prim" id="memok">📥 Ensinar</button>
        <label class="arq" title=".md (cada '## título' vira um item), .txt ou .json [{titulo, texto}]">📄 enviar arquivos
          <input type="file" id="memarq" multiple accept=".md,.txt,.json"></label>
        <a href="/exporta.jsonl" class="t" style="margin-left:auto" title="as conversas (com as correções) para treinar o LoRA do ER-1">⬇ conversas .jsonl</a></div>
      <div id="memult"></div></div></details></section>
</div>
<div class="col">
  <section class="cartao chat">
    <h2>Conversa</h2>
    <div id="chat"></div>
    <form class="compor" id="f">
      <input id="txt" placeholder="Pergunte algo… ou escreva um objetivo (ex.: pegue a caneca e coloque no coador)" autocomplete="off">
      <div class="botoes">
        <button type="submit" class="prim" title="só conversa (Enter)">Perguntar</button>
        <button type="button" class="verde" id="obj" title="o ER-1 monta o PLANO (nada vai ao robô)">🎯 Planejar</button>
        <button type="button" class="fant" id="falar" title="o robô fala este texto">🔊 Falar</button>
        <button type="button" class="fant" id="reset" title="apaga a memória da conversa e limpa o chat">🧹 Novo contexto</button>
      </div></form>
  </section>
  <section class="cartao"><h2>Controle do robô</h2>
    <div class="linha">
      <label class="tg"><input type="checkbox" id="lib"><span>robô liberado</span></label>
      <button type="button" class="conf" id="conf" disabled>✔ Confirmar plano</button>
      <span style="flex:1"></span>
      <button type="button" class="parar" id="parar">⏹ PARAR</button></div>
    <div class="grupo">
      <div class="sub"><h3>Voz do robô</h3>
        <label class="tg" title="fale 'Prometheus, ...' perto do robô"><input type="checkbox" id="ouv"><span>🎤 ouvir "Prometheus"</span></label>
        <label class="tg" title="uma ordem falada vira plano e o robô executa depois do seu 'sim' falado (trava em 10 min)">
          <input type="checkbox" id="vozexe"><span>🎤 voz pode executar</span></label></div>
      <div class="sub"><h3>Rosto</h3>
        <label class="linha">luz do rosto <input type="color" id="cor" value="#000000"></label>
        <span id="ouviu" class="t"></span></div>
    </div></section>
</div></main>
<script>
// 28/10: a câmera vem do servidor do ER-1 (funciona logo depois do init, sem o executor do WLA)
let roboLigado = false;
setInterval(() => { if (!roboLigado) return; const i = new Image();
  i.onload = () => { document.getElementById("cam").src = i.src; }; i.src = "/cam.jpg?t=" + Date.now(); }, 250);
let n = 0;
async function post(c, b){ await fetch(c, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(b||{})}); }
document.getElementById("f").onsubmit = e => { e.preventDefault(); const t = txt.value.trim(); if(t){ post("/conversa",{texto:t}); txt.value=""; } };
document.getElementById("obj").onclick = () => { const t = txt.value.trim(); if(t){ post("/objetivo",{texto:t}); txt.value=""; } };
document.getElementById("parar").onclick = () => post("/parar");
document.getElementById("reset").onclick = () => { post("/reset"); document.getElementById("chat").innerHTML = ""; n = 0; };
document.getElementById("falar").onclick = () => { const t = txt.value.trim(); if(t){ post("/falar",{texto:t}); txt.value=""; } };
document.getElementById("lib").onchange = e => post("/libera", {sim: e.target.checked});
document.getElementById("conf").onclick = () => { if(confirm("Executar no robô?")) post("/confirma"); };
document.getElementById("ouv").onchange = e => post("/ouvido", {sim: e.target.checked});
document.getElementById("vozexe").onchange = e => post("/libera_voz", {sim: e.target.checked});
document.getElementById("cor").onchange = e => post("/led", {cor: e.target.value});
// 30/10: voz no navegador e memória
let audioN = null; const vozweb = document.getElementById("vozweb");
const filaAudio = []; let tocando = false;
function tocaFila(){
  if (tocando || !filaAudio.length) return;
  tocando = true; const a = new Audio("/voz.wav?n=" + filaAudio.shift());
  const prox = () => { tocando = false; tocaFila(); };
  a.onended = prox; a.onerror = prox; a.play().catch(prox);
}
try { vozweb.checked = localStorage.getItem("vozweb") !== "0"; } catch(e){}
document.getElementById("vozsel").onchange = e => post("/voz", {voz: e.target.value});
document.getElementById("saida").onchange = e => post("/saida_voz", {saida: e.target.value});
vozweb.onchange = () => { try { localStorage.setItem("vozweb", vozweb.checked ? "1" : "0"); } catch(e){} };
document.getElementById("memok").onclick = async () => {
  const t = memtxt.value.trim(); if(!t) return;
  await post("/aprende", {titulo: memtit.value.trim() || t.slice(0, 60), texto: t, tipo: memtipo.value});
  memtxt.value = ""; memtit.value = ""; memoria(); };
document.getElementById("memarq").onchange = async e => {
  for (const f of e.target.files) await post("/documento", {nome: f.name, conteudo: await f.text()});
  e.target.value = ""; memoria(); };
async function memoria(){
  try{
    const m = await (await fetch("/memoria")).json();
    document.getElementById("memresumo").textContent = Object.entries(m.tipos).map(([k,v]) => `${v} ${k}`).join(" · ") +
      ` · ${m.conversas} conversas (${m.corrigidas} corrigidas)`;
    const ul = document.getElementById("memult"); ul.innerHTML = m.ultimos.length ? '<div class="t" style="margin-top:4px">ensinados por último</div>' : "";
    for (const x of m.ultimos){
      const d = document.createElement("div"); d.className = "it";
      const s = document.createElement("span"); s.textContent = `#${x.id} · ${x.tipo} · ${x.titulo}`; d.appendChild(s);
      const b = document.createElement("button"); b.className = "fant"; b.textContent = "🗑"; b.title = "esquecer";
      b.onclick = async () => { if(confirm("Esquecer isto?")){ await post("/esquece", {id: x.id}); memoria(); } };
      d.appendChild(b); ul.appendChild(d);
    }
  }catch(err){}
}
memoria(); setInterval(memoria, 8000);
function chip(id, ok, texto){ const c = document.getElementById(id); c.className = "chip " + (ok ? "on" : "off");
  c.querySelector("span").textContent = texto; }
async function atualiza(){
  try{
    const e = await (await fetch("/estado")).json();
    roboLigado = e.robo;
    const vs = document.getElementById("vozsel"); if (document.activeElement !== vs && e.voz) vs.value = e.voz;
    const sv = document.getElementById("saida"); if (document.activeElement !== sv && e.saida_voz) sv.value = e.saida_voz;
    chip("c_robo", e.robo, e.robo ? "robô ligado" : "robô desligado");
    chip("c_wla", e.wla, e.wla ? "WLA online" : "WLA offline");
    const ov = e.ouvido;
    chip("c_ouv", !!(ov && ov.ligado), !ov ? "ouvido indisponível" : ov.ligado ? "ouvindo" : "ouvido desligado");
    document.getElementById("camoff").style.display = e.robo ? "none" : "flex";
    if (!e.robo) document.getElementById("cam").removeAttribute("src");
    if (audioN === null) audioN = e.audio_n;
    else if (e.audio_n > audioN) {   // 30/10: frase a frase, em ordem (stream)
      for (let k = audioN + 1; k <= e.audio_n; k++) if (vozweb.checked) filaAudio.push(k);
      audioN = e.audio_n; tocaFila(); }
    document.getElementById("estado").textContent = e.rodando ? `executando etapa ${e.passo+1}/${e.plano.length}` : "parado";
    document.getElementById("lib").checked = e.liberado;
    document.getElementById("conf").disabled = !(e.pendente && e.liberado && (!e.rodando || e.retry));
    const hab = document.getElementById("hab");
    if (!hab.dataset.ok){ hab.innerHTML = '<span style="background:none;border:0;padding-left:0">O WLA sabe:</span>';
      for (const h of e.habilidades){ const s = document.createElement("span"); s.textContent = h; hab.appendChild(s); }
      hab.dataset.ok = 1; }
    document.getElementById("ouv").checked = !!(ov && ov.ligado);
    document.getElementById("vozexe").checked = !!e.voz_executa;
    document.getElementById("ouv").disabled = !ov || !e.robo;
    document.getElementById("ouviu").textContent = !ov ? "(ouvido desligado: --ouvido 1)" : !e.robo ? "" : !ov.pronto ? "carregando o Whisper..." :
      (ov.pacotes ? "" : "⚠️ sem áudio do microfone · ") + (ov.ultimo ? "última fala: " + ov.ultimo : "");
    const chat = document.getElementById("chat");
    if (e.msgs.length < n) { chat.innerHTML = ""; n = 0; }
    const embaixo = chat.scrollHeight - chat.scrollTop - chat.clientHeight < 60;
    for(const m of e.msgs.slice(n)){
      const d = document.createElement("div");
      d.className = "m " + (m.quem === "você" ? "voce" : m.quem.startsWith("robô") ? "robô" : m.quem.startsWith("ER-1") ? "" : m.quem) + (m.fino ? " fino" : "");
      const meta = document.createElement("div"); meta.className = "meta";
      meta.textContent = `${m.t} · ${m.quem}${m.ms ? " · " + (m.ms/1000).toFixed(1) + " s" : ""}`; d.appendChild(meta);
      const tx = document.createElement("div"); tx.className = "txt"; tx.textContent = m.texto; d.appendChild(tx);
      if ((m.fontes && m.fontes.length) || m.id){
        const pe = document.createElement("div"); pe.className = "pe";
        if (m.fontes && m.fontes.length){ const f = document.createElement("span"); f.className = "fontes";
          f.textContent = "📚 " + m.fontes.join(" · "); pe.appendChild(f); }
        if (m.id){ const b = document.createElement("button"); b.className = "corr"; b.textContent = "✏️ corrigir";
          b.onclick = () => { const c = prompt("Qual seria a resposta certa?"); if (c && c.trim()) post("/corrige", {id: m.id, texto: c.trim()}); };
          pe.appendChild(b); }
        d.appendChild(pe);
      }
      chat.appendChild(d);
    }
    if (e.msgs.length > n && embaixo) chat.scrollTop = chat.scrollHeight;
    n = e.msgs.length;
  }catch(err){}
  setTimeout(atualiza, 700);
}
atualiza();
</script>
<script type="importmap">{ "imports": { "three": "./static/three.module.min.js",
  "three/examples/jsm/loaders/STLLoader.js": "./static/STLLoader.js",
  "three/examples/jsm/loaders/ColladaLoader.js": "./static/ColladaLoader.js" } }</script>
<script type="module">
// 03/11: o mapa de ocupação do lidar (mapa_ocupacao.py na PGX) com o G1 nas juntas medidas
import { montaMundo } from './static/mapa3d.js';
montaMundo(document.getElementById('mundo3d'), {info: document.getElementById('m3dinfo')});
document.getElementById('m3dlimpa').onclick = e => { e.preventDefault(); if (confirm('Zerar o mapa do lidar?')) fetch('mapa_limpa'); };
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--porta", type=int, default=8092)
    ap.add_argument("--er1", default="http://127.0.0.1:8098")
    ap.add_argument("--cabine", default="http://127.0.0.1:8090", help="a cabine do robô (sobe_cabine_wla.sh)")
    ap.add_argument("--painel", default="http://192.168.123.164:8095", help="painel do robô (voz: POST /fala)")
    ap.add_argument("--piper", default=str(__import__("pathlib").Path.home() / "wla_testes/tts"))
    ap.add_argument("--voz", default="pt_BR-faber-medium.onnx", help="voz Piper (pasta --piper)")
    ap.add_argument("--fala-respostas", type=int, default=1, help="1 = o robô fala as respostas do ER-1 na conversa")
    ap.add_argument("--memoria", type=int, default=6, help="quantas trocas anteriores da conversa o ER-1 recebe")
    ap.add_argument("--cabine-navegador", default=None,
                    help="endereço da cabine do robô como o NAVEGADOR o vê (para a câmera); padrão: o mesmo host da página, :8090")
    ap.add_argument("--checa-s", type=float, default=5.0, help="segundos entre os 'terminou?'")
    ap.add_argument("--tempo-max", type=float, default=120.0, help="segundos por tentativa de cada etapa")
    ap.add_argument("--volta-s", type=float, default=10.0, help="segundos esperando voltar à pose inicial")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--ouvido", type=int, default=1, help="1 = ouve 'Prometheus, ...' pelo microfone do robô (Whisper)")
    ap.add_argument("--sim-s", type=float, default=25.0, help="segundos esperando o 'sim' falado depois do plano")
    ap.add_argument("--ouvir-ao-iniciar", type=int, default=0, help="1 = já começa ouvindo 'Prometheus' (padrão: "
                    "desligado, liga na chave da página)")
    ap.add_argument("--ip-local", default="192.168.123.165", help="IP da PGX na rede do robô (multicast do microfone)")
    ap.add_argument("--led-voz", type=int, default=1, help="1 = a luz do rosto mostra acordou/pensando/falando")
    ap.add_argument("--memoria-db", default=str(__import__("pathlib").Path.home() / "er1_memoria/memoria.db"),
                    help="a memória do ER-1 (memoria_er1.py)")
    ap.add_argument("--mapa", default="http://127.0.0.1:8603", help="mapa_ocupacao.py (o lidar)")
    ap.add_argument("--saida-voz", default="auto", choices=("auto", "robo", "pagina", "ambos"),
                    help="onde ele fala (auto = por onde foi perguntado)")
    ap.add_argument("--voz-navegador", type=int, default=1, help="1 = a voz também toca na página (/voz.wav)")
    ap.add_argument("--tts-url", default="http://127.0.0.1:8099", help="voz Kokoro (voz_kokoro.py); vazio = só Piper")
    ap.add_argument("--tts-voz", default="pm_alex", help="pm_alex | pm_santa | pf_dora (Kokoro) | faber (Piper fluido)")
    ap.add_argument("--tts-vel", type=float, default=1.05, help="velocidade da fala")
    ap.add_argument("--stream", type=int, default=1, help="1 = fala frase a frase enquanto o ER-1 gera")
    a = ap.parse_args()
    o = Orquestrador(a)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _json(self, obj, codigo=200):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(codigo)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _bin(self, b, tipo, cache=0):
            self.send_response(200)
            self.send_header("Content-Type", tipo)
            self.send_header("Cache-Control", f"max-age={cache}" if cache else "no-store")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith("/estado"):
                with o.trava:
                    msgs = list(o.msgs)
                pend = (o.pendente is not None and time.time() - o.pendente_t <= 60) or o.retry is not None
                return self._json({"msgs": msgs, "rodando": o.rodando, "passo": o.passo,
                                   "plano": [PT[p] for p in o.plano], "habilidades": [PT[h] for h in HABILIDADES],
                                   "pendente": pend, "liberado": o.esta_liberado(), "retry": o.retry is not None,
                                   "wla": o.wla_online(), "voz_executa": o.voz_liberada(),
                                   "robo": o.robo_ligado(), "audio_n": o.audio_n, "voz": o.voz, "saida_voz": o.saida_voz,
                                   "ouvido": None if o.ouvido is None else {
                                       "ligado": o.ouvido.ligado, "pronto": o.ouvido.pronto(), "pacotes": o.ouvido.pacotes,
                                       "ultimo": ("✅ " if o.ouvido.ultimo["ativou"] else "· ") + o.ouvido.ultimo["texto"]
                                       if o.ouvido.ultimo["texto"] else ""}})
            if self.path.startswith("/voz.wav"):
                m_ = re.search(r"[?&]n=(\d+)", self.path)
                b = o.audios.get(int(m_.group(1)), b"") if m_ else o.audio_wav
                self.send_response(200 if b else 404)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
                return
            if self.path.startswith("/memoria"):
                return self._json(o.mem.resumo())
            if self.path.startswith("/exporta.jsonl"):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".jsonl") as f:
                    o.mem.exporta(f.name)
                    b = open(f.name, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="conversas_prometheus.jsonl"')
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
                return
            if self.path.startswith("/cam.jpg"):
                try:
                    b = urllib.request.urlopen(f"{a.er1}/camera/head_stereo_left.jpg", timeout=2).read()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(b)))
                    self.end_headers()
                    self.wfile.write(b)
                except Exception:  # noqa: BLE001
                    self.send_response(404)
                    self.end_headers()
                return
            if self.path.startswith(("/mapa.json", "/mapa_limpa", "/vivo_mapa.json")):   # 03/11: mapa_ocupacao.py
                try:
                    rota = ("/mapa.json" if self.path.startswith("/mapa.json") else
                            "/vivo.json" if self.path.startswith("/vivo_mapa.json") else "/limpa")
                    b = urllib.request.urlopen(f"{a.mapa}{rota}", timeout=1.5).read()
                except Exception:  # noqa: BLE001
                    b = b'{"pronto": false}'
                return self._bin(b, "application/json")
            if self.path.startswith(("/static/", "/robo/")):   # 03/11: three.js e o URDF do G1, os mesmos da :8090
                base = CABINE_DIR / ("static" if self.path.startswith("/static/") else "../../assets/g1")
                rel = self.path.split("?")[0].split("/", 2)[2]
                arq = (base / rel).resolve()
                if base.resolve() not in arq.parents or not arq.is_file():
                    self.send_response(404)
                    self.end_headers()
                    return
                tipo = {".js": "text/javascript", ".urdf": "application/xml"}.get(arq.suffix, "application/octet-stream")
                # as bibliotecas (three.js, malhas) ficam em cache; o nosso mapa3d.js muda: sempre o novo
                return self._bin(arq.read_bytes(), tipo, cache=0 if arq.name == "mapa3d.js" else 86400)
            host = (self.headers.get("Host") or "localhost").split(":")[0]
            cab = a.cabine_navegador or f"http://{host}:8090"
            b = PAGINA.replace("__CABINE__", cab).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                corpo = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                corpo = {}
            texto = (corpo.get("texto") or "").strip()
            if self.path == "/conversa" and texto:
                threading.Thread(target=o.conversa, args=(texto,), daemon=True).start()
            elif self.path == "/objetivo" and texto:
                threading.Thread(target=o.objetivo, args=(texto,), daemon=True).start()
            elif self.path == "/parar":
                o.parar()
            elif self.path == "/reset":
                o.reset()
            elif self.path == "/confirma":
                threading.Thread(target=o.confirma, daemon=True).start()
            elif self.path == "/libera":
                o.libera(bool(corpo.get("sim")))
            elif self.path == "/ouvido" and o.ouvido is not None:
                o.ouvido.ligado = bool(corpo.get("sim"))
                o.diz("sistema", "🎤 ouvindo 'Prometheus'" if o.ouvido.ligado else "🎤 ouvido desligado")
            elif self.path == "/libera_voz":
                o.libera_voz(bool(corpo.get("sim")))
            elif self.path == "/led":
                c = (corpo.get("cor") or "#000000").lstrip("#")
                o.led_base = tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
                o.led(o.led_base)
            elif self.path == "/aprende" and texto:
                i = o.mem.adiciona(corpo.get("titulo") or texto[:60], texto, corpo.get("tipo") or "fato", "página")
                o.diz("sistema", f"📚 aprendi (#{i}): {corpo.get('titulo') or texto[:60]}", fino=True)
            elif self.path == "/documento" and corpo.get("conteudo"):
                nome = corpo.get("nome") or "documento.txt"
                n_ = o.mem.adiciona_documento(nome, corpo["conteudo"], f"upload:{nome}")
                o.diz("sistema", f"📚 '{nome}' entrou na memória ({n_} itens)", fino=True)
            elif self.path == "/corrige" and texto:
                r = o.corrige(texto, corpo.get("id"))
                o.diz("ER-1", r)
            elif self.path == "/saida_voz" and corpo.get("saida") in ("auto", "robo", "pagina", "ambos"):
                o.saida_voz = corpo["saida"]
                o.diz("sistema", f"🔊 a voz sai: {o.saida_voz}", fino=True)
            elif self.path == "/voz" and corpo.get("voz") in ("pm_alex", "pm_santa", "pf_dora", "faber"):
                o.voz = corpo["voz"]
                threading.Thread(target=o.fala, args=("Pronto, esta é a minha voz agora.", "voz"), daemon=True).start()
            elif self.path == "/esquece" and corpo.get("id"):
                o.mem.apaga(int(corpo["id"]))
                o.diz("sistema", f"🗑 esqueci o item #{corpo['id']}", fino=True)
            elif self.path == "/falar" and texto:
                threading.Thread(target=o.fala, args=(texto, "você"), kwargs={"no_robo": True}, daemon=True).start()
            return self._json({"ok": True})

    o.diz("sistema", "cabine do ER-1 pronta. Perguntar = conversa · Planejar = o ER-1 monta o plano (nada vai ao "
                     "robô) · CONFIRMAR executa (precisa da chave 'robô liberado') · Falar = o robô fala o texto · "
                     "PARAR a qualquer momento.")
    print(f"🧠 cabine do ER-1 em http://0.0.0.0:{a.porta}  (ER-1 {a.er1}, cabine do robô {a.cabine})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.porta), H).serve_forever()


if __name__ == "__main__":
    main()
