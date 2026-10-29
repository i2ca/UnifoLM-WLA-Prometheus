"""OUVIDO do G1 (28/10): o microfone interno do robô -> fala em texto -> "Prometheus, ..." vira pergunta.

O painel do robô liga o microfone (voice API 1008 {"mode": 1}); o arranjo de 4 microfones manda PCM 16 kHz mono s16
por multicast UDP 239.168.123.161:5555. Aqui (na PGX, na rede do robô):
  1. detector de voz Silero (VAD): junta um trecho de fala até ~0,8 s de silêncio (no máx. --max-s);
  2. Whisper large-v3-turbo (transformers, GPU) em português;
  3. PALAVRA DE ATIVAÇÃO: só reage ao que começa com (ou contém) "Prometheus" — o Whisper escreve Prometeu,
     Prometeus, Prometius...; "Prometheus" sozinho abre uma janela de --janela-s para a pergunta vir na fala seguinte;
  4. enquanto o robô fala, o ouvido fica SURDO (senão ele se ouve e responde a si mesmo).

    o = Ouvido(ao_ouvir=lambda texto: ..., estado=lambda e: ...)   # estado: "ouvindo", "entendendo", "acordou"
    o.surdo_ate(time.time() + duracao_da_fala)
"""
import re
import socket
import struct
import threading
import time
import unicodedata

import numpy as np

GRUPO, PORTA = "239.168.123.161", 5555
SR = 16000
# Prometheus / Prometeu / Prometeus / Prometius / Prometeo (e não "prometo")
ATIVA = re.compile(r"\b(prom[ei]th?(eus|eu|ius|eo|eos|ios|iu|euz))\b[\s,.!?:;-]*", re.IGNORECASE)


def _sem_acento(t):
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


# 28/10: o Whisper em português ouve "Prometheus" como "O meteoro", "Prometeu"... — além do prompt, aceita o que
# SOA parecido nas 3 primeiras palavras (distância de edição sobre as letras juntas)
ALVO = "prometheus"


def _distancia(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        ant, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            ant, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, ant + (ca != cb))
    return d[-1]


# o que o Whisper já escreveu no lugar de "Prometheus" (log de 28/10), só no COMEÇO da frase
APELIDOS = re.compile(r"^\W*((o|ô|oi|ei|bom|bem)[\s,]+)?(o\s+)?(meteoro|prometeo|promete\s+os|prometi\s+us|pro\s+meteus|"
                      r"metheus|meteus|matheus|mateus)\b[\s,.!?:;-]*", re.IGNORECASE)
INTERJ = {"o", "oi", "ei", "hey", "ola", "e"}


def acha_ativacao(texto):
    """(início, fim) do "Prometheus" no texto, ou None. 1) o nome (regex); 2) apelidos que o Whisper já escreveu, no
    começo; 3) o que SOA parecido (distância <= 3) na 1ª palavra (ou na 2ª, depois de "oi"/"ô") — "prometo" no meio
    da frase não ativa."""
    t = _sem_acento(texto)
    m = ATIVA.search(t)
    if m:
        return m.start(), m.end()
    m = APELIDOS.search(t)
    if m:
        return m.start(), m.end()
    pals = list(re.finditer(r"[a-zA-Z]+", t))[:3]
    ini = 1 if pals and pals[0].group().lower() in INTERJ else 0
    for j in range(ini, min(ini + 2, len(pals))):   # 1 ou 2 palavras juntas a partir do começo
        junto = "".join(p.group().lower() for p in pals[ini:j + 1]).replace("h", "")
        if 7 <= len(junto) <= 12 and min(_distancia(junto, "prometeus"), _distancia(junto, "prometeu")) <= 2:
            fim = pals[j].end()
            while fim < len(t) and t[fim] in " ,.!?:;-":
                fim += 1
            return pals[0].start(), fim
    return None


# o que o Whisper inventa em cima de ruído (silêncio, ventilador)
INVENTADAS = {"obrigado", "muito obrigado", "obrigada", "e ai", "tchau", "legendas pela comunidade amara.org",
              "inscreva-se no canal", "valeu"}


def inventada(texto):
    t = _sem_acento(texto).lower().strip(" .,!?")
    if t in INVENTADAS or all(p.strip(" .,!?") in INVENTADAS for p in re.split(r"[.!?]", t) if p.strip(" .,!?")):
        return True
    pal = t.split()
    return len(pal) >= 6 and len(set(pal)) <= len(pal) / 3   # "e ai e ai e ai ..." / "que é que é que é"


class Ouvido:
    def __init__(self, ao_ouvir, estado=None, ip_local="192.168.123.165", modelo="openai/whisper-large-v3-turbo",
                 janela_s=8.0, max_s=8.0, silencio_s=0.7, gpu_lock=None, log=print):
        self.ao_ouvir, self.estado, self.log = ao_ouvir, estado or (lambda e: None), log
        self.ip_local, self.modelo_nome = ip_local, modelo
        self.janela_s, self.max_s, self.silencio_s = janela_s, max_s, silencio_s
        self.gpu_lock = gpu_lock or threading.Lock()
        self._surdo = 0.0
        self._acordado_ate = 0.0
        self.ligado = True
        self.ultimo = {"t": 0.0, "texto": "", "ativou": False}
        self.pacotes = 0
        self.nivel_ruido = 300.0
        self._asr = None
        self._prompt = None
        threading.Thread(target=self._carrega, daemon=True, name="whisper").start()
        threading.Thread(target=self._laco, daemon=True, name="ouvido").start()

    # ── controle ──
    def surdo_ate(self, t):
        self._surdo = max(self._surdo, t)

    def pronto(self):
        return self._asr is not None

    def _carrega(self):
        import torch
        from transformers import pipeline
        t0 = time.time()
        self._asr = pipeline("automatic-speech-recognition", model=self.modelo_nome, dtype=torch.float16,
                             device="cuda:0")
        # contexto antes da fala: o Whisper passa a preferir a palavra "Prometheus" (em vez de "O meteoro")
        # 28/10: SÓ o nome — com frases de exemplo no prompt, o Whisper repetia as frases em cima do ruído
        self._prompt = torch.tensor(self._asr.tokenizer.get_prompt_ids("Prometheus."), device="cuda:0")
        self.log(f"🎤 Whisper pronto ({self.modelo_nome}, {time.time() - t0:.0f} s)")

    # ── áudio ──
    def _socket(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", PORTA))
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                     struct.pack("4s4s", socket.inet_aton(GRUPO), socket.inet_aton(self.ip_local)))
        s.settimeout(1.0)
        return s

    def _laco(self):
        while True:
            try:
                self._escuta(self._socket())
            except Exception as e:  # noqa: BLE001
                self.log(f"🎤 ouvido caiu ({e}); tentando de novo em 3 s")
                time.sleep(3)

    def _escuta(self, s):
        """28/10: SILERO VAD (rede de detecção de voz) em blocos de 512 amostras = 1 pacote do microfone do G1. O
        detector por energia não fechava o trecho (o ruído do ventilador varia) e gravava até o limite."""
        import collections

        import torch
        from silero_vad import VADIterator, load_silero_vad
        torch.set_num_threads(1)
        vad = VADIterator(load_silero_vad(), sampling_rate=SR, threshold=0.5, min_silence_duration_ms=int(
            self.silencio_s * 1000), speech_pad_ms=150)
        antes = collections.deque(maxlen=20)   # ~0,6 s antes do início (o começo do "Prometheus")
        fala, falando = [], False
        while True:
            try:
                d, _ = s.recvfrom(65536)
            except socket.timeout:
                continue
            self.pacotes += 1
            x = np.frombuffer(d, dtype=np.int16)
            if not self.ligado or time.time() < self._surdo:
                if falando:
                    self.estado("parado")
                fala, falando = [], False
                antes.clear()
                vad.reset_states()
                continue
            ev = None
            for i in range(0, len(x) - 511, 512):
                # o microfone chega BAIXO: 4x só para o detector (o ventilador fica em ~0,2 de confiança, voz > 0,5)
                e = vad(torch.from_numpy(np.clip(x[i:i + 512].astype(np.float32) * (4.0 / 32768.0), -1.0, 1.0)))
                ev = e or ev
            if not falando:
                antes.append(x)
                if ev and "start" in ev:
                    falando, fala = True, list(antes)
                    self.estado("ouvindo")
                continue
            fala.append(x)
            total = sum(len(f) for f in fala) / SR
            if (ev and "end" in ev) or total >= self.max_s:
                falando = False
                antes.clear()
                vad.reset_states()
                audio = np.concatenate(fala).astype(np.float32) / 32768.0
                fala = []
                if total >= 0.5:
                    threading.Thread(target=self._entende, args=(audio,), daemon=True).start()
                else:
                    self.estado("parado")

    def _guarda(self, audio):
        """Os últimos 10 trechos em ~/ouvido_trechos/*.wav (para ouvir o que o microfone pegou)."""
        try:
            import wave
            from pathlib import Path
            pasta = Path.home() / "ouvido_trechos"
            pasta.mkdir(exist_ok=True)
            w = wave.open(str(pasta / time.strftime("%H%M%S.wav")), "wb")
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
            w.close()
            for velho in sorted(pasta.glob("*.wav"))[:-10]:
                velho.unlink()
        except Exception:  # noqa: BLE001
            pass

    # ── fala -> texto -> ativação ──
    def _entende(self, audio):
        if self._asr is None:
            self.estado("parado")
            return
        self.estado("entendendo")
        t0 = time.time()
        self._guarda(audio)
        pico = float(np.abs(audio).max()) or 1.0
        audio = audio * min(0.9 / pico, 20.0)   # normaliza o volume (o microfone chega baixo)
        kw = {"language": "portuguese", "task": "transcribe"}
        if self._prompt is not None:
            kw["prompt_ids"] = self._prompt
        with self.gpu_lock:
            r = self._asr({"raw": audio, "sampling_rate": SR}, generate_kwargs=kw)
        texto = (r.get("text") or "").strip()
        # 28/10: siglas que o Whisper escreve do jeito que ouve
        texto = re.sub(r"\b(L[- ]?Card|L[- ]?Cad|Elcad|Éle ?cad[ei]?)\b", "LCAD", texto, flags=re.IGNORECASE)
        texto = re.sub(r"\b(U ?fes|Ufez)\b", "UFES", texto, flags=re.IGNORECASE)
        # o Whisper às vezes devolve o próprio prompt ("Prometheus.") quando só tem ruído: isso NÃO é ativação
        if _sem_acento(texto).lower().strip(" .,!?") in ("prometheus", ""):
            self.log("🎤 (só o nome do prompt, ignorado)")
            self.estado("parado")
            return
        ms = (time.time() - t0) * 1000
        if not texto or len(_sem_acento(texto).strip(" .,!?")) < 2 or inventada(texto):
            if texto:
                self.log(f"🎤 (ruído, ignorado) {texto[:60]}")
            self.estado("parado")
            return
        m = acha_ativacao(texto)
        agora = time.time()
        if m:
            pergunta = (texto[:m[0]] + " " + texto[m[1]:]).strip(" ,.!?;:-")
            pergunta = ATIVA.sub(" ", _sem_acento(pergunta)) if ATIVA.search(_sem_acento(pergunta)) else pergunta
            pergunta = re.sub(r"\s+", " ", pergunta).strip(" ,.!?;:-")
        elif agora < self._acordado_ate:
            pergunta = texto
        else:
            self.ultimo = {"t": agora, "texto": texto, "ativou": False}
            self.log(f"🎤 (sem 'Prometheus', ignorado, {ms:.0f} ms) {texto}")
            self.estado("parado")
            return
        self.ultimo = {"t": agora, "texto": texto, "ativou": True}
        if not pergunta:
            self._acordado_ate = agora + self.janela_s
            self.log(f"🎤 'Prometheus' — esperando a pergunta ({self.janela_s:.0f} s)")
            self.estado("acordou")
            return
        self._acordado_ate = 0.0
        self.log(f"🎤 ({ms:.0f} ms) {texto}")
        self.ao_ouvir(pergunta)
