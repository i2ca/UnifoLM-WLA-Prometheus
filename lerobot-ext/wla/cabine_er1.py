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
import json
import re
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# o que o WLA atual sabe fazer (frases do treino; o servidor traduz para as descritivas)
HABILIDADES = [
    "Pick up the white mug.",
    "Place the white mug under the coffee strainer.",
    "Pick up the apple.",
    "Pick up the apple and place it on the black X.",
]
# a mesa do café (objetos_cafe/objetos.json): o ER-1 CONFIRMA um a um (sim/não) antes de responder — com a lista
# inteira no texto ele citava tudo, até a maçã que não estava na mesa (27/10)
OBJETOS_MESA = {   # nome em português (o que o ER-1 deve usar) -> perguntas sim/não (qualquer "sim" vale)
    "chaleira elétrica de inox": ["Is there an electric kettle on the table?", "Is there a kettle on the table?"],
    "coador de café (marrom-claro, no suporte)": ["Is there a light brown coffee dripper (coffee strainer) on the table?",
                                                  "Is there a coffee filter cone on a stand?"],
    "caneca branca": ["Is there a white mug on the table?"],
    "pote de café (marrom-escuro)": ["Is there a dark brown coffee jar on the table?"],
    "tampa do pote (fora do pote)": ["Is there a round dark brown lid lying on the table, separate from the jar?"],
    "concha de café preta": ["Is there a small black coffee scoop on the table?"],
    "X preto na mesa": ["Is there a black X mark on the table?"],
    "maçã vermelha": ["Is there a red apple on the table?"],
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
        self.ouvido = None
        if getattr(a, "ouvido", 0):
            from ouvido_g1 import Ouvido
            self.ouvido = Ouvido(self.voz_pergunta, self.voz_estado, ip_local=a.ip_local,
                                 log=lambda t: print(t, flush=True))

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
    def _pcm(self, texto):
        """Voz Piper em português (pt_BR-faber, gerada aqui na PGX) -> PCM 16 kHz mono. A voz interna do G1 só fala
        inglês/chinês; o painel do robô toca o PCM pelo mesmo caminho do alarme de pânico."""
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([f"{self.a.piper}/.venv/bin/piper", "-m", f"{self.a.piper}/{self.a.voz}", "-f", f"{d}/a.wav"],
                           input=texto.encode(), check=True, capture_output=True)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", f"{d}/a.wav", "-ar", "16000", "-ac", "1",
                            "-f", "s16le", f"{d}/a.pcm"], check=True)
            return open(f"{d}/a.pcm", "rb").read()

    # ── luz do rosto e voz ──
    CORES = {"vermelh": ((255, 0, 0), "vermelha"), "verde": ((0, 255, 0), "verde"), "azul": ((0, 80, 255), "azul"),
             "amarel": ((255, 200, 0), "amarela"), "laranja": ((255, 110, 0), "laranja"),
             "rox": ((150, 0, 255), "roxa"), "lilás": ((200, 120, 255), "lilás"), "rosa": ((255, 60, 150), "rosa"),
             "branc": ((255, 255, 255), "branca"), "ciano": ((0, 220, 255), "ciano"),
             "turquesa": ((0, 220, 180), "turquesa")}
    LUZ = re.compile(r"\b(luz|led|cor|olhos?|rosto)\b", re.IGNORECASE)

    def led(self, rgb):
        try:
            r, g, b = rgb
            json.loads(urllib.request.urlopen(urllib.request.Request(
                f"{self.a.painel}/led?r={r}&g={g}&b={b}", data=b"", method="POST"), timeout=3).read())
        except Exception:  # noqa: BLE001
            pass

    def voz_estado(self, e):
        """O ouvido mudou de estado. Só "acordou" (ouviu 'Prometheus' sozinho) acende — barulho não pisca a luz."""
        if self.a.led_voz and e == "acordou":
            self.led((0, 120, 255))
            self.fala("Sim?", "voz", registra=False)

    def voz_pergunta(self, texto):
        """Ouviu "Prometheus, <texto>": pergunta de luz é resolvida aqui; o resto vai para a conversa (uma ORDEM vira
        PLANO e precisa de CONFIRMAR na tela, como digitada — a voz não libera nada)."""
        if self.a.led_voz:
            self.led((150, 0, 255))   # pensando
        t = texto.lower()
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

    def fala(self, texto, origem="ER-1", registra=True):
        """O robô fala em português (pelo painel do robô). registra=False: o texto já está no chat (resposta do ER-1
        marcada com 🔊) — não repete a mesma frase duas vezes."""
        try:
            pcm = self._pcm(texto)
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

    # ── travas ──
    def libera(self, sim):
        self.liberado, self.liberado_t = bool(sim), time.time()
        self.diz("sistema", "🔓 robô LIBERADO para executar planos confirmados" if sim else "🔒 robô TRAVADO")

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
        """Pergunta sim/não (ER-1 original) para cada objeto conhecido; devolve (vistos, ausentes) em português."""
        vistos, ausentes = [], []
        for nome, qs in OBJETOS_MESA.items():
            sim = False
            for q in qs:
                try:
                    r, _ = self.er1(q + " Answer only yes or no.", max_tok=3, lora=0, timeout=20)
                    if r.lower().startswith("yes"):
                        sim = True
                        break
                except Exception:  # noqa: BLE001
                    pass
            (vistos if sim else ausentes).append(nome)
        print("[ER-1] confirmei na mesa: " + (", ".join(vistos) or "nada da lista"), flush=True)
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

    PERGUNTA_TAREFA = re.compile(r"(sabe|consegue|pode|capaz).{0,25}(fazer|pegar|ajudar)|o que (você|vc) faz|"
                                 r"tarefa|habilidade|ajud", re.IGNORECASE)

    def sem_repeticao(self, r):
        """Tira as frases da resposta que já apareceram (quase iguais) nas respostas anteriores da conversa."""
        def norma(f):
            return re.sub(r"[^a-zà-ú ]", "", f.lower()).strip()
        ditas = {norma(f) for _, ant in self.historico for f in re.split(r"(?<=[.!?])\s+", ant) if norma(f)}
        frases = re.split(r"(?<=[.!?])\s+", r)
        novas = [f for f in frases if norma(f) not in ditas]
        return " ".join(novas).strip() or frases[0]

    def conversa(self, texto, voz=False):
        if voz:
            self.diz("sistema", "🎤 ouvi pela voz do robô", fino=True)
        if self.COMANDO.match(texto):   # ordem ("coloca o copo no coador"): vira PLANO (nada vai ao robô sem CONFIRMAR)
            print("[sistema] ordem no chat: vira plano (nada vai ao robô sem CONFIRMAR)", flush=True)
            return self.objetivo(texto)
        self.diz("você", texto)
        try:
            # 27/10: com a pergunta ANTES das instruções o ER-1 só repetia a pergunta; e sem a lista ele trocava nomes
            # ("copo de vidro" = coador, "xícara" = pote). Agora: papel -> objetos que podem estar na mesa -> pergunta.
            vistos, ausentes = self.objetos_presentes()
            conversa = "".join(f"Usuário: {u}\nRobô: {r_}\n" for u, r_ in self.historico[-self.a.memoria:])
            q = ("Você é o cérebro de um robô humanoide (Unitree G1). A imagem é da câmera da sua cabeça, olhando a mesa "
                 "à sua frente.\n"
                 f"Objetos CONFIRMADOS na mesa agora (use EXATAMENTE estes nomes em português): "
                 f"{'; '.join(vistos) if vistos else 'nenhum dos conhecidos'}.\n"
                 f"Objetos que NÃO estão na mesa: {'; '.join(ausentes) if ausentes else 'nenhum'}.\n"
                 "Nunca diga que um objeto confirmado não está na mesa. 'Copo' e 'caneca' são a mesma coisa.\n"
                 # 27/10: com a lista de tarefas em TODA pergunta ele terminava toda resposta com "consigo pegar a
                 # caneca e colocar embaixo do coador" (e a própria conversa reforçava). Só entra quando perguntam.
                 + (f"As ÚNICAS tarefas que você sabe fazer com as mãos: {'; '.join(PT[h] for h in HABILIDADES)}. "
                    "Não diga que sabe fazer outras.\n" if self.PERGUNTA_TAREFA.search(texto) else
                    "Não ofereça tarefas nem diga o que sabe fazer: o usuário não perguntou isso.\n")
                 + (f"Conversa até agora:\n{conversa}" if conversa else "")
                 + f"Usuário: {texto}\nResponda SÓ ao que o usuário perguntou, em português do Brasil, em 1 a 3 "
                   "frases curtas, sem repetir o que já disse antes na conversa.\nRobô:")
            r, ms = self.er1(q, max_tok=self.a.max_tokens)
            r = self.traduz(r)
            r = re.sub(r"^\s*(rob[oô]|robot|ER-1)\s*:\s*", "", r, flags=re.IGNORECASE).strip()   # tira o rótulo "Robô:"
            r = self.sem_repeticao(r)
            self.historico.append((texto, r))
            self.historico = self.historico[-20:]
            fala = bool(self.a.fala_respostas)
            self.diz("ER-1" + (" 🔊" if fala else ""), r, ms=ms)
            if fala:
                self.fala(r, "ER-1", registra=False)
        except Exception as e:  # noqa: BLE001
            self.diz("sistema", f"ER-1 indisponível: {e}")

    # ── objetivo: planejar + executar ──
    def objetivo(self, texto):
        if self.rodando:
            self.diz("sistema", "já estou executando um objetivo — aperte PARAR primeiro")
            return
        self.diz("você", f"🎯 objetivo: {texto}")
        lista = "\n".join(f"{i}. {h}" for i, h in enumerate(HABILIDADES, 1))
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
        self.pendente, self.pendente_t = plano, time.time()
        self.diz("sistema", "⚠️ nada foi mandado ao robô. Confira o plano: com o robô LIBERADO, aperte CONFIRMAR "
                            "em até 60 s para executar.", pendente=True)
        # o robô fala o plano ANTES de fazer qualquer coisa e pede a confirmação
        etapas = ", depois ".join(PT[p] for p in plano)
        self.fala(f"Meu plano é {etapas}. Posso começar? Confirme na tela.", "plano")

    def confirma(self):
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
        if not self.esta_liberado():
            self.diz("sistema", "🔒 o robô está TRAVADO: ligue a chave 'robô liberado' antes de confirmar")
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
            return
        self.pendente = None
        self.plano, self.passo, self.rodando = plano, -1, True
        self.cancela.clear()
        self.diz("você", "✔ CONFIRMAR plano")
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
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Cabine do ER-1</title>
<style>
:root{--fundo:#0f1115;--cartao:#181b22;--texto:#e7e9ee;--fraco:#8a90a0;--acento:#5aa9ff;--ok:#3ecf8e;--erro:#ff6b6b}
*{box-sizing:border-box}body{margin:0;background:var(--fundo);color:var(--texto);font:15px system-ui,sans-serif}
main{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:12px;padding:12px;height:100vh}
@media(max-width:900px){main{grid-template-columns:1fr;height:auto}}
.cartao{background:var(--cartao);border-radius:10px;padding:10px;display:flex;flex-direction:column;min-height:0}
img{width:100%;border-radius:8px;background:#000}h1{font-size:16px;margin:0 0 8px}
#chat{flex:1;overflow:auto;display:flex;flex-direction:column;gap:6px;min-height:300px}
.m{padding:7px 9px;border-radius:8px;max-width:95%;white-space:pre-wrap}.voce{align-self:flex-end;background:#25406b}
.ER-1{background:#24262e}.ER-1.🔊{background:#24262e}.orquestrador{background:#1f3326}.sistema{background:#3a2424}.fino{opacity:.6;font-size:13px}
.t{color:var(--fraco);font-size:11px;margin-right:6px}
form{display:flex;gap:6px;margin-top:8px}input{flex:1;padding:9px;border-radius:8px;border:1px solid #333;background:#0d0f13;color:var(--texto)}
button{padding:9px 12px;border:0;border-radius:8px;background:var(--acento);color:#fff;cursor:pointer}
button.obj{background:var(--ok)}button.parar{background:var(--erro);font-weight:700}
button.conf{background:#d98e04}button:disabled{opacity:.35;cursor:not-allowed}
.barra{display:flex;gap:8px;align-items:center;margin-top:8px;flex-wrap:wrap}.chave{display:flex;gap:6px;align-items:center}
.chave input{width:18px;height:18px}.robô{background:#2d2550}#hab{color:var(--fraco);font-size:13px;margin-top:8px}
</style></head><body><main>
<section class="cartao"><h1>Câmera da cabeça</h1><img id="cam" alt="câmera da cabeça">
<div id="hab"></div></section>
<section class="cartao"><h1>Conversa com o ER-1 <span id="estado" class="t"></span></h1><div id="chat"></div>
<form id="f"><input id="txt" placeholder="pergunte algo, ou escreva um objetivo (ex.: pegue a caneca e coloque no coador)" autocomplete="off">
<button type="submit" title="só conversa (Enter)">Perguntar</button>
<button type="button" class="obj" id="obj" title="o ER-1 monta o PLANO (nada vai ao robô)">Planejar</button>
<button type="button" id="falar" title="o robô fala este texto">🔊 Falar</button></form>
<div class="barra"><label class="chave"><input type="checkbox" id="lib"> robô liberado</label>
<button type="button" class="conf" id="conf" disabled>✔ CONFIRMAR plano</button>
<button type="button" class="parar" id="parar">⏹ PARAR</button>
<button type="button" id="reset" title="apaga a memória da conversa e limpa o chat">🧹 Novo contexto</button></div>
<div class="barra"><label class="chave" title="fale 'Prometheus, ...' perto do robô"><input type="checkbox" id="ouv"> 🎤 ouvir "Prometheus"</label>
<label class="chave">luz do rosto <input type="color" id="cor" value="#000000" style="width:42px;height:28px"></label>
<span id="ouviu" class="t"></span></div></section></main>
<script>
const CAB = "__CABINE__";
document.getElementById("cam").src = CAB + "/camera/head_stereo_left.mjpg";
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
document.getElementById("cor").onchange = e => post("/led", {cor: e.target.value});
async function atualiza(){
  try{
    const e = await (await fetch("/estado")).json();
    document.getElementById("estado").textContent = e.rodando ? `executando etapa ${e.passo+1}/${e.plano.length}` : "parado";
    document.getElementById("lib").checked = e.liberado;
    document.getElementById("conf").disabled = !(e.pendente && e.liberado && (!e.rodando || e.retry));
    document.getElementById("hab").textContent = "O WLA sabe: " + e.habilidades.join(" · ");
    const ov = e.ouvido;
    document.getElementById("ouv").checked = !!(ov && ov.ligado);
    document.getElementById("ouv").disabled = !ov;
    document.getElementById("ouviu").textContent = !ov ? "(ouvido desligado: --ouvido 1)" : !ov.pronto ? "carregando o Whisper..." :
      (ov.pacotes ? "" : "⚠️ sem áudio do microfone (painel do robô liga) · ") + (ov.ultimo ? "última fala: " + ov.ultimo : "");
    const chat = document.getElementById("chat");
    if (e.msgs.length < n) { document.getElementById("chat").innerHTML = ""; n = 0; }
    for(const m of e.msgs.slice(n)){
      const d = document.createElement("div"); d.className = "m " + (m.quem === "você" ? "voce" : m.quem.startsWith("robô") ? "robô" : m.quem) + (m.fino ? " fino" : "");
      d.innerHTML = `<span class="t">${m.t} ${m.quem}${m.ms ? " · " + m.ms + " ms" : ""}</span>`;
      d.appendChild(document.createTextNode(m.texto)); chat.appendChild(d); chat.scrollTop = chat.scrollHeight;
    }
    n = e.msgs.length;
  }catch(err){}
  setTimeout(atualiza, 700);
}
atualiza();
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
    ap.add_argument("--ip-local", default="192.168.123.165", help="IP da PGX na rede do robô (multicast do microfone)")
    ap.add_argument("--led-voz", type=int, default=1, help="1 = a luz do rosto mostra acordou/pensando/falando")
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

        def do_GET(self):
            if self.path.startswith("/estado"):
                with o.trava:
                    msgs = list(o.msgs)
                pend = (o.pendente is not None and time.time() - o.pendente_t <= 60) or o.retry is not None
                return self._json({"msgs": msgs, "rodando": o.rodando, "passo": o.passo,
                                   "plano": [PT[p] for p in o.plano], "habilidades": [PT[h] for h in HABILIDADES],
                                   "pendente": pend, "liberado": o.esta_liberado(), "retry": o.retry is not None,
                                   "ouvido": None if o.ouvido is None else {
                                       "ligado": o.ouvido.ligado, "pronto": o.ouvido.pronto(), "pacotes": o.ouvido.pacotes,
                                       "ultimo": ("✅ " if o.ouvido.ultimo["ativou"] else "· ") + o.ouvido.ultimo["texto"]
                                       if o.ouvido.ultimo["texto"] else ""}})
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
            elif self.path == "/led":
                c = (corpo.get("cor") or "#000000").lstrip("#")
                o.led_base = tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
                o.led(o.led_base)
            elif self.path == "/falar" and texto:
                threading.Thread(target=o.fala, args=(texto, "você"), daemon=True).start()
            return self._json({"ok": True})

    o.diz("sistema", "cabine do ER-1 pronta. Perguntar = conversa · Planejar = o ER-1 monta o plano (nada vai ao "
                     "robô) · CONFIRMAR executa (precisa da chave 'robô liberado') · Falar = o robô fala o texto · "
                     "PARAR a qualquer momento.")
    print(f"🧠 cabine do ER-1 em http://0.0.0.0:{a.porta}  (ER-1 {a.er1}, cabine do robô {a.cabine})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.porta), H).serve_forever()


if __name__ == "__main__":
    main()
