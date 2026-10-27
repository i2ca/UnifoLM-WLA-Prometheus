#!/usr/bin/env python
"""CABINE DO ER-1 (27/10): conversar com o ER-1 vendo a câmera do robô, e deixar ele PLANEJAR e EXECUTAR um objetivo
com as tarefas que o WLA sabe fazer — uma de cada vez, checando "terminou?" antes de passar para a próxima.

  conversa  pergunta livre ao ER-1 (com a imagem da cabeça, adaptador desligado), resposta em português
  objetivo  o ER-1 olha a mesa e escolhe a SEQUÊNCIA de tarefas (só as da lista HABILIDADES); o orquestrador manda
            cada uma para a cabine do robô (POST /tarefa), pergunta "terminou?" ao ER-1 (LoRA, a cada --checa-s) e só
            avança com dois "sim" seguidos. Não terminou em --tempo-max s: volta à pose inicial e tenta de novo uma vez;
            falhou de novo: para e avisa.
  parar     manda a cabine do robô parar (POST /parar) e cancela o objetivo

Precisa no ar: o ER-1 (sobe_er1_pergunta.sh, :8098) e a cabine do robô com o WLA (sobe_cabine_wla.sh, :8090).
Rode a cabine do robô SEM TERMINOU=1 (quem checa agora é este orquestrador).

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
        return json.loads(urllib.request.urlopen(req, timeout=10).read())

    # ── conversa ──
    def conversa(self, texto):
        self.diz("você", texto)
        try:
            q = (f"{texto}\n\n(You are the reasoning brain of a humanoid robot (Unitree G1) looking at a table through "
                 "its head camera. Answer in Brazilian Portuguese, briefly.)")
            r, ms = self.er1(q, max_tok=self.a.max_tokens)
            self.diz("ER-1", r, ms=ms)
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
        self.plano, self.passo, self.rodando = plano, -1, True
        self.cancela.clear()
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
                    self.diz("orquestrador", f"▶ etapa {i + 1}/{len(self.plano)}: {PT[frase]} "
                                             f"(tentativa {tentativa}) — mandando ao WLA")
                    self.cabine("/tarefa", {"texto": frase})
                    t0, sims = time.time(), 0
                    while time.time() - t0 < self.a.tempo_max:
                        if self._espera(self.a.checa_s):
                            return
                        try:
                            sim = self.terminou(frase)
                        except Exception as e:  # noqa: BLE001
                            self.diz("sistema", f"ER-1 (terminou?) falhou: {e}")
                            continue
                        sims = sims + 1 if sim else 0
                        self.diz("ER-1", f"terminou \"{PT[frase]}\"? {'sim' if sim else 'não'} ({sims}/2)", fino=True)
                        if sims >= 2:
                            break
                    if sims >= 2:
                        self.diz("orquestrador", f"✅ etapa {i + 1} concluída: {PT[frase]}")
                        break
                    if tentativa == 1:
                        self.diz("orquestrador", f"⏱️ {self.a.tempo_max:.0f} s sem concluir — voltando à pose inicial "
                                                 "e tentando de novo")
                        self.cabine("/tarefa", {"texto": ""})
                        self.cabine("/inicial")
                        if self._espera(self.a.volta_s):
                            return
                else:
                    self.diz("orquestrador", f"❌ não consegui \"{PT[frase]}\" em duas tentativas — parei")
                    self.cabine("/tarefa", {"texto": ""})
                    return
            self.diz("orquestrador", "🏁 objetivo concluído!")
            self.cabine("/tarefa", {"texto": ""})
        except Exception as e:  # noqa: BLE001
            self.diz("sistema", f"erro na execução: {e}")
        finally:
            self.rodando, self.passo = False, -1

    def parar(self):
        self.cancela.set()
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
.ER-1{background:#24262e}.orquestrador{background:#1f3326}.sistema{background:#3a2424}.fino{opacity:.6;font-size:13px}
.t{color:var(--fraco);font-size:11px;margin-right:6px}
form{display:flex;gap:6px;margin-top:8px}input{flex:1;padding:9px;border-radius:8px;border:1px solid #333;background:#0d0f13;color:var(--texto)}
button{padding:9px 12px;border:0;border-radius:8px;background:var(--acento);color:#fff;cursor:pointer}
button.obj{background:var(--ok)}button.parar{background:var(--erro)}#hab{color:var(--fraco);font-size:13px;margin-top:8px}
</style></head><body><main>
<section class="cartao"><h1>Câmera da cabeça</h1><img id="cam" alt="câmera da cabeça">
<div id="hab"></div></section>
<section class="cartao"><h1>Conversa com o ER-1 <span id="estado" class="t"></span></h1><div id="chat"></div>
<form id="f"><input id="txt" placeholder="pergunte algo, ou escreva um objetivo (ex.: pegue a caneca e coloque no coador)" autocomplete="off">
<button type="submit" title="só conversa">Perguntar</button><button type="button" class="obj" id="obj" title="o ER-1 planeja e executa">Executar</button>
<button type="button" class="parar" id="parar">PARAR</button></form></section></main>
<script>
const CAB = "__CABINE__";
document.getElementById("cam").src = CAB + "/camera/head_stereo_left.mjpg";
let n = 0;
async function post(c, b){ await fetch(c, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(b||{})}); }
document.getElementById("f").onsubmit = e => { e.preventDefault(); const t = txt.value.trim(); if(t){ post("/conversa",{texto:t}); txt.value=""; } };
document.getElementById("obj").onclick = () => { const t = txt.value.trim(); if(t){ post("/objetivo",{texto:t}); txt.value=""; } };
document.getElementById("parar").onclick = () => post("/parar");
async function atualiza(){
  try{
    const e = await (await fetch("/estado")).json();
    document.getElementById("estado").textContent = e.rodando ? `executando etapa ${e.passo+1}/${e.plano.length}` : "parado";
    document.getElementById("hab").textContent = "O WLA sabe: " + e.habilidades.join(" · ");
    const chat = document.getElementById("chat");
    for(const m of e.msgs.slice(n)){
      const d = document.createElement("div"); d.className = "m " + (m.quem === "você" ? "voce" : m.quem) + (m.fino ? " fino" : "");
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
    ap.add_argument("--cabine-navegador", default=None,
                    help="endereço da cabine do robô como o NAVEGADOR o vê (para a câmera); padrão: o mesmo host da página, :8090")
    ap.add_argument("--checa-s", type=float, default=5.0, help="segundos entre os 'terminou?'")
    ap.add_argument("--tempo-max", type=float, default=120.0, help="segundos por tentativa de cada etapa")
    ap.add_argument("--volta-s", type=float, default=10.0, help="segundos esperando voltar à pose inicial")
    ap.add_argument("--max-tokens", type=int, default=200)
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
                return self._json({"msgs": msgs, "rodando": o.rodando, "passo": o.passo,
                                   "plano": [PT[p] for p in o.plano], "habilidades": [PT[h] for h in HABILIDADES]})
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
            return self._json({"ok": True})

    o.diz("sistema", "cabine do ER-1 pronta. 'Perguntar' só conversa; 'Executar' faz o ER-1 planejar e mandar as tarefas "
                     "ao robô, uma de cada vez.")
    print(f"🧠 cabine do ER-1 em http://0.0.0.0:{a.porta}  (ER-1 {a.er1}, cabine do robô {a.cabine})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.porta), H).serve_forever()


if __name__ == "__main__":
    main()
