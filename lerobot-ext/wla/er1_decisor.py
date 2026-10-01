#!/usr/bin/env python
"""
ER-1 COMO DECISOR — UnifoLM-ER-1 escolhe a instrução, o WLA-1.0 executa
========================================================================
O ER-1 (Qwen3-VL-4B de raciocínio espacial da Unitree) olha a câmera da cabeça (ZED esquerda),
LOCALIZA a fruta e o prato e DECIDE a próxima instrução para o WLA, a partir de um objetivo em
texto. A instrução escolhida vai para a cabine do `roda_wla_real.py` (POST /tarefa na 8090),
que o WLA lê a cada consulta. O ER-1 NÃO comanda junta nenhuma: ele só troca a frase.

Página: http://<pgx>:8096 — imagem com os pontos que o ER-1 marcou, o raciocínio, a decisão
e uma caixa para mudar o objetivo.

As instruções possíveis são frases do tipo das que o WLA viu no treino (o dataset de fruta tem
"Pick up the fruit and place it on the plate." e variações). "stop" só é MOSTRADO; com
--pode-parar ele também aperta o "parar" da cabine (o que aciona o pânico por software).

Isto é uma montagem NOSSA: a Unitree não publicou código ligando o ER-1 ao WLA.

    cd ~/DEV/unifolm-wla && ~/miniforge3/envs/wla/bin/python \\
        ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/er1_decisor.py
"""
import argparse
import base64
import json
import re
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch
import zmq
from PIL import Image

ER1 = "playground/Pretrained_models/UnifoLM-ER-1"
INSTRUCOES = [
    "Pick up the fruit and place it on the plate.",
    "Pick up the fruit.",
    "Place the fruit on the plate.",
    "stop",
]
PROMPT = """You are the high-level decision module of a humanoid robot with two hands, seen from its head camera.
Goal given by the operator: "{objetivo}"
Current instruction being executed by the low-level policy: "{atual}"

1. Locate the objects relevant to the goal (the fruit, the plate). Give each one as a point in 0-1000 image coordinates.
2. Decide the next instruction for the low-level policy. Choose EXACTLY one of:
{opcoes}
Choose "stop" only if the goal is already achieved or impossible.

Answer ONLY with JSON, no other text:
{{"objects": [{{"label": "...", "point_2d": [x, y]}}], "scene": "<one short sentence>", "instruction": "<one of the options>", "reason": "<short>"}}"""


class Camera:
    def __init__(self, robo, porta, nome):
        self.rgb, self.t, self.nome = None, 0.0, nome
        s = zmq.Context.instance().socket(zmq.SUB)
        s.setsockopt_string(zmq.SUBSCRIBE, "")
        s.setsockopt(zmq.RCVHWM, 2)
        s.connect(f"tcp://{robo}:{porta}")
        threading.Thread(target=self._laco, args=(s,), daemon=True).start()

    def _laco(self, s):
        while True:
            try:
                m = json.loads(s.recv_multipart()[0])
                b = (m.get("images") or {}).get(self.nome)
                if b:
                    # o servidor codifica o array RGB como se fosse BGR: o imdecode devolve RGB
                    self.rgb = cv2.imdecode(np.frombuffer(base64.b64decode(b), np.uint8), cv2.IMREAD_COLOR)
                    self.t = time.time()
            except Exception:
                time.sleep(0.05)


class Estado:
    def __init__(self, objetivo):
        self.objetivo = objetivo
        self.decisao = {}
        self.jpeg = None
        self.historico = []
        self.lock = threading.Lock()


PAGINA = """<!doctype html><html lang="pt-br"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>ER-1 Decisor</title>
<style>:root{color-scheme:dark}body{margin:0;background:#0f1115;color:#e4e6eb;font:14px system-ui,sans-serif}
main{padding:12px 16px;max-width:1100px}img{width:100%;max-width:960px;border-radius:6px;display:block;background:#000}
form{display:flex;gap:8px;margin:10px 0}input{flex:1;background:#171a21;color:#e4e6eb;border:1px solid #2a2f3a;
padding:8px;border-radius:6px;font:inherit}button{background:#1f6f3a;color:#fff;border:1px solid #3fb950;border-radius:6px;
padding:8px 14px;font:inherit;cursor:pointer}.caixa{background:#171a21;border:1px solid #2a2f3a;border-radius:8px;padding:10px 12px;margin:8px 0}
.inst{font-size:18px;font-weight:700;color:#3fb950}.m{color:#8b93a1}pre{white-space:pre-wrap;margin:0;color:#8b93a1}</style></head>
<body><main><h2 style="margin:4px 0">UnifoLM-ER-1 · decisão → UnifoLM-WLA-1.0 · execução</h2>
<form onsubmit="event.preventDefault();fetch('objetivo',{method:'POST',body:JSON.stringify({objetivo:document.getElementById('o').value})})">
<input id="o" placeholder="objetivo para o ER-1"><button>definir objetivo</button></form>
<div class="caixa"><span class="m">instrução decidida pelo ER-1 (vai para o WLA):</span><div class="inst" id="i">—</div>
<div id="r" class="m"></div></div><img src="quadro.jpg" id="q"><div class="caixa"><pre id="h"></pre></div></main>
<script>const q=document.getElementById('q');q.onload=()=>setTimeout(()=>q.src='quadro.jpg?'+Date.now(),500);
q.onerror=()=>setTimeout(()=>q.src='quadro.jpg?'+Date.now(),1500);
async function t(){try{const e=await (await fetch('estado.json',{cache:'no-store'})).json();const d=e.decisao||{};
if(document.activeElement.id!=='o')document.getElementById('o').value=e.objetivo;
document.getElementById('i').textContent=d.instruction||'—';
document.getElementById('r').textContent=(d.scene?'cena: '+d.scene+' · ':'')+(d.reason?'motivo: '+d.reason:'')+
(d.ms?' · '+d.ms+' ms':'')+(d.enviada?' · enviada à cabine':'');
document.getElementById('h').textContent=(e.historico||[]).slice(-12).reverse().map(x=>x).join('\\n')}catch(_){}
setTimeout(t,700)}t()</script></body></html>"""


def sobe_http(est, porta):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _manda(self, corpo, tipo, codigo=200):
            self.send_response(codigo)
            self.send_header("Content-Type", tipo)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(corpo)

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/":
                return self._manda(PAGINA.encode(), "text/html; charset=utf-8")
            if p == "/estado.json":
                with est.lock:
                    return self._manda(json.dumps({"objetivo": est.objetivo, "decisao": est.decisao,
                                                   "historico": est.historico[-30:]}).encode(), "application/json")
            if p == "/quadro.jpg" and est.jpeg:
                return self._manda(est.jpeg, "image/jpeg")
            self._manda(b"", "text/plain", 404)

        def do_POST(self):
            if self.path == "/objetivo":
                n = int(self.headers.get("Content-Length", 0))
                o = (json.loads(self.rfile.read(n) or b"{}").get("objetivo") or "").strip()
                if o:
                    with est.lock:
                        est.objetivo = o
                        est.historico.append(f"{time.strftime('%H:%M:%S')} objetivo -> {o}")
                return self._manda(b'{"ok":true}', "application/json")
            self._manda(b"", "text/plain", 404)

    srv = ThreadingHTTPServer(("0.0.0.0", porta), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def extrai_json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def desenha(rgb, d):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = img.shape[:2]
    for o in (d or {}).get("objects") or []:
        try:
            x, y = o["point_2d"]
            px, py = int(x / 1000 * w), int(y / 1000 * h)
        except Exception:
            continue
        cv2.circle(img, (px, py), 9, (0, 0, 0), 4)
        cv2.circle(img, (px, py), 9, (60, 220, 255), 2)
        cv2.putText(img, str(o.get("label", "?")), (px + 12, py + 5), 0, 0.6, (0, 0, 0), 4)
        cv2.putText(img, str(o.get("label", "?")), (px + 12, py + 5), 0, 0.6, (60, 220, 255), 2)
    cv2.rectangle(img, (0, h - 30), (w, h), (20, 20, 20), -1)
    cv2.putText(img, f"ER-1 -> WLA: {(d or {}).get('instruction', '?')}"[:80], (8, h - 10), 0, 0.55, (80, 230, 120), 2)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robo", default="192.168.123.164")
    ap.add_argument("--camera", default="head_stereo_left")
    ap.add_argument("--porta", type=int, default=8096)
    ap.add_argument("--cabine", default="http://127.0.0.1:8090")
    ap.add_argument("--objetivo", default="pick up the apple and put it on the plate")
    ap.add_argument("--periodo", type=float, default=10.0,
                    help="segundos entre decisões (o ER-1 disputa a GPU com o WLA: 3 s deixava o WLA 2,5x mais lento)")
    ap.add_argument("--pode-parar", action="store_true", help="'stop' aperta o parar da cabine")
    ap.add_argument("--log", default=str(Path.home() / f"er1_decisor_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"))
    a = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(ER1)
    modelo = AutoModelForImageTextToText.from_pretrained(ER1, dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"ER-1 carregado em {time.time() - t0:.0f} s", flush=True)

    est = Estado(a.objetivo)
    sobe_http(est, a.porta)
    cam = Camera(a.robo, 5555, a.camera)
    print(f"🧠 ER-1 decisor em http://0.0.0.0:{a.porta}/ | câmera {a.camera} | cabine {a.cabine}", flush=True)
    log = open(a.log, "a")
    atual = None
    while True:
        t_ini = time.time()
        if cam.rgb is None or time.time() - cam.t > 1.0:
            time.sleep(0.5)
            continue
        try:
            with urllib.request.urlopen(a.cabine + "/estado.json", timeout=1) as r:
                atual = json.loads(r.read()).get("tarefa") or atual
        except Exception:
            pass
        rgb = cam.rgb.copy()
        with est.lock:
            objetivo = est.objetivo
        texto = PROMPT.format(objetivo=objetivo, atual=atual or "none",
                              opcoes="\n".join(f'- "{i}"' for i in INSTRUCOES))
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(rgb)},
                                             {"type": "text", "text": texto}]}]
        entrada = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt").to(modelo.device)
        tq = time.perf_counter()
        with torch.inference_mode():
            saida = modelo.generate(**entrada, max_new_tokens=220, do_sample=False)
        ms = round((time.perf_counter() - tq) * 1000)
        cru = proc.decode(saida[0, entrada["input_ids"].shape[1]:], skip_special_tokens=True)
        d = extrai_json(cru) or {"instruction": None, "reason": "resposta sem JSON", "cru": cru[:300]}
        d["ms"] = ms
        inst = d.get("instruction")
        d["enviada"] = False
        if inst in INSTRUCOES and inst != "stop" and inst != atual:
            try:
                req = urllib.request.Request(a.cabine + "/tarefa", data=json.dumps({"texto": inst}).encode(),
                                             headers={"Content-Type": "application/json"}, method="POST")
                urllib.request.urlopen(req, timeout=2).read()
                d["enviada"] = True
                atual = inst
            except Exception as e:  # noqa: BLE001
                d["erro_cabine"] = str(e)
        elif inst == "stop" and a.pode_parar:
            try:
                urllib.request.urlopen(urllib.request.Request(a.cabine + "/parar", data=b"", method="POST"), timeout=2)
                d["enviada"] = "parar"
            except Exception as e:  # noqa: BLE001
                d["erro_cabine"] = str(e)
        with est.lock:
            est.decisao = d
            est.jpeg = desenha(rgb, d)
            est.historico.append(f"{time.strftime('%H:%M:%S')} {ms:5d} ms | {inst} | "
                                 f"{[(o.get('label'), o.get('point_2d')) for o in d.get('objects') or []]} | "
                                 f"{d.get('reason', '')}")
        log.write(json.dumps({"t": time.time(), "objetivo": objetivo, **d}, default=str) + "\n")
        log.flush()
        print(f"[ER-1] {ms} ms | {inst} | {d.get('scene', '')} | enviada={d['enviada']}", flush=True)
        time.sleep(max(0.0, a.periodo - (time.time() - t_ini)))


if __name__ == "__main__":
    main()
