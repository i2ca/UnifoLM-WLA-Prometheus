#!/usr/bin/env python
"""Servidor de ações do UnifoLM-WLA com mapa de atenção AO VIVO no navegador.

É o servidor oficial (model_server.action_server_wbc_msgpack_unitree), mesmo protocolo e
mesma ação, envolvido por fora — nada do repositório do WLA é editado. A cada get_action:
  - mede a atenção dos tokens de ação sobre os tokens do VLM (ver mapa_atencao.py);
  - desenha o mapa sobre as 3 câmeras numa thread separada (não atrasa a resposta);
  - publica numa página web: http://<pgx>:8080 (stream MJPEG + números por chunk).

    cd ~/DEV/unifolm-wla
    python ~/wla_testes/servidor_atencao.py --port 8601 --http_port 8080 --unnorm_key UnifoLM_G1_Dex1

Na ponte: --servidor ws://192.168.123.52:8601
"""
import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mapa_atencao as ma  # noqa: E402  (instala o espião em mmdit.dispatch_attention_fn)

srv_mod = ma.srv_mod
REG = ma.REG

PAGINA = """<!doctype html>
<html lang="pt-br"><head><meta charset="utf-8"><title>WLA atenção ao vivo</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 :root { --bg:#111418; --fg:#e8eaed; --mut:#9aa0a6; --ac:#4fc3f7; --card:#1b1f24; }
 body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, sans-serif; }
 header { padding:10px 16px; display:flex; gap:24px; align-items:baseline; flex-wrap:wrap; }
 h1 { font-size:16px; margin:0; }
 .mut { color:var(--mut); }
 main { padding:0 16px 16px; }
 img { width:100%; max-width:1400px; border-radius:6px; display:block; background:#000; }
 .cards { display:flex; gap:12px; flex-wrap:wrap; margin:12px 0; }
 .card { background:var(--card); border-radius:6px; padding:8px 12px; min-width:120px; }
 .card b { display:block; font-size:20px; }
 svg { background:var(--card); border-radius:6px; width:100%; max-width:1400px; height:140px; }
</style></head><body>
<header><h1>UnifoLM-WLA · atenção das ações sobre as câmeras</h1>
<span class="mut" id="inst"></span></header>
<main>
<img src="/stream" alt="mapa de atenção">
<div class="cards">
 <div class="card"><span class="mut">chunk</span><b id="n">–</b></div>
 <div class="card"><span class="mut">imagens</span><b id="img">–</b></div>
 <div class="card"><span class="mut">cabeça</span><b id="c0">–</b></div>
 <div class="card"><span class="mut">punho esq.</span><b id="c1">–</b></div>
 <div class="card"><span class="mut">punho dir.</span><b id="c2">–</b></div>
 <div class="card"><span class="mut">estado</span><b id="st">–</b></div>
 <div class="card"><span class="mut">inferência</span><b id="lat">–</b></div>
 <div class="card"><span class="mut">garra E / D</span><b id="gr">–</b></div>
</div>
<div class="mut">% da atenção nas imagens por chunk (azul = total; linhas finas = cabeça, punho esq., punho dir.)</div>
<svg id="g" viewBox="0 0 1000 140" preserveAspectRatio="none"></svg>
</main>
<script>
const cores = ["#4fc3f7", "#ffb74d", "#81c784", "#e57373"];
function linha(v, cor, larg, max) {
  if (v.length < 2) return "";
  const pts = v.map((y, i) => `${(i / (v.length - 1)) * 1000},${140 - (y / max) * 130 - 5}`).join(" ");
  return `<polyline fill="none" stroke="${cor}" stroke-width="${larg}" points="${pts}"/>`;
}
async function tick() {
  try {
    const s = await (await fetch("/stats")).json();
    const u = s.ultimo || {};
    const pct = x => x == null ? "–" : (100 * x).toFixed(1) + "%";
    document.getElementById("inst").textContent = u.instrucao ? "“" + u.instrucao + "”" : "";
    document.getElementById("n").textContent = s.n;
    document.getElementById("img").textContent = pct(u.imagens);
    ["c0", "c1", "c2"].forEach((id, i) => document.getElementById(id).textContent = pct((u.cameras || [])[i]));
    document.getElementById("st").textContent = pct(u.estado);
    document.getElementById("lat").textContent = u.inferencia_ms ? u.inferencia_ms.toFixed(0) + " ms" : "–";
    document.getElementById("gr").textContent = u.garra ? u.garra.map(x => x.toFixed(2)).join(" / ") : "–";
    const h = s.historico;
    const tot = h.map(x => 100 * x.imagens), max = Math.max(70, ...tot);
    let svg = linha(tot, cores[0], 3, max);
    for (let c = 0; c < 3; c++) svg += linha(h.map(x => 100 * x.cameras[c]), cores[c + 1], 1.5, max);
    document.getElementById("g").innerHTML = svg;
  } catch (e) {}
  setTimeout(tick, 500);
}
tick();
</script></body></html>
"""


class Estado:
    def __init__(self):
        self.jpeg = None
        self.versao = 0
        self.cond = threading.Condition()
        self.historico = []
        self.ultimo = {}


EST = Estado()


def servidor_http(porta):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/":
                corpo = PAGINA.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(corpo)))
                self.end_headers()
                self.wfile.write(corpo)
            elif self.path == "/stats":
                corpo = json.dumps({"n": len(EST.historico), "ultimo": EST.ultimo,
                                    "historico": EST.historico[-300:]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(corpo)
            elif self.path == "/frame.jpg" and EST.jpeg:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(EST.jpeg)
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=quadro")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                visto = -1
                try:
                    while True:
                        with EST.cond:
                            EST.cond.wait_for(lambda: EST.versao != visto and EST.jpeg is not None, timeout=10)
                            jpeg, visto = EST.jpeg, EST.versao
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--quadro\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    ThreadingHTTPServer(("0.0.0.0", porta), H).serve_forever()


class ServidorComAtencao(srv_mod.ActionServerWBCMsgpack):
    def __init__(self, args):
        super().__init__(args)
        iface = self.model.qwen_vl_interface
        self._img_tok = iface.model.config.image_token_id
        self._st_tok = getattr(iface.model.config, "robot_state_token_id", None)
        self._inp = None
        orig = iface.build_qwenvl_inputs

        def build(*x, **kw):
            out = orig(*x, **kw)
            self._inp = out
            REG.L = out["input_ids"].shape[1]
            return out

        iface.build_qwenvl_inputs = build
        self._fila = queue.Queue(maxsize=2)
        threading.Thread(target=self._desenhista, daemon=True).start()

    def _build_example(self, obs):
        prep = super()._build_example(obs)
        self._prep = prep
        return prep

    def get_action(self, obs):
        t0 = time.perf_counter()
        REG.soma, REG.n, REG.ligado = None, 0, True
        try:
            acao = super().get_action(obs)
        finally:
            REG.ligado = False
        ms = (time.perf_counter() - t0) * 1e3
        if REG.n:
            job = ((REG.soma / REG.n).float().cpu().numpy(), self._inp, self._prep["example"],
                   str(self._prep["example"]["lang"]), ms,
                   [float(acao["action.left_gripper"][0, 0, 0]), float(acao["action.right_gripper"][0, 0, 0])])
            try:
                self._fila.put_nowait(job)
            except queue.Full:
                pass  # o desenho ficou para trás: pula este chunk, a ação não espera
        return acao

    def _desenhista(self):
        while True:
            att, inp, example, instr, ms, garra = self._fila.get()
            try:
                n = len(EST.historico)
                fig, massas, m_estado, m_resto = ma.desenha(att, inp, example, self._img_tok, self._st_tok,
                                                            f"chunk {n} | '{instr}'")
                ok, buf = cv2.imencode(".jpg", fig, [cv2.IMWRITE_JPEG_QUALITY, 85])
                reg = {"imagens": float(sum(massas)), "cameras": [float(x) for x in massas],
                       "estado": m_estado, "resto": m_resto, "inferencia_ms": ms, "garra": garra,
                       "instrucao": instr, "t": time.time()}
                with EST.cond:
                    EST.jpeg = buf.tobytes()
                    EST.versao += 1
                    EST.historico.append(reg)
                    EST.ultimo = reg
                    EST.cond.notify_all()
            except Exception:
                logging.exception("falha ao desenhar o mapa de atenção")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", default="playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors")
    p.add_argument("--instruction", default="")
    p.add_argument("--unnorm_key", default="UnifoLM_G1_Dex1")
    p.add_argument("--use_bf16", action="store_true", default=True)
    p.add_argument("--image_size", type=int, nargs=2, default=[320, 448])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8601)
    p.add_argument("--http_port", type=int, default=8080)
    p.add_argument("--debug_save_dir", default=None)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    threading.Thread(target=servidor_http, args=(a.http_port,), daemon=True).start()
    logging.info("página de atenção em http://0.0.0.0:%d", a.http_port)
    ServidorComAtencao(a).run(a.host, a.port)


if __name__ == "__main__":
    main()
