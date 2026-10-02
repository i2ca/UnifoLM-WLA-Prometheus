#!/usr/bin/env python
"""
ER-1 PERGUNTA SIM/NÃO — o UnifoLM-ER-1 fica carregado e responde perguntas sobre a câmera AGORA
=============================================================================================
Para checar se uma tarefa foi CONCLUÍDA olhando a imagem (ex.: "Is the white cup under the coffee
strainer?"). Pega o quadro mais novo da câmera (cameras_wla_server, porta 5555), pergunta ao ER-1 e
devolve a resposta. Cada pergunta fica salva em ~/er1_perguntas/ (imagem + resposta).

    GET /pergunta?q=<pergunta em inglês>[&cam=head_stereo_left][&livre=1]
        -> {"resposta": "yes", "sim": true, "ms": 850, "imagem": ".../0003.jpg", "cru": "..."}
        (&max=N limita o tamanho da resposta livre; sem livre=1 a pergunta ganha "Answer only yes or no."; com livre=1 a resposta é texto livre)
    GET /traduz?t=<texto>   tradução para PT-BR (só texto)
    GET /ultima.jpg    a imagem da última pergunta

    cd ~/DEV/unifolm-wla && ~/miniforge3/envs/wla/bin/python \\
        ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/er1_pergunta.py          # porta 8098
    curl "http://127.0.0.1:8098/pergunta?q=Is%20the%20white%20cup%20under%20the%20coffee%20strainer%3F"

O ER-1 disputa a GPU com o WLA: pergunte de vez em quando (a cada poucos segundos), não a cada consulta.
"""
import argparse
import base64
import json
import re
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch
import zmq
from PIL import Image

ER1 = "playground/Pretrained_models/UnifoLM-ER-1"


class Cameras:
    """Último quadro de cada câmera da mensagem do cameras_wla_server (JPEG em BGR, como o OpenCV)."""

    def __init__(self, robo, porta):
        self.bgr, self.t = {}, {}
        s = zmq.Context.instance().socket(zmq.SUB)
        s.setsockopt_string(zmq.SUBSCRIBE, "")
        s.setsockopt(zmq.RCVHWM, 2)
        s.connect(f"tcp://{robo}:{porta}")
        threading.Thread(target=self._laco, args=(s,), daemon=True).start()

    def _laco(self, s):
        while True:
            try:
                m = json.loads(s.recv_multipart()[0])
                for nome, b in (m.get("images") or {}).items():
                    if isinstance(b, str):
                        self.bgr[nome] = cv2.imdecode(np.frombuffer(base64.b64decode(b), np.uint8), cv2.IMREAD_COLOR)
                        self.t[nome] = time.time()
            except Exception:
                time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robo", default="192.168.123.164")
    ap.add_argument("--porta", type=int, default=8098)
    ap.add_argument("--rgb", action="store_true", help="se as cores saírem trocadas (maçã azul), use isto")
    ap.add_argument("--pasta", default=str(Path.home() / "er1_perguntas"))
    a = ap.parse_args()
    pasta = Path(a.pasta)
    pasta.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(ER1)
    modelo = AutoModelForImageTextToText.from_pretrained(ER1, dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"ER-1 carregado em {time.time() - t0:.0f} s", flush=True)
    cams = Cameras(a.robo, 5555)
    trava = threading.Lock()
    estado = {"n": len(list(pasta.glob("*.json"))), "ultima": None}

    def pergunta(q, cam, livre, max_tokens=120):
        bgr, t = cams.bgr.get(cam), cams.t.get(cam, 0)
        if bgr is None or time.time() - t > 2.0:
            return {"erro": f"sem imagem recente da câmera '{cam}' (chegando: {sorted(cams.bgr)})"}
        bgr = bgr.copy()
        rgb = bgr if a.rgb else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        texto = q if livre else f"{q}\nLook carefully at the image. Answer only yes or no."
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(rgb)},
                                             {"type": "text", "text": texto}]}]
        with trava:
            entrada = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt").to(modelo.device)
            tq = time.perf_counter()
            with torch.inference_mode():
                saida = modelo.generate(**entrada, max_new_tokens=8 if not livre else max_tokens, do_sample=False)
            ms = round((time.perf_counter() - tq) * 1000)
            cru = proc.decode(saida[0, entrada["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            estado["n"] += 1
            n = estado["n"]
        m = re.search(r"\b(yes|no)\b", cru.lower())
        r = {"pergunta": q, "camera": cam, "resposta": m.group(1) if m else cru, "sim": (m.group(1) == "yes") if m else None,
             "ms": ms, "cru": cru, "hora": time.strftime("%H:%M:%S")}
        img = pasta / f"{n:04d}.jpg"
        cv2.imwrite(str(img), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        r["imagem"] = str(img)
        (pasta / f"{n:04d}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        estado["ultima"] = img
        print(f"[ER-1] {r['hora']} {ms} ms | {cam} | {q} -> {r['resposta']}", flush=True)
        return r

    def traduz(texto):
        """Só texto, sem imagem: traduz para português do Brasil (histórico de falas da cabine, 02/10)."""
        msgs = [{"role": "user", "content": [{"type": "text", "text":
                 "Translate to Brazilian Portuguese. Reply with only the translation, nothing else.\n\n" + texto}]}]
        with trava:
            entrada = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt").to(modelo.device)
            tq = time.perf_counter()
            with torch.inference_mode():
                saida = modelo.generate(**entrada, max_new_tokens=80, do_sample=False)
            ms = round((time.perf_counter() - tq) * 1000)
            pt = proc.decode(saida[0, entrada["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"[ER-1] tradução {ms} ms | {texto} -> {pt}", flush=True)
        return {"texto": texto, "pt": pt, "ms": ms}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def _manda(self, corpo, tipo, codigo=200):
            self.send_response(codigo)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(corpo)))
            self.end_headers()
            self.wfile.write(corpo)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(u.query)
            if u.path == "/pergunta" and qs.get("q"):
                r = pergunta(qs["q"][0], qs.get("cam", ["head_stereo_left"])[0], qs.get("livre", ["0"])[0] == "1",
                             int(qs.get("max", ["120"])[0]))
                return self._manda(json.dumps(r, ensure_ascii=False).encode(), "application/json")
            if u.path == "/traduz" and qs.get("t"):
                return self._manda(json.dumps(traduz(qs["t"][0]), ensure_ascii=False).encode(), "application/json")
            if u.path == "/ultima.jpg" and estado["ultima"]:
                return self._manda(Path(estado["ultima"]).read_bytes(), "image/jpeg")
            self._manda(b"use /pergunta?q=...", "text/plain", 404)

    srv = ThreadingHTTPServer(("0.0.0.0", a.porta), H)
    print(f"🧠 ER-1 pergunta sim/não em http://0.0.0.0:{a.porta}/pergunta?q=... | imagens em {pasta}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
