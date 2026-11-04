#!/usr/bin/env python
"""VOZ NATURAL DO PROMETHEUS (30/10): Kokoro-82M (Apache-2.0) em português do Brasil, carregado uma vez e servido por
HTTP para a cabine (o Piper soava robótico). ~0,3 s na GPU para 10 s de fala.

    GET /fala?texto=...&voz=pm_alex&vel=1.0   -> audio/wav (24 kHz mono)
    GET /vozes                                -> as vozes pt-BR

Roda no venv próprio (kokoro_venv, com os pacotes do wla visíveis; não mexe no ambiente do ER-1/WLA):
    ~/wla_testes/tts/kokoro_venv/bin/python voz_kokoro.py --porta 8099
"""
import argparse
import io
import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VOZES = {"pm_alex": "masculina (Alex)", "pm_santa": "masculina (Santa)", "pf_dora": "feminina (Dora)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--porta", type=int, default=8099)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline
    t0 = time.time()
    pipe = KPipeline(lang_code="p", device=a.device)
    trava = threading.Lock()
    with trava:   # aquece (a 1ª síntese de cada voz é mais lenta)
        for v in VOZES:
            for _ in pipe("Olá.", voice=v):
                pass
    print(f"🗣️ Kokoro pronto em {time.time() - t0:.0f} s (vozes {list(VOZES)})", flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *_):
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
            if u.path == "/vozes":
                return self._manda(json.dumps(VOZES, ensure_ascii=False).encode(), "application/json")
            if u.path == "/fala" and qs.get("texto"):
                voz = qs.get("voz", ["pm_alex"])[0]
                voz = voz if voz in VOZES else "pm_alex"
                vel = float(qs.get("vel", ["1.0"])[0])
                t = time.perf_counter()
                with trava:
                    partes = [x.audio.numpy() for x in pipe(qs["texto"][0], voice=voz, speed=vel) if x.audio is not None]
                if not partes:
                    return self._manda(b"sem audio", "text/plain", 500)
                buf = io.BytesIO()
                sf.write(buf, np.concatenate(partes), 24000, format="WAV", subtype="PCM_16")
                print(f"[voz] {voz} {time.perf_counter() - t:.2f} s | {qs['texto'][0][:60]}", flush=True)
                return self._manda(buf.getvalue(), "audio/wav")
            self._manda(b"use /fala?texto=...", "text/plain", 404)

    ThreadingHTTPServer(("127.0.0.1", a.porta), H).serve_forever()


if __name__ == "__main__":
    main()
