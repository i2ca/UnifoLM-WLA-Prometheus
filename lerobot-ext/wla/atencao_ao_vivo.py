#!/usr/bin/env python
"""ATENÇÃO AO VIVO do WLA (03/11): dentro do servidor (servidor_wla_dex3.py --atencao 1), a cada consulta, para onde a
cabeça de ação (DiT) olha na saída do VLM — o mesmo espião do atencao_video_v7.py, resumido e servido em JSON para a
cabine do robô desenhar (GET http://<pgx>:8602/atencao.json):

  partes    parcela da atenção em cada câmera, na frase, no estado e no resto do prompt
  camadas   a mesma divisão por camada do DiT (média dos passos do fluxo)
  mapas     por câmera: a grade de tokens (linhas x colunas) em "vezes a média da câmera"
  palavras  [palavra, parcela] da frase que o MODELO recebeu (a descritiva do treino)

Custo: um softmax(q·k) por camada e passo do fluxo, só da 1ª amostra (~10-20 ms).
"""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "treino"))
PAPEIS = ["head_left", "head_depth", "cam_wrist_left", "cam_wrist_right"]
NOMES = ["cabeça", "profundidade", "punho E", "punho D", "frase", "estado", "resto"]


class AtencaoAoVivo:
    def __init__(self, modelo, porta=8602, passos_fluxo=4):
        from atencao_video_v7 import Espiao, segmentos
        self._segmentos = segmentos
        self.m, self.nfl = modelo, max(1, int(passos_fluxo))
        self.esp = Espiao()
        self.esp.instala()
        iface = modelo.qwen_vl_interface
        self.tok = iface.processor.tokenizer
        self.guard, self.ultima, self.trava = {}, {"pronto": False}, threading.Lock()
        self.prof = None   # (t, Z[::8, ::8] em metros, rgb[::8, ::8]) — o servidor preenche a cada consulta
        orig = iface.build_qwenvl_inputs

        def build(*x, **kw):
            out = orig(*x, **kw)
            self.guard["inp"] = out
            self.guard["roles"] = (kw.get("image_roles") or [None])[0]
            self.esp.L = out["input_ids"].shape[1]
            return out
        iface.build_qwenvl_inputs = build
        self._sobe(porta)

    def inicia(self):
        self.esp.chamadas, self.esp.ligado, self.t0 = [], True, time.time()

    def fecha(self, frase_cabine=""):
        self.esp.ligado = False
        try:
            self.ultima = self._resume(frase_cabine)
        except Exception as e:  # noqa: BLE001  (a atenção nunca derruba a ação)
            self.ultima = {"pronto": False, "erro": repr(e)[:200]}

    def _resume(self, frase_cabine):
        inp, roles = self.guard.get("inp"), self.guard.get("roles") or PAPEIS
        if inp is None or not self.esp.chamadas:
            return {"pronto": False, "erro": "sem chamadas do DiT"}
        C = np.stack(self.esp.chamadas)                                   # (chamadas, L)
        cams, frase, pos_est = self._segmentos(inp, self.tok, roles)
        n_cam = {r: C[:, p].sum(1) for r, (p, _) in cams.items()}
        pos_f = [i for ps, _ in frase for i in ps]
        n_fr = C[:, pos_f].sum(1) if pos_f else np.zeros(len(C))
        n_st = C[:, pos_est].sum(1) if len(pos_est) else np.zeros(len(C))
        cat = np.stack([n_cam.get(r, np.zeros(len(C))) for r in PAPEIS]
                       + [n_fr, n_st, 1 - sum(n_cam.values()) - n_fr - n_st], 1)
        por = max(1, len(cat) // self.nfl)
        camadas = cat[: por * self.nfl].reshape(self.nfl, por, 7).mean(0)
        media = C.mean(0)
        mapas = {}
        for r, (p, (gh, gw)) in cams.items():
            mp = media[p].reshape(gh, gw)
            mapas[r] = {"linhas": int(gh), "colunas": int(gw),
                        "valores": np.round(mp / (mp.mean() + 1e-12), 3).ravel().tolist()}
        tot_f = max(float(media[pos_f].sum()), 1e-12) if pos_f else 1.0
        palavras = [[w, round(float(media[ps].sum()) / tot_f, 4)] for ps, w in frase]
        tot = float(cat.mean(0).sum()) or 1.0
        return {"pronto": True, "t": time.time(), "ms": round((time.time() - self.t0) * 1000),
                "frase_modelo": " ".join(w for _, w in frase), "frase_cabine": frase_cabine,
                "partes": {n: round(float(v) / tot, 4) for n, v in zip(NOMES, cat.mean(0))},
                "camadas": np.round(camadas, 4).tolist(), "nomes": NOMES, "mapas": mapas, "palavras": palavras}

    def _nuvem(self):
        """A profundidade da cabeça reduzida: Z (m, None onde não há) e a cor, na grade de 8 em 8 pixels da 640x480."""
        if self.prof is None:
            return {"pronto": False}
        t, Z, rgb = self.prof
        Zl = [None if not np.isfinite(z) else round(float(z), 3) for z in Z.ravel()]
        return {"pronto": True, "t": t, "w": 640, "h": 480, "passo": 8, "linhas": int(Z.shape[0]),
                "colunas": int(Z.shape[1]), "Z": Zl, "cor": rgb.reshape(-1).astype(int).tolist()}

    def _sobe(self, porta):
        dono = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                if self.path.startswith("/profundidade.json"):
                    b = json.dumps(dono._nuvem()).encode()
                else:
                    b = json.dumps(dono.ultima, ensure_ascii=False).encode()
                self.send_response(200 if self.path.startswith(("/atencao.json", "/profundidade.json")) else 404)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

        srv = ThreadingHTTPServer(("0.0.0.0", porta), H)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True, name="atencao").start()
        print(f"👁  atenção ao vivo em http://0.0.0.0:{porta}/atencao.json", flush=True)
