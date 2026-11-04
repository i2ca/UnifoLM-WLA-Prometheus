#!/usr/bin/env python
"""Cabine — o painel web do Prometheus, com várias pessoas olhando ao mesmo tempo.

É só a CASCA: guarda o último quadro de cada câmera, o último estado do robô e a
tarefa pedida, e serve tudo por HTTP. Quem enche isso é o laço de controle
(`cerebro.py`), e quem pede tarefa é o navegador ou a super IA pelo MCP
(`mcp_prometheus.py`).

    python -m pontes.cabine.servidor --porta 8090       # sozinho, com quadros de teste

── Por que MJPEG e não WebRTC/WebSocket ────────────────────────────────────
MJPEG é um `<img src="...">` e funciona em qualquer navegador, inclusive no do
headset e no do celular, sem negociação, sem TLS e sem biblioteca. O custo é
banda; numa LAN de laboratório isso não é problema. O painel do FastWAM-D
(`viz_debug_fastwamd.py`) já provou o caminho — esta é a mesma ideia, mas com
uma câmera por stream em vez de um mosaico, para o MCP conseguir pedir UMA
câmera específica.

── Várias conexões ao mesmo tempo ──────────────────────────────────────────
`ThreadingHTTPServer`: cada cliente ganha uma thread. Os quadros são
compartilhados sob um `Condition`, então N espectadores NÃO custam N
codificações de JPEG — a codificação acontece uma vez, quando o laço publica.
Isso é o que permite deixar o painel aberto no notebook, no celular e na TV da
sala sem afogar o laço de controle.

── Sem autenticação, de propósito ──────────────────────────────────────────
Ferramenta de bancada em rede fechada, igual ao painel do FastWAM-D. Se um dia
isto for exposto para fora do laboratório, o lugar de pôr token é o
`_autorizado()` — está marcado lá embaixo.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


LIMITE = b"--quadro"
ESTATICO = Path(__file__).resolve().parent / "static"
ATENCAO_PORTA = 8602   # servidor_wla_dex3.py --atencao 1
MAPA_PORTA = 8603      # 03/11: wla/mapa_ocupacao.py (o lidar da cabeça)
ROBO = Path(__file__).resolve().parents[2] / "assets" / "g1"   # URDF + malhas do G1 (o mesmo do fk_g1.py)


class Cabine:
    """Estado compartilhado entre o laço de controle e todo mundo que olha.

    O laço chama `publica_quadro` / `publica_estado` e segue em frente; nada
    aqui bloqueia o laço. A codificação JPEG acontece no `publica_quadro`, na
    thread do laço — é ~2 ms para 848x480 e evita manter uma thread de render
    só para isso. Se um dia virar gargalo, é aqui que se move para outra thread.
    """

    def __init__(self, qualidade: int = 80):
        self.qualidade = int(qualidade)
        self._trava = threading.Lock()
        self._novo = threading.Condition(self._trava)
        self._jpeg: dict[str, bytes] = {}
        self._quando: dict[str, float] = {}
        self._seq = 0
        self._estado: dict[str, Any] = {}
        self._tarefa = ""
        self.atalhos: list[tuple[str, str]] = []   # (rótulo, frase) -> botões na página (02/10)
        self._tarefa_seq = 0
        self._parada = False
        self._pedido_copo = False
        self._pedido_inicial = False
        self._objetos_pedidos = []
        self._clientes = 0
        # HISTÓRICO DO QUE O ROBÔ FALOU (02/10) e quem traduz para PT-BR (texto -> texto; o executor liga no ER-1)
        self._falas: list[dict[str, Any]] = []
        self.tradutor = None
        # 03/11: blocos extras para a página desenhar (o executor: "plano" a cada trecho, "vivo" a ~5 Hz)
        self._extras: dict[str, Any] = {}

    # ── quem escreve: o laço de controle ────────────────────────────────
    def publica_quadro(self, camera: str, rgb: np.ndarray) -> None:
        if cv2 is None:
            raise RuntimeError("a cabine precisa de opencv para codificar JPEG")
        ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1],
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.qualidade])
        if not ok:
            return
        with self._novo:
            self._jpeg[camera] = buf.tobytes()
            self._quando[camera] = time.time()
            self._seq += 1
            self._novo.notify_all()

    def publica_estado(self, estado: dict[str, Any]) -> None:
        with self._trava:
            self._estado = dict(estado)
            self._estado["quando"] = time.time()

    def publica_extra(self, nome: str, dados: dict[str, Any]) -> None:
        """Um bloco à parte do estado (não é apagado pelo publica_estado): plano, vivo..."""
        with self._trava:
            self._extras[nome] = dados

    def registra_fala(self, texto: str, origem: str = "") -> None:
        with self._trava:
            i = len(self._falas)
            self._falas.append({"i": i, "hora": time.strftime("%H:%M:%S"), "texto": texto,
                                "origem": origem, "pt": None})
        if self.tradutor is not None:   # tradução AUTOMÁTICA, numa thread: não segura quem falou (02/10)
            threading.Thread(target=self._traduz_quieto, args=(i,), daemon=True).start()

    def _traduz_quieto(self, i: int) -> None:
        try:
            self.traduz_fala(i)
        except Exception:  # noqa: BLE001
            pass                        # fica o botão PT-BR para tentar de novo

    def traduz_fala(self, i: int) -> str | None:
        """Traduz a fala i para PT-BR (uma vez; fica guardada). Bloqueia quem pediu, não o laço."""
        with self._trava:
            if not (0 <= i < len(self._falas)):
                return None
            f = self._falas[i]
            if f["pt"] or self.tradutor is None:
                return f["pt"]
            texto = f["texto"]
        pt = self.tradutor(texto)
        with self._trava:
            f["pt"] = pt
        return pt

    # ── quem lê: o laço de controle ─────────────────────────────────────
    def tarefa(self) -> tuple[str, int]:
        """Texto pedido e um contador que muda a cada pedido NOVO.

        O contador existe porque pedir duas vezes a mesma frase é um pedido
        legítimo ("de novo"), e comparar só o texto perderia o segundo.
        """
        with self._trava:
            return self._tarefa, self._tarefa_seq

    def parada_pedida(self) -> bool:
        with self._trava:
            return self._parada

    # ── quem escreve: o navegador e o MCP ───────────────────────────────
    def define_tarefa(self, texto: str) -> int:
        texto = (texto or "").strip()
        with self._trava:
            self._tarefa = texto
            self._tarefa_seq += 1
            # Uma tarefa nova limpa a parada anterior, senão o pedido de parar
            # de ontem mataria o movimento de hoje no primeiro passo.
            self._parada = False
            return self._tarefa_seq

    def pede_parada(self) -> None:
        with self._trava:
            self._parada = True

    def pede_objeto(self, nome: str, x: float, y: float) -> None:
        """Posição pedida para UM objeto, em metros no mundo."""
        with self._trava:
            self._objetos_pedidos.append((nome, float(x), float(y)))

    def consome_objetos(self) -> list:
        with self._trava:
            v, self._objetos_pedidos = self._objetos_pedidos, []
            return v

    def pede_inicial(self) -> None:
        with self._trava:
            self._pedido_inicial = True

    def consome_inicial(self) -> bool:
        """Verdadeiro UMA vez por clique no botão 'posição inicial' (o executor volta à pose de partida)."""
        with self._trava:
            v, self._pedido_inicial = self._pedido_inicial, False
            return v

    def pede_copo(self) -> None:
        with self._trava:
            self._pedido_copo = True

    def consome_pedido_copo(self) -> bool:
        """Verdadeiro UMA vez por pedido: quem consome é o laço de controle."""
        with self._trava:
            v, self._pedido_copo = self._pedido_copo, False
            return v

    # ── leitura interna do servidor ─────────────────────────────────────
    def _instantaneo(self, camera: str) -> bytes | None:
        with self._trava:
            return self._jpeg.get(camera)

    def _resumo(self) -> dict[str, Any]:
        with self._trava:
            agora = time.time()
            return {
                **self._estado,
                "cameras": {c: round(agora - t, 2) for c, t in self._quando.items()},
                "tarefa": self._tarefa,
                "tarefa_seq": self._tarefa_seq,
                "atalhos": [{"rotulo": r, "frase": f} for r, f in self.atalhos],
                "parada_pedida": self._parada,
                "falas": self._falas[-60:],
                "tradutor": self.tradutor is not None,
                "fluxos_abertos": self._clientes,
                # 27/10: o executor com o ER-1 checando "terminou?" (TERMINOU=1) avisa aqui a última tarefa concluída;
                # a cabine do ER-1 (orquestrador) lê daqui em vez de disputar a GPU com o WLA
                "terminou_ativo": getattr(self, "terminou_ativo", False),
                "concluida": getattr(self, "concluida", None),
                **getattr(self, "_extras", {}),
            }


def _autorizado(_handler) -> bool:
    """Ponto único para exigir token, se um dia isto sair da LAN. Hoje: aberto."""
    return True


class _Handler(BaseHTTPRequestHandler):
    cabine: Cabine = None            # preenchido pelo `sobe`
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):       # silencia o log por requisição
        pass

    # ── GET ─────────────────────────────────────────────────────────────
    def do_GET(self):
        if not _autorizado(self):
            return self._erro(401, "sem autorização")
        caminho = self.path.split("?")[0]
        if caminho in ("/", "/index.html"):
            return self._texto(PAGINA, "text/html; charset=utf-8")
        if caminho == "/estado.json":
            return self._texto(json.dumps(self.cabine._resumo()), "application/json")
        if caminho.startswith("/quadro/") and caminho.endswith(".jpg"):
            nome = caminho[len("/quadro/"):-len(".jpg")]
            dado = self.cabine._instantaneo(nome)
            if dado is None:
                return self._erro(404, f"sem quadro da câmera {nome!r}")
            return self._binario(dado, "image/jpeg")
        if caminho.startswith("/camera/") and caminho.endswith(".mjpg"):
            return self._stream(caminho[len("/camera/"):-len(".mjpg")])
        if caminho in ("/atencao.json", "/profundidade.json", "/mapa.json", "/vivo_mapa.json"):   # 03/11: repassa
            try:
                import urllib.request
                porta = MAPA_PORTA if caminho in ("/mapa.json", "/vivo_mapa.json") else ATENCAO_PORTA
                rota = "/vivo.json" if caminho == "/vivo_mapa.json" else caminho
                b = urllib.request.urlopen(f"http://127.0.0.1:{porta}{rota}", timeout=1.0).read()
                return self._binario(b, "application/json")
            except Exception:  # noqa: BLE001
                return self._texto(json.dumps({"pronto": False, "erro": "servidor do WLA sem --atencao"}),
                                   "application/json")
        if caminho.startswith("/robo/"):     # 03/11: o URDF e as malhas do G1 para o mundo 3D
            rel = Path(caminho[len("/robo/"):])
            arq = (ROBO / rel).resolve()
            if ROBO.resolve() not in arq.parents or not arq.is_file():
                return self._erro(404, "não existe")
            tipo = "application/xml" if arq.suffix == ".urdf" else "model/stl"
            return self._binario(arq.read_bytes(), tipo, cache=86400)
        if caminho.startswith("/static/"):   # 03/11: three.js local (a rede do robô pode não ter internet)
            nome = Path(caminho[len("/static/"):]).name
            arq = ESTATICO / nome
            if not arq.is_file():
                return self._erro(404, "não existe")
            tipo = "text/javascript" if nome.endswith(".js") else "application/octet-stream"
            return self._binario(arq.read_bytes(), tipo, cache=0 if nome == "mapa3d.js" else 86400)   # o nosso muda
        return self._erro(404, "não existe")

    # ── POST ────────────────────────────────────────────────────────────
    def do_POST(self):
        if not _autorizado(self):
            return self._erro(401, "sem autorização")
        caminho = self.path.split("?")[0]
        if caminho == "/parar":
            self.cabine.pede_parada()
            return self._texto(json.dumps({"ok": True}), "application/json")
        if caminho == "/inicial":
            self.cabine.pede_inicial()
            return self._texto(json.dumps({"ok": True}), "application/json")
        if caminho == "/traduz":
            n = int(self.headers.get("Content-Length", 0))
            c = json.loads(self.rfile.read(n) or b"{}")
            try:
                pt = self.cabine.traduz_fala(int(c.get("i", -1)))
            except Exception as e:  # noqa: BLE001
                return self._erro(502, f"tradução falhou: {e}")
            return self._texto(json.dumps({"ok": pt is not None, "pt": pt}), "application/json")
        if caminho == "/copo":
            self.cabine.pede_copo()
            return self._texto(json.dumps({"ok": True}), "application/json")
        if caminho == "/objeto":
            n = int(self.headers.get("Content-Length", 0))
            c = json.loads(self.rfile.read(n) or b"{}")
            self.cabine.pede_objeto(c.get("nome", ""), c.get("x", 0), c.get("y", 0))
            return self._texto(json.dumps({"ok": True}), "application/json")
        if caminho != "/tarefa":
            return self._erro(404, "não existe")
        n = int(self.headers.get("Content-Length", 0))
        try:
            corpo = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._erro(400, "corpo não é JSON")
        seq = self.cabine.define_tarefa(corpo.get("texto", ""))
        return self._texto(json.dumps({"ok": True, "tarefa_seq": seq}), "application/json")

    # ── MJPEG ───────────────────────────────────────────────────────────
    def _stream(self, camera: str):
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={LIMITE.decode()[2:]}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with self.cabine._trava:
            self.cabine._clientes += 1
        visto = -1
        try:
            while True:
                with self.cabine._novo:
                    # O timeout não é decoração: sem ele, um cliente preso numa
                    # câmera que parou de publicar nunca acorda, e a thread dele
                    # fica pendurada até o processo morrer.
                    self.cabine._novo.wait_for(lambda: self.cabine._seq != visto, timeout=5.0)
                    visto = self.cabine._seq
                    dado = self.cabine._jpeg.get(camera)
                if dado is None:
                    continue
                self.wfile.write(LIMITE + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(dado)}\r\n\r\n".encode())
                self.wfile.write(dado)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass                     # espectador fechou a aba; normal
        finally:
            with self.cabine._trava:
                self.cabine._clientes -= 1

    # ── respostas ───────────────────────────────────────────────────────
    def _texto(self, corpo: str, tipo: str):
        self._binario(corpo.encode("utf-8"), tipo)

    def _binario(self, corpo: bytes, tipo: str, cache: int = 0):
        self.send_response(200)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(corpo)))
        self.send_header("Cache-Control", f"max-age={cache}" if cache else "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(corpo)

    def _erro(self, codigo: int, msg: str):
        corpo = json.dumps({"erro": msg}).encode()
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)


def sobe(cabine: Cabine, porta: int = 8090, host: str = "0.0.0.0") -> ThreadingHTTPServer:
    """Sobe o servidor numa thread e devolve. Não bloqueia."""
    _Handler.cabine = cabine
    httpd = ThreadingHTTPServer((host, porta), _Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True, name="cabine").start()
    return httpd


PAGINA = """<!doctype html>
<html lang="pt-BR">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prometheus — cabine do robô</title>
<style>
  :root { color-scheme: dark;
    --fundo:#0c0e12; --cartao:#151922; --cartao2:#1b2030; --borda:#262c3a; --texto:#e6e9ef; --fraco:#8b93a7;
    --acento:#4f9dff; --ok:#34c38f; --aviso:#e8a33d; --erro:#ff5d5d; --plano:#3ee6ff; --exec:#7CFF6B; --esq:#7a8cff; }
  * { box-sizing:border-box }
  body { margin:0; background:var(--fundo); color:var(--texto); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
  header { position:sticky; top:0; z-index:5; display:flex; flex-wrap:wrap; gap:10px; align-items:center;
           padding:10px 16px; background:#0f1218ee; backdrop-filter:blur(6px); border-bottom:1px solid var(--borda); }
  header h1 { font-size:16px; margin:0 6px 0 0; font-weight:650; white-space:nowrap }
  .chip { display:inline-flex; align-items:center; gap:6px; padding:3px 10px; border-radius:999px; font-size:12.5px;
          background:var(--cartao2); border:1px solid var(--borda); color:var(--fraco); white-space:nowrap }
  .chip i { width:8px; height:8px; border-radius:50%; background:var(--fraco) }
  .chip.on i { background:var(--ok) } .chip.off i { background:var(--erro) } .chip.warn i { background:var(--aviso) }
  .chip.on { color:var(--texto) }
  /* 03/11: a tarefa numa LINHA SÓ DELA embaixo dos chips (antes o campo ficava espremido entre a lista e os botões) */
  form { display:flex; flex-wrap:wrap; gap:8px; flex:1 1 100%; min-width:0; order:10 }
  form select { flex:0 1 280px; min-width:200px }
  input[type=text] { flex:1 1 380px; min-width:240px; background:#0a0c10; color:var(--texto); border:1px solid var(--borda);
          padding:8px 11px; font:inherit; border-radius:9px }
  button { font:inherit; padding:8px 13px; border-radius:9px; border:1px solid var(--borda); background:var(--cartao2);
           color:var(--texto); cursor:pointer; white-space:nowrap } button:hover { filter:brightness(1.15) }
  .prim { background:var(--acento); border-color:transparent } .parar { background:#5a2228; border-color:#8a3c3c; font-weight:650 }
  .volta { background:#22314f; border-color:#3c5a8a } #atalhos { display:flex; gap:6px; flex-wrap:wrap }
  main { display:grid; grid-template-columns:minmax(0,1.35fr) minmax(0,1fr); gap:14px; padding:14px 16px }
  @media (max-width:1100px) { main { grid-template-columns:1fr } }
  .col { display:flex; flex-direction:column; gap:14px; min-width:0 }
  .cartao { background:var(--cartao); border:1px solid var(--borda); border-radius:12px; padding:12px; min-width:0 }
  .cartao h2 { margin:0 0 10px; font-size:12.5px; letter-spacing:.06em; text-transform:uppercase; color:var(--fraco);
               font-weight:600; display:flex; align-items:center; gap:10px }
  .cartao h2 .sp { flex:1 } .cartao h2 label { text-transform:none; letter-spacing:0; font-weight:500; color:var(--fraco);
               display:inline-flex; gap:5px; align-items:center; cursor:pointer }
  #camwrap { position:relative; width:100%; aspect-ratio:4/3; background:#000; border-radius:9px; overflow:hidden }
  #camwrap img, #camwrap canvas { position:absolute; inset:0; width:100%; height:100% }
  #camwrap img { object-fit:contain }
  #camvazia { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:var(--fraco) }
  .legenda { display:flex; flex-wrap:wrap; gap:14px; margin-top:8px; font-size:12px; color:var(--fraco) }
  .legenda span::before { content:""; display:inline-block; width:18px; height:3px; margin-right:6px; vertical-align:middle;
                          background:var(--c) }
  #mundo { width:100%; aspect-ratio:4/3; border-radius:9px; overflow:hidden; background:#0a0d14 }
  .barra { display:flex; height:18px; border-radius:6px; overflow:hidden; background:#0a0c10; border:1px solid var(--borda) }
  .barra div { height:100% } .partes { display:flex; flex-wrap:wrap; gap:10px; margin:8px 0 10px; font-size:12px; color:var(--fraco) }
  .partes b { color:var(--texto) } .partes span::before { content:"■ "; color:var(--c) }
  #palavras { display:flex; flex-wrap:wrap; gap:5px; margin-bottom:10px }
  #palavras span { padding:2px 7px; border-radius:6px; font-size:13px; border:1px solid var(--borda) }
  .dupla { display:grid; grid-template-columns:1fr 1fr; gap:10px }
  canvas.grafico { width:100%; height:120px; background:#0a0c10; border:1px solid var(--borda); border-radius:8px }
  .temps { display:flex; flex-wrap:wrap; gap:6px }
  .t { font-size:12px; color:var(--fraco) }
  #thumbs { display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:8px }
  #thumbs figure { margin:0 } #thumbs img { width:100%; border-radius:7px; display:block; background:#000 }
  #thumbs figcaption { font-size:11.5px; color:var(--fraco); padding-top:3px }
  #falas-lista { max-height:190px; overflow-y:auto; font-size:13px }
  #falas-lista div { padding:4px 0; border-bottom:1px solid #1e2330 }
  details summary { cursor:pointer; color:var(--fraco) }
  table { width:100%; border-collapse:collapse; font-size:12.5px } td { padding:3px 6px; border-bottom:1px solid #1e2330 }
  td:first-child { color:var(--fraco); width:42% }
</style>
<header>
  <h1>🤖 Prometheus · cabine</h1>
  <span class="chip" id="c_cam"><i></i><span>câmera</span></span>
  <span class="chip" id="c_wla"><i></i><span>WLA</span></span>
  <span class="chip" id="c_atn"><i></i><span>atenção</span></span>
  <span class="chip" id="c_tarefa"><span>sem tarefa</span></span>
  <form onsubmit="manda(event)">
    <select id="tarefas" title="tarefas que o WLA sabe" style="background:#0a0c10;color:var(--texto);border:1px solid var(--borda);border-radius:9px;padding:7px 8px;font:inherit">
      <option value="">— escolha uma tarefa —</option>
      <option value="Pick up the white mug.">pegar a caneca branca</option>
      <option value="Place the white mug under the coffee strainer.">colocar a caneca embaixo do coador</option>
      <option value="Pick up the apple.">pegar a maçã</option>
      <option value="Pick up the apple and place it on the black X.">pegar a maçã e colocar no X</option>
    </select>
    <input type="text" id="t" placeholder="o que o robô deve fazer (ex.: Pick up the white mug.)" autocomplete="off">
    <button class="prim">▶ mandar</button>
    <button type="button" class="parar" onclick="fetch('parar',{method:'POST'})" title="mantém a posição e espera outra tarefa">⏸ parar</button>
    <button type="button" class="volta" onclick="fetch('inicial',{method:'POST'})" title="volta devagar à posição de partida">↩ inicial</button>
  </form>
  <span id="atalhos"></span>
  <button type="button" id="copo" style="display:none" onclick="fetch('copo',{method:'POST'})">mudar xícara (ç)</button>
</header>

<main>
  <div class="col">
    <section class="cartao">
      <h2>Câmera da cabeça — o que o WLA planeja <span class="sp"></span>
        <label><input type="checkbox" id="v_plano" checked> plano</label>
        <label><input type="checkbox" id="v_atn" checked> atenção</label>
        <label><input type="checkbox" id="v_esq"> mão esq.</label></h2>
      <div id="camwrap"><div id="camvazia">aguardando a câmera…</div><img id="cam" alt=""><canvas id="sobre"></canvas></div>
      <div class="legenda">
        <span style="--c:var(--plano)">plano da mão direita (1 s)</span>
        <span style="--c:var(--exec)">já executado</span>
        <span style="--c:#fff">mão agora</span>
        <span style="--c:var(--erro)">ponto onde fecha</span>
        <span style="--c:linear-gradient(90deg,#1b3cff,#20e0ff,#ffe020,#ff3020)">atenção do DiT (× média)</span>
      </div>
    </section>
    <section class="cartao">
      <h2>Mãos e cintura no tempo <span class="sp"></span><span class="t" id="trecho"></span></h2>
      <div class="dupla">
        <div><div class="t">dedos da mão direita — plano (linha) e medido (ponto)</div><canvas class="grafico" id="g_dedos"></canvas></div>
        <div><div class="t">giro da cintura (yaw) — plano, alvo e medido</div><canvas class="grafico" id="g_yaw"></canvas></div>
      </div>
    </section>
    <section class="cartao">
      <h2>Mapa de atenção por câmera <span class="sp"></span><span class="t">vermelho = onde a ação mais consulta</span></h2>
      <div id="mapas_atn" style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px"></div>
    </section>
    <section class="cartao">
      <h2>Outras câmeras</h2>
      <div id="thumbs"></div>
    </section>
  </div>

  <div class="col">
    <section class="cartao">
      <h2>Mundo 3D — mãos, cintura e a cena <span class="sp"></span>
        <label><input type="checkbox" id="v_nuvem" checked> profundidade (nuvem)</label>
        <label><input type="checkbox" id="v_mapa" checked> mapa do lidar</label>
        <span class="t">arraste p/ girar · roda p/ zoom</span></h2>
      <div id="mundo"></div>
      <div class="legenda">
        <span style="--c:var(--plano)">plano mão dir.</span><span style="--c:var(--esq)">plano mão esq.</span>
        <span style="--c:#9aa3b8">rastro</span><span style="--c:var(--aviso)">cintura medida</span><span style="--c:#ff7bd5">cintura pedida</span>
        <span style="--c:#37d39b">mesa (lidar)</span><span style="--c:#c08bff">tampo deduzido</span>
      </div>
      <div class="legenda" id="mapa_info" style="opacity:.8"></div>
    </section>
    <section class="cartao">
      <h2>Atenção dos tokens (ao vivo) <span class="sp"></span><span class="t" id="atn_meta"></span></h2>
      <div class="barra" id="atn_barra"></div>
      <div class="partes" id="atn_partes"></div>
      <div class="t" style="margin-bottom:4px">palavras da frase que o modelo recebeu (mais forte = mais atenção)</div>
      <div id="palavras"><span class="t">—</span></div>
      <div class="t" style="margin-bottom:4px">por camada do DiT (linhas) × parte da entrada (colunas)</div>
      <canvas class="grafico" id="g_camadas" style="height:150px"></canvas>
    </section>
    <section class="cartao">
      <h2>Temperaturas</h2>
      <div class="temps" id="temps"><span class="t">—</span></div>
    </section>
    <section class="cartao">
      <h2>O que o robô falou <span class="sp"></span><button type="button" id="traduz-tudo" onclick="traduzTudo()" style="padding:3px 10px;font-size:12px">🇧🇷 traduzir tudo</button></h2>
      <div id="falas-lista" class="t">nada falado ainda</div>
    </section>
    <section class="cartao">
      <details><summary>Detalhes técnicos da última consulta</summary><table id="detalhes"></table></details>
    </section>
    <section class="cartao" id="sim" style="display:none">
      <h2>Simulação</h2>
      <section id="objetos"></section>
      <canvas id="pred" width="1200" height="300" style="width:100%;margin-top:8px;background:#0a0c10;border-radius:8px"></canvas>
    </section>
  </div>
</main>

<script type="importmap">{ "imports": { "three": "./static/three.module.min.js",
  "three/examples/jsm/loaders/STLLoader.js": "./static/STLLoader.js",
  "three/examples/jsm/loaders/ColladaLoader.js": "./static/ColladaLoader.js" } }</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from './static/OrbitControls.js';
import URDFLoader from './static/URDFLoader.js';
import { CamadaMapa, resumoMapa } from './static/mapa3d.js';

const $ = id => document.getElementById(id);
let E = {}, ATN = null;
const CAM_PRINCIPAL = ['head_stereo_left', 'head_camera', 'head_left'];

// ── tarefa e botões ────────────────────────────────────────────────
$('tarefas').onchange = e => { if (e.target.value) { $('t').value = e.target.value; $('t').focus(); } };
window.manda = async ev => {
  ev.preventDefault();
  await fetch('tarefa', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({texto: $('t').value})});
  $('t').select();
};
let atalhosVistos = '';
function atalhos(e) {
  const lista = e.atalhos || [], chave = JSON.stringify(lista);
  if (chave === atalhosVistos) return; atalhosVistos = chave;
  const el = $('atalhos'); el.innerHTML = '';
  for (const a of lista) {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = a.rotulo; b.title = a.frase;
    b.onclick = () => fetch('tarefa', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({texto: a.frase})});
    el.appendChild(b);
  }
}
function chip(id, estado, texto) { const c = $(id); c.className = 'chip ' + estado; c.querySelector('span').textContent = texto; }

// ── câmera principal + sobreposição ────────────────────────────────
const img = $('cam'), sobre = $('sobre'), gctx = sobre.getContext('2d');
let camNome = null;
function liga() { if (camNome) img.src = 'quadro/' + camNome + '.jpg?t=' + Date.now(); }
img.onload = () => { $('camvazia').style.display = 'none'; desenhaSobre(); setTimeout(liga, 66); };
img.onerror = () => setTimeout(liga, 800);
function cor_calor(v) {   // 0..1 -> azul, ciano, amarelo, vermelho
  const s = [[27,60,255],[32,224,255],[255,224,32],[255,48,32]], x = Math.min(0.999, Math.max(0, v)) * 3, i = Math.floor(x), f = x - i;
  return s[i].map((c, k) => Math.round(c + (s[i + 1][k] - c) * f));
}
function calorCanvas(m) {   // grade da atenção -> canvas (gh x gw) com o mapa de cores e transparência
  const c = document.createElement('canvas'); c.width = m.colunas; c.height = m.linhas;
  const g = c.getContext('2d'), d = g.createImageData(m.colunas, m.linhas);
  for (let k = 0; k < m.valores.length; k++) {
    const v = Math.min(1, Math.max(0, (m.valores[k] - 1.3) / 2.7)), [R, G, B] = cor_calor(v);   // 1,3x..4x a média
    d.data.set([R, G, B, m.valores[k] < 1.3 ? 0 : Math.round(255 * (0.25 + 0.55 * v))], 4 * k);
  }
  g.putImageData(d, 0, 0); return c;
}
function desenhaSobre() {
  const r = sobre.getBoundingClientRect();
  if (sobre.width !== r.width * devicePixelRatio) { sobre.width = r.width * devicePixelRatio; sobre.height = r.height * devicePixelRatio; }
  const g = gctx, W = sobre.width, H = sobre.height;
  g.clearRect(0, 0, W, H);
  if (!img.naturalWidth) return;
  // a imagem fica "contain" na caixa 4:3
  const ar = img.naturalWidth / img.naturalHeight, w = Math.min(W, H * ar), h = w / ar, ox = (W - w) / 2, oy = (H - h) / 2;
  // atenção (head_left) — grade de tokens, em vezes a média da câmera (0..4)
  if ($('v_atn').checked && ATN && ATN.pronto && ATN.mapas && ATN.mapas.head_left) {
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high'; g.globalAlpha = 0.85;
    g.drawImage(calorCanvas(ATN.mapas.head_left), ox, oy, w, h); g.globalAlpha = 1;
  }
  const p = E.plano, vivo = E.vivo; if (!$('v_plano').checked || !p || !p.maos) return;
  const sx = w / (p.img?.w || 640), sy = h / (p.img?.h || 480), P = (u) => [ox + u[0] * sx, oy + u[1] * sy];
  const cursor = vivo?.passo ?? 0;
  const traco = (pts, de, ate, cor, larg, sombra = true) => {
    const q = pts.slice(de, ate).filter(Boolean); if (q.length < 2) return;
    if (sombra) { g.strokeStyle = 'rgba(0,0,0,.65)'; g.lineWidth = larg + 4 * devicePixelRatio; g.lineCap = g.lineJoin = 'round';
      g.beginPath(); q.forEach((u, k) => { const [x, y] = P(u); k ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke(); }
    g.strokeStyle = cor; g.lineWidth = larg; g.beginPath(); q.forEach((u, k) => { const [x, y] = P(u); k ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
  };
  for (const lado of ($('v_esq').checked ? ['left', 'right'] : ['right'])) {
    const m = p.maos[lado]; if (!m || !m.px) continue;
    const cor = lado === 'right' ? '#3ee6ff' : '#7a8cff', L = 5 * devicePixelRatio;
    traco(m.px, 0, m.px.length, cor, L);
    traco(m.px, 0, Math.min(cursor + 1, m.px.length), '#7CFF6B', L + 1.5 * devicePixelRatio, false);
    // pontos a cada 5 passos e o primeiro passo em que pede para fechar
    m.px.forEach((u, k) => { if (!u || k % 5) return; const [x, y] = P(u); g.fillStyle = cor; g.beginPath(); g.arc(x, y, 3 * devicePixelRatio, 0, 7); g.fill(); });
    for (let k = 9; k < m.px.length; k += 10) {   // setas: para onde a mão vai
      const a0 = m.px[k - 3], a1 = m.px[k]; if (!a0 || !a1) continue;
      const [x0, y0] = P(a0), [x1, y1] = P(a1), ang = Math.atan2(y1 - y0, x1 - x0), s = 10 * devicePixelRatio;
      if (Math.hypot(x1 - x0, y1 - y0) < 2) continue;
      g.fillStyle = cor; g.beginPath(); g.moveTo(x1, y1); g.lineTo(x1 - s * Math.cos(ang - .45), y1 - s * Math.sin(ang - .45));
      g.lineTo(x1 - s * Math.cos(ang + .45), y1 - s * Math.sin(ang + .45)); g.closePath(); g.fill();
    }
    const kf = (m.dedos || []).findIndex(v => v > 0.5);
    if (kf >= 0 && m.px[kf]) { const [x, y] = P(m.px[kf]); g.strokeStyle = '#ff5d5d'; g.lineWidth = 3 * devicePixelRatio;
      g.beginPath(); g.arc(x, y, 9 * devicePixelRatio, 0, 7); g.stroke(); }
    const fim = m.px[m.px.length - 1];
    if (fim) { const [x, y] = P(fim); g.fillStyle = cor; g.beginPath(); g.arc(x, y, 6 * devicePixelRatio, 0, 7); g.fill();
      g.font = `${12 * devicePixelRatio}px system-ui`; g.fillStyle = '#fff'; g.fillText('1 s', x + 9 * devicePixelRatio, y - 6 * devicePixelRatio); }
  }
  for (const lado of ['left', 'right']) {
    const u = vivo?.maos?.[lado]?.px; if (!u || (lado === 'left' && !$('v_esq').checked)) continue;
    const [x, y] = P(u), s = 7 * devicePixelRatio;
    g.fillStyle = '#fff'; g.strokeStyle = '#000'; g.lineWidth = 2; g.fillRect(x - s, y - s, 2 * s, 2 * s); g.strokeRect(x - s, y - s, 2 * s, 2 * s);
  }
}
['v_plano', 'v_atn', 'v_esq'].forEach(id => $(id).onchange = desenhaSobre);
addEventListener('resize', desenhaSobre);

// ── outras câmeras (miniaturas) ────────────────────────────────────
const vistas = new Set();
function miniatura(nome) {
  if (vistas.has(nome)) return; vistas.add(nome);
  const f = document.createElement('figure'), i = document.createElement('img'), c = document.createElement('figcaption');
  miniImg[nome] = i;
  const v = () => i.src = 'quadro/' + nome + '.jpg?t=' + Date.now();
  i.onload = () => setTimeout(v, 150); i.onerror = () => setTimeout(v, 1500); v();
  c.textContent = nome; f.append(i, c); $('thumbs').append(f);
}

// ── mundo 3D (pelvis: x frente, y esquerda, z cima) ────────────────
const caixa = $('mundo'), cena = new THREE.Scene();
const camera3 = new THREE.PerspectiveCamera(45, 4 / 3, 0.01, 20);
camera3.up.set(0, 0, 1); camera3.position.set(-1.05, -0.75, 0.55);
const rend = new THREE.WebGLRenderer({antialias: true}); rend.setPixelRatio(devicePixelRatio); rend.setClearColor(0x0a0d14);
caixa.appendChild(rend.domElement);
const ctrl = new OrbitControls(camera3, rend.domElement); ctrl.target.set(0.2, -0.05, -0.15); ctrl.enableDamping = true; ctrl.update();
cena.add(new THREE.AmbientLight(0xffffff, 0.7)); const sol = new THREE.DirectionalLight(0xffffff, 0.9); sol.position.set(1, -1, 2); cena.add(sol);
const grade = new THREE.GridHelper(2.4, 24, 0x34405a, 0x1d2433); grade.rotation.x = Math.PI / 2; grade.position.set(0.3, 0, -0.74); cena.add(grade);   // o chão (pelve a ~0,74 m)
cena.add(new THREE.AxesHelper(0.15));
const tronco = new THREE.Group(); cena.add(tronco);
const mat_corpo = new THREE.MeshStandardMaterial({color: 0x56607a, metalness: .2, roughness: .7, transparent: true, opacity: .35});
const corpo = new THREE.Mesh(new THREE.BoxGeometry(0.10, 0.24, 0.30), mat_corpo); corpo.position.set(-0.04, 0, 0.17); tronco.add(corpo);
const cabeca = new THREE.Mesh(new THREE.SphereGeometry(0.055, 24, 16), mat_corpo); cabeca.position.set(-0.02, 0, 0.40); tronco.add(cabeca);
const frente = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0.02, 0, 0.17), 0.12, 0xb0b8cc, 0.03, 0.02); tronco.add(frente);
// o G1 de verdade (URDF g1_body29_hand14 + malhas), juntas = as medidas do robô
let robo = null;
new URDFLoader().load('robo/g1_body29_hand14.urdf', r => {
  r.traverse(o => { if (o.isMesh) { o.material = new THREE.MeshStandardMaterial({color: 0xa9b1c4, metalness: .35, roughness: .55}); } });
  robo = r; cena.add(r); tronco.visible = false;
}, undefined, e => console.warn('URDF', e));
function seta(cor, tracejada) {
  const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0.3), new THREE.Vector3(0.38, 0, 0.3)]);
  const m = tracejada ? new THREE.LineDashedMaterial({color: cor, dashSize: .03, gapSize: .02}) : new THREE.LineBasicMaterial({color: cor});
  const l = new THREE.Line(g, m); if (tracejada) l.computeLineDistances(); cena.add(l); return l;
}
const yawMed = seta(0xe8a33d, false), yawPed = seta(0xff7bd5, true);
const esfera = cor => { const s = new THREE.Mesh(new THREE.SphereGeometry(0.025, 20, 14), new THREE.MeshStandardMaterial({color: cor, emissive: cor, emissiveIntensity: .25})); cena.add(s); return s; };
const maoD = esfera(0xffffff), maoE = esfera(0xb7c0ff);
const linha = cor => { const l = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({color: cor})); cena.add(l); return l; };
const planoD = linha(0x3ee6ff), planoE = linha(0x7a8cff), rastroD = linha(0x9aa3b8), rastroE = linha(0x5d6680);
// o plano como TUBO (linha do WebGL tem 1 px) + bolinhas a cada passo
const tuboMat = cor => new THREE.MeshStandardMaterial({color: cor, emissive: cor, emissiveIntensity: .35});
let tuboD = null, tuboE = null;
function tubo(antigo, arr, cor, raio) {
  if (antigo) { cena.remove(antigo); antigo.geometry.dispose(); }
  const v = (arr || []).filter(Boolean).map(p => new THREE.Vector3(p[0], p[1], p[2]));
  if (v.length < 2 || v[0].distanceTo(v.at(-1)) < 1e-4) return null;
  const m = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(v), 48, raio, 8, false), tuboMat(cor));
  cena.add(m); return m;
}
const fimD = esfera(0x3ee6ff); fimD.scale.setScalar(1.1);
const pontos = (l, arr) => { const v = (arr || []).filter(Boolean).map(p => new THREE.Vector3(p[0], p[1], p[2])); l.geometry.setFromPoints(v); l.visible = v.length > 1; };
// nuvem de pontos da profundidade estéreo da cabeça (servidor do WLA, 1 a cada 8 px), reprojetada com a câmera na pelvis
const nuvemGeo = new THREE.BufferGeometry();
const nuvem = new THREE.Points(nuvemGeo, new THREE.PointsMaterial({size: 0.012, vertexColors: true, sizeAttenuation: true}));
cena.add(nuvem);
let nuvemT = 0, PROF = null;
async function buscaNuvem() {
  // 03/11: com o mapa_ocupacao.py no ar, a nuvem é a da ZED CALIBRADA (distorção, disparidade, ~4 Hz) — ver buscaZed
  if (!$('v_nuvem').checked || mapa.zedT) { nuvem.visible = false; return; }
  try { PROF = await (await fetch('profundidade.json', {cache: 'no-store'})).json(); } catch (_) { PROF = null; }
  const cam = E.vivo?.camera; if (!PROF || !PROF.pronto || !cam || PROF.t === nuvemT) return;
  nuvemT = PROF.t;
  const pos = [], cor = [], R = cam.R, t = cam.t, f = cam.f, sx = cam.w / PROF.w, sy = cam.h / PROF.h;
  for (let i = 0; i < PROF.linhas; i++) for (let j = 0; j < PROF.colunas; j++) {
    const k = i * PROF.colunas + j, Z = PROF.Z[k]; if (Z == null || Z < 0.15 || Z > 2.0) continue;
    const u = (j * PROF.passo + PROF.passo / 2) * sx, v = (i * PROF.passo + PROF.passo / 2) * sy;
    const xc = (u - (cam.cx ?? cam.w / 2)) / f * Z, yc = (v - (cam.cy ?? cam.h / 2)) / f * Z, zc = Z;
    pos.push(R[0][0] * xc + R[0][1] * yc + R[0][2] * zc + t[0], R[1][0] * xc + R[1][1] * yc + R[1][2] * zc + t[1], R[2][0] * xc + R[2][1] * yc + R[2][2] * zc + t[2]);
    cor.push(PROF.cor[3 * k] / 255, PROF.cor[3 * k + 1] / 255, PROF.cor[3 * k + 2] / 255);
  }
  nuvemGeo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  nuvemGeo.setAttribute('color', new THREE.Float32BufferAttribute(cor, 3));
  nuvemGeo.computeBoundingSphere(); nuvem.visible = true;
  desenhaProf();
}
$('v_nuvem').onchange = () => { nuvemT = 0; buscaNuvem(); };
// 03/11: o MAPA DE OCUPAÇÃO do lidar (mapa_ocupacao.py na PGX); sem o executor, o G1 segue as juntas do mapa
const mapa = new CamadaMapa(cena);
async function buscaMapa() {
  mapa.visivel = $('v_mapa').checked; if (!$('v_mapa').checked) { $('mapa_info').textContent = ''; return; }
  const D = await mapa.busca(); $('mapa_info').textContent = resumoMapa(D);
}
$('v_mapa').onchange = buscaMapa; setInterval(buscaMapa, 1000);
async function buscaZed() {
  mapa.verZed = $('v_nuvem').checked;
  const V = await mapa.buscaVivo();
  if (robo && V?.juntas && !(E.vivo?.juntas && Date.now() / 1000 - (E.vivo.t || 0) < 1))
    for (const [n, q] of Object.entries(V.juntas)) if (robo.joints[n]) robo.joints[n].setJointValue(q);
  setTimeout(buscaZed, 250);
}
buscaZed();
setInterval(buscaNuvem, 1200);
function atualiza3D() {
  const v = E.vivo, p = E.plano;
  if (v && v.maos) {
    if (v.maos.right) { maoD.position.set(...v.maos.right.xyz); pontos(rastroD, v.maos.right.rastro); }
    if (v.maos.left) { maoE.position.set(...v.maos.left.xyz); pontos(rastroE, v.maos.left.rastro); }
    tronco.rotation.z = v.yaw_medido || 0; yawMed.rotation.z = v.yaw_medido || 0; yawPed.rotation.z = v.yaw_alvo || 0;
    if (robo && v.juntas) for (const [n, q] of Object.entries(v.juntas)) if (robo.joints[n]) robo.joints[n].setJointValue(q);
    const fd = v.maos.right?.fecha_pedido; maoD.material.color.set(fd > 0.5 ? 0xff5d5d : 0xffffff);
  }
  if (p && p.maos) {
    if (p.n !== atualiza3D.ultimo) { atualiza3D.ultimo = p.n;
      tuboD = tubo(tuboD, p.maos.right?.xyz, 0x3ee6ff, 0.012); tuboE = tubo(tuboE, p.maos.left?.xyz, 0x7a8cff, 0.008); }
    planoD.visible = planoE.visible = false;
    const f = p.maos.right?.xyz?.at(-1); if (f) fimD.position.set(...f);
  }
}
function loop3D() {
  const r = caixa.getBoundingClientRect();
  if (rend.domElement.width !== Math.round(r.width * devicePixelRatio)) { rend.setSize(r.width, r.width * 0.75); camera3.aspect = 4 / 3; camera3.updateProjectionMatrix(); }
  ctrl.update(); rend.render(cena, camera3); requestAnimationFrame(loop3D);
}
loop3D();

// ── gráficos no tempo ──────────────────────────────────────────────
function prepara(c) { const r = c.getBoundingClientRect(); if (c.width !== Math.round(r.width * devicePixelRatio)) { c.width = r.width * devicePixelRatio; c.height = r.height * devicePixelRatio; } return c.getContext('2d'); }
function serie(c, vals, opc) {
  const g = prepara(c), W = c.width, H = c.height, d = devicePixelRatio; g.clearRect(0, 0, W, H);
  if (!vals || !vals.length) { g.fillStyle = '#8b93a7'; g.font = `${12 * d}px system-ui`; g.fillText('aguardando o plano…', 10 * d, 20 * d); return; }
  const extra = [opc.medido, opc.alvo].filter(v => v != null);
  let lo = Math.min(...vals, ...extra, opc.lo ?? Infinity), hi = Math.max(...vals, ...extra, opc.hi ?? -Infinity); if (hi - lo < 1e-6) { hi += 0.05; lo -= 0.05; }
  const x0 = 34 * d, x1 = W - 10 * d, y0 = 10 * d, y1 = H - 18 * d;
  const X = k => x0 + (x1 - x0) * k / (vals.length - 1), Y = v => y1 - (y1 - y0) * (v - lo) / (hi - lo);
  g.strokeStyle = '#262c3a'; g.strokeRect(x0, y0, x1 - x0, y1 - y0);
  g.fillStyle = '#8b93a7'; g.font = `${10.5 * d}px system-ui`; g.fillText(opc.fmt(hi), 2 * d, y0 + 8 * d); g.fillText(opc.fmt(lo), 2 * d, y1);
  g.fillText('agora', x0, H - 4 * d); g.fillText('+1 s', x1 - 22 * d, H - 4 * d);
  if (opc.limiar != null) { g.setLineDash([4 * d, 4 * d]); g.strokeStyle = '#ff5d5d88'; g.beginPath(); g.moveTo(x0, Y(opc.limiar)); g.lineTo(x1, Y(opc.limiar)); g.stroke(); g.setLineDash([]); }
  g.strokeStyle = opc.cor; g.lineWidth = 2.2 * d; g.beginPath(); vals.forEach((v, k) => g[k ? 'lineTo' : 'moveTo'](X(k), Y(v))); g.stroke();
  const cur = Math.min(opc.cursor ?? 0, vals.length - 1); g.strokeStyle = '#7CFF6B'; g.lineWidth = 1.5 * d; g.beginPath(); g.moveTo(X(cur), y0); g.lineTo(X(cur), y1); g.stroke();
  if (opc.alvo != null) { g.fillStyle = '#ff7bd5'; g.beginPath(); g.arc(X(cur), Y(opc.alvo), 4 * d, 0, 7); g.fill(); }
  if (opc.medido != null) { g.fillStyle = '#fff'; g.beginPath(); g.arc(X(cur), Y(opc.medido), 4.5 * d, 0, 7); g.fill(); }
}
function graficos() {
  const p = E.plano, v = E.vivo || {}, cur = v.passo ?? 0;
  serie($('g_dedos'), p?.maos?.right?.dedos, {cor: '#3ee6ff', cursor: cur, lo: 0, hi: 1, limiar: 0.5, medido: v.maos?.right?.fecha_medido, fmt: x => x.toFixed(2)});
  serie($('g_yaw'), (p?.yaw || []).map(r => r * 57.3), {cor: '#e8a33d', cursor: cur, medido: v.yaw_medido != null ? v.yaw_medido * 57.3 : null,
                                                        alvo: v.yaw_alvo != null ? v.yaw_alvo * 57.3 : null, fmt: x => x.toFixed(0) + '°'});
  $('trecho').textContent = p && p.n != null ? `consulta ${p.n} · ${p.ms} ms · passo ${cur}/30` : '';
}

// ── atenção ────────────────────────────────────────────────────────
const CORES_PARTE = {'cabeça': '#4f9dff', 'profundidade': '#9b6bff', 'punho E': '#34c38f', 'punho D': '#2a9d6f', 'frase': '#ffd23f', 'estado': '#ff7b54', 'resto': '#3a4256'};
async function buscaAtencao() {
  try {
    const r = await fetch('atencao.json', {cache: 'no-store'});
    ATN = await r.json();
  } catch (_) { ATN = null; }
  const ok = ATN && ATN.pronto, idade = ok ? (Date.now() / 1000 - ATN.t) : 1e9;
  chip('c_atn', ok && idade < 5 ? 'on' : ok ? 'warn' : 'off', ok ? `atenção · ${idade < 5 ? 'ao vivo' : Math.round(idade) + ' s atrás'}` : 'atenção: servidor sem --atencao');
  if (!ok) return;
  const pt = ATN.partes; $('atn_barra').innerHTML = Object.entries(pt).map(([k, v]) => `<div title="${k} ${(v * 100).toFixed(1)}%" style="width:${v * 100}%;background:${CORES_PARTE[k]}"></div>`).join('');
  $('atn_partes').innerHTML = Object.entries(pt).map(([k, v]) => `<span style="--c:${CORES_PARTE[k]}">${k} <b>${(v * 100).toFixed(1)}%</b></span>`).join('');
  $('atn_meta').textContent = `consulta de ${ATN.ms} ms`;
  const mx = Math.max(...ATN.palavras.map(p => p[1]), 1e-6);
  $('palavras').innerHTML = '';
  for (const [w, v] of ATN.palavras) {
    const s = document.createElement('span'), a = v / mx; s.textContent = w;
    s.style.background = `rgba(255,210,63,${0.08 + 0.72 * a})`; s.style.color = a > 0.55 ? '#111' : '#e6e9ef'; s.title = (v * 100).toFixed(1) + '% da atenção na frase';
    $('palavras').append(s);
  }
  // camadas x partes
  const c = $('g_camadas'), g = prepara(c), W = c.width, H = c.height, d = devicePixelRatio, M = ATN.camadas, nC = ATN.nomes.length;
  g.clearRect(0, 0, W, H); const x0 = 26 * d, y0 = 16 * d, cw = (W - x0 - 4 * d) / nC, ch = (H - y0 - 4 * d) / M.length;
  const mxv = Math.max(...M.flat(), 1e-6);
  g.font = `${10 * d}px system-ui`; g.fillStyle = '#8b93a7';
  ATN.nomes.forEach((n, j) => g.fillText(n, x0 + j * cw + 3 * d, 11 * d));
  M.forEach((lin, i) => { g.fillStyle = '#8b93a7'; g.fillText(String(i + 1), 4 * d, y0 + i * ch + ch * 0.75);
    lin.forEach((v, j) => { const [R, G, B] = cor_calor(v / mxv); g.fillStyle = `rgb(${R},${G},${B})`; g.fillRect(x0 + j * cw + 1, y0 + i * ch + 1, cw - 2, ch - 2); }); });
  desenhaSobre(); mapasPorCamera();
}
const PAPEL_CAM = {head_left: ['cabeça', 'head_stereo_left'], head_depth: ['profundidade', null],
                   cam_wrist_left: ['punho esquerdo', 'wrist_left'], cam_wrist_right: ['punho direito', 'wrist_right']};
const fotos = {}, miniImg = {};
function desenhaProf() {   // a profundidade (Z) como imagem colorida (perto = quente), para o mapa da câmera "profundidade"
  if (!PROF || !PROF.pronto) return;
  const c = document.createElement('canvas'); c.width = PROF.colunas; c.height = PROF.linhas;
  const g = c.getContext('2d'), d = g.createImageData(PROF.colunas, PROF.linhas);
  PROF.Z.forEach((z, k) => { const v = z == null ? 0 : Math.max(0, Math.min(1, (1.2 - z) / 0.9)), [R, G, B] = cor_calor(v);
    d.data.set(z == null ? [0, 0, 0, 255] : [R * .55, G * .55, B * .55, 255], 4 * k); });
  g.putImageData(d, 0, 0); fotos.prof = c;
}
function mapasPorCamera() {
  const el = $('mapas_atn');
  for (const [papel, [nome, cam]] of Object.entries(PAPEL_CAM)) {
    let c = document.getElementById('ma_' + papel);
    if (!c) {
      const f = document.createElement('figure'); f.style.margin = 0;
      c = document.createElement('canvas'); c.id = 'ma_' + papel; c.style.cssText = 'width:100%;aspect-ratio:4/3;border-radius:7px;background:#000';
      const cap = document.createElement('figcaption'); cap.id = 'mac_' + papel; cap.className = 't'; cap.textContent = nome;
      f.append(c, cap); el.append(f);
    }
    const g = prepara(c), W = c.width, H = c.height; g.fillStyle = '#05070b'; g.fillRect(0, 0, W, H);
    const foto = papel === 'head_left' ? $('cam') : papel === 'head_depth' ? fotos.prof : miniImg[cam];
    if (foto && (foto.naturalWidth || foto.width)) { g.globalAlpha = papel === 'head_depth' ? 1 : 0.85; g.drawImage(foto, 0, 0, W, H); g.globalAlpha = 1; }
    const m = ATN?.mapas?.[papel];
    const parte = ATN?.partes?.[{head_left: 'cabeça', head_depth: 'profundidade', cam_wrist_left: 'punho E', cam_wrist_right: 'punho D'}[papel]];
    $('mac_' + papel).textContent = `${nome}${parte != null ? ` · ${(parte * 100).toFixed(1)}% da atenção` : ' · desligada'}`;
    if (!m) continue;
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high'; g.drawImage(calorCanvas(m), 0, 0, W, H);
  }
}

// ── temperaturas, falas, detalhes ──────────────────────────────────
function temps(v) {
  const t = v?.temps; if (!t) return;
  $('temps').innerHTML = Object.entries(t).map(([k, c]) => {
    const st = c >= 70 ? 'off' : c >= 55 ? 'warn' : 'on';
    return `<span class="chip ${st}"><i></i><span>${k.replace('_', ' ')} <b>${Math.round(c)} °C</b></span></span>`;
  }).join('');
}
let falasVistas = '';
function falas(e) {
  const lista = e.falas || []; $('traduz-tudo').style.display = e.tradutor ? '' : 'none'; if (!lista.length) return;
  const chave = JSON.stringify(lista.map(f => [f.i, f.pt])); if (chave === falasVistas) return; falasVistas = chave;
  const el = $('falas-lista'); el.innerHTML = '';
  for (const f of lista.slice().reverse()) {
    const d = document.createElement('div');
    d.innerHTML = `<span class="t">${f.hora} · ${f.origem || '—'}</span> <span></span>` + (f.pt ? `<div style="color:#9ccc9c">🇧🇷 </div>` : '');
    d.children[1].textContent = f.texto; if (f.pt) d.lastChild.append(f.pt);
    el.append(d);
  }
}
window.traduzTudo = async () => {
  for (const f of (E.falas || []).slice().reverse()) if (!f.pt)
    await fetch('traduz', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({i: f.i})}).catch(() => 0);
};
const ROTULOS = {modo: 'modo', consulta: 'consulta', acao_ms: 'tempo da inferência (ms)', tarefa_no_modelo: 'frase no modelo', passo_inicial: 'primeiro passo usado',
  passos_usados: 'passos usados', delta_pedido_pelo_modelo_cm: 'deslocamento pedido (cm)', delta_executado_cm: 'deslocamento executado (cm)',
  queda_medido_vs_comandado_cm: 'medido × comandado (cm)', erro_ik_mm: 'erro da IK (mm)', passos_limitados: 'passos limitados',
  maos_fechamento_0a1: 'fechamento das mãos', maior_desvio_da_partida_rad: 'maior desvio da partida (rad)'};
function detalhes(e) {
  const linhas = Object.entries(ROTULOS).filter(([k]) => e[k] !== undefined)
    .map(([k, r]) => `<tr><td>${r}</td><td>${typeof e[k] === 'object' ? Object.entries(e[k] || {}).map(([a, b]) => `${a}: ${Array.isArray(b) ? b.join(', ') : b}`).join(' · ') : e[k]}</td></tr>`);
  $('detalhes').innerHTML = linhas.join('') || '<tr><td>—</td><td>sem consulta ainda</td></tr>';
}

// ── simulação (MuJoCo): objetos e predição antiga, só se existirem ─
function simulacao(e) {
  const tem = e.objetos_mundo || e.predicao; $('sim').style.display = tem ? '' : 'none'; $('copo').style.display = e.objetos_mundo ? '' : 'none';
  if (e.objetos_mundo && !document.activeElement?.closest?.('#objetos')) {
    $('objetos').innerHTML = Object.entries(e.objetos_mundo).map(([n, v]) =>
      `<label style="margin-right:12px">${n} x <input type="text" data-n="${n}" data-c="x" value="${v[0].toFixed(3)}" style="width:66px;flex:none"> y <input type="text" data-n="${n}" data-c="y" value="${v[1].toFixed(3)}" style="width:66px;flex:none"></label>`).join('')
      + '<button type="button" id="mover">mover</button>';
    $('mover').onclick = async () => { const v = {}; document.querySelectorAll('#objetos input').forEach(i => { (v[i.dataset.n] ||= {})[i.dataset.c] = parseFloat(i.value); });
      for (const [nome, xy] of Object.entries(v)) await fetch('objeto', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({nome, ...xy})}); };
  }
}

// ── laço ───────────────────────────────────────────────────────────
async function tique() {
  try {
    E = await (await fetch('estado.json', {cache: 'no-store'})).json();
    const cams = Object.keys(E.cameras || {});
    const principal = CAM_PRINCIPAL.find(n => cams.includes(n));
    if (principal && principal !== camNome) { camNome = principal; liga(); }
    cams.filter(n => n !== camNome).forEach(miniatura);
    const idade = Math.max(...Object.values(E.cameras || {99: 99}));
    chip('c_cam', idade < 2 ? 'on' : 'off', idade < 2 ? 'câmera ok' : 'câmera parada');
    const ip = E.plano ? Date.now() / 1000 - E.plano.t : 1e9;
    chip('c_wla', ip < 3 ? 'on' : E.plano ? 'warn' : 'off', E.plano ? `WLA · ${E.plano.ms} ms${ip < 3 ? '' : ' · parado'}` : 'WLA · sem plano');
    chip('c_tarefa', E.tarefa ? 'on' : '', E.tarefa ? `tarefa: ${E.tarefa}` : 'sem tarefa (segurando)');
    atalhos(E); falas(E); temps(E.vivo); detalhes(E); simulacao(E); atualiza3D(); graficos(); desenhaSobre();
  } catch (_) { chip('c_cam', 'off', 'cabine sem resposta'); }
}
tique(); setInterval(tique, 200); buscaAtencao(); setInterval(buscaAtencao, 700);
</script>
</html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--porta", type=int, default=8090)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    cabine = Cabine()
    sobe(cabine, args.porta, args.host)
    print(f"cabine de teste em http://<ip>:{args.porta}/  (ctrl-c para sair)")
    t = 0
    while True:                       # quadros de teste, para provar o caminho
        t += 1
        for nome, cor in (("head_camera", 0), ("right_wrist_camera", 120)):
            img = np.zeros((240, 424, 3), np.uint8)
            img[:] = ((t * 3 + cor) % 180, 60, 90)
            cabine.publica_quadro(nome, img)
        cabine.publica_estado({"passo": t, "tarefa_ativa": cabine.tarefa()[0]})
        time.sleep(1 / 15)


if __name__ == "__main__":
    main()
