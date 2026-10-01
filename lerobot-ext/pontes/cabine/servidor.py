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
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


LIMITE = b"--quadro"


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
        self._tarefa_seq = 0
        self._parada = False
        self._pedido_copo = False
        self._objetos_pedidos = []
        self._clientes = 0

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
                "parada_pedida": self._parada,
                "fluxos_abertos": self._clientes,
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
        return self._erro(404, "não existe")

    # ── POST ────────────────────────────────────────────────────────────
    def do_POST(self):
        if not _autorizado(self):
            return self._erro(401, "sem autorização")
        caminho = self.path.split("?")[0]
        if caminho == "/parar":
            self.cabine.pede_parada()
            return self._texto(json.dumps({"ok": True}), "application/json")
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

    def _binario(self, corpo: bytes, tipo: str):
        self.send_response(200)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(corpo)))
        self.send_header("Cache-Control", "no-store")
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
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prometheus — cabine</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0e0e0e; color:#dcdcdc;
         font:14px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  header { display:flex; gap:10px; align-items:center; flex-wrap:wrap;
           padding:10px 14px; background:#171717; border-bottom:1px solid #2a2a2a; }
  #luz { width:10px; height:10px; border-radius:50%; background:#666; }
  #luz.on { background:#5cc45c } #luz.off { background:#d05050 }
  form { display:flex; gap:8px; flex:1 1 320px; }
  input { flex:1; background:#0e0e0e; color:#dcdcdc; border:1px solid #333;
          padding:7px 10px; font:inherit; border-radius:4px; }
  button { background:#2a4d2a; color:#dcdcdc; border:1px solid #3c6b3c;
           padding:7px 14px; font:inherit; border-radius:4px; cursor:pointer; }
  main { display:grid; grid-template-columns:repeat(auto-fit,minmax(380px,1fr));
         gap:10px; padding:10px; }
  figure { margin:0; background:#171717; border:1px solid #2a2a2a; border-radius:4px; }
  figure img { display:block; width:100%; height:auto; }
  figcaption { padding:5px 9px; color:#8a8a8a; border-top:1px solid #2a2a2a; }
  pre { margin:0; padding:10px 14px; color:#9a9a9a; white-space:pre-wrap;
        border-top:1px solid #2a2a2a; background:#141414; }
</style>
<header>
  <span id="luz"></span>
  <form onsubmit="manda(event)">
    <input id="t" placeholder="o que o robô deve fazer" autocomplete="off">
    <button>mandar</button>
    <button type="button" onclick="fetch('parar',{method:'POST'})">parar</button>
    <button type="button" onclick="fetch('copo',{method:'POST'})" title="mesma coisa que a tecla ç na janela do MuJoCo">mudar xícara (ç)</button>
  </form>
  <small id="meta"></small>
</header>
<section id="objetos" style="display:none;padding:8px 12px;background:#141414;
         border-bottom:1px solid #2a2a2a"></section>
<section id="grafico" style="display:none;padding:10px 10px 0">
  <canvas id="pred" width="1200" height="300"
          style="width:100%;max-width:1400px;background:#171717;border:1px solid #2a2a2a;border-radius:4px"></canvas>
</section>
<main id="cams"></main>
<pre id="estado">carregando…</pre>
<script>
const cams = document.getElementById('cams');
const vistas = new Set();

async function manda(e) {
  e.preventDefault();
  const t = document.getElementById('t');
  await fetch('tarefa', {method:'POST', headers:{'Content-Type':'application/json'},
                         body: JSON.stringify({texto: t.value})});
  // O texto FICA na caixa: repetir a mesma tarefa é o uso mais comum aqui.
  t.select();
}

// FOTO POR FOTO, e não MJPEG. Cada stream MJPEG prende uma conexão HTTP para
// sempre, e o navegador só abre SEIS por servidor: com sete câmeras na tela, as
// seis conexões ficavam todas presas em vídeo e o POST do botão "mandar" ficava
// na fila para sempre — o pedido nunca chegava (visto em 22/09, tarefa_seq
// parado em 1). Aqui cada quadro é uma requisição curta; a próxima só sai
// quando a anterior chegou, então uma câmera lenta não empilha pedidos.
function garante(nome) {
  if (vistas.has(nome)) return;
  vistas.add(nome);
  const f = document.createElement('figure');
  const i = document.createElement('img');
  const liga = () => i.src = 'quadro/' + nome + '.jpg?t=' + Date.now();
  i.onload = () => setTimeout(liga, 66);          // ~15 quadros/s por câmera
  i.onerror = () => setTimeout(liga, 1000);
  liga();
  const c = document.createElement('figcaption');
  c.textContent = nome;
  f.append(i, c); cams.append(f);
}

// Campos para arrastar os objetos pela mesa sem reiniciar. As coordenadas são
// do MUNDO, em metros: x para a frente do robô, y para a esquerda dele.
let mexendo = false;
function objetos(e) {
  const o = e.objetos_mundo;
  if (!o) return;
  const div = document.getElementById('objetos');
  div.style.display = '';
  if (mexendo) return;                       // não reescreve enquanto se digita
  const campos = Object.entries(o).map(([nome, v]) => `
    <label style="margin-right:14px">${nome}
      x <input data-n="${nome}" data-c="x" value="${v[0].toFixed(3)}" style="width:62px">
      y <input data-n="${nome}" data-c="y" value="${v[1].toFixed(3)}" style="width:62px">
    </label>`).join('');
  div.innerHTML = campos + `<button onclick="move()">mover</button>
    <small style="color:#7a7a7a"> x = para a frente, y = para a esquerda (metros)</small>`;
  div.querySelectorAll('input').forEach(i => {
    i.onfocus = () => mexendo = true;
    i.onblur = () => setTimeout(() => mexendo = false, 200);
    i.onkeydown = ev => { if (ev.key === 'Enter') move(); };
  });
}
async function move() {
  const v = {};
  document.querySelectorAll('#objetos input').forEach(i => {
    v[i.dataset.n] = v[i.dataset.n] || {};
    v[i.dataset.n][i.dataset.c] = parseFloat(i.value);
  });
  for (const [nome, xy] of Object.entries(v))
    await fetch('objeto', {method:'POST', headers:{'Content-Type':'application/json'},
                           body: JSON.stringify({nome, ...xy})});
  mexendo = false;
}

async function tique() {
  try {
    const e = await (await fetch('estado.json', {cache:'no-store'})).json();
    for (const nome of Object.keys(e.cameras || {})) garante(nome);
    const idades = Object.values(e.cameras || {});
    const velho = idades.length ? Math.max(...idades) : 99;
    document.getElementById('luz').className = velho > 2 ? 'off' : 'on';
    document.getElementById('meta').textContent =
      `tarefa: ${e.tarefa || '—'} · ${e.fluxos_abertos} fluxo(s) de vídeo`;
    document.getElementById('estado').textContent = JSON.stringify(e, null, 2);
    desenha(e); objetos(e);
  } catch (_) { document.getElementById('luz').className = 'off'; }
}
// ── Predição, desenhada AQUI a partir do JSON ─────────────────────────
// Era um PNG de matplotlib feito no servidor: 173 ms por pedaço segurando o
// GIL do laço de controle. No navegador custa nada e acompanha o cursor.
function desenha(e) {
  const p = e.predicao, a = e.acao || {}, l = e.laco || {};
  if (!p) return;
  document.getElementById('grafico').style.display = '';
  const c = document.getElementById('pred'), g = c.getContext('2d');
  const W = c.width, H = c.height, larg = W / 3;
  g.clearRect(0, 0, W, H); g.font = '13px monospace';
  const P = p.mao_dir_xyz, cur = l.cursor_no_pedaco ?? a.passo_do_pedaco ?? 0;
  const quadro = (i, titulo) => {
    g.strokeStyle = '#2a2a2a'; g.strokeRect(i * larg + 4, 4, larg - 8, H - 8);
    g.fillStyle = '#aaa'; g.fillText(titulo, i * larg + 12, 22);
  };
  // 1) de cima: y na horizontal (esquerda do robô à esquerda), x para cima
  quadro(0, `de cima (pelvis) — pedaço ${p.pedaco}, rede ${p.tempo_s}s, pulou ${p.pulados_por_atraso}`);
  const pts = P.map(v => [v[0], v[1]]).concat(
    Object.values(p.objetos_pelvis || {'o': p.copo_pelvis}).map(v => v.slice(0, 2)),
    [p.mao_dir_na_observacao.slice(0, 2)]);
  const xs = pts.map(v => v[0]), ys = pts.map(v => v[1]);
  const cx = (Math.max(...xs) + Math.min(...xs)) / 2, cy = (Math.max(...ys) + Math.min(...ys)) / 2;
  const esc = Math.min((larg - 60) / (Math.max(...ys) - Math.min(...ys) + 0.05),
                       (H - 70) / (Math.max(...xs) - Math.min(...xs) + 0.05));
  const tx = v => larg / 2 - (v[1] - cy) * esc, ty = v => H / 2 + 10 - (v[0] - cx) * esc;
  const linha = (arr, de, ate, cor, esp) => {
    g.strokeStyle = cor; g.lineWidth = esp; g.beginPath();
    for (let i = de; i < ate; i++) { const f = i === de ? 'moveTo' : 'lineTo'; g[f](tx(arr[i]), ty(arr[i])); }
    g.stroke(); g.lineWidth = 1;
  };
  linha(P, 0, P.length, '#555', 2);
  linha(P, 0, Math.min(cur + 1, P.length), '#5cc45c', 3);
  const objs = p.objetos_pelvis || {'objeto': p.copo_pelvis};
  const cores = ['#e0a030', '#e0e0ff', '#d070d0'];
  Object.entries(objs).forEach(([nome, v], k) => {
    g.fillStyle = cores[k % cores.length]; g.fillRect(tx(v) - 6, ty(v) - 6, 12, 12);
    g.fillText(nome, tx(v) + 9, ty(v) + 4);
  });
  g.fillStyle = '#e0e0e0'; g.beginPath(); g.arc(tx(p.mao_dir_na_observacao), ty(p.mao_dir_na_observacao), 4, 0, 7); g.fill();
  g.fillStyle = '#888';
  g.fillText('■ objetos  ● mão na observação  verde = já executado', 12, H - 12);
  // 2 e 3) séries no tempo do pedaço
  const serie = (i, vals, titulo, cor, ref) => {
    quadro(i, titulo);
    const lo = Math.min(...vals, ref ?? Infinity), hi = Math.max(...vals, ref ?? -Infinity);
    const x0 = i * larg + 20, x1 = (i + 1) * larg - 20, y0 = 40, y1 = H - 30;
    const X = k => x0 + (x1 - x0) * k / (vals.length - 1), Y = v => y1 - (y1 - y0) * (v - lo) / (hi - lo + 1e-9);
    if (ref !== undefined) { g.strokeStyle = '#e0a030'; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(x0, Y(ref)); g.lineTo(x1, Y(ref)); g.stroke(); g.setLineDash([]); }
    g.strokeStyle = cor; g.lineWidth = 2; g.beginPath();
    vals.forEach((v, k) => g[k ? 'lineTo' : 'moveTo'](X(k), Y(v))); g.stroke(); g.lineWidth = 1;
    g.strokeStyle = '#5cc45c'; g.beginPath(); g.moveTo(X(cur), y0); g.lineTo(X(cur), y1); g.stroke();
    g.fillStyle = '#777'; g.fillText(hi.toFixed(3), x0, y0 - 4); g.fillText(lo.toFixed(3), x0, y1 + 14);
  };
  serie(1, P.map(v => v[2]), 'altura da mão z (m) — tracejado: xícara', '#e0e0e0', p.copo_pelvis[2]);
  serie(2, p.garra_dir, 'garra dir (4,5 = aberta, 0 = fechada)', '#6aa0e0');
}

tique(); setInterval(tique, 250);
</script>
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
