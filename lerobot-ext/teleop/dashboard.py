"""Dashboard HTTP do teleop: o que está indo para o VR, visto do navegador.

Sobe junto com a teleoperação e serve, na porta 8080 do PC:

    /                     página do dashboard
    /api/state            estado ao vivo em JSON
    /stream/head          MJPEG da câmera da cabeça (a mesma que vai ao headset)
    /stream/wrist         MJPEG da câmera de pulso
    /qr.png               QR code com o link do VR, para escanear no Quest

Por que HTTP puro e não HTTPS: o Vuer precisa de TLS porque WebXR exige página
segura, mas o dashboard é só uma tela de acompanhamento. Deixar em HTTP evita
que você tenha que aceitar o certificado autoassinado de novo em cada máquina
que abrir a página.

Só stdlib + OpenCV — nada de dependência nova. O QR sai do
cv2.QRCodeEncoder, que já vem no opencv-python.
"""

import json
import logging
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def rota_local(destino: str) -> str | None:
    """IP local que o sistema usaria para falar com `destino`.

    O `connect` de um socket UDP só consulta a tabela de rotas — não sai
    pacote nenhum. É a forma mais barata de descobrir por qual perna da rede
    (cabo ou Wi-Fi) um endereço vai sair.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((destino, 80))
        return s.getsockname()[0]
    except Exception:
        return None
    finally:
        s.close()


def alcancavel(host: str, port: int, timeout: float = 0.6) -> bool:
    """TCP connect curto — só para separar 'ninguém escutando' de 'rede caída'."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def diagnostico_rede(endpoints: dict[str, tuple[str, int]], ip_do_vr: str) -> dict:
    """Por onde cada tráfego vai sair, e se o outro lado responde.

    Existe porque a separação certa é invisível quando funciona e silenciosa
    quando quebra: o VR quer o Wi-Fi (é onde o headset está) e o ZMQ das
    câmeras e da ponte quer o cabo, que é onde há banda para gravar dataset.
    Se o cabo cai, tudo continua "configurado certo" e simplesmente não chega
    imagem nenhuma — sem esta checagem isso vira tela preta sem explicação.
    """
    def _rede24(ip: str) -> str:
        return ip.rsplit(".", 1)[0]

    linhas = []
    for nome, (host, port) in endpoints.items():
        src = rota_local(host)
        # Sair de uma /24 rumo a OUTRA /24 pela mesma interface do VR é a
        # assinatura de rota caindo no gateway padrão — na prática, cabo fora.
        fora_de_casa = bool(src and _rede24(src) != _rede24(host))
        linhas.append({
            "nome": nome,
            "host": host,
            "port": port,
            "origem": src or "sem rota",
            "ok": alcancavel(host, port) if src else False,
            "pelo_vr": bool(src and src == ip_do_vr),
            "fora_de_casa": fora_de_casa,
        })

    avisos = []
    sem_rota = [l["nome"] for l in linhas if l["origem"] == "sem rota"]
    caidos = [l["nome"] for l in linhas if l["origem"] != "sem rota" and not l["ok"]]
    cabo_fora = [l["nome"] for l in linhas if l["pelo_vr"] and l["fora_de_casa"]]
    mesmo_wifi = [l["nome"] for l in linhas if l["pelo_vr"] and not l["fora_de_casa"]]

    if cabo_fora:
        avisos.append(
            f"CABO FORA: {', '.join(cabo_fora)} aponta para outra sub-rede mas está saindo por "
            f"{ip_do_vr} (a do VR). A rota do cabo sumiu e o tráfego caiu no gateway padrão — "
            "nada vai chegar. Conecte o cabo do robô e confira `ip -brief -4 addr`."
        )
    elif mesmo_wifi:
        avisos.append(
            f"{', '.join(mesmo_wifi)} está saindo pela MESMA interface do VR ({ip_do_vr}). "
            "Vídeo do headset e ZMQ disputando o Wi-Fi derrubam o FPS da gravação — use o cabo."
        )
    if sem_rota:
        avisos.append(f"Sem rota até: {', '.join(sem_rota)}. Interface caída.")
    if caidos and not cabo_fora:
        avisos.append(f"Rota existe mas ninguém responde em: {', '.join(caidos)}. Servidor do robô no ar?")

    return {"linhas": linhas, "avisos": avisos}


def listar_ips_locais() -> list[str]:
    """IPv4 não-loopback desta máquina, com a rota padrão na frente.

    A ordem importa: a máquina tem uma perna no cabo do robô (192.168.123.x) e
    outra no Wi-Fi da casa. O headset só enxerga a do Wi-Fi, que é justamente a
    da rota padrão — por isso ela vem primeiro e vira a URL oficial.
    """
    ips: list[str] = []
    padrao = rota_local("8.8.8.8")
    if padrao:
        ips.append(padrao)

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass

    return ips


def gerar_qr_png(texto: str, lado: int = 420) -> bytes | None:
    """QR code do texto, como PNG. None se o OpenCV desta máquina não tiver o encoder."""
    try:
        encoder = cv2.QRCodeEncoder_create()
        matriz = encoder.encode(texto)
    except Exception as e:
        logger.warning(f"Sem QR code: {e}")
        return None

    # INTER_NEAREST de propósito: interpolar suaviza as bordas dos módulos e
    # atrapalha a leitura. Uma borda branca (quiet zone) também é obrigatória
    # pela norma — sem ela muitos leitores simplesmente não enxergam o código.
    img = cv2.resize(matriz, (lado, lado), interpolation=cv2.INTER_NEAREST)
    borda = max(8, lado // 24)
    img = cv2.copyMakeBorder(img, borda, borda, borda, borda, cv2.BORDER_CONSTANT, value=255)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else None


class _Feed:
    """Último quadro de uma câmera, já em JPEG, com contagem de FPS."""

    def __init__(self, nome: str):
        self.nome = nome
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._novo = threading.Condition(self._lock)
        self._seq = 0
        self._marcas: list[float] = []

    def publicar(self, img_rgb: np.ndarray, qualidade: int = 70) -> None:
        # Recebe RGB e converte aqui: cv2.imencode assume BGR, e mandar RGB
        # direto deixaria o dashboard com vermelho e azul trocados em relação
        # ao que o operador vê no headset.
        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), qualidade],
        )
        if not ok:
            return
        agora = time.time()
        with self._novo:
            self._jpeg = buf.tobytes()
            self._seq += 1
            self._marcas.append(agora)
            if len(self._marcas) > 30:
                self._marcas = self._marcas[-30:]
            self._novo.notify_all()

    @property
    def fps(self) -> float:
        with self._lock:
            marcas = list(self._marcas)
        if len(marcas) < 2:
            return 0.0
        span = marcas[-1] - marcas[0]
        # Feed parado há muito tempo é 0, não o FPS histórico congelado.
        if time.time() - marcas[-1] > 2.0 or span <= 0:
            return 0.0
        return (len(marcas) - 1) / span

    def esperar(self, ultimo_seq: int, timeout: float = 1.0):
        """Bloqueia até haver um quadro mais novo que `ultimo_seq`."""
        with self._novo:
            if self._seq == ultimo_seq:
                self._novo.wait(timeout)
            return self._jpeg, self._seq


class TeleopDashboard:
    """Servidor do dashboard. `start()` sobe numa thread e não bloqueia nada."""

    def __init__(self, port: int = 8080, vr_url: str = "", info: dict | None = None):
        self.port = port
        self.vr_url = vr_url
        self.info = info or {}
        self.head = _Feed("head")
        self.wrist = _Feed("wrist")
        self._estado: dict = {}
        self._estado_lock = threading.Lock()
        self._qr = gerar_qr_png(vr_url) if vr_url else None
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ── API usada pelo teleop ────────────────────────────────────────────
    def atualizar_estado(self, **kwargs) -> None:
        with self._estado_lock:
            self._estado.update(kwargs)

    def snapshot_estado(self) -> dict:
        with self._estado_lock:
            estado = dict(self._estado)
        estado.update(
            head_fps=round(self.head.fps, 1),
            wrist_fps=round(self.wrist.fps, 1),
            vr_url=self.vr_url,
            tem_qr=self._qr is not None,
            **self.info,
        )
        return estado

    def start(self) -> None:
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # silencia o log de acesso no terminal
                pass

            def _headers(self, tipo: str, tamanho: int | None = None, cache: bool = False):
                self.send_response(200)
                self.send_header("Content-Type", tipo)
                if tamanho is not None:
                    self.send_header("Content-Length", str(tamanho))
                if not cache:
                    self.send_header("Cache-Control", "no-store")
                self.end_headers()

            def _bytes(self, corpo: bytes, tipo: str, cache: bool = False):
                self._headers(tipo, len(corpo), cache)
                self.wfile.write(corpo)

            def _mjpeg(self, feed: _Feed):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=quadro")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seq = -1
                try:
                    while True:
                        jpeg, seq = feed.esperar(seq, timeout=1.0)
                        if jpeg is None:
                            time.sleep(0.1)
                            continue
                        self.wfile.write(b"--quadro\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass  # aba fechada — normal, não é erro

            def do_GET(self):
                caminho = self.path.split("?")[0]
                if caminho == "/":
                    self._bytes(PAGINA.encode("utf-8"), "text/html; charset=utf-8")
                elif caminho == "/api/state":
                    self._bytes(json.dumps(dashboard.snapshot_estado()).encode(), "application/json")
                elif caminho == "/stream/head":
                    self._mjpeg(dashboard.head)
                elif caminho == "/stream/wrist":
                    self._mjpeg(dashboard.wrist)
                elif caminho == "/qr.png" and dashboard._qr:
                    self._bytes(dashboard._qr, "image/png", cache=True)
                else:
                    self.send_error(404)

        self._httpd = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None


PAGINA = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prometheus VLA — Teleop G1</title>
<style>
  :root{
    --bg:#0e1116; --card:#161b22; --linha:#262d36; --txt:#e6edf3; --fraco:#8b949e;
    --verde:#3fb950; --vermelho:#f0554e; --amarelo:#d29922; --azul:#58a6ff;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);
       font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
  header{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;
         padding:16px 22px;border-bottom:1px solid var(--linha)}
  h1{font-size:18px;margin:0;letter-spacing:.3px}
  .sub{color:var(--fraco);font-size:13px}
  .live{margin-left:auto;display:flex;align-items:center;gap:7px;font-size:13px;color:var(--fraco)}
  .ponto{width:9px;height:9px;border-radius:50%;background:var(--vermelho);
         animation:pisca 1.6s infinite}
  @keyframes pisca{50%{opacity:.25}}
  .wrap{padding:22px;display:grid;gap:18px;
        grid-template-columns:minmax(0,2fr) minmax(280px,1fr)}
  @media(max-width:900px){.wrap{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--linha);border-radius:12px;overflow:hidden}
  .card h2{font-size:12px;text-transform:uppercase;letter-spacing:.9px;color:var(--fraco);
           margin:0;padding:11px 14px;border-bottom:1px solid var(--linha);
           display:flex;justify-content:space-between;align-items:center}
  .card .corpo{padding:14px}
  .fps{font-variant-numeric:tabular-nums;color:var(--fraco);font-weight:400}
  img.feed{display:block;width:100%;background:#000;aspect-ratio:16/9;object-fit:contain}
  /* A câmera de pulso é 224×224 na origem: esticar para a largura da coluna só
     produz borrão. Fica no tamanho nativo, centralizada. */
  img.feed.quadrada{aspect-ratio:1/1;max-width:300px;margin:14px auto}
  .chips{display:flex;flex-wrap:wrap;gap:8px}
  .chip{display:inline-flex;align-items:center;gap:7px;padding:6px 12px;border-radius:999px;
        border:1px solid var(--linha);font-size:13px;font-weight:600;background:#0d1117}
  .chip .bola{width:8px;height:8px;border-radius:50%;background:currentColor}
  .verde{color:var(--verde);border-color:#1d572b}
  .vermelho{color:var(--vermelho);border-color:#5c2220}
  .amarelo{color:var(--amarelo);border-color:#5a4415}
  .cinza{color:var(--fraco)}
  table{width:100%;border-collapse:collapse;font-size:13px}
  td{padding:6px 0;border-bottom:1px solid var(--linha)}
  td:first-child{color:var(--fraco)}
  td:last-child{text-align:right;font-variant-numeric:tabular-nums}
  tr:last-child td{border-bottom:none}
  .barra{height:8px;background:#0d1117;border:1px solid var(--linha);border-radius:6px;
         position:relative;margin-top:8px}
  .barra i{position:absolute;top:50%;width:12px;height:12px;margin:-6px 0 0 -6px;
           border-radius:50%;background:var(--verde);transition:left .12s linear}
  .barra span{position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--linha)}
  .qr{text-align:center}
  .qr img{width:100%;max-width:260px;border-radius:8px;background:#fff;padding:8px}
  .qr code{display:block;margin-top:10px;font-size:11px;color:var(--azul);word-break:break-all}
  .dica{color:var(--fraco);font-size:12px;margin-top:8px}
  .alerta{margin:0 22px;padding:11px 14px;border-radius:10px;font-size:13px;
          background:#3b1d1c;border:1px solid #5c2220;color:#ffb4b0}
  ul.ips{list-style:none;margin:8px 0 0;padding:0;font-size:12px}
  ul.ips li{padding:3px 0;color:var(--fraco)}
  ul.ips a{color:var(--azul);text-decoration:none}
</style>
</head>
<body>
<header>
  <h1>Prometheus VLA · Teleop G1</h1>
  <span class="sub" id="sub">conectando…</span>
  <span class="live"><span class="ponto"></span> ao vivo</span>
</header>

<div id="alerta" class="alerta" hidden></div>

<div class="wrap">
  <div style="display:grid;gap:18px">
    <div class="card">
      <h2>Câmera da cabeça — o que vai para o headset
          <span class="fps" id="fps-head">— fps</span></h2>
      <img class="feed" id="head" src="/stream/head" alt="câmera da cabeça">
    </div>
    <div class="card">
      <h2>Câmera de pulso <span class="fps" id="fps-wrist">— fps</span></h2>
      <img class="feed quadrada" id="wrist" src="/stream/wrist" alt="câmera de pulso">
    </div>
  </div>

  <div style="display:grid;gap:18px;align-content:start">
    <div class="card">
      <h2>Estado</h2>
      <div class="corpo">
        <div class="chips" id="chips"></div>
        <div class="barra" style="margin-top:14px"><span></span><i id="tronco-pino"></i></div>
        <div class="dica" id="tronco-txt">tronco —</div>
      </div>
    </div>

    <div class="card">
      <h2>Abrir no headset</h2>
      <div class="corpo qr">
        <img src="/qr.png" alt="QR code do link do VR" id="qr">
        <code id="vrurl">—</code>
        <div class="dica">
          No Quest: abra o navegador e use o leitor de QR, ou digite o endereço.
          Aceite o aviso de certificado e toque em <b>Enter VR</b>.
        </div>
        <ul class="ips" id="ips"></ul>
      </div>
    </div>

    <div class="card">
      <h2>Rede — Wi-Fi para o VR, cabo para o ZMQ</h2>
      <div class="corpo">
        <table id="rede"></table>
        <div class="dica" id="rede-avisos"></div>
      </div>
    </div>

    <div class="card">
      <h2>Ligações</h2>
      <div class="corpo">
        <table id="infos"></table>
      </div>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);

function chip(texto, classe){
  return `<span class="chip ${classe}"><span class="bola"></span>${texto}</span>`;
}

async function tick(){
  let s;
  try{
    s = await (await fetch('/api/state', {cache:'no-store'})).json();
  }catch(e){
    $('sub').textContent = 'teleop offline';
    return;
  }

  $('sub').textContent = s.modo_entrada
    ? `${s.modo_entrada} · ${s.modo_display}` : '';

  const chips = [];
  chips.push(s.destravado ? chip('LIVRE','verde') : chip('TRAVADO — botão X','vermelho'));
  if(s.gravando)      chips.push(s.pausado ? chip('REC EM PAUSA','amarelo') : chip('GRAVANDO','vermelho'));
  else                chips.push(chip('sem gravação','cinza'));
  if(s.andando)       chips.push(chip(`ANDANDO ${(s.vx??0).toFixed(2)} m/s · ${(s.vyaw??0).toFixed(2)} rad/s`,'amarelo'));
  if(s.estabilizando) chips.push(chip(`ESTABILIZANDO ${s.estabilizando}s`,'amarelo'));
  chips.push(s.painel_pulso ? chip('painel de pulso ligado','cinza')
                            : chip('painel de pulso desligado','cinza'));
  $('chips').innerHTML = chips.join('');

  const lim = s.tronco_limite || 1;
  const frac = Math.max(-1, Math.min(1, (s.tronco||0)/lim));
  $('tronco-pino').style.left = (50 + frac*50) + '%';
  $('tronco-txt').textContent = `tronco ${(s.tronco||0).toFixed(2)} rad (limite ±${lim})`;

  $('fps-head').textContent  = `${s.head_fps ?? 0} fps`;
  $('fps-wrist').textContent = `${s.wrist_fps ?? 0} fps`;

  if(s.vr_url){ $('vrurl').textContent = s.vr_url; }
  if(!s.tem_qr){ $('qr').hidden = true; }

  $('infos').innerHTML = [
    ['robô',            s.robot_ip || '—'],
    ['câmera da cabeça', (s.img_server_ip||'—') + ':5555'],
    ['câmera de pulso',  s.wrist_cam ? (s.img_server_ip||'—') + ':' + s.wrist_port : 'desligada'],
    ['locomoção',        s.locomocao ? (s.loco_ip||'—') + ':' + s.loco_port : 'desligada'],
    ['servidor VR',      s.vuer_port ? (s.vuer_host||'') + ':' + s.vuer_port : '—'],
    ['dashboard',        (s.vuer_host||'') + ':' + (location.port||80)],
  ].map(([k,v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');

  if(s.ips && s.ips.length > 1){
    $('ips').innerHTML = '<li>outros endereços deste PC:</li>' + s.ips.slice(1).map(
      ip => `<li><a href="${(s.vr_url||'').replace(/\\/\\/[^:]+/, '//'+ip).replace(/wss:\\/\\/[^:]+/, 'wss://'+ip)}">${ip}</a></li>`
    ).join('');
  }

  const rede = s.rede || [];
  $('rede').innerHTML = [
    `<tr><td>VR + dashboard</td><td>${s.vuer_host||'—'} <b style="color:var(--verde)">Wi-Fi</b></td></tr>`
  ].concat(rede.map(l => {
    const cor = l.ok ? 'var(--verde)' : 'var(--vermelho)';
    const est = l.ok ? 'ok' : (l.origem === 'sem rota' ? 'sem rota' : 'sem resposta');
    return `<tr><td>${l.nome}</td><td>${l.origem} <b style="color:${cor}">${est}</b></td></tr>`;
  })).join('');
  $('rede-avisos').innerHTML = (s.avisos_rede||[]).map(a => '⚠️ ' + a).join('<br>');

  const avisos = [];
  if(s.aviso) avisos.push(s.aviso);
  (s.avisos_rede||[]).forEach(a => avisos.push(a));
  if(avisos.length){ $('alerta').innerHTML = avisos.map(a => '⚠️ ' + a).join('<br>'); $('alerta').hidden = false; }
  else { $('alerta').hidden = true; }
}

tick();
setInterval(tick, 500);
</script>
</body>
</html>
"""
