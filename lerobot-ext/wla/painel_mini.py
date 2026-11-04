#!/usr/bin/env python
"""PAINEL MINI do Prometheus (04/11) — uma página de 480x320 para a telinha de 3,5" (Raspberry Pi 5 ligado na PGX).

Roda na PGX (porta 8097) e junta:
  - do painel do robô (http://192.168.123.164:8095/estado.json): a trava da ponte (pânico), o modo (loco), a bateria,
    as temperaturas das juntas e das mãos, a pose inicial;
  - da PGX: se o servidor do WLA (8601), a cabine do executor (8090), o ER-1 (8092) e o mapa do lidar (8603) estão no ar.
Os botões repassam para o painel do robô (POST /rearmar, /panico, /pose_inicial?tipo=...): quem decide é a PONTE (o
rearme continua exigindo o cogumelo solto há >= 1 s). TRAVAR é imediato; os outros pedem confirmação na tela.

    ~/miniforge3/envs/wla/bin/python -u lerobot-ext/wla/painel_mini.py        # abra http://<pgx>:8097
"""
import argparse
import json
import socket
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# juntas que esquentam e importam (índice no lowstate, nome curto)
JUNTAS = [(13, "cintura roll"), (14, "cintura pitch"), (12, "cintura yaw"), (15, "ombro E"), (22, "ombro D"),
          (16, "ombro roll E"), (23, "ombro roll D"), (18, "cotovelo E"), (25, "cotovelo D")]
SERVICOS = [("WLA", 8601), ("executor", 8090), ("ER-1", 8092), ("mapa", 8603)]


def no_ar(porta, host="127.0.0.1"):
    try:
        with socket.create_connection((host, porta), timeout=0.25):
            return True
    except OSError:
        return False


class Estado:
    def __init__(self, painel):
        self.painel, self.ultimo, self.t, self.erro = painel, None, 0.0, None
        threading.Thread(target=self._laco, daemon=True).start()

    def _laco(self):
        while True:
            try:
                self.ultimo = json.loads(urllib.request.urlopen(f"{self.painel}/estado.json", timeout=1.5).read())
                self.t, self.erro = time.time(), None
            except Exception as e:  # noqa: BLE001
                self.erro = str(e)[:80]
            time.sleep(0.5)

    def resumo(self):
        e = self.ultimo if time.time() - self.t < 3 else None
        r = {"t": time.time(), "robo": e is not None, "erro": None if e else (self.erro or "sem resposta"),
             "servicos": {n: no_ar(p) for n, p in SERVICOS}}
        if not e:
            return r
        pan = e.get("panico") or {}
        r["trava"] = None if not pan else {"travada": bool(pan.get("panico")), "texto": pan.get("estado") or pan.get("motivo") or ""}
        r["modo"] = e.get("modo")
        b = e.get("bateria") or {}
        r["bateria"] = b.get("soc")
        r["pose"] = e.get("pose_inicial")
        temps = []
        J = (e.get("corpo") or {}).get("juntas") or []
        for i, nome in JUNTAS:
            if i < len(J) and J[i].get("temp"):
                temps.append([nome, max(J[i]["temp"])])
        for lado, nome in (("mao_left", "mão E"), ("mao_right", "mão D")):
            m = e.get(lado)
            if m:
                temps.append([nome, max(max(j["temp"]) for j in m["juntas"] if j.get("temp"))])
        if J:
            quente = max(J, key=lambda j: max(j.get("temp") or [0]))
            r["mais_quente"] = [quente.get("nome"), max(quente.get("temp") or [0])]
        r["temps"] = temps
        return r


PAGINA = r"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=480,height=320,initial-scale=1,user-scalable=no">
<title>Prometheus · mini</title>
<style>
:root{color-scheme:dark;--fundo:#0b0e13;--caixa:#151a22;--borda:#2a3140;--txt:#e6e9ef;--fraco:#8b93a1;
--ok:#3fb950;--aviso:#e3b341;--erro:#f85149;--azul:#58a6ff}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent;user-select:none}
html,body{margin:0;width:480px;height:320px;overflow:hidden;background:var(--fundo);color:var(--txt);
font:13px/1.25 system-ui,-apple-system,"Segoe UI",sans-serif}
#topo{display:flex;gap:5px;align-items:center;padding:6px 7px;height:38px;border-bottom:1px solid var(--borda)}
#trava{flex:1;padding:5px 8px;border-radius:8px;font-weight:700;font-size:13.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.livre{background:#173a22;border:1px solid var(--ok)} .presa{background:#4a1a1a;border:1px solid var(--erro)}
.semsinal{background:#2a2f3a;border:1px solid var(--borda);color:var(--fraco)}
.pil{padding:4px 7px;border-radius:99px;border:1px solid var(--borda);font-size:12px;color:var(--fraco);white-space:nowrap}
.pil.ok{color:var(--ok);border-color:#2c5a36}.pil.ruim{color:var(--erro);border-color:#6a2a2a}
#meio{display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;padding:6px 7px;height:178px;align-content:start}
.t{background:var(--caixa);border:1px solid var(--borda);border-radius:7px;padding:4px 6px;height:40px}
.t b{display:block;font-size:17px;line-height:1.1}.t span{font-size:11px;color:var(--fraco)}
.quente b{color:var(--aviso)}.muito b{color:var(--erro)}
#serv{display:flex;gap:4px;padding:0 7px;height:22px;align-items:center;font-size:11.5px;color:var(--fraco)}
#serv i{display:inline-block;width:8px;height:8px;border-radius:50%;margin:0 3px 0 6px;background:var(--erro)}
#serv i.ok{background:var(--ok)}
#botoes{display:grid;grid-template-columns:repeat(5,1fr);gap:5px;padding:5px 7px;height:82px}
button{font:inherit;font-weight:650;font-size:13px;border-radius:9px;border:1px solid var(--borda);color:var(--txt);
background:#1f2633;padding:2px}
button:active{filter:brightness(1.4)}
#bTrava{background:#8a1f1f;border-color:var(--erro)} #bRearma{background:#1f6f3a;border-color:var(--ok)}
#bGrav{background:#1f4f8a;border-color:var(--azul)} #bIni{background:#24405f} #bDesc{background:#4a3a1f;border-color:#d29922}
#conf{position:fixed;inset:0;background:#000c;display:none;flex-direction:column;align-items:center;justify-content:center;gap:14px}
#conf p{margin:0;font-size:18px;font-weight:700;text-align:center;padding:0 20px}
#conf div{display:flex;gap:14px}#conf button{width:150px;height:64px;font-size:18px}
#sim{background:#1f6f3a;border-color:var(--ok)}
#msg{position:fixed;left:7px;right:7px;bottom:90px;background:#222b3a;border:1px solid var(--borda);border-radius:8px;
padding:5px 8px;display:none;font-size:12.5px}
</style></head><body>
<div id="topo"><div id="trava" class="semsinal">conectando…</div>
  <span class="pil" id="pModo">—</span><span class="pil" id="pBat">—</span></div>
<div id="meio"></div>
<div id="serv"></div>
<div id="botoes">
  <button id="bTrava" onclick="acao('panico')">TRAVAR</button>
  <button id="bRearma" onclick="pede('rearmar','Rearmar a ponte? (o cogumelo precisa estar solto)')">Rearmar</button>
  <button id="bGrav" onclick="pede('pose:gravacao','Ir para a pose de GRAVAÇÃO?')">Pose de<br>gravação</button>
  <button id="bIni" onclick="pede('pose:inicial','Ir para a pose INICIAL?')">Pose<br>inicial</button>
  <button id="bDesc" onclick="pede('pose:descanso','Ir para o DESCANSO (punho fechado)?')">Descanso</button>
</div>
<div id="conf"><p id="confTxt"></p><div><button onclick="fecha()">Não</button><button id="sim">Sim</button></div></div>
<div id="msg"></div>
<script>
const $ = id => document.getElementById(id);
let pendente = null, timerConf = null;
function pede(a, txt) { pendente = a; $('confTxt').textContent = txt; $('conf').style.display = 'flex';
  clearTimeout(timerConf); timerConf = setTimeout(fecha, 6000); }
function fecha() { $('conf').style.display = 'none'; pendente = null; }
$('sim').onclick = () => { const a = pendente; fecha(); if (a) acao(a); };
function aviso(t) { $('msg').textContent = t; $('msg').style.display = 'block'; setTimeout(() => $('msg').style.display = 'none', 3500); }
async function acao(a) {
  try { const r = await (await fetch('acao', {method: 'POST', body: JSON.stringify({acao: a})})).json(); aviso(r.msg || 'ok'); }
  catch (e) { aviso('sem resposta da PGX'); }
}
function cor(t) { return t >= 75 ? 'muito' : t >= 60 ? 'quente' : ''; }
async function tique() {
  try {
    const e = await (await fetch('estado.json', {cache: 'no-store'})).json();
    const tv = $('trava');
    if (!e.robo) { tv.className = 'semsinal'; tv.textContent = 'robô sem resposta (init desligado?)'; }
    else if (!e.trava) { tv.className = 'semsinal'; tv.textContent = 'ponte sem sinal'; }
    else { tv.className = e.trava.travada ? 'presa' : 'livre'; tv.textContent = (e.trava.travada ? '🔒 TRAVADA' : '🟢 LIVRE') + (e.trava.texto ? ' · ' + e.trava.texto : ''); }
    $('pModo').textContent = e.modo || 'modo —'; $('pModo').className = 'pil ' + (e.modo === 'loco' ? 'ok' : '');
    $('pBat').textContent = e.bateria != null ? `🔋 ${e.bateria}%` : '🔋 —'; $('pBat').className = 'pil ' + (e.bateria != null && e.bateria < 25 ? 'ruim' : e.bateria != null ? 'ok' : '');
    $('meio').innerHTML = (e.temps || []).map(([n, t]) => `<div class="t ${cor(t)}"><b>${Math.round(t)}°</b><span>${n}</span></div>`).join('') +
      (e.mais_quente ? `<div class="t ${cor(e.mais_quente[1])}"><b>${Math.round(e.mais_quente[1])}°</b><span>+ quente: ${e.mais_quente[0]}</span></div>` : '') +
      (e.pose ? `<div class="t" style="grid-column:span 3;height:auto"><span>pose: ${e.pose}</span></div>` : '');
    $('serv').innerHTML = Object.entries(e.servicos || {}).map(([n, ok]) => `<i class="${ok ? 'ok' : ''}"></i>${n}`).join('');
  } catch (_) { $('trava').className = 'semsinal'; $('trava').textContent = 'sem resposta da PGX'; }
  setTimeout(tique, 700);
}
tique();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--porta", type=int, default=8097)
    ap.add_argument("--painel", default="http://192.168.123.164:8095", help="o painel do robô (init)")
    a = ap.parse_args()
    est = Estado(a.painel)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _manda(self, corpo, tipo, codigo=200):
            self.send_response(codigo)
            self.send_header("Content-Type", tipo)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(corpo)))
            self.end_headers()
            self.wfile.write(corpo)

        def do_GET(self):
            if self.path.startswith("/estado.json"):
                return self._manda(json.dumps(est.resumo(), ensure_ascii=False).encode(), "application/json")
            return self._manda(PAGINA.encode(), "text/html; charset=utf-8")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                pedido = json.loads(self.rfile.read(n) or b"{}").get("acao", "")
            except Exception:  # noqa: BLE001
                pedido = ""
            rota = {"panico": "/panico", "rearmar": "/rearmar", "pose:gravacao": "/pose_inicial?tipo=gravacao",
                    "pose:inicial": "/pose_inicial?tipo=inicial", "pose:descanso": "/pose_inicial?tipo=descanso"}.get(pedido)
            if rota is None:
                return self._manda(json.dumps({"msg": "pedido desconhecido"}).encode(), "application/json", 400)
            try:
                r = urllib.request.urlopen(urllib.request.Request(f"{a.painel}{rota}", data=b"", method="POST"), timeout=3).read()
                try:
                    msg = json.loads(r).get("msg") or "enviado"
                except Exception:  # noqa: BLE001
                    msg = "enviado"
                print(f"[mini] {pedido} -> {msg}", flush=True)
            except Exception as e:  # noqa: BLE001
                msg = f"o robô não respondeu ({str(e)[:40]})"
            return self._manda(json.dumps({"msg": {"panico": "TRAVADO", "rearmar": "pedido de rearme enviado"}.get(pedido, msg)},
                                          ensure_ascii=False).encode(), "application/json")

    srv = ThreadingHTTPServer(("0.0.0.0", a.porta), H)
    srv.daemon_threads = True
    print(f"📟 painel mini em http://0.0.0.0:{a.porta} (robô {a.painel})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
