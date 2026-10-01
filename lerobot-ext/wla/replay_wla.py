#!/usr/bin/env python
"""
REPLAY de uma rodada gravada pelo roda_wla_real.py — nada é enviado ao robô.

  1. ASSISTIR (sempre): gera <pasta>/replay.html — play/pausa/barra de tempo com a cabeça, os
     punhos e a trajetória prevista de cada consulta, e os números ao lado.

  2. REINFERIR (--reinferir): manda de novo CADA observação gravada ao servidor do WLA (8600) e
     compara a resposta nova com a da hora. Serve para ver se a rede é consistente na mesma cena, e
     para testar variações sem o robô:
        --frase "Pick up the apple."     outra instrução
        --so-cabeca                      sem as câmeras dos punhos
     Grava <pasta>/reinferencia[_<rótulo>].jsonl e mostra, por consulta, a diferença (cm) entre o
     fim do trecho de agora e o gravado.

    cd ~/DEV/unifolm-wla && ~/miniforge3/envs/wla/bin/python \\
        ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/replay_wla.py ~/wla_real_runs/20260929_182732 --reinferir
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent

PAGINA = """<!doctype html><html lang="pt-br"><head><meta charset="utf-8"><title>Replay WLA __NOME__</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
:root{color-scheme:dark}body{margin:0;background:#0f1115;color:#e4e6eb;font:14px system-ui,sans-serif}
main{padding:12px 16px}.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:8px 0}
button{background:#1f4f8a;color:#fff;border:1px solid #58a6ff;border-radius:6px;padding:6px 14px;font:inherit;cursor:pointer}
input[type=range]{flex:1;min-width:200px}.g{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:8px}
.g figure{margin:0}.g img{width:100%;border-radius:4px;background:#000;display:block}figcaption{color:#8b93a1;font-size:12px}
pre{background:#171a21;border:1px solid #2a2f3a;border-radius:6px;padding:10px;white-space:pre-wrap;color:#c9d1d9}</style></head>
<body><main><h2 style="margin:4px 0">Replay — rodada __NOME__</h2>
<div class="bar"><button id="p">▶ play</button><button onclick="va(-1)">◀</button><button onclick="va(1)">▶</button>
<input type="range" id="r" min="0" value="0"><span id="c"></span>
<label>velocidade <select id="v"><option value="1">1x (tempo real)</option><option value="0.5">2x</option>
<option value="2">0,5x</option></select></label></div>
<div class="g"><figure><img id="i0"><figcaption>cabeça (ZED esq.) — o que a IA viu</figcaption></figure>
<figure><img id="i1"><figcaption>punho esquerdo</figcaption></figure><figure><img id="i2"><figcaption>punho direito</figcaption></figure>
<figure><img id="i3"><figcaption>o que a IA quis (branco = agora, linha = 30 passos)</figcaption></figure></div>
<pre id="t"></pre></main><script>
const L = __DADOS__;
const r = document.getElementById('r'); r.max = L.length - 1; let k = 0, tocando = null;
const f = n => String(n).padStart(4, '0');
function mostra() {
  const x = L[k];
  ['cabeca', 'punho_esq', 'punho_dir', 'traj'].forEach((s, j) => document.getElementById('i' + j).src = f(x.n) + '_' + s + '.jpg');
  r.value = k; document.getElementById('c').textContent = `consulta ${x.n} de ${L[L.length - 1].n} · t = ${x.t.toFixed(1)} s`;
  const ri = x.reinferencia ? `\\nREINFERÊNCIA: fim do trecho difere do gravado em esq ${x.reinferencia.left} cm, dir ${x.reinferencia.right} cm` : '';
  document.getElementById('t').textContent =
    `frase: ${x.frase}\\nΔ pedido pela IA (fim do trecho, cm): esq ${JSON.stringify(x.delta_pedido_fim_cm.left)}  dir ${JSON.stringify(x.delta_pedido_fim_cm.right)}\\n` +
    `mão medida (cm, pelvis): esq ${JSON.stringify(x.mao_medida_cm.left)}  dir ${JSON.stringify(x.mao_medida_cm.right)}\\n` +
    `garra agora ${JSON.stringify(x.garras_agora)} → prevista no fim ${JSON.stringify(x.garra_fim_prevista)} (0 fecha, 5,5 abre)\\n` +
    `inferência ${x.ms} ms · trava ${JSON.stringify(x.trava)}` + ri;
}
function va(d) { k = Math.max(0, Math.min(L.length - 1, k + d)); mostra(); }
function passo() {
  if (k >= L.length - 1) { para(); return; }
  const dt = (L[k + 1].t - L[k].t) * 1000 * parseFloat(document.getElementById('v').value);
  tocando = setTimeout(() => { va(1); passo(); }, dt);
}
function para() { clearTimeout(tocando); tocando = null; document.getElementById('p').textContent = '▶ play'; }
document.getElementById('p').onclick = () => { if (tocando) return para(); if (k >= L.length - 1) k = 0;
  document.getElementById('p').textContent = '⏸ pausa'; passo(); };
r.oninput = () => { k = +r.value; mostra(); };
document.onkeydown = e => { if (e.key === 'ArrowRight') va(1); if (e.key === 'ArrowLeft') va(-1); if (e.key === ' ') { e.preventDefault(); document.getElementById('p').click(); } };
mostra();
</script></body></html>"""


def reinferir(pasta, L, url, frase, so_cabeca, rotulo):
    import cv2
    sys.path.insert(0, str(AQUI))
    from roda_wla_real import ClienteWLA
    cli = ClienteWLA(url)
    saida = open(pasta / f"reinferencia{'_' + rotulo if rotulo else ''}.jsonl", "w")
    difs = {"left": [], "right": []}
    for x in L:
        n = f"{x['n']:04d}"
        z = np.load(pasta / f"{n}.npz")
        obs = {"observation.images.cam_left_high": cv2.imread(str(pasta / f"{n}_cabeca.jpg")),
               "observation.state.left_ee_6d": z["ee_medido_left"], "observation.state.right_ee_6d": z["ee_medido_right"],
               "observation.state.left_gripper": z["garras"][:1].astype(np.float32),
               "observation.state.right_gripper": z["garras"][1:].astype(np.float32),
               "observation.state.lower_body": z["q_medido"][:15].astype(np.float32),
               "instruction": frase or x["frase"]}
        if not so_cabeca and (pasta / f"{n}_punho_esq.jpg").exists() and (pasta / f"{n}_punho_dir.jpg").exists():
            obs["observation.images.cam_left_wrist"] = cv2.imread(str(pasta / f"{n}_punho_esq.jpg"))
            obs["observation.images.cam_right_wrist"] = cv2.imread(str(pasta / f"{n}_punho_dir.jpg"))
        _, acao, ms, _, _ = cli.acao(obs)
        r = {"n": x["n"], "ms": round(ms)}
        for l in ("left", "right"):
            novo = acao[f"action.{l}_ee_rpy"][0][-1, :3]
            velho = z[f"acao__{l}_ee_rpy"][-1, :3]
            d = float(np.linalg.norm(novo - velho) * 100)
            difs[l].append(d)
            r[l] = round(d, 1)
            r[f"delta_novo_{l}_cm"] = np.round((novo - z[f"ee_medido_{l}"][:3]) * 100, 1).tolist()
            r[f"garra_nova_{l}"] = round(float(acao[f"action.{l}_gripper"][0][-1, 0]), 2)
        saida.write(json.dumps(r) + "\n")
        x["reinferencia"] = {"left": r["left"], "right": r["right"]}
        print(f"  #{x['n']:3d} {ms:4.0f} ms | fim do trecho difere do gravado: esq {r['left']:5.1f} cm, dir "
              f"{r['right']:5.1f} cm | Δ novo esq {r['delta_novo_left_cm']} dir {r['delta_novo_right_cm']} | garra "
              f"{r['garra_nova_left']}/{r['garra_nova_right']}", flush=True)
    for l in difs:
        print(f"{'esquerda' if l == 'left' else 'direita '}: diferença mediana {np.median(difs[l]):.1f} cm, "
              f"máxima {np.max(difs[l]):.1f} cm", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pasta", nargs="?")
    ap.add_argument("--reinferir", action="store_true")
    ap.add_argument("--servidor", default="ws://127.0.0.1:8600")
    ap.add_argument("--frase", default=None)
    ap.add_argument("--so-cabeca", action="store_true")
    a = ap.parse_args()
    pasta = Path(a.pasta) if a.pasta else max((Path.home() / "wla_real_runs").iterdir(), key=lambda p: p.stat().st_mtime)
    L = [json.loads(x) for x in open(pasta / "consultas.jsonl")]
    if a.reinferir:
        rot = "_".join(filter(None, ["frase" if a.frase else "", "so_cabeca" if a.so_cabeca else ""]))
        reinferir(pasta, L, a.servidor, a.frase, a.so_cabeca, rot)
    (pasta / "replay.html").write_text(PAGINA.replace("__NOME__", pasta.name).replace("__DADOS__", json.dumps(L)))
    print(pasta / "replay.html")


if __name__ == "__main__":
    main()
