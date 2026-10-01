#!/usr/bin/env python
"""Relatório de uma rodada gravada pelo roda_wla_real.py: uma página HTML com, por consulta, o que a
IA viu (cabeça + punhos), o que ela quis (trajetória, garra, deslocamento) e onde as mãos estavam.

    python relatorio_wla_real.py                         # a rodada mais recente de ~/wla_real_runs
    python relatorio_wla_real.py ~/wla_real_runs/20260929_190000 --cada 2
Gera <pasta>/relatorio.html (abre no navegador; as imagens ficam ao lado).
"""
import argparse
import html
import json
from pathlib import Path


def svg_series(pts, cores, rotulos, titulo, unidade, w=900, h=180):
    todos = [v for s in pts for v in s if v is not None]
    if not todos:
        return ""
    lo, hi = min(todos + [0]), max(todos + [0])
    hi = hi if hi > lo else lo + 1
    n = max(len(s) for s in pts)
    X = lambda i: 40 + (w - 60) * i / max(1, n - 1)  # noqa: E731
    Y = lambda v: 20 + (h - 40) * (hi - v) / (hi - lo)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" style="width:100%;max-width:{w}px;background:#171a21;border-radius:6px">',
           f'<text x="8" y="14" fill="#aaa" font-size="12">{html.escape(titulo)} ({unidade})</text>',
           f'<line x1="40" x2="{w - 20}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="#444"/>',
           f'<text x="4" y="{Y(hi) + 4:.0f}" fill="#777" font-size="10">{hi:.1f}</text>',
           f'<text x="4" y="{Y(lo):.0f}" fill="#777" font-size="10">{lo:.1f}</text>']
    for s, c, r in zip(pts, cores, rotulos):
        p = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(s) if v is not None)
        out.append(f'<polyline fill="none" stroke="{c}" stroke-width="1.6" points="{p}"><title>{r}</title></polyline>')
    x = 50
    for c, r in zip(cores, rotulos):
        out.append(f'<text x="{x}" y="{h - 4}" fill="{c}" font-size="11">■ {html.escape(r)}</text>')
        x += 12 + 7 * len(r)
    out.append("</svg>")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta", nargs="?")
    ap.add_argument("--cada", type=int, default=1, help="mostra uma consulta a cada N")
    a = ap.parse_args()
    pasta = Path(a.pasta) if a.pasta else max((Path.home() / "wla_real_runs").iterdir(), key=lambda p: p.stat().st_mtime)
    L = [json.loads(x) for x in open(pasta / "consultas.jsonl")]
    cfg = json.loads((pasta / "config.json").read_text())
    cores = ["#58a6ff", "#3fb950", "#e3b341"]
    graf = ""
    for l, nome in (("left", "mão esquerda"), ("right", "mão direita")):
        graf += svg_series([[x["delta_pedido_fim_cm"][l][k] for x in L] for k in range(3)], cores,
                           ["x frente", "y esquerda", "z cima"], f"{nome}: deslocamento pedido pela IA no fim do trecho", "cm")
        graf += svg_series([[x["mao_medida_cm"][l][k] for x in L] for k in range(3)], cores,
                           ["x", "y", "z"], f"{nome}: posição MEDIDA (pelvis)", "cm")
    graf += svg_series([[x["garra_fim_prevista"]["left"] for x in L], [x["garra_fim_prevista"]["right"] for x in L]],
                       ["#58a6ff", "#f85149"], ["esquerda", "direita"], "garra prevista (0 fecha … 5,5 abre)", "")
    linhas = []
    for x in L[::a.cada]:
        n = f"{x['n']:04d}"
        ims = "".join(f'<img src="{n}_{k}.jpg" loading="lazy" title="{k}">' for k in ("cabeca", "punho_esq", "punho_dir")
                      if (pasta / f"{n}_{k}.jpg").exists())
        info = (f"<b>#{x['n']}</b> t={x['t']:.1f}s · {x['ms']} ms<br>{html.escape(x['frase'])}<br>"
                f"Δ pedido esq {x['delta_pedido_fim_cm']['left']} cm<br>Δ pedido dir {x['delta_pedido_fim_cm']['right']} cm<br>"
                f"garra agora {x['garras_agora']} → fim {x['garra_fim_prevista']}<br>"
                f"mão medida esq {x['mao_medida_cm']['left']} dir {x['mao_medida_cm']['right']}<br>"
                f"trava {x['trava'].get('panico')}")
        linhas.append(f'<div class="c"><div class="i">{info}</div><div class="f">{ims}'
                      f'<img src="{n}_traj.jpg" loading="lazy" title="trajetória"></div></div>')
    pagina = f"""<!doctype html><html lang="pt-br"><head><meta charset="utf-8"><title>WLA rodada {pasta.name}</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
:root{{color-scheme:dark}}body{{margin:0;background:#0f1115;color:#e4e6eb;font:13px system-ui,sans-serif}}
main{{padding:12px 16px}}.c{{display:flex;gap:10px;border-top:1px solid #2a2f3a;padding:8px 0;flex-wrap:wrap}}
.i{{flex:0 0 300px;color:#c9d1d9}}.f{{display:flex;gap:6px;flex-wrap:wrap}}.f img{{height:180px;border-radius:4px}}
pre{{color:#8b93a1;white-space:pre-wrap}}</style></head><body><main>
<h2>UnifoLM-WLA-1.0 no G1 — rodada {pasta.name} ({len(L)} consultas)</h2>
<pre>{html.escape(json.dumps(cfg, ensure_ascii=False))}</pre>{graf}{''.join(linhas)}</main></body></html>"""
    (pasta / "relatorio.html").write_text(pagina)
    print(pasta / "relatorio.html")


if __name__ == "__main__":
    main()
