#!/usr/bin/env python
"""Banco de RECORTES de canecas e maçãs para os distratores do v5 (23/10).

No v4 o modelo ignorou o marcador: a arrumação da mesa já dizia qual objeto pegar (o alvo sempre na frente à
direita). O v5 cola canecas e maçãs FALSAS em lugares aleatórios da mesa na imagem da cabeça, inclusive perto do
alvo — só o marcador continua apontando o verdadeiro. Aqui saem os recortes: um quadrado em volta do ponto que o
ER-1 deu para o objeto a pegar (alvos.jsonl), nos episódios de TREINO, já na escala da imagem do treino (448x336).

    python gera_recortes.py --alvos /data/mrwlker/wla_v2/alvos.jsonl --desc /data/mrwlker/wla_v2/descricao \\
        --raiz /home/mrwlker/unifolm_data/Prometheus_G1_Dex3_Mix_Dataset --saida /data/mrwlker/wla_v2/recortes.pt
Sai {"img": (N, 3, L, L) float 0..1, "objeto": [...]} .
"""
import argparse
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rotula_alvos_er1 import decodifica  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alvos", required=True)
    ap.add_argument("--desc", required=True)
    ap.add_argument("--raiz", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--lado", type=int, default=48, help="px do recorte na imagem 448x336")
    ap.add_argument("--por-episodio", type=int, default=3)
    ap.add_argument("--processos", type=int, default=32)
    a = ap.parse_args()
    rnd = random.Random(0)
    split = json.load(open(Path(a.desc) / "split.json"))
    treino = {(ds, int(e)) for ds, s in split.items() for e in s["treino"]}
    eps = {(d["dataset"], d["episodio"]): d for d in map(json.loads, open(Path(a.desc) / "episodios.jsonl"))}
    pts = {}
    for d in map(json.loads, open(a.alvos)):
        if d["alvo"] == "pegar" and d.get("ponto") and (d["dataset"], d["episodio"]) in treino:
            # só antes da pega: depois a mão cobre o objeto
            pg = eps[(d["dataset"], d["episodio"])].get("pega_quadro") or 10 ** 9
            if d["quadro"] < pg - 15:
                pts.setdefault((d["dataset"], d["episodio"]), []).append(d)
    jobs, escolha = [], {}
    for k, lst in pts.items():
        sel = rnd.sample(lst, min(a.por_episodio, len(lst)))
        escolha[k] = {d["quadro"]: d for d in sel}
        e = dict(eps[k])
        e["_quadros"] = sorted(escolha[k])
        jobs.append(e)

    def pedidos(e, _passo):   # só os quadros escolhidos
        return [(q, "pegar", None) for q in e["_quadros"]]
    import rotula_alvos_er1 as R
    R.pedidos_do_episodio = pedidos
    imgs, objs = [], []
    L = a.lado
    with Pool(a.processos) as pool:
        for e, itens in pool.imap_unordered(decodifica, [(a.raiz, e, 1) for e in jobs]):
            for q, _alvo, _o, im in itens:
                if im is None:
                    continue
                d = escolha[(e["dataset"], e["episodio"])][q]
                im = np.asarray(im.resize((448, 336)), dtype=np.float32) / 255
                cx, cy = int(d["ponto"][0] / 1000 * 448), int(d["ponto"][1] / 1000 * 336)
                x0, y0 = cx - L // 2, cy - L // 2
                if x0 < 0 or y0 < 0 or x0 + L > 448 or y0 + L > 336:
                    continue
                imgs.append(torch.from_numpy(im[y0:y0 + L, x0:x0 + L].copy()).permute(2, 0, 1))
                objs.append(d["objeto"])
    banco = {"img": torch.stack(imgs), "objeto": objs}
    torch.save(banco, a.saida)
    print(f"✅ {a.saida}: {len(objs)} recortes {L}x{L} | " +
          ", ".join(f"{o}: {objs.count(o)}" for o in sorted(set(objs))))


if __name__ == "__main__":
    main()
