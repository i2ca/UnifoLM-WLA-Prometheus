#!/usr/bin/env python
"""Perguntas de APONTAR para o co-treino do v4 (22/10): o VLM do WLA continua sabendo onde estão as coisas (no v3,
treinando o VLM inteiro, ele passou a errar a chaleira) e liga o nome do objeto ao lugar na imagem.

Dos episódios de TREINO: até --por-alvo quadros por episódio e alvo (pegar/colocar) do alvos.jsonl, com a resposta
do ER-1 no formato dele. Com --er1, o ER-1 aponta também os OUTROS objetos da mesa nesses quadros (retém o geral).
Junta com --junta (as perguntas do v3: etapas + sim/não) num jsonl só para trainer.cotreino.

    CUDA_VISIBLE_DEVICES=2 python gera_aponta_vqa.py --desc /data/mrwlker/wla_v2/descricao \\
        --alvos /data/mrwlker/wla_v2/alvos.jsonl --raiz /home/mrwlker/unifolm_data/Prometheus_G1_Dex3_Mix_Dataset \\
        --er1 /data/mrwlker/UnifoLM-ER-1 --junta /data/mrwlker/wla_v2/vqa/treino.jsonl --saida /data/mrwlker/wla_v2/vqa_v4
"""
import argparse
import json
import random
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "er1"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gera_dataset_er1 import _tarefa  # noqa: E402
from rotula_alvos_er1 import CAM, PERGUNTA  # noqa: E402

OUTROS = ["the white mug", "the red apple", "the kettle", "the black pot", "the coffee strainer", "the black X"]


def resposta(x, y, obj):
    return f'```json\n[{{"point_2d": [{int(round(x))}, {int(round(y))}], "label": "{obj}"}}]\n```'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desc", required=True)
    ap.add_argument("--alvos", required=True)
    ap.add_argument("--raiz", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--er1", default=None, help="ER-1 para apontar também os outros objetos")
    ap.add_argument("--junta", nargs="*", default=[], help="jsonl de perguntas para juntar")
    ap.add_argument("--por-alvo", type=int, default=3)
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--processos", type=int, default=32)
    a = ap.parse_args()
    rnd = random.Random(0)
    out = Path(a.saida)
    (out / "quadros").mkdir(parents=True, exist_ok=True)
    split = json.load(open(Path(a.desc) / "split.json"))
    treino = {(ds, int(e)) for ds, s in split.items() for e in s["treino"]}
    grupos = {}
    for d in map(json.loads, open(a.alvos)):
        if d.get("ponto") and (d["dataset"], d["episodio"]) in treino:
            grupos.setdefault((d["dataset"], d["episodio"], d["alvo"]), []).append(d)
    escolhidos = [x for g in grupos.values() for x in rnd.sample(g, min(a.por_alvo, len(g)))]
    # quadros a extrair, por arquivo de vídeo
    import pandas as pd
    pedidos = {}
    for ds in sorted({d["dataset"] for d in escolhidos}):
        info = json.loads((Path(a.raiz) / ds / "meta/info.json").read_text())
        eps = pd.concat([pd.read_parquet(f) for f in sorted((Path(a.raiz) / ds / "meta/episodes").rglob("*.parquet"))])
        eps = eps.set_index("episode_index")
        col = f"videos/observation.images.{CAM}"
        for d in (x for x in escolhidos if x["dataset"] == ds):
            row = eps.loc[d["episodio"]]
            mp4 = str(Path(a.raiz) / ds / info["video_path"].format(video_key=f"observation.images.{CAM}",
                                                                    chunk_index=int(row[f"{col}/chunk_index"]),
                                                                    file_index=int(row[f"{col}/file_index"])))
            t = round(float(row[f"{col}/from_timestamp"]) + d["quadro"] / info["fps"], 4)
            pedidos.setdefault((ds, mp4, d["episodio"]), {})[t] = (d["episodio"], "aponta", d["quadro"])
    jobs = [(mp4, tempos, ds, CAM, str(out / "quadros")) for (ds, mp4, _e), tempos in pedidos.items()]
    caminho = {}
    with Pool(a.processos) as pool:
        for feitos in pool.imap_unordered(_tarefa, jobs):
            for (_c, ep, _t, k), p in feitos.items():
                caminho[(Path(p).name.split(f"_ep{ep:03d}_")[0], ep, k)] = p
    linhas = []
    for d in escolhidos:
        p = caminho.get((d["dataset"], d["episodio"], d["quadro"]))
        if p:
            linhas.append({"imagens": [p], "pergunta": PERGUNTA.format(d["objeto"]),
                           "resposta": resposta(*d["ponto"], d["objeto"]), "tipo": f"aponta_{d['alvo']}",
                           "dataset": d["dataset"], "episodio": d["episodio"]})
    print(f"{len(linhas)} perguntas de apontar o alvo, {len(caminho)} quadros", flush=True)
    if a.er1:   # os outros objetos da mesa, pelo ER-1
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor
        proc = AutoProcessor.from_pretrained(a.er1)
        proc.tokenizer.padding_side = "left"
        m = AutoModelForImageTextToText.from_pretrained(a.er1, dtype=torch.bfloat16, device_map="cuda").eval()
        fila = [(p, o) for (ds, ep, k), p in caminho.items() for o in rnd.sample(OUTROS, 3)]
        for i in range(0, len(fila), a.lote):
            lt = fila[i:i + a.lote]
            textos = [proc.apply_chat_template([{"role": "user", "content": [{"type": "image"},
                                                {"type": "text", "text": PERGUNTA.format(o)}]}],
                                               tokenize=False, add_generation_prompt=True) for _p, o in lt]
            ent = proc(text=textos, images=[Image.open(p).convert("RGB") for p, _o in lt], return_tensors="pt",
                       padding=True).to(m.device)
            with torch.inference_mode():
                s = m.generate(**ent, max_new_tokens=28, do_sample=False)
            for (p, o), seq in zip(lt, s[:, ent["input_ids"].shape[1]:]):
                mt = re.search(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", proc.decode(seq, skip_special_tokens=True))
                if mt:
                    linhas.append({"imagens": [p], "pergunta": PERGUNTA.format(o),
                                   "resposta": resposta(float(mt.group(1)), float(mt.group(2)), o), "tipo": "aponta_outro"})
            if (i // a.lote) % 20 == 0:
                print(f"  ER-1 outros objetos: {i}/{len(fila)}", flush=True)
    for j in a.junta:
        linhas += [json.loads(x) for x in open(j)]
    rnd.shuffle(linhas)
    with open(out / "treino.jsonl", "w") as f:
        for l in linhas:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
    tipos = {}
    for l in linhas:
        tipos[l.get("tipo", "?")] = tipos.get(l.get("tipo", "?"), 0) + 1
    print(f"✅ {out}/treino.jsonl: {len(linhas)} perguntas {tipos}")


if __name__ == "__main__":
    main()
