#!/usr/bin/env python
"""Rotula os ALVOS dos episódios com o ER-1 (22/10): onde está, na câmera da cabeça, o objeto a pegar e o lugar de
colocar — para o treino v4 desenhar um MARCADOR na imagem (o WLA segue a imagem muito melhor que a frase).

Por episódio (episodios.jsonl do descreve_datasets.py), um quadro a cada --passo:
  pegar    do início até a pega (+0,3 s): "Point to the white mug." / "the red apple"
  colocar  da pega até soltar (+0,3 s), nas tarefas de colocar: "the black X" / "the coffee strainer"
(o caneca→coador começa segurando: só colocar). Perguntas em LOTE na GPU (padding à esquerda).

    CUDA_VISIBLE_DEVICES=2 python rotula_alvos_er1.py --desc /data/mrwlker/wla_v2/descricao \\
        --raiz /home/mrwlker/unifolm_data/Prometheus_G1_Dex3_Mix_Dataset --er1 /data/mrwlker/UnifoLM-ER-1 \\
        --saida /data/mrwlker/wla_v2/alvos.jsonl
Sai uma linha por quadro: {dataset, episodio, quadro, alvo, objeto, ponto: [x, y] em 0-1000 ou null, cru}.
Retoma de onde parou (pula os episódios já no arquivo).
"""
import argparse
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "er1"))
from gera_dataset_er1 import quadros_do_video  # noqa: E402

PERGUNTA = ("Point to {}. Answer only JSON: [{{\"point_2d\": [x, y], \"label\": \"...\"}}] "
            "with coordinates in 0-1000.")
COLOCAR = {"Maca_X_Preto": "the black X", "Copo_No_Coador": "the coffee strainer"}
CAM = "head_stereo_left"


def objeto(frase):
    f = frase.lower()
    return "the white mug" if ("mug" in f or "cup" in f) else "the red apple" if "apple" in f else None


def pedidos_do_episodio(e, passo):
    n, pg, sl = e["quadros"], e.get("pega_quadro"), e.get("solta_quadro")
    m = 10
    out = []
    obj = objeto(e["tarefa"])
    if obj and not e.get("comeca_segurando"):
        fim = min(n, (pg + m) if pg is not None else n)
        out += [(k, "pegar", obj) for k in range(0, fim, passo)]
    lugar = next((v for k, v in COLOCAR.items() if k in e["dataset"]), None)
    if lugar:
        ini = pg if (pg is not None and not e.get("comeca_segurando")) else 0
        fim = min(n, (sl + m) if sl is not None else n)
        out += [(k, "colocar", lugar) for k in range(ini, fim, passo)]
    return out


def decodifica(job):
    """Um episódio: decodifica os quadros pedidos da câmera da cabeça (processo do Pool)."""
    raiz, e, passo = job
    ds = Path(raiz) / e["dataset"]
    info = json.loads((ds / "meta/info.json").read_text())
    eps = pd.concat([pd.read_parquet(f) for f in sorted((ds / "meta/episodes").rglob("*.parquet"))])
    row = eps.set_index("episode_index").loc[e["episodio"]]
    col = f"videos/observation.images.{CAM}"
    mp4 = ds / info["video_path"].format(video_key=f"observation.images.{CAM}", chunk_index=int(row[f"{col}/chunk_index"]),
                                         file_index=int(row[f"{col}/file_index"]))
    ped = pedidos_do_episodio(e, passo)
    if not ped:
        return e, []
    t0 = float(row[f"{col}/from_timestamp"])
    tempos = {round(t0 + k / info["fps"], 4): k for k in sorted({k for k, _, _ in ped})}
    imgs = {tempos[t]: im.convert("RGB") for t, im in quadros_do_video(mp4, list(tempos)).items()}
    return e, [(k, alvo, obj, imgs.get(k)) for k, alvo, obj in ped]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desc", required=True)
    ap.add_argument("--raiz", required=True)
    ap.add_argument("--er1", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--passo", type=int, default=10, help="quadros entre perguntas (10 = 3 por segundo)")
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--processos", type=int, default=24)
    a = ap.parse_args()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(a.er1)
    proc.tokenizer.padding_side = "left"
    m = AutoModelForImageTextToText.from_pretrained(a.er1, dtype=torch.bfloat16, device_map="cuda").eval()
    eps = [json.loads(x) for x in open(Path(a.desc) / "episodios.jsonl")]
    feitos = set()
    if Path(a.saida).exists():
        feitos = {(d["dataset"], d["episodio"]) for d in map(json.loads, open(a.saida))}
    eps = [e for e in eps if (e["dataset"], e["episodio"]) not in feitos]
    print(f"{len(eps)} episódios a rotular ({len(feitos)} já feitos)", flush=True)
    f = open(a.saida, "a")
    fila = []

    def roda(fila):
        textos = [proc.apply_chat_template([{"role": "user", "content": [{"type": "image"},
                                            {"type": "text", "text": PERGUNTA.format(obj)}]}],
                                           tokenize=False, add_generation_prompt=True) for *_, obj, _im in fila]
        ent = proc(text=textos, images=[im for *_, im in fila], return_tensors="pt", padding=True).to(m.device)
        with torch.inference_mode():
            s = m.generate(**ent, max_new_tokens=28, do_sample=False)
        for (e, k, alvo, obj, _im), seq in zip(fila, s[:, ent["input_ids"].shape[1]:]):
            cru = proc.decode(seq, skip_special_tokens=True)
            mt = re.search(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", cru)
            f.write(json.dumps({"dataset": e["dataset"], "episodio": e["episodio"], "quadro": k, "alvo": alvo,
                                "objeto": obj, "ponto": [float(mt.group(1)), float(mt.group(2))] if mt else None,
                                "cru": cru.strip()[:80]}, ensure_ascii=False) + "\n")
        f.flush()

    n = 0
    with Pool(a.processos) as pool:
        for e, itens in pool.imap(decodifica, [(a.raiz, e, a.passo) for e in eps]):
            for k, alvo, obj, im in itens:
                if im is not None:
                    fila.append((e, k, alvo, obj, im))
                if len(fila) >= a.lote:
                    roda(fila)
                    fila = []
            n += 1
            if n % 10 == 0:
                print(f"  {n}/{len(eps)} episódios", flush=True)
    if fila:
        roda(fila)
    print(f"✅ {a.saida}")


if __name__ == "__main__":
    main()
