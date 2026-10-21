#!/usr/bin/env python
"""Perguntas das ETAPAS para o co-treino do VLM do WLA (21/10), dos episódios de TREINO (split.json).

Do etapas.jsonl do descreve_datasets.py (alcançar / fechar / levantar / carregar / soltar / voltar), quadros do
MEIO dos episódios — o que as perguntas sim/não do ER-1 (só começo e fim) não cobriam:

  etapa    "What is the robot doing right now?"           -> "I am reaching for the white mug with my right hand."
  objeto   "Which object is the robot's right hand going to grasp: the white mug or the red apple?"
                                                           -> "The white mug."   (só alcançar/fechar)
  pega     "How should the robot's right hand grasp the white mug?"
                                                           -> "From the side, with the hand upright."
           (o giro do punho na pega, medido: caneca ~-7°, maçã ~-80°; é a diferença que o modelo confundia)

Junta com as perguntas sim/não do ER-1 (tirando os episódios de validação) num só jsonl para trainer.cotreino.

    python gera_etapas_vqa.py --desc /data/mrwlker/wla_v2/descricao \\
        --raiz /home/mrwlker/unifolm_data/Prometheus_G1_Dex3_Mix_Dataset \\
        --er1 /data/mrwlker/er1_lora/dados/treino.jsonl --saida /data/mrwlker/wla_v2/vqa
"""
import argparse
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "er1"))
from gera_dataset_er1 import _tarefa  # noqa: E402

CAMS = ["head_stereo_left", "wrist_right"]
ETAPA = ["What is the robot doing right now? Answer in one short sentence.",
         "Describe what the robot is doing at this moment, in one short sentence."]
OBJETO = "Which object is the robot's right hand going to grasp: the white mug or the red apple?"
PEGA = {"the white mug": "From the side, with the hand upright.",
        "the red apple": "From above, with the palm facing down."}
# nome da pasta convertida -> nome do dataset original (as perguntas do ER-1 usam o original)
ORIGINAL = {"Copo_Branco": "copo_branco", "Copo_No_Coador": "copo_no_coador", "Maca_X_Preto": "maca_x_preto",
            "Pegar_Caneca_Mesa_Nova": "pegar_caneca_mesa_nova", "Pegar_Maca_Mesa_Nova": "pegar_maca_mesa_nova"}


def objeto(frase):
    f = frase.lower()
    return "the white mug" if ("mug" in f or "cup" in f) else "the red apple" if "apple" in f else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desc", required=True, help="saída do descreve_datasets.py (etapas.jsonl, split.json)")
    ap.add_argument("--raiz", required=True, help="pasta multi_task convertida (com os vídeos)")
    ap.add_argument("--er1", default=None, help="treino.jsonl das perguntas sim/não do ER-1 (junta, sem a validação)")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--por-etapa", type=int, default=2, help="quadros sorteados por etapa e episódio")
    ap.add_argument("--processos", type=int, default=64)
    ap.add_argument("--semente", type=int, default=0)
    a = ap.parse_args()
    rnd = random.Random(a.semente)
    desc, raiz, out = Path(a.desc), Path(a.raiz), Path(a.saida)
    (out / "quadros").mkdir(parents=True, exist_ok=True)
    split = json.load(open(desc / "split.json"))
    etapas = [json.loads(x) for x in open(desc / "etapas.jsonl")]
    treino = {(ds, int(e)) for ds, s in split.items() for e in s["treino"]}
    etapas = [e for e in etapas if (e["dataset"], e["episodio"]) in treino]

    pedidos, meta = {}, {}   # (cam, ds, chunk, file) -> {tempo: (ep, "etapa", quadro)}
    for ds in sorted({e["dataset"] for e in etapas}):
        info = json.loads((raiz / ds / "meta/info.json").read_text())
        fps = info["fps"]
        eps = pd.concat([pd.read_parquet(f) for f in sorted((raiz / ds / "meta/episodes").rglob("*.parquet"))])
        eps = eps.set_index("episode_index")
        for e in (x for x in etapas if x["dataset"] == ds):
            ini, fim = e["ini"], e["fim"]
            ks = sorted(rnd.sample(range(ini, fim), min(a.por_etapa, fim - ini)))
            row = eps.loc[e["episodio"]]
            for k in ks:
                meta[(ds, e["episodio"], k)] = e
                for cam in CAMS:
                    col = f"videos/observation.images.{cam}"
                    mp4 = str(raiz / ds / info["video_path"].format(video_key=f"observation.images.{cam}",
                                                                    chunk_index=int(row[f"{col}/chunk_index"]),
                                                                    file_index=int(row[f"{col}/file_index"])))
                    t = round(float(row[f"{col}/from_timestamp"]) + k / fps, 4)
                    pedidos.setdefault((cam, ds, mp4, e["episodio"]), {})[t] = (e["episodio"], "etapa", k)
    jobs = [(mp4, tempos, ds, cam, str(out / "quadros")) for (cam, ds, mp4, _ep), tempos in pedidos.items()]
    print(f"{len(meta)} quadros de {len(etapas)} etapas, {len(jobs)} trechos de vídeo", flush=True)
    # _tarefa devolve {(cam, ep, trecho, k): jpg} sem o dataset (o mesmo ep/quadro existe em vários datasets):
    # a chave é refeita com o dataset tirado do nome do arquivo, job a job, sem juntar num dict antes
    por_arq = {}
    with Pool(a.processos) as pool:
        for feitos in pool.imap_unordered(_tarefa, jobs):
            for (cam, ep, _tr, k), p in feitos.items():
                por_arq[(Path(p).name.split(f"_ep{ep:03d}_")[0], ep, k, cam)] = p

    linhas = []
    for (ds, ep, k), e in meta.items():
        cams = [por_arq.get((ds, ep, k, c)) for c in CAMS]
        if None in cams:
            continue
        frase = e["frase"]
        base = {"imagens": cams, "dataset": ds, "episodio": ep}
        linhas.append({**base, "pergunta": rnd.choice(ETAPA), "resposta": "I am " + frase + ".", "tipo": "etapa"})
        obj = objeto(e["tarefa"])
        if obj and (frase.startswith("reaching") or frase.startswith("closing")):
            linhas.append({**base, "pergunta": OBJETO, "resposta": obj[0].upper() + obj[1:] + ".",
                           "tipo": "objeto"})
            linhas.append({**base, "pergunta": f"How should the robot's right hand grasp {obj}?",
                           "resposta": PEGA[obj], "tipo": "pega"})
    n_etapas = len(linhas)
    if a.er1:   # as perguntas sim/não do ER-1, sem os episódios de validação
        # o ER-1 usa o nome do dataset original, às vezes com a data (maca_x_preto_2026-09-30): casa pelo começo
        val = {(ORIGINAL.get(ds.removeprefix("G1_Dex3_"), ds), int(x)) for ds, s in split.items() for x in s["val"]}
        n0 = 0
        for l in map(json.loads, open(a.er1)):
            n0 += 1
            nome, ep = l.get("dataset", ""), int(l.get("episodio", -1))
            if not any(nome.startswith(o) and ep == v for o, v in val):
                linhas.append(l)
        print(f"ER-1: {len(linhas) - n_etapas} de {n0} perguntas (fora as da validação)", flush=True)
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
