#!/usr/bin/env python
"""Dataset de perguntas SIM/NÃO para o LoRA do ER-1 (02/10), tirado dos NOSSOS datasets de teleoperação.

A resposta certa vem de ONDE o quadro está no episódio — ninguém anota à mão:
  - começo do episódio (primeiros --inicio da duração): a tarefa ainda NÃO foi feita;
  - fim do episódio (últimos --fim, depois do A de salvar): a tarefa FOI feita.
Para cada dataset, a lista PERGUNTAS diz o que vale no começo e no fim (tarefa concluída, mão segurando o
objeto, objeto no lugar). Quadros do meio ficam de fora (ambíguos).

Separação por EPISÓDIO (não por quadro): --teste dos episódios de cada dataset vão só para o teste.

    python gera_dataset_er1.py --saida /data/mrwlker/er1_lora/dados \\
        --dataset /data/mrwlker/datasets_prometheus/pegar_caneca_mesa_nova ...

Saída: <saida>/quadros/*.jpg (RGB, 640x480, como a câmera chega ao ER-1 com --rgb) e treino.jsonl / teste.jsonl
com {"imagens": [...], "pergunta": ..., "resposta": "yes"|"no", "tipo": ..., "dataset": ..., "episodio": ...}.
"""
import argparse
import json
import random
from multiprocessing import Pool
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image

SUFIXO = " Answer only yes or no."
FEITA = ['The robot\'s task is: "{t}" Has the robot completed this task?',
         'Task: "{t}" Is this task finished?',
         'Has the robot already done this: "{t}"?']

# Por dataset (nome da pasta contém a chave): (pergunta, câmeras, resposta no COMEÇO, resposta no FIM)
SEGURA_CANECA = ["Is the robot's right hand holding the white mug?", "Is the white mug in the robot's right hand?"]
SEGURA_MACA = ["Is the robot's right hand holding the red apple?", "Is the red apple in the robot's right hand?"]
MACA_NO_X = ["Is the red apple on the black X?", "Is the black X covered by the red apple?"]
CANECA_COADOR = ["Is there a mug directly below the coffee strainer?", "Is the white mug under the coffee strainer?"]
PERGUNTAS = {
    "pegar_caneca": [(SEGURA_CANECA, ["head_stereo_left", "wrist_right"], "no", "yes")],
    "copo_branco": [(SEGURA_CANECA, ["head_stereo_left", "wrist_right"], "no", "yes")],
    "pegar_maca": [(SEGURA_MACA, ["head_stereo_left", "wrist_right"], "no", "yes")],
    "maca_x_preto": [(SEGURA_MACA, ["head_stereo_left", "wrist_right"], "no", "no"),
                     (MACA_NO_X, ["head_stereo_left"], "no", "yes")],
    "copo_no_coador": [(SEGURA_CANECA, ["head_stereo_left", "wrist_right"], "yes", "no"),
                       (CANECA_COADOR, ["head_stereo_left"], "no", "yes")],
}


def quadros_do_video(caminho, tempos):
    """Decodifica os quadros mais próximos de cada tempo (s, no arquivo inteiro) -> {tempo: PIL RGB}.
    Pula (seek) para perto do primeiro tempo: cada tarefa é um episódio, não o arquivo inteiro."""
    alvo = sorted(tempos)
    saida = {}
    with av.open(str(caminho)) as c:
        st = c.streams.video[0]
        c.seek(max(0, int((alvo[0] - 1.0) / st.time_base)), stream=st, backward=True)
        i = 0
        for fr in c.decode(st):
            t = float(fr.pts * st.time_base)
            while i < len(alvo) and t >= alvo[i] - 1e-3:
                saida[alvo[i]] = fr.to_image()
                i += 1
            if i >= len(alvo):
                break
    return saida


def _tarefa(job):
    """Um episódio de uma câmera: decodifica e salva os JPEGs. Roda num processo do Pool."""
    mp4, tempos, nome, cam, pasta = job
    feitos = {}
    for t, img in quadros_do_video(mp4, list(tempos)).items():
        ei, trecho, k = tempos[t]
        p = Path(pasta) / f"{nome}_ep{ei:03d}_{trecho}_{k:04d}_{cam}.jpg"
        img.convert("RGB").save(p, quality=92)
        feitos[(cam, ei, trecho, k)] = str(p)
    return feitos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", required=True, help="pastas LeRobot v3 (originais, não convertidas)")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--inicio", type=float, default=0.15, help="fração do começo do episódio = tarefa não feita")
    ap.add_argument("--fim", type=float, default=0.04, help="fração do fim do episódio = tarefa feita")
    ap.add_argument("--quadros", type=int, default=3, help="quadros sorteados por trecho (começo/fim) e episódio")
    ap.add_argument("--teste", type=float, default=0.15, help="fração dos episódios só para teste")
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--processos", type=int, default=64, help="processos decodificando vídeo em paralelo")
    a = ap.parse_args()
    rnd = random.Random(a.semente)
    saida = Path(a.saida)
    (saida / "quadros").mkdir(parents=True, exist_ok=True)
    linhas = {"treino": [], "teste": []}

    for raiz in map(Path, a.dataset):
        nome = raiz.name
        chave = next((k for k in PERGUNTAS if k in nome), None)
        if chave is None:
            print(f"⚠️  {nome}: sem perguntas definidas, pulando")
            continue
        info = json.loads((raiz / "meta/info.json").read_text())
        fps = info["fps"]
        ep = pd.concat([pd.read_parquet(f) for f in sorted((raiz / "meta/episodes").rglob("*.parquet"))])
        tarefas = pd.read_parquet(raiz / "meta/tasks.parquet")
        n_eps = len(ep)
        eps_teste = set(rnd.sample(sorted(ep["episode_index"]), max(1, round(a.teste * n_eps))))
        cams = sorted({c for p in PERGUNTAS[chave] for c in p[1]})
        # pedidos por arquivo de vídeo: (câmera, chunk, file) -> {tempo: (episódio, trecho, k)}
        pedidos, meta_q = {}, {}
        for _, e in ep.iterrows():
            ei, n = int(e["episode_index"]), int(e["length"])
            tarefa = list(e["tasks"])[0] if len(e["tasks"]) else tarefas.index[0]
            idx_ini = rnd.sample(range(0, max(1, int(n * a.inicio))), min(a.quadros, max(1, int(n * a.inicio))))
            idx_fim = rnd.sample(range(max(0, n - max(1, int(n * a.fim))), n), min(a.quadros, max(1, int(n * a.fim))))
            for trecho, idxs in (("inicio", idx_ini), ("fim", idx_fim)):
                for k in idxs:
                    meta_q[(ei, trecho, k)] = tarefa
                    for cam in cams:
                        col = f"videos/observation.images.{cam}"
                        arq = (cam, int(e[f"{col}/chunk_index"]), int(e[f"{col}/file_index"]), ei)
                        t = round(float(e[f"{col}/from_timestamp"]) + k / fps, 4)
                        pedidos.setdefault(arq, {})[t] = (ei, trecho, k)
        jobs = [(str(raiz / info["video_path"].format(video_key=f"observation.images.{cam}", chunk_index=ch,
                                                      file_index=fi)), tempos, nome, cam, str(saida / "quadros"))
                for (cam, ch, fi, _ei), tempos in pedidos.items()]
        imgs = {}   # (cam, ei, trecho, k) -> caminho jpg
        with Pool(a.processos) as pool:
            for feitos in pool.imap_unordered(_tarefa, jobs):
                imgs.update(feitos)
        n_l = 0
        for (ei, trecho, k), tarefa in meta_q.items():
            parte = "teste" if ei in eps_teste else "treino"
            casos = [([FEITA[rnd.randrange(len(FEITA))].format(t=tarefa)], ["head_stereo_left"], "no", "yes")] \
                + PERGUNTAS[chave]
            for textos, cams_q, r_ini, r_fim in casos:
                caminhos = [imgs.get((c, ei, trecho, k)) for c in cams_q]
                if None in caminhos:
                    continue
                linhas[parte].append({"imagens": caminhos, "pergunta": rnd.choice(textos) + SUFIXO,
                                      "resposta": r_ini if trecho == "inicio" else r_fim,
                                      "tipo": "tarefa_feita" if textos[0].startswith(("The robot's task", "Task:", "Has the"))
                                      else "estado", "dataset": nome, "episodio": ei, "trecho": trecho})
                n_l += 1
        print(f"{nome}: {n_eps} episódios ({len(eps_teste)} de teste), {len(imgs)} imagens, {n_l} perguntas", flush=True)

    for parte, ls in linhas.items():
        rnd.shuffle(ls)
        with open(saida / f"{parte}.jsonl", "w") as f:
            for l in ls:
                f.write(json.dumps(l, ensure_ascii=False) + "\n")
        sim = sum(l["resposta"] == "yes" for l in ls)
        print(f"{parte}: {len(ls)} perguntas ({sim} sim, {len(ls) - sim} não)")


if __name__ == "__main__":
    main()
