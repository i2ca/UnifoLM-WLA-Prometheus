#!/usr/bin/env python
"""Varre episódios INTEIROS e pergunta ao ER-1 (original e com LoRA) o que está acontecendo, quadro a quadro.

Para cada dataset sorteia um episódio de TESTE (o LoRA não viu), pega um quadro a cada --passo-s e faz as
perguntas da tarefa (concluída? segurando? no lugar?) às duas versões: o adaptador é ligado/desligado no mesmo
modelo. Gera uma linha do tempo por episódio (PNG com miniaturas + respostas) e um JSON com tudo.

    CUDA_VISIBLE_DEVICES=1 python varre_episodio_er1.py --adaptador /data/mrwlker/er1_lora/run1/adaptador_ep1 \\
        --dados /data/mrwlker/er1_lora/dados --saida /data/mrwlker/er1_lora/varredura_ep1 \\
        --dataset /data/mrwlker/datasets_prometheus/pegar_caneca_mesa_nova ...
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from gera_dataset_er1 import FEITA, PERGUNTAS, SUFIXO, quadros_do_video
from treina_lora_er1 import ER1, responde, sim_nao


def perguntas_da_tarefa(chave, tarefa):
    ps = [("concluída?", FEITA[0].format(t=tarefa) + SUFIXO, ["head_stereo_left"])]
    for textos, cams, _ini, _fim in PERGUNTAS[chave]:
        rot = ("segurando?" if "holding" in textos[0] or "hand" in textos[0]
               else "no X?" if "X" in textos[0] else "no coador?")
        ps.append((rot, textos[0] + SUFIXO, cams))
    return ps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", required=True)
    ap.add_argument("--dados", required=True, help="pasta do gera_dataset_er1 (para saber os episódios de teste)")
    ap.add_argument("--adaptador", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--passo-s", type=float, default=1.0)
    ap.add_argument("--semente", type=int, default=1)
    a = ap.parse_args()
    rnd = random.Random(a.semente)
    saida = Path(a.saida)
    (saida / "quadros").mkdir(parents=True, exist_ok=True)
    teste = [json.loads(x) for x in open(Path(a.dados) / "teste.jsonl")]
    eps_teste = {}
    for l in teste:
        eps_teste.setdefault(l["dataset"], set()).add(l["episodio"])

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(ER1)
    base = AutoModelForImageTextToText.from_pretrained(ER1, dtype=torch.bfloat16, device_map="cuda").eval()
    modelo = PeftModel.from_pretrained(base, a.adaptador).eval()
    print(f"ER-1 + adaptador {a.adaptador}", flush=True)
    tudo = {}
    try:
        fonte = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        fonte = ImageFont.load_default()

    for raiz in map(Path, a.dataset):
        nome = raiz.name
        chave = next((k for k in PERGUNTAS if k in nome), None)
        if chave is None or nome not in eps_teste:
            continue
        info = json.loads((raiz / "meta/info.json").read_text())
        fps = info["fps"]
        ep = pd.concat([pd.read_parquet(f) for f in sorted((raiz / "meta/episodes").rglob("*.parquet"))])
        ei = rnd.choice(sorted(eps_teste[nome]))
        e = ep[ep["episode_index"] == ei].iloc[0]
        n, tarefa = int(e["length"]), list(e["tasks"])[0]
        ks = list(range(0, n, max(1, int(a.passo_s * fps)))) + [n - 1]
        ps = perguntas_da_tarefa(chave, tarefa)
        cams = sorted({c for _r, _q, cs in ps for c in cs})
        imgs = {}
        for cam in cams:
            col = f"videos/observation.images.{cam}"
            mp4 = raiz / info["video_path"].format(video_key=f"observation.images.{cam}",
                                                   chunk_index=int(e[f"{col}/chunk_index"]),
                                                   file_index=int(e[f"{col}/file_index"]))
            t0 = float(e[f"{col}/from_timestamp"])
            tempos = {round(t0 + k / fps, 4): k for k in ks}
            for t, img in quadros_do_video(mp4, list(tempos)).items():
                p = saida / "quadros" / f"{nome}_ep{ei}_{tempos[t]:04d}_{cam}.jpg"
                img.convert("RGB").save(p, quality=90)
                imgs[(cam, tempos[t])] = str(p)
        linhas = []
        for k in ks:
            r = {"quadro": k, "t": round(k / fps, 1)}
            for rot, q, cs in ps:
                caminhos = [imgs[(c, k)] for c in cs]
                with modelo.disable_adapter():
                    r[f"{rot}|original"] = sim_nao(responde(modelo, proc, caminhos, q))
                r[f"{rot}|lora"] = sim_nao(responde(modelo, proc, caminhos, q))
            linhas.append(r)
            print(nome, r, flush=True)
        tudo[nome] = {"episodio": ei, "tarefa": tarefa, "linhas": linhas}

        # linha do tempo: miniatura da cabeça + respostas (verde = sim, cinza = não) original / LoRA
        W, mini_h = 160, 120
        lin_h = 18
        H = mini_h + 30 + lin_h * 2 * len(ps) + 10
        fig = Image.new("RGB", (120 + W * len(ks), H), (24, 24, 28))
        d = ImageDraw.Draw(fig)
        d.text((6, 6), f"{nome} ep {ei}: {tarefa}", fill=(230, 230, 230), font=fonte)
        for j, (k, r) in enumerate(zip(ks, linhas)):
            x = 120 + j * W
            fig.paste(Image.open(imgs[("head_stereo_left", k)]).resize((W - 4, mini_h)), (x, 24))
            d.text((x + 3, 26), f"{r['t']} s", fill=(255, 255, 0), font=fonte)
            for i, (rot, _q, _c) in enumerate(ps):
                for m, versao in enumerate(("original", "lora")):
                    y = mini_h + 30 + (2 * i + m) * lin_h
                    v = r[f"{rot}|{versao}"]
                    cor = (60, 170, 90) if v == "yes" else (90, 90, 95) if v == "no" else (200, 120, 40)
                    d.rectangle([x, y, x + W - 4, y + lin_h - 3], fill=cor)
                    d.text((x + 4, y + 1), v, fill=(255, 255, 255), font=fonte)
                    if j == 0:
                        d.text((4, y + 1), f"{rot} {'orig' if m == 0 else 'LoRA'}", fill=(220, 220, 220), font=fonte)
        fig.save(saida / f"linha_do_tempo_{nome}.png")
    (saida / "varredura.json").write_text(json.dumps(tudo, ensure_ascii=False, indent=1))
    print(f"✅ {saida}", flush=True)


if __name__ == "__main__":
    main()
