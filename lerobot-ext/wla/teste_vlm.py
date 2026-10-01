#!/usr/bin/env python
"""Testa as saídas de TEXTO do VLM do UnifoLM-WLA-1.0 (o ER-Flow que vem dentro do checkpoint)
e do UnifoLM-ER-1, nas mesmas imagens: apontar, caixas, plano, e sondagem dos tokens de
modelo de mundo (<seg_begin>...) e de ação discreta (<|EEF_START|>...).

Roda na PGX, env `wla`, da raiz do repo do WLA (não modifica nada dele):

    cd ~/DEV/unifolm-wla
    python ~/wla_testes/teste_vlm.py --obs ~/wla_testes/obs/real_ep0/obs_0000.npz ~/wla_testes/obs/fruta2/obs_0000.npz \
        --saida ~/wla_testes/vlm

As coordenadas do Qwen3-VL são relativas, de 0 a 1000.
"""
import argparse
import glob
import json
import logging
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())

PERGUNTAS = {
    "apontar": 'Point to the fruit and to the plate. Answer only with JSON: '
               '[{"point_2d": [x, y], "label": "fruit"}, {"point_2d": [x, y], "label": "plate"}]',
    "caixas": 'Detect the fruit and the plate. Answer only with JSON: '
              '[{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]',
    "plano": "You are the robot seen in this image. Task: pick up the fruit and place it on the plate. "
             "Describe, step by step, what your left and right grippers should do.",
}
SONDAS = {"mundo": "<seg_begin>", "acao_discreta": "<|EEF_START|>"}
INSTRUCAO_SONDA = "Pick up the fruit and place it on the plate."


def carrega_imagem(caminho):
    z = np.load(caminho, allow_pickle=True)
    bgr = z["observation__images__cam_left_high"]
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def gera(model, processor, img, texto, prefixo="", max_new=256):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": texto}]}]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefixo
    inp = processor(text=[prompt], images=[img], return_tensors="pt").to(model.device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]


def desenha(img, respostas, titulo):
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR).copy()
    h, w = bgr.shape[:2]
    cores = {"fruit": (0, 140, 255), "plate": (255, 200, 0)}
    for tipo in ("apontar", "caixas"):
        m = re.search(r"\[.*\]", respostas.get(tipo, ""), re.S)
        if not m:
            continue
        try:
            itens = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        for it in itens:
            rot = str(it.get("label", "?"))
            cor = cores.get(rot.split()[0].lower(), (255, 255, 255))
            if not isinstance(it, dict):
                continue
            if len(it.get("point_2d", [])) >= 2:
                x, y = it["point_2d"][:2]
                cv2.drawMarker(bgr, (int(x / 1000 * w), int(y / 1000 * h)), cor, cv2.MARKER_CROSS, 24, 3)
            if len(it.get("bbox_2d", [])) >= 4:
                x1, y1, x2, y2 = it["bbox_2d"][:4]
                p1 = (int(x1 / 1000 * w), int(y1 / 1000 * h))
                cv2.rectangle(bgr, p1, (int(x2 / 1000 * w), int(y2 / 1000 * h)), cor, 2)
                cv2.putText(bgr, rot, (p1[0], max(14, p1[1] - 4)), 0, 0.5, cor, 1)
    cv2.putText(bgr, titulo, (8, 22), 0, 0.6, (255, 255, 255), 2)
    return bgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", default="playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors")
    ap.add_argument("--obs", nargs="+", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--sem_er1", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)

    from unifolm_wla.model.framework.base_framework import baseframework
    wla = baseframework.from_pretrained(a.ckpt_path).to(torch.bfloat16).to("cuda").eval()
    modelos = {"WLA-1.0 (VLM interno)": (wla.qwen_vl_interface.model, wla.qwen_vl_interface.processor)}
    if not a.sem_er1:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        er1 = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--unitreerobotics--UnifoLM-ER-1/snapshots/*"))[0]
        modelos["ER-1"] = (Qwen3VLForConditionalGeneration.from_pretrained(er1, dtype=torch.bfloat16).to("cuda").eval(),
                           AutoProcessor.from_pretrained(er1))

    relatorio = {}
    for caminho in a.obs:
        img = carrega_imagem(caminho)
        nome_obs = f"{Path(caminho).parent.name}_{Path(caminho).stem}"
        linhas = []
        for nome_m, (model, proc) in modelos.items():
            resp = {k: gera(model, proc, img, q) for k, q in PERGUNTAS.items()}
            for k, pref in SONDAS.items():
                resp[k] = pref + gera(model, proc, img, INSTRUCAO_SONDA, prefixo=pref, max_new=96)
            relatorio[f"{nome_obs} | {nome_m}"] = resp
            linhas.append(desenha(img, resp, f"{nome_m} | {nome_obs}"))
            print(f"\n===== {nome_obs} | {nome_m}", flush=True)
            for k, v in resp.items():
                print(f"--- {k}: {v[:400]}", flush=True)
        cv2.imwrite(str(saida / f"{nome_obs}.jpg"), np.hstack(linhas))
    json.dump(relatorio, open(saida / "respostas.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
