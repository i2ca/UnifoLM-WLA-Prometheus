#!/usr/bin/env python
"""Pergunta ao VLM DENTRO do WLA (o que guia o DiT) as perguntas sim/não de TESTE do gera_dataset_er1.py
(episódios que o treino não viu) — mede se o co-treino ensinou o VLM a reconhecer a nossa cena pelo nome,
ou se ele só decorou as perguntas de treino. Compara runs (ex.: mix = VLM intocado x co-treino = VLM com LoRA).

    cd ~/DEV/unifolm-wla-lora && CUDA_VISIBLE_DEVICES=0 python avalia_perguntas_wla.py \\
        --run playground/Checkpoints/lora_prometheus_dex3_mix_athena:30000 \\
              playground/Checkpoints/lora_prometheus_dex3_cotreino:20000 \\
        --teste /data/mrwlker/er1_lora/dados/teste.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, ".")
from avalia_checkpoints_wla import carrega_lora  # noqa: E402


@torch.inference_mode()
def responde(iface, imagens, pergunta, tam):
    h, w = tam
    imgs = [Image.open(p).convert("RGB").resize((w, h)) for p in imagens]
    ent = iface.build_qwenvl_inputs(images=[imgs], instructions=[pergunta], add_generation_prompt=True)
    ent.pop("assistant_mask", None)
    out = iface.model.generate(**ent, max_new_tokens=4, do_sample=False)
    t = iface.processor.decode(out[0, ent["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
    return "yes" if t.startswith("yes") else "no" if t.startswith("no") else t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", required=True, help="pasta_do_run:passo")
    ap.add_argument("--teste", required=True)
    ap.add_argument("--base", default="playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--tamanho", type=int, nargs=2, default=[336, 448])
    a = ap.parse_args()
    teste = [json.loads(x) for x in open(a.teste)]
    if a.max:
        teste = teste[:a.max]
    resultado = {}
    for rp in a.run:
        run, passo = rp.rsplit(":", 1)
        m = carrega_lora(a.base, Path(run) / f"checkpoints/steps_{passo}_model.safetensors", run)
        iface = m.qwen_vl_interface
        por = defaultdict(lambda: [0, 0])
        for l in teste:
            ok = responde(iface, l["imagens"], l["pergunta"], a.tamanho) == l["resposta"]
            for k in ("total", f"tipo:{l['tipo']}", f"dataset:{l['dataset']}"):
                por[k][0] += ok
                por[k][1] += 1
        r = {k: round(x / n, 4) for k, (x, n) in sorted(por.items())}
        resultado[rp] = r
        print(f"\n=== {rp} ===\n" + "\n".join(f"  {k:40s} {v:.3f}" for k, v in r.items()), flush=True)
        del m
        torch.cuda.empty_cache()
    Path("avaliacao_perguntas_wla.json").write_text(json.dumps(resultado, indent=1))


if __name__ == "__main__":
    main()
