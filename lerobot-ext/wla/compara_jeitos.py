#!/usr/bin/env python
"""Jeitos de PEDIR a mesma tarefa (31/10): nas consultas reais em que a tarefa era pegar a caneca, compara o plano com
"Pick up the white mug." (a frase do treino) contra pedidos em português que ele nunca viu e contra pegar a MAÇÃ
(referência de "outra tarefa"). Δ pequeno = entendeu igual.

    cd ~/DEV/unifolm-wla-lora && CUDA_VISIBLE_DEVICES=2 python .../compara_jeitos.py --modelos RUN:PASSO ... --rodadas ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
REF = "Pick up the white mug."
JEITOS = ["pegar a caneca branca", "pega a caneca branca", "Prometheus, pega a caneca branca pra mim",
          "pega o copo branco", "me dá a caneca", "pegue a xícara branca", "agarre a caneca",
          "pegue a caneca", "Pick up the apple."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", required=True)
    ap.add_argument("--rodadas", nargs="+", required=True)
    ap.add_argument("--base", default="playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors")
    a = ap.parse_args()
    import torch
    from replay_offline import obs_da_consulta, servidor
    cons = []
    for r in a.rodadas:
        p = Path(r)
        cons += [(p, x) for x in map(json.loads, open(p / "consultas.jsonl"))
                 if "mug" in x["frase"].lower() and "strainer" not in x["frase"].lower()
                 and (p / f"{x['n']:04d}.npz").exists()][::3]
    print(f"{len(cons)} consultas reais de PEGAR A CANECA", flush=True)
    for m in a.modelos:
        run, passo = m.rsplit(":", 1)
        s = servidor(a.base, run, int(passo))
        D = {j: [] for j in JEITOS}
        F = {j: [] for j in JEITOS}
        for p, x in cons:
            obs, _ = obs_da_consulta(p, f"{x['n']:04d}", REF)

            def plano(frase):
                o = dict(obs)
                o["instruction"] = frase
                torch.manual_seed(1000 + x["n"])
                r = s.get_action(o)
                return (np.asarray(r["action.right_ee_rpy"])[0][-1, :3],
                        float(np.mean(np.asarray(r["action.right_fig6d"])[0][-1])))
            m0, f0 = plano(REF)
            for j in JEITOS:
                m1, f1 = plano(j)
                D[j].append(np.linalg.norm(m1 - m0) * 100)
                F[j].append(abs(f1 - f0))
        print(f"\n== {Path(run).name} @ {passo}: diferença do plano para \"{REF}\"", flush=True)
        for j in JEITOS:
            print(f"   {j:42s} Δ mão {np.mean(D[j]):5.2f} cm | Δ dedos {np.mean(F[j]):.3f}", flush=True)
        del s
        torch.cuda.empty_cache()
    print("FIM", flush=True)


if __name__ == "__main__":
    main()
