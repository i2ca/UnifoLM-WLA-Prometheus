#!/usr/bin/env python
"""O PESO DA FRASE numa inferência de verdade (31/10): as consultas que o robô gravou (imagem, estado, frase —
gravador_wla.py) passam pelo modelo como no servidor (mesma montagem, mesma amostragem, mesma semente), e a frase é
trocada. Mede quanto a ação muda em relação à frase certa:
  Δ mão  (cm, posição da mão direita no fim do trecho de 1 s)   Δ dedos (0..1)   Δ yaw (graus)
Variações: outra tarefa (caneca <-> maçã <-> coador), frase sem sentido, frase vazia, a mesma tarefa em português e
uma tarefa NUNCA treinada. Quanto maior o Δ para "outra tarefa" e menor para "português", mais ele USA a frase
(e entende o sentido, não as palavras).

    cd ~/DEV/unifolm-wla-lora && CUDA_VISIBLE_DEVICES=2 python .../compara_frase.py \\
        --modelos playground/Checkpoints/prometheus_v7:20000 playground/Checkpoints/prometheus_v8:1000 \\
        --rodadas /data/mrwlker/wla_real_runs/20261029_142620 /data/mrwlker/wla_real_runs/20261029_144233
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

PT = {"Pick up the white mug.": "Pegue a caneca branca.", "Pick up the apple.": "Pegue a maçã.",
      "Place the white mug under the coffee strainer.": "Coloque a caneca embaixo do coador.",
      "Pick up the apple and place it on the black X.": "Pegue a maçã e coloque no X preto."}
OUTRA = {"Pick up the white mug.": "Pick up the apple.", "Pick up the apple.": "Pick up the white mug.",
         "Place the white mug under the coffee strainer.": "Pick up the white mug.",
         "Pick up the apple and place it on the black X.": "Pick up the white mug."}
VARIACOES = {
    "outra tarefa": lambda f: OUTRA.get(f, "Pick up the apple."),
    "sem sentido": lambda f: "Dance with the blue chair next to the window.",
    "vazia": lambda f: "",
    "português": lambda f: PT.get(f, f),
    "nunca treinada": lambda f: "Pick up the blue kettle and lift it.",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelos", nargs="+", required=True, help="run:passo")
    ap.add_argument("--rodadas", nargs="+", required=True)
    ap.add_argument("--cada", type=int, default=6, help="usa 1 consulta a cada N")
    ap.add_argument("--base", default="playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors")
    ap.add_argument("--saida", default="/data/mrwlker/compara_frase.json")
    a = ap.parse_args()
    import torch
    from replay_offline import obs_da_consulta, servidor

    consultas = []
    for r in a.rodadas:
        p = Path(r)
        L = [json.loads(x) for x in open(p / "consultas.jsonl")]
        consultas += [(p, x) for x in L[::a.cada] if (p / f"{x['n']:04d}.npz").exists()]
    print(f"{len(consultas)} consultas reais de {len(a.rodadas)} rodadas", flush=True)

    def roda(s, obs, sem):
        torch.manual_seed(sem)
        r = s.get_action(obs)
        mao = np.asarray(r["action.right_ee_rpy"])[0][-1, :3]
        dedos = float(np.mean(np.asarray(r["action.right_fig6d"])[0][-1]))
        yaw = float(np.asarray(r["action.lower_body"])[0][-1, 12])
        return mao, dedos, yaw

    resumo = {}
    for m in a.modelos:
        run, passo = m.rsplit(":", 1)
        s = servidor(a.base, run, int(passo))
        D = {k: {"mao_cm": [], "dedos": [], "yaw_graus": []} for k in VARIACOES}
        for p, x in consultas:
            obs, _ = obs_da_consulta(p, f"{x['n']:04d}", x["frase"])
            sem = 1000 + x["n"]
            m0, d0, y0 = roda(s, obs, sem)
            for nome, f in VARIACOES.items():
                o = dict(obs)
                o["instruction"] = f(x["frase"])
                m1, d1, y1 = roda(s, o, sem)
                D[nome]["mao_cm"].append(float(np.linalg.norm(m1 - m0) * 100))
                D[nome]["dedos"].append(abs(d1 - d0))
                D[nome]["yaw_graus"].append(abs(np.degrees(y1 - y0)))
        resumo[m] = {k: {q: round(float(np.mean(v)), 2) for q, v in d.items()} for k, d in D.items()}
        print(f"\n== {Path(run).name} @ {passo}  (média sobre {len(consultas)} consultas reais)", flush=True)
        print(f"   {'troca da frase':16s} {'Δ mão (cm)':>11s} {'Δ dedos':>9s} {'Δ yaw (°)':>10s}")
        for k, d in resumo[m].items():
            print(f"   {k:16s} {d['mao_cm']:11.2f} {d['dedos']:9.3f} {d['yaw_graus']:10.2f}", flush=True)
        del s
        torch.cuda.empty_cache()
    json.dump(resumo, open(a.saida, "w"), indent=1, ensure_ascii=False)
    print("FIM", flush=True)


if __name__ == "__main__":
    main()
