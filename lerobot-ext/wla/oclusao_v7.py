#!/usr/bin/env python
"""MAPA DE OCLUSÃO (30/10): o que, nas imagens, o WLA USA de verdade para decidir — teste causal, sem robô.

Para uma consulta gravada no robô (~/wla_real_runs/<rodada>, gravador_wla.py), um quadrado cinza passa por cada parte
de cada câmera, um de cada vez, e mede-se quanto a AÇÃO muda em relação à cena intacta (mesma semente do fluxo):
  Δ mão   = o quanto o fim do trecho da mão direita se desloca (cm)
  Δ fecha = o quanto o fechamento da mão direita muda (0..1; + = tapar faz FECHAR mais, - = faz ABRIR)
Onde tapar muda muito, o modelo depende daquilo. Diferente do mapa de atenção (por onde a informação passa), este diz
DE ONDE ela vem.

    cd ~/DEV/unifolm-wla-lora && CUDA_VISIBLE_DEVICES=2 python .../oclusao_v7.py --run playground/Checkpoints/prometheus_v7 \\
        --passo 8000 --quadros /data/mrwlker/wla_real_runs/20261029_142620:85 ... --saida /data/mrwlker/oclusao_v7
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
CAMS = [("observation.images.cam_left_high", "cabeça"), ("observation.images.cam_left_wrist", "punho esquerdo"),
        ("observation.images.cam_right_wrist", "punho direito")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--passo", type=int, required=True)
    ap.add_argument("--quadros", nargs="+", required=True, help="pasta_da_rodada:n_da_consulta[:rótulo]")
    ap.add_argument("--base", default="playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors")
    ap.add_argument("--lado", type=int, default=80, help="lado do quadrado (px, na imagem 640x480)")
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from replay_offline import obs_da_consulta, servidor

    s = servidor(a.base, a.run, a.passo)
    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)

    def acao(obs, semente):
        torch.manual_seed(semente)
        r = s.get_action(obs)
        return (np.asarray(r["action.right_ee_rpy"])[0][-1, :3],
                float(np.mean(np.asarray(r["action.right_fig6d"])[0][-1])))

    resumo = []
    for q in a.quadros:
        partes = q.split(":")
        pasta, n = Path(partes[0]), int(partes[1])
        rot = partes[2] if len(partes) > 2 else ""
        L = {json.loads(x)["n"]: json.loads(x) for x in open(pasta / "consultas.jsonl")}
        frase = L[n]["frase"]
        obs, _ = obs_da_consulta(pasta, f"{n:04d}", frase)
        sem = 1000 + n
        ee0, f0 = acao(obs, sem)
        fig, ax = plt.subplots(len(CAMS), 3, figsize=(15, 4 * len(CAMS)), facecolor="#0f1115")
        linha = {"rodada": pasta.name, "n": n, "frase": frase, "fecha_base": round(f0, 2)}
        for c, (chave, nome) in enumerate(CAMS):
            img = obs.get(chave)
            for k in range(3):
                ax[c, k].axis("off")
            if img is None:
                continue
            H, W = img.shape[:2]
            gh, gw = H // a.lado, W // a.lado
            dm, df = np.zeros((gh, gw)), np.zeros((gh, gw))
            for i in range(gh):
                for j in range(gw):
                    o = dict(obs)
                    t = img.copy()
                    t[i * a.lado:(i + 1) * a.lado, j * a.lado:(j + 1) * a.lado] = 127
                    o[chave] = t
                    ee, f = acao(o, sem)
                    dm[i, j] = np.linalg.norm(ee - ee0) * 100
                    df[i, j] = f - f0
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax[c, 0].imshow(rgb)
            ax[c, 0].set_title(f"{nome}", color="w", fontsize=13)
            ax[c, 1].imshow(rgb)
            h1 = ax[c, 1].imshow(cv2.resize(dm, (W, H), interpolation=cv2.INTER_NEAREST), cmap="inferno", alpha=0.6,
                                 vmin=0, vmax=max(2.0, dm.max()))
            ax[c, 1].set_title(f"Δ mão ao tapar (máx {dm.max():.1f} cm)", color="w", fontsize=13)
            cb = fig.colorbar(h1, ax=ax[c, 1], fraction=0.035)
            cb.ax.tick_params(colors="w")
            ax[c, 2].imshow(rgb)
            lim = max(0.1, float(np.abs(df).max()))
            h2 = ax[c, 2].imshow(cv2.resize(df, (W, H), interpolation=cv2.INTER_NEAREST), cmap="coolwarm", alpha=0.6,
                                 vmin=-lim, vmax=lim)
            ax[c, 2].set_title(f"Δ fechamento ao tapar (vermelho = fecha mais, azul = abre)", color="w", fontsize=11)
            cb = fig.colorbar(h2, ax=ax[c, 2], fraction=0.035)
            cb.ax.tick_params(colors="w")
            linha[f"{nome}_dmao_max_cm"] = round(float(dm.max()), 1)
            linha[f"{nome}_dfecha_max"] = round(float(np.abs(df).max()), 2)
        fig.suptitle(f"OCLUSÃO — {pasta.name} consulta #{n} {rot} | '{frase}' | WLA v7 @ {a.passo} | "
                     f"mão fechada (sem tapar) = {f0:.2f}", color="w", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        nome_png = saida / f"{pasta.name}_{n:04d}_{a.passo}.png"
        fig.savefig(nome_png, facecolor=fig.get_facecolor(), dpi=90)
        plt.close(fig)
        resumo.append(linha)
        print(f"✅ {nome_png.name}: {linha}", flush=True)
    json.dump(resumo, open(saida / f"resumo_{a.passo}.json", "w"), indent=1, ensure_ascii=False)
    print("FIM", flush=True)


if __name__ == "__main__":
    main()
