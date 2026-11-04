#!/usr/bin/env python
"""VÍDEO: o que a FRASE faz o braço planejar (31/10). Consultas que o robô gravou de verdade (gravador_wla.py), a mesma
inferência do servidor e a mesma semente; só a frase muda. Em cada quadro, para cada modelo (linha), a imagem da
cabeça com o CAMINHO planejado da mão direita (1 s, projetado com a calibração da ZED, servo_camera.json) para cada
frase, e ao lado quanto a mão (cm) e a cintura (yaw, graus) mudam em relação à frase certa.

    cd ~/DEV/unifolm-wla-lora && CUDA_VISIBLE_DEVICES=2 python .../video_frase.py \\
        --modelos playground/Checkpoints/prometheus_v7:20000 playground/Checkpoints/prometheus_v8:1000 \\
        --rodadas /data/mrwlker/wla_real_runs/20261029_142620 /data/mrwlker/wla_real_runs/20261029_144233 \\
        --camera ~/servo_camera.json --saida /data/mrwlker/video_frase
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from compara_frase import OUTRA, PT  # noqa: E402

VARS = [("correct", lambda f: f, "#ffffff"),
        ("other task", lambda f: OUTRA.get(f, "Pick up the apple."), "#ff4d4d"),
        ("empty", lambda f: "", "#ffd23f"),
        ("nonsense", lambda f: "Dance with the blue chair next to the window.", "#b07cff"),
        ("Portuguese", lambda f: PT.get(f, f), "#3ecf8e"),
        ("unseen task", lambda f: "Pick up the coffee scoop and put it in the jar.", "#4fa3ff")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelos", nargs="+", required=True, help="run:passo (uma linha do vídeo cada)")
    ap.add_argument("--rodadas", nargs="+", required=True)
    ap.add_argument("--cada", type=int, default=6)
    ap.add_argument("--camera", default=str(Path.home() / "servo_camera.json"))
    ap.add_argument("--base", default="playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors")
    ap.add_argument("--fps-video", type=float, default=1.0)
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from fk_g1 import FK
    from replay_offline import obs_da_consulta, servidor
    from servo_er1 import Camera, torso_na_pelvis

    saida = Path(a.saida)
    (saida / "quadros").mkdir(parents=True, exist_ok=True)
    cam, fk = Camera.carrega(a.camera), FK()
    consultas = []
    for r in a.rodadas:
        p = Path(r)
        L = [json.loads(x) for x in open(p / "consultas.jsonl")]
        consultas += [(p, x) for x in L[::a.cada] if (p / f"{x['n']:04d}.npz").exists()]

    # 1) inferência: caminho da mão (30 passos) e yaw, para cada modelo x consulta x frase
    R = {}
    for m in a.modelos:
        run, passo = m.rsplit(":", 1)
        s = servidor(a.base, run, int(passo))
        R[m] = []
        for p, x in consultas:
            obs, z = obs_da_consulta(p, f"{x['n']:04d}", x["frase"])
            out = {}
            for nome, f, _ in VARS:
                o = dict(obs)
                o["instruction"] = f(x["frase"])
                torch.manual_seed(1000 + x["n"])
                r = s.get_action(o)
                out[nome] = (np.asarray(r["action.right_ee_rpy"])[0][:, :3],
                             float(np.asarray(r["action.lower_body"])[0][-1, 12]))
            R[m].append(out)
        print(f"✅ {m}: {len(consultas)} consultas x {len(VARS)} frases", flush=True)
        del s
        torch.cuda.empty_cache()

    # 2) quadros
    nm = {m: f"{Path(m.rsplit(':', 1)[0]).name.replace('prometheus_', 'WLA ')} @ {m.rsplit(':', 1)[1]}"
          for m in a.modelos}
    for k, (p, x) in enumerate(consultas):
        img = cv2.cvtColor(cv2.imread(str(p / f"{x['n']:04d}_cabeca.jpg")), cv2.COLOR_BGR2RGB)
        z = np.load(p / f"{x['n']:04d}.npz")
        R_tp, p_tp = torso_na_pelvis(fk, z["q_medido"])
        mao0 = z["ee_medido_right"][:3]
        fig, ax = plt.subplots(len(a.modelos), 2, figsize=(16, 5.6 * len(a.modelos)), facecolor="#0f1115",
                               gridspec_kw={"width_ratios": [1.45, 1]}, squeeze=False)
        for i, m in enumerate(a.modelos):
            out = R[m][k]
            a0, a1 = ax[i, 0], ax[i, 1]
            a0.imshow(img)
            a0.axis("off")
            u0 = cam.projeta(mao0, R_tp, p_tp)
            todos = [u0]
            for nome, _, cor in VARS:
                traj = np.array([cam.projeta(P, R_tp, p_tp) for P in out[nome][0]])
                todos += list(traj)
                a0.plot(traj[:, 0], traj[:, 1], "-", color="k", lw=6 if nome == "correct" else 5, alpha=0.6,
                        zorder=3)   # contorno
                a0.plot(traj[:, 0], traj[:, 1], "-", color=cor, lw=4 if nome == "correct" else 3,
                        alpha=0.95, zorder=5 if nome == "correct" else 4)
                a0.plot(traj[-1, 0], traj[-1, 1], "o", color=cor, ms=11, mec="k", mew=1.5, zorder=6)
            a0.plot(u0[0], u0[1], "s", color="w", ms=11, mec="k", mew=1.5, zorder=7)
            # ZOOM em volta da mão e dos caminhos (1 s de movimento é pouco na imagem inteira)
            P = np.array(todos)
            cx, cy = np.clip(P.mean(0), 0, [img.shape[1], img.shape[0]])
            meia_l = max(110, (P[:, 0].max() - P[:, 0].min()) / 2 + 45)
            meia_a = meia_l * 0.75
            a0.set_xlim(cx - meia_l, cx + meia_l)
            a0.set_ylim(cy + meia_a, cy - meia_a)
            a0.set_title(f"{nm[m]} — planned right-hand path (1 s) per instruction (zoom on the hand)", color="w",
                         fontsize=13)
            ref, yref = out["correct"]
            nomes = [n for n, _, _ in VARS[1:]]
            dm = [np.linalg.norm(out[n][0][-1] - ref[-1]) * 100 for n in nomes]
            dy = [abs(np.degrees(out[n][1] - yref)) for n in nomes]
            y = np.arange(len(nomes))
            cores = [c for _, _, c in VARS[1:]]
            a1.barh(y - 0.2, dm, 0.38, color=cores, label="Δ hand (cm)")
            a1.barh(y + 0.2, [d / 5 for d in dy], 0.38, color=cores, alpha=0.45, hatch="//", label="Δ waist yaw (°/5)")
            for yy, v1, v2 in zip(y, dm, dy):
                a1.text(v1 + 0.1, yy - 0.2, f"{v1:.1f} cm", va="center", color="w", fontsize=10)
                a1.text(v2 / 5 + 0.1, yy + 0.2, f"{v2:.0f}°", va="center", color="#bbb", fontsize=10)
            a1.set_yticks(y, nomes, color="w")
            a1.invert_yaxis()
            a1.set_xlim(0, max(8, max(dm + [d / 5 for d in dy]) * 1.25))
            a1.set_facecolor("#0f1115")
            a1.tick_params(colors="w")
            for sp in a1.spines.values():
                sp.set_color("#444")
            a1.set_title("change vs. the correct instruction", color="w", fontsize=12)
            if i == 0:
                a1.legend(facecolor="#222", labelcolor="w", fontsize=9, loc="lower right")
        leg = " · ".join(f"{n}" for n, _, _ in VARS)
        fig.suptitle(f"Real robot query {k + 1}/{len(consultas)} ({p.name} #{x['n']}) — instruction given: "
                     f"\"{x['frase']}\"", color="w", fontsize=15)
        fig.text(0.5, 0.005, "path colors: " + " | ".join(
            f"{n} = {c}" for n, _, c in VARS).replace("#ffffff", "white").replace("#ff4d4d", "red").replace(
            "#ffd23f", "yellow").replace("#b07cff", "purple").replace("#3ecf8e", "green").replace("#4fa3ff", "blue")
            + "  ·  white square = hand now", ha="center", color="#aaa", fontsize=11)
        fig.tight_layout(rect=(0, 0.02, 1, 0.95))
        fig.savefig(saida / "quadros" / f"{k:04d}.png", facecolor=fig.get_facecolor(), dpi=80)
        plt.close(fig)

    # 3) resumo (último quadro) e vídeo
    fig, axr = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0f1115")
    nomes = [n for n, _, _ in VARS[1:]]
    for j, (chave, rot) in enumerate((("m", "Δ hand (cm)"), ("y", "Δ waist yaw (°)"))):
        w = 0.8 / len(a.modelos)
        for i, m in enumerate(a.modelos):
            vals = []
            for n in nomes:
                if chave == "m":
                    v = [np.linalg.norm(o[n][0][-1] - o["correct"][0][-1]) * 100 for o in R[m]]
                else:
                    v = [abs(np.degrees(o[n][1] - o["correct"][1])) for o in R[m]]
                vals.append(np.mean(v))
            xs = np.arange(len(nomes)) + i * w
            axr[j].bar(xs, vals, w, label=nm[m])
            for xx, vv in zip(xs, vals):
                axr[j].text(xx, vv, f"{vv:.1f}", ha="center", va="bottom", color="w", fontsize=10)
        axr[j].set_xticks(np.arange(len(nomes)) + w * (len(a.modelos) - 1) / 2, nomes, color="w")
        axr[j].set_title(f"mean {rot} over {len(consultas)} real queries", color="w", fontsize=13)
        axr[j].set_facecolor("#0f1115")
        axr[j].tick_params(colors="w")
        axr[j].legend(facecolor="#222", labelcolor="w")
    fig.suptitle("How much the instruction changes the plan (same image, same state, same noise)", color="w",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for r in range(4):   # o resumo fica 4 s na tela
        fig.savefig(saida / "quadros" / f"{len(consultas) + r:04d}.png", facecolor=fig.get_facecolor(), dpi=80)
    plt.close(fig)
    video = saida / "instruction_effect.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(a.fps_video), "-i",
                    str(saida / "quadros" / "%04d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=25",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(video)], check=True)
    print(f"🎬 {video}", flush=True)


if __name__ == "__main__":
    main()
