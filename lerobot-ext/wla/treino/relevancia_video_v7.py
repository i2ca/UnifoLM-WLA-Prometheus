#!/usr/bin/env python
"""VÍDEO DE RELEVÂNCIA CAUSAL do WLA v7 (30/10) nos episódios de VALIDAÇÃO: o que, em cada quadro, MUDA A AÇÃO.

O mapa de atenção do DiT mostrava "pontos aleatórios" (tokens de borda usados como memória — Darcet et al., 2024): onde
a ação olha não é de onde vem a informação. Aqui, por OCLUSÃO, em todos os quadros:
  - cada câmera é tapada por partes (quadrado cinza, uma parte por vez) e mede-se quanto a ação da mão direita muda;
  - cada PALAVRA da instrução é removida, uma por vez, e mede-se o mesmo;
  - todas as versões de um quadro vão num LOTE só, com o MESMO ruído do fluxo (a diferença vem só do que foi tirado).
Relevância = deslocamento médio da trajetória prevista da mão direita (cm) + 10 x mudança do fechamento (0..1)
(0,1 de fechamento ~ 1 cm). Abaixo de --limiar não é pintado (ruído).

    cd ~/DEV/unifolm-wla-lora && CUDA_VISIBLE_DEVICES=2 python .../relevancia_video_v7.py --run playground/Checkpoints/prometheus_v7 \\
        --passo 17000 --tarefa Pegar_Caneca --episodios 5 --saida /data/mrwlker/relevancia_v7 --video-unico pick_mug_v7.mp4
"""
import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parent))

TXT = {
    "en": {"titulo": "What drives the ACTION — causal relevance by occlusion", "ep": "VALIDATION episode (never seen in "
           "training)", "frame": "frame", "err": "hand error", "cams": {"head_left": "head (ZED)", "head_depth": "depth",
           "cam_wrist_left": "left wrist", "cam_wrist_right": "right wrist"}, "share": "of relevance",
           "frase": "instruction — each word removed alone (number = Δ action)", "escala": "Δ action (cm-equiv.)",
           "acao": "right hand: predicted vs demonstration", "fecha": "hand closure (0..1) + waist yaw (rad)",
           "pred": "predicted", "demo": "demonstration", "passo": "chunk step (30 = 1 s)", "desloc": "displacement (cm)",
           "off": "(off)", "rodape": "Relevance = how much the predicted right-hand action changes when that part of the "
           "input is removed (gray square over the image region, or the word deleted), all variants of the frame in one "
           "batch with the same flow noise. Units: mean displacement of the predicted trajectory (cm) + 10 x change of "
           "the hand closure. Regions below {lim} cm-equiv. are not painted. Circles = the 3 most relevant regions.",
           "card": "Episode {i} of {n} — validation (never seen in training)", "card2": "task: {t}",
           "card3": "{ds} #{ep} | WLA v7 @ {p} | mean hand error {e:.1f} cm"},
    "pt": {"titulo": "O que muda a AÇÃO — relevância causal por oclusão", "ep": "episódio de VALIDAÇÃO (nunca visto no "
           "treino)", "frame": "quadro", "err": "erro da mão", "cams": {"head_left": "cabeça (ZED)",
           "head_depth": "profundidade", "cam_wrist_left": "punho esquerdo", "cam_wrist_right": "punho direito"},
           "share": "da relevância", "frase": "instrução — cada palavra removida sozinha (número = Δ ação)",
           "escala": "Δ ação (cm-equiv.)", "acao": "mão direita: prevista x demonstração",
           "fecha": "fechamento da mão (0..1) + yaw da cintura (rad)", "pred": "prevista", "demo": "demonstração",
           "passo": "passo do trecho (30 = 1 s)", "desloc": "deslocamento (cm)", "off": "(desligada)",
           "rodape": "Relevância = quanto a ação prevista da mão direita muda quando aquela parte da entrada é retirada "
           "(quadrado cinza na região da imagem, ou a palavra apagada), todas as versões do quadro num lote com o mesmo "
           "ruído do fluxo. Unidade: deslocamento médio da trajetória prevista (cm) + 10 x mudança do fechamento. Abaixo "
           "de {lim} cm-equiv. não é pintado. Círculos = as 3 regiões mais relevantes.",
           "card": "Episódio {i} de {n} — validação (nunca visto no treino)", "card2": "tarefa: {t}",
           "card3": "{ds} #{ep} | WLA v7 @ {p} | erro médio da mão {e:.1f} cm"},
}
PAPEIS = ["head_left", "head_depth", "cam_wrist_left", "cam_wrist_right"]
GRADE = {"head_left": (6, 8), "head_depth": (4, 6), "cam_wrist_left": (4, 6), "cam_wrist_right": (4, 6)}   # (linhas, cols)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--passo", required=True)
    ap.add_argument("--split", default="/data/mrwlker/wla_v7/descricao/split.json")
    ap.add_argument("--dados", default="unifolm_wla/dataloader/multi_source_dataset/configs/prometheus_v7.yaml")
    ap.add_argument("--base", default="playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors")
    ap.add_argument("--tarefa", default=None)
    ap.add_argument("--episodios", type=int, default=5)
    ap.add_argument("--passo-quadros", type=int, default=8)
    ap.add_argument("--fps-video", type=int, default=4)
    ap.add_argument("--lote", type=int, default=40)
    ap.add_argument("--limiar", type=float, default=0.4, help="cm-equiv.: abaixo disso não pinta")
    ap.add_argument("--escala", type=float, default=4.0, help="cm-equiv. do topo da escala de cor")
    ap.add_argument("--lingua", choices=["pt", "en"], default="en")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--video-unico", default=None)
    a = ap.parse_args()
    T = TXT[a.lingua]

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    import wla_v2
    from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES
    from unifolm_wla.dataloader.multi_source_dataset.dataloader import collate_fn, create_training_dataloader
    from valida_wla import carrega

    wla_v2.instala_split(a.split, "val")
    _, ds = create_training_dataloader(a.dados, batch_size=1, num_workers=0, pin_memory=False, shuffle=False)
    m = carrega(a.base, Path(a.run), a.passo)
    frases = wla_v2.frases_do_run(Path(a.run))
    import treina_wla_v7
    from omegaconf import OmegaConf
    cfg_run = Path(a.run) / "config.full.yaml"
    rivais = treina_wla_v7._rivais_da_config(OmegaConf.load(cfg_run)) if cfg_run.exists() else {}
    H = m.action_horizon
    sx, fd, yw = SLICES["right_xyz_rotvec"], SLICES["right_fig6d"], SLICES["waist_joint"].start
    xyz = slice(sx.start, sx.start + 3)
    randn_orig = torch.randn

    def randn_igual(*args, **kw):   # o MESMO ruído em todas as linhas do lote
        size = kw.pop("size", None) or (args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args)
        size = tuple(size)
        return randn_orig((1,) + size[1:], **kw).expand(size).contiguous()

    def prediz(itens, semente):
        out = []
        for i in range(0, len(itens), a.lote):
            b = collate_fn(itens[i:i + a.lote])
            torch.manual_seed(semente)
            torch.randn = randn_igual
            try:
                with torch.no_grad():
                    out.append(m.predict_action(examples=b)["normalized_actions"])
            finally:
                torch.randn = randn_orig
        return np.concatenate(out, 0)

    escolhidos = []
    cum = np.array(ds._cum_lengths)
    for k, (td, d) in enumerate(zip(ds._task_dirs, ds._datasets)):
        if a.tarefa and a.tarefa not in Path(td).name:
            continue
        t = d.hf_dataset.data
        ep, fr = t.column("episode_index").to_numpy(), t.column("frame_index").to_numpy()
        for e in list(dict.fromkeys(ep.tolist()))[: a.episodios]:
            loc = np.where(ep == e)[0]
            loc = loc[np.argsort(fr[loc])][:: a.passo_quadros]
            escolhidos.append((Path(td).name, int(e), (cum[k - 1] if k > 0 else 0) + loc))
    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)
    cmap_p = plt.get_cmap("magma")
    videos = []
    for i_ep, (nome_ds, ep, idx) in enumerate(escolhidos, 1):
        pasta = saida / f"{nome_ds}_ep{ep}_{a.passo}_{a.lingua}"
        pasta.mkdir(exist_ok=True)
        errs = []
        print(f"▶ {nome_ds} #{ep}: {len(idx)} quadros", flush=True)
        for q, j in enumerate(idx):
            item = ds[int(j)]
            frase = frases.get(item["task"], item["task"])
            frase = frase[0] if isinstance(frase, list) else frase
            item["task"] = frase
            palavras = frase.split()
            itens, rotulos = [item], [("base",)]
            for papel in PAPEIS:
                img = item["images"].get(papel)
                if img is None or not bool(item["image_mask"].get(papel, False)):
                    continue
                gl, gc = GRADE[papel]
                _, Hh, Ww = img.shape
                for r in range(gl):
                    for c in range(gc):
                        t2 = img.clone()
                        t2[:, r * Hh // gl:(r + 1) * Hh // gl, c * Ww // gc:(c + 1) * Ww // gc] = 0.5
                        it = dict(item)
                        it["images"] = dict(item["images"])
                        it["images"][papel] = t2
                        itens.append(it)
                        rotulos.append(("img", papel, r, c))
            for w in range(len(palavras)):
                it = dict(item)
                it["task"] = " ".join(palavras[:w] + palavras[w + 1:])
                itens.append(it)
                rotulos.append(("pal", w))
            rival = rivais.get(frase)
            if rival:   # a instrução de OUTRA tarefa na mesma cena (o teste de linguagem que importa)
                it = dict(item)
                it["task"] = rival
                itens.append(it)
                rotulos.append(("rival",))
            P = prediz(itens, 1000 + q)
            esc = item["action_norm_scale"].numpy()
            base = P[0]
            traj0, f0 = base[:, xyz] * esc[xyz] * 100, (base[:, fd].mean(-1) + 1) / 2
            rel = np.zeros(len(P))
            for i in range(1, len(P)):
                dt = np.linalg.norm(P[i][:, xyz] * esc[xyz] * 100 - traj0, axis=-1).mean()
                dfe = np.abs((P[i][:, fd].mean(-1) + 1) / 2 - f0).mean()
                rel[i] = dt + 10 * dfe
            mapas = {p: np.zeros(GRADE[p]) for p in PAPEIS}
            pal = np.zeros(len(palavras))
            rel_rival = None
            for i, rot in enumerate(rotulos):
                if rot[0] == "rival":
                    rel_rival = rel[i]
                if rot[0] == "img":
                    mapas[rot[1]][rot[2], rot[3]] = rel[i]
                elif rot[0] == "pal":
                    pal[rot[1]] = rel[i]
            alvo = item["action"].numpy()[-H:]
            da = alvo[:, xyz] * esc[xyz] * 100
            err = float(np.linalg.norm(traj0 - da, axis=-1).mean())
            errs.append(err)
            tot = sum(v.sum() for v in mapas.values()) + 1e-9

            fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#0f1115")
            gs = gridspec.GridSpec(3, 4, height_ratios=[1.25, 0.3, 1.05], hspace=0.42, wspace=0.22,
                                   left=0.05, right=0.94, top=0.9, bottom=0.15)
            fig.text(0.05, 0.955, f"{T['titulo']}  —  WLA v7 @ {a.passo}", color="w", fontsize=20, weight="bold")
            fig.text(0.05, 0.925, f"{T['ep']}: {nome_ds} #{ep} | {T['frame']} {q * a.passo_quadros} "
                     f"({q * a.passo_quadros / 30:.1f} s) | {T['err']} {err:.1f} cm", color="#bbb", fontsize=13)
            for c, papel in enumerate(PAPEIS):
                ax = fig.add_subplot(gs[0, c])
                ax.axis("off")
                img = item["images"].get(papel)
                if img is None or not bool(item["image_mask"].get(papel, False)):
                    ax.set_title(f"{T['cams'][papel]} {T['off']}", color="#777", fontsize=13)
                    continue
                rgb = (img.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
                mp = mapas[papel]
                cal = cv2.resize(mp.astype(np.float32), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
                cal = np.ma.masked_less(cal, a.limiar)          # abaixo do limiar: transparente (é ruído)
                ax.imshow(rgb)
                ax.imshow(cal, cmap="inferno", alpha=0.55, vmin=0, vmax=a.escala)
                gl, gc = mp.shape
                for r_, ix in enumerate(np.argsort(mp.ravel())[::-1][:3], 1):
                    if mp.ravel()[ix] < a.limiar:
                        break
                    cy, cx = divmod(int(ix), gc)
                    ax.add_patch(plt.Rectangle((cx * rgb.shape[1] / gc, cy * rgb.shape[0] / gl), rgb.shape[1] / gc,
                                               rgb.shape[0] / gl, fill=False, color="#7CFC00", lw=2.5))
                    ax.text((cx + 0.12) * rgb.shape[1] / gc, (cy + 0.3) * rgb.shape[0] / gl, str(r_), color="#7CFC00",
                            fontsize=13, weight="bold")
                ax.set_title(f"{T['cams'][papel]} — {100 * mp.sum() / tot:.0f}% {T['share']}", color="w", fontsize=14)
            cax = fig.add_axes([0.952, 0.6, 0.007, 0.28])
            cb = fig.colorbar(plt.cm.ScalarMappable(cmap="inferno", norm=matplotlib.colors.Normalize(0, a.escala)),
                              cax=cax)
            cb.set_label(T["escala"], color="w", fontsize=11)
            cb.ax.tick_params(colors="w")
            # palavras
            axf = fig.add_subplot(gs[1, :])
            axf.axis("off")
            titulo_f = T["frase"]
            if rel_rival is not None:
                titulo_f += (f"   |   swapped to the OTHER task ('{rivais[frase][:45]}…'): Δ action = {rel_rival:.1f} cm-equiv."
                             if a.lingua == "en" else
                             f"   |   trocada pela da OUTRA tarefa ('{rivais[frase][:45]}…'): Δ ação = {rel_rival:.1f} cm-equiv.")
            axf.set_title(titulo_f, color="#ffd166" if rel_rival is not None else "#bbb", fontsize=13, loc="left")
            x = 0.0
            for w, v in zip(palavras, pal):
                t_ = axf.text(x, 0.2, w + " ", fontsize=22, color="w", transform=axf.transAxes,
                              bbox=dict(facecolor=cmap_p(0.12 + 0.85 * min(1.0, v / a.escala)), edgecolor="none", pad=4))
                fig.canvas.draw()
                x = t_.get_window_extent().transformed(axf.transAxes.inverted()).x1 + 0.006
                axf.text(x - 0.006, -0.45, f"{v:.1f}", fontsize=10, color="#999", transform=axf.transAxes, ha="right")
            # ação
            axa = fig.add_subplot(gs[2, 0:2])
            axa.set_facecolor("#16181d")
            for k_, (cor, nome) in enumerate(zip(["#ff6b6b", "#4dd4ac", "#5aa9ff"], "xyz")):
                axa.plot(traj0[:, k_], color=cor, lw=2.5, label=f"{nome} {T['pred']}")
                axa.plot(da[:, k_], color=cor, lw=1.5, ls="--", label=f"{nome} {T['demo']}")
            axa.set_title(T["acao"], color="w", fontsize=14)
            axa.set_xlabel(T["passo"], color="w")
            axa.set_ylabel(T["desloc"], color="w")
            axa.tick_params(colors="w")
            axa.legend(fontsize=10, ncol=3, labelcolor="w", loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
            axg = fig.add_subplot(gs[2, 2:4])
            axg.set_facecolor("#16181d")
            axg.plot(f0, color="#ffd166", lw=2.5, label=f"closure {T['pred']}" if a.lingua == "en" else f"fechamento {T['pred']}")
            axg.plot((alvo[:, fd].mean(-1) + 1) / 2, color="#ffd166", lw=1.5, ls="--", label=T["demo"])
            axg.plot(base[:, yw] * esc[yw], color="#c792ea", lw=2, label=f"yaw {T['pred']}")
            axg.plot(alvo[:, yw] * esc[yw], color="#c792ea", lw=1.2, ls="--", label=f"yaw {T['demo']}")
            axg.set_ylim(-0.6, 1.05)
            axg.set_title(T["fecha"], color="w", fontsize=14)
            axg.set_xlabel(T["passo"], color="w")
            axg.tick_params(colors="w")
            axg.legend(fontsize=10, ncol=4, labelcolor="w", loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
            fig.text(0.05, 0.008, "\n".join(textwrap.wrap(T["rodape"].format(lim=a.limiar), 215)), color="#9aa0a6",
                     fontsize=10.5)
            fig.savefig(pasta / f"{q:04d}.png", facecolor=fig.get_facecolor())
            plt.close(fig)
            if q % 10 == 0:
                top = {p: round(float(mapas[p].max()), 1) for p in PAPEIS}
                print(f"   {q}/{len(idx)} | erro {err:.1f} cm | máx por câmera {top} | palavra mais relevante "
                      f"'{palavras[int(pal.argmax())]}' {pal.max():.2f} | outra tarefa {rel_rival}", flush=True)
        mp4 = saida / f"{pasta.name}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps_video), "-i",
                        str(pasta / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(mp4)],
                       check=True)
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#0f1115")
        fig.text(0.5, 0.56, T["card"].format(i=i_ep, n=len(escolhidos)), color="w", fontsize=40, ha="center",
                 weight="bold")
        fig.text(0.5, 0.45, T["card2"].format(t=frase), color="#bbb", fontsize=24, ha="center")
        fig.text(0.5, 0.38, T["card3"].format(ds=nome_ds, ep=ep, p=a.passo, e=np.mean(errs)), color="#888",
                 fontsize=18, ha="center")
        card = pasta / "titulo.png"
        fig.savefig(card, facecolor=fig.get_facecolor())
        plt.close(fig)
        mp4c = saida / f"{pasta.name}_titulo.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", str(a.fps_video), "-i",
                        str(card), "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(mp4c)],
                       check=True)
        videos += [mp4c, mp4]
        print(f"🎬 {mp4} (erro médio {np.mean(errs):.1f} cm)", flush=True)
    if a.video_unico and videos:
        lista = saida / "lista.txt"
        lista.write_text("".join(f"file '{v}'\n" for v in videos))
        unico = saida / a.video_unico
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lista),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r", str(a.fps_video), str(unico)],
                       check=True)
        print(f"🎬🎬 {unico}", flush=True)
    print("FIM", flush=True)


if __name__ == "__main__":
    main()
