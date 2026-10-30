#!/usr/bin/env python
"""VÍDEO DA ATENÇÃO do WLA v7 (30/10) nos episódios de VALIDAÇÃO (que o treino não viu): para onde a cabeça de ação
(o DiT) olha a cada passo — em cada CÂMERA (mapa de calor), em cada PALAVRA da frase, no ESTADO do robô — camada por
camada, e a AÇÃO prevista contra a demonstração. Um quadro de vídeo por consulta (a cada --passo-quadros quadros).

Como é medido (o mesmo do mapa_atencao.py): no DiT, os tokens de ação fazem atenção conjunta sobre [ação, saída do VLM].
Envolvemos dispatch_attention_fn e, em cada chamada (cada camada x passo do fluxo), calculamos softmax(q_ação · k) e
somamos a massa que cai em cada token do VLM (média sobre cabeças e tokens de ação). Os tokens são separados em:
imagem de cada câmera (pelo image_token_id + image_grid_thw), palavras da frase (entre "Task:" e "State:"), estado
(<|robot_state|>) e o resto do prompt (rótulos, "Control Mode"...).

    cd ~/DEV/unifolm-wla-lora && CUDA_VISIBLE_DEVICES=2 python .../atencao_video_v7.py \\
        --run playground/Checkpoints/prometheus_v7 --passo 16000 --saida ~/atencao_v7 [--episodios 1] [--lingua en]
"""
import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parent))

TXT = {
    "pt": {"cams": {"head_left": "cabeça (ZED)", "head_depth": "profundidade", "cam_wrist_left": "punho esquerdo",
                    "cam_wrist_right": "punho direito"},
           "cat": ["cabeça", "profund.", "punho E", "punho D", "frase", "estado", "resto"],
           "titulo": "Para onde a AÇÃO olha (atenção do DiT sobre o VLM)", "frase": "frase (cor = atenção)",
           "camadas": "camada do DiT", "parte": "parte da entrada", "acao": "ação: mão direita (prevista x demonstração)",
           "fecha": "mão fechada (0..1)", "pred": "prevista", "demo": "demonstração", "passo": "passo do trecho (30 = 1 s)",
           "desloc": "deslocamento (cm)", "ep": "episódio de VALIDAÇÃO", "quadro": "quadro", "share": "atenção total",
           "desligada": "(desligada)", "da_atencao": "da atenção", "escala": "atenção da região (1× = média da câmera)",
           "media": "(média)",
           "rodape": "Mapa de calor: quanto os tokens de AÇÃO do DiT consultam cada região da imagem (saída do VLM), em vezes "
                     "a média da câmera; círculos 1-5 = regiões mais consultadas; % = parcela da atenção total. Frase: "
                     "cor = atenção de cada palavra. Matriz: atenção por camada do DiT e por parte da entrada (média dos "
                     "4 passos do fluxo). Regiões de BORDA muito consultadas costumam ser tokens que o transformer usa como "
                     "memória/registrador (Darcet et al., 2024) — onde a ação olha não é necessariamente de onde vem a "
                     "informação; o teste de oclusão mede isso. Episódio de validação, nunca visto no treino.",
           "card": "Episódio {i} de {n} — validação (nunca visto no treino)", "card2": "tarefa: {t}"},
    "en": {"cams": {"head_left": "head (ZED)", "head_depth": "depth", "cam_wrist_left": "left wrist",
                    "cam_wrist_right": "right wrist"},
           "cat": ["head", "depth", "wrist L", "wrist R", "instruction", "state", "other"],
           "titulo": "Where the ACTION looks (DiT attention over the VLM)", "frase": "instruction (color = attention)",
           "camadas": "DiT layer", "parte": "input part", "acao": "action: right hand (predicted vs demonstration)",
           "fecha": "hand closed (0..1)", "pred": "predicted", "demo": "demonstration", "passo": "chunk step (30 = 1 s)",
           "desloc": "displacement (cm)", "ep": "VALIDATION episode", "quadro": "frame", "share": "total attention",
           "desligada": "(off)", "da_atencao": "of attention", "escala": "region attention (1× = camera mean)",
           "media": "(mean)",
           "rodape": "Heat map: how much the DiT ACTION tokens attend to each image region (VLM output), in multiples of "
                     "the camera mean; circles 1-5 = most attended regions; % = share of total attention. Instruction: "
                     "color = attention per word. Matrix: attention per DiT layer and input part (mean over the 4 flow "
                     "steps). Highly attended BORDER regions are often tokens the transformer uses as memory/registers "
                     "(Darcet et al., 2024) — where the action looks is not necessarily where the information comes "
                     "from; the occlusion test measures that. Validation episode, never seen in training.",
           "card": "Episode {i} of {n} — validation (never seen in training)", "card2": "task: {t}"},
}
PAPEIS = ["head_left", "head_depth", "cam_wrist_left", "cam_wrist_right"]
ESCALA_X = 4.0   # o mapa vai de 0 a 4x a média da câmera (fixo em todos os quadros)


class Espiao:
    """Guarda a massa de atenção ação -> cada token do VLM, por chamada (camada x passo do fluxo)."""
    def __init__(self):
        self.L, self.chamadas, self.ligado = 0, [], False

    def instala(self):
        from unifolm_wla.model.modules.action_model.DiT_modules import mmdit
        orig = mmdit.dispatch_attention_fn
        esp = self

        def espiao(q, k, v, attn_mask=None, **kw):
            if esp.ligado:
                B, S, H, D = q.shape
                A = S - esp.L
                if A > 0:   # camadas de self-atenção só da ação (interleave) não olham o VLM
                    with torch.no_grad():
                        lg = torch.einsum("bqhd,bkhd->bhqk", q[:1, :A].float(), k[:1].float()) / math.sqrt(D)
                        if attn_mask is not None:
                            lg = lg.masked_fill(~attn_mask[:1].reshape(1, 1, 1, S).bool(), float("-inf"))
                        p = lg.softmax(-1)[0, :, :, A:].mean(dim=(0, 1))
                    esp.chamadas.append(p.cpu().numpy())
            return orig(q, k, v, attn_mask=attn_mask, **kw)

        mmdit.dispatch_attention_fn = espiao


def segmentos(inp, tok, roles):
    """Índices dos tokens de cada parte: {papel: (posições, grade)}, frase [(posições, palavra)], estado."""
    ids = inp["input_ids"][0].cpu().numpy()
    img_tok = tok.convert_tokens_to_ids("<|image_pad|>")
    pos_img = np.where(ids == img_tok)[0]
    cams, ini = {}, 0
    for papel, g in zip(roles, inp["image_grid_thw"].cpu().numpy()):
        t, h, w = (int(x) for x in g)
        n = t * (h // 2) * (w // 2)
        cams[papel] = (pos_img[ini:ini + n], (h // 2, w // 2))
        ini += n
    st = set(tok.convert_tokens_to_ids(x) for x in ("<|robot_state|>", "<|robot_state_implicit_stats|>"))
    pos_estado = np.array([i for i, x in enumerate(ids) if int(x) in st])
    # frase: tokens entre "Task:" e "State:" (decodificando token a token)
    pecas = [tok.decode([int(x)]) for x in ids]
    acum, ini_f, fim_f = "", None, None
    for i, p in enumerate(pecas):
        acum += p
        if ini_f is None and acum.endswith("Task:"):
            ini_f = i + 1
        elif ini_f is not None and fim_f is None and acum.rstrip().endswith("State:"):
            fim_f = i
            break
    frase = []
    if ini_f is not None and fim_f is not None:
        # junta sub-palavras em palavras (o token que começa com espaço abre palavra nova)
        atual = None
        for i in range(ini_f, fim_f):
            p = pecas[i]
            if p.strip() == "" or p.strip() == "State":
                continue
            if atual is None or p.startswith(" "):
                atual = ([i], p.strip())
                frase.append(atual)
            else:
                atual[0].append(i)
                frase[-1] = (atual[0], atual[1] + p.strip())
                atual = frase[-1]
    return cams, frase, pos_estado


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--passo", required=True)
    ap.add_argument("--split", default="/data/mrwlker/wla_v7/descricao/split.json")
    ap.add_argument("--dados", default="unifolm_wla/dataloader/multi_source_dataset/configs/prometheus_v7.yaml")
    ap.add_argument("--base", default="playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors")
    ap.add_argument("--episodios", type=int, default=1, help="episódios de validação por tarefa")
    ap.add_argument("--passo-quadros", type=int, default=3, help="uma consulta a cada N quadros (30 fps)")
    ap.add_argument("--fps-video", type=int, default=10)
    ap.add_argument("--lingua", choices=["pt", "en"], default="pt")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--tarefa", default=None, help="só os datasets cujo nome contém isto (ex.: Pegar_Caneca)")
    ap.add_argument("--video-unico", default=None, help="junta todos os episódios num vídeo só com este nome")
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

    esp = Espiao()
    esp.instala()
    wla_v2.instala_split(a.split, "val")
    _, ds = create_training_dataloader(a.dados, batch_size=1, num_workers=0, pin_memory=False, shuffle=False)
    m = carrega(a.base, Path(a.run), a.passo)
    iface = m.qwen_vl_interface
    tok = iface.processor.tokenizer
    guardado = {}
    orig_build = iface.build_qwenvl_inputs

    def build(*x, **kw):
        out = orig_build(*x, **kw)
        guardado["inp"], guardado["roles"] = out, (kw.get("image_roles") or [None])[0]
        esp.L = out["input_ids"].shape[1]
        return out

    iface.build_qwenvl_inputs = build
    frases = wla_v2.frases_do_run(Path(a.run))
    H = m.action_horizon
    sx, fd, yw = SLICES["right_xyz_rotvec"], SLICES["right_fig6d"], SLICES["waist_joint"].start

    # episódios de validação por tarefa (o dataset de dentro: um sub-dataset por pasta)
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
    cmap = plt.get_cmap("magma")

    videos = []
    for i_ep, (nome_ds, ep, idx) in enumerate(escolhidos, 1):
        pasta = saida / f"{nome_ds}_ep{ep}_{a.passo}_{a.lingua}"
        pasta.mkdir(exist_ok=True)
        hist_err = []
        print(f"▶ {nome_ds} episódio {ep}: {len(idx)} consultas", flush=True)
        for q, j in enumerate(idx):
            item = ds[int(j)]
            b = collate_fn([item])
            b["task"] = [frases.get(t, t) for t in b["task"]]
            if isinstance(b["task"][0], list):
                b["task"] = [b["task"][0][0]]
            esp.chamadas, esp.ligado = [], True
            torch.manual_seed(1000 + q)
            with torch.no_grad():
                pred = m.predict_action(examples=b)["normalized_actions"][0]
            esp.ligado = False
            alvo = b["action"][0].cpu().numpy()[-H:]
            esc = b["action_norm_scale"][0].cpu().numpy()
            inp, roles = guardado["inp"], guardado["roles"] or [r for r in PAPEIS if b["image_mask"].get(r, [False])[0]]
            C = np.stack(esp.chamadas)                         # (chamadas, L)
            cams, frase, pos_estado = segmentos(inp, tok, roles)
            n_cam = {r: C[:, p].sum(1) for r, (p, _) in cams.items()}
            n_fr = C[:, [i for ps, _ in frase for i in ps]].sum(1) if frase else np.zeros(len(C))
            n_st = C[:, pos_estado].sum(1) if len(pos_estado) else np.zeros(len(C))
            cat = np.stack([n_cam.get(r, np.zeros(len(C))) for r in PAPEIS] + [n_fr, n_st,
                           1 - sum(n_cam.values()) - n_fr - n_st], 1)   # (chamadas, 7)
            # camadas: as chamadas vêm camada a camada (só as que olham o VLM; as de self-atenção da ação ficam
            # de fora), repetidas a cada passo do fluxo -> média sobre os passos
            nfl = int(getattr(m.action_model, "num_inference_timesteps", 4) or 4)
            por_passo = max(1, len(cat) // nfl)
            por_camada = cat[: por_passo * nfl].reshape(nfl, por_passo, 7).mean(0)
            media = C.mean(0)
            xyz = slice(sx.start, sx.start + 3)
            dp = pred[:, xyz] * esc[xyz] * 100
            da = alvo[:, xyz] * esc[xyz] * 100
            err = float(np.linalg.norm(dp - da, axis=-1).mean())
            hist_err.append(err)

            fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#0f1115")
            gs = gridspec.GridSpec(3, 4, height_ratios=[1.25, 0.3, 1.05], hspace=0.42, wspace=0.22,
                                   left=0.075, right=0.94, top=0.9, bottom=0.15)
            frase_txt = b["task"][0]
            fig.text(0.03, 0.955, f"{T['titulo']}  —  WLA v7 @ {a.passo}", color="w", fontsize=20, weight="bold")
            fig.text(0.03, 0.925, f"{T['ep']}: {nome_ds} #{ep} | {T['quadro']} {q * a.passo_quadros} "
                     f"({q * a.passo_quadros / 30:.1f} s) | erro mão {err:.1f} cm", color="#bbb", fontsize=13)
            # 1. câmeras — escala FIXA: atenção de cada região / média da câmera (1x = média), 0..ESCALA_X
            for c, papel in enumerate(PAPEIS):
                ax = fig.add_subplot(gs[0, c])
                ax.axis("off")
                img = item["images"].get(papel)
                if img is None or papel not in cams:
                    ax.set_title(f"{T['cams'][papel]} — {T['desligada']}", color="#777", fontsize=13)
                    continue
                rgb = (img.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
                p, (gh, gw) = cams[papel]
                mapa = media[p].reshape(gh, gw)
                rel = mapa / (mapa.mean() + 1e-12)
                calor = cv2.resize(np.clip(rel / ESCALA_X, 0, 1).astype(np.float32), (rgb.shape[1], rgb.shape[0]),
                                   interpolation=cv2.INTER_CUBIC)
                ax.imshow(rgb)
                ax.imshow(calor, cmap="jet", alpha=0.45, vmin=0, vmax=1)
                for r_, idx_ in enumerate(np.argsort(rel.ravel())[::-1][:5], 1):   # as 5 regiões mais consultadas
                    cy, cx = divmod(int(idx_), gw)
                    px, py = (cx + 0.5) * rgb.shape[1] / gw, (cy + 0.5) * rgb.shape[0] / gh
                    ax.add_patch(plt.Circle((px, py), 0.06 * rgb.shape[1], fill=False, color="w", lw=2))
                    ax.text(px, py, str(r_), color="w", fontsize=11, weight="bold", ha="center", va="center")
                ax.set_title(f"{T['cams'][papel]} — {100 * float(cat[:, c].mean()):.0f}% {T['da_atencao']}",
                             color="w", fontsize=14)
            cax = fig.add_axes([0.952, 0.6, 0.007, 0.28])
            cb = fig.colorbar(plt.cm.ScalarMappable(cmap="jet", norm=matplotlib.colors.Normalize(0, ESCALA_X)), cax=cax)
            cb.set_label(T["escala"], color="w", fontsize=11)
            cb.ax.tick_params(colors="w")
            cb.set_ticks([0, 1, 2, 3, 4])
            cb.set_ticklabels(["0", "1×", "2×", "3×", "4×+"])
            # 2. frase colorida pela atenção
            axf = fig.add_subplot(gs[1, :])
            axf.axis("off")
            axf.set_title(T["frase"], color="#bbb", fontsize=13, loc="left")
            if frase:
                pesos = np.array([media[ps].sum() for ps, _ in frase])
                pesos = pesos / (pesos.max() + 1e-9)
                x = 0.0
                for (ps, w), pw in zip(frase, pesos):
                    t_ = axf.text(x, 0.2, w + " ", fontsize=22, color="w", transform=axf.transAxes,
                                  bbox=dict(facecolor=cmap(0.15 + 0.8 * pw), edgecolor="none", pad=4, alpha=0.95))
                    fig.canvas.draw()
                    bb = t_.get_window_extent().transformed(axf.transAxes.inverted())
                    x = bb.x1 + 0.006
            else:
                axf.text(0, 0.35, frase_txt, fontsize=20, color="w", transform=axf.transAxes)
            # 3. camada x parte
            axh = fig.add_subplot(gs[2, 0:2])
            im = axh.imshow(por_camada.T * 100, aspect="auto", cmap="magma", vmin=0)
            axh.set_yticks(range(7), T["cat"], color="w", fontsize=12)
            axh.set_xticks(range(por_camada.shape[0]), [str(i + 1) for i in range(por_camada.shape[0])], color="w")
            axh.set_xlabel(T["camadas"], color="w", fontsize=13)
            axh.set_title(f"{T['share']} (%) — {T['parte']} x {T['camadas']}", color="w", fontsize=14)
            cb = fig.colorbar(im, ax=axh, fraction=0.025)
            cb.ax.tick_params(colors="w")
            # 4. ação
            axa = fig.add_subplot(gs[2, 2])
            axa.set_facecolor("#16181d")
            for k_, (cor, nome) in enumerate(zip(["#ff6b6b", "#4dd4ac", "#5aa9ff"], "xyz")):
                axa.plot(dp[:, k_], color=cor, lw=2.5, label=f"{nome} {T['pred']}")
                axa.plot(da[:, k_], color=cor, lw=1.5, ls="--", label=f"{nome} {T['demo']}")
            axa.set_title(T["acao"], color="w", fontsize=14)
            axa.set_xlabel(T["passo"], color="w")
            axa.set_ylabel(T["desloc"], color="w")
            axa.tick_params(colors="w")
            axa.legend(fontsize=9, ncol=3, facecolor="#16181d", labelcolor="w", loc="upper center",
                       bbox_to_anchor=(0.5, -0.2), frameon=False)
            axg = fig.add_subplot(gs[2, 3])
            axg.set_facecolor("#16181d")
            axg.plot((pred[:, fd].mean(-1) + 1) / 2, color="#ffd166", lw=2.5, label=T["pred"])
            axg.plot((alvo[:, fd].mean(-1) + 1) / 2, color="#ffd166", lw=1.5, ls="--", label=T["demo"])
            axg.plot(pred[:, yw] * esc[yw], color="#c792ea", lw=2, label="yaw (rad) " + T["pred"])
            axg.plot(alvo[:, yw] * esc[yw], color="#c792ea", lw=1.2, ls="--", label="yaw (rad) " + T["demo"])
            axg.set_ylim(-0.6, 1.05)
            axg.set_title(T["fecha"] + " + yaw", color="w", fontsize=14)
            axg.set_xlabel(T["passo"], color="w")
            axg.tick_params(colors="w")
            axg.legend(fontsize=9, ncol=2, facecolor="#16181d", labelcolor="w", loc="upper center",
                       bbox_to_anchor=(0.5, -0.2), frameon=False)
            import textwrap
            fig.text(0.075, 0.008, "\n".join(textwrap.wrap(T["rodape"], 205)), color="#9aa0a6", fontsize=10.5)
            fig.savefig(pasta / f"{q:04d}.png", facecolor=fig.get_facecolor())
            plt.close(fig)
            if q == 0:
                print(f"   {len(C)} chamadas de atenção ação->VLM por consulta ({nfl} passos do fluxo x {por_passo} "
                      f"camadas) | frase: {[w for _, w in frase]} | estado: {len(pos_estado)} tokens", flush=True)
            if q % 20 == 0:
                print(f"   {q}/{len(idx)} | erro {err:.1f} cm | atenção câmeras {np.round(100 * cat[:, :4].mean(0), 0)} "
                      f"frase {100 * cat[:, 4].mean():.0f}% estado {100 * cat[:, 5].mean():.1f}%", flush=True)
        mp4 = saida / f"{pasta.name}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps_video), "-i",
                        str(pasta / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(mp4)],
                       check=True)
        print(f"🎬 {mp4} ({len(idx)} quadros, erro médio da mão {np.mean(hist_err):.1f} cm)", flush=True)
        # tela de título deste episódio (2 s)
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#0f1115")
        fig.text(0.5, 0.56, T["card"].format(i=i_ep, n=len(escolhidos)), color="w", fontsize=40, ha="center",
                 weight="bold")
        fig.text(0.5, 0.45, T["card2"].format(t=frase_txt), color="#bbb", fontsize=24, ha="center")
        fig.text(0.5, 0.38, f"{nome_ds} #{ep} | WLA v7 @ {a.passo} | erro médio da mão {np.mean(hist_err):.1f} cm",
                 color="#888", fontsize=18, ha="center")
        card = pasta / "titulo.png"
        fig.savefig(card, facecolor=fig.get_facecolor())
        plt.close(fig)
        mp4c = saida / f"{pasta.name}_titulo.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", str(a.fps_video), "-i",
                        str(card), "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(mp4c)],
                       check=True)
        videos += [mp4c, mp4]
    if a.video_unico and videos:
        lista = saida / "lista.txt"
        lista.write_text("".join(f"file '{v}'\n" for v in videos))
        unico = saida / a.video_unico
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lista),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r", str(a.fps_video), str(unico)],
                       check=True)
        print(f"🎬🎬 vídeo único: {unico}", flush=True)
    print("FIM", flush=True)


if __name__ == "__main__":
    main()
