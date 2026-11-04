#!/usr/bin/env python
"""MESMA CENA, OUTRA FRASE (31/10): um quadro FIXO da câmera da cabeça com todos os objetos da mesa do café, e a frase
trocada a cada ~5 s — inclusive objetos que NUNCA foram tarefa no treino (chaleira, pote, tampa). Para cada frase e
cada modelo (linha):
  - RELEVÂNCIA causal na imagem da cabeça: cada pedaço (grade 6x8) é tapado e mede-se quanto o plano muda (mesmo
    ruído do fluxo em todos) -> onde ele está olhando PARA ESTA frase;
  - o CAMINHO planejado da mão direita (1 s), projetado na imagem (pose da mão do dataset + ação relativa à mão +
    calibração da ZED servo_camera.json).

    cd ~/DEV/unifolm-wla-lora && CUDA_VISIBLE_DEVICES=2 python .../atencao_tarefas.py \\
        --modelos playground/Checkpoints/prometheus_v7:20000 playground/Checkpoints/prometheus_v8:2000 \\
        --saida /data/mrwlker/video_tarefas
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))

FRASES = [
    ("Pick up the white mug.", "trained"),
    ("Pick up the apple.", "trained"),
    ("Place the white mug under the coffee strainer.", "trained"),
    ("Pick up the kettle.", "never a task in our data"),
    ("Pick up the brown coffee jar.", "never a task"),
    ("Pick up the black lid.", "never a task"),
    ("Pegue a maçã.", "trained task, in Portuguese"),
]
GRADE = (6, 8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelos", nargs="+", required=True, help="run:passo")
    ap.add_argument("--dados", default="unifolm_wla/dataloader/multi_source_dataset/configs/prometheus_v7.yaml")
    ap.add_argument("--base", default="playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors")
    ap.add_argument("--tarefa", default="Pegar_Caneca_Mesa_Nova")
    ap.add_argument("--episodio", type=int, default=0)
    ap.add_argument("--quadro", type=int, default=90, help="quadro do episódio (90 = 3 s: a mão já na borda da mesa, "
                    "indo; no início 1 s de plano é só levantar o braço)")
    ap.add_argument("--camera", default=str(Path.home() / "servo_camera.json"))
    ap.add_argument("--segundos", type=float, default=5.0)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation

    import wla_v2
    from fk_g1 import FK
    from servo_er1 import Camera, torso_na_pelvis
    from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES
    from unifolm_wla.dataloader.multi_source_dataset.dataloader import collate_fn, create_training_dataloader
    from valida_wla import carrega

    _, ds = create_training_dataloader(a.dados, batch_size=1, num_workers=0, pin_memory=False, shuffle=False)
    cum = np.array(ds._cum_lengths)
    j = None
    for k, (td, d) in enumerate(zip(ds._task_dirs, ds._datasets)):
        if a.tarefa in Path(td).name:
            t = d.hf_dataset.data
            ep, fr = t.column("episode_index").to_numpy(), t.column("frame_index").to_numpy()
            loc = int(np.where((ep == a.episodio) & (fr == a.quadro))[0][0])
            j = (cum[k - 1] if k > 0 else 0) + loc
            linha = d.hf_dataset[loc]
            break
    item0 = ds[int(j)]
    pos = np.asarray(linha["observation.state.right_ee_pose_gripper_base"], float)
    p_s, R_s = pos[:3], Rotation.from_euler("xyz", pos[3:]).as_matrix()
    q = np.zeros(29)
    q[12:15] = np.asarray(linha["observation.state.waist_state_joint"], float)
    fk, cam = FK(), Camera.carrega(a.camera)
    R_tp, p_tp = torso_na_pelvis(fk, q)
    img = item0["images"]["head_left"]
    img_np = img.permute(1, 2, 0).numpy()
    img_np = img_np / 255.0 if img_np.max() > 1.5 else img_np
    Hh, Ww = img.shape[1:]
    sx = SLICES["right_xyz_rotvec"]
    xyz = slice(sx.start, sx.start + 3)
    randn_orig = torch.randn

    def randn_igual(*args, **kw):
        size = kw.pop("size", None) or (args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args)
        size = tuple(size)
        return randn_orig((1,) + size[1:], **kw).expand(size).contiguous()

    saida = Path(a.saida)
    (saida / "quadros").mkdir(parents=True, exist_ok=True)
    resultados = {}
    for mod in a.modelos:
        run, passo = mod.rsplit(":", 1)
        m = carrega(a.base, Path(run), int(passo))
        frases_run = wla_v2.frases_do_run(Path(run))

        def prediz(itens):
            out = []
            for i in range(0, len(itens), a.lote):
                b = collate_fn(itens[i:i + a.lote])
                torch.manual_seed(1234)
                torch.randn = randn_igual
                try:
                    with torch.no_grad():
                        out.append(m.predict_action(examples=b)["normalized_actions"])
                finally:
                    torch.randn = randn_orig
            return np.concatenate(out, 0)

        resultados[mod] = []
        for frase, _ in FRASES:
            f = frases_run.get(frase, frase)
            f = f[0] if isinstance(f, list) else f
            base = dict(item0)
            base["task"] = f
            itens = [base]
            gl, gc = GRADE
            for r in range(gl):
                for c in range(gc):
                    t2 = img.clone()
                    t2[:, r * Hh // gl:(r + 1) * Hh // gl, c * Ww // gc:(c + 1) * Ww // gc] = 0.5 * (255 if img.max() > 1.5 else 1)
                    it = dict(base)
                    it["images"] = dict(base["images"])
                    it["images"]["head_left"] = t2
                    itens.append(it)
            P = prediz(itens)
            esc, off = item0["action_norm_scale"].numpy(), item0["action_norm_offset"].numpy()
            rel = P[:, :, xyz] * esc[xyz] + off[xyz]                       # metros, no referencial da MÃO
            caminho = p_s[None] + rel[0] @ R_s.T                          # pelvis
            dif = np.linalg.norm((rel[1:] - rel[0][None]) * 100, axis=-1).mean(-1)   # cm por pedaço tapado
            resultados[mod].append({"frase": frase, "usada": f, "caminho": caminho, "mapa": dif.reshape(gl, gc)})
            print(f"✅ {Path(run).name}@{passo} | {frase!r} -> fim da mão {np.round((caminho[-1] - p_s) * 100, 1)} cm | "
                  f"relevância máx {dif.max():.2f}", flush=True)
        del m
        torch.cuda.empty_cache()

    nm = {mo: f"{Path(mo.rsplit(':', 1)[0]).name.replace('prometheus_', 'WLA ')} @ {mo.rsplit(':', 1)[1]}"
          for mo in a.modelos}
    u0 = cam.projeta(p_s, R_tp, p_tp)
    for k_f, (frase, tipo) in enumerate(FRASES):
        fig, ax = plt.subplots(1, len(a.modelos), figsize=(8.2 * len(a.modelos), 7.2), facecolor="#0f1115", squeeze=False)
        for i, mo in enumerate(a.modelos):
            r = resultados[mo][k_f]
            a0 = ax[0, i]
            a0.imshow(img_np)
            mp = r["mapa"] / max(r["mapa"].max(), 1e-6)
            gl, gc = mp.shape
            from matplotlib.patches import Rectangle
            for rr in range(gl):          # só o que importa: retângulos vermelhos, opacidade = importância
                for cc in range(gc):
                    v = mp[rr, cc]
                    if v < 0.35:
                        continue
                    x0, y0 = cc * Ww / gc, rr * Hh / gl
                    a0.add_patch(Rectangle((x0, y0), Ww / gc, Hh / gl, facecolor=(1, 0.15, 0.1, 0.15 + 0.45 * v),
                                           edgecolor=(1, 0.2, 0.1, 0.9), lw=1.5 + 2 * v, zorder=3))
            traj = np.array([cam.projeta(P_, R_tp, p_tp) for P_ in r["caminho"]]) * [Ww / cam.w, Hh / cam.h]
            u = u0 * [Ww / cam.w, Hh / cam.h]
            a0.plot(traj[:, 0], traj[:, 1], "-", color="k", lw=6, alpha=0.6)
            a0.plot(traj[:, 0], traj[:, 1], "-", color="#3ef0ff", lw=3.5)
            a0.plot(traj[-1, 0], traj[-1, 1], "o", color="#3ef0ff", ms=12, mec="k", mew=1.5)
            a0.plot(u[0], u[1], "s", color="w", ms=11, mec="k", mew=1.5)
            a0.set_xlim(0, Ww)
            a0.set_ylim(Hh, 0)
            a0.axis("off")
            d = (r["caminho"][-1] - p_s) * 100
            a0.set_title(f"{nm[mo]}\nhand moves {np.linalg.norm(d):.1f} cm in 1 s · strongest region "
                         f"{r['mapa'].max():.1f} cm", color="w", fontsize=14)
        fig.suptitle(f"Instruction {k_f + 1}/{len(FRASES)}:  \"{frase}\"   ({tipo})", color="#ffd23f", fontsize=18)
        fig.text(0.5, 0.02, "same frozen image, same robot state, same noise — only the instruction changes  ·  "
                 "red = where hiding the image changes the plan (what it looks at)  ·  cyan = planned right-hand "
                 "path (1 s)  ·  white square = hand now", ha="center", color="#aaa", fontsize=11)
        fig.tight_layout(rect=(0, 0.04, 1, 0.93))
        fig.savefig(saida / "quadros" / f"{k_f:03d}.png", facecolor=fig.get_facecolor(), dpi=80)
        plt.close(fig)
    video = saida / "same_scene_other_instruction.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", f"1/{a.segundos}", "-i",
                    str(saida / "quadros" / "%03d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=25",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", str(video)], check=True)
    print(f"🎬 {video}", flush=True)


if __name__ == "__main__":
    main()
