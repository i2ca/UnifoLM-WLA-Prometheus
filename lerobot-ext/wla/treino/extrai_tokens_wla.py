#!/usr/bin/env python
"""TOKENS DO WLA para um "leitor" (03/11): em todos os episódios dos nossos datasets (inclusive os de PEGA ERRADA,
que ficaram fora do treino), uma consulta a cada --cada quadros pelo WLA, guardando o que ele tem por dentro:
  vlm_img / vlm_txt / vlm_est / vlm_tudo  saída do VLM (2560) que o DiT lê, média nos tokens da cabeça, da frase,
                                          do estado e em todos
  dit                                     saída do DiT (último passo do fluxo), média nos tokens de ação
  plano                                   o trecho de ação previsto (mão dir. xyz, dedos, yaw — 30 passos)
e, para comparar, o que um leitor teria SEM o WLA: imagem crua reduzida (24x32) e o estado (60).
Rótulos: progresso no episódio, dedos agora e daqui a 1 s, tarefa, episódio de pega errada (fora do split).
Sai um .npz por dataset em --saida (o sonda_tokens.py treina os leitores).

    cd ~/DEV/unifolm-wla-lora && CUDA_VISIBLE_DEVICES=0 python .../extrai_tokens_wla.py \\
        --run playground/Checkpoints/prometheus_v8 --passo 12000 --saida /data/mrwlker/tokens_v8
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--passo", required=True)
    ap.add_argument("--dados", default="unifolm_wla/dataloader/multi_source_dataset/configs/prometheus_v7.yaml")
    ap.add_argument("--split", default="/data/mrwlker/wla_v7/descricao/split.json")
    ap.add_argument("--base", default="playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors")
    ap.add_argument("--cada", type=int, default=10)
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    import torch.nn.functional as F

    import wla_v2
    from atencao_video_v7 import segmentos
    from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES, STATE_SLICES
    from unifolm_wla.dataloader.multi_source_dataset.dataloader import collate_fn, create_training_dataloader
    from valida_wla import carrega

    split = json.load(open(a.split))
    _, ds = create_training_dataloader(a.dados, batch_size=1, num_workers=0, pin_memory=False, shuffle=False)
    m = carrega(a.base, Path(a.run), a.passo)
    iface = m.qwen_vl_interface
    tok = iface.processor.tokenizer
    frases = wla_v2.frases_do_run(Path(a.run))
    guard = {}
    orig_build = iface.build_qwenvl_inputs

    def build(*x, **kw):
        out = orig_build(*x, **kw)
        guard["inp"], guard["roles"] = out, (kw.get("image_roles") or [None])[0]
        return out
    iface.build_qwenvl_inputs = build

    # o que o DiT recebe do VLM (o tensor [B, L, 2560]) e o que ele devolve (último passo do fluxo)
    dit = m.action_model.model

    def pre(mod, args, kwargs):
        for v in list(args) + list(kwargs.values()):
            if torch.is_tensor(v) and v.dim() == 3 and v.shape[-1] == 2560:
                guard["vl"] = v.detach()
        return None

    def pos(mod, args, out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(o):
            guard["dit"] = o.detach()
    dit.register_forward_pre_hook(pre, with_kwargs=True)
    dit.register_forward_hook(pos)

    sx, fd, yw = SLICES["right_xyz_rotvec"], SLICES["right_fig6d"], SLICES["waist_joint"].start
    sel = list(range(sx.start, sx.start + 3)) + list(range(fd.start, fd.stop)) + [yw]
    fds = STATE_SLICES["right_fig6d"]
    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)
    cum = np.array(ds._cum_lengths)
    for k, (td, d) in enumerate(zip(ds._task_dirs, ds._datasets)):
        nome = Path(td).name
        sp = split.get(nome, {})
        bons = set(int(e) for e in sp.get("treino", [])) | set(int(e) for e in sp.get("val", []))
        val = set(int(e) for e in sp.get("val", []))
        t = d.hf_dataset.data
        ep, fr = t.column("episode_index").to_numpy(), t.column("frame_index").to_numpy()
        dedos_col = np.stack(t.column("observation.state.right_fig6d").to_numpy(zero_copy_only=False)).mean(-1)
        R = {c: [] for c in ("vlm_img", "vlm_txt", "vlm_est", "vlm_tudo", "dit", "plano", "img_crua", "estado",
                             "ep", "quadro", "T", "dedos_agora", "dedos_1s", "falha", "val")}
        eps = list(dict.fromkeys(ep.tolist()))
        print(f"▶ {nome}: {len(eps)} episódios ({len(eps) - len(bons & set(eps))} de pega errada)", flush=True)
        for e in eps:
            loc = np.where(ep == e)[0]
            loc = loc[np.argsort(fr[loc])]
            T = len(loc)
            for i in range(0, T, a.cada):
                j = (cum[k - 1] if k > 0 else 0) + loc[i]
                item = ds[int(j)]
                b = collate_fn([item])
                f = frases.get(b["task"][0], b["task"][0])
                b["task"] = [f[0] if isinstance(f, list) else f]
                torch.manual_seed(1000 + i)
                with torch.no_grad():
                    pred = m.predict_action(examples=b)["normalized_actions"][0]
                inp = guard["inp"]
                L = inp["input_ids"].shape[1]
                vl = guard["vl"][0, -L:].float()                     # os tokens do VLM (o fim é o prompt)
                cams, frase_pos, est_pos = segmentos(inp, tok, guard["roles"])[:3]
                pos_img = cams.get("head_left", (np.array([], int), None))[0]
                pos_f = np.array([i_ for ps, _ in frase_pos for i_ in ps], int)
                media = (lambda ps: vl[torch.as_tensor(ps, dtype=torch.long)].mean(0) if len(ps) else vl.mean(0))
                R["vlm_img"].append(media(pos_img).cpu().numpy().astype(np.float16))
                R["vlm_txt"].append(media(pos_f).cpu().numpy().astype(np.float16))
                R["vlm_est"].append(media(np.asarray(est_pos, int)).cpu().numpy().astype(np.float16))
                R["vlm_tudo"].append(vl.mean(0).cpu().numpy().astype(np.float16))
                R["dit"].append(guard["dit"][0].float().mean(0).cpu().numpy().astype(np.float16))
                R["plano"].append(np.asarray(pred)[:, sel].reshape(-1).astype(np.float16))
                im = item["images"]["head_left"].float()
                im = im / 255.0 if im.max() > 1.5 else im
                R["img_crua"].append(F.interpolate(im[None], size=(24, 32), mode="area")[0].reshape(-1).numpy()
                                     .astype(np.float16))
                R["estado"].append(item["state"].numpy().astype(np.float16))
                R["ep"].append(e)
                R["quadro"].append(i)
                R["T"].append(T)
                R["dedos_agora"].append(float(dedos_col[loc[i]]))
                R["dedos_1s"].append(float(dedos_col[loc[min(i + 30, T - 1)]]))
                R["falha"].append(int(e not in bons))
                R["val"].append(int(e in val))
            print(f"   ep {e}: {len(range(0, T, a.cada))} consultas", flush=True)
        np.savez_compressed(saida / f"{nome}.npz", **{c: np.asarray(v) for c, v in R.items()}, tarefa=nome)
        print(f"✅ {nome}: {len(R['ep'])} consultas -> {saida / (nome + '.npz')}", flush=True)
    print("FIM", flush=True)


if __name__ == "__main__":
    main()
