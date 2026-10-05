#!/usr/bin/env python
"""
Carrega checkpoints do fine-tuning LoRA do UnifoLM-WLA e compara, nos MESMOS trechos do dataset, o quanto
cada um reproduz as demonstrações. (O repositório não tem carregador de checkpoint com LoRA para
inferência: aqui é base + injeção do LoRA com a config do treino + pesos afinados + normalização do treino.)

Métricas (sobre ~N lotes fixos, mesma semente para todos os checkpoints):
  mse_total        erro quadrático médio da ação normalizada, só nas partes ativas (o mse_score do treino,
                   mas em centenas de amostras em vez de 4)
  por parte        mão esq/dir (pose relativa), fig6d esq/dir (dedos), cintura, pernas, andar
  dedos_dir_mae    erro médio dos dedos da mão direita em 0..1
  fecha_dir_acerto % dos passos em que a mão direita está "fechada" (fig6d médio > 0,5) nos dois

ATENÇÃO: os 50 episódios foram todos usados no treino — isto mede o AJUSTE às demonstrações, não a
generalização para cenas novas.

    cd ~/DEV/unifolm-wla-lora && CUDA_VISIBLE_DEVICES=2 python avalia_checkpoints_wla.py \\
        --run playground/Checkpoints/lora_prometheus_dex3_maca_x_preto --passos 4000 8000 --lotes 50
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, ".")
from safetensors.torch import load_file  # noqa: E402

from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES  # noqa: E402
from unifolm_wla.dataloader.multi_source_dataset.dataloader import create_training_dataloader  # noqa: E402
from unifolm_wla.model.framework.base_framework import baseframework  # noqa: E402
from unifolm_wla.training.trainer_utils.trainer_tools import TrainerUtils  # noqa: E402

PARTES = ["left_xyz_rotvec", "right_xyz_rotvec", "left_fig6d", "right_fig6d", "waist_joint",
          "left_leg_joint", "right_leg_joint", "base_vx_vy", "base_vw", "height"]


def aplica_lora(m, lora_cfg):
    """Injeta o LoRA do treino no DiT e/ou no VLM. Aceita target_modules como LISTA (nomes, o oficial) ou STRING
    (regex do peft — o co-treino de 03/10 põe LoRA nas camadas de linguagem do VLM com regex)."""
    from peft import LoraConfig, inject_adapter_in_model
    if not lora_cfg or not lora_cfg.get("enabled", False):
        return m
    for nome, alvo in (("qwen_vl_interface", lambda: m.qwen_vl_interface.model), ("action_model", lambda: m.action_model.model)):
        sub = lora_cfg.get(nome) or {}
        if not sub.get("enabled", False):
            continue
        tm = sub.get("target_modules", [])
        inject_adapter_in_model(LoraConfig(r=sub.get("r", 16), lora_alpha=sub.get("lora_alpha", 32),
                                           lora_dropout=sub.get("lora_dropout", 0.05), bias=sub.get("bias", "none"),
                                           target_modules=tm if isinstance(tm, str) else list(tm)), alvo())
    return m


def carrega_lora(base_ckpt, fino_ckpt, run_dir):
    """Base -> injeta o LoRA (config do treino) -> pesos afinados -> normalização do treino."""
    m = baseframework.from_pretrained(base_ckpt)
    cfg = yaml.safe_load(open(Path(run_dir) / "config.yaml"))
    aplica_lora(m, cfg["trainer"]["lora"])
    sd = load_file(fino_ckpt)
    falta, sobra = m.load_state_dict(sd, strict=False)
    if sobra:
        raise SystemExit(f"chaves do checkpoint que o modelo não tem: {sobra[:5]}")
    if falta and "delta" not in Path(fino_ckpt).name:   # um "delta" só tem o que mudou: o resto vem do base
        raise SystemExit(f"o modelo ficou sem {len(falta)} tensores: {falta[:5]}")
    m.norm_stats = json.load(open(Path(run_dir) / "dataset_statistics.json"))
    return m.to(torch.bfloat16).cuda().eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", default="playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors")
    ap.add_argument("--passos", type=int, nargs="+", default=[4000, 8000])
    ap.add_argument("--lotes", type=int, default=50)
    ap.add_argument("--lote", type=int, default=8)
    a = ap.parse_args()
    run = Path(a.run)
    data_cfg = yaml.safe_load(open(run / "config.yaml"))["datasets"]["vla_data"]["data_config_path"]

    # os MESMOS lotes para todos os checkpoints
    torch.manual_seed(0)
    np.random.seed(0)
    dl, _ = create_training_dataloader(data_cfg, batch_size=a.lote, num_workers=4, pin_memory=False, shuffle=True)
    lotes = []
    for i, b in enumerate(dl):
        lotes.append(b)
        if len(lotes) >= a.lotes:
            break
    print(f"{len(lotes)} lotes x {a.lote} = {len(lotes) * a.lote} trechos de 1 s", flush=True)

    resultados = {}
    for passo in a.passos:
        m = carrega_lora(a.base, run / f"checkpoints/steps_{passo}_model.safetensors", run)
        H = m.action_horizon
        erros = {p: [] for p in PARTES}
        tot, dedos, fecha = [], [], []
        por_tarefa = {}   # frase -> (lista de mse por amostra, lista de acerto aberto/fechado da mão direita)
        for i, b in enumerate(lotes):
            torch.manual_seed(1000 + i)   # mesmo ruído inicial do fluxo para todos os checkpoints
            with torch.no_grad():
                pred = m.predict_action(examples=b)["normalized_actions"]
            alvo = b["action"].cpu().numpy()[:, -H:, :]
            mask = b["action_mask"].cpu().numpy()[:, None, :].repeat(H, 1)
            d2 = (pred - alvo) ** 2
            tot.append(d2[mask].mean())
            for p in PARTES:
                s = SLICES[p]
                if mask[:, :, s].any():
                    erros[p].append(d2[:, :, s][mask[:, :, s]].mean())
            # dedos da direita: minmax_q 0..1 -> -1..1  => valor 0..1 = (x+1)/2
            fp = (pred[:, :, SLICES["right_fig6d"]] + 1) / 2
            fa = (alvo[:, :, SLICES["right_fig6d"]] + 1) / 2
            dedos.append(np.abs(fp - fa).mean())
            fecha.append(((fp.mean(-1) > 0.5) == (fa.mean(-1) > 0.5)).mean())
            for j, frase in enumerate(b.get("task", [None] * len(pred))):
                mj = mask[j]
                e = por_tarefa.setdefault(frase, ([], []))
                e[0].append(float(d2[j][mj].mean()))
                e[1].append(float(((fp[j].mean(-1) > 0.5) == (fa[j].mean(-1) > 0.5)).mean()))
        r = {"mse_total": float(np.mean(tot)), "dedos_dir_mae": float(np.mean(dedos)),
             "fecha_dir_acerto_%": float(100 * np.mean(fecha)),
             **{f"mse_{p}": float(np.mean(v)) for p, v in erros.items() if v}}
        for frase, (ms, ac) in sorted(por_tarefa.items(), key=lambda x: str(x[0])):
            r[f"tarefa: {frase}"] = {"amostras": len(ms), "mse_total": float(np.mean(ms)),
                                     "fecha_dir_acerto_%": float(100 * np.mean(ac))}
        resultados[passo] = r
        print(f"\n=== passo {passo} ===", flush=True)
        for k, v in r.items():
            print(f"  {k:24s} {v:.4f}" if not isinstance(v, dict) else f"  {k}: {v}", flush=True)
        del m
        torch.cuda.empty_cache()
    json.dump(resultados, open(run / "avaliacao_checkpoints.json", "w"), indent=1)
    print("\nsalvo em", run / "avaliacao_checkpoints.json")


if __name__ == "__main__":
    main()
