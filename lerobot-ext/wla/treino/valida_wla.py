#!/usr/bin/env python
"""Validação do WLA nos episódios que o treino NÃO viu (split.json do descreve_datasets.py: 5 por tarefa) — 21/10.

Para cada checkpoint, nos MESMOS trechos de 1 s (mesma semente, mesmo ruído do fluxo):

  erro_mao_dir_cm       erro médio da posição da mão direita ao longo do trecho, em cm
  erro_mao_dir_fim_cm   o mesmo só no fim do trecho (1 s à frente) — o que mais pesa para chegar no objeto
  dedos_dir_acerto_%    passos em que a mão direita está aberta/fechada igual à demonstração
  fecha_no_tempo_%      dos trechos em que a demonstração FECHA a mão, em quantos o modelo também fecha
                        (até 0,3 s de diferença) — é a pega
  lingua_%              TROCA DE FRASE: mesma imagem e estado, frase da outra tarefa (caneca <-> maçã). % dos
                        trechos em que a frase CERTA reproduz a demonstração melhor que a trocada. 50% = o modelo
                        ignora a frase (decide pela imagem/posição — o "pega o copo como se fosse a maçã")
  lingua_razao          erro da mão dir. com a frase trocada / com a certa (> 1 = a frase muda o movimento)
  mse_total, mse_<parte>  erro na ação normalizada (como o treino)
Tudo também por tarefa. Sai <run>/validacao.json e uma tabela.

    cd ~/DEV/unifolm-wla-lora && CUDA_VISIBLE_DEVICES=0 python valida_wla.py \\
        --run playground/Checkpoints/prometheus_v2 --passos 10000 20000 \\
        --split /data/mrwlker/wla_v2/descricao/split.json --lotes 60
    # comparar com um run antigo (LoRA): --run playground/Checkpoints/lora_prometheus_dex3_cotreino --passos 20000
    #   (os runs antigos treinaram com TODOS os episódios: neles a validação é otimista)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from safetensors.torch import load_file  # noqa: E402

import wla_v2  # noqa: E402
from avalia_checkpoints_wla import PARTES, carrega_lora  # noqa: E402

FPS = 30


def carrega(base, run, passo):
    from unifolm_wla.model.framework.base_framework import baseframework
    run = Path(run)
    ck = run / f"checkpoints/steps_{passo}_model.safetensors"
    if not ck.exists() and (run / "final_model/model.safetensors").exists() and str(passo) == "final":
        ck = run / "final_model/model.safetensors"
    full = run / "config.full.yaml"
    cfg = yaml.safe_load(open(full if full.exists() else run / "config.yaml"))
    if (cfg.get("trainer", {}).get("lora") or {}).get("enabled", False):
        return carrega_lora(base, ck, run)
    m = baseframework.from_pretrained(base)
    falta, sobra = m.load_state_dict(load_file(str(ck)), strict=False)
    if sobra or len(falta) > 10:
        raise SystemExit(f"checkpoint não bate com o modelo: faltam {len(falta)}, sobram {len(sobra)}")
    m.norm_stats = json.load(open(run / "dataset_statistics.json"))
    return m.to(torch.bfloat16).cuda().eval()


def objeto(frase):
    f = (frase or "").lower()
    return "caneca" if ("mug" in f or "cup" in f) else "maca" if "apple" in f else "?"


def rivais(frases):
    """frase -> frase de OUTRA tarefa com o outro objeto (caneca <-> maçã); prefere a mais parecida."""
    r = {}
    for f in frases:
        outras = [g for g in frases if objeto(g) not in (objeto(f), "?")] or [g for g in frases if g != f]
        if not outras:
            continue
        pal = set(f.lower().split())
        r[f] = max(outras, key=lambda g: len(pal & set(g.lower().split())))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--passos", nargs="+", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--dados", default=None, help="config de dados (padrão: a do run)")
    ap.add_argument("--base", default="playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors")
    ap.add_argument("--lotes", type=int, default=60)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--sem-troca", action="store_true", help="pula o teste de troca de frase (2x mais rápido)")
    a = ap.parse_args()
    from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES
    from unifolm_wla.dataloader.multi_source_dataset.dataloader import create_training_dataloader
    run = Path(a.run)
    dados = a.dados or yaml.safe_load(open(run / "config.yaml"))["datasets"]["vla_data"]["data_config_path"]
    wla_v2.instala_split(a.split, "val")

    torch.manual_seed(0)
    np.random.seed(0)
    dl, _ = create_training_dataloader(dados, batch_size=a.lote, num_workers=8, pin_memory=False, shuffle=True)
    lotes = []
    for b in dl:
        lotes.append(b)
        if len(lotes) >= a.lotes:
            break
    frases = sorted({t for b in lotes for t in b["task"]})
    riv = rivais(frases)
    print(f"{len(lotes) * a.lote} trechos de 1 s dos episódios de VALIDAÇÃO | tarefas: {frases}\n"
          f"troca de frase: {riv}", flush=True)
    sx = SLICES["right_xyz_rotvec"]
    xyz = slice(sx.start, sx.start + 3)
    fd = SLICES["right_fig6d"]

    resultados = {}
    for passo in a.passos:
        m = carrega(a.base, run, passo)
        H = m.action_horizon
        A = {k: [] for k in ("mse", "cm", "cm_fim", "dedos", "lingua", "razao")}
        partes = {p: [] for p in PARTES}
        fecha = []
        por = {}
        for i, b in enumerate(lotes):
            torch.manual_seed(1000 + i)
            with torch.no_grad():
                pred = m.predict_action(examples=b)["normalized_actions"]
            alvo = b["action"].cpu().numpy()[:, -H:, :]
            mask = b["action_mask"].cpu().numpy()[:, None, :].repeat(H, 1)
            esc = b["action_norm_scale"].cpu().numpy()[:, None, :]
            d2 = (pred - alvo) ** 2
            cm = np.linalg.norm((pred - alvo)[:, :, xyz] * esc[:, :, xyz], axis=-1) * 100   # (B, H)
            fp, fa = (pred[:, :, fd].mean(-1) + 1) / 2 > 0.5, (alvo[:, :, fd].mean(-1) + 1) / 2 > 0.5
            troca = None
            if not a.sem_troca:
                bt = dict(b)
                bt["task"] = [riv.get(t, t) for t in b["task"]]
                torch.manual_seed(1000 + i)
                with torch.no_grad():
                    pt = m.predict_action(examples=bt)["normalized_actions"]
                troca = np.linalg.norm((pt - alvo)[:, :, xyz] * esc[:, :, xyz], axis=-1).mean(1) * 100
            for j, t in enumerate(b["task"]):
                mj = mask[j]
                e = por.setdefault(t, {k: [] for k in A} | {"fecha": []})
                vals = {"mse": float(d2[j][mj].mean()), "cm": float(cm[j].mean()), "cm_fim": float(cm[j, -1]),
                        "dedos": float((fp[j] == fa[j]).mean())}
                if troca is not None and riv.get(t, t) != t:
                    vals["lingua"] = float(cm[j].mean() < troca[j])
                    vals["razao"] = float(troca[j] / max(cm[j].mean(), 1e-3))
                for k, v in vals.items():
                    A[k].append(v)
                    e[k].append(v)
                # evento de fechar na demonstração: a mão passa de aberta para fechada dentro do trecho
                if (~fa[j][0]) and fa[j].any():
                    k_a = int(np.argmax(fa[j]))
                    ok = bool(fp[j].any() and abs(int(np.argmax(fp[j])) - k_a) <= int(0.3 * FPS))
                    fecha.append(ok)
                    e["fecha"].append(ok)
            for p in PARTES:
                s = SLICES[p]
                if mask[:, :, s].any():
                    partes[p].append(float(d2[:, :, s][mask[:, :, s]].mean()))

        def resume(D, fch):
            r = {"trechos": len(D["mse"]), "erro_mao_dir_cm": np.mean(D["cm"]), "erro_mao_dir_fim_cm": np.mean(D["cm_fim"]),
                 "dedos_dir_acerto_%": 100 * np.mean(D["dedos"]), "mse_total": np.mean(D["mse"])}
            if fch:
                r["fecha_no_tempo_%"] = 100 * np.mean(fch)
                r["eventos_fecha"] = len(fch)
            if D["lingua"]:
                r["lingua_%"] = 100 * np.mean(D["lingua"])
                r["lingua_razao"] = float(np.median(D["razao"]))
            return {k: round(float(v), 4) for k, v in r.items()}

        r = resume(A, fecha)
        r.update({f"mse_{p}": round(float(np.mean(v)), 5) for p, v in partes.items() if v})
        r["por_tarefa"] = {t: resume(e, e["fecha"]) for t, e in sorted(por.items())}
        resultados[str(passo)] = r
        print(f"\n=== {run.name} passo {passo} ===", flush=True)
        for k, v in r.items():
            if k != "por_tarefa":
                print(f"  {k:24s} {v}", flush=True)
        for t, rt in r["por_tarefa"].items():
            print(f"  · {t[:48]:48s} " + " | ".join(f"{k} {v}" for k, v in rt.items()), flush=True)
        del m
        torch.cuda.empty_cache()
    saida = run / "validacao.json"
    antigo = json.load(open(saida)) if saida.exists() else {}
    antigo.update(resultados)
    json.dump(antigo, open(saida, "w"), indent=1, ensure_ascii=False)
    print(f"\nsalvo em {saida}")


if __name__ == "__main__":
    main()
