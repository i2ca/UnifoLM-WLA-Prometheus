#!/usr/bin/env python
"""Descreve COMPLETAMENTE cada dataset convertido (formato do WLA) e separa a validação (21/10).

Por episódio, só pelos sensores gravados (sem anotação à mão):
  - mão usada (a que mais andou), duração, giro da cintura;
  - PEGA: quadro em que os dedos da mão usada passam de 50% do fechamento (entre o mínimo e o máximo do episódio)
    e ficam fechados; a posição da mão nesse quadro = onde o OBJETO estava (referencial da base do robô);
  - SOLTA: quando os dedos voltam a abrir depois de pegar; posição = onde o objeto foi colocado;
  - quanto levantou depois de pegar;
  - ETAPAS com frase (alcançar / fechar / levantar-carregar / soltar / voltar) e os quadros de cada uma — servem
    para co-treinar o VLM (pergunta "o que o robô está fazendo?") e para o ER-1 narrar;
  - SUSPEITOS: sem pega, pega no 1º segundo, muito curto/longo, solta sem levantar, etc.

Validação: --val N episódios por tarefa, ESPALHADOS pela posição do objeto (quantis de y da pega), nunca suspeitos.

    python descreve_datasets.py --raiz /home/mrwlker/unifolm_data/Prometheus_G1_Dex3_Mix_Dataset --val 5 \\
        --saida /data/mrwlker/wla_v2/descricao
Sai: DATASETS.md (o relatório), episodios.jsonl (uma linha por episódio), etapas.jsonl, split.json
({tarefa_dir: {"val": [...], "treino": [...], "excluidos": [...]}}).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 30


def carrega(td):
    arqs = sorted((td / "data").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(a) for a in arqs], ignore_index=True)
    tarefas = pd.read_parquet(td / "meta" / "tasks.parquet")
    # LeRobot v3: o índice do DataFrame é o texto da tarefa e a coluna task_index o número
    if "task_index" in tarefas.columns:
        nomes = {int(r.task_index): str(i) for i, r in tarefas.iterrows()}
    else:
        nomes = dict(enumerate(map(str, tarefas.index)))
    return df, nomes


def col(ep, k):
    return np.stack(ep[k].to_numpy()).astype(np.float64)


def fechamento(f6):
    """fig6d -> 0 (aberta) .. 1 (fechada), normalizado no próprio episódio."""
    m = f6.mean(1)
    lo, hi = np.percentile(m, 2), np.percentile(m, 98)
    if hi - lo < 1e-3:
        return np.zeros_like(m), hi - lo
    return np.clip((m - lo) / (hi - lo), 0, 1), hi - lo


def descreve_episodio(ep, nome_tarefa):
    n = len(ep)
    d = {"episodio": int(ep["episode_index"].iloc[0]), "tarefa": nome_tarefa, "quadros": n,
         "segundos": round(n / FPS, 1), "suspeito": []}
    pos = {l: col(ep, f"observation.state.{l}_ee_pose_gripper_base")[:, :3] for l in ("left", "right")}
    caminho = {l: float(np.linalg.norm(np.diff(pos[l], axis=0), axis=1).sum()) for l in pos}
    mao = max(caminho, key=caminho.get)
    d["mao"] = "direita" if mao == "right" else "esquerda"
    d["caminho_cm"] = {k: round(v * 100) for k, v in caminho.items()}
    p = pos[mao]
    d["inicio_cm"] = np.round(p[0] * 100, 1).tolist()
    d["fim_cm"] = np.round(p[-1] * 100, 1).tolist()
    if "observation.state.waist_state_joint" in ep:
        w = col(ep, "observation.state.waist_state_joint")
        d["cintura_yaw_graus"] = [round(float(np.degrees(w[:, 0].min())), 1), round(float(np.degrees(w[:, 0].max())), 1)]
    c, amp = fechamento(col(ep, f"observation.state.{mao}_fig6d"))
    if amp < 0.05:   # estado da Dex3 = 0 nos datasets de 30/09-01/10: usa a AÇÃO dos dedos
        c, amp = fechamento(col(ep, f"action.{mao}_fig6d"))
        d["dedos_de"] = "acao"
    else:
        d["dedos_de"] = "estado"
    fechada = c > 0.5
    # trechos fechados (>= 0,3 s). A PEGA é o trecho mais longo; os anteriores são TENTATIVAS que falharam
    # (fecha ao lado do objeto, abre, tenta de novo — o modelo copia isso). Trecho que começa no quadro 0 =
    # o episódio COMEÇA SEGURANDO (ex.: caneca no coador: a caneca já está na mão).
    trechos, i = [], 0
    while i < n:
        if fechada[i]:
            j = i
            while j < n and fechada[j]:
                j += 1
            if j - i >= FPS * 3 // 10:
                trechos.append((i, j))
            i = j
        else:
            i += 1
    d["trechos_fechados"] = [[a_, b_] for a_, b_ in trechos]
    d["comeca_segurando"] = bool(trechos and trechos[0][0] == 0)
    pega = solta = None
    if trechos:
        k = max(range(len(trechos)), key=lambda t: trechos[t][1] - trechos[t][0])
        pega, solta = trechos[k][0], (trechos[k][1] if trechos[k][1] < n else None)
        d["tentativas_falhas"] = k - (1 if d["comeca_segurando"] and k > 0 else 0)
        if d["tentativas_falhas"] > 0:
            d["suspeito"].append(f"{d['tentativas_falhas']} tentativa(s) de pegar que falharam antes da pega")
        if len(trechos) > k + 1:
            d["suspeito"].append(f"fechou de novo {len(trechos) - k - 1}x depois de soltar")
    d["pega_quadro"], d["solta_quadro"] = pega, solta
    if pega is None:
        d["suspeito"].append("sem pega (dedos não fecharam por 0,3 s)")
    else:
        d["pega_s"] = round(pega / FPS, 1)
        fim_carrega = solta if solta is not None else n
        d["levantou_cm"] = round(float((p[pega:fim_carrega, 2].max() - p[pega, 2]) * 100), 1)
        if not d["comeca_segurando"]:
            d["objeto_cm"] = np.round(p[pega] * 100, 1).tolist()
            if pega < FPS:
                d["suspeito"].append("pega no 1º segundo")
        if d["levantou_cm"] < 2 and solta is None:
            d["suspeito"].append("pegou e não levantou")
        if solta is not None:
            d["solta_s"] = round(solta / FPS, 1)
            d["colocado_cm"] = np.round(p[solta] * 100, 1).tolist()
            d["carregou_cm"] = round(float(np.linalg.norm(p[solta, :2] - p[pega, :2]) * 100), 1)
    if n < 3 * FPS:
        d["suspeito"].append("curto (< 3 s)")
    if n > 60 * FPS:
        d["suspeito"].append("longo (> 60 s)")
    d["etapas"] = etapas(d, n, nome_tarefa)
    return d


OBJ = [("mug", "the white mug"), ("cup", "the white mug"), ("apple", "the red apple")]


def objeto_da_tarefa(t):
    t = t.lower()
    return next((o for k, o in OBJ if k in t), "the object")


def etapas(d, n, tarefa):
    """[(ini, fim, frase)] — frases curtas no estilo da narração (1ª pessoa)."""
    obj = objeto_da_tarefa(tarefa)
    mao = "right" if d["mao"] == "direita" else "left"
    pg, sl = d["pega_quadro"], d["solta_quadro"]
    if pg is None:
        return [(0, n, f"moving my {mao} hand")]
    m = FPS // 3
    if d.get("comeca_segurando"):
        e = [(0, min(n, pg + m), f"holding {obj} in my {mao} hand")]
    else:
        e = [(0, max(0, pg - m), f"reaching for {obj} with my {mao} hand"),
             (max(0, pg - m), min(n, pg + m), f"closing my {mao} hand on {obj}")]
    if sl is None:
        e.append((min(n, pg + m), n, f"lifting {obj}" if d.get("levantou_cm", 0) > 3 else f"holding {obj}"))
    else:
        e += [(min(n, pg + m), max(pg + m, sl - m), f"carrying {obj}"),
              (max(pg + m, sl - m), min(n, sl + m), f"releasing {obj}"),
              (min(n, sl + m), n, f"moving my {mao} hand back")]
    return [(int(a), int(b), f) for a, b, f in e if b > a]


def escolhe_val(eps, nval):
    pos = lambda e: e.get("objeto_cm") or e.get("colocado_cm")   # noqa: E731
    bons = [e for e in eps if not e["suspeito"] and pos(e)]
    if len(bons) <= nval:
        return [e["episodio"] for e in bons[:max(0, len(bons) - 1)]]
    bons.sort(key=lambda e: (pos(e)[1], pos(e)[0]))
    idx = np.unique(np.linspace(0, len(bons) - 1, nval + 2)[1:-1].round().astype(int))
    return sorted(bons[i]["episodio"] for i in idx)


def resumo(xs):
    a = np.array(xs, float)
    return f"{np.median(a):.1f} (mín {a.min():.1f}, máx {a.max():.1f})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", required=True, help="pasta multi_task (subpastas = tarefas) ou uma tarefa")
    ap.add_argument("--val", type=int, default=5)
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    raiz, out = Path(a.raiz), Path(a.saida)
    out.mkdir(parents=True, exist_ok=True)
    tds = [raiz] if (raiz / "meta").exists() else sorted(p for p in raiz.iterdir() if (p / "meta").exists())
    md = ["# Datasets do treino — descrição automática\n",
          f"Raiz: `{raiz}`. Posições em cm no referencial da base do robô (x frente, y esquerda, z cima). "
          "Objeto = posição da mão usada quando os dedos fecharam. Validação: episódios espalhados pela posição do "
          f"objeto ({a.val} por tarefa), nunca suspeitos.\n"]
    split, f_eps, f_et = {}, open(out / "episodios.jsonl", "w"), open(out / "etapas.jsonl", "w")
    for td in tds:
        df, nomes = carrega(td)
        eps = []
        for e_idx, ep in df.groupby("episode_index", sort=True):
            nome = nomes.get(int(ep["task_index"].iloc[0]), "?")
            d = descreve_episodio(ep.sort_values("frame_index"), nome)
            d["dataset"] = td.name
            eps.append(d)
            f_eps.write(json.dumps(d, ensure_ascii=False) + "\n")
            for ini, fim, frase in d["etapas"]:
                f_et.write(json.dumps({"dataset": td.name, "episodio": d["episodio"], "ini": ini, "fim": fim,
                                       "frase": frase, "tarefa": nome}, ensure_ascii=False) + "\n")
        val = escolhe_val(eps, a.val)
        excl = [e["episodio"] for e in eps if e["suspeito"]]
        split[td.name] = {"val": val, "excluidos": excl,
                          "treino": [e["episodio"] for e in eps if e["episodio"] not in val and e["episodio"] not in excl]}
        tarefas = sorted({e["tarefa"] for e in eps})
        ok = [e for e in eps if "objeto_cm" in e]
        maos = {m: sum(e["mao"] == m for e in eps) for m in ("direita", "esquerda")}
        md += [f"\n## {td.name}\n",
               f"- **Frase(s):** " + "; ".join(f"*{t}*" for t in tarefas),
               f"- **Episódios:** {len(eps)} ({sum(e['quadros'] for e in eps)} quadros, "
               f"{sum(e['quadros'] for e in eps) / FPS / 60:.1f} min) | duração {resumo([e['segundos'] for e in eps])} s",
               f"- **Mão usada:** direita {maos['direita']}, esquerda {maos['esquerda']} | dedos lidos do "
               f"{'estado' if all(e['dedos_de'] == 'estado' for e in eps) else 'comando (estado da Dex3 = 0)'}"]
        if ok:
            O = np.array([e["objeto_cm"] for e in ok])
            md += [f"- **Onde estava o objeto (pega):** x {O[:, 0].min():.0f}–{O[:, 0].max():.0f}, "
                   f"y {O[:, 1].min():.0f}–{O[:, 1].max():.0f}, z {O[:, 2].min():.0f}–{O[:, 2].max():.0f} cm "
                   f"(mediana {np.round(np.median(O, 0), 1).tolist()})",
                   f"- **Pega aos** {resumo([e['pega_s'] for e in ok])} s | **levantou** "
                   f"{resumo([e['levantou_cm'] for e in ok])} cm"]
        seg = sum(e.get("comeca_segurando", False) for e in eps)
        if seg:
            md.append(f"- ⚠️ **Começa SEGURANDO o objeto em {seg}/{len(eps)}** episódios (não ensina a pegar)")
        tf = [e for e in eps if e.get("tentativas_falhas", 0) > 0]
        if tf:
            md.append(f"- ⚠️ **Tentativas que falharam antes da pega em {len(tf)}/{len(eps)}** episódios "
                      f"({sum(e['tentativas_falhas'] for e in tf)} no total) — o modelo aprende a fechar, abrir e tentar de novo")
        so = [e for e in eps if "colocado_cm" in e]
        if so:
            C = np.array([e["colocado_cm"] for e in so])
            md.append(f"- **Soltou** em {len(so)}/{len(eps)}: x {C[:, 0].min():.0f}–{C[:, 0].max():.0f}, "
                      f"y {C[:, 1].min():.0f}–{C[:, 1].max():.0f} cm, carregou {resumo([e['carregou_cm'] for e in so])} cm")
        if any("cintura_yaw_graus" in e for e in eps):
            yw = np.array([e["cintura_yaw_graus"] for e in eps])
            md.append(f"- **Cintura (yaw):** {yw[:, 0].min():.0f}° a {yw[:, 1].max():.0f}°")
        md.append(f"- **Validação ({len(val)}):** {val}")
        if excl:
            md.append(f"- **Suspeitos (fora do treino e da validação, {len(excl)}):**")
            md += [f"  - ep {e['episodio']}: {', '.join(e['suspeito'])}" for e in eps if e["suspeito"]]
        md.append(f"- **Treino:** {len(split[td.name]['treino'])} episódios")
        print(f"{td.name}: {len(eps)} eps, val {val}, suspeitos {excl}", flush=True)
    (out / "split.json").write_text(json.dumps(split, indent=1))
    (out / "DATASETS.md").write_text("\n".join(md) + "\n")
    print(f"✅ {out}/DATASETS.md, split.json, episodios.jsonl, etapas.jsonl")


if __name__ == "__main__":
    main()
