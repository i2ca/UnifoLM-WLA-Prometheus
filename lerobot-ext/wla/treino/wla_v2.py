"""Ajustes do treino v2 (21/10), por cima do código da Unitree sem editá-lo (monkeypatch). Usado pelo
treina_wla_v2.py e pelo valida_wla.py.

  instala_split(split.json, "treino"|"val")  — cada pasta de tarefa só carrega os episódios daquela parte
                                               (o split.json vem do descreve_datasets.py: val = 5 por tarefa,
                                               suspeitos fora dos dois).
  instala_balanceamento(alfa)                — amostra as TAREFAS por igual (alfa=1) em vez de proporcional ao
                                               número de quadros (maçã→X tinha 3x os quadros de pegar caneca).
  instala_pesos_acao({parte: peso})          — perda da ação com peso por parte (mão direita e dedos mais; pernas,
                                               base e altura, que o robô parado não usa, quase nada).
"""
import inspect
import json
import random
import textwrap
from pathlib import Path

import numpy as np
import torch

# partes da ação unificada (54) — action_mapping.SLICES
PESOS_PADRAO = {
    "left_xyz_rotvec": 1.0, "left_gripper": 0.5, "left_fig6d": 0.5,
    "right_xyz_rotvec": 3.0, "right_gripper": 1.0, "right_fig6d": 2.0,
    "waist_joint": 0.5, "torso_joint": 0.1,
    "base_vx_vy": 0.05, "base_vw": 0.05, "base_rotvec": 0.05, "height": 0.05,
    "left_leg_joint": 0.05, "right_leg_joint": 0.05,
}


PARTE_FORCADA = None


def instala_split(caminho, parte):
    from unifolm_wla.dataloader.multi_source_dataset import lerobot_wrapper as LW
    split = json.load(open(caminho))
    cls = LW.LeRobotDatasetWithSelectedKeys
    if getattr(cls, "_split_v2", None):
        return
    orig = cls.__init__

    def __init__(self, *args, **kw):
        nome = Path(kw.get("root") or "").name
        p = PARTE_FORCADA or parte   # a validação dentro do treino cria o dataset de "val" com o ajuste já instalado
        if nome in split and kw.get("episodes") is None:
            kw["episodes"] = sorted(int(e) for e in split[nome][p])
            print(f"📂 {nome}: {p} = {len(kw['episodes'])} episódios", flush=True)
        orig(self, *args, **kw)

    cls.__init__ = __init__
    cls._split_v2 = parte


class Balanceado(torch.utils.data.Dataset):
    """Invólucro do SingleSourceDataset com os índices repetidos por tarefa e, opcionalmente, a FRASE da tarefa
    trocada pela descritiva (frases = {frase do dataset: frase nova}). Fica no nível do módulo para viajar
    (pickle) até os leitores, que são processos novos (spawn): um ajuste na CLASSE não chegava lá (21/10)."""

    def __init__(self, ds, mapa, frases=None):
        self.ds, self.mapa, self.frases = ds, mapa, frases or {}

    def __len__(self):
        return len(self.mapa)

    def __getitem__(self, i):
        item = self.ds[int(self.mapa[i])]
        if self.frases and item.get("task") in self.frases:
            v = self.frases[item["task"]]
            item["task"] = random.choice(v) if isinstance(v, (list, tuple)) else v   # lista = variações sorteadas
        return item

    def __getattr__(self, nome):   # normalização, estatísticas etc. vêm do dataset de dentro
        if nome in ("ds", "mapa", "frases") or nome.startswith("__"):
            raise AttributeError(nome)
        return getattr(self.ds, nome)


def instala_balanceamento(alfa=1.0, frases=None):
    """Repete os quadros das tarefas menores: fator = (maior / tamanho) ** alfa (0 = sem repetir). frases: troca
    a frase da tarefa (ex.: "Pick up the white mug." -> "... from the side with the right hand upright.")."""
    from unifolm_wla.dataloader.multi_source_dataset import dataloader as DL
    if getattr(DL, "_balanceado", None) is not None or (alfa <= 0 and not frases):
        return
    orig = DL.create_single_source_dataset
    if frases:
        print("🗣️  frases da tarefa trocadas:\n" + "\n".join(f"   {k!r} -> {v!r}" for k, v in frases.items()), flush=True)

    def cria(*args, **kw):
        ds = orig(*args, **kw)
        tam = [len(d) for d in ds._datasets]
        maior, mapa, ini = max(tam), [], 0
        for td, n in zip(ds._task_dirs, tam):
            k = (maior / n) ** alfa if alfa > 0 else 1.0
            idx = np.arange(ini, ini + n)
            rep = np.concatenate([np.tile(idx, int(k)), idx[: int(round((k - int(k)) * n))]])
            mapa.append(rep)
            print(f"⚖️  {Path(td).name}: {n} quadros x{k:.2f} -> {len(rep)}", flush=True)
            ini += n
        return Balanceado(ds, np.concatenate(mapa), frases)

    DL.create_single_source_dataset = cria
    DL._balanceado = alfa


def instala_pesos_acao(pesos=None):
    """Troca o peso da perda (= máscara de DoFs válidos) por máscara x peso da parte. Reescreve a linha
    `weight = mask_part ...` do forward do MMDiT (falha alto se a Unitree mudar o código)."""
    from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES, UNIFIED_DIM
    from unifolm_wla.model.modules.action_model import MMDiT_ActionHeader as H
    p = {**PESOS_PADRAO, **(pesos or {})}
    vetor = torch.ones(UNIFIED_DIM)
    for nome, s in SLICES.items():
        vetor[s] = float(p.get(nome, 1.0))
    cls = next(c for c in vars(H).values() if inspect.isclass(c) and hasattr(c, "predict_action")
               and c.__module__ == H.__name__)
    nome_f = next(n for n, f in vars(cls).items() if callable(f) and "sq_err" in (inspect.getsource(f) if
                  inspect.isfunction(f) else ""))
    src = textwrap.dedent(inspect.getsource(getattr(cls, nome_f)))
    velho = "weight = mask_part if mask_part.shape[2] > 0 else None"
    if velho not in src:
        raise RuntimeError("instala_pesos_acao: a linha do peso mudou no MMDiT_ActionHeader — revise")
    src = src.replace(velho, "weight = (mask_part * _PESOS_V2.to(mask_part)) if mask_part.shape[2] > 0 else None")
    ns = dict(vars(H))
    ns["_PESOS_V2"] = vetor
    exec(compile(src, H.__file__, "exec"), ns)   # noqa: S102
    setattr(cls, nome_f, ns[nome_f])
    print("🎚️  pesos da perda por parte: " + ", ".join(f"{k} {v:g}" for k, v in p.items()), flush=True)
    return vetor


def frases_do_run(run_dir):
    """A troca de frases que o run usou (config.full.yaml: trainer.v2.frases) — para o servidor e a validação
    mandarem ao modelo a mesma frase do treino. Lista de variações -> a primeira."""
    import yaml
    run_dir = Path(run_dir)
    full = run_dir / "config.full.yaml"
    arq = full if full.exists() else run_dir / "config.yaml"
    if not arq.exists():
        return {}
    v2 = ((yaml.safe_load(open(arq)) or {}).get("trainer", {}) or {}).get("v2", {}) or {}
    return {k: (v[0] if isinstance(v, (list, tuple)) else v) for k, v in (v2.get("frases") or {}).items()}


# ── validação (dentro do treino e no valida_wla.py) ─────────────────────────────────────────────────────────
def objeto(frase):
    f = (frase or "").lower()
    return "caneca" if ("mug" in f or "cup" in f) else "maca" if "apple" in f else "?"


def rivais(frases):
    """frase -> frase de OUTRA tarefa com o outro objeto (caneca <-> maçã); prefere a mais parecida."""
    r = {}
    for f in frases:
        outras = [g for g in frases if objeto(g) not in (objeto(f), "?")] or [g for g in frases if g != f]
        if outras:
            pal = set(f.lower().split())
            r[f] = max(outras, key=lambda g: len(pal & set(g.lower().split())))
    return r


def lotes_validacao(dados_cfg, n_lotes, lote, frases_fixas=None, cache=None, leitores=16):
    """Lotes FIXOS dos episódios de validação (o split já instalado). frases_fixas: a frase do dataset -> a do treino
    (a 1ª variação), igual para todos os checkpoints. cache: arquivo .pt — com ele pronto, carrega em segundos
    (21/10: montar no passo 0, um a um, levou > 10 min e o NCCL derrubou o treino esperando o processo 0)."""
    global PARTE_FORCADA
    if cache and Path(cache).exists():
        return torch.load(cache, weights_only=False)
    from unifolm_wla.dataloader.multi_source_dataset.dataloader import collate_fn, create_training_dataloader
    PARTE_FORCADA = "val"
    try:
        _, ds = create_training_dataloader(dados_cfg, batch_size=lote, num_workers=0, pin_memory=False, shuffle=False)
    finally:
        PARTE_FORCADA = None
    if isinstance(ds, Balanceado):
        ds = ds.ds   # sem repetir quadros nem sortear a variação da frase
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[: n_lotes * lote].tolist()
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=lote, shuffle=False,
                                     num_workers=leitores, collate_fn=collate_fn)
    lotes = []
    for b in dl:
        if frases_fixas:
            b["task"] = [frases_fixas.get(t, t) for t in b["task"]]
        lotes.append(b)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save(lotes, cache)
    return lotes


def _orig_frase(frases_fixas):
    inv = {}
    for k, v in (frases_fixas or {}).items():
        inv[v[0] if isinstance(v, (list, tuple)) else v] = k
    return inv


@torch.no_grad()
def metricas_validacao(m, lotes, troca_frase=True):
    """erro da mão direita (cm, no trecho e no fim), dedos, fecha no tempo, troca de frase — total e por objeto."""
    from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES
    sx, fd = SLICES["right_xyz_rotvec"], SLICES["right_fig6d"]
    xyz = slice(sx.start, sx.start + 3)
    riv = rivais(sorted({t for b in lotes for t in b["task"]}))
    H = m.action_horizon
    A = {k: [] for k in ("cm", "cm_fim", "dedos", "lingua", "razao", "fecha")}
    por = {}
    for i, b in enumerate(lotes):
        torch.manual_seed(1000 + i)
        pred = m.predict_action(examples=b)["normalized_actions"]
        alvo = b["action"].cpu().numpy()[:, -H:, :]
        esc = b["action_norm_scale"].cpu().numpy()[:, None, :]
        cm = np.linalg.norm((pred - alvo)[:, :, xyz] * esc[:, :, xyz], axis=-1) * 100
        fp, fa = (pred[:, :, fd].mean(-1) + 1) / 2 > 0.5, (alvo[:, :, fd].mean(-1) + 1) / 2 > 0.5
        tr = None
        if troca_frase:
            bt = dict(b)
            bt["task"] = [riv.get(t, t) for t in b["task"]]
            torch.manual_seed(1000 + i)
            pt = m.predict_action(examples=bt)["normalized_actions"]
            tr = np.linalg.norm((pt - alvo)[:, :, xyz] * esc[:, :, xyz], axis=-1).mean(1) * 100
        for j, t in enumerate(b["task"]):
            v = {"cm": cm[j].mean(), "cm_fim": cm[j, -1], "dedos": (fp[j] == fa[j]).mean()}
            if tr is not None and riv.get(t, t) != t:
                v["lingua"] = float(cm[j].mean() < tr[j])
                v["razao"] = tr[j] / max(cm[j].mean(), 1e-3)
            if (~fa[j][0]) and fa[j].any():
                v["fecha"] = float(fp[j].any() and abs(int(np.argmax(fp[j])) - int(np.argmax(fa[j]))) <= 9)
            e = por.setdefault(objeto(t), {k: [] for k in A})
            for k, x in v.items():
                A[k].append(float(x))
                e[k].append(float(x))
    nomes = {"cm": "erro_mao_dir_cm", "cm_fim": "erro_mao_dir_fim_cm", "dedos": "dedos_dir_acerto_%",
             "lingua": "lingua_%", "razao": "lingua_razao", "fecha": "fecha_no_tempo_%"}

    def resume(D):
        r = {}
        for k, x in D.items():
            if x:
                val = np.median(x) if k == "razao" else np.mean(x)
                r[nomes[k]] = float(val * (100 if k in ("dedos", "lingua", "fecha") else 1))
        return r
    r = resume(A)
    for o, D in por.items():
        r.update({f"{o}/{k}": v for k, v in resume(D).items()})
    return r


if __name__ == "__main__":   # prepara o cache da validação ANTES do treino (CPU):  python wla_v2.py <config_treino.yaml>
    import sys as _s
    import yaml as _y
    _s.path.insert(0, ".")
    cfg = _y.safe_load(open(_s.argv[1]))
    v2 = cfg["trainer"]["v2"]
    instala_split(v2["split"], "treino")
    fixas = {k: (v[0] if isinstance(v, list) else v) for k, v in (v2.get("frases") or {}).items()}
    lt = lotes_validacao(cfg["datasets"]["vla_data"]["data_config_path"], v2.get("val_lotes", 24), v2.get("val_lote", 4),
                         fixas, v2["val_cache"])
    print(f"✅ {len(lt)} lotes de validação em {v2['val_cache']}: {sorted({t for b in lt for t in b['task']})}")
