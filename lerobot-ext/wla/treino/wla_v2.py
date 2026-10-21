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


def instala_split(caminho, parte):
    from unifolm_wla.dataloader.multi_source_dataset import lerobot_wrapper as LW
    split = json.load(open(caminho))
    cls = LW.LeRobotDatasetWithSelectedKeys
    if getattr(cls, "_split_v2", None):
        return
    orig = cls.__init__

    def __init__(self, *args, **kw):
        nome = Path(kw.get("root") or "").name
        if nome in split and kw.get("episodes") is None:
            kw["episodes"] = sorted(int(e) for e in split[nome][parte])
            print(f"📂 {nome}: {parte} = {len(kw['episodes'])} episódios", flush=True)
        orig(self, *args, **kw)

    cls.__init__ = __init__
    cls._split_v2 = parte


class Balanceado(torch.utils.data.Dataset):
    """Invólucro do SingleSourceDataset com os índices repetidos por tarefa. Fica no nível do módulo para viajar
    (pickle) até os leitores, que são processos novos (spawn): um ajuste na CLASSE não chegava lá (21/10)."""

    def __init__(self, ds, mapa):
        self.ds, self.mapa = ds, mapa

    def __len__(self):
        return len(self.mapa)

    def __getitem__(self, i):
        return self.ds[int(self.mapa[i])]

    def __getattr__(self, nome):   # normalização, estatísticas etc. vêm do dataset de dentro
        if nome in ("ds", "mapa") or nome.startswith("__"):
            raise AttributeError(nome)
        return getattr(self.ds, nome)


def instala_balanceamento(alfa=1.0):
    """Repete os quadros das tarefas menores: fator = (maior / tamanho) ** alfa."""
    from unifolm_wla.dataloader.multi_source_dataset import dataloader as DL
    if getattr(DL, "_balanceado", None) is not None or alfa <= 0:
        return
    orig = DL.create_single_source_dataset

    def cria(*args, **kw):
        ds = orig(*args, **kw)
        tam = [len(d) for d in ds._datasets]
        maior, mapa, ini = max(tam), [], 0
        for td, n in zip(ds._task_dirs, tam):
            k = (maior / n) ** alfa
            idx = np.arange(ini, ini + n)
            rep = np.concatenate([np.tile(idx, int(k)), idx[: int(round((k - int(k)) * n))]])
            mapa.append(rep)
            print(f"⚖️  {Path(td).name}: {n} quadros x{k:.2f} -> {len(rep)}", flush=True)
            ini += n
        return Balanceado(ds, np.concatenate(mapa))

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
