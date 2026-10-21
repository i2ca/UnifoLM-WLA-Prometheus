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
