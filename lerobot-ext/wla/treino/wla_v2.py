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



# ── MARCADOR (v4, 22/10): o alvo desenhado na imagem da cabeça — o WLA segue a imagem muito melhor que a frase ──
COR_MARCADOR = (1.0, 0.0, 1.0)   # magenta: não existe na cena


def desenha_marcador(img, x, y, raio=0.035, espessura=3.0):
    """Anel no ponto (x, y) em 0..1 da imagem. img: tensor CHW float 0..1 (treino) ou numpy HWC uint8 (robô).
    Devolve uma cópia."""
    if isinstance(img, torch.Tensor):
        c, h, w = img.shape
        yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
        d = ((xx - x * w) ** 2 + (yy - y * h) ** 2).sqrt()
        anel = (d - raio * w).abs() <= espessura / 2
        out = img.clone()
        for k in range(min(c, 3)):
            out[k][anel] = COR_MARCADOR[k] * (1.0 if out.max() <= 1.0 + 1e-3 else 255.0)
        return out
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((xx - x * w) ** 2 + (yy - y * h) ** 2)
    out = img.copy()
    out[np.abs(d - raio * w) <= espessura / 2] = (np.array(COR_MARCADOR) * 255).astype(out.dtype)
    return out


def cola_distratores(img, banco, alvo, pool, n, perto=0.5, dist_min=0.07):
    """v5: cola n recortes (canecas/maçãs de outros quadros) na imagem da cabeça (CHW float), com borda suave, em
    lugares reais de objetos (pool) ou perto do alvo, nunca a menos de dist_min do alvo (x, y em 0..1)."""
    import torch.nn.functional as F
    c, h, w = img.shape
    out = img.clone()
    for _ in range(n):
        for _tent in range(20):
            if random.random() < perto:
                ang, r = random.uniform(0, 2 * np.pi), random.uniform(dist_min, 0.22)
                x, y = alvo[0] + r * np.cos(ang), alvo[1] + r * np.sin(ang) * 0.8
            else:
                x, y = pool[random.randrange(len(pool))] + np.random.normal(0, 0.02, 2)
            if 0.06 < x < 0.94 and 0.08 < y < 0.92 and np.hypot(x - alvo[0], y - alvo[1]) >= dist_min:
                break
        else:
            continue
        rc = banco[random.randrange(len(banco))]
        L = max(16, int(rc.shape[-1] * random.uniform(0.8, 1.2) * w / 448))
        rc = F.interpolate(rc[None], size=(L, L), mode="bilinear", align_corners=False)[0]
        rc = (rc * random.uniform(0.9, 1.1)).clamp(0, 1)
        yy, xx = torch.meshgrid(torch.arange(L, dtype=torch.float32), torch.arange(L, dtype=torch.float32), indexing="ij")
        d = ((xx - L / 2 + 0.5) ** 2 + (yy - L / 2 + 0.5) ** 2).sqrt()
        alfa = ((0.44 * L - d) / 4).clamp(0, 1)
        x0, y0 = int(x * w - L / 2), int(y * h - L / 2)
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(w, x0 + L), min(h, y0 + L)
        if xb <= xa or yb <= ya:
            continue
        sub = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
        a_ = alfa[sub]
        out[:3, ya:yb, xa:xb] = out[:3, ya:yb, xa:xb] * (1 - a_) + rc[:3][(slice(None),) + sub] * a_
    return out


class Marcadores:
    """Onde desenhar o marcador em cada quadro: os pontos que o ER-1 deu (alvos.jsonl, a cada 10 quadros) e a fase
    pelo episodios.jsonl — o objeto a PEGAR até a pega (+10 quadros), depois o lugar de COLOCAR (se a tarefa tem),
    depois nada. É a mesma regra que o robô usa ao vivo (pegar até os dedos fecharem)."""

    def __init__(self, alvos_jsonl, episodios_jsonl):
        self.pts, self.pega = {}, {}
        for d in map(json.loads, open(episodios_jsonl)):
            self.pega[(d["dataset"], d["episodio"])] = (None if d.get("comeca_segurando") else d.get("pega_quadro"))
        self.descartados = 0
        for d in map(json.loads, open(alvos_jsonl)):
            if d.get("ponto"):
                x, y = d["ponto"][0] / 1000, d["ponto"][1] / 1000
                # o alvo está sempre na mesa: ponto no fundo (alto) ou na borda da imagem é erro do ER-1 (25% do
                # "colocar no X" na cena antiga caía na borda, 23/10)
                if y < 0.2 or x < 0.08 or x > 0.9:
                    self.descartados += 1
                    continue
                self.pts.setdefault((d["dataset"], d["episodio"], d["alvo"]), []).append(
                    (d["quadro"], d["ponto"][0] / 1000, d["ponto"][1] / 1000))
        self.pool = None   # lugares reais de objetos na mesa (para colar distratores do v5)
        # SALTOS ISOLADOS do ER-1 (um quadro aponta outra coisa e o seguinte volta): mediana móvel de 5 pontos
        # (~1,7 s). Não é a mediana do episódio: a cabeça gira com a cintura e o alvo anda na imagem de verdade.
        self.corrigidos = 0
        for k, v in self.pts.items():
            v.sort()
            a = np.array([[x, y] for _, x, y in v])
            for i in range(len(v)):
                viz = a[max(0, i - 2):i + 3]
                med = np.median(viz, 0)
                if len(viz) >= 3 and np.linalg.norm(a[i] - med) > 0.08:
                    v[i] = (v[i][0], float(med[0]), float(med[1]))
                    self.corrigidos += 1
        # lugares de objetos POR CENA (dataset): o distrator vai para a mesa daquela cena, não para o ar
        self.pool = {}
        for (ds, _e, alvo), v in self.pts.items():
            if alvo == "pegar":
                self.pool.setdefault(ds, []).extend([x, y] for _, x, y in v)
        self.pool = {ds: np.array(v, dtype=np.float32) for ds, v in self.pool.items()}
        todos = np.concatenate(list(self.pool.values())) if self.pool else np.zeros((1, 2), np.float32)
        self.pool_geral = todos

    def ponto(self, ds, ep, f):
        pg = self.pega.get((ds, ep))
        for alvo in ("pegar", "colocar"):
            lst = self.pts.get((ds, ep, alvo))
            if not lst:
                continue
            if alvo == "pegar" and pg is not None and f > pg + 10:
                continue
            if alvo == "colocar" and lst[0][0] > f + 10:
                continue
            q = min(lst, key=lambda t: abs(t[0] - f))
            return q[1], q[2]
        return None


# ── V7 (28/10): AUMENTO DE IMAGEM e CÂMERAS DESLIGADAS (só no treino) ────────────────────────────────────────
# As imagens chegam como tensor (C, H, W) em 0..1. A cabeça e a profundidade dela recebem a MESMA geometria
# (continuam alinhadas); cada punho, a sua. Cor e ruído não entram na profundidade (o mapa de cor É a medida).
# Câmera desligada = image_mask False: o modelo da Unitree tira o papel do prompt (QwenMMDiT, linha
# `roles = [r ... if batch["image_mask"][r][i]]`), igual a uma câmera que não existe — e não uma imagem preta.
CABECA = ("head_left", "head_depth")
PUNHOS = ("cam_wrist_left", "cam_wrist_right")


def _geometria(imgs, graus, desloca):
    import torchvision.transforms.v2.functional as F
    if not imgs:
        return imgs
    _, h, w = imgs[0].shape
    ang = random.uniform(-graus, graus)
    tx, ty = (int(round(random.uniform(-desloca, desloca) * n)) for n in (w, h))
    esc = 1.0 + random.uniform(0, desloca)          # um pouco de zoom: não aparece borda preta na rotação
    return [F.affine(x, angle=ang, translate=[tx, ty], scale=esc, shear=[0.0, 0.0]) for x in imgs]


def _cor(x, c):
    import torchvision.transforms.v2.functional as F
    x = F.adjust_brightness(x, 1 + random.uniform(-c["brilho"], c["brilho"]))
    x = F.adjust_contrast(x, 1 + random.uniform(-c["contraste"], c["contraste"]))
    x = F.adjust_saturation(x, 1 + random.uniform(-c["saturacao"], c["saturacao"]))
    if c.get("tom", 0) > 0:
        x = F.adjust_hue(x, random.uniform(-c["tom"], c["tom"]))
    if c.get("ruido_std", 0) > 0:
        x = x + torch.randn_like(x) * c["ruido_std"]
    return x.clamp(0, 1)


def aumenta_imagens(item, c):
    """c = trainer.v2.aumento_imagem. Mexe em item["images"] (papel -> tensor CHW 0..1)."""
    imgs, mask = item["images"], item.get("image_mask", {})
    grupos = [[r for r in CABECA if r in imgs]] + [[r] for r in PUNHOS if r in imgs]
    for g in grupos:
        g = [r for r in g if mask.get(r, True) and torch.is_tensor(imgs[r]) and imgs[r].is_floating_point()]
        novas = _geometria([imgs[r] for r in g], c.get("rotacao_graus", 3.0), c.get("desloca_frac", 0.04))
        for r, x in zip(g, novas):
            imgs[r] = x if (r == "head_depth" and not c.get("na_profundidade", False)) else _cor(x, c)
    return item


def desliga_cameras(item, c):
    """c = trainer.v2.desliga_camera {punhos, um_punho, cabeca}: uma situação por amostra, nunca todas."""
    imgs, mask = item["images"], item.setdefault("image_mask", {})
    sorteio, fora = random.random(), ()
    p1, p2, p3 = c.get("punhos", 0.0), c.get("um_punho", 0.0), c.get("cabeca", 0.0)
    if sorteio < p1:
        fora = PUNHOS
    elif sorteio < p1 + p2:
        fora = (random.choice(PUNHOS),)
    elif sorteio < p1 + p2 + p3:
        fora = CABECA
    vivas = [r for r in imgs if mask.get(r, True) and r not in fora]
    if not vivas:          # sobraria nenhuma: não desliga
        return item
    for r in fora:
        if r in imgs:
            imgs[r] = torch.zeros_like(imgs[r])
            mask[r] = False
    return item


class Balanceado(torch.utils.data.Dataset):
    """Invólucro do SingleSourceDataset com os índices repetidos por tarefa e, opcionalmente, a FRASE da tarefa
    trocada pela descritiva (frases = {frase do dataset: frase nova}). Fica no nível do módulo para viajar
    (pickle) até os leitores, que são processos novos (spawn): um ajuste na CLASSE não chegava lá (21/10)."""

    def __init__(self, ds, mapa, frases=None, marcadores=None, papel_cabeca="head_left", treino=True, texto_vazio=0.0,
                 distratores=None, aumento=None, desliga=None):
        self.ds, self.mapa, self.frases = ds, mapa, frases or {}
        self.aumento, self.desliga = aumento, desliga   # v7: aumento de imagem e câmeras desligadas (só treino)
        self.distratores = distratores   # v5: {"banco": (N,3,L,L), "prob": 0.7, "max": 3}
        self.texto_vazio = texto_vazio   # CFG: fração das amostras com a frase VAZIA (o modelo aprende "sem frase")
        self.marcadores, self.papel, self.treino = marcadores, papel_cabeca, treino
        self.ep_quadro = None
        if marcadores is not None:   # episódio e quadro de cada índice local (para achar o ponto do ER-1)
            self.nomes = [Path(td).name for td in ds._task_dirs]
            self.ep_quadro = []
            for d in ds._datasets:
                t = d.hf_dataset.data
                self.ep_quadro.append((t.column("episode_index").to_numpy(), t.column("frame_index").to_numpy()))
            self.cum = np.array(ds._cum_lengths)

    def __len__(self):
        return len(self.mapa)

    def onde(self, j):
        k = int(np.searchsorted(self.cum, j, side="right"))
        local = j - (self.cum[k - 1] if k > 0 else 0)
        ep, fr = self.ep_quadro[k]
        return self.nomes[k], int(ep[local]), int(fr[local])

    def marca(self, item, j, sortear):
        """Desenha o marcador do quadro j na imagem da cabeça. sortear: tremido de ~1,5% e 10% sem marcador."""
        cena = self.onde(j)
        p = self.marcadores.ponto(*cena)
        if p is None or (sortear and random.random() < 0.1):
            item["marcador"] = torch.tensor([-1.0, -1.0])
            return item
        x, y = p
        if sortear:
            x, y = x + random.gauss(0, 0.015), y + random.gauss(0, 0.015)
        x, y = float(np.clip(x, 0.02, 0.98)), float(np.clip(y, 0.02, 0.98))
        dz = self.distratores
        if sortear and dz is not None and random.random() < dz.get("prob", 0.7):
            item["images"][self.papel] = cola_distratores(item["images"][self.papel], dz["banco"], (x, y),
                                                          self.marcadores.pool.get(cena[0], self.marcadores.pool_geral),
                                                          random.randint(1, dz.get("max", 3)))
        item["cabeca_limpa"] = item["images"][self.papel]
        item["images"][self.papel] = desenha_marcador(item["images"][self.papel], x, y)
        item["marcador"] = torch.tensor([x, y])
        return item

    def __getitem__(self, i):
        j = int(self.mapa[i])
        item = self.ds[j]
        if self.frases and item.get("task") in self.frases:
            v = self.frases[item["task"]]
            item["task"] = random.choice(v) if isinstance(v, (list, tuple)) else v   # lista = variações sorteadas
        if self.marcadores is not None:
            item = self.marca(item, j, sortear=self.treino)
            item.pop("cabeca_limpa", None)
        if self.treino and self.texto_vazio > 0 and random.random() < self.texto_vazio:
            item["task"] = ""
        if self.treino and self.aumento:
            item = aumenta_imagens(item, self.aumento)
        if self.treino and self.desliga:
            item = desliga_cameras(item, self.desliga)
        return item

    def __getattr__(self, nome):   # normalização, estatísticas etc. vêm do dataset de dentro
        if nome in ("ds", "mapa", "frases", "marcadores", "ep_quadro", "papel", "treino", "nomes", "cum", "texto_vazio", "distratores",
                    "aumento", "desliga") \
                or nome.startswith("__"):
            raise AttributeError(nome)
        return getattr(self.ds, nome)


def instala_balanceamento(alfa=1.0, frases=None, marcadores=None, texto_vazio=0.0, distratores=None, aumento=None,
                          desliga=None):
    """Repete os quadros das tarefas menores: fator = (maior / tamanho) ** alfa (0 = sem repetir). frases: troca
    a frase da tarefa (ex.: "Pick up the white mug." -> "... from the side with the right hand upright.")."""
    from unifolm_wla.dataloader.multi_source_dataset import dataloader as DL
    if getattr(DL, "_balanceado", None) is not None or (alfa <= 0 and not frases and marcadores is None
                                                        and not texto_vazio and not aumento and not desliga):
        return
    if aumento:
        print(f"🎨 aumento de imagem: {aumento}", flush=True)
    if desliga:
        print(f"📷 câmeras desligadas no treino: {desliga}", flush=True)
    if distratores:
        print(f"🧸 distratores: {len(distratores['banco'])} recortes, em {distratores.get('prob', 0.7):.0%} das amostras, "
              f"até {distratores.get('max', 3)} por imagem", flush=True)
    if texto_vazio:
        print(f"🔇 frase vazia em {texto_vazio:.0%} das amostras (para guiar pela frase na inferência, CFG)", flush=True)
    if marcadores is not None:
        print(f"🎯 marcador na imagem da cabeça: {len(marcadores.pts)} trechos rotulados pelo ER-1 "
              f"({marcadores.corrigidos} saltos isolados corrigidos, {marcadores.descartados} fora da mesa descartados)",
              flush=True)
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
        return Balanceado(ds, np.concatenate(mapa), frases, marcadores, texto_vazio=texto_vazio, distratores=distratores,
                          aumento=aumento, desliga=desliga)

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
# v7 (28/10): frases em inglês, português e espanhol
_CANECA = ("mug", "cup", "caneca", "xícara", "xicara", "copo", "taza", "tazón")
_MACA = ("apple", "maçã", "maca", "manzana")
_COADOR = ("strainer", "coador", "filtro", "dripper", "colador", "cafetera")


def objeto(frase):
    f = (frase or "").lower()
    return "caneca" if any(k in f for k in _CANECA) else "maca" if any(k in f for k in _MACA) else "?"


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
    base = ds.ds if isinstance(ds, Balanceado) else ds   # sem repetir quadros nem sortear a variação da frase
    vds = base
    if isinstance(ds, Balanceado) and ds.marcadores is not None:   # v4: marcador SEM tremido + a cabeça limpa
        vds = ValMarcada(Balanceado(base, np.arange(len(base)), None, ds.marcadores, ds.papel, treino=False))
    idx = torch.randperm(len(base), generator=torch.Generator().manual_seed(0))[: n_lotes * lote].tolist()
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(vds, idx), batch_size=lote, shuffle=False,
                                     num_workers=leitores, collate_fn=_collate_val)
    lotes = []
    for b in dl:
        if frases_fixas:
            b["task"] = [frases_fixas.get(t, t) for t in b["task"]]
        lotes.append(b)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save(lotes, cache)
    return lotes


class ValMarcada(torch.utils.data.Dataset):
    """Validação do v4: marcador fixo, guardando a imagem da cabeça LIMPA para mover o marcador depois."""

    def __init__(self, bal):
        self.bal = bal

    def __len__(self):
        return len(self.bal.mapa)

    def __getitem__(self, j):
        item = self.bal.ds[j]
        item = self.bal.marca(item, j, sortear=False)
        if "cabeca_limpa" not in item:
            item["cabeca_limpa"] = item["images"][self.bal.papel]
        return item


def _collate_val(itens):
    from unifolm_wla.dataloader.multi_source_dataset.dataloader import collate_fn
    extra = {k: torch.stack([it.pop(k) for it in itens]) for k in ("marcador", "cabeca_limpa") if k in itens[0]}
    b = collate_fn(itens)
    b.update(extra)
    return b


def _orig_frase(frases_fixas):
    inv = {}
    for k, v in (frases_fixas or {}).items():
        inv[v[0] if isinstance(v, (list, tuple)) else v] = k
    return inv


def tarefa(frase):
    """v7: a tarefa (não só o objeto) — o coador separado do pegar caneca."""
    f = (frase or "").lower()
    return "coador" if any(k in f for k in _COADOR) else objeto(frase)


@torch.no_grad()
def metricas_validacao(m, lotes, troca_frase=True):
    """erro da mão direita (cm, no trecho e no fim), dedos, fecha no tempo, troca de frase — total e por objeto."""
    from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES
    sx, fd = SLICES["right_xyz_rotvec"], SLICES["right_fig6d"]
    xyz = slice(sx.start, sx.start + 3)
    riv = rivais(sorted({t for b in lotes for t in b["task"]}))
    H = m.action_horizon
    A = {k: [] for k in ("cm", "cm_fim", "dedos", "lingua", "razao", "fecha", "yaw", "estado", "sem_punhos")}
    por = {}
    for i, b in enumerate(lotes):
        torch.manual_seed(1000 + i)
        pred = m.predict_action(examples=b)["normalized_actions"]
        if i < 6:
            # v7: o modelo USA o estado? (v4..v6: estado aleatório mudava a ação em só 0,005 — projetor congelado)
            bs = dict(b)
            bs["state"] = torch.randn_like(b["state"])
            torch.manual_seed(1000 + i)
            A["estado"].append(float(np.abs(m.predict_action(examples=bs)["normalized_actions"] - pred).mean()))
            # v7: sem as câmeras dos punhos (como no treino com câmeras desligadas): erro da mão direita
            if "image_mask" in b and any(r in b["image_mask"] for r in PUNHOS):
                bp = dict(b)
                bp["image_mask"] = {r: (torch.zeros_like(v) if r in PUNHOS else v) for r, v in b["image_mask"].items()}
                torch.manual_seed(1000 + i)
                pp = m.predict_action(examples=bp)["normalized_actions"]
                al = b["action"].cpu().numpy()[:, -m.action_horizon:, :]
                es = b["action_norm_scale"].cpu().numpy()[:, None, :]
                A["sem_punhos"] += (np.linalg.norm((pp - al)[:, :, xyz] * es[:, :, xyz], axis=-1).mean(1) * 100).tolist()
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
            # v7: o YAW da cintura acompanha a demonstração? (trechos em que ela gira > 0,05 rad: giro previsto/real)
            yw = SLICES["waist_joint"].start
            gd = (alvo[j, -1, yw] - alvo[j, 0, yw]) * esc[j, 0, yw]
            if abs(gd) > 0.05:
                v["yaw"] = ((pred[j, -1, yw] - pred[j, 0, yw]) * esc[j, 0, yw]) / gd
            e = por.setdefault(tarefa(t), {k: [] for k in ("cm", "cm_fim", "dedos", "lingua", "razao", "fecha",
                                                            "yaw")})
            for k, x in v.items():
                A[k].append(float(x))
                e[k].append(float(x))
    # SEGUE O MARCADOR (v4): o mesmo quadro com o marcador 12% mais à DIREITA na imagem (= à direita do robô, -y):
    # a mão prevista no fim do trecho tem que ir para a direita. marcador_sentido_% = em quantos foi (> 0,5 cm),
    # marcador_desloca_cm = quanto a ponta da trajetória mudou.
    if "marcador" in lotes[0]:
        A["m_sentido"], A["m_desloca"] = [], []
        y_i = sx.start + 1
        for i, b in enumerate(lotes):
            ok = (b["marcador"][:, 0] >= 0).numpy()
            if not ok.any():
                continue
            esc = b["action_norm_scale"].cpu().numpy()
            torch.manual_seed(1000 + i)
            p0 = m.predict_action(examples=b)["normalized_actions"]
            bm = dict(b)
            bm["images"] = dict(b["images"])
            cab = b["cabeca_limpa"].clone()
            papel = next(r for r in b["images"] if "head" in r)
            for j in range(len(cab)):
                if ok[j]:
                    x, y = b["marcador"][j].tolist()
                    cab[j] = desenha_marcador(cab[j], min(x + 0.12, 0.98), y)
            bm["images"][papel] = cab
            torch.manual_seed(1000 + i)
            p1 = m.predict_action(examples=bm)["normalized_actions"]
            for j in np.nonzero(ok)[0]:
                dy = (p1[j, -1, y_i] - p0[j, -1, y_i]) * esc[j, y_i] * 100
                dfim = np.linalg.norm((p1[j, -1, xyz] - p0[j, -1, xyz]) * esc[j, xyz]) * 100
                A["m_sentido"].append(float(dy < -0.5))
                A["m_desloca"].append(float(dfim))
    nomes = {"cm": "erro_mao_dir_cm", "cm_fim": "erro_mao_dir_fim_cm", "dedos": "dedos_dir_acerto_%",
             "lingua": "lingua_%", "razao": "lingua_razao", "fecha": "fecha_no_tempo_%",
             "m_sentido": "marcador_sentido_%", "m_desloca": "marcador_desloca_cm", "yaw": "yaw_giro_razao",
             "estado": "estado_sensibilidade", "sem_punhos": "sem_punhos_erro_cm"}

    def resume(D):
        r = {}
        for k, x in D.items():
            if x:
                val = np.median(x) if k in ("razao", "yaw") else np.mean(x)
                r[nomes[k]] = float(val * (100 if k in ("dedos", "lingua", "fecha", "m_sentido") else 1))
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
    mk = v2.get("marcador")
    instala_balanceamento(float(v2.get("balanceia", 1.0)), None, Marcadores(mk["alvos"], mk["episodios"]) if mk else None)
    fixas = {k: (v[0] if isinstance(v, list) else v) for k, v in (v2.get("frases") or {}).items()}
    lt = lotes_validacao(cfg["datasets"]["vla_data"]["data_config_path"], v2.get("val_lotes", 24), v2.get("val_lote", 4),
                         fixas, v2["val_cache"])
    print(f"✅ {len(lt)} lotes de validação em {v2['val_cache']}: {sorted({t for b in lt for t in b['task']})}")
