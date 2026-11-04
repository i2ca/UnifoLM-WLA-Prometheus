#!/usr/bin/env python
"""LEITORES dos tokens do WLA (03/11): a informação sobre "o que está acontecendo" está DENTRO do WLA? Leitores
LINEARES simples (ridge / regressão logística) sobre os tokens extraídos (extrai_tokens_wla.py), validação cruzada
POR EPISÓDIO (5 dobras: nunca o mesmo episódio no treino e no teste), contra leitores que só têm a imagem crua e/ou
o estado — o que um ER-1 teria SEM ler o WLA.

Perguntas:
  progresso      em que ponto do episódio está (0..1)                      -> erro médio (% do episódio), menor = melhor
  vai_fechar     a mão vai FECHAR no próximo 1 s (aberta agora)             -> AUC, maior = melhor (0,5 = chute)
  tarefa         qual das 3 tarefas                                         -> acerto
  vai_falhar     episódio de PEGA ERRADA, olhando só ANTES da 1ª pega       -> AUC por episódio (caneca + maçã)

    python sonda_tokens.py --tokens /data/mrwlker/tokens_v8
"""
import argparse
from pathlib import Path

import numpy as np
import torch

CONJUNTOS = {
    "WLA: VLM (imagem)": ["vlm_img"],
    "WLA: VLM (tudo)": ["vlm_tudo"],
    "WLA: DiT": ["dit"],
    "WLA: plano de ação": ["plano"],
    "WLA: VLM + DiT + plano": ["vlm_tudo", "dit", "plano"],
    "sem WLA: imagem crua": ["img_crua"],
    "sem WLA: estado": ["estado"],
    "sem WLA: imagem crua + estado": ["img_crua", "estado"],
}


def carrega(pasta):
    D = [dict(np.load(f, allow_pickle=True)) for f in sorted(Path(pasta).glob("*.npz"))]
    X = {k: np.concatenate([d[k].astype(np.float32) for d in D]) for k in
         ("vlm_img", "vlm_txt", "vlm_est", "vlm_tudo", "dit", "plano", "img_crua", "estado")}
    meta = {k: np.concatenate([d[k] for d in D]) for k in ("ep", "quadro", "T", "dedos_agora", "dedos_1s", "falha")}
    meta["tarefa"] = np.concatenate([np.full(len(d["ep"]), i) for i, d in enumerate(D)])
    meta["grupo"] = meta["tarefa"] * 10000 + meta["ep"]
    nomes = [str(d["tarefa"]) for d in D]
    return X, meta, nomes


def padroniza(tr, te):
    mu, sd = tr.mean(0), tr.std(0) + 1e-6
    return (tr - mu) / sd, (te - mu) / sd


def ridge(Xtr, ytr, Xte, lam):
    Xtr, Xte = torch.tensor(Xtr), torch.tensor(Xte)
    A = Xtr.T @ Xtr + lam * len(Xtr) * torch.eye(Xtr.shape[1])
    w = torch.linalg.solve(A, Xtr.T @ (torch.tensor(ytr) - ytr.mean()))
    return (Xte @ w).numpy() + ytr.mean()


def logistica(Xtr, ytr, Xte, lam, classes=2):
    Xtr, Xte = torch.tensor(Xtr), torch.tensor(Xte)
    y = torch.tensor(ytr).long()
    W = torch.zeros(Xtr.shape[1], classes, requires_grad=True)
    b = torch.zeros(classes, requires_grad=True)
    peso = torch.bincount(y, minlength=classes).float()
    peso = (len(y) / (classes * peso.clamp(min=1)))
    opt = torch.optim.LBFGS([W, b], max_iter=200, line_search_fn="strong_wolfe")

    def f():
        opt.zero_grad()
        l = torch.nn.functional.cross_entropy(Xtr @ W + b, y, weight=peso) + lam * (W ** 2).sum()
        l.backward()
        return l
    opt.step(f)
    with torch.no_grad():
        return torch.softmax(Xte @ W + b, -1).numpy()


def auc(p, y):
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    ordem = np.argsort(p)
    r = np.empty(len(p))
    r[ordem] = np.arange(1, len(p) + 1)
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def dobras(grupos, k=5, semente=0):
    g = np.unique(grupos)
    np.random.default_rng(semente).shuffle(g)
    for i in range(k):
        te = np.isin(grupos, g[i::k])
        yield ~te, te


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--lam", type=float, default=1e-2)
    a = ap.parse_args()
    X, M, nomes = carrega(a.tokens)
    n = len(M["ep"])
    prog = (M["quadro"] / np.maximum(M["T"] - 1, 1)).astype(np.float32)
    # dedos MEDIDOS: aberta ~0,08, fechada no objeto ~0,31-0,39 (param no contato)
    vai_fechar = ((M["dedos_agora"] < 0.15) & (M["dedos_1s"] > 0.2)).astype(int)
    aberta = M["dedos_agora"] < 0.15
    # antes da 1ª pega de cada episódio (só caneca e maçã: o coador começa com a caneca na mão)
    antes = np.zeros(n, bool)
    for g in np.unique(M["grupo"]):
        s = np.where(M["grupo"] == g)[0]
        fech = np.where(M["dedos_agora"][s] > 0.2)[0]
        lim = fech[0] if len(fech) else len(s)
        antes[s[:lim]] = True
    pega = np.array(["Coador" not in nomes[t] for t in M["tarefa"]])
    print(f"{n} consultas | {len(np.unique(M['grupo']))} episódios | tarefas {nomes}\n"
          f"vai_fechar: {vai_fechar[aberta].sum()} positivos de {aberta.sum()} (mão aberta) | episódios de pega errada: "
          f"{len(np.unique(M['grupo'][M['falha'] == 1]))}\n")
    print(f"{'leitor':32s} {'progresso (erro %)':>19s} {'vai fechar (AUC)':>17s} {'tarefa (%)':>11s} "
          f"{'vai falhar (AUC ep.)':>21s}")
    for nome, chaves in CONJUNTOS.items():
        Xc = np.concatenate([X[k] for k in chaves], 1)
        e_prog, p_fecha, y_fecha, ac_tar = [], [], [], []
        p_falha_ep, y_falha_ep = [], []
        for tr, te in dobras(M["grupo"]):
            Xtr, Xte = padroniza(Xc[tr], Xc[te])
            e_prog.append(np.abs(np.clip(ridge(Xtr, prog[tr], Xte, a.lam), 0, 1) - prog[te]))
            m_tr, m_te = tr & aberta, te & aberta
            if vai_fechar[m_tr].sum() > 3:
                Xa, Xb = padroniza(Xc[m_tr], Xc[m_te])
                p_fecha.append(logistica(Xa, vai_fechar[m_tr], Xb, a.lam)[:, 1])
                y_fecha.append(vai_fechar[m_te])
            ac_tar.append(logistica(Xtr, M["tarefa"][tr], Xte, a.lam, 3).argmax(1) == M["tarefa"][te])
            f_tr, f_te = tr & antes & pega, te & antes & pega
            if M["falha"][f_tr].sum() > 0 and f_te.sum():
                Xa, Xb = padroniza(Xc[f_tr], Xc[f_te])
                p = logistica(Xa, M["falha"][f_tr], Xb, a.lam)[:, 1]
                for g in np.unique(M["grupo"][f_te]):
                    s = M["grupo"][f_te] == g
                    p_falha_ep.append(p[s].mean())
                    y_falha_ep.append(M["falha"][f_te][s][0])
        print(f"{nome:32s} {100 * np.concatenate(e_prog).mean():19.1f} "
              f"{auc(np.concatenate(p_fecha), np.concatenate(y_fecha)):17.2f} "
              f"{100 * np.concatenate(ac_tar).mean():11.1f} {auc(np.array(p_falha_ep), np.array(y_falha_ep)):21.2f}",
              flush=True)


if __name__ == "__main__":
    main()
