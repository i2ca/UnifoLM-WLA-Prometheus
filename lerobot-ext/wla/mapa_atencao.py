#!/usr/bin/env python
"""Mapa de atenção do UnifoLM-WLA: para onde as AÇÕES olham em cada câmera.

Roda na PGX, no env `wla`, a partir da raiz do repositório do WLA (não modifica nada dele):

    cd ~/DEV/unifolm-wla
    python ~/wla_testes/mapa_atencao.py --obs ~/wla_testes/obs_sim/obs_0000.npz ... --saida ~/wla_testes/atencao

Cada `obs_*.npz` é uma observação no protocolo do servidor (a ponte grava com --salvar_obs).

O que é medido: no DiT de ação (mmdit.py), os tokens de ação fazem atenção conjunta sobre
[ação, saída do VLM]. Envolvemos `dispatch_attention_fn` e, em cada chamada (16 camadas x 4
passos de difusão), calculamos softmax(q_ação · k) com a mesma máscara, e somamos a massa que
cai em cada token do VLM (média sobre cabeças e tokens de ação). Os tokens de imagem são
localizados pelo `image_token_id` e remontados na grade (h/2 x w/2) de `image_grid_thw`.
A ação é a mesma que o servidor devolveria: usamos o próprio ActionServerWBCMsgpack.
"""
import argparse
import logging
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "model_server"))
from unifolm_wla.model.modules.action_model.DiT_modules import mmdit  # noqa: E402
from model_server import action_server_wbc_msgpack_unitree as srv_mod  # noqa: E402


class Registro:
    ligado = False
    L = 0
    soma = None
    n = 0


REG = Registro()
_orig = mmdit.dispatch_attention_fn


def _espiao(q, k, v, attn_mask=None, **kw):
    if REG.ligado:
        B, S, H, D = q.shape
        A = S - REG.L
        with torch.no_grad():
            logit = torch.einsum("bqhd,bkhd->bhqk", q[:, :A].float(), k.float()) / math.sqrt(D)
            if attn_mask is not None:
                logit = logit.masked_fill(~attn_mask.reshape(B, 1, 1, S).bool(), float("-inf"))
            p = logit.softmax(-1)[0, :, :, A:].mean(dim=(0, 1))  # [L]: massa em cada token do VLM
            REG.soma = p if REG.soma is None else REG.soma + p
            REG.n += 1
    return _orig(q, k, v, attn_mask=attn_mask, **kw)


mmdit.dispatch_attention_fn = _espiao


def carrega_obs(caminho):
    z = np.load(caminho, allow_pickle=True)
    obs = {k.replace("__", "."): z[k] for k in z.files}
    if "instruction" in obs:
        obs["instruction"] = str(obs["instruction"])
    return obs


def desenha(att, inp, example, img_tok, st_tok, titulo):
    """att: [L] massa de atenção das ações em cada token do VLM -> (figura BGR, massas por câmera,
    massa no token de estado, resto)."""
    ids = inp["input_ids"][0].cpu().numpy()
    grades = inp["image_grid_thw"].cpu().numpy()
    pos_img = np.where(ids == img_tok)[0]
    paineis, massas, mapas, ini = [], [], [], 0
    for g in grades:
        t, h, w = (int(x) for x in g)
        n = t * (h // 2) * (w // 2)
        m = att[pos_img[ini:ini + n]].reshape(h // 2, w // 2)
        mapas.append(m)
        massas.append(float(m.sum()))
        ini += n
    # escala pelo percentil 99 de todos os tokens de imagem: poucos tokens de borda
    # ("sumidouros" de atenção) saturam e o resto do mapa fica visível
    vmax = float(np.percentile(np.concatenate([m.ravel() for m in mapas]), 99))
    for img, m, papel, massa in zip(example["image"], mapas, example["image_roles"], massas):
        bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        calor = cv2.resize((np.clip(m / vmax, 0, 1) * 255).astype(np.uint8), (bgr.shape[1], bgr.shape[0]),
                           interpolation=cv2.INTER_CUBIC)
        sobre = cv2.addWeighted(bgr, 0.55, cv2.applyColorMap(calor, cv2.COLORMAP_JET), 0.45, 0)
        gh, gw = m.shape
        for r in np.argsort(m.ravel())[::-1][:5]:  # 5 tokens mais fortes: círculo branco
            cy, cx = divmod(int(r), gw)
            cv2.circle(sobre, (int((cx + 0.5) * bgr.shape[1] / gw), int((cy + 0.5) * bgr.shape[0] / gh)),
                       10, (255, 255, 255), 2)
        cv2.putText(sobre, f"{papel}: {100 * massa:.1f}%", (8, 26), 0, 0.7, (255, 255, 255), 2)
        paineis.append(sobre)
    m_estado = float(att[ids == st_tok].sum()) if st_tok is not None else 0.0
    m_texto = 1.0 - sum(massas) - m_estado
    faixa = np.full((40, sum(p.shape[1] for p in paineis), 3), 30, np.uint8)
    cv2.putText(faixa, f"{titulo} | imagens {100 * sum(massas):.1f}%  estado {100 * m_estado:.1f}%  "
                f"texto/resto {100 * m_texto:.1f}%", (8, 27), 0, 0.55, (255, 255, 255), 1)
    return np.vstack([faixa, np.hstack(paineis)]), massas, m_estado, m_texto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", default="playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors")
    ap.add_argument("--unnorm_key", default="UnifoLM_G1_Dex1")
    ap.add_argument("--obs", nargs="+", required=True)
    ap.add_argument("--instrucao", default=None, help="sobrescreve a instrução gravada na obs")
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()

    # igual ao main() do servidor: sem isso o handler Rich do pacote transforma em exceção o
    # logging.info("... %s (backend=%s)", ckpt) com argumento faltando, no __init__ deles
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    args_srv = argparse.Namespace(ckpt_path=a.ckpt_path, instruction="", unnorm_key=a.unnorm_key, use_bf16=True,
                                  image_size=[320, 448], host="", port=0, debug_save_dir=None)
    srv = srv_mod.ActionServerWBCMsgpack(args_srv)
    iface = srv.model.qwen_vl_interface
    img_tok = iface.model.config.image_token_id
    st_tok = getattr(iface.model.config, "robot_state_token_id", None)
    guardado = {}
    orig_build = iface.build_qwenvl_inputs

    def build(*x, **kw):
        out = orig_build(*x, **kw)
        guardado["inp"] = out
        REG.L = out["input_ids"].shape[1]
        return out

    iface.build_qwenvl_inputs = build
    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)

    for caminho in a.obs:
        obs = carrega_obs(caminho)
        if a.instrucao:
            obs["instruction"] = a.instrucao
        prep = srv._build_example(obs)
        REG.soma, REG.n, REG.ligado = None, 0, True
        with torch.no_grad():
            srv.model.predict_action(examples=[prep["example"]])
        REG.ligado = False
        att = (REG.soma / REG.n).float().cpu().numpy()  # [L]
        titulo = f"{Path(caminho).parent.name}/{Path(caminho).stem} | '{obs['instruction']}'"
        figura, massas, m_estado, m_texto = desenha(att, guardado["inp"], prep["example"], img_tok, st_tok, titulo)
        nome = saida / f"{Path(caminho).parent.name}_{Path(caminho).stem}.jpg"
        cv2.imwrite(str(nome), figura)
        print(f"{nome.name}: imagens {[round(100 * x, 1) for x in massas]} estado {100 * m_estado:.1f} "
              f"texto {100 * m_texto:.1f} ({REG.n} chamadas de atenção)", flush=True)


if __name__ == "__main__":
    main()
