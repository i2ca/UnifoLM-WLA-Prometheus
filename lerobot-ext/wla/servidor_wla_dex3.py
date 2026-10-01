#!/usr/bin/env python
"""
Servidor do UnifoLM-WLA para o G1 com mão Dex3 (fig6d) — extensão do servidor oficial.

O `action_server_wbc_msgpack_unitree` oficial só serve a garra Dex1 (a própria doc da Unitree diz: para
mão de dedos é preciso acrescentar left/right_fig6d à máscara e ao protocolo). Esta subclasse faz
exatamente isso para o modelo afinado com o dataset do Prometheus (converte_dataset_dex3_wla.py):

  máscara da ação:  pose das mãos + left/right_fig6d (NÃO a garra) + cintura + andar/altura + pernas
  obs a mais:       observation.state.left_fig6d / right_fig6d   (6, 0..1 — ver o conversor)
  ação a mais:      action.left_fig6d / right_fig6d               (1, T, 6), 0..1
  (as chaves de garra do protocolo oficial deixam de ser exigidas na observação)

Mesmo protocolo websocket/msgpack do oficial; o roda_wla_real.py detecta pelo metadata.

    cd ~/DEV/unifolm-wla && ~/miniforge3/envs/wla/bin/python \\
        ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/servidor_wla_dex3.py \\
        --ckpt_path playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors \\
        --lora_run playground/Checkpoints/lora_prometheus_dex3_maca_x_preto --lora_passo 8000 --port 8601

Com --lora_run, --ckpt_path é o modelo BASE: o LoRA é injetado com a config do treino e os pesos afinados
(o checkpoint completo ou o delta) são carregados por cima — o repositório não tem carregador de LoRA para
inferência (o from_pretrained no checkpoint afinado perde as chaves base_layer/lora_).
"""
import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "model_server"))
from model_server import action_server_wbc_msgpack_unitree as srv  # noqa: E402
from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES, STATE_DIM, STATE_SLICES  # noqa: E402

PARTES_ACAO = ["left_xyz_rotvec", "left_fig6d", "right_xyz_rotvec", "right_fig6d", "waist_joint",
               "base_vx_vy", "base_vw", "height", "left_leg_joint", "right_leg_joint"]
CHAVES_EXTRA = ["observation.state.left_fig6d", "observation.state.right_fig6d",
                "action.left_fig6d", "action.right_fig6d"]


def mascara_dex3() -> np.ndarray:
    m = np.zeros(SLICES["right_leg_joint"].stop, dtype=bool)
    for k in PARTES_ACAO:
        m[SLICES[k]] = True
    return m


class ServidorDex3(srv.ActionServerWBCMsgpack):
    def __init__(self, args):
        super().__init__(args)
        self._action_mask = mascara_dex3()

    @property
    def metadata(self):
        md = dict(super().metadata)
        md["env"] = "unifolm_wla_unitree_dex3_fig6d"
        md["data_keys"] = [k for k in md["data_keys"] if "gripper" not in k] + CHAVES_EXTRA
        md["maos"] = "dex3_fig6d"
        return md

    def _build_state_unnorm(self, obs):
        state = np.zeros(STATE_DIM, dtype=np.float32)
        mask = np.zeros(STATE_DIM, dtype=bool)

        def poe(chave_obs, fatia, n):
            v = srv._last_frame(obs[chave_obs]).astype(np.float32).ravel()[:n]
            state[STATE_SLICES[fatia]] = v
            mask[STATE_SLICES[fatia]] = True

        poe("observation.state.left_ee_6d", "left_xyz_rot6d", 9)
        poe("observation.state.right_ee_6d", "right_xyz_rot6d", 9)
        poe("observation.state.left_fig6d", "left_fig6d", 6)
        poe("observation.state.right_fig6d", "right_fig6d", 6)
        lower = srv._last_frame(obs["observation.state.lower_body"]).astype(np.float32).ravel()
        for fatia, (a, b) in (("left_leg_joint", (0, 6)), ("right_leg_joint", (6, 12)), ("waist_joint", (12, 15))):
            state[STATE_SLICES[fatia]] = lower[a:b]
            mask[STATE_SLICES[fatia]] = True
        return state, mask

    def _encode_action(self, pred_norm, unnorm_key, state_unnorm):
        out = super()._encode_action(pred_norm, unnorm_key, state_unnorm)
        norm = self._norm_arrays[unnorm_key]
        unnorm = (pred_norm * norm["action_scale"] + norm["action_offset"]).astype(np.float32)
        for lado in ("left", "right"):
            out[f"action.{lado}_fig6d"] = np.clip(unnorm[:, SLICES[f"{lado}_fig6d"]], 0.0, 1.0)[None]
            out.pop(f"action.{lado}_gripper", None)
        return out


def instala_carregador_lora(run, passo):
    """Troca o from_pretrained que o servidor oficial chama por: base -> LoRA -> pesos afinados -> normalização."""
    from pathlib import Path

    import json
    import yaml
    from safetensors.torch import load_file
    from unifolm_wla.training.trainer_utils.trainer_tools import TrainerUtils

    run = Path(run)
    ck = run / "checkpoints"
    fino = next((p for p in (ck / f"steps_{passo}_model.safetensors", ck / f"steps_{passo}_delta.safetensors")
                 if p.exists()), None)
    if fino is None:
        sys.exit(f"sem steps_{passo}_model/_delta.safetensors em {ck}")
    original = srv.baseframework.from_pretrained

    def carrega(base_ckpt, *args, **kw):
        m = original(base_ckpt, *args, **kw)
        TrainerUtils.apply_lora_adapters(m, yaml.safe_load(open(run / "config.yaml"))["trainer"]["lora"])
        falta, sobra = m.load_state_dict(load_file(str(fino)), strict=False)
        if sobra:
            sys.exit(f"chaves do checkpoint que o modelo não tem: {sobra[:5]}")
        if falta and "delta" not in fino.name:
            sys.exit(f"o modelo ficou sem {len(falta)} tensores: {falta[:5]}")
        m.norm_stats = json.load(open(run / "dataset_statistics.json"))
        print(f"LoRA carregado: {fino} | normalização {list(m.norm_stats)}", flush=True)
        return m

    srv.baseframework.from_pretrained = carrega


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--instruction", default="")
    p.add_argument("--unnorm_key", default="Prometheus_G1_Dex3")
    p.add_argument("--use_bf16", action="store_true", default=True)
    p.add_argument("--image_size", type=int, nargs=2, default=[320, 448])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8601)
    p.add_argument("--debug_save_dir", default=None)
    p.add_argument("--lora_run", default=None, help="pasta do treino LoRA (config.yaml, dataset_statistics.json)")
    p.add_argument("--lora_passo", type=int, default=8000)
    a = p.parse_args()
    if a.lora_run:
        instala_carregador_lora(a.lora_run, a.lora_passo)
    # WARNING: o servidor oficial tem um logging.info com 2 "%s" e 1 argumento, que quebra em INFO.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    s = ServidorDex3(a)
    if a.unnorm_key not in s._norm_arrays:
        sys.exit(f"normalização '{a.unnorm_key}' não está no checkpoint: {list(s._norm_arrays)}")
    print(f"servidor WLA Dex3 (fig6d) | normalização {a.unnorm_key} | ws://{a.host}:{a.port}", flush=True)
    s.run(a.host, a.port)


if __name__ == "__main__":
    main()
