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
        aplica_lora(m, yaml.safe_load(open(run / "config.yaml"))["trainer"]["lora"])
        falta, sobra = m.load_state_dict(load_file(str(fino)), strict=False)
        if sobra:
            sys.exit(f"chaves do checkpoint que o modelo não tem: {sobra[:5]}")
        if falta and "delta" not in fino.name:
            sys.exit(f"o modelo ficou sem {len(falta)} tensores: {falta[:5]}")
        m.norm_stats = json.load(open(run / "dataset_statistics.json"))
        print(f"LoRA carregado: {fino} | normalização {list(m.norm_stats)}", flush=True)
        return m

    srv.baseframework.from_pretrained = carrega


def instala_amostragem(passos, amostras, escala_ruido):
    """MENOS RUÍDO NA AÇÃO (03/10): o DiT gera cada trecho por flow matching partindo de ruído aleatório, em 4 passos.
    Com o modelo ainda aprendendo, trechos seguidos discordam e a mão "desvia" do lugar. Aqui:
      passos       mais passos de integração do fluxo (trajetória mais limpa);
      amostras     K trechos com ruídos diferentes no MESMO lote e a MÉDIA deles (corta a variância);
      escala_ruido ruído inicial menor (<1 = mais perto da trajetória mais provável)."""
    import threading
    import torch
    from unifolm_wla.model.modules.action_model import MMDiT_ActionHeader as mh
    orig = mh.MMDiTFlowmatchingActionHead.predict_action
    trava = threading.Lock()

    def predict_action(self, vl_embs, state=None, action_mask=None, encoder_attention_mask=None, body_type_ids=None):
        if passos:
            self.num_inference_timesteps = passos
        K = max(1, amostras)
        rep = (lambda x: None if x is None else x.repeat_interleave(K, 0)) if K > 1 else (lambda x: x)
        with trava:
            randn = torch.randn
            if escala_ruido != 1.0:
                mh.torch.randn = lambda *a, **k: randn(*a, **k) * escala_ruido
            try:
                out = orig(self, rep(vl_embs), state=rep(state), action_mask=rep(action_mask),
                           encoder_attention_mask=rep(encoder_attention_mask), body_type_ids=rep(body_type_ids))
            finally:
                mh.torch.randn = randn
        if K > 1:
            out = out.view(vl_embs.shape[0], K, *out.shape[1:]).mean(dim=1)
        return out

    mh.MMDiTFlowmatchingActionHead.predict_action = predict_action
    print(f"amostragem: {passos or 'padrão'} passos de fluxo | média de {max(1, amostras)} trechos | "
          f"ruído inicial x{escala_ruido}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--instruction", default="")
    p.add_argument("--unnorm_key", default="Prometheus_G1_Dex3")
    p.add_argument("--use_bf16", action="store_true", default=True)
    p.add_argument("--image_size", type=int, nargs=2, default=[336, 448],
                   help="tamanho que o VLM recebe; 03/10: era 320x448, mas o treino (configs de dados) usa 336x448")
    # Latência medida na PGX (03/10): fluxo 4 x 1 amostra = 660 ms | 4 x 4 = 1390 ms | 6 x 2 = 1305 ms | 10 x 4 = 3200 ms.
    # O trecho dura 1,5 s: acima de ~1,1 s ele chega tarde. Padrão: 4 passos, média de 2, ruído x0,6.
    p.add_argument("--passos-fluxo", type=int, default=4, help="passos do flow matching do DiT (o treino usa 4)")
    p.add_argument("--amostras", type=int, default=2, help="trechos sorteados por consulta; a ação é a média")
    p.add_argument("--escala-ruido", type=float, default=0.6, help="escala do ruído inicial (1 = o original)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8601)
    p.add_argument("--debug_save_dir", default=None)
    p.add_argument("--lora_run", default=None, help="pasta do treino LoRA (config.yaml, dataset_statistics.json)")
    p.add_argument("--lora_passo", type=int, default=8000)
    a = p.parse_args()
    if a.lora_run:
        instala_carregador_lora(a.lora_run, a.lora_passo)
    instala_amostragem(a.passos_fluxo, a.amostras, a.escala_ruido)
    # WARNING: o servidor oficial tem um logging.info com 2 "%s" e 1 argumento, que quebra em INFO.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    s = ServidorDex3(a)
    if a.unnorm_key not in s._norm_arrays:
        sys.exit(f"normalização '{a.unnorm_key}' não está no checkpoint: {list(s._norm_arrays)}")
    print(f"servidor WLA Dex3 (fig6d) | normalização {a.unnorm_key} | ws://{a.host}:{a.port}", flush=True)
    s.run(a.host, a.port)


if __name__ == "__main__":
    main()
