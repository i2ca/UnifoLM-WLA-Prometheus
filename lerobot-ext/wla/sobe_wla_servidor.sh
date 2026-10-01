#!/bin/bash
# Sobe o SERVIDOR OFICIAL do UnifoLM-WLA-1.0 (websocket/msgpack, porta 8600) e deixa carregado.
# O executor do robô (roda_wla_real.py) só conecta nele. Log: ~/wla_servidor.log
WLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # esta pasta (lerobot-ext/wla)
UNIFOLM_WLA="${UNIFOLM_WLA:-$HOME/DEV/unifolm-wla}"                  # clone do unitreerobotics/unifolm-wla
PY="${WLA_PY:-$HOME/miniforge3/envs/wla/bin/python}"
for p in $(pgrep -f "action_server_wbc_msgpack_[u]nitree"); do kill $p; done
sleep 2
cd "$UNIFOLM_WLA"
setsid nohup "$PY" -u -m model_server.action_server_wbc_msgpack_unitree \
    --ckpt_path playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors \
    --unnorm_key UnifoLM_G1_Dex1 --host 0.0.0.0 --port 8600 > ~/wla_servidor.log 2>&1 < /dev/null &
echo "servidor WLA subindo (≈80 s) — ws://$(hostname -I | awk '{print $1}'):8600 — log ~/wla_servidor.log"
