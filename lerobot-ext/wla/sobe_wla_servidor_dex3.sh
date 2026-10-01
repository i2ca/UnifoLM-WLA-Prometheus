#!/bin/bash
# Sobe o servidor do WLA AFINADO para a Dex3 (servidor_wla_dex3.py: base + LoRA do passo 8000 do treino da
# maçã, fig6d nos dedos) na porta 8601 e deixa carregado. O executor conecta com
#   roda_wla_real.py --servidor ws://127.0.0.1:8601 --pose maca --tarefa "Pick up the apple and place it on the black X."
# Uso: [RUN=<pasta do treino>] sobe_wla_servidor_dex3.sh [passo]      (padrão 8000)      Log: ~/wla_servidor_dex3.log
WLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # esta pasta (lerobot-ext/wla)
UNIFOLM_WLA="${UNIFOLM_WLA:-$HOME/DEV/unifolm-wla}"                  # clone do unitreerobotics/unifolm-wla
PY="${WLA_PY:-$HOME/miniforge3/envs/wla/bin/python}"
PASSO=${1:-8000}
RUN=${RUN:-lora_prometheus_dex3_maca_medido}   # 01/10: treino com a mão MEDIDA (o de 30/09: lora_prometheus_dex3_maca_x_preto)
for p in $(pgrep -f "servidor_wla_[d]ex3.py"); do kill $p; done
sleep 2
cd "$UNIFOLM_WLA"
setsid nohup "$PY" -u "$WLA_DIR"/servidor_wla_dex3.py \
    --ckpt_path playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors \
    --lora_run playground/Checkpoints/$RUN --lora_passo "$PASSO" \
    --host 0.0.0.0 --port 8601 > ~/wla_servidor_dex3.log 2>&1 < /dev/null &
echo "servidor WLA Dex3 ($RUN, passo $PASSO) subindo (≈2 min) — ws://$(hostname -I | awk '{print $1}'):8601 — log ~/wla_servidor_dex3.log"
