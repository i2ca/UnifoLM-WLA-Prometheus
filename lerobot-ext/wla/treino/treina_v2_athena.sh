#!/bin/bash
# TREINO V2 na Athena (21/10), de ponta a ponta — rode DENTRO de uma screen:
#   screen -S treino_v2
#   GPU=0,1 bash ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/treino/treina_v2_athena.sh
#   v3 (frases descritivas + etapas): CONFIG_V=prometheus_v3 RUN_ID=prometheus_v3 GPU=0,1 bash .../treina_v2_athena.sh
# 1. descreve os 5 datasets do mix e separa a validação (5 episódios por tarefa; suspeitos fora)
# 2. treina do WLA-1.0-Base com tudo descongelado, tarefas balanceadas, peso na ação (treina_wla_v2.py)
# Validação depois:  CUDA_VISIBLE_DEVICES=0 python valida_wla.py --run playground/Checkpoints/prometheus_v2 \
#                        --passos 10000 20000 30000 --split $DESC/split.json
set -e
REPO=$HOME/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/treino
WLA=${UNIFOLM_WLA:-$HOME/DEV/unifolm-wla-lora}
PY=${PY:-$HOME/miniconda3/envs/unifolm-wla/bin/python}
DADOS_RAIZ=${DADOS_RAIZ:-/home/mrwlker/unifolm_data/Prometheus_G1_Dex3_Mix_Dataset}
DESC=${DESC:-/data/mrwlker/wla_v2/descricao}
RUN_ID=${RUN_ID:-prometheus_v2}
bash "$REPO/instala_treino.sh" "$WLA"
if [ ! -f "$DESC/split.json" ]; then
    "$PY" "$REPO/descreve_datasets.py" --raiz "$DADOS_RAIZ" --val 5 --saida "$DESC"
fi
cat "$DESC/DATASETS.md" | grep -v "^  - ep" | head -80
cd "$WLA"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${WANDB_MODE:-online} WANDB_LOG_MODEL=false WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true
N=$(echo "${GPU:-0,1}" | tr ',' '\n' | wc -l)
GPU=${GPU:-0,1} NPROC=$N LOTE=${LOTE:-4} ACUMULA=${ACUMULA:-2} LEITORES=${LEITORES:-32} PASSOS=${PASSOS:-30000} RUN_ID=$RUN_ID \
TREINO_PY=treina_wla_v2.py CONFIG=./unifolm_wla/config/training/${CONFIG_V:-prometheus_v2}.yaml \
DADOS=./unifolm_wla/dataloader/multi_source_dataset/configs/prometheus_v2.yaml \
    bash treina_lora_prometheus_dex3.sh --trainer.v2.split "$DESC/split.json" "$@"
