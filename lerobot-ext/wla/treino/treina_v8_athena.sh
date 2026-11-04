#!/bin/bash
# TREINO V8 na Athena (31/10): LoRA em cima do v7 (passo 20000) com os nossos 3 datasets + 7 de terceiros — rode DENTRO
# de uma screen:
#   screen -S treino_v8
#   bash ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/treino/treina_v8_athena.sh              # TESTE de 200 passos
#   CURTO=0 bash ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/treino/treina_v8_athena.sh      # treino longo
# Antes: converte_terceiros_wla.py (Terceiros_G1_Dex3/). Validação = os MESMOS lotes do v7 (val_v7.pt).
set -e
REPO=$HOME/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/treino
WLA=${UNIFOLM_WLA:-$HOME/DEV/unifolm-wla-lora}
PY=${PY:-$HOME/miniconda3/envs/unifolm-wla/bin/python}
DESC=/data/mrwlker/wla_v7/descricao      # o split do v7 (os 5 eps de validação de cada um dos nossos)
D=/data/mrwlker/unifolm_data
T=$D/Terceiros_G1_Dex3
RUN_ID=${RUN_ID:-prometheus_v8}
CURTO=${CURTO:-200}
bash "$REPO/instala_treino.sh" "$WLA"
# uma pasta de links por câmera de cabeça (cada entrada da config lê UMA chave de vídeo)
mkdir -p $D/T8_cam_left_high $D/T8_cam_rgb_high $D/T8_head_camera $D/T8_cam_head $D/T8_ego_view
ln -sfn $T/bae-air-lab__G1_Dex3_PickVeggis_605ep $D/T8_cam_left_high/
ln -sfn $T/memory-vla__unitree_g1_dex3_press_knob $D/T8_cam_left_high/
ln -sfn $T/Breno-de-Angelo__unitree-g1-dex3-1-pick-kettle-white-table $D/T8_cam_rgb_high/
ln -sfn $T/Breno-de-Angelo__unitree-g1-dex3-1-pick-kettle-v3 $D/T8_cam_rgb_high/
ln -sfn $T/lewislf__G1_Dex3_pick_white_cup_right8_1squeeze $D/T8_head_camera/
ln -sfn $T/alexandrosnt__g1_ee_dex3_green_cube_blue_box $D/T8_cam_head/
ln -sfn $T/rahul-ai-01__unitree_g1_upper_body_pick_and_place_data $D/T8_ego_view/
ls $D/T8_*
cd "$WLA"
test -f /data/mrwlker/wla_v7/val_v7.pt || { echo "falta o val_v7.pt"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${WANDB_MODE:-online} WANDB_LOG_MODEL=false WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true
GPU=${GPU:-0,1} NPROC=2 LOTE=${LOTE:-4} ACUMULA=${ACUMULA:-2} LEITORES=${LEITORES:-32} PASSOS=${PASSOS:-12000} \
RUN_ID=$RUN_ID TREINO_PY=treina_wla_v8.py CONFIG=./unifolm_wla/config/training/prometheus_v8.yaml \
DADOS=./unifolm_wla/dataloader/multi_source_dataset/configs/prometheus_v8.yaml \
    bash treina_lora_prometheus_dex3.sh --trainer.v2.split "$DESC/split.json" --trainer.teste_curto "$CURTO" "$@"
