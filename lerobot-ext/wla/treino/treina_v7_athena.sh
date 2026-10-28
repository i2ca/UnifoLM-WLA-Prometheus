#!/bin/bash
# TREINO V7 na Athena (28/10): modelo novo do WLA-1.0-Base, tudo descongelado — rode DENTRO de uma screen:
#   screen -S treino_v7
#   bash ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/treino/treina_v7_athena.sh                 # TESTE de 200 passos
#   CURTO=0 bash ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/treino/treina_v7_athena.sh         # treino longo
# Antes: profundidade_v7.py (Prometheus_G1_Dex3_V7 com head_depth). Split = o do café para os 3 datasets (mesmos
# episódios de validação do v5/v6: dá para comparar).
set -e
REPO=$HOME/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/treino
WLA=${UNIFOLM_WLA:-$HOME/DEV/unifolm-wla-lora}
PY=${PY:-$HOME/miniconda3/envs/unifolm-wla/bin/python}
DESC=/data/mrwlker/wla_v7/descricao
RUN_ID=${RUN_ID:-prometheus_v7}
CURTO=${CURTO:-200}
bash "$REPO/instala_treino.sh" "$WLA"
ln -sfn /data/mrwlker/unifolm_data/Prometheus_G1_Dex3_V7 /home/mrwlker/unifolm_data/Prometheus_G1_Dex3_V7
mkdir -p "$DESC"
"$PY" - <<PYEOF
import json
s = json.load(open("/data/mrwlker/wla_cafe/descricao/split.json"))
v7 = {k: s[k] for k in ("G1_Dex3_Cafe_Caneca_No_Coador", "G1_Dex3_Pegar_Caneca_Mesa_Nova", "G1_Dex3_Pegar_Maca_Mesa_Nova")}
json.dump(v7, open("$DESC/split.json", "w"), indent=1)
print({k: v["val"] for k, v in v7.items()})
PYEOF
cd "$WLA"
CUDA_VISIBLE_DEVICES="" "$PY" "$REPO/wla_v2.py" ./unifolm_wla/config/training/prometheus_v7.yaml   # lotes de validação
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${WANDB_MODE:-online} WANDB_LOG_MODEL=false WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true
GPU=${GPU:-0,1} NPROC=2 LOTE=${LOTE:-4} ACUMULA=${ACUMULA:-2} LEITORES=${LEITORES:-32} PASSOS=${PASSOS:-15000} \
RUN_ID=$RUN_ID TREINO_PY=treina_wla_v7.py CONFIG=./unifolm_wla/config/training/prometheus_v7.yaml \
DADOS=./unifolm_wla/dataloader/multi_source_dataset/configs/prometheus_v7.yaml \
    bash treina_lora_prometheus_dex3.sh --trainer.v2.split "$DESC/split.json" --trainer.teste_curto "$CURTO" "$@"
