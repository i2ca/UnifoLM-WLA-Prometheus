#!/bin/bash
# Fine-tuning LoRA do UnifoLM-WLA-1.0-Base (VLM congelado, LoRA no DiT) com o dataset do Prometheus
# (G1 + Dex3 em fig6d, câmeras ZED + punhos). Mesmo procedimento do run_lora_finetune_mmdit_frozen_vlm.sh
# da Unitree, com o ambiente conda da Athena, a GPU escolhida e atenção SDPA (sem flash_attn aqui).
#   GPU=0 bash treina_lora_prometheus_dex3.sh        # log: playground/Checkpoints/<run_id>/treino.log
#   GPU=0,2 NPROC=2 LOTE=4 PASSOS=8000 bash treina_lora_prometheus_dex3.sh   # 2 GPUs, lote efetivo 8
#   ação = mão MEDIDA (01/10): DADOS=./unifolm_wla/dataloader/multi_source_dataset/configs/prometheus_dex3_medido.yaml RUN_ID=lora_prometheus_dex3_maca_medido
#   na PGX (GB10, flash-attn): ATENCAO=flash_attention_2 GPU=0 NPROC=1 ... bash treina_lora_prometheus_dex3.sh --trainer.save_interval 5000
#   (argumentos extras vão para o train_unifolm_wla.py)
# Lote: 8 amostras por passo de uma vez (LOTE=8, ACUMULA=1) — mesmo lote efetivo da receita da Unitree
# (1 x 8 acumulado), mas a GPU trabalha cheia: com 1 por vez ela ficava em ~21% de uso (30/09).
cd "${UNIFOLM_WLA:-$HOME/DEV/unifolm-wla}"   # clone do unifolm-wla com os configs deste repo (instala_treino.sh)
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export NCCL_ASYNC_ERROR_HANDLING=1 NCCL_TIMEOUT=10000 NCCL_SOCKET_TIMEOUT_MS=360000 WANDB_MODE=disabled
base_model_dir=playground/Pretrained_models/UnifoLM-WLA-1.0-Base
config_yaml=./unifolm_wla/config/training/lora_prometheus_dex3.yaml
data_config_path=${DADOS:-./unifolm_wla/dataloader/multi_source_dataset/configs/prometheus_dex3.yaml}
run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-lora_prometheus_dex3_maca_x_preto}
mkdir -p ${run_root_dir}/${run_id}
cp $0 ${run_root_dir}/${run_id}/
"${ACCELERATE:-$HOME/miniconda3/envs/unifolm-wla/bin/accelerate}" launch \
  --config_file unifolm_wla/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NPROC:-1} \
  unifolm_wla/training/train_unifolm_wla.py \
  --config_yaml ${config_yaml} \
  --framework.qwenvl.base_vlm ${base_model_dir}/tokenizer \
  --framework.qwenvl.attn_implementation ${ATENCAO:-sdpa} \
  --trainer.pretrained_checkpoint ${base_model_dir}/checkpoints/model.safetensors \
  --datasets.vla_data.data_config_path ${data_config_path} \
  --datasets.vla_data.per_device_batch_size ${LOTE:-8} \
  --datasets.vla_data.num_workers ${LEITORES:-8} \
  --trainer.gradient_accumulation_steps ${ACUMULA:-1} \
  --trainer.max_train_steps ${PASSOS:-20000} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} "$@" 2>&1 | tee ${run_root_dir}/${run_id}/treino.log
