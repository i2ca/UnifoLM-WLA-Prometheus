#!/bin/bash
# LoRA do ER-1 na Athena, de ponta a ponta (rode dentro de uma screen:  screen -S er1_lora).
#   1. gera as perguntas sim/não dos datasets (CPU, ~10 s)  -> $DADOS/{treino,teste}.jsonl + quadros/
#   2. treina o LoRA (GPU, ~2 h com a GPU dividida)          -> $RUN/adaptador_epN, avaliacao_*.json
#   3. varre um episódio de teste de cada tarefa             -> $RUN/varredura/linha_do_tempo_*.png
# Variáveis: GPU (1), CPUS (64-127), EPOCAS (2), DS (datasets originais), DADOS, RUN
set -e
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-$HOME/miniconda3/envs/unifolm-wla/bin/python}
GPU=${GPU:-1}; CPUS=${CPUS:-64-127}; EPOCAS=${EPOCAS:-2}
DS=${DS:-/data/mrwlker/datasets_prometheus}
DADOS=${DADOS:-/data/mrwlker/er1_lora/dados}
RUN=${RUN:-/data/mrwlker/er1_lora/run_$(date +%Y%m%d)}
DATASETS="$DS/pegar_caneca_mesa_nova $DS/pegar_maca_mesa_nova $DS/copo_branco_2026-10-01 $DS/copo_no_coador_2026-10-01 $DS/maca_x_preto_2026-09-30"
cd "$AQUI"
[ -f "$DADOS/treino.jsonl" ] || taskset -c "$CPUS" "$PY" gera_dataset_er1.py --saida "$DADOS" --dataset $DATASETS
mkdir -p "$RUN"
CUDA_VISIBLE_DEVICES=$GPU taskset -c "$CPUS" "$PY" -u treina_lora_er1.py --dados "$DADOS" --saida "$RUN" --epocas "$EPOCAS" 2>&1 \
    | grep --line-buffered -v "it/s\]" | tee "$RUN/treino.log"
CUDA_VISIBLE_DEVICES=$GPU taskset -c "$CPUS" "$PY" -u varre_episodio_er1.py --adaptador "$RUN/adaptador_ep$EPOCAS" \
    --dados "$DADOS" --saida "$RUN/varredura" --dataset $DATASETS 2>&1 | grep --line-buffered -v "it/s\]" | tee "$RUN/varredura.log"
echo "✅ pronto: $RUN (adaptadores, avaliações, linhas do tempo)"
