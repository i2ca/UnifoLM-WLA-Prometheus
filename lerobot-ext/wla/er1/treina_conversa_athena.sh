#!/bin/bash
# LoRA de CONVERSA do ER-1 na Athena (30/10), de ponta a ponta — rode numa screen:  screen -S er1_conversa
#   1. gera (GPUs 0 e 1 em paralelo): perguntas da memória + respostas do ER-1 original com o trecho
#   2. treino (GPU 0): adaptador que responde igual SEM o trecho
#   3. avalia (GPU 0): perguntas guardadas fora do treino
# Variáveis: DB (memoria.db copiada da PGX), RUN, EPOCAS (3)
set -e
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-$HOME/miniconda3/envs/unifolm-wla/bin/python}
DB=${DB:-/data/mrwlker/er1_conversa/memoria.db}
RUN=${RUN:-/data/mrwlker/er1_conversa/run_$(date +%Y%m%d_%H%M)}
EPOCAS=${EPOCAS:-3}
S="$AQUI/treina_conversa_er1.py"
mkdir -p "$RUN"
echo "RUN=$RUN"
CUDA_VISIBLE_DEVICES=0 "$PY" -u "$S" --fase gera --parte 0 --partes 2 --db "$DB" --saida "$RUN" > "$RUN/gera_0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 "$PY" -u "$S" --fase gera --parte 1 --partes 2 --db "$DB" --saida "$RUN" > "$RUN/gera_1.log" 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0 "$PY" -u "$S" --fase treino --db "$DB" --saida "$RUN" --epocas "$EPOCAS" 2>&1 | tee "$RUN/treino.log"
CUDA_VISIBLE_DEVICES=0 "$PY" -u "$S" --fase avalia --db "$DB" --saida "$RUN" --epocas "$EPOCAS" 2>&1 | tee "$RUN/avalia.log"
echo "✅ FIM: $RUN"
