#!/bin/bash
# Sobe o servidor de PERGUNTAS do ER-1 (er1_pergunta.py, porta 8098) na PGX e deixa carregado (~60 s).
# Usa o adaptador LoRA da nossa cena (--adaptador, padrão em playground/Checkpoints/er1_lora_prometheus/adaptador_ep2)
# nas perguntas sim/não; narração e tradução usam o ER-1 original. Mata o anterior antes. Log: ~/er1_pergunta.log
#   bash sobe_er1.sh                 # padrão
#   bash sobe_er1.sh --adaptador ""  # só o ER-1 original
WLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIFOLM_WLA="${UNIFOLM_WLA:-$HOME/DEV/unifolm-wla}"
PY="${WLA_PY:-$HOME/miniforge3/envs/wla/bin/python}"
for p in $(pgrep -f "python.*er1_pergunt[a].py"); do kill "$p"; done
sleep 2
cd "$UNIFOLM_WLA" || exit 1
setsid nohup "$PY" -u "$WLA_DIR/er1_pergunta.py" --rgb "$@" > ~/er1_pergunta.log 2>&1 < /dev/null &
echo -n "⏳ carregando o ER-1"
for i in $(seq 1 60); do
    grep -q "ER-1 pergunta" ~/er1_pergunta.log 2>/dev/null && break
    grep -q "Traceback" ~/er1_pergunta.log 2>/dev/null && { echo; tail -n 15 ~/er1_pergunta.log; exit 1; }
    echo -n "."; sleep 3
done
echo; grep -E "adaptador|ER-1 pergunta" ~/er1_pergunta.log
