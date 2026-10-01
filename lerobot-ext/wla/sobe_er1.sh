#!/bin/bash
# Sobe o ER-1 decisor (página :8096). Mata o anterior sem se matar. Argumentos extras vão para o er1_decisor.py.
WLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # esta pasta (lerobot-ext/wla)
UNIFOLM_WLA="${UNIFOLM_WLA:-$HOME/DEV/unifolm-wla}"                  # clone do unitreerobotics/unifolm-wla
PY="${WLA_PY:-$HOME/miniforge3/envs/wla/bin/python}"
for p in $(pgrep -f "wla/er1_[d]ecisor.py"); do kill $p; done
sleep 2
cd "$UNIFOLM_WLA"
setsid nohup "$PY" -u "$WLA_DIR"/er1_decisor.py "$@" > ~/er1_decisor.log 2>&1 < /dev/null &
echo "ER-1 decisor subindo (≈60 s) — http://$(hostname -I | awk '{print $1}'):8096 — log ~/er1_decisor.log"
