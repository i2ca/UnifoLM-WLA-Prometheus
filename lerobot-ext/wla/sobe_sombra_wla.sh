#!/bin/bash
# Sobe a SOMBRA do WLA-1.0 na PGX (o modelo roda e mostra na cabine :8090 o que faria, sem mandar nada ao
# robô). Mata a anterior sem se matar. Argumentos extras vão para o sombra_wla.py.
WLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # esta pasta (lerobot-ext/wla)
UNIFOLM_WLA="${UNIFOLM_WLA:-$HOME/DEV/unifolm-wla}"                  # clone do unitreerobotics/unifolm-wla
PY="${WLA_PY:-$HOME/miniforge3/envs/wla/bin/python}"
for p in $(pgrep -f "lerobot-ext/wla/sombra_[w]la.py"); do kill $p; done
sleep 3
cd "$UNIFOLM_WLA"
setsid nohup "$PY" -u "$WLA_DIR"/sombra_wla.py "$@" > ~/sombra_wla.log 2>&1 < /dev/null &
echo "sombra WLA lançada: $*"
