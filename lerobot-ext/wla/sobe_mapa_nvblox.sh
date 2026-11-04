#!/bin/bash
# 03/11: o mapa 3D com o NVBLOX (wla/mapa_nvblox.py) no ambiente separado ~/nvblox_teste/env (torch cu130).
# Sobe no screen "mapa" no lugar do mapa_ocupacao.py (mesma porta 8603, as cabines não mudam).
#   bash lerobot-ext/wla/sobe_mapa_nvblox.sh
B=~/nvblox_teste
export LD_LIBRARY_PATH=$B/inst/lib:$(cat $B/cudnn_dir.txt):$LD_LIBRARY_PATH GLOG_minloglevel=2
cd ~/DEV/UnifoLM-WLA-Prometheus
exec $B/env/bin/python -u lerobot-ext/wla/mapa_nvblox.py "$@"
