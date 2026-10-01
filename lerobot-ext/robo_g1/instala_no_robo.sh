#!/bin/bash
# Copia os programas que rodam NA PLACA do G1 (Jetson, unitree@192.168.123.164) para ~/Script_Prometheus_int,
# onde o init_prometheus-vla.sh os procura. Uso:  bash instala_no_robo.sh [usuario@ip]
set -e
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBO="${1:-unitree@192.168.123.164}"
rsync -avL --exclude=__pycache__ \
    "$AQUI"/{cameras_wla_server.py,gera_ponte_v3_panico.py,dex3_g1_server_v3_panico.py,painel_diagnostico.py,painel_diagnostico.html,pose_partida_dex1.json,reenumera_zed.sh,pose_bloqueada.wav,pose_liberada.wav,realsense_estereo_server.py,diagnostico_botao.py,teste_botao_panico.py,init_prometheus-vla.sh} \
    "$ROBO":Script_Prometheus_int/
echo "ok. O reenumera_zed.sh precisa de sudo no robô (ver wiki Robo-G1). Reinicie o init para carregar o painel novo."
