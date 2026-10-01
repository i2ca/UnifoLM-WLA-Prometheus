#!/bin/bash

# Caminho absoluto para a pasta do seu projeto
PROJECT_DIR=~/prometheus-vla/lerobot-ext/Scripts_Prometheus_int

# Modo do WBC: "true" = High Level/Loco (rt/arm_sdk, WBC da Unitree cuida das pernas)
#              "false" = Low Level/Debug (rt/lowcmd, robô precisa estar suspenso/apoiado)
USE_LOCO=true

# 1. Inicializa o Conda usando o ativador exato do seu terminal
source ~/miniconda3/bin/activate

# 2. Ativa o ambiente
conda activate g1

# 3. Função de Segurança (Mata tudo quando você der Ctrl+C)
cleanup() {
    echo -e "\n\n🛑 [Ctrl+C] Pressionado! Encerrando os servidores..."
    # Mata os processos em segundo plano através dos PIDs
    kill $PID_CAM $PID_CAM_RIGHT $PID_DEX
    echo "✅ Servidores desligados. Robô liberado."
    exit 0
}

# Prepara a armadilha para o sinal de interrupção (SIGINT)
trap cleanup SIGINT

echo "🤖 Iniciando infraestrutura do Prometheus VLA..."

# 4. Inicia o Servidor da Câmera (RealSense) em background (&)
python $PROJECT_DIR/full_realsenser_server.py &
PID_CAM=$!
echo "   [OK] Servidor RealSense ZMQ HEAD (PID: $PID_CAM)"

# 4b. Inicia o Servidor da Câmera do Pulso Direito em background (&)
python $PROJECT_DIR/right_arm_realsense_server.py &
PID_CAM_RIGHT=$!
echo "   [OK] Servidor RealSense ZMQ RIGHT_WRIST (PID: $PID_CAM_RIGHT)"

# Dá um respiro de 1 segundo para as câmeras inicializarem sem gargalar a USB/CPU
sleep 1

# 5. Inicia o Servidor da Mão + Corpo (Dex3 Bridge v2) em background (&)
if [ "$USE_LOCO" = "true" ]; then
    python $PROJECT_DIR/dex3_g1_server_v2.py --loco &
    echo "   [!!] Modo HIGH LEVEL / LOCO — WBC da Unitree assume as pernas"
else
    python $PROJECT_DIR/dex3_g1_server_v2.py &
    echo "   [!!] Modo LOW LEVEL / DEBUG — robô precisa estar suspenso/apoiado"
fi
PID_DEX=$!
echo "   [OK] Servidor Dex3 Bridge v2 (PID: $PID_DEX)"

echo "-------------------------------------------------------"
echo "🚀 Sistema 100% online! Aguardando conexão do LeRobot."
echo "💡 Pressione [Ctrl + C] para finalizar tudo com segurança."
echo "-------------------------------------------------------"

# O comando 'wait' segura este terminal aberto monitorando os processos de fundo
wait
