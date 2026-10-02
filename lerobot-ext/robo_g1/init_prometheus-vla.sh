#!/bin/bash

# Caminho absoluto para a pasta do seu projeto
PROJECT_DIR=~/Script_Prometheus_int

# Modo do WBC: "true" = High Level/Loco (rt/arm_sdk, WBC da Unitree cuida das pernas)
#              "false" = Low Level/Debug (rt/lowcmd, robô precisa estar suspenso/apoiado)
USE_LOCO=true

# Ponte com BOTÃO DE PÂNICO (v3, 28/09/2026): cogumelo NC entre GPIO4 e GPIO6.
#   "true"  = dex3_g1_server_v3_panico.py — braços congelam no aperto, fala
#             "Warning. Pose locked." no alto-falante, LED vermelho; braços lentos e leves.
#             A ponte SOBE BLOQUEADA: o PC tem que pedir o rearme (porta 6006) com o
#             botão solto há >= 1 s.
#   "false" = dex3_g1_server_v2.py, a ponte antiga, SEM pânico.
USE_PANICO=true
# Limites da v3 (só valem com USE_PANICO=true)
VEL_BRACO=0.3     # rad/s máx. do alvo de braço
KP_BRACO=80       # 02/10: 80 = o mesmo da teleoperação/gravação (com 40 o braço cedia 10+ cm e o executor descia o braço).  teto de kp de ombro/cotovelo (a classe do LeRobot manda 80). Era 20: com os braços
                  # esticados à frente o braço caía ~12 cm (teste do WLA, 29/09)
KP_PUNHO=40       # 02/10: 40 = o mesmo da gravação.  teto de kp de punho (a classe manda 40). Era 10: o punho não segurava o pitch com o braço esticado (29/09)
VEL_MAO=5.0       # rad/s máx. dos dedos Dex3. Era 1,0 (padrão da v3): a mão levava ~1,5 s para fechar (30/09)
VOLUME=100        # volume da fala "Warning. Pose locked."
VOLUME_BIP=50     # bipe de emergência enquanto o cogumelo está apertado (0-100)

# Painel de diagnóstico no navegador (juntas, temperaturas, mãos, bateria, câmeras, trava):
#   http://192.168.123.164:8095/   — só LÊ o DDS; rearme/pânico só pelos botões da página.
USE_DIAGNOSTICO=true

# Câmeras: "true" = cameras_wla_server.py — formato do dataset do WLA, 640x480 (4:3), 30 fps, numa
#   mensagem só na 5555: head_stereo_left/right (ZED, cru), wrist_left (RealSense 138422074380),
#   wrist_right (RealSense 141722078588), head_camera (D435i da cabeça). Sem profundidade.
#   (Antes: realsense_estereo_server.py — cor + IR da D435i.) O servidor antigo do pulso (5556)
#   NÃO sobe neste modo: usaria a mesma câmera do punho direito.
#   cor em head_camera + par estéreo infravermelho em head_stereo_left/right, SEM profundidade.
#   "false" = full_realsenser_server.py (848x480 + profundidade, para gravar dataset).
CAMERA_ESTEREO=true
ZED_INVERTIDA=true      # ZED montada de cabeça para baixo (29/09): gira 180° e troca os olhos
CABECA_REALSENSE=false  # a ZED tapou a D435i da cabeça: não abre a head_camera
PUNHO_DIR_INVERTIDO=true  # câmera do punho direito de cabeça para baixo no suporte novo (30/09)
ZED_ENQUADRAMENTO=largo # largo = olho inteiro da ZED (~90°) com faixas pretas; 4x3 = recorte (~73°).
                        # A câmera do dataset do WLA tem ~100° (medido 29/09): 4x3 era 1,5x de zoom.

# 1. Inicializa o Conda usando o ativador exato do seu terminal
source ~/miniconda3/bin/activate

# 2. Ativa o ambiente
conda activate g1

# 3. Função de Segurança (Mata tudo quando você der Ctrl+C)
cleanup() {
    echo -e "\n\n🛑 [Ctrl+C] Pressionado! Encerrando os servidores..."
    # Mata os processos em segundo plano através dos PIDs
    kill $PID_CAM $PID_CAM_RIGHT $PID_DEX $PID_DIAG 2>/dev/null
    echo "✅ Servidores desligados. Robô liberado."
    exit 0
}

# Prepara a armadilha para o sinal de interrupção (SIGINT)
trap cleanup SIGINT

echo "🤖 Iniciando infraestrutura do Prometheus VLA..."

# 4. Inicia o Servidor da Câmera (RealSense) em background (&)
if [ "$CAMERA_ESTEREO" = "true" ]; then
    # A ZED às vezes não enumera ao ligar o robô: reinicia o hub USB das câmeras se ela faltar.
    # Precisa da regra do sudoers (ver reenumera_zed.sh); sem ela, só avisa.
    if [ -x /usr/local/sbin/reenumera_zed.sh ]; then
        sudo -n /usr/local/sbin/reenumera_zed.sh || echo "   [!!] ZED fora do USB: desconecte e conecte o cabo dela"
    fi
    CAM_EXTRA=""
    [ "$ZED_INVERTIDA" = "true" ] && CAM_EXTRA="$CAM_EXTRA --zed-invertida"
    [ "$PUNHO_DIR_INVERTIDO" = "true" ] && CAM_EXTRA="$CAM_EXTRA --punho-dir-invertido"
    CAM_EXTRA="$CAM_EXTRA --zed-enquadramento ${ZED_ENQUADRAMENTO:-4x3}"
    [ "$CABECA_REALSENSE" != "true" ] && CAM_EXTRA="$CAM_EXTRA --sem-cabeca-realsense"
    python $PROJECT_DIR/cameras_wla_server.py $CAM_EXTRA &
    PID_CAM=$!
    echo "   [OK] Câmeras do WLA 640x480: ZED estéreo + punhos, sem profundidade [$CAM_EXTRA ] (PID: $PID_CAM)"
else
    python $PROJECT_DIR/full_realsenser_server.py &
    PID_CAM=$!
    echo "   [OK] Servidor RealSense ZMQ HEAD (PID: $PID_CAM)"
fi

# 4b. Inicia o Servidor da Câmera do Pulso Direito em background (&)
if [ "$CAMERA_ESTEREO" != "true" ]; then
    python $PROJECT_DIR/right_arm_realsense_server.py &
    PID_CAM_RIGHT=$!
    echo "   [OK] Servidor RealSense ZMQ RIGHT_WRIST (PID: $PID_CAM_RIGHT)"
fi

# Dá um respiro de 1 segundo para as câmeras inicializarem sem gargalar a USB/CPU
sleep 1

# 5. Inicia o Servidor da Mão + Corpo (Dex3 Bridge) em background (&)
if [ "$USE_PANICO" = "true" ]; then
    PONTE=$PROJECT_DIR/dex3_g1_server_v3_panico.py
    EXTRA="--vel-braco $VEL_BRACO --kp-braco $KP_BRACO --kp-punho $KP_PUNHO --vel-mao $VEL_MAO --volume $VOLUME --volume-bip $VOLUME_BIP"
    NOME="v3 (PÂNICO GPIO4/GPIO6)"
else
    PONTE=$PROJECT_DIR/dex3_g1_server_v2.py
    EXTRA=""
    NOME="v2 (SEM pânico)"
fi
if [ "$USE_LOCO" = "true" ]; then
    python $PONTE --loco $EXTRA &
    echo "   [!!] Modo HIGH LEVEL / LOCO — WBC da Unitree assume as pernas"
else
    # Sem flag o servidor sobe em LOCO; o debug agora precisa ser pedido.
    python $PONTE --debug $EXTRA &
    echo "   [!!] Modo LOW LEVEL / DEBUG — robô precisa estar suspenso/apoiado"
fi
PID_DEX=$!
echo "   [OK] Servidor Dex3 Bridge $NOME (PID: $PID_DEX)"
if [ "$USE_DIAGNOSTICO" = "true" ]; then
    python -u $PROJECT_DIR/painel_diagnostico.py &
    PID_DIAG=$!
    echo "   [OK] Painel de diagnóstico http://192.168.123.164:8095/ (PID: $PID_DIAG)"
fi
if [ "$USE_PANICO" = "true" ]; then
    echo "   [!!] Ponte sobe BLOQUEADA: LED AMARELO. Solte o cogumelo e rearme no painel :8095 (ou pelo PC, porta 6006)."
fi

echo "-------------------------------------------------------"
echo "🚀 Sistema 100% online! Aguardando conexão do LeRobot."
echo "💡 Pressione [Ctrl + C] para finalizar tudo com segurança."
echo "-------------------------------------------------------"

# O comando 'wait' segura este terminal aberto monitorando os processos de fundo
wait
