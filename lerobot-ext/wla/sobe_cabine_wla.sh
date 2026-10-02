#!/bin/bash
# MODO CABINE (02/10): um comando para testar as tarefas pelo navegador.
#   1. sobe o servidor do WLA Dex3 (:8601) se ele não estiver no ar (RUN / PASSO escolhem o modelo);
#   2. roda o executor no robô real: vai para a POSE DE GRAVAÇÃO e fica SEGURANDO, sem tarefa;
#   3. na cabine (http://<pgx>:8090) você aperta "pegar caneca", depois "caneca → coador" (ou escreve a frase);
#      "⏸ parar" = o robô MANTÉM a posição em que está e espera outra tarefa (emergência: o cogumelo).
# Argumentos extras vão para o roda_wla_real.py (ex.: --pose inicial, --braco-esquerdo ativo).
#   bash sobe_cabine_wla.sh
WLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIFOLM_WLA="${UNIFOLM_WLA:-$HOME/DEV/unifolm-wla}"
PY="${WLA_PY:-$HOME/miniforge3/envs/wla/bin/python}"
export RUN="${RUN:-lora_prometheus_dex3_tres_tarefas_athena}"
PASSO="${PASSO:-30000}"

if (echo > /dev/tcp/127.0.0.1/8601) 2>/dev/null; then
    echo "✅ servidor do WLA já está no ar (:8601) — $(pgrep -af 'servidor_wla_[d]ex3' | grep -o 'lora_run [^ ]*' | head -1)"
else
    bash "$WLA_DIR/sobe_wla_servidor_dex3.sh" "$PASSO"
    echo -n "⏳ carregando o modelo"
    for i in $(seq 1 90); do
        grep -q "servidor WLA Dex3 (fig6d)" ~/wla_servidor_dex3.log 2>/dev/null && break
        grep -q "Traceback" ~/wla_servidor_dex3.log 2>/dev/null && { echo; tail -20 ~/wla_servidor_dex3.log; exit 1; }
        echo -n "."; sleep 3
    done
    echo; grep -E "LoRA carregado|servidor WLA Dex3" ~/wla_servidor_dex3.log | tail -2
fi

cd "$UNIFOLM_WLA" || exit 1
exec "$PY" -u "$WLA_DIR/roda_wla_real.py" --servidor ws://127.0.0.1:8601 --pose gravacao --tarefa "" --segundos 0 \
    --braco-esquerdo ativo --juntas-travadas 20 "$@"
