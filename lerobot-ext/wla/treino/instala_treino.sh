#!/bin/bash
# Coloca os arquivos de TREINO deste repositório dentro do clone do UnifoLM-WLA da Unitree
# (github.com/unitreerobotics/unifolm-wla, testado no commit 0a1aa87) — por link simbólico, então editar
# aqui vale lá. Uso:  bash instala_treino.sh [caminho do clone]   (padrão ~/DEV/unifolm-wla)
set -e
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WLA="${1:-$HOME/DEV/unifolm-wla}"
test -d "$WLA/unifolm_wla" || { echo "não achei $WLA/unifolm_wla — clone o unifolm-wla primeiro"; exit 1; }
ln -sf "$AQUI/lora_prometheus_dex3.yaml" "$WLA/unifolm_wla/config/training/lora_prometheus_dex3.yaml"
for f in "$AQUI"/prometheus_dex3*.yaml; do f=$(basename "$f")
    ln -sf "$AQUI/$f" "$WLA/unifolm_wla/dataloader/multi_source_dataset/configs/$f"
done
for f in treina_lora_prometheus_dex3.sh avalia_checkpoints_wla.py extrai_delta_lora.py testa_dados_wla.py; do
    ln -sf "$AQUI/$f" "$WLA/$f"
done
echo "ok: configs e scripts de treino ligados em $WLA"
echo "ATENÇÃO: confira data_base/cache_dir em prometheus_dex3*.yaml (caminhos da máquina de treino)"
