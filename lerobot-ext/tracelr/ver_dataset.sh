#!/bin/bash
# Abre um dataset LeRobot do Prometheus no tracelr com o URDF do G1 (trajetória das DUAS mãos).
#   bash ver_dataset.sh ../meu_dataset/copo_branco_2026-10-01 [--annotate] [--camera wrist_right]
# Abre na câmera da CABEÇA; C troca de câmera, T mostra a trajetória (mão direita + esquerda), G grade.
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS="$(readlink -f "$1")"; shift
# o bindgen do ffmpeg precisa dos headers do compilador (sem isso: "'limits.h' file not found")
export BINDGEN_EXTRA_CLANG_ARGS="${BINDGEN_EXTRA_CLANG_ARGS:--I$(ls -d /usr/lib/gcc/x86_64-linux-gnu/*/include 2>/dev/null | sort -V | tail -1)}"
cd "$AQUI" && cargo run --profile opt-dev -- --urdf "$AQUI/../assets/g1/g1_body29_hand14.urdf" "$DS" "$@"
