#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ─────────────────────────────────────────────────────────────────────────────
# GPU: NVIDIA por PRIME offload — LIGADO POR PADRÃO.
#
# Em notebook híbrido o MuJoCo cai na Intel integrada. Medido no cenário
# completo (teclado + janela do MuJoCo + render offscreen 640×480):
#
#     Intel (Mesa ARL)         57 fps
#     NVIDIA RTX 5070        1141 fps     ← 20×
#
# Na Intel a renderização das câmeras vira o gargalo do loop e a teleoperação
# fica intragável.
#
# Isto chegou a quebrar a janela do MuJoCo, mas a causa era outra: o SDL do
# pygame e o GLFW do MuJoCo em backends diferentes no Wayland. Corrigido em
# `teleop/unitree_g1/keyboard_g1_arm.py::connect`, que alinha os dois.
#
# Precisa vir antes de qualquer contexto OpenGL — o libGL lê na criação, não
# depois. Por isso fica aqui, e não dentro de connect().
#
# Para desligar (máquina sem NVIDIA, ou para comparar):
#     PROMETHEUS_FORCE_NVIDIA=0 python init_lerobot_teleoparate_v2.py ...
# ─────────────────────────────────────────────────────────────────────────────
import os as _os

if _os.environ.get("PROMETHEUS_FORCE_NVIDIA") != "0":
    _os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    _os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

from .config_unitree_g1 import UnitreeG1Config
from .unitree_g1 import UnitreeG1
from .unitree_g1_dex3 import UnitreeG1Dex3, UnitreeG1Dex3Config

from . import unitree_g1
from . import unitree_g1_dex3