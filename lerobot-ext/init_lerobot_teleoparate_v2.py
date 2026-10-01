#!/usr/bin/env python
"""
Teleoperation Entry Point V2 for Unitree G1 with Dex3 Hands.
Inclui SISTEMA HANDS-FREE por Voz para Congelar/Destravar o robô.
"""

import os

# ── OpenMP com uma thread: precisa vir ANTES de numpy/casadi/pinocchio ────────
# O ipopt e a BLAS deste ambiente vieram do conda-forge e são multithread. Para o
# NLP da IK, que tem 14 variáveis e converge em 2 iterações, o custo de abrir e
# sincronizar threads domina o cálculo inteiro: `solve_ik` media 83 ms com uma
# variância absurda (6 ms a 170 ms, ao sabor da carga da máquina) contra 0,8 ms
# estáveis com uma thread só. Medido: o laço de teleoperação sai de 6-8 Hz para
# ~24 Hz só com esta linha.
#
# As estatísticas do ipopt mostram o mesmo `iter_count` nos dois casos e
# `t_wall_nlp_*` perto de zero — o tempo não está no solver, está na orquestração
# de threads em volta dele. Por isso `OPENBLAS_NUM_THREADS`/`BLIS_NUM_THREADS`
# não mudam nada aqui, e só `OMP_NUM_THREADS` resolve.
#
# A runtime OpenMP lê isto ao ser carregada, junto com o primeiro import pesado;
# depois disso, mudar a variável não tem efeito. Daí estar na primeira linha.
# Respeita quem já tiver definido algo — se você precisa de CPU multithread para
# alguma etapa, exporte `OMP_NUM_THREADS` antes de chamar o script.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time
import threading

# 1. Register custom G1 modules into the global registry
try:
    import robot.unitree_g1
    import teleop.unitree_g1
    import teleop
except ImportError as e:
    print(f"\n[IMPORT ERROR]: Failed to load custom G1 modules: {e}")
    sys.exit(1)

# =========================================================================
# 🧊 INJEÇÃO 1: Hack de Congelamento Motor (Pause/Play)
# =========================================================================
from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3

robot_paused = False
frozen_action = None

original_send_action = UnitreeG1Dex3.send_action

def patched_send_action(self, action):
    global robot_paused, frozen_action
    
    if robot_paused:
        # 🧊 MODO ESTÁTUA: Mantém os motores travados na última posição
        if frozen_action is not None:
            return original_send_action(self, frozen_action)
        else:
            return original_send_action(self, action)
    else:
        # ▶️ MODO NORMAL: Atualiza a posição alvo
        frozen_action = {k: v for k, v in action.items()}
        return original_send_action(self, action)

UnitreeG1Dex3.send_action = patched_send_action

# =========================================================================
# 🎤 INJEÇÃO 2: Captura de Eventos de Teclado (Para a voz poder encerrar)
# =========================================================================
# Na 0.6.1 o `lerobot.utils.control_utils` deixou de existir: o módulo virou
# `lerobot.common.control_utils`, e o `init_keyboard_listener` NÃO foi junto — ele mora
# agora em `lerobot.utils.keyboard_input`. O patch precisa acontecer antes de qualquer
# `from lerobot.scripts... import`, porque os consumidores ligam o nome no topo do
# próprio módulo (`from lerobot.utils.keyboard_input import init_keyboard_listener`) e
# depois disso já não olham mais para cá.
import lerobot.utils.keyboard_input
original_init_keyboard = lerobot.utils.keyboard_input.init_keyboard_listener

global_events = None

def patched_init_keyboard():
    global global_events
    listener, events = original_init_keyboard()
    global_events = events
    return listener, events

lerobot.utils.keyboard_input.init_keyboard_listener = patched_init_keyboard

# =========================================================================
# 🎙️ INJEÇÃO 3: Controle de Voz Hands-Free
# =========================================================================
def voice_commander_loop():
    global robot_paused
    try:
        import speech_recognition as sr
    except ImportError:
        print("⚠️ Libs de voz não instaladas. Controle desativado.")
        return

    print("⏳ [VOZ] Aguardando estabilização do VR...")
    time.sleep(3)

    recognizer = sr.Recognizer()
    print("\n" + "="*50)
    print("🎙️ TELEOPERAÇÃO POR VOZ ATIVA!")
    print("   🧊 CONGELAR : 'pausar'")
    print("   ▶️ DESTRAVAR: 'continuar'")
    print("   🛑 ENCERRAR : 'sair'")
    print("="*50 + "\n")

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        while True:
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=2)
                texto = recognizer.recognize_google(audio, language="pt-BR").lower()

                # --- 🧊 CONGELAR O ROBÔ (PAUSE) ---
                if any(cmd in texto for cmd in ["pausar"]):
                    print(f"\n   🗣️ Detectado: '{texto}'")
                    print("   🧊 Ação: CONGELANDO O ROBÔ NA POSIÇÃO ATUAL!")
                    robot_paused = True

                # --- ▶️ DESTRAVAR O ROBÔ (PLAY) ---
                elif any(cmd in texto for cmd in ["continuar"]):
                    print(f"\n   🗣️ Detectado: '{texto}'")
                    print("   ▶️ Ação: DESTRAVANDO O ROBÔ! (Cuidado com trancos)")
                    robot_paused = False
                
                # --- 🛑 FINALIZAR TUDO ---
                elif any(cmd in texto for cmd in ["sair"]):
                    print(f"\n   🗣️ Detectado: '{texto}'")
                    print("   🛑 Ação: Encerrando teleoperação de forma segura...")
                    if global_events is not None:
                        global_events["exit_early"] = True

            except sr.WaitTimeoutError:
                pass 
            except sr.UnknownValueError:
                pass 
            except Exception:
                time.sleep(1)

voice_thread = threading.Thread(target=voice_commander_loop, daemon=True, name="VoiceTeleop")
voice_thread.start()

# =========================================================================
# INICIALIZAÇÃO OFICIAL DO LEROBOT
# =========================================================================
from lerobot.scripts.lerobot_teleoperate import main

def display_help():
    """Prints a professional usage guide and CLI documentation."""
    print("\n" + "="*70)
    print("LEROBOT CUSTOM TELEOPERATION INTERFACE (HANDS-FREE)")
    print("="*70)
    print("\nUSAGE:")
    print("  python init_lerobot_teleoparate_v2.py --config_path=<PATH_TO_YAML> [OPTIONS]")
    print("\nARGUMENTS:")
    print("  --config_path     Path to the YAML configuration file.")
    print("  --sim             Short flag to force simulation mode.")
    print("  --simulation=true Force simulation mode.")
    print("  -h, --help        Show this help message and exit.")
    
    # Listagem de arquivos yaml
    print("\nAVAILABLE CONFIG FILES:")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base_dir, "config")
    if os.path.exists(config_dir) and os.path.isdir(config_dir):
        yaml_files = [f for f in os.listdir(config_dir) if f.endswith(('.yaml', '.yml'))]
        if yaml_files:
            for yml in sorted(yaml_files):
                print(f"  - config/{yml}")
        else:
            print("  (No .yaml files found in 'config/' folder)")
    else:
        print("  (Folder 'config/' not found)")
    print("="*70 + "\n")

def preflight_porta_camera(porta: int = 5555) -> None:
    """
    Aborta cedo se a porta das câmeras já estiver ocupada.

    Por que existe: quando um run anterior morre (erro, Ctrl+C mal dado), o
    SUBPROCESSO de publicação de imagem sobrevive e continua segurando a 5555.
    O run seguinte falha assim:

        zmq.error.ZMQError: Address already in use (addr='tcp://*:5555')
        ...
        RuntimeError: Failed to connect to ZMQCamera(head_camera@127.0.0.1:5555):
                      async_read timeout after 6000ms

    A mensagem real (o ZMQError) sai no meio do log de um subprocesso e passa
    despercebida; o que se vê no fim é o timeout da câmera, que parece problema
    de rede ou de simulador. Já custou várias horas de diagnóstico — daí a
    checagem explícita, 6 segundos antes.
    """
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("", porta))
    except OSError:
        print(
            f"\n[ERRO] A porta {porta} (câmeras) já está em uso.\n"
            "\n"
            "  Quase sempre é um processo zumbi de um run anterior que morreu:\n"
            "  o subprocesso de publicação de imagem sobrevive ao pai.\n"
            "\n"
            "  Quem está segurando:\n"
            f"      ss -tlnp | grep {porta}\n"
            "\n"
            "  Para limpar:\n"
            "      pkill -9 -f init_lerobot_teleoparate_v2\n"
            "      pkill -9 -f init_lerobot_record\n"
            "      pkill -9 -f run_sim\n"
        )
        sys.exit(1)
    finally:
        s.close()


if __name__ == "__main__":
    cli_args = sys.argv[:]

    if any(flag in cli_args for flag in ["-h", "--help", "-help"]):
        display_help()
        sys.exit(0)

    # Só faz sentido em simulação local — no robô real quem publica é a máquina
    # do robô, e a porta local estar livre não diz nada.
    if any(a in cli_args for a in ["--sim", "--simulation=true"]):
        preflight_porta_camera()

    if not any("--config_path" in arg for arg in cli_args):
        print("\n[CRITICAL ERROR]: Mandatory '--config_path' argument is missing.")
        sys.exit(1)

    # Verifica simulação
    force_sim = False
    for arg in cli_args:
        if arg in ["--sim", "--simulation=true"]:
            force_sim = True
            if arg in sys.argv:
                sys.argv.remove(arg)

    if force_sim:
        sys.argv.append("--robot.is_simulation=true")
        sys.argv.append("--teleop.is_simulation=true")
        print("[INFO]: Overriding YAML config: robot.is_simulation set to TRUE")
    else:
        sys.argv.append("--robot.is_simulation=false")
        sys.argv.append("--teleop.is_simulation=false")
        print("[INFO]: Using Real Robot mode (robot.is_simulation=false)")

    # Launch teleoperation loop
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[SYSTEM]: Teleoperation terminated by user.")
        sys.exit(0)