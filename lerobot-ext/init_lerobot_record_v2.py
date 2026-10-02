#!/usr/bin/env python
"""
Data Collection Entry Point - HACKER EDITION V9 + VOICE CONTROL
Corrigindo validação de Tuplas e adicionando controle Hands-Free.
"""

import os

# Ver a nota longa no init_lerobot_teleoparate_v2.py: o ipopt/BLAS multithread do
# conda-forge faz a IK custar 83 ms em vez de 0,8 ms, e a variável tem que ser
# definida antes do primeiro import que carregue a runtime OpenMP (numpy já basta).
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import logging
import numpy as np
import threading
import time

frame_count = 0

# Buffer global de contrabando
buffer_pressao = {"left": np.zeros(33, dtype=np.float32), "right": np.zeros(33, dtype=np.float32)}

try:
    import robot.unitree_g1
    import teleop.unitree_g1
except ImportError as e:
    print(f"\n[IMPORT ERROR]: {e}")
    sys.exit(1)

# =========================================================================
# 💉 INJEÇÃO 1: Rouba a pressão do Robô
# =========================================================================
from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3

original_get_obs = UnitreeG1Dex3.get_observation

def patched_get_observation(self):
    global frame_count
    obs = original_get_obs(self)
    
    if obs is not None:
        frame_count += 1
        
        if "left_hand_pressure" in obs:
            lp = obs.pop("left_hand_pressure")
            rp = obs.pop("right_hand_pressure")
            buffer_pressao["left"] = lp
            buffer_pressao["right"] = rp
            
            if frame_count % 50 == 0:
                max_l = np.max(lp)
                max_r = np.max(rp)
                status = "🟢 SENSOR ATIVO" if (max_l > 0 or max_r > 0) else "⚪ ZERADO (Aguardando Toque)"
                #print(f"[DEBUG] Frame {frame_count} | {status} | Max L: {max_l:.0f} | Max R: {max_r:.0f}")
        #else:
            #if frame_count % 50 == 0:
                #print(f"[ERRO] Frame {frame_count} | 🔴 DRIVER NÃO ENVIOU DADOS DE PRESSÃO!")
                
    return obs

UnitreeG1Dex3.get_observation = patched_get_observation

# =========================================================================
# 💉 INJEÇÃO 2: Editar a Planta Baixa do Parquet (TUPLAS!)
# =========================================================================
# Na 0.6.1 os dois helpers saíram de `lerobot.datasets.utils` e foram para
# `lerobot.utils.feature_utils` — módulo propositalmente leve, para poder ser
# importado sem arrastar o `datasets` do HuggingFace junto.
#
# Trocar o atributo só no módulo de origem NÃO basta: quem consome faz
# `from lerobot.utils.feature_utils import build_dataset_frame`, ou seja, copia
# a referência para o próprio namespace no momento do import. Quem já foi
# importado (o `robot.unitree_g1` lá em cima arrasta meio lerobot) continuaria
# com a função original. `_patch_lerobot` reescreve a referência em todo módulo
# lerobot já carregado; trocar no módulo de origem cobre os que ainda vão ser
# importados — entre eles o `lerobot.scripts.lerobot_record` lá embaixo.
import lerobot.utils.feature_utils as lr_feature_utils
from lerobot.utils.constants import OBS_STR


def _patch_lerobot(nome: str, func_nova, func_velha) -> None:
    setattr(lr_feature_utils, nome, func_nova)
    for mod_nome, mod in list(sys.modules.items()):
        if mod is None or mod is lr_feature_utils or not mod_nome.startswith("lerobot"):
            continue
        if getattr(mod, nome, None) is func_velha:
            setattr(mod, nome, func_nova)


original_hw_to_dataset_features = lr_feature_utils.hw_to_dataset_features


# *args/**kwargs de propósito: o terceiro parâmetro se chama `use_video` na 0.6.1
# (era `use_videos`) e há chamador que passa por nome. Repassar cru evita casar
# assinatura com uma API que ainda está se mexendo.
def patched_hw_to_dataset_features(*args, **kwargs):
    dataset_features = original_hw_to_dataset_features(*args, **kwargs)

    if "observation.state" in dataset_features:
        print("\n[HACK LEROBOT] 🗜️ Configurando colunas do Parquet para Pressão...")

        old_names = dataset_features["observation.state"].get("names", [])
        new_names = [n for n in old_names if "pressure" not in n]
        dataset_features["observation.state"]["names"] = new_names
        dataset_features["observation.state"]["shape"] = (len(new_names),)

        dataset_features["observation.left_hand_pressure"] = {
            "dtype": "float32", "shape": (33,), "names": [f"left_hand_pressure_{i}" for i in range(33)]
        }
        dataset_features["observation.right_hand_pressure"] = {
            "dtype": "float32", "shape": (33,), "names": [f"right_hand_pressure_{i}" for i in range(33)]
        }

    # A marcação `video.is_depth_map` da head_camera_depth saiu daqui de propósito.
    # Na 0.6.1 essa flag deixou de ser enfeite de metadado: `meta.depth_keys` a lê e
    # manda a câmera inteira para OUTRO pipeline — quadros gravados como TIFF 16 bits
    # e vídeo encodado pelo `DepthEncoderConfig`, que quantiza profundidade métrica de
    # 1 canal. O nosso `realsense_server.py` publica profundidade já normalizada para
    # cinza uint8 de 3 canais (`cv2.merge([d, d, d])`), que é o que o `_cameras_ft`
    # declara. Ligar a flag mandaria dado de 3 canais para o encoder de 1 canal.
    # Quando a profundidade métrica de verdade for publicada (uint16, 1 canal), o
    # caminho certo é declarar (H, W, 1) no `_cameras_ft`: o próprio
    # `hw_to_dataset_features` marca `info["is_depth_map"] = True` sozinho.

    return dataset_features


_patch_lerobot("hw_to_dataset_features", patched_hw_to_dataset_features, original_hw_to_dataset_features)

# =========================================================================
# 💉 INJEÇÃO 3: Contrabando de volta pro Empacotador
# =========================================================================
original_build_dataset_frame = lr_feature_utils.build_dataset_frame


def patched_build_dataset_frame(ds_features, values, prefix, *args, **kwargs):
    # O `prefix` perdeu o default na 0.6.1 e virou "observation" (sem ponto), o
    # mesmo `OBS_STR` que o record passa. A mesma função também empacota a ação —
    # aí não há pressão nenhuma para contrabandear.
    if prefix == OBS_STR:
        lp = buffer_pressao["left"]
        rp = buffer_pressao["right"]

        for i in range(33):
            values[f"left_hand_pressure_{i}"] = float(lp[i])
            values[f"right_hand_pressure_{i}"] = float(rp[i])

    return original_build_dataset_frame(ds_features, values, prefix, *args, **kwargs)


_patch_lerobot("build_dataset_frame", patched_build_dataset_frame, original_build_dataset_frame)

# =========================================================================
# 🎤 INJEÇÃO 4: Comandos de Voz e Teclado (Setas, Pulo Duplo e PAUSE!)
# =========================================================================
# Ver a nota do init_lerobot_teleoparate_v2.py: na 0.6.1 o `init_keyboard_listener`
# migrou de `lerobot.utils.control_utils` para `lerobot.utils.keyboard_input`. Aqui o
# patch tem efeito de verdade — o `lerobot_record` chama a função —, mas só porque o
# `from lerobot.scripts.lerobot_record import main` lá embaixo vem DEPOIS desta linha.
import lerobot.utils.keyboard_input
import threading
import time

original_init_keyboard = lerobot.utils.keyboard_input.init_keyboard_listener

global_events = None

def patched_init_keyboard():
    global global_events
    listener, events = original_init_keyboard()
    global_events = events  
    return listener, events

lerobot.utils.keyboard_input.init_keyboard_listener = patched_init_keyboard

# --- ⌨️ NOVO: Listener de Teclado Paralelo (Setas e Espaço) ---
try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    print("⚠️ Lib 'pynput' não instalada. O atalho de teclado não funcionará. (pip install pynput)")
    pynput_keyboard = None

if pynput_keyboard:
    def on_press(key):
        global robot_paused # Usa a mesma variável da injeção 1
        try:
            # Seta para Baixo (PAUSAR / CONGELAR)
            if key == pynput_keyboard.Key.down:
                if not robot_paused:
                    robot_paused = True
                    print("\n   ⬇️ [TECLADO] Ação: CONGELANDO O ROBÔ NA POSIÇÃO ATUAL! 🧊")

            # Seta para Cima (CONTINUAR / DESTRAVAR)
            elif key == pynput_keyboard.Key.up:
                if robot_paused:
                    robot_paused = False
                    print("\n   ⬆️ [TECLADO] Ação: DESTRAVANDO O ROBÔ! ▶️ (Cuidado com trancos)")

            # Toggle alternativo (Barra de Espaço ou 'P')
            elif key == pynput_keyboard.Key.space or (hasattr(key, 'char') and key.char.lower() == 'p'):
                robot_paused = not robot_paused
                if robot_paused:
                    print("\n   ⌨️ [TECLADO] Ação: CONGELANDO O ROBÔ NA POSIÇÃO ATUAL! 🧊")
                else:
                    print("\n   ⌨️ [TECLADO] Ação: DESTRAVANDO O ROBÔ! ▶️ (Cuidado com trancos)")
                    
        except AttributeError:
            pass

    kb_listener = pynput_keyboard.Listener(on_press=on_press)
    kb_listener.daemon = True
    kb_listener.start()

# --- 🎙️ Função de Voz Original Atualizada ---
def voice_commander_loop():
    global robot_paused # Puxa a variável global de congelamento
    
    try:
        import speech_recognition as sr
    except ImportError:
        print("⚠️ Libs de voz não instaladas. Controle desativado.")
        return

    print("⏳ [VOZ] Aguardando os motores e câmeras iniciarem...")
    
    # Substitua "frame_count" pela variável real de inicialização do seu código, se necessário.
    time.sleep(3) 

    recognizer = sr.Recognizer()
    print("\n🎙️ [VOZ & TECLADO] SISTEMA ATIVO! Comandos:")
    print("   ✅ SALVAR: Voz: 'salvar', 'gravar'")
    print("   ❌ DESCARTAR: Voz: 'errei', 'reboot'")
    print("   🧊 CONGELAR: Voz: 'pausar'    | Teclado: Seta para Baixo (↓)")
    print("   ▶️ DESTRAVAR: Voz: 'continuar' | Teclado: Seta para Cima (↑)")
    print("   🛑 ENCERRAR: Voz: 'sair', 'finalizar'\n")

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        while True:
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=2)
                texto = recognizer.recognize_google(audio, language="pt-BR").lower()

                if global_events is None:
                    continue

                # --- 1. SUCESSO: SALVAR ---
                if any(cmd in texto for cmd in ["gravar", "salvar", "próximo"]):
                    print(f"\n   🗣️ Detectado: '{texto}'")
                    _evento_gravacao("save")

                # --- 2. ERRO: DESCARTAR ---
                elif any(cmd in texto for cmd in ["errei", "reboot", "voltar"]):
                    print(f"\n   🗣️ Detectado: '{texto}'")
                    _evento_gravacao("discard")

                # --- 3. 🧊 CONGELAR O ROBÔ (PAUSE) ---
                elif any(cmd in texto for cmd in ["pausar", "congelar", "travar"]):
                    if not robot_paused:
                        print(f"\n   🗣️ Detectado: '{texto}'")
                        print("   🧊 Ação: CONGELANDO O ROBÔ NA POSIÇÃO ATUAL!")
                        robot_paused = True

                # --- 4. ▶️ DESTRAVAR O ROBÔ (PLAY) ---
                elif any(cmd in texto for cmd in ["continuar", "destravar", "play"]):
                    if robot_paused:
                        print(f"\n   🗣️ Detectado: '{texto}'")
                        print("   ▶️ Ação: DESTRAVANDO O ROBÔ! (Cuidado com trancos)")
                        robot_paused = False
                
                # --- 5. FINALIZAR TUDO ---
                elif any(cmd in texto for cmd in ["finalizar", "sair", "fechar"]):
                    print(f"\n   🗣️ Detectado: '{texto}'")
                    print("   🛑 Ação: Encerrando gravação geral...")
                    global_events["stop_recording"] = True
                    global_events["exit_early"] = True

            except sr.WaitTimeoutError:
                pass 
            except sr.UnknownValueError:
                pass 
            except Exception:
                time.sleep(1)

voice_thread = threading.Thread(target=voice_commander_loop, daemon=True, name="VoiceCommander")
voice_thread.start()
# =========================================================================

# =========================================================================
# 🧊 INJEÇÃO 5: Hack de Congelamento Motor (Pause/Play)
# =========================================================================
robot_paused = False
frozen_action = None

original_send_action = UnitreeG1Dex3.send_action

def patched_send_action(self, action):
    global robot_paused, frozen_action
    
    if robot_paused:
        # 🧊 MODO ESTÁTUA: Ignora o VR e manda o robô segurar a última pose com força
        if frozen_action is not None:
            return original_send_action(self, frozen_action)
        else:
            return original_send_action(self, action)
    else:
        # ▶️ MODO NORMAL: Salva a posição atual e obedece o VR
        frozen_action = {k: v for k, v in action.items()}
        return original_send_action(self, action)

UnitreeG1Dex3.send_action = patched_send_action

# =========================================================================
# 🔁 INJEÇÃO 6: o robô continua sob controle enquanto o episódio é SALVO (01/10)
# =========================================================================
# O `lerobot_record` para o laço de controle no `dataset.save_episode()` (vídeos das 4 câmeras
# encodados ali, vários segundos): o robô ficava sem comando — "congelava"/perdia a conexão. Esta
# thread percebe o laço parado (> 0,15 s sem send_action) e faz ela mesma VR -> robô, a 30 Hz, até o
# laço voltar. Com o robô em PAUSA (seta ↓ / "pausar") ela manda a pose congelada, como o laço faria.
# Se o VR falhar, segura a última ação. A trava garante que ela e o laço nunca comandam ao mesmo tempo.
from teleop.xr_g1_arm import XRG1Arm

_trava_ctrl = threading.RLock()
_ctrl = {"robo": None, "tele": None, "t_laco": 0.0, "ultima": None, "ativo": False}

_orig_send_action_6 = UnitreeG1Dex3.send_action
_orig_get_action_6 = XRG1Arm.get_action


def _send_action_6(self, action):
    with _trava_ctrl:
        _ctrl["robo"] = self
        if not _ctrl["ativo"]:          # chamada do LAÇO do lerobot (não da thread de manutenção)
            _ctrl["t_laco"] = time.time()
        _ctrl["ultima"] = dict(action)
        return _orig_send_action_6(self, action)


def _semeia_pose_atual(tele, robo):
    """Os braços começam ONDE ESTÃO (01/10). O XRG1Arm nasce com todas as juntas em 0 e, travado, manda
    essa pose — no G1, 0 é o cotovelo a 90° com o antebraço à frente: os braços iam para o "L" ao começar
    a gravar. Aqui as juntas guardadas na teleoperação viram as MEDIDAS do robô (ex.: a pose de gravação
    do painel), e o robô só se mexe depois de destravar no VR."""
    low = getattr(robo, "_lowstate", None)
    if low is None:
        return False
    from robot.unitree_g1.g1_utils import G1_29_JointIndex   # o mesmo enum que o XRG1Arm usa nas chaves
    n = 0
    for motor in G1_29_JointIndex:
        chave = f"{motor.name}.q"
        if chave in tele.body_joints:
            tele.body_joints[chave] = float(low.motor_state[motor.value].q)
            n += 1
    print(f"   🦾 teleoperação começa da pose ATUAL do robô ({n} juntas medidas) — não vai para o 'L'", flush=True)
    return True


def _get_action_6(self):
    with _trava_ctrl:
        _ctrl["tele"] = self
        r = _ctrl.get("robo_obs") or _ctrl["robo"]
        if not _ctrl.get("semeado") and r is not None:
            _ctrl["semeado"] = _semeia_pose_atual(self, r)
        return _orig_get_action_6(self)


_orig_get_obs_6 = UnitreeG1Dex3.get_observation


def _get_obs_6(self):
    _ctrl["robo_obs"] = self      # o laço lê a observação ANTES de pedir a ação: dá o robô para a semente
    return _orig_get_obs_6(self)


UnitreeG1Dex3.get_observation = _get_obs_6
UnitreeG1Dex3.send_action = _send_action_6
XRG1Arm.get_action = _get_action_6


def _mantem_controle():
    avisou = False
    while True:
        time.sleep(1 / 30)
        r, t = _ctrl["robo"], _ctrl["tele"]
        if r is not None and not getattr(r, "is_connected", True):   # fim da gravação: robô desconectado
            _ctrl["robo"] = None
            continue
        if r is None or time.time() - _ctrl["t_laco"] < 0.15:
            if avisou:
                print("   🔁 laço de gravação de volta — controle normal", flush=True)
                avisou = False
            continue
        if not avisou:
            print("\n   🔁 salvando o episódio: o controle VR continua ativo (mantendo o robô)", flush=True)
            avisou = True
        with _trava_ctrl:
            if time.time() - _ctrl["t_laco"] < 0.15:
                continue
            _ctrl["ativo"] = True
            try:
                acao = t.get_action() if t is not None else _ctrl["ultima"]
            except Exception:  # noqa: BLE001 — VR caiu: segura a última
                acao = _ctrl["ultima"]
            try:
                if acao is not None:
                    r.send_action(acao)
            except Exception as e:  # noqa: BLE001
                print(f"   ⚠️ manutenção do controle falhou: {e}", flush=True)
            finally:
                _ctrl["ativo"] = False


threading.Thread(target=_mantem_controle, daemon=True, name="MantemControle").start()

# INICIALIZAÇÃO OFICIAL
from lerobot.scripts.lerobot_record import main

# =========================================================================
# 🎬 INJEÇÃO 7: SALVAR / DESCARTAR sabendo em que FASE a gravação está (01/10)
# =========================================================================
# O botão A e a voz "salvar" disparavam exit_early DUAS vezes (agora e 1 s depois: uma encerra o
# episódio, a outra pula o tempo de arrumar a cena). Com o salvar rápido, o segundo disparo caía DENTRO
# do episódio seguinte: ele terminava na hora, vazio, e o save_episode de um episódio vazio derrubava o
# programa (mãos soltas, robô "morrendo"). Agora:
#   A / "salvar"    no EPISÓDIO -> encerra e PULA o tempo de arrumar | no tempo de arrumar -> encerra ele
#   B / "errei"     no EPISÓDIO -> descarta e PULA o tempo de arrumar | no tempo de arrumar -> descarta
#   fora das duas fases (salvando) -> ignorado, com aviso
# Cada fase começa com exit_early limpo (disparo velho não encerra episódio novo), e um episódio sem
# nenhum quadro é descartado em vez de salvo.
import lerobot.scripts.lerobot_record as _lr
from lerobot.datasets.lerobot_dataset import LeRobotDataset

_fase = {"atual": None, "pular_reset": False}
_orig_record_loop = _lr.record_loop

# HUD DA GRAVAÇÃO (02/10): o XRG1Arm lê `__main__.hud_gravacao` e mostra no VR o episódio atual, quantos já
# foram salvos e o aviso grande de SALVANDO / SALVO / DESCARTADO / VAZIO.
hud_gravacao = {"ep": None, "salvos": None, "total": None, "fase": None, "aviso": None, "cor": None, "t_aviso": 0.0}


def _hud_aviso(texto, cor):
    hud_gravacao.update(aviso=texto, cor=cor, t_aviso=time.time())


def _record_loop_fase(*args, **kw):
    fase = "episodio" if kw.get("dataset") is not None else "arrumar"
    ev = kw.get("events")
    if fase == "arrumar" and _fase["pular_reset"]:
        _fase["pular_reset"] = False
        print("   ⏭️  tempo de arrumar pulado (salvar/descartar no episódio)", flush=True)
        return
    if ev is not None:
        ev["exit_early"] = False
    _fase["atual"] = fase
    ds = kw.get("dataset")
    if ds is not None:
        hud_gravacao.update(ep=ds.num_episodes + 1, salvos=ds.num_episodes)
        print(f"\n   🎬 GRAVANDO episódio {ds.num_episodes + 1} ({ds.num_episodes} já salvos)", flush=True)
    hud_gravacao["fase"] = fase
    try:
        return _orig_record_loop(*args, **kw)
    finally:
        _fase["atual"] = None


_lr.record_loop = _record_loop_fase


def _evento_gravacao(tipo):
    ev = global_events
    if ev is None:
        return
    fase = _fase["atual"]
    if fase is None:
        print(f"\n   ⏳ [{tipo}] ignorado: salvando o episódio anterior — espere começar o próximo", flush=True)
        return
    if tipo == "discard":
        ev["rerecord_episode"] = True
        print("\n   ❌ DESCARTANDO este episódio e recomeçando...", flush=True)
        _hud_aviso(f"EP {hud_gravacao.get('ep') or '?'} DESCARTADO", "vermelho")
    else:
        print("\n   ✅ SALVANDO e indo para o próximo..." if fase == "episodio"
              else "\n   ✅ fim do tempo de arrumar — salvando...", flush=True)
    if fase == "episodio":
        _fase["pular_reset"] = True
    ev["exit_early"] = True


def _trigger_7(self, action_type):
    if action_type in ("save", "discard"):
        return _evento_gravacao(action_type)
    return _orig_trigger_7(self, action_type)


_orig_trigger_7 = XRG1Arm._trigger_record_event
XRG1Arm._trigger_record_event = _trigger_7

_orig_save_episode_7 = LeRobotDataset.save_episode


def _save_episode_7(self, *a, **k):
    p = getattr(self, "has_pending_frames", None)
    tem = p() if callable(p) else p
    if tem is False:
        print("   ⚠️ episódio sem nenhum quadro — descartado (não salvo)", flush=True)
        _hud_aviso("EPISODIO VAZIO - NAO SALVO", "vermelho")
        self.clear_episode_buffer()
        return
    n = self.num_episodes + 1
    hud_gravacao["fase"] = "salvando"
    _hud_aviso(f"SALVANDO EP {n}...", "amarelo")
    print(f"   💾 SALVANDO episódio {n}...", flush=True)
    t0 = time.time()
    try:
        r = _orig_save_episode_7(self, *a, **k)
    except Exception:
        _hud_aviso(f"ERRO AO SALVAR EP {n}", "vermelho")
        raise
    hud_gravacao.update(salvos=self.num_episodes, fase=None)
    _hud_aviso(f"EP {n} SALVO  ({self.num_episodes} no total)", "verde")
    print(f"   ✅ episódio {n} SALVO em {time.time() - t0:.1f} s — {self.num_episodes} no dataset", flush=True)
    return r


LeRobotDataset.save_episode = _save_episode_7

class IgnoreFPSWarningFilter(logging.Filter):
    def filter(self, record):
        return "Record loop is running slower" not in record.getMessage()

if __name__ == "__main__":
    cli_args = sys.argv[:]
    
    if "--config_path" not in str(cli_args):
        print("\n[ERRO]: O argumento '--config_path' é obrigatório.")
        sys.exit(1)

    force_sim = "--sim" in cli_args or "--simulation=true" in cli_args
    if "--sim" in sys.argv: sys.argv.remove("--sim")

    if force_sim:
        sys.argv.append("--robot.is_simulation=true")
        sys.argv.append("--teleop.is_simulation=true")
    else:
        sys.argv.append("--robot.is_simulation=false")
        sys.argv.append("--teleop.is_simulation=false")

    logging.getLogger().addFilter(IgnoreFPSWarningFilter())
    logging.getLogger("lerobot").addFilter(IgnoreFPSWarningFilter())

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[SYSTEM]: Gravação finalizada pelo usuário.")
        sys.exit(0)
    except Exception as e:
        import traceback
        print(f"\n[ERRO DE EXECUÇÃO]:")
        traceback.print_exc()
        sys.exit(1)