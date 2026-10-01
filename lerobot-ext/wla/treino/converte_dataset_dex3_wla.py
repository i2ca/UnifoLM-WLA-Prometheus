#!/usr/bin/env python
"""
Converte um dataset gravado pela teleoperação VR (Prometheus: juntas + Dex3, câmeras cameras_wla_server)
para o FORMATO DE DADOS DO UnifoLM-WLA — sem reduzir a mão a garra (Dex1): a Dex3 vai como `fig6d`,
o espaço de mão de dedos do WLA (o mesmo das mãos de 5 dedos do dataset WBT da Unitree).

Colunas geradas (os NOMES da Unitree, para a normalização pré-calculada dela valer direto):

  action.left_ee_pose_gripper_base / right   (6)  xyz + rpy 'xyz' da mão na pelvis — FK (fk_g1.py) das
  observation.state.*_ee_pose_gripper_base   (6)  juntas do braço (estado = medidas; ação = MEDIDAS também,
                                                  com --acao medida, o padrão — ver abaixo)
  action.left_fig6d / right                  (6)  Dex3 -> 0..1 (0 aberto, 1 no limite de fechar), na ordem
  observation.state.left_fig6d / right       (6)  [polegar1, polegar2, indicador0, indicador1, médio0, médio1]
  action.waist_action_joint                  (3)  cintura yaw (comandado), roll, pitch (medidos)
  observation.state.waist_state_joint        (3)  yaw, roll, pitch medidos
  action.base_command                        (4)  vx, vy, vyaw, altura (base.vx/vy/vyaw/height)
  action.left_leg / right  e  observation.state.left_leg / right (6)  pernas medidas (o WBC é quem manda)
  observation.images.head_stereo_left/right, wrist_left, wrist_right   (vídeos copiados como estão)

--acao medida (padrão, 01/10): a pose da mão e a cintura da AÇÃO saem das juntas MEDIDAS, não das
comandadas. O WLA aprende a ação relativa à pose atual (T_estado(t)^-1 · T_ação(t+k)); na teleoperação o
comando ficava 8-15 cm acima da mão real (punho esquerdo travado, braço cedendo, comando adiantado no
movimento), e o modelo afinado com isso aprendeu "o alvo é ~10 cm acima de onde a mão está" — no robô as
mãos subiam sem parar. Com a ação medida, k=0 dá deslocamento zero e o modelo aprende o movimento REAL.
Os dedos (fig6d) continuam com o COMANDO: para segurar, o dedo é mandado além do ponto de contato.
--acao comando reproduz o dataset antigo.

A rotação do polegar (polegar0, a 7ª junta) não cabe no fig6d: fica fora, e o executor a reconstrói da
pinça (é assim que a teleoperação a comanda: polegar0 = -0,5 x gatilho do indicador).

    python wla/treino/converte_dataset_dex3_wla.py meu_dataset/maca_x_preto_2026-09-30 \\
        ~/unifolm_data/Prometheus_G1_Dex3_Dataset/G1_Dex3_Maca_X_Preto
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # lerobot-ext/wla (fk_g1)
from fk_g1 import FK, NOMES  # noqa: E402

# Dex3, ordem do SDK. Esq: polegar0-2, médio0-1, indicador0-1. Dir: polegar0-2, indicador0-1, médio0-1.
# Limite na direção de FECHAR (g1_utils DEX3_*_LIMITS): esq polegar1/2 +, dedos -; dir espelhado.
FECHA = {"left": np.array([0.0, 0.92, 1.74, -1.57, -1.74, -1.57, -1.74]),
         "right": np.array([0.0, -0.92, -1.74, 1.57, 1.74, 1.57, 1.74])}
# índices (na ordem do SDK) de [polegar1, polegar2, indicador0, indicador1, médio0, médio1]
ORDEM_FIG6D = {"left": [1, 2, 5, 6, 3, 4], "right": [1, 2, 3, 4, 5, 6]}
NOME_MAO = {"left": ["kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2", "kLeftHandMiddle0",
                     "kLeftHandMiddle1", "kLeftHandIndex0", "kLeftHandIndex1"],
            "right": ["kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2", "kRightHandIndex0",
                      "kRightHandIndex1", "kRightHandMiddle0", "kRightHandMiddle1"]}
PERNAS = NOMES[:12]
CINTURA = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
SDK = {n: s for n, s in zip(NOMES, [
    "kLeftHipPitch", "kLeftHipRoll", "kLeftHipYaw", "kLeftKnee", "kLeftAnklePitch", "kLeftAnkleRoll",
    "kRightHipPitch", "kRightHipRoll", "kRightHipYaw", "kRightKnee", "kRightAnklePitch", "kRightAnkleRoll",
    "kWaistYaw", "kWaistRoll", "kWaistPitch",
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow", "kLeftWristRoll",
    "kLeftWristPitch", "kLeftWristYaw", "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw",
    "kRightElbow", "kRightWristRoll", "kRightWristPitch", "kRightWristYaw"])}


def fig6d(q7, lado):
    """(N, 7) juntas Dex3 -> (N, 6) fechamento 0..1 na ordem ORDEM_FIG6D (polegar0 fica de fora)."""
    idx = ORDEM_FIG6D[lado]
    return np.clip(np.asarray(q7)[:, idx] / FECHA[lado][idx], 0.0, 1.0).astype(np.float32)


def nome_mao_nos_dados(nomes_colunas, lado):
    """O LeRobot grava a mão com os nomes de g1_utils (LEFT/RIGHT_HAND_JOINT_NAMES); acha as 7 colunas."""
    achou = [i for i, n in enumerate(nomes_colunas) if n.startswith(f"{lado}_hand_") and n.endswith(".q")]
    if len(achou) != 7:
        raise SystemExit(f"esperava 7 juntas da mão {lado}, achei {len(achou)}: {[nomes_colunas[i] for i in achou]}")
    return achou


def pose6(fk, q29, lado):
    R, p = fk.pose(q29, lado)
    return np.concatenate([p, Rotation.from_matrix(R).as_euler("xyz")]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("origem")
    ap.add_argument("destino")
    ap.add_argument("--altura", type=float, default=0.74, help="se o dataset não tiver base.height")
    ap.add_argument("--acao", choices=["medida", "comando"], default="medida",
                    help="de onde sai a pose da mão/cintura da ação (ver o topo do arquivo)")
    a = ap.parse_args()
    src, dst = Path(a.origem), Path(a.destino)
    info = json.load(open(src / "meta/info.json"))
    nomes_a = info["features"]["action"]["names"]
    nomes_s = info["features"]["observation.state"]["names"]
    # sem diferenciar maiúsculas: o enum do SDK grava "kLeftWristyaw" (y minúsculo)
    ia = {n.lower(): i for i, n in enumerate(nomes_a)}
    is_ = {n.lower(): i for i, n in enumerate(nomes_s)}
    faltam = [f"{SDK[n]}.q" for n in NOMES if f"{SDK[n]}.q".lower() not in is_]
    if faltam:
        raise SystemExit(f"o estado não tem {faltam} — grave com gravar_pernas_cintura: true")
    mao_a = {l: nome_mao_nos_dados(nomes_a, l) for l in ("left", "right")}
    mao_s = {l: nome_mao_nos_dados(nomes_s, l) for l in ("left", "right")}
    fk = FK()

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src / "videos", dst / "videos")
    shutil.copytree(src / "meta", dst / "meta")
    (dst / "data").mkdir(parents=True)

    n_quadros = 0
    for arq in sorted((src / "data").rglob("*.parquet")):
        d = pd.read_parquet(arq)
        A = np.stack(d["action"].values).astype(np.float64)
        S = np.stack(d["observation.state"].values).astype(np.float64)
        n = len(d)
        # q29 MEDIDO (estado) e COMANDADO (ação nos braços + cintura yaw; roll/pitch/pernas = medidos)
        qs = np.stack([S[:, is_[f"{SDK[nm]}.q".lower()]] for nm in NOMES], 1)
        qa = qs.copy()
        if a.acao == "comando":
            for j, nm in enumerate(NOMES):
                k = f"{SDK[nm]}.q".lower()
                if k in ia:
                    qa[:, j] = A[:, ia[k]]
        out = {}
        for pref, q in (("observation.state", qs), ("action", qa)):
            for l in ("left", "right"):
                out[f"{pref}.{l}_ee_pose_gripper_base"] = [pose6(fk, q[t], l) for t in range(n)]
        for l in ("left", "right"):
            out[f"action.{l}_fig6d"] = list(fig6d(A[:, mao_a[l]], l))
            out[f"observation.state.{l}_fig6d"] = list(fig6d(S[:, mao_s[l]], l))
        out["action.waist_action_joint"] = list(qa[:, 12:15].astype(np.float32))
        out["observation.state.waist_state_joint"] = list(qs[:, 12:15].astype(np.float32))
        base = np.zeros((n, 4), np.float32)
        for c, k in enumerate(("base.vx", "base.vy", "base.vyaw", "base.height")):
            base[:, c] = A[:, ia[k]] if k in ia else (a.altura if c == 3 else 0.0)
        out["action.base_command"] = list(base)
        out["action.left_leg"] = out["observation.state.left_leg"] = list(qs[:, 0:6].astype(np.float32))
        out["action.right_leg"] = out["observation.state.right_leg"] = list(qs[:, 6:12].astype(np.float32))
        novo = pd.DataFrame({k: v for k, v in out.items()})
        for c in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            novo[c] = d[c].values
        destino = dst / "data" / arq.relative_to(src / "data")
        destino.parent.mkdir(parents=True, exist_ok=True)
        novo.to_parquet(destino, index=False)
        n_quadros += n

    # meta/info.json: troca action/observation.state pelas colunas novas (vídeos e o resto iguais)
    feats = {k: v for k, v in info["features"].items()
             if k not in ("action", "observation.state", "observation.left_hand_pressure",
                          "observation.right_hand_pressure")}
    dims = {"ee_pose_gripper_base": ["x", "y", "z", "roll", "pitch", "yaw"],
            "fig6d": ["polegar1", "polegar2", "indicador0", "indicador1", "medio0", "medio1"]}
    for pref in ("action", "observation.state"):
        for l in ("left", "right"):
            for k, nomes in dims.items():
                feats[f"{pref}.{l}_{k}"] = {"dtype": "float32", "shape": [len(nomes)], "names": nomes}
            feats[f"{pref}.{l}_leg"] = {"dtype": "float32", "shape": [6], "names": [n.replace("_joint", "") for n in
                                        (PERNAS[:6] if l == "left" else PERNAS[6:])]}
    feats["action.waist_action_joint"] = {"dtype": "float32", "shape": [3], "names": ["yaw", "roll", "pitch"]}
    feats["observation.state.waist_state_joint"] = {"dtype": "float32", "shape": [3], "names": ["yaw", "roll", "pitch"]}
    feats["action.base_command"] = {"dtype": "float32", "shape": [4], "names": ["vx", "vy", "angle_z", "height"]}
    info["features"] = feats
    info["robot_type"] = "unitree_g1_dex3"
    json.dump(info, open(dst / "meta/info.json", "w"), indent=4)
    # stats.json por dataset não é usado (o WLA usa as estatísticas pré-calculadas da Unitree), mas o
    # LeRobot o lê: fica o original, sem as chaves velhas que não existem mais.
    st = json.load(open(src / "meta/stats.json"))
    json.dump({k: v for k, v in st.items() if k in feats}, open(dst / "meta/stats.json", "w"), indent=1)
    print(f"ok: {n_quadros} quadros -> {dst}")


if __name__ == "__main__":
    main()
