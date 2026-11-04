#!/usr/bin/env python
"""Datasets G1+Dex3 de TERCEIROS (HuggingFace) -> formato do WLA (31/10), para o LoRA v8.

Mesmas colunas do converte_dataset_dex3_wla.py (pose da mão por FK na pelvis, Dex3 -> fig6d, cintura, base,
pernas), com o que cada dataset tem de diferente:
  - nomes das juntas em 3 grafias (kLeftShoulderPitch / kLeftShoulderPitch.q / left_hand_thumb_0_joint.q) ou sem
    nomes (polvos: ordem padrão da teleoperação da Unitree, 14 braço + 14 mão);
  - G1 de 23 DoF (PickVeggis): punho só com roll -> pitch/yaw = 0 na FK;
  - só o braço direito (lewislf right8): o esquerdo fica na POSE DE DESCANSO dos nossos datasets (é o que
    queremos que ele faça);
  - pose da mão pronta (alexandrosnt, xyz + rpy, garra 0..1): a garra vira os 6 dedos;
  - pernas e cintura: não gravadas (o robô em pé, cintura travada) -> 0; base: parado, altura 0,74.
Ação da mão = pose MEDIDA (estado), como nos nossos; dedos = comando quando existe, senão o medido.
Só a câmera da CABEÇA fica no info.json (punhos e profundidade de outra câmera ficam de fora).
Os v2.x ganham episodes_stats.jsonl e passam pelo conversor oficial do LeRobot para v3.0 (o LeRobot 0.5 só lê v3).

    python converte_terceiros_wla.py --origem /data/mrwlker/datasets_terceiros --destino /data/mrwlker/unifolm_data/Terceiros_G1_Dex3
"""
import argparse
import json
import shutil
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from converte_dataset_dex3_wla import NOME_MAO, fig6d, pose6  # noqa: E402
from fk_g1 import FK, NOMES  # noqa: E402

BRACO = {"left": ["kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow", "kLeftWristRoll",
                  "kLeftWristPitch", "kLeftWristYaw"],
         "right": ["kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow", "kRightWristRoll",
                   "kRightWristPitch", "kRightWristYaw"]}
PADRAO28 = BRACO["left"] + BRACO["right"] + NOME_MAO["left"] + NOME_MAO["right"]   # ordem da teleoperação Unitree
# POLVOS (sem nomes): pelas faixas das colunas (31/10) é [braço esq, mão esq, braço dir, mão dir], e a mão com os
# dedos ANTES do polegar (dir: 4 colunas 0..+1,2 fechando; polegar1/2 negativos fechando, como na Dex3 direita)
POLVOS28 = (BRACO["left"] + [f"kLeftHand{d}" for d in ("Index0", "Index1", "Middle0", "Middle1", "Thumb0", "Thumb1",
                                                         "Thumb2")]
            + BRACO["right"] + [f"kRightHand{d}" for d in ("Index0", "Index1", "Middle0", "Middle1", "Thumb0", "Thumb1",
                                                           "Thumb2")])
# pose de DESCANSO do braço esquerdo nos nossos datasets (mediana do estado, pelvis): xyz + rpy, e os dedos
DESCANSO_ESQ = np.array([0.29, 0.289, 0.213, -0.135, -0.235, 0.567], np.float32)
DEDOS_ESQ = np.array([0.038, 0.025, 0.042, 0.022, 0.036, 0.019], np.float32)

DATASETS = {
    "bae-air-lab__G1_Dex3_PickVeggis_605ep": {"cabeca": "cam_left_high"},
    "rahul-ai-01__unitree_g1_upper_body_pick_and_place_data": {"cabeca": "ego_view", "sem_nomes": True},   # 20 fps
    "lewislf__G1_Dex3_pick_white_cup_right8_1squeeze": {"cabeca": "head_camera", "so_direita": True},
    # lewislf v2 e v3_grasp: os MESMOS 97 episódios, todos já dentro do right8 (238): ficam de fora
    "Breno-de-Angelo__unitree-g1-dex3-1-pick-kettle-white-table": {"cabeca": "cam_rgb_high"},
    "Breno-de-Angelo__unitree-g1-dex3-1-pick-kettle-v3": {"cabeca": "cam_rgb_high"},
    "memory-vla__unitree_g1_dex3_press_knob": {"cabeca": "cam_left_high"},
    "alexandrosnt__g1_ee_dex3_green_cube_blue_box": {"cabeca": "cam_head", "ee": True},
}


def canon(n):
    """Nome de junta em qualquer grafia -> a do SDK (kLeftHandThumb0, kRightElbow...), minúsculo."""
    n = n.split(".")[0]
    if n.endswith("_joint") and "_hand_" in n:   # left_hand_thumb_0_joint -> kLeftHandThumb0
        lado, _, dedo, k, _ = n.split("_")
        n = f"k{lado.capitalize()}Hand{dedo.capitalize()}{k}"
    return n.lower()


def nomes_de(info, chave):
    n = info["features"].get(chave, {}).get("names")
    if isinstance(n, dict):
        n = list(n.values())[0]
    if n and isinstance(n[0], list):
        n = n[0]
    return n


def colunas(nomes, quero):
    """índice de cada nome de `quero` em `nomes` (None se não tem)."""
    m = {canon(x): i for i, x in enumerate(nomes or [])}
    return [m.get(q.lower()) for q in quero]


def converte_arquivo(args):
    arq, dst, cfg, nomes_s, nomes_a = args
    d = pd.read_parquet(arq)
    S = np.stack(d["observation.state"].values).astype(np.float64)
    A = np.stack(d["action"].values).astype(np.float64)
    n = len(d)
    out = {}
    if cfg.get("ee"):   # alexandrosnt: [esq xyz rpy, dir xyz rpy, garra esq, garra dir]; garra da AÇÃO = comando 0..1
        for pref in ("observation.state", "action"):
            out[f"{pref}.left_ee_pose_gripper_base"] = list(S[:, 0:6].astype(np.float32))
            out[f"{pref}.right_ee_pose_gripper_base"] = list(S[:, 6:12].astype(np.float32))
        for k, l in ((12, "left"), (13, "right")):
            g = np.clip(A[:, k], 0, 1).astype(np.float32)[:, None].repeat(6, 1)
            out[f"action.{l}_fig6d"] = out[f"observation.state.{l}_fig6d"] = list(g)
    else:
        fk = FK()
        q = np.zeros((n, 29))   # pernas e cintura 0 (em pé, cintura travada)
        for l in ("left", "right"):
            ic = colunas(nomes_s, BRACO[l])
            for j, (nome, i) in enumerate(zip(BRACO[l], ic)):
                if i is not None:
                    q[:, 15 + (0 if l == "left" else 7) + j] = S[:, i]
        tem_esq = not cfg.get("so_direita")
        for l in ("left", "right"):
            if l == "left" and not tem_esq:
                p = np.repeat(DESCANSO_ESQ[None], n, 0)
                f = np.repeat(DEDOS_ESQ[None], n, 0)
                out["observation.state.left_ee_pose_gripper_base"] = out["action.left_ee_pose_gripper_base"] = list(p)
                out["observation.state.left_fig6d"] = out["action.left_fig6d"] = list(f)
                continue
            p = np.stack([pose6(fk, q[t], l) for t in range(n)])
            out[f"observation.state.{l}_ee_pose_gripper_base"] = out[f"action.{l}_ee_pose_gripper_base"] = list(p)
            ms = colunas(nomes_s, NOME_MAO[l])
            ma = colunas(nomes_a, NOME_MAO[l])
            out[f"observation.state.{l}_fig6d"] = list(fig6d(S[:, ms], l))
            out[f"action.{l}_fig6d"] = list(fig6d(A[:, ma], l) if None not in ma else fig6d(S[:, ms], l))
    z3, z6 = np.zeros((n, 3), np.float32), np.zeros((n, 6), np.float32)
    out["action.waist_action_joint"] = out["observation.state.waist_state_joint"] = list(z3)
    base = np.zeros((n, 4), np.float32)
    base[:, 3] = 0.74
    out["action.base_command"] = list(base)
    for l in ("left", "right"):
        out[f"action.{l}_leg"] = out[f"observation.state.{l}_leg"] = list(z6)
    novo = pd.DataFrame(out)
    for c in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        novo[c] = d[c].values
    novo.to_parquet(dst, index=False)
    return n


def episodes_stats(dst, info):
    """v2.x -> o v2.1 que o conversor oficial pede: estatística por episódio (as da Unitree é que valem no treino;
    estas só precisam existir). Vídeo: estatística neutra."""
    linhas = []
    vids = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    for arq in sorted((dst / "data").rglob("*.parquet")):
        d = pd.read_parquet(arq)
        st = {}
        for c in d.columns:
            v = np.stack(d[c].values).astype(np.float64)
            v = v.reshape(len(v), -1)
            st[c] = {"min": v.min(0).tolist(), "max": v.max(0).tolist(), "mean": v.mean(0).tolist(),
                     "std": v.std(0).tolist(), "count": [len(v)]}
        for k in vids:
            z = [[[0.5]], [[0.5]], [[0.5]]]
            st[k] = {"min": [[[0.0]]] * 3, "max": [[[1.0]]] * 3, "mean": z, "std": [[[0.25]]] * 3, "count": [len(d)]}
        linhas.append({"episode_index": int(d["episode_index"].iloc[0]), "stats": st})
    with open(dst / "meta/episodes_stats.jsonl", "w") as f:
        for l in sorted(linhas, key=lambda x: x["episode_index"]):
            f.write(json.dumps(l) + "\n")


def feats_novas(info, cabeca):
    feats = {k: v for k, v in info["features"].items() if not k.startswith(("action", "observation."))
             or k == f"observation.images.{cabeca}"}
    dims = {"ee_pose_gripper_base": ["x", "y", "z", "roll", "pitch", "yaw"],
            "fig6d": ["polegar1", "polegar2", "indicador0", "indicador1", "medio0", "medio1"]}
    for pref in ("action", "observation.state"):
        for l in ("left", "right"):
            for k, nm in dims.items():
                feats[f"{pref}.{l}_{k}"] = {"dtype": "float32", "shape": [len(nm)], "names": nm}
            feats[f"{pref}.{l}_leg"] = {"dtype": "float32", "shape": [6], "names": [f"j{i}" for i in range(6)]}
    feats["action.waist_action_joint"] = {"dtype": "float32", "shape": [3], "names": ["yaw", "roll", "pitch"]}
    feats["observation.state.waist_state_joint"] = {"dtype": "float32", "shape": [3], "names": ["yaw", "roll", "pitch"]}
    feats["action.base_command"] = {"dtype": "float32", "shape": [4], "names": ["vx", "vy", "angle_z", "height"]}
    return feats


def converte(nome, cfg, origem, destino, procs, py):
    src = origem / nome
    info = json.load(open(src / "meta/info.json"))
    v = info.get("codebase_version", "v2.1")
    tmp = destino / (nome + ("__v2" if v.startswith("v2") else ""))
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "data").mkdir(parents=True)
    shutil.copytree(src / "meta", tmp / "meta")
    cab = f"observation.images.{cfg['cabeca']}"
    # só o vídeo da cabeça, por link (não copia 30 GB)
    for p in (src / "videos").rglob("*.mp4"):
        if cab in p.parts:
            alvo = tmp / p.relative_to(src)
            alvo.parent.mkdir(parents=True, exist_ok=True)
            alvo.symlink_to(p.resolve())
    nomes_s = POLVOS28 if cfg.get("sem_nomes") else nomes_de(info, "observation.state")
    nomes_a = POLVOS28 if cfg.get("sem_nomes") else nomes_de(info, "action")
    arqs = sorted((src / "data").rglob("episode_*.parquet" if v.startswith("v2") else "*.parquet"))   # press_knob
    # tem um file-000.parquet quebrado no meio dos episode_*.parquet (sobra de outra versão)
    tarefas = []
    for a in arqs:
        dst = tmp / "data" / a.relative_to(src / "data")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tarefas.append((a, dst, cfg, nomes_s, nomes_a))
    with Pool(procs) as pool:
        n = sum(pool.map(converte_arquivo, tarefas, chunksize=1))
    info["features"] = feats_novas(info, cfg["cabeca"])
    info["robot_type"] = "unitree_g1_dex3"
    json.dump(info, open(tmp / "meta/info.json", "w"), indent=4)
    if (tmp / "meta/stats.json").exists():
        st = json.load(open(tmp / "meta/stats.json"))
        json.dump({k: v_ for k, v_ in st.items() if k in info["features"]}, open(tmp / "meta/stats.json", "w"))
    for velho in ("episodes_stats.jsonl",):
        (tmp / "meta" / velho).unlink(missing_ok=True)
    if not v.startswith("v2"):
        print(f"✅ {nome}: {n} quadros (v3.0) -> {tmp}", flush=True)
        return
    # v2.x -> v3.0 pelo conversor oficial
    info["codebase_version"] = "v2.1"
    json.dump(info, open(tmp / "meta/info.json", "w"), indent=4)
    episodes_stats(tmp, info)
    final = destino / nome
    if final.exists():
        shutil.rmtree(final)
    r = subprocess.run([py, "-m", "lerobot.datasets.v30.convert_dataset_v21_to_v30", f"--repo-id=terceiros/{nome}",
                        f"--root={tmp}", "--push-to-hub=false"], capture_output=True, text=True)
    if r.returncode == 0:   # converte NO LUGAR: tmp vira v3.0 e o v2 fica em tmp_old
        shutil.move(str(tmp), str(final))
        shutil.rmtree(str(tmp) + "_old", ignore_errors=True)
    else:
        print(r.stdout[-1500:], r.stderr[-3000:], flush=True)
    print(f"{'✅' if r.returncode == 0 else '❌'} {nome}: {n} quadros (v2 -> v3.0) | código {r.returncode}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origem", default="/data/mrwlker/datasets_terceiros")
    ap.add_argument("--destino", default="/data/mrwlker/unifolm_data/Terceiros_G1_Dex3")
    ap.add_argument("--so", nargs="*", help="só estes datasets")
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    origem, destino = Path(a.origem), Path(a.destino)
    destino.mkdir(parents=True, exist_ok=True)
    for nome, cfg in DATASETS.items():
        if a.so and nome not in a.so:
            continue
        converte(nome, cfg, origem, destino, a.procs, sys.executable)


if __name__ == "__main__":
    main()
