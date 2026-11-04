#!/usr/bin/env python
"""DEMONSTRAÇÃO da cabine do robô SEM o robô (03/11): uma rodada gravada (gravador_wla.py) toca na cabine nova como
se fosse ao vivo — cada consulta gravada vai de novo ao servidor do WLA (que calcula a atenção, se --atencao 1), e o
plano, as mãos, a cintura e as câmeras são publicados como o executor faz. NADA vai ao robô.

    cd ~/DEV/unifolm-wla && ~/miniforge3/envs/wla/bin/python .../demo_cabine.py ~/wla_real_runs/20261103_083335 \\
        --porta 8093      # e abra http://<pgx>:8093
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pasta")
    ap.add_argument("--porta", type=int, default=8093)
    ap.add_argument("--servidor", default="ws://127.0.0.1:8601")
    ap.add_argument("--de", type=int, default=0, help="consulta inicial")
    ap.add_argument("--voltas", type=int, default=1000)
    a = ap.parse_args()
    import cv2
    from fk_g1 import FK
    from pontes.cabine.servidor import Cabine, sobe
    from roda_wla_real import ClienteWLA
    from viz_cabine import VizCabine

    pasta = Path(a.pasta)
    L = [json.loads(x) for x in open(pasta / "consultas.jsonl")][a.de:]
    cab = Cabine()
    sobe(cab, a.porta)
    viz, fk = VizCabine(cab, FK()), None
    cli = ClienteWLA(a.servidor)
    cab.publica_estado({"modo": f"DEMONSTRAÇÃO — {pasta.name} (nada vai ao robô)"})
    print(f"🎬 demonstração em http://0.0.0.0:{a.porta}  ({len(L)} consultas de {pasta.name})", flush=True)
    for _ in range(a.voltas):
        for x in L:
            n = f"{x['n']:04d}"
            if not (pasta / f"{n}.npz").exists():
                continue
            z = np.load(pasta / f"{n}.npz")
            ims = {k: cv2.imread(str(pasta / f"{n}_{s}.jpg")) for k, s in
                   (("head_stereo_left", "cabeca"), ("head_stereo_right", "cabeca_dir"), ("wrist_left", "punho_esq"),
                    ("wrist_right", "punho_dir"))}
            for k, im in ims.items():
                if im is not None:
                    cab.publica_quadro(k, cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            obs = {"observation.images.cam_left_high": ims["head_stereo_left"],
                   "observation.state.left_ee_6d": z["ee_medido_left"], "observation.state.right_ee_6d": z["ee_medido_right"],
                   "observation.state.left_gripper": z["garras"][:1].astype(np.float32),
                   "observation.state.right_gripper": z["garras"][1:].astype(np.float32),
                   "observation.state.left_fig6d": np.zeros(6, np.float32),
                   "observation.state.right_fig6d": np.zeros(6, np.float32),
                   "observation.state.lower_body": z["q_medido"][:15].astype(np.float32), "instruction": x["frase"]}
            if ims["head_stereo_right"] is not None:
                obs["observation.images.head_stereo_right"] = ims["head_stereo_right"]
            if ims["wrist_left"] is not None and ims["wrist_right"] is not None:
                obs["observation.images.cam_left_wrist"] = ims["wrist_left"]
                obs["observation.images.cam_right_wrist"] = ims["wrist_right"]
            _, acao, ms, _, _ = cli.acao(obs)
            q = z["q_medido"]
            viz.plano({"acao": acao, "n": x["n"], "ms": ms, "frase": x["frase"]}, q)
            cab.define_tarefa(x["frase"]) if cab.tarefa()[0] != x["frase"] else None
            cab.publica_estado({"modo": f"DEMONSTRAÇÃO — {pasta.name} (nada vai ao robô)", "consulta": x["n"],
                                "acao_ms": round(ms), "tarefa_no_modelo": x["frase"],
                                "delta_pedido_pelo_modelo_cm": {l: np.round((np.asarray(acao[f"action.{l}_ee_rpy"])[0][-1, :3]
                                                                             - z[f"ee_medido_{l}"][:3]) * 100, 1).tolist()
                                                                for l in ("left", "right")}})
            yaw_plano = np.asarray(acao["action.lower_body"])[0][:, 12]
            f6 = np.asarray(acao["action.right_fig6d"])[0].mean(-1)
            t0 = time.time()
            for passo in range(0, 30, 3):   # o trecho "andando" em ~1 s
                viz.t = 0
                viz.vivo(time.time(), q, float(yaw_plano[passo]), {"left": 0.0, "right": float(f6[passo])}, passo,
                         {"cintura_roll": 45.0, "cintura_pitch": 41.0, "ombro_dir": 48.0, "ombro_esq": 47.0})
                time.sleep(max(0.0, t0 + (passo + 3) / 30 - time.time()))


if __name__ == "__main__":
    main()
