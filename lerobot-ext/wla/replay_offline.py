#!/usr/bin/env python
"""REPLAY OFFLINE (29/10): as consultas que o robô gravou (~/wla_real_runs/<rodada>, gravador_wla.py) passadas por
OUTROS checkpoints, sem robô e sem rede — o modelo é carregado aqui, como o servidor_wla_dex3.py faz (mesmas frases,
mesma montagem das câmeras e do estado), e cada consulta usa a MESMA semente do ruído do fluxo em todos.

Compara, por consulta: o fechamento da mão direita que o modelo pede no fim do trecho (0 aberta .. 1 fechada), para onde
a mão direita vai (cm em relação à medida) e o yaw da cintura. Sai <rodada>/replay_<run>_<passo>.csv e um resumo.

    cd ~/DEV/unifolm-wla-lora && CUDA_VISIBLE_DEVICES=2 python .../replay_offline.py --run playground/Checkpoints/prometheus_v7 \\
        --passos 8000 9000 --rodadas /data/mrwlker/wla_real_runs/20261029_142620 ...

Rodadas sem o olho direito (antes de 29/10 15h) ficam SEM profundidade em todos os checkpoints (comparação justa entre
eles, mas não idêntica ao robô).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))


def obs_da_consulta(pasta, n, frase):
    import cv2
    z = np.load(pasta / f"{n}.npz")
    obs = {"observation.images.cam_left_high": cv2.imread(str(pasta / f"{n}_cabeca.jpg")),
           "observation.state.left_ee_6d": z["ee_medido_left"].astype(np.float32),
           "observation.state.right_ee_6d": z["ee_medido_right"].astype(np.float32),
           "observation.state.left_fig6d": np.zeros(6, np.float32),    # o executor manda 0 (--dedos-estado zero)
           "observation.state.right_fig6d": np.zeros(6, np.float32),
           "observation.state.lower_body": z["q_medido"][:15].astype(np.float32),
           "instruction": frase}
    if (pasta / f"{n}_cabeca_dir.jpg").exists():
        obs["observation.images.head_stereo_right"] = cv2.imread(str(pasta / f"{n}_cabeca_dir.jpg"))
    if (pasta / f"{n}_punho_esq.jpg").exists() and (pasta / f"{n}_punho_dir.jpg").exists():
        obs["observation.images.cam_left_wrist"] = cv2.imread(str(pasta / f"{n}_punho_esq.jpg"))
        obs["observation.images.cam_right_wrist"] = cv2.imread(str(pasta / f"{n}_punho_dir.jpg"))
    return obs, z


_ORIG = {}


def servidor(base, run, passo):
    import servidor_wla_dex3 as S
    from unifolm_wla.model.modules.action_model import MMDiT_ActionHeader as mh
    # os ajustes do servidor EMBRULHAM funções da Unitree: volta às originais antes de cada checkpoint (senão empilham)
    if not _ORIG:
        _ORIG["fp"] = S.srv.baseframework.from_pretrained
        _ORIG["pa"] = mh.MMDiTFlowmatchingActionHead.predict_action
    S.srv.baseframework.from_pretrained = _ORIG["fp"]
    mh.MMDiTFlowmatchingActionHead.predict_action = _ORIG["pa"]
    S.instala_carregador_lora(run, passo)
    S.instala_amostragem(4, 2, 0.6)   # o padrão do servidor do robô
    import logging
    # o servidor da Unitree tem um logging.info com 2 "%s" e 1 argumento (o servidor_wla_dex3 faz o mesmo)
    logging.basicConfig(level=logging.WARNING, force=True)
    logging.getLogger().setLevel(logging.WARNING)
    a = argparse.Namespace(ckpt_path=base, instruction="", unnorm_key="Prometheus_G1_Dex3", use_bf16=True,
                           image_size=[336, 448], debug_save_dir=None, lora_run=run, lora_passo=passo,
                           profundidade="auto", grava_prof_s=0, passos_fluxo=4, amostras=2, escala_ruido=0.6,
                           host="0.0.0.0", port=0)
    return S.ServidorDex3(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--passos", nargs="+", type=int, required=True)
    ap.add_argument("--rodadas", nargs="+", required=True)
    ap.add_argument("--base", default="playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors")
    a = ap.parse_args()
    import torch
    rodadas = []
    for r in a.rodadas:
        p = Path(r)
        L = [json.loads(x) for x in open(p / "consultas.jsonl")]
        rodadas.append((p, L))
    for passo in a.passos:
        s = servidor(a.base, a.run, passo)
        for pasta, L in rodadas:
            linhas, fecha_tarefa = [], {}
            for x in L:
                n = f"{x['n']:04d}"
                if not (pasta / f"{n}.npz").exists():
                    continue
                obs, z = obs_da_consulta(pasta, n, x["frase"])
                torch.manual_seed(1000 + x["n"])
                acao = s.get_action(obs)
                f6 = np.asarray(acao["action.right_fig6d"])[0]           # (T, 6) 0..1
                fecha = float(np.mean(f6[-1]))
                ee = np.asarray(acao["action.right_ee_rpy"])[0][-1, :3]
                d = (ee - z["ee_medido_right"][:3]) * 100
                yaw = float(np.asarray(acao["action.lower_body"])[0][-1, 12])
                f_grav = float(np.mean(z["acao__right_fig6d"][-1]))
                linhas.append([x["n"], round(x["t"], 1), x["frase"][:30], round(fecha, 2), round(f_grav, 2),
                               *np.round(d, 1).tolist(), round(yaw, 3), round(float(z["acao__lower_body"][-1, 12]), 3)])
                fecha_tarefa.setdefault(x["frase"], []).append(fecha)
            nome = pasta / f"replay_{Path(a.run).name}_{passo}.csv"
            with open(nome, "w") as f:
                f.write("n,t,frase,fecha_novo,fecha_gravado,dx_cm,dy_cm,dz_cm,yaw_novo,yaw_gravado\n")
                for l in linhas:
                    f.write(",".join(str(v) for v in l) + "\n")
            res = " | ".join(f"{fr[:22]}: fecha>0,5 em {sum(v > 0.5 for v in vs)}/{len(vs)} consultas"
                             for fr, vs in fecha_tarefa.items())
            print(f"✅ {pasta.name} @ {passo}: {res}", flush=True)
        del s
        torch.cuda.empty_cache()
    print("FIM", flush=True)


if __name__ == "__main__":
    main()
