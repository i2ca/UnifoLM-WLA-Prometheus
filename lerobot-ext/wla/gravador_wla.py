"""Grava cada consulta do roda_wla_real.py para avaliar depois (relatorio_wla_real.py).

    ~/wla_real_runs/<AAAAMMDD_HHMMSS>/
        config.json                 argumentos da rodada
        consultas.jsonl             uma linha por consulta: tempo, frase, garras, deltas pedidos, trava...
        NNNN_cabeca.jpg             a imagem de cabeça que o MODELO recebeu (ZED esq., 640x480)
        NNNN_punho_esq.jpg          idem, punhos (se foram)
        NNNN_punho_dir.jpg
        NNNN_traj.jpg               trajetória prevista das duas mãos (de cima e de lado)
        NNNN.npz                    tudo em número: o trecho inteiro da IA (30 passos, todas as chaves),
                                    juntas medidas, juntas comandadas, pose das mãos, garras

A escrita é numa thread com fila: o controle nunca espera o disco (se a fila encher, a consulta
é pulada na gravação e contada em `pulados`).
"""
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import sombra_wla as sw

def garra_equivalente(a, lado):
    """Garra Dex1 (0 fechada .. 5,5 aberta); no modelo Dex3, da média do fig6d."""
    if f"action.{lado}_gripper" in a:
        return float(np.asarray(a[f"action.{lado}_gripper"])[0, -1, 0])
    return float(5.5 * (1.0 - np.mean(np.asarray(a[f"action.{lado}_fig6d"])[0, -1])))


IMAGENS = {"observation.images.cam_left_high": "cabeca", "observation.images.cam_left_wrist": "punho_esq",
           "observation.images.cam_right_wrist": "punho_dir"}


class Gravador:
    def __init__(self, raiz, config):
        self.pasta = Path(raiz) / time.strftime("%Y%m%d_%H%M%S")
        self.pasta.mkdir(parents=True, exist_ok=True)
        (self.pasta / "config.json").write_text(json.dumps(config, indent=1, default=str))
        self.jsonl = open(self.pasta / "consultas.jsonl", "a")
        self.fila = queue.Queue(maxsize=20)
        self.pulados = 0
        self.t0 = time.time()
        self.th = threading.Thread(target=self._laco, daemon=True, name="gravador")
        self.th.start()

    def consulta(self, tr, obs, q_cmd, trava, fk):
        imgs = {nome: obs[k] for k, nome in IMAGENS.items() if k in obs}
        try:
            self.fila.put_nowait((tr, imgs, np.array(q_cmd), dict(trava or {}),
                                  {l: fk.ee9(q_cmd, l) for l in ("left", "right")}))
        except queue.Full:
            self.pulados += 1

    def _laco(self):
        while True:
            item = self.fila.get()
            if item is None:
                return
            try:
                self._grava(*item)
            except Exception as e:  # noqa: BLE001
                print(f"[gravador] falhou: {e}", flush=True)

    def _grava(self, tr, imgs, q_cmd, trava, ee_cmd):
        n = tr["n"]
        base = self.pasta / f"{n:04d}"
        for nome, bgr in imgs.items():
            cv2.imwrite(f"{base}_{nome}.jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        fig = sw.desenha_trajetoria(tr["ee"], tr["acao"], tr["garras"])
        cv2.imwrite(f"{base}_traj.jpg", np.ascontiguousarray(fig[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 85])
        acao = {f"acao__{k.replace('action.', '')}": np.asarray(v)[0] for k, v in tr["acao"].items()}
        np.savez_compressed(f"{base}.npz", q_medido=tr["q"], q_comandado=q_cmd,
                            ee_medido_left=tr["ee"]["left"], ee_medido_right=tr["ee"]["right"],
                            ee_comandado_left=ee_cmd["left"], ee_comandado_right=ee_cmd["right"],
                            garras=np.array([tr["garras"]["left"], tr["garras"]["right"]]),
                            t_obs=tr["t_obs"], ms=tr["ms"], **acao)
        a = tr["acao"]
        linha = {
            "n": n, "t": round(tr["t_obs"] - self.t0, 3), "ms": round(tr["ms"]), "frase": tr["frase"],
            "imagens": sorted(imgs), "garras_agora": {k: round(v, 2) for k, v in tr["garras"].items()},
            "garra_fim_prevista": {l: round(garra_equivalente(a, l), 2) for l in ("left", "right")},
            "delta_pedido_fim_cm": {l: np.round((np.asarray(a[f"action.{l}_ee_rpy"])[0, -1, :3] - tr["ee"][l][:3]) * 100,
                                                1).tolist() for l in ("left", "right")},
            "mao_medida_cm": {l: np.round(tr["ee"][l][:3] * 100, 1).tolist() for l in ("left", "right")},
            "mao_comandada_cm": {l: np.round(ee_cmd[l][:3] * 100, 1).tolist() for l in ("left", "right")},
            "base_fim": np.round(np.asarray(a["action.base_command"])[0, -1], 3).tolist(),
            "trava": {k: trava.get(k) for k in ("panico", "motivo")},
        }
        self.jsonl.write(json.dumps(linha) + "\n")
        self.jsonl.flush()

    def fecha(self):
        self.fila.put(None)
        self.th.join(timeout=10)
        self.jsonl.close()
        if self.pulados:
            print(f"[gravador] {self.pulados} consultas não gravadas (disco lento)", flush=True)
