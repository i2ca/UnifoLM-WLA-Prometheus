#!/usr/bin/env python
"""
MODO SOMBRA — UnifoLM-WLA-1.0 INTEIRO no G1 real (câmera da cabeça + juntas), ZERO comando
==========================================================================================
Cabine http://<pgx>:8090 (caixa de tarefa, parar, MCP da super IA na 8091). A cada consulta,
da MESMA entrada (imagem da cabeça + frase + estado do robô no formato de pré-treino):

  1. AÇÃO CONTÍNUA — `predict_action` oficial (VLM -> DiT/MMDiT): 30 passos de pose das duas
     garras (xyz + rpy, pelvis), abertura da garra Dex1 (0 fechada ... 5,5 aberta), pernas,
     cintura e base. Exatamente o que o servidor oficial devolveria.
  2. TEXTO DO VLM (ER-Flow) — o MESMO prompt de pré-treino (câmera rotulada, Task, State com
     o estado injetado, Control Mode), mas com o turno do assistente aberto: o VLM continua
     sozinho. Mostramos o texto cru, com os tokens especiais (<segNNN>, <|POS...|>, ...).
     O repositório publicado não tem decodificador dos tokens discretos (VQ/RVQ não saiu):
     aqui eles aparecem crus, e é para ver SE e COMO ele os emite.
  3. ATENÇÃO do DiT sobre a imagem (para onde as ações olham).

Adaptações ao nosso robô (só leitura, nada é executado):
  - uma câmera só: a cabeça (RealSense), rotulada head_left — o treino tinha cabeça + 2 punhos;
  - Dex3 -> "garra" Dex1: fechamento médio dos dedos, 0 rad = aberta (5,5) ... 1,5 rad = fechada (0);
  - estado das garras em xyz+rot6d por FK (fk_g1.py, URDF do G1, pelvis fixa).

Abre na rede do robô só SUB: 5555 (câmera), 6001 (lowstate), 6002 (mãos), 6007 (trava).

    cd ~/DEV/unifolm-wla && ~/miniforge3/envs/wla/bin/python \\
        ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/sombra_wla.py
"""
import argparse
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import zmq

AQUI = Path(__file__).resolve().parent
EXT = AQUI.parent
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "model_server"))
sys.path.insert(0, str(EXT))
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(Path.home() / "wla_testes"))

import mapa_atencao as ma  # noqa: E402  (instala o espião de atenção no DiT; importa o servidor oficial)
from fk_g1 import FK  # noqa: E402
from pontes.cabine.servidor import Cabine, sobe  # noqa: E402
from robot.Scripts_Prometheus_int.sim.sensor_utils import ImageUtils, SensorClient  # noqa: E402

srv_mod = ma.srv_mod
REG = ma.REG
FRASE_TREINO = "Pick up the fruit and place it on the plate."


class Leitor:
    def __init__(self, ctx, robo, porta):
        self.msg, self.t = None, 0.0
        s = ctx.socket(zmq.SUB)
        s.setsockopt_string(zmq.SUBSCRIBE, "")
        s.setsockopt(zmq.RCVHWM, 2)
        s.connect(f"tcp://{robo}:{porta}")
        threading.Thread(target=self._laco, args=(s,), daemon=True).start()

    def _laco(self, s):
        while True:
            try:
                self.msg, self.t = json.loads(s.recv().decode("utf-8")), time.time()
            except Exception:
                time.sleep(0.02)

    def vivo(self, idade=0.5):
        return self.msg is not None and time.time() - self.t < idade


class Maos:
    def __init__(self, ctx, robo):
        self.q, self.t = {}, {}
        s = ctx.socket(zmq.SUB)
        s.setsockopt_string(zmq.SUBSCRIBE, "")
        s.connect(f"tcp://{robo}:6002")
        threading.Thread(target=self._laco, args=(s,), daemon=True).start()

    def _laco(self, s):
        while True:
            try:
                d = json.loads(s.recv().decode("utf-8"))["data"]
                self.q[d["side"]] = [float(m["q"]) for m in d["motor_state"]]
                self.t[d["side"]] = time.time()
            except Exception:
                time.sleep(0.02)

    def vivo(self):
        return all(time.time() - self.t.get(k, 0) < 0.5 for k in ("left", "right"))

    def garra(self, lado):
        """Dex3 -> escala da garra Dex1 do WLA (5,5 aberta ... 0 fechada)."""
        q = self.q.get(lado)
        if q is None:
            return 5.5
        fech = float(np.clip(np.mean(np.abs(q)) / 1.5, 0, 1))
        return 5.5 * (1 - fech)


def para_4x3(img, w=640, h=480):
    """Formato da cabeça do dataset do WLA: 640x480. 16:9 (a câmera antiga, 848x480) é recortado
    no centro para 4:3 em vez de espremido."""
    ih, iw = img.shape[:2]
    if abs(iw / ih - w / h) > 0.01:
        nw = int(round(ih * w / h))
        if nw <= iw:
            img = img[:, (iw - nw) // 2:(iw - nw) // 2 + nw]
        else:
            nh = int(round(iw * h / w))
            img = img[(ih - nh) // 2:(ih - nh) // 2 + nh]
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


class Camera:
    """Quadro mais novo de UMA imagem da mensagem (head_camera = cor; head_stereo_left = IR esq.),
    já em 640x480 RGB. Publica na cabine a ~15 fps."""

    def __init__(self, robo, porta, cabine, nome="head_camera"):
        self.rgb, self.t, self.fps, self.nome, self.cabine = None, 0.0, 0.0, nome, cabine
        self.origem = None
        self.cli = SensorClient()
        self.cli.start_client(server_ip=robo, port=porta)
        threading.Thread(target=self._laco, daemon=True).start()

    def _laco(self):
        ult, n, t0 = 0.0, 0, time.time()
        while True:
            msg = self.cli.receive_message()
            img = (msg.get("images") or {}).get(self.nome)
            if img is None:
                continue
            bruto = ImageUtils.decode_image(img)
            self.origem = bruto.shape[1::-1]
            self.rgb, self.t = para_4x3(bruto), time.time()
            n += 1
            if self.t - t0 >= 2:
                self.fps, n, t0 = n / (self.t - t0), 0, self.t
            if self.t - ult > 1 / 15:
                ult = self.t
                self.cabine.publica_quadro(self.nome, self.rgb)


class Sombra(srv_mod.ActionServerWBCMsgpack):
    """O servidor oficial, chamado em processo. Guarda as entradas do VLM para a atenção."""

    def __init__(self, args):
        super().__init__(args)
        iface = self.model.qwen_vl_interface
        self.iface = iface
        self.img_tok = iface.model.config.image_token_id
        self.st_tok = iface.model.config.robot_state_token_id
        self._inp = None
        orig = iface.build_qwenvl_inputs

        def build(*x, **kw):
            out = orig(*x, **kw)
            self._inp = out
            REG.L = out["input_ids"].shape[1]
            return out

        iface.build_qwenvl_inputs = build
        self._orig_build = orig

    def acao(self, obs):
        prep = self._build_example(obs)
        REG.soma, REG.n, REG.ligado = None, 0, True
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                out = self.model.predict_action(examples=[prep["example"]])
            torch.cuda.synchronize()
        finally:
            REG.ligado = False
        ms = (time.perf_counter() - t0) * 1e3
        acao = self._encode_action(out["normalized_actions"][0], prep["unnorm_key"], prep["state_unnorm"])
        att = (REG.soma / REG.n).float().cpu().numpy() if REG.n else None
        return prep, acao, ms, att, self._inp

    @torch.inference_mode()
    def texto(self, prep, max_tokens):
        """Mesmo prompt da ação, com o turno do assistente aberto: o VLM continua sozinho."""
        m = self.model
        ex = prep["example"]
        from unifolm_wla.model.framework.VLM4A import QwenMMDiT as qm
        imgs = [qm.to_pil_preserve(ex["image"])]
        tam = getattr(m.config.datasets.vla_data, "obs_image_size", None)
        if tam:
            imgs = qm.resize_images(imgs, target_size=tam)
        inp = self._orig_build(images=imgs, instructions=[ex["lang"]], image_roles=[ex["image_roles"]],
                               arm_types=[ex["arm_type"]], add_generation_prompt=True)
        estados = m._build_projector_state([ex])
        proj = m.qwen_vl_interface.model.robot_state_projector
        tok = self.st_tok

        def gancho(_mod, args, saida):
            ids = args[0] if args else None
            if ids is None or estados is None:
                return saida
            mask = ids == tok
            if int(mask.sum()) != estados.shape[0]:
                return saida   # passos de decodificação: não há token de estado
            p = next(proj.parameters())
            e = proj(estados.to(device=p.device, dtype=p.dtype)).to(device=saida.device, dtype=saida.dtype)
            return saida.masked_scatter(mask.unsqueeze(-1).expand_as(saida), e)

        h = m.qwen_vl_interface.model.get_input_embeddings().register_forward_hook(gancho)
        t0 = time.perf_counter()
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ger = m.qwen_vl_interface.model.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
        finally:
            h.remove()
        ms = (time.perf_counter() - t0) * 1e3
        novos = ger[0, inp["input_ids"].shape[1]:]
        proc = m.qwen_vl_interface.processor
        cru = proc.tokenizer.decode(novos, skip_special_tokens=False)
        return cru, int(novos.shape[0]), ms


def desenha_trajetoria(ee_agora, acao, garras_agora, h=480, w=640):
    """Vista de cima (x para frente, y para a esquerda) e de lado (x, z) das garras, pelvis."""
    g = np.full((h, w, 3), 24, np.uint8)
    cv2.rectangle(g, (0, 0), (w, 26), (0, 0, 160), -1)
    cv2.putText(g, "WLA-1.0 SOMBRA - nada e enviado ao robo", (8, 19), 0, 0.55, (255, 255, 255), 2)
    cores = {"left": (255, 180, 80), "right": (80, 200, 255)}
    paineis = [("de cima (x frente, y esq)", 0, 1, 10), ("de lado (x frente, z cima)", 0, 2, w // 2 + 5)]
    for titulo, ia, ib, x0 in paineis:
        W, H, y0 = w // 2 - 15, h - 140, 40
        cv2.rectangle(g, (x0, y0), (x0 + W, y0 + H), (70, 70, 70), 1)
        cv2.putText(g, titulo, (x0 + 6, y0 + 16), 0, 0.42, (200, 200, 200), 1)
        pts = [ee_agora[l][[ia, ib]] for l in ee_agora] + [acao[f"action.{l}_ee_rpy"][0][:, [ia, ib]] for l in ee_agora]
        todos = np.vstack([np.atleast_2d(p) for p in pts])
        c = todos.mean(0)
        esc = min(W, H) * 0.8 / max(0.3, np.ptp(todos, 0).max())

        def px(v):
            if ib == 1:   # de cima: y para a esquerda da tela, x para cima
                return int(x0 + W / 2 - (v[1] - c[1]) * esc), int(y0 + H / 2 - (v[0] - c[0]) * esc)
            return int(x0 + W / 2 + (v[0] - c[0]) * esc), int(y0 + H / 2 - (v[1] - c[1]) * esc)
        for l, cor in cores.items():
            tr = acao[f"action.{l}_ee_rpy"][0][:, [ia, ib]]
            for k in range(len(tr) - 1):
                cv2.line(g, px(tr[k]), px(tr[k + 1]), cor, 2)
            cv2.circle(g, px(tr[-1]), 6, cor, -1)
            cv2.circle(g, px(ee_agora[l][[ia, ib]]), 7, (255, 255, 255), 2)
    for k, l in enumerate(("left", "right")):
        d = acao[f"action.{l}_ee_rpy"][0][-1, :3] - ee_agora[l][:3]
        gr = (acao[f"action.{l}_gripper"][0][:, 0] if f"action.{l}_gripper" in acao
              else 5.5 * (1.0 - np.asarray(acao[f"action.{l}_fig6d"][0]).mean(axis=1)))
        cv2.putText(g, f"{'esq' if l == 'left' else 'dir'}: delta fim xyz {np.round(d * 100, 1)} cm | "
                    f"garra {garras_agora[l]:.1f} -> {gr[-1]:.1f} (0=fecha 5.5=abre)", (10, h - 80 + k * 22), 0,
                    0.45, cores[l], 1)
    base = acao["action.base_command"][0][-1]
    cv2.putText(g, f"base vx {base[0]:+.2f} vy {base[1]:+.2f} vw {base[2]:+.2f} altura {base[3]:.2f}",
                (10, h - 30), 0, 0.45, (200, 200, 200), 1)
    cv2.putText(g, "branco = agora (FK)  linha = 30 passos previstos  bolinha = fim", (10, h - 10), 0, 0.42,
                (160, 160, 160), 1)
    return g[:, :, ::-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robo", default="192.168.123.164")
    ap.add_argument("--porta-cam", type=int, default=5555)
    ap.add_argument("--porta", type=int, default=8090)
    ap.add_argument("--ckpt_path", default="playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors")
    ap.add_argument("--unnorm_key", default="UnifoLM_G1_Dex1")
    ap.add_argument("--tokens", type=int, default=48, help="tokens gerados pelo VLM por consulta (0 desliga)")
    ap.add_argument("--texto-cada", type=int, default=1, help="gera o texto a cada N consultas")
    ap.add_argument("--log", default=str(Path.home() / f"sombra_wla_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    cabine = Cabine()
    sobe(cabine, a.porta)
    cabine.define_tarefa(FRASE_TREINO)
    cabine.publica_estado({"modo": "carregando o UnifoLM-WLA-1.0 ..."})
    ctx = zmq.Context.instance()
    low, pan, maos = Leitor(ctx, a.robo, 6001), Leitor(ctx, a.robo, 6007), Maos(ctx, a.robo)
    cam = Camera(a.robo, a.porta_cam, cabine)
    fk = FK()

    ns = argparse.Namespace(ckpt_path=a.ckpt_path, instruction=FRASE_TREINO, unnorm_key=a.unnorm_key, use_bf16=True,
                            image_size=[320, 448], debug_save_dir=None)
    t = time.time()
    S = Sombra(ns)
    print(f"🕶  WLA-1.0 carregado em {time.time() - t:.0f} s | cabine http://0.0.0.0:{a.porta}/ | "
          f"nada é enviado ao robô", flush=True)
    log = open(a.log, "a")
    n, ultimo, ultimo_txt = 0, {}, {}
    while True:
        frase = cabine.tarefa()[0] or FRASE_TREINO
        real = low.vivo() and maos.vivo()
        base = {"modo": "SOMBRA WLA-1.0 — zero comando ao robô", "diagnostico_robo": f"http://{a.robo}:8095/",
                "camera": {"fps": round(cam.fps, 1), "idade_s": round(time.time() - cam.t, 2) if cam.t else None},
                "estado_das_juntas": "real (6001/6002)" if real else "SEM ESTADO: ponte fora do ar",
                "trava_do_robo": pan.msg if pan.vivo(1.0) else "sem sinal da ponte v3"}
        if cabine.parada_pedida() or cam.rgb is None or not real:
            motivo = "parado pelo botão" if cabine.parada_pedida() else "sem câmera" if cam.rgb is None else "sem estado"
            cabine.publica_estado({**base, "consultando": False, "motivo": motivo, **ultimo, **ultimo_txt})
            time.sleep(0.3)
            continue
        q = np.array([float(m["q"]) for m in low.msg["data"]["motor_state"][:29]])
        ee = {l: fk.ee9(q, l) for l in ("left", "right")}
        garras = {l: maos.garra(l) for l in ("left", "right")}
        obs = {"observation.images.cam_left_high": cv2.cvtColor(cam.rgb, cv2.COLOR_RGB2BGR),  # o servidor espera BGR
               "observation.state.left_ee_6d": ee["left"], "observation.state.right_ee_6d": ee["right"],
               "observation.state.left_gripper": np.array([garras["left"]], np.float32),
               "observation.state.right_gripper": np.array([garras["right"]], np.float32),
               "observation.state.lower_body": q[:15].astype(np.float32), "instruction": frase}
        prep, acao, ms, att, inp = S.acao(obs)
        n += 1
        cabine.publica_quadro("wla_trajetoria", desenha_trajetoria(ee, acao, garras))
        if att is not None:
            try:
                fig, *massas = ma.desenha(att, inp, prep["example"], S.img_tok, S.st_tok, f"consulta {n} | '{frase}'")
                cabine.publica_quadro("wla_atencao", np.ascontiguousarray(fig[:, :, ::-1]))
            except Exception as e:  # noqa: BLE001
                massas = [str(e)]
        else:
            massas = None
        ultimo = {"consulta": n, "acao_ms": round(ms), "tarefa_no_modelo": frase,
                  "garras_agora": {k: round(v, 2) for k, v in garras.items()},
                  "fim_do_chunk": {k.removeprefix("action."): np.round(v[0][-1], 3).tolist() for k, v in acao.items()},
                  "delta_ee_cm": {l: np.round((acao[f"action.{l}_ee_rpy"][0][-1, :3] - ee[l][:3]) * 100, 1).tolist()
                                  for l in ("left", "right")},
                  "atencao": {"imagem": round(float(sum(massas[0])), 3), "estado": round(float(massas[1]), 3)}
                  if massas and not isinstance(massas[0], str) else massas}
        if a.tokens and n % a.texto_cada == 0:
            cru, nt, tms = S.texto(prep, a.tokens)
            ultimo_txt = {"vlm_texto": cru, "vlm_tokens": nt, "vlm_ms": round(tms)}
        cabine.publica_estado({**base, "consultando": True, **ultimo, **ultimo_txt})
        log.write(json.dumps({"t": time.time(), **ultimo, **ultimo_txt}, default=str) + "\n")
        log.flush()
        print(f"[{n:04d}] ação {ms:4.0f} ms | Δ dir {ultimo['delta_ee_cm']['right']} cm | garra dir "
              f"{garras['right']:.1f}->{acao['action.right_gripper'][0][-1, 0]:.1f} | VLM: "
              f"{ultimo_txt.get('vlm_texto', '')[:90]!r}", flush=True)


if __name__ == "__main__":
    main()
