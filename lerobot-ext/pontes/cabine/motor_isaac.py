#!/usr/bin/env python
"""Motor da cabine que fala com o simulador da Unitree (IsaacLab), sem janela.

    ~/sobe_sim_dex3.sh                                     # o simulador, headless
    TAREFA=Isaac-PickPlace-Cafe-G129-Dex3-Joint ~/sobe_sim_dex3.sh
    python -m pontes.cabine.cerebro --motor isaac          # a rede, e o painel

Por que existe: o MuJoCo com janela VAZA memória no `sync()` do visualizador
(~1 GB/min, medido em 22/09). Aqui ninguém renderiza na tela: o simulador roda
headless, publica as câmeras por ZMQ, e o painel do navegador mostra esses
mesmos quadros. O que o robô vê é o que você vê.

── Três coisas que não são óbvias, herdadas do `roda_politica_g1_isaaclab.py` ──
  * o DDS do simulador fala no **domínio 1**, não no 0. Cliente no domínio
    errado não recebe nada e não dá erro nenhum;
  * as câmeras NÃO vêm por DDS: são publicadores ZMQ (55555 cabeça, 55556 punho
    esquerdo, 55557 punho direito) mandando JPEG;
  * o simulador ignora kp/kd do `rt/lowcmd` — ele lê só as posições das 29
    juntas. Então basta preencher `motor_cmd[j].q`;
  * e o mais traiçoeiro: o `get_action` deles faz `full_action.zero_()` a cada
    passo e SÓ preenche se houver comando novo no buffer. Como o termo de ação
    da cena usa `use_default_offset=True`, todo passo sem comando nosso puxa o
    braço de volta para a pose padrão. Publicando a 30 Hz, metade dos passos do
    simulador ficava sem comando e desfazia o movimento — o robô parecia
    parado. Por isso aqui uma THREAD publica o último alvo sem parar, a 200 Hz,
    e o laço da política só troca esse alvo. É o que a ponte do robô real faz.

A mão Dex3 tem tópico próprio, `rt/dex3/{left,right}/{state,cmd}`, com 7 juntas
por mão — a mesma da ponte do robô real.

── O que este teste mede, e o que NÃO mede ────────────────────────────────
A rede foi treinada com as câmeras ANTIGAS do nosso MuJoCo e com a mesa marrom.
Aqui a cena é a do IsaacLab (mesa clara, câmeras na pose real do URDF). É outra
distribuição visual: o que se mede é se o laço fecha — imagem entra, junta sai,
o robô se mexe — e não se a política resolve a tarefa.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "pontes" / "unifolm-vla"))
sys.path.insert(0, str(RAIZ / "maquinas" / "pgx"))

DOMINIO = 1
# papel que o treino usa -> (nome no simulador, porta ZMQ)
CAMERAS = {"head_left": ("cam_left_high", 55555), "cam_wrist_right": ("cam_right_wrist", 55557)}
BRACOS = list(range(15, 29))     # as 14 juntas dos braços, ordem do dataset
CINTURA = 12


class MotorIsaac:
    """A UnifoLM-WLA no laço com o IsaacLab. Mesma interface do `MotorWLA`."""

    def __init__(self, pasta: Path, base_vlm: Path, lead: int = 12, hz: float = 30.0,
                 max_consultas: int = 40, host: str = "127.0.0.1", reseta: bool = False):
        from pontes.wla.inferencia_wla import PoliticaWLA
        from roda_politica_g1_isaaclab import CameraZMQ
        import roda_unifolm_mujoco as ponte

        self.ponte = ponte
        self.cin = ponte.Cinematica()
        t0 = time.time()
        print(f"⏳ carregando a WLA de {pasta}…")
        self.pol = PoliticaWLA(pasta, base_vlm)
        print(f"✅ WLA carregada em {time.time() - t0:.0f} s — {self.pol.carga}")

        self.cameras = {papel: CameraZMQ(nome, porta, host)
                        for papel, (nome, porta) in CAMERAS.items()}
        for c in self.cameras.values():
            c.start()
        self._abre_dds()

        self.lead, self.hz, self.max_consultas = int(lead), float(hz), int(max_consultas)
        self.reseta = reseta
        self.info, self.cabine = {}, None
        self.pedido_copo = threading.Event()
        self.tarefa_treino = "pick up the white cup"
        self._abre_trabalhador()

    # ── DDS ─────────────────────────────────────────────────────────────
    def _abre_dds(self):
        from unitree_sdk2py.core.channel import (ChannelFactoryInitialize, ChannelPublisher,
                                                 ChannelSubscriber)
        from unitree_sdk2py.idl.default import (unitree_go_msg_dds__MotorCmd_,
                                                unitree_hg_msg_dds__LowCmd_)
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState
        from unitree_sdk2py.utils.crc import CRC

        ChannelFactoryInitialize(DOMINIO)
        self._estado = {"corpo": None, "mao_e": None, "mao_d": None}
        self._subs = []
        for topico, tipo, chave in (("rt/lowstate", hg_LowState, "corpo"),
                                    ("rt/dex3/left/state", MotorStates_, "mao_e"),
                                    ("rt/dex3/right/state", MotorStates_, "mao_d")):
            s = ChannelSubscriber(topico, tipo)
            s.Init(lambda m, c=chave: self._estado.__setitem__(c, m), 10)
            self._subs.append(s)

        self._pub_corpo = ChannelPublisher("rt/lowcmd", hg_LowCmd)
        self._pub_corpo.Init()
        self._pub_mao = {}
        for lado in ("left", "right"):
            p = ChannelPublisher(f"rt/dex3/{lado}/cmd", MotorCmds_)
            p.Init()
            self._pub_mao[lado] = p
        self._pub_reset = ChannelPublisher("rt/reset_pose/cmd", String_)
        self._pub_reset.Init()

        self._msg = unitree_hg_msg_dds__LowCmd_()
        self._crc = CRC()
        self._MotorCmd, self._MotorCmds = unitree_go_msg_dds__MotorCmd_, MotorCmds_
        self._String = String_
        self._alvo = None                 # (q14, yaw, dedos_esq, dedos_dir)
        self._trava_alvo = threading.Lock()
        threading.Thread(target=self._bombeia, daemon=True, name="lowcmd").start()

    def _bombeia(self, hz: float = 200.0):
        """Repete o último alvo sem parar. Ver a terceira armadilha no cabeçalho."""
        periodo = 1.0 / hz
        while True:
            with self._trava_alvo:
                alvo = self._alvo
            if alvo is not None:
                q14, yaw, dedos_e, dedos_d = alvo
                for k, j in enumerate(BRACOS):
                    self._msg.motor_cmd[j].q = float(q14[k])
                self._msg.motor_cmd[CINTURA].q = float(yaw)
                self._msg.crc = self._crc.Crc(self._msg)
                self._pub_corpo.Write(self._msg)
                self._manda_mao("left", dedos_e)
                self._manda_mao("right", dedos_d)
            time.sleep(periodo)

    def define_alvo(self, q14, yaw, dedos_e, dedos_d) -> None:
        with self._trava_alvo:
            self._alvo = (np.asarray(q14, float).copy(), float(yaw),
                          np.asarray(dedos_e, float).copy(), np.asarray(dedos_d, float).copy())

    def espera_simulador(self, segundos: float = 30.0) -> bool:
        """O simulador pode estar carregando: 30 s é o tempo que ele leva a frio."""
        t0 = time.time()
        while time.time() - t0 < segundos:
            if self._estado["corpo"] is not None and all(c.le() is not None
                                                         for c in self.cameras.values()):
                return True
            time.sleep(0.5)
        return False

    def _manda_mao(self, lado: str, angulos) -> None:
        cmds = []
        for v in angulos:
            c = self._MotorCmd()
            c.q = float(v)
            cmds.append(c)
        self._pub_mao[lado].Write(self._MotorCmds(cmds=cmds))

    def _le_estado(self):
        c = self._estado["corpo"]
        if c is None:
            return None
        q14 = np.array([c.motor_state[j].q for j in BRACOS], float)
        yaw = float(c.motor_state[CINTURA].q)
        mao = {}
        for chave, lado in (("mao_e", "esq"), ("mao_d", "dir")):
            m = self._estado[chave]
            mao[lado] = (np.array([x.q for x in m.states[:7]], float) if m is not None
                         else np.zeros(7))
        return q14, yaw, mao

    # ── a thread da rede: UMA, pela vida do motor ───────────────────────
    def _abre_trabalhador(self):
        import queue
        self.pedidos, self.respostas = queue.Queue(maxsize=1), queue.Queue()
        self._fim = threading.Event()

        def trabalhador():
            while not self._fim.is_set():
                try:
                    passo_obs, obs = self.pedidos.get(timeout=0.2)
                except queue.Empty:
                    continue
                t0 = time.perf_counter()
                out = self.pol.age(obs["imagens"], obs["texto"], obs["T_esq"], obs["T_dir"],
                                   obs["garra_esq"], obs["garra_dir"], obs["cintura3"])
                self.respostas.put((passo_obs, out, time.perf_counter() - t0,
                                    obs["T_dir"][:3, 3]))

        self.trab = threading.Thread(target=trabalhador, daemon=True, name="wla-isaac")
        self.trab.start()

    def _T(self, par):
        T = np.eye(4)
        T[:3, 3], T[:3, :3] = par
        return T

    def _observa(self, texto, q14, yaw, mao):
        ponte = self.ponte
        imagens = {papel: c.le() for papel, c in self.cameras.items()}
        if self.cabine is not None:                      # o painel vê o MESMO quadro
            for papel, img in imagens.items():
                self.cabine.publica_quadro(f"{papel}_isaac", img)
        esq, dir_ = self.cin.fk(q14)
        return {
            "imagens": imagens, "texto": texto,
            "T_esq": self._T(esq), "T_dir": self._T(dir_),
            "garra_esq": ponte.dedos_para_garra(mao["esq"], "esq"),
            "garra_dir": ponte.dedos_para_garra(mao["dir"], "dir"),
            "cintura3": np.array([0.0, 0.0, yaw], np.float32),
        }

    def executa(self, texto: str, n: int, ao_passo, dorme: float = 0.0) -> str:
        import queue

        if not self.espera_simulador():
            return ("o simulador não respondeu: confira se o `sobe_sim_dex3.sh` está de pé "
                    "(estado do corpo por DDS no domínio 1 e as câmeras em 55555/55557)")
        if self.reseta:
            # Recoloca a cena (copo e coador) sem descarregar a rede da GPU.
            # DESLIGADO por padrão: em 22/09 o robô ficou sem obedecer depois
            # dele, e o simulador precisou ser reiniciado.
            self._pub_reset.Write(self._String(data="1"))
            time.sleep(1.0)

        while not self.respostas.empty():
            self.respostas.get_nowait()
        periodo = 1.0 / self.hz
        passo, pedaco, cursor, n_pedacos, em_voo, parado = 0, None, 0, 0, False, 0
        tempos = []
        q14, yaw, mao = self._le_estado()
        self.pedidos.put((0, self._observa(texto, q14, yaw, mao)))
        em_voo = True
        proximo = time.perf_counter()
        while True:
            try:
                r = self.respostas.get(timeout=30) if pedaco is None else self.respostas.get_nowait()
                passo_obs, pedaco, dt, mao_obs = r
                em_voo = False
                n_pedacos += 1
                tempos.append(dt)
                cursor = min(passo - passo_obs, len(pedaco["pose_dir"]) - 1)
                self._publica_consulta(n_pedacos, pedaco, mao_obs, texto, dt, passo - passo_obs)
            except queue.Empty:
                pass

            q14, yaw, mao = self._le_estado()
            restante = len(pedaco["pose_dir"]) - cursor
            if not em_voo and restante <= self.lead and n_pedacos < self.max_consultas:
                self.pedidos.put((passo, self._observa(texto, q14, yaw, mao)))
                em_voo = True

            if restante > 0:
                i = cursor
                pe, pd = pedaco["pose_esq"][i], pedaco["pose_dir"][i]
                q14 = self.cin.resolve((pe[:3, 3], pe[:3, :3]), (pd[:3, 3], pd[:3, :3]), q14)
                yaw_alvo = float(np.clip(pedaco["cintura"][i][2], -1.0, 1.0))
                g = float(pedaco["garra_dir"][i])
                self.define_alvo(q14, yaw_alvo,
                                 self.ponte.garra_para_dedos(float(pedaco["garra_esq"][i]), "esq"),
                                 self.ponte.garra_para_dedos(g, "dir"))
                chegou = self.cin.fk(q14)[1][0]
                self.info["acao"] = {
                    "passo_do_pedaco": i,
                    "alvo_mao_dir": [round(float(v), 4) for v in pd[:3, 3]],
                    "mao_dir_chegou": [round(float(v), 4) for v in chegou],
                    "erro_mao_cm": round(float(np.linalg.norm(chegou - pd[:3, 3])) * 100, 2),
                    "garra_dir": round(g, 3), "cintura_yaw": round(yaw_alvo, 4),
                    "simulador": "isaaclab",
                }
                cursor += 1
            else:
                parado += 1
            if not em_voo and restante <= 0:
                break

            ao_passo(f"isaac {n_pedacos}/{self.max_consultas}", q14)
            passo += 1
            self.info["laco"] = {
                "passo": passo, "pedaco": n_pedacos, "cursor_no_pedaco": cursor,
                "restam_no_pedaco": max(restante - 1, 0), "lead": self.lead,
                "rede_calculando": em_voo, "passos_parado_esperando_rede": parado,
                "quadros_recebidos": {p: c.recebidos for p, c in self.cameras.items()},
                "memoria_gpu_gb": round(self.pol.torch.cuda.memory_reserved() / 1e9, 2),
            }
            proximo += periodo
            espera = proximo - time.perf_counter()
            if espera > 0:
                time.sleep(espera)
            else:
                proximo = time.perf_counter()

        return (f"{n_pedacos} pedaços, rede {np.mean(tempos):.2f} s, parado {parado} passos "
                f"(IsaacLab; o resultado da tarefa se julga pela imagem)")

    def _publica_consulta(self, k, out, mao_obs, texto, dt, pulados):
        P = out["pose_dir"][:, :3, 3]
        self.info["predicao"] = {
            "pedaco": k, "texto_enviado": texto, "tempo_s": round(dt, 3),
            "pulados_por_atraso": int(pulados),
            "mao_dir_na_observacao": [round(float(v), 4) for v in mao_obs],
            "copo_pelvis": [0.0, 0.0, 0.0],     # o IsaacLab não nos diz onde está o copo
            "mao_dir_xyz": [[round(float(v), 4) for v in p] for p in P],
            "garra_dir": [round(float(v), 3) for v in out["garra_dir"]],
            "cintura_yaw": [round(float(v), 4) for v in out["cintura"][:, 2]],
        }
        if self.cabine is not None:
            for papel, img in out.get("atencao", {}).items():
                self.cabine.publica_quadro(f"atencao_{papel}", img)
