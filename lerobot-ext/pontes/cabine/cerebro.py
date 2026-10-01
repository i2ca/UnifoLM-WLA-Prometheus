#!/usr/bin/env python
"""Cérebro da cabine — MuJoCo de um lado, tarefa em texto do outro.

    python -m pontes.cabine.cerebro                      # motor roteirizado
    ~/envs/wla-inferencia/bin/python -m pontes.cabine.cerebro --motor wla

Abre `http://<ip>:8090/`, escreve a tarefa na caixa (ou manda pelo MCP) e o robô
executa. O painel mostra as duas câmeras que a política vê e o estado das juntas.

── O motor é trocável, e isso é o ponto ────────────────────────────────────
`--motor roteirizado` usa o especialista do `demo_pega_copo_mujoco.py`: o mesmo
que gravou os 1.828 episódios, confiável, e que NÃO precisa de checkpoint. Serve
para a demonstração da super IA existir antes de a rede estar boa.
`--motor wla` põe a UnifoLM-WLA no lugar. A casca — painel, MCP, tarefas — é a
mesma nos dois casos, de propósito: o dia em que a rede ficar boa, troca-se uma
palavra na linha de comando.

── O roteador de texto é burro, e está escrito para ser ────────────────────
Hoje ele procura palavra-chave e escolhe entre três roteiros. Isso NÃO é
compreensão de linguagem; é um mapa de mesa. Quem entende texto de verdade é a
VLA, e no `--motor wla` a frase vai inteira para o campo `lang` do modelo, sem
passar por aqui. O roteador só existe para o motor roteirizado ter o que fazer.

── Por que o simulador roda numa thread e não no processo do servidor ──────
O laço do MuJoCo com IK é síncrono e demora ~18 s por episódio. Se ele rodasse
na thread do HTTP, o painel congelaria durante a execução — exatamente quando
alguém está olhando. Aqui o laço é dono da sua thread e só deposita quadros.
"""
from __future__ import annotations

import os
import sys

# O backend do GL é escolhido no import do mujoco, antes de qualquer argparse —
# por isso o `--janela` é lido na unha, direto do argv (mesmo truque do
# `roda_pi05_g1_sim.py`). EGL renderiza offscreen mas NÃO abre janela; GLFW abre
# janela E renderiza offscreen (medido lá). Sem `--janela`, EGL: quem assiste,
# assiste pelo navegador.
_JANELA = "--janela" in sys.argv
os.environ.setdefault("MUJOCO_GL", "glfw" if _JANELA else "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")   # ipopt multithread come 600% de CPU

import argparse
import threading
import time
import traceback
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

import mujoco  # noqa: E402

from pontes.cabine.servidor import Cabine, sobe  # noqa: E402


# Câmeras publicadas no painel. As duas primeiras são os `image_roles` do treino
# (`head_left` e `cam_wrist_right` no `dados_prometheus.yaml`) — é o que a
# política enxerga. A `global_view` é só para quem assiste entender a cena.
# `head_camera_antiga` está aqui de propósito: o dataset `nosso_sim_pega`, que é
# 97% do que treinou a rede, foi gravado com a pose errada. Enquanto for assim, é
# ELA que casa com o que a política viu — a `head_camera` (pose real, corrigida
# em 21/09) fica ao lado para dar para comparar as duas no painel.
CAMERAS = {
    "head_camera": (480, 848),
    "head_camera_antiga": (480, 848),
    "right_wrist_camera": (480, 640),
    "global_view": (480, 640),
}
# Qual delas alimenta a política. Câmeras que não existem na cena são ignoradas.
CAMERA_DA_POLITICA = os.environ.get("CABINE_CAM_CABECA", "head_camera_antiga")


class MotorRoteirizado:
    """Especialista do `demo_pega_copo_mujoco.py`, sem rede neural nenhuma."""

    # Palavra-chave → roteiro. Ordem importa: "põe" ganha de "pega" numa frase
    # que tem as duas ("pegue a xícara e ponha no filtro" é o roteiro completo).
    MAPA = (
        (("completo", "café", "cafe", "tudo"), "completo"),
        (("filtro", "coador", "dripper", "ponha", "poe", "põe", "place", "put"), "poe"),
        (("xícara", "xicara", "caneca", "copo", "cup", "pegue", "pega", "pick"), "pega"),
    )

    def __init__(self):
        import demo_pega_copo_mujoco as demo

        # O demo faz um remendo no módulo do IK antes de expor a classe (ver o
        # comentário dele por quê); importar `G1_29_ArmIK` direto do módulo
        # original pula esse remendo e quebra.
        self.demo = demo
        self.cena = demo.Cena(com_janela=False)
        self.ik = demo.G1_29_ArmIK(
            visualization=False, travar_punho_dir=False,
            postura_ref=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0], float),
        )
        self.descanso = np.array([0.42, -0.26, self.cena.z_copo + 0.22])
        self.cena.define_pose_partida(self.ik, self.descanso)
        self.cena.descanso = self.descanso
        self.rng = np.random.default_rng(0)

    def roteiro_de(self, texto: str) -> str | None:
        t = (texto or "").lower()
        for chaves, roteiro in self.MAPA:
            if any(c in t for c in chaves):
                return roteiro
        return None

    def executa(self, texto: str, n: int, ao_passo, dorme: float = 0.0) -> str:
        roteiro = self.roteiro_de(texto)
        if roteiro is None:
            return (f"não sei fazer {texto!r}. O motor roteirizado conhece: pegar a "
                    f"xícara, pôr no filtro, ou o roteiro completo.")
        self.demo.episodio(self.cena, self.ik, self.rng, n, dorme=dorme,
                           ao_passo=ao_passo, tarefa=roteiro)
        return f"executei o roteiro {roteiro!r}"


class MotorWLA:
    """UnifoLM-WLA no laço, sobre a MESMA cena do motor roteirizado.

    ── Pedaço (chunk) e antecedência (lead) ────────────────────────────────
    O braço anda a `hz` fixos (30, o fps do dataset: cada passo previsto é
    1/30 s) e a rede roda NUMA THREAD À PARTE. Quando faltam `lead` passos para
    o pedaço atual acabar, a observação daquele instante vai para a rede; a
    resposta chega ~0,22 s depois (≈ 7 passos) e é ALINHADA: os passos que já
    passaram enquanto ela pensava são pulados, porque foram previstos para um
    instante que já é passado. O braço nunca para para esperar, a não ser que a
    rede demore mais que o `lead` inteiro — aí ele segura a última pose.

    A versão anterior era síncrona: executava 10 passos, PARAVA ~0,6 s para
    perguntar (predição + gráfico), e repetia. Metade do tempo parado.

    ── Custos, medidos em 22/09 na GB10 ────────────────────────────────────
    predição 0,22 s · IK 2,9 ms/passo · render 1,6–2,2 ms/câmera · JPEG 1,8 ms.
    O gráfico em matplotlib custava 173 ms por consulta segurando o GIL, e saiu:
    o navegador desenha a predição a partir do JSON.

    ── As câmeras são as ANTIGAS, de propósito ─────────────────────────────
    O `nosso_sim_pega`, 97% do treino, foi gravado com as poses de câmera que
    valiam até 21/09. A rede só reconhece a cena por elas. Por isso a cabeça é a
    `head_camera_antiga` (480x848) e o pulso a `right_wrist_camera_antiga`
    (224x224, a resolução do dataset) — as corretas continuam no painel, para
    quem assiste.

    ── A cinemática é a do CONVERSOR ───────────────────────────────────────
    `Cinematica` de `roda_unifolm_mujoco.py`: o mesmo modelo reduzido e os
    mesmos frames `L_ee`/`R_ee` que o `converte_dataset_wla.py` usou para gerar
    a pose da mão do dataset. Qualquer outra FK daria um deslocamento
    constante entre o que o modelo aprendeu e o que ele recebe.

    ── A pega é cinemática ─────────────────────────────────────────────────
    Esta cena usa `mj_forward`, sem física de contato (ver o cabeçalho do
    demo). O roteirizado carrega a xícara com `poe_copo(ponto_pega())`; aqui é
    igual: se a garra fecha com a pinça a menos de `RAIO_PEGA` da xícara, ela
    passa a seguir a mão até a garra abrir. É uma conveniência do simulador,
    não mérito do modelo — o que se avalia é se ele LEVA a mão até lá e fecha.
    """

    CAM_CABECA = ("head_camera_antiga", 480, 848)
    CAM_PULSO = ("right_wrist_camera_antiga", 224, 224)
    RAIO_PEGA = 0.04
    ROTULO = "wla"            # como a fase aparece no painel

    CENAS = {
        # nome: (arquivo da cena, frase do treino)
        # Os três usam a MESMA cena; no `copo` a fruta e o prato ficam guardados
        # fora de vista. Assim a física e a pega são iguais nos três modos.
        "copo": ("scene_fruta.xml", "pick up the white cup"),
        "fruta": ("scene_fruta.xml", "Pick up the fruit and place it on the plate."),
        # xícara + fruta + prato na mesma mesa: a MESMA imagem serve às duas
        # frases, e o que se testa é se a rede escolhe o objeto pelo texto.
        "multi": ("scene_fruta.xml", "pick up the white cup"),
        # A cena do `sim_copo_mujoco` e do `cotreino_completo`: xícara na
        # metade direita, coador na esquerda, na posição fixa do `sorteia_alvo`.
        "coador": ("scene_fruta.xml", "place the white cup on the dripper"),
    }

    def __init__(self, pasta: Path, base_vlm: Path, lead: int = 12, hz: float = 30.0,
                 max_consultas: int = 40, copo_xy=None, cena: str = "copo",
                 fruta: str = "limao", fisica: bool = True, forca: str = "forte"):
        sys.path.insert(0, str(RAIZ / "pontes" / "unifolm-vla"))
        import roda_unifolm_mujoco as ponte

        # Cena e pose de partida idênticas às do roteirizado, que é quem gravou
        # o dataset: o primeiro quadro que o modelo vê tem que ser familiar.
        # A cena da fruta é a mesma, com prato e fruta a mais: a `Cena` do demo
        # lê o caminho do módulo na hora de carregar, então basta trocá-lo antes.
        import demo_pega_copo_mujoco as _demo
        self.modo = cena
        self.tarefa_treino = self.CENAS[cena][1]
        _demo.CENA = _demo.CENA.parent / self.CENAS[cena][0]
        base = MotorRoteirizado()
        self.demo, self.cena, self.rng = base.demo, base.cena, base.rng
        self.ponte = ponte
        self.cin = ponte.Cinematica()
        self._carrega_politica(pasta, base_vlm)
        self.lead = int(lead)
        self.hz = float(hz)
        self.max_consultas = max_consultas
        # Pedido de trocar a xícara de lugar (tecla ç na janela, botão no painel).
        # É só uma bandeira: quem mexe no MjData é o laço, nunca outra thread.
        self.pedido_copo = threading.Event()
        # None = sorteia na mesma caixa do dataset (x 0,45–0,48, y −0,06…−0,16).
        # Fora dela é fora da distribuição de treino — vale para testar, mas o
        # resultado diz tanto sobre isso quanto sobre a rede.
        self.copo_xy = copo_xy
        self.render = {
            nome: mujoco.Renderer(self.cena.m, h, w)
            for nome, h, w in (self.CAM_CABECA, self.CAM_PULSO)
        }
        m = self.cena.m
        adr = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
        self.adr_mao = {
            "esq": [adr(n[:-2]) for n in ponte.NOMES_MAO_ESQ],
            "dir": [adr(n[:-2]) for n in ponte.NOMES_MAO_DIR],
        }
        self.ultima_consulta_s = 0.0
        self.info = {}
        self.cabine = None
        self._prepara_objeto(fruta)
        self.fisica = fisica
        self.forca = forca
        self._prepara_fisica()
        self._abre_trabalhador()

    # ── o que muda de uma rede para outra (o MotorPI05 sobrescreve) ──────
    def _carrega_politica(self, pasta, base_vlm):
        from pontes.wla.inferencia_wla import PoliticaWLA
        t0 = time.time()
        print(f"⏳ carregando a WLA de {pasta} (VLM em {base_vlm})…")
        self.pol = PoliticaWLA(pasta, base_vlm)
        print(f"✅ WLA carregada em {time.time() - t0:.0f} s — {self.pol.carga}")

    def _consulta(self, obs):
        return self.pol.age(obs["imagens"], obs["texto"], obs["T_esq"], obs["T_dir"],
                            obs["garra_esq"], obs["garra_dir"], obs["cintura3"])

    @staticmethod
    def _tamanho(pedaco):
        return len(pedaco["pose_dir"])

    def _memoria_gpu(self):
        return round(self.pol.torch.cuda.memory_reserved() / 1e9, 2)

    # ── a thread que roda a rede: UMA para sempre ───────────────────────
    # Era criada a cada comando. Cada thread nova que toca em CUDA ganha as
    # suas áreas de trabalho (cuBLAS e afins) e elas NÃO voltam quando a
    # thread morre: a memória subia a cada tarefa mandada, e em 22/09 uma
    # sessão chegou a 81 GB — com o torch marcando 10,7 GB — até o navegador
    # travar por falta de memória na GB10, que é unificada.
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
                out = self._consulta(obs)
                self.respostas.put((passo_obs, out, time.perf_counter() - t0,
                                    obs["mao_dir"]))

        self.trab = threading.Thread(target=trabalhador, daemon=True, name="wla")
        self.trab.start()

    def _foto(self, nome):
        r = self.render[nome]
        r.update_scene(self.cena.d, camera=nome)
        return r.render()

    def _T(self, par):
        T = np.eye(4)
        T[:3, 3], T[:3, :3] = par
        return T

    def _observa(self, texto):
        """Tudo que a rede recebe, copiado AGORA. Roda na thread do laço, que é
        a dona do contexto de GL dos renderizadores."""
        cena, d, ponte = self.cena, self.cena.d, self.ponte
        q = d.qpos[cena.q_braco].copy()
        esq, dir_ = self.cin.fk(q)
        return {
            "imagens": {"head_left": self._foto(self.CAM_CABECA[0]).copy(),
                        "cam_wrist_right": self._foto(self.CAM_PULSO[0]).copy()},
            "texto": texto,
            "T_esq": self._T(esq), "T_dir": self._T(dir_),
            "garra_esq": ponte.dedos_para_garra(d.qpos[self.adr_mao["esq"]], "esq"),
            "garra_dir": ponte.dedos_para_garra(d.qpos[self.adr_mao["dir"]], "dir"),
            "cintura3": np.array([0.0, 0.0, float(d.qpos[cena.adr_cintura])], np.float32),
            "mao_dir": dir_[0],
        }

    # ── o objeto que a mão leva: a xícara, ou a fruta no modo fruta ──────
    RAIO_PRATO = 0.09
    LONGE = np.array([3.0, 3.0, 0.05])     # fora de qualquer câmera
    # Modo fruta: mais PERTO do robô que a xícara do dataset (x 0,45–0,48). Lá
    # o limite era o cotovelo raspar a mesa no roteirizado; aqui não há
    # roteirizado, e a 0,46 os dois ficavam fundos demais na mesa (22/09).
    # A mesa vai de x=0,08 a 0,88. Ajustável com --prato e --copo.
    X_FRUTA = (0.34, 0.40)
    Y_FRUTA = (0.10, 0.18)                 # módulo; a fruta nasce à direita (y<0)
    # No modo multi a xícara vai mais para trás e mais ao centro que a fruta,
    # senão as duas nascem no mesmo canto e encostam uma na outra.
    X_XICARA = (0.44, 0.50)
    Y_XICARA = (-0.04, 0.06)
    SEPARACAO = 0.18                       # distância mínima entre os objetos
    PRATO = (0.36, 0.12)

    def _prepara_objeto(self, fruta):
        """Monta a lista de objetos pegáveis da cena.

        Cada objeto: endereço no qpos/qvel, a altura em que ele repousa na mesa
        e o deslocamento da origem até o CENTRO (a xícara tem a origem no
        centro; as malhas do YCB, na base). `self.objetos[0]` é o principal —
        o que aparece no gráfico e o que o `--copo` posiciona.
        """
        cena, m = self.cena, self.cena.m
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        eid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, n)
        copo = {"nome": "xícara", "qpos": cena.copo_qpos, "dof": cena.copo_dof,
                "z0": cena.z_copo, "centro": 0.0,
                "geom": gid("colisao_copo"), "solda": eid("pega_copo")}
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"junta_{fruta}")
        if jid < 0:
            raise ValueError(f"fruta desconhecida: {fruta!r} (limao ou banana)")
        # A malha do YCB tem a base em z=0: pousada, a origem fica no tampo.
        f = {"nome": fruta, "qpos": m.jnt_qposadr[jid], "dof": m.jnt_dofadr[jid],
             "z0": cena.z_mesa, "centro": 0.027,
             "geom": gid(f"colisao_{fruta}"), "solda": eid(f"pega_{fruta}")}
        self.objetos = {"copo": [copo], "coador": [copo], "fruta": [f],
                        "multi": [f, copo]}[self.modo]
        em_uso = {o["qpos"] for o in self.objetos}
        self.guardados = [
            m.jnt_qposadr[j] for j in range(m.njnt)
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.jnt_qposadr[j] not in em_uso
            and m.jnt_qposadr[j] != 0                   # a base do robô
        ]
        self.segurando = None
        # Corpos da mão direita: é o contato com ELES que conta como pega.
        self.corpos_mao = {b for b in range(m.nbody) if m.body(b).name.startswith("right_hand_")}
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "prato")
        self.mocap_prato = int(m.body_mocapid[b])
        # A base da malha escaneada fica ABAIXO do zero dela: com o mocap no
        # tampo, o prato afundava 2,5 mm na mesa (medido de lado, 22/09). Mede
        # o vértice mais baixo com o prato na pose de uso e ergue por ele.
        d = cena.d
        d.mocap_pos[self.mocap_prato] = [*self.PRATO, cena.z_mesa]
        mujoco.mj_forward(m, d)
        g = gid("visual_prato")
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        z_min = float((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min())
        self.pos_prato = np.array([*self.PRATO, cena.z_mesa + (cena.z_mesa - z_min) + 0.0005])

    @property
    def obj_z0(self):
        return self.objetos[0]["z0"]

    def _obj_pos(self, o=None):
        o = o or self.objetos[0]
        return self.cena.d.qpos[o["qpos"]:o["qpos"] + 3].copy()

    def _centro(self, o):
        return self.cena.d.geom_xpos[o["geom"]].copy()

    def _poe_obj(self, pos, o=None):
        o = o or self.objetos[0]
        d = self.cena.d
        if self.segurando is o and self.fisica:
            self._larga()
        pos = np.asarray(pos, float).copy()
        pos[2] = max(pos[2], o["z0"])              # o tampo como piso
        d.qpos[o["qpos"]:o["qpos"] + 3] = pos
        d.qpos[o["qpos"] + 3:o["qpos"] + 7] = [1, 0, 0, 0]   # em pé
        d.qvel[o["dof"]:o["dof"] + 6] = 0

    def mundo_dos_objetos(self) -> dict:
        return {o["nome"]: [round(float(v), 3) for v in self._obj_pos(o)[:2]]
                for o in self.objetos}

    def poe_por_nome(self, nome: str, x: float, y: float) -> bool:
        for o in self.objetos:
            if o["nome"] == nome:
                self._poe_obj([x, y, o["z0"]], o)
                mujoco.mj_forward(self.cena.m, self.cena.d)
                return True
        return False

    def _sorteia_obj(self):
        """Fruta na caixa perto do robô; no `multi`, a xícara na faixa do
        dataset dela, sem encostar na fruta."""
        if self.modo in ("copo", "coador"):
            self.cena.sorteia_copo(self.rng)
            return
        fruta = self.objetos[0]
        x = self.rng.uniform(*self.X_FRUTA)
        y = -self.rng.uniform(*self.Y_FRUTA)
        self._poe_obj([x, y, fruta["z0"]], fruta)
        if self.modo == "multi":
            copo = self.objetos[1]
            for _ in range(80):
                c = [self.rng.uniform(*self.X_XICARA), self.rng.uniform(*self.Y_XICARA)]
                if np.hypot(c[0] - x, c[1] - y) > self.SEPARACAO:
                    break
            self._poe_obj([c[0], c[1], copo["z0"]], copo)

    def _arruma_cena(self):
        """Tira de cena o que não está em uso; prato só nos modos com fruta.
        O coador fica no modo copo (é a cena do dataset) e sai nos outros."""
        cena, d = self.cena, self.cena.d
        for k, adr in enumerate(self.guardados):
            d.qpos[adr:adr + 3] = self.LONGE + [0, 0.3 * (k + 1), 0]
        if self.modo in ("copo", "coador"):
            d.mocap_pos[self.mocap_prato] = self.LONGE + [0, -0.5, 0]
            if self.modo == "coador":
                self.pos_coador = cena.sorteia_alvo(self.rng)
                cena.desenha_alvo(self.pos_coador)
            return
        d.mocap_pos[cena.mocap_alvo] = self.LONGE + [0.5, 0, 0]      # o coador
        d.mocap_pos[self.mocap_prato] = self.pos_prato

    def _solta(self, o):
        """Garra abriu: o objeto cai no prato se estiver em cima dele."""
        p = self._obj_pos(o)
        if self.modo in ("fruta", "multi") and np.linalg.norm(p[:2] - self.pos_prato[:2]) < self.RAIO_PRATO:
            p[2] = self.pos_prato[2] + 0.012 + (o["z0"] - self.cena.z_mesa)
        else:
            p[2] = o["z0"]
        self._poe_obj(p, o)

    # ── física: PD por motor, base presa ─────────────────────────────────
    # Ganhos por grupo, (kp, kd). O torque sai com a compensação de gravidade
    # (`qfrc_bias`) somada, então o PD só corrige erro e não precisa ser duro.
    # Os tetos são os `ctrlrange` do MJCF: braço 25 Nm, punho 5–25, dedos
    # 1,4–2,4 Nm. Na mão o ganho é baixo de propósito: o dedo encosta no limão
    # e para, e a força de aperto é kp x (quanto falta para fechar).
    GANHOS = {"mao": (6.0, 0.15), "punho": (60.0, 2.0), "braco": (250.0, 8.0),
              "cintura": (400.0, 12.0), "perna": (300.0, 8.0)}
    # Com os torques REAIS (braço 25 Nm) o braço fica para trás do alvo: medido
    # em 22/09, erro médio de 1,66 cm e 9,4 cm de pico, e aí a rede recebe um
    # estado que ela nunca viu — o dataset foi gravado de forma CINEMÁTICA, com
    # o braço teleportando para o alvo da IK, erro zero por construção. A
    # resposta dela, nesse estado, é subir a mão e não fechar a garra.
    #
    # `forte` tira o teto de torque do BRAÇO (aplica a força direto, com ganho
    # alto) e mantém a física em tudo o mais: contato, colisão, dedos, objetos.
    # O braço vira "forte demais" para o robô real, e é uma dívida conhecida:
    # o certo é regravar o dataset com esta mesma física. `real` respeita os
    # limites do MJCF e serve para ver o tamanho dessa dívida.
    # No modo `forte` o ganho NÃO é fixo: é escalado pela inércia de cada junta,
    # kp = M·ωn² e kd = 2·M·ωn, o que dá amortecimento crítico igual em todas
    # elas. Ganho fixo alto (kp 2000) explodiu no passo de 2 ms — medido: erro
    # médio de 29 cm. Com ωn = 40 rad/s, ωn·dt = 0,08, que é estável.
    OMEGA = 40.0

    def _prepara_fisica(self):
        m = self.cena.m
        ganhos = self.GANHOS
        self.act_q = np.array([m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        self.act_v = np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        kp, kd = [], []
        for a in range(m.nu):
            n = m.joint(m.actuator_trnid[a, 0]).name
            k = ("mao" if "hand" in n else "cintura" if "waist" in n
                 else "perna" if any(t in n for t in ("hip", "knee", "ankle"))
                 else "punho" if "wrist" in n else "braco")
            kp.append(ganhos[k][0]); kd.append(ganhos[k][1])
        self.kp, self.kd = np.array(kp), np.array(kd)
        self.lim = m.actuator_ctrlrange.copy()
        self.sub = max(1, round((1.0 / self.hz) / m.opt.timestep))
        self._sincroniza_alvo()

    def _sincroniza_alvo(self):
        """Alvo = onde o robô está agora; base congelada aqui."""
        d = self.cena.d
        self.alvo = d.qpos[self.act_q].copy()
        self.base = d.qpos[0:7].copy()
        d.qvel[:] = 0
        mujoco.mj_forward(self.cena.m, d)

    def _fisica(self):
        m, d = self.cena.m, self.cena.d
        M = np.zeros((m.nv, m.nv)) if self.forca == "forte" else None
        for _ in range(self.sub):
            if self.forca == "forte":
                mujoco.mj_fullM(m, M, d.M)   # nesta versão do mujoco o nome é `d.M`
                inercia = np.maximum(M[self.act_v, self.act_v], 1e-4)
                kp, kd = inercia * self.OMEGA ** 2, 2 * inercia * self.OMEGA
            else:
                kp, kd = self.kp, self.kd
            tau = (kp * (self.alvo - d.qpos[self.act_q]) - kd * d.qvel[self.act_v]
                   + d.qfrc_bias[self.act_v])
            if self.forca == "forte":
                # Direto no grau de liberdade, sem o teto do atuador. O contato
                # continua sendo resolvido pela física: se o braço encostar na
                # mesa, ele para, só que com mais força disponível.
                d.qfrc_applied[self.act_v] = tau
            else:
                d.ctrl[:] = np.clip(tau, self.lim[:, 0], self.lim[:, 1])
            mujoco.mj_step(m, d)
            # Base presa, como a ElasticBand do simulador: sem isto o robô cai,
            # e as pernas não estão sob controle nenhum aqui.
            d.qpos[0:7] = self.base
            d.qvel[0:6] = 0
        mujoco.mj_forward(m, d)

    def _alvo_de_juntas(self, qpos_adr, valores):
        """Põe no vetor de alvos os valores das juntas dadas por endereço de qpos."""
        idx = {int(a): k for k, a in enumerate(self.act_q)}
        for a, v in zip(np.atleast_1d(qpos_adr), np.atleast_1d(valores)):
            self.alvo[idx[int(a)]] = v

    def _dedos_em(self, o):
        """Quantas partes DIFERENTES da mão direita encostam no objeto agora."""
        m, d = self.cena.m, self.cena.d
        partes = set()
        for i in range(d.ncon):
            c = d.contact[i]
            if o["geom"] in (c.geom1, c.geom2):
                outro = c.geom2 if c.geom1 == o["geom"] else c.geom1
                b = int(m.geom_bodyid[outro])
                if b in self.corpos_mao:
                    partes.add(b)
        return len(partes)

    def _atualiza_pega(self, g):
        """Solda quando a garra FECHOU e dois dedos encostam; larga quando abre.
        Quem decide fechar, e onde a mão está ao fechar, é a rede."""
        fechada = g < 0.5 * self.ponte.GARRA_ABERTA
        aberta = g > 0.8 * self.ponte.GARRA_ABERTA
        if self.segurando is None and fechada:
            for o in self.objetos:
                if self._dedos_em(o) >= 2:
                    self._solda(o)
                    break
        elif self.segurando is not None and aberta:
            self._larga()
        return self.segurando

    def _solda(self, o):
        """Liga a solda na pose relativa ATUAL: o objeto fica onde está na mão."""
        m, d = self.cena.m, self.cena.d
        e = o["solda"]
        b1, b2 = m.eq_obj1id[e], m.eq_obj2id[e]
        R1 = d.xmat[b1].reshape(3, 3)
        m.eq_data[e][3:6] = R1.T @ (d.xpos[b2] - d.xpos[b1])
        q1inv = np.zeros(4); mujoco.mju_negQuat(q1inv, d.xquat[b1])
        rel = np.zeros(4); mujoco.mju_mulQuat(rel, q1inv, d.xquat[b2])
        m.eq_data[e][6:10] = rel
        d.eq_active[e] = 1
        self.segurando = o

    def _larga(self):
        if self.segurando is not None:
            self.cena.d.eq_active[self.segurando["solda"]] = 0
        self.segurando = None

    def _preso(self):
        return self.segurando

    def _copo_pelvis(self):
        d = self.cena.d
        pos = self._obj_pos()
        return self.cena.para_pelvis(self._T((pos, np.eye(3))))[:3, 3]

    def executa(self, texto: str, n: int, ao_passo, dorme: float = 0.0) -> str:
        import queue

        cena, d, ponte = self.cena, self.cena.d, self.ponte
        # Cada comando repõe SÓ o robô. Os objetos ficam onde estão: quem os
        # muda de lugar é o `ç` (e a montagem inicial, no `main`). Antes cada
        # comando sorteava tudo de novo, e não dava para repetir uma frase na
        # mesma cena nem comparar duas frases na mesma cena (22/09).
        q = cena.reset_braco()
        if self.fisica:
            # O que estava na mão cai sozinho: é física.
            self._larga()
            self._sincroniza_alvo()
        else:
            # O que ficou no ar, preso na mão do comando anterior, cai no lugar:
            # no prato se estiver em cima dele, senão na mesa.
            for o in self.objetos:
                if self._obj_pos(o)[2] > o["z0"] + 0.01:
                    self._solta(o)
        mujoco.mj_forward(cena.m, d)

        pedidos, respostas = self.pedidos, self.respostas
        while not respostas.empty():        # sobras da tarefa anterior
            respostas.get_nowait()

        periodo = 1.0 / self.hz
        passo, pedaco, cursor, pulados = 0, None, 0, 0
        n_pedacos, em_voo, na_mao, parado = 0, False, None, 0
        tempos = []
        try:
            pedidos.put((passo, self._observa(texto)))
            em_voo = True
            proximo = time.perf_counter()
            while True:
                if self.pedido_copo.is_set():
                    self.pedido_copo.clear()
                    na_mao = None
                    self._sorteia_obj()
                    mujoco.mj_forward(cena.m, d)

                # ── chegou pedaço novo? (o primeiro espera; os outros não) ──
                try:
                    if pedaco is None:
                        r = respostas.get(timeout=30)
                    else:
                        r = respostas.get_nowait()
                    passo_obs, pedaco, dt, mao_obs = r
                    em_voo = False
                    n_pedacos += 1
                    tempos.append(dt)
                    # Alinhamento: o pedaço vale a partir do instante da
                    # observação; o que passou enquanto a rede pensava já era.
                    pulados = passo - passo_obs
                    cursor = min(pulados, self._tamanho(pedaco) - 1)
                    self._publica_consulta(n_pedacos, pedaco, mao_obs, texto, dt, pulados)
                except queue.Empty:
                    pass

                restante = self._tamanho(pedaco) - cursor
                if (not em_voo and restante <= self.lead and n_pedacos < self.max_consultas):
                    pedidos.put((passo, self._observa(texto)))
                    em_voo = True

                if restante > 0:
                    i = cursor
                    q, g, na_mao = self._aplica(pedaco, i, q, na_mao)
                    cursor += 1
                else:
                    parado += 1          # a rede atrasou mais que o lead
                if not em_voo and restante <= 0:
                    break                # acabou o orçamento de consultas

                ao_passo(f"{self.ROTULO} {n_pedacos}/{self.max_consultas}"
                         + (f" [{na_mao['nome']} na mão]" if na_mao else ""), q)
                passo += 1
                self.info["laco"] = {
                    "passo": passo, "pedaco": n_pedacos, "cursor_no_pedaco": cursor,
                    "restam_no_pedaco": max(restante - 1, 0), "lead": self.lead,
                    "rede_calculando": em_voo, "passos_parado_esperando_rede": parado,
                    "memoria_gpu_gb": self._memoria_gpu(),
                }
                proximo += periodo
                espera = proximo - time.perf_counter()
                if espera > 0:
                    time.sleep(espera)
                else:
                    proximo = time.perf_counter()   # atrasou: não tenta recuperar
        finally:
            # A thread NÃO morre aqui: ela é da vida do motor (ver
            # `_abre_trabalhador`). Só se espera o pedido em voo terminar, para
            # ele não cair no meio da próxima tarefa.
            if em_voo:
                try:
                    respostas.get(timeout=20)
                except queue.Empty:
                    pass

        linhas = []
        for o in self.objetos:
            p = self._centro(o)
            if na_mao is o:
                est = "NA MÃO" + (", levantada" if p[2] > o["z0"] + o["centro"] + 0.05 else "")
            elif self.modo == "coador" and np.linalg.norm(p[:2] - self.pos_coador[:2]) < 0.05:
                est = "NO COADOR"
            elif self.modo in ("fruta", "multi") and np.linalg.norm(p[:2] - self.pos_prato[:2]) < self.RAIO_PRATO:
                est = "NO PRATO"
            else:
                est = "na mesa"
            linhas.append(f"{o['nome']}: {est}")
        return (f"{n_pedacos} pedaços, rede {np.mean(tempos):.2f} s, parado {parado} passos; "
                + " · ".join(linhas))

    def _aplica(self, pedaco, i, q, na_mao):
        # `na_mao`: o dicionário do objeto preso, ou None.
        """Um passo previsto: IK, dedos, cintura, e a pega cinemática."""
        cena, d, ponte = self.cena, self.cena.d, self.ponte
        pe, pd = pedaco["pose_esq"][i], pedaco["pose_dir"][i]
        q = self.cin.resolve((pe[:3, 3], pe[:3, :3]), (pd[:3, 3], pd[:3, :3]), q)
        # O dataset guarda a cintura como [0, 0, yaw]: o yaw é o índice 2.
        yaw = float(np.clip(pedaco["cintura"][i][2], -cena.lim_cintura, cena.lim_cintura))
        g = float(pedaco["garra_dir"][i])
        dedos_d = ponte.garra_para_dedos(g, "dir")
        dedos_e = ponte.garra_para_dedos(float(pedaco["garra_esq"][i]), "esq")
        if self.fisica:
            # A rede e a IK dão o ALVO; quem move é o PD, e o limão só sobe se
            # os dedos o apertarem de fato.
            self._alvo_de_juntas(cena.q_braco, q)
            self._alvo_de_juntas(cena.adr_cintura, yaw)
            self._alvo_de_juntas(self.adr_mao["dir"], dedos_d)
            self._alvo_de_juntas(self.adr_mao["esq"], dedos_e)
            self._fisica()
            na_mao = self._atualiza_pega(g)
            chegou = self.cin.fk(d.qpos[cena.q_braco].copy())[1][0]
            self._info_acao(i, pd, chegou, g, pedaco)
            return q, g, na_mao
        d.qpos[cena.q_braco] = q
        d.qpos[cena.adr_cintura] = yaw
        d.qpos[self.adr_mao["dir"]] = dedos_d
        d.qpos[self.adr_mao["esq"]] = dedos_e
        mujoco.mj_forward(cena.m, d)
        na_mao = self._pega_cinematica(g, na_mao)
        chegou = self.cin.fk(q)[1][0]
        self._info_acao(i, pd, chegou, g, pedaco)
        return q, g, na_mao

    def _pega_cinematica(self, g, na_mao):
        """Sem física: o objeto gruda na pinça se ela FECHA perto dele, e cai
        quando ela abre. Devolve o objeto preso (ou None)."""
        cena, d, ponte = self.cena, self.cena.d, self.ponte
        fechada = g < 0.5 * ponte.GARRA_ABERTA
        # `na_mao` é o OBJETO preso (ou None). A pinça pega o que estiver mais
        # perto dela quando a rede fecha — a escolha de qual é da rede.
        pinca = cena.ponto_pega()
        if na_mao is None and fechada:
            dist = [(np.linalg.norm(pinca - self._centro(o)), k) for k, o in enumerate(self.objetos)]
            dmin, k = min(dist)
            if dmin < self.RAIO_PEGA:
                na_mao = self.objetos[k]
        elif na_mao is not None and g > 0.8 * ponte.GARRA_ABERTA:
            self._solta(na_mao)
            na_mao = None
            mujoco.mj_forward(cena.m, d)
        if na_mao is not None:
            self._poe_obj(pinca - [0, 0, na_mao["centro"]], na_mao)
            mujoco.mj_forward(cena.m, d)
        return na_mao

    def _info_acao(self, i, pd, chegou, g, pedaco):
        # Quanto a mão ficou longe do que a rede pediu. Sem física é só a IK;
        # com física soma o atraso do PD e o que a colisão impediu (mesa,
        # objeto na mão). Se cresce muito, o que se vê NÃO é mais a rede.
        self.info["acao"] = {
            "passo_do_pedaco": i,
            "alvo_mao_dir": [round(float(v), 4) for v in pd[:3, 3]],
            "mao_dir_chegou": [round(float(v), 4) for v in chegou],
            "erro_mao_cm": round(float(np.linalg.norm(chegou - pd[:3, 3])) * 100, 2),
            "garra_dir": round(g, 3),
            "cintura_yaw": round(float(pedaco["cintura"][i][2]), 4),
            "fisica": self.fisica,
        }

    def _publica_consulta(self, k, out, mao_obs, texto, dt, pulados):
        """A predição inteira no JSON (o navegador desenha) e a atenção."""
        P = out["pose_dir"][:, :3, 3]
        self.info["predicao"] = {
            "pedaco": k,
            "texto_enviado": texto,
            "tempo_s": round(dt, 3),
            "pulados_por_atraso": int(pulados),
            "mao_dir_na_observacao": [round(float(v), 4) for v in mao_obs],
            "copo_pelvis": [round(float(v), 4) for v in self._copo_pelvis()],
            "objetos_pelvis": {
                o["nome"]: [round(float(v), 4) for v in
                            self.cena.para_pelvis(self._T((self._centro(o), np.eye(3))))[:3, 3]]
                for o in self.objetos},
            "mao_dir_xyz": [[round(float(v), 4) for v in p] for p in P],
            "garra_dir": [round(float(v), 3) for v in out["garra_dir"]],
            "cintura_yaw": [round(float(v), 4) for v in out["cintura"][:, 2]],
        }
        if self.cabine is not None:
            for papel, img in out.get("atencao", {}).items():
                self.cabine.publica_quadro(f"atencao_{papel}", img)


class MotorPI05(MotorWLA):
    """O π0.5 do `pega_copo_sem_prof_2026-09-18` na mesma cena, painel e laço.

    Herda do `MotorWLA` tudo menos a rede: cena, objetos, `ç`, pega cinemática,
    o laço de pedaço/antecedência e a thread única da rede. Muda o que entra e
    o que sai.

    ── Entra e sai JUNTA, não pose de mão ──────────────────────────────────
    O dataset foi gravado pelo `gerar_dataset_mujoco.py`, que guarda o MESMO
    vetor de 29 como estado e como ação (cinemática pura: a junta comandada é a
    atingida). Ordem do `NOMES_29`: 0..13 braços na ordem do `JUNTAS_BRACO`,
    14 yaw da cintura, 15..21 mão esquerda (sempre zero no dataset), 22..28 mão
    direita na ordem de `cena.mao`. A ação vai direto para o `qpos`, sem IK.

    ── Câmeras ─────────────────────────────────────────────────────────────
    As mesmas `_antiga` do `MotorWLA`, nas mesmas resoluções: são as que
    gravaram este dataset (cabeça 480x848, pulso 224x224).

    Frase do treino: "pick up the white cup" — a ÚNICA do dataset. Qualquer
    outra frase é fora da distribuição.
    """

    ROTULO = "pi05"

    def _carrega_politica(self, pasta, base_vlm):
        import torch
        import init_lerobot_inference_v3 as v3   # os carregadores do cliente do robô

        self.torch = torch
        t0 = time.time()
        self.pol, tipo = v3.load_policy(str(pasta), torch.device("cuda"))
        self.pre, self.pos = v3.load_pre_post_processors(str(pasta), self.pol)
        self.chaves_img = list(self.pol.config.image_features)
        print(f"✅ π0.5 ({tipo}) carregado em {time.time() - t0:.0f} s — "
              f"câmeras {self.chaves_img}, pedaço {self.pol.config.chunk_size}")

    def _memoria_gpu(self):
        return round(self.torch.cuda.memory_reserved() / 1e9, 2)

    @staticmethod
    def _tamanho(pedaco):
        return len(pedaco["acoes"])

    def _estado29(self):
        cena, d = self.cena, self.cena.d
        e = np.zeros(29, np.float32)
        e[:14] = d.qpos[cena.q_braco]
        e[14] = d.qpos[cena.adr_cintura]
        for k, j in enumerate(cena.mao[:7]):
            e[22 + k] = d.qpos[j["adr"]]
        return e

    def _observa(self, texto):
        torch = self.torch
        fotos = {"head_camera": self._foto(self.CAM_CABECA[0]).copy(),
                 "right_wrist_camera": self._foto(self.CAM_PULSO[0]).copy()}
        bruto = {"observation.state": torch.from_numpy(self._estado29()), "task": texto}
        for chave in self.chaves_img:
            nome = chave.rsplit(".", 1)[-1]
            if nome in fotos:
                bruto[chave] = torch.from_numpy(fotos[nome]).permute(2, 0, 1).float().div(255.0)
        return {"bruto": bruto,
                "mao_dir": self.cin.fk(self.cena.d.qpos[self.cena.q_braco].copy())[1][0]}

    def _consulta(self, obs):
        torch = self.torch
        lote = self.pre(obs["bruto"])
        lote.pop("action", None)
        lote = {k: v for k, v in lote.items() if isinstance(v, torch.Tensor)}
        with torch.inference_mode():
            norm = self.pol.predict_action_chunk(lote)          # (1, 50, 29)
            acoes = self.pos(norm)
        if isinstance(acoes, dict):
            acoes = acoes["action"]
        return {"acoes": acoes[0].float().cpu().numpy()}

    def _aplica(self, pedaco, i, q, na_mao):
        cena, d, ponte = self.cena, self.cena.d, self.ponte
        a = pedaco["acoes"][i]
        q = a[:14].astype(float)
        d.qpos[cena.q_braco] = q
        d.qpos[cena.adr_cintura] = float(np.clip(a[14], -cena.lim_cintura, cena.lim_cintura))
        for k, j in enumerate(cena.mao[:7]):
            d.qpos[j["adr"]] = a[22 + k]
        mujoco.mj_forward(cena.m, d)
        g = float(ponte.dedos_para_garra(d.qpos[self.adr_mao["dir"]], "dir"))
        na_mao = self._pega_cinematica(g, na_mao)
        self.info["acao"] = {
            "passo_do_pedaco": i,
            "juntas_braco_dir": [round(float(v), 4) for v in q[7:14]],
            "garra_dir": round(g, 3),
            "cintura_yaw": round(float(a[14]), 4),
            "fisica": False,
        }
        return q, g, na_mao

    def _publica_consulta(self, k, out, mao_obs, texto, dt, pulados):
        """Mesmo JSON do `MotorWLA`: a trajetória da mão vem da FK das juntas
        previstas, para o gráfico do painel funcionar igual."""
        A = out["acoes"]
        P = [self.cin.fk(a[:14].astype(float))[1][0] for a in A]
        garra = [float(self.ponte.dedos_para_garra(a[22:29], "dir")) for a in A]
        self.info["predicao"] = {
            "pedaco": k,
            "texto_enviado": texto,
            "tempo_s": round(dt, 3),
            "pulados_por_atraso": int(pulados),
            "mao_dir_na_observacao": [round(float(v), 4) for v in mao_obs],
            "copo_pelvis": [round(float(v), 4) for v in self._copo_pelvis()],
            "objetos_pelvis": {
                o["nome"]: [round(float(v), 4) for v in
                            self.cena.para_pelvis(self._T((self._centro(o), np.eye(3))))[:3, 3]]
                for o in self.objetos},
            "mao_dir_xyz": [[round(float(v), 4) for v in p] for p in P],
            "garra_dir": [round(v, 3) for v in garra],
            "cintura_yaw": [round(float(a[14]), 4) for a in A],
        }


class Cerebro:
    def __init__(self, cabine: Cabine, motor, passos_por_quadro: int = 3,
                 dorme: float = 0.0, janela: bool = False, janela_hz: float = 10.0):
        self.cabine = cabine
        self.motor = motor
        # Segundos de sono por passo. Zero = o mais rápido que a CPU deixa, e um
        # episódio inteiro passa em ~4 s — ótimo para testar, ruim para gravar
        # vídeo, porque o movimento fica picotado. Para o vídeo da super IA, use
        # algo como 0.004 (o passo do MuJoCo), que dá tempo quase real.
        self.dorme = float(dorme)
        self.janela_periodo = 1.0 / max(1e-3, float(janela_hz))
        self.passos_por_quadro = passos_por_quadro
        # O motor do IsaacLab não tem cena de MuJoCo: as câmeras vêm por ZMQ do
        # simulador e ele mesmo as publica no painel.
        self.cena = getattr(motor, "cena", None)
        self.renderizadores = {} if self.cena is None else {
            nome: mujoco.Renderer(self.cena.m, alt, larg)
            for nome, (alt, larg) in CAMERAS.items()
            if mujoco.mj_name2id(self.cena.m, mujoco.mjtObj.mjOBJ_CAMERA, nome) >= 0
        }
        # O motor da WLA publica imagens próprias (atenção, gráfico da predição)
        # e campos próprios no estado; os outros motores ignoram isto.
        motor.cabine = cabine
        # Visualizador do MuJoCo, passivo: ele desenha na thread dele e só lê o
        # `MjData`; quem mexe no modelo continua sendo o laço.
        #
        # O `sync()` VAZA MEMÓRIA nesta versão do mujoco: medido em 22/09, só a
        # janela aberta e sincronizando a 30 Hz, sem rede e sem física, a
        # máquina subiu ~1 GB por minuto (e o `close()` terminou em segmentation
        # fault). Com a rede junto, uma sessão chegou a 112 GB e travou o
        # navegador — na GB10 a memória é unificada, então quem paga é o sistema
        # inteiro. Por isso a janela é OPCIONAL e sincroniza devagar; o painel do
        # navegador mostra as mesmas câmeras e não vaza nada.
        self.viewer = None
        self._ultimo_sync = 0.0
        if janela and self.cena is None:
            print("   (sem janela: este motor não tem cena de MuJoCo para desenhar)")
            janela = False
        if janela:
            import mujoco.viewer as visor  # `import mujoco.viewer` puro tornaria `mujoco` local aqui
            # Sem os painéis laterais (ajuda, juntas, opções de render): só o mundo.
            # Ainda dá para reabri-los com Tab (esquerdo) e Shift+Tab (direito).
            self.viewer = visor.launch_passive(self.cena.m, self.cena.d,
                                               show_left_ui=False, show_right_ui=False,
                                               key_callback=self._tecla)
        self._parar = threading.Event()
        self._contador = 0
        self._fase = "parado"
        self._ultimo_resultado = ""

    # ── publicação ──────────────────────────────────────────────────────
    def _publica(self, fase: str, q) -> None:
        self._contador += 1
        if self._contador % self.passos_por_quadro:
            return
        for nome, r in self.renderizadores.items():
            try:
                r.update_scene(self.cena.d, camera=nome)
                self.cabine.publica_quadro(nome, r.render())
            except Exception:
                pass          # um render que falha não pode derrubar o laço
        self.cabine.publica_estado({
            "fase": fase,
            "passo": self._contador,
            "juntas_braco": [round(float(v), 4) for v in np.atleast_1d(q)],
            **({"copo": [round(float(v), 3) for v in self.motor._obj_pos()]}
               if hasattr(self.motor, "_obj_pos") else {}),
            "motor": type(self.motor).__name__,
            "ultimo_resultado": self._ultimo_resultado,
            **getattr(self.motor, "info", {}),
            "objetos_mundo": (self.motor.mundo_dos_objetos()
                              if hasattr(self.motor, "mundo_dos_objetos") else {}),
        })

    def _tecla(self, codigo):
        """Ç no ABNT2 chega como 59 (o ; do layout americano); 186 em alguns
        sistemas. Mesma convenção do `unitree-g1-mujoco/sim/base_sim.py`. Roda
        na thread do visualizador: só levanta a bandeira."""
        if codigo in (59, 186):
            self._pede_copo()

    def _pede_copo(self):
        if hasattr(self.motor, "pedido_copo"):
            self.motor.pedido_copo.set()

    def _atende_copo_ocioso(self):
        """Parado, ninguém consome a bandeira do motor: troca aqui mesmo."""
        ev = getattr(self.motor, "pedido_copo", None)
        if self.cena is None:
            return
        if ev is not None and ev.is_set():
            ev.clear()
            if hasattr(self.motor, "_sorteia_obj"):
                self.motor._arruma_cena()
                self.motor._sorteia_obj()
            else:
                self.cena.sorteia_copo(self.motor.rng)
            mujoco.mj_forward(self.cena.m, self.cena.d)

    def _sincroniza_janela(self):
        if self.viewer is None or not self.viewer.is_running():
            return
        agora = time.perf_counter()
        if agora - self._ultimo_sync < self.janela_periodo:
            return
        self._ultimo_sync = agora
        self.viewer.sync()

    def _atende_pedidos(self):
        if self.cabine.consome_pedido_copo():
            self._pede_copo()
        for nome, x, y in self.cabine.consome_objetos():
            if hasattr(self.motor, "poe_por_nome"):
                self.motor.poe_por_nome(nome, x, y)

    def _ao_passo(self, fase, q):
        self._fase = fase
        self._atende_pedidos()
        self._sincroniza_janela()
        self._publica(fase, q)
        if self._parar.is_set() or self.cabine.parada_pedida():
            # `episodio` não tem cancelamento; levantar é o jeito honesto de
            # sair no meio sem deixar o MuJoCo num estado pela metade.
            raise KeyboardInterrupt("parada pedida pela cabine")

    # ── laço principal ──────────────────────────────────────────────────
    def roda(self) -> None:
        visto = -1
        n = 0
        self._publica("ocioso", np.zeros(14))
        while True:
            texto, seq = self.cabine.tarefa()
            if seq == visto or not texto:
                self._atende_pedidos()
                self._atende_copo_ocioso()
                self._publica("ocioso", np.zeros(14) if self.cena is None
                              else self.cena.d.qpos[self.cena.q_braco])
                self._sincroniza_janela()
                time.sleep(0.1)
                continue
            visto = seq
            n += 1
            self._parar.clear()
            self._ultimo_resultado = f"executando: {texto}"
            try:
                self._ultimo_resultado = self.motor.executa(
                    texto, n, self._ao_passo, dorme=self.dorme)
            except KeyboardInterrupt as e:
                self._ultimo_resultado = f"interrompido: {e}"
            except Exception as e:
                self._ultimo_resultado = f"falhou: {e}"
                traceback.print_exc()
            print(f"[cabine] {texto!r} → {self._ultimo_resultado}")

    def para(self) -> None:
        self._parar.set()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--porta", type=int, default=8090)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--motor", choices=("roteirizado", "wla", "pi05", "isaac"), default="roteirizado",
                   help="isaac: fecha o laço com o simulador da Unitree (IsaacLab) por DDS e "
                        "ZMQ, sem renderizar nada na tela. Suba o `sobe_sim_dex3.sh` antes.")
    p.add_argument("--modelo", type=Path, default=None,
                   help="WLA: pasta do treino (config.yaml + final_model/), padrão "
                        "~/modelos/wla_prometheus_copo. pi05: a `pretrained_model` do "
                        "checkpoint, padrão ~/ckpts_nossos/pi05_pega_copo_sim")
    p.add_argument("--base-vlm", type=Path, default=Path.home() / "modelos/UnifoLM-ER-1-Qwen3-VL",
                   help="o VLM base; o NOME da pasta escolhe a classe, não renomeie")
    p.add_argument("--tarefa", default="", help="já começa executando esta frase")
    p.add_argument("--copo", default=None, metavar="X,Y",
                   help="posição FIXA do objeto (xícara, ou a fruta no modo fruta), em "
                        "metros no mundo (ex.: 0.40,-0.10). Sem isto, sorteia.")
    p.add_argument("--consultas", type=int, default=40,
                   help="quantos pedaços de 30 passos a rede produz por tarefa")
    p.add_argument("--lead", type=int, default=12,
                   help="com quantos passos de antecedência pedir o próximo pedaço. "
                        "Tem que cobrir o tempo da rede: 0,22 s a 30 Hz = 7 passos")
    p.add_argument("--hz", type=float, default=30.0,
                   help="ritmo do braço; 30 = o fps do dataset, tempo real")
    p.add_argument("--cena", choices=("copo", "fruta", "multi", "coador"), default="copo",
                   help="copo: a do `nosso_sim_pega`. fruta: a mesma mesa com prato e fruta "
                        "do YCB, para a tarefa da Unitree (10%% do treino). multi: xícara, "
                        "fruta e prato juntos, para ver se a rede escolhe pelo texto. "
                        "coador: xícara e coador, a cena do `place the white cup on the dripper`")
    p.add_argument("--fruta", choices=("limao", "banana"), default="limao")
    p.add_argument("--forca", choices=("forte", "real"), default="forte",
                   help="forte: o braço segue o alvo de perto (o dataset é cinemático). "
                        "real: respeita os torques do MJCF, e o braço fica para trás")
    p.add_argument("--sem-fisica", action="store_true",
                   help="volta à pega cinemática (o objeto gruda na pinça quando a "
                        "garra fecha perto dele). Padrão: física de verdade")
    p.add_argument("--prato", default=None, metavar="X,Y",
                   help="posição do prato no modo fruta (padrão 0.36,0.12)")
    p.add_argument("--janela", action="store_true",
                   help="abre também o visualizador do MuJoCo (precisa de tela). ATENÇÃO: "
                        "o sync() dele vaza memória nesta versão do mujoco, ~1 GB por minuto "
                        "a 30 Hz. O painel do navegador mostra o mesmo sem vazar.")
    p.add_argument("--janela-hz", type=float, default=10.0,
                   help="quadros por segundo da janela do MuJoCo; menos = menos vazamento")
    p.add_argument("--dorme", type=float, default=0.0, metavar="SEGUNDOS",
                   help="sono por passo; 0.004 dá movimento quase em tempo real, "
                        "que é o que o vídeo pede. O padrão 0 corre solto.")
    args = p.parse_args()

    if args.modelo is None:
        args.modelo = Path.home() / ("ckpts_nossos/pi05_pega_copo_sim" if args.motor == "pi05"
                                     else "modelos/wla_prometheus_copo")
    print(f"⏳ montando o motor {args.motor}…")
    if args.motor == "isaac":
        from pontes.cabine.motor_isaac import MotorIsaac
        motor = MotorIsaac(args.modelo, args.base_vlm, lead=args.lead, hz=args.hz,
                           max_consultas=args.consultas)
        print(f"   IsaacLab: esperando simulador… {'OK' if motor.espera_simulador() else 'NÃO respondeu'}")
    elif args.motor in ("wla", "pi05"):
        copo = tuple(float(v) for v in args.copo.split(",")) if args.copo else None
        Motor = MotorPI05 if args.motor == "pi05" else MotorWLA
        if args.motor == "pi05" and not args.sem_fisica:
            # O π0.5 escreve junta direto no qpos; a física com PD não existe
            # para ele (e o dataset é cinemático).
            print("   π0.5: física desligada (a ação vai direto para as juntas)")
            args.sem_fisica = True
        motor = Motor(args.modelo, args.base_vlm, lead=args.lead, hz=args.hz,
                         max_consultas=args.consultas, copo_xy=copo,
                         cena=args.cena, fruta=args.fruta, fisica=not args.sem_fisica, forca=args.forca)
        if args.motor == "pi05" and args.cena != "coador":
            motor.tarefa_treino = "pick up the white cup"
        if args.prato:
            motor.pos_prato[:2] = [float(v) for v in args.prato.split(",")]
        # Parado, o motor já mostra a cena montada, com prato e fruta no lugar.
        motor._arruma_cena()
        if copo is None:
            motor._sorteia_obj()
        else:
            motor._poe_obj([copo[0], copo[1], motor.obj_z0])
        mujoco.mj_forward(motor.cena.m, motor.cena.d)
        motor._sincroniza_alvo()
        print(f"   cena {args.cena}: a frase do treino é {motor.tarefa_treino!r}"
              f" | física: {'SIM' if motor.fisica else 'não'}")
    else:
        motor = MotorRoteirizado()

    cabine = Cabine()
    cerebro = Cerebro(cabine, motor, dorme=args.dorme, janela=args.janela,
                      janela_hz=args.janela_hz)
    sobe(cabine, args.porta, args.host)
    print(f"✅ cabine no ar: http://<ip>:{args.porta}/")
    print(f"   câmeras: {', '.join(cerebro.renderizadores)}")
    if args.tarefa:
        cabine.define_tarefa(args.tarefa)
    try:
        cerebro.roda()
    except KeyboardInterrupt:
        print("\nsaindo.")


if __name__ == "__main__":
    main()
