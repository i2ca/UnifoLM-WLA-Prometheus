#!/usr/bin/env python
"""
Ajuste do SUPORTE e das CÂMERAS do sim do café pelo TECLADO, vendo ao vivo o que a IA vai receber
=================================================================================================
Abre o viewer oficial do MuJoCo (vista 3D, robô parado na pose de partida do dataset) e uma janela
"cameras" com a ZED da cabeça e as duas D435 dos punhos, do jeito que o sim_cafe_wla.py as envia.
Clique na janela "cameras" e use as teclas; com G grava no g1_29dof_with_hand_wla.xml (pos + euler, em
radianos — o mesmo formato que você edita à mão).

  1 peça do suporte ESQ.   2 câmera ESQ.   3 conjunto ESQ. (suporte + câmera juntos)
  4 peça do suporte DIR.   5 câmera DIR.   6 conjunto DIR.
  7 ZED da cabeça
  W / S  x (para os dedos / para trás)    A / D  y    Q / E  z (sobe / desce)
  I / K  muda o 1º número do euler   J / L  o 2º   U / O  o 3º
         (exatamente o que você mudaria à mão no XML: euler="x y z", em radianos)
  + / -  passo maior / menor (mm e graus)
  G grava no XML    R recarrega do XML (desfaz o que não foi gravado)    ESC sai

    cd ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla
    conda run --no-capture-output -n prometheus-vla python ajusta_cameras.py
"""
import re
import sys
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import sim_cafe_wla as sim  # noqa: E402

ROBO_XML = sim.CENA.parent / "g1_29dof_with_hand_wla.xml"
# tecla: (rótulo, "body"/"geom", nome no XML, a câmera gira a imagem na lente com U/O?)
ALVOS = {ord("1"): ("peça do suporte ESQ.", "geom", "suporte_punho_esquerdo", False),
         ord("2"): ("câmera ESQ. (D435)", "body", "d435_punho_esquerdo", True),
         ord("3"): ("conjunto ESQ. (suporte + câmera)", "body", "suporte_punho_esquerdo", False),
         ord("4"): ("peça do suporte DIR.", "geom", "suporte_punho_direito", False),
         ord("5"): ("câmera DIR. (D435)", "body", "d435_punho_direito", True),
         ord("6"): ("conjunto DIR. (suporte + câmera)", "body", "suporte_punho_direito", False),
         ord("7"): ("ZED da cabeça", "body", "zed_cabeca", True)}
ANDA = {ord("w"): (0, 1), ord("s"): (0, -1), ord("a"): (1, 1), ord("d"): (1, -1), ord("q"): (2, 1), ord("e"): (2, -1)}
GIRA = {ord("i"): (0, 1), ord("k"): (0, -1), ord("j"): (1, 1), ord("l"): (1, -1), ord("u"): (2, 1), ord("o"): (2, -1)}


def q_de_euler(e):
    """euler do MuJoCo (eulerseq "xyz" = intrínseco, radianos) -> quat wxyz."""
    x, y, z, w = Rotation.from_euler("XYZ", e).as_quat()
    return np.array([w, x, y, z])


def euler_de_q(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("XYZ")


def mul(a, b):
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, a, b)
    return q


def gira_q(eixo, ang):
    q = np.zeros(4)
    v = np.zeros(3)
    v[eixo] = 1
    mujoco.mju_axisAngle2Quat(q, v, ang)
    return q


def fmt(v, casas=4):
    s = []
    for x in v:
        t = f"{x:.{casas}f}".rstrip("0").rstrip(".")
        s.append("0" if t in ("-0", "") else t)
    return " ".join(s)


def tag(texto, tipo, nome):
    return re.search(rf'<{tipo} name="{nome}"[^>]*>', texto)


def attr(t, nome):
    m = re.search(rf'\s{nome}="([^"]*)"', t)
    return np.array([float(x) for x in m.group(1).split()]) if m else None


class Estado:
    """Valores do XML (pos + quat) de cada alvo e como aplicá-los no modelo compilado sem recompilar."""

    def __init__(self, m):
        self.m = m
        self.xml, self.off, self.euler = {}, {}, {}
        txt = ROBO_XML.read_text()
        for _, tipo, nome, _ in ALVOS.values():
            if tipo == "body":
                b = m.body(nome)
                self.xml[(tipo, nome)] = [b.pos.copy(), b.quat.copy()]
                e = attr(tag(txt, "body", nome).group(0), "euler")
                if e is not None:
                    self.euler[(tipo, nome)] = e
            else:
                t = tag(txt, "geom", nome).group(0)
                p = attr(t, "pos")
                e = attr(t, "euler")
                p = np.zeros(3) if p is None else p
                q = q_de_euler(np.zeros(3) if e is None else e)
                g = m.geom(nome)
                # o MuJoCo recentra a malha no centro de massa: geom = (pos, quat do XML) ∘ deslocamento da malha
                qi = q.copy()
                qi[1:] *= -1
                off_q = mul(qi, g.quat)
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, q)
                off_p = R.reshape(3, 3).T @ (g.pos - p)
                self.off[nome] = (off_p, off_q)
                self.xml[(tipo, nome)] = [p, q]
                if e is not None:
                    self.euler[(tipo, nome)] = e

    def aplica(self):
        for (tipo, nome), (p, q) in self.xml.items():
            if tipo == "body":
                self.m.body(nome).pos[:] = p
                self.m.body(nome).quat[:] = q
            else:
                off_p, off_q = self.off[nome]
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, q)
                self.m.geom(nome).pos[:] = p + R.reshape(3, 3) @ off_p
                self.m.geom(nome).quat[:] = mul(q, off_q)

    def grava(self):
        txt = ROBO_XML.read_text()
        for (tipo, nome), (p, q) in self.xml.items():
            mt = tag(txt, tipo, nome)
            t = mt.group(0)
            novo = re.sub(r'\s(?:zaxis|quat|euler|xyaxes|axisangle)="[^"]*"', "", t)
            e = f'euler="{fmt(self.euler.get((tipo, nome), euler_de_q(q)))}"'
            p_ = f'pos="{fmt(p)}"'
            if re.search(r'\spos="', novo):
                novo = re.sub(r'pos="[^"]*"', f"{p_} {e}", novo, count=1)
            else:
                novo = novo.replace(f'name="{nome}"', f'name="{nome}" {p_} {e}', 1)
            txt = txt[:mt.start()] + novo + txt[mt.end():]
        ROBO_XML.write_text(txt)
        print(f"💾 gravado em {ROBO_XML}", flush=True)


def carrega():
    a = type("A", (), {"maca": [0.40, -0.10], "x": [0.33, 0.10]})()
    m = sim.monta_cena(a)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("maca").id)
    mujoco.mj_forward(m, d)
    return m, d


def painel(cams, d, est, sel, passo_mm, passo_grau):
    imgs = cams.renderiza(d)
    topo = np.hstack([imgs["wrist_left"], imgs["wrist_right"]])
    zed = imgs["head_stereo_left"].copy()
    rot, tipo, nome, _ = ALVOS[sel]
    p, q = est.xml[(tipo, nome)]
    eu = est.euler.get((tipo, nome), euler_de_q(q))
    info = np.zeros_like(zed)
    linhas = [f"AJUSTANDO: {rot}",
              f"  <{tipo} name=\"{nome}\">",
              f"  pos   = {fmt(p, 3)}",
              f"  euler = {fmt(eu, 3)}  (rad)",
              f"  passo: {passo_mm:.0f} mm / {passo_grau:.0f} graus  (+/-)",
              "",
              "1 peca ESQ  2 camera ESQ  3 conjunto ESQ",
              "4 peca DIR  5 camera DIR  6 conjunto DIR  7 ZED",
              "W/S x   A/D y   Q/E z",
              "I/K euler 1o   J/L euler 2o   U/O euler 3o",
              "G grava no XML   R recarrega   ESC sai"]
    for i, s in enumerate(linhas):
        cv2.putText(info, s, (12, 32 + 36 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
    for img, nome_img in ((topo[:, :640], "D435 esquerda"), (topo[:, 640:], "D435 direita"),
                          (zed, "ZED (o que a IA ve)")):
        cv2.putText(img, nome_img, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([topo, np.hstack([zed, info])])


def main():
    m, d = carrega()
    est = Estado(m)
    cams = sim.Cameras(m)
    sel, passo_mm, passo_grau = ord("1"), 5.0, 5.0
    viewer = mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False)
    cv2.namedWindow("cameras", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("cameras", 1280, 960)
    print(__doc__.split("\n\n")[1], flush=True)
    while viewer.is_running():
        cv2.imshow("cameras", painel(cams, d, est, sel, passo_mm, passo_grau))
        k = cv2.waitKey(30) & 0xFF
        if k == 255:
            viewer.sync()
            continue
        if k == 27:
            break
        k = ord(chr(k).lower()) if 65 <= k <= 90 else k
        _, tipo, nome, _ = ALVOS[sel]
        with viewer.lock():
            if k in ALVOS:
                sel = k
                print(f"→ {ALVOS[sel][0]}", flush=True)
            elif k in (ord("+"), ord("=")):
                passo_mm, passo_grau = min(passo_mm * 2, 40), min(passo_grau * 2, 45)
            elif k in (ord("-"), ord("_")):
                passo_mm, passo_grau = max(passo_mm / 2, 1), max(passo_grau / 2, 1)
            elif k in ANDA:
                eixo, s = ANDA[k]
                est.xml[(tipo, nome)][0][eixo] += s * passo_mm / 1000
            elif k in GIRA:
                eixo, s = GIRA[k]
                # muda UM número do euler, como na edição à mão
                e = est.euler.get((tipo, nome))
                if e is None:
                    e = euler_de_q(est.xml[(tipo, nome)][1])
                e = e.copy()
                e[eixo] += np.radians(s * passo_grau)
                est.euler[(tipo, nome)] = e
                est.xml[(tipo, nome)][1] = q_de_euler(e)
            elif k == ord("g"):
                est.grava()
            elif k == ord("r"):
                m2, _ = carrega()
                novo = Estado(m2)
                est.xml, est.off, est.euler = novo.xml, novo.off, novo.euler
                print("↺ recarregado do XML", flush=True)
            est.aplica()
            mujoco.mj_forward(m, d)
        viewer.sync()
    viewer.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
