"""FK do G1 em numpy puro (sem pinocchio), lendo o URDF g1_body29_hand14.

Mesma convenção da `ponte_wla_isaac.py` (validada nos dados do WLA): pelvis fixa na origem,
ponto da garra = *_wrist_yaw_link + T_GARRA no eixo do punho, rotação = a do wrist_yaw_link.
Devolve xyz + rot6d (as 2 primeiras colunas de R), o estado que o servidor do WLA espera.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF = Path(__file__).resolve().parent.parent / "assets/g1/g1_body29_hand14.urdf"
T_GARRA = {"left": np.array([0.1081, 0.0037, 0.0]), "right": np.array([0.1097, -0.0010, 0.0])}
NOMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _eixo(u, a):
    u = u / np.linalg.norm(u)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


class FK:
    def __init__(self, urdf=URDF):
        raiz = ET.parse(urdf).getroot()
        self.junta_do_filho = {}
        for j in raiz.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            lim = j.find("limit")
            self.junta_do_filho[j.find("child").get("link")] = {
                "nome": j.get("name"), "tipo": j.get("type"), "pai": j.find("parent").get("link"),
                "R": _rpy(*rpy), "p": xyz,
                "eixo": np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else None,
                "lim": (float(lim.get("lower")), float(lim.get("upper"))) if lim is not None else (-np.pi, np.pi)}
        self.idx = {n: i for i, n in enumerate(NOMES)}
        self.cadeia = {lado: self._cadeia(f"{lado}_wrist_yaw_link") for lado in ("left", "right")}
        por_nome = {j["nome"]: j for j in self.junta_do_filho.values()}
        self.lo = np.array([por_nome[n]["lim"][0] for n in NOMES])
        self.hi = np.array([por_nome[n]["lim"][1] for n in NOMES])

    def _cadeia(self, link):
        c = []
        while link in self.junta_do_filho:
            j = self.junta_do_filho[link]
            c.append(j)
            link = j["pai"]
        return c[::-1]   # da pelvis para o punho

    def pose(self, q29, lado):
        R, p = np.eye(3), np.zeros(3)
        for j in self.cadeia[lado]:
            p = p + R @ j["p"]
            R = R @ j["R"]
            if j["tipo"] in ("revolute", "continuous") and j["nome"] in self.idx:
                R = R @ _eixo(j["eixo"], q29[self.idx[j["nome"]]])
        return R, p + R @ T_GARRA[lado]

    def ee9(self, q29, lado):
        R, p = self.pose(q29, lado)
        return np.concatenate([p, R[:, 0], R[:, 1]]).astype(np.float32)

    def pose_jac(self, q29, lado):
        """Pose da garra e jacobiano geométrico (6 x 29, referencial da pelvis; linhas v, w)."""
        R, p = np.eye(3), np.zeros(3)
        eixos = []
        for j in self.cadeia[lado]:
            p = p + R @ j["p"]
            R = R @ j["R"]
            if j["tipo"] in ("revolute", "continuous") and j["nome"] in self.idx:
                eixos.append((self.idx[j["nome"]], R @ (j["eixo"] / np.linalg.norm(j["eixo"])), p.copy()))
                R = R @ _eixo(j["eixo"], q29[self.idx[j["nome"]]])
        pe = p + R @ T_GARRA[lado]
        J = np.zeros((6, 29))
        for i, z, o in eixos:
            J[:3, i] = np.cross(z, pe - o)
            J[3:, i] = z
        return R, pe, J

    def ik(self, q29, lado, R_alvo, p_alvo, iters=30, amort=1e-4, peso_rot=0.5, lo=None, hi=None,
           q_repouso=None, peso_repouso=0.0):
        """Mínimos quadrados amortecidos; mexe SÓ nas 7 juntas do braço `lado`, dentro dos limites
        (os do URDF, ou `lo`/`hi` de 29 posições, p.ex. a caixa em volta da pose de partida).
        `q_repouso` + `peso_repouso`: mola leve puxando para uma postura (o braço tem 7 juntas e sobra
        liberdade: sem isso, com pouco peso na orientação, cotovelo e punho "rolam")."""
        idx = BRACO[lado]
        lo = self.lo if lo is None else np.maximum(self.lo, lo)
        hi = self.hi if hi is None else np.minimum(self.hi, hi)
        q = np.array(q29, float)
        W = np.diag([1, 1, 1, peso_rot, peso_rot, peso_rot])
        for _ in range(iters):
            R, p, J = self.pose_jac(q, lado)
            e = np.concatenate([p_alvo - p, rotvec(R_alvo @ R.T)])
            if np.linalg.norm(e[:3]) < 5e-4 and np.linalg.norm(e[3:]) < 5e-3:
                break
            Jb = W @ J[:, idx]
            if q_repouso is not None and peso_repouso > 0:
                lam = amort + peso_repouso
                dq = np.linalg.solve(Jb.T @ Jb + lam * np.eye(len(idx)),
                                     Jb.T @ (W @ e) + peso_repouso * (np.asarray(q_repouso)[idx] - q[idx]))
            else:
                dq = Jb.T @ np.linalg.solve(Jb @ Jb.T + amort * np.eye(6), W @ e)
            q[idx] = np.clip(q[idx] + dq, lo[idx], hi[idx])
        R, p, _ = self.pose_jac(q, lado)
        return q, float(np.linalg.norm(p_alvo - p)), float(np.linalg.norm(rotvec(R_alvo @ R.T)))


BRACO = {"left": list(range(15, 22)), "right": list(range(22, 29))}


def rotvec(R):
    """log de SO(3) -> vetor de rotação."""
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    a = np.arccos(c)
    if a < 1e-8:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - a < 1e-4:   # perto de 180°: pelo eixo da diagonal
        k = np.argmax(np.diag(R))
        u = R[:, k] + np.eye(3)[k]
        return a * u / np.linalg.norm(u)
    return a * v / (2 * np.sin(a))


def expvec(w):
    return _eixo(w, np.linalg.norm(w)) if np.linalg.norm(w) > 1e-12 else np.eye(3)


def ik_varias_sementes(fk, q29, lado, R_alvo, p_alvo, iters=200):
    """IK para um alvo LONGE da pose atual (ex.: pose inicial): tenta várias sementes de 'braço à
    frente' além da pose atual e fica com a que acerta o alvo com o braço mais perto da atual.
    Uma semente só (a pose atual, braço caído) cai em ramos torcidos, com ombro no limite."""
    s = 1.0 if lado == "left" else -1.0
    idx = BRACO[lado]
    sementes = [np.array(q29, float)]
    for sp in (-1.2, -0.8, -0.4, 0.0):
        for cot in (0.2, 0.7, 1.2):
            for sy in (-0.3, 0.0, 0.3):
                q = np.array(q29, float)
                q[idx] = [sp, 0.2 * s, sy, cot, 0.0, 0.0, 0.0]
                sementes.append(q)
    melhor = None
    for q0 in sementes:
        q, ep, er = fk.ik(q0, lado, R_alvo, p_alvo, iters=iters)
        margem = np.minimum(q[idx] - fk.lo[idx], fk.hi[idx] - q[idx]).min()   # longe dos limites
        custo = 20 * ep + er + 0.1 * np.abs(q[idx] - np.asarray(q29)[idx]).max() + (0.5 if margem < 0.1 else 0)
        if melhor is None or custo < melhor[0]:
            melhor = (custo, q, ep, er)
    return melhor[1], melhor[2], melhor[3]
