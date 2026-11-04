#!/usr/bin/env python3
"""TELA DO PROMETHEUS (04/11) — o painelzinho de 3,5" (480x320) do Raspberry Pi 5 ligado na PGX pela USB-C.

Programa NATIVO (pygame, desenha direto na tela: KMS/DRM no Pi e no QEMU, sem navegador — as telinhas SPI de 3,5" e
o QEMU emulando ARM são lentos demais para um Chromium). Mostra:
  - a TRAVA da ponte (livre / travada), o modo (loco) e a bateria;
  - as temperaturas que importam (cintura roll/pitch/yaw, ombros, cotovelos, mãos) em cores, e a junta mais quente;
  - se WLA, executor, ER-1 e mapa estão no ar na PGX;
  - os botões TRAVAR (imediato), Rearmar, Pose de gravação, Pose inicial e Descanso (estes pedem confirmação).

Fontes de dados:
  --fonte http://10.55.0.1:8097   o painel_mini.py da PGX (GET /estado.json, POST /acao) — o de verdade
  --falso                         dados FAKE (para testar sem robô: as temperaturas andam, os botões mudam o estado)

    python3 tela.py --falso --janela          # no PC, numa janela de 480x320
    python3 tela.py --fonte http://10.55.0.1:8097   # no Pi (tela cheia; o serviço prometheus-tela faz isso)
"""
import argparse
import json
import math
import os
import random
import threading
import time
import urllib.request

L, A = 480, 320
COR = {"fundo": (11, 14, 19), "caixa": (21, 26, 34), "borda": (42, 49, 64), "txt": (230, 233, 239),
       "fraco": (139, 147, 161), "ok": (63, 185, 80), "aviso": (227, 179, 65), "erro": (248, 81, 73),
       "azul": (88, 166, 255), "livre": (23, 58, 34), "presa": (74, 26, 26), "semsinal": (42, 47, 58)}
BOTOES = [("TRAVAR", "panico", (138, 31, 31), None),
          ("Rearmar", "rearmar", (31, 111, 58), "Rearmar a ponte?\n(o cogumelo precisa estar solto)"),
          ("Pose de\ngravação", "pose:gravacao", (31, 79, 138), "Ir para a pose de GRAVAÇÃO?"),
          ("Pose\ninicial", "pose:inicial", (36, 64, 95), "Ir para a pose INICIAL?"),
          ("Descanso", "pose:descanso", (74, 58, 31), "Ir para o DESCANSO\n(punho fechado)?")]


# ── fontes de dados ─────────────────────────────────────────────────────────────────────────────────────
class Falso:
    """Um robô de mentira: temperaturas andando devagar, a trava e a pose respondendo aos botões."""

    def __init__(self):
        self.t0 = time.time()
        self.travada, self.pose, self.bateria = True, "parada", 78.0
        self.base = {"cintura roll": 52, "cintura pitch": 49, "cintura yaw": 41, "ombro E": 46, "ombro D": 58,
                     "ombro roll E": 44, "ombro roll D": 55, "cotovelo E": 40, "cotovelo D": 47, "mão E": 36, "mão D": 39}
        self.fase = {k: random.random() * 6 for k in self.base}
        self.pose_fim = 0.0

    def estado(self):
        t = time.time() - self.t0
        temps = [[k, v + 6 * math.sin(t / 40 + self.fase[k]) + (18 if k == "ombro D" and int(t / 60) % 2 else 0)]
                 for k, v in self.base.items()]
        self.bateria = max(5.0, 78.0 - t / 120)
        if self.pose_fim and time.time() < self.pose_fim:
            pose = f"{self.pose}: {100 - 100 * (self.pose_fim - time.time()) / 6:.0f}%"
        else:
            pose, self.pose_fim = ("parada" if not self.pose_fim else f"{self.pose}: pronto"), self.pose_fim and -1
        quente = max(temps, key=lambda x: x[1])
        return {"robo": True, "trava": {"travada": self.travada, "texto": "botão solto, falta REARMAR" if self.travada else ""},
                "modo": "loco", "bateria": int(self.bateria), "temps": temps[:9], "mais_quente": quente, "pose": pose,
                "servicos": {"WLA": True, "executor": int(t / 30) % 3 != 0, "ER-1": True, "mapa": True}, "falso": True}

    def acao(self, a):
        if a == "panico":
            self.travada = True
            return "TRAVADO"
        if a == "rearmar":
            self.travada = False
            return "rearmado (falso)"
        if a.startswith("pose:"):
            if self.travada:
                return "a ponte está TRAVADA: rearme primeiro"
            self.pose, self.pose_fim = "indo para " + a[5:], time.time() + 6
            return "indo"
        return "?"


class Rede:
    """O painel_mini.py da PGX."""

    def __init__(self, url):
        self.url, self.ultimo = url.rstrip("/"), {"robo": False, "erro": "conectando…"}
        threading.Thread(target=self._laco, daemon=True).start()

    def _laco(self):
        while True:
            try:
                self.ultimo = json.loads(urllib.request.urlopen(f"{self.url}/estado.json", timeout=1.5).read())
            except Exception as e:  # noqa: BLE001
                self.ultimo = {"robo": False, "erro": f"sem a PGX ({str(e)[:30]})", "pgx": False}
            time.sleep(0.7)

    def estado(self):
        return self.ultimo

    def acao(self, a):
        try:
            r = urllib.request.urlopen(urllib.request.Request(f"{self.url}/acao", data=json.dumps({"acao": a}).encode(),
                                                              method="POST"), timeout=3).read()
            return json.loads(r).get("msg", "enviado")
        except Exception as e:  # noqa: BLE001
            return f"sem resposta ({str(e)[:30]})"


# ── desenho ────────────────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fonte", default=os.environ.get("TELA_FONTE", "http://10.55.0.1:8097"))
    ap.add_argument("--falso", action="store_true", default=os.environ.get("TELA_FALSO") == "1")
    ap.add_argument("--janela", action="store_true", help="janela 480x320 (no PC); sem isso: tela cheia")
    a = ap.parse_args()
    import pygame
    pygame.init()
    tela = pygame.display.set_mode((L, A), 0 if a.janela else pygame.FULLSCREEN)
    pygame.display.set_caption("Prometheus · tela")
    pygame.mouse.set_visible(a.janela)
    sup = pygame.Surface((L, A))
    f = {k: pygame.font.Font(None, s) for k, s in (("p", 17), ("m", 20), ("g", 27), ("gg", 32))}
    fonte = Falso() if a.falso else Rede(a.fonte)
    caixas_bot = [pygame.Rect(7 + i * 94, 233, 88, 80) for i in range(5)]
    conf, msg, msg_ate = None, "", 0.0
    relogio = pygame.time.Clock()

    def texto(s, fnt, cor, pos, centro=False, larg=None):
        linhas = s.split("\n")
        alt = fnt.get_linesize()
        y0 = pos[1] - (alt * len(linhas)) / 2 if centro else pos[1]
        for i, ln in enumerate(linhas):
            r = fnt.render(ln, True, cor)
            if larg and r.get_width() > larg:   # corta com reticências
                while ln and fnt.size(ln + "…")[0] > larg:
                    ln = ln[:-1]
                r = fnt.render(ln + "…", True, cor)
            x = pos[0] - r.get_width() / 2 if centro else pos[0]
            sup.blit(r, (x, y0 + i * alt))

    def cor_t(t):
        return COR["erro"] if t >= 75 else COR["aviso"] if t >= 60 else COR["txt"]

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE and a.janela):
                return
            toque = None
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                toque = ev.pos
            elif ev.type == pygame.FINGERDOWN:
                toque = (int(ev.x * L), int(ev.y * A))
            if toque is None:
                continue
            if conf is not None:
                acao_c, _, t_c = conf
                if pygame.Rect(250, 200, 160, 64).collidepoint(toque):      # SIM
                    msg, msg_ate, conf = fonte.acao(acao_c), time.time() + 3.5, None
                elif pygame.Rect(70, 200, 160, 64).collidepoint(toque):     # NÃO
                    conf = None
                continue
            for r, (_, acao_b, _, pergunta) in zip(caixas_bot, BOTOES):
                if r.collidepoint(toque):
                    if pergunta is None:           # TRAVAR: na hora
                        msg, msg_ate = fonte.acao(acao_b), time.time() + 3.5
                    else:
                        conf = (acao_b, pergunta, time.time())
        if conf is not None and time.time() - conf[2] > 6:
            conf = None

        e = fonte.estado()
        sup.fill(COR["fundo"])
        # topo: a trava, o modo, a bateria
        tv = e.get("trava")
        if not e.get("robo"):
            fundo_t, txt_t = COR["semsinal"], e.get("erro") or "robô sem resposta"
        elif not tv:
            fundo_t, txt_t = COR["semsinal"], "ponte sem sinal"
        else:
            fundo_t = COR["presa"] if tv["travada"] else COR["livre"]
            txt_t = ("TRAVADA" if tv["travada"] else "LIVRE") + (f" · {tv['texto']}" if tv.get("texto") else "")
        pygame.draw.rect(sup, fundo_t, (7, 6, 300, 30), border_radius=8)
        pygame.draw.rect(sup, COR["erro"] if tv and tv["travada"] else COR["ok"] if tv else COR["borda"], (7, 6, 300, 30), 1, border_radius=8)
        texto(txt_t, f["m"], COR["txt"], (15, 13), larg=286)
        for i, (rot, ok) in enumerate(((e.get("modo") or "modo —", e.get("modo") == "loco"),
                                       (f"bat {e['bateria']}%" if e.get("bateria") is not None else "bat —",
                                        (e.get("bateria") or 0) >= 25))):
            x = 314 + i * 84
            pygame.draw.rect(sup, COR["borda"], (x, 8, 78, 26), 1, border_radius=13)
            texto(rot, f["p"], COR["ok"] if ok else COR["fraco"], (x + 39, 21), centro=True)
        # meio: as temperaturas (3 colunas)
        itens = list(e.get("temps") or [])
        if e.get("mais_quente"):
            itens.append([f"+quente: {e['mais_quente'][0]}", e["mais_quente"][1]])
        for k, (nome, t) in enumerate(itens[:12]):
            cx, cy = 7 + (k % 3) * 156, 42 + (k // 3) * 44
            pygame.draw.rect(sup, COR["caixa"], (cx, cy, 150, 40), border_radius=7)
            pygame.draw.rect(sup, COR["borda"], (cx, cy, 150, 40), 1, border_radius=7)
            texto(f"{t:.0f}°", f["g"], cor_t(t), (cx + 7, cy + 3))
            texto(nome, f["p"], COR["fraco"], (cx + 7, cy + 24), larg=138)
        # serviços da PGX + a pose
        x = 9
        for nome, ok in (e.get("servicos") or {}).items():
            pygame.draw.circle(sup, COR["ok"] if ok else COR["erro"], (x + 4, 220), 4)
            texto(nome, f["p"], COR["fraco"], (x + 12, 214))
            x += 22 + f["p"].size(nome)[0]
        if e.get("pose"):
            texto(f"pose: {e['pose']}", f["p"], COR["fraco"], (x + 6, 214), larg=L - x - 14)
        if e.get("falso"):
            texto("DADOS FALSOS", f["p"], COR["aviso"], (L - 100, 214))
        # botões
        for r, (rot, _, cor_b, _) in zip(caixas_bot, BOTOES):
            pygame.draw.rect(sup, cor_b, r, border_radius=9)
            pygame.draw.rect(sup, COR["borda"], r, 1, border_radius=9)
            texto(rot, f["m"], COR["txt"], r.center, centro=True)
        # mensagem
        if time.time() < msg_ate:
            pygame.draw.rect(sup, (34, 43, 58), (7, 188, 466, 24), border_radius=6)
            texto(msg, f["p"], COR["txt"], (14, 193), larg=450)
        # confirmação
        if conf is not None:
            sombra = pygame.Surface((L, A), pygame.SRCALPHA)
            sombra.fill((0, 0, 0, 215))
            sup.blit(sombra, (0, 0))
            texto(conf[1], f["gg"], COR["txt"], (L / 2, 110), centro=True)
            for rr, rot, cc in ((pygame.Rect(70, 200, 160, 64), "Não", (31, 38, 51)), (pygame.Rect(250, 200, 160, 64), "Sim", (31, 111, 58))):
                pygame.draw.rect(sup, cc, rr, border_radius=10)
                pygame.draw.rect(sup, COR["borda"], rr, 1, border_radius=10)
                texto(rot, f["gg"], COR["txt"], rr.center, centro=True)
        tela.blit(pygame.transform.scale(sup, tela.get_size()) if tela.get_size() != (L, A) else sup, (0, 0))
        pygame.display.flip()
        relogio.tick(10)
        if os.environ.get("TELA_CAPTURA"):   # testes: salva um quadro (TELA_CAPTURA=arquivo.png) e sai
            cap = os.environ["TELA_CAPTURA"]
            if os.environ.get("TELA_CLIQUE") and not getattr(main, "_clicou", False):   # simula um toque (x,y)
                main._clicou = True
                x, y = (int(v) for v in os.environ["TELA_CLIQUE"].split(","))
                pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(x, y), button=1))
            elif time.time() - getattr(main, "_t0", time.time()) > 1.5:
                pygame.image.save(sup, cap)
                return
            main._t0 = getattr(main, "_t0", time.time())


if __name__ == "__main__":
    main()
