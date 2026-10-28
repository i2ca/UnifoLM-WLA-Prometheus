"""Profundidade HÍBRIDA (28/10): Depth Anything V2 (denso, bordas nítidas, sem escala) + estéreo da ZED (escala em m).

O estéreo puro (estereo_zed.py) deixa buracos onde não acha o par: toalha branca, inox, bordas, a faixa esquerda.
Aqui o Depth Anything dá a forma da cena inteira (profundidade INVERSA relativa) e, em cada quadro, uma reta
1/Z = s * d_rel + t é ajustada nos pixels em que o estéreo mediu bem (0,25–2 m, com descarte de discrepantes).
Quadro sem estéreo suficiente: usa o último (s, t). As faixas pretas do enquadramento "largo" ficam sem medida.

    from profundidade_hibrida import ProfundidadeHibrida
    p = ProfundidadeHibrida(dispositivo="cuda")
    Zs = p.lote([(esq_rgb, dir_rgb), ...])          # metros, (480, 640), NaN = sem medida
    rgb = colore(Zs[0])                             # mapa de cor (o que o modelo vê como head_depth)
"""
import numpy as np

from estereo_zed import EstereoZED

MODELO_DA = "depth-anything/Depth-Anything-V2-Small-hf"


class ProfundidadeHibrida:
    def __init__(self, dispositivo="cuda", modelo=MODELO_DA, min_pixels=800):
        import cv2
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.cv2, self.torch = cv2, torch
        self.z = EstereoZED()
        z = self.z
        yy, xx = np.mgrid[0:z.h, 0:z.w].astype(np.float32)
        pts = cv2.undistortPoints(np.stack([xx.ravel(), yy.ravel()], 1)[:, None], z.K1, z.d1, R=z.R1, P=z.P1)
        self.mx, self.my = pts[:, 0, 0].reshape(z.h, z.w), pts[:, 0, 1].reshape(z.h, z.w)   # retificada -> original
        self.proc = AutoImageProcessor.from_pretrained(modelo)
        self.da = AutoModelForDepthEstimation.from_pretrained(modelo).to(dispositivo).eval()
        self.dev = dispositivo
        self.min_pixels = min_pixels
        self.st = None          # último (s, t) bom
        self.n_sem_estereo = 0

    def estereo(self, esq_rgb, dir_rgb):
        """Metros na grade original 640x480 do olho esquerdo (NaN = sem medida)."""
        z, cv2 = self.z, self.cv2
        Z = z.profundidade(esq_rgb[:, :, ::-1].copy(), dir_rgb[:, :, ::-1].copy(), retificada=True)
        out = np.full((z.H, z.w), np.nan, np.float32)
        out[z.y0:z.y0 + z.h] = cv2.remap(Z, self.mx, self.my, cv2.INTER_NEAREST, borderValue=np.nan)
        return out

    def relativa(self, imgs_rgb):
        """Depth Anything: profundidade INVERSA relativa (maior = mais perto), na resolução da imagem."""
        torch = self.torch
        with torch.no_grad():
            x = self.proc(images=list(imgs_rgb), return_tensors="pt").to(self.dev)
            d = self.da(**x).predicted_depth                       # (B, h', w')
            d = torch.nn.functional.interpolate(d[:, None], size=imgs_rgb[0].shape[:2], mode="bilinear",
                                                align_corners=False)[:, 0]
        return d.float().cpu().numpy()

    def ajusta(self, d, Zs):
        """Reta 1/Z = s*d + t nos pixels bons do estéreo; 2 rodadas (descarta > 2,5 MAD)."""
        ok = np.isfinite(Zs) & (Zs > 0.25) & (Zs < 2.0)
        if ok.sum() < self.min_pixels:
            return None
        x, y = d[ok].astype(np.float64), 1.0 / Zs[ok].astype(np.float64)
        if len(x) > 20000:
            i = np.random.default_rng(0).choice(len(x), 20000, replace=False)
            x, y = x[i], y[i]
        for _ in range(2):
            A = np.stack([x, np.ones_like(x)], 1)
            (s, t), *_ = np.linalg.lstsq(A, y, rcond=None)
            r = np.abs(y - (s * x + t))
            mad = np.median(r) + 1e-9
            boa = r < 2.5 * 1.4826 * mad
            if boa.sum() < self.min_pixels // 2:
                break
            x, y = x[boa], y[boa]
        return (s, t) if s > 0 else None

    def lote(self, pares):
        """[(esq_rgb, dir_rgb)] -> [Z em metros (480, 640)]."""
        z = self.z
        rel = self.relativa([e for e, _ in pares])
        saida = []
        for (e, dr), d in zip(pares, rel):
            Zs = self.estereo(e, dr)
            st = self.ajusta(d, Zs)
            if st is None:
                self.n_sem_estereo += 1
                st = self.st
            elif self.st is not None:   # suaviza a escala entre quadros (sem piscar)
                st = (0.7 * self.st[0] + 0.3 * st[0], 0.7 * self.st[1] + 0.3 * st[1])
            self.st = st
            Z = np.full(d.shape, np.nan, np.float32)
            if st is not None:
                inv = st[0] * d + st[1]
                Z = np.where(inv > 1e-3, 1.0 / np.maximum(inv, 1e-3), np.nan).astype(np.float32)
            Z[:z.y0] = np.nan                  # faixas pretas do enquadramento: não há cena
            Z[z.y0 + z.h:] = np.nan
            saida.append(Z)
        return saida


def colore(Z, faixa=(0.3, 1.2)):
    """TURBO de 1/Z na faixa (perto = vermelho, longe = azul); sem medida = preto. RGB uint8."""
    import cv2
    perto, longe = faixa
    v = (1.0 / np.where(np.isfinite(Z), Z, np.inf) - 1.0 / longe) / (1.0 / perto - 1.0 / longe)
    ok = np.isfinite(Z)
    u8 = (np.clip(v, 0, 1) * 255).astype(np.uint8)
    rgb = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)[:, :, ::-1]
    rgb[~ok] = 0
    return np.ascontiguousarray(rgb)
