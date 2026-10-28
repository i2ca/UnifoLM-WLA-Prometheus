#!/usr/bin/env python
"""V7 (28/10): PROFUNDIDADE da ZED da cabeça (estéreo SN19436) para os datasets do v7, offline e em paralelo.

Para cada dataset: lê head_stereo_left + head_stereo_right (LeRobot v3: vários episódios num file-000.mp4), calcula a
profundidade HÍBRIDA (zed/profundidade_hibrida.py: Depth Anything V2 denso + escala do estéreo SN19436; o estéreo
puro deixava buracos na toalha, no inox e nas bordas) na grade do head_stereo_left e grava
observation.images.head_depth como vídeo em mapa de cor (TURBO de 1/Z, faixa --faixa m; sem medida = preto). Divide os quadros em trechos para --cpus processos
e junta com o ffmpeg (-c copy). Monta Prometheus_G1_Dex3_V7/<dataset>: links para data/ e os vídeos de antes + meta
copiada com a câmera nova (info.json, stats.json, meta/episodes). Os originais não são tocados.

    CUDA_VISIBLE_DEVICES=2 python profundidade_v7.py --cpus 32     # ~/DEV/unifolm-wla-lora, env unifolm-wla
    python profundidade_v7.py --teste 90           # só 90 quadros do 1º dataset + amostras PNG (velocidade/aspecto)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "zed"))
sys.path.insert(0, str(AQUI / "zed"))
DATASETS = ["G1_Dex3_Cafe_Caneca_No_Coador", "G1_Dex3_Pegar_Caneca_Mesa_Nova", "G1_Dex3_Pegar_Maca_Mesa_Nova"]
ESQ, DIR, PROF = ("observation.images.head_stereo_left", "observation.images.head_stereo_right",
                  "observation.images.head_depth")
FPS = 30
_p = None
LOTE = 16


def _hibrida():
    """Um ProfundidadeHibrida por processo (Depth Anything na GPU, estéreo na CPU)."""
    global _p
    if _p is None:
        import cv2
        import torch
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        from profundidade_hibrida import ProfundidadeHibrida
        _p = ProfundidadeHibrida(dispositivo="cuda" if torch.cuda.is_available() else "cpu")
    return _p


def quadros(caminho, a, b):
    """Quadros [a, b) do vídeo, em RGB, com índice exato pelo pts (o vídeo tem g=2: o seek é barato)."""
    import av
    with av.open(str(caminho)) as c:
        s = c.streams.video[0]
        tb = s.time_base
        c.seek(int(max(0, a - 2) / FPS / tb), stream=s, backward=True)
        for f in c.decode(s):
            i = int(round(float(f.pts * tb) * FPS))
            if i < a:
                continue
            if i >= b:
                break
            yield i, f.to_ndarray(format="rgb24")


def trecho(ds_dir, a, b, saida, faixa):
    import av
    esq = ds_dir / "videos" / ESQ / "chunk-000" / "file-000.mp4"
    dir_ = ds_dir / "videos" / DIR / "chunk-000" / "file-000.mp4"
    t0 = time.time()
    with av.open(str(saida), "w") as o:
        s = o.add_stream("libx264", rate=FPS)
        s.width, s.height, s.pix_fmt = 640, 480, "yuv420p"
        s.options = {"crf": "18", "g": "2", "threads": "1"}
        s.codec_context.time_base = Fraction(1, FPS)
        n = 0
        ph = _hibrida()
        ph.st, ph.n_sem_estereo = None, 0       # cada trecho começa sem escala anterior
        from profundidade_hibrida import colore

        def grava(pares):
            nonlocal n
            for Z in ph.lote(pares):
                fr = av.VideoFrame.from_ndarray(colore(Z, faixa), format="rgb24")
                fr.pts = n
                for p in s.encode(fr):
                    o.mux(p)
                n += 1
        pares = []
        for (i, fe), (j, fd) in zip(quadros(esq, a, b), quadros(dir_, a, b)):
            if i != j:
                raise RuntimeError(f"quadros desalinhados {i} != {j}")
            pares.append((fe, fd))
            if len(pares) == LOTE:
                grava(pares)
                pares = []
        if pares:
            grava(pares)
        for p in s.encode():
            o.mux(p)
    if n != b - a:
        raise RuntimeError(f"{saida.name}: {n} quadros, esperava {b - a}")
    return saida, n, time.time() - t0, ph.n_sem_estereo


def total_quadros(ds_dir):
    info = json.load(open(ds_dir / "meta/info.json"))
    return int(info["total_frames"])


def monta_v7(nome, origem, destino, video):
    """Prometheus_G1_Dex3_V7/<nome>: links para o original + meta com a câmera nova."""
    import pandas as pd
    d = destino / nome
    if d.exists():
        shutil.rmtree(d)
    (d / "videos").mkdir(parents=True)
    os.symlink(origem / "data", d / "data")
    for v in (origem / "videos").iterdir():
        os.symlink(v, d / "videos" / v.name)
    (d / "videos" / PROF / "chunk-000").mkdir(parents=True)
    shutil.move(str(video), d / "videos" / PROF / "chunk-000" / "file-000.mp4")
    shutil.copytree(origem / "meta", d / "meta", symlinks=False)
    info = json.load(open(d / "meta/info.json"))
    info["features"][PROF] = json.loads(json.dumps(info["features"][ESQ]))
    info["features"][PROF]["info"]["video.crf"] = 18
    json.dump(info, open(d / "meta/info.json", "w"), indent=4)
    st = json.load(open(d / "meta/stats.json"))
    if ESQ in st:
        st[PROF] = st[ESQ]   # as imagens não são normalizadas por estas estatísticas; só para o formato bater
    json.dump(st, open(d / "meta/stats.json", "w"), indent=4)
    for f in (d / "meta/episodes").rglob("*.parquet"):
        e = pd.read_parquet(f)
        for c in [c for c in e.columns if f"/{ESQ}/" in c]:
            e[c.replace(f"/{ESQ}/", f"/{PROF}/")] = e[c]
        e.to_parquet(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="/home/mrwlker/unifolm_data/Prometheus_G1_Dex3_Cafe")
    ap.add_argument("--destino", default="/data/mrwlker/unifolm_data/Prometheus_G1_Dex3_V7")
    ap.add_argument("--tmp", default="/data/mrwlker/wla_v7/profundidade_tmp")
    ap.add_argument("--cpus", type=int, default=32, help="processos (cada um: estéreo na CPU + Depth Anything na GPU)")
    ap.add_argument("--trecho", type=int, default=300, help="quadros por tarefa")
    ap.add_argument("--faixa", type=float, nargs=2, default=[0.3, 1.2], help="m: perto, longe")
    ap.add_argument("--teste", type=int, default=0, help="só N quadros do 1º dataset + PNGs de amostra")
    a = ap.parse_args()
    tmp, destino = Path(a.tmp), Path(a.destino)
    tmp.mkdir(parents=True, exist_ok=True)
    destino.mkdir(parents=True, exist_ok=True)

    if a.teste:
        import cv2
        ds = Path(a.base) / DATASETS[0]
        t0 = time.time()
        _, n, dt, sem = trecho(ds, 3000, 3000 + a.teste, tmp / "teste.mp4", a.faixa)
        print(f"teste: {n} quadros em {dt:.1f} s ({1000 * dt / n:.0f} ms/quadro num processo) | sem estéreo: {sem}")
        esq = dict(quadros(ds / "videos" / ESQ / "chunk-000" / "file-000.mp4", 3000, 3000 + a.teste))
        prof = dict(quadros(tmp / "teste.mp4", 0, a.teste))
        for k in (0, a.teste // 2, a.teste - 1):
            cv2.imwrite(str(tmp / f"amostra_{k}.png"), np.hstack([esq[3000 + k], prof[k]])[:, :, ::-1])
        print(f"amostras em {tmp}/amostra_*.png | {time.time() - t0:.0f} s")
        return

    for nome in DATASETS:
        ds = Path(a.base) / nome
        origem = ds.resolve()
        N = total_quadros(ds)
        pasta = tmp / nome
        pasta.mkdir(exist_ok=True)
        cortes = [(i, min(i + a.trecho, N)) for i in range(0, N, a.trecho)]
        t0 = time.time()
        print(f"\n▶ {nome}: {N} quadros em {len(cortes)} trechos, {a.cpus} CPUs", flush=True)
        prontos, sem_total = 0, [0]
        import multiprocessing as mp
        with ProcessPoolExecutor(a.cpus, mp_context=mp.get_context("spawn")) as ex:
            fut = {}
            for i, j in cortes:
                seg = pasta / f"seg_{i:06d}.mp4"
                if seg.exists() and seg.with_suffix(".ok").exists():   # retomada
                    prontos += 1
                    continue
                fut[ex.submit(trecho, ds, i, j, seg, a.faixa)] = seg
            for f in as_completed(fut):
                seg, n, dt, sem = f.result()
                sem_total[0] += sem
                seg.with_suffix(".ok").write_text(str(n))
                prontos += 1
                if prontos % 10 == 0 or prontos == len(cortes):
                    el = time.time() - t0
                    print(f"   {prontos}/{len(cortes)} trechos | {el / 60:.1f} min", flush=True)
        lista = pasta / "lista.txt"
        lista.write_text("".join(f"file '{pasta / f'seg_{i:06d}.mp4'}'\n" for i, _ in cortes))
        junto = pasta / "file-000.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lista),
                        "-c", "copy", str(junto)], check=True)
        import av
        with av.open(str(junto)) as c:
            n = sum(1 for _ in c.decode(c.streams.video[0]))
        if n != N:
            raise SystemExit(f"{nome}: vídeo de profundidade com {n} quadros, o dataset tem {N}")
        monta_v7(nome, origem, destino, junto)
        print(f"✅ {nome}: {n} quadros ({sem_total[0]} sem estéreo suficiente: escala do anterior) | "
              f"{(time.time() - t0) / 60:.1f} min -> {destino / nome}", flush=True)
    print("FIM")


if __name__ == "__main__":
    main()
