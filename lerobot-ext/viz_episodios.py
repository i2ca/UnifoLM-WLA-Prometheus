#!/usr/bin/env python
"""Navegador de episódios do dataset no Rerun, pelo terminal.

O `lerobot-dataset-viz` abre UM episódio e morre: para ver o próximo você
mata a janela e roda tudo de novo, esperando o visualizador subir outra vez.
Este script mantém a janela aberta e troca o episódio por comando — o jeito
prático de varrer 27 tomadas atrás da que ficou ruim.

Também loga o que o visualizador de fábrica ignora: as duas colunas de
pressão das mãos (33 canais cada) viram séries temporais como a ação e o
estado. Elas são metade do motivo de existir deste dataset.

    cd lerobot-ext
    python viz_episodios.py --root meu_dataset/white_cup_on_dripper_2026-08-11

Comandos (dentro do prompt):

    <enter> ou n   próximo episódio          p    episódio anterior
    <número>       vai direto para ele       l    lista os episódios
    r              recarrega o atual         q    sai (a janela continua)
"""

import argparse
import logging
import os

# O LeRobot consulta o Hub para validar a versão do `repo_id` ANTES de olhar o
# disco, e toma 401 num repo que não existe lá — mesmo com o dataset inteiro
# aqui do lado. Offline ele vai direto ao `--root`. Use --online se o dataset
# que você quer ver só existe no Hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.scripts.lerobot_dataset_viz import (
    get_feature_names,
    to_hwc_float32_numpy,
    to_hwc_uint8_numpy,
)
from lerobot.utils.constants import ACTION, OBS_STATE

# Endereço padrão do visualizador do Rerun: é onde o `spawn` o coloca a ouvir.
URL_PADRAO = "rerun+http://127.0.0.1:9876/proxy"

# Colunas que o dataset carrega por obrigação e que não dizem nada num gráfico.
COLUNAS_DE_SERVICO = {
    "timestamp", "frame_index", "episode_index", "index", "task_index", ACTION, OBS_STATE,
}


def features_escalares_extras(dataset: LeRobotDataset) -> list[str]:
    """Vetores float32 do dataset fora ação/estado — aqui, a pressão das mãos."""
    extras = []
    for chave, ft in dataset.features.items():
        if chave in COLUNAS_DE_SERVICO or ft["dtype"] != "float32":
            continue
        if len(ft["shape"]) == 1 and ft["shape"][0] > 0:
            extras.append(chave)
    return sorted(extras)


def monta_blueprint(dataset: LeRobotDataset, extras: list[str]):
    """Mesmo layout do visualizador de fábrica, mais um painel por vetor extra.

    Os painéis são montados aqui em vez de estender o blueprint pronto do
    `build_blueprint_from_dataset`: remendar o objeto do Rerun por dentro
    dependeria de detalhes internos dele, que mudam entre versões.
    """
    import rerun as rr
    import rerun.blueprint as rrb

    painéis = [rrb.Spatial2DView(origin=chave, name=chave) for chave in dataset.meta.camera_keys]

    for origem, chave in ((ACTION, ACTION), ("state", OBS_STATE), *((e, e) for e in extras)):
        if chave not in dataset.features:
            continue
        estilo = rr.SeriesLines(names=get_feature_names(dataset, chave))
        painéis.append(rrb.TimeSeriesView(origin=origem, name=origem, overrides={origem: estilo}))

    return rrb.Blueprint(rrb.Grid(*painéis))


def faixas_de_profundidade(dataset: LeRobotDataset) -> dict[str, tuple[float, float]]:
    """q01/q99 de cada câmera de profundidade — mesma escala do visualizador de fábrica.

    Usar min/max crus deixaria a imagem lavada: um punhado de pixels sem medida
    (ou o teto da sala) estica a escala e o resto do quadro vira uma cor só.
    """
    faixas = {}
    for chave in dataset.meta.depth_keys:
        stats = (dataset.meta.stats or {}).get(chave)
        if not stats:
            continue
        baixo = stats.get("q01", stats["min"])
        alto = stats.get("q99", stats["max"])
        faixas[chave] = (float(np.asarray(baixo).item()), float(np.asarray(alto).item()))
    return faixas


def mostra_episodio(repo_id, root, indice, primeira_vez, url, batch_size, num_workers) -> LeRobotDataset:
    import rerun as rr
    from lerobot.configs.video import DEPTH_MILLIMETER_UNIT

    dataset = LeRobotDataset(repo_id, episodes=[indice], root=root)
    extras = features_escalares_extras(dataset)
    blueprint = monta_blueprint(dataset, extras)

    # Cada episódio entra como uma GRAVAÇÃO nova, com o mesmo application_id.
    # É o que faz a janela trocar de conteúdo em vez de empilhar os quadros de
    # dois episódios na mesma linha do tempo. As anteriores continuam na lista
    # da esquerda, dá para voltar nelas sem recarregar.
    rr.init(f"{repo_id}", recording_id=f"episodio_{indice}", default_blueprint=blueprint)
    if primeira_vez and url is None:
        rr.spawn(default_blueprint=blueprint)  # sobe a janela na primeira vez
    else:
        rr.connect_grpc(url or URL_PADRAO, default_blueprint=blueprint)

    carregador = torch.utils.data.DataLoader(
        dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False
    )

    # A profundidade já sai desquantizada na unidade do dataset; o `meter` diz
    # ao Rerun quantas unidades valem 1 metro.
    metro = 1000.0 if dataset.depth_output_unit == DEPTH_MILLIMETER_UNIT else 1.0
    faixas = faixas_de_profundidade(dataset)

    primeiro = None
    for lote in carregador:
        if primeiro is None:
            primeiro = lote["index"][0].item()

        for i in range(len(lote["index"])):
            rr.set_time("frame_index", sequence=lote["index"][i].item() - primeiro)
            rr.set_time("timestamp", timestamp=lote["timestamp"][i].item())

            for chave in dataset.meta.camera_keys:
                if chave in dataset.meta.depth_keys:
                    rr.log(chave, rr.DepthImage(
                        to_hwc_float32_numpy(lote[chave][i]),
                        meter=metro,
                        colormap=rr.components.Colormap.Viridis,
                        depth_range=faixas.get(chave),
                    ))
                else:
                    rr.log(chave, rr.Image(to_hwc_uint8_numpy(lote[chave][i])))

            if ACTION in lote:
                rr.log(ACTION, rr.Scalars(lote[ACTION][i].numpy()))
            if OBS_STATE in lote:
                rr.log("state", rr.Scalars(lote[OBS_STATE][i].numpy()))
            for chave in extras:
                rr.log(chave, rr.Scalars(lote[chave][i].numpy()))

    return dataset


def lista_episodios(meta: LeRobotDatasetMetadata, atual: int) -> None:
    print()
    for linha in meta.episodes:
        indice = linha["episode_index"]
        segundos = linha["length"] / meta.fps
        tarefas = ", ".join(linha["tasks"]) if isinstance(linha["tasks"], list) else linha["tasks"]
        marca = "→" if indice == atual else " "
        print(f" {marca} {indice:3d}  {linha['length']:5d} quadros  {segundos:5.1f}s  {tarefas}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="Pasta do dataset (a que tem meta/, data/, videos/).")
    parser.add_argument("--repo-id", default=None, help="Só um rótulo quando o dataset é local; o padrão vem do --root.")
    parser.add_argument("--episode", type=int, default=0, help="Episódio inicial.")
    parser.add_argument("--url", default=None,
                        help=f"Conecta numa janela do Rerun já aberta em vez de subir outra (ex.: {URL_PADRAO}).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--online", action="store_true", help="Deixa o LeRobot consultar o Hub.")
    args = parser.parse_args()

    if args.online:
        os.environ.pop("HF_HUB_OFFLINE", None)

    logging.getLogger().setLevel(logging.WARNING)  # o log por episódio só atrapalha aqui

    repo_id = args.repo_id or f"local/{os.path.basename(os.path.normpath(args.root))}"
    meta = LeRobotDatasetMetadata(repo_id, root=args.root)
    total = meta.total_episodes

    print(f"\n{repo_id}  —  {total} episódios, {meta.total_frames} quadros, {meta.fps} fps")
    print("  <enter>/n próximo   p anterior   <número> vai para ele   l lista   r recarrega   q sai\n")

    atual = max(0, min(args.episode, total - 1))
    primeira_vez = True

    while True:
        print(f"carregando episódio {atual}...", flush=True)
        dataset = mostra_episodio(repo_id, args.root, atual, primeira_vez, args.url,
                                  args.batch_size, args.num_workers)
        primeira_vez = False
        quadros = dataset.num_frames
        print(f"episódio {atual}/{total - 1} no visualizador — {quadros} quadros, {quadros / meta.fps:.1f}s")

        while True:
            try:
                comando = input(f"[ep {atual}] > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return

            if comando in ("q", "sair", "exit"):
                return
            if comando in ("", "n", "next"):
                if atual + 1 >= total:
                    print("já é o último episódio.")
                    continue
                atual += 1
                break
            if comando in ("p", "prev", "anterior"):
                if atual == 0:
                    print("já é o primeiro episódio.")
                    continue
                atual -= 1
                break
            if comando in ("r", "reload"):
                break
            if comando in ("l", "lista", "ls"):
                lista_episodios(meta, atual)
                continue
            if comando.isdigit():
                escolhido = int(comando)
                if not 0 <= escolhido < total:
                    print(f"fora da faixa: escolha entre 0 e {total - 1}.")
                    continue
                atual = escolhido
                break
            print("não entendi. use: <enter>/n, p, <número>, l, r, q")


if __name__ == "__main__":
    main()
