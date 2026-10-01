# Ver os episódios

## `viz_episodios.py` (Rerun)

Ele mantém a janela aberta e troca de episódio por comando: Enter ou `n` para o próximo, `p` para o
anterior, um número para ir direto a um episódio e `l` para listar. Mostra as câmeras, a ação, o estado e
a pressão das mãos.

```bash
cd lerobot-ext && python viz_episodios.py --root meu_dataset/maca_x_preto_2026-09-30
```

## `tracelr` (Rust, rápido, com a trajetória das duas mãos)

É um visualizador de datasets LeRobot (v2.1 e v3.0):
- vídeo em tempo real;
- grade com vários episódios ao mesmo tempo (tecla `G`);
- anotação de frases (`--annotate`);
- **trajetória 3D da mão**, calculada por cinemática a partir do URDF.

**Adaptado para o G1:**
- **Duas mãos:** a trajetória da mão **direita** sai na cor do tema e a da **esquerda** em **azul**. O fim
  de cada cadeia é a palma (`*_hand_palm_link`).
- **Juntas casadas pelo nome:** as colunas do SDK da Unitree (`kLeftShoulderPitch.q`) são ligadas às juntas
  do URDF (`left_shoulder_pitch_joint`). Com as pernas e a cintura gravadas, o tronco entra na conta.
- **Unidades:** `.q` é lido em radianos e `.pos` em graus, como no SO101.
- **Conferido contra o MuJoCo** na pose de partida: as duas palmas batem com 1,6 mm de diferença
  (`cargo test`, `TRACELR_FK_REF`).

```bash
cd lerobot-ext/tracelr && bash ver_dataset.sh ../meu_dataset/copo_branco_2026-10-01
```

O `ver_dataset.sh` já passa o URDF do G1 e contorna o erro de compilação do FFmpeg
(`'limits.h' file not found`) com `BINDGEN_EXTRA_CLANG_ARGS`.

## Exemplos

Os datasets são grandes demais para o git e ficam no Hugging Face Hub (público) e no `/data` da Athena:

| Dataset | Episódios | Frase |
|---|---|---|
| [Mrwlker/maca_x_preto_2026-09-30](https://huggingface.co/datasets/Mrwlker/maca_x_preto_2026-09-30) | 50 (27 569 quadros, 422 MB) | Pick up the apple and place it on the black X. |

Para baixar:

```bash
huggingface-cli download Mrwlker/maca_x_preto_2026-09-30 --repo-type dataset --local-dir lerobot-ext/meu_dataset/maca_x_preto_2026-09-30
cd lerobot-ext/tracelr && bash ver_dataset.sh ../meu_dataset/maca_x_preto_2026-09-30
```

As rodadas do robô real (`~/wla_real_runs/<data>`) guardam, a cada consulta ao modelo:
- as imagens;
- o trecho previsto;
- as juntas;
- a trajetória.

Elas podem ser vistas com `relatorio_wla_real.py` e `replay_wla.py` (ver [Servidor e execução](Servidor-e-Execucao)).
