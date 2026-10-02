# UnifoLM-WLA-Prometheus

O **UnifoLM-WLA-1.0** da Unitree rodando no **nosso G1 + mãos Dex3 (Prometheus, i2ca)**: teleoperação em
VR, gravação de datasets, conversão para o formato do WLA, fine-tuning LoRA para a Dex3 (sem reduzir a mão
a garra), servidor do modelo, executor no robô real com botão de pânico, simulação MuJoCo da cena do café
e ferramentas para ver os episódios.

É a parte "WLA" do [prometheus-vla](https://github.com/i2ca/prometheus-vla), separada e organizada.
**Documentação completa na [wiki](../../wiki)** (também em [`docs/wiki/`](docs/wiki/)).

```
 VR (Quest)          notebook                     G1 (Jetson)                 Athena (3x A100)
 ──────────   ┌────────────────────────┐    ┌───────────────────────┐     ┌──────────────────────┐
  cabeça  ───▶│ teleop xr_g1_arm        │──▶│ ponte v3 + pânico      │     │ converte (mão MEDIDA) │
  mãos        │ record (LeRobot 0.6.1)  │◀──│ câmeras: ZED + 2 D435  │     │ LoRA no DiT (fig6d)   │
              │ dataset LeRobot v3      │──────────────────────────────────▶│ avalia / extrai delta │
              │ viz_episodios / tracelr │    │ painel :8095           │     └──────────┬───────────┘
              │ sim MuJoCo (café)       │    └──────────▲────────────┘                │ checkpoint
              └────────────────────────┘               │ ZMQ                          ▼
                                             ┌──────────┴────────────┐     ┌──────────────────────┐
                                             │ PGX (GB10)            │◀────│ servidor WLA Dex3     │
                                             │ roda_wla_real.py      │     │ :8601 (+ ER-1 :8098)  │
                                             └───────────────────────┘     └──────────────────────┘
```

## Estrutura

| Pasta | O quê | Roda em |
|---|---|---|
| [`lerobot-ext/robot/`](lerobot-ext/robot) | plugin LeRobot do G1 + Dex3 (estado, ação, câmeras WLA, coluna reta) | notebook |
| [`lerobot-ext/teleop/`](lerobot-ext/teleop) | teleoperação VR (`xr_g1_arm.py`, televuer, IK, retargeting) | notebook |
| [`lerobot-ext/init_lerobot_record_v2.py`](lerobot-ext/init_lerobot_record_v2.py) | gravação (voz, teclado, controle VR, robô vivo ao salvar) | notebook |
| [`lerobot-ext/config/`](lerobot-ext/config) | configs de teleop e de gravação das tarefas | notebook |
| [`lerobot-ext/robo_g1/`](lerobot-ext/robo_g1) | **na placa do robô**: init, ponte v3 com pânico, câmeras, painel | G1 |
| [`lerobot-ext/wla/`](lerobot-ext/wla) | servidor Dex3, executor no robô, sombra, gravador/relatório/replay, ER-1, sim | PGX / notebook |
| [`lerobot-ext/wla/treino/`](lerobot-ext/wla/treino) | conversão do dataset, configs e script do LoRA, avaliação | Athena |
| [`lerobot-ext/viz_episodios.py`](lerobot-ext/viz_episodios.py) | navegador de episódios no Rerun | notebook |
| [`lerobot-ext/tracelr/`](lerobot-ext/tracelr) | visualizador rápido (Rust) com a trajetória das DUAS mãos do G1 | notebook |
| [`unitree-g1-mujoco/`](unitree-g1-mujoco) | cena do café com a maçã/caneca, G1 com ZED e D435 nos suportes | notebook |
| [`third_party/`](third_party) | submódulos: LeRobot (fork), UnifoLM-WLA (Unitree), unitree_sdk2_python | — |

## Começo rápido

```bash
git clone --recursive https://github.com/i2ca/UnifoLM-WLA-Prometheus ~/DEV/UnifoLM-WLA-Prometheus
```

1. [Instalação](docs/wiki/Instalacao.md) de cada máquina (notebook, robô, PGX, Athena).
2. [Gravar um dataset](docs/wiki/Teleop-e-Gravacao.md) por teleoperação VR.
3. [Converter e treinar](docs/wiki/Conversao-e-Treino.md) o LoRA na Athena.
4. [Subir o servidor e rodar no robô](docs/wiki/Servidor-e-Execucao.md), com o cogumelo de pânico à mão.

Antes de tudo, leia as [lições aprendidas](docs/wiki/Licoes-Aprendidas.md): metade dos problemas que
tivemos está lá, com a causa e a correção.

## Datasets (Hugging Face, públicos)

Gravados por teleoperação VR no G1 + Dex3, com a ZED na cabeça e duas D435 nos punhos (640x480, 30 fps),
em formato LeRobot v3.0. Os da **cena mista** (02/10) têm todos os objetos na mesa em todos os episódios
(o modelo escolhe a tarefa pela frase) e o estado da Dex3 medido de verdade (nos de 30/09–01/10 ele saiu 0).

| Dataset | Tarefa (frase) | Episódios | Quadros |
|---|---|---|---|
| [Mrwlker/maca_x_preto_2026-09-30](https://huggingface.co/datasets/Mrwlker/maca_x_preto_2026-09-30) | Pick up the apple and place it on the black X. | 50 | 27 569 |
| [Mrwlker/copo_branco_2026-10-01](https://huggingface.co/datasets/Mrwlker/copo_branco_2026-10-01) | Pick up the white mug. | 47 | 9 533 |
| [Mrwlker/copo_no_coador_2026-10-01](https://huggingface.co/datasets/Mrwlker/copo_no_coador_2026-10-01) | Place the white mug under the coffee strainer. | 50 | 15 531 |
| [Mrwlker/pegar_caneca_mesa_nova_2026-10-02](https://huggingface.co/datasets/Mrwlker/pegar_caneca_mesa_nova_2026-10-02) | Pick up the white mug. *(cena mista, mesa nova)* | 55 | 21 355 |
| [Mrwlker/pegar_maca_mesa_nova_2026-10-02](https://huggingface.co/datasets/Mrwlker/pegar_maca_mesa_nova_2026-10-02) | Pick up the apple. *(cena mista, mesa nova)* | 52 | 25 008 |

```bash
huggingface-cli download Mrwlker/copo_branco_2026-10-01 --repo-type dataset --local-dir lerobot-ext/meu_dataset/copo_branco_2026-10-01
```

## Créditos e licenças

- **UnifoLM-WLA** e **UnifoLM-ER-1**: Unitree Robotics ([unifolm-wla](https://github.com/unitreerobotics/unifolm-wla), Apache-2.0).
- **LeRobot**: Hugging Face, via o fork [Breno-de-Angelo/lerobot](https://github.com/Breno-de-Angelo/lerobot) (Apache-2.0).
- **televuer / teleimager / robot_control**: Unitree xr_teleoperate (licenças nas pastas).
- **ZED 1**: malha da Stereolabs ([zed-ros2-description](https://github.com/stereolabs/zed-ros2-description), Apache-2.0).
- **RealSense D435i**: malha do MuJoCo Menagerie (licença em `unitree-g1-mujoco/assets/realsense_d435i/`).
- **G1** (URDF e malhas): Unitree Robotics.
- **tracelr**: visualizador de datasets LeRobot (Rust), adaptado aqui para o G1.
