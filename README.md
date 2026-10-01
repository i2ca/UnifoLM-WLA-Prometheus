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

## Créditos e licenças

- **UnifoLM-WLA** e **UnifoLM-ER-1**: Unitree Robotics ([unifolm-wla](https://github.com/unitreerobotics/unifolm-wla), Apache-2.0).
- **LeRobot**: Hugging Face, via o fork [Breno-de-Angelo/lerobot](https://github.com/Breno-de-Angelo/lerobot) (Apache-2.0).
- **televuer / teleimager / robot_control**: Unitree xr_teleoperate (licenças nas pastas).
- **ZED 1**: malha da Stereolabs ([zed-ros2-description](https://github.com/stereolabs/zed-ros2-description), Apache-2.0).
- **RealSense D435i**: malha do MuJoCo Menagerie (licença em `unitree-g1-mujoco/assets/realsense_d435i/`).
- **G1** (URDF e malhas): Unitree Robotics.
- **tracelr**: visualizador de datasets LeRobot (Rust), adaptado aqui para o G1.
