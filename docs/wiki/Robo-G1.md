# Robô G1 (placa Jetson)

Tudo o que roda **no robô** fica em [`lerobot-ext/robo_g1/`](../../lerobot-ext/robo_g1) e é copiado para
`~/Script_Prometheus_int` pelo `instala_no_robo.sh`. Quem sobe tudo é o `init_prometheus-vla.sh`.

## O que o init sobe

| Programa | Porta | Faz |
|---|---|---|
| `dex3_g1_server_v3_panico.py` (ponte v3) | 6000–6007 | DDS ↔ ZMQ para braços, cintura e mãos, com **botão de pânico**, braços lentos e kp limitado |
| `cameras_wla_server.py` | 5555 | ZED na cabeça (`head_stereo_left/right`) e duas D435 nos punhos (`wrist_left/right`), 640x480 a 30 Hz |
| `painel_diagnostico.py` | 8095 | página com juntas, temperaturas, mãos, bateria, câmeras, pânico, poses e voz |

Variáveis no topo do init. Estes são os valores que usamos:

```bash
KP_BRACO=40  KP_PUNHO=20  VEL_MAO=5.0          # tetos da ponte (braço / punho / velocidade dos dedos)
USE_DIAGNOSTICO=true                           # painel :8095
CAMERA_ESTEREO=true ZED_INVERTIDA=true         # ZED montada de cabeça para baixo (gira 180°)
ZED_ENQUADRAMENTO=largo                        # olho inteiro (~90°) com faixas pretas, como o dataset
PUNHO_DIR_INVERTIDO=true CABECA_REALSENSE=false
```

## Botão de pânico (cogumelo)

- **Ligação:** cogumelo NC entre o GPIO4 (PI.04, saída em 0) e o GPIO6 (PCC.03, entrada com pull-up interno).
  O GPIO4 **tem que** ficar em 0: com ele em 1, o GPIO6 lia 1 nos dois estados do botão.
- **Apertado, ou cabo aberto:** a ponte entra em **pânico**, descarta os comandos do PC e segura a pose medida a 100 Hz.
- **LED do peito:**

  | Cor | Estado |
  |---|---|
  | vermelho, com bipe | botão apertado |
  | **amarelo** | botão solto, mas a ponte ainda está travada esperando o rearme |
  | verde por 2 s | rearmado |

  A ponte sempre sobe **amarela**.
- **Rearme:** o botão precisa estar solto há pelo menos 1 s, e alguém precisa pedir o rearme pelo painel `:8095`
  (botão "Rearmar") ou mandando `{"rearmar": true}` para a porta 6006.
- **Voz:** "Warning. Pose locked." / "Pose unlocked.", na voz interna do G1 (`TtsMaker`, voz 1), com o
  `.wav` como reserva. O volume já está no máximo do SDK (100).

## Câmeras

| Câmera | Onde | Imagem que vai para o modelo |
|---|---|---|
| **ZED 1** (Stereolabs, USB3) | frente da cabeça, montada **de cabeça para baixo** | olho esquerdo, 672x376 (modo VGA), reduzido para 640 de largura com faixas pretas até 480 (`largo`, ~90° na horizontal) |
| D435 punho esquerdo (`138422074380`) | suporte impresso na mão esquerda | 640x480 cor |
| D435 punho direito (`141722078588`) | suporte espelhado na mão direita, **de cabeça para baixo** | 640x480 cor, girada 180° no software |

- **Quadros corrompidos da ZED:** no modo HD vinham ~12 % de quadros corrompidos, com duas imagens emendadas.
  O modo VGA mais um detector de emenda descarta esses quadros, e a contagem aparece no status do servidor.
- **Se a ZED não aparecer depois de ligar o robô:** o init chama o `reenumera_zed.sh`, que precisa de sudo
  (ver [Instalação](Instalacao)). Ligue a ZED num hub USB3: num hub USB2 ela fica a 12 Mbps.

## Painel de diagnóstico (`http://192.168.123.164:8095`)

- **Leitura** (por DDS, sem mandar nada ao robô): as 29 juntas com as duas temperaturas, as Dex3, a IMU,
  a bateria, a placa-mãe, as temperaturas da Jetson e as câmeras.
- **Botões:**

  | Botão | Faz |
  |---|---|
  | Rearmar / Pânico | o mesmo que o cogumelo, pelo navegador |
  | **Assumir pose inicial** | coluna reta, mãos afastadas |
  | **Pose de gravação** | coluna reta, braços abertos e levantados, **fora da vista da ZED**, para começar a gravar |

  As duas poses vão devagar e, no fim, **travam a ponte** segurando a pose. Para continuar, rearme.
- **Fala:** um `POST /fala` faz o G1 falar. O executor usa isso para anunciar a tarefa.

As poses ficam em [`wla/pose_partida_dex1.json`](../../lerobot-ext/wla/pose_partida_dex1.json):
`inicial`, `elevada`, `dataset` (Dex1 da Unitree), `maca` (início do nosso dataset da maçã) e `gravacao`.

## Calor da cintura

Com os braços estendidos à frente, o peso deles faz ~13 Nm no pitch da cintura. Com kp alto e alvo fixo,
o roll e o pitch da cintura chegam a 50–60 °C.

- **Use o guindaste** para aliviar o peso nos testes longos.
- **Vigie a "cintura pitch" no painel.**
- **Os ganhos estão em 300/8** com `cintura_reta: true`, só para a gravação (ver
  [Teleop e gravação](Teleop-e-Gravacao)). Se esquentar, baixe para 250/7 no `config_unitree_g1.py`.
