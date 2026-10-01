# Servidor do modelo e execução no robô (PGX)

## Servidor Dex3 (`:8601`)

```bash
RUN=lora_prometheus_dex3_maca_medido bash lerobot-ext/wla/sobe_wla_servidor_dex3.sh      # ~80 s para carregar
python lerobot-ext/wla/testa_servidor_dex3.py --imgs <pasta com cabeca/punho_esq/punho_dir.jpg>   # teste de fumaça
```

O [`servidor_wla_dex3.py`](../../lerobot-ext/wla/servidor_wla_dex3.py) é o servidor oficial da Unitree
(websocket + msgpack), com o que o oficial não tem:
- **Mãos Dex3:** `observation.state.*_fig6d` na entrada e `action.*_fig6d` na saída, sem garra Dex1.
- **Carregador de LoRA** (`--lora_run`, `--lora_passo`): monta o modelo base, injeta o LoRA com a config do
  treino, carrega os pesos afinados e a normalização. O repositório da Unitree não tem carregador de LoRA
  para inferência.

Na PGX (GB10) cada trecho de 30 passos leva **~665 ms**.

O servidor original da Unitree (Dex1, `sobe_wla_servidor.sh`, porta `:8600`) continua disponível sem
nenhuma alteração.

## Executor no robô real (`roda_wla_real.py`)

```bash
cd ~/DEV/unifolm-wla && ~/miniforge3/envs/wla/bin/python ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/roda_wla_real.py \
    --servidor ws://127.0.0.1:8601 --pose maca --tarefa "Pick up the apple and place it on the black X." --segundos 120
```

O ciclo do executor:
1. lê as câmeras (5555) e as juntas (6001);
2. pergunta ao servidor (assíncrono, a 20 Hz);
3. converte a pose das mãos (pelvis) em juntas por IK (`fk_g1.py`);
4. comanda braços, cintura e dedos pela ponte v3.

Ao conectar, ele deve mostrar **"mãos: Dex3 pelos DEDOS (fig6d)"**.

### Camadas de segurança

| Camada | O quê |
|---|---|
| Pose de partida | vai **devagar** (10 s) até a pose do início do dataset (`--pose maca` / `gravacao` / `inicial`) |
| Caixa da mão | a mão fica a ±25 cm (frente/trás), ±30 cm (lados), 30 cm abaixo e 15 cm acima da partida, **dentro da IK** |
| Passo máximo | ≤ 1 cm de mão e ≤ 0,03 rad de junta por passo, encolhendo o movimento inteiro |
| Mãos separadas | no mínimo 8 cm de distância lateral |
| Integrador | corrige o braço que cede ao peso, só com o alvo e o braço parados (±0,3 rad); para de forçar junta presa |
| **Força dos dedos** | o alvo do dedo fica ≤ **0,25 rad** da posição medida, com kp 0,5 (≈0,12 N·m por junta) e fechamento ≤ 85 %. Sem isso, o servo da Dex3 desligava ao apertar a maçã |
| Tronco | **segue o yaw da cintura pedido pelo modelo** (`--cintura yaw`, ±0,35 rad, devagar); o roll e o pitch ficam seguros. `--cintura parada` segura tudo |
| Ponte v3 | 0,3 rad/s, kp 40 nos braços e 20 nos punhos, **cogumelo** |
| Software | pânico por software se o estado parar, num erro, no fim ou no botão "parar" da cabine (`:8090`) |

O ajuste fino dos dedos é por `--kp-mao`, `--folga-mao` e `--fecha-max`.

### Cada rodada fica gravada

A pasta `~/wla_real_runs/<data_hora>/` guarda, a cada consulta:
- as imagens que o modelo recebeu;
- o trecho previsto inteiro;
- as juntas medidas e comandadas;
- a trajetória.

Para ver uma rodada:

```bash
python relatorio_wla_real.py ~/wla_real_runs/20261001_132709           # resumo e gráficos
python replay_wla.py ~/wla_real_runs/20261001_132709 [--reinferir]     # replay (e re-consulta ao modelo)
```

## Sombra e cabine

- **Sombra:** o `sombra_wla.py` roda o modelo **sem mandar nada ao robô** e mostra na cabine (`:8090`) a
  trajetória que ele faria. É o primeiro teste com uma câmera ou uma cena novas.
- **Sem robô:** o executor também roda contra o [simulador](Simulacao) (`--robo 127.0.0.1`).
