# Teleoperação VR e gravação de datasets

Tudo roda no **notebook**, no ambiente `prometheus-vla`, dentro de `lerobot-ext/`.

```bash
python init_lerobot_teleoparate_v2.py --config_path=config/teleop/teleop_vr_wla.yaml          # só teleop (treino)
python init_lerobot_record_v2.py --config_path=config/record/record_vr_wla_copo_branco.yaml   # gravar
```

No óculos (Quest), abra a página HTTPS que o terminal mostrar, aceite o certificado e entre no modo VR.

## Controles

| Onde | O quê |
|---|---|
| **Cabeça** | girar a cabeça gira o **tronco** (yaw da cintura, até ±57°), a partir de para onde você olhava ao destravar |
| Controle esquerdo, analógico | **anda**: Y frente/trás, X para os lados |
| Controle direito, analógico X | **gira** o corpo inteiro |
| **X** | trava / destrava o robô |
| **Y** | encerra |
| **A** | **salva** o episódio |
| **B** | **descarta** o episódio |
| Gatilho de grip (dedo médio) | fecha a mão inteira |
| Gatilho do indicador | pinça |
| Os dois analógicos clicados | **damping**: robô mole e travado (parada de emergência macia) |
| Teclado ↓ / ↑ (ou voz "pausar" / "continuar") | congela / destrava o robô |
| Voz "salvar" / "errei" / "finalizar" | igual ao A / B / Y |

## Ordem para gravar

1. **No robô:** o init no ar, e no painel `:8095` aperte **"Pose de gravação"**: coluna reta, braços abertos fora da cena.
2. **Rearme** a ponte, porque a pose deixa a ponte travada.
3. Rode o `init_lerobot_record_v2.py`. Os braços **ficam onde estão** (a teleop começa da pose medida do robô, não
   do "L" de juntas em zero) até você destravar com o **X**. Deve aparecer no terminal:
   `🦾 teleoperação começa da pose ATUAL do robô (29 juntas medidas)`.
4. Ponha as suas mãos na posição das mãos do robô e só então destrave, para ele não dar um tranco.
5. Faça a tarefa e aperte **A uma vez** no fim. O episódio é salvo e o próximo já começa, sem tempo de arrumar.

### O que a gravação faz por você (`init_lerobot_record_v2.py`)

- **Robô vivo ao salvar:** enquanto o LeRobot salva o episódio, uma thread continua mandando VR → robô
  ("🔁 salvando o episódio: o controle VR continua ativo").
- **Salvar/descartar por fase:**

  | Quando você aperta | A / "salvar" | B / "errei" |
  |---|---|---|
  | no episódio | encerra e **pula** o tempo de arrumar | descarta e recomeça |
  | no tempo de arrumar | encerra o tempo de arrumar | descarta |
  | durante o salvamento | ignorado, com aviso | ignorado |

  Antes, o disparo era duplo e matava o episódio seguinte vazio, derrubando o programa.
- **Episódio vazio:** é descartado em vez de quebrar.
- **Vídeos codificados durante o episódio** (`streaming_encoding: true`), então salvar é rápido. Se o notebook
  ficar lento, troque para `false`.

## Configs de gravação

| Config | Frase (`single_task`) | Pasta |
|---|---|---|
| `record_vr_wla_maca_x_preto.yaml` | Pick up the apple and place it on the black X. | `meu_dataset/maca_x_preto_2026-09-30` |
| `record_vr_wla_copo_branco.yaml` | **Pick up the white mug.** | `meu_dataset/copo_branco_2026-10-01` |
| `record_vr_wla_copo_no_coador.yaml` | **Place the white mug under the coffee strainer.** | `meu_dataset/copo_no_coador_2026-10-01` |

Todas gravam o mesmo conjunto:
- **Câmeras:** `head_stereo_left/right` (ZED) e `wrist_left/right` (D435), em 640x480.
- **Braços e cintura:** as juntas, o yaw da cintura e as Dex3.
- **Pernas** (`gravar_pernas_cintura`): as 12 juntas e o roll/pitch da cintura, medidos.
- **Locomoção** (`gravar_locomocao`): `base.vx/vy/vyaw/height`, mostrando se o robô andou.

O pegar e o colocar foram separados de propósito: com uma frase para cada um, dá para pedir as tarefas
uma por uma ou em sequência.

### Coluna reta (`cintura_reta: true`)

A trava do roll e do pitch da cintura antes só **amortecia**: a cada ciclo ela usava como alvo a posição
medida, e o tronco ficava inclinado onde o peso dos braços deixasse. Com `cintura_reta: true`, o alvo é
**0** (reto), com kp/kd 300/8. Vigie a temperatura (ver [Robô G1](Robo-G1)).

## Dicas para um dataset que funciona

- **Comece sempre da mesma pose** (a "Pose de gravação"), porque o executor vai partir dela (`--pose gravacao`).
- **Varie a posição dos objetos** de 10 a 20 cm entre episódios. Com 50 episódios numa posição só, o modelo
  decora o lugar e trava quando a cena muda (ver [Resultados](Resultados)).
- **Ninguém no campo da câmera da cabeça.**
- **Não deixe tempo parado no começo nem no fim.**
- **Confira o punho esquerdo (pitch).** No dataset da maçã ele esteve travado em 0,32 rad o tempo todo.
- O braço ceder na teleoperação **não estraga** o dataset: o conversor usa a mão medida
  (ver [Conversão e treino](Conversao-e-Treino)).
