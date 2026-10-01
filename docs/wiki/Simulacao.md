# Simulação (MuJoCo, cena do café)

## O "robô falso"

O [`wla/sim_cafe_wla.py`](../../lerobot-ext/wla/sim_cafe_wla.py) publica **as mesmas portas ZMQ do robô
real** (câmeras 5555, ponte 6000–6007), com os mesmos limites da ponte v3. Assim, o `roda_wla_real.py` roda
contra ele **sem nenhuma mudança**, com o modelo na PGX ou na Athena:

```bash
python lerobot-ext/wla/sim_cafe_wla.py                       # janela do MuJoCo + portas
python lerobot-ext/wla/sim_cafe_wla.py --foto /tmp/sim.jpg   # só as 4 câmeras na pose de partida
ssh -N -L 8601:localhost:8601 <usuario>@<maquina do servidor>
python lerobot-ext/wla/roda_wla_real.py --robo 127.0.0.1 --servidor ws://127.0.0.1:8601 --pose maca --voz nenhuma
```

A cena ([`unitree-g1-mujoco/assets/scene_cafe_wla.xml`](../../unitree-g1-mujoco/assets/scene_cafe_wla.xml)) é
a do café, ajustada ao dataset:
- **Mesa:** baixada para a altura do dataset (tampo 11 cm abaixo da pelvis), na cor de papelão.
- **Objetos:** a **maçã** (7,6 cm e 150 g, dá para pegar), o **X preto**, a caneca e o coador.
- **Robô:** a pelvis fica presa, como no guindaste.
- **Pose de partida:** o keyframe `maca` deixa o robô na pose do início do dataset.

A imagem simulada é diferente da real, então o modelo treinado com imagens reais não deve ser julgado
pelo desempenho no simulador. O simulador serve para testar o **encanamento**: o protocolo, as convenções
de pose, a IK, os dedos e a segurança.

## Câmeras no XML (ZED + D435 nos suportes)

Em [`g1_29dof_with_hand_wla.xml`](../../unitree-g1-mujoco/assets/g1_29dof_with_hand_wla.xml), procure "EDITE AQUI":

| Body | O quê |
|---|---|
| `zed_cabeca` | malha oficial da ZED 1; olhos a ±6 cm (12 cm entre as lentes), 672x376 a 90° |
| `suporte_punho_esquerdo` | o **suporte impresso** (`suporte_d435_punho.stl`, em mm); move o conjunto |
| `d435_punho_esquerdo` | a D435 em cima do suporte (malha oficial); move só a câmera |
| `suporte_punho_direito` / `d435_punho_direito` | o mesmo, com o suporte **espelhado em X** (`scale="-0.001 0.001 0.001"`) |

Os eixos são os do punho: **x** para os dedos, **z** para o lado do indicador e **y** para o polegar. O `euler` está em radianos.

### Ajustar pelo teclado, vendo o que a IA vê

```bash
python lerobot-ext/wla/ajusta_cameras.py
```

Abre o viewer oficial do MuJoCo e uma janela com as 4 câmeras como o modelo as recebe. Clique na janela das câmeras e use as teclas:

| Tecla | Faz |
|---|---|
| `1` / `2` / `3` | peça do suporte / câmera / conjunto, do lado **esquerdo** |
| `4` / `5` / `6` | o mesmo, do lado **direito** |
| `7` | ZED |
| `W` `S`, `A` `D`, `Q` `E` | move em x, y, z |
| `I` `K`, `J` `L`, `U` `O` | muda o 1º, 2º e 3º número do `euler` |
| `+` / `-` | passo maior / menor |
| **`G`** | grava no XML (no mesmo formato `pos`/`euler` que se edita à mão) |
| `R` | recarrega do XML |

### Ver o XML no VS Code

A extensão **MuJoCo Viewer** (`julienblanchon.mujoco-viewer`) mostra a cena e atualiza ao editar o XML.
Abra o `scene_cafe_wla.xml` e use "Open with MuJoCo Viewer to the Side".

Se ela mostrar `mjCError` ou "memory access out of bounds" sem mais detalhes, o motivo costuma ser **um erro
no XML**, como uma aspa faltando. A mensagem não é de memória.

## Isaac (unitree_sim_env)

O `wla/ponte_wla_isaac.py` liga o WLA ao simulador Isaac da Unitree (tarefa `PickPlace-RedBlock`, DDS domínio 1).
Use o ambiente `g1` com cyclonedds 0.10.2, porque o 11 derruba o simulador.
