# ER-1: perguntar à visão o que está acontecendo

O **UnifoLM-ER-1** (Qwen3-VL-4B de raciocínio espacial da Unitree) é um modelo de imagem e texto
**separado** do WLA: o WLA move o braço, o ER-1 **olha a câmera e responde**. No Prometheus ele faz três
coisas:

1. **"A tarefa terminou?"**: o executor pergunta a cada ~8 s e o robô diz *"Task completed."* depois de dois
   "sim" seguidos.
2. **Narração**: diz o objeto mais perto da mão direita, que entra na frase montada pelos sensores (o
   quanto a mão andou, dedos abertos/fechados/segurando).
3. **Tradução** das falas para PT-BR no histórico da cabine.

Para as perguntas sim/não usamos um **LoRA treinado na nossa cena** (acerto de 74,5% → 94,9%, ver
[Treinar o LoRA](#treinar-o-lora-do-er-1-na-athena)). A narração e a tradução usam o ER-1 original: o
adaptador é ligado e desligado por pergunta, sem carregar dois modelos.

## Subir na PGX

```bash
bash ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/sobe_er1_pergunta.sh          # porta 8098, ~60 s
```

- Carrega o ER-1 de `~/DEV/unifolm-wla/playground/Pretrained_models/UnifoLM-ER-1` (um link para o cache do
  Hugging Face) e o adaptador de `~/DEV/unifolm-wla/playground/Checkpoints/er1_lora_prometheus/adaptador_ep2`.
- O log fica em `~/er1_pergunta.log`. Está pronto quando aparecer `🧠 ER-1 pergunta sim/não em http://0.0.0.0:8098`.
- `--rgb` vai ligado: a câmera da cabeça já chega em RGB. Sem ele, a maçã vermelha aparece **azul** para o ER-1.
- Só o ER-1 original: `bash sobe_er1_pergunta.sh --adaptador ""`.

**Com o robô (cabine):** a cabine sobe o ER-1 sozinha quando você pede:

```bash
NARRA=1 bash ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/sobe_cabine_wla.sh --dedos-estado medido
```

Sem `NARRA=1` o executor roda **sem ER-1** e o WLA fica sozinho na GPU.

**Conferir se está no ar:**

```bash
(echo > /dev/tcp/127.0.0.1/8098) 2>/dev/null && echo "ER-1 no ar" || echo "ER-1 fora"
```

## Perguntar (API HTTP)

O servidor pega o quadro **mais novo** da câmera do robô (`cameras_wla_server`, porta 5555) a cada pergunta.

```bash
# sim/não (com o adaptador): "yes"/"no" em ~0,2 s
curl "http://127.0.0.1:8098/pergunta?q=Is%20the%20robot's%20right%20hand%20holding%20the%20white%20mug%3F"

# texto livre (ER-1 original), câmera do punho direito
curl "http://127.0.0.1:8098/pergunta?q=What%20object%20is%20closest%20to%20the%20robot's%20right%20hand%3F&livre=1&max=8&cam=wrist_right"

# tradução para PT-BR (só texto)
curl "http://127.0.0.1:8098/traduz?t=I%20will%20pick%20up%20the%20white%20mug."

# a imagem da última pergunta
curl -o ultima.jpg http://127.0.0.1:8098/ultima.jpg
```

| Parâmetro | O quê |
|---|---|
| `q` | a pergunta, em inglês |
| `cam` | `head_stereo_left` (padrão), `wrist_right`, `wrist_left`; várias separadas por vírgula (vão juntas na mesma pergunta) |
| `livre=1` | resposta em texto livre. Sem ele a pergunta ganha "Answer only yes or no." e a resposta vira `yes`/`no` |
| `max` | tamanho máximo da resposta livre, em tokens |
| `escala` | reduz a imagem (ex.: 0,5) para responder mais rápido |
| `lora=0/1` | força o adaptador ligado/desligado. Padrão: ligado no sim/não, desligado no texto livre |

Cada pergunta fica salva em `~/er1_perguntas/` (imagem + JSON com a resposta).

**A pergunta "terminou?" do executor** usa exatamente o formato do treino do LoRA:

```
The robot's task is: "Pick up the white mug." Has the robot completed this task? Answer only yes or no.
```

Mude o formato e o acerto cai: o adaptador aprendeu essas frases.

**Disputa a GPU com o WLA.** O executor faz as perguntas numa fila com o WLA, um de cada vez. Mesmo assim,
pergunte a cada poucos segundos, não a cada consulta.

## Treinar o LoRA do ER-1 na Athena

### Onde está tudo

| O quê | Onde |
|---|---|
| Código que rodou (cópia do repositório) | Athena `~/er1_lora/`: `gera_dataset_er1.py`, `treina_lora_er1.py`, `varre_episodio_er1.py`, `er1_lora.sh`, `er1_varre.sh` |
| Código no repositório | `lerobot-ext/wla/er1/` (os mesmos scripts + `treina_er1_athena.sh`, que faz tudo) |
| ER-1 original | Athena `/data/mrwlker/UnifoLM-ER-1` |
| Datasets originais (LeRobot v3) | Athena `/data/mrwlker/datasets_prometheus/` (5 datasets) |
| Perguntas geradas | Athena `/data/mrwlker/er1_lora/dados/`: `treino.jsonl` (3 084), `teste.jsonl` (564), `quadros/*.jpg` |
| Treino de 02–03/10 | Athena `/data/mrwlker/er1_lora/run1/`: `adaptador_ep1`, **`adaptador_ep2`** (o que está na PGX), `avaliacao_{base,ep1,ep2}.json`, `geral_*.jsonl` |
| Log do treino | Athena `/data/mrwlker/er1_lora/run1.log` |
| Linhas do tempo (episódio inteiro, original × LoRA) | Athena `/data/mrwlker/er1_lora/varredura_ep1/linha_do_tempo_*.png` |
| Ambiente Python | Athena `~/miniconda3/envs/unifolm-wla` (transformers 5.5, peft 0.21) |
| Adaptador em uso | PGX `~/DEV/unifolm-wla/playground/Checkpoints/er1_lora_prometheus/adaptador_ep2` |

### Como funciona

1. **`gera_dataset_er1.py`**: tira quadros do **começo** (tarefa não feita) e do **fim** (tarefa feita) de
   cada episódio e monta perguntas sim/não com a resposta certa, sem anotação à mão: tarefa concluída?,
   mão segurando a caneca ou a maçã?, maçã no X?, caneca embaixo do coador? Separa ~15% dos **episódios**
   só para teste. Usa 64 processos (CPU), ~10 s.
2. **`treina_lora_er1.py`**: LoRA só nas camadas de **linguagem** (a visão fica intacta). Para não esquecer
   o que já sabia, o ER-1 original responde perguntas gerais nas nossas imagens e essas respostas entram
   no treino (auto-destilação). Mede antes e depois: acerto nas perguntas de teste e **retenção**
   (concordância com o original nas perguntas gerais).
3. **`varre_episodio_er1.py`**: pega um episódio de teste de cada tarefa, pergunta quadro a quadro (1 s) às
   duas versões e desenha a linha do tempo.

### Rodar de novo (ex.: com datasets novos)

```bash
ssh mrwlker@10.9.8.252
screen -S er1_lora
cd ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/er1     # ou ~/er1_lora (a cópia que rodou)
GPU=1 DADOS=/data/mrwlker/er1_lora/dados_v2 bash treina_er1_athena.sh
```

- **Datasets novos:** acrescente em `DATASETS`, dentro de `treina_er1_athena.sh`, e diga em `PERGUNTAS`
  (`gera_dataset_er1.py`) o que vale no começo e no fim daquela tarefa.
- **`DADOS` novo** gera as perguntas de novo. Se a pasta já existir, ele reaproveita.
- **Saída:** `/data/mrwlker/er1_lora/run_<data>/`. Escolha a época pelo `avaliacao_epN.json`: maior
  `total` sem a `retencao_sim_nao_geral` cair muito.
- Para sair da screen sem parar o treino: Ctrl+A e depois D. Para voltar: `screen -r er1_lora`.

### Levar para a PGX

```bash
# na Athena (a PGX na rede do laboratório, 10.9.8.80)
scp -r /data/mrwlker/er1_lora/run_<data>/adaptador_ep2 \
    mrwlker@10.9.8.80:DEV/unifolm-wla/playground/Checkpoints/er1_lora_prometheus/adaptador_novo
# na PGX
bash ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/sobe_er1_pergunta.sh \
    --adaptador ~/DEV/unifolm-wla/playground/Checkpoints/er1_lora_prometheus/adaptador_novo
```

### Resultados (run1, 564 perguntas de episódios nunca vistos)

| | ER-1 original | LoRA época 1 | **LoRA época 2** |
|---|---|---|---|
| Total | 74,5% | 94,3% | **94,9%** |
| Retenção geral | 100% | 97,1% | **97,9%** |

Por tarefa, o maior ganho foi em maçã → X (60% → 99%) e em "tarefa concluída?" (68% → 95%).

**Limitação conhecida:** as perguntas vêm só do começo e do fim dos episódios. O meio (carregar o objeto)
não entrou no treino, e na linha do tempo da maçã → X o LoRA errou o "segurando" durante o carregamento.

## Próximos passos

- **Rotular o episódio inteiro pelos sensores**: dedos fechados no objeto + mão subindo = "segurando", também
  no meio. Isso resolve a limitação acima e gera frases de narração ("lowering my right hand toward the
  mug") para o ER-1 aprender a descrever a etapa sem inventar.
- **Apontar o alvo + profundidade da ZED** para o ajuste fino da pegada no executor: o ER-1 aponta a caneca,
  a estéreo dá a posição 3D, o executor corrige os últimos centímetros.
- **ER-1 como planejador**: quebrar "fazer café" em etapas e checar cada uma com as perguntas acima.
- As mesmas perguntas já entram no **co-treino do WLA** (`treina_cotreino_wla.py`), ver
  [Conversão e treino](Conversao-e-Treino).

## Testes antigos (sem LoRA, 01/10)

Pergunta "a caneca está embaixo do coador?", com a caneca retirada e com a caneca embaixo:

| Forma | Caneca retirada | Caneca embaixo | Confiável? |
|---|---|---|---|
| *Is the white cup under the coffee strainer?* | yes ❌ | yes ✅ | não: diz sempre "yes" |
| ***Is there a mug directly below the coffee strainer?*** | **no ✅** | **yes ✅** | **sim** |
| *Is the coffee strainer standing on its own base…, with no cup under it?* | yes ✅ | yes ❌ | não |
| ***Apontar*** a caneca e o coador e aplicar uma regra geométrica | não embaixo ✅ | embaixo ✅ | **sim** |

O ER-1 original **tende a concordar com a pergunta**: antes de usar uma pergunta nova, teste nos dois casos.
O `er1_decisor.py` (porta 8096, `sobe_er1.sh`) é a versão antiga, que escolhia a próxima frase para o WLA.
