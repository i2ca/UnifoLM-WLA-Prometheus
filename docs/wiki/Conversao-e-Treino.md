# Conversão do dataset e fine-tuning LoRA (Athena)

Arquivos em [`lerobot-ext/wla/treino/`](../../lerobot-ext/wla/treino). O `instala_treino.sh` liga esses
arquivos (configs e scripts) dentro do clone do `unifolm-wla` por link simbólico.

## 1. Converter para o formato do WLA

```bash
python lerobot-ext/wla/treino/converte_dataset_dex3_wla.py \
    meu_dataset/maca_x_preto_2026-09-30 /data/<usuario>/unifolm_data/Prometheus_G1_Dex3_Medido_Dataset/G1_Dex3_Maca_X_Preto
```

| Coluna WLA | De onde vem |
|---|---|
| `*_ee_pose_gripper_base` (estado **e ação**) | FK (`wla/fk_g1.py`, conferida com 2 mm de diferença) das juntas **MEDIDAS** do braço: xyz + rpy `xyz` na pelvis |
| `*_fig6d` (ação) | Dex3 **comandada**, em 0..1 na ordem [polegar1, polegar2, indicador0, indicador1, médio0, médio1] |
| `*_fig6d` (estado) | Dex3 medida |
| `waist_*_joint` | cintura **medida** |
| `base_command` | `base.vx/vy/vyaw/height` (se o robô andou) |
| `*_leg` | pernas medidas (quem comanda é o WBC) |
| vídeos | copiados como estão (`head_stereo_left`, `wrist_left`, `wrist_right`) |

**A mão nunca é reduzida a garra (Dex1):** ela vai como `fig6d`, o espaço de mão de dedos do próprio WLA.
A rotação do polegar (polegar0) fica de fora, porque nos dados ela era sempre 0.

### Por que a ação é a mão MEDIDA (`--acao medida`, o padrão)

O WLA aprende a ação **relativa à pose atual**: `T_estado(t)⁻¹ · T_ação(t+k)`. No primeiro dataset, a ação
era o **comando** da teleoperação, que ficava **8 a 15 cm acima da mão real**. Os motivos eram o punho
esquerdo travado, o braço cedendo e o comando adiantado durante o movimento. O modelo aprendeu "o alvo
fica ~10 cm acima de onde a mão está", e no robô **as mãos subiam sem parar**. Com a mão medida, o
deslocamento em `k=0` é zero e o modelo aprende o movimento real. Os **dedos** continuam com o comando:
para segurar, o dedo é mandado além do ponto de contato. A opção `--acao comando` reproduz o dataset antigo.

## 2. Treinar o LoRA

```bash
cd ~/DEV/unifolm-wla
GPU=0,1,2 NPROC=3 LOTE=4 PASSOS=8000 \
DADOS=./unifolm_wla/dataloader/multi_source_dataset/configs/prometheus_dex3_medido.yaml \
RUN_ID=lora_prometheus_dex3_maca_medido bash treina_lora_prometheus_dex3.sh
```

- **Receita da Unitree para fine-tuning** (`lora_prometheus_dex3.yaml`):
  - o VLM (Qwen3-VL) fica **congelado**;
  - o LoRA (r=16, alpha=32) vai nas atenções do **DiT**, e as entradas e saídas dele treinam inteiras (~24 M parâmetros);
  - o `robot_state_projector` vem do checkpoint afinado.
- **Normalização:** usa as estatísticas pré-calculadas da Unitree, porque as colunas têm os nomes deles; o
  relativo usa z-score. A chave da normalização é `Prometheus_G1_Dex3`.
- **Atenção:** SDPA, porque a Athena não tem `flash_attn`.
- **Tempo:** 8 000 passos levam ~1h54 em 3 A100 (lote 12) ou ~1h30 em 2 (lote 8). Com 2 ou 3 GPUs a vazão
  é quase a mesma; com 3, cada passo vê mais amostras.
- **Espaço em disco:** um checkpoint completo a cada 2 000 passos, de **12,5 GB cada**. Use o `/data`.
- **Sem wandb:** `WANDB_MODE=disabled`.

## 3. Avaliar

```bash
CUDA_VISIBLE_DEVICES=2 python avalia_checkpoints_wla.py \
    --run playground/Checkpoints/lora_prometheus_dex3_maca_medido --passos 4000 8000 --lotes 50
```

O script mede, em ~400 trechos fixos do dataset (mesma semente para todos os checkpoints):
- o erro da ação normalizada por parte do corpo;
- o erro dos dedos da mão direita;
- a % de passos em que a mão direita está aberta/fechada certo.

Ele mede o **ajuste às demonstrações**, não a generalização, porque os episódios são os do treino. Para
medir generalização, grave alguns episódios que não entram no treino.

## 4. Levar à PGX

- **Mesma rede:** copie o checkpoint inteiro (`steps_8000_model.safetensors`, `config.yaml`,
  `dataset_statistics.json`) com `rsync`.
- **Pela VPN:** use `extrai_delta_lora.py`, que guarda só o que mudou em relação ao modelo base (LoRA e
  partes do DiT, algumas dezenas de MB). O servidor aceita `steps_<N>_delta.safetensors` no lugar do modelo inteiro.

## Várias tarefas num modelo só

Treine **um** LoRA com todos os datasets na config de dados, um bloco por dataset, cada um com a sua
frase. O modelo escolhe a tarefa pela frase. Cada tarefa nova pede um novo treino com todas (~2 h). Para
não esquecer as tarefas da Unitree, misture uma parte do dataset deles com um peso menor.
