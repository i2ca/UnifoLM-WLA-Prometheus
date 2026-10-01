# Instalação

```bash
git clone --recursive https://github.com/i2ca/UnifoLM-WLA-Prometheus ~/DEV/UnifoLM-WLA-Prometheus
```

Os submódulos em `third_party/` ficam presos nas versões que testamos:

| Submódulo | Repositório | Versão |
|---|---|---|
| `third_party/lerobot` | [Breno-de-Angelo/lerobot](https://github.com/Breno-de-Angelo/lerobot) | `d52a098b` (LeRobot 0.6.1 + câmera ZMQ) |
| `third_party/unifolm-wla` | [unitreerobotics/unifolm-wla](https://github.com/unitreerobotics/unifolm-wla) | `0a1aa87` |
| `third_party/unitree_sdk2_python` | [unitreerobotics/unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) | `a035ade` |

## Notebook (teleop, gravação, visualização, simulação)

Ambiente conda `prometheus-vla`, com Python 3.12. As versões testadas são:

| Pacote | Versão |
|---|---|
| lerobot | 0.6.1, editável, de `third_party/lerobot` |
| torch | 2.11 (cu128) |
| mujoco | 3.11.0 |
| unitree_sdk2py | 1.0.1, editável, de `third_party/unitree_sdk2_python` |
| cyclonedds | 11.0.1 |
| vuer | 0.0.60 |
| casadi | 3.7.2 |
| rerun-sdk | 0.33.1 |
| pyzmq | 27.1 |

```bash
conda create -n prometheus-vla python=3.12 -y && conda activate prometheus-vla
pip install -e third_party/lerobot -e third_party/unitree_sdk2_python
pip install mujoco==3.11.0 vuer==0.0.60 casadi rerun-sdk pynput SpeechRecognition pyzmq msgpack websockets
```

**Certificado do VR:** o óculos só abre a página por HTTPS. O certificado é gerado uma vez, dentro de `lerobot-ext/`:

```bash
cd lerobot-ext && openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key.pem -out cert.pem -subj "/CN=prometheus"
```

O `.gitignore` não deixa esse certificado entrar no repositório.

**tracelr** (opcional): precisa de Rust (`rustup`) e das bibliotecas do FFmpeg (`sudo apt install pkg-config libavcodec-dev libavformat-dev libswscale-dev libavutil-dev`).

## Robô G1 (Jetson, `unitree@192.168.123.164`)

```bash
bash lerobot-ext/robo_g1/instala_no_robo.sh          # copia para ~/Script_Prometheus_int
```

O `reenumera_zed.sh` precisa de sudo no robô. O operador instala uma vez:

```bash
sudo install -m 755 ~/Script_Prometheus_int/reenumera_zed.sh /usr/local/sbin/
echo "unitree ALL=(root) NOPASSWD: /usr/local/sbin/reenumera_zed.sh" | sudo tee /etc/sudoers.d/reenumera_zed
```

Detalhes na página [Robô G1](Robo-G1).

## PGX (DGX Spark GB10, aarch64): servidor e executor

- **Clone da Unitree:** o `unifolm-wla` vai em `~/DEV/unifolm-wla`, com o modelo base em
  `playground/Pretrained_models/UnifoLM-WLA-1.0-Base` (e o `UnifoLM-ER-1` ao lado).
- **Ambiente `wla`** (miniforge): as dependências do `uv.lock` da Unitree, mais:
  - torch 2.8 e torchvision 0.23 do índice `cu129` (aarch64);
  - `flash-attn` 2.8.3 compilado para **sm_121**, porque o DiT exige essa biblioteca;
  - `peft` 0.21.1.
- **O modelo afinado:** vai em `~/DEV/unifolm-wla/playground/Checkpoints/<run>/` (`config.yaml`,
  `dataset_statistics.json`, `checkpoints/steps_8000_model.safetensors`).

## Athena (3x A100): treino

```bash
git clone https://github.com/unitreerobotics/unifolm-wla ~/DEV/unifolm-wla && cd ~/DEV/unifolm-wla && git checkout 0a1aa87
bash ~/DEV/UnifoLM-WLA-Prometheus/lerobot-ext/wla/treino/instala_treino.sh ~/DEV/unifolm-wla
```

Na Athena:
- o ambiente conda é o `unifolm-wla`, com `peft`;
- não há `flash-attn`, então o treino usa `--framework.qwenvl.attn_implementation sdpa`, que o script já passa;
- **dados e checkpoints ficam em `/data`**, não na home. Cada checkpoint completo tem 12,5 GB e encheu o disco uma vez.
