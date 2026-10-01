# lerobot-ext (parte WLA do Prometheus)

Mesma estrutura do `lerobot-ext/` do [prometheus-vla](https://github.com/i2ca/prometheus-vla), só com o que o
UnifoLM-WLA usa. Mantida igual de propósito: os scripts acham uns aos outros por caminho relativo.

| | |
|---|---|
| `robot/unitree_g1/` | plugin LeRobot do G1 + Dex3 (`cintura_reta`, câmeras WLA, locomoção e pernas no dataset) |
| `robot/Scripts_Prometheus_int/` | servidores antigos do robô (ponte v2, base da v3) e `sim/sensor_utils.py` (ZMQ das câmeras) |
| `teleop/` | teleoperação VR (`xr_g1_arm.py`), televuer, teleimager, IK e retargeting das mãos |
| `config/teleop`, `config/record` | configs de teleop e das gravações do WLA |
| `init_lerobot_record_v2.py` | gravação com voz/teclado/VR, robô vivo ao salvar, salvar/descartar por fase |
| `init_lerobot_teleoparate_v2.py` | só teleoperação |
| `viz_episodios.py` | navegador de episódios no Rerun |
| `tracelr/` | visualizador Rust com a trajetória das duas mãos do G1 (`ver_dataset.sh`) |
| `robo_g1/` | **roda no robô**: init, ponte v3 + pânico, câmeras, painel `:8095` (`instala_no_robo.sh`) |
| `pontes/cabine/` | a cabine (página `:8090`) usada pelo executor e pela sombra |
| `wla/` | servidor Dex3, executor no robô, sombra, gravador/relatório/replay, ER-1, simulador, câmeras |
| `wla/treino/` | conversão, configs e script do LoRA, avaliação, delta (`instala_treino.sh`) |
| `assets/g1/` | URDF e malhas do G1 (FK do `fk_g1.py` e do tracelr) |

Documentação: [`../docs/wiki/`](../docs/wiki/Home.md).
