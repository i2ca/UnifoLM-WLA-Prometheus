# UnifoLM-WLA-Prometheus

O UnifoLM-WLA-1.0 da Unitree no **G1 + Dex3 do Prometheus**: do VR ao robô pegando a maçã.

## O caminho inteiro

1. **[Instalação](Instalacao)**: notebook, placa do robô, PGX e Athena.
2. **[Robô G1](Robo-G1)**: init, ponte com botão de pânico, câmeras (ZED + 2 D435 nos suportes), painel `:8095`.
3. **[Teleop e gravação](Teleop-e-Gravacao)**: VR, controles, pose de gravação, coluna reta, salvar e descartar.
4. **[Ver os episódios](Visualizar-Datasets)**: `viz_episodios.py` (Rerun) e `tracelr` (as duas mãos do G1).
5. **[Conversão e treino](Conversao-e-Treino)**: formato do WLA, Dex3 como `fig6d`, **ação = mão medida**, LoRA na Athena.
6. **[Servidor e execução](Servidor-e-Execucao)**: servidor Dex3 `:8601`, executor no robô real e camadas de segurança.
7. **[Simulação](Simulacao)**: cena do café no MuJoCo, que finge ser o robô nas mesmas portas, e o ajuste das câmeras.
8. **[ER-1](ER-1)**: perguntar à visão se a tarefa foi concluída.
9. **[Lições aprendidas](Licoes-Aprendidas)**: os problemas reais, a causa e a correção. **Leia antes de gravar.**
10. **[Resultados](Resultados)**: avaliações e testes no robô.

## Máquinas e portas

| Máquina | Endereço | Faz |
|---|---|---|
| Notebook | 192.168.123.99 | teleop, gravação, visualização, simulação |
| G1 (Jetson) | 192.168.123.164 | ponte v3 + pânico, câmeras, painel |
| PGX (DGX Spark GB10) | 192.168.123.165 na rede do robô | servidor do WLA, executor, ER-1 |
| Athena (3x A100 80 GB) | rede do laboratório | conversão, treino, avaliação |

| Porta | Onde | O quê |
|---|---|---|
| 5555 | G1 | câmeras (`cameras_wla_server.py`: `head_stereo_left/right`, `wrist_left/right`) |
| 6000–6007 | G1 | ponte v3: lowcmd, lowstate, mãos, modo, rearme/pânico |
| 8095 | G1 | painel de diagnóstico (juntas, temperaturas, câmeras, pânico, pose de gravação) |
| 8080 | notebook | dashboard do teleop |
| 8090 | PGX | cabine do executor (trajetória prevista, botão parar) |
| 8600 | PGX | servidor WLA original (Dex1, da Unitree) |
| 8601 | PGX | **servidor WLA Dex3 (nosso fine-tuning)** |
| 8096 / 8098 | PGX | ER-1 decisor / ER-1 pergunta sim/não |
