# Resultados

## Dataset da maçã (30/09)

50 episódios de "Pick up the apple and place it on the black X.", com 27 569 quadros a 30 fps (18,4 s em média).

## Avaliação nos dados de treino (~400 trechos fixos, passo 8 000)

| Métrica | Ação = comando (30/09) | **Ação = mão medida (01/10)** |
|---|---|---|
| erro total da ação (MSE normalizado) | 0,115 | **0,027** |
| pose da mão direita | 0,52 | **0,073** |
| pose da mão esquerda | 0,13 | **0,020** |
| dedos da mão direita (erro médio, 0..1) | 0,058 | **0,034** |
| mão direita aberta/fechada certa | 94,0 % | **96,8 %** |

As duas colunas medem coisas diferentes, porque o alvo do treino mudou. Mesmo assim, o modelo com a mão
medida reproduz as demonstrações bem mais de perto.

## No robô real

| Data | Modelo | Resultado |
|---|---|---|
| 29/09 | WLA base (Dex1) | errava a maçã e levava o braço ao lugar errado (câmeras e garra diferentes das do treino) |
| 01/10 manhã | afinado, ação = comando | **mãos subindo sem parar** (+21 cm em 16 s) |
| 01/10 13:00 | afinado, ação = medida | parado numa cena diferente da do dataset; com a maçã na mão, fechou e levou ao X |
| 01/10 ~13:15 | idem, com a cena como no dataset | **pegou a maçã**; aperto forte demais (o servo da Dex3 desligou) |
| 01/10 13:27 | idem, com força dos dedos limitada | **completou a tarefa** (maçã no X); tronco parado (o executor ainda segurava a cintura) |
| 01/10 tarde | idem, com o tronco seguindo o yaw do modelo | **completou a tarefa**, com o tronco girando |

## ER-1 como juiz de conclusão

Ver [ER-1](ER-1): uma forma de perguntar e o "apontar" acertaram nos dois casos testados.
