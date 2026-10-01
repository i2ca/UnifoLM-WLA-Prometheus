# Lições aprendidas

Cada item é um problema real, de 28/09 a 01/10, com a causa e a correção.

## Dados e treino

1. **A ação tem que ser a mão MEDIDA, não o comando.** O comando da teleoperação ficava 8 a 15 cm acima da
   mão real. O modelo aprendeu "o alvo fica ~10 cm acima", e no robô as mãos subiam sem parar. O conversor
   agora usa a FK das juntas medidas (`--acao medida`). Os dedos continuam com o comando.
2. **50 episódios numa cena só não generalizam.** Com a caixa mais perto e a maçã encostada na mão, o modelo
   ficou parado: previu ~0 cm por mais de 50 consultas. Com a cena montada como no dataset, ele pegou a maçã.
   Varie os objetos de 10 a 20 cm e comece sempre da mesma pose.
3. **Confira o hardware antes de gravar.** O pitch do punho esquerdo (junta 20) esteve travado em 0,32 rad
   **em todos** os episódios da maçã. O modelo aprende a mão esquerda torta.
4. **Disco:** cada checkpoint completo tem 12,5 GB e um treino salva 4 deles. Use o `/data`, não a home.
5. **O `mse_score` do log é ruído:** ele é calculado em 4 amostras. Avalie com `avalia_checkpoints_wla.py`
   (~400 trechos fixos).
6. **O modelo afinado esquece as tarefas da Unitree.** O modelo base continua intacto (servidor `:8600`).
   Para várias tarefas, treine um modelo só com todos os datasets.

## Robô

7. **A força dos dedos derrubava o servo.** O modelo pede "fechar até o limite", e com a maçã no meio o motor
   ficava empurrando com tudo até a Dex3 desligar, e era preciso reiniciar o robô. Agora o alvo fica a no
   máximo 0,25 rad da posição medida, com kp 0,5.
8. **Calor da cintura:** os braços estendidos fazem ~13 Nm no pitch. Use o guindaste e vigie a temperatura no painel.
9. **"Coluna reta" que não era:** a trava antes só amortecia (alvo = posição medida a cada ciclo).
   `cintura_reta: true` põe o alvo em 0.
10. **Braços no "L" ao começar a gravar:** a teleop começava com todas as juntas em 0 (no G1, cotovelo a 90°).
    Agora ela começa da pose medida.
11. **Robô "morrendo" ao salvar:** o laço de controle parava no `save_episode`. Pior: o salvar disparava duas
    vezes, a segunda matava o episódio seguinte vazio e o erro derrubava o programa, que soltava as mãos.
    Ver [Teleop e gravação](Teleop-e-Gravacao).
12. **Tronco parado:** o modelo pedia até 0,25 rad de yaw da cintura, mas o executor segurava a cintura. Agora
    ele segue o yaw (`--cintura yaw`).
13. **ZED:** no modo HD vinham ~12 % de quadros corrompidos (use VGA mais o descarte). Ela também pode não
    aparecer no boot (`reenumera_zed.sh`) e precisa de um hub USB3.
14. **`pkill -f` por ssh se mata:** o padrão aparece na própria linha de comando. Use `[x]yz` no padrão.

## Execução

15. **Só o delta do modelo, sobre a pose comandada:** o braço cede sob o kp limitado. Medir o "peso" a cada
    consulta transformava atraso em erro, e a mão foi parar no rosto. Hoje há um integrador, só com o braço parado.
16. **Caixa de segurança dentro da IK, não depois:** cortar junta por junta depois da IK fazia o punho
    compensar, o "jab" do braço esquerdo.
17. **FOV da câmera da cabeça:** o dataset viu ~90 a 100°. O recorte 4:3 da ZED dava 73°, um zoom de 1,5 vez.
    Por isso existe o enquadramento `largo`.
18. **ER-1 concorda com a pergunta:** teste cada pergunta nos dois casos (ver [ER-1](ER-1)).
