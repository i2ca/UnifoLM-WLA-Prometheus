# ER-1: perguntar à visão se a tarefa terminou

O **UnifoLM-ER-1** (Qwen3-VL-4B de raciocínio espacial da Unitree) é um modelo de imagem e texto
**separado** do WLA. O WLA não diz quando terminou; o ER-1 pode olhar a câmera e responder.

```bash
python lerobot-ext/wla/er1_pergunta.py          # carrega (~55 s) e fica na memória, porta 8098
curl "http://127.0.0.1:8098/pergunta?q=Is%20there%20a%20mug%20directly%20below%20the%20coffee%20strainer%3F"
curl "http://127.0.0.1:8098/pergunta?q=...&livre=1"     # resposta em texto livre (ou apontar objetos)
```

- **Resposta:** em ~0,2 s para sim/não, ou ~2,9 s para apontar objetos.
- **Registro:** cada pergunta fica salva em `~/er1_perguntas/`, com a imagem e a resposta.
- **Disputa a GPU com o WLA:** pergunte a cada poucos segundos, não a cada consulta. Rodar o decisor a cada
  3 s deixou o WLA 2,5 vezes mais lento.

## O que testamos (caneca embaixo / fora do coador, 01/10)

| Forma | Caneca retirada | Caneca embaixo | Confiável? |
|---|---|---|---|
| *Is the white cup under the coffee strainer?* | yes ❌ | yes ✅ | não: diz sempre "yes" |
| ***Is there a mug directly below the coffee strainer?*** | **no ✅** | **yes ✅** | **sim** |
| *Is the coffee strainer standing on its own base…, with no cup under it?* | yes ✅ | yes ❌ | não |
| *Is the white mug next to the coffee strainer, not under it?* | yes ✅ | yes ❌ | não |
| ***Apontar*** a caneca e o coador e aplicar uma regra geométrica | não embaixo ✅ (dx 180/1000) | embaixo ✅ (dx 30/1000) | **sim** |

**Ele tende a concordar com a pergunta.** Antes de usar uma pergunta, teste-a nos **dois** casos (feito e
não feito). Para checar a conclusão no executor:
- exija que a pergunta **e** o apontar concordem;
- exija duas confirmações seguidas.

Para o "pegar", os sensores bastam: dedos fechados e parados antes do fim, e a mão levantada.

## Decisor

O `er1_decisor.py` (`:8096`) olha a cena, localiza os objetos e **escolhe a próxima frase** para o WLA
(por exemplo "pegar" e depois "colocar"). Ele não comanda junta nenhuma.
