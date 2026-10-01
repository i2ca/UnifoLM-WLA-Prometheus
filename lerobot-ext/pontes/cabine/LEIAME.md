# Cabine — painel web e MCP do Prometheus

Três processos pequenos que, juntos, deixam uma pessoa (pelo navegador) e uma
super IA (pelo MCP) mandarem tarefas em texto no robô e acompanharem por imagem.

```
  navegador ─┐
  celular  ──┼──HTTP──►  servidor.py  ◄──HTTP──  mcp_prometheus.py  ◄──MCP──  super IA
  TV       ──┘              ▲   │                                            (outra rede)
                            │   │ tarefa
                    quadros │   ▼
                         cerebro.py ──► MuJoCo + motor (roteirizado ou WLA)
```

## Subir

```bash
python -m pontes.cabine.cerebro --porta 8090                      # simulador + painel
python -m pontes.cabine.mcp_prometheus --porta 8091 --cabine http://127.0.0.1:8090
```

Painel em `http://<ip>:8090/`. MCP em `http://<ip>:8091/mcp`.

Para gravar vídeo, `--dorme 0.004`: sem isso um episódio inteiro passa em ~4 s e
o movimento fica picotado na tela.

## Ferramentas que a super IA enxerga

| ferramenta | o que faz |
|---|---|
| `definir_tarefa(texto)` | manda a tarefa e **volta na hora**, sem esperar |
| `estado()` | fase, juntas, onde está a xícara, resultado da última tarefa |
| `ver_cena(camera)` | uma foto: `head_camera`, `right_wrist_camera`, `global_view` |
| `parar()` | interrompe o movimento no próximo passo |

`definir_tarefa` não bloqueia de propósito: um movimento leva ~20 s em tempo real
e a maioria dos clientes MCP estoura o tempo limite bem antes. O desenho é o do
cronograma do IJCNN — a super IA manda a tarefa e **monitora por imagem**.

## Três coisas que custaram tempo

1. **A proteção contra DNS rebinding do MCP 2.x** recusa com 421 qualquer `Host`
   fora da lista, e a lista **não aceita curinga**: `allowed_hosts=["*"]` recusou
   até o próprio `127.0.0.1:8091`. Como a super IA vem de outra rede, ela fica
   desligada (`enable_dns_rebinding_protection=False`). Vale numa LAN fechada e
   não vale na internet.

2. **O MCP 2.x renomeou tudo para snake_case.** `FastMCP` virou `MCPServer`,
   `streamablehttp_client` virou `streamable_http_client` e ele devolve DUAS
   coisas, não três; `serverInfo` virou `server_info`, `inputSchema` virou
   `input_schema`, `mimeType` virou `mime_type`. Código de tutorial v1 não roda.

3. **O IK vem do `demo_pega_copo_mujoco`, não do módulo original.** O demo faz
   um remendo no módulo antes de expor a classe; importar `G1_29_ArmIK` direto
   de `robot.unitree_g1.robot_control.g1_arm_ik` pula o remendo e quebra na
   assinatura.

## O motor é trocável, e é esse o ponto

`--motor roteirizado` (padrão) usa o especialista que gravou os 1.828 episódios:
confiável, sem checkpoint, sem GPU. É o que torna a demonstração da super IA
possível antes de a rede estar boa.

`--motor wla` põe a UnifoLM-WLA no lugar. **Ainda não fecha o laço**: falta
desnormalizar a saída (54 dims, `minmax_q`) com as estatísticas do treino e
remapear para as nossas juntas — o inverso do `pontes/wla/converte_dataset_wla.py`.

O painel mostra `head_camera` e `head_camera_antiga` lado a lado porque o
dataset que treinou a rede foi gravado com a pose errada da câmera. Enquanto for
assim, é a antiga que casa com o que a política viu.
