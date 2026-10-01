#!/usr/bin/env python
"""Servidor MCP do Prometheus — o que a super IA enxerga do robô.

    python -m pontes.cabine.mcp_prometheus --porta 8091 --cabine http://127.0.0.1:8090

A super IA (GPT-6 Astra, Claude, o que for) aponta o cliente MCP dela para
`http://<ip-desta-maquina>:8091/mcp` e ganha quatro ferramentas: mandar tarefa,
olhar a cena, ler o estado e parar. É o desenho do cronograma do IJCNN: ela gera
a sequência de tarefas, manda cada uma por texto e acompanha por imagem.

── Processo separado do cérebro, de propósito ──────────────────────────────
Este servidor só fala HTTP com a cabine. Ele não importa MuJoCo, não carrega
rede neural nenhuma e sobe em menos de um segundo. Dá para reiniciá-lo, trocar
de porta ou pô-lo noutra máquina sem tocar no simulador que está no meio de um
episódio. O preço é um salto de rede por chamada, que é irrelevante ao lado dos
segundos que um movimento leva.

── O que ele NÃO faz ───────────────────────────────────────────────────────
Não bloqueia esperando a tarefa acabar. `definir_tarefa` volta na hora, e quem
quiser saber se terminou chama `estado` ou `ver_cena` — que é exatamente o
"monitora por imagem a sua execução" do plano. Um tool que bloqueasse por 20 s
estouraria o tempo limite da maioria dos clientes MCP.

── Segurança ───────────────────────────────────────────────────────────────
Sem autenticação: é laboratório. A proteção contra DNS rebinding do MCP 2.x vem
LIGADA e recusa com 421 qualquer `Host` que não esteja na lista — e a lista não
aceita curinga, `allowed_hosts=["*"]` não funciona (testado: ele recusou até o
próprio 127.0.0.1:8091). Como a super IA vem de outra rede e o `Host` dela não é
previsível, a proteção fica desligada aqui. Isso é aceitável numa LAN fechada e
NÃO é aceitável na internet: se um dia isto sair do laboratório, o caminho é
`auth=AuthSettings(...)` mais `allowed_hosts` com os nomes de verdade.
"""
from __future__ import annotations

import argparse
import base64
import json
import urllib.error
import urllib.request

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ImageContent

CABINE = "http://127.0.0.1:8090"
TEMPO_LIMITE = 10.0


def _pega(caminho: str) -> bytes:
    with urllib.request.urlopen(f"{CABINE}{caminho}", timeout=TEMPO_LIMITE) as r:
        return r.read()


def _poe(caminho: str, corpo: dict) -> dict:
    req = urllib.request.Request(
        f"{CABINE}{caminho}", method="POST",
        data=json.dumps(corpo).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=TEMPO_LIMITE) as r:
        return json.loads(r.read())


mcp = MCPServer(
    name="prometheus",
    instructions=(
        "Robô humanoide Unitree G1 (real ou simulado no MuJoCo) da equipe Prometheus. "
        "Mande UMA tarefa por vez em linguagem natural com `definir_tarefa`; ela volta "
        "imediatamente, sem esperar o movimento terminar. Para saber se terminou, chame "
        "`estado` e leia `fase` e `ultimo_resultado`, ou `ver_cena` para olhar. "
        "O robô hoje sabe pegar uma xícara branca e pô-la sob o filtro de café."
    ),
)


@mcp.tool()
def definir_tarefa(texto: str) -> str:
    """Manda o robô executar uma tarefa descrita em linguagem natural.

    Volta na hora, sem esperar o movimento acabar. Exemplos que ele conhece:
    "pegue a xícara branca", "ponha a xícara no filtro", "faça o café".

    Args:
        texto: a tarefa, em uma frase.
    """
    try:
        r = _poe("/tarefa", {"texto": texto})
    except urllib.error.URLError as e:
        return f"a cabine não respondeu ({e}). O simulador está de pé?"
    return f"tarefa {texto!r} aceita (pedido nº {r['tarefa_seq']}). Use `estado` para acompanhar."


@mcp.tool()
def estado() -> str:
    """Lê o estado do robô: fase atual, juntas dos braços, onde está a xícara,
    qual motor está no laço e o resultado da última tarefa. Devolve JSON."""
    try:
        return _pega("/estado.json").decode()
    except urllib.error.URLError as e:
        return json.dumps({"erro": f"a cabine não respondeu: {e}"})


@mcp.tool()
def ver_cena(camera: str = "head_camera") -> ImageContent:
    """Devolve uma foto do que o robô está vendo agora.

    Args:
        camera: `head_camera` (a da cabeça, que é o que a política vê),
            `right_wrist_camera` (a da mão direita) ou `global_view`
            (terceira pessoa, boa para julgar se a tarefa deu certo).
    """
    dado = _pega(f"/quadro/{camera}.jpg")
    return ImageContent(type="image", mimeType="image/jpeg",
                        data=base64.b64encode(dado).decode())


@mcp.tool()
def parar() -> str:
    """Interrompe o movimento em andamento. Use se algo parecer errado na imagem."""
    try:
        _poe("/parar", {})
    except urllib.error.URLError as e:
        return f"a cabine não respondeu ({e})."
    return "parada pedida; o robô interrompe no próximo passo."


def main():
    global CABINE
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--porta", type=int, default=8091)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--cabine", default=CABINE, help="onde o cerebro está servindo")
    args = p.parse_args()
    CABINE = args.cabine.rstrip("/")

    print(f"MCP do Prometheus em http://<ip>:{args.porta}/mcp")
    print(f"   falando com a cabine em {CABINE}")
    mcp.run(
        transport="streamable-http", host=args.host, port=args.porta,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False),
    )


if __name__ == "__main__":
    main()
