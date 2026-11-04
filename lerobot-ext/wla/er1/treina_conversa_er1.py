#!/usr/bin/env python
"""LoRA de CONVERSA do ER-1 (30/10): o Prometheus guarda nos pesos o que hoje vem da memória (RAG) e as correções.

DESTILAÇÃO DE CONTEXTO, em 3 fases (cada uma grava e é pulada se já existir):
  gera    o ER-1 ORIGINAL lê cada item da memória, escreve perguntas como uma pessoa falaria com o robô e as responde
          COM o trecho no prompt (o "professor"). Também responde perguntas gerais sem trecho (retenção: o adaptador
          não pode piorar o resto). Divide entre GPUs: --parte i --partes n.     -> <saida>/gera_<i>.jsonl
  treino  o "aluno" (ER-1 + LoRA nas camadas de linguagem) aprende a mesma resposta SEM o trecho (metade das vezes) e
          COM ele (a outra metade: na cabine o RAG continua ligado); correções da memória entram 3x.
                                                                                  -> <saida>/adaptador_epN
  avalia  perguntas guardadas fora do treino (1 de cada item): ER-1 sem trecho vs. LoRA sem trecho vs. LoRA com
          trecho, medindo quanto da resposta do professor ele reproduz (sobreposição de palavras) -> avaliacao.json

    # na Athena (GPUs 0 e 1): treina_conversa_athena.sh
    CUDA_VISIBLE_DEVICES=0 python treina_conversa_er1.py --fase gera --parte 0 --partes 2 --db memoria.db --saida RUN
"""
import argparse
import json
import math
import random
import re
import sqlite3
import time
import unicodedata
from pathlib import Path

import torch

ER1 = "/data/mrwlker/UnifoLM-ER-1"
PERSONA = ("Você é o Prometheus, um robô humanoide criado pela equipe do LCAD (Laboratório de Computação de Alto "
           "Desempenho) da UFES. Você fala português do Brasil, sempre em primeira pessoa, como um assistente de voz "
           "simpático e natural: frases curtas, tom de conversa, sem listas nem símbolos, e responde exatamente ao que "
           "perguntaram.\n")
DESLIGADO = ("O seu corpo de robô está DESLIGADO agora: só o seu cérebro, no computador, está ligado. Você não enxerga "
             "nada (câmeras desligadas), não mexe os braços e está conversando pelo computador. Se perguntarem o que "
             "você vê ou faz, diga isso com naturalidade.\n")
EXEMPLOS = ("Exemplos de conversa:\nUsuário: Boa tarde!\nPrometheus: Boa tarde! Tudo bem com você?\n"
            "Usuário: Quem é você?\nPrometheus: Eu sou o Prometheus, um robô humanoide criado pela equipe do LCAD, da "
            "UFES.\nUsuário: Obrigado!\nPrometheus: De nada! Precisando, é só chamar.\n\n")
# retenção: o jeito geral de conversar do ER-1 original (sem trecho da memória) não pode mudar
GERAIS = """Bom dia! Tudo bem?|Boa noite, Prometheus.|Como você está hoje?|Qual é o seu nome?|Você gosta de conversar?
Me conta uma curiosidade.|Quanto é sete vezes oito?|Quanto é cem dividido por quatro?|Qual é a capital do Brasil?
Qual é a capital do Espírito Santo?|Me dá uma dica para estudar melhor.|O que é inteligência artificial?
O que é um robô humanoide?|Você sabe contar até cinco?|Qual é a cor do céu num dia limpo?|Me conta uma piada.
Que dia vem depois de sexta-feira?|Quantos minutos tem uma hora?|O que é uma rede neural?|Você dorme?
Você tem sentimentos?|Do que você mais gosta?|Me ajuda a escolher um nome para um gato.|O que é visão computacional?
Por que o céu é azul?|Como faço para economizar energia em casa?|Qual a diferença entre chá e café?
Obrigado pela ajuda!|Tchau, até amanhã!|Você pode repetir mais devagar?|Fala uma frase motivadora.
Quantas pernas tem uma aranha?|O que é a UFES?|Me explica o que é um algoritmo.|Qual é o maior planeta do sistema solar?
Você fala inglês?|Que horas são mais ou menos agora?|Você conhece a cidade de Vitória?|O que você acha de música?
Qual é a fruta mais rica em vitamina C?|Como se diz obrigado em inglês?|Você pode me lembrar de beber água?
Me diz três cores primárias.|O que é um supercomputador?|Qual é o oposto de quente?|Você consegue aprender coisas novas?
Escreve uma frase bonita sobre o mar.|Quanto é dois mais dois?|Qual é o animal mais rápido do mundo?""".replace("\n", "|").split("|")
PEDE_PERGUNTAS = ("Leia o texto abaixo e escreva {n} perguntas diferentes, em português do Brasil, que uma pessoa faria "
                  "em voz alta para um robô e que o texto responde. Varie o jeito de perguntar: informal, curta, mais "
                  "longa, com 'você sabe...', 'me fala...'. Uma pergunta por linha, sem numeração e sem mais nada.\n\n"
                  "Texto:\n{t}")


def normaliza(t):
    t = unicodedata.normalize("NFD", t.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def palavras(t):
    return set(p for p in re.findall(r"[a-z0-9]+", normaliza(t)) if len(p) > 3)


def prompt(pergunta, trecho=None, desligado=False):
    saber = (f"O que você sabe sobre isso (use se ajudar; não leia tudo):\n- {trecho['titulo']}: {trecho['texto'][:900]}\n"
             if trecho else "")
    return PERSONA + (DESLIGADO if desligado else "") + saber + EXEMPLOS + f"Usuário: {pergunta}\nPrometheus:"


def limpa(r):
    r = re.sub(r"^\s*(prometheus|rob[oô])\s*:\s*", "", r, flags=re.IGNORECASE).strip()
    return re.split(r"\n\s*(Usuário|Prometheus)\s*:", r)[0].strip()


def carrega(a):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(a.er1)
    modelo = AutoModelForImageTextToText.from_pretrained(a.er1, dtype=torch.bfloat16, device_map="cuda").eval()
    return proc, modelo


def msgs(texto, resposta=None):
    m = [{"role": "user", "content": [{"type": "text", "text": texto}]}]
    if resposta is not None:
        m.append({"role": "assistant", "content": [{"type": "text", "text": resposta}]})
    return m


@torch.inference_mode()
def gera_texto(modelo, proc, texto, max_tokens=220):
    ent = proc.apply_chat_template(msgs(texto), tokenize=True, add_generation_prompt=True, return_dict=True,
                                   return_tensors="pt").to(modelo.device)
    out = modelo.generate(**ent, max_new_tokens=max_tokens, do_sample=False, repetition_penalty=1.05)
    return proc.decode(out[0, ent["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def itens_da_memoria(db):
    c = sqlite3.connect(db)
    itens = [{"id": i, "titulo": t, "texto": x, "tipo": tp} for i, t, x, tp in
             c.execute("SELECT id, titulo, texto, tipo FROM conhecimento WHERE ativo=1 ORDER BY id")]
    return [x for x in itens if x["tipo"] != "correcao"], [x for x in itens if x["tipo"] == "correcao"]


# ── fase 1 ──
def fase_gera(a):
    saida = Path(a.saida)
    arq = saida / f"gera_{a.parte}.jsonl"
    if arq.exists():
        print(f"{arq.name} já existe", flush=True)
        return
    itens, _ = itens_da_memoria(a.db)
    meus = [x for k, x in enumerate(itens) if k % a.partes == a.parte]
    gerais = [g for k, g in enumerate(GERAIS) if k % a.partes == a.parte]
    proc, modelo = carrega(a)
    t0 = time.time()
    with open(str(arq) + ".tmp", "w") as f:
        for k, it in enumerate(meus):
            cru = gera_texto(modelo, proc, PEDE_PERGUNTAS.format(n=a.perguntas, t=f"{it['titulo']}. {it['texto']}"), 300)
            qs = [re.sub(r"^\s*([-•*]|\d+[.)])\s*", "", l).strip().strip('"') for l in cru.split("\n")]
            # 30/10: o ER-1 às vezes esquece o "?" — vale qualquer linha com 3+ palavras que não seja título/rótulo
            qs = [q if q.endswith("?") else q.rstrip(".!") + "?" for q in qs
                  if 8 < len(q) < 200 and len(q.split()) >= 3 and not q.endswith(":")]
            qs = list(dict.fromkeys(qs))[:a.perguntas]
            if not qs:
                print(f"   ⚠️ sem perguntas; o ER-1 escreveu: {cru[:200]!r}", flush=True)
            for j, q in enumerate(qs):
                desl = random.Random(f"{it['id']}-{j}").random() < 0.5
                r = limpa(gera_texto(modelo, proc, prompt(q, it, desl)))
                f.write(json.dumps({"tipo": "memoria", "item": it["id"], "titulo": it["titulo"], "pergunta": q,
                                    "resposta": r, "desligado": desl, "teste": j == 0}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[parte {a.parte}] item {k + 1}/{len(meus)} ({len(qs)} perguntas) | {it['titulo'][:50]} | "
                  f"{(time.time() - t0) / (k + 1):.1f} s/item", flush=True)
        for g in gerais:
            desl = random.Random(g).random() < 0.5
            r = limpa(gera_texto(modelo, proc, prompt(g, None, desl)))
            f.write(json.dumps({"tipo": "geral", "pergunta": g, "resposta": r, "desligado": desl,
                                "teste": random.Random(g + "t").random() < 0.15}, ensure_ascii=False) + "\n")
    Path(str(arq) + ".tmp").rename(arq)
    print(f"✅ {arq} em {time.time() - t0:.0f} s", flush=True)


def le_gerados(saida):
    linhas = []
    for f in sorted(Path(saida).glob("gera_*.jsonl")):
        linhas += [json.loads(x) for x in open(f)]
    return linhas


def exemplos(a):
    itens, corr = itens_da_memoria(a.db)
    por_id = {x["id"]: x for x in itens}
    tr, te = [], []
    for l in le_gerados(a.saida):
        (te if l["teste"] else tr).append(l)
    out = []
    for l in tr:
        it = por_id.get(l.get("item"))
        out.append(prompt(l["pergunta"], None, l["desligado"]) + "\x00" + l["resposta"])   # sem trecho: guardar
        if it is not None:
            out.append(prompt(l["pergunta"], it, l["desligado"]) + "\x00" + l["resposta"])   # com trecho (RAG ligado)
    for c in corr:   # "Pergunta: ...\nResposta certa: ..." vindo das correções na cabine
        m = re.match(r"Pergunta:\s*(.*?)\nResposta certa:\s*(.*)", c["texto"], re.S)
        if m:
            out += [prompt(m.group(1).strip(), None, k == 1) + "\x00" + m.group(2).strip() for k in range(3)]
    return out, te, por_id


def tokeniza(proc, ex, dev):
    pergunta, resposta = ex.split("\x00")
    m = msgs(pergunta, resposta)
    cheio = proc.apply_chat_template(m, tokenize=True, return_dict=True, return_tensors="pt")
    pre = proc.apply_chat_template(m[:1], tokenize=True, add_generation_prompt=True, return_dict=True,
                                   return_tensors="pt")
    n = pre["input_ids"].shape[1]
    labels = cheio["input_ids"].clone()
    labels[0, :n] = -100
    cheio["labels"] = labels
    return {k: v.to(dev) for k, v in cheio.items()}


# ── fase 2 ──
def fase_treino(a):
    saida = Path(a.saida)
    todos, te, _ = exemplos(a)
    proc, modelo = carrega(a)
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                     target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    modelo = get_peft_model(modelo, cfg)
    modelo.print_trainable_parameters()
    modelo.gradient_checkpointing_enable()
    modelo.enable_input_require_grads()
    passos = math.ceil(len(todos) * a.epocas / a.acumula)
    opt = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    aquece = max(1, passos // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / aquece) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / passos))))
    print(f"treino: {len(todos)} exemplos ({len(te)} perguntas guardadas para teste) | {a.epocas} épocas | {passos} "
          f"passos", flush=True)
    rnd, passo = random.Random(0), 0
    for ep in range(1, a.epocas + 1):
        modelo.train()
        rnd.shuffle(todos)
        t0, soma, n = time.time(), 0.0, 0
        for i, ex in enumerate(todos):
            perda = modelo(**tokeniza(proc, ex, modelo.device)).loss / a.acumula
            perda.backward()
            soma, n = soma + perda.item() * a.acumula, n + 1
            if (i + 1) % a.acumula == 0 or i + 1 == len(todos):
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                passo += 1
                if passo % 10 == 0:
                    print(f"época {ep} passo {passo}/{passos} | perda {soma / n:.4f} | lr {sched.get_last_lr()[0]:.2e} "
                          f"| {(time.time() - t0) / (i + 1):.2f} s/exemplo", flush=True)
                    soma, n = 0.0, 0
        modelo.save_pretrained(saida / f"adaptador_ep{ep}")
        print(f"✅ adaptador_ep{ep} salvo", flush=True)


# ── fase 3 ──
def fase_avalia(a):
    saida = Path(a.saida)
    _, te, por_id = exemplos(a)
    proc, modelo = carrega(a)
    from peft import PeftModel
    modelo = PeftModel.from_pretrained(modelo, str(saida / f"adaptador_ep{a.epocas}")).eval()

    def sobrepoe(r, ref):
        a_, b_ = palavras(r), palavras(ref)
        return len(a_ & b_) / max(1, len(b_))

    linhas, soma = [], {"base_sem": 0.0, "lora_sem": 0.0, "lora_com": 0.0, "base_com": 0.0}
    nmem = 0
    for l in te:
        it = por_id.get(l.get("item"))
        with modelo.disable_adapter():
            b_sem = limpa(gera_texto(modelo, proc, prompt(l["pergunta"], None, l["desligado"])))
        l_sem = limpa(gera_texto(modelo, proc, prompt(l["pergunta"], None, l["desligado"])))
        l_com = limpa(gera_texto(modelo, proc, prompt(l["pergunta"], it, l["desligado"]))) if it else l_sem
        row = {"tipo": l["tipo"], "pergunta": l["pergunta"], "professor": l["resposta"], "base_sem_trecho": b_sem,
               "lora_sem_trecho": l_sem, "lora_com_trecho": l_com}
        if l["tipo"] == "memoria":
            nmem += 1
            soma["base_sem"] += sobrepoe(b_sem, l["resposta"])
            soma["lora_sem"] += sobrepoe(l_sem, l["resposta"])
            soma["lora_com"] += sobrepoe(l_com, l["resposta"])
        linhas.append(row)
        print(f"• {l['pergunta']}\n   base: {b_sem[:120]}\n   LoRA: {l_sem[:120]}", flush=True)
    ger = [r for r in linhas if r["tipo"] == "geral"]
    res = {"n_memoria": nmem, "n_geral": len(ger),
           "memoria_base_sem_trecho": round(soma["base_sem"] / max(1, nmem), 3),
           "memoria_lora_sem_trecho": round(soma["lora_sem"] / max(1, nmem), 3),
           "memoria_lora_com_trecho": round(soma["lora_com"] / max(1, nmem), 3),
           "geral_concordancia_lora_vs_original": round(sum(sobrepoe(r["lora_sem_trecho"], r["professor"])
                                                            for r in ger) / max(1, len(ger)), 3)}
    (saida / "avaliacao.json").write_text(json.dumps({"resumo": res, "respostas": linhas}, ensure_ascii=False, indent=1))
    print("RESUMO", json.dumps(res, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fase", choices=["gera", "treino", "avalia"], required=True)
    ap.add_argument("--db", required=True, help="a memória da cabine (~/er1_memoria/memoria.db da PGX)")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--er1", default=ER1)
    ap.add_argument("--parte", type=int, default=0)
    ap.add_argument("--partes", type=int, default=1)
    ap.add_argument("--perguntas", type=int, default=6, help="perguntas geradas por item da memória")
    ap.add_argument("--epocas", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--acumula", type=int, default=8)
    a = ap.parse_args()
    Path(a.saida).mkdir(parents=True, exist_ok=True)
    {"gera": fase_gera, "treino": fase_treino, "avalia": fase_avalia}[a.fase](a)


if __name__ == "__main__":
    main()
