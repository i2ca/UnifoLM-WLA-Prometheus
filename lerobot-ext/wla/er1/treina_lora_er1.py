#!/usr/bin/env python
"""LoRA do UnifoLM-ER-1 (Qwen3-VL) para a NOSSA cena (02/10): tarefa concluída? mão segurando? objeto no lugar?

Sem desaprender o que ele já sabia:
  1. o ER-1 original fica congelado; o LoRA é um adaptador à parte (dá para ligar/desligar por pergunta);
  2. LoRA só nas camadas de LINGUAGEM (a visão fica intacta), posto baixo, lr baixo, poucas épocas;
  3. "auto-destilação": o ER-1 ORIGINAL responde perguntas gerais nas nossas imagens e essas respostas entram
     no treino junto com as nossas — o adaptador aprende o novo sem se afastar do jeito antigo de responder;
  4. mede ANTES e DEPOIS: acerto nas nossas perguntas (episódios de teste) e CONCORDÂNCIA com o original nas
     perguntas gerais de sim/não (retenção).

Fases (cada uma grava o resultado e é pulada se já existir):
  geral   -> <saida>/geral_treino.jsonl, geral_teste.jsonl  (respostas do ER-1 original)
  base    -> <saida>/avaliacao_base.json
  treino  -> <saida>/adaptador_epN/ (um por época)
  depois  -> <saida>/avaliacao_epN.json

    CUDA_VISIBLE_DEVICES=1 python treina_lora_er1.py --dados /data/mrwlker/er1_lora/dados --saida /data/mrwlker/er1_lora/run1
"""
import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

ER1 = "/data/mrwlker/UnifoLM-ER-1"
ABERTAS = ["Describe this image in one sentence.", "What objects are on the table?",
           "What is the robot doing in this image?", "Where is the robot's right hand?",
           "How many robot hands can you see?", "What color is the largest object on the table?"]
SN_GERAIS = ["Is there a kettle in this image?", "Is there a chair in this image?", "Is there a person in this image?",
             "Is there a computer or laptop in this image?", "Is there a black X mark on the table?",
             "Is there a cardboard box in this image?", "Is there a cooking pot in this image?",
             "Is there a white cup in this image?", "Is the floor visible in this image?",
             "Is there a red apple in this image?"]
SUFIXO = " Answer only yes or no."


def msgs(imagens, pergunta, resposta=None):
    m = [{"role": "user", "content": [{"type": "image", "image": Image.open(p).convert("RGB")} for p in imagens]
          + [{"type": "text", "text": pergunta}]}]
    if resposta is not None:
        m.append({"role": "assistant", "content": [{"type": "text", "text": resposta}]})
    return m


@torch.inference_mode()
def responde(modelo, proc, imagens, pergunta, max_tokens=8):
    ent = proc.apply_chat_template(msgs(imagens, pergunta), tokenize=True, add_generation_prompt=True,
                                   return_dict=True, return_tensors="pt").to(modelo.device)
    out = modelo.generate(**ent, max_new_tokens=max_tokens, do_sample=False)
    return proc.decode(out[0, ent["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def sim_nao(texto):
    t = texto.lower().strip()
    return "yes" if t.startswith("yes") else "no" if t.startswith("no") else t


def avalia(modelo, proc, teste, geral_teste, rotulo):
    t0 = time.time()
    por = defaultdict(lambda: [0, 0])
    for l in teste:
        ok = sim_nao(responde(modelo, proc, l["imagens"], l["pergunta"])) == l["resposta"]
        for k in ("total", f"tipo:{l['tipo']}", f"dataset:{l['dataset']}"):
            por[k][0] += ok
            por[k][1] += 1
    ret = [0, 0]
    for l in geral_teste:
        ret[0] += sim_nao(responde(modelo, proc, l["imagens"], l["pergunta"])) == l["resposta"]
        ret[1] += 1
    r = {k: round(a / b, 4) for k, (a, b) in sorted(por.items())}
    r["retencao_sim_nao_geral"] = round(ret[0] / max(1, ret[1]), 4)
    r["n_teste"], r["n_geral"], r["segundos"] = len(teste), len(geral_teste), round(time.time() - t0)
    print(f"[{rotulo}] acerto {r['total']:.3f} | " + " | ".join(f"{k} {v:.3f}" for k, v in r.items()
                                                                   if k.startswith(("tipo:", "dataset:")))
          + f" | retenção geral {r['retencao_sim_nao_geral']:.3f}", flush=True)
    return r


def exemplo_treino(proc, l, dev):
    """input_ids/labels com a perda SÓ na resposta (o resto -100)."""
    m = msgs(l["imagens"], l["pergunta"], l["resposta"])
    cheio = proc.apply_chat_template(m, tokenize=True, return_dict=True, return_tensors="pt")
    pre = proc.apply_chat_template(m[:1], tokenize=True, add_generation_prompt=True, return_dict=True,
                                   return_tensors="pt")
    n = pre["input_ids"].shape[1]
    assert torch.equal(cheio["input_ids"][0, :n], pre["input_ids"][0]), "o template não começa igual ao prompt"
    labels = cheio["input_ids"].clone()
    labels[0, :n] = -100
    cheio["labels"] = labels
    return {k: v.to(dev) for k, v in cheio.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--n-geral", type=int, default=700, help="quadros com perguntas gerais (auto-destilação)")
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--acumula", type=int, default=8, help="exemplos por passo do otimizador")
    ap.add_argument("--max", type=int, default=0, help="limita os exemplos (teste rápido do script)")
    ap.add_argument("--max-teste", type=int, default=0)
    a = ap.parse_args()
    rnd = random.Random(0)
    dados, saida = Path(a.dados), Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)
    treino = [json.loads(x) for x in open(dados / "treino.jsonl")]
    teste = [json.loads(x) for x in open(dados / "teste.jsonl")]
    if a.max:
        treino = treino[:a.max]
    if a.max_teste:
        teste = teste[:a.max_teste]

    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(ER1)
    modelo = AutoModelForImageTextToText.from_pretrained(ER1, dtype=torch.bfloat16, device_map="cuda").eval()
    dev = modelo.device
    print(f"ER-1 carregado | treino {len(treino)} | teste {len(teste)}", flush=True)

    # ── 1. respostas do ER-1 ORIGINAL para perguntas gerais (auto-destilação + prova de retenção) ──
    arq_g = {p: saida / f"geral_{p}.jsonl" for p in ("treino", "teste")}
    if not all(f.exists() for f in arq_g.values()):
        cabecas = sorted({l["imagens"][0] for l in treino})
        cabecas_t = sorted({l["imagens"][0] for l in teste})
        rnd.shuffle(cabecas)
        rnd.shuffle(cabecas_t)
        n_t = a.n_geral if not a.max else 10
        gerais = {"treino": [], "teste": []}
        t0 = time.time()
        for parte, imgs, n in (("treino", cabecas, n_t), ("teste", cabecas_t, max(5, n_t // 5))):
            for p in imgs[:n]:
                q = rnd.choice(ABERTAS)
                gerais[parte].append({"imagens": [p], "pergunta": q, "tipo": "geral_aberta",
                                      "resposta": responde(modelo, proc, [p], q, max_tokens=48)})
                q = rnd.choice(SN_GERAIS) + SUFIXO
                gerais[parte].append({"imagens": [p], "pergunta": q, "tipo": "geral_sim_nao",
                                      "resposta": sim_nao(responde(modelo, proc, [p], q))})
        for parte, ls in gerais.items():
            with open(arq_g[parte], "w") as f:
                for l in ls:
                    f.write(json.dumps(l, ensure_ascii=False) + "\n")
        print(f"respostas gerais do ER-1 original: {sum(map(len, gerais.values()))} em {time.time() - t0:.0f} s",
              flush=True)
    geral_tr = [json.loads(x) for x in open(arq_g["treino"])]
    geral_te = [json.loads(x) for x in open(arq_g["teste"]) if json.loads(x)["tipo"] == "geral_sim_nao"]

    # ── 2. linha de base ──
    arq_b = saida / "avaliacao_base.json"
    if not arq_b.exists():
        arq_b.write_text(json.dumps(avalia(modelo, proc, teste, geral_te, "ER-1 original"), indent=1))
    else:
        print(f"[ER-1 original] {json.loads(arq_b.read_text())}", flush=True)

    # ── 3. LoRA só nas camadas de linguagem ──
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                     target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    modelo = get_peft_model(modelo, cfg)
    modelo.print_trainable_parameters()
    todos = treino + geral_tr
    passos = math.ceil(len(todos) * a.epocas / a.acumula)
    opt = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    aquece = max(1, passos // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / aquece) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / passos))))
    print(f"treino: {len(treino)} nossas + {len(geral_tr)} gerais | {a.epocas} épocas | {passos} passos", flush=True)
    passo = 0
    for ep in range(1, a.epocas + 1):
        modelo.train()
        rnd.shuffle(todos)
        t0, soma, n = time.time(), 0.0, 0
        for i, l in enumerate(todos):
            perda = modelo(**exemplo_treino(proc, l, dev)).loss / a.acumula
            perda.backward()
            soma += perda.item() * a.acumula
            n += 1
            if (i + 1) % a.acumula == 0 or i + 1 == len(todos):
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                passo += 1
                if passo % 20 == 0:
                    print(f"época {ep} passo {passo}/{passos} | perda {soma / n:.4f} | lr {sched.get_last_lr()[0]:.2e} | "
                          f"{(time.time() - t0) / (i + 1):.2f} s/exemplo", flush=True)
                    soma, n = 0.0, 0
        pasta = saida / f"adaptador_ep{ep}"
        modelo.save_pretrained(pasta)
        modelo.eval()
        r = avalia(modelo, proc, teste, geral_te, f"LoRA época {ep}")
        (saida / f"avaliacao_ep{ep}.json").write_text(json.dumps(r, indent=1))
        print(f"✅ adaptador salvo em {pasta}", flush=True)


if __name__ == "__main__":
    main()
