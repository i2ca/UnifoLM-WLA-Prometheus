#!/usr/bin/env python
"""MEDE a velocidade do ER-1 na PGX (30/10) com as técnicas de aceleração, nas mesmas perguntas da cabine:
  base      generate padrão do HF (o que o er1_pergunta.py usa hoje)
  estatico  cache_implementation="static" (memória do KV fixa, sem realocar a cada token)
  compile   estático + torch.compile do forward (decodificação em grafo)
  espec     decodificação especulativa: um Qwen3 pequeno (mesmo vocabulário) chuta tokens e o ER-1 confere
Cada uma: tempo até o 1º token, tokens/s, e se a resposta é IGUAL à da base (greedy: tem de ser).

    ~/miniforge3/envs/wla/bin/python bench_er1.py --modos base estatico compile espec
"""
import argparse
import json
import time

import torch

ER1 = "playground/Pretrained_models/UnifoLM-ER-1"
PERGUNTAS = ["Boa noite, tudo bem?", "Como eu faço um café coado?", "O que é a IARA do LCAD?",
             "Me conta uma curiosidade sobre robôs."]
PERSONA = ("Você é o Prometheus, um robô humanoide criado pela equipe do LCAD (Laboratório de Computação de Alto "
           "Desempenho) da UFES. Você fala português do Brasil, sempre em primeira pessoa, como um assistente de voz "
           "simpático e natural: frases curtas, tom de conversa, sem listas nem símbolos.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modos", nargs="+", default=["base", "estatico", "compile", "espec"])
    ap.add_argument("--rascunho", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max", type=int, default=120)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer
    proc = AutoProcessor.from_pretrained(ER1)
    m = AutoModelForImageTextToText.from_pretrained(ER1, dtype=torch.bfloat16, device_map="cuda").eval()

    def entrada(q):
        msgs = [{"role": "user", "content": [{"type": "text", "text": PERSONA + f"Usuário: {q}\nPrometheus:"}]}]
        return proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                        return_tensors="pt").to(m.device)

    def roda(nome, **kw):
        import threading
        res = []
        for rep in range(2):   # a 1ª passada aquece (compila); vale a 2ª
            tot_tok, tot_s, ttft = 0, 0.0, []
            textos = []
            for q in PERGUNTAS:
                ent = entrada(q)
                st = TextIteratorStreamer(proc.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=180)
                t0 = time.perf_counter()
                th = threading.Thread(target=lambda: m.generate(**ent, max_new_tokens=a.max, do_sample=False,
                                                                streamer=st, **kw))
                th.start()
                primeiro, txt = None, ""
                for pedaco in st:
                    if primeiro is None and pedaco.strip():
                        primeiro = time.perf_counter() - t0
                    txt += pedaco
                th.join()
                dt = time.perf_counter() - t0
                n = len(proc.tokenizer(txt)["input_ids"])
                tot_tok, tot_s = tot_tok + n, tot_s + dt
                ttft.append(primeiro or dt)
                textos.append(txt.strip())
            res = {"modo": nome, "tok_s": round(tot_tok / tot_s, 1), "primeiro_token_s": round(sum(ttft) / len(ttft), 3),
                   "s_por_resposta": round(tot_s / len(PERGUNTAS), 2), "textos": textos}
            print(f"[{nome}] passada {rep}: {res['tok_s']} tok/s | 1º token {res['primeiro_token_s']} s | "
                  f"{res['s_por_resposta']} s/resposta", flush=True)
        return res

    out = {}
    if "base" in a.modos:
        out["base"] = roda("base")
    if "estatico" in a.modos:
        out["estatico"] = roda("estatico", cache_implementation="static", disable_compile=True)
    if "compile" in a.modos:
        try:
            fwd = m.forward
            m.forward = torch.compile(m.forward, mode="reduce-overhead", fullgraph=False)
            out["compile"] = roda("compile", cache_implementation="static")
            m.forward = fwd
        except Exception as e:  # noqa: BLE001
            print(f"[compile] falhou: {e!r}"[:400], flush=True)
    if "espec" in a.modos:
        try:
            r = AutoModelForCausalLM.from_pretrained(a.rascunho, dtype=torch.bfloat16, device_map="cuda").eval()
            try:   # mesmo vocabulário (Qwen3): especulação padrão
                out["espec"] = roda("espec", assistant_model=r)
            except Exception as e:  # noqa: BLE001
                print(f"[espec] padrão falhou ({e!r}"[:300] + "); tentando com os dois tokenizadores", flush=True)
                out["espec"] = roda("espec", assistant_model=r, tokenizer=proc.tokenizer,
                                    assistant_tokenizer=__import__("transformers").AutoTokenizer.from_pretrained(a.rascunho))
        except Exception as e:  # noqa: BLE001
            print(f"[espec] falhou: {e!r}"[:400], flush=True)
    base = out.get("base", {}).get("textos")
    for k, v in out.items():
        if base and k != "base":
            v["igual_a_base"] = sum(x == y for x, y in zip(v["textos"], base))
        print("RESUMO", k, {x: y for x, y in v.items() if x != "textos"}, flush=True)
    json.dump(out, open("/tmp/bench_er1.json", "w"), ensure_ascii=False, indent=1)
    print("FIM", flush=True)


if __name__ == "__main__":
    main()
