#!/usr/bin/env python
"""MEMÓRIA DO ER-1 (30/10): o que o Prometheus aprende conversando, guardado na PGX (SQLite), e buscado a cada
pergunta (RAG) — o ER-1 é pequeno: em vez de um prompt enorme, entram só os 2-3 trechos mais parecidos com a pergunta.

  conhecimento  fatos, receitas, documentos e CORREÇÕES ("o certo é ...") — o que é buscado
  conversa      todas as trocas (pergunta, resposta, se o robô estava ligado) — vira dataset para o LoRA depois

Busca BM25 (palavras, sem acento, sem as palavras vazias do português): rápida na CPU e sem modelo extra; a interface
(busca/adiciona) é a mesma se um dia trocarmos por embeddings. Os arquivos .md/.txt/.json de --pasta-base entram
sozinhos na primeira vez (e de novo se mudarem): cada "## título" de um .md vira um item.

    python memoria_er1.py --busca "receita de café coado"        # teste
    python memoria_er1.py --exporta ~/er1_memoria/conversas.jsonl
"""
import argparse
import hashlib
import json
import math
import re
import sqlite3
import threading
import time
import unicodedata
from collections import Counter
from pathlib import Path

VAZIAS = set("""a o e é de da do das dos em no na nos nas um uma uns umas que se por para pra com sem como mais menos
ao aos à às ou eu tu ele ela nós vós eles elas me te lhe nos vos meu minha seu sua isso isto esse essa este esta aquilo
foi ser ter tem está esta estou vai vou sou era são há já não sim também muito bem qual quais quando onde porque
você voce vc o_que the of and to is in on
oi ola olá bom boa dia tarde noite tudo td tranquilo beleza obrigado obrigada valeu favor por_favor me te
ensina ensinar fazer faz faço fazendo sabe saber sei pode podia consegue conhece conta fala diga diz quero queria
gostaria explica mostra preciso agora aí ai então entao sobre quem algo alguma coisa coisas""".split())
PADRAO_BASE = Path(__file__).resolve().parent / "er1" / "conhecimento"


def normaliza(t):
    t = unicodedata.normalize("NFD", t.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def palavras(t):
    out = []
    for p in re.findall(r"[a-z0-9]+", normaliza(t)):
        if p in VAZIAS or len(p) < 2:
            continue
        p = p[:-1] if len(p) > 4 and p.endswith("s") else p   # plural simples: cafés -> cafe
        out.append(p[:5])   # radical por corte (segurando/segura -> segur; receitas -> recei): simples e simétrico
        if len(p) > 5:      # 03/11: + o radical curto (fundou/fundada -> fund; o de 5 letras separava "fundo" e "funda")
            out.append(p[:4] + "~")
    return out


class Memoria:
    def __init__(self, caminho, pasta_base=PADRAO_BASE):
        self.caminho = Path(caminho).expanduser()
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self.trava = threading.Lock()
        self.db = sqlite3.connect(str(self.caminho), check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS conhecimento (id INTEGER PRIMARY KEY, t REAL, titulo TEXT, texto TEXT,
                tipo TEXT, fonte TEXT, ativo INTEGER DEFAULT 1);
            CREATE TABLE IF NOT EXISTS conversa (id INTEGER PRIMARY KEY, t REAL, pergunta TEXT, resposta TEXT,
                via TEXT, robo_ligado INTEGER, trechos TEXT, correcao TEXT);
            CREATE TABLE IF NOT EXISTS arquivos (nome TEXT PRIMARY KEY, hash TEXT);""")
        self.db.commit()
        if pasta_base:
            self.carrega_pasta(pasta_base)
        self._indexa()

    # ── índice BM25 (refeito a cada mudança; milhares de itens indexam em ms) ──
    def _indexa(self):
        with self.trava:
            linhas = self.db.execute("SELECT id, titulo, texto, tipo FROM conhecimento WHERE ativo=1").fetchall()
        self.docs = []
        for i, titulo, texto, tipo in linhas:
            ps = palavras(titulo) * 3 + palavras(texto)   # o título pesa mais
            self.docs.append((i, titulo, texto, tipo, Counter(ps), len(ps)))
        n = len(self.docs) or 1
        df = Counter(p for d in self.docs for p in d[4])
        self.idf = {p: math.log(1 + (n - c + 0.5) / (c + 0.5)) for p, c in df.items()}
        self.media = sum(d[5] for d in self.docs) / n if self.docs else 1

    def busca(self, pergunta, k=3, minimo=2.5):
        q = palavras(pergunta)
        res = []
        for i, titulo, texto, tipo, tf, tam in self.docs:
            s = 0.0
            for p in q:
                f = tf.get(p, 0)
                if f:
                    s += self.idf.get(p, 0) * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * tam / self.media))
            if tipo == "correcao":
                s *= 1.5   # o que você corrigiu vem primeiro
            if s >= minimo:
                res.append((s, {"id": i, "titulo": titulo, "texto": texto, "tipo": tipo, "nota": round(s, 1)}))
        res.sort(key=lambda x: -x[0])
        # só os que chegam perto do melhor (o 3º trecho fraco só confunde o ER-1)
        return [r for s, r in res[:k] if s >= 0.6 * res[0][0]]

    # ── escrever ──
    def adiciona(self, titulo, texto, tipo="fato", fonte="cabine", reindexa=True):
        with self.trava:
            cur = self.db.execute("INSERT INTO conhecimento (t, titulo, texto, tipo, fonte) VALUES (?,?,?,?,?)",
                                  (time.time(), titulo.strip()[:200], texto.strip(), tipo, fonte))
            self.db.commit()
        if reindexa:
            self._indexa()
        return cur.lastrowid

    def apaga(self, i):
        with self.trava:
            self.db.execute("UPDATE conhecimento SET ativo=0 WHERE id=?", (i,))
            self.db.commit()
        self._indexa()

    def adiciona_documento(self, nome, conteudo, fonte):
        """Um arquivo inteiro: .md -> um item por "## título" (ou "# "); .json -> lista de {titulo, texto[, tipo]};
        texto solto -> pedaços de ~800 caracteres por parágrafo."""
        n = 0
        if nome.endswith(".json"):
            for x in json.loads(conteudo):
                self.adiciona(x.get("titulo", nome), x["texto"], x.get("tipo", "documento"), fonte, reindexa=False)
                n += 1
        else:
            secoes = re.split(r"\n(?=#{1,3} )", "\n" + conteudo)
            tipo_arq = "receita" if "receita" in nome.lower() else "documento"
            for s in secoes:
                s = s.strip()
                if not s:
                    continue
                m = re.match(r"#{1,3} (.+)\n?", s)
                titulo = m.group(1).strip() if m else Path(nome).stem
                corpo = s[m.end():].strip() if m else s
                if not corpo:
                    continue
                pedaco = ""
                for par in re.split(r"\n\s*\n", corpo):
                    if len(pedaco) + len(par) > 900 and pedaco:
                        self.adiciona(titulo, pedaco, tipo_arq, fonte, reindexa=False)
                        n += 1
                        pedaco = ""
                    pedaco += ("\n\n" if pedaco else "") + par
                if pedaco:
                    self.adiciona(titulo, pedaco, tipo_arq, fonte, reindexa=False)
                    n += 1
        self._indexa()
        return n

    def carrega_pasta(self, pasta):
        pasta = Path(pasta)
        if not pasta.is_dir():
            return
        for f in sorted(pasta.glob("*")):
            if f.suffix not in (".md", ".txt", ".json"):
                continue
            h = hashlib.md5(f.read_bytes()).hexdigest()
            with self.trava:
                ant = self.db.execute("SELECT hash FROM arquivos WHERE nome=?", (f.name,)).fetchone()
            if ant and ant[0] == h:
                continue
            with self.trava:   # mudou: tira a versão antiga desse arquivo e põe a nova
                self.db.execute("UPDATE conhecimento SET ativo=0 WHERE fonte=?", (f"arquivo:{f.name}",))
                self.db.commit()
            n = self.adiciona_documento(f.name, f.read_text(encoding="utf-8"), f"arquivo:{f.name}")
            with self.trava:
                self.db.execute("INSERT OR REPLACE INTO arquivos VALUES (?,?)", (f.name, h))
                self.db.commit()
            print(f"[memória] {f.name}: {n} itens", flush=True)

    def registra(self, pergunta, resposta, via, robo_ligado, trechos):
        with self.trava:
            cur = self.db.execute("INSERT INTO conversa (t, pergunta, resposta, via, robo_ligado, trechos) "
                                  "VALUES (?,?,?,?,?,?)", (time.time(), pergunta, resposta, via, int(robo_ligado),
                                                           json.dumps([x["id"] for x in trechos])))
            self.db.commit()
        return cur.lastrowid

    def corrige(self, id_conversa, pergunta, certo):
        """A resposta da conversa id estava errada: guarda a certa (vira conhecimento buscável e dado de treino)."""
        with self.trava:
            self.db.execute("UPDATE conversa SET correcao=? WHERE id=?", (certo, id_conversa))
            self.db.commit()
        return self.adiciona(pergunta, f"Pergunta: {pergunta}\nResposta certa: {certo}", "correcao", "correção")

    # ── ler ──
    def pergunta_de(self, id_conversa):
        with self.trava:
            r = self.db.execute("SELECT pergunta FROM conversa WHERE id=?", (id_conversa,)).fetchone()
        return r[0] if r else None

    def resumo(self):
        with self.trava:
            tipos = dict(self.db.execute("SELECT tipo, COUNT(*) FROM conhecimento WHERE ativo=1 GROUP BY tipo"))
            nconv = self.db.execute("SELECT COUNT(*) FROM conversa").fetchone()[0]
            ncor = self.db.execute("SELECT COUNT(*) FROM conversa WHERE correcao IS NOT NULL").fetchone()[0]
            ult = self.db.execute("SELECT id, titulo, tipo FROM conhecimento WHERE ativo=1 AND fonte NOT LIKE "
                                  "'arquivo:%' ORDER BY id DESC LIMIT 8").fetchall()
        return {"tipos": tipos, "conversas": nconv, "corrigidas": ncor,
                "ultimos": [{"id": i, "titulo": t, "tipo": tp} for i, t, tp in ult]}

    def exporta(self, destino):
        """Dataset para o LoRA do ER-1: cada troca com a resposta FINAL (a corrigida, se houve correção)."""
        with self.trava:
            linhas = self.db.execute("SELECT t, pergunta, resposta, correcao, robo_ligado, via FROM conversa "
                                     "ORDER BY id").fetchall()
        with open(Path(destino).expanduser(), "w", encoding="utf-8") as f:
            for t, p, r, c, lig, via in linhas:
                f.write(json.dumps({"t": t, "pergunta": p, "resposta": c or r, "original": r, "corrigida": c is not None,
                                    "robo_ligado": bool(lig), "via": via}, ensure_ascii=False) + "\n")
        return len(linhas)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="~/er1_memoria/memoria.db")
    ap.add_argument("--busca")
    ap.add_argument("--exporta")
    a = ap.parse_args()
    m = Memoria(a.db)
    print(m.resumo())
    if a.busca:
        for x in m.busca(a.busca):
            print(f"  [{x['nota']}] ({x['tipo']}) {x['titulo']}: {x['texto'][:120]!r}")
    if a.exporta:
        print("exportadas", m.exporta(a.exporta))
