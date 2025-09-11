from typing import List, Dict
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
import os, re
from dotenv import load_dotenv

load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)

SYSTEM_PROMPT = """Você é um assistente que responde SOMENTE com base no IPCC AR6 (SYR Longer Report).
Regras:
- Responda SOMENTE no idioma português (pt-br), mantendo termos técnicos do IPCC.
- Cada afirmação factual deve citar a(s) página(s) no formato EXATO [p.X]. Ex.: "[p.34]".
- Se não houver evidência suficiente nos trechos fornecidos, responda exatamente:
  "Não encontrei evidência suficiente no IPCC para responder com segurança."
- Não invente fontes nem use conhecimento externo. Seja conciso.
"""

# Normaliza variações comuns de citação para o formato [p.X]
_CIT_PATTS = [
    (re.compile(r"\(p\.?\s*(\d+)\)", re.I), r"[p.\1]"),   # (p. 34) -> [p.34]
    (re.compile(r"\[p\s*(\d+)\]", re.I), r"[p.\1]"),       # [p 34]  -> [p.34]
    (re.compile(r"\[pg\.?\s*(\d+)\]", re.I), r"[p.\1]"),   # [pg.34] -> [p.34]
]

def _normalize_citations(txt: str) -> str:
    for patt, rep in _CIT_PATTS:
        txt = patt.sub(rep, txt)
    return txt

def _build_context(ctxs: List[Dict]) -> str:
    """Formata os trechos recuperados como contexto, um por linha."""
    if not ctxs:
        return "(sem trechos recuperados)"
    lines = []
    for c in ctxs:
        m = c.get("metadata", {})
        page = m.get("page", "?")
        text = c["text"].strip().replace("\n", " ")
        if len(text) > 700:
            text = text[:700] + "…"
        lines.append(f"[p.{page}] {text}")
    return "\n\n".join(lines)

def format_citations(ctxs: List[Dict]) -> str:
    """Gera lista única de páginas como links para o PDF oficial (markdown)."""
    base = "https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_LongerReport.pdf#page="
    pages: list[int] = []
    for c in ctxs:
        p = c.get("metadata", {}).get("page", None)
        try:
            p = int(p)
            pages.append(p)
        except (TypeError, ValueError):
            continue
    pages = sorted(set(pages))
    return " ".join(f"[p.{p}]({base}{p})" for p in pages) if pages else "[p.?]"

def answer(query: str, ctxs: List[Dict]) -> Dict:
    context_text = _build_context(ctxs)
    user = (
        "Pergunta:\n"
        f"{query}\n\n"
        "Trechos do IPCC (use APENAS o que está abaixo):\n"
        f"{context_text}\n\n"
        "Instruções de citação: inclua [p.X] após cada afirmação suportada."
    )
    msg = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)]
    resp = llm.invoke(msg)
    response_text = _normalize_citations(resp.content.strip())

    # Fallback mínimo caso o modelo ignore a regra de citação
    if "[p." not in response_text and "Não encontrei evidência" not in response_text:
        response_text += "\n\n(Cuidado: nenhuma citação detectada; verifique evidências)"

    response_text += "\n\nCitações: " + format_citations(ctxs)
    return {"answer": response_text, "contexts": ctxs}
