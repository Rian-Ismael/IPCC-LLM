from typing import List, Dict
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
import os, re
from dotenv import load_dotenv

load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct-q4_K_M")

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.0)

SYSTEM_PROMPT = """You are an assistant that answers ONLY based on the IPCC AR6 (SYR Longer Report).

Rules:
- Respond ONLY in Brazilian Portuguese (pt-BR), keeping technical IPCC terms.
- Each factual statement must cite the page(s) in the exact format [p.X].
- If the provided excerpts do not contain enough evidence to answer, reply only with:
  "Não encontrei evidência suficiente no IPCC para responder com segurança."
- Do not add this sentence if there is already enough evidence.
- Do not invent sources or use external knowledge.
- Be concise.
"""



_CIT_PATTS = [
    (re.compile(r"\(p\.\s*\[(\d+)\]\)", re.I), r"[p.\1]"),     # (p. [34]) → [p.34]
    (re.compile(r"p\.\s*\[(\d+)\]", re.I), r"[p.\1]"),         # p. [34]   → [p.34]
    (re.compile(r"\(p\.?\s*(\d+)\)", re.I), r"[p.\1]"),        # (p. 34)   → [p.34]
    (re.compile(r"\[p\s*\.?\s*(\d+)\]", re.I), r"[p.\1]"),     # [p 34]/[p. 34] → [p.34]
    (re.compile(r"\[pg\.?\s*(\d+)\]", re.I), r"[p.\1]"),       # [pg.34]   → [p.34]
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
        m = c.get("metadata", {}) or {}
        page = m.get("page", "?")
        text = (c.get("text") or "").strip().replace("\n", " ")
        if len(text) > 700:
            text = text[:700] + "…"
        lines.append(f"[p.{page}] {text}")
    return "\n\n".join(lines)

def format_citations(ctxs: List[Dict]) -> str:
    """Gera lista única de páginas como links para o PDF oficial (markdown)."""
    base = "https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_LongerReport.pdf#page="
    pages: list[int] = []
    for c in ctxs:
        p = (c.get("metadata") or {}).get("page", None)
        try:
            pages.append(int(p))
        except (TypeError, ValueError):
            continue
    pages = sorted(set(pages))
    return " ".join(f"[p.{p}]({base}{p})" for p in pages) if pages else "[p.?]"

def _extract_text(raw) -> str:
    try:
        if hasattr(raw, "content") and isinstance(raw.content, str):
            return raw.content
        if hasattr(raw, "text") and isinstance(raw.text, str):
            return raw.text
        if hasattr(raw, "generations"):
            gens = raw.generations
            if gens and gens[0] and hasattr(gens[0][0], "text"):
                return gens[0][0].text
    except Exception:
        pass
    return str(raw or "")

def answer(query: str, ctxs: List[Dict]) -> Dict:
    if not ctxs:
        refuse = "Não encontrei evidência suficiente no IPCC para responder com segurança."
        refuse += "\n\nCitações: [p.?]"
        return {"answer": refuse, "contexts": []}

    context_text = _build_context(ctxs)
    user = (
        "Pergunta:\n"
        f"{query}\n\n"
        "Trechos do IPCC (use APENAS o que está abaixo):\n"
        f"{context_text}\n\n"
        "Instruções de citação: inclua [p.X] após cada afirmação suportada."
    )
    msg = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)]

    raw = llm.invoke(msg)
    response_text = _extract_text(raw).strip()
    if not response_text:
        raw = llm.invoke(msg)
        response_text = _extract_text(raw).strip()

    response_text = _normalize_citations(response_text)

    if "[p." not in response_text and "Não encontrei evidência" not in response_text:
        response_text += "\n\n(Cuidado: nenhuma citação detectada; verifique evidências)"

    response_text += "\n\nCitações: " + format_citations(ctxs)
    return {"answer": response_text, "contexts": ctxs}
