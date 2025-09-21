# src/nodes/answerer.py
from typing import List, Dict
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
import os, re
from dotenv import load_dotenv

load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

# modelo fraco → temp baixa pra reduzir alucinação, mas sem engessar formato
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)

SYSTEM_PROMPT = """Use ONLY the provided excerpts from the IPCC AR6 Synthesis Report – Longer Report.

Rules (strict):
- English; keep IPCC terminology. No external knowledge.
- Reproduce numbers, units, ranges, scenario labels (e.g., SSP1-2.6) and confidence terms EXACTLY as written.
- Mention scenarios, time windows, regions or qualifiers ONLY if they appear in the excerpts.
- Every factual sentence MUST end with a page citation like [p.X]. If multiple excerpts support a sentence, you may add multiple citations like [p.X][p.Y].
- Use ONLY page numbers that appear in the provided excerpts.
- When multiple provided excerpts add directly relevant quantified details, include them as additional bullet points (up to ~6), each with its own [p.X].
- If the excerpts are insufficient to answer, reply EXACTLY:
I have not found sufficient evidence in the IPCC to answer with confidence.
- Prefer clarity and completeness when the excerpts justify it; do not omit supported details.

"""

# Normalização de variações de citação → [p.X]
_CIT_PATTS = [
    (re.compile(r"\(p\.\s*\[(\d+)\]\)", re.I), r"[p.\1]"),
    (re.compile(r"p\.\s*\[(\d+)\]", re.I), r"[p.\1]"),
    (re.compile(r"\(p\.?\s*(\d+)\)", re.I), r"[p.\1]"),
    (re.compile(r"\[p\s*\.?\s*(\d+)\]", re.I), r"[p.\1]"),
    (re.compile(r"\[pg\.?\s*(\d+)\]", re.I), r"[p.\1]"),
]

def _normalize_citations(txt: str) -> str:
    for patt, rep in _CIT_PATTS:
        txt = patt.sub(rep, txt)
    return txt

def _build_context(ctxs: List[Dict]) -> str:
    """Formata os excertos recuperados como contexto, um por parágrafo, prefixados com [p.X]."""
    if not ctxs:
        return "(no retrieved excerpts)"
    lines = []
    for c in ctxs:
        m = c.get("metadata", {}) or {}
        page = m.get("page", "?")
        text = (c.get("text") or "").strip().replace("\n", " ")
        if len(text) > 700:  # só para não explodir o prompt
            text = text[:700] + "…"
        lines.append(f"[p.{page}] {text}")
    return "\n\n".join(lines)

def format_citations(ctxs: List[Dict]) -> str:
    """Lista TODAS as páginas dos contextos, como links para o PDF oficial."""
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

FALLBACK = "I have not found sufficient evidence in the IPCC to answer with confidence."

def answer(query: str, ctxs: List[Dict]) -> Dict:
    # Sem contexto → recusa limpa
    if not ctxs:
        refuse = f"{FALLBACK}\n\nCitations: [p.?]"
        return {"answer": refuse, "contexts": []}

    context_text = _build_context(ctxs)
    user = (
        "Question:\n"
        f"{query}\n\n"
        "IPCC excerpts (use ONLY what is below):\n"
        f"{context_text}\n\n"
        "Remember: end EACH factual sentence with [p.X] (or multiple like [p.X][p.Y])."
    )
    msg = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)]

    raw = llm.invoke(msg)
    response_text = _extract_text(raw).strip()
    if not response_text:
        raw = llm.invoke(msg)
        response_text = _extract_text(raw).strip()

    # normaliza as citações para [p.X]
    response_text = _normalize_citations(response_text)

    # Rodapé: TODAS as páginas dos contextos (sem limitar)
    response_text += "\n\nCitations: " + format_citations(ctxs)
    return {"answer": response_text, "contexts": ctxs}
