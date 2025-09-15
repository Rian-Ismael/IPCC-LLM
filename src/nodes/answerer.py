# LLM + prompting utilities (keep this in your src/… module as you prefer)
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
- Respond ONLY in English, keeping technical IPCC terms.
- Each factual statement must cite the page(s) in the exact format [p.X].
- If the provided excerpts do not contain enough evidence to answer, reply only with:
  "I have not found sufficient evidence in the IPCC to answer with confidence."
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
    """Formats retrieved excerpts as context, one per paragraph."""
    if not ctxs:
        return "(no retrieved excerpts)"
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
    """Generates a unique page list as links to the official PDF (markdown)."""
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
        refuse = "I have not found sufficient evidence in the IPCC to answer with confidence."
        refuse += "\n\nCitations: [p.?]"
        return {"answer": refuse, "contexts": []}

    context_text = _build_context(ctxs)
    user = (
        "Question:\n"
        f"{query}\n\n"
        "IPCC excerpts (use ONLY what is below):\n"
        f"{context_text}\n\n"
        "Citation rule: include [p.X] right after each supported claim."
    )
    msg = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)]

    raw = llm.invoke(msg)
    response_text = _extract_text(raw).strip()
    if not response_text:
        raw = llm.invoke(msg)
        response_text = _extract_text(raw).strip()

    response_text = _normalize_citations(response_text)

    if "[p." not in response_text and "I have not found sufficient evidence" not in response_text:
        response_text += "\n\n(Caution: no citation detected; please verify evidence)"

    response_text += "\n\nCitations: " + format_citations(ctxs)
    return {"answer": response_text, "contexts": ctxs}
