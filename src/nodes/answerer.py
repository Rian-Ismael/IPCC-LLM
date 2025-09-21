# src/nodes/answerer.py
from typing import List, Dict
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
import os, re
from dotenv import load_dotenv

load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

# modelo fraco: temperatura baixa p/ obedecer formato e não "viajar"
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.1)

SYSTEM_PROMPT = """Use ONLY the provided excerpts from the IPCC AR6 Synthesis Report – Longer Report.

Rules (strict):
- English; keep IPCC terms. No external knowledge.
- Reproduce numbers/units/scenario labels exactly as shown.
- Mention scenarios/time windows/qualifiers ONLY if present in excerpts.
- Every factual sentence MUST end with a page citation like [p.X].
  If multiple excerpts support it, append multiple citations separately: [p.X][p.Y].
  Use ONLY pages present in the excerpts for that sentence.
- If excerpts are insufficient, answer EXACTLY:
I have not found sufficient evidence in the IPCC to answer with confidence.
- Do NOT output the sentence above if you wrote any sentence with [p.X].
- Keep it compact: 1–5 bullets OR 2–3 short sentences.
"""

# normalização de citações → [p.X]
_CIT_PATTS = [
    (re.compile(r"\(p\.\s*\[(\d+)\]\)", re.I), r"[p.\1]"),
    (re.compile(r"p\.\s*\[(\d+)\]", re.I), r"[p.\1]"),
    (re.compile(r"\(p\.?\s*(\d+)\)", re.I), r"[p.\1]"),
    (re.compile(r"\[p\s*\.?\s*(\d+)\]", re.I), r"[p.\1]"),
    (re.compile(r"\[pg\.?\s*(\d+)\]", re.I), r"[p.\1]"),
]
_CIT_FINDER = re.compile(r"\[p\.(\d+)\]")  # páginas citadas no corpo

def _normalize_citations(txt: str) -> str:
    for patt, rep in _CIT_PATTS:
        txt = patt.sub(rep, txt)
    # deduplica citações consecutivas idênticas: [p.37][p.37] -> [p.37]
    txt = re.sub(r'(\[p\.\d+\])(?:\1)+', r'\1', txt)
    return txt

def _pages_used_in(text: str) -> list[int]:
    pages = [int(m.group(1)) for m in _CIT_FINDER.finditer(text)]
    return sorted(set(pages))

def _build_context(ctxs: List[Dict]) -> str:
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

def _format_citations_links(pages: list[int]) -> str:
    base = "https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_LongerReport.pdf#page="
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
    # fallback se não há contexto
    if not ctxs:
        refuse = f"{FALLBACK}\n\nCitations: [p.?]"
        return {"answer": refuse, "contexts": []}

    context_text = _build_context(ctxs)
    user = (
        "Question:\n"
        f"{query}\n\n"
        "IPCC excerpts (use ONLY what is below):\n"
        f"{context_text}\n\n"
        "Remember: put [p.X] (or multiple like [p.X][p.Y]) at the END of each factual sentence."
    )

    msg = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)]
    raw = llm.invoke(msg)
    response_text = _extract_text(raw).strip()
    if not response_text:
        raw = llm.invoke(msg)
        response_text = _extract_text(raw).strip()

    # normaliza e deduplica citações
    response_text = _normalize_citations(response_text)

    # páginas permitidas (vindas dos contextos)
    allowed_pages = sorted({
        int(c.get("metadata", {}).get("page"))
        for c in ctxs
        if str(c.get("metadata", {}).get("page", "")).isdigit()
    })

    # páginas realmente citadas no corpo, filtradas pelas permitidas
    used_pages = [p for p in _pages_used_in(response_text) if p in allowed_pages]
    if not used_pages:
        # se o modelo não citou nada no corpo, use páginas dos contextos como fallback de rodapé
        used_pages = allowed_pages

    response_text += "\n\nCitations: " + _format_citations_links(used_pages)
    return {"answer": response_text, "contexts": ctxs}
