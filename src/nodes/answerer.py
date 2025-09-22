from typing import List, Dict
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
import os, re, textwrap
from dotenv import load_dotenv

load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
    num_ctx=2048,
    num_predict=256,
    keep_alive="30m",
)

FALLBACK = "I have not found sufficient evidence in the IPCC to answer with confidence."

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

_STOPWORDS = {
    "the","a","an","and","or","of","on","in","at","to","for","with","by","from",
    "is","are","was","were","be","been","being","that","this","these","those",
    "what","which","who","whom","why","how","when","where","than","as","into",
    "about","over","under","it","its","their","there","we","you","your","our",
}

_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")

def _build_context(ctxs: List[Dict]) -> str:
    blocks = []
    for c in ctxs:
        txt = (c.get("text") or c.get("page_content") or "").strip()
        pg  = c.get("page") or (c.get("metadata") or {}).get("page")
        if not txt or not pg:
            continue
        blocks.append(f"[p.{pg}]\n{txt}")
    return "\n\n---\n\n".join(blocks)

def _normalize_citations(text: str) -> str:
    text = re.sub(r"\[\s*p\s*\.?\s*(\d+)\s*\]", r"[p.\1]", text)
    text = re.sub(r"\(\s*p\s*\.?\s*(\d+)\s*\)", r"[p.\1]", text)
    text = re.sub(r"\s+\[p\.(\d+)\]", r" [p.\1]", text)
    return text

def _has_any_citation(text: str) -> bool:
    return bool(re.search(r"\[p\.\d+\]", text))

def _keywords(q: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9\-\./]+", q.lower())
    return [t for t in toks if t not in _STOPWORDS and len(t) > 2]

def _extractive_fallback(query: str, ctxs: List[Dict], max_sents: int = 5, min_sents:int = 2) -> str:
    kws = set(_keywords(query))
    picked: List[str] = []
    seen = set()
    for c in ctxs:
        txt = (c.get("text") or c.get("page_content") or "").strip()
        pg  = c.get("page") or (c.get("metadata") or {}).get("page")
        if not txt or not pg:
            continue
        for sent in _SENT_SPLIT.split(txt):
            s = sent.strip()
            if not s:
                continue
            lower = s.lower()
            if kws and not any(k in lower for k in kws):
                continue
            sig = re.sub(r"\s+", " ", lower)
            if sig in seen:
                continue
            seen.add(sig)
            if not re.search(r"\[p\.\d+\]\s*$", s):
                s = s.rstrip(". ") + f" [p.{pg}]"
            picked.append(s)
            if len(picked) >= max_sents:
                break
        if len(picked) >= max_sents:
            break

    if len(picked) < min_sents:
        return FALLBACK

    if len(picked) == 1:
        return picked[0]
    return "\n".join(f"- {s}" for s in picked)

def answer(query: str, ctxs: List[Dict]) -> Dict:
    if not ctxs:
        return {"answer": FALLBACK, "contexts": []}

    context_text = _build_context(ctxs)
    user = textwrap.dedent(f"""
    Question:
    {query}

    IPCC excerpts (use ONLY what is below):
    {context_text}

    Remember: end EACH factual sentence with [p.X] (or multiple like [p.X][p.Y]).
    """)

    msgs = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)]
    out = llm.invoke(msgs)
    ans = (out.content or "").strip()
    ans = _normalize_citations(ans)

    if not ans or ans.strip() == FALLBACK or not _has_any_citation(ans):
        ans = _extractive_fallback(query, ctxs)

    ans = re.sub(r"[ \t]+", " ", ans).strip()
    return {"answer": ans, "contexts": ctxs}
