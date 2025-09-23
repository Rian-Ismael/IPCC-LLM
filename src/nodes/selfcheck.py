from typing import Dict, Set
import re

RE_CIT = re.compile(r"\[p\.?\s*(\d+)\]", re.I)
FALLBACK = "I have not found sufficient evidence in the IPCC to answer with confidence."

def _strip_fallback(txt: str) -> str:
    out = txt.replace(FALLBACK, "").strip()
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out

def _pages_from_contexts(ctxs) -> Set[int]:
    pages: Set[int] = set()
    for c in (ctxs or []):
        pg = c.get("page")
        if pg is None:
            pg = (c.get("metadata") or {}).get("page")
        try:
            if pg is not None:
                pages.add(int(pg))
        except Exception:
            continue
    return pages

def self_check(ans: Dict) -> Dict:
    """
    Self-RAG (light) acceptance:
      - Every factual sentence must have [p.X] (enforced by answerer).
      - Coverage proxy: at least one cited page must belong to the pages of top-k contexts.
      - If not satisfied → refuse with FALLBACK (safe).
    Returns: {"answer": str, "contexts": list, "ok": bool}
    """
    txt = (ans or {}).get("answer", "") or ""
    ctxs = (ans or {}).get("contexts", []) or []

    has_refusal = FALLBACK in txt
    cited_pages = {int(m) for m in RE_CIT.findall(txt)}
    ctx_pages = _pages_from_contexts(ctxs)
    has_cit = len(cited_pages) > 0

    # Content with [p.X] + refusal → strip refusal and then check coverage
    if has_cit and has_refusal:
        cleaned = _strip_fallback(txt) or FALLBACK
        ok = cleaned != FALLBACK and len(cited_pages & ctx_pages) > 0
        return {"answer": cleaned, "contexts": ctxs, "ok": ok}

    # Only refusal (no citations) → keep refusal
    if (not has_cit) and has_refusal:
        return {"answer": FALLBACK, "contexts": ctxs, "ok": False}

    # Content without [p.X] → refuse
    if (not has_cit) and (not has_refusal):
        return {"answer": FALLBACK, "contexts": ctxs, "ok": False}

    # Content with [p.X] (no refusal) → accept if we have coverage against top-k pages
    ok = len(cited_pages & ctx_pages) > 0
    if not ok:
        # No explicit coverage → safe refusal
        return {"answer": FALLBACK, "contexts": ctxs, "ok": False}
    return {"answer": txt, "contexts": ctxs, "ok": True}
