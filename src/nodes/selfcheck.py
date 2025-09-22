from typing import Dict
import re

RE_CIT = re.compile(r"\[p\.?\s*\d+\]", re.I)
FALLBACK = "I have not found sufficient evidence in the IPCC to answer with confidence."

def _strip_fallback(txt: str) -> str:
    out = txt.replace(FALLBACK, "").strip()
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out

def self_check(ans: Dict) -> Dict:
    """
    Tabela-verdade:
      1) Conteúdo com [p.X] + recusa  → remove recusa, mantém conteúdo.
      2) Só recusa (sem [p.X])        → mantém recusa limpa.
      3) Conteúdo sem [p.X]           → substitui por recusa limpa.
      4) Conteúdo com [p.X] (sem recusa) → passa.
    Retorna sempre {"answer": ..., "contexts": ...}
    """
    txt = (ans or {}).get("answer", "") or ""
    ctxs = (ans or {}).get("contexts", []) or []

    has_cit = bool(RE_CIT.search(txt))
    has_refusal = FALLBACK in txt

    if has_cit and has_refusal:
        cleaned = _strip_fallback(txt)
        cleaned = cleaned if cleaned else FALLBACK
        return {"answer": cleaned, "contexts": ctxs}

    if (not has_cit) and has_refusal:
        return {"answer": FALLBACK, "contexts": ctxs}

    if (not has_cit) and (not has_refusal):
        return {"answer": FALLBACK, "contexts": ctxs}

    return {"answer": txt, "contexts": ctxs}
