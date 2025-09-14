from typing import Dict
import re

RE_CIT = re.compile(
    r"\[(?:p|pg)\s*\.?\s*\d+\]|\(p\.?\s*\d+\)|p\.\s*\[\s*\d+\s*\]",
    re.I
)

def self_check(ans: Dict) -> Dict:
    txt = ans.get("answer", "")
    if RE_CIT.search(txt) or "Não encontrei evidência suficiente" in txt:
        return {"ok": True}
    if "Citações:" in txt and re.search(r"\bp\.\s*\d+\b", txt):
        return {"ok": True, "note": "Citações apenas no rodapé"}
    return {"ok": False, "reason": "Resposta sem citação de página do IPCC."}
