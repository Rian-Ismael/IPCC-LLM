from typing import Dict
import re

# aceita [p.34], [pg.34] ou (p. 34)
RE_CIT = re.compile(r"\[(?:p|pg)\.\s*\d+\]|\(p\.?\s*\d+\)", re.I)

def self_check(ans: Dict) -> Dict:
    txt = ans.get("answer", "")
    if RE_CIT.search(txt) or "Não encontrei evidência" in txt:
        return {"ok": True}
    return {"ok": False, "reason": "Resposta sem citação de página do IPCC."}
