from typing import Dict
import re

RE_CIT = re.compile(
    r"\[(?:p|pg)\s*\.?\s*\d+\]|\(p\.?\s*\d+\)|p\.\s*\[\s*\d+\s*\]",
    re.I
)

def self_check(ans: Dict) -> Dict:
    txt = ans.get("answer", "")
    if RE_CIT.search(txt) or "I have not found sufficient evidence" in txt:
        return {"ok": True}
    if "Citations:" in txt and re.search(r"\bp\.\s*\d+\b", txt):
        return {"ok": True, "note": "Citations only in the footer"}
    return {"ok": False, "reason": "Response without IPCC page citation."}