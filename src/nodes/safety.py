from typing import Dict

FALLBACK = "I have not found sufficient evidence in the IPCC to answer with confidence."

DISCLAIMER = (
    "Nota: síntese com trechos do IPCC AR6 (SYR). "
    "Verifique sempre o relatório completo para evidências, escopo e incertezas."
)
def apply_safety(ans_obj: Dict) -> Dict:
    if not isinstance(ans_obj, dict):
        return ans_obj
    
    if ans_obj.get("rejected"):
        return ans_obj
    ans_txt = ans_obj.get("answer" or "").strip()
    if ans_txt and "Nota:" not in ans_txt:
       ans_obj = dict(ans_obj)
       ans_obj["answer"] = f"{ans_txt}\n\n{DISCLAIMER}"
    return ans_obj