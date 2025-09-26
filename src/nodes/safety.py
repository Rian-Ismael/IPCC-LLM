from typing import Dict

FALLBACK = "I have not found sufficient evidence in the IPCC to answer with confidence."

def apply_safety(ans: Dict) -> Dict:
    txt = (ans or {}).get("answer", "") or ""
    if txt.strip() == FALLBACK:
        return ans

    disclaimer = (
        "\n\n---\n"
        "_Aviso_: Conteúdo informativo com base no IPCC AR6 (SYR). "
        "Não substitui interpretação oficial. Consulte o relatório completo."
    )
    ans["answer"] = txt + disclaimer
    return ans
