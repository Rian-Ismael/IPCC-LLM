from typing import Dict

def apply_safety(ans: Dict) -> Dict:
    disclaimer = (
        "\n\n---\n"
        "_Disclaimer_: Conteúdo informativo baseado no IPCC AR6 (SYR). "
        "Não substitui interpretação oficial. Consulte o relatório completo."
    )
    ans["answer"] = ans.get("answer", "") + disclaimer
    return ans
