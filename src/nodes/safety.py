from typing import Dict

FALLBACK = "I have not found sufficient evidence in the IPCC to answer with confidence."

def apply_safety(ans: Dict) -> Dict:
    txt = (ans or {}).get("answer", "") or ""
    disclaimer = (
        "\n\n---\n"
        "_Disclaimer_: Informational content based on IPCC AR6 (SYR). "
        "This does not replace official interpretation. See the full report."
    )
    ans["answer"] = txt + disclaimer
    return ans
