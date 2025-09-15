from typing import Dict

def apply_safety(ans: Dict) -> Dict:
    disclaimer = (
        "\n\n---\n"
        "_Disclaimer_: Informational content based on IPCC AR6 (SYR). "
        "This does not replace official interpretation. See the full report."
    )
    ans["answer"] = ans.get("answer", "") + disclaimer
    return ans