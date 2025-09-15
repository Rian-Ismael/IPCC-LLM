from typing import Literal

def route_intent(query: str) -> Literal["ipcc"]:
    # PoC: a single domain (IPCC)
    return "ipcc"