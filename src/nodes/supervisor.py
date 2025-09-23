from typing import Dict, Any, Literal
from src.nodes.answerer import FALLBACK

class Supervisor:
    def __call__(self, s: Dict[str, Any]) -> Dict[str, Any]:
        s.setdefault("tries", 0)
        s.setdefault("stage", "start")
        s.setdefault("agent_logs", [])

        ans_txt = (s.get("answer") or {}).get("answer")
        ans_flag = "FALLBACK" if ans_txt == FALLBACK else ("OK" if ans_txt else "None")

        s["agent_logs"].append(
            f"[Supervisor] stage={s['stage']} tries={s['tries']} "
            f"contexts={len(s.get('contexts', []) or [])} ans={ans_flag}"
        )
        return s

    def decide_next(self, s: Dict[str, Any]) -> Literal["retrieve", "answer", "selfcheck", "safety"]:
        stage = s.get("stage", "start")

        if stage == "start":
            return "retrieve"
        if stage == "retrieved":
            return "answer"
        if stage == "answered":
            return "selfcheck"
        if stage == "retry":
            return "retrieve"
        if stage == "safety":
            return "safety"

        return "safety"
