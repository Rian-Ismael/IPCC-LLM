from typing import Dict, Any

class Supervisor:
    """
    Coordinator agent (no keyword heuristics).
    Decisions are based ONLY on state:
      - if no answer yet and no contexts -> retrieve
      - if no answer yet but contexts exist -> answer
      - if answer exists and ok is None -> selfcheck
      - if ok == False:
          - if should_retry -> retrieve
          - else -> safety
      - if ok == True -> safety

    * Retry budget (should_retry) is set by selfcheck node (not here).
    * Pure state-driven router (no string matching).
    """

    def __call__(self, s: Dict[str, Any]) -> Dict[str, Any]:
        # Ensure defaults
        s.setdefault("tries", 0)
        s.setdefault("agent_logs", [])
        s["agent_logs"].append(
            f"[Supervisor] state: has_answer={isinstance(s.get('answer'), dict)} "
            f"contexts={len(s.get('contexts', []) or [])} ok={s.get('ok')}"
        )
        return s

    def decide_next(self, s: Dict[str, Any]) -> str:
        has_answer = isinstance(s.get("answer"), dict)
        has_contexts_key = "contexts" in s  # may be empty list when retrieve ran but found nothing
        ok = s.get("ok", None)
        should_retry = bool(s.get("should_retry", False))

        # Entry / before answering
        if not has_answer:
            # no answer yet
            if not has_contexts_key:
                return "retrieve"
            return "answer"

        # After answer, we need to validate
        if ok is None:
            return "selfcheck"

        # After selfcheck
        if ok is False:
            return "retrieve" if should_retry else "safety"

        # ok is True
        return "safety"
