# src/graph.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict

from src.nodes.retriever import retrieve
from src.nodes.answerer import answer, FALLBACK
from src.nodes.selfcheck import self_check
from src.nodes.safety import apply_safety
from src.nodes.supervisor import Supervisor


class State(TypedDict, total=False):
    query: str
    contexts: List[Dict]
    answer: Dict
    ok: bool              # set by selfcheck
    tries: int
    should_retry: bool    # set by selfcheck
    agent_logs: List[str]


def build_graph():
    g = StateGraph(State)

    sup = Supervisor()

    # --- Nodes ---
    def node_retrieve(s: State):
        s["contexts"] = retrieve(s["query"])
        return s

    def node_answer(s: State):
        s["answer"] = answer(s["query"], s.get("contexts", []))
        return s

    def node_selfcheck(s: State):
        # self_check returns {"answer": str, "contexts": list, "ok": bool}
        res = self_check(s.get("answer", {}))
        s["answer"] = {
            "answer": res["answer"],
            "contexts": res.get("contexts", s.get("contexts", [])),
        }
        s["ok"] = bool(res.get("ok", False))

        # Retry budget control LIVES HERE (avoid mutating inside predicates)
        if not s["ok"]:
            s["tries"] = s.get("tries", 0)
            if s["tries"] < 1:
                s["tries"] += 1
                s["should_retry"] = True
            else:
                s["should_retry"] = False
        else:
            s["should_retry"] = False

        return s

    def node_safety(s: State):
        # Always attach disclaimer
        base_answer = s.get("answer") or {"answer": FALLBACK, "contexts": s.get("contexts", [])}
        s["answer"] = apply_safety(base_answer)
        return s

    # --- Register nodes ---
    g.add_node("supervisor", sup)
    g.add_node("retrieve", node_retrieve)
    g.add_node("answer", node_answer)
    g.add_node("selfcheck", node_selfcheck)
    g.add_node("safety", node_safety)

    g.set_entry_point("supervisor")

    # Supervisor decides next hop by STATE (no heuristics)
    g.add_conditional_edges(
        "supervisor",
        sup.decide_next,
        {
            "retrieve": "retrieve",
            "answer": "answer",
            "selfcheck": "selfcheck",
            "safety": "safety",
        },
    )

    # Workers → Supervisor
    g.add_edge("retrieve", "supervisor")
    g.add_edge("answer", "supervisor")
    g.add_edge("selfcheck", "supervisor")

    # Safety ends
    g.add_edge("safety", END)

    return g.compile()
