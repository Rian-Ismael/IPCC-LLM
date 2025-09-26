from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict

from src.nodes.retriever import retrieve
from src.nodes.answerer import answer, FALLBACK
from src.nodes.selfcheck import self_check
from src.nodes.safety import apply_safety
from src.nodes.supervisor import Supervisor
from src.nodes.moderator import moderate, REJECTION_OFF_TOPIC, REJECTION_UNSAFE

class State(TypedDict, total=False):
    query: str
    contexts: List[Dict]
    answer: Dict
    stage: str
    tries: int
    agent_logs: List[str]

def build_graph():
    g = StateGraph(State)
    sup = Supervisor()

    def node_moderate(s: State):
        dec = moderate(s["query"])
        if dec == "reject_unsafe":
            s["answer"] = {"answer": REJECTION_UNSAFE, "contexts": [], "rejected": True}
            s["stage"] = "moderated_reject"
        elif dec == "reject_off_topic":
            s["answer"] = {"answer": REJECTION_OFF_TOPIC, "contexts": [], "rejected": True}
            s["stage"] = "moderated_reject"
        else:
            s["stage"] = "moderated_ok"
        return s

    def node_retrieve(s: State):
        s["contexts"] = retrieve(s["query"])
        s["stage"] = "retrieved"
        return s

    def node_answer(s: State):
        s["answer"] = answer(s["query"], s.get("contexts", []))
        s["stage"] = "answered"
        return s

    def node_selfcheck(s: State):
        s["answer"] = self_check(s.get("answer", {}))
        ans_txt = (s["answer"] or {}).get("answer", "")
        if ans_txt == FALLBACK and s.get("tries", 0) < 1:
            s["tries"] = s.get("tries", 0) + 1
            s["stage"] = "retry"
        else:
            s["stage"] = "safety"
        return s

    def node_safety(s: State):
        base = s.get("answer") or {"answer": FALLBACK, "contexts": s.get("contexts", [])}
        s["answer"] = apply_safety(base)
        s["stage"] = "safety"
        return s

    g.add_node("moderate", node_moderate)
    g.add_node("retrieve", node_retrieve)
    g.add_node("answer", node_answer)
    g.add_node("selfcheck", node_selfcheck)
    g.add_node("safety", node_safety)
    g.add_node("supervisor", sup)

    g.set_entry_point("supervisor")

    g.add_conditional_edges(
        "supervisor",
        sup.decide_next,
        {
            "moderate": "moderate",
            "retrieve": "retrieve",
            "answer": "answer",
            "selfcheck": "selfcheck",
            "safety": "safety",
            "end": END, 
        },
    )

    g.add_edge("moderate", "supervisor")
    g.add_edge("retrieve", "supervisor")
    g.add_edge("answer", "supervisor")
    g.add_edge("selfcheck", "supervisor")

    g.add_edge("safety", END)

    return g.compile()
