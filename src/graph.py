# src/graph.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict

from src.nodes.supervisor import route_intent
from src.nodes.retriever import retrieve
from src.nodes.answerer import answer
from src.nodes.selfcheck import self_check
from src.nodes.safety import apply_safety


class State(TypedDict):
    query: str
    contexts: List[Dict]
    answer: Dict


def build_graph():
    g = StateGraph(State)

    def node_supervisor(s: State):
        _ = route_intent(s["query"])  # único destino no PoC
        return s

    def node_retrieve(s: State):
        s["contexts"] = retrieve(s["query"])
        return s

    def node_answer(s: State):
        s["answer"] = answer(s["query"], s["contexts"])
        return s

    def node_selfcheck(s: State):
        # ✅ NOVO: self_check já normaliza e devolve {"answer": ..., "contexts": ...}
        s["answer"] = self_check(s["answer"])
        return s

    def node_safety(s: State):
        s["answer"] = apply_safety(s["answer"])
        return s

    g.add_node("supervisor", node_supervisor)
    g.add_node("retrieve", node_retrieve)
    g.add_node("answer", node_answer)
    g.add_node("selfcheck", node_selfcheck)
    g.add_node("safety", node_safety)

    g.set_entry_point("supervisor")
    g.add_edge("supervisor", "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", "selfcheck")
    g.add_edge("selfcheck", "safety")
    g.add_edge("safety", END)

    return g.compile()
