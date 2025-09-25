# src/graph.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict

from src.nodes.retriever import retrieve
from src.nodes.answerer import answer, FALLBACK
from src.nodes.selfcheck import self_check
from src.nodes.safety import apply_safety
from src.nodes.supervisor import Supervisor          # mantém o supervisor
from src.nodes.moderator import moderate, REJECTION_OFF_TOPIC, REJECTION_UNSAFE

class State(TypedDict, total=False):
    query: str
    contexts: List[Dict]
    answer: Dict
    stage: str

def build_graph():
    g = StateGraph(State)
    sup = Supervisor()  # você mantém sua lógica interna, logs, etc.

    # --- 1) Moderador: decide se rejeita ou prossegue ---
    def node_moderate(s: State):
        decision = moderate(s["query"])
        if decision == "reject_unsafe":
            s["answer"] = {"answer": REJECTION_UNSAFE, "contexts": [], "rejected": True}
            s["stage"] = "reject"
        elif decision == "reject_off_topic":
            s["answer"] = {"answer": REJECTION_OFF_TOPIC, "contexts": [], "rejected": True}
            s["stage"] = "reject"
        else:
            s["stage"] = "proceed"
        return s

    def decide_after_moderate(s: State) -> str:
        return "reject" if s.get("stage") == "reject" else "proceed"

    # --- 2) Supervisor (mantido), mas sem decidir aresta dinâmica agora ---
    #     Use o nó para logging/telemetria/heurística leve se quiser.
    def node_supervisor(s: State):
        _ = sup  # se quiser, chame um método de logging/roteamento interno aqui
        # forçando a 1ª passada a ir para retrieve
        s["stage"] = "go_retrieve"
        return s

    # --- 3) Retrieve → Answer → Self-Check → Safety (linha reta) ---
    def node_retrieve(s: State):
        s["contexts"] = retrieve(s["query"])
        return s

    def node_answer(s: State):
        # diagnóstico rápido: veja se vieram contextos
        print(f"[diag] ctxs: {len(s.get('contexts', []))}")
        s["answer"] = answer(s["query"], s.get("contexts", []))
        return s

    def node_selfcheck(s: State):
        s["answer"] = self_check(s.get("answer", {}))
        return s

    def node_safety(s: State):
        base = s.get("answer") or {"answer": FALLBACK, "contexts": s.get("contexts", [])}
        s["answer"] = apply_safety(base)
        return s

    # --- registro dos nós ---
    g.add_node("moderate", node_moderate)
    g.add_node("supervisor", node_supervisor)
    g.add_node("retrieve", node_retrieve)
    g.add_node("answer", node_answer)
    g.add_node("selfcheck", node_selfcheck)
    g.add_node("safety", node_safety)

    # --- entrada ---
    g.set_entry_point("moderate")

    # --- arestas condicionais só no moderador ---
    g.add_conditional_edges("moderate", decide_after_moderate, {
        "reject": "safety",
        "proceed": "supervisor",
    })

    # --- caminho determinístico depois do supervisor ---
    g.add_edge("supervisor", "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", "selfcheck")
    g.add_edge("selfcheck", "safety")
    g.add_edge("safety", END)

    return g.compile()