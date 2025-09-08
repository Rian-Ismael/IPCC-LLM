from src.graph import build_graph

def test_every_answer_has_citation_token():
    g = build_graph()
    out = g.invoke({"query": "What are near-term climate risks?", "contexts": [], "answer": {}})
    assert "[p." in out["answer"]["answer"] or "Não encontrei evidência" in out["answer"]["answer"]