import importlib
import time
import os

def _get_graph():
    os.environ.setdefault("MODE_DEMO", "1")  # força modo leve se existir
    mod = importlib.import_module("src.graph")
    return mod.build_graph()

def test_graph_returns_minimal_contract():
    g = _get_graph()
    out = g.invoke({"query": "What is assessed in SYR?", "contexts": [], "answer": {}})
    assert isinstance(out, dict)

    ans = out.get("answer", {})
    txt = ans.get("answer") if isinstance(ans, dict) else out.get("answer")
    assert isinstance(txt, str)

    ctxs = out.get("contexts") or (ans.get("contexts") if isinstance(ans, dict) else [])
    assert isinstance(ctxs, list)

def test_graph_is_reasonably_fast():
    g = _get_graph()
    t0 = time.time()
    _ = g.invoke({"query": "Give a short overview", "contexts": [], "answer": {}})
    assert (time.time() - t0) < 5.0, "Resposta do grafo está lenta (>5s) no modo leve"

def test_graph_is_deterministic_in_demo_mode():
    g = _get_graph()
    q = "Determinism check in demo mode"
    out1 = g.invoke({"query": q, "contexts": [], "answer": {}})
    out2 = g.invoke({"query": q, "contexts": [], "answer": {}})
    a1 = (out1.get("answer", {}) or {}).get("answer") or out1.get("answer", "")
    a2 = (out2.get("answer", {}) or {}).get("answer") or out2.get("answer", "")
    assert a1 == a2, "Respostas diferentes com mesma entrada em MODE_DEMO=1"
