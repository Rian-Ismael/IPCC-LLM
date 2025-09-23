import importlib

def test_build_graph_invoke_shape():
    mod = importlib.import_module("src.graph")
    graph = mod.build_graph()
    out = graph.invoke({"query": "What is assessed in SYR?", "contexts": [], "answer": {}})
    assert isinstance(out, dict)
    ans = out.get("answer", {})
    if isinstance(ans, dict):
        txt = ans.get("answer", "")
    else:
        txt = out.get("answer", "")
    assert isinstance(txt, str)
    ctxs = out.get("contexts") or (ans.get("contexts") if isinstance(ans, dict) else [])
    assert isinstance(ctxs, list)
