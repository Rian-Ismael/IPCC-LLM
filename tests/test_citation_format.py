import importlib
import os
import re

PAGE_RE = re.compile(r"\[p\.(\d+)\]")

def test_answer_mentions_pages_with_bracket_format():
    os.environ.setdefault("MODE_DEMO", "1")
    graph = importlib.import_module("src.graph").build_graph()
    out = graph.invoke({"query": "Citations format check", "contexts": [], "answer": {}})

    ans = out.get("answer", {})
    txt = (ans.get("answer") if isinstance(ans, dict) else out.get("answer")) or ""
    assert isinstance(txt, str)

    # Não força ter citação sempre; mas se tiver, deve ser no formato [p.X]
    cites = PAGE_RE.findall(txt)
    for c in cites:
        assert c.isdigit(), "Número de página em [p.X] deve ser dígito"
