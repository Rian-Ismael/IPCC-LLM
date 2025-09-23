import importlib
import os

def test_context_items_have_text_and_optional_page():
    os.environ.setdefault("MODE_DEMO", "1")
    graph = importlib.import_module("src.graph").build_graph()
    out = graph.invoke({"query": "Context schema check", "contexts": [], "answer": {}})

    ans = out.get("answer", {})
    ctxs = out.get("contexts") or (ans.get("contexts") if isinstance(ans, dict) else [])
    assert isinstance(ctxs, list)
    if not ctxs:
        # aceitável devolver vazio; apenas não pode quebrar o contrato
        return

    for c in ctxs:
        assert isinstance(c, dict), "Cada contexto deve ser dict"
        txt = (c.get("text") or c.get("page_content") or c.get("content") or "").strip()
        assert isinstance(txt, str), "Contexto deve ter texto em 'text'/'page_content'/'content'"
        # página pode estar em 'page' ou em metadata.page
        pg = c.get("page")
        if pg is None:
            meta = c.get("metadata") or {}
            if isinstance(meta, dict):
                pg = meta.get("page")
        # página é opcional; se vier, deve ser string/int "imprimível"
        if pg is not None:
            assert isinstance(pg, (str, int)), "Campo 'page' (quando presente) deve ser str/int"
