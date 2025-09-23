# tests/smoke_graph.py
import json
from src.graph import build_graph

def pretty(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))

def run(q):
    g = build_graph()
    return g.invoke({"query": q})

if __name__ == "__main__":
    print(">>\n")
    print("#1 about/how-to (out-of-domain → fallback+disclaimer)")
    out1 = run("how to use this app?")
    pretty({
        "query": "how to use this app?",
        "answer": out1["answer"]["answer"][:300] + "..." if len(out1["answer"]["answer"])>300 else out1["answer"]["answer"],
        "has_contexts": bool(out1.get("contexts")),
    })
    print("---\n")

    print("#2 factual (should retrieve+answer+selfcheck+disclaimer)")
    out2 = run("What are the main drivers of recent global warming?")
    pretty({
        "query": "What are the main drivers of recent global warming?",
        "answer_head": out2["answer"]["answer"][:400],
    })
    print("---\n")

    print("#3 out-of-domain hard (fallback+disclaimer)")
    out3 = run("como usar um cavalo?")
    pretty({
        "query": "como usar um cavalo?",
        "answer": out3["answer"]["answer"],
    })
    print("---")
