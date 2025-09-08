import os, pandas as pd, time
from dotenv import load_dotenv
from chromadb import Client
from chromadb.config import Settings

from giskard.rag import KnowledgeBase, QATestset, RAGReport, evaluate, AgentAnswer, generate_testset
from giskard.rag.metrics.ragas_metrics import ragas_context_recall, ragas_context_precision

from src.graph import build_graph

load_dotenv()
INDEX_DIR = os.getenv("INDEX_DIR", "data/index")
DB = Client(Settings(persist_directory=INDEX_DIR))
COLL = DB.get_collection(name="ipcc")

# Recupera docs do Chroma
res = COLL.get(include=["metadatas", "documents"])
docs = res["documents"]
metas = res["metadatas"]
rows = []
for txt, meta in zip(docs, metas):
    rows.append({"text": txt, "page": meta.get("page", "?"), "section": meta.get("section", "")})
kb_df = pd.DataFrame(rows)

# KnowledgeBase
knowledge_base = KnowledgeBase(kb_df.rename(columns={"text": "text"}))

# Testset (gera se não existir)
TESTSET_PATH = "eval/ipcc_testset.jsonl"
if os.path.exists(TESTSET_PATH):
    testset = QATestset.load(TESTSET_PATH)
else:
    testset = generate_testset(
        knowledge_base, num_questions=25,
        agent_description="Chatbot RAG sobre o IPCC AR6 SYR (Longer Report)"
    )
    testset.save(TESTSET_PATH)

# Grafo LangGraph
graph = build_graph()


def answer_fn(question: str, history: list[dict] | None = None) -> AgentAnswer:
    start = time.time()
    out = graph.invoke({"query": question, "contexts": [], "answer": {}})
    latency = time.time() - start
    answer_text = out["answer"]["answer"] + f"\n\n[latency={latency:.2f}s]"
    sources = []
    for c in out.get("contexts", []):
        m = c.get("metadata", {})
        page = m.get("page", "?")
        sources.append(f"(p.{page}) {c['text']}")
    return AgentAnswer(message=answer_text, documents=sources)

# Avaliação Giskard
report = evaluate(
    answer_fn,
    testset=testset,
    knowledge_base=knowledge_base,
    metrics=[ragas_context_recall, ragas_context_precision],
)

os.makedirs("eval/reports", exist_ok=True)
report.save("eval/reports/ipcc_raget_report")
with open("eval/reports/ipcc_raget_report.html", "w", encoding="utf-8") as f:
    f.write(report.to_html(embed=True))

print("✅ Giskard RAGET finalizado → eval/reports/ipcc_raget_report.html")