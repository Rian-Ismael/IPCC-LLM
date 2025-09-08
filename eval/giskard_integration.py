# eval/giskard_integration.py
import os, json, pandas as pd, time
from pathlib import Path
from dotenv import load_dotenv

# Chroma persistente (igual ingest)
from chromadb import PersistentClient

# ===== Giskard (evitar import circular!) =====
from giskard.rag.knowledge_base import KnowledgeBase
from giskard.rag.testset import QATestset
from giskard.rag.evaluate import evaluate, RAGReport
from giskard.rag.datatypes import AgentAnswer
from giskard.rag.metrics.ragas_metrics import ragas_context_recall, ragas_context_precision
from giskard.llm.embeddings import SentenceTransformerEmbedding
# ============================================

from src.graph import build_graph

load_dotenv()
INDEX_DIR = os.getenv("INDEX_DIR", "data/index")
EVAL_SRC = Path("eval/eval_set.jsonl")           # seu arquivo (question + gold_page)
TESTSET_PATH = Path("eval/ipcc_testset.jsonl")   # arquivo no formato Giskard
REPORT_DIR = Path("eval/reports")

# --- Abre coleção Chroma ---
DB = PersistentClient(path=INDEX_DIR)
try:
    COLL = DB.get_collection(name="ipcc")
except Exception:
    raise SystemExit(
        f"❌ Coleção 'ipcc' não encontrada em {INDEX_DIR}.\n"
        "Rode a ingestão primeiro:\n"
        "  python -m ingest.build_index --pdf data\\corpus\\IPCC_AR6_SYR_LongerReport.pdf --index-dir data\\index"
    )

probe = COLL.get(limit=1, include=["documents"])
if not probe.get("documents"):
    raise SystemExit(
        f"❌ Coleção 'ipcc' está vazia em {INDEX_DIR}.\n"
        "Recrie o índice:\n"
        "  Remove-Item -Recurse -Force data\\index\n"
        "  python -m ingest.build_index --pdf data\\corpus\\IPCC_AR6_SYR_LongerReport.pdf --index-dir data\\index"
    )

# --- Monta Knowledge Base do Giskard com embedding local ---
res = COLL.get(include=["metadatas", "documents"])
docs, metas = res["documents"], res["metadatas"]

rows = []
for txt, meta in zip(docs, metas):
    rows.append({"text": txt, "page": meta.get("page", "?"), "section": meta.get("section", "")})
kb_df = pd.DataFrame(rows)

embedding = SentenceTransformerEmbedding("sentence-transformers/all-MiniLM-L6-v2")
knowledge_base = KnowledgeBase(kb_df, embedding_model=embedding)

# --- Constrói testset Giskard a partir do seu eval_set.jsonl (sem OpenAI) ---
def build_testset_from_evalset(src_path: Path, dst_path: Path):
    if not src_path.exists():
        raise SystemExit(f"❌ Não encontrei {src_path.as_posix()}. Crie o arquivo com linhas JSON: "
                         '{"question":"...", "gold_page": 34}')

    # indexa docs por página para pegar contexto verdadeiro
    page_to_chunks = {}
    for t, m in zip(docs, metas):
        p = m.get("page", "?")
        page_to_chunks.setdefault(str(p), []).append(t)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with src_path.open("r", encoding="utf-8") as fin, dst_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            q = item.get("question")
            gp = str(item.get("gold_page"))

            # monta um reference_context simples com 1–2 chunks da página
            chunks = page_to_chunks.get(gp, [])
            if not chunks:
                # se não achar, deixa um contexto placeholder (a métrica vai penalizar)
                ref_ctx = f"[p.{gp}] (nenhum chunk encontrado nesta página no índice)"
            else:
                # concatena 1–2 trechos curtos
                picked = chunks[:2]
                short = []
                for c in picked:
                    c1 = c.replace("\n", " ").strip()
                    if len(c1) > 500:
                        c1 = c1[:500] + "…"
                    short.append(c1)
                ref_ctx = " ".join(f"[p.{gp}] {s}" for s in short)

            # QATestset do Giskard suporta referência livre; aqui não usamos answer, só métricas de contexto
            rec = {
                "question": q,
                "reference_answer": "",
                "reference_context": ref_ctx,
                "conversation_history": [],
                "metadata": {"page": gp}
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    if written == 0:
        raise SystemExit(f"❌ {src_path.as_posix()} está vazio.")
    print(f"✅ Test set Giskard gerado: {dst_path.as_posix()} ({written} perguntas)")

# se não existir o testset no formato Giskard, gera agora a partir do seu eval_set.jsonl
if not TESTSET_PATH.exists():
    build_testset_from_evalset(EVAL_SRC, TESTSET_PATH)

# carrega o testset para o Giskard
testset = QATestset.load(TESTSET_PATH.as_posix())

# --- Grafo LangGraph ---
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

# --- Avaliação Giskard ---
report = evaluate(
    answer_fn,
    testset=testset,
    knowledge_base=knowledge_base,
    metrics=[ragas_context_recall, ragas_context_precision],
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
report.save(REPORT_DIR.joinpath("ipcc_raget_report"))
with REPORT_DIR.joinpath("ipcc_raget_report.html").open("w", encoding="utf-8") as f:
    f.write(report.to_html(embed=True))

print("✅ Giskard RAGET finalizado → eval/reports/ipcc_raget_report.html")
