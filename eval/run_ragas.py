# eval/run_ragas.py
import os, json, time, csv
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from chromadb import PersistentClient
from src.graph import build_graph

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.embeddings import HuggingFaceEmbeddings  # <- embeddings locais do RAGAS
from langchain_ollama import ChatOllama

# estabilidade
os.environ.setdefault("RAGAS_MAX_WORKERS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

load_dotenv()

INDEX_DIR = os.getenv("INDEX_DIR", "data/index")
EVAL_SET = Path("eval/eval_set.jsonl")
REPORT_DIR = Path("eval/reports"); REPORT_DIR.mkdir(parents=True, exist_ok=True)

# --- Chroma
db = PersistentClient(path=INDEX_DIR)
try:
    coll = db.get_collection("ipcc")
except Exception:
    raise SystemExit(
        f"❌ Coleção 'ipcc' não encontrada em {INDEX_DIR}.\n"
        "Rode a ingestão:\n"
        "  python -m ingest.build_index --pdf data\\corpus\\IPCC_AR6_SYR_LongerReport.pdf --index-dir data\\index"
    )

probe = coll.get(limit=1, include=["documents"])
if not probe.get("documents"):
    raise SystemExit(
        f"❌ Coleção 'ipcc' está vazia em {INDEX_DIR}.\n"
        "Recrie o índice:\n"
        "  Remove-Item -Recurse -Force data\\index\n"
        "  python -m ingest.build_index --pdf data\\corpus\\IPCC_AR6_SYR_LongerReport.pdf --index-dir data\\index"
    )

dump = coll.get(include=["documents", "metadatas"])
docs, metas = dump["documents"], dump["metadatas"]
kb_df = pd.DataFrame([{"text": t, "page": str(m.get("page", "?"))} for t, m in zip(docs, metas)])

# --- Eval set
if not EVAL_SET.exists():
    raise SystemExit('❌ Falta eval/eval_set.jsonl (question + gold_page).')

items = []
with EVAL_SET.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            q = obj.get("question"); gp = obj.get("gold_page")
            if q is None or gp is None:
                continue
            items.append({"question": str(q), "gold_page": str(gp)})
if not items:
    raise SystemExit("❌ eval/eval_set.jsonl está vazio ou mal formatado.")

graph = build_graph()

def ground_truth_for_page(page: str, max_snippets: int = 3) -> list[str]:
    rows = kb_df[kb_df["page"] == str(page)]["text"].tolist()
    gts = []
    for r in rows[:max_snippets]:
        txt = r.replace("\n", " ").strip()
        if len(txt) > 800:
            txt = txt[:800] + "…"
        gts.append(txt)
    return gts or [f"[p.{page}] (nenhum chunk encontrado nesta página no índice)"]

# --- Executa perguntas
user_inputs, responses, retrieved_contexts, references, latencies = [], [], [], [], []

for it in items:
    q = it["question"]; gp = it["gold_page"]
    t0 = time.time()
    out = graph.invoke({"query": q, "contexts": [], "answer": {}})
    latencies.append(time.time() - t0)

    ans = out["answer"]["answer"]
    ctxs = []
    for c in out.get("contexts", []):
        t = (c.get("text") or "").replace("\n", " ").strip()
        if t:
            ctxs.append(t)

    ref_str = "\n".join(ground_truth_for_page(gp))

    user_inputs.append(q)
    responses.append(ans)
    retrieved_contexts.append(ctxs)
    references.append(ref_str)

# --- LLM local p/ RAGAS (modelo obediente a JSON ajuda o parser)
llm = LangchainLLMWrapper(
    ChatOllama(
        model=os.getenv("RAGAS_LLM", "llama3.2:1b"),  # altere p/ "qwen2.5:3b" se preferir
        temperature=0,
        format="json",
        num_ctx=2048,
        num_predict=256,
    )
)

# --- Embeddings locais do RAGAS (evita fallback p/ OpenAI)
embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# --- Dataset no schema esperado (RAGAS 0.3.x)
data = Dataset.from_dict({
    "user_input": user_inputs,
    "response": responses,
    "retrieved_contexts": retrieved_contexts,
    "reference": references,
})

print("Primeiro exemplo de avaliação:", data[0])

# --- Métricas pedidas pela rubrica
metrics = [faithfulness, answer_relevancy]

# Timeout e 1 worker
run_cfg = RunConfig(max_workers=1, timeout=120)

result = evaluate(
    data,
    metrics=metrics,
    llm=llm,
    embeddings=embeddings,   # <- ESSENCIAL para não tentar OpenAI
    run_config=run_cfg,
)

# --- Salva
def as_float(v):
    for getter in (lambda x: float(x),
                   lambda x: float(x.get("score")),
                   lambda x: float(getattr(x, "score", float("nan")))):
        try:
            return getter(v)
        except Exception:
            pass
    return float("nan")

scores = {m.name: as_float(result[m.name]) for m in metrics}
scores["latency_avg_s"] = sum(latencies) / max(1, len(latencies))

with REPORT_DIR.joinpath("ragas_scores.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["metric", "value"])
    for k, v in scores.items():
        w.writerow([k, f"{v:.4f}" if isinstance(v, float) else v])

per_q = pd.DataFrame({
    "question": user_inputs,
    "contexts_count": [len(c) for c in retrieved_contexts],
    "latency_s": latencies,
})
for m in metrics:
    try:
        per_q[m.name] = [float(x) if x is not None else "" for x in result[m.name].samples]
    except Exception:
        pass

per_q.to_csv(REPORT_DIR.joinpath("ragas_details.csv"), index=False, encoding="utf-8")

print("✅ RAGAS finalizado!")
print(f"   → Médias: {REPORT_DIR/'ragas_scores.csv'}")
print(f"   → Detalhes: {REPORT_DIR/'ragas_details.csv'}")
