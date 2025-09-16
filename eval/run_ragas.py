# eval/run_ragas.py
import os, json, time, csv
from pathlib import Path
import pandas as pd

from dotenv import load_dotenv
from chromadb import PersistentClient

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.graph import build_graph

# Estabilidade
os.environ.setdefault("RAGAS_MAX_WORKERS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

load_dotenv()

INDEX_DIR = os.getenv("INDEX_DIR", "data/index")
EVAL_SET = Path("eval/eval_set.jsonl")
REPORT_DIR = Path("eval/reports"); REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Chroma: valida coleção
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
kb_df = pd.DataFrame(
    [{"text": t, "page": str(m.get("page", "?"))} for t, m in zip(docs, metas)]
)

# ---- Eval set: leitura robusta (BOM/linhas vazias/comentários)
if not EVAL_SET.exists():
    raise SystemExit('❌ Falta eval/eval_set.jsonl (question + gold_page).')

items = []
with EVAL_SET.open("r", encoding="utf-8-sig") as f:
    for raw in f:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            # ignora linhas ruins silenciosamente
            continue
        q = obj.get("question"); gp = obj.get("gold_page")
        if q is None or gp is None:
            continue
        items.append({"question": str(q), "gold_page": str(gp)})
if not items:
    raise SystemExit("❌ eval/eval_set.jsonl está vazio ou mal formatado.")

# ---- Util
def ground_truth_for_page(page: str, max_snippets: int = 3) -> list[str]:
    rows = kb_df[kb_df["page"] == str(page)]["text"].tolist()
    out = []
    for r in rows[:max_snippets]:
        txt = r.replace("\n", " ").strip()
        if len(txt) > 800:
            txt = txt[:800] + "…"
        out.append(txt)
    return out or [f"[p.{page}] (nenhum chunk encontrado para esta página no índice)"]

# ---- Executa o grafo para cada pergunta
graph = build_graph()

user_inputs, responses, retrieved_contexts, references_list, latencies = [], [], [], [], []

for it in items:
    q = it["question"]; gp = it["gold_page"]

    t0 = time.time()
    out = graph.invoke({"query": q, "contexts": [], "answer": {}})
    lat = time.time() - t0

    ans = out.get("answer", {}).get("answer", "")
    ctxs = []
    for c in out.get("contexts", []) or []:
        t = (c.get("text") or "").replace("\n", " ").strip()
        if t:
            ctxs.append(t)

    # RAGAS 0.3.x espera reference como list[str]
    ref_snips = ground_truth_for_page(gp)

    user_inputs.append(q)
    responses.append(ans)
    retrieved_contexts.append(ctxs)
    references_list.append(ref_snips)
    latencies.append(lat)

print("Primeiro exemplo:", {
    "user_input": user_inputs[0],
    "response_head": responses[0][:160],
    "contexts_count": len(retrieved_contexts[0]),
    "reference_count": len(references_list[0]),
})

# ---- LLM do RAGAS: sem JSON-mode pra evitar parser error em modelos pequenos
llm = LangchainLLMWrapper(
    ChatOllama(
        model=os.getenv("RAGAS_LLM", os.getenv("OLLAMA_MODEL", "llama3.2:3b")),
        temperature=0,
        num_ctx=2048,
        num_predict=512,
    )
)

# ---- Embeddings corretas (wrapper LC -> RAGAS)
lc_emb = HuggingFaceEmbeddings(
    model_name=os.getenv("RAGAS_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
)
embeddings = LangchainEmbeddingsWrapper(lc_emb)

# ---- Dataset no schema esperado
data = Dataset.from_dict({
    "user_input": user_inputs,                # list[str]
    "response": responses,                    # list[str]
    "retrieved_contexts": retrieved_contexts, # list[list[str]]
    "reference": references_list,             # list[list[str]]
})

# ---- Métricas pedidas
metrics = [faithfulness, answer_relevancy]
run_cfg = RunConfig(max_workers=1, timeout=180)

result = evaluate(
    data,
    metrics=metrics,
    llm=llm,
    embeddings=embeddings,
    run_config=run_cfg,
)

# ---- Persistência
def _as_float(v):
    try:
        return float(v)
    except Exception:
        try:
            return float(getattr(v, "score", "nan"))
        except Exception:
            try:
                return float(v.get("score"))
            except Exception:
                return float("nan")

scores = {m.name: _as_float(result[m.name]) for m in metrics}
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
