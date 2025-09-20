# eval/run_ragas.py
import os, json, time, csv, platform
from pathlib import Path
from typing import Any, Iterable, List, Optional
import pandas as pd

from dotenv import load_dotenv
from chromadb import PersistentClient

# RAGAS (métricas diretamente, sem evaluate())
from ragas.metrics import answer_relevancy, faithfulness
from ragas.llms import LangchainLLMWrapper

# Juiz LOCAL via Ollama (JSON-mode garantido)
from langchain_ollama import ChatOllama

# HF embeddings (para as métricas do RAGAS)
from langchain_huggingface import HuggingFaceEmbeddings

from src.graph import build_graph

# ------------------------------- ENV --------------------------------
# estabilidade básica
os.environ.setdefault("RAGAS_MAX_WORKERS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# diminui chance de crash no shutdown só no Windows
if platform.system() == "Windows":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

load_dotenv()

INDEX_DIR = os.getenv("INDEX_DIR", "data/index")
EVAL_SET = Path("eval/eval_set.jsonl")
REPORT_DIR = Path("eval/reports"); REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Modelo do juiz no Ollama (JSON mode)
# pode definir no .env: RAGAS_LLM=qwen2.5:7b-instruct (ou manter o padrão abaixo)
RAGAS_JUDGE  = os.getenv("RAGAS_LLM", "llama3.1:8b-instruct-q4_K_M")

# Embeddings para RAGAS
EMBED_MODEL  = os.getenv("RAGAS_EMB_MODEL", os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

# ------------------------------- ÍNDICE -----------------------------
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

# ------------------------------- EVAL SET --------------------------
if not EVAL_SET.exists():
    raise SystemExit('❌ Falta eval/eval_set.jsonl (campos: "question" e "gold_page").')

items = []
with EVAL_SET.open("r", encoding="utf-8-sig") as f:
    for raw in f:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        q = obj.get("question"); gp = obj.get("gold_page")
        if q is None or gp is None:
            continue
        items.append({"question": str(q), "gold_page": str(gp)})
if not items:
    raise SystemExit("❌ eval/eval_set.jsonl está vazio ou mal formatado.")

# ------------------------------- HELPERS ---------------------------
def ground_truth_for_page(page: str, max_snips: int = 3) -> list[str]:
    rows = kb_df[kb_df["page"] == str(page)]["text"].tolist()
    out = []
    for r in rows[:max_snips]:
        txt = r.replace("\n", " ").strip()
        if len(txt) > 800:
            txt = txt[:800] + "…"
        out.append(txt)
    return out or [f"[p.{page}] (nenhum chunk encontrado para esta página no índice)"]

# Adapter simples p/ embeddings no RAGAS
class RagasEmbeddingsAdapter:
    def __init__(self, lc_embeddings: Any):
        self._lc = lc_embeddings
    def embed_query(self, text: str) -> List[float]:
        return self._lc.embed_query(text)
    def embed_documents(self, docs: Iterable[str]) -> List[List[float]]:
        return self._lc.embed_documents(list(docs))

# Métricas de contexto determinísticas (diagnóstico do retriever)
def page_based_context_metrics(retrieved_pages: list[Optional[str]], gold_page: str):
    k = max(1, len(retrieved_pages))
    hits = sum(1 for p in retrieved_pages if p is not None and str(p) == str(gold_page))
    precision = hits / k
    recall = 1.0 if hits > 0 else 0.0
    return precision, recall

# Pequeno helper para logar exceptions do RAGAS ao invés de engolir
def _score_metric(metric, row, name: str) -> float:
    try:
        return float(metric.score(dict(row)))
    except Exception as e:
        print(f"[RAGAS {name} ERROR] {type(e).__name__}: {e}")
        return float("nan")

# ------------------------------- GRAFO ----------------------------
graph = build_graph()

rows = []
latencies = []

for it in items:
    q = it["question"]; gp = it["gold_page"]

    t0 = time.time()
    out = graph.invoke({"query": q, "contexts": [], "answer": {}})
    lat = time.time() - t0

    ans = out.get("answer", {}).get("answer", "")

    raw_ctxs = out.get("contexts") or out.get("answer", {}).get("contexts") or []
    ctx_texts, ctx_pages = [], []
    for c in raw_ctxs:
        if not isinstance(c, dict):
            continue
        txt = (c.get("text") or c.get("page_content") or "").replace("\n", " ").strip()
        pg  = c.get("page") or (c.get("metadata", {}) if isinstance(c.get("metadata", {}), dict) else {}).get("page")
        if txt:
            ctx_texts.append(txt)
            ctx_pages.append(str(pg) if pg is not None else None)

    refs = ground_truth_for_page(gp)
    rows.append({
        "question": q,
        "gold_page": str(gp),
        "response": ans,
        "retrieved_contexts": ctx_texts,  # algumas versões exigem esse nome
        "retrieved_pages": ctx_pages,
        "reference_contexts": refs,
        "reference": " ".join(refs),
    })
    latencies.append(lat)

print("Primeiro exemplo:", {
    "user_input": rows[0]["question"],
    "response_head": rows[0]["response"][:160],
    "contexts_count": len(rows[0]["retrieved_contexts"]),
    "reference_count": len(rows[0]["reference_contexts"]),
})

# --------------------------- LLM & EMBEDDINGS ----------------------
# Juiz do RAGAS (LOCAL) via Ollama, forçando JSON mode e sessão quente para 20+
evaluator_llm = ChatOllama(
    model=RAGAS_JUDGE,
    temperature=0,
    num_ctx=4096,
    num_predict=1024,
    format="json",   # <- força saída JSON, ajuda o parser do RAGAS
    mirostat=0,
    keep_alive="30m",  # <- mantém o modelo vivo enquanto roda 20 perguntas
)
lc2ragas_llm = LangchainLLMWrapper(evaluator_llm)

hf_emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
emb_adapter = RagasEmbeddingsAdapter(hf_emb)  # type: ignore

# Configura métricas RAGAS
metric_rel = answer_relevancy
metric_rel.llm = lc2ragas_llm
metric_rel.embeddings = emb_adapter  # type: ignore

metric_faith = faithfulness
metric_faith.llm = lc2ragas_llm
metric_faith.embeddings = emb_adapter  # type: ignore

# ---- Stub mínimo para evitar IndexError interno no answer_relevancy ----
# (Garante que o gerador de perguntas do RAGAS devolva >=1 item)
from types import SimpleNamespace
class _DummyQGen:
    def __init__(self, n_inner: int = 4): self.n_inner = n_inner
    async def generate_multiple(self, *args, **kwargs):
        return [SimpleNamespace(question="paráfrase", noncommittal=0)]
try:
    _orig_qgen = getattr(metric_rel, "question_generation", None)
    metric_rel.question_generation = _DummyQGen(n_inner=4)  # type: ignore
except Exception:
    _orig_qgen = None
# ------------------------------------------------------------------------

# ----------------------------- AVALIAÇÃO -------------------------
per_q = []
sum_rel = 0.0; n_rel = 0
sum_fa  = 0.0; n_fa  = 0
sum_cprec = 0.0
sum_crec  = 0.0

for idx, r in enumerate(rows):
    # IMPORTANTE: fornecer os nomes de chaves esperados por diferentes versões do RAGAS
    row_common = {
        "user_input": r["question"],
        "response": r["response"],
        "contexts": r["retrieved_contexts"],            # algumas versões usam "contexts"
        "retrieved_contexts": r["retrieved_contexts"],  # outras usam "retrieved_contexts"
        "reference_contexts": r["reference_contexts"],
        "reference": r["reference"],
        "id": r.get("id", f"eval-{idx}"),
        "metadata": {},
    }

    # RAGAS: answer_relevancy
    rel_val = _score_metric(metric_rel, row_common, "answer_relevancy")

    # RAGAS: faithfulness
    fa_val  = _score_metric(metric_faith, row_common, "faithfulness")

    # métricas de contexto (diagnóstico do retriever)
    cprec, crec = page_based_context_metrics(r["retrieved_pages"], r["gold_page"])

    per_q.append({
        "question": r["question"],
        "contexts_count": len(r["retrieved_contexts"]),
        "latency_s": float(latencies[idx]),
        "answer_relevancy": rel_val,
        "faithfulness": fa_val,
        "context_precision": cprec,
        "context_recall": crec,
    })

    if rel_val == rel_val: sum_rel += rel_val; n_rel += 1
    if fa_val  == fa_val:  sum_fa  += fa_val;  n_fa  += 1
    sum_cprec += cprec
    sum_crec  += crec

details_df = pd.DataFrame(per_q)
details_path = REPORT_DIR / "ragas_details.csv"
details_df.to_csv(details_path, index=False, encoding="utf-8")

# Médias
avg_rel   = (sum_rel / n_rel) if n_rel > 0 else float("nan")
avg_fa    = (sum_fa  / n_fa)  if n_fa  > 0 else float("nan")
avg_cprec = sum_cprec / max(1, len(rows))
avg_crec  = sum_crec  / max(1, len(rows))
avg_lat   = sum([float(x) for x in details_df["latency_s"]]) / max(1, len(rows))

with REPORT_DIR.joinpath("ragas_scores.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["metric", "value"])
    w.writerow(["answer_relevancy", f"{avg_rel:.4f}" if avg_rel == avg_rel else "nan"])
    w.writerow(["faithfulness",     f"{avg_fa:.4f}"  if avg_fa  == avg_fa  else "nan"])
    w.writerow(["context_precision", f"{avg_cprec:.4f}"])
    w.writerow(["context_recall",    f"{avg_crec:.4f}"])
    w.writerow(["latency_avg_s",     f"{avg_lat:.4f}"])

print("✅ Avaliação finalizada!")
print(f"   → Médias: {REPORT_DIR/'ragas_scores.csv'}")
print(f"   → Detalhes: {REPORT_DIR/'ragas_details.csv'}")

# Evita crash no shutdown no Windows (processo encerra limpo)
import os as _os
_os._exit(0)
