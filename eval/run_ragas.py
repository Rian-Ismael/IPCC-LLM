import os, json, time, csv, platform, sys, asyncio, inspect, re, math
from pathlib import Path
from typing import Any, Iterable, List, Optional
import pandas as pd

try:
    import psutil 
except Exception:
    psutil = None

from dotenv import load_dotenv
from chromadb import PersistentClient

FALLBACK = "I have not found sufficient evidence in the IPCC to answer with confidence."

try:
    from importlib.metadata import version as _pkgver
    RAGAS_VER = _pkgver("ragas")
except Exception:
    RAGAS_VER = "unknown"

from ragas.metrics import answer_relevancy, faithfulness
from ragas.llms import LangchainLLMWrapper
try:
    from ragas.embeddings import RagasEmbeddingsAdapter as RagasEmbAdapter
except Exception:
    RagasEmbAdapter = None

SingleTurnSample = None
for mod, name in [
    ("ragas.types", "SingleTurnSample"),             # 0.3.x
    ("ragas.dataset_schema", "SingleTurnSample"),    # 0.2.x
]:
    try:
        SingleTurnSample = __import__(mod, fromlist=[name]).__dict__[name]
        break
    except Exception:
        pass

from langchain_huggingface import HuggingFaceEmbeddings
from src.graph import build_graph

load_dotenv(override=True)
os.environ.setdefault("RAGAS_MAX_WORKERS", os.getenv("RAGAS_MAX_WORKERS", "1"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if platform.system() == "Windows":
    for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

INDEX_DIR   = os.getenv("INDEX_DIR", "data/index")
EVAL_SET    = Path("eval/eval_set.jsonl")
REPORT_DIR  = Path("eval/reports"); REPORT_DIR.mkdir(parents=True, exist_ok=True)

RAGAS_JUDGE = os.getenv("RAGAS_LLM", os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"))
JUDGE_NUM_CTX     = int(os.getenv("RAGAS_JUDGE_NUM_CTX", "1536"))
JUDGE_NUM_PREDICT = int(os.getenv("RAGAS_JUDGE_NUM_PREDICT", "512"))

_MET = [m.strip().lower() for m in os.getenv("RAGAS_METRICS", "rel,faith").split(",") if m.strip()]
USE_REL   = "rel" in _MET
USE_FAITH = "faith" in _MET

EMBED_MODEL = os.getenv("RAGAS_EMB_MODEL", os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

db = PersistentClient(path=INDEX_DIR)
try:
    coll = db.get_collection("ipcc")
except Exception:
    raise SystemExit(
        f"Collection 'ipcc' not found in {INDEX_DIR}.\n"
        "Run ingestion:\n"
        "  python -m ingest.build_index --pdf data\\corpus\\IPCC_AR6_SYR_LongerReport.pdf --index-dir data\\index"
    )

probe = coll.get(limit=1, include=["documents"])
if not probe.get("documents"):
    raise SystemExit(
        f"Collection 'ipcc' is empty in {INDEX_DIR}.\n"
        "Rebuild the index:\n"
        "  Remove-Item -Recurse -Force data\\index\n"
        "  python -m ingest.build_index --pdf data\\corpus\\IPCC_AR6_SYR_LongerReport.pdf --index-dir data\\index"
    )

dump = coll.get(include=["documents", "metadatas"])
docs, metas = dump["documents"], dump["metadatas"]
kb_df = pd.DataFrame([{"text": t, "page": str(m.get("page","?"))} for t,m in zip(docs, metas)])


if not EVAL_SET.exists():
    raise SystemExit('Missing eval/eval_set.jsonl (fields: "question" and "gold_page").')

items = []
with EVAL_SET.open("r", encoding="utf-8-sig") as f:
    for raw in f:
        line = raw.strip()
        if not line or line.startswith("#"): continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        q = obj.get("question"); gp = obj.get("gold_page")
        if q is None or gp is None: continue
        items.append({"question": str(q), "gold_page": str(gp)})
if not items:
    raise SystemExit("eval/eval_set.jsonl is empty or malformed.")

_NUM_RE = re.compile(r'[-+]?\d+(?:\.\d+)?')

def _extract_numbers(text: str) -> list[str]:
    return _NUM_RE.findall(text or "")

def _strip_citations(text: str) -> str:
    return re.sub(r'\[p\.\d+\]', '', text or '').strip()

def _cosine(u: List[float], v: List[float]) -> float:
    dot = sum((a*b) for a,b in zip(u, v))
    nu = math.sqrt(sum((a*a) for a in u))
    nv = math.sqrt(sum((b*b) for b in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (nu * nv)

def ground_truth_for_page(page: str, max_snips: int = 3) -> list[str]:
    rows = kb_df[kb_df["page"] == str(page)]["text"].tolist()
    out = []
    for r in rows[:max_snips]:
        txt = (r or "").replace("\n", " ").strip()
        if len(txt) > 800: txt = txt[:800] + "…"
        if txt: out.append(txt)
    return out or [f"[p.{page}] (no chunk found for this page in the index)"]

class _LocalRagasEmbAdapter:
    def __init__(self, lc_embeddings: Any): self._lc = lc_embeddings
    def embed_query(self, text: str) -> List[float]: return self._lc.embed_query(text)
    def embed_documents(self, docs: Iterable[str]) -> List[List[float]]: return self._lc.embed_documents(list(docs))

def page_based_context_metrics(retrieved_pages: list[Optional[str]], gold_page: str):
    k = max(1, len(retrieved_pages))
    hits = sum(1 for p in retrieved_pages if p is not None and str(p) == str(gold_page))
    precision = hits / k
    recall = 1.0 if hits > 0 else 0.0
    return precision, recall

def clamp01(x: float) -> float:
    try: x = float(x)
    except Exception: return float("nan")
    if x != x: return x
    return 0.0 if x < 0 else (1.0 if x > 1 else x)

def faithfulness_proxy(answer: str, ctx_texts: list[str], emb: HuggingFaceEmbeddings) -> float:
    """
    Proxy conservador: fração de frases da resposta que têm suporte
    nos contextos por similaridade >= 0.52 E números preservados.
    """
    if not answer or not ctx_texts:
        return float("nan")

    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', _strip_citations(answer)) if s.strip()]
    if not sents:
        return float("nan")

    try:
        ctx_embs: List[List[float]] = emb.embed_documents(ctx_texts)
    except Exception:
        return float("nan")

    ctx_blob = " ".join(ctx_texts)
    supported = 0

    for s in sents:
        try:
            s_emb: List[float] = emb.embed_query(s)
        except Exception:
            continue
        max_sim = max((_cosine(s_emb, ce) for ce in ctx_embs), default=0.0)

        nums = _extract_numbers(s)
        nums_ok = all(n in ctx_blob for n in nums) if nums else True

        if max_sim >= 0.52 and nums_ok:
            supported += 1

    return supported / max(1, len(sents))

def _await_if_needed(v):
    if inspect.isawaitable(v):
        try:
            return asyncio.run(v)
        except RuntimeError:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                return loop.run_until_complete(v)
            raise
    return v

def _build_sample(row: dict):
    """Constrói SingleTurnSample quando disponível (ragas 0.3.x)."""
    if SingleTurnSample is None:
        return row
    try:
        return SingleTurnSample(**row)
    except Exception:
        return row

from langchain_ollama import ChatOllama
evaluator_llm = ChatOllama(
    model=RAGAS_JUDGE,
    temperature=0,
    num_ctx=JUDGE_NUM_CTX,
    num_predict=JUDGE_NUM_PREDICT,
    mirostat=0,
    keep_alive="30m",
)
print(f"[RAGAS] versão {RAGAS_VER} | Judge = Ollama:{RAGAS_JUDGE} | metrics={','.join([m for m in ['rel' if USE_REL else '', 'faith' if USE_FAITH else ''] if m])}")

lc2ragas_llm = LangchainLLMWrapper(evaluator_llm)
hf_emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
emb_adapter = (RagasEmbAdapter(hf_emb) if RagasEmbAdapter else _LocalRagasEmbAdapter(hf_emb)) 

metric_rel = answer_relevancy;  metric_rel.llm = lc2ragas_llm;  metric_rel.embeddings = emb_adapter 
metric_faith = faithfulness;    metric_faith.llm = lc2ragas_llm; metric_faith.embeddings = emb_adapter 


graph = build_graph()

proc = psutil.Process(os.getpid()) if psutil else None
peak_mem_bytes = 0
cpu_samples = []

rows = []; latencies = []
for it in items:
    q  = it["question"]; gp = it["gold_page"]
    t0 = time.time()
    out = graph.invoke({"query": q, "contexts": [], "answer": {}})
    lat = time.time() - t0

    ans = out.get("answer", {}).get("answer", "")
    raw_ctxs = out.get("contexts") or out.get("answer", {}).get("contexts") or []
    ctx_texts, ctx_pages = [], []
    for c in raw_ctxs:
        if not isinstance(c, dict): continue
        txt = (c.get("text") or c.get("page_content") or "").replace("\n", " ").strip()
        pg  = c.get("page") or (c.get("metadata", {}) if isinstance(c.get("metadata", {}), dict) else {}).get("page")
        if txt:
            ctx_texts.append(txt)
            ctx_pages.append(str(pg) if pg is not None else None)

    refs = ground_truth_for_page(gp)
    is_abstain = isinstance(ans, str) and FALLBACK.lower() in ans.lower()
    rows.append({
        "user_input": q,
        "response": ans,
        "retrieved_contexts": ctx_texts,
        "contexts": ctx_texts,
        "reference_contexts": refs,
        "reference": " ".join(refs),
        "retrieved_pages": ctx_pages,
        "gold_page": str(gp),
        "abstained": is_abstain,
        "id": f"eval-{len(rows)}",
        "metadata": {},
    })
    latencies.append(lat)

    if proc:
        try:
            mem_now = proc.memory_info().rss
            if mem_now > peak_mem_bytes:
                peak_mem_bytes = mem_now
            cpu_samples.append(psutil.cpu_percent(interval=0.0))
        except Exception:
            pass

print("Primeiro exemplo:", {
    "user_input": rows[0]["user_input"],
    "response_head": rows[0]["response"][:160],
    "contexts_count": len(rows[0]["retrieved_contexts"]),
    "reference_count": len(rows[0]["reference_contexts"]),
})


per_q = []
sum_rel = sum_cprec = sum_crec = 0.0
n_rel = 0
sum_fa_answered = 0.0
n_fa_answered = 0

def _safe_ascore(metric, row, name: str) -> float:
    try:
        if hasattr(metric, "single_turn_ascore"):
            sample = _build_sample(row)
            res = metric.single_turn_ascore(sample)
        else:
            res = metric.score(row)  # compat 0.2.x
        res = _await_if_needed(res)
        return float(res)
    except Exception as e:
        print(f"[RAGAS {name} ERROR] {type(e).__name__}: {e}")
        return float("nan")

for r, lat in zip(rows, latencies):
    row_common = {
        "user_input": r["user_input"],
        "response": r["response"],
        "contexts": r["contexts"],
        "retrieved_contexts": r["retrieved_contexts"],
        "reference_contexts": r["reference_contexts"],
        "reference": r["reference"],
        "id": r["id"],
        "metadata": r["metadata"],
    }

    rel_val = float("nan")
    if USE_REL:
        rel_raw = clamp01(_safe_ascore(metric_rel, row_common, "answer_relevancy"))
        if rel_raw == rel_raw:
            rel_val = rel_raw
            sum_rel += rel_val
            n_rel += 1

    fa_val = float("nan")
    fa_src = ""
    if USE_FAITH and not r.get("abstained"):
        fa_raw = clamp01(_safe_ascore(metric_faith, row_common, "faithfulness"))
        if fa_raw == fa_raw:
            fa_val = fa_raw
            fa_src = "ragas"
            sum_fa_answered += fa_val
            n_fa_answered += 1
        else:
            try:
                fa_proxy = clamp01(faithfulness_proxy(
                    answer=r["response"],
                    ctx_texts=r["retrieved_contexts"],
                    emb=hf_emb,
                ))
                if fa_proxy == fa_proxy:
                    fa_val = fa_proxy
                    fa_src = "proxy"
                    sum_fa_answered += fa_val
                    n_fa_answered += 1
            except Exception:
                fa_src = "error"

    cprec, crec = page_based_context_metrics(r["retrieved_pages"], r["gold_page"])

    per_q.append({
        "question": r["user_input"],
        "contexts_count": len(r["retrieved_contexts"]),
        "latency_s": float(lat),
        "answer_relevancy": rel_val,
        "faithfulness": fa_val,
        "faithfulness_source": fa_src,
        "context_precision": cprec,
        "context_recall": crec,
    })

details_df = pd.DataFrame(per_q)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
details_df.to_csv(REPORT_DIR / "ragas_details.csv", index=False, encoding="utf-8")

avg_rel   = (sum_rel / n_rel) if n_rel > 0 else float("nan")
avg_cprec = sum([x["context_precision"] for x in per_q]) / max(1, len(per_q))
avg_crec  = sum([x["context_recall"]    for x in per_q]) / max(1, len(per_q))
avg_lat   = sum([float(x["latency_s"])  for x in per_q]) / max(1, len(per_q))
faith_answered = (sum_fa_answered / n_fa_answered) if n_fa_answered > 0 else float("nan")

if psutil and peak_mem_bytes > 0:
    peak_mem_mb = peak_mem_bytes / (1024 * 1024)
    cpu_avg = (sum(cpu_samples) / max(1, len(cpu_samples))) if cpu_samples else 0.0
else:
    peak_mem_mb = float("nan")
    cpu_avg = float("nan")

with (REPORT_DIR / "ragas_scores.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["metric", "value"])
    w.writerow(["answer_relevancy", f"{avg_rel:.4f}" if avg_rel == avg_rel else "nan"])
    w.writerow(["faithfulness",     f"{faith_answered:.4f}" if faith_answered == faith_answered else "nan"])
    w.writerow(["context_precision", f"{avg_cprec:.4f}"])
    w.writerow(["context_recall",    f"{avg_crec:.4f}"])
    w.writerow(["latency_avg_s",     f"{avg_lat:.4f}"])
    w.writerow(["memory_peak_mb",    f"{peak_mem_mb:.1f}" if peak_mem_mb == peak_mem_mb else "nan"])
    w.writerow(["cpu_percent_avg",   f"{cpu_avg:.1f}" if cpu_avg == cpu_avg else "nan"])

print("Avaliação finalizada!")
print(f"   → Médias: {REPORT_DIR/'ragas_scores.csv'}")
print(f"   → Detalhes: {REPORT_DIR/'ragas_details.csv'}")

sys.exit(0)