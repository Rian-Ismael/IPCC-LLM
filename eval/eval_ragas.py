import os, sys, time, json, threading, re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, cast

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import psutil
import pandas as pd
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph import build_graph, State

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI


_env_eval = os.getenv("EVAL_PATH")
if _env_eval:
    EVAL_PATH = _env_eval
else:
    gt = Path("eval/eval_set.gt.jsonl")
    plain = Path("eval/eval_set.jsonl")
    EVAL_PATH = str(gt if gt.exists() else plain)

REPORTS_DIR = Path(os.getenv("EVAL_REPORTS_DIR", "eval/reports"))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-2.5-pro")
USE_GEMINI_JUDGE = os.getenv("USE_GEMINI_JUDGE", "1").strip()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", "qwen2.5:7b-instruct")

def make_judge():
    """
    Se USE_GEMINI_JUDGE == "1" e GOOGLE_API_KEY existir, usa Gemini 2.5 Pro (remote, alivia CPU/RAM).
    Caso contrário, usa Ollama local (fallback).
    """
    use_gemini = (USE_GEMINI_JUDGE == "1") and bool(GOOGLE_API_KEY)
    if use_gemini:
        model_kwargs = {
            "response_mime_type": "application/json",
            "max_output_tokens": 2048,
        }
        judge = ChatGoogleGenerativeAI(
            model=GEMINI_JUDGE_MODEL,
            temperature=0.0,
            convert_system_message_to_human=True,
            model_kwargs=model_kwargs,
        )
        print(f"[judge] Gemini ativo: model={GEMINI_JUDGE_MODEL}")
        return judge
    else:
        judge = ChatOllama(
            model=OLLAMA_JUDGE_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.0,
            num_ctx=1024,
            num_thread=2,
        )
        print(f"[judge] Ollama ativo: model={OLLAMA_JUDGE_MODEL} @ {OLLAMA_BASE_URL}")
        return judge

def make_embeddings():
    model_name = (
        os.getenv("RAGAS_EMBEDDINGS_MODEL")
        or os.getenv("EMBEDDINGS_MODEL")
        or "sentence-transformers/all-MiniLM-L6-v2"
    )
    return HuggingFaceEmbeddings(model_name=model_name)

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval set não encontrado: {path}")
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            q = obj.get("question") or obj.get("pergunta") or obj.get("query")
            if not q:
                raise ValueError(f"Linha sem 'question': {obj}")
            rows.append({
                "question": q,
                "gold_page": obj.get("gold_page"),
                "ground_truth": obj.get("ground_truth") or obj.get("expected_answer")
            })
    return rows

def extract_answer_and_contexts(state_out: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Compatível com sua pipeline:
      - out['answer'] é um dict com chave 'answer' (string) e, às vezes, 'contexts'
      - out['contexts'] (top-level) vem do retriever (lista de dicts)
    """
    ans_dict = cast(Dict[str, Any], state_out.get("answer") or {})
    answer_txt = (ans_dict.get("answer") or "").strip()

    raw_ctxs = state_out.get("contexts") or ans_dict.get("contexts") or []
    ctx_texts: List[str] = []
    for c in raw_ctxs:
        if isinstance(c, str):
            txt = c
        elif isinstance(c, dict):
            txt = c.get("text") or c.get("page_content") or c.get("content") or ""
        else:
            txt = ""
        if isinstance(txt, str) and txt.strip():
            ctx_texts.append(txt)
    if not ctx_texts:
        ctx_texts = [""]
    return answer_txt, ctx_texts

class FootprintSampler:
    def __init__(self, period_sec: float = 0.5):
        self.proc = psutil.Process(os.getpid())
        self.period = period_sec
        self._stop = threading.Event()
        self.samples_cpu = []
        self.peak_rss = 0

        try:
            self.proc.cpu_percent(None)
        except Exception:
            pass

    def _tick(self):
        while not self._stop.is_set():
            try:
                cpu = self.proc.cpu_percent(None)
                rss = self.proc.memory_info().rss
                self.samples_cpu.append(cpu)
                if rss > self.peak_rss:
                    self.peak_rss = rss
            except Exception:
                pass
            time.sleep(self.period)

    def start(self):
        self._stop.clear()
        self._th = threading.Thread(target=self._tick, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        if hasattr(self, "_th"):
            self._th.join(timeout=2.0)

    @property
    def cpu_avg(self) -> Optional[float]:
        return float(sum(self.samples_cpu) / len(self.samples_cpu)) if self.samples_cpu else None

    @property
    def mem_peak_mb(self) -> Optional[float]:
        return round(self.peak_rss / (1024 * 1024), 1) if self.peak_rss else None

def pages_in_texts(texts: Any) -> set:
    """
    Extrai números de página a partir de tags como [p.X] presentes nos textos dos contextos.
    Se seus contextos tiverem metadados de página, adapte aqui para ler do dict.
    """
    pages = set()
    if isinstance(texts, list):
        for t in texts:
            if not isinstance(t, str):
                continue
            for m in re.finditer(r"\[p\.(\d+)\]", t):
                try:
                    pages.add(int(m.group(1)))
                except Exception:
                    pass
    return pages

def gold_hit_row(row: pd.Series) -> Optional[bool]:
    gold = row.get("gold_page")
    if pd.isna(gold) or gold is None:
        return None
    try:
        gold = int(gold)
    except Exception:
        return None
    ctx_pages = pages_in_texts(row.get("contexts"))
    return gold in ctx_pages if ctx_pages else False

def run_eval(eval_path: str = EVAL_PATH) -> None:
    print(f"[eval] Usando arquivo: {eval_path}")

    items = load_jsonl(eval_path)

    app = build_graph()

    sampler = FootprintSampler(period_sec=0.5)
    sampler.start()

    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(items, start=1):
        q: str = item["question"]
        gt: Optional[str] = item.get("ground_truth") or ""
        t0 = time.time()

        empty_contexts: List[Dict[str, Any]] = []
        empty_answer: Dict[str, Any] = {}
        init_state: State = {"query": q, "contexts": empty_contexts, "answer": empty_answer}

        out_state: Dict[str, Any] = app.invoke(init_state)
        elapsed_ms = int((time.time() - t0) * 1000)

        answer_txt, ctx_texts = extract_answer_and_contexts(out_state)

        rows.append({
            "question": q,
            "answer": answer_txt,
            "contexts": ctx_texts,
            "ground_truth": gt,
            "gold_page": item.get("gold_page"),
            "latency_ms": elapsed_ms,
        })
        print(f"[{i:02d}/{len(items)}] {elapsed_ms} ms | '{q[:70]}...'")

    sampler.stop()

    df = pd.DataFrame(rows)
    ds_cols = ["question", "answer", "contexts"]
    if "ground_truth" in df.columns and df["ground_truth"].notna().any():
        ds_cols.append("ground_truth")
    ds = Dataset.from_pandas(df[ds_cols])

    judge = make_judge()
    embedder = make_embeddings()

    ragas_result = ragas_evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
        embeddings=embedder,
        run_config=RunConfig(timeout=600, max_workers=1),
    )

    df_scores = ragas_result.to_pandas()
    merge_key = "user_input" if "user_input" in df_scores.columns else "question"
    df_merged = df.merge(df_scores, left_on="question", right_on=merge_key, how="left")

    df_merged["gold_hit"] = df_merged.apply(gold_hit_row, axis=1)
    gold_rate = float(df_merged["gold_hit"].mean(skipna=True)) if "gold_hit" in df_merged else None

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "ragas_scores.csv").write_text(df_scores.to_csv(index=False), encoding="utf-8")
    (REPORTS_DIR / "raw_results.csv").write_text(df_merged.to_csv(index=False), encoding="utf-8")

    lat = df_merged["latency_ms"].astype(float)
    p50 = int(lat.quantile(0.50)) if len(lat) else 0
    p95 = int(lat.quantile(0.95)) if len(lat) else 0

    def safe_mean(col: str) -> Optional[float]:
        return float(df_merged[col].mean()) if (col in df_merged and df_merged[col].notna().any()) else None

    metrics_summary = {
        "faithfulness_mean": safe_mean("faithfulness"),
        "answer_relevancy_mean": safe_mean("answer_relevancy"),
        "context_precision_mean": safe_mean("context_precision"),
        "context_recall_mean": safe_mean("context_recall"),
    }

    summary = {
        "samples": int(len(df_merged)),
        "metrics": {k: (None if v is None else round(v, 4)) for k, v in metrics_summary.items()},
        "latency_ms": {
            "mean": int(lat.mean()) if len(lat) else None,
            "p50": p50,
            "p95": p95,
            "min": int(lat.min()) if len(lat) else None,
            "max": int(lat.max()) if len(lat) else None,
        },
        "footprint": {
            "memory_peak_mb": sampler.mem_peak_mb,
            "cpu_percent_avg": None if sampler.cpu_avg is None else round(sampler.cpu_avg, 1),
        },
        "audits": {
            "gold_hit_rate": None if gold_rate is None else round(gold_rate, 4)
        },
        "paths": {
            "scores_csv": str(REPORTS_DIR / "ragas_scores.csv"),
            "raw_csv": str(REPORTS_DIR / "raw_results.csv"),
        },
    }

    (REPORTS_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (REPORTS_DIR / "report.md").open("w", encoding="utf-8") as f:
        f.write("# Resultado de Avaliação (RAGAS)\n\n")
        f.write(f"- **Amostras**: {summary['samples']}\n")
        m = summary["metrics"]
        f.write(f"- **Faithfulness (média)**: {m['faithfulness_mean']}\n")
        f.write(f"- **Answer Relevancy (média)**: {m['answer_relevancy_mean']}\n")
        f.write(f"- **Context Precision (média)**: {m['context_precision_mean']}\n")
        f.write(f"- **Context Recall (média)**: {m['context_recall_mean']}\n\n")

        f.write("## Latência (ms)\n")
        lm = summary["latency_ms"]
        f.write(f"- média: {lm['mean']} | min: {lm['min']} | max: {lm['max']}\n")
        f.write(f"- p50: {lm['p50']} | p95: {lm['p95']}\n\n")

        f.write("## Footprint (média aproximada do processo)\n")
        fp = summary["footprint"]
        f.write(f"- **Pico de memória**: {fp['memory_peak_mb']} MB\n")
        f.write(f"- **CPU média do processo**: {fp['cpu_percent_avg']}%\n\n")

        f.write("## Auditoria do Retriever (gold_page)\n")
        gh = summary["audits"]["gold_hit_rate"]
        if gh is None:
            f.write("- **gold_hit_rate**: n/d (sem gold_page suficiente)\n\n")
        else:
            f.write(f"- **gold_hit_rate**: {gh:.2%}\n\n")

    print("[DONE] Avaliação salva em:")
    print(f"- {REPORTS_DIR / 'report.md'}")
    print(f"- {REPORTS_DIR / 'raw_results.csv'}")
    print(f"- {REPORTS_DIR / 'ragas_scores.csv'}")
    print(f"- {REPORTS_DIR / 'summary.json'}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-path", default=EVAL_PATH, help="Caminho para o JSONL (padrão autodetect)")
    args = ap.parse_args()
    run_eval(args.eval_path)
