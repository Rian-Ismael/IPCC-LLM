from typing import List, Dict, Any
import os, math

from src.utils.settings import COLL, EMB

try:
    from src.utils.pdf_loader import normalize_text
except Exception:
    def normalize_text(s: str) -> str:
        return (s or "").strip()

K = int(os.getenv("TOP_K", "6"))

MIN_SIM = float(os.getenv("RETRIEVER_MIN_SIM", "0.05"))

UNIQ_BY_PAGE = os.getenv("RETRIEVER_UNIQUE_PAGES", "1") == "1"

RERANK_ENABLE = os.getenv("RERANK_ENABLE", "1") == "1"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", str(max(K * 3, 12))))
RERANK_ALPHA = float(os.getenv("RERANK_ALPHA", "0.7"))

_RERANKER = None 


def _get_reranker():
    """Carrega o CrossEncoder sob demanda. Fallback silencioso se não der."""
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER
    if not RERANK_ENABLE:
        return None
    try:
        from sentence_transformers import CrossEncoder
        _RERANKER = CrossEncoder(RERANK_MODEL)
        return _RERANKER
    except Exception as e:
        print(f"[retriever] Rerank desabilitado: {e}")
        _RERANKER = None
        return None


def _cosine_sim_from_distance(d) -> float:
    try:
        return max(0.0, 1.0 - float(d))
    except Exception:
        return 0.0


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _apply_rerank(query_text: str, cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reranqueia top-N com CrossEncoder e mistura com score vetorial."""
    reranker = _get_reranker()
    if reranker is None or not cands:
        return cands 

    pool = sorted(cands, key=lambda x: x.get("vector_score", 0.0), reverse=True)[:RERANK_TOP_K]
    pairs = [(query_text, d["text"]) for d in pool]

    try:
        scores = reranker.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
    except Exception as e:
        print(f"[retriever] Falha no rerank: {e}")
        return cands

    out = []
    for item, ce_raw in zip(pool, scores):
        ce_norm = _sigmoid(float(ce_raw))             
        vec = float(item.get("vector_score", 0.0))
        final = RERANK_ALPHA * ce_norm + (1.0 - RERANK_ALPHA) * vec

        new_item = dict(item)
        new_item["rerank_score_raw"] = float(ce_raw)
        new_item["rerank_score"] = ce_norm
        new_item["score"] = final
        out.append(new_item)

    out.sort(key=lambda x: x["score"], reverse=True)

    used = {i["id"] for i in out}
    rest = [d for d in cands if d["id"] not in used]
    out.extend(rest)
    return out


def retrieve(query: str) -> List[Dict[str, Any]]:
    q_norm = normalize_text(query)
    qv = EMB.encode([q_norm], convert_to_numpy=True).tolist()[0]

    n = max(K * 3, K)
    res = COLL.query(
        query_embeddings=[qv],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    if not res.get("documents"):
        return []

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res.get("distances", [[None] * len(docs)])[0]

    raw_ids = res.get("ids")
    ids = raw_ids[0] if raw_ids and len(raw_ids) > 0 else [f"idx-{i}" for i in range(len(docs))]

    prelim: List[Dict[str, Any]] = []
    for id_, doc, meta, dist in zip(ids, docs, metas, dists):
        vec_sim = _cosine_sim_from_distance(dist)
        if MIN_SIM > 0 and vec_sim < MIN_SIM:
            continue
        prelim.append({
            "id": id_,
            "text": doc,
            "metadata": meta,
            "page": (meta or {}).get("page"),
            "vector_score": vec_sim,
            "score": vec_sim,
        })

    if not prelim:
        for id_, doc, meta, dist in zip(ids, docs, metas, dists):
            vec_sim = _cosine_sim_from_distance(dist)
            prelim.append({
                "id": id_,
                "text": doc,
                "metadata": meta,
                "page": (meta or {}).get("page"),
                "vector_score": vec_sim,
                "score": vec_sim,
            })

    ranked = _apply_rerank(q_norm, prelim)

    out: List[Dict[str, Any]] = []
    seen_pages = set()
    for h in ranked:
        if UNIQ_BY_PAGE:
            pg = h.get("page")
            if pg in seen_pages:
                continue
            seen_pages.add(pg)
        out.append(h)
        if len(out) >= K:
            break

    if len(out) < K:
        for h in ranked:
            if h in out:
                continue
            if UNIQ_BY_PAGE and h.get("page") in seen_pages:
                continue
            out.append(h)
            if len(out) >= K:
                break

    return out
