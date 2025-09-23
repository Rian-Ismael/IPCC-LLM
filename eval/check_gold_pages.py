import os, json, re, argparse
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from dotenv import load_dotenv
from chromadb import PersistentClient

load_dotenv()

INDEX_DIR = os.getenv("INDEX_DIR", "data/index")
EVAL_SET = Path("eval/eval_set.jsonl")

def load_index_df(index_dir: str) -> pd.DataFrame:
    db = PersistentClient(path=index_dir)
    try:
        coll = db.get_collection("ipcc")
    except Exception:
        raise SystemExit(f"Collection 'ipcc' not found in {index_dir}. Rode a ingestão primeiro.")
    dump = coll.get(include=["documents", "metadatas"])
    docs, metas = dump["documents"], dump["metadatas"]
    if not docs:
        raise SystemExit(f"Índice vazio em {index_dir}. Rode a ingestão.")
    df = pd.DataFrame([{"text": t or "", "page": str(m.get("page","?"))} for t,m in zip(docs, metas)])
    return df

def load_eval_items(eval_path: Path) -> List[dict]:
    if not eval_path.exists():
        raise SystemExit("Falta eval/eval_set.jsonl")
    items, bad = [], []
    with eval_path.open("r", encoding="utf-8-sig") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"): continue
            try:
                obj = json.loads(line)
                q = str(obj.get("question", "")).strip()
                gp = str(obj.get("gold_page", "")).strip()
                if q and gp: items.append({"question": q, "gold_page": gp})
                else: bad.append((lineno, "missing question/gold_page"))
            except json.JSONDecodeError as e:
                bad.append((lineno, f"JSONDecodeError: {e}"))
    if not items:
        try:
            data = json.loads(eval_path.read_text(encoding="utf-8-sig"))
            if isinstance(data, list):
                for idx, obj in enumerate(data, start=1):
                    q = str(obj.get("question", "")).strip()
                    gp = str(obj.get("gold_page", "")).strip()
                    if q and gp: items.append({"question": q, "gold_page": gp})
                    else: bad.append((idx, "missing question/gold_page (array mode)"))
        except Exception:
            pass
    if bad:
        print("\n[WARN] Algumas linhas foram ignoradas:")
        for ln, msg in bad: print(f"  linha {ln}: {msg}")
    if not items:
        raise SystemExit("eval_set.jsonl inválido (nenhuma entrada válida)")
    return items

def find_pages(df: pd.DataFrame, substr: str, top=8) -> List[str]:
    if not substr: return []
    pat = re.escape(substr)
    m = df[df["text"].str.contains(pat, case=False, na=False)][["page"]]
    seen, uniq = set(), []
    for p in m["page"].tolist():
        if p not in seen:
            seen.add(p); uniq.append(p)
            if len(uniq) >= top: break
    return uniq

def sample_texts(df: pd.DataFrame, page: str, n=2) -> List[str]:
    rows = df[df["page"] == str(page)]["text"].head(n).tolist()
    out = []
    for r in rows:
        r = (r or "").replace("\n"," ").strip()
        out.append(r[:180] + ("..." if len(r) > 180 else ""))
    return out

_SSP_PAT = re.compile(r"SSP\s*([1-5])\s*[-–—]?\s*(\d(?:\.\d)?)", re.IGNORECASE)
_DEGC_PAT = re.compile(r"(\d(?:\.\d)?)\s*°?\s*C")
_YEAR_PAT = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2}|2100)\b")
_PHRASES = {
    "drivers": ["well-mixed GHG", "greenhouse gas", "unequivocally caused"],
    "losses": ["losses and damages"],
    "sea level": ["global mean sea level", "sea level rise"],
    "ocean heat": ["ocean heat content"],
    "hot extremes": ["hot extremes"],
    "biome": ["biome shifts"],
    "tropical marine": ["tropical marine species"],
    "near-term": ["near-term", "near term"],
    "overshoot": ["overshoot"],
    "figure 3.2": ["Figure 3.2"],
    "risks escalate": ["escalate with every increment"],
    "water scarcity": ["water scarcity in drylands"],
}

def normalize_ssp(text: str) -> List[str]:
    out = []
    for m in _SSP_PAT.finditer(text):
        ssp = f"SSP{m.group(1)}-{m.group(2)}"
        out.append(ssp)
    return out

def extract_keywords_from_question(q: str) -> List[str]:
    ql = q.lower()
    kws = set()
    for s in normalize_ssp(q): kws.add(s)
    for m in _DEGC_PAT.finditer(q):
        v = m.group(1)
        kws.add(f"{v}°C"); kws.add(f"{v} °C")
    for y in _YEAR_PAT.findall(q): kws.add(y)
    for key, arr in _PHRASES.items():
        if key in ql:
            for a in arr: kws.add(a)
    extras = []
    if "projected" in ql or "projection" in ql: extras += ["projected", "projection"]
    if "warming" in ql: extras += ["global warming", "warming"]
    if "near-term" in ql or "near term" in ql: extras += ["near-term"]
    if "co-benefits" in ql or "co benefits" in ql: extras += ["co-benefits", "co benefits"]
    for e in extras: kws.add(e)
    return [k for k in kws if k]

def suggest_pages_for_question(df: pd.DataFrame, q: str, top=8, extra_kw: List[str] | None = None) -> List[Tuple[str,str]]:
    terms = extract_keywords_from_question(q)
    if extra_kw:
        for k in extra_kw:
            if k and k not in terms: terms.append(k)
    scores = {}
    for t in terms:
        pages = find_pages(df, t, top=top)
        for rank, p in enumerate(pages):
            scores.setdefault(p, 0)
            bonus = 3 if re.search(r"SSP|°C|losses and damages|Figure 3\.2|ocean heat|sea level", t, re.I) else 1
            scores[p] += (top - rank) * bonus
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(p, str(score)) for p, score in ordered[:top]]

def main():
    ap = argparse.ArgumentParser(description="Checa gold_page e sugere páginas candidatas.")
    ap.add_argument("--file", default=str(EVAL_SET), help="Caminho do eval_set.jsonl")
    ap.add_argument("--question", "-q", default=None, help="Pergunta ad-hoc para sugerir páginas")
    ap.add_argument("--kw", default=None, help="Palavras-chave extras separadas por vírgula")
    ap.add_argument("--top", type=int, default=8, help="Quantidade de sugestões")
    args = ap.parse_args()

    kb_df = load_index_df(INDEX_DIR)

    if args.question:
        extras = [k.strip() for k in (args.kw.split(",") if args.kw else []) if k.strip()]
        print(f"\n[AD-HOC] Pergunta: {args.question}")
        sugg = suggest_pages_for_question(kb_df, args.question, top=args.top, extra_kw=extras)
        if not sugg:
            print("  (sem sugestões)")
        else:
            print("  Sugestões de páginas (com score simples):")
            for p, sc in sugg:
                smp = sample_texts(kb_df, p, n=1)
                print(f"   - p.{p}  score={sc}")
                if smp: print(f"       ex: {smp[0]}")
        return

    items = load_eval_items(Path(args.file))

    print("\n[DEBUG] Checando gold_page(s) no índice...")
    pages_in_index = set(kb_df["page"].unique().tolist())
    for it in items:
        q = it["question"]; gp = str(it["gold_page"])
        if gp in pages_in_index:
            ex = sample_texts(kb_df, gp, n=2)
            print(f"[OK] gold_page={gp} existe. Exemplos:")
            for e in ex:
                print(f"- {e}")
        else:
            print(f"[WARN] gold_page={gp} NÃO existe no índice!")

        sugg = suggest_pages_for_question(kb_df, q, top=args.top)
        if sugg:
            pack = ", ".join([f"p.{p}({sc})" for p, sc in sugg[:min(5, len(sugg))]])
            print(f"  → Sugestões pelo texto da pergunta: {pack}")
        else:
            print("  → Sem sugestões via texto da pergunta.")

    print("\n[DEBUG] Páginas candidatas por palavra-chave (globais):")
    keywords = [
        "well-mixed GHG", "greenhouse gas", "unequivocally caused global warming",
        "SSP2-4.5", "SSP5-8.5", "2100", "losses and damages",
        "ocean heat content", "global mean sea level", "Figure 3.2", "overshoot",
    ]
    for kw in keywords:
        pages = find_pages(kb_df, kw, top=10)
        print(f"  {kw:<35} → {pages}")

    print("\n[HINT] Use também:")
    print("  python -m eval.check_gold_pages --question \"Projected warming under SSP2-4.5 by 2100?\"")
    print("  python -m eval.check_gold_pages --question \"What happened to ocean heat content since the 1970s?\" --top 12")
    print("  python -m eval.check_gold_pages --question \"Beyond 4°C, what risks...\" --kw \"Figure 3.2\"")

if __name__ == "__main__":
    main()
