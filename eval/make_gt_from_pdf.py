# eval/make_gt_from_pdf.py
import json, argparse
from pathlib import Path
import fitz  # PyMuPDF
import re
import textwrap

def read_jsonl(p: Path):
    rows=[]
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln: 
                continue
            rows.append(json.loads(ln))
    return rows

def write_jsonl(p: Path, rows):
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def clean_page_text(txt: str) -> str:
    # remove números de página soltos e cabeçalhos óbvios
    txt = re.sub(r"\n{2,}", "\n\n", txt)          # normaliza quebras
    txt = re.sub(r"[ \t]+", " ", txt)             # espaços múltiplos
    # des-hifenização simples (palavra-\ncontinua -> palavracontinua)
    txt = re.sub(r"(\w)-\n(\w)", r"\1\2", txt)
    # quebra longa em 1 parágrafo (ajuda o matcher do RAGAS)
    txt = textwrap.dedent(txt).strip()
    return txt

def extract_page_text(pdf_path: Path, page_num: int) -> str:
    # gold_page é 1-based; PyMuPDF é 0-based
    idx = max(0, page_num - 1)
    with fitz.open(pdf_path) as doc:
        if idx >= len(doc):
            return ""
        page = doc[idx]
        return page.get_text("text").strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="JSONL de entrada (com gold_page)")
    ap.add_argument("--pdf", dest="pdf", required=True, help="PDF fonte")
    ap.add_argument("--out", dest="out", required=True, help="JSONL de saída com ground_truth preenchido")
    ap.add_argument("--max-chars", type=int, default=4000, help="Limite de caracteres do ground_truth")
    args = ap.parse_args()

    inp = Path(args.inp); pdf = Path(args.pdf); outp = Path(args.out)
    rows = read_jsonl(inp)

    new=[]
    for r in rows:
        gp = r.get("gold_page")
        gt_current = r.get("ground_truth") or r.get("expected_answer") or ""
        if (not gt_current) and isinstance(gp, (int, float, str)):
            try:
                gp_int = int(gp)
                txt = extract_page_text(pdf, gp_int)
                txt = clean_page_text(txt)
                r["ground_truth"] = txt[:args.max_chars] if txt else ""
            except Exception:
                # se der erro no parse dessa linha, só segue
                pass
        new.append(r)

    outp.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(outp, new)
    print(f"[OK] Escrevi {len(new)} linhas em {outp}")

if __name__ == "__main__":
    main()
