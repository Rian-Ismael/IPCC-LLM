import json
from pathlib import Path

def test_evalset_no_duplicates_and_numeric_pages():
    p = Path("eval/eval_set.jsonl")
    assert p.exists(), "Faltando eval/eval_set.jsonl"

    seen = set()
    n = 0
    with p.open("r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            q = obj.get("question"); gp = obj.get("gold_page")
            assert isinstance(q, str) and q.strip(), "question deve ser string não vazia"
            assert isinstance(gp, (str, int)), "gold_page deve ser str/int"
            # páginas devem ser numéricas (mas aceitamos str " 40 ")
            try:
                _ = int(str(gp).strip())
            except Exception:
                raise AssertionError(f"gold_page não numérico: {gp!r}")
            # duplicatas de pergunta
            assert q not in seen, f"Pergunta duplicada: {q}"
            seen.add(q)
            n += 1
    assert n > 0, "Conjunto de avaliação vazio"
