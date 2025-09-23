import json
from pathlib import Path

def test_evalset_schema_minimo():
    p = Path("eval/eval_set.jsonl")
    assert p.exists(), "Faltando eval/eval_set.jsonl"
    ok = 0
    with p.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            assert "question" in obj and "gold_page" in obj, "Linha sem campos obrigatórios"
            ok += 1
    assert ok > 0, "Conjunto de avaliação vazio"
