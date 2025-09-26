from pathlib import Path
import os

def test_basic_paths_exist():
    index_dir = Path(os.getenv("INDEX_DIR", "data/index"))
    corpus_dir = Path("data/corpus")
    assert corpus_dir.exists(), "Esperava data/corpus com PDFs públicos"
    if not index_dir.exists():
        index_dir.mkdir(parents=True, exist_ok=True)
    assert index_dir.exists()
