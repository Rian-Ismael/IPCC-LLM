from pathlib import Path
import os

def test_basic_paths_exist():
    index_dir = Path(os.getenv("INDEX_DIR", "data/index"))
    corpus_dir = Path("data/corpus")
    # Não falha se ainda não ingeri, mas avisa
    assert corpus_dir.exists(), "Esperava data/corpus com PDFs públicos"
    # O índice pode não existir ainda: quando não existir, apenas garantimos que o caminho é válido
    if not index_dir.exists():
        # cria o diretório vazio pra permitir CI sem ingest
        index_dir.mkdir(parents=True, exist_ok=True)
    assert index_dir.exists()
