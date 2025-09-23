# scripts/clean_index.py
import os, shutil, sys
from pathlib import Path
try:
    from dotenv import load_dotenv
except ImportError:
    print("[clean_index] python-dotenv não instalado. Rode: pip install python-dotenv", file=sys.stderr)
    sys.exit(1)

def main():
    load_dotenv()  # carrega .env para este processo
    idx = os.getenv("INDEX_DIR", "data/index")
    p = Path(idx)
    print(f"[clean_index] INDEX_DIR lido do .env = {idx}")
    print(f"[clean_index] Caminho absoluto = {p.resolve()}")
    if p.exists():
        try:
            shutil.rmtree(p, ignore_errors=True)
            print("[clean_index] Pasta removida.")
        except Exception as e:
            print(f"[clean_index] ERRO ao remover: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("[clean_index] Nada para remover (pasta não existe).")
    # garantia: recria vazia (opcional)
    p.mkdir(parents=True, exist_ok=True)
    print("[clean_index] OK.")

if __name__ == "__main__":
    main()
