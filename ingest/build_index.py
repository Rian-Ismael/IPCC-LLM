# ingest/build_index.py
import argparse, os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from src.utils.pdf_loader import load_pdf_with_metadata  # já traz texto normalizado

load_dotenv()

DEFAULT_EMB = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def main(pdf_path: str, index_dir: str):
    os.makedirs(index_dir, exist_ok=True)

    # 1) Carrega o PDF em páginas (texto já normalizado pelo loader)
    docs = load_pdf_with_metadata(pdf_path)  # [{text, page}, ...]

    # 2) Splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

    chunks = []
    for d in docs:
        parts = splitter.split_text(d["text"])
        for c in parts:
            c = (c or "").strip()
            if not c:
                continue  # evita chunk vazio
            chunks.append({
                "text": c,
                "metadata": {"page": d["page"], "section": d.get("section", "")},
            })

    # 3) Embeddings
    emb = SentenceTransformer(DEFAULT_EMB)

    # 4) Chroma (persistente)
    client = PersistentClient(path=index_dir)

    # 4.1) Zera a coleção se já existir (evita duplicar documentos)
    try:
        existing = {c.name for c in client.list_collections()}
        if "ipcc" in existing:
            client.delete_collection(name="ipcc")
    except Exception:
        pass

    coll = client.get_or_create_collection(name="ipcc")

    # 5) Adiciona documentos
    ids, texts, metas = [], [], []
    for i, ch in enumerate(chunks):
        ids.append(f"ipcc-{i}")
        texts.append(ch["text"])
        metas.append(ch["metadata"])

    vecs = emb.encode(texts, convert_to_numpy=True).tolist()
    coll.add(ids=ids, documents=texts, metadatas=metas, embeddings=vecs)

    print(f"Indexed {len(texts)} chunks → {index_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--index-dir", required=True)
    args = ap.parse_args()
    main(args.pdf, args.index_dir)
