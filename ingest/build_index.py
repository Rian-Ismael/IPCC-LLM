# ingest/build_index.py
import argparse, os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from src.utils.pdf_loader import load_pdf_with_metadata

load_dotenv()

DEFAULT_EMB = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def main(pdf_path: str, index_dir: str):
    os.makedirs(index_dir, exist_ok=True)

    # 1) carrega páginas (já normalizadas no loader)
    docs = load_pdf_with_metadata(pdf_path)  # [{text, page}, ...]
    num_pages = len(docs)

    # 2) chunking
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
                "metadata": {"page": d["page"]},
            })

    # 3) embedders
    emb = SentenceTransformer(DEFAULT_EMB)

    # 4) Chroma persistente em COSINE (drop para garantir métrica)
    client = PersistentClient(path=index_dir)
    try:
        client.delete_collection("ipcc")
    except Exception:
        pass

    coll = client.get_or_create_collection(
        name="ipcc",
        metadata={"hnsw:space": "cosine"}  # ESSENCIAL
    )

    # 5) adiciona
    ids = [f"ipcc-{i}" for i in range(len(chunks))]
    texts = [ch["text"] for ch in chunks]
    metas = [ch["metadata"] for ch in chunks]

    vecs = emb.encode(texts, convert_to_numpy=True).tolist()
    coll.add(ids=ids, documents=texts, metadatas=metas, embeddings=vecs)

    print(f"Indexed {len(texts)} chunks from {num_pages} pages → {index_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--index-dir", required=True)
    args = ap.parse_args()
    main(args.pdf, args.index_dir)
