# src/utils/settings.py
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

load_dotenv()

EMB_NAME = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_DIR = os.getenv("INDEX_DIR", "data/index")

EMB = SentenceTransformer(EMB_NAME)
DB = PersistentClient(path=INDEX_DIR)
COLL = DB.get_or_create_collection(name="ipcc")
