from typing import List, Dict
from src.utils.settings import COLL, EMB

K = 4

def retrieve(query: str) -> List[Dict]:
    qv = EMB.encode([query], convert_to_numpy=True).tolist()[0]
    res = COLL.query(query_embeddings=[qv], n_results=K)
    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "metadata": res["metadatas"][0][i]
        })
    return hits