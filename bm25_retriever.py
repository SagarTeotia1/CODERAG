from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
import threading


class BM25Retriever:
    def __init__(self):
        self._lock = threading.Lock()
        self._bm25 = None
        self._docs: List[str] = []
        self._ids: List[str] = []
        self._metas: List[Dict[str, Any]] = []

    def build(self, documents: List[str], ids: List[str], metadatas: List[Dict[str, Any]]):
        tokens = [d.lower().split() for d in documents]
        with self._lock:
            self._bm25 = BM25Okapi(tokens)
            self._docs = documents
            self._ids = ids
            self._metas = metadatas

    def is_ready(self) -> bool:
        with self._lock:
            return self._bm25 is not None

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float, Dict[str, Any]]]:
        with self._lock:
            if self._bm25 is None:
                return []
            scores = self._bm25.get_scores(query.lower().split())
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
            results: List[Tuple[str, float, Dict[str, Any]]] = []
            for idx, score in ranked:
                results.append((self._ids[idx], float(score), self._metas[idx]))
            return results


