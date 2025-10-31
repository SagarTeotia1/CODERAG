from typing import List, Dict, Tuple
import math


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def fast_semantic_rerank(query_emb: List[float], doc_id_to_best_emb: Dict[str, List[float]], doc_id_to_term_cov: Dict[str, float], top_k: int = 50) -> List[Tuple[str, float]]:
    scored: List[Tuple[str, float]] = []
    for doc_id, emb in doc_id_to_best_emb.items():
        sim = cosine_similarity(query_emb, emb)
        coverage = doc_id_to_term_cov.get(doc_id, 0.0)
        score = 0.7 * sim + 0.3 * coverage
        scored.append((doc_id, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def mmr_select(doc_ids: List[str], doc_embs: Dict[str, List[float]], query_emb: List[float], lambda_param: float = 0.75, final_k: int = 5) -> List[str]:
    if not doc_ids:
        return []
    selected: List[str] = [doc_ids[0]]
    candidates = set(doc_ids[1:])
    while len(selected) < min(final_k, len(doc_ids)):
        best_id = None
        best_score = -1e9
        for cid in candidates:
            rel = cosine_similarity(query_emb, doc_embs.get(cid, []))
            div = 0.0
            for sid in selected:
                div = max(div, cosine_similarity(doc_embs.get(cid, []), doc_embs.get(sid, [])))
            score = lambda_param * rel - (1 - lambda_param) * div
            if score > best_score:
                best_score = score
                best_id = cid
        if best_id is None:
            break
        selected.append(best_id)
        candidates.remove(best_id)
    return selected[:final_k]


