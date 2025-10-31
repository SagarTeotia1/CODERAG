from typing import Dict, Tuple

def rrf_score(rankings: Dict[str, Dict[str, int]], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for method, doc_to_rank in rankings.items():
        for doc_id, rank in doc_to_rank.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores

def apply_boost(scores: Dict[str, float], boosts: Dict[str, float], weight: float = 0.2) -> Dict[str, float]:
    out = dict(scores)
    for doc_id, b in boosts.items():
        out[doc_id] = out.get(doc_id, 0.0) * (1.0 + weight * b)
    return out


