import os
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
import google.generativeai as genai


# Configure Gemini once on import
load_dotenv(override=True)
_API_KEY = os.getenv("GEMINI_API_KEY")
if not _API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set. Create .env with GEMINI_API_KEY=<key>.")
genai.configure(api_key=_API_KEY)

_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")


def backend_fingerprint() -> str:
    """Identify the embedding backend and model for cache keys.

    The value should be stable across runs when the backend/model are unchanged.
    """
    return f"gemini:{_MODEL}"


def _embed_single(text: str, task_type: str) -> List[float]:
    # Use task_type to hint retrieval direction (query vs document)
    # Gemini supports task_type but tolerates absence; we pass it when available.
    resp = genai.embed_content(model=_MODEL, content=text, task_type=task_type)
    return resp["embedding"] if isinstance(resp, dict) else resp.embedding


def embed_texts(texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
    """Embed a list of texts using the configured Gemini model.

    - Returns embeddings in the same order as inputs.
    - Uses a thread pool for concurrency to improve throughput.
    """
    if not texts:
        return []

    # Try batch call first if a sequence is provided (best-effort; API may change)
    try:
        batch_resp = genai.embed_content(model=_MODEL, content=texts, task_type=task_type)
        if isinstance(batch_resp, list):
            return [r["embedding"] if isinstance(r, dict) else r.embedding for r in batch_resp]
    except Exception:
        # Fall back to concurrent single calls
        pass

    workers = int(os.getenv("EMBED_MAX_WORKERS", os.getenv("MAX_WORKERS", "10")))
    results: List[List[float]] = [None] * len(texts)  # type: ignore
    with ThreadPoolExecutor(max_workers=min(max(1, workers), len(texts))) as executor:
        futures = {executor.submit(_embed_single, t, task_type): i for i, t in enumerate(texts)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception:
                # Graceful degradation: return a zero vector with expected dim (768 for text-embedding-004)
                results[i] = [0.0] * 768
    # type: ignore
    return results


