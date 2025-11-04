import os
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
import google.generativeai as genai

# Optional GPU backend
_USE_GPU = os.getenv("USE_GPU_EMBEDDINGS", "0") in ("1", "true", "True")
_GPU_MODEL_NAME = os.getenv("GPU_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_GPU_BATCH_SIZE = int(os.getenv("GPU_BATCH_SIZE", "256"))
_gpu_model = None
_gpu_device = None


# Configure backends once on import
load_dotenv(override=True)

if _USE_GPU:
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        _gpu_device = "cuda" if torch.cuda.is_available() else "cpu"
        _gpu_model = SentenceTransformer(_GPU_MODEL_NAME, device=_gpu_device)
        # Warm-up small encode to initialize GPU
        _gpu_model.encode(["warmup"], batch_size=1, convert_to_numpy=True)
    except Exception as e:
        # Fallback to Gemini if GPU path fails
        _USE_GPU = False

if not _USE_GPU:
    _API_KEY = os.getenv("GEMINI_API_KEY")
    if not _API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set. Create .env with GEMINI_API_KEY=<key>.")
    genai.configure(api_key=_API_KEY)

_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")


def backend_fingerprint() -> str:
    """Identify the embedding backend and model for cache keys.

    The value should be stable across runs when the backend/model are unchanged.
    """
    if _USE_GPU and _gpu_model is not None:
        return f"gpu:{_GPU_MODEL_NAME}"
    return f"gemini:{_MODEL}"


def _embed_single(text: str, task_type: str) -> List[float]:
    if _USE_GPU and _gpu_model is not None:
        # Single encode on GPU is inefficient; batch path handles most loads
        vec = _gpu_model.encode([text], batch_size=1, convert_to_numpy=True)[0]
        return vec.tolist()
    # Use task_type to hint retrieval direction (query vs document)
    resp = genai.embed_content(model=_MODEL, content=text, task_type=task_type)
    return resp["embedding"] if isinstance(resp, dict) else resp.embedding


def embed_texts(texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
    """Embed a list of texts using GPU (sentence-transformers) or Gemini.

    - Returns embeddings in the same order as inputs.
    - Uses GPU batching when available; otherwise Gemini batch API then threaded fallback.
    """
    if not texts:
        return []

    if _USE_GPU and _gpu_model is not None:
        try:
            vecs = _gpu_model.encode(texts, batch_size=_GPU_BATCH_SIZE, convert_to_numpy=True, show_progress_bar=False)
            return [v.tolist() for v in vecs]
        except Exception:
            # If GPU path fails mid-run, fall back to Gemini below
            pass

    # Gemini path: try batch call first
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
                # Graceful degradation: default to 384 dims for MiniLM, 768 for Gemini
                default_dim = 384 if _USE_GPU else 768
                results[i] = [0.0] * default_dim
    # type: ignore
    return results


