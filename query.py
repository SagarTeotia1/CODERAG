import os
import json
import time
from typing import List, Dict, Any, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm

from query_processor import multi_query_generate, classify_query, extract_query_signals, keyword_only
from bm25_retriever import BM25Retriever
from fusion import rrf_score
from reranker import fast_semantic_rerank, mmr_select
from embedder import embed_texts, backend_fingerprint


@dataclass
class Config:
    persist_dir: str
    collection_name: str
    model: str
    batch_size: int
    max_workers: int
    embedding_batch_size: int
    dense_top_k_per_query: int
    sparse_top_k: int
    final_top_k: int


def load_config() -> Config:
    load_dotenv(override=True)
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
    collection_name = os.getenv("COLLECTION_NAME", "mock_dataset_v1")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Create .env with GEMINI_API_KEY=<key>.")
    genai.configure(api_key=api_key)
    print(f"Loaded config. persist_dir={persist_dir}, collection={collection_name}", flush=True)
    return Config(
        persist_dir=persist_dir,
        collection_name=collection_name,
        model="text-embedding-004",
        batch_size=int(os.getenv("QUERY_BATCH_SIZE", "10")),
        max_workers=int(os.getenv("MAX_WORKERS", "5")),
        embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "20")),
        dense_top_k_per_query=int(os.getenv("DENSE_TOP_K_PER_QUERY", "50")),
        sparse_top_k=int(os.getenv("SPARSE_TOP_K", "100")),
        final_top_k=int(os.getenv("FINAL_TOP_K", "5")),
    )


_EMB_CACHE: Dict[Tuple[str, str], List[float]] = {}
_EMB_LOCK = threading.Lock()


def embed_queries_batch(texts: List[str], model: str, task_type: str = "retrieval_query") -> List[List[float]]:
    if not texts:
        return []
    # Serve from cache where possible
    # Include backend/model fingerprint in cache key so we can switch backends safely
    model_key = backend_fingerprint()
    keys = [(f"{t}", f"{task_type}:{model_key}") for t in texts]
    results: List[List[float]] = [None] * len(texts)  # type: ignore
    to_compute: List[Tuple[int, str]] = []
    with _EMB_LOCK:
        for i, k in enumerate(keys):
            if k in _EMB_CACHE:
                results[i] = _EMB_CACHE[k]
            else:
                to_compute.append((i, texts[i]))
    if to_compute:
        compute_texts = [t for _, t in to_compute]
        # Use shared embedder (local GPU or Gemini)
        computed = embed_texts(compute_texts, task_type=task_type)
        with _EMB_LOCK:
            for (i, t), emb in zip(to_compute, computed):
                _EMB_CACHE[(t, f"{task_type}:{model_key}")] = emb
                results[i] = emb
    # type: ignore
    return results  # filled


def fetch_all_from_collection(collection) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    limit = 1000
    offset = 0
    while True:
        print(f"Fetching collection batch offset={offset} limit={limit}...", flush=True)
        batch = collection.get(include=["metadatas", "documents"], limit=limit, offset=offset)
        if not batch or not batch.get("ids"):
            break
        ids.extend(batch.get("ids", []))
        documents.extend(batch.get("documents", []))
        metadatas.extend(batch.get("metadatas", []))
        if len(batch.get("ids", [])) < limit:
            break
        offset += limit
    return ids, documents, metadatas


def dense_search_variations(collection, model: str, query_variants: List[str], top_k: int) -> Tuple[Dict[str, Dict[str, int]], Dict[str, List[Tuple[Dict[str, Any], str]]]]:
    """Vectorized dense search across variants to reduce RPC overhead."""
    rankings: Dict[str, Dict[str, int]] = {}
    best_snippets_by_variant: Dict[str, List[Tuple[Dict[str, Any], str]]] = {}
    v_embs = embed_queries_batch(query_variants, model)
    # Single batched query for all variants
    res = collection.query(query_embeddings=v_embs, n_results=top_k, include=["documents", "metadatas"])
    res_docs = res.get("documents", []) or []
    res_metas = res.get("metadatas", []) or []
    for v, docs_list, metas_list in zip(query_variants, res_docs, res_metas):
        doc_to_rank: Dict[str, int] = {}
        pairs: List[Tuple[Dict[str, Any], str]] = []
        for rank, (meta, doc) in enumerate(zip(metas_list or [], docs_list or []), start=1):
            doc_name = meta.get("doc_name") or meta.get("source") or meta.get("chunk_id")
            if doc_name:
                if doc_name not in doc_to_rank:
                    doc_to_rank[doc_name] = rank
                pairs.append((meta, doc))
        rankings[f"dense:{v}"] = doc_to_rank
        best_snippets_by_variant[v] = pairs
    return rankings, best_snippets_by_variant


def sparse_search_bm25(bm25: BM25Retriever, query_variants: List[str], top_k: int) -> Dict[str, Dict[str, int]]:
    rankings: Dict[str, Dict[str, int]] = {}
    for v in query_variants:
        hits = bm25.search(v, top_k=top_k)
        doc_to_rank: Dict[str, int] = {}
        for rank, (_, _score, meta) in enumerate(hits, start=1):
            doc_name = meta.get("doc_name") or meta.get("source") or meta.get("chunk_id")
            if doc_name and doc_name not in doc_to_rank:
                doc_to_rank[doc_name] = rank
        rankings[f"bm25:{v}"] = doc_to_rank
    return rankings


def stage_rerank_and_mmr(model: str, query_text: str, candidate_docs: List[str], doc_best_snippet: Dict[str, str], final_top_k: int) -> List[str]:
    # Embed query
    q_emb = embed_queries_batch([query_text], model)[0]
    # Embed best snippets per doc
    snippets = [doc_best_snippet.get(d, "") for d in candidate_docs]
    emb_snips = embed_queries_batch(snippets, model, task_type="retrieval_document")
    doc_to_emb = {d: e for d, e in zip(candidate_docs, emb_snips)}
    # Term coverage
    keywords = set(keyword_only(query_text).split())
    coverage: Dict[str, float] = {}
    for d in candidate_docs:
        text = doc_best_snippet.get(d, "").lower()
        if not keywords:
            coverage[d] = 0.0
        else:
            hits = sum(1 for k in keywords if k in text)
            coverage[d] = hits / max(1, len(keywords))
    # Fast rerank to 20 then MMR to final
    top50 = fast_semantic_rerank(q_emb, doc_to_emb, coverage, top_k=min(50, len(candidate_docs)))
    ordered = [d for d, _ in top50]
    final_ids = mmr_select(ordered, doc_to_emb, q_emb, lambda_param=float(os.getenv("DIVERSITY_LAMBDA", "0.75")), final_k=final_top_k)
    return final_ids


def _process_single_query(q: Dict[str, str], collection, bm25: BM25Retriever, cfg: Config) -> Dict[str, Any]:
    q_text = q["query"]
    fast_mode = os.getenv("FAST_MODE", "0") in ("1", "true", "True")
    num_variants = 1 if fast_mode else int(os.getenv("NUM_QUERY_VARIATIONS", "3"))
    variants = multi_query_generate(q_text, num_variants=num_variants)
    
    if not fast_mode:
        _ = classify_query(q_text)
        _ = extract_query_signals(q_text)

    dense_rankings, best_snippets_by_variant = dense_search_variations(collection, cfg.model, variants, top_k=cfg.dense_top_k_per_query)
    use_bm25 = os.getenv("USE_BM25", "1") not in ("0", "false", "False")
    sparse_rankings = sparse_search_bm25(bm25, variants, top_k=cfg.sparse_top_k) if (use_bm25 and not fast_mode) else {}

    if fast_mode:
        # Fast mode: just use top dense results, no fusion/reranking
        top_docs = []
        for v in variants:
            doc_to_rank = dense_rankings.get(f"dense:{v}", {})
            top_docs.extend(list(doc_to_rank.keys())[:cfg.final_top_k])
        final_docs = list(dict.fromkeys(top_docs))[:cfg.final_top_k]  # Dedup and limit
    else:
        fused_scores = rrf_score({**dense_rankings, **sparse_rankings}, k=60)
        top_docs_sorted = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        top_doc_names = [d for d, _ in top_docs_sorted[:min(200, len(top_docs_sorted))]]

        doc_best_snippet: Dict[str, str] = {}
        for v in variants:
            for meta, doc in best_snippets_by_variant.get(v, []):
                name = meta.get("doc_name") or meta.get("source") or meta.get("chunk_id")
                if name in top_doc_names and name not in doc_best_snippet:
                    doc_best_snippet[name] = doc
            if len(doc_best_snippet) >= len(top_doc_names):
                break

        # Simplified reranking: skip expensive MMR if SKIP_RERANK is set
        if os.getenv("SKIP_RERANK", "0") in ("1", "true", "True"):
            final_docs = top_doc_names[:cfg.final_top_k]
        else:
            final_docs = stage_rerank_and_mmr(cfg.model, q_text, top_doc_names, doc_best_snippet, cfg.final_top_k)
    
    return {"query_num": q["query_num"], "query": q_text, "response": final_docs}


def main():
    print("Starting query pipeline...", flush=True)
    cfg = load_config()
    print("Connecting to Chroma...", flush=True)
    client = chromadb.PersistentClient(path=cfg.persist_dir, settings=Settings(anonymized_telemetry=False))
    print("Opening collection...", flush=True)
    collection = client.get_or_create_collection(name=cfg.collection_name, metadata={"hnsw:space": "cosine"})
    try:
        cnt = collection.count()
        print(f"Collection open. Vector records: {cnt}", flush=True)
    except Exception:
        pass

    # Build BM25 index once (optional)
    use_bm25 = os.getenv("USE_BM25", "1") not in ("0", "false", "False")
    bm25 = BM25Retriever() if use_bm25 else None
    if use_bm25:
        print("Fetching documents from collection for BM25...", flush=True)
        ids, docs, metas = fetch_all_from_collection(collection)
        if docs:
            print(f"Building BM25 over {len(docs)} docs...", flush=True)
            bm25.build(docs, ids, metas)

    queries_path = os.getenv("QUERIES_PATH", "./Queries.json")
    if not os.path.exists(queries_path):
        raise FileNotFoundError(f"Queries.json not found at {queries_path}")

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    valid_queries = []
    for item in queries:
        qnum = str(item.get("query_num"))
        qtext = str(item.get("query", "")).strip()
        if qnum and qtext:
            valid_queries.append({"query_num": qnum, "query": qtext})

    # Optional sharding for multi-process parallelism
    shard_total = int(os.getenv("SHARD_TOTAL", "1"))
    shard_index = int(os.getenv("SHARD_INDEX", "0"))
    if shard_total > 1:
        sliced = []
        for i, q in enumerate(valid_queries):
            if i % shard_total == shard_index:
                sliced.append(q)
        valid_queries = sliced

    print(f"Processing {len(valid_queries)} queries with hybrid retrieval and reranking...", flush=True)
    t0 = time.time()

    all_results: List[Dict[str, Any]] = []
    # Aggressive parallelism: increase workers significantly
    workers = int(os.getenv("QUERY_MAX_WORKERS", os.getenv("MAX_WORKERS", str(max(32, cfg.max_workers * 4)))))
    print(f"Using {workers} worker threads for parallel processing...", flush=True)
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_single_query, q, collection, bm25, cfg) for q in valid_queries]
        pbar = tqdm(total=len(futures), desc="Queries", unit="q")
        for fut in as_completed(futures):
            try:
                res = fut.result()
                all_results.append(res)
            except Exception as e:
                # On error, add empty response but keep going
                all_results.append({"query_num": "unknown", "query": "", "response": [], "error": str(e)})
            pbar.update(1)
        pbar.close()

    # Write results
    print("Writing results to files...")
    for result in tqdm(all_results, desc="Writing files", unit="file"):
        qnum = result["query_num"]
        out_obj = {"query": result["query"], "response": result["response"]}
        with open(f"query_{qnum}.json", "w", encoding="utf-8") as outf:
            json.dump(out_obj, outf, ensure_ascii=False, indent=2)
        print(f"Wrote query_{qnum}.json")

    total_time = time.time() - t0
    avg_time_per_query = total_time / max(1, len(valid_queries))
    print(f"\n{'='*50}")
    print("Query processing complete!")
    print(f"Total queries processed: {len(valid_queries)}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per query: {avg_time_per_query:.2f} seconds")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
