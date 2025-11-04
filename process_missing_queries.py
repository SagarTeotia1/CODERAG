import os
import json
import sys

# Import query.py as a module
import query
from bm25_retriever import BM25Retriever
from dotenv import load_dotenv

def process_missing_queries(query_nums: list):
    """Process specific query numbers and write JSON files."""
    load_dotenv(override=True)
    cfg = query.load_config()
    
    # Connect to Chroma
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=cfg.persist_dir, settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name=cfg.collection_name, metadata={"hnsw:space": "cosine"})
    
    # BM25 is optional (skip if USE_BM25=0)
    use_bm25 = os.getenv("USE_BM25", "1") not in ("0", "false", "False")
    bm25 = BM25Retriever() if use_bm25 else None
    if use_bm25:
        print("Building BM25 index...", flush=True)
        ids, docs, metas = query.fetch_all_from_collection(collection)
        if docs:
            bm25.build(docs, ids, metas)
    else:
        print("Skipping BM25 (USE_BM25=0)", flush=True)
    
    # Load queries
    queries_path = os.getenv("QUERIES_PATH", "./Queries.json")
    with open(queries_path, "r", encoding="utf-8") as f:
        all_queries = json.load(f)
    
    # Find queries by query_num
    query_nums_set = set(str(q) for q in query_nums)
    target_queries = []
    for item in all_queries:
        qnum = str(item.get("query_num"))
        if qnum in query_nums_set:
            qtext = str(item.get("query", "")).strip()
            if qnum and qtext:
                target_queries.append({"query_num": qnum, "query": qtext})
    
    if not target_queries:
        print(f"Warning: No queries found for query_nums {query_nums}", flush=True)
        return
    
    print(f"Processing {len(target_queries)} missing queries: {[q['query_num'] for q in target_queries]}", flush=True)
    
    # Process each query
    for q in target_queries:
        try:
            result = query._process_single_query(q, collection, bm25, cfg)
            qnum = result["query_num"]
            out_obj = {"query": result["query"], "response": result["response"]}
            out_path = f"query_{qnum}.json"
            with open(out_path, "w", encoding="utf-8") as outf:
                json.dump(out_obj, outf, ensure_ascii=False, indent=2)
            print(f"✓ Wrote {out_path} ({len(result['response'])} docs)", flush=True)
        except Exception as e:
            print(f"✗ Error processing query {q.get('query_num')}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # Write empty response
            qnum = q.get("query_num", "unknown")
            out_obj = {"query": q.get("query", ""), "response": [], "error": str(e)}
            with open(f"query_{qnum}.json", "w", encoding="utf-8") as outf:
                json.dump(out_obj, outf, ensure_ascii=False, indent=2)
    
    print("Done!", flush=True)

if __name__ == "__main__":
    # Missing query numbers from user
    missing = [3007, 3008, 9003, 9005, 9006, 9011]
    process_missing_queries(missing)
