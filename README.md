# RAG Platform (Gemini Embeddings + ChromaDB)

## Setup

1. Install Python 3.10+
2. Create a virtual environment and activate it
3. Install dependencies:

```
pip install -r requirements.txt
```

4. Create a `.env` file with:

```
GEMINI_API_KEY=your_key_here
CHROMA_PERSIST_DIR=./chroma_store
COLLECTION_NAME=mock_dataset_v1
CHUNK_TOKENS=800
CHUNK_OVERLAP=100
BATCH_SIZE=64
MAX_WORKERS=10
EMBEDDING_BATCH_SIZE=100
```

5. Place `mock_dataset_v1.zip` and `Queries.json` in this directory.

## Indexing (run once)

```
python ingest.py
```

- Unzips and reads all `.txt` files inside the zip (in-memory)
- Cleans text, chunks by word-count with overlap
- Generates Gemini embeddings (cached in SQLite)
- Upserts into ChromaDB (persistent)

## Querying

```
python query.py
```

- Reads `Queries.json` (array of {"query_num", "query"})
- Embeds each query with Gemini
- Searches ChromaDB and writes `query_<query_num>.json` with top-5 unique doc names

## Performance Optimizations

- **Concurrent Processing**: Uses ThreadPoolExecutor for parallel API calls
- **Batch Embedding**: Processes multiple texts simultaneously when possible
- **Connection Pooling**: Optimized database operations with connection reuse
- **Caching**: SQLite-based embedding cache to avoid recomputation
- **Progress Tracking**: Real-time speed metrics during processing

## Configuration

- `MAX_WORKERS`: Number of concurrent threads for API calls (default: 10)
- `EMBEDDING_BATCH_SIZE`: Size of embedding batches (default: 100)
- `BATCH_SIZE`: ChromaDB upsert batch size (default: 64)

## Notes

- Uses `text-embedding-004` model
- Persistence path: `CHROMA_PERSIST_DIR`
- Change batching/chunk sizes via `.env`
- Optimized for speed with concurrent processing and caching
