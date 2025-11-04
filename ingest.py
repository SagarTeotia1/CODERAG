import os
import io
import json
import hashlib
import sqlite3
import zipfile
import asyncio
import aiohttp
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm

import numpy as np


@dataclass
class Config:
	persist_dir: str
	collection_name: str
	chunk_tokens: int
	chunk_overlap: int
	batch_size: int
	zip_path: str
	dataset_dir: str
	cache_path: str
	gemini_model: str = "text-embedding-004"
	max_workers: int = 10
	embedding_batch_size: int = 100
	# Multi-granularity (defaults; can be overridden via env)
	parent_chunk: int = 1024
	standard_chunk: int = 512
	micro_chunk: int = 256
	standard_overlap: int = 128
	micro_overlap: int = 64


def load_config() -> Config:
	load_dotenv(override=True)
	persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
	collection_name = os.getenv("COLLECTION_NAME", "shortlisting_dataset")
	chunk_tokens = int(os.getenv("CHUNK_TOKENS", "800"))
	chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
	batch_size = int(os.getenv("BATCH_SIZE", "64"))
	zip_path = os.getenv("DATASET_ZIP", "./shortlisting_dataset.zip")
	dataset_dir = os.getenv("DATASET_DIR", "./Shortlisting Dataset")
	cache_path = os.path.join(persist_dir, "emb_cache.sqlite")
	max_workers = int(os.getenv("MAX_WORKERS", "10"))
	embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
	api_key = os.getenv("GEMINI_API_KEY")
	if not api_key:
		raise RuntimeError("GEMINI_API_KEY not set. Create .env with GEMINI_API_KEY=<key>.")
	genai.configure(api_key=api_key)
	return Config(
		persist_dir=persist_dir,
		collection_name=collection_name,
		chunk_tokens=chunk_tokens,
		chunk_overlap=chunk_overlap,
		batch_size=batch_size,
		zip_path=zip_path,
		dataset_dir=dataset_dir,
		cache_path=cache_path,
		max_workers=max_workers,
		embedding_batch_size=embedding_batch_size,
		parent_chunk=int(os.getenv("PARENT_CHUNK_SIZE", "1024")),
		standard_chunk=int(os.getenv("STANDARD_CHUNK_SIZE", "512")),
		micro_chunk=int(os.getenv("MICRO_CHUNK_SIZE", "256")),
		standard_overlap=int(os.getenv("STANDARD_CHUNK_OVERLAP", "128")),
		micro_overlap=int(os.getenv("MICRO_CHUNK_OVERLAP", "64")),
	)


def normalize_text(text: str) -> str:
	text = text.replace("\r\n", "\n").replace("\r", "\n")
	# Collapse excessive whitespace
	return "\n".join(" ".join(line.split()) for line in text.split("\n")).strip()


def hash_text(text: str, model: str) -> str:
	m = hashlib.sha1()
	m.update(model.encode("utf-8"))
	m.update(b"\x00")
	m.update(text.encode("utf-8"))
	return m.hexdigest()


def detect_sections(text: str) -> List[str]:
	lines = text.split("\n")
	sections: List[str] = []
	current: List[str] = []
	def is_heading(line: str) -> bool:
		l = line.strip()
		return (
			l.startswith("#") or
			(len(l) > 0 and l.isupper()) or
			re.match(r"^(\d+\.|[A-Z]\))\s", l or "") is not None
		)
	for line in lines:
		if is_heading(line) and current:
			sections.append("\n".join(current).strip())
			current = [line]
		else:
			current.append(line)
	if current:
		sections.append("\n".join(current).strip())
	return [s for s in sections if s.strip()]


def word_chunk(text: str, chunk_tokens: int, overlap: int) -> List[str]:
	words = text.split()
	if not words:
		return []
	chunks: List[str] = []
	start = 0
	while start < len(words):
		end = min(start + chunk_tokens, len(words))
		chunk = " ".join(words[start:end]).strip()
		if chunk:
			chunks.append(chunk)
		if end == len(words):
			break
		start = max(0, end - overlap)
	return chunks


def chunk_multi_granularity(text: str, cfg: Config) -> List[Tuple[str, str, str, int, str]]:
	results: List[Tuple[str, str, str, int, str]] = []
	sections = detect_sections(text) or [text]
	parent_index = 0
	for sec in sections:
		parents = word_chunk(sec, cfg.parent_chunk, 0) or [sec]
		for ppos, pchunk in enumerate(parents):
			parent_suffix = f"parent_{parent_index}"
			results.append((parent_suffix, "parent", pchunk, ppos, parent_suffix))
			stds = word_chunk(pchunk, cfg.standard_chunk, cfg.standard_overlap)
			for spos, s in enumerate(stds):
				std_suffix = f"{parent_suffix}_std_{spos}"
				results.append((std_suffix, "standard", s, spos, parent_suffix))
				micros = word_chunk(s, cfg.micro_chunk, cfg.micro_overlap)
				for mpos, m in enumerate(micros):
					micro_suffix = f"{std_suffix}_micro_{mpos}"
					results.append((micro_suffix, "micro", m, mpos, parent_suffix))
			parent_index += 1
	return results


def compute_doc_stats(text: str) -> Dict[str, float]:
	words = [w for w in re.split(r"\W+", text) if w]
	word_count = len(words)
	unique_terms = len(set(w.lower() for w in words))
	sentences = [s for s in re.split(r"[.!?]+\s", text) if s.strip()]
	avg_sentence_length = (sum(len(s.split()) for s in sentences) / max(1, len(sentences))) if sentences else 0.0
	return {
		"word_count": float(word_count),
		"unique_terms_count": float(unique_terms),
		"avg_sentence_length": float(avg_sentence_length),
	}


def extract_simple_entities(text: str) -> Dict[str, List[str]]:
	emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
	urls = re.findall(r"https?://\S+", text)
	years = re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", text)
	numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
	return {
		"emails": list(set(emails)),
		"urls": list(set(urls)),
		"years": list(set(years)),
		"numbers": list(set(numbers)),
	}


STOPWORDS = set(["the","is","at","which","on","and","a","an","of","to","in","for","by","with","as","from","that","this"])


def top_keywords(text: str, k: int = 10) -> List[str]:
	counts: Dict[str, int] = {}
	for tok in re.split(r"\W+", text.lower()):
		if not tok or tok in STOPWORDS:
			continue
		counts[tok] = counts.get(tok, 0) + 1
	return [w for w, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:k]]


class EmbeddingCache:
	def __init__(self, sqlite_path: str):
		self.sqlite_path = sqlite_path
		os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
		self._init_db()
		# Use connection pooling for better performance
		self._conn_pool = []

	def _init_db(self) -> None:
		conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
		try:
			c = conn.cursor()
			c.execute(
				"""
				CREATE TABLE IF NOT EXISTS embeddings (
					id TEXT PRIMARY KEY,
					model TEXT NOT NULL,
					vector BLOB NOT NULL
				)
				"""
			)
			# Create index for faster lookups
			c.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_id ON embeddings(id)")
			conn.commit()
		finally:
			conn.close()

	def _get_connection(self):
		"""Get a connection from pool or create new one"""
		if self._conn_pool:
			return self._conn_pool.pop()
		return sqlite3.connect(self.sqlite_path, check_same_thread=False)

	def _return_connection(self, conn):
		"""Return connection to pool"""
		if len(self._conn_pool) < 5:  # Limit pool size
			self._conn_pool.append(conn)
		else:
			conn.close()

	def get_many(self, ids: List[str]) -> List[Tuple[str, np.ndarray]]:
		if not ids:
			return []
		conn = self._get_connection()
		try:
			c = conn.cursor()
			placeholders = ",".join(["?"] * len(ids))
			c.execute(f"SELECT id, vector FROM embeddings WHERE id IN ({placeholders})", ids)
			rows = c.fetchall()
			result: List[Tuple[str, np.ndarray]] = []
			for row in rows:
				vec = np.frombuffer(row[1], dtype=np.float32)
				result.append((row[0], vec))
			return result
		finally:
			self._return_connection(conn)

	def put_many(self, entries: List[Tuple[str, str, np.ndarray]]) -> None:
		if not entries:
			return
		conn = self._get_connection()
		try:
			c = conn.cursor()
			c.executemany(
				"INSERT OR REPLACE INTO embeddings(id, model, vector) VALUES(?,?,?)",
				[(eid, model, vec.astype(np.float32).tobytes()) for (eid, model, vec) in entries],
			)
			conn.commit()
		finally:
			self._return_connection(conn)

	def close(self):
		"""Close all connections in pool"""
		for conn in self._conn_pool:
			conn.close()
		self._conn_pool.clear()


def embed_texts_gemini_batch(texts: List[str], model: str) -> List[List[float]]:
	"""Fast batch embedding using concurrent processing"""
	if not texts:
		return []
	
	# Use batch API if available, otherwise process in parallel
	try:
		# Try batch embedding first (if supported by the API)
		responses = genai.embed_content(
			model=model, 
			content=texts,
			task_type="retrieval_document"
		)
		if isinstance(responses, list):
			return [resp["embedding"] if isinstance(resp, dict) else resp.embedding for resp in responses]
		else:
			# Fallback to individual calls if batch not supported
			raise NotImplementedError("Batch not supported")
	except Exception:
		# Fallback to concurrent individual calls
		return embed_texts_gemini_concurrent(texts, model)


def embed_texts_gemini_concurrent(texts: List[str], model: str) -> List[List[float]]:
	"""Concurrent individual embedding calls for maximum compatibility"""
	def embed_single(text: str) -> List[float]:
		try:
			resp = genai.embed_content(model=model, content=text)
			emb = resp["embedding"] if isinstance(resp, dict) else resp.embedding
			return emb
		except Exception as e:
			print(f"Error embedding text: {e}")
			# Return zero vector as fallback
			return [0.0] * 768  # text-embedding-004 has 768 dimensions
	
	embeddings: List[List[float]] = []
	
	# Use ThreadPoolExecutor for concurrent API calls (env-configurable)
	workers = int(os.getenv("EMBED_MAX_WORKERS", os.getenv("MAX_WORKERS", "10")))
	with ThreadPoolExecutor(max_workers=min(workers, len(texts))) as executor:
		future_to_text = {executor.submit(embed_single, text): text for text in texts}
		
		# Maintain order by using a list
		text_to_embedding = {}
		for future in as_completed(future_to_text):
			text = future_to_text[future]
			try:
				embedding = future.result()
				text_to_embedding[text] = embedding
			except Exception as e:
				print(f"Error processing {text[:50]}...: {e}")
				text_to_embedding[text] = [0.0] * 768
		
		# Return embeddings in original order
		for text in texts:
			embeddings.append(text_to_embedding[text])
	
	return embeddings


def process_embedding_batch(batch_data: Tuple[List[str], str, str]) -> Tuple[List[str], List[List[float]]]:
	"""Process a batch of texts for embedding - used by concurrent workers"""
	texts, model, batch_id = batch_data
	embeddings = embed_texts_gemini_batch(texts, model)
	return batch_id, embeddings


def main():
	cfg = load_config()

	client = chromadb.PersistentClient(path=cfg.persist_dir, settings=Settings(anonymized_telemetry=False))
	collection = client.get_or_create_collection(
		name=cfg.collection_name,
		metadata={"hnsw:space": "cosine"},
	)

	cache = EmbeddingCache(cfg.cache_path)

	# Prefer directory of .txt files if provided; otherwise fall back to zip
	use_dir = bool(cfg.dataset_dir and os.path.isdir(cfg.dataset_dir))
	use_zip = bool(os.path.exists(cfg.zip_path))

	if not use_dir and not use_zip:
		raise FileNotFoundError(
			f"No dataset found. Set DATASET_DIR to a folder of .txt files (e.g., ./mockdataset) or DATASET_ZIP to a zip archive (e.g., ./mock_dataset_v1.zip)."
		)

	all_ids: List[str] = []
	all_texts: List[str] = []
	all_metadatas: List[dict] = []
	all_embeddings: List[List[float]] = []

	if use_dir:
		file_paths: List[str] = []
		for root, _dirs, files in os.walk(cfg.dataset_dir):
			for name in files:
				if name.lower().endswith('.txt'):
					file_paths.append(os.path.join(root, name))
		if not file_paths:
			raise RuntimeError(f"No .txt files found under {cfg.dataset_dir}")
		pbar = tqdm(file_paths, desc="Reading & chunking", unit="file")
		for fpath in pbar:
			with open(fpath, 'rb') as fh:
				data = fh.read()
			try:
				text = data.decode('utf-8', errors='ignore')
			except Exception:
				text = data.decode('latin-1', errors='ignore')
			text = normalize_text(text)
			doc_name = os.path.basename(fpath)
			stats = compute_doc_stats(text)
			entities = extract_simple_entities(text)
			keywords = top_keywords(text)
			mg = chunk_multi_granularity(text, cfg)
			for suffix, chunk_type, ch, pos, parent_suffix in mg:
				chunk_id = f"{doc_name}::{suffix}"
				all_ids.append(chunk_id)
				all_texts.append(ch)
				qfocus = f"This document discusses: {ch[:200]}"
				all_metadatas.append({
					"chunk_id": chunk_id,
					"doc_name": doc_name,
					"source": os.path.relpath(fpath, cfg.dataset_dir),
					"chunk_type": chunk_type,
					"parent_id": f"{doc_name}::{parent_suffix}",
					"position": pos,
					"doc_stats": json.dumps(stats, ensure_ascii=False),
					"doc_entities": json.dumps(entities, ensure_ascii=False),
					"doc_top_keywords": json.dumps(keywords, ensure_ascii=False),
					"query_focused_text": qfocus,
				})
	else:
		with zipfile.ZipFile(cfg.zip_path, 'r') as zf:
			# Collect .txt files
			txt_members = [m for m in zf.infolist() if m.filename.lower().endswith('.txt')]
			if not txt_members:
				raise RuntimeError("No .txt files found in the zip archive.")
			pbar = tqdm(txt_members, desc="Reading & chunking", unit="file")
			for info in pbar:
				with zf.open(info, 'r') as fh:
					data = fh.read()
					try:
						text = data.decode('utf-8', errors='ignore')
					except Exception:
						text = data.decode('latin-1', errors='ignore')
				text = normalize_text(text)
				doc_name = os.path.basename(info.filename)
				# Document-level enrichments
				stats = compute_doc_stats(text)
				entities = extract_simple_entities(text)
				keywords = top_keywords(text)
				# Multi-granularity chunking
				mg = chunk_multi_granularity(text, cfg)
				for suffix, chunk_type, ch, pos, parent_suffix in mg:
					chunk_id = f"{doc_name}::{suffix}"
					all_ids.append(chunk_id)
					all_texts.append(ch)
					qfocus = f"This document discusses: {ch[:200]}"
					all_metadatas.append({
						"chunk_id": chunk_id,
						"doc_name": doc_name,
						"source": info.filename,
						"chunk_type": chunk_type,
						"parent_id": f"{doc_name}::{parent_suffix}",
						"position": pos,
						"doc_stats": json.dumps(stats, ensure_ascii=False),
						"doc_entities": json.dumps(entities, ensure_ascii=False),
						"doc_top_keywords": json.dumps(keywords, ensure_ascii=False),
						"query_focused_text": qfocus,
					})

	# Resolve embeddings with cache - optimized version (runs for both dir and zip cases)
	print(f"Processing {len(all_texts)} texts with optimized embedding...")
	start_time = time.time()

	# Process in larger batches for better efficiency
	embedding_batch_size = cfg.embedding_batch_size
	pbar = tqdm(range(0, len(all_texts), embedding_batch_size), desc="Embedding & caching", unit="batch")

	for start in pbar:
		end = min(start + embedding_batch_size, len(all_texts))
		batch_ids = all_ids[start:end]
		batch_texts = all_texts[start:end]

		model = cfg.gemini_model
		cache_ids = [hash_text(t, model) for t in batch_texts]
		cached = {k: v for k, v in cache.get_many(cache_ids)}
		to_compute: List[Tuple[int, str, str]] = []  # (local_index, cache_id, text)
		batch_embeddings: List[np.ndarray] = [None] * len(batch_texts)  # type: ignore

		for i, (cid, txt) in enumerate(zip(cache_ids, batch_texts)):
			if cid in cached:
				batch_embeddings[i] = cached[cid]
			else:
				to_compute.append((i, cid, txt))

		if to_compute:
			# Use optimized batch embedding
			computed_vecs = embed_texts_gemini_batch([t for (_, _, t) in to_compute], model)
			entries_to_store: List[Tuple[str, str, np.ndarray]] = []
			for (i_local, cid, _), emb in zip(to_compute, computed_vecs):
				vec = np.asarray(emb, dtype=np.float32)
				batch_embeddings[i_local] = vec
				entries_to_store.append((cid, model, vec))
			cache.put_many(entries_to_store)

		# Append to full list
		for vec in batch_embeddings:
			assert vec is not None
			all_embeddings.append(vec.tolist())

		# Update progress with speed info
		elapsed = time.time() - start_time
		processed = end
		speed = processed / elapsed if elapsed > 0 else 0
		pbar.set_postfix({"speed": f"{speed:.1f} texts/sec"})

	# Upsert into Chroma in batches
	pbar = tqdm(range(0, len(all_texts), cfg.batch_size), desc="Writing to Chroma", unit="batch")
	for start in pbar:
		end = min(start + cfg.batch_size, len(all_texts))
		collection.upsert(
			ids=all_ids[start:end],
			embeddings=all_embeddings[start:end],
			documents=all_texts[start:end],
			metadatas=all_metadatas[start:end],
		)

	# Cleanup
	cache.close()
	
	# Performance summary
	total_time = time.time() - start_time
	total_texts = len(all_texts)
	avg_speed = total_texts / total_time if total_time > 0 else 0
	
	print(f"\n{'='*50}")
	print(f"Ingestion complete!")
	print(f"Total texts processed: {total_texts}")
	print(f"Total time: {total_time:.2f} seconds")
	print(f"Average speed: {avg_speed:.1f} texts/second")
	print(f"Chroma persisted at: {cfg.persist_dir}")
	print(f"{'='*50}")


if __name__ == "__main__":
	main()
