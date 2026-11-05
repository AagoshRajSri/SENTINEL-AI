# src/cache.py
"""
Semantic Caching Layer for Sentinel-AI with Redis.
Intercepts identical and semantically equivalent customer inquiries (>0.95 cosine similarity)
before LLM inference, slashing latency from seconds to single-digit milliseconds.
"""

import hashlib
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv()

from src.pipeline import PipelineResponse, run_full_pipeline
from src.retriever import _get_model

logger = logging.getLogger("SentinelSemanticCache")

REDIS_URL_ENV = os.getenv("REDIS_URL")
DEFAULT_PORTS = [6380, 6379]
CACHE_KEY_PREFIX = "sentinel:semcache"
SIMILARITY_THRESHOLD_DEFAULT = 0.95


class SemanticCache:
    """Enterprise Redis-backed semantic vector cache with cosine similarity matching."""

    def __init__(
        self,
        redis_url: Optional[str] = REDIS_URL_ENV,
        similarity_threshold: float = SIMILARITY_THRESHOLD_DEFAULT,
        ttl_seconds: int = 86400,  # 24 hour TTL
    ):
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.redis_client = None
        self._connected = False
        self._local_in_memory_fallback: Dict[str, Dict[str, Any]] = {}

        # Connection telemetry
        self.hits = 0
        self.misses = 0

        self._init_connection(redis_url)

    def _init_connection(self, redis_url: Optional[str]):
        """Initializes connection to Redis, attempting configured URL or standard ports."""
        try:
            import redis

            if redis_url:
                client = redis.Redis.from_url(redis_url, decode_responses=False)
                client.ping()
                self.redis_client = client
                self._connected = True
                logger.info(f"Connected to Redis at: {redis_url}")
                return

            # Auto-detect local ports
            for port in DEFAULT_PORTS:
                try:
                    candidate_url = f"redis://localhost:{port}"
                    client = redis.Redis.from_url(candidate_url, decode_responses=False)
                    client.ping()
                    self.redis_client = client
                    self._connected = True
                    logger.info(f"Auto-detected and connected to Redis at: {candidate_url}")
                    return
                except Exception:
                    continue

            logger.warning("No live Redis broker detected. Operating in in-memory fallback mode.")
        except ImportError:
            logger.warning("redis python library not available. Operating in in-memory fallback mode.")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _encode(self, text: str) -> np.ndarray:
        """Encodes query string into a normalized 384-dimensional dense vector."""
        model = _get_model()
        vec = model.encode([text.strip()], normalize_embeddings=True)[0]
        return vec.astype(np.float32)

    def lookup(
        self,
        query: str,
        threshold: Optional[float] = None
    ) -> Optional[Tuple[PipelineResponse, float, str]]:
        """
        Searches semantic cache for query.
        Returns (PipelineResponse, similarity_score, matched_cached_query) if >= threshold, else None.
        """
        if not query or not query.strip():
            return None

        clean_query = query.strip()
        min_sim = threshold if threshold is not None else self.similarity_threshold
        query_hash = hashlib.sha256(clean_query.lower().encode("utf-8")).hexdigest()

        # Tier 0A: Instant O(1) Exact Hash Match
        if self._connected and self.redis_client:
            try:
                exact_key = f"{CACHE_KEY_PREFIX}:exact:{query_hash}"
                entry_id_bytes = self.redis_client.get(exact_key)
                if entry_id_bytes:
                    entry_id = entry_id_bytes.decode("utf-8")
                    data = self.redis_client.hgetall(f"{CACHE_KEY_PREFIX}:entry:{entry_id}")
                    if data and b"response" in data:
                        resp_json = json.loads(data[b"response"].decode("utf-8"))
                        cached_query = data.get(b"query", b"").decode("utf-8")
                        self.hits += 1
                        self.redis_client.hincrby(f"{CACHE_KEY_PREFIX}:entry:{entry_id}", "hits", 1)
                        return (PipelineResponse(**resp_json), 1.0, cached_query)
            except Exception as e:
                logger.error(f"Error during Redis exact lookup: {e}")

        # Tier 0B: Semantic Vector Cosine Similarity Search
        query_vec = self._encode(clean_query)

        if self._connected and self.redis_client:
            try:
                # Fetch all cached entry IDs
                index_key = f"{CACHE_KEY_PREFIX}:index"
                entry_ids = self.redis_client.smembers(index_key)
                if not entry_ids:
                    self.misses += 1
                    return None

                best_id = None
                best_sim = -1.0
                best_query = ""

                # Batch fetch vector representations
                pipe = self.redis_client.pipeline()
                entry_id_list = list(entry_ids)
                for eid in entry_id_list:
                    pipe.hget(f"{CACHE_KEY_PREFIX}:entry:{eid.decode('utf-8')}", "vector")
                    pipe.hget(f"{CACHE_KEY_PREFIX}:entry:{eid.decode('utf-8')}", "query")
                results = pipe.execute()

                vectors = []
                queries = []
                valid_ids = []

                for i, eid in enumerate(entry_id_list):
                    vec_bytes = results[i * 2]
                    q_bytes = results[i * 2 + 1]
                    if vec_bytes:
                        vec = np.frombuffer(vec_bytes, dtype=np.float32)
                        vectors.append(vec)
                        queries.append(q_bytes.decode("utf-8") if q_bytes else "")
                        valid_ids.append(eid.decode("utf-8"))

                if not vectors:
                    self.misses += 1
                    return None

                # Compute cosine similarities via dot product (vectors are L2-normalized)
                matrix = np.vstack(vectors)
                sims = np.dot(matrix, query_vec)
                max_idx = int(np.argmax(sims))
                best_sim = float(sims[max_idx])

                if best_sim >= min_sim:
                    best_id = valid_ids[max_idx]
                    best_query = queries[max_idx]
                    resp_bytes = self.redis_client.hget(f"{CACHE_KEY_PREFIX}:entry:{best_id}", "response")
                    if resp_bytes:
                        resp_json = json.loads(resp_bytes.decode("utf-8"))
                        self.hits += 1
                        self.redis_client.hincrby(f"{CACHE_KEY_PREFIX}:entry:{best_id}", "hits", 1)
                        return (PipelineResponse(**resp_json), best_sim, best_query)

            except Exception as e:
                logger.error(f"Error during Redis semantic vector search: {e}")

        # In-Memory Fallback Evaluation
        if not self._connected:
            best_sim = -1.0
            best_entry = None
            for eid, entry in self._local_in_memory_fallback.items():
                sim = float(np.dot(entry["vector"], query_vec))
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

            if best_sim >= min_sim and best_entry:
                self.hits += 1
                return (PipelineResponse(**best_entry["response"]), best_sim, best_entry["query"])

        self.misses += 1
        return None

    def store(
        self,
        query: str,
        response: PipelineResponse,
        vector: Optional[np.ndarray] = None
    ) -> str:
        """Stores query, embedding vector, and PipelineResponse in Redis."""
        clean_query = query.strip()
        query_hash = hashlib.sha256(clean_query.lower().encode("utf-8")).hexdigest()
        entry_id = query_hash[:16]

        if vector is None:
            vector = self._encode(clean_query)

        resp_dict = response.model_dump()

        if self._connected and self.redis_client:
            try:
                entry_key = f"{CACHE_KEY_PREFIX}:entry:{entry_id}"
                exact_key = f"{CACHE_KEY_PREFIX}:exact:{query_hash}"
                index_key = f"{CACHE_KEY_PREFIX}:index"

                pipe = self.redis_client.pipeline()
                pipe.hset(entry_key, mapping={
                    "query": clean_query.encode("utf-8"),
                    "vector": vector.tobytes(),
                    "response": json.dumps(resp_dict, ensure_ascii=False).encode("utf-8"),
                    "created_at": str(time.time()).encode("utf-8"),
                    "hits": b"0"
                })
                pipe.set(exact_key, entry_id.encode("utf-8"))
                pipe.sadd(index_key, entry_id.encode("utf-8"))

                if self.ttl_seconds > 0:
                    pipe.expire(entry_key, self.ttl_seconds)
                    pipe.expire(exact_key, self.ttl_seconds)

                pipe.execute()
                logger.debug(f"Stored query [{clean_query[:30]}...] in Redis with ID {entry_id}")
            except Exception as e:
                logger.error(f"Failed to store entry in Redis: {e}")

        # Keep in local fallback mirror as well
        self._local_in_memory_fallback[entry_id] = {
            "query": clean_query,
            "vector": vector,
            "response": resp_dict,
            "created_at": time.time(),
        }

        return entry_id

    def clear(self):
        """Flushes all semantic cache keys from Redis."""
        if self._connected and self.redis_client:
            try:
                pattern = f"{CACHE_KEY_PREFIX}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} semantic cache keys from Redis.")
            except Exception as e:
                logger.error(f"Failed to clear Redis semantic cache: {e}")
        self._local_in_memory_fallback.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Returns cache telemetry and hit rate statistics."""
        total_queries = self.hits + self.misses
        hit_rate = (self.hits / total_queries * 100.0) if total_queries > 0 else 0.0

        cached_count = len(self._local_in_memory_fallback)
        if self._connected and self.redis_client:
            try:
                cached_count = self.redis_client.scard(f"{CACHE_KEY_PREFIX}:index")
            except Exception:
                pass

        return {
            "connected": self._connected,
            "cached_entries": cached_count,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(hit_rate, 2)
        }


# Global Singleton Instance
_global_semantic_cache: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    """Returns the process-wide SemanticCache singleton."""
    global _global_semantic_cache
    if _global_semantic_cache is None:
        _global_semantic_cache = SemanticCache()
    return _global_semantic_cache


def triage_with_semantic_cache(
    text: str,
    threshold: float = SIMILARITY_THRESHOLD_DEFAULT,
    cache: Optional[SemanticCache] = None
) -> Tuple[PipelineResponse, bool, float, float]:
    """
    Tier 0 Semantic Caching Interceptor.
    Checks cache first:
      - If hit: returns (PipelineResponse, True, similarity, latency_ms) in milliseconds!
      - If miss: runs run_full_pipeline(), stores in Redis, returns (PipelineResponse, False, 0.0, latency_ms).
    """
    c = cache or get_semantic_cache()
    t0 = time.perf_counter()

    # Step 1: Cache Lookup
    cached_match = c.lookup(text, threshold=threshold)
    if cached_match is not None:
        res, similarity, _ = cached_match
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return (res, True, similarity, latency_ms)

    # Step 2: Cache Miss -> Full Two-Pass Pipeline Execution
    res = run_full_pipeline(text)
    c.store(text, res)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return (res, False, 0.0, latency_ms)

