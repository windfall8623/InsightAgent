"""Qdrant Vector Store Adapter for InsightAgent.

Provides:
1. Connection to remote Qdrant (via QDRANT_URL) with automatic fallback to embedded in-memory mode (:memory:).
2. Deterministic BLAKE2b float32 dense embeddings (128-dim) for offline/CI reproducibility.
3. Collection management and payload indexing (e.g. integer index on review_score).
4. Single-stage payload filtering during HNSW graph traversal.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import time
import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from insight_agent.app.rag import tokenize

logger = logging.getLogger(__name__)


class BGEEmbeddingAdapter:
    """Adapter for BAAI BGE-M3 / BGE-Large dense embeddings with remote API & zero-infra fallback."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        dims: int = 128,
        device: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.dims = dims
        self.model_name = (
            model_name or os.environ.get("INSIGHT_EMBEDDING_MODEL") or "BAAI/bge-m3"
        )
        self.device = device or os.environ.get("INSIGHT_DEVICE", "cpu")
        self.url = url or os.environ.get("INSIGHT_EMBEDDING_URL")
        self.api_key = api_key or os.environ.get("INSIGHT_EMBEDDING_API_KEY")
        self._model: Any = None
        self._is_neural = False
        # Counts how many times a configured remote endpoint failed and this
        # adapter silently fell back to the hash embedding. The eval snapshots
        # this to attribute embedding degradation to a specific case.
        self.remote_fallback_count = 0
        # Client-side throttle: the gateway rate-limits bge-m3 per key (HTTP 429
        # under bursts). Spacing calls at least this far apart keeps us under the
        # cap so the 429 never fires. Tunable via env for a looser/tighter key.
        from insight_agent.app.http_util import RateLimiter

        self._rate_limiter = RateLimiter(
            float(os.environ.get("INSIGHT_EMBEDDING_MIN_INTERVAL_S", "0.6"))
        )
        # Cache remote embeddings by text: the eval re-embeds the same schema
        # docs and repeated queries hundreds of times, and every avoided call is
        # one fewer chance to hit the gateway's connection drops.
        self._emb_cache: dict[str, list[float]] = {}

        enable_flag = os.environ.get("INSIGHT_BGE_ENABLED", "0").lower() in ("1", "true", "yes")
        if enable_flag and not self.url:
            try:
                from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-not-found]

                self._model = BGEM3FlagModel(
                    self.model_name, use_fp16=True, device=self.device
                )
                self._is_neural = True
                logger.info(f"BGEEmbeddingAdapter loaded locally: {self.model_name}")
            except Exception as exc:
                logger.warning(
                    f"Could not load FlagEmbedding ({exc}); falling back to deterministic hashing"
                )

    @property
    def is_neural(self) -> bool:
        return self._is_neural or bool(self.url)

    def _call_remote_embedding(self, text: str) -> list[float] | None:
        """Call standard OpenAI/TEI-compatible /v1/embeddings endpoint."""
        from insight_agent.app.http_util import post_json_with_retry

        if getattr(self, "_remote_disabled_until", 0) > time.time():
            return None

        if not self.url:
            return None
        target_url = self.url.rstrip("/")
        if not target_url.endswith("/embeddings"):
            target_url = f"{target_url}/embeddings"

        payload = {"input": text, "model": self.model_name}
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) InsightAgent/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data, exc = post_json_with_retry(
            target_url,
            payload,
            headers,
            timeout=8.0,
            max_attempts=6,
            base_delay=2.0,
            max_delay=60.0,
            rate_limiter=self._rate_limiter,
        )
        if data is not None:
            try:
                embedding = data["data"][0]["embedding"]
                return [float(x) for x in embedding]
            except (KeyError, IndexError, TypeError) as parse_exc:
                logger.warning(f"Remote embedding response malformed ({parse_exc}); falling back")
                return None

        err_str = str(exc)
        if any(c in err_str for c in ("401", "402", "403")):
            self._remote_disabled_until = time.time() + 300
            logger.warning(f"Remote embedding endpoint billing/auth issue ({exc}); circuit breaker enabled for 300s")
        else:
            logger.warning(f"Remote embedding call failed after retries ({exc}); falling back")
        return None

    def embed(self, text: str, dims: int | None = None) -> list[float]:
        target_dims = dims or self.dims

        # 1. Try remote HTTP embedding endpoint if configured (text-cached)
        if self.url:
            remote_vec = self._emb_cache.get(text)
            if remote_vec is None:
                remote_vec = self._call_remote_embedding(text)
                if remote_vec:
                    self._emb_cache[text] = remote_vec
                else:
                    # Configured remote endpoint but the call failed: this embed()
                    # will degrade to the hash fallback below. Record it so the
                    # eval can re-run every case that touched a degraded embedding.
                    self.remote_fallback_count += 1
            if remote_vec:
                if len(remote_vec) == target_dims:
                    return remote_vec
                sub = remote_vec[:target_dims]
                norm = math.sqrt(sum(v * v for v in sub)) or 1.0
                return [v / norm for v in sub]

        # 2. Try local neural model if loaded
        if self._is_neural and self._model is not None:
            try:
                output = self._model.encode([text], return_dense=True)
                dense_vec = output["dense_vecs"][0]
                vec_list = [float(v) for v in dense_vec]
                if len(vec_list) == target_dims:
                    return vec_list
                sub = vec_list[:target_dims]
                norm = math.sqrt(sum(v * v for v in sub)) or 1.0
                return [v / norm for v in sub]
            except Exception as exc:
                logger.warning(f"Error in local neural embed: {exc}; fallback to hash")

        # 3. Fallback to deterministic hash dense embedding
        return compute_hash_dense_embedding(text, dims=target_dims)


_DEFAULT_BGE_ADAPTER: BGEEmbeddingAdapter | None = None


def get_default_bge_adapter() -> BGEEmbeddingAdapter:
    global _DEFAULT_BGE_ADAPTER
    if _DEFAULT_BGE_ADAPTER is None:
        _DEFAULT_BGE_ADAPTER = BGEEmbeddingAdapter()
    return _DEFAULT_BGE_ADAPTER


def compute_hash_dense_embedding(text: str, dims: int = 128) -> list[float]:
    """Compute deterministic normalized dense float vector using BLAKE2b hashing."""
    vec = [0.0] * dims
    tokens = tokenize(text)
    if not tokens:
        vec[0] = 1.0
        return vec

    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest, "big") % dims
        vec[bucket] += 1.0

    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        vec[0] = 1.0
        return vec
    return [v / norm for v in vec]


def compute_dense_embedding(text: str, dims: int = 128) -> list[float]:
    """Compute normalized dense vector using BGE adapter with automatic zero-infra fallback."""
    return get_default_bge_adapter().embed(text, dims=dims)


class QdrantVectorStore:
    """Production-ready Qdrant adapter with embedded fallback."""

    def __init__(
        self,
        collection_name: str,
        *,
        dims: int = 128,
        url: str | None = None,
        api_key: str | None = None,
        force_memory: bool = False,
    ) -> None:
        self.collection_name = collection_name
        self.dims = dims
        self.is_memory = False

        target_url = url or os.environ.get("QDRANT_URL")
        target_api_key = api_key or os.environ.get("QDRANT_API_KEY")

        if force_memory or not target_url:
            self.client = QdrantClient(":memory:")
            self.is_memory = True
            logger.info("Initialized Qdrant in embedded :memory: mode for collection '%s'.", collection_name)
        else:
            try:
                self.client = QdrantClient(
                    url=target_url,
                    api_key=target_api_key,
                    timeout=5,
                    check_compatibility=False,
                )
                # Quick healthcheck ping
                self.client.get_collections()
                logger.info("Connected to remote Qdrant at %s for collection '%s'.", target_url, collection_name)
            except Exception as e:
                logger.warning(
                    "Failed to connect to remote Qdrant at %s (%s). Falling back to in-memory mode.",
                    target_url,
                    e,
                )
                self.client = QdrantClient(":memory:")
                self.is_memory = True

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if not present."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dims,
                    distance=models.Distance.COSINE,
                ),
            )
            # Create payload index on server Qdrant for review_score to accelerate single-stage pre-filtering
            if not self.is_memory:
                import contextlib

                with contextlib.suppress(Exception):
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="review_score",
                        field_schema=models.PayloadSchemaType.INTEGER,
                    )

    def _to_point_id(self, doc_id: str) -> str:
        """Convert arbitrary string document ID to deterministic UUID string."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))

    def upsert(
        self,
        doc_id: str,
        text_or_vector: str | list[float],
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Upsert a single vector point with payload."""
        if isinstance(text_or_vector, str):
            vector = compute_dense_embedding(text_or_vector, dims=self.dims)
        else:
            vector = text_or_vector

        data_payload = dict(payload or {})
        data_payload["_doc_id"] = doc_id

        point = models.PointStruct(
            id=self._to_point_id(doc_id),
            vector=vector,
            payload=data_payload,
        )
        self.client.upsert(collection_name=self.collection_name, points=[point])

    def upsert_batch(
        self,
        items: list[tuple[str, str | list[float], dict[str, Any] | None]],
    ) -> None:
        """Batch upsert vector points."""
        points: list[models.PointStruct] = []
        for doc_id, text_or_vec, payload in items:
            if isinstance(text_or_vec, str):
                vec = compute_dense_embedding(text_or_vec, dims=self.dims)
            else:
                vec = text_or_vec
            data_payload = dict(payload or {})
            data_payload["_doc_id"] = doc_id
            points.append(
                models.PointStruct(
                    id=self._to_point_id(doc_id),
                    vector=vec,
                    payload=data_payload,
                )
            )

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        query: str | list[float],
        *,
        top_k: int = 5,
        query_filter: models.Filter | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Search similar vectors with optional single-stage payload filter.

        Returns:
            List of (doc_id, cosine_score, payload) tuples.
        """
        if isinstance(query, str):
            query_vector = compute_dense_embedding(query, dims=self.dims)
        else:
            query_vector = query

        res = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
        )

        results: list[tuple[str, float, dict[str, Any]]] = []
        for pt in res.points:
            score = float(pt.score) if pt.score is not None else 0.0
            payload = dict(pt.payload or {})
            doc_id = str(payload.get("_doc_id", str(pt.id)))
            results.append((doc_id, score, payload))
        return results

    def count(self) -> int:
        """Count total vectors in collection."""
        res = self.client.count(collection_name=self.collection_name)
        return int(res.count)

    def close(self) -> None:
        """Close client resources."""
        self.client.close()
