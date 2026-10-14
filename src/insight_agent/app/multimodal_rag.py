"""Multi-Modal Agentic RAG: Cross-retrieval over relational schema and customer reviews.

Features:
1. Indexes Olist 9-table schema metadata (columns, keys, data types).
2. Indexes unstructured customer reviews (filtering by 1-star / 2-star complaints).
3. Hybrid BM25 + BLAKE2b deterministic hashing vector index + RRF fusion.
4. Value-matching and sentiment boosting for root-cause evidence discovery.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from qdrant_client import models

from insight_agent.app.qdrant_store import QdrantVectorStore
from insight_agent.app.rag import BM25Index, HashingVectorStore, SchemaDoc


@dataclass(slots=True)
class ReviewEvidence:
    review_id: str
    order_id: str
    review_score: int
    title: str
    message: str
    score: float = 0.0

    @property
    def text(self) -> str:
        return self.message

    @property
    def metadata(self) -> dict[str, Any]:
        return {"review_score": self.review_score, "order_id": self.order_id}

    def to_evidence(self) -> dict[str, Any]:
        return {
            "kind": "review",
            "review_id": self.review_id,
            "order_id": self.order_id,
            "review_score": self.review_score,
            "snippet": f"[{self.review_score}★] {self.title or ''}: {self.message or ''}".strip(),
            "relevance_score": round(self.score, 4),
        }



from insight_agent.app.schema_vocab import get_all_table_aliases

TABLE_SYNONYMS: dict[str, list[str]] = get_all_table_aliases()


class OlistMultiModalRetriever:
    """Retrieves both structural schema context and unstructured review complaints with Qdrant."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        rrf_k: int = 60,
        qdrant_url: str | None = None,
    ) -> None:
        self.conn = conn
        self.rrf_k = rrf_k

        # Detect dataset
        cursor = self.conn.cursor()
        has_cn = bool(
            cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='cn_orders'").fetchone()
        )
        schema_col = "cn_schema" if has_cn else "olist_schema"
        reviews_col = "cn_reviews" if has_cn else "olist_reviews"

        # Schema leg
        self.schema_bm25 = BM25Index()
        self.schema_vectors = HashingVectorStore()
        self.schema_qdrant = QdrantVectorStore(schema_col, url=qdrant_url)
        self.schema_docs: dict[str, SchemaDoc] = {}

        # Review leg
        self.review_bm25 = BM25Index()
        self.review_vectors = HashingVectorStore()
        self.review_qdrant = QdrantVectorStore(reviews_col, url=qdrant_url)
        self.reviews: dict[str, ReviewEvidence] = {}
        from insight_agent.app.bge_reranker import BGEReranker

        self.reranker = BGEReranker()

        self._index_schema()
        self._index_reviews()

    def _index_schema(self) -> None:
        """Index table names, column names, and domain relationships."""
        cursor = self.conn.cursor()
        tables = [
            r[0]
            for r in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        for tbl in tables:
            cols = [c[1] for c in cursor.execute(f'PRAGMA table_info("{tbl}")')]
            syns = TABLE_SYNONYMS.get(tbl, [])
            doc_id = f"table:{tbl}"
            body = f"Table {tbl} with columns: {', '.join(cols)}. Synonyms: {', '.join(syns)}"
            doc = SchemaDoc(
                id=doc_id,
                kind="table",
                table=tbl,
                title=tbl,
                body=body,
                keywords=cols + [tbl, "olist", "cn_retail"] + syns,
            )
            self.schema_docs[doc_id] = doc
            text = f"{doc.table} {doc.body} {' '.join(doc.keywords)}"
            self.schema_bm25.add(doc_id, text)
            self.schema_vectors.add(doc_id, text)
            from insight_agent.app.qdrant_store import compute_hash_dense_embedding

            hash_vec = compute_hash_dense_embedding(text, dims=self.schema_qdrant.dims)
            self.schema_qdrant.upsert(
                doc_id=doc_id,
                text_or_vector=hash_vec,
                payload={"table": tbl, "kind": doc.kind, "keywords": doc.keywords},
            )

    def _index_reviews(self) -> None:
        """Index low-score customer review complaints (score <= 3)."""
        cursor = self.conn.cursor()
        has_cn_reviews = bool(
            cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='cn_reviews'").fetchone()
        )
        if has_cn_reviews:
            query = (
                "SELECT review_id, order_id, rating, tags, comment_text "
                "FROM cn_reviews WHERE comment_text IS NOT NULL AND rating <= 3 "
                "LIMIT 1000"
            )
        else:
            query = (
                "SELECT review_id, order_id, review_score, review_comment_title, review_comment_message "
                "FROM olist_order_reviews WHERE review_comment_message IS NOT NULL AND review_score <= 3 "
                "LIMIT 1000"
            )

        try:
            from insight_agent.app.qdrant_store import compute_hash_dense_embedding

            rows = cursor.execute(query).fetchall()
            batch_items: list[tuple[str, Any, dict[str, Any]]] = []
            for rid, oid, score, title, msg in rows:
                ev = ReviewEvidence(
                    review_id=rid,
                    order_id=oid,
                    review_score=score,
                    title=title or "",
                    message=msg or "",
                )
                self.reviews[rid] = ev
                text = f"{title or ''} {msg or ''} score_{score}"
                self.review_bm25.add(rid, text)
                self.review_vectors.add(rid, text)
                # Use local fast deterministic hash embedding for bulk index initialization
                hash_vec = compute_hash_dense_embedding(text, dims=self.review_qdrant.dims)
                batch_items.append(
                    (
                        rid,
                        hash_vec,
                        {
                            "review_id": rid,
                            "order_id": oid,
                            "review_score": score,
                            "title": title or "",
                            "message": msg or "",
                        },
                    )
                )
            if batch_items:
                self.review_qdrant.upsert_batch(batch_items)
        except Exception:
            pass

    def search_schema(self, query: str, top_k: int = 5) -> list[tuple[SchemaDoc, float]]:
        """Retrieve relevant schema tables using RRF."""
        bm25_hits = self.schema_bm25.search(query, top_k * 3)
        qdrant_hits = self.schema_qdrant.search(query, top_k=top_k * 3)
        fused: dict[str, float] = {}

        for rank, (doc_id, _) in enumerate(bm25_hits):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, (doc_id, _, _) in enumerate(qdrant_hits):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        # Fallback if Qdrant returned nothing
        if not qdrant_hits:
            vec_hits = self.schema_vectors.search(query, top_k * 3)
            for rank, (doc_id, _) in enumerate(vec_hits):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        ranked = sorted(fused.items(), key=lambda it: -it[1])
        from insight_agent.app.bge_reranker import RerankCandidate

        candidates: list[RerankCandidate] = []
        for doc_id, score in ranked[: top_k * 2]:
            if doc_id in self.schema_docs:
                doc = self.schema_docs[doc_id]
                candidates.append(
                    RerankCandidate(
                        doc_id=doc_id,
                        text=f"{doc.table} {doc.body} {' '.join(doc.keywords)}",
                        initial_score=score,
                        metadata={"table": doc.table},
                    )
                )

        reranked_cands = self.reranker.rerank(query, candidates, top_k=top_k)
        results = []
        for c in reranked_cands:
            if c.doc_id in self.schema_docs:
                results.append((self.schema_docs[c.doc_id], c.rerank_score))
        return results

    def search_reviews(
        self, query: str, *, top_k: int = 5, max_score: int = 3
    ) -> list[ReviewEvidence]:
        """Hybrid search with Qdrant single-stage payload filter and BM25 RRF fusion."""
        bm25_hits = self.review_bm25.search(query, top_k * 4)

        # Single-stage pre-filtering in Qdrant HNSW graph traversal
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="review_score",
                    range=models.Range(lte=max_score),
                )
            ]
        )
        qdrant_hits = self.review_qdrant.search(
            query,
            top_k=top_k * 4,
            query_filter=qdrant_filter,
        )

        fused: dict[str, float] = {}

        for rank, (rid, _) in enumerate(bm25_hits):
            if rid in self.reviews and self.reviews[rid].review_score <= max_score:
                fused[rid] = fused.get(rid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        for rank, (rid, _, _) in enumerate(qdrant_hits):
            fused[rid] = fused.get(rid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        if not qdrant_hits:
            vec_hits = self.review_vectors.search(query, top_k * 4)
            for rank, (rid, _) in enumerate(vec_hits):
                if rid in self.reviews and self.reviews[rid].review_score <= max_score:
                    fused[rid] = fused.get(rid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        ranked = sorted(fused.items(), key=lambda it: -it[1])
        from insight_agent.app.bge_reranker import RerankCandidate

        candidates: list[RerankCandidate] = []
        for rid, score in ranked[: top_k * 3]:
            if rid in self.reviews:
                ev = self.reviews[rid]
                text = f"{ev.title} {ev.message} score_{ev.review_score}"
                candidates.append(
                    RerankCandidate(
                        doc_id=rid,
                        text=text,
                        initial_score=score,
                        metadata={"order_id": ev.order_id, "score": ev.review_score},
                    )
                )

        reranked_cands = self.reranker.rerank(query, candidates, top_k=top_k)
        results = []
        for c in reranked_cands:
            if c.doc_id in self.reviews:
                item = self.reviews[c.doc_id]
                item.score = c.rerank_score
                results.append(item)
        return results

    def render_schema_context(self, tables: list[str]) -> str:
        """Render precise schema string for planning."""
        cursor = self.conn.cursor()
        lines = []
        for tbl in tables:
            cols = [c[1] for c in cursor.execute(f'PRAGMA table_info("{tbl}")')]
            lines.append(f"{tbl}({', '.join(cols)})")
        return "\n".join(lines)

    def retrieve_schema(self, query: str, top_k: int = 5) -> list[SchemaDoc]:
        """Convenience method returning SchemaDoc items."""
        return [doc for doc, _ in self.search_schema(query, top_k=top_k)]

    def retrieve_reviews(self, query: str, top_k: int = 5) -> list[ReviewEvidence]:
        """Convenience method returning ReviewEvidence items."""
        return self.search_reviews(query, top_k=top_k)


# Generalized and domestic aliases for clean domain naming
MultiModalRetriever = OlistMultiModalRetriever
CNMultiModalRetriever = OlistMultiModalRetriever


