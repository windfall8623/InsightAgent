"""Unit and integration tests for QdrantVectorStore and Single-Stage Payload Filtering."""

from __future__ import annotations

from qdrant_client import models

from insight_agent.app.qdrant_store import QdrantVectorStore, compute_dense_embedding


def test_dense_embedding_deterministic() -> None:
    """Ensure BLAKE2b dense embedding produces normalized 128-dim vectors consistently."""
    text1 = "olist_order_items price freight"
    vec1 = compute_dense_embedding(text1, dims=128)
    vec2 = compute_dense_embedding(text1, dims=128)

    assert len(vec1) == 128
    assert vec1 == vec2
    norm_sq = sum(x * x for x in vec1)
    assert abs(norm_sq - 1.0) < 1e-4

    # Empty text fallback
    empty_vec = compute_dense_embedding("", dims=128)
    assert len(empty_vec) == 128
    assert empty_vec[0] == 1.0


def test_qdrant_vector_store_in_memory_crud() -> None:
    """Test QdrantVectorStore embedded in-memory lifecycle."""
    store = QdrantVectorStore("test_memory_collection", dims=64, force_memory=True)
    assert store.is_memory is True

    # Upsert single
    store.upsert(
        doc_id="doc_1",
        text_or_vector="pedil o reembolso atraso entrega",
        payload={"review_score": 1, "state": "SP"},
    )

    # Upsert batch
    batch_items = [
        ("doc_2", "produto excelente entrega rapida", {"review_score": 5, "state": "RJ"}),
        ("doc_3", "produto quebrado pessimo atendimento", {"review_score": 1, "state": "RJ"}),
        ("doc_4", "chegou no prazo mas cor errada", {"review_score": 2, "state": "SP"}),
    ]
    store.upsert_batch(batch_items)

    assert store.count() == 4

    # Unfiltered search
    hits = store.search("atraso pessimo", top_k=10)
    assert len(hits) == 4

    # Single-stage payload filtering: only score <= 2
    filter_condition = models.Filter(
        must=[
            models.FieldCondition(
                key="review_score",
                range=models.Range(lte=2),
            )
        ]
    )
    filtered_hits = store.search("entrega", top_k=10, query_filter=filter_condition)
    assert len(filtered_hits) > 0
    for doc_id, _score, payload in filtered_hits:
        assert payload["review_score"] <= 2
        assert doc_id != "doc_2"  # doc_2 has score 5 and MUST be excluded

    store.close()


def test_qdrant_fallback_on_unreachable_url() -> None:
    """Test graceful fallback to in-memory mode when remote Qdrant URL is unreachable."""
    unreachable_url = "http://127.0.0.1:59999"
    store = QdrantVectorStore("test_fallback", dims=32, url=unreachable_url)
    assert store.is_memory is True

    store.upsert("fallback_1", "test item", {"status": "ok"})
    hits = store.search("test", top_k=1)
    assert len(hits) == 1
    assert hits[0][0] == "fallback_1"
    store.close()
