"""Tests for BAAI BGE Two-Stage Reranker and Retrieval Upgrades."""


from insight_agent.app.bge_reranker import (
    HeuristicCrossAttentionReranker,
    RerankCandidate,
)
from insight_agent.app.multimodal_rag import OlistMultiModalRetriever
from insight_agent.app.qdrant_store import (
    BGEEmbeddingAdapter,
    compute_dense_embedding,
)
from insight_agent.app.rag import HybridRetriever, SchemaDoc
from insight_agent.data.olist_db import connect_olist


def test_heuristic_reranker_basic_ranking():
    reranker = HeuristicCrossAttentionReranker()
    query = "里约 订单 延误"
    cands = [
        RerankCandidate(doc_id="doc_irrelevant", text="用户 手机 充值 记录", initial_score=0.5),
        RerankCandidate(
            doc_id="doc_perfect", text="里约热内卢 订单 严重 延误 调查报告", initial_score=0.4
        ),
        RerankCandidate(doc_id="doc_partial", text="圣保罗 订单 统计", initial_score=0.6),
    ]

    results = reranker.rerank(query, cands, top_k=2)
    assert len(results) == 2
    assert results[0].doc_id == "doc_perfect"
    assert results[0].rerank_score > results[1].rerank_score


def test_bge_cross_lingual_portuguese_alignment():
    reranker = HeuristicCrossAttentionReranker()
    # Chinese query about damaged goods
    query = "包装 破损 缺陷 差评"
    cands = [
        RerankCandidate(
            doc_id="rev_delay",
            text="demorou muito para chegar correios atraso",  # delay
            initial_score=0.5,
        ),
        RerankCandidate(
            doc_id="rev_damaged",
            text="o produto veio quebrado e a embalagem danificada pessimo",  # broken/damaged
            initial_score=0.4,
        ),
        RerankCandidate(
            doc_id="rev_neutral",
            text="tudo certo conforme anunciado",
            initial_score=0.5,
        ),
    ]

    results = reranker.rerank(query, cands, top_k=3)
    # The broken/damaged Portuguese review should rank #1 due to cross-lingual alignment
    assert results[0].doc_id == "rev_damaged"
    assert results[0].rerank_score > results[1].rerank_score


def test_bge_adapter_zero_infra_fallback():
    adapter = BGEEmbeddingAdapter()
    vec = adapter.embed("测试中文文本", dims=128)
    assert len(vec) == 128
    assert isinstance(vec[0], float)

    dense_vec = compute_dense_embedding("里约热内卢", dims=64)
    assert len(dense_vec) == 64


def test_hybrid_retriever_two_stage_rerank():
    retriever = HybridRetriever(use_qdrant=False)
    docs = [
        SchemaDoc(
            id="tbl:orders",
            kind="table",
            table="orders",
            title="Orders Table",
            body="Contains order_id, customer_id, order_status, purchase_timestamp",
            keywords=["orders", "订单", "购买", "支付"],
        ),
        SchemaDoc(
            id="tbl:customers",
            kind="table",
            table="customers",
            title="Customers Table",
            body="Customer addresses, state, city Rio de Janeiro",
            keywords=["customers", "客户", "里约", "城市"],
            values=["Rio de Janeiro", "Shanghai"],
        ),
    ]
    retriever.index(docs)

    results = retriever.search("里约的客户订单", top_k=2)
    assert len(results) == 2
    doc_ids = [doc.id for doc, _ in results]
    assert "tbl:customers" in doc_ids or "tbl:orders" in doc_ids


def test_olist_multimodal_retriever_with_reranker():
    conn = connect_olist()
    retriever = OlistMultiModalRetriever(conn)

    # 1. Test schema search
    schema_results = retriever.search_schema("里约 延误 承运商", top_k=3)
    assert len(schema_results) >= 1

    # 2. Test review search with cross-lingual query
    reviews = retriever.search_reviews("商品损坏 破损", top_k=3)
    assert len(reviews) <= 3
    for r in reviews:
        assert r.review_score <= 3
