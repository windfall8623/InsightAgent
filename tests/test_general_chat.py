"""Tests for General Chit-Chat (chat_general) intent classification and handling."""

import pytest
from httpx import ASGITransport, AsyncClient

from insight_agent.api import create_app
from insight_agent.app.router import classify_intent


def test_classify_intent_greetings() -> None:
    for greeting in ["你好", "您好", "hello", "hi", "你是谁", "介绍一下自己", "你能做什么", "帮助"]:
        dec = classify_intent(greeting)
        assert dec.intent == "chat_general", f"Expected chat_general for '{greeting}', got '{dec.intent}'"
        assert not dec.is_anomaly


def test_classify_intent_business_queries_not_hijacked() -> None:
    # Business queries should NOT be classified as chat_general
    for q in [
        "统计2024年各月份华东地区的订单总数",
        "查询退款率最高的 Top 5 商品品类",
        "分析华东江浙沪地区订单履约时效大幅延误的原因",
    ]:
        dec = classify_intent(q)
        assert dec.intent != "chat_general", f"'{q}' should not be chat_general"


@pytest.mark.asyncio
async def test_ask_endpoint_general_chat() -> None:
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/ask", json={"question": "你好", "user_id": "test_user"})
        assert resp.status_code == 200
        data = resp.json()
        assert "InsightAgent" in data["answer"]
        assert data["sql"] == ""
        assert data["rows"] == []
        assert data["route_reason"].startswith("chat_general")
