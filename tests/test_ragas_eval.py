"""Unit tests for the Outer-Loop Ragas-Style Evaluation Framework."""

from __future__ import annotations

from insight_agent.eval.outer_loop_ragas import (
    AnswerRelevanceEvaluator,
    NarrativeFaithfulnessEvaluator,
    OuterLoopRagasEvaluator,
    UnstructuredContextEvaluator,
    _extract_numbers,
    _numbers_match,
)


class TestNarrativeFaithfulness:
    def test_extract_numbers_various_formats(self) -> None:
        text = "本月销售额为 3500 万元，退款率为 15.8%，同比增长 2.5 倍，统计日期为 2024-03-01。"
        nums = _extract_numbers(text)
        # Should extract 3500, 0.158 (15.8%), and 2.5, ignoring date parts
        assert 3500.0 in nums
        assert 0.158 in nums
        assert 2.5 in nums

    def test_numbers_match_tolerance_and_percentage(self) -> None:
        assert _numbers_match(0.158, 0.158)
        assert _numbers_match(15.8, 0.158)  # 15.8% vs 0.158
        assert _numbers_match(0.158, 15.8)
        assert not _numbers_match(0.158, 0.88)

    def test_faithfulness_grounded_report(self) -> None:
        evaluator = NarrativeFaithfulnessEvaluator()
        sql_rows = [{"category": "生鲜", "refund_rate": 0.158, "orders_count": 3500}]
        report = "经排查，生鲜品类的退款率为 15.8%，涉及订单总数为 3500 笔。"

        result = evaluator.evaluate(report, sql_rows)
        assert result.score >= 0.95
        assert result.numeric_grounded_ratio == 1.0
        assert len(result.hallucinated_numbers) == 0

    def test_faithfulness_detects_secondary_hallucination(self) -> None:
        evaluator = NarrativeFaithfulnessEvaluator()
        # Ground truth is 12.0%
        sql_rows = [{"category": "生鲜", "refund_rate": 0.12, "orders_count": 100}]
        # Report claims 85.5% (fabricated number)
        report = "经排查，生鲜品类的退款率高达 85.5%，异常暴增。"

        result = evaluator.evaluate(report, sql_rows)
        # 85.5% is not in ground truth
        assert 0.855 in result.hallucinated_numbers
        assert result.numeric_grounded_ratio < 0.5
        assert result.score < 0.6


class TestUnstructuredContextQuality:
    def test_context_precision_rank_discounting(self) -> None:
        evaluator = UnstructuredContextEvaluator()
        query = "顺丰物流延误卡在转运中心"

        # Case 1: Relevant review at Rank 1
        contexts_good = [
            "顺丰快递卡在杭州转运中心三天没动静，严重延误！",
            "衣服尺码偏大，不满意。",
            "包装挺好，好评。",
        ]
        res_good = evaluator.evaluate(query, contexts_good)

        # Case 2: Relevant review at Rank 3
        contexts_bad = [
            "衣服尺码偏大，不满意。",
            "包装挺好，好评。",
            "顺丰快递卡在杭州转运中心三天没动静，严重延误！",
        ]
        res_bad = evaluator.evaluate(query, contexts_bad)

        assert res_good.context_precision > res_bad.context_precision
        assert res_good.top_k_ranks == [1]
        assert res_bad.top_k_ranks == [3]

    def test_context_relevance_noise_ratio(self) -> None:
        evaluator = UnstructuredContextEvaluator()
        query = "商品破损严重漏液"

        contexts = [
            "洗发水包装破损，漏液搞得箱子全湿了",
            "今天天气不错",
        ]
        res = evaluator.evaluate(query, contexts)
        # 1 out of 2 is relevant
        assert res.relevant_chunks == 1
        assert res.total_chunks == 2
        assert res.context_relevance == 0.5


class TestAnswerRelevance:
    def test_evasion_penalty(self) -> None:
        evaluator = AnswerRelevanceEvaluator()
        res = evaluator.evaluate("为什么退款率上升", "抱歉没有数据，我不知道原因。")
        assert res.score <= 0.5
        assert "evasive_or_unhelpful" in res.penalties

    def test_action_and_intent_match(self) -> None:
        evaluator = AnswerRelevanceEvaluator()
        res = evaluator.evaluate(
            "华东地区生鲜退款率异常原因及补救方案",
            "根因分析显示生鲜退款率因包装破损升高。建议立即对受影响客户执行 issue_voucher 优惠券赔偿并优化冷链。",
            remediation_action="issue_voucher",
        )
        assert res.score >= 0.9
        assert res.intent_match is True
        assert res.action_match is True
        assert len(res.penalties) == 0


class TestOuterLoopRagasComposite:
    def test_composite_aggregation(self) -> None:
        ragas = OuterLoopRagasEvaluator()

        res1 = ragas.evaluate_case(
            case_id="case_1",
            query="排查物流延误原因",
            report_text="物流延误率为 25.0%，建议补偿优惠券 issue_voucher。",
            sql_rows=[{"delay_rate": 0.25}],
            retrieved_contexts=["中通快递物流延误严重"],
            remediation_action="issue_voucher",
        )

        res2 = ragas.evaluate_case(
            case_id="case_2",
            query="排查包装破损原因",
            report_text="包装破损导致退款，实际损失高达 99999 元。",  # Hallucinated 99999
            sql_rows=[{"refund_loss": 500}],
            retrieved_contexts=["箱子破了漏液"],
        )

        report = ragas.aggregate([res1, res2])
        assert report.total_cases == 2
        assert 0.0 < report.mean_composite_score <= 1.0
        # Case 2 has a hallucinated number
        assert report.secondary_hallucination_rate == 0.5
