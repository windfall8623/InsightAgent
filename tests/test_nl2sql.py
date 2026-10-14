"""Unit tests for the question -> plan -> SQL translator."""

from __future__ import annotations

from insight_agent.app.nl2sql import compile_sql, parse_question


class TestMetricDetection:
    def test_sales_amount(self) -> None:
        assert parse_question("总销售额是多少").metric == "amount"

    def test_order_count_variants(self) -> None:
        for q in ("订单数是多少", "订单总数", "有多少订单", "订单量多少", "各城市订单"):
            assert parse_question(q).metric == "order_count", q

    def test_refund_rate_beats_generic_amount(self) -> None:
        # "退款金额" must not fall through to the generic amount metric.
        assert parse_question("退款金额是多少").metric == "refund_amount"
        assert parse_question("退款率是多少").metric == "refund_rate"

    def test_payment_rate_is_not_read_as_a_status_filter(self) -> None:
        # Regression: "支付" used to imply status='paid', making every payment
        # success rate trivially 100%.
        plan = parse_question("支付成功率是多少")
        assert plan.metric == "pay_success_rate"
        assert not any("status = 'paid'" in f for f in plan.filters)

    def test_unknown_question_is_not_guessed(self) -> None:
        assert not parse_question("今天天气怎么样").understood


class TestFilterExtraction:
    def test_city_filter(self) -> None:
        assert "u.city = 'Beijing'" in parse_question("北京有多少订单").filters

    def test_english_token_next_to_cjk_is_found(self) -> None:
        # \b fails against CJK because Chinese characters are word characters.
        assert "u.member_level = 'gold'" in parse_question("gold会员的订单数").filters

    def test_pay_method_and_status(self) -> None:
        filters = parse_question("支付宝支付成功了多少笔").filters
        assert "p.pay_method = 'alipay'" in filters
        assert "p.pay_status = 'success'" in filters

    def test_total_cue_suppresses_the_dimension(self) -> None:
        # "商品总库存" contains 商品 but asks for one number, not a breakdown.
        assert parse_question("商品总库存是多少").dimension is None

    def test_grouping_cue_keeps_the_dimension(self) -> None:
        assert parse_question("各商品库存").dimension == "product"


class TestDimensionConflict:
    def test_filter_pinned_dimension_is_dropped(self) -> None:
        plan = parse_question("小程序渠道的订单数")
        assert plan.dimension is None
        assert "o.channel = 'mini_program'" in plan.filters

    def test_enumerating_statuses_keeps_the_dimension(self) -> None:
        plan = parse_question("支付成功率和支付失败率分别是多少")
        assert plan.dimension == "pay_status"
        assert not any(f.startswith("p.pay_status") for f in plan.filters)


class TestDateRanges:
    def test_iso_range(self) -> None:
        plan = parse_question("2024-06-01到2024-06-30销售额")
        assert (plan.start, plan.end) == ("2024-06-01", "2024-06-30")

    def test_chinese_month(self) -> None:
        assert (parse_question("2024年3月的订单数").start) == "2024-03-01"

    def test_chinese_day(self) -> None:
        plan = parse_question("2024年3月15日的订单数")
        assert (plan.start, plan.end) == ("2024-03-15", "2024-03-15")

    def test_relative_window_is_anchored_to_as_of(self) -> None:
        plan = parse_question("最近7天的销售额")
        assert (plan.start, plan.end) == ("2024-12-25", "2024-12-31")


class TestSQLGeneration:
    def test_dimension_pins_a_join(self) -> None:
        plan = parse_question("北京有多少订单")
        assert "JOIN users" in compile_sql(plan)

    def test_item_level_sum_avoids_fanout(self) -> None:
        # Summing o.total_amount through order_items repeats each order once per
        # line item; measured 148.8M against a true total of 49.4M.
        sql = compile_sql(parse_question("各品类的销售额"))
        assert "oi.unit_price * oi.quantity" in sql
        assert "SUM(o.total_amount)" not in sql

    def test_inventory_does_not_start_from_orders(self) -> None:
        sql = compile_sql(parse_question("商品总库存是多少"))
        assert "FROM products" in sql
        assert "FROM orders" not in sql

    def test_order_by_uses_the_real_alias(self) -> None:
        sql = compile_sql(parse_question("各品类的退款率"))
        assert "ORDER BY refund_rate_pct" in sql

    def test_unparsed_question_cannot_compile(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            compile_sql(parse_question("今天天气怎么样"))

    def test_limit_is_always_present(self) -> None:
        assert "LIMIT" in compile_sql(parse_question("总销售额是多少"))
