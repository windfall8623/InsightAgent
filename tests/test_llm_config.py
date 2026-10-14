"""Tests for LLM configuration and the planner abstraction.

Every test here runs **without an API key**. That is the point: the project must
be fully testable offline, so configuration selection, key handling, fallback
behaviour and SQL extraction are all verified without a network call.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from insight_agent.app.planner import (
    LLMPlanner,
    RulesPlanner,
    _extract_sql,
    build_planner,
    schema_hint,
)
from insight_agent.config import (
    DEFAULT_MODELS,
    PROVIDERS,
    LLMConfig,
    build_client,
    load_config,
    load_dotenv_file,
)
from insight_agent.core.llm import Message, Turn


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from a known-empty environment and no local .env.

    Without the second part these tests would read the developer's real .env,
    select a live provider, and make network calls — turning a unit test into
    an integration test that fails whenever the key is absent.
    """

    import insight_agent.config as config_module

    monkeypatch.setattr(config_module, "DEFAULT_DOTENV", Path("nonexistent-for-tests/.env"))

    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "INSIGHT_API_KEY",
        "API_KEY",
        "INSIGHT_PROVIDER",
        "INSIGHT_MODEL",
        "INSIGHT_BASE_URL",
        "INSIGHT_ANTHROPIC_BASE_URL",
        "INSIGHT_OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "OPENAI_BASE_URL",
        "INSIGHT_MAX_TOKENS",
        "INSIGHT_TEMPERATURE",
    ):
        monkeypatch.delenv(var, raising=False)


class TestConfigDefaults:
    def test_defaults_to_auto_with_no_key(self, store_dir: Path) -> None:
        """Auto by default, but a missing key degrades to rules rather than failing."""

        config = load_config(dotenv=store_dir / "absent.env")
        assert config.provider == "auto"
        assert config.wants_live
        assert not config.is_live
        assert config.missing_reason() is not None
        assert isinstance(build_planner(config), RulesPlanner)

    def test_config_is_serialisable_and_hides_keys(self) -> None:
        config = LLMConfig(provider="openai", openai_api_key="sk-secret")
        payload = config.to_payload()
        json.dumps(payload)
        assert "sk-secret" not in json.dumps(payload)
        assert payload["has_openai_key"] is True

    def test_unknown_provider_falls_back_to_rules(self, store_dir: Path) -> None:
        path = store_dir / ".env"
        path.write_text("INSIGHT_PROVIDER=nonsense\n", encoding="utf-8")
        assert load_config(dotenv=path).provider == "rules"

    def test_explicit_provider_overrides_the_file(self, store_dir: Path) -> None:
        path = store_dir / ".env"
        path.write_text("INSIGHT_PROVIDER=anthropic\n", encoding="utf-8")
        assert load_config(dotenv=path, provider="rules").provider == "rules"

    def test_known_providers(self) -> None:
        assert set(PROVIDERS) == {"rules", "auto", "anthropic", "openai"}
        assert set(DEFAULT_MODELS) == {"anthropic", "openai"}


class TestDotenvLoading:
    def test_reads_keys_and_settings(self, store_dir: Path) -> None:
        path = store_dir / ".env"
        path.write_text(
            "INSIGHT_PROVIDER=openai\n"
            "OPENAI_API_KEY=sk-abc123\n"
            "INSIGHT_MODEL=my-local-model\n"
            "INSIGHT_BASE_URL=http://localhost:11434/v1\n"
            "INSIGHT_MAX_TOKENS=512\n",
            encoding="utf-8",
        )
        config = load_config(dotenv=path)
        assert config.provider == "openai"
        assert config.openai_api_key == "sk-abc123"
        assert config.model == "my-local-model"
        assert config.base_url == "http://localhost:11434/v1"
        assert config.max_tokens == 512

    def test_missing_file_is_not_an_error(self, store_dir: Path) -> None:
        assert load_dotenv_file(store_dir / "nope.env") is None

    def test_existing_environment_wins_by_default(
        self, store_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        path = store_dir / ".env"
        path.write_text("OPENAI_API_KEY=from-file\n", encoding="utf-8")
        assert load_config(dotenv=path).openai_api_key == "from-env"

    def test_comments_and_blank_lines_are_skipped(self, store_dir: Path) -> None:
        path = store_dir / ".env"
        path.write_text("# a comment\n\nOPENAI_API_KEY=k\n\n", encoding="utf-8")
        assert load_config(dotenv=path).openai_api_key == "k"

    def test_quotes_are_stripped(self, store_dir: Path) -> None:
        path = store_dir / ".env"
        path.write_text("OPENAI_API_KEY='quoted'\n", encoding="utf-8")
        assert load_config(dotenv=path).openai_api_key == "quoted"


class TestMissingKeys:
    @pytest.mark.parametrize("provider", ["anthropic", "openai"])
    def test_live_provider_without_a_key_is_reported(self, provider: str) -> None:
        config = LLMConfig(provider=provider)
        assert config.missing_reason() is not None
        assert "not set" in config.missing_reason()

    def test_api_key_selects_the_right_field(self) -> None:
        assert LLMConfig(provider="anthropic", anthropic_api_key="a").api_key == "a"
        assert LLMConfig(provider="openai", openai_api_key="o").api_key == "o"

    def test_build_client_refuses_without_a_key(self) -> None:
        with pytest.raises(RuntimeError, match="not set"):
            build_client(LLMConfig(provider="openai"))

    def test_build_client_has_no_rules_client(self) -> None:
        with pytest.raises(ValueError, match="no client"):
            build_client(LLMConfig(provider="rules"))


class TestPlannerSelection:
    def test_rules_is_the_default(self) -> None:
        assert isinstance(build_planner(LLMConfig()), RulesPlanner)

    def test_missing_key_degrades_to_rules(self) -> None:
        """A key is an upgrade, never a requirement."""

        planner = build_planner(LLMConfig(provider="anthropic"))
        assert isinstance(planner, RulesPlanner)

    def test_configured_key_selects_the_llm_planner(self) -> None:
        config = LLMConfig(provider="anthropic", anthropic_api_key="k")
        planner = build_planner(config, client=object())  # type: ignore[arg-type]
        assert isinstance(planner, LLMPlanner)

    def test_planner_names_differ(self) -> None:
        assert RulesPlanner().name == "rules"
        assert LLMPlanner(object()).name == "llm"  # type: ignore[arg-type]


class TestRulesPlanner:
    async def test_produces_sql_for_a_known_question(self) -> None:
        result = await RulesPlanner().plan("总销售额是多少")
        assert result.understood
        assert result.sql.upper().startswith("SELECT")
        assert result.model in ("rules", "rules-cn")

    async def test_reports_an_unrecognised_question(self) -> None:
        result = await RulesPlanner().plan("今天天气怎么样")
        assert not result.understood
        assert result.notes


class TestSqlExtraction:
    def test_plain_select(self) -> None:
        assert _extract_sql("SELECT 1") == "SELECT 1"

    def test_fenced_block(self) -> None:
        assert _extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"

    def test_fenced_without_language(self) -> None:
        assert _extract_sql("```\nSELECT 2\n```") == "SELECT 2"

    def test_leading_prose(self) -> None:
        assert _extract_sql("Sure! SELECT 3") == "SELECT 3"

    def test_trailing_semicolon_is_kept(self) -> None:
        assert _extract_sql("SELECT 4;").startswith("SELECT 4")

    def test_prose_only_yields_nothing(self) -> None:
        assert _extract_sql("I cannot answer that.") == ""

    def test_empty_yields_nothing(self) -> None:
        assert _extract_sql("") == ""

    def test_ddl_is_not_returned_as_a_select(self) -> None:
        assert _extract_sql("DROP TABLE orders") == ""

    def test_chinese_prose_mentioning_select_is_not_extracted(self) -> None:
        assert _extract_sql("SELECT 语句。") == ""
        assert _extract_sql("请提供有效的 SELECT 语句。") == ""
        assert _extract_sql("你好！你可以输入诸如 SELECT 查询。") == ""
        # Valid SQL with Chinese string literal should still be accepted
        assert _extract_sql("SELECT count(*) FROM orders WHERE category = '生鲜'") == "SELECT count(*) FROM orders WHERE category = '生鲜'"


class _StubLLM:
    """Stands in for a provider client without any network call."""

    model = "stub-model"

    def __init__(self, text: str) -> None:
        self._text = text
        self.seen: list[Sequence[Message]] = []

    async def complete(self, messages: Sequence[Message], tools: Sequence[dict]) -> Turn:
        self.seen.append(list(messages))
        return Turn(text=self._text)


class TestLLMPlanner:
    async def test_extracts_sql_from_a_fenced_response(self) -> None:
        planner = LLMPlanner(_StubLLM("```sql\nSELECT COUNT(*) FROM orders\n```"))  # type: ignore[arg-type]
        result = await planner.plan("有多少订单")
        assert result.sql == "SELECT COUNT(*) FROM orders"
        assert result.understood
        assert result.model == "stub-model"

    async def test_receives_the_schema_in_the_system_prompt(self) -> None:
        client = _StubLLM("SELECT 1")
        await LLMPlanner(client).plan("q", schema_hint="orders(id INT)")  # type: ignore[arg-type]
        system = client.seen[0][0].content
        assert "orders(id INT)" in system

    async def test_records_a_note_when_no_sql_is_produced(self) -> None:
        result = await LLMPlanner(_StubLLM("I don't know")).plan("q")  # type: ignore[arg-type]
        assert not result.understood
        assert result.notes

    async def test_system_prompt_forbids_writes(self) -> None:
        """The prompt states the same rule the tool policy enforces anyway."""

        client = _StubLLM("SELECT 1")
        await LLMPlanner(client).plan("q")  # type: ignore[arg-type]
        assert "SELECT" in client.seen[0][0].content


class TestSchemaHint:
    def test_lists_tables_and_columns(self, conn) -> None:
        hint = schema_hint(conn)
        assert "orders" in hint
        assert "total_amount" in hint

    def test_is_deterministic(self, conn) -> None:
        assert schema_hint(conn) == schema_hint(conn)


class TestCliConfigSurface:
    def test_show_config_prints_json_without_keys(self, capsys) -> None:
        from insight_agent.cli import main

        assert main(["--show-config"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["provider"] == "auto"
        assert payload["live"] is False

    def test_missing_key_warns_but_still_answers(self, capsys) -> None:
        from insight_agent.cli import main

        assert main(["--provider", "anthropic", "总销售额"]) == 0
        out = capsys.readouterr().out
        # Assert the shape of the answer rather than a literal amount: the
        # sandbox warehouse is regenerated with injected anomalies, so any
        # hard-coded total goes stale the moment the fixtures change.
        assert "total_sales" in out
        assert any(len(t) >= 6 for t in re.findall(r"\d[\d,]*\.\d+", out)), (
            f"expected a sales figure in the output, got: {out}"
        )

    def test_rejects_an_unknown_provider(self) -> None:
        from insight_agent.cli import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(["--provider", "bogus", "q"])


class TestEvaluationHonoursThePlanner:
    """The harness must actually drive the configured planner.

    A regression here is silent: the report would say "rules" while every
    number came from somewhere else, which is the exact confusion this project
    exists to avoid.
    """

    def test_runner_accepts_a_planner_and_scores_it(self, conn) -> None:
        import asyncio

        from insight_agent.app.orchestrator import RunConfig
        from insight_agent.eval.runner import Variant, run_variant

        class Stub:
            model = "stub"

            def __init__(self) -> None:
                self.calls = 0

            async def complete(self, messages: Sequence[Message], tools: Sequence[dict]) -> Turn:
                self.calls += 1
                return Turn(text="```sql\nSELECT COUNT(*) AS n FROM orders\n```")

        stub = Stub()
        from insight_agent.eval.dataset import load_cases

        cases = load_cases()[:40]  # a slice: the full set is a long sweep
        report = asyncio.run(
            run_variant(
                Variant("llm", RunConfig(), "stub"),
                conn=conn,
                planner=LLMPlanner(stub),  # type: ignore[arg-type]
                cases=cases,
            )
        )
        assert report.total == len(cases)
        assert stub.calls > 0, "the runner never invoked the planner"
        # A deliberately wrong stub must not score perfectly; if it did, the
        # harness would be scoring something other than what the planner returned.
        assert report.task_success_rate < 1.0

    def test_planner_may_return_no_structured_plan(self, conn) -> None:
        """An LLM emits SQL, not a QueryPlan; the orchestrator must cope.

        The question is deliberately outside the deterministic compiler's
        modelled surface so the router keeps it on the live planner -- the
        compiler would otherwise handle it and the assertion below would be
        testing the wrong path.
        """

        import asyncio

        from insight_agent.app.orchestrator import InsightOrchestrator, InsightRequest

        class Stub:
            model = "stub"

            async def complete(self, messages: Sequence[Message], tools: Sequence[dict]) -> Turn:
                return Turn(text="```sql\nSELECT 1 AS n FROM orders\n```")

        orchestrator = InsightOrchestrator(
            conn,
            planner=LLMPlanner(Stub()),  # type: ignore[arg-type]
        )
        question = "用户留存和复购的关联性分析"
        assert not orchestrator._rules_understands(question)
        result = asyncio.run(orchestrator.run(InsightRequest(question=question, user_id="u")))
        assert result.planner == "llm"
        assert not result.blocked
        assert result.sql.startswith("SELECT")


class TestBaseUrl:
    def test_provider_specific_wins_over_shared(self) -> None:
        config = LLMConfig(
            provider="openai",
            base_url="http://shared/v1",
            openai_base_url="http://openai/v1",
            anthropic_base_url="http://anthropic/v1",
        )
        assert config.resolved_base_url == "http://openai/v1"

    def test_shared_is_used_when_no_specific_value(self) -> None:
        config = LLMConfig(provider="anthropic", base_url="http://shared/v1")
        assert config.resolved_base_url == "http://shared/v1"

    def test_blank_specific_falls_through(self) -> None:
        config = LLMConfig(provider="openai", base_url="http://shared/v1", openai_base_url="")
        assert config.resolved_base_url == "http://shared/v1"

    def test_rules_has_no_endpoint(self) -> None:
        assert LLMConfig(provider="rules", base_url="http://x/v1").resolved_base_url == ""

    def test_unprefixed_variable_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Relay services document ANTHROPIC_BASE_URL without our prefix."""

        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://relay.example.com")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        config = load_config(provider="anthropic")
        assert config.anthropic_base_url == "https://relay.example.com"

    def test_prefixed_wins_over_unprefixed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://unprefixed")
        monkeypatch.setenv("INSIGHT_ANTHROPIC_BASE_URL", "https://prefixed")
        config = load_config(provider="anthropic")
        assert config.anthropic_base_url == "https://prefixed"

    def test_both_protocols_configure_independently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
        monkeypatch.setenv("OPENAI_API_KEY", "o")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://relay")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        assert load_config(provider="anthropic").resolved_base_url == "https://relay"
        assert load_config(provider="openai").resolved_base_url == "http://localhost:11434/v1"


class TestCliOverrides:
    def test_base_url_flag_overrides_the_file(self, capsys) -> None:
        import json as _json

        from insight_agent.cli import main

        assert main(["--show-config", "--provider", "openai", "--base-url", "http://cli/v1"]) == 0
        payload = _json.loads(capsys.readouterr().out)
        assert payload["base_url"] == "http://cli/v1"

    def test_model_flag_overrides_the_file(self, capsys) -> None:
        import json as _json

        from insight_agent.cli import main

        assert main(["--show-config", "--provider", "openai", "--model", "cli-model"]) == 0
        assert _json.loads(capsys.readouterr().out)["model"] == "cli-model"


class TestProtocolRouting:
    """The router must pick a protocol from the endpoint, not from the user."""

    def test_auto_is_the_default_provider(self) -> None:
        assert LLMConfig().provider == "auto"

    def test_three_values_are_enough(self, store_dir: Path) -> None:
        """base_url + model + key, nothing else."""
        path = store_dir / ".env"
        path.write_text(
            "INSIGHT_BASE_URL=https://relay.example.com/v1\n"
            "INSIGHT_MODEL=some-model\n"
            "INSIGHT_API_KEY=sk-one\n",
            encoding="utf-8",
        )
        config = load_config(dotenv=path)
        assert config.is_live
        assert config.missing_reason() is None
        assert config.resolved_base_url == "https://relay.example.com/v1"
        assert config.model == "some-model"

    def test_one_generic_key_feeds_both_protocols(self) -> None:
        config = LLMConfig(anthropic_api_key="k", openai_api_key="k")
        assert config.api_key == "k"

    def test_no_key_falls_back_to_rules(self) -> None:
        assert isinstance(build_planner(LLMConfig()), RulesPlanner)

    def test_error_routing_is_limited_to_protocol_shaped_failures(self) -> None:
        from insight_agent.core.providers.auto import looks_like_protocol_error

        class NotFound(Exception):
            status_code = 404

        class Unauthorized(Exception):
            status_code = 401

        assert looks_like_protocol_error(NotFound())
        assert not looks_like_protocol_error(Unauthorized())
        assert not looks_like_protocol_error(RuntimeError("rate limit exceeded"))


class _RecordingClient:
    """A protocol stand-in that succeeds or fails on command."""

    def __init__(self, name: str, *, fail: Exception | None = None) -> None:
        self.model = f"{name}-model"
        self.calls = 0
        self._fail = fail

    async def complete(self, messages: Sequence[Message], tools: Sequence[dict]) -> Turn:
        self.calls += 1
        if self._fail is not None:
            raise self._fail
        return Turn(text="SELECT 1")


class _NotFound(Exception):
    status_code = 404


class TestAutoClient:
    def _client(self, **fails: Exception) -> tuple[Any, dict[str, Any]]:
        from insight_agent.core.providers.auto import AutoClient

        clients = {name: _RecordingClient(name, fail=err) for name, err in fails.items()}
        return AutoClient(clients=clients), clients

    async def test_falls_back_to_the_other_protocol(self) -> None:
        client, clients = self._client(openai=_NotFound(), anthropic=None)
        turn = await client.complete([Message(role="user", content="q")], ())
        assert turn.text == "SELECT 1"
        assert client.protocol == "anthropic"
        assert clients["openai"].calls == 1
        assert clients["anthropic"].calls == 1

    async def test_does_not_retry_a_non_protocol_error(self) -> None:
        client, clients = self._client(openai=RuntimeError("quota exceeded"), anthropic=None)
        with pytest.raises(RuntimeError, match="quota"):
            await client.complete([Message(role="user", content="q")], ())
        assert clients["anthropic"].calls == 0

    async def test_caches_the_working_protocol(self) -> None:
        client, clients = self._client(openai=None, anthropic=None)
        for _ in range(3):
            await client.complete([Message(role="user", content="q")], ())
        assert client.protocol == "openai"
        assert clients["anthropic"].calls == 0

    async def test_exhausting_both_protocols_is_reported_clearly(self) -> None:
        client, _ = self._client(openai=_NotFound(), anthropic=_NotFound())
        with pytest.raises(RuntimeError, match="could not speak"):
            await client.complete([Message(role="user", content="q")], ())
