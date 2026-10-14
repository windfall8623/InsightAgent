"""Tests for the JD corpus and skill-frequency analysis.

The analysis feeds the resume and the selection report, so a bad match or a
corrupted corpus must fail loudly rather than quietly skew the numbers.
"""

from __future__ import annotations

import pytest

from insight_agent.eval.jd_analysis import (
    REQUIRED_SKILLS,
    SKILL_TERMS,
    analyse,
    business_group_breakdown,
    experience_breakdown,
)
from insight_agent.eval.jd_corpus import corpus, corpus_size


class TestCorpus:
    def test_meets_the_twenty_posting_requirement(self) -> None:
        assert corpus_size() >= 20

    def test_titles_are_unique(self) -> None:
        titles = [p.title for p in corpus()]
        assert len(titles) == len(set(titles))

    def test_every_posting_has_provenance(self) -> None:
        for p in corpus():
            assert p.title and p.city and p.business_group and p.work_year
            assert p.family in {"Agent", "大模型应用", "RAG", "智能体", "大模型应用+RAG"}
            assert p.scope, p.title

    def test_breakdowns_sum_to_the_corpus(self) -> None:
        total = corpus_size()
        assert sum(business_group_breakdown().values()) == total
        assert sum(experience_breakdown().values()) == total


class TestFrequencyAnalysis:
    def test_report_shape(self) -> None:
        report = analyse()
        assert report["total_postings"] == corpus_size()
        assert len(report["skills"]) == len(SKILL_TERMS)

    def test_counts_are_monotonically_sorted(self) -> None:
        counts = [s["postings"] for s in analyse()["skills"]]
        assert counts == sorted(counts, reverse=True)

    def test_shares_are_consistent_with_counts(self) -> None:
        report = analyse()
        for stat in report["skills"]:
            assert stat["postings"] <= report["total_postings"]
            # Shares are rounded to 4 decimal places in the payload.
            assert stat["share"] == pytest.approx(
                stat["postings"] / report["total_postings"], abs=1e-4
            )

    def test_most_frequent_skill_is_tool_related_or_product(self) -> None:
        """Anchors the analysis against known corpus content."""

        top = analyse()["skills"][0]["skill"]
        assert top in {"产品规划", "工具调用/Tool use"}

    def test_evaluation_is_a_top_tier_skill(self) -> None:
        """The finding that justifies the project's evaluation emphasis."""

        by_skill = {s["skill"]: s["share"] for s in analyse()["skills"]}
        assert by_skill["评测/Evaluation"] >= 0.30

    def test_a2a_absent_but_mcp_present(self) -> None:
        """Records the finding that A2A is not yet a market requirement."""

        by_skill = {s["skill"]: s["postings"] for s in analyse()["skills"]}
        assert by_skill["A2A"] == 0
        assert by_skill["MCP"] > 0

    def test_required_skills_all_appear_in_the_term_table(self) -> None:
        for skill in REQUIRED_SKILLS:
            assert skill in SKILL_TERMS

    def test_analysis_is_deterministic(self) -> None:
        assert analyse() == analyse()

    def test_empty_corpus_is_handled(self) -> None:
        report = analyse([])
        assert report["total_postings"] == 0
        assert all(s["share"] == 0.0 for s in report["skills"])
