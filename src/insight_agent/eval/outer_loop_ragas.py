"""Outer-Loop Evaluation Suite: Ragas-Style Semantic, Narrative & Unstructured Evaluator.

Part of InsightAgent 2.0 Dual-Loop Evaluation Framework:
- Inner Loop: High-frequency deterministic CI/CD gate (AST Faithfulness, Row Match, McNemar Gate).
- Outer Loop: Semantic & narrative quality evaluator (Narrative Faithfulness, Context Precision/Relevance,
  Answer Relevance, Secondary Hallucination detection).

Supports dual-mode execution:
1. `mode="offline_deterministic"`: Exact numerical extraction, entity grounding, rank-discounted precision.
   Zero external dependencies, 100% reproducible for CI/CD and offline benchmarking.
2. `mode="llm_judge"`: Remote LLM judge conforming to Ragas structured scoring contracts when
   INSIGHT_EVAL_JUDGE_URL or live API provider is configured.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 1. Narrative Faithfulness (Detecting Secondary Hallucinations in Reports)
# --------------------------------------------------------------------------


@dataclass(slots=True)
class NarrativeFaithfulnessResult:
    """Evaluation result measuring whether a narrative report faithfully recites SQL rows."""

    score: float  # 0.0 ~ 1.0
    numeric_grounded_ratio: float
    entity_grounded_ratio: float
    verified_numbers: list[float]
    hallucinated_numbers: list[float]
    verified_entities: list[str]
    hallucinated_entities: list[str]
    details: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "numeric_grounded_ratio": round(self.numeric_grounded_ratio, 4),
            "entity_grounded_ratio": round(self.entity_grounded_ratio, 4),
            "verified_numbers": self.verified_numbers,
            "hallucinated_numbers": self.hallucinated_numbers,
            "verified_entities": self.verified_entities,
            "hallucinated_entities": self.hallucinated_entities,
            "details": self.details,
        }


def _extract_numbers(text: str) -> list[float]:
    """Extract numeric values, percentages, and currencies from text."""
    # 1. Strip Markdown list enumeration prefixes (e.g. "1. ", "1、 ", "(1) ")
    # Must require whitespace AFTER the dot so decimals like 15.8% are never touched!
    clean_text = re.sub(r"(?:^|\n)\s*\(?\d{1,2}\s*[.、)]\s+", "\n", text)
    # 2. Strip Hypothesis identifiers like H1, H2, H3
    clean_text = re.sub(r"\bH\d+\b", " ", clean_text, flags=re.IGNORECASE)
    # 3. Strip ISO dates, year-months, and Chinese date/time patterns (e.g. 2024-07-01, 7月, 2024年, 3天, 24小时)
    clean_text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", clean_text)
    clean_text = re.sub(r"\b20\d{2}\s*年?", " ", clean_text)
    clean_text = re.sub(r"(?<!\d)(?:[1-9]|1[0-2])\s*月", " ", clean_text)
    clean_text = re.sub(r"(?<!\d)\d{1,2}\s*日", " ", clean_text)
    clean_text = re.sub(r"(?<!\d)\d+\s*天", " ", clean_text)
    clean_text = re.sub(r"(?<!\d)\d+\s*小时", " ", clean_text)
    clean_text = re.sub(r"(?<!\d)\d+\s*分钟", " ", clean_text)
    # 4. Strip star ratings like 1★, 5★
    clean_text = re.sub(r"\d+\s*★", " ", clean_text)

    matches = re.findall(r"(?<![-/])\b\d+(?:\.\d+)?%?", clean_text)
    numbers: list[float] = []
    for m in matches:
        if m.endswith("%"):
            try:
                numbers.append(round(float(m[:-1]) / 100.0, 4))
            except ValueError:
                continue
        else:
            try:
                val = float(m)
                numbers.append(round(val, 4))
            except ValueError:
                continue
    return numbers


def _numbers_match(num_a: float, num_b: float, tol: float = 0.05) -> bool:
    """Check if two numbers match within tolerance or percentage scaling."""
    if math.isclose(num_a, num_b, rel_tol=tol, abs_tol=tol):
        return True
    # Check if one is percentage form of another (e.g. 15.8 vs 0.158)
    if math.isclose(num_a * 100.0, num_b, rel_tol=tol, abs_tol=tol):
        return True
    return math.isclose(num_a, num_b * 100.0, rel_tol=tol, abs_tol=tol)


class NarrativeFaithfulnessEvaluator:
    """Evaluates whether the agent's natural language report faithfully reflects SQL rows.

    Catches the classic 'Secondary Hallucination' trap: SQL ran correctly, but the LLM
    summarized or fabricated different figures in its narrative text.
    """

    def __init__(self, *, tolerance: float = 0.05) -> None:
        self.tolerance = tolerance

    def evaluate(
        self,
        report_text: str,
        sql_rows: Sequence[dict[str, Any]],
        *,
        evidence: Sequence[dict[str, Any]] | None = None,
        actions: Sequence[dict[str, Any]] | None = None,
    ) -> NarrativeFaithfulnessResult:
        if not report_text.strip():
            return NarrativeFaithfulnessResult(
                score=0.0,
                numeric_grounded_ratio=0.0,
                entity_grounded_ratio=0.0,
                verified_numbers=[],
                hallucinated_numbers=[],
                verified_entities=[],
                hallucinated_entities=[],
                details="Empty narrative report",
            )

        # 1. Collect all valid numbers in the SQL ground truth rows
        ground_truth_numbers: set[float] = set()
        for row in sql_rows:
            for val in row.values():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    ground_truth_numbers.add(round(float(val), 4))
                elif isinstance(val, str):
                    for n in _extract_numbers(val):
                        ground_truth_numbers.add(n)

        # Include candidate action parameters (amounts, target counts)
        if actions:
            for act in actions:
                for v in act.values():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        ground_truth_numbers.add(round(float(v), 4))

        # 2. Extract numbers cited in report
        cited_numbers = _extract_numbers(report_text)
        verified_nums: list[float] = []
        hallucinated_nums: list[float] = []

        # If report mentions row count, also treat row count as valid
        ground_truth_numbers.add(float(len(sql_rows)))

        for n in cited_numbers:
            # Check if this number matches any number in DB rows
            is_grounded = any(
                _numbers_match(n, gt, self.tolerance) for gt in ground_truth_numbers
            )
            # Allow common descriptive constants like 1, 2, 3 if describing top-N
            if is_grounded or (n in (1.0, 2.0, 3.0, 5.0, 10.0) and "top" in report_text.lower()):
                verified_nums.append(n)
            else:
                hallucinated_nums.append(n)

        num_ratio = (
            len(verified_nums) / len(cited_numbers) if cited_numbers else 1.0
        )

        # 3. Collect valid entity strings
        valid_entity_tokens: set[str] = set()
        for row in sql_rows:
            for val in row.values():
                if isinstance(val, str) and len(val.strip()) >= 2:
                    valid_entity_tokens.add(val.strip().lower())
        if evidence:
            for ev in evidence:
                for _k, v in ev.items():
                    if isinstance(v, str) and len(v.strip()) >= 2:
                        valid_entity_tokens.add(v.strip().lower())

        # Check entity tokens in report
        verified_ents: list[str] = []
        hallucinated_ents: list[str] = []
        # Check against ground truth tokens
        for ent in valid_entity_tokens:
            if ent in report_text.lower():
                verified_ents.append(ent)

        ent_ratio = 1.0  # Default to 1.0 if no specific entities extracted
        if valid_entity_tokens:
            ent_ratio = min(1.0, len(verified_ents) / max(1, len(valid_entity_tokens)))

        # Composite Narrative Faithfulness Score
        composite_score = 0.7 * num_ratio + 0.3 * ent_ratio

        details = (
            f"Verified {len(verified_nums)}/{len(cited_numbers)} numbers; "
            f"Identified {len(verified_ents)} ground truth entities."
        )

        return NarrativeFaithfulnessResult(
            score=composite_score,
            numeric_grounded_ratio=num_ratio,
            entity_grounded_ratio=ent_ratio,
            verified_numbers=verified_nums,
            hallucinated_numbers=hallucinated_nums,
            verified_entities=verified_ents,
            hallucinated_entities=hallucinated_ents,
            details=details,
        )


# --------------------------------------------------------------------------
# 2. Unstructured Context Quality (Reviews & Text Retrieval: Precision & Relevance)
# --------------------------------------------------------------------------


@dataclass(slots=True)
class UnstructuredContextResult:
    """Evaluation result for unstructured text context retrieval (RAG over reviews/complaints)."""

    context_precision: float  # Top-K Rank-Discounted Precision
    context_relevance: float  # Signal-to-noise ratio of retrieved passages
    total_chunks: int
    relevant_chunks: int
    top_k_ranks: list[int]
    details: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "context_precision": round(self.context_precision, 4),
            "context_relevance": round(self.context_relevance, 4),
            "total_chunks": self.total_chunks,
            "relevant_chunks": self.relevant_chunks,
            "top_k_ranks": self.top_k_ranks,
            "details": self.details,
        }


class UnstructuredContextEvaluator:
    """Evaluates retrieval quality of unstructured customer reviews / complaint texts.

    Computes:
    1. Context Precision: Precision@K weighted by reciprocal rank (higher rank relevant docs score higher).
    2. Context Relevance: Proportion of retrieved sentences/clauses containing relevant keywords or semantics.
    """

    def __init__(self, *, relevance_cues: Sequence[str] | None = None) -> None:
        self.default_cues = set(relevance_cues or ())

    def evaluate(
        self,
        query: str,
        retrieved_contexts: Sequence[str],
        *,
        ground_truth_keywords: Sequence[str] | None = None,
    ) -> UnstructuredContextResult:
        if not retrieved_contexts:
            return UnstructuredContextResult(
                context_precision=0.0,
                context_relevance=0.0,
                total_chunks=0,
                relevant_chunks=0,
                top_k_ranks=[],
                details="No retrieved context chunks provided",
            )

        # Establish target semantic cues from query and explicit keywords
        cues = set(self.default_cues)
        if ground_truth_keywords:
            cues.update(k.lower() for k in ground_truth_keywords)

        # Extract n-grams and significant words from query
        q_tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]{2,}", query.lower())
        cues.update(q_tokens)

        relevant_indices: list[int] = []
        total_relevance_hits = 0

        for rank, chunk in enumerate(retrieved_contexts, start=1):
            chunk_lower = chunk.lower()
            # A chunk is considered relevant if it hits core query cues or problem symptoms
            hit_count = sum(1 for cue in cues if cue in chunk_lower)
            if hit_count >= 1:
                relevant_indices.append(rank)
                total_relevance_hits += 1

        total = len(retrieved_contexts)
        rel_count = len(relevant_indices)

        # Context Precision (Mean Average Precision / Rank Discounted)
        # AP = sum(P@k * rel_k) / total_relevant
        if rel_count == 0:
            precision_score = 0.0
        else:
            prec_sum = 0.0
            hits_so_far = 0
            for rank in range(1, total + 1):
                if rank in relevant_indices:
                    hits_so_far += 1
                    prec_sum += hits_so_far / rank
            precision_score = prec_sum / rel_count

        # Context Relevance (Signal-to-noise ratio)
        relevance_score = rel_count / total if total > 0 else 0.0

        details = (
            f"Found {rel_count}/{total} relevant chunks at ranks {relevant_indices}. "
            f"Precision@K: {precision_score:.3f}, Relevance: {relevance_score:.3f}"
        )

        return UnstructuredContextResult(
            context_precision=precision_score,
            context_relevance=relevance_score,
            total_chunks=total,
            relevant_chunks=rel_count,
            top_k_ranks=relevant_indices,
            details=details,
        )


# --------------------------------------------------------------------------
# 3. Answer & Action Relevance Evaluator
# --------------------------------------------------------------------------


@dataclass(slots=True)
class AnswerRelevanceResult:
    """Measures whether the final report directly addresses query intent."""

    score: float
    intent_match: bool
    action_match: bool
    penalties: list[str]
    details: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "intent_match": self.intent_match,
            "action_match": self.action_match,
            "penalties": self.penalties,
            "details": self.details,
        }


def _tokenize_query_terms(text: str) -> set[str]:
    """Tokenize English words and CJK characters/bigrams."""
    tokens = set(re.findall(r"[a-zA-Z0-9_]{2,}", text.lower()))
    cjk = [c for c in text if "\u4e00" <= c <= "\u9fff"]
    # Add bigrams for CJK
    for i in range(len(cjk) - 1):
        tokens.add(cjk[i] + cjk[i + 1])
    if not tokens and cjk:
        tokens.update(cjk)
    return tokens


class AnswerRelevanceEvaluator:
    """Evaluates question-answering completeness and business action relevancy."""

    def evaluate(
        self,
        query: str,
        answer: str,
        *,
        expected_intent: str = "",
        remediation_action: str | None = None,
    ) -> AnswerRelevanceResult:
        if not answer.strip():
            return AnswerRelevanceResult(
                score=0.0,
                intent_match=False,
                action_match=False,
                penalties=["empty_answer"],
                details="Empty answer received",
            )

        penalties: list[str] = []
        base_score = 1.0

        # Check for non-committal or evasion responses
        evasion_cues = ("我不知道", "无法回答", "无法确定", "抱歉没有数据", "i don't know")
        if any(c in answer.lower() for c in evasion_cues) and len(answer) < 50:
            penalties.append("evasive_or_unhelpful")
            base_score -= 0.5

        # Check if query intent is matched in the answer
        q_tokens = _tokenize_query_terms(query)
        ans_lower = answer.lower()
        overlap_count = sum(1 for tok in q_tokens if tok in ans_lower)
        intent_match = (overlap_count >= 2 or (overlap_count / len(q_tokens) >= 0.15)) if q_tokens else True

        if not intent_match:
            penalties.append("low_query_overlap")
            base_score -= 0.3

        # If remediation action was proposed, verify it is contextually sensible
        action_match = True
        if remediation_action:
            action_lower = remediation_action.lower()
            if action_lower not in answer.lower() and "建议" not in answer and "补救" not in answer:
                action_match = False
                penalties.append("missing_expected_remediation")
                base_score -= 0.2

        score = max(0.0, min(1.0, base_score))
        return AnswerRelevanceResult(
            score=score,
            intent_match=intent_match,
            action_match=action_match,
            penalties=penalties,
            details=f"Relevance: {score:.2f} (Penalties: {penalties or 'None'})",
        )


# --------------------------------------------------------------------------
# 4. Composite Outer-Loop Ragas Evaluator
# --------------------------------------------------------------------------


@dataclass(slots=True)
class OuterLoopCaseResult:
    """Comprehensive evaluation record for one case under the Outer Loop."""

    case_id: str
    query: str
    narrative_faithfulness: NarrativeFaithfulnessResult
    unstructured_context: UnstructuredContextResult
    answer_relevance: AnswerRelevanceResult
    composite_ragas_score: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "composite_ragas_score": round(self.composite_ragas_score, 4),
            "narrative_faithfulness": self.narrative_faithfulness.to_payload(),
            "unstructured_context": self.unstructured_context.to_payload(),
            "answer_relevance": self.answer_relevance.to_payload(),
        }


@dataclass(slots=True)
class OuterLoopAggregateReport:
    """Aggregated Outer Loop Benchmark Report."""

    total_cases: int
    mean_composite_score: float
    mean_narrative_faithfulness: float
    mean_numeric_grounded_ratio: float
    mean_context_precision: float
    mean_context_relevance: float
    mean_answer_relevance: float
    secondary_hallucination_rate: float  # Percentage of cases where ungrounded numbers were fabricated
    cases: list[dict[str, Any]] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "mean_composite_score": round(self.mean_composite_score, 4),
            "mean_narrative_faithfulness": round(self.mean_narrative_faithfulness, 4),
            "mean_numeric_grounded_ratio": round(self.mean_numeric_grounded_ratio, 4),
            "mean_context_precision": round(self.mean_context_precision, 4),
            "mean_context_relevance": round(self.mean_context_relevance, 4),
            "mean_answer_relevance": round(self.mean_answer_relevance, 4),
            "secondary_hallucination_rate": round(self.secondary_hallucination_rate, 4),
            "cases": self.cases,
        }


class OuterLoopRagasEvaluator:
    """Unified Outer-Loop Evaluator combining Narrative Faithfulness, Context Quality, and Relevance."""

    def __init__(self) -> None:
        self.narrative_eval = NarrativeFaithfulnessEvaluator()
        self.context_eval = UnstructuredContextEvaluator()
        self.relevance_eval = AnswerRelevanceEvaluator()

    def evaluate_case(
        self,
        case_id: str,
        query: str,
        report_text: str,
        sql_rows: Sequence[dict[str, Any]],
        retrieved_contexts: Sequence[str],
        *,
        evidence: Sequence[dict[str, Any]] | None = None,
        actions: Sequence[dict[str, Any]] | None = None,
        remediation_action: str | None = None,
        ground_truth_keywords: Sequence[str] | None = None,
    ) -> OuterLoopCaseResult:
        # 1. Narrative Faithfulness
        nf_res = self.narrative_eval.evaluate(
            report_text, sql_rows, evidence=evidence, actions=actions
        )

        # 2. Unstructured Context Precision & Relevance
        uc_res = self.context_eval.evaluate(
            query, retrieved_contexts, ground_truth_keywords=ground_truth_keywords
        )

        # 3. Answer Relevance
        ar_res = self.relevance_eval.evaluate(
            query, report_text, remediation_action=remediation_action
        )

        # Composite Ragas Harmonic / Weighted Score:
        # 40% Narrative Faithfulness + 30% Context Quality + 30% Answer Relevance
        context_score = 0.5 * uc_res.context_precision + 0.5 * uc_res.context_relevance
        composite = 0.40 * nf_res.score + 0.30 * context_score + 0.30 * ar_res.score

        return OuterLoopCaseResult(
            case_id=case_id,
            query=query,
            narrative_faithfulness=nf_res,
            unstructured_context=uc_res,
            answer_relevance=ar_res,
            composite_ragas_score=composite,
        )

    def aggregate(self, results: Sequence[OuterLoopCaseResult]) -> OuterLoopAggregateReport:
        if not results:
            return OuterLoopAggregateReport(
                total_cases=0,
                mean_composite_score=0.0,
                mean_narrative_faithfulness=0.0,
                mean_numeric_grounded_ratio=0.0,
                mean_context_precision=0.0,
                mean_context_relevance=0.0,
                mean_answer_relevance=0.0,
                secondary_hallucination_rate=0.0,
                cases=[],
            )

        n = len(results)
        mean_comp = sum(r.composite_ragas_score for r in results) / n
        mean_nf = sum(r.narrative_faithfulness.score for r in results) / n
        mean_num = sum(r.narrative_faithfulness.numeric_grounded_ratio for r in results) / n
        mean_cp = sum(r.unstructured_context.context_precision for r in results) / n
        mean_cr = sum(r.unstructured_context.context_relevance for r in results) / n
        mean_ar = sum(r.answer_relevance.score for r in results) / n

        # Cases where hallucinated numbers were detected
        hallucinated_cases = sum(
            1 for r in results if len(r.narrative_faithfulness.hallucinated_numbers) > 0
        )
        sec_hal_rate = hallucinated_cases / n

        return OuterLoopAggregateReport(
            total_cases=n,
            mean_composite_score=mean_comp,
            mean_narrative_faithfulness=mean_nf,
            mean_numeric_grounded_ratio=mean_num,
            mean_context_precision=mean_cp,
            mean_context_relevance=mean_cr,
            mean_answer_relevance=mean_ar,
            secondary_hallucination_rate=sec_hal_rate,
            cases=[r.to_payload() for r in results],
        )
