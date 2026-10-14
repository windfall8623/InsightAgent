"""Deterministic function-calling engine (Berkeley Function Calling Leaderboard Evaluator).

Architecture Positioning:
-------------------------
- `app.funccall` provides a deterministic AST-level function-calling engine and
  rule-based baseline evaluator.
- Primary purpose:
  * Serving as the evaluation harness adapter for the Berkeley Function Calling
    Leaderboard (BFCL) in `insight_agent.eval.bfcl`.
  * Measuring AST-level schema matching and parameter extraction against canonical benchmarks.

Stands in for an LLM the same way :mod:`insight_agent.app.nl2sql` does: given a
natural-language request and a set of callable function schemas, decide *which*
function to call and *with what arguments*.

The pipeline is the same shape as the NL2SQL translator so the two can be
compared:

1. **Select** — score each candidate function by how well its name and
   description match the request. Alias and token overlap both count; a
   description match is weighted higher because that is where a user states
   intent in natural language.
2. **Decide** — call nothing when no candidate clears the relevance threshold.
   This is what makes the ``irrelevance`` category measurable: a system that
   always calls *something* scores zero there.
3. **Fill** — extract an argument value for each required parameter by looking
   for the parameter's tokens in the request and taking the nearest plausible
   value.

This is a rules baseline, and it is expected to be weak. That is the point of
scoring it on BFCL: an external canonical benchmark produces a realistic number
where a self-written dataset produces a flattering one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# Function names arrive as "namespace.call"; the bare verb is the better
# selection signal, but the full name is kept for the emitted call.
_NAME_SPLIT = re.compile(r"[._]")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "for",
        "with",
        "and",
        "to",
        "in",
        "on",
        "at",
        "is",
        "are",
        "please",
        "can",
        "you",
        "i",
        "me",
        "my",
        "it",
        "that",
        "this",
        "get",
        "give",
        "tell",
        "find",
        "what",
        "how",
        "many",
        "much",
    }
)


@dataclass(slots=True)
class FunctionSpec:
    """One callable function, parsed out of the benchmark's schema format."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def bare_name(self) -> str:
        return _NAME_SPLIT.split(self.name)[-1]

    @property
    def required(self) -> list[str]:
        req = self.parameters.get("required") or []
        return [r for r in req if r in self.parameters.get("properties", {})]

    @property
    def property_names(self) -> list[str]:
        return list(self.parameters.get("properties", {}))

    @classmethod
    def from_raw(cls, raw: Any) -> FunctionSpec:
        # BFCL ships the function blob as a JSON *string* inside the record.
        if isinstance(raw, str):
            raw = json.loads(raw)
        first = raw[0] if isinstance(raw, list) else raw
        return cls(
            name=first.get("name", ""),
            description=first.get("description", ""),
            parameters=first.get("parameters", {}) or {},
        )


@dataclass(slots=True)
class FunctionCall:
    """The engine's decision for one request."""

    calls: list[dict[str, Any]] = field(default_factory=list)

    def names(self) -> list[str]:
        return [c.get("name", "") for c in self.calls]


def _tokens(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9_]+", text.lower()))
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _score_function(spec: FunctionSpec, request: str) -> float:
    """Relevance of one function for the request, in [0, 1]."""

    req_tokens = _tokens(request)
    if not req_tokens:
        return 0.0

    name_tokens = _tokens(spec.bare_name.replace("_", " "))
    desc_tokens = _tokens(spec.description)

    # A whole-word hit on the function verb is the strongest signal available.
    verb = spec.bare_name.lower()
    if re.search(rf"\b{re.escape(verb)}\b", request.lower()):
        name_score = 1.0
    else:
        overlap = len(name_tokens & req_tokens)
        name_score = min(1.0, overlap / max(len(name_tokens), 1))

    desc_score = len(desc_tokens & req_tokens) / max(len(desc_tokens), 1)
    return 0.45 * name_score + 0.55 * desc_score


def _extract_argument(param: str, request: str) -> Any:
    """Best-effort value for one parameter.

    Tries, in order: an explicit ``"param": value`` pattern, a number sitting
    near the parameter's words, then any number in the request. Numeric coercion
    follows, because ground truth stores numbers as strings.
    """

    lowered = request.lower()

    explicit = re.search(
        rf"{re.escape(param.lower())}\s*(?:is|=|:|of)?\s*(-?\d+(?:\.\d+)?)", lowered
    )
    if explicit:
        return _coerce(explicit.group(1))

    words = [w for w in re.findall(r"[a-z]+", param.lower()) if w not in _STOPWORDS]
    for word in words:
        near = re.search(rf"{re.escape(word)}\D{{0,20}}?(-?\d+(?:\.\d+)?)", lowered)
        if near:
            return _coerce(near.group(1))

    any_number = _NUMBER.search(request)
    if any_number:
        return _coerce(any_number.group(0))
    return None


def _coerce(raw: str) -> Any:
    value: float | int | str = float(raw)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def decide(
    request: str,
    functions: list[FunctionSpec],
    *,
    relevance_threshold: float = 0.18,
    max_calls: int = 4,
) -> FunctionCall:
    """Choose the functions to call, and the arguments to pass."""

    scored = sorted(
        ((spec, _score_function(spec, request)) for spec in functions),
        key=lambda pair: (-pair[1], pair[0].name),
    )
    if not scored or scored[0][1] < relevance_threshold:
        return FunctionCall(calls=[])

    best = scored[0][1]
    # Parallel category: take every candidate within a small margin of the best.
    selected = [spec for spec, score in scored if score >= best - 0.02][:max_calls]

    calls: list[dict[str, Any]] = []
    for spec in selected:
        arguments: dict[str, Any] = {}
        for param in spec.required:
            value = _extract_argument(param, request)
            if value is not None:
                arguments[param] = value
        calls.append({"name": spec.name, "arguments": arguments})
    return FunctionCall(calls=calls)
