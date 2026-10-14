"""Conservative SQL extraction shared by planners and the execution guard.

The lexer owns statement boundaries, not regexes: comments, quoted identifiers
and string literals cannot hide a terminator. SQLite's built-in parser checks
syntax using EXPLAIN on an isolated in-memory connection; the candidate query
is never executed. Name resolution and execution permissions still belong to
the live-schema/SQL guard and the database authorizer.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from unicodedata import category


@dataclass(frozen=True, slots=True)
class SQLParseResult:
    sql: str = ""
    outcome: str = "no_sql"
    error: str = ""


@dataclass(frozen=True, slots=True)
class _Token:
    text: str
    start: int
    end: int
    kind: str

    @property
    def keyword(self) -> str:
        return self.text.upper() if self.kind == "word" else ""


_WRITE_WORDS = frozenset({
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE",
    "TRUNCATE", "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX",
    "GRANT", "REVOKE", "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT",
    "RELEASE", "ANALYZE", "INTO",
})
_STATEMENT_STARTS = _WRITE_WORDS | {"SELECT", "WITH", "EXPLAIN", "VALUES", "END"}
_UNSAFE_FUNCTIONS = frozenset({"load_extension", "writefile"})
_FENCE_OPEN = re.compile(
    r"(?m)^[ \t]*(?P<mark>`{3,}|~{3,})(?P<lang>[\w-]*)[ \t]*"
    r"(?:\r?\n|(?=(?:SELECT|WITH)\b))",
    re.IGNORECASE,
)
_PROSE_LINE = re.compile(
    r"(?im)^[ \t]*(?:(?:explanation|note)\s*:|this query\b|"
    r"(?:说明|解释|注意)\s*[：:]|(?:以上|该|此)(?:SQL|查询))"
)
_REFUSAL = re.compile(
    r"\b(?:i\s+(?:cannot|can't|am unable to)|unable to)\s+"
    r"(?:answer|provide|generate|help|assist|comply|execute)\b|"
    r"(?:无法|不能|不支持)(?:回答|提供|生成|执行|协助)|抱歉[，,：:]",
    re.IGNORECASE,
)


def _word_char(char: str) -> bool:
    return char.isalnum() or char in "_$" or category(char).startswith("M")


class _LexError(ValueError):
    """Keep safely scanned tokens so an unfinished tail cannot hide a write."""

    def __init__(self, error: str, tokens: list[_Token]) -> None:
        super().__init__(error)
        self.tokens = tokens


def _lex(
    text: str, *, prose: bool = False, fence: str = ""
) -> tuple[list[_Token], int]:
    """Tokenize without normalising SQL; optionally stop at an unquoted fence."""
    tokens: list[_Token] = []
    i, size = 0, len(text)
    while i < size:
        start = i
        char = text[i]
        if char.isspace() or char == "\ufeff":
            i += 1
            continue
        if fence and text.startswith(fence, i):
            return tokens, i
        if text.startswith("--", i):
            i += 2
            while i < size and text[i] not in "\r\n":
                i += 1
            kind = "comment"
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                raise _LexError("unterminated SQL block comment", tokens)
            i, kind = end + 2, "comment"
        elif char in "'\"`[":
            # English wrapper contractions are not SQL string delimiters.
            if prose and char == "'" and i and text[i - 1].isalnum():
                i += 1
                tokens.append(_Token(char, start, i, "symbol"))
                continue
            closer = "]" if char == "[" else char
            kind = "literal" if char == "'" else "quoted"
            i += 1
            while i < size:
                if text[i] == closer:
                    if closer != "]" and i + 1 < size and text[i + 1] == closer:
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            else:
                raise _LexError("unterminated SQL string or quoted identifier", tokens)
        elif _word_char(char):
            i += 1
            while i < size and _word_char(text[i]):
                i += 1
            kind = "word"
            if text[start:i].upper() in _STATEMENT_STARTS:
                prose = False
        else:
            i += 1
            kind = "symbol"
        tokens.append(_Token(text[start:i], start, i, kind))
    return tokens, size


def sql_skeleton(sql: str, *, mask_identifiers: bool = True) -> str:
    """Mask comments/literals/quoted names for guard heuristics, preserving lines.

    With mask_identifiers=False, quoted names become word placeholders instead
    of whitespace, allowing e.g. the guard's comma-join detector to see them
    without mistaking a keyword inside a name for executable SQL.
    Raises ValueError on unfinished quotes/comments. This is not an execution
    approval; call validate_readonly_sql and the normal policy/authorizer too.
    """
    tokens, _ = _lex(sql)
    chars = list(sql)
    for token in tokens:
        if token.kind in {"comment", "literal", "quoted"}:
            fill = "_" if token.kind == "quoted" and not mask_identifiers else " "
            chars[token.start:token.end] = [
                "\n" if char == "\n" else fill for char in token.text
            ]
    return "".join(chars)


def _syntax_error(sql: str) -> str:
    # Preparing EXPLAIN compiles the complete SQL without running any SELECT,
    # recursive CTE or function. Do not use the caller's connection here.
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("EXPLAIN " + sql)
    except sqlite3.Error as exc:
        error = str(exc)
        # The parser has accepted the grammar before resolving these names.
        # Missing tables/functions/bindings must be checked against the actual
        # connection later, rather than fabricated in this scratch database.
        if error.startswith((
            "no such table:", "no such column:", "no such function:",
            "no such collation sequence:", "ambiguous column name:",
            "Incorrect number of bindings supplied.",
        )):
            return ""
        return error
    finally:
        conn.close()
    return ""


def _lex_failure(exc: _LexError) -> SQLParseResult:
    tokens = [token for token in exc.tokens if token.kind != "comment"]
    # Prose never turns known unsafe SQL into a retryable lexical failure.
    # Conversely, a quoted/commented keyword is absent from this word stream.
    prefix_end = next((t.start for t in tokens if t.keyword in _STATEMENT_STARTS), None)
    if tokens and prefix_end is not None:
        for token in tokens:
            if token.keyword in {"EXPLAIN", "END"} and token.start == prefix_end:
                return SQLParseResult(outcome="unsafe_sql", error="only SELECT or read-only WITH is allowed")
    for i, token in enumerate(tokens):
        if token.kind == "symbol" and token.text == ";" and i + 1 < len(tokens):
            return SQLParseResult(outcome="multiple_statements", error="additional tokens after a SQL terminator")
        next_text = tokens[i + 1].text if i + 1 < len(tokens) else ""
        if token.keyword in _WRITE_WORDS and not (token.keyword == "REPLACE" and next_text == "("):
            return SQLParseResult(outcome="unsafe_sql", error=f"non-read SQL keyword: {token.keyword}")
        if next_text == "(" and token.text.strip('"`[]').lower() in _UNSAFE_FUNCTIONS:
            return SQLParseResult(outcome="unsafe_sql", error="non-read SQL function")
    return SQLParseResult(outcome="invalid_sql", error=str(exc))


def validate_readonly_sql(sql: str) -> SQLParseResult:
    """Accept one complete SELECT or read-only WITH statement, without wrappers.

    No prefix is salvaged from stacked or malformed SQL. The returned SQL keeps
    comments, spelling, identifier quoting, and the optional final semicolon.
    """
    candidate = sql.strip()
    if not candidate:
        return SQLParseResult(outcome="empty", error="empty SQL")
    try:
        raw_tokens, _ = _lex(candidate)
    except _LexError as exc:
        return _lex_failure(exc)
    tokens = [token for token in raw_tokens if token.kind != "comment"]
    if not tokens:
        return SQLParseResult(outcome="no_sql", error="only SQL comments were returned")
    terminators = [i for i, token in enumerate(tokens) if token.text == ";" and token.kind == "symbol"]
    if terminators and (len(terminators) != 1 or terminators[0] != len(tokens) - 1):
        return SQLParseResult(outcome="multiple_statements", error="multiple SQL statements are not allowed")
    if tokens[0].keyword not in {"SELECT", "WITH"}:
        return SQLParseResult(outcome="unsafe_sql", error="only SELECT or read-only WITH is allowed")

    depth = 0
    for i, token in enumerate(tokens):
        next_text = tokens[i + 1].text if i + 1 < len(tokens) else ""
        keyword = token.keyword
        if keyword in _WRITE_WORDS and not (keyword == "REPLACE" and next_text == "("):
            return SQLParseResult(outcome="unsafe_sql", error=f"non-read SQL keyword: {keyword}")
        name = token.text.strip('"`[]').lower()
        if next_text == "(" and name in _UNSAFE_FUNCTIONS:
            return SQLParseResult(outcome="unsafe_sql", error=f"non-read SQL function: {name}")
        if token.kind != "symbol":
            continue
        if token.text == "(":
            depth += 1
        elif token.text == ")":
            depth -= 1
            if depth < 0:
                return SQLParseResult(outcome="invalid_sql", error="unbalanced SQL parentheses")
        elif any(ord(char) > 127 for char in token.text):
            # Chinese identifiers are valid; unquoted prose punctuation is not
            # a SQL separator (SQLite otherwise treats e.g. 语句。 as a name).
            return SQLParseResult(outcome="invalid_sql", error="prose punctuation outside a quoted SQL value")
    if depth:
        return SQLParseResult(outcome="invalid_sql", error="unbalanced SQL parentheses")
    error = _syntax_error(candidate)
    if error:
        return SQLParseResult(outcome="invalid_sql", error=error)
    return SQLParseResult(sql=candidate, outcome="accepted")


def _statement_start(text: str) -> tuple[int | None, list[_Token]]:
    tokens, _ = _lex(text, prose=True)
    for i, token in enumerate(tokens):
        if token.keyword in _STATEMENT_STARTS:
            # Keep a leading SQL comment, but not the prose preceding it.
            while i > 0 and tokens[i - 1].kind == "comment":
                i -= 1
            return tokens[i].start, tokens
    return None, tokens


def _has_extra_sql(text: str) -> bool:
    try:
        start, tokens = _statement_start(text)
    except ValueError:
        # Ambiguous/unclosed wrappers cannot justify dropping trailing text.
        return True
    return start is not None or any(token.text == ";" for token in tokens if token.kind == "symbol")


def _extract_plain(text: str) -> SQLParseResult:
    try:
        start, _ = _statement_start(text)
    except _LexError as exc:
        return _lex_failure(exc)
    prefix = text[:start] if start is not None else text
    if _REFUSAL.search(prefix):
        return SQLParseResult(outcome="refusal", error="model declined to produce SQL")
    if start is None:
        return SQLParseResult(outcome="no_sql", error="no SELECT or WITH statement found")
    candidate = text[start:].strip()
    parsed = validate_readonly_sql(candidate)
    if parsed.outcome == "accepted":
        return parsed
    try:
        tokens, _ = _lex(candidate)
    except ValueError:
        return parsed
    terminator = next((t for t in tokens if t.kind == "symbol" and t.text == ";"), None)
    if terminator is not None:
        suffix = candidate[terminator.end:]
        if _has_extra_sql(suffix):
            return SQLParseResult(outcome="multiple_statements", error="additional SQL after the first statement")
        # Only an independently complete statement followed by non-SQL prose
        # can be unwrapped; literal/comment semicolons never reach this branch.
        return validate_readonly_sql(candidate[:terminator.end])
    if parsed.outcome == "invalid_sql":
        for match in _PROSE_LINE.finditer(candidate):
            if not _has_extra_sql(candidate[match.start():]):
                prefix_result = validate_readonly_sql(candidate[:match.start()])
                if prefix_result.outcome == "accepted":
                    return prefix_result
    return parsed


def extract_readonly_sql(text: str) -> SQLParseResult:
    """Unwrap one SQL response; reject alternatives, writes, and stacked SQL."""
    stripped = text.strip()
    if not stripped:
        return SQLParseResult(outcome="empty", error="empty model response")
    # A raw SQL statement owns all of its contents, including any markdown
    # marker inside a literal/comment. Never search it for a later SELECT.
    try:
        prefix_tokens, _ = _lex(stripped, prose=True, fence="```")
        first = next((t for t in prefix_tokens if t.kind != "comment"), None)
        if first is not None and first.keyword in _STATEMENT_STARTS:
            return _extract_plain(stripped)
    except ValueError:
        # A fenced response can contain wrapper apostrophes. Inspect its body
        # separately, rather than treating a prose quote as an SQL delimiter.
        pass

    opening = _FENCE_OPEN.search(stripped)
    if opening is None:
        if "```" in stripped or "~~~" in stripped:
            return SQLParseResult(outcome="invalid_sql", error="unclosed or unsupported SQL fence")
        return _extract_plain(stripped)
    if _REFUSAL.search(stripped[:opening.start()]):
        return SQLParseResult(outcome="refusal", error="model declined to produce SQL")
    if opening.group("lang").lower() not in {"", "sql", "sqlite", "sqlite3"}:
        return SQLParseResult(outcome="invalid_sql", error="unsupported SQL fence language")
    body_start = opening.end()
    marker = opening.group("mark")
    try:
        _, body_end = _lex(stripped[body_start:], fence=marker)
    except _LexError as exc:
        return _lex_failure(exc)
    closing = body_start + body_end
    if closing == len(stripped):
        return SQLParseResult(outcome="invalid_sql", error="unclosed SQL fence")
    suffix = stripped[closing + len(marker):]
    if _FENCE_OPEN.search(suffix) or "```" in suffix or "~~~" in suffix:
        return SQLParseResult(outcome="multiple_statements", error="multiple SQL code blocks are not allowed")
    if _has_extra_sql(stripped[:opening.start()]) or _has_extra_sql(suffix):
        return SQLParseResult(outcome="multiple_statements", error="SQL outside the selected code block")
    return validate_readonly_sql(stripped[body_start:closing])


__all__ = ["SQLParseResult", "extract_readonly_sql", "sql_skeleton", "validate_readonly_sql"]
