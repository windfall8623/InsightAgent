"""Agentic RAG over database schema documents.

Retrieval is hybrid on purpose. Dense vectors alone miss exact identifiers
(`pay_status`), while keyword search alone misses paraphrases ("退款率" ->
refund rate). Running both and fusing them recovers each other's misses, and
the ablation report measures exactly how much each leg contributes.

The store is in-process by design: the project must run on one machine with no
external service. :class:`VectorStore` is an interface, so swapping in a real
vector database is a drop-in change documented in the README.
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens plus CJK character bigrams.

    Chinese has no spaces, so word-level BM25 degenerates; character bigrams
    give it something to match on.
    """

    lowered = text.lower()
    tokens = _TOKEN.findall(lowered)
    cjk = re.findall(r"[一-鿿]+", text)
    for run in cjk:
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


@dataclass(slots=True)
class SchemaDoc:
    """One retrievable unit: a table, or a column within a table."""

    id: str
    kind: str  # "table" | "column"
    table: str
    title: str
    body: str
    keywords: list[str] = field(default_factory=list)
    # Canonical values a filter may compare against, e.g. city names.
    values: list[str] = field(default_factory=list)

    def to_evidence(self, score: float) -> dict[str, Any]:
        return {
            "id": self.id,
            "table": self.table,
            "kind": self.kind,
            "title": self.title,
            "score": round(score, 4),
        }


class VectorStore(Protocol):
    """Minimal vector index interface, so a real vector DB can replace this."""

    def add(self, doc_id: str, text: str) -> None: ...

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]: ...


class HashingVectorStore:
    """Deterministic hashed bag-of-tokens embedding.

    No model download, no network, and — critically — **stable across processes**.
    Python salts string hashing per process (PYTHONHASHSEED), so the builtin
    ``hash()`` would make retrieval order differ between runs and quietly break
    the reproducibility guarantee the evaluation report relies on. BLAKE2b gives
    the same digest in every process.

    It is weaker than a learned embedding, and the ablation report treats it as
    such.
    """

    def __init__(self, dims: int = 512) -> None:
        self.dims = dims
        self._vectors: dict[str, dict[int, float]] = {}
        self._norms: dict[str, float] = {}

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dims

    def _embed(self, text: str) -> dict[int, float]:
        vec: dict[int, float] = {}
        for token in tokenize(text):
            idx = self._bucket(token)
            vec[idx] = vec.get(idx, 0.0) + 1.0
        return vec

    def add(self, doc_id: str, text: str) -> None:
        vec = self._embed(text)
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        self._vectors[doc_id] = vec
        self._norms[doc_id] = norm

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        q = self._embed(query)
        qnorm = math.sqrt(sum(v * v for v in q.values())) or 1.0
        scored: list[tuple[str, float]] = []
        for doc_id, vec in self._vectors.items():
            if len(q) > len(vec):
                shared = vec
                other = q
            else:
                shared = q
                other = vec
            dot = sum(v * other.get(k, 0.0) for k, v in shared.items())
            scored.append((doc_id, dot / (qnorm * self._norms[doc_id])))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]


class BM25Index:
    """Classic BM25 over tokenised documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: dict[str, list[str]] = {}
        self._freqs: dict[str, Counter[str]] = {}
        self._df: Counter[str] = Counter()
        self._avg_len = 0.0

    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        self._docs[doc_id] = tokens
        freqs = Counter(tokens)
        self._freqs[doc_id] = freqs
        for term in freqs:
            self._df[term] += 1
        self._avg_len = sum(len(t) for t in self._docs.values()) / max(len(self._docs), 1)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        terms = tokenize(query)
        n = len(self._docs)
        results: list[tuple[str, float]] = []
        for doc_id, tokens in self._docs.items():
            freqs = self._freqs[doc_id]
            length = len(tokens) or 1
            score = 0.0
            for term in terms:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                df = self._df[term]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b * length / (self._avg_len or 1))
                score += idf * (tf * (self.k1 + 1)) / denom
            if score > 0:
                results.append((doc_id, score))
        results.sort(key=lambda item: (-item[1], item[0]))
        return results[:top_k]


class HybridRetriever:
    """Fuses BM25 and vector hits with reciprocal rank fusion, then reranks.

    Reciprocal rank fusion is used instead of score normalisation because BM25
    and cosine scores are on incomparable scales; ranks are not.
    """

    dims = 512

    def __init__(
        self,
        *,
        rrf_k: int = 60,
        rerank_weight: float = 0.25,
        use_qdrant: bool = True,
        qdrant_url: str | None = None,
    ) -> None:
        self.bm25 = BM25Index()
        self.vectors = HashingVectorStore()
        self.qdrant: Any | None = None
        if use_qdrant:
            try:
                from insight_agent.app.qdrant_store import QdrantVectorStore

                self.qdrant = QdrantVectorStore("schema_catalog", url=qdrant_url)
            except Exception:
                self.qdrant = None
        self.docs: dict[str, SchemaDoc] = {}
        self.rrf_k = rrf_k
        self.rerank_weight = rerank_weight
        from insight_agent.app.bge_reranker import BGEReranker

        self.bge_reranker = BGEReranker()

    def index(self, docs: list[SchemaDoc]) -> None:
        for doc in docs:
            self.docs[doc.id] = doc
            text = " ".join([doc.table, doc.title, doc.body, *doc.keywords, *doc.values])
            self.bm25.add(doc.id, text)
            self.vectors.add(doc.id, text)
            if self.qdrant is not None:
                from insight_agent.app.qdrant_store import compute_hash_dense_embedding

                hash_vec = compute_hash_dense_embedding(text, dims=self.qdrant.dims)
                self.qdrant.upsert(
                    doc_id=doc.id,
                    text_or_vector=hash_vec,
                    payload={"table": doc.table, "kind": doc.kind, "keywords": doc.keywords},
                )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> list[tuple[SchemaDoc, float]]:
        """Retrieve the most relevant schema docs.

        ``mode`` selects the retrieval strategy and exists so the ablation study
        can disable one leg at a time without touching call sites.
        """

        if mode == "keyword":
            # BM25 scores have no common scale with cosine, but within a single
            # leg rank order is all that matters, so convert rank to a weight.
            hits = self.bm25.search(query, top_k * 3)
            scored = [
                (doc_id, 1.0 - rank / max(len(hits), 1)) for rank, (doc_id, _) in enumerate(hits)
            ]
        elif mode == "vector":
            if self.qdrant is not None:
                q_hits = self.qdrant.search(query, top_k=top_k * 3)
                scored = [(doc_id, score) for doc_id, score, _ in q_hits] if q_hits else self.vectors.search(query, top_k * 3)
            else:
                scored = self.vectors.search(query, top_k * 3)
        else:
            scored = self._fuse(query)

        scored = [(doc_id, score) for doc_id, score in scored if doc_id in self.docs]
        reranked = self._rerank(query, scored)
        # Ties break on doc id so retrieval order is reproducible run to run.
        reranked.sort(key=lambda item: (-item[1], item[0]))
        return [(self.docs[doc_id], score) for doc_id, score in reranked[:top_k]]

    def _fuse(self, query: str) -> list[tuple[str, float]]:
        """Reciprocal rank fusion over both legs."""

        fused: dict[str, float] = {}
        for rank, (doc_id, _) in enumerate(self.bm25.search(query, 20)):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        
        vec_hits: list[tuple[str, float]] = []
        if self.qdrant is not None:
            q_hits = self.qdrant.search(query, top_k=20)
            if q_hits:
                vec_hits = [(doc_id, score) for doc_id, score, _ in q_hits]
        if not vec_hits:
            vec_hits = self.vectors.search(query, 20)

        for rank, (doc_id, _) in enumerate(vec_hits):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        return sorted(fused.items(), key=lambda item: (-item[1], item[0]))

    def _rerank(
        self,
        query: str,
        scored: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Stage 2: Cross-Encoder semantic reranking with domain value boosting."""
        from insight_agent.app.bge_reranker import RerankCandidate

        terms = set(tokenize(query))
        candidates: list[RerankCandidate] = []
        for doc_id, score in scored:
            doc = self.docs[doc_id]
            hits = sum(1 for value in doc.values if value in terms or value in query)
            boosted = score * (1 + self.rerank_weight * hits)
            text = f"{doc.table} {doc.title} {doc.body} {' '.join(doc.keywords)}"
            candidates.append(
                RerankCandidate(
                    doc_id=doc_id,
                    text=text,
                    initial_score=boosted,
                    metadata={"table": doc.table},
                )
            )

        reranked_cands = self.bge_reranker.rerank(query, candidates, top_k=len(candidates))
        return [(c.doc_id, c.rerank_score) for c in reranked_cands]

    def search_tables(
        self, query: str, *, top_k: int = 3, mode: str = "hybrid"
    ) -> list[tuple[str, float]]:
        """Rank *tables* by aggregating the scores of their columns.

        Ranking columns and then inferring tables from the top-k is the wrong
        order: a handful of strong column matches (orders.total_amount for
        "销售额") crowd out the table a question is actually about, and the
        planner is then forbidden from using it. Schema linking works at table
        granularity, so retrieval should too.
        """

        per_table: dict[str, float] = {}
        for doc, score in self.search(query, top_k=self.dims * 4, mode=mode):
            per_table[doc.table] = max(per_table.get(doc.table, 0.0), score)
        ranked = sorted(per_table.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_k]


# --------------------------------------------------------------------------
# Schema corpus construction
# --------------------------------------------------------------------------

from insight_agent.app.schema_vocab import (
    get_all_table_aliases,
    get_all_column_aliases,
    METRIC_DEFINITIONS,
)

_TABLE_ALIASES: dict[str, list[str]] = get_all_table_aliases()


def read_schema(conn: sqlite3.Connection) -> dict[str, tuple[list[Any], dict[str, list[str]]]]:
    """Read table names, columns, and sampled literals in one locked pass.

    All database access happens here, under the connection lock. Schema
    introspection runs on the same connection as the request path, so a bare
    ``conn.execute`` outside the lock deadlocks under concurrent use.
    """

    from insight_agent.data.database import locked

    with locked(conn):
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]
        schema: dict[str, tuple[list[Any], dict[str, list[str]]]] = {}
        for table in tables:
            columns = list(conn.execute(f'PRAGMA table_info("{table}")'))
            values: dict[str, list[str]] = {}
            for column in columns:
                values[column[1]] = _distinct_values_locked(conn, table, column[1])
            schema[table] = (list(columns), values)
    return schema


def build_schema_docs(conn: sqlite3.Connection) -> list[SchemaDoc]:
    """Read the live schema and wrap each table and column as a document."""

    schema = read_schema(conn)
    docs: list[SchemaDoc] = []

    # Augment table aliases with metric synonyms
    table_metric_synonyms: dict[str, list[str]] = {}
    for m in METRIC_DEFINITIONS.values():
        for tbl in m.get("tables", []):
            table_metric_synonyms.setdefault(tbl, []).extend(m.get("synonyms", []))

    for table, (columns, literals) in schema.items():
        aliases = list(_TABLE_ALIASES.get(table, []))
        if table in table_metric_synonyms:
            aliases.extend(table_metric_synonyms[table])
        col_words = [c[1] for c in columns]
        docs.append(
            SchemaDoc(
                id=f"table:{table}",
                kind="table",
                table=table,
                title=f"table {table}",
                body=(
                    f"{table} ({', '.join(col_words)}) contains "
                    f"{', '.join(aliases) or table} records"
                ),
                keywords=[table, *aliases],
            )
        )

        for column in columns:
            cname, ctype = column[1], column[2]
            doc_aliases = [a for a in aliases if _matches_column(a, cname)]
            docs.append(
                SchemaDoc(
                    id=f"column:{table}.{cname}",
                    kind="column",
                    table=table,
                    title=f"{table}.{cname}",
                    body=(
                        f"{table}.{cname} {ctype} column"
                        f"{(' of ' + table) if table else ''}"
                        f"{(' also called ' + '/'.join(doc_aliases)) if doc_aliases else ''}"
                    ),
                    keywords=[cname, *doc_aliases, *_field_words(cname)],
                    values=literals.get(cname, []),
                )
            )
    return docs


def _field_words(column: str) -> list[str]:
    return [p for p in column.split("_") if len(p) > 2]


# Chinese names for columns. Tables already had these; columns did not, which
# meant a question phrased in Chinese could never retrieve the column it needed
# -- "各城市销售额" failed to reach users.city, and the planner was then told to
# use only retrieved tables, so it correctly refused a question it could answer.
_COLUMN_ALIASES: dict[str, list[str]] = get_all_column_aliases()


def _matches_column(alias: str, column: str) -> bool:
    return alias.lower() in column.lower() or column.lower() in alias.lower()


def _column_aliases(column: str) -> list[str]:
    """Chinese and English names for a column, used as retrieval keywords."""

    return list(_COLUMN_ALIASES.get(column.lower(), ()))


def _distinct_values_locked(
    conn: sqlite3.Connection, table: str, column: str, cap: int = 60
) -> list[str]:
    """Sample literal values so the retriever can match filter arguments.

    Must be called with the connection lock held (see :func:`read_schema`).
    Only low-cardinality columns are sampled; scanning a large fact table to
    index it would be wasteful.
    """

    try:
        rows = conn.execute(
            f'SELECT DISTINCT "{column}" FROM "{table}" LIMIT ?', (cap + 1,)
        ).fetchall()
    except sqlite3.Error:
        return []
    values = [str(r[0]) for r in rows if r[0] is not None]
    return values if len(values) <= cap else []


def build_retriever(conn: sqlite3.Connection) -> HybridRetriever:
    retriever = HybridRetriever()
    retriever.index(build_schema_docs(conn))
    return retriever
