"""BAAI BGE Reranker Adapter for InsightAgent Two-Stage Retrieval.

Implements:
1. Cross-Encoder reranking protocol: transforms (query, doc) pairs into deep semantic relevance scores.
2. Production integration with BAAI/bge-reranker-v2-m3 and bge-reranker-large.
3. Cross-lingual semantic alignment (Chinese query <-> Portuguese Olist review comments).
4. Zero-Infra fallback: when heavy neural frameworks (FlagEmbedding/torch) are not installed,
   gracefully falls back to a high-fidelity heuristic cross-attention scorer, guaranteeing
   reproducible, millisecond-latency execution in CI/CD without model downloads.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from insight_agent.app.rag import tokenize

logger = logging.getLogger(__name__)

# Cross-lingual and domestic retail semantic alignment dictionary
_CROSS_LINGUAL_SYNONYMS: dict[str, list[str]] = {
    # Domestic Retail Chinese supply chain synonyms
    "软果": ["软烂", "变质", "发软", "发酸", "出水", "坏果", "烂果", "不新鲜"],
    "变质": ["发霉", "长毛", "发黑", "发酸", "异味", "臭了", "坏了", "氧化"],
    "发黑": ["氧化", "变黑", "霉烂", "发霉", "长毛", "变色"],
    "漏液": ["漏水", "冰袋融化", "化冻", "流汤", "渗漏", "包装渗水"],
    "超时": ["延误", "迟到", "慢", "未送达", "积压", "滞留", "超时"],
    "虚假妥投": ["未收到", "没收到", "未送达", "显示签收", "虚假签收"],
    "少件": ["缺件", "漏发", "少发", "短少", "空箱"],
    # Portuguese (Olist e-commerce) <-> Chinese
    "破损": ["quebrado", "danificado", "rasgado", "amassado", "estragado", "defeito", "pessimo", "破裂", "压烂", "摔坏"],
    "损坏": ["danificado", "quebrado", "avariado", "defeito", "quebrou", "弄坏", "损坏"],
    "质量": ["qualidade", "material", "ruim", "pessima", "defeito", "falso", "品质", "劣质"],
    "假货": ["falso", "pirata", "replica", "enganoso", "山寨", "高仿"],
    "延误": ["atraso", "demorou", "atrasado", "demora", "prazo", "nao chegou", "espera", "超时", "晚到"],
    "迟到": ["atraso", "demorou", "tarde", "demorado", "prazo", "晚送达"],
    "慢": ["lento", "demorado", "devagar", "demorou", "龟速"],
    "没收到": ["nao recebi", "nao chegou", "nunca recebi", "falta", "extraviado", "丢件", "漏送"],
    "丢失": ["extraviado", "sumiu", "perdeu", "perda", "丢件", "丢失"],
    "退款": ["reembolso", "estorno", "devolucao", "cancelamento", "dinheiro", "退钱", "退货退款", "仅退款"],
    "差评": ["pessimo", "horrivel", "ruim", "insatisfeito", "nao recomendo", "reclamacao", "避雷", "差评", "踩坑"],
    "包装": ["embalagem", "caixa", "pacote", "embrulho", "外包装", "纸箱", "泡沫箱"],
    "卖家": ["vendedor", "loja", "商家", "店铺", "客服"],
    "物流": ["correios", "transportadora", "entrega", "envio", "frete", "快递", "承运商", "顺丰", "中通", "京东"],
    "运费": ["frete", "valor do frete", "caro", "快递费", "运费险"],
}


@dataclass(slots=True)
class RerankCandidate:
    """A candidate document to be scored by the reranker."""

    doc_id: str
    text: str
    initial_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    rerank_score: float = 0.0


class BaseReranker(Protocol):
    """Protocol for two-stage retrieval rerankers."""

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        *,
        top_k: int = 5,
    ) -> list[RerankCandidate]:
        """Score candidate documents against the query and return the top_k."""
        ...


class HeuristicCrossAttentionReranker:
    """High-fidelity heuristic cross-encoder for zero-infra offline & CI execution.

    Evaluates:
    1. Exact query-token coverage with domain weighting.
    2. CJK bigram & Portuguese cross-lingual semantic alignment.
    3. Contiguous substring matches (phrase matching).
    4. BM25 length-normalization penalty.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def compute_score(self, query: str, text: str) -> float:
        if not query or not text:
            return 0.0

        query_lower = query.lower()
        text_lower = text.lower()

        # 1. Base tokenization
        q_tokens = tokenize(query_lower)
        t_tokens = set(tokenize(text_lower))
        if not q_tokens or not t_tokens:
            return 0.0

        # 2. Token overlap & positional phrase matching
        direct_matches = sum(1 for t in q_tokens if t in t_tokens)
        token_coverage = direct_matches / max(len(q_tokens), 1)

        # Exact substring bonus (e.g. "里约", "orders")
        phrase_bonus = 0.0
        for token in q_tokens:
            if len(token) > 1 and token in text_lower:
                phrase_bonus += 0.25

        # 3. Cross-lingual semantic alignment (Chinese -> Portuguese)
        cross_lingual_bonus = 0.0
        for zh_key, pt_words in _CROSS_LINGUAL_SYNONYMS.items():
            if zh_key in query_lower:
                for pt_word in pt_words:
                    if pt_word in text_lower:
                        cross_lingual_bonus += 0.45
                        break

        # 4. Length normalization penalty (avoid overly long noisy text dominating)
        length_penalty = 1.0 / (1.0 + math.log1p(max(len(text_lower) - 50, 0) / 100.0))

        raw_score = (token_coverage * 1.5 + phrase_bonus + cross_lingual_bonus) * length_penalty

        # Sigmoid normalization into [0.0, 1.0]
        sigmoid_score = 1.0 / (1.0 + math.exp(-raw_score * self.temperature))
        return round(sigmoid_score, 4)

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        *,
        top_k: int = 5,
    ) -> list[RerankCandidate]:
        if not candidates:
            return []

        scored: list[RerankCandidate] = []
        for cand in candidates:
            score = self.compute_score(query, cand.text)
            # Combine initial recall score with deep semantic rerank score
            combined = 0.3 * cand.initial_score + 0.7 * score
            cand.rerank_score = round(combined, 4)
            scored.append(cand)

        # Sort by rerank_score descending, break ties deterministically by doc_id
        scored.sort(key=lambda c: (-c.rerank_score, c.doc_id))
        return scored[:top_k]


class ONNXBGEReranker:
    """Production ONNX Runtime Cross-Encoder Reranker.

    Provides fast, lightweight CPU inference (<150MB) without requiring heavy PyTorch or CUDA.
    Executes quantized ONNX graphs (e.g. bge-reranker-base.onnx) on standard CPU cores.
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        tokenizer_path: str | os.PathLike[str] | None = None,
        *,
        max_length: int = 512,
        threads: int = 2,
    ) -> None:
        raw_model = model_path or os.environ.get("INSIGHT_ONNX_RERANKER_PATH")
        raw_tok = tokenizer_path or os.environ.get("INSIGHT_ONNX_TOKENIZER_PATH")
        self.model_path = str(raw_model) if raw_model else None
        self.tokenizer_path = str(raw_tok) if raw_tok else None
        self.max_length = max_length
        self.session: Any = None
        self.tokenizer: Any = None
        self.is_available: bool = False
        self._heuristic = HeuristicCrossAttentionReranker()

        self._init_session(threads=threads)

    def _init_session(self, threads: int = 2) -> None:
        if not self.model_path:
            return

        import pathlib

        model_p = pathlib.Path(self.model_path)
        if not model_p.exists():
            logger.info("ONNX reranker model not found at %s", model_p)
            return

        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = threads
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                str(model_p), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self.is_available = True
            logger.info("ONNXBGEReranker initialized from %s", model_p)
        except Exception as exc:
            logger.warning("Failed to initialize ONNX Runtime session (%s)", exc)
            self.is_available = False

        if self.tokenizer_path and pathlib.Path(self.tokenizer_path).exists():
            try:
                from tokenizers import Tokenizer

                self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
            except Exception:
                self.tokenizer = None

    def _tokenize_pairs(self, pairs: list[tuple[str, str]]) -> dict[str, Any]:
        """Convert (query, text) pairs into input_ids and attention_mask numpy tensors."""
        import numpy as np

        if self.tokenizer is not None:
            encoded = self.tokenizer.encode_batch(pairs)
            input_ids = [enc.ids[: self.max_length] for enc in encoded]
            attention_mask = [enc.attention_mask[: self.max_length] for enc in encoded]
            max_len = max(len(ids) for ids in input_ids) if input_ids else 1
            padded_ids = [ids + [0] * (max_len - len(ids)) for ids in input_ids]
            padded_masks = [m + [0] * (max_len - len(m)) for m in attention_mask]
            return {
                "input_ids": np.array(padded_ids, dtype=np.int64),
                "attention_mask": np.array(padded_masks, dtype=np.int64),
            }

        # Deterministic lightweight cross-encoder tokenizer fallback
        # Constructs [CLS] + hash_tokens(query) + [SEP] + hash_tokens(doc) + [SEP]
        batch_ids = []
        batch_masks = []
        cls_id = 101
        sep_id = 102

        for q, t in pairs:
            q_toks = tokenize(q.lower())[: self.max_length // 3]
            t_toks = tokenize(t.lower())[: (self.max_length * 2) // 3]

            def hash_tok(tok: str) -> int:
                import hashlib

                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:6], 16)
                return (h % 30000) + 1000

            ids = [cls_id] + [hash_tok(w) for w in q_toks] + [sep_id] + [hash_tok(w) for w in t_toks] + [sep_id]
            ids = ids[: self.max_length]
            batch_ids.append(ids)
            batch_masks.append([1] * len(ids))

        max_len = max(len(ids) for ids in batch_ids) if batch_ids else 1
        padded_ids = [ids + [0] * (max_len - len(ids)) for ids in batch_ids]
        padded_masks = [m + [0] * (max_len - len(m)) for m in batch_masks]
        return {
            "input_ids": np.array(padded_ids, dtype=np.int64),
            "attention_mask": np.array(padded_masks, dtype=np.int64),
        }

    def compute_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Compute sigmoid normalized semantic relevance scores using ONNX Runtime."""
        if not pairs:
            return []
        if not self.is_available or self.session is None:
            return [self._heuristic.compute_score(q, t) for q, t in pairs]

        import numpy as np

        inputs = self._tokenize_pairs(pairs)
        # Match input tensor names from the session
        session_input_names = [inp.name for inp in self.session.get_inputs()]
        session_inputs = {}
        for name in session_input_names:
            if name in inputs:
                session_inputs[name] = inputs[name]
            elif "mask" in name.lower():
                session_inputs[name] = inputs["attention_mask"]
            else:
                session_inputs[name] = inputs["input_ids"]

        outputs = self.session.run(None, session_inputs)
        logits = outputs[0]
        raw_scores = logits.flatten() if hasattr(logits, "flatten") else np.array(logits).flatten()
        sigmoid_scores = 1.0 / (1.0 + np.exp(-raw_scores))
        return [float(round(s, 4)) for s in sigmoid_scores]

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        *,
        top_k: int = 5,
    ) -> list[RerankCandidate]:
        if not candidates:
            return []
        if not self.is_available or self.session is None:
            return self._heuristic.rerank(query, candidates, top_k=top_k)

        pairs = [(query, cand.text) for cand in candidates]
        scores = self.compute_scores(pairs)
        scored: list[RerankCandidate] = []
        for cand, score in zip(candidates, scores, strict=True):
            cand.rerank_score = round(0.3 * cand.initial_score + 0.7 * score, 4)
            scored.append(cand)

        scored.sort(key=lambda c: (-c.rerank_score, c.doc_id))
        return scored[:top_k]


class BGEReranker:
    """Production BAAI BGE Cross-Encoder Reranker with zero-infra fallback.

    Automatically resolves inference backends in priority order:
    1. Remote Rerank API (if url is set)
    2. Lightweight ONNX Runtime (if onnx_model_path exists or INSIGHT_ONNX_RERANKER_PATH is set)
    3. Local FlagEmbedding / PyTorch (if INSIGHT_BGE_ENABLED=1)
    4. Deterministic Heuristic Cross-Attention Fallback
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        use_fp16: bool = True,
        url: str | None = None,
        api_key: str | None = None,
        onnx_model_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.model_name = (
            model_name
            or os.environ.get("INSIGHT_RERANKER_MODEL")
            or "BAAI/bge-reranker-v2-m3"
        )
        self.device = device or os.environ.get("INSIGHT_DEVICE", "cpu")
        self.url = url or os.environ.get("INSIGHT_RERANKER_URL")
        self.api_key = api_key or os.environ.get("INSIGHT_RERANKER_API_KEY")
        self.onnx_model_path = str(onnx_model_path) if onnx_model_path else os.environ.get("INSIGHT_ONNX_RERANKER_PATH")
        self.use_fp16 = use_fp16
        self._model: Any = None
        self._onnx_reranker: ONNXBGEReranker | None = None
        self._is_neural = False
        self.backend_type: str = "heuristic"
        self._heuristic = HeuristicCrossAttentionReranker()
        # Monotonic count of rerank() calls that were configured to use the
        # remote reranker but fell back to a weaker backend. The eval harness
        # snapshots this before/after each case to attribute degradation to the
        # exact instance (one rerank call per case, strictly sequential).
        self.remote_fallback_count = 0

        self._init_backend()

    def _init_backend(self) -> None:
        """Attempt to load real backend in order of preference."""
        if self.url:
            self.backend_type = "remote"
            logger.info("BGEReranker configured with remote URL: %s", self.url)
            return

        if self.onnx_model_path:
            self._onnx_reranker = ONNXBGEReranker(self.onnx_model_path)
            if self._onnx_reranker.is_available:
                self.backend_type = "onnx"
                self._is_neural = True
                logger.info("BGEReranker using ONNX Runtime backend: %s", self.onnx_model_path)
                return

        enable_flag = os.environ.get("INSIGHT_BGE_ENABLED", "0").lower() in ("1", "true", "yes")
        if enable_flag:
            try:
                from FlagEmbedding import FlagReranker  # type: ignore[import-not-found]

                logger.info("Loading real BAAI FlagReranker: %s on %s", self.model_name, self.device)
                self._model = FlagReranker(self.model_name, use_fp16=self.use_fp16, device=self.device)
                self._is_neural = True
                self.backend_type = "flag_embedding"
                return
            except Exception as exc:
                logger.warning(
                    "FlagEmbedding unavailable (%s); falling back to heuristic", exc
                )
                self._model = None
                self._is_neural = False

        self.backend_type = "heuristic"
        logger.info("BGEReranker initialized with heuristic fallback")

    @property
    def is_neural(self) -> bool:
        """Whether neural Cross-Encoder is active (either remote API, ONNX, or FlagEmbedding)."""
        return self._is_neural or bool(self.url) or (self._onnx_reranker is not None and self._onnx_reranker.is_available)

    def _call_remote_rerank(
        self, query: str, candidates: list[RerankCandidate]
    ) -> list[float] | None:
        """Call standard TEI / Jina / BAAI / SiliconFlow compatible /v1/rerank endpoint."""
        from insight_agent.app.http_util import post_json_with_retry

        if getattr(self, "_remote_disabled_until", 0) > time.time():
            return None

        if not self.url:
            return None
        target_url = self.url.rstrip("/")
        if not (target_url.endswith("/rerank") or target_url.endswith("/rerank/")):
            target_url = f"{target_url}/rerank"

        docs = [cand.text for cand in candidates]
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": docs,
        }
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) InsightAgent/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data, exc = post_json_with_retry(
            target_url,
            payload,
            headers,
            timeout=8.0,
            base_delay=2.0,
            max_delay=60.0,
        )
        if data is not None:
            items = data.get("results") or data.get("data") or []
            scores = [0.0] * len(candidates)
            for item in items:
                idx = item.get("index", 0)
                score = float(item.get("relevance_score", item.get("score", 0.0)))
                if 0 <= idx < len(scores):
                    scores[idx] = score
            return scores

        err_str = str(exc)
        if any(c in err_str for c in ("401", "402", "403")):
            self._remote_disabled_until = time.time() + 300
            logger.warning("Remote rerank endpoint billing/auth issue (%s); circuit breaker enabled for 300s", exc)
        else:
            logger.warning("Remote rerank call failed after retries (%s); falling back to local heuristic", exc)
        return None

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        *,
        top_k: int = 5,
    ) -> list[RerankCandidate]:
        if not candidates:
            return []

        # 1. Try remote Rerank API if URL configured
        if self.url:
            remote_scores = self._call_remote_rerank(query, candidates)
            if remote_scores:
                scored: list[RerankCandidate] = []
                for cand, r_score in zip(candidates, remote_scores, strict=True):
                    cand.rerank_score = round(0.3 * cand.initial_score + 0.7 * r_score, 4)
                    scored.append(cand)
                scored.sort(key=lambda c: (-c.rerank_score, c.doc_id))
                return scored[:top_k]
            # Remote backend was configured but returned nothing (timeout /
            # disconnect / circuit-breaker). This call is about to degrade to a
            # weaker backend; record it so the harness can mark this instance.
            self.remote_fallback_count += 1

        # 2. Try ONNX Runtime backend if active
        if self.backend_type == "onnx" and self._onnx_reranker is not None and self._onnx_reranker.is_available:
            return self._onnx_reranker.rerank(query, candidates, top_k=top_k)

        # 3. Try local FlagEmbedding model if loaded
        if self._is_neural and self._model is not None:
            try:
                pairs = [[query, cand.text] for cand in candidates]
                scores = self._model.compute_score(pairs, normalize=True)
                if isinstance(scores, float):
                    scores = [scores]

                scored_local: list[RerankCandidate] = []
                for cand, score in zip(candidates, scores, strict=True):
                    norm_score = float(score)
                    cand.rerank_score = round(0.3 * cand.initial_score + 0.7 * norm_score, 4)
                    scored_local.append(cand)

                scored_local.sort(key=lambda c: (-c.rerank_score, c.doc_id))
                return scored_local[:top_k]
            except Exception as exc:
                logger.warning("Neural reranker execution error: %s; falling back to heuristic", exc)

        # 4. Fallback to heuristic cross-attention scorer
        return self._heuristic.rerank(query, candidates, top_k=top_k)
