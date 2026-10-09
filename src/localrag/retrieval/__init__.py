"""Retrieval strategies behind one registry.

A strategy is a function ``(ctx, query, config, k) -> [(Document, score)]``, best first. Payload
filters and LLM re-ranking compose with every strategy, so adding one (say, a ColBERT or SPLADE
retriever) is a single decorated function::

    @register("my_strategy")
    def my_strategy(ctx, query, config, k):
        ...
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from localrag.config import RetrievalConfig
from localrag.retrieval.bm25 import BM25Index
from localrag.retrieval.fusion import reciprocal_rank_fusion
from localrag.retrieval.rerank import LLMReranker, Reranker, parse_score
from localrag.store import metadata_filter

__all__ = [
    "BM25Index",
    "LLMReranker",
    "Reranker",
    "RetrievalContext",
    "parse_score",
    "reciprocal_rank_fusion",
    "register",
    "retrieve",
    "strategies",
]

Scored = list[tuple[Document, float]]


@dataclass
class RetrievalContext:
    """What a strategy may use. ``bm25`` is built lazily because only hybrid needs it."""

    vectorstore: QdrantVectorStore
    load_documents: Callable[[], list[Document]]
    _bm25: BM25Index | None = field(default=None, repr=False)

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index(self.load_documents())
        return self._bm25


Strategy = Callable[[RetrievalContext, str, RetrievalConfig, int], Scored]
_REGISTRY: dict[str, Strategy] = {}


def register(name: str) -> Callable[[Strategy], Strategy]:
    def decorator(fn: Strategy) -> Strategy:
        _REGISTRY[name] = fn
        return fn

    return decorator


def strategies() -> list[str]:
    return sorted(_REGISTRY)


@register("dense")
def dense(ctx: RetrievalContext, query: str, config: RetrievalConfig, k: int) -> Scored:
    """Cosine kNN over the embeddings."""
    return ctx.vectorstore.similarity_search_with_score(
        query, k=k, filter=metadata_filter(config.filters)
    )


@register("mmr")
def mmr(ctx: RetrievalContext, query: str, config: RetrievalConfig, k: int) -> Scored:
    """Maximal marginal relevance: from ``fetch_k`` nearest candidates, greedily pick chunks
    that are relevant to the query *and* dissimilar to those already picked."""
    vector = ctx.vectorstore.embeddings.embed_query(query)
    return ctx.vectorstore.max_marginal_relevance_search_with_score_by_vector(
        vector,
        k=k,
        fetch_k=max(config.fetch_k, k),
        lambda_mult=config.lambda_mult,
        filter=metadata_filter(config.filters),
    )


@register("hybrid")
def hybrid(ctx: RetrievalContext, query: str, config: RetrievalConfig, k: int) -> Scored:
    """BM25 and dense rankings (``fetch_k`` deep each) fused with weighted RRF.

    The returned score is the fused RRF score, not a similarity.
    """
    depth = max(config.fetch_k, k)
    dense_hits = dense(ctx, query, config, depth)
    lexical_hits = ctx.bm25.search(query, depth, config.filters)
    by_id: dict[str, Document] = {}
    for doc, _ in [*dense_hits, *lexical_hits]:
        by_id.setdefault(doc.metadata["_id"], doc)
    fused = reciprocal_rank_fusion(
        [
            [d.metadata["_id"] for d, _ in lexical_hits],
            [d.metadata["_id"] for d, _ in dense_hits],
        ],
        weights=[config.bm25_weight, 1.0 - config.bm25_weight],
        c=config.rrf_c,
    )
    return [(by_id[pid], score) for pid, score in fused[:k]]


def retrieve(
    ctx: RetrievalContext,
    query: str,
    config: RetrievalConfig,
    reranker: Reranker | None = None,
    rerank_question: str | None = None,
) -> Scored:
    """Run ``config.strategy`` and, if ``config.rerank``, re-rank its candidates.

    Args:
        ctx: Vector store and BM25 access.
        query: Text to search with (possibly a rewritten query).
        config: Strategy and parameters.
        reranker: Required when ``config.rerank`` is set.
        rerank_question: What the re-ranker judges relevance against; defaults to ``query``.
            The engine passes the user's original question here.
    """
    try:
        strategy = _REGISTRY[config.strategy]
    except KeyError:
        raise ValueError(f"Unknown strategy {config.strategy!r}; have {strategies()}") from None
    if not config.rerank:
        return strategy(ctx, query, config, config.k)
    if reranker is None:
        raise ValueError("config.rerank is set but no reranker was provided")
    candidates = strategy(ctx, query, config, max(config.rerank_candidates, config.k))
    reranked = reranker.rerank(rerank_question or query, [doc for doc, _ in candidates])
    return reranked[: config.k]
