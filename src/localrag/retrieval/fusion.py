"""Reciprocal-rank fusion (Cormack, Clarke & Büttcher, SIGIR 2009)."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TypeVar

K = TypeVar("K", bound=Hashable)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[K]],
    weights: Sequence[float] | None = None,
    c: int = 60,
) -> list[tuple[K, float]]:
    """Fuse several rankings into one: ``score(d) = Σ_i w_i / (c + rank_i(d))``.

    RRF uses only ranks, never raw scores, so it can combine lists whose scores live on
    incomparable scales (BM25 is unbounded; cosine similarity is in [-1, 1]). The constant ``c``
    damps the advantage of the very top ranks, so a document ranked well by several lists beats
    one ranked first by a single list.

    Args:
        ranked_lists: Each list holds item keys, best first. Ranks are 1-based. A key repeated
            within one list counts only at its best rank.
        weights: One weight per list; defaults to equal weights of 1.
        c: Smoothing constant.

    Returns:
        ``(key, score)`` pairs, best first. Ties keep the order in which keys were first seen,
        so the result is deterministic.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must have one entry per ranked list")
    if c < 0:
        raise ValueError("c must be non-negative")

    scores: dict[K, float] = {}
    for items, weight in zip(ranked_lists, weights, strict=True):
        seen: set[K] = set()
        for rank, key in enumerate(items, start=1):
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + weight / (c + rank)
    # sorted() is stable, and dict order is first-seen order, so ties are deterministic.
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
