"""Deterministic metrics: no LLM judge, so every score can be recomputed and audited by hand."""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from localrag.evaluation.dataset import EvalItem

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")


def normalize(text: str) -> str:
    """Lowercase, straighten quotes, and drop thousands separators (``10,000`` → ``10000``)."""
    text = text.lower().replace("’", "'").replace("‘", "'")
    return _THOUSANDS.sub("", text)


def term_matches(text: str, term: str) -> bool:
    """Whether any ``|``-separated alternative of ``term`` occurs in normalised ``text``.

    Alternatives match at a word start, so stems work (``downsiz`` matches "downsized").
    Numeric alternatives must also end at a non-digit, so ``100`` does not match ``10000``.
    """
    norm = normalize(text)
    for alt in term.split("|"):
        alt = normalize(alt.strip())
        tail = r"(?!\d)" if alt[-1:].isdigit() else ""
        if re.search(rf"(?<!\w){re.escape(alt)}{tail}", norm):
            return True
    return False


def content_correct(item: EvalItem, answer: str) -> bool:
    """All ``must_include`` terms present and no ``must_not_include`` term present."""
    return all(term_matches(answer, t) for t in item.must_include) and not any(
        term_matches(answer, t) for t in item.must_not_include
    )


def hit_at_k(retrieved: Sequence[str], gold: Sequence[str]) -> bool | None:
    """Any retrieved source is a gold source. ``None`` when the item has no gold sources."""
    if not gold:
        return None
    return any(src in gold for src in retrieved)


def reciprocal_rank(retrieved: Sequence[str], gold: Sequence[str]) -> float | None:
    """1 / rank of the first gold source (0 if none retrieved); ``None`` without gold sources."""
    if not gold:
        return None
    for rank, src in enumerate(retrieved, start=1):
        if src in gold:
            return 1.0 / rank
    return 0.0


@dataclass
class ItemResult:
    """One (config, question) outcome."""

    config: str
    item_id: str
    type: str
    question: str
    answer: str
    abstained: bool
    correct: bool
    sources: list[str]
    hit: bool | None
    rr: float | None
    seconds: float
    guardrail: str | None = None
    rewritten_query: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_item(
    config: str,
    item: EvalItem,
    answer: str,
    abstained: bool,
    sources: list[str],
    seconds: float,
    guardrail: str | None = None,
    rewritten_query: str | None = None,
) -> ItemResult:
    """Score one answer.

    An answerable item is correct when its content matches. An unanswerable item is correct
    when the answer abstains; anything else counts as a hallucination.
    """
    correct = content_correct(item, answer) if item.answerable else abstained
    return ItemResult(
        config=config,
        item_id=item.id,
        type=item.type,
        question=item.question,
        answer=answer,
        abstained=abstained,
        correct=correct,
        sources=sources,
        hit=hit_at_k(sources, item.gold_sources),
        rr=reciprocal_rank(sources, item.gold_sources),
        seconds=seconds,
        guardrail=guardrail,
        rewritten_query=rewritten_query,
    )


@dataclass
class Summary:
    """Aggregate metrics for one config. Rates are in [0, 1]; ``None`` when undefined."""

    config: str
    n: int
    answerable: int
    unanswerable: int
    accuracy: float
    answer_accuracy: float | None
    abstention_accuracy: float | None
    hallucination_rate: float | None
    false_abstention_rate: float | None
    hit_at_k: float | None
    mrr: float | None
    p50_seconds: float
    p95_seconds: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean(values: Sequence[float | bool]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: Sequence[float], q: float) -> float:
    """Inclusive-method percentile; 0 for an empty list."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


def summarize(config: str, results: Sequence[ItemResult]) -> Summary:
    answerable = [r for r in results if r.type != "unanswerable"]
    unanswerable = [r for r in results if r.type == "unanswerable"]
    hits = [r.hit for r in results if r.hit is not None]
    rrs = [r.rr for r in results if r.rr is not None]
    latencies = [r.seconds for r in results]
    return Summary(
        config=config,
        n=len(results),
        answerable=len(answerable),
        unanswerable=len(unanswerable),
        accuracy=_mean([r.correct for r in results]) or 0.0,
        answer_accuracy=_mean([r.correct for r in answerable]),
        abstention_accuracy=_mean([r.abstained for r in unanswerable]),
        hallucination_rate=_mean([not r.abstained for r in unanswerable]),
        false_abstention_rate=_mean([r.abstained for r in answerable]),
        hit_at_k=_mean(hits),
        mrr=_mean(rrs),
        p50_seconds=percentile(latencies, 50),
        p95_seconds=percentile(latencies, 95),
    )
