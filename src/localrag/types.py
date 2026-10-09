"""Typed results returned by the engine. Every object serializes to plain JSON via ``to_dict``."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceChunk:
    """One retrieved chunk, as shown to the generator and returned to the caller.

    Attributes:
        source: Document path relative to the data directory.
        text: Chunk text exactly as stored (including any ingest-time header).
        rank: 1-based position in the final context.
        score: Strategy-specific score: cosine similarity (dense/MMR), fused RRF score
            (hybrid), or the LLM's 0-10 relevance score (re-ranked).
        metadata: Payload metadata (extracted fields, ``chunk_index``, ``content_hash``).
    """

    source: str
    text: str
    rank: int
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def snippet(self, width: int = 160) -> str:
        """First ``width`` characters on one line, for tables and logs."""
        flat = " ".join(self.text.split())
        return flat if len(flat) <= width else flat[: width - 1] + "…"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GuardrailDecision:
    """Outcome of the scope guardrail for one question.

    Attributes:
        action: ``"abstain"`` (answered without calling the LLM), ``"filter"`` (scoped the
            search to the values named in the question), or ``"pass"``.
        field: Metadata field that was checked.
        requested: Values found in the question.
        available: Values present in the collection.
        reason: Human-readable explanation.
    """

    action: str
    field: str
    requested: tuple[int, ...]
    available: tuple[int, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Answer:
    """A generated answer plus everything needed to audit it.

    Attributes:
        question: The user's original question (always what the generator answered).
        text: The answer.
        sources: Chunks that were placed in the prompt, in order.
        collection: Qdrant collection searched.
        retrieval: Label of the retrieval configuration used.
        context_format: How the chunks were rendered for the generator.
        abstained: Whether the answer declines to answer.
        rewritten_query: Search query produced by the rewriter, if enabled.
        guardrail: Guardrail decision, if the guardrail ran.
        timings: Seconds per stage: ``rewrite``, ``retrieve`` (including any re-ranking),
            ``generate``, and ``total``.
    """

    question: str
    text: str
    sources: list[SourceChunk]
    collection: str
    retrieval: str
    context_format: str
    abstained: bool
    rewritten_query: str | None = None
    guardrail: GuardrailDecision | None = None
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "abstained": self.abstained,
            "collection": self.collection,
            "retrieval": self.retrieval,
            "context_format": self.context_format,
            "rewritten_query": self.rewritten_query,
            "guardrail": self.guardrail.to_dict() if self.guardrail else None,
            "timings": {k: round(v, 4) for k, v in self.timings.items()},
            "sources": [s.to_dict() for s in self.sources],
        }


@dataclass(frozen=True)
class IngestReport:
    """What ``ingest`` did.

    Attributes:
        collection: Collection written.
        documents: Files loaded.
        chunks: Chunks in the collection after the run.
        added: Chunks embedded and upserted this run.
        removed: Stale chunks deleted this run.
        rebuilt: Whether the collection was dropped and recreated.
        vector_size: Embedding dimensionality (probed, not configured).
        field_values: Distinct values seen per extracted metadata field.
        seconds: Wall-clock duration.
    """

    collection: str
    documents: int
    chunks: int
    added: int
    removed: int
    rebuilt: bool
    vector_size: int
    field_values: dict[str, list[Any]] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def unchanged(self) -> bool:
        """True when the run embedded nothing and deleted nothing."""
        return self.added == 0 and self.removed == 0 and not self.rebuilt

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "unchanged": self.unchanged}
