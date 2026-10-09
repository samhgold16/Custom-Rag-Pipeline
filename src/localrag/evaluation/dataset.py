"""Gold question sets: one JSON object per line, validated on load."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

QUESTION_TYPES = ("factual", "summary", "unanswerable")


@dataclass(frozen=True)
class EvalItem:
    """One gold question.

    Attributes:
        id: Unique identifier.
        question: Text sent to the engine.
        type: ``factual``, ``summary``, or ``unanswerable``.
        gold_sources: Source paths (relative to the data directory) that contain the answer.
            Empty means any source is acceptable, and retrieval metrics skip the item.
        must_include: Every term must appear in the answer; ``"a|b"`` accepts either.
        must_not_include: No term may appear in the answer.
        notes: Why the item exists or what makes it hard.
    """

    id: str
    question: str
    type: str
    gold_sources: tuple[str, ...] = ()
    must_include: tuple[str, ...] = ()
    must_not_include: tuple[str, ...] = ()
    notes: str = ""

    @property
    def answerable(self) -> bool:
        return self.type != "unanswerable"


def load_dataset(path: str | Path) -> list[EvalItem]:
    """Read and validate a JSONL dataset.

    Raises:
        ValueError: on malformed lines, unknown types, duplicate ids, or answerable items
            without any ``must_include`` term (they could never be scored).
    """
    items: list[EvalItem] = []
    seen: set[str] = set()
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            item = EvalItem(
                id=str(row["id"]),
                question=str(row["question"]),
                type=str(row["type"]),
                gold_sources=tuple(row.get("gold_sources", [])),
                must_include=tuple(row.get("must_include", [])),
                must_not_include=tuple(row.get("must_not_include", [])),
                notes=str(row.get("notes", "")),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{lineno}: invalid row ({exc})") from exc
        if item.type not in QUESTION_TYPES:
            raise ValueError(f"{path}:{lineno}: type must be one of {QUESTION_TYPES}")
        if item.id in seen:
            raise ValueError(f"{path}:{lineno}: duplicate id {item.id!r}")
        if item.answerable and not item.must_include:
            raise ValueError(f"{path}:{lineno}: answerable item {item.id!r} needs must_include")
        seen.add(item.id)
        items.append(item)
    if not items:
        raise ValueError(f"{path}: dataset is empty")
    return items
