"""In-memory BM25 over the chunks stored in a collection."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens.

    The default ``str.split()`` tokenizer is case-sensitive and keeps punctuation, so
    ``"Frustration."`` would never match a query for ``frustration``.
    """
    return _TOKEN.findall(text.lower())


def matches(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Python mirror of ``store.metadata_filter``, so BM25 honours the same payload filters."""
    for name, wanted in filters.items():
        value = metadata.get(name)
        if isinstance(wanted, list | tuple | set):
            if value not in wanted:
                return False
        elif value != wanted:
            return False
    return True


class BM25Index:
    """Okapi BM25 over a fixed list of chunks.

    IDF statistics come from the whole collection; filters are applied to the scored results.
    That keeps one index per collection instead of one per filter value.
    """

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self._bm25 = BM25Okapi([tokenize(d.page_content) for d in documents]) if documents else None

    def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> list[tuple[Document, float]]:
        """Top ``k`` chunks by BM25 score, best first. Zero-score chunks are dropped."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            (
                (doc, float(score))
                for doc, score in zip(self.documents, scores, strict=True)
                if score > 0 and (not filters or matches(doc.metadata, filters))
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:k]
