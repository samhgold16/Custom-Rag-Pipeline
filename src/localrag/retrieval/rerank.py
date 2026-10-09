"""Pointwise LLM re-ranking.

The generator model scores each (question, chunk) pair on 0-10. It reads both texts together,
which a bi-encoder cannot, at the cost of one LLM call per candidate. A cross-encoder such as
``BAAI/bge-reranker-base`` would be faster and better calibrated, but pulls in torch (~2 GB); the
``Reranker`` protocol is the seam for adding one.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

RERANK_PROMPT = (
    "Rate how well the document answers the question, on a scale of 0 to 10.\n"
    "10 means the document contains the answer. 0 means it is unrelated.\n"
    "Reply with the number only, nothing else.\n\n"
    "Question: {question}\n\n"
    "Document: {document}\n\n"
    "Score:"
)

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


class Reranker(Protocol):
    def rerank(self, question: str, documents: list[Document]) -> list[tuple[Document, float]]: ...


def parse_score(reply: str) -> float | None:
    """First number in ``reply``, clamped to [0, 10]; ``None`` if there is no number.

    Taking the *first* number matters: concatenating every digit turns ``"8/10"`` into 810.
    """
    match = _NUMBER.search(reply)
    if match is None:
        return None
    return min(max(float(match.group()), 0.0), 10.0)


class LLMReranker:
    """Scores each candidate with one chat-model call and sorts by score (stable on ties)."""

    def __init__(self, llm: BaseChatModel, prompt: str = RERANK_PROMPT) -> None:
        self.llm = llm
        self.prompt = prompt

    def rerank(self, question: str, documents: list[Document]) -> list[tuple[Document, float]]:
        scored: list[tuple[Document, float]] = []
        for doc in documents:
            reply = str(
                self.llm.invoke(
                    self.prompt.format(question=question, document=doc.page_content)
                ).content
            )
            score = parse_score(reply)
            if score is None:
                logger.warning("Re-ranker reply had no score, using 0: %r", reply[:80])
                score = 0.0
            scored.append((doc, score))
        return sorted(scored, key=lambda pair: pair[1], reverse=True)
