import logging

import pytest
from langchain_core.documents import Document

from localrag.retrieval.rerank import LLMReranker, parse_score
from tests.conftest import make_llm


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("8", 8.0),
        ("Score: 8/10", 8.0),  # the original parser read this as 810
        ("8 out of 10", 8.0),  # ... and this too
        ("7.5.", 7.5),
        ("I'd say 9", 9.0),
        ("12", 10.0),
        ("N/A", None),
        ("", None),
    ],
)
def test_parse_score(reply, expected):
    assert parse_score(reply) == expected


def test_reranker_sorts_by_llm_score_and_falls_back_to_zero(caplog):
    docs = [Document(page_content=t) for t in ("low", "unparseable", "high")]
    reranker = LLMReranker(make_llm("2", "no idea", "Score: 9/10"))
    with caplog.at_level(logging.WARNING):
        ranked = reranker.rerank("q", docs)
    assert [(d.page_content, s) for d, s in ranked] == [
        ("high", 9.0),
        ("low", 2.0),
        ("unparseable", 0.0),
    ]
    assert "no score" in caplog.text


def test_rerank_keeps_top_k_and_judges_the_original_question(engine):
    from localrag import RetrievalConfig

    engine._chat_model = make_llm("1", "9", "3", "5")
    sources = engine.retrieve("qubits", RetrievalConfig(k=2, rerank=True, rerank_candidates=4))
    assert [s.score for s in sources] == [9.0, 5.0]
