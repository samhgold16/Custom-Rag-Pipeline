import pytest
from langchain_core.documents import Document

from localrag import RetrievalConfig
from localrag.retrieval import register, retrieve, strategies
from localrag.retrieval.bm25 import BM25Index, matches, tokenize


def test_tokenizer_ignores_case_and_punctuation():
    assert tokenize("Frustration. AGGRESSIVE steps, quantum-AI!") == [
        "frustration",
        "aggressive",
        "steps",
        "quantum",
        "ai",
    ]


def test_bm25_finds_the_literal_term_and_honours_filters():
    docs = [
        Document(page_content="Frustration with competition.", metadata={"year": 2032, "_id": "a"}),
        Document(page_content="Record growth and optimism.", metadata={"year": 2035, "_id": "b"}),
        Document(page_content="We share your frustration.", metadata={"year": 2033, "_id": "c"}),
    ]
    index = BM25Index(docs)
    assert {d.metadata["_id"] for d, _ in index.search("frustration", k=5)} == {"a", "c"}
    assert [
        d.metadata["_id"] for d, _ in index.search("frustration", k=5, filters={"year": 2032})
    ] == ["a"]
    assert index.search("unrelated", k=5) == []
    assert BM25Index([]).search("x", k=3) == []


def test_python_filter_mirrors_payload_filter():
    assert matches({"year": 2032}, {"year": [2031, 2032]})
    assert not matches({"year": 2030}, {"year": 2032})
    assert not matches({}, {"year": 2032})


@pytest.mark.parametrize("strategy", ["dense", "mmr", "hybrid"])
def test_every_strategy_respects_the_year_filter(engine, strategy):
    config = RetrievalConfig(strategy=strategy, k=4, filters={"year": 2032})
    sources = engine.retrieve("error correction frustration", config)
    assert sources and {s.metadata["year"] for s in sources} == {2032}


@pytest.mark.parametrize("strategy", ["dense", "mmr", "hybrid"])
def test_strategies_return_at_most_k_unique_chunks(engine, strategy):
    sources = engine.retrieve("qubits", RetrievalConfig(strategy=strategy, k=3))
    assert 0 < len(sources) <= 3
    assert len({s.metadata["content_hash"] for s in sources}) == len(sources)
    assert [s.rank for s in sources] == list(range(1, len(sources) + 1))


def test_filter_can_return_fewer_than_k(engine):
    sources = engine.retrieve("anything", RetrievalConfig(k=4, filters={"year": 2030}))
    assert len(sources) == 1


def test_custom_strategies_plug_into_the_registry(engine):
    @register("first_only")
    def first_only(ctx, query, config, k):
        doc = ctx.load_documents()[0]
        return [(doc, 1.0)]

    assert "first_only" in strategies()
    sources = engine.retrieve("q", RetrievalConfig(strategy="first_only"))
    assert sources[0].source == "2030.txt"


def test_unknown_strategy_and_missing_reranker_are_errors(engine):
    ctx = engine._ctx()
    with pytest.raises(ValueError, match="Unknown strategy"):
        retrieve(ctx, "q", RetrievalConfig(strategy="nope"))
    with pytest.raises(ValueError, match="reranker"):
        retrieve(ctx, "q", RetrievalConfig(rerank=True))
