import json

import pytest

from localrag import RAGEngine, Settings
from tests.conftest import build_engine, make_llm


def test_ask_returns_an_auditable_answer(engine):
    answer = engine.ask("Who is the CEO?", guardrail=False)
    assert answer.text == "QuantumMind's CEO is Dr. Quentin Bohr."
    assert not answer.abstained
    assert answer.sources and answer.sources[0].rank == 1
    assert {"retrieve", "generate", "total"} <= answer.timings.keys()
    assert answer.context_format == "header"
    payload = json.loads(json.dumps(answer.to_dict()))
    assert payload["sources"][0]["metadata"]["year"] in range(2030, 2034)
    assert "_id" not in payload["sources"][0]["metadata"]


def test_abstention_is_flagged(engine):
    engine._chat_model = make_llm("I don't know the answer.")
    assert engine.ask("Anything?", guardrail=False).abstained


def test_querying_before_ingest_is_a_clear_error(corpus):
    engine = build_engine(corpus)
    with pytest.raises(LookupError, match="rag ingest"):
        engine.ask("q", guardrail=False)


def test_variant_shares_models_and_client(engine):
    variant = engine.variant(header_fields=("year",))
    assert variant.store.client is engine.store.client
    assert variant.chat_model is engine.chat_model
    assert variant.collection != engine.collection


def test_preflight_is_skipped_when_models_are_injected(engine):
    engine.preflight()


def test_default_provider_is_ollama_and_lazy():
    engine = RAGEngine(Settings(qdrant_path=":memory:"))
    assert type(engine.chat_model).__name__ == "ChatOllama"
    assert engine.chat_model.reasoning is False
    assert type(engine.embeddings).__name__ == "OllamaEmbeddings"
