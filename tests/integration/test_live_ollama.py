"""End-to-end checks against a real Ollama. Run with ``uv run pytest -m integration``."""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from localrag import RAGEngine, Settings
from localrag.providers import check_ollama
from tests.conftest import DEMO_CORPUS

pytestmark = pytest.mark.integration

SETTINGS = Settings(data_dir=str(DEMO_CORPUS), qdrant_path=":memory:")
U02 = "what was the max qubits that Quantum mind reached in 2023?"


@pytest.fixture(scope="module")
def engine() -> RAGEngine:
    health = check_ollama(SETTINGS.ollama_host, [SETTINGS.chat_model, SETTINGS.embedding_model])
    if not health.ok:
        pytest.skip(f"Ollama not ready at {SETTINGS.ollama_host}: {health.error or health.missing}")
    eng = RAGEngine(SETTINGS, client=QdrantClient(":memory:"))
    eng.ingest()
    return eng


def test_ceo_question_end_to_end(engine):
    answer = engine.ask("Who is the CEO of QuantumMind?")
    assert "bohr" in answer.text.lower()


def test_headline_baseline_is_recorded(engine, record_property):
    """The original pipeline fabricates here. Record, don't assert: model drift may change it."""
    answer = engine.ask(U02, context_format="plain", guardrail=False)
    record_property("plain_answer", answer.text)
    record_property("plain_abstained", answer.abstained)


def test_year_in_context_prevents_the_fabrication(engine):
    answer = engine.ask(U02, context_format="header", guardrail=False)
    assert answer.abstained or "10,000" not in answer.text


def test_year_header_in_embeddings_prevents_the_fabrication(engine):
    variant = engine.variant(header_fields=("year",))
    variant.ingest()
    answer = variant.ask(U02, context_format="plain", guardrail=False)
    assert answer.abstained or "10,000" not in answer.text


def test_guardrail_abstains_deterministically(engine):
    answer = engine.ask(U02)
    assert answer.abstained and answer.guardrail.action == "abstain"
