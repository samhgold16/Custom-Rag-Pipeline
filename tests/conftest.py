"""Shared fixtures. Everything here is offline: fake models and in-memory Qdrant."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from qdrant_client import QdrantClient

from localrag import RAGEngine, Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_CORPUS = REPO_ROOT / "data" / "quantummind"

# Real answers produced by the original pipeline (baseline run).
BASELINE_FABRICATION = (
    "In 2023, QuantumMind reached 10,000 qubits while maintaining unprecedented coherence times."
)
BASELINE_ABSTENTIONS = (
    "I don't know. The provided context doesn't mention any CEO scandal.",
    "I don't know the answer.",
)


class CountingChatModel(FakeListChatModel):
    """Scripted replies that also count how often the model was called."""

    calls: int = 0

    def _call(self, *args, **kwargs):  # type: ignore[override]
        self.calls += 1
        return super()._call(*args, **kwargs)


def make_llm(*responses: str) -> CountingChatModel:
    return CountingChatModel(responses=list(responses) or ["stub answer"])


@pytest.fixture
def embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=32)


@pytest.fixture
def llm() -> CountingChatModel:
    return make_llm("QuantumMind's CEO is Dr. Quentin Bohr.")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A tiny year-named corpus that is cheap to mutate per test."""
    root = tmp_path / "letters"
    root.mkdir()
    texts = {
        "2030.txt": "We built our first 100-qubit processor. Partnerships grew.",
        "2031.txt": "We reached 250 qubits and launched a quantum cloud service.",
        "2032.txt": "Error correction struggles caused frustration. Aggressive steps are planned.",
        "2033.txt": "We had to downsize our workforce. The stock hit an all-time low.",
    }
    for name, text in texts.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


def build_engine(data_dir: Path, llm=None, embeddings=None, **settings) -> RAGEngine:
    return RAGEngine(
        Settings(data_dir=str(data_dir), qdrant_path=":memory:", **settings),
        chat_model=llm or make_llm(),
        embeddings=embeddings or DeterministicFakeEmbedding(size=32),
        client=QdrantClient(":memory:"),
    )


@pytest.fixture
def engine(corpus: Path, llm, embeddings) -> RAGEngine:
    eng = build_engine(corpus, llm, embeddings, chunk_size=200, chunk_overlap=20)
    eng.ingest()
    return eng
