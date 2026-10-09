import pytest
from langchain_core.documents import Document

from localrag.generation import (
    BASE_SYSTEM_PROMPT,
    CITATION_RULE,
    AnswerGenerator,
    format_context,
    is_abstention,
    system_prompt,
)
from tests.conftest import BASELINE_ABSTENTIONS, BASELINE_FABRICATION, make_llm


@pytest.mark.parametrize("text", BASELINE_ABSTENTIONS)
def test_real_baseline_abstentions_are_detected(text):
    assert is_abstention(text)


@pytest.mark.parametrize(
    "text",
    [
        BASELINE_FABRICATION,
        "QuantumMind's first quantum processor achieved 100 qubits.",
        "The CEO of QuantumMind is Dr. Quentin Bohr.",
    ],
)
def test_real_answers_are_not_abstentions(text):
    assert not is_abstention(text)


def test_curly_apostrophes_are_normalised():
    assert is_abstention("I don’t know.")


DOCS = [
    Document(page_content="Body A", metadata={"source": "2030.txt", "year": 2030}),
    Document(
        page_content="Metadata: year=2031\nBody B", metadata={"source": "2031.txt", "year": 2031}
    ),
]


def test_plain_context_is_the_raw_text():
    assert format_context(DOCS, "plain") == "Body A\n\nMetadata: year=2031\nBody B"


def test_header_context_adds_the_header_once():
    out = format_context(DOCS, "header", ["year"])
    assert out == "Metadata: year=2030\nBody A\n\nMetadata: year=2031\nBody B"


def test_tagged_context_labels_each_passage():
    out = format_context(DOCS[:1], "tagged", ["year"])
    assert out == "[source: 2030.txt | year: 2030]\nBody A"


def test_only_tagged_context_asks_for_citations():
    assert CITATION_RULE not in system_prompt("plain")
    assert CITATION_RULE not in system_prompt("header")
    assert CITATION_RULE in system_prompt("tagged")
    assert system_prompt("plain") == BASE_SYSTEM_PROMPT + "\n\n{context}"


def test_generator_runs_the_lcel_chain():
    assert AnswerGenerator(make_llm("  Answer.  ")).generate("q", DOCS, "plain") == "Answer."
