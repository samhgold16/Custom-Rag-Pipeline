import pytest

from localrag.query import QueryRewriter, extract_years, guardrail_filter, scope_guardrail
from tests.conftest import make_llm

CORPUS_YEARS = range(2030, 2036)


@pytest.mark.parametrize(
    ("text", "years"),
    [
        ("what was the max qubits that Quantum mind reached in 2023?", [2023]),
        ("In 2033, the stock fell; by 2034 it recovered. 2033 again.", [2033, 2034]),
        ("a stable 1000-qubit system", []),
        ("scaled to 10,000 qubits", []),
        ("version 3.2031 and 12030", []),
    ],
)
def test_extract_years(text, years):
    assert extract_years(text) == years


def test_guardrail_passes_questions_without_years():
    assert scope_guardrail("Who is the CEO?", "year", CORPUS_YEARS).action == "pass"


def test_guardrail_abstains_outside_the_corpus():
    decision = scope_guardrail("max qubits in 2023?", "year", CORPUS_YEARS)
    assert decision.action == "abstain"
    assert (
        decision.reason == "The indexed documents cover year 2030–2035; there is nothing for 2023."
    )


def test_guardrail_filters_to_years_inside_the_corpus():
    decision = scope_guardrail("Compare 2023 and 2032.", "year", CORPUS_YEARS)
    assert decision.action == "filter"
    assert guardrail_filter(decision) == {"year": [2032]}


def test_out_of_scope_question_never_reaches_the_llm(engine):
    answer = engine.ask(
        "what was the max qubits that Quantum mind reached in 2023?", guardrail=True
    )
    assert answer.abstained and answer.sources == []
    assert answer.guardrail.action == "abstain"
    assert engine.chat_model.calls == 0


def test_in_scope_year_becomes_a_payload_filter(engine):
    answer = engine.ask("What happened in 2032?", guardrail=True)
    assert {s.metadata["year"] for s in answer.sources} == {2032}
    assert "year:[2032]" in answer.retrieval


def test_rewrite_is_used_for_search_but_the_original_is_answered(engine):
    engine._chat_model = make_llm("QuantumMind workforce reduction measures", "They downsized.")
    answer = engine.ask("what did they do to staff?", rewrite=True, guardrail=False)
    assert answer.rewritten_query == "QuantumMind workforce reduction measures"
    assert answer.question == "what did they do to staff?"
    assert answer.text == "They downsized."
    assert "rewrite" in answer.timings


@pytest.mark.parametrize(
    "reply",
    [
        "max qubits QuantumMind reached in 2031",  # year swapped: the drift seen in the baseline
        "",
        "x" * 400,
    ],
)
def test_rewriter_rejects_drift_and_garbage(reply):
    question = "max qubits in 2023?"
    assert QueryRewriter(make_llm(reply)).rewrite(question) == question


def test_rewriter_keeps_the_first_line_without_quotes():
    reply = '"QuantumMind qubit count in 2023"\nExplanation: ...'
    assert (
        QueryRewriter(make_llm(reply)).rewrite("qubits 2023?") == "QuantumMind qubit count in 2023"
    )
