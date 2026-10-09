from pathlib import Path

import pytest

from localrag.evaluation.dataset import EvalItem, load_dataset
from localrag.evaluation.metrics import (
    content_correct,
    hit_at_k,
    percentile,
    reciprocal_rank,
    score_item,
    summarize,
    term_matches,
)

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("text", "term", "expected"),
    [
        ("It achieved 100 qubits.", "100", True),
        ("It scaled to 10,000 qubits.", "100", False),
        ("It scaled to 10,000 qubits.", "10000", True),
        ("A stable 1,000-qubit system.", "1000", True),
        ("They downsized the workforce.", "downsiz", True),
        ("Drug discovery speedups.", "pharmaceutical|drug discovery", True),
        ("They rehired key personnel.", "hire", False),  # matches only at a word start
        ("QuantumMind’s CEO", "quantummind's", True),
    ],
)
def test_term_matches(text, term, expected):
    assert term_matches(text, term) is expected


def test_content_correct_requires_all_terms_and_no_forbidden_ones():
    item = EvalItem(
        "q", "?", "factual", must_include=("financ", "breakthrough"), must_not_include=("rehir",)
    )
    assert content_correct(item, "Financial trouble, then a breakthrough.")
    assert not content_correct(item, "Financial trouble.")
    assert not content_correct(item, "Financial trouble, a breakthrough, and they rehired staff.")


def test_retrieval_metrics():
    assert hit_at_k(["2030.txt", "2033.txt"], ["2033.txt"]) is True
    assert hit_at_k(["2030.txt"], ["2033.txt"]) is False
    assert hit_at_k(["2030.txt"], []) is None
    assert reciprocal_rank(["2030.txt", "2031.txt", "2033.txt"], ["2033.txt"]) == pytest.approx(
        1 / 3
    )
    assert reciprocal_rank(["2030.txt"], ["2033.txt"]) == 0.0
    assert reciprocal_rank([], []) is None


def test_scoring_and_summary():
    factual = EvalItem("f", "?", "factual", gold_sources=("a",), must_include=("yes",))
    unanswerable = EvalItem("u", "?", "unanswerable")
    results = [
        score_item("cfg", factual, "yes", False, ["b", "a"], 1.0),
        score_item("cfg", factual, "I don't know", True, [], 2.0),
        score_item("cfg", unanswerable, "I don't know", True, ["c"], 3.0),
        score_item("cfg", unanswerable, "It was 10,000.", False, ["c"], 4.0),
    ]
    s = summarize("cfg", results)
    assert (s.n, s.answerable, s.unanswerable) == (4, 2, 2)
    assert s.accuracy == 0.5
    assert s.answer_accuracy == 0.5
    assert s.abstention_accuracy == 0.5
    assert s.hallucination_rate == 0.5
    assert s.false_abstention_rate == 0.5
    assert s.hit_at_k == 0.5  # unanswerable items have no gold sources
    assert s.mrr == pytest.approx(0.25)
    assert s.p50_seconds == 2.5


def test_percentile_edges():
    assert percentile([], 50) == 0.0
    assert percentile([3.0], 95) == 3.0


def test_bundled_dataset_is_valid():
    items = load_dataset(REPO / "eval" / "quantummind.jsonl")
    assert len(items) == 22
    assert sum(not i.answerable for i in items) == 4
    letters = {p.name for p in (REPO / "data" / "quantummind").iterdir()}
    assert all(set(i.gold_sources) <= letters for i in items)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('{"id": "a", "question": "?", "type": "trivia"}', "type must be"),
        ('{"id": "a", "question": "?", "type": "factual"}', "needs must_include"),
        ('{"question": "?"}', "invalid row"),
        ("not json", "invalid row"),
    ],
)
def test_dataset_validation(tmp_path, line, message):
    path = tmp_path / "d.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_dataset(path)


def test_duplicate_ids_are_rejected(tmp_path):
    row = '{"id": "a", "question": "?", "type": "unanswerable"}'
    path = tmp_path / "d.jsonl"
    path.write_text(f"{row}\n{row}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(path)
