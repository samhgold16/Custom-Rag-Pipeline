import json

import pytest

from localrag import RetrievalConfig
from localrag.evaluation import GRIDS, EvalConfig, EvalItem, render_markdown, run_eval, write_report
from localrag.evaluation import runner as runner_mod
from localrag.providers import HealthReport

ITEMS = [
    EvalItem(
        "q1",
        "How many qubits in 2031?",
        "factual",
        gold_sources=("2031.txt",),
        must_include=("250",),
    ),
    EvalItem("u02", "What happened in 2023?", "unanswerable"),
]

CONFIGS = [
    EvalConfig("baseline", "plain"),
    EvalConfig("hdr-embedded", "header in text", header_fields=("year",)),
    EvalConfig(
        "hybrid+guardrail",
        "guarded",
        context_format="header",
        retrieval=RetrievalConfig(strategy="hybrid"),
        guardrail=True,
    ),
]


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    monkeypatch.setattr(runner_mod, "check_ollama", lambda host, models: HealthReport(host, False))
    monkeypatch.setattr(runner_mod, "server_version", lambda host: None)


def test_run_eval_scores_every_cell(engine):
    seen = []
    run = run_eval(engine, ITEMS, CONFIGS, on_result=seen.append)
    assert len(run.results) == len(seen) == 6
    assert [s.config for s in run.summaries] == ["baseline", "hdr-embedded", "hybrid+guardrail"]
    guarded = {r.item_id: r for r in run.results if r.config == "hybrid+guardrail"}
    assert guarded["u02"].abstained and guarded["u02"].guardrail == "abstain"
    assert guarded["q1"].sources == ["2031.txt"]
    assert engine.store.exists(engine.collection + "__hdr-year")
    assert run.environment["chunk_size"] == engine.settings.chunk_size


def test_report_contains_tables_and_highlights(engine, tmp_path):
    run = run_eval(engine, ITEMS, CONFIGS, dataset_path="eval/demo.jsonl")
    markdown = render_markdown(run, highlight=("u02",))
    assert "| Config | Context | Retrieval | Accuracy |" in markdown
    assert "## Per-question results" in markdown
    assert "### `u02`: What happened in 2023?" in markdown
    assert "none (guardrail: abstain)" in markdown
    md_path, json_path = write_report(run, tmp_path, name="report")
    assert md_path.read_text(encoding="utf-8").startswith("# Evaluation report: `demo`")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(data["results"]) == 6 and data["configs"][1]["header_fields"] == ["year"]


def test_grids_are_well_formed():
    names = [c.name for c in GRIDS["default"]]
    assert len(names) == len(set(names))
    assert names[0] == "baseline"
    assert {c.name for c in GRIDS["quick"]} <= set(names)
