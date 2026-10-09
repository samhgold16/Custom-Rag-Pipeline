import json

import pytest
from typer.testing import CliRunner

from localrag import cli
from localrag.providers import HealthReport
from tests.conftest import build_engine, make_llm

runner = CliRunner()


@pytest.fixture
def fake_engine(corpus, monkeypatch):
    engine = build_engine(corpus, make_llm("QuantumMind's CEO is Dr. Quentin Bohr."))
    monkeypatch.setattr(cli, "ENGINE_FACTORY", lambda settings: engine)
    return engine


def test_help_lists_every_command():
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for command in ("doctor", "ingest", "ask", "eval", "demo"):
        assert command in result.output


def test_ingest_reports_then_skips(fake_engine):
    first = runner.invoke(cli.app, ["ingest"])
    second = runner.invoke(cli.app, ["ingest", "--json"])
    assert first.exit_code == 0 and "4 documents → 4 chunks" in first.output
    assert json.loads(second.output)["unchanged"] is True


def test_ask_json(fake_engine):
    fake_engine.ingest()
    result = runner.invoke(
        cli.app, ["ask", "Who is the CEO?", "--json", "--no-guardrail", "-k", "2"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["answer"] == "QuantumMind's CEO is Dr. Quentin Bohr."
    assert len(payload["sources"]) == 2
    assert payload["retrieval"] == "dense k=2"


def test_ask_guardrail_and_year_filter(fake_engine):
    fake_engine.ingest()
    out = runner.invoke(cli.app, ["ask", "max qubits in 2023?"])
    assert out.exit_code == 0 and "nothing for 2023" in out.output
    filtered = runner.invoke(cli.app, ["ask", "What happened?", "--year", "2032", "--json"])
    assert {s["metadata"]["year"] for s in json.loads(filtered.output)["sources"]} == {2032}


def test_ask_before_ingest_fails_cleanly(fake_engine):
    result = runner.invoke(cli.app, ["ask", "q"])
    assert result.exit_code == 1
    assert "rag ingest" in result.output


def test_unknown_strategy_is_rejected(fake_engine):
    result = runner.invoke(cli.app, ["ask", "q", "--strategy", "telepathy"])
    assert result.exit_code == 1 and "Unknown strategy" in result.output


def test_doctor_fails_when_ollama_is_down(monkeypatch):
    monkeypatch.setenv("RAG_QDRANT_PATH", ":memory:")
    monkeypatch.setattr(
        cli, "check_ollama", lambda host, models: HealthReport(host, False, error="ConnectError")
    )
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 1
    assert "ollama serve" in result.output


def test_doctor_flags_a_missing_chat_template(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_QDRANT_PATH", ":memory:")
    monkeypatch.setattr(
        cli,
        "check_ollama",
        lambda host, models: HealthReport(host, True, digests={m: "abc123def456" for m in models}),
    )
    monkeypatch.setattr(cli, "server_version", lambda host: "0.20.3")
    monkeypatch.setattr(cli, "chat_template_problem", lambda host, model: "no chat template")
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 1 and "no chat template" in result.output


def test_eval_quick_grid_writes_a_report(fake_engine, tmp_path, monkeypatch):
    from localrag.evaluation import runner as runner_mod

    monkeypatch.setattr(runner_mod, "check_ollama", lambda host, models: HealthReport(host, False))
    monkeypatch.setattr(runner_mod, "server_version", lambda host: None)
    dataset = tmp_path / "mini.jsonl"
    dataset.write_text(
        '{"id": "q1", "question": "CEO?", "type": "factual", "must_include": ["bohr"]}\n',
        encoding="utf-8",
    )
    result = runner.invoke(
        cli.app,
        [
            "eval",
            "--dataset",
            str(dataset),
            "--grid",
            "quick",
            "--out",
            str(tmp_path),
            "--name",
            "r",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "r.md").exists() and (tmp_path / "r.json").exists()
    assert "baseline" in result.output
