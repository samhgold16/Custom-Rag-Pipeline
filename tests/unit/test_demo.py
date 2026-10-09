from rich.console import Console

from localrag.demo import run_demo
from tests.conftest import build_engine, make_llm


def test_demo_runs_every_step_offline(corpus):
    engine = build_engine(corpus, make_llm("I'm not sure.", "Dr. Quentin Bohr.", "7", "3"))
    console = Console(record=True, width=120)
    run_demo(engine, console)
    text = console.export_text()
    for heading in ("1. The model alone", "2. Ingest", "5. The hallucination case", "6. Re-rank"):
        assert heading in text
    assert "nothing for 2023" in text  # the guardrail row
