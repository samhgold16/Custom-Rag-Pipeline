"""``rag demo``: a narrated walk through the pipeline on the bundled QuantumMind corpus."""

from __future__ import annotations

from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from localrag.config import RetrievalConfig
from localrag.engine import RAGEngine

CEO_QUESTION = "Who is the CEO of QuantumMind?"
OUT_OF_SCOPE_QUESTION = "what was the max qubits that Quantum mind reached in 2023?"
RERANK_QUESTION = "In which year did they build the 100 qubit quantum processor?"


def _step(console: Console, n: int, title: str, note: str) -> None:
    console.print()
    console.print(Rule(f"[bold]{n}. {title}", align="left"))
    console.print(f"[dim]{note}[/]")


def run_demo(engine: RAGEngine, console: Console) -> None:
    """Run each stage once and show what it produced."""
    engine.preflight()
    fields = [rule.name for rule in engine.settings.metadata_fields]

    _step(
        console,
        1,
        "The model alone",
        "No retrieval. The company and years are fictional, so the model has nothing to go on.",
    )
    reply = engine.chat_model.invoke(CEO_QUESTION)
    console.print(f"[bold]Q:[/] {CEO_QUESTION}\n[bold]A:[/] {str(reply.content).strip()}")

    _step(
        console,
        2,
        "Ingest",
        "Load → split → extract metadata from file names → embed → upsert. "
        "Point IDs are content hashes, so a second run is a no-op.",
    )
    report = engine.ingest()
    console.print(
        f"{report.documents} documents → {report.chunks} chunks in [bold]{report.collection}[/], "
        f"{report.vector_size}-d vectors; embedded {report.added} this run ({report.seconds:.2f}s)."
    )

    _step(
        console,
        3,
        "Retrieve",
        "Dense cosine kNN, k=4. The CEO's name is only in each letter's "
        "signature, so the closing chunks come back.",
    )
    table = Table("#", "Source", *fields, "Cosine", "Excerpt", header_style="bold")
    for src in engine.retrieve(CEO_QUESTION):
        table.add_row(
            str(src.rank),
            src.source,
            *[str(src.metadata.get(f, "")) for f in fields],
            f"{src.score:.3f}",
            src.snippet(70),
        )
    console.print(table)

    _step(console, 4, "Retrieve + generate", "Same question as step 1; only the context changed.")
    answer = engine.ask(CEO_QUESTION)
    console.print(f"[bold]A:[/] {answer.text}")

    _step(
        console,
        5,
        "The hallucination case",
        "The corpus covers 2030–2035. The question asks "
        "about 2023. Same retrieval each time; only what the model sees differs.",
    )
    rows = [
        ("Plain chunks (original pipeline)", dict(context_format="plain", guardrail=False)),
        ("Year header added at prompt time", dict(context_format="header", guardrail=False)),
        ("Scope guardrail (no LLM call)", dict(context_format="header", guardrail=True)),
    ]
    table = Table("Variant", "Answer", "Abstained", header_style="bold")
    for label, kwargs in rows:
        result = engine.ask(OUT_OF_SCOPE_QUESTION, **kwargs)
        table.add_row(label, result.text, "yes" if result.abstained else "[red]no[/]")
    console.print(f"[bold]Q:[/] {OUT_OF_SCOPE_QUESTION}")
    console.print(table)

    _step(
        console,
        6,
        "Re-rank",
        "The LLM scores each (question, chunk) pair 0–10 and the list is "
        "re-sorted. One extra LLM call per candidate.",
    )
    dense = engine.retrieve(RERANK_QUESTION, RetrievalConfig(k=4))
    reranked = engine.retrieve(
        RERANK_QUESTION, RetrievalConfig(k=4, rerank=True, rerank_candidates=4)
    )
    table = Table("#", "Dense order", "Re-ranked", "LLM score", header_style="bold")
    for i, (d, r) in enumerate(zip(dense, reranked, strict=False), start=1):
        table.add_row(str(i), d.source, r.source, f"{r.score:.0f}")
    console.print(f"[bold]Q:[/] {RERANK_QUESTION}")
    console.print(table)
    console.print('\n[green]Done.[/] Next: `rag ask "…"`, or `rag eval` to measure every strategy.')
