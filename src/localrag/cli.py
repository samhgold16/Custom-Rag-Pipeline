"""``rag``: command-line interface (Typer + Rich)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
from ollama import ResponseError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from localrag import __version__
from localrag.config import Settings
from localrag.engine import RAGEngine
from localrag.log import setup_logging
from localrag.providers import ProviderError, chat_template_problem, check_ollama, server_version
from localrag.retrieval import strategies
from localrag.store import StoreLockedError, open_client
from localrag.types import Answer, SourceChunk

app = typer.Typer(
    name="rag",
    help="Local-first RAG: ingest a folder, ask questions, and measure grounding.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    add_completion=False,
)
console = Console()
err = Console(stderr=True)
logger = logging.getLogger(__name__)

# Tests replace this to inject fake models and an in-memory Qdrant.
ENGINE_FACTORY: Callable[[Settings], RAGEngine] = RAGEngine


class _State:
    config: Path | None = None
    host: str = "the configured Ollama host"


state = _State()


def _settings(**overrides: Any) -> Settings:
    try:
        settings = Settings.load(state.config, **overrides)
    except (ValueError, FileNotFoundError) as exc:
        _fail(f"Configuration error: {exc}")
    state.host = settings.ollama_host
    return settings


def _fail(message: str, hint: str | None = None) -> None:
    err.print(f"[bold red]Error:[/] {message}")
    if hint:
        err.print(f"[dim]{hint}[/]")
    raise typer.Exit(1)


def _run(fn: Callable[[], Any]) -> Any:
    """Translate expected operational failures into one-line messages instead of tracebacks."""
    try:
        return fn()
    except StoreLockedError as exc:
        _fail(str(exc))
    except ProviderError as exc:
        _fail(str(exc), "Run `rag doctor` for a full check.")
    except httpx.ConnectError:
        _fail(
            f"Cannot reach Ollama at {state.host}.",
            "Start it with `ollama serve`, then run `rag doctor`.",
        )
    except ResponseError as exc:
        _fail(f"Ollama error: {exc.error}", "Run `rag doctor` to check that the models are pulled.")
    except (LookupError, FileNotFoundError) as exc:
        _fail(str(exc).strip("'\""))


@app.callback()
def main(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="TOML config (default: ./localrag.toml if present)."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show pipeline logs.")] = False,
) -> None:
    setup_logging(verbose, err)
    state.config = config


@app.command()
def version() -> None:
    """Print the localrag version."""
    console.print(f"localrag {__version__}")


@app.command()
def doctor() -> None:
    """Check Ollama, the models, the chat template, the Qdrant folder, and the corpus."""
    s = _settings()
    table = Table(show_header=False, box=None, padding=(0, 2))
    ok = True

    def row(passed: bool, label: str, detail: str) -> None:
        nonlocal ok
        ok &= passed
        table.add_row("[green]✓[/]" if passed else "[red]✗[/]", label, detail)

    health = check_ollama(s.ollama_host, [s.chat_model, s.embedding_model])
    version_str = server_version(s.ollama_host) if health.reachable else None
    row(
        health.reachable,
        "Ollama",
        f"{s.ollama_host} (v{version_str})"
        if health.reachable
        else f"unreachable at {s.ollama_host}: run `ollama serve`",
    )
    if health.reachable:
        for model in (s.chat_model, s.embedding_model):
            present = model not in health.missing
            row(
                present,
                "Model",
                f"{model} ({health.digests[model][:12]})"
                if present
                else f"{model} missing: run `ollama pull {model}`",
            )
        if s.chat_model not in health.missing:
            problem = chat_template_problem(s.ollama_host, s.chat_model)
            row(problem is None, "Chat template", problem or "present")

    if s.qdrant_path == ":memory:":
        row(True, "Qdrant", "in-memory")
    else:
        try:
            n_collections = len(open_client(s.qdrant_path).get_collections().collections)
            row(
                True,
                "Qdrant",
                f"{s.qdrant_path} (embedded, {n_collections} collections)",
            )
        except StoreLockedError as exc:
            row(False, "Qdrant", str(exc))

    data = Path(s.data_dir)
    files = (
        sorted({p for g in s.globs for p in data.glob(g) if p.is_file()}) if data.is_dir() else []
    )
    row(
        bool(files),
        "Corpus",
        f"{s.data_dir}: {len(files)} file(s)" if files else f"no files in {s.data_dir}",
    )
    console.print(table)
    if not ok:
        raise typer.Exit(1)
    console.print("[green]Ready.[/]")


@app.command()
def ingest(
    data_dir: Annotated[str | None, typer.Option(help="Folder of .txt/.md files.")] = None,
    header: Annotated[
        list[str] | None,
        typer.Option(
            "--header",
            help="Write this metadata field into chunk text before embedding (repeatable).",
        ),
    ] = None,
    rebuild: Annotated[bool, typer.Option(help="Drop and re-embed even if unchanged.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Index the corpus. Idempotent: an unchanged corpus is skipped without embedding."""
    s = _settings(data_dir=data_dir, header_fields=tuple(header) if header else None)
    engine = ENGINE_FACTORY(s)
    report = _run(lambda: engine.ingest(rebuild=rebuild))
    if as_json:
        console.print_json(json.dumps(report.to_dict()))
        return
    status = (
        "[dim]unchanged, nothing embedded[/]"
        if report.unchanged
        else (
            f"[green]+{report.added}[/] / [red]-{report.removed}[/]"
            + (" (rebuilt)" if report.rebuilt else "")
        )
    )
    console.print(
        f"[bold]{report.collection}[/]: {report.documents} documents → {report.chunks} chunks, "
        f"{report.vector_size}-d vectors · {status} · {report.seconds:.2f}s"
    )
    for name, values in report.field_values.items():
        console.print(f"  {name}: {_span(values)}")


def _span(values: list[Any]) -> str:
    if (
        values
        and all(isinstance(v, int) for v in values)
        and values == list(range(values[0], values[-1] + 1))
    ):
        return f"{values[0]}–{values[-1]} ({len(values)} values)"
    return ", ".join(map(str, values)) or "none found"


def _parse_filters(raw: list[str] | None, year: int | None) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    for item in raw or []:
        if "=" not in item:
            _fail(f"--filter expects field=value, got {item!r}")
        key, value = item.split("=", 1)
        filters[key] = int(value) if value.lstrip("-").isdigit() else value
    if year is not None:
        filters["year"] = year
    return filters


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The question.")],
    strategy: Annotated[
        str | None, typer.Option("--strategy", "-s", help=f"One of: {', '.join(strategies())}.")
    ] = None,
    k: Annotated[int | None, typer.Option("-k", help="Chunks to retrieve.")] = None,
    year: Annotated[int | None, typer.Option(help="Shortcut for --filter year=YEAR.")] = None,
    filter_: Annotated[
        list[str] | None, typer.Option("--filter", help="Payload filter field=value (repeatable).")
    ] = None,
    rerank: Annotated[
        bool | None, typer.Option("--rerank/--no-rerank", help="Re-rank candidates with the LLM.")
    ] = None,
    rewrite: Annotated[
        bool | None, typer.Option("--rewrite/--no-rewrite", help="Two-step query rewriting.")
    ] = None,
    guardrail: Annotated[
        bool | None, typer.Option("--guardrail/--no-guardrail", help="Year-scope guardrail.")
    ] = None,
    context: Annotated[
        str | None, typer.Option(help="Context format: plain, header, or tagged.")
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the full answer object as JSON.")
    ] = False,
) -> None:
    """Answer a question from the indexed corpus, with sources."""
    s = _settings(context_format=context)
    if strategy and strategy not in strategies():
        _fail(f"Unknown strategy {strategy!r}; choose from {', '.join(strategies())}")
    overrides: dict[str, Any] = {"strategy": strategy, "k": k, "rerank": rerank}
    filters = _parse_filters(filter_, year)
    if filters:
        overrides["filters"] = filters
    retrieval = replace(s.retrieval, **{key: v for key, v in overrides.items() if v is not None})
    engine = ENGINE_FACTORY(s)
    answer: Answer = _run(
        lambda: engine.ask(question, retrieval, guardrail=guardrail, rewrite=rewrite)
    )
    if as_json:
        console.print_json(json.dumps(answer.to_dict(), ensure_ascii=False))
        return
    _render_answer(answer, [r.name for r in s.metadata_fields])


def _render_answer(answer: Answer, fields: list[str]) -> None:
    subtitle = f"{answer.collection} · {answer.retrieval} · context={answer.context_format}"
    style = "yellow" if answer.abstained else "green"
    console.print(Panel(answer.text, title="Answer", subtitle=subtitle, border_style=style))
    if answer.guardrail and answer.guardrail.action != "pass":
        console.print(
            f"[yellow]Guardrail ({answer.guardrail.action}):[/] {answer.guardrail.reason}"
        )
    if answer.rewritten_query:
        console.print(f"[cyan]Search query (rewritten):[/] {answer.rewritten_query}")
    if answer.sources:
        console.print(_sources_table(answer.sources, fields))
    timings = " · ".join(f"{k} {v:.2f}s" for k, v in answer.timings.items())
    console.print(f"[dim]{timings}[/]")


def _sources_table(sources: list[SourceChunk], fields: list[str]) -> Table:
    table = Table(title="Sources", title_justify="left", header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Source")
    for name in fields:
        table.add_column(name)
    table.add_column("Score", justify="right")
    table.add_column("Excerpt", overflow="fold", ratio=1)
    for src in sources:
        table.add_row(
            str(src.rank),
            src.source,
            *[str(src.metadata.get(name, "")) for name in fields],
            f"{src.score:.3f}",
            src.snippet(110),
        )
    return table


@app.command(name="eval")
def evaluate(
    dataset: Annotated[Path, typer.Option(help="Gold JSONL dataset.")] = Path(
        "eval/quantummind.jsonl"
    ),
    grid: Annotated[str, typer.Option(help="Config grid: default or quick.")] = "default",
    only: Annotated[
        list[str] | None, typer.Option(help="Run only these config names (repeatable).")
    ] = None,
    out: Annotated[Path, typer.Option(help="Report directory.")] = Path("results"),
    name: Annotated[
        str | None, typer.Option(help="Report file stem (default: <date>-<dataset>).")
    ] = None,
) -> None:
    """Run a grid of pipeline configurations over a gold dataset and write a report."""
    from localrag.evaluation import GRIDS, load_dataset, run_eval, write_report

    if grid not in GRIDS:
        _fail(f"Unknown grid {grid!r}; choose from {', '.join(GRIDS)}")
    configs = [c for c in GRIDS[grid] if not only or c.name in only]
    if not configs:
        _fail(f"No configs named {only} in grid {grid!r}")
    try:
        items = load_dataset(dataset)
    except (ValueError, FileNotFoundError) as exc:
        _fail(str(exc))
    s = _settings()
    engine = ENGINE_FACTORY(s)
    _run(engine.preflight)

    total = len(items) * len(configs)
    done = 0

    def progress(result: Any) -> None:
        nonlocal done
        done += 1
        mark = "[green]✓[/]" if result.correct else "[red]✗[/]"
        err.print(f"[dim]{done:>4}/{total}[/] {mark} {result.config:<17} {result.item_id}")

    console.print(f"Evaluating {len(configs)} config(s) × {len(items)} questions on {s.chat_model}")
    run = _run(
        lambda: run_eval(engine, items, configs, dataset_path=str(dataset), on_result=progress)
    )
    md_path, json_path = write_report(run, out, name)

    table = Table(header_style="bold")
    for col in (
        "Config",
        "Accuracy",
        "Answerable",
        "Abstained (unans.)",
        "False abst.",
        "Hit@k",
        "MRR",
        "p50 s",
    ):
        table.add_column(col, justify="left" if col == "Config" else "right")
    for summary in run.summaries:
        table.add_row(
            summary.config,
            f"{summary.accuracy:.0%}",
            _pct(summary.answer_accuracy),
            _pct(summary.abstention_accuracy),
            _pct(summary.false_abstention_rate),
            _pct(summary.hit_at_k),
            "–" if summary.mrr is None else f"{summary.mrr:.2f}",
            f"{summary.p50_seconds:.2f}",
        )
    console.print(table)
    console.print(f"Report: {md_path}\nData:   {json_path}")


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value:.0%}"


@app.command()
def demo() -> None:
    """Narrated tour of the pipeline on the demo corpus (needs Ollama)."""
    from localrag.demo import run_demo

    engine = ENGINE_FACTORY(_settings())
    _run(lambda: run_demo(engine, console))


if __name__ == "__main__":
    app()
