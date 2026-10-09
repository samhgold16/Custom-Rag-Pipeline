# AGENTS.md

Guide for coding agents (and people) working on `localrag`.

## What this is

`localrag` is a local-first RAG engine: it ingests a folder of `.txt`/`.md` files into embedded
Qdrant, answers questions with an Ollama chat model, and measures grounding with a
deterministic evaluation harness. Retrieval strategies (dense, MMR, hybrid BM25+dense with
RRF, LLM re-ranking), payload filters, a year-scope guardrail, and query rewriting all sit
behind one `RAGEngine` facade and a Typer CLI (`rag`). The bundled demo corpus
(`data/quantummind/`) is fictional, so every correct answer must come from retrieval.

## Layout

| Path | Contents |
| ---- | -------- |
| `src/localrag/` | The package. `engine.py` is the facade; see `docs/architecture.md` for every module |
| `src/localrag/retrieval/` | Strategy registry, `fusion.py` (RRF), `bm25.py`, `rerank.py` |
| `src/localrag/evaluation/` | Dataset loader, metrics, grid runner, report writer |
| `tests/unit/` | Offline tests: fake models + `QdrantClient(":memory:")` |
| `tests/integration/` | `@pytest.mark.integration`, needs a running Ollama |
| `data/quantummind/` | Demo corpus. **Do not edit**: the eval set and the case study depend on the exact text |
| `eval/quantummind.jsonl` | Gold questions |
| `results/` | Generated eval reports (committed) |
| `examples/` | Runnable scripts using only the public API |
| `docs/` | Architecture, configuration, evaluation, design decisions, case study |
| `localrag.toml` | Config for the demo corpus |

## Commands

```bash
uv sync                              # install (Python >= 3.11)
uv run pytest                        # unit tests, offline, < 5 s
uv run pytest -m integration         # live tests, needs `ollama serve` + both models
uv run ruff check . && uv run ruff format --check .
uv run rag doctor | ingest | ask "…" | eval | demo
```

Always use `uv run`; never bare `python` or `pip`.

## Conventions

- **Dependency injection.** `RAGEngine` takes `chat_model`, `embeddings`, `client`, `provider`.
  Never construct Ollama or Qdrant clients inside library functions.
- **Logging.** Library modules use `logging.getLogger(__name__)` and never configure logging
  or `print`. Only `cli.py` calls `setup_logging()`.
- **Prompts** live in `generation.py` (answering), `query.py` (rewrite), `retrieval/rerank.py`
  (re-rank). `BASE_SYSTEM_PROMPT` is the original pipeline's prompt verbatim; keep it that way
  so the `baseline` config stays comparable.
- **Types and docstrings.** Type hints everywhere; Google-style docstrings that explain *why*.
- **Unit tests must not need a network.** Use `tests/conftest.py` (`build_engine`, `make_llm`,
  `CountingChatModel`). Anything that needs Ollama is an integration test.
- Line length 100; ruff rules `E,F,I,UP,B,SIM,RUF`.

## Embedded Qdrant rules

1. One client per folder per process (`store.open_client` caches them). Never run two pipeline
   commands against the same `qdrant_path` concurrently; you will get `StoreLockedError`.
2. Clients are closed in an `atexit` hook. Do not close them mid-run or in `__del__`, or
   qdrant-client prints `ImportError: sys.meta_path is None` at shutdown.
3. LangChain nests metadata under the `metadata` payload key: filter on `metadata.year`.
4. Filters are `qdrant_client.models.Filter` objects (`store.metadata_filter`), not dicts.
5. Point IDs are deterministic (`ingest.point_id`). Never switch to random IDs; idempotent
   ingest depends on them.
6. A text-header enricher changes the vectors, so it must produce a different collection name
   (`Settings.collection_name` handles this).

## Dependencies

LangChain 1.x, split packages only: `langchain-core`, `langchain-ollama`, `langchain-qdrant`,
`langchain-text-splitters`. Do not add the `langchain` umbrella, `langchain-community`, or
`langchain-classic`: generation is LCEL, BM25 uses `rank-bm25` directly, and RRF is our own.
No hosted-model SDKs; a new provider implements `providers.ModelProvider`.

Ollama must be recent enough to apply the chat model's GGUF chat template. Old servers fall
back to `{{ .Prompt }}`, silently dropping the system prompt (and with it, the retrieved
context). `rag doctor` checks this.

## Evaluation integrity

- Never fabricate or hand-edit numbers in `results/`, the README, or `docs/`. Regenerate with
  `uv run rag eval` and copy from the report.
- **Headline invariant.** After changing chunking, prompts, context formatting, or models,
  run `uv run python examples/hallucination_case_study.py`. The `baseline` variant fabricating
  an answer for 2023 and the header variants abstaining is the project's central finding. If it
  stops reproducing, say so and update the docs. Do not tune the question until it does.
- Report techniques that do not help. The results table is not supposed to be all wins.

## Extending

- **Retrieval strategy:** in `retrieval/__init__.py` (or any imported module),
  `@register("name") def fn(ctx, query, config, k) -> list[tuple[Document, float]]`. Honour
  `config.filters` (`store.metadata_filter` for Qdrant, `bm25.matches` for Python). Add a unit
  test with the `engine` fixture and an `EvalConfig` to `GRIDS`.
- **Enricher:** a pure `list[Document] -> list[Document]` in `enrichers.py`, wired into
  `default_enrichers`. If it changes `page_content`, it must change the collection name.
- **Metadata field:** usually just config: add a `[[metadata.fields]]` rule.
- **Eval rows:** append to `eval/quantummind.jsonl` (schema in `docs/evaluation.md`). Check every
  `must_include` term against the letters.

## Git

Commit only when asked. No AI co-author trailers unless the owner asks for them. Never commit
`.env`, `.venv/`, `qdrant_db/`, or caches.
