"""Run a grid of pipeline configurations over a gold dataset."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from localrag.config import ContextFormat, RetrievalConfig
from localrag.engine import RAGEngine
from localrag.evaluation.dataset import EvalItem
from localrag.evaluation.metrics import ItemResult, Summary, score_item, summarize
from localrag.providers import check_ollama, server_version


@dataclass(frozen=True)
class EvalConfig:
    """One cell of the grid.

    Attributes:
        name: Short id used in tables.
        description: What the cell tests.
        header_fields: Ingest-time text header (selects the collection variant).
        context_format: Prompt-time presentation of chunks.
        retrieval: Retrieval strategy and parameters.
        guardrail: Apply the scope guardrail (bypasses the LLM for out-of-scope years).
        rewrite: Two-step query rewriting.
    """

    name: str
    description: str
    header_fields: tuple[str, ...] = ()
    context_format: ContextFormat = "plain"
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    guardrail: bool = False
    rewrite: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "header_fields": list(self.header_fields),
            "context_format": self.context_format,
            "retrieval": self.retrieval.label(),
            "guardrail": self.guardrail,
            "rewrite": self.rewrite,
        }


GRIDS: dict[str, list[EvalConfig]] = {
    "default": [
        EvalConfig("baseline", "Original pipeline: raw chunks, plain context, dense k=4"),
        EvalConfig(
            "hdr-embedded",
            "Year header written into chunk text before embedding (the original fix)",
            header_fields=("year",),
        ),
        EvalConfig(
            "hdr-prompt",
            "Same header added at prompt time only; embeddings untouched",
            context_format="header",
        ),
        EvalConfig(
            "tagged",
            "[source | year] label at prompt time plus a citation instruction",
            context_format="tagged",
        ),
        EvalConfig(
            "mmr",
            "MMR (fetch_k=20, λ=0.5), header context",
            context_format="header",
            retrieval=RetrievalConfig(strategy="mmr"),
        ),
        EvalConfig(
            "mmr-k2",
            "MMR with k=2 (hand-tuned on one question originally), header context",
            context_format="header",
            retrieval=RetrievalConfig(strategy="mmr", k=2),
        ),
        EvalConfig(
            "hybrid",
            "BM25 + dense, weighted RRF (0.5/0.5, c=60), header context",
            context_format="header",
            retrieval=RetrievalConfig(strategy="hybrid"),
        ),
        EvalConfig(
            "rerank",
            "Dense top-8 re-ranked by the LLM to top-4, header context",
            context_format="header",
            retrieval=RetrievalConfig(rerank=True),
        ),
        EvalConfig(
            "rewrite",
            "Two-step query rewrite (constraint-preserving), header context",
            context_format="header",
            rewrite=True,
        ),
        EvalConfig(
            "guardrail",
            "Year-scope guardrail + auto payload filter, header context, dense",
            context_format="header",
            guardrail=True,
        ),
        EvalConfig(
            "hybrid+guardrail",
            "Guardrail + hybrid retrieval, header context",
            context_format="header",
            retrieval=RetrievalConfig(strategy="hybrid"),
            guardrail=True,
        ),
    ],
}
GRIDS["quick"] = [c for c in GRIDS["default"] if c.name in ("baseline", "hdr-prompt", "hybrid")]


@dataclass
class EvalRun:
    """Everything an evaluation produced, plus the environment that produced it."""

    dataset: str
    configs: list[EvalConfig]
    results: list[ItemResult]
    summaries: list[Summary]
    environment: dict[str, Any]
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "environment": self.environment,
            "seconds": round(self.seconds, 2),
            "configs": [c.to_dict() for c in self.configs],
            "summaries": [s.to_dict() for s in self.summaries],
            "results": [r.to_dict() for r in self.results],
        }


def run_eval(
    engine: RAGEngine,
    items: list[EvalItem],
    configs: list[EvalConfig],
    *,
    dataset_path: str = "",
    on_result: Callable[[ItemResult], None] | None = None,
) -> EvalRun:
    """Evaluate every config on every item.

    Each distinct ``header_fields`` value gets its own collection, ingested idempotently, so a
    re-run reuses existing vectors. All variants share the engine's models and Qdrant client.
    """
    started = time.perf_counter()
    variants: dict[tuple[str, ...], RAGEngine] = {}
    for config in configs:
        if config.header_fields not in variants:
            variant = engine.variant(header_fields=config.header_fields)
            variant.ingest()
            variants[config.header_fields] = variant

    results: list[ItemResult] = []
    for config in configs:
        variant = variants[config.header_fields]
        for item in items:
            answer = variant.ask(
                item.question,
                config.retrieval,
                context_format=config.context_format,
                guardrail=config.guardrail,
                rewrite=config.rewrite,
            )
            result = score_item(
                config.name,
                item,
                answer.text,
                answer.abstained,
                [s.source for s in answer.sources],
                answer.timings.get("total", 0.0),
                guardrail=answer.guardrail.action if answer.guardrail else None,
                rewritten_query=answer.rewritten_query,
            )
            results.append(result)
            if on_result:
                on_result(result)

    summaries = [summarize(c.name, [r for r in results if r.config == c.name]) for c in configs]
    return EvalRun(
        dataset=dataset_path,
        configs=configs,
        results=results,
        summaries=summaries,
        environment=capture_environment(engine, dataset_path),
        seconds=time.perf_counter() - started,
    )


def capture_environment(engine: RAGEngine, dataset_path: str = "") -> dict[str, Any]:
    """Record what is needed to reproduce (or distrust) a set of numbers."""
    s = engine.settings
    health = check_ollama(s.ollama_host, [s.chat_model, s.embedding_model])
    packages = {}
    for name in (
        "localrag",
        "langchain-core",
        "langchain-ollama",
        "langchain-qdrant",
        "qdrant-client",
        "rank-bm25",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    dataset_sha = (
        hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()[:12]
        if dataset_path and Path(dataset_path).exists()
        else None
    )
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "ollama_version": server_version(s.ollama_host),
        "chat_model": s.chat_model,
        "chat_model_digest": health.digests.get(s.chat_model, "")[:12] or None,
        "embedding_model": s.embedding_model,
        "embedding_model_digest": health.digests.get(s.embedding_model, "")[:12] or None,
        "temperature": s.temperature,
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
        "data_dir": s.data_dir,
        "dataset_sha256": dataset_sha,
    }


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()
