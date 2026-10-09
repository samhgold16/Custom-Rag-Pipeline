"""``RAGEngine``: the public facade over ingest, retrieval, and generation."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from qdrant_client import QdrantClient

from localrag import ingest as ingest_mod
from localrag.config import ContextFormat, RetrievalConfig, Settings
from localrag.generation import AnswerGenerator, is_abstention
from localrag.providers import ModelProvider, OllamaProvider
from localrag.query import QueryRewriter, guardrail_filter, scope_guardrail
from localrag.retrieval import LLMReranker, RetrievalContext, retrieve
from localrag.store import Store
from localrag.types import Answer, GuardrailDecision, IngestReport, SourceChunk

logger = logging.getLogger(__name__)


class RAGEngine:
    """Ingest a folder of documents and answer questions over it.

    Models and the vector store are injectable, which is how the test suite runs the real
    pipeline with fake models and an in-memory Qdrant. When omitted they are built from
    ``settings`` through the provider (Ollama by default) and the embedded Qdrant folder.

    Args:
        settings: Configuration; defaults to ``Settings.load()``.
        chat_model: Generator (also used for re-ranking and rewriting).
        embeddings: Embedding model.
        client: Qdrant client. Engines for different collection variants can share one.
        provider: Builds whichever of ``chat_model``/``embeddings`` were not given.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        chat_model: BaseChatModel | None = None,
        embeddings: Embeddings | None = None,
        client: QdrantClient | None = None,
        provider: ModelProvider | None = None,
    ) -> None:
        self.settings = settings or Settings.load()
        self._provider = provider or OllamaProvider(self.settings)
        self._chat_model = chat_model
        self._embeddings = embeddings
        if client is not None:
            manifest_dir = (
                None if self.settings.qdrant_path == ":memory:" else self.settings.qdrant_path
            )
            self.store = Store(client, manifest_dir)
        else:
            self.store = Store.from_path(self.settings.qdrant_path)
        self._retrieval_ctx: RetrievalContext | None = None
        self._scope_values: set[int] | None = None

    # -- lazily built dependencies ----------------------------------------------------------

    @property
    def chat_model(self) -> BaseChatModel:
        if self._chat_model is None:
            self._chat_model = self._provider.chat_model()
        return self._chat_model

    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            self._embeddings = self._provider.embeddings()
        return self._embeddings

    def preflight(self) -> None:
        """Fail fast with a fix-it message if the provider cannot serve the configured models.

        Skipped when both models were injected (as in tests).
        """
        health = getattr(self._provider, "health", None)
        if health is not None and (self._chat_model is None or self._embeddings is None):
            health().raise_for_status()

    @property
    def collection(self) -> str:
        return self.settings.collection_name

    def variant(self, **overrides: Any) -> RAGEngine:
        """A sibling engine with different settings sharing this one's models and Qdrant client.

        Used to compare collection variants (e.g. ``header_fields=("year",)``) side by side.
        """
        return RAGEngine(
            replace(self.settings, **overrides),
            chat_model=self._chat_model,
            embeddings=self._embeddings,
            client=self.store.client,
            provider=self._provider,
        )

    # -- ingest -----------------------------------------------------------------------------

    def ingest(self, *, rebuild: bool = False) -> IngestReport:
        """Index ``settings.data_dir`` into this engine's collection. Idempotent."""
        report = ingest_mod.ingest(self.settings, self.store, self.embeddings, rebuild=rebuild)
        self._retrieval_ctx = None
        self._scope_values = None
        return report

    def _require_collection(self) -> None:
        if not self.store.exists(self.collection):
            raise LookupError(
                f"Collection {self.collection!r} does not exist yet. Run `rag ingest` "
                "(or engine.ingest()) first."
            )

    def _ctx(self) -> RetrievalContext:
        self._require_collection()
        if self._retrieval_ctx is None:
            self._retrieval_ctx = RetrievalContext(
                vectorstore=self.store.vectorstore(self.collection, self.embeddings),
                load_documents=lambda: self.store.documents(self.collection),
            )
        return self._retrieval_ctx

    def scope_values(self) -> set[int]:
        """Values of ``settings.scope_field`` present in the collection (cached)."""
        field = self.settings.scope_field
        if field is None:
            return set()
        if self._scope_values is None:
            self._require_collection()
            self._scope_values = {int(v) for v in self.store.field_values(self.collection, field)}
        return self._scope_values

    # -- query ------------------------------------------------------------------------------

    def retrieve(self, query: str, retrieval: RetrievalConfig | None = None) -> list[SourceChunk]:
        """Retrieve without generating. Useful for inspecting what a strategy returns."""
        config = retrieval or self.settings.retrieval
        reranker = LLMReranker(self.chat_model) if config.rerank else None
        hits = retrieve(self._ctx(), query, config, reranker=reranker)
        return _to_sources(hits)

    def ask(
        self,
        question: str,
        retrieval: RetrievalConfig | None = None,
        *,
        context_format: ContextFormat | None = None,
        guardrail: bool | None = None,
        rewrite: bool | None = None,
    ) -> Answer:
        """Answer ``question`` from the collection.

        Pipeline: scope guardrail → optional rewrite → retrieve (+ payload filter) → optional
        re-rank → format context → generate. Arguments left as ``None`` fall back to settings.

        Returns:
            The answer with its sources, decisions, and per-stage timings.
        """
        config = retrieval or self.settings.retrieval
        context_format = context_format or self.settings.context_format
        use_guardrail = self.settings.guardrail if guardrail is None else guardrail
        use_rewrite = self.settings.rewrite if rewrite is None else rewrite
        timings: dict[str, float] = {}
        started = time.perf_counter()

        decision: GuardrailDecision | None = None
        if use_guardrail and self.settings.scope_field:
            decision = scope_guardrail(question, self.settings.scope_field, self.scope_values())
            if decision.action == "abstain":
                timings["total"] = time.perf_counter() - started
                return Answer(
                    question=question,
                    text=f"I don't know. {decision.reason}",
                    sources=[],
                    collection=self.collection,
                    retrieval=config.label(),
                    context_format=context_format,
                    abstained=True,
                    guardrail=decision,
                    timings=timings,
                )
            if decision.action == "filter":
                config = replace(config, filters={**config.filters, **guardrail_filter(decision)})

        search_query = question
        rewritten: str | None = None
        if use_rewrite:
            t = time.perf_counter()
            rewritten = QueryRewriter(self.chat_model).rewrite(question)
            search_query = rewritten
            timings["rewrite"] = time.perf_counter() - t

        t = time.perf_counter()
        reranker = LLMReranker(self.chat_model) if config.rerank else None
        hits = retrieve(
            self._ctx(), search_query, config, reranker=reranker, rerank_question=question
        )
        timings["retrieve"] = time.perf_counter() - t

        t = time.perf_counter()
        label_fields = [rule.name for rule in self.settings.metadata_fields]
        text = AnswerGenerator(self.chat_model).generate(
            question, [doc for doc, _ in hits], context_format, label_fields
        )
        timings["generate"] = time.perf_counter() - t
        timings["total"] = time.perf_counter() - started

        return Answer(
            question=question,
            text=text,
            sources=_to_sources(hits),
            collection=self.collection,
            retrieval=config.label(),
            context_format=context_format,
            abstained=is_abstention(text),
            rewritten_query=rewritten,
            guardrail=decision,
            timings=timings,
        )


def _to_sources(hits: list) -> list[SourceChunk]:
    return [
        SourceChunk(
            source=doc.metadata.get("source", "unknown"),
            text=doc.page_content,
            rank=rank,
            score=float(score),
            metadata={k: v for k, v in doc.metadata.items() if not k.startswith("_")},
        )
        for rank, (doc, score) in enumerate(hits, start=1)
    ]
