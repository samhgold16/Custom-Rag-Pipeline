"""Model provider seam.

The engine only needs a LangChain ``BaseChatModel`` and ``Embeddings``. ``ModelProvider`` is the
one place that knows where they come from. Ollama is the only implementation: the project is
local-first by design, and a hosted provider would be a new class here, not a change elsewhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings

from localrag.config import Settings

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """The model backend is unreachable or misconfigured. The message says how to fix it."""


class ModelProvider(Protocol):
    """Builds the chat and embedding models for an engine."""

    def chat_model(self) -> BaseChatModel: ...

    def embeddings(self) -> Embeddings: ...


@dataclass(frozen=True)
class HealthReport:
    """Result of probing the Ollama server.

    Attributes:
        host: URL probed.
        reachable: Whether ``/api/tags`` answered.
        missing: Required models that are not pulled.
        digests: Digest of each required model that is present (recorded in eval reports).
        error: Connection error text, if unreachable.
    """

    host: str
    reachable: bool
    missing: list[str] = field(default_factory=list)
    digests: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.reachable and not self.missing

    def raise_for_status(self) -> None:
        """Raise ``ProviderError`` with a fix-it hint unless the backend is ready."""
        if not self.reachable:
            raise ProviderError(
                f"Cannot reach Ollama at {self.host} ({self.error}). Start it with `ollama serve`."
            )
        if self.missing:
            pulls = "; ".join(f"ollama pull {m}" for m in self.missing)
            raise ProviderError(f"Missing model(s): {', '.join(self.missing)}. Run: {pulls}")


def server_version(host: str, timeout: float = 5.0) -> str | None:
    """Ollama server version, or ``None`` if unreachable."""
    try:
        response = httpx.get(f"{host.rstrip('/')}/api/version", timeout=timeout)
        response.raise_for_status()
        return str(response.json().get("version"))
    except (httpx.HTTPError, ValueError):
        return None


def chat_template_problem(host: str, model: str, timeout: float = 10.0) -> str | None:
    """Detect a chat model that Ollama will run as a bare completion model.

    If Ollama cannot convert a GGUF's chat template (older servers and newer architectures),
    it falls back to ``TEMPLATE {{ .Prompt }}``. Chat requests then silently drop the system
    message, which is where RAG puts the retrieved context: the model answers from nothing.

    Returns:
        A description of the problem, or ``None`` if the template looks usable.
    """
    try:
        response = httpx.post(
            f"{host.rstrip('/')}/api/show", json={"model": model}, timeout=timeout
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    template = str(response.json().get("template", "")).strip()
    if template in ("", "{{ .Prompt }}"):
        return (
            f"{model} has no chat template in this Ollama server ({template or 'empty'}), so the "
            "system prompt and retrieved context would be dropped. Upgrade Ollama."
        )
    return None


def check_ollama(host: str, models: list[str], timeout: float = 5.0) -> HealthReport:
    """Check that Ollama is up and every model in ``models`` is pulled.

    Ollama lists untagged models with an implicit ``:latest`` (``nomic-embed-text`` appears as
    ``nomic-embed-text:latest``), so both spellings count as present.
    """
    try:
        response = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return HealthReport(host=host, reachable=False, error=type(exc).__name__)

    installed = {m["name"]: m.get("digest", "") for m in response.json().get("models", [])}
    digests: dict[str, str] = {}
    missing: list[str] = []
    for model in models:
        name = model if model in installed else f"{model}:latest"
        if name in installed:
            digests[model] = installed[name]
        else:
            missing.append(model)
    return HealthReport(host=host, reachable=True, missing=missing, digests=digests)


class OllamaProvider:
    """Chat and embedding models served by a local Ollama."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def chat_model(self) -> BaseChatModel:
        s = self.settings
        return ChatOllama(
            model=s.chat_model,
            base_url=s.ollama_host,
            temperature=s.temperature,
            reasoning=s.reasoning,
            num_predict=s.num_predict,
        )

    def embeddings(self) -> Embeddings:
        return OllamaEmbeddings(
            model=self.settings.embedding_model, base_url=self.settings.ollama_host
        )

    def health(self) -> HealthReport:
        return check_ollama(
            self.settings.ollama_host, [self.settings.chat_model, self.settings.embedding_model]
        )
