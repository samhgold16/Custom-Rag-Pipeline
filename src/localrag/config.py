"""Configuration: defaults < ``localrag.toml`` < ``RAG_*`` env vars < explicit overrides.

Everything corpus-specific (where the files live, which metadata to pull out of file paths,
which field scopes the guardrail) is configuration, so pointing the engine at a new domain
is a TOML file, not a code change.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

ContextFormat = Literal["plain", "header", "tagged"]

DEFAULT_CONFIG_FILE = "localrag.toml"


@dataclass(frozen=True)
class FieldRule:
    """Extract one metadata field from each document's path with a regex.

    The first capture group becomes the value. Documents whose path does not match are left
    without the field rather than rejected, so mixed corpora still ingest.

    Attributes:
        name: Metadata key to set, e.g. ``"year"``.
        pattern: Regex searched against the path relative to the data directory.
        type: ``"int"`` casts the match; ``"str"`` keeps it as text.
    """

    name: str
    pattern: str
    type: Literal["int", "str"] = "str"

    def __post_init__(self) -> None:
        compiled = re.compile(self.pattern)
        if compiled.groups < 1:
            raise ValueError(f"FieldRule {self.name!r}: pattern needs a capture group")
        if self.type not in ("int", "str"):
            raise ValueError(f"FieldRule {self.name!r}: type must be 'int' or 'str'")


@dataclass(frozen=True)
class RetrievalConfig:
    """How to fetch context for one question.

    Attributes:
        strategy: A registered strategy name. Built in: ``dense`` (cosine kNN), ``mmr``
            (maximal marginal relevance), ``hybrid`` (BM25 + dense with reciprocal-rank fusion).
        k: Chunks handed to the generator.
        fetch_k: Candidate pool for MMR and for each ranked list in hybrid.
        lambda_mult: MMR trade-off; 1.0 is pure relevance, 0.0 pure diversity.
        bm25_weight: Weight of the lexical list in RRF; the dense list gets ``1 - bm25_weight``.
        rrf_c: RRF smoothing constant (60 in Cormack et al., 2009).
        filters: Exact-match payload filters, ``{field: value}`` or ``{field: [values]}``.
            Applied inside the vector search, so they narrow the space rather than post-filter.
        rerank: Re-score candidates with the LLM and keep the top ``k``.
        rerank_candidates: How many chunks to retrieve before re-ranking.
    """

    strategy: str = "dense"
    k: int = 4
    fetch_k: int = 20
    lambda_mult: float = 0.5
    bm25_weight: float = 0.5
    rrf_c: int = 60
    filters: dict[str, Any] = field(default_factory=dict)
    rerank: bool = False
    rerank_candidates: int = 8

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if not 0.0 <= self.bm25_weight <= 1.0:
            raise ValueError("bm25_weight must be in [0, 1]")
        if not 0.0 <= self.lambda_mult <= 1.0:
            raise ValueError("lambda_mult must be in [0, 1]")

    def label(self) -> str:
        """Short human-readable description used in reports and the CLI."""
        parts = [self.strategy, f"k={self.k}"]
        if self.strategy == "mmr":
            parts.append(f"λ={self.lambda_mult}")
        if self.strategy == "hybrid":
            parts.append(f"bm25={self.bm25_weight}")
        if self.filters:
            parts.append("filter=" + ",".join(f"{k}:{v}" for k, v in self.filters.items()))
        if self.rerank:
            parts.append(f"rerank@{self.rerank_candidates}")
        return " ".join(parts)


@dataclass(frozen=True)
class Settings:
    """Engine configuration. Defaults reproduce the original baseline pipeline.

    Attributes:
        ollama_host: Ollama base URL. Read from the unprefixed ``OLLAMA_HOST``, as Ollama does.
        chat_model: Generator model tag.
        embedding_model: Embedding model tag.
        temperature: 0 for repeatable evaluation (the MiniCPM5 card suggests 1.0 for chat).
        reasoning: MiniCPM5 is a thinking model; ``False`` keeps its chain of thought out of
            the answer text, which the abstention and re-rank parsers depend on.
        num_predict: Generation token cap.
        data_dir: Folder of documents to index.
        globs: File patterns to load, relative to ``data_dir``.
        qdrant_path: Embedded Qdrant folder, or ``":memory:"``.
        collection: Base collection name; defaults to the data folder's name.
        chunk_size: Characters per chunk.
        chunk_overlap: Characters shared between consecutive chunks.
        metadata_fields: Rules that lift metadata out of file paths.
        header_fields: Fields written into each chunk's text *before embedding*
            (``"Metadata: year=2033"``). Changes the vectors, so it gets its own collection.
        scope_field: Integer year field the guardrail checks questions against; ``None``
            disables the guardrail.
        context_format: How chunks are shown to the generator, decided at prompt time and
            independent of what was embedded: ``plain`` text; ``header``, a
            ``Metadata: field=value`` line per chunk; or ``tagged``, a ``[source | field]`` label
            plus an instruction to cite it.
        guardrail: Apply the scope guardrail in ``ask``.
        rewrite: Rewrite the query with the LLM before retrieval.
        retrieval: Default retrieval configuration.
    """

    ollama_host: str = "http://localhost:11434"
    chat_model: str = "hf.co/openbmb/MiniCPM5-2B-GGUF:Q4_K_M"
    embedding_model: str = "nomic-embed-text"
    temperature: float = 0.0
    reasoning: bool = False
    num_predict: int = 4096

    data_dir: str = "data/quantummind"
    globs: tuple[str, ...] = ("**/*.txt", "**/*.md")
    qdrant_path: str = "./qdrant_db"
    collection: str | None = None
    chunk_size: int = 1000
    chunk_overlap: int = 200

    metadata_fields: tuple[FieldRule, ...] = (FieldRule("year", r"(\d{4})", "int"),)
    header_fields: tuple[str, ...] = ()
    scope_field: str | None = "year"

    context_format: ContextFormat = "header"
    guardrail: bool = True
    rewrite: bool = False
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.context_format not in ("plain", "header", "tagged"):
            raise ValueError(f"Unknown context_format: {self.context_format!r}")
        known = {rule.name for rule in self.metadata_fields}
        missing = [f for f in self.header_fields if f not in known]
        if missing:
            raise ValueError(f"header_fields {missing} are not defined in metadata_fields")

    @property
    def collection_name(self) -> str:
        """Collection for this ingest variant.

        Variants that change the embedded text get a suffix, so they can sit side by side in
        one Qdrant folder and be compared without re-embedding.
        """
        base = self.collection or _slug(Path(self.data_dir).name) or "documents"
        if self.header_fields:
            return f"{base}__hdr-{'-'.join(self.header_fields)}"
        return base

    @classmethod
    def load(cls, config_path: str | Path | None = None, **overrides: Any) -> Settings:
        """Build settings from the TOML file, then the environment, then ``overrides``.

        Args:
            config_path: Explicit TOML path. If ``None``, ``./localrag.toml`` is used when present.
            **overrides: Final values; ``None`` entries are ignored.

        Returns:
            The merged settings.
        """
        load_dotenv()
        values: dict[str, Any] = {}
        path = Path(config_path) if config_path else Path(DEFAULT_CONFIG_FILE)
        if config_path and not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        if path.exists():
            values.update(_from_toml(tomllib.loads(path.read_text(encoding="utf-8"))))
        values.update(_from_env())
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)


_ENV_FIELDS: dict[str, tuple[str, type]] = {
    "OLLAMA_HOST": ("ollama_host", str),
    "RAG_CHAT_MODEL": ("chat_model", str),
    "RAG_EMBEDDING_MODEL": ("embedding_model", str),
    "RAG_TEMPERATURE": ("temperature", float),
    "RAG_DATA_DIR": ("data_dir", str),
    "RAG_QDRANT_PATH": ("qdrant_path", str),
    "RAG_COLLECTION": ("collection", str),
    "RAG_CHUNK_SIZE": ("chunk_size", int),
    "RAG_CHUNK_OVERLAP": ("chunk_overlap", int),
    "RAG_CONTEXT_FORMAT": ("context_format", str),
}


def _from_env() -> dict[str, Any]:
    return {
        name: cast(os.environ[var])
        for var, (name, cast) in _ENV_FIELDS.items()
        if os.environ.get(var)
    }


def _from_toml(doc: dict[str, Any]) -> dict[str, Any]:
    """Map the sectioned TOML layout onto flat ``Settings`` fields."""
    out: dict[str, Any] = {}
    sections = {
        "models": {
            "host": "ollama_host",
            "chat": "chat_model",
            "embedding": "embedding_model",
            "temperature": "temperature",
            "reasoning": "reasoning",
            "num_predict": "num_predict",
        },
        "corpus": {
            "data_dir": "data_dir",
            "globs": "globs",
            "collection": "collection",
            "qdrant_path": "qdrant_path",
        },
        "chunking": {"size": "chunk_size", "overlap": "chunk_overlap"},
        "metadata": {"header": "header_fields", "scope_field": "scope_field"},
        "generation": {
            "context_format": "context_format",
            "guardrail": "guardrail",
            "rewrite": "rewrite",
        },
    }
    unknown_sections = set(doc) - set(sections) - {"retrieval"}
    if unknown_sections:
        raise ValueError(f"Unknown config sections: {sorted(unknown_sections)}")
    for section, mapping in sections.items():
        table = doc.get(section, {})
        for key, value in table.items():
            if key == "fields" and section == "metadata":
                out["metadata_fields"] = tuple(FieldRule(**rule) for rule in value)
                continue
            if key not in mapping:
                raise ValueError(f"Unknown key [{section}].{key}")
            out[mapping[key]] = tuple(value) if isinstance(value, list) else value
    if "scope_field" in out and out["scope_field"] == "":
        out["scope_field"] = None
    if "retrieval" in doc:
        allowed = {f.name for f in fields(RetrievalConfig)}
        bad = set(doc["retrieval"]) - allowed
        if bad:
            raise ValueError(f"Unknown keys in [retrieval]: {sorted(bad)}")
        out["retrieval"] = RetrievalConfig(**doc["retrieval"])
    return out


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", text.lower()).strip("-")
