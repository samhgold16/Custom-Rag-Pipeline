"""localrag: a local-first RAG engine with pluggable retrieval and measurable grounding."""

from localrag.config import FieldRule, RetrievalConfig, Settings
from localrag.engine import RAGEngine
from localrag.types import Answer, GuardrailDecision, IngestReport, SourceChunk

__all__ = [
    "Answer",
    "FieldRule",
    "GuardrailDecision",
    "IngestReport",
    "RAGEngine",
    "RetrievalConfig",
    "Settings",
    "SourceChunk",
]

__version__ = "0.2.0"
