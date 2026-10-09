"""Chunk enrichers: small pure functions ``list[Document] -> list[Document]`` applied at ingest.

Two different things can be done with a piece of metadata, and they fix different problems:

* **Payload field** (``extract_fields``): stored next to the vector. Enables exact payload
  filters and the scope guardrail. Invisible to the embedding and to the LLM unless the context
  formatter shows it.
* **Text header** (``text_header``): written into ``page_content`` before embedding. Both the
  vector and the LLM's view change, which is why header variants live in their own collection.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence

from langchain_core.documents import Document

from localrag.config import FieldRule, Settings

Enricher = Callable[[list[Document]], list[Document]]


def extract_fields(rules: Sequence[FieldRule]) -> Enricher:
    """Set ``metadata[rule.name]`` from a regex over each chunk's ``source`` path.

    Chunks whose path does not match are left without the field, so a corpus that only partly
    follows a naming convention still ingests.
    """
    compiled = [(rule, re.compile(rule.pattern)) for rule in rules]

    def enrich(chunks: list[Document]) -> list[Document]:
        out = []
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            for rule, pattern in compiled:
                match = pattern.search(metadata.get("source", ""))
                if match:
                    value = match.group(1)
                    metadata[rule.name] = int(value) if rule.type == "int" else value
            out.append(Document(page_content=chunk.page_content, metadata=metadata))
        return out

    return enrich


def format_header(metadata: dict, fields: Sequence[str]) -> str:
    """``"Metadata: year=2033"``: the header format the original grounding fix used."""
    pairs = [f"{name}={metadata[name]}" for name in fields if metadata.get(name) is not None]
    return f"Metadata: {', '.join(pairs)}" if pairs else ""


def text_header(fields: Sequence[str]) -> Enricher:
    """Prepend a ``Metadata: field=value`` line to each chunk's text before it is embedded."""

    def enrich(chunks: list[Document]) -> list[Document]:
        out = []
        for chunk in chunks:
            header = format_header(chunk.metadata, fields)
            text = f"{header}\n{chunk.page_content}" if header else chunk.page_content
            out.append(Document(page_content=text, metadata=dict(chunk.metadata)))
        return out

    return enrich


def content_hash(chunks: list[Document]) -> list[Document]:
    """Record a short hash of the final chunk text, used in the deterministic point ID."""
    out = []
    for chunk in chunks:
        digest = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[:16]
        out.append(
            Document(
                page_content=chunk.page_content, metadata={**chunk.metadata, "content_hash": digest}
            )
        )
    return out


def default_enrichers(settings: Settings) -> list[Enricher]:
    """Enricher chain for ``settings``. Order matters: the header needs the extracted fields,
    and the hash must see the final text."""
    chain: list[Enricher] = [extract_fields(settings.metadata_fields)]
    if settings.header_fields:
        chain.append(text_header(settings.header_fields))
    chain.append(content_hash)
    return chain


def apply(chunks: list[Document], enrichers: Sequence[Enricher]) -> list[Document]:
    for enrich in enrichers:
        chunks = enrich(chunks)
    return chunks
