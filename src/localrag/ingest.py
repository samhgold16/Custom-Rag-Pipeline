"""Ingest: load → split → enrich → embed → upsert, idempotently.

Point IDs are content-addressed (``uuid5`` of source, chunk index, and text hash), so the set of
IDs *is* the desired state of the collection. Each run diffs it against what is stored: new IDs
are embedded and upserted, vanished IDs are deleted, and an unchanged corpus costs no embedding
calls at all. A change of embedding model or vector size forces a full rebuild, because old and
new vectors would not be comparable.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from localrag import enrichers
from localrag.config import Settings
from localrag.store import Store
from localrag.types import IngestReport

logger = logging.getLogger(__name__)

# Fixed namespace so the same chunk always maps to the same point ID across machines.
POINT_NAMESPACE = uuid.UUID("6f1d3c2e-8a4b-4f5e-9c7d-2b1a0e9f8d7c")


def load_documents(
    data_dir: str | Path, globs: tuple[str, ...] = ("**/*.txt", "**/*.md")
) -> list[Document]:
    """Read every matching file as UTF-8, sorted by path so runs are repeatable.

    ``metadata["source"]`` is the POSIX path relative to ``data_dir``, which keeps payloads and
    point IDs stable no matter where the repository is checked out.

    Raises:
        FileNotFoundError: if ``data_dir`` does not exist or contains no matching files.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {root}")
    paths = sorted({p for pattern in globs for p in root.glob(pattern) if p.is_file()})
    if not paths:
        raise FileNotFoundError(f"No files matching {list(globs)} in {root}")
    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": path.relative_to(root).as_posix()},
        )
        for path in paths
    ]


def split_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Recursive character splitting, with a per-document ``chunk_index``."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    counters: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        source = chunk.metadata["source"]
        chunk.metadata["chunk_index"] = counters[source]
        counters[source] += 1
    return chunks


def point_id(chunk: Document) -> str:
    m = chunk.metadata
    return str(uuid.uuid5(POINT_NAMESPACE, f"{m['source']}:{m['chunk_index']}:{m['content_hash']}"))


def prepare_chunks(settings: Settings) -> list[Document]:
    """Everything up to (not including) embedding. Cheap, deterministic, and side-effect free."""
    documents = load_documents(settings.data_dir, settings.globs)
    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)
    return enrichers.apply(chunks, enrichers.default_enrichers(settings))


def ingest(
    settings: Settings, store: Store, embeddings: Embeddings, *, rebuild: bool = False
) -> IngestReport:
    """Bring ``settings.collection_name`` in line with the files in ``settings.data_dir``.

    Args:
        settings: Corpus, chunking, and enrichment configuration.
        store: Target Qdrant store.
        embeddings: Embedding model.
        rebuild: Drop and re-embed everything even if nothing changed.

    Returns:
        A report of what changed.
    """
    started = time.perf_counter()
    collection = settings.collection_name
    chunks = prepare_chunks(settings)
    desired = {point_id(c): c for c in chunks}

    manifest = store.manifest(collection) or {}
    same_model = manifest.get("embedding_model") == settings.embedding_model
    must_rebuild = rebuild or not store.exists(collection) or not same_model

    if must_rebuild:
        vector_size = len(embeddings.embed_query("dimension probe"))
        store.recreate(collection, vector_size)
        existing: set[str] = set()
    else:
        vector_size = int(manifest["vector_size"])
        existing = store.point_ids(collection)

    to_add = [cid for cid in desired if cid not in existing]
    to_remove = sorted(existing - desired.keys())

    store.delete_points(collection, to_remove)
    if to_add:
        logger.info("Embedding %d chunk(s) into %s", len(to_add), collection)
        store.vectorstore(collection, embeddings).add_texts(
            [desired[cid].page_content for cid in to_add],
            metadatas=[desired[cid].metadata for cid in to_add],
            ids=to_add,
        )

    field_values = {
        rule.name: sorted({c.metadata[rule.name] for c in chunks if rule.name in c.metadata})
        for rule in settings.metadata_fields
    }
    store.write_manifest(
        collection,
        {
            "data_dir": settings.data_dir,
            "embedding_model": settings.embedding_model,
            "vector_size": vector_size,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "header_fields": list(settings.header_fields),
            "metadata_fields": [rule.name for rule in settings.metadata_fields],
            "chunks": len(desired),
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    report = IngestReport(
        collection=collection,
        documents=len({c.metadata["source"] for c in chunks}),
        chunks=store.count(collection),
        added=len(to_add),
        removed=len(to_remove),
        rebuilt=must_rebuild,
        vector_size=vector_size,
        field_values=field_values,
        seconds=time.perf_counter() - started,
    )
    logger.info(
        "Ingested %s: %d chunks (+%d / -%d)%s",
        collection,
        report.chunks,
        report.added,
        report.removed,
        " [rebuilt]" if report.rebuilt else "",
    )
    return report
