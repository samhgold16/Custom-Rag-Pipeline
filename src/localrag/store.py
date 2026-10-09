"""Qdrant lifecycle, collection management, and the ingest manifest.

Qdrant runs in embedded ("local") mode: the client *is* the database, persisted to a folder,
with no server or container. Two consequences shape this module:

* Only one client may open a folder, and the lock is held for the life of the process. Clients
  are therefore cached per path and shared by every collection.
* Local mode is a pure-Python/NumPy implementation intended for small corpora. Filters work by
  scan (payload indexes are a no-op locally); pointing ``QdrantClient`` at a server is the
  scale-up path and needs no other change here.
"""

from __future__ import annotations

import atexit
import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

# LangChain nests document metadata under this payload key, so a filter on "year" must be
# written as "metadata.year".
METADATA_KEY = "metadata"
MANIFEST_FILE = "localrag_manifest.json"

_clients: dict[str, QdrantClient] = {}


class StoreLockedError(RuntimeError):
    """Another process holds the embedded Qdrant folder."""


def open_client(path: str) -> QdrantClient:
    """Return this process's client for ``path``, creating it on first use.

    ``":memory:"`` always returns a fresh, independent client.

    Raises:
        StoreLockedError: if another process already has the folder open.
    """
    if path == ":memory:":
        return QdrantClient(":memory:")
    key = str(Path(path).resolve())
    if key not in _clients:
        try:
            _clients[key] = QdrantClient(path=path)
        except RuntimeError as exc:
            if "already accessed by another instance" not in str(exc):
                raise
            raise StoreLockedError(
                f"Another process is using the Qdrant folder {path}. Embedded Qdrant allows one "
                "process at a time: stop the other command and retry."
            ) from exc
    return _clients[key]


@atexit.register
def _close_clients() -> None:
    # Closing during normal interpreter shutdown is too late: qdrant-client then fails with
    # "ImportError: sys.meta_path is None" and prints a traceback after the program's output.
    while _clients:
        _, client = _clients.popitem()
        client.close()


def metadata_filter(filters: dict[str, Any]) -> models.Filter | None:
    """Translate ``{field: value | [values]}`` into a Qdrant payload filter (AND of fields)."""
    if not filters:
        return None
    conditions: list[models.Condition] = []
    for name, value in filters.items():
        if isinstance(value, list | tuple | set):
            match: models.MatchValue | models.MatchAny = models.MatchAny(any=list(value))
        else:
            match = models.MatchValue(value=value)
        conditions.append(models.FieldCondition(key=f"{METADATA_KEY}.{name}", match=match))
    return models.Filter(must=conditions)


class Store:
    """A Qdrant client plus a small JSON manifest describing how each collection was built.

    The manifest lets ``ingest`` tell an unchanged corpus from one that needs work, and makes a
    rebuild mandatory when the embedding model or vector size changes (old vectors would be
    silently incomparable with new queries).
    """

    def __init__(self, client: QdrantClient, manifest_dir: str | None) -> None:
        self.client = client
        self._manifest_path = Path(manifest_dir) / MANIFEST_FILE if manifest_dir else None
        self._memory_manifest: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_path(cls, path: str) -> Store:
        return cls(open_client(path), None if path == ":memory:" else path)

    # -- manifest ---------------------------------------------------------------------------

    def _read_manifest(self) -> dict[str, dict[str, Any]]:
        if self._manifest_path is None:
            return self._memory_manifest
        if not self._manifest_path.exists():
            return {}
        return json.loads(self._manifest_path.read_text(encoding="utf-8"))

    def manifest(self, collection: str) -> dict[str, Any] | None:
        return self._read_manifest().get(collection)

    def write_manifest(self, collection: str, entry: dict[str, Any]) -> None:
        data = self._read_manifest()
        data[collection] = entry
        if self._manifest_path is None:
            self._memory_manifest = data
            return
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    # -- collections ------------------------------------------------------------------------

    def exists(self, collection: str) -> bool:
        return self.client.collection_exists(collection)

    def count(self, collection: str) -> int:
        return self.client.count(collection).count if self.exists(collection) else 0

    def recreate(self, collection: str, vector_size: int) -> None:
        if self.exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def vectorstore(self, collection: str, embeddings: Embeddings) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.client,
            collection_name=collection,
            embedding=embeddings,
            metadata_payload_key=METADATA_KEY,
        )

    def point_ids(self, collection: str) -> set[str]:
        return {str(p.id) for p in self._scroll(collection, with_payload=False)}

    def delete_points(self, collection: str, ids: list[str]) -> None:
        if ids:
            self.client.delete(collection, points_selector=models.PointIdsList(points=ids))

    def documents(self, collection: str) -> list[Document]:
        """Every stored chunk as a ``Document``, ordered by source then chunk index.

        Used to rebuild the in-memory BM25 index after a restart without re-reading files.
        """
        docs = [
            Document(
                page_content=p.payload.get("page_content", ""),
                metadata={**p.payload.get(METADATA_KEY, {}), "_id": str(p.id)},
            )
            for p in self._scroll(collection, with_payload=True)
            if p.payload
        ]
        docs.sort(key=lambda d: (d.metadata.get("source", ""), d.metadata.get("chunk_index", 0)))
        return docs

    def field_values(self, collection: str, field: str) -> set[Any]:
        """Distinct values of one metadata field across the collection."""
        return {
            d.metadata[field]
            for d in self.documents(collection)
            if d.metadata.get(field) is not None
        }

    def _scroll(self, collection: str, with_payload: bool) -> list[models.Record]:
        records: list[models.Record] = []
        offset = None
        while True:
            batch, offset = self.client.scroll(
                collection, limit=256, offset=offset, with_payload=with_payload, with_vectors=False
            )
            records.extend(batch)
            if offset is None:
                return records
