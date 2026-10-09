from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from localrag.store import Store, StoreLockedError, metadata_filter, open_client


def test_metadata_filter_targets_the_nested_payload_key():
    flt = metadata_filter({"year": 2032, "team": ["ops", "ml"]})
    assert isinstance(flt, models.Filter)
    keys = [c.key for c in flt.must]
    assert keys == ["metadata.year", "metadata.team"]
    assert isinstance(flt.must[0].match, models.MatchValue)
    assert isinstance(flt.must[1].match, models.MatchAny)
    assert metadata_filter({}) is None


def test_second_process_style_open_is_translated(tmp_path: Path):
    path = str(tmp_path / "db")
    holder = QdrantClient(path=path)  # simulates another process holding the folder
    try:
        with pytest.raises(StoreLockedError, match="one process at a time"):
            open_client(path)
    finally:
        holder.close()


def test_clients_are_shared_per_path(tmp_path: Path):
    path = str(tmp_path / "shared")
    assert open_client(path) is open_client(path)


def test_manifest_round_trips_on_disk(tmp_path: Path):
    path = str(tmp_path / "manifest_db")
    store = Store.from_path(path)
    store.write_manifest("c", {"vector_size": 768})
    assert Store(store.client, path).manifest("c") == {"vector_size": 768}
    assert (tmp_path / "manifest_db" / "localrag_manifest.json").exists()


def test_in_memory_manifest():
    store = Store(QdrantClient(":memory:"), None)
    assert store.manifest("c") is None
    store.write_manifest("c", {"chunks": 3})
    assert store.manifest("c") == {"chunks": 3}
