from collections import Counter
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from localrag import Settings
from localrag.ingest import load_documents, point_id, prepare_chunks, split_documents
from tests.conftest import DEMO_CORPUS, build_engine


def test_load_documents_is_sorted_relative_and_utf8(tmp_path: Path):
    (tmp_path / "b.md").write_text("# B – dash", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("x", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert [d.metadata["source"] for d in docs] == ["b.md", "sub/a.txt"]
    assert docs[0].page_content == "# B – dash"


def test_load_documents_errors_are_explicit(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_documents(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="No files"):
        load_documents(tmp_path)


def test_demo_corpus_chunks_as_documented():
    chunks = split_documents(load_documents(DEMO_CORPUS), 1000, 200)
    per_file = Counter(c.metadata["source"] for c in chunks)
    assert len(chunks) == 14
    assert per_file == {
        "2030.txt": 3,
        "2031.txt": 2,
        "2032.txt": 2,
        "2033.txt": 2,
        "2034.txt": 2,
        "2035.txt": 3,
    }
    assert [c.metadata["chunk_index"] for c in chunks if c.metadata["source"] == "2030.txt"] == [
        0,
        1,
        2,
    ]


def test_point_ids_are_deterministic():
    settings = Settings(data_dir=str(DEMO_CORPUS))
    first = [point_id(c) for c in prepare_chunks(settings)]
    second = [point_id(c) for c in prepare_chunks(settings)]
    assert first == second and len(set(first)) == 14


def test_ingest_is_idempotent(corpus: Path):
    engine = build_engine(corpus)
    first = engine.ingest()
    second = engine.ingest()
    assert first.rebuilt and first.added == first.chunks
    assert second.unchanged and second.chunks == first.chunks
    assert first.field_values == {"year": [2030, 2031, 2032, 2033]}


def test_editing_one_file_only_touches_its_chunks(corpus: Path):
    engine = build_engine(corpus)
    engine.ingest()
    (corpus / "2031.txt").write_text("Revised 2031 letter.", encoding="utf-8")
    report = engine.ingest()
    assert (report.added, report.removed, report.rebuilt) == (1, 1, False)
    sources = [d.metadata["source"] for d in engine.store.documents(engine.collection)]
    assert sources.count("2031.txt") == 1


def test_deleted_and_new_files_are_synced(corpus: Path):
    engine = build_engine(corpus)
    engine.ingest()
    (corpus / "2030.txt").unlink()
    (corpus / "2034.md").write_text("CEO reinstated after a breakthrough.", encoding="utf-8")
    report = engine.ingest()
    assert (report.added, report.removed) == (1, 1)
    assert report.field_values["year"] == [2031, 2032, 2033, 2034]


def test_changing_the_embedding_model_forces_a_rebuild(corpus: Path):
    engine = build_engine(corpus)
    engine.ingest()
    other = engine.variant(embedding_model="another-embedder")
    other._embeddings = DeterministicFakeEmbedding(size=16)
    report = other.ingest()
    assert report.rebuilt and report.vector_size == 16


def test_header_variant_is_a_separate_collection(corpus: Path):
    engine = build_engine(corpus)
    engine.ingest()
    header = engine.variant(header_fields=("year",))
    report = header.ingest()
    assert report.collection.endswith("__hdr-year")
    text = header.store.documents(header.collection)[0].page_content
    assert text.startswith("Metadata: year=")
    assert engine.store.count(engine.collection) == header.store.count(header.collection)
