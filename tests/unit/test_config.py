from pathlib import Path

import pytest

from localrag.config import FieldRule, RetrievalConfig, Settings


def test_defaults_reproduce_the_baseline_pipeline():
    s = Settings()
    assert (s.chunk_size, s.chunk_overlap, s.retrieval.k) == (1000, 200, 4)
    assert s.temperature == 0.0 and s.reasoning is False
    assert s.metadata_fields == (FieldRule("year", r"(\d{4})", "int"),)


def test_collection_name_derives_from_data_dir_and_header_variant():
    assert Settings(data_dir="data/Quantum Mind").collection_name == "quantum-mind"
    assert Settings(collection="docs", header_fields=("year",)).collection_name == "docs__hdr-year"


def test_header_fields_must_be_extracted_fields():
    with pytest.raises(ValueError, match="not defined"):
        Settings(header_fields=("author",))


def test_field_rule_requires_a_capture_group():
    with pytest.raises(ValueError, match="capture group"):
        FieldRule("year", r"\d{4}")


def test_chunk_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        Settings(chunk_size=100, chunk_overlap=100)


def test_retrieval_config_validation_and_label():
    with pytest.raises(ValueError):
        RetrievalConfig(k=0)
    with pytest.raises(ValueError):
        RetrievalConfig(bm25_weight=1.5)
    label = RetrievalConfig(strategy="hybrid", filters={"year": 2032}, rerank=True).label()
    assert label == "hybrid k=4 bm25=0.5 filter=year:2032 rerank@8"


def test_toml_env_and_override_precedence(tmp_path: Path, monkeypatch):
    config = tmp_path / "custom.toml"
    config.write_text(
        """
[corpus]
data_dir = "notes"
collection = "notes"

[chunking]
size = 500
overlap = 50

[[metadata.fields]]
name = "team"
pattern = '^([a-z]+)/'

[metadata]
header = ["team"]
scope_field = ""

[retrieval]
strategy = "hybrid"
k = 6
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_CHUNK_SIZE", "800")
    s = Settings.load(config, context_format="tagged")
    assert s.data_dir == "notes"
    assert s.chunk_size == 800  # env beats file
    assert s.chunk_overlap == 50
    assert s.metadata_fields == (FieldRule("team", "^([a-z]+)/"),)
    assert s.header_fields == ("team",)
    assert s.scope_field is None
    assert s.retrieval.strategy == "hybrid" and s.retrieval.k == 6
    assert s.context_format == "tagged"  # explicit override beats everything
    assert s.collection_name == "notes__hdr-team"


def test_unknown_toml_keys_are_rejected(tmp_path: Path):
    config = tmp_path / "bad.toml"
    config.write_text("[chunking]\nsizee = 10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown key"):
        Settings.load(config)


def test_missing_explicit_config_file_is_an_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        Settings.load(tmp_path / "nope.toml")


def test_repository_config_loads():
    root = Path(__file__).resolve().parents[2]
    s = Settings.load(root / "localrag.toml")
    assert s.collection_name == "quantummind"
    assert s.scope_field == "year"
