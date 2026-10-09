from langchain_core.documents import Document

from localrag.config import FieldRule, Settings
from localrag.enrichers import (
    apply,
    content_hash,
    default_enrichers,
    extract_fields,
    format_header,
    text_header,
)


def docs(*sources: str) -> list[Document]:
    return [Document(page_content=f"text of {s}", metadata={"source": s}) for s in sources]


def test_extract_fields_casts_ints_and_keeps_strings():
    rules = [FieldRule("year", r"(\d{4})", "int"), FieldRule("team", r"^([a-z]+)/")]
    out = extract_fields(rules)(docs("ops/2033.txt"))
    assert out[0].metadata["year"] == 2033
    assert out[0].metadata["team"] == "ops"


def test_missing_field_is_left_unset_not_an_error():
    out = extract_fields([FieldRule("year", r"(\d{4})", "int")])(docs("README.md"))
    assert "year" not in out[0].metadata


def test_enrichers_do_not_mutate_their_input():
    original = docs("2030.txt")
    extract_fields([FieldRule("year", r"(\d{4})", "int")])(original)
    assert "year" not in original[0].metadata


def test_header_uses_the_original_format():
    assert format_header({"year": 2033}, ["year"]) == "Metadata: year=2033"
    assert format_header({}, ["year"]) == ""
    enriched = text_header(["year"])([Document(page_content="body", metadata={"year": 2030})])
    assert enriched[0].page_content == "Metadata: year=2030\nbody"


def test_content_hash_tracks_the_final_text():
    a, b = content_hash([Document(page_content="x"), Document(page_content="y")])
    assert a.metadata["content_hash"] != b.metadata["content_hash"]
    assert len(a.metadata["content_hash"]) == 16


def test_default_chain_hashes_text_after_the_header():
    plain = apply(docs("2030.txt"), default_enrichers(Settings()))
    header = apply(docs("2030.txt"), default_enrichers(Settings(header_fields=("year",))))
    assert header[0].page_content.startswith("Metadata: year=2030\n")
    assert plain[0].metadata["content_hash"] != header[0].metadata["content_hash"]
