"""Unit tests for ennoia.utils.ids — semantic vector id conventions."""

from __future__ import annotations

from ennoia.utils.ids import (
    SEMANTIC_VECTOR_ID_SEP,
    extract_source_id,
    make_semantic_vector_id,
    parse_semantic_vector_id,
)


def test_separator_is_colon():
    assert SEMANTIC_VECTOR_ID_SEP == ":"


def test_make_and_parse_round_trip():
    vid = make_semantic_vector_id("doc_001", "Summary")
    assert vid == "doc_001:Summary"
    assert parse_semantic_vector_id(vid) == ("doc_001", "Summary")


def test_parse_without_separator_returns_whole_id():
    assert parse_semantic_vector_id("doc_001") == ("doc_001", "")


def test_extract_source_id_prefers_metadata():
    vid = make_semantic_vector_id("wrong", "Idx")
    assert extract_source_id({"source_id": "right"}, vid) == "right"


def test_extract_source_id_falls_back_to_vector_id():
    vid = make_semantic_vector_id("doc_001", "Summary")
    assert extract_source_id({}, vid) == "doc_001"


def test_extract_source_id_ignores_non_string_metadata_value():
    vid = make_semantic_vector_id("doc_001", "Summary")
    assert extract_source_id({"source_id": 123}, vid) == "doc_001"


def test_make_semantic_vector_id_permits_empty_components():
    # Empty source or index yields a valid round-trippable id as long as the
    # separator is present. Stricter validation is a Stage 2 concern.
    assert make_semantic_vector_id("", "Idx") == ":Idx"
    assert make_semantic_vector_id("doc", "") == "doc:"
    assert parse_semantic_vector_id(":Idx") == ("", "Idx")
    assert parse_semantic_vector_id("doc:") == ("doc", "")


def test_parse_semantic_vector_id_splits_on_first_separator_only():
    # Index names are free-form; embedded colons stay with the index half.
    assert parse_semantic_vector_id("doc:layer:sub") == ("doc", "layer:sub")


def test_extract_source_id_returns_empty_string_metadata_as_is():
    # Current contract: any str metadata wins over the vector_id fallback.
    vid = make_semantic_vector_id("doc_001", "Summary")
    assert extract_source_id({"source_id": ""}, vid) == ""


def test_extract_source_id_with_plain_vector_id_falls_back():
    # Vector id without separator: parsing returns it whole, extract uses that.
    assert extract_source_id({}, "doc_001") == "doc_001"
