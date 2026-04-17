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
    assert parse_semantic_vector_id(vid) == ("doc_001", "Summary", None)


def test_make_and_parse_round_trip_with_unique():
    vid = make_semantic_vector_id("doc_001", "ContractParties", "abc123")
    assert vid == "doc_001:ContractParties:abc123"
    assert parse_semantic_vector_id(vid) == ("doc_001", "ContractParties", "abc123")


def test_parse_without_separator_returns_whole_id():
    assert parse_semantic_vector_id("doc_001") == ("doc_001", "", None)


def test_extract_source_id_prefers_metadata():
    vid = make_semantic_vector_id("wrong", "Idx")
    assert extract_source_id({"source_id": "right"}, vid) == "right"


def test_extract_source_id_falls_back_to_vector_id():
    vid = make_semantic_vector_id("doc_001", "Summary")
    assert extract_source_id({}, vid) == "doc_001"


def test_extract_source_id_falls_back_to_triple_vector_id():
    vid = make_semantic_vector_id("doc_001", "ContractParties", "hash1")
    assert extract_source_id({}, vid) == "doc_001"


def test_extract_source_id_ignores_non_string_metadata_value():
    vid = make_semantic_vector_id("doc_001", "Summary")
    assert extract_source_id({"source_id": 123}, vid) == "doc_001"


def test_make_semantic_vector_id_permits_empty_components():
    assert make_semantic_vector_id("", "Idx") == ":Idx"
    assert make_semantic_vector_id("doc", "") == "doc:"
    assert parse_semantic_vector_id(":Idx") == ("", "Idx", None)
    assert parse_semantic_vector_id("doc:") == ("doc", "", None)


def test_parse_semantic_vector_id_limits_to_three_parts():
    # Only the first two separators are consumed; anything after stays as the
    # unique suffix. This keeps unique keys free-form (user-defined).
    assert parse_semantic_vector_id("doc:Idx:unique:with:colons") == (
        "doc",
        "Idx",
        "unique:with:colons",
    )


def test_extract_source_id_returns_empty_string_metadata_as_is():
    vid = make_semantic_vector_id("doc_001", "Summary")
    assert extract_source_id({"source_id": ""}, vid) == ""


def test_extract_source_id_with_plain_vector_id_falls_back():
    assert extract_source_id({}, "doc_001") == "doc_001"
