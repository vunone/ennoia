"""IndexResult.summary() — confidence stripping, semantic truncation."""

from __future__ import annotations

from typing import Literal

from ennoia import BaseStructure
from ennoia.index.extractor import CONFIDENCE_KEY
from ennoia.index.result import IndexResult, SearchHit, SearchResult


class _Doc(BaseStructure):
    """Doc metadata."""

    cat: Literal["legal", "medical"]


def _instance_with_confidence() -> _Doc:
    inst = _Doc(cat="legal")
    # ``extra='allow'`` means we can stash the ridealong confidence on the model.
    setattr(inst, CONFIDENCE_KEY, 0.91)
    return inst


def test_summary_strips_confidence_from_structural_payload() -> None:
    result = IndexResult(
        source_id="doc_1",
        structural={"Doc": _instance_with_confidence()},
        confidences={"Doc": 0.91},
    )
    summary = result.summary()
    assert CONFIDENCE_KEY not in summary["structural"]["Doc"]
    assert summary["structural"]["Doc"] == {"cat": "legal"}
    assert summary["confidences"] == {"Doc": 0.91}
    assert summary["rejected"] is False


def test_summary_truncates_long_semantic_answers() -> None:
    long_text = "x" * 200
    short_text = "short"
    result = IndexResult(
        source_id="doc_1",
        semantic={"Long": long_text, "Short": short_text},
    )
    summary = result.summary()
    assert summary["semantic"]["Long"].endswith("...")
    assert len(summary["semantic"]["Long"]) == 120 + len("...")
    assert summary["semantic"]["Short"] == "short"


def test_summary_rejected_flag_passes_through() -> None:
    result = IndexResult(source_id="x", rejected=True)
    assert result.summary()["rejected"] is True


def test_search_result_supports_len_iter_bool() -> None:
    empty = SearchResult()
    assert len(empty) == 0
    assert bool(empty) is False
    assert list(empty) == []

    populated = SearchResult(hits=[SearchHit(source_id="a", score=1.0, structural={"x": 1})])
    assert len(populated) == 1
    assert bool(populated) is True
    assert [h.source_id for h in populated] == ["a"]
