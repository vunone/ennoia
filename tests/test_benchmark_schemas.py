"""Unit tests for the ESCI product DDI schemas."""

from __future__ import annotations

import pytest

from benchmark.pipelines.schemas import ProductMeta, ProductSummary


def test_product_meta_requires_all_fields() -> None:
    meta = ProductMeta(
        brand="ACME",
        color="black",
        category="electronics",
        price_usd=79,
    )
    assert meta.brand == "ACME"
    assert meta.color == "black"
    assert meta.category == "electronics"
    assert meta.price_usd == 79


def test_product_meta_rejects_missing_fields() -> None:
    with pytest.raises(ValueError):
        ProductMeta(brand="ACME")  # type: ignore[call-arg]


def test_product_meta_rejects_non_int_price() -> None:
    with pytest.raises(ValueError):
        ProductMeta(
            brand="ACME",
            color="black",
            category="electronics",
            price_usd="cheap",  # type: ignore[arg-type]
        )


def test_product_summary_is_semantic_marker() -> None:
    assert ProductSummary.__ennoia_kind__ == "semantic"
    # Docstring is the extraction prompt — must be non-empty.
    assert (ProductSummary.__doc__ or "").strip()
