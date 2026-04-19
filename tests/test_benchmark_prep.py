"""Unit tests for the ESCI prep script.

Tests cover: streaming slice determinism under the seeded start offset,
locale filtering, empty-bullet normalisation, per-band price rule
(one band carries a ``under $X`` ceiling with X > actual price, the
other two forbid price), JSON parsing of fenced LLM replies, seeded
price/band assignment, and threads-cap respect.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from benchmark.data import prep
from benchmark.data.prep import Product

# -- _normalise_bullets -----------------------------------------------------


def test_normalise_bullets_handles_json_list() -> None:
    raw = json.dumps(["a", "b", " c "])
    assert prep._normalise_bullets(raw) == ["a", "b", "c"]


def test_normalise_bullets_handles_actual_list() -> None:
    assert prep._normalise_bullets(["  x", "", "y"]) == ["x", "y"]


def test_normalise_bullets_falls_back_to_newlines() -> None:
    assert prep._normalise_bullets("alpha\nbeta\n\ngamma") == ["alpha", "beta", "gamma"]


def test_normalise_bullets_none_and_empty() -> None:
    assert prep._normalise_bullets(None) == []
    assert prep._normalise_bullets("") == []
    assert prep._normalise_bullets("   ") == []


# -- _assign_price + _pick_price_band ---------------------------------------


def test_assign_price_in_range_and_deterministic() -> None:
    a = prep._assign_price("abc", 42)
    b = prep._assign_price("abc", 42)
    assert a == b
    assert prep._PRICE_MIN_USD <= a <= prep._PRICE_MAX_USD


def test_assign_price_varies_by_docid_and_seed() -> None:
    prices = {prep._assign_price(f"p{i}", 42) for i in range(100)}
    # A deterministic log-uniform draw over 100 ids should give many distinct
    # integers — guard against accidentally collapsing all products to one price.
    assert len(prices) > 50
    # Different seed → different distribution for the same docid.
    assert prep._assign_price("p0", 42) != prep._assign_price("p0", 43)


def test_pick_price_band_is_deterministic_and_covers_all_bands() -> None:
    assert prep._pick_price_band("abc", 42) == prep._pick_price_band("abc", 42)
    bands = {prep._pick_price_band(f"p{i}", 42) for i in range(60)}
    # With 60 draws over 3 bands every band should turn up at least once.
    assert bands == {"broad", "medium", "high"}


# -- _record_to_product ------------------------------------------------------


def test_record_to_product_filters_non_us() -> None:
    record = {"docid": "x", "title": "T", "locale": "uk"}
    assert prep._record_to_product(record, seed=1) is None


def test_record_to_product_drops_missing_title_or_docid() -> None:
    assert prep._record_to_product({"docid": "", "title": "T"}, seed=1) is None
    assert prep._record_to_product({"docid": "x", "title": "  "}, seed=1) is None


def test_record_to_product_assigns_price_and_defaults() -> None:
    record = {
        "docid": "p1",
        "title": "Widget",
        "text": "desc",
        "bullet_points": json.dumps(["f1"]),
        "brand": "",
        "color": None,
        "locale": "us",
    }
    product = prep._record_to_product(record, seed=42)
    assert product is not None
    assert product["docid"] == "p1"
    assert product["title"] == "Widget"
    assert product["bullet_points"] == ["f1"]
    assert product["brand"] == "unknown"
    assert product["color"] == "unknown"
    assert product["price_usd"] == prep._assign_price("p1", 42)


# -- _contains_price + _extract_price_ceiling -------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "laptop stand under $100",
        "cheap usb hub",
        "something for less than 50",
        "under 100 dollars",
        "100 usd or below",
        "price-friendly option",
        "an expensive-looking gadget",
    ],
)
def test_price_patterns_flag_common_leaks(text: str) -> None:
    assert prep._contains_price(text)


@pytest.mark.parametrize("text", ["aluminium laptop stand", "rgb mechanical keyboard"])
def test_price_patterns_let_clean_queries_through(text: str) -> None:
    assert not prep._contains_price(text)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("laptop stand under $50", 50.0),
        ("bluetooth speaker below $120", 120.0),
        ("something up to 75", 75.0),
        ("gadget less than $25", 25.0),
    ],
)
def test_extract_price_ceiling_finds_upper_bound(text: str, expected: float) -> None:
    assert prep._extract_price_ceiling(text) == expected


def test_extract_price_ceiling_returns_none_without_ceiling_phrase() -> None:
    assert prep._extract_price_ceiling("over $50 gadget") is None
    assert prep._extract_price_ceiling("no price here") is None


# -- _price_rules_ok ---------------------------------------------------------


def test_price_rules_ok_valid_target_band_with_ceiling_above_actual() -> None:
    parsed = {"broad": "a thing", "medium": "a widget under $50", "high": "acme widget"}
    assert prep._price_rules_ok(parsed, "medium", actual_price=30) is True


def test_price_rules_ok_rejects_price_in_non_target_band() -> None:
    parsed = {"broad": "thing under $50", "medium": "a widget under $50", "high": "acme widget"}
    # Target is medium, but broad also mentions price — reject.
    assert prep._price_rules_ok(parsed, "medium", actual_price=30) is False


def test_price_rules_ok_rejects_target_band_without_price() -> None:
    parsed = {"broad": "a thing", "medium": "a widget", "high": "acme widget"}
    assert prep._price_rules_ok(parsed, "medium", actual_price=30) is False


def test_price_rules_ok_rejects_ceiling_at_or_below_actual_price() -> None:
    parsed = {"broad": "a thing", "medium": "widget under $30", "high": "acme widget"}
    assert prep._price_rules_ok(parsed, "medium", actual_price=30) is False  # not strictly >
    parsed = {"broad": "a thing", "medium": "widget under $25", "high": "acme widget"}
    assert prep._price_rules_ok(parsed, "medium", actual_price=30) is False


def test_price_rules_ok_rejects_target_band_with_non_ceiling_price() -> None:
    # "$50" with no "under/below" phrase — no parseable ceiling.
    parsed = {"broad": "a thing", "medium": "widget $50", "high": "acme widget"}
    assert prep._price_rules_ok(parsed, "medium", actual_price=30) is False


# -- _parse_query_json -------------------------------------------------------


def test_parse_query_json_plain() -> None:
    raw = '{"broad": "a", "medium": "b", "high": "c"}'
    assert prep._parse_query_json(raw) == {"broad": "a", "medium": "b", "high": "c"}


def test_parse_query_json_fenced_block() -> None:
    raw = "```json\n" + json.dumps({"broad": "a", "medium": "b", "high": "c"}) + "\n```"
    assert prep._parse_query_json(raw) == {"broad": "a", "medium": "b", "high": "c"}


def test_parse_query_json_extracts_object_from_prose() -> None:
    raw = 'Sure! Here you go: {"broad":"q1","medium":"q2","high":"q3"}. Enjoy.'
    assert prep._parse_query_json(raw) == {"broad": "q1", "medium": "q2", "high": "q3"}


def test_parse_query_json_rejects_missing_key() -> None:
    raw = '{"broad": "a", "medium": "b"}'
    assert prep._parse_query_json(raw) is None


def test_parse_query_json_rejects_non_dict() -> None:
    assert prep._parse_query_json("[1, 2, 3]") is None


def test_parse_query_json_invalid_json_inside_braces() -> None:
    assert prep._parse_query_json("{not valid json}") is None


def test_parse_query_json_rejects_non_string_value() -> None:
    raw = '{"broad": 1, "medium": "b", "high": "c"}'
    assert prep._parse_query_json(raw) is None


def test_parse_query_json_handles_garbage() -> None:
    assert prep._parse_query_json("no json here") is None


# -- _render_query_prompt ---------------------------------------------------


def _make_product(docid: str = "abc", price: int = 40) -> Product:
    return {
        "docid": docid,
        "title": "Widget",
        "text": "A useful widget.",
        "bullet_points": ["feature one", "feature two"],
        "brand": "ACME",
        "color": "black",
        "price_usd": price,
    }


def test_render_query_prompt_embeds_band_and_price() -> None:
    prompt = prep._render_query_prompt(_make_product(price=40), price_band="medium")
    assert "medium" in prompt
    # Price is referenced in the prompt (as "$40") so the LLM knows the
    # actual price when asked to produce a strictly-greater ceiling.
    assert "$40" in prompt or "40" in prompt
    assert "Widget" in prompt


# -- generate_queries_for_product -------------------------------------------


class _FakeLLM:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls = 0

    async def complete_text(self, prompt: str) -> str:
        self.calls += 1
        return self._replies.pop(0)


async def test_generate_queries_happy_path_with_price_band() -> None:
    llm = _FakeLLM(['{"broad":"a thing","medium":"a widget under $50","high":"acme widget black"}'])
    product = _make_product(price=40)
    result = await prep.generate_queries_for_product(product, "medium", llm)  # type: ignore[arg-type]
    assert result == {
        "docid": "abc",
        "broad": "a thing",
        "medium": "a widget under $50",
        "high": "acme widget black",
        "price_band": "medium",
    }
    assert llm.calls == 1


async def test_generate_queries_retries_on_wrong_band_price() -> None:
    # First reply leaks price in broad (wrong band); second is clean.
    llm = _FakeLLM(
        [
            '{"broad":"thing under $50","medium":"a widget","high":"acme widget"}',
            '{"broad":"a thing","medium":"a widget under $60","high":"acme widget"}',
        ]
    )
    product = _make_product(price=40)
    result = await prep.generate_queries_for_product(product, "medium", llm)  # type: ignore[arg-type]
    assert result is not None
    assert result["price_band"] == "medium"
    assert llm.calls == 2


async def test_generate_queries_retries_when_ceiling_below_actual() -> None:
    # Target-band ceiling is less than the actual price; must retry.
    llm = _FakeLLM(
        [
            '{"broad":"a thing","medium":"widget under $30","high":"acme widget"}',
            '{"broad":"a thing","medium":"widget under $60","high":"acme widget"}',
        ]
    )
    product = _make_product(price=40)
    result = await prep.generate_queries_for_product(product, "medium", llm)  # type: ignore[arg-type]
    assert result is not None
    assert llm.calls == 2


async def test_generate_queries_returns_none_on_persistent_violation() -> None:
    llm = _FakeLLM(
        [
            '{"broad":"thing under $50","medium":"a widget","high":"acme widget"}',
            '{"broad":"thing under $50","medium":"a widget","high":"acme widget"}',
        ]
    )
    result = await prep.generate_queries_for_product(
        _make_product(price=40),
        "medium",
        llm,  # type: ignore[arg-type]
    )
    assert result is None


async def test_generate_queries_returns_none_on_unparseable_output() -> None:
    llm = _FakeLLM(["nope", "still nope"])
    result = await prep.generate_queries_for_product(
        _make_product(price=40),
        "medium",
        llm,  # type: ignore[arg-type]
    )
    assert result is None


# -- build_dataset ----------------------------------------------------------


def _make_build_dataset(
    monkeypatch: pytest.MonkeyPatch,
    products: list[Product] | list[list[Product]],
    replies: list[str],
) -> _FakeLLM:
    """Patch streaming + LLM adapter for build_dataset() tests."""
    product_slices = list(products) if products and isinstance(products[0], list) else [products]
    calls = {"n": 0}

    def fake_stream(slug: str, offset: int, n: int, seed: int) -> list[Product]:
        calls["n"] += 1
        idx = min(calls["n"] - 1, len(product_slices) - 1)
        return list(product_slices[idx])[:n]

    monkeypatch.setattr(prep, "_stream_products", fake_stream)
    llm = _FakeLLM(replies)
    monkeypatch.setattr(prep, "OpenRouterAdapter", lambda **kw: llm)
    return llm


def _seeded_products(n: int, seed: int = 42) -> list[Product]:
    """Build in-memory products with deterministic prices matching prep._assign_price."""
    out: list[Product] = []
    for i in range(n):
        docid = f"p{i}"
        out.append(
            {
                "docid": docid,
                "title": f"Product {i}",
                "text": "desc",
                "bullet_points": ["f"],
                "brand": "ACME",
                "color": "black",
                "price_usd": prep._assign_price(docid, seed),
            }
        )
    return out


async def test_build_dataset_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    products = _seeded_products(3, seed=42)

    # Reply tailored to each product's assigned band + price.
    class _SmartLLM:
        def __init__(self, products: list[Product]) -> None:
            self._by_title = {p["title"]: p for p in products}

        async def complete_text(self, prompt: str) -> str:
            for title, prod in self._by_title.items():
                if title in prompt:
                    band = prep._pick_price_band(prod["docid"], 42)
                    ceiling = prod["price_usd"] + 50
                    queries = {
                        "broad": "a generic thing",
                        "medium": "a widget",
                        "high": "acme widget",
                    }
                    queries[band] = f"{queries[band]} under ${ceiling}"
                    return json.dumps(queries)
            raise AssertionError(f"unknown product in prompt: {prompt[:200]}")

    def fake_stream(slug: str, offset: int, n: int, seed: int) -> list[Product]:
        return list(products)[:n]

    monkeypatch.setattr(prep, "_stream_products", fake_stream)
    monkeypatch.setattr(prep, "OpenRouterAdapter", lambda **kw: _SmartLLM(products))

    dataset = await prep.build_dataset(num_samples=3, seed=42, threads=2, model="gemma-free")
    assert [p["docid"] for p in dataset["products"]] == ["p0", "p1", "p2"]
    assert [q["docid"] for q in dataset["queries"]] == ["p0", "p1", "p2"]
    assert dataset["seed"] == 42
    assert dataset["model"] == "gemma-free"
    for query, product in zip(dataset["queries"], dataset["products"], strict=True):
        expected_band = prep._pick_price_band(product["docid"], 42)
        assert query["price_band"] == expected_band
        assert "under $" in query[expected_band]


async def test_build_dataset_drops_products_on_persistent_rule_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    products = _seeded_products(2, seed=42)

    class _KeyedLLM:
        async def complete_text(self, prompt: str) -> str:
            # p0 always violates (no ceiling phrase); p1 returns a valid query
            # for its assigned band + a ceiling above its actual price.
            if "Product 0" in prompt:
                return '{"broad":"a thing","medium":"a widget","high":"acme widget"}'
            for p in products[1:2]:
                band = prep._pick_price_band(p["docid"], 42)
                ceiling = p["price_usd"] + 50
                base = {"broad": "a thing", "medium": "a widget", "high": "acme widget"}
                base[band] = f"{base[band]} under ${ceiling}"
                return json.dumps(base)
            raise AssertionError("unreachable")

    def fake_stream(slug: str, offset: int, n: int, seed: int) -> list[Product]:
        return list(products)[:n]

    monkeypatch.setattr(prep, "_stream_products", fake_stream)
    monkeypatch.setattr(prep, "OpenRouterAdapter", lambda **kw: _KeyedLLM())

    dataset = await prep.build_dataset(num_samples=2, seed=42, threads=1, model="m")
    assert [p["docid"] for p in dataset["products"]] == ["p1"]


async def test_build_dataset_halves_offset_on_short_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    products_third: list[Product] = [
        {
            "docid": "p0",
            "title": "P0",
            "text": "",
            "bullet_points": [],
            "brand": "b",
            "color": "c",
            "price_usd": prep._assign_price("p0", 42),
        },
    ]
    band = prep._pick_price_band("p0", 42)
    ceiling = products_third[0]["price_usd"] + 20
    base = {"broad": "a", "medium": "b", "high": "c"}
    base[band] = f"{base[band]} under ${ceiling}"
    replies = [json.dumps(base)]
    _make_build_dataset(monkeypatch, [[], [], products_third], replies)  # type: ignore[list-item]

    dataset = await prep.build_dataset(num_samples=1, seed=42, threads=1, model="m")
    assert len(dataset["products"]) == 1
    assert dataset["start_offset"] >= 0


async def test_build_dataset_falls_back_to_zero_offset_when_all_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product: Product = {
        "docid": "p0",
        "title": "P0",
        "text": "",
        "bullet_points": [],
        "brand": "b",
        "color": "c",
        "price_usd": prep._assign_price("p0", 0),
    }
    calls: list[int] = []

    def fake_stream(slug: str, offset: int, n: int, seed: int) -> list[Product]:
        calls.append(offset)
        if offset == 0:
            return [product]
        return []

    monkeypatch.setattr(prep, "_stream_products", fake_stream)

    band = prep._pick_price_band("p0", 0)
    ceiling = product["price_usd"] + 20
    base = {"broad": "a", "medium": "b", "high": "c"}
    base[band] = f"{base[band]} under ${ceiling}"

    class _LLM:
        async def complete_text(self, prompt: str) -> str:
            return json.dumps(base)

    monkeypatch.setattr(prep, "OpenRouterAdapter", lambda **kw: _LLM())
    ds = await prep.build_dataset(num_samples=1, seed=0, threads=1, model="m")
    assert ds["start_offset"] == 0
    assert len(ds["products"]) == 1


def test_stream_products_maps_hf_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """_stream_products should call the HF loader and filter unusable rows."""

    class _FakeStream:
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self._rows = rows

        def skip(self, n: int) -> _FakeStream:
            return _FakeStream(self._rows[n:])

        def take(self, n: int) -> _FakeStream:
            return _FakeStream(self._rows[:n])

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(self._rows)

    rows = [
        {"docid": "a", "title": "A", "locale": "us"},
        {"docid": "b", "title": "B", "locale": "uk"},  # filtered
        {"docid": "c", "title": "C", "locale": "us"},
    ]

    import sys as _sys
    import types

    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = lambda slug, split, streaming: _FakeStream(rows)  # type: ignore[attr-defined]
    monkeypatch.setitem(_sys.modules, "datasets", fake_module)

    products = prep._stream_products("slug", 0, 3, seed=42)
    assert [p["docid"] for p in products] == ["a", "c"]
    # Streamed products must carry deterministic prices.
    assert products[0]["price_usd"] == prep._assign_price("a", 42)


async def test_build_dataset_respects_thread_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    products = _seeded_products(6, seed=42)

    def fake_stream(slug: str, offset: int, n: int, seed: int) -> list[Product]:
        return list(products)[:n]

    monkeypatch.setattr(prep, "_stream_products", fake_stream)

    max_seen = {"n": 0}
    running = {"n": 0}

    class _SpyLLM:
        async def complete_text(self, prompt: str) -> str:
            running["n"] += 1
            max_seen["n"] = max(max_seen["n"], running["n"])
            await asyncio.sleep(0)  # yield so concurrency shows
            running["n"] -= 1
            for p in products:
                if p["title"] in prompt:
                    band = prep._pick_price_band(p["docid"], 42)
                    ceiling = p["price_usd"] + 50
                    base = {"broad": "a", "medium": "b", "high": "c"}
                    base[band] = f"{base[band]} under ${ceiling}"
                    return json.dumps(base)
            raise AssertionError(f"unknown product: {prompt[:200]}")

    monkeypatch.setattr(prep, "OpenRouterAdapter", lambda **kw: _SpyLLM())
    await prep.build_dataset(num_samples=6, seed=42, threads=2, model="m")
    assert max_seen["n"] <= 2


# -- save/load dataset + main ------------------------------------------------


def _full_dataset() -> prep.Dataset:
    return {
        "products": [
            {
                "docid": "p0",
                "title": "T",
                "text": "",
                "bullet_points": [],
                "brand": "b",
                "color": "c",
                "price_usd": 42,
            }
        ],
        "queries": [
            {
                "docid": "p0",
                "broad": "a",
                "medium": "b under $60",
                "high": "c",
                "price_band": "medium",
            },
        ],
        "seed": 1,
        "num_samples": 1,
        "start_offset": 0,
        "model": "m",
    }


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    dataset = _full_dataset()
    target = tmp_path / "ds.json"
    prep.save_dataset(dataset, target)
    assert prep.load_dataset(target) == dataset


def test_main_cli_writes_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def fake_build(**kw: Any) -> prep.Dataset:
        return _full_dataset() | {"seed": kw["seed"], "model": kw["model"]}  # type: ignore[return-value]

    monkeypatch.setattr(prep, "build_dataset", fake_build)
    out = tmp_path / "ds.json"
    rc = prep.main(
        [
            "--num-samples=1",
            "--seed=7",
            "--threads=1",
            f"--output={out}",
            "--llm=google/gemma-4-26b-a4b-it",
        ]
    )
    assert rc == 0
    loaded = prep.load_dataset(out)
    assert loaded["seed"] == 7
    assert loaded["model"] == "google/gemma-4-26b-a4b-it"
