"""Prep: stream a slice of the ESCI product corpus and generate 3-band queries.

The ESCI dataset (``spacemanidol/ESCI-product-dataset-corpus-us``) ships
~1.21M US product listings. We draw a deterministic seeded slice, assign
each product a synthesized price (ESCI has no price field of its own),
then ask a free OpenRouter model (Gemma by default) to generate three
shopper-style search phrases per product:

- ``broad``:  no brand, no product name — generic attribute only
- ``medium``: product name or one specific attribute, still no brand
- ``high``:   brand + product name or feature

Exactly ONE of the three bands (randomly chosen per product, seeded on
the docid) includes a price ceiling of the form ``under $X`` where X is
strictly greater than the product's actual price — this produces a
realistic search pattern the retrievers can filter on. The other two
bands are post-checked to confirm they do NOT mention price.

CLI:

  python -m benchmark.data.prep \\
      --num-samples=1000 --seed=42 --threads=16 \\
      --output=benchmark/data/dataset.json \\
      --llm=google/gemma-4-26b-a4b-it
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from tqdm.asyncio import tqdm_asyncio

from benchmark.config import (
    DATASET_PATH,
    DEFAULT_NUM_SAMPLES,
    HF_ESCI_DATASET,
    MAX_START_OFFSET,
    MODEL_LLM,
    SEED,
    THREADS,
)
from benchmark.pipelines._retry import async_with_retry
from ennoia.adapters.llm.openrouter import OpenRouterAdapter
from ennoia.prompts import load_prompt

__all__ = [
    "Dataset",
    "PriceBand",
    "Product",
    "QueryCase",
    "build_dataset",
    "generate_queries_for_product",
    "load_dataset",
    "main",
    "save_dataset",
]


PriceBand = Literal["broad", "medium", "high"]
_PRICE_BANDS: tuple[PriceBand, ...] = ("broad", "medium", "high")

# Synthesized product prices: log-uniform in $10 – $300, whole-dollar
# integers. Log-uniform spreads the price distribution more realistically
# than plain uniform (more cheap items, fewer expensive ones), matching
# how e-commerce catalogues skew.
_PRICE_MIN_USD = 10
_PRICE_MAX_USD = 300


class Product(TypedDict):
    docid: str
    title: str
    text: str
    bullet_points: list[str]
    brand: str
    color: str
    price_usd: int


class QueryCase(TypedDict):
    docid: str
    broad: str
    medium: str
    high: str
    price_band: PriceBand


class Dataset(TypedDict):
    products: list[Product]
    queries: list[QueryCase]
    seed: int
    num_samples: int
    start_offset: int
    model: str


# Any of these substrings in a generated query is considered a price mention.
_PRICE_PATTERNS = re.compile(
    r"(\$|usd|\bprice\b|\bcheap(er|est)?\b|\bexpensive\b|\bunder\s*\d+\b|\bover\s*\d+\b|\bless\s+than\s+\d+\b)",
    re.IGNORECASE,
)

# Upper-bound ceiling phrases — the only legal shape in the price band.
_PRICE_CEILING_RE = re.compile(
    r"(?:under|below|less\s+than|up\s+to)\s*\$?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _assign_price(docid: str, seed: int) -> int:
    """Deterministic per-product price, log-uniform in [$10, $300]."""
    rng = random.Random(f"price::{seed}::{docid}")
    lo, hi = math.log(_PRICE_MIN_USD), math.log(_PRICE_MAX_USD)
    return int(math.exp(rng.uniform(lo, hi)))


def _pick_price_band(docid: str, seed: int) -> PriceBand:
    """Deterministic per-product band pick — which query gets the price constraint."""
    rng = random.Random(f"band::{seed}::{docid}")
    return rng.choice(_PRICE_BANDS)


def _normalise_bullets(raw: Any) -> list[str]:
    """Tolerant parse of ESCI ``bullet_points`` into a clean list.

    ESCI records ship this field as a JSON-encoded string, a plain newline
    string, or occasionally ``None``. We try ``json.loads`` first, fall
    back to newline-split, and always return a list of trimmed non-empty
    strings.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    text = str(raw).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [line.strip() for line in text.split("\n") if line.strip()]


def _record_to_product(record: dict[str, Any], seed: int) -> Product | None:
    """Return a ``Product`` from one ESCI row, or ``None`` if unusable.

    Filters out non-US locales defensively, drops rows missing a
    docid/title, and assigns a deterministic synthesized price keyed on
    the docid + run seed.
    """
    if str(record.get("locale", "us")).lower() != "us":
        return None
    docid = str(record.get("docid", "")).strip()
    title = str(record.get("title", "")).strip()
    if not docid or not title:
        return None
    return {
        "docid": docid,
        "title": title,
        "text": str(record.get("text", "") or "").strip(),
        "bullet_points": _normalise_bullets(record.get("bullet_points")),
        "brand": str(record.get("brand", "") or "").strip() or "unknown",
        "color": str(record.get("color", "") or "").strip() or "unknown",
        "price_usd": _assign_price(docid, seed),
    }


def _stream_products(
    slug: str,
    start_offset: int,
    num_samples: int,
    seed: int,
) -> list[Product]:
    """Stream ``num_samples`` ``Product``s starting at ``start_offset``.

    If the slice comes back short (offset too high for the dataset), the
    caller retries with a halved offset — handled by ``build_dataset``.
    """
    from datasets import load_dataset as _hf_load  # type: ignore[import-untyped]

    stream = _hf_load(slug, split="train", streaming=True)
    stream = stream.skip(start_offset).take(num_samples)
    products: list[Product] = []
    for raw in stream:
        record = cast(dict[str, Any], raw)
        product = _record_to_product(record, seed)
        if product is not None:
            products.append(product)
    return products


def _contains_price(text: str) -> bool:
    return _PRICE_PATTERNS.search(text) is not None


def _extract_price_ceiling(text: str) -> float | None:
    """Return the numeric ceiling from an ``under $X`` / ``below $X`` phrase.

    Returns ``None`` if no ceiling phrase is present (malformed query). The
    regex only captures ``\\d+(\\.\\d+)?`` so ``float()`` on the match is
    always well-defined.
    """
    m = _PRICE_CEILING_RE.search(text)
    if m is None:
        return None
    return float(m.group(1))


def _price_rules_ok(parsed: dict[str, str], price_band: PriceBand, actual_price: int) -> bool:
    """Verify the target band has a usable price ceiling, the others don't.

    - Target band: must contain a price mention AND a ceiling strictly
      greater than ``actual_price`` (so filtering with ``price__lt=ceiling``
      would include this product).
    - Other two bands: must NOT mention price at all.
    """
    for band in _PRICE_BANDS:
        has_price = _contains_price(parsed[band])
        if band == price_band:
            if not has_price:
                return False
            ceiling = _extract_price_ceiling(parsed[band])
            if ceiling is None or ceiling <= actual_price:
                return False
        else:
            if has_price:
                return False
    return True


def _parse_query_json(raw: str) -> dict[str, str] | None:
    """Extract the ``{"broad","medium","high"}`` object from an LLM reply.

    Free models are chatty — they sometimes emit fenced blocks or prose.
    The regex below finds the first ``{...}`` block regardless of fences,
    so we don't need to strip them explicitly.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    out: dict[str, str] = {}
    for key in ("broad", "medium", "high"):
        value = data.get(key) if isinstance(data, dict) else None
        if not isinstance(value, str):
            return None
        out[key] = value.strip().strip('"').strip("'")
    return out


def _render_query_prompt(product: Product, price_band: PriceBand) -> str:
    payload = {
        "title": product["title"],
        "brand": product["brand"],
        "color": product["color"],
        "price_usd": product["price_usd"],
        "bullet_points": product["bullet_points"][:6],
        "description": product["text"][:600],
    }
    return load_prompt("benchmark_queries").format(
        product_json=json.dumps(payload, ensure_ascii=False, indent=2),
        price_band=price_band,
        price_usd=product["price_usd"],
    )


async def generate_queries_for_product(
    product: Product,
    price_band: PriceBand,
    llm: OpenRouterAdapter,
) -> QueryCase | None:
    """Generate one ``QueryCase`` or return ``None`` after two failed attempts.

    Retries transient network errors inside :func:`async_with_retry`, then
    re-prompts once if the LLM violates the per-band price rules (price
    leaks into the wrong band, or the target band's ceiling is ≤ the
    product's actual price).
    """
    prompt = _render_query_prompt(product, price_band)
    for _attempt in range(2):
        raw = await async_with_retry(lambda: llm.complete_text(prompt))
        parsed = _parse_query_json(raw)
        if parsed is None:
            continue
        if not _price_rules_ok(parsed, price_band, product["price_usd"]):
            continue
        return {
            "docid": product["docid"],
            "broad": parsed["broad"],
            "medium": parsed["medium"],
            "high": parsed["high"],
            "price_band": price_band,
        }
    return None


async def build_dataset(
    num_samples: int,
    seed: int,
    threads: int,
    model: str,
    slug: str = HF_ESCI_DATASET,
) -> Dataset:
    """Download, slice, assign prices, and generate queries for ``num_samples`` products."""
    rng = random.Random(seed)
    start_offset = rng.randrange(0, MAX_START_OFFSET)
    offset = start_offset
    products = _stream_products(slug, offset, num_samples, seed)
    # If the slice comes back empty (offset past end of dataset), halve
    # until we land on a populated region. The final iteration halves to
    # offset=0 which is the guaranteed-populated fallback.
    while not products and offset > 0:
        offset = offset // 2
        products = _stream_products(slug, offset, num_samples, seed)
        start_offset = offset

    llm = OpenRouterAdapter(model=model)
    sem = asyncio.Semaphore(max(1, threads))
    dropped: list[str] = []

    async def one(product: Product) -> QueryCase | None:
        band = _pick_price_band(product["docid"], seed)
        async with sem:
            try:
                return await generate_queries_for_product(product, band, llm)
            except Exception as exc:  # pragma: no cover - network noise path
                print(
                    f"[prep] {product['docid']} query-gen failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return None

    results = await tqdm_asyncio.gather(
        *(one(p) for p in products),
        desc="prep queries",
        unit="product",
    )
    queries: list[QueryCase] = []
    kept_products: list[Product] = []
    for product, result in zip(products, results, strict=True):
        if result is None:
            dropped.append(product["docid"])
            continue
        queries.append(result)
        kept_products.append(product)
    if dropped:
        print(
            f"[prep] dropped {len(dropped)} products after price-rule or parse failure: "
            f"{dropped[:5]}{'...' if len(dropped) > 5 else ''}",
            file=sys.stderr,
        )

    return {
        "products": kept_products,
        "queries": queries,
        "seed": seed,
        "num_samples": num_samples,
        "start_offset": start_offset,
        "model": model,
    }


def save_dataset(dataset: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")


def load_dataset(path: Path) -> Dataset:
    return cast(Dataset, json.loads(path.read_text(encoding="utf-8")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmark.data.prep")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--output", type=Path, default=DATASET_PATH)
    parser.add_argument("--llm", type=str, default=MODEL_LLM)
    args = parser.parse_args(argv)

    dataset = asyncio.run(
        build_dataset(
            num_samples=args.num_samples,
            seed=args.seed,
            threads=args.threads,
            model=args.llm,
        )
    )
    save_dataset(dataset, args.output)
    print(
        f"[prep] wrote {args.output} "
        f"(products={len(dataset['products'])}, queries={len(dataset['queries'])}, "
        f"start_offset={dataset['start_offset']})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
