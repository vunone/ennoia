"""DDI schemas tailored to e-commerce product listings (ESCI benchmark).

- :class:`ProductMeta` (structural) — filterable catalogue metadata
  (brand, color, category, price). High-precision queries reach the
  right listing via structural filters on brand + category; price
  ceilings (``price_usd__lt=50``) drive the price-band queries generated
  by the prep script.
- :class:`ProductSummary` (semantic) — one-sentence gist of the listing.
  Broad queries ("a thing to cool my laptop") fall back to this index
  when no structural field narrows the candidate set.
"""

from __future__ import annotations

from typing import Annotated

from ennoia import BaseSemantic, BaseStructure, Field


class ProductMeta(BaseStructure):
    """Extract filterable product-catalogue metadata from an e-commerce listing.

    Read the listing (title, description, bullets, price) and identify:
    - The brand or manufacturer as it appears on the listing.
    - The primary colour as a single lowercase word (e.g. ``black``,
      ``navy``). Use ``unknown`` when colour is not stated.
    - A high-level category in lowercase snake-case
      (e.g. ``electronics``, ``home_garden``, ``apparel``, ``toys``).
    - The price in US dollars, read directly from the ``Price: $X`` line
      in the listing, as a whole-dollar integer.

    Use ``unknown`` for any text field that is not explicitly stated.
    """

    brand: Annotated[str, Field(description="Brand or manufacturer, as written on the listing.")]
    color: Annotated[str, Field(description="Primary colour, single lowercase word, or 'unknown'.")]
    category: Annotated[
        str, Field(description="High-level category in lowercase snake-case (e.g. 'electronics').")
    ]
    price_usd: Annotated[
        int,
        Field(
            description=(
                "Price in US dollars as a whole-dollar integer, read from the "
                "'Price: $X' line in the listing."
            )
        ),
    ]


class ProductName(BaseSemantic):
    """Extract the product name, brand and base characteristics in a single product tagline.

    Example: A pink sillicon spoon of brand HandyMan"""


class ProductSummary(BaseSemantic):
    """Summarize what this product is and who it is for.
    Capture the product's purpose and intended user in plain English.
    Preserve concrete specifics (materials, size, power source) when the
    listing states them, but do not quote the brand or product name.
    """
