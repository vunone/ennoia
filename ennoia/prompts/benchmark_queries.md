# Product-discovery query generator — benchmark prompt

You generate three shopper-style search phrases for one product in an
online-store catalogue. The phrases simulate what a real user would type
into a search box when looking for this item. Three difficulty bands:

- **`broad`**: Vague request. NO brand, NO product name. Describe only a
  rough use-case or generic attribute ("a thing to cool down my laptop",
  "something to hold plants on my balcony"). Intentionally ambiguous.
- **`medium`**: Moderate request. Mentions the product category / type OR
  one specific attribute ("external laptop cooler with usb power",
  "hanging rail planter for balcony"). NOT the brand.
- **`high`**: Precise request. Mentions the brand AND either the product
  name or a distinguishing feature ("ASUS ROG laptop stand with RGB",
  "Keter Easy Grow hanging balcony planter").

## Price rule (exactly one band has a price ceiling)

- **Target band:** `{price_band}` — this is the ONLY band that may
  mention price.
- **Product's actual price:** `${price_usd}`.
- In the `{price_band}` query, include a phrase of the form `under $X`
  where **X is a whole-dollar ceiling strictly greater than `${price_usd}`**.
  Pick a clean round number: the next multiple of 10 / 25 / 50 above the
  actual price. Examples:
  - Product at $40 → `under $50`.
  - Product at $72 → `under $100`.
  - Product at $210 → `under $250`.
- In the OTHER two bands, NEVER mention price, `$`, `USD`, `cheap`,
  `expensive`, `under N`, `over N`, or any numeric ceiling.

## Hard rules

- Output lowercase, no trailing punctuation.
- No leading/trailing whitespace.
- No quotes, no markdown.
- Do not copy the product title verbatim into the `broad` variant; it
  must be generic enough that a dozen unrelated products could match.

## Product

```
{product_json}
```

## Output contract

Reply with ONLY a JSON object in this exact shape — no prose, no fences:

{{"broad": "...", "medium": "...", "high": "..."}}
