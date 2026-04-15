# Schema authoring

Schemas are the user-facing API. Ennoia ships no domain-specific defaults;
you declare what to extract, and Ennoia runs it.

## BaseStructure

A `BaseStructure` is a Pydantic `BaseModel` subclass whose fields describe
structured metadata. The class **docstring is the extraction prompt** —
the LLM reads it verbatim alongside the JSON Schema.

```python
from datetime import date
from typing import Literal, Optional

from ennoia import BaseStructure


class CaseDocument(BaseStructure):
    """Extract case metadata from the document."""

    jurisdiction: Literal["WA", "NY", "TX"]
    date_decided: date
    is_overruled: bool
    tags: list[str]
    overruled_by: Optional[str] = None
```

Field types drive both validation and **filter-operator inference** — see
[filters.md](filters.md) for the mapping. No additional declarations
required for the common case.

## BaseSemantic

A semantic schema is a marker class whose docstring is the question the
LLM should answer about a document. The answer text is embedded and
stored for vector search.

```python
from ennoia import BaseSemantic


class Holding(BaseSemantic):
    """What is the core legal holding of this case?"""
```

## extend()

`BaseStructure.extend()` runs after the instance is extracted and returns
further schemas (structural or semantic) to apply to the same document.
The populated parent instance is available through `self` — including its
self-reported `_confidence` — so conditional logic is just Python.

```python
class CaseDocument(BaseStructure):
    """Extract case metadata."""

    jurisdiction: Literal["WA", "NY", "TX"]

    def extend(self):
        if self.jurisdiction == "WA" and getattr(self, "_confidence", 0.0) >= 0.8:
            return [WashingtonAppellateSchema]
        return []
```

Children run in the next execution layer; siblings within a layer run in
parallel. The parent's extracted values are injected as `Additional
Context` in the child's prompt, so the LLM can reason about the parent
output when filling in the child.

## Field overrides

Operators for a specific field can be restricted or the field itself
excluded from filtering using `Field(...)`:

```python
from typing import Annotated

from ennoia import BaseStructure, Field


class DocMeta(BaseStructure):
    """Extract doc metadata."""

    title: Annotated[str, Field(description="Doc title", operators=["eq", "contains"])]
    notes: Annotated[str, Field(description="Internal", filterable=False)]
```

`operators=[...]` replaces the inferred operator list; `filterable=False`
omits the field from both the discovery payload and the filter validator
(so a query referencing it is rejected as *unknown field*).

## Confidence

The extractor dynamically appends a `_confidence` property to the JSON
Schema the LLM sees — it is **not** a declared field on `BaseStructure`.
This guarantees the model emits a self-reported score *after* filling in
the real fields (evaluation, not guessing). The confidence is surfaced in
two places:

- `IndexResult.confidences`: `{schema_name: float}` — the summary exposed
  to pipeline callers.
- `instance._confidence`: set on the validated model via
  `ConfigDict(extra="allow")` so `extend()` can branch on it.

The confidence is **stripped from the structured-store payload** — it is
observability, not data. If the model omits `_confidence`, it defaults to
1.0 with a warning.

## RejectException

Raise `RejectException` from any schema's validator or `extend()` to drop
the entire document. The pipeline catches it, returns
`IndexResult(rejected=True)`, and writes nothing to the stores.

```python
from ennoia import BaseStructure, RejectException


class Gatekeeper(BaseStructure):
    """Confirm the document is in scope."""

    looks_like_case_law: bool

    def extend(self):
        if not self.looks_like_case_law:
            raise RejectException("Out of scope.")
        return []
```

## Discovery

`ennoia.describe(schemas)` returns the canonical filter contract for a
schema set — useful for agents that need to build valid filters
autonomously. See [filters.md §Schema Discovery](filters.md).
