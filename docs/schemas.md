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

## Extension manifest

Every class `extend()` may return must be declared up front in
`Schema.extensions` on the emitting schema. This lets Ennoia compute the
full field space (the *superschema*) at pipeline initialization, validate
schema composition before any document is indexed, and reject runtime
emissions that were not declared.

```python
class CaseDocument(BaseStructure):
    """Extract case metadata."""

    jurisdiction: Literal["WA", "NY", "TX"]

    class Schema:
        extensions = [WashingtonDetails, NewYorkDetails, TexasDetails]

    def extend(self):
        if self.jurisdiction == "WA":
            return [WashingtonDetails]
        if self.jurisdiction == "NY":
            return [NewYorkDetails]
        return [TexasDetails]
```

`Schema` is a plain inner class — no base to extend, no decorator. Both
attributes are optional:

- `extensions: list[type]` — default `[]`. Must list every class
  `extend()` may return, structural or semantic. Undeclared returns raise
  `SchemaError` at index time.
- `namespace: str | None` — default `None` (flat merge). See below.

Multi-level manifests are fine: an extension may itself declare
`extensions`, producing a tree the framework traverses transitively.

## Field merging

When the manifest is resolved, every reachable structural schema
contributes its fields to the superschema. There are three cases:

**Flat merge (default).** Fields from every contributing schema appear
at the top level. This is the common case for contextual extensions —
children add fields that semantically belong to the parent document.

**Multi-source flat fields.** When the same field name appears on more
than one schema with compatible types, it merges into a single superschema
entry whose `sources` list records every contributing class. Used when
multiple extraction strategies target the same underlying concept (e.g.,
citation formats that vary by case era).

```python
class OldCaseFormat(BaseStructure):
    """Historic case format."""
    citation: Literal["federal", "state"] = "federal"

class NewCaseFormat(BaseStructure):
    """Modern case format."""
    citation: Literal["state", "international"] = "state"
```

The merged `citation` field becomes `Literal["federal", "state", "international"]`.

**Namespaced merge.** When `Schema.namespace = "ns"`, the schema's own
fields are prefixed with `ns__` in the superschema. Namespaces are an
escape hatch — use them when fields would otherwise collide, when
provenance matters, or when multiple instances of the same schema might
be present.

```python
class WashingtonDetails(BaseStructure):
    """Washington-specific fields."""
    court_type: Literal["appellate", "supreme", "district"]

    class Schema:
        namespace = "wa"
```

In the superschema, this becomes `wa__court_type`. Namespaces apply only
to their declaring class's own fields — they do **not** propagate into
descendants reachable via that schema's `extensions`.

The delimiter `__` is reserved and may not appear inside a namespace; a
namespace must also be a valid Python identifier and must not match any
reserved filter-operator name (`eq`, `in`, `gt`, …). Field names
themselves must not match reserved operators either — Ennoia rejects
such schemas at pipeline init with a clear `SchemaError`.

## Type compatibility

When a field is declared by more than one schema under flat merge, its
types are merged per these rules:

- Identical types merge to themselves.
- `Literal[...]` ∪ `Literal[...]` merges to the union of values.
- `Optional[T]` + `T` merges to `Optional[T]`.
- `list[T]` + `list[U]` merges to `list[merge(T, U)]` recursively.

Any other pairing is incompatible and raises `SchemaError` at pipeline
init, identifying both source classes and both types. When descriptions
differ across sources, Ennoia uses the first-declared description and
emits a non-fatal `SchemaWarning`; the `has_divergent_descriptions` flag
on the discovery payload alerts downstream agents.

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
