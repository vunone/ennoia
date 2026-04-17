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

## BaseCollection

A `BaseCollection` extracts a **list** of structured entities the LLM
finds in a document — parties to a contract, citations in a brief, every
PII mention. Where `BaseSemantic` yields one free-text answer per document,
`BaseCollection` yields N entities, each shaped by its own Pydantic fields
and rendered to text via `template()` for embedding.

```python
from typing import Annotated
from ennoia import BaseCollection, Field


class ContractParty(BaseCollection):
    """Extract every party mentioned in the contract."""

    company_name: str
    context: Annotated[str, Field(description="Short context this company is mentioned in")]
    participation_year: int

    def template(self) -> str:
        return f"{self.company_name} ({self.participation_year}): {self.context}"
```

### How extraction works

The pipeline asks the LLM for a wrapper object of the shape

```json
{
  "entities_list": [ <entity>, <entity>, ... ],
  "is_done": true
}
```

and re-runs the call with a `<PreviouslyExtracted>` block listing what it
already captured, until one of:

- `is_done == true`,
- `entities_list` is empty,
- no new unique valid entity was added this iteration, or
- `Schema.max_iterations` was reached (default: `None` — unbounded).

Each extracted item is validated by Pydantic **and** by the subclass's
`is_valid()`. Invalid items are dropped silently; the loop keeps going.

### Customisable hooks

| Method | Default | Why override |
|---|---|---|
| `extract_prompt()` | the docstring | almost never; the docstring *is* the prompt |
| `template()` | `str(self.model_dump(mode="json"))` | produce a tight natural-language rendering for better retrieval |
| `get_unique()` | a random token per call | return a deterministic key (e.g. a tuple-hash of selected fields) to dedup identical extractions |
| `is_valid()` | no-op | raise `SkipItem` to drop just this entity; raise `RejectException` to drop the whole document |
| `extend()` | `[]` | emit child schemas (structural, semantic, or collection) for downstream extraction — called once per extracted entity |

### Schema config

`Schema` is a plain inner class, same pattern as `BaseStructure`. All
attributes are optional:

- `extensions: list[type]` — classes `extend()` may return; undeclared
  returns raise `SchemaError` at index time.
- `max_iterations: int | None` — hard cap on iteration count.
  `None` (default) trusts the other termination conditions.

```python
class ContractParty(BaseCollection):
    """Extract every party."""

    company_name: str
    context: str

    class Schema:
        extensions = [FaangScorecard]
        max_iterations = 20

    def extend(self):
        if self.company_name.lower() in {"meta", "google", "apple"}:
            return [FaangScorecard]
        return []
```

### Persistence

Each extracted entity becomes one row in the vector store — keyed by
`(source_id, schema_name, unique)` — with its `template()` text embedded
as the vector. From storage's perspective a `BaseCollection` with N
entities behaves identically to N `BaseSemantic` answers under the same
index name. Structural fields of the entity are **not** persisted as
tabular columns; they exist only to shape the prompt and the template
output.

`IndexResult.collections` exposes the typed instances. Per-entity
confidences live in `IndexResult.collection_confidences`; the top-level
`IndexResult.confidences[schema_name]` is the mean so one flat map
continues to work for existing consumers.

### SkipItem vs. RejectException

- `SkipItem` (raised from `is_valid()`) drops just the offending entity;
  the collection continues and the rest of the pipeline runs.
- `RejectException` (raised anywhere — `is_valid`, `extend`, etc.) drops
  the **whole document**: `IndexResult(rejected=True)`, nothing persisted.

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
