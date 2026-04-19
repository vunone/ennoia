# Filter language

Ennoia provides a unified filter language across every surface — Python
SDK, CLI, MCP, REST. The language is predictable for developers and
unambiguous for AI agents.

## Syntax

A filter is a key-value pair where the key is a field name, optionally
followed by an operator suffix separated by double underscore. Missing
suffix means equality.

```
field_name                → equals
field_name__operator      → applies operator
```

Filters are combined with logical AND — every condition must hold.

## Operators

| Operator | Meaning | Example |
|---|---|---|
| `eq` | Equals (default) | `jurisdiction=WA` |
| `gt` | Greater than | `date_decided__gt=2020-01-01` |
| `gte` | Greater than or equal | `date_decided__gte=2020-01-01` |
| `lt` | Less than | `date_decided__lt=2023-06-01` |
| `lte` | Less than or equal | `date_decided__lte=2023-06-01` |
| `in` | Matches any value in a list | `jurisdiction__in=WA,NY,TX` |
| `contains` | String contains substring / list contains element | `title__contains=bankruptcy` |
| `startswith` | String starts with prefix | `title__startswith=In re` |
| `contains_all` | List contains all elements | `tags__contains_all=tort,negligence` |
| `contains_any` | List contains any element | `tags__contains_any=tort,contract` |
| `is_null` | Field is null (`true`) or not null (`false`) | `overruled_by__is_null=true` |

## Operator inference

Applying `gt` to an enum or `contains` to a date is meaningless, so
Ennoia infers valid operators from Pydantic types.

| Field Type | Supported Operators |
|---|---|
| `bool` | `eq` |
| `str` | `eq`, `contains`, `startswith` |
| `int`, `float` | `eq`, `gt`, `gte`, `lt`, `lte` |
| `date`, `datetime` | `eq`, `gt`, `gte`, `lt`, `lte` |
| `Literal[...]` | `eq`, `in` |
| `list[T]` | `contains`, `contains_all`, `contains_any` |
| `Optional[T]` / `T \| None` | operators of `T` plus `is_null` |

For example:

```python
class CaseDocument(BaseStructure):
    """Extract case metadata."""
    jurisdiction: Literal["WA", "NY", "TX"]
    date_decided: date
    tags: list[str]
    overruled_by: Optional[str] = None
```

yields:

```
jurisdiction  → eq, in
date_decided  → eq, gt, gte, lt, lte
tags          → contains, contains_all, contains_any
overruled_by  → eq, contains, startswith, is_null
```

## Overrides

Narrow the operator set or exclude a field entirely from filtering using
`ennoia.Field`:

```python
from typing import Annotated
from ennoia import BaseStructure, Field

class DocMeta(BaseStructure):
    """Extract doc metadata."""
    title: Annotated[str, Field(description="Title", operators=["eq", "contains"])]
    notes: Annotated[str, Field(description="Internal", filterable=False)]
```

## Schema discovery

`ennoia.describe(schemas)` returns the canonical filter contract — the
same payload an MCP server exposes over `discover_schema()`. The shape
reflects the **superschema**: fields from every class reachable via
`Schema.extensions` (see [schemas.md](schemas.md#extension-manifest))
are merged into a unified field space. Namespaced schemas contribute
fields under the `{namespace}__{field}` convention.

```json
{
  "structural_fields": [
    {"name": "jurisdiction", "type": "enum", "options": ["WA", "NY", "TX"],
     "operators": ["eq", "in"], "sources": ["CaseDocument"], "description": null},
    {"name": "citation", "type": "enum",
     "options": ["federal", "state", "international"],
     "operators": ["eq", "in"],
     "sources": ["OldCaseFormat", "NewCaseFormat"],
     "description": "Case citation.",
     "has_divergent_descriptions": false},
    {"name": "wa__court_type", "type": "enum",
     "options": ["appellate", "supreme", "district"],
     "operators": ["eq", "in"],
     "sources": ["WashingtonDetails"],
     "description": "Washington state court type."}
  ],
  "semantic_indices": [
    {"name": "Holding", "description": "What is the core legal holding of this case?"}
  ]
}
```

`sources` lists every class contributing to that field (one entry for
single-source fields, multiple for merged). `has_divergent_descriptions`
is present only when a field has more than one source, set to `true`
when the contributing descriptions differ.

## Validation errors

`Pipeline.search()` validates every filter before touching the store.
Unknown fields, unsupported operators, and un-coercible values raise
`FilterValidationError` whose `.to_dict()` matches:

```json
{
  "error": "invalid_filter",
  "field": "jurisdiction",
  "operator": "gt",
  "message": "Field 'jurisdiction' (type: enum) does not support operator 'gt'. Supported operators: eq, in."
}
```

### Nested filter values are rejected

The flat `{field__op: value}` form is canonical — the nested
`{field: {op: value}}` form (Elasticsearch/MongoDB style) is not
accepted, because silently swallowing it would collapse to a false
equality and match zero rows. Mapping values trigger a
`FilterValidationError` that tells the caller (or agent) how to rewrite
the filter:

```json
{
  "error": "invalid_filter",
  "field": "date_decided",
  "operator": "eq",
  "message": "Filter value for 'date_decided' must be a scalar, not a mapping. Use the flat convention {'date_decided__<operator>': <value>}, e.g. {'date_decided__eq': <value>}."
}
```

## Interface consistency

The same filter expressed across surfaces:

```python
# Python
pipeline.search(query="...", filters={
    "jurisdiction__in": ["WA", "NY"],
    "date_decided__gte": date(2020, 1, 1),
    "is_overruled": False,
})
```

```bash
# CLI
ennoia search "..." \
  --filter "jurisdiction__in=WA,NY" \
  --filter "date_decided__gte=2020-01-01" \
  --filter "is_overruled=false"
```

CLI string values are coerced against the schema types at evaluation
time. Comma-separated values in `__in`, `__contains_all`, and
`__contains_any` are split into lists automatically.
