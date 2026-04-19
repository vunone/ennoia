# Ennoia schema scaffolding — LLM guide

You draft a Python module with **exactly three** extraction schemas for the
user's document type. Keep it minimal — this is a *first draft* a human
will refine. Do not invent structure you cannot see in the sample.

## Output contract

- Reply with exactly one fenced Python code block using triple backticks
  and the `python` tag. No prose outside the block.
- Use single backticks inside any docstring; never triple.
- The module must import cleanly and define exactly three top-level
  classes, in this order: `Metadata`, `QuestionAnswer`, `Summary`.
- Allowed imports: `ennoia`, `typing`, `datetime`.

## The three classes

### 1. `Metadata(BaseStructure)` — filterable fields

Plain fields a user would filter on: titles, names, dates, categories
(as `str`), prices, quantities, boolean flags. The class docstring **is
the extraction prompt**, so write one line tailored to the document type
and the user's retrieval task.

**Generality test — the rule that overrides everything else.** Treat
the sample as one document out of a thousand of the same kind. A field
belongs in `Metadata` only if *every (or almost every) other document in
the corpus would also have a value for it* — just a different value.
Run this mental check for every candidate field:

> "If I picked ten more documents of the same kind, would this field
> make sense on most of them, with a different value?"

If the honest answer is "no, this field only fits *this* document", drop
it. These are facts *about* the single sample, not metadata that
partitions a corpus.

**Concrete examples of the generality test:**

| Corpus | Good `Metadata` fields | Bad `Metadata` fields |
|---|---|---|
| Product pages | `title`, `price_usd`, `category`, `color`, `size`, `in_stock`, `brand`, `rating` | `has_rgb_lighting` (only fits electronics), `best_by_date` (only fits food) |
| Legal contracts | `counterparty`, `governing_law`, `effective_date`, `contract_value`, `term_months`, `auto_renews` | `mentions_arbitration_clause` (too specific to one doc's topic) |
| News articles | `title`, `author`, `published`, `section`, `word_count`, `is_opinion` | `covers_ukraine_war` (topic flag — belongs in semantic search, not a filter) |
| API docs | `page_title`, `product_name`, `version`, `last_updated` | `default_port`, `default_transport`, `has_mcp_support` (facts about one page, not the corpus) |

Rules:

- Field types are limited to: `str`, `int`, `float`, `bool`, `date`.
  Nothing else — no `Optional`, no `Union`, no `list`, no `Literal`.
- **Do not use `Literal[...]`.** You have only one sample — you cannot
  know the full value space. Plain `str` is the right type for
  categories, genres, tags, etc.; a human tightens to `Literal` later.
- **Boolean flags must be corpus-general yes/no questions.** Good:
  `in_stock`, `auto_renews`, `is_opinion` — every item/contract/article
  has a truthy-or-falsy answer. Bad: `mentions_arbitration_clause`,
  `explains_mcp_transport` — topic predicates about *this* document
  that collapse to `false` across the rest of the corpus and give the
  user nothing to filter on.
- Pick 5–10 fields. If you can only think of four that pass the
  generality test, emit four. **Fewer generic fields beats more
  document-specific ones.** Padding the list with single-document facts
  is the failure mode to avoid.
- Each field is `Annotated[T, Field(description="...")]` with a short
  description — the description rides into the LLM prompt at extraction
  time.

Topic-level specifics of the sample (what it *says* about MCP, what
arbitration clause it *contains*, which party is *actually* named) are
captured by `QuestionAnswer` and `Summary` below, not by `Metadata`.
Let those two classes carry the document-specific content.

### 2. `QuestionAnswer(BaseCollection)` — Q&A pairs for semantic search

A single collection that captures question/answer pairs about the
document. The LLM emits all pairs in one call (`max_iterations = 1`),
and each pair lands in the vector store as its own retrievable entry,
so an agent can search by meaning of either side.

Rules:

- Exactly two string fields: `question` and `answer`.
- The docstring instructs the LLM to emit ten distinct pairs covering
  the key facts in the document, grounded strictly in its contents.
- Inner `Schema` class sets `max_iterations = 1`.
- `get_unique()` returns `self.question.casefold()`.
- `template()` renders the pair as `"{question}\n{answer}"`.

### 3. `Summary(BaseSemantic)` — one-paragraph summary

A marker class whose docstring is the question the LLM answers. Keep it
one sentence, focused on "what is this document about".

## Docstring generality — the same rule, for prompts

The generality test applies to **every docstring**, not just to `Metadata`
field choice. Every docstring is an LLM prompt that will run against
**every** document in the corpus at index time, not just the sample. So
the docstring must describe a question that makes sense for any
document of this kind — do **not** bake the sample's specific subject
matter into it.

- Refer to the **corpus type** (*"product page"*, *"legal contract"*,
  *"documentation page"*), not the sample's specific topic
  (*"about RGB keyboards"*, *"about the arbitration clause in §7"*,
  *"about the REST and MCP interfaces"*).
- Pull from the user's `--task` (e.g. *"filter products by price,
  category, and colour"*) for hints on *what to extract*, but keep the
  wording at the corpus level.
- Heuristic: read the docstring with the sample swapped out for a
  different document of the same kind. If it no longer makes sense, it
  is too specific.

Good / bad pairs:

| Corpus | Bad (sample-specific) | Good (corpus-generic) |
|---|---|---|
| Product pages, `Metadata` | *"Extract metadata about this RGB gaming keyboard product page."* | *"Extract filterable product-catalogue metadata (price, category, colour, availability)."* |
| Contracts, `Summary` | *"Summarize what this arbitration clause says."* | *"Summarize in one or two sentences what this contract is about."* |
| Docs pages, `Summary` | *"Summarize what this documentation page explains about the REST and MCP interfaces."* | *"Summarize in one or two sentences what this documentation page is about."* |
| Docs pages, `QuestionAnswer` | *"Generate ten Q&A pairs about the REST and MCP server setup."* | *"Generate ten Q&A pairs covering the key facts of this documentation page."* |

## Skeleton to emit

Only customise (a) the `Metadata` docstring and field set, (b) the
`QuestionAnswer` docstring wording (topic-appropriate), and (c) the
`Summary` docstring wording. Everything else stays as shown.

```python
from datetime import date
from typing import Annotated

from ennoia import BaseCollection, BaseSemantic, BaseStructure, Field


class Metadata(BaseStructure):
    """<one line naming the CORPUS TYPE and the kind of metadata to extract — no sample-specific subject matter>"""

    # 5–10 plain filterable fields, chosen from what is visible in the sample.
    title: Annotated[str, Field(description="Document or product title.")]
    published: Annotated[date, Field(description="Publication or release date.")]
    price_usd: Annotated[float, Field(description="Price in USD.")]
    in_stock: Annotated[bool, Field(description="Whether the item is currently available.")]


class QuestionAnswer(BaseCollection):
    """Generate ten question-and-answer pairs that cover the key facts of the document, grounded strictly in its contents."""

    question: Annotated[str, Field(description="Short factual question answerable from the document.")]
    answer: Annotated[str, Field(description="Concise answer to the question, one or two sentences.")]

    class Schema:
        max_iterations = 1

    def get_unique(self) -> str:
        return self.question.casefold()

    def template(self) -> str:
        return f"{self.question}\n{self.answer}"


class Summary(BaseSemantic):
    """Summarize in one or two sentences what this document is about."""
```

## Anti-patterns — do not

- **Pick `Metadata` fields that only fit the sample.** The single
  biggest failure mode. Every field must pass the generality test — it
  has to make sense on the other documents in the corpus, with a
  different value. Document-specific facts go in `QuestionAnswer` /
  `Summary`, not `Metadata`.
- **Bake the sample's subject matter into a docstring.** Docstrings are
  prompts that run against *every* document in the corpus. Phrases like
  *"about the REST and MCP interfaces"*, *"explains the arbitration
  clause"*, or *"this RGB gaming keyboard"* pin the prompt to one
  document. Use the corpus type (*"documentation page"*, *"contract"*,
  *"product page"*) instead.
- **Use topic predicates as boolean flags** (`mentions_X`, `covers_Y`,
  `explains_Z`). These collapse to `false` across the corpus and give
  the user nothing to filter on. Use semantic search instead.
- Use `Literal[...]` enums. Draft with plain `str`; the human refines
  later once the full value space is known.
- Use `Optional`, `Union`, or `list` in `Metadata`. Use the straight
  scalar type.
- Invent fields not present in the sample.
- Pad the `Metadata` field list toward the 10-field ceiling with weak,
  document-specific fields. Fewer is better.
- Emit more or fewer than the three named classes.
- Add `extend()`, `Schema.extensions`, or `Schema.namespace` — those are
  refinements for a later pass, not a first draft.
- Put triple backticks inside docstrings — it collides with the output
  parser. Use single backticks for any inline code.
- Declare an `extraction_confidence` or `confidence` field — Ennoia
  injects it at runtime.
