"""Prompt construction and single-schema LLM extraction.

Confidence plumbing
-------------------

``_confidence`` is never declared on :class:`BaseStructure`. If it were, it
would appear at the top of the JSON Schema and bias the LLM into picking a
score before filling the real fields. Instead, the extractor **dynamically
appends** ``_confidence`` as the final property of the prompted schema and
instructs the model to emit it last. This encourages the model to evaluate
itself *after* generating the extraction.

``BaseStructure.model_config = ConfigDict(extra="allow")`` (see
:mod:`ennoia.schema.base`) allows the returned ``_confidence`` value to ride
on the validated instance so ``extend()`` can consult it. The Pipeline strips
it before persistence.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, cast

from pydantic import ValidationError

from ennoia.index.exceptions import ExtractionError, SkipItem
from ennoia.schema.base import (
    BaseCollection,
    BaseSemantic,
    BaseStructure,
    get_schema_max_iterations,
)

if TYPE_CHECKING:
    from ennoia.adapters.llm.base import LLMAdapter

__all__ = [
    "CONFIDENCE_KEY",
    "augment_json_schema_with_confidence",
    "build_collection_prompt",
    "build_collection_schema",
    "build_semantic_prompt",
    "build_structural_prompt",
    "extract_collection",
    "extract_semantic",
    "extract_structural",
]

_log = logging.getLogger(__name__)

CONFIDENCE_KEY = "_confidence"

_STRUCTURAL_ROLE = (
    "You are a document structure expert. Your goal is to carefully read the given "
    "document and extract the necessary data according to the **Task** and output a "
    "JSON-object strictly according to **Output Format**."
)

_STRUCTURAL_OUTPUT_FORMAT = (
    "# Output Format\n"
    "- Respond strictly according to JSON-Schema provided below\n"
    "- Emit JSON properties in the exact order shown in the schema\n"
    f"- The `{CONFIDENCE_KEY}` property MUST be the FINAL property in the JSON object, "
    "evaluated only after every other field has been filled — it reflects your confidence "
    "(0.0–1.0) in the extraction as a whole\n"
    "- Avoid any prose / comments before or after a JSON-object output\n"
    "- Do not use any formatting (e.g, markdown) to wrap the output JSON-object"
)

_SEMANTIC_ROLE = (
    "You are a document analyst. Your goal is to carefully read the given document "
    "and answer the question in the **Task** according to the **Output Format**."
)

_SEMANTIC_OUTPUT_FORMAT = (
    "# Output Format\n"
    "- Answer in one concise paragraph grounded in the document below\n"
    "- Avoid any prose / comments before or after the answer\n"
    "- Do not use any formatting (e.g, markdown)\n"
    "- After your answer, append a single line of the form "
    "`<confidence>0.xx</confidence>` reporting your confidence (0.0–1.0) in the answer"
)

_SEMANTIC_CONFIDENCE_RE = re.compile(
    r"<confidence>\s*([0-9]*\.?[0-9]+)\s*</confidence>\s*$",
    re.IGNORECASE,
)


def augment_json_schema_with_confidence(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``schema`` with ``_confidence`` appended as the last property.

    Python dicts preserve insertion order, so appending here guarantees that
    the JSON rendering — and the model's expected emission order — places
    ``_confidence`` at the end of the object.
    """
    augmented: dict[str, Any] = {k: v for k, v in schema.items() if k != "properties"}
    properties: dict[str, Any] = dict(schema.get("properties", {}))
    properties[CONFIDENCE_KEY] = {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
        "description": (
            "Self-reported confidence (0.0–1.0) in the extraction. "
            "Evaluate this AFTER filling every field above. MUST be the last property."
        ),
    }
    augmented["properties"] = properties

    required = list(schema.get("required", []))
    if CONFIDENCE_KEY not in required:
        required.append(CONFIDENCE_KEY)
    augmented["required"] = required
    return augmented


def build_structural_prompt(
    schema: type[BaseStructure],
    document_text: str,
    context_additions: list[str] | None = None,
) -> str:
    sections: list[str] = [
        _STRUCTURAL_ROLE,
        f"# Task\n{schema.extract_prompt()}",
    ]

    if context_additions:
        bullets = "\n".join(f"- {addition}" for addition in context_additions)
        sections.append(f"# Additional Context\n{bullets}")

    schema_json = json.dumps(augment_json_schema_with_confidence(schema.json_schema()), indent=2)
    sections.append(f"{_STRUCTURAL_OUTPUT_FORMAT}\n\n```json\n{schema_json}\n```")
    sections.append(f"<DocumentContent>\n{document_text}\n</DocumentContent>")

    return "\n\n".join(sections)


def build_semantic_prompt(schema: type[BaseSemantic], document_text: str) -> str:
    sections = [
        _SEMANTIC_ROLE,
        f"# Task\n{schema.extract_prompt()}",
        _SEMANTIC_OUTPUT_FORMAT,
        f"<DocumentContent>\n{document_text}\n</DocumentContent>",
    ]
    return "\n\n".join(sections)


def _split_confidence(raw: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """Pop ``_confidence`` from ``raw`` and return ``(raw_without_confidence, confidence)``.

    The instance still receives the full dict (``extra="allow"`` keeps the
    field on the model) so ``extend()`` can read ``self._confidence``. Pipeline
    strips it again at persistence time.
    """
    if CONFIDENCE_KEY not in raw:
        _log.warning("LLM omitted %s; defaulting confidence to 1.0", CONFIDENCE_KEY)
        return raw, 1.0
    value = raw.get(CONFIDENCE_KEY, 1.0)
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        _log.warning("Non-numeric %s=%r; defaulting confidence to 1.0", CONFIDENCE_KEY, value)
        confidence = 1.0
    return raw, max(0.0, min(1.0, confidence))


async def extract_structural(
    schema: type[BaseStructure],
    text: str,
    context_additions: list[str],
    llm: LLMAdapter,
) -> tuple[BaseStructure, float]:
    """Extract a structural schema from ``text``; return the instance + confidence.

    Retries once on Pydantic validation failure with the error message
    appended to the prompt. Raises :class:`ExtractionError` if the second
    attempt also fails.
    """
    prompt = build_structural_prompt(schema, text, context_additions)
    try:
        raw = await llm.complete_json(prompt)
        raw, confidence = _split_confidence(dict(raw))
        return schema.model_validate(raw), confidence
    except ValidationError as first_err:
        retry_prompt = (
            prompt + "\n\nThe previous JSON failed validation with these errors. "
            "Return corrected JSON only:\n" + str(first_err)
        )
        try:
            raw = await llm.complete_json(retry_prompt)
            raw, confidence = _split_confidence(dict(raw))
            return schema.model_validate(raw), confidence
        except ValidationError as second_err:
            raise ExtractionError(
                f"Failed to extract {schema.__name__} after retry: {second_err}"
            ) from second_err


_COLLECTION_ROLE = (
    "You are a document structure expert. Your goal is to carefully read the given "
    "document and extract **every** entity described in the **Task** into a JSON-object "
    "strictly according to **Output Format**. You may be called several times in a row; "
    "on each call, the `<PreviouslyExtracted>` block lists entities that have already "
    "been captured — do not re-emit them, but continue extracting any entities not yet "
    "present there."
)


_COLLECTION_OUTPUT_FORMAT = (
    "# Output Format\n"
    "- Respond strictly according to JSON-Schema provided below\n"
    "- Emit JSON properties in the exact order shown in the schema\n"
    "- Each entity in `entities_list` MUST include its own "
    f"`{CONFIDENCE_KEY}` property as the FINAL property of the entity object, "
    "evaluated only after every other entity field has been filled\n"
    "- `is_done` MUST be set to false if the document still mentions entities not "
    "captured in either the current `entities_list` or the `<PreviouslyExtracted>` block\n"
    "- Avoid any prose / comments before or after the JSON object\n"
    "- Do not use any formatting (e.g, markdown) to wrap the output JSON object"
)


def build_collection_schema(schema: type[BaseCollection]) -> dict[str, Any]:
    """Build the ``{entities_list, is_done}`` wrapper schema for a collection.

    The item schema is ``schema.json_schema()`` augmented with ``_confidence``
    as its final property — the same mechanism structural extractions use, but
    applied per-entity instead of top-level.
    """
    item_schema = augment_json_schema_with_confidence(schema.json_schema())
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "entities_list": {
                "type": "array",
                "items": item_schema,
            },
            "is_done": {
                "type": "boolean",
                "description": (
                    "Set to false if the document contains entities not mentioned in "
                    "either `entities_list` field or `<PreviouslyExtracted>` block."
                ),
            },
        },
        "required": ["entities_list", "is_done"],
    }


def _render_previously_extracted(collected: list[BaseCollection]) -> str:
    if not collected:
        return ""
    lines: list[str] = []
    for idx, instance in enumerate(collected, start=1):
        dumped = instance.model_dump(mode="json")
        dumped.pop(CONFIDENCE_KEY, None)
        lines.append(f"{idx}. " + json.dumps(dumped, default=str, sort_keys=True))
    body = "\n".join(lines)
    return f"<PreviouslyExtracted>\n{body}\n</PreviouslyExtracted>"


def build_collection_prompt(
    schema: type[BaseCollection],
    document_text: str,
    context_additions: list[str] | None,
    collected: list[BaseCollection],
) -> str:
    sections: list[str] = [
        _COLLECTION_ROLE,
        f"# Task\n{schema.extract_prompt()}",
    ]

    if context_additions:
        bullets = "\n".join(f"- {addition}" for addition in context_additions)
        sections.append(f"# Additional Context\n{bullets}")

    previously = _render_previously_extracted(collected)
    if previously:
        sections.append(previously)

    schema_json = json.dumps(build_collection_schema(schema), indent=2)
    sections.append(f"{_COLLECTION_OUTPUT_FORMAT}\n\n```json\n{schema_json}\n```")
    sections.append(f"<DocumentContent>\n{document_text}\n</DocumentContent>")

    return "\n\n".join(sections)


async def extract_collection(
    schema: type[BaseCollection],
    text: str,
    context_additions: list[str],
    llm: LLMAdapter,
) -> tuple[list[BaseCollection], list[float]]:
    """Run the multi-iteration collection loop; return the unique validated entities.

    Termination (whichever comes first):

    1. ``is_done == True`` in the LLM response.
    2. ``entities_list`` is empty on the current response.
    3. No new unique valid entity was added this iteration.
    4. ``schema.Schema.max_iterations`` (when set) reached.

    Individual malformed items and items raising :class:`SkipItem` from
    :meth:`BaseCollection.is_valid` are silently dropped. :class:`RejectException`
    raised from ``is_valid`` propagates up to :meth:`Pipeline.aindex`, which drops
    the whole document.
    """
    max_iterations = get_schema_max_iterations(schema)
    collected: list[BaseCollection] = []
    confidences: list[float] = []
    seen_uniques: set[str] = set()
    iterations = 0

    while True:
        iterations += 1
        prompt = build_collection_prompt(schema, text, context_additions, collected)

        try:
            raw = await llm.complete_json(prompt)
        except ValidationError as err:  # pragma: no cover — complete_json does not validate
            raise ExtractionError(f"Collection {schema.__name__} failed: {err}") from err

        raw_entities = raw.get("entities_list")
        entities: list[Any] = (
            list(cast("list[Any]", raw_entities)) if isinstance(raw_entities, list) else []
        )
        is_done = bool(raw.get("is_done", False))

        new_this_iter = 0
        for raw_item in entities:
            if not isinstance(raw_item, dict):
                continue
            item_dict = cast(dict[str, Any], raw_item)
            item_raw, confidence = _split_confidence(dict(item_dict))
            try:
                instance = schema.model_validate(item_raw)
            except ValidationError:
                _log.warning("Collection %s: dropping malformed item %r", schema.__name__, item_raw)
                continue
            try:
                instance.is_valid()
            except SkipItem:
                continue
            unique = instance.get_unique()
            if unique in seen_uniques:
                continue
            seen_uniques.add(unique)
            collected.append(instance)
            confidences.append(confidence)
            new_this_iter += 1

        if not entities:
            break
        if is_done:
            break
        if new_this_iter == 0:
            break
        if max_iterations is not None and iterations >= max_iterations:
            break

    return collected, confidences


async def extract_semantic(
    schema: type[BaseSemantic],
    text: str,
    llm: LLMAdapter,
) -> tuple[str, float]:
    """Extract a semantic answer + confidence.

    Confidence is pulled from a trailing ``<confidence>0.xx</confidence>`` tag;
    when absent it defaults to 1.0 without failing.
    """
    prompt = build_semantic_prompt(schema, text)
    response = (await llm.complete_text(prompt)).strip()

    match = _SEMANTIC_CONFIDENCE_RE.search(response)
    if match:
        # Regex guarantees ``match.group(1)`` is a well-formed decimal literal,
        # so ``float`` never raises here.
        confidence = max(0.0, min(1.0, float(match.group(1))))
        response = response[: match.start()].rstrip()
    else:
        confidence = 1.0

    return response, confidence
