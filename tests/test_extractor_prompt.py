"""Prompt builder shape tests."""

from __future__ import annotations

import json
import re
from typing import Literal

from ennoia import BaseSemantic, BaseStructure, Field
from ennoia.index.extractor import (
    CONFIDENCE_KEY,
    augment_json_schema_with_confidence,
    build_semantic_prompt,
    build_structural_prompt,
)


class Meta(BaseStructure):
    """Extract document metadata."""

    category: Literal["legal", "medical"]
    title: str = Field(description="The document title.")


class Nested(BaseStructure):
    """Extract a wrapper carrying inner metadata."""

    inner: Meta


class Summary(BaseSemantic):
    """What is the main topic?"""


_JSON_BLOCK = re.compile(r"```json\n(?P<body>.*?)\n```", re.DOTALL)


def _extract_json_block(prompt: str) -> dict[str, object]:
    match = _JSON_BLOCK.search(prompt)
    assert match is not None, "prompt is missing a ```json``` block"
    return json.loads(match.group("body"))


def test_structural_prompt_contains_role_task_and_schema():
    prompt = build_structural_prompt(Meta, "Doc body.")

    assert prompt.startswith("You are a document structure expert.")
    assert "# Task\nExtract document metadata." in prompt
    assert "# Output Format" in prompt
    expected_schema = augment_json_schema_with_confidence(Meta.model_json_schema())
    assert _extract_json_block(prompt) == expected_schema
    # The confidence property must be appended last and also required.
    props = list(_extract_json_block(prompt)["properties"])
    assert props[-1] == CONFIDENCE_KEY
    assert CONFIDENCE_KEY in _extract_json_block(prompt)["required"]
    assert "<DocumentContent>\nDoc body.\n</DocumentContent>" in prompt
    assert "# Additional Context" not in prompt


def test_structural_prompt_renders_additional_context_when_provided():
    prompt = build_structural_prompt(
        Meta,
        "Doc body.",
        context_additions=["prior: foo", "prior: bar"],
    )

    assert "# Additional Context\n- prior: foo\n- prior: bar" in prompt


def test_structural_prompt_omits_additional_context_when_empty():
    prompt = build_structural_prompt(Meta, "Doc body.", context_additions=[])

    assert "# Additional Context" not in prompt


def test_structural_prompt_embeds_nested_schema_under_defs():
    prompt = build_structural_prompt(Nested, "Doc body.")
    schema = _extract_json_block(prompt)

    assert "$defs" in schema
    assert "Meta" in schema["$defs"]
    assert schema["$defs"]["Meta"]["properties"]["category"]["enum"] == [
        "legal",
        "medical",
    ]


def test_semantic_prompt_contains_role_task_and_document():
    prompt = build_semantic_prompt(Summary, "Doc body.")

    assert prompt.startswith("You are a document analyst.")
    assert "# Task\nWhat is the main topic?" in prompt
    assert "# Output Format" in prompt
    assert "<DocumentContent>\nDoc body.\n</DocumentContent>" in prompt
    assert "# Additional Context" not in prompt
    assert "```json" not in prompt
