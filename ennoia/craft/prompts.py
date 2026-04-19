"""Prompt builders for the ``ennoia craft`` LLM loop."""

from __future__ import annotations

from ennoia.prompts import load_prompt

__all__ = [
    "build_retry_prompt",
    "build_system_prompt",
    "build_user_prompt",
]


_FORMAT_REMINDER = (
    "\n\n---\n"
    "Formatting reminder:\n"
    "- Reply with exactly one ```python fenced block and no prose outside it.\n"
    "- The code must import cleanly from `ennoia`, `typing`, `datetime`, `pydantic`.\n"
    "- Use single backticks inside docstrings so the closing fence is unambiguous."
)


def build_system_prompt() -> str:
    """Return the full system prompt (bundled authoring guide + reminder)."""
    return load_prompt("craft") + _FORMAT_REMINDER


def build_user_prompt(
    *,
    task: str,
    document: str,
    existing_schema: str | None,
) -> str:
    """Compose the user prompt from the ``--task`` and the sample document.

    When ``existing_schema`` is provided, it is included so the LLM
    *improves* it instead of rewriting from scratch.
    """
    parts = [
        "# Task",
        task.strip(),
        "",
        "# Document sample",
        document,
    ]
    if existing_schema is not None and existing_schema.strip():
        parts.extend(
            [
                "",
                "# Current schema (improve this; preserve the parts that already work)",
                "```python",
                existing_schema,
                "```",
            ]
        )
    parts.extend(
        [
            "",
            "# Deliver",
            "A single ```python fenced block containing exactly three classes "
            "in order — Metadata (BaseStructure), QuestionAnswer (BaseCollection), "
            "Summary (BaseSemantic) — following the skeleton from the guide.",
        ]
    )
    return "\n".join(parts)


def build_retry_prompt(
    *,
    previous_user_prompt: str,
    previous_reply: str,
    error: str,
) -> str:
    """Compose a retry prompt that surfaces the previous failure to the LLM."""
    return "\n".join(
        [
            previous_user_prompt,
            "",
            "# Your previous reply",
            previous_reply,
            "",
            "# It failed with",
            error,
            "",
            "Return a corrected ```python fenced block. Fix the reported error "
            "and preserve the parts of the previous reply that already work.",
        ]
    )
