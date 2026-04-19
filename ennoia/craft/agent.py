"""Orchestrate the LLM → extract → validate → retry loop."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from ennoia.craft.codeblock import CraftParseError, extract_python_block
from ennoia.craft.prompts import build_retry_prompt, build_system_prompt, build_user_prompt
from ennoia.craft.validate import validate_schema_file

if TYPE_CHECKING:
    from ennoia.adapters.llm.base import LLMAdapter

__all__ = [
    "CraftError",
    "CraftLLMError",
    "CraftValidationError",
    "run_craft_loop",
]


class CraftError(Exception):
    """Base class for errors surfaced by the craft loop."""


class CraftLLMError(CraftError):
    """The LLM call itself failed (network error, context length, etc.)."""


class CraftValidationError(CraftError):
    """The generated schema file could not be validated within the retry budget."""


ProgressCallback = Callable[[int, str], None]


async def run_craft_loop(
    *,
    llm: LLMAdapter,
    task: str,
    document: str,
    output_path: Path,
    existing_schema: str | None,
    max_retries: int = 2,
    on_attempt: ProgressCallback | None = None,
) -> None:
    """Run the craft loop until validation succeeds or the budget is exhausted.

    The loop runs at most ``max_retries + 1`` attempts (``max_retries = 2``
    → up to 3 LLM calls). The first call builds a fresh schema from the
    document + task; each subsequent call sees its predecessor's reply
    and the validator's error string, and is asked to fix it while
    preserving the working parts.

    On success, the validated schema sits at ``output_path``. On
    validation failure after the final attempt, the partially-written
    file is left on disk for the user to inspect and a
    :class:`CraftValidationError` is raised. Provider-side errors
    (authentication, context length, etc.) short-circuit the loop with
    :class:`CraftLLMError`.
    """
    user_prompt = build_user_prompt(
        task=task,
        document=document,
        existing_schema=existing_schema,
    )
    system_prompt = build_system_prompt()
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    last_error = ""
    for attempt in range(max_retries + 1):
        if on_attempt is not None:
            on_attempt(attempt, "calling LLM")

        try:
            reply = await llm.complete_text(full_prompt)
        except Exception as exc:
            raise CraftLLMError(
                f"LLM call failed: {exc}. If the document exceeds the model's "
                "context window, try a model with a larger context."
            ) from exc

        try:
            code = extract_python_block(reply)
        except CraftParseError as err:
            last_error = f"Your reply did not contain a python code block: {err}"
        else:
            output_path.write_text(code, encoding="utf-8")
            if on_attempt is not None:
                on_attempt(attempt, "validating schema")
            validation_error = validate_schema_file(output_path)
            if validation_error is None:
                return
            last_error = validation_error

        full_prompt = build_retry_prompt(
            previous_user_prompt=user_prompt,
            previous_reply=reply,
            error=last_error,
        )

    raise CraftValidationError(last_error or "craft loop exhausted retries")
