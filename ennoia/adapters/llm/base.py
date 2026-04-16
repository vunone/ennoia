"""Abstract base class every LLM adapter must implement.

These ABCs replace the former ``typing.Protocol`` definitions. Inheritance
(rather than structural typing) is used so that concrete adapters have a
single, discoverable contract and so that instantiating a partially
implemented adapter fails loudly with ``TypeError``. Shared JSON parsing
for json-mode adapters lives here as a module-level helper; Anthropic's
tool_use path does not share it, so it is not a method on the ABC.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, cast

from ennoia.index.exceptions import ExtractionError

__all__ = ["LLMAdapter", "parse_json_object"]


def parse_json_object(content: str, source: str) -> dict[str, Any]:
    """Parse ``content`` as a JSON object; raise ``ExtractionError`` on failure."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as err:
        raise ExtractionError(f"{source} returned non-JSON content: {content!r}") from err
    if not isinstance(parsed, dict):
        raise ExtractionError(f"Expected a JSON object from {source}, got {type(parsed).__name__}.")
    return cast(dict[str, Any], parsed)


class LLMAdapter(ABC):
    """Minimal contract every LLM backend implements."""

    @abstractmethod
    async def complete_json(self, prompt: str) -> dict[str, Any]: ...

    @abstractmethod
    async def complete_text(self, prompt: str) -> str: ...
