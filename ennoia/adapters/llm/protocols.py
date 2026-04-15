"""LLMAdapter protocol — minimal interface every LLM backend implements."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["LLMAdapter"]


@runtime_checkable
class LLMAdapter(Protocol):
    async def complete_json(self, prompt: str) -> dict[str, Any]: ...

    async def complete_text(self, prompt: str) -> str: ...
