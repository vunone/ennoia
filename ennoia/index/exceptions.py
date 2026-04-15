"""Exceptions raised by the indexing pipeline."""

from __future__ import annotations

from typing import Any

__all__ = ["ExtractionError", "FilterValidationError", "RejectException"]


class ExtractionError(Exception):
    """Raised when LLM output cannot be validated against a schema after retry."""


class RejectException(Exception):
    """Raise from user code to explicitly drop a document from indexing.

    The pipeline catches this, returns an `IndexResult(rejected=True)`,
    and writes nothing to the stores. Intended for schema validators or
    `extend()` logic that detects an out-of-scope document.
    """


class FilterValidationError(ValueError):
    """Raised when a search filter references unknown fields or operators.

    The error payload mirrors ``docs/filters.md §Filter Validation`` so every
    surface (SDK, CLI, MCP, REST) can render it identically.
    """

    def __init__(
        self,
        *,
        field: str,
        operator: str,
        message: str,
        supported: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.field = field
        self.operator = operator
        self.message = message
        self.supported = list(supported) if supported is not None else []

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": "invalid_filter",
            "field": self.field,
            "operator": self.operator,
            "message": self.message,
        }
