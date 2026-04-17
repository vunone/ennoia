"""Exceptions raised by the indexing pipeline."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ExtractionError",
    "FilterValidationError",
    "RejectException",
    "SchemaError",
    "SchemaWarning",
    "SkipItem",
]


class ExtractionError(Exception):
    """Raised when LLM output cannot be validated against a schema after retry."""


class RejectException(Exception):
    """Raise from user code to explicitly drop a document from indexing.

    The pipeline catches this, returns an `IndexResult(rejected=True)`,
    and writes nothing to the stores. Intended for schema validators or
    `extend()` logic that detects an out-of-scope document.
    """


class SkipItem(Exception):
    """Raise from ``BaseCollection.is_valid`` to drop a single extracted entity.

    Unlike :class:`RejectException` (which drops the whole document), this
    only discards the entity it is raised from; the collection loop continues
    and the rest of the pipeline proceeds. Caught exclusively by the collection
    extractor.
    """


class SchemaError(Exception):
    """Raised when a schema graph is malformed or emits undeclared classes.

    Used both at pipeline init (manifest validation — cycles, incompatible
    field merges, reserved field names, invalid namespace) and at runtime
    (``extend()`` returning a class not listed in ``Schema.extensions``).
    """


class SchemaWarning(UserWarning):
    """Emitted during pipeline init when the manifest is well-formed but
    has soft issues — for example, divergent descriptions on a field that
    is merged from multiple sources. Does not block initialization.
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
