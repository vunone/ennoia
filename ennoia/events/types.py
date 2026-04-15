"""Typed event dataclasses emitted by the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["ExtractionEvent", "IndexEvent", "SearchEvent"]


@dataclass
class ExtractionEvent:
    """A single schema extraction finished (or failed)."""

    document_id: str
    schema_name: str
    duration_ms: float
    confidence: float
    llm_tokens_used: int = 0
    success: bool = True
    error: str | None = None


@dataclass
class IndexEvent:
    """A document-indexing cycle (all schemas) finished."""

    source_id: str
    schemas_extracted: list[str] = field(default_factory=list[str])
    rejected: bool = False
    duration_ms: float = 0.0


@dataclass
class SearchEvent:
    """A search call finished."""

    query: str
    filters: dict[str, Any] = field(default_factory=dict[str, Any])
    hit_count: int = 0
    duration_ms: float = 0.0
