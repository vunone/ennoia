"""Typed event bus for ennoia pipelines.

The emitter is an optional, pluggable observability hook. Every extraction,
indexing, and search operation emits a typed dataclass event; handlers
register per event type and run synchronously. See ``docs/concepts.md`` for
the rationale (cost tracking, confidence monitoring, debugging).
"""

from __future__ import annotations

from ennoia.events.emitter import Emitter, NullEmitter
from ennoia.events.types import ExtractionEvent, IndexEvent, SearchEvent

__all__ = [
    "Emitter",
    "ExtractionEvent",
    "IndexEvent",
    "NullEmitter",
    "SearchEvent",
]
