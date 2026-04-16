"""Ennoia — LLM-powered document pre-indexing and hybrid retrieval."""

from ennoia.events import Emitter, ExtractionEvent, IndexEvent, SearchEvent
from ennoia.index import Pipeline
from ennoia.index.exceptions import ExtractionError, FilterValidationError, RejectException
from ennoia.index.result import IndexResult, SearchHit, SearchResult
from ennoia.schema import BaseSemantic, BaseStructure, Field, describe
from ennoia.store import Store

__all__ = [
    "BaseSemantic",
    "BaseStructure",
    "Emitter",
    "ExtractionError",
    "ExtractionEvent",
    "Field",
    "FilterValidationError",
    "IndexEvent",
    "IndexResult",
    "Pipeline",
    "RejectException",
    "SearchEvent",
    "SearchHit",
    "SearchResult",
    "Store",
    "describe",
]

__version__ = "0.3.0"
