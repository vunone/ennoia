"""Result types returned by Pipeline.index / Pipeline.search."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from ennoia.index.extractor import CONFIDENCE_KEY
from ennoia.schema.base import BaseStructure

__all__ = ["IndexResult", "SearchHit", "SearchResult"]


@dataclass
class IndexResult:
    source_id: str
    structural: dict[str, BaseStructure] = field(default_factory=dict[str, BaseStructure])
    semantic: dict[str, str] = field(default_factory=dict[str, str])
    confidences: dict[str, float] = field(default_factory=dict[str, float])
    rejected: bool = False

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source_id": self.source_id,
            "rejected": self.rejected,
            "structural": {
                name: {
                    k: v for k, v in model.model_dump(mode="json").items() if k != CONFIDENCE_KEY
                }
                for name, model in self.structural.items()
            },
            "semantic": {
                name: (text[:120] + "..." if len(text) > 120 else text)
                for name, text in self.semantic.items()
            },
            "confidences": dict(self.confidences),
        }
        return out


@dataclass
class SearchHit:
    source_id: str
    score: float
    structural: dict[str, Any]
    semantic: dict[str, str] = field(default_factory=dict[str, str])
    confidences: dict[str, float] = field(default_factory=dict[str, float])


@dataclass
class SearchResult:
    hits: list[SearchHit] = field(default_factory=list[SearchHit])

    def __iter__(self) -> Iterator[SearchHit]:
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)

    def __bool__(self) -> bool:
        return bool(self.hits)
