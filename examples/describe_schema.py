"""Print the canonical filter contract for a schema set.

The payload matches the shape an MCP server would expose via
``discover_schema()`` — ``docs/filters.md §Schema discovery``.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Annotated, Literal

from ennoia import BaseSemantic, BaseStructure, Field, describe


class CaseDocument(BaseStructure):
    """Extract case metadata."""

    jurisdiction: Literal["WA", "NY", "TX"]
    date_decided: date
    court_level: Literal["supreme", "appellate", "district"]
    is_overruled: bool
    tags: list[str]
    overruled_by: str | None = None
    title: Annotated[str, Field(description="Case title", operators=["eq", "contains"])] = ""


class Holding(BaseSemantic):
    """What is the core holding of this case?"""


class Facts(BaseSemantic):
    """Summarize the key facts of the case."""


def main() -> None:
    payload = describe([CaseDocument, Holding, Facts])
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
