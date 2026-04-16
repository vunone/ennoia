"""Example schemas for the CLI walkthrough.

Paired with ``examples/fixtures/`` and the shell script
``examples/cli_walkthrough.sh``.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from ennoia import BaseSemantic, BaseStructure, Field


class DocMeta(BaseStructure):
    """Extract basic document metadata."""

    category: Literal["legal", "medical", "financial"]
    doc_date: Annotated[date, Field(description="Datetime in ISO 8601 format")]


class Summary(BaseSemantic):
    """What is the main topic of this document?"""
