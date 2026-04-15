"""Example schemas for the CLI walkthrough.

Paired with ``examples/fixtures/`` and the shell script
``examples/cli_walkthrough.sh``.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from ennoia import BaseSemantic, BaseStructure


class DocMeta(BaseStructure):
    """Extract basic document metadata."""

    category: Literal["legal", "medical", "financial"]
    doc_date: date


class Summary(BaseSemantic):
    """What is the main topic of this document?"""
