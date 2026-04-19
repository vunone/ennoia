"""Schema crafting — LLM-driven schema scaffolding from a sample document.

``ennoia craft`` points an LLM at a document and a retrieval task,
extracts the ```python``` block from its reply, writes it to ``--output``,
and validates the file by importing it. If the import fails, the
traceback is fed back to the LLM on the next attempt.
"""

from __future__ import annotations

from ennoia.craft.agent import (
    CraftError,
    CraftLLMError,
    CraftValidationError,
    run_craft_loop,
)
from ennoia.craft.codeblock import CraftParseError, extract_python_block
from ennoia.craft.validate import validate_schema_file

__all__ = [
    "CraftError",
    "CraftLLMError",
    "CraftParseError",
    "CraftValidationError",
    "extract_python_block",
    "run_craft_loop",
    "validate_schema_file",
]
