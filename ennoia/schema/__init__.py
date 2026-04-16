"""User-facing schema declaration layer."""

from __future__ import annotations

from typing import Any

from ennoia.schema.base import BaseSemantic, BaseStructure
from ennoia.schema.fields import Field

__all__ = [
    "BaseSemantic",
    "BaseStructure",
    "Field",
    "describe",
]


def describe(
    schemas: list[type[BaseStructure] | type[BaseSemantic]],
) -> dict[str, Any]:
    """Return the superschema discovery payload for ``schemas``.

    ``schemas`` is the list of root schemas (the same set passed to
    ``Pipeline(schemas=...)``). The function computes the emission
    manifest (transitive closure of ``Schema.extensions``) and the
    superschema, then renders the JSON discovery payload documented in
    ``docs/filters.md §Schema Discovery``.

    Fields from schemas reachable via ``Schema.extensions`` appear
    correctly merged — flat by default, prefixed under ``{namespace}__``
    when a source declares ``Schema.namespace``.
    """
    from ennoia.schema.manifest import build_manifest
    from ennoia.schema.merging import build_superschema

    manifest = build_manifest(schemas)
    superschema = build_superschema(manifest)
    return superschema.to_discovery_payload()
