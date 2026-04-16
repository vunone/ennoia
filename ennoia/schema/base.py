"""Base classes users subclass to declare extraction schemas."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ennoia.schema.operators import describe_field

__all__ = [
    "BaseSemantic",
    "BaseStructure",
    "get_schema_extensions",
    "get_schema_namespace",
]


def get_schema_namespace(cls: type) -> str | None:
    """Return the namespace declared on ``cls.Schema`` (None when unset).

    User schemas declare their inner ``Schema`` as a bare class with no
    base, so we read attributes via ``getattr`` with defaults rather than
    rely on inheritance of a framework marker class.
    """
    schema = getattr(cls, "Schema", None)
    if schema is None:
        return None
    ns = getattr(schema, "namespace", None)
    return ns if isinstance(ns, str) and ns else None


def get_schema_extensions(cls: type) -> list[type]:
    """Return a fresh list of classes declared in ``cls.Schema.extensions``.

    Returns ``[]`` when the schema has no inner ``Schema`` or no
    ``extensions`` attribute. A copy is returned so callers may iterate
    without risking mutation of the declaration.
    """
    schema = getattr(cls, "Schema", None)
    if schema is None:
        return []
    exts = getattr(schema, "extensions", None)
    if exts is None:
        return []
    return list(exts)


def _clean_docstring(obj: type) -> str:
    doc = obj.__dict__.get("__doc__")
    if not doc:
        raise ValueError(
            f"{obj.__name__} has no docstring. The docstring is the extraction prompt "
            "and must be provided."
        )
    return doc.strip()


class BaseStructure(BaseModel):
    """Base class for structured extraction schemas.

    Subclasses declare fields whose types drive validation and filter-operator
    inference. The subclass docstring is the extraction prompt passed to the
    LLM. ``model_config`` allows extra fields so the extractor can retain a
    trailing ``_confidence`` value on the instance for ``extend()`` to consult
    without polluting the static JSON schema.
    """

    model_config = ConfigDict(extra="allow")

    __ennoia_kind__: ClassVar[str] = "structural"

    class Schema:
        """Ennoia-level schema configuration, orthogonal to ``model_config``.

        Subclasses override by declaring their own bare inner ``class Schema``
        (no base needed). The framework reads attributes via ``getattr`` with
        defaults; a user subclass that omits an attribute inherits the default
        transparently even though Python does not merge inherited nested
        classes.

        - ``namespace``: when set, fields merge into the superschema under the
          prefix ``{namespace}__``; otherwise fields merge flat.
        - ``extensions``: the complete list of classes ``extend()`` may return.
          Enforced at runtime — undeclared emissions raise ``SchemaError``.
        """

        namespace: ClassVar[str | None] = None
        extensions: ClassVar[list[type]] = []

    @classmethod
    def extract_prompt(cls) -> str:
        return _clean_docstring(cls)

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        return cls.model_json_schema()

    @classmethod
    def describe_schema(cls) -> dict[str, Any]:
        """Return the filter-contract description for this schema.

        See ``docs/filters.md §Schema Discovery`` for the payload shape.
        """
        fields: list[dict[str, Any]] = []
        for name, info in cls.model_fields.items():
            record = describe_field(name, info)
            if record is not None:
                fields.append(record)
        return {"name": cls.__name__, "fields": fields}

    def extend(self) -> list[type[BaseStructure] | type[BaseSemantic]]:
        """Return additional schemas to apply after this one is extracted.

        The pipeline calls ``extend()`` on the populated instance and queues
        the returned schemas (structural or semantic) for further extraction
        against the same document. The parent instance's extracted values —
        including ``_confidence`` — are available via ``self``. This is the
        sole mechanism for inter-schema dependencies.

        Default: no extension.
        """
        return []


class BaseSemantic:
    """Base class for semantic (free-text) extraction schemas.

    A semantic schema is a marker class whose docstring is the question the
    LLM should answer about a document. The answer text is embedded and
    stored for vector search, linked to the source document by reference id.
    """

    __ennoia_kind__: ClassVar[str] = "semantic"

    @classmethod
    def extract_prompt(cls) -> str:
        return _clean_docstring(cls)
