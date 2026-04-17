"""Base classes users subclass to declare extraction schemas."""

from __future__ import annotations

import secrets
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ennoia.schema.operators import describe_field

__all__ = [
    "BaseCollection",
    "BaseSemantic",
    "BaseStructure",
    "get_schema_extensions",
    "get_schema_max_iterations",
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


def get_schema_max_iterations(cls: type) -> int | None:
    """Return ``cls.Schema.max_iterations`` or ``None`` when unset.

    Only meaningful for :class:`BaseCollection` subclasses; structural and
    semantic schemas do not iterate. The helper tolerates missing attributes
    so the inner ``Schema`` can stay a bare class without forcing users to
    spell out every knob.
    """
    schema = getattr(cls, "Schema", None)
    if schema is None:
        return None
    value = getattr(schema, "max_iterations", None)
    if value is None:
        return None
    if not isinstance(value, int):
        raise TypeError(
            f"{cls.__name__}.Schema.max_iterations must be int or None, got {type(value).__name__}."
        )
    if value < 1:
        raise ValueError(
            f"{cls.__name__}.Schema.max_iterations must be >= 1 (or None), got {value}."
        )
    return value


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

    def extend(self) -> list[type[BaseStructure] | type[BaseSemantic] | type[BaseCollection]]:
        """Return additional schemas to apply after this one is extracted.

        The pipeline calls ``extend()`` on the populated instance and queues
        the returned schemas (structural, semantic, or collection) for further
        extraction against the same document. The parent instance's extracted
        values — including ``_confidence`` — are available via ``self``. This
        is the sole mechanism for inter-schema dependencies.

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


class BaseCollection(BaseModel):
    """Base class for iterative list extraction.

    A collection schema declares the shape of **one** entity the LLM should
    extract; the pipeline repeatedly asks for more until the model signals
    completion (``is_done``) or a guardrail trips (see
    :attr:`Schema.max_iterations`). Each extracted entity is turned into an
    embeddable string by :meth:`template` and stored as a semantic entry —
    from storage's perspective, a collection with N entities behaves like N
    :class:`BaseSemantic` answers under the same index name.

    The per-entity fields drive prompt shape and :meth:`template` output; they
    are **not** persisted as tabular columns in the structured store.
    """

    model_config = ConfigDict(extra="allow")

    __ennoia_kind__: ClassVar[str] = "collection"

    class Schema:
        """Ennoia-level configuration, orthogonal to ``model_config``.

        - ``extensions``: classes ``extend()`` may return (structural, semantic,
          or other collections). Enforced at runtime.
        - ``max_iterations``: cap on iteration count. ``None`` means unbounded;
          the loop then relies on ``is_done``, empty ``entities_list``, or
          no-new-unique-items for termination.
        - ``namespace``: present for parity with :class:`BaseStructure`, but has
          no persistence effect — collections don't contribute tabular fields.
        """

        namespace: ClassVar[str | None] = None
        extensions: ClassVar[list[type]] = []
        max_iterations: ClassVar[int | None] = None

    @classmethod
    def extract_prompt(cls) -> str:
        return _clean_docstring(cls)

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        return cls.model_json_schema()

    def is_valid(self) -> None:
        """Validate this entity. Raise :class:`SkipItem` to drop just this one.

        Raise :class:`RejectException` to drop the entire document. Default
        implementation does nothing, so the entity is always accepted.
        """
        return None

    def get_unique(self) -> str:
        """Return a dedup key for this entity.

        Default: a fresh random token, which means two identical extractions
        are both kept. Override to return a deterministic key (e.g., a hash of
        chosen fields) when duplicates should be collapsed. The key is not
        included in the prompt; it is only used by the loop to detect repeats.
        """
        return secrets.token_hex(16)

    def template(self) -> str:
        """Return the embeddable string representation of this entity.

        Default: the JSON dump of the model. Override to produce a tighter
        natural-language rendering for better retrieval (e.g.,
        ``f"{self.name} ({self.year}): {self.context}"``).
        """
        return str(self.model_dump(mode="json"))

    def extend(self) -> list[type[BaseStructure] | type[BaseSemantic] | type[BaseCollection]]:
        """Return additional schemas to apply after this entity is extracted.

        Called once per *entity* (not once per collection). Returned classes
        must be declared in :attr:`Schema.extensions`. Default: no extension.
        """
        return []
