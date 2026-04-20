"""Root-detection rule for ``ennoia_schema`` and the CLI fallback."""

from __future__ import annotations

from typing import ClassVar

from ennoia import BaseCollection, BaseSemantic, BaseStructure
from ennoia.schema.roots import identify_roots


def test_flat_list_without_extensions_every_class_is_root() -> None:
    class A(BaseStructure):
        """A."""

        x: str

    class B(BaseSemantic):
        """B?"""

    class C(BaseCollection):
        """C."""

        y: str

    assert identify_roots([A, B, C]) == [A, B, C]


def test_single_parent_with_extensions_is_sole_root() -> None:
    class Product(BaseStructure):
        """Product fields."""

        name: str

    class Summary(BaseSemantic):
        """Summary?"""

    class Page(BaseStructure):
        """Classify page."""

        kind: str

        class Schema:
            extensions: ClassVar[list[type]] = [Product, Summary]

    assert identify_roots([Product, Summary, Page]) == [Page]


def test_multiple_parents_both_are_roots() -> None:
    class Leaf1(BaseSemantic):
        """Leaf?"""

    class Leaf2(BaseSemantic):
        """Leaf?"""

    class ParentA(BaseStructure):
        """A."""

        x: str

        class Schema:
            extensions: ClassVar[list[type]] = [Leaf1]

    class ParentB(BaseStructure):
        """B."""

        y: str

        class Schema:
            extensions: ClassVar[list[type]] = [Leaf2]

    assert identify_roots([Leaf1, ParentA, Leaf2, ParentB]) == [ParentA, ParentB]


def test_roots_preserve_declaration_order() -> None:
    class L1(BaseSemantic):
        """L?"""

    class Parent(BaseStructure):
        """P."""

        x: str

        class Schema:
            extensions: ClassVar[list[type]] = [L1]

    class L2(BaseSemantic):
        """L?"""

    class OtherParent(BaseStructure):
        """OP."""

        y: str

        class Schema:
            extensions: ClassVar[list[type]] = [L2]

    # Caller passes classes in arbitrary order; the helper keeps that order.
    assert identify_roots([OtherParent, L1, Parent, L2]) == [OtherParent, Parent]


def test_empty_extensions_list_counts_as_no_extensions() -> None:
    # A class with an empty extensions list has no children it pulls in, so
    # it does not self-select as a DAG parent — every class stays a root.
    class A(BaseStructure):
        """A."""

        x: str

        class Schema:
            extensions: ClassVar[list[type]] = []

    class B(BaseSemantic):
        """B?"""

    assert identify_roots([A, B]) == [A, B]


def test_empty_input_returns_empty_list() -> None:
    assert identify_roots([]) == []
