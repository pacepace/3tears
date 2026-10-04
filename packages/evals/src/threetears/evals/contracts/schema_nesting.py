"""Where one JSON Schema can hold another — the one answer every schema walker in the contracts reads.

Several readers walk a schema tree: the honoured-subset audit (``world_schema.honoured_kind``), the
registry's self-contradiction check (``world._schema_defects``) and the prose gate's path addressing
(``prose.schema_nodes_at``). Each once kept its own list of the positions a schema can nest at, and they
disagreed — one descended into ``items`` alone, another stopped at ``anyOf`` — so a defect or a prose field
written at a position one walker skipped was invisible to it. A keyword the honoured subset gains next
(``oneOf``, ``prefixItems``) is added here, once, and every walker sees it.

A leaf module, because both halves of the contracts reach it: ``prose`` sits beneath the host package that
``world_schema`` lives in, and importing upward from it would be a cycle.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Literal, NamedTuple

__all__ = ["NestedSchema", "NestingKeyword", "nested_schemas"]

#: The keywords under which a schema holds another schema.
NestingKeyword = Literal["items", "properties", "additionalProperties", "anyOf"]


class NestedSchema(NamedTuple):
    """One schema written inside another, and where it sits."""

    keyword: NestingKeyword
    """The keyword that holds it."""

    key: str | int | None
    """The property name under ``properties``, the branch index under ``anyOf``; None otherwise."""

    schema: Any
    """What is written there, as written. It may not be a schema at all: whether that is a refusal (the
    honoured-subset audit) or simply nothing to check (the self-contradiction check) is the reader's call."""

    def at(self, parent: str) -> str:
        """The path this nested schema sits at, given the path of the schema holding it.

        Args:
            parent: Where the holding schema sits — a dimension name or a seed path.

        Returns:
            ``parent.items``, ``parent.<property>``, ``parent.additionalProperties`` or ``parent.anyOf[i]``.
        """
        if self.keyword == "properties":
            return f"{parent}.{self.key}"
        if self.keyword == "anyOf":
            return f"{parent}.anyOf[{self.key}]"
        return f"{parent}.{self.keyword}"


def nested_schemas(schema: Mapping[str, Any]) -> Iterator[NestedSchema]:
    """Every schema written directly inside ``schema``, in declaration order.

    ``items``; each property; an ``additionalProperties`` that is not a boolean (a boolean opens or closes
    the object and holds no schema); each ``anyOf`` branch. A ``properties`` that is not a mapping, or an
    ``anyOf`` that is not a list, holds nothing a walker could descend into, so it yields nothing — a reader
    that must refuse such a shape checks for it itself.

    Args:
        schema: The holding schema.

    Yields:
        Each nested position. Not recursive: a walker descends by calling this on what it yields.
    """
    if "items" in schema:
        yield NestedSchema("items", None, schema["items"])
    if isinstance(properties := schema.get("properties"), Mapping):
        for name, nested in properties.items():
            yield NestedSchema("properties", name, nested)
    if "additionalProperties" in schema and not isinstance(extra := schema["additionalProperties"], bool):
        yield NestedSchema("additionalProperties", None, extra)
    if isinstance(branches := schema.get("anyOf"), list):
        for index, branch in enumerate(branches):
            yield NestedSchema("anyOf", index, branch)
