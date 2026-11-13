"""where a statement's key columns come from: a declared key source, never a name that says "key".

A statement written over a table its code names at run time (``SELECT * FROM {table} WHERE {pk} = $1``) is keyed when
the column it interpolates is the table's key. The walk cannot see the table, so it asks where the column came from.
It is the key only when it flows from a DECLARED key source:

- a parameter, a class attribute (a dataclass field, ``self.x: KeyColumns = ...``, a property) or a function's return
  annotated :data:`~threetears.interfaces.keyed.KeyColumn` / :data:`~threetears.interfaces.keyed.KeyColumns` /
  :data:`~threetears.interfaces.keyed.KeyValues`, or ``Annotated[..., DeclaredKey]``
  (``def fetch(self, table: str, pk: KeyColumn)``, ``primary_key_columns: KeyColumns``,
  ``def _pk_columns(...) -> KeyColumns``);
- one of 3tears' own (:data:`THREETEARS_KEY_SOURCES`), which a consumer repo reads without walking 3tears;
- a local the function binds exactly once, never changing it after, to one of these (or to a copy of one:
  ``tuple(x)``, ``list(x)``, ``sorted(x)``).

The name the column is held under says nothing: ``pk_cols`` holding the table's other columns is a predicate the walk
cannot see, and ``names`` bound from ``self._pk_columns(...)`` is the key. A rename never changes a verdict.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

__all__ = [
    "THREETEARS_KEY_SOURCES",
    "KeySources",
    "annotated_key",
    "declared_key_sources",
]

#: the annotations that declare a key source
_KEY_TYPES: Final[frozenset[str]] = frozenset({"KeyColumn", "KeyColumns", "KeyValues", "DeclaredKey"})

#: 3tears' declared key sources a consumer repo reads (``self.primary_key_columns`` of a collection, a backend's
#: ``primary_key`` attribute): each is annotated with the key annotations (``KeyColumn``, ``KeyColumns``, ``KeyValues``, ``DeclaredKey``) where 3tears declares it, which
#: ``tests/enforcement/test_key_sources.py`` holds, so the list cannot outlive a rename. Attributes and functions
#: alike, by the name read (``x.<name>``, ``x.<name>(...)``).
THREETEARS_KEY_SOURCES: Final[frozenset[str]] = frozenset(
    {
        "primary_key_columns",
        "primary_key_column",
        "_pk_columns",
        "pk_columns",
        "conflict_columns",
    }
)


def annotated_key(annotation: ast.expr | None) -> bool:
    """whether an annotation declares a key source: the key annotations (``KeyColumn``, ``KeyColumns``, ``KeyValues``, ``DeclaredKey``) anywhere in it (``KeyColumn``,
    ``KeyColumn | KeyColumns``, ``KeyColumns | None``, ``"KeyColumns"``).

    :param annotation: the annotation
    :ptype annotation: ast.expr | None
    :return: whether it does
    :rtype: bool
    """
    if annotation is None:
        return False
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name) and node.id in _KEY_TYPES:
            return True
        if isinstance(node, ast.Attribute) and node.attr in _KEY_TYPES:
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                parsed = ast.parse(node.value, mode="eval").body
            except SyntaxError:
                continue  # NOSILENT: a string annotation that is not an expression declares nothing
            if annotated_key(parsed):
                return True
    return False


@dataclass(frozen=True)
class KeySources:
    """the repo's declared key sources, by the name they are read under.

    :ivar names: attributes and functions annotated as key sources (``x.<name>``, ``x.<name>(...)``, ``<name>(...)``)
    """

    names: frozenset[str] = THREETEARS_KEY_SOURCES


def declared_key_sources(trees: Iterable[ast.Module], *, base: frozenset[str] = THREETEARS_KEY_SOURCES) -> KeySources:
    """every key source the repo's modules declare, beside 3tears' own: class-body annotations
    (``primary_key: KeyColumns``), ``self.x: KeyColumns = ...`` in a method, and functions (properties too) whose
    return is annotated as one.

    :param trees: the repo's modules
    :ptype trees: Iterable[ast.Module]
    :param base: the sources taken as declared without walking them (3tears' own, for a consumer repo)
    :ptype base: frozenset[str]
    :return: the sources
    :rtype: KeySources
    """
    names: set[str] = set(base)
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and annotated_key(node.returns):
                names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                for statement in node.body:
                    if (
                        isinstance(statement, ast.AnnAssign)
                        and isinstance(statement.target, ast.Name)
                        and annotated_key(statement.annotation)
                    ):
                        names.add(statement.target.id)
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Attribute)
                and isinstance(node.target.value, ast.Name)
                and node.target.value.id == "self"
                and annotated_key(node.annotation)
            ):
                names.add(node.target.attr)
    return KeySources(frozenset(names))
