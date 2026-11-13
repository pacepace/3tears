"""The keys each declared table holds, by name: what a keyset pager's pins and order must hold to read every row.

A keyset page after the first starts past the last row read in the page's order. When the order (with the columns
the read pins to one value) holds no key of the table, every other row sharing that row's order values is never read:
a listing, a copy or a drain comes back short, silently. The no-whole-reads rule names such a call where its order is
written out (``non-unique-order``); an order built at run time it cannot see, so the runners refuse it as well
(:func:`refuse_non_unique_order`), against the keys the process's table declarations register here.

What registers: each ``TableSchema`` (``threetears.core.collections.schema_backed``) on construction, its primary
key, its ``UNIQUE`` constraints and its unique indexes. A partial unique index is a key of the rows its predicate
holds: one whose predicate says a column ``IS NOT NULL`` is a key where the read pins that column to a value. A key's
partition column (``Column(..., partition=True)``) places a row; it never tells two rows apart within the rest of the
key, so a read need not hold it. A partial unique index whose predicate says a column ``IS NULL`` is a key where the
read pins that column to NULL. Keys of one name from several declarations (one table in many schemas, a test's
own) are all kept: an order holding any of them reads every row.

The refusal never refuses what the declarations cannot prove: a table no declaration registers (a derived relation,
a warehouse's, one whose collection names it per instance) passes, and so does a unique index whose predicate is
anything but one ``IS NOT NULL`` or ``IS NULL``. The static rule stays the gate; this catches the order a run builds.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from threetears.interfaces.keyed.filters import AnyOf
from threetears.interfaces.keyed.page import Sort

__all__ = [
    "TableKey",
    "not_null_predicate",
    "null_predicate",
    "refuse_non_unique_order",
    "register_table_keys",
]


@dataclass(frozen=True)
class TableKey:
    """one key of a table: columns no two of its rows share every value of.

    :ivar columns: the key's columns, in order
    :ivar not_null: a partial unique index's predicate columns (``WHERE customer_id IS NOT NULL``): the key holds only
        among rows where each is a value, so a read must pin each to one
    :ivar placement: the key's partition columns: they place a row and never tell two apart, so a read need not hold
        them
    :ivar null: a partial unique index's predicate columns that are NULL (``WHERE customer_id IS NULL``): the key
        holds only among rows where each is NULL, so a read must pin each to NULL
    """

    columns: tuple[str, ...]
    not_null: tuple[str, ...] = ()
    placement: tuple[str, ...] = ()
    null: tuple[str, ...] = ()


#: every ``(bare table name, key)`` a declaration registered, once each: appended, never looked up and filled (a
#: list a reader may walk while another thread appends), so two declarations registering at once each add their keys
_KEYS: list[tuple[str, TableKey]] = []

#: a partial index's predicate that one column is not NULL
_NOT_NULL: Final[re.Pattern[str]] = re.compile(r'\s*\(?\s*"?(\w+)"?\s+IS\s+NOT\s+NULL\s*\)?\s*', re.IGNORECASE)

#: a partial index's predicate that one column is NULL
_NULL: Final[re.Pattern[str]] = re.compile(r'\s*\(?\s*"?(\w+)"?\s+IS\s+NULL\s*\)?\s*', re.IGNORECASE)


def _bare(table: str) -> str:
    """the table a relation names, as keys are registered: its last dotted part, unquoted, lower-cased."""
    return table.rsplit(".", 1)[-1].strip().strip('"').lower()


def not_null_predicate(where: str) -> str | None:
    """the column a partial index's predicate says is not NULL (``customer_id IS NOT NULL``); None for any other.

    :param where: the predicate as declared
    :ptype where: str
    :return: the column, or None
    :rtype: str | None
    """
    found = _NOT_NULL.fullmatch(where)
    return found.group(1) if found is not None else None


def null_predicate(where: str) -> str | None:
    """the column a partial index's predicate says is NULL (``customer_id IS NULL``); None for any other.

    :param where: the predicate as declared
    :ptype where: str
    :return: the column, or None
    :rtype: str | None
    """
    found = _NULL.fullmatch(where)
    return found.group(1) if found is not None else None


def register_table_keys(table: str, keys: Iterable[TableKey]) -> None:
    """record ``table``'s keys, beside any another declaration of its name registered.

    :param table: the table's name (``schema.table`` or bare)
    :ptype table: str
    :param keys: its keys
    :ptype keys: Iterable[TableKey]
    :return: nothing
    :rtype: None
    """
    bare = _bare(table)
    # a schema built again (``dataclasses.replace``, one per agent schema) registers nothing new: the list stays
    # one entry a key however often its declarations are built
    _KEYS.extend(pair for pair in {(bare, key) for key in keys} if pair not in _KEYS)


def _registered_keys(table: str) -> frozenset[TableKey]:
    """every key registered for ``table``'s name; empty when no declaration names it.

    :param table: the table's name (``schema.table`` or bare)
    :ptype table: str
    :return: the keys
    :rtype: frozenset[TableKey]
    """
    bare = _bare(table)
    return frozenset(key for name, key in _KEYS if name == bare)


def refuse_non_unique_order(table: Any, by: Mapping[Any, Any], order: Sequence[Sort]) -> None:
    """refuse a keyset read of ``table`` whose pins and order hold none of its registered keys.

    A column pinned to one value (or to NULL) holds it; one pinned to a set (:class:`AnyOf`) does not. A sort by an
    expression holds no column. A relation that is not a table's name (a derived one) and a table with no registered
    key pass.

    :param table: the relation read
    :ptype table: Any
    :param by: the read's pins
    :ptype by: Mapping[Any, Any]
    :param order: the read's sort
    :ptype order: Sequence[Sort]
    :return: nothing
    :rtype: None
    :raises ValueError: when the pins and order hold no registered key
    """
    if not isinstance(table, str):
        return
    keys = _registered_keys(table)
    if not keys:
        return
    valued = {column for column, value in by.items() if isinstance(column, str) and not isinstance(value, AnyOf)}
    nulls = {column for column, value in by.items() if isinstance(column, str) and value is None}
    valued -= nulls
    held = valued | nulls | {sort.column for sort in order if sort.expression is None}
    for key in keys:
        if (
            all(column in held or column in key.placement for column in key.columns)
            and all(column in valued for column in key.not_null)
            and all(column in nulls for column in key.null)
        ):
            return
    named = " | ".join("(" + ", ".join(key.columns) + ")" for key in sorted(keys, key=lambda k: k.columns))
    raise ValueError(
        f"a keyset read of {table} pinning ({', '.join(sorted(valued | nulls))}) in order "
        f"({', '.join(sort.column for sort in order)}) holds none of its keys {named}: every row sharing a page's "
        "last order values would be skipped; end the order with the key"
    )
