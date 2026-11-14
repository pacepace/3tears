"""The keys each declared L3 table holds: what a keyset pager's pins and order must hold to read every row.

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
read pins that column to NULL. Keys of one relation from several declarations (its ``TableSchema`` and its
migration's keys registered beside it) are all kept: an order holding any of them reads every row.

**A table is known by its identity, never its bare name.** Every declaration here is an L3 table's (PostgreSQL or
YugabyteDB, as its ``TableSchema`` or migration creates it), registered under the name it declares: ``schema.table``
when it names its schema, else the bare name the connection's search path finds. A read is judged only when it goes to
L3 (the pager says so: a datasource's reader, a warehouse's, an in-process DuckDB or SQLite never does) and names that
same relation: ``"platform_audit"."audit_events"`` is the declaration ``platform_audit.audit_events``, ``memories`` is
the bare declaration ``memories``. A warehouse relation that shares a bare name with an L3 table, the same name in
another schema, or a schema-qualified read of a table declared bare is another relation, unregistered here, and passes:
judging it by another table's keys could refuse a read that holds its own key (an outage) or pass one that does not.

The refusal never refuses what the declarations cannot prove: a relation no declaration registers (a derived one, a
warehouse's, one whose collection names it per instance) passes, and so does a unique index whose predicate is
anything but one ``IS NOT NULL`` or ``IS NULL``. The static rule stays the gate; this catches the order a run builds.

**Indexes, and the one a read names to YugabyteDB** (:func:`serving_index`, :func:`index_hint`).
YugabyteDB's planner costs a secondary index it would have to read the table through above the primary key, even when
the primary key holds only the read's tenant and the secondary index holds the tenant, the read's other pins and its
order. One agent's conversations of one status, newest first: the primary key ``(agent_id, conversation_id)`` filtering
the status read 13-17 rows for a page of 9, against 9 through ``(agent_id, status, conversation_id)``; it chose so on
every one of 15 fresh worlds, its estimate one row whatever the statistics said (``ANALYZE`` and
``yb_enable_optimizer_statistics`` on or off). So a keyed read names the index its declaration says serves it, as
pg_hint_plan reads a statement's first comment: ``/*+ IndexOnlyScan(t <index>) */``, an index-only scan where the index
holds every column read, else an index scan of that same index.

What registers: each ``TableSchema`` (``threetears.core.collections.schema_backed``) on construction, its primary key
and each plain B-tree index it declares (no predicate, no operator class, no collation, no other access method). A table
is known by its identity, as for its keys: ``schema.table`` when declared so, else the bare name the search path
finds.

Which index serves a read (:func:`serving_index`): one whose leading columns are each pinned to one value by the read
(``by``; a set, ``AnyOf``, is no single value), followed by the read's sort columns in its order (a sort column ``by``
pins is a constant, and orders nothing), every one in the sort's direction or every one against it (a backward scan).
Of those, the one holding the most pins; then the fewest columns; the primary key over a secondary index holding as
many, which needs no hint. No index at all for a relation no declaration registers, a sort by an expression, or a read
with nothing left to sort by.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from threetears.interfaces.keyed.filters import AnyOf
from threetears.interfaces.keyed.page import Sort

__all__ = [
    "DeclaredIndex",
    "TableKey",
    "index_hint",
    "not_null_predicate",
    "null_predicate",
    "refuse_non_unique_order",
    "register_table_indexes",
    "register_table_keys",
    "serving_index",
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

    def __post_init__(self) -> None:
        """refuse a key naming anything but columns: an expression index (``lower(name)``) is no key a read's pins
        or sort columns can hold, and taken as one it would stand for columns it does not name.

        :raises ValueError: when a name is not a plain column name, or the key names no column
        """
        if not self.columns:
            raise ValueError("a key names at least one column")
        for name in (*self.columns, *self.not_null, *self.placement, *self.null):
            if not isinstance(name, str) or _COLUMN.fullmatch(name) is None:
                raise ValueError(f"a key names columns only, not {name!r}: an expression index is no column key")


#: a plain column name, as a pager's pins and sort columns name one
_COLUMN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")


#: an L3 relation's identity: its schema as declared (None for a bare name, which the search path finds) and its
#: table, each as the database folds it (an unquoted name lower-cased, a quoted one as written)
Relation = tuple[str | None, str]

#: every ``(relation, key)`` a declaration registered, once each: appended, never looked up and filled (a list a
#: reader may walk while another thread appends), so two declarations registering at once each add their keys
_KEYS: list[tuple[Relation, TableKey]] = []

#: one SQL identifier: a quoted one (``""`` an escaped quote) or a plain one
_IDENTIFIER: Final = r'"(?:[^"]|"")+"|[A-Za-z_][A-Za-z0-9_$]*'

#: a relation's name: a table, or a schema and a table
_RELATION: Final[re.Pattern[str]] = re.compile(rf"\s*({_IDENTIFIER})(?:\s*\.\s*({_IDENTIFIER}))?\s*")

#: a partial index's predicate that one column is not NULL
_NOT_NULL: Final[re.Pattern[str]] = re.compile(r'\s*\(?\s*"?(\w+)"?\s+IS\s+NOT\s+NULL\s*\)?\s*', re.IGNORECASE)

#: a partial index's predicate that one column is NULL
_NULL: Final[re.Pattern[str]] = re.compile(r'\s*\(?\s*"?(\w+)"?\s+IS\s+NULL\s*\)?\s*', re.IGNORECASE)


def _folded(identifier: str) -> str:
    """an identifier as the database holds it: a quoted one as written (``""`` one quote), a plain one lower-cased."""
    if identifier.startswith('"'):
        return identifier[1:-1].replace('""', '"')
    return identifier.lower()


def _relation(table: str) -> Relation | None:
    """the relation ``table`` names (``schema.table`` or a bare table, either part quoted or not); None for any other
    text (a subquery, a three-part name, an alias), which names no declared table.

    :param table: the name
    :ptype table: str
    :return: its schema (None when bare) and table, folded, or None
    :rtype: Relation | None
    """
    found = _RELATION.fullmatch(table)
    if found is None:
        return None
    first, second = found.group(1), found.group(2)
    return (None, _folded(first)) if second is None else (_folded(first), _folded(second))


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
    """record the keys of the L3 table ``table`` names, beside any another declaration of that same relation
    registered.

    :param table: the table's name as declared: ``schema.table``, or bare for the one the search path finds
    :ptype table: str
    :param keys: its keys
    :ptype keys: Iterable[TableKey]
    :return: nothing
    :rtype: None
    :raises ValueError: when ``table`` is no relation's name
    """
    relation = _relation(table)
    if relation is None:
        raise ValueError(f"{table!r} names no table (schema.table, or a bare table): its keys cannot be registered")
    # a schema built again (``dataclasses.replace``) registers nothing new: the list stays one entry a key however
    # often its declarations are built
    _KEYS.extend(pair for pair in {(relation, key) for key in keys} if pair not in _KEYS)


def _registered_keys(table: str) -> frozenset[TableKey]:
    """every key registered for the L3 relation ``table`` names; empty when no declaration names that relation.

    :param table: the relation's name as read (``schema.table`` or bare)
    :ptype table: str
    :return: the keys
    :rtype: frozenset[TableKey]
    """
    relation = _relation(table)
    if relation is None:
        return frozenset()
    return frozenset(key for named, key in _KEYS if named == relation)


def refuse_non_unique_order(table: Any, by: Mapping[Any, Any], order: Sequence[Sort], *, l3: bool) -> None:
    """refuse a keyset read of the L3 table ``table`` whose pins and order hold none of its registered keys.

    A column pinned to one value (or to NULL) holds it; one pinned to a set (:class:`AnyOf`) does not. A sort by an
    expression holds no column. A read that does not go to L3, a relation that is not a table's name (a derived one),
    and a relation no declaration registers under that same identity pass.

    :param table: the relation read
    :ptype table: Any
    :param by: the read's pins
    :ptype by: Mapping[Any, Any]
    :param order: the read's sort
    :ptype order: Sequence[Sort]
    :param l3: whether the read goes to L3 (PostgreSQL or YugabyteDB, where the declarations' tables are); a
        datasource's reader (a warehouse), an in-process DuckDB or SQLite does not
    :ptype l3: bool
    :return: nothing
    :rtype: None
    :raises ValueError: when the pins and order hold no registered key
    """
    if not l3 or not isinstance(table, str):
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


@dataclass(frozen=True)
class DeclaredIndex:
    """one B-tree index a table declares, as a read's pins and order can be served by it.

    :ivar name: the index's name, a plain SQL identifier (the hint names it unquoted)
    :ivar columns: its columns, in order
    :ivar descending: the columns it sorts descending
    :ivar primary: whether it is the table's primary key (the planner's own choice; never hinted)
    """

    name: str
    columns: tuple[str, ...]
    descending: frozenset[str] = frozenset()
    primary: bool = False

    def __post_init__(self) -> None:
        """refuse an index naming anything but plain columns, or a name a hint could not carry unquoted.

        :raises ValueError: for such an index
        """
        if not self.columns:
            raise ValueError(f"index {self.name!r} names at least one column")
        for name in (self.name, *self.columns, *self.descending):
            if not isinstance(name, str) or _COLUMN.fullmatch(name) is None:
                raise ValueError(f"a declared index names plain identifiers only, not {name!r}")
        stray = self.descending - set(self.columns)
        if stray:
            raise ValueError(f"index {self.name!r}: descending names {sorted(stray)}, not among its columns")


#: every ``(relation, index)`` a declaration registered, once each, appended only (as the keys are)
_INDEXES: list[tuple[Relation, DeclaredIndex]] = []


def register_table_indexes(table: str, indexes: Iterable[DeclaredIndex]) -> None:
    """record the B-tree indexes of the L3 table ``table`` names, beside any another declaration of that same
    relation registered.

    :param table: the table's name as declared: ``schema.table``, or bare for the one the search path finds
    :ptype table: str
    :param indexes: its indexes
    :ptype indexes: Iterable[DeclaredIndex]
    :return: nothing
    :rtype: None
    :raises ValueError: when ``table`` is no relation's name
    """
    relation = _relation(table)
    if relation is None:
        raise ValueError(f"{table!r} names no table (schema.table, or a bare table): its indexes cannot be registered")
    _INDEXES.extend(pair for pair in {(relation, index) for index in indexes} if pair not in _INDEXES)


def _rank(index: DeclaredIndex, pinned: set[str], sorts: Sequence[Sort]) -> tuple[int, int, bool] | None:
    """how well ``index`` serves a read pinning ``pinned`` sorted by ``sorts`` (more pins held, then fewer columns,
    then the primary key); None when it does not serve it."""
    held = 0
    while held < len(index.columns) and index.columns[held] in pinned:
        held += 1
    if held == 0 or index.columns[held : held + len(sorts)] != tuple(sort.column for sort in sorts):
        return None
    against = {(column in index.descending) != sort.descending for column, sort in zip(index.columns[held:], sorts)}
    if len(against) > 1:
        return None
    return held, -len(index.columns), index.primary


def serving_index(table: Any, by: Mapping[Any, Any], order: Sequence[Sort]) -> DeclaredIndex | None:
    """the declared index that serves a keyed read of ``table`` pinning ``by``, in ``order``; None when none does.

    :param table: the relation read (a derived relation, or a name no declaration registers, has none)
    :ptype table: Any
    :param by: the read's pins (a column to a value or None; an ``AnyOf`` pins no single value)
    :ptype by: Mapping[Any, Any]
    :param order: the read's sort
    :ptype order: Sequence[Sort]
    :return: the index, or None
    :rtype: DeclaredIndex | None
    """
    relation = _relation(table) if isinstance(table, str) else None
    if relation is None:
        return None
    pinned = {column for column, value in by.items() if isinstance(column, str) and not isinstance(value, AnyOf)}
    sorts = [sort for sort in order if sort.expression is not None or sort.column not in pinned]
    if not sorts or any(sort.expression is not None for sort in sorts):
        return None
    ranked = [
        (rank, index)
        for named, index in _INDEXES
        if named == relation and (rank := _rank(index, pinned, sorts)) is not None
    ]
    if not ranked:
        return None
    return max(ranked, key=lambda pair: pair[0])[1]


def index_hint(index: DeclaredIndex | None, alias: str = "t") -> str:
    """the hint naming ``index`` as the read of ``alias``, as a statement's first words; nothing for no index or the
    primary key (the planner's own choice).

    ``IndexOnlyScan``: pg_hint_plan reads the index alone where it holds every column read, else scans that same index
    and reads the table by it. Every leg of a page (each ``... FROM <table> AS t`` of its ``WITH``) takes it.

    :param index: the index, or None
    :ptype index: DeclaredIndex | None
    :param alias: the alias the statement reads the table by
    :ptype alias: str
    :return: the hint and a space, or the empty string
    :rtype: str
    """
    if index is None or index.primary:
        return ""
    return f"/*+ IndexOnlyScan({alias} {index.name}) */ "
