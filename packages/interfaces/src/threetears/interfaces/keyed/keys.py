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
and each plain B-tree index it declares (no operator class, no collation, no other access method; no predicate, or one
of nothing but ``column = 'text'`` conjuncts, which serves only a read pinning each to that text). A table
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
    "register_partial_predicates",
    "not_null_guards",
    "partial_predicate",
    "primary_key",
    "stated_literals",
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
    :ivar stated: a partial index's predicate as the ``(column, text)`` pairs it names (``WHERE status = 'proposed'``,
        :func:`partial_predicate`): it serves only a read pinning each column to exactly that text; empty for a whole
        index
    :ivar guarded: the columns a partial index's predicate holds ``IS NOT NULL`` (``WHERE alias IS NOT NULL``,
        :func:`partial_predicate`): it serves only a read whose own guards state exactly that of each; empty for none
    :ivar unique: whether no two rows share every value of its columns (a unique index or ``UNIQUE`` constraint): a
        read binding each of its columns, a set of values allowed (``AnyOf``), reads one row a value
    """

    name: str
    columns: tuple[str, ...]
    descending: frozenset[str] = frozenset()
    primary: bool = False
    stated: frozenset[tuple[str, str]] = frozenset()
    guarded: frozenset[str] = frozenset()
    unique: bool = False

    def __post_init__(self) -> None:
        """refuse an index naming anything but plain columns, or a name a hint could not carry unquoted.

        :raises ValueError: for such an index
        """
        if not self.columns:
            raise ValueError(f"index {self.name!r} names at least one column")
        for name in (self.name, *self.columns, *self.descending, *(column for column, _ in self.stated), *self.guarded):
            if not isinstance(name, str) or _COLUMN.fullmatch(name) is None:
                raise ValueError(f"a declared index names plain identifiers only, not {name!r}")
        stray = self.descending - set(self.columns)
        if stray:
            raise ValueError(f"index {self.name!r}: descending names {sorted(stray)}, not among its columns")


#: every ``(relation, column, text)`` a declared partial index's predicate names as ``column = 'text'``, once each,
#: appended only (as the keys are): a keyed read pinning that column to that text writes it as the literal
#: (:func:`stated_literals`, ``by_conditions(stated=)``)
_STATED: list[tuple[Relation, str, str]] = []

#: one conjunct of a partial index's predicate naming a column equal to a text literal: ``status = 'proposed'``
_STATED_EQUALITY: Final[re.Pattern[str]] = re.compile(r"\s*\(?\s*\"?(\w+)\"?\s*=\s*'((?:[^']|'')*)'\s*\)?\s*")


#: a read's guard that one column (of its alias, or bare) is not NULL: the column captured
_NOT_NULL_GUARD: Final[re.Pattern[str]] = re.compile(
    r"\s*\(?\s*(?:\w+\.)?\"?(\w+)\"?\s+IS\s+NOT\s+NULL\s*\)?\s*", re.IGNORECASE
)


def partial_predicate(where: str) -> tuple[frozenset[tuple[str, str]], frozenset[str]] | None:
    """a partial index's predicate as the ``(column, text)`` pairs it names and the columns it holds ``IS NOT NULL``,
    when it is nothing but ``column = 'text'`` and ``column IS NOT NULL`` conjuncts; None for any other predicate,
    which no keyed read is judged to state.

    WHY ``IS NOT NULL`` (D7, coordinator ruling 2026-11-15): left out, ``ix_memories_user_alias (agent_id, user_id,
    alias) WHERE alias IS NOT NULL`` was never named to YugabyteDB, so a user merge's alias-collision lock (pinning the
    agent and the user, guarded ``t.alias IS NOT NULL``) was hinted to the user's every memory by date. A read states
    the guard in its own ``where=`` (:func:`not_null_guards`), never by its pins, so the index serves only a read
    carrying exactly that guard; nothing measured argued against it (it was left out as unstateable, e124fe266).

    :param where: the predicate, as declared
    :ptype where: str
    :return: the pairs and the guarded columns, or None
    :rtype: tuple[frozenset[tuple[str, str]], frozenset[str]] | None
    """
    pairs: set[tuple[str, str]] = set()
    guarded: set[str] = set()
    for conjunct in re.split(r"\s+AND\s+", where, flags=re.IGNORECASE):
        equal = _STATED_EQUALITY.fullmatch(conjunct)
        held = None if equal is not None else _NOT_NULL.fullmatch(conjunct)
        if equal is not None:
            pairs.add((equal.group(1).lower(), equal.group(2).replace("''", "'")))
        elif held is not None:
            guarded.add(held.group(1).lower())
        else:
            return None
    return (frozenset(pairs), frozenset(guarded)) if pairs or guarded else None


def not_null_guards(guards: Iterable[str]) -> frozenset[str]:
    """the columns a read's guards hold ``IS NOT NULL``, each a guard of nothing else (``t.alias IS NOT NULL``): what
    the read states of a ``column IS NOT NULL`` partial index (:func:`partial_predicate`).

    :param guards: the read's guards, as SQL
    :ptype guards: Iterable[str]
    :return: the columns
    :rtype: frozenset[str]
    """
    return frozenset(
        found.group(1).lower() for guard in guards if (found := _NOT_NULL_GUARD.fullmatch(guard)) is not None
    )


def register_partial_predicates(table: str, predicates: Iterable[str]) -> None:
    """record each ``column = 'text'`` a partial index of the L3 table ``table`` names in its predicate (each conjunct
    of an ``AND``), beside any another declaration of that same relation registered. A predicate of any other shape
    names none; nothing is written as a literal for it.

    WHY (coordinator ruling 2026-11-14, as Pace's boolean ruling): ``status = $3`` cannot prove a partial index's
    ``WHERE status = 'proposed'`` under PostgreSQL's generic plan (a prepared statement's after five runs), so the
    planner passes over the index the read pins; written ``status = 'proposed'`` it states the predicate every time.
    Pace extended the ruling 2026-11-16 to any fixed text from our own code, never a caller's or an end user's
    (:class:`~threetears.interfaces.keyed.filters.Const`): a bound discriminator also kept the generic plan from
    being cached at all (the why and numbers at ``Const``).

    :param table: the table's name as declared: ``schema.table``, or bare for the one the search path finds
    :ptype table: str
    :param predicates: its partial indexes' predicates, as declared
    :ptype predicates: Iterable[str]
    :return: nothing
    :rtype: None
    :raises ValueError: when ``table`` is no relation's name
    """
    relation = _relation(table)
    if relation is None:
        raise ValueError(
            f"{table!r} names no table (schema.table, or a bare table): its predicates cannot be registered"
        )
    for predicate in predicates:
        for conjunct in re.split(r"\s+AND\s+", predicate, flags=re.IGNORECASE):
            found = _STATED_EQUALITY.fullmatch(conjunct)
            if found is None:
                continue
            stated = (relation, found.group(1).lower(), found.group(2).replace("''", "'"))
            if stated not in _STATED:
                _STATED.append(stated)


def _states(index: DeclaredIndex, by: Mapping[Any, Any], guarded: frozenset[str] = frozenset()) -> bool:
    """whether a read pinning ``by`` states ``index``'s partial predicate: each of its columns pinned to exactly its
    text (a whole index's empty predicate is stated by every read).

    WHY: a partial index holds only the rows its predicate admits; named to YugabyteDB for a read that does not state
    it, the read would miss every other row. Stated, the index holds every row asked for (identity's pending queue:
    ``(agent_id, user_id, date_created) WHERE status = 'proposed'``; unhinted, YugabyteDB read 400 rows and
    sorted them for a page of 10, 3tears #98).
    """
    return index.guarded <= guarded and all(
        isinstance(by.get(column), str) and by[column] == text for column, text in index.stated
    )


def stated_literals(table: Any) -> frozenset[tuple[str, str]]:
    """the ``(column, text)`` pairs the partial indexes declared for the relation ``table`` names state; none for a
    relation no declaration registers (a derived one, a warehouse's), which writes every pin as ``$n``.

    :param table: the relation a read names
    :ptype table: Any
    :return: the pairs
    :rtype: frozenset[tuple[str, str]]
    """
    relation = _relation(table) if isinstance(table, str) else None
    if relation is None:
        return frozenset()
    return frozenset((column, text) for named, column, text in list(_STATED) if named == relation)


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


def primary_key(table: Any) -> tuple[str, ...] | None:
    """the primary key columns of the table ``table`` names, as its declaration registered them; None for a derived
    relation, a name no declaration registers, or one registering no primary key.

    :param table: the relation read
    :ptype table: Any
    :return: the key's columns, in order, or None
    :rtype: tuple[str, ...] | None
    """
    relation = _relation(table) if isinstance(table, str) else None
    if relation is None:
        return None
    return next((index.columns for named, index in _INDEXES if named == relation and index.primary), None)


def _rank(index: DeclaredIndex, pinned: set[str], sorts: Sequence[Sort]) -> tuple[int, int, bool] | None:
    """how well ``index`` serves a read pinning ``pinned`` sorted by ``sorts`` (more pins held, then fewer columns,
    then the primary key); None when it does not serve it."""
    held = _leading_held(index, pinned)
    if held == 0 or index.columns[held : held + len(sorts)] != tuple(sort.column for sort in sorts):
        return None
    against = {(column in index.descending) != sort.descending for column, sort in zip(index.columns[held:], sorts)}
    if len(against) > 1:
        return None
    return held, -len(index.columns), index.primary


def _leading_held(index: DeclaredIndex, pinned: set[str]) -> int:
    """how many of ``index``'s leading columns ``pinned`` binds."""
    held = 0
    while held < len(index.columns) and index.columns[held] in pinned:
        held += 1
    return held


def _unordered_serving(
    relation: Relation,
    pinned: set[str],
    by: Mapping[Any, Any],
    guarded: frozenset[str] = frozenset(),
    sets: frozenset[str] = frozenset(),
) -> DeclaredIndex | None:
    """the declared index that serves an unordered read of ``relation`` pinning ``pinned``: a secondary index binding
    more of the pins than the primary key (the one binding most, then the one binding more leading columns, then the
    narrower); else the primary key where it binds any (always when it binds all its columns: a point read); else
    None. A secondary index binds its pinned leading columns and, when it binds at least one, each column its partial
    predicate names that the read pins to exactly the predicate's text.

    YugabyteDB plans such a read through the primary key's prefix and filters the rest (3tears #95: 100 rows scanned
    for 20); the coordinator's ruling (#96) names the index that binds more of the pins.

    WHY a stated predicate column counts as bound (coordinator ruling, 3tears #100): ``idx_wake_schedules_conv_active``
    ``(conversation_id) WHERE status = 'active'`` binds one leading column, as the primary key ``(conversation_id,
    schedule_id)`` does, so it was never named; YugabyteDB counted a conversation's 10 active schedules through the
    primary key, reading all 400 of its rows, on 6 of 6 fresh ANALYZEs (S/o99/probe/wakecount-yb.log). The index holds
    only the rows its predicate admits, so a read pinning the predicate's exact text is narrowed by it as by a key
    column. :func:`_states` has already refused an index whose text the read does not pin exactly (another text, a
    set, none): naming it there would miss every other row. At equal binding the index binding more leading columns
    wins (it narrows inside the index too; the earlier contract, test_keyed_index_hint's
    ``o88_conv_agent_user_status``)."""
    indexes = [index for named, index in _INDEXES if named == relation and _states(index, by, guarded)]
    primary = next((index for index in indexes if index.primary), None)
    floor = _leading_held(primary, pinned) if primary is not None else 0
    if primary is not None and floor == len(primary.columns):
        return primary
    # WHY a set binds a unique index (D8 via the coordinator, 2026-11-15): an ``AnyOf`` pins no single value, so it
    # bound nothing here, and datasources' schema priming (``name = ANY(...)`` and ``status = 'active'``) was hinted on
    # YugabyteDB to idx_datasources_status, every active datasource read and filtered, rather than the unique name
    # index, one row a name. A unique index whose every column is pinned or bound to a set reads one row a value: it
    # serves the read before any index binding more pins one value each (the narrowest such index)
    keyed = [index for index in indexes if index.unique and not index.primary and set(index.columns) <= pinned | sets]
    if keyed:
        return min(keyed, key=lambda index: len(index.columns))
    ranked = [
        ((held + len(index.stated) + len(index.guarded), held, -len(index.columns)), index)
        for index in indexes
        if not index.primary
        and (held := _leading_held(index, pinned)) > 0
        and held + len(index.stated) + len(index.guarded) > floor
    ]
    if ranked:
        return max(ranked, key=lambda pair: pair[0])[1]
    return primary if floor else None


def serving_index(
    table: Any, by: Mapping[Any, Any], order: Sequence[Sort], *, guarded: frozenset[str] = frozenset()
) -> DeclaredIndex | None:
    """the declared index that serves a keyed read of ``table`` pinning ``by``, in ``order``; None when none does.

    An unordered read (no sort, or one sorting only pinned columns) is served by the index binding most of its pins
    (:func:`_unordered_serving`); so is a read sorting by an expression, which no plain index holds, unless only the
    primary key binds its pins (None: the planner's own choice).

    :param table: the relation read (a derived relation, or a name no declaration registers, has none)
    :ptype table: Any
    :param by: the read's pins (a column to a value or None; an ``AnyOf`` pins no single value)
    :ptype by: Mapping[Any, Any]
    :param order: the read's sort
    :ptype order: Sequence[Sort]
    :param guarded: the columns the read's own guards hold ``IS NOT NULL`` (:func:`not_null_guards`), which a
        ``column IS NOT NULL`` partial index needs stated
    :ptype guarded: frozenset[str]
    :return: the index, or None
    :rtype: DeclaredIndex | None
    """
    relation = _relation(table) if isinstance(table, str) else None
    if relation is None:
        return None
    pinned = {column for column, value in by.items() if isinstance(column, str) and not isinstance(value, AnyOf)}
    sorts = [sort for sort in order if sort.expression is not None or sort.column not in pinned]
    sets = frozenset(column for column, value in by.items() if isinstance(column, str) and isinstance(value, AnyOf))
    if not sorts:
        return _unordered_serving(relation, pinned, by, guarded, sets)
    if any(sort.expression is not None for sort in sorts):
        # WHY an order by an expression names the index binding most pins (D5 via the coordinator, 2026-11-15): no
        # plain B-tree index holds such an order, so the read sorts whatever rows it reads and the fewest is the best
        # it can do: the unordered read's choice. Unnamed, YugabyteDB read a user's enabled skills (recency first,
        # COALESCE(last_used_at, '-infinity')) through the user's prefix of uq_skills_agent_user_name, all 2,000 of
        # them, rather than the 200 enabled ones of idx_skills_agent_user_enabled, which binds all three pins
        # (skills' test_skills_agent_user_enabled on YugabyteDB: 2,000 rows scanned before, 200 after).
        # WHY not an order by plain columns that no declared index holds: an index the declarations do not name may
        # hold it, and naming a declared one overrides it. The hub's provider model pages (gateway_models by
        # provider, by status / type / dates) read v160's (provider_id, <sort>, id) indexes, which its TableSchema
        # does not declare; named its declared idx_gw_models_provider_model instead, YugabyteDB read 333 and 1,021
        # rows for a page of 21 (YB1, 2026-11-16). The primary key stays unnamed (the planner's own choice).
        fallback = _unordered_serving(relation, pinned, by, guarded, sets)
        return None if fallback is None or fallback.primary else fallback
    ranked = [
        (rank, index)
        for named, index in _INDEXES
        if named == relation and _states(index, by, guarded) and (rank := _rank(index, pinned, sorts)) is not None
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
