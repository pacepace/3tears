"""the joins of a keyed read, applied in Python: each joined table read by the values the read's rows hold (one keyed
batch, :func:`~threetears.core.keyed.read_joined`), each row matched to the joined rows holding its values and
answered once for each, as the hand-written join answered it.

Shared by the reads that join (:func:`~threetears.core.keyed.read_by_joined`, :func:`~threetears.core.keyed.read_nearest`),
so a join is spelled one way whichever read it follows.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Any
from uuid import UUID

from threetears.core.ddl.row_epoch import GONE_COLUMN
from threetears.core.keyed._filters import Selected
from threetears.core.keyed.rows import read_joined
from threetears.interfaces.keyed.filters import by_conditions, quote_identifier, quoted_table

__all__ = ["Joined", "cast_of", "check_joins", "join_rows", "joined_columns", "lateral_joins"]


@dataclass(frozen=True)
class Joined:
    """one table a :func:`read_by_joined` read joins: ``JOIN <table> ON <table>.<col> = <row's column> ...``.

    :param table: the joined table, a TRUSTED identifier
    :ptype table: str
    :param on: its columns -> the row's columns each equals (a column the read reads, or a name an earlier join adds),
        a leading part of its key (or of an index declared for the caller's read)
    :ptype on: Mapping[str, str]
    :param columns: the name each of its columns lands under in the row -> that column, TRUSTED identifiers
    :ptype columns: Mapping[str, str]
    :param left: keep a row no joined row matches (``LEFT JOIN``), each joined column None
    :ptype left: bool
    :param by: its columns pinned beside the join (``ON n.row_scope = 'customer'``), each a value or an
        :class:`~threetears.core.keyed.AnyOf`
    :ptype by: Mapping[str, Any] | None
    :param matched: the name a row says under whether a joined row matched it (``(n.id IS NOT NULL) AS live`` over a
        ``LEFT JOIN`` on ``n.id``); none when None
    :ptype matched: str | None
    :param casts: each ``on`` column's SQL type; read from the row's value when absent (a ``UUID`` is ``uuid``, an
        ``int`` ``bigint``, a ``str`` ``text``)
    :ptype casts: Mapping[str, str]
    :param counts: a counted join's: each name the row says a count under -> the joined column a matching row is
        counted by when it is not NULL (``COUNT(dc.description)``), or None to count every matching row
        (``COUNT(*)``); every read row is then kept once, with no joined column (``columns`` empty)
    :ptype counts: Mapping[str, str | None]
    """

    table: str
    on: Mapping[str, str]
    columns: Mapping[str, str]
    left: bool = False
    by: Mapping[str, Any] | None = None
    matched: str | None = None
    casts: Mapping[str, str] = field(default_factory=dict)
    counts: Mapping[str, str | None] = field(default_factory=dict)
    #: what a statement joining this table is written from (every field, in order, each pin's value beside its
    #: type), computed once: None when a pin's value is not hashable (an AnyOf of a list)
    shape: Hashable | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """hold each mapping as a read-only copy and compute :attr:`shape` once.

        WHY (2026-11-26): a walk's held statement is found by its call's shape
        (``threetears.core.keyed.joins._statement_key``), and rebuilding the joins' part from their mappings every call
        cost ~3 us of the visible-customer set's ~24 us of Python (its joins a module constant, rebuilt each call;
        test_visible_customers_speed_live). The mappings are copied read-only so the shape computed here stays true.
        """
        for name in ("on", "columns", "casts", "counts"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        if self.by is not None:
            object.__setattr__(self, "by", MappingProxyType(dict(self.by)))
        shape: Hashable | None = (
            self.table,
            tuple(self.on.items()),
            tuple(self.columns.items()),
            self.left,
            None if self.by is None else tuple((type(c), c, type(v), v) for c, v in self.by.items()),
            self.matched,
            tuple(self.casts.items()),
            tuple(self.counts.items()),
        )
        try:
            hash(shape)
        except TypeError:
            # NOSILENT: an unhashable pin (an AnyOf of a list) only means a statement joining this table is not held
            # by its shape: it is written for its call (joins._statement_key answers None)
            shape = None
        object.__setattr__(self, "shape", shape)


#: a Python value's SQL type, for an ``on`` column given no cast
_CASTS: tuple[tuple[type, str], ...] = (
    (bool, "boolean"),
    (int, "bigint"),
    (UUID, "uuid"),
    (str, "text"),
    (Decimal, "numeric"),
    (datetime, "timestamptz"),
    (date, "date"),
)


def cast_of(values: Sequence[Any]) -> str:
    """the SQL type a list of key values binds as, read from its first non-None value (``text`` when every value is
    None or no listed type matches): the cast a join's ``unnest($n::<type>[])`` of the values takes.

    :param values: the values
    :ptype values: Sequence[Any]
    :return: the SQL type's name
    :rtype: str
    """
    held = next((value for value in values if value is not None), None)
    return next((cast for kind, cast in _CASTS if isinstance(held, kind)), "text")


def _names(columns: Sequence[str | Selected] | None) -> set[str] | None:
    return None if columns is None else {column.alias if isinstance(column, Selected) else column for column in columns}


def check_joins(table: str, joins: Sequence[Joined]) -> None:
    """refuse joins no read can apply: none, one naming nothing to join on, a counted one that widens the row.

    :param table: the table read, as a refusal names it
    :ptype table: str
    :param joins: the joins
    :ptype joins: Sequence[Joined]
    :return: nothing
    :rtype: None
    :raises ValueError: as above
    """
    if not joins:
        raise ValueError(f"a joined read of {table} joins at least one table (read_by reads one)")
    for join in joins:
        if not join.on:
            raise ValueError(f"a join of {join.table} names how it joins the row")
        if join.counts and (join.columns or join.left or join.matched):
            raise ValueError(f"a join of {join.table} that counts keeps each row once: no columns, left or matched")


def joined_columns(
    columns: Sequence[str | Selected] | None, joins: Sequence[Joined]
) -> tuple[list[str | Selected] | None, list[str]]:
    """the columns a joined read reads, and those of them it drops after joining: each column a join is matched on
    that the read does not select (and no join names) is read, then dropped from the rows.

    :param columns: the read's own columns; every one when None
    :ptype columns: Sequence[str | Selected] | None
    :param joins: the joins
    :ptype joins: Sequence[Joined]
    :return: the columns read (None: every one), and the ones dropped after
    :rtype: tuple[list[str | Selected] | None, list[str]]
    """
    selected = _names(columns)
    added = [] if selected is None else sorted({column for join in joins for column in join.on.values()} - selected)
    named: set[str] = set(selected or ()) | {
        name for join in joins for name in (*join.columns, *join.counts, join.matched) if name
    }
    hidden = [column for column in added if column not in named]
    return (None if columns is None else [*columns, *hidden]), hidden


async def join_rows(
    conn: Any,
    rows: list[dict[str, Any]],
    joins: Sequence[Joined],
    *,
    batch: int | None = None,
    hidden: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """``rows`` joined to each table of ``joins`` in order, as the hand-written joins answered them: a row no joined
    row matches dropped (``JOIN``) or kept once with each joined column None (``LEFT JOIN``); a row matching several
    answered once for each, in the read's order; a counted join keeps each row once beside its counts. A NULL value
    matches nothing, as ``=`` matches nothing.

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param rows: the read's rows, as dicts
    :ptype rows: list[dict[str, Any]]
    :param joins: the tables joined, in order; each joins the row as the joins before it left it
    :ptype joins: Sequence[Joined]
    :param batch: the most values a joined table's statement names; None: the connection's
    :ptype batch: int | None
    :param hidden: the read's columns dropped from the rows once joined (:func:`joined_columns`)
    :ptype hidden: Sequence[str]
    :return: the joined rows
    :rtype: list[dict[str, Any]]
    """
    for join in joins:
        inner = list(join.on)
        # the joined table's columns read under names of their own, so none overwrites the row's before it is placed
        counted = [column for column in join.counts.values() if column is not None]
        wanted = list(dict.fromkeys([*inner, *join.columns.values(), *counted]))
        casts = {column: join.casts.get(column) or cast_of([row[join.on[column]] for row in rows]) for column in inner}
        matches = (
            await read_joined(
                conn, rows, join.table, on=dict(join.on), columns=wanted, casts=casts, by=join.by, batch=batch
            )
            if rows
            else {}
        )
        joined: list[dict[str, Any]] = []
        for row in rows:
            value = tuple(row[join.on[column]] for column in inner)
            found = matches.get(value) if all(part is not None for part in value) else None
            if join.counts:
                held = found or []
                joined.append(
                    {
                        **row,
                        **{
                            name: sum(1 for other in held if column is None or other[column] is not None)
                            for name, column in join.counts.items()
                        },
                    }
                )
            elif found:
                for other in found:
                    widened = {**row, **{name: other[column] for name, column in join.columns.items()}}
                    if join.matched:
                        widened[join.matched] = True
                    joined.append(widened)
            elif join.left:
                widened = {**row, **dict.fromkeys(join.columns)}
                if join.matched:
                    widened[join.matched] = False
                joined.append(widened)
        rows = joined
    if hidden:
        for row in rows:
            for column in hidden:
                del row[column]
    return rows


def lateral_joins(
    joins: Sequence[Joined],
    binds: list[Any],
    *,
    source: str = "r",
    fenced: bool = False,
    guarded: frozenset[str] = frozenset(),
) -> tuple[list[str], str]:
    """each join as a ``LATERAL`` lookup of the rows of its table the row ``source`` names (``j1."group_id" =
    r."group_id"`` and the join's own pins, a later join's ``on`` naming an earlier one's column by the name it lands
    under), and the select-list items its columns land under: the one spelling of a join inside one statement, driven
    from the read's few rows by key, never a join the planner may hash from the joined side (as a plain join over a
    small derived set, YugabyteDB hashed every row of the joined table). A ``left`` join keeps a row none matches
    (``LEFT JOIN LATERAL ... ON TRUE``); any other drops it (``CROSS JOIN LATERAL``).

    :param joins: the joins, in order; none counted or matched
    :ptype joins: Sequence[Joined]
    :param binds: the statement's binds so far; each join's pins are appended
    :ptype binds: list[Any]
    :param source: the alias of the row the joins look up from
    :ptype source: str
    :param fenced: end each lookup in ``OFFSET 0``, so the planner keeps it a lookup a row and never pulls it up into
        a join it may hash from the joined side (a source whose row count it cannot estimate, a recursive walk's)
    :ptype fenced: bool
    :param guarded: the joined tables read as stamped L3 tables (``_filters.stamped_names``): each lookup of one
        answers its live rows alone (``NOT j."gone"``), so a join never brings back a row a delete marked gone
    :ptype guarded: frozenset[str]
    :return: the select-list items, and the joins' text (each after a space)
    :rtype: tuple[list[str], str]
    :raises ValueError: when a join counts or says whether it matched (a lookup answers neither)
    """
    qualifier: dict[str, str] = {}
    selected: list[str] = []
    laterals: list[str] = []
    for number, join in enumerate(joins, start=1):
        if join.counts or join.matched:
            raise ValueError(f"a join of {join.table} inside one statement neither counts nor says it matched")
        alias = f"j{number}"
        conditions = [
            f"j.{quote_identifier(column)} = {qualifier.get(row, f'{source}.' + quote_identifier(row))}"
            for column, row in join.on.items()
        ]
        pinned, values = by_conditions(join.by or {}, first=len(binds) + 1, alias="j")
        binds += values
        if join.table in guarded:
            pinned = [*pinned, f"NOT j.{quote_identifier(GONE_COLUMN)}"]
        wanted = ", ".join(quote_identifier(column) for column in dict.fromkeys(join.columns.values()))
        kind = "LEFT JOIN LATERAL" if join.left else "CROSS JOIN LATERAL"
        tail = " ON TRUE" if join.left else ""
        fence = " OFFSET 0" if fenced else ""
        laterals.append(
            f" {kind} (SELECT {wanted} FROM {quoted_table(join.table)} AS j"
            f" WHERE {' AND '.join([*conditions, *pinned])}{fence}) AS {alias}{tail}"
        )
        for name, column in join.columns.items():
            selected.append(f"{alias}.{quote_identifier(column)} AS {quote_identifier(name)}")
            qualifier[name] = f"{alias}.{quote_identifier(column)}"
    return selected, "".join(laterals)
