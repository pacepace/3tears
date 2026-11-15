"""the joins of a keyed read, applied in Python: each joined table read by the values the read's rows hold (one keyed
batch, :func:`~threetears.core.keyed.read_joined`), each row matched to the joined rows holding its values and
answered once for each, as the hand-written join answered it.

Shared by the reads that join (:func:`~threetears.core.keyed.read_by_joined`, :func:`~threetears.core.keyed.read_nearest`),
so a join is spelled one way whichever read it follows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from threetears.core.keyed._filters import Selected
from threetears.core.keyed.rows import read_joined

__all__ = ["Joined", "cast_of", "check_joins", "join_rows", "joined_columns"]


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
