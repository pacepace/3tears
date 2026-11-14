"""a point read with joins, as keyed reads: the rows a ``WHERE`` names, then each joined table by the values they hold.

The one spelling of what a module used to write as one joined statement::

    SELECT dt.*, d.name AS datasource_name FROM datasource_tables dt
      JOIN datasources d ON d.id = dt.datasource_id
     WHERE dt.id = $1

is :func:`read_by_joined`: the pinned table's rows by its pins (:func:`~threetears.core.keyed.read_by`), then each
joined table by the values those rows hold (:func:`~threetears.core.keyed.read_joined`, one keyed batch), never a join
the planner may drive from the joined side. Each read row is matched to the joined rows holding its values, in Python,
and answered once for each, as the join answers it: a row no joined row matches is dropped (``JOIN``) or kept once
with every joined column None (``LEFT JOIN``); a NULL value matches nothing, as ``=`` matches nothing. The rows come in
the read's ``order``; the rows one read row matches, in no promised order, as the join left them.

A correlated count in the select list (``(SELECT COUNT(*) FROM datasource_columns dc WHERE dc.table_id = dt.id AND
dc.description IS NOT NULL) AS described``) is a counted join (:class:`Joined` with ``counts``): the joined table read
by the same keyed batch, and each read row kept once beside how many of its rows match it (0 when none does).

The no-whole-reads rule judges a call by the statements it sends (the read by its pins, each joined table by the
columns it is joined on), each keyed only as the hand-written join's tables were.

IO only: one statement for the read, one keyed batch a joined table.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from threetears.core.keyed._filters import Fragment, Selected
from threetears.core.keyed.point import read_by
from threetears.core.keyed.rows import read_joined, read_keyed
from threetears.interfaces.keyed.page import Sort
from threetears.interfaces.keyed.rows import KeyBatch

__all__ = [
    "Joined",
    "read_by_joined",
    "read_reached",
]


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


def _cast_of(values: Sequence[Any]) -> str:
    held = next((value for value in values if value is not None), None)
    return next((cast for kind, cast in _CASTS if isinstance(held, kind)), "text")


def _names(columns: Sequence[str | Selected] | None) -> set[str] | None:
    return None if columns is None else {column.alias if isinstance(column, Selected) else column for column in columns}


async def read_by_joined(
    conn: Any,
    table: str,
    *,
    by: Mapping[str, Any],
    columns: Sequence[str | Selected] | None,
    joins: Sequence[Joined],
    where: Sequence[Fragment] = (),
    order: Sequence[Sort] = (),
    batch: int | None = None,
) -> list[dict[str, Any]]:
    """the rows of ``table`` the pins name, each beside the rows of each joined table it matches: what ``fetch`` of
    the hand-written join answered.

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param table: the table pinned, a TRUSTED identifier
    :ptype table: str
    :param by: its pins, as :func:`~threetears.core.keyed.read_by` takes them; at least one
    :ptype by: Mapping[str, Any]
    :param columns: its columns in the row, as :func:`~threetears.core.keyed.read_by` takes them; every one when None
    :ptype columns: Sequence[str | Selected] | None
    :param joins: the tables joined, in order; each joins the row as the joins before it left it
    :ptype joins: Sequence[Joined]
    :param where: guards beside the pins, as :func:`~threetears.core.keyed.read_by` takes them
    :ptype where: Sequence[Fragment]
    :param order: the read's order, its own columns; none promised when empty
    :ptype order: Sequence[Sort]
    :param batch: the most values a joined table's statement names; None: the connection's
    :ptype batch: int | None
    :return: the rows, as dicts: the read's columns, then each join's
    :rtype: list[dict[str, Any]]
    :raises ValueError: when nothing is joined, a join names nothing to join on, or a counted join widens the row
    """
    if not joins:
        raise ValueError(f"a joined read of {table} joins at least one table (read_by reads one)")
    for join in joins:
        if not join.on:
            raise ValueError(f"a join of {join.table} names how it joins the row")
        if join.counts and (join.columns or join.left or join.matched):
            raise ValueError(f"a join of {join.table} that counts keeps each row once: no columns, left or matched")
    selected = _names(columns)
    # the columns of the read a join needs and the read does not select: read, then dropped from the rows
    added = [] if selected is None else sorted({column for join in joins for column in join.on.values()} - selected)
    named: set[str] = set(selected or ()) | {
        name for join in joins for name in (*join.columns, *join.counts, join.matched) if name
    }
    hidden = [column for column in added if column not in named]
    read = None if columns is None else [*columns, *hidden]
    rows: list[dict[str, Any]] = await read_by(conn, table, by=by, columns=read, where=where, order=order)
    for join in joins:
        inner = list(join.on)
        # the joined table's columns read under names of their own, so none overwrites the row's before it is placed
        counted = [column for column in join.counts.values() if column is not None]
        wanted = list(dict.fromkeys([*inner, *join.columns.values(), *counted]))
        casts = {column: join.casts.get(column) or _cast_of([row[join.on[column]] for row in rows]) for column in inner}
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


async def read_reached(
    conn: Any,
    table: str,
    *,
    start: Mapping[str, Any],
    on: str,
    to: str,
    columns: Sequence[str],
    by: Mapping[str, Any] | None = None,
    follow: Mapping[str, Any] | None = None,
    batch: int | None = None,
) -> list[dict[str, Any]]:
    """every row a walk of ``table``'s links reaches, each once: what a recursive statement walking them answered
    (``WITH RECURSIVE reached(node, path) AS (... UNION ALL SELECT s.<to>, ... JOIN reached r ON s.<on> = r.node
    WHERE ... AND NOT (s.<to> = ANY(r.path))) SELECT DISTINCT ...``), as a keyed read a level.

    The rows ``start`` pins are read first (:func:`~threetears.core.keyed.read_by`); each row names a node in ``to``
    (one whose ``follow`` columns hold their values; every row when None), and the rows whose ``on`` holds a node not
    walked yet (within ``by``) are read next, one keyed batch a level, until a level names no new node. A cycle ends
    where it comes back. Each read is keyed as ``on`` leads an index of the table (``start``'s own pins likewise).

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param start: the first rows' pins, as :func:`~threetears.core.keyed.read_by` takes them; when it pins ``on``,
        that value is the walk's first node
    :ptype start: Mapping[str, Any]
    :param on: the column a row is reached by (it holds a node), a TRUSTED identifier
    :ptype on: str
    :param to: the column naming the node a row leads to, a TRUSTED identifier
    :ptype to: str
    :param columns: the columns answered, TRUSTED identifiers; the rows answered distinct over them
    :ptype columns: Sequence[str]
    :param by: columns pinned on every row read after the first (``source_kind = 'dataset'``)
    :ptype by: Mapping[str, Any] | None
    :param follow: the values a row's columns hold for its ``to`` to be walked; every row's when None
    :ptype follow: Mapping[str, Any] | None
    :param batch: the most nodes a statement names; None: the connection's
    :ptype batch: int | None
    :return: the rows reached, each its ``columns``, distinct, level by level
    :rtype: list[dict[str, Any]]
    :raises ValueError: when ``start`` pins nothing or no column is answered
    """
    if not start:
        raise ValueError(f"a walk of {table} starts at the rows its pins name")
    if not columns:
        raise ValueError(f"a walk of {table} answers at least one column")
    followed = dict(follow or {})
    read = list(dict.fromkeys([on, to, *followed, *columns]))
    walked: set[Any] = {start[on]} if on in start and start[on] is not None else set()
    answered: dict[tuple[Any, ...], dict[str, Any]] = {}
    level: list[dict[str, Any]] = await read_by(conn, table, by=start, columns=read)
    cast: str | None = None
    while level:
        nodes: list[Any] = []
        for row in level:
            answered.setdefault(tuple(row[column] for column in columns), {column: row[column] for column in columns})
            node = row[to]
            if node is None or node in walked or any(row[column] != value for column, value in followed.items()):
                continue
            walked.add(node)
            nodes.append(node)
        if not nodes:
            break
        cast = cast or _cast_of(nodes)
        level = await read_keyed(
            conn, table, keys=KeyBatch((on,), [(node,) for node in nodes], (cast,)), columns=read, by=by, batch=batch
        )
    return list(answered.values())
