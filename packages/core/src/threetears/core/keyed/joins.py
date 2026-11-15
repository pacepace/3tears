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
from typing import Any

from threetears.core.keyed._filters import Fragment, Selected
from threetears.core.keyed._joining import Joined, cast_of, check_joins, join_rows, joined_columns
from threetears.core.keyed.page import keyset_page
from threetears.core.keyed.point import read_by
from threetears.core.keyed.rows import read_keyed
from threetears.interfaces.keyed.page import Sort
from threetears.interfaces.keyed.rows import KeyBatch

__all__ = [
    "Joined",
    "read_by_joined",
    "read_reached",
]


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
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """the rows of ``table`` the pins name, each beside the rows of each joined table it matches: what ``fetch`` of
    the hand-written join answered.

    With ``limit``, the read is the first page of its rows in ``order`` (:func:`~threetears.core.keyed.keyset_page`,
    a seek past the page before in ``where``): ``... LEFT JOIN ... ORDER BY ... LIMIT n``, each join a ``LEFT`` (or
    counted) one, so the page holds the rows the statement's did.

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
    :param limit: the most rows read, the first in ``order``; every row the pins name when None
    :ptype limit: int | None
    :return: the rows, as dicts: the read's columns, then each join's
    :rtype: list[dict[str, Any]]
    :raises ValueError: when nothing is joined, a join names nothing to join on, a counted join widens the row, or a
        page has no order or an inner join
    """
    check_joins(table, joins)
    if limit is not None and not order:
        raise ValueError(f"a page of a joined read of {table} names its order")
    if limit is not None and any(not (join.left or join.counts) for join in joins):
        raise ValueError(f"a page of a joined read of {table} joins LEFT only (an inner join drops rows past the page)")
    read, hidden = joined_columns(columns, joins)
    rows: list[dict[str, Any]] = (
        await read_by(conn, table, by=by, columns=read, where=where, order=order)
        if limit is None
        else (await keyset_page(conn, table, columns=read, by=by, order=order, limit=limit, where=where)).rows
    )
    return await join_rows(conn, rows, joins, batch=batch, hidden=hidden)


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
        cast = cast or cast_of(nodes)
        level = await read_keyed(
            conn, table, keys=KeyBatch((on,), [(node,) for node in nodes], (cast,)), columns=read, by=by, batch=batch
        )
    return list(answered.values())
