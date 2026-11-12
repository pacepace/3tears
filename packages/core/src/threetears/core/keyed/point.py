"""point reads: the rows one key names, bound as a hand-written ``WHERE a = $1 AND b = $2`` binds it.

The one spelling of the read a module used to write as text and send through ``fetch``, ``fetchrow`` or ``fetchval``::

    SELECT t."id", t."name" FROM "agents" AS t WHERE t."id" = $1 AND t."is_deleted" = $2

Each pinned column is compared with one parameter, ``= $n``, whose type the database infers from the column, exactly as
it inferred it for the statement written by hand, so a ``uuid`` key takes a ``UUID`` (or its text) and its index is read
by it; a ``None`` pins ``IS NULL``; an :class:`~threetears.core.keyed.AnyOf` pins a set. Where :func:`read_keyed` reads a
batch of keys, this reads one: the rows of one key, the first of them, or the first column of the first.

The read is keyed when its pinned columns cover a leading part of the table's key, or the leading column of an index
declared for the caller's read; the no-whole-reads rule judges each call by the statement it sends
(:mod:`threetears.enforcement.no_whole_reads.helpers`), under the caller's name. A read pinning nothing reads the table
whole, and is refused.

IO only: one statement, awaited on the caller's connection (anything with an asyncpg-shaped ``fetch``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, overload

from threetears.core.keyed._filters import Fragment, Lock, Selected, sent_by
from threetears.interfaces.keyed.page import Sort
from threetears.interfaces.keyed.point import read_by_statement, read_row_statement

__all__ = [
    "read_by",
    "read_row",
    "read_value",
]


@overload
async def read_by(
    conn: Any,
    table: str,
    *,
    by: Mapping[str, Any],
    columns: Sequence[str | Selected] | None = ...,
    records: Literal[False] = ...,
    lock: Lock | None = ...,
    where: Sequence[Fragment] = ...,
    order: Sequence[Sort] = ...,
) -> list[dict[str, Any]]: ...


@overload
async def read_by(
    conn: Any,
    table: str,
    *,
    by: Mapping[str, Any],
    columns: Sequence[str | Selected] | None = ...,
    records: Literal[True],
    lock: Lock | None = ...,
    where: Sequence[Fragment] = ...,
    order: Sequence[Sort] = ...,
) -> list[Any]: ...


async def read_by(
    conn: Any,
    table: str,
    *,
    by: Mapping[str, Any],
    columns: Sequence[str | Selected] | None = None,
    records: bool = False,
    lock: Lock | None = None,
    where: Sequence[Fragment] = (),
    order: Sequence[Sort] = (),
) -> list[dict[str, Any]] | list[Any]:
    """every row of ``table`` the pinned columns name: what ``fetch`` of the hand-written statement answered.

    :param conn: anything with an asyncpg-shaped ``fetch(query, *params)``
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param by: column -> its value (``= $n``, typed by the column), ``None`` (``IS NULL``) or an
        :class:`~threetears.core.keyed.AnyOf`; columns are TRUSTED identifiers; at least one
    :ptype by: Mapping[str, Any]
    :param columns: the columns read, TRUSTED identifiers or :class:`~threetears.core.keyed.Selected` expressions;
        every column of the table, in its order, when None
    :ptype columns: Sequence[str | Selected] | None
    :param records: answer the rows as the connection answers them (asyncpg ``Record``s on a direct connection), with
        no copy into a dict; dicts when False
    :ptype records: bool
    :param lock: the row lock taken on the rows read (``FOR UPDATE OF t NOWAIT``, say), in the caller's transaction;
        none when None
    :ptype lock: Lock | None
    :param where: guards beside the pins, each a fragment written as ``t."col"`` from ``$1``; none when empty
    :ptype where: Sequence[Fragment]
    :param order: the sort the rows come in, the columns after the pins in the key or index serving the read; none
        when empty
    :ptype order: Sequence[Sort]
    :return: the rows, in ``order``, else in no promised order
    :rtype: list[dict[str, Any]] | list[Any]
    :raises ValueError: when nothing is pinned, or ``columns`` is empty
    """
    sql, params = read_by_statement(table, by=by, columns=columns, lock=lock, where=where, order=order)
    rows = list(await sent_by(conn).fetch(sql, *params))
    return rows if records else [dict(row) for row in rows]


@overload
async def read_row(
    conn: Any,
    table: str,
    *,
    by: Mapping[str, Any],
    columns: Sequence[str | Selected] | None = ...,
    records: Literal[False] = ...,
    lock: Lock | None = ...,
    where: Sequence[Fragment] = ...,
) -> dict[str, Any] | None: ...


@overload
async def read_row(
    conn: Any,
    table: str,
    *,
    by: Mapping[str, Any],
    columns: Sequence[str | Selected] | None = ...,
    records: Literal[True],
    lock: Lock | None = ...,
    where: Sequence[Fragment] = ...,
) -> Any: ...


async def read_row(
    conn: Any,
    table: str,
    *,
    by: Mapping[str, Any],
    columns: Sequence[str | Selected] | None = None,
    records: bool = False,
    lock: Lock | None = None,
    where: Sequence[Fragment] = (),
) -> dict[str, Any] | Any | None:
    """the first row of ``table`` the pinned columns name, or None: what ``fetchrow`` of the hand-written statement
    answered (one row asked for, ``LIMIT 1``; which one, where the key names several, is the database's, as it was).

    :param conn: anything with an asyncpg-shaped ``fetch(query, *params)``
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param by: as :func:`read_by`'s
    :ptype by: Mapping[str, Any]
    :param columns: as :func:`read_by`'s
    :ptype columns: Sequence[str | Selected] | None
    :param records: as :func:`read_by`'s; the row is then typed as the connection's (``Any``), as ``fetchrow``'s was
    :ptype records: bool
    :param lock: the row lock taken on the rows read (``FOR UPDATE OF t NOWAIT``, say), in the caller's transaction;
        none when None
    :ptype lock: Lock | None
    :param where: guards beside the pins, each a fragment written as ``t."col"`` from ``$1``; none when empty
    :ptype where: Sequence[Fragment]
    :return: the row, or None when the key names none
    :rtype: dict[str, Any] | Any | None
    :raises ValueError: when nothing is pinned, or ``columns`` is empty
    """
    sql, params = read_row_statement(table, by=by, columns=columns, lock=lock, where=where)
    rows = list(await sent_by(conn).fetch(sql, *params))
    if not rows:
        return None
    return rows[0] if records else dict(rows[0])


async def read_value(conn: Any, table: str, *, by: Mapping[str, Any], column: str) -> Any:
    """one column of the first row of ``table`` the pinned columns name, or None: what ``fetchval`` of the
    hand-written statement answered (None both when the key names no row and when the column holds none).

    :param conn: anything with an asyncpg-shaped ``fetch(query, *params)``
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param by: as :func:`read_by`'s
    :ptype by: Mapping[str, Any]
    :param column: the column, a TRUSTED identifier
    :ptype column: str
    :return: its value, or None
    :rtype: Any
    :raises ValueError: when nothing is pinned
    """
    sql, params = read_row_statement(table, by=by, columns=(column,))
    rows = list(await sent_by(conn).fetch(sql, *params))
    return rows[0][column] if rows else None
