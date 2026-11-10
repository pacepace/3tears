"""a write by key: the rows a key names updated or deleted, in one statement, never a table's rows by a scan.

A collection's targeted write (a status flip, a stamp, a fence-checked save, a delete of one row or of a parent's
rows) names its rows by a leading part of a key, as a keyed read does::

    UPDATE <table> SET "c1" = $1, "c2" = now() WHERE "k1" = $2 AND "k2" = ANY($3::uuid[]) AND <where> RETURNING ...
    DELETE FROM <table> WHERE "k1" = $1 AND <where> RETURNING ...

``by`` is that key (a value, ``None`` for ``IS NULL``, or an :class:`~threetears.core.keyed.AnyOf`); ``where`` adds
the write's own guards (a status it must still hold, a fence value), each a fragment written as ``"col"`` from
``$1``; a value set may be an expression (a :class:`~threetears.core.keyed.Fragment`: ``now()``,
``LEAST(1.0, "salience" + $1)``). A write names a key: an empty ``by`` is refused, never a write of every row.

IO only: one statement on the caller's connection, awaited.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from threetears.core.keyed._filters import Fragment, by_conditions, quoted_table, renumbered
from threetears.core.keyed.counts import rows_affected
from threetears.core.sql_fragments import quote_identifier

__all__ = ["KeyedWrite", "delete_keyed", "update_keyed"]


@dataclass(frozen=True)
class KeyedWrite:
    """what a keyed write did.

    :param count: the rows it wrote (the statement's own status, or its returned rows)
    :ptype count: int
    :param rows: the rows it returned (``returning``), in no promised order; empty when it returned none
    :ptype rows: list[dict[str, Any]]
    """

    count: int
    rows: list[dict[str, Any]]


def _guarded(by: Mapping[str, Any], where: Sequence[Fragment], table: str, first: int) -> tuple[str, list[Any]]:
    """the ``WHERE`` naming a write's rows: ``by`` then ``where``, placeholders from ``first``.

    :raises ValueError: when ``by`` names no column
    """
    if not by:
        raise ValueError(f"a keyed write of {table} names the key of the rows it writes (by=)")
    conditions, params = by_conditions(by, first=first)
    for fragment in where:
        conditions.append(renumbered(fragment, first - 1 + len(params)))
        params += list(fragment.params)
    return " WHERE " + " AND ".join(conditions), params


def _returning(columns: Sequence[str]) -> str:
    return f" RETURNING {', '.join(quote_identifier(column) for column in columns)}" if columns else ""


async def _run(conn: Any, sql: str, params: Sequence[Any], returning: Sequence[str]) -> KeyedWrite:
    if returning:
        rows = [dict(row) for row in await conn.fetch(sql, *params)]
        return KeyedWrite(count=len(rows), rows=rows)
    status = await conn.execute(sql, *params)
    return KeyedWrite(count=rows_affected(str(status or "")), rows=[])


async def update_keyed(
    conn: Any,
    table: str,
    *,
    values: Mapping[str, Any],
    by: Mapping[str, Any],
    where: Sequence[Fragment] = (),
    returning: Sequence[str] = (),
) -> KeyedWrite:
    """set ``values`` on the rows of ``table`` that ``by`` (and ``where``) name.

    :param conn: anything with an asyncpg-shaped ``execute`` (and ``fetch`` when ``returning``)
    :ptype conn: Any
    :param table: the table (``schema.table``), a TRUSTED identifier
    :ptype table: str
    :param values: column -> its new value, or a :class:`~threetears.core.keyed.Fragment` expression (``now()``,
        ``"n" + $1``, its placeholders from ``$1``); columns are TRUSTED identifiers
    :ptype values: Mapping[str, Any]
    :param by: the key of the rows written, a leading part of the table's key (or of an index declared for the
        write), each a value, ``None`` (``IS NULL``) or an :class:`~threetears.core.keyed.AnyOf`
    :ptype by: Mapping[str, Any]
    :param where: guards beside the key, each a fragment written as ``"col"`` from ``$1`` (a status the row must
        hold, a fence)
    :ptype where: Sequence[Fragment]
    :param returning: the columns the written rows return; none when empty
    :ptype returning: Sequence[str]
    :return: the count written, and the rows returned
    :rtype: KeyedWrite
    :raises ValueError: when ``values`` or ``by`` is empty
    """
    if not values:
        raise ValueError(f"an update of {table} sets at least one column")
    assignments: list[str] = []
    params: list[Any] = []
    for column, value in values.items():
        if isinstance(value, Fragment):
            assignments.append(f"{quote_identifier(column)} = {renumbered(value, len(params))}")
            params += list(value.params)
        else:
            params.append(value)
            assignments.append(f"{quote_identifier(column)} = ${len(params)}")
    clause, guarded = _guarded(by, where, table, len(params) + 1)
    sql = f"UPDATE {quoted_table(table)} SET {', '.join(assignments)}{clause}{_returning(returning)}"
    return await _run(conn, sql, [*params, *guarded], returning)


async def delete_keyed(
    conn: Any,
    table: str,
    *,
    by: Mapping[str, Any],
    where: Sequence[Fragment] = (),
    returning: Sequence[str] = (),
) -> KeyedWrite:
    """delete the rows of ``table`` that ``by`` (and ``where``) name.

    :param conn: anything with an asyncpg-shaped ``execute`` (and ``fetch`` when ``returning``)
    :ptype conn: Any
    :param table: the table (``schema.table``), a TRUSTED identifier
    :ptype table: str
    :param by: the key of the rows deleted, as :func:`update_keyed` takes it
    :ptype by: Mapping[str, Any]
    :param where: guards beside the key, each a fragment written as ``"col"`` from ``$1``
    :ptype where: Sequence[Fragment]
    :param returning: the columns the deleted rows return; none when empty
    :ptype returning: Sequence[str]
    :return: the count deleted, and the rows returned
    :rtype: KeyedWrite
    :raises ValueError: when ``by`` is empty
    """
    clause, params = _guarded(by, where, table, 1)
    sql = f"DELETE FROM {quoted_table(table)}{clause}{_returning(returning)}"
    return await _run(conn, sql, params, returning)
