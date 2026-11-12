"""a migration's backfill: a column of a table's existing rows filled a page at a time by key, each page's writes
keyed updates, never one ``UPDATE`` of every row.

A schema change that adds or reshapes a column fills the rows written before it. One ``UPDATE ... WHERE <col> IS
NULL`` does that in a single statement over the whole table: it holds every row's lock to the end, scans the table
(a hashed key on YugabyteDB answers no range), and grows with the table. :func:`backfill_pages` reads the rows in key
order a page at a time (:func:`~threetears.core.keyset.keyset_pages`, walking a hashed lead on YugabyteDB), lets the
migration decide each row in Python (the same rule a read side applies to a row cached before the migration ran), and
writes each page's rows of one value by their keys (:func:`~threetears.core.keyed.update_keyed`). It lives with the
migration machinery: a schema change reads every row once, by design, and runs only from a migration.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from threetears.core.collections.l2_order import (
    L2_EPOCH_COLUMN,
    L2_ORDER_COLUMNS,
    L2_ORDER_FLOOR,
    L2_REVISION_COLUMN,
)
from threetears.core.keyed import KeyBatch, update_keyed
from threetears.core.keyset import keyset_pages
from threetears.core.sql_fragments import quote_identifier

__all__ = ["BACKFILL_PAGE", "backfill_l2_order", "backfill_pages"]


#: rows a backfill page reads
BACKFILL_PAGE: Final = 500


async def backfill_pages(
    conn: Any,
    table: str,
    key: Sequence[str],
    *,
    columns: Sequence[str],
    fill: Callable[[Mapping[str, Any]], Mapping[str, Any] | None],
    casts: Sequence[str] = (),
    lead_array: str | None = None,
    page_size: int = BACKFILL_PAGE,
) -> int:
    """fill a column of a table's existing rows a page at a time in key order, each page's writes keyed updates: a
    migration's backfill, never one ``UPDATE`` of every row.

    Each page reads ``key`` and ``columns`` (:func:`keyset_pages`); ``fill`` answers, for a row, the values it is to
    hold (a value, ``None``, or a :class:`~threetears.core.keyed.Fragment` over the row's own columns such as
    ``LEFT("content", 1000)``), or None when the row needs no write. The page's rows that answer the same values are
    written by one :func:`~threetears.core.keyed.update_keyed` naming their keys. A ``fill`` that answers None for a
    row already filled makes a replay write nothing.

    :param conn: anything with an asyncpg-shaped ``fetch`` and ``execute``, bound where ``table`` resolves
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier (``schema.table`` or bare)
    :ptype table: str
    :param key: its primary key, TRUSTED identifiers, in key order
    :ptype key: Sequence[str]
    :param columns: the columns ``fill`` reads beside the key, TRUSTED identifiers
    :ptype columns: Sequence[str]
    :param fill: a row -> the values it is to hold, or None to leave it
    :ptype fill: Callable[[Mapping[str, Any]], Mapping[str, Any] | None]
    :param casts: each key column's SQL type, as :class:`~threetears.core.keyed.KeyBatch` takes them; ``text`` for
        every column when empty
    :ptype casts: Sequence[str]
    :param lead_array: the key's hashed lead's array type, as :func:`keyset_pages` takes it
    :ptype lead_array: str | None
    :param page_size: rows a page reads
    :ptype page_size: int
    :return: the rows written
    :rtype: int
    :raises ValueError: when ``page_size`` is under one
    """
    key = tuple(key)
    read = ", ".join(quote_identifier(column) for column in dict.fromkeys((*key, *columns)))
    written = 0
    async for rows in keyset_pages(
        conn, f"SELECT {read} FROM {table}", key, page_size=page_size, lead_array=lead_array
    ):
        groups: list[tuple[Mapping[str, Any], list[tuple[Any, ...]]]] = []
        for row in rows:
            values = fill(row)
            if values is None:
                continue
            found = next((keys for held, keys in groups if held == values), None)
            if found is None:
                found = []
                groups.append((values, found))
            found.append(tuple(row[column] for column in key))
        for values, keys in groups:
            batch = KeyBatch(key, keys, casts=tuple(casts))
            written += (await update_keyed(conn, table, values=values, keys=batch)).count
    return written


def _floor_unless_ordered(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """:data:`L2_ORDER_FLOOR` for a row that carries no order, None for one that does."""
    if row[L2_EPOCH_COLUMN] is not None and row[L2_REVISION_COLUMN] is not None:
        return None
    return {L2_EPOCH_COLUMN: L2_ORDER_FLOOR.epoch, L2_REVISION_COLUMN: L2_ORDER_FLOOR.revision}


async def backfill_l2_order(
    conn: Any, table_name: str, key: Sequence[str], *, casts: Sequence[str] = (), lead_array: str | None = None
) -> int:
    """write :data:`L2_ORDER_FLOOR` into every row of ``table_name`` that carries no order, a page at a time in key
    order (:func:`backfill_pages`), so the first compare-and-swap after the migration
    supersedes them and every later one is fenced. A row that carries an order is not written, so a replay writes
    nothing.

    :param conn: anything with an asyncpg-shaped ``fetch`` and ``execute``
    :ptype conn: Any
    :param table_name: the table, unqualified or ``schema.table``, a TRUSTED identifier
    :ptype table_name: str
    :param key: its primary key, in key order
    :ptype key: Sequence[str]
    :param casts: each key column's SQL type (``text`` for every column when empty)
    :ptype casts: Sequence[str]
    :param lead_array: the key's hashed lead's array type, as :func:`~threetears.core.keyset.keyset_pages` takes it
    :ptype lead_array: str | None
    :return: the rows written
    :rtype: int
    """
    return await backfill_pages(
        conn, table_name, key, columns=L2_ORDER_COLUMNS, fill=_floor_unless_ordered, casts=casts, lead_array=lead_array
    )
