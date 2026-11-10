"""the rows past a bound deleted (or updated) a page at a time, through the index on the bound's column: expiry sweeps and caps.

A sweep of expired rows (leases, tokens, dedup entries) deletes the rows whose ``expires_at`` passed. Written as one
``DELETE ... WHERE expires_at < $1`` it reads every row of the table; written as a page it reads only the page::

    DELETE FROM <table> WHERE ("k1", "k2") IN (
        SELECT "k1", "k2" FROM <table> WHERE <by> AND "expires_at" < $n ORDER BY "expires_at" LIMIT <page>)

and the sweep repeats until a page deletes fewer rows than it asked for, which the statement's own status says
(:func:`~threetears.core.keyed.rows_affected`), never a count. The index it reads is ``(<by columns>, <column>)``,
declared with the sweep as its ``purpose``.

IO only: one statement a page on the caller's connection, awaited in turn.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from threetears.core.keyed._filters import IN_SUBQUERY, batched_join_hint, by_conditions, quoted_table
from threetears.core.keyed.counts import rows_affected
from threetears.core.keyed.threaded import ThreadedDuckDB, ThreadedSQLite
from threetears.core.sql_fragments import quote_identifier

if TYPE_CHECKING:
    # the annotation only: the collections package imports these helpers
    from threetears.core.collections.base import BaseCollection

__all__ = ["SweptRows", "delete_before", "delete_oldest", "update_before"]


@dataclass(frozen=True)
class SweptRows:
    """what a paged sweep deleted (or a paged update touched).

    :param deleted: the rows deleted, or updated
    :ptype deleted: int
    :param statements: the DELETEs sent, one a page
    :ptype statements: int
    """

    deleted: int
    statements: int


async def delete_before(
    table: str,
    *,
    collection: BaseCollection[Any] | None,
    key: Sequence[str],
    column: str,
    before: Any,
    page: int,
    by: Mapping[str, Any] | None = None,
    most_pages: int | None = None,
    conn: Any = None,
) -> SweptRows:
    """delete the rows of ``table`` (within ``by``) whose ``column`` is before ``before``, ``page`` rows a statement.

    Each page is one statement inside ``collection``'s
    :meth:`~threetears.core.collections.base.BaseCollection.bypassing_write`, its deleted keys named by ``RETURNING``,
    so every cache of the rows it deleted is settled (evicted and broadcast, or enrolled in the caller's transaction).

    :param table: the table, a TRUSTED identifier (``schema.table`` quoted a segment at a time)
    :ptype table: str
    :param collection: the collection serving ``table``, whose caches the deleted rows are settled in; None, said so,
        for a table no collection or cache serves (a store's own table on its pool), read on ``conn``
    :ptype collection: BaseCollection | None
    :param key: its primary key's columns, TRUSTED identifiers: the rows a page deletes are named by them
    :ptype key: Sequence[str]
    :param column: the column compared with ``before`` (``expires_at``), led by ``by`` in its index
    :ptype column: str
    :param before: the bound: rows whose ``column`` is less are deleted
    :ptype before: Any
    :param page: the most rows a statement deletes
    :ptype page: int
    :param by: columns pinned ahead of ``column`` (a purpose, a tenant)
    :ptype by: Mapping[str, Any] | None
    :param most_pages: the most statements this call sends; every page until a short one when None
    :ptype most_pages: int | None
    :param conn: the caller's transaction (a ``CallerTransaction``'s connection), or the pool of a table no collection
        serves; the collection's pool when None
    :ptype conn: Any
    :return: the rows deleted and the statements sent
    :rtype: SweptRows
    :raises ValueError: when ``key`` is empty or ``page`` / ``most_pages`` is under one
    :raises TypeError: when neither a collection nor a connection is given
    """
    if not key:
        raise ValueError(f"a sweep of {table} names its key")
    if page < 1 or (most_pages is not None and most_pages < 1):
        raise ValueError(f"a sweep of {table} deletes at least one row in at least one page")
    conditions, params = by_conditions(by or {})
    conditions.append(f"{quote_identifier(column)} < ${len(params) + 1}")
    params.append(before)
    return await _paged(collection, conn, _Paged(table, key, column, conditions, page), params, most_pages)


@dataclass(frozen=True)
class _Paged:
    """one paged write: the rows of ``table`` matching ``conditions``, ``page`` of them a statement in ``column``'s
    order, deleted, or set to ``assignments`` when there are any."""

    table: str
    key: Sequence[str]
    column: str
    conditions: Sequence[str]
    page: int
    assignments: Sequence[str] = ()


async def _paged(
    collection: BaseCollection[Any] | None, conn: Any, write: _Paged, params: Sequence[Any], most_pages: int | None
) -> SweptRows:
    """run a paged write until a page touches fewer rows than it asked for, each page inside the collection's
    ``bypassing_write`` so the keys it returned are settled in every cache."""
    named = ", ".join(quote_identifier(part) for part in write.key)
    written = ", ".join(f"t.{quote_identifier(part)}" for part in write.key)
    table = quoted_table(write.table)
    # the page's keys joined back to the table by its key, batched on YugabyteDB (batched_join_hint): one storage
    # request a page, not one a row
    hint = batched_join_hint(IN_SUBQUERY, "t")
    rows_of = (
        f"({written}) IN (SELECT {named} FROM {table} WHERE {' AND '.join(write.conditions)} "
        f"ORDER BY {quote_identifier(write.column)} LIMIT {write.page}) RETURNING {written}"
    )
    if write.assignments:
        sql = f"{hint}UPDATE {table} AS t SET {', '.join(write.assignments)} WHERE {rows_of}"
    else:
        sql = f"{hint}DELETE FROM {table} AS t WHERE {rows_of}"
    if conn is not None:
        reader = conn
    elif collection is not None:
        reader = collection.required_l3_pool
    else:
        raise TypeError(f"a paged write of {write.table} names its collection, or (with collection=None) its conn")
    touched = statements = 0
    while most_pages is None or statements < most_pages:
        if collection is None:
            rows = await _unserved_page(reader, sql, params)
        else:
            async with collection.bypassing_write(conn=conn) as settled:
                rows = await reader.fetch(sql, *params)
                settled.touches(*(_entity_id(row, write.key) for row in rows))
        statements += 1
        touched += len(rows)
        if len(rows) < write.page:
            break
    return SweptRows(deleted=touched, statements=statements)


async def _unserved_page(reader: Any, sql: str, params: Sequence[Any]) -> list[Any]:
    """one page of a paged write on a table no collection or cache serves (``collection=None``, said so by the
    caller): nothing to settle.

    :param reader: the caller's connection
    :ptype reader: Any
    :param sql: the page's statement
    :ptype sql: str
    :param params: its parameters
    :ptype params: Sequence[Any]
    :return: the keys it returned
    :rtype: list[Any]
    """
    return list(await reader.fetch(sql, *params))


def _entity_id(row: Mapping[str, Any], key: Sequence[str]) -> Any:
    """a row's key as a collection names it: the value of a one-column key, a tuple of a composite one."""
    return row[key[0]] if len(key) == 1 else tuple(row[column] for column in key)


async def update_before(
    table: str,
    *,
    collection: BaseCollection[Any] | None,
    key: Sequence[str],
    column: str,
    before: Any,
    set_to: Mapping[str, Any],
    page: int,
    by: Mapping[str, Any] | None = None,
    most_pages: int | None = None,
    conn: Any = None,
) -> SweptRows:
    """set ``set_to`` on the rows of ``table`` (within ``by``) whose ``column`` is before ``before``, a page at a time.

    The reap of stale rows (fires stuck ``dispatching``): ``UPDATE <table> SET ... WHERE (<key>) IN (SELECT <key> FROM
    <table> WHERE <by> AND <column> < $n ORDER BY <column> LIMIT <page>)``, repeated until a page touches fewer rows
    than it asked for. Each update must move its rows out of the pages it reads (``set_to`` names a column of ``by``,
    or ``column`` itself), or the next page would read them again: that is refused.

    :param collection: the collection serving ``table``: each page runs in its ``bypassing_write``, the keys it
        updated named by ``RETURNING``, so every cache of them is settled; None, said so, for a table no collection or
        cache serves, read on ``conn``
    :ptype collection: BaseCollection | None
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param key: its primary key's columns, TRUSTED identifiers
    :ptype key: Sequence[str]
    :param column: the column compared with ``before``, led by ``by`` in its index
    :ptype column: str
    :param before: the bound
    :ptype before: Any
    :param set_to: column -> its new value, each bound
    :ptype set_to: Mapping[str, Any]
    :param page: the most rows a statement updates
    :ptype page: int
    :param by: columns pinned ahead of ``column`` (a status, a tenant)
    :ptype by: Mapping[str, Any] | None
    :param most_pages: the most statements this call sends; every page until a short one when None
    :ptype most_pages: int | None
    :param conn: the caller's transaction (a ``CallerTransaction``'s connection), or the pool of a table no collection
        serves; the collection's pool when None
    :ptype conn: Any
    :return: the rows updated and the statements sent
    :rtype: SweptRows
    :raises ValueError: when ``key`` or ``set_to`` is empty, a bound is under one, or the update leaves its rows in
        the pages it reads
    :raises TypeError: when neither a collection nor a connection is given
    """
    pinned = dict(by or {})
    if not key or not set_to:
        raise ValueError(f"a paged update of {table} names its key and what it sets")
    if page < 1 or (most_pages is not None and most_pages < 1):
        raise ValueError(f"a paged update of {table} touches at least one row in at least one page")
    if not set(set_to) & (set(pinned) | {column}):
        raise ValueError(
            f"a paged update of {table} must move its rows out of the pages it reads: set a column of by, or {column}"
        )
    assignments = []
    params: list[Any] = []
    for name, value in set_to.items():
        params.append(value)
        assignments.append(f"{quote_identifier(name)} = ${len(params)}")
    conditions, values = by_conditions(pinned, first=len(params) + 1)
    params += values
    conditions.append(f"{quote_identifier(column)} < ${len(params) + 1}")
    params.append(before)
    return await _paged(
        collection, conn, _Paged(table, key, column, conditions, page, tuple(assignments)), params, most_pages
    )


async def delete_oldest(
    conn: Any,
    table: str,
    *,
    count: int,
    key: str = "rowid",
    by: Mapping[str, Any] | None = None,
) -> int:
    """delete ``table``'s ``count`` oldest rows (within ``by``) by its key's order: a cache's trim back to its cap.

    ``DELETE FROM t WHERE <by> AND "rowid" <= COALESCE((SELECT "rowid" FROM t WHERE <by> ORDER BY "rowid" LIMIT 1
    OFFSET $n), (SELECT max("rowid") FROM t WHERE <by>))``: an L1 table's rowid is the key it is stored by, so the
    trim seeks its ``count``-th oldest key and deletes the range up to it, every row when fewer are held, none from
    an empty table. The key is unique within ``by`` (rowid, a growing id): a range to it holds ``count`` rows. The
    caller knows how far past its cap the table is from a count it keeps (a trigger's) or a bounded probe
    (:func:`~threetears.core.keyed.count_at_most` with no ``by``, ``at_most`` the cap plus the most it trims).

    :param conn: the local cache's connection: :class:`~threetears.core.keyed.ThreadedSQLite` or
        :class:`~threetears.core.keyed.ThreadedDuckDB`. An L1 table is no L3 row and follows no epoch; an L3 table's
        rows are deleted through its collection (:func:`delete_before`)
    :ptype conn: ThreadedSQLite | ThreadedDuckDB
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param count: how many rows to delete; none when under one
    :ptype count: int
    :param key: the column whose order is the rows' age: SQLite's ``rowid``, or a key column (a growing id)
    :ptype key: str
    :param by: columns pinned ahead of it
    :ptype by: Mapping[str, Any] | None
    :return: the rows deleted
    :rtype: int
    :raises TypeError: when ``conn`` is not a local cache's
    """
    if not isinstance(conn, ThreadedSQLite | ThreadedDuckDB):
        raise TypeError(f"delete_oldest trims a local cache's table; {table} is read through {type(conn).__name__}")
    if count < 1:
        return 0
    conditions, params = by_conditions(by or {})
    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    pinned = "".join(f"{condition} AND " for condition in conditions)
    named = quote_identifier(key)
    quoted = quoted_table(table)
    # one range delete up to the count-th oldest key, found by one seek of the key's order (its OFFSET), or to the
    # newest when fewer are held: no list of keys built and probed (0.37 ms against 0.54 ms for 3,000 of 300,000)
    sql = (
        f"DELETE FROM {quoted} WHERE {pinned}{named} <= COALESCE("
        f"(SELECT {named} FROM {quoted}{where} ORDER BY {named} LIMIT 1 OFFSET ${len(params) + 1}), "
        f"(SELECT max({named}) FROM {quoted}{where}))"
    )
    return rows_affected(await conn.execute(sql, *params, count - 1))
