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
from typing import Any

from threetears.core.keyed._filters import by_conditions
from threetears.core.keyed.counts import rows_affected
from threetears.core.sql_fragments import quote_identifier

__all__ = ["SweptRows", "delete_before", "update_before"]


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
    conn: Any,
    table: str,
    *,
    key: Sequence[str],
    column: str,
    before: Any,
    page: int,
    by: Mapping[str, Any] | None = None,
    most_pages: int | None = None,
) -> SweptRows:
    """delete the rows of ``table`` (within ``by``) whose ``column`` is before ``before``, ``page`` rows a statement.

    :param conn: anything with an asyncpg-shaped ``execute`` answering the status (``DELETE 3``)
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
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
    :return: the rows deleted and the statements sent
    :rtype: SweptRows
    :raises ValueError: when ``key`` is empty or ``page`` / ``most_pages`` is under one
    """
    if not key:
        raise ValueError(f"a sweep of {table} names its key")
    if page < 1 or (most_pages is not None and most_pages < 1):
        raise ValueError(f"a sweep of {table} deletes at least one row in at least one page")
    conditions, params = by_conditions(by or {})
    conditions.append(f"{quote_identifier(column)} < ${len(params) + 1}")
    params.append(before)
    named = ", ".join(quote_identifier(part) for part in key)
    sql = (
        f"DELETE FROM {quote_identifier(table)} WHERE ({named}) IN (SELECT {named} FROM {quote_identifier(table)} "
        f"WHERE {' AND '.join(conditions)} ORDER BY {quote_identifier(column)} LIMIT {page})"
    )
    deleted = statements = 0
    while most_pages is None or statements < most_pages:
        swept = rows_affected(await conn.execute(sql, *params))
        statements += 1
        deleted += swept
        if swept < page:
            break
    return SweptRows(deleted=deleted, statements=statements)


async def update_before(
    conn: Any,
    table: str,
    *,
    key: Sequence[str],
    column: str,
    before: Any,
    set_to: Mapping[str, Any],
    page: int,
    by: Mapping[str, Any] | None = None,
    most_pages: int | None = None,
) -> SweptRows:
    """set ``set_to`` on the rows of ``table`` (within ``by``) whose ``column`` is before ``before``, a page at a time.

    The reap of stale rows (fires stuck ``dispatching``): ``UPDATE <table> SET ... WHERE (<key>) IN (SELECT <key> FROM
    <table> WHERE <by> AND <column> < $n ORDER BY <column> LIMIT <page>)``, repeated until a page touches fewer rows
    than it asked for. Each update must move its rows out of the pages it reads (``set_to`` names a column of ``by``,
    or ``column`` itself), or the next page would read them again: that is refused.

    :param conn: anything with an asyncpg-shaped ``execute`` answering the status (``UPDATE 3``)
    :ptype conn: Any
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
    :return: the rows updated and the statements sent
    :rtype: SweptRows
    :raises ValueError: when ``key`` or ``set_to`` is empty, a bound is under one, or the update leaves its rows in
        the pages it reads
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
    named = ", ".join(quote_identifier(part) for part in key)
    sql = (
        f"UPDATE {quote_identifier(table)} SET {', '.join(assignments)} WHERE ({named}) IN (SELECT {named} FROM "
        f"{quote_identifier(table)} WHERE {' AND '.join(conditions)} ORDER BY {quote_identifier(column)} LIMIT {page})"
    )
    updated = statements = 0
    while most_pages is None or statements < most_pages:
        touched = rows_affected(await conn.execute(sql, *params))
        statements += 1
        updated += touched
        if touched < page:
            break
    return SweptRows(deleted=updated, statements=statements)
