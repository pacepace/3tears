"""the rows past a bound deleted a page at a time, through the index on the bound's column: expiry sweeps and caps.

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

__all__ = ["SweptRows", "delete_before"]


@dataclass(frozen=True)
class SweptRows:
    """what a paged sweep deleted.

    :param deleted: the rows deleted
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
