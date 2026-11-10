"""counts without a count of a table: a statement's own rows, the planner's estimate, or a bounded probe.

A ``count(*)`` of a table reads the table. Each question a count answered has a cheaper answer, and this module is the
one place each is spelled:

- **how many rows a write touched**: the statement says so itself (``UPDATE 3``, ``DELETE 0``, ``INSERT 0 5``):
  :func:`rows_affected` reads it off the status;
- **about how big a table is** (a progress bar, a size class, a choice of plan): the planner's estimate,
  ``pg_class.reltuples`` on PostgreSQL and YugabyteDB, ``duckdb_tables().estimated_size`` on DuckDB:
  :func:`estimated_rows`. None for a table never analysed: an estimate that does not exist is not zero;
- **whether any row is held** (an empty copy, a scope still holding rows): :func:`holds_any`, one ``EXISTS`` of one
  row, and :func:`empty_tables` for several tables in one statement;
- **how many rows each key holds** (a run's rows, to tell a complete load from a cut one): :func:`counts_by`, a
  ``GROUP BY`` of the key's own rows, a bounded batch of keys a statement;
- **how many, up to a bound** (a page's "more than 100", a cap): :func:`count_at_most`, a count of at most
  ``at_most + 1`` rows the key names, never more.

Each read that takes a ``by`` reads the rows it names (:func:`~threetears.core.keyed.read_keyed`'s ``by``): a leading
part of the key, or an index declared for the caller's read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from threetears.core.keyed._filters import AnyOf, by_conditions
from threetears.core.sql_fragments import quote_identifier

__all__ = [
    "BoundedCount",
    "Dialect",
    "count_at_most",
    "counts_by",
    "empty_tables",
    "estimated_rows",
    "holds_any",
    "rows_affected",
]

#: the verbs whose status ends in the rows they touched (``INSERT`` carries an oid before it)
_COUNTED_VERBS: Final[frozenset[str]] = frozenset(
    {"INSERT", "UPDATE", "DELETE", "SELECT", "COPY", "MERGE", "MOVE", "FETCH"}
)


class Dialect(StrEnum):
    """the database a statistics read is spelled for. PostgreSQL and YugabyteDB keep the same catalog."""

    POSTGRES = "postgres"
    YUGABYTE = "yugabyte"
    DUCKDB = "duckdb"


#: the planner's row estimate of a table, by dialect: ``$1`` the table, ``$2`` its schema (the current one when null)
_ESTIMATE_SQL: Final[Mapping[Dialect, str]] = {
    Dialect.POSTGRES: (
        "SELECT c.reltuples::bigint AS estimate FROM pg_catalog.pg_class c "
        "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
        "WHERE c.relname = $1 AND n.nspname = COALESCE($2, current_schema())"
    ),
    Dialect.DUCKDB: (
        "SELECT estimated_size AS estimate FROM duckdb_tables() "
        "WHERE table_name = $1 AND schema_name = COALESCE($2, current_schema())"
    ),
}


def rows_affected(status: str) -> int:
    """the rows a statement touched, read off its own status (asyncpg's ``execute`` answer).

    :param status: ``UPDATE 3``, ``DELETE 0``, ``INSERT 0 5``, ...
    :ptype status: str
    :return: the rows
    :rtype: int
    :raises ValueError: when the status names no row count (``CREATE TABLE``)
    """
    words = status.split()
    if len(words) < 2 or words[0].upper() not in _COUNTED_VERBS or not words[-1].isdigit():
        raise ValueError(f"the status {status!r} names no row count")
    return int(words[-1])


async def estimated_rows(
    conn: Any, table: str, *, schema: str | None = None, dialect: Dialect = Dialect.POSTGRES
) -> int | None:
    """the planner's estimate of ``table``'s rows: no row of the table is read.

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param table: the table's name, unquoted, as the catalog holds it
    :ptype table: str
    :param schema: its schema; the connection's current one when None
    :ptype schema: str | None
    :param dialect: the database's catalog to ask
    :ptype dialect: Dialect
    :return: the estimate; None when the table is unknown or was never analysed
    :rtype: int | None
    """
    sql = _ESTIMATE_SQL[Dialect.POSTGRES if dialect is Dialect.YUGABYTE else dialect]
    rows = await conn.fetch(sql, table, schema)
    if not rows:
        return None
    estimate = rows[0]["estimate"]
    # -1 is PostgreSQL 14+'s "never analysed"; YugabyteDB says 0 until its first ANALYZE
    if estimate is None or int(estimate) < 0:
        return None
    return int(estimate)


@dataclass(frozen=True)
class BoundedCount:
    """a count of at most a bound's rows.

    :param count: the rows counted, never more than the bound
    :ptype count: int
    :param more: whether rows past the bound are held
    :ptype more: bool
    """

    count: int
    more: bool


def _where(by: Mapping[str, Any]) -> tuple[str, list[Any]]:
    conditions, values = by_conditions(by)
    return (f" WHERE {' AND '.join(conditions)}" if conditions else ""), values


async def count_at_most(conn: Any, table: str, *, by: Mapping[str, Any], at_most: int) -> BoundedCount:
    """how many of the rows ``by`` names ``table`` holds, counting at most ``at_most + 1`` of them.

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param by: the rows to count: a leading part of the key, each a value or an :class:`~threetears.core.keyed.AnyOf`
    :ptype by: Mapping[str, Any]
    :param at_most: the bound
    :ptype at_most: int
    :return: the count, and whether rows past the bound are held
    :rtype: BoundedCount
    :raises ValueError: when ``by`` names nothing or ``at_most`` is under zero
    """
    if not by:
        raise ValueError(f"a bounded count of {table} names the rows it counts")
    if at_most < 0:
        raise ValueError(f"a bounded count counts at least none, got {at_most}")
    where, values = _where(by)
    sql = f"SELECT count(*) AS held FROM (SELECT 1 FROM {quote_identifier(table)}{where} LIMIT {at_most + 1}) AS held"
    rows = await conn.fetch(sql, *values)
    held = int(rows[0]["held"])
    return BoundedCount(count=min(held, at_most), more=held > at_most)


async def holds_any(conn: Any, table: str, *, by: Mapping[str, Any] | None = None) -> bool:
    """whether ``table`` holds any row ``by`` names (any row at all when None): one row read, at most.

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param by: the rows asked about, a leading part of the key; None asks whether the table is empty
    :ptype by: Mapping[str, Any] | None
    :return: whether a row is held
    :rtype: bool
    """
    where, values = _where(by or {})
    rows = await conn.fetch(f"SELECT 1 AS held FROM {quote_identifier(table)}{where} LIMIT 1", *values)
    return bool(rows)


async def empty_tables(conn: Any, tables: Sequence[str]) -> list[str]:
    """which of ``tables`` hold no row, in one statement: a probe of one row each.

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param tables: the tables, TRUSTED identifiers
    :ptype tables: Sequence[str]
    :return: the empty ones, in the order given
    :rtype: list[str]
    """
    if not tables:
        return []
    probes = ", ".join(
        f"NOT EXISTS (SELECT 1 FROM {quote_identifier(table)} LIMIT 1) AS e{index}"
        for index, table in enumerate(tables)
    )
    rows = await conn.fetch(f"SELECT {probes}")
    answer = rows[0]
    return [table for index, table in enumerate(tables) if answer[f"e{index}"]]


async def counts_by(
    conn: Any,
    table: str,
    *,
    column: str,
    values: AnyOf,
    by: Mapping[str, Any] | None = None,
    batch: int = 999,
) -> dict[Any, int]:
    """how many rows of ``table`` each of ``values`` holds in ``column`` (within ``by``), ``batch`` values a statement.

    ``SELECT "column", count(*) FROM t WHERE <by> AND "column" = ANY($n::<cast>[]) GROUP BY "column"``: each statement
    counts the rows its own keys name, so ``by`` and ``column`` together must be a leading part of the key (or of an
    index declared for the caller's read). A batch answers at most ``batch`` groups, under the rail's row cap.

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param column: the column counted by, a TRUSTED identifier
    :ptype column: str
    :param values: its values, and the array type they bind as
    :ptype values: AnyOf
    :param by: the prefix pinned ahead of it (a state, a tenant)
    :ptype by: Mapping[str, Any] | None
    :param batch: the most values a statement names
    :ptype batch: int
    :return: value -> its rows; a value holding none is absent
    :rtype: dict[Any, int]
    :raises ValueError: when ``batch`` is under one, or ``by`` names ``column`` too
    """
    if batch < 1:
        raise ValueError(f"a keyed count names at least one value a statement, got {batch}")
    pinned = dict(by or {})
    if column in pinned:
        raise ValueError(f"{table}: {column} is counted by, so it is not also pinned")
    unique = list(dict.fromkeys(values.values))
    counted: dict[Any, int] = {}
    quoted = quote_identifier(column)
    for start in range(0, len(unique), batch):
        chunk = AnyOf(unique[start : start + batch], values.cast)
        conditions, params = by_conditions({**pinned, column: chunk})
        sql = (
            f"SELECT {quoted} AS counted, count(*) AS held FROM {quote_identifier(table)} "
            f"WHERE {' AND '.join(conditions)} GROUP BY {quoted}"
        )
        for row in await conn.fetch(sql, *params):
            counted[row["counted"]] = int(row["held"])
    return counted
