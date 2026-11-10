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

from threetears.core.keyed._filters import _SQL_TYPE, AnyOf, Derived, Fragment, by_conditions, relation, renumbered
from threetears.core.sql_fragments import quote_identifier

__all__ = [
    "BoundedCount",
    "Dialect",
    "count_at_most",
    "counts_by",
    "counts_by_relations",
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


def _ordered(table: str | Derived, order: Sequence[str]) -> str:
    """the ``ORDER BY`` a probe walks its rows in: the table's key (or a declared index's columns) after ``by``, so the
    bound is a walk of that index, never a scan stopped early. A scan the planner stops at a LIMIT reads every row
    before the ones it keeps: a key whose rows lie at the end of the heap is a whole read.

    :raises ValueError: when ``order`` names no column
    """
    if not order:
        raise ValueError(f"a probe of {table} names the key it walks (order=)")
    return " ORDER BY " + ", ".join(f"r.{quote_identifier(column)}" for column in order)


async def count_at_most(
    conn: Any,
    table: str | Derived,
    *,
    at_most: int,
    order: Sequence[str],
    by: Mapping[str, Any] | None = None,
) -> BoundedCount:
    """how many of the rows ``by`` names ``table`` holds (any of its rows when None), counting at most
    ``at_most + 1`` of them: ``SELECT count(*) FROM (SELECT 1 FROM t WHERE <by> ORDER BY <order> LIMIT $n)``.

    Unkeyed, it is a bounded probe of the table's first rows, never more than the bound: the size class of a table no
    estimate is kept for (an L1 cache, a table never analysed).

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param table: the table (``schema.table`` quoted a segment at a time), or a derived relation
    :ptype table: str | Derived
    :param at_most: the bound
    :ptype at_most: int
    :param order: the key the probe walks (the table's key, or a declared index's columns): ``by``'s columns and those
        after them
    :ptype order: Sequence[str]
    :param by: the rows to count: a leading part of the key, each a value or an :class:`~threetears.core.keyed.AnyOf`;
        any rows when None
    :ptype by: Mapping[str, Any] | None
    :return: the count, and whether rows past the bound are held
    :rtype: BoundedCount
    :raises ValueError: when ``at_most`` is under zero
    """
    if at_most < 0:
        raise ValueError(f"a bounded count counts at least none, got {at_most}")
    where, values = _where(by or {})
    ordered = _ordered(table, order)
    sql = (
        f"SELECT count(*) AS held FROM (SELECT 1 FROM {relation(table, 'r')}{where}{ordered} "
        f"LIMIT ${len(values) + 1}) AS held"
    )
    rows = await conn.fetch(sql, *values, at_most + 1)
    held = int(rows[0]["held"])
    return BoundedCount(count=min(held, at_most), more=held > at_most)


async def holds_any(
    conn: Any, table: str | Derived, *, order: Sequence[str], by: Mapping[str, Any] | None = None
) -> bool:
    """whether ``table`` holds any row ``by`` names (any row at all when None): one row read, at most.

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param order: the key the probe walks (the table's key, or a declared index's columns)
    :ptype order: Sequence[str]
    :param by: the rows asked about, a leading part of the key; None asks whether the table is empty
    :ptype by: Mapping[str, Any] | None
    :return: whether a row is held
    :rtype: bool
    """
    where, values = _where(by or {})
    rows = await conn.fetch(
        f"SELECT 1 AS held FROM {relation(table, 'r')}{where}{_ordered(table, order)} LIMIT 1", *values
    )
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
    table: str | Derived,
    *,
    column: str,
    values: AnyOf | None = None,
    where: Sequence[Fragment] = (),
    by: Mapping[str, Any] | None = None,
    batch: int = 999,
    cast: str | None = "text",
) -> dict[Any, int]:
    """how many rows of ``table`` each value of ``column`` holds (within ``by``), ``batch`` values a statement.

    With neither ``values`` nor ``where``, every value of ``column`` within the prefix ``by`` is counted, ``batch``
    values a statement in the column's order (``... AND "column" > $last GROUP BY "column" ORDER BY "column" LIMIT
    batch``): each statement reads the rows of its own groups through the ``(by, column)`` index, never the prefix's
    rows at once. ``cast`` is the column's SQL type the page's last value binds as (None binds it as it is: SQLite).

    ``SELECT "column", count(*) FROM t WHERE <by> AND "column" = ANY($n::<cast>[]) GROUP BY "column"``: each statement
    counts the rows its own keys name, so ``by`` and ``column`` together must be a leading part of the key (or of an
    index declared for the caller's read). The values are named by ``values``, or by ``where`` (a bounded subquery on
    the source's own record, ``"run_id" IN (SELECT run_id FROM runs WHERE status = 'ok' AND ...)``) in one statement.

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param table: the table (``schema.table``), or a derived relation
    :ptype table: str | Derived
    :param column: the column counted by, a TRUSTED identifier
    :ptype column: str
    :param values: its values, and how they bind (an array, or a list for a warehouse); None when ``where`` names them
    :ptype values: AnyOf | None
    :param where: filters naming the rows counted beside ``by``, each a fragment from ``$1``
    :ptype where: Sequence[Fragment]
    :param by: the prefix pinned ahead of it (a state, a tenant)
    :ptype by: Mapping[str, Any] | None
    :param batch: the most values a statement names (or, paging every value, the most groups a statement answers)
    :ptype batch: int
    :param cast: the column's SQL type, binding the last value of a page when every value is counted; None binds it
        as it is (SQLite)
    :ptype cast: str | None
    :return: value -> its rows; a value holding none is absent
    :rtype: dict[Any, int]
    :raises ValueError: when ``batch`` is under one, ``by`` names ``column`` too, or neither ``values``, ``where``
        nor a prefix ``by`` names the rows
    """
    if values is None and not where:
        return await _counts_paged(conn, table, column=column, by=dict(by or {}), batch=batch, cast=cast)
    counted = await counts_by_relations(
        conn, {"": table}, column=column, values=values, where=where, by=by, batch=batch
    )
    return counted[""]


async def _counts_paged(
    conn: Any, table: str | Derived, *, column: str, by: dict[str, Any], batch: int, cast: str | None
) -> dict[Any, int]:
    """every value of ``column`` within the prefix ``by`` counted, ``batch`` groups a statement in its order.

    :raises ValueError: when ``by`` is empty (that is the whole table), names ``column``, or ``batch`` is under one
    """
    if not by:
        raise ValueError(
            f"a keyed count names its values of {column}, a filter naming them, or the prefix of {table} it counts "
            "every value within (by=)"
        )
    if column in by:
        raise ValueError(f"{column} is counted by, so it is not also pinned")
    if batch < 1:
        raise ValueError(f"a keyed count answers at least one group a statement, got {batch}")
    if cast is not None and not _SQL_TYPE.match(cast):
        raise ValueError(f"counts_by: {cast!r} is not a SQL type name")
    quoted = quote_identifier(column)
    counted: dict[Any, int] = {}
    last: Any = None
    while True:
        conditions, params = by_conditions(by)
        conditions.append(f"{quoted} IS NOT NULL")
        if last is not None:
            params.append(last)
            conditions.append(f"{quoted} > ${len(params)}{'' if cast is None else f'::{cast}'}")
        sql = (
            f"SELECT {quoted} AS counted, count(*) AS held FROM {relation(table, 'r')} WHERE {' AND '.join(conditions)}"
            f" GROUP BY {quoted} ORDER BY {quoted} LIMIT {batch}"
        )
        rows = await conn.fetch(sql, *params)
        for row in rows:
            counted[row["counted"]] = int(row["held"])
        if len(rows) < batch:
            break
        last = rows[-1]["counted"]
    # the NULL group, which no "> last" reaches: one keyed count of its own
    conditions, params = by_conditions({**by, column: None})
    rows = await conn.fetch(
        f"SELECT count(*) AS held FROM {relation(table, 'r')} WHERE {' AND '.join(conditions)}", *params
    )
    if rows and int(rows[0]["held"]):
        counted[None] = int(rows[0]["held"])
    return counted


async def counts_by_relations(
    conn: Any,
    tables: Mapping[str, str | Derived],
    *,
    column: str,
    values: AnyOf | None = None,
    where: Sequence[Fragment] = (),
    by: Mapping[str, Any] | None = None,
    batch: int = 999,
) -> dict[str, dict[Any, int]]:
    """:func:`counts_by` over several relations at once: one statement a batch, a ``UNION ALL`` of each relation's
    keyed count, so a warehouse answers every relation in the statements one relation took.

    A statement answers at most ``batch`` groups over every relation together: each names ``batch // len(tables)``
    values (at least one).

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param tables: a label -> each relation (``schema.table``, or a derived relation)
    :ptype tables: Mapping[str, str | Derived]
    :param column: the column counted by, in every relation
    :ptype column: str
    :param values: its values; None when ``where`` names them
    :ptype values: AnyOf | None
    :param where: filters naming the rows counted beside ``by``, each a fragment from ``$1``, the same in every
        relation
    :ptype where: Sequence[Fragment]
    :param by: the prefix pinned in every relation
    :ptype by: Mapping[str, Any] | None
    :param batch: the most groups a statement answers
    :ptype batch: int
    :return: label -> value -> its rows; every label present
    :rtype: dict[str, dict[Any, int]]
    :raises ValueError: as :func:`counts_by`, or when no relation is named
    """
    if batch < 1:
        raise ValueError(f"a keyed count answers at least one group a statement, got {batch}")
    if not tables:
        raise ValueError("a keyed count names at least one relation")
    pinned = dict(by or {})
    if column in pinned:
        raise ValueError(f"{column} is counted by, so it is not also pinned")
    if values is None and not where:
        raise ValueError(f"a keyed count names its values of {column}, or a filter naming them")
    labels = list(tables)
    counted: dict[str, dict[Any, int]] = {label: {} for label in labels}
    quoted = quote_identifier(column)
    if values is None:
        chunks: list[dict[str, Any]] = [{}]
    else:
        unique = list(dict.fromkeys(values.values))
        size = max(1, batch // len(labels))
        chunks = [
            {column: AnyOf(unique[start : start + size], values.cast, as_list=values.as_list)}
            for start in range(0, len(unique), size)
        ]
    for chunk in chunks:
        conditions, params = by_conditions({**pinned, **chunk})
        for fragment in where:
            conditions.append(renumbered(fragment, len(params)))
            params += list(fragment.params)
        clause = " AND ".join(conditions)
        parts = [
            f"SELECT {index} AS relation, {quoted} AS counted, count(*) AS held FROM {relation(tables[label], 'r')} "
            f"WHERE {clause} GROUP BY {quoted}"
            for index, label in enumerate(labels)
        ]
        for row in await conn.fetch(" UNION ALL ".join(parts), *params):
            counted[labels[int(row["relation"])]][row["counted"]] = int(row["held"])
    return counted
