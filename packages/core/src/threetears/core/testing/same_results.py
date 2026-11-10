"""the one equivalence harness: a helper call gives the rows the hand-written statement it replaces gave.

Every conversion of a hand-written statement to a 3tears helper (the ``use-the-helper`` rule) proves it changes no
answer: :func:`assert_same_results` seeds the statement's table for each of :func:`_cases` -- empty, one row,
many rows, NULLs, ties in the sort order, a tenant boundary, a page boundary and one past it -- runs the old statement
and the helper call on the same rows, and compares what they return as multisets, and in order when the old statement
ordered its rows. Each case runs in a transaction rolled back after it, so the table is as the test left it.

It runs on any asyncpg-shaped connection (``fetch``, ``execute``, ``executemany``, ``transaction``): PostgreSQL and
YugabyteDB in a package's integration tests, the same calls on each.

.. code-block:: python

    await assert_same_results(
        conn,
        table="memories",
        row=lambda seed: {"agent_id": AGENTS[seed.tenant], "memory_id": uuid5(NS, str(seed.index)), ...},
        old_sql="SELECT memory_id FROM memories WHERE agent_id = $1 ORDER BY memory_id",
        params=[AGENTS[0]],
        helper=lambda c: keyed.keyset_page(c, "memories", by={"agent_id": AGENTS[0]}, ...),
        order_by=["memory_id"],
    )
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

__all__ = [
    "Seeded",
    "assert_same_results",
]

_ORDER_BY: Final = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)


@dataclass(frozen=True)
class _SeedCase:
    """one shape of seeded table.

    :ivar name: what it seeds, as a failure names it
    :ivar rows: how many rows
    :ivar tenants: how many tenants the rows are split between (2: a tenant boundary)
    :ivar nulls: every other row's nullable columns NULL
    :ivar ties: rows share their sort value in runs of three
    """

    name: str
    rows: int
    tenants: int = 1
    nulls: bool = False
    ties: bool = False


@dataclass(frozen=True)
class Seeded:
    """what the row factory is handed for one row.

    :ivar index: the row's place, 0 up, unique within the case (a key may be built from it)
    :ivar case: the case's name
    :ivar tenant: which tenant the row belongs to, 0 or 1
    :ivar null: whether the row's nullable columns should be NULL
    :ivar tie: the row's sort value: its index, or shared by a run of three in the ties case
    """

    index: int
    case: str
    tenant: int
    null: bool
    tie: int


def _cases(page_size: int) -> tuple[_SeedCase, ...]:
    """the seeds every comparison runs, the page boundary at ``page_size``.

    :param page_size: the helper's page, when it pages
    :ptype page_size: int
    :return: the cases
    :rtype: tuple[_SeedCase, ...]
    """
    return (
        _SeedCase("empty", 0),
        _SeedCase("one row", 1),
        _SeedCase("many rows", 25),
        _SeedCase("NULLs", 12, nulls=True),
        _SeedCase("ties in the sort order", 12, ties=True),
        _SeedCase("a tenant boundary", 12, tenants=2),
        _SeedCase("a page boundary", page_size),
        _SeedCase("one past a page boundary", page_size + 1),
    )


def _seeded(case: _SeedCase) -> list[Seeded]:
    """each row's seed for one case: the tenants in contiguous runs (the boundary between them), NULLs every other
    row, ties in runs of three.

    :param case: the case
    :ptype case: _SeedCase
    :return: the seeds
    :rtype: list[Seeded]
    """
    per_tenant = max(1, -(-case.rows // case.tenants))
    return [
        Seeded(
            index=index,
            case=case.name,
            tenant=min(case.tenants - 1, index // per_tenant),
            null=case.nulls and index % 2 == 1,
            tie=index // 3 if case.ties else index,
        )
        for index in range(case.rows)
    ]


def _normal(row: Mapping[str, Any], columns: Sequence[str] | None) -> tuple[tuple[str, Any], ...]:
    """a row as a comparable value: its columns (or the ones named) in name order.

    :param row: the row
    :ptype row: Mapping[str, Any]
    :param columns: the columns compared; every column when None
    :ptype columns: Sequence[str] | None
    :return: ``(column, value)`` pairs
    :rtype: tuple[tuple[str, Any], ...]
    """
    names = sorted(row) if columns is None else sorted(columns)
    return tuple((name, row[name]) for name in names)


def _differ(
    case: str,
    old: list[tuple[tuple[str, Any], ...]],
    new: list[tuple[tuple[str, Any], ...]],
    ordered_by: Sequence[str] | None,
    ordered: bool,
) -> str | None:
    """what differs between the old statement's rows and the helper's, or None.

    :param case: the seed case
    :ptype case: str
    :param old: the old statement's rows, normalised
    :ptype old: list[tuple[tuple[str, Any], ...]]
    :param new: the helper's rows, normalised
    :ptype new: list[tuple[tuple[str, Any], ...]]
    :param ordered_by: the columns whose order is compared (ties among them may come in either order); every column
        when None
    :ptype ordered_by: Sequence[str] | None
    :param ordered: whether order is compared at all
    :ptype ordered: bool
    :return: the difference, or None
    :rtype: str | None
    """
    lost = Counter(old) - Counter(new)
    extra = Counter(new) - Counter(old)
    if lost or extra:
        return (
            f"{case}: the old statement gave {len(old)} rows, the helper {len(new)}; missing from the helper "
            f"{sorted(lost.elements())[:3]}, not in the old {sorted(extra.elements())[:3]}"
        )
    if ordered:

        def keyed(rows: list[tuple[tuple[str, Any], ...]]) -> list[tuple[Any, ...]]:
            if ordered_by is None:
                return [tuple(value for _, value in row) for row in rows]
            return [tuple(dict(row)[column] for column in ordered_by) for row in rows]

        if keyed(old) != keyed(new):
            return f"{case}: the same rows in another order (by {list(ordered_by or ['every column'])})"
    return None


async def _seed(conn: Any, table: str, rows: list[Mapping[str, Any]]) -> None:
    """empty ``table`` and insert ``rows`` (the caller's transaction rolls both back).

    :param conn: the connection
    :ptype conn: Any
    :param table: the TRUSTED table name
    :ptype table: str
    :param rows: the rows, every one with the same columns
    :ptype rows: list[Mapping[str, Any]]
    :return: nothing
    :rtype: None
    """
    await conn.execute(f"DELETE FROM {table}")  # noqa: S608 - a test's own table, in a transaction rolled back
    if rows:
        columns = list(rows[0])
        marks = ", ".join(f"${index}" for index in range(1, len(columns) + 1))
        await conn.executemany(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({marks})",  # noqa: S608 - trusted test identifiers
            [tuple(row[column] for column in columns) for row in rows],
        )


async def assert_same_results(
    conn: Any,
    *,
    table: str,
    row: Callable[[Seeded], Mapping[str, Any]],
    old_sql: str,
    params: Sequence[Any] = (),
    helper: Callable[[Any], Awaitable[Sequence[Mapping[str, Any]]]],
    columns: Sequence[str] | None = None,
    order_by: Sequence[str] | None = None,
    ordered: bool | None = None,
    page_size: int = 5,
) -> None:
    """seed ``table`` for every case, run ``old_sql`` and ``helper`` on the same rows, and fail on any difference.

    :param conn: an asyncpg-shaped connection (``fetch``, ``execute``, ``executemany``, ``transaction``)
    :ptype conn: Any
    :param table: the table both read or write, a TRUSTED name; emptied and seeded inside each case's transaction
    :ptype table: str
    :param row: the row the table holds for a seed (:class:`Seeded`): every row the same columns, its key built from
        ``seed.index``, its tenant column from ``seed.tenant``, its nullable columns NULL when ``seed.null``, its sort
        column from ``seed.tie``
    :ptype row: Callable[[Seeded], Mapping[str, Any]]
    :param old_sql: the hand-written statement the helper replaces
    :ptype old_sql: str
    :param params: its parameters (tenant 0's, say, so the tenant boundary case shows a filter dropped)
    :ptype params: Sequence[Any]
    :param helper: the helper call, handed the same connection, returning its rows
    :ptype helper: Callable[[Any], Awaitable[Sequence[Mapping[str, Any]]]]
    :param columns: the columns compared, when the helper returns more than the old statement selected; every
        column when None
    :ptype columns: Sequence[str] | None
    :param order_by: the columns the old statement orders by, compared in order (rows tied on them may come in
        either order); every column when None
    :ptype order_by: Sequence[str] | None
    :param ordered: whether order is compared; when None, whether ``old_sql`` has an ``ORDER BY``
    :ptype ordered: bool | None
    :param page_size: the helper's page, where the page boundary cases sit
    :ptype page_size: int
    :return: nothing
    :rtype: None
    :raises AssertionError: naming each case whose rows differ, and how
    """
    compare_order = bool(_ORDER_BY.search(old_sql)) if ordered is None else ordered
    differences: list[str] = []
    for case in _cases(page_size):
        transaction = conn.transaction()
        await transaction.start()
        try:
            await _seed(conn, table, [row(seed) for seed in _seeded(case)])
            old = [_normal(dict(found), columns) for found in await conn.fetch(old_sql, *params)]
            new = [_normal(dict(found), columns) for found in await helper(conn)]
        finally:
            await transaction.rollback()
        difference = _differ(case.name, old, new, order_by, compare_order)
        if difference is not None:
            differences.append(difference)
    assert not differences, "the helper does not give the old statement's rows:\n" + "\n".join(differences)
