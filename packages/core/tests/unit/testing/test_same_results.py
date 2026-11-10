"""the equivalence harness catches every way a conversion could change an answer, and passes the one that does not.

Run on DuckDB through an asyncpg-shaped stand-in (``$n`` placeholders, a transaction rolled back per case): the
same statement passes; a helper that drops the tenant filter, adds a row, loses a row's NULL, or returns the same rows
in another order fails, naming the case that showed it; rows tied on the old statement's sort may come in either
order when ``order_by`` names the sort.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import duckdb
import pytest

from threetears.core.testing.same_results import Seeded, assert_same_results


class _Transaction:
    def __init__(self, db: duckdb.DuckDBPyConnection) -> None:
        self._db = db

    async def start(self) -> None:
        self._db.execute("BEGIN")

    async def rollback(self) -> None:
        self._db.execute("ROLLBACK")


class _Duck:
    """the asyncpg calls the harness makes, on one DuckDB connection."""

    def __init__(self, db: duckdb.DuckDBPyConnection) -> None:
        self._db = db

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        cursor = self._db.execute(sql, list(params))
        names = [column[0] for column in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]

    async def execute(self, sql: str, *params: Any) -> None:
        self._db.execute(sql, list(params))

    async def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        self._db.executemany(sql, [list(row) for row in rows])

    def transaction(self) -> _Transaction:
        return _Transaction(self._db)


@pytest.fixture
async def conn() -> AsyncIterator[_Duck]:
    db = duckdb.connect()
    db.execute("CREATE TABLE t (tenant VARCHAR NOT NULL, k INTEGER NOT NULL, v VARCHAR, s INTEGER NOT NULL)")
    yield _Duck(db)
    db.close()


def _row(seed: Seeded) -> Mapping[str, Any]:
    return {"tenant": f"t{seed.tenant}", "k": seed.index, "v": None if seed.null else f"v{seed.index}", "s": seed.tie}


_OLD = "SELECT k, v FROM t WHERE tenant = $1 ORDER BY k"


def _asking(sql: str, *params: Any) -> Any:
    async def helper(conn: _Duck) -> list[dict[str, Any]]:
        return await conn.fetch(sql, *params)

    return helper


async def test_the_same_statement_gives_the_same_rows(conn: _Duck) -> None:
    """Flips if the harness reports a difference where there is none, in any case."""
    await assert_same_results(conn, table="t", row=_row, old_sql=_OLD, params=["t0"], helper=_asking(_OLD, "t0"))


async def test_a_dropped_filter_is_caught_at_the_tenant_boundary(conn: _Duck) -> None:
    """Flips if a helper reading every tenant's rows passes for one reading tenant 0's."""
    with pytest.raises(AssertionError, match="a tenant boundary: the old statement gave 6 rows, the helper 12"):
        await assert_same_results(
            conn, table="t", row=_row, old_sql=_OLD, params=["t0"], helper=_asking("SELECT k, v FROM t ORDER BY k")
        )


async def test_an_extra_row_is_caught(conn: _Duck) -> None:
    """Flips if a helper returning one row more passes."""

    async def one_more(conn: _Duck) -> list[dict[str, Any]]:
        return [*(await conn.fetch(_OLD, "t0")), {"k": -1, "v": "extra"}]

    with pytest.raises(AssertionError, match="empty: the old statement gave 0 rows, the helper 1"):
        await assert_same_results(conn, table="t", row=_row, old_sql=_OLD, params=["t0"], helper=one_more)


async def test_a_lost_null_is_caught(conn: _Duck) -> None:
    """Flips if a helper that turns NULL into a value passes (the NULLs case shows it)."""
    filled = _asking("SELECT k, coalesce(v, '') AS v FROM t WHERE tenant = $1 ORDER BY k", "t0")
    with pytest.raises(AssertionError, match="NULLs: "):
        await assert_same_results(conn, table="t", row=_row, old_sql=_OLD, params=["t0"], helper=filled)


async def test_a_lost_order_is_caught(conn: _Duck) -> None:
    """Flips if the old statement's order is not compared: the same rows reversed pass."""
    reversed_ = _asking("SELECT k, v FROM t WHERE tenant = $1 ORDER BY k DESC", "t0")
    with pytest.raises(AssertionError, match="many rows: the same rows in another order"):
        await assert_same_results(conn, table="t", row=_row, old_sql=_OLD, params=["t0"], helper=reversed_)


async def test_rows_tied_on_the_sort_may_come_in_either_order(conn: _Duck) -> None:
    """Flips if rows tied on ``order_by`` must come in the old statement's order, or if the sort itself goes
    uncompared (a helper ordering by another column fails)."""
    old = "SELECT k, s FROM t WHERE tenant = $1 ORDER BY s, k"
    ties_reversed = _asking("SELECT k, s FROM t WHERE tenant = $1 ORDER BY s, k DESC", "t0")
    await assert_same_results(
        conn, table="t", row=_row, old_sql=old, params=["t0"], helper=ties_reversed, order_by=["s"]
    )
    by_key_descending = _asking("SELECT k, s FROM t WHERE tenant = $1 ORDER BY k DESC", "t0")
    with pytest.raises(AssertionError, match="another order"):
        await assert_same_results(
            conn, table="t", row=_row, old_sql=old, params=["t0"], helper=by_key_descending, order_by=["s"]
        )


async def test_every_case_leaves_the_table_as_it_was(conn: _Duck) -> None:
    """Flips if a case's seed outlives it (the transaction not rolled back)."""
    await conn.execute("INSERT INTO t VALUES ('kept', 1, 'x', 1)")
    await assert_same_results(conn, table="t", row=_row, old_sql=_OLD, params=["t0"], helper=_asking(_OLD, "t0"))
    assert await conn.fetch("SELECT tenant, k FROM t") == [{"tenant": "kept", "k": 1}]


_STATE = "SELECT tenant, k, v, s FROM t"


async def test_a_write_is_compared_by_the_table_it_leaves(conn: _Duck) -> None:
    """Flips if a write's comparison reads only what it returned: the same update passes; one that drops its tenant
    filter (returning nothing either way) fails on the table after it, naming the case."""
    update = "UPDATE t SET v = 'w' WHERE tenant = $1 AND k = $2"
    await assert_same_results(
        conn, table="t", row=_row, old_sql=update, params=["t0", 0], helper=_asking(update, "t0", 0), state_sql=_STATE
    )
    with pytest.raises(AssertionError, match=r"a tenant boundary \(the table after\)"):
        await assert_same_results(
            conn,
            table="t",
            row=_row,
            old_sql="UPDATE t SET v = 'w' WHERE tenant = $1 AND k >= $2",
            params=["t0", 0],
            helper=_asking("UPDATE t SET v = 'w' WHERE k >= $1", 0),
            state_sql=_STATE,
        )
