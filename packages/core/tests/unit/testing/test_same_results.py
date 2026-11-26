"""the equivalence harness catches every way a conversion could change an answer, and passes the one that does not.

Run on DuckDB through an asyncpg-shaped stand-in (``$n`` placeholders, a transaction rolled back per case): the
same statement passes; a helper that drops the tenant filter, adds a row, loses a row's NULL, or returns the same rows
in another order fails, naming the case that showed it; rows tied on the old statement's sort may come in either
order when ``order_by`` names the sort.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import duckdb
import pytest

from threetears.core.collections.schema_backed import INT_TYPE, STRING_TYPE, Column, TableSchema
from threetears.core.ddl.row_epoch import EPOCH_COLUMN, ROW_STAMPED
from threetears.core.testing.l1_tables import TEST_L1_TABLE
from threetears.core.testing.same_results import Seeded, assert_same_results


class _Transaction:
    def __init__(self, db: duckdb.DuckDBPyConnection) -> None:
        self._db = db

    async def start(self) -> None:
        await asyncio.to_thread(self._db.execute, "BEGIN")

    async def rollback(self) -> None:
        await asyncio.to_thread(self._db.execute, "ROLLBACK")


def _fetched(db: duckdb.DuckDBPyConnection, sql: str, params: list[Any]) -> list[dict[str, Any]]:
    cursor = db.execute(sql, params)
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


class _Duck:
    """the asyncpg calls the harness makes, on one DuckDB connection. Each statement runs on a thread, as asyncpg's
    wait on the server is off the loop: the harness seeds every case's rows by ``executemany``, a DuckDB statement a
    row, which on the loop held it past the loop-stall guard's limit (threetears.core.testing.loop_stall)."""

    def __init__(self, db: duckdb.DuckDBPyConnection) -> None:
        self._db = db

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        return await asyncio.to_thread(_fetched, self._db, sql, list(params))

    async def execute(self, sql: str, *params: Any) -> None:
        await asyncio.to_thread(self._db.execute, sql, list(params))

    async def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        await asyncio.to_thread(lambda: self._db.executemany(sql, [list(row) for row in rows]))

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


async def test_rows_differing_by_a_null_beside_a_value_are_reported_not_crashed_on(conn: _Duck) -> None:
    """Flips if a difference whose rows hold a NULL where others hold a value (a LEFT JOIN's unmatched row) ends in a
    TypeError from sorting them, not the difference reported: the report sorts the rows by their text."""

    async def a_null_and_a_number(conn: _Duck) -> list[dict[str, Any]]:
        return [*(await conn.fetch(_OLD, "t0")), {"k": None, "v": None}, {"k": 7, "v": None}]

    with pytest.raises(AssertionError, match="the old statement gave 0 rows, the helper 2"):
        await assert_same_results(conn, table="t", row=_row, old_sql=_OLD, params=["t0"], helper=a_null_and_a_number)


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


@pytest.fixture
async def documents() -> AsyncIterator[_Duck]:
    """a table whose rows hold an array, an object, and an object nesting an array and an object (decoded jsonb)."""
    db = duckdb.connect()
    db.execute(
        "CREATE TABLE d (tenant VARCHAR NOT NULL, k INTEGER NOT NULL, tags VARCHAR[], "
        "meta STRUCT(a INTEGER, b VARCHAR), deep STRUCT(names VARCHAR[], inner_ STRUCT(x INTEGER)))"
    )
    yield _Duck(db)
    db.close()


def _document(seed: Seeded) -> Mapping[str, Any]:
    return {
        "tenant": f"t{seed.tenant}",
        "k": seed.index,
        "tags": [] if seed.null else [f"tag{seed.index}", "shared"],
        "meta": {"a": seed.tie, "b": f"b{seed.index}"},
        "deep": {"names": [f"n{seed.index}"], "inner_": {"x": seed.index}},
    }


_OLD_DOCUMENTS = "SELECT k, tags, meta, deep FROM d WHERE tenant = $1 ORDER BY k"


async def test_rows_holding_lists_and_objects_compare_by_their_values(documents: _Duck) -> None:
    """Flips if a row holding an array, an object or a nested one cannot be compared (an unhashable value), or if
    equal ones are reported as differing."""
    await assert_same_results(
        documents,
        table="d",
        row=_document,
        old_sql=_OLD_DOCUMENTS,
        params=["t0"],
        helper=_asking(_OLD_DOCUMENTS, "t0"),
    )


@pytest.mark.parametrize(
    ("column", "changed"),
    [
        ("tags", "list_append(tags, 'extra')"),
        ("meta", "struct_pack(a := meta.a + 1, b := meta.b)"),
        ("deep", "struct_pack(names := deep.names, inner_ := struct_pack(x := deep.inner_.x + 1))"),
    ],
)
async def test_a_changed_list_object_or_nested_value_is_caught(documents: _Duck, column: str, changed: str) -> None:
    """Flips if a helper whose array, object, or object nested in an object differs passes for the old statement."""
    others = ", ".join(name for name in ("tags", "meta", "deep") if name != column)
    helper_sql = f"SELECT k, {others}, {changed} AS {column} FROM d WHERE tenant = $1 ORDER BY k"
    with pytest.raises(AssertionError, match="one row: "):
        await assert_same_results(
            documents,
            table="d",
            row=_document,
            old_sql=_OLD_DOCUMENTS,
            params=["t0"],
            helper=_asking(helper_sql, "t0"),
        )


async def test_related_rows_are_seeded_in_each_case_and_the_old_rows_answered(conn: _Duck) -> None:
    """Flips if ``related`` is not handed each case's seeded rows inside the case (a statement reading another
    table's rows beside the seeded one would read none, on both sides), or if the old statement's rows per case are
    not answered (a clause proof reads them to show its clause restricted something)."""
    await conn.execute("CREATE TABLE p (k INTEGER NOT NULL, flag BOOLEAN NOT NULL)")
    seen: list[int] = []

    async def parents(c: _Duck, rows: list[Mapping[str, Any]]) -> None:
        seen.append(len(rows))
        await c.execute("DELETE FROM p")
        if rows:
            await c.executemany("INSERT INTO p VALUES ($1, $2)", [(row["k"], row["k"] % 2 == 0) for row in rows])

    old = "SELECT k FROM t WHERE tenant = $1 AND EXISTS (SELECT 1 FROM p WHERE p.k = t.k AND p.flag) ORDER BY k"
    answered = await assert_same_results(
        conn, table="t", row=_row, old_sql=old, params=["t0"], helper=_asking(old, "t0"), related=parents
    )
    assert seen == [0, 1, 25, 12, 12, 12, 5, 6]
    assert [len(rows) for rows in answered] == [0, 1, 13, 6, 6, 3, 3, 3]
    assert await conn.fetch("SELECT * FROM p") == []


#: a ``ROW_STAMPED`` table: its declaration registers it stamped (``keys.stamped_table``), as a product table's does
_STAMPED = TableSchema(
    name="same_results_stamped",
    primary_key="k",
    columns=[Column("tenant", STRING_TYPE), Column("k", INT_TYPE), Column("v", STRING_TYPE)],
    row_epoch=ROW_STAMPED,
    l1_eviction=TEST_L1_TABLE,
)

#: its L3 shape as the seed meets it: the epoch NOT NULL with no default, so an unstamped row is refused
_STAMPED_DDL = (
    f"CREATE TABLE {_STAMPED.name} (tenant VARCHAR NOT NULL, k INTEGER NOT NULL, v VARCHAR, "
    f"{EPOCH_COLUMN} BIGINT NOT NULL, gone BOOLEAN NOT NULL DEFAULT false)"
)

_OLD_STAMPED = f"SELECT k, v, {EPOCH_COLUMN} FROM {_STAMPED.name} WHERE tenant = $1 ORDER BY k"


def _stamped_row(seed: Seeded) -> Mapping[str, Any]:
    return {"tenant": f"t{seed.tenant}", "k": seed.index, "v": None if seed.null else f"v{seed.index}"}


async def test_a_stamped_tables_rows_are_seeded_with_an_epoch_each(conn: _Duck) -> None:
    """Flips if a ``ROW_STAMPED`` table's seed names no epoch (the database refuses the row: its epoch is NOT NULL with
    no default), or if its rows share one rather than each its own fresh stamp, as each write's is; a row naming its own
    epoch keeps it."""
    await conn.execute(_STAMPED_DDL)
    answered = await assert_same_results(
        conn,
        table=_STAMPED.name,
        row=_stamped_row,
        old_sql=_OLD_STAMPED,
        params=["t0"],
        helper=_asking(_OLD_STAMPED, "t0"),
    )
    epochs = [dict(row)[EPOCH_COLUMN] for rows in answered for row in rows]
    assert epochs and all(epoch > 0 for epoch in epochs)
    assert len(set(epochs)) == len(epochs)

    answered = await assert_same_results(
        conn,
        table=_STAMPED.name,
        row=lambda seed: {**_stamped_row(seed), EPOCH_COLUMN: 7},
        old_sql=_OLD_STAMPED,
        params=["t0"],
        helper=_asking(_OLD_STAMPED, "t0"),
    )
    assert {dict(row)[EPOCH_COLUMN] for rows in answered for row in rows} == {7}


async def test_an_unstamped_tables_rows_take_no_epoch(conn: _Duck) -> None:
    """Flips if a table no stamped declaration names is seeded with an epoch column (it has none)."""
    await assert_same_results(conn, table="t", row=_row, old_sql=_OLD, params=["t0"], helper=_asking(_OLD, "t0"))
