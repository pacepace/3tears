"""a DuckDB connection driven inline (:class:`InlineDuckDB`, :func:`run_inline`): the helpers' statements on the
caller's own connection, in its transaction, seeing its registered relations, with no event loop."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any
from collections.abc import Iterator

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from threetears.core.keyed import (
    ANY_ROWS_CAP,
    Fragment,
    InlineDuckDB,
    KeyBatch,
    Sort,
    any_rows,
    apply_stage,
    keyset_rows,
    read_keyed,
    run_inline,
)
from threetears.core.testing.same_results import Seeded, assert_same_results


@pytest.fixture
def conn() -> Iterator[duckdb.DuckDBPyConnection]:
    db = duckdb.connect()
    db.execute("CREATE TABLE t (a VARCHAR, b BIGINT, v INTEGER, PRIMARY KEY (a, b))")
    db.execute("INSERT INTO t SELECT 'k' || (i % 4), i, i FROM range(40) r(i)")
    yield db
    db.close()


def test_a_helper_runs_in_the_caller_s_transaction_and_sees_its_relation(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if a helper's statements run beside the caller's transaction (a cursor: its rollback would not undo
    them) or cannot see a relation the caller registered."""
    conn.register("chunk", pa.table({"a": ["k1", "k1"], "b": [1, 999], "v": [-1, 7]}))
    conn.execute("BEGIN TRANSACTION")
    applied = run_inline(apply_stage(InlineDuckDB(conn), "t", "chunk", key=["a", "b"], columns=["a", "b", "v"]))
    assert (applied.deleted, applied.written) == (1, 2)
    assert conn.execute("SELECT v FROM t WHERE a = 'k1' AND b IN (1, 999) ORDER BY b").fetchall() == [(-1,), (7,)]
    conn.execute("ROLLBACK")
    assert conn.execute("SELECT count(*), sum(v) FROM t").fetchall() == [(40, sum(range(40)))]


def test_an_append_runs_inline_and_writes_what_the_insert_it_replaces_writes(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if apply_stage(mode=append) cannot run on an inline DuckDB, deletes a row, or leaves the table other than
    the plain ``INSERT INTO t SELECT ... FROM stage`` it replaces: the stage's keys are keys each case's table holds
    (re-staged), and some are staged twice."""
    conn.execute("CREATE TABLE seen (a VARCHAR, b BIGINT, v INTEGER)")
    conn.execute("CREATE TABLE stage (a VARCHAR, b BIGINT, v INTEGER)")
    conn.execute(
        "INSERT INTO stage SELECT 'k' || (i % 2), i, -i FROM range(60) r(i) WHERE i % 2 = 0 "
        "UNION ALL SELECT 'k' || (i % 2), i, -i FROM range(60) r(i) WHERE i % 3 = 0"
    )

    def row(seed: Seeded) -> dict[str, object]:
        return {"a": f"k{seed.tenant}", "b": seed.index, "v": None if seed.null else seed.tie}

    async def helper(c: Any) -> Any:
        staged = await c.fetch("SELECT a, b, v FROM stage")
        await apply_stage(c, "seen", "stage", key=["a", "b"], columns=["a", "b", "v"], mode="append")
        return staged

    run_inline(
        assert_same_results(
            InlineDuckDB(conn),
            table="seen",
            row=row,
            old_sql="INSERT INTO seen (a, b, v) SELECT a, b, v FROM stage RETURNING a, b, v",
            helper=helper,
            ordered=False,
            state_sql="SELECT a, b, v FROM seen",
        )
    )


def test_a_distinct_difference_writes_what_the_except_it_replaces_writes(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if apply_stage(mode=difference, distinct=True) writes a row staged twice twice, or leaves the table
    other than the set difference it replaces (``INSERT INTO t SELECT ... FROM stage EXCEPT SELECT ... FROM t``),
    over every seed: new keys, each staged twice, one of them identical."""
    conn.execute("CREATE TABLE seen (a VARCHAR, b BIGINT, v INTEGER)")
    conn.execute("CREATE TABLE stage (a VARCHAR, b BIGINT, v INTEGER)")
    conn.execute(
        "INSERT INTO stage SELECT 'k' || (i % 2), 1000 + i, i FROM range(20) r(i) "
        "UNION ALL SELECT 'k' || (i % 2), 1000 + i, i FROM range(20) r(i) WHERE i % 3 = 0"
    )

    def row(seed: Seeded) -> dict[str, object]:
        return {"a": f"k{seed.tenant}", "b": seed.index, "v": None if seed.null else seed.tie}

    async def helper(c: Any) -> Any:
        written = await c.fetch("SELECT DISTINCT a, b, v FROM stage")
        await apply_stage(c, "seen", "stage", key=["a", "b"], columns=["a", "b", "v"], mode="difference", distinct=True)
        return written

    run_inline(
        assert_same_results(
            InlineDuckDB(conn),
            table="seen",
            row=row,
            old_sql="INSERT INTO seen (a, b, v) SELECT a, b, v FROM stage EXCEPT SELECT a, b, v FROM seen RETURNING a, b, v",
            helper=helper,
            ordered=False,
            state_sql="SELECT a, b, v FROM seen",
        )
    )


def test_its_transaction_commits_on_a_clean_exit_and_rolls_back_on_an_error(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if ``async with db.transaction()`` keeps a failed block's writes or drops a clean one's."""
    db = InlineDuckDB(conn)

    async def write(fail: bool) -> None:
        async with db.transaction():
            await db.execute("DELETE FROM t WHERE a = $1", "k2")
            await db.executemany("INSERT INTO t VALUES ($1, $2, $3)", [("k9", 1, 1), ("k9", 2, 2)])
            if fail:
                raise LookupError("the block failed")

    with pytest.raises(LookupError):
        run_inline(write(True))
    assert conn.execute("SELECT count(*) FROM t WHERE a IN ('k2', 'k9')").fetchall() == [(10,)]
    run_inline(write(False))
    assert conn.execute("SELECT a, count(*) FROM t WHERE a IN ('k2', 'k9') GROUP BY a").fetchall() == [("k9", 2)]


def test_reads_answer_as_the_threaded_reader_does(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if an inline keyed read or drain answers other rows than the plain statement."""
    db = InlineDuckDB(conn)
    rows = run_inline(read_keyed(db, "t", keys=KeyBatch(("a",), [("k3",)]), columns=["a", "b"]))
    assert sorted((row["a"], row["b"]) for row in rows) == sorted(
        conn.execute("SELECT a, b FROM t WHERE a = 'k3'").fetchall()
    )
    drained = run_inline(
        keyset_rows(db, "t", columns=["a", "b"], by={}, order=[Sort("a"), Sort("b", cast="bigint")], page=6)
    )
    assert [(row["a"], row["b"]) for row in drained] == conn.execute("SELECT a, b FROM t ORDER BY a, b").fetchall()


def test_a_coroutine_that_waits_on_io_is_refused_not_left_half_run() -> None:
    """Flips if run_inline answers a coroutine that awaits a real future (or drives a non-coroutine), or leaves it
    suspended rather than closed; a bare ``sleep(0)`` is stepped past."""
    loop = asyncio.new_event_loop()
    try:
        future = loop.create_future()
        reached: list[str] = []

        async def waits() -> int:
            await asyncio.sleep(0)
            reached.append("before")
            await future
            reached.append("after")
            return 1

        coroutine = waits()
        with pytest.raises(RuntimeError, match="never waits on IO"):
            run_inline(coroutine)
        assert reached == ["before"]
        assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
        with pytest.raises(TypeError, match="drives a coroutine"):
            run_inline(future)  # type: ignore[arg-type]
    finally:
        loop.close()


def test_the_equivalence_harness_runs_through_it(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if the harness (seed in a rolled-back transaction, ``executemany``, both sides) cannot run on an inline
    DuckDB, or a helper read it compares answers other rows than the statement it replaces."""

    def row(seed: Seeded) -> dict[str, object]:
        return {"a": f"k{seed.tenant}", "b": seed.index, "v": None if seed.null else seed.tie}

    run_inline(
        assert_same_results(
            InlineDuckDB(conn),
            table="t",
            row=row,
            old_sql="SELECT a, b, v FROM t WHERE a = $1 ORDER BY b",
            params=["k0"],
            helper=lambda c: keyset_rows(
                c, "t", columns=["a", "b", "v"], by={"a": "k0"}, order=[Sort("b", cast="bigint")], page=5
            ),
            order_by=["b"],
        )
    )
    assert conn.execute("SELECT count(*) FROM t").fetchall() == [(40,)]


def test_a_bounded_look_takes_at_most_its_limit_of_the_rows_it_names(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if any_rows reads past its limit, a row outside ``by``/``where``, or takes a limit outside 1..cap; and,
    with a limit past the rows held, if it answers other rows than the statement it replaces (a set: no order)."""
    db = InlineDuckDB(conn)
    rows = run_inline(any_rows(db, "t", ["a", "b"], limit=3, by={"a": "k1"}, where=[Fragment('t."b" > $1', (5,))]))
    assert len(rows) == 3 and all(row["a"] == "k1" and row["b"] > 5 for row in rows)
    for limit in (0, ANY_ROWS_CAP + 1):
        with pytest.raises(ValueError, match="bounded look"):
            run_inline(any_rows(db, "t", ["a"], limit=limit))

    def row(seed: Seeded) -> dict[str, object]:
        return {"a": f"k{seed.tenant}", "b": seed.index, "v": None if seed.null else seed.tie}

    run_inline(
        assert_same_results(
            db,
            table="t",
            row=row,
            old_sql="SELECT a, b, v FROM t WHERE a = $1 LIMIT 1000",
            params=["k0"],
            helper=lambda c: any_rows(c, "t", ["a", "b", "v"], limit=1000, by={"a": "k0"}),
            ordered=False,
        )
    )
