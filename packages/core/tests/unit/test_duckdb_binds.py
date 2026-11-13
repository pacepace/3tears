"""DuckDB statements with their long list parameters bound through Arrow (:func:`run_bound`): the same rows, types and
plans as binding the Python values, without DuckDB's per-value pandas probe (seconds for a page of keys where pandas
is not installed)."""

from __future__ import annotations

import asyncio
import datetime as dt
import uuid
from collections.abc import Iterator
from decimal import Decimal
from typing import Any

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from sqlalchemy import MetaData

from threetears.core.cache.duckdb import DuckDBBackend, query_arrow
from threetears.core.cache.duckdb_binds import run_bound
from threetears.core.keyed import InlineDuckDB, ThreadedDuckDB, run_inline
from threetears.core.testing.calibrate import best_seconds_interleaved, budget_problem

#: keys a reports page binds at once is in the thousands; the copy follower's large commit bound 70,000
_REAL_KEYS = 10_000

#: rows the keyed table holds
_REAL_ROWS = 200_000


@pytest.fixture
def conn() -> Iterator[duckdb.DuckDBPyConnection]:
    db = duckdb.connect()
    db.execute("CREATE TABLE t (k BIGINT PRIMARY KEY, s VARCHAR, ts TIMESTAMPTZ, d DATE, f DOUBLE, ok BOOLEAN, b BLOB)")
    db.execute(
        "INSERT INTO t SELECT i, 's' || i, TIMESTAMPTZ '2026-01-01 00:00:00+00' + to_seconds(i), "
        "DATE '2026-01-01' + (i % 300)::INTEGER, i / 4, i % 2 = 0, ('b' || i)::BLOB FROM range(1000) r(i)"
    )
    yield db
    db.close()


def _fetch(result: Any) -> list[tuple[Any, ...]]:
    return list(result.fetchall())


_AT = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)

#: (statement, parameters): every kind of list run_bound takes through Arrow, and every kind it leaves bound
_CASES: list[tuple[str, list[Any]]] = [
    ("SELECT k FROM t WHERE k = ANY($1) ORDER BY k", [[1, 5, 9, 700, 2000]]),
    ("SELECT k FROM t WHERE k = ANY($1) ORDER BY k", [[1, 5, 9, 2**40]]),
    ("SELECT k FROM t WHERE k = ANY($1) ORDER BY k", [(3, 4, 5, 6)]),
    ("SELECT k FROM t WHERE k = ANY($1) ORDER BY k", [[None, 3, None, 4]]),
    ("SELECT k FROM t WHERE k = ANY($1) ORDER BY k", [[None, None, None, None]]),
    ("SELECT k FROM t WHERE s = ANY($1) AND k > $2 ORDER BY k", [["s1", "s2", "s30", "nope"], 1]),
    ("SELECT k FROM t WHERE k > $2 AND s = ANY($1) AND k < $3 ORDER BY k", [["s1", "s2", "s30", "s31"], 1, 31]),
    ("SELECT k FROM t WHERE ts = ANY($1) ORDER BY k", [[_AT + dt.timedelta(seconds=i) for i in (0, 4, 8, 12)]]),
    ("SELECT k FROM t WHERE d = ANY($1) AND k < 20 ORDER BY k", [[dt.date(2026, 1, i) for i in (1, 2, 3, 4)]]),
    ("SELECT k FROM t WHERE f = ANY($1) ORDER BY k", [[0.25, 0.5, 2.0, 2.25]]),
    ("SELECT k FROM t WHERE ok = ANY($1) AND k < 6 ORDER BY k", [[True, True, True, True]]),
    ("SELECT k FROM t WHERE b = ANY($1) ORDER BY k", [[b"b1", b"b2", b"b3", b"zz"]]),
    ("SELECT k FROM t WHERE k = ANY(?) AND s <> ? ORDER BY k", [[1, 2, 3, 4], "s2"]),
    ("SELECT k FROM t WHERE k = ANY($1) OR k = ANY($1) ORDER BY k", [[7, 8, 9, 10]]),
    ("SELECT '$1 ?' AS lit, \"k\" FROM t WHERE k = ANY($1) ORDER BY k -- $2 ?", [[7, 8, 9, 10]]),
    ("SELECT k FROM t WHERE k = ANY($1) ORDER BY k", [[1, 2]]),
    ("SELECT k FROM t WHERE k = $1", [5]),
    ("SELECT count(*) FROM t WHERE k = ANY($1)", [[]]),
    ("SELECT typeof($1), typeof($2), typeof($3)", [[1, 2, 3, 4], [1, 2, 3, 2**40], ["a", "b", "c", "d"]]),
    ("SELECT typeof($1), typeof($2)", [[_AT] * 4, [dt.datetime(2026, 1, 1)] * 4]),
    ("SELECT typeof($1), typeof($2), typeof($3)", [[None] * 4, [dt.date(2026, 1, 1)] * 4, [1.5] * 4]),
    ("SELECT typeof($1)", [[uuid.UUID(int=i) for i in range(4)]]),
    ("SELECT typeof($1)", [[Decimal("1.5"), Decimal("2.25"), Decimal("3"), Decimal("4")]]),
    ("SELECT typeof($1)", [[[1], [2, 3], [4], [5]]]),
    ("SELECT typeof($1)", [[1, 2, 3, 2**70]]),
    ("SELECT typeof($1)", [[True, 2, 3, 4]]),
    ("SELECT typeof($1)", [[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]]),
]


@pytest.mark.parametrize(("query", "params"), _CASES)
def test_a_statement_answers_what_binding_its_values_answers(
    conn: duckdb.DuckDBPyConnection, query: str, params: list[Any]
) -> None:
    """Flips if a list read through Arrow answers other rows, or is typed otherwise (INTEGER[] against BIGINT[]),
    than the same list bound value by value; or if a list Arrow cannot type as DuckDB does is not bound as before."""
    assert run_bound(conn, query, params, _fetch) == conn.execute(query, params).fetchall()


def test_the_variable_is_gone_once_the_answer_is_read(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if a statement's keys stay held on the connection after its answer (a page of keys per connection)."""
    run_bound(conn, "SELECT k FROM t WHERE k = ANY($1)", [list(range(50))], _fetch)
    assert conn.execute("SELECT getvariable('_threetears_bound')").fetchall() == [(None,)]
    tables = conn.execute("SELECT count(*) FROM duckdb_views() WHERE view_name = '_threetears_bound_row'").fetchall()
    assert tables == [(0,)]


def test_a_failing_statement_raises_its_own_error(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if a statement's failure inside a transaction surfaces as the cleanup's error (the transaction is
    aborted, so resetting the variable fails) instead of its own."""
    conn.execute("BEGIN TRANSACTION")
    with pytest.raises(duckdb.ConstraintException):
        run_bound(conn, "INSERT INTO t (k) SELECT unnest($1)", [[1, 2, 3, 4]], _fetch)
    conn.execute("ROLLBACK")
    assert run_bound(conn, "SELECT count(*) FROM t WHERE k = ANY($1)", [[1, 2, 3, 4]], _fetch) == [(4,)]


def test_a_write_in_the_caller_s_transaction_rolls_back_with_it(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if the Arrow round (register, variable) commits or breaks the caller's open transaction."""
    conn.execute("BEGIN TRANSACTION")
    assert run_inline(InlineDuckDB(conn).execute("DELETE FROM t WHERE k = ANY($1)", list(range(10, 20)))) == "DELETE 10"
    conn.execute("ROLLBACK")
    assert conn.execute("SELECT count(*) FROM t").fetchall() == [(1000,)]


def test_threaded_and_backend_reads_bind_through_arrow_too(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if ThreadedDuckDB, duckdb_arrow (fetch_arrow) or the cache's query_arrow answer other rows for a long
    list than binding it does."""
    keys = list(range(0, 400, 7))
    by_hand = conn.execute("SELECT k, s FROM t WHERE k = ANY($1) ORDER BY k", [keys]).fetchall()
    threaded = asyncio.run(ThreadedDuckDB(conn).fetch("SELECT k, s FROM t WHERE k = ANY($1) ORDER BY k", keys))
    assert [(r["k"], r["s"]) for r in threaded] == by_hand
    arrow = run_inline(
        InlineDuckDB(conn).fetch_arrow("SELECT k, s FROM t WHERE k = ANY($1) ORDER BY k", keys, types=None)
    )
    assert list(zip(arrow.column("k").to_pylist(), arrow.column("s").to_pylist(), strict=True)) == by_hand
    queried = query_arrow(conn, "SELECT k, s FROM t WHERE k = ANY(?) ORDER BY k", [keys])
    assert list(zip(queried.column("k").to_pylist(), queried.column("s").to_pylist(), strict=True)) == by_hand
    backend = DuckDBBackend()
    backend.initialize(MetaData())
    try:
        backend.execute_query("CREATE TABLE u AS SELECT range AS k FROM range(100)")
        assert backend.execute_query("SELECT count(*) AS n FROM u WHERE k = ANY(?)", ([1, 2, 3, 4, 500],)) == [{"n": 4}]
    finally:
        backend.reset()


def test_a_long_list_binds_at_real_size_within_twice_the_statement_on_keys_duckdb_holds() -> None:
    """Flips if a page of keys bound to an InlineDuckDB statement is bound value by value: DuckDB tries
    ``import pandas`` twice per value and, pandas absent, each try searches the whole import path (10,000 keys took
    2.7 s on a laptop, 14.7 s on a pool runner). Baseline: the same statement with its keys already in DuckDB, put
    there by hand from the same Python list (one Arrow table, one variable), its rows made the same dicts: the
    statement's own cost, so the budget bounds what binding the keys adds."""
    db = duckdb.connect()
    db.execute("CREATE TABLE r (id BIGINT PRIMARY KEY, v VARCHAR)")
    db.execute(f"INSERT INTO r SELECT i, 'v' || i FROM range({_REAL_ROWS}) t(i)")
    ids = list(range(0, _REAL_KEYS * 13, 13))
    inline = InlineDuckDB(db)

    async def bare() -> list[dict[str, Any]]:
        db.register("_keys", pa.table({"id": ids}))
        try:
            db.execute("SET VARIABLE keys = (SELECT list(id) FROM _keys)")
        finally:
            db.unregister("_keys")
        result = db.execute("SELECT r.id, r.v FROM r WHERE r.id = ANY(getvariable('keys'))")
        names = [column[0] for column in result.description]
        rows = [dict(zip(names, row, strict=True)) for row in result.fetchall()]
        db.execute("RESET VARIABLE keys")
        return rows

    async def served() -> list[dict[str, Any]]:
        return await inline.fetch("SELECT r.id, r.v FROM r WHERE r.id = ANY($1)", ids)

    try:
        assert sorted(r["id"] for r in run_inline(served())) == sorted(r["id"] for r in run_inline(bare())) == ids
        baseline, took = asyncio.run(best_seconds_interleaved(bare, served))
    finally:
        db.close()
    print(
        f"{_REAL_KEYS} keys: InlineDuckDB.fetch {took * 1000:.1f} ms, keys put in DuckDB by hand {baseline * 1000:.1f} ms"
    )
    problem = budget_problem(f"an InlineDuckDB read of {_REAL_KEYS} bound keys", took, baseline)
    assert problem is None, problem


def test_a_short_list_is_bound_as_it_is(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if the Arrow round is taken for a list too short to gain from it (3 values: it costs more than their
    probes), or not taken from 4 values on."""
    query = "SELECT getvariable('_threetears_bound') IS NULL FROM t WHERE k = ANY($1)"
    assert run_bound(conn, query, [list(range(3))], _fetch) == [(True,)] * (3)
    assert run_bound(conn, query, [list(range(4))], _fetch) == [(False,)] * 4
