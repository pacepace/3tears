"""DuckDB statements with their long list parameters bound through Arrow (:func:`run_bound`): the same rows, types and
plans as binding the Python values, without DuckDB's per-value pandas probe (seconds for a page of keys where pandas
is not installed)."""

from __future__ import annotations

import asyncio
import datetime as dt
import sys
import uuid
from collections.abc import Iterator
from decimal import Decimal
from typing import Any

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from sqlalchemy import MetaData

from threetears.core.cache.duckdb import DuckDBBackend, query_arrow
from threetears.core.cache.duckdb_binds import KEYS, run_bound, run_joined
from threetears.core.keyed import InlineDuckDB, ThreadedDuckDB, run_inline
from threetears.core.testing.calibrate import best_seconds_interleaved, budget_problem

#: keys a reports page binds at once is in the thousands; the copy follower's large commit bound 70,000
_REAL_KEYS = 10_000

#: rows the keyed table holds
_REAL_ROWS = 200_000

#: the threads a DuckDB point query runs on, for the speed ratio's CPU trust check (coordinator ruling)
_POINT_QUERY_THREADS = 2


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
    ("SELECT typeof($1)", [[dt.time(1, 2), dt.time(3, 4, 5, 6), dt.time(7), dt.time(8)]]),
    ("SELECT typeof($1)", [[dt.timedelta(seconds=i) for i in range(4)]]),
    ("SELECT k FROM t WHERE s = $1 AND k = ANY($2) ORDER BY k", ["s2", [1, 2, 3, 4]]),
    ("SELECT k FROM t WHERE k = ANY($2) AND s <> $1 AND k < $3 ORDER BY k", ["s2", [1, 2, 3, 4], 2**40]),
    ("SELECT k FROM t WHERE k = ANY($1) AND ($2 IS NULL) ORDER BY k", [[1, 2, 3, 4], None]),
    ("SELECT k FROM t WHERE k = ANY($1) AND k <> ALL($2) ORDER BY k", [[1, 2, 3, 4], [2]]),
    ("SELECT k FROM t WHERE k = ANY($1) AND $2 = $2 ORDER BY k", [[1, 2, 3, 4], Decimal("1.5")]),
    (
        "SELECT typeof($1), typeof($2), typeof($3), typeof($4), typeof($5), typeof($6), typeof($7), typeof($8)",
        [1, 2**40, 1.5, "a", True, b"b", _AT, dt.date(2026, 1, 1)],
    ),
    (
        "SELECT typeof($1), typeof($2), typeof($3), typeof($4), typeof($5), $1, $2, $3, $4, $5",
        [uuid.UUID(int=7), dt.time(1, 2), dt.timedelta(seconds=3), dt.datetime(2026, 1, 1), -(2**31)],
    ),
    ("SELECT k FROM t WHERE k >= $1 AND k <= $2 AND s <> $3 AND k <> $4 AND k <> $5 ORDER BY k", [3, 9, "s4", 5, 6]),
]


@pytest.mark.parametrize(("query", "params"), _CASES)
def test_a_statement_answers_what_binding_its_values_answers(
    conn: duckdb.DuckDBPyConnection, query: str, params: list[Any]
) -> None:
    """Flips if a list read through Arrow answers other rows, or is typed otherwise (INTEGER[] against BIGINT[]),
    than the same list bound value by value; or if a list Arrow cannot type as DuckDB does is not bound as before."""
    assert run_bound(conn, query, params, _fetch) == conn.execute(query, params).fetchall()


#: (key columns, the values of each): every kind of key run_joined registers through Arrow, and every kind it binds
#: as lists to unnest instead; read against ``t`` by (k, s), the composite key a pair of columns makes
_JOINED: list[Any] = [
    pytest.param([1, 5, 9, 700, 2000], ["s1", "s5", "nope", "s700", "s2000"], id="int-and-text"),
    pytest.param([1, 2**40, 3], ["s1", "s2", "s3"], id="past-32-bits"),
    pytest.param([None, 3, 4], ["s2", None, "s4"], id="nulls-match-nothing"),
    pytest.param([7, 7, 8], ["s7", "s7", "s8"], id="a-key-twice-answers-its-row-once"),
    pytest.param([1], ["s1"], id="one-key"),
    pytest.param([Decimal("1"), Decimal("2")], ["s1", "s2"], id="decimal-bound-as-lists"),
]


@pytest.mark.parametrize(("ks", "ss"), _JOINED)
def test_a_composite_key_s_relation_answers_what_unnesting_its_bound_lists_answers(
    conn: duckdb.DuckDBPyConnection, ks: list[Any], ss: list[Any]
) -> None:
    """Flips if a composite key read as a registered Arrow relation answers other rows than the same keys bound as
    lists and unnested (value by value), or a key Arrow cannot type as DuckDB does is not bound as before."""
    joined = f"SELECT k, s FROM t WHERE (k, s) IN ({KEYS}) ORDER BY k"
    unnested = "SELECT k, s FROM t WHERE (k, s) IN (SELECT unnest($1), unnest($2)) ORDER BY k"
    assert run_joined(conn, joined, [ks, ss], _fetch) == conn.execute(unnested, [ks, ss]).fetchall()


def test_a_composite_key_s_relation_is_gone_once_the_answer_is_read(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if a statement's keys stay registered on the connection after its answer, or a failing statement
    leaves them."""
    run_joined(conn, f"SELECT k FROM t WHERE (k, s) IN ({KEYS})", [[1, 2, 3, 4], ["s1", "s2", "s3", "s4"]], _fetch)
    with pytest.raises(duckdb.Error):
        run_joined(conn, f"SELECT nope FROM t WHERE (k, s) IN ({KEYS})", [[1], ["s1"]], _fetch)
    views = conn.execute("SELECT count(*) FROM duckdb_views() WHERE view_name = '_threetears_keys'").fetchall()
    assert views == [(0,)]


def test_a_composite_key_s_statement_must_name_its_keys_and_its_columns_agree_in_length(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    with pytest.raises(ValueError, match="names no"):
        run_joined(conn, "SELECT k FROM t", [[1], ["s1"]], _fetch)
    with pytest.raises(ValueError, match="differ in length"):
        run_joined(conn, f"SELECT k FROM t WHERE (k, s) IN ({KEYS})", [[1, 2], ["s1"]], _fetch)


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


# times against the host's quiet (best_seconds_interleaved): run apart, never beside other work (tests/timing_suite.py)
@pytest.mark.loop_timing
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
        baseline, took = asyncio.run(best_seconds_interleaved(bare, served, threads=_POINT_QUERY_THREADS))
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


def test_five_scalars_take_the_arrow_round_and_four_do_not(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if a statement of scalars only takes the Arrow round when its probes cost less than it (four scalars),
    or binds them value by value when they cost as much as a long list's (five)."""
    held = "getvariable('_threetears_bound') IS NULL"
    assert run_bound(conn, f"SELECT {held} FROM t WHERE k IN ($1, $2, $3, $4)", [1, 2, 3, 4], _fetch) == [(True,)] * 4
    five = f"SELECT {held} FROM t WHERE k IN ($1, $2, $3, $4, $5)"
    assert run_bound(conn, five, [1, 2, 3, 4, 5], _fetch) == [(False,)] * 5


class _PandasTries:
    """a finder on ``sys.meta_path`` that counts each ``import pandas`` reaching the path search, finding nothing."""

    def __init__(self) -> None:
        self.tries = 0

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
        if name == "pandas" or name.startswith("pandas."):
            self.tries += 1


def test_a_keyed_read_binds_no_python_value(conn: duckdb.DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flips if a keyed read binds any of its values through DuckDB's Python layer, which tries ``import pandas``
    twice a value and, pandas absent, searches the whole import path each time: a scope bound beside a page of keys,
    or keys Arrow types as DuckDB does but the Arrow round once refused (UUIDs: 100,000 of them took 46 s of loop CPU
    and 200,002 failed imports on a laptop, 0.2 s and none through Arrow). The absent-pandas entry in sys.modules is
    taken out while it runs, so each try a bound value makes reaches the finders and is counted."""
    if sys.modules.get("pandas") is not None:
        pytest.skip("pandas is imported here: DuckDB's probe finds it and never searches the path")
    monkeypatch.delitem(sys.modules, "pandas", raising=False)
    conn.execute("CREATE TABLE ids AS SELECT uuid() AS u, i AS k FROM range(2000) r(i)")
    uuids = [row[0] for row in conn.execute("SELECT u FROM ids ORDER BY k").fetchall()]
    reads: list[tuple[str, list[Any]]] = [
        ("SELECT k FROM t WHERE s <> $1 AND k = ANY($2)", ["s2", list(range(1000))]),
        ("SELECT k FROM ids WHERE u = ANY($1)", [uuids]),
        ("SELECT k FROM ids WHERE u = ANY($1) AND k >= $2", [uuids[:10], 3]),
        ("SELECT k FROM t WHERE k = ANY($1) AND ts >= $2 AND d <= $3", [list(range(500)), _AT, dt.date(2027, 1, 1)]),
    ]
    run_bound(conn, *reads[0], _fetch)  # pyarrow's own first look for pandas, once a process, is not the read's
    finder = _PandasTries()
    sys.meta_path.insert(0, finder)
    try:
        answered = [len(run_bound(conn, query, params, _fetch)) for query, params in reads]
    finally:
        sys.meta_path.remove(finder)
    assert answered == [999, 2000, 7, 500]
    assert finder.tries == 0


def test_an_absent_pandas_is_never_searched_for_on_a_scalar_bind(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if, pandas not installed, binding a scalar (what the Arrow round leaves bound) makes DuckDB's
    ``import pandas`` search the import path: duckdb_binds marks pandas absent in sys.modules once, so the import
    fails at once (a single-scalar bind 0.79 -> 0.26 ms on a laptop, a 100-value list 55.6 -> 6.4 ms)."""
    if sys.modules.get("pandas") is not None:
        pytest.skip("pandas is installed here: DuckDB finds it and never searches the path")
    finder = _PandasTries()
    sys.meta_path.insert(0, finder)
    try:
        answered = run_bound(conn, "SELECT s FROM t WHERE k = $1", [5], _fetch)
    finally:
        sys.meta_path.remove(finder)
    assert answered == [("s5",)]
    assert finder.tries == 0
