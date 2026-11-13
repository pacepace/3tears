"""``DuckDBBackend.select_batch``: a page of keys bound as one list a key column (through Arrow, by ``run_bound``), never a
placeholder a value -- the same rows as the placeholder-a-value statement it replaced, without DuckDB's per-value
pandas probe (where pandas is not installed, each bound value searches the whole import path)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table

from threetears.core.cache.duckdb import DuckDBBackend
from threetears.core.testing.calibrate import best_seconds_interleaved, budget_problem

#: keys one call names at real size: a collection's get_many of a page
_REAL_KEYS = 1_000

#: rows the table holds at real size
_REAL_ROWS = 200_000


def _metadata() -> MetaData:
    metadata = MetaData()
    Table("by_text", metadata, Column("id", String, primary_key=True), Column("v", Integer), Column("note", String))
    Table("by_int", metadata, Column("id", Integer, primary_key=True), Column("v", Integer))
    Table(
        "by_pair",
        metadata,
        Column("a", String, primary_key=True),
        Column("b", Integer, primary_key=True),
        Column("v", Integer),
    )
    return metadata


@pytest.fixture
def backend() -> Iterator[DuckDBBackend]:
    made = DuckDBBackend()
    made.initialize(_metadata())
    made.execute_query("INSERT INTO by_text SELECT 't' || i, i, 'n' || i FROM range(500) r(i)")
    made.execute_query("INSERT INTO by_int SELECT i, i * 3 FROM range(500) r(i)")
    made.execute_query("INSERT INTO by_pair SELECT 'a' || (i % 50), i // 50, i FROM range(500) r(i)")
    try:
        yield made
    finally:
        made.reset()


def _placeholder_a_value(
    backend: DuckDBBackend, table: str, ids: list[Any], pk: tuple[str, ...]
) -> list[dict[str, Any]]:
    """the statement select_batch ran before: ``pk IN (?, ?, ...)``, or an OR of ``(a = ? AND b = ?)`` a key."""
    if len(pk) == 1:
        where = f'"{pk[0]}" IN ({", ".join("?" for _ in ids)})'
        params: list[Any] = list(ids)
    else:
        one = " AND ".join(f'"{c}" = ?' for c in pk)
        where = " OR ".join(f"({one})" for _ in ids)
        params = [value for key in ids for value in key]
    return backend.execute_query(f'SELECT * FROM "{table}" WHERE {where}', tuple(params))  # noqa: S608


def _sorted(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return sorted(tuple(sorted(row.items())) for row in rows)


_CASES: list[tuple[str, list[Any], tuple[str, ...]]] = [
    ("by_text", ["t1", "t7", "t499"], ("id",)),
    ("by_text", ["t1", "t7", "t8", "t9", "t300", "nope"], ("id",)),
    ("by_text", ["t2", "t2", "t3", "t3", "t4"], ("id",)),
    ("by_text", ["nope", "none", "never", "gone"], ("id",)),
    ("by_int", [0, 1, 2], ("id",)),
    ("by_int", [0, 9, 99, 499, 500, 2**40], ("id",)),
    ("by_pair", [("a1", 0), ("a2", 9)], ("a", "b")),
    ("by_pair", [("a1", 0), ("a2", 9), ("a49", 9), ("a3", 3), ("a1", 0), ("nope", 1), ("a1", 99)], ("a", "b")),
]


@pytest.mark.parametrize(("table", "ids", "pk"), _CASES)
def test_a_batch_answers_the_rows_the_placeholder_a_value_statement_answered(
    backend: DuckDBBackend, table: str, ids: list[Any], pk: tuple[str, ...]
) -> None:
    """Flips if any key kind (text, integer, a composite key; short and long lists, repeats, misses) answers other
    rows than the statement select_batch ran before."""
    primary_key: str | tuple[str, ...] = pk[0] if len(pk) == 1 else pk
    assert _sorted(backend.select_batch(table, ids, primary_key)) == _sorted(
        _placeholder_a_value(backend, table, ids, pk)
    )


def test_named_columns_are_the_only_ones_answered(backend: DuckDBBackend) -> None:
    rows = backend.select_batch("by_text", ["t1", "t2", "t3", "t4", "t5"], columns=["id", "note"])
    assert sorted((r["id"], r["note"]) for r in rows) == [(f"t{i}", f"n{i}") for i in range(1, 6)]
    assert all(set(r) == {"id", "note"} for r in rows)


def _real_size(backend: DuckDBBackend) -> list[str]:
    backend.execute_query(f"INSERT INTO by_text SELECT 'r' || i, i, 'n' || i FROM range({_REAL_ROWS}) r(i)")
    return [f"r{i}" for i in range(0, _REAL_KEYS * 97, 97)]


def test_a_page_of_keys_is_read_within_twice_the_statement_on_keys_duckdb_holds(backend: DuckDBBackend) -> None:
    """Flips if select_batch binds its keys a placeholder a value: DuckDB tries ``import pandas`` twice per value and,
    pandas absent, each try searches the whole import path (1,000 keys ~320 ms on a laptop against ~4.5 ms). Baseline:
    the same rows read with the keys already in DuckDB, put there by hand from the same Python list (one Arrow table,
    one variable), made the same dicts: the statement's own cost, so the budget bounds what binding the keys adds."""
    ids = _real_size(backend)
    db = backend.get_connection()

    async def bare() -> list[dict[str, Any]]:
        db.register("_keys", pa.table({"id": ids}))
        try:
            db.execute("SET VARIABLE keys = (SELECT list(id) FROM _keys)")
        finally:
            db.unregister("_keys")
        result = db.execute("SELECT * FROM by_text WHERE id = ANY(getvariable('keys'))")
        names = [column[0] for column in result.description]
        rows = [dict(zip(names, row, strict=True)) for row in result.fetchall()]
        db.execute("RESET VARIABLE keys")
        return rows

    async def served() -> list[dict[str, Any]]:
        return backend.select_batch("by_text", ids)

    assert sorted(r["id"] for r in asyncio.run(served())) == sorted(r["id"] for r in asyncio.run(bare())) == sorted(ids)
    baseline, took = asyncio.run(best_seconds_interleaved(bare, served))
    print(f"{_REAL_KEYS} keys: select_batch {took * 1000:.1f} ms, keys put in DuckDB by hand {baseline * 1000:.1f} ms")
    problem = budget_problem(f"select_batch of {_REAL_KEYS} keys", took, baseline)
    assert problem is None, problem


def test_a_page_of_composite_keys_is_read_within_twice_the_join_on_keys_duckdb_holds(backend: DuckDBBackend) -> None:
    """Flips if a composite key's page is an OR of one ``(a = ? AND b = ?)`` a key (300 keys ~490 ms against ~4 ms):
    baseline is the semi-join on the keys put in DuckDB by hand."""
    backend.execute_query(f"INSERT INTO by_pair SELECT 'p' || (i % 1000), i // 1000, i FROM range({_REAL_ROWS}) r(i)")
    keys = [(f"p{i % 1000}", (i * 7) % 200) for i in range(300)]
    db = backend.get_connection()

    async def bare() -> list[dict[str, Any]]:
        db.register("_keys", pa.table({"a": [k[0] for k in keys], "b": pa.array([k[1] for k in keys], pa.int32())}))
        try:
            result = db.execute("SELECT * FROM by_pair WHERE (a, b) IN (SELECT a, b FROM _keys)")
            names = [column[0] for column in result.description]
            rows = [dict(zip(names, row, strict=True)) for row in result.fetchall()]
        finally:
            db.unregister("_keys")
        return rows

    async def served() -> list[dict[str, Any]]:
        return backend.select_batch("by_pair", keys, ("a", "b"))

    assert _sorted(asyncio.run(served())) == _sorted(asyncio.run(bare()))
    baseline, took = asyncio.run(best_seconds_interleaved(bare, served))
    print(
        f"{len(keys)} composite keys: select_batch {took * 1000:.1f} ms, keys joined by hand {baseline * 1000:.1f} ms"
    )
    problem = budget_problem(f"select_batch of {len(keys)} composite keys", took, baseline)
    assert problem is None, problem
