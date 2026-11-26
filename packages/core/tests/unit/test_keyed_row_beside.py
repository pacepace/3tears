"""a point row read beside other rows in one statement (:mod:`threetears.core.keyed.point`): ``read_row(beside=)``,
a table-less derived row read beside the table's.

Correctness runs on an in-process DuckDB (the helpers' SQL is one spelling for PostgreSQL, YugabyteDB and DuckDB).
"""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
import pytest

from threetears.core.keyed import Derived, Selected, ThreadedDuckDB, read_row
from threetears.interfaces.keyed import read_row_statement


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    """``ring``: one row naming the current slot and a floor."""
    conn = duckdb.connect()
    conn.execute("CREATE TABLE ring (id INTEGER PRIMARY KEY, current_ring INTEGER, floor_at INTEGER)")
    conn.execute("INSERT INTO ring VALUES (1, 1, 10)")
    yield conn
    conn.close()


async def test_a_derived_row_beside_is_read_before_the_select_list(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if read_row's ``beside=`` row is not read beside the table's (``b``), or a read with none changes."""
    beside = Derived("SELECT 41 + 1 AS was OFFSET 0")
    row = await read_row(
        ThreadedDuckDB(db), "ring", by={"id": 1}, columns=[Selected("b.was", "was"), "current_ring"], beside=beside
    )
    assert row == {"was": 42, "current_ring": 1}
    sql, _ = read_row_statement("ring", by={"id": 1}, columns=["current_ring"], beside=beside)
    assert ' FROM (SELECT 41 + 1 AS was OFFSET 0) AS b CROSS JOIN "ring" AS t WHERE ' in sql
    plain, _ = read_row_statement("ring", by={"id": 1}, columns=["current_ring"])
    assert plain == 'SELECT t."current_ring" FROM "ring" AS t WHERE t."id" = $1 LIMIT 1'
