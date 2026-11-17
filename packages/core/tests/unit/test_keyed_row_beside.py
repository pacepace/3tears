"""a point row read beside other rows in one statement (:mod:`threetears.core.keyed.point`): ``read_row(beside=)``,
a table-less derived row read beside the table's, and ``read_row_beside``, a row and the rows of a step its columns
decide.

Correctness runs on an in-process DuckDB (the helpers' SQL is one spelling for PostgreSQL, YugabyteDB and DuckDB); the
PostgreSQL / YugabyteDB half is the change-note ring's (``packages/core/tests/integration/test_noted_rows_scale_live.py``,
``test_noted_keys_page_live.py``), the two reads these were made for.
"""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
import pytest

from threetears.core.keyed import (
    Derived,
    Fragment,
    Selected,
    Step,
    ThreadedDuckDB,
    distinct,
    read_row,
    read_row_beside,
    scan,
    union,
)
from threetears.interfaces.keyed import read_row_beside_statement, read_row_statement


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    """``ring``: one row naming the current slot and a floor; ``slot_0`` / ``slot_1``: rows by name, at a time."""
    conn = duckdb.connect()
    conn.execute("CREATE TABLE ring (id INTEGER PRIMARY KEY, current_ring INTEGER, floor_at INTEGER)")
    conn.execute("INSERT INTO ring VALUES (1, 1, 10)")
    for slot in (0, 1):
        conn.execute(f"CREATE TABLE slot_{slot} (name VARCHAR, seen_at INTEGER, v VARCHAR)")
        conn.execute(
            f"INSERT INTO slot_{slot} SELECT 'n' || (i % 3), i, 's{slot}-' || (i % 4) FROM range(30) r(i)"  # noqa: S608
        )
    yield conn
    conn.close()


#: the ring row's columns the step reads
_RING = ["current_ring", "floor_at"]


def _step(name: str) -> Step:
    legs = [
        scan(
            f"slot_{slot}",
            ["v"],
            where=[
                Fragment("r.current_ring = $1", (slot,)),
                Fragment("t.name = $1 AND t.seen_at >= r.floor_at", (name,)),
            ],
        )
        for slot in (0, 1)
    ]
    return distinct(union(*legs, keep_all=True), ["v"])


async def test_a_row_beside_a_step_reads_the_rows_its_columns_decide(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if the step's rows are not the ones the row's columns decide (its current slot, its floor), or the row is
    not answered once with no step row when the step holds none, or at all when the pins name none."""
    conn = ThreadedDuckDB(db)
    rows = await read_row_beside(
        conn, "ring", by={"id": 1}, columns=_RING, beside=_step("n1"), limit=10, present="seen"
    )
    want = {
        row[0] for row in db.execute("SELECT DISTINCT v FROM slot_1 WHERE name = 'n1' AND seen_at >= 10").fetchall()
    }
    assert {row["v"] for row in rows} == want and all(row["seen"] is True for row in rows)
    assert all(row["current_ring"] == 1 and row["floor_at"] == 10 for row in rows)
    assert len(rows) == len(want)
    alone = await read_row_beside(
        conn, "ring", by={"id": 1}, columns=_RING, beside=_step("none"), limit=10, present="seen"
    )
    assert [(row["current_ring"], row["seen"], row["v"]) for row in alone] == [(1, None, None)]
    missing = await read_row_beside(
        conn, "ring", by={"id": 2}, columns=_RING, beside=_step("n1"), limit=10, present="s"
    )
    assert missing == []


async def test_a_row_beside_a_step_reads_at_most_its_limit(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if the step reads past its limit."""
    rows = await read_row_beside(
        ThreadedDuckDB(db), "ring", by={"id": 1}, columns=_RING, beside=_step("n1"), limit=2, present="seen"
    )
    assert len(rows) == 2


def test_the_statement_binds_the_pins_then_the_step_then_the_limit() -> None:
    """Flips if the step's placeholders are not moved past the pins', or the limit is not the last bind; or a limit
    under one row is taken."""
    sql, params = read_row_beside_statement(
        "s.ring", by={"id": 1}, columns=["a"], beside=_step("n1"), limit=5, present="seen"
    )
    assert params == [1, 0, "n1", 1, "n1", 5]
    assert sql.startswith('SELECT r.*, s.* FROM (SELECT t."a" FROM "s"."ring" AS t WHERE t."id" = $1 LIMIT 1) AS r ')
    assert 'LEFT JOIN LATERAL (SELECT TRUE AS "seen", b.* FROM (' in sql and sql.endswith("LIMIT $6) AS s ON TRUE")
    assert "r.current_ring = $2" in sql and "t.name = $3" in sql and "r.current_ring = $4" in sql
    with pytest.raises(ValueError, match="at least one row"):
        read_row_beside_statement("ring", by={"id": 1}, columns=["a"], beside=_step("n1"), limit=0, present="s")


async def test_a_row_beside_a_step_answers_the_row_once_when_asked(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if ``first`` does not answer the row's columns on one step row alone (TRUE in ``first``, NULL elsewhere)
    while the step still reads them, or on the row alone when the step holds none; or if the limit can drop the row
    that carries them."""
    conn = ThreadedDuckDB(db)
    rows = await read_row_beside(
        conn, "ring", by={"id": 1}, columns=_RING, beside=_step("n1"), limit=10, present="seen", first="once"
    )
    want = {
        row[0] for row in db.execute("SELECT DISTINCT v FROM slot_1 WHERE name = 'n1' AND seen_at >= 10").fetchall()
    }
    assert {row["v"] for row in rows} == want and len(rows) == len(want) and all(row["seen"] for row in rows)
    assert [(row["current_ring"], row["floor_at"]) for row in rows if row["once"]] == [(1, 10)]
    assert all(row["current_ring"] is None and row["floor_at"] is None for row in rows if not row["once"])
    for limit in (1, 2):
        cut = await read_row_beside(
            conn, "ring", by={"id": 1}, columns=_RING, beside=_step("n1"), limit=limit, present="seen", first="once"
        )
        assert len(cut) == limit and [row["floor_at"] for row in cut if row["once"]] == [10]
    alone = await read_row_beside(
        conn, "ring", by={"id": 1}, columns=_RING, beside=_step("none"), limit=10, present="seen", first="once"
    )
    assert [(row["current_ring"], row["seen"], row["once"], row["v"]) for row in alone] == [(1, None, None, None)]
    sql, params = read_row_beside_statement(
        "ring", by={"id": 1}, columns=_RING, beside=_step("n1"), limit=5, present="seen", first="once"
    )
    assert sql.startswith('SELECT CASE WHEN s."once" IS NOT FALSE THEN r."current_ring" END AS "current_ring", ')
    assert 'row_number() OVER () = 1 AS "once", k.* FROM (SELECT b.* FROM (' in sql and sql.endswith(
        "LIMIT $6) AS k) AS s ON TRUE"
    )
    assert params == [1, 0, "n1", 1, "n1", 5]


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
