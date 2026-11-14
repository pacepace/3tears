"""a joined read's counted join (:class:`Joined` with ``counts``): each read row once, beside how many of the joined
table's rows match it (or match it with a column not NULL), as a correlated ``(SELECT COUNT(*) ...)`` answers it; on
an in-process DuckDB."""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
import pytest

from threetears.core.keyed import InlineDuckDB, Joined, read_by_joined


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    conn = duckdb.connect()
    conn.execute("CREATE TABLE tables (id VARCHAR PRIMARY KEY, source VARCHAR, schema_name VARCHAR, name VARCHAR)")
    conn.execute(
        "CREATE TABLE columns (source VARCHAR, schema_name VARCHAR, table_name VARCHAR, column_name VARCHAR, "
        "description VARCHAR)"
    )
    conn.execute(
        "INSERT INTO tables VALUES ('t1', 's', 'a', 'x'), ('t2', 's', 'a', 'y'), ('t3', 's', 'b', 'x'), "
        "('t4', 's', NULL, 'z'), ('t5', 'other', 'a', 'x')"
    )
    conn.execute(
        "INSERT INTO columns VALUES ('s', 'a', 'x', 'c1', 'one'), ('s', 'a', 'x', 'c2', NULL), "
        "('s', 'a', 'x', 'c3', 'three'), ('s', 'b', 'x', 'c1', NULL), ('s', 'b', 'y', 'c1', 'stray'), "
        "(NULL, 'a', 'x', 'c9', 'no source'), ('other', 'a', 'x', 'c1', 'o')"
    )
    yield conn
    conn.close()


_ON = {"source": "source", "schema_name": "schema_name", "table_name": "name"}

_OLD = (
    "SELECT t.*, (SELECT count(*) FROM columns c WHERE c.source = t.source AND c.schema_name = t.schema_name "
    "AND c.table_name = t.name) AS n, (SELECT count(*) FROM columns c WHERE c.source = t.source "
    "AND c.schema_name = t.schema_name AND c.table_name = t.name AND c.description IS NOT NULL) AS described "
    "FROM tables t WHERE t.source = ? ORDER BY t.id"
)


async def test_a_counted_join_answers_each_row_once_with_its_correlated_counts(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a counted join drops a row nothing matches (its counts are 0), answers a row once a match, counts a
    NULL description, matches a NULL value, or differs from the correlated sub-selects over the same rows."""
    rows = await read_by_joined(
        InlineDuckDB(db),
        "tables",
        by={"source": "s"},
        columns=None,
        joins=[Joined("columns", on=_ON, columns={}, counts={"n": None, "described": "description"})],
    )
    expected = db.execute(_OLD, ["s"]).fetchall()
    assert sorted(tuple(row.values()) for row in rows) == sorted(expected)
    assert {row["id"]: (row["n"], row["described"]) for row in rows}["t1"] == (3, 2)


async def test_a_counted_join_takes_no_joined_column() -> None:
    """Flips if a join that counts also widens the row with the joined rows' columns (one row a read row, never one a
    match)."""
    with pytest.raises(ValueError, match="counts"):
        await read_by_joined(
            None,
            "tables",
            by={"source": "s"},
            columns=None,
            joins=[Joined("columns", on=_ON, columns={"c": "column_name"}, counts={"n": None})],
        )
