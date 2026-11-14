"""the rows a walk of a table's links reaches (:func:`read_reached`), as a recursive statement walking them answers
(``WITH RECURSIVE ... NOT (next = ANY(path))``, then ``DISTINCT``): a keyed read a level, each node read once; on an
in-process DuckDB."""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
import pytest

from threetears.core.keyed import InlineDuckDB, read_reached

#: a dataset's sources: d1 <- d2 <- d3 <- d1 (a cycle), d2 <- t9 (a table, not walked), d4 alone, d5 <- d2
_EDGES = (
    ("d1", "dataset", "d2"),
    ("d1", "datasource", "s1"),
    ("d2", "dataset", "d3"),
    ("d2", "table", "t9"),
    ("d3", "dataset", "d1"),
    ("d3", "datasource", "s2"),
    ("d4", "datasource", "s1"),
    ("d5", "dataset", "d2"),
)


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    conn = duckdb.connect()
    conn.execute("CREATE TABLE sources (dataset_id VARCHAR, source_kind VARCHAR, source_id VARCHAR)")
    conn.executemany("INSERT INTO sources VALUES (?, ?, ?)", list(_EDGES))
    yield conn
    conn.close()


_CLOSURE = (
    "WITH RECURSIVE reachable(dataset_id, path) AS ( SELECT ?::VARCHAR, [?::VARCHAR] UNION ALL "
    "SELECT s.source_id, list_append(r.path, s.source_id) FROM sources s JOIN reachable r ON s.dataset_id = r.dataset_id "
    "WHERE s.source_kind = 'dataset' AND NOT list_contains(r.path, s.source_id) ) "
    "SELECT DISTINCT s.source_kind, s.source_id FROM sources s JOIN reachable r ON s.dataset_id = r.dataset_id"
)
_DOWNSTREAM = (
    "WITH RECURSIVE downstream(dataset_id, path) AS ( SELECT s.dataset_id, [s.dataset_id] FROM sources s "
    "WHERE s.source_kind = ? AND s.source_id = ? UNION ALL SELECT s.dataset_id, list_append(d.path, s.dataset_id) "
    "FROM sources s JOIN downstream d ON s.source_kind = 'dataset' AND s.source_id = d.dataset_id "
    "WHERE NOT list_contains(d.path, s.dataset_id) ) SELECT DISTINCT dataset_id FROM downstream"
)


@pytest.mark.parametrize("start", ["d1", "d2", "d4", "nothing"])
async def test_a_closure_reaches_the_rows_the_recursive_statement_reaches(
    db: duckdb.DuckDBPyConnection, start: str
) -> None:
    """Flips if the walk from a node misses a row a node it reaches holds, follows a row whose link is not to be
    followed (a table source), loops on a cycle, or answers a row twice."""
    rows = await read_reached(
        InlineDuckDB(db),
        "sources",
        start={"dataset_id": start},
        on="dataset_id",
        to="source_id",
        follow={"source_kind": "dataset"},
        columns=("source_kind", "source_id"),
    )
    expected = db.execute(_CLOSURE, [start, start]).fetchall()
    assert sorted((row["source_kind"], row["source_id"]) for row in rows) == sorted(expected)
    assert len(rows) == len({(row["source_kind"], row["source_id"]) for row in rows})


@pytest.mark.parametrize(("kind", "source"), [("datasource", "s1"), ("dataset", "d2"), ("table", "t9"), ("x", "y")])
async def test_a_downstream_walk_reaches_what_the_recursive_statement_reaches(
    db: duckdb.DuckDBPyConnection, kind: str, source: str
) -> None:
    """Flips if a walk started by pins other than its link's (a source's kind and id) and following by another column
    than it starts by misses a dataset downstream, or loops on the cycle."""
    rows = await read_reached(
        InlineDuckDB(db),
        "sources",
        start={"source_kind": kind, "source_id": source},
        on="source_id",
        to="dataset_id",
        by={"source_kind": "dataset"},
        columns=("dataset_id",),
    )
    expected = db.execute(_DOWNSTREAM, [kind, source]).fetchall()
    assert sorted(row["dataset_id"] for row in rows) == sorted(value for (value,) in expected)
