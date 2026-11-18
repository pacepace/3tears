"""the rows a walk of a table's links reaches (:func:`read_reached`), as a recursive statement walking them answers
(``WITH RECURSIVE ... NOT (next = ANY(path))``, then ``DISTINCT``): a keyed read a level, each node read once; on an
in-process DuckDB."""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
import pytest

from threetears.core.keyed import InlineDuckDB, Joined, read_reached

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


def _levels(start: str, depth: int) -> set[tuple[str, str]]:
    """the rows a walk from ``start`` reaches within ``depth`` levels (the start's own rows the first), following
    dataset links: the expected answer, by hand."""
    found: set[tuple[str, str]] = set()
    level = {start}
    for _ in range(depth):
        rows = [(kind, source) for dataset, kind, source in _EDGES if dataset in level]
        found |= set(rows)
        level = {source for kind, source in rows if kind == "dataset"}
    return found


@pytest.mark.parametrize("start", ["d1", "d2", "d4", "d5", "nothing"])
@pytest.mark.parametrize("depth", [1, 2, 3, 5])
async def test_a_capped_walk_reaches_the_rows_within_its_depth_in_one_statement(
    monkeypatch: pytest.MonkeyPatch, db: duckdb.DuckDBPyConnection, start: str, depth: int
) -> None:
    """Flips if a walk given a ``depth`` reaches a row past the cap (a level too many, a cycle walked round past it),
    misses one within it, answers a row twice, or sends more than one statement."""
    sent: list[str] = []
    inline = InlineDuckDB(db)
    fetch = inline.fetch

    async def recorded(sql: str, *args: object) -> object:
        sent.append(sql)
        return await fetch(sql, *args)

    monkeypatch.setattr(inline, "fetch", recorded)
    rows = await read_reached(
        inline,
        "sources",
        start={"dataset_id": start},
        on="dataset_id",
        to="source_id",
        follow={"source_kind": "dataset"},
        columns=("source_kind", "source_id"),
        depth=depth,
    )
    assert {(row["source_kind"], row["source_id"]) for row in rows} == _levels(start, depth)
    assert len(rows) == len({(row["source_kind"], row["source_id"]) for row in rows})
    assert len([sql for sql in sent if "sources" in sql]) == 1, sent


async def test_a_capped_walk_joins_what_it_reaches_in_the_same_statement(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if each reached row is not joined to its table's rows by its column (a LEFT join keeping a row none
    matches, an inner one dropping it), or the join's own pins are left out."""
    db.execute("CREATE TABLE owners (source_id VARCHAR, owner VARCHAR, kind VARCHAR)")
    db.executemany(
        "INSERT INTO owners VALUES (?, ?, ?)",
        [("d2", "ann", "person"), ("d2", "bot", "agent"), ("s1", "cy", "person"), ("d3", "di", "person")],
    )

    async def walk(left: bool) -> set[tuple[object, ...]]:
        rows = await read_reached(
            InlineDuckDB(db),
            "sources",
            start={"dataset_id": "d1"},
            on="dataset_id",
            to="source_id",
            follow={"source_kind": "dataset"},
            columns=("source_id",),
            depth=2,
            joins=[
                Joined(
                    "owners",
                    on={"source_id": "source_id"},
                    by={"kind": "person"},
                    columns={"owner": "owner"},
                    left=left,
                )
            ],
        )
        return {(row["source_id"], row["owner"]) for row in rows}

    # d1 reaches d2 and s1 (level 1), then d3 and t9 (level 2); d1 itself again only past the cap
    assert await walk(left=True) == {("d2", "ann"), ("s1", "cy"), ("d3", "di"), ("t9", None)}
    assert await walk(left=False) == {("d2", "ann"), ("s1", "cy"), ("d3", "di")}


async def test_a_walk_refuses_joins_without_a_depth_or_a_depth_under_one(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a walk joins what it reaches a read a level (no depth), or walks no level."""
    joined = [Joined("owners", on={"source_id": "source_id"}, columns={"owner": "owner"}, left=True)]
    with pytest.raises(ValueError, match="give it a depth"):
        await read_reached(
            InlineDuckDB(db),
            "sources",
            start={"dataset_id": "d1"},
            on="dataset_id",
            to="source_id",
            columns=("source_id",),
            joins=joined,
        )
    with pytest.raises(ValueError, match="at least one level"):
        await read_reached(
            InlineDuckDB(db),
            "sources",
            start={"dataset_id": "d1"},
            on="dataset_id",
            to="source_id",
            columns=("source_id",),
            depth=0,
        )
