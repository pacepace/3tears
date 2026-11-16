"""SQLiteRTree: a SQLite R*Tree of 2-D boxes by text key on an L1, written by key and read by the boxes a box overlaps."""

from __future__ import annotations

import asyncio
import re
import sqlite3

import pytest

from threetears.core.cache.rtree import SQLiteRTree
from threetears.core.testing.calibrate import best_seconds_interleaved, budget_problem

#: features one cache holds at real size: FeatureCache's default bound (max_cached_rows)
_REAL_FEATURES = 100_000


@pytest.fixture
def conn() -> sqlite3.Connection:
    return sqlite3.connect(":memory:")


def _tree(conn: sqlite3.Connection) -> SQLiteRTree:
    tree = SQLiteRTree(conn, "boxes", "boxes_map")
    tree.create()
    return tree


def _sent(conn: sqlite3.Connection, run: object, verb: str) -> list[str]:
    """the statements of ``verb`` the call sends, as written (their binds as placeholders again)."""
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    try:
        run()  # type: ignore[operator]
    finally:
        conn.set_trace_callback(None)
    # the traced text has the binds inlined: back to placeholders, so the statement is planned as sent
    written = [re.sub(r"'[^']*'|(?<![\w.])-?\d+(?:\.\d+)?", "?", statement) for statement in statements]
    return [statement for statement in written if statement.lstrip().upper().startswith(verb)]


def _plan(conn: sqlite3.Connection, statement: str) -> str:
    params = [0] * statement.count("?")
    return " | ".join(str(row[3]) for row in conn.execute(f"EXPLAIN QUERY PLAN {statement}", params).fetchall())


def test_the_overlap_read_walks_the_tree(conn: sqlite3.Connection) -> None:
    """Flips if the overlap read SQLiteRTree.overlapping sends is not answered from the tree with all four coordinates
    as its constraints (``VIRTUAL TABLE INDEX 2:`` and four constraint codes), each entry's map row then found by the
    map's key (its id): a read the tree does not bound by every coordinate may reach every entry along the unbounded
    one. The contrast, one axis bounded, plans two."""
    tree = _tree(conn)
    tree.put([(f"k{i}", i, i + 1.0, -i, -i + 1.0) for i in range(1, 200)])
    (select,) = _sent(conn, lambda: tree.overlapping(10.0, 20.0, -20.0, -10.0, prefix="k"), "SELECT")
    plan = _plan(conn, select)
    found = re.search(r"SCAN r VIRTUAL TABLE INDEX 2:((?:[A-H]\d)*)", plan)
    assert found is not None, plan
    assert sorted(found.group(1)[1::2]) == ["0", "1", "2", "3"], plan
    assert "SEARCH m USING INTEGER PRIMARY KEY (rowid=?)" in plan, plan
    one_axis = _plan(conn, "SELECT id FROM boxes WHERE max_x >= ? AND min_x <= ?")
    contrast = re.search(r"VIRTUAL TABLE INDEX 2:((?:[A-H]\d)*)", one_axis)
    assert contrast is not None and len(contrast.group(1)) == 4, one_axis


def test_a_remove_finds_its_keys_by_the_key_index(conn: sqlite3.Connection) -> None:
    """Flips if SQLiteRTree.remove's delete from the map is not a search of its key index (``boxes_map_key``), or its
    delete from the tree is not by the tree's id."""
    tree = _tree(conn)
    tree.put([(f"k{i}", i, i + 1.0, i, i + 1.0) for i in range(200)])
    deletes = _sent(conn, lambda: tree.remove(["k3", "k5"]), "DELETE")
    assert len(deletes) == 3, deletes
    assert "INDEX boxes_map_key (key=?)" in _plan(conn, deletes[0]), deletes[0]
    assert "VIRTUAL TABLE INDEX 1:" in _plan(conn, deletes[1]), deletes[1]


def test_answers_the_boxes_overlapping_a_box_edges_included(conn: sqlite3.Connection) -> None:
    """Flips if the overlap read misses a box inside, straddling or touching the asked box's edge, or answers one
    apart from it."""
    tree = _tree(conn)
    tree.put(
        [
            ("inside", 0.0, 1.0, 0.0, 1.0),
            ("corner", -5.0, 0.5, -5.0, 0.5),
            ("touches", 2.0, 3.0, 0.0, 1.0),
            ("right", 2.5, 3.0, 0.0, 1.0),
            ("above", 0.0, 1.0, 5.0, 6.0),
        ]
    )
    assert sorted(tree.overlapping(0.0, 2.0, 0.0, 2.0)) == ["corner", "inside", "touches"]
    tree.put([("a_b", 0.0, 1.0, 0.0, 1.0), ("axb", 0.0, 1.0, 0.0, 1.0)])
    assert tree.overlapping(0.0, 2.0, 0.0, 2.0, prefix="a_") == ["a_b"], "the prefix is plain text, never LIKE"


def test_a_put_replaces_a_key_s_box_and_a_remove_drops_only_its_keys(conn: sqlite3.Connection) -> None:
    """Flips if a second put of a key adds a box (or an id) beside its first, if remove drops another key's box or
    leaves its own map row, or if a dropped tree still has its tables."""
    tree = _tree(conn)
    tree.put([("a", 0.0, 1.0, 0.0, 1.0), ("b", 0.0, 1.0, 0.0, 1.0)])
    tree.put([("a", 10.0, 11.0, 10.0, 11.0)])
    assert sorted(tree.overlapping(0.0, 1.0, 0.0, 1.0)) == ["b"]
    assert tree.overlapping(10.0, 11.0, 10.0, 11.0) == ["a"]
    assert conn.execute("SELECT count(*) FROM boxes").fetchone()[0] == 2
    assert conn.execute("SELECT count(*) FROM boxes_map").fetchone()[0] == 2
    tree.remove(["b", "never-held"])
    assert tree.overlapping(0.0, 1.0, 0.0, 1.0) == []
    assert tree.overlapping(10.0, 11.0, 10.0, 11.0) == ["a"]
    assert conn.execute("SELECT count(*) FROM boxes").fetchone()[0] == 1
    assert conn.execute("SELECT key FROM boxes_map").fetchall() == [("a",)]
    tree.put([])
    tree.remove([])
    tree.drop()
    assert conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'boxes%'").fetchall() == []


def test_a_write_or_read_past_one_batch_names_every_key(conn: sqlite3.Connection) -> None:
    """Flips if a put, a remove or a read naming more keys than one statement takes drops the keys past the first
    batch."""
    tree = _tree(conn)
    tree.put([(f"k{i}", 0.0, 1.0, 0.0, 1.0) for i in range(1_234)])
    assert len(tree.overlapping(0.0, 1.0, 0.0, 1.0)) == 1_234
    tree.remove([f"k{i}" for i in range(1_200)])
    assert sorted(tree.overlapping(0.0, 1.0, 0.0, 1.0)) == sorted(f"k{i}" for i in range(1_200, 1_234))
    assert conn.execute("SELECT count(*) FROM boxes").fetchone()[0] == 34


def test_a_backend_is_asked_for_its_connection_at_each_statement() -> None:
    """Flips if the tree keeps the first connection a backend answered rather than asking at each statement (a
    backend's connection is the calling thread's own)."""
    first, second = sqlite3.connect(":memory:"), sqlite3.connect(":memory:")

    class _Backend:
        def __init__(self) -> None:
            self.conn = first

        def get_connection(self) -> sqlite3.Connection:
            return self.conn

    backend = _Backend()
    tree = SQLiteRTree(backend, "boxes", "boxes_map")
    tree.create()
    backend.conn = second
    with pytest.raises(sqlite3.OperationalError):
        tree.overlapping(0.0, 1.0, 0.0, 1.0)
    backend.conn = first
    assert tree.overlapping(0.0, 1.0, 0.0, 1.0) == []


#: the one statement FeatureCache read its tree and map by before (to 2026-11-15, its SQL in the cache): the speed
#: baseline the helper's read is held to
_OLD_JOIN = (
    "SELECT m.key AS feature_key FROM boxes r JOIN boxes_map m ON m.id = r.id "
    "WHERE r.max_x >= ? AND r.min_x <= ? AND r.max_y >= ? AND r.min_y <= ? AND substr(m.key, 1, ?) = ?"
)


# times against the host's quiet (best_seconds_interleaved): run apart, never beside other work (tests/timing_suite.py)
@pytest.mark.loop_timing
@pytest.mark.parametrize(
    ("what", "box"),
    [
        # a z12 tile's rectangle: a few features
        ("a tile", (-100.0, -99.9, 40.0, 40.1)),
        # a z8 chunk's: a few hundred
        ("a chunk", (-101.4, -100.0, 39.0, 40.4)),
        # a wide read's: past a thousand
        ("a wide read", (-104.0, -100.0, 36.0, 40.0)),
    ],
)
def test_the_overlap_read_is_within_twice_the_join_at_real_size(
    conn: sqlite3.Connection, what: str, box: tuple[float, float, float, float]
) -> None:
    """Flips if the overlap read takes past twice the statement FeatureCache read the same rows by before it moved
    into this helper (the same connection, the same answer), at FeatureCache's default bound of 100,000 held
    features."""
    tree = _tree(conn)
    side = 0.1
    tree.put(
        [
            (
                f"cache\x1fprecincts\x1f1\x1f{i:06d}",
                -120 + (i % 500) * side,
                -120 + (i % 500) * side + 0.05,
                25 + (i // 500) * side,
                25 + (i // 500) * side + 0.04,
            )
            for i in range(_REAL_FEATURES)
        ]
    )
    prefix = "cache\x1fprecincts\x1f1\x1f"
    asked = (*box, len(prefix), prefix)
    expected = sorted(str(row[0]) for row in conn.execute(_OLD_JOIN, asked).fetchall())
    assert expected and sorted(tree.overlapping(*box, prefix=prefix)) == expected

    async def bare() -> list[object]:
        return conn.execute(_OLD_JOIN, asked).fetchall()

    async def path() -> list[str]:
        return tree.overlapping(*box, prefix=prefix)

    baseline, took = asyncio.run(best_seconds_interleaved(bare, path))
    print(f"{what}, {len(expected)} features: overlapping {took * 1e6:.0f} us, the old join {baseline * 1e6:.0f} us")
    problem = budget_problem(f"the overlap read of {what}", took, baseline)
    assert problem is None, problem
