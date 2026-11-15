"""SQLiteRTree: a SQLite R*Tree of 2-D boxes on an L1, written by id and read by the boxes a box overlaps."""

from __future__ import annotations

import re
import sqlite3

import pytest

from threetears.core.cache.rtree import SQLiteRTree


@pytest.fixture
def conn() -> sqlite3.Connection:
    return sqlite3.connect(":memory:")


def _traced(conn: sqlite3.Connection) -> list[str]:
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    return statements


def _plan(conn: sqlite3.Connection, statement: str, params: tuple[float, ...]) -> str:
    return " | ".join(str(row[3]) for row in conn.execute(f"EXPLAIN QUERY PLAN {statement}", params).fetchall())


def test_the_overlap_read_walks_the_tree(conn: sqlite3.Connection) -> None:
    """Flips if the overlap read SQLiteRTree.overlapping sends is not answered from the tree with all four coordinates
    as its constraints (``VIRTUAL TABLE INDEX 2:`` and four constraint codes): a read the tree does not bound by
    every coordinate may reach every entry along the unbounded one. The contrast, one axis bounded, plans two."""
    tree = SQLiteRTree(conn, "boxes")
    tree.create()
    tree.put([(i, i, i + 1.0, -i, -i + 1.0) for i in range(1, 200)])
    box = (10.0, 20.0, -20.0, -10.0)
    statements = _traced(conn)
    assert tree.overlapping(*box)
    conn.set_trace_callback(None)
    (sent,) = [statement for statement in statements if statement.lstrip().upper().startswith("SELECT")]
    # the traced text has the binds inlined; plan the statement as sent, its binds as placeholders again
    written = re.sub(r"(>=|<=) -?[\d.]+", r"\1 ?", sent)
    plan = _plan(conn, written, box)
    found = re.search(r"VIRTUAL TABLE INDEX 2:((?:[A-H]\d)*)", plan)
    assert found is not None, plan
    assert sorted(found.group(1)[1::2]) == ["0", "1", "2", "3"], plan
    one_axis = _plan(conn, "SELECT id FROM boxes WHERE max_x >= ? AND min_x <= ?", box[:2])
    contrast = re.search(r"VIRTUAL TABLE INDEX 2:((?:[A-H]\d)*)", one_axis)
    assert contrast is not None and len(contrast.group(1)) == 4, one_axis


def test_answers_the_boxes_overlapping_a_box_edges_included(conn: sqlite3.Connection) -> None:
    """Flips if the overlap read misses a box inside, straddling or touching the asked box's edge, or answers one
    apart from it."""
    tree = SQLiteRTree(conn, "boxes")
    tree.create()
    tree.put(
        [
            (1, 0.0, 1.0, 0.0, 1.0),  # inside
            (2, -5.0, 0.5, -5.0, 0.5),  # straddles the corner
            (3, 2.0, 3.0, 0.0, 1.0),  # touches the right edge
            (4, 2.5, 3.0, 0.0, 1.0),  # apart, right
            (5, 0.0, 1.0, 5.0, 6.0),  # apart, above
        ]
    )
    assert sorted(tree.overlapping(0.0, 2.0, 0.0, 2.0)) == [1, 2, 3]


def test_a_put_replaces_an_id_s_box_and_a_remove_drops_only_its_ids(conn: sqlite3.Connection) -> None:
    """Flips if a second put of an id adds a box beside its first, if remove drops another id's box, or if a removed
    or dropped tree still answers."""
    tree = SQLiteRTree(conn, "boxes")
    tree.create()
    tree.put([(1, 0.0, 1.0, 0.0, 1.0), (2, 0.0, 1.0, 0.0, 1.0)])
    tree.put([(1, 10.0, 11.0, 10.0, 11.0)])
    assert sorted(tree.overlapping(0.0, 1.0, 0.0, 1.0)) == [2]
    assert tree.overlapping(10.0, 11.0, 10.0, 11.0) == [1]
    tree.remove([2])
    assert tree.overlapping(0.0, 1.0, 0.0, 1.0) == []
    assert tree.overlapping(10.0, 11.0, 10.0, 11.0) == [1]
    tree.put([])
    tree.remove([])
    tree.drop()
    assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'boxes'").fetchall() == []


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
    tree = SQLiteRTree(backend, "boxes")
    tree.create()
    backend.conn = second
    with pytest.raises(sqlite3.OperationalError):
        tree.overlapping(0.0, 1.0, 0.0, 1.0)
    backend.conn = first
    assert tree.overlapping(0.0, 1.0, 0.0, 1.0) == []
