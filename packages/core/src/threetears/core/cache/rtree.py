"""a SQLite R*Tree on an L1: a spatial index of 2-D boxes by integer id, read by the boxes a box overlaps.

SQLite's R*Tree module is built in (no SpatiaLite), and an R*Tree is a virtual table keyed by its integer ``id`` whose
coordinate columns are themselves the spatial index: a read bounding every coordinate (an overlap of a box) is answered
by ``xBestIndex`` from the tree, reading only the entries the box can reach. This is the one place an L1 writes and
reads one, so its every statement is the tree's own: by id, or by a box.

.. code-block:: python

    tree = SQLiteRTree(l1, "geo_features_x_rtree")  # l1: an L1 backend (``get_connection()``) or a sqlite3.Connection
    tree.create()
    tree.put([(1, -112.2, -112.0, 33.3, 33.5)])  # (id, min_x, max_x, min_y, max_y)
    tree.overlapping(-112.5, -111.5, 33.0, 34.0)  # [1]
    tree.remove([1])

The tree stores 32-bit floats, its boxes rounded outward: it may answer a box that only touches the asked box's edge
after rounding, and never misses one that overlaps it. A caller wanting the exact overlap tests its own rows after.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

__all__ = ["SQLiteRTree"]

#: the tree's layout: its key, then the box as (min, max) per axis. Its purpose and plan test are the no-whole-reads
#: gate's: a read bounding every coordinate walks the tree (``SPATIAL_METHODS`` in the gate's schema)
_RTREE_DDL: Final[str] = (
    "-- purpose: packages/core/src/threetears/core/cache/rtree.py::SQLiteRTree.overlapping\n"
    "-- plan_test: packages/core/tests/unit/cache/test_rtree.py::test_the_overlap_read_walks_the_tree\n"
    "CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING rtree(id, min_x, max_x, min_y, max_y)"
)


class SQLiteRTree:
    """one R*Tree of 2-D boxes on a SQLite connection; each write is its own transaction, committed.

    :param conn: an L1 backend whose ``get_connection()`` answers the calling thread's ``sqlite3.Connection`` (asked
        at each statement, so a backend that rebinds its connection is followed), or a ``sqlite3.Connection``
    :ptype conn: Any
    :param table: the tree's name, a TRUSTED identifier
    :ptype table: str
    """

    def __init__(self, conn: Any, table: str) -> None:
        self._conn = conn
        self._table = table

    @property
    def table(self) -> str:
        """the tree's name."""
        return self._table

    def _connection(self) -> Any:
        found = getattr(self._conn, "get_connection", None)
        return found() if found is not None else self._conn

    def create(self) -> None:
        """create the tree if absent."""
        conn = self._connection()
        conn.execute(_RTREE_DDL.format(table=self._table))
        conn.commit()

    def put(self, boxes: Sequence[tuple[int, float, float, float, float]]) -> None:
        """record each box under its id, replacing the box an id held; one transaction.

        :param boxes: ``(id, min_x, max_x, min_y, max_y)`` each
        :ptype boxes: Sequence[tuple[int, float, float, float, float]]
        """
        if not boxes:
            return
        conn = self._connection()
        conn.executemany(
            f"INSERT OR REPLACE INTO {self._table} (id, min_x, max_x, min_y, max_y) VALUES (?, ?, ?, ?, ?)", boxes
        )
        conn.commit()

    def remove(self, ids: Sequence[int]) -> None:
        """drop the boxes of ``ids``, each by its key; one transaction.

        :param ids: the ids
        :ptype ids: Sequence[int]
        """
        if not ids:
            return
        conn = self._connection()
        conn.executemany(f"DELETE FROM {self._table} WHERE id = ?", [(box_id,) for box_id in ids])
        conn.commit()

    def overlapping(self, min_x: float, max_x: float, min_y: float, max_y: float) -> list[int]:
        """the ids of the boxes overlapping ``[min_x, max_x] x [min_y, max_y]``, edges included, in no promised order.

        :param min_x: the box's least x
        :ptype min_x: float
        :param max_x: its greatest x
        :ptype max_x: float
        :param min_y: its least y
        :ptype min_y: float
        :param max_y: its greatest y
        :ptype max_y: float
        :return: the ids
        :rtype: list[int]
        """
        cursor = self._connection().execute(
            f"SELECT id FROM {self._table} WHERE max_x >= ? AND min_x <= ? AND max_y >= ? AND min_y <= ?",
            (min_x, max_x, min_y, max_y),
        )
        try:
            return [int(row[0]) for row in cursor.fetchall()]
        finally:
            cursor.close()

    def drop(self) -> None:
        """drop the tree, every box with it."""
        conn = self._connection()
        conn.execute(f"DROP TABLE IF EXISTS {self._table}")
        conn.commit()
