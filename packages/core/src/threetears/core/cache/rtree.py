"""a SQLite R*Tree on an L1: a spatial index of 2-D boxes by text key, read by the boxes a box overlaps.

SQLite's R*Tree module is built in (no SpatiaLite), and an R*Tree is a virtual table keyed by an integer ``id`` whose
coordinate columns are themselves the spatial index: a read bounding every coordinate (an overlap of a box) is answered
by ``xBestIndex`` from the tree, reading only the entries the box can reach. The tree takes integer ids only, so a
companion map table assigns one per text key. This is the one place an L1 writes and reads them, so its every
statement is keyed: the tree by a box or its id, the map by its id or its key.

.. code-block:: python

    tree = SQLiteRTree(l1, "geo_features_x_rtree", "geo_features_x_rtree_map")  # l1: an L1 backend (``get_connection()``) or a sqlite3.Connection
    tree.create()
    tree.put([("a", -112.2, -112.0, 33.3, 33.5)])  # (key, min_x, max_x, min_y, max_y)
    tree.overlapping(-112.5, -111.5, 33.0, 34.0)  # ["a"]
    tree.remove(["a"])

The tree stores 32-bit floats, its boxes rounded outward: it may answer a box that only touches the asked box's edge
after rounding, and never misses one that overlaps it. A caller wanting the exact overlap tests its own rows after.

WHY a map table, not a dict in the caller (measured 2026-11-15, macOS arm64, Python 3.14, SQLite 3.50; scratchpad
d6/mem/measure.py: FeatureCache holding ``04013-PCT-000123``-shaped rows through ``_hold`` in 2,000-row chunks, process
RSS growth, two runs each). Held as Python dicts beside the rows, the map grew RSS by +14.2 MB at 100,000 held
features, FeatureCache's default bound (50.0 -> 64.2 MB), and the hub builds one cache per layer. In this table it
costs what FeatureCache's own map table did before it moved here: 100,000 held 50.0 -> 50.5 MB, 180,000 (every
precinct of a national layer) 104.6 -> 105.1 MB. The overlap read is that cache's one join, timed against it in
packages/core/tests/unit/cache/test_rtree.py (2x budget; the numbers at :meth:`SQLiteRTree.overlapping`).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Final

from threetears.core.eviction import Declared, Policy

__all__ = ["SQLiteRTree"]

#: the tree's layout: its id, then the box as (min, max) per axis. A read bounding every coordinate walks the tree
#: (``SPATIAL_METHODS`` in the no-whole-reads gate's schema)
_RTREE_DDL: Final[str] = (
    "-- purpose: packages/core/src/threetears/core/cache/rtree.py::SQLiteRTree.overlapping\n"
    "-- plan_test: packages/core/tests/unit/cache/test_rtree.py::test_the_overlap_read_walks_the_tree\n"
    "CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING rtree(id, min_x, max_x, min_y, max_y)"
)

#: the map: each key's id, read by the id the tree answers
#: an R-tree's and its map's L1 policy (threetears.core.eviction)
_KEPT: Final = Declared(Policy.KEEP, "a spatial index of the rows beside it, whole with them; its owner bounds it")
_MAP_DDL: Final[str] = "CREATE TABLE IF NOT EXISTS {table} (id INTEGER PRIMARY KEY, key TEXT NOT NULL)"

#: the map's key, written and removed by
_MAP_KEY_DDL: Final[str] = (
    "-- purpose: packages/core/src/threetears/core/cache/rtree.py::SQLiteRTree.remove\n"
    "-- plan_test: packages/core/tests/unit/cache/test_rtree.py::test_a_remove_finds_its_keys_by_the_key_index\n"
    "CREATE UNIQUE INDEX IF NOT EXISTS {index} ON {table} (key)"
)

#: the most keys or ids one statement names (SQLite allows 32,766 binds; a page this size keeps a statement's text
#: and its plan small)
_KEY_BATCH: Final[int] = 500


def _batches(values: Sequence[Any]) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), _KEY_BATCH):
        yield values[start : start + _KEY_BATCH]


def _marks(count: int) -> str:
    return ", ".join(["?"] * count)


class SQLiteRTree:
    """one R*Tree of 2-D boxes by text key on a SQLite connection; each write is its own transaction, committed.

    :param conn: an L1 backend whose ``get_connection()`` answers the calling thread's ``sqlite3.Connection`` (asked
        at each statement, so a backend that rebinds its connection is followed), or a ``sqlite3.Connection``
    :ptype conn: Any
    :param table: the tree's name, a TRUSTED identifier
    :ptype table: str
    :param map_table: its map's name, a TRUSTED identifier
    :ptype map_table: str
    """

    def __init__(self, conn: Any, table: str, map_table: str) -> None:
        self._conn = conn
        self._table = table
        self._map = map_table

    @property
    def table(self) -> str:
        """the tree's name."""
        return self._table

    def _connection(self) -> Any:
        found = getattr(self._conn, "get_connection", None)
        return found() if found is not None else self._conn

    def create(self) -> None:
        """create the tree and its map if absent."""
        conn = self._connection()
        conn.execute(_RTREE_DDL.format(table=self._table))
        conn.execute(_MAP_DDL.format(table=self._map))
        conn.execute(_MAP_KEY_DDL.format(index=f"{self._map}_key", table=self._map))
        conn.commit()
        declare = getattr(self._conn, "declare", None)
        if declare is not None:
            # made by DDL, not SQLiteBackend.initialize, so declared here: the tree and its map index the rows beside
            # them and must stay whole with them; their owner bounds what they hold (FeatureCache.max_cached_rows)
            for table in (self._table, self._map):
                declare(table, _KEPT)

    def put(self, boxes: Sequence[tuple[str, float, float, float, float]]) -> None:
        """record each box under its key, replacing the box a key held (its id kept); one transaction.

        :param boxes: ``(key, min_x, max_x, min_y, max_y)`` each
        :ptype boxes: Sequence[tuple[str, float, float, float, float]]
        """
        if not boxes:
            return
        conn = self._connection()
        ids: dict[str, int] = {}
        for batch in _batches([box[0] for box in boxes]):
            # a key held already answers its own id: the no-op update makes RETURNING name it
            cursor = conn.execute(
                f"INSERT INTO {self._map} (key) VALUES {', '.join(['(?)'] * len(batch))} "
                "ON CONFLICT (key) DO UPDATE SET key = excluded.key RETURNING id, key",
                list(batch),
            )
            ids.update((str(row[1]), int(row[0])) for row in cursor.fetchall())
        conn.executemany(
            f"INSERT OR REPLACE INTO {self._table} (id, min_x, max_x, min_y, max_y) VALUES (?, ?, ?, ?, ?)",
            [(ids[key], *bounds) for key, *bounds in boxes],
        )
        conn.commit()

    def remove(self, keys: Sequence[str]) -> None:
        """drop the boxes of ``keys``, each found by its key; one transaction.

        :param keys: the keys
        :ptype keys: Sequence[str]
        """
        if not keys:
            return
        conn = self._connection()
        gone: list[int] = []
        for batch in _batches(list(keys)):
            cursor = conn.execute(f"DELETE FROM {self._map} WHERE key IN ({_marks(len(batch))}) RETURNING id", batch)
            gone.extend(int(row[0]) for row in cursor.fetchall())
        conn.executemany(f"DELETE FROM {self._table} WHERE id = ?", [(box_id,) for box_id in gone])
        conn.commit()

    def overlapping(self, min_x: float, max_x: float, min_y: float, max_y: float, *, prefix: str = "") -> list[str]:
        """the keys of the boxes overlapping ``[min_x, max_x] x [min_y, max_y]``, edges included, that start with
        ``prefix``, in no promised order.

        One statement: the tree walked by the box, each entry's map row joined by its id (the map's key). Two keyed
        statements instead (the tree's ids, then the map by them in batches) took 2.0-2.2x the join at real size
        (100,000 boxes, 2026-11-15: 225 keys 89 us vs 43 us, 1,681 keys 633 us vs 283 us); the join is the read. Against
        the statement FeatureCache sent before (the same join, its SQL in the cache): 4 keys 6 vs 6 us, 225 keys 60 vs
        56 us, 1,681 keys 435 vs 417 us (test_the_overlap_read_is_within_twice_the_join_at_real_size).

        :param min_x: the box's least x
        :ptype min_x: float
        :param max_x: its greatest x
        :ptype max_x: float
        :param min_y: its least y
        :ptype min_y: float
        :param max_y: its greatest y
        :ptype max_y: float
        :param prefix: the start every key answered has (a plain prefix: ``%`` and ``_`` are no wildcards); every key
            when empty
        :ptype prefix: str
        :return: the keys
        :rtype: list[str]
        """
        rows = (
            self._connection()
            .execute(
                f"SELECT m.key FROM {self._table} AS r JOIN {self._map} AS m ON m.id = r.id "
                "WHERE r.max_x >= ? AND r.min_x <= ? AND r.max_y >= ? AND r.min_y <= ? AND substr(m.key, 1, ?) = ?",
                (min_x, max_x, min_y, max_y, len(prefix), prefix),
            )
            .fetchall()
        )
        return [str(row[0]) for row in rows]

    def drop(self) -> None:
        """drop the tree and its map, every box with them."""
        conn = self._connection()
        conn.execute(f"DROP TABLE IF EXISTS {self._table}")
        conn.execute(f"DROP TABLE IF EXISTS {self._map}")
        conn.commit()
