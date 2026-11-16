"""Each table's rows and bytes in one SQLite L1 database, read from SQLite's own pages (``dbstat``).

One walk answers both for every table: a table's bytes are the pages of its own b-tree and of every index on it, its
rows the cells of its b-tree's leaf pages (exactly ``count(*)``, held by ``tests/unit/cache/test_l1_sizes.py``).
Shared by the size gauges (:mod:`~threetears.core.cache.l1_sizes`, SQ2) and the L1 budget's ledger
(:mod:`~threetears.core.cache.l1_lru`, which scales its per-row estimate by the measured bytes), so both read one
measurement. Its own module because the backend (:mod:`~threetears.core.cache.sqlite`) calls it and the gauges import
the backend. ``dbstat`` is SQLite's page catalog, bounded by the database as ``pg_class`` is (the whole-read gates
treat it as a catalog).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

__all__ = ["TableSize", "measure"]

#: every b-tree's bytes and leaf cells, by b-tree name, in one walk of the database's pages
_PAGES_SQL: Final = (
    "SELECT name, sum(pgsize) AS bytes, sum(CASE WHEN pagetype = 'leaf' THEN ncell ELSE 0 END) AS cells "
    "FROM dbstat GROUP BY name"
)

#: each b-tree's table: a table's own, an index's (autoindexes of a key included) the table it indexes
_OWNERS_SQL: Final = "SELECT type, name, tbl_name FROM sqlite_schema WHERE type IN ('table', 'index')"


@dataclass(frozen=True)
class TableSize:
    """one L1 table's size.

    :ivar rows: its rows
    :ivar bytes: the bytes of its pages and of its indexes' pages
    """

    rows: int
    bytes: int


def measure(conn: Any) -> dict[str, TableSize]:
    """every table's size in one SQLite database, from its pages.

    SQLite's own tables (``sqlite_*``) are left out.

    :param conn: a connection to the database on the calling thread (a ``sqlite3.Connection``, or the L1's own
        thread-local one, :meth:`~threetears.core.cache.sqlite.SQLiteBackend.get_connection`)
    :ptype conn: Any
    :return: table -> its size
    :rtype: dict[str, TableSize]
    :raises sqlite3.OperationalError: when this SQLite was built without ``dbstat``
    """
    owners: dict[str, tuple[str, str]] = {
        str(name): (str(kind), str(table)) for kind, name, table in conn.execute(_OWNERS_SQL).fetchall()
    }
    rows: dict[str, int] = {}
    held: dict[str, int] = {}
    for name, size, cells in conn.execute(_PAGES_SQL).fetchall():
        kind, table = owners.get(str(name), ("", ""))
        if not table or table.startswith("sqlite_"):
            continue
        held[table] = held.get(table, 0) + int(size or 0)
        if kind == "table":
            rows[table] = int(cells or 0)
    # a table with no pages yet still has its root page, so every table appears above; keep the shape regardless
    return {table: TableSize(rows=rows.get(table, 0), bytes=size) for table, size in held.items()}
