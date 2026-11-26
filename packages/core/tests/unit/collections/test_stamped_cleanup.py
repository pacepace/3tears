"""quiet-time cleanup removes a stamped table's gone rows below the floor F, by key, a bucket at a time; never a live row,
never one at or above F (append-only L3, decisions 2 and 5).

The table runs on an in-process DuckDB behind a reader that names no dialect (so the helpers take it for L3, where a
keyed DELETE of a stamped table is refused: cleanup's own statements are the one delete it takes). Its ``epoch_bucket``
is a plain column here. The seeded rows, F = 100:

- ``("CA", "r1")`` gone at 50, bucket 3: removed;
- ``("CA", "r2")`` gone at 60, bucket 3: removed;
- ``("CA", "r3")`` gone at 150, bucket 3: kept (at or above F, a restore may still read it);
- ``("NV", "r1")`` live at 40, bucket 3: kept (live);
- ``("TX", "r1")`` gone at 70, bucket 9: removed.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import duckdb
import pytest

from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, TableSchema
from threetears.core.collections.stamped_cleanup import clear_stamped_gone
from threetears.core.ddl.row_epoch import ROW_STAMPED, AppendOnlyWriteError
from threetears.core.keyed import ThreadedDuckDB, delete_keyed
from threetears.core.testing.l1_tables import TEST_L1_TABLE

_TABLE = "sc_runs"

_RUNS = TableSchema(
    name=_TABLE,
    primary_key=("state", "run_id"),
    columns=[Column("state", STRING_TYPE), Column("run_id", STRING_TYPE), Column("rows", BIGINT_TYPE)],
    row_epoch=ROW_STAMPED,
    l1_eviction=TEST_L1_TABLE,
)


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    conn = duckdb.connect()
    conn.execute(
        f"CREATE TABLE {_TABLE} (state VARCHAR NOT NULL, run_id VARCHAR NOT NULL, rows BIGINT NOT NULL, "
        "epoch BIGINT NOT NULL, gone BOOLEAN NOT NULL DEFAULT false, epoch_bucket SMALLINT NOT NULL, "
        "PRIMARY KEY (state, run_id))"
    )
    conn.execute(
        f"INSERT INTO {_TABLE} (state, run_id, rows, epoch, gone, epoch_bucket) VALUES "
        "('CA', 'r1', 1, 50, true, 3), ('CA', 'r2', 2, 60, true, 3), ('CA', 'r3', 3, 150, true, 3), "
        "('NV', 'r1', 4, 40, false, 3), ('TX', 'r1', 5, 70, true, 9)"
    )
    yield conn
    conn.close()


class _L3:
    """an L3 pool answering from DuckDB: it names no dialect, so the helpers take it for L3; it keeps every statement."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._inner = ThreadedDuckDB(conn)
        self.statements: list[str] = []

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append(query)
        return await self._inner.fetch(query, *params)

    async def execute(self, query: str, *params: Any) -> str:
        self.statements.append(query)
        return await self._inner.execute(query, *params)


def _held(db: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    return sorted(db.execute(f"SELECT state, run_id FROM {_TABLE}").fetchall())


async def test_cleanup_removes_the_gone_rows_below_the_floor_alone(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if cleanup leaves a gone row below F, removes a live row or a gone one at or above F, or reads the table
    by anything but each bucket's range of the (bucket, epoch) index (one bucket pinned a statement)."""
    pool = _L3(db)
    swept = await clear_stamped_gone(pool, _RUNS, since=0, floor=100, batch=1)
    assert swept.finished and swept.removed == 3
    assert _held(db) == [("CA", "r3"), ("NV", "r1")]
    reads = [sql for sql in pool.statements if sql.lstrip().startswith("SELECT") or "SELECT" in sql.split("FROM")[0]]
    assert reads and all('t."epoch_bucket" = $1' in sql for sql in reads if f'"{_TABLE}"' in sql), reads
    deletes = [sql for sql in pool.statements if sql.startswith("DELETE")]
    assert deletes and all('"gone"' in sql and '"epoch" < $' in sql for sql in deletes), deletes


async def test_cleanup_keeps_a_key_written_again_before_its_delete(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a key written again between cleanup's read and its delete (live, its epoch risen) is removed."""
    pool = _L3(db)
    real_fetch = pool.fetch

    async def fetch_then_revive(query: str, *params: Any) -> list[dict[str, Any]]:
        answered = await real_fetch(query, *params)
        # the writer brings CA r1 back after cleanup read it
        db.execute(f"UPDATE {_TABLE} SET gone = false, epoch = 200 WHERE state = 'CA' AND run_id = 'r1'")
        return answered

    pool.fetch = fetch_then_revive  # type: ignore[method-assign]
    await clear_stamped_gone(pool, _RUNS, since=0, floor=100)
    assert ("CA", "r1") in _held(db)


async def test_cleanup_stops_before_a_batch_when_a_load_starts(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if cleanup goes on deleting once a load started."""
    swept = await clear_stamped_gone(_L3(db), _RUNS, since=0, floor=100, stirred=lambda: True)
    assert not swept.finished and swept.removed == 0
    assert len(_held(db)) == 5


async def test_a_keyed_delete_of_the_table_is_still_refused(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if the keyed delete cleanup does not use is open to every caller."""
    with pytest.raises(AppendOnlyWriteError):
        await delete_keyed(_L3(db), _TABLE, by={"state": "CA", "run_id": "r1"})
