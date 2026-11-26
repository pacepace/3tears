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

import asyncio
from collections.abc import Callable, Iterator
from typing import Any, cast

import duckdb
import pytest

from threetears.core.backends.broker_generation import (
    CLEANUP_DELETE_FIELD,
    CLEANUP_FLOOR_FIELD,
    cleanup_deleting,
    stamp_fields,
)
from threetears.core.collections.cleanup_progress import CLEANUP_PROGRESS_TABLE
from threetears.core.collections.quiet_cleanup import STAMPED_SCOPE, LoadGate, QuietCleanup
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, TableSchema
from threetears.core.collections.scope_epochs import WHOLE, EpochSnapshot, ScopeEpochs
from threetears.core.collections.stamped_cleanup import clear_stamped_gone
from threetears.core.ddl.row_epoch import ROW_STAMPED, AppendOnlyWriteError, row_versions
from threetears.core.keyed import Fragment, ThreadedDuckDB, delete_keyed
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


class _Epochs:
    """the scope epochs QuietCleanup listens to: a quiet snapshot, nothing writing."""

    def __init__(self) -> None:
        self.snap = EpochSnapshot(version=1, writing=None, epochs={WHOLE: 1}, previous={})

    def on_change(self, listener: Callable[[], None]) -> Callable[[], None]:
        return lambda: None

    async def snapshot(self) -> EpochSnapshot:
        return self.snap


class _Marked(_L3):
    """:class:`_L3` keeping what each statement's request would say on the rail (stamp_fields)."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        super().__init__(conn)
        self.fields: list[tuple[str, dict[str, Any]]] = []

    async def execute(self, query: str, *params: Any) -> str:
        self.fields.append((query, stamp_fields()))
        return await super().execute(query, *params)


def _quiet(pool: Any, floor: int) -> QuietCleanup:
    async def window() -> int | None:
        return floor

    return QuietCleanup(pool, (_RUNS,), LoadGate(cast(ScopeEpochs, _Epochs())), window, run_markers=None)


async def test_quiet_cleanup_clears_a_stamped_tables_gone_rows_and_records_its_progress(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Flips if QuietCleanup refuses a stamped table, leaves its gone rows below F, or does not record that it cleared
    the table to F (so the next pass would read it again)."""
    db.execute(
        f"CREATE TABLE {CLEANUP_PROGRESS_TABLE} (table_name VARCHAR, scope VARCHAR, cleaned_to BIGINT, "
        "PRIMARY KEY (table_name, scope))"
    )
    pool = _Marked(db)
    done = await _quiet(pool, 100).clean()
    assert done.finished and done.removed == 3
    assert _held(db) == [("CA", "r3"), ("NV", "r1")]
    assert db.execute(f"SELECT table_name, scope, cleaned_to FROM {CLEANUP_PROGRESS_TABLE}").fetchall() == [
        (_TABLE, STAMPED_SCOPE, 100)
    ]
    # every delete of the table is sent as cleanup's, with its floor
    deletes = [fields for sql, fields in pool.fields if sql.startswith("DELETE")]
    assert deletes and all(f[CLEANUP_DELETE_FIELD] is True and f[CLEANUP_FLOOR_FIELD] == 100 for f in deletes)
    # a second pass at the same floor reads none of the table
    pool.statements.clear()
    again = await _quiet(pool, 100).clean()
    assert again.finished and again.removed == 0
    assert not [sql for sql in pool.statements if f'"{_TABLE}"' in sql or f" {_TABLE} " in sql], pool.statements


async def test_cleanups_own_keyed_delete_is_taken_inside_cleanup_deleting(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a keyed delete of an append-only table is refused inside cleanup_deleting (quiet-time cleanup's own
    delete, the one such a table takes), or taken outside it."""
    with cleanup_deleting(100):
        written = await delete_keyed(
            _L3(db), _TABLE, by={"state": "CA", "run_id": "r1"}, where=[Fragment('"epoch" < $1', (100,))]
        )
    assert written.count == 1
    with pytest.raises(AppendOnlyWriteError):
        await delete_keyed(_L3(db), _TABLE, by={"state": "CA", "run_id": "r2"})


class _HeardLog:
    """the change log's records as this process hears them (the hub's follower of the epoch log): each record handed
    to every listener subscribed, on the loop."""

    def __init__(self) -> None:
        self._listeners: list[Callable[..., None]] = []

    def on_record(self, listener: Callable[..., None]) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def hear(self, record: Any) -> None:
        for listener in list(self._listeners):
            listener(record)


class _Settled(_Marked):
    """:class:`_Marked` saying when cleanup recorded its progress: the end of a pass that cleared the table."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        super().__init__(conn)
        self.recorded = asyncio.Event()

    async def execute(self, query: str, *params: Any) -> str:
        answered = await super().execute(query, *params)
        if CLEANUP_PROGRESS_TABLE in query and query.lstrip().startswith("INSERT"):
            self.recorded.set()
        return answered


class _Timer:
    """one timer the gate armed: fired by the test, never by the clock."""

    def __init__(self, callback: Callable[..., None], args: tuple[Any, ...]) -> None:
        self._callback, self._args = callback, args
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        assert not self.cancelled
        self._callback(*self._args)


class _Timers:
    """the loop's ``call_later``: the gate's quiet timer kept for the test to fire, every other timer the loop's own
    (``asyncio.sleep``, ``wait_for``)."""

    def __init__(self, real: Callable[..., Any]) -> None:
        self.armed: list[_Timer] = []
        self._real = real

    def call_later(self, delay: float, callback: Callable[..., None], *args: Any, context: Any = None) -> Any:
        if getattr(callback, "__name__", "") != "_fell_quiet":
            return self._real(delay, callback, *args, context=context)
        timer = _Timer(callback, args)
        self.armed.append(timer)
        return timer

    def live(self) -> list[_Timer]:
        return [timer for timer in self.armed if not timer.cancelled]


async def test_a_change_log_gate_pauses_on_a_heard_record_and_resumes_when_quiet_with_no_l3_poll(
    db: duckdb.DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a database with no scope epochs (the hub) cannot gate cleanup, if a record heard does not pause it, if
    it does not resume once the log falls quiet, if quiet is anything but one timer re-armed on each record, or if
    anything reads L3 to learn whether a load is on or the log is quiet (a periodic poll): every statement is cleanup's
    own work, and none is sent while it waits."""
    db.execute(
        f"CREATE TABLE {CLEANUP_PROGRESS_TABLE} (table_name VARCHAR, scope VARCHAR, cleaned_to BIGINT, "
        "PRIMARY KEY (table_name, scope))"
    )
    loop = asyncio.get_running_loop()
    timers = _Timers(loop.call_later)
    # the quiet timer is fired by the test, not by the wall clock: what the test asserts waits on nothing timed
    monkeypatch.setattr(loop, "call_later", timers.call_later)
    pool = _Settled(db)
    heard = _HeardLog()
    gate = LoadGate.from_change_log(heard.on_record, quiet_after=30.0)

    async def window() -> int | None:
        return 100

    cleanup = QuietCleanup(pool, (_RUNS,), gate, window, run_markers=None)
    # a load writes: its record pauses cleanup at once, with no read
    heard.hear({"table": _TABLE})
    assert gate.stirred
    assert (await cleanup.clean()).paused
    assert pool.statements == []
    runner = asyncio.create_task(cleanup.run())
    try:
        # the load goes on: each record cancels the timer and arms one again, and cleanup sends nothing
        for _ in range(4):
            await asyncio.sleep(0)
            heard.hear({"table": _TABLE})
        await asyncio.sleep(0)
        assert len(timers.armed) == 5 and len(timers.live()) == 1
        assert pool.statements == [] and len(_held(db)) == 5
        # the log falls quiet: the one timer fires and cleanup resumes, clearing the table
        timers.live()[0].fire()
        await asyncio.wait_for(pool.recorded.wait(), timeout=5)
        assert _held(db) == [("CA", "r3"), ("NV", "r1")]
        # quiet and done: nothing more is sent and no timer armed while it waits (no poll of L3)
        sent, armed = len(pool.statements), len(timers.armed)
        for _ in range(5):
            await asyncio.sleep(0.01)
        assert len(pool.statements) == sent, pool.statements[sent:]
        assert len(timers.armed) == armed
    finally:
        runner.cancel()
        gate.close()


def test_a_change_log_gate_refuses_versioned_tables() -> None:
    """Flips if a gate that names no scopes is handed a versioned table, which it would never clean."""
    versioned = TableSchema(
        name="sc_versions",
        primary_key=("state", "run_id"),
        columns=[Column("state", STRING_TYPE), Column("run_id", STRING_TYPE)],
        row_epoch=row_versions("state"),
        l1_eviction=TEST_L1_TABLE,
    )

    async def window() -> int | None:
        return None

    with pytest.raises(ValueError, match="names no scopes"):
        QuietCleanup(object(), (versioned,), LoadGate.from_change_log(_HeardLog().on_record), window)
