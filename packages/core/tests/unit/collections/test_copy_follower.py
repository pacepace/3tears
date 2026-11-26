"""a copy follower keeps whole tables current by the keys each epoch changed, at one settled stamp.

The registry is the standard one, its marks moved by epoch change records a fake log holds (as the epoch bucket
does); L3 is a DuckDB answering the follower's own statements. The contract this pins:

- a first warm reads every table whole, and the first pass labels the copy with the settled stamp;
- a change reads from L3 exactly the keys its epochs named, whatever the table holds, and leaves the copy equal
  to L3 (an insert, an update and a delete alike);
- a write in progress, or a stamp that moves while a pass reads, applies nothing and labels nothing; the next pass
  tops up what was staged and applies it at its own stamp;
- an epoch the log cannot give leaves the copy as it was and the follower behind; never a drop;
- a whole-table record reads that table again, beside the copy;
- a pass that fails part way leaves the copy as it was, and the next pass applies the same keys again;
- a reader's stamp is exactly the change its snapshot holds;
- a commit too large to record (keys unknown, with its notes bracket) reaches the copy exactly from the keys its
  change notes name; notes that cannot name them, or a hub with no door to ask, leave the copy as it was and the
  follower behind with why, until the log is put back; one whose record names its delta file is applied from the
  file, so a follower restarted from a checkpoint older than the notes still catches up.
"""

from __future__ import annotations

import bisect
import asyncio
import json
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from typing import Any

import pytest
from sqlalchemy import BigInteger, Column as SaColumn, MetaData, String, Table
from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE

pa = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")

import threetears.core as core  # noqa: E402
from threetears.core.cache.duckdb import STAGE_PREFIX, DuckDBBackend, StagedChange  # noqa: E402
from threetears.core.cache.duckdb_binds import run_bound  # noqa: E402
from threetears.core.collections import CollectionRegistry  # noqa: E402
from threetears.core.collections.complete_copy import Unsettled  # noqa: E402
from threetears.core.collections.registry import BEHIND_REASONS, CacheInvalidationMessage  # noqa: E402
from threetears.core.collections.copy_checkpoint import Checkpoint, CopyCheckpoints  # noqa: E402
from threetears.core.backends.broker_generation import GENERATION_BY_TRIGGER  # noqa: E402
from threetears.core.collections.scope_epochs import ScopeEpochs  # noqa: E402
import threetears.core.collections.copy_follower as copy_follower  # noqa: E402
from threetears.core.collections.copy_queries import CopyQueries, CopyVersion  # noqa: E402
from threetears.core.collections.copy_key_work import INLINE_KEYS  # noqa: E402
from packages.core.tests.copy_workers_support import COPY_WORKERS as _WORKERS  # noqa: E402
from packages.core.tests.unit.collections import copy_query_jobs  # noqa: E402
from threetears.core.process_workers import ProcessWorkers  # noqa: E402
from threetears.core.collections.copy_follower import (  # noqa: E402
    FAILING_AFTER,
    WHOLE_READ_WHYS,
    CopyFollower,
    FollowedTable,
    FollowerPhase,
    HubCapabilityMissingError,
    PartialStamp,
    require_generation_by_trigger,
)
from threetears.core.testing.published_copy import PublishedCopy  # noqa: E402
from threetears.core.backends.nats_proxy import NatsProxyL3Backend  # noqa: E402
from threetears.core.collections.change_notes import NotedKeysCursor, NotedKeysPage  # noqa: E402
from threetears.core.collections.epoch_change import ChangedRow, EpochChange, EpochWalk, walk_by_count  # noqa: E402
from threetears.nats import Subject  # noqa: E402
from threetears.nats.errors import NoRespondersError, RequestTimeoutError  # noqa: E402
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, rail_arrow_types  # noqa: E402
from threetears.core.collections.delta_files import TableDelta, write_delta_file  # noqa: E402
from prometheus_client import REGISTRY  # noqa: E402
from threetears.core.testing.keyed_statements import as_lists  # noqa: E402
from threetears.core.testing.kv import FakeNatsClient  # noqa: E402

_TABLE = "votes"
_KEY = ("state", "geo_id")
_COLUMNS = (
    Column(name="state", column_type=STRING_TYPE),
    Column(name="geo_id", column_type=BIGINT_TYPE),
    Column(name="votes", column_type=BIGINT_TYPE),
)


def _whole_reads_counted(table: str, why: str) -> float:
    return REGISTRY.get_sample_value("copy_follower_whole_reads_total", {"table": table, "why": why}) or 0.0


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        _TABLE,
        metadata,
        SaColumn("state", String(8), primary_key=True),
        SaColumn("geo_id", BigInteger, primary_key=True),
        SaColumn("votes", BigInteger),
        info={L1_EVICTION_INFO: TEST_L1_TABLE},
    )
    return metadata


def _dict_rows(result: Any) -> list[dict[str, Any]]:
    names = [d[0] for d in result.description]
    return [dict(zip(names, row, strict=True)) for row in result.fetchall()]


#: how each DuckDB a test opened is closed when it ends (:func:`_close_what_the_test_opened`)
_OPENED: list[Callable[[], None]] = []


def _backend() -> DuckDBBackend:
    """a DuckDB L1 the test's end closes."""
    backend = DuckDBBackend()
    _OPENED.append(backend.reset)
    return backend


@pytest.fixture(autouse=True)
def _close_what_the_test_opened() -> Iterator[None]:
    """close every DuckDB the test opened once it ends.

    Each in-memory DuckDB runs a worker thread a core, and a test's L3 and copy are held in cycles (the follower, its
    registry), so left open they lived until a collection of the old generation: this module alone took one process
    to 725 threads on a 10-core host, and on a pool runner of many cores the next test to start one failed with
    "can't start new thread" (test_keyed_writes' TestApplyStageAppend[duckdb], 2026-11-16).
    """
    yield
    while _OPENED:
        _OPENED.pop()()


class _L3:
    """L3 as a DuckDB, answering ``fetch`` with ``$n`` placeholders; counts the rows each statement returned."""

    async def answers_from_yugabyte(self) -> bool:
        """a PostgreSQL reader's: its keyed batches bounded beside their keys, and the question sent nowhere."""
        return False

    def __init__(self) -> None:
        self.db = duckdb.connect(":memory:")
        _OPENED.append(self.db.close)
        self.db.execute(
            f"CREATE TABLE {_TABLE} (state VARCHAR, geo_id BIGINT, votes BIGINT, PRIMARY KEY (state, geo_id))"
        )
        self.rows_read = 0
        self.statements = 0
        self.fail_next = False

    def put(self, rows: Sequence[tuple[str, int, int]]) -> None:
        # one statement over the rows as a table: a large commit's tens of thousands, row by row, take minutes
        put = pa.table(
            {
                "state": pa.array([r[0] for r in rows], pa.string()),
                "geo_id": pa.array([r[1] for r in rows], pa.int64()),
                "votes": pa.array([r[2] for r in rows], pa.int64()),
            }
        )
        self.db.register("_put", put)
        try:
            self.db.execute(f"INSERT OR REPLACE INTO {_TABLE} (state, geo_id, votes) SELECT * FROM _put")  # noqa: S608
        finally:
            self.db.unregister("_put")

    def delete(self, keys: Sequence[tuple[str, int]]) -> None:
        # one statement over the keys as a table, as ``put``: a statement a key binds each value from Python (a failed
        # ``import pandas`` a value), which held the loop past the loop-stall guard's limit
        gone = pa.table(
            {"state": pa.array([k[0] for k in keys], pa.string()), "geo_id": pa.array([k[1] for k in keys], pa.int64())}
        )
        self.db.register("_gone", gone)
        try:
            self.db.execute(
                f"DELETE FROM {_TABLE} AS t USING _gone AS g WHERE t.state = g.state AND t.geo_id = g.geo_id"  # noqa: S608
            )
        finally:
            self.db.unregister("_gone")

    def held(self) -> list[tuple[Any, ...]]:
        return sorted(self.db.execute(f"SELECT state, geo_id, votes FROM {_TABLE}").fetchall())

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        await asyncio.sleep(0)
        if self.fail_next:
            self.fail_next = False
            raise ConnectionError("the L3 rail dropped the reply")
        cursor = self.db.cursor()
        # bound as the 3tears DuckDB readers bind (run_bound): a page of keys through Arrow, never value by value
        rows = run_bound(cursor, sql, params, _dict_rows)
        if sql.startswith("SELECT version()"):
            # which database it is (threetears.core.backends.dialect), asked beside a whole read: no read of the table
            return rows
        self.statements += 1
        self.rows_read += len(rows)
        return rows


class _Log:
    """one table's write generation and its change records, as the epoch bucket keeps them."""

    def __init__(self) -> None:
        self.count = 0
        self.records: dict[str, EpochChange] = {}

    async def current(self, table: str) -> str:
        return f"inc:{self.count}"

    def commit(self, keys: Sequence[tuple[Any, Any]] | None, *, keep: bool = True) -> str:
        self.count += 1
        token = f"inc:{self.count}"
        if keep:
            self.records[token] = EpochChange(
                table=_TABLE,
                token=token,
                whole_table=keys is None,
                rows=[ChangedRow(ids=[s, str(g)]) for s, g in keys or ()],
            )
        return token

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        return self.records.get(token)

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        return await walk_by_count(self, table_name, after, upto)


class _Seqlock:
    """the writer's stamp: a version, or a write in progress."""

    def __init__(self) -> None:
        self.version = 0
        self.writing = False
        self.reads = 0
        self.move_on_read: int | None = None

    async def settled(self) -> Any:
        self.reads += 1
        if self.move_on_read is not None and self.reads == self.move_on_read:
            # a write commits between this pass's first read and its second
            self.version += 1
        return Unsettled("a write is in progress") if self.writing else self.version


@pytest.fixture
def world() -> tuple[_L3, _Log, _Seqlock, DuckDBBackend, CopyFollower]:
    return _world()[0]


@pytest.fixture
def noted_world() -> tuple[tuple[_L3, _Log, _Seqlock, DuckDBBackend, CopyFollower], CollectionRegistry]:
    return _world()


def _world(
    retry_waits: tuple[float, float] = (1.0, 60.0),
    page_size: int = 50,
    workers: ProcessWorkers | None = None,
) -> tuple[tuple[_L3, _Log, _Seqlock, DuckDBBackend, CopyFollower], CollectionRegistry]:
    l3, log, lock = _L3(), _Log(), _Seqlock()
    backend = _backend()
    backend.initialize(_metadata())
    registry = CollectionRegistry()
    registry.follow_generation(_TABLE)
    published = PublishedCopy(backend)
    follower = CopyFollower(
        registry,
        [FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS)],
        published,
        l3,
        settled=lock.settled,
        generation=log.current,
        changes=log,
        page_size=page_size,
        retry_waits=retry_waits,
        drift_workers=_WORKERS if workers is None else workers,
        queries=published.queries,
    )
    follower.register()
    return (l3, log, lock, backend, follower), registry


def _copy(backend: DuckDBBackend) -> list[tuple[Any, ...]]:
    with backend.read_snapshot() as cursor:
        return sorted(cursor.execute(f"SELECT state, geo_id, votes FROM {_TABLE}").fetchall())


def _rows(state: str, count: int, votes: int = 1) -> list[tuple[str, int, int]]:
    return [(state, i, votes) for i in range(count)]


async def _warmed(world: tuple[_L3, _Log, _Seqlock, DuckDBBackend, CopyFollower], rows: int) -> None:
    l3, log, lock, backend, follower = world
    l3.put(_rows("VA", rows))
    log.commit([("VA", i) for i in range(rows)])
    await follower.warm()
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()


async def test_a_warm_reads_every_table_whole_and_the_first_pass_labels_it(world: Any) -> None:
    l3, _log, lock, backend, follower = world
    await _warmed(world, 120)
    assert follower.stamp == lock.version
    assert follower.status().phase is FollowerPhase.FOLLOWING
    assert backend.staged_stages() == []


async def test_a_change_reads_exactly_its_keys_and_leaves_the_copy_equal_to_l3(world: Any) -> None:
    l3, log, lock, backend, follower = world
    await _warmed(world, 120)
    l3.rows_read = 0
    l3.put([("VA", 3, 99), ("VA", 500, 7)])  # an update and an insert
    l3.delete([("VA", 4)])  # a delete
    log.commit([("VA", 3), ("VA", 500), ("VA", 4)])
    lock.version += 1
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()
    # the keys' rows that exist, never the table
    assert l3.rows_read == 2
    assert follower.stamp == lock.version


@pytest.mark.parametrize("held", [100, 1000])
async def test_a_changes_reads_do_not_grow_with_the_table(world: Any, held: int) -> None:
    l3, log, lock, _backend, follower = world
    await _warmed(world, held)
    l3.rows_read, l3.statements = 0, 0
    l3.put([("VA", 1, 5)])
    log.commit([("VA", 1)])
    lock.version += 1
    assert await follower.catch_up()
    assert (l3.rows_read, l3.statements) == (1, 1)


async def test_a_write_in_progress_applies_nothing(world: Any) -> None:
    l3, log, lock, backend, follower = world
    await _warmed(world, 10)
    before = _copy(backend)
    l3.put([("VA", 1, 5)])
    log.commit([("VA", 1)])
    lock.writing = True
    assert not await follower.catch_up()
    assert _copy(backend) == before
    assert follower.status().phase is FollowerPhase.WAITING
    lock.writing = False
    lock.version += 1
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()


async def test_a_stamp_that_moves_while_a_pass_reads_keeps_its_stage_and_the_next_pass_tops_it_up(world: Any) -> None:
    l3, log, lock, backend, follower = world
    await _warmed(world, 10)
    before, stamp = _copy(backend), follower.stamp
    l3.put([("VA", 1, 5)])
    log.commit([("VA", 1)])
    lock.move_on_read = lock.reads + 2
    assert not await follower.catch_up()
    assert _copy(backend) == before
    assert follower.stamp == stamp
    assert backend.staged_stages()  # kept for the next pass
    # the write that moved the stamp changed another key, and the first again
    l3.put([("VA", 2, 6), ("VA", 1, 8)])
    log.commit([("VA", 2), ("VA", 1)])
    l3.rows_read = 0
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()
    assert l3.rows_read == 2  # only the keys heard since
    assert follower.stamp == lock.version
    assert backend.staged_stages() == []


async def test_an_epoch_the_log_cannot_give_leaves_the_copy_and_says_behind(world: Any) -> None:
    l3, log, lock, backend, follower = world
    await _warmed(world, 10)
    before, stamp = _copy(backend), follower.stamp
    l3.put([("VA", 1, 5)])
    log.commit([("VA", 1)], keep=False)
    lock.version += 1
    assert not await follower.catch_up()
    assert _copy(backend) == before
    assert follower.stamp == stamp
    status = follower.status()
    assert status.phase is FollowerPhase.BEHIND
    assert _TABLE in status.behind
    # the log put back (the keeper's restore): the catch-up applies it
    log.records["inc:2"] = EpochChange(table=_TABLE, token="inc:2", rows=[ChangedRow(ids=["VA", "1"])])
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()


async def test_a_whole_table_record_reads_the_table_again_beside_the_copy(world: Any) -> None:
    l3, log, lock, backend, follower = world
    await _warmed(world, 10)
    l3.delete([("VA", i) for i in range(10)])
    l3.put(_rows("OH", 7, votes=3))
    log.commit(None)
    lock.version += 1
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()


async def test_a_pass_that_fails_part_way_leaves_the_copy_and_the_next_applies_the_same_keys(world: Any) -> None:
    l3, log, lock, backend, follower = world
    await _warmed(world, 10)
    before = _copy(backend)
    l3.put([("VA", 1, 5)])
    l3.delete([("VA", 2)])
    log.commit([("VA", 1), ("VA", 2)])
    lock.version += 1
    l3.fail_next = True
    with pytest.raises(ConnectionError):
        await follower.catch_up()
    assert _copy(backend) == before
    assert backend.staged_stages() == []
    assert follower.status().pending == {_TABLE: 2}
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()


async def _votes_of_geo_1(state: Any) -> list[Any]:
    """geo 1's votes in a pinned state of the copy, read in a query worker."""
    rows = await state.run_job(copy_query_jobs.sql, (f"SELECT votes FROM {_TABLE} WHERE geo_id = 1",))
    return list(rows.column("votes").to_pylist())


async def test_a_readers_stamp_is_the_change_its_pinned_state_holds(world: Any) -> None:
    l3, log, lock, backend, follower = world
    await _warmed(world, 10)
    async with follower.current() as (state, stamp, _marks):
        l3.put([("VA", 1, 5)])
        log.commit([("VA", 1)])
        lock.version += 1
        assert await follower.catch_up()
        # the old pin still reads the old rows under the old stamp
        assert await _votes_of_geo_1(state) == [1]
        assert stamp == lock.version - 1
    async with follower.current() as (state, stamp, _marks):
        assert await _votes_of_geo_1(state) == [5]
        assert stamp == lock.version


async def test_the_keyed_read_joins_the_keys_to_the_table_on_its_key(world: Any) -> None:
    l3, log, lock, _backend, follower = world
    await _warmed(world, 10)
    asked: list[str] = []
    fetch = l3.fetch

    async def recording(sql: str, *params: Any) -> list[dict[str, Any]]:
        # each key column spelled as the list it binds, however it was bound (the page's Arrow key columns: one text
        # a column, split back in SQL)
        asked.append(as_lists(sql, params)[0])
        rows: list[dict[str, Any]] = await fetch(sql, *params)
        return rows

    l3.fetch = recording
    l3.put([("VA", 1, 5), ("TX", 2, 6), ("CA", 3, 7)])
    log.commit([("VA", 1), ("TX", 2), ("CA", 3)])
    lock.version += 1
    assert await follower.catch_up()
    (sql,) = asked
    # the keys joined to the table as rows (each array cast whole), batched on YugabyteDB; never a column = ANY its
    # values, which names their cross product (3 states x 3 geographies: 9 keys for 3; keys sharing their lead are
    # bound to them too, test_keyed_helpers)
    assert sql.startswith("/*+ YbBatchedNL(ANY_subquery t) */ "), sql
    assert '(t."state", t."geo_id") IN (SELECT unnest($1::text[]) AS k0, unnest($2::text[]::bigint[]) AS k1)' in sql
    assert "= ANY(" not in sql, sql


# --- checkpoints ---------------------------------------------------------------------------------


class _Store:
    """an Object Store in memory; a put can be made to fail."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.fail_put_at: int | None = None
        self.puts = 0
        self.retired: list[str] = []

    async def put(self, name: str, data: bytes) -> None:
        self.puts += 1
        if self.fail_put_at is not None and self.puts == self.fail_put_at:
            raise ConnectionError("the store went away mid-write")
        self.objects[name] = data

    async def get(self, name: str) -> bytes:
        return self.objects[name]

    async def require_room(self, objects: Sequence[tuple[str, int]]) -> int:
        """no bound: the room a checkpoint asks for is always there."""
        return sum(size for _, size in objects)

    async def list_names(self, *, prefix: str = "") -> list[str]:
        return [n for n in self.objects if n.startswith(prefix)]

    async def retire(self, names: list[str]) -> None:
        self.retired.extend(names)
        for name in names:
            self.objects.pop(name, None)


class _Pointers:
    """a pointer bucket in memory: create-if-absent and compare-and-set by revision."""

    def __init__(self) -> None:
        self.entries: dict[str, tuple[bytes, int]] = {}
        self.revision = 0

    async def get_entry(self, *, key: str) -> tuple[bytes, int] | None:
        return self.entries.get(key)

    async def create(self, *, key: str, value: bytes, ttl: Any = None) -> int | None:
        if key in self.entries:
            return None
        self.revision += 1
        self.entries[key] = (value, self.revision)
        return self.revision

    async def update(self, *, key: str, value: bytes, revision: int, ttl: Any = None) -> int | None:
        held = self.entries.get(key)
        if held is None or held[1] != revision:
            return None
        self.revision += 1
        self.entries[key] = (value, self.revision)
        return self.revision

    async def delete(self, *, key: str, revision: int | None = None, marker_ttl: Any = None) -> bool:
        return self.entries.pop(key, None) is not None

    async def list_keys(self, *, prefix: str = "") -> list[str]:
        return [k for k in self.entries if k.startswith(prefix)]


def _replica(
    l3: _L3,
    log: _Log,
    lock: _Seqlock | _Scopes,
    store: _Store,
    pointers: _Pointers,
    owner: str,
    part_rows: int = 40,
    notes: Any = None,
    delta_files: bool = False,
    **stamps: Any,
) -> tuple[DuckDBBackend, CopyFollower, Any]:
    backend = _backend()
    backend.initialize(_metadata())
    tables = [
        FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS, scope=("state", "state:"), delta_files=delta_files)
    ]
    checkpoints = CopyCheckpoints(
        name="votes",
        tables=tables,
        backend=backend,
        store=store,
        pointers=pointers,
        retire=store.retire,
        owner=owner,
        part_rows=part_rows,
        **stamps,
    )
    registry = CollectionRegistry()
    registry.follow_generation(_TABLE)
    registry.set_noted_keys_reader(notes)
    published = PublishedCopy(backend)
    follower = CopyFollower(
        registry,
        tables,
        published,
        l3,
        settled=lock.settled,
        generation=log.current,
        changes=log,
        page_size=50,
        checkpoints=checkpoints,
        stamp_table="scope_epochs",
        drift_workers=_WORKERS,
        queries=published.queries,
    )
    return backend, follower, checkpoints


async def test_a_start_loads_the_checkpoint_and_walks_the_log_from_its_marks_never_reading_l3_whole() -> None:
    l3, log, lock, store, pointers = _L3(), _Log(), _Seqlock(), _Store(), _Pointers()
    l3.put(_rows("VA", 200))
    log.commit([("VA", i) for i in range(200)])
    first, writer, _ = _replica(l3, log, lock, store, pointers, "a")
    assert not await writer.start()  # no checkpoint yet: the first start reads L3
    assert await writer.catch_up()
    await writer.checkpointed()
    assert pointers.entries  # the first warm made one due
    # a change after the checkpoint, which the second replica must take from the log
    l3.put([("VA", 7, 70), ("VA", 900, 1)])
    log.commit([("VA", 7), ("VA", 900)])
    lock.version += 1
    l3.rows_read, l3.statements = 0, 0
    second, follower, _ = _replica(l3, log, lock, store, pointers, "b")
    assert await follower.start()
    assert await follower.catch_up()
    assert _copy(second) == l3.held()
    assert (l3.rows_read, l3.statements) == (2, 1)  # the change's keys alone
    assert follower.stamp == lock.version


async def test_a_checkpoint_written_part_way_moves_no_pointer_and_leaves_nothing_behind() -> None:
    l3, log, lock, store, pointers = _L3(), _Log(), _Seqlock(), _Store(), _Pointers()
    l3.put(_rows("VA", 200))
    log.commit([("VA", i) for i in range(200)])
    _, writer, checkpoints = _replica(l3, log, lock, store, pointers, "a")
    await writer.start()
    assert await writer.catch_up()
    await writer.checkpointed()
    before = dict(pointers.entries)
    kept = set(store.objects)
    store.fail_put_at = store.puts + 2
    # nothing changed since, so an incremental write would reuse every part: a table read whole again (a reset) writes
    # in full
    checkpoints.applied(1, {_TABLE: None}, whys={_TABLE: "reset"})
    with pytest.raises(ConnectionError):
        await checkpoints.write(writer.current)
    # the pointer still names the old checkpoint, every part of it is there, and no claim or lock is left
    assert pointers.entries == before
    assert set(store.objects) == kept
    # a start from it still loads whole
    second, follower, _ = _replica(l3, log, lock, store, pointers, "b")
    assert await follower.start()
    assert await follower.catch_up()
    assert _copy(second) == l3.held()


async def test_a_checkpoint_beaten_to_the_pointer_retires_its_own_parts() -> None:
    l3, log, lock, store, pointers = _L3(), _Log(), _Seqlock(), _Store(), _Pointers()
    l3.put(_rows("VA", 50))
    log.commit([("VA", i) for i in range(50)])
    _, writer, checkpoints = _replica(l3, log, lock, store, pointers, "a")
    await writer.start()
    assert await writer.catch_up()
    await writer.checkpointed()
    named = set(store.objects)
    real_update = pointers.update

    async def beaten(*, key: str, value: bytes, revision: int, ttl: Any = None) -> int | None:
        if key == checkpoints.base:
            await real_update(key=key, value=pointers.entries[key][0], revision=pointers.entries[key][1])
        return await real_update(key=key, value=value, revision=revision, ttl=ttl)

    pointers.update = beaten  # type: ignore[method-assign]
    assert await checkpoints.write(writer.current) is None
    assert set(store.objects) == named


async def test_a_lock_another_replica_holds_skips_the_write() -> None:
    l3, log, lock, store, pointers = _L3(), _Log(), _Seqlock(), _Store(), _Pointers()
    l3.put(_rows("VA", 5))
    log.commit([("VA", i) for i in range(5)])
    _, writer, checkpoints = _replica(l3, log, lock, store, pointers, "a")
    await writer.start()
    assert await writer.catch_up()
    await writer.checkpointed()
    await pointers.create(key=f"{checkpoints.base}-lock", value=b"b")
    assert await checkpoints.write(writer.current) is None


async def test_a_checkpoint_is_due_by_the_rows_applied_since_never_by_time() -> None:
    l3, log, lock, store, pointers = _L3(), _Log(), _Seqlock(), _Store(), _Pointers()
    l3.put(_rows("VA", 100))
    log.commit([("VA", i) for i in range(100)])
    _, writer, checkpoints = _replica(l3, log, lock, store, pointers, "a")
    await writer.start()
    assert await writer.catch_up()
    await writer.checkpointed()
    assert not checkpoints.due()
    checkpoints.applied(49, {})
    assert not checkpoints.due()
    checkpoints.applied(1, {})
    assert checkpoints.due()


async def test_a_part_that_does_not_read_back_whole_makes_the_start_read_l3() -> None:
    l3, log, lock, store, pointers = _L3(), _Log(), _Seqlock(), _Store(), _Pointers()
    l3.put(_rows("VA", 100))
    log.commit([("VA", i) for i in range(100)])
    _, writer, _ = _replica(l3, log, lock, store, pointers, "a")
    await writer.start()
    assert await writer.catch_up()
    await writer.checkpointed()
    name = next(iter(store.objects))
    store.objects[name] = store.objects[next(n for n in store.objects if n != name)][:0] or b"broken"
    second, follower, _ = _replica(l3, log, lock, store, pointers, "b")
    assert not await follower.start()
    assert second.staged_stages() == []
    assert await follower.catch_up()
    assert _copy(second) == l3.held()


# --- the hub it needs ----------------------------------------------------------------------------


class _Hub:
    """an L3 reached through the hub's broker, announcing what it was built with."""

    def __init__(self, announced: frozenset[str]) -> None:
        self.announced = announced

    async def hub_capabilities(self) -> frozenset[str]:
        return self.announced


async def test_a_hub_that_does_not_advance_trigger_switched_tables_is_refused_by_name() -> None:
    with pytest.raises(HubCapabilityMissingError, match=GENERATION_BY_TRIGGER):
        await require_generation_by_trigger(_Hub(frozenset()))


async def test_a_hub_that_announces_it_is_accepted() -> None:
    await require_generation_by_trigger(_Hub(frozenset({GENERATION_BY_TRIGGER, "other"})))


async def test_an_l3_with_no_broker_between_is_not_asked() -> None:
    await require_generation_by_trigger(_L3())


# --- a load's partition ----------------------------------------------------------------------------


async def test_a_loads_partition_reads_only_its_rows_and_leaves_the_copy_equal_to_l3(world: Any) -> None:
    l3, log, lock, backend, follower = world
    await _warmed(world, 200)
    # a load from files: every row it loaded holds its constants (here the state), recorded as the load's partition
    l3.put(_rows("OH", 30, votes=4))
    log.count += 1
    token = f"inc:{log.count}"
    log.records[token] = EpochChange(table=_TABLE, token=token, whole_table=True, partition={"state": "OH"})
    lock.version += 1
    l3.rows_read = 0
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()
    assert l3.rows_read == 30  # the load's rows, never the table's 230


async def test_a_partition_then_a_key_it_holds_leaves_the_keys_newest_row(world: Any) -> None:
    l3, log, lock, backend, follower = world
    await _warmed(world, 10)
    l3.put(_rows("OH", 5, votes=4))
    log.count += 1
    log.records[f"inc:{log.count}"] = EpochChange(
        table=_TABLE, token=f"inc:{log.count}", whole_table=True, partition={"state": "OH"}
    )
    l3.put([("OH", 2, 99)])
    l3.delete([("OH", 3)])
    log.commit([("OH", 2), ("OH", 3)])
    lock.version += 1
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()


async def test_a_partition_neither_the_copy_nor_l3_holds_a_row_of_names_no_stage_it_never_staged(world: Any) -> None:
    """Flips if a partition that staged nothing hands the backend a change naming its keys or rows stage anyway: the
    backend never made them, its apply failed on the missing table, and every pass after failed the same way, the
    copy stuck (local burst replay, 0.95.0 gate 1: ``__stage__lsf_projections__keys does not exist``, 99 in a row)."""
    l3, log, lock, backend, follower = world
    await _warmed(world, 10)
    # a load's partition of a state neither the copy nor L3 holds a row of (a reload that emptied it, an empty state)
    log.count += 1
    log.records[f"inc:{log.count}"] = EpochChange(
        table=_TABLE, token=f"inc:{log.count}", whole_table=True, partition={"state": "OH"}
    )
    lock.version += 1
    assert await follower.catch_up()
    assert follower.status().failures == 0
    assert _copy(backend) == l3.held()
    assert backend.staged_stages() == []
    # and a key after it applies: the follower is not stuck
    l3.put([("VA", 3, 7)])
    log.commit([("VA", 3)])
    lock.version += 1
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()


async def test_a_derived_cache_is_told_a_records_partition_and_never_dropped_whole_for_it() -> None:
    """Flips if a load's partition drops a derived cache whole (decision 33: every derived cache takes slices) instead
    of telling it the slice."""
    registry = CollectionRegistry()
    heard: list[Any] = []
    registry.register_derived_cache(
        _TABLE, on_row=lambda message: None, on_table_dropped=lambda: heard.append("all"), on_partition=heard.append
    )
    await registry.apply_epoch_change(
        EpochChange(table=_TABLE, token="inc:1", whole_table=True, partition={"state": "OH"})
    )
    assert heard == [{"state": "OH"}]


async def test_a_partition_on_a_column_the_copy_does_not_hold_filters_l3_by_it() -> None:
    """a load's constants include a version stamp the copy does not hold: L3 filters on it, and only that load's rows
    are read."""
    l3, log, lock = _L3(), _Log(), _Seqlock()
    l3.db.execute(f"ALTER TABLE {_TABLE} ADD COLUMN ver BIGINT DEFAULT 0")
    backend = _backend()
    backend.initialize(_metadata())
    registry = CollectionRegistry()
    registry.follow_generation(_TABLE)
    table = FollowedTable(
        name=_TABLE,
        key=_KEY,
        columns=_COLUMNS,
        l3_kinds=(("state", STRING_TYPE), ("geo_id", BIGINT_TYPE), ("votes", BIGINT_TYPE), ("ver", BIGINT_TYPE)),
    )
    published = PublishedCopy(backend)
    follower = CopyFollower(
        registry,
        [table],
        published,
        l3,
        settled=lock.settled,
        generation=log.current,
        changes=log,
        page_size=50,
        drift_workers=_WORKERS,
        queries=published.queries,
    )
    follower.register()
    l3.put(_rows("VA", 40))
    await follower.warm()
    assert await follower.catch_up()
    # one statement with no parameters: a statement a row binds each value from Python (a failed ``import pandas`` a
    # value), which held the loop past the loop-stall guard's limit
    l3.db.execute(f"INSERT INTO {_TABLE} SELECT 'OH', i, 2, 5 FROM range(12) AS r(i)")  # noqa: S608
    log.count += 1
    log.records[f"inc:{log.count}"] = EpochChange(
        table=_TABLE, token=f"inc:{log.count}", whole_table=True, partition={"ver": "5"}
    )
    lock.version += 1
    l3.rows_read = 0
    legacy = _whole_reads_counted(_TABLE, "legacy_whole_record")
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()
    # 0.95.0: a partition no key leads is an older writer's whole record, read once and counted, never a scan by it
    assert l3.rows_read == 53  # 52 rows, a page and its one-row look past
    assert _whole_reads_counted(_TABLE, "legacy_whole_record") == legacy + 1


class _Notes:
    """the table's change notes of the follower's tenant, paged as the broker's l3.noted_keys door pages them."""

    def __init__(self, notes: Sequence[tuple[datetime, tuple[str, ...]]]) -> None:
        self.notes = sorted(notes)
        #: each note's time, in the notes' order: a page found by bisection, as the broker's index seeks it, not a walk
        #: of every note a page (tens of thousands of them, which held the loop past the loop-stall guard's limit)
        self.times = [at for at, _ in self.notes]
        self.pages = 0

    async def noted_keys(
        self,
        table: str,
        *,
        noted_between: tuple[datetime, datetime],
        cursor: NotedKeysCursor | None,
        limit: int,
        max_bytes: int,
    ) -> NotedKeysPage | str:
        self.pages += 1
        since, until = noted_between
        after = cursor.after if cursor is not None and cursor.after is not None else since - timedelta(microseconds=1)
        start, end = bisect.bisect_right(self.times, after), bisect.bisect_right(self.times, until)
        taken, more = self.notes[start : min(end, start + limit)], end - start > limit
        if more:
            taken = [note for note in taken if note[0] < self.notes[start + limit][0]]
        return NotedKeysPage(
            keys=tuple(keys for _, keys in taken),
            whole_table=False,
            next=NotedKeysCursor(slot=0, after=taken[-1][0]) if more else None,
        )


def _too_large(log: _Log, *, keep: bool = True, since: datetime | None = None) -> tuple[datetime, datetime]:
    """a commit the broker recorded keys unknown, with its notes bracket (beginning now, unless ``since`` says)."""
    since = since if since is not None else datetime.now(UTC)
    until = since + timedelta(seconds=1)
    log.count += 1
    token = f"inc:{log.count}"
    if keep:
        log.records[token] = EpochChange(table=_TABLE, token=token, keys_unknown=True, noted_between=(since, until))
    return since, until


#: a commit naming more rows than sixteen change records of 256 KiB hold, at about sixty bytes a named row
_PAST_SIXTEEN_RECORDS = 70_000


def _committed_past_sixteen_records(l3: _L3) -> tuple[list[tuple[str, int, int]], list[tuple[str, int]]]:
    """one commit's rows put in L3: updates of the 120 rows held, inserts past them, and ten deletes."""
    changed = [("VA", i, 7) for i in range(60, _PAST_SIXTEEN_RECORDS)]
    deleted = [("VA", i) for i in range(10, 20)]
    l3.put(changed)
    l3.delete(deleted)
    return changed, deleted


def _notes_of(since: datetime, changed: Sequence[tuple[str, int, int]], deleted: Sequence[tuple[str, int]]) -> _Notes:
    """the commit's change notes from ``since``, four keys a microsecond."""
    keys = [(s, str(g)) for s, g, _ in changed] + [(s, str(g)) for s, g in deleted]
    return _Notes([(since + timedelta(microseconds=index // 4), key) for index, key in enumerate(keys)])


async def test_a_commit_too_large_to_record_reaches_the_copy_exactly_from_its_notes(noted_world: Any) -> None:
    world, registry = noted_world
    l3, log, lock, backend, follower = world
    await _warmed(world, 120)
    # updates of rows held, inserts past them, and deletes, all in one commit: its 70,000 rows and their notes built on
    # a thread, as setup; on the loop they held it past the loop-stall guard's limit (threetears.core.testing.loop_stall)
    changed, deleted = await asyncio.to_thread(_committed_past_sixteen_records, l3)
    since, _ = _too_large(log)
    notes = await asyncio.to_thread(_notes_of, since, changed, deleted)
    registry.set_noted_keys_reader(notes)
    lock.version += 1
    l3.rows_read = 0

    assert await follower.catch_up()

    copied, held = await asyncio.to_thread(lambda: (_copy(backend), l3.held()))
    assert copied == held, "the copy equals L3, row for row"
    assert l3.rows_read == len(changed), "the keys' rows that exist, never the table"
    assert notes.pages > 1
    assert follower.status().behind == {}
    assert follower.stamp == lock.version


async def test_a_large_commits_noted_keys_reach_a_copy_as_one_arrow_batch_a_page(noted_world: Any) -> None:
    """Flips if a page of the keys the change notes name reaches a cache that takes key batches (a copy, ``on_keys``)
    as anything but one Arrow batch a page (a row message a key: a Python object a key on the loop, decision 35), or
    if a row message is built at all when every cache of the table takes batches."""
    world, registry = noted_world
    l3, log, lock, _backend, follower = world
    await _warmed(world, 10)
    changed = [("VA", i, 3) for i in range(10, 25_010)]
    l3.put(changed)
    since, _ = _too_large(log)
    keys = [(s, str(g)) for s, g, _ in changed]
    notes = _Notes([(since + timedelta(microseconds=index), key) for index, key in enumerate(keys)])
    registry.set_noted_keys_reader(notes)
    batches: list[Any] = []
    rows: list[Any] = []
    registry.register_derived_cache(
        _TABLE,
        on_row=rows.append,
        on_table_dropped=lambda: None,
        on_partition=lambda _prefix: None,
        on_keys=lambda batch, epoch: batches.append((batch, epoch)),
    )
    lock.version += 1

    assert await follower.catch_up()

    assert rows == [], "no row message a key when every cache of the table takes batches"
    assert len(batches) == notes.pages > 1, "one batch a page"
    assert all(isinstance(batch, pa.Table) and epoch is None for batch, epoch in batches)
    heard = pa.concat_tables([batch for batch, _ in batches])
    assert sorted(zip(*(heard.column(at).to_pylist() for at in range(2)), strict=True)) == sorted(keys)


async def test_notes_that_cannot_name_a_large_commit_leave_the_copy_behind_until_the_log_is_put_back(
    noted_world: Any,
) -> None:
    world, registry = noted_world
    l3, log, lock, backend, follower = world
    await _warmed(world, 10)
    before, stamp = _copy(backend), follower.stamp
    l3.put([("VA", 1, 5)])
    _too_large(log)

    class _Emptied:
        async def noted_keys(self, table: str, **_: Any) -> str:
            return "not_covered"

    registry.set_noted_keys_reader(_Emptied())
    lock.version += 1
    assert not await follower.catch_up()
    assert (_copy(backend), follower.stamp) == (before, stamp), "never a drop, never a rebuild"
    status = follower.status()
    assert status.phase is FollowerPhase.BEHIND
    assert status.behind == {_TABLE: "notes_not_covered"}
    # the keeper's restore puts the log back from the notes: the catch-up applies it
    log.records["inc:2"] = EpochChange(table=_TABLE, token="inc:2", rows=[ChangedRow(ids=["VA", "1"])])
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()
    assert follower.status().behind == {}


def _silent_hub(error: Exception | None) -> NatsProxyL3Backend:
    """the pod's broker transport on a bus where no hub answers l3.noted_keys: none listens, or none replies."""

    async def request_raw(*, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        if error is not None:
            raise error
        assert timeout is not None
        await asyncio.sleep(timeout.total_seconds())
        raise RequestTimeoutError(f"no reply on {subject.path}")

    nats = MagicMock()
    nats.request_raw = AsyncMock(side_effect=request_raw)
    nats.max_payload = 1024 * 1024
    return NatsProxyL3Backend(
        nats_client=nats,
        namespace_prefix="test",
        default_namespace="tools.reports",
        identity_token=lambda: "token",
        timeout_ms=50,
    )


@pytest.mark.parametrize("error", [NoRespondersError("no responders"), None], ids=["no-responders", "no-reply"])
async def test_a_hub_with_no_noted_keys_door_leaves_the_follower_behind_never_hung(
    noted_world: Any, error: Exception | None
) -> None:
    world, registry = noted_world
    l3, log, lock, backend, follower = world
    await _warmed(world, 10)
    before = _copy(backend)
    l3.put([("VA", 1, 5)])
    _too_large(log)
    registry.set_noted_keys_reader(_silent_hub(error))
    lock.version += 1
    began = time.monotonic()
    async with asyncio.timeout(10):
        assert not await follower.catch_up()
    assert time.monotonic() - began < 5, "bounded by the backend's own timeout, attempts included"
    assert _copy(backend) == before
    assert follower.status().behind == {_TABLE: "notes_not_covered"}


async def test_the_door_answers_a_page_the_client_reads_back_whole() -> None:
    at = datetime(2026, 11, 9, 1, 2, 3, 4, tzinfo=UTC)
    reply = {
        "success": True,
        "answer": "page",
        "keys": [["VA", "1"], ["VA", "2"]],
        "columns": [None, {"state": "VA"}],
        "whole_table": False,
        "next": {"slot": 3, "after": at.isoformat()},
    }
    sent: list[dict[str, Any]] = []

    async def request_raw(*, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        sent.append({"subject": subject.path, **json.loads(payload)})
        return json.dumps(reply).encode()

    nats = MagicMock()
    nats.request_raw = AsyncMock(side_effect=request_raw)
    nats.max_payload = 1000
    proxy = NatsProxyL3Backend(
        nats_client=nats, namespace_prefix="test", default_namespace="tools.reports", identity_token=lambda: "token"
    )
    page = await proxy.noted_keys(
        "votes", noted_between=(at, at), cursor=NotedKeysCursor(slot=2, after=at), limit=10, max_bytes=1 << 20
    )
    assert page == NotedKeysPage(
        keys=(("VA", "1"), ("VA", "2")),
        columns=(None, {"state": "VA"}),
        whole_table=False,
        next=NotedKeysCursor(slot=3, after=at),
    )
    (asked,) = sent
    assert asked["subject"] == "test.l3.noted_keys"
    assert asked["max_bytes"] == 500, "held under half the payload the bus announces"
    assert asked["cursor"] == {"slot": 2, "after": at.isoformat()}
    assert "tenant" not in asked, "the broker names the tenant from the caller's verified schema, never the request"


# --- a checkpoint before the notes it needs can no longer be read ---------------------------------------------------


class _Covered(_Notes):
    """the change notes, until the ring has switched past them: then every bracket is refused (``not_covered``)."""

    def __init__(self, notes: Sequence[tuple[datetime, tuple[str, ...]]]) -> None:
        super().__init__(notes)
        self.emptied = False

    async def noted_keys(self, table: str, **asked: Any) -> NotedKeysPage | str:
        if self.emptied:
            return "not_covered"
        return await super().noted_keys(table, **asked)


def _pointer_seq(pointers: _Pointers) -> int:
    return Checkpoint.decode(pointers.entries["votes-ckpt"][0]).seq


def _notes_age(base: str = "votes-ckpt") -> float | None:
    return REGISTRY.get_sample_value("copy_checkpoint_notes_unsaved_age_seconds", {"base": base})


async def test_an_epoch_taken_from_the_notes_makes_a_checkpoint_due_before_they_go_so_a_restart_is_never_stuck() -> (
    None
):
    """a large commit's few rows never reach the rows ratio, so before the fix no checkpoint followed it: a restart
    loaded the checkpoint from before it, met its epoch again, found its notes emptied and stayed behind for good. Now
    an epoch applied from the notes a quarter fence old makes the checkpoint due at once, and the restart is current.
    """
    l3, log, lock, store, pointers = _L3(), _Log(), _Seqlock(), _Store(), _Pointers()
    l3.put(_rows("VA", 200))
    log.commit([("VA", i) for i in range(200)])
    since = datetime.now(UTC) - timedelta(minutes=3)  # past a quarter of the 10-minute fence
    notes = _Covered([(since, ("VA", "7")), (since, ("VA", "8"))])
    _, writer, _ = _replica(l3, log, lock, store, pointers, "a", notes=notes)
    assert not await writer.start()
    assert await writer.catch_up()
    await writer.checkpointed()
    assert _pointer_seq(pointers) == 1
    l3.put([("VA", 7, 70), ("VA", 8, 80)])
    _too_large(log, since=since)
    lock.version += 1

    assert await writer.catch_up()
    await writer.checkpointed()

    assert _pointer_seq(pointers) == 2, "two rows of 200 held: due by the notes' age alone"
    notes.emptied = True
    second, follower, _ = _replica(l3, log, lock, store, pointers, "b", notes=notes)
    assert await follower.start()
    assert await follower.catch_up()
    assert follower.status().behind == {}
    assert _copy(second) == l3.held()


async def test_a_restart_from_a_checkpoint_older_than_the_notes_catches_up_from_the_commits_delta_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a keys-unknown commit whose record names its delta file is still named from the change notes: on
    2026-11-23 a follower restarted from a checkpoint older than the notes' coverage met such an epoch, found its
    notes gone (``notes_not_covered``) and stayed behind. The file never expires, so the restart applies the whole
    commit from it, rows and deletes, reading neither the notes nor L3 (``design-pod-delta-files.md``, decision 21).
    """
    monkeypatch.setenv("THREETEARS_DELTA_FILES_DIR", str(tmp_path))
    l3, log, lock, store, pointers = _L3(), _Log(), _Seqlock(), _Store(), _Pointers()
    l3.put(_rows("VA", 200))
    log.commit([("VA", i) for i in range(200)])
    notes = _Covered([])
    _, writer, _ = _replica(l3, log, lock, store, pointers, "a", notes=notes, delta_files=True)
    assert not await writer.start()
    assert await writer.catch_up()
    await writer.checkpointed()
    assert _pointer_seq(pointers) == 1
    # a commit too big to list: updates, an insert and deletes, its file saved before it, its record keys unknown
    changed = [("VA", i, 1000 + i) for i in range(50, 120)] + [("VA", 900, 9)]
    delta = TableDelta(name=_TABLE, types=rail_arrow_types(_COLUMNS), key=_KEY)
    delta.rows([{"state": s, "geo_id": g, "votes": v} for s, g, v in changed], ["state", "geo_id", "votes"])
    delta.deletes([("VA", 3), ("VA", 4)])
    name = write_delta_file(tmp_path, delta, "big")
    l3.put(changed)
    l3.delete([("VA", 3), ("VA", 4)])
    since, until = _too_large(log)
    log.records[f"inc:{log.count}"] = EpochChange(
        table=_TABLE, token=f"inc:{log.count}", keys_unknown=True, noted_between=(since, until), delta_file=name
    )
    lock.version += 1
    # the ring switched past the commit's bracket before the restart: the notes can no longer name its keys
    notes.emptied = True
    l3.rows_read, l3.statements = 0, 0

    second, follower, _ = _replica(l3, log, lock, store, pointers, "b", notes=notes, delta_files=True)
    assert await follower.start()
    assert await follower.catch_up()

    assert follower.status().behind == {}, "never notes_not_covered: the file names the commit"
    assert _copy(second) == l3.held()
    assert (l3.statements, notes.pages) == (0, 0), "neither L3 nor the notes read"
    assert follower.stamp == lock.version


async def test_a_quiet_copy_still_checkpoints_an_epoch_taken_from_the_notes_before_they_go() -> None:
    """after a large commit nothing else may be applied: before the fix a checkpoint was asked only after an apply,
    so none came and the epoch's notes went unsaved. Now the follower asks again when the epoch makes one due."""
    l3, log, lock, store, pointers = _L3(), _Log(), _Seqlock(), _Store(), _Pointers()
    l3.put(_rows("VA", 200))
    log.commit([("VA", i) for i in range(200)])
    notes = _Covered([])
    _, writer, _ = _replica(l3, log, lock, store, pointers, "a", notes=notes)
    assert not await writer.start()
    assert await writer.catch_up()
    await writer.checkpointed()
    # due a quarter of the 10-minute fence after the bracket begins: 2 s from now
    due = datetime.now(UTC) + timedelta(seconds=2)
    since = due - timedelta(minutes=10) / 4
    notes.notes = [(since, ("VA", "7"))]
    l3.put([("VA", 7, 70)])
    _too_large(log, since=since)
    lock.version += 1
    assert await writer.catch_up()
    await writer.checkpointed()
    assert _pointer_seq(pointers) == 1, "not due yet"

    async with asyncio.timeout(10):
        while _pointer_seq(pointers) == 1:
            await asyncio.sleep(0.05)
    await writer.checkpointed()
    assert datetime.now(UTC) >= due
    writer.stop()


async def test_how_long_an_epoch_from_the_notes_waits_for_a_checkpoint_is_a_gauge() -> None:
    """the risk is seen before a restart: ``copy_checkpoint_notes_unsaved_age_seconds`` is the age of the oldest epoch
    applied from the notes that no checkpoint holds, and 0 once one does. Before the fix there was no such gauge."""
    l3, log, lock, store, pointers = _L3(), _Log(), _Seqlock(), _Store(), _Pointers()
    l3.put(_rows("VA", 200))
    log.commit([("VA", i) for i in range(200)])
    since = datetime.now(UTC) - timedelta(seconds=60)
    notes = _Covered([(since, ("VA", "7"))])
    _, writer, checkpoints = _replica(l3, log, lock, store, pointers, "a", notes=notes)
    assert not await writer.start()
    assert await writer.catch_up()
    await writer.checkpointed()
    assert _notes_age() == 0.0
    l3.put([("VA", 7, 70)])
    _too_large(log, since=since)
    lock.version += 1
    assert await writer.catch_up()
    await writer.checkpointed()

    age = _notes_age()
    assert age is not None and 60 <= age < 120, "a minute old, not yet due"
    assert await checkpoints.write(writer.current) is not None
    assert _notes_age() == 0.0
    writer.stop()


def test_the_notes_not_covered_reason_names_what_ends_it_for_a_follower_started_from_a_checkpoint() -> None:
    """a restart from a checkpoint older than the epoch adopts its mark and meets the same hole: the reason used to say
    a restart ends it; it names what does."""
    said = BEHIND_REASONS["notes_not_covered"]
    assert "otherwise at a restart" not in said
    assert "CopyCheckpoints.forget" in said
    assert "checkpoint past the epoch" in said


# --- a write in progress holds only its scopes (0.95.0) -------------------------------------------------------------


class _Scopes:
    """the writer's record as ScopeEpochs answers it: each scope's epoch, and the scopes a write in progress holds."""

    def __init__(self) -> None:
        self.version = 1
        self.epochs: dict[str, int] = {"state:VA": 1, "state:IL": 1}
        self.held: frozenset[str] | None = None
        #: the scopes a commit withheld (pending at a settled stamp: ScopeEpochs' "A commit may withhold scopes")
        self.withheld: frozenset[str] = frozenset()
        self.on_read: Any = None
        self.reads = 0

    async def settled(self) -> Any:
        self.reads += 1
        if self.on_read is not None and self.on_read[0] == self.reads:
            self.on_read[1]()
        if self.held is None:
            return SimpleNamespace(version=self.version, epochs=dict(self.epochs), pending=self.withheld)
        return Unsettled("a long write", epochs=dict(self.epochs), pending=self.held, version=self.version)

    def commit(self, *scopes: str) -> None:
        self.version += 1
        for scope in scopes:
            self.epochs[scope] = self.version


def _scoped_world(
    lock: _Scopes,
    *,
    checkpoint_from: DuckDBBackend | None = None,
    page_size: int = 50,
    workers: ProcessWorkers | None = None,
    registry: CollectionRegistry | None = None,
) -> tuple[_L3, _Log, DuckDBBackend, CopyFollower]:
    l3, log = _L3(), _Log()
    backend = _backend()
    backend.initialize(_metadata())
    registry = CollectionRegistry() if registry is None else registry
    registry.follow_generation(_TABLE)
    published = PublishedCopy(backend)
    follower = CopyFollower(
        registry,
        [FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS, scope=("state", "state:"))],
        published,
        l3,
        settled=lock.settled,
        generation=log.current,
        changes=log,
        page_size=page_size,
        stamp_table="scope_epochs",
        drift_workers=_WORKERS if workers is None else workers,
        queries=published.queries,
    )
    follower.register()
    return l3, log, backend, follower


def _of(backend: DuckDBBackend, state: str) -> list[tuple[Any, ...]]:
    return [row for row in _copy(backend) if row[0] == state]


def _l3_of(l3: _L3, state: str) -> list[tuple[Any, ...]]:
    return [row for row in l3.held() if row[0] == state]


async def test_a_commit_beside_a_held_write_reaches_the_copy_and_the_held_scope_stays_settled() -> None:
    lock = _Scopes()
    l3, log, backend, follower = _scoped_world(lock)
    l3.put(_rows("VA", 5) + _rows("IL", 5))
    log.commit([("VA", i) for i in range(5)] + [("IL", i) for i in range(5)])
    await follower.warm()
    assert await follower.catch_up()
    settled_il = _of(backend, "IL")
    # a long AP write holds IL: its rows land part way
    lock.held = frozenset({"state:IL"})
    l3.put([("IL", 0, 50), ("IL", 9, 50)])
    log.commit([("IL", 0), ("IL", 9)])
    # an LSF run commits VA beside it
    l3.put([("VA", 1, 77)])
    log.commit([("VA", 1)])
    lock.commit("state:VA")
    assert await follower.catch_up()
    assert _of(backend, "VA") == _l3_of(l3, "VA")
    assert _of(backend, "IL") == settled_il
    stamp = follower.stamp
    assert isinstance(stamp, PartialStamp)
    assert stamp.held == frozenset({"state:IL"})
    assert stamp.epoch("state:VA") == lock.version and stamp.epoch("state:IL") == 1
    assert follower.status().pending == {_TABLE: 2}
    # the write commits: IL follows
    lock.commit("state:IL")
    lock.held = None
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()
    assert follower.stamp.epochs == lock.epochs


async def test_a_scope_a_commit_withheld_stays_settled_until_a_later_commit_moves_it() -> None:
    """ScopeEpochs' "A commit may withhold scopes": a load whose IL page failed commits VA and withholds IL, whose rows
    it left part way. No write is in progress, so the stamp is settled, but IL stays pending: the copy applies VA and
    keeps IL as it was, until a later commit moves IL. Before the fix a settled stamp held nothing, so IL's half-written
    rows were applied."""
    lock = _Scopes()
    l3, log, backend, follower = _scoped_world(lock)
    l3.put(_rows("VA", 5) + _rows("IL", 5))
    log.commit([("VA", i) for i in range(5)] + [("IL", i) for i in range(5)])
    await follower.warm()
    assert await follower.catch_up()
    settled_il = _of(backend, "IL")
    # the load wrote VA whole and IL part way, then committed withholding IL
    l3.put([("VA", 1, 77), ("IL", 0, 50)])
    log.commit([("VA", 1), ("IL", 0)])
    lock.commit("state:VA")
    lock.withheld = frozenset({"state:IL"})
    assert await follower.catch_up()
    assert _of(backend, "VA") == _l3_of(l3, "VA")
    assert _of(backend, "IL") == settled_il, "a withheld scope's half-written rows were applied"
    stamp = follower.stamp
    assert isinstance(stamp, PartialStamp) and stamp.held == frozenset({"state:IL"})
    # the next load finishes IL and commits it: IL follows
    l3.put([("IL", 9, 90)])
    log.commit([("IL", 9)])
    lock.commit("state:IL")
    lock.withheld = frozenset()
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()
    assert follower.stamp.epochs == lock.epochs


async def test_a_new_replica_is_ready_during_a_long_write_for_every_scope_it_can() -> None:
    lock = _Scopes()
    l3, log, backend, follower = _scoped_world(lock)
    l3.put(_rows("VA", 5) + _rows("IL", 5))
    lock.held = frozenset({"state:IL"})
    await follower.warm()
    assert await follower.catch_up()
    assert _of(backend, "VA") == _l3_of(l3, "VA")
    assert _of(backend, "IL") == []
    assert follower.stamp is not None and follower.stamp.held == frozenset({"state:IL"})
    lock.commit("state:IL")
    lock.held = None
    l3.rows_read = 0
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()
    assert l3.rows_read == 5  # IL alone, as a key range: never the table again


def _encode_scoped(stamp: Any) -> Any:
    """a scoped stamp as JSON-able data (a checkpoint keeps it): a partial stamp, or a settled record."""
    if isinstance(stamp, PartialStamp):
        return {"epochs": dict(stamp.epochs), "held": sorted(stamp.held), "version": stamp.version}
    return {"epochs": dict(stamp.epochs), "version": stamp.version}


def _decode_scoped(data: Any) -> Any:
    if "held" in data:
        return PartialStamp(epochs=data["epochs"], held=frozenset(data["held"]), version=data["version"])
    return SimpleNamespace(version=data["version"], epochs=data["epochs"], pending=frozenset())


async def test_a_checkpoint_written_beside_a_held_write_keeps_the_mark_before_its_keys_so_a_start_from_it_hears_them() -> (
    None
):
    """Flips if a checkpoint records a table at the registry's mark while keys of a scope a write holds are still
    pending: a replica started from it adopts that mark, never hears those keys, and serves the held scope's old rows
    after the write commits (reports' live replica tests, 2026-11-10: the second replica's copy never held the load's
    candidates)."""
    lock = _Scopes()
    l3, log, store, pointers = _L3(), _Log(), _Store(), _Pointers()
    l3.put(_rows("VA", 5) + _rows("IL", 5))
    log.commit([("VA", i) for i in range(5)] + [("IL", i) for i in range(5)])
    codec: dict[str, Any] = {"encode_stamp": _encode_scoped, "decode_stamp": _decode_scoped}
    _, writer, _ = _replica(l3, log, lock, store, pointers, "a", **codec)
    assert not await writer.start()
    assert await writer.catch_up()
    await writer.checkpointed()
    assert "votes-ckpt" in pointers.entries
    # a long write holds IL: its rows land part way, their keys heard and left pending
    lock.held = frozenset({"state:IL"})
    l3.put([("IL", i, 50) for i in range(5)])
    log.commit([("IL", i) for i in range(5)])
    # VA commits beside it, enough rows that a checkpoint is due at its apply
    l3.put([("VA", i, 70) for i in range(5)])
    log.commit([("VA", i) for i in range(5)])
    lock.commit("state:VA")
    assert await writer.catch_up()
    await writer.checkpointed()
    assert writer.status().pending == {_TABLE: 5}
    written = Checkpoint.decode(pointers.entries["votes-ckpt"][0])
    assert written.seq == 2, "a checkpoint is written beside the held write"
    assert _decode_scoped(written.stamp).held == frozenset({"state:IL"})
    # the write commits, and a replica starts from the checkpoint written beside it
    lock.commit("state:IL")
    lock.held = None
    second, follower, _ = _replica(l3, log, lock, store, pointers, "b", **codec)
    assert await follower.start()
    assert await follower.catch_up()
    assert _copy(second) == l3.held(), "the replica started from the checkpoint lost the held write's keys"


async def test_another_replicas_commit_of_a_held_write_wakes_the_follower_by_the_stamps_own_row_messages() -> None:
    """Flips if a follower holding keys a write held waits for another key, a catch-up or a poke once the write
    commits on another replica: the commit moves only the writer's epochs (every key was heard before it), so nothing
    else wakes it, and it serves the held scope's old rows (reports' live replica tests, 2026-11-10). The epochs
    table's own row messages, heard from every replica, ask the pass."""
    lock = _Scopes()
    l3, log = _L3(), _Log()
    backend = _backend()
    backend.initialize(_metadata())
    registry = CollectionRegistry()
    registry.follow_generation(_TABLE)
    published = PublishedCopy(backend)
    follower = CopyFollower(
        registry,
        [FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS, scope=("state", "state:"))],
        published,
        l3,
        settled=lock.settled,
        generation=log.current,
        changes=log,
        page_size=50,
        stamp_table="scope_epochs",
        drift_workers=_WORKERS,
        queries=published.queries,
    )
    follower.register()
    bus = FakeNatsClient()
    writer = CollectionRegistry()
    await registry.start_invalidation_listener(bus)
    running = asyncio.create_task(follower.run())
    try:
        l3.put(_rows("IL", 5))
        log.commit([("IL", i) for i in range(5)])
        await follower.warm()
        assert await follower.catch_up()
        # a long write on another replica holds IL: its rows land, their keys heard and left pending
        lock.held = frozenset({"state:IL"})
        l3.put([("IL", i, 50) for i in range(5)])
        log.commit([("IL", i) for i in range(5)])
        follower.poke()
        for _ in range(100):
            if follower.status().pending == {_TABLE: 5}:
                break
            await asyncio.sleep(0.01)
        assert follower.status().pending == {_TABLE: 5}
        # it commits: the epochs move, and the writer's registry says so for the epochs' rows alone
        lock.commit("state:IL")
        lock.held = None
        await writer.publish_invalidation(bus, "scope_epochs", ["state:IL"])
        for _ in range(200):
            if _copy(backend) == l3.held():
                break
            await asyncio.sleep(0.01)
        assert _copy(backend) == l3.held(), "the commit on another replica never woke the follower"
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        await registry.stop_invalidation_listener()


def _stamped(tables: list[FollowedTable], settled: Any, **stamp: Any) -> CopyFollower:
    """a follower of ``tables`` over a fresh copy, its writer's stamp read by ``settled``."""
    backend = _backend()
    backend.initialize(_metadata())
    registry = CollectionRegistry()
    registry.follow_generation(_TABLE)
    log = _Log()
    published = PublishedCopy(backend)
    return CopyFollower(
        registry,
        tables,
        published,
        _L3(),
        settled=settled,
        generation=log.current,
        changes=log,
        **stamp,
        drift_workers=_WORKERS,
        queries=published.queries,
    )


def test_a_follower_of_scoped_tables_names_the_stamp_table_it_hears_or_construction_raises() -> None:
    """Flips if a follower of a table a write may hold can be built hearing no stamp table: it would hold that write's
    keys until some other key arrives (reports' live replica tests, 2026-11-10). A ``ScopeEpochs.settled`` names its
    own table; any other ``settled`` must name it, and the error says what to pass."""
    scoped = [FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS, scope=("state", "state:"))]
    with pytest.raises(ValueError, match="stamp_table="):
        _stamped(scoped, _Scopes().settled)
    _stamped(scoped, _Scopes().settled, stamp_table="scope_epochs").register()
    epochs = ScopeEpochs(SimpleNamespace(table_name="enr_scope_epochs"))  # type: ignore[arg-type]
    backend = _backend()
    backend.initialize(_metadata())
    registry = CollectionRegistry()
    registry.follow_generation(_TABLE)
    log = _Log()
    published = PublishedCopy(backend)
    derived = CopyFollower(
        registry,
        scoped,
        published,
        _L3(),
        settled=epochs.settled,
        generation=log.current,
        changes=log,
        drift_workers=_WORKERS,
        queries=published.queries,
    )
    derived.register()
    assert registry.has_derived_caches("enr_scope_epochs"), "a ScopeEpochs.settled names its own table"


def test_a_follower_none_of_whose_tables_has_a_scope_builds_hearing_no_stamp_table() -> None:
    """Flips if a follower none of whose tables has a scope must name a stamp table: no write can hold its keys, so it
    has none to hear."""
    plain = [FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS)]
    _stamped(plain, _Seqlock().settled).register()


async def test_a_scope_the_write_takes_while_a_pass_reads_is_never_applied_from_that_pass() -> None:
    lock = _Scopes()
    registry = CollectionRegistry()
    l3, log, backend, follower = _scoped_world(lock, registry=registry)
    l3.put(_rows("VA", 5))
    log.commit([("VA", i) for i in range(5)])
    await follower.warm()
    assert await follower.catch_up()
    before, stamp = _copy(backend), follower.stamp
    l3.put([("VA", 1, 9)])
    log.commit([("VA", 1)])
    # the write takes VA between the pass's two reads of the record
    lock.on_read = (lock.reads + 2, lambda: setattr(lock, "held", frozenset({"state:VA"})))
    assert not await follower.catch_up()
    assert _copy(backend) == before and follower.stamp == stamp
    assert backend.staged_stages() == [] or follower.status().pending == {_TABLE: 1}
    lock.commit("state:VA")
    lock.held = None
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()


@pytest.mark.parametrize("how", ["commits", "takes"])
async def test_a_scope_the_stamp_moves_under_while_a_pass_reads_is_left_out_and_the_rest_applies(how: str) -> None:
    """Flips if one moved scope keeps a whole change from applying: a pass that read for minutes while an LSF load
    committed a state every ~13 s found some staged scope moved on every pass, kept every stage, and the pages froze
    for the night (reports' burst replay, 2026-11-23). Only the moved scope's keys leave the change, read again by the
    next pass; the label keeps that scope at the epoch the copy holds it at."""
    lock = _Scopes()
    l3, log, backend, follower = _scoped_world(lock)
    l3.put(_rows("VA", 3) + _rows("IL", 3))
    log.commit([("VA", i) for i in range(3)] + [("IL", i) for i in range(3)])
    await follower.warm()
    assert await follower.catch_up()
    settled_va, copy_va_epoch = _of(backend, "VA"), lock.epochs["state:VA"]
    l3.put([("VA", 1, 40), ("IL", 2, 50)])
    log.commit([("VA", 1), ("IL", 2)])
    lock.commit("state:VA", "state:IL")
    il_epoch = lock.epochs["state:IL"]

    def moves() -> None:
        # between the pass's two reads of the record: another load commits VA, or a write takes it
        if how == "commits":
            l3.put([("VA", 1, 41)])
            log.commit([("VA", 1)])
            lock.commit("state:VA")
        else:
            lock.held = frozenset({"state:VA"})

    lock.on_read = (lock.reads + 2, moves)
    assert await follower.catch_up()
    assert _of(backend, "IL") == _l3_of(l3, "IL")
    assert _of(backend, "VA") == settled_va
    stamp = follower.stamp
    assert isinstance(stamp, PartialStamp)
    assert stamp.epoch("state:IL") == il_epoch
    assert stamp.epoch("state:VA") == copy_va_epoch
    assert "state:VA" in stamp.held
    assert follower.status().pending == {_TABLE: 1}
    assert backend.staged_stages() == []
    lock.on_read = None
    if how == "takes":
        lock.commit("state:VA")
        lock.held = None
    l3.rows_read = 0
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()
    assert l3.rows_read == 1  # the key left out, alone
    assert follower.stamp.epochs == lock.epochs


async def test_unstaging_keys_takes_their_rows_and_keys_out_of_each_stage_and_keeps_the_rest() -> None:
    """Flips if a key left out of a change stays in its stages (it would land beside a stamp that never held its
    scope still), or if taking it out takes any other key's row with it."""
    backend = _backend()
    backend.initialize(_metadata())
    rows_stage, keys_stage = f"{STAGE_PREFIX}t__rows", f"{STAGE_PREFIX}t__keys"
    staged = pa.table(
        {
            "state": ["VA", "VA", "IL"],
            "geo_id": pa.array([1, 2, 3], pa.int64()),
            "votes": pa.array([5, 6, 7], pa.int64()),
        }
    )
    backend.stage_keys(keys_stage, _TABLE, _KEY, staged.select(list(_KEY)), rows_stage=rows_stage)
    backend.stage_rows(rows_stage, _TABLE, staged, _KEY)
    leaving = pa.table({"state": ["VA"], "geo_id": ["2"]})
    assert backend.unstage_keys([keys_stage, rows_stage, f"{STAGE_PREFIX}absent"], _TABLE, _KEY, leaving) == 1
    backend.apply_staged([StagedChange(table=_TABLE, key=_KEY, keys=keys_stage, rows=rows_stage)])
    assert _copy(backend) == [("IL", 3, 7), ("VA", 1, 5)]


async def test_an_apply_that_fails_keeps_its_stages_and_the_next_pass_reads_nothing(world: Any) -> None:
    l3, log, lock, backend, follower = world
    await _warmed(world, 10)
    l3.put([("VA", 1, 5)])
    log.commit([("VA", 1)])
    lock.version += 1
    real = backend.apply_staged
    calls = {"n": 0}

    def failing(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("the copy's apply failed")
        return real(*args, **kwargs)

    backend.apply_staged = failing
    with pytest.raises(OSError):
        await follower.catch_up()
    assert backend.staged_stages()
    l3.rows_read, l3.statements = 0, 0
    assert await follower.catch_up()
    assert (l3.rows_read, l3.statements) == (0, 0)
    assert _copy(backend) == l3.held()


async def test_drops_heard_before_the_read_are_one_whole_read_and_counted(world: Any) -> None:
    l3, log, lock, backend, follower = world
    await _warmed(world, 10)
    l3.put(_rows("OH", 3))
    log.commit(None)
    log.commit(None)
    lock.version += 1
    l3.statements = 0
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()
    assert l3.statements == 1  # one page of 13 rows: the two records were one read


async def test_a_partition_no_key_leads_is_an_older_writers_whole_record_never_a_scan_by_it() -> None:
    lock = _Scopes()
    l3, log, backend, follower = _scoped_world(lock)
    l3.put(_rows("VA", 4))
    log.commit([("VA", i) for i in range(4)])
    await follower.warm()
    assert await follower.catch_up()
    l3.put([("VA", 9, 1)])
    log.count += 1
    log.records[f"inc:{log.count}"] = EpochChange(
        table=_TABLE, token=f"inc:{log.count}", whole_table=True, partition={"votes": "1"}
    )
    lock.commit("state:VA")
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()


# --- a load's or a clear's key prefixes, an operator's reset, a table behind (0.95.0) -------------------------------


def _record_whole(log: _Log, **reach: Any) -> str:
    log.count += 1
    token = f"inc:{log.count}"
    log.records[token] = EpochChange(table=_TABLE, token=token, whole_table=True, **reach)
    return token


async def test_a_load_of_many_key_prefixes_reads_each_as_a_key_range_in_one_batched_read(world: Any) -> None:
    """an AP load covers many races of a state: its record names every office's key prefix, and the follower reads
    those prefixes' rows, all of them in one statement, never the table."""
    l3, log, lock, backend, follower = world
    await _warmed(world, 200)
    l3.put(_rows("OH", 7, votes=4) + _rows("TX", 9, votes=4))
    l3.delete([("VA", 0)])
    _record_whole(log, prefixes=[{"state": "OH"}, {"state": "TX"}])
    lock.version += 1
    l3.rows_read, l3.statements = 0, 0
    before = _whole_reads_counted(_TABLE, "legacy_whole_record")
    assert await follower.catch_up()
    assert [row for row in _copy(backend) if row[0] != "VA"] == [row for row in l3.held() if row[0] != "VA"]
    assert (l3.rows_read, l3.statements) == (16, 1)
    assert _whole_reads_counted(_TABLE, "legacy_whole_record") == before


async def test_two_column_key_prefixes_read_only_their_rows_and_replace_them(world: Any) -> None:
    l3, log, lock, backend, follower = world
    await _warmed(world, 50)
    l3.put([("VA", 3, 30)])
    l3.delete([("VA", 4)])
    _record_whole(log, prefixes=[{"state": "VA", "geo_id": "3"}, {"state": "VA", "geo_id": "4"}])
    lock.version += 1
    l3.rows_read, l3.statements = 0, 0
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()
    assert (l3.rows_read, l3.statements) == (1, 1)


async def test_a_reset_reads_its_table_again_as_that_tables_cold_start_counted_apart(world: Any) -> None:
    l3, log, lock, backend, follower = world
    await _warmed(world, 30)
    l3.delete([("VA", i) for i in range(30)])
    l3.put(_rows("OH", 4))
    _record_whole(log, reason="reset")
    lock.version += 1
    resets, legacy = _whole_reads_counted(_TABLE, "reset"), _whole_reads_counted(_TABLE, "legacy_whole_record")
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()
    assert _whole_reads_counted(_TABLE, "reset") == resets + 1
    assert _whole_reads_counted(_TABLE, "legacy_whole_record") == legacy


async def test_a_reset_is_read_during_a_long_write_as_a_first_start_is() -> None:
    lock = _Scopes()
    l3, log, backend, follower = _scoped_world(lock)
    l3.put(_rows("VA", 5) + _rows("IL", 5))
    log.commit([("VA", i) for i in range(5)] + [("IL", i) for i in range(5)])
    await follower.warm()
    assert await follower.catch_up()
    lock.held = frozenset({"state:IL"})
    l3.put([("VA", 2, 40), ("IL", 1, 40)])
    _record_whole(log, reason="reset")
    lock.commit("state:VA")
    assert await follower.catch_up()
    assert _of(backend, "VA") == _l3_of(l3, "VA")
    assert _of(backend, "IL") == []  # the held scope is read once the write commits
    lock.commit("state:IL")
    lock.held = None
    l3.rows_read = 0
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()
    assert l3.rows_read == 5


async def test_a_whole_read_kept_across_a_moved_stamp_still_reads_the_scopes_it_left_for_the_write() -> None:
    # 2026-11-22 replay: a reset's whole read beside a long write left the held states out for the write; the stamp
    # moved, the whole stage was kept, and the next pass dropped those states' partitions as covered by it. The copy
    # then held none of their rows (IA, SC, SD, TN, TX, each thousands of rows in L3) for as long as it ran.
    lock = _Scopes()
    l3, log, backend, follower = _scoped_world(lock)
    l3.put(_rows("VA", 5) + _rows("IL", 5))
    log.commit([("VA", i) for i in range(5)] + [("IL", i) for i in range(5)])
    await follower.warm()
    assert await follower.catch_up()
    lock.held = frozenset({"state:IL"})
    l3.put([("VA", 2, 40), ("IL", 1, 40)])
    _record_whole(log, reason="reset")
    lock.commit("state:VA")
    # a write moves VA while the whole read is staged: the pass keeps its stage for the next one
    lock.on_read = (lock.reads + 2, lambda: lock.commit("state:VA"))
    assert not await follower.catch_up()
    lock.on_read = None
    # the long write commits IL and ends
    lock.commit("state:IL")
    lock.held = None
    assert await follower.catch_up()
    assert _of(backend, "IL") == _l3_of(l3, "IL")
    assert _copy(backend) == l3.held()


def _office_world(lock: _Scopes) -> tuple[_L3, _Log, DuckDBBackend, CopyFollower]:
    """a table keyed by a column that leads without the scope (an office's key), its scope in the row only."""
    l3, log = _L3(), _Log()
    backend = _backend()
    backend.initialize(_metadata())
    registry = CollectionRegistry()
    registry.follow_generation(_TABLE)
    published = PublishedCopy(backend)
    follower = CopyFollower(
        registry,
        [FollowedTable(name=_TABLE, key=("geo_id", "state"), columns=_COLUMNS, scope=("state", "state:"))],
        published,
        l3,
        settled=lock.settled,
        generation=log.current,
        changes=log,
        page_size=50,
        stamp_table="scope_epochs",
        drift_workers=_WORKERS,
        queries=published.queries,
    )
    follower.register()
    return l3, log, backend, follower


async def test_a_prefix_whose_scope_is_in_its_rows_applies_beside_a_held_write_unless_a_row_is_held() -> None:
    lock = _Scopes()
    l3, log, backend, follower = _office_world(lock)
    l3.put([("VA", 3, 1), ("VA", 7, 1), ("IL", 7, 1)])
    log.commit([("3", "VA"), ("7", "VA"), ("7", "IL")])
    await follower.warm()
    assert await follower.catch_up()
    settled = _copy(backend)
    lock.held = frozenset({"state:IL"})
    # a load of office 3 (VA's alone) and office 7 (rows of VA and of the held IL) commits beside the write
    l3.put([("VA", 3, 9), ("VA", 7, 9), ("IL", 7, 9)])
    _record_whole(log, prefixes=[{"geo_id": "3"}])
    _record_whole(log, prefixes=[{"geo_id": "7"}])
    lock.commit("state:VA")
    assert await follower.catch_up()
    assert ("VA", 3, 9) in _copy(backend)
    assert [row for row in _copy(backend) if row[1] == 7] == [row for row in settled if row[1] == 7]
    lock.commit("state:IL")
    lock.held = None
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()


async def test_a_table_the_log_lost_is_caught_up_from_its_notes_never_read_whole(noted_world: Any) -> None:
    world, registry = noted_world
    l3, log, lock, backend, follower = world
    await _warmed(world, 100)
    # an epoch the log lost: its record is gone, so the registry is behind on the table
    when = datetime.now(UTC)
    l3.put([("VA", 5, 50), ("VA", 500, 1)])
    l3.delete([("VA", 6)])
    log.commit(None, keep=False)
    registry.set_noted_keys_reader(_Notes([(when, ("VA", "5")), (when, ("VA", "500")), (when, ("VA", "6"))]))
    lock.version += 1
    l3.rows_read = 0
    cold = _whole_reads_counted(_TABLE, "cold_start")
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()
    assert l3.rows_read == 2  # the noted keys' rows, never the table's
    assert follower.status().behind == {}
    assert _whole_reads_counted(_TABLE, "cold_start") == cold


class _WholeNotes(_Notes):
    """notes naming a whole-table write in the bracket: a file load, a large copy, a reset."""

    async def noted_keys(self, table: str, **kwargs: Any) -> NotedKeysPage | str:
        page = await super().noted_keys(table, **kwargs)
        assert isinstance(page, NotedKeysPage)
        return NotedKeysPage(keys=page.keys, whole_table=True, next=None)


async def test_notes_naming_a_whole_write_leave_the_table_behind_never_read_whole(noted_world: Any) -> None:
    world, registry = noted_world
    l3, log, lock, backend, follower = world
    await _warmed(world, 20)
    before = _copy(backend)
    l3.put(_rows("OH", 3))
    log.commit(None, keep=False)
    registry.set_noted_keys_reader(_WholeNotes([(datetime.now(UTC), ("OH", "0"))]))
    lock.version += 1
    l3.rows_read = 0
    assert not await follower.catch_up()
    assert follower.status().behind == {_TABLE: "notes_whole"}
    assert _copy(backend) == before
    assert l3.rows_read == 0


# --- a table the follower's owner derives, held in its checkpoint ---------------------------------------------------

_DERIVED = "totals"
_DERIVED_TABLE = FollowedTable(
    name=_DERIVED,
    key=("state",),
    columns=(Column(name="state", column_type=STRING_TYPE), Column(name="total", column_type=BIGINT_TYPE)),
    scope=("state", "state:"),
)


def _derived_replica(
    l3: _L3, log: _Log, lock: _Seqlock, store: _Store, pointers: _Pointers, owner: str, *, holds_derived: bool = True
) -> tuple[DuckDBBackend, CopyFollower]:
    """a replica whose copy holds a table its owner derives (a settle's), which its checkpoint holds too."""
    metadata = _metadata()
    Table(
        _DERIVED,
        metadata,
        SaColumn("state", String(8), primary_key=True),
        SaColumn("total", BigInteger),
        info={L1_EVICTION_INFO: TEST_L1_TABLE},
    )
    backend = _backend()
    backend.initialize(metadata)
    followed = [FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS, scope=("state", "state:"))]
    checkpoints = CopyCheckpoints(
        name="votes",
        tables=[*followed, *([_DERIVED_TABLE] if holds_derived else [])],
        backend=backend,
        store=store,
        pointers=pointers,
        retire=store.retire,
        owner=owner,
        part_rows=40,
    )
    registry = CollectionRegistry()
    registry.follow_generation(_TABLE)  # the derived table is never followed: its owner derives it
    published = PublishedCopy(backend)
    follower = CopyFollower(
        registry,
        followed,
        published,
        l3,
        settled=lock.settled,
        generation=log.current,
        changes=log,
        page_size=50,
        checkpoints=checkpoints,
        stamp_table="scope_epochs",
        drift_workers=_WORKERS,
        queries=published.queries,
    )
    return backend, follower


def _derive(backend: DuckDBBackend, total: int) -> None:
    """the owner's settle: the derived table's rows, applied to the copy."""
    stage = "__stage__totals__rows"
    backend.stage_rows(stage, _DERIVED, pa.table({"state": ["VA"], "total": pa.array([total], pa.int64())}), ("state",))
    backend.apply_staged(
        [StagedChange(table=_DERIVED, key=("state",), rows=stage, whole=True, whole_kind="cold_start")]
    )


def _totals(backend: DuckDBBackend) -> list[tuple[Any, ...]]:
    with backend.read_snapshot() as cursor:
        return sorted(cursor.execute(f"SELECT state, total FROM {_DERIVED}").fetchall())


async def test_a_derived_table_in_the_checkpoint_loads_with_the_followed_ones() -> None:
    l3, log, lock, store, pointers = _L3(), _Log(), _Seqlock(), _Store(), _Pointers()
    l3.put(_rows("VA", 60))
    log.commit([("VA", i) for i in range(60)])
    first, writer = _derived_replica(l3, log, lock, store, pointers, "a")
    assert not await writer.start()
    _derive(first, 60)
    assert await writer.catch_up()
    await writer.checkpointed()
    second, follower = _derived_replica(l3, log, lock, store, pointers, "b")
    l3.rows_read = 0
    assert await follower.start()  # wholly from the checkpoint: the owner need not derive again
    assert await follower.catch_up()
    assert _copy(second) == l3.held() and l3.rows_read == 0
    assert _totals(second) == [("VA", 60)]


async def test_a_derived_table_the_checkpoint_cannot_give_is_left_to_its_owner_never_warmed() -> None:
    l3, log, lock, store, pointers = _L3(), _Log(), _Seqlock(), _Store(), _Pointers()
    l3.put(_rows("VA", 60))
    log.commit([("VA", i) for i in range(60)])
    # an older writer's checkpoint holds no derived table
    _, writer = _derived_replica(l3, log, lock, store, pointers, "a", holds_derived=False)
    assert not await writer.start()
    assert await writer.catch_up()
    await writer.checkpointed()
    second, follower = _derived_replica(l3, log, lock, store, pointers, "b")
    l3.rows_read = 0
    # the followed table loads; the derived one is the owner's to derive again, so the start is not whole
    assert not await follower.start()
    assert await follower.catch_up()
    assert _copy(second) == l3.held() and l3.rows_read == 0
    assert _totals(second) == []


async def test_a_key_whose_row_is_of_a_held_scope_is_read_once_across_the_write() -> None:
    """a table keyed without its scope (a candidate's key, its state in the row): a key the writer committed while
    its scope's write is still held waits on the copy's own row, never read from L3 until the write commits."""
    lock = _Scopes()
    l3, log = _L3(), _Log()
    backend = _backend()
    backend.initialize(_metadata())
    registry = CollectionRegistry()
    registry.follow_generation(_TABLE)
    published = PublishedCopy(backend)
    follower = CopyFollower(
        registry,
        [FollowedTable(name=_TABLE, key=("geo_id",), columns=_COLUMNS, scope=("state", "state:"))],
        published,
        l3,
        settled=lock.settled,
        generation=log.current,
        changes=log,
        page_size=50,
        stamp_table="scope_epochs",
        drift_workers=_WORKERS,
        queries=published.queries,
    )
    follower.register()
    l3.put([("VA", 1, 1), ("VA", 2, 1), ("IL", 11, 1), ("IL", 12, 1)])
    await follower.warm()
    assert await follower.catch_up()

    def commit(*geo_ids: int) -> None:
        log.count += 1
        token = f"inc:{log.count}"
        log.records[token] = EpochChange(table=_TABLE, token=token, rows=[ChangedRow(ids=[str(g)]) for g in geo_ids])

    # the writer's data commits first, its scope marked pending in that transaction; the scope commits after
    lock.held = frozenset({"state:IL"})
    l3.put([("IL", 11, 9)])
    commit(11)
    l3.rows_read = 0
    assert await follower.catch_up()
    assert l3.rows_read == 0  # the copy's own row says IL, held: no L3 read while the write holds it
    lock.commit("state:IL")
    lock.held = None
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()
    assert l3.rows_read == 1  # the changed key, read exactly once


# --- the one whole read: only a cold start, a reset, a schema change, or an older writer mid-roll --------------------


def _whole_reads_now() -> dict[str, float]:
    """``copy_follower_whole_reads_total`` of the table, by why: what ``CopyFollower._stage_whole`` counts each time
    it is reached."""
    return {why: _whole_reads_counted(_TABLE, why) for why in WHOLE_READ_WHYS}


def _risen(before: Mapping[str, float]) -> dict[str, float]:
    return {why: count - before[why] for why, count in _whole_reads_now().items() if count != before[why]}


async def test_stage_whole_is_reached_only_at_a_cold_start_a_reset_a_schema_change_or_a_legacy_whole_record() -> None:
    """the proving test of the whole reads declared in CopyFollower._stage_cold_start, CopyFollower._stage_schema_change
    and CopyFollower._stage_legacy_whole_record, and of CopyFollower._stage's declared call into each: every
    steady-state path (keys, a load's partition, a load's key prefixes, a stamp that moves, a failed pass, a start from
    a checkpoint and its checkpoints) reads nothing whole; only a cold start and an operator's reset
    (_stage_cold_start), a schema change (_stage_schema_change) and a whole record with neither prefixes nor a reason
    (a writer older than 0.95.0 mid-roll: _stage_legacy_whole_record) reach _stage_whole, each counted once by its why
    (copy_follower_whole_reads_total, which _stage_whole alone counts, by the why each of the three passes it)."""
    l3, log, lock, store, pointers = _L3(), _Log(), _Seqlock(), _Store(), _Pointers()
    l3.put(_rows("VA", 120))
    log.commit([("VA", i) for i in range(120)])
    _, first, _ = _replica(l3, log, lock, store, pointers, "a")
    before = _whole_reads_now()
    assert not await first.start()
    assert await first.catch_up()
    await first.checkpointed()
    assert _risen(before) == {"cold_start": 1}
    # a second replica starts from the checkpoint: no whole read
    backend, follower, _ = _replica(l3, log, lock, store, pointers, "b")
    counted = _whole_reads_now()
    l3.rows_read = 0
    assert await follower.start()
    assert await follower.catch_up()
    # keys: an insert, an update and a delete
    l3.put([("VA", 7, 70), ("VA", 500, 1)])
    l3.delete([("VA", 3)])
    log.commit([("VA", 7), ("VA", 500), ("VA", 3)])
    lock.version += 1
    assert await follower.catch_up()
    # a load's partition, and a load's key prefixes
    l3.put(_rows("OH", 9, votes=4))
    _record_whole(log, partition={"state": "OH"})
    lock.version += 1
    assert await follower.catch_up()
    l3.put(_rows("TX", 5, votes=2))
    _record_whole(log, prefixes=[{"state": "TX"}, {"state": "OH"}])
    lock.version += 1
    assert await follower.catch_up()
    # a stamp that moves while a pass reads, then the pass that tops it up
    l3.put([("VA", 8, 80)])
    log.commit([("VA", 8)])
    lock.version += 1
    lock.move_on_read = lock.reads + 2
    assert not await follower.catch_up()
    assert await follower.catch_up()
    # a pass that fails part way, then the one that applies the same keys
    l3.put([("VA", 9, 90)])
    log.commit([("VA", 9)])
    lock.version += 1
    l3.fail_next = True
    with pytest.raises(ConnectionError):
        await follower.catch_up()
    assert await follower.catch_up()
    await follower.checkpointed()
    assert _copy(backend) == l3.held()
    assert _risen(counted) == {}
    assert l3.rows_read < len(l3.held())  # never the table
    # the three that do read whole, each once and counted by its why
    _record_whole(log, reason="reset")
    lock.version += 1
    assert await follower.catch_up()
    follower.schema_changed(_TABLE)
    assert await follower.catch_up()
    _record_whole(log)  # a writer older than 0.95.0: neither prefixes nor a reason
    lock.version += 1
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()
    assert _risen(counted) == {"reset": 1, "schema_change": 1, "legacy_whole_record": 1}


def _recording_applies(backend: DuckDBBackend) -> list[str | None]:
    """the whole kind of every whole change ``backend.apply_staged`` is handed, in order (None for one with none)."""
    kinds: list[str | None] = []
    real = backend.apply_staged

    def apply_staged(changes: Sequence[StagedChange], *, mark: int | None = None) -> int:
        kinds.extend(change.whole_kind for change in changes if change.whole)
        return real(changes, mark=mark)

    backend.apply_staged = apply_staged  # type: ignore[method-assign]
    return kinds


async def test_a_whole_change_carries_the_declared_kind_that_staged_it() -> None:
    """the proving test of the whole empties declared in DuckDBBackend.apply_staged: a change is whole only with the
    declared whole read that staged it (cold_start, schema_change, rolling_upgrade), and every path that stages one
    carries that kind to the apply: a first start and a start from a checkpoint (cold_start), an operator's reset
    (cold_start), a schema change (schema_change) and an older writer's whole record (rolling_upgrade); keys,
    partitions and checkpoints apply no whole change. A whole change naming no kind, or another, is refused, and so
    is a kind on a change that is not whole."""
    with pytest.raises(ValueError, match="names the declared whole read"):
        StagedChange(table=_TABLE, key=_KEY, rows="__stage__x", whole=True)
    with pytest.raises(ValueError, match="names the declared whole read"):
        StagedChange(table=_TABLE, key=_KEY, rows="__stage__x", whole=True, whole_kind="restore")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="names no whole kind"):
        StagedChange(table=_TABLE, key=_KEY, rows="__stage__x", whole_kind="cold_start")
    l3, log, lock, store, pointers = _L3(), _Log(), _Seqlock(), _Store(), _Pointers()
    l3.put(_rows("VA", 60))
    log.commit([("VA", i) for i in range(60)])
    first_backend, first, _ = _replica(l3, log, lock, store, pointers, "a")
    first_kinds = _recording_applies(first_backend)
    assert not await first.start()
    assert await first.catch_up()
    await first.checkpointed()
    assert first_kinds == ["cold_start"]
    backend, follower, _ = _replica(l3, log, lock, store, pointers, "b")
    kinds = _recording_applies(backend)
    assert await follower.start()
    assert await follower.catch_up()
    assert kinds == ["cold_start"]  # the start from the checkpoint
    l3.put([("VA", 7, 70)])
    log.commit([("VA", 7)])
    lock.version += 1
    assert await follower.catch_up()
    l3.put(_rows("OH", 9, votes=4))
    _record_whole(log, partition={"state": "OH"})
    lock.version += 1
    assert await follower.catch_up()
    await follower.checkpointed()
    assert kinds == ["cold_start"]  # keys, a partition and a checkpoint apply nothing whole
    _record_whole(log, reason="reset")
    lock.version += 1
    assert await follower.catch_up()
    follower.schema_changed(_TABLE)
    assert await follower.catch_up()
    _record_whole(log)  # a writer older than 0.95.0
    lock.version += 1
    assert await follower.catch_up()
    assert kinds == ["cold_start", "cold_start", "schema_change", "rolling_upgrade"]
    assert _copy(backend) == l3.held()


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".")[:3])


def test_the_older_writers_whole_record_read_is_gone_by_0_96_0() -> None:
    """legacy_whole_record lives only for the 0.95.0 roll (Pace, 2026-11-09): from 3tears 0.96.0 it fails until the
    code is deleted: CopyFollower._on_dropped's re-read and the non-key branch of CopyFollower._on_partition (a whole
    record then is the registry's to refuse), WHOLE_READ_WHYS["legacy_whole_record"], this test's twin assertions in
    test_stage_whole_is_reached_only_at_..., and the rolling_upgrade declaration (until="0.96.0") in
    CopyFollower._stage and CopyFollower._stage_legacy_whole_record."""
    assert core.__version__ != "unknown", "3tears is not installed: its version cannot be judged"
    if _version_tuple(core.__version__) >= (0, 96, 0):
        assert "legacy_whole_record" not in WHOLE_READ_WHYS, (
            f"3tears {core.__version__} still re-reads a table whole for a writer older than 0.95.0: delete "
            "CopyFollower._on_dropped's legacy re-read, the non-key branch of CopyFollower._on_partition and "
            'WHOLE_READ_WHYS["legacy_whole_record"]'
        )


async def test_a_checkpoint_stage_is_reached_only_at_a_start() -> None:
    """the proving test of CopyCheckpoints.stage's ledger line: its whole load of each table from L2 runs once, at a
    follower's start; no catch-up, checkpoint write, reset or schema change reaches it again."""
    l3, log, lock, store, pointers = _L3(), _Log(), _Seqlock(), _Store(), _Pointers()
    l3.put(_rows("VA", 60))
    log.commit([("VA", i) for i in range(60)])
    _, writer, _ = _replica(l3, log, lock, store, pointers, "a")
    assert not await writer.start()
    assert await writer.catch_up()
    await writer.checkpointed()
    backend, follower, checkpoints = _replica(l3, log, lock, store, pointers, "b")
    stages: list[int] = []
    real = checkpoints.stage

    async def stage() -> Any:
        stages.append(1)
        return await real()

    checkpoints.stage = stage
    assert await follower.start()
    assert stages == [1]
    for version in range(1, 40):
        l3.put([("VA", version, version)])
        log.commit([("VA", version)])
        lock.version += 1
        assert await follower.catch_up()
    await follower.checkpointed()
    _record_whole(log, reason="reset")
    lock.version += 1
    assert await follower.catch_up()
    follower.schema_changed(_TABLE)
    assert await follower.catch_up()
    await follower.checkpointed()
    assert _copy(backend) == l3.held()
    assert stages == [1]


@pytest.mark.parametrize("field", ["order", "partition_order"])
def test_an_order_that_is_not_unique_is_refused_at_construction(field: str) -> None:
    """Flips if a FollowedTable takes a page order missing a key column: a keyset read by it skips every row sharing
    the cursor's values at each page boundary (reports' state-index order, 2026-11-13), so the copy holds fewer rows
    than L3 under L3's epochs."""
    with pytest.raises(ValueError, match=r"votes: .*not unique.*missing \['geo_id'\]"):
        FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS, **{field: ("state",)})  # type: ignore[arg-type]
    FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS, **{field: ("state", "votes", "geo_id")})  # type: ignore[arg-type]


async def test_a_warm_paged_by_its_order_over_one_shared_lead_value_reads_every_row() -> None:
    """Flips if the first warm, paged (50 a page) by an explicit order over 120 rows that all share its lead value,
    holds fewer rows than L3."""
    l3, log, lock = _L3(), _Log(), _Seqlock()
    backend = _backend()
    backend.initialize(_metadata())
    registry = CollectionRegistry()
    registry.follow_generation(_TABLE)
    published = PublishedCopy(backend)
    follower = CopyFollower(
        registry,
        [FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS, order=("state", "geo_id"))],
        published,
        l3,
        settled=lock.settled,
        generation=log.current,
        changes=log,
        page_size=50,
        drift_workers=_WORKERS,
        queries=published.queries,
    )
    follower.register()
    await _warmed((l3, log, lock, backend, follower), 120)
    assert len(_copy(backend)) == 120


def _dies_applying(backend: DuckDBBackend, deaths: int) -> dict[str, int]:
    """the apply worker dies applying the next ``deaths`` changes: its stages die with it (a worker-process backend's
    ``LookupError``), nothing is applied."""
    real = backend.apply_staged
    calls = {"n": 0}

    def dying(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] <= deaths:
            held = backend.staged_stages()
            if held:
                backend.drop_stage(*held)
            raise LookupError("the apply worker died applying the change; the change is read again")
        return real(*args, **kwargs)

    backend.apply_staged = dying  # type: ignore[method-assign]
    return calls


def _failures_counted(what: str) -> float:
    return REGISTRY.get_sample_value("copy_follower_pass_failures_total", {"what": what}) or 0.0


def _behind_since() -> float:
    return REGISTRY.get_sample_value("copy_follower_behind_since_timestamp_seconds") or 0.0


async def test_an_apply_whose_worker_died_reads_the_change_again_and_the_next_pass_applies_it(world: Any) -> None:
    l3, log, lock, backend, follower = world
    await _warmed(world, 10)
    before = _copy(backend)
    l3.put([("VA", 1, 5)])
    l3.delete([("VA", 2)])
    log.commit([("VA", 1), ("VA", 2)])
    lock.version += 1
    _dies_applying(backend, 1)
    lost = _failures_counted("stages_lost")
    with pytest.raises(LookupError):
        await follower.catch_up()
    # the copy keeps serving its last change; the change's keys are pending again, nothing is read whole
    assert _copy(backend) == before
    assert follower.status().pending == {_TABLE: 2}
    assert follower.status().failures == 1
    assert follower.status().behind_since is not None
    assert _failures_counted("stages_lost") == lost + 1
    assert _behind_since() > 0
    whole = _whole_reads_now()
    l3.rows_read = 0
    assert await follower.catch_up()
    # the change is read again -- its own keys, never the table
    assert l3.rows_read == 1
    assert _risen(whole) == {}
    assert _copy(backend) == l3.held()
    assert follower.status().failures == 0
    assert follower.status().behind_since is None
    assert follower.status().phase is FollowerPhase.FOLLOWING
    assert _behind_since() == 0


async def test_stages_lost_while_staging_are_forgotten_and_the_next_pass_reads_the_change_again(world: Any) -> None:
    l3, log, lock, backend, follower = world
    await _warmed(world, 10)
    l3.put([("VA", 3, 7)])
    log.commit([("VA", 3)])
    lock.version += 1
    real = backend.stage_rows
    calls = {"n": 0}

    def dying(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise LookupError("the apply worker holding this change's stages was replaced; the change is read again")
        return real(*args, **kwargs)

    backend.stage_rows = dying
    with pytest.raises(LookupError):
        await follower.catch_up()
    assert follower.status().pending == {_TABLE: 1}
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()


async def test_repeated_deaths_reach_the_failing_state_serving_the_last_change_and_a_good_change_then_applies(
    world: Any,
) -> None:
    l3, log, lock, backend, follower = world
    await _warmed(world, 10)
    before = _copy(backend)
    stamp = follower.stamp
    l3.put([("VA", 4, 9)])
    log.commit([("VA", 4)])
    lock.version += 1
    _dies_applying(backend, FAILING_AFTER)
    for failed in range(1, FAILING_AFTER + 1):
        with pytest.raises(LookupError):
            await follower.catch_up()
        status = follower.status()
        assert status.failures == failed
        assert status.phase is (FollowerPhase.FAILING if failed >= FAILING_AFTER else FollowerPhase.WAITING)
        # stale is fine: every reader still reads the last change applied, labelled with its stamp
        assert _copy(backend) == before
        async with follower.current() as (_state, read_stamp, _marks):
            assert read_stamp == stamp
    assert _behind_since() > 0
    l3.put([("VA", 5, 9)])
    log.commit([("VA", 5)])
    lock.version += 1
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()
    assert follower.status().phase is FollowerPhase.FOLLOWING
    assert follower.status().failures == 0
    assert _behind_since() == 0


async def test_the_running_follower_tries_a_failed_pass_again_after_a_wait_with_nothing_else_heard() -> None:
    (l3, log, lock, backend, follower), _registry = _world(retry_waits=(0.01, 0.05))
    await _warmed((l3, log, lock, backend, follower), 10)
    applied = follower.status().applied
    l3.put([("VA", 6, 3)])
    log.commit([("VA", 6)])
    lock.version += 1
    calls = _dies_applying(backend, 3)
    running = asyncio.create_task(follower.run())
    try:
        follower.poke()
        async with asyncio.timeout(10):
            while follower.status().applied == applied:
                await asyncio.sleep(0.01)
    finally:
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    # three deaths, each retried after its wait with no poke; the fourth try applied the change
    assert calls["n"] == 4
    assert _copy(backend) == l3.held()
    assert follower.status().failures == 0


# --- a pass that only relabels the copy (E1, 2026-11-15) -------------------------------------------------------------


def _applies_counted(backend: DuckDBBackend) -> list[int | None]:
    """each mark ``backend.apply_staged`` is handed, in order: one per change the worker is asked to apply."""
    marks: list[int | None] = []
    real = backend.apply_staged

    def apply_staged(changes: Sequence[StagedChange], *, mark: int | None = None) -> int:
        marks.append(mark)
        return real(changes, mark=mark)

    backend.apply_staged = apply_staged  # type: ignore[method-assign]
    return marks


async def test_a_pass_that_only_relabels_the_copy_asks_the_backend_for_no_apply(world: Any) -> None:
    """Flips if a pass with nothing heard and nothing staged sends the backend an empty change: on a pod's backend
    that is a worker round trip, a settle and a publish of unchanged rows for a new label alone (F1, 2026-11-15)."""
    l3, log, lock, backend, follower = world
    await _warmed(world, 10)
    applies = _applies_counted(backend)
    heard: list[Any] = []
    follower.on_change(heard.append)
    applied = follower.status().applied
    # the writer's stamp moves with no key of a followed table heard (a commit of nothing this copy holds)
    lock.version += 1
    assert await follower.catch_up()
    assert applies == []
    assert follower.stamp == lock.version
    async with follower.current() as (_state, stamp, marks):
        # a reader of the copy as it is reads it at the new label, with the marks it holds now
        assert stamp == lock.version
        assert marks == follower.marks()
    assert heard == [lock.version]
    assert follower.status().applied == applied
    assert follower.status().relabelled == 1
    # a change after it applies as ever, at its own mark, its readers labelled with its stamp
    l3.put([("VA", 1, 5)])
    log.commit([("VA", 1)])
    lock.version += 1
    assert await follower.catch_up()
    assert len(applies) == 1
    assert _copy(backend) == l3.held()
    async with follower.current() as (_state, stamp, _marks):
        assert stamp == lock.version


async def test_a_relabel_beside_a_write_that_holds_nothing_asks_for_no_apply_either_way() -> None:
    """Flips if the passes around a write in progress that holds no scope the copy is behind on (a PartialStamp in,
    the settled record out) each send the backend an empty change, as reports' pod did on every load (F1)."""
    lock = _Scopes()
    l3, log, backend, follower = _scoped_world(lock)
    l3.put(_rows("VA", 5) + _rows("IL", 5))
    log.commit([("VA", i) for i in range(5)] + [("IL", i) for i in range(5)])
    await follower.warm()
    assert await follower.catch_up()
    applies = _applies_counted(backend)
    before = _copy(backend)
    lock.held = frozenset()
    assert await follower.catch_up()
    assert isinstance(follower.stamp, PartialStamp)
    async with follower.current() as (_state, stamp, _marks):
        assert stamp == follower.stamp
    lock.held = None
    assert await follower.catch_up()
    assert not isinstance(follower.stamp, PartialStamp)
    async with follower.current() as (_state, stamp, _marks):
        assert stamp == follower.stamp
    assert applies == []
    assert _copy(backend) == before
    assert follower.status().relabelled == 2


async def test_a_relabel_never_hides_a_failed_apply_the_next_pass_tries_the_apply_again(world: Any) -> None:
    """Flips if a pass after a failed apply (its stages kept, the copy at the change before) relabels the copy instead
    of applying: readers would read the old rows under the new stamp."""
    l3, log, lock, backend, follower = world
    await _warmed(world, 10)
    before = follower.stamp
    l3.put([("VA", 1, 5)])
    log.commit([("VA", 1)])
    lock.version += 1
    real = backend.apply_staged

    def failing(changes: Sequence[StagedChange], *, mark: int | None = None) -> int:
        raise RuntimeError("the apply failed")

    backend.apply_staged = failing
    with pytest.raises(RuntimeError):
        await follower.catch_up()
    assert follower.stamp == before
    backend.apply_staged = real
    applies = _applies_counted(backend)
    assert await follower.catch_up()
    assert len(applies) == 1
    assert _copy(backend) == l3.held()
    async with follower.current() as (_state, stamp, _marks):
        assert stamp == lock.version
    assert follower.status().relabelled == 0


async def test_a_run_of_row_messages_is_heard_as_one_arrow_batch_a_table_in_order() -> None:
    """Flips if the owner's drain of row messages loses one, keeps one of a table it does not follow, or stops past a
    message that is not a row message (each table's run is built by Arrow off the messages, never a Python list a key:
    the 2026-11-24 ruling). A message naming the wrong number of key values fails the pass; every message before it
    is heard and every one after it is applied by the next pass."""
    (l3, _log, _lock, backend, follower), registry = _world()
    await _warmed((l3, _log, _lock, backend, follower), 2)
    l3.put([("VA", 1, 11), ("IL", 4, 44)])
    for message in (
        CacheInvalidationMessage(table=_TABLE, ids=["VA", "1"]),
        CacheInvalidationMessage(table="not_followed", ids=["x"]),
        CacheInvalidationMessage(table=_TABLE, ids=["IL", "4"]),
    ):
        registry.tell_derived_caches(message)
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()
    l3.put([("OH", 2, 22), ("TX", 7, 77)])
    for message in (
        CacheInvalidationMessage(table=_TABLE, ids=["OH", "2"]),
        CacheInvalidationMessage(table=_TABLE, ids=["VA"]),
        CacheInvalidationMessage(table=_TABLE, ids=["TX", "7"]),
    ):
        registry.tell_derived_caches(message)
    with pytest.raises(ValueError, match="names 1 key values"):
        await follower.catch_up()
    assert await follower.catch_up()
    assert _copy(backend) == l3.held(), "a row message before or after the wrong one was lost"


async def test_a_key_heard_again_at_the_epoch_its_kept_stage_holds_is_not_read_from_l3_again() -> None:
    """Flips if a key heard a second time at the epoch an earlier pass already staged it at (its row broadcast and
    its epoch's record, one landing after a pass the stamp moved under, which keeps its stages) is read from L3 again:
    each record's keys are read once (the live refresh test read one changed row twice, 2 runs in 6, 2026-11-24). A
    key heard at a newer epoch is still read again, and the copy ends equal to L3."""
    (l3, log, lock, backend, follower), registry = _world()
    await _warmed((l3, log, lock, backend, follower), 2)
    l3.put([("VA", 0, 500)])
    token = log.commit([("VA", 0)])
    heard = CacheInvalidationMessage(table=_TABLE, ids=["VA", "0"], generation=token)
    registry.tell_derived_caches(heard)
    lock.move_on_read = lock.reads + 2  # a write commits between this pass's two reads of the stamp
    assert not await follower.catch_up(), "the stamp moved: the pass keeps its stage for the next"
    read = l3.rows_read
    registry.tell_derived_caches(heard)  # the same change heard again, at the same epoch
    assert await follower.catch_up()
    assert l3.rows_read == read, "the key staged at this epoch was read from L3 again"
    assert _copy(backend) == l3.held()
    l3.put([("VA", 0, 600)])
    newer = log.commit([("VA", 0)])
    registry.tell_derived_caches(CacheInvalidationMessage(table=_TABLE, ids=["VA", "0"], generation=newer))
    assert await follower.catch_up()
    assert l3.rows_read > read, "a key heard at a newer epoch was not read again"
    assert _copy(backend) == l3.held()


# --- a pass's CPU leaves the serving process (copy_key_work) -------------------------------------------------------


#: the most keys a pass takes in this process, and the most row messages a turn of the loop drains: the follower's own
#: measured bounds (copy_follower: the take's on the loop, a drain's run), restated so a test says what it holds them to
_TAKEN_HERE = 500
_A_TURN = 2_000


@pytest.fixture
def jobs() -> Iterator[list[str]]:
    """the jobs the tests' worker process was sent, from the test's start."""
    _WORKERS.jobs.clear()
    yield _WORKERS.jobs
    _WORKERS.jobs.clear()


def _here(monkeypatch: pytest.MonkeyPatch, name: str, rows: Callable[..., int]) -> list[int]:
    """each call of one of ``copy_key_work``'s cores the follower makes in this process (any thread): the rows it was
    given. A call in the worker process is not seen. Set as the follower names it, so a follower that has no such core
    (an older one) is seen to make none."""
    seen: list[int] = []
    real = getattr(copy_follower, name, None)

    def watched(*args: Any) -> Any:
        seen.append(rows(*args))
        assert real is not None
        return real(*args)

    monkeypatch.setattr(copy_follower, name, watched, raising=False)
    return seen


async def test_a_large_pass_takes_its_keys_in_the_worker_process(
    monkeypatch: pytest.MonkeyPatch, jobs: list[str]
) -> None:
    """Flips if a pass's take of its keys heard (each key at each epoch once, those of a held scope left, the keys
    sorted and placed) runs in the serving process past its bound: on a thread it took 84 ms of CPU at 100,000 keys and
    1.06 s at 1,000,000 (2026-11-25), cores the pod's page serving needs. A small pass takes its keys here."""
    (l3, log, lock, backend, follower), _registry = _world(workers=_WORKERS)
    await _warmed((l3, log, lock, backend, follower), 2)
    taken = _here(monkeypatch, "taken_of", lambda key, scope, files, heard, *rest: heard.num_rows)
    count = _TAKEN_HERE * 6
    await asyncio.to_thread(l3.put, _rows("IL", count, votes=3))
    await asyncio.to_thread(log.commit, [("IL", i) for i in range(count)])
    lock.version += 1
    assert await follower.catch_up()
    assert "taken_job" in jobs, f"the pass's {count} keys were not taken in the worker process: {jobs}"
    assert not [rows for rows in taken if rows > _TAKEN_HERE], f"taken here: {taken}"
    assert _copy(backend) == l3.held()
    jobs.clear()
    l3.put([("IL", 0, 9)])
    log.commit([("IL", 0)])
    lock.version += 1
    assert await follower.catch_up()
    assert "taken_job" not in jobs, "a pass of one key went to the worker process: its round trip costs more"
    assert _copy(backend) == l3.held()


async def test_a_whole_read_and_a_partition_past_the_bound_are_judged_in_the_worker_process(
    monkeypatch: pytest.MonkeyPatch, jobs: list[str]
) -> None:
    """Flips if a page of a whole read, or of a partition, past the bound has its scopes found and judged in the
    serving process: on the loop a 200,000-row page took 6.5 ms, 19.7 ms with a scope held (2026-11-25). A first start
    beside a write holding IL leaves IL's rows out (their flags the worker's), then reads IL once the write commits."""
    lock = _Scopes()
    page = INLINE_KEYS * 2
    l3, log, backend, follower = _scoped_world(lock, page_size=page * 2, workers=_WORKERS)
    await asyncio.to_thread(l3.put, _rows("VA", page) + _rows("IL", page))
    await asyncio.to_thread(log.commit, [("VA", i) for i in range(page)] + [("IL", i) for i in range(page)])
    whole = _here(monkeypatch, "whole_page_of", lambda scope, rows, held: rows.num_rows)
    parts = _here(monkeypatch, "partition_page_of", lambda key, instants, scope, rows, held: rows.num_rows)
    lock.held = frozenset({"state:IL"})
    await follower.warm()
    assert await follower.catch_up()
    assert _of(backend, "VA") == _l3_of(l3, "VA") and _of(backend, "IL") == []
    lock.held = None
    lock.commit("state:IL")
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()
    assert {"whole_page_job", "partition_page_job"} <= set(jobs), f"judged in the serving process: {jobs}"
    assert not [rows for rows in (*whole, *parts) if rows >= INLINE_KEYS], f"here: {whole}, {parts}"


async def test_a_partition_the_copy_holds_many_rows_of_is_judged_in_the_worker_process(
    monkeypatch: pytest.MonkeyPatch, jobs: list[str]
) -> None:
    """Flips if the copy's own rows of a load's partition, past the bound, have their scopes and keys found in the
    serving process (the same work as a whole read's page, over a state's rows the copy holds)."""
    count = INLINE_KEYS * 2
    (l3, log, lock, backend, follower), _registry = _world(page_size=count, workers=_WORKERS)
    await asyncio.to_thread(l3.put, _rows("OH", count))
    await asyncio.to_thread(log.commit, [("OH", i) for i in range(count)])
    await follower.warm()
    assert await follower.catch_up()
    jobs.clear()
    mine = _here(monkeypatch, "partition_mine_of", lambda key, instants, scope, last, rows, held: rows.num_rows)
    await asyncio.to_thread(l3.delete, [("OH", i) for i in range(0, count, 2)])
    await asyncio.to_thread(l3.put, _rows("OH", count // 2, votes=4))
    log.count += 1
    token = f"inc:{log.count}"
    log.records[token] = EpochChange(table=_TABLE, token=token, whole_table=True, partition={"state": "OH"})
    lock.version += 1
    assert await follower.catch_up()
    assert await asyncio.to_thread(_copy, backend) == await asyncio.to_thread(l3.held)
    assert {"partition_mine_job", "partition_page_job"} <= set(jobs), f"judged in the serving process: {jobs}"
    assert not [rows for rows in mine if rows >= INLINE_KEYS], f"the copy's rows judged here: {mine}"


async def test_the_keys_an_apply_replaced_are_made_distinct_for_the_checkpoint_in_the_worker_process(
    monkeypatch: pytest.MonkeyPatch, jobs: list[str]
) -> None:
    """Flips if the keys a table of no scope replaced (what a checkpoint patches its one part by) are made distinct in
    the serving process past the bound: on the loop they held it 20 ms at 100,000 keys and 286 ms at 1,000,000
    (2026-11-25). They are made beside the apply, and the checkpoint still gets them."""
    l3, log, lock, store, pointers = _L3(), _Log(), _Seqlock(), _Store(), _Pointers()
    l3.put(_rows("VA", 10))
    log.commit([("VA", i) for i in range(10)])
    backend = _backend()
    backend.initialize(_metadata())
    tables = [FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS)]
    checkpoints = CopyCheckpoints(
        name="votes",
        tables=tables,
        backend=backend,
        store=store,
        pointers=pointers,
        retire=store.retire,
        owner="a",
        part_rows=40,
    )
    registry = CollectionRegistry()
    registry.follow_generation(_TABLE)
    published = PublishedCopy(backend)
    follower = CopyFollower(
        registry,
        tables,
        published,
        l3,
        settled=lock.settled,
        generation=log.current,
        changes=log,
        page_size=INLINE_KEYS * 4,
        checkpoints=checkpoints,
        drift_workers=_WORKERS,
        queries=published.queries,
    )
    assert not await follower.start()
    assert await follower.catch_up()
    jobs.clear()
    touched = _here(monkeypatch, "touched_of", lambda key, keys: keys.num_rows)
    patched: list[int] = []
    applied = checkpoints.applied

    def counted(rows: int, touched_scopes: Any, keys: Any = None, *args: Any, **kwargs: Any) -> Any:
        if keys:
            patched.append(keys[_TABLE].num_rows)
        return applied(rows, touched_scopes, keys, *args, **kwargs)

    monkeypatch.setattr(checkpoints, "applied", counted)
    count = INLINE_KEYS * 2
    await asyncio.to_thread(l3.put, _rows("IL", count, votes=5))
    await asyncio.to_thread(log.commit, [("IL", i) for i in range(count)])
    lock.version += 1
    assert await follower.catch_up()
    assert "touched_job" in jobs, f"the keys replaced were made distinct in the serving process: {jobs}"
    assert not [rows for rows in touched if rows >= INLINE_KEYS], f"made distinct here: {touched}"
    assert patched == [count], "the checkpoint was not told each key the apply replaced, once"
    # the checkpoint the apply made due is written beside the follower, its read of the copy on a thread: the test's
    # end closes the copy's DuckDB, so it waits for the write first (left running, its thread's unregister met a
    # closed connection: SIGSEGV in DuckDB's ClientContext::LockContext, brooks-14th-mbp, 2026-11-25)
    await follower.checkpointed()
    assert await asyncio.to_thread(_copy, backend) == await asyncio.to_thread(l3.held)


class _ReadMessage(CacheInvalidationMessage):
    """a row message that says when its table is read: the owner reads each message's once as it drains it."""

    def __getattribute__(self, name: str) -> Any:
        if name == "table" and _READS:
            _READS[-1] += 1
        return super().__getattribute__(name)


#: while a test watches: the table reads of row messages in each turn of the loop, the newest last; empty otherwise
_READS: list[int] = []


async def test_messages_heard_while_a_pass_reads_are_drained_a_bounded_run_a_turn() -> None:
    """Flips if the messages heard while a pass read are drained in one turn of the loop, however many: a run of row
    messages read into Arrow took the loop 1.24 ms for 2,000, 78 ms for 200,000 (2026-11-25). Every drain of a pass
    takes them :data:`~threetears.core.collections._A_TURN` at a time, and the next pass reads
    their keys."""
    lock = _Scopes()
    registry = CollectionRegistry()
    l3, log, backend, follower = _scoped_world(lock, registry=registry)
    l3.put(_rows("VA", 5))
    log.commit([("VA", i) for i in range(5)])
    await follower.warm()
    assert await follower.catch_up()
    count = _A_TURN * 4
    await asyncio.to_thread(l3.put, _rows("IL", count, votes=6))
    heard = await asyncio.to_thread(lambda: [_ReadMessage(table=_TABLE, ids=["IL", str(i)]) for i in range(count)])
    l3.put([("VA", 1, 70)])
    log.commit([("VA", 1)])
    lock.commit("state:VA")
    loop = asyncio.get_running_loop()

    def turned() -> None:
        # once a turn of the loop: the reads after it are the next turn's
        if _READS:
            _READS.append(0)
            loop.call_soon(turned)

    def tell_all() -> None:
        # heard while the pass reads the stamp after its stage, before its last drains; its own reads not counted
        for message in heard:
            registry.tell_derived_caches(message)
        _READS.append(0)
        loop.call_soon(turned)

    lock.on_read = (lock.reads + 2, tell_all)
    try:
        assert await follower.catch_up()
        turns = list(_READS)
    finally:
        _READS.clear()
    assert max(turns) <= _A_TURN, f"a turn drained {max(turns)} messages"
    assert sum(turns) >= count, f"the messages were not all drained: {sum(turns)} of {count}"
    lock.commit("state:IL")
    assert await follower.catch_up()
    assert await asyncio.to_thread(_copy, backend) == await asyncio.to_thread(l3.held)


async def test_a_large_stage_is_cut_and_a_moved_scope_left_out_of_it_in_the_worker_process(
    monkeypatch: pytest.MonkeyPatch, jobs: list[str]
) -> None:
    """Flips if a pass's stage past the bound has its keys' scopes found (``staged_of``), or a scope the stamp moved under is
    found among its keys (``leaving_keys_of``) and taken out of its bookkeeping (``without_keys_of``), in the serving
    process: on a thread they took 6.3 ms, 14.7 ms and 12.6 ms of its CPU at 100,000 keys and 64 ms, 163 ms and 156 ms
    at 1,000,000 (2026-11-25). The moved scope's keys still leave the change and the rest applies."""
    lock = _Scopes()
    count = INLINE_KEYS
    l3, log, backend, follower = _scoped_world(lock, page_size=count * 2)
    await asyncio.to_thread(l3.put, _rows("VA", count) + _rows("IL", count))
    await asyncio.to_thread(log.commit, [("VA", i) for i in range(count)] + [("IL", i) for i in range(count)])
    await follower.warm()
    assert await follower.catch_up()
    settled_va = await asyncio.to_thread(_of, backend, "VA")
    jobs.clear()
    seen = {
        name: _here(monkeypatch, name, lambda key, *rest: sum(getattr(t, "num_rows", 0) for t in rest))
        for name in ("staged_of", "leaving_keys_of", "without_keys_of")
    }
    await asyncio.to_thread(l3.put, _rows("VA", count, votes=4) + _rows("IL", count, votes=5))
    await asyncio.to_thread(log.commit, [("VA", i) for i in range(count)] + [("IL", i) for i in range(count)])
    lock.commit("state:VA", "state:IL")

    def moves() -> None:
        # between the pass's two reads of the record: another load commits VA
        l3.put([("VA", 1, 41)])
        log.commit([("VA", 1)])
        lock.commit("state:VA")

    lock.on_read = (lock.reads + 2, moves)
    assert await follower.catch_up()
    assert await asyncio.to_thread(_of, backend, "IL") == await asyncio.to_thread(_l3_of, l3, "IL")
    assert await asyncio.to_thread(_of, backend, "VA") == settled_va
    assert {"staged_scopes_job", "leaving_keys_job", "without_keys_job"} <= set(jobs), f"cut here: {jobs}"
    assert not [name for name, calls in seen.items() if calls], f"cut here: {seen}"
    lock.on_read = None
    assert await follower.catch_up()
    assert await asyncio.to_thread(_copy, backend) == await asyncio.to_thread(l3.held)


async def test_a_checkpoint_pinned_behind_the_last_change_pins_again_off_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flips if a checkpoint whose pinned state missed the last change (an apply landed while it pinned) is written
    from it, claiming a change it does not hold, or holds the event loop while it pins again. The stale pin is let go
    and the state pinned again (a lease and its mark file, read on a thread): the loop is never late, and the checkpoint
    is written from the fresh state."""
    l3, log, lock, store, pointers = _L3(), _Log(), _Seqlock(), _Store(), _Pointers()
    l3.put(_rows("VA", 200))
    log.commit([("VA", i) for i in range(200)])
    _backend_of, writer, _ = _replica(l3, log, lock, store, pointers, "a", ratio=0.0)
    assert not await writer.start()
    assert await writer.catch_up()
    await writer.checkpointed()
    pinned = CopyQueries.current
    pins: list[int | None] = []

    async def behind(self: CopyQueries) -> CopyVersion:
        # the first pin holds a change before the last applied (its mark read none): the checkpoint pins again
        version = await pinned(self)
        pins.append(version.mark)
        if len(pins) == 1:
            version.mark = None
        return version

    monkeypatch.setattr(CopyQueries, "current", behind)
    late = 0.0

    async def tick() -> None:
        nonlocal late
        while True:
            began = time.monotonic()
            await asyncio.sleep(0.01)
            late = max(late, time.monotonic() - began - 0.01)

    ticker = asyncio.create_task(tick())
    l3.put([("VA", 7, 70)])
    log.commit([("VA", 7)])
    lock.version += 1
    assert await writer.catch_up()
    await writer.checkpointed()
    ticker.cancel()
    await asyncio.gather(ticker, return_exceptions=True)

    assert late < 0.5, f"the event loop was {late:.2f} s late: a pin waited for a slot on it"
    assert _pointer_seq(pointers) == 2, "the checkpoint was not written"
    assert len(pins) == 2, f"the checkpoint pinned {len(pins)} times, not once again after the stale pin"
    writer.stop()
