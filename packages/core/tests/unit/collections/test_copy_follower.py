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
  follower behind with why, until the log is put back.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from typing import Any

import pytest
from sqlalchemy import BigInteger, Column as SaColumn, MetaData, String, Table

pa = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")

import threetears.core as core  # noqa: E402
from threetears.core.cache.duckdb import DuckDBBackend, StagedChange  # noqa: E402
from threetears.core.collections import CollectionRegistry  # noqa: E402
from threetears.core.collections.complete_copy import Unsettled  # noqa: E402
from threetears.core.collections.copy_checkpoint import CopyCheckpoints  # noqa: E402
from threetears.core.backends.broker_generation import GENERATION_BY_TRIGGER  # noqa: E402
from threetears.core.collections.copy_follower import (  # noqa: E402
    WHOLE_READ_WHYS,
    CopyFollower,
    FollowedTable,
    FollowerPhase,
    HubCapabilityMissingError,
    PartialStamp,
    require_generation_by_trigger,
)
from threetears.core.backends.nats_proxy import NatsProxyL3Backend  # noqa: E402
from threetears.core.collections.change_notes import NotedKeysCursor, NotedKeysPage  # noqa: E402
from threetears.core.collections.epoch_change import ChangedRow, EpochChange, EpochWalk, walk_by_count  # noqa: E402
from threetears.nats import Subject  # noqa: E402
from threetears.nats.errors import NoRespondersError, RequestTimeoutError  # noqa: E402
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column  # noqa: E402
from prometheus_client import REGISTRY  # noqa: E402

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
    )
    return metadata


class _L3:
    """L3 as a DuckDB, answering ``fetch`` with ``$n`` placeholders; counts the rows each statement returned."""

    def __init__(self) -> None:
        self.db = duckdb.connect(":memory:")
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
        self.db.executemany(f"DELETE FROM {_TABLE} WHERE state = ? AND geo_id = ?", [list(k) for k in keys])

    def held(self) -> list[tuple[Any, ...]]:
        return sorted(self.db.execute(f"SELECT state, geo_id, votes FROM {_TABLE}").fetchall())

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        await asyncio.sleep(0)
        if self.fail_next:
            self.fail_next = False
            raise ConnectionError("the L3 rail dropped the reply")
        cursor = self.db.cursor()
        result = cursor.execute(sql, list(params))
        names = [d[0] for d in result.description]
        rows = [dict(zip(names, row, strict=True)) for row in result.fetchall()]
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

    def commit(self, keys: Sequence[tuple[str, int]] | None, *, keep: bool = True) -> str:
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


def _world() -> tuple[tuple[_L3, _Log, _Seqlock, DuckDBBackend, CopyFollower], CollectionRegistry]:
    l3, log, lock = _L3(), _Log(), _Seqlock()
    backend = DuckDBBackend()
    backend.initialize(_metadata())
    registry = CollectionRegistry()
    registry.follow_generation(_TABLE)
    follower = CopyFollower(
        registry,
        [FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS)],
        backend,
        l3,
        settled=lock.settled,
        generation=log.current,
        changes=log,
        page_size=50,
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


async def test_a_readers_stamp_is_the_change_its_snapshot_holds(world: Any) -> None:
    l3, log, lock, backend, follower = world
    await _warmed(world, 10)
    with follower.reading() as (cursor, stamp):
        l3.put([("VA", 1, 5)])
        log.commit([("VA", 1)])
        lock.version += 1
        assert await follower.catch_up()
        # the old snapshot still reads the old rows under the old stamp
        assert cursor.execute(f"SELECT votes FROM {_TABLE} WHERE geo_id = 1").fetchone() == (1,)
        assert stamp == lock.version - 1
    with follower.reading() as (cursor, stamp):
        assert cursor.execute(f"SELECT votes FROM {_TABLE} WHERE geo_id = 1").fetchone() == (5,)
        assert stamp == lock.version


async def test_the_keyed_read_joins_the_keys_to_the_table_on_its_key(world: Any) -> None:
    l3, log, lock, _backend, follower = world
    await _warmed(world, 10)
    asked: list[str] = []
    fetch = l3.fetch

    async def recording(sql: str, *params: Any) -> list[dict[str, Any]]:
        asked.append(sql)
        rows: list[dict[str, Any]] = await fetch(sql, *params)
        return rows

    l3.fetch = recording
    l3.put([("VA", 1, 5)])
    log.commit([("VA", 1)])
    lock.version += 1
    assert await follower.catch_up()
    (sql,) = asked
    assert "JOIN (SELECT unnest($1::text[]) AS k0, unnest($2::text[]) AS k1) AS k" in sql
    assert 't."state" = CAST(k.k0 AS text) AND t."geo_id" = CAST(k.k1 AS bigint)' in sql


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
    l3: _L3, log: _Log, lock: _Seqlock, store: _Store, pointers: _Pointers, owner: str, part_rows: int = 40
) -> tuple[DuckDBBackend, CopyFollower, Any]:
    backend = DuckDBBackend()
    backend.initialize(_metadata())
    tables = [FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS, scope=("state", "state:"))]
    checkpoints = CopyCheckpoints(
        name="votes",
        tables=tables,
        backend=backend,
        store=store,
        pointers=pointers,
        retire=store.retire,
        owner=owner,
        part_rows=part_rows,
    )
    registry = CollectionRegistry()
    registry.follow_generation(_TABLE)
    follower = CopyFollower(
        registry,
        tables,
        backend,
        l3,
        settled=lock.settled,
        generation=log.current,
        changes=log,
        page_size=50,
        checkpoints=checkpoints,
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
    # nothing changed since, so an incremental write would reuse every part: changes of unknown reach write in full
    checkpoints.applied(1)
    with pytest.raises(ConnectionError):
        await checkpoints.write(writer.reading_marked)
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
    assert await checkpoints.write(writer.reading_marked) is None
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
    assert await checkpoints.write(writer.reading_marked) is None


async def test_a_checkpoint_is_due_by_the_rows_applied_since_never_by_time() -> None:
    l3, log, lock, store, pointers = _L3(), _Log(), _Seqlock(), _Store(), _Pointers()
    l3.put(_rows("VA", 100))
    log.commit([("VA", i) for i in range(100)])
    _, writer, checkpoints = _replica(l3, log, lock, store, pointers, "a")
    await writer.start()
    assert await writer.catch_up()
    await writer.checkpointed()
    assert not checkpoints.due()
    checkpoints.applied(49)
    assert not checkpoints.due()
    checkpoints.applied(1)
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


async def test_a_follower_without_partitions_reads_the_record_as_the_whole_table() -> None:
    registry = CollectionRegistry()
    dropped: list[str] = []
    registry.register_derived_cache(_TABLE, on_row=lambda message: None, on_table_dropped=lambda: dropped.append("all"))
    await registry.apply_epoch_change(
        EpochChange(table=_TABLE, token="inc:1", whole_table=True, partition={"state": "OH"})
    )
    assert dropped == ["all"]


async def test_a_partition_on_a_column_the_copy_does_not_hold_filters_l3_by_it() -> None:
    """a load's constants include a version stamp the copy does not hold: L3 filters on it, and only that load's rows
    are read."""
    l3, log, lock = _L3(), _Log(), _Seqlock()
    l3.db.execute(f"ALTER TABLE {_TABLE} ADD COLUMN ver BIGINT DEFAULT 0")
    backend = DuckDBBackend()
    backend.initialize(_metadata())
    registry = CollectionRegistry()
    registry.follow_generation(_TABLE)
    table = FollowedTable(
        name=_TABLE,
        key=_KEY,
        columns=_COLUMNS,
        l3_kinds=(("state", STRING_TYPE), ("geo_id", BIGINT_TYPE), ("votes", BIGINT_TYPE), ("ver", BIGINT_TYPE)),
    )
    follower = CopyFollower(
        registry, [table], backend, l3, settled=lock.settled, generation=log.current, changes=log, page_size=50
    )
    follower.register()
    l3.put(_rows("VA", 40))
    await follower.warm()
    assert await follower.catch_up()
    l3.db.executemany(f"INSERT INTO {_TABLE} VALUES (?, ?, ?, 5)", [["OH", i, 2] for i in range(12)])
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
        found = [note for note in self.notes if after < note[0] <= until]
        taken, more = found[:limit], len(found) > limit
        if more:
            taken = [note for note in taken if note[0] < found[limit][0]]
        return NotedKeysPage(
            keys=tuple(keys for _, keys in taken),
            whole_table=False,
            next=NotedKeysCursor(slot=0, after=taken[-1][0]) if more else None,
        )


def _too_large(log: _Log, *, keep: bool = True) -> tuple[datetime, datetime]:
    """a commit the broker recorded keys unknown, with its notes bracket."""
    since = datetime.now(UTC)
    until = since + timedelta(seconds=1)
    log.count += 1
    token = f"inc:{log.count}"
    if keep:
        log.records[token] = EpochChange(table=_TABLE, token=token, keys_unknown=True, noted_between=(since, until))
    return since, until


#: a commit naming more rows than sixteen change records of 256 KiB hold, at about sixty bytes a named row
_PAST_SIXTEEN_RECORDS = 70_000


async def test_a_commit_too_large_to_record_reaches_the_copy_exactly_from_its_notes(noted_world: Any) -> None:
    world, registry = noted_world
    l3, log, lock, backend, follower = world
    await _warmed(world, 120)
    # updates of rows held, inserts past them, and deletes, all in one commit
    changed = [("VA", i, 7) for i in range(60, _PAST_SIXTEEN_RECORDS)]
    deleted = [("VA", i) for i in range(10, 20)]
    l3.put(changed)
    l3.delete(deleted)
    since, _ = _too_large(log)
    keys = [(s, str(g)) for s, g, _ in changed] + [(s, str(g)) for s, g in deleted]
    notes = _Notes([(since + timedelta(microseconds=index // 4), key) for index, key in enumerate(keys)])
    registry.set_noted_keys_reader(notes)
    lock.version += 1
    l3.rows_read = 0

    assert await follower.catch_up()

    assert _copy(backend) == l3.held(), "the copy equals L3, row for row"
    assert l3.rows_read == len(changed), "the keys' rows that exist, never the table"
    assert notes.pages > 1
    assert follower.status().behind == {}
    assert follower.stamp == lock.version


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


# --- a write in progress holds only its scopes (0.95.0) -------------------------------------------------------------


class _Scopes:
    """the writer's record as ScopeEpochs answers it: each scope's epoch, and the scopes a write in progress holds."""

    def __init__(self) -> None:
        self.version = 1
        self.epochs: dict[str, int] = {"state:VA": 1, "state:IL": 1}
        self.held: frozenset[str] | None = None
        self.on_read: Any = None
        self.reads = 0

    async def settled(self) -> Any:
        self.reads += 1
        if self.on_read is not None and self.on_read[0] == self.reads:
            self.on_read[1]()
        if self.held is None:
            return SimpleNamespace(version=self.version, epochs=dict(self.epochs), pending=frozenset())
        return Unsettled("a long write", epochs=dict(self.epochs), pending=self.held, version=self.version)

    def commit(self, *scopes: str) -> None:
        self.version += 1
        for scope in scopes:
            self.epochs[scope] = self.version


def _scoped_world(
    lock: _Scopes, *, checkpoint_from: DuckDBBackend | None = None
) -> tuple[_L3, _Log, DuckDBBackend, CopyFollower]:
    l3, log = _L3(), _Log()
    backend = DuckDBBackend()
    backend.initialize(_metadata())
    registry = CollectionRegistry()
    registry.follow_generation(_TABLE)
    follower = CopyFollower(
        registry,
        [FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS, scope=("state", "state:"))],
        backend,
        l3,
        settled=lock.settled,
        generation=log.current,
        changes=log,
        page_size=50,
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


async def test_a_scope_the_write_takes_while_a_pass_reads_is_never_applied_from_that_pass() -> None:
    lock = _Scopes()
    l3, log, backend, follower = _scoped_world(lock)
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


async def test_an_apply_that_fails_keeps_its_stages_and_the_next_pass_reads_nothing(world: Any) -> None:
    l3, log, lock, backend, follower = world
    await _warmed(world, 10)
    l3.put([("VA", 1, 5)])
    log.commit([("VA", 1)])
    lock.version += 1
    real = backend.apply_staged
    calls = {"n": 0}

    def failing(*args: Any, **kwargs: Any) -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("the copy's apply failed")
        return real(*args, **kwargs)

    backend.apply_staged = failing  # type: ignore[method-assign]
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


def _office_world(lock: _Scopes) -> tuple[_L3, _Log, DuckDBBackend, CopyFollower]:
    """a table keyed by a column that leads without the scope (an office's key), its scope in the row only."""
    l3, log = _L3(), _Log()
    backend = DuckDBBackend()
    backend.initialize(_metadata())
    registry = CollectionRegistry()
    registry.follow_generation(_TABLE)
    follower = CopyFollower(
        registry,
        [FollowedTable(name=_TABLE, key=("geo_id", "state"), columns=_COLUMNS, scope=("state", "state:"))],
        backend,
        l3,
        settled=lock.settled,
        generation=log.current,
        changes=log,
        page_size=50,
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
    Table(_DERIVED, metadata, SaColumn("state", String(8), primary_key=True), SaColumn("total", BigInteger))
    backend = DuckDBBackend()
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
    follower = CopyFollower(
        registry,
        followed,
        backend,
        l3,
        settled=lock.settled,
        generation=log.current,
        changes=log,
        page_size=50,
        checkpoints=checkpoints,
    )
    return backend, follower


def _derive(backend: DuckDBBackend, total: int) -> None:
    """the owner's settle: the derived table's rows, applied to the copy."""
    stage = "__stage__totals__rows"
    backend.stage_rows(stage, _DERIVED, pa.table({"state": ["VA"], "total": pa.array([total], pa.int64())}), ("state",))
    backend.apply_staged([StagedChange(table=_DERIVED, key=("state",), rows=stage, whole=True)])


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
    backend = DuckDBBackend()
    backend.initialize(_metadata())
    registry = CollectionRegistry()
    registry.follow_generation(_TABLE)
    follower = CopyFollower(
        registry,
        [FollowedTable(name=_TABLE, key=("geo_id",), columns=_COLUMNS, scope=("state", "state:"))],
        backend,
        l3,
        settled=lock.settled,
        generation=log.current,
        changes=log,
        page_size=50,
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


def _spy_whole_reads(follower: CopyFollower) -> list[str]:
    """every why ``follower._stage_whole`` (the one whole read of a table from L3) is reached with, in order."""
    whys: list[str] = []
    real = follower._stage_whole  # noqa: SLF001 - the one whole read, spied on

    async def stage_whole(table: FollowedTable, rows_stage: str, why: str, held: frozenset[str]) -> set[str | None]:
        whys.append(why)
        return await real(table, rows_stage, why, held)

    follower._stage_whole = stage_whole  # type: ignore[method-assign]  # noqa: SLF001
    return whys


async def test_stage_whole_is_reached_only_at_a_cold_start_a_reset_a_schema_change_or_a_legacy_whole_record() -> None:
    """the proving test of CopyFollower._stage_whole and CopyFollower._stage's ledger lines: every steady-state path
    (keys, a load's partition, a load's key prefixes, a stamp that moves, a failed pass, a start from a checkpoint and
    its checkpoints) reads nothing whole; only a cold start, an operator's reset, a schema change and a whole record
    with neither prefixes nor a reason (a writer older than 0.95.0 mid-roll) reach _stage_whole, each counted."""
    l3, log, lock, store, pointers = _L3(), _Log(), _Seqlock(), _Store(), _Pointers()
    l3.put(_rows("VA", 120))
    log.commit([("VA", i) for i in range(120)])
    _, first, _ = _replica(l3, log, lock, store, pointers, "a")
    first_whys = _spy_whole_reads(first)
    assert not await first.start()
    assert await first.catch_up()
    await first.checkpointed()
    assert first_whys == ["cold_start"]
    # a second replica starts from the checkpoint: no whole read
    backend, follower, _ = _replica(l3, log, lock, store, pointers, "b")
    whys = _spy_whole_reads(follower)
    counted = {why: _whole_reads_counted(_TABLE, why) for why in WHOLE_READ_WHYS}
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
    assert whys == []
    assert {why: _whole_reads_counted(_TABLE, why) for why in WHOLE_READ_WHYS} == counted
    # the four that do read whole, each once and counted by its why
    _record_whole(log, reason="reset")
    lock.version += 1
    assert await follower.catch_up()
    follower.schema_changed(_TABLE)
    assert await follower.catch_up()
    _record_whole(log)  # a writer older than 0.95.0: neither prefixes nor a reason
    lock.version += 1
    assert await follower.catch_up()
    assert _copy(backend) == l3.held()
    assert whys == ["reset", "schema_change", "legacy_whole_record"]
    for why in ("reset", "schema_change", "legacy_whole_record"):
        assert _whole_reads_counted(_TABLE, why) == counted[why] + 1
    assert _whole_reads_counted(_TABLE, "cold_start") == counted["cold_start"]


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".")[:3])


def test_the_older_writers_whole_record_read_is_gone_by_0_96_0() -> None:
    """legacy_whole_record lives only for the 0.95.0 roll (Pace, 2026-11-09): from 3tears 0.96.0 it fails until the
    code is deleted: CopyFollower._on_dropped's re-read and the non-key branch of CopyFollower._on_partition (a whole
    record then is the registry's to refuse), WHOLE_READ_WHYS["legacy_whole_record"], this test's twin assertions in
    test_stage_whole_is_reached_only_at_..., and the why in its ledger reason."""
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

    checkpoints.stage = stage  # type: ignore[method-assign]
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
