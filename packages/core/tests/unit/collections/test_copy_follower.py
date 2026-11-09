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
- a reader's stamp is exactly the change its snapshot holds.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
from sqlalchemy import BigInteger, Column as SaColumn, MetaData, String, Table

pa = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")

from threetears.core.cache.duckdb import DuckDBBackend  # noqa: E402
from threetears.core.collections import CollectionRegistry  # noqa: E402
from threetears.core.collections.complete_copy import Unsettled  # noqa: E402
from threetears.core.collections.copy_checkpoint import CopyCheckpoints  # noqa: E402
from threetears.core.backends.broker_generation import GENERATION_BY_TRIGGER  # noqa: E402
from threetears.core.collections.copy_follower import (  # noqa: E402
    CopyFollower,
    FollowedTable,
    FollowerPhase,
    HubCapabilityMissingError,
    require_generation_by_trigger,
)
from threetears.core.collections.epoch_change import ChangedRow, EpochChange, EpochWalk, walk_by_count  # noqa: E402
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column  # noqa: E402

_TABLE = "votes"
_KEY = ("state", "geo_id")
_COLUMNS = (
    Column(name="state", column_type=STRING_TYPE),
    Column(name="geo_id", column_type=BIGINT_TYPE),
    Column(name="votes", column_type=BIGINT_TYPE),
)


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
        self.db.executemany(f"INSERT OR REPLACE INTO {_TABLE} VALUES (?, ?, ?)", [list(r) for r in rows])

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
    return l3, log, lock, backend, follower


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
    tables = [FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS)]
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
