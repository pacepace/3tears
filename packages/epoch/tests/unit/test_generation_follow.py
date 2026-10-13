"""following collection write generations over the epoch bucket: the catch-up pass and the key watcher.

Two registries on one in-process bus and bucket stand in for a writer pod and a follower pod, with
the real generation source, the real reader and the real pass. The contract this pins:

- a missed row broadcast is caught by ONE pass, with no sleep: the follower's row is gone;
- a replaced bucket is caught by one pass;
- an opted-out table, a collection built with ``NO_L2`` and a table that is not switched on put
  nothing in the epoch bucket at all;
- one transaction is one advance, and a follower that hears every row of it does not drop;
- a flush is one advance per table;
- a generation that cannot be read drops nothing and leaves the mark where it was;
- a cache derived from the table is evicted row by row for a heard change, and emptied only when
  the table drops;
- one table's failure does not abandon the rest of the pass;
- a key watcher judges each pushed generation the same way, after giving its broadcasts time.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, ClassVar, Literal

import pytest
from sqlalchemy import Column, DateTime, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import (
    NO_L2,
    WRITE_GENERATION,
    BaseCollection,
    CacheInvalidationMessage,
    CallerTransaction,
    CollectionRegistry,
    NoWriteGeneration,
    WriteBuffer,
    flush_pending,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient
from threetears.epoch import (
    EpochGenerationReader,
    EpochGenerationSource,
    follow_generation_key,
    generation_catchup_tick,
    generation_kv_key,
)
from threetears.nats import KvError, set_default_namespace

_TABLE = "group_members"
_OTHER = "role_assignments"

#: build the collection with whatever L2 client its registry offers
_FROM_REGISTRY = object()


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace("followprobe")


def _metadata() -> MetaData:
    metadata = MetaData()
    for table in (_TABLE, _OTHER):
        Table(
            table,
            metadata,
            Column("id", String(255), primary_key=True),
            Column("member_id", String(255)),
            Column("date_created", DateTime(timezone=True)),
            Column("date_updated", DateTime(timezone=True)),
        )
    return metadata


class _Row(BaseEntity):
    primary_key_field = "id"


class _Rows(BaseCollection[_Row]):
    """an in-process three-tier collection over an L3 dict every pod of the test shares."""

    datetime_columns: ClassVar[frozenset[str]] = frozenset({"date_created", "date_updated"})
    table: ClassVar[str] = _TABLE

    def __init__(
        self,
        registry: CollectionRegistry,
        rows: dict[str, dict[str, Any]],
        *,
        nats_client: Any = _FROM_REGISTRY,
        write_buffer: WriteBuffer | None = None,
    ) -> None:
        self._rows = rows
        if nats_client is _FROM_REGISTRY:
            super().__init__(registry, DefaultCoreConfig(), write_buffer=write_buffer)
        else:
            super().__init__(registry, DefaultCoreConfig(), nats_client, write_buffer=write_buffer)

    @property
    def table_name(self) -> str:
        return self.table

    @property
    def entity_class(self) -> type[_Row]:
        return _Row

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        row = self._rows.get(str(entity_id))
        return dict(row) if row is not None else None

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        self._rows[str(data["id"])] = dict(data)
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        self._rows.pop(str(entity_id), None)

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        row: dict[str, Any] = json.loads(data)
        return row


class _Members(_Rows):
    write_generation = WRITE_GENERATION
    invalidation_columns: ClassVar[tuple[str, ...]] = ("member_id",)


class _Assignments(_Rows):
    table: ClassVar[str] = _OTHER
    write_generation = WRITE_GENERATION


class _UndeclaredMembers(_Rows):
    """as every collection is until it is switched on."""


class _OptedOutMembers(_Rows):
    write_generation = NoWriteGeneration(reason="written on every turn; nothing reads a row by key")


class _WriteBehindMembers(_Members):
    l3_write_policy: ClassVar[Literal["synchronous", "write_behind"] | None] = "write_behind"


class _WriteBehindAssignments(_Assignments):
    l3_write_policy: ClassVar[Literal["synchronous", "write_behind"] | None] = "write_behind"


class _Lossy(FakeNatsClient):
    """a bus that loses every row message published while it is deaf."""

    def __init__(self) -> None:
        super().__init__()
        self.deaf = False

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:
        if self.deaf:
            return
        await super().publish(subject=subject, message=message, reply_to=reply_to)


class _Pod:
    """one registry with its own L1 and listener, wired as a hub wires one: a source and a reader."""

    def __init__(self, bus: _Lossy, *, scope: str) -> None:
        l1 = SQLiteBackend(db_name=f"generation_follow_{uuid.uuid4().hex[:8]}")
        l1.initialize(_metadata())
        self.bus = bus
        self.registry = CollectionRegistry()
        self.registry.configure(l1_backend=l1, l2_client=bus, l3_pool=object(), kv_key_scope=scope)  # type: ignore[arg-type]
        self.source = EpochGenerationSource(bus)
        self.registry.set_generation_source(self.source)
        self.reader = EpochGenerationReader(bus)

    async def listen(self) -> None:
        await self.registry.start_invalidation_listener(self.bus)  # type: ignore[arg-type]

    async def follow(self, *tables: str) -> None:
        """follow the tables and take each one's first mark, as a pod's first pass does."""
        for table in tables:
            self.registry.follow_generation(table)
        await generation_catchup_tick(self.registry, self.reader)

    async def tick(self) -> int:
        return await generation_catchup_tick(self.registry, self.reader)


async def _save(collection: _Rows, entity_id: str, member_id: str = "ada") -> None:
    await collection.save_entity(collection.create({"id": entity_id, "member_id": member_id}))


async def _member_id(collection: _Rows, entity_id: str) -> str | None:
    row = await collection.ensure(entity_id)
    return None if row is None else str(row["member_id"])


async def _epochs(bus: FakeNatsClient) -> FakeKvBucket:
    return await bus.kv_bucket(name="epochs")


async def _count(bus: FakeNatsClient, table: str) -> int | None:
    """the table's advance count as the bucket holds it, or ``None`` when it holds no generation."""
    raw = await (await _epochs(bus)).get(key=generation_kv_key(table))
    return None if raw is None else int(raw.decode().rpartition(":")[2])


async def _never_written(bus: FakeNatsClient, table: str) -> bool:
    """whether the epoch bucket has never had any message for the table's generation key."""
    _, revision = await (await _epochs(bus)).get_latest(key=generation_kv_key(table))
    return revision == 0


class _Conn:
    """a connection whose transaction does nothing, for a CallerTransaction to open."""

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[None]:
        yield

    def transaction(self, **options: Any) -> Any:
        return self._transaction()


@pytest.fixture
def bus() -> _Lossy:
    return _Lossy()


async def _two_pods(bus: _Lossy) -> tuple[_Pod, _Members, _Pod, _Members]:
    """a writer and a follower of the membership table, the follower listening and marked."""
    rows: dict[str, dict[str, Any]] = {}
    writer, follower = _Pod(bus, scope="writer"), _Pod(bus, scope="follower")
    writing, reading = _Members(writer.registry, rows), _Members(follower.registry, rows)
    await follower.listen()
    # the table's generation exists before the follower first looks, as it does on a running
    # platform; a table nobody has written yet is covered on its own below.
    await _save(writing, "seed")
    await follower.follow(_TABLE)
    return writer, writing, follower, reading


class TestAdvanceSaysWhatItWrote:
    async def test_the_token_returned_is_the_one_in_the_bucket(self, bus: _Lossy) -> None:
        source = EpochGenerationSource(bus)
        first = await source.advance(_TABLE)
        assert first == await source.current(_TABLE)
        second = await source.advance(_TABLE)
        assert second == await source.current(_TABLE)
        assert (first.rpartition(":")[0], first.rpartition(":")[2]) == (second.rpartition(":")[0], "1")
        assert second.endswith(":2")

    async def test_concurrent_advances_each_get_their_own_token(self, bus: _Lossy) -> None:
        source = EpochGenerationSource(bus)
        await source.current(_TABLE)
        tokens = await asyncio.gather(*(source.advance(_TABLE) for _ in range(12)))
        assert sorted(int(token.rpartition(":")[2]) for token in tokens) == list(range(1, 13))


class TestAReaderNeverWrites:
    async def test_a_table_with_no_generation_reads_as_none_and_stays_unwritten(self, bus: _Lossy) -> None:
        await _epochs(bus)  # the hub declared the bucket; the reader only binds it
        reader = EpochGenerationReader(bus)
        assert await reader.read(_TABLE) is None
        assert await _never_written(bus, _TABLE)

    async def test_it_reads_what_a_source_wrote(self, bus: _Lossy) -> None:
        token = await EpochGenerationSource(bus).advance(_TABLE)
        assert await EpochGenerationReader(bus).read(_TABLE) == token

    async def test_it_binds_the_bucket_and_does_not_create_it(self, bus: _Lossy) -> None:
        with pytest.raises(GenerationUnavailableError):
            await EpochGenerationReader(bus).read(_TABLE)
        assert not bus.bucket_exists("epochs")

    async def test_a_value_that_is_not_a_generation_is_unavailable(self, bus: _Lossy) -> None:
        await (await _epochs(bus)).put(key=generation_kv_key(_TABLE), value=b"not-a-generation")
        with pytest.raises(GenerationUnavailableError):
            await EpochGenerationReader(bus).read(_TABLE)


class TestAMissedBroadcastIsCaughtByOnePass:
    async def test_the_followers_row_is_gone_after_one_pass_and_no_sleep(self, bus: _Lossy) -> None:
        _, writing, follower, reading = await _two_pods(bus)
        await _save(writing, "m1", "ada")
        assert await _member_id(reading, "m1") == "ada"
        assert reading.exists_in_cache_sync("m1")

        bus.deaf = True
        await _save(writing, "m1", "grace")
        bus.deaf = False
        assert await _member_id(reading, "m1") == "ada", "nothing told the follower, so it serves what it cached"

        assert await follower.tick() == 1

        assert not reading.exists_in_cache_sync("m1")
        assert await _member_id(reading, "m1") == "grace"

    async def test_the_pass_after_a_drop_finds_nothing_to_do(self, bus: _Lossy) -> None:
        _, writing, follower, _ = await _two_pods(bus)
        bus.deaf = True
        await _save(writing, "m1")
        bus.deaf = False
        assert await follower.tick() == 1
        assert await follower.tick() == 0

    async def test_a_heard_write_drops_nothing(self, bus: _Lossy) -> None:
        _, writing, follower, reading = await _two_pods(bus)
        await _save(writing, "m1", "ada")
        await _save(writing, "m2", "grace")
        assert await _member_id(reading, "m2") == "grace"
        await _save(writing, "m1", "edsger")
        assert await follower.tick() == 0
        # the row that changed was evicted by its own broadcast; the row that did not is untouched
        assert not reading.exists_in_cache_sync("m1")
        assert reading.exists_in_cache_sync("m2")

    async def test_a_registry_that_both_writes_and_follows_does_not_drop_for_its_own_writes(self, bus: _Lossy) -> None:
        pod = _Pod(bus, scope="hub")
        members = _Members(pod.registry, {})
        await pod.listen()
        await _save(members, "seed")
        await pod.follow(_TABLE)
        assert await _member_id(members, "seed") == "ada"
        await _save(members, "m1")
        await members.delete("m1")
        assert await pod.tick() == 0
        assert members.exists_in_cache_sync("seed")


class TestAReplacedBucketIsCaught:
    async def test_a_bucket_emptied_and_written_again_drops_the_table(self, bus: _Lossy) -> None:
        _, writing, follower, reading = await _two_pods(bus)
        await _save(writing, "m1", "ada")
        assert await _member_id(reading, "m1") == "ada"

        (await _epochs(bus)).wipe()
        bus.deaf = True
        await _save(writing, "m1", "grace")
        bus.deaf = False

        assert await follower.tick() == 1
        assert await _member_id(reading, "m1") == "grace"
        assert await follower.tick() == 0

    async def test_a_bucket_emptied_and_not_yet_written_drops_the_table(self, bus: _Lossy) -> None:
        _, writing, follower, reading = await _two_pods(bus)
        await _save(writing, "m1", "ada")
        assert await _member_id(reading, "m1") == "ada"
        (await _epochs(bus)).wipe()
        assert await follower.tick() == 1
        assert not reading.exists_in_cache_sync("m1")
        assert await follower.tick() == 0

    async def test_hearing_every_row_under_the_new_incarnation_does_not_excuse_the_old_ones(self, bus: _Lossy) -> None:
        _, writing, follower, _ = await _two_pods(bus)
        (await _epochs(bus)).wipe()
        await _save(writing, "m1")
        assert await follower.tick() == 1


class TestWhatPutsNothingInTheEpochBucket:
    async def test_an_opted_out_table(self, bus: _Lossy) -> None:
        pod = _Pod(bus, scope="writer")
        members = _OptedOutMembers(pod.registry, {})
        await _save(members, "m1")
        await members.invalidate_cache_many(["m1", "m2"])
        await members.delete("m1")
        assert await _never_written(bus, _TABLE)
        assert (await _epochs(bus)).keys() == ()

    async def test_a_switched_on_collection_built_with_no_l2(self, bus: _Lossy) -> None:
        pod = _Pod(bus, scope="writer")
        members = _Members(pod.registry, {}, nats_client=NO_L2)
        await _save(members, "m1")
        await members.invalidate_cache_many(["m1", "m2"])
        await members.delete("m1")
        assert await _never_written(bus, _TABLE)
        assert (await _epochs(bus)).keys() == ()

    async def test_a_table_that_is_not_switched_on(self, bus: _Lossy) -> None:
        pod = _Pod(bus, scope="writer")
        members = _UndeclaredMembers(pod.registry, {})
        await _save(members, "m1")
        await members.invalidate_cache_many(["m1", "m2"])
        async with CallerTransaction(_Conn()) as transaction:
            transaction.enroll(members, "m1")
        await members.delete("m1")
        assert await _never_written(bus, _TABLE)
        assert (await _epochs(bus)).keys() == ()


class TestOneCommitIsOneAdvance:
    async def test_a_transaction_of_many_rows_moves_the_count_by_one_and_a_follower_that_heard_them_all_does_not_drop(
        self, bus: _Lossy
    ) -> None:
        _, writing, follower, reading = await _two_pods(bus)
        await _save(writing, "kept", "ada")
        assert await _member_id(reading, "kept") == "ada"
        before = await _count(bus, _TABLE)
        assert before is not None

        conn = _Conn()
        async with CallerTransaction(conn):
            for index in range(6):
                await writing.save_entity(writing.create({"id": f"m{index}", "member_id": "grace"}), conn=conn)

        assert await _count(bus, _TABLE) == before + 1
        assert await follower.tick() == 0
        assert reading.exists_in_cache_sync("kept")

    async def test_a_follower_that_missed_one_row_of_the_transaction_drops(self, bus: _Lossy) -> None:
        _, writing, follower, reading = await _two_pods(bus)
        await _save(writing, "kept", "ada")
        assert await _member_id(reading, "kept") == "ada"
        lost: list[str] = []
        deliver = FakeNatsClient.publish

        async def lose_one(*, subject: Any, message: Any, reply_to: Any = None) -> None:
            if isinstance(message, CacheInvalidationMessage) and message.ids == ["m3"]:
                lost.append(message.ids[0])
                return
            await deliver(bus, subject=subject, message=message, reply_to=reply_to)

        bus.publish = lose_one  # type: ignore[method-assign]
        conn = _Conn()
        async with CallerTransaction(conn):
            for index in range(6):
                await writing.save_entity(writing.create({"id": f"m{index}", "member_id": "grace"}), conn=conn)
        assert lost == ["m3"]

        assert await follower.tick() == 1
        assert not reading.exists_in_cache_sync("kept")

    async def test_a_flush_advances_once_per_table(self, bus: _Lossy) -> None:
        rows: dict[str, dict[str, Any]] = {}
        other_rows: dict[str, dict[str, Any]] = {}
        pod = _Pod(bus, scope="writer")
        buffer = WriteBuffer()
        members = _WriteBehindMembers(pod.registry, rows, write_buffer=buffer)
        assignments = _WriteBehindAssignments(pod.registry, other_rows, write_buffer=buffer)
        for index in range(5):
            await _save(members, f"m{index}")
        for index in range(3):
            await _save(assignments, f"a{index}")
        assert await _never_written(bus, _TABLE), "a write-behind save is not in L3 yet"
        assert await _never_written(bus, _OTHER)

        assert await flush_pending(buffer, pod.registry) == 8

        assert (len(rows), len(other_rows)) == (5, 3)
        assert await _count(bus, _TABLE) == 1
        assert await _count(bus, _OTHER) == 1

    async def test_a_follower_that_hears_a_flush_does_not_drop(self, bus: _Lossy) -> None:
        rows: dict[str, dict[str, Any]] = {}
        writer, follower = _Pod(bus, scope="writer"), _Pod(bus, scope="follower")
        buffer = WriteBuffer()
        writing = _WriteBehindMembers(writer.registry, rows, write_buffer=buffer)
        reading = _Members(follower.registry, rows)
        await follower.listen()
        await _save(writing, "seed")
        await flush_pending(buffer, writer.registry)
        await follower.follow(_TABLE)
        assert await _member_id(reading, "seed") == "ada"

        for index in range(4):
            await _save(writing, f"m{index}")
        await flush_pending(buffer, writer.registry)

        assert await follower.tick() == 0
        assert reading.exists_in_cache_sync("seed")


class TestAGenerationThatCannotBeReadIsNotAReset:
    async def test_it_drops_nothing_and_leaves_the_mark_where_it_was(self, bus: _Lossy) -> None:
        _, writing, follower, reading = await _two_pods(bus)
        await _save(writing, "m1", "ada")
        assert await _member_id(reading, "m1") == "ada"
        mark = follower.registry.generation_marks.recorded(_TABLE)
        assert mark is not None
        bucket = await _epochs(bus)

        bucket.become_unreachable(KvError("broker unreachable"))
        assert await follower.tick() == 0
        bucket.become_reachable()

        assert reading.exists_in_cache_sync("m1")
        assert follower.registry.generation_marks.recorded(_TABLE) == mark
        # and the pod is not wedged: the next pass reads, and finds it heard everything
        assert await follower.tick() == 0
        assert reading.exists_in_cache_sync("m1")

    async def test_a_write_missed_while_the_bucket_was_unreadable_is_still_caught_once_it_reads(
        self, bus: _Lossy
    ) -> None:
        _, writing, follower, reading = await _two_pods(bus)
        await _save(writing, "m1", "ada")
        assert await _member_id(reading, "m1") == "ada"
        bus.deaf = True
        await _save(writing, "m1", "grace")
        bus.deaf = False
        bucket = await _epochs(bus)

        bucket.become_unreachable(KvError("broker unreachable"))
        assert await follower.tick() == 0
        assert await _member_id(reading, "m1") == "ada"
        bucket.become_reachable()

        assert await follower.tick() == 1
        assert await _member_id(reading, "m1") == "grace"

    async def test_one_tables_failure_does_not_abandon_the_rest_of_the_pass(self, bus: _Lossy) -> None:
        rows: dict[str, dict[str, Any]] = {}
        writer, follower = _Pod(bus, scope="writer"), _Pod(bus, scope="follower")
        writing_members, writing_assignments = _Members(writer.registry, rows), _Assignments(writer.registry, {})
        reading_members = _Members(follower.registry, rows)
        await follower.listen()
        await _save(writing_members, "seed")
        await _save(writing_assignments, "seed")
        await follower.follow(_OTHER, _TABLE)
        assert await _member_id(reading_members, "seed") == "ada"
        bus.deaf = True
        await _save(writing_members, "seed", "grace")
        bus.deaf = False

        class _BrokenForOne(EpochGenerationReader):
            async def read(self, table_name: str) -> str | None:
                if table_name == _OTHER:
                    raise RuntimeError("a bug in the first table's read")
                return await super().read(table_name)

        with pytest.raises(RuntimeError, match="first table"):
            await generation_catchup_tick(follower.registry, _BrokenForOne(bus))

        # the table after the broken one was still judged, and dropped for the write it missed
        assert await _member_id(reading_members, "seed") == "grace"


class _PersonCache:
    """a cache derived from the membership table, keyed by the person, not by the row."""

    def __init__(self, registry: CollectionRegistry, people: tuple[str, ...]) -> None:
        self.entries = {person: f"access-of-{person}" for person in people}
        self.emptied = 0
        registry.register_derived_cache(_TABLE, on_row=self._on_row, on_table_dropped=self._on_table_dropped)

    def _on_row(self, message: CacheInvalidationMessage) -> None:
        assert message.columns is not None
        self.entries.pop(str(message.columns["member_id"]), None)

    def _on_table_dropped(self) -> None:
        self.emptied += 1
        self.entries.clear()


class TestADerivedCacheIsEvictedRowByRowAndEmptiedOnlyOnATableDrop:
    async def test_a_heard_change_evicts_one_entry_and_a_missed_one_empties_the_cache(self, bus: _Lossy) -> None:
        _, writing, follower, _ = await _two_pods(bus)
        cache = _PersonCache(follower.registry, ("ada", "grace", "edsger"))

        await _save(writing, "m1", "grace")
        assert await follower.tick() == 0
        assert cache.entries == {"ada": "access-of-ada", "edsger": "access-of-edsger"}
        assert cache.emptied == 0

        bus.deaf = True
        await _save(writing, "m2", "ada")
        bus.deaf = False
        assert cache.entries == {"ada": "access-of-ada", "edsger": "access-of-edsger"}
        assert await follower.tick() == 1
        assert cache.entries == {}
        assert cache.emptied == 1

    async def test_a_replaced_bucket_empties_the_cache(self, bus: _Lossy) -> None:
        _, _, follower, _ = await _two_pods(bus)
        cache = _PersonCache(follower.registry, ("ada",))
        (await _epochs(bus)).wipe()
        assert await follower.tick() == 1
        assert cache.emptied == 1

    async def test_an_unreadable_generation_empties_nothing(self, bus: _Lossy) -> None:
        _, _, follower, _ = await _two_pods(bus)
        cache = _PersonCache(follower.registry, ("ada",))
        bucket = await _epochs(bus)
        bucket.become_unreachable(KvError("broker unreachable"))
        assert await follower.tick() == 0
        assert cache.entries == {"ada": "access-of-ada"}
        assert cache.emptied == 0


async def _until(condition: Callable[[], bool]) -> None:
    """let the event loop run until ``condition`` holds; fail rather than wait for ever."""
    for _ in range(2000):
        if condition():
            return
        await asyncio.sleep(0)
    raise AssertionError("the watcher never reached the expected state")


@asynccontextmanager
async def _watching(pod: _Pod, table: str, *, grace: timedelta) -> AsyncIterator[None]:
    task = asyncio.create_task(follow_generation_key(pod.registry, pod.reader, table, grace=grace))
    try:
        yield
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


class TestAKeyWatcherFollowsWithoutAPass:
    async def test_a_heard_write_is_not_dropped_and_a_missed_one_is(self, bus: _Lossy) -> None:
        rows: dict[str, dict[str, Any]] = {}
        writer, follower = _Pod(bus, scope="writer"), _Pod(bus, scope="follower")
        writing, reading = _Members(writer.registry, rows), _Members(follower.registry, rows)
        await follower.listen()
        await _save(writing, "seed")
        cache = _PersonCache(follower.registry, ("ada", "grace"))
        marks = follower.registry.generation_marks

        async with _watching(follower, _TABLE, grace=timedelta(seconds=30)):
            # the key's latest value is pushed first, and is this pod's first sight of the table
            await _until(lambda: marks.follows(_TABLE) and marks.recorded(_TABLE) is not None)
            assert cache.emptied == 1
            cache.entries.update({"ada": "access-of-ada", "grace": "access-of-grace"})
            assert await _member_id(reading, "seed") == "ada"

            # the pushed generation arrives ahead of its broadcast; the watcher waits for it
            await _save(writing, "m1", "grace")
            await _until(lambda: marks.recorded(_TABLE) == f"{marks.recorded(_TABLE).rpartition(':')[0]}:2")  # type: ignore[union-attr]
            await asyncio.sleep(0)
            assert cache.entries == {"ada": "access-of-ada"}
            assert cache.emptied == 1
            assert reading.exists_in_cache_sync("seed")

        async with _watching(follower, _TABLE, grace=timedelta(0)):
            await _until(lambda: marks.follows(_TABLE))
            bus.deaf = True
            await _save(writing, "seed", "edsger")
            bus.deaf = False
            await _until(lambda: cache.emptied == 2)
            assert not reading.exists_in_cache_sync("seed")
            assert await _member_id(reading, "seed") == "edsger"

    async def test_a_new_incarnation_is_dropped_without_waiting_for_broadcasts(self, bus: _Lossy) -> None:
        rows: dict[str, dict[str, Any]] = {}
        writer, follower = _Pod(bus, scope="writer"), _Pod(bus, scope="follower")
        writing, reading = _Members(writer.registry, rows), _Members(follower.registry, rows)
        await follower.listen()
        await _save(writing, "seed")
        marks = follower.registry.generation_marks
        async with _watching(follower, _TABLE, grace=timedelta(hours=1)):
            await _until(lambda: marks.recorded(_TABLE) is not None)
            first = marks.recorded(_TABLE)
            assert await _member_id(reading, "seed") == "ada"
            (await _epochs(bus)).wipe()
            bus.deaf = True
            await _save(writing, "seed", "grace")
            bus.deaf = False
            await _until(lambda: marks.recorded(_TABLE) != first)
            assert await _member_id(reading, "seed") == "grace"

    async def test_a_reader_whose_bucket_cannot_be_opened_cannot_watch(self, bus: _Lossy) -> None:
        follower = _Pod(bus, scope="follower")
        with pytest.raises(GenerationUnavailableError):
            await follow_generation_key(follower.registry, follower.reader, _TABLE)
