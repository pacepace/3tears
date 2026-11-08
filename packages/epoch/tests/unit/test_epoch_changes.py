"""an epoch is its change set: a pod that misses a broadcast corrects exactly the rows it missed.

Two registries on one in-process bus and bucket stand in for a writer pod and a follower pod, with
the real generation source, the real reader and the real pass. The contract this pins:

- a lost invalidation is caught by one pass, and only the stale key is corrected: every other row
  the follower holds stays cached;
- an epoch whose change record L2 does not hold leaves the follower behind -- serving what it has,
  counted, untrusted by derived caches -- and the next pass that can read the record catches up;
- a write whose reach is the whole table invalidates the table, because that is the change;
- a follower's first sight of a generation takes it as its mark and drops nothing;
- the change log refuses a record past its cap, loudly, rather than evicting one.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, ClassVar

import pytest
from sqlalchemy import Column, DateTime, MetaData, String, Table

from threetears.core.backends.protocol import DurableStore
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import (
    WRITE_GENERATION,
    BaseCollection,
    CollectionRegistry,
    announce_unheard_writes,
)
from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient
from threetears.epoch import (
    EPOCH_CHANGES_BUCKET,
    EpochChangeRetention,
    EpochGenerationReader,
    EpochGenerationSource,
    epoch_change_kv_key,
    generation_catchup_tick,
)
from threetears.nats import set_default_namespace

_TABLE = "group_members"


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace("epochchanges")


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        _TABLE,
        metadata,
        Column("id", String(255), primary_key=True),
        Column("member_id", String(255)),
        Column("date_created", DateTime(timezone=True)),
        Column("date_updated", DateTime(timezone=True)),
    )
    return metadata


class _NoL3(DurableStore):
    """a durable L3 the registry does not wrap as SQL; the rows live in the collection's shared dict."""


class _Row(BaseEntity):
    primary_key_field = "id"


class _Members(BaseCollection[_Row]):
    """a switched-on three-tier collection over an L3 dict every pod of the test shares."""

    datetime_columns: ClassVar[frozenset[str]] = frozenset({"date_created", "date_updated"})
    write_generation = WRITE_GENERATION
    invalidation_columns: ClassVar[tuple[str, ...]] = ("member_id",)

    def __init__(self, registry: CollectionRegistry, rows: dict[str, dict[str, Any]]) -> None:
        self._rows = rows
        super().__init__(registry, DefaultCoreConfig())

    @property
    def table_name(self) -> str:
        return _TABLE

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
    """one registry with its own L1, listener, generation source and reader."""

    def __init__(self, bus: _Lossy, *, scope: str, retention: EpochChangeRetention | None = None) -> None:
        l1 = SQLiteBackend(db_name=f"epoch_changes_{uuid.uuid4().hex[:8]}")
        l1.initialize(_metadata())
        self.bus = bus
        self.registry = CollectionRegistry()
        self.registry.configure(l1_backend=l1, l2_client=bus, l3_pool=_NoL3(), kv_key_scope=scope)  # type: ignore[arg-type]
        self.source = (
            EpochGenerationSource(bus) if retention is None else EpochGenerationSource(bus, change_retention=retention)
        )
        self.registry.set_generation_source(self.source)
        self.reader = EpochGenerationReader(bus)

    async def tick(self) -> int:
        return await generation_catchup_tick(self.registry, self.reader)


@pytest.fixture
def bus() -> _Lossy:
    return _Lossy()


async def _save(collection: _Members, entity_id: str, member_id: str) -> None:
    await collection.save_entity(collection.create({"id": entity_id, "member_id": member_id}))


async def _member_id(collection: _Members, entity_id: str) -> str | None:
    row = await collection.ensure(entity_id)
    return None if row is None else str(row["member_id"])


async def _changes(bus: FakeNatsClient) -> FakeKvBucket:
    return await bus.kv_bucket(name=EPOCH_CHANGES_BUCKET)


async def _two_pods(bus: _Lossy) -> tuple[_Members, _Pod, _Members]:
    """a writer and a follower of the table, the follower listening and following, two rows cached."""
    rows: dict[str, dict[str, Any]] = {}
    writer, follower = _Pod(bus, scope="writer"), _Pod(bus, scope="follower")
    writing, reading = _Members(writer.registry, rows), _Members(follower.registry, rows)
    await follower.registry.start_invalidation_listener(follower.bus)  # type: ignore[arg-type]
    await _save(writing, "seed", "nobody")
    follower.registry.follow_generation(_TABLE)
    await follower.tick()
    await _save(writing, "m1", "ada")
    await _save(writing, "m2", "grace")
    assert await _member_id(reading, "m1") == "ada"
    assert await _member_id(reading, "m2") == "grace"
    return writing, follower, reading


class TestALostInvalidationCorrectsOnlyTheStaleKey:
    async def test_one_pass_corrects_the_missed_row_and_keeps_every_other_row(self, bus: _Lossy) -> None:
        writing, follower, reading = await _two_pods(bus)

        bus.deaf = True
        await _save(writing, "m1", "edsger")
        bus.deaf = False
        assert await _member_id(reading, "m1") == "ada", "nothing told the follower, so it serves what it cached"

        assert await follower.tick() == 1

        assert not reading.exists_in_cache_sync("m1"), "the missed row was evicted"
        assert reading.exists_in_cache_sync("m2"), "the row no epoch changed stays cached: no table drop"
        assert await _member_id(reading, "m1") == "edsger"
        assert not follower.registry.behind_on(_TABLE)
        assert await follower.tick() == 0

    async def test_the_record_names_exactly_the_rows_of_its_commit(self, bus: _Lossy) -> None:
        writing, _, _ = await _two_pods(bus)
        await _save(writing, "m1", "edsger")
        token = await EpochGenerationReader(bus).read(_TABLE)
        assert token is not None
        record = await EpochGenerationReader(bus).change(_TABLE, token)
        assert record is not None
        assert not record.whole_table
        assert record.rows == [ChangedRow(ids=["m1"], columns={"member_id": "edsger"})]

    async def test_several_missed_epochs_are_applied_in_turn(self, bus: _Lossy) -> None:
        writing, follower, reading = await _two_pods(bus)
        bus.deaf = True
        await _save(writing, "m1", "edsger")
        await _save(writing, "m2", "barbara")
        bus.deaf = False
        assert await follower.tick() == 1
        assert await _member_id(reading, "m1") == "edsger"
        assert await _member_id(reading, "m2") == "barbara"


class TestAMissingRecordLeavesThePodBehindAndServing:
    async def test_behind_until_the_record_is_back_then_caught_up(self, bus: _Lossy) -> None:
        writing, follower, reading = await _two_pods(bus)
        bus.deaf = True
        await _save(writing, "m1", "edsger")
        bus.deaf = False
        token = await EpochGenerationReader(bus).read(_TABLE)
        assert token is not None
        key = epoch_change_kv_key(_TABLE, token)
        bucket = await _changes(bus)
        held = await bucket.get(key=key)
        assert held is not None
        await bucket.delete(key=key)

        assert await follower.tick() == 0
        assert follower.registry.behind_on(_TABLE)
        assert reading.exists_in_cache_sync("m1"), "behind is never a drop: the pod serves what it holds"
        assert reading.exists_in_cache_sync("m2")
        assert await _member_id(reading, "m1") == "ada"
        follower.registry.watched_by(_TABLE, lambda: True)
        assert not follower.registry.tables_trusted([_TABLE]), "a derived cache does not trust a table it is behind on"

        await bucket.put(key=key, value=held)
        assert await follower.tick() == 1
        assert not follower.registry.behind_on(_TABLE)
        assert follower.registry.tables_trusted([_TABLE])
        assert await _member_id(reading, "m1") == "edsger"
        assert reading.exists_in_cache_sync("m2")

    async def test_a_replaced_generation_is_behind_not_a_drop(self, bus: _Lossy) -> None:
        writing, follower, reading = await _two_pods(bus)
        (await bus.kv_bucket(name="epochs")).wipe()
        await _save(writing, "m1", "edsger")
        await follower.tick()
        assert follower.registry.behind_on(_TABLE)
        assert reading.exists_in_cache_sync("m2")


class TestAWholeTableChangeIsTheChange:
    async def test_an_announced_migration_invalidates_the_table(self, bus: _Lossy) -> None:
        writing, follower, reading = await _two_pods(bus)
        await announce_unheard_writes(follower.source, [_TABLE])
        assert await follower.tick() == 1
        assert not reading.exists_in_cache_sync("m1")
        assert not reading.exists_in_cache_sync("m2")
        assert not follower.registry.behind_on(_TABLE)


class TestFirstSight:
    async def test_the_pods_own_l2_entries_are_read_through_l3_once(self, bus: _Lossy) -> None:
        rows: dict[str, dict[str, Any]] = {}
        writer, follower = _Pod(bus, scope="writer"), _Pod(bus, scope="follower")
        writing, reading = _Members(writer.registry, rows), _Members(follower.registry, rows)
        await _save(writing, "m1", "ada")
        assert await _member_id(reading, "m1") == "ada"  # now in the follower's own L2 key, which outlives it
        rows["m1"] = {**rows["m1"], "member_id": "changed-while-nobody-followed"}
        follower.registry.follow_generation(_TABLE)
        await follower.tick()
        assert not follower.registry.behind_on(_TABLE)
        assert await _member_id(reading, "m1") == "changed-while-nobody-followed"

    async def test_a_table_first_seen_with_no_generation_follows_its_first_incarnation(self, bus: _Lossy) -> None:
        rows: dict[str, dict[str, Any]] = {}
        writer, follower = _Pod(bus, scope="writer"), _Pod(bus, scope="follower")
        writing, reading = _Members(writer.registry, rows), _Members(follower.registry, rows)
        await bus.kv_bucket(name="epochs")  # the hub declared the bucket; nothing has written the table
        follower.registry.follow_generation(_TABLE)
        await follower.tick()
        await follower.registry.start_invalidation_listener(follower.bus)  # type: ignore[arg-type]
        await _save(writing, "m1", "ada")
        assert await _member_id(reading, "m1") == "ada"
        await follower.tick()
        assert not follower.registry.behind_on(_TABLE), "the first incarnation is the mark's own, not a replaced store"
        bus.deaf = True
        await _save(writing, "m1", "grace")
        bus.deaf = False
        assert await follower.tick() == 1
        assert await _member_id(reading, "m1") == "grace"


class TestAHoleHoldsOnlyTheMark:
    async def test_the_epochs_after_a_missing_record_are_still_applied(self, bus: _Lossy) -> None:
        writing, follower, reading = await _two_pods(bus)
        bus.deaf = True
        await _save(writing, "m1", "edsger")
        first = await EpochGenerationReader(bus).read(_TABLE)
        await _save(writing, "m2", "barbara")
        bus.deaf = False
        assert first is not None
        await (await _changes(bus)).delete(key=epoch_change_kv_key(_TABLE, first))
        await follower.tick()
        assert follower.registry.behind_on(_TABLE)
        assert await _member_id(reading, "m2") == "barbara", "the epoch after the hole was applied"
        assert await _member_id(reading, "m1") == "ada", "the hole's own row waits for its record"


class TestARecordStaysWithinItsBound:
    async def test_a_commit_too_large_to_record_by_row_is_recorded_as_the_whole_table(self, bus: _Lossy) -> None:
        source = EpochGenerationSource(bus)
        rows = [ChangedRow(ids=[f"row-{each:06d}"], columns={"member_id": "x" * 64}) for each in range(5000)]
        token = await source.advance_with_change(_TABLE, rows)
        record = await EpochGenerationReader(bus).change(_TABLE, token)
        assert record is not None
        assert record.whole_table
        assert record.rows == []


class TestBehindIsSaidOnce:
    async def test_a_table_that_stays_behind_is_logged_once(
        self, bus: _Lossy, caplog: pytest.LogCaptureFixture
    ) -> None:
        writing, follower, _ = await _two_pods(bus)
        bus.deaf = True
        await _save(writing, "m1", "edsger")
        bus.deaf = False
        token = await EpochGenerationReader(bus).read(_TABLE)
        assert token is not None
        await (await _changes(bus)).delete(key=epoch_change_kv_key(_TABLE, token))
        with caplog.at_level(logging.ERROR):
            for _ in range(3):
                await follower.tick()
        said = [r for r in caplog.records if r.getMessage().startswith("behind on a table")]
        assert len(said) == 1
        assert follower.registry.behind_on(_TABLE)


class TestTheLogRefusesPastItsCap:
    async def test_a_full_log_refuses_the_record_loudly(self, bus: _Lossy) -> None:
        source = EpochGenerationSource(bus, change_retention=EpochChangeRetention(max_bytes=600))
        await source.advance_with_change(_TABLE, [ChangedRow(ids=["m1"])])
        with pytest.raises(GenerationUnavailableError, match="could not be stored"):
            for each in range(20):
                await source.advance_with_change(_TABLE, [ChangedRow(ids=[f"row-{each}" * 8])])
        changes = await _changes(bus)
        assert changes.max_bytes == 600
        assert changes.storage == "memory"
        assert changes.keys(), "the records already written were kept, not evicted"
