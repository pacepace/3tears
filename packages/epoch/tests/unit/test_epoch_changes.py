"""an epoch is its change set: a pod that misses a broadcast corrects exactly the rows it missed.

Two registries on one in-process bus and bucket stand in for a writer pod and a follower pod, with
the real generation source, the real reader and the real pass. The contract this pins:

- a lost invalidation is caught by one pass, and only the stale key is corrected: every other row
  the follower holds stays cached;
- an epoch whose change record L2 does not hold leaves the follower behind -- serving what it has,
  counted, untrusted by derived caches -- and the next pass that can read the record catches up;
- a write whose reach is the whole table invalidates the table, because that is the change; a large
  commit is several keyed records, never a whole-table one;
- a log put back from the change notes carries a follower whose mark the notes cover to its start, where
  it walks every restored record and catches up; an older follower is behind (``record_lost``); and a
  lost log waits for its restore rather than starting under a new incarnation;
- a follower's first sight of a generation takes it as its mark and drops nothing;
- the change log refuses a record past its cap, loudly, rather than evicting one;
- a follower's quiet pass, reading only the change log, keeps a quiet table's mark inside retention and
  walks any table it holds an unapplied epoch of.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
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
from threetears.core.collections.change_notes import NotedChange
from threetears.core.collections.epoch_change import (
    EPOCH_CHANGE_MAX_AGE,
    ChangedRow,
    EpochChange,
    EpochWalk,
    advance_recording,
    changed_row_runs,
    walk_by_count,
)
from threetears.core.collections.generation import L2_WRITE_GENERATION
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.testing.kv import FakeChangeLog, FakeNatsClient
from threetears.epoch import (
    EPOCH_CHANGES_BUCKET,
    EpochChangeRetention,
    EpochGenerationReader,
    EpochGenerationSource,
    TableFollower,
    generation_catchup_tick,
    quiet_marks_tick,
    quiet_pass_interval,
)
from threetears.nats import set_default_namespace

_TABLE = "group_members"
_KEPT = "precincts"

#: the quiet pass runs this many times per retention window, as quiet_pass_interval documents
_PASSES_PER_RETENTION = 4


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
        #: the L3 rows every pod of the test shares, for a test that writes L3 behind the collection's back
        self.l3_rows = rows
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


class _L2Members(_Members):
    """the same table, its generation and each epoch's record kept in L2 beside its rows."""

    write_generation = L2_WRITE_GENERATION


@dataclass
class _L2Kept:
    """an L2-kept table's records, read through its collection."""

    collection: _L2Members

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        return await self.collection.l2_epoch_change(table_name, token)

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        return await walk_by_count(self, table_name, after, upto)


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


def _log(bus: FakeNatsClient) -> FakeChangeLog:
    return bus.change_log(EPOCH_CHANGES_BUCKET)


def _seq(token: str) -> int:
    return int(token.rpartition(":")[2])


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
    async def test_behind_while_aged_out_then_caught_up_once_the_log_is_restored(self, bus: _Lossy) -> None:
        writing, follower, reading = await _two_pods(bus)
        reader = EpochGenerationReader(bus)
        bus.deaf = True
        await _save(writing, "m1", "edsger")
        first = await reader.read(_TABLE)
        await _save(writing, "m2", "barbara")
        second = await reader.read(_TABLE)
        bus.deaf = False
        assert first is not None and second is not None
        records = [await reader.change(_TABLE, first), await reader.change(_TABLE, second)]
        assert all(record is not None for record in records)
        # retention takes the oldest records first; a follower whose mark is older cannot vouch for them
        _log(bus).age_out(_seq(first))

        assert await follower.tick() == 0
        assert follower.registry.behind_on(_TABLE)
        assert reading.exists_in_cache_sync("m1"), "behind is never a drop: the pod serves what it holds"
        assert await _member_id(reading, "m1") == "ada"
        assert await _member_id(reading, "m2") == "barbara", "the epoch the log still holds was applied"
        follower.registry.watched_by(_TABLE, lambda: True)
        assert follower.registry.tables_trusted([_TABLE]), "behind is said, never a cache withdrawn"

        # the broker loses the log, and the change-note keeper puts it back from the notes, which reach back
        # to the follower's mark
        mark = follower.registry.generation_marks.recorded(_TABLE)
        assert mark is not None
        _log(bus).vanish()
        notes = [_noted(_TABLE, "m1", "m2")]
        assert await EpochGenerationSource(bus).restore_from_notes(
            notes, lost=mark.rpartition(":")[0], covered_from=_seq(mark)
        )
        assert await follower.tick() == 1
        assert not follower.registry.behind_on(_TABLE)
        assert await _member_id(reading, "m1") == "edsger"

    async def test_a_replaced_generation_is_behind_not_a_drop(self, bus: _Lossy) -> None:
        writing, follower, reading = await _two_pods(bus)
        _log(bus).vanish()
        await _save(writing, "m1", "edsger")
        await follower.tick()
        assert follower.registry.behind_on(_TABLE)
        assert reading.exists_in_cache_sync("m2")


class TestAnEpochWhoseKeysAreUnknown:
    async def test_it_leaves_the_pod_behind_said_apart_from_a_lost_record(
        self, bus: _Lossy, caplog: pytest.LogCaptureFixture
    ) -> None:
        writing, follower, reading = await _two_pods(bus)
        bus.deaf = True
        await _save(writing, "m1", "edsger")
        bus.deaf = False
        # and what a broker records for a commit that named no rows
        await EpochGenerationSource(bus).advance(_TABLE)
        with caplog.at_level(logging.ERROR):
            await follower.tick()
        assert follower.registry.behind_on(_TABLE)
        assert reading.exists_in_cache_sync("m2"), "behind is never a drop"
        (said,) = [r.__dict__["extra_data"] for r in caplog.records if r.getMessage().startswith("behind on a table")]
        assert said["reason"] == "keys_unknown"

    async def test_a_writer_records_it_and_may_not_name_rows_with_it(self, bus: _Lossy) -> None:
        source = EpochGenerationSource(bus)
        token = await source.advance_with_change(_TABLE, None, keys_unknown=True)
        record = await EpochGenerationReader(bus).change(_TABLE, token)
        assert record is not None
        assert (record.keys_unknown, record.whole_table, record.rows) == (True, False, [])
        with pytest.raises(ValueError, match="rows=None"):
            await source.advance_with_change(_TABLE, [ChangedRow(ids=["m1"])], keys_unknown=True)
        assert await EpochGenerationReader(bus).read(_TABLE) == token, "a refused call advanced nothing"
        with pytest.raises(ValueError, match="rows=None"):
            await advance_recording(source, _TABLE, [ChangedRow(ids=["m1"])], keys_unknown=True)

    def test_a_record_is_exactly_one_of_its_three_forms(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            EpochChange(table=_TABLE, token="inc:1", whole_table=True, keys_unknown=True)
        with pytest.raises(ValueError, match="exactly one"):
            EpochChange(table=_TABLE, token="inc:1")


class TestAWholeTableChangeIsTheChange:
    async def test_an_announced_migration_invalidates_the_table(self, bus: _Lossy) -> None:
        writing, follower, reading = await _two_pods(bus)
        await announce_unheard_writes(follower.source, [_TABLE])
        assert await follower.tick() == 1
        assert not reading.exists_in_cache_sync("m1")
        assert not reading.exists_in_cache_sync("m2")
        assert not follower.registry.behind_on(_TABLE)


class TestFirstSight:
    async def test_the_head_is_the_mark_and_nothing_is_re_read(self, bus: _Lossy) -> None:
        rows: dict[str, dict[str, Any]] = {}
        writer, follower = _Pod(bus, scope="writer"), _Pod(bus, scope="follower")
        writing, reading = _Members(writer.registry, rows), _Members(follower.registry, rows)
        await _save(writing, "m1", "ada")
        assert await _member_id(reading, "m1") == "ada"
        head = await EpochGenerationReader(bus).read(_TABLE)
        follower.registry.follow_generation(_TABLE)
        await follower.tick()
        assert follower.registry.generation_marks.recorded(_TABLE) == head
        assert not follower.registry.behind_on(_TABLE)
        assert reading.exists_in_cache_sync("m1"), "a first sight drops and re-reads nothing"

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
        _log(bus).age_out(_seq(first))
        await follower.tick()
        assert follower.registry.behind_on(_TABLE)
        assert await _member_id(reading, "m2") == "barbara", "the epoch after the hole was applied"
        assert await _member_id(reading, "m1") == "ada", "the hole's own row waits for its record"


class TestARecordStaysWithinItsBound:
    async def test_a_commit_too_large_for_one_record_is_recorded_as_keyed_records_only(self, bus: _Lossy) -> None:
        source = EpochGenerationSource(bus)
        rows = [ChangedRow(ids=[f"row-{each:06d}"], columns={"member_id": "x" * 64}) for each in range(5000)]
        before = await source.current(_TABLE)
        token = await source.advance_with_change(_TABLE, rows)
        walk = await EpochGenerationReader(bus).changes_after(_TABLE, before, token)
        assert walk.complete
        assert len(walk.changes) > 1, "one record could not hold them all"
        assert not any(change.whole_table or change.keys_unknown for change in walk.changes)
        assert [row for change in walk.changes for row in change.rows] == rows, "every row, in order, by key"
        assert walk.changes[-1].token == token, "the write's epoch is its last record"

    async def test_a_pod_that_missed_it_applies_the_keys_and_drops_nothing(self, bus: _Lossy) -> None:
        writing, follower, reading = await _two_pods(bus)
        rows = [ChangedRow(ids=[f"row-{each:06d}"], columns={"member_id": "x" * 64}) for each in range(5000)]
        rows.append(ChangedRow(ids=["m1"], columns={"member_id": "edsger"}))
        writing.l3_rows["m1"]["member_id"] = "edsger"  # the shared L3 the commit changed
        await advance_recording(EpochGenerationSource(bus), _TABLE, rows)
        assert await _member_id(reading, "m1") == "ada", "nothing told the follower"

        await follower.tick()

        assert not follower.registry.behind_on(_TABLE)
        assert not reading.exists_in_cache_sync("m1"), "the row the commit named was evicted"
        assert reading.exists_in_cache_sync("m2"), "the row it did not name stays: no table drop"
        assert await _member_id(reading, "m1") == "edsger"

    async def test_a_table_kept_in_l2_records_a_large_commit_as_keyed_records_too(self, bus: _Lossy) -> None:
        rows: dict[str, dict[str, Any]] = {}
        writer, follower = _Pod(bus, scope="shared"), _Pod(bus, scope="shared")
        writing, reading = _L2Members(writer.registry, rows), _L2Members(follower.registry, rows)
        await _save(writing, "m1", "ada")
        await _save(writing, "m2", "grace")
        follower.registry.follow_generation(_TABLE)
        await follower.registry.catch_up_l2_generations()
        assert await _member_id(reading, "m1") == "ada"
        assert await _member_id(reading, "m2") == "grace"
        before = await writing.l2_write_generation()
        assert before is not None

        bus.deaf = True
        ids = [f"row-{each:06d}" for each in range(5000)] + ["m1"]
        async with writing.bypassing_write(*ids):
            rows["m1"]["member_id"] = "edsger"
        bus.deaf = False

        head = await writing.l2_write_generation()
        assert head is not None
        walk = await walk_by_count(_L2Kept(writing), _TABLE, before, head)
        assert walk.complete and len(walk.changes) > 1
        assert not any(change.whole_table for change in walk.changes), "keyed records only"
        await follower.registry.catch_up_l2_generations()
        assert not follower.registry.behind_on(_TABLE)
        assert reading.exists_in_cache_sync("m2"), "the row it did not name stays: no table drop"
        assert await _member_id(reading, "m1") == "edsger"

    def test_one_row_past_the_bound_is_refused_loudly(self) -> None:
        with pytest.raises(GenerationUnavailableError, match="record bound"):
            changed_row_runs(_TABLE, [ChangedRow(ids=["x" * (300 * 1024)])])


class TestAGapPastRetention:
    async def test_a_quiet_tables_follower_that_judges_keeps_its_mark_inside_retention(self, bus: _Lossy) -> None:
        _, follower, _ = await _two_pods(bus)
        source = EpochGenerationSource(bus)
        await follower.tick()
        for each in range(3):
            await source.advance_with_change(_KEPT, [ChangedRow(ids=[f"k{each}"])])
            await follower.tick()
        state = await _log(bus).state()
        assert state is not None
        _log(bus).age_out(state.last_seq - 1)  # every record of the quiet table is gone

        await follower.tick()
        assert not follower.registry.behind_on(_TABLE), "its mark moved with the log while nothing was written"

    async def test_a_follower_away_past_retention_is_behind_never_current(
        self, bus: _Lossy, caplog: pytest.LogCaptureFixture
    ) -> None:
        _, follower, reading = await _two_pods(bus)
        source = EpochGenerationSource(bus)
        for each in range(3):  # the follower is away: it judges nothing meanwhile
            await source.advance_with_change(_KEPT, [ChangedRow(ids=[f"k{each}"])])
        state = await _log(bus).state()
        assert state is not None
        _log(bus).age_out(state.last_seq - 1)  # the table's whole history aged out
        assert await EpochGenerationReader(bus).read(_TABLE) == f"{state.incarnation}:0"

        with caplog.at_level(logging.ERROR):
            await follower.tick()
        assert follower.registry.behind_on(_TABLE), "a gap past retention is behind, never current"
        said = [r.__dict__["extra_data"] for r in caplog.records if r.getMessage().startswith("behind on a table")]
        assert [each["reason"] for each in said] == ["aged_out"]
        assert reading.exists_in_cache_sync("m1"), "behind serves what it holds and drops nothing"


def _noted(table: str, *ids: str, columns: dict[str, str | None] | None = None) -> NotedChange:
    """what the change notes of ``table`` say changed: these rows."""
    return NotedChange(table=table, rows=tuple(ChangedRow(ids=[each], columns=columns) for each in ids))


class TestARestoreFromNotesCarriesACoveredFollowerOver:
    async def test_a_follower_at_or_past_the_floor_walks_the_restored_log_and_catches_up(self, bus: _Lossy) -> None:
        writing, follower, reading = await _two_pods(bus)
        reader = EpochGenerationReader(bus)
        floor = follower.registry.generation_marks.recorded(_TABLE)
        assert floor is not None
        lost = floor.rpartition(":")[0]
        bus.deaf = True
        await _save(writing, "m1", "edsger")  # missed by the follower; its note is in L3 with the row
        bus.deaf = False

        _log(bus).vanish()
        restored = await EpochGenerationSource(bus).restore_from_notes(
            [_noted(_TABLE, "m1", columns={"member_id": "edsger"})], lost=lost, covered_from=_seq(floor)
        )
        assert restored is not None and restored != lost
        assert await reader.continues(restored) == (lost, _seq(floor))

        await follower.tick()
        assert not follower.registry.behind_on(_TABLE), "covered: nothing is lost, so nothing is said lost"
        assert (follower.registry.generation_marks.recorded(_TABLE) or "").startswith(f"{restored}:")
        assert await _member_id(reading, "m1") == "edsger", "the missed epoch was applied from the restored log"
        assert reading.exists_in_cache_sync("m2"), "a row no note names stays cached: no table drop"

    async def test_a_non_l2_first_tables_records_come_back_beside_others(self, bus: _Lossy) -> None:
        writing, follower, reading = await _two_pods(bus)
        floor = follower.registry.generation_marks.recorded(_TABLE)
        assert floor is not None
        source = EpochGenerationSource(bus)
        await source.advance_with_change(_KEPT, [ChangedRow(ids=["k0"])])
        bus.deaf = True
        await _save(writing, "m1", "edsger")
        bus.deaf = False
        await source.advance_with_change(_KEPT, [ChangedRow(ids=["k1"])])

        _log(bus).vanish()
        notes = [_noted(_KEPT, "k0", "k1"), _noted(_TABLE, "m1", columns={"member_id": "edsger"})]
        assert await source.restore_from_notes(notes, lost=floor.rpartition(":")[0], covered_from=_seq(floor))
        await follower.tick()
        assert not follower.registry.behind_on(_TABLE)
        assert await _member_id(reading, "m1") == "edsger"

    async def test_a_follower_older_than_the_floor_is_behind_as_lost_and_drops_nothing(
        self, bus: _Lossy, caplog: pytest.LogCaptureFixture
    ) -> None:
        writing, follower, reading = await _two_pods(bus)
        floor = follower.registry.generation_marks.recorded(_TABLE)
        assert floor is not None
        await EpochGenerationSource(bus).advance_with_change(_KEPT, [ChangedRow(ids=["k0"])])
        bus.deaf = True
        await _save(writing, "m1", "edsger")
        bus.deaf = False
        head = await EpochGenerationReader(bus).read(_TABLE)
        assert head is not None

        _log(bus).vanish()
        # the notes reach back only to an epoch after the follower's mark
        assert await EpochGenerationSource(bus).restore_from_notes(
            [_noted(_TABLE, "m1")], lost=floor.rpartition(":")[0], covered_from=_seq(head)
        )
        with caplog.at_level(logging.ERROR):
            await follower.tick()
            await follower.tick()
        assert follower.registry.behind_on(_TABLE), "older than the notes reach: said, never read as current"
        said = [r.__dict__["extra_data"] for r in caplog.records if r.getMessage().startswith("behind on a table")]
        assert [each["reason"] for each in said] == ["record_lost"], "said once, as lost"
        assert await _member_id(reading, "m1") == "ada", "behind serves what it holds"
        assert reading.exists_in_cache_sync("m2"), "and drops nothing"

    async def test_a_restore_whose_notes_cover_nothing_leaves_every_follower_behind(self, bus: _Lossy) -> None:
        _, follower, reading = await _two_pods(bus)
        floor = follower.registry.generation_marks.recorded(_TABLE)
        assert floor is not None
        _log(bus).vanish()
        assert await EpochGenerationSource(bus).restore_from_notes([], lost=floor.rpartition(":")[0], covered_from=None)
        await follower.tick()
        assert follower.registry.behind_on(_TABLE)
        assert reading.exists_in_cache_sync("m2")


class TestALostLogWaitsForItsRestore:
    async def test_a_write_after_the_loss_never_starts_a_log_ahead_of_the_restore(self, bus: _Lossy) -> None:
        source = EpochGenerationSource(bus, restored_from_l3=True, restore_wait=timedelta(0))
        incarnation = await source.start_log()
        written = await source.advance_with_change(_TABLE, [ChangedRow(ids=["m1"])])

        _log(bus).vanish()
        with pytest.raises(GenerationUnavailableError, match="restore"):
            await source.advance_with_change(_TABLE, [ChangedRow(ids=["m2"])])
        with pytest.raises(GenerationUnavailableError, match="restore"):
            await source.current(_TABLE)
        assert await _log(bus).state() is None, "no log was started under a new incarnation"

        assert await source.restore_from_notes([_noted(_TABLE, "m1")], lost=incarnation, covered_from=0)
        after = await source.advance_with_change(_TABLE, [ChangedRow(ids=["m3"])])
        assert await source.continues(after.rpartition(":")[0]) == (incarnation, 0), (
            "the restored log continues the lost one"
        )
        assert after.rpartition(":")[0] != written.rpartition(":")[0]

    async def test_a_write_waits_for_the_restore_and_lands_in_the_restored_log(self, bus: _Lossy) -> None:
        source = EpochGenerationSource(bus, restored_from_l3=True, restore_wait=timedelta(seconds=5))
        incarnation = await source.start_log()
        await source.advance_with_change(_TABLE, [ChangedRow(ids=["m1"])])
        _log(bus).vanish()

        async def restore() -> str | None:
            await asyncio.sleep(0.2)
            return await EpochGenerationSource(bus).restore_from_notes(
                [_noted(_TABLE, "m1")], lost=incarnation, covered_from=0
            )

        written, restored = await asyncio.gather(
            source.advance_with_change(_TABLE, [ChangedRow(ids=["m2"])]), restore()
        )
        assert restored is not None, "the restore ran first: the waiting write started no log"
        assert written.rpartition(":")[0] == restored
        assert await source.continues(restored) == (incarnation, 0)


class TestTheQuietPass:
    async def test_a_quiet_table_stays_current_across_more_than_one_retention_window(self, bus: _Lossy) -> None:
        _, follower, reading = await _two_pods(bus)
        source = EpochGenerationSource(bus)
        held_from = (await _log(bus).state()).first_seq  # type: ignore[union-attr]
        # three retention windows, each a pass interval at a time: other tables are written, the follower's
        # quiet pass runs, and retention takes what the window has passed -- the quiet table's history first
        for each in range(3 * _PASSES_PER_RETENTION):
            await source.advance_with_change(_KEPT, [ChangedRow(ids=[f"k{each}"])])
            done = await quiet_marks_tick(follower.registry, follower.reader)
            assert done.moved == (_TABLE,) and done.walked == ()
            state = await _log(bus).state()
            assert state is not None
            _log(bus).age_out(state.last_seq - 1)
        state = await _log(bus).state()
        assert state is not None and state.first_seq > held_from, "the quiet table's every record aged out"

        await follower.tick()
        assert not follower.registry.behind_on(_TABLE), "never judged aged_out for a stretch nothing happened in"
        assert reading.exists_in_cache_sync("m1"), "and nothing was dropped"

    async def test_a_table_with_a_pending_record_is_never_skipped(self, bus: _Lossy) -> None:
        writing, follower, reading = await _two_pods(bus)
        bus.deaf = True
        await _save(writing, "m1", "edsger")
        bus.deaf = False
        await EpochGenerationSource(bus).advance_with_change(_KEPT, [ChangedRow(ids=["k0"])])
        missed = await EpochGenerationReader(bus).read(_TABLE)

        done = await quiet_marks_tick(follower.registry, follower.reader)
        assert done.walked == (_TABLE,) and done.moved == ()
        assert not follower.registry.behind_on(_TABLE)
        assert not reading.exists_in_cache_sync("m1"), "the unapplied epoch was walked and applied"
        assert await _member_id(reading, "m1") == "edsger"
        state = await _log(bus).state()
        assert state is not None
        assert follower.registry.generation_marks.recorded(_TABLE) == missed, "the mark is the applied epoch"

    async def test_the_pass_reads_only_the_change_log_never_l3(
        self, bus: _Lossy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        writing, follower, reading = await _two_pods(bus)
        bus.deaf = True
        await _save(writing, "m2", "barbara")
        bus.deaf = False
        await EpochGenerationSource(bus).advance_with_change(_KEPT, [ChangedRow(ids=["k0"])])

        async def no_l3(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("the quiet pass read L3")

        monkeypatch.setattr(reading, "fetch_from_store", no_l3)
        monkeypatch.setattr(reading, "save_to_store", no_l3)
        monkeypatch.setattr(reading, "delete_from_store", no_l3)
        done = await quiet_marks_tick(follower.registry, follower.reader)
        assert done.walked == (_TABLE,)

    async def test_the_interval_is_a_fraction_of_the_logs_own_retention(self, bus: _Lossy) -> None:
        await EpochGenerationSource(bus, change_retention=EpochChangeRetention(max_age=timedelta(hours=1))).start_log()
        state, _ = await EpochGenerationReader(bus).newest_epochs(())
        assert quiet_pass_interval(state) == timedelta(hours=1) / _PASSES_PER_RETENTION
        assert quiet_pass_interval(None) == EPOCH_CHANGE_MAX_AGE / _PASSES_PER_RETENTION

    async def test_a_table_follower_runs_the_pass_for_its_tables(self, bus: _Lossy) -> None:
        retention = EpochChangeRetention(max_age=timedelta(milliseconds=400))
        await EpochGenerationSource(bus, change_retention=retention).start_log()
        _, follower, _ = await _two_pods(bus)
        watcher = TableFollower(follower.registry, follower.reader, [_TABLE])
        watcher.start()
        try:
            head = await EpochGenerationSource(bus).advance_with_change(_KEPT, [ChangedRow(ids=["k0"])])
            for _ in range(100):
                if follower.registry.generation_marks.recorded(_TABLE) == head:
                    break
                await asyncio.sleep(0.02)
            assert follower.registry.generation_marks.recorded(_TABLE) == head, (
                "the pass moved the quiet table's mark to the log's head"
            )
        finally:
            await watcher.stop()


class TestBehindIsSaidOnce:
    async def test_a_table_that_stays_behind_is_logged_once(
        self, bus: _Lossy, caplog: pytest.LogCaptureFixture
    ) -> None:
        writing, follower, _ = await _two_pods(bus)
        bus.deaf = True
        await _save(writing, "m1", "edsger")
        token = await EpochGenerationReader(bus).read(_TABLE)
        await _save(writing, "m2", "barbara")
        bus.deaf = False
        assert token is not None
        _log(bus).age_out(_seq(token))
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
        with pytest.raises(GenerationUnavailableError, match="could not be appended"):
            for each in range(20):
                await source.advance_with_change(_TABLE, [ChangedRow(ids=[f"row-{each}" * 8])])
        assert _log(bus).max_bytes == 600
        state = await _log(bus).state()
        assert state is not None and state.first_seq == 1, "the records already written were kept, not evicted"
