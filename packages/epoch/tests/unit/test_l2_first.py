"""an L2-first collection commits in NATS, and a saver makes L3 match it.

Pods on one in-process bus and bucket, all of one principal (one L2 scope), stand in for a writer pod,
a follower pod and the one claimed worker that runs the saver and the refiller. L3 is a dict the test
can make refuse a row, as a unique constraint would. The contract this pins:

- the commit is the row in L2 plus its epoch record in the change log; a writer that dies after it
  loses nothing, because the saver carries the commit to L3 later, and saving twice is harmless;
- a caller that waits for L3 is told when L3 refuses the row, and L2 no longer holds it;
- a collection that does not wait has an L3 refusal counted and logged at ERROR, and the saver moves on;
- when NATS loses everything, the refiller puts rows, epoch records and the generation back from L3;
  until then a pod serves what it holds and is behind, and afterwards it catches up through the normal
  epoch path without reading L3;
- without L2 the same write goes straight to L3.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, ClassVar

import pytest
from sqlalchemy import Column, DateTime, MetaData, String, Table

from threetears.core.backends.protocol import DurableStore
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import NO_L2, WRITE_GENERATION, BaseCollection, CollectionRegistry
from threetears.core.collections.epoch_change import ChangedRow, EpochChange
from threetears.core.collections.l2_first import (
    L2_FIRST,
    L2First,
    L2LostError,
    L2Refiller,
    L3Saver,
    l2_first_status,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.kv import FakeNatsClient
from threetears.epoch import (
    EPOCH_CHANGES_BUCKET,
    EPOCH_CHANGES_FILL_WARN_RATIO,
    EpochChangeRetention,
    EpochGenerationReader,
    EpochGenerationSource,
    generation_catchup_tick,
)
from threetears.nats import set_default_namespace

_TABLE = "results"
_FEED = "results_feed"
_SCOPE = "fleet"


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace("l2first")


def _metadata(table: str) -> MetaData:
    metadata = MetaData()
    Table(
        table,
        metadata,
        Column("id", String(255), primary_key=True),
        Column("votes", String(255)),
        Column("date_created", DateTime(timezone=True)),
        Column("date_updated", DateTime(timezone=True)),
    )
    return metadata


class _NoL3(DurableStore):
    """a durable L3 the registry does not wrap as SQL; the rows live in the test's dict."""


class _UniqueViolation(Exception):
    """what L3 raises for a row its constraint refuses, with the SQLSTATE a driver reports."""

    sqlstate = "23505"


class _Crash(Exception):
    """a process dying part-way through a write, as the saver sees it."""


class _L3:
    """one table's L3: its rows, the rows it refuses, and how often it was read."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.refused: set[str] = set()
        self.crash_on: set[str] = set()
        self.lands_then_drops: set[str] = set()
        self.writes = 0


class _Row(BaseEntity):
    primary_key_field = "id"


class _Results(BaseCollection[_Row]):
    """an L2-first collection over an L3 dict every pod of the test shares."""

    datetime_columns: ClassVar[frozenset[str]] = frozenset({"date_created", "date_updated"})
    write_generation = WRITE_GENERATION
    l2_first = L2_FIRST
    _table: ClassVar[str] = _TABLE

    def __init__(self, registry: CollectionRegistry, l3: _L3, **kwargs: Any) -> None:
        self._l3 = l3
        self.l3_reads = 0
        super().__init__(registry, DefaultCoreConfig(), **kwargs)

    @property
    def table_name(self) -> str:
        return self._table

    @property
    def entity_class(self) -> type[_Row]:
        return _Row

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        self.l3_reads += 1
        row = self._l3.rows.get(str(entity_id))
        return dict(row) if row is not None else None

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        key = str(data["id"])
        if key in self._l3.crash_on:
            self._l3.crash_on.discard(key)
            raise _Crash(key)
        if key in self._l3.refused:
            raise _UniqueViolation(f"duplicate key value violates unique constraint ({key})")
        self._l3.rows[key] = dict(data)
        self._l3.writes += 1
        if key in self._l3.lands_then_drops:
            self._l3.lands_then_drops.discard(key)
            raise ConnectionResetError("the connection dropped before the commit was acknowledged")
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        self._l3.rows.pop(str(entity_id), None)

    def l3_pages(self, *, page_size: int) -> AsyncIterator[list[dict[str, Any]]]:
        rows = [dict(row) for row in self._l3.rows.values()]

        async def pages() -> AsyncIterator[list[dict[str, Any]]]:
            for start in range(0, len(rows), page_size):
                yield rows[start : start + page_size]

        return pages()

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        row: dict[str, Any] = json.loads(data)
        return row


class _Feed(_Results):
    """an L2-first collection whose loader can send its rows again, so its writes do not wait for L3."""

    l2_first = L2First(wait_for_l3=False, refetchable_from="the loader reads every row again from its source")
    _table: ClassVar[str] = _FEED


# parity-with: threetears.core.collections.l2_first.EpochLogStore
class FakeEpochLog:
    """the L3 epoch log, kept in a dict."""

    def __init__(self) -> None:
        self.records: dict[tuple[str, str, int], EpochChange] = {}
        self.order: list[tuple[str, str, int]] = []

    async def save(self, change: EpochChange) -> None:
        incarnation, _, count = change.token.rpartition(":")
        key = (change.table, incarnation, int(count))
        if key not in self.records:
            self.records[key] = change
            self.order.append(key)

    async def head(self, table_name: str) -> str | None:
        mine = [key for key in self.order if key[0] == table_name]
        if not mine:
            return None
        incarnation = mine[-1][1]
        count = max(key[2] for key in mine if key[1] == incarnation)
        return f"{incarnation}:{count}"

    async def since(self, table_name: str, token: str, *, limit: int) -> list[EpochChange]:
        incarnation, _, count = token.rpartition(":")
        found = sorted(
            (key for key in self.records if key[0] == table_name and key[1] == incarnation and key[2] > int(count)),
            key=lambda key: key[2],
        )
        return [self.records[key] for key in found[:limit]]


class _Pod:
    """one registry of the principal, with its own L1, generation source and reader."""

    def __init__(
        self, bus: FakeNatsClient, log: FakeEpochLog, *, retention: EpochChangeRetention | None = None
    ) -> None:
        l1 = SQLiteBackend(db_name=f"l2_first_{uuid.uuid4().hex[:8]}")
        l1.initialize(_metadata(_TABLE))
        l1.initialize(_metadata(_FEED))
        self.bus = bus
        self.registry = CollectionRegistry()
        self.registry.configure(l1_backend=l1, l2_client=bus, l3_pool=_NoL3(), kv_key_scope=_SCOPE)  # type: ignore[arg-type]
        self.source = (
            EpochGenerationSource(bus) if retention is None else EpochGenerationSource(bus, change_retention=retention)
        )
        self.registry.set_generation_source(self.source)
        self.registry.set_epoch_log(log)
        self.reader = EpochGenerationReader(bus)

    async def tick(self) -> int:
        return await generation_catchup_tick(self.registry, self.reader)


class _Fleet:
    """a writer, a follower and the claimed worker, over one L3 and one epoch log."""

    def __init__(self, bus: FakeNatsClient) -> None:
        self.bus = bus
        self.l3 = _L3()
        self.feed_l3 = _L3()
        self.log = FakeEpochLog()
        self.writer, self.follower, self.worker = _Pod(bus, self.log), _Pod(bus, self.log), _Pod(bus, self.log)
        self.writing = _Results(self.writer.registry, self.l3)
        self.reading = _Results(self.follower.registry, self.l3)
        self.feeding = _Feed(self.writer.registry, self.feed_l3)
        self.feed_read = _Feed(self.follower.registry, self.feed_l3)
        self.working = _Results(self.worker.registry, self.l3)
        self.feed_work = _Feed(self.worker.registry, self.feed_l3)
        self.saver = L3Saver(self.worker.registry, epochs=self.worker.source, log=self.log)
        self.refiller = L2Refiller(self.worker.registry, epochs=self.worker.source, log=self.log)

    async def start(self) -> None:
        """seed L2 from the empty L3 (the refiller's first sight) and start the follower following."""
        await self.refiller.run_pass()
        await self.follower.registry.start_invalidation_listener(self.bus)  # type: ignore[arg-type]
        self.follower.registry.follow_generation(_TABLE)
        self.follower.registry.follow_generation(_FEED)
        await self.follower.tick()

    async def lose_nats(self) -> None:
        """every memory bucket loses every entry, as a full broker restart does."""
        for name in ("collections", "epochs", EPOCH_CHANGES_BUCKET):
            (await self.bus.kv_bucket(name=name)).wipe()


async def _save(collection: _Results, entity_id: str, votes: str) -> None:
    await collection.save_entity(collection.create({"id": entity_id, "votes": votes}))


async def _votes(collection: _Results, entity_id: str) -> str | None:
    row = await collection.ensure(entity_id)
    return None if row is None else str(row["votes"])


@pytest.fixture
async def fleet() -> _Fleet:
    started = _Fleet(FakeNatsClient())
    await started.start()
    return started


class TestTheCommitIsInNats:
    async def test_a_waiting_save_is_in_l2_l3_and_the_l3_epoch_log_when_it_returns(self, fleet: _Fleet) -> None:
        await _save(fleet.writing, "precinct-1", "10")

        assert fleet.l3.rows["precinct-1"]["votes"] == "10"
        token = await fleet.writer.reader.read(_TABLE)
        assert token is not None
        assert await fleet.log.head(_TABLE) == token, "the epoch record reached the L3 epoch log too"
        record = await fleet.writer.reader.change(_TABLE, token)
        assert record is not None
        assert [row.ids for row in record.rows] == [["precinct-1"]]
        assert record.rows[0].l3_pending

    async def test_the_follower_reads_the_row_from_l2_never_from_l3(self, fleet: _Fleet) -> None:
        await _save(fleet.writing, "precinct-1", "10")
        assert await _votes(fleet.reading, "precinct-1") == "10"
        assert await _votes(fleet.reading, "precinct-404") is None, "L2 is complete, so a miss is an absence"
        assert fleet.reading.l3_reads == 0

    async def test_the_saver_does_not_save_a_settled_commit_again(self, fleet: _Fleet) -> None:
        await _save(fleet.writing, "precinct-1", "10")
        writes = fleet.l3.writes
        result = await fleet.saver.run_pass()
        assert result.settled == 1
        assert result.saved == 0
        assert fleet.l3.writes == writes


class TestACrashBetweenTheCommitAndTheL3Save:
    async def test_the_saver_carries_a_commit_nobody_saved(self, fleet: _Fleet) -> None:
        await _save(fleet.feeding, "precinct-1", "10")
        await _save(fleet.feeding, "precinct-2", "20")
        assert fleet.feed_l3.rows == {}, "a no-wait write returns at the commit, before L3 has it"

        result = await fleet.saver.run_pass()

        assert result.saved == 2
        assert fleet.feed_l3.rows["precinct-1"]["votes"] == "10"
        assert fleet.feed_l3.rows["precinct-2"]["votes"] == "20"
        token = await fleet.writer.reader.read(_FEED)
        assert await fleet.log.head(_FEED) == token

    async def test_a_saver_that_dies_mid_pass_is_resumed_and_saving_twice_is_harmless(self, fleet: _Fleet) -> None:
        await _save(fleet.feeding, "precinct-1", "10")
        await _save(fleet.feeding, "precinct-2", "20")
        fleet.feed_l3.crash_on.add("precinct-2")
        with pytest.raises(_Crash):
            await fleet.saver.run_pass()
        assert fleet.feed_l3.rows["precinct-1"]["votes"] == "10"
        assert "precinct-2" not in fleet.feed_l3.rows

        restarted = L3Saver(fleet.worker.registry, epochs=fleet.worker.source, log=fleet.log)
        result = await restarted.run_pass()

        assert result.saved == 1, "the saver resumed after the last epoch it finished"
        assert fleet.feed_l3.rows["precinct-2"]["votes"] == "20"
        again = L3Saver(fleet.worker.registry, epochs=fleet.worker.source, log=fleet.log)
        before = dict(fleet.feed_l3.rows)
        await again.run_pass()
        assert fleet.feed_l3.rows == before

    async def test_the_newest_value_wins_whatever_epoch_the_saver_reaches_first(self, fleet: _Fleet) -> None:
        await _save(fleet.feeding, "precinct-1", "10")
        await _save(fleet.feeding, "precinct-1", "11")
        await fleet.saver.run_pass()
        assert fleet.feed_l3.rows["precinct-1"]["votes"] == "11"

    async def test_a_delete_reaches_l3(self, fleet: _Fleet) -> None:
        await _save(fleet.feeding, "precinct-1", "10")
        await fleet.saver.run_pass()
        await fleet.feeding.delete("precinct-1")
        assert "precinct-1" in fleet.feed_l3.rows
        await fleet.saver.run_pass()
        assert "precinct-1" not in fleet.feed_l3.rows
        assert await _votes(fleet.feed_read, "precinct-1") is None
        assert fleet.feed_read.l3_reads == 0


class TestAWaitingCallerSeesAnL3Refusal:
    async def test_the_error_reaches_the_caller_and_l2_is_withdrawn(self, fleet: _Fleet) -> None:
        await _save(fleet.writing, "precinct-1", "10")
        assert await _votes(fleet.reading, "precinct-1") == "10"
        fleet.l3.refused.add("precinct-1")

        with pytest.raises(_UniqueViolation):
            await _save(fleet.writing, "precinct-1", "99")

        assert fleet.l3.rows["precinct-1"]["votes"] == "10"
        assert await _votes(fleet.reading, "precinct-1") == "10", "the follower never serves the refused value"
        assert await _votes(fleet.working, "precinct-1") == "10", "L2 holds the value L3 has"

    async def test_a_save_whose_outcome_is_unknown_converges_on_l2(self, fleet: _Fleet) -> None:
        await _save(fleet.writing, "precinct-1", "10")
        fleet.l3.lands_then_drops.add("precinct-1")
        with pytest.raises(ConnectionResetError):
            await _save(fleet.writing, "precinct-1", "99")
        assert fleet.l3.rows["precinct-1"]["votes"] == "99", "L3 took it, though the caller was told it failed"
        assert await _votes(fleet.reading, "precinct-1") == "10"
        await fleet.saver.run_pass()
        assert fleet.l3.rows["precinct-1"]["votes"] == "10", "the saver made L3 hold what L2 holds"

    async def test_a_refused_new_row_leaves_no_row(self, fleet: _Fleet) -> None:
        fleet.l3.refused.add("precinct-9")
        with pytest.raises(_UniqueViolation):
            await _save(fleet.writing, "precinct-9", "1")
        assert await _votes(fleet.reading, "precinct-9") is None
        result = await fleet.saver.run_pass()
        assert result.rejected == 0, "the writer settled its own refusal; the saver has nothing to alarm on"


class TestANoWaitRefusalIsLoud:
    async def test_counted_logged_and_never_wedging_the_saver(
        self, fleet: _Fleet, caplog: pytest.LogCaptureFixture
    ) -> None:
        fleet.feed_l3.refused.add("precinct-1")
        await _save(fleet.feeding, "precinct-1", "10")
        await _save(fleet.feeding, "precinct-2", "20")
        with caplog.at_level(logging.ERROR):
            result = await fleet.saver.run_pass()
        assert result.rejected == 1
        assert result.saved == 1
        assert fleet.feed_l3.rows["precinct-2"]["votes"] == "20"
        assert any("L3 refused" in record.getMessage() for record in caplog.records if record.levelno == logging.ERROR)
        assert (await fleet.saver.run_pass()).rejected == 0, "the saver moved past it"


class TestL2Lost:
    async def test_the_refiller_restores_and_the_pod_catches_up_without_reading_l3(self, fleet: _Fleet) -> None:
        await _save(fleet.writing, "precinct-1", "10")
        await _save(fleet.writing, "precinct-2", "20")
        assert await _votes(fleet.reading, "precinct-1") == "10"
        assert await _votes(fleet.reading, "precinct-2") == "20"
        await _save(fleet.writing, "precinct-3", "30")

        await fleet.lose_nats()

        assert await _votes(fleet.reading, "precinct-1") == "10", "a pod serves what it holds"
        with pytest.raises(L2LostError):
            await fleet.reading.ensure("precinct-3")
        await fleet.follower.tick()
        assert fleet.follower.registry.behind_on(_TABLE), "and reports itself behind"
        with pytest.raises(L2LostError):
            await _save(fleet.writing, "precinct-4", "40")

        assert await fleet.refiller.run_pass() >= 1

        await fleet.follower.tick()
        assert not fleet.follower.registry.behind_on(_TABLE)
        assert await _votes(fleet.reading, "precinct-3") == "30"
        assert await _votes(fleet.reading, "precinct-2") == "20"
        assert fleet.reading.l3_reads == 0, "no pod read L3"
        await _save(fleet.writing, "precinct-4", "40")
        assert await _votes(fleet.reading, "precinct-4") == "40"

    async def test_a_missed_epoch_is_restored_and_applied_through_the_normal_path(self, fleet: _Fleet) -> None:
        await _save(fleet.writing, "precinct-1", "10")
        assert await _votes(fleet.reading, "precinct-1") == "10"
        await fleet.follower.registry.stop_invalidation_listener()
        await _save(fleet.writing, "precinct-1", "11")
        await fleet.lose_nats()

        await fleet.refiller.run_pass()
        await fleet.follower.tick()

        assert await _votes(fleet.reading, "precinct-1") == "11"
        assert fleet.reading.l3_reads == 0


class TestWithoutL2:
    async def test_the_same_api_writes_straight_to_l3(self) -> None:
        l1 = SQLiteBackend(db_name=f"l2_first_{uuid.uuid4().hex[:8]}")
        l1.initialize(_metadata(_TABLE))
        registry = CollectionRegistry()
        registry.configure(l1_backend=l1, l3_pool=_NoL3())  # type: ignore[arg-type]
        l3 = _L3()
        collection = _Results(registry, l3, nats_client=NO_L2)
        await _save(collection, "precinct-1", "10")
        assert l3.rows["precinct-1"]["votes"] == "10"
        assert await _votes(collection, "precinct-1") == "10"


class TestTheDeclaration:
    def test_a_collection_that_does_not_wait_says_where_its_data_comes_from(self) -> None:
        with pytest.raises(ValueError, match="refetchable_from"):
            L2First(wait_for_l3=False)

    def test_waiting_is_the_default(self) -> None:
        assert L2_FIRST.wait_for_l3


class TestTheChangeLogWarnsBeforeItFills:
    async def test_the_fill_ratio_is_measured_and_warned_on(self, caplog: pytest.LogCaptureFixture) -> None:
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus, change_retention=EpochChangeRetention(max_bytes=3200))
        for each in range(11):
            await source.advance_with_change(_TABLE, [ChangedRow(ids=[f"row-{each:04d}"])])
        with caplog.at_level(logging.WARNING):
            ratio = await source.observe_change_log_fill()
        assert ratio > EPOCH_CHANGES_FILL_WARN_RATIO
        assert any("epoch change log" in record.getMessage() for record in caplog.records)


class TestTheOperatorSeesWhereEachTableStands:
    async def test_lost_then_refilled_then_saved(self, fleet: _Fleet) -> None:
        await _save(fleet.feeding, "precinct-1", "10")
        statuses = {
            s.table: s for s in await l2_first_status(fleet.worker.registry, epochs=fleet.worker.source, log=fleet.log)
        }
        assert statuses[_FEED].complete
        assert statuses[_FEED].unsaved_epochs == 1, "the no-wait write waits for the saver"
        await fleet.saver.run_pass()
        statuses = {
            s.table: s for s in await l2_first_status(fleet.worker.registry, epochs=fleet.worker.source, log=fleet.log)
        }
        assert statuses[_FEED].unsaved_epochs == 0
        await fleet.lose_nats()
        statuses = {
            s.table: s for s in await l2_first_status(fleet.worker.registry, epochs=fleet.worker.source, log=fleet.log)
        }
        assert statuses[_FEED].generation is None
        assert not statuses[_FEED].complete
        await fleet.refiller.run_pass()
        statuses = {
            s.table: s for s in await l2_first_status(fleet.worker.registry, epochs=fleet.worker.source, log=fleet.log)
        }
        assert statuses[_FEED].complete
        assert statuses[_FEED].generation == statuses[_FEED].l3_logged


class TestEveryReadAndWritePathStaysOnL2:
    async def test_a_reload_reads_l2_never_l3(self, fleet: _Fleet) -> None:
        await _save(fleet.writing, "precinct-1", "10")
        entity = await fleet.reading.get("precinct-1")
        assert entity is not None
        await _save(fleet.writing, "precinct-1", "12")
        await fleet.reading.reload_entity(entity)
        assert entity.to_dict()["votes"] == "12"
        assert fleet.reading.l3_reads == 0

    async def test_a_write_straight_to_l3_is_refused(self, fleet: _Fleet) -> None:
        with pytest.raises(ValueError, match="L2-first"):
            async with fleet.writing.bypassing_write("precinct-1"):
                pass

    async def test_a_save_cannot_join_a_callers_transaction(self, fleet: _Fleet) -> None:
        with pytest.raises(ValueError, match="L2-first"):
            await fleet.writing.save_entity(fleet.writing.create({"id": "precinct-1", "votes": "1"}), conn=object())
