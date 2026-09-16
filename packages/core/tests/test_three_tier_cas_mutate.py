"""``l2_cas_mutate`` on a collection with an L3 pool, and the per-collection L3 write policy.

The contract this pins:

- when L2 holds no live row, the mutation starts from L3's row, so a counter survives a broker
  wipe instead of restarting at zero -- and replicas racing to seed converge on one count;
- the won result lands in L3 synchronously, or through the write buffer, per ``l3_write_policy``;
  a delete always lands synchronously;
- the outcome says what was done;
- on a collection that caches absences, a CAS-created row invalidates recorded absences;
- wiring that cannot honour a declared policy is refused at construction.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Literal

import pytest
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.flush import WriteBuffer, flush_pending
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.kv import FakeNatsClient

_SCOPE = "cas-principal"
_TABLE = "attempt_counters"
_KEY = f"{_SCOPE}.{_TABLE}.acct-1"


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        _TABLE,
        metadata,
        Column("id", String(255), primary_key=True),
        Column("count", Integer),
        Column("date_created", DateTime(timezone=True)),
        Column("date_updated", DateTime(timezone=True)),
    )
    return metadata


class _Counter(BaseEntity):
    primary_key_field = "id"


class _Store:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.affect_nothing = False


class _Counters(BaseCollection[_Counter]):
    """a three-tier attempt counter over an in-process L3."""

    datetime_columns: ClassVar[frozenset[str]] = frozenset({"date_created", "date_updated"})

    def __init__(
        self, registry: CollectionRegistry, config: DefaultCoreConfig, store: _Store, buffer: WriteBuffer | None
    ) -> None:
        self._store = store
        super().__init__(registry, config, write_buffer=buffer)

    @property
    def table_name(self) -> str:
        return _TABLE

    @property
    def entity_class(self) -> type[_Counter]:
        return _Counter

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        row = self._store.rows.get(str(entity_id))
        return dict(row) if row is not None else None

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        if self._store.affect_nothing:
            return 0
        self._store.rows[str(data["id"])] = dict(data)
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        self._store.rows.pop(str(entity_id), None)

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        row: dict[str, Any] = json.loads(data)
        return row


class _SynchronousCounters(_Counters):
    l3_write_policy: ClassVar[Literal["synchronous", "write_behind"] | None] = "synchronous"


class _WriteBehindCounters(_Counters):
    l3_write_policy: ClassVar[Literal["synchronous", "write_behind"] | None] = "write_behind"


class _NegativeCachingWriteBehind(_WriteBehindCounters):
    negative_cache_max_age: ClassVar[timedelta | None] = timedelta(seconds=60)


class _NegativeCachingCounters(_SynchronousCounters):
    negative_cache_max_age: ClassVar[timedelta | None] = timedelta(seconds=60)


# parity-with: threetears.core.collections.generation.GenerationSource
class _FakeGenerations:
    def __init__(self) -> None:
        self.count = 0

    async def current(self, table_name: str) -> str:
        return f"i:{self.count}"

    async def advance(self, table_name: str) -> None:
        self.count += 1


class _Nats(FakeNatsClient):
    """the shared collections bucket, and the invalidation broadcast delivered to every listening replica."""

    def __init__(self) -> None:
        super().__init__()
        self._subscribers: list[tuple[Any, Any]] = []

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:
        for cb, message_type in list(self._subscribers):
            await cb(message_type.model_validate_json(message.model_dump_json()))

    async def subscribe_typed(self, *, subject: Any, cb: Any, message_type: Any, **_: Any) -> object:
        self._subscribers.append((cb, message_type))
        return object()


def _replica(
    cls: type[_Counters],
    nats: _Nats,
    store: _Store,
    *,
    buffer: WriteBuffer | None = None,
    config: DefaultCoreConfig | None = None,
    generations: _FakeGenerations | None = None,
    scope: str = _SCOPE,
) -> tuple[_Counters, CollectionRegistry]:
    l1 = SQLiteBackend(db_name=f"cas_{uuid.uuid4().hex[:8]}")
    l1.initialize(_metadata())
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l2_client=nats, l3_pool=object(), kv_key_scope=scope)  # type: ignore[arg-type]
    if generations is not None:
        registry.set_generation_source(generations)
    collection = cls(
        registry, config or DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""), store, buffer
    )
    return collection, registry


def _increment(row: dict[str, Any] | None) -> tuple[Literal["upsert", "delete", "noop"], dict[str, Any] | None]:
    count = 0 if row is None else int(row["count"])
    return "upsert", {"id": "acct-1", "count": count + 1}


async def _l2_count(nats: _Nats) -> int:
    bucket = await nats.kv_bucket(name="collections")
    raw = await bucket.get(key=_KEY)
    assert raw is not None
    return int(json.loads(raw)["count"])


class TestSeedingFromL3:
    @pytest.mark.asyncio
    async def test_a_counter_with_nothing_in_l2_continues_from_its_l3_value(self) -> None:
        nats, store = _Nats(), _Store()
        store.rows["acct-1"] = {"id": "acct-1", "count": 5}
        coll, _ = _replica(_SynchronousCounters, nats, store)
        outcome = await coll.l2_cas_mutate("acct-1", _increment)
        assert outcome.action == "updated"
        assert await _l2_count(nats) == 6
        assert store.rows["acct-1"]["count"] == 6

    @pytest.mark.asyncio
    async def test_a_counter_survives_a_broker_wipe(self) -> None:
        nats, store = _Nats(), _Store()
        coll, _ = _replica(_SynchronousCounters, nats, store)
        await coll.l2_cas_mutate("acct-1", _increment)
        await coll.l2_cas_mutate("acct-1", _increment)
        (await nats.kv_bucket(name="collections")).wipe()
        outcome = await coll.l2_cas_mutate("acct-1", _increment)
        assert outcome.row is not None and outcome.row["count"] == 3, "the wipe reset the counter"
        assert await _l2_count(nats) == 3

    @pytest.mark.asyncio
    async def test_replicas_racing_to_seed_converge_on_one_count(self) -> None:
        nats, store = _Nats(), _Store()
        store.rows["acct-1"] = {"id": "acct-1", "count": 5}
        replicas = [_replica(_SynchronousCounters, nats, store)[0] for _ in range(4)]
        await asyncio.gather(*(r.l2_cas_mutate("acct-1", _increment) for r in replicas))
        assert await _l2_count(nats) == 9
        assert store.rows["acct-1"]["count"] == 9

    @pytest.mark.asyncio
    async def test_without_an_l3_pool_the_l2_value_is_the_only_record(self) -> None:
        # the L1+L2-only behaviour presence relies on: no seeding, nothing persisted.
        nats, store = _Nats(), _Store()
        store.rows["acct-1"] = {"id": "acct-1", "count": 5}
        l1 = SQLiteBackend(db_name=f"cas_{uuid.uuid4().hex[:8]}")
        l1.initialize(_metadata())
        registry = CollectionRegistry()
        registry.configure(l1_backend=l1, l2_client=nats, kv_key_scope=_SCOPE)
        coll = _SynchronousCounters(
            registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""), store, None
        )
        outcome = await coll.l2_cas_mutate("acct-1", _increment)
        assert outcome.action == "created"
        assert await _l2_count(nats) == 1
        assert store.rows["acct-1"]["count"] == 5


class TestTheL3WritePolicy:
    @pytest.mark.asyncio
    async def test_write_behind_lands_in_l3_only_when_the_buffer_flushes(self) -> None:
        nats, store, buffer = _Nats(), _Store(), WriteBuffer()
        coll, registry = _replica(_WriteBehindCounters, nats, store, buffer=buffer)
        await coll.l2_cas_mutate("acct-1", _increment)
        await coll.l2_cas_mutate("acct-1", _increment)
        assert "acct-1" not in store.rows
        assert await flush_pending(buffer, registry) == 1
        assert store.rows["acct-1"]["count"] == 2

    @pytest.mark.asyncio
    async def test_a_declared_synchronous_policy_overrides_a_deferring_process_strategy(self) -> None:
        nats, store, buffer = _Nats(), _Store(), WriteBuffer()
        deferring = DefaultCoreConfig(collection_flush="ON_CHECKPOINT", collection_flush_tables=_TABLE)
        coll, _ = _replica(_SynchronousCounters, nats, store, buffer=buffer, config=deferring)
        await coll.l2_cas_mutate("acct-1", _increment)
        assert store.rows["acct-1"]["count"] == 1
        assert buffer.pending_count() == 0

    @pytest.mark.asyncio
    async def test_a_delete_lands_in_l3_synchronously_even_when_writes_are_behind(self) -> None:
        nats, store, buffer = _Nats(), _Store(), WriteBuffer()
        store.rows["acct-1"] = {"id": "acct-1", "count": 5}
        coll, _ = _replica(_WriteBehindCounters, nats, store, buffer=buffer)
        outcome = await coll.l2_cas_mutate("acct-1", lambda _row: ("delete", None))
        assert outcome.action == "deleted"
        assert "acct-1" not in store.rows

    @pytest.mark.asyncio
    async def test_a_synchronous_persist_that_affects_no_row_raises(self) -> None:
        nats, store = _Nats(), _Store()
        store.affect_nothing = True
        coll, _ = _replica(_SynchronousCounters, nats, store)
        with pytest.raises(RuntimeError, match="affected no L3 row"):
            await coll.l2_cas_mutate("acct-1", _increment)


class TestPeersListening:
    @pytest.mark.asyncio
    async def test_a_listening_peer_in_the_same_scope_keeps_the_counter(self) -> None:
        # write-behind: L3 holds nothing until a flush, so the shared L2 key is the only record
        # of the count. a peer that evicted it on every broadcast would restart the counter.
        nats, store = _Nats(), _Store()
        first, first_registry = _replica(_WriteBehindCounters, nats, store, buffer=WriteBuffer())
        second, second_registry = _replica(_WriteBehindCounters, nats, store, buffer=WriteBuffer())
        await first_registry.start_invalidation_listener(nats)  # type: ignore[arg-type]
        await second_registry.start_invalidation_listener(nats)  # type: ignore[arg-type]
        for replica in (first, second, first, second):
            await replica.l2_cas_mutate("acct-1", _increment)
        assert await _l2_count(nats) == 4, "a peer's listener evicted the compare-and-swap key"

    @pytest.mark.asyncio
    async def test_a_listener_in_another_scope_still_evicts_its_own_copy(self) -> None:
        nats, store = _Nats(), _Store()
        writer, _ = _replica(_SynchronousCounters, nats, store)
        await writer.l2_cas_mutate("acct-1", _increment)
        reader, reader_registry = _replica(_SynchronousCounters, nats, store, scope="other-principal")
        await reader_registry.start_invalidation_listener(nats)  # type: ignore[arg-type]
        assert await reader.get("acct-1") is not None  # pulls through into the reader's own key
        bucket = await nats.kv_bucket(name="collections")
        other_key = f"other-principal.{_TABLE}.acct-1"
        assert await bucket.get(key=other_key) is not None
        await writer.l2_cas_mutate("acct-1", _increment)
        assert await bucket.get(key=other_key) is None, "another principal's stale copy survived"


class TestPersistFailure:
    @pytest.mark.asyncio
    async def test_a_raising_persist_withdraws_the_l2_value_and_a_retry_counts_once(self) -> None:
        nats, store = _Nats(), _Store()
        store.rows["acct-1"] = {"id": "acct-1", "count": 5}
        coll, _ = _replica(_SynchronousCounters, nats, store)
        outage = RuntimeError("L3 unavailable")

        async def _refuse(data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None) -> int:
            raise outage

        coll.save_to_store = _refuse  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="L3 unavailable"):
            await coll.l2_cas_mutate("acct-1", _increment)
        bucket = await nats.kv_bucket(name="collections")
        assert await bucket.get(key=_KEY) is None, "L2 kept a write the caller was told failed"
        del coll.save_to_store
        outcome = await coll.l2_cas_mutate("acct-1", _increment)
        assert outcome.row is not None and outcome.row["count"] == 6, "the failed increment was counted"
        assert store.rows["acct-1"]["count"] == 6

    @pytest.mark.asyncio
    async def test_a_persist_affecting_no_row_withdraws_the_l2_value(self) -> None:
        nats, store = _Nats(), _Store()
        store.affect_nothing = True
        coll, _ = _replica(_SynchronousCounters, nats, store)
        with pytest.raises(RuntimeError, match="affected no L3 row"):
            await coll.l2_cas_mutate("acct-1", _increment)
        bucket = await nats.kv_bucket(name="collections")
        assert await bucket.get(key=_KEY) is None

    @pytest.mark.asyncio
    async def test_a_table_that_fences_every_l3_write_is_refused_before_l2_is_touched(self) -> None:
        class _Fenced(_SynchronousCounters):
            @property
            def emits_cas_fence(self) -> bool:
                return True

        nats, store = _Nats(), _Store()
        coll, _ = _replica(_Fenced, nats, store)
        with pytest.raises(ValueError, match="fences every L3 write"):
            await coll.l2_cas_mutate("acct-1", _increment)
        bucket = await nats.kv_bucket(name="collections")
        assert await bucket.get(key=_KEY) is None


class TestCreationTime:
    @pytest.mark.asyncio
    async def test_a_row_new_to_every_tier_is_stamped_even_over_an_l2_entry(self) -> None:
        nats, store = _Nats(), _Store()
        bucket = await nats.kv_bucket(name="collections")
        await bucket.put(key=_KEY, value=b'{"id": "acct-1", "date_created": "not-a-time"}')  # corrupt
        coll, _ = _replica(_SynchronousCounters, nats, store)
        outcome = await coll.l2_cas_mutate("acct-1", _increment)
        assert outcome.row is not None and outcome.row.get("date_created") is not None
        assert store.rows["acct-1"].get("date_created") is not None

    @pytest.mark.asyncio
    async def test_a_row_seeded_from_l3_keeps_its_creation_time(self) -> None:
        nats, store = _Nats(), _Store()
        created = datetime(2026, 1, 1, tzinfo=UTC)
        store.rows["acct-1"] = {"id": "acct-1", "count": 5, "date_created": created}
        coll, _ = _replica(_SynchronousCounters, nats, store)
        await coll.l2_cas_mutate("acct-1", _increment)
        assert store.rows["acct-1"]["date_created"] == created


class TestOutcomes:
    @pytest.mark.asyncio
    async def test_created_updated_deleted_and_noop_are_reported(self) -> None:
        nats, store = _Nats(), _Store()
        coll, _ = _replica(_SynchronousCounters, nats, store)
        assert (await coll.l2_cas_mutate("acct-1", _increment)).action == "created"
        assert (await coll.l2_cas_mutate("acct-1", _increment)).action == "updated"
        assert (await coll.l2_cas_mutate("acct-1", lambda _row: ("noop", None))).action == "noop"
        assert (await coll.l2_cas_mutate("acct-1", lambda _row: ("delete", None))).action == "deleted"


class TestNegativeCachingThroughCas:
    @pytest.mark.asyncio
    async def test_a_cas_created_row_invalidates_a_recorded_absence(self) -> None:
        nats, store, generations = _Nats(), _Store(), _FakeGenerations()
        reader, _ = _replica(_NegativeCachingCounters, nats, store, generations=generations)
        writer, _ = _replica(_NegativeCachingCounters, nats, store, generations=generations)
        assert await reader.get("acct-1") is None  # absence recorded
        await writer.l2_cas_mutate("acct-1", _increment)
        assert await reader.get("acct-1") is not None, "a recorded absence hid a CAS-created row"


class TestUnhonourablePoliciesAreRefused:
    def test_write_behind_without_a_write_buffer_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no write buffer"):
            _replica(_WriteBehindCounters, _Nats(), _Store())

    def test_negative_caching_with_write_behind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="defers its L3"):
            _replica(
                _NegativeCachingWriteBehind, _Nats(), _Store(), buffer=WriteBuffer(), generations=_FakeGenerations()
            )
