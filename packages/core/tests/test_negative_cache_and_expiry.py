"""BaseCollection negative caching and row expiry.

The contract this pins:

- a negative-caching collection records a full miss in L1 and L2, stamped with the table's write
  generation, so repeated lookups of a key nobody wrote reach L3 once per generation, across
  replicas;
- a recorded absence never answers after a committed write: every write advances the generation,
  so the absence is invalidated whatever happened to L2 in between -- a peer listener deleting the
  writer's fresh value, a writer in another principal scope, a reader whose L3 read straddled the
  commit;
- a generation that cannot be read means no absence is trusted or recorded;
- a writer that commits and cannot advance the generation, or cannot write L2, raises;
- the wiring that would let an absence lie is refused at construction;
- a row whose declared expiry has passed is absent to the reads that answer existence;
- a collection that opts into neither behaves exactly as before.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

import pytest
from sqlalchemy import Column, DateTime, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient
from threetears.nats import KvError

_SCOPE = "test-principal"
_OTHER_SCOPE = "other-principal"
_TABLE = "denylist_entries"
_MAX_AGE = timedelta(seconds=60)


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        _TABLE,
        metadata,
        Column("id", String(255), primary_key=True),
        Column("reason", String(255)),
        Column("expires_at", DateTime(timezone=True)),
        Column("date_created", DateTime(timezone=True)),
        Column("date_updated", DateTime(timezone=True)),
    )
    return metadata


class _Entry(BaseEntity):
    primary_key_field = "id"


class _Store:
    """the shared L3 every replica reads, counting how often it is asked."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.fetches = 0
        self.after_fetch: Callable[[], Awaitable[None]] | None = None


# parity-with: threetears.core.collections.generation.GenerationSource
class _FakeGenerations:
    """a generation source whose reads and advances a test can make fail."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.fail_current = False
        self.fail_advance = False

    async def current(self, table_name: str) -> str:
        if self.fail_current:
            raise GenerationUnavailableError("generation store unreachable")
        return f"incarnation-1:{self.counts.get(table_name, 0)}"

    async def advance(self, table_name: str) -> None:
        if self.fail_advance:
            raise GenerationUnavailableError("generation store unreachable")
        self.counts[table_name] = self.counts.get(table_name, 0) + 1


class _DenylistCollection(BaseCollection[_Entry]):
    """a three-tier collection with no opt-ins; subclasses turn them on."""

    datetime_columns: ClassVar[frozenset[str]] = frozenset({"expires_at", "date_created", "date_updated"})

    def __init__(self, registry: CollectionRegistry, config: DefaultCoreConfig, store: _Store) -> None:
        self._store = store
        super().__init__(registry, config)

    @property
    def table_name(self) -> str:
        return _TABLE

    @property
    def entity_class(self) -> type[_Entry]:
        return _Entry

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        self._store.fetches += 1
        row = self._store.rows.get(str(entity_id))
        found = dict(row) if row is not None else None
        hook, self._store.after_fetch = self._store.after_fetch, None
        if hook is not None:
            await hook()
        return found

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        self._store.rows[str(data["id"])] = dict(data)
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        self._store.rows.pop(str(entity_id), None)

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        row: dict[str, Any] = json.loads(data)
        return row


class _NegativeCaching(_DenylistCollection):
    negative_cache_max_age: ClassVar[timedelta | None] = _MAX_AGE


class _Expiring(_DenylistCollection):
    expires_at_column: ClassVar[str | None] = "expires_at"


class _NegativeCachingAndExpiring(_DenylistCollection):
    negative_cache_max_age: ClassVar[timedelta | None] = _MAX_AGE
    expires_at_column: ClassVar[str | None] = "expires_at"


class _HookedBucket(FakeKvBucket):
    """a real fake bucket whose writes a test can make fail."""

    def __init__(self, bucket_name: str) -> None:
        super().__init__(bucket_name)
        self.fail_put = False
        self.fail_delete = False

    async def put(self, *, key: str, value: bytes) -> int:
        if self.fail_put:
            raise KvError("simulated L2 write failure")
        return await super().put(key=key, value=value)

    async def delete(self, *, key: str, revision: int | None = None) -> bool:
        if self.fail_delete:
            raise KvError("simulated L2 delete failure")
        return await super().delete(key=key, revision=revision)


class _HookedNats(FakeNatsClient):
    """one shared hooked bucket for every replica, as the real collections bucket is."""

    def __init__(self) -> None:
        super().__init__()
        self.bucket = _HookedBucket("collections")
        self.published: list[Any] = []

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:
        """record the invalidation broadcast every write sends; no replica here subscribes."""
        self.published.append(message)

    async def kv_bucket(
        self,
        *,
        name: str,
        ttl: object | None = None,
        storage: str = "memory",
        create_if_missing: bool = True,
        history: int = 1,
    ) -> _HookedBucket:
        return self.bucket


def _replica(
    cls: type[_DenylistCollection],
    nats: _HookedNats | None,
    store: _Store,
    generations: _FakeGenerations | None,
    *,
    with_l3: bool = True,
    scope: str = _SCOPE,
    config: DefaultCoreConfig | None = None,
    l1: SQLiteBackend | None = None,
) -> _DenylistCollection:
    """one replica: its own L1 (the caller's, when a test inspects it), the shared L2, L3 and generation."""
    if l1 is None:
        l1 = SQLiteBackend(db_name=f"negcache_{uuid.uuid4().hex[:8]}")
    l1.initialize(_metadata())
    registry = CollectionRegistry()
    registry.configure(
        l1_backend=l1,
        l2_client=nats,
        l3_pool=object() if with_l3 else None,  # type: ignore[arg-type]
        kv_key_scope=scope if nats is not None else None,
    )
    if generations is not None:
        registry.set_generation_source(generations)
    return cls(registry, config or DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""), store)


async def _write(collection: _DenylistCollection, entity_id: str, *, expires_at: datetime | None = None) -> None:
    entity = collection.create({"id": entity_id, "reason": "revoked", "expires_at": expires_at})
    await collection.save_entity(entity)


def _wire() -> tuple[_HookedNats, _Store, _FakeGenerations]:
    return _HookedNats(), _Store(), _FakeGenerations()


class TestNegativeCaching:
    @pytest.mark.asyncio
    async def test_a_key_nobody_wrote_reaches_l3_once_across_replicas(self) -> None:
        nats, store, gens = _wire()
        a, b = _replica(_NegativeCaching, nats, store, gens), _replica(_NegativeCaching, nats, store, gens)
        assert await a.get("never-revoked") is None
        assert await a.get("never-revoked") is None  # L1 marker
        assert await b.get("never-revoked") is None  # L2 marker, for a replica that never asked
        assert store.fetches == 1

    @pytest.mark.asyncio
    async def test_without_opting_in_every_miss_reaches_l3(self) -> None:
        nats, store, gens = _wire()
        plain = _replica(_DenylistCollection, nats, store, gens)
        for _ in range(3):
            assert await plain.get("never-revoked") is None
        assert store.fetches == 3

    @pytest.mark.asyncio
    async def test_without_an_l3_pool_no_marker_is_written(self) -> None:
        nats, store, gens = _wire()
        l2_only = _replica(_NegativeCaching, nats, store, gens, with_l3=False)
        assert await l2_only.get("k") is None
        assert await nats.bucket.get(key=f"{_SCOPE}.{_TABLE}.k") is None

    @pytest.mark.asyncio
    async def test_a_write_invalidates_the_readers_markers_in_l1_and_l2(self) -> None:
        nats, store, gens = _wire()
        reader, writer, cold = (_replica(_NegativeCaching, nats, store, gens) for _ in range(3))
        assert await reader.get("tok") is None  # markers in the reader's L1 and in L2
        await _write(writer, "tok")
        assert await reader.get("tok") is not None, "the reader's own L1 marker outlived the write"
        assert await cold.get("tok") is not None

    @pytest.mark.asyncio
    async def test_a_peer_listener_deleting_the_writers_value_does_not_revive_the_absence(self) -> None:
        # the reviewed failure: a peer's invalidation listener evicts the writer's fresh L2 value,
        # and a reader then finds L2 empty. the absence must not come back.
        nats, store, gens = _wire()
        reader, writer = _replica(_NegativeCaching, nats, store, gens), _replica(_NegativeCaching, nats, store, gens)
        assert await reader.get("tok") is None
        await _write(writer, "tok")
        await nats.bucket.delete(key=f"{_SCOPE}.{_TABLE}.tok")  # what the listener does
        assert await reader.get("tok") is not None

    @pytest.mark.asyncio
    async def test_a_readers_l3_miss_that_straddles_a_commit_is_not_trusted(self) -> None:
        # the reader's L3 read finds nothing; the writer commits before the reader records the
        # absence. the absence is recorded under the generation the write already moved past.
        nats, store, gens = _wire()
        reader, writer = _replica(_NegativeCaching, nats, store, gens), _replica(_NegativeCaching, nats, store, gens)

        async def writer_commits_in_between() -> None:
            await _write(writer, "tok")

        store.after_fetch = writer_commits_in_between
        assert await reader.get("tok") is None  # its L3 read predated the write
        assert await reader.get("tok") is not None, "an absence recorded across a commit answered"

    @pytest.mark.asyncio
    async def test_a_writer_in_another_principal_invalidates_the_readers_absence(self) -> None:
        # different scopes never share an L2 key, and no broadcast reaches the reader here. only
        # the shared generation connects them.
        nats, store, gens = _wire()
        reader = _replica(_NegativeCaching, nats, store, gens, scope=_OTHER_SCOPE)
        writer = _replica(_NegativeCaching, nats, store, gens, scope=_SCOPE)
        assert await reader.get("tok") is None
        await _write(writer, "tok")
        assert await reader.get("tok") is not None

    @pytest.mark.asyncio
    async def test_a_writer_with_no_l2_client_still_invalidates_absences_other_pods_recorded(self) -> None:
        # absences are recorded by readers with L2; a writer wired without one must still advance
        # the generation they are stamped with, or they answer over its commit.
        nats, store, gens = _wire()
        reader = _replica(_NegativeCaching, nats, store, gens)
        l3_only_writer = _replica(_NegativeCaching, None, store, gens)
        assert await reader.get("tok") is None
        await _write(l3_only_writer, "tok")
        assert await reader.get("tok") is not None, "an L3-only writer left the reader's absence answering"

    @pytest.mark.asyncio
    async def test_expired_l1_markers_are_swept_without_their_keys_being_read_again(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # a denylist checks a different key per token, so a marker's own key is rarely seen twice;
        # expiry has to reach rows nobody looks up again.
        clock = [1_000.0]
        monkeypatch.setattr(time, "monotonic", lambda: clock[0])
        nats, store, gens = _wire()
        l1 = SQLiteBackend(db_name=f"negcache_{uuid.uuid4().hex[:8]}")
        coll = _replica(_NegativeCaching, nats, store, gens, l1=l1)
        for n in range(5):
            assert await coll.get(f"token-{n}") is None
        assert len(l1.execute_query("SELECT key FROM collection_absent_markers")) == 5

        clock[0] += _MAX_AGE.total_seconds() + 61.0  # past every deadline and the sweep interval
        assert await coll.get("a-new-token") is None  # a marker write triggers the sweep
        remaining = l1.execute_query("SELECT key FROM collection_absent_markers")
        assert len(remaining) == 1, "expired markers for keys nobody read again were left in L1"

    @pytest.mark.asyncio
    async def test_one_sweep_drains_a_backlog_larger_than_its_batch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # a sweep capped at one batch per interval falls behind any miss rate above batch/interval,
        # and the table then grows without bound.
        import threetears.core.collections.base as base_module

        clock = [1_000.0]
        monkeypatch.setattr(time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(base_module, "_ABSENT_MARKER_SWEEP_BATCH", 2)
        nats, store, gens = _wire()
        l1 = SQLiteBackend(db_name=f"negcache_{uuid.uuid4().hex[:8]}")
        coll = _replica(_NegativeCaching, nats, store, gens, l1=l1)
        for n in range(7):
            assert await coll.get(f"token-{n}") is None

        clock[0] += _MAX_AGE.total_seconds() + 61.0
        assert await coll.get("a-new-token") is None
        remaining = l1.execute_query("SELECT key FROM collection_absent_markers")
        assert len(remaining) == 1, "the sweep stopped after one batch and left the backlog"

    @pytest.mark.asyncio
    async def test_an_l2_marker_leaves_the_bucket_when_its_lifetime_passes(self) -> None:
        nats, store, gens = _wire()
        a = _replica(_NegativeCaching, nats, store, gens)
        assert await a.get("k") is None
        key = f"{_SCOPE}.{_TABLE}.k"
        assert await nats.bucket.get(key=key) is not None
        nats.bucket.advance_clock(_MAX_AGE)
        assert await nats.bucket.get(key=key) is None, "a marker nobody reads again stayed in the bucket"

    @pytest.mark.asyncio
    async def test_an_unreadable_generation_trusts_and_records_no_absence(self) -> None:
        nats, store, gens = _wire()
        a = _replica(_NegativeCaching, nats, store, gens)
        assert await a.get("k") is None  # recorded under a readable generation
        gens.fail_current = True
        assert await a.get("k") is None
        assert await a.get("k") is None
        assert store.fetches == 3  # both later reads went to L3

    @pytest.mark.asyncio
    async def test_a_writer_that_cannot_advance_the_generation_raises_after_committing(self) -> None:
        nats, store, gens = _wire()
        reader, writer = _replica(_NegativeCaching, nats, store, gens), _replica(_NegativeCaching, nats, store, gens)
        assert await reader.get("tok") is None
        gens.fail_advance = True
        with pytest.raises(GenerationUnavailableError):
            await _write(writer, "tok")
        assert "tok" in store.rows
        assert nats.published, "the invalidation broadcast still went out"

    @pytest.mark.asyncio
    async def test_a_marker_in_l2_does_not_break_a_cas_mutation(self) -> None:
        nats, store, gens = _wire()
        coll = _replica(_NegativeCaching, nats, store, gens)
        assert await coll.get("k") is None  # L2 now holds a marker for k
        await coll.l2_cas_mutate("k", lambda row: ("upsert", {"id": "k", "reason": "set"}))
        raw = await nats.bucket.get(key=f"{_SCOPE}.{_TABLE}.k")
        assert raw is not None and json.loads(raw)["reason"] == "set"

    @pytest.mark.asyncio
    async def test_an_undecodable_l2_entry_is_replaced_by_a_marker(self) -> None:
        nats, store, gens = _wire()
        coll = _replica(_NegativeCaching, nats, store, gens)
        key = f"{_SCOPE}.{_TABLE}.k"
        await nats.bucket.put(key=key, value=b'{"id": "k", "date_created": "not-a-time"}')
        assert await coll.get("k") is None
        other = _replica(_NegativeCaching, nats, store, gens)
        assert await other.get("k") is None
        assert store.fetches == 1, "the poisoned entry kept sending lookups to L3"


class TestFailedL2Writes:
    @pytest.mark.asyncio
    async def test_a_failed_l2_write_cannot_leave_an_absence_answering(self) -> None:
        # the L2 marker the write failed to replace was stamped before the commit advanced the
        # generation, so it stops answering without the write ever reaching L2.
        nats, store, gens = _wire()
        reader = _replica(_NegativeCaching, nats, store, gens)
        assert await reader.get("tok") is None  # absence recorded in L1 and L2
        writer = _replica(_NegativeCaching, nats, store, gens)
        nats.bucket.fail_put = True
        await _write(writer, "tok")
        nats.bucket.fail_put = False
        other = _replica(_NegativeCaching, nats, store, gens)
        assert await reader.get("tok") is not None, "this pod's own recorded absence hid the row"
        assert await other.get("tok") is not None, "the L2 marker hid the row"

    @pytest.mark.asyncio
    async def test_failed_l2_writes_still_degrade_without_opting_in(self) -> None:
        nats, store, gens = _wire()
        coll = _replica(_DenylistCollection, nats, store, gens)
        nats.bucket.fail_put = True
        nats.bucket.fail_delete = True
        await _write(coll, "tok")
        await coll.delete("tok")


class TestUnsoundWiringIsRefused:
    def test_no_generation_source_is_refused_at_construction(self) -> None:
        nats, store, _ = _wire()
        with pytest.raises(ValueError, match="generation source"):
            _replica(_NegativeCaching, nats, store, None)

    def test_no_generation_source_is_refused_even_without_an_l2_client(self) -> None:
        # an L3-only writer that cannot advance the generation is as unsound as a reader that
        # cannot read it.
        _, store, _ = _wire()
        with pytest.raises(ValueError, match="generation source"):
            _replica(_NegativeCaching, None, store, None)

    def test_deferred_l3_writes_are_refused_at_construction(self) -> None:
        nats, store, gens = _wire()
        deferred = DefaultCoreConfig(collection_flush="ON_CHECKPOINT", collection_flush_tables=_TABLE)
        with pytest.raises(ValueError, match="defers its L3"):
            _replica(_NegativeCaching, nats, store, gens, config=deferred)

    def test_subscript_writes_are_refused(self) -> None:
        nats, store, gens = _wire()
        coll = _replica(_NegativeCaching, nats, store, gens)
        with pytest.raises(TypeError, match="save_entity"):
            coll["tok"] = {"id": "tok", "reason": "r"}

    @pytest.mark.asyncio
    async def test_joining_a_callers_transaction_is_refused(self) -> None:
        nats, store, gens = _wire()
        coll = _replica(_NegativeCaching, nats, store, gens)
        entity = coll.create({"id": "tok", "reason": "r", "expires_at": None})
        with pytest.raises(ValueError, match="transaction"):
            await coll.save_entity(entity, conn=object())

    def test_an_expiry_column_that_is_not_a_datetime_column_is_refused_at_class_definition(self) -> None:
        with pytest.raises(TypeError, match="datetime_columns"):

            class _Bad(_DenylistCollection):
                datetime_columns: ClassVar[frozenset[str]] = frozenset({"date_created"})
                expires_at_column: ClassVar[str | None] = "expires_at"

    def test_a_max_age_under_one_second_is_refused_at_class_definition(self) -> None:
        with pytest.raises(TypeError, match="one second"):

            class _TooShort(_DenylistCollection):
                negative_cache_max_age: ClassVar[timedelta | None] = timedelta(milliseconds=500)


class TestRowExpiry:
    @pytest.mark.asyncio
    async def test_an_expired_row_is_absent_to_the_reads_that_answer_existence(self) -> None:
        nats, store, gens = _wire()
        coll = _replica(_Expiring, nats, store, gens)
        await _write(coll, "old", expires_at=datetime.now(UTC) - timedelta(seconds=1))
        assert await coll.get("old") is None
        assert await coll.ensure("old") is None
        with pytest.raises(KeyError):
            coll["old"]

    @pytest.mark.asyncio
    async def test_an_entity_held_past_its_expiry_can_still_be_written(self) -> None:
        # reporting reads serve the entity's own internals; hiding its row there would turn this
        # save into "L1 cache miss in to_dict()" rather than a write.
        nats, store, gens = _wire()
        coll = _replica(_Expiring, nats, store, gens)
        entity = coll.create({"id": "held", "reason": "r", "expires_at": datetime.now(UTC) - timedelta(seconds=1)})
        await coll.save_entity(entity)
        assert store.rows["held"]["reason"] == "r"

    @pytest.mark.asyncio
    async def test_an_expired_row_is_absent_from_l2_and_l3(self) -> None:
        nats, store, gens = _wire()
        writer = _replica(_Expiring, nats, store, gens)
        await _write(writer, "old", expires_at=datetime.now(UTC) - timedelta(seconds=1))
        cold = _replica(_Expiring, nats, store, gens)  # empty L1: must go through L2, then L3
        assert await cold.get("old") is None
        assert store.fetches == 1  # the expired L2 row sent it to L3, which is expired too

    @pytest.mark.asyncio
    async def test_a_row_expiring_later_and_a_row_that_never_expires_are_served(self) -> None:
        nats, store, gens = _wire()
        coll = _replica(_Expiring, nats, store, gens)
        await _write(coll, "live", expires_at=datetime.now(UTC) + timedelta(hours=1))
        await _write(coll, "forever", expires_at=None)
        cold = _replica(_Expiring, nats, store, gens)
        assert await cold.get("live") is not None
        assert await cold.get("forever") is not None

    @pytest.mark.asyncio
    async def test_an_expired_row_becomes_an_absence_that_a_later_write_supersedes(self) -> None:
        nats, store, gens = _wire()
        writer = _replica(_NegativeCachingAndExpiring, nats, store, gens)
        await _write(writer, "tok", expires_at=datetime.now(UTC) - timedelta(seconds=1))
        reader = _replica(_NegativeCachingAndExpiring, nats, store, gens)
        assert await reader.get("tok") is None  # expired row replaced by an absent-marker
        assert await reader.get("tok") is None
        assert store.fetches == 1

        await _write(writer, "tok", expires_at=datetime.now(UTC) + timedelta(hours=1))
        assert await reader.get("tok") is not None, "an expired row's absence masked a later write"
