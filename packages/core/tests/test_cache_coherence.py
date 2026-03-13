"""Tests for cross-pod cache coherence via NATS pub/sub invalidation signals.

TDD: these tests are written BEFORE implementation. They define the contract
for cache coherence — every write path must signal, every signal must evict L1.

Test architecture:
- Each "pod" = separate L1 (SQLiteBackend) + shared L2 (NATS mock) + shared L3 (dict)
- InMemoryNatsBus: mock that supports both KV operations AND pub/sub signaling
- Tests verify that writes on pod A cause L1 eviction on pod B via the signal path
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Any

import pytest
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.flush import WriteBuffer
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


def _make_metadata() -> MetaData:
    metadata = MetaData()
    Table(
        "test_entities",
        metadata,
        Column("id", String(255), primary_key=True),
        Column("name", String(255)),
        Column("score", Integer),
        Column("date_created", DateTime),
        Column("date_updated", DateTime),
    )
    return metadata


class StubEntity(BaseEntity):
    _primary_key_field = "id"


class StubCollection(BaseCollection[StubEntity]):
    """Concrete collection for coherence tests."""

    def __init__(
        self,
        registry: CollectionRegistry,
        config: DefaultCoreConfig,
        nats_client: Any = None,
        write_buffer: WriteBuffer | None = None,
        pg_store: dict[str, dict] | None = None,
    ) -> None:
        self._pg_store = pg_store if pg_store is not None else {}
        super().__init__(registry, config, nats_client, write_buffer)

    @property
    def table_name(self) -> str:
        return "test_entities"

    @property
    def entity_class(self) -> type[StubEntity]:
        return StubEntity

    async def _fetch_from_postgres(self, entity_id: object) -> dict | None:
        return self._pg_store.get(str(entity_id))

    async def _save_to_postgres(self, data: dict, original_timestamp: datetime | None = None) -> int:
        pk = data.get("id")
        if original_timestamp is not None:
            existing = self._pg_store.get(str(pk))
            if existing and existing.get("date_updated") != original_timestamp:
                return 0
        self._pg_store[str(pk)] = dict(data)
        return 1

    async def _delete_from_postgres(self, entity_id: object) -> None:
        self._pg_store.pop(str(entity_id), None)

    def _serialize(self, data: dict) -> bytes:
        return json.dumps(data, default=str).encode()

    def _deserialize(self, data: bytes) -> dict:
        return json.loads(data)


class InMemoryNatsBus:
    """Mock NATS client that supports both KV operations AND pub/sub.

    KV: in-memory dict store (same as existing test mocks).
    Pub/sub: maintains subscriber callbacks, dispatches on publish.
    Simulates cross-pod communication — all subscribers see all messages.
    """

    def __init__(self) -> None:
        self._kv_store: dict[str, bytes] = {}
        self._subscribers: dict[str, list[Any]] = {}  # subject -> [callbacks]
        self._publish_count: int = 0
        self._received_messages: list[tuple[str, bytes]] = []

    def bucket_name(self, suffix: str) -> str:
        return f"test-{suffix}"

    # --- KV operations ---

    async def get(self, bucket: str, key: str) -> bytes | None:
        return self._kv_store.get(key)

    async def put(self, bucket: str, key: str, value: bytes) -> bool:
        self._kv_store[key] = value
        return True

    async def delete(self, bucket: str, key: str) -> bool:
        self._kv_store.pop(key, None)
        return True

    # --- Pub/sub operations ---

    async def publish(self, subject: str, data: bytes) -> bool:
        """Publish a message. All subscribers on this subject receive it."""
        self._publish_count += 1
        self._received_messages.append((subject, data))
        callbacks = self._subscribers.get(subject, [])
        for cb in callbacks:
            try:
                await cb(data)
            except Exception:
                pass  # fail-open, like real NATS
        return True

    async def subscribe(self, subject: str, callback: Any) -> None:
        """Register a callback for a subject."""
        if subject not in self._subscribers:
            self._subscribers[subject] = []
        self._subscribers[subject].append(callback)


def _make_pod(
    nats: InMemoryNatsBus,
    pg_store: dict[str, dict],
    config: DefaultCoreConfig,
    write_buffer: WriteBuffer | None = None,
) -> tuple[StubCollection, CollectionRegistry]:
    """Create a collection + registry representing one pod (own L1, shared L2+L3)."""
    l1 = SQLiteBackend(db_name=f"test_pod_{uuid.uuid4().hex[:8]}")
    l1.initialize(_make_metadata())
    reg = CollectionRegistry()
    reg.configure(l1_backend=l1, l2_client=nats)
    coll = StubCollection(reg, config, nats_client=nats, write_buffer=write_buffer, pg_store=pg_store)
    return coll, reg


def _wait_for_publish(nats: InMemoryNatsBus, target_count: int, timeout: float = 5.0) -> None:
    """Poll until the NATS bus has received at least target_count publishes."""
    deadline = time.monotonic() + timeout
    while nats._publish_count < target_count:
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for {target_count} publishes (got {nats._publish_count})")
        time.sleep(0.02)


@pytest.fixture()
def config_always() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


@pytest.fixture()
def config_deferred() -> DefaultCoreConfig:
    return DefaultCoreConfig(
        collection_flush="ON_CHECKPOINT",
        collection_flush_tables="test_entities",
    )


@pytest.fixture()
def shared_nats() -> InMemoryNatsBus:
    return InMemoryNatsBus()


@pytest.fixture()
def shared_pg() -> dict[str, dict]:
    return {}


# ---------------------------------------------------------------------------
# NatsClient pub/sub tests
# ---------------------------------------------------------------------------


class TestNatsClientPubSub:
    """Tests for publish/subscribe on the NATS client."""

    @pytest.mark.asyncio
    async def test_publish_and_subscribe_basic(self) -> None:
        """Published message is received by subscriber."""
        nats = InMemoryNatsBus()
        received: list[bytes] = []

        async def on_msg(data: bytes) -> None:
            received.append(data)

        await nats.subscribe("test.subject", on_msg)
        await nats.publish("test.subject", b"hello")

        assert len(received) == 1
        assert received[0] == b"hello"

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self) -> None:
        """Multiple subscribers all receive the same message."""
        nats = InMemoryNatsBus()
        received_a: list[bytes] = []
        received_b: list[bytes] = []

        await nats.subscribe("test.subject", lambda d: received_a.append(d) or asyncio.sleep(0))
        await nats.subscribe("test.subject", lambda d: received_b.append(d) or asyncio.sleep(0))
        await nats.publish("test.subject", b"hello")

        assert len(received_a) == 1
        assert len(received_b) == 1

    @pytest.mark.asyncio
    async def test_publish_no_subscribers(self) -> None:
        """Publishing with no subscribers does not error."""
        nats = InMemoryNatsBus()
        result = await nats.publish("nobody.listening", b"hello")
        assert result is True

    @pytest.mark.asyncio
    async def test_subscriber_error_does_not_break_other_subscribers(self) -> None:
        """A failing subscriber does not prevent other subscribers from receiving."""
        nats = InMemoryNatsBus()
        received: list[bytes] = []

        async def bad_subscriber(data: bytes) -> None:
            raise RuntimeError("boom")

        async def good_subscriber(data: bytes) -> None:
            received.append(data)

        await nats.subscribe("test.subject", bad_subscriber)
        await nats.subscribe("test.subject", good_subscriber)
        await nats.publish("test.subject", b"hello")

        assert len(received) == 1


# ---------------------------------------------------------------------------
# Core invalidation signal tests — verify the signal path works
# ---------------------------------------------------------------------------


class TestInvalidationSignalContract:
    """Verify that the invalidation signal protocol works end-to-end.

    These test the signal mechanism in isolation: publish a signal,
    verify the correct L1 eviction happens.
    """

    @pytest.mark.asyncio
    async def test_invalidation_signal_evicts_l1(
        self, shared_nats: InMemoryNatsBus, shared_pg: dict, config_always: DefaultCoreConfig
    ) -> None:
        """Receiving an invalidation signal evicts the entity from L1."""
        pod_a, reg_a = _make_pod(shared_nats, shared_pg, config_always)
        pod_b, reg_b = _make_pod(shared_nats, shared_pg, config_always)

        # Start invalidation listeners on both pods
        await reg_a.start_invalidation_listener(shared_nats)
        await reg_b.start_invalidation_listener(shared_nats)

        # Seed data in both pods' L1
        shared_pg["e1"] = {"id": "e1", "name": "Alice", "score": 42}
        await pod_a.ensure("e1")
        await pod_b.ensure("e1")
        assert "e1" in pod_a
        assert "e1" in pod_b

        # Manually publish an invalidation signal
        signal = json.dumps({"table": "test_entities", "entity_id": "e1"}).encode()
        await shared_nats.publish("threetears.cache.invalidate", signal)

        # Both pods' L1 should be evicted
        assert "e1" not in pod_a
        assert "e1" not in pod_b

    @pytest.mark.asyncio
    async def test_invalidation_for_unknown_table_is_ignored(
        self, shared_nats: InMemoryNatsBus, shared_pg: dict, config_always: DefaultCoreConfig
    ) -> None:
        """Signal for an unregistered table is silently ignored."""
        pod_a, reg_a = _make_pod(shared_nats, shared_pg, config_always)
        await reg_a.start_invalidation_listener(shared_nats)

        shared_pg["e1"] = {"id": "e1", "name": "Alice", "score": 42}
        await pod_a.ensure("e1")

        # Signal for a table that doesn't exist in this registry
        signal = json.dumps({"table": "nonexistent_table", "entity_id": "e1"}).encode()
        await shared_nats.publish("threetears.cache.invalidate", signal)

        # Pod A's data should be untouched
        assert "e1" in pod_a

    @pytest.mark.asyncio
    async def test_invalidation_for_missing_entity_is_safe(
        self, shared_nats: InMemoryNatsBus, shared_pg: dict, config_always: DefaultCoreConfig
    ) -> None:
        """Evicting a non-existent entity does not crash."""
        pod_a, reg_a = _make_pod(shared_nats, shared_pg, config_always)
        await reg_a.start_invalidation_listener(shared_nats)

        signal = json.dumps({"table": "test_entities", "entity_id": "nonexistent"}).encode()
        await shared_nats.publish("threetears.cache.invalidate", signal)

        # No crash, no side effects

    @pytest.mark.asyncio
    async def test_malformed_signal_is_ignored(
        self, shared_nats: InMemoryNatsBus, shared_pg: dict, config_always: DefaultCoreConfig
    ) -> None:
        """Malformed signal payload is logged and ignored, not crashed."""
        pod_a, reg_a = _make_pod(shared_nats, shared_pg, config_always)
        await reg_a.start_invalidation_listener(shared_nats)

        shared_pg["e1"] = {"id": "e1", "name": "Alice", "score": 42}
        await pod_a.ensure("e1")

        # Publish garbage
        await shared_nats.publish("threetears.cache.invalidate", b"not json")

        # Pod A's data should be untouched — no crash
        assert "e1" in pod_a


# ---------------------------------------------------------------------------
# Write path signal emission — every write must publish invalidation
# ---------------------------------------------------------------------------


class TestSetterEmitsSignal:
    """Verify that __setitem__ publishes invalidation signals."""

    @pytest.mark.asyncio
    async def test_field_setter_emits_signal(
        self, shared_nats: InMemoryNatsBus, shared_pg: dict, config_always: DefaultCoreConfig
    ) -> None:
        """collection[id, field] = value emits an invalidation signal."""
        pod_a, reg_a = _make_pod(shared_nats, shared_pg, config_always)
        pod_b, reg_b = _make_pod(shared_nats, shared_pg, config_always)
        await reg_a.start_invalidation_listener(shared_nats)
        await reg_b.start_invalidation_listener(shared_nats)

        # Both pods load entity
        shared_pg["e1"] = {"id": "e1", "name": "Alice", "score": 42}
        await pod_a.ensure("e1")
        await pod_b.ensure("e1")
        assert "e1" in pod_b

        initial_publish = shared_nats._publish_count

        # Pod A updates via setter
        pod_a["e1", "name"] = "Bob"

        # Wait for fire-and-forget to propagate
        _wait_for_publish(shared_nats, initial_publish + 1)

        # Pod B's L1 should be evicted by the signal
        assert "e1" not in pod_b

        # Pod B can still read the updated value via L2/L3 pull-through
        entity_b = await pod_b.get("e1")
        assert entity_b is not None
        assert entity_b.name == "Bob"

    @pytest.mark.asyncio
    async def test_dict_setter_emits_signal(
        self, shared_nats: InMemoryNatsBus, shared_pg: dict, config_always: DefaultCoreConfig
    ) -> None:
        """collection[id] = data_dict emits an invalidation signal."""
        pod_a, reg_a = _make_pod(shared_nats, shared_pg, config_always)
        pod_b, reg_b = _make_pod(shared_nats, shared_pg, config_always)
        await reg_a.start_invalidation_listener(shared_nats)
        await reg_b.start_invalidation_listener(shared_nats)

        # Pod B has old data in L1
        shared_pg["e1"] = {"id": "e1", "name": "Alice", "score": 42}
        await pod_b.ensure("e1")

        initial_publish = shared_nats._publish_count

        # Pod A writes full dict
        pod_a["e1"] = {"id": "e1", "name": "Replaced", "score": 99}
        _wait_for_publish(shared_nats, initial_publish + 1)

        # Pod B's L1 evicted
        assert "e1" not in pod_b


class TestSaveEntityEmitsSignal:
    """Verify that save_entity() publishes invalidation signals."""

    @pytest.mark.asyncio
    async def test_save_new_entity_emits_signal(
        self, shared_nats: InMemoryNatsBus, shared_pg: dict, config_always: DefaultCoreConfig
    ) -> None:
        """Saving a new entity emits an invalidation signal."""
        pod_a, reg_a = _make_pod(shared_nats, shared_pg, config_always)
        pod_b, reg_b = _make_pod(shared_nats, shared_pg, config_always)
        await reg_a.start_invalidation_listener(shared_nats)
        await reg_b.start_invalidation_listener(shared_nats)

        initial_publish_count = shared_nats._publish_count

        entity = pod_a.create({"id": "e1", "name": "New", "score": 0})
        await pod_a.save_entity(entity)

        # Signal was published
        assert shared_nats._publish_count > initial_publish_count

        # Pod B can read the new entity (via L2/L3)
        entity_b = await pod_b.get("e1")
        assert entity_b is not None
        assert entity_b.name == "New"

    @pytest.mark.asyncio
    async def test_save_existing_entity_evicts_other_pod_l1(
        self, shared_nats: InMemoryNatsBus, shared_pg: dict, config_always: DefaultCoreConfig
    ) -> None:
        """Saving an existing entity evicts stale L1 on other pods."""
        pod_a, reg_a = _make_pod(shared_nats, shared_pg, config_always)
        pod_b, reg_b = _make_pod(shared_nats, shared_pg, config_always)
        await reg_a.start_invalidation_listener(shared_nats)
        await reg_b.start_invalidation_listener(shared_nats)

        # Both pods load entity
        shared_pg["e1"] = {"id": "e1", "name": "Alice", "score": 42}
        await pod_a.ensure("e1")
        await pod_b.ensure("e1")

        # Pod A modifies and saves
        entity_a = await pod_a.get("e1")
        entity_a.name = "Updated"
        await pod_a.save_entity(entity_a)

        # Pod B's L1 should be evicted
        assert "e1" not in pod_b

        # Pod B reads fresh data
        entity_b = await pod_b.get("e1")
        assert entity_b is not None
        assert entity_b.name == "Updated"

    @pytest.mark.asyncio
    async def test_deferred_save_emits_signal(
        self, shared_nats: InMemoryNatsBus, shared_pg: dict, config_deferred: DefaultCoreConfig
    ) -> None:
        """Deferred save still emits invalidation (L1+L2 updated, other pods must evict)."""
        buf = WriteBuffer()
        pod_a, reg_a = _make_pod(shared_nats, shared_pg, config_deferred, write_buffer=buf)
        pod_b, reg_b = _make_pod(shared_nats, shared_pg, config_deferred, write_buffer=buf)
        await reg_a.start_invalidation_listener(shared_nats)
        await reg_b.start_invalidation_listener(shared_nats)

        # Pod B has data in L1
        shared_pg["e1"] = {"id": "e1", "name": "Alice", "score": 42}
        await pod_b.ensure("e1")

        # Pod A creates and saves (deferred — no L3 write)
        entity = pod_a.create({"id": "e1", "name": "Deferred", "score": 0})
        await pod_a.save_entity(entity)

        # Pod B's L1 should still be evicted (L2 has new data)
        assert "e1" not in pod_b


class TestDeleteEmitsSignal:
    """Verify that delete() publishes invalidation signals."""

    @pytest.mark.asyncio
    async def test_delete_evicts_other_pod_l1(
        self, shared_nats: InMemoryNatsBus, shared_pg: dict, config_always: DefaultCoreConfig
    ) -> None:
        """Deleting on pod A evicts pod B's L1 cache."""
        pod_a, reg_a = _make_pod(shared_nats, shared_pg, config_always)
        pod_b, reg_b = _make_pod(shared_nats, shared_pg, config_always)
        await reg_a.start_invalidation_listener(shared_nats)
        await reg_b.start_invalidation_listener(shared_nats)

        shared_pg["e1"] = {"id": "e1", "name": "Alice", "score": 42}
        await pod_a.ensure("e1")
        await pod_b.ensure("e1")

        await pod_a.delete("e1")

        # Pod B's L1 should be evicted
        assert "e1" not in pod_b

        # Pod B should get None (entity deleted from all tiers)
        entity_b = await pod_b.get("e1")
        assert entity_b is None


class TestReloadEmitsSignal:
    """Verify that reload_entity() publishes invalidation signals."""

    @pytest.mark.asyncio
    async def test_reload_evicts_other_pod_l1(
        self, shared_nats: InMemoryNatsBus, shared_pg: dict, config_always: DefaultCoreConfig
    ) -> None:
        """Reloading from L3 on pod A evicts pod B's stale L1."""
        pod_a, reg_a = _make_pod(shared_nats, shared_pg, config_always)
        pod_b, reg_b = _make_pod(shared_nats, shared_pg, config_always)
        await reg_a.start_invalidation_listener(shared_nats)
        await reg_b.start_invalidation_listener(shared_nats)

        shared_pg["e1"] = {"id": "e1", "name": "Alice", "score": 42}
        await pod_a.ensure("e1")
        await pod_b.ensure("e1")

        # External system updates L3 directly
        shared_pg["e1"]["name"] = "ExternalUpdate"

        # Pod A reloads from L3
        entity_a = await pod_a.get("e1")
        await pod_a.reload_entity(entity_a)

        # Pod B's stale L1 should be evicted
        assert "e1" not in pod_b

        # Pod B reads fresh data
        entity_b = await pod_b.get("e1")
        assert entity_b is not None
        assert entity_b.name == "ExternalUpdate"


class TestInvalidateCacheEmitsSignal:
    """Verify that invalidate_cache() publishes invalidation signals."""

    @pytest.mark.asyncio
    async def test_invalidate_cache_evicts_other_pod_l1(
        self, shared_nats: InMemoryNatsBus, shared_pg: dict, config_always: DefaultCoreConfig
    ) -> None:
        """Explicit cache invalidation on pod A evicts pod B's L1."""
        pod_a, reg_a = _make_pod(shared_nats, shared_pg, config_always)
        pod_b, reg_b = _make_pod(shared_nats, shared_pg, config_always)
        await reg_a.start_invalidation_listener(shared_nats)
        await reg_b.start_invalidation_listener(shared_nats)

        shared_pg["e1"] = {"id": "e1", "name": "Alice", "score": 42}
        await pod_a.ensure("e1")
        await pod_b.ensure("e1")

        await pod_a.invalidate_cache("e1")

        # Pod B's L1 evicted
        assert "e1" not in pod_b


# ---------------------------------------------------------------------------
# Convergence and stress tests
# ---------------------------------------------------------------------------


class TestCacheConvergence:
    """Verify that caches converge to consistent state under concurrent operations."""

    @pytest.mark.asyncio
    async def test_rapid_setter_updates_all_signals_received(
        self, shared_nats: InMemoryNatsBus, shared_pg: dict, config_always: DefaultCoreConfig
    ) -> None:
        """Rapid sequential updates via save_entity; pod B converges."""
        pod_a, reg_a = _make_pod(shared_nats, shared_pg, config_always)
        pod_b, reg_b = _make_pod(shared_nats, shared_pg, config_always)
        await reg_a.start_invalidation_listener(shared_nats)
        await reg_b.start_invalidation_listener(shared_nats)

        shared_pg["e1"] = {"id": "e1", "name": "v0", "score": 0}
        await pod_a.ensure("e1")
        await pod_b.ensure("e1")

        initial_publish = shared_nats._publish_count

        # 10 sequential save_entity calls (async, so signals are synchronous)
        for i in range(10):
            entity = await pod_a.get("e1")
            entity.name = f"v{i + 1}"
            await pod_a.save_entity(entity)

        # All 10 saves produced signals (save_entity publishes synchronously)
        assert shared_nats._publish_count >= initial_publish + 10

        # Pod B's L1 should be evicted
        assert "e1" not in pod_b

        # Pod B reads the latest value
        entity_b = await pod_b.get("e1")
        assert entity_b is not None
        assert entity_b.name == "v10"

    @pytest.mark.asyncio
    async def test_both_pods_write_caches_converge(
        self, shared_nats: InMemoryNatsBus, shared_pg: dict, config_always: DefaultCoreConfig
    ) -> None:
        """Both pods write different values; after settling, both see the last write."""
        pod_a, reg_a = _make_pod(shared_nats, shared_pg, config_always)
        pod_b, reg_b = _make_pod(shared_nats, shared_pg, config_always)
        await reg_a.start_invalidation_listener(shared_nats)
        await reg_b.start_invalidation_listener(shared_nats)

        shared_pg["e1"] = {"id": "e1", "name": "initial", "score": 0}
        await pod_a.ensure("e1")
        await pod_b.ensure("e1")

        # Pod A writes
        entity_a = await pod_a.get("e1")
        entity_a.name = "from_pod_a"
        await pod_a.save_entity(entity_a)

        # Pod B writes (last writer wins)
        entity_b = await pod_b.get("e1")
        entity_b.name = "from_pod_b"
        await pod_b.save_entity(entity_b)

        # After settling, both pods should see pod B's value (last writer wins)
        refreshed_a = await pod_a.get("e1")
        refreshed_b = await pod_b.get("e1")
        assert refreshed_a is not None
        assert refreshed_b is not None
        assert refreshed_a.name == "from_pod_b"
        assert refreshed_b.name == "from_pod_b"

    @pytest.mark.asyncio
    async def test_three_pods_all_converge(
        self, shared_nats: InMemoryNatsBus, shared_pg: dict, config_always: DefaultCoreConfig
    ) -> None:
        """Three pods: write on one, all three converge."""
        pod_a, reg_a = _make_pod(shared_nats, shared_pg, config_always)
        pod_b, reg_b = _make_pod(shared_nats, shared_pg, config_always)
        pod_c, reg_c = _make_pod(shared_nats, shared_pg, config_always)
        await reg_a.start_invalidation_listener(shared_nats)
        await reg_b.start_invalidation_listener(shared_nats)
        await reg_c.start_invalidation_listener(shared_nats)

        shared_pg["e1"] = {"id": "e1", "name": "original", "score": 42}
        await pod_a.ensure("e1")
        await pod_b.ensure("e1")
        await pod_c.ensure("e1")

        # Pod B updates
        entity_b = await pod_b.get("e1")
        entity_b.name = "from_pod_b"
        await pod_b.save_entity(entity_b)

        # All pods converge to pod B's value
        assert (await pod_a.get("e1")).name == "from_pod_b"
        assert (await pod_b.get("e1")).name == "from_pod_b"
        assert (await pod_c.get("e1")).name == "from_pod_b"


# ---------------------------------------------------------------------------
# Failure resilience — signals are fire-and-forget, writes must not fail
# ---------------------------------------------------------------------------


class TestSignalFailureResilience:
    """Verify that signal failures don't break write operations."""

    @pytest.mark.asyncio
    async def test_signal_publish_failure_does_not_break_save(self, config_always: DefaultCoreConfig) -> None:
        """If NATS publish fails, save_entity still succeeds."""
        nats = InMemoryNatsBus()
        pg_store: dict[str, dict] = {}
        pod_a, reg_a = _make_pod(nats, pg_store, config_always)

        # Sabotage publish
        async def broken_publish(subject: str, data: bytes) -> bool:
            raise ConnectionError("NATS down")

        nats.publish = broken_publish

        entity = pod_a.create({"id": "e1", "name": "Alice", "score": 0})
        await pod_a.save_entity(entity)

        # Save succeeded despite publish failure
        assert "e1" in pg_store
        assert pg_store["e1"]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_signal_publish_failure_does_not_break_setter(self, config_always: DefaultCoreConfig) -> None:
        """If NATS publish fails, setter still writes to L1."""
        nats = InMemoryNatsBus()
        pg_store: dict[str, dict] = {}
        pod_a, reg_a = _make_pod(nats, pg_store, config_always)

        pod_a._l1.upsert("test_entities", {"id": "e1", "name": "Alice", "score": 42})

        # Sabotage publish
        async def broken_publish(subject: str, data: bytes) -> bool:
            raise ConnectionError("NATS down")

        nats.publish = broken_publish

        pod_a["e1", "name"] = "Bob"

        # L1 write succeeded
        row = pod_a._l1.select_by_id("test_entities", "e1")
        assert row["name"] == "Bob"

    @pytest.mark.asyncio
    async def test_signal_publish_failure_does_not_break_delete(self, config_always: DefaultCoreConfig) -> None:
        """If NATS publish fails, delete still succeeds."""
        nats = InMemoryNatsBus()
        pg_store = {"e1": {"id": "e1", "name": "Alice", "score": 42}}
        pod_a, reg_a = _make_pod(nats, pg_store, config_always)

        async def broken_publish(subject: str, data: bytes) -> bool:
            raise ConnectionError("NATS down")

        nats.publish = broken_publish

        result = await pod_a.delete("e1")
        assert result is True
        assert "e1" not in pg_store

    @pytest.mark.asyncio
    async def test_no_nats_client_no_signal_no_crash(self, config_always: DefaultCoreConfig) -> None:
        """Collection with no NATS client skips signaling entirely."""
        l1 = SQLiteBackend(db_name=f"test_nosignal_{uuid.uuid4().hex[:8]}")
        l1.initialize(_make_metadata())
        reg = CollectionRegistry()
        reg.configure(l1_backend=l1)
        pg_store: dict[str, dict] = {}
        coll = StubCollection(reg, config_always, nats_client=None, pg_store=pg_store)

        entity = coll.create({"id": "e1", "name": "Alice", "score": 0})
        await coll.save_entity(entity)

        assert "e1" in pg_store
        assert coll["e1", "name"] == "Alice"


# ---------------------------------------------------------------------------
# Listener lifecycle
# ---------------------------------------------------------------------------


class TestListenerLifecycle:
    """Verify start/stop of the invalidation listener."""

    @pytest.mark.asyncio
    async def test_listener_not_started_no_eviction(
        self, shared_nats: InMemoryNatsBus, shared_pg: dict, config_always: DefaultCoreConfig
    ) -> None:
        """Without starting the listener, signals are not processed."""
        pod_a, reg_a = _make_pod(shared_nats, shared_pg, config_always)
        pod_b, reg_b = _make_pod(shared_nats, shared_pg, config_always)
        # NOTE: NOT calling start_invalidation_listener on reg_b

        shared_pg["e1"] = {"id": "e1", "name": "Alice", "score": 42}
        await pod_a.ensure("e1")
        await pod_b.ensure("e1")

        # Manually publish invalidation
        signal = json.dumps({"table": "test_entities", "entity_id": "e1"}).encode()
        await shared_nats.publish("threetears.cache.invalidate", signal)

        # Pod B's L1 is NOT evicted (no listener)
        assert "e1" in pod_b
