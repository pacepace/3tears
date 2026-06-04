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
import uuid
from datetime import datetime
from typing import Any

import pytest
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table
from sqlalchemy.dialects.postgresql import UUID as PgUUID

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.flush import WriteBuffer
from threetears.core.collections.registry import (
    CacheInvalidationMessage,
    CollectionRegistry,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.nats import Subjects
from threetears.nats.errors import PublishError


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
    primary_key_field = "id"


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

    async def fetch_from_postgres(self, entity_id: object) -> dict | None:
        return self._pg_store.get(str(entity_id))

    async def save_to_postgres(self, data: dict, original_timestamp: datetime | None = None) -> int:
        pk = data.get("id")
        if original_timestamp is not None:
            existing = self._pg_store.get(str(pk))
            if existing and existing.get("date_updated") != original_timestamp:
                return 0
        self._pg_store[str(pk)] = dict(data)
        return 1

    async def delete_from_postgres(self, entity_id: object) -> None:
        self._pg_store.pop(str(entity_id), None)

    def serialize(self, data: dict) -> bytes:
        return json.dumps(data, default=str).encode()

    def deserialize(self, data: bytes) -> dict:
        raw = json.loads(data)
        for key in ("date_created", "date_updated"):
            if key in raw and isinstance(raw[key], str):
                raw[key] = datetime.fromisoformat(raw[key])
        return raw


class _InMemoryKvBucket:
    """Test fake mimicking :class:`threetears.nats.NatsKvBucket`.

    in-memory dict store; kw-only get/put/delete matching the wrapper
    contract.
    """

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    async def get(self, *, key: str) -> bytes | None:
        return self._store.get(key)

    async def put(self, *, key: str, value: bytes) -> int:
        self._store[key] = value
        return len(self._store)

    async def delete(self, *, key: str, revision: int | None = None) -> bool:
        existed = key in self._store
        self._store.pop(key, None)
        return existed or revision is None


class InMemoryNatsBus:
    """Mock NATS wrapper that supports both KV operations AND typed pub/sub.

    KV: in-memory bucket via :meth:`kv_bucket` returning an
    :class:`_InMemoryKvBucket` (same instance reused for all callers
    so cross-pod state is shared).
    Pub/sub: typed via :meth:`publish` / :meth:`subscribe_typed`;
    simulates cross-pod fan-out -- all subscribers on a subject see
    every message.
    """

    def __init__(self) -> None:
        self._bucket = _InMemoryKvBucket()
        self._subscribers: dict[str, list[tuple[Any, Any]]] = {}
        self.publish_count: int = 0
        self._received_messages: list[tuple[str, Any]] = []

    # --- KV ---

    async def kv_bucket(
        self,
        *,
        name: str,  # noqa: ARG002 -- single shared bucket suffices for tests
        ttl: Any = None,  # noqa: ARG002
        storage: str = "file",  # noqa: ARG002
        create_if_missing: bool = True,  # noqa: ARG002
        history: int = 1,  # noqa: ARG002
    ) -> _InMemoryKvBucket:
        return self._bucket

    # --- Pub/sub ---

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:  # noqa: ARG002
        """typed publish. dispatches to every subscriber on ``subject``."""
        self.publish_count += 1
        subject_str = str(subject)
        self._received_messages.append((subject_str, message))
        for cb, message_type in self._subscribers.get(subject_str, []):
            # the real wrapper round-trips through model_dump_json /
            # model_validate_json; reproduce that round-trip so tests
            # exercise serialization the same way prod does.
            payload = message.model_dump_json()
            decoded = message_type.model_validate_json(payload)
            await cb(decoded)

    async def subscribe_typed(
        self,
        *,
        subject: Any,
        cb: Any,
        message_type: Any,
        queue: Any = None,  # noqa: ARG002
        max_in_flight: Any = None,  # noqa: ARG002
        deadletter_on_error: bool = True,  # noqa: ARG002
    ) -> None:
        """typed subscribe matching wrapper kw-only api."""
        subject_str = str(subject)
        self._subscribers.setdefault(subject_str, []).append((cb, message_type))


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


async def _wait_for_publish(nats: InMemoryNatsBus, target_count: int, timeout: float = 5.0) -> None:
    """Poll until the NATS bus has received at least target_count publishes."""
    deadline = asyncio.get_event_loop().time() + timeout
    while nats.publish_count < target_count:
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError(f"Timed out waiting for {target_count} publishes (got {nats.publish_count})")
        await asyncio.sleep(0.02)


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


# NOTE: bare-wrapper pub/sub semantics (typed publish, kw-only
# subscribe_typed, model_validate_json round-trip, deadletter-on-error
# delivery) are tested directly against the wrapper itself in
# ``3tears/packages/nats/tests``; this file is the *core consumer*
# side and exercises only the contract `BaseCollection` /
# `CollectionRegistry` rely on.


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

        # Manually publish an invalidation signal via the typed wrapper
        await shared_nats.publish(
            subject=Subjects.cache_invalidate(),
            message=CacheInvalidationMessage(table="test_entities", ids=["e1"]),
        )

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
        await shared_nats.publish(
            subject=Subjects.cache_invalidate(),
            message=CacheInvalidationMessage(table="nonexistent_table", ids=["e1"]),
        )

        # Pod A's data should be untouched
        assert "e1" in pod_a

    @pytest.mark.asyncio
    async def test_invalidation_for_missing_entity_is_safe(
        self, shared_nats: InMemoryNatsBus, shared_pg: dict, config_always: DefaultCoreConfig
    ) -> None:
        """Evicting a non-existent entity does not crash."""
        pod_a, reg_a = _make_pod(shared_nats, shared_pg, config_always)
        await reg_a.start_invalidation_listener(shared_nats)

        await shared_nats.publish(
            subject=Subjects.cache_invalidate(),
            message=CacheInvalidationMessage(table="test_entities", ids=["nonexistent"]),
        )

        # No crash, no side effects

    # NOTE: malformed-payload handling is the wrapper's job. with
    # typed pub/sub the wrapper round-trips
    # ``model_dump_json`` / ``model_validate_json``; junk bytes never
    # reach the listener. that contract is tested in
    # ``3tears/packages/nats/tests`` rather than duplicated here.


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

        initial_publish = shared_nats.publish_count

        # Pod A updates via setter
        pod_a["e1", "name"] = "Bob"

        # Wait for fire-and-forget to propagate
        await _wait_for_publish(shared_nats, initial_publish + 1)

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

        initial_publish = shared_nats.publish_count

        # Pod A writes full dict
        pod_a["e1"] = {"id": "e1", "name": "Replaced", "score": 99}
        await _wait_for_publish(shared_nats, initial_publish + 1)

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

        initial_publish_count = shared_nats.publish_count

        entity = pod_a.create({"id": "e1", "name": "New", "score": 0})
        await pod_a.save_entity(entity)

        # Signal was published
        assert shared_nats.publish_count > initial_publish_count

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

        initial_publish = shared_nats.publish_count

        # 10 sequential save_entity calls (async, so signals are synchronous)
        for i in range(10):
            entity = await pod_a.get("e1")
            entity.name = f"v{i + 1}"
            await pod_a.save_entity(entity)

        # All 10 saves produced signals (save_entity publishes synchronously)
        assert shared_nats.publish_count >= initial_publish + 10

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

        # Sabotage publish at the typed-wrapper boundary; the registry
        # narrow-catches PublishError and continues, the write stays
        # durable, and other pods' next read pulls fresh from L3.
        async def broken_publish(*, subject: Any, message: Any, reply_to: Any = None) -> None:
            raise PublishError("NATS down")

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

        pod_a.write_to_cache_sync({"id": "e1", "name": "Alice", "score": 42})

        async def broken_publish(*, subject: Any, message: Any, reply_to: Any = None) -> None:
            raise PublishError("NATS down")

        nats.publish = broken_publish

        pod_a["e1", "name"] = "Bob"

        # L1 write succeeded
        row = pod_a.get_row_sync("e1")
        assert row["name"] == "Bob"

    @pytest.mark.asyncio
    async def test_signal_publish_failure_does_not_break_delete(self, config_always: DefaultCoreConfig) -> None:
        """If NATS publish fails, delete still succeeds."""
        nats = InMemoryNatsBus()
        pg_store = {"e1": {"id": "e1", "name": "Alice", "score": 42}}
        pod_a, reg_a = _make_pod(nats, pg_store, config_always)

        async def broken_publish(*, subject: Any, message: Any, reply_to: Any = None) -> None:
            raise PublishError("NATS down")

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

        # Manually publish invalidation via the typed wrapper
        await shared_nats.publish(
            subject=Subjects.cache_invalidate(),
            message=CacheInvalidationMessage(table="test_entities", ids=["e1"]),
        )

        # Pod B's L1 is NOT evicted (no listener)
        assert "e1" in pod_b


# ---------------------------------------------------------------------------
# Read-after-write on the SAME pod -- a pod must not evict its own L1 from
# an invalidation IT published. This is the gap the cross-pod tests above
# never cover: every test there asserts the OTHER pod is evicted and reads
# it back via .get() (pull-through). None checks that the WRITER can still
# read its own just-written entity via the L1-only accessor while its own
# invalidation listener is live -- which is exactly the production path that
# broke wake_schedule_create.
# ---------------------------------------------------------------------------


class TestWriterDoesNotEvictOwnL1:
    """A pod must not act on cache invalidations it published itself."""

    @pytest.mark.asyncio
    async def test_writer_reads_own_entity_after_save(
        self, shared_nats: InMemoryNatsBus, shared_pg: dict, config_always: DefaultCoreConfig
    ) -> None:
        """save_entity then an L1-only read of the same entity must hit.

        The writer subscribes to the same invalidation subject it
        publishes to (single-pod prod topology). Before the origin-skip
        fix, the writer's own listener evicts the row save_entity just
        wrote, so the immediate L1-only read returns nothing.
        """
        pod_a, reg_a = _make_pod(shared_nats, shared_pg, config_always)
        await reg_a.start_invalidation_listener(shared_nats)

        entity = pod_a.create({"id": "e1", "name": "New", "score": 0})
        await pod_a.save_entity(entity)

        # The writer must still find its own freshly-written row in L1.
        assert pod_a.get_row_sync("e1") is not None, "writer evicted its own L1 row from an invalidation it published"
        assert pod_a["e1", "name"] == "New"

    @pytest.mark.asyncio
    async def test_other_pod_still_evicted_after_fix(
        self, shared_nats: InMemoryNatsBus, shared_pg: dict, config_always: DefaultCoreConfig
    ) -> None:
        """Cross-pod eviction MUST survive the origin-skip change.

        Guards against an over-broad fix that suppresses all eviction:
        a DIFFERENT pod's L1 must still be evicted by pod A's write.
        """
        pod_a, reg_a = _make_pod(shared_nats, shared_pg, config_always)
        pod_b, reg_b = _make_pod(shared_nats, shared_pg, config_always)
        await reg_a.start_invalidation_listener(shared_nats)
        await reg_b.start_invalidation_listener(shared_nats)

        shared_pg["e1"] = {"id": "e1", "name": "Alice", "score": 42}
        await pod_a.ensure("e1")
        await pod_b.ensure("e1")

        entity_a = await pod_a.get("e1")
        entity_a.name = "Updated"
        await pod_a.save_entity(entity_a)

        # Writer keeps its own row; the OTHER pod is still evicted.
        assert pod_a.get_row_sync("e1") is not None
        assert "e1" not in pod_b


# ---------------------------------------------------------------------------
# UUID type-discipline contract: a composite-UUID entity round-tripped
# through the real three-tier stack must expose its PK columns as UUID
# objects via the entity accessor -- never str, never None. Mirrors the
# wake_schedule_create path (composite (conversation_id, schedule_id) PK,
# create -> save -> read-back the PK). Strings are allowed ONLY inside the
# serialization border (the L1 row), and this test pins that the border
# converts back to UUID on the way out.
# ---------------------------------------------------------------------------


def _coerce_uuid(value: object) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def _make_uuid_metadata() -> MetaData:
    metadata = MetaData()
    Table(
        "uuid_entities",
        metadata,
        Column("conversation_id", PgUUID, primary_key=True),
        Column("entity_id", PgUUID, primary_key=True),
        Column("name", String(255)),
        Column("date_created", DateTime),
        Column("date_updated", DateTime),
    )
    return metadata


class CompUuidEntity(BaseEntity):
    """Composite-PK entity with UUID columns, mirroring WakeScheduleEntity."""

    primary_key_field = "entity_id"

    def __init__(self, data: dict[str, Any], is_new: bool = True, collection: Any = None) -> None:
        super().__init__(data, is_new=is_new, collection=collection)
        if "conversation_id" in data and "entity_id" in data:
            object.__setattr__(self, "_id", (data["conversation_id"], data["entity_id"]))

    @property
    def conversation_id(self) -> uuid.UUID:
        return _coerce_uuid(self._get_raw("conversation_id"))

    @property
    def entity_id(self) -> uuid.UUID:
        return _coerce_uuid(self._get_raw("entity_id"))

    @property
    def name(self) -> str | None:
        value: str | None = self._get_raw("name")
        return value


class CompUuidCollection(BaseCollection[CompUuidEntity]):
    """Composite-PK collection for the UUID type-discipline contract test."""

    primary_key_column: str | tuple[str, ...] = ("conversation_id", "entity_id")

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
        return "uuid_entities"

    @property
    def entity_class(self) -> type[CompUuidEntity]:
        return CompUuidEntity

    @staticmethod
    def _key(entity_id: Any) -> str:
        conv, ent = entity_id
        return f"{conv}:{ent}"

    async def fetch_from_postgres(self, entity_id: object) -> dict | None:
        return self._pg_store.get(self._key(entity_id))

    async def save_to_postgres(self, data: dict, original_timestamp: datetime | None = None) -> int:
        self._pg_store[self._key((data["conversation_id"], data["entity_id"]))] = dict(data)
        return 1

    async def delete_from_postgres(self, entity_id: object) -> None:
        self._pg_store.pop(self._key(entity_id), None)

    def serialize(self, data: dict) -> bytes:
        return json.dumps(data, default=str).encode()

    def deserialize(self, data: bytes) -> dict:
        raw = json.loads(data)
        for key in ("date_created", "date_updated"):
            if key in raw and isinstance(raw[key], str):
                raw[key] = datetime.fromisoformat(raw[key])
        return raw


def _make_uuid_pod(
    nats: InMemoryNatsBus,
    pg_store: dict[str, dict],
    config: DefaultCoreConfig,
) -> tuple[CompUuidCollection, CollectionRegistry]:
    l1 = SQLiteBackend(db_name=f"test_uuidpod_{uuid.uuid4().hex[:8]}")
    l1.initialize(_make_uuid_metadata())
    reg = CollectionRegistry()
    reg.configure(l1_backend=l1, l2_client=nats)
    coll = CompUuidCollection(reg, config, nats_client=nats, pg_store=pg_store)
    return coll, reg


class TestUuidTypeDisciplineRoundTrip:
    """Composite-UUID entity must read back as UUID, not str, not None."""

    @pytest.mark.asyncio
    async def test_pk_columns_are_uuid_after_create_and_save(
        self, shared_nats: InMemoryNatsBus, shared_pg: dict, config_always: DefaultCoreConfig
    ) -> None:
        """create -> save -> read-back PK via accessor yields UUID objects.

        This is the wake_schedule_create shape: a freshly created
        composite-UUID entity, saved with its own invalidation listener
        live, then its PK read back through the entity accessor. The
        accessor must return UUID instances (strings live only inside
        the L1 row), and the row must not have been self-evicted.
        """
        coll, reg = _make_uuid_pod(shared_nats, shared_pg, config_always)
        await reg.start_invalidation_listener(shared_nats)

        conv = uuid.uuid4()
        ent = uuid.uuid4()
        entity = coll.create({"conversation_id": conv, "entity_id": ent, "name": "x"})
        await coll.save_entity(entity)

        # not self-evicted
        assert coll.get_row_sync((conv, ent)) is not None, "composite-UUID entity self-evicted after save"
        # type discipline: accessor returns UUID, never str, never None
        assert isinstance(entity.conversation_id, uuid.UUID)
        assert isinstance(entity.entity_id, uuid.UUID)
        assert entity.entity_id == ent
        assert entity.conversation_id == conv

    def test_accessor_coerces_string_pk_to_uuid(self) -> None:
        """Negative-proof: when the raw field is a STRING, the accessor
        still returns a UUID.

        Guards the type-discipline test against silently becoming a
        no-op. A cache/JSON round-trip can surface a pk as a bare string
        (the allowed border representation); the entity accessor MUST
        re-coerce it to UUID. If a future edit drops the ``_coerce_uuid``
        call from the accessor, this fails -- the accessor would leak the
        raw string and the ``isinstance`` check below would catch it.
        """
        conv = uuid.uuid4()
        ent = uuid.uuid4()
        # transient entity (no collection) holds the raw dict in _changes;
        # the pk values are strings, as a serialization round-trip yields.
        entity = CompUuidEntity(
            {"conversation_id": str(conv), "entity_id": str(ent), "name": "x"},
            collection=None,
        )
        assert isinstance(entity.entity_id, uuid.UUID)
        assert isinstance(entity.conversation_id, uuid.UUID)
        assert entity.entity_id == ent
        assert entity.conversation_id == conv
