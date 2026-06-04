"""Tests for BaseCollection three-tier caching."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.flush import WriteBuffer
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import ConcurrentModificationError


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
    """Concrete collection for testing."""

    def __init__(
        self,
        registry: CollectionRegistry,
        config: DefaultCoreConfig,
        nats_client: AsyncMock | None = None,
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
                return 0  # Optimistic lock failure
        self._pg_store[str(pk)] = dict(data)
        return 1

    async def delete_from_postgres(self, entity_id: object) -> None:
        self._pg_store.pop(str(entity_id), None)

    def serialize(self, data: dict) -> bytes:
        return json.dumps(data, default=str).encode()

    def deserialize(self, data: bytes) -> dict:
        return json.loads(data)


def _make_nats_mock() -> AsyncMock:
    """Create a typed-wrapper NATS client mock with in-memory KV bucket.

    matches :class:`threetears.nats.NatsClient` /
    :class:`threetears.nats.NatsKvBucket` shapes: ``kv_bucket`` is
    awaited and returns a bucket whose ``get`` / ``put`` / ``delete``
    are kw-only. ``store`` is hung off the client for assertion
    convenience.
    """
    store: dict[str, bytes] = {}

    async def _get(*, key: str) -> bytes | None:
        return store.get(key)

    async def _put(*, key: str, value: bytes) -> int:
        store[key] = value
        return len(store)

    async def _delete(*, key: str, revision: int | None = None) -> bool:  # noqa: ARG001
        existed = key in store
        store.pop(key, None)
        return existed or revision is None

    bucket = AsyncMock()
    bucket.get = AsyncMock(side_effect=_get)
    bucket.put = AsyncMock(side_effect=_put)
    bucket.delete = AsyncMock(side_effect=_delete)

    nats = AsyncMock()
    nats.kv_bucket = AsyncMock(return_value=bucket)
    nats.publish = AsyncMock()
    nats.subscribe_typed = AsyncMock()
    nats.store = store  # expose for assertions
    nats.bucket = bucket  # expose for assertions
    return nats


@pytest.fixture()
def l1_backend() -> SQLiteBackend:
    b = SQLiteBackend(db_name=f"test_coll_{uuid.uuid4().hex[:8]}")
    b.initialize(_make_metadata())
    yield b
    from threetears.core._bridge import drain, shutdown

    drain()
    shutdown()
    b.reset()


@pytest.fixture()
def registry(l1_backend: SQLiteBackend) -> CollectionRegistry:
    reg = CollectionRegistry()
    reg.configure(l1_backend=l1_backend)
    return reg


@pytest.fixture()
def config_always() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


@pytest.fixture()
def config_deferred() -> DefaultCoreConfig:
    return DefaultCoreConfig(
        collection_flush="ON_CHECKPOINT",
        collection_flush_tables="test_entities",
    )


class TestThreeTierGet:
    """Tests for BaseCollection.get() three-tier read."""

    @pytest.mark.asyncio
    async def test_l1_hit(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        """L1 hit returns entity without touching L2/L3."""
        nats = _make_nats_mock()
        pg_store = {"e1": {"id": "e1", "name": "Alice", "score": 100}}
        coll = StubCollection(registry, config_always, nats_client=nats, pg_store=pg_store)

        # Pre-populate L1
        coll.write_to_cache_sync({"id": "e1", "name": "Alice", "score": 100})

        entity = await coll.get("e1")

        assert entity is not None
        assert entity.name == "Alice"
        # L2 should NOT have been called (L1 hit)
        nats.bucket.get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_l1_miss_l2_hit(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        """L1 miss, L2 hit promotes to L1 and returns entity."""
        nats = _make_nats_mock()
        l2_data = {"id": "e2", "name": "Bob", "score": 50}
        nats.store["test_entities.e2"] = json.dumps(l2_data).encode()
        coll = StubCollection(registry, config_always, nats_client=nats)

        entity = await coll.get("e2")

        assert entity is not None
        assert entity.name == "Bob"
        # Should be promoted to L1
        l1_row = coll.get_row_sync("e2")
        assert l1_row is not None
        assert l1_row["name"] == "Bob"

    @pytest.mark.asyncio
    async def test_l1_l2_miss_l3_hit(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        """L1+L2 miss, L3 hit promotes to both caches."""
        nats = _make_nats_mock()
        pg_store = {"e3": {"id": "e3", "name": "Carol", "score": 75}}
        coll = StubCollection(registry, config_always, nats_client=nats, pg_store=pg_store)

        entity = await coll.get("e3")

        assert entity is not None
        assert entity.name == "Carol"
        # Promoted to L1
        l1_row = coll.get_row_sync("e3")
        assert l1_row is not None
        # Promoted to L2
        assert "test_entities.e3" in nats.store

    @pytest.mark.asyncio
    async def test_all_miss_returns_none(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        """All tiers miss returns None."""
        nats = _make_nats_mock()
        coll = StubCollection(registry, config_always, nats_client=nats)

        entity = await coll.get("nonexistent")

        assert entity is None


class TestSaveEntity:
    """Tests for BaseCollection.save_entity()."""

    @pytest.mark.asyncio
    async def test_immediate_save(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        """Immediate save writes to L3 first, then caches."""
        nats = _make_nats_mock()
        pg_store: dict[str, dict] = {}
        coll = StubCollection(registry, config_always, nats_client=nats, pg_store=pg_store)

        entity = coll.create({"id": "e1", "name": "Alice", "score": 10})
        await coll.save_entity(entity)

        # Written to postgres
        assert "e1" in pg_store
        assert pg_store["e1"]["name"] == "Alice"
        # Written to L1
        l1_row = coll.get_row_sync("e1")
        assert l1_row is not None
        # Written to L2
        assert "test_entities.e1" in nats.store
        # Entity is clean
        assert entity.is_dirty is False
        assert entity.is_new is False

    @pytest.mark.asyncio
    async def test_deferred_save(self, registry: CollectionRegistry, config_deferred: DefaultCoreConfig) -> None:
        """Deferred save writes L1+L2+buffer, skips L3."""
        nats = _make_nats_mock()
        buf = WriteBuffer()
        pg_store: dict[str, dict] = {}
        coll = StubCollection(registry, config_deferred, nats_client=nats, write_buffer=buf, pg_store=pg_store)

        entity = coll.create({"id": "e1", "name": "Alice", "score": 10})
        await coll.save_entity(entity)

        # NOT written to postgres
        assert "e1" not in pg_store
        # Written to L1
        l1_row = coll.get_row_sync("e1")
        assert l1_row is not None
        # Written to L2
        assert "test_entities.e1" in nats.store
        # In write buffer
        assert buf.pending_count() == 1
        # Entity is clean
        assert entity.is_dirty is False

    @pytest.mark.asyncio
    async def test_optimistic_lock_failure(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """Optimistic lock failure raises ConcurrentModificationError."""
        nats = _make_nats_mock()
        ts_old = datetime(2025, 1, 1, tzinfo=UTC)
        ts_new = datetime(2025, 6, 1, tzinfo=UTC)
        pg_store = {
            "e1": {
                "id": "e1",
                "name": "Alice",
                "score": 10,
                "date_updated": ts_new,
            }
        }
        coll = StubCollection(registry, config_always, nats_client=nats, pg_store=pg_store)

        # Load entity with old timestamp
        coll.write_to_cache_sync(
            {"id": "e1", "name": "Alice", "score": 10, "date_updated": ts_old},
        )
        entity = await coll.get("e1")
        assert entity is not None
        entity.original_date_updated = ts_old

        entity.name = "Alice Updated"

        with pytest.raises(ConcurrentModificationError):
            await coll.save_entity(entity)


class TestReloadEntity:
    """Tests for BaseCollection.reload_entity()."""

    @pytest.mark.asyncio
    async def test_reload_from_l3(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        """Reload fetches from L3 and updates caches."""
        nats = _make_nats_mock()
        pg_store = {"e1": {"id": "e1", "name": "Original", "score": 10}}
        coll = StubCollection(registry, config_always, nats_client=nats, pg_store=pg_store)

        entity = coll.create({"id": "e1", "name": "Original", "score": 10})
        await coll.save_entity(entity)

        # Modify in postgres directly
        pg_store["e1"]["name"] = "Updated"

        await coll.reload_entity(entity)

        assert entity.name == "Updated"
        # L1 should be updated
        l1_row = coll.get_row_sync("e1")
        assert l1_row is not None
        assert l1_row["name"] == "Updated"

    @pytest.mark.asyncio
    async def test_reload_entity_not_found(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """Reload raises ValueError if entity not in L3."""
        coll = StubCollection(registry, config_always)

        entity = coll.create({"id": "missing", "name": "Ghost", "score": 0})

        with pytest.raises(ValueError, match="not found in storage"):
            await coll.reload_entity(entity)


class TestDelete:
    """Tests for BaseCollection.delete()."""

    @pytest.mark.asyncio
    async def test_delete_from_all_tiers(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        """Delete removes from all tiers."""
        nats = _make_nats_mock()
        pg_store = {"e1": {"id": "e1", "name": "Alice", "score": 10}}
        coll = StubCollection(registry, config_always, nats_client=nats, pg_store=pg_store)

        # Populate all tiers
        entity = await coll.get("e1")
        assert entity is not None

        result = await coll.delete("e1")

        assert result is True
        # Removed from L3
        assert "e1" not in pg_store
        # Removed from L1
        assert coll.get_row_sync("e1") is None
        # Removed from L2
        assert "test_entities.e1" not in nats.store


class TestCreate:
    """Tests for BaseCollection.create()."""

    @pytest.mark.asyncio
    async def test_create_returns_new_entity(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """Create returns a new entity with is_new=True."""
        coll = StubCollection(registry, config_always)

        entity = coll.create({"id": "new1", "name": "New Entity", "score": 0})

        assert entity.is_new is True
        assert entity.is_dirty is True
        assert entity.id == "new1"
        assert entity.name == "New Entity"


class TestFieldAccessors:
    """Tests for sync field accessors."""

    def test_get_field_sync(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        coll = StubCollection(registry, config_always)
        coll.write_to_cache_sync({"id": "e1", "name": "Alice", "score": 42})

        result = coll.get_field_sync("e1", "name")

        assert result == "Alice"

    def test_get_field_sync_missing_entity(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        from threetears.core.cache import MISSING

        coll = StubCollection(registry, config_always)

        result = coll.get_field_sync("nonexistent", "name")

        assert result is MISSING

    def test_set_field_sync(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        coll = StubCollection(registry, config_always)
        coll.write_to_cache_sync({"id": "e1", "name": "Alice", "score": 42})

        result = coll.set_field_sync("e1", "name", "Bob")

        assert result is True
        row = coll.get_row_sync("e1")
        assert row["name"] == "Bob"

    def test_set_field_sync_missing_entity(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        coll = StubCollection(registry, config_always)

        result = coll.set_field_sync("nonexistent", "name", "Bob")

        assert result is False

    def test_get_row_sync(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        coll = StubCollection(registry, config_always)
        coll.write_to_cache_sync({"id": "e1", "name": "Alice", "score": 42})

        row = coll.get_row_sync("e1")

        assert row is not None
        assert row["name"] == "Alice"
        assert row["score"] == 42

    def test_get_row_sync_missing(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        coll = StubCollection(registry, config_always)

        row = coll.get_row_sync("nonexistent")

        assert row is None

    def test_write_to_cache_sync(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        coll = StubCollection(registry, config_always)

        result = coll.write_to_cache_sync({"id": "e1", "name": "Alice", "score": 42})

        assert result is True
        row = coll.get_row_sync("e1")
        assert row is not None
        assert row["name"] == "Alice"

    def test_exists_in_cache_sync(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        coll = StubCollection(registry, config_always)

        assert coll.exists_in_cache_sync("e1") is False

        coll.write_to_cache_sync({"id": "e1", "name": "Alice", "score": 42})

        assert coll.exists_in_cache_sync("e1") is True


class TestNoL1Backend:
    """Tests for collection without L1 backend configured."""

    @pytest.fixture()
    def no_l1_registry(self) -> CollectionRegistry:
        return CollectionRegistry()

    @pytest.mark.asyncio
    async def test_get_from_l3_without_l1(
        self, no_l1_registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """Can still read from L3 when L1 is None."""
        pg_store = {"e1": {"id": "e1", "name": "Alice", "score": 10}}
        coll = StubCollection(no_l1_registry, config_always, pg_store=pg_store)

        entity = await coll.get("e1")

        assert entity is not None
        assert entity.name == "Alice"

    def test_field_accessors_return_missing_without_l1(
        self, no_l1_registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        from threetears.core.cache import MISSING

        coll = StubCollection(no_l1_registry, config_always)

        assert coll.get_field_sync("e1", "name") is MISSING
        assert coll.set_field_sync("e1", "name", "x") is False
        assert coll.get_row_sync("e1") is None
        assert coll.write_to_cache_sync({"id": "e1"}) is False
        assert coll.exists_in_cache_sync("e1") is False


class TestSubscriptGetterPullThrough:
    """Tests for __getitem__ transparent three-tier pull-through."""

    @pytest.mark.asyncio
    async def test_getitem_entity_l1_hit(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        """collection[id] returns entity from L1 without touching L2/L3."""
        nats = _make_nats_mock()
        coll = StubCollection(registry, config_always, nats_client=nats)
        coll.write_to_cache_sync({"id": "e1", "name": "Alice", "score": 42})

        entity = coll["e1"]

        assert entity.name == "Alice"
        assert entity.score == 42
        nats.bucket.get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_getitem_field_l1_hit(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        """collection[id, field] returns field value from L1."""
        coll = StubCollection(registry, config_always)
        coll.write_to_cache_sync({"id": "e1", "name": "Alice", "score": 42})

        assert coll["e1", "name"] == "Alice"
        assert coll["e1", "score"] == 42

    def test_getitem_entity_pulls_through_l3(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """collection[id] transparently pulls through L3 on L1 miss."""
        pg_store = {"e1": {"id": "e1", "name": "Alice", "score": 100}}
        coll = StubCollection(registry, config_always, pg_store=pg_store)

        entity = coll["e1"]

        assert entity.name == "Alice"
        # Should now be in L1
        l1_row = coll.get_row_sync("e1")
        assert l1_row is not None
        assert l1_row["name"] == "Alice"

    def test_getitem_field_pulls_through_l3(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """collection[id, field] transparently pulls through L3 on L1 miss."""
        pg_store = {"e1": {"id": "e1", "name": "Alice", "score": 100}}
        coll = StubCollection(registry, config_always, pg_store=pg_store)

        assert coll["e1", "name"] == "Alice"

    def test_getitem_pulls_through_l2(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        """collection[id] pulls through L2 on L1 miss."""
        nats = _make_nats_mock()
        l2_data = {"id": "e1", "name": "Bob", "score": 50}
        nats.store["test_entities.e1"] = json.dumps(l2_data).encode()
        coll = StubCollection(registry, config_always, nats_client=nats)

        entity = coll["e1"]

        assert entity.name == "Bob"
        # Promoted to L1
        l1_row = coll.get_row_sync("e1")
        assert l1_row is not None

    def test_getitem_raises_keyerror_for_missing(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """collection[id] raises KeyError if entity doesn't exist anywhere."""
        coll = StubCollection(registry, config_always)

        with pytest.raises(KeyError, match="entity not found"):
            _ = coll["nonexistent"]

    def test_getitem_field_raises_keyerror_for_missing_entity(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """collection[id, field] raises KeyError if entity doesn't exist."""
        coll = StubCollection(registry, config_always)

        with pytest.raises(KeyError, match="entity not found"):
            _ = coll["nonexistent", "name"]

    def test_getitem_field_raises_keyerror_for_missing_field(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """collection[id, field] raises KeyError if field doesn't exist on entity."""
        pg_store = {"e1": {"id": "e1", "name": "Alice"}}
        coll = StubCollection(registry, config_always, pg_store=pg_store)

        with pytest.raises(KeyError, match="field not found"):
            _ = coll["e1", "nonexistent_field"]

    def test_contains_checks_l1_only(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        """'in' operator checks L1 only, does not pull through."""
        pg_store = {"e1": {"id": "e1", "name": "Alice", "score": 10}}
        coll = StubCollection(registry, config_always, pg_store=pg_store)

        # In L3 but not L1
        assert "e1" not in coll

        # After pull-through, it's in L1
        _ = coll["e1"]
        assert "e1" in coll


class TestSubscriptSetterPropagation:
    """Tests for __setitem__ three-tier write propagation."""

    @pytest.mark.asyncio
    async def test_field_setter_updates_l1(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """collection[id, field] = value updates L1 immediately."""
        coll = StubCollection(registry, config_always)
        coll.write_to_cache_sync({"id": "e1", "name": "Alice", "score": 42})

        coll["e1", "name"] = "Bob"

        row = coll.get_row_sync("e1")
        assert row["name"] == "Bob"

    @pytest.mark.asyncio
    async def test_field_setter_propagates_to_l2(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """collection[id, field] = value propagates to L2 non-blocking."""
        nats = _make_nats_mock()
        coll = StubCollection(registry, config_always, nats_client=nats)
        coll.write_to_cache_sync({"id": "e1", "name": "Alice", "score": 42})

        coll["e1", "name"] = "Bob"

        # Yield to the event loop so the fire-and-forget task can complete
        await asyncio.sleep(0.1)

        assert "test_entities.e1" in nats.store
        l2_data = json.loads(nats.store["test_entities.e1"])
        assert l2_data["name"] == "Bob"

    @pytest.mark.asyncio
    async def test_field_setter_immediate_l3_write(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """With ALWAYS strategy, setter propagates to L3 non-blocking."""
        nats = _make_nats_mock()
        pg_store: dict[str, dict] = {}
        coll = StubCollection(registry, config_always, nats_client=nats, pg_store=pg_store)
        coll.write_to_cache_sync({"id": "e1", "name": "Alice", "score": 42})

        coll["e1", "name"] = "Bob"

        await asyncio.sleep(0.1)

        assert "e1" in pg_store
        assert pg_store["e1"]["name"] == "Bob"

    @pytest.mark.asyncio
    async def test_field_setter_deferred_l3_write(
        self, registry: CollectionRegistry, config_deferred: DefaultCoreConfig
    ) -> None:
        """With deferred strategy, setter buffers for L3 instead of writing immediately."""
        nats = _make_nats_mock()
        buf = WriteBuffer()
        pg_store: dict[str, dict] = {}
        coll = StubCollection(registry, config_deferred, nats_client=nats, write_buffer=buf, pg_store=pg_store)
        coll.write_to_cache_sync({"id": "e1", "name": "Alice", "score": 42})

        coll["e1", "name"] = "Bob"

        await asyncio.sleep(0.1)

        # NOT in L3
        assert "e1" not in pg_store
        # But IS in L2
        assert "test_entities.e1" in nats.store
        # And IS in write buffer
        assert buf.pending_count() == 1

    @pytest.mark.asyncio
    async def test_dict_setter_propagates(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        """collection[id] = data_dict propagates to L1, L2, and L3."""
        nats = _make_nats_mock()
        pg_store: dict[str, dict] = {}
        coll = StubCollection(registry, config_always, nats_client=nats, pg_store=pg_store)

        coll["e1"] = {"id": "e1", "name": "Alice", "score": 99}

        await asyncio.sleep(0.1)

        # L1
        row = coll.get_row_sync("e1")
        assert row is not None
        assert row["name"] == "Alice"
        # L2
        assert "test_entities.e1" in nats.store
        # L3
        assert "e1" in pg_store
        assert pg_store["e1"]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_setter_updates_date_updated(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """Setter sets date_updated on the propagated data."""
        nats = _make_nats_mock()
        pg_store: dict[str, dict] = {}
        coll = StubCollection(registry, config_always, nats_client=nats, pg_store=pg_store)
        coll.write_to_cache_sync({"id": "e1", "name": "Alice", "score": 42})

        before = datetime.now(UTC)
        coll["e1", "name"] = "Bob"
        await asyncio.sleep(0.1)

        assert "e1" in pg_store
        du = pg_store["e1"].get("date_updated")
        assert du is not None
        # date_updated should be close to now
        assert du >= before

    def test_setter_rejects_non_dict(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        """collection[id] = non_dict raises TypeError."""
        coll = StubCollection(registry, config_always)

        with pytest.raises(TypeError, match="requires a dict"):
            coll["e1"] = "not a dict"


class TestMultiPodSimulation:
    """Simulate two pods sharing L2 (NATS KV) + L3, with separate L1 caches."""

    def _make_pod(
        self,
        nats: AsyncMock,
        pg_store: dict[str, dict],
        config: DefaultCoreConfig,
        write_buffer: WriteBuffer | None = None,
    ) -> StubCollection:
        """Create a collection representing one pod (own L1, shared L2+L3)."""
        l1 = SQLiteBackend(db_name=f"test_pod_{uuid.uuid4().hex[:8]}")
        l1.initialize(_make_metadata())
        reg = CollectionRegistry()
        reg.configure(l1_backend=l1)
        return StubCollection(reg, config, nats_client=nats, write_buffer=write_buffer, pg_store=pg_store)

    @pytest.mark.asyncio
    async def test_write_on_pod_a_visible_on_pod_b_via_l2(self, config_always: DefaultCoreConfig) -> None:
        """Data written on pod A is visible on pod B through shared L2."""
        nats = _make_nats_mock()
        pg_store: dict[str, dict] = {}
        pod_a = self._make_pod(nats, pg_store, config_always)
        pod_b = self._make_pod(nats, pg_store, config_always)

        # Pod A creates and saves entity
        entity = pod_a.create({"id": "e1", "name": "Alice", "score": 42})
        await pod_a.save_entity(entity)

        # Pod B reads — should hit L2 (shared NATS KV)
        entity_b = await pod_b.get("e1")
        assert entity_b is not None
        assert entity_b.name == "Alice"

    @pytest.mark.asyncio
    async def test_stale_l1_on_pod_b_after_pod_a_update(self, config_always: DefaultCoreConfig) -> None:
        """Pod B's L1 cache becomes stale after pod A updates via setter.

        This demonstrates the cache coherence gap that signaling will fix.
        """
        nats = _make_nats_mock()
        pg_store: dict[str, dict] = {}
        pod_a = self._make_pod(nats, pg_store, config_always)
        pod_b = self._make_pod(nats, pg_store, config_always)

        # Both pods load the same entity
        pg_store["e1"] = {"id": "e1", "name": "Alice", "score": 42}
        await pod_a.ensure("e1")
        await pod_b.ensure("e1")

        # Pod A updates via setter
        pod_a["e1", "name"] = "Bob"
        await asyncio.sleep(0.1)  # Let propagation complete

        # Pod B's L1 is stale (still "Alice")
        stale_name = pod_b.get_field_sync("e1", "name")
        assert stale_name == "Alice"

        # But L2 has the update (shared NATS KV)
        l2_raw = nats.store.get("test_entities.e1")
        assert l2_raw is not None
        l2_data = json.loads(l2_raw)
        assert l2_data["name"] == "Bob"

        # Pod B invalidating its L1 and re-reading picks up the change
        await pod_b.invalidate_cache("e1")
        entity_b = await pod_b.get("e1")
        assert entity_b is not None
        assert entity_b.name == "Bob"

    @pytest.mark.asyncio
    async def test_setter_propagation_reaches_l3(self, config_always: DefaultCoreConfig) -> None:
        """Setter with ALWAYS strategy writes through to shared L3."""
        nats = _make_nats_mock()
        pg_store: dict[str, dict] = {}
        pod_a = self._make_pod(nats, pg_store, config_always)
        pod_b = self._make_pod(nats, pg_store, config_always)

        # Seed data
        pg_store["e1"] = {"id": "e1", "name": "Alice", "score": 42}
        await pod_a.ensure("e1")

        # Pod A updates via setter
        pod_a["e1", "name"] = "Updated"
        await asyncio.sleep(0.1)

        # L3 (shared postgres) has the update
        assert pg_store["e1"]["name"] == "Updated"

        # Pod B can see it via L3 even after its caches are cleared
        await pod_b.invalidate_cache("e1")
        entity_b = await pod_b.get("e1")
        assert entity_b is not None
        assert entity_b.name == "Updated"

    @pytest.mark.asyncio
    async def test_setter_deferred_does_not_reach_l3(self, config_deferred: DefaultCoreConfig) -> None:
        """Setter with deferred strategy buffers but doesn't write L3."""
        nats = _make_nats_mock()
        buf = WriteBuffer()
        pg_store = {"e1": {"id": "e1", "name": "Alice", "score": 42}}
        pod_a = self._make_pod(nats, pg_store, config_deferred, write_buffer=buf)
        await pod_a.ensure("e1")

        pod_a["e1", "name"] = "Deferred"
        await asyncio.sleep(0.1)

        # L3 still has old value
        assert pg_store["e1"]["name"] == "Alice"
        # L2 has new value
        l2_data = json.loads(nats.store["test_entities.e1"])
        assert l2_data["name"] == "Deferred"
        # Write buffer has pending entry
        assert buf.pending_count() == 1


class TestInvalidateCache:
    """Tests for BaseCollection.invalidate_cache()."""

    @pytest.mark.asyncio
    async def test_invalidate_removes_l1_and_l2(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        nats = _make_nats_mock()
        pg_store = {"e1": {"id": "e1", "name": "Alice", "score": 10}}
        coll = StubCollection(registry, config_always, nats_client=nats, pg_store=pg_store)

        # Populate caches
        await coll.get("e1")
        assert coll.get_row_sync("e1") is not None
        assert "test_entities.e1" in nats.store

        await coll.invalidate_cache("e1")

        assert coll.get_row_sync("e1") is None
        assert "test_entities.e1" not in nats.store


class TestL3PoolAccessor:
    """verify the public ``l3_pool`` attribute exposes the pool the
    registry handed the collection at construction time.

    the hub's ad-hoc-SQL extension seam depends on this identity
    relation: if it drifts (say, by copying or wrapping), hub code that
    relies on ``self.l3_pool.fetch(...)`` sees a different pool than
    the one the collection uses internally, which silently breaks
    transactions and connection-lifetime assumptions.
    """

    def test_l3_pool_returns_registry_pool_by_default(self, config_always: DefaultCoreConfig) -> None:
        """collection.l3_pool is the same object the registry holds."""
        sentinel_pool = object()
        reg = CollectionRegistry()
        reg.configure(l3_pool=sentinel_pool)
        coll = StubCollection(reg, config_always)
        assert coll.l3_pool is sentinel_pool

    def test_l3_pool_respects_per_collection_override(self, config_always: DefaultCoreConfig) -> None:
        """per-collection pool override wins over the registry default."""
        default_pool = object()
        override_pool = object()
        reg = CollectionRegistry()
        reg.configure(l3_pool=default_pool)
        # override must be registered BEFORE BaseCollection.__init__ reads it;
        # the collection's auto-register call happens last so we pre-stage
        # the override on the registry by hand.
        reg.bind_table("test_entities", l3_pool=override_pool)
        coll = StubCollection(reg, config_always)
        assert coll.l3_pool is override_pool

    def test_l3_pool_none_when_registry_has_no_pool(self, config_always: DefaultCoreConfig) -> None:
        """collection.l3_pool is None when the registry has no pool.

        callers MUST guard with ``if self.l3_pool is not None`` — this
        test pins that contract so the absent-pool case does not
        silently regress to a misleading truthy value.
        """
        reg = CollectionRegistry()
        coll = StubCollection(reg, config_always)
        assert coll.l3_pool is None
