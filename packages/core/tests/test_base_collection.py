"""Tests for BaseCollection three-tier caching."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table

from threetears.core.backends.sql import SqlL3Backend
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
        l3_rows: dict[str, dict] | None = None,
    ) -> None:
        self._l3_rows = l3_rows if l3_rows is not None else {}
        super().__init__(registry, config, nats_client, write_buffer)

    @property
    def table_name(self) -> str:
        return "test_entities"

    @property
    def entity_class(self) -> type[StubEntity]:
        return StubEntity

    async def fetch_from_store(self, entity_id: object) -> dict | None:
        return self._l3_rows.get(str(entity_id))

    async def save_to_store(self, data: dict, original_timestamp: datetime | None = None) -> int:
        pk = data.get("id")
        if original_timestamp is not None:
            existing = self._l3_rows.get(str(pk))
            if existing and existing.get("date_updated") != original_timestamp:
                return 0  # Optimistic lock failure
        self._l3_rows[str(pk)] = dict(data)
        return 1

    async def delete_from_store(self, entity_id: object) -> None:
        self._l3_rows.pop(str(entity_id), None)

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
        l3_rows = {"e1": {"id": "e1", "name": "Alice", "score": 100}}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)

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
        l3_rows = {"e3": {"id": "e3", "name": "Carol", "score": 75}}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)

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
        l3_rows: dict[str, dict] = {}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)

        entity = coll.create({"id": "e1", "name": "Alice", "score": 10})
        await coll.save_entity(entity)

        # Written to postgres
        assert "e1" in l3_rows
        assert l3_rows["e1"]["name"] == "Alice"
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
        l3_rows: dict[str, dict] = {}
        coll = StubCollection(registry, config_deferred, nats_client=nats, write_buffer=buf, l3_rows=l3_rows)

        entity = coll.create({"id": "e1", "name": "Alice", "score": 10})
        await coll.save_entity(entity)

        # NOT written to postgres
        assert "e1" not in l3_rows
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
        l3_rows = {
            "e1": {
                "id": "e1",
                "name": "Alice",
                "score": 10,
                "date_updated": ts_new,
            }
        }
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)

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
        l3_rows = {"e1": {"id": "e1", "name": "Original", "score": 10}}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)

        entity = coll.create({"id": "e1", "name": "Original", "score": 10})
        await coll.save_entity(entity)

        # Modify in postgres directly
        l3_rows["e1"]["name"] = "Updated"

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
        l3_rows = {"e1": {"id": "e1", "name": "Alice", "score": 10}}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)

        # Populate all tiers
        entity = await coll.get("e1")
        assert entity is not None

        result = await coll.delete("e1")

        assert result is True
        # Removed from L3
        assert "e1" not in l3_rows
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
        l3_rows = {"e1": {"id": "e1", "name": "Alice", "score": 10}}
        coll = StubCollection(no_l1_registry, config_always, l3_rows=l3_rows)

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
        l3_rows = {"e1": {"id": "e1", "name": "Alice", "score": 100}}
        coll = StubCollection(registry, config_always, l3_rows=l3_rows)

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
        l3_rows = {"e1": {"id": "e1", "name": "Alice", "score": 100}}
        coll = StubCollection(registry, config_always, l3_rows=l3_rows)

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
        l3_rows = {"e1": {"id": "e1", "name": "Alice"}}
        coll = StubCollection(registry, config_always, l3_rows=l3_rows)

        with pytest.raises(KeyError, match="field not found"):
            _ = coll["e1", "nonexistent_field"]

    def test_contains_checks_l1_only(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        """'in' operator checks L1 only, does not pull through."""
        l3_rows = {"e1": {"id": "e1", "name": "Alice", "score": 10}}
        coll = StubCollection(registry, config_always, l3_rows=l3_rows)

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
        l3_rows: dict[str, dict] = {}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)
        coll.write_to_cache_sync({"id": "e1", "name": "Alice", "score": 42})

        coll["e1", "name"] = "Bob"

        await asyncio.sleep(0.1)

        assert "e1" in l3_rows
        assert l3_rows["e1"]["name"] == "Bob"

    @pytest.mark.asyncio
    async def test_field_setter_deferred_l3_write(
        self, registry: CollectionRegistry, config_deferred: DefaultCoreConfig
    ) -> None:
        """With deferred strategy, setter buffers for L3 instead of writing immediately."""
        nats = _make_nats_mock()
        buf = WriteBuffer()
        l3_rows: dict[str, dict] = {}
        coll = StubCollection(registry, config_deferred, nats_client=nats, write_buffer=buf, l3_rows=l3_rows)
        coll.write_to_cache_sync({"id": "e1", "name": "Alice", "score": 42})

        coll["e1", "name"] = "Bob"

        await asyncio.sleep(0.1)

        # NOT in L3
        assert "e1" not in l3_rows
        # But IS in L2
        assert "test_entities.e1" in nats.store
        # And IS in write buffer
        assert buf.pending_count() == 1

    @pytest.mark.asyncio
    async def test_dict_setter_propagates(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        """collection[id] = data_dict propagates to L1, L2, and L3."""
        nats = _make_nats_mock()
        l3_rows: dict[str, dict] = {}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)

        coll["e1"] = {"id": "e1", "name": "Alice", "score": 99}

        await asyncio.sleep(0.1)

        # L1
        row = coll.get_row_sync("e1")
        assert row is not None
        assert row["name"] == "Alice"
        # L2
        assert "test_entities.e1" in nats.store
        # L3
        assert "e1" in l3_rows
        assert l3_rows["e1"]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_setter_updates_date_updated(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """Setter sets date_updated on the propagated data."""
        nats = _make_nats_mock()
        l3_rows: dict[str, dict] = {}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)
        coll.write_to_cache_sync({"id": "e1", "name": "Alice", "score": 42})

        before = datetime.now(UTC)
        coll["e1", "name"] = "Bob"
        await asyncio.sleep(0.1)

        assert "e1" in l3_rows
        du = l3_rows["e1"].get("date_updated")
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
        l3_rows: dict[str, dict],
        config: DefaultCoreConfig,
        write_buffer: WriteBuffer | None = None,
    ) -> StubCollection:
        """Create a collection representing one pod (own L1, shared L2+L3)."""
        l1 = SQLiteBackend(db_name=f"test_pod_{uuid.uuid4().hex[:8]}")
        l1.initialize(_make_metadata())
        reg = CollectionRegistry()
        reg.configure(l1_backend=l1)
        return StubCollection(reg, config, nats_client=nats, write_buffer=write_buffer, l3_rows=l3_rows)

    @pytest.mark.asyncio
    async def test_write_on_pod_a_visible_on_pod_b_via_l2(self, config_always: DefaultCoreConfig) -> None:
        """Data written on pod A is visible on pod B through shared L2."""
        nats = _make_nats_mock()
        l3_rows: dict[str, dict] = {}
        pod_a = self._make_pod(nats, l3_rows, config_always)
        pod_b = self._make_pod(nats, l3_rows, config_always)

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
        l3_rows: dict[str, dict] = {}
        pod_a = self._make_pod(nats, l3_rows, config_always)
        pod_b = self._make_pod(nats, l3_rows, config_always)

        # Both pods load the same entity
        l3_rows["e1"] = {"id": "e1", "name": "Alice", "score": 42}
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
        l3_rows: dict[str, dict] = {}
        pod_a = self._make_pod(nats, l3_rows, config_always)
        pod_b = self._make_pod(nats, l3_rows, config_always)

        # Seed data
        l3_rows["e1"] = {"id": "e1", "name": "Alice", "score": 42}
        await pod_a.ensure("e1")

        # Pod A updates via setter
        pod_a["e1", "name"] = "Updated"
        await asyncio.sleep(0.1)

        # L3 (shared postgres) has the update
        assert l3_rows["e1"]["name"] == "Updated"

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
        l3_rows = {"e1": {"id": "e1", "name": "Alice", "score": 42}}
        pod_a = self._make_pod(nats, l3_rows, config_deferred, write_buffer=buf)
        await pod_a.ensure("e1")

        pod_a["e1", "name"] = "Deferred"
        await asyncio.sleep(0.1)

        # L3 still has old value
        assert l3_rows["e1"]["name"] == "Alice"
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
        l3_rows = {"e1": {"id": "e1", "name": "Alice", "score": 10}}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)

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
    relation: if it drifts, hub code that relies on
    ``self.l3_pool.fetch(...)`` sees a different pool than the one the
    collection uses internally, which silently breaks transactions and
    connection-lifetime assumptions.

    L3B-03 introduced a deliberate normalization: the registry wraps a raw
    asyncpg-shaped pool in a :class:`SqlL3Backend` (so the collection CRUD
    lifecycle can route through the structured ``DurableStore`` ops). The
    wrapper is **identity-stable** (the registry resolves the SAME wrapper
    object every time) and **delegates** ``fetch``/``fetchrow``/``execute``/
    ``acquire``/``transaction`` straight to the wrapped pool — so the hub's
    raw-SQL seam and connection-lifetime contract are preserved. A pool that
    already satisfies ``DurableStore`` passes through un-wrapped.
    """

    def test_l3_pool_returns_registry_pool_by_default(self, config_always: DefaultCoreConfig) -> None:
        """collection.l3_pool is the SAME backend the registry resolves, wrapping the configured pool."""
        sentinel_pool = object()
        reg = CollectionRegistry()
        reg.configure(l3_pool=sentinel_pool)
        coll = StubCollection(reg, config_always)
        # identity-stable: the collection sees the exact backend the registry resolves
        assert coll.l3_pool is reg.get_l3_pool("test_entities")
        # the raw pool the collection hands the hub's ad-hoc-SQL seam is the configured one
        assert isinstance(coll.l3_pool, SqlL3Backend)
        assert coll.l3_pool._pool is sentinel_pool  # noqa: SLF001 -- introspect the wrapper's raw pool

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
        assert isinstance(coll.l3_pool, SqlL3Backend)
        assert coll.l3_pool._pool is override_pool  # noqa: SLF001 -- introspect the wrapper's raw pool

    def test_l3_pool_none_when_registry_has_no_pool(self, config_always: DefaultCoreConfig) -> None:
        """collection.l3_pool is None when the registry has no pool.

        callers MUST guard with ``if self.l3_pool is not None`` — this
        test pins that contract so the absent-pool case does not
        silently regress to a misleading truthy value.
        """
        reg = CollectionRegistry()
        coll = StubCollection(reg, config_always)
        assert coll.l3_pool is None


# ---------------------------------------------------------------------------
# IMPROVEMENT 1 — l2_key is grammar-safe by default
# ---------------------------------------------------------------------------


class CompositeStubEntity(BaseEntity):
    primary_key_field = "a"


class CompositeStubCollection(BaseCollection[CompositeStubEntity]):
    """composite-pk concrete collection, for l2_key shape tests."""

    primary_key_column = ("a", "b")

    @property
    def table_name(self) -> str:
        return "test_entities"

    @property
    def entity_class(self) -> type[CompositeStubEntity]:
        return CompositeStubEntity

    async def fetch_from_store(self, entity_id: object) -> dict | None:  # pragma: no cover - unused
        return None

    async def save_to_store(self, data: dict, original_timestamp: datetime | None = None) -> int:  # pragma: no cover
        return 1

    async def delete_from_store(self, entity_id: object) -> None:  # pragma: no cover - unused
        return None

    def serialize(self, data: dict) -> bytes:
        return json.dumps(data, default=str).encode()

    def deserialize(self, data: bytes) -> dict:
        return json.loads(data)


_HEX = set("0123456789abcdef")


class TestL2KeyGrammarSafe:
    """l2_key keeps grammar-safe pks readable and hashes out-of-grammar ones."""

    def test_safe_single_pk_unchanged(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        """a grammar-safe single pk keeps its readable key (backward-compatible)."""
        coll = StubCollection(registry, config_always)
        assert coll.l2_key("e1") == "test_entities.e1"
        # uuid-shaped pk (dashes are in-grammar) stays readable too.
        uid = "550e8400-e29b-41d4-a716-446655440000"
        assert coll.l2_key(uid) == f"test_entities.{uid}"

    def test_safe_composite_pk_unchanged(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        """a grammar-safe composite pk keeps the readable underscore-joined body."""
        coll = CompositeStubCollection(registry, config_always)
        assert coll.l2_key(("scope1", "grp7")) == "test_entities.scope1_grp7"

    def test_out_of_grammar_pk_is_hashed_and_valid(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """a colon-bearing (out-of-grammar) pk yields a valid SHA-256-hashed key."""
        coll = StubCollection(registry, config_always)
        prefix, _, body = coll.l2_key("cust:story:main:scene.md").partition(".")
        assert prefix == "test_entities"
        assert ":" not in body
        assert len(body) == 64 and set(body) <= _HEX

    def test_space_pk_is_hashed_and_valid(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        """a space (out-of-grammar) yields a valid hashed key, never a raw space."""
        coll = StubCollection(registry, config_always)
        body = coll.l2_key("my file.md").partition(".")[2]
        assert " " not in body
        assert len(body) == 64 and set(body) <= _HEX

    def test_naive_colon_to_eq_collision_is_avoided(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """two pks a naive ':'->'=' replace would collide map to DISTINCT keys."""
        coll = StubCollection(registry, config_always)
        # both collapse to "x=y=z" under a ':'->'=' replace.
        assert coll.l2_key("x=y:z") != coll.l2_key("x:y=z")

    def test_deterministic(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        """same pk always yields the same key (safe and hashed paths alike)."""
        coll = StubCollection(registry, config_always)
        assert coll.l2_key("e1") == coll.l2_key("e1")
        assert coll.l2_key("cust:story:f.md") == coll.l2_key("cust:story:f.md")


# ---------------------------------------------------------------------------
# IMPROVEMENT 2 — l2_cas_mutate (generic L1+L2 atomic read-modify-write)
# ---------------------------------------------------------------------------


class _CasKvBucket:
    """CAS-capable typed-wrapper KV bucket stand-in, matching ``NatsKvBucket``.

    optional ``conflict_first`` makes the first ``update``/``create``/``delete``
    return a conflict (``None``/``False``) regardless of revision, to drive the
    retry branch; ``always_conflict`` forces every write to conflict, to drive
    retry-budget exhaustion.
    """

    def __init__(self, *, conflict_first: bool = False, always_conflict: bool = False) -> None:
        self._store: dict[str, tuple[bytes, int]] = {}
        self._seq = 0
        self._conflict_first = conflict_first
        self._always_conflict = always_conflict

    def _next(self) -> int:
        self._seq += 1
        return self._seq

    def _should_conflict(self) -> bool:
        if self._always_conflict:
            return True
        if self._conflict_first:
            self._conflict_first = False
            return True
        return False

    async def get(self, *, key: str) -> bytes | None:
        entry = self._store.get(key)
        return entry[0] if entry is not None else None

    async def get_entry(self, *, key: str) -> tuple[bytes, int] | None:
        return self._store.get(key)

    async def put(self, *, key: str, value: bytes) -> int:
        rev = self._next()
        self._store[key] = (value, rev)
        return rev

    async def create(self, *, key: str, value: bytes) -> int | None:
        if self._should_conflict() or key in self._store:
            return None
        rev = self._next()
        self._store[key] = (value, rev)
        return rev

    async def update(self, *, key: str, value: bytes, revision: int) -> int | None:
        if self._should_conflict():
            return None
        entry = self._store.get(key)
        if entry is None or entry[1] != revision:
            return None
        rev = self._next()
        self._store[key] = (value, rev)
        return rev

    async def delete(self, *, key: str, revision: int | None = None) -> bool:
        if self._should_conflict():
            return False
        entry = self._store.get(key)
        if entry is None:
            return True
        if revision is not None and entry[1] != revision:
            return False
        del self._store[key]
        return True


def _make_cas_nats(bucket: _CasKvBucket) -> AsyncMock:
    """wrap a ``_CasKvBucket`` in a NATS-client mock for collection wiring."""
    nats = AsyncMock()
    nats.kv_bucket = AsyncMock(return_value=bucket)
    nats.publish = AsyncMock()
    nats.subscribe_typed = AsyncMock()
    nats.bucket = bucket
    return nats


def _append_member(member: str) -> "object":
    """build a mutate callback that appends ``member`` to a CSV ``name`` field.

    the stub L1 schema has no list column, so the "set" is modelled as a
    comma-separated string in the ``name`` column — the CAS contract is
    identical (read current → compute next → write), and the test reads
    members back by splitting on ``,``.
    """

    def _mutate(row: dict | None) -> tuple[str, dict | None]:
        if row is None:
            return "upsert", {"id": "r1", "name": member}
        members = [m for m in row.get("name", "").split(",") if m]
        if member in members:
            return "noop", None
        members.append(member)
        return "upsert", {**row, "name": ",".join(members)}

    return _mutate


def _members(row: dict | None) -> list[str]:
    """read the CSV ``name`` field back as a member list."""
    if row is None:
        return []
    return [m for m in row.get("name", "").split(",") if m]


class TestL2CasMutate:
    """the generic L1+L2 compare-and-swap read-modify-write primitive."""

    @pytest.mark.asyncio
    async def test_upsert_creates_when_absent(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """an upsert on an absent value create-if-absents it into L2 + L1."""
        bucket = _CasKvBucket()
        coll = StubCollection(registry, config_always, nats_client=_make_cas_nats(bucket))

        await coll.l2_cas_mutate("r1", _append_member("conn-1"))

        raw = await bucket.get(key="test_entities.r1")
        assert raw is not None
        assert _members(json.loads(raw)) == ["conn-1"]
        # L1 reconciled.
        assert _members(coll.get_row_sync("r1")) == ["conn-1"]

    @pytest.mark.asyncio
    async def test_upsert_updates_existing(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """a second upsert CAS-updates the existing value."""
        bucket = _CasKvBucket()
        coll = StubCollection(registry, config_always, nats_client=_make_cas_nats(bucket))

        await coll.l2_cas_mutate("r1", _append_member("conn-1"))
        await coll.l2_cas_mutate("r1", _append_member("conn-2"))

        raw = await bucket.get(key="test_entities.r1")
        assert _members(json.loads(raw)) == ["conn-1", "conn-2"]

    @pytest.mark.asyncio
    async def test_noop_returns_without_writing(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """a 'noop' action writes nothing and publishes no invalidation."""
        bucket = _CasKvBucket()
        nats = _make_cas_nats(bucket)
        coll = StubCollection(registry, config_always, nats_client=nats)

        await coll.l2_cas_mutate("r1", lambda _row: ("noop", None))

        assert await bucket.get(key="test_entities.r1") is None
        nats.publish.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delete_action_removes_value(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """a 'delete' action CAS-deletes the value from L2 and L1."""
        bucket = _CasKvBucket()
        coll = StubCollection(registry, config_always, nats_client=_make_cas_nats(bucket))
        await coll.l2_cas_mutate("r1", _append_member("conn-1"))
        assert coll.get_row_sync("r1") is not None  # L1 populated by the upsert

        await coll.l2_cas_mutate("r1", lambda _row: ("delete", None))

        assert await bucket.get(key="test_entities.r1") is None
        assert coll.get_row_sync("r1") is None

    @pytest.mark.asyncio
    async def test_retry_on_single_conflict_then_succeeds(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """one CAS conflict then success — the loop retries and lands the write."""
        # seed the value so the write path is update (which conflicts once).
        bucket = _CasKvBucket(conflict_first=True)
        await bucket.put(key="test_entities.r1", value=json.dumps({"id": "r1", "name": "seed"}).encode())
        coll = StubCollection(registry, config_always, nats_client=_make_cas_nats(bucket))

        await coll.l2_cas_mutate("r1", _append_member("conn-1"))

        raw = await bucket.get(key="test_entities.r1")
        assert _members(json.loads(raw)) == ["seed", "conn-1"]

    @pytest.mark.asyncio
    async def test_retry_budget_exhaustion_raises(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """an always-conflicting bucket exhausts the budget and raises CME."""
        bucket = _CasKvBucket(always_conflict=True)
        await bucket.put(key="test_entities.r1", value=json.dumps({"id": "r1", "name": "seed"}).encode())
        coll = StubCollection(registry, config_always, nats_client=_make_cas_nats(bucket))

        with pytest.raises(ConcurrentModificationError):
            await coll.l2_cas_mutate("r1", _append_member("conn-1"), max_retries=3)

    @pytest.mark.asyncio
    async def test_l1_only_fallback_upsert_and_delete(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """with no NATS wired, the primitive degrades to an L1 read-modify-write."""
        # nats_client=None -> _ensure_kv() resolves to None -> L1-only fallback path.
        coll = StubCollection(registry, config_always, nats_client=None)

        await coll.l2_cas_mutate("r1", _append_member("conn-1"))
        await coll.l2_cas_mutate("r1", _append_member("conn-2"))
        assert _members(coll.get_row_sync("r1")) == ["conn-1", "conn-2"]

        def _drop_conn1(row: dict | None) -> tuple[str, dict | None]:
            members = [m for m in _members(row) if m != "conn-1"]
            return "upsert", {**row, "name": ",".join(members)}

        await coll.l2_cas_mutate("r1", _drop_conn1)
        assert _members(coll.get_row_sync("r1")) == ["conn-2"]

        await coll.l2_cas_mutate("r1", lambda _row: ("delete", None))
        assert coll.get_row_sync("r1") is None


class TestStorageAgnosticL3Contract:
    """The L3 durable tier is a PLUGGABLE backend — storage-agnostic.

    A collection reaches L3 only through three methods: ``fetch_from_store`` /
    ``save_to_store`` / ``delete_from_store``. It never assumes SQL. ``StubCollection``
    backs them with a plain in-memory dict; the SQL backend backs them with
    Postgres; a ``GitL3Backend`` backs them with files in a git working tree —
    all three are first-class. This pins that contract: a NON-SQL L3 backend
    drives the full three-tier save → evict → pull-through → delete round-trip,
    so a git L3 implementing the same three methods slots in unchanged.
    """

    @pytest.mark.asyncio
    async def test_non_sql_l3_backend_drives_full_three_tier_roundtrip(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        nats = _make_nats_mock()
        # The ENTIRE durable store is a plain dict — no SQL anywhere. This is the
        # exact shape a GitL3Backend presents (fetch/save/delete a record by pk).
        l3_rows: dict[str, dict] = {}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)

        # save → save_to_store persists to the non-SQL durable tier
        await coll.save_entity(StubEntity({"id": "g1", "name": "Gitish", "score": 7}, is_new=True))
        assert l3_rows["g1"]["name"] == "Gitish"  # durable tier holds it; no SQL involved

        # evict L1 + L2 so the next read MUST fall through to the non-SQL L3
        await coll.invalidate_cache("g1")
        assert coll.get_row_sync("g1") is None  # L1 evicted

        # get → L1+L2 miss → fetch_from_store (the non-SQL backend) serves it + promotes up
        got = await coll.get("g1")
        assert got is not None and got.name == "Gitish"
        assert coll.get_row_sync("g1") is not None  # promoted back to L1
        assert "test_entities.g1" in nats.store  # promoted to L2

        # delete → delete_from_store removes it from the non-SQL durable tier
        await coll.delete("g1")
        assert "g1" not in l3_rows
