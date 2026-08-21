"""Tests for BaseCollection three-tier caching."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.base import _CACHED_AT_COLUMN
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.flush import WriteBuffer
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import ConcurrentModificationError
from threetears.nats.errors import KvError


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


class TestL2BucketResolutionDegradesGracefully:
    """_ensure_kv()'s KvError must degrade the same way a get/put/delete KvError does.

    regression coverage for a bug where ``kv = await self._ensure_kv()`` sat
    outside each method's ``try/except KvError`` block -- so a bucket-open
    failure (e.g. right after a NATS outage begins, before this collection's
    bucket handle has ever been resolved) propagated uncaught instead of
    degrading to None/False like every other L2 transport failure, breaking
    the "L2 is best-effort, L3 is source of truth" contract these methods'
    own docstrings promise. exercised through the public API (get/save_entity/
    delete), not the private _get_from_l2/_save_to_l2/_delete_from_l2 hooks,
    matching this suite's existing convention (cleanup 2A-4f drove test
    assertions on tier storage through the public API; 2A-4m-final zeroed
    SLF001 across the workspace).
    """

    @pytest.mark.asyncio
    async def test_get_degrades_on_bucket_open_failure(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """L1 miss + L2 bucket-open KvError still resolves via L3 fallback."""
        nats = _make_nats_mock()
        nats.kv_bucket = AsyncMock(side_effect=KvError("bucket open failed"))
        l3_rows = {"e1": {"id": "e1", "name": "Alice", "score": 10}}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)

        entity = await coll.get("e1")

        assert entity is not None
        assert entity.name == "Alice"

    @pytest.mark.asyncio
    async def test_save_entity_degrades_on_bucket_open_failure(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """L2 bucket-open KvError during save still durably writes to L3."""
        nats = _make_nats_mock()
        nats.kv_bucket = AsyncMock(side_effect=KvError("bucket open failed"))
        l3_rows: dict[str, dict] = {}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)

        entity = coll.create({"id": "e1", "name": "Alice", "score": 10})
        await coll.save_entity(entity)

        assert "e1" in l3_rows
        assert entity.is_dirty is False

    @pytest.mark.asyncio
    async def test_delete_degrades_on_bucket_open_failure(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """L2 bucket-open KvError during delete still durably removes from L3."""
        nats = _make_nats_mock()
        nats.kv_bucket = AsyncMock(side_effect=KvError("bucket open failed"))
        l3_rows = {"e1": {"id": "e1", "name": "Alice", "score": 10}}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)

        result = await coll.delete("e1")

        assert result is True
        assert "e1" not in l3_rows

    @pytest.mark.asyncio
    async def test_non_kverror_from_bucket_open_still_propagates(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """a genuine programming error in bucket resolution still propagates loudly."""
        nats = _make_nats_mock()
        nats.kv_bucket = AsyncMock(side_effect=TypeError("bad bucket name"))
        l3_rows = {"e1": {"id": "e1", "name": "Alice", "score": 10}}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)

        with pytest.raises(TypeError):
            await coll.get("e1")


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


class TestCorruptL2EntryFallsThroughToL3:
    """A cached value that will not decode is a cache miss, not a failed read.

    L2 is a cache and L3 is authoritative, so a corrupt entry must not be able to break a
    lookup that L3 could answer. The two alternatives were both worse and both plausible:
    letting the decode error propagate turns one poisoned key into a failed read, and handing
    back the undecoded value gives the caller a `str` in a column it declared as `datetime`,
    which fails far away and usually at the database border.

    Three packages had each written their own rehydration before this moved to the base, and
    all three answered this question differently. This is the answer.
    """

    @pytest.mark.asyncio
    async def test_a_corrupt_timestamp_in_l2_is_served_from_l3_instead(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        class _Timestamped(StubCollection):
            datetime_columns = frozenset({"date_created"})

        nats = _make_nats_mock()
        good = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)
        l3_rows = {"e1": {"id": "e1", "name": "Alice", "score": 10, "date_created": good}}
        coll = _Timestamped(registry, config_always, nats_client=nats, l3_rows=l3_rows)

        # A poisoned L2 entry: right key, right shape, one value that will not decode.
        nats.store[coll.l2_key("e1")] = json.dumps(
            {"id": "e1", "name": "FromCache", "score": 99, "date_created": "not-a-timestamp"}
        ).encode()

        entity = await coll.get("e1")

        assert entity is not None, "a corrupt cache entry broke a read that L3 could serve"
        assert entity.name == "Alice", "the corrupt L2 row was served instead of falling through"

    @pytest.mark.asyncio
    async def test_a_decodable_l2_entry_is_still_served_from_l2(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """The negative half. Without it, a base that always fell through to L3 would pass.

        That failure mode is invisible in behaviour and expensive in load, which is exactly the
        kind that survives.
        """

        class _Timestamped(StubCollection):
            datetime_columns = frozenset({"date_created"})

        nats = _make_nats_mock()
        l3_rows = {"e1": {"id": "e1", "name": "Alice", "score": 10}}
        coll = _Timestamped(registry, config_always, nats_client=nats, l3_rows=l3_rows)
        nats.store[coll.l2_key("e1")] = json.dumps(
            {"id": "e1", "name": "FromCache", "score": 99, "date_created": "2026-07-26T12:00:00+00:00"}
        ).encode()

        entity = await coll.get("e1")

        assert entity is not None
        assert entity.name == "FromCache", "a healthy L2 entry was bypassed"


class TestCacheAgeStampOnLowerTierReads:
    """A row's provenance is stamped wherever it arrives from a lower tier.

    The stamp records when a row was last obtained from a LOWER tier, never
    when it was last touched locally. That distinction is the mechanism: a row
    renewed by every local edit would be immortal, which is precisely the
    staleness this is meant to bound.
    """

    @staticmethod
    def _stored_stamp(backend: SQLiteBackend, entity_id: str) -> float | None:
        """Read the stamp straight from SQLite, since every read strips it."""
        conn = backend.get_connection()
        row = conn.execute(
            f'SELECT "{_CACHED_AT_COLUMN}" FROM test_entities WHERE id = ?',
            (entity_id,),
        ).fetchone()
        return None if row is None else row[0]

    @pytest.mark.asyncio
    async def test_a_pull_through_from_l3_stamps_the_l1_row(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig, l1_backend: SQLiteBackend
    ) -> None:
        nats = _make_nats_mock()
        l3_rows = {"p1": {"id": "p1", "name": "FromL3", "score": 1}}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)

        assert await coll.get("p1") is not None

        assert self._stored_stamp(l1_backend, "p1") is not None

    @pytest.mark.asyncio
    async def test_a_pull_through_from_l2_stamps_the_l1_row(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig, l1_backend: SQLiteBackend
    ) -> None:
        nats = _make_nats_mock()
        nats.store["test_entities.p2"] = json.dumps({"id": "p2", "name": "FromL2", "score": 2}).encode()
        coll = StubCollection(registry, config_always, nats_client=nats)

        assert await coll.get("p2") is not None

        assert self._stored_stamp(l1_backend, "p2") is not None

    @pytest.mark.asyncio
    async def test_a_locally_authored_write_is_not_stamped(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig, l1_backend: SQLiteBackend
    ) -> None:
        """This pod wrote it; no lower tier has served it yet."""
        nats = _make_nats_mock()
        coll = StubCollection(registry, config_always, nats_client=nats)

        coll.write_to_cache_sync({"id": "p3", "name": "Local", "score": 3})

        assert self._stored_stamp(l1_backend, "p3") is None

    @pytest.mark.asyncio
    async def test_reload_entity_stamps_the_row_it_fetched_from_l3(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig, l1_backend: SQLiteBackend
    ) -> None:
        """Reached via ``BaseEntity.reload()``, not via get/ensure -- and it still counts.

        It is not a pull-through, but it fetches from L3 and writes the result
        into L1, so the row's provenance is a lower tier. An unstamped reload
        would read as locally authored, and locally authored rows never expire.
        """
        nats = _make_nats_mock()
        l3_rows = {"p5": {"id": "p5", "name": "FromL3", "score": 5}}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)

        entity = coll.create({"id": "p5", "name": "Local", "score": 0})
        await coll.reload_entity(entity)

        assert self._stored_stamp(l1_backend, "p5") is not None

    @pytest.mark.asyncio
    async def test_the_stamp_never_reaches_the_caller(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        nats = _make_nats_mock()
        l3_rows = {"p4": {"id": "p4", "name": "FromL3", "score": 4}}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)

        entity = await coll.get("p4")
        assert entity is not None
        assert _CACHED_AT_COLUMN not in entity.to_dict()

        row = coll.get_row_sync("p4")
        assert row is not None
        assert _CACHED_AT_COLUMN not in row

        ensured = await coll.ensure("p4")
        assert ensured is not None
        assert _CACHED_AT_COLUMN not in ensured


class TestL1MaxAgePolicy:
    """Who gets a staleness bound, and who is structurally refused one."""

    def test_expiry_is_off_until_a_collection_asks(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        coll = StubCollection(registry, config_always, nats_client=_make_nats_mock())
        coll.l3_pool = object()
        assert coll.l1_max_age_seconds is None

    def test_a_configured_collection_with_an_l3_gets_its_bound(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        registry.set_l1_max_age("test_entities", 3600.0)
        coll = StubCollection(registry, config_always, nats_client=_make_nats_mock())
        coll.l3_pool = object()
        assert coll.l1_max_age_seconds == 3600.0

    def test_a_collection_with_no_l3_is_refused_a_bound_even_when_configured(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """The regression this chunk exists to prevent.

        With no L3 there is nothing to pull through from, so an expired row is
        not a miss that repairs itself -- it is a deletion. A caller that reads
        absence and writes goes on to replace live state with a fresh empty
        row, and nothing raises. Refusing the bound here is what makes that
        impossible rather than merely unlikely.
        """
        registry.set_l1_max_age("test_entities", 1.0)
        coll = StubCollection(registry, config_always, nats_client=_make_nats_mock())
        coll.l3_pool = None
        assert coll.l1_max_age_seconds is None

    def test_a_nonpositive_bound_is_refused_at_configuration_time(self, registry: CollectionRegistry) -> None:
        with pytest.raises(ValueError, match="positive"):
            registry.set_l1_max_age("test_entities", 0)

    @pytest.mark.asyncio
    async def test_an_expired_row_pulls_through_and_picks_up_a_peers_write(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig, l1_backend: SQLiteBackend
    ) -> None:
        """End to end, staged the way the failure actually happens.

        The residue this bounds is a peer's write whose invalidation never
        arrived. That write DID reach L2, which is shared -- only this pod's L1
        copy is stale. So expiry is checked against what L2 holds, not L3: a
        pull-through consults L2 first and stops there on a hit.
        """
        registry.set_l1_max_age("test_entities", 30.0)
        nats = _make_nats_mock()
        l3_rows = {"m1": {"id": "m1", "name": "Original", "score": 1}}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)
        # a real L3 handle, because a collection without one is refused a bound
        # outright -- that gate is the subject of the tests above.
        coll.l3_pool = object()

        assert await coll.get("m1") is not None
        # a peer writes; its invalidation is lost, so this pod's L1 keeps the
        # old row while the shared tier already holds the new one.
        nats.store["test_entities.m1"] = json.dumps({"id": "m1", "name": "PeerWrote", "score": 2}).encode()
        conn = l1_backend.get_connection()
        conn.execute(f'UPDATE test_entities SET "{_CACHED_AT_COLUMN}" = ? WHERE id = ?', (0.0, "m1"))

        entity = await coll.get("m1")

        assert entity is not None
        assert entity.name == "PeerWrote"

    @pytest.mark.asyncio
    async def test_without_the_bound_the_same_stale_row_is_served_forever(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig, l1_backend: SQLiteBackend
    ) -> None:
        """The control for the test above: no bound, no recovery.

        Same staging, expiry off. This is the unbounded staleness the chunk
        exists to close, and it is worth pinning so the previous test cannot
        quietly start passing for some other reason.
        """
        nats = _make_nats_mock()
        l3_rows = {"m2": {"id": "m2", "name": "Original", "score": 1}}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)
        # a real L3 handle, because a collection without one is refused a bound
        # outright -- that gate is the subject of the tests above.
        coll.l3_pool = object()

        assert await coll.get("m2") is not None
        nats.store["test_entities.m2"] = json.dumps({"id": "m2", "name": "PeerWrote", "score": 2}).encode()
        conn = l1_backend.get_connection()
        conn.execute(f'UPDATE test_entities SET "{_CACHED_AT_COLUMN}" = ? WHERE id = ?', (0.0, "m2"))

        entity = await coll.get("m2")

        assert entity is not None
        assert entity.name == "Original"


class TestExpiryDoesNotBreakNonRepairingReads:
    """Expiry converts a stale hit into a miss, which is only safe where a miss repairs.

    Three reviewers converged on this independently: routing the reporting
    readers through the expiring path made a field write vanish and an entity
    handle answer wrongly, because those callers treat a miss as "not cached"
    and do not fall back.
    """

    @staticmethod
    def _age_out(backend: SQLiteBackend, entity_id: str) -> None:
        conn = backend.get_connection()
        conn.execute(f'UPDATE test_entities SET "{_CACHED_AT_COLUMN}" = ? WHERE id = ?', (0.0, entity_id))

    @pytest.mark.asyncio
    async def test_a_field_write_survives_an_aged_out_row(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig, l1_backend: SQLiteBackend
    ) -> None:
        """``collection[id, "field"] = v`` must not silently drop the write.

        ``__setitem__`` reads the row back through ``get_row_sync`` and skips
        propagation on ``None``. If that read expired the row, the write landed
        nowhere and nothing raised.
        """
        registry.set_l1_max_age("test_entities", 30.0)
        nats = _make_nats_mock()
        l3_rows = {"w1": {"id": "w1", "name": "Original", "score": 1}}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)
        coll.l3_pool = object()

        assert await coll.get("w1") is not None
        self._age_out(l1_backend, "w1")

        coll["w1", "name"] = "Written"

        assert coll.get_row_sync("w1") is not None
        assert coll.get_row_sync("w1")["name"] == "Written"

    @pytest.mark.asyncio
    async def test_an_entity_handle_still_reads_its_fields_across_the_bound(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig, l1_backend: SQLiteBackend
    ) -> None:
        """A handle resolves attributes through the reporting readers.

        Expiring under it made ``to_dict()`` raise and field reads answer for a
        row that is present in every tier.
        """
        registry.set_l1_max_age("test_entities", 30.0)
        nats = _make_nats_mock()
        l3_rows = {"w2": {"id": "w2", "name": "Held", "score": 7}}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)
        coll.l3_pool = object()

        entity = await coll.get("w2")
        assert entity is not None
        self._age_out(l1_backend, "w2")

        assert entity.name == "Held"
        assert entity.to_dict()["score"] == 7

    @pytest.mark.asyncio
    async def test_the_repairing_read_still_expires(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig, l1_backend: SQLiteBackend
    ) -> None:
        """The control: narrowing expiry must not have turned it off.

        ``ensure`` repairs by pulling through, so it is one of the two callers
        that still applies the bound.
        """
        registry.set_l1_max_age("test_entities", 30.0)
        nats = _make_nats_mock()
        l3_rows = {"w3": {"id": "w3", "name": "Original", "score": 1}}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)
        coll.l3_pool = object()

        assert await coll.get("w3") is not None
        nats.store["test_entities.w3"] = json.dumps({"id": "w3", "name": "PeerWrote", "score": 2}).encode()
        self._age_out(l1_backend, "w3")

        refreshed = await coll.ensure("w3")
        assert refreshed is not None
        assert refreshed["name"] == "PeerWrote"


class TestAnOlderL1BackendStillWorks:
    """``L1Backend`` is a published Protocol, so implementers live out of repo.

    Passing ``max_age_seconds=`` on every read -- even as ``None`` -- raises
    ``TypeError`` on an implementation that predates the parameter, which is
    every cached read for a feature nobody opted into. Both in-repo backends
    accept the kwarg, so nothing else in the suite can catch a regression here.
    """

    class _PreExpiryBackend:
        """An L1 backend whose reads predate the expiry parameters."""

        def __init__(self) -> None:
            self.rows: dict[Any, dict[str, Any]] = {}

        def select_by_id(
            self,
            table: str,  # noqa: ARG002
            entity_id: Any,
            primary_key: Any = "id",  # noqa: ARG002
            columns: Any = None,  # noqa: ARG002
        ) -> dict[str, Any] | None:
            key = entity_id if not isinstance(entity_id, tuple) else entity_id[0]
            return self.rows.get(key)

    def test_a_read_without_a_bound_omits_the_new_kwargs(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        backend = self._PreExpiryBackend()
        backend.rows["old1"] = {"id": "old1", "name": "kept"}
        coll = StubCollection(registry, config_always, nats_client=_make_nats_mock())
        coll._l1 = backend  # noqa: SLF001 - substituting the tier under test

        row = coll.get_row_sync("old1")

        assert row is not None
        assert row["name"] == "kept"

    def test_a_configured_bound_does_not_reach_a_reporting_read_either(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """Even opted in, the reporting reads never pass the new parameters.

        ``get_row_sync`` reports whether a row is cached; it does not repair a
        miss, so it does not expire and has no reason to ask. That keeps an
        older backend working for the majority of reads on a collection that
        HAS opted in -- only the repairing paths reach the parameter.
        """
        registry.set_l1_max_age("test_entities", 30.0)
        backend = self._PreExpiryBackend()
        backend.rows["old2"] = {"id": "old2", "name": "kept"}
        coll = StubCollection(registry, config_always, nats_client=_make_nats_mock())
        coll._l1 = backend  # noqa: SLF001 - substituting the tier under test
        coll.l3_pool = object()

        assert coll.get_row_sync("old2") is not None

    @pytest.mark.asyncio
    async def test_only_a_repairing_read_under_a_bound_reaches_the_new_kwargs(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """And there the loud failure is the right one.

        A deployment that configures a bound against a backend too old to
        honour it should hear about it; the alternative is a bound that
        silently never fires, which is the staleness it was asking to remove.
        """
        registry.set_l1_max_age("test_entities", 30.0)
        backend = self._PreExpiryBackend()
        backend.rows["old3"] = {"id": "old3", "name": "kept"}
        coll = StubCollection(registry, config_always, nats_client=_make_nats_mock())
        coll._l1 = backend  # noqa: SLF001 - substituting the tier under test
        coll.l3_pool = object()

        with pytest.raises(TypeError):
            await coll.ensure("old3")


class TestTheBoundReachesTheSubscriptReadPath:
    """``collection[id]`` is the primary read, and the bound has to reach it.

    ``_resolve_row`` repairs a miss by pulling through, so it is a repairing
    caller. It briefly read L1 through ``get_row_sync``, which is a REPORTING
    read and does not expire -- so a stale row was returned before the
    pull-through below it could ever run, and every subscript read served it
    indefinitely on a collection that had opted into a bound.

    Nothing caught that: the collection-tier tests covered ``ensure`` and the
    reporting readers, and the backend-tier tests covered the predicate. The
    wiring between them was the gap.
    """

    @staticmethod
    def _age_out(backend: SQLiteBackend, entity_id: str) -> None:
        conn = backend.get_connection()
        conn.execute(f'UPDATE test_entities SET "{_CACHED_AT_COLUMN}" = ? WHERE id = ?', (0.0, entity_id))

    @pytest.mark.asyncio
    async def test_the_entity_subscript_expires_and_pulls_through(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig, l1_backend: SQLiteBackend
    ) -> None:
        """``collection[id]`` past the bound reloads instead of serving the stale row.

        :param registry: collection registry bound to the L1 backend
        :ptype registry: CollectionRegistry
        :param config_always: flush-always config
        :ptype config_always: DefaultCoreConfig
        :param l1_backend: the SQLite L1 backend, used to age the row out
        :ptype l1_backend: SQLiteBackend
        :return: nothing
        :rtype: None
        """
        registry.set_l1_max_age("test_entities", 30.0)
        nats = _make_nats_mock()
        l3_rows = {"s1": {"id": "s1", "name": "Original", "score": 1}}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)
        coll.l3_pool = object()

        assert await coll.get("s1") is not None
        nats.store["test_entities.s1"] = json.dumps({"id": "s1", "name": "PeerWrote", "score": 2}).encode()
        self._age_out(l1_backend, "s1")

        assert coll["s1"].name == "PeerWrote"

    @pytest.mark.asyncio
    async def test_the_field_subscript_expires_and_pulls_through(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig, l1_backend: SQLiteBackend
    ) -> None:
        """``collection[id, "field"]`` resolves through the same row read.

        :param registry: collection registry bound to the L1 backend
        :ptype registry: CollectionRegistry
        :param config_always: flush-always config
        :ptype config_always: DefaultCoreConfig
        :param l1_backend: the SQLite L1 backend, used to age the row out
        :ptype l1_backend: SQLiteBackend
        :return: nothing
        :rtype: None
        """
        registry.set_l1_max_age("test_entities", 30.0)
        nats = _make_nats_mock()
        l3_rows = {"s2": {"id": "s2", "name": "Original", "score": 1}}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)
        coll.l3_pool = object()

        assert await coll.get("s2") is not None
        nats.store["test_entities.s2"] = json.dumps({"id": "s2", "name": "PeerWrote", "score": 2}).encode()
        self._age_out(l1_backend, "s2")

        assert coll["s2", "name"] == "PeerWrote"

    @pytest.mark.asyncio
    async def test_a_row_inside_the_bound_is_served_without_pulling_through(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """The control: expiring the subscript path must not make every read a pull-through.

        :param registry: collection registry bound to the L1 backend
        :ptype registry: CollectionRegistry
        :param config_always: flush-always config
        :ptype config_always: DefaultCoreConfig
        :return: nothing
        :rtype: None
        """
        registry.set_l1_max_age("test_entities", 30.0)
        nats = _make_nats_mock()
        l3_rows = {"s3": {"id": "s3", "name": "Original", "score": 1}}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)
        coll.l3_pool = object()

        assert await coll.get("s3") is not None
        nats.store["test_entities.s3"] = json.dumps({"id": "s3", "name": "PeerWrote", "score": 2}).encode()

        # Not aged out: the L1 row is inside its window and must win.
        assert coll["s3"].name == "Original"

    @pytest.mark.asyncio
    async def test_an_unbounded_collection_keeps_serving_the_old_row(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig, l1_backend: SQLiteBackend
    ) -> None:
        """No opt-in, no expiry -- the subscript path must not have become always-expiring.

        :param registry: collection registry bound to the L1 backend
        :ptype registry: CollectionRegistry
        :param config_always: flush-always config
        :ptype config_always: DefaultCoreConfig
        :param l1_backend: the SQLite L1 backend, used to age the row out
        :ptype l1_backend: SQLiteBackend
        :return: nothing
        :rtype: None
        """
        nats = _make_nats_mock()
        l3_rows = {"s4": {"id": "s4", "name": "Original", "score": 1}}
        coll = StubCollection(registry, config_always, nats_client=nats, l3_rows=l3_rows)
        coll.l3_pool = object()

        assert await coll.get("s4") is not None
        nats.store["test_entities.s4"] = json.dumps({"id": "s4", "name": "PeerWrote", "score": 2}).encode()
        self._age_out(l1_backend, "s4")

        assert coll["s4"].name == "Original"
