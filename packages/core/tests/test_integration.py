"""Integration tests: full CRUD through L1 -> L2 -> L3 with a concrete entity/collection."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from threetears.core import (
    BaseCollection,
    BaseEntity,
    CollectionRegistry,
    ConcurrentModificationError,
    DefaultCoreConfig,
)
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.flush import WriteBuffer
from threetears.core.models import SoftDeleteMixin, TimestampMixin, UUIDPrimaryKeyMixin


# Module-level declarative base and composed model for mixin tests.
# Defined at module level so SQLAlchemy can resolve Mapped annotations.
class _MixinTestBase(DeclarativeBase):
    pass


class _ComposedModel(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, _MixinTestBase):
    __tablename__ = "composed_test"
    name: Mapped[str] = mapped_column(Text, nullable=False)


# ---- Verify public API re-exports work ----


class TestPublicAPIImports:
    """Verify that the documented public API surface is importable."""

    def test_base_entity_import(self) -> None:
        from threetears.core import BaseEntity

        assert BaseEntity is not None

    def test_base_collection_import(self) -> None:
        from threetears.core import BaseCollection

        assert BaseCollection is not None

    def test_concurrent_modification_error_import(self) -> None:
        from threetears.core import ConcurrentModificationError

        assert ConcurrentModificationError is not None

    def test_config_imports(self) -> None:
        from threetears.core import CoreConfig, DefaultCoreConfig

        assert CoreConfig is not None
        assert DefaultCoreConfig is not None

    def test_data_layer_unavailable_error_import(self) -> None:
        from threetears.core import DataLayerUnavailableError

        assert DataLayerUnavailableError is not None

    def test_subpackage_imports(self) -> None:
        from threetears.core.cache.sqlite import SQLiteBackend
        from threetears.core.collections.registry import CollectionRegistry

        assert SQLiteBackend is not None
        assert CollectionRegistry is not None


# ---- Mixin tests ----


class TestModelMixins:
    """Verify SQLAlchemy mixin classes define expected columns."""

    def test_uuid_primary_key_mixin_has_id(self) -> None:
        assert hasattr(UUIDPrimaryKeyMixin, "id")

    def test_timestamp_mixin_has_date_columns(self) -> None:
        assert hasattr(TimestampMixin, "date_created")
        assert hasattr(TimestampMixin, "date_updated")

    def test_soft_delete_mixin_has_delete_columns(self) -> None:
        assert hasattr(SoftDeleteMixin, "is_deleted")
        assert hasattr(SoftDeleteMixin, "date_deleted")

    def test_mixins_compose_into_declarative_model(self) -> None:
        """Mixins can be composed into a full SQLAlchemy declarative model."""
        table = _ComposedModel.__table__
        col_names = {c.name for c in table.columns}
        assert "id" in col_names
        assert "date_created" in col_names
        assert "date_updated" in col_names
        assert "is_deleted" in col_names
        assert "date_deleted" in col_names
        assert "name" in col_names

        # Verify id is the primary key
        pk_names = [c.name for c in table.primary_key.columns]
        assert pk_names == ["id"]


# ---- Integration test fixtures and stubs ----


def _make_metadata() -> MetaData:
    metadata = MetaData()
    Table(
        "stub_items",
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
    """Concrete collection backed by an in-memory dict (simulating Postgres)."""

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
        return "stub_items"

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


def _make_nats_mock() -> AsyncMock:
    """Create a NATS client mock with in-memory KV store."""
    store: dict[str, bytes] = {}
    nats = AsyncMock()
    nats.bucket_name = MagicMock(return_value="test_collections")

    async def _get(bucket: str, key: str) -> bytes | None:
        return store.get(key)

    async def _put(bucket: str, key: str, value: bytes) -> bool:
        store[key] = value
        return True

    async def _delete(bucket: str, key: str) -> bool:
        store.pop(key, None)
        return True

    nats.get = AsyncMock(side_effect=_get)
    nats.put = AsyncMock(side_effect=_put)
    nats.delete = AsyncMock(side_effect=_delete)
    nats._store = store
    return nats


@pytest.fixture()
def l1_backend() -> SQLiteBackend:
    b = SQLiteBackend(db_name=f"test_integ_{uuid.uuid4().hex[:8]}")
    b.initialize(_make_metadata())
    yield b  # type: ignore[misc]
    b.reset()


@pytest.fixture()
def registry(l1_backend: SQLiteBackend) -> CollectionRegistry:
    reg = CollectionRegistry()
    reg.configure(l1_backend=l1_backend)
    return reg


@pytest.fixture()
def config() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


# ---- Full three-tier integration tests ----


class TestThreeTierIntegration:
    """Exercise the full create -> save -> get -> invalidate -> get -> reload -> delete flow."""

    @pytest.mark.asyncio
    async def test_full_crud_lifecycle(self, registry: CollectionRegistry, config: DefaultCoreConfig) -> None:
        """Full CRUD lifecycle through all three tiers.

        Flow:
        1. Create entity
        2. Save (writes to L3, promotes to L1 + L2)
        3. Get (L1 hit)
        4. Invalidate L1 + L2
        5. Get again (L1/L2 miss -> L3 hit, re-promotes)
        6. Modify and save
        7. Reload from L3
        8. Delete from all tiers
        """
        nats = _make_nats_mock()
        pg_store: dict[str, dict] = {}
        coll = StubCollection(registry, config, nats_client=nats, pg_store=pg_store)

        # 1. Create entity
        entity = coll.create({"id": "item-1", "name": "Widget", "score": 42})
        assert entity.is_new is True
        assert entity.is_dirty is True
        assert entity.id == "item-1"
        assert entity.name == "Widget"

        # 2. Save — persists to L3, then promotes to L1 + L2
        await entity.save()
        assert entity.is_new is False
        assert entity.is_dirty is False
        assert "item-1" in pg_store
        assert pg_store["item-1"]["name"] == "Widget"
        # L1 populated
        l1_row = coll._l1.select_by_id("stub_items", "item-1")
        assert l1_row is not None
        assert l1_row["name"] == "Widget"
        # L2 populated
        assert "stub_items.item-1" in nats._store

        # 3. Get — should hit L1
        nats.get.reset_mock()
        fetched = await coll.get("item-1")
        assert fetched is not None
        assert fetched.name == "Widget"
        assert fetched.score == 42
        nats.get.assert_not_awaited()  # L1 hit, no L2 call

        # 4. Invalidate L1 + L2 caches
        await coll.invalidate_cache("item-1")
        assert coll._l1.select_by_id("stub_items", "item-1") is None
        assert "stub_items.item-1" not in nats._store

        # 5. Get again — L1 miss, L2 miss, L3 hit -> re-promotes to L1 + L2
        fetched2 = await coll.get("item-1")
        assert fetched2 is not None
        assert fetched2.name == "Widget"
        # Re-promoted to L1
        l1_after = coll._l1.select_by_id("stub_items", "item-1")
        assert l1_after is not None
        # Re-promoted to L2
        assert "stub_items.item-1" in nats._store

        # 6. Modify and save
        fetched2.name = "Gadget"
        fetched2.score = 99
        assert fetched2.is_dirty is True
        await fetched2.save()
        assert fetched2.is_dirty is False
        assert pg_store["item-1"]["name"] == "Gadget"
        assert pg_store["item-1"]["score"] == 99

        # 7. Modify in "postgres" directly and reload
        pg_store["item-1"]["name"] = "Thingamajig"
        pg_store["item-1"]["score"] = 7
        await fetched2.reload()
        assert fetched2.name == "Thingamajig"
        assert fetched2.score == 7
        assert fetched2.is_dirty is False
        # L1 updated
        l1_reloaded = coll._l1.select_by_id("stub_items", "item-1")
        assert l1_reloaded is not None
        assert l1_reloaded["name"] == "Thingamajig"

        # 8. Delete from all tiers
        result = await coll.delete("item-1")
        assert result is True
        assert "item-1" not in pg_store
        assert coll._l1.select_by_id("stub_items", "item-1") is None
        assert "stub_items.item-1" not in nats._store

        # Verify get returns None after delete
        gone = await coll.get("item-1")
        assert gone is None

    @pytest.mark.asyncio
    async def test_l2_fallback_when_l1_invalidated(
        self, registry: CollectionRegistry, config: DefaultCoreConfig
    ) -> None:
        """When L1 is invalidated but L2 still has data, get() hits L2 and re-promotes to L1."""
        nats = _make_nats_mock()
        pg_store: dict[str, dict] = {}
        coll = StubCollection(registry, config, nats_client=nats, pg_store=pg_store)

        entity = coll.create({"id": "item-2", "name": "Sprocket", "score": 5})
        await entity.save()

        # Manually remove from L1 only (simulate L1 eviction), leave L2 intact
        coll._l1.delete_by_id("stub_items", "item-2")
        assert coll._l1.select_by_id("stub_items", "item-2") is None
        assert "stub_items.item-2" in nats._store  # L2 still has it

        # Get should hit L2 and re-promote to L1
        fetched = await coll.get("item-2")
        assert fetched is not None
        assert fetched.name == "Sprocket"

        # L1 re-promoted
        l1_row = coll._l1.select_by_id("stub_items", "item-2")
        assert l1_row is not None
        assert l1_row["name"] == "Sprocket"

    @pytest.mark.asyncio
    async def test_concurrent_modification_detected(
        self, registry: CollectionRegistry, config: DefaultCoreConfig
    ) -> None:
        """Optimistic locking detects concurrent modification through the full stack."""
        nats = _make_nats_mock()
        ts_v1 = datetime(2025, 1, 1, tzinfo=UTC)
        ts_v2 = datetime(2025, 6, 1, tzinfo=UTC)
        pg_store = {
            "item-3": {
                "id": "item-3",
                "name": "Original",
                "score": 10,
                "date_updated": ts_v1,
            }
        }
        coll = StubCollection(registry, config, nats_client=nats, pg_store=pg_store)

        # Load entity (L3 -> L1 promotion)
        entity = await coll.get("item-3")
        assert entity is not None
        assert entity.original_date_updated == ts_v1

        # Simulate another process updating L3 behind our back
        pg_store["item-3"]["date_updated"] = ts_v2
        pg_store["item-3"]["name"] = "Modified by other"

        # Our modification should fail with ConcurrentModificationError
        entity.name = "My update"
        with pytest.raises(ConcurrentModificationError):
            await entity.save()

    @pytest.mark.asyncio
    async def test_multiple_entities_independent(self, registry: CollectionRegistry, config: DefaultCoreConfig) -> None:
        """Multiple entities in the same collection are independent."""
        nats = _make_nats_mock()
        pg_store: dict[str, dict] = {}
        coll = StubCollection(registry, config, nats_client=nats, pg_store=pg_store)

        e1 = coll.create({"id": "a", "name": "Alpha", "score": 1})
        e2 = coll.create({"id": "b", "name": "Beta", "score": 2})
        await e1.save()
        await e2.save()

        # Modify one, verify the other is unaffected
        e1.name = "Alpha-Updated"
        await e1.save()

        fetched_b = await coll.get("b")
        assert fetched_b is not None
        assert fetched_b.name == "Beta"
        assert fetched_b.score == 2

        fetched_a = await coll.get("a")
        assert fetched_a is not None
        assert fetched_a.name == "Alpha-Updated"
