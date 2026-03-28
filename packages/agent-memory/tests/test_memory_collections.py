"""Tests for MemoriesCollection three-tier caching."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Boolean, Column, DateTime, MetaData, String, Table, Text

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.agent.memory.collections import MemoriesCollection
from threetears.agent.memory.entities import MemoryEntity


def _make_metadata() -> MetaData:
    metadata = MetaData()
    Table(
        "memories",
        metadata,
        Column("memory_id", String(255), primary_key=True),
        Column("agent_id", String(255)),
        Column("customer_id", String(255)),
        Column("user_id", String(255)),
        Column("conversation_id", String(255)),
        Column("message_id_source", String(255)),
        Column("type_memory", String(50)),
        Column("content", Text),
        Column("embedding", Text),  # stored as JSON text in SQLite
        Column("media_id", String(255)),
        Column("is_deleted", Boolean),
        Column("date_created", DateTime),
        Column("date_deleted", DateTime),
        Column("date_updated", DateTime),
    )
    return metadata


def _sample_data() -> dict:
    return {
        "memory_id": uuid.uuid7(),
        "agent_id": uuid.uuid7(),
        "customer_id": uuid.uuid7(),
        "user_id": uuid.uuid7(),
        "conversation_id": uuid.uuid7(),
        "message_id_source": uuid.uuid7(),
        "type_memory": "preference",
        "content": "User prefers dark mode",
        "embedding": [0.1, 0.2, 0.3],
        "media_id": None,
        "is_deleted": False,
        "date_created": datetime.now(UTC),
        "date_deleted": None,
        "date_updated": None,
    }


def _make_pg_mock(store: dict[str, dict] | None = None) -> AsyncMock:
    """Create a mock asyncpg pool backed by an in-memory dict."""
    if store is None:
        store = {}
    pg = AsyncMock()

    async def _fetchrow(query: str, *args: object) -> dict | None:
        entity_id = args[0] if args else None
        return store.get(str(entity_id))

    async def _execute(query: str, *args: object) -> str:
        if "INSERT" in query:
            data_keys = [
                "memory_id",
                "agent_id",
                "customer_id",
                "user_id",
                "conversation_id",
                "message_id_source",
                "type_memory",
                "content",
                "embedding",
                "is_deleted",
                "media_id",
                "date_created",
                "date_deleted",
                "date_updated",
            ]
            data = dict(zip(data_keys, args))
            store[str(data["memory_id"])] = data
            return "INSERT 0 1"
        elif "UPDATE" in query:
            entity_id = str(args[0])
            existing = store.get(entity_id)
            if existing is None:
                return "UPDATE 0"
            # Check optimistic lock
            if len(args) > 6 and args[6] is not None:
                if existing.get("date_updated") != args[6]:
                    return "UPDATE 0"
            existing["content"] = args[1]
            existing["embedding"] = args[2]
            existing["is_deleted"] = args[3]
            existing["date_deleted"] = args[4]
            existing["date_updated"] = args[5]
            return "UPDATE 1"
        elif "DELETE" in query:
            entity_id = str(args[0])
            store.pop(entity_id, None)
            return "DELETE 1"
        return "0"

    async def _fetch(query: str, *args: object) -> list[dict]:
        user_id = str(args[0]) if args else None
        include_deleted = "is_deleted" not in query
        results = []
        for row in store.values():
            if str(row.get("user_id")) == user_id:
                if include_deleted or not row.get("is_deleted", False):
                    results.append(dict(row))
        return results

    pg.fetchrow = AsyncMock(side_effect=_fetchrow)
    pg.execute = AsyncMock(side_effect=_execute)
    pg.fetch = AsyncMock(side_effect=_fetch)
    pg._store = store
    return pg


def _make_nats_mock() -> AsyncMock:
    kv_store: dict[str, bytes] = {}
    nats = AsyncMock()
    nats.bucket_name = MagicMock(return_value="test_collections")

    async def _get(bucket: str, key: str) -> bytes | None:
        return kv_store.get(key)

    async def _put(bucket: str, key: str, value: bytes) -> bool:
        kv_store[key] = value
        return True

    async def _delete(bucket: str, key: str) -> bool:
        kv_store.pop(key, None)
        return True

    nats.get = AsyncMock(side_effect=_get)
    nats.put = AsyncMock(side_effect=_put)
    nats.delete = AsyncMock(side_effect=_delete)
    nats._store = kv_store
    return nats


@pytest.fixture()
def l1_backend() -> SQLiteBackend:
    b = SQLiteBackend(db_name=f"test_mem_{uuid.uuid7().hex[:8]}")
    b.initialize(_make_metadata())
    yield b
    b.reset()


@pytest.fixture()
def registry(l1_backend: SQLiteBackend) -> CollectionRegistry:
    reg = CollectionRegistry()
    reg.configure(l1_backend=l1_backend)
    return reg


@pytest.fixture()
def config_always() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


class TestMemoriesCollectionGet:
    async def test_l1_hit(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        nats = _make_nats_mock()
        pg = _make_pg_mock()
        coll = MemoriesCollection(registry, config_always, postgres_pool=pg, nats_client=nats)

        data = _sample_data()
        # Serialize UUIDs to strings for SQLite L1
        l1_data = {k: str(v) if isinstance(v, uuid.UUID) else v for k, v in data.items()}
        coll._l1.upsert("memories", l1_data, "memory_id")

        entity = await coll.get(data["memory_id"])

        assert entity is not None
        assert entity.content == "User prefers dark mode"
        nats.get.assert_not_awaited()

    async def test_l3_hit_promotes(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        nats = _make_nats_mock()
        data = _sample_data()
        pg = _make_pg_mock({str(data["memory_id"]): data})
        coll = MemoriesCollection(registry, config_always, postgres_pool=pg, nats_client=nats)

        entity = await coll.get(data["memory_id"])

        assert entity is not None
        assert entity.type_memory == "preference"
        # Promoted to L2
        assert f"memories.{data['memory_id']}" in nats._store

    async def test_all_miss_returns_none(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        pg = _make_pg_mock()
        coll = MemoriesCollection(registry, config_always, postgres_pool=pg)

        entity = await coll.get(uuid.uuid7())
        assert entity is None


class TestMemoriesCollectionSave:
    async def test_save_new_entity(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        pg_store: dict[str, dict] = {}
        pg = _make_pg_mock(pg_store)
        nats = _make_nats_mock()
        coll = MemoriesCollection(registry, config_always, postgres_pool=pg, nats_client=nats)

        data = _sample_data()
        entity = coll.create(data)
        await coll.save_entity(entity)

        assert str(data["memory_id"]) in pg_store
        assert entity.is_dirty is False
        assert entity.is_new is False

    async def test_save_updates_entity(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        data = _sample_data()
        pg_store = {str(data["memory_id"]): dict(data)}
        pg = _make_pg_mock(pg_store)
        coll = MemoriesCollection(registry, config_always, postgres_pool=pg)

        entity = await coll.get(data["memory_id"])
        assert entity is not None

        entity.content = "Updated preference"
        await coll.save_entity(entity)

        assert entity.is_dirty is False


class TestMemoriesCollectionDelete:
    async def test_delete_removes_from_all_tiers(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        data = _sample_data()
        pg_store = {str(data["memory_id"]): dict(data)}
        pg = _make_pg_mock(pg_store)
        nats = _make_nats_mock()
        coll = MemoriesCollection(registry, config_always, postgres_pool=pg, nats_client=nats)

        # Load into caches
        await coll.get(data["memory_id"])

        result = await coll.delete(data["memory_id"])

        assert result is True
        assert str(data["memory_id"]) not in pg_store


class TestMemoriesCollectionSoftDelete:
    async def test_soft_delete_sets_flags(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        data = _sample_data()
        pg_store = {str(data["memory_id"]): dict(data)}
        pg = _make_pg_mock(pg_store)
        coll = MemoriesCollection(registry, config_always, postgres_pool=pg)

        entity = await coll.get(data["memory_id"])
        assert entity is not None

        await coll.soft_delete(entity)

        assert entity.is_deleted is True
        assert entity.date_deleted is not None


class TestMemoriesCollectionFindByUser:
    async def test_find_by_user_returns_entities(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        user_id = uuid.uuid7()
        data1 = _sample_data()
        data1["user_id"] = user_id
        data2 = _sample_data()
        data2["user_id"] = user_id
        pg_store = {
            str(data1["memory_id"]): data1,
            str(data2["memory_id"]): data2,
        }
        pg = _make_pg_mock(pg_store)
        coll = MemoriesCollection(registry, config_always, postgres_pool=pg)

        entities = await coll.find_by_user(user_id)

        assert len(entities) == 2
        assert all(isinstance(e, MemoryEntity) for e in entities)

    async def test_find_by_user_excludes_deleted(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        user_id = uuid.uuid7()
        data1 = _sample_data()
        data1["user_id"] = user_id
        data1["is_deleted"] = False
        data2 = _sample_data()
        data2["user_id"] = user_id
        data2["is_deleted"] = True
        pg_store = {
            str(data1["memory_id"]): data1,
            str(data2["memory_id"]): data2,
        }
        pg = _make_pg_mock(pg_store)
        coll = MemoriesCollection(registry, config_always, postgres_pool=pg)

        entities = await coll.find_by_user(user_id, include_deleted=False)

        assert len(entities) == 1

    async def test_find_by_user_includes_deleted(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        user_id = uuid.uuid7()
        data1 = _sample_data()
        data1["user_id"] = user_id
        data1["is_deleted"] = False
        data2 = _sample_data()
        data2["user_id"] = user_id
        data2["is_deleted"] = True
        pg_store = {
            str(data1["memory_id"]): data1,
            str(data2["memory_id"]): data2,
        }
        pg = _make_pg_mock(pg_store)
        coll = MemoriesCollection(registry, config_always, postgres_pool=pg)

        entities = await coll.find_by_user(user_id, include_deleted=True)

        assert len(entities) == 2


class TestMemoriesCollectionSerialization:
    def test_serialize_deserialize_roundtrip(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        pg = _make_pg_mock()
        coll = MemoriesCollection(registry, config_always, postgres_pool=pg)

        data = _sample_data()
        data["date_created"] = datetime(2025, 6, 1, tzinfo=UTC)
        data["date_updated"] = datetime(2025, 6, 2, tzinfo=UTC)

        serialized = coll._serialize(data)
        assert isinstance(serialized, bytes)

        deserialized = coll._deserialize(serialized)

        assert deserialized["memory_id"] == data["memory_id"]
        assert deserialized["user_id"] == data["user_id"]
        assert deserialized["type_memory"] == "preference"
        assert deserialized["embedding"] == [0.1, 0.2, 0.3]
        assert deserialized["is_deleted"] is False
        assert deserialized["media_id"] is None
        assert isinstance(deserialized["date_created"], datetime)

    def test_deserialize_handles_none_values(
        self, registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        pg = _make_pg_mock()
        coll = MemoriesCollection(registry, config_always, postgres_pool=pg)

        raw = json.dumps(
            {
                "memory_id": str(uuid.uuid7()),
                "media_id": None,
                "date_deleted": None,
            }
        ).encode("utf-8")

        result = coll._deserialize(raw)
        assert result["media_id"] is None
        assert result["date_deleted"] is None


class TestMemoriesCollectionTableName:
    def test_table_name(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        pg = _make_pg_mock()
        coll = MemoriesCollection(registry, config_always, postgres_pool=pg)
        assert coll.table_name == "memories"

    def test_entity_class(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        pg = _make_pg_mock()
        coll = MemoriesCollection(registry, config_always, postgres_pool=pg)
        assert coll.entity_class is MemoryEntity

    def test_primary_key_column(self, registry: CollectionRegistry, config_always: DefaultCoreConfig) -> None:
        pg = _make_pg_mock()
        coll = MemoriesCollection(registry, config_always, postgres_pool=pg)
        assert coll._primary_key_column == "memory_id"
