"""Tests for MemoriesCollection three-tier caching.

Registry-bound pool pattern (namespace-task-01 phase 8.5b): the
``postgres_pool`` constructor parameter is retired — callers bind the
pool to the registry via :meth:`CollectionRegistry.configure` (default)
or :meth:`CollectionRegistry.bind_table` (per-table override) BEFORE
constructing the Collection. :class:`BaseCollection.__init__` reads the
pool through :meth:`CollectionRegistry.get_l3_pool` at construction and
caches the reference on ``self.l3_pool``.
"""

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

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import MemoriesCollection
from threetears.agent.memory.entities import MemoryEntity


def _make_metadata() -> MetaData:
    """build a SQLite-compatible MetaData mirror of the memories table.

    collections-task-04 made ``agent_id`` part of the composite primary
    key on ``memories``; the L1 metadata mirrors that so
    :class:`SQLiteBackend` keys rows on the full ``(agent_id,
    memory_id)`` tuple just like L3 does.

    :return: SQLAlchemy metadata
    :rtype: MetaData
    """
    metadata = MetaData()
    Table(
        "memories",
        metadata,
        Column("agent_id", String(255), primary_key=True),
        Column("memory_id", String(255), primary_key=True),
        Column("customer_id", String(255)),
        Column("user_id", String(255)),
        Column("conversation_id", String(255)),
        Column("message_id_source", String(255)),
        Column("type_memory", String(50)),
        Column("content", Text),
        Column("embedding", Text),
        Column("media_id", String(255)),
        Column("is_deleted", Boolean),
        Column("date_created", DateTime),
        Column("date_deleted", DateTime),
        Column("date_updated", DateTime),
    )
    return metadata


def _sample_data() -> dict:
    """build a sample memory row for tests.

    :return: row data
    :rtype: dict
    """
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
    """Create a mock asyncpg pool backed by an in-memory dict.

    collections-task-04 partitioned the memories table on ``agent_id``
    and rewrote the primary key to the composite ``(agent_id,
    memory_id)``. the mock continues to key its in-memory dict by
    ``memory_id`` (UNIQUE constraint preserved on the table) so test
    setup stays terse, but every fetch / fetchval / fetch path now
    branches on the leading ``agent_id`` parameter the SQL emits and
    filters accordingly.

    :param store: initial pool contents keyed by memory_id string
    :ptype store: dict[str, dict] | None
    :return: asyncpg-shape mock
    :rtype: AsyncMock
    """
    if store is None:
        store = {}
    pg = AsyncMock()

    async def _fetchrow(query: str, *args: object) -> dict | None:
        # composite-pk fetch: SQL is ``WHERE agent_id = $1 AND memory_id = $2``
        # so ``args`` is ``(agent_id, memory_id)``; lookup keyed by
        # memory_id (the UNIQUE side) keeps the mock terse but still
        # filters on agent_id explicitly.
        if "agent_id = $1" in query and len(args) >= 2:
            agent_id = str(args[0])
            memory_id = str(args[1])
            row = store.get(memory_id)
            if row is None or str(row.get("agent_id")) != agent_id:
                return None
            return row
        # fetch_content_for_recall: agent_id, memory_id, user_id
        return store.get(str(args[0]) if args else None)

    async def _fetchval(query: str, *args: object) -> bool:
        # count_by_user query: agent_id $1, user_id $2
        agent_id = str(args[0]) if args else None
        user_id = str(args[1]) if len(args) > 1 else None
        return any(
            str(row.get("agent_id")) == agent_id
            and str(row.get("user_id")) == user_id
            for row in store.values()
        )

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
            # composite-pk CAS: $1=agent_id, $2=memory_id, then
            # mutable column values, with the CAS fence as the last
            # parameter.
            agent_id = str(args[0])
            memory_id = str(args[1])
            existing = store.get(memory_id)
            if existing is None or str(existing.get("agent_id")) != agent_id:
                return "UPDATE 0"
            cas_fence = args[-1]
            if cas_fence is not None and existing.get("date_updated") != cas_fence:
                return "UPDATE 0"
            # mutable columns in declared order:
            # content, embedding, is_deleted, date_deleted, date_updated
            existing["content"] = args[2]
            existing["embedding"] = args[3]
            existing["is_deleted"] = args[4]
            existing["date_deleted"] = args[5]
            existing["date_updated"] = args[6]
            return "UPDATE 1"
        elif "DELETE" in query:
            # composite-pk delete: $1=agent_id, $2=memory_id
            memory_id = str(args[1] if len(args) > 1 else args[0])
            store.pop(memory_id, None)
            return "DELETE 1"
        return "0"

    async def _fetch(query: str, *args: object) -> list[dict]:
        # find_by_user: agent_id $1, customer_id $2, user_id $3
        agent_id = str(args[0]) if args else None
        user_id = str(args[2]) if len(args) > 2 else None
        include_deleted = "is_deleted" not in query
        results = []
        for row in store.values():
            if (
                str(row.get("agent_id")) == agent_id
                and str(row.get("user_id")) == user_id
            ):
                if include_deleted or not row.get("is_deleted", False):
                    results.append(dict(row))
        return results

    pg.fetchrow = AsyncMock(side_effect=_fetchrow)
    pg.fetchval = AsyncMock(side_effect=_fetchval)
    pg.execute = AsyncMock(side_effect=_execute)
    pg.fetch = AsyncMock(side_effect=_fetch)
    pg.store = store
    return pg


def _make_nats_mock() -> AsyncMock:
    """Create a mock NATS KV client backed by an in-memory dict.

    :return: mock KV client
    :rtype: AsyncMock
    """
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
    nats.store = kv_store
    return nats


@pytest.fixture()
def l1_backend() -> SQLiteBackend:
    """build an initialized SQLiteBackend for one test."""
    b = SQLiteBackend(db_name=f"test_mem_{uuid.uuid7().hex[:8]}")
    b.initialize(_make_metadata())
    yield b
    b.reset()


@pytest.fixture()
def pg_pool() -> AsyncMock:
    """build a mock pool reused across a test's construction cycles."""
    return _make_pg_mock()


@pytest.fixture()
def registry(
    l1_backend: SQLiteBackend,
    pg_pool: AsyncMock,
) -> CollectionRegistry:
    """build a registry pre-configured with L1 + a default mock L3 pool.

    binding the pool to the registry BEFORE Collection construction
    matches the production wiring path (see
    :mod:`aibots_agents.runtime.three_tier_stack.build_three_tier_stack`).
    """
    reg = CollectionRegistry()
    reg.configure(l1_backend=l1_backend, l3_pool=pg_pool)
    return reg


@pytest.fixture()
def config_always() -> DefaultCoreConfig:
    """build a ``CollectionConfig(flush=ALWAYS)`` for immediate L3 writes."""
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


def _rebind_pool(registry: CollectionRegistry, pool: AsyncMock) -> None:
    """helper: override the registry's default L3 pool mid-test.

    used by tests that want a pool pre-seeded with data — we swap the
    default pool before constructing the Collection.
    """
    registry.configure(l3_pool=pool)


class TestMemoriesCollectionGet:
    async def test_l1_hit(
        self,
        registry: CollectionRegistry,
        config_always: DefaultCoreConfig,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        nats = _make_nats_mock()
        coll = MemoriesCollection(
            registry,
            config_always,
            authorizer=permissive_memory_authorizer,
            nats_client=nats,
        )

        data = _sample_data()
        l1_data = {k: str(v) if isinstance(v, uuid.UUID) else v for k, v in data.items()}
        coll.write_to_cache_sync(l1_data, ("agent_id", "memory_id"))

        entity = await coll.get((data["agent_id"], data["memory_id"]))

        assert entity is not None
        assert entity.content == "User prefers dark mode"
        nats.get.assert_not_awaited()

    async def test_l3_hit_promotes(
        self,
        registry: CollectionRegistry,
        config_always: DefaultCoreConfig,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        nats = _make_nats_mock()
        data = _sample_data()
        pg = _make_pg_mock({str(data["memory_id"]): data})
        _rebind_pool(registry, pg)
        coll = MemoriesCollection(
            registry,
            config_always,
            authorizer=permissive_memory_authorizer,
            nats_client=nats,
        )

        entity = await coll.get((data["agent_id"], data["memory_id"]))

        assert entity is not None
        assert entity.type_memory == "preference"
        assert any(
            str(data["memory_id"]) in key for key in nats.store
        )

    async def test_all_miss_returns_none(
        self,
        registry: CollectionRegistry,
        config_always: DefaultCoreConfig,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        coll = MemoriesCollection(
            registry,
            config_always,
            authorizer=permissive_memory_authorizer,
        )

        entity = await coll.get((uuid.uuid7(), uuid.uuid7()))
        assert entity is None


class TestMemoriesCollectionSave:
    async def test_save_new_entity(
        self,
        registry: CollectionRegistry,
        config_always: DefaultCoreConfig,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        pg_store: dict[str, dict] = {}
        pg = _make_pg_mock(pg_store)
        _rebind_pool(registry, pg)
        nats = _make_nats_mock()
        coll = MemoriesCollection(
            registry,
            config_always,
            authorizer=permissive_memory_authorizer,
            nats_client=nats,
        )

        data = _sample_data()
        entity = coll.create(data)
        await coll.save_entity(entity)

        assert str(data["memory_id"]) in pg_store
        assert entity.is_dirty is False
        assert entity.is_new is False

    async def test_save_updates_entity(
        self,
        registry: CollectionRegistry,
        config_always: DefaultCoreConfig,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        data = _sample_data()
        pg_store = {str(data["memory_id"]): dict(data)}
        pg = _make_pg_mock(pg_store)
        _rebind_pool(registry, pg)
        coll = MemoriesCollection(
            registry,
            config_always,
            authorizer=permissive_memory_authorizer,
        )

        entity = await coll.get((data["agent_id"], data["memory_id"]))
        assert entity is not None

        entity.content = "Updated preference"
        await coll.save_entity(entity)

        assert entity.is_dirty is False


class TestMemoriesCollectionDelete:
    async def test_delete_removes_from_all_tiers(
        self,
        registry: CollectionRegistry,
        config_always: DefaultCoreConfig,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        data = _sample_data()
        pg_store = {str(data["memory_id"]): dict(data)}
        pg = _make_pg_mock(pg_store)
        _rebind_pool(registry, pg)
        nats = _make_nats_mock()
        coll = MemoriesCollection(
            registry,
            config_always,
            authorizer=permissive_memory_authorizer,
            nats_client=nats,
        )

        pk = (data["agent_id"], data["memory_id"])
        await coll.get(pk)
        result = await coll.delete(pk)

        assert result is True
        assert str(data["memory_id"]) not in pg_store


class TestMemoriesCollectionSoftDelete:
    async def test_soft_delete_sets_flags(
        self,
        registry: CollectionRegistry,
        config_always: DefaultCoreConfig,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        data = _sample_data()
        pg_store = {str(data["memory_id"]): dict(data)}
        pg = _make_pg_mock(pg_store)
        _rebind_pool(registry, pg)
        coll = MemoriesCollection(
            registry,
            config_always,
            authorizer=permissive_memory_authorizer,
        )

        entity = await coll.get((data["agent_id"], data["memory_id"]))
        assert entity is not None

        await coll.soft_delete(entity)

        assert entity.is_deleted is True
        assert entity.date_deleted is not None


class TestMemoriesCollectionFindByUser:
    async def test_find_by_user_returns_entities(
        self,
        registry: CollectionRegistry,
        config_always: DefaultCoreConfig,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        user_id = uuid.uuid7()
        agent_id = uuid.uuid7()
        customer_id = uuid.uuid7()
        data1 = _sample_data()
        data1["user_id"] = user_id
        data1["agent_id"] = agent_id
        data1["customer_id"] = customer_id
        data2 = _sample_data()
        data2["user_id"] = user_id
        data2["agent_id"] = agent_id
        data2["customer_id"] = customer_id
        pg_store = {
            str(data1["memory_id"]): data1,
            str(data2["memory_id"]): data2,
        }
        pg = _make_pg_mock(pg_store)
        _rebind_pool(registry, pg)
        coll = MemoriesCollection(
            registry,
            config_always,
            authorizer=permissive_memory_authorizer,
        )

        entities = await coll.find_by_user(
            user_id, agent_id=agent_id, customer_id=customer_id,
        )

        assert len(entities) == 2
        assert all(isinstance(e, MemoryEntity) for e in entities)

    async def test_find_by_user_excludes_deleted(
        self,
        registry: CollectionRegistry,
        config_always: DefaultCoreConfig,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        user_id = uuid.uuid7()
        agent_id = uuid.uuid7()
        customer_id = uuid.uuid7()
        data1 = _sample_data()
        data1["user_id"] = user_id
        data1["agent_id"] = agent_id
        data1["customer_id"] = customer_id
        data1["is_deleted"] = False
        data2 = _sample_data()
        data2["user_id"] = user_id
        data2["agent_id"] = agent_id
        data2["customer_id"] = customer_id
        data2["is_deleted"] = True
        pg_store = {
            str(data1["memory_id"]): data1,
            str(data2["memory_id"]): data2,
        }
        pg = _make_pg_mock(pg_store)
        _rebind_pool(registry, pg)
        coll = MemoriesCollection(
            registry,
            config_always,
            authorizer=permissive_memory_authorizer,
        )

        entities = await coll.find_by_user(
            user_id,
            include_deleted=False,
            agent_id=agent_id,
            customer_id=customer_id,
        )

        assert len(entities) == 1

    async def test_find_by_user_includes_deleted(
        self,
        registry: CollectionRegistry,
        config_always: DefaultCoreConfig,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        user_id = uuid.uuid7()
        agent_id = uuid.uuid7()
        customer_id = uuid.uuid7()
        data1 = _sample_data()
        data1["user_id"] = user_id
        data1["agent_id"] = agent_id
        data1["customer_id"] = customer_id
        data1["is_deleted"] = False
        data2 = _sample_data()
        data2["user_id"] = user_id
        data2["agent_id"] = agent_id
        data2["customer_id"] = customer_id
        data2["is_deleted"] = True
        pg_store = {
            str(data1["memory_id"]): data1,
            str(data2["memory_id"]): data2,
        }
        pg = _make_pg_mock(pg_store)
        _rebind_pool(registry, pg)
        coll = MemoriesCollection(
            registry,
            config_always,
            authorizer=permissive_memory_authorizer,
        )

        entities = await coll.find_by_user(
            user_id,
            include_deleted=True,
            agent_id=agent_id,
            customer_id=customer_id,
        )

        assert len(entities) == 2


class TestMemoriesCollectionSerialization:
    def test_serialize_deserialize_roundtrip(
        self,
        registry: CollectionRegistry,
        config_always: DefaultCoreConfig,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        coll = MemoriesCollection(
            registry,
            config_always,
            authorizer=permissive_memory_authorizer,
        )

        data = _sample_data()
        data["date_created"] = datetime(2025, 6, 1, tzinfo=UTC)
        data["date_updated"] = datetime(2025, 6, 2, tzinfo=UTC)

        serialized = coll.serialize(data)
        assert isinstance(serialized, bytes)

        deserialized = coll.deserialize(serialized)

        assert deserialized["memory_id"] == data["memory_id"]
        assert deserialized["user_id"] == data["user_id"]
        assert deserialized["type_memory"] == "preference"
        assert deserialized["embedding"] == [0.1, 0.2, 0.3]
        assert deserialized["is_deleted"] is False
        assert deserialized["media_id"] is None
        assert isinstance(deserialized["date_created"], datetime)

    def test_deserialize_handles_none_values(
        self,
        registry: CollectionRegistry,
        config_always: DefaultCoreConfig,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        coll = MemoriesCollection(
            registry,
            config_always,
            authorizer=permissive_memory_authorizer,
        )

        raw = json.dumps(
            {
                "memory_id": str(uuid.uuid7()),
                "media_id": None,
                "date_deleted": None,
            }
        ).encode("utf-8")

        result = coll.deserialize(raw)
        assert result["media_id"] is None
        assert result["date_deleted"] is None


class TestMemoriesCollectionTableName:
    def test_table_name(
        self,
        registry: CollectionRegistry,
        config_always: DefaultCoreConfig,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        coll = MemoriesCollection(
            registry,
            config_always,
            authorizer=permissive_memory_authorizer,
        )
        assert coll.table_name == "memories"

    def test_entity_class(
        self,
        registry: CollectionRegistry,
        config_always: DefaultCoreConfig,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        coll = MemoriesCollection(
            registry,
            config_always,
            authorizer=permissive_memory_authorizer,
        )
        assert coll.entity_class is MemoryEntity

    def testprimary_key_column(
        self,
        registry: CollectionRegistry,
        config_always: DefaultCoreConfig,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        coll = MemoriesCollection(
            registry,
            config_always,
            authorizer=permissive_memory_authorizer,
        )
        assert coll.primary_key_column == ("agent_id", "memory_id")


class TestMemoriesCollectionCountByUser:
    async def test_count_by_user_present(
        self,
        registry: CollectionRegistry,
        config_always: DefaultCoreConfig,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        data = _sample_data()
        pg_store = {str(data["memory_id"]): data}
        pg = _make_pg_mock(pg_store)
        _rebind_pool(registry, pg)
        coll = MemoriesCollection(
            registry,
            config_always,
            authorizer=permissive_memory_authorizer,
        )

        result = await coll.count_by_user(
            data["user_id"], agent_id=data["agent_id"],
        )
        assert result is True

    async def test_count_by_user_absent(
        self,
        registry: CollectionRegistry,
        config_always: DefaultCoreConfig,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        coll = MemoriesCollection(
            registry,
            config_always,
            authorizer=permissive_memory_authorizer,
        )

        result = await coll.count_by_user(uuid.uuid7(), agent_id=uuid.uuid7())
        assert result is False
