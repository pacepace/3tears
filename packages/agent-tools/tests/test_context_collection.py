"""Tests for ContextItemCollection three-tier operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.agent.tools.collections import ContextItemCollection
from threetears.agent.tools.entities import ContextItemEntity

from testing_utils import FakePool, make_context_metadata, make_nats_mock


@pytest.fixture()
def l1_backend() -> SQLiteBackend:
    b = SQLiteBackend(db_name=f"test_ctx_{uuid.uuid4().hex[:8]}")
    b.initialize(make_context_metadata())
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
def config() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


@pytest.fixture()
def pool() -> FakePool:
    return FakePool()


@pytest.fixture()
def collection(registry: CollectionRegistry, config: DefaultCoreConfig, pool: FakePool) -> ContextItemCollection:
    nats = make_nats_mock()
    coll = ContextItemCollection(registry, config, nats_client=nats)
    coll.l3_pool = pool
    return coll


def _make_item(
    conversation_id: str = "00000000-0000-0000-0000-000000000001",
    context_type: str = "tool_result",
    key: str = "calc",
    short_desc: str = "42",
    long_desc: str = "",
    content: str = "The answer is 42",
    date_accessed: datetime | None = None,
) -> dict:
    now = datetime.now(UTC)
    return {
        "context_id": uuid.uuid4(),
        "conversation_id": uuid.UUID(conversation_id),
        "context_type": context_type,
        "key": key,
        "short_desc": short_desc,
        "long_desc": long_desc,
        "content": content,
        "metadata": {},
        "date_accessed": date_accessed or now,
        "date_created": now,
        "date_updated": now,
    }


CONV_ID = "00000000-0000-0000-0000-000000000001"
CONV_UUID = uuid.UUID(CONV_ID)


class TestFindByConversation:
    @pytest.mark.asyncio
    async def test_returns_entities(self, collection: ContextItemCollection, pool: FakePool) -> None:
        item = _make_item()
        pool.rows[str(item["context_id"])] = item

        entities = await collection.find_by_conversation(CONV_UUID)

        assert len(entities) == 1
        assert isinstance(entities[0], ContextItemEntity)

    @pytest.mark.asyncio
    async def test_populates_l1(self, collection: ContextItemCollection, pool: FakePool) -> None:
        item = _make_item()
        pool.rows[str(item["context_id"])] = item

        await collection.find_by_conversation(CONV_UUID)

        # Should be in L1 now (composite-pk row keyed on tuple).
        l1_row = collection.get_row_sync((CONV_UUID, item["context_id"]))
        assert l1_row is not None

    @pytest.mark.asyncio
    async def test_empty_conversation(self, collection: ContextItemCollection) -> None:
        entities = await collection.find_by_conversation(CONV_UUID)
        assert entities == []


class TestUpsertVariable:
    @pytest.mark.asyncio
    async def test_insert_new(self, collection: ContextItemCollection) -> None:
        item = _make_item(context_type="variable", key="name", short_desc="Alice", content="Alice")
        returned_id = await collection.upsert_variable(CONV_UUID, item)
        assert returned_id == item["context_id"]

    @pytest.mark.asyncio
    async def test_upsert_existing(self, collection: ContextItemCollection, pool: FakePool) -> None:
        item1 = _make_item(context_type="variable", key="name", short_desc="Alice", content="Alice")
        await collection.upsert_variable(CONV_UUID, item1)

        item2 = _make_item(context_type="variable", key="name", short_desc="Bob", content="Bob")
        returned_id = await collection.upsert_variable(CONV_UUID, item2)

        # Should return the original context_id (conflict resolution)
        assert returned_id == item1["context_id"]
        # Value should be updated
        assert pool.rows[str(item1["context_id"])]["content"] == "Bob"


class TestTouch:
    @pytest.mark.asyncio
    async def test_updates_date_accessed(self, collection: ContextItemCollection, pool: FakePool) -> None:
        old_time = datetime.now(UTC) - timedelta(hours=1)
        item = _make_item(date_accessed=old_time)
        pool.rows[str(item["context_id"])] = item
        # Populate L1
        collection.write_to_cache_sync(item)

        await collection.touch(CONV_UUID, str(item["context_id"]))

        # L1 should have updated date_accessed. composite-pk row keyed
        # on (conversation_id, context_id) tuple.
        l1_row = collection.get_row_sync((CONV_UUID, item["context_id"]))
        assert l1_row is not None
        # date_accessed should be more recent than old_time
        assert l1_row["date_accessed"] > old_time


class TestEvictLru:
    @pytest.mark.asyncio
    async def test_evicts_oldest_tool_results(self, collection: ContextItemCollection, pool: FakePool) -> None:
        now = datetime.now(UTC)
        items = []
        for i in range(5):
            item = _make_item(key=f"tool{i}", date_accessed=now + timedelta(seconds=i))
            pool.rows[str(item["context_id"])] = item
            collection.write_to_cache_sync(item)
            items.append(item)

        evicted = await collection.evict_lru(CONV_UUID, result_limit=3)

        assert evicted == 2
        # Oldest two should be gone from L3
        assert str(items[0]["context_id"]) not in pool.rows
        assert str(items[1]["context_id"]) not in pool.rows
        # Newest three should remain
        assert str(items[2]["context_id"]) in pool.rows
        assert str(items[3]["context_id"]) in pool.rows
        assert str(items[4]["context_id"]) in pool.rows

    @pytest.mark.asyncio
    async def test_does_not_evict_variables(self, collection: ContextItemCollection, pool: FakePool) -> None:
        now = datetime.now(UTC)
        # Add a variable with very old date_accessed
        var = _make_item(
            context_type="variable",
            key="name",
            short_desc="Alice",
            content="Alice",
            date_accessed=now - timedelta(days=1),
        )
        pool.rows[str(var["context_id"])] = var
        # Add tool results
        for i in range(3):
            item = _make_item(key=f"tool{i}", date_accessed=now + timedelta(seconds=i))
            pool.rows[str(item["context_id"])] = item

        evicted = await collection.evict_lru(CONV_UUID, result_limit=2)

        assert evicted == 1
        # Variable should still be there
        assert str(var["context_id"]) in pool.rows

    @pytest.mark.asyncio
    async def test_no_eviction_under_limit(self, collection: ContextItemCollection, pool: FakePool) -> None:
        for i in range(3):
            item = _make_item(key=f"tool{i}")
            pool.rows[str(item["context_id"])] = item

        evicted = await collection.evict_lru(CONV_UUID, result_limit=5)
        assert evicted == 0


class TestThreeTierReadWrite:
    @pytest.mark.asyncio
    async def test_save_and_get(self, collection: ContextItemCollection, pool: FakePool) -> None:
        item = _make_item()
        pool.rows[str(item["context_id"])] = item

        entity = await collection.get((CONV_UUID, item["context_id"]))

        assert entity is not None
        assert entity.key == "calc"
        assert entity.content == "The answer is 42"
