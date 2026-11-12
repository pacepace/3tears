"""a raw salience or supersession UPDATE evicts every row it may have changed, however it ends.

``bump_salience`` and ``mark_superseded`` write L3 directly and then evict
the rows they touched. Two ways that eviction used to go missing:

- the UPDATE raised after reaching L3 (a connection lost once the statement was sent). Its
  outcome is unknown, the rows may have changed, and nothing was evicted.
- the task was cancelled partway through the eviction loop. The UPDATE had committed; the rows
  the loop had not reached stayed cached, and ``asyncio.CancelledError`` is not an ``Exception``
  so nothing caught it on the way out.

Both are asserted on the invalidation broadcast, not on a local re-read.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from threetears.core.collections.registry import CacheInvalidationMessage, CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeNatsClient

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import MemoriesCollection


# parity-with: asyncpg.Pool (execute / fetch: the two calls the raw salience writes make)
class _Pool:
    """an L3 whose execute raises after the statement reached the server, and whose fetch returns fixed keys."""

    def __init__(self, returned: list[tuple[Any, ...]] | None = None) -> None:
        self.returned = returned or []

    async def execute(self, sql: str, *args: Any) -> str:
        del sql, args
        raise ConnectionError("connection lost after the statement was sent")

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        del sql, args
        return [{"agent_id": agent_id, "memory_id": memory_id} for agent_id, memory_id in self.returned]


def _collection(pool: Any, nats: FakeNatsClient, authorizer: MemoryAuthorizerDependencies) -> MemoriesCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool, kv_key_scope="salience-principal")
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return MemoriesCollection(registry, config, authorizer=authorizer, nats_client=nats)


def _evicted(nats: FakeNatsClient) -> set[tuple[str, ...]]:
    return {
        tuple(message.ids)
        for message in nats.published
        if isinstance(message, CacheInvalidationMessage) and message.table == "memories"
    }


class TestAnUpdateThatRaisesStillEvicts:
    @pytest.mark.asyncio
    async def test_bump_salience(self, permissive_memory_authorizer: MemoryAuthorizerDependencies) -> None:
        nats = FakeNatsClient()
        agent_id, ids = uuid.uuid4(), [uuid.uuid4(), uuid.uuid4()]
        collection = _collection(_Pool(), nats, permissive_memory_authorizer)

        with pytest.raises(ConnectionError):
            await collection.bump_salience(ids, agent_id=agent_id, access_bump=0.1)

        assert _evicted(nats) == {(str(agent_id), str(i)) for i in ids}

    @pytest.mark.asyncio
    async def test_mark_superseded(self, permissive_memory_authorizer: MemoryAuthorizerDependencies) -> None:
        nats = FakeNatsClient()
        agent_id, ids = uuid.uuid4(), [uuid.uuid4(), uuid.uuid4()]
        collection = _collection(_Pool(), nats, permissive_memory_authorizer)

        with pytest.raises(ConnectionError):
            await collection.mark_superseded(agent_id=agent_id, source_memory_ids=ids, gist_id=uuid.uuid4())

        assert _evicted(nats) == {(str(agent_id), str(i)) for i in ids}
