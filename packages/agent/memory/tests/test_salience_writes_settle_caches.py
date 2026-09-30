"""a raw salience or supersession UPDATE evicts every row it may have changed, however it ends.

``bump_salience``, ``mark_superseded`` and ``decay_salience`` write L3 directly and then evict
the rows they touched. Two ways that eviction used to go missing:

- the UPDATE raised after reaching L3 (a connection lost once the statement was sent). Its
  outcome is unknown, the rows may have changed, and nothing was evicted.
- the task was cancelled partway through the eviction loop. The UPDATE had committed; the rows
  the loop had not reached stayed cached, and ``asyncio.CancelledError`` is not an ``Exception``
  so nothing caught it on the way out.

Both are asserted on the invalidation broadcast, not on a local re-read.
"""

from __future__ import annotations

import asyncio
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


class _SuspendingNats(FakeNatsClient):
    """a NATS client whose first publish suspends until released, where a real broker acknowledgement would."""

    def __init__(self) -> None:
        super().__init__()
        self.first_publish_started = asyncio.Event()
        self.release = asyncio.Event()

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:
        if not self.first_publish_started.is_set():
            self.first_publish_started.set()
            await self.release.wait()
        await super().publish(subject=subject, message=message, reply_to=reply_to)


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


class TestACancelledEvictionFinishes:
    @pytest.mark.asyncio
    async def test_decay_salience(self, permissive_memory_authorizer: MemoryAuthorizerDependencies) -> None:
        """cancelled while evicting the first decayed row: every decayed row is still evicted."""
        nats = _SuspendingNats()
        agent_id = uuid.uuid4()
        decayed = [(agent_id, uuid.uuid4()) for _ in range(3)]
        collection = _collection(_Pool(decayed), nats, permissive_memory_authorizer)

        task = asyncio.create_task(collection.decay_salience(half_life_days=30.0, floor=0.05))
        await asyncio.wait_for(nats.first_publish_started.wait(), timeout=5.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        nats.release.set()
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        await asyncio.gather(*pending)

        assert _evicted(nats) == {(str(a), str(m)) for a, m in decayed}
