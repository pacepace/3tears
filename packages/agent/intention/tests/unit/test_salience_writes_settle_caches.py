"""a raw salience UPDATE on intentions evicts every row it may have changed, however it ends.

``intention_log``'s dedup-refresh reads a want with ``get()`` and saves it back, so a cached
pre-write salience is written over L3. The eviction after ``bump_salience`` / ``decay_salience``
is what prevents that; these pin that it runs when the UPDATE raises after reaching L3, and that
a cancellation cannot stop it partway. Asserted on the broadcast, not a local re-read.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest

from threetears.core.collections.registry import CacheInvalidationMessage, CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeNatsClient

from threetears.agent.intention.collections import IntentionsCollection


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
        return [{"agent_id": agent_id, "intention_id": intention_id} for agent_id, intention_id in self.returned]


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


def _collection(pool: Any, nats: FakeNatsClient) -> IntentionsCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool, kv_key_scope="salience-principal")
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return IntentionsCollection(registry, config, nats_client=nats)


def _evicted(nats: FakeNatsClient) -> set[tuple[str, ...]]:
    return {
        tuple(message.ids)
        for message in nats.published
        if isinstance(message, CacheInvalidationMessage) and message.table == "intentions"
    }


@pytest.mark.asyncio
async def test_a_bump_that_raises_still_evicts() -> None:
    nats = FakeNatsClient()
    agent_id, ids = uuid.uuid4(), [uuid.uuid4(), uuid.uuid4()]

    with pytest.raises(ConnectionError):
        await _collection(_Pool(), nats).bump_salience(ids, agent_id=agent_id, access_bump=0.1)

    assert _evicted(nats) == {(str(agent_id), str(i)) for i in ids}


@pytest.mark.asyncio
async def test_a_cancelled_decay_still_evicts_every_decayed_row() -> None:
    nats = _SuspendingNats()
    agent_id = uuid.uuid4()
    decayed = [(agent_id, uuid.uuid4()) for _ in range(3)]

    task = asyncio.create_task(_collection(_Pool(decayed), nats).decay_salience(half_life_days=14.0, floor=0.05))
    await asyncio.wait_for(nats.first_publish_started.wait(), timeout=5.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    nats.release.set()
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    await asyncio.gather(*pending)

    assert _evicted(nats) == {(str(a), str(i)) for a, i in decayed}
