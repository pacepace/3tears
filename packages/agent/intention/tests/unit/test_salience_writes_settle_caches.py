"""a raw salience UPDATE on intentions evicts every row it may have changed, however it ends.

``intention_log``'s dedup-refresh reads a want with ``get()`` and saves it back, so a cached
pre-write salience is written over L3. The eviction after ``bump_salience``
is what prevents that; these pin that it runs when the UPDATE raises after reaching L3, and that
a cancellation cannot stop it partway. Asserted on the broadcast, not a local re-read.
"""

from __future__ import annotations

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
