"""integration: ``on_memory_created`` gets a readable memory after a slow summary.

metallm 0.56.0, twice in one run: the summary callback waited out its 120 s
model timeout, and in that window the next turn's ambient retrieval surfaced the
new memory. ``bump_salience`` invalidates each surfaced row, which drops it from
L1. The extractor then handed ``on_memory_created`` the entity it built before
the wait. Its field reads are L1-only, so ``memory_id`` read ``None`` and the
coercer raised "badly formed hexadecimal UUID string"; the push was lost.

The summary callback below does what that turn did, with a real SQLite L1 and a
real Postgres L3, so the eviction is the production one.
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import MagicMock

import asyncpg
import pytest
from sqlalchemy import MetaData

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import MemoriesCollection
from threetears.agent.memory.entities import MemoryEntity
from threetears.agent.memory.extraction import MemoryExtractor
from threetears.agent.memory.migrations import register as register_memory
from threetears.agent.memory.types import MemoryConfig
from threetears.conversations.migrations import register as register_conversations
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore

pytestmark = pytest.mark.integration

_AGENT_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
_CUSTOMER_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")


class _Embedding:
    """a fixed 1024-dim vector, the width of the pgvector column."""

    async def aembed_query(self, text: str) -> list[float]:
        _ = text
        return [0.1] * 1024

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1024 for _ in texts]


class _ChatModelFactory:
    """worthy, one fact, nothing to resolve against."""

    def __init__(self, content: str) -> None:
        self._content = content

    async def create_chat_model(self, purpose: str = "extraction") -> Any:
        payloads = {
            "worthiness": json.dumps({"worthy": True}),
            "extraction": json.dumps([{"type": "fact", "content": self._content}]),
        }
        model = MagicMock()

        async def _ainvoke(messages: list[Any], **kwargs: Any) -> Any:
            _ = messages, kwargs
            response = MagicMock()
            response.content = payloads.get(purpose, "[]")
            return response

        model.ainvoke = _ainvoke
        return model


@pytest.fixture
async def pool(pg_schema: tuple[str, str]) -> Any:
    url, schema = pg_schema
    runner = MigrationRunner()
    register_conversations(runner)
    register_memory(runner)
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        await runner.apply_for_agent_schema(AsyncpgStore(conn))  # type: ignore[arg-type]
    finally:
        await conn.close()
    created = await asyncpg.create_pool(
        dsn=url,
        min_size=1,
        max_size=4,
        server_settings={"search_path": f"{schema}, public"},
        init=init_connection,
    )
    yield created
    await created.close()


def _memories(pool: asyncpg.Pool, authorizer: MemoryAuthorizerDependencies) -> MemoriesCollection:
    metadata = MetaData()
    MemoriesCollection.schema.to_sqlalchemy_table(metadata)
    l1 = SQLiteBackend(db_name=f"mem_cb_{uuid.uuid4().hex[:8]}")
    l1.initialize(metadata)
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l3_pool=pool)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return MemoriesCollection(registry=registry, config=config, authorizer=authorizer)


async def test_created_callback_reads_the_memory_after_retrieval_evicted_it(
    pool: asyncpg.Pool,
    permissive_memory_authorizer: MemoryAuthorizerDependencies,
) -> None:
    memories = _memories(pool, permissive_memory_authorizer)
    user_id = uuid.uuid7()
    conversation_id = uuid.uuid7()
    evicted: list[uuid.UUID] = []

    async def _slow_summary(memory_id: str, content: str) -> None:
        # the next turn's ambient retrieval, landing while the summary waits
        _ = content
        surfaced = uuid.UUID(memory_id)
        await memories.bump_salience([surfaced], agent_id=_AGENT_ID, access_bump=0.1)
        assert memories.get_row_sync((_AGENT_ID, surfaced)) is None, "retrieval did not evict the row"
        evicted.append(surfaced)

    pushed: list[tuple[uuid.UUID, uuid.UUID | None, uuid.UUID, str]] = []

    async def _push(entity: MemoryEntity) -> None:
        pushed.append((entity.memory_id, entity.user_id, entity.conversation_id, entity.type_memory))

    extractor = MemoryExtractor(
        config=MemoryConfig(),
        embedding_provider=_Embedding(),
        chat_model_factory=_ChatModelFactory("Takes coffee black, no sugar"),
        authorizer=permissive_memory_authorizer,
        memories_collection=memories,
        summary_callback=_slow_summary,
        on_memory_created=_push,
    )
    await extractor.extract(
        user_id=user_id,
        conversation_id=conversation_id,
        message_id_source=uuid.uuid7(),
        user_message="x" * 50,
        assistant_response="y" * 200,
        turn_count=10,
        agent_id=_AGENT_ID,
        customer_id=_CUSTOMER_ID,
    )

    assert len(evicted) == 1
    assert pushed == [(evicted[0], user_id, conversation_id, "fact")]
