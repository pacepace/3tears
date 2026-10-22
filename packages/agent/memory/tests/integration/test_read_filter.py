"""a host's read filter restricts every memory read, inside the SQL, on real Postgres.

One owner holds three memories tagged ``keep``, ``hide`` and ``secret``. Each
has a passage (chunk) and a file (media row with a content row) under it. Every
filtered method is read four ways:

- no filter, and a filter that answers ``None``: all three come back;
- a filter binding one value (``hide``): ``keep`` and ``secret`` come back;
- a filter binding two values (``hide``, ``secret``): only ``keep`` comes back.

The two-value filter is what proves the placeholder numbering: its values land
after whatever the method binds itself (an embedding, dates, a scope, a cursor,
a limit), and a number off by one either fails the statement or tests the
wrong value. Chunk and media reads are decided on their memory's tags.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import asyncpg
import pytest
import uuid_utils

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import (
    MediaContentCollection,
    MemoriesCollection,
    MemoryChunkCollection,
    MemoryConsolidationsCollection,
)
from threetears.agent.memory.extraction import ExtractionResult, MemoryExtractor
from threetears.agent.memory.migrations import register as register_memory
from threetears.agent.memory.read_filter import ReadFilter
from threetears.agent.memory.retrieval_scope import RetrievalScope
from threetears.agent.memory.tools import load_memory_keep_tool
from threetears.agent.memory.types import MemoryConfig
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore
from .memory_support import PurposeChatModelFactory, StubEmbeddings, make_pool, vec, vec_sql

pytestmark = pytest.mark.integration

LABELS = frozenset({"keep", "hide", "secret"})
_WEIGHTS = {"semantic": 0.5, "keyword": 0.3, "recency": 0.2}
_CHUNK_WEIGHTS = {"semantic": 0.6, "keyword": 0.4}
_NO_CHUNK = uuid.UUID(int=0)  # a cursor below every uuid7, so a cursor read still sees every passage


def _hide_one(tags: str, bind: Callable[[Any], str]) -> str:
    return f"NOT (COALESCE({tags}, '[]'::jsonb) ? {bind('hide')})"


def _hide_two(tags: str, bind: Callable[[Any], str]) -> str:
    held = f"COALESCE({tags}, '[]'::jsonb)"
    return f"NOT ({held} ? {bind('hide')}) AND NOT ({held} ? {bind('secret')})"


def _no_restriction(tags: str, bind: Callable[[Any], str]) -> None:
    _ = tags, bind


#: each filter and what it leaves readable
_FILTERED: list[tuple[ReadFilter, frozenset[str]]] = [
    (_hide_one, frozenset({"keep", "secret"})),
    (_hide_two, frozenset({"keep"})),
]


@dataclass
class World:
    """one owner's three memories, their passages and files, and the collections over them."""

    pool: asyncpg.Pool
    memories: MemoriesCollection
    chunks: MemoryChunkCollection
    media: MediaContentCollection
    edges: MemoryConsolidationsCollection
    agent: uuid.UUID = field(default_factory=uuid.uuid4)
    customer: uuid.UUID = field(default_factory=uuid.uuid4)
    user: uuid.UUID = field(default_factory=uuid.uuid4)
    conversation: uuid.UUID = field(default_factory=uuid.uuid4)
    memory: dict[str, uuid.UUID] = field(default_factory=dict)
    chunk: dict[str, uuid.UUID] = field(default_factory=dict)
    content: dict[str, uuid.UUID] = field(default_factory=dict)
    message: dict[str, uuid.UUID] = field(default_factory=dict)

    def use(self, read_filter: ReadFilter | None) -> None:
        for collection in (self.memories, self.chunks, self.media):
            collection.set_read_filter(read_filter)

    def labels(self, kind: dict[str, uuid.UUID], ids: Any) -> set[str]:
        back = {value: label for label, value in kind.items()}
        return {back[uuid.UUID(str(i))] for i in ids}

    @property
    def owner(self) -> dict[str, uuid.UUID]:
        return {"agent_id": self.agent, "user_id": self.user}

    @property
    def window(self) -> dict[str, datetime]:
        now = datetime.now(UTC)
        return {"date_after": now - timedelta(days=1), "date_before": now + timedelta(days=1)}


async def expect_filtered(world: World, read: Callable[[], Awaitable[set[str]]]) -> None:
    """``read`` returns every label with no restriction, and only the readable ones under each filter."""
    world.use(None)
    assert await read() == LABELS, "no filter"
    world.use(_no_restriction)
    assert await read() == LABELS, "a filter answering None"
    for read_filter, readable in _FILTERED:
        world.use(read_filter)
        assert await read() == readable, read_filter.__name__


@pytest.fixture
async def world(pg_schema: tuple[str, str]) -> AsyncIterator[World]:
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
    pool = await make_pool(url, schema)
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    w = World(
        pool=pool,
        memories=MemoriesCollection(
            registry=registry, config=config, authorizer=MagicMock(spec=MemoryAuthorizerDependencies)
        ),
        chunks=MemoryChunkCollection(registry=registry, config=config),
        media=MediaContentCollection(registry=registry, config=config),
        edges=MemoryConsolidationsCollection(registry=registry, config=config, nats_client=None),
    )
    try:
        for label in sorted(LABELS):
            await _seed(w, label)
        yield w
    finally:
        await pool.close()


async def _seed(w: World, label: str) -> None:
    memory_id, message_id, chunk_id = uuid.uuid4(), uuid.uuid4(), uuid.UUID(str(uuid_utils.uuid7()))
    media_id, content_id = uuid.uuid4(), uuid.uuid4()
    text = f"the lighthouse keeper logs the harbour tide ({label})"
    await w.pool.execute(
        "INSERT INTO memories (memory_id, agent_id, customer_id, user_id, conversation_id, message_id_source, "
        "type_memory, content, tags, alias, embedding, date_created, date_updated) "
        "VALUES ($1, $2, $3, $4, $5, $6, 'fact', $7, $8::jsonb, $9, $10::text::public.vector, now(), now())",
        memory_id,
        w.agent,
        w.customer,
        w.user,
        w.conversation,
        message_id,
        text,
        ["doc", label],
        f"alias-{label}",
        vec_sql(1.0),
    )
    await w.pool.execute(
        "INSERT INTO memory_chunks (chunk_id, agent_id, customer_id, memory_id, user_id, chunk_index, content, "
        "token_count, embedding, message_id_start, message_id_end, date_created) "
        "VALUES ($1, $2, $3, $4, $5, 0, $6, 1, $7::text::public.vector, $8, $8, now())",
        chunk_id,
        w.agent,
        w.customer,
        memory_id,
        w.user,
        text,
        vec_sql(1.0),
        message_id,
    )
    await w.pool.execute(
        "INSERT INTO media (media_id, memory_id, agent_id, customer_id, user_id, media_category, metadata_json, "
        "date_created, date_updated) VALUES ($1, $2, $3, $4, $5, 'document', '{}'::jsonb, now(), now())",
        media_id,
        memory_id,
        w.agent,
        w.customer,
        w.user,
    )
    await w.pool.execute(
        "INSERT INTO media_content (content_id, media_id, agent_id, customer_id, user_id, content_type, content, "
        "embedding, date_created) VALUES ($1, $2, $3, $4, $5, 'ocr', $6, $7::text::public.vector, now())",
        content_id,
        media_id,
        w.agent,
        w.customer,
        w.user,
        text,
        vec_sql(1.0),
    )
    w.memory[label], w.message[label], w.chunk[label], w.content[label] = memory_id, message_id, chunk_id, content_id


async def _each(probe: Callable[[str], Awaitable[bool]]) -> set[str]:
    """the labels a per-row read finds."""
    return {label for label in LABELS if await probe(label)}


class TestMemoriesReads:
    async def test_find_by_user(self, world: World) -> None:
        async def read() -> set[str]:
            found = await world.memories.find_by_user(world.user, agent_id=world.agent, customer_id=world.customer)
            return world.labels(world.memory, [e.memory_id for e in found])

        await expect_filtered(world, read)

    async def test_find_by_scope(self, world: World) -> None:
        async def read() -> set[str]:
            found = await world.memories.find_by_scope(world.agent, world.customer, world.user)
            return world.labels(world.memory, [e.memory_id for e in found])

        await expect_filtered(world, read)

    async def test_find_ids_by_tags(self, world: World) -> None:
        async def read() -> set[str]:
            found = await world.memories.find_ids_by_tags(
                agent_id=world.agent, customer_id=world.customer, user_id=world.user, tags_any=("doc", "other")
            )
            return world.labels(world.memory, found)

        await expect_filtered(world, read)

    async def test_find_similar_for_dedup(self, world: World) -> None:
        async def read() -> set[str]:
            found = await world.memories.find_similar_for_dedup(
                **world.owner, embedding=vec(1.0), top_k=10, threshold=0.0
            )
            return world.labels(world.memory, [r["memory_id"] for r in found])

        await expect_filtered(world, read)

    async def test_find_same_event(self, world: World) -> None:
        async def read() -> set[str]:
            found = await world.memories.find_same_event(
                **world.owner, embedding=vec(1.0), message_ids=list(world.message.values()), threshold=0.0, top_k=10
            )
            return world.labels(world.memory, [r["memory_id"] for r in found])

        await expect_filtered(world, read)

    async def test_find_by_alias(self, world: World) -> None:
        async def found(label: str) -> bool:
            return await world.memories.find_by_alias(**world.owner, alias=f"alias-{label}") is not None

        await expect_filtered(world, lambda: _each(found))

    async def _hybrid(self, world: World, **kwargs: Any) -> set[str]:
        defaults: dict[str, Any] = {
            "customer_id": world.customer,
            "embedding": vec(1.0),
            "user_text": "",
            "top_k": 10,
            "candidate_limit": 10,
            "similarity_threshold": 0.0,
            "recency_half_life_hours": 24.0,
            "signal_weights": _WEIGHTS,
        }
        found = await world.memories.hybrid_search(**world.owner, **{**defaults, **kwargs})
        return world.labels(world.memory, [r["memory_id"] for r in found])

    async def test_hybrid_search_vector_arm(self, world: World) -> None:
        await expect_filtered(world, lambda: self._hybrid(world))

    async def test_hybrid_search_keyword_arm(self, world: World) -> None:
        # no row clears a similarity above 1, so every result was admitted by the keyword arm alone
        await expect_filtered(
            world, lambda: self._hybrid(world, user_text="lighthouse harbour", similarity_threshold=1.1)
        )

    async def test_hybrid_search_history_arm_with_dates_and_scope(self, world: World) -> None:
        # a negative window puts every row in the history pool, so the history scan runs and returns them,
        # after the method's own date, scope, salience and history placeholders
        await expect_filtered(
            world,
            lambda: self._hybrid(
                world,
                user_text="lighthouse",
                scope=RetrievalScope(tags_any=("doc",)),
                recent_cap=5,
                recent_window_hours=-1.0,
                **world.window,
            ),
        )

    async def test_search_by_ids(self, world: World) -> None:
        async def read() -> set[str]:
            found = await world.memories.search_by_ids(list(world.memory.values()), world.user, agent_id=world.agent)
            return world.labels(world.memory, [r["memory_id"] for r in found])

        await expect_filtered(world, read)

    async def test_search_by_semantic(self, world: World) -> None:
        async def read() -> set[str]:
            found = await world.memories.search_by_semantic(
                **world.owner,
                embedding=vec(1.0),
                max_results=10,
                similarity_threshold=0.0,
                type_filter="fact",
                scope=RetrievalScope(tags_any=("doc",)),
                **world.window,
            )
            return world.labels(world.memory, [r["memory_id"] for r in found])

        await expect_filtered(world, read)

    async def test_search_by_fts(self, world: World) -> None:
        async def read() -> set[str]:
            found = await world.memories.search_by_fts(
                **world.owner,
                fts_text="lighthouse",
                max_results=10,
                type_filter="fact",
                scope=RetrievalScope(tags_any=("doc",)),
                **world.window,
            )
            return world.labels(world.memory, [r["memory_id"] for r in found])

        await expect_filtered(world, read)

    async def test_find_for_recall(self, world: World) -> None:
        async def found(label: str) -> bool:
            row = await world.memories.find_for_recall(memory_id=world.memory[label], **world.owner)
            return row is not None

        await expect_filtered(world, lambda: _each(found))

    async def test_list_for_user(self, world: World) -> None:
        async def read() -> set[str]:
            rows, total = await world.memories.list_for_user(**world.owner, type_filter="fact", **world.window)
            assert total == len(rows), "the count and the page disagree"
            return world.labels(world.memory, [r["memory_id"] for r in rows])

        await expect_filtered(world, read)

    async def test_fetch_content_for_recall(self, world: World) -> None:
        async def found(label: str) -> bool:
            return (
                await world.memories.fetch_content_for_recall(memory_id=world.memory[label], **world.owner) is not None
            )

        await expect_filtered(world, lambda: _each(found))

    @pytest.mark.parametrize("with_customer", [False, True], ids=["agents-only", "with-customer"])
    async def test_find_for_user_in_agents(self, world: World, with_customer: bool) -> None:
        async def read() -> set[str]:
            found = await world.memories.find_for_user_in_agents(
                user_id=world.user, agent_ids=(world.agent,), customer_id=world.customer if with_customer else None
            )
            return world.labels(world.memory, [e.memory_id for e in found])

        await expect_filtered(world, read)

    async def test_find_active_for_consolidation(self, world: World) -> None:
        async def read() -> set[str]:
            found = await world.memories.find_active_for_consolidation(
                world.agent, customer_id=world.customer, user_id=world.user
            )
            return world.labels(world.memory, [r["memory_id"] for r in found])

        await expect_filtered(world, read)


class TestChunkReads:
    """a passage answers to its parent memory's tags."""

    async def _hybrid(self, world: World, **kwargs: Any) -> set[str]:
        defaults: dict[str, Any] = {
            "customer_id": world.customer,
            "embedding": vec(1.0),
            "user_text": "",
            "candidate_k": 10,
            "similarity_threshold": 0.0,
            "chunk_signal_weights": _CHUNK_WEIGHTS,
        }
        found = await world.chunks.hybrid_search(**world.owner, **{**defaults, **kwargs})
        return world.labels(world.chunk, [r["chunk_id"] for r in found])

    async def test_hybrid_search_vector_arm(self, world: World) -> None:
        await expect_filtered(world, lambda: self._hybrid(world))

    async def test_hybrid_search_keyword_arm_with_cursor(self, world: World) -> None:
        await expect_filtered(
            world,
            lambda: self._hybrid(world, user_text="lighthouse", similarity_threshold=1.1, chunk_id_after=_NO_CHUNK),
        )

    async def test_search_by_ids(self, world: World) -> None:
        async def read() -> set[str]:
            found = await world.chunks.search_by_ids(list(world.chunk.values()), world.user, agent_id=world.agent)
            return world.labels(world.chunk, [r["chunk_id"] for r in found])

        await expect_filtered(world, read)

    async def test_search_by_semantic(self, world: World) -> None:
        async def read() -> set[str]:
            found = await world.chunks.search_by_semantic(
                **world.owner, embedding=vec(1.0), max_results=10, similarity_threshold=0.0, **world.window
            )
            return world.labels(world.chunk, [r["chunk_id"] for r in found])

        await expect_filtered(world, read)

    async def test_find_for_recall(self, world: World) -> None:
        async def found(label: str) -> bool:
            return await world.chunks.find_for_recall(chunk_id=world.chunk[label], **world.owner) is not None

        await expect_filtered(world, lambda: _each(found))

    async def test_fetch_content_for_recall(self, world: World) -> None:
        async def found(label: str) -> bool:
            return await world.chunks.fetch_content_for_recall(chunk_id=world.chunk[label], **world.owner) is not None

        await expect_filtered(world, lambda: _each(found))

    async def test_find_by_memory_id_with_cursor(self, world: World) -> None:
        async def found(label: str) -> bool:
            rows = await world.chunks.find_by_memory_id(
                world.memory[label], **world.owner, customer_id=world.customer, chunk_id_after=_NO_CHUNK
            )
            return bool(rows)

        await expect_filtered(world, lambda: _each(found))

    async def test_find_by_chunk_indexes(self, world: World) -> None:
        async def found(label: str) -> bool:
            rows = await world.chunks.find_by_chunk_indexes(
                world.memory[label], **world.owner, customer_id=world.customer, chunk_indexes=[0]
            )
            return bool(rows)

        await expect_filtered(world, lambda: _each(found))

    async def test_find_by_conversation_id_with_cursor(self, world: World) -> None:
        async def read() -> set[str]:
            rows = await world.chunks.find_by_conversation_id(
                world.conversation, **world.owner, customer_id=world.customer, chunk_id_after=_NO_CHUNK
            )
            return world.labels(world.chunk, [r["chunk_id"] for r in rows])

        await expect_filtered(world, read)

    @pytest.mark.parametrize(
        ("user_text", "threshold"), [("", 0.0), ("lighthouse", 1.1)], ids=["vector-arm", "keyword-arm"]
    )
    async def test_hybrid_search_within_memory(self, world: World, user_text: str, threshold: float) -> None:
        async def found(label: str) -> bool:
            rows = await world.chunks.hybrid_search_within_memory(
                memory_id=world.memory[label],
                **world.owner,
                customer_id=world.customer,
                embedding=vec(1.0),
                user_text=user_text,
                candidate_k=10,
                similarity_threshold=threshold,
                chunk_signal_weights=_CHUNK_WEIGHTS,
            )
            return bool(rows)

        await expect_filtered(world, lambda: _each(found))

    async def test_count_by_memory_id(self, world: World) -> None:
        async def found(label: str) -> bool:
            count = await world.chunks.count_by_memory_id(
                world.memory[label], **world.owner, customer_id=world.customer
            )
            return count > 0

        await expect_filtered(world, lambda: _each(found))


class TestMediaContentReads:
    """a file's content answers to the tags of the memory its media row is paired with."""

    async def _hybrid(self, world: World, **kwargs: Any) -> set[str]:
        defaults: dict[str, Any] = {
            "customer_id": world.customer,
            "embedding": vec(1.0),
            "user_text": "",
            "top_k": 10,
            "candidate_limit": 10,
            "similarity_threshold": 0.0,
            "recency_half_life_hours": 24.0,
            "signal_weights": _WEIGHTS,
        }
        found = await world.media.hybrid_search(**world.owner, **{**defaults, **kwargs})
        return world.labels(world.content, [r["content_id"] for r in found])

    async def test_hybrid_search_vector_arm(self, world: World) -> None:
        await expect_filtered(world, lambda: self._hybrid(world))

    async def test_hybrid_search_keyword_arm(self, world: World) -> None:
        await expect_filtered(world, lambda: self._hybrid(world, user_text="lighthouse", similarity_threshold=1.1))

    async def test_search_by_ids(self, world: World) -> None:
        async def read() -> set[str]:
            found = await world.media.search_by_ids(list(world.content.values()), world.user, agent_id=world.agent)
            return world.labels(world.content, [r["content_id"] for r in found])

        await expect_filtered(world, read)

    async def test_search_by_semantic(self, world: World) -> None:
        async def read() -> set[str]:
            found = await world.media.search_by_semantic(
                **world.owner, embedding=vec(1.0), max_results=10, similarity_threshold=0.0, **world.window
            )
            return world.labels(world.content, [r["content_id"] for r in found])

        await expect_filtered(world, read)

    async def test_search_by_fts(self, world: World) -> None:
        async def read() -> set[str]:
            found = await world.media.search_by_fts(
                **world.owner, fts_text="lighthouse", max_results=10, **world.window
            )
            return world.labels(world.content, [r["content_id"] for r in found])

        await expect_filtered(world, read)

    async def test_fetch_content_for_recall(self, world: World) -> None:
        async def found(label: str) -> bool:
            text = await world.media.fetch_content_for_recall(content_id=world.content[label], **world.owner)
            return text is not None

        await expect_filtered(world, lambda: _each(found))


class TestWhatTheFilterDoesNotGate:
    async def test_a_delete_by_tag_reaches_rows_the_task_cannot_read(self, world: World) -> None:
        world.use(_hide_two)
        deleted = await world.memories.delete_by_tags(
            agent_id=world.agent, customer_id=world.customer, user_id=world.user, tags_any=("doc",)
        )
        assert deleted == 3
        assert await world.pool.fetchval("SELECT count(*) FROM memories") == 0


class TestTheFilterIsAskedOnEveryRead:
    async def test_two_tasks_with_different_context_read_differently(self, world: World) -> None:
        hidden: contextvars.ContextVar[str | None] = contextvars.ContextVar("hidden", default=None)

        def per_task(tags: str, bind: Callable[[Any], str]) -> str | None:
            label = hidden.get()
            return None if label is None else f"NOT (COALESCE({tags}, '[]'::jsonb) ? {bind(label)})"

        world.use(per_task)

        async def read_as(label: str | None) -> set[str]:
            hidden.set(label)
            found = await world.memories.search_by_ids(list(world.memory.values()), world.user, agent_id=world.agent)
            return world.labels(world.memory, [r["memory_id"] for r in found])

        # each task runs in its own copy of the context, concurrently, over the one collection
        results = await asyncio.gather(read_as("hide"), read_as("secret"), read_as(None))
        assert results == [{"keep", "secret"}, {"keep", "hide"}, set(LABELS)]


class TestMemoryKeep:
    async def test_a_memory_the_task_cannot_read_is_not_found_to_keep(
        self, world: World, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        [keep] = load_memory_keep_tool(
            world.user, world.agent, world.customer, permissive_memory_authorizer, world.memories
        )
        world.use(_hide_one)

        refused = await keep.ainvoke({"memory_id": str(world.memory["hide"])})
        kept = await keep.ainvoke({"memory_id": str(world.memory["keep"])})

        assert "No memory" in refused
        assert "harbour" not in refused, "the refusal shows the content it should not"
        assert "kept permanently" in kept
        evergreen = await world.pool.fetchval(
            "SELECT evergreen FROM memories WHERE memory_id = $1", world.memory["hide"]
        )
        assert evergreen is False


class TestExtractionTargets:
    """the resolution model names the memory it updates or retires; one this task cannot read is left alone."""

    async def _resolve(
        self, world: World, authorizer: MemoryAuthorizerDependencies, action: dict[str, Any]
    ) -> ExtractionResult:
        extractor = MemoryExtractor(
            config=MemoryConfig(),
            embedding_provider=StubEmbeddings(vec(1.0)),
            chat_model_factory=PurposeChatModelFactory(
                worthiness=json.dumps({"worthy": True, "reason": "has facts"}),
                extraction=json.dumps([{"type": "fact", "content": action.get("content", "the harbour tide")}]),
                resolution=json.dumps([{"index": 0, **action}]),
            ),
            authorizer=authorizer,
            memories_collection=world.memories,
            consolidations_collection=world.edges,
        )
        return await extractor.extract(
            user_id=world.user,
            conversation_id=world.conversation,
            message_id_source=uuid.uuid4(),
            user_message="x" * 50,
            assistant_response="y" * 200,
            turn_count=10,
            agent_id=world.agent,
            customer_id=world.customer,
        )

    async def _state(self, world: World, label: str) -> tuple[float, Any, list[str]]:
        row = await world.pool.fetchrow(
            "SELECT salience, superseded_by, tags FROM memories WHERE memory_id = $1", world.memory[label]
        )
        assert row is not None
        tags = row["tags"] if isinstance(row["tags"], list) else json.loads(row["tags"])
        return float(row["salience"]), row["superseded_by"], tags

    @pytest.mark.parametrize("verb", ["DELETE", "UPDATE"])
    async def test_a_hidden_target_is_skipped_and_a_readable_one_is_not(
        self, world: World, permissive_memory_authorizer: MemoryAuthorizerDependencies, verb: str
    ) -> None:
        world.use(_hide_one)
        extra = {"content": "the harbour tide turns at noon"} if verb == "UPDATE" else {}

        await self._resolve(
            world, permissive_memory_authorizer, {"action": verb, "memory_id": str(world.memory["hide"]), **extra}
        )
        await self._resolve(
            world, permissive_memory_authorizer, {"action": verb, "memory_id": str(world.memory["keep"]), **extra}
        )

        assert await self._state(world, "hide") == (0.5, None, ["doc", "hide"]), "a hidden memory was changed"
        salience, superseded_by, tags = await self._state(world, "keep")
        assert (salience == 0.0 and "retracted" in tags) if verb == "DELETE" else superseded_by is not None
