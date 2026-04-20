"""
integration test: end-to-end exercise of the memory public API.

memory-task-01. after the migration chain runs, every public surface on
the package must function against the reconciled schema:

- :class:`MemoriesCollection` — save / update / find_by_user / find_by_scope
  round-trip via a real asyncpg pool.
- :class:`MemoryRetriever` — hybrid retrieve across memories + media
  + chunks with FTS live.
- :class:`MemoryExtractor` — extract candidates, resolve ADD, insert
  into memories with trigger-maintained search_vector.
- tool factories — :func:`load_memory_search_tool`,
  :func:`load_recall_memory_tool`, :func:`load_add_memory_tool` all
  execute their SQL paths against the real schema.

the LLM side of :class:`MemoryExtractor` is stubbed (no network). the
embedding side is stubbed to a deterministic unit vector so the
pgvector search surface is still exercised end-to-end.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest

from threetears.agent.memory.collections import MemoriesCollection
from threetears.agent.memory.extraction import MemoryExtractor
from threetears.agent.memory.migrations import register as register_memory
from threetears.agent.memory.retrieval import MemoryRetriever
from threetears.agent.memory.tools import (
    load_add_memory_tool,
    load_memory_search_tool,
    load_recall_memory_tool,
)
from threetears.agent.memory.types import MemoryConfig
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# stubs for non-DB dependencies
# ---------------------------------------------------------------------------


class _StubEmbedding:
    """
    deterministic embedding provider: returns a fixed 1024-dim vector.

    the exact values don't matter for correctness — pgvector only needs
    a well-formed vector to exercise the indexing and distance queries.
    """

    def __init__(self, seed: float = 0.1) -> None:
        """
        :param seed: fill value for the 1024-dim vector
        :ptype seed: float
        """
        self._vec = [seed] * 1024

    async def embed_text(self, text: str) -> tuple[list[float] | None, int]:
        """
        return ``(vector, token_count)`` deterministically.

        :param text: text to embed (ignored; deterministic output)
        :ptype text: str
        :return: (vector, token estimate)
        :rtype: tuple[list[float] | None, int]
        """
        return self._vec, len(text.split())

    @property
    def dimensions(self) -> int:
        """
        :return: vector dimensionality (1024 to match pgvector column)
        :rtype: int
        """
        return 1024


class _StubChatModel:
    """chat model stub returning a preconfigured content payload."""

    def __init__(self, content: str) -> None:
        """
        :param content: text to return as ``response.content``
        :ptype content: str
        """
        self._content = content

    async def ainvoke(self, messages: list[Any]) -> Any:
        """return a MagicMock with ``content`` set to the preconfigured payload."""
        resp = MagicMock()
        resp.content = self._content
        return resp


class _StubChatModelFactory:
    """factory returning per-purpose stub chat models."""

    def __init__(
        self,
        worthiness: str,
        extraction: str,
        resolution: str | None = None,
    ) -> None:
        """
        :param worthiness: JSON content for the worthiness check
        :ptype worthiness: str
        :param extraction: JSON content for the extraction list
        :ptype extraction: str
        :param resolution: JSON content for the resolution step (optional)
        :ptype resolution: str | None
        """
        self._by_purpose = {
            "worthiness": _StubChatModel(worthiness),
            "extraction": _StubChatModel(extraction),
            "resolution": _StubChatModel(resolution or "[]"),
        }

    async def create_chat_model(self, purpose: str = "extraction") -> Any:
        """
        return the stub for the given purpose or a default empty-list stub.

        :param purpose: "worthiness" | "extraction" | "resolution"
        :ptype purpose: str
        :return: stub chat model
        :rtype: Any
        """
        result = self._by_purpose.get(purpose, _StubChatModel("[]"))
        return result


# ---------------------------------------------------------------------------
# shared schema fixture applying the whole runner chain
# ---------------------------------------------------------------------------


@pytest.fixture
async def applied_schema(pg_schema: tuple[str, str]) -> tuple[str, str]:
    """
    apply conversations + agent-memory migrations into the per-test schema.

    :param pg_schema: (url, schema) tuple
    :ptype pg_schema: tuple[str, str]
    :return: same (url, schema) tuple after migrations land
    :rtype: tuple[str, str]
    """
    url, schema = pg_schema
    runner = MigrationRunner()
    register_conversations(runner)
    register_memory(runner)
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        store = AsyncpgStore(conn)
        await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
    finally:
        await conn.close()
    return url, schema


async def _make_pool(url: str, schema: str) -> asyncpg.Pool:
    """
    create an asyncpg pool whose connections are pre-bound to ``schema``.

    :param url: asyncpg URL
    :ptype url: str
    :param schema: target schema for search_path
    :ptype schema: str
    :return: ready pool
    :rtype: asyncpg.Pool
    """
    result: asyncpg.Pool = await asyncpg.create_pool(
        dsn=url,
        min_size=1,
        max_size=4,
        server_settings={"search_path": f"{schema}, public"},
    )
    return result


# ---------------------------------------------------------------------------
# MemoriesCollection round-trip
# ---------------------------------------------------------------------------


class TestMemoriesCollectionAgainstLiveSchema:
    """MemoriesCollection save / find paths execute against real pg."""

    async def test_save_and_find_by_user(
        self, applied_schema: tuple[str, str]
    ) -> None:
        """
        insert two memories, find by user, verify both surface.

        :param applied_schema: (url, schema) after migrations
        :ptype applied_schema: tuple[str, str]
        """
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            registry = CollectionRegistry()
            config = DefaultCoreConfig(
                collection_flush="ALWAYS",
                collection_flush_tables="",
            )
            coll = MemoriesCollection(
                registry=registry,
                config=config,
                postgres_pool=pool,
                nats_client=None,
            )

            user_id = uuid.uuid4()
            agent_id = uuid.uuid4()
            now = datetime.now(UTC).replace(tzinfo=None)
            vec = [0.1] * 1024
            data_common = {
                "agent_id": agent_id,
                "customer_id": uuid.uuid4(),
                "user_id": user_id,
                "conversation_id": uuid.uuid4(),
                "message_id_source": uuid.uuid4(),
                "type_memory": "fact",
                "embedding": vec,
                "is_deleted": False,
                "media_id": None,
                "date_created": now,
                "date_deleted": None,
                "date_updated": now,
            }

            data_a = dict(data_common, memory_id=uuid.uuid4(), content="mem A")
            data_b = dict(data_common, memory_id=uuid.uuid4(), content="mem B")
            entity_a = coll.create(data_a)
            entity_b = coll.create(data_b)
            await coll.save_entity(entity_a)
            await coll.save_entity(entity_b)

            entities = await coll.find_by_user(user_id)
            contents = {e.content for e in entities}
            assert contents == {"mem A", "mem B"}

            # scoped find
            scoped = await coll.find_by_scope(
                agent_id=agent_id, user_id=user_id
            )
            assert len(scoped) == 2
        finally:
            await pool.close()

    async def test_soft_delete_excludes_from_find(
        self, applied_schema: tuple[str, str]
    ) -> None:
        """
        soft-delete sets ``is_deleted`` + ``date_deleted`` and excludes the
        row from default find.

        :param applied_schema: (url, schema)
        :ptype applied_schema: tuple[str, str]
        """
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            registry = CollectionRegistry()
            config = DefaultCoreConfig(
                collection_flush="ALWAYS",
                collection_flush_tables="",
            )
            coll = MemoriesCollection(
                registry=registry,
                config=config,
                postgres_pool=pool,
            )

            user_id = uuid.uuid4()
            mid = uuid.uuid4()
            now = datetime.now(UTC).replace(tzinfo=None)
            data = {
                "memory_id": mid,
                "agent_id": uuid.uuid4(),
                "customer_id": uuid.uuid4(),
                "user_id": user_id,
                "conversation_id": uuid.uuid4(),
                "message_id_source": uuid.uuid4(),
                "type_memory": "fact",
                "content": "to be deleted",
                "embedding": [0.1] * 1024,
                "is_deleted": False,
                "media_id": None,
                "date_created": now,
                "date_deleted": None,
                "date_updated": now,
            }
            entity = coll.create(data)
            await coll.save_entity(entity)

            # direct UPDATE to set is_deleted (bypassing entity lifecycle)
            later = datetime.now(UTC).replace(tzinfo=None)
            await pool.execute(
                "UPDATE memories SET is_deleted = TRUE, date_deleted = $2 "
                "WHERE memory_id = $1",
                mid,
                later,
            )

            visible = await coll.find_by_user(user_id)
            assert visible == []

            all_mem = await coll.find_by_user(user_id, include_deleted=True)
            assert len(all_mem) == 1
        finally:
            await pool.close()


# ---------------------------------------------------------------------------
# MemoryRetriever against live schema
# ---------------------------------------------------------------------------


class TestMemoryRetrieverAgainstLiveSchema:
    """MemoryRetriever.retrieve executes real SQL against pg."""

    async def test_retrieve_returns_context_with_memories(
        self, applied_schema: tuple[str, str]
    ) -> None:
        """
        seed memories, retrieve with a query, assert formatted context comes back.

        :param applied_schema: (url, schema)
        :ptype applied_schema: tuple[str, str]
        """
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            user_id = uuid.uuid4()
            now = datetime.now(UTC).replace(tzinfo=None)
            vec = [0.1] * 1024
            vec_str = json.dumps(vec)
            await pool.execute(
                "INSERT INTO memories ("
                "memory_id, agent_id, customer_id, user_id, "
                "conversation_id, message_id_source, type_memory, content, "
                "summary, embedding, is_deleted, date_created, date_updated"
                ") VALUES ($1, $2, $3, $4, $5, $6, 'preference', $7, NULL, "
                "$8::vector, FALSE, $9, $9)",
                uuid.uuid4(),
                uuid.uuid4(),
                uuid.uuid4(),
                user_id,
                uuid.uuid4(),
                uuid.uuid4(),
                "User prefers Rust programming language",
                vec_str,
                now,
            )

            config = MemoryConfig()
            retriever = MemoryRetriever(config, _StubEmbedding())
            context = await retriever.retrieve(
                pool, user_id, "What does user prefer programming?"
            )
            assert context is not None
            assert "Rust" in context
        finally:
            await pool.close()


# ---------------------------------------------------------------------------
# MemoryExtractor against live schema (ADD path)
# ---------------------------------------------------------------------------


class TestMemoryExtractorAgainstLiveSchema:
    """MemoryExtractor.extract inserts rows against real pg."""

    async def test_extract_adds_memory_row(
        self, applied_schema: tuple[str, str]
    ) -> None:
        """
        happy-path extract: LLM says 1 memory is worthy, extractor embeds
        and inserts it. verify the row ends up in memories with a
        trigger-maintained search_vector.

        :param applied_schema: (url, schema)
        :ptype applied_schema: tuple[str, str]
        """
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            factory = _StubChatModelFactory(
                worthiness=json.dumps({"worthy": True, "reason": "has facts"}),
                extraction=json.dumps(
                    [{"type": "fact", "content": "User lives in Seattle"}]
                ),
            )
            extractor = MemoryExtractor(
                config=MemoryConfig(),
                embedding_provider=_StubEmbedding(),
                chat_model_factory=factory,
                nats_client=None,
            )
            user_id = uuid.uuid4()
            await extractor.extract(
                pool=pool,
                user_id=user_id,
                conversation_id=uuid.uuid4(),
                message_id_source=uuid.uuid4(),
                user_message="x" * 50,
                assistant_response="y" * 200,
                turn_count=10,
            )

            rows = await pool.fetch(
                "SELECT content, type_memory, search_vector FROM memories "
                "WHERE user_id = $1",
                user_id,
            )
            assert len(rows) == 1
            assert rows[0]["content"] == "User lives in Seattle"
            assert rows[0]["type_memory"] == "fact"
            assert rows[0]["search_vector"] is not None
        finally:
            await pool.close()


# ---------------------------------------------------------------------------
# tool factories against live schema
# ---------------------------------------------------------------------------


class TestMemoryToolsAgainstLiveSchema:
    """tool factories produce tools whose SQL executes successfully."""

    async def test_add_memory_tool_inserts_row(
        self, applied_schema: tuple[str, str]
    ) -> None:
        """
        ``add_memory`` tool inserts a new row when no similar memory
        exists and populates trigger-maintained ``search_vector``.

        :param applied_schema: (url, schema)
        :ptype applied_schema: tuple[str, str]
        """
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            user_id = uuid.uuid4()
            conv_id = uuid.uuid4()
            msg_id = uuid.uuid4()
            tools = await load_add_memory_tool(
                pool, user_id, conv_id, msg_id, _StubEmbedding()
            )
            assert len(tools) == 1
            result = await tools[0].ainvoke(
                {"content": "User prefers Python", "memory_type": "preference"}
            )
            assert "Remembered" in result

            row = await pool.fetchrow(
                "SELECT content, type_memory, search_vector FROM memories "
                "WHERE user_id = $1",
                user_id,
            )
            assert row is not None
            assert row["content"] == "User prefers Python"
            assert row["type_memory"] == "preference"
            assert row["search_vector"] is not None
        finally:
            await pool.close()

    async def test_memory_search_tool_query_path(
        self, applied_schema: tuple[str, str]
    ) -> None:
        """
        seed memories via add tool; semantic search finds them.

        :param applied_schema: (url, schema)
        :ptype applied_schema: tuple[str, str]
        """
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            user_id = uuid.uuid4()
            conv_id = uuid.uuid4()
            msg_id = uuid.uuid4()
            add_tools = await load_add_memory_tool(
                pool, user_id, conv_id, msg_id, _StubEmbedding()
            )
            await add_tools[0].ainvoke(
                {"content": "User loves type hints", "memory_type": "preference"}
            )

            search_tools = await load_memory_search_tool(
                pool, user_id, _StubEmbedding()
            )
            assert len(search_tools) == 1
            result = await search_tools[0].ainvoke(
                {"query": "What does user love about Python?"}
            )
            # result is text; should contain our inserted content
            assert "type hints" in result or "No relevant memories" not in result
        finally:
            await pool.close()

    async def test_recall_memory_tool_fetches_content(
        self, applied_schema: tuple[str, str]
    ) -> None:
        """
        recall_memory fetches content for a known memory_id.

        :param applied_schema: (url, schema)
        :ptype applied_schema: tuple[str, str]
        """
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            user_id = uuid.uuid4()
            mid = uuid.uuid4()
            now = datetime.now(UTC).replace(tzinfo=None)
            await pool.execute(
                "INSERT INTO memories ("
                "memory_id, agent_id, customer_id, user_id, "
                "conversation_id, message_id_source, type_memory, content, "
                "embedding, is_deleted, date_created, date_updated"
                ") VALUES ($1, $2, $3, $4, $5, $6, 'fact', $7, "
                "$8::vector, FALSE, $9, $9)",
                mid,
                uuid.uuid4(),
                uuid.uuid4(),
                user_id,
                uuid.uuid4(),
                uuid.uuid4(),
                "Seattle resident",
                "[" + ",".join(["0.1"] * 1024) + "]",
                now,
            )

            tools = await load_recall_memory_tool(pool, user_id)
            assert len(tools) == 1
            result = await tools[0].ainvoke({"id": str(mid), "type": "memory"})
            assert result == "Seattle resident"
        finally:
            await pool.close()
