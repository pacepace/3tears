"""the two memory reads that look their rows up by key behind a MATERIALIZED fence give the rows the plain joins gave.

``MediaContentCollection.search_by_ids`` and ``MemoryChunkCollection.find_by_conversation_id`` read their keyed rows
first (the ids, the conversation's memories) and check the owner after, so PostgreSQL and YugabyteDB both read by the
key (``test_index_plans``). Each is compared, through
:func:`~threetears.core.testing.same_results.assert_same_results`, with the statement it replaces: empty, one row,
many, NULLs, ties in the sort, a tenant boundary, a page boundary and one past it.

A content row's user is not constrained to be its file's (the key to the file is ``(agent_id, media_id)`` alone), so
``search_by_ids`` binds both. The seeds hold a content row of another user on the caller's file, which no form may
return; a content row of the caller's on another user's file is the one row binding both no longer returns, asserted
on its own.

Runs on PostgreSQL and YugabyteDB (the session's ``db_container``), a schema of its own.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator, Mapping
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import asyncpg
import pytest
import pytest_asyncio

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import MediaContentCollection, MemoriesCollection, MemoryChunkCollection
from threetears.agent.memory.migrations import register as register_memory
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.core.testing.same_results import Seeded, assert_same_results

from .conftest import AsyncpgStore

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

_NS = uuid.UUID("0c7f3e7a-51b2-4f43-9d0e-6a1f9b3e2d10")
_AGENTS = (uuid.uuid5(_NS, "agent-0"), uuid.uuid5(_NS, "agent-1"))
_USER = uuid.uuid5(_NS, "user")
_OTHER_USER = uuid.uuid5(_NS, "other-user")
_CUSTOMER = uuid.uuid5(_NS, "customer")
_CONVERSATION = uuid.uuid5(_NS, "conversation")
_T0 = datetime(2026, 9, 30, 12, 0)  # noqa: DTZ001 - the tables' columns are TIMESTAMP
#: a multiple of the ties' run of three, so a LIMIT never cuts a tie
_PAGE = 6


def _id(kind: str, index: int) -> uuid.UUID:
    return uuid.uuid5(_NS, f"{kind}-{index}")


#: memories each case's rows hang from: in each agent, the caller's memory in the conversation, another user's
#: memory in the conversation, and the caller's memory in another conversation
_MEMORIES = {
    (agent, name): (_id(f"memory-{name}", index), user, conversation)
    for index, agent in enumerate(_AGENTS)
    for name, user, conversation in (
        ("mine", _USER, _CONVERSATION),
        ("theirs", _OTHER_USER, _CONVERSATION),
        ("elsewhere", _USER, uuid.uuid5(_NS, "another conversation")),
    )
}
#: files each case's content rows hang from: in each agent, the caller's file and another user's file
_FILES = {
    (agent, name): (_id(f"file-{name}", index), user, _MEMORIES[(agent, "mine")][0])
    for index, agent in enumerate(_AGENTS)
    for name, user in (("mine", _USER), ("theirs", _OTHER_USER))
}


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def conn(pg_url: str) -> AsyncIterator[asyncpg.Connection]:
    """a fresh schema holding the memory tables."""
    schema = f"mem_same_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(pg_url)
    await connection.execute('CREATE EXTENSION IF NOT EXISTS "vector"')
    await connection.execute(f'CREATE SCHEMA "{schema}"')
    await connection.execute(f'SET search_path TO "{schema}", public')
    runner = MigrationRunner()
    register_conversations(runner)
    register_memory(runner)
    await runner.apply_for_agent_schema(AsyncpgStore(connection))
    try:
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


@contextlib.asynccontextmanager
async def _parents(connection: asyncpg.Connection) -> AsyncIterator[None]:
    """the memories and files the seeded rows hang from, for one test; gone after it."""
    await connection.executemany(
        "INSERT INTO memories (memory_id, agent_id, customer_id, user_id, conversation_id, type_memory, content,"
        " date_created, date_updated) VALUES ($1, $2, $3, $4, $5, 'fact', 'a memory', $6, $6)",
        [(memory, agent, _CUSTOMER, user, conv, _T0) for (agent, _), (memory, user, conv) in _MEMORIES.items()],
    )
    await connection.executemany(
        "INSERT INTO media (media_id, memory_id, agent_id, customer_id, user_id, date_created, date_updated)"
        " VALUES ($1, $2, $3, $4, $5, $6, $6)",
        [(media, memory, agent, _CUSTOMER, user, _T0) for (agent, _), (media, user, memory) in _FILES.items()],
    )
    try:
        yield
    finally:
        await connection.execute("DELETE FROM media")
        await connection.execute("DELETE FROM memories")


def _collection(kind: type[Any], pool: Any) -> Any:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    if kind is MemoriesCollection:
        return kind(registry, config, authorizer=MagicMock(spec=MemoryAuthorizerDependencies))
    return kind(registry=registry, config=config)


#: search_by_ids before the fence: the plain join, the user bound on the content
_IDS_SQL = """
    SELECT mc.content_id, mc.content, mc.content_type, mc.media_id,
           med.media_category, med.metadata_json, med.date_created
    FROM media_content mc
    JOIN media med
      ON mc.agent_id = med.agent_id
     AND mc.media_id = med.media_id
    WHERE mc.agent_id = $1
      AND (mc.content_id = ANY($2::uuid[]) OR mc.media_id = ANY($2::uuid[]))
      AND mc.user_id = $3
    ORDER BY mc.media_id, mc.date_created DESC
"""


def _content(seed: Seeded) -> Mapping[str, Any]:
    """a content row on the caller's file of its tenant's agent; every fourth one written by another user."""
    agent = _AGENTS[seed.tenant]
    return {
        "content_id": _id("content", seed.index),
        "media_id": _FILES[(agent, "mine")][0],
        "agent_id": agent,
        "customer_id": _CUSTOMER,
        "user_id": _OTHER_USER if seed.index % 4 == 3 else _USER,
        "content_type": "text",
        "content": f"content {seed.index}",
        "summary": None if seed.null else f"summary {seed.index}",
        "date_created": _T0 + timedelta(minutes=seed.tie),
    }


async def test_search_by_ids_gives_the_plain_joins_rows(conn: asyncpg.Connection) -> None:
    """Flips if the fenced search_by_ids returns a row the plain join did not, or misses one: by a file's id and by
    content ids, a content row of another user on the caller's file returned by neither."""
    agent = _AGENTS[0]
    ids = [_FILES[(agent, "mine")][0], _id("content", 0), _id("content", 3), _id("content", 30)]
    async with _parents(conn):
        await assert_same_results(
            conn,
            table="media_content",
            row=_content,
            old_sql=_IDS_SQL,
            params=[agent, ids, _USER],
            helper=lambda c: _collection(MediaContentCollection, c).search_by_ids(ids, _USER, agent_id=agent),
            order_by=["media_id", "date_created"],
            page_size=_PAGE,
        )


async def test_search_by_ids_leaves_out_the_callers_content_on_another_users_file(conn: asyncpg.Connection) -> None:
    """Flips if search_by_ids returns a content row of the caller's on a file another user owns: the file's user is
    bound beside the content's, as nothing makes them equal."""
    agent = _AGENTS[0]
    theirs = _FILES[(agent, "theirs")][0]
    async with _parents(conn):
        await _mine_on_theirs(conn, agent, theirs)


async def _mine_on_theirs(conn: asyncpg.Connection, agent: uuid.UUID, theirs: uuid.UUID) -> None:
    transaction = conn.transaction()
    await transaction.start()
    try:
        await conn.execute(
            "INSERT INTO media_content (content_id, media_id, agent_id, customer_id, user_id, content_type, content,"
            " date_created) VALUES ($1, $2, $3, $4, $5, 'text', 'mine on theirs', $6)",
            _id("content", 999),
            theirs,
            agent,
            _CUSTOMER,
            _USER,
            _T0,
        )
        contents = _collection(MediaContentCollection, conn)
        assert await contents.search_by_ids([theirs, _id("content", 999)], _USER, agent_id=agent) == []
    finally:
        await transaction.rollback()


#: find_by_conversation_id before the fence: the plain join, the owner bound on the chunk
_CONVERSATION_SQL = """
    SELECT mc.chunk_id, mc.memory_id, mc.chunk_index, mc.content, mc.summary,
           mc.heading_context, mc.page_number,
           mc.message_id_start, mc.message_id_end,
           mc.date_created
    FROM memory_chunks mc
    JOIN memories m
      ON m.agent_id = mc.agent_id
     AND m.memory_id = mc.memory_id
    WHERE mc.agent_id = $1
      AND m.conversation_id = $2
      AND mc.user_id = $3
      AND mc.customer_id = $4
      AND mc.message_id_start IS NOT NULL
    ORDER BY mc.message_id_start ASC, mc.chunk_id ASC
    LIMIT $5
"""
_CHUNK_MEMORIES = ("mine", "theirs", "elsewhere")


def _chunk(seed: Seeded) -> Mapping[str, Any]:
    """a chunk of its tenant's agent, under the caller's memory in the conversation, another user's memory in it, or
    the caller's memory in another conversation in turn; every fifth one written by another user."""
    agent = _AGENTS[seed.tenant]
    return {
        "chunk_id": _id("chunk", seed.index),
        "memory_id": _MEMORIES[(agent, _CHUNK_MEMORIES[seed.index % 3])][0],
        "agent_id": agent,
        "customer_id": _CUSTOMER,
        "user_id": _OTHER_USER if seed.index % 5 == 4 else _USER,
        "chunk_index": seed.index,
        "content": f"chunk {seed.index}",
        "heading_context": None if seed.null else "a heading",
        "page_number": None if seed.null else seed.index,
        "date_created": _T0,
        "message_id_start": None if seed.null else _id("message", seed.tie),
    }


async def test_find_by_conversation_id_gives_the_plain_joins_rows(conn: asyncpg.Connection) -> None:
    """Flips if the fenced find_by_conversation_id returns a chunk the plain join did not, misses one, or orders them
    differently."""
    agent = _AGENTS[0]
    async with _parents(conn):
        await assert_same_results(
            conn,
            table="memory_chunks",
            row=_chunk,
            old_sql=_CONVERSATION_SQL,
            params=[agent, _CONVERSATION, _USER, _CUSTOMER, _PAGE],
            helper=lambda c: _collection(MemoryChunkCollection, c).find_by_conversation_id(
                _CONVERSATION, user_id=_USER, agent_id=agent, customer_id=_CUSTOMER, limit=_PAGE
            ),
            order_by=["message_id_start", "chunk_id"],
            page_size=_PAGE,
        )


#: hybrid_search's history pool before the fence: the owner triple and the date bound side by side
_HISTORY_SQL = """
    SELECT memory_id, content, summary, type_memory, date_created,
           embedding::text AS embedding,
           1 - (embedding OPERATOR(public.<=>) $1::text::public.vector) AS similarity
    FROM memories
    WHERE agent_id = $2 AND customer_id = $3 AND user_id = $4
      AND embedding IS NOT NULL
      AND (superseded_by IS NULL OR NOT EXISTS (
            SELECT 1 FROM memories g
            WHERE g.agent_id = memories.agent_id AND g.memory_id = memories.superseded_by))
      AND NOT (COALESCE(tags, '[]'::jsonb) ? 'retracted')
      AND salience >= $6 AND date_created < $7
    ORDER BY embedding OPERATOR(public.<=>) $1::text::public.vector
    LIMIT $5
"""
#: the embedding width the package registers by default
_DIM = 1024
_OTHER_CUSTOMER = uuid.uuid5(_NS, "other-customer")


class _Asked:
    """a pool that keeps each statement asked of it and answers nothing."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self.asked.append((sql, args))
        return []

    async def fetchrow(self, sql: str, *args: Any) -> None:
        self.asked.append((sql, args))

    async def fetchval(self, sql: str, *args: Any) -> int:
        self.asked.append((sql, args))
        return 0

    async def execute(self, sql: str, *args: Any) -> str:
        self.asked.append((sql, args))
        return ""


def _memory(seed: Seeded) -> Mapping[str, Any]:
    """a memory of its tenant's agent, older than the window on even rows and newer on odd ones; every fifth one of
    another user, every seventh one of another customer, its distance from the query its tie."""
    return {
        "memory_id": _id("memory", seed.index),
        "agent_id": _AGENTS[seed.tenant],
        "customer_id": _OTHER_CUSTOMER if seed.index % 7 == 6 else _CUSTOMER,
        "user_id": _OTHER_USER if seed.index % 5 == 4 else _USER,
        "conversation_id": _CONVERSATION,
        "type_memory": "fact",
        "content": f"memory {seed.index}",
        "date_created": _T0 - timedelta(days=30, minutes=seed.index) if seed.index % 2 == 0 else datetime(2099, 1, 1),  # noqa: DTZ001
        "date_updated": _T0,
        "embedding": None if seed.null else "[" + ",".join(["1"] * (_DIM - 1) + [str(seed.tie)]) + "]",
    }


async def test_history_pool_gives_the_plain_reads_rows(conn: asyncpg.Connection) -> None:
    """Flips if hybrid_search's history pool, the user's older memories read inside a subquery and the customer
    checked after, returns a memory the owner-and-date read did not, misses one, or orders them differently."""
    agent = _AGENTS[0]
    asked = _Asked()
    await _collection(MemoriesCollection, asked).hybrid_search(
        user_id=_USER,
        agent_id=agent,
        customer_id=_CUSTOMER,
        embedding=[0.5] * _DIM,
        user_text="memory",
        top_k=_PAGE,
        candidate_limit=_PAGE,
        similarity_threshold=0.0,
        recency_half_life_hours=24.0,
        signal_weights={"semantic": 1.0, "keyword": 1.0, "recency": 1.0},
        recent_cap=2,
    )
    (history_sql, history_args), *_ = [(sql, args) for sql, args in asked.asked if "date_created <" in sql]
    # the seeds write embeddings as their text form: pgvector's type is an extension type on PostgreSQL and a
    # built-in one on YugabyteDB (which asyncpg then refuses without a codec), so name the codec where it lives
    vector_schema = await conn.fetchval(
        "SELECT n.nspname FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace WHERE t.typname = 'vector'"
    )
    await conn.set_type_codec("vector", schema=vector_schema, encoder=str, decoder=str, format="text")
    await assert_same_results(
        conn,
        table="memories",
        row=_memory,
        old_sql=_HISTORY_SQL,
        params=history_args,
        helper=lambda c: c.fetch(history_sql, *history_args),
        order_by=["similarity"],
        page_size=_PAGE,
    )
