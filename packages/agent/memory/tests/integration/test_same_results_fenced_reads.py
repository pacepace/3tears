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

import uuid
from typing import Any

import asyncpg
import pytest

from threetears.agent.memory.collections import MediaContentCollection, MemoriesCollection, MemoryChunkCollection
from threetears.core.testing.same_results import assert_same_results

from .memory_same_results import (
    text_vectors,
    AGENTS,
    CONVERSATION,
    CUSTOMER,
    DIM,
    FILES,
    PAGE,
    T0,
    USER,
    chunk_row,
    collection,
    conn,
    content_row,
    ident,
    memory_row,
    parents,
)

__all__ = ["conn"]

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]


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


async def test_search_by_ids_gives_the_plain_joins_rows(conn: asyncpg.Connection) -> None:
    """Flips if the fenced search_by_ids returns a row the plain join did not, or misses one: by a file's id and by
    content ids, a content row of another user on the caller's file returned by neither."""
    agent = AGENTS[0]
    ids = [FILES[(agent, "mine")][0], ident("content", 0), ident("content", 3), ident("content", 30)]
    async with parents(conn):
        await assert_same_results(
            conn,
            table="media_content",
            row=content_row,
            old_sql=_IDS_SQL,
            params=[agent, ids, USER],
            helper=lambda c: collection(MediaContentCollection, c).search_by_ids(ids, USER, agent_id=agent),
            order_by=["media_id", "date_created"],
            page_size=PAGE,
        )


async def test_search_by_ids_leaves_out_the_callers_content_on_another_users_file(conn: asyncpg.Connection) -> None:
    """Flips if search_by_ids returns a content row of the caller's on a file another user owns: the file's user is
    bound beside the content's, as nothing makes them equal."""
    agent = AGENTS[0]
    theirs = FILES[(agent, "theirs")][0]
    async with parents(conn):
        await _mine_on_theirs(conn, agent, theirs)


async def _mine_on_theirs(conn: asyncpg.Connection, agent: uuid.UUID, theirs: uuid.UUID) -> None:
    transaction = conn.transaction()
    await transaction.start()
    try:
        await conn.execute(
            "INSERT INTO media_content (content_id, media_id, agent_id, customer_id, user_id, content_type, content,"
            " date_created) VALUES ($1, $2, $3, $4, $5, 'text', 'mine on theirs', $6)",
            ident("content", 999),
            theirs,
            agent,
            CUSTOMER,
            USER,
            T0,
        )
        contents = collection(MediaContentCollection, conn)
        assert await contents.search_by_ids([theirs, ident("content", 999)], USER, agent_id=agent) == []
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


async def test_find_by_conversation_id_gives_the_plain_joins_rows(conn: asyncpg.Connection) -> None:
    """Flips if the fenced find_by_conversation_id returns a chunk the plain join did not, misses one, or orders them
    differently."""
    agent = AGENTS[0]
    async with parents(conn):
        await assert_same_results(
            conn,
            table="memory_chunks",
            row=chunk_row,
            old_sql=_CONVERSATION_SQL,
            params=[agent, CONVERSATION, USER, CUSTOMER, PAGE],
            helper=lambda c: collection(MemoryChunkCollection, c).find_by_conversation_id(
                CONVERSATION, user_id=USER, agent_id=agent, customer_id=CUSTOMER, limit=PAGE
            ),
            order_by=["message_id_start", "chunk_id"],
            page_size=PAGE,
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


async def test_history_pool_gives_the_plain_reads_rows(conn: asyncpg.Connection) -> None:
    """Flips if hybrid_search's history pool, the user's older memories read inside a subquery and the customer
    checked after, returns a memory the owner-and-date read did not, misses one, or orders them differently."""
    agent = AGENTS[0]
    asked = _Asked()
    await collection(MemoriesCollection, asked).hybrid_search(
        user_id=USER,
        agent_id=agent,
        customer_id=CUSTOMER,
        embedding=[0.5] * DIM,
        user_text="memory",
        top_k=PAGE,
        candidate_limit=PAGE,
        similarity_threshold=0.0,
        recency_half_life_hours=24.0,
        signal_weights={"semantic": 1.0, "keyword": 1.0, "recency": 1.0},
        recent_cap=2,
    )
    (history_sql, history_args), *_ = [(sql, args) for sql, args in asked.asked if "date_created <" in sql]
    await text_vectors(conn)
    await assert_same_results(
        conn,
        table="memories",
        row=memory_row,
        old_sql=_HISTORY_SQL,
        params=history_args,
        helper=lambda c: c.fetch(history_sql, *history_args),
        order_by=["similarity"],
        page_size=PAGE,
    )
