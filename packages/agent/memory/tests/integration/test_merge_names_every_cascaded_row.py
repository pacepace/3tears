"""integration: a user merge names every row the alias-collision cascade removed.

The merge deletes a source memory whose alias a master memory already holds, and the delete
cascades to the memory's media, that media's media_content, the memory's chunks, and every
consolidation edge touching it. None of those come back from the DELETE, and
the hub can evict only what it is told about, so a pod that had read one kept serving it by id
after L3 lost it. This drives :func:`repoint_user` against Postgres with the full chain seeded
and checks that every row the cascade removed is named, and that nothing the merge left behind
is.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import asyncpg
import pytest

from threetears.agent.memory.merge import repoint_user
from threetears.agent.memory.migrations import register as register_memory
from threetears.conversations.migrations import register as register_conversations
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration

_EMBEDDING = "[" + ",".join(["0.1"] * 1024) + "]"


async def _memory(conn: asyncpg.Connection, *, agent_id: uuid.UUID, user_id: uuid.UUID, alias: str | None) -> uuid.UUID:
    memory_id = uuid.uuid4()
    now = datetime.now(UTC)
    await conn.execute(
        "INSERT INTO memories ("
        "memory_id, agent_id, customer_id, user_id, conversation_id, type_memory, content, alias, "
        "date_created, date_updated"
        ") VALUES ($1, $2, $3, $4, $5, 'fact', $6, $7, $8, $8)",
        memory_id,
        agent_id,
        uuid.uuid4(),
        user_id,
        uuid.uuid4(),
        f"memory {memory_id}",
        alias,
        now,
    )
    return memory_id


async def _media(
    conn: asyncpg.Connection, *, agent_id: uuid.UUID, user_id: uuid.UUID, memory_id: uuid.UUID
) -> uuid.UUID:
    media_id = uuid.uuid4()
    now = datetime.now(UTC)
    await conn.execute(
        "INSERT INTO media ("
        "media_id, memory_id, agent_id, customer_id, user_id, media_category, metadata_json, "
        "date_created, date_updated"
        ") VALUES ($1, $2, $3, $4, $5, 'document', '{}'::jsonb, $6, $6)",
        media_id,
        memory_id,
        agent_id,
        uuid.uuid4(),
        user_id,
        now,
    )
    return media_id


async def _content(
    conn: asyncpg.Connection, *, agent_id: uuid.UUID, user_id: uuid.UUID, media_id: uuid.UUID
) -> uuid.UUID:
    content_id = uuid.uuid4()
    await conn.execute(
        "INSERT INTO media_content ("
        "content_id, media_id, agent_id, customer_id, user_id, content_type, content, embedding, date_created"
        ") VALUES ($1, $2, $3, $4, $5, 'ocr', 'text', $6::text::public.vector, $7)",
        content_id,
        media_id,
        agent_id,
        uuid.uuid4(),
        user_id,
        _EMBEDDING,
        datetime.now(UTC),
    )
    return content_id


async def _chunk(
    conn: asyncpg.Connection, *, agent_id: uuid.UUID, user_id: uuid.UUID, memory_id: uuid.UUID
) -> uuid.UUID:
    chunk_id = uuid.uuid4()
    await conn.execute(
        "INSERT INTO memory_chunks ("
        "chunk_id, memory_id, agent_id, customer_id, user_id, content, embedding, date_created"
        ") VALUES ($1, $2, $3, $4, $5, 'chunk', $6::text::public.vector, $7)",
        chunk_id,
        memory_id,
        agent_id,
        uuid.uuid4(),
        user_id,
        _EMBEDDING,
        datetime.now(UTC),
    )
    return chunk_id


async def _edge(conn: asyncpg.Connection, *, agent_id: uuid.UUID, gist_id: uuid.UUID, source_id: uuid.UUID) -> None:
    now = datetime.now(UTC)
    await conn.execute(
        "INSERT INTO memory_consolidations ("
        "agent_id, consolidated_memory_id, source_memory_id, rationale, date_created, date_updated"
        ") VALUES ($1, $2, $3, 'merged', $4, $4)",
        agent_id,
        gist_id,
        source_id,
        now,
    )


async def test_every_row_the_collision_cascade_removed_is_named(pg_schema: tuple[str, str]) -> None:
    url, schema = pg_schema
    runner = MigrationRunner()
    register_conversations(runner)
    register_memory(runner)
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        await runner.apply_for_agent_schema(AsyncpgStore(conn))  # type: ignore[arg-type]

        agent = uuid.uuid4()
        source, master = uuid.uuid4(), uuid.uuid4()
        await _memory(conn, agent_id=agent, user_id=master, alias="home")
        colliding = await _memory(conn, agent_id=agent, user_id=source, alias="home")
        survivor = await _memory(conn, agent_id=agent, user_id=source, alias="work")
        gist = await _memory(conn, agent_id=agent, user_id=source, alias=None)

        media = await _media(conn, agent_id=agent, user_id=source, memory_id=colliding)
        content = await _content(conn, agent_id=agent, user_id=source, media_id=media)
        first_chunk = await _chunk(conn, agent_id=agent, user_id=source, memory_id=colliding)
        second_chunk = await _chunk(conn, agent_id=agent, user_id=source, memory_id=colliding)
        await _edge(conn, agent_id=agent, gist_id=gist, source_id=colliding)
        # rows the merge moves but does not delete: named as repointed, never as cascaded.
        kept_media = await _media(conn, agent_id=agent, user_id=source, memory_id=survivor)
        await _edge(conn, agent_id=agent, gist_id=gist, source_id=survivor)

        async with conn.transaction():
            result = await repoint_user(conn, from_user_id=source, to_user_id=master)

        assert result.alias_collisions_deleted == [(agent, colliding)]
        assert result.alias_collision_media == [(agent, media)]
        assert result.alias_collision_media_content == [(agent, content)]
        assert sorted(result.alias_collision_memory_chunks) == sorted([(agent, first_chunk), (agent, second_chunk)])
        assert result.alias_collision_memory_consolidations == [(agent, gist, colliding)]
        assert (agent, kept_media) in result.media

        # and the cascade removed exactly what was named.
        assert await conn.fetchval("SELECT count(*) FROM media WHERE media_id = $1", media) == 0
        assert await conn.fetchval("SELECT count(*) FROM media_content WHERE content_id = $1", content) == 0
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM memory_chunks WHERE chunk_id = ANY($1::uuid[])", [first_chunk, second_chunk]
            )
            == 0
        )
        assert (
            await conn.fetchval("SELECT count(*) FROM memory_consolidations WHERE source_memory_id = $1", colliding)
            == 0
        )
        assert (
            await conn.fetchval("SELECT count(*) FROM memory_consolidations WHERE source_memory_id = $1", survivor) == 1
        )
    finally:
        await conn.close()


async def test_a_merge_with_no_collision_deletes_and_names_nothing(pg_schema: tuple[str, str]) -> None:
    url, schema = pg_schema
    runner = MigrationRunner()
    register_conversations(runner)
    register_memory(runner)
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        await runner.apply_for_agent_schema(AsyncpgStore(conn))  # type: ignore[arg-type]

        agent = uuid.uuid4()
        source, master = uuid.uuid4(), uuid.uuid4()
        moved = await _memory(conn, agent_id=agent, user_id=source, alias="home")

        async with conn.transaction():
            result = await repoint_user(conn, from_user_id=source, to_user_id=master)

        assert result.alias_collisions_deleted == []
        assert result.alias_collision_media == []
        assert result.alias_collision_memory_chunks == []
        assert result.alias_collision_memory_consolidations == []
        assert result.memories == [(agent, moved)]
    finally:
        await conn.close()
