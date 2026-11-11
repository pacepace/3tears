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
from typing import Any

import asyncpg
import pytest

from threetears.agent.memory.merge import repoint_user
from threetears.agent.memory.migrations import register as register_memory
from threetears.conversations import repoint_user as repoint_conversations_user
from threetears.core.collections import repoint_user_rows
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
            result = await repoint_user(conn, from_user_id=source, to_user_id=master, agent_id=agent)

        assert result.removed["memories"] == [(agent, colliding)]
        assert result.removed["media"] == [(agent, media)]
        assert result.removed["media_content"] == [(agent, content)]
        assert sorted(result.removed["memory_chunks"]) == sorted([(agent, first_chunk), (agent, second_chunk)])
        assert result.removed["memory_consolidations"] == [(agent, gist, colliding)]
        assert (agent, kept_media) in result.repointed["media"]
        # the one map the caller evicts from names both.
        assert {(agent, media), (agent, kept_media)} <= set(result.evict["media"])

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
            result = await repoint_user(conn, from_user_id=source, to_user_id=master, agent_id=agent)

        assert all(keys == [] for keys in result.removed.values())
        assert result.repointed["memories"] == [(agent, moved)]
    finally:
        await conn.close()


#: each table a merge touches, by its key, with the owner the merge rewrites
_STATE_SQL = (
    "SELECT agent_id, memory_id, user_id FROM memories ORDER BY 1, 2",
    "SELECT agent_id, media_id, user_id FROM media ORDER BY 1, 2",
    "SELECT agent_id, content_id, user_id FROM media_content ORDER BY 1, 2",
    "SELECT agent_id, chunk_id, user_id FROM memory_chunks ORDER BY 1, 2",
    "SELECT agent_id, consolidated_memory_id, source_memory_id FROM memory_consolidations ORDER BY 1, 2, 3",
    "SELECT agent_id, conversation_id, user_id FROM conversations ORDER BY 1, 2",
)


#: the merge's statements before it named the schema's agent: the collision delete and each repoint by user alone
_BY_USER_COLLISIONS = (
    "DELETE FROM memories m WHERE m.user_id = $1 AND m.alias IS NOT NULL AND EXISTS ("
    "SELECT 1 FROM memories m2 WHERE m2.user_id = $2 AND m2.agent_id = m.agent_id AND m2.alias = m.alias"
    ") RETURNING agent_id, memory_id"
)
_BY_USER_REPOINTS = (
    ("conversations", "conversation_id", "date_updated"),
    ("memories", "memory_id", "date_updated"),
    ("media", "media_id", None),
    ("media_content", "content_id", None),
    ("memory_chunks", "chunk_id", None),
)


async def _rolled_back(conn: asyncpg.Connection, merge: Any) -> tuple[object, ...]:
    """``merge`` in a transaction rolled back after it: the keys it moved per table (sorted), the memories it
    deleted, and every table's rows after.

    :param conn: the schema's connection
    :ptype conn: asyncpg.Connection
    :param merge: the merge, answering ``(moved by table, deleted memories)``
    :ptype merge: Any
    :return: what it moved, what it deleted, what it left
    :rtype: tuple[object, ...]
    """
    transaction = conn.transaction()
    await transaction.start()
    try:
        moved, deleted = await merge()
        after = [[tuple(row) for row in await conn.fetch(sql)] for sql in _STATE_SQL]
    finally:
        await transaction.rollback()
    return {table: sorted(keys) for table, keys in moved.items()}, sorted(deleted), after


async def test_a_merge_naming_its_agent_moves_the_same_rows(pg_schema: tuple[str, str]) -> None:
    """Flips if the merge pinning the schema's agent (each statement by agent and user) moves, deletes or leaves other
    rows than the statements by user alone it replaced: the schema holds one agent's rows, so the two are the same
    merge."""
    url, schema = pg_schema
    runner = MigrationRunner()
    register_conversations(runner)
    register_memory(runner)
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        await runner.apply_for_agent_schema(AsyncpgStore(conn))  # type: ignore[arg-type]

        agent = uuid.uuid4()
        source, master, bystander = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        await _memory(conn, agent_id=agent, user_id=master, alias="home")
        colliding = await _memory(conn, agent_id=agent, user_id=source, alias="home")
        survivor = await _memory(conn, agent_id=agent, user_id=source, alias="work")
        gist = await _memory(conn, agent_id=agent, user_id=source, alias=None)
        other = await _memory(conn, agent_id=agent, user_id=bystander, alias="home")
        for memory, owner in ((colliding, source), (survivor, source), (other, bystander)):
            media = await _media(conn, agent_id=agent, user_id=owner, memory_id=memory)
            await _content(conn, agent_id=agent, user_id=owner, media_id=media)
            await _chunk(conn, agent_id=agent, user_id=owner, memory_id=memory)
        await _edge(conn, agent_id=agent, gist_id=gist, source_id=colliding)
        await _edge(conn, agent_id=agent, gist_id=gist, source_id=survivor)
        for owner in (source, source, master, bystander):
            await conn.execute(
                "INSERT INTO conversations (agent_id, conversation_id, customer_id, user_id, channel_type, status,"
                " date_created, date_updated, message_count) VALUES ($1, $2, $3, $4, 'web', 'active', now(), now(), 0)",
                agent,
                uuid.uuid4(),
                uuid.uuid4(),
                owner,
            )

        async def by_user() -> tuple[dict[str, list[tuple[Any, ...]]], list[tuple[Any, ...]]]:
            deleted = [tuple(row) for row in await conn.fetch(_BY_USER_COLLISIONS, source, master)]
            moved = {
                table: await repoint_user_rows(
                    conn,
                    table=table,
                    user_column="user_id",
                    pk_columns=["agent_id", key],
                    from_user_id=source,
                    to_user_id=master,
                    touch_column=touch,
                )
                for table, key, touch in _BY_USER_REPOINTS
            }
            return moved, deleted

        async def by_agent() -> tuple[dict[str, list[tuple[Any, ...]]], list[tuple[Any, ...]]]:
            conversations = await repoint_conversations_user(
                conn, from_user_id=source, to_user_id=master, agent_id=agent
            )
            result = await repoint_user(conn, from_user_id=source, to_user_id=master, agent_id=agent)
            return {"conversations": conversations, **result.repointed}, result.removed["memories"]

        old = await _rolled_back(conn, by_user)
        new = await _rolled_back(conn, by_agent)

        assert new == old
        moved, deleted, _ = new
        # the case is not empty: the merge moved rows in every table and deleted the colliding memory
        assert len(moved["conversations"]) == 2
        assert deleted == [(agent, colliding)]
        assert all(moved[table] for table in ("memories", "media", "media_content", "memory_chunks"))
    finally:
        await conn.close()
