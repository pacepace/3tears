"""finding and deleting memories by tag, within one owner's rows.

a host that files derived memories under a tag (scriob: a story's facts under ``scene:<id>``) must drop
them when their source goes: find them by the tag, or delete them by it. three rules, each a test here:

- **scoped to the owner.** the same tag on another user's, customer's or agent's rows is not theirs to
  find or delete.
- **through the collection.** a deleted memory is gone from every tier, so reading it by id afterwards
  answers nothing (a raw ``DELETE`` would leave it in the row cache).
- **empty is not absent.** an empty tag tuple names nothing, so it finds and deletes nothing; it never
  means "every row", which here would be a whole owner's corpus gone.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import asyncpg
import pytest
from sqlalchemy import MetaData

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import MemoriesCollection, memories_table
from threetears.agent.memory.migrations import register as register_memory
from threetears.conversations.migrations import register as register_conversations
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore
from .memory_support import vec_sql

pytestmark = pytest.mark.integration


@pytest.fixture
async def applied_schema(pg_schema: tuple[str, str]) -> tuple[str, str]:
    """apply conversations + memory migrations into the per-test schema."""
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
    return url, schema


async def _make_pool(url: str, schema: str) -> asyncpg.Pool:
    pool: asyncpg.Pool = await asyncpg.create_pool(
        dsn=url, min_size=1, max_size=4, server_settings={"search_path": f"{schema}, public"}, init=init_connection
    )
    return pool


def _build_collection(pool: asyncpg.Pool) -> MemoriesCollection:
    """the collection over a real L1 row cache (the package's own table factory), so a test can see a
    deleted memory left answering from a cache tier."""
    metadata = MetaData()
    memories_table(metadata)
    l1 = SQLiteBackend(db_name=f"mem_by_tag_l1_{uuid.uuid4().hex[:8]}")
    l1.initialize(metadata)
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l3_pool=pool)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return MemoriesCollection(registry=registry, config=config, authorizer=MagicMock(spec=MemoryAuthorizerDependencies))


async def _insert(
    pool: asyncpg.Pool,
    *,
    agent_id: uuid.UUID,
    customer_id: uuid.UUID,
    user_id: uuid.UUID,
    content: str,
    tags: list[str],
) -> uuid.UUID:
    memory_id = uuid.uuid4()
    await pool.execute(
        "INSERT INTO memories ("
        "memory_id, agent_id, customer_id, user_id, conversation_id, "
        "type_memory, content, tags, embedding, date_created, date_updated"
        ") VALUES ($1,$2,$3,$4,$5,'fact',$6,$7::jsonb,$8::text::public.vector,now(),now())",
        memory_id,
        agent_id,
        customer_id,
        user_id,
        uuid.uuid4(),
        content,
        tags,
        vec_sql(1.0),
    )
    return memory_id


class TestMemoriesByTag:
    async def test_finds_and_deletes_only_this_owners_tagged_rows(self, applied_schema: tuple[str, str]) -> None:
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            memories = _build_collection(pool)
            agent, customer, user = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            owner = {"agent_id": agent, "customer_id": customer, "user_id": user}
            first = await _insert(pool, **owner, content="a", tags=["scene:1", "subj:mara"])
            second = await _insert(pool, **owner, content="b", tags=["scene:2"])
            untouched = await _insert(pool, **owner, content="c", tags=["scene:3"])
            # the same tags, owned by someone else on each axis: never found, never deleted
            others = [
                await _insert(pool, **{**owner, "user_id": uuid.uuid4()}, content="x", tags=["scene:1"]),
                await _insert(pool, **{**owner, "customer_id": uuid.uuid4()}, content="y", tags=["scene:1"]),
                await _insert(pool, **{**owner, "agent_id": uuid.uuid4()}, content="z", tags=["scene:2"]),
            ]
            # read once through the collection, so the row cache holds them before the delete
            assert await memories.get((agent, first)) is not None

            found = await memories.find_ids_by_tags(**owner, tags_any=("scene:1", "scene:2"))
            assert set(found) == {first, second}

            deleted = await memories.delete_by_tags(**owner, tags_any=("scene:1", "scene:2"))

            assert deleted == 2
            assert await memories.get((agent, first)) is None, "a deleted memory still answers from a cache tier"
            assert await memories.get((agent, second)) is None
            assert await memories.get((agent, untouched)) is not None
            remaining = {row["memory_id"] for row in await pool.fetch("SELECT memory_id FROM memories")}
            assert remaining == {untouched, *others}
        finally:
            await pool.close()

    async def test_an_empty_tag_tuple_finds_and_deletes_nothing(self, applied_schema: tuple[str, str]) -> None:
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            memories = _build_collection(pool)
            owner = {"agent_id": uuid.uuid4(), "customer_id": uuid.uuid4(), "user_id": uuid.uuid4()}
            kept = await _insert(pool, **owner, content="a", tags=["scene:1"])

            assert await memories.find_ids_by_tags(**owner, tags_any=()) == []
            assert await memories.delete_by_tags(**owner, tags_any=()) == 0
            assert await memories.get((owner["agent_id"], kept)) is not None
        finally:
            await pool.close()

    async def test_a_callers_delete_is_authorized_as_a_memory_write_first(
        self, applied_schema: tuple[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given a caller, the bulk delete evaluates ``memory.write`` before anything is found; a refusal
        deletes nothing. Without a caller it is the agent-internal path, as ``find_by_scope``'s is."""
        import threetears.agent.memory.collections as collections_module
        from threetears.agent.memory.authorize import MemoryAccessDenied

        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            memories = _build_collection(pool)
            owner = {"agent_id": uuid.uuid4(), "customer_id": uuid.uuid4(), "user_id": uuid.uuid4()}
            kept = await _insert(pool, **owner, content="a", tags=["scene:1"])
            asked: list[dict[str, object]] = []

            async def deny(**kwargs: object) -> None:
                asked.append(kwargs)
                raise MemoryAccessDenied("no")

            monkeypatch.setattr(collections_module, "authorize_memory_access", deny)
            caller = uuid.uuid4()

            with pytest.raises(MemoryAccessDenied):
                await memories.delete_by_tags(**owner, tags_any=("scene:1",), caller_user_id=caller)

            assert asked[0]["action"] == "memory.write" and asked[0]["caller_user_id"] == caller
            assert await memories.get((owner["agent_id"], kept)) is not None, "a refused delete deletes nothing"
        finally:
            await pool.close()
