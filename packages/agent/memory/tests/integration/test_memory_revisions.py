"""integration: nothing an agent remembers is destroyed, and a permanent memory is never touched.

Against real pgvector tables: dream's people guard and its permanence
judgment, extraction's UPDATE as a linked revision and its DELETE as a
retraction, and every move refusing a permanent (``evergreen``) memory.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any
from unittest.mock import MagicMock

import asyncpg
import pytest

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import MemoriesCollection, MemoryConsolidationsCollection
from threetears.agent.memory.extraction import MemoryExtractor
from threetears.agent.memory.revisions import RETRACTED_TAG, retract, supersede
from threetears.agent.memory.tools import load_memory_add_tool, load_memory_keep_tool
from threetears.agent.memory.types import MemoryConfig
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.agent.memory.migrations import register as register_memory
from threetears.conversations.migrations import register as register_conversations
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore
from .memory_support import PurposeChatModelFactory, StubEmbeddings, insert_source, make_pool, make_service, vec

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


def _collections(pool: asyncpg.Pool) -> tuple[MemoriesCollection, MemoryConsolidationsCollection]:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    memories = MemoriesCollection(
        registry=registry, config=config, authorizer=MagicMock(spec=MemoryAuthorizerDependencies)
    )
    return memories, MemoryConsolidationsCollection(registry=registry, config=config, nats_client=None)


async def _row(conn: asyncpg.Connection, memory_id: uuid.UUID) -> dict[str, Any]:
    row = await conn.fetchrow(
        "SELECT content, salience, evergreen, superseded_by, tags FROM memories WHERE memory_id = $1", memory_id
    )
    assert row is not None, "a memory was deleted"
    return dict(row)


async def _two_near_duplicates(conn: asyncpg.Connection) -> tuple[uuid.UUID, uuid.UUID, list[uuid.UUID]]:
    agent_id, user_id = uuid.uuid4(), uuid.uuid4()
    ids = []
    for i in range(2):
        mid, _ = await insert_source(
            conn, agent_id=agent_id, customer_id=user_id, user_id=user_id, seed=0.05, content=f"likes tea {i}"
        )
        ids.append(mid)
    return agent_id, user_id, ids


class TestDreamJudges:
    async def test_memories_about_different_subjects_are_not_merged(self, applied_schema: tuple[str, str]) -> None:
        url, schema = applied_schema
        pool, conn = await make_pool(url, schema), await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            agent_id, user_id, ids = await _two_near_duplicates(conn)
            service = make_service(
                pool,
                gist_vector=vec(0.2),
                reflector_content='{"gist": "they like tea", "rationale": "x", "one_subject": false}',
            )

            result = await service.run_consolidation(agent_id, customer_id=user_id, user_id=user_id)

            assert result.gists_created == 0
            for i in ids:
                assert (await _row(conn, i))["superseded_by"] is None
        finally:
            await conn.close()
            await pool.close()

    async def test_a_gist_dream_judges_lasting_is_permanent(self, applied_schema: tuple[str, str]) -> None:
        url, schema = applied_schema
        pool, conn = await make_pool(url, schema), await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            agent_id, user_id, _ids = await _two_near_duplicates(conn)
            service = make_service(
                pool,
                gist_vector=vec(0.2),
                reflector_content='{"gist": "likes tea", "rationale": "x", "one_subject": true, "permanent": true}',
            )

            result = await service.run_consolidation(agent_id, customer_id=user_id, user_id=user_id)

            [gist] = result.gist_ids
            assert (await _row(conn, uuid.UUID(str(gist))))["evergreen"] is True
        finally:
            await conn.close()
            await pool.close()


class TestThePermanentAreUntouched:
    async def test_supersede_and_retract_refuse_a_permanent_memory(self, applied_schema: tuple[str, str]) -> None:
        url, schema = applied_schema
        pool, conn = await make_pool(url, schema), await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            agent_id, user_id, (kept, other) = await _two_near_duplicates(conn)
            await conn.execute("UPDATE memories SET evergreen = true WHERE memory_id = $1", kept)
            memories, edges = _collections(pool)

            with pytest.raises(ValueError):
                await supersede(
                    memories,
                    edges,
                    agent_id=agent_id,
                    source_ids=[kept],
                    fields={
                        "customer_id": user_id,
                        "user_id": user_id,
                        "conversation_id": uuid.uuid4(),
                        "type_memory": "fact",
                        "content": "new",
                        "embedding": vec(0.3),
                        "salience": 0.5,
                    },
                    rationale="x",
                )
            assert not await retract(memories, agent_id=agent_id, memory_id=kept, reason="x")
            row = await _row(conn, kept)
            assert (row["content"], row["superseded_by"], float(row["salience"])) == ("likes tea 0", None, 0.5)

            # an ordinary memory is retracted, not deleted
            assert await retract(memories, agent_id=agent_id, memory_id=other, reason="no longer true")
            gone = await _row(conn, other)
            assert float(gone["salience"]) == 0.0
            assert RETRACTED_TAG in gone["tags"] and "retracted_because:no longer true" in gone["tags"]
        finally:
            await conn.close()
            await pool.close()


class TestARetractedMemoryIsLeftAlone:
    async def test_dream_and_dedup_skip_it(self, applied_schema: tuple[str, str]) -> None:
        url, schema = applied_schema
        pool, conn = await make_pool(url, schema), await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            agent_id, user_id, (gone, kept) = await _two_near_duplicates(conn)
            memories, _edges = _collections(pool)
            assert await retract(memories, agent_id=agent_id, memory_id=gone, reason="wrong")

            similar = await memories.find_similar_for_dedup(
                user_id=user_id, agent_id=agent_id, embedding=vec(0.05), top_k=5, threshold=0.0
            )
            result = await make_service(pool, gist_vector=vec(0.2)).run_consolidation(
                agent_id, customer_id=user_id, user_id=user_id
            )

            for floor in (0.0, 0.2):  # explicit search reads every salience; ambient recall cuts low ones
                found = await memories.hybrid_search(
                    user_id=user_id,
                    agent_id=agent_id,
                    customer_id=user_id,
                    embedding=vec(0.05),
                    user_text="likes tea",
                    top_k=5,
                    candidate_limit=20,
                    similarity_threshold=0.0,
                    recency_half_life_hours=24.0,
                    signal_weights={"semantic": 0.55, "keyword": 0.15, "recency": 0.30},
                    salience_ambient_floor=floor,
                )
                assert gone not in {r["memory_id"] for r in found}, floor
                assert kept in {r["memory_id"] for r in found}, floor
            assert [r["memory_id"] for r in similar] == [kept]
            assert result.gists_created == 0  # one live memory is no cluster
        finally:
            await conn.close()
            await pool.close()


class TestExtractionRevises:
    async def _act(
        self,
        pool: asyncpg.Pool,
        authorizer: MemoryAuthorizerDependencies,
        action: dict[str, Any],
        *,
        agent_id: uuid.UUID,
        user_id: uuid.UUID,
        with_edges: bool = True,
    ) -> Any:
        """Run one extraction whose resolution step answers ``action`` for its one candidate."""
        memories, edges = _collections(pool)
        extractor = MemoryExtractor(
            config=MemoryConfig(),
            embedding_provider=StubEmbeddings(vec(0.4)),
            chat_model_factory=PurposeChatModelFactory(
                worthiness=json.dumps({"worthy": True, "reason": "has facts"}),
                extraction=json.dumps([{"type": "fact", "content": action.get("content", "about tea")}]),
                resolution=json.dumps([{"index": 0, **action}]),
            ),
            authorizer=authorizer,
            memories_collection=memories,
            consolidations_collection=edges if with_edges else None,
        )
        return await extractor.extract(
            user_id=user_id,
            conversation_id=uuid.uuid4(),
            message_id_source=uuid.uuid4(),
            user_message="x" * 50,
            assistant_response="y" * 200,
            turn_count=10,
            agent_id=agent_id,
            customer_id=user_id,
        )

    async def test_an_update_is_a_new_memory_and_the_old_is_superseded_not_overwritten(
        self, applied_schema: tuple[str, str], permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        url, schema = applied_schema
        pool, conn = await make_pool(url, schema), await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            agent_id, user_id, (old, _) = await _two_near_duplicates(conn)

            await self._act(
                pool,
                permissive_memory_authorizer,
                {"action": "UPDATE", "memory_id": str(old), "content": "likes green tea"},
                agent_id=agent_id,
                user_id=user_id,
            )

            row = await _row(conn, old)
            assert row["content"] == "likes tea 0" and row["superseded_by"] is not None
            assert (await _row(conn, row["superseded_by"]))["content"] == "likes green tea"
            assert (
                await conn.fetchval("SELECT rationale FROM memory_consolidations WHERE source_memory_id = $1", old)
                == "revised by a later conversation"
            )
        finally:
            await conn.close()
            await pool.close()

    async def test_an_update_to_a_permanent_memory_is_written_beside_it(
        self, applied_schema: tuple[str, str], permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        url, schema = applied_schema
        pool, conn = await make_pool(url, schema), await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            agent_id, user_id, (kept, _) = await _two_near_duplicates(conn)
            await conn.execute("UPDATE memories SET evergreen = true WHERE memory_id = $1", kept)

            await self._act(
                pool,
                permissive_memory_authorizer,
                {"action": "UPDATE", "memory_id": str(kept), "content": "likes green tea"},
                agent_id=agent_id,
                user_id=user_id,
            )

            row = await _row(conn, kept)
            assert (row["content"], row["superseded_by"]) == ("likes tea 0", None)
            assert await conn.fetchval("SELECT count(*) FROM memories WHERE content = 'likes green tea'") == 1
        finally:
            await conn.close()
            await pool.close()

    async def test_without_edges_an_update_changes_nothing(
        self, applied_schema: tuple[str, str], permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        url, schema = applied_schema
        pool, conn = await make_pool(url, schema), await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            agent_id, user_id, (old, _) = await _two_near_duplicates(conn)

            await self._act(
                pool,
                permissive_memory_authorizer,
                {"action": "UPDATE", "memory_id": str(old), "content": "likes green tea"},
                agent_id=agent_id,
                user_id=user_id,
                with_edges=False,
            )

            assert await conn.fetchval("SELECT count(*) FROM memories WHERE content = 'likes green tea'") == 0
            assert (await _row(conn, old))["content"] == "likes tea 0"
        finally:
            await conn.close()
            await pool.close()

    async def test_a_delete_retracts_and_keeps_the_row(
        self, applied_schema: tuple[str, str], permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        url, schema = applied_schema
        pool, conn = await make_pool(url, schema), await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            agent_id, user_id, (old, kept) = await _two_near_duplicates(conn)
            await conn.execute("UPDATE memories SET evergreen = true WHERE memory_id = $1", kept)
            for target in (old, kept):
                await self._act(
                    pool,
                    permissive_memory_authorizer,
                    {"action": "DELETE", "memory_id": str(target)},
                    agent_id=agent_id,
                    user_id=user_id,
                )

            assert await conn.fetchval("SELECT count(*) FROM memories WHERE agent_id = $1", agent_id) == 2
            assert float((await _row(conn, old))["salience"]) == 0.0
            assert float((await _row(conn, kept))["salience"]) == 0.5
        finally:
            await conn.close()
            await pool.close()


class _Ctx:
    def __init__(self) -> None:
        self.conversation_id = uuid.uuid4()
        self.correlation_id = uuid.uuid4()


class TestTheAgentDecides:
    async def _tools(
        self, pool: asyncpg.Pool, authorizer: MemoryAuthorizerDependencies, *, agent_id: uuid.UUID, user_id: uuid.UUID
    ) -> tuple[Any, Any]:
        memories, edges = _collections(pool)
        [add] = await load_memory_add_tool(
            user_id,
            StubEmbeddings(vec(0.4)),
            agent_id,
            user_id,
            authorizer,
            memories,
            context_resolver=_Ctx,
            consolidations_collection=edges,
        )
        [keep] = load_memory_keep_tool(user_id, agent_id, user_id, authorizer, memories)
        return add, keep

    async def test_a_near_duplicate_replaces_the_old_which_stays(
        self, applied_schema: tuple[str, str], permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        url, schema = applied_schema
        pool, conn = await make_pool(url, schema), await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            agent_id, user_id, (old, _) = await _two_near_duplicates(conn)
            await conn.execute("UPDATE memories SET superseded_by = NULL WHERE agent_id = $1", agent_id)
            await conn.execute("DELETE FROM memories WHERE agent_id = $1 AND memory_id <> $2", agent_id, old)
            add, _keep = await self._tools(pool, permissive_memory_authorizer, agent_id=agent_id, user_id=user_id)

            said = await add.ainvoke({"content": "likes green tea", "permanent": True})

            row = await _row(conn, old)
            assert row["content"] == "likes tea 0" and row["superseded_by"] is not None
            new = await _row(conn, row["superseded_by"])
            assert (new["content"], new["evergreen"]) == ("likes green tea", True)
            assert f"replaces [memory:{old}]" in said and "Kept permanently" in said
        finally:
            await conn.close()
            await pool.close()

    async def test_a_kept_memory_is_never_replaced_and_keep_pins(
        self,
        applied_schema: tuple[str, str],
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        url, schema = applied_schema
        pool, conn = await make_pool(url, schema), await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            agent_id, user_id, (old, _) = await _two_near_duplicates(conn)
            await conn.execute("DELETE FROM memories WHERE agent_id = $1 AND memory_id <> $2", agent_id, old)
            add, keep = await self._tools(pool, permissive_memory_authorizer, agent_id=agent_id, user_id=user_id)

            pinned = await keep.ainvoke({"memory_id": f"[memory:{old}]"})
            with caplog.at_level(logging.WARNING, logger="threetears.agent.memory.tools"):
                said = await add.ainvoke({"content": "likes green tea"})

            # the tool knows a kept memory is not replaced; it is not an error it recovered from
            assert not [r for r in caplog.records if r.levelno >= logging.WARNING], caplog.text
            row = await _row(conn, old)
            assert (row["content"], row["evergreen"], row["superseded_by"]) == ("likes tea 0", True, None)
            assert "kept permanently" in pinned and "replaces" not in said
            assert await conn.fetchval("SELECT count(*) FROM memories WHERE agent_id = $1", agent_id) == 2
        finally:
            await conn.close()
            await pool.close()
