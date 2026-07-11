"""integration: Dream consolidation end-to-end (A5, v0.15.0).

Exercises :meth:`DreamService.run_consolidation` against a real
pgvector/pg16 database with a faked embedder + reflector:

1. a cluster of near-duplicate memories is merged into one gist -- the gist
   commits (embedded, salience seeded, scope inherited, conversation_id
   inherited from the newest source), the provenance edges are recorded,
   and every source is superseded by the gist while staying directly
   recallable (non-destructive).
2. scope isolation -- a user-scoped run only touches that user's rows; a
   second user's near-duplicates are untouched.
3. self-healing supersession -- a source pointed at a gist that no longer
   exists becomes an eligible candidate again, so a fresh gist regenerates.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import asyncpg
import pytest

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import (
    MemoriesCollection,
    MemoryConsolidationsCollection,
)
from threetears.agent.memory.dream import DreamService
from threetears.agent.memory.migrations import register as register_memory
from threetears.agent.memory.types import MemoryConfig
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration

_DIM = 1024


def _vec(seed: float) -> list[float]:
    """a constant 1024-dim vector; equal seeds -> cosine 1.0 (cluster)."""
    return [seed] * _DIM


def _vec_sql(seed: float) -> str:
    return "[" + ",".join([str(seed)] * _DIM) + "]"


class StubEmbeddings:
    """embedder returning a fixed gist vector (distinct from source vectors)."""

    def __init__(self, gist_vector: list[float]) -> None:
        self._gist = gist_vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._gist for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        _ = text
        return self._gist

    async def aembed_query(self, text: str) -> list[float]:
        _ = text
        return self._gist

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._gist for _ in texts]


class StubChatModel:
    def __init__(self, content: str) -> None:
        self._content = content

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> Any:
        _ = kwargs
        resp = MagicMock()
        resp.content = self._content
        return resp


class StubReflectorFactory:
    def __init__(self, content: str) -> None:
        self._content = content

    async def create_chat_model(self, purpose: str = "consolidation") -> Any:
        _ = purpose
        return StubChatModel(self._content)


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
        store = AsyncpgStore(conn)
        await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
    finally:
        await conn.close()
    return url, schema


async def _make_pool(url: str, schema: str) -> asyncpg.Pool:
    pool: asyncpg.Pool = await asyncpg.create_pool(
        dsn=url,
        min_size=1,
        max_size=4,
        server_settings={"search_path": f"{schema}, public"},
        init=init_connection,
    )
    return pool


def _make_service(
    pool: asyncpg.Pool,
    *,
    gist_vector: list[float],
    reflector_content: str = '{"gist": "merged gist", "rationale": "near-duplicates"}',
) -> DreamService:
    """wire a DreamService over real collections + stubbed embed/reflect."""
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    authorizer = MagicMock(spec=MemoryAuthorizerDependencies)
    memories = MemoriesCollection(registry=registry, config=config, authorizer=authorizer)
    edges = MemoryConsolidationsCollection(registry=registry, config=config, nats_client=None)
    return DreamService(
        config=MemoryConfig(),
        embedding_provider=StubEmbeddings(gist_vector),
        chat_model_factory=StubReflectorFactory(reflector_content),
        memories_collection=memories,
        consolidations_collection=edges,
    )


async def _insert_source(
    conn: asyncpg.Connection,
    *,
    agent_id: uuid.UUID,
    customer_id: uuid.UUID | None,
    user_id: uuid.UUID | None,
    seed: float,
    content: str,
    conversation_id: uuid.UUID | None = None,
    age_days: int = 0,
    superseded_by: uuid.UUID | None = None,
    type_memory: str = "fact",
) -> tuple[uuid.UUID, uuid.UUID]:
    """insert one embedded source memory; return (memory_id, conversation_id)."""
    memory_id = uuid.uuid4()
    conv_id = conversation_id or uuid.uuid4()
    created = datetime.now(UTC) - timedelta(days=age_days)
    await conn.execute(
        "INSERT INTO memories ("
        "memory_id, agent_id, customer_id, user_id, conversation_id, "
        "type_memory, content, embedding, salience, superseded_by, "
        "date_created, date_updated"
        f") VALUES ($1,$2,$3,$4,$5,$6,$7,$8::text::public.vector,0.5,$9,$10,$10)",
        memory_id,
        agent_id,
        customer_id,
        user_id,
        conv_id,
        type_memory,
        content,
        _vec_sql(seed),
        superseded_by,
        created,
    )
    return memory_id, conv_id


async def _superseded_by(conn: asyncpg.Connection, memory_id: uuid.UUID) -> str | None:
    # normalise to str at the border: asyncpg yields stdlib uuid.UUID while a
    # gist id is a uuid_utils.UUID (uuid7); the two compare unequal despite an
    # identical value (the project's recurring UUID-border gotcha).
    val = await conn.fetchval("SELECT superseded_by FROM memories WHERE memory_id = $1", memory_id)
    return str(val) if val is not None else None


class TestDreamEndToEnd:
    async def test_cluster_merges_into_gist_non_destructively(self, applied_schema: tuple[str, str]) -> None:
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            agent_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            user_id = uuid.uuid4()

            # three near-duplicates (identical embedding -> cosine 1.0) with
            # distinct ages so the newest is deterministic, plus one
            # orthogonal singleton that must NOT be merged.
            s_ids = []
            convs = []
            for i, age in enumerate((5, 3, 1)):
                mid, conv = await _insert_source(
                    conn,
                    agent_id=agent_id,
                    customer_id=customer_id,
                    user_id=user_id,
                    seed=0.05,
                    content=f"prefers postgres reason {i}",
                    age_days=age,
                )
                s_ids.append(mid)
                convs.append((age, conv))
            singleton, _ = await _insert_source(
                conn,
                agent_id=agent_id,
                customer_id=customer_id,
                user_id=user_id,
                seed=-0.05,
                content="lives in seattle",
            )
            newest_conv = min(convs, key=lambda t: t[0])[1]  # smallest age = newest

            service = _make_service(pool, gist_vector=_vec(0.9))
            result = await service.run_consolidation(agent_id, customer_id=customer_id, user_id=user_id)

            assert result.gists_created == 1
            assert result.clusters_formed == 1
            assert result.sources_superseded == 3
            gist_id = result.gist_ids[0]

            # the gist row: embedded, salience seeded, scope inherited,
            # conversation_id inherited from the newest source.
            gist = await conn.fetchrow(
                "SELECT customer_id, user_id, conversation_id, content, salience, "
                "evergreen, superseded_by, embedding IS NOT NULL AS has_embed "
                "FROM memories WHERE agent_id = $1 AND memory_id = $2",
                agent_id,
                gist_id,
            )
            assert gist["customer_id"] == customer_id
            assert gist["user_id"] == user_id
            assert gist["conversation_id"] == newest_conv
            assert gist["content"] == "merged gist"
            assert abs(float(gist["salience"]) - 0.5) < 1e-6
            assert gist["evergreen"] is False
            assert gist["superseded_by"] is None
            assert gist["has_embed"] is True

            # provenance edges recorded (gist <- each of the 3 sources).
            edges = MemoryConsolidationsCollection(
                registry=_registry(pool), config=_cfg(), nats_client=None
            )
            found = await edges.find_sources(agent_id, gist_id)
            assert sorted(str(s) for s in found) == sorted(str(s) for s in s_ids)

            # sources superseded by the gist; the singleton untouched.
            for s in s_ids:
                assert await _superseded_by(conn, s) == str(gist_id)
            assert await _superseded_by(conn, singleton) is None

            # non-destructive: sources still exist AND stay directly recallable.
            mems = MemoriesCollection(
                registry=_registry(pool), config=_cfg(), authorizer=MagicMock(spec=MemoryAuthorizerDependencies)
            )
            direct = await mems.search_by_ids(s_ids, user_id, agent_id=agent_id)
            assert len(direct) == 3

            # a second consolidation load no longer surfaces the superseded
            # sources (a live gist now represents them).
            active = await mems.find_active_for_consolidation(
                agent_id, customer_id=customer_id, user_id=user_id
            )
            active_ids = {c["memory_id"] for c in active}
            assert not (set(s_ids) & active_ids)
        finally:
            await conn.close()
            await pool.close()

    async def test_scope_isolation_across_users(self, applied_schema: tuple[str, str]) -> None:
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            agent_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            user_a = uuid.uuid4()
            user_b = uuid.uuid4()

            a_ids = [
                (await _insert_source(conn, agent_id=agent_id, customer_id=customer_id, user_id=user_a, seed=0.05, content=f"a{i}"))[0]
                for i in range(2)
            ]
            b_ids = [
                (await _insert_source(conn, agent_id=agent_id, customer_id=customer_id, user_id=user_b, seed=0.05, content=f"b{i}"))[0]
                for i in range(2)
            ]

            service = _make_service(pool, gist_vector=_vec(0.9))
            result = await service.run_consolidation(agent_id, customer_id=customer_id, user_id=user_a)

            assert result.gists_created == 1
            gist_id = result.gist_ids[0]

            # only user A's rows superseded; user B's are untouched.
            for s in a_ids:
                assert await _superseded_by(conn, s) == str(gist_id)
            for s in b_ids:
                assert await _superseded_by(conn, s) is None

            # the gist is user-A scoped (isolation is the user_id boundary).
            gist_user = await conn.fetchval(
                "SELECT user_id FROM memories WHERE memory_id = $1", gist_id
            )
            assert gist_user == user_a
        finally:
            await conn.close()
            await pool.close()

    async def test_orphaned_supersession_self_heals(self, applied_schema: tuple[str, str]) -> None:
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            agent_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            user_id = uuid.uuid4()

            # a real live gist supersedes a source -> that source stays excluded.
            live_gist, _ = await _insert_source(
                conn, agent_id=agent_id, customer_id=customer_id, user_id=user_id, seed=0.9, content="live gist"
            )
            live_superseded, _ = await _insert_source(
                conn, agent_id=agent_id, customer_id=customer_id, user_id=user_id,
                seed=0.05, content="represented by a live gist", superseded_by=live_gist,
            )
            # a DEAD gist (id never inserted) supersedes two sources -> orphaned;
            # they must become eligible candidates again.
            dead_gist = uuid.uuid4()
            orphan_a, _ = await _insert_source(
                conn, agent_id=agent_id, customer_id=customer_id, user_id=user_id,
                seed=0.05, content="orphan a", superseded_by=dead_gist,
            )
            orphan_b, _ = await _insert_source(
                conn, agent_id=agent_id, customer_id=customer_id, user_id=user_id,
                seed=0.05, content="orphan b", superseded_by=dead_gist,
            )

            mems = MemoriesCollection(
                registry=_registry(pool), config=_cfg(), authorizer=MagicMock(spec=MemoryAuthorizerDependencies)
            )
            active = await mems.find_active_for_consolidation(
                agent_id, customer_id=customer_id, user_id=user_id
            )
            active_ids = {c["memory_id"] for c in active}

            # orphaned-by-dead-gist sources are eligible again; the
            # live-gist-superseded source is not.
            assert orphan_a in active_ids
            assert orphan_b in active_ids
            assert live_superseded not in active_ids
        finally:
            await conn.close()
            await pool.close()


def _registry(pool: asyncpg.Pool) -> CollectionRegistry:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    return registry


def _cfg() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
