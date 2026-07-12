"""integration: memory_consolidations edge table + DAG cycle-guard (v026).

Presence/aliveness program (3tears v0.15.0), chunk A4. Exercises the N:1
provenance edge against a real pgvector/pg16 database:

1. edge insert + N-source fan-in (many sources -> one gist).
2. ON DELETE CASCADE — deleting either endpoint memory removes its edges
   while leaving the surviving endpoint's own row (and salience) intact.
3. ``find_sources`` / ``find_consolidated_into`` read the forward + back
   edges.
4. the cycle-guard's DB path (``assert_no_cycle``) accepts a fresh /
   diamond consolidation and rejects a self- or descendant-merge, reading
   real edges through ``find_sources``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import asyncpg
import pytest

from threetears.agent.memory.collections import (
    ConsolidationCycleError,
    MemoryConsolidationsCollection,
)
from threetears.agent.memory.migrations import register as register_memory
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration


@pytest.fixture
async def applied_schema(pg_schema: tuple[str, str]) -> tuple[str, str]:
    """apply conversations + memory migrations into the per-test schema.

    :param pg_schema: ``(url, schema)`` tuple from the shared fixture
    :ptype pg_schema: tuple[str, str]
    :return: same tuple with migrations applied
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
    """build an asyncpg pool with search_path pre-bound to the schema.

    :param url: asyncpg URL
    :ptype url: str
    :param schema: per-test schema name
    :ptype schema: str
    :return: pool ready for Collection use
    :rtype: asyncpg.Pool
    """
    from threetears.core.collections import init_connection

    result: asyncpg.Pool = await asyncpg.create_pool(
        dsn=url,
        min_size=1,
        max_size=4,
        server_settings={"search_path": f"{schema}, public"},
        init=init_connection,
    )
    return result


def _build_collection(pool: asyncpg.Pool) -> MemoryConsolidationsCollection:
    """construct the collection over an L3-only registry (reads + guard).

    The forward/back-edge reads and the cycle-guard touch only
    ``l3_pool``; L1 / L2 are unnecessary for this suite.

    :param pool: pg pool bound to the per-test schema
    :ptype pool: asyncpg.Pool
    :return: a read-ready collection
    :rtype: MemoryConsolidationsCollection
    """
    reg = CollectionRegistry()
    reg.configure(l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return MemoryConsolidationsCollection(registry=reg, config=cfg, nats_client=None)


async def _insert_memory(conn: asyncpg.Connection, agent_id: uuid.UUID, memory_id: uuid.UUID) -> None:
    """insert a minimal fully-scoped memory row (FK target for edges).

    :param conn: live asyncpg connection
    :ptype conn: asyncpg.Connection
    :param agent_id: partition agent id
    :ptype agent_id: uuid.UUID
    :param memory_id: memory id to create
    :ptype memory_id: uuid.UUID
    :return: nothing
    :rtype: None
    """
    now = datetime.now(UTC)
    await conn.execute(
        "INSERT INTO memories ("
        "memory_id, agent_id, customer_id, user_id, "
        "conversation_id, type_memory, content, date_created, date_updated"
        ") VALUES ($1, $2, $3, $4, $5, 'fact', $6, $7, $7)",
        memory_id,
        agent_id,
        uuid.uuid4(),
        uuid.uuid4(),
        uuid.uuid4(),
        f"memory {memory_id}",
        now,
    )


async def _insert_edge(
    conn: asyncpg.Connection,
    agent_id: uuid.UUID,
    gist_id: uuid.UUID,
    source_id: uuid.UUID,
    rationale: str | None = None,
) -> None:
    """insert one consolidation edge (gist <- source).

    :param conn: live asyncpg connection
    :ptype conn: asyncpg.Connection
    :param agent_id: partition agent id
    :ptype agent_id: uuid.UUID
    :param gist_id: the consolidated (gist) memory id
    :ptype gist_id: uuid.UUID
    :param source_id: the source memory id
    :ptype source_id: uuid.UUID
    :param rationale: optional audit text
    :ptype rationale: str | None
    :return: nothing
    :rtype: None
    """
    now = datetime.now(UTC)
    await conn.execute(
        "INSERT INTO memory_consolidations ("
        "agent_id, consolidated_memory_id, source_memory_id, rationale, "
        "date_created, date_updated"
        ") VALUES ($1, $2, $3, $4, $5, $5)",
        agent_id,
        gist_id,
        source_id,
        rationale,
        now,
    )


class TestMemoryConsolidationsEdge:
    """edge insert, N-source fan-in, CASCADE, and the read helpers."""

    async def test_n_source_fan_in_and_find_sources(self, applied_schema: tuple[str, str]) -> None:
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            agent_id = uuid.uuid4()
            gist = uuid.uuid4()
            sources = [uuid.uuid4() for _ in range(3)]
            async with pool.acquire() as conn:
                await _insert_memory(conn, agent_id, gist)
                for s in sources:
                    await _insert_memory(conn, agent_id, s)
                    await _insert_edge(conn, agent_id, gist, s, rationale="near-dup")

            coll = _build_collection(pool)
            found = await coll.find_sources(agent_id, gist)
            assert sorted(str(s) for s in found) == sorted(str(s) for s in sources)

            # back-edge: each source resolves to the one gist it fed.
            for s in sources:
                into = await coll.find_consolidated_into(agent_id, s)
                assert [str(g) for g in into] == [str(gist)]
        finally:
            await pool.close()

    async def test_delete_source_cascades_edge_but_keeps_gist(self, applied_schema: tuple[str, str]) -> None:
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            agent_id = uuid.uuid4()
            gist = uuid.uuid4()
            source = uuid.uuid4()
            async with pool.acquire() as conn:
                await _insert_memory(conn, agent_id, gist)
                await _insert_memory(conn, agent_id, source)
                await _insert_edge(conn, agent_id, gist, source)

                # delete the SOURCE memory: its edge cascades away, but the
                # gist row itself survives (non-destructive to the gist).
                await conn.execute(
                    "DELETE FROM memories WHERE agent_id = $1 AND memory_id = $2",
                    agent_id,
                    source,
                )
                edge_count = await conn.fetchval(
                    "SELECT count(*) FROM memory_consolidations WHERE agent_id = $1",
                    agent_id,
                )
                gist_exists = await conn.fetchval(
                    "SELECT count(*) FROM memories WHERE agent_id = $1 AND memory_id = $2",
                    agent_id,
                    gist,
                )
            assert edge_count == 0
            assert gist_exists == 1
        finally:
            await pool.close()

    async def test_delete_gist_cascades_edge_but_keeps_source(self, applied_schema: tuple[str, str]) -> None:
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            agent_id = uuid.uuid4()
            gist = uuid.uuid4()
            source = uuid.uuid4()
            async with pool.acquire() as conn:
                await _insert_memory(conn, agent_id, gist)
                await _insert_memory(conn, agent_id, source)
                await _insert_edge(conn, agent_id, gist, source)

                await conn.execute(
                    "DELETE FROM memories WHERE agent_id = $1 AND memory_id = $2",
                    agent_id,
                    gist,
                )
                edge_count = await conn.fetchval(
                    "SELECT count(*) FROM memory_consolidations WHERE agent_id = $1",
                    agent_id,
                )
                source_exists = await conn.fetchval(
                    "SELECT count(*) FROM memories WHERE agent_id = $1 AND memory_id = $2",
                    agent_id,
                    source,
                )
            assert edge_count == 0
            assert source_exists == 1
        finally:
            await pool.close()


class TestMemoryConsolidationsCycleGuardDb:
    """the cycle-guard reading real edges via find_sources."""

    async def test_fresh_consolidation_passes(self, applied_schema: tuple[str, str]) -> None:
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            agent_id = uuid.uuid4()
            gist = uuid.uuid4()
            sources = [uuid.uuid4(), uuid.uuid4()]
            async with pool.acquire() as conn:
                await _insert_memory(conn, agent_id, gist)
                for s in sources:
                    await _insert_memory(conn, agent_id, s)

            coll = _build_collection(pool)
            # no existing edges -> no cycle.
            await coll.assert_no_cycle(
                agent_id,
                consolidated_memory_id=gist,
                source_memory_ids=sources,
            )
        finally:
            await pool.close()

    async def test_self_merge_rejected(self, applied_schema: tuple[str, str]) -> None:
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            agent_id = uuid.uuid4()
            gist = uuid.uuid4()
            coll = _build_collection(pool)
            with pytest.raises(ConsolidationCycleError):
                await coll.assert_no_cycle(
                    agent_id,
                    consolidated_memory_id=gist,
                    source_memory_ids=[gist],
                )
        finally:
            await pool.close()

    async def test_descendant_merge_rejected(self, applied_schema: tuple[str, str]) -> None:
        # existing edge: source S was consolidated from gist G (S <- G).
        # consolidating G from S now would close the loop.
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            agent_id = uuid.uuid4()
            gist = uuid.uuid4()
            source = uuid.uuid4()
            async with pool.acquire() as conn:
                await _insert_memory(conn, agent_id, gist)
                await _insert_memory(conn, agent_id, source)
                # S is a gist consolidated from G in a prior round.
                await _insert_edge(conn, agent_id, source, gist)

            coll = _build_collection(pool)
            with pytest.raises(ConsolidationCycleError):
                await coll.assert_no_cycle(
                    agent_id,
                    consolidated_memory_id=gist,
                    source_memory_ids=[source],
                )
        finally:
            await pool.close()

    async def test_diamond_dag_accepted(self, applied_schema: tuple[str, str]) -> None:
        # s1 and s2 both consolidated from a common ancestor `a` — a valid
        # DAG. Consolidating a NEW gist from {s1, s2} must not false-positive.
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            agent_id = uuid.uuid4()
            gist = uuid.uuid4()
            s1, s2, ancestor = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            async with pool.acquire() as conn:
                for m in (gist, s1, s2, ancestor):
                    await _insert_memory(conn, agent_id, m)
                await _insert_edge(conn, agent_id, s1, ancestor)
                await _insert_edge(conn, agent_id, s2, ancestor)

            coll = _build_collection(pool)
            await coll.assert_no_cycle(
                agent_id,
                consolidated_memory_id=gist,
                source_memory_ids=[s1, s2],
            )
        finally:
            await pool.close()
