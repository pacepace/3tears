"""integration: v024 salience decay, reinforcement, and ambient gating.

Exercises the decay/reinforcement cycle and the ambient-vs-direct
retrieval split against real Postgres:

- ``MemoriesCollection.decay_salience`` -- cadence-independent decay
  (two steps over 20d + 10d equal one step over 30d), the floor asymptote
  (never below floor, never deleted), and evergreen exclusion.
- ``MemoriesCollection.bump_salience`` -- reinforcement increment clamped
  to 1.0, ``last_accessed`` stamped, evergreen skipped.
- ``MemoriesCollection.hybrid_search`` -- ambient retrieval drops rows
  below the salience floor and rows superseded by a consolidation gist,
  while ``search_by_ids`` (direct recall) still returns them.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import asyncpg
import pytest

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import MemoriesCollection
from threetears.agent.memory.migrations import register as register_memory
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration

_DIM = 1024
_EMBED = "[" + ",".join(["0.1"] * _DIM) + "]"


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
    """asyncpg pool with search_path pre-bound to the test schema."""
    pool: asyncpg.Pool = await asyncpg.create_pool(
        dsn=url,
        min_size=1,
        max_size=4,
        server_settings={"search_path": f"{schema}, public"},
        init=init_connection,
    )
    return pool


def _build_collection(pool: asyncpg.Pool) -> MemoriesCollection:
    """registry-bound MemoriesCollection; authorizer unused by these paths."""
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    authorizer = MagicMock(spec=MemoryAuthorizerDependencies)
    return MemoriesCollection(registry=registry, config=config, authorizer=authorizer)


async def _insert_memory(
    conn: asyncpg.Connection,
    *,
    agent_id: uuid.UUID,
    user_id: uuid.UUID,
    customer_id: uuid.UUID,
    salience: float = 0.5,
    evergreen: bool = False,
    superseded_by: uuid.UUID | None = None,
    last_decayed_at: datetime | None = None,
    content: str = "a durable fact",
    with_embedding: bool = False,
) -> uuid.UUID:
    """insert one memory row; returns its memory_id."""
    memory_id = uuid.uuid4()
    now = datetime.now(UTC)
    embedding_sql = "$11::text::public.vector" if with_embedding else "NULL"
    params: list[Any] = [
        memory_id,
        agent_id,
        customer_id,
        user_id,
        uuid.uuid4(),  # conversation_id
        content,
        salience,
        evergreen,
        superseded_by,
        last_decayed_at,
    ]
    if with_embedding:
        params.append(_EMBED)
    await conn.execute(
        "INSERT INTO memories ("
        "memory_id, agent_id, customer_id, user_id, conversation_id, "
        "type_memory, content, salience, evergreen, superseded_by, "
        "last_decayed_at, embedding, date_created, date_updated"
        f") VALUES ($1,$2,$3,$4,$5,'fact',$6,$7,$8,$9,$10,{embedding_sql},now(),now())",
        *params,
    )
    return memory_id


async def _salience(conn: asyncpg.Connection, memory_id: uuid.UUID) -> float:
    val = await conn.fetchval("SELECT salience FROM memories WHERE memory_id = $1", memory_id)
    return float(val)


class TestDecaySalience:
    async def test_two_step_decay_equals_single_step(self, applied_schema: tuple[str, str]) -> None:
        """20d + 10d in two calls equals a single 30d decay (cadence-safe).

        The formula is multiplicative and anchored on ``last_decayed_at``,
        so 0.5^(20/h) * 0.5^(10/h) = 0.5^(30/h) regardless of how the run
        is split.
        """
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            agent_id = uuid.uuid4()
            user_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            coll = _build_collection(pool)

            mem_id = await _insert_memory(
                conn,
                agent_id=agent_id,
                user_id=user_id,
                customer_id=customer_id,
                salience=1.0,
            )
            # anchor 20 days in the past
            await conn.execute(
                "UPDATE memories SET last_decayed_at = now() - interval '20 days' WHERE memory_id = $1",
                mem_id,
            )
            await coll.decay_salience(half_life_days=60.0, floor=0.1)
            # simulate 10 more days elapsing since the last decay run
            await conn.execute(
                "UPDATE memories SET last_decayed_at = now() - interval '10 days' WHERE memory_id = $1",
                mem_id,
            )
            await coll.decay_salience(half_life_days=60.0, floor=0.1)

            got = await _salience(conn, mem_id)
            expected = 0.5 ** (30.0 / 60.0)  # single 30-day decay of salience 1.0
            assert abs(got - expected) < 0.01, f"cadence drift: {got} vs {expected}"
        finally:
            await conn.close()
            await pool.close()

    async def test_floor_is_asymptote_never_deletes(self, applied_schema: tuple[str, str]) -> None:
        """Salience decays toward the floor and never drops below it."""
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            coll = _build_collection(pool)
            mem_id = await _insert_memory(
                conn,
                agent_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                customer_id=uuid.uuid4(),
                salience=0.2,
            )
            await conn.execute(
                "UPDATE memories SET last_decayed_at = now() - interval '3650 days' WHERE memory_id = $1",
                mem_id,
            )
            await coll.decay_salience(half_life_days=60.0, floor=0.1)
            assert abs(await _salience(conn, mem_id) - 0.1) < 1e-4

            # row still exists (never deleted) and never sinks below floor
            await coll.decay_salience(half_life_days=60.0, floor=0.1)
            assert abs(await _salience(conn, mem_id) - 0.1) < 1e-4
            exists = await conn.fetchval("SELECT count(*) FROM memories WHERE memory_id = $1", mem_id)
            assert exists == 1
        finally:
            await conn.close()
            await pool.close()

    async def test_evergreen_is_excluded_from_decay(self, applied_schema: tuple[str, str]) -> None:
        """Evergreen memories never decay."""
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            coll = _build_collection(pool)
            mem_id = await _insert_memory(
                conn,
                agent_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                customer_id=uuid.uuid4(),
                salience=0.8,
                evergreen=True,
            )
            await conn.execute(
                "UPDATE memories SET last_decayed_at = now() - interval '3650 days' WHERE memory_id = $1",
                mem_id,
            )
            await coll.decay_salience(half_life_days=60.0, floor=0.1)
            assert abs(await _salience(conn, mem_id) - 0.8) < 1e-4
        finally:
            await conn.close()
            await pool.close()


class TestBumpSalience:
    async def test_bump_increments_and_stamps_and_caps(self, applied_schema: tuple[str, str]) -> None:
        """Reinforcement adds the bump, stamps last_accessed, caps at 1.0."""
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            agent_id = uuid.uuid4()
            user_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            coll = _build_collection(pool)

            mid = await _insert_memory(conn, agent_id=agent_id, user_id=user_id, customer_id=customer_id, salience=0.5)
            near_cap = await _insert_memory(
                conn, agent_id=agent_id, user_id=user_id, customer_id=customer_id, salience=0.98
            )

            await coll.bump_salience([mid, near_cap], agent_id=agent_id, access_bump=0.05)

            assert abs(await _salience(conn, mid) - 0.55) < 1e-4
            assert abs(await _salience(conn, near_cap) - 1.0) < 1e-4  # LEAST(1.0, 1.03)
            stamped = await conn.fetchval("SELECT last_accessed FROM memories WHERE memory_id = $1", mid)
            assert stamped is not None
        finally:
            await conn.close()
            await pool.close()

    async def test_bump_skips_evergreen(self, applied_schema: tuple[str, str]) -> None:
        """Evergreen memories are not bumped (pinned)."""
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            agent_id = uuid.uuid4()
            coll = _build_collection(pool)
            mid = await _insert_memory(
                conn,
                agent_id=agent_id,
                user_id=uuid.uuid4(),
                customer_id=uuid.uuid4(),
                salience=0.5,
                evergreen=True,
            )
            await coll.bump_salience([mid], agent_id=agent_id, access_bump=0.05)
            assert abs(await _salience(conn, mid) - 0.5) < 1e-4
        finally:
            await conn.close()
            await pool.close()


class TestAmbientGatingVsDirectRecall:
    async def test_ambient_floor_and_supersession_vs_direct(self, applied_schema: tuple[str, str]) -> None:
        """Ambient search drops dormant + superseded; direct recall keeps them."""
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            agent_id = uuid.uuid4()
            user_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            coll = _build_collection(pool)

            visible = await _insert_memory(
                conn,
                agent_id=agent_id,
                user_id=user_id,
                customer_id=customer_id,
                salience=0.9,
                with_embedding=True,
                content="visible salient memory",
            )
            dormant = await _insert_memory(
                conn,
                agent_id=agent_id,
                user_id=user_id,
                customer_id=customer_id,
                salience=0.15,
                with_embedding=True,
                content="dormant faded memory",
            )
            superseded = await _insert_memory(
                conn,
                agent_id=agent_id,
                user_id=user_id,
                customer_id=customer_id,
                salience=0.9,
                superseded_by=uuid.uuid4(),
                with_embedding=True,
                content="superseded merged memory",
            )

            results = await coll.hybrid_search(
                user_id=user_id,
                agent_id=agent_id,
                customer_id=customer_id,
                embedding=[0.1] * _DIM,
                user_text="memory",
                top_k=10,
                candidate_limit=50,
                similarity_threshold=0.0,
                recency_half_life_hours=24.0,
                signal_weights={"semantic": 0.55, "keyword": 0.15, "recency": 0.30},
                salience_ambient_floor=0.2,
            )
            ambient_ids = {r["memory_id"] for r in results}
            assert visible in ambient_ids
            assert dormant not in ambient_ids  # below the 0.2 floor
            assert superseded not in ambient_ids  # replaced by a gist

            # direct recall bypasses BOTH gates -- dormant, not gone
            direct = await coll.search_by_ids([dormant, superseded], user_id, agent_id=agent_id)
            direct_ids = {r["memory_id"] for r in direct}
            assert dormant in direct_ids
            assert superseded in direct_ids
        finally:
            await conn.close()
            await pool.close()
