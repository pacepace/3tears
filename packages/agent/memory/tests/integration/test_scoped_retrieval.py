"""integration matrix for :class:`RetrievalScope` (scoped retrieval).

the two tests that matter most, per the proposal that specified this:

- **under-fill negative control** -- eligible rows ranked BELOW the
  unfiltered top-k must still fill ``top_k``. this is the test that fails
  if anyone reimplements the scope as a post-filter.
- **FTS-arm coverage** -- an ineligible row that only the FTS arm would
  surface (weak vector similarity, strong lexical match) must not appear.
  catches a predicate applied to the vector arm only.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import asyncpg
import pytest

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import MemoriesCollection
from threetears.agent.memory.migrations import register as register_memory
from threetears.agent.memory.retrieval_scope import RetrievalScope
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore

pytestmark = pytest.mark.integration

_DIM = 1024
_HALF = _DIM // 2


def _vec(front: float, back: float) -> str:
    """1024-dim vector: first half ``front``, second half ``back``.

    two blocks give controllable cosine angles: (1,0) query against a
    (1,0) row is similarity 1.0; against a (0,1) row it is 0.0.
    """
    return "[" + ",".join([str(front)] * _HALF + [str(back)] * _HALF) + "]"


_QUERY_EMBEDDING = [1.0] * _HALF + [0.0] * _HALF


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
    assert pool is not None
    return pool


def _build_collection(pool: asyncpg.Pool) -> MemoriesCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    authorizer = MagicMock(spec=MemoryAuthorizerDependencies)
    return MemoriesCollection(registry=registry, config=config, authorizer=authorizer)


async def _insert(
    pool: asyncpg.Pool,
    *,
    agent_id: uuid.UUID,
    customer_id: uuid.UUID,
    user_id: uuid.UUID,
    content: str,
    embedding: str,
    tags: list[str] | None = None,
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
        tags or [],
        embedding,
    )
    return memory_id


_HYBRID_DEFAULTS: dict[str, Any] = {
    "top_k": 3,
    "candidate_limit": 20,
    "similarity_threshold": -2.0,
    "recency_half_life_hours": 999999.0,
    "signal_weights": {"semantic": 1.0, "keyword": 1.0, "recency": 0.0},
}


class TestScopedRetrieval:
    """the proposal's matrix, driven against a real database."""

    async def test_no_scope_is_todays_behaviour(self, applied_schema: tuple[str, str]) -> None:
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            agent, customer, user = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            for i in range(3):
                await _insert(
                    pool,
                    agent_id=agent,
                    customer_id=customer,
                    user_id=user,
                    content=f"fact {i}",
                    embedding=_vec(1.0, 0.0),
                )
            coll = _build_collection(pool)
            unscoped = await coll.search_by_semantic(
                user_id=user,
                agent_id=agent,
                embedding=_QUERY_EMBEDDING,
                max_results=10,
                similarity_threshold=-2.0,
            )
            with_none = await coll.search_by_semantic(
                user_id=user,
                agent_id=agent,
                embedding=_QUERY_EMBEDDING,
                max_results=10,
                similarity_threshold=-2.0,
                scope=None,
            )
            assert [r["memory_id"] for r in unscoped] == [r["memory_id"] for r in with_none]
            assert len(unscoped) == 3
        finally:
            await pool.close()

    async def test_tags_and_id_predicates_bound_the_result(self, applied_schema: tuple[str, str]) -> None:
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            agent, customer, user = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            pa = await _insert(
                pool,
                agent_id=agent,
                customer_id=customer,
                user_id=user,
                content="pa row",
                embedding=_vec(1.0, 0.0),
                tags=["state:PA", "org:durp"],
            )
            oh = await _insert(
                pool,
                agent_id=agent,
                customer_id=customer,
                user_id=user,
                content="oh row",
                embedding=_vec(1.0, 0.0),
                tags=["state:OH"],
            )
            await _insert(
                pool,
                agent_id=agent,
                customer_id=customer,
                user_id=user,
                content="untagged row",
                embedding=_vec(1.0, 0.0),
            )
            coll = _build_collection(pool)

            any_hit = await coll.search_by_semantic(
                user_id=user,
                agent_id=agent,
                embedding=_QUERY_EMBEDDING,
                max_results=10,
                similarity_threshold=-2.0,
                scope=RetrievalScope(tags_any=("state:PA", "state:MI")),
            )
            assert [r["memory_id"] for r in any_hit] == [str(pa)]

            all_hit = await coll.search_by_semantic(
                user_id=user,
                agent_id=agent,
                embedding=_QUERY_EMBEDDING,
                max_results=10,
                similarity_threshold=-2.0,
                scope=RetrievalScope(tags_all=("state:PA", "org:durp")),
            )
            assert [r["memory_id"] for r in all_hit] == [str(pa)]

            restricted = await coll.search_by_semantic(
                user_id=user,
                agent_id=agent,
                embedding=_QUERY_EMBEDDING,
                max_results=10,
                similarity_threshold=-2.0,
                scope=RetrievalScope(restrict_to_ids=frozenset({oh})),
            )
            assert [r["memory_id"] for r in restricted] == [str(oh)]

            excluded = await coll.search_by_semantic(
                user_id=user,
                agent_id=agent,
                embedding=_QUERY_EMBEDDING,
                max_results=10,
                similarity_threshold=-2.0,
                scope=RetrievalScope(exclude_ids=frozenset({pa, oh})),
            )
            assert str(pa) not in [r["memory_id"] for r in excluded]
            assert str(oh) not in [r["memory_id"] for r in excluded]
            assert len(excluded) == 1
        finally:
            await pool.close()

    async def test_an_empty_allow_list_is_fail_closed(self, applied_schema: tuple[str, str]) -> None:
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            agent, customer, user = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            await _insert(
                pool,
                agent_id=agent,
                customer_id=customer,
                user_id=user,
                content="present row",
                embedding=_vec(1.0, 0.0),
            )
            coll = _build_collection(pool)
            empty_scope = RetrievalScope(restrict_to_ids=frozenset())
            assert (
                await coll.search_by_semantic(
                    user_id=user,
                    agent_id=agent,
                    embedding=_QUERY_EMBEDDING,
                    max_results=10,
                    similarity_threshold=-2.0,
                    scope=empty_scope,
                )
                == []
            )
            assert (
                await coll.search_by_fts(
                    user_id=user,
                    agent_id=agent,
                    fts_text="present",
                    max_results=10,
                    scope=empty_scope,
                )
                == []
            )
            assert (
                await coll.hybrid_search(
                    user_id=user,
                    agent_id=agent,
                    customer_id=customer,
                    embedding=_QUERY_EMBEDDING,
                    user_text="present",
                    scope=empty_scope,
                    **_HYBRID_DEFAULTS,
                )
                == []
            )
        finally:
            await pool.close()

    async def test_under_fill_negative_control(self, applied_schema: tuple[str, str]) -> None:
        """eligible rows below the unfiltered top-k still fill top_k.

        the test that fails if the scope is a post-filter.
        """
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            agent, customer, user = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            # five near, untagged rows dominate the unfiltered top-k...
            for i in range(5):
                await _insert(
                    pool,
                    agent_id=agent,
                    customer_id=customer,
                    user_id=user,
                    content=f"near untagged {i}",
                    embedding=_vec(1.0, 0.0),
                )
            # ...while the three eligible rows rank far below it.
            eligible = [
                await _insert(
                    pool,
                    agent_id=agent,
                    customer_id=customer,
                    user_id=user,
                    content=f"far tagged {i}",
                    embedding=_vec(0.0, 1.0),
                    tags=["wanted"],
                )
                for i in range(3)
            ]
            coll = _build_collection(pool)
            rows = await coll.hybrid_search(
                user_id=user,
                agent_id=agent,
                customer_id=customer,
                embedding=_QUERY_EMBEDDING,
                user_text="",
                scope=RetrievalScope(tags_any=("wanted",)),
                **_HYBRID_DEFAULTS,
            )
            assert len(rows) == 3, f"top_k under-filled: {len(rows)} of 3 -- scope ran as a post-filter"
            assert {str(r["memory_id"]) for r in rows} == {str(m) for m in eligible}
        finally:
            await pool.close()

    async def test_fts_arm_cannot_leak_an_ineligible_row(self, applied_schema: tuple[str, str]) -> None:
        """a row only the FTS arm would surface stays out when ineligible."""
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            agent, customer, user = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            eligible = await _insert(
                pool,
                agent_id=agent,
                customer_id=customer,
                user_id=user,
                content="ordinary eligible fact",
                embedding=_vec(1.0, 0.0),
            )
            # strong lexical match, weak vector similarity: the FTS arm's darling.
            await _insert(
                pool,
                agent_id=agent,
                customer_id=customer,
                user_id=user,
                content="zanzibar zanzibar zanzibar",
                embedding=_vec(0.0, 1.0),
            )
            coll = _build_collection(pool)
            rows = await coll.hybrid_search(
                user_id=user,
                agent_id=agent,
                customer_id=customer,
                embedding=_QUERY_EMBEDDING,
                user_text="zanzibar",
                scope=RetrievalScope(restrict_to_ids=frozenset({eligible})),
                **_HYBRID_DEFAULTS,
            )
            contents = [r["content"] for r in rows]
            assert all("zanzibar" not in c for c in contents), contents
            assert [str(r["memory_id"]) for r in rows] == [str(eligible)]
        finally:
            await pool.close()

    async def test_scope_composes_with_the_date_filter(self, applied_schema: tuple[str, str]) -> None:
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            agent, customer, user = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            tagged = await _insert(
                pool,
                agent_id=agent,
                customer_id=customer,
                user_id=user,
                content="tagged now",
                embedding=_vec(1.0, 0.0),
                tags=["keep"],
            )
            await _insert(
                pool,
                agent_id=agent,
                customer_id=customer,
                user_id=user,
                content="untagged now",
                embedding=_vec(1.0, 0.0),
            )
            # push one tagged row before the date window.
            old = await _insert(
                pool,
                agent_id=agent,
                customer_id=customer,
                user_id=user,
                content="tagged but old",
                embedding=_vec(1.0, 0.0),
                tags=["keep"],
            )
            await pool.execute(
                "UPDATE memories SET date_created = now() - interval '10 days' WHERE agent_id = $1 AND memory_id = $2",
                agent,
                old,
            )
            coll = _build_collection(pool)
            rows = await coll.search_by_semantic(
                user_id=user,
                agent_id=agent,
                embedding=_QUERY_EMBEDDING,
                max_results=10,
                similarity_threshold=-2.0,
                date_after=__import__("datetime").datetime.now(__import__("datetime").UTC)
                - __import__("datetime").timedelta(days=1),
                scope=RetrievalScope(tags_any=("keep",)),
            )
            assert [r["memory_id"] for r in rows] == [str(tagged)]
        finally:
            await pool.close()
