"""integration: agent-intention v001 migration + three-tier collection CRUD.

Exercises against a real pgvector/pg16 container:

- the v001 migration chain applies cleanly (table + enum + indexes),
  replay is a no-op, and the ``intention_status`` enum carries exactly
  the four lifecycle values;
- the ``status`` server default (``open``) applies to a raw INSERT that
  omits it;
- ``IntentionsCollection`` save + get round-trips through L1 / L2 / L3;
- ``find_by_user`` isolates by ``user_id`` (the required boundary) and
  ranks by salience -- one metallm user's wants never surface for
  another sharing the same ``agent_id``.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest

from threetears.agent.intention.types import INTENTION_STATUS_VALUES

from .conftest import (
    AsyncpgStore,
    build_collection,
    make_pool,
    runner_apply,
)
from .conftest import InMemoryNatsBus as _InMemoryNatsBus


pytestmark = pytest.mark.integration


def _make_row(agent_id: uuid.UUID, user_id: uuid.UUID, *, salience: float, content: str) -> dict[str, Any]:
    """an open-status intention write dict for the collection create path."""
    now = datetime.now(UTC)
    return {
        "intention_id": uuid.uuid4(),
        "agent_id": agent_id,
        "customer_id": uuid.uuid4(),
        "user_id": user_id,
        "status": "open",
        "content": content,
        "salience": salience,
        "date_created": now,
        "date_updated": now,
    }


class TestMigrationChain:
    """v001 applies cleanly, replays as a no-op, and builds the enum."""

    async def test_chain_applies_replays_and_builds_schema(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = AsyncpgStore(conn)
            count = await runner_apply(conn, store)
            assert count > 0

            # table + expected columns present
            cols = {
                r["column_name"]
                for r in await conn.fetch(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = $1 AND table_name = 'intentions'",
                    schema,
                )
            }
            for expected in (
                "intention_id",
                "agent_id",
                "customer_id",
                "user_id",
                "status",
                "content",
                "embedding",
                "salience",
                "last_decayed_at",
                "last_surfaced_at",
                "source_memory_id",
                "source_conversation_id",
                "date_created",
                "date_updated",
            ):
                assert expected in cols, f"missing column {expected!r}"

            # customer_id / user_id nullable scope grains; salience NOT NULL.
            nullability = {
                r["column_name"]: r["is_nullable"]
                for r in await conn.fetch(
                    "SELECT column_name, is_nullable FROM information_schema.columns "
                    "WHERE table_schema = $1 AND table_name = 'intentions'",
                    schema,
                )
            }
            assert nullability["customer_id"] == "YES"
            assert nullability["user_id"] == "YES"
            assert nullability["salience"] == "NO"

            # all three indexes present
            index_names = {
                r["indexname"]
                for r in await conn.fetch(
                    "SELECT indexname FROM pg_indexes WHERE schemaname = $1 AND tablename = 'intentions'",
                    schema,
                )
            }
            assert "idx_intentions_open_ranked" in index_names
            assert "idx_intentions_last_surfaced" in index_names
            assert "ix_intentions_embedding_hnsw" in index_names

            # re-apply is a no-op
            count2 = await runner_apply(conn, store)
            assert count2 == 0
        finally:
            await conn.close()

    async def test_intention_status_enum_values(self, pg_schema: tuple[str, str]) -> None:
        """The live PG enum carries exactly the four lifecycle values, in order."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = AsyncpgStore(conn)
            await runner_apply(conn, store)

            labels = [
                r["enumlabel"]
                for r in await conn.fetch(
                    "SELECT e.enumlabel FROM pg_enum e "
                    "JOIN pg_type t ON t.oid = e.enumtypid "
                    "JOIN pg_namespace n ON n.oid = t.typnamespace "
                    "WHERE t.typname = 'intention_status' AND n.nspname = $1 "
                    "ORDER BY e.enumsortorder",
                    schema,
                )
            ]
            assert tuple(labels) == INTENTION_STATUS_VALUES
        finally:
            await conn.close()

    async def test_status_server_default_applies(self, pg_schema: tuple[str, str]) -> None:
        """A raw INSERT omitting status takes the ``open`` server default."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = AsyncpgStore(conn)
            await runner_apply(conn, store)

            now = datetime.now(UTC)
            agent_id = uuid.uuid4()
            intention_id = uuid.uuid4()
            await conn.execute(
                "INSERT INTO intentions (intention_id, agent_id, content, date_created) VALUES ($1, $2, $3, $4)",
                intention_id,
                agent_id,
                "a want with no explicit status",
                now,
            )
            row = await conn.fetchrow(
                "SELECT status, salience FROM intentions WHERE agent_id = $1 AND intention_id = $2",
                agent_id,
                intention_id,
            )
            assert row is not None
            assert row["status"] == "open"
            assert float(row["salience"]) == 0.5  # salience server default too
        finally:
            await conn.close()


class TestIntentionsCollectionThreeTier:
    """save + get round-trips through the tiers; find_by_user isolates."""

    async def test_save_and_get_round_trip(self, applied_schema: tuple[str, str]) -> None:
        url, schema = applied_schema
        pool = await make_pool(url, schema)
        try:
            nats = _InMemoryNatsBus()
            coll, _l1 = build_collection(pool, nats)

            agent_id = uuid.uuid4()
            user_id = uuid.uuid4()
            row = _make_row(agent_id, user_id, salience=0.7, content="check the wake threads")
            entity = coll.create(row)
            await coll.save_entity(entity)

            # L3 row present with the mutable fields
            async with pool.acquire() as conn:
                db = await conn.fetchrow(
                    "SELECT status, content, salience FROM intentions WHERE agent_id = $1 AND intention_id = $2",
                    agent_id,
                    row["intention_id"],
                )
            assert db is not None
            assert db["status"] == "open"
            assert db["content"] == "check the wake threads"
            assert float(db["salience"]) == 0.7

            # get resolves by the composite pk tuple
            fetched = await coll.get((agent_id, row["intention_id"]))
            assert fetched is not None
            assert fetched.content == "check the wake threads"
            assert fetched.status == "open"
            assert fetched.user_id == user_id
        finally:
            await pool.close()

    async def test_find_by_user_isolates_and_ranks(self, applied_schema: tuple[str, str]) -> None:
        """find_by_user returns only the caller's wants, salience-ranked."""
        url, schema = applied_schema
        pool = await make_pool(url, schema)
        try:
            nats = _InMemoryNatsBus()
            coll, _l1 = build_collection(pool, nats)

            agent_id = uuid.uuid4()  # shared across users (metallm reality)
            user_a = uuid.uuid4()
            user_b = uuid.uuid4()

            a_low = _make_row(agent_id, user_a, salience=0.3, content="A: low")
            a_high = _make_row(agent_id, user_a, salience=0.9, content="A: high")
            b_only = _make_row(agent_id, user_b, salience=0.8, content="B: only")
            for row in (a_low, a_high, b_only):
                await coll.save_entity(coll.create(row))

            a_wants = await coll.find_by_user(user_a, agent_id=agent_id)
            # isolation: B's want never surfaces for A
            assert [w.content for w in a_wants] == ["A: high", "A: low"]
            assert all(w.user_id == user_a for w in a_wants)

            b_wants = await coll.find_by_user(user_b, agent_id=agent_id)
            assert [w.content for w in b_wants] == ["B: only"]
        finally:
            await pool.close()


#: the hand-written dedup read ``find_similar_for_dedup`` sent before it moved onto ``read_nearest`` (2026-11-14)
_OLD_DEDUP_SQL = (
    "SELECT intention_id, content, salience, "
    "1 - (embedding OPERATOR(public.<=>) $1::text::public.vector) AS similarity "
    "FROM intentions "
    "WHERE agent_id = $2 AND user_id = $3 AND embedding IS NOT NULL "
    "AND status = 'open' "
    "ORDER BY embedding OPERATOR(public.<=>) $1::text::public.vector "
    "LIMIT $4"
)


class TestFindSimilarForDedupSameResults:
    """``find_similar_for_dedup`` answers what its hand-written statement answered."""

    async def test_the_keyed_read_answers_the_old_statements_rows(self, applied_schema: tuple[str, str]) -> None:
        """Flips if the dedup read stops pinning the agent, the user or the open status, lets a row with no embedding
        in, orders other than nearest first, takes more than ``top_k``, or filters the threshold otherwise; checked
        against the old statement's own answer over the same rows."""
        url, schema = applied_schema
        pool = await make_pool(url, schema)
        try:
            async with pool.acquire() as conn:
                width = await conn.fetchval(
                    "SELECT atttypmod FROM pg_attribute WHERE attrelid = 'intentions'::regclass AND attname = 'embedding'"
                )

            def vector(lean: float) -> str:
                return "[" + ",".join(["1", str(lean), *["0"] * (width - 2)]) + "]"

            agent_id, user_id = uuid.uuid4(), uuid.uuid4()
            seeds: list[tuple[uuid.UUID, uuid.UUID, str, str | None]] = [
                (agent_id, user_id, "open", vector(0.1 * i)) for i in range(8)
            ]
            seeds += [
                (agent_id, user_id, "granted", vector(0.01)),  # another status, nearer than any open row
                (agent_id, user_id, "open", None),  # no embedding
                (agent_id, uuid.uuid4(), "open", vector(0.02)),  # another user, same agent
                (uuid.uuid4(), user_id, "open", vector(0.03)),  # another agent, same user
            ]
            now = datetime.now(UTC)
            async with pool.acquire() as conn:
                for seed_agent, seed_user, status, embedding in seeds:
                    await conn.execute(
                        "INSERT INTO intentions (intention_id, agent_id, customer_id, user_id, status, content, "
                        "embedding, salience, date_created, date_updated) "
                        "VALUES ($1, $2, $3, $4, $5::intention_status, $6, $7::text::public.vector, 0.5, $8, $8)",
                        uuid.uuid4(),
                        seed_agent,
                        uuid.uuid4(),
                        seed_user,
                        status,
                        f"{status} {embedding is not None}",
                        embedding,
                        now,
                    )
            coll, _l1 = build_collection(pool, _InMemoryNatsBus())
            query = [1.0, 0.0, *[0.0] * (width - 2)]
            for top_k, threshold in ((4, 0.0), (8, 0.9), (20, 0.0), (0, 0.0)):
                async with pool.acquire() as conn:
                    old = await conn.fetch(_OLD_DEDUP_SQL, json.dumps(query), agent_id, user_id, top_k)
                expected = [
                    {
                        "intention_id": row["intention_id"],
                        "content": row["content"],
                        "salience": float(row["salience"]),
                        "similarity": float(row["similarity"]),
                    }
                    for row in old
                    if float(row["similarity"]) >= threshold
                ]
                found = await coll.find_similar_for_dedup(
                    user_id=user_id, agent_id=agent_id, embedding=query, top_k=top_k, threshold=threshold
                )
                assert found == expected
                assert len(found) == {(4, 0.0): 4, (8, 0.9): 5, (20, 0.0): 8, (0, 0.0): 0}[(top_k, threshold)]
        finally:
            await pool.close()
