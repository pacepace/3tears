"""integration: the three intention tools + decay, against real pgvector.

Exercises the B2 surface end-to-end through the three-tier collection on
a pgvector/pg16 container:

- ``intention_log`` creates an ``open`` want, dedups a near-duplicate
  (refresh, not duplicate), and creates a second row for a distinct want;
- ``intention_list`` returns ``open`` wants outside the surfacing cooldown,
  salience-ranked, isolated by ``user_id``;
- ``intention_mark_surfaced`` stamps the cooldown clock, enforces
  ownership (a foreign user's want reads back not-found), and emits the
  presence event (surfaced on ``asked``, resolved on ``granted`` /
  ``dropped``);
- ``IntentionsCollection.decay_salience`` sinks an abandoned want.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import asyncpg
import pytest

from threetears.agent.acl import AclCache

from threetears.agent.intention.authorize import IntentionAuthorizerDependencies
from threetears.agent.intention.events import (
    IntentionResolvedEvent,
    IntentionSurfacedEvent,
)
from threetears.agent.intention.tools import (
    load_intention_list_tool,
    load_intention_log_tool,
    load_intention_mark_surfaced_tool,
)

from .conftest import apply_migrations, build_collection, make_pool
from .conftest import InMemoryNatsBus as _InMemoryNatsBus


pytestmark = pytest.mark.integration

_DIM = 1024


class _OneHotEmbedder:
    """deterministic embedder: identical text -> identical unit vector.

    Maps each distinct text to a one-hot 1024-vector at ``hash % 1024``.
    Identical strings collide on the same axis (cosine 1.0 >= dedup
    threshold); different strings land on different axes (cosine 0.0),
    so dedup fires exactly when two logs carry the same want text.
    """

    async def aembed_query(self, text: str) -> list[float]:
        vec = [0.0] * _DIM
        vec[hash(text) % _DIM] = 1.0
        return vec


def _authorizer() -> IntentionAuthorizerDependencies:
    """owner-only authorizer with empty ACL loaders (short-circuit path)."""

    class _EmptyMembershipLoader:
        async def load_for_user(self, user_id: uuid.UUID) -> tuple[Any, ...]:
            _ = user_id
            return ()

        async def load_for_agent(self, agent_id: uuid.UUID) -> tuple[Any, ...]:
            _ = agent_id
            return ()

    class _EmptyGrantLoader:
        async def load_assignments_for_groups(self, group_ids: Any, namespace: Any) -> tuple[Any, ...]:
            _ = group_ids, namespace
            return ()

        async def load_roles(self, role_ids: Any) -> dict[Any, Any]:
            _ = role_ids
            return {}

        async def load_groups(self, group_ids: Any) -> dict[Any, Any]:
            _ = group_ids
            return {}

    return IntentionAuthorizerDependencies(
        acl_cache=AclCache(
            membership_loader=_EmptyMembershipLoader(),
            grant_loader=_EmptyGrantLoader(),
        ),
    )


async def _count_open(pool: asyncpg.Pool, agent_id: uuid.UUID, user_id: uuid.UUID) -> int:
    async with pool.acquire() as conn:
        n = await conn.fetchval(
            "SELECT count(*) FROM intentions WHERE agent_id = $1 AND user_id = $2 AND status = 'open'",
            agent_id,
            user_id,
        )
    return int(n)


class TestIntentionLog:
    async def test_log_creates_open_want(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        await apply_migrations(url, schema)
        pool = await make_pool(url, schema)
        try:
            coll, _l1 = build_collection(pool, _InMemoryNatsBus())
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            (log_tool,) = await load_intention_log_tool(
                user_id,
                _OneHotEmbedder(),
                agent_id,
                customer_id,
                _authorizer(),
                coll,  # type: ignore[arg-type]
            )
            out = await log_tool.ainvoke({"content": "learn what the user is building"})
            assert "[intention:" in out
            assert await _count_open(pool, agent_id, user_id) == 1
        finally:
            await pool.close()

    async def test_log_dedups_refreshes_not_duplicates(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        await apply_migrations(url, schema)
        pool = await make_pool(url, schema)
        try:
            coll, _l1 = build_collection(pool, _InMemoryNatsBus())
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            (log_tool,) = await load_intention_log_tool(
                user_id,
                _OneHotEmbedder(),
                agent_id,
                customer_id,
                _authorizer(),
                coll,  # type: ignore[arg-type]
                salience_seed=0.5,
                near_dup_bump=0.05,
            )
            first = await log_tool.ainvoke({"content": "remember to ask about the trip"})
            second = await log_tool.ainvoke({"content": "remember to ask about the trip"})
            assert "Logged as" in first
            assert "Refreshed existing" in second
            # exactly one row, salience bumped 0.5 -> 0.55
            assert await _count_open(pool, agent_id, user_id) == 1
            async with pool.acquire() as conn:
                sal = await conn.fetchval(
                    "SELECT salience FROM intentions WHERE agent_id = $1 AND user_id = $2",
                    agent_id,
                    user_id,
                )
            assert float(sal) == pytest.approx(0.55, abs=1e-6)
        finally:
            await pool.close()

    async def test_log_distinct_creates_second(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        await apply_migrations(url, schema)
        pool = await make_pool(url, schema)
        try:
            coll, _l1 = build_collection(pool, _InMemoryNatsBus())
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            (log_tool,) = await load_intention_log_tool(
                user_id,
                _OneHotEmbedder(),
                agent_id,
                customer_id,
                _authorizer(),
                coll,  # type: ignore[arg-type]
            )
            await log_tool.ainvoke({"content": "want A"})
            await log_tool.ainvoke({"content": "a totally unrelated want B"})
            assert await _count_open(pool, agent_id, user_id) == 2
        finally:
            await pool.close()

    async def test_log_refresh_save_failure_soft_fails(self, pg_schema: tuple[str, str]) -> None:
        """a save failure on the dedup-refresh path returns the soft-fail string, not a raise."""
        url, schema = pg_schema
        await apply_migrations(url, schema)
        pool = await make_pool(url, schema)
        try:
            coll, _l1 = build_collection(pool, _InMemoryNatsBus())
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            (log_tool,) = await load_intention_log_tool(
                user_id,
                _OneHotEmbedder(),
                agent_id,
                customer_id,
                _authorizer(),
                coll,  # type: ignore[arg-type]
            )
            await log_tool.ainvoke({"content": "same want, will re-log"})

            # force the refresh save to fail (simulates a CAS conflict / L3 error)
            async def _boom(entity: Any) -> None:
                raise RuntimeError("simulated CAS conflict")

            coll.save_entity = _boom  # type: ignore[method-assign]
            out = await log_tool.ainvoke({"content": "same want, will re-log"})
            assert "[TOOL ERROR] intention_log: refresh failed" in out
            assert "simulated CAS conflict" in out
        finally:
            await pool.close()

    async def test_log_stamps_source_conversation_from_resolver(self, pg_schema: tuple[str, str]) -> None:
        """a context_resolver's conversation_id lands on source_conversation_id."""
        url, schema = pg_schema
        await apply_migrations(url, schema)
        pool = await make_pool(url, schema)
        try:
            coll, _l1 = build_collection(pool, _InMemoryNatsBus())
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            conv_id = uuid.uuid4()

            class _Ctx:
                conversation_id = conv_id

            (log_tool,) = await load_intention_log_tool(
                user_id,
                _OneHotEmbedder(),
                agent_id,
                customer_id,
                _authorizer(),
                coll,  # type: ignore[arg-type]
                context_resolver=lambda: _Ctx(),
            )
            out = await log_tool.ainvoke({"content": "a want tied to this chat"})
            intention_id = out.split("[intention:")[1].split("]")[0]
            async with pool.acquire() as conn:
                stored = await conn.fetchval(
                    "SELECT source_conversation_id FROM intentions WHERE agent_id = $1 AND intention_id = $2",
                    agent_id,
                    uuid.UUID(intention_id),
                )
            assert stored == conv_id
        finally:
            await pool.close()


class TestIntentionList:
    async def test_list_open_outside_cooldown_ranked_isolated(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        await apply_migrations(url, schema)
        pool = await make_pool(url, schema)
        try:
            coll, _l1 = build_collection(pool, _InMemoryNatsBus())
            agent_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            user_a, user_b = uuid.uuid4(), uuid.uuid4()
            now = datetime.now(UTC)

            async def _insert(
                user_id: uuid.UUID, content: str, salience: float, last_surfaced: datetime | None
            ) -> None:
                async with pool.acquire() as conn:
                    await conn.execute(
                        "INSERT INTO intentions (intention_id, agent_id, customer_id, user_id, status, "
                        "content, salience, last_surfaced_at, date_created, date_updated) "
                        "VALUES ($1,$2,$3,$4,'open',$5,$6,$7,now(),now())",
                        uuid.uuid4(),
                        agent_id,
                        customer_id,
                        user_id,
                        content,
                        salience,
                        last_surfaced,
                    )

            await _insert(user_a, "A high", 0.9, None)
            await _insert(user_a, "A low", 0.3, None)
            await _insert(user_a, "A recently surfaced", 0.95, now)  # inside cooldown -> excluded
            await _insert(user_b, "B only", 0.8, None)

            (list_tool,) = await load_intention_list_tool(
                user_a, agent_id, customer_id, _authorizer(), coll, cooldown_days=7
            )
            out = await list_tool.ainvoke({})
            # ranked, isolated, cooldown-filtered
            assert out.index("A high") < out.index("A low")
            assert "A recently surfaced" not in out
            assert "B only" not in out
        finally:
            await pool.close()

    async def test_list_empty_when_all_in_cooldown(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        await apply_migrations(url, schema)
        pool = await make_pool(url, schema)
        try:
            coll, _l1 = build_collection(pool, _InMemoryNatsBus())
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO intentions (intention_id, agent_id, customer_id, user_id, status, "
                    "content, salience, last_surfaced_at, date_created, date_updated) "
                    "VALUES ($1,$2,$3,$4,'open','fresh',0.9,now(),now(),now())",
                    uuid.uuid4(),
                    agent_id,
                    customer_id,
                    user_id,
                )
            (list_tool,) = await load_intention_list_tool(
                user_id, agent_id, customer_id, _authorizer(), coll, cooldown_days=7
            )
            out = await list_tool.ainvoke({})
            assert "No open wants" in out
        finally:
            await pool.close()


class TestIntentionMarkSurfaced:
    async def test_mark_asked_stamps_clock_and_emits_surfaced(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        await apply_migrations(url, schema)
        pool = await make_pool(url, schema)
        try:
            coll, _l1 = build_collection(pool, _InMemoryNatsBus())
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            (log_tool,) = await load_intention_log_tool(
                user_id,
                _OneHotEmbedder(),
                agent_id,
                customer_id,
                _authorizer(),
                coll,  # type: ignore[arg-type]
            )
            out = await log_tool.ainvoke({"content": "surface this one"})
            intention_id = out.split("[intention:")[1].split("]")[0]

            (mark_tool,) = await load_intention_mark_surfaced_tool(user_id, agent_id, customer_id, _authorizer(), coll)
            captured: list[Any] = []

            async def _capture(event: Any, *, config: Any = None) -> None:
                captured.append(event)

            with patch("threetears.agent.intention.tools.dispatch_event", new=_capture):
                res = await mark_tool.ainvoke({"intention_id": intention_id, "new_status": "asked"})
            assert "asked" in res

            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT status, last_surfaced_at FROM intentions WHERE agent_id = $1 AND intention_id = $2",
                    agent_id,
                    uuid.UUID(intention_id),
                )
            assert row["status"] == "asked"
            assert row["last_surfaced_at"] is not None
            assert len(captured) == 1
            assert isinstance(captured[0], IntentionSurfacedEvent)
            assert captured[0].user_id == str(user_id)

        finally:
            await pool.close()

    async def test_mark_granted_emits_resolved(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        await apply_migrations(url, schema)
        pool = await make_pool(url, schema)
        try:
            coll, _l1 = build_collection(pool, _InMemoryNatsBus())
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            (log_tool,) = await load_intention_log_tool(
                user_id,
                _OneHotEmbedder(),
                agent_id,
                customer_id,
                _authorizer(),
                coll,  # type: ignore[arg-type]
            )
            out = await log_tool.ainvoke({"content": "resolve this one"})
            intention_id = out.split("[intention:")[1].split("]")[0]
            (mark_tool,) = await load_intention_mark_surfaced_tool(user_id, agent_id, customer_id, _authorizer(), coll)
            captured: list[Any] = []

            async def _capture(event: Any, *, config: Any = None) -> None:
                captured.append(event)

            with patch("threetears.agent.intention.tools.dispatch_event", new=_capture):
                await mark_tool.ainvoke({"intention_id": intention_id, "new_status": "granted"})
            assert isinstance(captured[0], IntentionResolvedEvent)
            assert captured[0].new_status == "granted"
        finally:
            await pool.close()

    async def test_mark_foreign_want_not_found(self, pg_schema: tuple[str, str]) -> None:
        """a want owned by another user reads back not-found (isolation)."""
        url, schema = pg_schema
        await apply_migrations(url, schema)
        pool = await make_pool(url, schema)
        try:
            coll, _l1 = build_collection(pool, _InMemoryNatsBus())
            agent_id, customer_id = uuid.uuid4(), uuid.uuid4()
            owner, intruder = uuid.uuid4(), uuid.uuid4()
            (owner_log,) = await load_intention_log_tool(
                owner,
                _OneHotEmbedder(),
                agent_id,
                customer_id,
                _authorizer(),
                coll,  # type: ignore[arg-type]
            )
            out = await owner_log.ainvoke({"content": "owner's private want"})
            intention_id = out.split("[intention:")[1].split("]")[0]

            # a tool minted for the intruder must not be able to touch it
            (intruder_mark,) = await load_intention_mark_surfaced_tool(
                intruder, agent_id, customer_id, _authorizer(), coll
            )
            res = await intruder_mark.ainvoke({"intention_id": intention_id, "new_status": "dropped"})
            assert "No want found" in res
            # the owner's row is untouched (still open)
            async with pool.acquire() as conn:
                status = await conn.fetchval(
                    "SELECT status FROM intentions WHERE agent_id = $1 AND intention_id = $2",
                    agent_id,
                    uuid.UUID(intention_id),
                )
            assert status == "open"
        finally:
            await pool.close()

    async def test_invalid_status_rejected(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        await apply_migrations(url, schema)
        pool = await make_pool(url, schema)
        try:
            coll, _l1 = build_collection(pool, _InMemoryNatsBus())
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            (mark_tool,) = await load_intention_mark_surfaced_tool(user_id, agent_id, customer_id, _authorizer(), coll)
            res = await mark_tool.ainvoke({"intention_id": str(uuid.uuid4()), "new_status": "open"})
            assert "invalid new_status" in res
        finally:
            await pool.close()


class TestIntentionDecay:
    async def test_decay_sinks_abandoned_want(self, pg_schema: tuple[str, str]) -> None:
        """decay_salience ages a want with an old last_decayed_at toward the floor."""
        url, schema = pg_schema
        await apply_migrations(url, schema)
        pool = await make_pool(url, schema)
        try:
            coll, _l1 = build_collection(pool, _InMemoryNatsBus())
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            intention_id = uuid.uuid4()
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO intentions (intention_id, agent_id, customer_id, user_id, status, "
                    "content, salience, last_decayed_at, date_created, date_updated) "
                    "VALUES ($1,$2,$3,$4,'open','abandoned',0.9, now() - interval '120 days', now(), now())",
                    intention_id,
                    agent_id,
                    customer_id,
                    user_id,
                )
                # a 60-day half-life over 120 days -> ~1/4 of 0.9 = ~0.225
                decayed = await coll.decay_salience(half_life_days=60.0, floor=0.1)
                assert decayed >= 1
                sal = await conn.fetchval(
                    "SELECT salience FROM intentions WHERE agent_id = $1 AND intention_id = $2",
                    agent_id,
                    intention_id,
                )
            assert float(sal) < 0.9
            assert float(sal) >= 0.1  # never below the floor
        finally:
            await pool.close()


class TestIntentionCacheCoherence:
    """review R2 fix: salience + last_decayed_at are immutable to the entity
    UPDATE, so a mark_surfaced / refresh save can't revert a scheduled decay
    (matching memory's guard)."""

    async def test_decay_survives_stale_entity_save(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        await apply_migrations(url, schema)
        pool = await make_pool(url, schema)
        try:
            coll, _l1 = build_collection(pool, _InMemoryNatsBus())
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            wid = uuid.uuid4()
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO intentions (intention_id, agent_id, customer_id, user_id, status, "
                    "content, salience, last_decayed_at, date_created, date_updated) "
                    "VALUES ($1,$2,$3,$4,'open','a want',0.9, now() - interval '60 days', now(), now())",
                    wid,
                    agent_id,
                    customer_id,
                    user_id,
                )
            # hold a stale entity carrying the pre-decay salience + old anchor
            entity = await coll.get((agent_id, wid))
            assert entity is not None
            assert float(entity.salience) == 0.9

            # decay L3 DIRECTLY (raw) so the held entity + its L1 stay at the
            # pre-decay values -- reproduces the multi-pod window where a pod
            # holds a stale entity while another pod's decay has committed.
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE intentions SET salience = 0.45, last_decayed_at = now() WHERE intention_id = $1",
                    wid,
                )

            # a full-entity save from the stale entity (mark it surfaced) must
            # NOT revert salience / last_decayed_at -- both are immutable to
            # the entity-UPDATE generator.
            entity.status = "asked"
            entity.last_surfaced_at = datetime.now(UTC)
            await coll.save_entity(entity)

            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT salience, status, last_decayed_at FROM intentions WHERE intention_id = $1",
                    wid,
                )
            assert abs(float(row["salience"]) - 0.45) < 1e-6  # decay NOT reverted to 0.9
            assert row["status"] == "asked"  # the status update still landed
            # the decay anchor was NOT rolled back to the 60-days-ago value
            assert (datetime.now(UTC) - row["last_decayed_at"]) < timedelta(hours=1)
        finally:
            await pool.close()

    async def test_decay_and_bump_invalidate_cache(self, pg_schema: tuple[str, str]) -> None:
        """decay_salience + bump_salience each invalidate the affected pks (defense B)."""
        url, schema = pg_schema
        await apply_migrations(url, schema)
        pool = await make_pool(url, schema)
        try:
            coll, _l1 = build_collection(pool, _InMemoryNatsBus())
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            wid = uuid.uuid4()
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO intentions (intention_id, agent_id, customer_id, user_id, status, "
                    "content, salience, last_decayed_at, date_created, date_updated) "
                    "VALUES ($1,$2,$3,$4,'open','a want',0.9, now() - interval '60 days', now(), now())",
                    wid,
                    agent_id,
                    customer_id,
                    user_id,
                )
            calls: list[Any] = []
            original = coll.invalidate_cache

            async def _spy(entity_id: Any) -> None:
                calls.append(entity_id)
                await original(entity_id)

            coll.invalidate_cache = _spy  # type: ignore[method-assign]

            await coll.decay_salience(half_life_days=60.0, floor=0.1)
            assert (agent_id, wid) in calls

            calls.clear()
            await coll.bump_salience([wid], agent_id=agent_id, access_bump=0.05)
            assert (agent_id, wid) in calls
        finally:
            await pool.close()

    async def test_user_id_is_immutable_across_saves(self, pg_schema: tuple[str, str]) -> None:
        """user_id (the sole isolation boundary) can't be moved by an entity save."""
        url, schema = pg_schema
        await apply_migrations(url, schema)
        pool = await make_pool(url, schema)
        try:
            coll, _l1 = build_collection(pool, _InMemoryNatsBus())
            agent_id, customer_id = uuid.uuid4(), uuid.uuid4()
            owner, other = uuid.uuid4(), uuid.uuid4()
            wid = uuid.uuid4()
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO intentions (intention_id, agent_id, customer_id, user_id, status, "
                    "content, salience, date_created, date_updated) "
                    "VALUES ($1,$2,$3,$4,'open','a want',0.5, now(), now())",
                    wid,
                    agent_id,
                    customer_id,
                    owner,
                )
            entity = await coll.get((agent_id, wid))
            assert entity is not None
            # a stray attempt to move the want to another user must NOT persist
            entity.user_id = other
            entity.status = "asked"
            await coll.save_entity(entity)
            async with pool.acquire() as conn:
                db_user = await conn.fetchval("SELECT user_id FROM intentions WHERE intention_id = $1", wid)
            assert db_user == owner  # isolation boundary held; only status changed
        finally:
            await pool.close()
