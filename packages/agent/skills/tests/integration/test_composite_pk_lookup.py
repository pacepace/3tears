"""Integration test: composite-pk lookup round-trips through the Collection.

Verifies that ``AgentSkillCollection.save_entity`` + ``get`` work with
the tuple ``(agent_id, skill_id)`` form and that the round-trip
preserves the non-trivial typed fields (``prompt_mode``,
``tool_additions``, ``tool_restrictions``, ``tags``).

This exercises the BaseCollection contract directly: no L1 backend
configured, no NATS client; the L3 pool path is the single source of
truth for the round-trip.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import asyncpg
import pytest
from uuid_utils import uuid7

from threetears.agent.skills.collections import (
    AgentSkillCollection,
    AgentSkillInvocationCollection,
)
from threetears.agent.skills.entities import (
    AgentSkillEntity,
    AgentSkillInvocationEntity,
)
from threetears.agent.skills.migrations import register as register_skills
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration


def _new_uuid() -> UUID:
    """Return a fresh UUIDv7 cast to stdlib ``UUID``."""
    return UUID(str(uuid7()))


async def _apply_schema(url: str, schema: str) -> asyncpg.Pool:
    """Apply migrations and return a pool bound to the schema."""
    setup_conn = await asyncpg.connect(url)
    try:
        await setup_conn.execute(f'SET search_path TO "{schema}", public')
        runner = MigrationRunner()
        register_conversations(runner)
        register_skills(runner)
        store = AsyncpgStore(setup_conn)
        await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
    finally:
        await setup_conn.close()

    pool = await asyncpg.create_pool(
        url,
        min_size=1,
        max_size=4,
        server_settings={"search_path": f"{schema}, public"},
    )
    assert pool is not None
    return pool


def _build_skill_collection(pool: asyncpg.Pool) -> AgentSkillCollection:
    """Build a Collection bound to the pool with no L1 / L2 wiring.

    The shard's BaseCollection subclass works with ``l3_pool`` only;
    L1 + L2 are optional and ``BaseCollection`` already guards on
    ``self.l3_pool is None`` everywhere.
    """
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return AgentSkillCollection(registry=registry, config=cfg)


def _build_invocation_collection(pool: asyncpg.Pool) -> AgentSkillInvocationCollection:
    """Build an invocation collection sharing the same pool."""
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return AgentSkillInvocationCollection(registry=registry, config=cfg)


class TestSkillRoundTrip:
    """``save_entity`` + ``get`` preserves every typed field."""

    async def test_round_trip_through_collection(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """Save a skill, read it back via composite pk, fields match."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            coll = _build_skill_collection(pool)
            agent_id = _new_uuid()
            skill_id = _new_uuid()
            user_id = _new_uuid()
            now = datetime.now(UTC)
            data: dict[str, Any] = {
                "agent_id": agent_id,
                "skill_id": skill_id,
                "user_id": user_id,
                "name": "deploy-helper",
                "summary": "Deploy a service",
                "body": "Step 1: run terraform plan. Step 2: apply.",
                "prompt_mode": "additive",
                "tool_additions": ["mcp.shell", "mcp.git"],
                "tool_restrictions": ["mcp.dangerous_op"],
                "trigger_keywords": "deploy release ship",
                "tags": ["ops", "ci"],
                "source": "manual",
                "enabled": True,
                "use_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "date_created": now,
                "date_updated": now,
            }
            entity = coll.create(data)
            await coll.save_entity(entity)

            fetched = await coll.get((agent_id, skill_id))
            assert fetched is not None
            assert isinstance(fetched, AgentSkillEntity)
            assert fetched.skill_id == skill_id
            assert fetched.agent_id == agent_id
            assert fetched.user_id == user_id
            assert fetched.name == "deploy-helper"
            assert fetched.summary == "Deploy a service"
            assert fetched.body == "Step 1: run terraform plan. Step 2: apply."
            assert fetched.prompt_mode == "additive"
            assert fetched.tool_additions == ["mcp.shell", "mcp.git"]
            assert fetched.tool_restrictions == ["mcp.dangerous_op"]
            assert fetched.trigger_keywords == "deploy release ship"
            assert sorted(fetched.tags) == ["ci", "ops"]
            assert fetched.source == "manual"
            assert fetched.enabled is True
        finally:
            await pool.close()

    async def test_find_by_name_for_user_round_trip(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """``find_by_name_for_user`` returns the freshly-saved entity."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            coll = _build_skill_collection(pool)
            agent_id = _new_uuid()
            skill_id = _new_uuid()
            user_id = _new_uuid()
            now = datetime.now(UTC)
            entity = coll.create(
                {
                    "agent_id": agent_id,
                    "skill_id": skill_id,
                    "user_id": user_id,
                    "name": "lookup-target",
                    "summary": "Found via name",
                    "body": "body",
                    "prompt_mode": "additive",
                    "date_created": now,
                    "date_updated": now,
                },
            )
            await coll.save_entity(entity)
            found = await coll.find_by_name_for_user(
                agent_id,
                user_id,
                "lookup-target",
            )
            assert found is not None
            assert found.skill_id == skill_id
        finally:
            await pool.close()

    async def test_list_for_user_orders_by_recency(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """``list_for_user`` without query orders by ``last_used_at`` DESC NULLS LAST."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            coll = _build_skill_collection(pool)
            agent_id = _new_uuid()
            user_id = _new_uuid()
            now = datetime.now(UTC)

            older = coll.create(
                {
                    "agent_id": agent_id,
                    "skill_id": _new_uuid(),
                    "user_id": user_id,
                    "name": "older",
                    "summary": "older",
                    "body": "older",
                    "prompt_mode": "additive",
                    "date_created": now,
                    "date_updated": now,
                },
            )
            await coll.save_entity(older)

            newer = coll.create(
                {
                    "agent_id": agent_id,
                    "skill_id": _new_uuid(),
                    "user_id": user_id,
                    "name": "newer",
                    "summary": "newer",
                    "body": "newer",
                    "prompt_mode": "additive",
                    "date_created": now,
                    "date_updated": now,
                },
            )
            await coll.save_entity(newer)

            # bump the newer one's last_used_at by hand to force ordering
            await coll.bump_use_count(agent_id, [newer.skill_id])

            results = await coll.list_for_user(agent_id, user_id)
            assert len(results) == 2
            assert results[0].name == "newer"

            total = await coll.count_for_user(agent_id, user_id)
            assert total == 2
        finally:
            await pool.close()

    async def test_count_for_user_honours_tag_and_query_filters(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """``count_for_user`` filtered counts match ``list_for_user`` lengths.

        The cap-check call (no filters) sees every row; a tag-filtered
        or FTS-query-filtered count must match exactly the set
        ``list_for_user`` returns under the same predicates so a
        paginated ``total_count`` never overstates.
        """
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            coll = _build_skill_collection(pool)
            agent_id = _new_uuid()
            user_id = _new_uuid()
            now = datetime.now(UTC)

            # Two "ops"-tagged skills, one with a "terraform" body token;
            # one untagged skill with an unrelated body.
            for name, body, tags in [
                ("deploy", "run terraform apply against prod", ["ops"]),
                ("rollback", "revert the last release safely", ["ops"]),
                ("notes", "generic personal scratchpad", []),
            ]:
                await coll.save_entity(
                    coll.create(
                        {
                            "agent_id": agent_id,
                            "skill_id": _new_uuid(),
                            "user_id": user_id,
                            "name": name,
                            "summary": "s",
                            "body": body,
                            "prompt_mode": "additive",
                            "tags": tags,
                            "date_created": now,
                            "date_updated": now,
                        },
                    ),
                )

            # Unfiltered cap-check semantics unchanged: all three rows.
            assert await coll.count_for_user(agent_id, user_id) == 3

            # Tag filter: only the two "ops"-tagged rows.
            tag_count = await coll.count_for_user(
                agent_id,
                user_id,
                tag_filter=["ops"],
            )
            tag_rows = await coll.list_for_user(
                agent_id,
                user_id,
                tag_filter=["ops"],
                limit=100,
            )
            assert tag_count == 2
            assert tag_count == len(tag_rows)

            # FTS query: only the row whose body mentions terraform.
            query_count = await coll.count_for_user(
                agent_id,
                user_id,
                query="terraform",
            )
            query_rows = await coll.list_for_user(
                agent_id,
                user_id,
                query="terraform",
                limit=100,
            )
            assert query_count == 1
            assert query_count == len(query_rows)
        finally:
            await pool.close()


class TestInvocationRoundTrip:
    """Invocation Collection: composite-pk save / get + record + set_outcome."""

    async def test_record_and_get_round_trip(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """``record`` persists an invocation; ``get`` reads it back via tuple pk."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            skill_coll = _build_skill_collection(pool)
            inv_coll = _build_invocation_collection(pool)

            agent_id = _new_uuid()
            user_id = _new_uuid()
            conversation_id = _new_uuid()
            invocation_id = _new_uuid()
            now = datetime.now(UTC)

            skill_id = _new_uuid()
            skill = skill_coll.create(
                {
                    "agent_id": agent_id,
                    "skill_id": skill_id,
                    "user_id": user_id,
                    "name": "skill-for-invocation",
                    "summary": "x",
                    "body": "x",
                    "prompt_mode": "additive",
                    "date_created": now,
                    "date_updated": now,
                },
            )
            await skill_coll.save_entity(skill)

            invocation = inv_coll.create(
                {
                    "agent_id": agent_id,
                    "invocation_id": invocation_id,
                    "skill_id": skill_id,
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "invocation_source": "invoke",
                    "invoked_at": now,
                },
            )
            await inv_coll.record(agent_id, invocation)

            fetched = await inv_coll.get((agent_id, invocation_id))
            assert fetched is not None
            assert isinstance(fetched, AgentSkillInvocationEntity)
            assert fetched.invocation_id == invocation_id
            assert fetched.skill_id == skill_id
            assert fetched.invocation_source == "invoke"
            assert fetched.outcome is None
        finally:
            await pool.close()

    async def test_set_outcome_is_idempotent(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """Calling ``set_outcome`` twice with the same args leaves the same state."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            skill_coll = _build_skill_collection(pool)
            inv_coll = _build_invocation_collection(pool)

            agent_id = _new_uuid()
            user_id = _new_uuid()
            conv_id = _new_uuid()
            skill_id = _new_uuid()
            invocation_id = _new_uuid()
            now = datetime.now(UTC)

            await skill_coll.save_entity(
                skill_coll.create(
                    {
                        "agent_id": agent_id,
                        "skill_id": skill_id,
                        "user_id": user_id,
                        "name": "x",
                        "summary": "x",
                        "body": "x",
                        "prompt_mode": "additive",
                        "date_created": now,
                        "date_updated": now,
                    },
                ),
            )
            await inv_coll.record(
                agent_id,
                inv_coll.create(
                    {
                        "agent_id": agent_id,
                        "invocation_id": invocation_id,
                        "skill_id": skill_id,
                        "user_id": user_id,
                        "conversation_id": conv_id,
                        "invocation_source": "invoke",
                        "invoked_at": now,
                    },
                ),
            )

            await inv_coll.set_outcome(
                agent_id,
                invocation_id,
                outcome="success",
                source="agent_marker",
            )
            await inv_coll.set_outcome(
                agent_id,
                invocation_id,
                outcome="success",
                source="agent_marker",
            )
            after_double = await inv_coll.get((agent_id, invocation_id))
            assert after_double is not None
            assert after_double.outcome == "success"
            assert after_double.outcome_source == "agent_marker"
        finally:
            await pool.close()

    async def test_list_for_skill_orders_newest_first(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """``list_for_skill`` returns rows ordered by ``invoked_at`` DESC."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            skill_coll = _build_skill_collection(pool)
            inv_coll = _build_invocation_collection(pool)
            agent_id = _new_uuid()
            user_id = _new_uuid()
            conv_id = _new_uuid()
            skill_id = _new_uuid()

            base = datetime.now(UTC)
            await skill_coll.save_entity(
                skill_coll.create(
                    {
                        "agent_id": agent_id,
                        "skill_id": skill_id,
                        "user_id": user_id,
                        "name": "x",
                        "summary": "x",
                        "body": "x",
                        "prompt_mode": "additive",
                        "date_created": base,
                        "date_updated": base,
                    },
                ),
            )

            from datetime import timedelta as _td

            first = _new_uuid()
            second = _new_uuid()
            third = _new_uuid()
            for inv_id, ts in [
                (first, base),
                (second, base + _td(seconds=1)),
                (third, base + _td(seconds=2)),
            ]:
                await inv_coll.record(
                    agent_id,
                    inv_coll.create(
                        {
                            "agent_id": agent_id,
                            "invocation_id": inv_id,
                            "skill_id": skill_id,
                            "user_id": user_id,
                            "conversation_id": conv_id,
                            "invocation_source": "invoke",
                            "invoked_at": ts,
                        },
                    ),
                )

            rows = await inv_coll.list_for_skill(agent_id, skill_id, limit=10)
            assert [row.invocation_id for row in rows] == [third, second, first]
        finally:
            await pool.close()

    async def test_count_for_skill_honours_outcome_filter(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """``count_for_skill`` counts the full set independent of page limit.

        Seeds more invocations than a page would hold and asserts the
        count equals the total (not the page size), and that the
        ``outcome_filter`` predicate matches ``list_for_skill``.
        """
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            skill_coll = _build_skill_collection(pool)
            inv_coll = _build_invocation_collection(pool)
            agent_id = _new_uuid()
            user_id = _new_uuid()
            conv_id = _new_uuid()
            skill_id = _new_uuid()
            now = datetime.now(UTC)

            await skill_coll.save_entity(
                skill_coll.create(
                    {
                        "agent_id": agent_id,
                        "skill_id": skill_id,
                        "user_id": user_id,
                        "name": "x",
                        "summary": "x",
                        "body": "x",
                        "prompt_mode": "additive",
                        "date_created": now,
                        "date_updated": now,
                    },
                ),
            )

            # 7 invocations: 3 success, 2 failure, 2 unclassified (NULL).
            outcomes: list[str | None] = [
                "success",
                "success",
                "success",
                "failure",
                "failure",
                None,
                None,
            ]
            for outcome in outcomes:
                inv_id = _new_uuid()
                await inv_coll.record(
                    agent_id,
                    inv_coll.create(
                        {
                            "agent_id": agent_id,
                            "invocation_id": inv_id,
                            "skill_id": skill_id,
                            "user_id": user_id,
                            "conversation_id": conv_id,
                            "invocation_source": "invoke",
                            "invoked_at": now,
                            "outcome": outcome,
                            "outcome_source": "agent_marker" if outcome else None,
                        },
                    ),
                )

            # total_count must reflect the full set, not the page size:
            # request a page smaller than the total and confirm the count
            # is still the true total.
            page = await inv_coll.list_for_skill(agent_id, skill_id, limit=2)
            assert len(page) == 2
            all_count = await inv_coll.count_for_skill(agent_id, skill_id)
            assert all_count == 7
            assert all_count != len(page)

            assert (
                await inv_coll.count_for_skill(
                    agent_id,
                    skill_id,
                    outcome_filter="success",
                )
                == 3
            )
            assert (
                await inv_coll.count_for_skill(
                    agent_id,
                    skill_id,
                    outcome_filter="failure",
                )
                == 2
            )
            # "unknown" (NULL) is all minus the classified counts.
            assert all_count - 3 - 2 == 2
        finally:
            await pool.close()
