"""End-to-end integration tests for the seven skill-tool factories.

Each test:

1. Spins up a fresh schema via the canonical ``pg_schema`` fixture.
2. Applies the conversations + skills migrations.
3. Constructs real :class:`AgentSkillCollection` /
   :class:`AgentSkillInvocationCollection` against the live pool.
4. Loads the tool factory and exercises its public surface via
   ``ainvoke``.

Covers the full happy-path lifecycle, cross-user isolation, ACL,
first-invoke-wins, and the 200-prose-skill cap. The
:class:`SkillRegistryClient` is replaced with a deterministic fake so
the tests stay decoupled from any ACL evaluator / tool registry
deployment.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

import asyncpg
import pytest
from uuid_utils import uuid7

from threetears.agent.skills.collections import (
    AgentSkillCollection,
    AgentSkillInvocationCollection,
)
from threetears.agent.skills.migrations import register as register_skills
from threetears.agent.skills.tools import (
    SkillEligibleTool,
    SkillRegistryClient,
    SkillToolIntrospect,
    load_skill_create_tool,
    load_skill_delete_tool,
    load_skill_get_tool,
    load_skill_introspect_tool,
    load_skill_invoke_tool,
    load_skill_list_tool,
    load_skill_update_tool,
)
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration


def _new_uuid() -> UUID:
    """Return a fresh UUIDv7 cast to stdlib ``UUID``."""
    return UUID(str(uuid7()))


class _FakeRegistry(SkillRegistryClient):
    """In-memory implementation used across the integration tests."""

    def __init__(
        self,
        *,
        permitted_tools: set[str] | None = None,
        skill_eligible: list[SkillEligibleTool] | None = None,
        introspect_payloads: dict[str, SkillToolIntrospect] | None = None,
    ) -> None:
        self._permitted = permitted_tools or set()
        self._skill_eligible = list(skill_eligible or [])
        self._introspect = dict(introspect_payloads or {})

    async def acl_permits(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        tool_name: str,
    ) -> bool:
        return tool_name in self._permitted

    async def list_skill_eligible_tools(
        self,
        *,
        actor_user_id: UUID,
        actor_agent_id: UUID,
    ) -> list[SkillEligibleTool]:
        return list(self._skill_eligible)

    async def get_tool_introspect(
        self,
        *,
        actor_user_id: UUID,
        actor_agent_id: UUID,
        mcp_name: str,
    ) -> SkillToolIntrospect | None:
        return self._introspect.get(mcp_name)


async def _apply(conn: asyncpg.Connection, schema: str) -> None:
    """Apply conversations + skills migrations to ``schema``."""
    await conn.execute(f'SET search_path TO "{schema}", public')
    runner = MigrationRunner()
    register_conversations(runner)
    register_skills(runner)
    store = AsyncpgStore(conn)
    await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]


def _build_collections(
    pool: asyncpg.Pool,
) -> tuple[AgentSkillCollection, AgentSkillInvocationCollection]:
    """Build skills + invocations Collections bound to ``pool``."""
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    config = DefaultCoreConfig(
        collection_flush="ALWAYS",
        collection_flush_tables="",
    )
    skills = AgentSkillCollection(
        registry=registry,
        config=config,
        nats_client=None,
    )
    invocations = AgentSkillInvocationCollection(
        registry=registry,
        config=config,
        nats_client=None,
    )
    return skills, invocations


def _build_collections_with_l1(
    pool: asyncpg.Pool,
) -> tuple[AgentSkillCollection, AgentSkillInvocationCollection]:
    """Build skills + invocations Collections backed by an L1 SQLite cache.

    The default :func:`_build_collections` wires L3 only; a consumer that
    runs with the real three-tier cache (e.g. a consumer app via ``main.py``)
    holds entities as L1 cache proxies. Tool code that reads entity
    fields AFTER a Collection mutation (``delete``) must snapshot them
    first, because ``delete`` evicts the L1 row and subsequent proxy
    reads return ``None``. This builder reproduces that path so the
    regression is caught here, not only downstream.
    """
    from sqlalchemy import MetaData
    from threetears.agent.skills.tables import (
        agent_skill_invocations_table,
        agent_skills_table,
    )
    from threetears.core.cache.sqlite import SQLiteBackend

    metadata = MetaData()
    agent_skills_table(metadata)
    agent_skill_invocations_table(metadata)
    l1 = SQLiteBackend(db_name=f"skills_cache_{uuid7()}")
    l1.initialize(metadata)

    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l3_pool=pool)
    config = DefaultCoreConfig(
        collection_flush="ALWAYS",
        collection_flush_tables="",
    )
    skills = AgentSkillCollection(
        registry=registry,
        config=config,
        nats_client=None,
    )
    invocations = AgentSkillInvocationCollection(
        registry=registry,
        config=config,
        nats_client=None,
    )
    return skills, invocations


@pytest.fixture
async def pool_with_schema(
    pg_schema: tuple[str, str],
) -> AsyncIterator[asyncpg.Pool]:
    """Yield an ``asyncpg.Pool`` pointed at a freshly-migrated schema."""
    url, schema = pg_schema
    # Apply migrations on a one-off connection first; the pool then
    # binds to the same schema via ``search_path``.
    conn = await asyncpg.connect(url)
    try:
        await _apply(conn, schema)
    finally:
        await conn.close()

    pool: asyncpg.Pool = await asyncpg.create_pool(
        url,
        min_size=1,
        max_size=2,
        server_settings={"search_path": f"{schema}, public"},
    )
    try:
        yield pool
    finally:
        await pool.close()


# --- skill_create + skill_get + skill_list + skill_update + skill_delete ---


class TestSkillLifecycle:
    """create -> list -> introspect -> update -> delete round-trip."""

    async def test_full_lifecycle(
        self,
        pool_with_schema: asyncpg.Pool,
    ) -> None:
        agent_id = _new_uuid()
        user_id = _new_uuid()
        skills, _ = _build_collections(pool_with_schema)
        registry = _FakeRegistry(permitted_tools={"mcp.shell"})

        [create_tool] = load_skill_create_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=skills,
            registry=registry,
        )
        [list_tool] = load_skill_list_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=skills,
            registry=registry,
        )
        [get_tool] = load_skill_get_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=skills,
        )
        [update_tool] = load_skill_update_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=skills,
            registry=registry,
        )
        [delete_tool] = load_skill_delete_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=skills,
        )

        # Create
        create_out = await create_tool.ainvoke(
            {
                "name": "deploy",
                "summary": "Deploy via helm",
                "body": "Run helm install",
                "tool_additions": ["mcp.shell"],
            }
        )
        assert create_out.startswith("[skill:")
        # Extract the skill_id
        skill_id_str = create_out.split("[skill:")[1].split("]")[0]
        skill_id = UUID(skill_id_str)

        # List should surface it
        list_out = await list_tool.ainvoke({})
        assert skill_id_str in list_out
        assert "deploy" in list_out

        # Get returns full body + minimal-token shape
        get_out = await get_tool.ainvoke({"skill_id": skill_id_str})
        assert "kind: prose" in get_out
        assert "Run helm install" in get_out
        assert "use_count" not in get_out  # minimal-token shape

        # Update summary; verify persisted
        update_out = await update_tool.ainvoke(
            {"skill_id": skill_id_str, "summary": "Deploy with helm chart"},
        )
        assert "[TOOL ERROR]" not in update_out
        get_out2 = await get_tool.ainvoke({"skill_id": skill_id_str})
        assert "Deploy with helm chart" in get_out2

        # Delete
        del_out = await delete_tool.ainvoke({"skill_id": skill_id_str})
        assert del_out.startswith("Deleted")
        list_out2 = await list_tool.ainvoke({})
        assert skill_id_str not in list_out2

    async def test_at_least_one_payload_enforced_at_db_level(
        self,
        pool_with_schema: asyncpg.Pool,
    ) -> None:
        """The tool's check fires before the DB CHECK constraint can."""
        agent_id = _new_uuid()
        user_id = _new_uuid()
        skills, _ = _build_collections(pool_with_schema)
        registry = _FakeRegistry()
        [create_tool] = load_skill_create_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=skills,
            registry=registry,
        )
        out = await create_tool.ainvoke(
            {"name": "empty", "summary": "nothing"},
        )
        assert "[TOOL ERROR]" in out

    async def test_delete_under_l1_cache_returns_success_message(
        self,
        pool_with_schema: asyncpg.Pool,
    ) -> None:
        """skill_delete must not read evicted entity fields post-delete.

        Regression: with a live L1 cache (the production three-tier
        path), the loaded entity is a cache proxy. ``delete`` evicts the
        L1 row, so reading ``entity.name`` / ``entity.skill_id`` AFTER
        the delete returns ``None`` and ``[skill:None]`` raises in the
        UUID coercion. The tool snapshots both fields before delete.
        """
        agent_id = _new_uuid()
        user_id = _new_uuid()
        skills, _ = _build_collections_with_l1(pool_with_schema)
        registry = _FakeRegistry()
        [create_tool] = load_skill_create_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=skills,
            registry=registry,
        )
        [delete_tool] = load_skill_delete_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=skills,
        )

        create_out = await create_tool.ainvoke(
            {"name": "ephemeral", "summary": "to be deleted", "body": "x"},
        )
        skill_id_str = create_out.split("[skill:")[1].split("]")[0]

        del_out = await delete_tool.ainvoke({"skill_id": skill_id_str})
        # The success message renders the snapshotted name + id; no crash.
        assert del_out.startswith("Deleted skill 'ephemeral'")
        assert skill_id_str in del_out


class TestCrossUserIsolation:
    """A skill belonging to one user must never surface for another."""

    async def test_user_b_cannot_get_user_a_skill(
        self,
        pool_with_schema: asyncpg.Pool,
    ) -> None:
        agent_id = _new_uuid()
        user_a = _new_uuid()
        user_b = _new_uuid()
        skills, _ = _build_collections(pool_with_schema)
        registry = _FakeRegistry()

        [create_a] = load_skill_create_tool(
            agent_id=agent_id,
            user_id=user_a,
            skills_collection=skills,
            registry=registry,
        )
        [get_b] = load_skill_get_tool(
            agent_id=agent_id,
            user_id=user_b,
            skills_collection=skills,
        )
        [list_b] = load_skill_list_tool(
            agent_id=agent_id,
            user_id=user_b,
            skills_collection=skills,
            registry=registry,
        )
        out = await create_a.ainvoke(
            {"name": "secret", "summary": "private", "body": "private body"},
        )
        skill_id_str = out.split("[skill:")[1].split("]")[0]

        list_b_out = await list_b.ainvoke({})
        assert skill_id_str not in list_b_out
        assert "secret" not in list_b_out

        get_b_out = await get_b.ainvoke({"skill_id": skill_id_str})
        assert "[TOOL ERROR]" in get_b_out
        assert "not found" in get_b_out

    async def test_user_b_cannot_update_user_a_skill(
        self,
        pool_with_schema: asyncpg.Pool,
    ) -> None:
        agent_id = _new_uuid()
        user_a = _new_uuid()
        user_b = _new_uuid()
        skills, _ = _build_collections(pool_with_schema)
        registry = _FakeRegistry()

        [create_a] = load_skill_create_tool(
            agent_id=agent_id,
            user_id=user_a,
            skills_collection=skills,
            registry=registry,
        )
        [update_b] = load_skill_update_tool(
            agent_id=agent_id,
            user_id=user_b,
            skills_collection=skills,
            registry=registry,
        )
        out = await create_a.ainvoke(
            {"name": "orig", "summary": "private", "body": "private"},
        )
        skill_id_str = out.split("[skill:")[1].split("]")[0]

        update_out = await update_b.ainvoke(
            {"skill_id": skill_id_str, "summary": "hijacked"},
        )
        assert "[TOOL ERROR]" in update_out
        assert "not found" in update_out


class TestCapEnforcement:
    """The 200-prose-skill cap (configurable) holds against the DB."""

    async def test_cap_blocks_creation(
        self,
        pool_with_schema: asyncpg.Pool,
    ) -> None:
        agent_id = _new_uuid()
        user_id = _new_uuid()
        skills, _ = _build_collections(pool_with_schema)
        registry = _FakeRegistry()
        [create_tool] = load_skill_create_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=skills,
            registry=registry,
            max_prose_skills_per_user=3,
        )
        for i in range(3):
            out = await create_tool.ainvoke(
                {
                    "name": f"skill-{i}",
                    "summary": "s",
                    "body": "b",
                },
            )
            assert out.startswith("[skill:")
        # 4th rejected
        out = await create_tool.ainvoke(
            {"name": "skill-4", "summary": "s", "body": "b"},
        )
        assert "[TOOL ERROR]" in out
        assert "max 3 prose skills" in out


# --- skill_invoke against real schema ---


class TestSkillInvokeIntegration:
    """``skill_invoke`` writes an ``agent_skill_invocations`` row."""

    async def test_invocation_row_persisted(
        self,
        pool_with_schema: asyncpg.Pool,
    ) -> None:
        agent_id = _new_uuid()
        user_id = _new_uuid()
        conversation_id = _new_uuid()
        skills, invocations = _build_collections(pool_with_schema)
        registry = _FakeRegistry()
        [create_tool] = load_skill_create_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=skills,
            registry=registry,
        )

        active: dict[str, Any] = {"id": None}

        def _probe() -> UUID | None:
            return active["id"]

        def _setter(sid: UUID) -> None:
            active["id"] = sid

        [invoke_tool] = load_skill_invoke_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=skills,
            invocations_collection=invocations,
            conversation_id_resolver=lambda: conversation_id,
            active_skill_probe=_probe,
            active_skill_setter=_setter,
        )

        # No FK from agent_skill_invocations.conversation_id to
        # conversations (consumer-owned table; messages-style soft link
        # per shard-01 design notes).
        out = await create_tool.ainvoke(
            {"name": "test-skill", "summary": "s", "body": "do this"},
        )
        skill_id = UUID(out.split("[skill:")[1].split("]")[0])

        invoke_out = await invoke_tool.ainvoke({"skill_id": str(skill_id)})
        assert "[ACTIVE SKILL: test-skill]" in invoke_out
        assert "do this" in invoke_out
        assert active["id"] == skill_id

        # Verify invocation row landed
        rows = await pool_with_schema.fetch(
            "SELECT skill_id, invocation_source, conversation_id FROM agent_skill_invocations WHERE agent_id = $1",
            agent_id,
        )
        assert len(rows) == 1
        assert rows[0]["skill_id"] == skill_id
        assert rows[0]["invocation_source"] == "invoke"
        assert rows[0]["conversation_id"] == conversation_id


# --- skill_list UNION with tool-skill from registry ---


class TestSkillListUnion:
    async def test_tool_skill_surfaces_via_registry(
        self,
        pool_with_schema: asyncpg.Pool,
    ) -> None:
        agent_id = _new_uuid()
        user_id = _new_uuid()
        skills, _ = _build_collections(pool_with_schema)
        registry = _FakeRegistry(
            skill_eligible=[
                SkillEligibleTool(
                    mcp_name="loki.query",
                    summary="Query Loki logs",
                ),
            ],
        )
        [list_tool] = load_skill_list_tool(
            agent_id=agent_id,
            user_id=user_id,
            skills_collection=skills,
            registry=registry,
        )
        out = await list_tool.ainvoke({})
        assert "loki.query" in out
        assert "kind=tool" in out
