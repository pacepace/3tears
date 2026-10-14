"""End-to-end integration tests for the schedule + webhook tool factories.

Spins up a fresh schema via the canonical ``pg_schema`` fixture,
applies the conversations + skills + wake migrations, and exercises
the real Collections against live Postgres.

Covers:

- Schedule lifecycle: create -> list -> pause -> resume -> delete.
- Cap-of-10 enforcement against a live row count.
- Webhook lifecycle: create -> list -> rotate -> delete.
- Cycle detection rejects an A->A loop on update.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import asyncpg
import pytest
from uuid_utils import uuid7

from threetears.agent.wake.collections import (
    WakeFireCollection,
    WakeScheduleCollection,
    WebhookSubscriptionCollection,
)
from threetears.agent.wake.manual import claim_manual_fire
from threetears.agent.wake.migrations import register as register_wake
from threetears.agent.wake.tools import (
    WakeRegistryClient,
    load_wake_fires_tool,
    load_wake_schedule_create_tool,
    load_wake_schedule_delete_tool,
    load_wake_schedule_get_tool,
    load_wake_schedule_list_tool,
    load_wake_schedule_pause_tool,
    load_wake_schedule_resume_tool,
    load_wake_schedule_update_tool,
    load_webhook_subscription_create_tool,
    load_webhook_subscription_delete_tool,
    load_webhook_subscription_get_tool,
    load_webhook_subscription_list_tool,
    load_webhook_subscription_pause_tool,
    load_webhook_subscription_resume_tool,
    next_fires,
    load_webhook_subscription_rotate_secret_tool,
    load_webhook_subscription_update_tool,
)
from threetears.agent.skills.migrations import register as register_skills
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration


def _new_uuid() -> UUID:
    return UUID(str(uuid7()))


# parity-with: threetears.agent.wake.tools.schedule_tools.WakeRegistryClient
class _PermissiveRegistry(WakeRegistryClient):
    """All skill IDs permitted; no name lookups."""

    async def acl_permits_skill(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        skill_id: UUID,
    ) -> bool:
        del user_id, agent_id, skill_id
        return True

    async def skill_name_for_id(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        skill_id: UUID,
    ) -> str | None:
        del user_id, agent_id, skill_id
        return None


# parity-with: threetears.agent.wake.tools.schedule_tools.WakeRegistryClient
class _RestrictiveRegistry(WakeRegistryClient):
    """Denies every skill_id ACL probe; returns no names."""

    async def acl_permits_skill(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        skill_id: UUID,
    ) -> bool:
        del user_id, agent_id, skill_id
        return False

    async def skill_name_for_id(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        skill_id: UUID,
    ) -> str | None:
        del user_id, agent_id, skill_id
        return None


# parity-with: threetears.agent.wake.entities.EncryptionService
class _IdentityEncryption:
    """Identity encryption for the integration tests (no real crypto)."""

    def encrypt(self, plaintext: bytes) -> bytes:
        return bytes(plaintext)

    def decrypt(self, ciphertext: bytes) -> str:
        return ciphertext.decode("utf-8")


async def _apply_schema(url: str, schema: str) -> asyncpg.Pool:
    setup_conn = await asyncpg.connect(url)
    try:
        await setup_conn.execute(f'SET search_path TO "{schema}", public')
        runner = MigrationRunner()
        register_conversations(runner)
        register_skills(runner)
        register_wake(runner)
        store = AsyncpgStore(setup_conn)
        await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
    finally:
        await setup_conn.close()
    pool = await asyncpg.create_pool(
        url,
        min_size=2,
        max_size=8,
        server_settings={"search_path": f"{schema}, public"},
        init=init_connection,
    )
    assert pool is not None
    return pool


def _build_collections(
    pool: asyncpg.Pool,
) -> tuple[WakeScheduleCollection, WebhookSubscriptionCollection]:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    config = DefaultCoreConfig(
        collection_flush="ALWAYS",
        collection_flush_tables="",
    )
    return (
        WakeScheduleCollection(registry=registry, config=config),
        WebhookSubscriptionCollection(registry=registry, config=config),
    )


@pytest.mark.asyncio
async def test_schedule_lifecycle_create_list_pause_resume_delete(
    pg_schema: tuple[str, str],
) -> None:
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        schedules, _ = _build_collections(pool)
        conv_id = _new_uuid()
        user_id = _new_uuid()
        agent_id = _new_uuid()
        registry = _PermissiveRegistry()

        create_tool = load_wake_schedule_create_tool(
            conversation_id=conv_id,
            user_id=user_id,
            agent_id=agent_id,
            schedules_collection=schedules,
            registry=registry,
        )[0]
        list_tool = load_wake_schedule_list_tool(
            user_id=user_id,
            agent_id=agent_id,
            schedules_collection=schedules,
            registry=registry,
        )[0]
        pause_tool = load_wake_schedule_pause_tool(
            agent_id=agent_id,
            schedules_collection=schedules,
        )[0]
        resume_tool = load_wake_schedule_resume_tool(
            agent_id=agent_id,
            schedules_collection=schedules,
        )[0]
        delete_tool = load_wake_schedule_delete_tool(
            agent_id=agent_id,
            schedules_collection=schedules,
        )[0]

        # Create
        create_result = await create_tool.ainvoke(
            {
                "schedule_type": "interval",
                "schedule_config": {"seconds": 600},
                "name": "lifecycle-test",
            },
        )
        assert create_result.startswith("[schedule:"), create_result
        sched_id_str = create_result.split("]")[0].removeprefix("[schedule:")
        sched_id = UUID(sched_id_str)

        # List
        list_result = await list_tool.ainvoke({})
        assert "lifecycle-test" in list_result

        # Pause
        pause_result = await pause_tool.ainvoke({"schedule_id": str(sched_id)})
        assert "Paused" in pause_result
        row = await schedules.get((conv_id, sched_id))
        assert row is not None and row.status == "paused"

        # Resume
        resume_result = await resume_tool.ainvoke({"schedule_id": str(sched_id)})
        assert "Resumed" in resume_result
        row = await schedules.get((conv_id, sched_id))
        assert row is not None and row.status == "active"
        assert row.next_fire_at is not None

        # Delete
        delete_result = await delete_tool.ainvoke({"schedule_id": str(sched_id)})
        assert "Deleted" in delete_result
        assert await schedules.get((conv_id, sched_id)) is None
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_schedule_include_conversation_history_persists_and_updates(
    pg_schema: tuple[str, str],
) -> None:
    """The history switch defaults on, persists when set off, and flips on update."""
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        schedules, _ = _build_collections(pool)
        conv_id = _new_uuid()
        user_id = _new_uuid()
        agent_id = _new_uuid()
        registry = _PermissiveRegistry()

        create_tool = load_wake_schedule_create_tool(
            conversation_id=conv_id,
            user_id=user_id,
            agent_id=agent_id,
            schedules_collection=schedules,
            registry=registry,
        )[0]
        update_tool = load_wake_schedule_update_tool(
            user_id=user_id,
            agent_id=agent_id,
            schedules_collection=schedules,
            registry=registry,
        )[0]

        # Default: omitting the flag persists ``True`` (prior behavior).
        default_result = await create_tool.ainvoke(
            {"schedule_type": "interval", "schedule_config": {"seconds": 600}},
        )
        default_id = UUID(default_result.split("]")[0].removeprefix("[schedule:"))
        default_row = await schedules.get((conv_id, default_id))
        assert default_row is not None
        assert default_row.include_conversation_history is True

        # Explicit off persists ``False``.
        off_result = await create_tool.ainvoke(
            {
                "schedule_type": "interval",
                "schedule_config": {"seconds": 600},
                "include_conversation_history": False,
            },
        )
        off_id = UUID(off_result.split("]")[0].removeprefix("[schedule:"))
        off_row = await schedules.get((conv_id, off_id))
        assert off_row is not None
        assert off_row.include_conversation_history is False

        # Update flips it back on.
        await update_tool.ainvoke(
            {
                "schedule_id": str(off_id),
                "include_conversation_history": True,
            },
        )
        updated_row = await schedules.get((conv_id, off_id))
        assert updated_row is not None
        assert updated_row.include_conversation_history is True
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_schedule_create_enforces_the_agent_cap_across_conversations(
    pg_schema: tuple[str, str],
) -> None:
    """The cap counts the agent's active wakes in every conversation, under the agent's lock."""
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        schedules, _ = _build_collections(pool)
        user_id = _new_uuid()
        agent_id = _new_uuid()
        cap = 3

        def create_in(conversation_id: UUID) -> Any:
            return load_wake_schedule_create_tool(
                conversation_id=conversation_id,
                user_id=user_id,
                agent_id=agent_id,
                schedules_collection=schedules,
                registry=_PermissiveRegistry(),
                max_active_schedules_per_agent=cap,
            )[0]

        for i in range(cap):
            res = await create_in(_new_uuid()).ainvoke(
                {
                    "schedule_type": "interval",
                    "schedule_config": {"seconds": 600 + i},
                    "name": f"sched-{i}",
                },
            )
            assert res.startswith("[schedule:"), res

        rejected = await create_in(_new_uuid()).ainvoke(
            {
                "schedule_type": "interval",
                "schedule_config": {"seconds": 999},
            },
        )
        assert rejected.startswith("[TOOL ERROR]")
        assert f"{cap} active wakes" in rejected
        # another agent is not affected by this agent's count
        other = load_wake_schedule_create_tool(
            conversation_id=_new_uuid(),
            user_id=user_id,
            agent_id=_new_uuid(),
            schedules_collection=schedules,
            registry=_PermissiveRegistry(),
            max_active_schedules_per_agent=cap,
        )[0]
        assert (await other.ainvoke({"schedule_type": "interval", "schedule_config": {"seconds": 60}})).startswith(
            "[schedule:"
        )
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_schedule_list_reaches_across_the_agents_conversations(
    pg_schema: tuple[str, str],
) -> None:
    """A wake created in one conversation is listed from another of the same agent, and by no other agent."""
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        schedules, _ = _build_collections(pool)
        conv_a = _new_uuid()
        user_id = _new_uuid()
        agent_id = _new_uuid()

        create_tool_a = load_wake_schedule_create_tool(
            conversation_id=conv_a,
            user_id=user_id,
            agent_id=agent_id,
            schedules_collection=schedules,
            registry=_PermissiveRegistry(),
        )[0]
        res = await create_tool_a.ainvoke(
            {"schedule_type": "interval", "schedule_config": {"seconds": 600}, "name": "from-a"},
        )
        assert res.startswith("[schedule:")

        list_tool = load_wake_schedule_list_tool(
            user_id=user_id,
            agent_id=agent_id,
            schedules_collection=schedules,
            registry=_PermissiveRegistry(),
        )[0]
        list_result = await list_tool.ainvoke({})
        assert "from-a" in list_result
        assert f"[conversation:{conv_a}]" in list_result

        stranger_list = load_wake_schedule_list_tool(
            user_id=user_id,
            agent_id=_new_uuid(),
            schedules_collection=schedules,
            registry=_PermissiveRegistry(),
        )[0]
        assert "no wake schedules" in await stranger_list.ainvoke({})
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_schedule_update_rejects_self_context_from_cycle(
    pg_schema: tuple[str, str],
) -> None:
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        schedules, _ = _build_collections(pool)
        conv_id = _new_uuid()
        user_id = _new_uuid()
        agent_id = _new_uuid()
        registry = _PermissiveRegistry()

        create_tool = load_wake_schedule_create_tool(
            conversation_id=conv_id,
            user_id=user_id,
            agent_id=agent_id,
            schedules_collection=schedules,
            registry=registry,
        )[0]
        update_tool = load_wake_schedule_update_tool(
            user_id=user_id,
            agent_id=agent_id,
            schedules_collection=schedules,
            registry=registry,
        )[0]

        res = await create_tool.ainvoke(
            {"schedule_type": "interval", "schedule_config": {"seconds": 600}},
        )
        sched_id_str = res.split("]")[0].removeprefix("[schedule:")
        sched_id = UUID(sched_id_str)

        # Try to set context_from to itself.
        update_result = await update_tool.ainvoke(
            {
                "schedule_id": str(sched_id),
                "context_from_schedule_id": str(sched_id),
            },
        )
        assert update_result.startswith("[TOOL ERROR]")
        assert "cycle" in update_result
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_schedule_create_acl_denied_skill_does_not_persist(
    pg_schema: tuple[str, str],
) -> None:
    """ACL-denied skill_id on create must NOT persist the schedule row.

    Covers the Critic warning that the ACL-denied path was only
    exercised at the unit level against an in-memory fake collection;
    this test runs the rejection against testcontainers Postgres so the
    "no row written on ACL denial" contract has a real-stack guard.
    """
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        schedules, _ = _build_collections(pool)
        conv_id = _new_uuid()
        user_id = _new_uuid()
        agent_id = _new_uuid()
        forbidden_skill = _new_uuid()

        create_tool = load_wake_schedule_create_tool(
            conversation_id=conv_id,
            user_id=user_id,
            agent_id=agent_id,
            schedules_collection=schedules,
            registry=_RestrictiveRegistry(),
        )[0]
        list_tool = load_wake_schedule_list_tool(
            user_id=user_id,
            agent_id=agent_id,
            schedules_collection=schedules,
            registry=_PermissiveRegistry(),
        )[0]

        result = await create_tool.ainvoke(
            {
                "schedule_type": "interval",
                "schedule_config": {"seconds": 600},
                "skill_id": str(forbidden_skill),
                "name": "should-not-persist",
            },
        )
        assert result.startswith("[TOOL ERROR]"), result
        assert "not authorized" in result

        # No row landed in Postgres.
        list_result = await list_tool.ainvoke({})
        assert "no wake schedules" in list_result, list_result
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_schedule_update_cross_agent_returns_not_found(
    pg_schema: tuple[str, str],
) -> None:
    """Agent B cannot read or mutate a schedule owned by agent A.

    Seeds via agent A's create tool against real Postgres, then attempts
    update via agent B's update tool. The attempt MUST surface as "not
    found" (existence is not leaked) and MUST NOT mutate the row.
    """
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        schedules, _ = _build_collections(pool)
        conv_id = _new_uuid()
        user_a = _new_uuid()
        user_b = _new_uuid()
        agent_id = _new_uuid()

        # User A creates a schedule.
        create_a = load_wake_schedule_create_tool(
            conversation_id=conv_id,
            user_id=user_a,
            agent_id=agent_id,
            schedules_collection=schedules,
            registry=_PermissiveRegistry(),
        )[0]
        create_result = await create_a.ainvoke(
            {
                "schedule_type": "interval",
                "schedule_config": {"seconds": 600},
                "name": "owned-by-a",
            },
        )
        assert create_result.startswith("[schedule:"), create_result
        sched_id = UUID(create_result.split("]")[0].removeprefix("[schedule:"))

        # User B tries to update it.
        update_b = load_wake_schedule_update_tool(
            user_id=user_b,
            agent_id=_new_uuid(),
            schedules_collection=schedules,
            registry=_PermissiveRegistry(),
        )[0]
        update_result = await update_b.ainvoke(
            {
                "schedule_id": str(sched_id),
                "name": "hijacked",
            },
        )
        assert update_result.startswith("[TOOL ERROR]")
        assert "not found" in update_result

        # The row still belongs to user A and the name is untouched.
        row = await schedules.get((conv_id, sched_id))
        assert row is not None
        assert row.user_id == user_a
        assert row.name == "owned-by-a"
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_webhook_update_cross_agent_returns_not_found(
    pg_schema: tuple[str, str],
) -> None:
    """Agent B cannot read or mutate a webhook subscription owned by agent A."""
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        _, subs = _build_collections(pool)
        conv_id = _new_uuid()
        user_a = _new_uuid()
        user_b = _new_uuid()
        agent_id = _new_uuid()
        enc = _IdentityEncryption()

        create_a = load_webhook_subscription_create_tool(
            conversation_id=conv_id,
            user_id=user_a,
            agent_id=agent_id,
            subscriptions_collection=subs,
            encryption_service=enc,
            registry=_PermissiveRegistry(),
        )[0]
        create_result = await create_a.ainvoke(
            {
                "task_prompt_template": "x: {{event.type}}",
                "name": "owned-by-a",
            },
        )
        assert "[webhook:" in create_result
        sub_id = UUID(create_result.split("[webhook:")[1].split("]")[0])

        update_b = load_webhook_subscription_update_tool(
            user_id=user_b,
            agent_id=_new_uuid(),
            subscriptions_collection=subs,
            registry=_PermissiveRegistry(),
        )[0]
        update_result = await update_b.ainvoke(
            {
                "subscription_id": str(sub_id),
                "name": "hijacked",
            },
        )
        assert update_result.startswith("[TOOL ERROR]")
        assert "not found" in update_result

        row = await subs.get((conv_id, sub_id))
        assert row is not None
        assert row.user_id == user_a
        assert row.name == "owned-by-a"
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_webhook_lifecycle_create_list_rotate_delete(
    pg_schema: tuple[str, str],
) -> None:
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        _, subs = _build_collections(pool)
        conv_id = _new_uuid()
        user_id = _new_uuid()
        agent_id = _new_uuid()
        enc = _IdentityEncryption()
        registry = _PermissiveRegistry()

        create_tool = load_webhook_subscription_create_tool(
            conversation_id=conv_id,
            user_id=user_id,
            agent_id=agent_id,
            subscriptions_collection=subs,
            encryption_service=enc,
            registry=registry,
        )[0]
        list_tool = load_webhook_subscription_list_tool(
            user_id=user_id,
            agent_id=agent_id,
            subscriptions_collection=subs,
            registry=registry,
        )[0]
        rotate_tool = load_webhook_subscription_rotate_secret_tool(
            agent_id=agent_id,
            subscriptions_collection=subs,
            encryption_service=enc,
        )[0]
        delete_tool = load_webhook_subscription_delete_tool(
            agent_id=agent_id,
            subscriptions_collection=subs,
        )[0]
        update_tool = load_webhook_subscription_update_tool(
            user_id=user_id,
            agent_id=agent_id,
            subscriptions_collection=subs,
            registry=registry,
        )[0]

        create_result = await create_tool.ainvoke(
            {
                "task_prompt_template": "event: {{event.type}}",
                "name": "gh-push",
            },
        )
        assert "[webhook:" in create_result
        assert "secret" in create_result
        sub_id_str = create_result.split("[webhook:")[1].split("]")[0]
        sub_id = UUID(sub_id_str)
        original_row = await subs.get((conv_id, sub_id))
        assert original_row is not None
        original_ciphertext = bytes(original_row.secret_ciphertext)

        list_result = await list_tool.ainvoke({})
        assert "gh-push" in list_result

        # Update the template.
        update_result = await update_tool.ainvoke(
            {
                "subscription_id": str(sub_id),
                "task_prompt_template": "updated: {{event.id}}",
            },
        )
        assert "[webhook:" in update_result
        updated_row = await subs.get((conv_id, sub_id))
        assert updated_row is not None
        assert updated_row.task_prompt_template == "updated: {{event.id}}"

        # Rotate the secret.
        rotate_result = await rotate_tool.ainvoke({"subscription_id": str(sub_id)})
        assert "Rotated" in rotate_result
        new_row = await subs.get((conv_id, sub_id))
        assert new_row is not None
        assert bytes(new_row.secret_ciphertext) != original_ciphertext

        # Delete.
        delete_result = await delete_tool.ainvoke({"subscription_id": str(sub_id)})
        assert "Deleted" in delete_result
        assert await subs.get((conv_id, sub_id)) is None
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_the_same_wake_made_twice_is_made_once_and_a_different_one_is_made(
    pg_schema: tuple[str, str],
) -> None:
    """A wake made twice fires twice; the second create names the first and makes nothing."""
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        schedules, _ = _build_collections(pool)
        conv_id, other_conv, user_id, agent_id = _new_uuid(), _new_uuid(), _new_uuid(), _new_uuid()

        def _create(conversation: UUID, agent: UUID = agent_id) -> Any:
            return load_wake_schedule_create_tool(
                conversation_id=conversation,
                user_id=user_id,
                agent_id=agent,
                schedules_collection=schedules,
                registry=_PermissiveRegistry(),
            )[0]

        wake = {
            "schedule_type": "interval",
            "schedule_config": {"seconds": 600},
            "name": "mail check",
            "task_prompt": "Read the inbox.",
        }
        first = await _create(conv_id).ainvoke(wake)
        assert first.startswith("[schedule:"), first
        first_id = first.split("]")[0].removeprefix("[schedule:")

        # the same wake again, from another conversation of the same agent, in another turn
        again = await _create(other_conv).ainvoke(dict(wake))
        assert again.startswith("[NO CHANGE] wake_schedule_create: you already have this wake"), again
        assert f"[schedule:{first_id}]" in again and "(active)" in again and "Nothing was made." in again
        assert await pool.fetchval("SELECT count(*) FROM agent_wake_schedules WHERE agent_id = $1", agent_id) == 1

        # good traffic: a wake that differs in any one part is another wake, and is made
        for changed in (
            {"schedule_config": {"seconds": 900}},
            {"task_prompt": "Read the inbox and the calendar."},
            {"name": "mail check 2"},
            {"include_conversation_history": False},
            {"missed_fire_policy": "catch_up"},
            {"schedule_type": "one_shot_at", "schedule_config": {"fire_at_iso": "2099-01-01T00:00:00+00:00"}},
        ):
            made = await _create(conv_id).ainvoke({**wake, **changed})
            assert made.startswith("[schedule:"), (changed, made)
        # and another agent's identical wake is its own
        theirs = await _create(_new_uuid(), agent=_new_uuid()).ainvoke(dict(wake))
        assert theirs.startswith("[schedule:"), theirs
        assert await pool.fetchval("SELECT count(*) FROM agent_wake_schedules WHERE agent_id = $1", agent_id) == 7
        # two wakes with no name and no text say nothing of what they are for: both are made
        bare = {"schedule_type": "interval", "schedule_config": {"seconds": 600}}
        assert (await _create(conv_id).ainvoke(bare)).startswith("[schedule:")
        assert (await _create(conv_id).ainvoke(dict(bare))).startswith("[schedule:")

        # a wake that was deleted no longer stands in the way
        await pool.execute("DELETE FROM agent_wake_schedules WHERE schedule_id = $1", UUID(first_id))
        remade = await _create(conv_id).ainvoke(dict(wake))
        assert remade.startswith("[schedule:"), remade
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_a_listing_that_cannot_be_read_does_not_cost_the_create(
    pg_schema: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        schedules, _ = _build_collections(pool)

        async def _down(*_args: Any, **_kwargs: Any) -> Any:
            raise OSError("the listing is down")

        monkeypatch.setattr(schedules, "list_for_agent", _down)
        tool = load_wake_schedule_create_tool(
            conversation_id=_new_uuid(),
            user_id=_new_uuid(),
            agent_id=_new_uuid(),
            schedules_collection=schedules,
            registry=_PermissiveRegistry(),
        )[0]
        made = await tool.ainvoke({"schedule_type": "interval", "schedule_config": {"seconds": 600}, "name": "n"})
        assert made.startswith("[schedule:"), made
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# What a result states: the schedule, its fires, what changed
# ---------------------------------------------------------------------------

_LA = "America/Los_Angeles"


def _fires(pool: asyncpg.Pool) -> WakeFireCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    return WakeFireCollection(
        registry=registry, config=DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    )


async def _made_schedule(pool: asyncpg.Pool, schedules: WakeScheduleCollection, agent_id: UUID, **create: Any) -> UUID:
    tool = load_wake_schedule_create_tool(
        conversation_id=_new_uuid(),
        user_id=_new_uuid(),
        agent_id=agent_id,
        schedules_collection=schedules,
        registry=_PermissiveRegistry(),
    )[0]
    result = await tool.ainvoke(create)
    assert result.startswith("[schedule:"), result
    return UUID(result.split("]")[0].removeprefix("[schedule:"))


@pytest.mark.asyncio
async def test_the_listing_and_the_get_state_the_schedule_and_its_fires_in_the_callers_zone(
    pg_schema: tuple[str, str],
) -> None:
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        schedules, _ = _build_collections(pool)
        agent_id, user_id = _new_uuid(), _new_uuid()
        sid = await _made_schedule(
            pool,
            schedules,
            agent_id,
            schedule_type="interval",
            schedule_config={"seconds": 5400},
            name="every ninety",
            task_prompt="look at the queue " + "and again " * 300,
        )
        due = datetime(2026, 10, 10, 20, 0, tzinfo=UTC)
        await pool.execute(
            "UPDATE agent_wake_schedules SET next_fire_at = $2, last_fired_at = $3 WHERE schedule_id = $1",
            sid,
            due,
            due - timedelta(minutes=90),
        )
        listed = await load_wake_schedule_list_tool(
            user_id=user_id, agent_id=agent_id, schedules_collection=schedules, registry=_PermissiveRegistry(), tz=_LA
        )[0].ainvoke({})
        assert (
            " · interval · every 90 minutes · next: Oct 10, 2026 at 1:00 PM PDT · last: Oct 10, 2026 at 11:30 AM PDT · "
            in listed
        ), listed

        got = await load_wake_schedule_get_tool(
            user_id=user_id, agent_id=agent_id, schedules_collection=schedules, registry=_PermissiveRegistry(), tz=_LA
        )[0].ainvoke({"schedule_id": f"[schedule:{sid}]"})
        assert "fires: every 90 minutes" in got
        assert (
            "next fires: Oct 10, 2026 at 1:00 PM PDT, Oct 10, 2026 at 2:30 PM PDT, Oct 10, 2026 at 4:00 PM PDT" in got
        ), got
        assert "last fire: Oct 10, 2026 at 11:30 AM PDT" in got
        assert 'schedule_config: {"seconds": 5400}' in got
        # the text is whole: nothing a later edit would shorten by being written back
        assert got.endswith("look at the queue " + "and again " * 300)

        missing = await load_wake_schedule_get_tool(
            user_id=user_id,
            agent_id=_new_uuid(),
            schedules_collection=schedules,
            registry=_PermissiveRegistry(),
        )[0].ainvoke({"schedule_id": str(sid)})
        assert missing.startswith("[TOOL ERROR] wake_schedule_get: schedule not found"), "another agent's wake"
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_next_fires_steps_as_the_tick_will_for_each_repeating_type(pg_schema: tuple[str, str]) -> None:
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        schedules, _ = _build_collections(pool)
        agent_id = _new_uuid()
        daily = await _made_schedule(
            pool, schedules, agent_id, schedule_type="daily_at", schedule_config={"hour": 3, "minute": 0}, name="d"
        )
        once = await _made_schedule(
            pool, schedules, agent_id, schedule_type="relative_delay", schedule_config={"delay": "30m"}, name="o"
        )
        entity = await schedules.find_for_agent(agent_id, daily)
        assert entity is not None and entity.next_fire_at is not None
        three = next_fires(entity, 3)
        assert three == [entity.next_fire_at + timedelta(days=n) for n in range(3)]
        one_shot = await schedules.find_for_agent(agent_id, once)
        assert one_shot is not None
        assert next_fires(one_shot, 3) == [one_shot.next_fire_at], "a one-shot has one fire"
        await schedules.pause(entity.conversation_id, daily)
        paused = await schedules.find_for_agent(agent_id, daily)
        assert paused is not None and next_fires(paused, 3) == []
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_an_update_says_what_changed_and_the_same_values_change_nothing(pg_schema: tuple[str, str]) -> None:
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        schedules, _ = _build_collections(pool)
        agent_id, user_id = _new_uuid(), _new_uuid()
        sid = await _made_schedule(
            pool, schedules, agent_id, schedule_type="interval", schedule_config={"seconds": 3600}, name="hourly"
        )
        update = load_wake_schedule_update_tool(
            user_id=user_id, agent_id=agent_id, schedules_collection=schedules, registry=_PermissiveRegistry(), tz=_LA
        )[0]
        before = await pool.fetchrow(
            "SELECT next_fire_at, date_updated FROM agent_wake_schedules WHERE schedule_id = $1", sid
        )

        same = await update.ainvoke({"schedule_id": str(sid), "schedule_config": {"seconds": 3600}, "name": "hourly"})
        assert same.startswith("[NO CHANGE] wake_schedule_update: every value given is the one"), same
        assert "every hour" in same
        after = await pool.fetchrow(
            "SELECT next_fire_at, date_updated FROM agent_wake_schedules WHERE schedule_id = $1", sid
        )
        assert dict(after) == dict(before), "the same values write nothing and move no fire"

        changed = await update.ainvoke(
            {"schedule_id": str(sid), "schedule_config": {"seconds": 7200}, "name": "two-hourly"}
        )
        assert "\nChanged: name: hourly -> two-hourly; schedule: every hour -> every 2 hours; next fire: " in changed
        assert " · every 2 hours · " in changed.splitlines()[0]
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_pause_resume_and_delete_state_what_the_wake_was(pg_schema: tuple[str, str]) -> None:
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        schedules, _ = _build_collections(pool)
        agent_id = _new_uuid()
        sid = await _made_schedule(
            pool, schedules, agent_id, schedule_type="interval", schedule_config={"seconds": 600}, name="ten"
        )
        pause = load_wake_schedule_pause_tool(agent_id=agent_id, schedules_collection=schedules, tz=_LA)[0]
        resume = load_wake_schedule_resume_tool(agent_id=agent_id, schedules_collection=schedules, tz=_LA)[0]
        delete = load_wake_schedule_delete_tool(agent_id=agent_id, schedules_collection=schedules, tz=_LA)[0]

        paused = await pause.ainvoke({"schedule_id": str(sid)})
        assert paused.startswith(f"Paused [schedule:{sid}] (ten). It was every 10 minutes; next fire "), paused
        assert "PDT; last fire never." in paused or "PST; last fire never." in paused
        again = await pause.ainvoke({"schedule_id": str(sid)})
        assert again.startswith("[NO CHANGE] wake_schedule_pause:"), again

        resumed = await resume.ainvoke({"schedule_id": str(sid)})
        assert resumed.startswith(f"Resumed [schedule:{sid}] (ten). It was paused. It is every 10 minutes. "), resumed

        deleted = await delete.ainvoke({"schedule_id": str(sid)})
        assert deleted.startswith(f"Deleted [schedule:{sid}] (ten). It was active: every 10 minutes; next fire "), (
            deleted
        )
        assert deleted.endswith("Its history went with it, and it cannot be brought back.")
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_wake_fires_shows_each_fire_how_it_started_and_how_it_ended(pg_schema: tuple[str, str]) -> None:
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        schedules, subs = _build_collections(pool)
        fires = _fires(pool)
        agent_id = _new_uuid()
        sid = await _made_schedule(
            pool, schedules, agent_id, schedule_type="interval", schedule_config={"seconds": 3600}, name="hourly"
        )
        entity = await schedules.find_for_agent(agent_id, sid)
        assert entity is not None
        conv = entity.conversation_id
        tool = load_wake_fires_tool(
            agent_id=agent_id,
            schedules_collection=schedules,
            fires_collection=fires,
            subscriptions_collection=subs,
            tz=_LA,
        )[0]
        assert (await tool.ainvoke({"schedule_id": str(sid)})).endswith("It has not fired yet.")

        late, started = _new_uuid(), _new_uuid()
        due = datetime(2026, 10, 10, 9, 0, tzinfo=UTC)
        await fires.create_dispatching(
            fire_id=late,
            schedule_id=sid,
            webhook_subscription_id=None,
            conversation_id=conv,
            scheduled_fire_at=due,
            actual_fired_at=due + timedelta(minutes=52),
            fire_source="scheduled_tick",
            execution_mode="spawn",
        )
        await pool.execute("UPDATE wake_fires SET started_conversation_id = $2 WHERE fire_id = $1", late, started)
        await fires.finalize_failed(conv, late, error="the model\n timed out", latency_ms=41000)
        claim = await claim_manual_fire(
            pool, entity, requested_by="user:abc", min_gap=timedelta(minutes=10), now=due + timedelta(hours=2)
        )
        await fires.finalize_success(conv, claim.fire_id, status="fired_silent", latency_ms=2000)

        shown = (await tool.ainvoke({"schedule_id": str(sid)})).splitlines()
        assert shown[0].startswith(f"[schedule:{sid}] hourly: every hour; next fire ")
        assert shown[1] == "Its newest 2 fire(s), newest first:"
        assert (
            shown[2] == "- Oct 10, 2026 at 4:00 AM PDT · run by hand, asked by user:abc · ran, and stayed silent · 2s"
        )
        assert shown[3] == (
            "- Oct 10, 2026 at 2:52 AM PDT · on schedule · due Oct 10, 2026 at 2:00 AM PDT, 52 min late · "
            f"failed: the model timed out · 41s · in [conversation:{started}]"
        )
        assert (await tool.ainvoke({})).startswith("[TOOL ERROR] wake_fires: give a schedule_id or a subscription_id")
        other = load_wake_fires_tool(agent_id=_new_uuid(), schedules_collection=schedules, fires_collection=fires)[0]
        assert (await other.ainvoke({"schedule_id": str(sid)})).startswith(
            "[TOOL ERROR] wake_fires: schedule not found"
        )
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_a_webhook_is_read_in_full_its_changes_are_said_and_its_refusals_are_shown(
    pg_schema: tuple[str, str],
) -> None:
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        schedules, subs = _build_collections(pool)
        fires = _fires(pool)
        conv_id, user_id, agent_id = _new_uuid(), _new_uuid(), _new_uuid()
        base = "https://example.test/api/v1/webhooks/"
        common: dict[str, Any] = {"user_id": user_id, "agent_id": agent_id, "subscriptions_collection": subs}
        create = load_webhook_subscription_create_tool(
            conversation_id=conv_id,
            encryption_service=_IdentityEncryption(),
            registry=_PermissiveRegistry(),
            endpoint_base_url=base,
            **common,
        )[0]
        template = "a push to {{event.repo}}: " + "read it carefully. " * 150
        made = await create.ainvoke(
            {
                "task_prompt_template": template,
                "name": "pushes",
                "allowed_source_pattern": "^10\\.",
                "rate_limit_per_minute": 5,
            }
        )
        sub_id = UUID(made.split("[webhook:")[1].split("]")[0])
        secret = [line for line in made.splitlines() if "secret (copy now" in line][0].split(":")[-1].strip()
        assert made.endswith(f"endpoint: https://example.test/api/v1/webhooks/{sub_id}")

        # the same subscription again is refused and names the one that stands
        twice = await create.ainvoke(
            {
                "task_prompt_template": template,
                "name": "pushes",
                "allowed_source_pattern": "^10\\.",
                "rate_limit_per_minute": 5,
            }
        )
        assert twice.startswith(
            f"[NO CHANGE] webhook_subscription_create: you already have this subscription, [webhook:{sub_id}] (active)"
        ), twice
        assert f"https://example.test/api/v1/webhooks/{sub_id}" in twice
        assert await pool.fetchval("SELECT count(*) FROM webhook_subscriptions WHERE agent_id = $1", agent_id) == 1
        # good traffic: one that differs in any part is made
        other = await create.ainvoke({"task_prompt_template": template, "name": "pushes to another repo"})
        assert other.startswith("[webhook:") and "secret (copy now" in other

        get = load_webhook_subscription_get_tool(
            registry=_PermissiveRegistry(), endpoint_base_url=base, tz=_LA, **common
        )[0]
        got = await get.ainvoke({"subscription_id": f"[webhook:{sub_id}]"})
        assert f"address: https://example.test/api/v1/webhooks/{sub_id}" in got
        assert "only from senders matching: ^10\\." in got
        assert "limit: 5 a minute" in got
        assert "refused deliveries kept: 0" in got
        assert got.endswith(template), "the template is whole"
        assert secret not in got, "the secret is never shown again"

        update = load_webhook_subscription_update_tool(registry=_PermissiveRegistry(), tz=_LA, **common)[0]
        changed = await update.ainvoke(
            {"subscription_id": str(sub_id), "name": "all pushes", "clear_rate_limit_per_minute": True}
        )
        assert changed.endswith("Changed: name: pushes -> all pushes; rate limit per minute: 5 -> the default"), changed
        same = await update.ainvoke({"subscription_id": str(sub_id), "name": "all pushes"})
        assert same.startswith("[NO CHANGE] webhook_subscription_update: every value given is the one"), same
        both = await update.ainvoke(
            {"subscription_id": str(sub_id), "rate_limit_per_minute": 3, "clear_rate_limit_per_minute": True}
        )
        assert both.startswith("[TOOL ERROR] webhook_subscription_update: rate_limit_per_minute and clear_"), both
        assert "limit: the default, 60 a minute" in await get.ainvoke({"subscription_id": str(sub_id)})

        entity = await subs.find_for_agent(agent_id, sub_id)
        assert entity is not None
        at = datetime(2026, 10, 10, 19, 0, 30, tzinfo=UTC)
        for _ in range(3):
            await subs.record_refusal(entity.conversation_id, sub_id, reason="bad_signature", status_code=403, at=at)
        delivered = _new_uuid()
        await fires.create_dispatching(
            fire_id=delivered,
            schedule_id=None,
            webhook_subscription_id=sub_id,
            conversation_id=entity.conversation_id,
            scheduled_fire_at=None,
            actual_fired_at=at - timedelta(hours=1),
            fire_source="webhook",
            execution_mode="spawn",
        )
        await fires.finalize_success(entity.conversation_id, delivered, status="fired", latency_ms=9000)
        history = await load_wake_fires_tool(
            agent_id=agent_id,
            schedules_collection=schedules,
            fires_collection=fires,
            subscriptions_collection=subs,
            tz=_LA,
        )[0].ainvoke({"subscription_id": str(sub_id)})
        assert "- Oct 10, 2026 at 11:00 AM PDT · a delivery · ran · 9s" in history
        assert (
            "- Oct 10, 2026 at 12:00 PM PDT · 3 refused · the signature did not match the secret · answered 403"
            in history
        ), history
        assert "refused deliveries kept: 3 (wake_fires with this subscription_id shows them)" in await get.ainvoke(
            {"subscription_id": str(sub_id)}
        )

        listed = await load_webhook_subscription_list_tool(
            registry=_PermissiveRegistry(), endpoint_base_url=base, **common
        )[0].ainvoke({})
        assert listed.endswith("Each one's address is https://example.test/api/v1/webhooks/<its id>.")
        nowhere = await load_webhook_subscription_list_tool(registry=_PermissiveRegistry(), **common)[0].ainvoke({})
        assert nowhere.endswith("Address: no public address is configured here, so no sender can reach it yet.")

        pause = load_webhook_subscription_pause_tool(agent_id=agent_id, subscriptions_collection=subs)[0]
        resume = load_webhook_subscription_resume_tool(agent_id=agent_id, subscriptions_collection=subs)[0]
        assert (await pause.ainvoke({"subscription_id": str(sub_id)})).startswith(
            f"Paused [webhook:{sub_id}] (all pushes). It was active."
        )
        assert (await pause.ainvoke({"subscription_id": str(sub_id)})).startswith(
            "[NO CHANGE] webhook_subscription_pause"
        )
        assert (await resume.ainvoke({"subscription_id": str(sub_id)})).startswith(
            f"Resumed [webhook:{sub_id}] (all pushes). It was paused."
        )
        rotated = await load_webhook_subscription_rotate_secret_tool(
            agent_id=agent_id,
            subscriptions_collection=subs,
            encryption_service=_IdentityEncryption(),
            endpoint_base_url=base,
        )[0].ainvoke({"subscription_id": str(sub_id)})
        assert rotated.endswith(f"endpoint: https://example.test/api/v1/webhooks/{sub_id}")
        gone = await load_webhook_subscription_delete_tool(agent_id=agent_id, subscriptions_collection=subs)[0].ainvoke(
            {"subscription_id": str(sub_id)}
        )
        assert gone.startswith(f"Deleted [webhook:{sub_id}] (all pushes). It was active.")
    finally:
        await pool.close()
