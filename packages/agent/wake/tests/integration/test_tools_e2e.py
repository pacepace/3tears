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

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import asyncpg
import pytest
from uuid_utils import uuid7

from threetears.agent.wake.collections import (
    WakeScheduleCollection,
    WebhookSubscriptionCollection,
)
from threetears.agent.wake.migrations import register as register_wake
from threetears.agent.wake.tools import (
    DEFAULT_MAX_SCHEDULES_PER_CONVERSATION,
    WakeRegistryClient,
    load_wake_schedule_create_tool,
    load_wake_schedule_delete_tool,
    load_wake_schedule_list_tool,
    load_wake_schedule_pause_tool,
    load_wake_schedule_resume_tool,
    load_wake_schedule_update_tool,
    load_webhook_subscription_create_tool,
    load_webhook_subscription_delete_tool,
    load_webhook_subscription_list_tool,
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
            conversation_id=conv_id,
            user_id=user_id,
            agent_id=agent_id,
            schedules_collection=schedules,
            registry=registry,
        )[0]
        pause_tool = load_wake_schedule_pause_tool(
            conversation_id=conv_id,
            user_id=user_id,
            schedules_collection=schedules,
        )[0]
        resume_tool = load_wake_schedule_resume_tool(
            conversation_id=conv_id,
            user_id=user_id,
            schedules_collection=schedules,
        )[0]
        delete_tool = load_wake_schedule_delete_tool(
            conversation_id=conv_id,
            user_id=user_id,
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
async def test_schedule_create_enforces_cap_of_10(
    pg_schema: tuple[str, str],
) -> None:
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        schedules, _ = _build_collections(pool)
        conv_id = _new_uuid()
        user_id = _new_uuid()
        agent_id = _new_uuid()

        create_tool = load_wake_schedule_create_tool(
            conversation_id=conv_id,
            user_id=user_id,
            agent_id=agent_id,
            schedules_collection=schedules,
            registry=_PermissiveRegistry(),
        )[0]

        for i in range(DEFAULT_MAX_SCHEDULES_PER_CONVERSATION):
            res = await create_tool.ainvoke(
                {
                    "schedule_type": "interval",
                    "schedule_config": {"seconds": 600 + i},
                    "name": f"sched-{i}",
                },
            )
            assert res.startswith("[schedule:"), res

        rejected = await create_tool.ainvoke(
            {
                "schedule_type": "interval",
                "schedule_config": {"seconds": 999},
            },
        )
        assert rejected.startswith("[TOOL ERROR]")
        assert "max" in rejected
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_schedule_cross_conversation_isolation(
    pg_schema: tuple[str, str],
) -> None:
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        schedules, _ = _build_collections(pool)
        conv_a = _new_uuid()
        conv_b = _new_uuid()
        user_id = _new_uuid()
        agent_id = _new_uuid()

        # Create one schedule under conv_a.
        create_tool_a = load_wake_schedule_create_tool(
            conversation_id=conv_a,
            user_id=user_id,
            agent_id=agent_id,
            schedules_collection=schedules,
            registry=_PermissiveRegistry(),
        )[0]
        res = await create_tool_a.ainvoke(
            {"schedule_type": "interval", "schedule_config": {"seconds": 600}},
        )
        assert res.startswith("[schedule:")

        # conv_b's list tool sees nothing.
        list_tool_b = load_wake_schedule_list_tool(
            conversation_id=conv_b,
            user_id=user_id,
            agent_id=agent_id,
            schedules_collection=schedules,
            registry=_PermissiveRegistry(),
        )[0]
        list_result = await list_tool_b.ainvoke({})
        assert "No wake schedules" in list_result
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
            conversation_id=conv_id,
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
            conversation_id=conv_id,
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
        assert "No wake schedules" in list_result, list_result
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_schedule_update_cross_user_returns_not_found(
    pg_schema: tuple[str, str],
) -> None:
    """User B cannot read or mutate a schedule owned by user A.

    Seeds via user A's create tool against real Postgres, then attempts
    update via user B's update tool. The cross-user attempt MUST
    surface as "not found" (existence is not leaked) and MUST NOT
    mutate the row.
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
            conversation_id=conv_id,
            user_id=user_b,
            agent_id=agent_id,
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
async def test_webhook_update_cross_user_returns_not_found(
    pg_schema: tuple[str, str],
) -> None:
    """User B cannot read or mutate a webhook subscription owned by user A."""
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
            conversation_id=conv_id,
            user_id=user_b,
            agent_id=agent_id,
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
            conversation_id=conv_id,
            user_id=user_id,
            agent_id=agent_id,
            subscriptions_collection=subs,
            registry=registry,
        )[0]
        rotate_tool = load_webhook_subscription_rotate_secret_tool(
            conversation_id=conv_id,
            user_id=user_id,
            subscriptions_collection=subs,
            encryption_service=enc,
        )[0]
        delete_tool = load_webhook_subscription_delete_tool(
            conversation_id=conv_id,
            user_id=user_id,
            subscriptions_collection=subs,
        )[0]
        update_tool = load_webhook_subscription_update_tool(
            conversation_id=conv_id,
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
