"""Integration test: cross-table FK cascade behaviour.

Three behaviours under test:

1. Deleting an ``agent_wake_schedules`` row CASCADE-deletes its
   ``wake_fires`` rows (the schedule-side FK is ``ON DELETE CASCADE``
   so that analytics for a deleted schedule do not survive as
   orphans).
2. Deleting a ``webhook_subscriptions`` row leaves matching
   ``wake_fires`` rows in place but sets
   ``webhook_subscription_id`` to NULL (the subscription-side FK is
   ``ON DELETE SET NULL`` so audit history outlives subscription
   deletes).
3. Deleting an ``agent_skills`` row sets
   ``agent_wake_schedules.skill_id`` and
   ``webhook_subscriptions.default_skill_id`` to NULL (cross-package
   FKs are ``ON DELETE SET NULL`` so the wake / subscription stays
   active but unbound).
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import asyncpg
import pytest
from uuid_utils import uuid7

from threetears.agent.skills.migrations import register as register_skills
from threetears.agent.wake.migrations import register as register_wake
from threetears.conversations.migrations import register as register_conversations
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration


def _new_uuid() -> UUID:
    """Return a fresh UUIDv7 cast to stdlib ``UUID``."""
    return UUID(str(uuid7()))


async def _apply(conn: asyncpg.Connection, schema: str) -> None:
    """Apply the full migration chain (conversations + skills + wake)."""
    await conn.execute(f'SET search_path TO "{schema}", public')
    runner = MigrationRunner()
    register_conversations(runner)
    register_skills(runner)
    register_wake(runner)
    store = AsyncpgStore(conn)
    await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]


async def _seed_skill(
    conn: asyncpg.Connection,
    *,
    agent_id: UUID,
    user_id: UUID,
) -> UUID:
    """Insert a minimal-payload skill and return its id."""
    skill_id = _new_uuid()
    await conn.execute(
        "INSERT INTO agent_skills (agent_id, skill_id, user_id, name, summary, body) VALUES ($1, $2, $3, $4, $5, $6)",
        agent_id,
        skill_id,
        user_id,
        f"skill-{skill_id}",
        "summary",
        "body",
    )
    return skill_id


async def _seed_schedule(
    conn: asyncpg.Connection,
    *,
    conv_id: UUID,
    user_id: UUID,
    agent_id: UUID,
    skill_id: UUID | None = None,
) -> UUID:
    """Insert a minimal schedule row and return its id."""
    schedule_id = _new_uuid()
    await conn.execute(
        "INSERT INTO agent_wake_schedules "
        "(conversation_id, schedule_id, user_id, agent_id, skill_id, schedule_type) "
        "VALUES ($1, $2, $3, $4, $5, $6)",
        conv_id,
        schedule_id,
        user_id,
        agent_id,
        skill_id,
        "daily_at",
    )
    return schedule_id


async def _seed_fire(
    conn: asyncpg.Connection,
    *,
    conv_id: UUID,
    schedule_id: UUID | None = None,
    subscription_id: UUID | None = None,
) -> UUID:
    """Insert a fire row and return its id."""
    fire_id = _new_uuid()
    await conn.execute(
        "INSERT INTO wake_fires "
        "(conversation_id, fire_id, schedule_id, webhook_subscription_id, "
        " actual_fired_at, status) "
        "VALUES ($1, $2, $3, $4, $5, $6)",
        conv_id,
        fire_id,
        schedule_id,
        subscription_id,
        datetime.now(UTC),
        "fired",
    )
    return fire_id


async def _seed_subscription(
    conn: asyncpg.Connection,
    *,
    conv_id: UUID,
    user_id: UUID,
    agent_id: UUID,
    default_skill_id: UUID | None = None,
) -> UUID:
    """Insert a subscription row and return its id."""
    sub_id = _new_uuid()
    await conn.execute(
        "INSERT INTO webhook_subscriptions "
        "(conversation_id, subscription_id, user_id, agent_id, "
        " default_skill_id, secret_ciphertext) "
        "VALUES ($1, $2, $3, $4, $5, $6)",
        conv_id,
        sub_id,
        user_id,
        agent_id,
        default_skill_id,
        b"\x00\x01",
    )
    return sub_id


class TestScheduleDeleteCascadesFires:
    """``agent_wake_schedules`` DELETE cascades ``wake_fires``."""

    async def test_delete_cascades(self, pg_schema: tuple[str, str]) -> None:
        """A deleted schedule's fires are removed in the same txn."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            conv = _new_uuid()
            user = _new_uuid()
            agent = _new_uuid()
            schedule_id = await _seed_schedule(
                conn,
                conv_id=conv,
                user_id=user,
                agent_id=agent,
            )
            fire_a = await _seed_fire(conn, conv_id=conv, schedule_id=schedule_id)
            fire_b = await _seed_fire(conn, conv_id=conv, schedule_id=schedule_id)

            await conn.execute(
                "DELETE FROM agent_wake_schedules WHERE conversation_id = $1 AND schedule_id = $2",
                conv,
                schedule_id,
            )
            for fire_id in (fire_a, fire_b):
                row = await conn.fetchrow(
                    "SELECT fire_id FROM wake_fires WHERE conversation_id = $1 AND fire_id = $2",
                    conv,
                    fire_id,
                )
                assert row is None, f"fire {fire_id} should have been cascade-deleted"
        finally:
            await conn.close()


class TestSubscriptionDeleteSetsFireSubIdNull:
    """``webhook_subscriptions`` DELETE sets ``wake_fires.webhook_subscription_id`` NULL."""

    async def test_delete_set_null(self, pg_schema: tuple[str, str]) -> None:
        """A deleted subscription leaves the fire row with subscription_id NULL."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            conv = _new_uuid()
            user = _new_uuid()
            agent = _new_uuid()
            sub_id = await _seed_subscription(
                conn,
                conv_id=conv,
                user_id=user,
                agent_id=agent,
            )
            fire_id = await _seed_fire(conn, conv_id=conv, subscription_id=sub_id)

            await conn.execute(
                "DELETE FROM webhook_subscriptions WHERE conversation_id = $1 AND subscription_id = $2",
                conv,
                sub_id,
            )
            row = await conn.fetchrow(
                "SELECT schedule_id, webhook_subscription_id FROM wake_fires "
                "WHERE conversation_id = $1 AND fire_id = $2",
                conv,
                fire_id,
            )
            assert row is not None, "fire row should still exist"
            assert row["webhook_subscription_id"] is None
        finally:
            await conn.close()


class TestSkillDeleteSetsScheduleSkillIdNull:
    """``agent_skills`` DELETE sets ``agent_wake_schedules.skill_id`` NULL."""

    async def test_skill_delete_unbinds_schedule(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """A deleted skill leaves the wake schedule active but skill_id NULL."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            conv = _new_uuid()
            user = _new_uuid()
            agent = _new_uuid()
            skill_id = await _seed_skill(conn, agent_id=agent, user_id=user)
            schedule_id = await _seed_schedule(
                conn,
                conv_id=conv,
                user_id=user,
                agent_id=agent,
                skill_id=skill_id,
            )

            await conn.execute(
                "DELETE FROM agent_skills WHERE agent_id = $1 AND skill_id = $2",
                agent,
                skill_id,
            )
            row = await conn.fetchrow(
                "SELECT skill_id FROM agent_wake_schedules WHERE conversation_id = $1 AND schedule_id = $2",
                conv,
                schedule_id,
            )
            assert row is not None
            assert row["skill_id"] is None
        finally:
            await conn.close()

    async def test_skill_delete_unbinds_subscription(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """A deleted skill leaves the webhook subscription with default_skill_id NULL."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            conv = _new_uuid()
            user = _new_uuid()
            agent = _new_uuid()
            skill_id = await _seed_skill(conn, agent_id=agent, user_id=user)
            sub_id = await _seed_subscription(
                conn,
                conv_id=conv,
                user_id=user,
                agent_id=agent,
                default_skill_id=skill_id,
            )

            await conn.execute(
                "DELETE FROM agent_skills WHERE agent_id = $1 AND skill_id = $2",
                agent,
                skill_id,
            )
            row = await conn.fetchrow(
                "SELECT default_skill_id FROM webhook_subscriptions "
                "WHERE conversation_id = $1 AND subscription_id = $2",
                conv,
                sub_id,
            )
            assert row is not None
            assert row["default_skill_id"] is None
        finally:
            await conn.close()
