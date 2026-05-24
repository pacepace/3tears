"""Integration test: cross-package FK targets ``agent_skills.skill_id`` UNIQUE.

The wake-side schedule + subscription tables FK to the bare
``agent_skills.skill_id`` column via the standalone ``UNIQUE
(skill_id)`` constraint that agent-skills v001 declared. This test
locks in the cross-package contract:

- Inserting a schedule / subscription with a ``skill_id`` /
  ``default_skill_id`` referring to a non-existent skill fails with
  ``ForeignKeyViolationError`` (the FK target column is the standalone
  UNIQUE, not the composite primary key, so the constraint is legal).
- Inserting a schedule / subscription with NULL ``skill_id`` /
  ``default_skill_id`` succeeds (the FK columns are nullable).
"""

from __future__ import annotations

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
    """Apply the full migration chain."""
    await conn.execute(f'SET search_path TO "{schema}", public')
    runner = MigrationRunner()
    register_conversations(runner)
    register_skills(runner)
    register_wake(runner)
    store = AsyncpgStore(conn)
    await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]


class TestScheduleSkillIdFk:
    """``agent_wake_schedules.skill_id`` FK behaviour."""

    async def test_orphan_skill_id_rejected(self, pg_schema: tuple[str, str]) -> None:
        """Inserting a schedule with a non-existent skill_id fails."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            with pytest.raises(asyncpg.exceptions.ForeignKeyViolationError):
                await conn.execute(
                    "INSERT INTO agent_wake_schedules "
                    "(conversation_id, schedule_id, user_id, agent_id, "
                    " skill_id, schedule_type) "
                    "VALUES ($1, $2, $3, $4, $5, $6)",
                    _new_uuid(),
                    _new_uuid(),
                    _new_uuid(),
                    _new_uuid(),
                    _new_uuid(),  # ghost skill_id
                    "daily_at",
                )
        finally:
            await conn.close()

    async def test_null_skill_id_accepted(self, pg_schema: tuple[str, str]) -> None:
        """A NULL skill_id is accepted (skill-less wake)."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            await conn.execute(
                "INSERT INTO agent_wake_schedules "
                "(conversation_id, schedule_id, user_id, agent_id, "
                " skill_id, schedule_type) "
                "VALUES ($1, $2, $3, $4, NULL, $5)",
                _new_uuid(),
                _new_uuid(),
                _new_uuid(),
                _new_uuid(),
                "daily_at",
            )
        finally:
            await conn.close()

    async def test_real_skill_id_accepted(self, pg_schema: tuple[str, str]) -> None:
        """A schedule referencing a real skill_id is accepted."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            agent = _new_uuid()
            user = _new_uuid()
            skill_id = _new_uuid()
            await conn.execute(
                "INSERT INTO agent_skills "
                "(agent_id, skill_id, user_id, name, summary, body) "
                "VALUES ($1, $2, $3, $4, $5, $6)",
                agent,
                skill_id,
                user,
                "deploy",
                "deploy",
                "body",
            )
            await conn.execute(
                "INSERT INTO agent_wake_schedules "
                "(conversation_id, schedule_id, user_id, agent_id, "
                " skill_id, schedule_type) "
                "VALUES ($1, $2, $3, $4, $5, $6)",
                _new_uuid(),
                _new_uuid(),
                user,
                agent,
                skill_id,
                "daily_at",
            )
        finally:
            await conn.close()


class TestSubscriptionDefaultSkillIdFk:
    """``webhook_subscriptions.default_skill_id`` FK behaviour."""

    async def test_orphan_default_skill_id_rejected(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """Inserting a subscription with a non-existent default_skill_id fails."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            with pytest.raises(asyncpg.exceptions.ForeignKeyViolationError):
                await conn.execute(
                    "INSERT INTO webhook_subscriptions "
                    "(conversation_id, subscription_id, user_id, agent_id, "
                    " default_skill_id, secret_ciphertext) "
                    "VALUES ($1, $2, $3, $4, $5, $6)",
                    _new_uuid(),
                    _new_uuid(),
                    _new_uuid(),
                    _new_uuid(),
                    _new_uuid(),  # ghost
                    b"\x00",
                )
        finally:
            await conn.close()

    async def test_null_default_skill_id_accepted(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """A NULL default_skill_id is accepted."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            await conn.execute(
                "INSERT INTO webhook_subscriptions "
                "(conversation_id, subscription_id, user_id, agent_id, "
                " default_skill_id, secret_ciphertext) "
                "VALUES ($1, $2, $3, $4, NULL, $5)",
                _new_uuid(),
                _new_uuid(),
                _new_uuid(),
                _new_uuid(),
                b"\x00",
            )
        finally:
            await conn.close()
