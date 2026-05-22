"""Integration test: deleting a skill cascade-deletes its invocations.

The composite FK
``agent_skill_invocations(agent_id, skill_id) REFERENCES
agent_skills(agent_id, skill_id) ON DELETE CASCADE`` ensures the
invocation rows are removed in the same transaction.

Also verifies that inserting an invocation whose ``(agent_id,
skill_id)`` pair does not exist on ``agent_skills`` raises a
``ForeignKeyViolationError`` -- the composite FK shape is the
defensive layer that catches orphan inserts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import asyncpg
import pytest
from uuid_utils import uuid7

from threetears.agent.skills.migrations import register as register_skills
from threetears.conversations.migrations import register as register_conversations
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration


def _new_uuid() -> UUID:
    """Return a fresh UUIDv7 cast to stdlib ``UUID``."""
    return UUID(str(uuid7()))


async def _apply(conn: asyncpg.Connection, schema: str) -> None:
    """Apply conversations + skills migrations to the schema."""
    await conn.execute(f'SET search_path TO "{schema}", public')
    runner = MigrationRunner()
    register_conversations(runner)
    register_skills(runner)
    store = AsyncpgStore(conn)
    await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]


async def _insert_skill_minimal(
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


async def _insert_invocation(
    conn: asyncpg.Connection,
    *,
    agent_id: UUID,
    skill_id: UUID,
    user_id: UUID,
    conversation_id: UUID,
    invocation_source: str,
) -> UUID:
    """Insert an invocation row and return its id."""
    invocation_id = _new_uuid()
    await conn.execute(
        "INSERT INTO agent_skill_invocations "
        "(agent_id, invocation_id, skill_id, user_id, conversation_id, "
        " invocation_source, invoked_at) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7)",
        agent_id,
        invocation_id,
        skill_id,
        user_id,
        conversation_id,
        invocation_source,
        datetime.now(UTC),
    )
    return invocation_id


class TestSkillDeleteCascadesInvocations:
    """Deleting a skill removes every invocation row for that skill."""

    async def test_delete_cascades(self, pg_schema: tuple[str, str]) -> None:
        """Delete one skill; observe its invocations have vanished."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            agent_id = _new_uuid()
            user_id = _new_uuid()
            conversation_id = _new_uuid()

            skill_id = await _insert_skill_minimal(
                conn,
                agent_id=agent_id,
                user_id=user_id,
            )
            inv_a = await _insert_invocation(
                conn,
                agent_id=agent_id,
                skill_id=skill_id,
                user_id=user_id,
                conversation_id=conversation_id,
                invocation_source="invoke",
            )
            inv_b = await _insert_invocation(
                conn,
                agent_id=agent_id,
                skill_id=skill_id,
                user_id=user_id,
                conversation_id=conversation_id,
                invocation_source="wake",
            )

            await conn.execute(
                "DELETE FROM agent_skills WHERE agent_id = $1 AND skill_id = $2",
                agent_id,
                skill_id,
            )

            for inv_id in (inv_a, inv_b):
                row = await conn.fetchrow(
                    "SELECT invocation_id FROM agent_skill_invocations WHERE invocation_id = $1",
                    inv_id,
                )
                assert row is None, f"invocation {inv_id} should have been cascade-deleted"
        finally:
            await conn.close()

    async def test_unrelated_skill_invocations_survive(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """Deleting one skill leaves invocations of another skill intact."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            agent_id = _new_uuid()
            user_id = _new_uuid()
            conversation_id = _new_uuid()

            skill_a = await _insert_skill_minimal(
                conn,
                agent_id=agent_id,
                user_id=user_id,
            )
            skill_b = await _insert_skill_minimal(
                conn,
                agent_id=agent_id,
                user_id=user_id,
            )
            await _insert_invocation(
                conn,
                agent_id=agent_id,
                skill_id=skill_a,
                user_id=user_id,
                conversation_id=conversation_id,
                invocation_source="invoke",
            )
            survivor_inv = await _insert_invocation(
                conn,
                agent_id=agent_id,
                skill_id=skill_b,
                user_id=user_id,
                conversation_id=conversation_id,
                invocation_source="invoke",
            )

            await conn.execute(
                "DELETE FROM agent_skills WHERE agent_id = $1 AND skill_id = $2",
                agent_id,
                skill_a,
            )

            row = await conn.fetchrow(
                "SELECT invocation_id FROM agent_skill_invocations WHERE invocation_id = $1",
                survivor_inv,
            )
            assert row is not None
        finally:
            await conn.close()


class TestOrphanInvocationRejected:
    """A composite FK violation on insert raises ``ForeignKeyViolationError``."""

    async def test_orphan_skill_id_rejected(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """Inserting an invocation pointing at a non-existent skill fails."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            agent_id = _new_uuid()
            user_id = _new_uuid()
            conversation_id = _new_uuid()
            ghost_skill_id = _new_uuid()

            with pytest.raises(asyncpg.exceptions.ForeignKeyViolationError):
                await _insert_invocation(
                    conn,
                    agent_id=agent_id,
                    skill_id=ghost_skill_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    invocation_source="invoke",
                )
        finally:
            await conn.close()
