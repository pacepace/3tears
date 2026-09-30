"""Integration test: DB-side CHECK constraints fire as expected.

Covers:

- ``agent_skills_prompt_mode_check`` rejects ``prompt_mode`` values
  outside ``('additive', 'replace')``.
- ``agent_skills_payload_check`` rejects rows with ``body IS NULL``
  and both ``tool_additions`` / ``tool_restrictions`` empty.
- ``agent_skill_invocations_outcome_check`` rejects ``outcome`` values
  outside ``('success', 'failure')`` (NULL remains valid).
- ``agent_skill_invocations_source_check`` rejects ``invocation_source``
  outside ``('wake', 'invoke')``.
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
    """Apply conversations + skills migrations."""
    await conn.execute(f'SET search_path TO "{schema}", public')
    runner = MigrationRunner()
    register_conversations(runner)
    register_skills(runner)
    store = AsyncpgStore(conn)
    await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]


class TestSkillCheckConstraints:
    """CHECK constraints on ``agent_skills``."""

    async def test_prompt_mode_enum_rejected(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """Inserting ``prompt_mode='invalid'`` raises ``CheckViolationError``."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            with pytest.raises(asyncpg.exceptions.CheckViolationError):
                await conn.execute(
                    "INSERT INTO agent_skills "
                    "(agent_id, skill_id, user_id, name, summary, body, prompt_mode) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7)",
                    _new_uuid(),
                    _new_uuid(),
                    _new_uuid(),
                    "bad-mode",
                    "summary",
                    "body",
                    "invalid",
                )
        finally:
            await conn.close()

    async def test_payload_check_rejects_empty_skill(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """``body IS NULL`` plus empty ``tool_additions`` + ``tool_restrictions`` fails."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            with pytest.raises(asyncpg.exceptions.CheckViolationError):
                await conn.execute(
                    "INSERT INTO agent_skills "
                    "(agent_id, skill_id, user_id, name, summary, body, "
                    " tool_additions, tool_restrictions) "
                    "VALUES ($1, $2, $3, $4, $5, NULL, '{}', '{}')",
                    _new_uuid(),
                    _new_uuid(),
                    _new_uuid(),
                    "empty",
                    "summary",
                )
        finally:
            await conn.close()

    async def test_payload_check_allows_tool_only_skill(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """A skill with no body but one tool_additions entry is accepted."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            await conn.execute(
                "INSERT INTO agent_skills "
                "(agent_id, skill_id, user_id, name, summary, body, "
                " tool_additions, tool_restrictions) "
                "VALUES ($1, $2, $3, $4, $5, NULL, $6, '{}')",
                _new_uuid(),
                _new_uuid(),
                _new_uuid(),
                "tool-only",
                "summary",
                ["mcp.x"],
            )
        finally:
            await conn.close()

    async def test_payload_check_allows_restrictions_only_skill(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """A skill with no body but tool_restrictions entries is accepted."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            await conn.execute(
                "INSERT INTO agent_skills "
                "(agent_id, skill_id, user_id, name, summary, body, "
                " tool_additions, tool_restrictions) "
                "VALUES ($1, $2, $3, $4, $5, NULL, '{}', $6)",
                _new_uuid(),
                _new_uuid(),
                _new_uuid(),
                "restrictions-only",
                "summary",
                ["mcp.dangerous"],
            )
        finally:
            await conn.close()


class TestInvocationCheckConstraints:
    """CHECK constraints on ``agent_skill_invocations``."""

    async def _seed_skill(
        self,
        conn: asyncpg.Connection,
        agent_id: UUID,
        user_id: UUID,
    ) -> UUID:
        """Insert one minimal skill so the FK can be satisfied."""
        skill_id = _new_uuid()
        await conn.execute(
            "INSERT INTO agent_skills "
            "(agent_id, skill_id, user_id, name, summary, body) "
            "VALUES ($1, $2, $3, $4, $5, $6)",
            agent_id,
            skill_id,
            user_id,
            f"skill-{skill_id}",
            "summary",
            "body",
        )
        return skill_id

    async def test_invocation_source_enum_rejected(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """``invocation_source='cron'`` raises ``CheckViolationError``."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            agent_id = _new_uuid()
            user_id = _new_uuid()
            skill_id = await self._seed_skill(conn, agent_id, user_id)
            with pytest.raises(asyncpg.exceptions.CheckViolationError):
                await conn.execute(
                    "INSERT INTO agent_skill_invocations "
                    "(agent_id, invocation_id, skill_id, user_id, conversation_id, "
                    " invocation_source, invoked_at) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7)",
                    agent_id,
                    _new_uuid(),
                    skill_id,
                    user_id,
                    _new_uuid(),
                    "cron",
                    datetime.now(UTC),
                )
        finally:
            await conn.close()

    async def test_outcome_partial_rejected(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """``outcome='partial'`` raises ``CheckViolationError``."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            agent_id = _new_uuid()
            user_id = _new_uuid()
            skill_id = await self._seed_skill(conn, agent_id, user_id)
            with pytest.raises(asyncpg.exceptions.CheckViolationError):
                await conn.execute(
                    "INSERT INTO agent_skill_invocations "
                    "(agent_id, invocation_id, skill_id, user_id, conversation_id, "
                    " invocation_source, invoked_at, outcome) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
                    agent_id,
                    _new_uuid(),
                    skill_id,
                    user_id,
                    _new_uuid(),
                    "invoke",
                    datetime.now(UTC),
                    "partial",
                )
        finally:
            await conn.close()

    async def test_outcome_null_allowed(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """A row with NULL ``outcome`` (no marker present) is accepted."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            agent_id = _new_uuid()
            user_id = _new_uuid()
            skill_id = await self._seed_skill(conn, agent_id, user_id)
            await conn.execute(
                "INSERT INTO agent_skill_invocations "
                "(agent_id, invocation_id, skill_id, user_id, conversation_id, "
                " invocation_source, invoked_at, outcome) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, NULL)",
                agent_id,
                _new_uuid(),
                skill_id,
                user_id,
                _new_uuid(),
                "invoke",
                datetime.now(UTC),
            )
        finally:
            await conn.close()


class TestToolCallCheckConstraints:
    """v004's checks: a row is a body skill or a tool-call skill, never both."""

    async def _insert(
        self, conn: asyncpg.Connection, *, body: str | None, tool: str | None, arguments: str | None
    ) -> None:
        """Insert one skill with the given body / tool / arguments (arguments as JSON text)."""
        await conn.execute(
            "INSERT INTO agent_skills (agent_id, skill_id, user_id, name, summary, body, tool, arguments) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8::text::jsonb)",
            _new_uuid(),
            _new_uuid(),
            _new_uuid(),
            f"skill-{_new_uuid()}",
            "summary",
            body,
            tool,
            arguments,
        )

    @pytest.mark.parametrize(
        ("body", "tool", "arguments", "constraint"),
        [
            ("steps", "loki.query", None, "agent_skills_body_or_tool_check"),
            ("", "loki.query", None, "agent_skills_body_or_tool_check"),
            ("steps", None, '{"q": 1}', "agent_skills_arguments_need_tool_check"),
            (None, "loki.query", "[1, 2]", "agent_skills_arguments_object_check"),
        ],
    )
    async def test_refused(
        self,
        pg_schema: tuple[str, str],
        body: str | None,
        tool: str | None,
        arguments: str | None,
        constraint: str,
    ) -> None:
        """Each refused shape is refused by the check named for it."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            with pytest.raises(asyncpg.exceptions.CheckViolationError) as caught:
                await self._insert(conn, body=body, tool=tool, arguments=arguments)
            assert caught.value.constraint_name == constraint
        finally:
            await conn.close()

    async def test_tool_only_skill_satisfies_payload_check(self, pg_schema: tuple[str, str]) -> None:
        """No body and no tool lists, only a tool: v004's payload check accepts it (v001's refused it)."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            await self._insert(conn, body=None, tool="loki.query", arguments='{"q": "error"}')
            await self._insert(conn, body=None, tool="loki.query", arguments=None)
            count = await conn.fetchval("SELECT COUNT(*) FROM agent_skills WHERE tool = 'loki.query'")
            assert count == 2
        finally:
            await conn.close()
