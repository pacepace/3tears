"""Integration test: agent-wake migrations apply cleanly.

Verifies that running v001 + v002 + v003 against a fresh schema
(after the conversations + agent-skills migrations have applied --
declared via ``depends_on``):

- Creates ``agent_wake_schedules`` + ``wake_fires`` +
  ``webhook_subscriptions`` with the expected column inventory and
  indexes.
- Installs the cross-package FKs to ``agent_skills`` and the
  retro-added FK on ``wake_fires.webhook_subscription_id``.
- Is idempotent on re-apply.
- Topological ordering: conversations + agent-skills migrations run
  before any wake migration (the runner's
  ``apply_for_agent_schema`` walks ``depends_on`` and orders by
  declared dependencies).
"""

from __future__ import annotations

import asyncpg
import pytest

from threetears.agent.skills.migrations import register as register_skills
from threetears.agent.wake.migrations import register as register_wake
from threetears.conversations.migrations import register as register_conversations
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration


def _build_runner() -> MigrationRunner:
    """Register conversations + agent-skills + agent-wake on a fresh runner.

    All three are required because wake declares
    ``depends_on=("conversations", "agent_skills")``.
    """
    runner = MigrationRunner()
    register_conversations(runner)
    register_skills(runner)
    register_wake(runner)
    return runner


async def _columns(
    conn: asyncpg.Connection,
    schema: str,
    table: str,
) -> dict[str, str]:
    """Return ``column_name -> data_type`` for the named table."""
    rows = await conn.fetch(
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = $1 AND table_name = $2",
        schema,
        table,
    )
    return {r["column_name"]: r["data_type"] for r in rows}


async def _index_exists(
    conn: asyncpg.Connection,
    schema: str,
    index_name: str,
) -> bool:
    """Return whether ``schema.index_name`` exists."""
    row = await conn.fetchrow(
        "SELECT 1 FROM pg_indexes WHERE schemaname = $1 AND indexname = $2",
        schema,
        index_name,
    )
    return row is not None


async def _constraint_exists(
    conn: asyncpg.Connection,
    schema: str,
    constraint_name: str,
) -> bool:
    """Return whether ``schema.constraint_name`` exists in ``pg_constraint``."""
    row = await conn.fetchrow(
        """
        SELECT 1 FROM pg_constraint c
          JOIN pg_namespace ns ON ns.oid = c.connamespace
         WHERE ns.nspname = $1 AND c.conname = $2
        """,
        schema,
        constraint_name,
    )
    return row is not None


class TestSchemaShape:
    """The v001 + v002 + v003 chain produces the documented schema."""

    async def test_migration_applies_and_creates_tables(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """All three tables + every documented column exist after apply."""
        url, schema = pg_schema
        runner = _build_runner()
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = AsyncpgStore(conn)
            count = await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
            assert count > 0

            sched_cols = await _columns(conn, schema, "agent_wake_schedules")
            expected_sched = {
                "conversation_id",
                "schedule_id",
                "user_id",
                "agent_id",
                "skill_id",
                "schedule_type",
                "schedule_config",
                "task_prompt",
                "execution_mode",
                "status",
                "next_fire_at",
                "last_fired_at",
                "name",
                "missed_fire_policy",
                "context_from_schedule_id",
                "date_created",
                "date_updated",
            }
            assert expected_sched.issubset(sched_cols.keys())

            fire_cols = await _columns(conn, schema, "wake_fires")
            expected_fire = {
                "conversation_id",
                "fire_id",
                "schedule_id",
                "webhook_subscription_id",
                "scheduled_fire_at",
                "actual_fired_at",
                "status",
                "display_suppressed",
                "output_text",
                "latency_ms",
                "error",
                "date_created",
            }
            assert expected_fire.issubset(fire_cols.keys())

            sub_cols = await _columns(conn, schema, "webhook_subscriptions")
            expected_sub = {
                "conversation_id",
                "subscription_id",
                "user_id",
                "agent_id",
                "default_skill_id",
                "name",
                "secret_ciphertext",
                "allowed_source_pattern",
                "execution_mode",
                "task_prompt_template",
                "verification_scheme",
                "status",
                "rate_limit_per_minute",
                "last_fired_at",
                "date_created",
                "date_updated",
            }
            assert expected_sub.issubset(sub_cols.keys())
        finally:
            await conn.close()

    async def test_indexes_present(self, pg_schema: tuple[str, str]) -> None:
        """Every named index exists after apply."""
        url, schema = pg_schema
        runner = _build_runner()
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = AsyncpgStore(conn)
            await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
            for index_name in (
                "idx_wake_schedules_next_fire",
                "idx_wake_schedules_conv_status",
                "idx_wake_schedules_user",
                "idx_wake_schedules_context_from",
                "idx_wake_fires_schedule_time",
                "idx_wake_fires_webhook_time",
                "idx_wake_fires_conv_time",
                "idx_webhook_subs_conv",
                "idx_webhook_subs_user",
            ):
                assert await _index_exists(conn, schema, index_name), index_name
        finally:
            await conn.close()

    async def test_check_constraints_present(self, pg_schema: tuple[str, str]) -> None:
        """All declared CHECK / FK constraints exist after apply."""
        url, schema = pg_schema
        runner = _build_runner()
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = AsyncpgStore(conn)
            await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
            for constraint in (
                "agent_wake_schedules_skill_fk",
                "agent_wake_schedules_context_from_fk",
                "agent_wake_schedules_execution_mode_check",
                "agent_wake_schedules_status_check",
                "agent_wake_schedules_missed_fire_policy_check",
                "wake_fires_schedule_fk",
                "wake_fires_one_source_check",
                "wake_fires_status_check",
                "wake_fires_webhook_subscription_fk",
                "webhook_subscriptions_default_skill_fk",
                "webhook_subscriptions_execution_mode_check",
                "webhook_subscriptions_verification_scheme_check",
                "webhook_subscriptions_status_check",
            ):
                assert await _constraint_exists(conn, schema, constraint), constraint
        finally:
            await conn.close()

    async def test_standalone_unique_on_schedule_id(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """``UNIQUE (schedule_id)`` exists so cross-package FKs can reference the bare id."""
        url, schema = pg_schema
        runner = _build_runner()
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = AsyncpgStore(conn)
            await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
            row = await conn.fetchrow(
                """
                SELECT 1 FROM pg_constraint c
                  JOIN pg_namespace ns ON ns.oid = c.connamespace
                 WHERE ns.nspname = $1
                   AND c.conrelid::regclass::text IN ('agent_wake_schedules', $2 || '.agent_wake_schedules')
                   AND c.contype = 'u'
                   AND array_length(c.conkey, 1) = 1
                """,
                schema,
                schema,
            )
            assert row is not None, "expected a standalone UNIQUE on agent_wake_schedules"
        finally:
            await conn.close()

    async def test_standalone_unique_on_subscription_id(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """``UNIQUE (subscription_id)`` exists so the receiver can look up by bare id."""
        url, schema = pg_schema
        runner = _build_runner()
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = AsyncpgStore(conn)
            await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
            row = await conn.fetchrow(
                """
                SELECT 1 FROM pg_constraint c
                  JOIN pg_namespace ns ON ns.oid = c.connamespace
                 WHERE ns.nspname = $1
                   AND c.conrelid::regclass::text IN ('webhook_subscriptions', $2 || '.webhook_subscriptions')
                   AND c.contype = 'u'
                   AND array_length(c.conkey, 1) = 1
                """,
                schema,
                schema,
            )
            assert row is not None, "expected a standalone UNIQUE on webhook_subscriptions"
        finally:
            await conn.close()


class TestIdempotency:
    """Re-applying the migration chain is a no-op."""

    async def test_re_apply_is_no_op(self, pg_schema: tuple[str, str]) -> None:
        """The second ``apply_for_agent_schema`` returns 0 applied."""
        url, schema = pg_schema
        runner = _build_runner()
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = AsyncpgStore(conn)
            first = await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
            assert first > 0
            second = await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
            assert second == 0
        finally:
            await conn.close()
