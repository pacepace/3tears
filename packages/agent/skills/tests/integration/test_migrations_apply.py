"""Integration test: agent-skills migrations apply cleanly.

Verifies that running v001-v004 against a fresh schema:

- Creates ``agent_skills`` + ``agent_skill_invocations`` with the
  expected column inventory and indexes.
- Leaves no GIN index on ``agent_skills``: v003 drops the two v001
  created, because every predicate that could use them goes through
  ``gin_filter``.
- Installs the FTS trigger function + trigger.
- Is idempotent on re-apply.
- Produces zero rows on the first ``apply_for_agent_schema`` after
  initial apply (SK-07 alembic-autogenerate parity equivalent at the
  runner level).
"""

from __future__ import annotations

import asyncpg
import pytest

from threetears.agent.skills.migrations import add_tool_call_columns, drop_gin_indexes
from threetears.agent.skills.migrations.v004_add_tool_call_columns import PAYLOAD_CHECK_ENGINE_DEF
from threetears.agent.skills.migrations import register as register_skills
from threetears.conversations.migrations import register as register_conversations
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration

#: the v001 GIN indexes v003 drops.
_DROPPED_GIN_INDEXES: tuple[str, ...] = ("idx_skills_search_vector", "idx_skills_tags")


def _build_runner() -> MigrationRunner:
    """Register conversations + agent-skills on a fresh runner.

    Conversations is required because skills declares
    ``depends_on=("conversations",)``.
    """
    runner = MigrationRunner()
    register_conversations(runner)
    register_skills(runner)
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


async def _trigger_exists(
    conn: asyncpg.Connection,
    schema: str,
    trigger_name: str,
) -> bool:
    """Return whether the named trigger exists on any table in the schema."""
    row = await conn.fetchrow(
        """
        SELECT 1 FROM information_schema.triggers
         WHERE trigger_schema = $1 AND trigger_name = $2
        """,
        schema,
        trigger_name,
    )
    return row is not None


class TestSchemaShape:
    """The v001-v004 chain produces the documented schema."""

    async def test_migration_applies_and_creates_tables(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """Both tables + every documented column exist after apply."""
        url, schema = pg_schema
        runner = _build_runner()
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = AsyncpgStore(conn)
            count = await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
            assert count > 0

            skill_cols = await _columns(conn, schema, "agent_skills")
            expected_skill_cols = {
                "agent_id",
                "skill_id",
                "user_id",
                "name",
                "summary",
                "body",
                "tool",
                "arguments",
                "prompt_mode",
                "tool_additions",
                "tool_restrictions",
                "trigger_keywords",
                "tags",
                "source",
                "enabled",
                "use_count",
                "last_used_at",
                "success_count",
                "failure_count",
                "last_failure_at",
                "date_created",
                "date_updated",
                "search_vector",
            }
            assert expected_skill_cols.issubset(skill_cols.keys())

            invocation_cols = await _columns(conn, schema, "agent_skill_invocations")
            expected_invocation_cols = {
                "agent_id",
                "invocation_id",
                "skill_id",
                "user_id",
                "conversation_id",
                "message_id",
                "invocation_source",
                "invoked_at",
                "outcome",
                "outcome_source",
                "notes",
            }
            assert expected_invocation_cols.issubset(invocation_cols.keys())
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
                "uq_skills_agent_user_name",
                "idx_skills_agent_user_enabled",
                "idx_skill_invocations_skill_time",
                "idx_skill_invocations_conv",
            ):
                assert await _index_exists(conn, schema, index_name), index_name
            for index_name in _DROPPED_GIN_INDEXES:
                assert not await _index_exists(conn, schema, index_name), index_name
        finally:
            await conn.close()

    async def test_v003_drops_both_gin_indexes(self, pg_schema: tuple[str, str]) -> None:
        """A schema migrated to v002 carries both GIN indexes; v003 removes them.

        The v002 checkpoint is the non-vacuity guard: it proves each name
        asserted absent afterwards really existed on an upgraded schema. The
        ``search_vector`` column and its FTS trigger survive the drop.
        """
        url, schema = pg_schema
        runner = _build_runner()
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = AsyncpgStore(conn)
            await runner.apply_for_agent_schema(store, target=2)  # type: ignore[arg-type]
            for index_name in _DROPPED_GIN_INDEXES:
                assert await _index_exists(conn, schema, index_name), f"{index_name} missing at v002"

            applied = await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
            assert applied >= 1

            for index_name in _DROPPED_GIN_INDEXES:
                assert not await _index_exists(conn, schema, index_name), f"{index_name} survived v003"
            assert "search_vector" in await _columns(conn, schema, "agent_skills")
            assert await _trigger_exists(conn, schema, "trg_agent_skills_search_vector")

            # the body tolerates indexes that are already gone.
            await drop_gin_indexes(store)  # type: ignore[arg-type]
        finally:
            await conn.close()

    async def test_fts_trigger_installed(self, pg_schema: tuple[str, str]) -> None:
        """The FTS trigger is installed on ``agent_skills``."""
        url, schema = pg_schema
        runner = _build_runner()
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = AsyncpgStore(conn)
            await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
            assert await _trigger_exists(
                conn,
                schema,
                "trg_agent_skills_search_vector",
            )
        finally:
            await conn.close()

    async def test_composite_fk_present(self, pg_schema: tuple[str, str]) -> None:
        """The composite FK from invocations to skills exists with CASCADE."""
        url, schema = pg_schema
        runner = _build_runner()
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = AsyncpgStore(conn)
            await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
            assert await _constraint_exists(
                conn,
                schema,
                "agent_skill_invocations_skill_fk",
            )
            # confdeltype 'c' = CASCADE
            row = await conn.fetchrow(
                """
                SELECT confdeltype FROM pg_constraint c
                  JOIN pg_namespace ns ON ns.oid = c.connamespace
                 WHERE ns.nspname = $1
                   AND c.conname = 'agent_skill_invocations_skill_fk'
                """,
                schema,
            )
            assert row is not None
            assert row["confdeltype"] == b"c"
        finally:
            await conn.close()

    async def test_standalone_unique_on_skill_id(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """``UNIQUE (skill_id)`` exists so wake-side FKs can reference the bare column."""
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
                   AND c.conrelid::regclass::text IN ('agent_skills', $2 || '.agent_skills')
                   AND c.contype = 'u'
                   AND array_length(c.conkey, 1) = 1
                """,
                schema,
                schema,
            )
            assert row is not None, "expected a standalone UNIQUE constraint covering one column"
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


#: the constraints v004 adds or replaces.
_V004_CHECKS: tuple[str, ...] = (
    "agent_skills_body_or_tool_check",
    "agent_skills_arguments_need_tool_check",
    "agent_skills_arguments_object_check",
    "agent_skills_payload_check",
)


async def _check_oids(conn: asyncpg.Connection, schema: str) -> dict[str, tuple[int, str]]:
    """Return ``name -> (oid, definition)`` for v004's checks in ``schema``."""
    rows = await conn.fetch(
        """
        SELECT c.conname, c.oid::bigint AS oid, pg_get_constraintdef(c.oid) AS def
          FROM pg_constraint c
          JOIN pg_namespace ns ON ns.oid = c.connamespace
         WHERE ns.nspname = $1 AND c.conname = ANY($2::text[])
        """,
        schema,
        list(_V004_CHECKS),
    )
    return {r["conname"]: (r["oid"], r["def"]) for r in rows}


class TestV004ToolCallColumns:
    """v004 adds ``tool`` / ``arguments`` and the checks, safely and repeatably."""

    async def test_upgrade_leaves_existing_skills_untouched(self, pg_schema: tuple[str, str]) -> None:
        """A skill written at v003 reads back identical after v004, with no tool and no arguments."""
        url, schema = pg_schema
        runner = _build_runner()
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = AsyncpgStore(conn)
            await runner.apply_for_agent_schema(store, target=3)  # type: ignore[arg-type]
            assert "tool" not in await _columns(conn, schema, "agent_skills")
            await conn.execute(
                "INSERT INTO agent_skills (agent_id, skill_id, user_id, name, summary, body, tool_additions) "
                "VALUES (gen_random_uuid(), gen_random_uuid(), gen_random_uuid(), 'old', 'sum', 'steps', '{a.b}'), "
                "(gen_random_uuid(), gen_random_uuid(), gen_random_uuid(), 'bare', 'sum', NULL, '{a.b}')"
            )
            before = [dict(r) for r in await conn.fetch("SELECT * FROM agent_skills ORDER BY name")]

            assert await runner.apply_for_agent_schema(store) >= 1  # type: ignore[arg-type]

            after = [dict(r) for r in await conn.fetch("SELECT * FROM agent_skills ORDER BY name")]
            assert [{k: v for k, v in r.items() if k not in {"tool", "arguments"}} for r in after] == before
            assert all(r["tool"] is None and r["arguments"] is None for r in after)
            columns = await _columns(conn, schema, "agent_skills")
            assert columns["tool"] == "text"
            assert columns["arguments"] == "jsonb"
        finally:
            await conn.close()

    async def test_applies_twice_without_change(self, pg_schema: tuple[str, str]) -> None:
        """Running v004's body again keeps every check, with the same OID and definition."""
        url, schema = pg_schema
        runner = _build_runner()
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = AsyncpgStore(conn)
            await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
            first = await _check_oids(conn, schema)
            assert set(first) == set(_V004_CHECKS)
            assert "tool IS NOT NULL" in first["agent_skills_payload_check"][1]

            await add_tool_call_columns(store)  # type: ignore[arg-type]
            assert await _check_oids(conn, schema) == first
        finally:
            await conn.close()

    async def test_payload_check_stored_as_declared(self, pg_schema: tuple[str, str]) -> None:
        """The swap's short-circuit compares against the definition Postgres actually stores."""
        url, schema = pg_schema
        runner = _build_runner()
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            await runner.apply_for_agent_schema(AsyncpgStore(conn))  # type: ignore[arg-type]
            stored = (await _check_oids(conn, schema))["agent_skills_payload_check"][1]
            assert stored == PAYLOAD_CHECK_ENGINE_DEF
        finally:
            await conn.close()

    async def test_each_agent_schema_gets_its_own_checks(self, pg_url: str) -> None:
        """A sibling schema already at v004 does not make this schema's probes skip.

        Every catalog probe is ``current_schema()``-scoped. Unscoped, the
        second schema would see the first one's constraints and add none.
        """
        schemas = [f"sk_v004_{_i}_{id(object())}" for _i in range(2)]
        conn = await asyncpg.connect(pg_url)
        try:
            for schema in schemas:
                await conn.execute(f'CREATE SCHEMA "{schema}"')
                await conn.execute(f'SET search_path TO "{schema}", public')
                await _build_runner().apply_for_agent_schema(AsyncpgStore(conn))  # type: ignore[arg-type]
            for schema in schemas:
                checks = await _check_oids(conn, schema)
                assert set(checks) == set(_V004_CHECKS), schema
                assert "tool IS NOT NULL" in checks["agent_skills_payload_check"][1], schema
        finally:
            for schema in schemas:
                await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            await conn.close()
