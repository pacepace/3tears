"""
unit tests for 3tears-agent-tools migration registration and DDL shape.

agent-tools owns the ``context_items`` table (moved from agent-memory
during the migrations-task-01 ownership reshape). these tests verify
the package contributes exactly the expected migration in the
expected scope and order, with the expected ``depends_on`` edge to
the conversations package.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from threetears.agent.tools.migrations import (
    PACKAGE_NAME,
    align_context_items_shape,
    create_context_items_table,
    datetime_to_datetimetz,
    register,
)
from threetears.core.data.migrations import (
    DuplicateVersionError,
    MigrationRunner,
    MigrationScope,
)


class _FakeDataStore:
    """
    in-memory DataStore stub capturing every executed statement.

    same shape as the fakes in conversations / agent-memory test
    suites so the runner exercise mirrors those packages exactly.

    :ivar executed: list of (sql, params) tuples
    :ptype executed: list[tuple[str, tuple[Any, ...]]]
    :ivar migrations_rows: emulated ``_schema_migrations`` rows
    :ptype migrations_rows: list[dict[str, Any]]
    :ivar migrations_table_created: whether the bookkeeping table
        has been materialized
    :ptype migrations_table_created: bool
    """

    def __init__(self) -> None:
        """initialize an empty in-memory store."""
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.migrations_rows: list[dict[str, Any]] = []
        self.migrations_table_created = False

    async def execute(self, sql: str, *params: Any) -> str:
        """
        record an execute call and emulate ``_schema_migrations`` writes.

        :param sql: SQL statement text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: synthetic status string
        :rtype: str
        """
        self.executed.append((sql, params))
        normalized = " ".join(sql.split()).upper()
        result: str
        if "CREATE TABLE IF NOT EXISTS _SCHEMA_MIGRATIONS" in normalized:
            self.migrations_table_created = True
            result = "CREATE TABLE"
            return result
        if normalized.startswith("INSERT INTO _SCHEMA_MIGRATIONS"):
            self.migrations_rows.append(
                {
                    "version": params[0],
                    "package": params[1],
                    "description": params[2],
                }
            )
            result = "INSERT 0 1"
            return result
        result = "EXECUTE"
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """
        emulate the two ``_schema_migrations`` queries the runner issues.

        :param sql: SQL query text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        normalized = " ".join(sql.split()).upper()
        result: list[dict[str, Any]]
        if "SELECT VERSION, PACKAGE FROM _SCHEMA_MIGRATIONS" in normalized:
            result = [{"version": row["version"], "package": row["package"]} for row in self.migrations_rows]
            return result
        if "COALESCE(MAX(VERSION)" in normalized:
            max_version = max((row["version"] for row in self.migrations_rows), default=0)
            result = [{"max_version": max_version}]
            return result
        result = []
        return result


def _joined_executed_sql(store: _FakeDataStore) -> str:
    """
    join every executed statement into a single normalized string.

    :param store: fake data store with execution history
    :ptype store: _FakeDataStore
    :return: joined, whitespace-normalized statements
    :rtype: str
    """
    return "\n".join(" ".join(sql.split()) for sql, _params in store.executed)


class TestRegisterAgentToolsMigrations:
    """tests for the register factory and apply flow."""

    async def test_register_returns_agent_scoped_package(self) -> None:
        """register produces a PackageMigrations scoped to AGENT."""
        runner = MigrationRunner()
        pkg = register(runner)
        assert pkg.name == PACKAGE_NAME
        assert pkg.scope == MigrationScope.AGENT

    async def test_register_depends_on_conversations(self) -> None:
        """conversations is the parent of context_items.conversation_id."""
        runner = MigrationRunner()
        pkg = register(runner)
        assert pkg.depends_on == ("conversations",)

    async def test_register_populates_versions_one_through_three(self) -> None:
        """register attaches v001 (create), v002 (datetime promote),
        v003 (align context_items shape with prod parity).
        """
        runner = MigrationRunner()
        pkg = register(runner)
        assert set(pkg.versions.keys()) == {1, 2, 3}

    async def test_apply_in_isolation_runs_all_versions(self) -> None:
        """
        apply_package runs the package's migrations against a target
        store without resolving dependencies; this is the harness path
        for per-package isolation tests.
        """
        runner = MigrationRunner()
        register(runner)
        store = _FakeDataStore()
        first_count = await runner.apply_package(store, PACKAGE_NAME)
        assert first_count == 3
        assert [row["version"] for row in store.migrations_rows] == [1, 2, 3]

    async def test_apply_emits_context_items_create_statement(self) -> None:
        """the CREATE TABLE statement carries every column and type."""
        runner = MigrationRunner()
        register(runner)
        store = _FakeDataStore()
        await runner.apply_package(store, PACKAGE_NAME)
        joined = _joined_executed_sql(store)
        assert re.search(r"CREATE TABLE IF NOT EXISTS context_items", joined)
        assert re.search(r"conversation_id UUID NOT NULL", joined)
        assert re.search(r"context_id UUID NOT NULL", joined)
        assert "PRIMARY KEY (conversation_id, context_id)" in joined
        assert "context_type VARCHAR(50) NOT NULL" in joined
        assert "key VARCHAR(255) NOT NULL" in joined
        assert "short_desc VARCHAR(200)" in joined
        assert "long_desc VARCHAR(1000)" in joined
        assert "content TEXT" in joined
        assert "metadata JSONB" in joined
        assert re.search(r"date_accessed TIMESTAMP NOT NULL", joined)
        assert re.search(r"date_created TIMESTAMP NOT NULL", joined)
        assert re.search(r"date_updated TIMESTAMP NOT NULL", joined)

    async def test_apply_emits_two_indexes(self) -> None:
        """conversation and conversation+type lookup indexes are emitted."""
        runner = MigrationRunner()
        register(runner)
        store = _FakeDataStore()
        await runner.apply_package(store, PACKAGE_NAME)
        joined = _joined_executed_sql(store)
        assert re.search(
            r"CREATE INDEX IF NOT EXISTS idx_ctx_conversation "
            r"ON context_items \(conversation_id\)",
            joined,
        )
        assert re.search(
            r"CREATE INDEX IF NOT EXISTS idx_ctx_conversation_type "
            r"ON context_items \(conversation_id, context_type\)",
            joined,
        )

    async def test_apply_does_not_qualify_with_schema_name(self) -> None:
        """
        DDL statements stay unqualified; ``search_path`` governs.
        """
        runner = MigrationRunner()
        register(runner)
        store = _FakeDataStore()
        await runner.apply_package(store, PACKAGE_NAME)
        joined = _joined_executed_sql(store)
        assert not re.search(r"agent_[0-9a-f]{32}\.", joined)


class TestDirectMigrationFunction:
    """tests exercising create_context_items_table directly."""

    async def test_direct_call_issues_three_statements(self) -> None:
        """one CREATE TABLE plus two CREATE INDEX statements."""
        store = _FakeDataStore()
        await create_context_items_table(store)  # type: ignore[arg-type]
        assert len(store.executed) == 3

    async def test_direct_call_leaves_migrations_table_untouched(self) -> None:
        """direct invocation does not touch ``_schema_migrations``."""
        store = _FakeDataStore()
        await create_context_items_table(store)  # type: ignore[arg-type]
        assert store.migrations_table_created is False
        assert store.migrations_rows == []


class TestDatetimeToDatetimetzMigration:
    """
    tests for v002: TIMESTAMP -> TIMESTAMPTZ promotion of every
    datetime column on ``context_items``.

    collections-task-05 requires every per-column ALTER to appear as a
    literal SQL string (not a templated DO block iterating a list) so
    the column-type-alignment AST walker in
    ``packages/core/tests/enforcement/test_column_type_alignment.py``
    can match each ``(table, column) -> TIMESTAMPTZ`` pair against its
    ``Column(..., DATETIMETZ_TYPE, ...)`` declaration in
    ``collections.py``. these tests pin that pattern so a future
    refactor cannot regress it.
    """

    async def test_direct_call_issues_three_per_column_alters(self) -> None:
        """one DO block per (table, column) pair: 3 statements."""
        store = _FakeDataStore()
        await datetime_to_datetimetz(store)  # type: ignore[arg-type]
        assert len(store.executed) == 3

    async def test_direct_call_targets_every_datetime_column(self) -> None:
        """every datetime column on context_items has its own ALTER."""
        store = _FakeDataStore()
        await datetime_to_datetimetz(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert "ALTER TABLE context_items ALTER COLUMN date_accessed TYPE TIMESTAMPTZ" in joined
        assert "ALTER TABLE context_items ALTER COLUMN date_created TYPE TIMESTAMPTZ" in joined
        assert "ALTER TABLE context_items ALTER COLUMN date_updated TYPE TIMESTAMPTZ" in joined

    async def test_direct_call_uses_at_time_zone_utc(self) -> None:
        """every ALTER asserts UTC semantics on the bare TIMESTAMP cell."""
        store = _FakeDataStore()
        await datetime_to_datetimetz(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert "USING date_accessed AT TIME ZONE 'UTC'" in joined
        assert "USING date_created AT TIME ZONE 'UTC'" in joined
        assert "USING date_updated AT TIME ZONE 'UTC'" in joined

    async def test_direct_call_is_guarded_by_information_schema(self) -> None:
        """each ALTER lives inside a DO block that probes data_type."""
        store = _FakeDataStore()
        await datetime_to_datetimetz(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert joined.count("information_schema.columns") == 3
        assert joined.count("'timestamp without time zone'") == 3

    async def test_direct_call_leaves_migrations_table_untouched(self) -> None:
        """direct invocation does not touch ``_schema_migrations``."""
        store = _FakeDataStore()
        await datetime_to_datetimetz(store)  # type: ignore[arg-type]
        assert store.migrations_table_created is False
        assert store.migrations_rows == []


class TestAlignContextItemsShapeMigration:
    """tests for v003: align ``context_items`` shape with prod parity.

    v003 drops the v001 legacy indexes, creates the four v0.8.0
    indexes (matching prod metallm), promotes ``long_desc`` to NOT
    NULL DEFAULT '', and adds the FK on ``conversation_id`` -
    everything required for the v0.8.0 parity gate to stay clean.
    """

    async def test_direct_call_drops_legacy_indexes(self) -> None:
        """v003 drops the v001 ``idx_ctx_*`` indexes that diverge from prod."""
        store = _FakeDataStore()
        await align_context_items_shape(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert "DROP INDEX IF EXISTS idx_ctx_conversation" in joined
        assert "DROP INDEX IF EXISTS idx_ctx_conversation_type" in joined

    async def test_direct_call_creates_v080_indexes(self) -> None:
        """v003 creates the four v0.8.0 indexes matching prod."""
        store = _FakeDataStore()
        await align_context_items_shape(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert "CREATE INDEX IF NOT EXISTS ix_context_items_conv" in joined
        assert "CREATE INDEX IF NOT EXISTS ix_context_items_type" in joined
        assert "CREATE INDEX IF NOT EXISTS ix_context_items_lru" in joined
        assert "CREATE UNIQUE INDEX IF NOT EXISTS ix_context_items_var_key" in joined
        # the var_key index is partial-unique on context_type=variable
        assert "WHERE context_type = 'variable'" in joined

    async def test_direct_call_backfills_and_promotes_long_desc(self) -> None:
        """v003 backfills NULL long_desc -> '' then promotes NOT NULL."""
        store = _FakeDataStore()
        await align_context_items_shape(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert "UPDATE context_items SET long_desc = '' WHERE long_desc IS NULL" in joined
        # the SET NOT NULL is inside a DO block; the inner ALTER is
        # what matters for the parity gate.
        assert "ALTER TABLE context_items ALTER COLUMN long_desc SET DEFAULT ''" in joined
        assert "ALTER TABLE context_items ALTER COLUMN long_desc SET NOT NULL" in joined

    async def test_direct_call_drops_legacy_conversation_fk(self) -> None:
        """v003 drops any legacy ``fk_context_items_conversation`` FK.

        See v003 module docstring "FK decision" -- the 3tears
        ``conversations`` table has composite PK
        ``(agent_id, conversation_id)`` and ``context_items`` lacks
        ``agent_id``, so no FK shape is legal. Earlier drafts of
        v003 added a single-column FK that the v0.8.0 shard 04.6
        rename surfaced as illegal. The migration now drops the FK
        by name (idempotent via ``IF EXISTS``) so any agent schema
        that ran an earlier v003 draft converges to the FK-free
        shape.
        """
        store = _FakeDataStore()
        await align_context_items_shape(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert "DROP CONSTRAINT IF EXISTS fk_context_items_conversation" in joined
        # explicitly assert the migration does NOT try to add the
        # FK -- a regression here would mean the composite-PK +
        # missing-agent_id combination breaks again.
        assert "ADD CONSTRAINT fk_context_items_conversation" not in joined

    async def test_direct_call_leaves_migrations_table_untouched(self) -> None:
        """direct invocation does not touch ``_schema_migrations``."""
        store = _FakeDataStore()
        await align_context_items_shape(store)  # type: ignore[arg-type]
        assert store.migrations_table_created is False
        assert store.migrations_rows == []


class TestDuplicateVersionGuard:
    """tests confirming the runner rejects duplicate version registration."""

    async def test_duplicate_version_registration_raises(self) -> None:
        """registering a second callable at v1 raises DuplicateVersionError."""
        runner = MigrationRunner()
        pkg = register(runner)
        with pytest.raises(DuplicateVersionError):
            pkg.version(1)(create_context_items_table)
