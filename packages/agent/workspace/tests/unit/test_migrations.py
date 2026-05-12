"""unit tests for workspace migration registration and SQL shape."""

from __future__ import annotations

import re
from typing import Any

import pytest

from threetears.agent.workspace.migrations import (
    PACKAGE_NAME,
    add_date_deleted_column,
    create_workspace_tables,
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
    in-memory DataStore stub that captures executed SQL.

    mirrors the DataStore.execute and DataStore.query surface used by
    MigrationRunner. state lives in three members: executed captures
    every execute call's (sql, params); migrations_rows emulates the
    _schema_migrations bookkeeping row set; migrations_table_created
    flips the first time the runner issues the CREATE TABLE.
    """

    def __init__(self) -> None:
        """initialize empty execution log and migrations tracker."""
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.migrations_rows: list[dict[str, Any]] = []
        self.migrations_table_created = False

    async def execute(self, sql: str, *params: Any) -> str:
        """
        record SQL execution and emulate _schema_migrations side effects.

        :param sql: SQL statement string
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
        # treat user migrations as no-op for capture purposes
        result = "EXECUTE"
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """
        emulate DataStore.query for the two statements MigrationRunner issues.

        :param sql: SQL query string
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
    join every captured SQL statement into a single normalized string.

    :param store: fake data store with execution history
    :ptype store: _FakeDataStore
    :return: single-line string of all statements separated by newlines
    :rtype: str
    """
    return "\n".join(" ".join(sql.split()) for sql, _params in store.executed)


class TestRegisterWorkspaceMigrations:
    """tests for the register factory and apply flow."""

    async def test_register_returns_package_with_versions_one_through_four(self) -> None:
        """register populates the PackageMigrations with versions 1, 2, 3, and 4."""
        runner = MigrationRunner()
        pkg = register(runner)
        assert pkg.name == PACKAGE_NAME
        assert pkg.scope == MigrationScope.AGENT
        assert set(pkg.versions.keys()) == {1, 2, 3, 4}

    async def test_apply_runs_all_versions_then_idempotent(self) -> None:
        """apply records v1-v4 in _schema_migrations and runs no second time."""
        runner = MigrationRunner()
        register(runner)
        store = _FakeDataStore()
        first_count = await runner.apply_for_agent_schema(store)
        assert first_count == 4
        assert store.migrations_table_created is True
        assert [row["version"] for row in store.migrations_rows] == [1, 2, 3, 4]
        second_count = await runner.apply_for_agent_schema(store)
        assert second_count == 0

    async def test_apply_emits_v2_add_date_deleted_column(self) -> None:
        """v2 migration emits ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS date_deleted."""
        runner = MigrationRunner()
        register(runner)
        store = _FakeDataStore()
        await runner.apply_for_agent_schema(store)
        joined = _joined_executed_sql(store)
        assert re.search(
            r"ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS date_deleted TIMESTAMP NULL",
            joined,
        )

    async def test_register_records_package_name(self) -> None:
        """registered package is retrievable via the canonical name."""
        runner = MigrationRunner()
        pkg = register(runner)
        assert pkg.name == "agent_workspace"

    async def test_apply_emits_workspaces_create_statement(self) -> None:
        """workspaces CREATE TABLE statement contains every required column and constraint."""
        runner = MigrationRunner()
        register(runner)
        store = _FakeDataStore()
        await runner.apply_for_agent_schema(store)
        joined = _joined_executed_sql(store)
        assert re.search(r"CREATE TABLE IF NOT EXISTS workspaces", joined)
        assert "id UUID PRIMARY KEY" in joined
        assert re.search(r"agent_id UUID NOT NULL", joined)
        assert re.search(r"name VARCHAR\(255\) NOT NULL", joined)
        assert "description TEXT" in joined
        assert "template_name VARCHAR(255)" in joined
        assert re.search(r"created_by UUID NOT NULL", joined)
        assert re.search(r"current_version INTEGER NOT NULL DEFAULT 0", joined)
        assert re.search(r"date_created TIMESTAMP NOT NULL", joined)
        assert re.search(r"date_updated TIMESTAMP NOT NULL", joined)
        assert re.search(r"UNIQUE \(agent_id, name\)", joined)

    async def test_apply_emits_workspace_files_create_statement(self) -> None:
        """workspace_files CREATE TABLE statement has BYTEA content, FK, UNIQUE, and index."""
        runner = MigrationRunner()
        register(runner)
        store = _FakeDataStore()
        await runner.apply_for_agent_schema(store)
        joined = _joined_executed_sql(store)
        assert re.search(r"CREATE TABLE IF NOT EXISTS workspace_files", joined)
        assert re.search(
            r"workspace_id UUID NOT NULL REFERENCES workspaces\(id\) ON DELETE CASCADE",
            joined,
        )
        assert "relative_path VARCHAR(512) NOT NULL" in joined
        assert "content BYTEA NOT NULL" in joined
        assert "sha256 CHAR(64) NOT NULL" in joined
        assert "version INTEGER NOT NULL" in joined
        assert "date_updated TIMESTAMP NOT NULL" in joined
        assert re.search(r"UNIQUE \(workspace_id, relative_path\)", joined)
        assert re.search(
            r"CREATE INDEX IF NOT EXISTS idx_workspace_files_workspace ON workspace_files \(workspace_id\)",
            joined,
        )

    async def test_apply_emits_workspace_file_versions_create_statement(self) -> None:
        """workspace_file_versions CREATE TABLE has journal schema with triple UNIQUE and history index."""
        runner = MigrationRunner()
        register(runner)
        store = _FakeDataStore()
        await runner.apply_for_agent_schema(store)
        joined = _joined_executed_sql(store)
        assert re.search(r"CREATE TABLE IF NOT EXISTS workspace_file_versions", joined)
        assert re.search(
            r"workspace_id UUID NOT NULL REFERENCES workspaces\(id\) ON DELETE CASCADE",
            joined,
        )
        assert "relative_path VARCHAR(512) NOT NULL" in joined
        assert "version INTEGER NOT NULL" in joined
        assert "content BYTEA NOT NULL" in joined
        assert "sha256 CHAR(64) NOT NULL" in joined
        assert "action VARCHAR(32) NOT NULL" in joined
        assert "label VARCHAR(255)" in joined
        assert re.search(r"actor_id UUID NOT NULL", joined)
        assert re.search(r"correlation_id UUID NOT NULL", joined)
        assert re.search(r"UNIQUE \(workspace_id, relative_path, version\)", joined)
        assert re.search(
            r"CREATE INDEX IF NOT EXISTS idx_workspace_file_versions_history ON workspace_file_versions \(workspace_id, date_created\)",
            joined,
        )

    async def test_apply_does_not_qualify_with_schema_name(self) -> None:
        """
        table statements are unqualified; search_path at the L3 layer scopes them.
        this guards against accidental reintroduction of hard-coded
        agent_{hex}.table references that would break the broker contract.
        regex matches a 32-hex-char schema prefix exactly as the L3 broker
        would emit (``agent_<32hex>.``); legitimate column names like
        ``agent_id`` and constraint names like ``uq_workspaces_agent_name``
        must not trigger.
        """
        runner = MigrationRunner()
        register(runner)
        store = _FakeDataStore()
        await runner.apply_for_agent_schema(store)
        joined = _joined_executed_sql(store)
        assert not re.search(r"agent_[0-9a-f]{32}\.", joined)


class TestDirectMigrationFunction:
    """tests exercising create_workspace_tables directly without the runner."""

    async def test_direct_call_issues_expected_statement_count(self) -> None:
        """calling create_workspace_tables issues exactly five execute calls."""
        store = _FakeDataStore()
        await create_workspace_tables(store)  # type: ignore[arg-type]
        # three CREATE TABLE plus two CREATE INDEX
        assert len(store.executed) == 5

    async def test_direct_call_leaves_migrations_table_untouched(self) -> None:
        """direct invocation does not touch _schema_migrations bookkeeping."""
        store = _FakeDataStore()
        await create_workspace_tables(store)  # type: ignore[arg-type]
        assert store.migrations_table_created is False
        assert store.migrations_rows == []


class TestDatetimeToDatetimetzMigration:
    """
    tests for v004: TIMESTAMP -> TIMESTAMPTZ promotion of every datetime
    column in the agent-workspace partition.

    collections-task-05 requires every per-column ALTER to appear as a
    literal SQL string (not a templated DO block iterating a list) so
    the column-type-alignment AST walker in
    ``packages/core/tests/enforcement/test_column_type_alignment.py``
    can match each ``(table, column) -> TIMESTAMPTZ`` pair against its
    ``Column(..., DATETIMETZ_TYPE, ...)`` declaration in
    ``collections.py``. these tests pin that pattern so a future
    refactor cannot regress it.
    """

    async def test_direct_call_issues_five_per_column_alters(self) -> None:
        """one DO block per (table, column) pair: 5 statements."""
        store = _FakeDataStore()
        await datetime_to_datetimetz(store)  # type: ignore[arg-type]
        assert len(store.executed) == 5

    async def test_direct_call_targets_every_datetime_column(self) -> None:
        """every datetime column has its own ALTER literal."""
        store = _FakeDataStore()
        await datetime_to_datetimetz(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert "ALTER TABLE workspaces ALTER COLUMN date_created TYPE TIMESTAMPTZ" in joined
        assert "ALTER TABLE workspaces ALTER COLUMN date_updated TYPE TIMESTAMPTZ" in joined
        assert "ALTER TABLE workspaces ALTER COLUMN date_deleted TYPE TIMESTAMPTZ" in joined
        assert "ALTER TABLE workspace_files ALTER COLUMN date_updated TYPE TIMESTAMPTZ" in joined
        assert "ALTER TABLE workspace_file_versions ALTER COLUMN date_created TYPE TIMESTAMPTZ" in joined

    async def test_direct_call_uses_at_time_zone_utc(self) -> None:
        """every ALTER asserts UTC semantics on the bare TIMESTAMP cell."""
        store = _FakeDataStore()
        await datetime_to_datetimetz(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        # five (table, column) pairs each emit a USING <col> AT TIME ZONE 'UTC'
        # clause; the column names overlap (date_created, date_updated)
        # across tables so this assertion confirms presence rather than count.
        assert joined.count("AT TIME ZONE 'UTC'") == 5
        assert "USING date_created AT TIME ZONE 'UTC'" in joined
        assert "USING date_updated AT TIME ZONE 'UTC'" in joined
        assert "USING date_deleted AT TIME ZONE 'UTC'" in joined

    async def test_direct_call_is_guarded_by_information_schema(self) -> None:
        """each ALTER lives inside a DO block that probes data_type."""
        store = _FakeDataStore()
        await datetime_to_datetimetz(store)  # type: ignore[arg-type]
        joined = _joined_executed_sql(store)
        assert joined.count("information_schema.columns") == 5
        assert joined.count("'timestamp without time zone'") == 5

    async def test_direct_call_leaves_migrations_table_untouched(self) -> None:
        """direct invocation does not touch ``_schema_migrations``."""
        store = _FakeDataStore()
        await datetime_to_datetimetz(store)  # type: ignore[arg-type]
        assert store.migrations_table_created is False
        assert store.migrations_rows == []


class TestDuplicateVersionGuard:
    """tests confirming the runner rejects duplicate version registration."""

    async def test_duplicate_version_registration_raises(self) -> None:
        """registering a second migration at version 1 raises DuplicateVersionError."""
        runner = MigrationRunner()
        pkg = register(runner)
        with pytest.raises(DuplicateVersionError):
            pkg.version(1)(create_workspace_tables)


class TestAddDateDeletedColumnDirect:
    """tests exercising add_date_deleted_column directly without the runner."""

    async def test_direct_call_issues_alter_table(self) -> None:
        """calling add_date_deleted_column emits exactly one ALTER TABLE statement."""
        store = _FakeDataStore()
        await add_date_deleted_column(store)  # type: ignore[arg-type]
        assert len(store.executed) == 1
        sql_text = " ".join(store.executed[0][0].split())
        assert "ALTER TABLE workspaces" in sql_text
        assert "ADD COLUMN IF NOT EXISTS date_deleted" in sql_text
        assert "TIMESTAMP NULL" in sql_text
