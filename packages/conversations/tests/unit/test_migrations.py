"""
unit tests for 3tears-conversations migration registration and DDL shape.

these tests mirror the pattern used by agent-workspace and agent-memory:
a :class:`_FakeDataStore` captures executed SQL so the tests can assert
statement shape + idempotent re-apply without touching a real database.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from threetears.conversations.migrations import (
    PACKAGE_NAME,
    add_name_column,
    create_conversations_table,
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

    :ivar executed: list of (sql, params) tuples in execution order
    :ptype executed: list[tuple[str, tuple[Any, ...]]]
    :ivar migrations_rows: emulated ``_schema_migrations`` rows
    :ptype migrations_rows: list[dict[str, Any]]
    :ivar migrations_table_created: whether the bookkeeping table has
        been materialized
    :ptype migrations_table_created: bool
    """

    def __init__(self) -> None:
        """
        initialize an empty in-memory store.
        """
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
        :return: synthetic status string (``EXECUTE`` or an insert tag)
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


class TestRegisterConversationsMigrations:
    """tests for the register factory and apply flow."""

    async def test_register_returns_agent_scoped_package(self) -> None:
        """register produces a PackageMigrations scoped to AGENT."""
        runner = MigrationRunner()
        pkg = register(runner)
        assert pkg.name == PACKAGE_NAME
        assert pkg.scope == MigrationScope.AGENT

    async def test_register_has_no_depends_on_edges(self) -> None:
        """conversations is the root of the agent dependency graph."""
        runner = MigrationRunner()
        pkg = register(runner)
        assert pkg.depends_on == ()

    async def test_register_populates_versions_one_two_and_three(self) -> None:
        """register wires v001 (create), v002 (message_count), v003 (name)."""
        runner = MigrationRunner()
        pkg = register(runner)
        assert set(pkg.versions.keys()) == {1, 2, 3}

    async def test_apply_runs_three_versions_then_idempotent(self) -> None:
        """apply records v1+v2+v3 and re-running is a no-op."""
        runner = MigrationRunner()
        register(runner)
        store = _FakeDataStore()
        first_count = await runner.apply_for_agent_schema(store)
        assert first_count == 3
        assert store.migrations_table_created is True
        assert [row["version"] for row in store.migrations_rows] == [1, 2, 3]
        second_count = await runner.apply_for_agent_schema(store)
        assert second_count == 0

    async def test_apply_emits_name_column_add(self) -> None:
        """v003 emits an ADD COLUMN IF NOT EXISTS for ``name``."""
        runner = MigrationRunner()
        register(runner)
        store = _FakeDataStore()
        await runner.apply_for_agent_schema(store)
        joined = _joined_executed_sql(store)
        assert re.search(
            r"ALTER TABLE conversations\s+ADD COLUMN IF NOT EXISTS name TEXT",
            joined,
            re.IGNORECASE,
        )

    async def test_apply_emits_conversations_create_statement(self) -> None:
        """the CREATE TABLE statement carries every column and type."""
        runner = MigrationRunner()
        register(runner)
        store = _FakeDataStore()
        await runner.apply_for_agent_schema(store)
        joined = _joined_executed_sql(store)
        assert re.search(r"CREATE TABLE IF NOT EXISTS conversations", joined)
        assert re.search(r"agent_id UUID NOT NULL", joined)
        assert re.search(r"id UUID NOT NULL", joined)
        assert "PRIMARY KEY (agent_id, id)" in joined
        assert re.search(r"customer_id UUID NOT NULL", joined)
        assert re.search(r"user_id UUID NOT NULL", joined)
        assert "channel_type VARCHAR(50) NOT NULL" in joined
        assert "conversation_ref VARCHAR(500)" in joined
        assert "status VARCHAR(20) NOT NULL" in joined
        assert "summary TEXT" in joined
        assert re.search(r"date_created TIMESTAMP NOT NULL", joined)
        assert re.search(r"date_updated TIMESTAMP NOT NULL", joined)
        assert "date_last_message TIMESTAMP" in joined
        assert "metadata JSONB" in joined

    async def test_apply_emits_three_indexes(self) -> None:
        """lookup indexes on user, customer, and status are emitted."""
        runner = MigrationRunner()
        register(runner)
        store = _FakeDataStore()
        await runner.apply_for_agent_schema(store)
        joined = _joined_executed_sql(store)
        assert re.search(
            r"CREATE INDEX IF NOT EXISTS idx_conv_user "
            r"ON conversations \(user_id, date_created\)",
            joined,
        )
        assert re.search(
            r"CREATE INDEX IF NOT EXISTS idx_conv_customer "
            r"ON conversations \(customer_id, date_created\)",
            joined,
        )
        assert re.search(
            r"CREATE INDEX IF NOT EXISTS idx_conv_status "
            r"ON conversations \(status\)",
            joined,
        )

    async def test_apply_does_not_qualify_with_schema_name(self) -> None:
        """
        DDL statements stay unqualified so ``search_path`` governs.

        protects against accidental hard-coded ``agent_<32hex>.`` prefixes
        that would break the L3 broker contract.
        """
        runner = MigrationRunner()
        register(runner)
        store = _FakeDataStore()
        await runner.apply_for_agent_schema(store)
        joined = _joined_executed_sql(store)
        assert not re.search(r"agent_[0-9a-f]{32}\.", joined)


class TestDirectMigrationFunction:
    """tests exercising create_conversations_table directly."""

    async def test_direct_call_issues_four_statements(self) -> None:
        """one CREATE TABLE plus three CREATE INDEX statements."""
        store = _FakeDataStore()
        await create_conversations_table(store)  # type: ignore[arg-type]
        assert len(store.executed) == 4

    async def test_direct_call_leaves_migrations_table_untouched(self) -> None:
        """direct invocation does not touch ``_schema_migrations``."""
        store = _FakeDataStore()
        await create_conversations_table(store)  # type: ignore[arg-type]
        assert store.migrations_table_created is False
        assert store.migrations_rows == []


class TestDuplicateVersionGuard:
    """tests confirming the runner rejects duplicate version registration."""

    async def test_duplicate_version_registration_raises(self) -> None:
        """registering a second callable at v1 raises DuplicateVersionError."""
        runner = MigrationRunner()
        pkg = register(runner)
        with pytest.raises(DuplicateVersionError):
            pkg.version(1)(create_conversations_table)
