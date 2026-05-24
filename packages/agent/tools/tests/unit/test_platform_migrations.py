"""unit tests for the agent-tools platform-scope migrations.

agent-tools-eligibility shard 01 (TE-04 / TE-11): verify the
platform-scope migration package registers under the canonical
runner, declares ``PLATFORM`` scope, and emits the two
``ADD COLUMN IF NOT EXISTS`` statements that bring the
``tool_eligible`` + ``skill_eligible`` columns to the platform
``namespaces`` table.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from threetears.agent.tools.platform_migrations import (
    PACKAGE_NAME,
    add_tool_eligibility_columns,
    register,
)
from threetears.core.data.migrations import (
    DuplicateVersionError,
    MigrationRunner,
    MigrationScope,
)


class _FakeDataStore:
    """recording stand-in for :class:`DataStore`.

    same shape as the agent-tools agent-scope migration test fake
    (kept private rather than shared because the canonical
    ``_FakeDataStore`` lives next to its own tests; sharing across
    test modules adds an import-order coupling we don't need here).

    :ivar executed: list of (sql, params) tuples
    :ptype executed: list[tuple[str, tuple[Any, ...]]]
    :ivar migrations_rows: emulated ``_schema_migrations`` rows
    :ptype migrations_rows: list[dict[str, Any]]
    """

    def __init__(self) -> None:
        """initialize an empty store."""
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.migrations_rows: list[dict[str, Any]] = []
        self.migrations_table_created = False

    async def execute(self, sql: str, *params: Any) -> str:
        """record one execute call and emulate the bookkeeping write.

        :param sql: SQL statement text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: synthetic status string
        :rtype: str
        """
        self.executed.append((sql, params))
        normalized = " ".join(sql.split()).upper()
        if "CREATE TABLE IF NOT EXISTS _SCHEMA_MIGRATIONS" in normalized:
            self.migrations_table_created = True
            return "CREATE TABLE"
        if normalized.startswith("INSERT INTO _SCHEMA_MIGRATIONS"):
            self.migrations_rows.append(
                {
                    "version": params[0],
                    "package": params[1],
                    "description": params[2],
                }
            )
            return "INSERT 0 1"
        return "EXECUTE"

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """emulate the runner's bookkeeping selects.

        :param sql: SQL query text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        normalized = " ".join(sql.split()).upper()
        if "SELECT VERSION, PACKAGE FROM _SCHEMA_MIGRATIONS" in normalized:
            return [{"version": row["version"], "package": row["package"]} for row in self.migrations_rows]
        if "COALESCE(MAX(VERSION)" in normalized:
            max_version = max(
                (row["version"] for row in self.migrations_rows),
                default=0,
            )
            return [{"max_version": max_version}]
        return []


def _joined_sql(store: _FakeDataStore) -> str:
    """join every executed statement into one whitespace-normalized string.

    :param store: fake store
    :ptype store: _FakeDataStore
    :return: joined statements
    :rtype: str
    """
    return "\n".join(" ".join(sql.split()) for sql, _params in store.executed)


class TestRegisterPlatformMigrations:
    """package registration + scope + version contract."""

    def test_register_returns_platform_scoped_package(self) -> None:
        """register produces a PackageMigrations scoped to PLATFORM."""
        runner = MigrationRunner()
        pkg = register(runner)
        assert pkg.name == PACKAGE_NAME
        assert pkg.scope == MigrationScope.PLATFORM

    def test_register_declares_no_dependencies(self) -> None:
        """the ``namespaces`` table is platform-managed; no 3tears dep."""
        runner = MigrationRunner()
        pkg = register(runner)
        assert pkg.depends_on == ()

    def test_register_attaches_v001(self) -> None:
        """only v001 is registered today."""
        runner = MigrationRunner()
        pkg = register(runner)
        assert set(pkg.versions.keys()) == {1}

    async def test_apply_in_isolation_runs_v001(self) -> None:
        """apply_package walks v001 against the fake store."""
        runner = MigrationRunner()
        register(runner)
        store = _FakeDataStore()
        first_count = await runner.apply_package(store, PACKAGE_NAME)
        assert first_count == 1
        assert [row["version"] for row in store.migrations_rows] == [1]


class TestAddToolEligibilityColumnsMigration:
    """direct invocation of the v001 callable."""

    async def test_issues_two_alter_statements(self) -> None:
        """one ``ADD COLUMN IF NOT EXISTS`` per flag column."""
        store = _FakeDataStore()
        await add_tool_eligibility_columns(store)  # type: ignore[arg-type]
        assert len(store.executed) == 2

    async def test_alter_targets_tool_eligible_with_default_true(self) -> None:
        """``tool_eligible BOOLEAN NOT NULL DEFAULT TRUE`` matches the
        shard contract."""
        store = _FakeDataStore()
        await add_tool_eligibility_columns(store)  # type: ignore[arg-type]
        joined = _joined_sql(store)
        assert re.search(
            r"ALTER TABLE namespaces "
            r"ADD COLUMN IF NOT EXISTS tool_eligible "
            r"BOOLEAN NOT NULL DEFAULT TRUE",
            joined,
        )

    async def test_alter_targets_skill_eligible_with_default_false(self) -> None:
        """``skill_eligible BOOLEAN NOT NULL DEFAULT FALSE`` matches."""
        store = _FakeDataStore()
        await add_tool_eligibility_columns(store)  # type: ignore[arg-type]
        joined = _joined_sql(store)
        assert re.search(
            r"ALTER TABLE namespaces "
            r"ADD COLUMN IF NOT EXISTS skill_eligible "
            r"BOOLEAN NOT NULL DEFAULT FALSE",
            joined,
        )

    async def test_statements_unqualified_so_search_path_governs(self) -> None:
        """no statement carries an explicit ``platform.`` schema prefix."""
        store = _FakeDataStore()
        await add_tool_eligibility_columns(store)  # type: ignore[arg-type]
        joined = _joined_sql(store)
        assert "platform.namespaces" not in joined.lower()

    async def test_idempotent_via_if_not_exists(self) -> None:
        """both ALTERs use IF NOT EXISTS so replays are no-ops."""
        store = _FakeDataStore()
        await add_tool_eligibility_columns(store)  # type: ignore[arg-type]
        joined = _joined_sql(store)
        assert joined.count("ADD COLUMN IF NOT EXISTS") == 2

    async def test_direct_call_does_not_touch_migrations_table(self) -> None:
        """direct invocation must NOT write to ``_schema_migrations``."""
        store = _FakeDataStore()
        await add_tool_eligibility_columns(store)  # type: ignore[arg-type]
        assert store.migrations_table_created is False
        assert store.migrations_rows == []


class TestDuplicateRegistrationGuard:
    """re-registering v1 raises the canonical duplicate-version error."""

    def test_duplicate_version_registration_raises(self) -> None:
        """``pkg.version(1)`` twice raises :class:`DuplicateVersionError`."""
        runner = MigrationRunner()
        pkg = register(runner)
        with pytest.raises(DuplicateVersionError):
            pkg.version(1)(add_tool_eligibility_columns)
