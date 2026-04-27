"""
unit tests for agent-workspace v003 (workspace_namespace_backfill).

v003 is a one-statement data-translation migration that heals pre-task-19
history: every live row in ``<agent_schema>.workspaces`` gets a matching
``platform.namespaces`` row stamped with the same id and
``namespace_type='workspace'``. these tests verify the emitted SQL shape
against a ``_CaptureStore`` stub; end-to-end behavior is covered by the
integration test in
``14-eng-ai-bot/tests/integration/test_workspaces_as_namespaces_migration.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.agent.workspace.migrations import (
    PACKAGE_NAME,
    register,
    workspace_namespace_backfill,
)
from threetears.core.data.migrations import (
    MigrationRunner,
    MigrationScope,
)


class _CaptureStore:
    """DataStore-shaped stub capturing executed SQL for assertions."""

    def __init__(self) -> None:
        """initialize empty execution log."""
        self.executed: list[str] = []

    async def execute(self, sql: str, *params: Any) -> str:
        """
        record SQL execution and return synthetic status.

        :param sql: SQL statement text
        :ptype sql: str
        :param params: positional parameters (ignored)
        :ptype params: Any
        :return: synthetic status string
        :rtype: str
        """
        self.executed.append(sql)
        return "EXECUTE"


def _joined_sql(store: _CaptureStore) -> str:
    """
    join every captured SQL statement into a single whitespace-collapsed
    string so asserts can pattern-match against a stable surface.

    :param store: capture store
    :ptype store: _CaptureStore
    :return: normalized SQL text
    :rtype: str
    """
    return "\n".join(" ".join(sql.split()) for sql in store.executed)


class TestWorkspaceNamespaceBackfillShape:
    """tests verifying the v003 migration emits the expected backfill SQL."""

    @pytest.mark.asyncio
    async def test_inserts_into_platform_namespaces(self) -> None:
        """v003 targets platform.namespaces (cross-schema write)."""
        store = _CaptureStore()
        await workspace_namespace_backfill(store)  # type: ignore[arg-type]
        joined = _joined_sql(store)
        assert "INSERT INTO platform.namespaces" in joined

    @pytest.mark.asyncio
    async def test_sets_namespace_type_to_workspace(self) -> None:
        """every inserted namespace row carries namespace_type='workspace'."""
        store = _CaptureStore()
        await workspace_namespace_backfill(store)  # type: ignore[arg-type]
        joined = _joined_sql(store)
        assert "'workspace' AS namespace_type" in joined

    @pytest.mark.asyncio
    async def test_uses_current_schema_for_schema_name(self) -> None:
        """schema_name points at the current agent schema via search_path."""
        store = _CaptureStore()
        await workspace_namespace_backfill(store)  # type: ignore[arg-type]
        joined = _joined_sql(store)
        assert "current_schema() AS schema_name" in joined

    @pytest.mark.asyncio
    async def test_joins_platform_agents_for_customer_id(self) -> None:
        """customer_id resolves via join against platform.agents."""
        store = _CaptureStore()
        await workspace_namespace_backfill(store)  # type: ignore[arg-type]
        joined = _joined_sql(store)
        assert "JOIN platform.agents a ON a.id = w.agent_id" in joined
        assert "a.customer_id" in joined

    @pytest.mark.asyncio
    async def test_shares_primary_key_with_workspaces(self) -> None:
        """the inserted id column is pulled straight from workspaces.id."""
        store = _CaptureStore()
        await workspace_namespace_backfill(store)  # type: ignore[arg-type]
        joined = _joined_sql(store)
        assert "SELECT w.id," in joined

    @pytest.mark.asyncio
    async def test_skips_soft_deleted_workspaces(self) -> None:
        """only live (date_deleted IS NULL) workspaces are backfilled."""
        store = _CaptureStore()
        await workspace_namespace_backfill(store)  # type: ignore[arg-type]
        joined = _joined_sql(store)
        assert "WHERE w.date_deleted IS NULL" in joined

    @pytest.mark.asyncio
    async def test_is_idempotent_via_on_conflict(self) -> None:
        """replay is safe: ON CONFLICT (id) DO NOTHING."""
        store = _CaptureStore()
        await workspace_namespace_backfill(store)  # type: ignore[arg-type]
        joined = _joined_sql(store)
        assert "ON CONFLICT (id) DO NOTHING" in joined

    @pytest.mark.asyncio
    async def test_namespace_name_prefix_avoids_agent_collision(self) -> None:
        """
        namespaces.name is UNIQUE. the agent-scope namespace uses
        ``agent.<uuid>``; workspaces prefix with ``workspace.`` so the
        two ranges cannot collide.
        """
        store = _CaptureStore()
        await workspace_namespace_backfill(store)  # type: ignore[arg-type]
        joined = _joined_sql(store)
        assert "'workspace.' || w.id::text AS name" in joined


class TestRegisterIncludesV003:
    """tests verifying the new version is wired into the package registration."""

    async def test_register_returns_package_with_versions_one_two_three(self) -> None:
        """register populates the PackageMigrations with versions 1, 2, and 3."""
        runner = MigrationRunner()
        pkg = register(runner)
        assert pkg.name == PACKAGE_NAME
        assert pkg.scope == MigrationScope.AGENT
        assert set(pkg.versions.keys()) == {1, 2, 3}
