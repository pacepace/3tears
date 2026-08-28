"""
unit tests for agent-workspace v003, which is retired and applies nothing.

v003 used to issue one cross-schema INSERT that healed pre-task-19 history:
every live row in ``<agent_schema>.workspaces`` got a matching row in the
hub's ``namespaces`` table. Reaching a second schema from a connection whose
``search_path`` is the agent's requires naming that schema in the SQL, and the
statement named the hub's REMOVED default. That name is correct on one
deployment and wrong on every other, the correct one cannot be threaded into a
migration body, and the row is written by the hub off ``workspace_create``'s
emitted event now. So the body is gone and the version number stays claimed.

These tests hold the retirement: v003 executes NOTHING, and in particular
issues no statement naming another schema. Its admitted twin is
``test_migrations.py``'s registration assertions, which still expect version 3
to exist -- retired is not the same as renumbered.
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


class TestWorkspaceNamespaceBackfillIsRetired:
    """tests pinning that v003 issues no statement at all."""

    @pytest.mark.asyncio
    async def test_executes_no_statement(self) -> None:
        """the retired body runs nothing against the store."""
        store = _CaptureStore()
        await workspace_namespace_backfill(store)  # type: ignore[arg-type]
        assert store.executed == []

    @pytest.mark.asyncio
    async def test_names_no_schema_in_any_statement(self) -> None:
        """no statement qualifies a table with a schema name.

        stated separately from "executes nothing" on purpose: if a future
        change gives v003 a body again, this is the assertion that has to be
        confronted, and the answer cannot be a hardcoded schema.
        """
        store = _CaptureStore()
        await workspace_namespace_backfill(store)  # type: ignore[arg-type]
        qualified = [sql for sql in store.executed if "platform." in sql or "aibots." in sql]
        assert qualified == []

    @pytest.mark.asyncio
    async def test_is_replay_safe(self) -> None:
        """applying it twice is indistinguishable from applying it once."""
        store = _CaptureStore()
        await workspace_namespace_backfill(store)  # type: ignore[arg-type]
        await workspace_namespace_backfill(store)  # type: ignore[arg-type]
        assert store.executed == []


class TestRegisterStillClaimsVersionThree:
    """the version number survives the retirement; renumbering does not happen."""

    async def test_register_includes_v003(self) -> None:
        """register still wires version 3, now to the retired callable.

        the number must stay claimed. ``_verify_ledger_identity`` compares
        the recorded ``description`` (the callable's ``__name__``) against
        what the build registers at that version, so shifting v004 down into
        3 would make every database carrying the old ledger read the shifted
        version as already applied and never run its body.
        """
        runner = MigrationRunner()
        pkg = register(runner)
        assert pkg.name == PACKAGE_NAME
        assert pkg.scope == MigrationScope.AGENT
        assert {1, 2, 3}.issubset(set(pkg.versions.keys()))
        assert pkg.versions[3] is workspace_namespace_backfill
