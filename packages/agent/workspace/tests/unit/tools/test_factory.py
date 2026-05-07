"""tests for :mod:`threetears.agent.workspace.factory` registry and builder."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

import threetears.agent.workspace.tools  # noqa: F401  -- registers builders
from threetears.agent.acl import (
    AclCache,
    GroupMembership,
    Namespace,
    Role,
    RoleAssignment,
)
from threetears.agent.tools.base_tool import TearsTool
from threetears.agent.workspace.factory import _TOOL_BUILDERS, build_workspace_tools
from _helpers.asyncpg_shims import FakeAsyncpgAcquireCM, FakeAsyncpgConnection, FakeAsyncpgPool, FakeAsyncpgTransaction
from _helpers.workspace_shims import (
    FakeWorkspaceCollection,
    FakeWorkspaceContext,
    FakeWorkspaceEntity,
    FakeWorkspaceFile,
    FakeWorkspaceFileCollection,
    FakeWorkspaceFileVersionCollection,
    FakeWorkspaceSandbox,
)


class _FakeCollection:
    """minimal collection stub satisfying the WorkspaceListTool/UseTool deps."""

    async def find_by_agent(self, agent_id: Any) -> list[Any]:
        return []

    async def find_by_agent_and_name(self, agent_id: Any, name: str) -> Any:
        return None

    async def find_by_workspace(self, workspace_id: Any) -> list[Any]:
        return []


class _FakeContext(FakeWorkspaceContext):
    """sentinel context object returned by the provider closure."""


class _FakeSandbox(FakeWorkspaceSandbox):
    """sandbox stub for tools that accept it but never invoke it in build."""


class _FakePool(FakeAsyncpgPool):
    """asyncpg pool stub for tools that accept it but never invoke it in build."""


class _NoopMembershipLoader:
    """membership loader stub yielding empty memberships."""

    async def load_for_user(
        self,
        user_id: UUID,
    ) -> tuple[GroupMembership, ...]:
        del user_id
        return ()

    async def load_for_agent(
        self,
        agent_id: UUID,
    ) -> tuple[GroupMembership, ...]:
        del agent_id
        return ()


class _NoopGrantLoader:
    """grant loader stub yielding empty grants."""

    async def load_assignments_for_groups(
        self,
        group_ids: tuple[UUID, ...],
        namespace: Namespace,
    ) -> tuple[RoleAssignment, ...]:
        del group_ids, namespace
        return ()

    async def load_roles(
        self,
        role_ids: tuple[UUID, ...],
    ) -> dict[UUID, Role]:
        del role_ids
        return {}

    async def load_groups(
        self,
        group_ids: tuple[UUID, ...],
    ) -> dict[UUID, object]:
        del group_ids
        return {}


def _make_acl_cache() -> AclCache:
    """build a real :class:`AclCache` with noop loaders for factory tests."""
    return AclCache(
        membership_loader=_NoopMembershipLoader(),
        grant_loader=_NoopGrantLoader(),
        ttl_seconds=60,
    )


class _FakeNamespaceCollection:
    """stub that satisfies the ``namespace_collection`` shape at build time.

    :class:`WorkspaceCreateTool` captures the reference at construction
    and only dereferences ``entity_class`` / ``save_entity`` inside
    :meth:`execute`. factory tests never drive a create, so the
    collection attribute exists purely to keep the constructor happy.
    """

    async def save_entity(self, entity: Any) -> None:
        """no-op save placeholder for the factory builder path."""
        del entity

    class entity_class:  # noqa: N801 -- matches BaseCollection attribute
        """dummy entity class placeholder for construction tests."""

        def __init__(
            self,
            data: Any,
            *,
            is_new: bool,
            collection: Any,
        ) -> None:
            """capture kwargs for parity with the real entity signature."""
            self.data = data
            self.is_new = is_new
            self.collection = collection


def _minimal_deps() -> dict[str, Any]:
    """build the minimum deps bundle every workspace tool requires."""
    return {
        "acl_cache": _make_acl_cache(),
        "namespace_collection": _FakeNamespaceCollection(),
        "workspace_collection": _FakeCollection(),
        "workspace_file_collection": _FakeCollection(),
        "workspace_file_version_collection": _FakeCollection(),
        "sandbox": _FakeSandbox(),
        "agent_id": uuid4(),
        "context_provider": lambda: _FakeContext(),
        "db_pool": _FakePool(),
    }


def test_tool_builders_registry_has_nineteen_after_history_tools() -> None:
    """importing the tools subpackage must register all nineteen tools.

    six meta + lifecycle (shards 09+10) plus four fs_* tools (shard 11)
    plus three doc_* tools (shard 12) plus four history tools (shard 13:
    history, diff, checkpoint, rollback_to) plus the refresh_from_disk
    live-sync tool that landed alongside bind's watcher, plus the
    flush_to_disk one-shot that projects L3 back onto disk.
    """
    assert len(_TOOL_BUILDERS) == 19


def test_build_workspace_tools_returns_nineteen_tools() -> None:
    """build_workspace_tools instantiates every registered builder."""
    tools = build_workspace_tools(**_minimal_deps())

    assert len(tools) == 19
    assert all(isinstance(t, TearsTool) for t in tools)


def test_build_workspace_tools_includes_each_expected_mcp_name() -> None:
    """built tools include exactly the nineteen expected mcp_name strings."""
    tools = build_workspace_tools(**_minimal_deps())

    names = {t.mcp_name() for t in tools}
    assert names == {
        "threetears.workspace.list",
        "threetears.workspace.use",
        "threetears.workspace.current",
        "threetears.workspace.create",
        "threetears.workspace.reset",
        "threetears.workspace.delete",
        "threetears.workspace.fs_read",
        "threetears.workspace.fs_write",
        "threetears.workspace.fs_list",
        "threetears.workspace.fs_edit",
        "threetears.workspace.doc_get",
        "threetears.workspace.doc_set",
        "threetears.workspace.doc_merge",
        "threetears.workspace.history",
        "threetears.workspace.diff",
        "threetears.workspace.checkpoint",
        "threetears.workspace.rollback_to",
        "threetears.workspace.refresh_from_disk",
        "threetears.workspace.flush_to_disk",
    }


def test_build_workspace_tools_returns_fresh_instances() -> None:
    """each call returns new instances; tools are not singletons."""
    first = build_workspace_tools(**_minimal_deps())
    second = build_workspace_tools(**_minimal_deps())

    first_ids = {id(t) for t in first}
    second_ids = {id(t) for t in second}
    assert first_ids.isdisjoint(second_ids)


def test_build_workspace_tools_tolerates_missing_optional_deps() -> None:
    """unused deps default to None so callers can pass only what tools need."""
    deps = _minimal_deps()
    tools = build_workspace_tools(**deps)

    assert len(tools) == 19


def test_register_tool_builder_appends_to_registry() -> None:
    """register_tool_builder appends the builder so it is emitted on next build."""
    from threetears.agent.workspace.factory import register_tool_builder

    sentinel_calls: list[dict[str, Any]] = []

    class _SentinelTool(TearsTool):
        async def execute(self, **kwargs: Any) -> Any:  # pragma: no cover - not exercised
            return None

        def mcp_schema(self) -> Any:  # pragma: no cover - not exercised
            return None

        def mcp_name(self) -> str:
            return "threetears.workspace._sentinel"

        def mcp_version(self) -> str:
            return "0.0"

    def _build(**kwargs: Any) -> _SentinelTool:
        sentinel_calls.append(kwargs)
        return _SentinelTool()

    initial = list(_TOOL_BUILDERS)
    register_tool_builder(_build)
    try:
        tools = build_workspace_tools(**_minimal_deps())
        assert any(t.mcp_name() == "threetears.workspace._sentinel" for t in tools)
        assert len(sentinel_calls) == 1
    finally:
        # restore the registry so test ordering does not affect counts
        _TOOL_BUILDERS.clear()
        _TOOL_BUILDERS.extend(initial)


@pytest.mark.parametrize(
    "expected",
    [
        "threetears.workspace.list",
        "threetears.workspace.use",
        "threetears.workspace.current",
        "threetears.workspace.create",
        "threetears.workspace.reset",
        "threetears.workspace.delete",
        "threetears.workspace.fs_read",
        "threetears.workspace.fs_write",
        "threetears.workspace.fs_list",
        "threetears.workspace.fs_edit",
        "threetears.workspace.doc_get",
        "threetears.workspace.doc_set",
        "threetears.workspace.doc_merge",
        "threetears.workspace.history",
        "threetears.workspace.diff",
        "threetears.workspace.checkpoint",
        "threetears.workspace.rollback_to",
        "threetears.workspace.refresh_from_disk",
        "threetears.workspace.flush_to_disk",
    ],
)
def test_each_expected_mcp_name_present(expected: str) -> None:
    """each of the nineteen required mcp_name strings is emitted."""
    tools = build_workspace_tools(**_minimal_deps())

    names = [t.mcp_name() for t in tools]
    assert expected in names
