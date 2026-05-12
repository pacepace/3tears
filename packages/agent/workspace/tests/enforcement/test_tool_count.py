"""enforcement: workspace tool registry matches frozen 19-name set.

importing :mod:`threetears.agent.workspace.tools` fires every tool
module's ``register_tool_builder`` side-effect, populating
:data:`threetears.agent.workspace.factory._TOOL_BUILDERS`. this test
constructs every registered builder with stub dependencies, collects
``mcp_name`` from each, and asserts the set exactly equals the frozen
expected set. any drift -- a rename, an addition, a removal, or a missed
registration -- fails the test with the specific set difference.

the stub dependencies pass :class:`unittest.mock.MagicMock` for every
collection / sandbox / lease / pool argument because the tool
constructors only store their dependencies -- no method is invoked at
construction time. ``context_provider`` is a zero-arg callable returning
a mock; :func:`build_workspace_tools` does not call it.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

from threetears.agent.acl import (
    AclCache,
    GroupMembership,
    Namespace,
    Role,
    RoleAssignment,
)


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


_EXPECTED_WORKSPACE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "threetears.workspace.list",
        "threetears.workspace.use",
        "threetears.workspace.current",
        "threetears.workspace.create",
        "threetears.workspace.reset",
        "threetears.workspace.delete",
        "threetears.workspace.history",
        "threetears.workspace.diff",
        "threetears.workspace.checkpoint",
        "threetears.workspace.rollback_to",
        "threetears.workspace.fs_read",
        "threetears.workspace.fs_write",
        "threetears.workspace.fs_list",
        "threetears.workspace.fs_edit",
        "threetears.workspace.doc_get",
        "threetears.workspace.doc_set",
        "threetears.workspace.doc_merge",
        "threetears.workspace.refresh_from_disk",
        "threetears.workspace.flush_to_disk",
    }
)
_EXPECTED_WORKSPACE_TOOL_COUNT = 19
_NAMESPACE_PREFIX = "threetears.workspace."


def _stub_dependencies() -> dict[str, Any]:
    """
    build the dependency bundle passed to every registered tool builder.

    every collection / sandbox / lease / pool arg is a
    :class:`MagicMock`. ``context_provider`` is a zero-arg callable
    returning a mock; tool constructors only store these references
    (none are invoked at construction time). ``agent_id`` / ``pod_id``
    are fresh UUIDs so any tool that records them as attributes sees a
    well-formed value.

    :return: mapping of factory kwargs to stub values
    :rtype: dict[str, Any]
    """
    return {
        "acl_cache": AclCache(
            membership_loader=_NoopMembershipLoader(),
            grant_loader=_NoopGrantLoader(),
            ttl_seconds=60,
        ),
        "namespace_collection": MagicMock(),
        "workspace_collection": MagicMock(),
        "workspace_file_collection": MagicMock(),
        "workspace_file_version_collection": MagicMock(),
        "sandbox": MagicMock(),
        "lease": MagicMock(),
        "context_provider": lambda: MagicMock(),
        "nats_client": MagicMock(),
        "namespace": "enforcement-test",
        "agent_id": uuid4(),
        "pod_id": uuid4(),
        "config": None,
        "db_pool": MagicMock(),
        "validators": None,
    }


class TestWorkspaceToolCount:
    """frozen tool-count + name-set enforcement."""

    def test_factory_emits_exactly_nineteen_tools(self) -> None:
        """
        import the tools package, build via factory, assert exact count.

        importing :mod:`threetears.agent.workspace.tools` fires each
        module's ``register_tool_builder`` side-effect. we then call
        :func:`build_workspace_tools` with stub dependencies and count
        the returned list.

        :return: None
        :rtype: None
        """
        # side-effect import: registers all 19 builders.
        from threetears.agent.workspace import tools as _tools  # noqa: F401
        from threetears.agent.workspace.factory import (
            _TOOL_BUILDERS,
            build_workspace_tools,
        )

        assert len(_TOOL_BUILDERS) == _EXPECTED_WORKSPACE_TOOL_COUNT, (
            f"expected {_EXPECTED_WORKSPACE_TOOL_COUNT} registered builders; found {len(_TOOL_BUILDERS)}"
        )
        tools = build_workspace_tools(**_stub_dependencies())
        assert len(tools) == _EXPECTED_WORKSPACE_TOOL_COUNT

    def test_tool_names_match_frozen_set(self) -> None:
        """
        every ``mcp_name`` matches the canonical frozen set, no drift.

        :return: None
        :rtype: None
        """
        from threetears.agent.workspace import tools as _tools  # noqa: F401
        from threetears.agent.workspace.factory import build_workspace_tools

        tools = build_workspace_tools(**_stub_dependencies())
        actual_names = {t.mcp_name() for t in tools}
        missing = _EXPECTED_WORKSPACE_TOOL_NAMES - actual_names
        unexpected = actual_names - _EXPECTED_WORKSPACE_TOOL_NAMES
        assert not missing and not unexpected, (
            f"tool name set drift:\n"
            f"  missing (in expected, not in actual): {sorted(missing)}\n"
            f"  unexpected (in actual, not in expected): {sorted(unexpected)}"
        )

    def test_every_tool_name_uses_namespace_prefix(self) -> None:
        """
        every ``mcp_name`` starts with ``threetears.workspace.``.

        :return: None
        :rtype: None
        """
        from threetears.agent.workspace import tools as _tools  # noqa: F401
        from threetears.agent.workspace.factory import build_workspace_tools

        tools = build_workspace_tools(**_stub_dependencies())
        violations = [t.mcp_name() for t in tools if not t.mcp_name().startswith(_NAMESPACE_PREFIX)]
        assert not violations, f"{len(violations)} tool(s) lack the {_NAMESPACE_PREFIX!r} prefix: {violations}"
