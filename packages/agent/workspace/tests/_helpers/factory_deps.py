"""the minimum dependency bundle :func:`build_workspace_tools` needs, for factory tests.

every registered workspace tool is constructed from this bundle; no builder dereferences the
collections, sandbox or pool at construction, so stand-ins that satisfy the constructors are
enough. shared by the factory tests and the subprocess they run a registration probe in.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from threetears.agent.acl import (
    AclCache,
    GroupMembership,
    Namespace,
    Role,
    RoleAssignment,
)

from packages.agent.workspace.tests._helpers.asyncpg_shims import FakeAsyncpgPool
from packages.agent.workspace.tests._helpers.workspace_shims import (
    FakeWorkspaceContext,
    FakeWorkspaceSandbox,
)

__all__ = ["minimal_tool_deps"]


# parity-exempt: workspace-collection subset for the workspace tools factory test; the test exercises factory wiring only and does not call collection methods directly
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

    async def load_for_group(self, group_id: UUID) -> tuple[GroupMembership, ...]:
        """return parent-group memberships -- none; these fixtures use flat groups.

        :param group_id: child group UUID
        :ptype group_id: UUID
        :return: empty tuple
        :rtype: tuple[GroupMembership, ...]
        """
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


# parity-exempt: NamespaceCollection subset for the workspace tools factory test exposing only the get_by_name lookup the namespace-emit surface uses
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


def minimal_tool_deps() -> dict[str, Any]:
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
