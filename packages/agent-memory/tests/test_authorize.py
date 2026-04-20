"""tests for the memory authorize helper (namespace-task-01 phase 3)."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from threetears.agent.acl import (
    Group,
    GroupMembership,
    MemberType,
    Role,
    RoleAssignment,
    ScopeType,
)

from threetears.agent.memory.authorize import (
    ACTION_MEMORY_EXTRACT,
    ACTION_MEMORY_READ,
    ACTION_MEMORY_WRITE,
    MEMORY_OWNER_ROLE_NAME,
    MemoryAccessDenied,
    MemoryAuthorizerDependencies,
    MemoryNamespaceRow,
    authorize_memory_access,
    memory_namespace_name,
)


def _ns_row(
    *,
    agent_id: UUID,
    customer_id: UUID,
) -> MemoryNamespaceRow:
    return MemoryNamespaceRow(
        id=uuid4(),
        namespace_type="memory",
        owner_agent_id=agent_id,
        customer_id=customer_id,
    )


def _owner_role() -> Role:
    return Role(
        id=uuid4(),
        name=MEMORY_OWNER_ROLE_NAME,
        permissions={
            "memory": frozenset(
                {ACTION_MEMORY_READ, ACTION_MEMORY_WRITE, ACTION_MEMORY_EXTRACT},
            ),
        },
        is_built_in=True,
    )


def _reader_role() -> Role:
    return Role(
        id=uuid4(),
        name="MemoryReader",
        permissions={"memory": frozenset({ACTION_MEMORY_READ})},
        is_built_in=True,
    )


def _deps(
    *,
    resolver_returns: MemoryNamespaceRow | None,
    memberships_for_user: tuple[GroupMembership, ...] = (),
    memberships_for_agent: tuple[GroupMembership, ...] = (),
    assignments: tuple[RoleAssignment, ...] = (),
    roles: dict[UUID, Role] | None = None,
    groups: dict[UUID, Group] | None = None,
) -> tuple[MemoryAuthorizerDependencies, AsyncMock]:
    ensure_mock = AsyncMock()
    membership_loader = AsyncMock()
    membership_loader.load_for_user = AsyncMock(return_value=memberships_for_user)
    membership_loader.load_for_agent = AsyncMock(return_value=memberships_for_agent)
    grant_loader = AsyncMock()
    grant_loader.load_assignments_for_groups = AsyncMock(return_value=assignments)
    grant_loader.load_roles = AsyncMock(return_value=roles or {})
    grant_loader.load_groups = AsyncMock(return_value=groups or {})
    namespace_resolver = AsyncMock(return_value=resolver_returns)
    deps = MemoryAuthorizerDependencies(
        membership_loader=membership_loader,
        grant_loader=grant_loader,
        namespace_resolver=namespace_resolver,
        assignment_ensurer=ensure_mock,
    )
    return deps, ensure_mock


class TestMemoryNamespaceName:
    def test_shape(self) -> None:
        agent_id = UUID("019470a8-b5c3-7def-8123-456789abcdef")
        customer_id = UUID("11112222-3333-4444-5555-666677778888")
        name = memory_namespace_name(agent_id, customer_id)
        assert name == "memory:019470a8:11112222"


class TestAuthorizeMemoryAccess:
    async def test_namespace_unresolved_raises(self) -> None:
        deps, _ = _deps(resolver_returns=None)
        with pytest.raises(MemoryAccessDenied, match="could not be resolved"):
            await authorize_memory_access(
                action=ACTION_MEMORY_READ,
                agent_id=uuid4(),
                customer_id=uuid4(),
                caller_user_id=uuid4(),
                caller_agent_id=None,
                deps=deps,
            )

    async def test_owner_shortcut_allows_agent_without_grant(self) -> None:
        agent_id = uuid4()
        customer_id = uuid4()
        ns = _ns_row(agent_id=agent_id, customer_id=customer_id)
        deps, _ = _deps(resolver_returns=ns)

        result = await authorize_memory_access(
            action=ACTION_MEMORY_WRITE,
            agent_id=agent_id,
            customer_id=customer_id,
            caller_user_id=None,
            caller_agent_id=agent_id,
            deps=deps,
        )
        assert result is ns

    async def test_user_without_grant_denied(self) -> None:
        agent_id = uuid4()
        customer_id = uuid4()
        ns = _ns_row(agent_id=agent_id, customer_id=customer_id)
        deps, _ = _deps(resolver_returns=ns)
        with pytest.raises(MemoryAccessDenied, match="evaluator denied"):
            await authorize_memory_access(
                action=ACTION_MEMORY_READ,
                agent_id=agent_id,
                customer_id=customer_id,
                caller_user_id=uuid4(),
                caller_agent_id=None,
                deps=deps,
            )

    async def test_user_with_reader_grant_allowed(self) -> None:
        agent_id = uuid4()
        customer_id = uuid4()
        user_id = uuid4()
        group_id = uuid4()
        role = _reader_role()
        ns = _ns_row(agent_id=agent_id, customer_id=customer_id)
        assignment = RoleAssignment(
            id=uuid4(),
            role_id=role.id,
            group_id=group_id,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=ns.id,
            scope_namespace_type=None,
            scope_customer_id=None,
        )
        membership = GroupMembership(
            group_id=group_id,
            member_type=MemberType.USER,
            member_id=user_id,
            customer_id=customer_id,
        )
        group = Group(id=group_id, name="memory-owner:x", customer_id=customer_id)
        deps, _ = _deps(
            resolver_returns=ns,
            memberships_for_user=(membership,),
            assignments=(assignment,),
            roles={role.id: role},
            groups={group_id: group},
        )
        result = await authorize_memory_access(
            action=ACTION_MEMORY_READ,
            agent_id=agent_id,
            customer_id=customer_id,
            caller_user_id=user_id,
            caller_agent_id=None,
            deps=deps,
        )
        assert result is ns

    async def test_user_with_reader_grant_cannot_write(self) -> None:
        agent_id = uuid4()
        customer_id = uuid4()
        user_id = uuid4()
        group_id = uuid4()
        role = _reader_role()
        ns = _ns_row(agent_id=agent_id, customer_id=customer_id)
        assignment = RoleAssignment(
            id=uuid4(),
            role_id=role.id,
            group_id=group_id,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=ns.id,
            scope_namespace_type=None,
            scope_customer_id=None,
        )
        membership = GroupMembership(
            group_id=group_id,
            member_type=MemberType.USER,
            member_id=user_id,
            customer_id=customer_id,
        )
        group = Group(id=group_id, name="x", customer_id=customer_id)
        deps, _ = _deps(
            resolver_returns=ns,
            memberships_for_user=(membership,),
            assignments=(assignment,),
            roles={role.id: role},
            groups={group_id: group},
        )
        with pytest.raises(MemoryAccessDenied):
            await authorize_memory_access(
                action=ACTION_MEMORY_WRITE,
                agent_id=agent_id,
                customer_id=customer_id,
                caller_user_id=user_id,
                caller_agent_id=None,
                deps=deps,
            )

    async def test_owner_role_grants_extract(self) -> None:
        agent_id = uuid4()
        customer_id = uuid4()
        user_id = uuid4()
        group_id = uuid4()
        role = _owner_role()
        ns = _ns_row(agent_id=agent_id, customer_id=customer_id)
        assignment = RoleAssignment(
            id=uuid4(),
            role_id=role.id,
            group_id=group_id,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=ns.id,
            scope_namespace_type=None,
            scope_customer_id=None,
        )
        membership = GroupMembership(
            group_id=group_id,
            member_type=MemberType.USER,
            member_id=user_id,
            customer_id=customer_id,
        )
        group = Group(id=group_id, name="x", customer_id=customer_id)
        deps, _ = _deps(
            resolver_returns=ns,
            memberships_for_user=(membership,),
            assignments=(assignment,),
            roles={role.id: role},
            groups={group_id: group},
        )
        result = await authorize_memory_access(
            action=ACTION_MEMORY_EXTRACT,
            agent_id=agent_id,
            customer_id=customer_id,
            caller_user_id=user_id,
            caller_agent_id=None,
            deps=deps,
        )
        assert result is ns
