"""shared test fixtures for agent-memory.

the ``permissive_memory_authorizer`` fixture constructs a
:class:`MemoryAuthorizerDependencies` bundle that authorizes every
evaluate call and materializes a deterministic namespace for every
``(agent_id, customer_id)`` a caller asks about. three-tier-task-01
phase D retired the bespoke resolver / ensurer callables; the fixture
now passes in in-memory Collection stand-ins with the method surface
:func:`authorize_memory_access` + :func:`ensure_memory_owner_assignment`
rely on. tests that DO exercise rbac construct their own bundle with
realistic loaders + real Collections.

using this fixture makes the test-only bypass explicit at every
call site — there is no silent ``authorizer=None`` shim in the
production API, so tests declare "permissive" in the same line
they pass the authorizer.
"""

from __future__ import annotations


from uuid import UUID, uuid4

import pytest

from threetears.agent.acl import (
    AclCache,
    Group,
    GroupMembership,
    MemberType,
    Role,
    RoleAssignment,
    ScopeType,
)
from .authorizer_fakes import (
    FakeGroupCollection,
    FakeGroupMemberCollection,
    FakeNamespaceCollection,
    FakeRoleAssignmentCollection,
    FakeRoleCollection,
)
from threetears.agent.memory.authorize import (
    ACTION_MEMORY_EXTRACT,
    ACTION_MEMORY_READ,
    ACTION_MEMORY_WRITE,
    MemoryAuthorizerDependencies,
)


class _PermissiveMembershipLoader:
    """MembershipLoader that returns a single synthetic group membership.

    the membership binds every caller (user or agent) to a common
    permissive group; combined with
    :class:`_PermissiveGrantLoader` the caller is authorized for
    every memory action on every namespace.
    """

    def __init__(self, group_id: UUID) -> None:
        """store the shared group UUID every actor is a member of.

        :param group_id: UUID of the synthetic permissive group
        :ptype group_id: UUID
        """
        self._group_id = group_id

    async def load_for_user(
        self,
        user_id: UUID,
    ) -> tuple[GroupMembership, ...]:
        """return one membership naming the user as member of the group.

        :param user_id: user UUID
        :ptype user_id: UUID
        :return: tuple with one :class:`GroupMembership`
        :rtype: tuple[GroupMembership, ...]
        """
        return (
            GroupMembership(
                group_id=self._group_id,
                member_type=MemberType.USER,
                member_id=user_id,
                customer_id=None,
            ),
        )

    async def load_for_agent(
        self,
        agent_id: UUID,
    ) -> tuple[GroupMembership, ...]:
        """return one membership naming the agent as member of the group.

        :param agent_id: agent UUID
        :ptype agent_id: UUID
        :return: tuple with one :class:`GroupMembership`
        :rtype: tuple[GroupMembership, ...]
        """
        return (
            GroupMembership(
                group_id=self._group_id,
                member_type=MemberType.AGENT,
                member_id=agent_id,
                customer_id=None,
            ),
        )

    async def load_for_group(self, group_id: UUID) -> tuple[GroupMembership, ...]:
        """return parent-group memberships -- none; these fixtures use flat groups.

        :param group_id: child group UUID
        :ptype group_id: UUID
        :return: empty tuple
        :rtype: tuple[GroupMembership, ...]
        """
        return ()


class _PermissiveGrantLoader:
    """GrantLoader that surfaces one all-actions role for every namespace.

    the single synthetic role carries every canonical memory action,
    scoped to ``ScopeType.ALL`` so the evaluator admits the
    assignment against any memory namespace.
    """

    def __init__(self, group_id: UUID, role_id: UUID) -> None:
        """store the shared group + role UUIDs reused across every load.

        :param group_id: UUID of the synthetic permissive group
        :ptype group_id: UUID
        :param role_id: UUID of the synthetic permissive role
        :ptype role_id: UUID
        """
        self._group_id = group_id
        self._role_id = role_id
        self._role = Role(
            id=role_id,
            name="PermissiveTestRole",
            permissions={
                "memory": frozenset(
                    {
                        ACTION_MEMORY_READ,
                        ACTION_MEMORY_WRITE,
                        ACTION_MEMORY_EXTRACT,
                    },
                ),
            },
            is_built_in=True,
        )

    async def load_assignments_for_groups(
        self,
        group_ids: tuple[UUID, ...],
        namespace: object,
    ) -> tuple[RoleAssignment, ...]:
        """return one all-scope assignment per known group id.

        :param group_ids: group UUIDs to inspect
        :ptype group_ids: tuple[UUID, ...]
        :param namespace: namespace under evaluation (unused)
        :ptype namespace: object
        :return: tuple of role assignments
        :rtype: tuple[RoleAssignment, ...]
        """
        assignments: list[RoleAssignment] = []
        for group_id in group_ids:
            if group_id != self._group_id:
                continue
            assignments.append(
                RoleAssignment(
                    id=uuid4(),
                    role_id=self._role_id,
                    group_id=group_id,
                    scope_type=ScopeType.ALL,
                    scope_namespace_id=None,
                    scope_namespace_type=None,
                    scope_customer_id=None,
                ),
            )
        return tuple(assignments)

    async def load_roles(
        self,
        role_ids: tuple[UUID, ...],
    ) -> dict[UUID, Role]:
        """return the synthetic role for every requested role id it owns.

        :param role_ids: role UUIDs to resolve
        :ptype role_ids: tuple[UUID, ...]
        :return: mapping role_id -> :class:`Role`
        :rtype: dict[UUID, Role]
        """
        return {rid: self._role for rid in role_ids if rid == self._role_id}

    async def load_groups(
        self,
        group_ids: tuple[UUID, ...],
    ) -> dict[UUID, object]:
        """resolve every known group id to a synthetic platform-scoped group.

        the evaluator requires every assignment's group to resolve
        through this method or the assignment is skipped (see
        :func:`threetears.agent.acl.evaluator._walk_assignments`).
        return a platform-scoped group for the shared synthetic id so
        the permissive assignment actually contributes actions.

        :param group_ids: group UUIDs to resolve
        :ptype group_ids: tuple[UUID, ...]
        :return: mapping group_id -> :class:`Group`
        :rtype: dict[UUID, object]
        """
        result: dict[UUID, object] = {}
        for gid in group_ids:
            if gid == self._group_id:
                result[gid] = Group(
                    id=gid,
                    name="PermissiveTestGroup",
                    customer_id=None,
                )
        return result


@pytest.fixture
def permissive_memory_authorizer() -> MemoryAuthorizerDependencies:
    """TEST-ONLY permissive :class:`MemoryAuthorizerDependencies` bundle.

    every evaluator call allows; every
    :meth:`NamespaceCollection.get_by_owner_and_customer` call
    returns a synthetic stub namespace; every ensure-path Collection
    method is a no-op. intended for unit tests that focus on memory
    plumbing (SQL shape, embeddings, dedup, ...) and not on rbac
    behaviour. tests that exercise rbac construct their own bundle
    with production loaders + Collections.

    three-tier-task-01 phase D retired the bespoke resolver /
    ensurer callables the fixture previously supplied; the returned
    bundle carries permissive Collection stand-ins whose method
    surface matches what the production Collections expose.

    :return: permissive authorizer dependency bundle
    :rtype: MemoryAuthorizerDependencies
    """
    shared_group_id = uuid4()
    shared_role_id = uuid4()
    membership_loader = _PermissiveMembershipLoader(shared_group_id)
    grant_loader = _PermissiveGrantLoader(shared_group_id, shared_role_id)
    return MemoryAuthorizerDependencies(
        acl_cache=AclCache(
            membership_loader=membership_loader,
            grant_loader=grant_loader,
        ),
        namespace_collection=FakeNamespaceCollection(),
        group_collection=FakeGroupCollection(),
        group_member_collection=FakeGroupMemberCollection(),
        role_collection=FakeRoleCollection(),
        role_assignment_collection=FakeRoleAssignmentCollection(),
    )
