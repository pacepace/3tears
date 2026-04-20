"""unit tests for :class:`RbacEvaluatorAuthorizer`.

namespace-task-01 phase 2 replaced :class:`KvAgentToolAuthorizer`
with :class:`RbacEvaluatorAuthorizer`. three-tier-task-01 phase D
retired the bespoke resolver callable alias and the parallel
tool-namespace-row value object; the authorizer now takes a
``NamespaceCollection`` handle directly. these tests drive the
authorizer with an in-memory Collection stand-in that duck-types
:meth:`get_by_name`, keeping the unit focused on the
evaluator-interaction branches:

- allow path: user + agent in a group that holds a role granting
  ``tool.call`` on the tool namespace
- deny path: user + agent membership empty
- platform tool path: ``owner_agent_id=None`` + ``customer_id=None``
  row is reachable only via an explicit assignment (no implicit
  ownership short-circuit)
- user_id absent: dispatch without user identity denied (defense in
  depth)
- unresolvable tool name: Collection returns ``None`` (tool
  registered race) -> denied
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.acl import (
    Group,
    GroupMembership,
    MemberType,
    Namespace as AclNamespace,
    Role,
    RoleAssignment,
    ScopeType,
)
from threetears.registry.rbac_authorizer import RbacEvaluatorAuthorizer


class _StubToolNamespace:
    """duck-typed namespace entity exposing the four fields the evaluator reads."""

    __slots__ = ("id", "namespace_type", "owner_agent_id", "customer_id")

    def __init__(
        self,
        *,
        id: UUID,
        namespace_type: str,
        owner_agent_id: UUID | None,
        customer_id: UUID | None,
    ) -> None:
        """initialize a stub namespace entity.

        :param id: namespace UUID
        :ptype id: UUID
        :param namespace_type: namespace type discriminator
        :ptype namespace_type: str
        :param owner_agent_id: owning agent UUID or ``None``
        :ptype owner_agent_id: UUID | None
        :param customer_id: owning customer UUID or ``None``
        :ptype customer_id: UUID | None
        """
        self.id = id
        self.namespace_type = namespace_type
        self.owner_agent_id = owner_agent_id
        self.customer_id = customer_id


class _FakeNamespaceCollection:
    """duck-typed ``NamespaceCollection`` with a preconfigured ``get_by_name``.

    the authorizer only reads :meth:`get_by_name`, so the fake
    intentionally omits the rest of the Collection surface.
    """

    def __init__(self, entity: _StubToolNamespace | None) -> None:
        """store the entity returned for every ``get_by_name`` call.

        :param entity: preconfigured stub entity or ``None``
        :ptype entity: _StubToolNamespace | None
        """
        self._entity = entity

    async def get_by_name(self, name: str) -> _StubToolNamespace | None:
        """return the preconfigured stub (may be ``None``).

        :param name: tool namespace name (unused)
        :ptype name: str
        :return: preconfigured stub or ``None``
        :rtype: _StubToolNamespace | None
        """
        _ = name
        return self._entity


class _FakeMembershipLoader:
    """in-memory membership loader keyed on actor UUID."""

    def __init__(
        self,
        *,
        users: dict[UUID, tuple[GroupMembership, ...]] | None = None,
        agents: dict[UUID, tuple[GroupMembership, ...]] | None = None,
    ) -> None:
        """store fixture memberships for user + agent sides.

        :param users: user membership fixture
        :ptype users: dict[UUID, tuple[GroupMembership, ...]] | None
        :param agents: agent membership fixture
        :ptype agents: dict[UUID, tuple[GroupMembership, ...]] | None
        """
        self._users = users or {}
        self._agents = agents or {}

    async def load_for_user(
        self, user_id: UUID,
    ) -> tuple[GroupMembership, ...]:
        """return user fixture memberships.

        :param user_id: user UUID
        :ptype user_id: UUID
        :return: memberships or empty tuple
        :rtype: tuple[GroupMembership, ...]
        """
        return self._users.get(user_id, ())

    async def load_for_agent(
        self, agent_id: UUID,
    ) -> tuple[GroupMembership, ...]:
        """return agent fixture memberships.

        :param agent_id: agent UUID
        :ptype agent_id: UUID
        :return: memberships or empty tuple
        :rtype: tuple[GroupMembership, ...]
        """
        return self._agents.get(agent_id, ())


class _FakeGrantLoader:
    """in-memory grant loader keyed on group UUID."""

    def __init__(
        self,
        *,
        assignments: dict[UUID, tuple[RoleAssignment, ...]] | None = None,
        roles: dict[UUID, Role] | None = None,
        groups: dict[UUID, Group] | None = None,
    ) -> None:
        """store fixture assignments / roles / groups.

        :param assignments: assignments keyed on group id
        :ptype assignments: dict[UUID, tuple[RoleAssignment, ...]] | None
        :param roles: role fixture keyed on role id
        :ptype roles: dict[UUID, Role] | None
        :param groups: group fixture keyed on group id
        :ptype groups: dict[UUID, Group] | None
        """
        self._assignments = assignments or {}
        self._roles = roles or {}
        self._groups = groups or {}

    async def load_assignments_for_groups(
        self,
        group_ids: tuple[UUID, ...],
        namespace: AclNamespace,
    ) -> tuple[RoleAssignment, ...]:
        """return every assignment across the supplied groups.

        the evaluator re-checks coverage so over-returning is safe.

        :param group_ids: candidate group UUIDs
        :ptype group_ids: tuple[UUID, ...]
        :param namespace: namespace under evaluation (ignored)
        :ptype namespace: AclNamespace
        :return: assignments
        :rtype: tuple[RoleAssignment, ...]
        """
        out: list[RoleAssignment] = []
        for gid in group_ids:
            out.extend(self._assignments.get(gid, ()))
        return tuple(out)

    async def load_roles(
        self, role_ids: tuple[UUID, ...],
    ) -> dict[UUID, Role]:
        """return role mapping for every supplied role id.

        :param role_ids: requested role UUIDs
        :ptype role_ids: tuple[UUID, ...]
        :return: role mapping subset
        :rtype: dict[UUID, Role]
        """
        return {rid: self._roles[rid] for rid in role_ids if rid in self._roles}

    async def load_groups(
        self, group_ids: tuple[UUID, ...],
    ) -> dict[UUID, Any]:
        """return group mapping for every supplied group id.

        :param group_ids: requested group UUIDs
        :ptype group_ids: tuple[UUID, ...]
        :return: group mapping subset
        :rtype: dict[UUID, Any]
        """
        return {
            gid: self._groups[gid] for gid in group_ids if gid in self._groups
        }


class TestRbacEvaluatorAuthorizer:
    """cover allow / deny / platform / no-user / resolver-miss."""

    @pytest.mark.asyncio
    async def test_allow_when_user_and_agent_both_grant(self) -> None:
        """valid two-sided grant on the tool namespace allows dispatch."""
        user_id = uuid4()
        agent_id = uuid4()
        customer_id = uuid4()
        group_id = uuid4()
        role_id = uuid4()
        namespace_id = uuid4()

        group = Group(
            id=group_id,
            name="tool-access:agent-abc",
            customer_id=customer_id,
        )
        role = Role(
            id=role_id,
            name="ToolCaller",
            permissions={"tool": frozenset({"tool.call"})},
            is_built_in=True,
        )
        user_membership = GroupMembership(
            group_id=group_id,
            member_id=user_id,
            member_type=MemberType.USER,
            customer_id=customer_id,
        )
        agent_membership = GroupMembership(
            group_id=group_id,
            member_id=agent_id,
            member_type=MemberType.AGENT,
            customer_id=customer_id,
        )
        assignment = RoleAssignment(
            id=uuid4(),
            group_id=group_id,
            role_id=role_id,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=namespace_id,
            scope_namespace_type=None,
            scope_customer_id=None,
        )

        authorizer = RbacEvaluatorAuthorizer(
            acl_cache=None,
            membership_loader=_FakeMembershipLoader(
                users={user_id: (user_membership,)},
                agents={agent_id: (agent_membership,)},
            ),
            grant_loader=_FakeGrantLoader(
                assignments={group_id: (assignment,)},
                roles={role_id: role},
                groups={group_id: group},
            ),
            namespace_collection=_FakeNamespaceCollection(
                _StubToolNamespace(
                    id=namespace_id,
                    namespace_type="tool",
                    owner_agent_id=None,
                    customer_id=customer_id,
                ),
            ),
        )

        result = await authorizer.is_authorized(
            str(agent_id), str(user_id), "aibots.calc",
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_deny_when_no_memberships(self) -> None:
        """actor without memberships is denied."""
        user_id = uuid4()
        agent_id = uuid4()
        customer_id = uuid4()
        namespace_id = uuid4()

        authorizer = RbacEvaluatorAuthorizer(
            acl_cache=None,
            membership_loader=_FakeMembershipLoader(),
            grant_loader=_FakeGrantLoader(),
            namespace_collection=_FakeNamespaceCollection(
                _StubToolNamespace(
                    id=namespace_id,
                    namespace_type="tool",
                    owner_agent_id=None,
                    customer_id=customer_id,
                ),
            ),
        )

        result = await authorizer.is_authorized(
            str(agent_id), str(user_id), "aibots.calc",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_deny_when_user_id_is_none(self) -> None:
        """dispatch without user identity is denied (defense in depth)."""
        agent_id = uuid4()
        customer_id = uuid4()
        namespace_id = uuid4()

        authorizer = RbacEvaluatorAuthorizer(
            acl_cache=None,
            membership_loader=_FakeMembershipLoader(),
            grant_loader=_FakeGrantLoader(),
            namespace_collection=_FakeNamespaceCollection(
                _StubToolNamespace(
                    id=namespace_id,
                    namespace_type="tool",
                    owner_agent_id=None,
                    customer_id=customer_id,
                ),
            ),
        )

        result = await authorizer.is_authorized(
            str(agent_id), None, "aibots.calc",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_deny_when_namespace_lookup_returns_none(self) -> None:
        """missing tool namespace row (registration race) -> denied."""
        agent_id = uuid4()
        user_id = uuid4()

        authorizer = RbacEvaluatorAuthorizer(
            acl_cache=None,
            membership_loader=_FakeMembershipLoader(),
            grant_loader=_FakeGrantLoader(),
            namespace_collection=_FakeNamespaceCollection(None),
        )

        result = await authorizer.is_authorized(
            str(agent_id), str(user_id), "aibots.unknown",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_platform_tool_requires_explicit_grant(self) -> None:
        """platform tool (NULL customer) still needs a real assignment."""
        user_id = uuid4()
        agent_id = uuid4()
        namespace_id = uuid4()

        authorizer = RbacEvaluatorAuthorizer(
            acl_cache=None,
            membership_loader=_FakeMembershipLoader(),
            grant_loader=_FakeGrantLoader(),
            namespace_collection=_FakeNamespaceCollection(
                _StubToolNamespace(
                    id=namespace_id,
                    namespace_type="tool",
                    owner_agent_id=None,
                    customer_id=None,
                ),
            ),
        )

        # no memberships => no assignments => deny, even on platform tool
        result = await authorizer.is_authorized(
            str(agent_id), str(user_id), "platform.time.now",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_invalid_agent_id_denied(self) -> None:
        """malformed ``agent_id`` surfaces as a deny rather than crash."""
        user_id = uuid4()
        namespace_id = uuid4()
        customer_id = uuid4()

        authorizer = RbacEvaluatorAuthorizer(
            acl_cache=None,
            membership_loader=_FakeMembershipLoader(),
            grant_loader=_FakeGrantLoader(),
            namespace_collection=_FakeNamespaceCollection(
                _StubToolNamespace(
                    id=namespace_id,
                    namespace_type="tool",
                    owner_agent_id=None,
                    customer_id=customer_id,
                ),
            ),
        )

        result = await authorizer.is_authorized(
            "not-a-uuid", str(user_id), "aibots.calc",
        )
        assert result is False
