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
    AclCache,
    Group,
    GroupMembership,
    MemberType,
    Namespace as AclNamespace,
    Role,
    RoleAssignment,
    ScopeType,
)
from threetears.core.namespaces import build_agent_namespace_name
from threetears.registry.rbac_authorizer import RbacEvaluatorAuthorizer


def _cache(membership_loader: Any, grant_loader: Any) -> AclCache:
    """build an :class:`AclCache` wrapping the supplied loaders.

    test helper used in place of the per-test ``acl_cache=None`` +
    loaders kwargs the authorizer used to accept. one cache is
    created per test so ttl state never leaks between cases.

    :param membership_loader: actor -> memberships resolver
    :ptype membership_loader: Any
    :param grant_loader: groups -> assignments resolver
    :ptype grant_loader: Any
    :return: fresh cache wrapping the loaders
    :rtype: AclCache
    """
    return AclCache(
        membership_loader=membership_loader,
        grant_loader=grant_loader,
    )


class _StubToolNamespace:
    """duck-typed namespace entity exposing the fields the evaluator reads."""

    __slots__ = (
        "id",
        "namespace_type",
        "owner_agent_id",
        "customer_id",
        "owner_namespace",
    )

    def __init__(
        self,
        *,
        id: UUID,
        namespace_type: str,
        owner_agent_id: UUID | None,
        customer_id: UUID | None,
        owner_namespace: str | None = None,
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
        :param owner_namespace: canonical name of the namespace that
            owns this row, or ``None``. a platform tool namespace has no
            owner at all -- it is reached by grant, which is what
            ``test_platform_tool_requires_explicit_grant`` asserts --
            so ``None`` is the shape these cases actually want
        :ptype owner_namespace: str | None
        """
        self.id = id
        self.namespace_type = namespace_type
        self.owner_agent_id = owner_agent_id
        self.customer_id = customer_id
        self.owner_namespace = owner_namespace


# parity-exempt: subset shim for the post-discovery NamespaceCollection exposing only the get_by_name lookup the rbac authorizer evaluates against
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
        self.last_get_by_name: str | None = None

    async def get_by_name(self, name: str) -> _StubToolNamespace | None:
        """return the preconfigured stub (may be ``None``).

        records the name argument on ``self.last_get_by_name`` so
        tests can assert the authorizer constructs the canonical
        sanitized form (``tools.<sanitized-mcp>.<sanitized-version>``)
        from the dispatch's ``(mcp_name, mcp_version)`` rather than
        passing the raw ``mcp_name`` directly.

        :param name: tool namespace name
        :ptype name: str
        :return: preconfigured stub or ``None``
        :rtype: _StubToolNamespace | None
        """
        self.last_get_by_name = name
        return self._entity


# parity-with: threetears.agent.acl.loader.MembershipLoader
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
        self,
        user_id: UUID,
    ) -> tuple[GroupMembership, ...]:
        """return user fixture memberships.

        :param user_id: user UUID
        :ptype user_id: UUID
        :return: memberships or empty tuple
        :rtype: tuple[GroupMembership, ...]
        """
        return self._users.get(user_id, ())

    async def load_for_agent(
        self,
        agent_id: UUID,
    ) -> tuple[GroupMembership, ...]:
        """return agent fixture memberships.

        :param agent_id: agent UUID
        :ptype agent_id: UUID
        :return: memberships or empty tuple
        :rtype: tuple[GroupMembership, ...]
        """
        return self._agents.get(agent_id, ())


# parity-with: threetears.agent.acl.loader.GrantLoader
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
        self,
        role_ids: tuple[UUID, ...],
    ) -> dict[UUID, Role]:
        """return role mapping for every supplied role id.

        :param role_ids: requested role UUIDs
        :ptype role_ids: tuple[UUID, ...]
        :return: role mapping subset
        :rtype: dict[UUID, Role]
        """
        return {rid: self._roles[rid] for rid in role_ids if rid in self._roles}

    async def load_groups(
        self,
        group_ids: tuple[UUID, ...],
    ) -> dict[UUID, Any]:
        """return group mapping for every supplied group id.

        :param group_ids: requested group UUIDs
        :ptype group_ids: tuple[UUID, ...]
        :return: group mapping subset
        :rtype: dict[UUID, Any]
        """
        return {gid: self._groups[gid] for gid in group_ids if gid in self._groups}


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
            acl_cache=_cache(
                _FakeMembershipLoader(
                    users={user_id: (user_membership,)},
                    agents={agent_id: (agent_membership,)},
                ),
                _FakeGrantLoader(
                    assignments={group_id: (assignment,)},
                    roles={role_id: role},
                    groups={group_id: group},
                ),
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
            str(agent_id),
            str(user_id),
            "3tears.calc",
            "1.0",
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_deny_cross_customer_tool_call(self) -> None:
        """a fully valid grant in customer A does NOT authorize a tool owned by customer B.

        The evaluator's cross-customer wall drops a customer-scoped membership/group against
        another customer's namespace, so even an otherwise-complete two-sided grant yields a
        DENY. This locks tenant isolation on the tool-call path specifically (v0.13.9 auth C5):
        post-C3 the proxy feeds ``is_authorized`` the VERIFIED caller identity, and a verified
        caller cannot reach across the customer line into another tenant's tools -- the wall
        keys off the DB-sourced membership customer, never the (self-asserted) envelope.
        """
        user_id = uuid4()
        agent_id = uuid4()
        customer_a = uuid4()  # the caller's grant lives here
        customer_b = uuid4()  # the tool namespace is owned here
        group_id = uuid4()
        role_id = uuid4()
        namespace_id = uuid4()

        group = Group(id=group_id, name="tool-access:agent-abc", customer_id=customer_a)
        role = Role(
            id=role_id,
            name="ToolCaller",
            permissions={"tool": frozenset({"tool.call"})},
            is_built_in=True,
        )
        user_membership = GroupMembership(
            group_id=group_id, member_id=user_id, member_type=MemberType.USER, customer_id=customer_a
        )
        agent_membership = GroupMembership(
            group_id=group_id, member_id=agent_id, member_type=MemberType.AGENT, customer_id=customer_a
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
            acl_cache=_cache(
                _FakeMembershipLoader(
                    users={user_id: (user_membership,)},
                    agents={agent_id: (agent_membership,)},
                ),
                _FakeGrantLoader(
                    assignments={group_id: (assignment,)},
                    roles={role_id: role},
                    groups={group_id: group},
                ),
            ),
            namespace_collection=_FakeNamespaceCollection(
                _StubToolNamespace(
                    id=namespace_id,
                    namespace_type="tool",
                    owner_agent_id=None,
                    customer_id=customer_b,  # DIFFERENT customer than the grant
                ),
            ),
        )

        result = await authorizer.is_authorized(
            str(agent_id),
            str(user_id),
            "3tears.calc",
            "1.0",
        )
        assert result is False  # cross-customer wall denies despite a valid same-customer grant

    @pytest.mark.asyncio
    async def test_deny_when_no_memberships(self) -> None:
        """actor without memberships is denied."""
        user_id = uuid4()
        agent_id = uuid4()
        customer_id = uuid4()
        namespace_id = uuid4()

        authorizer = RbacEvaluatorAuthorizer(
            acl_cache=_cache(_FakeMembershipLoader(), _FakeGrantLoader()),
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
            str(agent_id),
            str(user_id),
            "3tears.calc",
            "1.0",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_deny_when_user_id_is_none(self) -> None:
        """dispatch without user identity is denied (defense in depth)."""
        agent_id = uuid4()
        customer_id = uuid4()
        namespace_id = uuid4()

        authorizer = RbacEvaluatorAuthorizer(
            acl_cache=_cache(_FakeMembershipLoader(), _FakeGrantLoader()),
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
            str(agent_id),
            None,
            "3tears.calc",
            "1.0",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_deny_when_namespace_lookup_returns_none(self) -> None:
        """missing tool namespace row (registration race) -> denied."""
        agent_id = uuid4()
        user_id = uuid4()

        authorizer = RbacEvaluatorAuthorizer(
            acl_cache=_cache(_FakeMembershipLoader(), _FakeGrantLoader()),
            namespace_collection=_FakeNamespaceCollection(None),
        )

        result = await authorizer.is_authorized(
            str(agent_id),
            str(user_id),
            "3tears.unknown",
            "1.0",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_platform_tool_requires_explicit_grant(self) -> None:
        """platform tool (NULL customer) still needs a real assignment."""
        user_id = uuid4()
        agent_id = uuid4()
        namespace_id = uuid4()

        authorizer = RbacEvaluatorAuthorizer(
            acl_cache=_cache(_FakeMembershipLoader(), _FakeGrantLoader()),
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
            str(agent_id),
            str(user_id),
            "platform.time.now",
            "1.0",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_ownership_supplies_the_agent_side_with_no_agent_grant(self) -> None:
        """the owner short-circuit is WIRED at tool dispatch, not only in the evaluator.

        This is the site the ownership move names, and an evaluator unit
        test cannot cover it: the authorizer builds its OWN
        ``AclNamespace``, so a construction site that forgot to carry
        ``owner_namespace`` would leave every owner falling through to
        grants it does not hold -- and the failure would look like an
        ordinary deny.

        Dispatch stays TWO-SIDED. Ownership opens the agent side and
        nothing more, so the user still needs a real grant; what this
        asserts is that the AGENT side needs no assignment at all when
        the agent owns the namespace. The companion below removes the
        ownership and shows the same setup then denies, which is what
        makes this test about ownership rather than about the user
        grant.
        """
        user_id = uuid4()
        agent_id = uuid4()
        customer_id = uuid4()
        group_id = uuid4()
        role_id = uuid4()
        namespace_id = uuid4()

        group = Group(id=group_id, name="user-side", customer_id=customer_id)
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
        assignment = RoleAssignment(
            id=uuid4(),
            group_id=group_id,
            role_id=role_id,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=namespace_id,
            scope_namespace_type=None,
            scope_customer_id=None,
        )
        # the USER is a member; the AGENT is a member of nothing.
        loaders = (
            _FakeMembershipLoader(users={user_id: (user_membership,)}, agents={}),
            _FakeGrantLoader(
                assignments={group_id: (assignment,)},
                roles={role_id: role},
                groups={group_id: group},
            ),
        )

        owned = RbacEvaluatorAuthorizer(
            acl_cache=_cache(*loaders),
            namespace_collection=_FakeNamespaceCollection(
                _StubToolNamespace(
                    id=namespace_id,
                    namespace_type="tool",
                    owner_agent_id=agent_id,
                    customer_id=customer_id,
                    owner_namespace=build_agent_namespace_name(agent_id),
                ),
            ),
        )
        assert await owned.is_authorized(str(agent_id), str(user_id), "example.own_tool", "1.0") is True

    @pytest.mark.asyncio
    async def test_the_same_call_denies_when_the_agent_does_not_own_the_row(self) -> None:
        """the A/B that makes the case above about ownership.

        identical user grant, identical absence of an agent grant, and
        the ONLY difference is whose namespace the row records as its
        owner. without that, the previous test would pass just as well
        against a gate that had stopped checking the agent side at all.
        """
        user_id = uuid4()
        agent_id = uuid4()
        other_agent = uuid4()
        customer_id = uuid4()
        group_id = uuid4()
        role_id = uuid4()
        namespace_id = uuid4()

        group = Group(id=group_id, name="user-side", customer_id=customer_id)
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
        assignment = RoleAssignment(
            id=uuid4(),
            group_id=group_id,
            role_id=role_id,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=namespace_id,
            scope_namespace_type=None,
            scope_customer_id=None,
        )

        peer = RbacEvaluatorAuthorizer(
            acl_cache=_cache(
                _FakeMembershipLoader(users={user_id: (user_membership,)}, agents={}),
                _FakeGrantLoader(
                    assignments={group_id: (assignment,)},
                    roles={role_id: role},
                    groups={group_id: group},
                ),
            ),
            namespace_collection=_FakeNamespaceCollection(
                _StubToolNamespace(
                    id=namespace_id,
                    namespace_type="tool",
                    owner_agent_id=other_agent,
                    customer_id=customer_id,
                    owner_namespace=build_agent_namespace_name(other_agent),
                ),
            ),
        )
        assert await peer.is_authorized(str(agent_id), str(user_id), "example.own_tool", "1.0") is False

    @pytest.mark.asyncio
    async def test_invalid_agent_id_denied(self) -> None:
        """malformed ``agent_id`` surfaces as a deny rather than crash."""
        user_id = uuid4()
        namespace_id = uuid4()
        customer_id = uuid4()

        authorizer = RbacEvaluatorAuthorizer(
            acl_cache=_cache(_FakeMembershipLoader(), _FakeGrantLoader()),
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
            "not-a-uuid",
            str(user_id),
            "3tears.calc",
            "1.0",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_lookup_uses_the_canonical_name(self) -> None:
        """authorizer constructs the canonical name from (mcp_name, mcp_version).

        the dispatch carries the natural ``mcp_name`` /
        ``mcp_version`` shape (e.g. ``3tears.admin.customer_management`` /
        ``1.0``); the namespace ``name`` column carries the rooted form
        (``tools.3tears.admin.customer_management.1-0``), in which the
        mcp name is interpolated unchanged and only the version is
        sanitized. without this canonicalization step the lookup never
        resolves the row and every dispatch denies even when a valid
        grant exists.

        the expectation is asserted BOTH against the shared grammar and
        against the literal, because those fail differently: the first
        catches this call site drifting away from the builder, and the
        second catches the builder itself changing shape under a
        deployment whose rows were written by the older one.
        """
        from threetears.core.namespaces import build_tool_namespace_name

        user_id = uuid4()
        agent_id = uuid4()
        ns_coll = _FakeNamespaceCollection(None)

        authorizer = RbacEvaluatorAuthorizer(
            acl_cache=_cache(_FakeMembershipLoader(), _FakeGrantLoader()),
            namespace_collection=ns_coll,
        )

        await authorizer.is_authorized(
            str(agent_id),
            str(user_id),
            "3tears.admin.customer_management",
            "1.0",
        )

        assert ns_coll.last_get_by_name == build_tool_namespace_name(
            "3tears.admin.customer_management",
            "1.0",
        )
        assert ns_coll.last_get_by_name == "tools.3tears.admin.customer_management.1-0"


class TestAMalformedToolNameDeniesRatherThanRaises:
    """the dispatch envelope is untrusted, and this is the authorization hot path.

    ``tool_name`` arrives on a proxy request. The namespace-name builder
    REFUSES a name carrying an empty component, and correctly so -- no
    registration could have produced one. But an exception escaping here
    is worse than a deny in two ways: the caller gets an error instead of
    a refusal it can read, and an authorizer that can raise is one whose
    failure mode is not "denied".

    So a name that cannot compose denies, which is the same answer as a
    name that composes and matches no row.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_name",
        ["a..b", ".leading", "trailing.", "tools.already.rooted", "tools"],
    )
    async def test_a_malformed_tool_name_denies(self, tool_name: str) -> None:
        ns_coll = _FakeNamespaceCollection(None)
        authorizer = RbacEvaluatorAuthorizer(
            acl_cache=_cache(_FakeMembershipLoader(), _FakeGrantLoader()),
            namespace_collection=ns_coll,
        )

        result = await authorizer.is_authorized(str(uuid4()), str(uuid4()), tool_name, "1.0")

        assert result is False

    @pytest.mark.asyncio
    async def test_a_malformed_tool_name_is_never_looked_up(self) -> None:
        """no row can be named by it, so the lookup would be a wasted read."""
        ns_coll = _FakeNamespaceCollection(None)
        authorizer = RbacEvaluatorAuthorizer(
            acl_cache=_cache(_FakeMembershipLoader(), _FakeGrantLoader()),
            namespace_collection=ns_coll,
        )

        await authorizer.is_authorized(str(uuid4()), str(uuid4()), "a..b", "1.0")

        assert ns_coll.last_get_by_name is None

    @pytest.mark.asyncio
    async def test_a_malformed_version_denies(self) -> None:
        ns_coll = _FakeNamespaceCollection(None)
        authorizer = RbacEvaluatorAuthorizer(
            acl_cache=_cache(_FakeMembershipLoader(), _FakeGrantLoader()),
            namespace_collection=ns_coll,
        )

        result = await authorizer.is_authorized(str(uuid4()), str(uuid4()), "pentest.sqlmap", "")

        assert result is False
