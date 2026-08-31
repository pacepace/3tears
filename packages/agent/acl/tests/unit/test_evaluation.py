"""evaluator truth table.

every branch of :func:`evaluate_with_trail` is exercised here:

- ownership shortcut (agent is the namespace owner) -> allow with no trails
- single-actor evaluation (user only) -> action set = user-side union
- single-actor evaluation (agent only, not owner) -> action set = agent-side union
- intersection (both sides) -> action set = user ∩ agent
- empty user side in intersection -> deny with `LimitingSide.NEITHER`
- empty agent side in intersection -> deny with `LimitingSide.NEITHER`
- wildcard role permissions
- type_customer scope coverage
- all scope coverage (platform-admin universal)
- mixed-membership group splits across user / agent sides
- cross-customer group never contributes (cross-customer wall)
- limiting_side classification (USER, AGENT, EQUAL, USER on owner shortcut)

every test asserts on the trail structure as well as the bool
decision per RBAC-15: a regression that flips which assignment
grants access fails a specific test, not just the bool.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from threetears.agent.acl import (
    EvaluationContext,
    Group,
    GroupMembership,
    LimitingSide,
    MemberType,
    Namespace,
    Role,
    RoleAssignment,
    ScopeType,
    WILDCARD_RESOURCE_TYPE,
    evaluate_decision,
    evaluate_with_trail,
)
from threetears.agent.acl.cache import ActorMembershipKey
from threetears.agent.acl.evaluator import _agent_owns_namespace

from threetears.core.namespaces import build_agent_namespace_name

from ._fake_loaders import FakeStore, make_cache


# ---------------------------------------------------------------------------
# fixture builders
# ---------------------------------------------------------------------------


def _ns(
    *,
    customer_id: UUID,
    owner_agent_id: UUID,
    namespace_type: str = "workspace",
) -> Namespace:
    """build a :class:`Namespace` with a fresh id.

    :param customer_id: customer UUID
    :ptype customer_id: UUID
    :param owner_agent_id: owning agent UUID
    :ptype owner_agent_id: UUID
    :param namespace_type: type discriminator
    :ptype namespace_type: str
    :return: namespace record
    :rtype: Namespace
    """
    return Namespace(
        id=uuid4(),
        customer_id=customer_id,
        namespace_type=namespace_type,
        owner_agent_id=owner_agent_id,
        # a real row records its owner as the NAMESPACE that owns it,
        # and for an agent-owned row that is the agent's own namespace.
        # the fixture writes both columns so the assertions below keep
        # asking the question they always asked.
        owner_namespace=build_agent_namespace_name(owner_agent_id),
    )


def _role(
    *,
    name: str,
    permissions: dict[str, list[str]],
    is_built_in: bool = True,
) -> Role:
    """build a :class:`Role` with frozenset-coerced permissions.

    :param name: role name
    :ptype name: str
    :param permissions: ``{resource_type: [actions]}`` mapping
    :ptype permissions: dict[str, list[str]]
    :param is_built_in: whether the role is platform-shipped
    :ptype is_built_in: bool
    :return: role record
    :rtype: Role
    """
    return Role(
        id=uuid4(),
        name=name,
        permissions={k: frozenset(v) for k, v in permissions.items()},
        is_built_in=is_built_in,
    )


def _group(*, name: str, customer_id: UUID | None) -> Group:
    """build a :class:`Group` with a fresh id.

    :param name: group name
    :ptype name: str
    :param customer_id: owning customer UUID, or ``None`` for platform scope
    :ptype customer_id: UUID | None
    :return: group record
    :rtype: Group
    """
    return Group(id=uuid4(), name=name, customer_id=customer_id)


def _assignment(
    *,
    role: Role,
    group: Group,
    scope_type: ScopeType,
    scope_namespace_id: UUID | None = None,
    scope_namespace_type: str | None = None,
    scope_customer_id: UUID | None = None,
) -> RoleAssignment:
    """build a :class:`RoleAssignment` with a fresh id.

    :param role: role being granted
    :ptype role: Role
    :param group: group receiving the grant
    :ptype group: Group
    :param scope_type: scope shape
    :ptype scope_type: ScopeType
    :param scope_namespace_id: namespace UUID for namespace-scope
    :ptype scope_namespace_id: UUID | None
    :param scope_namespace_type: namespace type for type_customer-scope
    :ptype scope_namespace_type: str | None
    :param scope_customer_id: customer UUID for type_customer-scope
    :ptype scope_customer_id: UUID | None
    :return: assignment record
    :rtype: RoleAssignment
    """
    return RoleAssignment(
        id=uuid4(),
        role_id=role.id,
        group_id=group.id,
        scope_type=scope_type,
        scope_namespace_id=scope_namespace_id,
        scope_namespace_type=scope_namespace_type,
        scope_customer_id=scope_customer_id,
    )


# ---------------------------------------------------------------------------
# 1. ownership short-circuit
# ---------------------------------------------------------------------------


class TestAgentOwnershipShortCircuit:
    """agent owns the namespace -> agent side bypasses every loader."""

    @pytest.mark.asyncio
    async def test_agent_only_owner_allows_without_loaders(self) -> None:
        """agent-only call where agent is the owner allows with empty trails."""
        customer = uuid4()
        agent = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=agent)
        store = FakeStore()
        ctx = EvaluationContext(
            namespace=namespace,
            action="read",
            agent_id=agent,
        )
        result = await evaluate_with_trail(
            ctx,
            cache=make_cache(store),
        )
        assert result.decision is True
        assert result.agent_owner_short_circuited is True
        assert result.agent_trails == ()
        assert result.trails == ()

    @pytest.mark.asyncio
    async def test_intersection_owner_uses_user_side_as_cap(self) -> None:
        """owner agent + user with read grant -> effective = read."""
        customer = uuid4()
        agent = uuid4()
        user = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=agent)

        # user has Reader on this namespace.
        reader = _role(name="Reader", permissions={"workspace": ["read"]})
        engineering = _group(name="Engineering", customer_id=customer)
        membership = GroupMembership(
            group_id=engineering.id,
            member_type=MemberType.USER,
            member_id=user,
            customer_id=customer,
        )
        assignment = _assignment(
            role=reader,
            group=engineering,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=namespace.id,
        )
        store = FakeStore()
        store.add_role(reader)
        store.add_group(engineering)
        store.add_membership(membership)
        store.add_assignment(assignment)

        ctx = EvaluationContext(
            namespace=namespace,
            action="read",
            user_id=user,
            agent_id=agent,
        )
        result = await evaluate_with_trail(
            ctx,
            cache=make_cache(store),
        )
        assert result.decision is True
        assert result.effective_actions == frozenset({"read"})
        assert result.agent_owner_short_circuited is True
        # user is the cap because the agent has every action by ownership
        assert result.limiting_side == LimitingSide.USER
        # user side has one trail naming the engineering group
        assert len(result.user_trails) == 1
        trail = result.user_trails[0]
        assert trail.group.name == "Engineering"
        assert trail.role.name == "Reader"
        assert trail.contributed_actions == frozenset({"read"})

    @pytest.mark.asyncio
    async def test_intersection_owner_writes_when_user_can(self) -> None:
        """owner agent + user with write grant -> effective contains write."""
        customer = uuid4()
        agent = uuid4()
        user = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=agent)
        writer = _role(
            name="Writer",
            permissions={"workspace": ["read", "write"]},
        )
        admins = _group(name="Admins", customer_id=customer)
        store = FakeStore()
        store.add_role(writer)
        store.add_group(admins)
        store.add_membership(
            GroupMembership(
                group_id=admins.id,
                member_type=MemberType.USER,
                member_id=user,
                customer_id=customer,
            )
        )
        store.add_assignment(
            _assignment(
                role=writer,
                group=admins,
                scope_type=ScopeType.NAMESPACE,
                scope_namespace_id=namespace.id,
            )
        )
        ctx = EvaluationContext(
            namespace=namespace,
            action="write",
            user_id=user,
            agent_id=agent,
        )
        result = await evaluate_with_trail(
            ctx,
            cache=make_cache(store),
        )
        assert result.decision is True
        assert "write" in result.effective_actions


# ---------------------------------------------------------------------------
# 2. single-side evaluation
# ---------------------------------------------------------------------------


class TestSingleSideUser:
    """user-only evaluation collects trails on the single side."""

    @pytest.mark.asyncio
    async def test_user_with_grant_allows(self) -> None:
        """user in a group with a covering assignment is allowed."""
        customer = uuid4()
        owner = uuid4()
        user = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=owner)
        reader = _role(name="Reader", permissions={"workspace": ["read"]})
        engineering = _group(name="Engineering", customer_id=customer)
        store = FakeStore()
        store.add_role(reader)
        store.add_group(engineering)
        store.add_membership(
            GroupMembership(
                group_id=engineering.id,
                member_type=MemberType.USER,
                member_id=user,
                customer_id=customer,
            )
        )
        store.add_assignment(
            _assignment(
                role=reader,
                group=engineering,
                scope_type=ScopeType.NAMESPACE,
                scope_namespace_id=namespace.id,
            )
        )
        ctx = EvaluationContext(
            namespace=namespace,
            action="read",
            user_id=user,
        )
        result = await evaluate_with_trail(
            ctx,
            cache=make_cache(store),
        )
        assert result.decision is True
        assert result.effective_actions == frozenset({"read"})
        assert len(result.trails) == 1
        assert result.trails[0].role.name == "Reader"
        # intersection-only fields stay at defaults
        assert result.user_trails == ()
        assert result.agent_trails == ()
        assert result.limiting_side == LimitingSide.NEITHER

    @pytest.mark.asyncio
    async def test_user_no_grant_denies(self) -> None:
        """user in no group is denied with no trails."""
        customer = uuid4()
        owner = uuid4()
        user = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=owner)
        store = FakeStore()
        ctx = EvaluationContext(
            namespace=namespace,
            action="read",
            user_id=user,
        )
        result = await evaluate_with_trail(
            ctx,
            cache=make_cache(store),
        )
        assert result.decision is False
        assert result.effective_actions == frozenset()
        assert result.trails == ()


class TestSingleSideAgent:
    """agent-only evaluation, agent is not the owner."""

    @pytest.mark.asyncio
    async def test_agent_in_group_with_grant_allows(self) -> None:
        """agent in a group with a covering assignment is allowed."""
        customer = uuid4()
        owner = uuid4()
        agent = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=owner)
        # owner != agent so the shortcut does not fire
        assert namespace.owner_agent_id != agent
        reader = _role(name="Reader", permissions={"workspace": ["read"]})
        bots = _group(name="ReadonlyBots", customer_id=customer)
        store = FakeStore()
        store.add_role(reader)
        store.add_group(bots)
        store.add_membership(
            GroupMembership(
                group_id=bots.id,
                member_type=MemberType.AGENT,
                member_id=agent,
                customer_id=customer,
            )
        )
        store.add_assignment(
            _assignment(
                role=reader,
                group=bots,
                scope_type=ScopeType.NAMESPACE,
                scope_namespace_id=namespace.id,
            )
        )
        ctx = EvaluationContext(
            namespace=namespace,
            action="read",
            agent_id=agent,
        )
        result = await evaluate_with_trail(
            ctx,
            cache=make_cache(store),
        )
        assert result.decision is True
        assert result.effective_actions == frozenset({"read"})
        assert result.agent_owner_short_circuited is False


# ---------------------------------------------------------------------------
# 3. intersection truth table
# ---------------------------------------------------------------------------


class TestIntersection:
    """user ∩ agent -- the production hot path."""

    @pytest.mark.asyncio
    async def test_user_admin_agent_reader_caps_to_read(self) -> None:
        """admin user calling through reader bot -> effective = {read}."""
        customer = uuid4()
        owner = uuid4()
        user = uuid4()
        agent = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=owner)

        admin = _role(
            name="Admin",
            permissions={WILDCARD_RESOURCE_TYPE: ["read", "write", "delete"]},
        )
        reader = _role(name="Reader", permissions={"workspace": ["read"]})
        admins = _group(name="Admins", customer_id=customer)
        bots = _group(name="ReadonlyBots", customer_id=customer)
        store = FakeStore()
        store.add_role(admin)
        store.add_role(reader)
        store.add_group(admins)
        store.add_group(bots)
        store.add_membership(
            GroupMembership(
                group_id=admins.id,
                member_type=MemberType.USER,
                member_id=user,
                customer_id=customer,
            )
        )
        store.add_membership(
            GroupMembership(
                group_id=bots.id,
                member_type=MemberType.AGENT,
                member_id=agent,
                customer_id=customer,
            )
        )
        store.add_assignment(
            _assignment(
                role=admin,
                group=admins,
                scope_type=ScopeType.NAMESPACE,
                scope_namespace_id=namespace.id,
            )
        )
        store.add_assignment(
            _assignment(
                role=reader,
                group=bots,
                scope_type=ScopeType.NAMESPACE,
                scope_namespace_id=namespace.id,
            )
        )

        ctx = EvaluationContext(
            namespace=namespace,
            action="write",
            user_id=user,
            agent_id=agent,
        )
        result = await evaluate_with_trail(
            ctx,
            cache=make_cache(store),
        )
        # admin ∩ reader = {read}; write is denied
        assert result.decision is False
        assert result.effective_actions == frozenset({"read"})
        assert result.user_actions == frozenset({"read", "write", "delete"})
        assert result.agent_actions == frozenset({"read"})
        assert result.limiting_side == LimitingSide.AGENT
        # both trails populated
        assert len(result.user_trails) == 1
        assert len(result.agent_trails) == 1
        assert result.user_trails[0].role.name == "Admin"
        assert result.agent_trails[0].role.name == "Reader"

    @pytest.mark.asyncio
    async def test_empty_user_side_denies(self) -> None:
        """user with no group memberships denies even when agent allows."""
        customer = uuid4()
        owner = uuid4()
        user = uuid4()
        agent = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=owner)
        reader = _role(name="Reader", permissions={"workspace": ["read"]})
        bots = _group(name="Bots", customer_id=customer)
        store = FakeStore()
        store.add_role(reader)
        store.add_group(bots)
        store.add_membership(
            GroupMembership(
                group_id=bots.id,
                member_type=MemberType.AGENT,
                member_id=agent,
                customer_id=customer,
            )
        )
        store.add_assignment(
            _assignment(
                role=reader,
                group=bots,
                scope_type=ScopeType.NAMESPACE,
                scope_namespace_id=namespace.id,
            )
        )
        ctx = EvaluationContext(
            namespace=namespace,
            action="read",
            user_id=user,
            agent_id=agent,
        )
        result = await evaluate_with_trail(
            ctx,
            cache=make_cache(store),
        )
        assert result.decision is False
        assert result.effective_actions == frozenset()
        assert result.limiting_side == LimitingSide.NEITHER
        assert result.user_trails == ()
        # agent-side trail still surfaces -- the operator wants to see
        # what the agent could have done if the user had matching grants
        assert len(result.agent_trails) == 1

    @pytest.mark.asyncio
    async def test_empty_agent_side_denies(self) -> None:
        """agent with no membership denies even when user allows.

        agent is not the namespace owner so the shortcut does not fire.
        """
        customer = uuid4()
        owner = uuid4()
        user = uuid4()
        agent = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=owner)
        admin = _role(
            name="Admin",
            permissions={WILDCARD_RESOURCE_TYPE: ["read", "write"]},
        )
        admins = _group(name="Admins", customer_id=customer)
        store = FakeStore()
        store.add_role(admin)
        store.add_group(admins)
        store.add_membership(
            GroupMembership(
                group_id=admins.id,
                member_type=MemberType.USER,
                member_id=user,
                customer_id=customer,
            )
        )
        store.add_assignment(
            _assignment(
                role=admin,
                group=admins,
                scope_type=ScopeType.NAMESPACE,
                scope_namespace_id=namespace.id,
            )
        )
        ctx = EvaluationContext(
            namespace=namespace,
            action="read",
            user_id=user,
            agent_id=agent,
        )
        result = await evaluate_with_trail(
            ctx,
            cache=make_cache(store),
        )
        assert result.decision is False
        assert result.effective_actions == frozenset()
        assert result.limiting_side == LimitingSide.NEITHER

    @pytest.mark.asyncio
    async def test_equal_action_sets_classify_equal(self) -> None:
        """user and agent both readers -> EQUAL limiting side."""
        customer = uuid4()
        owner = uuid4()
        user = uuid4()
        agent = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=owner)
        reader = _role(name="Reader", permissions={"workspace": ["read"]})
        users = _group(name="Users", customer_id=customer)
        bots = _group(name="Bots", customer_id=customer)
        store = FakeStore()
        store.add_role(reader)
        store.add_group(users)
        store.add_group(bots)
        store.add_membership(
            GroupMembership(
                group_id=users.id,
                member_type=MemberType.USER,
                member_id=user,
                customer_id=customer,
            )
        )
        store.add_membership(
            GroupMembership(
                group_id=bots.id,
                member_type=MemberType.AGENT,
                member_id=agent,
                customer_id=customer,
            )
        )
        store.add_assignment(
            _assignment(
                role=reader,
                group=users,
                scope_type=ScopeType.NAMESPACE,
                scope_namespace_id=namespace.id,
            )
        )
        store.add_assignment(
            _assignment(
                role=reader,
                group=bots,
                scope_type=ScopeType.NAMESPACE,
                scope_namespace_id=namespace.id,
            )
        )
        ctx = EvaluationContext(
            namespace=namespace,
            action="read",
            user_id=user,
            agent_id=agent,
        )
        result = await evaluate_with_trail(
            ctx,
            cache=make_cache(store),
        )
        assert result.decision is True
        assert result.limiting_side == LimitingSide.EQUAL


# ---------------------------------------------------------------------------
# 4. wildcard role
# ---------------------------------------------------------------------------


class TestWildcardRole:
    """role with ``"*"`` permissions covers every resource type."""

    @pytest.mark.asyncio
    async def test_wildcard_role_covers_workspace_type(self) -> None:
        """admin user (wildcard role) on workspace namespace -> read allowed."""
        customer = uuid4()
        owner = uuid4()
        user = uuid4()
        namespace = _ns(
            customer_id=customer,
            owner_agent_id=owner,
            namespace_type="workspace",
        )
        admin = _role(
            name="Admin",
            permissions={WILDCARD_RESOURCE_TYPE: ["read", "write"]},
        )
        admins = _group(name="Admins", customer_id=customer)
        store = FakeStore()
        store.add_role(admin)
        store.add_group(admins)
        store.add_membership(
            GroupMembership(
                group_id=admins.id,
                member_type=MemberType.USER,
                member_id=user,
                customer_id=customer,
            )
        )
        store.add_assignment(
            _assignment(
                role=admin,
                group=admins,
                scope_type=ScopeType.NAMESPACE,
                scope_namespace_id=namespace.id,
            )
        )
        ctx = EvaluationContext(
            namespace=namespace,
            action="write",
            user_id=user,
        )
        result = await evaluate_with_trail(
            ctx,
            cache=make_cache(store),
        )
        assert result.decision is True
        assert "write" in result.effective_actions
        assert result.trails[0].contributed_actions == frozenset({"read", "write"})

    @pytest.mark.asyncio
    async def test_wildcard_unioned_with_typed_bucket(self) -> None:
        """role with both wildcard and type-specific buckets unions both."""
        customer = uuid4()
        owner = uuid4()
        user = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=owner, namespace_type="workspace")
        # wildcard grants read; workspace bucket grants write
        role = _role(
            name="WorkspaceEditor",
            permissions={
                WILDCARD_RESOURCE_TYPE: ["read"],
                "workspace": ["write"],
            },
        )
        editors = _group(name="Editors", customer_id=customer)
        store = FakeStore()
        store.add_role(role)
        store.add_group(editors)
        store.add_membership(
            GroupMembership(
                group_id=editors.id,
                member_type=MemberType.USER,
                member_id=user,
                customer_id=customer,
            )
        )
        store.add_assignment(
            _assignment(
                role=role,
                group=editors,
                scope_type=ScopeType.NAMESPACE,
                scope_namespace_id=namespace.id,
            )
        )
        ctx = EvaluationContext(
            namespace=namespace,
            action="write",
            user_id=user,
        )
        result = await evaluate_with_trail(
            ctx,
            cache=make_cache(store),
        )
        assert result.decision is True
        assert result.effective_actions == frozenset({"read", "write"})


# ---------------------------------------------------------------------------
# 5. type_customer + all scopes
# ---------------------------------------------------------------------------


class TestTypeCustomerScope:
    """``type_customer`` scope covers every namespace of a type within a customer."""

    @pytest.mark.asyncio
    async def test_covers_matching_type_and_customer(self) -> None:
        """auditor with type_customer scope covers any workspace ns."""
        customer = uuid4()
        owner = uuid4()
        user = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=owner, namespace_type="workspace")
        auditor = _role(name="Auditor", permissions={"workspace": ["read"]})
        auditors = _group(name="Auditors", customer_id=customer)
        store = FakeStore()
        store.add_role(auditor)
        store.add_group(auditors)
        store.add_membership(
            GroupMembership(
                group_id=auditors.id,
                member_type=MemberType.USER,
                member_id=user,
                customer_id=customer,
            )
        )
        store.add_assignment(
            _assignment(
                role=auditor,
                group=auditors,
                scope_type=ScopeType.TYPE_CUSTOMER,
                scope_namespace_type="workspace",
                scope_customer_id=customer,
            )
        )
        ctx = EvaluationContext(
            namespace=namespace,
            action="read",
            user_id=user,
        )
        result = await evaluate_with_trail(
            ctx,
            cache=make_cache(store),
        )
        assert result.decision is True

    @pytest.mark.asyncio
    async def test_does_not_cover_different_type(self) -> None:
        """type_customer scope on workspace does not reach an agent ns."""
        customer = uuid4()
        owner = uuid4()
        user = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=owner, namespace_type="agent")
        auditor = _role(name="Auditor", permissions={"workspace": ["read"]})
        auditors = _group(name="Auditors", customer_id=customer)
        store = FakeStore()
        store.add_role(auditor)
        store.add_group(auditors)
        store.add_membership(
            GroupMembership(
                group_id=auditors.id,
                member_type=MemberType.USER,
                member_id=user,
                customer_id=customer,
            )
        )
        store.add_assignment(
            _assignment(
                role=auditor,
                group=auditors,
                scope_type=ScopeType.TYPE_CUSTOMER,
                scope_namespace_type="workspace",
                scope_customer_id=customer,
            )
        )
        ctx = EvaluationContext(
            namespace=namespace,
            action="read",
            user_id=user,
        )
        result = await evaluate_with_trail(
            ctx,
            cache=make_cache(store),
        )
        assert result.decision is False


class TestAllScope:
    """``all`` scope is universal, intended for platform admin only."""

    @pytest.mark.asyncio
    async def test_all_scope_covers_any_namespace(self) -> None:
        """assignment with scope_type=ALL covers a customer-scoped namespace."""
        customer = uuid4()
        owner = uuid4()
        user = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=owner)
        support = _role(
            name="Support",
            permissions={WILDCARD_RESOURCE_TYPE: ["read"]},
        )
        # platform-scoped group
        platform_admins = _group(name="PlatformAdmins", customer_id=None)
        store = FakeStore()
        store.add_role(support)
        store.add_group(platform_admins)
        store.add_membership(
            GroupMembership(
                group_id=platform_admins.id,
                member_type=MemberType.USER,
                member_id=user,
                customer_id=None,  # admin not bound to one customer
            )
        )
        store.add_assignment(
            _assignment(
                role=support,
                group=platform_admins,
                scope_type=ScopeType.ALL,
            )
        )
        ctx = EvaluationContext(
            namespace=namespace,
            action="read",
            user_id=user,
        )
        result = await evaluate_with_trail(
            ctx,
            cache=make_cache(store),
        )
        assert result.decision is True


# ---------------------------------------------------------------------------
# 6. mixed-membership group
# ---------------------------------------------------------------------------


class TestMixedMembership:
    """one group containing both user and agent members splits across sides."""

    @pytest.mark.asyncio
    async def test_user_member_does_not_satisfy_agent_side(self) -> None:
        """mixed group: user member never counts as agent-side membership."""
        customer = uuid4()
        owner = uuid4()
        user = uuid4()
        agent = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=owner)
        reader = _role(name="Reader", permissions={"workspace": ["read"]})
        mixed = _group(name="Mixed", customer_id=customer)
        store = FakeStore()
        store.add_role(reader)
        store.add_group(mixed)
        # user is a USER member of mixed; agent is NOT a member at all
        store.add_membership(
            GroupMembership(
                group_id=mixed.id,
                member_type=MemberType.USER,
                member_id=user,
                customer_id=customer,
            )
        )
        store.add_assignment(
            _assignment(
                role=reader,
                group=mixed,
                scope_type=ScopeType.NAMESPACE,
                scope_namespace_id=namespace.id,
            )
        )
        ctx = EvaluationContext(
            namespace=namespace,
            action="read",
            user_id=user,
            agent_id=agent,
        )
        result = await evaluate_with_trail(
            ctx,
            cache=make_cache(store),
        )
        # agent side empty -> deny
        assert result.decision is False
        assert result.agent_actions == frozenset()
        assert result.user_actions == frozenset({"read"})


# ---------------------------------------------------------------------------
# 7. cross-customer wall
# ---------------------------------------------------------------------------


class TestCrossCustomerWall:
    """customer-scoped group never contributes against another customer's namespace."""

    @pytest.mark.asyncio
    async def test_customer_scoped_group_blocked_against_other_customer(self) -> None:
        """group scoped to customer A cannot contribute on customer B's namespace.

        loader returns the membership row honestly (with customer A on
        the membership); the evaluator's filter rejects it because
        the namespace is customer B.
        """
        customer_a = uuid4()
        customer_b = uuid4()
        owner = uuid4()
        user = uuid4()
        namespace = _ns(customer_id=customer_b, owner_agent_id=owner)
        reader = _role(name="Reader", permissions={"workspace": ["read"]})
        a_engineering = _group(name="Engineering-A", customer_id=customer_a)
        store = FakeStore()
        store.add_role(reader)
        store.add_group(a_engineering)
        store.add_membership(
            GroupMembership(
                group_id=a_engineering.id,
                member_type=MemberType.USER,
                member_id=user,
                customer_id=customer_a,
            )
        )
        store.add_assignment(
            _assignment(
                role=reader,
                group=a_engineering,
                scope_type=ScopeType.NAMESPACE,
                scope_namespace_id=namespace.id,
            )
        )
        ctx = EvaluationContext(
            namespace=namespace,
            action="read",
            user_id=user,
        )
        result = await evaluate_with_trail(
            ctx,
            cache=make_cache(store),
        )
        # filter cuts the membership row -> no eligible groups -> empty side
        assert result.decision is False
        assert result.trails == ()


# ---------------------------------------------------------------------------
# 8. evaluate_decision wrapper
# ---------------------------------------------------------------------------


class TestEvaluateDecisionWrapper:
    """:func:`evaluate_decision` returns the bool from the trail-mode call."""

    @pytest.mark.asyncio
    async def test_returns_bool(self) -> None:
        """wrapper returns True on allow, False on deny."""
        customer = uuid4()
        owner = uuid4()
        user = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=owner)
        store = FakeStore()
        ctx = EvaluationContext(
            namespace=namespace,
            action="read",
            user_id=user,
        )
        decision = await evaluate_decision(
            ctx,
            cache=make_cache(store),
        )
        assert decision is False  # no grants

    @pytest.mark.asyncio
    async def test_owner_shortcut_returns_true(self) -> None:
        """owner shortcut surfaces True through the wrapper."""
        customer = uuid4()
        agent = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=agent)
        store = FakeStore()
        ctx = EvaluationContext(
            namespace=namespace,
            action="anything",
            agent_id=agent,
        )
        decision = await evaluate_decision(
            ctx,
            cache=make_cache(store),
        )
        assert decision is True


# ---------------------------------------------------------------------------
# 9. context validation
# ---------------------------------------------------------------------------


class TestContextValidation:
    """evaluator rejects an :class:`EvaluationContext` without any actor."""

    @pytest.mark.asyncio
    async def test_no_actor_raises(self) -> None:
        """user_id and agent_id both None -> ValueError."""
        customer = uuid4()
        owner = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=owner)
        store = FakeStore()
        ctx = EvaluationContext(namespace=namespace, action="read")
        with pytest.raises(ValueError, match="at least one"):
            await evaluate_with_trail(
                ctx,
                cache=make_cache(store),
            )


# ---------------------------------------------------------------------------
# 10. limiting_side classification
# ---------------------------------------------------------------------------


class TestLimitingSide:
    """:class:`LimitingSide` classification covers each branch."""

    @pytest.mark.asyncio
    async def test_user_subset_classifies_user(self) -> None:
        """user actions are a strict subset of agent actions -> USER."""
        customer = uuid4()
        owner = uuid4()
        user = uuid4()
        agent = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=owner)
        reader = _role(name="Reader", permissions={"workspace": ["read"]})
        admin = _role(
            name="Admin",
            permissions={"workspace": ["read", "write", "delete"]},
        )
        ug = _group(name="UG", customer_id=customer)
        ag = _group(name="AG", customer_id=customer)
        store = FakeStore()
        store.add_role(reader)
        store.add_role(admin)
        store.add_group(ug)
        store.add_group(ag)
        store.add_membership(
            GroupMembership(
                group_id=ug.id,
                member_type=MemberType.USER,
                member_id=user,
                customer_id=customer,
            )
        )
        store.add_membership(
            GroupMembership(
                group_id=ag.id,
                member_type=MemberType.AGENT,
                member_id=agent,
                customer_id=customer,
            )
        )
        store.add_assignment(
            _assignment(
                role=reader,
                group=ug,
                scope_type=ScopeType.NAMESPACE,
                scope_namespace_id=namespace.id,
            )
        )
        store.add_assignment(
            _assignment(
                role=admin,
                group=ag,
                scope_type=ScopeType.NAMESPACE,
                scope_namespace_id=namespace.id,
            )
        )
        ctx = EvaluationContext(
            namespace=namespace,
            action="read",
            user_id=user,
            agent_id=agent,
        )
        result = await evaluate_with_trail(
            ctx,
            cache=make_cache(store),
        )
        assert result.decision is True
        # user_actions = {read}; agent_actions = {read, write, delete}
        # user is the strict subset, so user is the cap.
        assert result.limiting_side == LimitingSide.USER


# ---------------------------------------------------------------------------
# 13. scope coverage of platform-shared namespaces (customer_id NULL)
# ---------------------------------------------------------------------------


class TestPlatformSharedNamespaceScopeCoverage:
    """which assignment shapes reach a ``customer_id=NULL`` namespace.

    platform built-in tools materialize as ``tool`` namespaces with
    both owner columns NULL (``ToolServer._upsert_tool_namespace``), and
    the evaluator does not special-case them -- coverage is decided
    entirely by :meth:`RoleAssignment.covers`. these tests pin the
    resulting matrix, because
    :class:`~threetears.registry.rbac_authorizer.RbacEvaluatorAuthorizer`
    documents it for admins seeding the first grant and a docstring
    cannot be checked by the test suite.

    the counter-intuitive entry is the last one: ``type_customer``
    matches a NULL-customer row only when the assignment's own
    ``scope_customer_id`` is NULL too, which makes it a platform-wide
    grant rather than the per-customer one its name suggests.
    """

    def _platform_tool_namespace(self) -> Namespace:
        """build a platform built-in tool namespace (both owners NULL).

        :return: namespace record with ``customer_id`` and
            ``owner_agent_id`` NULL
        :rtype: Namespace
        """
        return Namespace(
            id=uuid4(),
            customer_id=None,
            namespace_type="tool",
            owner_agent_id=None,
        )

    def test_type_customer_bound_to_a_customer_never_covers(self) -> None:
        """the shape an admin reaches for first, and it grants nothing."""
        namespace = self._platform_tool_namespace()
        role = _role(name="tool_user", permissions={"tool": ["tool.call"]})
        group = _group(name="default tool access", customer_id=uuid4())
        assignment = _assignment(
            role=role,
            group=group,
            scope_type=ScopeType.TYPE_CUSTOMER,
            scope_namespace_type="tool",
            scope_customer_id=uuid4(),
        )

        assert assignment.covers(namespace) is False

    def test_type_customer_with_null_customer_covers(self) -> None:
        """NULL scope_customer_id matches the row's NULL customer_id.

        a platform-scope assignment: ``RoleAssignmentCollection``
        derives ``row_scope='platform'`` for precisely this shape.
        """
        namespace = self._platform_tool_namespace()
        role = _role(name="tool_user", permissions={"tool": ["tool.call"]})
        group = _group(name="default tool access", customer_id=None)
        assignment = _assignment(
            role=role,
            group=group,
            scope_type=ScopeType.TYPE_CUSTOMER,
            scope_namespace_type="tool",
            scope_customer_id=None,
        )

        assert assignment.covers(namespace) is True

    def test_type_customer_with_null_customer_still_honours_the_type(self) -> None:
        """the NULL-customer shape is not a backdoor past the type predicate."""
        namespace = self._platform_tool_namespace()
        role = _role(name="ws_user", permissions={"workspace": ["read"]})
        group = _group(name="default workspace access", customer_id=None)
        assignment = _assignment(
            role=role,
            group=group,
            scope_type=ScopeType.TYPE_CUSTOMER,
            scope_namespace_type="workspace",
            scope_customer_id=None,
        )

        assert assignment.covers(namespace) is False

    def test_namespace_scope_covers(self) -> None:
        """the per-tool grant, and the narrowest shape available."""
        namespace = self._platform_tool_namespace()
        role = _role(name="tool_user", permissions={"tool": ["tool.call"]})
        group = _group(name="default tool access", customer_id=uuid4())
        assignment = _assignment(
            role=role,
            group=group,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=namespace.id,
        )

        assert assignment.covers(namespace) is True

    def test_all_scope_covers(self) -> None:
        """platform-admin universal grant reaches it unconditionally."""
        namespace = self._platform_tool_namespace()
        role = _role(name="super", permissions={WILDCARD_RESOURCE_TYPE: ["tool.call"]})
        group = _group(name="platform admins", customer_id=None)
        assignment = _assignment(
            role=role,
            group=group,
            scope_type=ScopeType.ALL,
        )

        assert assignment.covers(namespace) is True

    def test_agent_spun_tool_is_reached_by_the_ordinary_customer_grant(self) -> None:
        """the contrast: a real customer_id makes type_customer work normally."""
        customer = uuid4()
        namespace = Namespace(
            id=uuid4(),
            customer_id=customer,
            namespace_type="tool",
            owner_agent_id=uuid4(),
        )
        role = _role(name="tool_user", permissions={"tool": ["tool.call"]})
        group = _group(name="default tool access", customer_id=customer)
        assignment = _assignment(
            role=role,
            group=group,
            scope_type=ScopeType.TYPE_CUSTOMER,
            scope_namespace_type="tool",
            scope_customer_id=customer,
        )

        assert assignment.covers(namespace) is True


# ---------------------------------------------------------------------------
# ownership is recorded as a NAMESPACE, not as an agent
# ---------------------------------------------------------------------------


class TestOwnershipIsANamespace:
    """the short-circuit reads ``owner_namespace`` and nothing else.

    ownership moved off ``owner_agent_id`` because an agent is only one
    of the four kinds of principal that own namespaces -- a tool pod, a
    capability source and an in-hub component own them too, and none of
    those is an agent. an owner expressed as a namespace covers all
    four with one column.
    """

    @pytest.mark.asyncio
    async def test_an_agent_owner_grants_the_owner(self) -> None:
        customer = uuid4()
        agent = uuid4()
        namespace = Namespace(
            id=uuid4(),
            customer_id=customer,
            namespace_type="workspace",
            owner_agent_id=agent,
            owner_namespace=build_agent_namespace_name(agent),
        )
        result = await evaluate_with_trail(
            EvaluationContext(namespace=namespace, action="read", agent_id=agent),
            cache=make_cache(FakeStore()),
        )
        assert result.decision is True
        assert result.agent_owner_short_circuited is True

    @pytest.mark.asyncio
    async def test_an_agent_owner_denies_a_peer(self) -> None:
        customer = uuid4()
        owner = uuid4()
        peer = uuid4()
        namespace = Namespace(
            id=uuid4(),
            customer_id=customer,
            namespace_type="workspace",
            owner_agent_id=owner,
            owner_namespace=build_agent_namespace_name(owner),
        )
        result = await evaluate_with_trail(
            EvaluationContext(namespace=namespace, action="read", agent_id=peer),
            cache=make_cache(FakeStore()),
        )
        assert result.decision is False
        assert result.agent_owner_short_circuited is False

    @pytest.mark.asyncio
    async def test_a_provider_owner_grants_the_owner(self) -> None:
        """the owner need not be an agent namespace at all.

        a tool leaf is owned by its PROVIDER node (``tools.pentest``),
        not by an agent. the short-circuit compares names and does not
        care which kind of namespace the owner is, which is the whole
        reason the column names a namespace.
        """
        customer = uuid4()
        agent = uuid4()
        namespace = Namespace(
            id=uuid4(),
            customer_id=customer,
            namespace_type="tool",
            owner_agent_id=None,
            # the caller's own namespace IS the provider node
            owner_namespace=build_agent_namespace_name(agent),
        )
        result = await evaluate_with_trail(
            EvaluationContext(namespace=namespace, action="tool.call", agent_id=agent),
            cache=make_cache(FakeStore()),
        )
        assert result.decision is True
        assert result.agent_owner_short_circuited is True

    @pytest.mark.asyncio
    async def test_a_provider_owner_denies_a_peer(self) -> None:
        customer = uuid4()
        peer = uuid4()
        namespace = Namespace(
            id=uuid4(),
            customer_id=customer,
            namespace_type="tool",
            owner_agent_id=None,
            owner_namespace="tools.pentest",
        )
        result = await evaluate_with_trail(
            EvaluationContext(namespace=namespace, action="tool.call", agent_id=peer),
            cache=make_cache(FakeStore()),
        )
        assert result.decision is False
        assert result.agent_owner_short_circuited is False

    @pytest.mark.asyncio
    async def test_a_null_owner_matches_nothing(self) -> None:
        """``owner_namespace`` of ``None`` grants NOBODY.

        this is the ``is not None`` guard, asserted directly rather than
        through a decision that could pass for another reason. under the
        previous ``owner_agent_id == agent_id`` comparison a null owner
        could not match, because the enclosing branch guaranteed a
        non-null agent id on the other side. that accident does not
        survive the move to a string: two ``None`` values compare EQUAL
        in python, so without the explicit guard an unowned namespace
        would be handed to any caller whose derived name came out null.
        """
        namespace = Namespace(
            id=uuid4(),
            customer_id=uuid4(),
            namespace_type="workspace",
            owner_agent_id=None,
            owner_namespace=None,
        )
        assert _agent_owns_namespace(namespace, uuid4()) is False
        # and the same row denies every caller, not merely this one
        for _ in range(5):
            assert _agent_owns_namespace(namespace, uuid4()) is False

    @pytest.mark.asyncio
    async def test_owner_agent_id_alone_no_longer_grants(self) -> None:
        """a row whose only owner marking is ``owner_agent_id`` grants nobody.

        this is the stale-gate assertion at the evaluator. after this
        change ``owner_agent_id`` still records which agent owns a
        workspace or a memory store, and it is true wherever it is set
        -- but it no longer DECIDES. a gate still keyed on it would be
        checking a column that does not govern.
        """
        agent = uuid4()
        namespace = Namespace(
            id=uuid4(),
            customer_id=uuid4(),
            namespace_type="workspace",
            owner_agent_id=agent,
            owner_namespace=None,
        )
        result = await evaluate_with_trail(
            EvaluationContext(namespace=namespace, action="read", agent_id=agent),
            cache=make_cache(FakeStore()),
        )
        assert result.decision is False
        assert result.agent_owner_short_circuited is False

    @pytest.mark.asyncio
    async def test_the_comparison_is_exact(self) -> None:
        """no case folding, no whitespace tolerance, no prefix matching.

        the owner is one namespace, not a subtree of them: a caller that
        owns ``agents.<x>`` does not thereby own everything under it,
        and a value carrying stray whitespace or the wrong case grants
        nothing rather than being silently normalised into one that
        grants.
        """
        agent = uuid4()
        canonical = build_agent_namespace_name(agent)
        for spelled in (canonical.upper(), canonical + " ", " " + canonical, canonical + ".child"):
            namespace = Namespace(
                id=uuid4(),
                customer_id=uuid4(),
                namespace_type="workspace",
                owner_agent_id=agent,
                owner_namespace=spelled,
            )
            assert _agent_owns_namespace(namespace, agent) is False

    @pytest.mark.asyncio
    async def test_ownership_survives_a_deleted_and_recreated_owner_only_as_identity(
        self,
    ) -> None:
        """a recreated owner node does not, BY ITSELF, transfer ownership.

        UNREACHABLE RATHER THAN PREVENTED, and the distinction is the
        point. The owner is stored as a NAME, so a node deleted and
        recreated under the same name is the same owner value, and the
        children still name it. What stops a transfer today is that
        being the owner requires the caller to independently BE that
        namespace -- and the only namespace any principal currently IS
        is its own ``agents.<uuid>``, which no operator recreates as an
        interior provider node.

        So the guarantee is "no principal can currently occupy a
        recreated node", not "the children were detached". PREVENTING
        it needs a principal whose identity is a namespace row with its
        own id, which is the chunk that gives a tool pod its own
        namespace -- not this one. Recorded so whoever reaches that
        chunk finds the pointer.
        """
        recreated_node = "tools.pentest"
        namespace = Namespace(
            id=uuid4(),
            customer_id=None,
            namespace_type="tool",
            owner_agent_id=None,
            owner_namespace=recreated_node,
        )
        # every agent-shaped caller is refused, because an agent's own
        # namespace name can never equal an interior provider node
        for _ in range(5):
            assert _agent_owns_namespace(namespace, uuid4()) is False


class TestGroupInGroupMembership:
    """a group may be a member of a group, resolved at read, depth-capped.

    the EVD shape: user -> org group -> state group, where the grant sits
    on the state group. one membership row per person, one row per licensed
    state. resolution WALKS at read (never expands at write), so changing
    the org group's parents invalidates exactly one cache key.
    """

    async def test_a_user_reaches_a_grant_through_a_parent_group(self) -> None:
        """user in org group, org group member of state group, grant on state."""
        customer = uuid4()
        user = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=None)

        reader = _role(name="Reader", permissions={"workspace": ["read"]})
        org = _group(name="durp-com", customer_id=customer)
        state = _group(name="states-PA", customer_id=customer)
        store = FakeStore()
        store.add_role(reader)
        store.add_group(org)
        store.add_group(state)
        store.add_membership(
            GroupMembership(group_id=org.id, member_type=MemberType.USER, member_id=user, customer_id=customer)
        )
        store.add_membership(
            GroupMembership(group_id=state.id, member_type=MemberType.GROUP, member_id=org.id, customer_id=customer)
        )
        store.add_assignment(
            _assignment(role=reader, group=state, scope_type=ScopeType.NAMESPACE, scope_namespace_id=namespace.id)
        )

        ctx = EvaluationContext(namespace=namespace, action="read", user_id=user)
        result = await evaluate_with_trail(ctx, cache=make_cache(store))
        assert result.decision is True

    async def test_the_depth_cap_stops_the_walk(self) -> None:
        """user -> A -> B -> C with the grant on C only: beyond depth 2, denied."""
        customer = uuid4()
        user = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=None)

        reader = _role(name="Reader", permissions={"workspace": ["read"]})
        group_a = _group(name="A", customer_id=customer)
        group_b = _group(name="B", customer_id=customer)
        group_c = _group(name="C", customer_id=customer)
        store = FakeStore()
        store.add_role(reader)
        for g in (group_a, group_b, group_c):
            store.add_group(g)
        store.add_membership(
            GroupMembership(group_id=group_a.id, member_type=MemberType.USER, member_id=user, customer_id=customer)
        )
        store.add_membership(
            GroupMembership(
                group_id=group_b.id, member_type=MemberType.GROUP, member_id=group_a.id, customer_id=customer
            )
        )
        store.add_membership(
            GroupMembership(
                group_id=group_c.id, member_type=MemberType.GROUP, member_id=group_b.id, customer_id=customer
            )
        )
        store.add_assignment(
            _assignment(role=reader, group=group_c, scope_type=ScopeType.NAMESPACE, scope_namespace_id=namespace.id)
        )

        ctx = EvaluationContext(namespace=namespace, action="read", user_id=user)
        result = await evaluate_with_trail(ctx, cache=make_cache(store))
        assert result.decision is False

    async def test_a_parent_group_of_another_customer_does_not_leak(self) -> None:
        """the customer wall applies to the walked edge exactly as to a direct one."""
        customer = uuid4()
        other_customer = uuid4()
        user = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=None)

        reader = _role(name="Reader", permissions={"workspace": ["read"]})
        org = _group(name="org", customer_id=customer)
        foreign_state = _group(name="foreign-state", customer_id=other_customer)
        store = FakeStore()
        store.add_role(reader)
        store.add_group(org)
        store.add_group(foreign_state)
        store.add_membership(
            GroupMembership(group_id=org.id, member_type=MemberType.USER, member_id=user, customer_id=customer)
        )
        # the group-edge row carries the OTHER customer, so the wall drops it.
        store.add_membership(
            GroupMembership(
                group_id=foreign_state.id, member_type=MemberType.GROUP, member_id=org.id, customer_id=other_customer
            )
        )
        store.add_assignment(
            _assignment(
                role=reader, group=foreign_state, scope_type=ScopeType.NAMESPACE, scope_namespace_id=namespace.id
            )
        )

        ctx = EvaluationContext(namespace=namespace, action="read", user_id=user)
        result = await evaluate_with_trail(ctx, cache=make_cache(store))
        assert result.decision is False

    async def test_the_walk_reads_through_the_group_keyed_cache_layer(self) -> None:
        """a parent lookup is cached under ("group", group_id), not under the user.

        the issue's whole point: changing durp-com's parents must invalidate
        exactly one key, and every user's own membership entry stays valid
        because it never contained the group's contents.
        """
        customer = uuid4()
        user = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=None)

        reader = _role(name="Reader", permissions={"workspace": ["read"]})
        org = _group(name="org", customer_id=customer)
        state = _group(name="state", customer_id=customer)
        store = FakeStore()
        store.add_role(reader)
        store.add_group(org)
        store.add_group(state)
        store.add_membership(
            GroupMembership(group_id=org.id, member_type=MemberType.USER, member_id=user, customer_id=customer)
        )
        store.add_membership(
            GroupMembership(group_id=state.id, member_type=MemberType.GROUP, member_id=org.id, customer_id=customer)
        )
        store.add_assignment(
            _assignment(role=reader, group=state, scope_type=ScopeType.NAMESPACE, scope_namespace_id=namespace.id)
        )
        cache = make_cache(store)
        ctx = EvaluationContext(namespace=namespace, action="read", user_id=user)
        assert (await evaluate_with_trail(ctx, cache=cache)).decision is True

        # the org group's parent set is cached under its own key.
        group_entry = cache.get_membership(ActorMembershipKey(actor_kind="group", actor_id=org.id))
        assert group_entry is not None
        # the user's entry holds only the DIRECT membership -- no expansion.
        user_entry = cache.get_membership(ActorMembershipKey(actor_kind="user", actor_id=user))
        assert user_entry is not None
        assert all(m.member_type is MemberType.USER for m in user_entry.memberships)

        # dropping ONE key -- the group's -- forces a re-walk that sees a changed
        # parent set, while the user's entry was never touched.
        cache.invalidate_membership(ActorMembershipKey(actor_kind="group", actor_id=org.id))
        store.memberships = [
            m for m in store.memberships if not (m.member_type is MemberType.GROUP and m.member_id == org.id)
        ]
        assert (await evaluate_with_trail(ctx, cache=cache)).decision is False
