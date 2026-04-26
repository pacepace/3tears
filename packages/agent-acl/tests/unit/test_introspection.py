"""trail-mode (:func:`evaluate_with_trail`) introspection unit tests.

these tests treat the evaluator as the answer surface for the future
introspection api endpoints. they assert on the structure of
:class:`EvaluationResult` — number of trails, group / role names on
each trail, contributed action sets — so the introspection api can
package the same data into json without losing fidelity.

scenarios:

- multiple paths to the same allow surface as multiple trails (alice
  is in two groups that both grant read).
- per-side trails are independent (user trail names the user-side
  group, agent trail names the agent-side group; one is never
  mistaken for the other).
- empty contribution still records a trail when group + assignment +
  role triple matched (the role just had no actions for the
  namespace's type).
- owner shortcut hides the agent-side trail (ownership is not a
  grant, no row to surface).
- limiting_side classifications round-trip into the result.
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
    evaluate_with_trail,
)

from tests.unit._fake_loaders import FakeStore, make_cache


def _ns(
    *, customer_id: UUID, owner_agent_id: UUID, namespace_type: str = "workspace",
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
        id=uuid4(), customer_id=customer_id,
        namespace_type=namespace_type, owner_agent_id=owner_agent_id,
    )


def _role(*, name: str, permissions: dict[str, list[str]]) -> Role:
    """build a :class:`Role`.

    :param name: role name
    :ptype name: str
    :param permissions: permissions mapping
    :ptype permissions: dict[str, list[str]]
    :return: role record
    :rtype: Role
    """
    return Role(
        id=uuid4(), name=name, is_built_in=True,
        permissions={k: frozenset(v) for k, v in permissions.items()},
    )


def _group(*, name: str, customer_id: UUID | None) -> Group:
    """build a :class:`Group`.

    :param name: group name
    :ptype name: str
    :param customer_id: customer UUID or None for platform scope
    :ptype customer_id: UUID | None
    :return: group record
    :rtype: Group
    """
    return Group(id=uuid4(), name=name, customer_id=customer_id)


def _assignment(
    *, role: Role, group: Group, scope_type: ScopeType,
    scope_namespace_id: UUID | None = None,
    scope_namespace_type: str | None = None,
    scope_customer_id: UUID | None = None,
) -> RoleAssignment:
    """build a :class:`RoleAssignment`.

    :param role: role granted
    :ptype role: Role
    :param group: receiving group
    :ptype group: Group
    :param scope_type: scope shape
    :ptype scope_type: ScopeType
    :param scope_namespace_id: namespace UUID for namespace scope
    :ptype scope_namespace_id: UUID | None
    :param scope_namespace_type: type for type_customer scope
    :ptype scope_namespace_type: str | None
    :param scope_customer_id: customer for type_customer scope
    :ptype scope_customer_id: UUID | None
    :return: assignment record
    :rtype: RoleAssignment
    """
    return RoleAssignment(
        id=uuid4(), role_id=role.id, group_id=group.id,
        scope_type=scope_type,
        scope_namespace_id=scope_namespace_id,
        scope_namespace_type=scope_namespace_type,
        scope_customer_id=scope_customer_id,
    )


# ---------------------------------------------------------------------------
# multiple-path trails
# ---------------------------------------------------------------------------


class TestMultiplePathsSurfaceMultipleTrails:
    """alice in two groups, both granting read -> two trails on user side."""

    @pytest.mark.asyncio
    async def test_two_groups_grant_same_action_two_trails(self) -> None:
        """alice in Engineering AND Admins, both with Reader -> 2 trails."""
        customer = uuid4()
        owner = uuid4()
        alice = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=owner)
        reader = _role(name="Reader", permissions={"workspace": ["read"]})
        engineering = _group(name="Engineering", customer_id=customer)
        admins = _group(name="Admins", customer_id=customer)
        store = FakeStore()
        store.add_role(reader)
        store.add_group(engineering)
        store.add_group(admins)
        store.add_membership(GroupMembership(
            group_id=engineering.id, member_type=MemberType.USER,
            member_id=alice, customer_id=customer,
        ))
        store.add_membership(GroupMembership(
            group_id=admins.id, member_type=MemberType.USER,
            member_id=alice, customer_id=customer,
        ))
        store.add_assignment(_assignment(
            role=reader, group=engineering,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=namespace.id,
        ))
        store.add_assignment(_assignment(
            role=reader, group=admins,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=namespace.id,
        ))
        ctx = EvaluationContext(
            namespace=namespace, action="read", user_id=alice,
        )
        result = await evaluate_with_trail(
            ctx, cache=make_cache(store),
        )
        assert result.decision is True
        # both trails surface; operator sees every grant path
        assert len(result.trails) == 2
        group_names = {t.group.name for t in result.trails}
        assert group_names == {"Engineering", "Admins"}
        for trail in result.trails:
            assert trail.role.name == "Reader"
            assert trail.contributed_actions == frozenset({"read"})


# ---------------------------------------------------------------------------
# per-side trail independence
# ---------------------------------------------------------------------------


class TestPerSideTrailIndependence:
    """user_trails name user-side groups, agent_trails name agent-side groups."""

    @pytest.mark.asyncio
    async def test_user_and_agent_trails_distinct(self) -> None:
        """alice in user-side group, bot in agent-side group; trails name each separately."""
        customer = uuid4()
        owner = uuid4()
        alice = uuid4()
        bot = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=owner)
        reader = _role(name="Reader", permissions={"workspace": ["read"]})
        users = _group(name="Users", customer_id=customer)
        bots = _group(name="Bots", customer_id=customer)
        store = FakeStore()
        store.add_role(reader)
        store.add_group(users)
        store.add_group(bots)
        store.add_membership(GroupMembership(
            group_id=users.id, member_type=MemberType.USER,
            member_id=alice, customer_id=customer,
        ))
        store.add_membership(GroupMembership(
            group_id=bots.id, member_type=MemberType.AGENT,
            member_id=bot, customer_id=customer,
        ))
        store.add_assignment(_assignment(
            role=reader, group=users,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=namespace.id,
        ))
        store.add_assignment(_assignment(
            role=reader, group=bots,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=namespace.id,
        ))
        ctx = EvaluationContext(
            namespace=namespace, action="read",
            user_id=alice, agent_id=bot,
        )
        result = await evaluate_with_trail(
            ctx, cache=make_cache(store),
        )
        assert result.decision is True
        assert len(result.user_trails) == 1
        assert result.user_trails[0].group.name == "Users"
        assert len(result.agent_trails) == 1
        assert result.agent_trails[0].group.name == "Bots"
        # single-side trail field stays empty
        assert result.trails == ()


# ---------------------------------------------------------------------------
# empty contribution still records a trail
# ---------------------------------------------------------------------------


class TestEmptyContributionRecordsTrail:
    """role grants nothing for namespace.type but the chain still surfaces."""

    @pytest.mark.asyncio
    async def test_role_with_unrelated_type_records_trail(self) -> None:
        """role grants on type "memory" but namespace is "workspace" -> trail with empty contribution.

        the trail is still recorded so the operator sees that this
        group/assignment/role chain matched the namespace, even though
        no actions were contributed. drives the "why does my role
        appear to do nothing here?" diagnostic.
        """
        customer = uuid4()
        owner = uuid4()
        alice = uuid4()
        namespace = _ns(
            customer_id=customer, owner_agent_id=owner,
            namespace_type="workspace",
        )
        # role grants only on "memory" type
        memory_only = _role(
            name="MemoryReader", permissions={"memory": ["read"]},
        )
        engineering = _group(name="Engineering", customer_id=customer)
        store = FakeStore()
        store.add_role(memory_only)
        store.add_group(engineering)
        store.add_membership(GroupMembership(
            group_id=engineering.id, member_type=MemberType.USER,
            member_id=alice, customer_id=customer,
        ))
        store.add_assignment(_assignment(
            role=memory_only, group=engineering,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=namespace.id,
        ))
        ctx = EvaluationContext(
            namespace=namespace, action="read", user_id=alice,
        )
        result = await evaluate_with_trail(
            ctx, cache=make_cache(store),
        )
        # decision denies because no actions for workspace type
        assert result.decision is False
        # but the trail is still there showing the chain matched
        assert len(result.trails) == 1
        trail = result.trails[0]
        assert trail.role.name == "MemoryReader"
        assert trail.contributed_actions == frozenset()


# ---------------------------------------------------------------------------
# owner shortcut hides agent-side trails
# ---------------------------------------------------------------------------


class TestOwnerShortcutNoAgentTrail:
    """ownership short-circuit produces no agent-side trail rows."""

    @pytest.mark.asyncio
    async def test_owner_intersection_agent_trails_empty(self) -> None:
        """owner agent + user with grant -> agent_trails is empty."""
        customer = uuid4()
        agent = uuid4()
        user = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=agent)
        reader = _role(name="Reader", permissions={"workspace": ["read"]})
        users = _group(name="Users", customer_id=customer)
        store = FakeStore()
        store.add_role(reader)
        store.add_group(users)
        store.add_membership(GroupMembership(
            group_id=users.id, member_type=MemberType.USER,
            member_id=user, customer_id=customer,
        ))
        store.add_assignment(_assignment(
            role=reader, group=users,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=namespace.id,
        ))
        ctx = EvaluationContext(
            namespace=namespace, action="read",
            user_id=user, agent_id=agent,
        )
        result = await evaluate_with_trail(
            ctx, cache=make_cache(store),
        )
        assert result.decision is True
        assert result.agent_owner_short_circuited is True
        # ownership is not a grant; agent side has no trail row
        assert result.agent_trails == ()
        assert len(result.user_trails) == 1


# ---------------------------------------------------------------------------
# limiting_side round-trip
# ---------------------------------------------------------------------------


class TestLimitingSideRoundTrip:
    """LimitingSide values appear as expected on the result."""

    @pytest.mark.asyncio
    async def test_limiting_side_user_when_owner(self) -> None:
        """owner agent shortcut -> limiting_side = USER."""
        customer = uuid4()
        agent = uuid4()
        user = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=agent)
        reader = _role(name="Reader", permissions={"workspace": ["read"]})
        users = _group(name="Users", customer_id=customer)
        store = FakeStore()
        store.add_role(reader)
        store.add_group(users)
        store.add_membership(GroupMembership(
            group_id=users.id, member_type=MemberType.USER,
            member_id=user, customer_id=customer,
        ))
        store.add_assignment(_assignment(
            role=reader, group=users,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=namespace.id,
        ))
        ctx = EvaluationContext(
            namespace=namespace, action="read",
            user_id=user, agent_id=agent,
        )
        result = await evaluate_with_trail(
            ctx, cache=make_cache(store),
        )
        assert result.limiting_side == LimitingSide.USER

    @pytest.mark.asyncio
    async def test_limiting_side_neither_on_empty_side(self) -> None:
        """one side empty -> limiting_side = NEITHER (deny)."""
        customer = uuid4()
        owner = uuid4()
        user = uuid4()
        agent = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=owner)
        reader = _role(name="Reader", permissions={"workspace": ["read"]})
        users = _group(name="Users", customer_id=customer)
        store = FakeStore()
        store.add_role(reader)
        store.add_group(users)
        # only user side has membership
        store.add_membership(GroupMembership(
            group_id=users.id, member_type=MemberType.USER,
            member_id=user, customer_id=customer,
        ))
        store.add_assignment(_assignment(
            role=reader, group=users,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=namespace.id,
        ))
        ctx = EvaluationContext(
            namespace=namespace, action="read",
            user_id=user, agent_id=agent,
        )
        result = await evaluate_with_trail(
            ctx, cache=make_cache(store),
        )
        assert result.limiting_side == LimitingSide.NEITHER
        assert result.decision is False


# ---------------------------------------------------------------------------
# trail ordering
# ---------------------------------------------------------------------------


class TestTrailOrderingDeterministic:
    """trails come out in stable (group_id, assignment_id) order."""

    @pytest.mark.asyncio
    async def test_repeated_runs_produce_same_trail_order(self) -> None:
        """two evaluations on the same fixture yield identical trail lists."""
        customer = uuid4()
        owner = uuid4()
        alice = uuid4()
        namespace = _ns(customer_id=customer, owner_agent_id=owner)
        reader = _role(name="Reader", permissions={"workspace": ["read"]})
        # several groups, each with a separate assignment
        groups = [_group(name=f"G{i}", customer_id=customer) for i in range(5)]
        store = FakeStore()
        store.add_role(reader)
        for g in groups:
            store.add_group(g)
            store.add_membership(GroupMembership(
                group_id=g.id, member_type=MemberType.USER,
                member_id=alice, customer_id=customer,
            ))
            store.add_assignment(_assignment(
                role=reader, group=g,
                scope_type=ScopeType.NAMESPACE,
                scope_namespace_id=namespace.id,
            ))
        ctx = EvaluationContext(
            namespace=namespace, action="read", user_id=alice,
        )
        result1 = await evaluate_with_trail(
            ctx, cache=make_cache(store),
        )
        result2 = await evaluate_with_trail(
            ctx, cache=make_cache(store),
        )
        # same trail order both times
        ids1 = [t.assignment.id for t in result1.trails]
        ids2 = [t.assignment.id for t in result2.trails]
        assert ids1 == ids2
        assert len(ids1) == 5
