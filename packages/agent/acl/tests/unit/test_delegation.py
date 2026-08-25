"""the ceiling on what a delegated admin may hand out.

two things are under test, and they are separable on purpose:

- the ROLE-OWNERSHIP wall in the evaluator: a customer-authored role
  contributes nothing against another customer's namespace, and it does
  so INDEPENDENTLY of the group wall — the platform-scoped-group case
  is the one the group wall cannot catch, so it gets its own test.
- the DELEGATION CEILING in :mod:`threetears.agent.acl.delegation`:
  what the caller holds is the intersection across every namespace a
  grant could reach, an empty namespace set yields an empty ceiling
  rather than a universal one, and the wildcard bucket is refused.

the built-in-roles-unchanged test is the regression guard the shard
asked for: a platform-owned role (``customer_id is None``) must
evaluate exactly as it did before ownership existed.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from threetears.agent.acl import (
    EvaluationContext,
    Group,
    GroupMembership,
    HeldPermissions,
    MemberType,
    Namespace,
    PermissionEscalation,
    Role,
    RoleAssignment,
    ScopeType,
    WILDCARD_RESOURCE_TYPE,
    enforce_within_held_permissions,
    escalating_permissions,
    evaluate_with_trail,
    held_actions_on,
    resolve_held_permissions,
)

from ._fake_loaders import FakeStore, make_cache


# ---------------------------------------------------------------------------
# fixture builders
# ---------------------------------------------------------------------------


def _ns(
    *,
    customer_id: UUID,
    namespace_type: str = "workspace",
) -> Namespace:
    """build a :class:`Namespace` with a fresh id and no owning agent.

    :param customer_id: owning customer UUID
    :ptype customer_id: UUID
    :param namespace_type: type discriminator
    :ptype namespace_type: str
    :return: namespace value
    :rtype: Namespace
    """
    return Namespace(
        id=uuid4(),
        customer_id=customer_id,
        namespace_type=namespace_type,
        owner_agent_id=None,
    )


def _grant(
    store: FakeStore,
    *,
    user_id: UUID,
    namespace: Namespace,
    actions: frozenset[str],
    group_customer_id: UUID | None,
    role_customer_id: UUID | None,
) -> tuple[Group, Role, RoleAssignment]:
    """wire one complete grant path into ``store`` for ``user_id``.

    one group, one role, one namespace-scoped assignment binding them.
    both customer dimensions are parameters because the two walls under
    test are exactly those two fields.

    :param store: fake loader store to populate
    :ptype store: FakeStore
    :param user_id: user who becomes a member of the group
    :ptype user_id: UUID
    :param namespace: namespace the assignment is scoped to
    :ptype namespace: Namespace
    :param actions: actions the role grants on the namespace's type
    :ptype actions: frozenset[str]
    :param group_customer_id: owning customer of the group, or ``None``
        for a platform-scoped group
    :ptype group_customer_id: UUID | None
    :param role_customer_id: owning customer of the role, or ``None``
        for a platform-owned role
    :ptype role_customer_id: UUID | None
    :return: the ``(group, role, assignment)`` triple written
    :rtype: tuple[Group, Role, RoleAssignment]
    """
    group = store.add_group(
        Group(id=uuid4(), name=f"g-{uuid4().hex[:8]}", customer_id=group_customer_id),
    )
    role = store.add_role(
        Role(
            id=uuid4(),
            name=f"r-{uuid4().hex[:8]}",
            permissions={namespace.namespace_type: actions},
            is_built_in=False,
            customer_id=role_customer_id,
        ),
    )
    assignment = store.add_assignment(
        RoleAssignment(
            id=uuid4(),
            role_id=role.id,
            group_id=group.id,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=namespace.id,
            scope_namespace_type=None,
            scope_customer_id=None,
        ),
    )
    store.add_membership(
        GroupMembership(
            group_id=group.id,
            member_type=MemberType.USER,
            member_id=user_id,
            customer_id=group_customer_id,
        ),
    )
    return group, role, assignment


# parity-exempt: not a fake OF NamespaceCollection -- a stub of the one structural method
class _FakeNamespaceCollection:
    """namespace collection stub exposing only ``find_by_type_and_customer``.

    :func:`resolve_held_permissions` types its collection argument
    ``Any`` and calls exactly one structural method on it, for the same
    reason :func:`~threetears.agent.acl.authorize.authorize` types its
    own that way: the concrete class lives in the consuming app, above
    this package. So the test supplies that one method rather than a
    whole Collection.

    That is also why this class is parity-exempt rather than
    parity-declared. Declaring parity against the full
    :class:`~threetears.agent.acl.collections.NamespaceCollection` would
    demand thirty unused methods and assert a contract this code does
    not depend on -- a fake harder to read without being any more
    faithful to what the function actually calls.

    :ivar by_type: ``{(namespace_type, customer_id): [namespace, ...]}``
    """

    def __init__(self, by_type: dict[tuple[str, UUID], list[Namespace]]) -> None:
        """store the canned answer set.

        :param by_type: mapping the stub answers from
        :ptype by_type: dict[tuple[str, UUID], list[Namespace]]
        :return: nothing
        :rtype: None
        """
        self.by_type = by_type

    async def find_by_type_and_customer(
        self,
        *,
        namespace_type: str,
        customer_id: UUID,
    ) -> list[Namespace]:
        """return the canned namespaces for the pair.

        a :class:`Namespace` already exposes the four attributes
        :func:`resolve_held_permissions` reads off an entity, so the
        value type doubles as the entity here.

        :param namespace_type: type discriminator
        :ptype namespace_type: str
        :param customer_id: owning customer UUID
        :ptype customer_id: UUID
        :return: matching namespaces, empty when none were canned
        :rtype: list[Namespace]
        """
        return list(self.by_type.get((namespace_type, customer_id), []))


# ---------------------------------------------------------------------------
# evaluator: the role-ownership wall
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_customer_authored_role_denies_against_another_customer() -> None:
    """a customer-authored role must not reach a second customer's namespace."""
    store = FakeStore()
    customer_a = uuid4()
    customer_b = uuid4()
    user_id = uuid4()
    # the namespace belongs to customer B; the role is authored by A.
    # the GROUP is platform-scoped, so the group wall (evaluator.py's
    # ``group.customer_id is not None`` check) passes -- only the role
    # wall can refuse this.
    namespace = _ns(customer_id=customer_b)
    _grant(
        store,
        user_id=user_id,
        namespace=namespace,
        actions=frozenset({"write"}),
        group_customer_id=None,
        role_customer_id=customer_a,
    )

    result = await evaluate_with_trail(
        EvaluationContext(namespace=namespace, action="write", user_id=user_id),
        cache=make_cache(store),
    )

    assert result.decision is False
    assert result.effective_actions == frozenset()
    assert result.trails == ()


@pytest.mark.asyncio
async def test_customer_authored_role_allows_inside_its_own_customer() -> None:
    """the same role resolves normally against its owning customer."""
    store = FakeStore()
    customer_a = uuid4()
    user_id = uuid4()
    namespace = _ns(customer_id=customer_a)
    _, role, _ = _grant(
        store,
        user_id=user_id,
        namespace=namespace,
        actions=frozenset({"write"}),
        group_customer_id=customer_a,
        role_customer_id=customer_a,
    )

    result = await evaluate_with_trail(
        EvaluationContext(namespace=namespace, action="write", user_id=user_id),
        cache=make_cache(store),
    )

    assert result.decision is True
    assert result.effective_actions == frozenset({"write"})
    assert [t.role.id for t in result.trails] == [role.id]


@pytest.mark.asyncio
async def test_two_customers_author_identically_named_roles_independently() -> None:
    """same role name in two tenants; neither reaches the other."""
    store = FakeStore()
    customer_a = uuid4()
    customer_b = uuid4()
    admin_a = uuid4()
    admin_b = uuid4()
    ns_a = _ns(customer_id=customer_a)
    ns_b = _ns(customer_id=customer_b)

    for user_id, customer_id, namespace, actions in (
        (admin_a, customer_a, ns_a, frozenset({"read", "write"})),
        (admin_b, customer_b, ns_b, frozenset({"read"})),
    ):
        group = store.add_group(Group(id=uuid4(), name="field-managers", customer_id=customer_id))
        role = store.add_role(
            Role(
                id=uuid4(),
                # the point: one name, two rows, two owners.
                name="Field Manager",
                permissions={"workspace": actions},
                is_built_in=False,
                customer_id=customer_id,
            ),
        )
        store.add_assignment(
            RoleAssignment(
                id=uuid4(),
                role_id=role.id,
                group_id=group.id,
                scope_type=ScopeType.TYPE_CUSTOMER,
                scope_namespace_id=None,
                scope_namespace_type="workspace",
                scope_customer_id=customer_id,
            ),
        )
        store.add_membership(
            GroupMembership(
                group_id=group.id,
                member_type=MemberType.USER,
                member_id=user_id,
                customer_id=customer_id,
            ),
        )

    cache = make_cache(store)
    a_on_a = await evaluate_with_trail(
        EvaluationContext(namespace=ns_a, action="write", user_id=admin_a),
        cache=cache,
    )
    b_on_b = await evaluate_with_trail(
        EvaluationContext(namespace=ns_b, action="read", user_id=admin_b),
        cache=cache,
    )
    b_on_a = await evaluate_with_trail(
        EvaluationContext(namespace=ns_a, action="read", user_id=admin_b),
        cache=cache,
    )

    assert a_on_a.decision is True
    assert b_on_b.decision is True
    assert b_on_a.decision is False


@pytest.mark.asyncio
async def test_platform_owned_role_evaluation_is_unchanged() -> None:
    """a role with ``customer_id is None`` behaves exactly as before.

    the ownership wall must be inert for every built-in and every
    operator-authored platform role, including the platform-scoped-group
    path that deliberately spans customers (shared agents rely on it).
    """
    store = FakeStore()
    customer_b = uuid4()
    user_id = uuid4()
    namespace = _ns(customer_id=customer_b)
    _, role, _ = _grant(
        store,
        user_id=user_id,
        namespace=namespace,
        actions=frozenset({"read"}),
        group_customer_id=None,
        role_customer_id=None,
    )

    result = await evaluate_with_trail(
        EvaluationContext(namespace=namespace, action="read", user_id=user_id),
        cache=make_cache(store),
    )

    assert result.decision is True
    assert [t.role.id for t in result.trails] == [role.id]
    assert role.customer_id is None


def test_role_defaults_to_platform_ownership() -> None:
    """every pre-existing ``Role(...)`` call site stays platform-owned.

    the field is last and defaulted, which is what makes the upgrade a
    no-op for callers that never heard of ownership.
    """
    role = Role(id=uuid4(), name="Reader", permissions={}, is_built_in=True)
    assert role.customer_id is None


# ---------------------------------------------------------------------------
# delegation ceiling: held_actions_on
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_held_actions_intersects_rather_than_unions() -> None:
    """holding an action on ONE namespace does not license it on all."""
    store = FakeStore()
    customer_id = uuid4()
    admin = uuid4()
    ns_one = _ns(customer_id=customer_id)
    ns_two = _ns(customer_id=customer_id)
    _grant(
        store,
        user_id=admin,
        namespace=ns_one,
        actions=frozenset({"read", "write"}),
        group_customer_id=customer_id,
        role_customer_id=None,
    )
    _grant(
        store,
        user_id=admin,
        namespace=ns_two,
        actions=frozenset({"read"}),
        group_customer_id=customer_id,
        role_customer_id=None,
    )

    held, trails = await held_actions_on(
        namespaces=[ns_one, ns_two],
        user_id=admin,
        agent_id=None,
        cache=make_cache(store),
    )

    assert held == frozenset({"read"})
    assert trails != ()


@pytest.mark.asyncio
async def test_held_actions_on_no_namespaces_is_empty_not_universal() -> None:
    """the intersection identity is universal; here it must be empty."""
    store = FakeStore()
    held, trails = await held_actions_on(
        namespaces=[],
        user_id=uuid4(),
        agent_id=None,
        cache=make_cache(store),
    )
    assert held == frozenset()
    assert trails == ()


@pytest.mark.asyncio
async def test_held_actions_stops_once_the_intersection_empties() -> None:
    """an admin holding nothing on one namespace holds nothing overall."""
    store = FakeStore()
    customer_id = uuid4()
    admin = uuid4()
    granted = _ns(customer_id=customer_id)
    ungranted = _ns(customer_id=customer_id)
    _grant(
        store,
        user_id=admin,
        namespace=granted,
        actions=frozenset({"read", "write"}),
        group_customer_id=customer_id,
        role_customer_id=None,
    )

    held, _ = await held_actions_on(
        namespaces=[granted, ungranted],
        user_id=admin,
        agent_id=None,
        cache=make_cache(store),
    )

    assert held == frozenset()


# ---------------------------------------------------------------------------
# delegation ceiling: resolve_held_permissions + enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_held_permissions_bounds_by_customer_namespaces() -> None:
    """the ceiling is per resource type, across that type's namespaces."""
    store = FakeStore()
    customer_id = uuid4()
    admin = uuid4()
    ws_one = _ns(customer_id=customer_id, namespace_type="workspace")
    ws_two = _ns(customer_id=customer_id, namespace_type="workspace")
    for namespace, actions in ((ws_one, frozenset({"read", "write"})), (ws_two, frozenset({"read"}))):
        _grant(
            store,
            user_id=admin,
            namespace=namespace,
            actions=actions,
            group_customer_id=customer_id,
            role_customer_id=None,
        )
    collection = _FakeNamespaceCollection({("workspace", customer_id): [ws_one, ws_two]})

    held = await resolve_held_permissions(
        namespace_collection=collection,
        customer_id=customer_id,
        resource_types=["workspace"],
        user_id=admin,
        agent_id=None,
        cache=make_cache(store),
    )

    assert held.actions_by_resource_type == {"workspace": frozenset({"read"})}
    assert held.namespace_count == 2
    # authoring "read" is within the ceiling; "write" is not, even
    # though the admin holds it on ws_one.
    assert escalating_permissions({"workspace": ["read"]}, held) == ()
    violations = escalating_permissions({"workspace": ["read", "write"]}, held)
    assert [v.action for v in violations] == ["write"]


@pytest.mark.asyncio
async def test_resource_type_the_tenant_has_no_namespaces_of_is_refused() -> None:
    """authoring against an unused resource type grants nothing, loudly."""
    store = FakeStore()
    customer_id = uuid4()
    admin = uuid4()
    collection = _FakeNamespaceCollection({})

    held = await resolve_held_permissions(
        namespace_collection=collection,
        customer_id=customer_id,
        resource_types=["datasource"],
        user_id=admin,
        agent_id=None,
        cache=make_cache(store),
    )

    assert held.actions_by_resource_type == {"datasource": frozenset()}
    assert held.namespace_count == 0
    with pytest.raises(PermissionEscalation) as excinfo:
        enforce_within_held_permissions({"datasource": ["datasource.read"]}, held)
    assert excinfo.value.violations[0].action == "datasource.read"


@pytest.mark.asyncio
async def test_wildcard_bucket_is_never_resolved_for_a_tenant() -> None:
    """``"*"`` is skipped on resolve, which is what makes it refused."""
    store = FakeStore()
    customer_id = uuid4()
    collection = _FakeNamespaceCollection({})

    held = await resolve_held_permissions(
        namespace_collection=collection,
        customer_id=customer_id,
        resource_types=[WILDCARD_RESOURCE_TYPE, "workspace"],
        user_id=uuid4(),
        agent_id=None,
        cache=make_cache(store),
    )

    assert WILDCARD_RESOURCE_TYPE not in held.actions_by_resource_type
    violations = escalating_permissions({WILDCARD_RESOURCE_TYPE: ["read"]}, held)
    assert len(violations) == 1
    assert violations[0].action is None
    assert violations[0].resource_type == WILDCARD_RESOURCE_TYPE


def test_wildcard_bucket_is_allowed_when_the_caller_holds_one() -> None:
    """a platform caller supplying a wildcard ceiling is not refused."""
    held = HeldPermissions(
        actions_by_resource_type={WILDCARD_RESOURCE_TYPE: frozenset({"read"})},
        trails=(),
        namespace_count=0,
    )
    assert escalating_permissions({WILDCARD_RESOURCE_TYPE: ["read"]}, held) == ()
    assert escalating_permissions({WILDCARD_RESOURCE_TYPE: ["write"]}, held)[0].action == "write"


def test_violations_are_deterministically_ordered() -> None:
    """two runs over the same refusal read the same way."""
    held = HeldPermissions(
        actions_by_resource_type={"workspace": frozenset(), "datasource": frozenset()},
        trails=(),
        namespace_count=1,
    )
    violations = escalating_permissions(
        {"workspace": ["write", "read"], "datasource": ["datasource.write", "datasource.read"]},
        held,
    )
    assert [(v.resource_type, v.action) for v in violations] == [
        ("datasource", "datasource.read"),
        ("datasource", "datasource.write"),
        ("workspace", "read"),
        ("workspace", "write"),
    ]


@pytest.mark.asyncio
async def test_refusal_carries_the_trail_that_justifies_the_ceiling() -> None:
    """a refusal cites what the caller does hold, not just what they do not."""
    store = FakeStore()
    customer_id = uuid4()
    admin = uuid4()
    workspace = _ns(customer_id=customer_id, namespace_type="workspace")
    _, role, _ = _grant(
        store,
        user_id=admin,
        namespace=workspace,
        actions=frozenset({"read"}),
        group_customer_id=customer_id,
        role_customer_id=None,
    )
    collection = _FakeNamespaceCollection({("workspace", customer_id): [workspace]})

    held = await resolve_held_permissions(
        namespace_collection=collection,
        customer_id=customer_id,
        resource_types=["workspace"],
        user_id=admin,
        agent_id=None,
        cache=make_cache(store),
    )
    with pytest.raises(PermissionEscalation) as excinfo:
        enforce_within_held_permissions({"workspace": ["write"]}, held)

    assert [t.role.id for t in excinfo.value.trails] == [role.id]
    assert "write" in str(excinfo.value)


def test_enforcement_passes_silently_when_within_the_ceiling() -> None:
    """the raising form returns nothing when there is nothing to refuse."""
    held = HeldPermissions(
        actions_by_resource_type={"workspace": frozenset({"read", "write"})},
        trails=(),
        namespace_count=3,
    )
    assert enforce_within_held_permissions({"workspace": ["read"]}, held) is None
