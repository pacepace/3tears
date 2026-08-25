"""tenant and zone isolation for a human-in-the-loop display session.

these tests answer one question: does the namespace name
:func:`threetears.core.namespaces.build_hitl_namespace_name` produces
let the SHIPPED evaluator do the isolating, with no check written for
this path anywhere. every refusal below comes out of
:func:`evaluate_with_trail`; nothing here inspects a customer id and
decides.

the fixture derives each namespace row's id from its NAME, the way the
platform's own rows are unique by name and resolved through
``get_by_name``. that is what gives the assertions their teeth: a name
builder that collapsed two zones, or two customers, into one string
would hand both evaluations the SAME row, and the deny cases would go
green in the one direction that matters.

the operator is evaluated user-side only. an attaching human is a user
and no agent is on the call, so the intersection path never runs and
the agent-ownership short-circuit cannot mask anything.
"""

from __future__ import annotations

from uuid import NAMESPACE_DNS, UUID, uuid4, uuid5

from threetears.agent.acl import (
    EvaluationContext,
    Group,
    GroupMembership,
    MemberType,
    Namespace,
    Role,
    RoleAssignment,
    ScopeType,
    WILDCARD_RESOURCE_TYPE,
    evaluate_with_trail,
)
from threetears.core.namespaces import (
    HITL_NAMESPACE_TYPE,
    PLURAL_PREFIX_TOOL,
    build_hitl_namespace_name,
    build_namespace_name,
)

from ._fake_loaders import FakeStore, make_cache

# Two customers sharing their first eight hex characters, so a
# truncated customer spelling would put both on one row rather than
# failing visibly. See the same pair in
# ``packages/core/tests/unit/test_namespaces.py``.
CUSTOMER_X = UUID("7f3c9a1d-1111-4111-8111-000000000001")
CUSTOMER_Z = UUID("7f3c9a1d-2222-4222-8222-000000000002")

# A network zone is a distinct registered tool, so two zones are two
# tool namespace names. The versions differ only in their patch
# component, which is what a truncated version would collapse.
TOOL_NS_ALPHA = build_namespace_name(PLURAL_PREFIX_TOOL, "scrape.zone_alpha", "1.0.0")
TOOL_NS_BETA = build_namespace_name(PLURAL_PREFIX_TOOL, "scrape.zone_beta", "1.0.0")
TOOL_NS_ALPHA_NEXT = build_namespace_name(PLURAL_PREFIX_TOOL, "scrape.zone_alpha", "1.0.1")

# The action is data as far as this package is concerned: `authorize()`
# takes an arbitrary string. Declared here because the tests need one,
# NOT shipped as vocabulary -- the platform that seeds the role owns
# the spelling.
ACTION_ATTACH = "hitl.attach"


def _row(name: str, *, customer_id: UUID | None, namespace_type: str) -> Namespace:
    """build a namespace row whose id is derived from its name.

    ``platform.namespaces`` rows are resolved by name, so two identical
    names are one row. deriving the id the same way makes a name
    collision show up here as a shared row rather than as two rows that
    happen to be distinct because the fixture minted two UUIDs.

    :param name: canonical namespace name
    :ptype name: str
    :param customer_id: owning customer, or ``None`` for a
        platform-scoped row
    :ptype customer_id: UUID | None
    :param namespace_type: type discriminator
    :ptype namespace_type: str
    :return: namespace record
    :rtype: Namespace
    """
    return Namespace(
        id=uuid5(NAMESPACE_DNS, name),
        customer_id=customer_id,
        namespace_type=namespace_type,
        owner_agent_id=None,
    )


def _hitl_row(tool_namespace_name: str, customer_id: UUID) -> Namespace:
    """build the row a session against ``tool_namespace_name`` authorizes on.

    :param tool_namespace_name: serving tool's canonical namespace name
    :ptype tool_namespace_name: str
    :param customer_id: customer whose session it is
    :ptype customer_id: UUID
    :return: namespace record named by the hitl builder
    :rtype: Namespace
    """
    return _row(
        build_hitl_namespace_name(tool_namespace_name, customer_id),
        customer_id=customer_id,
        namespace_type=HITL_NAMESPACE_TYPE,
    )


def _tool_row(tool_namespace_name: str) -> Namespace:
    """build the serving tool's OWN namespace row.

    platform tool namespaces materialize with ``customer_id`` NULL,
    which is the fact the wrong shape founders on.

    :param tool_namespace_name: canonical tool namespace name
    :ptype tool_namespace_name: str
    :return: namespace record with no customer
    :rtype: Namespace
    """
    return _row(tool_namespace_name, customer_id=None, namespace_type="tool")


def _operator_store(
    *,
    group_customer_id: UUID | None,
    member_customer_id: UUID | None,
    permissions: dict[str, list[str]],
) -> tuple[FakeStore, UUID, Group, Role]:
    """build a store holding one operator, one group and one role.

    no assignment is added: each test names the scope its own point
    needs, which is the variable under test.

    :param group_customer_id: customer owning the operator group, or
        ``None`` for a platform-scoped group
    :ptype group_customer_id: UUID | None
    :param member_customer_id: customer the operator belongs to, or
        ``None`` for an operator bound to no customer
    :ptype member_customer_id: UUID | None
    :param permissions: role permission map
    :ptype permissions: dict[str, list[str]]
    :return: ``(store, operator_id, group, role)``
    :rtype: tuple[FakeStore, UUID, Group, Role]
    """
    store = FakeStore()
    operator_id = uuid4()
    group = store.add_group(Group(id=uuid4(), name="scrape-operators", customer_id=group_customer_id))
    role = store.add_role(
        Role(
            id=uuid4(),
            name="HitlAttach",
            permissions={k: frozenset(v) for k, v in permissions.items()},
            is_built_in=True,
        ),
    )
    store.add_membership(
        GroupMembership(
            group_id=group.id,
            member_type=MemberType.USER,
            member_id=operator_id,
            customer_id=member_customer_id,
        ),
    )
    return store, operator_id, group, role


def _grant_on(store: FakeStore, *, role: Role, group: Group, namespace: Namespace) -> RoleAssignment:
    """give ``group`` ``role`` on exactly one namespace.

    :param store: fake store to append to
    :ptype store: FakeStore
    :param role: role being granted
    :ptype role: Role
    :param group: group receiving the grant
    :ptype group: Group
    :param namespace: the one namespace the grant covers
    :ptype namespace: Namespace
    :return: the assignment added
    :rtype: RoleAssignment
    """
    return store.add_assignment(
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


def _attach_ctx(namespace: Namespace, operator_id: UUID) -> EvaluationContext:
    """build the question "may this operator attach to this display?".

    :param namespace: namespace under evaluation
    :ptype namespace: Namespace
    :param operator_id: the attaching human
    :ptype operator_id: UUID
    :return: evaluation context
    :rtype: EvaluationContext
    """
    return EvaluationContext(namespace=namespace, action=ACTION_ATTACH, user_id=operator_id, agent_id=None)


class TestCrossTenantIsolation:
    """one customer's operator group cannot reach another's display."""

    async def test_another_customers_session_is_refused_by_the_evaluator(self) -> None:
        store, operator_id, group, role = _operator_store(
            group_customer_id=CUSTOMER_X,
            member_customer_id=CUSTOMER_X,
            permissions={HITL_NAMESPACE_TYPE: [ACTION_ATTACH]},
        )
        ours = _hitl_row(TOOL_NS_ALPHA, CUSTOMER_X)
        theirs = _hitl_row(TOOL_NS_ALPHA, CUSTOMER_Z)
        assert ours.id != theirs.id
        assignment = _grant_on(store, role=role, group=group, namespace=ours)
        cache = make_cache(store)

        allowed = await evaluate_with_trail(_attach_ctx(ours, operator_id), cache=cache)
        refused = await evaluate_with_trail(_attach_ctx(theirs, operator_id), cache=cache)

        # the allow half is what proves the refusal is about the
        # namespace and not about a fixture that grants nothing: same
        # operator, same group, same role, same action.
        assert allowed.decision is True
        assert [trail.assignment.id for trail in allowed.trails] == [assignment.id]
        assert refused.decision is False
        assert refused.trails == ()
        assert refused.effective_actions == frozenset()

    async def test_a_grant_written_against_another_tenants_row_still_refuses(self) -> None:
        """the row being another customer's is enough on its own.

        the test above refuses through scope coverage: the operator's
        assignment names its own tenant's namespace, and another
        tenant's is a different row. that is the ordinary path and it
        is worth pinning, but it would also refuse for two distinct
        rows of any kind. here the assignment names the OTHER tenant's
        row directly, so coverage passes and the evaluator's
        cross-customer wall is the only thing left standing.
        """
        store, operator_id, group, role = _operator_store(
            group_customer_id=CUSTOMER_X,
            member_customer_id=CUSTOMER_X,
            permissions={HITL_NAMESPACE_TYPE: [ACTION_ATTACH]},
        )
        theirs = _hitl_row(TOOL_NS_ALPHA, CUSTOMER_Z)
        assignment = _grant_on(store, role=role, group=group, namespace=theirs)
        cache = make_cache(store)

        refused = await evaluate_with_trail(_attach_ctx(theirs, operator_id), cache=cache)

        assert assignment.covers(theirs) is True
        assert refused.decision is False
        assert refused.trails == ()


class TestCrossZoneIsolation:
    """the SAME customer's display in another zone is refused too.

    written separately from the cross-tenant case rather than folded
    into it: a customer-only namespace passes that one and fails this
    one, and that difference is the whole reason the name carries the
    tool.
    """

    async def test_the_same_customers_other_zone_is_refused(self) -> None:
        store, operator_id, group, role = _operator_store(
            group_customer_id=CUSTOMER_X,
            member_customer_id=CUSTOMER_X,
            permissions={HITL_NAMESPACE_TYPE: [ACTION_ATTACH]},
        )
        alpha = _hitl_row(TOOL_NS_ALPHA, CUSTOMER_X)
        beta = _hitl_row(TOOL_NS_BETA, CUSTOMER_X)
        # both rows are the same tenant's. only the zone differs, and
        # it differs only because the NAME carries the tool.
        assert alpha.customer_id == beta.customer_id
        assert alpha.id != beta.id
        assignment = _grant_on(store, role=role, group=group, namespace=alpha)
        cache = make_cache(store)

        allowed = await evaluate_with_trail(_attach_ctx(alpha, operator_id), cache=cache)
        refused = await evaluate_with_trail(_attach_ctx(beta, operator_id), cache=cache)

        assert allowed.decision is True
        assert [trail.assignment.id for trail in allowed.trails] == [assignment.id]
        assert refused.decision is False
        assert refused.trails == ()

    async def test_a_type_customer_grant_spans_every_zone_deliberately(self) -> None:
        """a grant written over the TYPE reaches every zone, by design.

        the name isolates zones for a namespace-scoped assignment. an
        administrator who writes a ``type_customer`` assignment over
        the hitl type is asking for "this customer's displays,
        wherever they are", and gets it. recorded as a test because
        the zone isolation would otherwise read as a property of the
        name alone, and the entry point that bypasses it belongs in
        the record beside the one it closes.
        """
        store, operator_id, group, role = _operator_store(
            group_customer_id=CUSTOMER_X,
            member_customer_id=CUSTOMER_X,
            permissions={HITL_NAMESPACE_TYPE: [ACTION_ATTACH]},
        )
        alpha = _hitl_row(TOOL_NS_ALPHA, CUSTOMER_X)
        beta = _hitl_row(TOOL_NS_BETA, CUSTOMER_X)
        store.add_assignment(
            RoleAssignment(
                id=uuid4(),
                role_id=role.id,
                group_id=group.id,
                scope_type=ScopeType.TYPE_CUSTOMER,
                scope_namespace_id=None,
                scope_namespace_type=HITL_NAMESPACE_TYPE,
                scope_customer_id=CUSTOMER_X,
            ),
        )
        cache = make_cache(store)

        assert (await evaluate_with_trail(_attach_ctx(alpha, operator_id), cache=cache)).decision is True
        assert (await evaluate_with_trail(_attach_ctx(beta, operator_id), cache=cache)).decision is True

        # and it still stops at the tenant line.
        other_tenant = _hitl_row(TOOL_NS_ALPHA, CUSTOMER_Z)
        assert (await evaluate_with_trail(_attach_ctx(other_tenant, operator_id), cache=cache)).decision is False


class TestVersionIsPartOfTheEntitlement:
    """a tool version is its own row, so a session against it is too.

    ``tools.<mcp>.<version>`` carries its own role assignments. a
    session namespace that dropped the version would leave a version
    bump able to produce the mismatch where somebody may call the tool
    and not attach to the display it raised, or the reverse.
    """

    async def test_another_version_of_the_same_zone_is_refused(self) -> None:
        store, operator_id, group, role = _operator_store(
            group_customer_id=CUSTOMER_X,
            member_customer_id=CUSTOMER_X,
            permissions={HITL_NAMESPACE_TYPE: [ACTION_ATTACH]},
        )
        granted = _hitl_row(TOOL_NS_ALPHA, CUSTOMER_X)
        bumped = _hitl_row(TOOL_NS_ALPHA_NEXT, CUSTOMER_X)
        assert granted.id != bumped.id
        _grant_on(store, role=role, group=group, namespace=granted)
        cache = make_cache(store)

        assert (await evaluate_with_trail(_attach_ctx(granted, operator_id), cache=cache)).decision is True
        assert (await evaluate_with_trail(_attach_ctx(bumped, operator_id), cache=cache)).decision is False


class TestToolNamespaceCannotCarryTheSession:
    """why the tool's own namespace is not an option for this.

    the roles here use the WILDCARD resource bucket so the role
    contributes the same action on a ``tool`` row and on a ``hitl``
    row. without that, a refusal on the tool row would prove only that
    the role had nothing to say about that type -- a second producer
    of the same denial, and the assertion would stop meaning anything.
    """

    async def test_a_tenant_scoped_operator_cannot_be_granted_on_a_tool_namespace(self) -> None:
        store, operator_id, group, role = _operator_store(
            group_customer_id=CUSTOMER_X,
            member_customer_id=CUSTOMER_X,
            permissions={WILDCARD_RESOURCE_TYPE: [ACTION_ATTACH]},
        )
        tool_row = _tool_row(TOOL_NS_ALPHA)
        hitl_row = _hitl_row(TOOL_NS_ALPHA, CUSTOMER_X)
        assert tool_row.customer_id is None
        _grant_on(store, role=role, group=group, namespace=tool_row)
        _grant_on(store, role=role, group=group, namespace=hitl_row)
        cache = make_cache(store)

        on_tool = await evaluate_with_trail(_attach_ctx(tool_row, operator_id), cache=cache)
        on_hitl = await evaluate_with_trail(_attach_ctx(hitl_row, operator_id), cache=cache)

        # identical operator, group, role, action and scope shape. the
        # row's customer is the only operative difference, and the
        # evaluator counts a customer-scoped group or membership only
        # against a namespace of that same customer.
        assert on_hitl.decision is True
        assert on_tool.decision is False
        assert on_tool.trails == ()

    async def test_the_only_grant_that_reaches_a_tool_namespace_names_no_customer(self) -> None:
        store, operator_id, group, role = _operator_store(
            group_customer_id=None,
            member_customer_id=None,
            permissions={WILDCARD_RESOURCE_TYPE: [ACTION_ATTACH]},
        )
        tool_row = _tool_row(TOOL_NS_ALPHA)
        _grant_on(store, role=role, group=group, namespace=tool_row)
        cache = make_cache(store)

        allowed = await evaluate_with_trail(_attach_ctx(tool_row, operator_id), cache=cache)

        # this is the grant the previous test's tenant-scoped one had
        # to become in order to work at all: both the group and the
        # membership carry no customer. it is therefore not one
        # tenant's, and the row it covers has no customer either, so
        # no evaluation against it can turn on whose display it is.
        assert allowed.decision is True
        assert group.customer_id is None
        assert tool_row.customer_id is None
