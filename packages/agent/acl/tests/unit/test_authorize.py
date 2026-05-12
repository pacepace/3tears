"""tests for the canonical :func:`authorize` primitive.

verifies:

- happy path: namespace resolves, evaluator allows -> returns the
  full :class:`EvaluationResult` with non-empty effective_actions
- :func:`authorize_with_trail` returns ``(result, ns_entity)`` so
  resource wrappers needing the entity skip a second lookup
- namespace lookup miss -> :class:`NamespaceNotFound`
  (subclass of :class:`AccessDenied`)
- evaluator deny -> :class:`AccessDenied` carrying action +
  namespace_name + caller identity for downstream audit fan-out
- cache hit-rate: the second authorize call against the same
  ``(actor, namespace)`` serves membership + per-namespace layers
  from cache. instrumentation: count loader method invocations
  across two calls and assert the second adds zero loader hits.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

from threetears.agent.acl import (
    AccessDenied,
    AclCache,
    Group,
    GroupMembership,
    MemberType,
    NamespaceNotFound,
    Role,
    RoleAssignment,
    ScopeType,
    authorize,
    authorize_with_trail,
)

from tests.unit._fake_loaders import FakeStore


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@dataclass
class _StubNamespace:
    """duck-typed namespace entity exposing the four fields the primitive reads."""

    id: UUID
    customer_id: UUID
    namespace_type: str
    owner_agent_id: UUID


class _NamespaceCollectionStub:
    """fake namespace collection backed by a name -> entity dict.

    duck-types :meth:`get_by_name` so tests never reach a real
    Collection. records every lookup on ``call_count`` so cache
    hit-rate tests can assert the namespace lookup is not bypassed
    (the primitive always re-hits the namespace collection because
    that lookup is L1-served and outside the AclCache's scope).
    """

    def __init__(self, entries: dict[str, _StubNamespace]) -> None:
        self._entries = dict(entries)
        self.call_count = 0

    async def get_by_name(self, name: str) -> _StubNamespace | None:
        """return preconfigured entity for ``name`` or ``None``.

        :param name: namespace name to resolve
        :ptype name: str
        :return: entity or None
        :rtype: _StubNamespace | None
        """
        self.call_count += 1
        return self._entries.get(name)


class _CountingFakeStore(FakeStore):
    """:class:`FakeStore` variant that tracks loader invocation counts.

    used by cache-hit-rate tests to assert subsequent authorize calls
    against the same actor + namespace serve from cache without
    further loader trips.
    """

    def __init__(self) -> None:
        super().__init__()
        self.user_membership_calls = 0
        self.agent_membership_calls = 0
        self.assignment_calls = 0

    async def load_for_user(self, user_id: UUID) -> tuple[GroupMembership, ...]:
        """count + delegate.

        :param user_id: user UUID
        :ptype user_id: UUID
        :return: memberships
        :rtype: tuple[GroupMembership, ...]
        """
        self.user_membership_calls += 1
        return await super().load_for_user(user_id)

    async def load_for_agent(self, agent_id: UUID) -> tuple[GroupMembership, ...]:
        """count + delegate.

        :param agent_id: agent UUID
        :ptype agent_id: UUID
        :return: memberships
        :rtype: tuple[GroupMembership, ...]
        """
        self.agent_membership_calls += 1
        return await super().load_for_agent(agent_id)

    async def load_assignments_for_groups(
        self,
        group_ids: tuple[UUID, ...],
        namespace,  # type: ignore[no-untyped-def]
    ) -> tuple[RoleAssignment, ...]:
        """count + delegate.

        :param group_ids: group UUIDs
        :ptype group_ids: tuple[UUID, ...]
        :param namespace: namespace under evaluation
        :return: assignments
        :rtype: tuple[RoleAssignment, ...]
        """
        self.assignment_calls += 1
        return await super().load_assignments_for_groups(group_ids, namespace)


def _grant_user_read(
    *,
    store: _CountingFakeStore | FakeStore,
    user_id: UUID,
    namespace_id: UUID,
    customer_id: UUID,
    namespace_type: str = "memory",
) -> None:
    """seed a single grant: ``user_id`` reads on ``namespace_id``.

    builds a Reader role on the namespace_type bucket, a customer-
    scoped group with ``user_id`` as a member, and a namespace-scope
    assignment binding the group to the role.

    :param store: in-memory loader store
    :ptype store: FakeStore
    :param user_id: user UUID to grant
    :ptype user_id: UUID
    :param namespace_id: namespace UUID to scope the assignment to
    :ptype namespace_id: UUID
    :param customer_id: customer scope for both group + namespace
    :ptype customer_id: UUID
    :param namespace_type: namespace_type discriminator the role
        grants on; also stamped onto the role's permissions bucket
    :ptype namespace_type: str
    :return: nothing
    :rtype: None
    """
    role = Role(
        id=uuid4(),
        name="Reader",
        permissions={namespace_type: frozenset(["read"])},
        is_built_in=True,
    )
    group = Group(id=uuid4(), name="readers", customer_id=customer_id)
    membership = GroupMembership(
        group_id=group.id,
        member_type=MemberType.USER,
        member_id=user_id,
        customer_id=customer_id,
    )
    assignment = RoleAssignment(
        id=uuid4(),
        role_id=role.id,
        group_id=group.id,
        scope_type=ScopeType.NAMESPACE,
        scope_namespace_id=namespace_id,
        scope_namespace_type=None,
        scope_customer_id=None,
    )
    store.add_role(role)
    store.add_group(group)
    store.add_membership(membership)
    store.add_assignment(assignment)


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


class TestAuthorizeAllow:
    """allow path: namespace resolves and evaluator grants action."""

    @pytest.mark.asyncio
    async def test_returns_evaluation_result(self) -> None:
        """authorize returns full result with non-empty effective_actions."""
        customer = uuid4()
        owner = uuid4()
        user = uuid4()
        ns = _StubNamespace(
            id=uuid4(),
            customer_id=customer,
            namespace_type="memory",
            owner_agent_id=owner,
        )
        store = FakeStore()
        _grant_user_read(
            store=store,
            user_id=user,
            namespace_id=ns.id,
            customer_id=customer,
        )
        cache = AclCache(membership_loader=store, grant_loader=store)
        ns_collection = _NamespaceCollectionStub({"memories.test": ns})

        result = await authorize(
            namespace_collection=ns_collection,
            namespace_name="memories.test",
            action="read",
            user_id=user,
            agent_id=None,
            cache=cache,
        )
        assert result.decision is True
        assert "read" in result.effective_actions

    @pytest.mark.asyncio
    async def test_with_trail_returns_entity(self) -> None:
        """authorize_with_trail returns ``(result, ns_entity)``."""
        customer = uuid4()
        owner = uuid4()
        user = uuid4()
        ns = _StubNamespace(
            id=uuid4(),
            customer_id=customer,
            namespace_type="memory",
            owner_agent_id=owner,
        )
        store = FakeStore()
        _grant_user_read(
            store=store,
            user_id=user,
            namespace_id=ns.id,
            customer_id=customer,
        )
        cache = AclCache(membership_loader=store, grant_loader=store)
        ns_collection = _NamespaceCollectionStub({"memories.test": ns})

        result, returned_ns = await authorize_with_trail(
            namespace_collection=ns_collection,
            namespace_name="memories.test",
            action="read",
            user_id=user,
            agent_id=None,
            cache=cache,
        )
        assert result.decision is True
        assert returned_ns is ns


# ---------------------------------------------------------------------------
# denial paths
# ---------------------------------------------------------------------------


class TestAuthorizeDeny:
    """denial paths surface as :class:`AccessDenied` subclasses."""

    @pytest.mark.asyncio
    async def test_namespace_missing_raises_namespace_not_found(self) -> None:
        """absent namespace row -> :class:`NamespaceNotFound`."""
        store = FakeStore()
        cache = AclCache(membership_loader=store, grant_loader=store)
        ns_collection = _NamespaceCollectionStub({})

        with pytest.raises(NamespaceNotFound) as exc_info:
            await authorize(
                namespace_collection=ns_collection,
                namespace_name="memories.absent",
                action="read",
                user_id=uuid4(),
                agent_id=None,
                cache=cache,
            )
        assert exc_info.value.reason == "namespace_not_found"
        assert exc_info.value.namespace_name == "memories.absent"
        # NamespaceNotFound is a subclass of AccessDenied so generic
        # catchers still trip.
        assert isinstance(exc_info.value, AccessDenied)

    @pytest.mark.asyncio
    async def test_evaluator_deny_raises_access_denied(self) -> None:
        """evaluator deny -> :class:`AccessDenied` with reason."""
        customer = uuid4()
        owner = uuid4()
        user = uuid4()
        ns = _StubNamespace(
            id=uuid4(),
            customer_id=customer,
            namespace_type="memory",
            owner_agent_id=owner,
        )
        store = FakeStore()  # no grants seeded
        cache = AclCache(membership_loader=store, grant_loader=store)
        ns_collection = _NamespaceCollectionStub({"memories.test": ns})

        with pytest.raises(AccessDenied) as exc_info:
            await authorize(
                namespace_collection=ns_collection,
                namespace_name="memories.test",
                action="read",
                user_id=user,
                agent_id=None,
                cache=cache,
            )
        assert exc_info.value.reason == "evaluator_deny"
        assert exc_info.value.action == "read"
        assert exc_info.value.namespace_name == "memories.test"
        assert exc_info.value.user_id == user


# ---------------------------------------------------------------------------
# cache hit rate
# ---------------------------------------------------------------------------


class TestCacheHitRate:
    """cache layers are consulted on subsequent calls."""

    @pytest.mark.asyncio
    async def test_second_call_serves_membership_from_cache(self) -> None:
        """two authorize calls produce one membership loader hit."""
        customer = uuid4()
        owner = uuid4()
        user = uuid4()
        ns = _StubNamespace(
            id=uuid4(),
            customer_id=customer,
            namespace_type="memory",
            owner_agent_id=owner,
        )
        store = _CountingFakeStore()
        _grant_user_read(
            store=store,
            user_id=user,
            namespace_id=ns.id,
            customer_id=customer,
        )
        cache = AclCache(membership_loader=store, grant_loader=store)
        ns_collection = _NamespaceCollectionStub({"memories.test": ns})

        for _ in range(2):
            await authorize(
                namespace_collection=ns_collection,
                namespace_name="memories.test",
                action="read",
                user_id=user,
                agent_id=None,
                cache=cache,
            )
        # first call hits loader once, second call serves from cache
        assert store.user_membership_calls == 1
        # per-namespace assignment layer also caches: one assignment
        # call across two authorize invocations
        assert store.assignment_calls == 1

    @pytest.mark.asyncio
    async def test_cache_layers_populate(self) -> None:
        """one authorize call leaves entries in membership + per-ns layers."""
        customer = uuid4()
        owner = uuid4()
        user = uuid4()
        ns = _StubNamespace(
            id=uuid4(),
            customer_id=customer,
            namespace_type="memory",
            owner_agent_id=owner,
        )
        store = _CountingFakeStore()
        _grant_user_read(
            store=store,
            user_id=user,
            namespace_id=ns.id,
            customer_id=customer,
        )
        cache = AclCache(membership_loader=store, grant_loader=store)
        ns_collection = _NamespaceCollectionStub({"memories.test": ns})

        assert cache.size == 0
        await authorize(
            namespace_collection=ns_collection,
            namespace_name="memories.test",
            action="read",
            user_id=user,
            agent_id=None,
            cache=cache,
        )
        # one membership entry + one per-namespace assignment entry
        assert cache.membership_size == 1
        assert cache.group_namespace_size == 1
