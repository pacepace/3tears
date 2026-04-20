"""unit tests for :class:`AclCache` — three layers, ttl, invalidation.

verifies:

- each layer (membership, per-namespace assignment, per-type+customer
  assignment) stores and retrieves entries independently
- ttl expiry returns ``None`` and evicts the entry
- targeted invalidation clears only the named entries
- bulk invalidation (group, namespace, all) fans out correctly
- size accessors report the right counts per layer
- thread-safety: the lock holds across put/get pairs without
  corrupting state under concurrent access (smoke-tested with two
  threads racing inserts; verifies the dict never trips a
  ``RuntimeError: dictionary changed size during iteration`` on
  the invalidate-during-walk path)
"""

from __future__ import annotations

import time
from threading import Thread
from uuid import UUID, uuid4

from threetears.agent.acl import (
    AclCache,
    ActorMembershipKey,
    GroupMembership,
    GroupNamespaceKey,
    GroupTypeCustomerKey,
    Namespace,
    Role,
    RoleAssignment,
)


class _NoopMembershipLoader:
    """membership loader stub returning empty tuples.

    cache-layer tests never exercise loader traversal; the cache
    stores and retrieves entries by direct :meth:`put_*` /
    :meth:`get_*` calls. the loader handle still has to satisfy the
    :class:`AclCache` constructor contract, so this stub exists to
    supply a valid reference without carrying test state.
    """

    async def load_for_user(self, user_id: UUID) -> tuple[GroupMembership, ...]:
        """
        return empty tuple for every user id.

        :param user_id: user UUID (ignored)
        :ptype user_id: UUID
        :return: empty tuple
        :rtype: tuple[GroupMembership, ...]
        """
        del user_id
        return ()

    async def load_for_agent(self, agent_id: UUID) -> tuple[GroupMembership, ...]:
        """
        return empty tuple for every agent id.

        :param agent_id: agent UUID (ignored)
        :ptype agent_id: UUID
        :return: empty tuple
        :rtype: tuple[GroupMembership, ...]
        """
        del agent_id
        return ()


class _NoopGrantLoader:
    """grant loader stub returning empty assignments / roles / groups.

    same rationale as :class:`_NoopMembershipLoader`: the cache's
    direct-access tests never traverse the loader path, so the stub
    satisfies the constructor without side effects.
    """

    async def load_assignments_for_groups(
        self,
        group_ids: tuple[UUID, ...],
        namespace: Namespace,
    ) -> tuple[RoleAssignment, ...]:
        """
        return empty tuple for every group set.

        :param group_ids: group UUIDs (ignored)
        :ptype group_ids: tuple[UUID, ...]
        :param namespace: target namespace (ignored)
        :ptype namespace: Namespace
        :return: empty tuple
        :rtype: tuple[RoleAssignment, ...]
        """
        del group_ids
        del namespace
        return ()

    async def load_roles(self, role_ids: tuple[UUID, ...]) -> dict[UUID, Role]:
        """
        return empty mapping for every role set.

        :param role_ids: role UUIDs (ignored)
        :ptype role_ids: tuple[UUID, ...]
        :return: empty mapping
        :rtype: dict[UUID, Role]
        """
        del role_ids
        return {}

    async def load_groups(self, group_ids: tuple[UUID, ...]) -> dict[UUID, object]:
        """
        return empty mapping for every group set.

        :param group_ids: group UUIDs (ignored)
        :ptype group_ids: tuple[UUID, ...]
        :return: empty mapping
        :rtype: dict[UUID, object]
        """
        del group_ids
        return {}


def _make_cache(ttl_seconds: int = 60) -> AclCache:
    """
    construct an :class:`AclCache` wired with noop loaders for unit tests.

    :param ttl_seconds: TTL forwarded to the cache
    :ptype ttl_seconds: int
    :return: ready-to-use cache instance
    :rtype: AclCache
    """
    return AclCache(
        membership_loader=_NoopMembershipLoader(),
        grant_loader=_NoopGrantLoader(),
        ttl_seconds=ttl_seconds,
    )


# ---------------------------------------------------------------------------
# membership layer
# ---------------------------------------------------------------------------


class TestMembershipLayer:
    """basic put/get/invalidate on the membership layer."""

    def test_put_then_get_returns_entry(self) -> None:
        """entry is round-tripped intact."""
        cache = _make_cache(ttl_seconds=60)
        user = uuid4()
        groups = (uuid4(), uuid4())
        key = ActorMembershipKey(actor_kind="user", actor_id=user)
        cache.put_membership(key, groups)
        entry = cache.get_membership(key)
        assert entry is not None
        assert entry.group_ids == groups

    def test_get_unknown_returns_none(self) -> None:
        """miss returns None without raising."""
        cache = _make_cache(ttl_seconds=60)
        key = ActorMembershipKey(actor_kind="user", actor_id=uuid4())
        assert cache.get_membership(key) is None

    def test_invalidate_drops_entry(self) -> None:
        """invalidate clears the cached entry."""
        cache = _make_cache(ttl_seconds=60)
        key = ActorMembershipKey(actor_kind="user", actor_id=uuid4())
        cache.put_membership(key, ())
        cache.invalidate_membership(key)
        assert cache.get_membership(key) is None

    def test_invalidate_for_actor_helper(self) -> None:
        """convenience helper builds the key from actor_kind + actor_id."""
        cache = _make_cache(ttl_seconds=60)
        user = uuid4()
        key = ActorMembershipKey(actor_kind="user", actor_id=user)
        cache.put_membership(key, ())
        cache.invalidate_membership_for_actor("user", user)
        assert cache.get_membership(key) is None

    def test_user_and_agent_keys_are_distinct(self) -> None:
        """``actor_kind`` is part of the key; same id gives two entries."""
        cache = _make_cache(ttl_seconds=60)
        actor_id = uuid4()
        user_key = ActorMembershipKey(actor_kind="user", actor_id=actor_id)
        agent_key = ActorMembershipKey(actor_kind="agent", actor_id=actor_id)
        cache.put_membership(user_key, (uuid4(),))
        cache.put_membership(agent_key, (uuid4(),))
        assert cache.get_membership(user_key) is not None
        assert cache.get_membership(agent_key) is not None
        cache.invalidate_membership(user_key)
        assert cache.get_membership(user_key) is None
        assert cache.get_membership(agent_key) is not None


# ---------------------------------------------------------------------------
# per-namespace layer
# ---------------------------------------------------------------------------


class TestGroupNamespaceLayer:
    """put/get/invalidate on the per-namespace layer."""

    def test_put_then_get(self) -> None:
        """entry round-trip preserves actions and trails."""
        cache = _make_cache(ttl_seconds=60)
        key = GroupNamespaceKey(group_id=uuid4(), namespace_id=uuid4())
        cache.put_group_namespace(key, frozenset({"read"}), ())
        entry = cache.get_group_namespace(key)
        assert entry is not None
        assert entry.actions == frozenset({"read"})

    def test_invalidate_namespace_drops_every_group(self) -> None:
        """``invalidate_namespace`` drops every group's entry for the namespace."""
        cache = _make_cache(ttl_seconds=60)
        ns = uuid4()
        key_a = GroupNamespaceKey(group_id=uuid4(), namespace_id=ns)
        key_b = GroupNamespaceKey(group_id=uuid4(), namespace_id=ns)
        # entry for a different namespace should survive
        key_other = GroupNamespaceKey(group_id=uuid4(), namespace_id=uuid4())
        cache.put_group_namespace(key_a, frozenset({"read"}), ())
        cache.put_group_namespace(key_b, frozenset({"read"}), ())
        cache.put_group_namespace(key_other, frozenset({"read"}), ())
        cache.invalidate_namespace(ns)
        assert cache.get_group_namespace(key_a) is None
        assert cache.get_group_namespace(key_b) is None
        assert cache.get_group_namespace(key_other) is not None


# ---------------------------------------------------------------------------
# type+customer layer
# ---------------------------------------------------------------------------


class TestGroupTypeCustomerLayer:
    """put/get/invalidate on the type+customer layer."""

    def test_put_then_get(self) -> None:
        """entry round-trip works."""
        cache = _make_cache(ttl_seconds=60)
        key = GroupTypeCustomerKey(
            group_id=uuid4(), namespace_type="workspace", customer_id=uuid4(),
        )
        cache.put_group_type_customer(key, frozenset({"read"}), ())
        entry = cache.get_group_type_customer(key)
        assert entry is not None
        assert entry.actions == frozenset({"read"})

    def test_invalidate_specific_key(self) -> None:
        """targeted invalidate drops one entry without touching others."""
        cache = _make_cache(ttl_seconds=60)
        key = GroupTypeCustomerKey(
            group_id=uuid4(), namespace_type="workspace", customer_id=uuid4(),
        )
        other = GroupTypeCustomerKey(
            group_id=uuid4(), namespace_type="workspace", customer_id=uuid4(),
        )
        cache.put_group_type_customer(key, frozenset({"read"}), ())
        cache.put_group_type_customer(other, frozenset({"read"}), ())
        cache.invalidate_group_type_customer(key)
        assert cache.get_group_type_customer(key) is None
        assert cache.get_group_type_customer(other) is not None


# ---------------------------------------------------------------------------
# group fan-out invalidation
# ---------------------------------------------------------------------------


class TestGroupFanOutInvalidation:
    """``invalidate_group`` cleans both assignment layers but not membership."""

    def test_drops_per_namespace_entries_for_group(self) -> None:
        """every per-namespace entry naming the group is removed."""
        cache = _make_cache(ttl_seconds=60)
        group = uuid4()
        other_group = uuid4()
        ns_a, ns_b = uuid4(), uuid4()
        cache.put_group_namespace(
            GroupNamespaceKey(group_id=group, namespace_id=ns_a),
            frozenset({"read"}), (),
        )
        cache.put_group_namespace(
            GroupNamespaceKey(group_id=group, namespace_id=ns_b),
            frozenset({"read"}), (),
        )
        cache.put_group_namespace(
            GroupNamespaceKey(group_id=other_group, namespace_id=ns_a),
            frozenset({"read"}), (),
        )
        cache.invalidate_group(group)
        assert cache.get_group_namespace(
            GroupNamespaceKey(group_id=group, namespace_id=ns_a),
        ) is None
        assert cache.get_group_namespace(
            GroupNamespaceKey(group_id=group, namespace_id=ns_b),
        ) is None
        # other group untouched
        assert cache.get_group_namespace(
            GroupNamespaceKey(group_id=other_group, namespace_id=ns_a),
        ) is not None

    def test_drops_type_customer_entries_for_group(self) -> None:
        """type+customer entries for the group are also dropped."""
        cache = _make_cache(ttl_seconds=60)
        group = uuid4()
        customer = uuid4()
        cache.put_group_type_customer(
            GroupTypeCustomerKey(
                group_id=group, namespace_type="workspace",
                customer_id=customer,
            ),
            frozenset({"read"}), (),
        )
        cache.invalidate_group(group)
        assert cache.get_group_type_customer(
            GroupTypeCustomerKey(
                group_id=group, namespace_type="workspace",
                customer_id=customer,
            ),
        ) is None

    def test_does_not_touch_membership_layer(self) -> None:
        """``invalidate_group`` is scoped to the assignment layers."""
        cache = _make_cache(ttl_seconds=60)
        actor = uuid4()
        group = uuid4()
        m_key = ActorMembershipKey(actor_kind="user", actor_id=actor)
        cache.put_membership(m_key, (group,))
        cache.invalidate_group(group)
        # membership layer is the caller's job to invalidate separately
        assert cache.get_membership(m_key) is not None


# ---------------------------------------------------------------------------
# ttl
# ---------------------------------------------------------------------------


class TestTtl:
    """expired entries return None and are evicted."""

    def test_membership_expires(self) -> None:
        """membership entry past ttl is dropped on read."""
        cache = _make_cache(ttl_seconds=0)  # ttl 0s -> always expired
        key = ActorMembershipKey(actor_kind="user", actor_id=uuid4())
        cache.put_membership(key, ())
        # tiny sleep so the freshness check sees a non-zero age
        time.sleep(0.001)
        assert cache.get_membership(key) is None
        # eviction happened: size drops to 0
        assert cache.membership_size == 0

    def test_per_namespace_expires(self) -> None:
        """per-namespace entry past ttl is dropped on read."""
        cache = _make_cache(ttl_seconds=0)
        key = GroupNamespaceKey(group_id=uuid4(), namespace_id=uuid4())
        cache.put_group_namespace(key, frozenset({"read"}), ())
        time.sleep(0.001)
        assert cache.get_group_namespace(key) is None
        assert cache.group_namespace_size == 0

    def test_type_customer_expires(self) -> None:
        """type+customer entry past ttl is dropped on read."""
        cache = _make_cache(ttl_seconds=0)
        key = GroupTypeCustomerKey(
            group_id=uuid4(), namespace_type="workspace",
            customer_id=uuid4(),
        )
        cache.put_group_type_customer(key, frozenset({"read"}), ())
        time.sleep(0.001)
        assert cache.get_group_type_customer(key) is None
        assert cache.group_type_customer_size == 0


# ---------------------------------------------------------------------------
# bulk operations
# ---------------------------------------------------------------------------


class TestBulkOperations:
    """``invalidate_all`` and the size accessors."""

    def test_invalidate_all_clears_every_layer(self) -> None:
        """one call drops everything across the three layers."""
        cache = _make_cache(ttl_seconds=60)
        cache.put_membership(
            ActorMembershipKey(actor_kind="user", actor_id=uuid4()), (),
        )
        cache.put_group_namespace(
            GroupNamespaceKey(group_id=uuid4(), namespace_id=uuid4()),
            frozenset({"read"}), (),
        )
        cache.put_group_type_customer(
            GroupTypeCustomerKey(
                group_id=uuid4(), namespace_type="workspace",
                customer_id=uuid4(),
            ),
            frozenset({"read"}), (),
        )
        assert cache.size == 3
        cache.invalidate_all()
        assert cache.size == 0
        assert cache.membership_size == 0
        assert cache.group_namespace_size == 0
        assert cache.group_type_customer_size == 0

    def test_size_accessors_per_layer(self) -> None:
        """each layer reports its own size correctly."""
        cache = _make_cache(ttl_seconds=60)
        for _ in range(3):
            cache.put_membership(
                ActorMembershipKey(actor_kind="user", actor_id=uuid4()), (),
            )
        for _ in range(2):
            cache.put_group_namespace(
                GroupNamespaceKey(group_id=uuid4(), namespace_id=uuid4()),
                frozenset({"read"}), (),
            )
        cache.put_group_type_customer(
            GroupTypeCustomerKey(
                group_id=uuid4(), namespace_type="workspace",
                customer_id=uuid4(),
            ),
            frozenset({"read"}), (),
        )
        assert cache.membership_size == 3
        assert cache.group_namespace_size == 2
        assert cache.group_type_customer_size == 1
        assert cache.size == 6


# ---------------------------------------------------------------------------
# concurrent access smoke test
# ---------------------------------------------------------------------------


class TestConcurrentAccess:
    """two threads racing inserts + a fan-out invalidation do not crash."""

    def test_invalidate_during_inserts(self) -> None:
        """invalidate_all between inserts is safe under the cache lock.

        regression guard for the dict-mutation race the bulk-invalidate
        paths used to hit before the lock was added. running 1000
        iterations with two threads inserting and one fan-out
        invalidation should never raise.
        """
        cache = _make_cache(ttl_seconds=60)
        errors: list[BaseException] = []

        def inserter() -> None:
            try:
                for _ in range(500):
                    cache.put_membership(
                        ActorMembershipKey(
                            actor_kind="user", actor_id=uuid4(),
                        ),
                        (),
                    )
            except BaseException as exc:  # capture for the parent
                errors.append(exc)

        def invalidator() -> None:
            try:
                for _ in range(500):
                    cache.invalidate_all()
            except BaseException as exc:
                errors.append(exc)

        threads = [
            Thread(target=inserter),
            Thread(target=inserter),
            Thread(target=invalidator),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
