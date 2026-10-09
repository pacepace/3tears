"""an AclCache is evicted row by row from the access tables, and never stores what an eviction overtook.

The contract this pins (epoch-task-06, stage 3):

- the evaluator records, per assignment entry, the roles its resolution read, so a role edit evicts
  exactly those entries;
- an entry computed from rows read before an eviction is not stored after it (the read fence);
- a group's row evicts every membership entry naming the group, its own parent entry, and its
  assignment entries;
- a row broadcast that does not say what it names empties only the layer derived from its table,
  and a dropped table empties what was derived from it and nothing more.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from threetears.agent.acl import (
    ActorMembershipKey,
    EvaluationContext,
    Group,
    GroupMembership,
    GroupNamespaceKey,
    MemberType,
    Namespace,
    Role,
    RoleAssignment,
    ScopeType,
    bind_acl_cache_to_access_tables,
    evaluate_decision,
)
from threetears.core.collections import CacheInvalidationMessage, CollectionRegistry

from .fake_loaders import FakeStore, make_cache


def _granted_store() -> tuple[FakeStore, Namespace, UUID, Role, Group]:
    customer, user = uuid4(), uuid4()
    namespace = Namespace(
        id=uuid4(), customer_id=customer, namespace_type="workspace", owner_agent_id=None, owner_namespace=None
    )
    role = Role(id=uuid4(), name="Reader", permissions={"workspace": frozenset({"read"})}, is_built_in=True)
    group = Group(id=uuid4(), name="eng", customer_id=customer)
    store = FakeStore()
    store.add_role(role)
    store.add_group(group)
    store.add_membership(GroupMembership(group.id, MemberType.USER, user, customer))
    store.add_assignment(
        RoleAssignment(
            id=uuid4(),
            role_id=role.id,
            group_id=group.id,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=namespace.id,
            scope_namespace_type=None,
            scope_customer_id=None,
        )
    )
    return store, namespace, user, role, group


class TestARoleEditEvictsOnlyTheEntriesThatReadIt:
    async def test_the_entry_records_its_roles(self) -> None:
        store, namespace, user, role, group = _granted_store()
        cache = make_cache(store)
        assert await evaluate_decision(EvaluationContext(namespace=namespace, action="read", user_id=user), cache=cache)
        key = GroupNamespaceKey(group.id, namespace.id)
        entry = cache.get_group_namespace(key)
        assert entry is not None
        assert entry.role_ids == frozenset({role.id})
        cache.evict_role_row(uuid4())
        assert cache.get_group_namespace(key) is not None
        cache.evict_role_row(role.id)
        assert cache.get_group_namespace(key) is None

    def test_an_entry_that_did_not_record_its_roles_is_evicted_by_any_role(self) -> None:
        cache = make_cache(FakeStore())
        key = GroupNamespaceKey(uuid4(), uuid4())
        cache.put_group_namespace(key, frozenset(), ())
        cache.evict_role_row(uuid4())
        assert cache.get_group_namespace(key) is None


class TestTheReadFence:
    async def test_an_eviction_during_the_read_keeps_the_entry_out(self) -> None:
        store, namespace, user, _role, _group = _granted_store()
        cache = make_cache(store)
        load = store.load_for_user

        async def load_then_revoked(user_id: UUID) -> tuple[GroupMembership, ...]:
            rows = await load(user_id)
            # the membership changes while the evaluator holds the rows it read
            cache.evict_group_member_row("user", user_id)
            return rows

        store.load_for_user = load_then_revoked  # type: ignore[method-assign]
        await evaluate_decision(EvaluationContext(namespace=namespace, action="read", user_id=user), cache=cache)
        assert cache.get_membership(ActorMembershipKey("user", user)) is None

    async def test_with_no_eviction_the_entry_is_stored(self) -> None:
        store, namespace, user, _role, _group = _granted_store()
        cache = make_cache(store)
        await evaluate_decision(EvaluationContext(namespace=namespace, action="read", user_id=user), cache=cache)
        assert cache.get_membership(ActorMembershipKey("user", user)) is not None


class TestAGroupRow:
    def test_evicts_every_membership_entry_naming_it_and_its_own_entries(self) -> None:
        cache = make_cache(FakeStore())
        group, other, ns = uuid4(), uuid4(), uuid4()
        ada, bob = uuid4(), uuid4()
        cache.put_membership(ActorMembershipKey("user", ada), (GroupMembership(group, MemberType.USER, ada, None),))
        cache.put_membership(ActorMembershipKey("user", bob), (GroupMembership(other, MemberType.USER, bob, None),))
        cache.put_membership(ActorMembershipKey("group", group), ())
        cache.put_group_namespace(GroupNamespaceKey(group, ns), frozenset(), ())
        cache.put_group_namespace(GroupNamespaceKey(other, ns), frozenset(), ())
        cache.evict_group_row(group)
        assert cache.get_membership(ActorMembershipKey("user", ada)) is None
        assert cache.get_membership(ActorMembershipKey("group", group)) is None
        assert cache.get_group_namespace(GroupNamespaceKey(group, ns)) is None
        assert cache.get_membership(ActorMembershipKey("user", bob)) is not None
        assert cache.get_group_namespace(GroupNamespaceKey(other, ns)) is not None


def _bound() -> tuple[CollectionRegistry, object]:
    registry = CollectionRegistry()
    cache = make_cache(FakeStore())
    bind_acl_cache_to_access_tables(registry, cache)
    cache.put_membership(ActorMembershipKey("user", uuid4()), ())
    cache.put_group_namespace(GroupNamespaceKey(uuid4(), uuid4()), frozenset(), (), role_ids=frozenset())
    return registry, cache


class TestAnUnknownReachEmptiesOnlyItsTablesLayer:
    def test_a_membership_row_that_does_not_name_its_member(self) -> None:
        registry, cache = _bound()
        registry.tell_derived_caches(CacheInvalidationMessage(table="group_members", ids=[f"{uuid4()}", f"{uuid4()}"]))
        assert cache.membership_size == 0  # type: ignore[attr-defined]
        assert cache.group_namespace_size == 1  # type: ignore[attr-defined]

    def test_an_assignment_row_that_does_not_name_its_group(self) -> None:
        registry, cache = _bound()
        registry.tell_derived_caches(CacheInvalidationMessage(table="role_assignments", ids=["customer", f"{uuid4()}"]))
        assert cache.group_namespace_size == 0  # type: ignore[attr-defined]
        assert cache.membership_size == 1  # type: ignore[attr-defined]

    def test_a_named_row_empties_nothing(self) -> None:
        registry, cache = _bound()
        registry.tell_derived_caches(
            CacheInvalidationMessage(
                table="group_members",
                ids=[f"{uuid4()}", f"{uuid4()}"],
                columns={"member_type": "user", "member_id": f"{uuid4()}"},
            )
        )
        registry.tell_derived_caches(
            CacheInvalidationMessage(
                table="role_assignments", ids=["customer", f"{uuid4()}"], columns={"group_id": f"{uuid4()}"}
            )
        )
        assert cache.size == 2  # type: ignore[attr-defined]

    def test_a_dropped_table_empties_what_was_derived_from_it(self) -> None:
        registry, cache = _bound()
        registry.drop_table("group_members", reason="missed")
        assert (cache.membership_size, cache.group_namespace_size) == (0, 1)  # type: ignore[attr-defined]
        registry.drop_table("roles", reason="missed")
        assert cache.group_namespace_size == 0  # type: ignore[attr-defined]
