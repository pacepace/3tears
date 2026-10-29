"""a person's access is a function of their membership shape, and a shape's actor answers exactly as they do.

The contract this pins:

- a shape is the sorted set of a person's direct ``(group, membership customer)`` pairs: order and
  duplicates do not change it, the customer stamp does, and rows that are not a user's are not in it;
- a shape's actor is a stable id no real principal has, whose memberships are the shape; a person
  is still read from the real loader, and an actor whose shape was let go is refused, never read
  from the database as a member of nothing;
- every evaluation as a shape's actor equals the same evaluation as any person with the shape;
- a footprint reaches every group the shape and the signing agents reach through nesting, the roles
  those groups hold, and the agents;
- a per-person cache of membership rows is dropped only by a row naming that person.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4, uuid7

import pytest

from threetears.agent.acl import (
    AccessShape,
    CallerAccessCache,
    CallerKey,
    ShapeActors,
    bind_caller_cache_to_memberships,
    evaluate_many_with_trail,
)
from threetears.agent.acl.access_shapes import access_footprint
from threetears.agent.acl.cache import AclCache
from threetears.agent.acl.types import (
    EvaluationContext,
    Group,
    GroupMembership,
    MemberType,
    Namespace,
    Role,
    RoleAssignment,
    ScopeType,
)
from threetears.core.collections import CollectionRegistry
from threetears.core.collections.registry import CacheInvalidationMessage

from .fake_loaders import FakeBatchingStore, FakeStore


class _HeldStore(FakeBatchingStore):
    """a store that can list the assignments a set of groups holds, and counts what it is asked."""

    def __init__(self) -> None:
        super().__init__()
        self.reads: list[str] = []

    async def load_for_user(self, user_id: UUID) -> tuple[GroupMembership, ...]:
        self.reads.append("load_for_user")
        return await super().load_for_user(user_id)

    async def load_assignments_held_by_groups(self, group_ids: tuple[UUID, ...]) -> tuple[RoleAssignment, ...]:
        self.reads.append("load_assignments_held_by_groups")
        ids = set(group_ids)
        return tuple(a for a in self.assignments if a.group_id in ids)


def _grant(store: FakeStore, group: UUID, role: UUID, namespace: Namespace) -> None:
    store.add_assignment(
        RoleAssignment(
            id=uuid4(),
            role_id=role,
            group_id=group,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=namespace.id,
            scope_namespace_type=None,
            scope_customer_id=None,
        )
    )


class _Estate:
    """two readers in the same bundle nested in two state groups; a third in one state alone; an agent."""

    def __init__(self, store: FakeStore) -> None:
        self.customer = uuid4()
        self.reader, self.twin, self.other, self.agent = uuid7(), uuid7(), uuid7(), uuid4()
        self.read_role = store.add_role(Role(uuid4(), "Reader", {"tool": frozenset({"read"})}, is_built_in=True)).id
        self.call_role = store.add_role(
            Role(uuid4(), "Caller", {"tool": frozenset({"tool.call"})}, is_built_in=True)
        ).id
        self.bundle = store.add_group(Group(uuid4(), "bundle", self.customer)).id
        self.states = [store.add_group(Group(uuid4(), f"state-{i}", self.customer)).id for i in range(3)]
        self.agents = store.add_group(Group(uuid4(), "signers", None)).id
        for person in (self.reader, self.twin):
            store.add_membership(GroupMembership(self.bundle, MemberType.USER, person, self.customer))
        store.add_membership(GroupMembership(self.states[2], MemberType.USER, self.other, self.customer))
        for state in self.states[:2]:
            store.add_membership(GroupMembership(state, MemberType.GROUP, self.bundle, self.customer))
        store.add_membership(GroupMembership(self.agents, MemberType.AGENT, self.agent, None))
        self.namespaces = [
            Namespace(
                id=uuid4(), customer_id=self.customer, namespace_type="tool", owner_agent_id=None, owner_namespace=None
            )
            for _ in range(6)
        ]
        for index, namespace in enumerate(self.namespaces):
            _grant(store, self.states[index % 3], self.read_role, namespace)
            _grant(store, self.agents, self.call_role, namespace)
            _grant(store, self.agents, self.read_role, namespace)


def _row(group: UUID, person: UUID, customer: UUID | None, kind: MemberType = MemberType.USER) -> GroupMembership:
    return GroupMembership(group_id=group, member_type=kind, member_id=person, customer_id=customer)


class TestAShape:
    def test_order_and_duplicates_do_not_change_it(self) -> None:
        person, a, b, customer = uuid4(), uuid4(), uuid4(), uuid4()
        one = AccessShape.of([_row(a, person, customer), _row(b, person, None)])
        two = AccessShape.of([_row(b, person, None), _row(a, person, customer), _row(a, person, customer)])
        assert one == two
        assert one.digest == two.digest

    def test_two_people_with_the_same_rows_have_one_shape(self) -> None:
        group, customer = uuid4(), uuid4()
        assert AccessShape.of([_row(group, uuid4(), customer)]) == AccessShape.of([_row(group, uuid4(), customer)])

    def test_the_membership_customer_is_part_of_it(self) -> None:
        person, group = uuid4(), uuid4()
        assert (
            AccessShape.of([_row(group, person, uuid4())]).digest != AccessShape.of([_row(group, person, None)]).digest
        )

    def test_rows_that_are_not_a_users_are_not_in_it(self) -> None:
        person, group = uuid4(), uuid4()
        shape = AccessShape.of([_row(group, person, None), _row(uuid4(), person, None, MemberType.AGENT)])
        assert shape.group_ids == frozenset({group})

    def test_no_rows_is_the_empty_shape(self) -> None:
        assert AccessShape.of([]).memberships == ()


class TestAShapesActor:
    def test_it_is_stable_across_processes_and_no_real_principals(self) -> None:
        shape = AccessShape.of([_row(uuid4(), uuid4(), None)])
        actor = ShapeActors(FakeStore()).actor_for(shape)
        assert actor == ShapeActors(FakeStore()).actor_for(AccessShape(shape.memberships))
        # a version-8 id, where every principal is version 4 or 7
        assert actor.version == 8

    def test_different_shapes_have_different_actors(self) -> None:
        actors = ShapeActors(FakeStore())
        assert actors.actor_for(AccessShape.of([_row(uuid4(), uuid4(), None)])) != actors.actor_for(AccessShape.of([]))

    async def test_its_memberships_are_the_shape_and_a_person_is_read_from_the_real_loader(self) -> None:
        store = _HeldStore()
        estate = _Estate(store)
        actors = ShapeActors(store)
        shape = AccessShape.of(await store.load_for_user(estate.reader))
        actor = actors.actor_for(shape)
        rows = await actors.load_for_user(actor)
        assert {(row.group_id, row.customer_id) for row in rows} == set(shape.memberships)
        assert all(row.member_id == actor and row.member_type is MemberType.USER for row in rows)
        store.reads.clear()
        assert await actors.load_for_user(estate.reader) == tuple(await store.load_for_user(estate.reader))
        assert store.reads == ["load_for_user", "load_for_user"]

    async def test_an_actor_whose_shape_was_let_go_is_refused(self) -> None:
        store = _HeldStore()
        actors = ShapeActors(store, max_shapes=1)
        first = actors.actor_for(AccessShape.of([_row(uuid4(), uuid4(), None)]))
        actors.actor_for(AccessShape.of([_row(uuid4(), uuid4(), None)]))
        store.reads.clear()
        with pytest.raises(LookupError):
            await actors.load_for_user(first)
        assert store.reads == []

    async def test_groups_and_agents_are_read_from_the_real_loader(self) -> None:
        store = _HeldStore()
        estate = _Estate(store)
        actors = ShapeActors(store)
        assert await actors.load_for_agent(estate.agent) == tuple(await store.load_for_agent(estate.agent))
        assert await actors.load_for_group(estate.bundle) == tuple(await store.load_for_group(estate.bundle))
        assert set(await actors.load_for_groups((estate.bundle,))) == set(await store.load_for_group(estate.bundle))
        # a loader that reads one group at a time is asked one group at a time
        plain = ShapeActors(FakeStore(memberships=list(store.memberships)))
        assert set(await plain.load_for_groups((estate.bundle,))) == set(await store.load_for_group(estate.bundle))

    def test_it_holds_at_least_one_shape(self) -> None:
        with pytest.raises(ValueError):
            ShapeActors(FakeStore(), max_shapes=0)

    async def test_every_evaluation_as_the_actor_equals_it_as_a_person_with_the_shape(self) -> None:
        store = _HeldStore()
        estate = _Estate(store)
        actors = ShapeActors(store)
        shape = AccessShape.of(await store.load_for_user(estate.reader))
        actor = actors.actor_for(shape)

        def contexts(user: UUID) -> list[EvaluationContext]:
            found: list[EvaluationContext] = []
            for namespace in estate.namespaces:
                for action in ("read", "tool.call"):
                    found.append(EvaluationContext(namespace=namespace, action=action, user_id=user))
                    found.append(
                        EvaluationContext(namespace=namespace, action=action, user_id=user, agent_id=estate.agent)
                    )
            return found

        def decisions(results: list[Any]) -> list[tuple[bool, frozenset[str]]]:
            return [(result.decision, frozenset(result.effective_actions)) for result in results]

        as_actor = await evaluate_many_with_trail(
            contexts(actor), cache=AclCache(membership_loader=actors, grant_loader=store)
        )
        for person in (estate.reader, estate.twin):
            as_person = await evaluate_many_with_trail(
                contexts(person), cache=AclCache(membership_loader=store, grant_loader=store)
            )
            assert decisions(as_actor) == decisions(as_person)
        assert any(allowed for allowed, _actions in decisions(as_actor))
        other = await evaluate_many_with_trail(
            contexts(estate.other), cache=AclCache(membership_loader=store, grant_loader=store)
        )
        assert decisions(other) != decisions(as_actor)


class TestAFootprint:
    async def test_it_reaches_nested_groups_the_agents_groups_their_roles_and_the_agents(self) -> None:
        store = _HeldStore()
        estate = _Estate(store)
        actors = ShapeActors(store)
        shape = AccessShape.of(await store.load_for_user(estate.reader))
        footprint = await access_footprint(
            AclCache(membership_loader=actors, grant_loader=store), shape=shape, agent_ids=[estate.agent]
        )
        assert footprint.groups == frozenset({estate.bundle, *estate.states[:2], estate.agents})
        assert footprint.roles == frozenset({estate.read_role, estate.call_role})
        assert footprint.agents == frozenset({estate.agent})
        deps = footprint.deps(["namespaces"])
        assert f"g:{estate.states[0].hex}" in deps
        assert f"g:{estate.states[2].hex}" not in deps
        assert f"r:{estate.read_role.hex}" in deps
        assert f"a:{estate.agent.hex}" in deps
        assert "t:namespaces" in deps

    async def test_roles_it_cannot_list_make_every_role_a_dep(self) -> None:
        store = FakeStore()
        estate = _Estate(store)
        shape = AccessShape.of(await store.load_for_user(estate.reader))
        footprint = await access_footprint(
            AclCache(membership_loader=ShapeActors(store), grant_loader=store), shape=shape, agent_ids=[]
        )
        assert footprint.roles is None
        assert "r:*" in footprint.deps([])


def _member_row(member_type: str, member_id: Any, table: str = "group_members") -> CacheInvalidationMessage:
    return CacheInvalidationMessage(
        table=table,
        ids=[f"{uuid4()}", f"{uuid4()}"],
        columns={"member_type": member_type, "member_id": f"{member_id}"},
    )


class TestAPerPersonMembershipCache:
    def _filled(self) -> tuple[CollectionRegistry, CallerAccessCache[Any], CallerKey, CallerKey]:
        registry = CollectionRegistry()
        cache: CallerAccessCache[Any] = CallerAccessCache()
        cache.followed_by(lambda: True)
        bind_caller_cache_to_memberships(registry, cache)
        ingress = uuid4()
        reader, other = CallerKey(ingress, uuid4()), CallerKey(ingress, uuid4())
        for key in (reader, other):
            assert cache.put(key, AccessShape.of([]), fence=cache.read_fence())
        return registry, cache, reader, other

    def test_a_row_naming_the_person_drops_them_alone(self) -> None:
        registry, cache, reader, other = self._filled()
        registry.tell_derived_caches(_member_row("user", reader.user_id))
        assert cache.get(reader) is None
        assert cache.get(other) is not None

    def test_a_nested_groups_row_drops_nobody(self) -> None:
        registry, cache, reader, other = self._filled()
        registry.tell_derived_caches(_member_row("group", uuid4()))
        assert cache.get(reader) is not None
        assert cache.get(other) is not None

    @pytest.mark.parametrize("table", ["groups", "roles", "role_assignments", "namespaces"])
    def test_no_other_access_tables_row_drops_anybody(self, table: str) -> None:
        registry, cache, reader, _other = self._filled()
        registry.tell_derived_caches(CacheInvalidationMessage(table=table, ids=[f"{uuid4()}"], columns={}))
        assert cache.get(reader) is not None

    def test_a_row_that_does_not_say_what_it_names_drops_everybody(self) -> None:
        registry, cache, reader, other = self._filled()
        registry.tell_derived_caches(CacheInvalidationMessage(table="group_members", ids=["x"], columns={}))
        assert cache.get(reader) is None
        assert cache.get(other) is None

    def test_a_dropped_table_drops_everybody(self) -> None:
        registry, cache, reader, _other = self._filled()
        registry.drop_table("group_members", reason="test")
        assert cache.get(reader) is None


def test_an_access_cache_says_whether_anything_follows_it() -> None:
    cache = AclCache(membership_loader=FakeStore(), grant_loader=FakeStore())
    assert cache.trusted and not cache.followed, "a scratch cache is trusted, and nothing follows it"
    cache.followed_by(lambda: True)
    assert cache.followed
    cache.followed_by(None)
    assert not cache.followed
