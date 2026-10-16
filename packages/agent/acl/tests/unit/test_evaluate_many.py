"""evaluate_many_with_trail: the answers of evaluating one at a time, at reads that do not grow per namespace.

The shape it exists for: a person in a subscriber bundle that is a member of dozens of state
groups, each granting its own namespaces. Evaluated one namespace at a time, every (group,
namespace) pair the cache did not hold cost its own assignment and group reads -- tens of
thousands of round trips for one audit snapshot.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from threetears.agent.acl import HeldAssignmentsLoader, evaluate_decision, evaluate_many_with_trail, evaluate_with_trail
from threetears.agent.acl.cache import GroupNamespaceKey
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
from threetears.core.namespaces import build_agent_namespace_name

from .fake_loaders import FakeStore, make_cache


class _CountingStore(FakeStore):
    """a :class:`FakeStore` that counts every read the evaluator makes."""

    def __init__(self) -> None:
        super().__init__()
        self.reads: dict[str, int] = {}

    def _count(self, name: str) -> None:
        self.reads[name] = self.reads.get(name, 0) + 1

    async def load_for_user(self, user_id: UUID) -> tuple[GroupMembership, ...]:
        self._count("load_for_user")
        return await super().load_for_user(user_id)

    async def load_for_agent(self, agent_id: UUID) -> tuple[GroupMembership, ...]:
        self._count("load_for_agent")
        return await super().load_for_agent(agent_id)

    async def load_for_group(self, group_id: UUID) -> tuple[GroupMembership, ...]:
        self._count("load_for_group")
        return await super().load_for_group(group_id)

    async def load_assignments_for_groups(
        self, group_ids: tuple[UUID, ...], namespace: Namespace
    ) -> tuple[RoleAssignment, ...]:
        self._count("load_assignments_for_groups")
        return await super().load_assignments_for_groups(group_ids, namespace)

    async def load_roles(self, role_ids: tuple[UUID, ...]) -> dict[UUID, Role]:
        self._count("load_roles")
        return await super().load_roles(role_ids)

    async def load_groups(self, group_ids: tuple[UUID, ...]) -> dict[UUID, object]:
        self._count("load_groups")
        return await super().load_groups(group_ids)


class _HeldStore(_CountingStore):
    """a counting store that is also a :class:`HeldAssignmentsLoader`."""

    async def load_assignments_held_by_groups(self, group_ids: tuple[UUID, ...]) -> tuple[RoleAssignment, ...]:
        self._count("load_assignments_held_by_groups")
        ids = set(group_ids)
        return tuple(a for a in self.assignments if a.group_id in ids)


class _Bundle:
    """a person in a bundle nested in ``states`` state groups, each granting ``per_state`` namespaces.

    alongside it, what the walls must keep out: a state group of another customer granting on
    this customer's namespaces, a customer-authored role of another customer, and a namespace of
    another customer granted by one of the states. one namespace is granted by two states with
    different roles, and an agent owns one namespace outright.
    """

    def __init__(self, store: FakeStore, *, states: int, per_state: int) -> None:
        self.customer, other = uuid4(), uuid4()
        self.user, self.agent = uuid4(), uuid4()
        reader = store.add_role(Role(uuid4(), "Reader", {"tool": frozenset({"read"})}, is_built_in=True))
        caller = store.add_role(Role(uuid4(), "Caller", {"tool": frozenset({"tool.call"})}, is_built_in=True))
        foreign_role = store.add_role(
            Role(uuid4(), "Theirs", {"tool": frozenset({"delete"})}, is_built_in=False, customer_id=other)
        )
        self.bundle = store.add_group(Group(uuid4(), "bundle", self.customer)).id
        store.add_membership(GroupMembership(self.bundle, MemberType.USER, self.user, self.customer))
        store.add_membership(GroupMembership(self.bundle, MemberType.AGENT, self.agent, self.customer))
        self.namespaces: list[Namespace] = []
        state_ids: list[UUID] = []
        for s in range(states):
            state = store.add_group(Group(uuid4(), f"state-{s}", self.customer)).id
            state_ids.append(state)
            store.add_membership(GroupMembership(state, MemberType.GROUP, self.bundle, self.customer))
            for _ in range(per_state):
                namespace = self._namespace(self.customer)
                self._grant(store, state, reader.id, namespace)
                self._grant(store, state, foreign_role.id, namespace)  # another customer's role: nothing
        shared = self._namespace(self.customer)
        self._grant(store, state_ids[0], reader.id, shared)
        self._grant(store, state_ids[-1], caller.id, shared)
        foreign_state = store.add_group(Group(uuid4(), "their-state", other)).id
        store.add_membership(GroupMembership(foreign_state, MemberType.GROUP, self.bundle, other))
        self._grant(store, foreign_state, caller.id, self.namespaces[0])  # another customer's group: nothing
        theirs = self._namespace(other)
        self._grant(store, state_ids[0], reader.id, theirs)  # a customer group on another customer's namespace
        self.owned = Namespace(
            id=uuid4(),
            customer_id=self.customer,
            namespace_type="tool",
            owner_agent_id=self.agent,
            owner_namespace=build_agent_namespace_name(self.agent),
        )
        self.namespaces.append(self.owned)

    def _namespace(self, customer: UUID) -> Namespace:
        namespace = Namespace(
            id=uuid4(), customer_id=customer, namespace_type="tool", owner_agent_id=None, owner_namespace=None
        )
        self.namespaces.append(namespace)
        return namespace

    @staticmethod
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

    def contexts(self) -> list[EvaluationContext]:
        """a person, a person with their agent, and a direct member of the bundle, on every namespace."""
        found: list[EvaluationContext] = []
        for namespace in self.namespaces:
            found.append(EvaluationContext(namespace=namespace, action="read", user_id=self.user))
            found.append(EvaluationContext(namespace=namespace, action="read", user_id=self.user, agent_id=self.agent))
            found.append(EvaluationContext(namespace=namespace, action="tool.call", group_id=self.bundle))
        return found


async def _one_at_a_time(store: FakeStore, contexts: list[EvaluationContext]) -> list[object]:
    """each context evaluated alone, on a cache of its own."""
    return [await evaluate_with_trail(ctx, cache=make_cache(store)) for ctx in contexts]


class TestTheAnswersAreEvaluatingOneAtATime:
    async def test_with_a_held_assignments_loader(self) -> None:
        store = _HeldStore()
        contexts = _Bundle(store, states=4, per_state=3).contexts()
        assert await evaluate_many_with_trail(contexts, cache=make_cache(store)) == await _one_at_a_time(
            store, contexts
        )

    async def test_with_a_loader_that_reads_per_namespace(self) -> None:
        store = _CountingStore()
        assert not isinstance(store, HeldAssignmentsLoader)
        contexts = _Bundle(store, states=4, per_state=3).contexts()
        assert await evaluate_many_with_trail(contexts, cache=make_cache(store)) == await _one_at_a_time(
            store, contexts
        )

    async def test_the_walls_and_the_union_hold(self) -> None:
        store = _HeldStore()
        bundle = _Bundle(store, states=2, per_state=2)
        results = await evaluate_many_with_trail(bundle.contexts()[::3], cache=make_cache(store))
        by_namespace = {ctx.namespace.id: r.effective_actions for ctx, r in zip(bundle.contexts()[::3], results)}
        reached = {k: v for k, v in by_namespace.items() if v}
        # every state namespace reads, the shared one unions both states, and nothing else counts
        assert reached == {
            **{n.id: frozenset({"read"}) for n in bundle.namespaces[:4]},
            bundle.namespaces[4].id: frozenset({"read", "tool.call"}),
        }

    async def test_a_cached_pair_is_served_from_the_cache(self) -> None:
        store = _HeldStore()
        contexts = _Bundle(store, states=2, per_state=2).contexts()
        cache = make_cache(store)
        first = await evaluate_many_with_trail(contexts, cache=cache)
        store.reads.clear()
        assert await evaluate_many_with_trail(contexts, cache=cache) == first
        assert "load_assignments_held_by_groups" not in store.reads


class TestTheReadsDoNotGrowPerNamespace:
    async def _reads(self, store: _CountingStore, per_state: int) -> dict[str, int]:
        bundle = _Bundle(store, states=6, per_state=per_state)
        contexts = [EvaluationContext(namespace=n, action="read", user_id=bundle.user) for n in bundle.namespaces]
        await evaluate_many_with_trail(contexts, cache=make_cache(store))
        return store.reads

    async def test_tenfold_the_namespaces_reads_the_same(self) -> None:
        small, large = await self._reads(_HeldStore(), 5), await self._reads(_HeldStore(), 50)
        assert small == large
        # one read of what the groups hold, one of the roles, one of the groups
        assert (
            large["load_assignments_held_by_groups"],
            large["load_roles"],
            large["load_groups"],
        ) == (1, 1, 1)
        assert "load_assignments_for_groups" not in large

    async def test_a_loader_reading_per_namespace_reads_once_per_namespace_not_per_group(self) -> None:
        store = _CountingStore()
        reads = await self._reads(store, 5)
        # the states' own, the shared one and the owned one; on the other customer's namespace the
        # wall leaves the person no group, so there is nothing to read
        namespaces = 6 * 5 + 2
        assert reads["load_assignments_for_groups"] == namespaces
        assert (reads["load_roles"], reads["load_groups"]) == (1, 1)


class TestOneDecisionDoesNotReadPerGroup:
    """a two-sided ``tool.call`` decision -- a cache token's, per shareable read -- for a reader of many states."""

    async def _reads(self, states: int) -> dict[str, int]:
        store = _HeldStore()
        bundle = _Bundle(store, states=states, per_state=2)
        ctx = EvaluationContext(
            namespace=bundle.namespaces[0], action="read", user_id=bundle.user, agent_id=bundle.agent
        )
        assert await evaluate_decision(ctx, cache=make_cache(store))
        return store.reads

    async def test_tenfold_the_states_reads_the_same(self) -> None:
        assert await self._reads(5) == await self._reads(50)


class TestTheFenceStillGuardsTheBulkRead:
    async def test_an_eviction_during_the_read_keeps_the_pairs_out_and_still_answers(self) -> None:
        store = _HeldStore()
        bundle = _Bundle(store, states=2, per_state=2)
        cache = make_cache(store)
        read = store.load_assignments_held_by_groups

        async def read_then_evicted(group_ids: tuple[UUID, ...]) -> tuple[RoleAssignment, ...]:
            rows = await read(group_ids)
            cache.evict_role_assignment_row(group_ids[0])
            return rows

        store.load_assignments_held_by_groups = read_then_evicted  # type: ignore[method-assign]
        namespace = bundle.namespaces[0]
        [result] = await evaluate_many_with_trail(
            [EvaluationContext(namespace=namespace, action="read", user_id=bundle.user)], cache=cache
        )
        assert result.decision
        assert cache.get_group_namespace(GroupNamespaceKey(bundle.bundle, namespace.id)) is None
