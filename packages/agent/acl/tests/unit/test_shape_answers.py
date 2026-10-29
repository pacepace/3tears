"""answers per access shape: from L1 when hot, shared through L2, derived once, and dropped only by what they reach.

Two replicas are two collections on two registries, each with its own L1 and access cache, sharing one
L2 bucket under one key scope, as the hub's replicas do. A heard change is a row broadcast handed to
each registry's derived caches, as its invalidation listener hands it.

The contract this pins:

- two people with one shape share one derivation, and a hot shape is answered from L1 with no L2 and
  no database read; a replica that did not derive it reads it from L2 once, then from its own L1;
- one derivation for concurrent asks, in one process and across replicas;
- a change reaching a shape's footprint drops it from L1 and L2 on every replica that hears it, and
  nothing else; a person's own membership row drops nothing;
- an answer derived while a change landed is returned and kept nowhere;
- an answer a replica put in L2 before it heard a change is never served by one that has, and a
  replica trusts nothing stamped before it began following;
- nothing is served or kept while the tables are not followed;
- an answer derived as a shape's actor is the answer for every person with the shape.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID, uuid4, uuid7

import pytest
from sqlalchemy import MetaData

from threetears.agent.acl import (
    ACCESS_TABLES,
    AccessShape,
    AccessShapeAnswers,
    ShapeActors,
    ShapeAnswerOutcome,
    ShapeDependencies,
    evaluate_many_with_trail,
)
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
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import tier_reads_total
from threetears.core.collections.registry import CacheInvalidationMessage, CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient

from .fake_loaders import FakeBatchingStore

_SCOPE = "hub-shapes"
_TABLE = "access_shape_answers"
_EXTRA = "rest_shares"
_TABLES = (*ACCESS_TABLES, _EXTRA)
_KIND = "cache-token"
_CONTEXT = "routes-1"


class _Store(FakeBatchingStore):
    """the database: lists held assignments in one read, and counts every read."""

    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    async def load_for_user(self, user_id: UUID) -> tuple[GroupMembership, ...]:
        self.reads += 1
        return await super().load_for_user(user_id)

    async def load_for_agent(self, agent_id: UUID) -> tuple[GroupMembership, ...]:
        self.reads += 1
        return await super().load_for_agent(agent_id)

    async def load_for_groups(self, group_ids: tuple[UUID, ...]) -> tuple[GroupMembership, ...]:
        self.reads += 1
        return await super().load_for_groups(group_ids)

    async def load_assignments_held_by_groups(self, group_ids: tuple[UUID, ...]) -> tuple[RoleAssignment, ...]:
        self.reads += 1
        ids = set(group_ids)
        return tuple(a for a in self.assignments if a.group_id in ids)

    async def load_roles(self, role_ids: tuple[UUID, ...]) -> dict[UUID, Role]:
        self.reads += 1
        return await super().load_roles(role_ids)

    async def load_groups(self, group_ids: tuple[UUID, ...]) -> dict[UUID, object]:
        self.reads += 1
        return await super().load_groups(group_ids)


class _Estate:
    """readers in a bundle nested in two states, a reader in a third state, and a signing agent."""

    def __init__(self, store: _Store) -> None:
        self.customer = uuid4()
        self.reader, self.twin, self.other, self.agent = uuid7(), uuid7(), uuid7(), uuid4()
        self.role = store.add_role(Role(uuid4(), "Reader", {"tool": frozenset({"read"})}, is_built_in=True)).id
        self.other_role = store.add_role(Role(uuid4(), "Other", {"tool": frozenset({"read"})}, is_built_in=True)).id
        self.bundle = store.add_group(Group(uuid4(), "bundle", self.customer)).id
        self.states = [store.add_group(Group(uuid4(), f"state-{i}", self.customer)).id for i in range(3)]
        self.signers = store.add_group(Group(uuid4(), "signers", None)).id
        for person in (self.reader, self.twin):
            store.add_membership(GroupMembership(self.bundle, MemberType.USER, person, self.customer))
        store.add_membership(GroupMembership(self.states[2], MemberType.USER, self.other, self.customer))
        for state in self.states[:2]:
            store.add_membership(GroupMembership(state, MemberType.GROUP, self.bundle, self.customer))
        store.add_membership(GroupMembership(self.signers, MemberType.AGENT, self.agent, None))
        self.namespaces = [
            Namespace(
                id=uuid4(), customer_id=self.customer, namespace_type="tool", owner_agent_id=None, owner_namespace=None
            )
            for _ in range(3)
        ]
        for index, namespace in enumerate(self.namespaces):
            role = self.other_role if index == 2 else self.role
            for group in (self.states[index], self.signers):
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

    async def shape(self, store: _Store, person: UUID) -> AccessShape:
        return AccessShape.of(await store.load_for_user(person))


class _Replica:
    def __init__(self, registry: CollectionRegistry, answers: AccessShapeAnswers, store: _Store) -> None:
        self.registry = registry
        self.answers = answers
        self.store = store
        self.followed = True

    def hear(self, message: CacheInvalidationMessage) -> None:
        self.registry.tell_derived_caches(message)

    async def settle(self) -> None:
        for _ in range(5):
            await asyncio.sleep(0)
        await self.answers.settled()


def _sighted(registry: CollectionRegistry, token: str = "inc:0") -> None:
    for table in _TABLES:
        registry.generation_marks.follow(table)
        registry.generation_marks.settle(table, token)


async def _replica(bus: FakeNatsClient, store: _Store, *, token: str = "inc:0") -> _Replica:
    l1 = SQLiteBackend(db_name=f"shapes-l1-{uuid.uuid4().hex}")
    l1.initialize(MetaData())
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l2_client=bus, l3_pool=None, kv_key_scope=_SCOPE)  # type: ignore[arg-type]
    actors = ShapeActors(store)
    dependencies = ShapeDependencies(registry, _TABLES)
    answers = AccessShapeAnswers(
        registry,
        DefaultCoreConfig(),
        bus,
        table_name=_TABLE,
        acl_cache=AclCache(membership_loader=actors, grant_loader=store),
        actors=actors,
        dependencies=dependencies,
    )
    await registry.start_invalidation_listener(bus)  # type: ignore[arg-type]
    _sighted(registry, token)
    dependencies.bind()
    replica = _Replica(registry, answers, store)
    dependencies.followed_by(lambda: replica.followed)
    return replica


class _Computer:
    """what a consumer derives: counts its calls, and may hold until released."""

    def __init__(self, text: str = '{"routes": ["/a"]}', *, hold: asyncio.Event | None = None) -> None:
        self.text = text
        self.calls = 0
        self.actors: list[UUID] = []
        self.started = asyncio.Event()
        self._hold = hold

    async def __call__(self, actor: UUID) -> str:
        self.calls += 1
        self.actors.append(actor)
        self.started.set()
        if self._hold is not None:
            await self._hold.wait()
        return self.text


async def _ask(
    replica: _Replica,
    estate: _Estate,
    shape: AccessShape,
    compute: Callable[[UUID], Awaitable[str | None]],
    *,
    context: str = _CONTEXT,
) -> Any:
    return await replica.answers.answer(
        kind=_KIND, customer_id=estate.customer, shape=shape, context=context, agent_ids=[estate.agent], compute=compute
    )


def _row(
    table: str, *, generation: str | None, ids: list[str] | None = None, **columns: str
) -> CacheInvalidationMessage:
    ids = ids if ids is not None else [f"{uuid4()}"]
    return CacheInvalidationMessage(
        table=table, ids=ids, columns=columns, generation=generation, bump_rows=1 if generation else None
    )


def _assignment_row(group: UUID, generation: str | None = "inc:1") -> CacheInvalidationMessage:
    return CacheInvalidationMessage(
        table="role_assignments",
        ids=[f"{uuid4()}"],
        columns={"group_id": f"{group}"},
        generation=generation,
        bump_rows=1 if generation else None,
    )


async def _bucket(bus: FakeNatsClient) -> FakeKvBucket:
    bucket: FakeKvBucket = await bus.kv_bucket(name="collections")
    return bucket


def _l2_keys(bucket: FakeKvBucket) -> list[str]:
    return [key for key in bucket.keys() if f".{_TABLE}." in key]


def _l2_counted(bucket: FakeKvBucket) -> list[str]:
    calls: list[str] = []
    for name in ("get", "get_entry", "get_latest", "create", "update", "put", "delete"):
        original = getattr(bucket, name)

        def counted(*args: Any, _name: str = name, _original: Any = original, **kwargs: Any) -> Any:
            calls.append(_name)
            return _original(*args, **kwargs)

        setattr(bucket, name, counted)
    return calls


@pytest.fixture
def bus() -> FakeNatsClient:
    return FakeNatsClient()


class TestOneDerivationServesEveryoneWithTheShape:
    async def test_two_people_with_one_shape_share_one_derivation(self, bus: FakeNatsClient) -> None:
        store = _Store()
        estate = _Estate(store)
        replica = await _replica(bus, store)
        compute = _Computer()
        first = await _ask(replica, estate, await estate.shape(store, estate.reader), compute)
        second = await _ask(replica, estate, await estate.shape(store, estate.twin), compute)
        assert first.outcome is ShapeAnswerOutcome.DERIVED
        assert second.outcome is ShapeAnswerOutcome.L1
        assert first.body == second.body
        assert compute.calls == 1

    async def test_a_hot_shape_is_served_from_l1_with_no_l2_or_database_read(self, bus: FakeNatsClient) -> None:
        store = _Store()
        estate = _Estate(store)
        replica = await _replica(bus, store)
        shape = await estate.shape(store, estate.reader)
        await _ask(replica, estate, shape, _Computer())
        calls = _l2_counted(await _bucket(bus))
        reads_before, tiers_before = store.reads, tier_reads_total().get(_TABLE, {}).get("l1", 0)
        compute = _Computer()
        given = await _ask(replica, estate, shape, compute)
        assert given.outcome is ShapeAnswerOutcome.L1
        assert calls == []
        assert store.reads == reads_before
        assert compute.calls == 0
        assert tier_reads_total()[_TABLE]["l1"] == tiers_before + 1

    async def test_another_replica_reads_it_from_l2_once_then_from_its_own_l1(self, bus: FakeNatsClient) -> None:
        store = _Store()
        estate = _Estate(store)
        one, two = await _replica(bus, store), await _replica(bus, store)
        shape = await estate.shape(store, estate.reader)
        await _ask(one, estate, shape, _Computer())
        compute = _Computer()
        assert (await _ask(two, estate, shape, compute)).outcome is ShapeAnswerOutcome.L2
        assert (await _ask(two, estate, shape, compute)).outcome is ShapeAnswerOutcome.L1
        assert compute.calls == 0

    async def test_a_different_shape_is_derived_on_its_own(self, bus: FakeNatsClient) -> None:
        store = _Store()
        estate = _Estate(store)
        replica = await _replica(bus, store)
        compute = _Computer()
        await _ask(replica, estate, await estate.shape(store, estate.reader), compute)
        await _ask(replica, estate, await estate.shape(store, estate.other), compute)
        assert compute.calls == 2
        assert compute.actors[0] != compute.actors[1]

    async def test_another_context_is_derived_again(self, bus: FakeNatsClient) -> None:
        store = _Store()
        estate = _Estate(store)
        replica = await _replica(bus, store)
        shape = await estate.shape(store, estate.reader)
        compute = _Computer()
        await _ask(replica, estate, shape, compute)
        given = await _ask(replica, estate, shape, compute, context="routes-2")
        assert given.outcome is ShapeAnswerOutcome.DERIVED
        assert compute.calls == 2

    async def test_nothing_answered_is_kept_nowhere(self, bus: FakeNatsClient) -> None:
        store = _Store()
        estate = _Estate(store)
        replica = await _replica(bus, store)

        async def nothing(_actor: UUID) -> str | None:
            return None

        assert await _ask(replica, estate, await estate.shape(store, estate.reader), nothing) is None
        assert replica.answers.held == 0
        assert _l2_keys(await _bucket(bus)) == []


class TestOneDerivationForConcurrentAsks:
    async def test_in_one_process(self, bus: FakeNatsClient) -> None:
        store = _Store()
        estate = _Estate(store)
        replica = await _replica(bus, store)
        shape = await estate.shape(store, estate.reader)
        release = asyncio.Event()
        compute = _Computer(hold=release)
        asks = [asyncio.create_task(_ask(replica, estate, shape, compute)) for _ in range(10)]
        await compute.started.wait()
        release.set()
        given = await asyncio.gather(*asks)
        assert compute.calls == 1
        assert {answer.body for answer in given} == {compute.text}

    async def test_across_replicas(self, bus: FakeNatsClient) -> None:
        store = _Store()
        estate = _Estate(store)
        one, two = await _replica(bus, store), await _replica(bus, store)
        shape = await estate.shape(store, estate.reader)
        release = asyncio.Event()
        compute = _Computer(hold=release)
        first = asyncio.create_task(_ask(one, estate, shape, compute))
        await compute.started.wait()
        second = asyncio.create_task(_ask(two, estate, shape, compute))
        await asyncio.sleep(0.05)
        release.set()
        given = await asyncio.gather(first, second)
        assert compute.calls == 1
        assert given[1].outcome is ShapeAnswerOutcome.L2


class TestAChangeDropsOnlyWhatItReaches:
    async def test_a_group_change_empties_its_shapes_from_every_tier_on_every_replica(
        self, bus: FakeNatsClient
    ) -> None:
        store = _Store()
        estate = _Estate(store)
        one, two = await _replica(bus, store), await _replica(bus, store)
        reached = await estate.shape(store, estate.reader)
        untouched = await estate.shape(store, estate.other)
        for replica in (one, two):
            await _ask(replica, estate, reached, _Computer())
            await _ask(replica, estate, untouched, _Computer())
        bucket = await _bucket(bus)
        assert len(_l2_keys(bucket)) == 2
        assert one.answers.held == two.answers.held == 2
        change = _assignment_row(estate.states[0])
        for replica in (one, two):
            replica.hear(change)
        for replica in (one, two):
            await replica.settle()
        assert one.answers.held == two.answers.held == 1
        assert _l2_keys(bucket) == [one.answers.l2_key(one.answers.key_of(_KIND, estate.customer, untouched))]
        for replica in (one, two):
            assert replica.answers.held_row(replica.answers.key_of(_KIND, estate.customer, reached)) is None
            kept = await _ask(replica, estate, untouched, _Computer())
            assert kept.outcome is ShapeAnswerOutcome.L1
            again = _Computer()
            assert (await _ask(replica, estate, reached, again)).outcome in (
                ShapeAnswerOutcome.DERIVED,
                ShapeAnswerOutcome.L2,
            )

    async def test_a_role_change_drops_only_the_shapes_holding_it(self, bus: FakeNatsClient) -> None:
        store = _Store()
        estate = _Estate(store)
        replica = await _replica(bus, store)
        reader = await estate.shape(store, estate.reader)
        other = await estate.shape(store, estate.other)
        await _ask(replica, estate, reader, _Computer())
        await _ask(replica, estate, other, _Computer())
        # the agent's group holds both roles, so the signer side reaches both; a role neither holds
        replica.hear(_row("roles", generation="inc:1", ids=[f"{uuid4()}"]))
        assert replica.answers.held == 2
        replica.hear(_row("roles", generation="inc:2", ids=[f"{estate.other_role}"]))
        assert replica.answers.held == 0

    async def test_a_persons_own_membership_row_drops_nothing(self, bus: FakeNatsClient) -> None:
        store = _Store()
        estate = _Estate(store)
        replica = await _replica(bus, store)
        shape = await estate.shape(store, estate.reader)
        await _ask(replica, estate, shape, _Computer())
        fence = replica.answers.dependencies.fence()
        replica.hear(_row("group_members", generation="inc:1", member_type="user", member_id=f"{estate.reader}"))
        assert replica.answers.held == 1
        assert replica.answers.dependencies.fence() == fence
        assert (await _ask(replica, estate, shape, _Computer())).outcome is ShapeAnswerOutcome.L1

    async def test_a_whole_table_change_drops_every_shape(self, bus: FakeNatsClient) -> None:
        store = _Store()
        estate = _Estate(store)
        replica = await _replica(bus, store)
        await _ask(replica, estate, await estate.shape(store, estate.reader), _Computer())
        replica.hear(_row(_EXTRA, generation="inc:1"))
        assert replica.answers.held == 0

    async def test_a_dropped_table_drops_every_shape(self, bus: FakeNatsClient) -> None:
        store = _Store()
        estate = _Estate(store)
        replica = await _replica(bus, store)
        shape = await estate.shape(store, estate.reader)
        await _ask(replica, estate, shape, _Computer())
        replica.registry.settle_generation("groups", "inc:9")
        assert replica.answers.held == 0
        compute = _Computer()
        assert (await _ask(replica, estate, shape, compute)).outcome is ShapeAnswerOutcome.DERIVED
        assert compute.calls == 1


class TestADropThatMovedNoMark:
    async def test_trusts_nothing_until_the_table_applies_a_later_generation(self, bus: FakeNatsClient) -> None:
        store = _Store()
        estate = _Estate(store)
        replica = await _replica(bus, store)
        shape = await estate.shape(store, estate.reader)
        await _ask(replica, estate, shape, _Computer())
        # a table changed with no row to say what (a truncate, a bulk load): dropped, no mark moved
        replica.registry.drop_table("groups", reason="changed with no row-level mark")
        compute = _Computer()
        assert (await _ask(replica, estate, shape, compute)).outcome is ShapeAnswerOutcome.UNCACHED
        assert compute.calls == 1
        replica.hear(_row("groups", generation="inc:1", ids=["platform", f"{uuid4()}"]))
        assert (await _ask(replica, estate, shape, compute)).outcome is ShapeAnswerOutcome.DERIVED
        assert (await _ask(replica, estate, shape, compute)).outcome is ShapeAnswerOutcome.L1


class TestTheReadFence:
    async def test_an_answer_derived_while_a_change_landed_is_kept_nowhere(self, bus: FakeNatsClient) -> None:
        store = _Store()
        estate = _Estate(store)
        replica = await _replica(bus, store)
        shape = await estate.shape(store, estate.reader)
        release = asyncio.Event()
        compute = _Computer(hold=release)
        ask = asyncio.create_task(_ask(replica, estate, shape, compute))
        await compute.started.wait()
        replica.hear(_assignment_row(estate.states[2]))
        release.set()
        given = await ask
        assert given.outcome is ShapeAnswerOutcome.UNCACHED
        assert given.body == compute.text
        assert replica.answers.held == 0
        assert _l2_keys(await _bucket(bus)) == []


class TestAStaleAnswerIsNeverServedByAReplicaThatHeardTheChange:
    async def test_one_put_before_its_replica_heard_the_change(self, bus: FakeNatsClient) -> None:
        store = _Store()
        estate = _Estate(store)
        late, heard = await _replica(bus, store), await _replica(bus, store)
        shape = await estate.shape(store, estate.reader)
        heard.hear(_assignment_row(estate.states[0]))
        # the late replica has not heard it: what it derives is stamped before the change
        await _ask(late, estate, shape, _Computer('{"stale": true}'))
        fresh = _Computer('{"fresh": true}')
        given = await _ask(heard, estate, shape, fresh)
        assert given.outcome is ShapeAnswerOutcome.DERIVED
        assert given.body == fresh.text
        late.hear(_assignment_row(estate.states[0]))
        assert late.answers.held == 0

    async def test_a_replica_trusts_nothing_stamped_before_it_began_following(self, bus: FakeNatsClient) -> None:
        store = _Store()
        estate = _Estate(store)
        early = await _replica(bus, store)
        shape = await estate.shape(store, estate.reader)
        await _ask(early, estate, shape, _Computer())
        joined = await _replica(bus, store, token="inc:5")
        compute = _Computer()
        given = await _ask(joined, estate, shape, compute)
        assert given.outcome is ShapeAnswerOutcome.DERIVED
        assert compute.calls == 1
        stamp = json.loads(joined.answers.held_row(joined.answers.key_of(_KIND, estate.customer, shape))["stamp"])
        assert set(stamp.values()) == {"inc:5"}

    async def test_a_row_naming_no_generation_stays_stale_until_a_later_one_is_applied(
        self, bus: FakeNatsClient
    ) -> None:
        store = _Store()
        estate = _Estate(store)
        replica = await _replica(bus, store)
        shape = await estate.shape(store, estate.reader)
        replica.hear(_assignment_row(estate.states[0], generation=None))
        first = _Computer()
        assert (await _ask(replica, estate, shape, first)).outcome is ShapeAnswerOutcome.UNCACHED
        assert (await _ask(replica, estate, shape, first)).outcome is ShapeAnswerOutcome.UNCACHED
        assert replica.answers.held == 0
        replica.hear(_assignment_row(estate.states[2], generation="inc:1"))
        assert (await _ask(replica, estate, shape, first)).outcome is ShapeAnswerOutcome.DERIVED
        assert (await _ask(replica, estate, shape, first)).outcome is ShapeAnswerOutcome.L1


class TestUntrusted:
    async def test_nothing_is_served_or_kept_while_not_followed(self, bus: FakeNatsClient) -> None:
        store = _Store()
        estate = _Estate(store)
        replica = await _replica(bus, store)
        replica.followed = False
        shape = await estate.shape(store, estate.reader)
        compute = _Computer()
        assert (await _ask(replica, estate, shape, compute)).outcome is ShapeAnswerOutcome.UNCACHED
        assert (await _ask(replica, estate, shape, compute)).outcome is ShapeAnswerOutcome.UNCACHED
        assert compute.calls == 2
        assert replica.answers.held == 0
        assert _l2_keys(await _bucket(bus)) == []

    async def test_nothing_held_before_a_watch_failed_is_served_once_it_runs_again(self, bus: FakeNatsClient) -> None:
        store = _Store()
        estate = _Estate(store)
        replica = await _replica(bus, store)
        shape = await estate.shape(store, estate.reader)
        await _ask(replica, estate, shape, _Computer())
        replica.followed = False
        await _ask(replica, estate, shape, _Computer())
        assert replica.answers.held == 0
        replica.followed = True
        compute = _Computer()
        # L2 still holds it, checked against its stamp: nothing was missed, so it is current
        assert (await _ask(replica, estate, shape, compute)).outcome is ShapeAnswerOutcome.L2
        assert compute.calls == 0

    async def test_losing_the_follower_drops_what_is_held(self, bus: FakeNatsClient) -> None:
        store = _Store()
        estate = _Estate(store)
        replica = await _replica(bus, store)
        await _ask(replica, estate, await estate.shape(store, estate.reader), _Computer())
        replica.answers.dependencies.followed_by(None)
        assert replica.answers.held == 0

    async def test_the_access_cache_must_read_through_the_actors(self, bus: FakeNatsClient) -> None:
        store = _Store()
        registry = CollectionRegistry()
        with pytest.raises(ValueError):
            AccessShapeAnswers(
                registry,
                DefaultCoreConfig(),
                bus,
                table_name=_TABLE,
                acl_cache=AclCache(membership_loader=store, grant_loader=store),
                actors=ShapeActors(store),
                dependencies=ShapeDependencies(registry, _TABLES),
            )


class TestTheAnswerIsEveryonesWithTheShape:
    async def test_derived_as_the_actor_it_equals_each_persons_own(self, bus: FakeNatsClient) -> None:
        store = _Store()
        estate = _Estate(store)
        replica = await _replica(bus, store)
        cache = replica.answers.acl_cache

        async def visible(as_user: UUID, evaluation_cache: AclCache) -> str:
            contexts = [
                EvaluationContext(namespace=namespace, action="read", user_id=as_user, agent_id=estate.agent)
                for namespace in estate.namespaces
            ]
            results = await evaluate_many_with_trail(contexts, cache=evaluation_cache)
            return json.dumps(sorted(f"{ctx.namespace.id}" for ctx, r in zip(contexts, results) if r.decision))

        async def as_actor(actor: UUID) -> str:
            return await visible(actor, cache)

        shape = await estate.shape(store, estate.reader)
        given = await _ask(replica, estate, shape, as_actor)
        for person in (estate.reader, estate.twin):
            assert given.body == await visible(person, AclCache(membership_loader=store, grant_loader=store))
        assert json.loads(given.body)
        other = await _ask(replica, estate, await estate.shape(store, estate.other), as_actor)
        assert other.body == await visible(estate.other, AclCache(membership_loader=store, grant_loader=store))
        assert other.body != given.body


def _reached(message: CacheInvalidationMessage) -> list[frozenset[str] | None]:
    """what one row reaches, as the answers' dependencies tell their listeners."""
    registry = CollectionRegistry()
    _sighted(registry)
    dependencies = ShapeDependencies(registry, _TABLES)
    dependencies.bind()
    heard: list[frozenset[str] | None] = []
    dependencies.add_listener(heard.append)
    registry.tell_derived_caches(message)
    return heard


_PERSON, _AGENT, _CHILD = (UUID(f"00000000-0000-4000-8000-00000000000{n}") for n in (1, 2, 3))


class TestWhatARowReaches:
    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            (_row("group_members", generation=None, member_type="user", member_id=f"{_PERSON}"), []),
            (
                _row("group_members", generation=None, member_type="agent", member_id=f"{_AGENT}"),
                [frozenset({f"a:{_AGENT.hex}"})],
            ),
            (
                _row("group_members", generation=None, member_type="group", member_id=f"{_CHILD}"),
                [frozenset({f"g:{_CHILD.hex}"})],
            ),
            (_row("group_members", generation=None), [frozenset({"t:group_members"})]),
            (
                _row("role_assignments", generation=None, group_id=f"{_CHILD}"),
                [frozenset({f"g:{_CHILD.hex}"})],
            ),
            (_row("role_assignments", generation=None), [frozenset({"t:role_assignments"})]),
            (_row("roles", generation=None, ids=[f"{_AGENT}"]), [frozenset({f"r:{_AGENT.hex}", "r:*"})]),
            (_row("groups", generation=None, ids=["platform", f"{_CHILD}"]), [frozenset({f"g:{_CHILD.hex}"})]),
            (_row("namespaces", generation=None, ids=["platform", f"{_PERSON}"]), [frozenset({"t:namespaces"})]),
            (_row(_EXTRA, generation=None), [frozenset({f"t:{_EXTRA}"})]),
        ],
    )
    def test_each_table(self, message: CacheInvalidationMessage, expected: list[frozenset[str] | None]) -> None:
        assert _reached(message) == expected
