"""answers per access shape over the real access tables: shared by two replicas and dropped by a real write.

A writer registry (the hub's admin path: real collections over Postgres, advancing each table's
generation through the epoch bucket) and two replicas (each with its own L1, access cache and shape
answers, sharing one L2 and hearing the writer's row broadcasts on one in-process bus). Answers are
real evaluations, as each shape's actor, of which namespaces a person may read.

- a shape derived on one replica is read from L2 by the other, and from L1 after;
- a grant written for a group drops, on both replicas and in L2, exactly the shapes that reach the
  group; the next answer for that shape carries the grant;
- a person's own membership row drops no shape.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

import asyncpg
import pytest
from sqlalchemy import MetaData

from threetears.agent.acl import (
    ACCESS_TABLES,
    AccessShape,
    AccessShapeAnswers,
    AclCache,
    CollectionGrantLoader,
    CollectionMembershipLoader,
    GroupCollection,
    GroupMemberCollection,
    RoleAssignmentCollection,
    RoleCollection,
    ShapeActors,
    ShapeAnswerOutcome,
    ShapeDependencies,
    bind_acl_cache_to_access_tables,
    evaluate_many_with_trail,
)
from threetears.agent.acl.types import EvaluationContext, Namespace
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient
from threetears.epoch import EpochGenerationReader, EpochGenerationSource, generation_catchup_tick
from threetears.nats import set_default_namespace

from .access_tables_ddl import ACCESS_TABLE_NAMES as _TABLES
from .access_tables_ddl import ACCESS_TABLES_DDL as _DDL

pytestmark = pytest.mark.integration

_TABLE = "access_shape_answers_it"


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace("shapegen")


@pytest.fixture
async def pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    from threetears.core.collections import init_connection

    pg: asyncpg.Pool = await asyncpg.create_pool(db_container, min_size=1, max_size=4, init=init_connection)
    try:
        async with pg.acquire() as conn:
            for table in _TABLES:
                await conn.execute(f"DROP TABLE IF EXISTS {table}")
            for ddl in _DDL:
                await conn.execute(ddl)
        yield pg
    finally:
        async with pg.acquire() as conn:
            for table in _TABLES:
                await conn.execute(f"DROP TABLE IF EXISTS {table}")
        await pg.close()


def _config() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


class _Writer:
    def __init__(self, pg: asyncpg.Pool, bus: FakeNatsClient) -> None:
        self.registry = CollectionRegistry()
        self.registry.configure(l3_pool=pg, l2_client=bus, kv_key_scope="writer")
        self.registry.set_generation_source(EpochGenerationSource(bus))
        self.groups = GroupCollection(self.registry, _config(), nats_client=bus)
        self.members = GroupMemberCollection(self.registry, _config(), nats_client=bus)
        self.roles = RoleCollection(self.registry, _config(), nats_client=bus)
        self.assignments = RoleAssignmentCollection(self.registry, _config(), nats_client=bus)

    async def add_member(self, group_id: UUID, member_id: UUID) -> None:
        entity = self.members.create(
            {
                "id": uuid.uuid4(),
                "group_id": group_id,
                "member_type": "user",
                "member_id": member_id,
                "customer_id": None,
            }
        )
        await self.members.save_entity(entity)

    async def grant(self, group_id: UUID, role_id: UUID, namespace_id: UUID) -> None:
        await self.assignments.ensure_group_role_assignment(
            group_id=group_id, role_id=role_id, scope_type="namespace", scope_id=namespace_id
        )


class _Replica:
    """a hub replica: its collections over Postgres, a followed access cache, and the shape answers."""

    def __init__(self, pg: asyncpg.Pool, bus: FakeNatsClient) -> None:
        self.bus = bus
        l1 = SQLiteBackend(db_name=f"shape-it-{uuid.uuid4().hex}")
        l1.initialize(MetaData())
        self.registry = CollectionRegistry()
        self.registry.configure(l1_backend=l1, l3_pool=pg, l2_client=bus, kv_key_scope="hub")
        members = GroupMemberCollection(self.registry, _config(), nats_client=bus)
        self.members = CollectionMembershipLoader(members)
        grants = CollectionGrantLoader(
            assignment_collection=RoleAssignmentCollection(self.registry, _config(), nats_client=bus),
            role_collection=RoleCollection(self.registry, _config(), nats_client=bus),
            group_collection=GroupCollection(self.registry, _config(), nats_client=bus),
        )
        self.actors = ShapeActors(self.members)
        self.cache = AclCache(membership_loader=self.actors, grant_loader=grants)
        self.dependencies = ShapeDependencies(self.registry, ACCESS_TABLES)
        self.answers = AccessShapeAnswers(
            self.registry,
            _config(),
            bus,
            table_name=_TABLE,
            acl_cache=self.cache,
            actors=self.actors,
            dependencies=self.dependencies,
        )

    async def start(self) -> None:
        await self.registry.start_invalidation_listener(self.bus)  # type: ignore[arg-type]
        for table in ACCESS_TABLES:
            self.registry.follow_generation(table)
        await generation_catchup_tick(self.registry, EpochGenerationReader(self.bus))
        # the access cache is bound first, as the hub binds it, so a row reaches it before the answers
        bind_acl_cache_to_access_tables(self.registry, self.cache)
        self.cache.followed_by(lambda: True)
        self.dependencies.bind()
        self.dependencies.followed_by(lambda: True)

    async def shape_of(self, person: UUID) -> AccessShape:
        return AccessShape.of(await self.members.load_for_user(person))

    async def settle(self) -> None:
        for _ in range(5):
            await asyncio.sleep(0)
        await self.answers.settled()


class _Estate:
    def __init__(self) -> None:
        self.customer = uuid.uuid4()
        self.state_a, self.state_b, self.role = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        self.ada, self.bea, self.cy = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        self.ns_a, self.ns_b = (
            Namespace(
                id=uuid.uuid4(),
                customer_id=self.customer,
                namespace_type="tool",
                owner_agent_id=None,
                owner_namespace=None,
            )
            for _ in range(2)
        )

    async def seed(self, writer: _Writer) -> None:
        for group, name in ((self.state_a, "state-a"), (self.state_b, "state-b")):
            await writer.groups.save_entity(
                writer.groups.create({"group_id": group, "customer_id": self.customer, "name": name})
            )
        role = writer.roles.create(
            {"role_id": self.role, "name": "Reader", "description": "", "permissions": json.dumps({"tool": ["read"]})}
        )
        await writer.roles.save_entity(role)
        await writer.add_member(self.state_a, self.ada)
        await writer.add_member(self.state_a, self.bea)
        await writer.add_member(self.state_b, self.cy)
        await writer.grant(self.state_a, self.role, self.ns_a.id)
        await writer.grant(self.state_b, self.role, self.ns_b.id)


def _visible(replica: _Replica, estate: _Estate) -> Any:
    async def compute(actor: UUID) -> str:
        namespaces = (estate.ns_a, estate.ns_b)
        contexts = [EvaluationContext(namespace=ns, action="read", user_id=actor) for ns in namespaces]
        results = await evaluate_many_with_trail(contexts, cache=replica.cache)
        return json.dumps(sorted(f"{ns.id}" for ns, result in zip(namespaces, results, strict=True) if result.decision))

    return compute


async def _ask(replica: _Replica, estate: _Estate, shape: AccessShape) -> Any:
    return await replica.answers.answer(
        kind="read",
        customer_id=estate.customer,
        shape=shape,
        context="it",
        agent_ids=[],
        compute=_visible(replica, estate),
    )


async def _bucket(bus: FakeNatsClient) -> FakeKvBucket:
    bucket: FakeKvBucket = await bus.kv_bucket(name="collections")
    return bucket


@pytest.fixture
async def estate(pool: asyncpg.Pool) -> AsyncIterator[tuple[_Writer, _Replica, _Replica, _Estate]]:
    bus = FakeNatsClient()
    writer = _Writer(pool, bus)
    for table in ACCESS_TABLES:
        await writer.registry.generation_source.advance(table)  # type: ignore[union-attr]
    one, two = _Replica(pool, bus), _Replica(pool, bus)
    await one.start()
    await two.start()
    seeded = _Estate()
    await seeded.seed(writer)
    yield writer, one, two, seeded


class TestTwoReplicasOverTheRealTables:
    async def test_one_derives_and_the_other_reads_it(
        self, estate: tuple[_Writer, _Replica, _Replica, _Estate]
    ) -> None:
        _writer, one, two, seeded = estate
        ada = await _ask(one, seeded, await one.shape_of(seeded.ada))
        bea = await _ask(two, seeded, await two.shape_of(seeded.bea))
        assert ada.outcome is ShapeAnswerOutcome.DERIVED
        assert bea.outcome is ShapeAnswerOutcome.L2
        assert json.loads(ada.body) == [f"{seeded.ns_a.id}"] == json.loads(bea.body)
        assert (await _ask(two, seeded, await two.shape_of(seeded.bea))).outcome is ShapeAnswerOutcome.L1

    async def test_a_grant_drops_exactly_the_shapes_reaching_its_group_everywhere(
        self, estate: tuple[_Writer, _Replica, _Replica, _Estate]
    ) -> None:
        writer, one, two, seeded = estate
        shape_a = await one.shape_of(seeded.ada)
        shape_b = await one.shape_of(seeded.cy)
        for replica in (one, two):
            await _ask(replica, seeded, shape_a)
            await _ask(replica, seeded, shape_b)
        bucket = await _bucket(one.bus)
        key_a = one.answers.key_of("read", seeded.customer, shape_a)
        key_b = one.answers.key_of("read", seeded.customer, shape_b)
        assert one.answers.l2_key(key_a) in bucket.keys()

        await writer.grant(seeded.state_a, seeded.role, seeded.ns_b.id)
        await one.settle()
        await two.settle()

        for replica in (one, two):
            assert replica.answers.held_row(key_a) is None
            assert replica.answers.held_row(key_b) is not None
        assert one.answers.l2_key(key_a) not in bucket.keys()
        assert one.answers.l2_key(key_b) in bucket.keys()
        again = await _ask(two, seeded, await two.shape_of(seeded.bea))
        assert again.outcome is ShapeAnswerOutcome.DERIVED
        assert json.loads(again.body) == sorted([f"{seeded.ns_a.id}", f"{seeded.ns_b.id}"])
        assert (await _ask(one, seeded, shape_b)).outcome is ShapeAnswerOutcome.L1

    async def test_a_persons_own_membership_drops_no_shape(
        self, estate: tuple[_Writer, _Replica, _Replica, _Estate]
    ) -> None:
        writer, one, _two, seeded = estate
        shape_a = await one.shape_of(seeded.ada)
        await _ask(one, seeded, shape_a)
        await writer.add_member(seeded.state_b, seeded.ada)
        await one.settle()
        assert one.answers.held == 1
        moved = await one.shape_of(seeded.ada)
        assert moved != shape_a
        assert (await _ask(one, seeded, shape_a)).outcome is ShapeAnswerOutcome.L1
        both = await _ask(one, seeded, moved)
        assert json.loads(both.body) == sorted([f"{seeded.ns_a.id}", f"{seeded.ns_b.id}"])
