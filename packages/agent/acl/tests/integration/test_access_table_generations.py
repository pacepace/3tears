"""the access tables carry write generations, and an AclCache on another registry is evicted row by row.

A writer registry (the hub's shape: Postgres directly, the epoch bucket's write) and a follower
registry (a pod's shape: listening, following, an :class:`AclCache` bound to the access tables and
no ``acl.*`` subscription anywhere) share one in-process bus and epoch bucket. The writes go through
the real collections against real Postgres. The contract this pins (epoch-task-06, stage 3):

- each access table's write advances its generation, and its row broadcast names what the row
  reaches: a membership row its member, an assignment row its group;
- a heard row evicts exactly the entries it reaches and leaves every other entry;
- the raw-SQL writes of :class:`RoleAssignmentCollection` (the ensure and the revoke) are announced
  with their advance, and a revoke that matches nothing advances nothing;
- a missed broadcast is caught by the follower, by a pass or by the key watcher, and empties only
  the layer derived from that table.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import asyncpg
import pytest

from threetears.agent.acl import (
    ActorMembershipKey,
    AclCache,
    GroupCollection,
    GroupMemberCollection,
    GroupNamespaceKey,
    RoleAssignmentCollection,
    NamespaceCollection,
    RoleCollection,
    bind_acl_cache_to_access_tables,
)
from threetears.agent.acl import ACCESS_TABLES
from threetears.agent.acl.generation_follow import AccessTableFollower, follow_access_tables
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeNatsClient
from threetears.epoch import EpochGenerationReader, EpochGenerationSource, generation_catchup_tick, generation_kv_key
from threetears.nats import set_default_namespace

pytestmark = pytest.mark.integration

_DDL = (
    """
    CREATE TABLE groups (
        row_scope varchar(8) NOT NULL,
        group_id uuid NOT NULL,
        customer_id uuid,
        name varchar(255) NOT NULL,
        description text,
        date_created timestamptz NOT NULL DEFAULT now(),
        date_updated timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (row_scope, group_id)
    )
    """,
    """
    CREATE TABLE group_members (
        id uuid NOT NULL,
        group_id uuid NOT NULL,
        member_type varchar(10) NOT NULL,
        member_id uuid NOT NULL,
        customer_id uuid,
        date_added timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (group_id, id)
    )
    """,
    """
    CREATE TABLE roles (
        role_id uuid PRIMARY KEY,
        name varchar(255) NOT NULL,
        description text NOT NULL DEFAULT '',
        permissions jsonb NOT NULL DEFAULT '{}',
        customer_id uuid,
        is_builtin boolean NOT NULL DEFAULT false,
        date_created timestamptz NOT NULL DEFAULT now(),
        date_updated timestamptz NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE role_assignments (
        row_scope varchar(8) NOT NULL,
        assignment_id uuid NOT NULL,
        role_id uuid NOT NULL,
        group_id uuid NOT NULL,
        scope_type varchar(16) NOT NULL,
        scope_namespace_id uuid,
        scope_namespace_type varchar(255),
        scope_customer_id uuid,
        scope_namespace_name varchar(255),
        granted_by uuid,
        date_granted timestamptz NOT NULL DEFAULT now(),
        managed_by varchar(32) NOT NULL DEFAULT 'manual',
        PRIMARY KEY (row_scope, assignment_id)
    )
    """,
    """
    CREATE TABLE namespaces (
        row_scope varchar(8) NOT NULL,
        namespace_id uuid NOT NULL UNIQUE,
        name varchar(255) NOT NULL,
        namespace_type varchar(20) NOT NULL,
        owner_agent_id uuid,
        owner_namespace varchar(255),
        customer_id uuid,
        schema_name varchar(100),
        metadata jsonb DEFAULT '{}'::jsonb,
        tool_eligible boolean NOT NULL DEFAULT true,
        skill_eligible boolean NOT NULL DEFAULT false,
        face_api boolean NOT NULL DEFAULT false,
        face_mcp boolean NOT NULL DEFAULT false,
        face_platform_tool boolean NOT NULL DEFAULT true,
        face_rest boolean NOT NULL DEFAULT false,
        face_rest_declaration jsonb,
        date_created timestamptz NOT NULL,
        date_updated timestamptz NOT NULL,
        PRIMARY KEY (row_scope, namespace_id)
    )
    """,
)
_TABLES = ("role_assignments", "roles", "group_members", "groups", "namespaces")


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace("aclgen")


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


class _Lossy(FakeNatsClient):
    """a bus that loses every message published while it is deaf."""

    def __init__(self) -> None:
        super().__init__()
        self.deaf = False

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:
        if self.deaf:
            return
        await super().publish(subject=subject, message=message, reply_to=reply_to)


def _config() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


class _Writer:
    """the hub's shape: Postgres directly, advancing through the epoch bucket."""

    def __init__(self, pg: asyncpg.Pool, bus: _Lossy) -> None:
        self.registry = CollectionRegistry()
        self.registry.configure(l3_pool=pg, l2_client=bus, kv_key_scope="hub")
        self.registry.set_generation_source(EpochGenerationSource(bus))
        self.groups = GroupCollection(self.registry, _config(), nats_client=bus)
        self.members = GroupMemberCollection(self.registry, _config(), nats_client=bus)
        self.roles = RoleCollection(self.registry, _config(), nats_client=bus)
        self.assignments = RoleAssignmentCollection(self.registry, _config(), nats_client=bus)
        self.namespaces = NamespaceCollection(self.registry, _config(), nats_client=bus)

    async def add_member(self, group_id: uuid.UUID, member_type: str, member_id: uuid.UUID) -> uuid.UUID:
        row_id = uuid.uuid4()
        entity = self.members.create(
            {
                "id": row_id,
                "group_id": group_id,
                "member_type": member_type,
                "member_id": member_id,
                "customer_id": None,
            }
        )
        await self.members.save_entity(entity)
        return row_id


class _Follower:
    """a pod's shape: listening, following the access tables, an AclCache bound to them."""

    def __init__(self, bus: _Lossy) -> None:
        self.bus = bus
        self.registry = CollectionRegistry()
        self.registry.configure(l2_client=bus, kv_key_scope="pod")
        self.reader = EpochGenerationReader(bus)
        loader: Any = object()
        self.cache = AclCache(membership_loader=loader, grant_loader=loader)
        self.unbind = bind_acl_cache_to_access_tables(self.registry, self.cache)

    async def start(self) -> None:
        await self.registry.start_invalidation_listener(self.bus)  # type: ignore[arg-type]
        for table in ACCESS_TABLES:
            self.registry.follow_generation(table)
        await generation_catchup_tick(self.registry, self.reader)

    def seed(
        self, *actors: tuple[str, uuid.UUID], groups: tuple[tuple[uuid.UUID, uuid.UUID, frozenset[uuid.UUID]], ...] = ()
    ) -> None:
        for kind, actor_id in actors:
            self.cache.put_membership(ActorMembershipKey(kind, actor_id), ())
        for group_id, namespace_id, role_ids in groups:
            self.cache.put_group_namespace(
                GroupNamespaceKey(group_id, namespace_id), frozenset({"read"}), (), role_ids=role_ids
            )

    def holds(self, kind: str, actor_id: uuid.UUID) -> bool:
        return self.cache.get_membership(ActorMembershipKey(kind, actor_id)) is not None

    def holds_group(self, group_id: uuid.UUID, namespace_id: uuid.UUID) -> bool:
        return self.cache.get_group_namespace(GroupNamespaceKey(group_id, namespace_id)) is not None


async def _count(bus: FakeNatsClient, table: str) -> int | None:
    raw = await (await bus.kv_bucket(name="epochs")).get(key=generation_kv_key(table))
    return None if raw is None else int(raw.decode().rpartition(":")[2])


@pytest.fixture
def bus() -> _Lossy:
    return _Lossy()


async def _pods(pool: asyncpg.Pool, bus: _Lossy) -> tuple[_Writer, _Follower]:
    writer, follower = _Writer(pool, bus), _Follower(bus)
    # each table's generation exists before the follower first looks, as on a running platform
    for table in _TABLES:
        await writer.registry.generation_source.advance(table)  # type: ignore[union-attr]
    await follower.start()
    return writer, follower


class TestAMembershipRowEvictsItsMember:
    async def test_a_new_member_evicts_that_member_and_nobody_else(self, pool: asyncpg.Pool, bus: _Lossy) -> None:
        writer, follower = await _pods(pool, bus)
        ada, bob, group, ns = uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        follower.seed(("user", ada), ("user", bob), groups=((group, ns, frozenset()),))
        before = await _count(bus, "group_members")
        await writer.add_member(group, "user", ada)
        assert await _count(bus, "group_members") == (before or 0) + 1
        assert not follower.holds("user", ada)
        assert follower.holds("user", bob)
        assert follower.holds_group(group, ns)
        # heard in full: the follower's pass drops nothing
        assert await generation_catchup_tick(follower.registry, follower.reader) == 0

    async def test_nesting_a_group_evicts_only_the_child_groups_parent_entry(
        self, pool: asyncpg.Pool, bus: _Lossy
    ) -> None:
        writer, follower = await _pods(pool, bus)
        ada, child, parent = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        follower.seed(("user", ada), ("group", child), ("group", parent))
        await writer.add_member(parent, "group", child)
        assert not follower.holds("group", child)
        assert follower.holds("user", ada)
        assert follower.holds("group", parent)

    async def test_a_removed_member_is_evicted_by_the_row_the_delete_read(
        self, pool: asyncpg.Pool, bus: _Lossy
    ) -> None:
        writer, follower = await _pods(pool, bus)
        ada, bob, group = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        row_id = await writer.add_member(group, "user", ada)
        follower.seed(("user", ada), ("user", bob))
        await writer.members.delete((group, row_id))
        assert not follower.holds("user", ada)
        assert follower.holds("user", bob)


class TestAnAssignmentRowEvictsItsGroup:
    async def test_an_ensured_grant_evicts_its_group_and_no_other(self, pool: asyncpg.Pool, bus: _Lossy) -> None:
        writer, follower = await _pods(pool, bus)
        granted, other, ns = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        follower.seed(("user", uuid.uuid4()), groups=((granted, ns, frozenset()), (other, ns, frozenset())))
        before = await _count(bus, "role_assignments")
        _, created = await writer.assignments.ensure_group_role_assignment(
            group_id=granted, role_id=uuid.uuid4(), scope_type="namespace", scope_id=ns
        )
        assert created
        assert await _count(bus, "role_assignments") == (before or 0) + 1
        assert not follower.holds_group(granted, ns)
        assert follower.holds_group(other, ns)
        assert follower.cache.membership_size == 1
        assert await generation_catchup_tick(follower.registry, follower.reader) == 0

    async def test_a_found_grant_writes_and_advances_nothing(self, pool: asyncpg.Pool, bus: _Lossy) -> None:
        writer, follower = await _pods(pool, bus)
        group, role, ns = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        await writer.assignments.ensure_group_role_assignment(
            group_id=group, role_id=role, scope_type="namespace", scope_id=ns
        )
        follower.seed(groups=((group, ns, frozenset()),))
        before = await _count(bus, "role_assignments")
        _, created = await writer.assignments.ensure_group_role_assignment(
            group_id=group, role_id=role, scope_type="namespace", scope_id=ns
        )
        assert not created
        assert await _count(bus, "role_assignments") == before
        assert follower.holds_group(group, ns)

    async def test_a_revocation_evicts_its_group_and_one_that_matches_nothing_advances_nothing(
        self, pool: asyncpg.Pool, bus: _Lossy
    ) -> None:
        writer, follower = await _pods(pool, bus)
        group, ns = uuid.uuid4(), uuid.uuid4()
        await writer.assignments.ensure_group_role_assignment(
            group_id=group, role_id=uuid.uuid4(), scope_type="namespace", scope_id=ns
        )
        follower.seed(groups=((group, ns, frozenset()),))
        before = await _count(bus, "role_assignments")
        assert (
            await writer.assignments.delete_by_group_and_scope(group_id=group, scope_type="namespace", scope_id=ns) == 1
        )
        assert await _count(bus, "role_assignments") == (before or 0) + 1
        assert not follower.holds_group(group, ns)
        follower.seed(groups=((group, ns, frozenset()),))
        assert (
            await writer.assignments.delete_by_group_and_scope(group_id=group, scope_type="namespace", scope_id=ns) == 0
        )
        assert await _count(bus, "role_assignments") == (before or 0) + 1
        assert follower.holds_group(group, ns)


class TestARoleOrGroupRowEvictsWhatWasResolvedThroughIt:
    async def test_a_role_edit_evicts_the_entries_that_read_it(self, pool: asyncpg.Pool, bus: _Lossy) -> None:
        writer, follower = await _pods(pool, bus)
        role, other_role = uuid.uuid4(), uuid.uuid4()
        reads_it, reads_other, ns = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        follower.seed(
            ("user", uuid.uuid4()),
            groups=((reads_it, ns, frozenset({role})), (reads_other, ns, frozenset({other_role}))),
        )
        entity = writer.roles.create(
            {"role_id": role, "name": "Editor", "description": "", "permissions": json.dumps({"*": ["read"]})}
        )
        await writer.roles.save_entity(entity)
        assert not follower.holds_group(reads_it, ns)
        assert follower.holds_group(reads_other, ns)
        assert follower.cache.membership_size == 1

    async def test_a_group_write_evicts_its_entries_and_every_member_entry_naming_it(
        self, pool: asyncpg.Pool, bus: _Lossy
    ) -> None:
        writer, follower = await _pods(pool, bus)
        group, other, ns = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        ada, bob = uuid.uuid4(), uuid.uuid4()
        from threetears.agent.acl import GroupMembership, MemberType

        follower.cache.put_membership(
            ActorMembershipKey("user", ada), (GroupMembership(group, MemberType.USER, ada, None),)
        )
        follower.cache.put_membership(
            ActorMembershipKey("user", bob), (GroupMembership(other, MemberType.USER, bob, None),)
        )
        follower.seed(("group", group), groups=((group, ns, frozenset()), (other, ns, frozenset())))
        await writer.groups.save_entity(writer.groups.create({"group_id": group, "customer_id": None, "name": "g"}))
        assert not follower.holds("user", ada)
        assert not follower.holds("group", group)
        assert not follower.holds_group(group, ns)
        assert follower.holds("user", bob)
        assert follower.holds_group(other, ns)


class TestANamespaceRowEvictsWhatWasResolvedForIt:
    async def test_ensuring_a_namespace_is_announced_and_heard(self, pool: asyncpg.Pool, bus: _Lossy) -> None:
        writer, follower = await _pods(pool, bus)
        before = await _count(bus, "namespaces")
        await writer.namespaces.ensure_namespace(
            namespace_id=uuid.uuid4(), name="tool.x", namespace_type="tool", owner_agent_id=None, customer_id=None
        )
        assert await _count(bus, "namespaces") == (before or 0) + 1
        assert await generation_catchup_tick(follower.registry, follower.reader) == 0

    async def test_a_rescope_evicts_every_group_entry_for_that_namespace_and_no_other(
        self, pool: asyncpg.Pool, bus: _Lossy
    ) -> None:
        writer, follower = await _pods(pool, bus)
        ns, other_ns, group = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        await writer.namespaces.ensure_namespace(
            namespace_id=ns, name="tool.y", namespace_type="tool", owner_agent_id=None, customer_id=None
        )
        follower.seed(
            groups=((group, ns, frozenset()), (uuid.uuid4(), ns, frozenset()), (group, other_ns, frozenset()))
        )
        before = await _count(bus, "namespaces")
        outcome = await writer.namespaces.rescope(ns, customer_id=uuid.uuid4())
        assert outcome.moved
        assert await _count(bus, "namespaces") == (before or 0) + 1
        assert follower.cache.group_namespace_size == 1
        assert follower.holds_group(group, other_ns)
        assert await generation_catchup_tick(follower.registry, follower.reader) == 0


class TestAMissedBroadcastEmptiesOnlyItsTablesLayer:
    async def test_by_one_pass(self, pool: asyncpg.Pool, bus: _Lossy) -> None:
        writer, follower = await _pods(pool, bus)
        ada, group, ns = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        follower.seed(("user", ada), ("user", uuid.uuid4()), groups=((group, ns, frozenset()),))
        bus.deaf = True
        await writer.add_member(uuid.uuid4(), "user", uuid.uuid4())
        bus.deaf = False
        assert follower.cache.membership_size == 2  # nothing heard yet
        assert await generation_catchup_tick(follower.registry, follower.reader) == 1
        assert follower.cache.membership_size == 0
        assert follower.holds_group(group, ns)

    async def test_by_the_key_watcher(self, pool: asyncpg.Pool, bus: _Lossy) -> None:
        writer, follower = await _pods(pool, bus)
        group, ns = uuid.uuid4(), uuid.uuid4()
        follower_ = AccessTableFollower(follower.registry, follower.reader, grace=timedelta(milliseconds=50))
        follower_.start()
        try:
            await asyncio.sleep(0.1)  # each watch is pushed its key's latest value
            follower.seed(("user", uuid.uuid4()), groups=((group, ns, frozenset()),))
            await writer.add_member(uuid.uuid4(), "user", uuid.uuid4())  # heard
            await asyncio.sleep(0.2)
            assert follower.holds_group(group, ns)
            bus.deaf = True
            await writer.assignments.ensure_group_role_assignment(
                group_id=uuid.uuid4(), role_id=uuid.uuid4(), scope_type="all", scope_id=None
            )
            bus.deaf = False
            for _ in range(50):
                if not follower.holds_group(group, ns):
                    break
                await asyncio.sleep(0.02)
            assert not follower.holds_group(group, ns)
            assert follower.cache.membership_size == 1
        finally:
            await follower_.stop()
        assert not follower_.running


class TestOneCallBindsAndFollows:
    async def test_follow_access_tables_evicts_heard_rows_drops_missed_ones_and_reports_health(
        self, pool: asyncpg.Pool, bus: _Lossy
    ) -> None:
        writer = _Writer(pool, bus)
        for table in _TABLES:
            await writer.registry.generation_source.advance(table)  # type: ignore[union-attr]
        registry = CollectionRegistry()
        registry.configure(l2_client=bus, kv_key_scope="pod")
        await registry.start_invalidation_listener(bus)  # type: ignore[arg-type]
        loader: Any = object()
        cache = AclCache(membership_loader=loader, grant_loader=loader)
        following = follow_access_tables(registry, cache, EpochGenerationReader(bus), grace=timedelta(milliseconds=50))
        try:
            for _ in range(50):
                if following.healthy:
                    break
                await asyncio.sleep(0.02)
            assert following.healthy
            ada, bob = uuid.uuid4(), uuid.uuid4()
            cache.put_membership(ActorMembershipKey("user", ada), ())
            cache.put_membership(ActorMembershipKey("user", bob), ())
            await writer.add_member(uuid.uuid4(), "user", ada)
            assert cache.get_membership(ActorMembershipKey("user", ada)) is None
            assert cache.get_membership(ActorMembershipKey("user", bob)) is not None
            bus.deaf = True
            await writer.add_member(uuid.uuid4(), "user", uuid.uuid4())
            bus.deaf = False
            for _ in range(50):
                if cache.membership_size == 0:
                    break
                await asyncio.sleep(0.02)
            assert cache.membership_size == 0
        finally:
            await following.stop()
        assert not following.follower.running
        assert not registry.has_derived_caches("group_members")
