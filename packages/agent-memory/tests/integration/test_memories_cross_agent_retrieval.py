"""integration: cross-agent memory retrieval through MemoryAccessService.

collections-task-04. exercises the full composition shape end-to-end:

- two agents (A, B) within one customer, each with a memory namespace
  rooted on its own ``owner_agent_id``.
- one user with memories under both agents (rows tagged with the
  matching ``agent_id`` partition column).
- a stub :class:`AclCache`-shape bundle whose loaders authorize one or
  the other namespace selectively per test scenario.
- :meth:`MemoryAccessService.find_for_user_across_authorized_agents`
  resolves the candidate namespaces, runs the unified evaluator on
  each, fans out to
  :meth:`MemoriesCollection.find_for_user_in_agents` only with the
  authorized agent ids, and returns the union.

positive case: caller authorized on both agents -> both agents'
memories surface.
negative case: caller authorized on agent A only -> only agent A's
memories surface (agent B's are never queried).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest

from threetears.agent.acl import (
    Group,
    GroupMembership,
    MemberType,
    Role,
    RoleAssignment,
    ScopeType,
)
from threetears.agent.memory.access import MemoryAccessService
from threetears.agent.memory.authorize import (
    ACTION_MEMORY_READ,
    MEMORY_NAMESPACE_TYPE,
    MemoryAuthorizerDependencies,
)
from threetears.agent.memory.collections import MemoriesCollection
from threetears.agent.memory.migrations import register as register_memory
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration


@pytest.fixture
async def applied_schema(pg_schema: tuple[str, str]) -> tuple[str, str]:
    """apply conversations + memory migrations into the per-test schema."""
    url, schema = pg_schema
    runner = MigrationRunner()
    register_conversations(runner)
    register_memory(runner)
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        store = AsyncpgStore(conn)
        await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
    finally:
        await conn.close()
    return url, schema


async def _make_pool(url: str, schema: str) -> asyncpg.Pool:
    """build an asyncpg pool with search_path pre-bound to the test schema."""
    result: asyncpg.Pool = await asyncpg.create_pool(
        dsn=url,
        min_size=1,
        max_size=4,
        server_settings={"search_path": f"{schema}, public"},
    )
    return result


def _build_memories_collection(
    pool: asyncpg.Pool,
    authorizer: MemoryAuthorizerDependencies,
) -> MemoriesCollection:
    """build a registry-bound :class:`MemoriesCollection` around the pool."""
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    config = DefaultCoreConfig(
        collection_flush="ALWAYS",
        collection_flush_tables="",
    )
    return MemoriesCollection(
        registry=registry,
        config=config,
        authorizer=authorizer,
    )


class _StubNamespaceEntity:
    """duck-typed stand-in for :class:`NamespaceEntity`.

    carries the four attributes :class:`MemoryAccessService` and
    :func:`evaluate_decision` read.
    """

    __slots__ = (
        "id",
        "name",
        "namespace_type",
        "owner_agent_id",
        "customer_id",
        "schema_name",
        "metadata",
        "date_created",
        "date_updated",
    )

    def __init__(
        self,
        *,
        id: uuid.UUID,
        owner_agent_id: uuid.UUID,
        customer_id: uuid.UUID,
        name: str,
    ) -> None:
        """initialize the stub namespace entity.

        :param id: namespace UUID
        :ptype id: uuid.UUID
        :param owner_agent_id: owning agent UUID
        :ptype owner_agent_id: uuid.UUID
        :param customer_id: owning customer UUID
        :ptype customer_id: uuid.UUID
        :param name: display name
        :ptype name: str
        """
        self.id = id
        self.name = name
        self.namespace_type = MEMORY_NAMESPACE_TYPE
        self.owner_agent_id = owner_agent_id
        self.customer_id = customer_id
        self.schema_name = ""
        self.metadata = {}
        self.date_created = datetime.now(UTC)
        self.date_updated = datetime.now(UTC)


class _NamespaceCollectionStub:
    """duck-typed :class:`NamespaceCollection` returning two memory namespaces."""

    def __init__(
        self,
        namespaces: list[_StubNamespaceEntity],
    ) -> None:
        """store the static namespace catalogue.

        :param namespaces: list of stub namespace entities to return
        :ptype namespaces: list[_StubNamespaceEntity]
        """
        self._namespaces = namespaces

    async def find_by_type_and_customer(
        self,
        *,
        namespace_type: str,
        customer_id: uuid.UUID,
    ) -> list[_StubNamespaceEntity]:
        """return the configured namespaces (no real filtering needed in tests).

        :param namespace_type: namespace type filter (unused here)
        :ptype namespace_type: str
        :param customer_id: customer filter (unused here)
        :ptype customer_id: uuid.UUID
        :return: list of stub namespaces
        :rtype: list[_StubNamespaceEntity]
        """
        _ = namespace_type, customer_id
        return list(self._namespaces)


class _SelectiveMembershipLoader:
    """MembershipLoader that binds the user only to the authorized groups."""

    def __init__(self, user_id: uuid.UUID, group_ids: tuple[uuid.UUID, ...]) -> None:
        """store the user UUID and the groups they belong to.

        :param user_id: the user the memberships speak for
        :ptype user_id: uuid.UUID
        :param group_ids: tuple of group UUIDs the user is a member of
        :ptype group_ids: tuple[uuid.UUID, ...]
        """
        self._user_id = user_id
        self._group_ids = group_ids

    async def load_for_user(
        self, user_id: uuid.UUID,
    ) -> tuple[GroupMembership, ...]:
        """return one membership per configured group.

        :param user_id: user UUID to resolve memberships for
        :ptype user_id: uuid.UUID
        :return: tuple of memberships
        :rtype: tuple[GroupMembership, ...]
        """
        if user_id != self._user_id:
            return ()
        return tuple(
            GroupMembership(
                group_id=gid,
                member_type=MemberType.USER,
                member_id=user_id,
                customer_id=None,
            )
            for gid in self._group_ids
        )

    async def load_for_agent(
        self, agent_id: uuid.UUID,
    ) -> tuple[GroupMembership, ...]:
        """no agent-side memberships in this test fixture.

        :param agent_id: agent UUID
        :ptype agent_id: uuid.UUID
        :return: empty tuple
        :rtype: tuple[GroupMembership, ...]
        """
        _ = agent_id
        return ()


class _SelectiveGrantLoader:
    """GrantLoader scoping ``memory.read`` per-namespace.

    ``namespace_to_group`` maps a namespace UUID to the group whose
    members get the read grant on it. enables tests to authorize one
    namespace and deny another.
    """

    def __init__(
        self,
        *,
        role_id: uuid.UUID,
        namespace_to_group: dict[uuid.UUID, uuid.UUID],
    ) -> None:
        """store the role and the namespace-to-group routing table.

        :param role_id: synthetic role UUID carrying ``memory.read``
        :ptype role_id: uuid.UUID
        :param namespace_to_group: mapping namespace UUID -> group UUID
            granted on that namespace
        :ptype namespace_to_group: dict[uuid.UUID, uuid.UUID]
        """
        self._role_id = role_id
        self._namespace_to_group = namespace_to_group
        self._role = Role(
            id=role_id,
            name="MemoryReader",
            permissions={"memory": frozenset({ACTION_MEMORY_READ})},
            is_built_in=True,
        )

    async def load_assignments_for_groups(
        self,
        group_ids: tuple[uuid.UUID, ...],
        namespace: object,
    ) -> tuple[RoleAssignment, ...]:
        """return assignments only for the group authorized on this namespace.

        :param group_ids: groups under evaluation
        :ptype group_ids: tuple[uuid.UUID, ...]
        :param namespace: namespace under evaluation
        :ptype namespace: object
        :return: tuple of assignments
        :rtype: tuple[RoleAssignment, ...]
        """
        ns_id = getattr(namespace, "id", None)
        if ns_id is None:
            return ()
        authorized_group = self._namespace_to_group.get(ns_id)
        if authorized_group is None:
            return ()
        if authorized_group not in group_ids:
            return ()
        return (
            RoleAssignment(
                id=uuid.uuid4(),
                role_id=self._role_id,
                group_id=authorized_group,
                scope_type=ScopeType.NAMESPACE,
                scope_namespace_id=ns_id,
                scope_namespace_type=None,
                scope_customer_id=None,
            ),
        )

    async def load_roles(
        self, role_ids: tuple[uuid.UUID, ...],
    ) -> dict[uuid.UUID, Role]:
        """return the synthetic role for every requested id this loader owns.

        :param role_ids: role UUIDs to resolve
        :ptype role_ids: tuple[uuid.UUID, ...]
        :return: mapping role_id -> :class:`Role`
        :rtype: dict[uuid.UUID, Role]
        """
        return {rid: self._role for rid in role_ids if rid == self._role_id}

    async def load_groups(
        self, group_ids: tuple[uuid.UUID, ...],
    ) -> dict[uuid.UUID, Any]:
        """resolve every group id to a platform-scoped group.

        the evaluator skips assignments whose group does not resolve
        through this method.

        :param group_ids: group UUIDs to resolve
        :ptype group_ids: tuple[uuid.UUID, ...]
        :return: mapping group_id -> :class:`Group`
        :rtype: dict[uuid.UUID, Any]
        """
        return {
            gid: Group(
                id=gid,
                name=f"group-{gid.hex[:8]}",
                customer_id=None,
            )
            for gid in group_ids
        }


def _AclCacheStub(  # noqa: N802
    *,
    membership_loader: _SelectiveMembershipLoader,
    grant_loader: _SelectiveGrantLoader,
) -> Any:
    """factory returning a real :class:`AclCache` wrapping the loaders.

    acl-evaluator-task-01 wired ``evaluate_decision`` through the
    canonical :class:`AclCache` membership and per-namespace layers;
    a hand-rolled stub no longer satisfies the contract. integration
    tests use the real cache so the on-pod cache hit path is exercised
    end-to-end.

    :param membership_loader: configured membership loader
    :ptype membership_loader: _SelectiveMembershipLoader
    :param grant_loader: configured grant loader
    :ptype grant_loader: _SelectiveGrantLoader
    :return: real :class:`AclCache` instance
    :rtype: AclCache
    """
    from threetears.agent.acl import AclCache
    return AclCache(
        membership_loader=membership_loader,
        grant_loader=grant_loader,
    )


def _seed_memory(
    pool: asyncpg.Pool,
    *,
    agent_id: uuid.UUID,
    customer_id: uuid.UUID,
    user_id: uuid.UUID,
    content: str,
) -> Any:
    """build an INSERT coroutine for one memory row in the given partition.

    :param pool: asyncpg pool
    :ptype pool: asyncpg.Pool
    :param agent_id: partition column value
    :ptype agent_id: uuid.UUID
    :param customer_id: customer scope
    :ptype customer_id: uuid.UUID
    :param user_id: row owner
    :ptype user_id: uuid.UUID
    :param content: content text
    :ptype content: str
    :return: awaitable coroutine ready for pool.execute
    :rtype: Any
    """
    now = datetime.now(UTC).replace(tzinfo=None)
    return pool.execute(
        "INSERT INTO memories ("
        "memory_id, agent_id, customer_id, user_id, "
        "conversation_id, message_id_source, type_memory, content, "
        "summary, embedding, is_deleted, date_created, date_updated"
        ") VALUES ($1, $2, $3, $4, $5, $6, 'fact', $7, NULL, "
        "$8::vector, FALSE, $9, $9)",
        uuid.uuid4(),
        agent_id,
        customer_id,
        user_id,
        uuid.uuid4(),
        uuid.uuid4(),
        content,
        "[" + ",".join(["0.1"] * 1024) + "]",
        now,
    )


class TestCrossAgentRetrieval:
    """:class:`MemoryAccessService` composes ACL with partition fan-out."""

    async def test_authorized_on_both_agents_returns_both(
        self,
        applied_schema: tuple[str, str],
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        """caller authorized on both agents sees both agents' memories.

        :param applied_schema: (url, schema) tuple after migrations
        :ptype applied_schema: tuple[str, str]
        :param permissive_memory_authorizer: permissive authorizer
            (used by :class:`MemoriesCollection` constructor; the
            access service uses the selective stub instead)
        :ptype permissive_memory_authorizer: MemoryAuthorizerDependencies
        """
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            customer_id = uuid.uuid4()
            user_id = uuid.uuid4()
            caller_user_id = user_id
            agent_a = uuid.uuid4()
            agent_b = uuid.uuid4()
            ns_a = _StubNamespaceEntity(
                id=uuid.uuid4(),
                owner_agent_id=agent_a,
                customer_id=customer_id,
                name="memories.agent-a",
            )
            ns_b = _StubNamespaceEntity(
                id=uuid.uuid4(),
                owner_agent_id=agent_b,
                customer_id=customer_id,
                name="memories.agent-b",
            )

            # seed one memory per agent
            await _seed_memory(
                pool,
                agent_id=agent_a,
                customer_id=customer_id,
                user_id=user_id,
                content="memory from agent A",
            )
            await _seed_memory(
                pool,
                agent_id=agent_b,
                customer_id=customer_id,
                user_id=user_id,
                content="memory from agent B",
            )

            group_a = uuid.uuid4()
            group_b = uuid.uuid4()
            membership = _SelectiveMembershipLoader(
                user_id=caller_user_id,
                group_ids=(group_a, group_b),
            )
            grant = _SelectiveGrantLoader(
                role_id=uuid.uuid4(),
                namespace_to_group={
                    ns_a.id: group_a,
                    ns_b.id: group_b,
                },
            )
            acl_cache = _AclCacheStub(
                membership_loader=membership,
                grant_loader=grant,
            )
            namespace_collection = _NamespaceCollectionStub([ns_a, ns_b])
            memories = _build_memories_collection(
                pool, permissive_memory_authorizer,
            )

            service = MemoryAccessService(
                acl_cache=acl_cache,
                namespace_collection=namespace_collection,
                memories_collection=memories,
            )

            results = await service.find_for_user_across_authorized_agents(
                user_id=user_id,
                caller_user_id=caller_user_id,
                customer_id=customer_id,
            )

            contents = {entity.content for entity in results}
            assert contents == {"memory from agent A", "memory from agent B"}
        finally:
            await pool.close()

    async def test_authorized_on_one_agent_returns_only_that_agent(
        self,
        applied_schema: tuple[str, str],
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        """caller authorized on agent A only does NOT see agent B's memories.

        verifies the negative grant path: revoking the grant on
        agent B (by removing it from ``namespace_to_group``) means
        agent B's namespace fails the evaluator and its
        ``owner_agent_id`` is never added to the authorized tuple
        passed to the Collection's fan-out method.

        :param applied_schema: (url, schema) tuple after migrations
        :ptype applied_schema: tuple[str, str]
        :param permissive_memory_authorizer: permissive authorizer
        :ptype permissive_memory_authorizer: MemoryAuthorizerDependencies
        """
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            customer_id = uuid.uuid4()
            user_id = uuid.uuid4()
            caller_user_id = user_id
            agent_a = uuid.uuid4()
            agent_b = uuid.uuid4()
            ns_a = _StubNamespaceEntity(
                id=uuid.uuid4(),
                owner_agent_id=agent_a,
                customer_id=customer_id,
                name="memories.agent-a",
            )
            ns_b = _StubNamespaceEntity(
                id=uuid.uuid4(),
                owner_agent_id=agent_b,
                customer_id=customer_id,
                name="memories.agent-b",
            )

            await _seed_memory(
                pool,
                agent_id=agent_a,
                customer_id=customer_id,
                user_id=user_id,
                content="memory from agent A",
            )
            await _seed_memory(
                pool,
                agent_id=agent_b,
                customer_id=customer_id,
                user_id=user_id,
                content="memory from agent B",
            )

            group_a = uuid.uuid4()
            membership = _SelectiveMembershipLoader(
                user_id=caller_user_id,
                group_ids=(group_a,),
            )
            grant = _SelectiveGrantLoader(
                role_id=uuid.uuid4(),
                namespace_to_group={
                    ns_a.id: group_a,
                    # ns_b deliberately absent — caller has no group
                    # mapped to that namespace, so the evaluator denies
                },
            )
            acl_cache = _AclCacheStub(
                membership_loader=membership,
                grant_loader=grant,
            )
            namespace_collection = _NamespaceCollectionStub([ns_a, ns_b])
            memories = _build_memories_collection(
                pool, permissive_memory_authorizer,
            )

            service = MemoryAccessService(
                acl_cache=acl_cache,
                namespace_collection=namespace_collection,
                memories_collection=memories,
            )

            results = await service.find_for_user_across_authorized_agents(
                user_id=user_id,
                caller_user_id=caller_user_id,
                customer_id=customer_id,
            )

            contents = {entity.content for entity in results}
            assert contents == {"memory from agent A"}
            assert "memory from agent B" not in contents
        finally:
            await pool.close()

    async def test_no_authorized_agents_returns_empty(
        self,
        applied_schema: tuple[str, str],
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        """caller with no grants returns an empty list (no Collection call).

        guards the short-circuit path: if every candidate namespace
        denies, the service must NOT call the Collection's
        ``@spans_partitions`` method (which would raise on an empty
        tuple). it returns ``[]`` early.

        :param applied_schema: (url, schema) tuple after migrations
        :ptype applied_schema: tuple[str, str]
        :param permissive_memory_authorizer: permissive authorizer
        :ptype permissive_memory_authorizer: MemoryAuthorizerDependencies
        """
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            customer_id = uuid.uuid4()
            user_id = uuid.uuid4()
            caller_user_id = user_id
            agent_a = uuid.uuid4()
            ns_a = _StubNamespaceEntity(
                id=uuid.uuid4(),
                owner_agent_id=agent_a,
                customer_id=customer_id,
                name="memories.agent-a",
            )

            await _seed_memory(
                pool,
                agent_id=agent_a,
                customer_id=customer_id,
                user_id=user_id,
                content="memory from agent A",
            )

            membership = _SelectiveMembershipLoader(
                user_id=caller_user_id,
                group_ids=(),  # caller has no group memberships
            )
            grant = _SelectiveGrantLoader(
                role_id=uuid.uuid4(),
                namespace_to_group={},
            )
            acl_cache = _AclCacheStub(
                membership_loader=membership,
                grant_loader=grant,
            )
            namespace_collection = _NamespaceCollectionStub([ns_a])
            memories = _build_memories_collection(
                pool, permissive_memory_authorizer,
            )

            service = MemoryAccessService(
                acl_cache=acl_cache,
                namespace_collection=namespace_collection,
                memories_collection=memories,
            )

            results = await service.find_for_user_across_authorized_agents(
                user_id=user_id,
                caller_user_id=caller_user_id,
                customer_id=customer_id,
            )

            assert results == []
        finally:
            await pool.close()
