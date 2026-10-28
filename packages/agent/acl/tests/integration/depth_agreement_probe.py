"""at the nesting this process was configured with, do the evaluator and the visibility clause agree?

run by ``test_query_visibility_nesting.py`` once per setting, each in a fresh interpreter with
``THREETEARS_ACL_MAX_GROUP_NESTING_LEVELS`` set, because the nesting is read once at import --
the way a deployment configures it. against the tables that module's fixture created, it seeds
one chain per length from 1 group to one past the configured depth (levels - 1 groups; the caller in the bottom group,
the grant on the top one), asks the evaluator through the production
:class:`CollectionMembershipLoader` and the SQL through :func:`caller_visible_customer_clause`,
and prints one JSON object: the levels and depth this process resolved and, per chain length, both verdicts.

usage: python depth_agreement_probe.py DSN
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid

import asyncpg

from threetears.agent.acl import (
    MAX_GROUP_MEMBERSHIP_DEPTH,
    AclCache,
    CollectionMembershipLoader,
    EvaluationContext,
    Group,
    GroupMemberCollection,
    Namespace,
    Role,
    RoleAssignment,
    ScopeType,
    caller_visible_customer_clause,
    evaluate_decision,
)
from threetears.agent.acl.types import MAX_GROUP_NESTING_LEVELS
from threetears.core.collections import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeNatsClient
from threetears.nats import set_default_namespace

READER = Role(id=uuid.uuid4(), name="Reader", permissions={"thing": frozenset({"read"})}, is_built_in=True)


class SeededGrants:
    """the evaluator's grant side over the fixture's ``role_assignments`` rows.

    every assignment grants :data:`READER`; groups belong to the customer they are evaluated
    against, which is the only shape this probe seeds.
    """

    def __init__(self, pool: asyncpg.Pool, customer_id: uuid.UUID) -> None:
        """capture the pool and the customer every seeded group belongs to.

        :param pool: pool over the fixture's tables
        :ptype pool: asyncpg.Pool
        :param customer_id: the customer every seeded row belongs to
        :ptype customer_id: uuid.UUID
        :return: nothing
        :rtype: None
        """
        self._pool = pool
        self._customer_id = customer_id

    async def load_assignments_for_groups(
        self, group_ids: tuple[uuid.UUID, ...], namespace: Namespace
    ) -> tuple[RoleAssignment, ...]:
        """every seeded assignment the groups hold; the evaluator re-checks coverage.

        :param group_ids: groups to inspect
        :ptype group_ids: tuple[uuid.UUID, ...]
        :param namespace: namespace under evaluation (the evaluator re-checks coverage)
        :ptype namespace: Namespace
        :return: the groups' assignments
        :rtype: tuple[RoleAssignment, ...]
        """
        _ = namespace
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT assignment_id, group_id, scope_namespace_type, scope_customer_id "
                "FROM role_assignments WHERE group_id = ANY($1::uuid[])",
                list(group_ids),
            )
        return tuple(
            RoleAssignment(
                id=row["assignment_id"],
                role_id=READER.id,
                group_id=row["group_id"],
                scope_type=ScopeType.TYPE_CUSTOMER,
                scope_namespace_id=None,
                scope_namespace_type=row["scope_namespace_type"],
                scope_customer_id=row["scope_customer_id"],
            )
            for row in rows
        )

    async def load_roles(self, role_ids: tuple[uuid.UUID, ...]) -> dict[uuid.UUID, Role]:
        """the reader role, the only one this probe grants.

        :param role_ids: role ids to resolve
        :ptype role_ids: tuple[uuid.UUID, ...]
        :return: the reader role where asked for
        :rtype: dict[uuid.UUID, Role]
        """
        return {rid: READER for rid in role_ids if rid == READER.id}

    async def load_groups(self, group_ids: tuple[uuid.UUID, ...]) -> dict[uuid.UUID, object]:
        """each group as the evaluated customer's.

        :param group_ids: group ids to resolve
        :ptype group_ids: tuple[uuid.UUID, ...]
        :return: one group row per id
        :rtype: dict[uuid.UUID, object]
        """
        return {gid: Group(id=gid, name=str(gid), customer_id=self._customer_id) for gid in group_ids}


async def _seed_chain(conn: asyncpg.Connection, groups: int, user_id: uuid.UUID, customer_id: uuid.UUID) -> None:
    """seed ``groups`` groups, each a member of the next, the caller in the bottom one, the grant on the top.

    :param conn: open connection
    :ptype conn: asyncpg.Connection
    :param groups: how many groups the chain holds
    :ptype groups: int
    :param user_id: the caller
    :ptype user_id: uuid.UUID
    :param customer_id: the customer the listed row and the grant belong to
    :ptype customer_id: uuid.UUID
    :return: nothing
    :rtype: None
    """
    chain = [uuid.uuid4() for _ in range(groups)]
    await conn.execute("INSERT INTO listed_things (thing_id, customer_id) VALUES ($1, $2)", uuid.uuid4(), customer_id)
    await conn.execute(
        "INSERT INTO role_assignments (assignment_id, group_id, scope_type, scope_customer_id, scope_namespace_type) "
        "VALUES ($1, $2, 'type_customer', $3, 'thing')",
        uuid.uuid4(),
        chain[-1],
        customer_id,
    )
    edges = [(parent, "group", child) for child, parent in zip(chain, chain[1:], strict=False)]
    for group_id, member_type, member_id in [*edges, (chain[0], "user", user_id)]:
        await conn.execute(
            "INSERT INTO group_members (id, group_id, member_type, member_id) VALUES ($1, $2, $3, $4)",
            uuid.uuid4(),
            group_id,
            member_type,
            member_id,
        )


async def _verdicts(pool: asyncpg.Pool, groups: int) -> dict[str, bool]:
    """seed one chain of ``groups`` groups and ask both surfaces whether its caller reaches the grant.

    :param pool: pool over the fixture's tables
    :ptype pool: asyncpg.Pool
    :param groups: how many groups the chain holds
    :ptype groups: int
    :return: ``{"granted": ..., "visible": ...}``
    :rtype: dict[str, bool]
    """
    customer_id, user_id = uuid.uuid4(), uuid.uuid4()
    async with pool.acquire() as conn:
        await _seed_chain(conn, groups, user_id, customer_id)
        fragment, params = caller_visible_customer_clause(
            user_id=user_id,
            customer_id_column="listed_things.customer_id",
            scope_namespace_type="thing",
            param_offset=1,
        )
        visible = await conn.fetchval(f"SELECT count(*) FROM listed_things WHERE {fragment}", *params)
    bus = FakeNatsClient()
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool, l2_client=bus, kv_key_scope="depthagree")
    members = GroupMemberCollection(registry, DefaultCoreConfig(), nats_client=bus)
    cache = AclCache(
        membership_loader=CollectionMembershipLoader(members), grant_loader=SeededGrants(pool, customer_id)
    )
    namespace = Namespace(id=uuid.uuid4(), customer_id=customer_id, namespace_type="thing", owner_agent_id=None)
    granted = await evaluate_decision(
        EvaluationContext(namespace=namespace, action="read", user_id=user_id), cache=cache
    )
    return {"granted": granted, "visible": visible == 1}


async def main(dsn: str) -> None:
    """print the resolved depth and both verdicts for every chain length up to one past it.

    :param dsn: the fixture database
    :ptype dsn: str
    :return: nothing
    :rtype: None
    """
    set_default_namespace("depthagree")
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2, init=init_connection)
    try:
        chains = {groups: await _verdicts(pool, groups) for groups in range(1, MAX_GROUP_MEMBERSHIP_DEPTH + 2)}
    finally:
        await pool.close()
    print(json.dumps({"levels": MAX_GROUP_NESTING_LEVELS, "depth": MAX_GROUP_MEMBERSHIP_DEPTH, "chains": chains}))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
