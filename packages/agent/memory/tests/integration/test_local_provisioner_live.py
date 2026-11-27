"""Integration: ``LocalMemoryNamespaceProvisioner`` on the hub's ``namespaces`` table shape.

The row a hub-less deployment materializes must be the row the hub writes, owner foreign key included:
every field is checked against the public helpers, and a second ensure is a no-op on the same row.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import asyncpg
import pytest

from threetears.agent.acl import NamespaceCollection
from threetears.agent.acl.migrations import register as register_acl
from threetears.agent.memory import (
    LocalMemoryNamespaceProvisioner,
    memory_namespace_id,
    memory_namespace_name,
    memory_namespace_schema_name,
)
from threetears.agent.tools.platform_migrations import register as register_tools_platform
from threetears.core.collections import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.core.ddl.row_epoch import next_epoch
from threetears.core.namespaces import build_agent_namespace_name

from .conftest import AsyncpgStore

pytestmark = pytest.mark.integration


async def _hub_namespaces(conn: asyncpg.Connection) -> None:
    """``namespaces`` as the hub has it, on the schema ``conn``'s search path names.

    agent-acl's migrations, then agent-tools platform's (the tool columns ``NamespaceCollection`` reads), never a
    hand-written copy, which drifted from the stamped table (``epoch``, ``gone``, ``owner_namespace_id``). On top, the
    hub's own keys (v199), which the acl package leaves to the deployment: a name unique among live rows, and the
    owner foreign key onto ``namespace_id``, a namespace's owner stored by id.

    :param conn: a connection whose search path is the test's schema
    :ptype conn: asyncpg.Connection
    :return: nothing
    :rtype: None
    """
    runner = MigrationRunner()
    register_acl(runner)
    register_tools_platform(runner, depends_on=("agent_acl",))
    await runner.apply_for_platform_schema(AsyncpgStore(conn))
    await conn.execute("CREATE UNIQUE INDEX uq_namespaces_name ON namespaces (name) WHERE NOT gone")
    await conn.execute(
        "ALTER TABLE namespaces ADD CONSTRAINT namespaces_owner_namespace_id_fkey"
        " FOREIGN KEY (owner_namespace_id) REFERENCES namespaces (namespace_id)"
    )


@pytest.fixture
async def pg_pool(pg_schema: tuple[str, str]) -> AsyncIterator[asyncpg.Pool]:
    url, schema = pg_schema
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}"')
        await _hub_namespaces(conn)
    finally:
        await conn.close()
    pool: asyncpg.Pool = await asyncpg.create_pool(
        url, min_size=1, max_size=4, init=init_connection, server_settings={"search_path": f'"{schema}"'}
    )
    try:
        yield pool
    finally:
        await pool.close()


async def _agent_row(pool: asyncpg.Pool, agent: uuid.UUID, customer: uuid.UUID) -> uuid.UUID:
    """the agent's own namespace, owned by itself as the hub provisions it, which the owner foreign key names; its id."""
    now = datetime.now(UTC)
    namespace_id = uuid.uuid4()
    await pool.execute(
        "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, owner_agent_id, owner_namespace, "
        "owner_namespace_id, customer_id, date_created, date_updated, epoch) "
        "VALUES ('customer', $1, $2, 'agent', $3, $2, $1, $4, $5, $5, $6)",
        namespace_id,
        build_agent_namespace_name(agent),
        agent,
        customer,
        now,
        next_epoch(),
    )
    return namespace_id


def _provisioner(pool: asyncpg.Pool) -> LocalMemoryNamespaceProvisioner:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool, kv_key_scope=f"local-prov-{uuid.uuid4().hex[:6]}")
    collection = NamespaceCollection(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""))
    return LocalMemoryNamespaceProvisioner(collection)


async def test_the_local_provisioner_writes_the_hubs_row(pg_pool: asyncpg.Pool) -> None:
    agent, customer = uuid.uuid4(), uuid.uuid4()
    owner = await _agent_row(pg_pool, agent, customer)
    provisioner = _provisioner(pg_pool)

    ref = await provisioner.ensure(agent_id=agent, customer_id=customer)
    again = await provisioner.ensure(agent_id=agent, customer_id=customer)

    assert ref == again
    row = await pg_pool.fetchrow("SELECT * FROM namespaces WHERE namespace_type = 'memory'")
    assert row is not None
    assert row["namespace_id"] == memory_namespace_id(agent, customer) == ref.id
    assert row["name"] == memory_namespace_name(agent, customer)
    assert row["owner_agent_id"] == agent and row["customer_id"] == customer
    assert row["owner_namespace"] == build_agent_namespace_name(agent) == ref.owner_namespace
    assert row["owner_namespace_id"] == owner, "the owner is stored by id, as the hub stores it"
    assert row["schema_name"] == memory_namespace_schema_name(agent, customer)
    assert await pg_pool.fetchval("SELECT count(*) FROM namespaces WHERE namespace_type = 'memory'") == 1


#: two agents a uuid7 generator could mint in one millisecond for one customer: identical
#: leading 48 bits (the timestamp), different random tails.
_AGENT_A = uuid.UUID("019470a8-b5c3-7def-8123-456789abcdef")
_AGENT_B = uuid.UUID("019470a8-b5c3-7a01-9fed-cba987654321")
_CUSTOMER = uuid.UUID("019470a8-b5c4-7000-8000-000000000001")


async def test_two_agents_minted_in_one_millisecond_both_get_a_memory_namespace(pg_pool: asyncpg.Pool) -> None:
    """The production failure, on the real unique indexes: the second agent's row could not be written."""
    assert _AGENT_A.int >> 80 == _AGENT_B.int >> 80, "the fixture must share the uuid7 timestamp"
    await _agent_row(pg_pool, _AGENT_A, _CUSTOMER)
    await _agent_row(pg_pool, _AGENT_B, _CUSTOMER)
    provisioner = _provisioner(pg_pool)

    first = await provisioner.ensure(agent_id=_AGENT_A, customer_id=_CUSTOMER)
    second = await provisioner.ensure(agent_id=_AGENT_B, customer_id=_CUSTOMER)

    assert (first.id, second.id) == (memory_namespace_id(_AGENT_A, _CUSTOMER), memory_namespace_id(_AGENT_B, _CUSTOMER))
    assert await pg_pool.fetchval("SELECT count(*) FROM namespaces WHERE namespace_type = 'memory'") == 2


async def test_a_row_named_by_the_earlier_rule_is_found_not_refused(pg_pool: asyncpg.Pool) -> None:
    """A row a deployed database already holds, named by the eight-character rule, keeps working.

    ``ensure_namespace`` refuses a row that disagrees with any field it is handed, the name included.
    The provisioner resolves by (type, owner agent, customer) first, so the existing row is returned as
    it is and never compared against the current rule's name.
    """
    owner = await _agent_row(pg_pool, _AGENT_A, _CUSTOMER)
    now = datetime.now(UTC)
    await pg_pool.execute(
        "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, owner_agent_id, owner_namespace, "
        "owner_namespace_id, customer_id, schema_name, date_created, date_updated, epoch) "
        "VALUES ('customer', $1, 'memories.019470a8.019470a8', 'memory', $2, $3, $4, $5, "
        "'memory__019470a8__019470a8', $6, $6, $7)",
        memory_namespace_id(_AGENT_A, _CUSTOMER),
        _AGENT_A,
        build_agent_namespace_name(_AGENT_A),
        owner,
        _CUSTOMER,
        now,
        next_epoch(),
    )

    ref = await _provisioner(pg_pool).ensure(agent_id=_AGENT_A, customer_id=_CUSTOMER)

    assert ref.id == memory_namespace_id(_AGENT_A, _CUSTOMER)
    assert ref.name == "memories.019470a8.019470a8"
    row = await pg_pool.fetchrow("SELECT name, schema_name FROM namespaces WHERE namespace_type = 'memory'")
    assert row is not None
    assert (row["name"], row["schema_name"]) == ("memories.019470a8.019470a8", "memory__019470a8__019470a8")
