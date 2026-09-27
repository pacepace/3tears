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
from threetears.agent.memory import (
    LocalMemoryNamespaceProvisioner,
    memory_namespace_id,
    memory_namespace_name,
    memory_namespace_schema_name,
)
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.namespaces import build_agent_namespace_name

pytestmark = pytest.mark.integration

#: the hub's table shape (copied from agent-acl's namespace integration tests, which own the canonical copy).
_NAMESPACES_DDL = """
CREATE TABLE namespaces (
    row_scope varchar(8) NOT NULL,
    namespace_id uuid NOT NULL,
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
    CONSTRAINT namespaces_row_scope_ck CHECK (row_scope IN ('platform', 'customer')),
    CONSTRAINT namespaces_row_scope_customer_ck CHECK (
        (row_scope = 'platform' AND customer_id IS NULL
            AND namespace_type IN ('system', 'model', 'tool', 'tool_provider', 'shared', 'knowledge'))
     OR (row_scope = 'customer' AND customer_id IS NOT NULL)),
    CONSTRAINT namespaces_pkey PRIMARY KEY (row_scope, namespace_id)
)
"""


@pytest.fixture
async def pg_pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    from threetears.core.collections import init_connection

    pool: asyncpg.Pool = await asyncpg.create_pool(db_container, min_size=1, max_size=4, init=init_connection)
    try:
        async with pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS namespaces")
            await conn.execute(_NAMESPACES_DDL)
            await conn.execute("CREATE UNIQUE INDEX namespaces_id_unique ON namespaces (namespace_id)")
            await conn.execute("CREATE UNIQUE INDEX idx_namespaces_name ON namespaces (name)")
            await conn.execute(
                "CREATE UNIQUE INDEX idx_namespaces_schema_name_non_workspace ON namespaces (schema_name)"
                " WHERE namespace_type <> 'workspace' AND schema_name IS NOT NULL"
            )
            await conn.execute(
                "ALTER TABLE namespaces ADD CONSTRAINT namespaces_owner_namespace_fkey"
                " FOREIGN KEY (owner_namespace) REFERENCES namespaces(name)"
            )
        yield pool
    finally:
        async with pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS namespaces")
        await pool.close()


async def _agent_row(pool: asyncpg.Pool, agent: uuid.UUID, customer: uuid.UUID) -> None:
    """the agent's own namespace, which the owner_namespace foreign key names (the hub provisions it)."""
    now = datetime.now(UTC)
    await pool.execute(
        "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, customer_id, date_created, "
        "date_updated) VALUES ('customer', $1, $2, 'agent', $3, $4, $4)",
        uuid.uuid4(),
        build_agent_namespace_name(agent),
        customer,
        now,
    )


def _provisioner(pool: asyncpg.Pool) -> LocalMemoryNamespaceProvisioner:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool, kv_key_scope=f"local-prov-{uuid.uuid4().hex[:6]}")
    collection = NamespaceCollection(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""))
    return LocalMemoryNamespaceProvisioner(collection)


async def test_the_local_provisioner_writes_the_hubs_row(pg_pool: asyncpg.Pool) -> None:
    agent, customer = uuid.uuid4(), uuid.uuid4()
    await _agent_row(pg_pool, agent, customer)
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
    await _agent_row(pg_pool, _AGENT_A, _CUSTOMER)
    now = datetime.now(UTC)
    await pg_pool.execute(
        "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, owner_agent_id, owner_namespace, "
        "customer_id, schema_name, date_created, date_updated) "
        "VALUES ('customer', $1, 'memories.019470a8.019470a8', 'memory', $2, $3, $4, "
        "'memory__019470a8__019470a8', $5, $5)",
        memory_namespace_id(_AGENT_A, _CUSTOMER),
        _AGENT_A,
        build_agent_namespace_name(_AGENT_A),
        _CUSTOMER,
        now,
    )

    ref = await _provisioner(pg_pool).ensure(agent_id=_AGENT_A, customer_id=_CUSTOMER)

    assert ref.id == memory_namespace_id(_AGENT_A, _CUSTOMER)
    assert ref.name == "memories.019470a8.019470a8"
    row = await pg_pool.fetchrow("SELECT name, schema_name FROM namespaces WHERE namespace_type = 'memory'")
    assert row is not None
    assert (row["name"], row["schema_name"]) == ("memories.019470a8.019470a8", "memory__019470a8__019470a8")
