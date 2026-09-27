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
                "ALTER TABLE namespaces ADD CONSTRAINT namespaces_owner_namespace_fkey"
                " FOREIGN KEY (owner_namespace) REFERENCES namespaces(name)"
            )
        yield pool
    finally:
        async with pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS namespaces")
        await pool.close()


async def test_the_local_provisioner_writes_the_hubs_row(pg_pool: asyncpg.Pool) -> None:
    agent, customer = uuid.uuid4(), uuid.uuid4()
    now = datetime.now(UTC)
    # the agent's own namespace, which the owner_namespace foreign key names (the hub provisions it)
    await pg_pool.execute(
        "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, customer_id, date_created, "
        "date_updated) VALUES ('customer', $1, $2, 'agent', $3, $4, $4)",
        uuid.uuid4(),
        build_agent_namespace_name(agent),
        customer,
        now,
    )
    registry = CollectionRegistry()
    registry.configure(l3_pool=pg_pool, kv_key_scope=f"local-prov-{uuid.uuid4().hex[:6]}")
    collection = NamespaceCollection(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""))
    provisioner = LocalMemoryNamespaceProvisioner(collection)

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
