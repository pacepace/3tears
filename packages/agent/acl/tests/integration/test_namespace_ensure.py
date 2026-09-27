"""``NamespaceCollection.ensure_namespace``: get-or-create a namespace row on a deterministic id.

The hub materializes a memory namespace on request; a deployment with no hub (one application that owns
its own control plane) had no supported way to, and wrote the row with raw SQL -- leaving
``owner_namespace`` NULL, so no agent owned it. This is that write, once, through the collection:
idempotent, convergent across pods on a deterministic id, and refusing a row that already means
something else. Measured against the real table shape, including the ``owner_namespace`` foreign key.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import asyncpg
import pytest

from threetears.agent.acl import NamespaceCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

pytestmark = pytest.mark.integration

#: the hub's table shape, as ``test_namespace_owned_by`` carries it (composite key, partition CHECK,
#: the unique name index and the owner_namespace foreign key onto it).
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


def _collection(pool: asyncpg.Pool) -> NamespaceCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool, kv_key_scope=f"ensure-{uuid.uuid4().hex[:6]}")
    return NamespaceCollection(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""))


async def _owner_row(pool: asyncpg.Pool, name: str, customer_id: uuid.UUID) -> None:
    """the owning agent's own namespace, which the owner_namespace foreign key names."""
    now = datetime.now(UTC)
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, customer_id, "
            "date_created, date_updated) VALUES ('customer', $1, $2, 'agent', $3, $4, $4)",
            uuid.uuid4(),
            name,
            customer_id,
            now,
        )


def _fields(customer: uuid.UUID, agent: uuid.UUID, namespace_id: uuid.UUID) -> dict[str, object]:
    return {
        "namespace_id": namespace_id,
        "name": f"memories.{agent.hex}.{customer.hex}",
        "namespace_type": "memory",
        "owner_agent_id": agent,
        "customer_id": customer,
        "owner_namespace": "agent-owner",
        "schema_name": f"memory__{namespace_id.hex}",
    }


async def test_ensure_creates_the_row_with_every_field(pg_pool: asyncpg.Pool) -> None:
    customer, agent, namespace_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await _owner_row(pg_pool, "agent-owner", customer)
    entity = await _collection(pg_pool).ensure_namespace(**_fields(customer, agent, namespace_id))  # type: ignore[arg-type]
    assert entity.id == namespace_id
    row = await pg_pool.fetchrow("SELECT * FROM namespaces WHERE namespace_id = $1", namespace_id)
    assert row is not None
    assert row["row_scope"] == "customer"
    assert (row["namespace_type"], row["owner_agent_id"], row["customer_id"]) == ("memory", agent, customer)
    assert row["owner_namespace"] == "agent-owner"
    assert row["schema_name"] == f"memory__{namespace_id.hex}"


async def test_ensure_is_idempotent_and_concurrent_ensures_converge(pg_pool: asyncpg.Pool) -> None:
    customer, agent, namespace_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await _owner_row(pg_pool, "agent-owner", customer)
    fields = _fields(customer, agent, namespace_id)
    pods = [_collection(pg_pool) for _ in range(4)]
    results = await asyncio.gather(*(pod.ensure_namespace(**fields) for pod in pods))  # type: ignore[arg-type]
    assert {r.id for r in results} == {namespace_id}
    again = await pods[0].ensure_namespace(**fields)  # type: ignore[arg-type]
    assert again.id == namespace_id
    count = await pg_pool.fetchval("SELECT count(*) FROM namespaces WHERE namespace_type = 'memory'")
    assert count == 1


async def test_ensure_refuses_a_row_that_already_means_something_else(pg_pool: asyncpg.Pool) -> None:
    customer, agent, namespace_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await _owner_row(pg_pool, "agent-owner", customer)
    collection = _collection(pg_pool)
    await collection.ensure_namespace(**_fields(customer, agent, namespace_id))  # type: ignore[arg-type]
    other_customer = uuid.uuid4()
    with pytest.raises(ValueError, match="already exists"):
        await collection.ensure_namespace(**{**_fields(customer, agent, namespace_id), "customer_id": other_customer})  # type: ignore[arg-type]


async def test_racing_ensures_never_raise_under_load(pg_pool: asyncpg.Pool) -> None:
    """One gather of four hid it: the loser of a real race hit the non-arbiter UNIQUE indexes."""
    customer, agent = uuid.uuid4(), uuid.uuid4()
    await _owner_row(pg_pool, "agent-owner", customer)
    pods = [_collection(pg_pool) for _ in range(8)]
    for _ in range(200):
        fields = _fields(customer, agent, uuid.uuid4())
        fields["name"] = f"ns-{uuid.uuid4().hex}"
        results = await asyncio.gather(*(pod.ensure_namespace(**fields) for pod in pods), return_exceptions=True)  # type: ignore[arg-type]
        errors = [r for r in results if isinstance(r, BaseException)]
        assert errors == [], errors[:1]


async def test_ensure_refuses_a_row_whose_other_fields_disagree(pg_pool: asyncpg.Pool) -> None:
    customer, agent, namespace_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await _owner_row(pg_pool, "agent-owner", customer)
    collection = _collection(pg_pool)
    await collection.ensure_namespace(**{**_fields(customer, agent, namespace_id), "owner_namespace": None})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="already exists"):
        await collection.ensure_namespace(**_fields(customer, agent, namespace_id))  # type: ignore[arg-type]
    row = await pg_pool.fetchrow("SELECT owner_namespace FROM namespaces WHERE namespace_id = $1", namespace_id)
    assert row is not None and row["owner_namespace"] is None, "an existing row is never overwritten"


async def test_a_name_taken_by_another_row_is_refused(pg_pool: asyncpg.Pool) -> None:
    customer, agent = uuid.uuid4(), uuid.uuid4()
    await _owner_row(pg_pool, "agent-owner", customer)
    collection = _collection(pg_pool)
    first = _fields(customer, agent, uuid.uuid4())
    await collection.ensure_namespace(**first)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="already taken"):
        await collection.ensure_namespace(**{**first, "namespace_id": uuid.uuid4()})  # type: ignore[arg-type]
