"""``NamespaceCollection.ensure_namespace``: get-or-create a namespace row on a deterministic id.

The hub materializes a memory namespace on request; a deployment with no hub (one application that owns
its own control plane) had no supported way to, and wrote the row with raw SQL -- leaving
``owner_namespace`` NULL, so no agent owned it. This is that write, once, through the collection:
idempotent, convergent across pods on a deterministic id, and refusing a row that already means
something else. Measured against the real table shape, including the ``owner_namespace_id`` foreign key.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import asyncpg
import pytest
import pytest_asyncio

from threetears.agent.acl import NamespaceCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.ddl.row_epoch import next_epoch

from .access_schema import access_pool, access_schema, empty_tables, owner_id_key

# one schema the acl migrations built, for the module's tests (each starts from empty tables)
pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def _module_schema(db_container: str) -> AsyncIterator[str]:
    """a schema of the module's own holding the namespaces the acl migrations build, with the hub's owner_namespace
    foreign key; built once, as a build is slow on YugabyteDB. Its name.
    """
    async with access_schema(db_container, "nsensure") as schema:
        pool = await access_pool(db_container, schema, max_size=1)
        try:
            await owner_id_key(pool)
        finally:
            await pool.close()
        yield schema


@pytest_asyncio.fixture(loop_scope="module")
async def pg_pool(db_container: str, _module_schema: str) -> AsyncIterator[asyncpg.Pool]:
    """a pool on the module's schema, its access tables emptied for this test."""
    pool = await access_pool(db_container, _module_schema, max_size=4)
    try:
        await empty_tables(pool)
        yield pool
    finally:
        await pool.close()


def _collection(pool: asyncpg.Pool) -> NamespaceCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool, kv_key_scope=f"ensure-{uuid.uuid4().hex[:6]}")
    return NamespaceCollection(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""))


async def _owner_row(pool: asyncpg.Pool, name: str, customer_id: uuid.UUID) -> uuid.UUID:
    """the owning agent's own namespace, which the owner_namespace_id foreign key names; its id."""
    now = datetime.now(UTC)
    owner_id = uuid.uuid4()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, customer_id, "
            "date_created, date_updated, epoch) VALUES ('customer', $1, $2, 'agent', $3, $4, $4, $5)",
            owner_id,
            name,
            customer_id,
            now,
            next_epoch(),
        )
    return owner_id


def _fields(customer: uuid.UUID, agent: uuid.UUID, namespace_id: uuid.UUID, owner: uuid.UUID) -> dict[str, object]:
    return {
        "namespace_id": namespace_id,
        "name": f"memories.{agent.hex}.{customer.hex}",
        "namespace_type": "memory",
        "owner_agent_id": agent,
        "customer_id": customer,
        "owner_namespace": "agent-owner",
        "owner_namespace_id": owner,
        "schema_name": f"memory__{namespace_id.hex}",
    }


async def test_ensure_creates_the_row_with_every_field(pg_pool: asyncpg.Pool) -> None:
    customer, agent, namespace_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    owner = await _owner_row(pg_pool, "agent-owner", customer)
    entity = await _collection(pg_pool).ensure_namespace(**_fields(customer, agent, namespace_id, owner))  # type: ignore[arg-type]
    assert entity.id == namespace_id
    row = await pg_pool.fetchrow("SELECT * FROM namespaces WHERE namespace_id = $1", namespace_id)
    assert row is not None
    assert row["row_scope"] == "customer"
    assert (row["namespace_type"], row["owner_agent_id"], row["customer_id"]) == ("memory", agent, customer)
    assert row["owner_namespace"] == "agent-owner"
    assert row["owner_namespace_id"] == owner
    assert row["schema_name"] == f"memory__{namespace_id.hex}"


async def test_ensure_is_idempotent_and_concurrent_ensures_converge(pg_pool: asyncpg.Pool) -> None:
    customer, agent, namespace_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    owner = await _owner_row(pg_pool, "agent-owner", customer)
    fields = _fields(customer, agent, namespace_id, owner)
    pods = [_collection(pg_pool) for _ in range(4)]
    results = await asyncio.gather(*(pod.ensure_namespace(**fields) for pod in pods))  # type: ignore[arg-type]
    assert {r.id for r in results} == {namespace_id}
    again = await pods[0].ensure_namespace(**fields)  # type: ignore[arg-type]
    assert again.id == namespace_id
    count = await pg_pool.fetchval("SELECT count(*) FROM namespaces WHERE namespace_type = 'memory'")
    assert count == 1


async def test_ensure_refuses_a_row_that_already_means_something_else(pg_pool: asyncpg.Pool) -> None:
    customer, agent, namespace_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    owner = await _owner_row(pg_pool, "agent-owner", customer)
    collection = _collection(pg_pool)
    await collection.ensure_namespace(**_fields(customer, agent, namespace_id, owner))  # type: ignore[arg-type]
    other = {**_fields(customer, agent, namespace_id, owner), "customer_id": uuid.uuid4()}
    with pytest.raises(ValueError, match="already exists"):
        await collection.ensure_namespace(**other)  # type: ignore[arg-type]


#: racing ensures of one namespace (a replica each), and the namespaces raced at once, each a lane of rounds
_RACERS = 8
_LANES = 6
_ROUNDS = 200


async def test_racing_ensures_never_raise_under_load(
    pg_pool: asyncpg.Pool, db_container: str, _module_schema: str
) -> None:
    """One gather of four hid it: the loser of a real race hit the non-arbiter UNIQUE indexes.

    Flips if a racing insert's YugabyteDB deadlock or conflict reaches the caller: one statement writes the row and each
    index entry to their own tablets at once, so racers deadlock (40P01) or conflict (40001), which YugabyteDB asks the
    client to retry. One lane of 200 rounds met it twice in a day of the suite's runs under load; six lanes at once,
    on a pool of a connection a racer, met it 113 times in 23,280 rounds (2026-11-26), so 1,200 rounds meet it nearly
    always on the old code. PostgreSQL never raises it.
    """
    customer, agent = uuid.uuid4(), uuid.uuid4()
    owner = await _owner_row(pg_pool, "agent-owner", customer)
    pool = await access_pool(db_container, _module_schema, max_size=_RACERS * _LANES)
    try:
        pods = [_collection(pool) for _ in range(_RACERS)]

        async def lane() -> list[BaseException]:
            for _ in range(_ROUNDS):
                fields = _fields(customer, agent, uuid.uuid4(), owner)
                fields["name"] = f"ns-{uuid.uuid4().hex}"
                ensures = [pod.ensure_namespace(**fields) for pod in pods]  # type: ignore[arg-type]
                results = await asyncio.gather(*ensures, return_exceptions=True)
                errors = [r for r in results if isinstance(r, BaseException)]
                if errors:
                    return errors
            return []

        errors = [error for found in await asyncio.gather(*(lane() for _ in range(_LANES))) for error in found]
        assert errors == [], errors[:1]
    finally:
        await pool.close()


#: rounds of racing re-ensures of a namespace a delete marked gone
_REVIVE_ROUNDS = 20


async def test_racing_revives_of_a_gone_namespace_never_raise(
    pg_pool: asyncpg.Pool, db_container: str, _module_schema: str
) -> None:
    """Flips if a racing revive's loser reads its empty revive as a unique value taken by another id.

    Every replica re-ensures a namespace a delete marked gone at once. One replica's revive makes the row live; every
    other's keyed ``UPDATE ... WHERE gone`` then matches nothing (on PostgreSQL it waits on the winner's lock and
    re-checks ``gone``; on YugabyteDB it is retried after a conflict and finds the row live). A loser that skipped the
    read-back raised "already taken by a row with another id" for a row that is live under its own id.
    """
    customer, agent = uuid.uuid4(), uuid.uuid4()
    owner = await _owner_row(pg_pool, "agent-owner", customer)
    pool = await access_pool(db_container, _module_schema, max_size=_RACERS)
    try:
        for _ in range(_REVIVE_ROUNDS):
            namespace_id = uuid.uuid4()
            fields = _fields(customer, agent, namespace_id, owner)
            fields["name"] = f"ns-{namespace_id.hex}"
            await _collection(pool).ensure_namespace(**fields)  # type: ignore[arg-type]
            # a delete marks the row gone, never deletes it (ROW_STAMPED)
            await pool.execute(
                "UPDATE namespaces SET gone = true, epoch = $2 WHERE namespace_id = $1", namespace_id, next_epoch()
            )
            pods = [_collection(pool) for _ in range(_RACERS)]
            results = await asyncio.gather(
                *(pod.ensure_namespace(**fields) for pod in pods),  # type: ignore[arg-type]
                return_exceptions=True,
            )
            errors = [r for r in results if isinstance(r, BaseException)]
            assert errors == [], errors[:1]
            assert {r.id for r in results if not isinstance(r, BaseException)} == {namespace_id}
            live = await pool.fetchval("SELECT NOT gone FROM namespaces WHERE namespace_id = $1", namespace_id)
            assert live is True
    finally:
        await pool.close()


async def test_ensure_refuses_a_row_whose_other_fields_disagree(pg_pool: asyncpg.Pool) -> None:
    customer, agent, namespace_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    owner = await _owner_row(pg_pool, "agent-owner", customer)
    collection = _collection(pg_pool)
    await collection.ensure_namespace(**{**_fields(customer, agent, namespace_id, owner), "owner_namespace": None})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="already exists"):
        await collection.ensure_namespace(**_fields(customer, agent, namespace_id, owner))  # type: ignore[arg-type]
    row = await pg_pool.fetchrow("SELECT owner_namespace FROM namespaces WHERE namespace_id = $1", namespace_id)
    assert row is not None and row["owner_namespace"] is None, "an existing row is never overwritten"


async def test_a_name_taken_by_another_row_is_refused(pg_pool: asyncpg.Pool) -> None:
    customer, agent = uuid.uuid4(), uuid.uuid4()
    owner = await _owner_row(pg_pool, "agent-owner", customer)
    collection = _collection(pg_pool)
    first = _fields(customer, agent, uuid.uuid4(), owner)
    await collection.ensure_namespace(**first)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="already taken"):
        await collection.ensure_namespace(**{**first, "namespace_id": uuid.uuid4()})  # type: ignore[arg-type]


async def test_ensure_refuses_a_row_owned_by_another_owner_id(pg_pool: asyncpg.Pool) -> None:
    """Flips if ensure stops comparing the stored owner (by id): a row of the same owner name under another owner row
    is something else."""
    customer, agent, namespace_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    owner = await _owner_row(pg_pool, "agent-owner", customer)
    other = await _owner_row(pg_pool, "agent-other", customer)
    collection = _collection(pg_pool)
    await collection.ensure_namespace(**_fields(customer, agent, namespace_id, owner))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="already exists"):
        await collection.ensure_namespace(**_fields(customer, agent, namespace_id, other))  # type: ignore[arg-type]


async def test_an_owner_id_naming_no_row_is_refused(pg_pool: asyncpg.Pool) -> None:
    """Flips if the hub's shape stops keying the owner by id: an owner id naming no row is refused by its foreign key."""
    customer, agent = uuid.uuid4(), uuid.uuid4()
    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await _collection(pg_pool).ensure_namespace(**_fields(customer, agent, uuid.uuid4(), uuid.uuid4()))  # type: ignore[arg-type]
