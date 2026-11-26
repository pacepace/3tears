"""the rows a namespace owns, against a real database.

A namespace's ``owner_namespace_id`` names another row by ID, through a foreign
key onto ``namespace_id`` (hub v199; its ``owner_namespace`` beside it is the
owner's name, the evaluator's key). Deleting an owner while anything still names it
is refused by that key -- so tearing a namespace down means finding everything it
owns and removing that first. ``NamespaceCollection.list_owned_by`` is the finding
half. Measured here rather than against a double, because both the partition
spread of the children and the foreign key's refusal are properties of the real
table.
"""

from __future__ import annotations

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
    async with access_schema(db_container, "nsowned") as schema:
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
    """a ``NamespaceCollection`` reading the real table, with no cache tiers in the way."""
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool, kv_key_scope="owned-by-test")
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return NamespaceCollection(registry, config)


async def _insert(
    pool: asyncpg.Pool,
    *,
    name: str,
    namespace_type: str,
    owner_namespace: str | None,
    customer_id: uuid.UUID | None,
    schema_name: str | None = None,
    owner_namespace_id: uuid.UUID | None = None,
) -> uuid.UUID:
    """write one namespace row the way the hub's emitters leave it."""
    namespace_id = uuid.uuid4()
    now = datetime.now(UTC)
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, owner_namespace,"
            " customer_id, schema_name, date_created, date_updated, epoch, owner_namespace_id)"
            " VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $8, $9, $10)",
            "platform" if customer_id is None else "customer",
            namespace_id,
            name,
            namespace_type,
            owner_namespace,
            customer_id,
            schema_name,
            now,
            next_epoch(),
            owner_namespace_id,
        )
    return namespace_id


async def _agent_with_children(pool: asyncpg.Pool, agent_id: uuid.UUID, customer_id: uuid.UUID) -> dict[str, uuid.UUID]:
    """an agent namespace that owns itself, a customer-scope channel and memory, and a platform-scope tool."""
    agent_ns = f"agents.{agent_id}"
    ids = {
        "agent": await _insert(
            pool, name=agent_ns, namespace_type="agent", owner_namespace=None, customer_id=customer_id
        )
    }
    async with pool.acquire() as conn:
        # an agent's namespace names itself as its owner: set after the insert, as
        # the foreign key refuses an owner that does not exist yet.
        await conn.execute(
            "UPDATE namespaces SET owner_namespace = name, owner_namespace_id = namespace_id, epoch = $2"
            " WHERE name = $1",
            agent_ns,
            next_epoch(),
        )
    owner = ids["agent"]
    ids["channel"] = await _insert(
        pool,
        name=f"channels.web.{agent_id.hex}",
        namespace_type="channel",
        owner_namespace=agent_ns,
        owner_namespace_id=owner,
        customer_id=customer_id,
    )
    ids["memory"] = await _insert(
        pool,
        name=f"memories.{agent_id}.{customer_id}",
        namespace_type="memory",
        owner_namespace=agent_ns,
        owner_namespace_id=owner,
        customer_id=customer_id,
    )
    ids["tool"] = await _insert(
        pool,
        name=f"tools.{agent_id.hex}.probe.1-0",
        namespace_type="tool",
        owner_namespace=agent_ns,
        owner_namespace_id=owner,
        customer_id=None,
    )
    return ids


class TestListOwnedBy:
    """``list_owned_by`` returns exactly the rows that name the owner, in both partitions."""

    async def test_it_returns_every_child_across_both_partitions_and_not_the_owner(self, pg_pool: asyncpg.Pool) -> None:
        """the owner's self-reference is not a child: a walker that followed it would never finish."""
        agent_id, customer_id = uuid.uuid4(), uuid.uuid4()
        ids = await _agent_with_children(pg_pool, agent_id, customer_id)

        owned = await _collection(pg_pool).list_owned_by(ids["agent"])

        assert {entity.id for entity in owned} == {ids["channel"], ids["memory"], ids["tool"]}

    async def test_it_does_not_return_another_owners_children(self, pg_pool: asyncpg.Pool) -> None:
        """a second agent's rows are not the first's, even under one customer."""
        customer_id = uuid.uuid4()
        first, second = uuid.uuid4(), uuid.uuid4()
        await _agent_with_children(pg_pool, first, customer_id)
        second_ids = await _agent_with_children(pg_pool, second, customer_id)

        owned = await _collection(pg_pool).list_owned_by(second_ids["agent"])

        assert {entity.id for entity in owned} == {second_ids["channel"], second_ids["memory"], second_ids["tool"]}

    async def test_a_row_naming_the_owner_by_name_alone_is_not_its_child(self, pg_pool: asyncpg.Pool) -> None:
        """the owner is stored by id: a row carrying the owner's name and no id is not read as owned by it.

        Seeded with exactly that row beside a real child, so a read by the owner's name WOULD return it.
        """
        agent_id, customer_id = uuid.uuid4(), uuid.uuid4()
        ids = await _agent_with_children(pg_pool, agent_id, customer_id)
        await _insert(
            pg_pool,
            name=f"channels.web.{agent_id.hex}.unkeyed",
            namespace_type="channel",
            owner_namespace=f"agents.{agent_id}",
            customer_id=customer_id,
        )

        owned = await _collection(pg_pool).list_owned_by(ids["agent"])

        assert {entity.id for entity in owned} == {ids["channel"], ids["memory"], ids["tool"]}

    async def test_an_owner_with_no_children_owns_nothing(self, pg_pool: asyncpg.Pool) -> None:
        """a leaf answers an empty list rather than raising."""
        agent_id, customer_id = uuid.uuid4(), uuid.uuid4()
        ids = await _agent_with_children(pg_pool, agent_id, customer_id)

        assert await _collection(pg_pool).list_owned_by(ids["channel"]) == []


class TestWhyTheListExists:
    """the foreign key refuses to delete an owner while a child still names it."""

    async def test_deleting_an_owner_before_its_children_is_refused(self, pg_pool: asyncpg.Pool) -> None:
        """the parent side IS enforced for DELETE, here as on YugabyteDB.

        What YugabyteDB leaves unenforced is a parent RENAME (per the hub's v089 migration);
        Postgres refuses that too. This test measures the delete only.
        """
        agent_id, customer_id = uuid.uuid4(), uuid.uuid4()
        await _agent_with_children(pg_pool, agent_id, customer_id)

        async with pg_pool.acquire() as conn:
            with pytest.raises(asyncpg.exceptions.ForeignKeyViolationError):
                await conn.execute("DELETE FROM namespaces WHERE name = $1", f"agents.{agent_id}")

    async def test_the_owner_deletes_once_its_children_are_gone(self, pg_pool: asyncpg.Pool) -> None:
        """leaves first, then the owner: its own self-reference does not block it."""
        agent_id, customer_id = uuid.uuid4(), uuid.uuid4()
        ids = await _agent_with_children(pg_pool, agent_id, customer_id)
        collection = _collection(pg_pool)

        async with pg_pool.acquire() as conn:
            for child in await collection.list_owned_by(ids["agent"]):
                await conn.execute("DELETE FROM namespaces WHERE namespace_id = $1", child.id)
            await conn.execute("DELETE FROM namespaces WHERE name = $1", f"agents.{agent_id}")
            remaining = await conn.fetchval("SELECT count(*) FROM namespaces")

        assert remaining == 0


class TestSchemaInUse:
    """``schema_in_use`` says whether any row still names a schema.

    A workspace namespace records its AGENT's schema as its own ``schema_name``,
    so one schema can be named by several rows. Whoever deletes a row may drop its
    schema only once no row names it any more.
    """

    async def test_a_schema_a_remaining_row_names_is_in_use(self, pg_pool: asyncpg.Pool) -> None:
        """the agent still names the schema its workspace shared."""
        agent_id, customer_id = uuid.uuid4(), uuid.uuid4()
        schema = f"agent_{agent_id.hex}"
        await _agent_with_children(pg_pool, agent_id, customer_id)
        workspace_id = await _insert(
            pg_pool,
            name=f"workspaces.{agent_id.hex}.notes",
            namespace_type="workspace",
            owner_namespace=f"agents.{agent_id}",
            customer_id=customer_id,
        )
        async with pg_pool.acquire() as conn:
            await conn.execute(
                "UPDATE namespaces SET schema_name = $1, epoch = $4 WHERE namespace_id = $2 OR name = $3",
                schema,
                workspace_id,
                f"agents.{agent_id}",
                next_epoch(),
            )
            await conn.execute("DELETE FROM namespaces WHERE namespace_id = $1", workspace_id)

        assert await _collection(pg_pool).schema_in_use(schema) is True

    async def test_a_schema_no_row_ever_named_is_free(self, pg_pool: asyncpg.Pool) -> None:
        """a schema nothing names is free to drop."""
        await _agent_with_children(pg_pool, uuid.uuid4(), uuid.uuid4())

        assert await _collection(pg_pool).schema_in_use(f"agent_{uuid.uuid4().hex}") is False

    async def test_the_schema_is_free_once_the_last_row_naming_it_is_gone(self, pg_pool: asyncpg.Pool) -> None:
        """deleting one of two rows leaves the schema in use; deleting the last frees it."""
        agent_id, customer_id = uuid.uuid4(), uuid.uuid4()
        agent_ns = f"agents.{agent_id}"
        schema = f"agent_{agent_id.hex}"
        await _insert(
            pg_pool,
            name=agent_ns,
            namespace_type="agent",
            owner_namespace=None,
            customer_id=customer_id,
            schema_name=schema,
        )
        workspace_id = await _insert(
            pg_pool,
            name=f"workspaces.{agent_id.hex}.notes",
            namespace_type="workspace",
            owner_namespace=agent_ns,
            customer_id=customer_id,
            schema_name=schema,
        )
        collection = _collection(pg_pool)

        async with pg_pool.acquire() as conn:
            await conn.execute("DELETE FROM namespaces WHERE namespace_id = $1", workspace_id)
        after_first = await collection.schema_in_use(schema)
        async with pg_pool.acquire() as conn:
            await conn.execute("DELETE FROM namespaces WHERE name = $1", agent_ns)
        after_last = await collection.schema_in_use(schema)

        assert after_first is True
        assert after_last is False

    async def test_a_platform_row_naming_the_schema_keeps_it_in_use(self, pg_pool: asyncpg.Pool) -> None:
        """the check spans both partitions: a platform row names a schema as surely as a customer row.

        The v052 system namespace is platform-scoped and carries a schema, so a check that
        read only the customer partition would report that schema free to drop.
        """
        await _insert(
            pg_pool,
            name="system",
            namespace_type="system",
            owner_namespace=None,
            customer_id=None,
            schema_name="platform",
        )

        assert await _collection(pg_pool).schema_in_use("platform") is True

    async def test_an_empty_schema_name_is_never_in_use(self, pg_pool: asyncpg.Pool) -> None:
        """an empty name is never in use, even where a row really carries the empty schema name.

        Seeded with exactly that row, so the query WOULD find it: this fails if the
        empty-name guard is removed, where a table holding only NULL schemas could not tell.
        """
        await _insert(
            pg_pool,
            name="shared.blank",
            namespace_type="shared",
            owner_namespace=None,
            customer_id=None,
            schema_name="",
        )

        assert await _collection(pg_pool).schema_in_use("") is False
