"""the rows a namespace owns, against a real database.

A namespace's ``owner_namespace`` names another row by NAME, through a foreign
key onto the unique name index. Deleting an owner while anything still names it
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

from threetears.agent.acl import NamespaceCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

pytestmark = pytest.mark.integration


#: the real shape: composite primary key, the partition CHECK, the
#: ``owner_namespace`` foreign key onto the unique name index (v089), and the
#: partial unique index that lets only workspace rows share a schema name (the
#: hub's ``idx_namespaces_schema_name_non_workspace``, applied in the fixture).
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
    """per-test pool over a fresh ``namespaces`` table carrying the owner foreign key."""
    from threetears.core.collections import init_connection

    pool: asyncpg.Pool = await asyncpg.create_pool(
        db_container,
        min_size=1,
        max_size=4,
        init=init_connection,
    )
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
) -> uuid.UUID:
    """write one namespace row the way the hub's emitters leave it."""
    namespace_id = uuid.uuid4()
    now = datetime.now(UTC)
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, owner_namespace,"
            " customer_id, schema_name, date_created, date_updated) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $8)",
            "platform" if customer_id is None else "customer",
            namespace_id,
            name,
            namespace_type,
            owner_namespace,
            customer_id,
            schema_name,
            now,
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
        await conn.execute("UPDATE namespaces SET owner_namespace = name WHERE name = $1", agent_ns)
    ids["channel"] = await _insert(
        pool,
        name=f"channels.web.{agent_id.hex}",
        namespace_type="channel",
        owner_namespace=agent_ns,
        customer_id=customer_id,
    )
    ids["memory"] = await _insert(
        pool,
        name=f"memories.{agent_id}.{customer_id}",
        namespace_type="memory",
        owner_namespace=agent_ns,
        customer_id=customer_id,
    )
    ids["tool"] = await _insert(
        pool, name=f"tools.{agent_id.hex}.probe.1-0", namespace_type="tool", owner_namespace=agent_ns, customer_id=None
    )
    return ids


class TestListOwnedBy:
    """``list_owned_by`` returns exactly the rows that name the owner, in both partitions."""

    async def test_it_returns_every_child_across_both_partitions_and_not_the_owner(self, pg_pool: asyncpg.Pool) -> None:
        """the owner's self-reference is not a child: a walker that followed it would never finish."""
        agent_id, customer_id = uuid.uuid4(), uuid.uuid4()
        ids = await _agent_with_children(pg_pool, agent_id, customer_id)

        owned = await _collection(pg_pool).list_owned_by(f"agents.{agent_id}")

        assert {entity.id for entity in owned} == {ids["channel"], ids["memory"], ids["tool"]}

    async def test_it_does_not_return_another_owners_children(self, pg_pool: asyncpg.Pool) -> None:
        """a second agent's rows are not the first's, even under one customer."""
        customer_id = uuid.uuid4()
        first, second = uuid.uuid4(), uuid.uuid4()
        await _agent_with_children(pg_pool, first, customer_id)
        second_ids = await _agent_with_children(pg_pool, second, customer_id)

        owned = await _collection(pg_pool).list_owned_by(f"agents.{second}")

        assert {entity.id for entity in owned} == {second_ids["channel"], second_ids["memory"], second_ids["tool"]}

    async def test_an_empty_owner_owns_nothing(self, pg_pool: asyncpg.Pool) -> None:
        """an empty name owns nothing, even where a row really is owned by the empty name.

        The table is seeded with exactly that row, so the query WOULD return it: this
        fails if the empty-name guard is removed, where a table holding only NULL owners
        could not tell.
        """
        customer_id = uuid.uuid4()
        await _insert(pg_pool, name="", namespace_type="agent", owner_namespace=None, customer_id=customer_id)
        await _insert(
            pg_pool, name="channels.web.orphan", namespace_type="channel", owner_namespace="", customer_id=customer_id
        )

        assert await _collection(pg_pool).list_owned_by("") == []

    async def test_an_owner_with_no_children_owns_nothing(self, pg_pool: asyncpg.Pool) -> None:
        """a leaf answers an empty list rather than raising."""
        agent_id, customer_id = uuid.uuid4(), uuid.uuid4()
        await _agent_with_children(pg_pool, agent_id, customer_id)

        assert await _collection(pg_pool).list_owned_by(f"channels.web.{agent_id.hex}") == []


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
        await _agent_with_children(pg_pool, agent_id, customer_id)
        collection = _collection(pg_pool)

        async with pg_pool.acquire() as conn:
            for child in await collection.list_owned_by(f"agents.{agent_id}"):
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
                "UPDATE namespaces SET schema_name = $1 WHERE namespace_id = $2 OR name = $3",
                schema,
                workspace_id,
                f"agents.{agent_id}",
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
