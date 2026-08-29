"""a namespace row that gains (or loses) a customer, against a real database.

``namespaces`` is keyed ``(row_scope, namespace_id)`` and ``row_scope`` is
DERIVED from ``customer_id`` (``platform`` <-> ``customer_id IS NULL``), so a
row that gains a customer changes its own PRIMARY KEY. The generated upsert
nominates ``(row_scope, namespace_id)`` as its conflict arbiter and therefore
does not match the existing row -- while the separate ``UNIQUE (namespace_id)``
index DOES, which turns the write into an unretryable
:class:`asyncpg.exceptions.UniqueViolationError`.

That is not reproducible against a double. Every assertion here runs against a
real Postgres carrying the real composite primary key, the real secondary
unique index, and the real ``role_assignments.scope_namespace_id`` foreign key
with ``ON DELETE CASCADE`` -- which is what makes the "grants survive the move"
claim a measurement rather than a hope.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest
from sqlalchemy import MetaData

from threetears.agent.acl import NamespaceCollection
from threetears.agent.acl.collections import NamespaceRescopeRefused
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

pytestmark = pytest.mark.integration


#: the real shape, not a paraphrase: composite primary key on
#: ``(row_scope, namespace_id)``, a SEPARATE unique index on ``namespace_id``
#: alone, and the derived-partition CHECK that ties ``row_scope`` to whether
#: ``customer_id`` is null. Take any one of the three away and the hazard this
#: module is about stops reproducing.
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

_ROLE_ASSIGNMENTS_DDL = """
CREATE TABLE role_assignments (
    row_scope varchar(8) NOT NULL,
    assignment_id uuid NOT NULL,
    scope_namespace_id uuid,
    managed_by varchar(32),
    CONSTRAINT role_assignments_pkey PRIMARY KEY (row_scope, assignment_id),
    CONSTRAINT role_assignments_scope_namespace_id_fkey
        FOREIGN KEY (scope_namespace_id) REFERENCES namespaces(namespace_id) ON DELETE CASCADE
)
"""


@pytest.fixture
async def pg_pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    """per-test pool over a fresh ``namespaces`` + ``role_assignments`` pair."""
    from threetears.core.collections import init_connection

    pool: asyncpg.Pool = await asyncpg.create_pool(
        db_container,
        min_size=1,
        max_size=4,
        init=init_connection,
    )
    try:
        async with pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS role_assignments")
            await conn.execute("DROP TABLE IF EXISTS namespaces")
            await conn.execute(_NAMESPACES_DDL)
            await conn.execute("CREATE UNIQUE INDEX namespaces_id_unique ON namespaces (namespace_id)")
            await conn.execute("CREATE UNIQUE INDEX idx_namespaces_name ON namespaces (name)")
            await conn.execute(_ROLE_ASSIGNMENTS_DDL)
        yield pool
    finally:
        async with pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS role_assignments")
            await conn.execute("DROP TABLE IF EXISTS namespaces")
        await pool.close()


class _InMemoryKvBucket:
    """typed-wrapper KV bucket stand-in, so L2 eviction is observable.

    mirrors the three methods :class:`~threetears.core.collections.base
    .BaseCollection` calls on a bucket handle (``get`` / ``put`` / ``delete``);
    the same shape ``packages/core/tests/integration`` uses.
    """

    def __init__(self) -> None:
        self.kv: dict[str, bytes] = {}

    async def get(self, *, key: str) -> bytes | None:
        return self.kv.get(key)

    async def put(self, *, key: str, value: bytes) -> int:
        self.kv[key] = value
        return len(self.kv)

    async def delete(self, *, key: str, revision: int | None = None) -> bool:  # noqa: ARG002
        existed = key in self.kv
        self.kv.pop(key, None)
        return existed


class _InMemoryNats:
    """typed-wrapper NATS stand-in: one KV bucket, publishes go nowhere.

    the invalidation publish is not what this suite is about -- the L2 delete
    is -- so it is accepted and dropped.
    """

    def __init__(self) -> None:
        self._bucket = _InMemoryKvBucket()

    @property
    def kv(self) -> dict[str, bytes]:
        return self._bucket.kv

    async def kv_bucket(self, **_kwargs: Any) -> _InMemoryKvBucket:
        return self._bucket

    async def publish(self, **_kwargs: Any) -> None:
        return None

    async def subscribe_typed(self, **_kwargs: Any) -> None:
        return None


def _build_collection(
    pool: asyncpg.Pool,
) -> tuple[NamespaceCollection, _InMemoryNats, SQLiteBackend]:
    """wire a real ``NamespaceCollection`` over L1 + L2 + the real table."""
    l1 = SQLiteBackend(db_name=f"acl_{uuid.uuid4().hex[:8]}")
    metadata = MetaData()
    NamespaceCollection.schema.to_sqlalchemy_table(metadata)
    l1.initialize(metadata)
    nats = _InMemoryNats()
    registry = CollectionRegistry()
    registry.configure(
        l1_backend=l1,
        l2_client=nats,
        l3_pool=pool,
        kv_key_scope="rescope-test",
    )
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    collection = NamespaceCollection(registry, config, nats_client=nats)
    return collection, nats, l1


async def _seed_platform_tool(
    pool: asyncpg.Pool,
    *,
    namespace_id: uuid.UUID,
    name: str = "tools.probe.example.1-0",
) -> None:
    """insert one platform-scoped ``tool`` row the way a pre-upgrade pod left it."""
    now = datetime.now(UTC)
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO namespaces"
            " (row_scope, namespace_id, name, namespace_type, customer_id, date_created, date_updated)"
            " VALUES ('platform', $1, $2, 'tool', NULL, $3, $3)",
            namespace_id,
            name,
            now,
        )


async def _grant_on(pool: asyncpg.Pool, namespace_id: uuid.UUID, *, managed_by: str = "manual") -> uuid.UUID:
    """bind one namespace-scoped assignment to the row, as an operator's grant."""
    assignment_id = uuid.uuid4()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO role_assignments (row_scope, assignment_id, scope_namespace_id, managed_by)"
            " VALUES ('platform', $1, $2, $3)",
            assignment_id,
            namespace_id,
            managed_by,
        )
    return assignment_id


def _tool_row(namespace_id: uuid.UUID, customer_id: uuid.UUID | None, name: str) -> dict[str, Any]:
    """the payload the tool-namespace emitter builds for one tool."""
    now = datetime.now(UTC)
    return {
        "namespace_id": namespace_id,
        "name": name,
        "namespace_type": "tool",
        "owner_agent_id": None,
        "owner_namespace": None,
        "customer_id": customer_id,
        "schema_name": None,
        "metadata": {"mcp_name": "probe.example", "mcp_version": "1.0"},
        "face_api": False,
        "face_mcp": False,
        "face_platform_tool": True,
        "face_rest": False,
        "face_rest_declaration": None,
        "date_created": now,
        "date_updated": now,
    }


class TestTheHazardItself:
    """what an unaided upsert does to a row whose derived partition moved."""

    async def test_the_ordinary_upsert_cannot_move_a_row_and_raises(
        self,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """a row registered platform-scoped, re-emitted WITH a customer, refuses.

        this is the bug: the conflict arbiter is the composite key, which no
        longer matches, while the bare-``namespace_id`` unique index does.
        """
        collection, _nats, l1 = _build_collection(pg_pool)
        namespace_id = uuid.uuid4()
        customer_id = uuid.uuid4()
        try:
            await _seed_platform_tool(pg_pool, namespace_id=namespace_id)
            entity = collection.entity_class(
                _tool_row(namespace_id, customer_id, "tools.probe.example.1-0"),
                is_new=True,
                collection=collection,
            )
            with pytest.raises(asyncpg.exceptions.UniqueViolationError):
                await collection.save_entity(entity)
        finally:
            l1.reset()

    async def test_deleting_the_row_first_would_destroy_every_grant_on_it(
        self,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """the admitted twin of the fix: why delete-and-reinsert is not it.

        ``role_assignments.scope_namespace_id`` carries ``ON DELETE CASCADE``,
        so the obvious "remove it and write it again under the new scope" loses
        every grant the row carried -- silently, and including the operator
        grants nothing rebuilds.
        """
        namespace_id = uuid.uuid4()
        await _seed_platform_tool(pg_pool, namespace_id=namespace_id)
        await _grant_on(pg_pool, namespace_id)
        async with pg_pool.acquire() as conn:
            before = await conn.fetchval("SELECT count(*) FROM role_assignments")
            await conn.execute("DELETE FROM namespaces WHERE namespace_id = $1", namespace_id)
            after = await conn.fetchval("SELECT count(*) FROM role_assignments")
        assert before == 1
        assert after == 0


class TestRescopeMovesTheRow:
    """the deliberate move, and what it does and does not take with it."""

    async def test_a_platform_row_gains_a_customer(self, pg_pool: asyncpg.Pool) -> None:
        """platform -> customer: the row moves partition and keeps its id."""
        collection, _nats, l1 = _build_collection(pg_pool)
        namespace_id = uuid.uuid4()
        customer_id = uuid.uuid4()
        try:
            await _seed_platform_tool(pg_pool, namespace_id=namespace_id)
            outcome = await collection.rescope(namespace_id, customer_id=customer_id)
            assert outcome.moved is True
            assert outcome.previous_row_scope == "platform"
            assert outcome.previous_customer_id is None
            assert outcome.row_scope == "customer"
            assert outcome.customer_id == customer_id
            async with pg_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT row_scope, customer_id FROM namespaces WHERE namespace_id = $1",
                    namespace_id,
                )
            assert row is not None
            assert row["row_scope"] == "customer"
            assert row["customer_id"] == customer_id
            async with pg_pool.acquire() as conn:
                assert await conn.fetchval("SELECT count(*) FROM namespaces") == 1
        finally:
            l1.reset()

    async def test_a_customer_row_loses_its_customer(self, pg_pool: asyncpg.Pool) -> None:
        """customer -> platform: the move runs in both directions."""
        collection, _nats, l1 = _build_collection(pg_pool)
        namespace_id = uuid.uuid4()
        customer_id = uuid.uuid4()
        try:
            await _seed_platform_tool(pg_pool, namespace_id=namespace_id)
            await collection.rescope(namespace_id, customer_id=customer_id)
            outcome = await collection.rescope(namespace_id, customer_id=None)
            assert outcome.moved is True
            assert outcome.previous_row_scope == "customer"
            assert outcome.previous_customer_id == customer_id
            assert outcome.row_scope == "platform"
            async with pg_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT row_scope, customer_id FROM namespaces WHERE namespace_id = $1",
                    namespace_id,
                )
            assert row is not None
            assert row["row_scope"] == "platform"
            assert row["customer_id"] is None
        finally:
            l1.reset()

    async def test_every_grant_on_the_row_survives_the_move(self, pg_pool: asyncpg.Pool) -> None:
        """the grants are carried, not cascaded: the row is UPDATED, never deleted."""
        collection, _nats, l1 = _build_collection(pg_pool)
        namespace_id = uuid.uuid4()
        customer_id = uuid.uuid4()
        try:
            await _seed_platform_tool(pg_pool, namespace_id=namespace_id)
            operator_grant = await _grant_on(pg_pool, namespace_id, managed_by="manual")
            rebuilt_grant = await _grant_on(pg_pool, namespace_id, managed_by="bootstrap")
            await collection.rescope(namespace_id, customer_id=customer_id)
            async with pg_pool.acquire() as conn:
                surviving = {
                    r["assignment_id"]
                    for r in await conn.fetch(
                        "SELECT assignment_id FROM role_assignments WHERE scope_namespace_id = $1",
                        namespace_id,
                    )
                }
            assert surviving == {operator_grant, rebuilt_grant}
        finally:
            l1.reset()

    async def test_the_ordinary_upsert_lands_once_the_row_has_moved(
        self,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """after the move the generated upsert finds its own row again."""
        collection, _nats, l1 = _build_collection(pg_pool)
        namespace_id = uuid.uuid4()
        customer_id = uuid.uuid4()
        try:
            await _seed_platform_tool(pg_pool, namespace_id=namespace_id)
            await collection.rescope(namespace_id, customer_id=customer_id)
            entity = collection.entity_class(
                _tool_row(namespace_id, customer_id, "tools.probe.example.1-0"),
                is_new=True,
                collection=collection,
            )
            await collection.save_entity(entity)
            async with pg_pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT row_scope, customer_id FROM namespaces WHERE namespace_id = $1",
                    namespace_id,
                )
            assert len(rows) == 1
            assert rows[0]["row_scope"] == "customer"
            assert rows[0]["customer_id"] == customer_id
        finally:
            l1.reset()

    async def test_the_old_partition_key_leaves_l1_and_l2(self, pg_pool: asyncpg.Pool) -> None:
        """a cached row under the OLD composite key would outlive the move."""
        collection, nats, l1 = _build_collection(pg_pool)
        namespace_id = uuid.uuid4()
        customer_id = uuid.uuid4()
        try:
            await _seed_platform_tool(pg_pool, namespace_id=namespace_id)
            # warm both tiers under the pre-move key
            before = await collection.get(("platform", namespace_id))
            assert before is not None
            assert collection.get_row_sync(("platform", namespace_id)) is not None
            assert any(f"{namespace_id}" in key for key in nats.kv)

            await collection.rescope(namespace_id, customer_id=customer_id)

            assert collection.get_row_sync(("platform", namespace_id)) is None
            assert not [key for key in nats.kv if "platform" in key]
        finally:
            l1.reset()


class TestRescopeRefusesAndNoOps:
    """what a move must NOT do, each paired with the case it admits."""

    async def test_a_move_between_two_customers_is_refused(self, pg_pool: asyncpg.Pool) -> None:
        """re-tenanting is a different act and has no path through here."""
        collection, _nats, l1 = _build_collection(pg_pool)
        namespace_id = uuid.uuid4()
        first_customer = uuid.uuid4()
        second_customer = uuid.uuid4()
        try:
            await _seed_platform_tool(pg_pool, namespace_id=namespace_id)
            await collection.rescope(namespace_id, customer_id=first_customer)
            with pytest.raises(NamespaceRescopeRefused):
                await collection.rescope(namespace_id, customer_id=second_customer)
            async with pg_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT customer_id FROM namespaces WHERE namespace_id = $1",
                    namespace_id,
                )
            assert row is not None
            assert row["customer_id"] == first_customer
        finally:
            l1.reset()

    async def test_a_row_already_in_the_target_scope_moves_nothing(
        self,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """the admitted twin: same scope, same customer, no write."""
        collection, _nats, l1 = _build_collection(pg_pool)
        namespace_id = uuid.uuid4()
        try:
            await _seed_platform_tool(pg_pool, namespace_id=namespace_id)
            async with pg_pool.acquire() as conn:
                before = await conn.fetchval(
                    "SELECT date_updated FROM namespaces WHERE namespace_id = $1",
                    namespace_id,
                )
            outcome = await collection.rescope(namespace_id, customer_id=None)
            assert outcome.moved is False
            assert outcome.previous_row_scope == "platform"
            async with pg_pool.acquire() as conn:
                after = await conn.fetchval(
                    "SELECT date_updated FROM namespaces WHERE namespace_id = $1",
                    namespace_id,
                )
            assert after == before
        finally:
            l1.reset()

    async def test_an_absent_row_moves_nothing(self, pg_pool: asyncpg.Pool) -> None:
        """a first registration has nothing to move; the caller just inserts."""
        collection, _nats, l1 = _build_collection(pg_pool)
        try:
            outcome = await collection.rescope(uuid.uuid4(), customer_id=uuid.uuid4())
            assert outcome.moved is False
            assert outcome.previous_row_scope is None
            async with pg_pool.acquire() as conn:
                assert await conn.fetchval("SELECT count(*) FROM namespaces") == 0
        finally:
            l1.reset()
