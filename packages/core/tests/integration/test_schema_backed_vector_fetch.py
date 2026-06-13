"""integration regression: declarative SchemaBackedCollection by-pk fetch
over a real pgvector pool.

the unit suite for :class:`SchemaBackedCollection` uses a recording-mock
pool, so its ``test_fetch_decodes_vector_string`` handed back a
PRE-STRINGIFIED vector and never exercised the asyncpg decode that fails
on a real pool. that masked a production bug: the generic by-pk fetch
emitted ``SELECT *``, which returns the ``vector`` column (OID 8078); no
3tears pool registers a pgvector codec (only the jsonb text codec in
:func:`threetears.core.collections.init_connection`), so asyncpg raised
``UnsupportedClientFeatureError: unhandled standard data type 'vector'``
on every get / update / delete of a row in a vector-bearing table -- the
hub knowledge get/update/delete 500 surfaced by the live smoke.

this suite exercises the EXACT production path (declarative
:class:`TableSchema` + the generic ``fetch_from_postgres`` ->
``_build_fetch_sql``) against a real pgvector container, in the two
shapes that occur in production:

1. the consumer DECLARES the vector column (the SDK / agent-memory
   retrieval shape): the fetch must cast it ``::text`` so asyncpg returns
   the bracketed string the read coercion parses back to a ``list``.
2. the consumer OMITS the vector column from its ``TableSchema`` (the hub
   knowledge shape -- the hub never reads vectors): the fetch must
   project only DECLARED columns, so the undeclared vector is never read
   and the missing codec is never hit.

harness-vs-production OID divergence (important): the testcontainer is
``pgvector/pg16``, where the extension's ``vector`` type gets a user-type
OID and asyncpg returns it as a TEXT string (no codec needed). the hub
runs YugabyteDB, where ``vector`` carries the FIXED low OID 8078 that
asyncpg treats as a standard type with no codec and RAISES
``UnsupportedClientFeatureError``. so on this harness the throw is NOT
reproduced -- the regression that goes red pre-fix is
:meth:`TestOmittedVectorFetch.test_get_does_not_read_undeclared_vector`,
which catches the ROOT cause (``SELECT *`` over-reads the undeclared
column; the leaked ``embedding`` string fails the assertion). the
declared-vector cases are forward-contract coverage: on pg16 they pass
even pre-fix (the text string parses), but they pin the ``::text`` cast
that is REQUIRED for a declared vector to be readable at all on YB. the
end-to-end proof of the YB throw-fix is the hub live smoke (by-pk
get/update/delete of a knowledge row on the real YugabyteDB hub).

guarded by ``@pytest.mark.integration``; skips when docker is
unavailable.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import asyncpg
import pytest

from threetears.core.collections import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    STRING_TYPE,
    TSVECTOR_TYPE,
    UUID_TYPE,
    VECTOR_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def db_image() -> str:
    """pin pgvector/pg16; the suite needs the ``vector`` extension."""
    return "pgvector/pgvector:pg16"


@pytest.fixture
async def pg_pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    """real asyncpg pool with the canonical 3tears connection init.

    ``init_connection`` registers ONLY the jsonb / json text codec --
    deliberately NO vector codec, exactly the production shape that makes
    a raw ``SELECT *`` over a vector column unreadable.
    """
    pool: asyncpg.Pool = await asyncpg.create_pool(
        db_container,
        min_size=1,
        max_size=4,
        init=init_connection,
    )
    try:
        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute("DROP TABLE IF EXISTS sb_vec_items")
            await conn.execute(
                "CREATE TABLE sb_vec_items ("
                "id UUID PRIMARY KEY, label TEXT NOT NULL, "
                "embedding VECTOR(4), search_vector TSVECTOR, "
                "date_created TIMESTAMPTZ, date_updated TIMESTAMPTZ)",
            )
        yield pool
    finally:
        async with pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS sb_vec_items")
        await pool.close()


class _StubEntity(BaseEntity):
    """minimal entity; the fetch path returns a coerced row dict."""

    primary_key_field = "id"


class _DeclaredVecCollection(SchemaBackedCollection[_StubEntity]):
    """TableSchema that DECLARES the vector column (SDK / memory shape)."""

    primary_key_column: str = "id"
    schema = TableSchema(
        name="sb_vec_items",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("label", STRING_TYPE),
            Column("embedding", VECTOR_TYPE, nullable=True, vector_dim=4),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        cas_column="date_updated",
    )

    @property
    def table_name(self) -> str:
        """return database table name.

        :return: table name string
        :rtype: str
        """
        return "sb_vec_items"

    @property
    def entity_class(self) -> type[_StubEntity]:
        """return entity class for this collection.

        :return: stub entity class
        :rtype: type[_StubEntity]
        """
        return _StubEntity


class _OmittedVecCollection(SchemaBackedCollection[_StubEntity]):
    """TableSchema that OMITS the vector column (hub knowledge shape).

    the table HAS the ``embedding`` column; this schema simply never
    declares it because the hub never reads vectors. the by-pk fetch must
    therefore never select it.
    """

    primary_key_column: str = "id"
    schema = TableSchema(
        name="sb_vec_items",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("label", STRING_TYPE),
            # embedding deliberately omitted -- the hub never reads vectors
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        cas_column="date_updated",
    )

    @property
    def table_name(self) -> str:
        """return database table name.

        :return: table name string
        :rtype: str
        """
        return "sb_vec_items"

    @property
    def entity_class(self) -> type[_StubEntity]:
        """return entity class for this collection.

        :return: stub entity class
        :rtype: type[_StubEntity]
        """
        return _StubEntity


class _DeclaredFtsCollection(SchemaBackedCollection[_StubEntity]):
    """TableSchema that DECLARES the tsvector column (memories shape).

    ``tsvector`` is the OTHER codec-less asyncpg type: like ``vector`` it
    has no binary codec on any 3tears pool, so a by-pk fetch that does not
    cast it ``::text`` raises the same ``UnsupportedClientFeatureError``.
    ``MemoriesCollection`` / ``MediaContentCollection`` /
    ``MemoryChunkCollection`` declare exactly this shape and inherit the
    generic ``fetch_from_postgres``.
    """

    primary_key_column: str = "id"
    schema = TableSchema(
        name="sb_vec_items",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("label", STRING_TYPE),
            Column(
                "search_vector",
                TSVECTOR_TYPE,
                nullable=True,
                immutable=True,
            ),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        cas_column="date_updated",
    )

    @property
    def table_name(self) -> str:
        """return database table name.

        :return: table name string
        :rtype: str
        """
        return "sb_vec_items"

    @property
    def entity_class(self) -> type[_StubEntity]:
        """return entity class for this collection.

        :return: stub entity class
        :rtype: type[_StubEntity]
        """
        return _StubEntity


def _nats() -> AsyncMock:
    """build a no-op typed-NATS wrapper mock for collection construction."""
    bucket = AsyncMock()
    bucket.get = AsyncMock(return_value=None)
    bucket.put = AsyncMock(return_value=1)
    bucket.delete = AsyncMock(return_value=True)
    nats = AsyncMock()
    nats.kv_bucket = AsyncMock(return_value=bucket)
    nats.publish = AsyncMock()
    nats.subscribe_typed = AsyncMock()
    return nats


def _registry(pool: asyncpg.Pool) -> CollectionRegistry:
    """build a registry wired to the real L3 pool (no L1: the by-pk L3
    fetch is the path under test)."""
    reg = CollectionRegistry()
    reg.configure(l3_pool=pool)
    return reg


def _config() -> DefaultCoreConfig:
    """build an always-flush config."""
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


async def _insert_row(
    pool: asyncpg.Pool,
    item_id: uuid.UUID,
    *,
    embedding: str | None,
) -> None:
    """insert one row, casting the text vector literal with ``::vector``.

    :param pool: real asyncpg pool
    :ptype pool: asyncpg.Pool
    :param item_id: row primary key
    :ptype item_id: uuid.UUID
    :param embedding: pgvector text literal (e.g. ``"[0.1, 0.2, 0.3, 0.4]"``)
        or ``None`` for a NULL vector
    :ptype embedding: str | None
    :return: nothing
    :rtype: None
    """
    now = datetime(2026, 1, 1, tzinfo=UTC)
    await pool.execute(
        "INSERT INTO sb_vec_items "
        "(id, label, embedding, search_vector, date_created, date_updated) "
        "VALUES ($1, $2, $3::vector, to_tsvector('english', $2), $4, $5)",
        item_id,
        "hello",
        embedding,
        now,
        now,
    )


class TestDeclaredVectorFetch:
    """schema declares the vector: by-pk fetch round-trips it as a list."""

    async def test_get_returns_parsed_vector(self, pg_pool: asyncpg.Pool) -> None:
        """fetch casts the declared vector ``::text`` and parses it back.

        pre-fix this raised ``UnsupportedClientFeatureError`` (``SELECT *``
        returned a binary ``vector`` with no codec); the real pgvector
        ``::text`` form has NO spaces, the shape the recording-mock unit
        test never exercised.
        """
        coll = _DeclaredVecCollection(
            _registry(pg_pool),
            _config(),
            nats_client=_nats(),
        )
        item_id = uuid.uuid4()
        await _insert_row(pg_pool, item_id, embedding="[0.1, 0.2, 0.3, 0.4]")

        row = await coll.fetch_from_postgres(item_id)

        assert row is not None
        assert row["embedding"] == [0.1, 0.2, 0.3, 0.4]
        assert row["label"] == "hello"

    async def test_get_handles_null_vector(self, pg_pool: asyncpg.Pool) -> None:
        """a NULL declared vector round-trips to ``None`` without throwing."""
        coll = _DeclaredVecCollection(
            _registry(pg_pool),
            _config(),
            nats_client=_nats(),
        )
        item_id = uuid.uuid4()
        await _insert_row(pg_pool, item_id, embedding=None)

        row = await coll.fetch_from_postgres(item_id)

        assert row is not None
        assert row["embedding"] is None


class TestOmittedVectorFetch:
    """schema OMITS the vector (hub knowledge shape): by-pk fetch over a
    vector-bearing table must never decode the undeclared vector.

    this is the exact production failure the live smoke surfaced: pre-fix
    ``SELECT *`` returned the ``embedding`` column and asyncpg raised
    ``UnsupportedClientFeatureError`` even when the vector value is NULL.
    """

    async def test_get_does_not_read_undeclared_vector(
        self,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """fetch projects declared columns only; the vector is never read."""
        coll = _OmittedVecCollection(
            _registry(pg_pool),
            _config(),
            nats_client=_nats(),
        )
        item_id = uuid.uuid4()
        await _insert_row(pg_pool, item_id, embedding="[0.5, 0.6, 0.7, 0.8]")

        row = await coll.fetch_from_postgres(item_id)

        assert row is not None
        assert row["label"] == "hello"
        # neither undeclared codec-less column is read: the projection is
        # declared-columns-only, so both the vector and the tsvector the
        # table carries are absent (and never hit the missing codec).
        assert "embedding" not in row
        assert "search_vector" not in row


class TestDeclaredTsvectorFetch:
    """schema declares a tsvector (``memories.search_vector`` shape):
    by-pk fetch casts it ``::text`` so the codec-less column is readable
    instead of raising.
    """

    async def test_get_returns_tsvector_as_text(
        self,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """a declared trigger-maintained tsvector round-trips as text.

        pre-fix the by-pk fetch would project ``search_vector`` raw and
        raise on the missing codec the moment a memory row was fetched by
        pk through the hub L3 broker; the ``::text`` cast returns the
        full-text string form (never consumed as data).
        """
        coll = _DeclaredFtsCollection(
            _registry(pg_pool),
            _config(),
            nats_client=_nats(),
        )
        item_id = uuid.uuid4()
        await _insert_row(pg_pool, item_id, embedding=None)

        row = await coll.fetch_from_postgres(item_id)

        assert row is not None
        assert row["label"] == "hello"
        assert isinstance(row["search_vector"], str)
        assert "hello" in row["search_vector"]
