"""integration tests for the DataStore dynamic path on a real asyncpg pool.

regression suite for the integration-guide review findings (PR #84):
the repo's prior integration coverage used a dict-returning stub pool,
which masked two real-driver failures in the L3 -> L1 re-promotion path:

1. issue #85 -- ``fetch_from_store`` returned raw ``asyncpg.Record``
   rows; iterating a Record yields values (not keys), so
   ``SQLiteBackend.upsert`` built ``INSERT INTO t () VALUES ()``.
2. issue #86 -- asyncpg deserializes ``uuid`` columns to
   ``pgproto.pgproto.UUID`` (a ``uuid.UUID`` subclass); sqlite3 adapter
   lookup is exact-type, so binding the re-promoted pk raw raised
   ``ProgrammingError``.

scenarios:

1. full CRUD round-trip with a **uuid primary key** and a **raw asyncpg
   pool** (no dict wrapper, no codec registration) -- the §8.1
   integration-guide shape.
2. two tables created through one ``DataStore`` sharing a single L1
   backend (additive ``SQLiteBackend.initialize``).
3. a ``vector`` column round-trips L3 -> L1 with pgvector text-form
   coercion at the L3 border.

the suite is guarded by ``@pytest.mark.integration`` and skips when
docker is unavailable.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator

import asyncpg
import pytest

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.schema import ColumnDef, TableDef
from threetears.core.data.store import DataStore

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def db_image() -> str:
    """pin pgvector/pg16; the vector scenario needs the extension."""
    return "pgvector/pgvector:pg16"


@pytest.fixture
async def pg_pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    """plain asyncpg pool -- deliberately no codecs, no dict wrapper.

    the whole point of this suite is exercising the raw driver shape
    the integration guide documents (and the stub pool masked).
    """
    pool: asyncpg.Pool = await asyncpg.create_pool(db_container, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        yield pool
    finally:
        await pool.close()


@pytest.fixture
def l1_backend() -> Iterator[SQLiteBackend]:
    backend = SQLiteBackend(db_name=f"it_datastore_{uuid.uuid4().hex[:8]}")
    yield backend
    backend.reset()


@pytest.fixture
def store(pg_pool: asyncpg.Pool, l1_backend: SQLiteBackend) -> DataStore:
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1_backend, l3_pool=pg_pool)
    return DataStore(uuid.uuid4(), registry, DefaultCoreConfig(collection_flush="ALWAYS"))


def _unique(name: str) -> str:
    """unique table name per test run (the container is session-scoped)."""
    return f"{name}_{uuid.uuid4().hex[:8]}"


class TestUuidPkRoundTrip:
    """§8.1 shape: uuid pk + raw asyncpg pool, end to end."""

    async def test_full_crud_round_trip(self, store: DataStore, l1_backend: SQLiteBackend) -> None:
        table = _unique("widgets")
        widgets = await store.create_table(
            TableDef(
                name=table,
                columns=[
                    ColumnDef(name="id", column_type="uuid", primary_key=True),
                    ColumnDef(name="name", column_type="text", nullable=False),
                    ColumnDef(name="score", column_type="integer"),
                ],
            )
        )

        wid = str(uuid.uuid4())
        entity = widgets.create({"id": wid, "name": "sprocket", "score": 42})
        await entity.save()

        got = await widgets.get(wid)
        assert got is not None
        got.score = 99
        await got.save()

        # evict from L1, then re-promote from L3 through the raw pool:
        # the row comes back as an asyncpg Record carrying a
        # pgproto.UUID pk -- the exact #85 + #86 failure shape.
        await widgets.invalidate_cache(wid)
        refreshed = await widgets.get(wid)
        assert refreshed is not None
        assert refreshed.score == 99
        assert refreshed.name == "sprocket"

        # the re-promoted row is readable from L1 by its string pk
        cached = l1_backend.select_by_id(table, wid)
        assert cached is not None
        assert cached["score"] == 99

        await widgets.delete(wid)
        assert await widgets.get(wid) is None


class TestMultiTableSharedL1:
    """two DataStore tables compose on one shared L1 backend."""

    async def test_second_table_round_trips(self, store: DataStore) -> None:
        first = _unique("first")
        second = _unique("second")
        columns = [
            ColumnDef(name="id", column_type="text", primary_key=True),
            ColumnDef(name="name", column_type="text"),
        ]
        first_coll = await store.create_table(TableDef(name=first, columns=columns))
        second_coll = await store.create_table(TableDef(name=second, columns=columns))

        a = first_coll.create({"id": "a", "name": "alpha"})
        await a.save()
        b = second_coll.create({"id": "b", "name": "beta"})
        await b.save()

        await first_coll.invalidate_cache("a")
        await second_coll.invalidate_cache("b")

        got_a = await first_coll.get("a")
        got_b = await second_coll.get("b")
        assert got_a is not None and got_a.name == "alpha"
        assert got_b is not None and got_b.name == "beta"


class TestVectorRoundTrip:
    """vector columns round-trip L3 -> L1 on a raw pool."""

    async def test_vector_column_round_trips(self, store: DataStore, l1_backend: SQLiteBackend) -> None:
        table = _unique("embeddings")
        embeddings = await store.create_table(
            TableDef(
                name=table,
                columns=[
                    ColumnDef(name="id", column_type="text", primary_key=True),
                    ColumnDef(name="embedding", column_type="vector", vector_dim=3),
                ],
            )
        )

        entity = embeddings.create({"id": "e1", "embedding": [0.1, 0.2, 0.3]})
        await entity.save()

        await embeddings.invalidate_cache("e1")
        refreshed = await embeddings.get("e1")
        assert refreshed is not None
        assert refreshed.embedding == [0.1, 0.2, 0.3]

        cached = l1_backend.select_by_id(table, "e1")
        assert cached is not None
        assert cached["embedding"] == [0.1, 0.2, 0.3]
