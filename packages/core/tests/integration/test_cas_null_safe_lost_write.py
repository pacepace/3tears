"""integration proof that ``cas_null_safe=True`` stops a silently lost write.

this is the test the defect needed and did not have. the unit suite
(``tests/unit/collections/test_cas_null_safe.py``) asserts the SQL SHAPE; only a
real Postgres can show that the shape actually costs the losing writer its row,
because the whole failure is in what the server does with
``ON CONFLICT DO UPDATE ... WHERE`` -- and specifically in the fact that plain
``=`` can never match a ``NULL``.

the setup is the one three ``14-eng-ai-survey`` collections live in: a
**derived** primary key (``uuid5`` of the business key, not a random id), so two
writers who have never seen the row still compute the SAME id and collide on its
FIRST insert. each runs the standard read-modify-write retry loop, treating "0
rows affected" as "i lost, re-read and try again".

``test_two_first_writers_do_not_lose_an_increment`` is the one that fails
without the fix: the losing first-writer's unfenced upsert reports one row
affected, its retry loop concludes it succeeded, and one increment vanishes.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    INT_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import ConcurrentModificationError

pytestmark = pytest.mark.integration

#: namespace for the derived (deterministic) primary key. two writers holding
#: the same business key compute the same id, which is the entire premise.
_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

#: retry budget for the read-modify-write loops below. generous enough that a
#: genuine failure is a correctness bug, not a budget shortfall.
_MAX_RETRIES = 60


class _CounterEntity(BaseEntity):
    primary_key_field = "id"


class _CounterCollection(SchemaBackedCollection[_CounterEntity]):
    """derived-id counter fenced NULL-safely on ``date_updated``."""

    primary_key_column: str = "id"
    schema = TableSchema(
        name="cas_counters",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("label", STRING_TYPE),
            Column("count", INT_TYPE),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE, nullable=True),
        ],
        cas_column="date_updated",
        cas_null_safe=True,
    )

    @property
    def table_name(self) -> str:
        """return table name."""
        return "cas_counters"

    @property
    def entity_class(self) -> type[_CounterEntity]:
        """return entity class."""
        return _CounterEntity


def _derived_id(label: str) -> uuid.UUID:
    """derive the primary key from the business key, as the dependants do.

    :param label: business key
    :ptype label: str
    :return: deterministic primary key
    :rtype: uuid.UUID
    """
    return uuid.uuid5(_NAMESPACE, label)


@pytest.fixture
async def pg_pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    """per-test pool over a fresh ``cas_counters`` table.

    ``date_updated`` is deliberately NULLABLE: the NULL is what a first write
    fences against, and a NOT NULL column would make the case untestable.
    """
    pool: asyncpg.Pool = await asyncpg.create_pool(db_container, min_size=1, max_size=12)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                DROP TABLE IF EXISTS cas_counters;
                CREATE TABLE cas_counters (
                    id UUID PRIMARY KEY,
                    label TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    date_created TIMESTAMPTZ NOT NULL,
                    date_updated TIMESTAMPTZ
                )
                """
            )
        yield pool
    finally:
        async with pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS cas_counters")
        await pool.close()


@pytest.fixture
def collection(pg_pool: asyncpg.Pool) -> _CounterCollection:
    """collection wired straight to L3 -- no L1, no L2, nothing to hide behind.

    the point of the suite is what the DATABASE does, so every read below goes
    to ``fetch_from_store`` rather than through a cache that could mask a lost
    write by serving the value from L1.

    :param pg_pool: asyncpg pool over the fresh table
    :ptype pg_pool: asyncpg.Pool
    :return: the collection under test
    :rtype: _CounterCollection
    """
    registry = CollectionRegistry()
    registry.configure(l3_pool=pg_pool)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return _CounterCollection(registry, config, nats_client=None)


def _fresh_row(entity_id: uuid.UUID, label: str) -> dict[str, Any]:
    """build the payload a writer that read NO row would construct.

    ``date_updated`` is absent, which is what makes ``original_date_updated``
    ``None`` and selects the NULL branch of the fence.

    :param entity_id: derived primary key
    :ptype entity_id: uuid.UUID
    :param label: business key
    :ptype label: str
    :return: row payload for a first write
    :rtype: dict[str, Any]
    """
    return {
        "id": entity_id,
        "label": label,
        "count": 1,
        "date_created": datetime.now(UTC),
    }


async def _increment(collection: _CounterCollection, label: str) -> int:
    """run one read-modify-write increment under the retry loop.

    this mirrors ``SplitAssignmentsData.increment_count``: read L3 directly
    (never a cache), build the entity ``is_new=False`` so a lost race raises the
    RETRYABLE ``ConcurrentModificationError`` rather than an unretryable
    ``RuntimeError``, and retry on loss.

    :param collection: collection under test
    :ptype collection: _CounterCollection
    :param label: business key
    :ptype label: str
    :return: the count this attempt persisted
    :rtype: int
    :raises AssertionError: when the retry budget is exhausted
    """
    entity_id = _derived_id(label)
    for _ in range(_MAX_RETRIES):
        try:
            existing = await collection.fetch_from_store(entity_id)
            if existing is None:
                entity = _CounterEntity(_fresh_row(entity_id, label), is_new=False, collection=collection)
                new_count = 1
            else:
                new_count = int(existing["count"]) + 1
                entity = _CounterEntity(existing, is_new=False, collection=collection)
                entity.count = new_count
            await entity.save()
            return new_count
        except ConcurrentModificationError:
            await asyncio.sleep(0)
    raise AssertionError(f"retry budget exhausted incrementing {label!r}")


async def _count_of(pg_pool: asyncpg.Pool, label: str) -> int:
    """read the persisted count straight from Postgres.

    :param pg_pool: asyncpg pool
    :ptype pg_pool: asyncpg.Pool
    :param label: business key
    :ptype label: str
    :return: stored count
    :rtype: int
    """
    async with pg_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT count FROM cas_counters WHERE id = $1", _derived_id(label))
    assert row is not None
    return int(row["count"])


class TestNullSafeFenceUnderContention:
    """what the fence buys, measured against a real Postgres."""

    @pytest.mark.asyncio
    async def test_two_first_writers_do_not_lose_an_increment(
        self,
        collection: _CounterCollection,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """THE REGRESSION. two writers race the FIRST insert of a derived id.

        the interleave is forced rather than raced, so the test is
        deterministic: writer B takes its "there is no row" read BEFORE writer A
        inserts, then writes afterwards. B's expected fence value is therefore
        ``NULL`` while the stored value is A's timestamp.

        with the fence, B affects 0 rows, retries against the row A actually
        wrote, and increments it to 2. WITHOUT the fence, B's upsert is an
        unfenced ``ON CONFLICT DO UPDATE`` that reports one row affected and
        overwrites A's row with ``count=1``: A's increment is gone, nothing
        raises, and the final count is 1.
        """
        label = "shared-key"
        entity_id = _derived_id(label)

        # B reads first and sees nothing.
        b_read = await collection.fetch_from_store(entity_id)
        assert b_read is None

        # A does a complete first write.
        assert await _increment(collection, label) == 1
        assert await _count_of(pg_pool, label) == 1

        # B now writes on the strength of its stale read. This is the exact
        # statement the defect let through unfenced.
        b_entity = _CounterEntity(_fresh_row(entity_id, label), is_new=False, collection=collection)
        with pytest.raises(ConcurrentModificationError):
            await b_entity.save()

        # A's row survived untouched.
        assert await _count_of(pg_pool, label) == 1

        # and B's retry lands on top of it rather than in place of it.
        assert await _increment(collection, label) == 2
        assert await _count_of(pg_pool, label) == 2

    @pytest.mark.asyncio
    async def test_twenty_concurrent_increments_all_land(
        self,
        collection: _CounterCollection,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """the live shape: 20 writers, none of which has ever seen the row.

        every one of them starts by computing the same derived id and reading
        nothing, so all 20 attempt a FIRST insert. exactly one can win it; the
        other 19 must be told they lost and must retry. anything less than 20
        is an increment that evaporated.
        """
        label = "hot-key"
        await asyncio.gather(*(_increment(collection, label) for _ in range(20)))
        assert await _count_of(pg_pool, label) == 20

    @pytest.mark.asyncio
    async def test_a_genuine_first_insert_is_not_blocked(
        self,
        collection: _CounterCollection,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """the fence must not stop a row being CREATED, only stop it being clobbered.

        with no conflicting row the plain INSERT applies and the ``ON CONFLICT``
        branch -- fence included -- is never evaluated.
        """
        label = "brand-new"
        entity = _CounterEntity(_fresh_row(_derived_id(label), label), is_new=False, collection=collection)
        await entity.save()
        assert await _count_of(pg_pool, label) == 1

    @pytest.mark.asyncio
    async def test_a_stale_non_null_fence_value_also_loses(
        self,
        collection: _CounterCollection,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """the ordinary (non-NULL) CAS case still works through the same statement.

        one statement now serves create and update alike, so this checks the
        NULL-safe operator did not cost the fence its normal job.
        """
        label = "stale-update"
        await _increment(collection, label)
        stale = await collection.fetch_from_store(_derived_id(label))
        assert stale is not None

        # somebody else advances the row.
        await _increment(collection, label)
        assert await _count_of(pg_pool, label) == 2

        # our write, still carrying the pre-advance fence value, must lose.
        loser = _CounterEntity(dict(stale), is_new=False, collection=collection)
        loser.count = 99
        with pytest.raises(ConcurrentModificationError):
            await loser.save()
        assert await _count_of(pg_pool, label) == 2

    @pytest.mark.asyncio
    async def test_independent_keys_do_not_interfere(
        self,
        collection: _CounterCollection,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """the fence is per-row: different business keys never contend."""
        await asyncio.gather(
            *(_increment(collection, f"key-{i}") for i in range(8)),
            *(_increment(collection, f"key-{i}") for i in range(8)),
        )
        for i in range(8):
            assert await _count_of(pg_pool, f"key-{i}") == 2
