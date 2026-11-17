"""Integration: a nearest read of a small owner beside a heavy one gets every row it asks for, on a platform pool.

Every vector reader pins an owner. At a heavy owner the planner walks the HNSW index and filters it to the owner, and
pgvector's HNSW scan stops at ``hnsw.ef_search`` (40) candidates: a 50-row owner beside a 20,000-row one got 0 of 10
rows on PostgreSQL 16 + pgvector 0.8.7 (2026-11-14). A platform pool
(:func:`~threetears.core.utils.pg_pool_kwargs.create_pool_with_startup_timeout`) starts its sessions with
``hnsw.iterative_scan = relaxed_order``, and ``read_nearest`` re-sorts the rows it gets by distance. YugabyteDB's
ybhnsw applies the owner pin inside its index scan, and refuses ``SET hnsw.*``: there the read gets every row with or
without the setting, and sends no ``SET``.

The plan is forced onto the index (``enable_seqscan = off``, a session default) as the heavy owner's plan is. The
re-sort's cost is budgeted against the same read unwrapped (the hand-written statement it was) on the same session.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.core.keyed import Nearest, read_nearest
from threetears.core.testing.calibrate import best_seconds_interleaved, budget_problem
from threetears.core.testing.plans import on_yugabyte
from threetears.core.utils.pg_pool_kwargs import create_pool_with_startup_timeout

pytestmark = pytest.mark.integration

_DIM = 16
_HEAVY = 20_000
_SMALL = 50
_K = 10
#: the index plan, as at a heavy owner (a session default: the read sends nothing but its statement)
_INDEX_PLAN = {"enable_seqscan": "off"}


def _random_vectors(count: int, owner: str) -> str:
    """an INSERT of ``count`` rows of ``owner``, each a random ``_DIM``-wide vector (seeded by the caller)."""
    parts = "array_to_string(ARRAY(SELECT random() + g * 0 FROM generate_series(1, %d)), ',')" % _DIM
    return (
        f"INSERT INTO m (id, owner, embedding) SELECT gen_random_uuid(), '{owner}'::uuid, "
        f"('[' || {parts} || ']')::vector FROM generate_series(1, {count}) AS g"
    )


@pytest.fixture
async def world(db_container: str) -> AsyncIterator[tuple[str, str, bool]]:
    """a schema of its own: ``m (id, owner, embedding vector(16))`` with an HNSW index, a 20,000-row owner and a 50-row
    one; its search path's URL, the small owner, and whether the database is YugabyteDB."""
    name = f"nearest_heavy_{uuid.uuid4().hex[:10]}"
    small = str(uuid.uuid4())
    held = await asyncpg.connect(db_container)
    try:
        yugabyte = await on_yugabyte(held)
        await held.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await held.execute(f'CREATE SCHEMA "{name}"')
        await held.execute(f'SET search_path TO "{name}", public')
        await held.execute(f"CREATE TABLE m (id uuid PRIMARY KEY, owner uuid NOT NULL, embedding vector({_DIM}))")
        await held.execute("SELECT setseed(0.7)")
        await held.execute(_random_vectors(_HEAVY, str(uuid.uuid4())))
        await held.execute(_random_vectors(_SMALL, small))
        method = "ybhnsw" if yugabyte else "hnsw"
        await held.execute(f"CREATE INDEX m_hnsw ON m USING {method} (embedding vector_cosine_ops)")
        await held.execute("ANALYZE m")
        yield name, small, yugabyte
    finally:
        await held.execute(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')
        await held.close()


#: the query vector: ``[1, 0, ...]``
_QUERY = "[" + ",".join(["1", *["0"] * (_DIM - 1)]) + "]"
#: reads a timed run makes: one read is a few milliseconds, under the timer's noise alone
_READS = 20


async def _nearest(pool: asyncpg.Pool, small: str) -> tuple[list[dict[str, Any]], list[str]]:
    """the small owner's nearest rows to ``[1, 0, ...]`` through ``read_nearest``, and every statement the read sent."""
    sent: list[str] = []
    async with pool.acquire() as conn:
        conn.add_query_logger(lambda record: sent.append(record.query))
        rows = await read_nearest(
            conn,
            "m",
            by={"owner": uuid.UUID(small)},
            nearest=Nearest("embedding", _QUERY, similarity="similarity"),
            limit=_K,
            columns=("id",),
        )
    return rows, sent


async def test_a_small_owner_beside_a_heavy_one_gets_every_row_on_a_platform_pool(
    db_container: str, world: tuple[str, str, bool]
) -> None:
    """Flips if a platform pool's nearest read of a 50-row owner beside a 20,000-row one, walked through the HNSW
    index, gets fewer than the 10 rows it asks for (the iterative scan off), gets them out of distance order (no
    re-sort), or sends a ``SET``."""
    schema, small, _yugabyte = world
    pool = await create_pool_with_startup_timeout(
        db_container,
        pool_name="nearest_heavy",
        min_size=1,
        max_size=1,
        server_settings={"search_path": f'"{schema}", public', **_INDEX_PLAN},
    )
    try:
        rows, sent = await _nearest(pool, small)
    finally:
        await pool.close()
    assert len(rows) == _K
    similarities = [row["similarity"] for row in rows]
    assert similarities == sorted(similarities, reverse=True)
    assert not [query for query in sent if query.lstrip().upper().startswith("SET")]


async def test_without_the_iterative_scan_postgres_gets_fewer_rows_than_asked(
    db_container: str, world: tuple[str, str, bool]
) -> None:
    """Flips if the world stops being one where the setting matters: on PostgreSQL a pool without the platform's
    settings must get fewer than the 10 rows from the HNSW walk (else the test above proves nothing there). On
    YugabyteDB the owner pin is applied inside the index scan, so every row comes without the setting."""
    schema, small, yugabyte = world
    pool = await create_pool_with_startup_timeout(
        db_container,
        pool_name="nearest_heavy_bare",
        min_size=1,
        max_size=1,
        platform_settings=False,
        server_settings={"search_path": f'"{schema}", public', **_INDEX_PLAN},
    )
    try:
        rows, _sent = await _nearest(pool, small)
    finally:
        await pool.close()
    if yugabyte:
        assert len(rows) == _K
    else:
        assert len(rows) < _K


@pytest.mark.loop_timing
async def test_the_re_sort_costs_within_twice_the_read_unwrapped(
    db_container: str, world: tuple[str, str, bool]
) -> None:
    """Flips if ``read_nearest``'s wrapper (the read under its LIMIT in a subquery, re-sorted by distance) takes more
    than twice the same read unwrapped, the hand-written statement it was, on the same platform session, for the
    small owner and the heavy one; or answers other rows than it."""
    schema, small, _yugabyte = world
    pool = await create_pool_with_startup_timeout(
        db_container,
        pool_name="nearest_heavy_speed",
        min_size=1,
        max_size=1,
        server_settings={"search_path": f'"{schema}", public', **_INDEX_PLAN},
    )
    try:
        async with pool.acquire() as conn:
            heavy = await conn.fetchval("SELECT owner FROM m WHERE owner <> $1 LIMIT 1", uuid.UUID(small))
            for owner in (uuid.UUID(small), heavy):
                bare_sql = (
                    "SELECT t.id, 1 - (t.embedding OPERATOR(public.<=>) $2::text::public.vector) AS similarity "
                    "FROM m AS t WHERE t.owner = $1 "
                    "ORDER BY t.embedding OPERATOR(public.<=>) $2::text::public.vector LIMIT $3"
                )

                async def bare(who: uuid.UUID = owner, sql: str = bare_sql) -> list[Any]:
                    return [await conn.fetch(sql, who, _QUERY, _K) for _ in range(_READS)]

                async def wrapped(who: uuid.UUID = owner) -> list[Any]:
                    return [
                        await read_nearest(
                            conn,
                            "m",
                            by={"owner": who},
                            nearest=Nearest("embedding", _QUERY, similarity="similarity"),
                            limit=_K,
                            columns=("id",),
                        )
                        for _ in range(_READS)
                    ]

                old = {row["id"] for row in (await bare())[0]}
                new = {row["id"] for row in (await wrapped())[0]}
                assert new == old
                baseline, took = await best_seconds_interleaved(bare, wrapped)
                problem = budget_problem(f"{_READS} nearest reads of owner {owner}", took, baseline)
                assert problem is None, problem
    finally:
        await pool.close()
