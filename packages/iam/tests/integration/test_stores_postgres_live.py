"""Integration: the Postgres stores' keyed statements on a real database.

Each claim, point read and purge is a shared keyed builder's statement (:mod:`threetears.interfaces.keyed`) run on
the caller's pool: a claim one ``DELETE ... RETURNING`` guarded by its expiry, a read ``LIMIT 1`` by the key, a purge
a page of expired rows a statement through the ``expires_at`` index until a short page. Against the session
``db_container`` (Postgres), in a schema of its own, on a pool registering the JSONB codec every 3tears pool does.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from datetime import timedelta

import asyncpg
import pytest

from threetears.iam.stores.postgres import (
    EXPIRES_INDEX_DDL,
    STATE_TABLE_DDL,
    TICKET_TABLE_DDL,
    PostgresStateStore,
    PostgresTicketStore,
)

pytestmark = pytest.mark.integration

_TTL = timedelta(minutes=30)


async def _jsonb(conn: asyncpg.Connection) -> None:
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


@pytest.fixture
async def schema(db_container: str) -> AsyncIterator[tuple[asyncpg.Pool, str]]:
    """a pool and a fresh schema holding one ticket table and one state table, each with its expiry index."""
    name = f"iam_live_{uuid.uuid4().hex[:10]}"
    pool: asyncpg.Pool = await asyncpg.create_pool(db_container, min_size=1, max_size=4, init=_jsonb)
    try:
        async with pool.acquire() as conn:
            await conn.execute(f"CREATE SCHEMA {name}")
            for table, ddl in ((f"{name}.tickets", TICKET_TABLE_DDL), (f"{name}.state", STATE_TABLE_DDL)):
                await conn.execute(ddl.format(table=table))
                await conn.execute(EXPIRES_INDEX_DDL.format(index=f"{table.split('.')[1]}_expires", table=table))
        yield pool, name
    finally:
        async with pool.acquire() as conn:
            await conn.execute(f"DROP SCHEMA IF EXISTS {name} CASCADE")
        await pool.close()


async def test_a_ticket_redeems_once_and_an_expired_one_never(schema: tuple[asyncpg.Pool, str]) -> None:
    """Flips if the claim's statement fails on Postgres, consumes nothing, or admits an expired row."""
    pool, name = schema
    store = PostgresTicketStore(pool, table=f"{name}.tickets")
    issued = await store.issue({"principal": "p-1"}, ttl=_TTL)
    expired = await store.issue({"principal": "p-2"}, ttl=timedelta(seconds=-1))
    assert await store.redeem(issued.secret) == {"principal": "p-1"}
    assert await store.redeem(issued.secret) is None
    assert await store.redeem(expired.secret) is None
    assert await pool.fetchval(f"SELECT count(*) FROM {name}.tickets WHERE hashed = $1", expired.hashed) == 1


async def test_state_reads_without_consuming_and_takes_once(schema: tuple[asyncpg.Pool, str]) -> None:
    """Flips if the point read or the take fails on Postgres, or either answers an expired value."""
    pool, name = schema
    store = PostgresStateStore(pool, table=f"{name}.state")
    await store.put("s-1", {"nonce": "n"}, ttl=_TTL)
    await store.put("s-2", {"nonce": "gone"}, ttl=timedelta(seconds=-1))
    assert await store.get("s-1") == {"nonce": "n"}
    assert await store.get("s-1") == {"nonce": "n"}
    assert await store.get("s-2") is None
    assert await store.take("s-2") is None
    assert await store.take("s-1") == {"nonce": "n"}
    assert await store.take("s-1") is None


async def test_a_purge_deletes_every_expired_row_page_after_page_and_no_live_one(
    schema: tuple[asyncpg.Pool, str],
) -> None:
    """Flips if the purge stops after its first page, deletes a live row, or fails on Postgres."""
    pool, name = schema
    tickets = PostgresTicketStore(pool, table=f"{name}.tickets")
    state = PostgresStateStore(pool, table=f"{name}.state")
    live = await tickets.issue({"n": 1}, ttl=_TTL)
    await state.put("live", {"n": 1}, ttl=_TTL)
    for table, key in ((f"{name}.tickets", "hashed"), (f"{name}.state", "key")):
        await pool.execute(
            f"INSERT INTO {table} ({key}, payload, expires_at) "  # noqa: S608 - the test's own schema
            "SELECT 'x' || i, '{}'::jsonb, now() - interval '1 hour' FROM generate_series(1, 2500) AS i"
        )
    await tickets.purge_expired()
    await state.purge_expired()
    assert [row["hashed"] for row in await pool.fetch(f"SELECT hashed FROM {name}.tickets")] == [live.hashed]
    assert [row["key"] for row in await pool.fetch(f"SELECT key FROM {name}.state")] == ["live"]
