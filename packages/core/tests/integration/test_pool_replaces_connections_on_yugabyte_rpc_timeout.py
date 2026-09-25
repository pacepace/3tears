"""a real asyncpg pool replaces its connections when one answers the YugabyteDB RPC timeout.

Postgres cannot wedge a tserver session, but it can raise the error a wedged one answers: SQLSTATE
XX000 with YugabyteDB's message, which asyncpg raises as ``InternalServerError`` exactly as it does
for YugabyteDB. these tests drive real pool mechanics -- release, reset, the generation check -- so
they show what the unit tests cannot: that the caller still sees the error its query raised, and
that the next acquire gets a different server backend.

the control test runs the same sequence on a pool without the recycler and sees the same backend
come back, which is the wedge's defect.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import asyncpg
import pytest

from threetears.core.utils.yugabyte_rpc_timeout import (
    YUGABYTE_RPC_TIMEOUT_MESSAGE_PREFIX,
    YugabyteRpcTimeoutRecycler,
)

pytestmark = pytest.mark.integration

_WEDGE_MESSAGE = f"{YUGABYTE_RPC_TIMEOUT_MESSAGE_PREFIX}, state: kRequestSent"

#: one statement on the simple-query path (``execute`` with no arguments)
_RAISE_WEDGE = f"DO $$ BEGIN RAISE EXCEPTION USING ERRCODE = 'XX000', MESSAGE = '{_WEDGE_MESSAGE}'; END $$"

#: a function, so a query on the extended-protocol path (``fetchval``) raises it too
_CREATE_WEDGE_FUNCTION = f"""
CREATE OR REPLACE FUNCTION pg_temp.answer_rpc_timeout(x int) RETURNS int LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION USING ERRCODE = 'XX000', MESSAGE = '{_WEDGE_MESSAGE}';
END $$
"""


async def _recycling_pool(dsn: str, *, size: int) -> tuple[asyncpg.Pool, YugabyteRpcTimeoutRecycler]:
    """a pool of ``size`` connections watched by a recycler, as a platform service builds one.

    :param dsn: database url
    :ptype dsn: str
    :param size: min and max pool size
    :ptype size: int
    :return: the pool and its recycler
    :rtype: tuple[asyncpg.Pool, YugabyteRpcTimeoutRecycler]
    """
    recycler = YugabyteRpcTimeoutRecycler(pool_name="it_pool")
    pool = await asyncpg.create_pool(dsn, min_size=size, max_size=size, init=recycler.init)
    recycler.bind(pool)
    return pool, recycler


@pytest.fixture
async def one_connection_pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    """a recycling pool holding exactly one connection, so every acquire is that connection unless replaced."""
    pool, _ = await _recycling_pool(db_container, size=1)
    try:
        yield pool
    finally:
        await pool.close()


async def _backend_pid(pool: asyncpg.Pool) -> int:
    """the server process id behind the connection the pool hands out next.

    :param pool: pool under test
    :ptype pool: asyncpg.Pool
    :return: backend pid
    :rtype: int
    """
    async with pool.acquire() as conn:
        pid: int = conn.get_server_pid()
    return pid


class TestTheCallerSeesItsOwnError:
    """replacing the connection never changes what the failed query raised."""

    async def test_pool_execute_raises_the_rpc_timeout(self, one_connection_pool: asyncpg.Pool) -> None:
        with pytest.raises(asyncpg.exceptions.InternalServerError, match=YUGABYTE_RPC_TIMEOUT_MESSAGE_PREFIX):
            await one_connection_pool.execute(_RAISE_WEDGE)

    async def test_a_query_inside_a_transaction_raises_the_rpc_timeout(self, one_connection_pool: asyncpg.Pool) -> None:
        """the rollback on the way out runs on the same connection, and still the query's error surfaces."""
        with pytest.raises(asyncpg.exceptions.InternalServerError, match=YUGABYTE_RPC_TIMEOUT_MESSAGE_PREFIX):
            async with one_connection_pool.acquire() as conn, conn.transaction():
                await conn.execute(_RAISE_WEDGE)


class TestTheWedgedConnectionIsReplaced:
    """the next acquire after the error gets a different backend."""

    async def test_after_pool_execute(self, one_connection_pool: asyncpg.Pool) -> None:
        before = await _backend_pid(one_connection_pool)

        with pytest.raises(asyncpg.exceptions.InternalServerError):
            await one_connection_pool.execute(_RAISE_WEDGE)

        assert await _backend_pid(one_connection_pool) != before

    async def test_after_an_extended_protocol_query(self, one_connection_pool: asyncpg.Pool) -> None:
        async with one_connection_pool.acquire() as conn:
            before = conn.get_server_pid()
            await conn.execute(_CREATE_WEDGE_FUNCTION)
            with pytest.raises(asyncpg.exceptions.InternalServerError):
                await conn.fetchval("SELECT pg_temp.answer_rpc_timeout($1)", 1)

        assert await _backend_pid(one_connection_pool) != before

    async def test_after_a_failed_transaction(self, one_connection_pool: asyncpg.Pool) -> None:
        before = await _backend_pid(one_connection_pool)

        with pytest.raises(asyncpg.exceptions.InternalServerError):
            async with one_connection_pool.acquire() as conn, conn.transaction():
                await conn.execute(_RAISE_WEDGE)

        assert await _backend_pid(one_connection_pool) != before

    async def test_every_connection_from_before_the_wedge_is_replaced(self, db_container: str) -> None:
        """the wedge takes every connection opened before it, so one error retires all of them."""
        pool, _ = await _recycling_pool(db_container, size=2)
        try:
            async with pool.acquire() as first, pool.acquire() as second:
                before = {first.get_server_pid(), second.get_server_pid()}
                with pytest.raises(asyncpg.exceptions.InternalServerError):
                    await first.execute(_RAISE_WEDGE)

            async with pool.acquire() as first, pool.acquire() as second:
                after = {first.get_server_pid(), second.get_server_pid()}
        finally:
            await pool.close()

        assert len(before) == 2
        assert not before & after

    async def test_another_error_keeps_the_connection(self, one_connection_pool: asyncpg.Pool) -> None:
        before = await _backend_pid(one_connection_pool)

        with pytest.raises(asyncpg.exceptions.DivisionByZeroError):
            await one_connection_pool.fetchval("SELECT 1 / 0")

        assert await _backend_pid(one_connection_pool) == before


class TestWithoutTheRecycler:
    """the control: an unwatched pool hands the same connection straight back, which is the wedge."""

    async def test_the_same_backend_comes_back(self, db_container: str) -> None:
        pool = await asyncpg.create_pool(db_container, min_size=1, max_size=1)
        try:
            before = await _backend_pid(pool)
            with pytest.raises(asyncpg.exceptions.InternalServerError):
                await pool.execute(_RAISE_WEDGE)
            after = await _backend_pid(pool)
        finally:
            await pool.close()

        assert after == before
