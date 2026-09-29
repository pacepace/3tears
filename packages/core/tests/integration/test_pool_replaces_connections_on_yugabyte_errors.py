"""a real asyncpg pool replaces its connections when one answers a YugabyteDB error its recycler keys on.

Postgres cannot wedge a tserver session or cache a stale table shape, but it can raise the errors
those answer: YugabyteDB's messages under a chosen SQLSTATE, which asyncpg raises exactly as it does
for YugabyteDB. these tests drive real pool mechanics -- release, reset, the generation check -- so
they show what the unit tests cannot: that the caller still sees the error its query raised, and
that the next acquire gets a different server backend. every test runs once per trigger.

the control test runs the same sequence on a pool without a recycler and sees the same backend come
back, which is the defect.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

import asyncpg
import pytest

from threetears.core.utils import (
    YUGABYTE_POOL_TRIGGERS,
    YUGABYTE_RPC_TIMEOUT,
    YUGABYTE_RPC_TIMEOUT_MESSAGE_PREFIX,
    YUGABYTE_STALE_TABLE_SHAPE,
    PoolExpiryTrigger,
    YugabytePoolRecycler,
)

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class _Trigger:
    """the triggers a recycler watches, and a statement that makes Postgres answer one of their errors."""

    triggers: tuple[PoolExpiryTrigger, ...]
    sqlstate: str
    message: str

    @property
    def raise_statement(self) -> str:
        """one statement on the simple-query path (``execute`` with no arguments)."""
        return f"DO $$ BEGIN RAISE EXCEPTION USING ERRCODE = '{self.sqlstate}', MESSAGE = '{self.message}'; END $$"

    @property
    def create_function(self) -> str:
        """a function, so a query on the extended-protocol path (``fetchval``) raises it too."""
        return f"""
CREATE OR REPLACE FUNCTION pg_temp.answer_the_error(x int) RETURNS int LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION USING ERRCODE = '{self.sqlstate}', MESSAGE = '{self.message}';
END $$
"""


_RPC_TIMEOUT = _Trigger(
    triggers=(YUGABYTE_RPC_TIMEOUT,),
    sqlstate="XX000",
    message=f"{YUGABYTE_RPC_TIMEOUT_MESSAGE_PREFIX}, state: kRequestSent",
)
# the stale-shape error's SQLSTATE has never been observed; XX000 is YugabyteDB's usual one for an
# internal consistency error, and the recycler matches the message whatever the class.
_STALE_TABLE_SHAPE = _Trigger(
    triggers=(YUGABYTE_STALE_TABLE_SHAPE,),
    sqlstate="XX000",
    message="Invalid column number 19 (table=eval_runs, schema_version=8, num_columns=18)",
)


@pytest.fixture(params=[_RPC_TIMEOUT, _STALE_TABLE_SHAPE], ids=["rpc-timeout", "stale-table-shape"])
def trigger(request: pytest.FixtureRequest) -> _Trigger:
    """each recycler in turn."""
    chosen: _Trigger = request.param
    return chosen


async def _recycling_pool(dsn: str, trigger: _Trigger, *, size: int) -> asyncpg.Pool:
    """a pool of ``size`` connections watched by the trigger's recycler, as a platform service builds one.

    :param dsn: database url
    :ptype dsn: str
    :param trigger: which recycler watches the pool
    :ptype trigger: _Trigger
    :param size: min and max pool size
    :ptype size: int
    :return: the pool
    :rtype: asyncpg.Pool
    """
    recycler = YugabytePoolRecycler(pool_name="it_pool", triggers=trigger.triggers)
    pool = await asyncpg.create_pool(dsn, min_size=size, max_size=size, init=recycler.init)
    recycler.bind(pool)
    return pool


@pytest.fixture
async def one_connection_pool(db_container: str, trigger: _Trigger) -> AsyncIterator[asyncpg.Pool]:
    """a recycling pool holding exactly one connection, so every acquire is that connection unless replaced."""
    pool = await _recycling_pool(db_container, trigger, size=1)
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

    async def test_pool_execute_raises_the_error(self, one_connection_pool: asyncpg.Pool, trigger: _Trigger) -> None:
        with pytest.raises(asyncpg.exceptions.InternalServerError, match=re.escape(trigger.message)):
            await one_connection_pool.execute(trigger.raise_statement)

    async def test_a_query_inside_a_transaction_raises_the_error(
        self, one_connection_pool: asyncpg.Pool, trigger: _Trigger
    ) -> None:
        """the rollback on the way out runs on the same connection, and still the query's error surfaces."""
        with pytest.raises(asyncpg.exceptions.InternalServerError, match=re.escape(trigger.message)):
            async with one_connection_pool.acquire() as conn, conn.transaction():
                await conn.execute(trigger.raise_statement)


class TestTheConnectionIsReplaced:
    """the next acquire after the error gets a different backend."""

    async def test_after_pool_execute(self, one_connection_pool: asyncpg.Pool, trigger: _Trigger) -> None:
        before = await _backend_pid(one_connection_pool)

        with pytest.raises(asyncpg.exceptions.InternalServerError):
            await one_connection_pool.execute(trigger.raise_statement)

        assert await _backend_pid(one_connection_pool) != before

    async def test_after_an_extended_protocol_query(self, one_connection_pool: asyncpg.Pool, trigger: _Trigger) -> None:
        async with one_connection_pool.acquire() as conn:
            before = conn.get_server_pid()
            await conn.execute(trigger.create_function)
            with pytest.raises(asyncpg.exceptions.InternalServerError):
                await conn.fetchval("SELECT pg_temp.answer_the_error($1)", 1)

        assert await _backend_pid(one_connection_pool) != before

    async def test_after_a_failed_transaction(self, one_connection_pool: asyncpg.Pool, trigger: _Trigger) -> None:
        before = await _backend_pid(one_connection_pool)

        with pytest.raises(asyncpg.exceptions.InternalServerError):
            async with one_connection_pool.acquire() as conn, conn.transaction():
                await conn.execute(trigger.raise_statement)

        assert await _backend_pid(one_connection_pool) != before

    async def test_every_connection_from_before_the_error_is_replaced(
        self, db_container: str, trigger: _Trigger
    ) -> None:
        """every connection opened before the cause shares it, so one error retires all of them."""
        pool = await _recycling_pool(db_container, trigger, size=2)
        try:
            async with pool.acquire() as first, pool.acquire() as second:
                before = {first.get_server_pid(), second.get_server_pid()}
                with pytest.raises(asyncpg.exceptions.InternalServerError):
                    await first.execute(trigger.raise_statement)

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


class TestTheStaleShapeRecyclerLeavesTheRetryableFenceAlone:
    """YugabyteDB's 40001 schema-version fence is retried on the SAME connection; replacing it gains nothing."""

    async def test_a_schema_version_mismatch_keeps_the_connection(self, db_container: str) -> None:
        pool = await _recycling_pool(db_container, _STALE_TABLE_SHAPE, size=1)
        fence = _Trigger(
            triggers=(YUGABYTE_STALE_TABLE_SHAPE,),
            sqlstate="40001",
            message="schema version mismatch for table 000034e10000300080000000000040d9: expected 1, got 0",
        )
        try:
            before = await _backend_pid(pool)
            with pytest.raises(asyncpg.exceptions.SerializationError):
                await pool.execute(fence.raise_statement)
            after = await _backend_pid(pool)
        finally:
            await pool.close()

        assert after == before


class TestTheDefaultYugabyteSet:
    """one recycler with the YugabyteDB default set, as a platform pool is: either error replaces its connections."""

    async def test_either_error_replaces_the_connection(self, db_container: str) -> None:
        recycler = YugabytePoolRecycler(pool_name="it_pool")
        assert recycler.triggers == YUGABYTE_POOL_TRIGGERS
        pool = await asyncpg.create_pool(db_container, min_size=1, max_size=1, init=recycler.init)
        recycler.bind(pool)
        try:
            first = await _backend_pid(pool)
            with pytest.raises(asyncpg.exceptions.InternalServerError):
                await pool.execute(_STALE_TABLE_SHAPE.raise_statement)
            second = await _backend_pid(pool)
            with pytest.raises(asyncpg.exceptions.InternalServerError):
                await pool.execute(_RPC_TIMEOUT.raise_statement)
            third = await _backend_pid(pool)
        finally:
            await pool.close()

        assert len({first, second, third}) == 3


class TestWithoutARecycler:
    """the control: an unwatched pool hands the same connection straight back, which is the defect."""

    async def test_the_same_backend_comes_back(self, db_container: str, trigger: _Trigger) -> None:
        pool = await asyncpg.create_pool(db_container, min_size=1, max_size=1)
        try:
            before = await _backend_pid(pool)
            with pytest.raises(asyncpg.exceptions.InternalServerError):
                await pool.execute(trigger.raise_statement)
            after = await _backend_pid(pool)
        finally:
            await pool.close()

        assert after == before
