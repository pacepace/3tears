"""tests for threetears.core.utils.yugabyte_rpc_timeout.

a pooled connection whose YugabyteDB tserver session wedged answers every query that reaches the
tserver with ``Timed out waiting kResponseSent``, while the pool's reset query and ``SELECT 1`` still
succeed on it, so nothing in asyncpg ever retires it. the recycler watches every query on every
connection of its pool and, on that error, expires the pool so asyncpg replaces each connection at
its next release or acquire.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any
from unittest.mock import create_autospec

import asyncpg
import pytest
from asyncpg.connection import LoggedQuery

from threetears.core.utils.yugabyte_rpc_timeout import (
    DEFAULT_MIN_SECONDS_BETWEEN_EXPIRIES,
    YugabyteRpcTimeoutRecycler,
    is_yugabyte_rpc_timeout,
)

#: the two shapes the wedge produced on a local stack after a Docker VM time discontinuity,
#: read from the hub's log on 2026-09-24.
_WEDGE_MESSAGES = (
    "Timed out waiting kResponseSent, state: kRequestSent",
    "Timed out waiting kResponseSent, state: kProcessingRequest",
)


class _Clock:
    """a monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _connection(pid: int = 4242) -> Any:
    """an autospecced asyncpg connection that keeps the query logger it is given.

    :param pid: backend pid the connection reports
    :ptype pid: int
    :return: connection double; ``loggers`` holds every logger registered on it
    :rtype: Any
    """
    conn = create_autospec(asyncpg.Connection, instance=True)
    conn.loggers = []
    conn.add_query_logger.side_effect = conn.loggers.append
    conn.get_server_pid.return_value = pid
    return conn


def _pool() -> Any:
    """an autospecced asyncpg pool.

    :return: pool double whose ``expire_connections`` is awaitable
    :rtype: Any
    """
    return create_autospec(asyncpg.Pool, instance=True)


def _record(exception: BaseException | None) -> LoggedQuery:
    """the record asyncpg hands a query logger after a query.

    :param exception: what the query raised, or None when it succeeded
    :ptype exception: BaseException | None
    :return: query log record
    :rtype: LoggedQuery
    """
    return LoggedQuery(
        query="SELECT id FROM agents WHERE agent_id = $1",
        args=("x",),
        timeout=30.0,
        elapsed=0.014,
        exception=exception,
        conn_addr=("yugabytedb", 5433),
        conn_params=None,
    )


async def _watched(recycler: YugabyteRpcTimeoutRecycler, conn: Any) -> Callable[[LoggedQuery], Any]:
    """register the recycler on ``conn`` and return the logger it installed.

    :param recycler: recycler under test
    :ptype recycler: YugabyteRpcTimeoutRecycler
    :param conn: connection double
    :ptype conn: Any
    :return: the query logger asyncpg would call after each query
    :rtype: Callable[[LoggedQuery], Any]
    """
    recycler.watch(conn)
    assert len(conn.loggers) == 1
    return conn.loggers[0]


class TestIsYugabyteRpcTimeout:
    """the predicate names the wedge's error and nothing else."""

    @pytest.mark.parametrize("message", _WEDGE_MESSAGES)
    def test_the_wedge_error_is_recognised(self, message: str) -> None:
        assert is_yugabyte_rpc_timeout(asyncpg.exceptions.InternalServerError(message))

    def test_another_internal_server_error_is_not(self) -> None:
        assert not is_yugabyte_rpc_timeout(asyncpg.exceptions.InternalServerError("could not open relation"))

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(RuntimeError(_WEDGE_MESSAGES[0]), id="runtime-error"),
            pytest.param(asyncpg.exceptions.PostgresConnectionError(_WEDGE_MESSAGES[0]), id="connection-error"),
            pytest.param(asyncpg.exceptions.QueryCanceledError(_WEDGE_MESSAGES[0]), id="query-canceled"),
        ],
    )
    def test_the_message_on_another_class_is_not(self, error: BaseException) -> None:
        """the wedge arrives as SQLSTATE XX000; the same words on another class are some other failure."""
        assert not is_yugabyte_rpc_timeout(error)

    def test_no_error_is_not(self) -> None:
        assert not is_yugabyte_rpc_timeout(None)


class TestRecyclerWatchesEveryConnection:
    """``init`` is the pool's ``init=`` hook: the canonical codec setup, then the watch."""

    async def test_init_registers_the_codecs_then_watches(self) -> None:
        recycler = YugabyteRpcTimeoutRecycler(pool_name="hub_l3")
        conn = _connection()

        await recycler.init(conn)

        codec_types = [call.args[0] for call in conn.set_type_codec.await_args_list]
        assert codec_types == ["jsonb", "json"]
        assert len(conn.loggers) == 1


class TestRecyclerExpiresTheWedgedPool:
    """a query answering the RPC timeout expires the pool, once per wedge."""

    @pytest.mark.parametrize("message", _WEDGE_MESSAGES)
    async def test_the_wedge_error_expires_the_pool(self, message: str, caplog: pytest.LogCaptureFixture) -> None:
        clock = _Clock()
        recycler = YugabyteRpcTimeoutRecycler(pool_name="hub_l3", clock=clock)
        pool = _pool()
        recycler.bind(pool)
        logger = await _watched(recycler, _connection(pid=4242))
        clock.now += 3600.0

        with caplog.at_level(logging.WARNING, logger="threetears.core.utils.yugabyte_rpc_timeout"):
            await logger(_record(asyncpg.exceptions.InternalServerError(message)))

        pool.expire_connections.assert_awaited_once_with()
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("hub_l3" in m and "4242" in m and "3600" in m for m in warnings), warnings

    @pytest.mark.parametrize(
        "exception",
        [
            pytest.param(None, id="query-succeeded"),
            pytest.param(asyncpg.exceptions.UniqueViolationError("duplicate key"), id="unique-violation"),
            pytest.param(asyncpg.exceptions.InternalServerError("could not open relation"), id="other-xx000"),
            pytest.param(asyncio.CancelledError(), id="cancelled"),
        ],
    )
    async def test_anything_else_leaves_the_pool_alone(self, exception: BaseException | None) -> None:
        recycler = YugabyteRpcTimeoutRecycler(pool_name="hub_l3")
        pool = _pool()
        recycler.bind(pool)
        logger = await _watched(recycler, _connection())

        await logger(_record(exception))

        pool.expire_connections.assert_not_awaited()

    async def test_a_connection_opened_before_the_expiry_does_not_expire_again(self) -> None:
        """every connection from before the wedge answers the timeout once; one expiry retires them all."""
        clock = _Clock()
        recycler = YugabyteRpcTimeoutRecycler(pool_name="hub_l3", clock=clock)
        pool = _pool()
        recycler.bind(pool)
        first = await _watched(recycler, _connection(pid=1))
        second = await _watched(recycler, _connection(pid=2))
        wedge = asyncpg.exceptions.InternalServerError(_WEDGE_MESSAGES[0])

        clock.now += 1.0
        await first(_record(wedge))
        clock.now += DEFAULT_MIN_SECONDS_BETWEEN_EXPIRIES + 1.0
        await second(_record(wedge))

        pool.expire_connections.assert_awaited_once_with()

    async def test_a_connection_opened_after_the_expiry_expires_again_past_the_floor(self) -> None:
        """a fresh connection answering the timeout is a new wedge, and the pool is expired again."""
        clock = _Clock()
        recycler = YugabyteRpcTimeoutRecycler(pool_name="hub_l3", clock=clock)
        pool = _pool()
        recycler.bind(pool)
        wedge = asyncpg.exceptions.InternalServerError(_WEDGE_MESSAGES[0])
        old = await _watched(recycler, _connection(pid=1))
        clock.now += 1.0
        await old(_record(wedge))

        clock.now += 1.0
        fresh = await _watched(recycler, _connection(pid=2))
        clock.now += DEFAULT_MIN_SECONDS_BETWEEN_EXPIRIES
        await fresh(_record(wedge))

        assert pool.expire_connections.await_count == 2

    async def test_the_floor_holds_off_a_second_expiry(self) -> None:
        """a cluster answering the timeout on every new connection is not answered with a reconnect storm."""
        clock = _Clock()
        recycler = YugabyteRpcTimeoutRecycler(pool_name="hub_l3", clock=clock)
        pool = _pool()
        recycler.bind(pool)
        wedge = asyncpg.exceptions.InternalServerError(_WEDGE_MESSAGES[0])
        old = await _watched(recycler, _connection(pid=1))
        clock.now += 1.0
        await old(_record(wedge))

        clock.now += 1.0
        fresh = await _watched(recycler, _connection(pid=2))
        clock.now += 1.0
        await fresh(_record(wedge))

        pool.expire_connections.assert_awaited_once_with()

    async def test_an_unbound_recycler_says_so_at_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """a recycler never bound to its pool cannot recover it, and the log names the missing bind."""
        recycler = YugabyteRpcTimeoutRecycler(pool_name="gateway_l3")
        logger = await _watched(recycler, _connection())

        with caplog.at_level(logging.ERROR, logger="threetears.core.utils.yugabyte_rpc_timeout"):
            await logger(_record(asyncpg.exceptions.InternalServerError(_WEDGE_MESSAGES[0])))

        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any("gateway_l3" in m and "bind" in m for m in errors), errors

    def test_binding_a_second_pool_is_refused(self) -> None:
        """one recycler watches one pool: a second bind would expire the wrong pool's connections."""
        recycler = YugabyteRpcTimeoutRecycler(pool_name="hub_l3")
        recycler.bind(_pool())

        with pytest.raises(RuntimeError, match="hub_l3"):
            recycler.bind(_pool())

    def test_the_floor_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="min_seconds_between_expiries"):
            YugabyteRpcTimeoutRecycler(pool_name="hub_l3", min_seconds_between_expiries=0.0)
