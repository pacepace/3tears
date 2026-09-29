"""tests for threetears.core.utils.yugabyte_pool_recycler.

two YugabyteDB failures leave a pooled connection answering an error a fresh connection would not:

- a wedged tserver session answers every query that reaches the tserver with ``Timed out waiting
  kResponseSent``, while the pool's reset query and ``SELECT 1`` still succeed on it;
- a connection opened before another session altered a table can hold the table's old shape and
  answer ``Invalid column number <n>``.

nothing in asyncpg retires either. one recycler watches every query on every connection of its pool
and, on any of its triggers' errors, expires the pool so asyncpg replaces each connection at its next
release or acquire. the mechanism is tested once per trigger alone, and again with the YugabyteDB
set every platform pool carries.
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

from threetears.core.utils import (
    DEFAULT_MIN_SECONDS_BETWEEN_EXPIRIES,
    YUGABYTE_POOL_TRIGGERS,
    YUGABYTE_RPC_TIMEOUT,
    YUGABYTE_STALE_TABLE_SHAPE,
    PoolExpiryTrigger,
    YugabytePoolRecycler,
    is_yugabyte_rpc_timeout,
    is_yugabyte_stale_table_shape,
)

_LOGGER = "threetears.core.utils.yugabyte_pool_recycler"

#: the two shapes the wedge produced on a local stack after a Docker VM time discontinuity,
#: read from the hub's log on 2026-09-24.
_WEDGE_MESSAGES = (
    "Timed out waiting kResponseSent, state: kRequestSent",
    "Timed out waiting kResponseSent, state: kProcessingRequest",
)

#: the one stale-table-shape error ever recorded, on cobalt-dev, two seconds after the system
#: migrations added columns to ``eval_runs``.
_STALE_SHAPE_MESSAGE = "Invalid column number 19 (table=...eval_runs, schema_version=8, num_columns=18)"

#: YugabyteDB fencing a write that raced a schema change: retryable on the same connection, and
#: produced routinely by an online index build.
_SCHEMA_VERSION_MISMATCH = asyncpg.exceptions.SerializationError(
    "schema version mismatch for table 000034e10000300080000000000040d9: expected 1, got 0",
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


async def _watched(recycler: YugabytePoolRecycler, conn: Any) -> Callable[[LoggedQuery], Any]:
    """register the recycler on ``conn`` and return the logger it installed.

    :param recycler: recycler under test
    :ptype recycler: YugabytePoolRecycler
    :param conn: connection double
    :ptype conn: Any
    :return: the query logger asyncpg would call after each query
    :rtype: Callable[[LoggedQuery], Any]
    """
    recycler.watch(conn)
    assert len(conn.loggers) == 1
    return conn.loggers[0]


_WEDGE = asyncpg.exceptions.InternalServerError(_WEDGE_MESSAGES[0])
_STALE_SHAPE = asyncpg.exceptions.InternalServerError(_STALE_SHAPE_MESSAGE)


class _Case:
    """a recycler watching some triggers, and an error one of them matches."""

    def __init__(self, triggers: tuple[PoolExpiryTrigger, ...], matching: BaseException) -> None:
        self.triggers = triggers
        self.matching = matching

    def build(self, **kwargs: Any) -> YugabytePoolRecycler:
        """a recycler over this case's triggers.

        :param kwargs: the rest of the constructor's arguments
        :ptype kwargs: Any
        :return: the recycler
        :rtype: YugabytePoolRecycler
        """
        return YugabytePoolRecycler(triggers=self.triggers, **kwargs)


_CASES = [
    pytest.param(_Case((YUGABYTE_RPC_TIMEOUT,), _WEDGE), id="rpc-timeout"),
    pytest.param(_Case((YUGABYTE_STALE_TABLE_SHAPE,), _STALE_SHAPE), id="stale-table-shape"),
    pytest.param(_Case(YUGABYTE_POOL_TRIGGERS, _WEDGE), id="default-set-rpc-timeout"),
    pytest.param(_Case(YUGABYTE_POOL_TRIGGERS, _STALE_SHAPE), id="default-set-stale-table-shape"),
]


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


class TestIsYugabyteStaleTableShape:
    """the stale-shape error by its message, and nothing that merely resembles it."""

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(asyncpg.exceptions.InternalServerError(_STALE_SHAPE_MESSAGE), id="xx000"),
            pytest.param(asyncpg.exceptions.InvalidParameterValueError(_STALE_SHAPE_MESSAGE), id="22023"),
            pytest.param(asyncpg.exceptions.InternalServerError(f"ERROR: {_STALE_SHAPE_MESSAGE}"), id="prefixed"),
        ],
    )
    def test_the_recorded_error_matches_whatever_its_class(self, error: BaseException) -> None:
        """the SQLSTATE it arrives with has never been observed, so the class does not decide."""
        assert is_yugabyte_stale_table_shape(error)

    def test_a_schema_version_mismatch_does_not_match(self) -> None:
        """the retryable 40001 fence an online build produces routinely must not churn the pool."""
        assert not is_yugabyte_stale_table_shape(_SCHEMA_VERSION_MISMATCH)

    @pytest.mark.parametrize(
        "message",
        [
            pytest.param("Invalid column number", id="no-number"),
            pytest.param("Invalid column numbers 19", id="plural"),
            pytest.param("xInvalid column number 19", id="mid-word"),
        ],
    )
    def test_near_misses_do_not_match(self, message: str) -> None:
        assert not is_yugabyte_stale_table_shape(asyncpg.exceptions.InternalServerError(message))

    def test_the_words_outside_a_server_error_do_not_match(self) -> None:
        """a client-side error quoting the words is not the server answering them."""
        assert not is_yugabyte_stale_table_shape(ValueError(_STALE_SHAPE_MESSAGE))
        assert not is_yugabyte_stale_table_shape(None)


class TestTriggers:
    """one recycler keys on a sequence of triggers; each trigger alone ignores the others' errors."""

    def test_the_default_is_the_yugabyte_set(self) -> None:
        assert YugabytePoolRecycler(pool_name="p").triggers == YUGABYTE_POOL_TRIGGERS
        assert YUGABYTE_POOL_TRIGGERS == (YUGABYTE_RPC_TIMEOUT, YUGABYTE_STALE_TABLE_SHAPE)

    def test_no_triggers_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one trigger"):
            YugabytePoolRecycler(pool_name="p", triggers=())

    async def test_one_logger_per_connection_for_every_trigger(self) -> None:
        conn = _connection()
        YugabytePoolRecycler(pool_name="hub_l3").watch(conn)
        assert len(conn.loggers) == 1

    async def test_the_rpc_timeout_alone_ignores_the_stale_shape_error(self) -> None:
        recycler = YugabytePoolRecycler(pool_name="hub_l3", triggers=(YUGABYTE_RPC_TIMEOUT,))
        pool = _pool()
        recycler.bind(pool)
        logger = await _watched(recycler, _connection())

        await logger(_record(_STALE_SHAPE))

        pool.expire_connections.assert_not_awaited()

    async def test_the_stale_shape_alone_ignores_the_rpc_timeout(self) -> None:
        recycler = YugabytePoolRecycler(pool_name="hub_l3", triggers=(YUGABYTE_STALE_TABLE_SHAPE,))
        pool = _pool()
        recycler.bind(pool)
        logger = await _watched(recycler, _connection())

        await logger(_record(_WEDGE))

        pool.expire_connections.assert_not_awaited()

    async def test_a_consumer_trigger_decides_by_its_own_predicate(self, caplog: pytest.LogCaptureFixture) -> None:
        """a consumer can key the mechanism on an error 3tears does not name, and its words reach the log."""
        trigger = PoolExpiryTrigger(
            error_name="test failure",
            matches=lambda error: isinstance(error, asyncpg.exceptions.DiskFullError),
            diagnosis="the test says so",
            persistent_cause="the test",
        )
        recycler = YugabytePoolRecycler(pool_name="custom_pool", triggers=(trigger,))
        pool = _pool()
        recycler.bind(pool)
        logger = await _watched(recycler, _connection())

        await logger(_record(_WEDGE))
        pool.expire_connections.assert_not_awaited()

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            await logger(_record(asyncpg.exceptions.DiskFullError("disk full")))
        pool.expire_connections.assert_awaited_once_with()
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("custom_pool" in m and "test failure" in m and "the test says so" in m for m in warnings), warnings


class TestTheDefaultSetSharesOnePoolRecord:
    """the pool's last expiry is one fact whatever caused it; each trigger keeps its own floor."""

    async def test_an_old_connection_answering_the_other_error_is_already_being_replaced(self) -> None:
        # an expiry retires every connection opened before it, whatever the cause, so a pre-expiry
        # connection answering the OTHER trigger changes nothing.
        clock = _Clock()
        recycler = YugabytePoolRecycler(pool_name="hub_l3", clock=clock)
        pool = _pool()
        recycler.bind(pool)
        first = await _watched(recycler, _connection(pid=1))
        second = await _watched(recycler, _connection(pid=2))

        clock.now += 1.0
        await first(_record(_WEDGE))
        clock.now += DEFAULT_MIN_SECONDS_BETWEEN_EXPIRIES + 1.0
        await second(_record(_STALE_SHAPE))

        pool.expire_connections.assert_awaited_once_with()

    async def test_a_new_cause_inside_the_other_triggers_floor_still_expires(self) -> None:
        # the floor says "this error on new connections means the cause is not the connections".
        # a stale shape seconds after a wedge was cleared is a different cause, not the wedge
        # persisting, so the wedge's floor does not hold it off.
        clock = _Clock()
        recycler = YugabytePoolRecycler(pool_name="hub_l3", clock=clock)
        pool = _pool()
        recycler.bind(pool)
        old = await _watched(recycler, _connection(pid=1))
        clock.now += 1.0
        await old(_record(_WEDGE))

        clock.now += 1.0
        fresh = await _watched(recycler, _connection(pid=2))
        clock.now += 1.0
        await fresh(_record(_STALE_SHAPE))

        assert pool.expire_connections.await_count == 2

    async def test_the_same_cause_inside_its_own_floor_is_held_off(self) -> None:
        clock = _Clock()
        recycler = YugabytePoolRecycler(pool_name="hub_l3", clock=clock)
        pool = _pool()
        recycler.bind(pool)
        old = await _watched(recycler, _connection(pid=1))
        clock.now += 1.0
        await old(_record(_STALE_SHAPE))

        clock.now += 1.0
        fresh = await _watched(recycler, _connection(pid=2))
        clock.now += 1.0
        await fresh(_record(_STALE_SHAPE))

        pool.expire_connections.assert_awaited_once_with()


class TestRecyclerWatchesEveryConnection:
    """``init`` is the pool's ``init=`` hook: the canonical codec setup, then the watch."""

    @pytest.mark.parametrize("case", _CASES)
    async def test_init_registers_the_codecs_then_watches(self, case: _Case) -> None:
        recycler = case.build(pool_name="hub_l3")
        conn = _connection()

        await recycler.init(conn)

        codec_types = [call.args[0] for call in conn.set_type_codec.await_args_list]
        assert codec_types == ["jsonb", "json"]
        assert len(conn.loggers) == 1


@pytest.mark.parametrize("case", _CASES)
class TestRecyclerExpiresThePool:
    """a query answering the trigger's error expires the pool, once per cause."""

    async def test_the_error_expires_the_pool(self, case: _Case, caplog: pytest.LogCaptureFixture) -> None:
        clock = _Clock()
        recycler = case.build(pool_name="hub_l3", clock=clock)
        pool = _pool()
        recycler.bind(pool)
        logger = await _watched(recycler, _connection(pid=4242))
        clock.now += 3600.0

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            await logger(_record(case.matching))

        pool.expire_connections.assert_awaited_once_with()
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "hub_l3" in m and "4242" in m and "3600" in m and any(t.error_name in m for t in recycler.triggers)
            for m in warnings
        ), warnings

    @pytest.mark.parametrize(
        "exception",
        [
            pytest.param(None, id="query-succeeded"),
            pytest.param(asyncpg.exceptions.UniqueViolationError("duplicate key"), id="unique-violation"),
            pytest.param(asyncpg.exceptions.InternalServerError("could not open relation"), id="other-xx000"),
            pytest.param(_SCHEMA_VERSION_MISMATCH, id="schema-version-mismatch"),
            pytest.param(asyncio.CancelledError(), id="cancelled"),
        ],
    )
    async def test_anything_else_leaves_the_pool_alone(self, case: _Case, exception: BaseException | None) -> None:
        recycler = case.build(pool_name="hub_l3")
        pool = _pool()
        recycler.bind(pool)
        logger = await _watched(recycler, _connection())

        await logger(_record(exception))

        pool.expire_connections.assert_not_awaited()

    async def test_a_connection_opened_before_the_expiry_does_not_expire_again(self, case: _Case) -> None:
        """every connection from before the cause answers the error once; one expiry retires them all."""
        clock = _Clock()
        recycler = case.build(pool_name="hub_l3", clock=clock)
        pool = _pool()
        recycler.bind(pool)
        first = await _watched(recycler, _connection(pid=1))
        second = await _watched(recycler, _connection(pid=2))

        clock.now += 1.0
        await first(_record(case.matching))
        clock.now += DEFAULT_MIN_SECONDS_BETWEEN_EXPIRIES + 1.0
        await second(_record(case.matching))

        pool.expire_connections.assert_awaited_once_with()

    async def test_a_connection_opened_after_the_expiry_expires_again_past_the_floor(self, case: _Case) -> None:
        """a fresh connection answering the error is a new cause, and the pool is expired again."""
        clock = _Clock()
        recycler = case.build(pool_name="hub_l3", clock=clock)
        pool = _pool()
        recycler.bind(pool)
        old = await _watched(recycler, _connection(pid=1))
        clock.now += 1.0
        await old(_record(case.matching))

        clock.now += 1.0
        fresh = await _watched(recycler, _connection(pid=2))
        clock.now += DEFAULT_MIN_SECONDS_BETWEEN_EXPIRIES
        await fresh(_record(case.matching))

        assert pool.expire_connections.await_count == 2

    async def test_the_floor_holds_off_a_second_expiry(self, case: _Case) -> None:
        """a cause that new connections answer too is not met with a reconnect storm."""
        clock = _Clock()
        recycler = case.build(pool_name="hub_l3", clock=clock)
        pool = _pool()
        recycler.bind(pool)
        old = await _watched(recycler, _connection(pid=1))
        clock.now += 1.0
        await old(_record(case.matching))

        clock.now += 1.0
        fresh = await _watched(recycler, _connection(pid=2))
        clock.now += 1.0
        await fresh(_record(case.matching))

        pool.expire_connections.assert_awaited_once_with()

    async def test_an_unbound_recycler_says_so_at_error(self, case: _Case, caplog: pytest.LogCaptureFixture) -> None:
        """a recycler never bound to its pool cannot recover it, and the log names the missing bind."""
        recycler = case.build(pool_name="gateway_l3")
        logger = await _watched(recycler, _connection())

        with caplog.at_level(logging.ERROR, logger=_LOGGER):
            await logger(_record(case.matching))

        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any("gateway_l3" in m and "bind" in m for m in errors), errors

    def test_binding_a_second_pool_is_refused(self, case: _Case) -> None:
        """one recycler watches one pool: a second bind would expire the wrong pool's connections."""
        recycler = case.build(pool_name="hub_l3")
        recycler.bind(_pool())

        with pytest.raises(RuntimeError, match="hub_l3"):
            recycler.bind(_pool())

    def test_the_floor_must_be_positive(self, case: _Case) -> None:
        with pytest.raises(ValueError, match="min_seconds_between_expiries"):
            case.build(pool_name="hub_l3", min_seconds_between_expiries=0.0)
