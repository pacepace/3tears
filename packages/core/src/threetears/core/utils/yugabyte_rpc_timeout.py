"""replace a pool's connections when YugabyteDB wedges the tserver sessions behind them.

the failure
-----------

a single-node YugabyteDB can wedge the tserver session behind an open connection -- on a Docker
Desktop VM time discontinuity or CPU throttling, with no host sleep required. every query on that
connection that reaches the tserver then fails in milliseconds with SQLSTATE XX000,
``Timed out waiting kResponseSent, state: <kRequestSent | kProcessingRequest>``, while a fresh
connection works. nothing in asyncpg retires such a connection: the pool's reset query and a
``SELECT 1`` never reach the tserver, so they succeed on it; and the pool is LIFO, so a connection
in use every few seconds is never idle long enough for ``max_inactive_connection_lifetime``. every
connection opened before the wedge is wedged, so terminating one only hands the next caller another.

the recovery
------------

:class:`YugabyteRpcTimeoutRecycler` registers an asyncpg query logger on every connection its pool
opens, so it sees every query, whether it came through a collection or a raw ``pool.fetch``. when a
query answers the RPC timeout it calls ``Pool.expire_connections()``, asyncpg's own way to retire a
pool's connections: each connection opened before the call is closed at its next release or
acquire and replaced with a fresh one. terminating the failed connection directly instead is not
safe -- the logger runs while the pool is still releasing that connection, and asyncpg would answer
the caller with ``InternalClientError`` in place of the error its query raised.

usage::

    recycler = YugabyteRpcTimeoutRecycler(pool_name="hub_l3")
    pool = await asyncpg.create_pool(dsn, init=recycler.init, **get_pg_pool_kwargs())
    recycler.bind(pool)

a consumer composing its own ``init`` calls :func:`~threetears.core.collections.init_connection`
and then :meth:`YugabyteRpcTimeoutRecycler.watch`.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import asyncpg

from threetears.core.collections.asyncpg_init import init_connection
from threetears.observe import get_logger

if TYPE_CHECKING:
    from asyncpg.connection import LoggedQuery

__all__ = [
    "DEFAULT_MIN_SECONDS_BETWEEN_EXPIRIES",
    "YUGABYTE_RPC_TIMEOUT_MESSAGE_PREFIX",
    "YugabyteRpcTimeoutRecycler",
    "is_yugabyte_rpc_timeout",
]

log = get_logger(__name__)

#: the start of every message YugabyteDB answers when a tserver RPC behind the connection times out.
#: the state that follows varies (``kRequestSent``, ``kProcessingRequest``).
YUGABYTE_RPC_TIMEOUT_MESSAGE_PREFIX = "Timed out waiting kResponseSent"

#: the least time between two expiries of one pool. a cluster that answers the RPC timeout on
#: brand-new connections too (a tserver genuinely down or overloaded) is not helped by replacing
#: connections again, and a reconnect for every failed query would add load to it.
DEFAULT_MIN_SECONDS_BETWEEN_EXPIRIES = 10.0


def is_yugabyte_rpc_timeout(error: BaseException | None) -> bool:
    """say whether ``error`` is YugabyteDB answering that a tserver RPC behind the connection timed out.

    the error arrives as SQLSTATE XX000, which asyncpg raises as
    :class:`asyncpg.exceptions.InternalServerError`. the same words on any other class are some other
    failure and do not match.

    :param error: what a query raised, or None when it succeeded
    :ptype error: BaseException | None
    :return: True for the YugabyteDB RPC timeout
    :rtype: bool
    """
    return isinstance(error, asyncpg.exceptions.InternalServerError) and str(error).startswith(
        YUGABYTE_RPC_TIMEOUT_MESSAGE_PREFIX
    )


class YugabyteRpcTimeoutRecycler:
    """expire one pool's connections when a query on any of them answers the YugabyteDB RPC timeout.

    one expiry retires every connection opened before it, so a connection opened before the last
    expiry answering the timeout changes nothing and is skipped. one opened after it is a new wedge,
    and expires the pool again once ``min_seconds_between_expiries`` has passed.

    :param pool_name: the pool's name in logs, as given to ``log_pool_created``
    :ptype pool_name: str
    :param min_seconds_between_expiries: the least time between two expiries of the pool
    :ptype min_seconds_between_expiries: float
    :param clock: monotonic clock in seconds; tests pass their own
    :ptype clock: Callable[[], float]
    :raises ValueError: if ``min_seconds_between_expiries`` is not positive
    """

    def __init__(
        self,
        *,
        pool_name: str,
        min_seconds_between_expiries: float = DEFAULT_MIN_SECONDS_BETWEEN_EXPIRIES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if min_seconds_between_expiries <= 0:
            raise ValueError(
                f"min_seconds_between_expiries must be positive, got {min_seconds_between_expiries!r}",
            )
        self._pool_name = pool_name
        self._min_seconds_between_expiries = min_seconds_between_expiries
        self._clock = clock
        self._pool: asyncpg.Pool | None = None
        self._expired_at: float | None = None

    def bind(self, pool: asyncpg.Pool) -> None:
        """name the pool this recycler expires; call once, right after the pool is created.

        :param pool: the pool whose ``init`` is :meth:`init` or calls :meth:`watch`
        :ptype pool: asyncpg.Pool
        :return: nothing
        :rtype: None
        :raises RuntimeError: if the recycler is already bound to another pool
        """
        if self._pool is not None and self._pool is not pool:
            raise RuntimeError(
                f"recycler for pool {self._pool_name} is already bound; one recycler watches one pool",
            )
        self._pool = pool

    async def init(self, conn: asyncpg.Connection) -> None:
        """pool ``init=`` hook: the canonical 3tears connection setup, then :meth:`watch`.

        :param conn: a connection the pool just opened
        :ptype conn: asyncpg.Connection
        :return: nothing
        :rtype: None
        """
        await init_connection(conn)
        self.watch(conn)

    def watch(self, conn: asyncpg.Connection) -> None:
        """register the query logger that reports this connection's RPC timeouts to the recycler.

        asyncpg keeps a query logger for the connection's life; it is dropped only when the
        connection closes.

        :param conn: a connection the pool just opened
        :ptype conn: asyncpg.Connection
        :return: nothing
        :rtype: None
        """
        conn.add_query_logger(self._logger_for(opened_at=self._clock(), backend_pid=conn.get_server_pid()))

    def _logger_for(self, *, opened_at: float, backend_pid: int) -> Callable[[LoggedQuery], Awaitable[None]]:
        """build the query logger for one connection.

        asyncpg runs a coroutine logger as its own task, scheduled when the query finishes, so the
        expiry is in place before the pool finishes releasing the connection.

        :param opened_at: clock reading when the connection was opened
        :ptype opened_at: float
        :param backend_pid: the connection's server process id, for the log line
        :ptype backend_pid: int
        :return: the logger asyncpg calls after each query on the connection
        :rtype: Callable[[LoggedQuery], Awaitable[None]]
        """

        async def _on_query(record: LoggedQuery) -> None:
            if is_yugabyte_rpc_timeout(record.exception):
                await self._on_rpc_timeout(opened_at=opened_at, backend_pid=backend_pid, error=record.exception)

        return _on_query

    async def _on_rpc_timeout(self, *, opened_at: float, backend_pid: int, error: BaseException | None) -> None:
        """expire the pool, unless an earlier expiry already covers this connection or the floor holds it off.

        :param opened_at: clock reading when the failing connection was opened
        :ptype opened_at: float
        :param backend_pid: the failing connection's server process id
        :ptype backend_pid: int
        :param error: the RPC timeout the query raised
        :ptype error: BaseException | None
        :return: nothing
        :rtype: None
        """
        now = self._clock()
        if self._pool is None:
            log.error(
                "a query on pool %s answered the YugabyteDB RPC timeout, but its recycler was never bound "
                "to the pool, so no connection is replaced; call bind(pool) right after creating it: "
                "backend_pid=%s error=%s",
                self._pool_name,
                backend_pid,
                error,
            )
        elif self._expired_at is not None and opened_at <= self._expired_at:
            log.debug(
                "pool %s: connection from before the last expiry answered the RPC timeout; already being replaced: "
                "backend_pid=%s",
                self._pool_name,
                backend_pid,
            )
        elif self._expired_at is not None and now - self._expired_at < self._min_seconds_between_expiries:
            log.warning(
                "pool %s: a connection opened after the last expiry answered the YugabyteDB RPC timeout %.1fs "
                "later; not expiring again within %.0fs, since new connections failing too means the cluster, "
                "not the connections: backend_pid=%s error=%s",
                self._pool_name,
                now - self._expired_at,
                self._min_seconds_between_expiries,
                backend_pid,
                error,
            )
        else:
            log.warning(
                "pool %s: a query answered the YugabyteDB RPC timeout, so the tserver session behind its "
                "connection is wedged; expiring every connection in the pool so each is replaced at its next "
                "release or acquire: backend_pid=%s connection_age_seconds=%.0f error=%s",
                self._pool_name,
                backend_pid,
                now - opened_at,
                error,
            )
            self._expired_at = now
            await self._pool.expire_connections()
