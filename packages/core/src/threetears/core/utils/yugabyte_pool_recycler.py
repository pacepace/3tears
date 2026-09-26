"""replace a pool's connections when YugabyteDB answers an error only a fresh connection escapes.

the failures
------------

**A wedged tserver session.** A single-node YugabyteDB can wedge the tserver session behind an
open connection -- on a Docker Desktop VM time discontinuity or CPU throttling, with no host sleep
required. Every query on that connection that reaches the tserver then fails in milliseconds with
SQLSTATE XX000, ``Timed out waiting kResponseSent, state: <kRequestSent | kProcessingRequest>``,
while a fresh connection works. Every connection opened before the wedge is wedged, so terminating
one only hands the next caller another.

**A stale table shape.** A pooled connection can hold a table's OLD shape after another session
altered the table. On cobalt-dev, two seconds after the system migrations added columns to
``eval_runs``, the hub's first query on it failed with ``Invalid column number 19
(table=...eval_runs, schema_version=8, num_columns=18)``: schema version 8 was the table before
those migrations, and the connection that answered had been opened before them. A new connection
loads the schema as it stands. A refresh at the moment a process KNOWS it changed a schema cannot
cover the rest -- another process's migrations, a data sync's ``ALTER TABLE`` on its own
connection, a table created from a template -- so this keys on the failure instead of the cause.

Nothing in asyncpg retires either kind of connection: the pool's reset query and a ``SELECT 1``
succeed on it, and the pool is LIFO, so a connection in use every few seconds is never idle long
enough for ``max_inactive_connection_lifetime``.

the recovery
------------

:class:`YugabytePoolRecycler` registers an asyncpg query logger on every connection its pool opens,
so it sees every query, whether it came through a collection or a raw ``pool.fetch``. When a query
answers one of its triggers' errors it calls ``Pool.expire_connections()``, asyncpg's own way to
retire a pool's connections: each connection opened before the call is closed at its next release
or acquire and replaced with a fresh one. Terminating the failed connection directly instead is not
safe -- the logger runs while the pool is still releasing that connection, and asyncpg would answer
the caller with ``InternalClientError`` in place of the error its query raised.

What it keys on is a sequence of :class:`PoolExpiryTrigger`. The YugabyteDB set,
:data:`YUGABYTE_POOL_TRIGGERS`, is the default, so a platform pool is one recycler, one logger per
connection, one ``bind`` and one ``init``, and a trigger added to the set reaches every such pool.

usage::

    recycler = YugabytePoolRecycler(pool_name="hub_l3")
    pool = await asyncpg.create_pool(dsn, init=recycler.init, **get_pg_pool_kwargs())
    recycler.bind(pool)

a consumer composing its own ``init`` calls :func:`~threetears.core.collections.init_connection` and
then :meth:`YugabytePoolRecycler.watch`.
"""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import asyncpg

from threetears.core.collections.asyncpg_init import init_connection
from threetears.observe import get_logger

if TYPE_CHECKING:
    from asyncpg.connection import LoggedQuery

__all__ = [
    "DEFAULT_MIN_SECONDS_BETWEEN_EXPIRIES",
    "YUGABYTE_POOL_TRIGGERS",
    "YUGABYTE_RPC_TIMEOUT",
    "YUGABYTE_RPC_TIMEOUT_MESSAGE_PREFIX",
    "YUGABYTE_STALE_TABLE_SHAPE",
    "PoolExpiryTrigger",
    "YugabytePoolRecycler",
    "is_yugabyte_rpc_timeout",
    "is_yugabyte_stale_table_shape",
]

log = get_logger(__name__)

#: the start of every message YugabyteDB answers when a tserver RPC behind the connection times out.
#: the state that follows varies (``kRequestSent``, ``kProcessingRequest``).
YUGABYTE_RPC_TIMEOUT_MESSAGE_PREFIX = "Timed out waiting kResponseSent"

#: the least time between two expiries of one pool by one recycler. new connections answering the
#: error too means the cause is not the connections -- a tserver genuinely down or overloaded, a
#: table whose schema is itself inconsistent -- and a reconnect for every failed query would only
#: add load.
DEFAULT_MIN_SECONDS_BETWEEN_EXPIRIES = 10.0

#: YugabyteDB's answer when a connection's cached table shape names a column the table's stored
#: schema version does not have. matched anywhere in the message; see
#: :func:`is_yugabyte_stale_table_shape`.
_STALE_TABLE_SHAPE = re.compile(r"\bInvalid column number \d+")


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


def is_yugabyte_stale_table_shape(error: BaseException | None) -> bool:
    """say whether ``error`` is YugabyteDB answering that a connection holds a table's old shape.

    matches ``Invalid column number <n>`` in the message of any server error, whatever SQLSTATE it
    arrives with: the error has been seen once, on cobalt-dev, and not reproduced (0 of 80 attempts
    in one session, 0 of 40 rounds of eight connections in another, on YugabyteDB 2026.1), so its
    SQLSTATE is not known and the class cannot decide. the same words outside a server error are a
    client quoting them, not the server answering them, and do not match.

    deliberately does NOT match ``schema version mismatch for table ...`` (SQLSTATE ``40001``): that is
    YugabyteDB fencing a write that raced a schema change, it is retryable on the SAME connection,
    and an online index build produces it routinely -- a writer racing three ``ALTER TABLE``
    statements met it 366 times in 40 rounds. expiring the pool on it would churn every connection
    for no gain.

    :param error: what a query raised, or None when it succeeded
    :ptype error: BaseException | None
    :return: True for the stale-table-shape error
    :rtype: bool
    """
    return isinstance(error, asyncpg.PostgresError) and _STALE_TABLE_SHAPE.search(str(error)) is not None


@dataclass(frozen=True, slots=True)
class PoolExpiryTrigger:
    """what a :class:`YugabytePoolRecycler` keys on, and the words its log lines use for it.

    :ivar error_name: the error's name in a log line, e.g. ``"YugabyteDB RPC timeout"``
    :ivar matches: says whether a query's exception (None when it succeeded) is this error
    :ivar diagnosis: what the error says about the connection that answered it, completing
        "a query answered the <error_name>, so <diagnosis>"
    :ivar persistent_cause: what new connections answering it too would mean instead, completing
        "new connections failing too means <persistent_cause>, not the connections"
    """

    error_name: str
    matches: Callable[[BaseException | None], bool]
    diagnosis: str
    persistent_cause: str


#: a wedged tserver session behind the connection: :func:`is_yugabyte_rpc_timeout`.
YUGABYTE_RPC_TIMEOUT = PoolExpiryTrigger(
    error_name="YugabyteDB RPC timeout",
    matches=is_yugabyte_rpc_timeout,
    diagnosis="the tserver session behind its connection is wedged",
    persistent_cause="the cluster",
)

#: a connection holding a table's shape from before another session altered it:
#: :func:`is_yugabyte_stale_table_shape`.
YUGABYTE_STALE_TABLE_SHAPE = PoolExpiryTrigger(
    error_name="YugabyteDB stale-table-shape error",
    matches=is_yugabyte_stale_table_shape,
    diagnosis=(
        "a connection holds a table's shape from before another session altered it, "
        "and a new connection reloads the schema"
    ),
    persistent_cause="the table's schema",
)


#: the YugabyteDB errors a platform pool is watched for: every one a fresh connection escapes.
#: A pool built with the default recycler carries all of them, so a trigger added here reaches every
#: pool at once rather than every consumer's hand-written composition.
YUGABYTE_POOL_TRIGGERS: tuple[PoolExpiryTrigger, ...] = (YUGABYTE_RPC_TIMEOUT, YUGABYTE_STALE_TABLE_SHAPE)


class YugabytePoolRecycler:
    """expire one pool's connections when a query on any of them answers one of its triggers' errors.

    **One pool, one record of when it was last expired.** An expiry retires every connection opened
    before it, whatever error caused it, so a connection opened before the last expiry answering
    any trigger changes nothing and is skipped.

    **One floor per trigger.** The floor says "new connections answering this error too means the
    cause is not the connections", which is a claim about one cause: a stale table shape answered
    seconds after a wedge was cleared is a new cause, not the wedge persisting, and expires the pool
    again. So each trigger has its own last expiry, and a connection opened after the pool's last
    expiry answering a trigger inside that trigger's floor is logged and left, not expired again.
    Two triggers alternating can therefore expire a pool at most once per floor each.

    :param pool_name: the pool's name in logs, as given to ``log_pool_created``
    :ptype pool_name: str
    :param triggers: the errors this recycler keys on, checked in order; the first that matches a
        query's error decides. Defaults to :data:`YUGABYTE_POOL_TRIGGERS`
    :ptype triggers: Sequence[PoolExpiryTrigger]
    :param min_seconds_between_expiries: the least time between two expiries of the pool for one
        trigger
    :ptype min_seconds_between_expiries: float
    :param clock: monotonic clock in seconds; tests pass their own
    :ptype clock: Callable[[], float]
    :raises ValueError: if ``triggers`` is empty or ``min_seconds_between_expiries`` is not positive
    """

    def __init__(
        self,
        *,
        pool_name: str,
        triggers: Sequence[PoolExpiryTrigger] = YUGABYTE_POOL_TRIGGERS,
        min_seconds_between_expiries: float = DEFAULT_MIN_SECONDS_BETWEEN_EXPIRIES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """hold the pool's name, the triggers, the expiry floor and the clock.

        :param pool_name: the pool's name in logs
        :ptype pool_name: str
        :param triggers: the errors this recycler keys on, in order
        :ptype triggers: Sequence[PoolExpiryTrigger]
        :param min_seconds_between_expiries: the least time between two expiries for one trigger
        :ptype min_seconds_between_expiries: float
        :param clock: monotonic clock in seconds
        :ptype clock: Callable[[], float]
        :return: nothing
        :rtype: None
        :raises ValueError: if ``triggers`` is empty or ``min_seconds_between_expiries`` is not positive
        """
        if not triggers:
            raise ValueError("a pool recycler needs at least one trigger; it would otherwise watch for nothing")
        if min_seconds_between_expiries <= 0:
            raise ValueError(
                f"min_seconds_between_expiries must be positive, got {min_seconds_between_expiries!r}",
            )
        self._pool_name = pool_name
        self._triggers = tuple(triggers)
        self._min_seconds_between_expiries = min_seconds_between_expiries
        self._clock = clock
        self._pool: asyncpg.Pool | None = None
        # the pool's last expiry, whatever caused it: every connection opened before it is retired.
        self._expired_at: float | None = None
        # each trigger's last expiry, for its floor.
        self._expired_at_by_trigger: dict[PoolExpiryTrigger, float] = {}

    @property
    def triggers(self) -> tuple[PoolExpiryTrigger, ...]:
        """the errors this recycler keys on, in the order they are checked.

        :return: the triggers
        :rtype: tuple[PoolExpiryTrigger, ...]
        """
        return self._triggers

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
        """register the one query logger that reports this connection's trigger errors to the recycler.

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
            """expire the pool when this query answered one of the triggers' errors.

            :param record: what asyncpg ran, and what it raised
            :ptype record: LoggedQuery
            :return: nothing
            :rtype: None
            """
            trigger = next((t for t in self._triggers if t.matches(record.exception)), None)
            if trigger is not None:
                await self._on_match(trigger, opened_at=opened_at, backend_pid=backend_pid, error=record.exception)

        return _on_query

    async def _on_match(
        self, trigger: PoolExpiryTrigger, *, opened_at: float, backend_pid: int, error: BaseException | None
    ) -> None:
        """expire the pool, unless an earlier expiry already covers this connection or the floor holds it off.

        :param trigger: the trigger whose error the query answered
        :ptype trigger: PoolExpiryTrigger
        :param opened_at: clock reading when the failing connection was opened
        :ptype opened_at: float
        :param backend_pid: the failing connection's server process id
        :ptype backend_pid: int
        :param error: the error the query raised
        :ptype error: BaseException | None
        :return: nothing
        :rtype: None
        """
        now = self._clock()
        error_name = trigger.error_name
        trigger_expired_at = self._expired_at_by_trigger.get(trigger)
        if self._pool is None:
            log.error(
                "a query on pool %s answered the %s, but its recycler was never bound to the pool, so no "
                "connection is replaced; call bind(pool) right after creating it: backend_pid=%s error=%s",
                self._pool_name,
                error_name,
                backend_pid,
                error,
            )
        elif self._expired_at is not None and opened_at <= self._expired_at:
            log.debug(
                "pool %s: connection from before the last expiry answered the %s; already being replaced: "
                "backend_pid=%s",
                self._pool_name,
                error_name,
                backend_pid,
            )
        elif trigger_expired_at is not None and now - trigger_expired_at < self._min_seconds_between_expiries:
            log.warning(
                "pool %s: a connection opened after the last expiry answered the %s %.1fs after the last "
                "expiry for it; not expiring again within %.0fs, since new connections failing too means %s, "
                "not the connections: backend_pid=%s error=%s",
                self._pool_name,
                error_name,
                now - trigger_expired_at,
                self._min_seconds_between_expiries,
                trigger.persistent_cause,
                backend_pid,
                error,
            )
        else:
            log.warning(
                "pool %s: a query answered the %s, so %s; expiring every connection in the pool so each is "
                "replaced at its next release or acquire: backend_pid=%s connection_age_seconds=%.0f error=%s",
                self._pool_name,
                error_name,
                trigger.diagnosis,
                backend_pid,
                now - opened_at,
                error,
            )
            self._expired_at = now
            self._expired_at_by_trigger[trigger] = now
            await self._pool.expire_connections()
