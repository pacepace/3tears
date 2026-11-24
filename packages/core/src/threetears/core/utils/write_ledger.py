"""a process's writer ledger: when its L3 writes began, in memory, so a lost epoch advance is never lost for good.

A write to a table whose collection follows write generations commits to L3, and its generation is advanced after,
outside the transaction (``BaseCollection``, the hub's broker). An advance that fails, is cancelled, or never runs
because the writer died between its commit and its advance leaves the rows committed and the generation unmoved:
every follower keeps what it holds. Nothing in the database marks the write (the per-row write-mark trigger did, at
three times an upsert's cost on YugabyteDB, plus a row inserted and deleted for every commit; Pace 2026-11-22: no L3
write per commit for this backstop). The writer itself knows instead:

- **The ledger** (:class:`WriteLedger`) holds, in this process's memory, the wall-clock start of every transaction
  open on a tracked connection, and of every statement run outside one, and keeps each start for
  :data:`WRITE_GRACE` after it ends. Its **watermark** is the oldest start it holds, or a lost advance's
  (:meth:`WriteLedger.lost_advance`) until it is repaired; ``now`` when it holds none. Every change note a write of
  this process left that may still lack its advance is stamped after the watermark (less the clocks' skew,
  :data:`CLOCK_SKEW`). Zero database round trips: a dict entry a statement.
- **The connection class** (:func:`write_ledger_connection_class`) keeps the ledger for a direct asyncpg pool.
- **A failed advance** is retried by the process itself: :func:`~threetears.core.collections.epoch_change
  .advance_recording` hands it to :meth:`WriteLedger.lost_advance`, and the process's backstop
  (:class:`~threetears.core.collections.lost_writes.WriteBackstop`) retries it until it lands, the watermark held at
  its start meanwhile.
- **A dead writer** is caught from its heartbeat: the backstop publishes the watermark to NATS every few seconds, and
  the replica holding the recovery lease announces a stale writer's whole window from the change notes.

The ledger is per process (:func:`process_write_ledger`): one watermark covers every pool the process writes through.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import count
from types import MappingProxyType
from typing import Any, Final

import asyncpg

from threetears.observe import contain, get_logger
from threetears.observe.metrics import counter, gauge

__all__ = [
    "CLOCK_SKEW",
    "WRITE_GRACE",
    "WriteLedger",
    "process_write_ledger",
    "write_ledger_connection_class",
]

log = get_logger(__name__)

#: how long a write's start is kept after its commit: the bound on commit-to-advance. An advance is a NATS KV write
#: and its change record, each bounded by its client's timeout (seconds), run right after the commit; one that fails
#: or is cancelled is handed to the repair before it returns, which holds the watermark from then on. A writer whose
#: advance has not even failed 60 s after its commit is hung, and a beat it still sends carries a watermark no later
#: than this before it. Wider costs nothing per write (one entry a second of commits); it only widens a dead writer's
#: window, which recovery reads by the notes' (table_name, noted_at) index
WRITE_GRACE: Final = timedelta(seconds=60)

#: how far a process's wall clock and a database node's may disagree: a note is stamped on the node's clock
#: (``clock_timestamp()``), a watermark on the process's. Pods run NTP (milliseconds); YugabyteDB bounds its nodes' skew
#: at max_clock_skew_usec (500 ms by default). Every window this backstop announces is widened by it at both ends: a
#: wider window only names keys other writes changed too, which a follower reads again harmlessly
CLOCK_SKEW: Final = timedelta(seconds=5)

_REPAIRS = counter(
    "collection_lost_advance_repairs_total",
    description="lost epoch advances this process retried, by outcome: queued (a failed advance handed over), "
    "landed, failed (retried again later)",
    label_names=("outcome",),
)
_PENDING = gauge(
    "collection_lost_advance_repairs_pending",
    description="lost epoch advances this process holds for retry; each holds its watermark",
)


@dataclass
class _LostAdvance:
    """one advance that did not land after its write committed, held until a retry lands.

    :ivar table: the table
    :ivar retry: makes the advance again (the same change, through the same source); raises when it fails
    :ivar held_from: the process's watermark when it failed, held until it lands
    :ivar attempts: retries made so far
    :ivar error: the last failure, as text
    :ivar key: its place in the ledger
    """

    table: str
    retry: Callable[[], Awaitable[Any]]
    held_from: datetime
    attempts: int = 0
    error: str = ""
    key: int = 0


class WriteLedger:
    """the starts of a process's open and recently ended L3 writes, and the advances it still owes.

    :param clock: wall-clock seconds since the epoch
    :ptype clock: Callable[[], float]
    :param grace: how long an ended write's start is kept (:data:`WRITE_GRACE`)
    :ptype grace: timedelta
    """

    def __init__(self, *, clock: Callable[[], float] = time.time, grace: timedelta = WRITE_GRACE) -> None:
        if grace <= timedelta(0):
            raise ValueError("a write's start is kept for some time after it ends")
        self._clock = clock
        self._grace = grace.total_seconds()
        self._tokens = count(1)
        #: each open write's start, by its token (as many as the process's open connections); replaced whole
        self._open: Mapping[int, float] = MappingProxyType({})
        #: the oldest start of the writes that ended in each whole second (the second they ended), kept for the grace:
        #: an entry a second rather than a list entry a write, so a busy writer's ledger stays a few dozen entries;
        #: replaced whole only when a second's oldest start changes, not per write
        self._ended: Mapping[int, float] = MappingProxyType({})
        #: the advances held for retry, by key; replaced whole
        self._lost: Mapping[int, _LostAdvance] = MappingProxyType({})
        self._lost_keys = count(1)
        #: whether a backstop retries this ledger's lost advances (:meth:`attach_repairer`)
        self.repairing = False

    def begin(self) -> int:
        """a write starts now.

        :return: its token, for :meth:`end`
        :rtype: int
        """
        token = next(self._tokens)
        self._open = MappingProxyType({**self._open, token: self._clock()})
        return token

    def end(self, token: int) -> None:
        """the write ``token`` names ended (committed, rolled back, or its connection closed); its start is kept for the
        grace.

        :param token: from :meth:`begin`
        :ptype token: int
        :return: nothing
        :rtype: None
        """
        start = self._open.get(token)
        if start is None:
            return
        self._open = MappingProxyType({held: begun for held, begun in self._open.items() if held != token})
        second = math.floor(self._clock())
        held = self._ended.get(second)
        if held is None or start < held:
            self._ended = MappingProxyType({**self._ended, second: start})

    @property
    def open_writes(self) -> int:
        """how many writes are open.

        :return: the count
        :rtype: int
        """
        return len(self._open)

    def watermark(self) -> datetime:
        """the oldest start among the open writes, the ended ones within their grace and the lost advances.

        :return: that start; now when the ledger holds none
        :rtype: datetime
        """
        now = self._clock()
        # a second's entry is kept while any write that ended in it may still be in its grace
        if any(second + 1 + self._grace <= now for second in self._ended):
            self._ended = MappingProxyType(
                {second: start for second, start in self._ended.items() if second + 1 + self._grace > now}
            )
        starts = [*self._open.values(), *self._ended.values()]
        oldest = min(starts, default=now)
        held = min((lost.held_from for lost in self._lost.values()), default=None)
        at = datetime.fromtimestamp(min(oldest, now), tz=UTC)
        return at if held is None else min(at, held)

    def lost_advance(self, table: str, retry: Callable[[], Awaitable[Any]], *, error: str = "") -> _LostAdvance:
        """hold an advance that did not land, from the watermark now, until :meth:`repair_pending` lands it.

        :param table: the table
        :ptype table: str
        :param retry: makes the advance again
        :ptype retry: Callable[[], Awaitable[Any]]
        :param error: why it failed
        :ptype error: str
        :return: what is held
        :rtype: _LostAdvance
        """
        lost = _LostAdvance(
            table=table, retry=retry, held_from=self.watermark(), error=error, key=next(self._lost_keys)
        )
        self._lost = MappingProxyType({**self._lost, lost.key: lost})
        _REPAIRS.labels(outcome="queued").inc()
        _PENDING.set(len(self._lost))
        if not self.repairing:
            log.error(
                "an epoch advance was lost after its write committed and this process runs no backstop to retry it: "
                "followers of the table keep what they hold until the table moves again or a recovery announces it",
                extra={"extra_data": {"table": table, "error": error}},
            )
        return lost

    @property
    def lost(self) -> tuple[_LostAdvance, ...]:
        """the advances held for retry, oldest first.

        :return: them
        :rtype: tuple[_LostAdvance, ...]
        """
        return tuple(self._lost.values())

    def attach_repairer(self) -> None:
        """a backstop retries this ledger's lost advances from now on.

        :return: nothing
        :rtype: None
        """
        self.repairing = True

    def detach_repairer(self) -> None:
        """no backstop retries this ledger's lost advances any more.

        :return: nothing
        :rtype: None
        """
        self.repairing = False

    async def repair_pending(self, *, deadline: float | None = None) -> int:
        """retry every held advance once, oldest first; release each that lands.

        Each retry is bounded by :data:`_RETRY_SECONDS` (and by ``deadline``): one that hangs (an epoch bucket that never
        answers) stays held like one that fails, so it neither stalls the others' repairs nor holds a stop past its
        drain (``WriteBackstop.stop``).

        :param deadline: a ``time.monotonic()`` past which no retry starts or runs on; ``None`` for none
        :ptype deadline: float | None
        :return: how many are still held
        :rtype: int
        """
        for lost in self._lost.values():
            budget = _RETRY_SECONDS if deadline is None else min(_RETRY_SECONDS, deadline - time.monotonic())
            if budget <= 0:
                break
            lost.attempts += 1
            try:
                async with asyncio.timeout(budget):
                    await lost.retry()
            # prawduct:allow prawduct/broad-except -- a retry that fails keeps its advance held (and the watermark with it) for the next pass
            except Exception as exc:
                lost.error = f"{type(exc).__name__}: {exc}"
                _REPAIRS.labels(outcome="failed").inc()
                contain(
                    exc,
                    log=log,
                    what="a lost epoch advance could not be made again yet; it stays held and is retried",
                    extra={"table": lost.table, "attempts": lost.attempts, "held_from": lost.held_from.isoformat()},
                    level="warning",
                )
                continue
            self._lost = MappingProxyType({key: held for key, held in self._lost.items() if key != lost.key})
            _REPAIRS.labels(outcome="landed").inc()
            log.info(
                "a lost epoch advance was made again",
                extra={"extra_data": {"table": lost.table, "attempts": lost.attempts}},
            )
        _PENDING.set(len(self._lost))
        return len(self._lost)


#: the longest one retry of a lost advance may take; past it the advance stays held for the next pass
_RETRY_SECONDS: Final = 10.0


_PROCESS_LEDGER: Final = WriteLedger()


def process_write_ledger() -> WriteLedger:
    """the ledger of this process: every tracked connection and every lost advance share it.

    :return: the ledger
    :rtype: WriteLedger
    """
    return _PROCESS_LEDGER


def write_ledger_connection_class(
    base: type[asyncpg.Connection] = asyncpg.Connection, *, ledger: WriteLedger | None = None
) -> type[asyncpg.Connection]:
    """a connection class that keeps the ledger of its writes: each transaction's start, and each statement's outside
    one, ended when the transaction (or the statement) ends.

    Wraps any connection class (``session_settings_connection_class(...)``, ``change_note_slot_connection_class(...)``)
    so the three compose: pass the result as ``connection_class`` to ``asyncpg.create_pool``. A transaction is seen by
    the statement that opens it (``BEGIN``, through ``execute``) and ended by the first statement after which the
    connection is outside one (``COMMIT``, ``ROLLBACK``, a pool's reset), or by the connection closing. Every public
    statement method is tracked: ``execute``, ``executemany``, the ``fetch`` family and the ``copy`` family. A
    prepared statement run outside a transaction is not (none of 3tears' writers do; one inside a transaction is
    covered by the transaction's start).

    :param base: the connection class to extend
    :ptype base: type[asyncpg.Connection]
    :param ledger: the ledger; ``None`` for the process's (:func:`process_write_ledger`)
    :ptype ledger: WriteLedger | None
    :return: the connection class
    :rtype: type[asyncpg.Connection]
    """
    kept = ledger if ledger is not None else _PROCESS_LEDGER

    class _WriteLedgerConnection(base):  # type: ignore[misc]
        """an asyncpg connection that keeps the start of each of its writes in the process's ledger."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._ledger_transaction: int | None = None
            self.add_termination_listener(self._ledger_on_terminated)

        def _ledger_on_terminated(self, connection: Any) -> None:
            """the connection closed or was terminated mid-transaction: the transaction ended with it.

            :param connection: this connection, as asyncpg hands it
            :ptype connection: Any
            :return: nothing
            :rtype: None
            """
            if self._ledger_transaction is not None:
                kept.end(self._ledger_transaction)
                self._ledger_transaction = None

        def _ledger_in_transaction(self) -> bool:
            try:
                return bool(self.is_in_transaction())
            # NOSILENT: a closed connection's protocol is gone; it is in no transaction
            except AttributeError:
                return False

        def _ledger_settle(self) -> None:
            """end the transaction the ledger holds once the connection is outside it.

            :return: nothing
            :rtype: None
            """
            if self._ledger_transaction is not None and not self._ledger_in_transaction():
                kept.end(self._ledger_transaction)
                self._ledger_transaction = None

        async def _tracked(self, call: Callable[[], Awaitable[Any]]) -> Any:
            self._ledger_settle()
            token: int | None = None
            if self._ledger_transaction is None:
                # a statement outside a transaction, or the BEGIN of one; or a transaction opened past these methods,
                # held from now (the best start known)
                token = kept.begin()
                if self._ledger_in_transaction():
                    self._ledger_transaction, token = token, None
            try:
                return await call()
            finally:
                if token is not None:
                    if self._ledger_in_transaction():
                        self._ledger_transaction = token
                    else:
                        kept.end(token)
                else:
                    self._ledger_settle()

        async def execute(self, query: str, *args: Any, timeout: float | None = None) -> str:
            status: str = await self._tracked(
                lambda: super(_WriteLedgerConnection, self).execute(query, *args, timeout=timeout)
            )
            return status

        async def executemany(self, command: str, args: Any, *, timeout: float | None = None) -> None:
            await self._tracked(lambda: super(_WriteLedgerConnection, self).executemany(command, args, timeout=timeout))

        async def fetch(self, query: str, *args: Any, **kwargs: Any) -> list[Any]:
            rows: list[Any] = await self._tracked(
                lambda: super(_WriteLedgerConnection, self).fetch(query, *args, **kwargs)
            )
            return rows

        async def fetchrow(self, query: str, *args: Any, **kwargs: Any) -> Any:
            return await self._tracked(lambda: super(_WriteLedgerConnection, self).fetchrow(query, *args, **kwargs))

        async def fetchval(self, query: str, *args: Any, **kwargs: Any) -> Any:
            return await self._tracked(lambda: super(_WriteLedgerConnection, self).fetchval(query, *args, **kwargs))

        async def fetchmany(self, query: str, args: Any, **kwargs: Any) -> list[Any]:
            rows: list[Any] = await self._tracked(
                lambda: super(_WriteLedgerConnection, self).fetchmany(query, args, **kwargs)
            )
            return rows

        async def copy_to_table(self, table_name: str, **kwargs: Any) -> str:
            status: str = await self._tracked(
                lambda: super(_WriteLedgerConnection, self).copy_to_table(table_name, **kwargs)
            )
            return status

        async def copy_records_to_table(self, table_name: str, **kwargs: Any) -> str:
            status: str = await self._tracked(
                lambda: super(_WriteLedgerConnection, self).copy_records_to_table(table_name, **kwargs)
            )
            return status

        async def copy_from_query(self, query: str, *args: Any, **kwargs: Any) -> str:
            status: str = await self._tracked(
                lambda: super(_WriteLedgerConnection, self).copy_from_query(query, *args, **kwargs)
            )
            return status

        async def copy_from_table(self, table_name: str, **kwargs: Any) -> str:
            status: str = await self._tracked(
                lambda: super(_WriteLedgerConnection, self).copy_from_table(table_name, **kwargs)
            )
            return status

    return _WriteLedgerConnection
