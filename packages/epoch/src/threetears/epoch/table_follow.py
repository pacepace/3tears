"""follow tables' write generations, one supervised key watch per table, for as long as a process runs.

A registry that follows a table keeps a mark of the generations it has heard every row of, and
drops the table -- its rows, its scans, every cache derived from it -- when the epoch bucket shows
an advance it did not hear (epoch-task-06). :func:`~threetears.epoch.generation_tick.follow_generation_key`
is one such watch; it ends when its connection closes and fails when the bucket cannot be reached,
and a table left followed with no watch has nothing to judge its mark.

:class:`TableFollower` runs one watch per table as a task, and starts a watch again when one ends
or fails, after a delay that doubles with each consecutive failure up to a cap and resets when a
watch is pushed a value. A restarted watch is first pushed the key's latest value, so an advance
missed while it was down is judged then. The delay is about the watch's liveness, not an age on any
cached value: nothing cached here expires. Each table's :class:`WatchHealth` is readable, so a
caller can report a follower whose watches keep failing, and the follower tells the registry
whether each table's watch is running (``CollectionRegistry.watched_by``), which is what a cache
derived from the table reads before it serves what it holds.

**One call for a consumer: :func:`follow_tables`.** It follows any tables on a registry and returns
the follower, started. A cache that must also be bound to the tables' row broadcasts has its own
call that does both (``threetears.agent.acl.generation_follow`` for the access tables).
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Final

from threetears.core.collections.registry import CollectionRegistry
from threetears.observe import contain, get_logger

from threetears.epoch.generation_tick import DEFAULT_BROADCAST_GRACE, GenerationWatcher, follow_generation_key

__all__ = [
    "DEFAULT_STOP_TIMEOUT",
    "DEFAULT_WATCH_RESTART_DELAY",
    "MAX_WATCH_RESTART_DELAY",
    "TableFollower",
    "WatchHealth",
    "follow_tables",
]

log = get_logger(__name__)

#: how long a watch that ended or failed for the first time in a row waits before it starts again
DEFAULT_WATCH_RESTART_DELAY: Final = timedelta(seconds=1)

#: the longest a failing watch waits between attempts, however many times in a row it has failed
MAX_WATCH_RESTART_DELAY: Final = timedelta(seconds=60)

#: the longest :meth:`TableFollower.stop` waits for its cancelled watches to end; a shutdown
#: never waits on a watch that will not stop
DEFAULT_STOP_TIMEOUT: Final = timedelta(seconds=5)


def _reader_closed(reader: object) -> bool:
    """whether the watcher says its connection is closed, so no watch can ever deliver again.

    :param reader: the watcher; one without a ``closed`` attribute is never closed
    :ptype reader: object
    :return: ``True`` once the watcher's connection is closed
    :rtype: bool
    """
    return getattr(reader, "closed", False) is True


@dataclass
class WatchHealth:
    """one table's watch, as its follower last saw it.

    :ivar consecutive_failures: watches that ended or failed in a row since the last push; ``0``
        while the watch is pushed values
    :ivar pushes: values the watch has been pushed since the follower started
    :ivar last_error: the last failure's description, or ``None``
    """

    consecutive_failures: int = 0
    pushes: int = 0
    last_error: str | None = None

    @property
    def healthy(self) -> bool:
        """whether the watch has been pushed a value since it last failed.

        :return: ``True`` when it has, ``False`` while it is failing or has not been pushed yet
        :rtype: bool
        """
        return self.consecutive_failures == 0 and self.pushes > 0


class _ObservedWatcher:
    """a generation watcher that records each push on the table's health."""

    def __init__(self, inner: GenerationWatcher, health: dict[str, WatchHealth]) -> None:
        self._inner = inner
        self._health = health

    async def watch(self, table_name: str) -> AsyncGenerator[str | None]:
        """pass the inner watch's values on, recording each one.

        :param table_name: the table
        :ptype table_name: str
        :return: the generations
        :rtype: AsyncGenerator[str | None]
        """
        health = self._health[table_name]
        generations = self._inner.watch(table_name)
        try:
            async for token in generations:
                health.pushes += 1
                health.consecutive_failures = 0
                yield token
        finally:
            await generations.aclose()


class TableFollower:
    """one supervised generation-key watch per table, on one registry.

    :param registry: the registry that follows the tables; its invalidation listener must be
        running, since it is what hears the rows a watch judges against
    :ptype registry: CollectionRegistry
    :param reader: reads and watches the epoch bucket
    :ptype reader: GenerationWatcher
    :param tables: the tables to follow
    :ptype tables: Sequence[str]
    :param grace: how long a watch waits for an advance's rows before judging them missed
    :ptype grace: timedelta
    :param restart_delay: how long a watch waits after its first failure in a row; doubled after
        each further one, up to ``max_restart_delay``
    :ptype restart_delay: timedelta
    :param max_restart_delay: the longest wait between attempts
    :ptype max_restart_delay: timedelta
    :param stop_timeout: how long :meth:`stop` waits for the cancelled watches to end
    :ptype stop_timeout: timedelta

    A watch whose connection is closed (the watcher's ``closed``) stops instead of starting again:
    the client is shutting down, and a retry loop against it would keep the process alive.
    """

    def __init__(
        self,
        registry: CollectionRegistry,
        reader: GenerationWatcher,
        tables: Sequence[str],
        *,
        grace: timedelta = DEFAULT_BROADCAST_GRACE,
        restart_delay: timedelta = DEFAULT_WATCH_RESTART_DELAY,
        max_restart_delay: timedelta = MAX_WATCH_RESTART_DELAY,
        stop_timeout: timedelta = DEFAULT_STOP_TIMEOUT,
    ) -> None:
        """capture what to follow; no I/O.

        :return: nothing
        :rtype: None
        """
        self._registry = registry
        self._tables = tuple(tables)
        self._grace = grace
        self._restart_delay = restart_delay
        self._max_restart_delay = max_restart_delay
        self._stop_timeout = stop_timeout
        self._health = {table: WatchHealth() for table in self._tables}
        self._source = reader
        self._reader = _ObservedWatcher(reader, self._health)
        self._tasks: list[asyncio.Task[None]] = []
        # table -> what this follower registered with the registry, withdrawn by stop()
        self._watching_by_table: dict[str, Callable[[], bool]] = {}

    @property
    def running(self) -> bool:
        """whether the watches are running.

        :return: ``True`` between :meth:`start` and :meth:`stop`
        :rtype: bool
        """
        return bool(self._tasks)

    @property
    def health(self) -> dict[str, WatchHealth]:
        """each table's watch, as last seen.

        :return: table to its health; a copy
        :rtype: dict[str, WatchHealth]
        """
        return {table: WatchHealth(h.consecutive_failures, h.pushes, h.last_error) for table, h in self._health.items()}

    @property
    def watching(self) -> bool:
        """whether the watches are running and none is failing now.

        Weaker than :attr:`healthy`: a watch on a key never written is pushed nothing, and is still
        watching. What a cache that may serve held answers needs: a watch that is running judges every
        advance it is pushed, its first push included, which is the key's latest.

        :return: ``True`` when every table's watch is running with no failure since its last push
        :rtype: bool
        """
        return self.running and all(h.consecutive_failures == 0 for h in self._health.values())

    def _table_watching(self, table: str) -> bool:
        """whether ``table``'s watch is running with no failure since its last push.

        :param table: the table
        :ptype table: str
        :return: ``True`` while the watch can judge the table
        :rtype: bool
        """
        return self.running and self._health[table].consecutive_failures == 0

    @property
    def healthy(self) -> bool:
        """whether every table's watch is running and has been pushed a value since it last failed.

        :return: ``True`` when the follower can judge every table it follows
        :rtype: bool
        """
        return self.running and all(h.healthy for h in self._health.values())

    def start(self) -> None:
        """follow every table on the registry now and start its watch; a no-op while running.

        Each table is followed before its watch's first push, so a row heard in between counts.

        :return: nothing
        :rtype: None
        :raises RuntimeError: when the registry's invalidation listener is not running
        """
        if self._tasks:
            return
        if not self._registry.invalidation_listener_running:
            raise RuntimeError(
                "a table follower needs the registry's invalidation listener running first: it "
                "hears the rows each watch judges against, and an advance judged without them reads as missed"
            )
        # every table followed first, then one watch task each: start() does not yield, so no watch is
        # pushed a value before its table is followed. The tasks are a fixed set, one per table, held to
        # be stopped -- nothing is accumulated or flushed
        # what a cache derived from each table reads to know whether it may serve what it holds
        self._watching_by_table = {table: functools.partial(self._table_watching, table) for table in self._tables}
        for table in self._tables:
            self._registry.follow_generation(table)
            self._registry.watched_by(table, self._watching_by_table[table])
        self._tasks = [
            asyncio.create_task(self._watch(table), name=f"follow-generation:{table}") for table in self._tables
        ]
        log.info("following tables' write generations", extra={"extra_data": {"tables": self._tables}})

    async def stop(self) -> None:
        """cancel every watch and wait, at most ``stop_timeout``, for them to end. Idempotent.

        A watch that does not end in time is left cancelled and logged: a shutdown never waits on
        it. A cancellation of ``stop`` itself propagates.

        :return: nothing
        :rtype: None
        """
        tasks, self._tasks = self._tasks, []
        watching_by_table, self._watching_by_table = self._watching_by_table, {}
        for table, watching in watching_by_table.items():
            self._registry.not_watched_by(table, watching)
        for task in tasks:
            task.cancel()
        if tasks:
            _done, pending = await asyncio.wait(tasks, timeout=self._stop_timeout.total_seconds())
            if pending:
                log.warning(
                    "write-generation watches did not end within the stop bound; left cancelled",
                    extra={
                        "extra_data": {
                            "tables": sorted(task.get_name() for task in pending),
                            "stop_timeout_seconds": self._stop_timeout.total_seconds(),
                        }
                    },
                )

    def _delay_after(self, failures: int) -> float:
        """the wait before the next attempt, after ``failures`` failures in a row.

        :param failures: consecutive failures, at least one
        :ptype failures: int
        :return: seconds
        :rtype: float
        """
        doubled: float = self._restart_delay.total_seconds() * float(2 ** min(failures - 1, 30))
        return min(doubled, self._max_restart_delay.total_seconds())

    async def _watch(self, table: str) -> None:
        """watch one table's generation key until cancelled, starting the watch again when it ends.

        :param table: the table
        :ptype table: str
        :return: nothing; it ends only when cancelled
        :rtype: None
        """
        health = self._health[table]
        while True:
            try:
                await follow_generation_key(self._registry, self._reader, table, grace=self._grace)
                health.last_error = "the watch ended"
            except asyncio.CancelledError:
                raise
            # prawduct:allow prawduct/broad-except -- a watch that fails for any reason is started again;
            # stopping would leave the table followed with nothing to judge its mark
            except Exception as exc:  # noqa: BLE001
                contain(exc, log=log, what="a write-generation watch failed", extra={"table": table})
                health.last_error = f"{type(exc).__name__}: {exc}"
            health.consecutive_failures += 1
            if _reader_closed(self._source):
                log.info(
                    "a write-generation watch stopped: its NATS connection is closed",
                    extra={"extra_data": {"table": table, "error": health.last_error}},
                )
                return
            delay = self._delay_after(health.consecutive_failures)
            log.warning(
                "a write-generation watch ended or failed; starting it again",
                extra={
                    "extra_data": {
                        "table": table,
                        "consecutive_failures": health.consecutive_failures,
                        "error": health.last_error,
                        "retry_in_seconds": delay,
                    }
                },
            )
            await asyncio.sleep(delay)


def follow_tables(
    registry: CollectionRegistry,
    reader: GenerationWatcher,
    tables: Sequence[str],
    *,
    grace: timedelta = DEFAULT_BROADCAST_GRACE,
    restart_delay: timedelta = DEFAULT_WATCH_RESTART_DELAY,
    max_restart_delay: timedelta = MAX_WATCH_RESTART_DELAY,
) -> TableFollower:
    """follow ``tables`` on ``registry`` by watching their generation keys, and return the follower, started.

    For tables whose derived caches live on the registry itself -- the visibility-scan cache
    (:meth:`~threetears.core.collections.registry.CollectionRegistry.scan_cache`), which serves an
    entry only while every table it depends on is followed this way. Stop the follower before the
    registry's invalidation listener.

    :param registry: the registry whose listener hears the rows, and which follows the tables
    :ptype registry: CollectionRegistry
    :param reader: reads and watches the epoch bucket
    :ptype reader: GenerationWatcher
    :param tables: the tables to follow
    :ptype tables: Sequence[str]
    :param grace: how long a watch waits for an advance's rows before judging them missed
    :ptype grace: timedelta
    :param restart_delay: a failing watch's first wait before it starts again
    :ptype restart_delay: timedelta
    :param max_restart_delay: the longest wait between a failing watch's attempts
    :ptype max_restart_delay: timedelta
    :return: the running follower
    :rtype: TableFollower
    :raises RuntimeError: when the registry's invalidation listener is not running
    """
    follower = TableFollower(
        registry,
        reader,
        tuple(tables),
        grace=grace,
        restart_delay=restart_delay,
        max_restart_delay=max_restart_delay,
    )
    follower.start()
    return follower
