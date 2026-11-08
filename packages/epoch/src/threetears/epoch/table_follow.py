"""follow tables' write generations, one supervised key watch per table, for as long as a process runs.

A registry that follows a table keeps a mark of the generations it has heard every row of, and
applies the change record of each advance the epoch bucket shows it did not hear -- evicting exactly
those rows from its L1, its scans and every cache derived from the table (epoch-task-06). :func:`~threetears.epoch.generation_tick.follow_generation_key`
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
from typing import Any, Final

from threetears.core.collections.epoch_change import (
    EpochChange,
    EpochChangeReader,
    EpochLogExtent,
    EpochQuiet,
    EpochWalk,
    walk_by_count,
)
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.collections.registry import CollectionRegistry
from threetears.observe import contain, get_logger

from threetears.epoch.generation_tick import (
    DEFAULT_BROADCAST_GRACE,
    GenerationWatcher,
    QuietReader,
    follow_generation_key,
    quiet_marks_tick,
    quiet_pass_interval,
)

__all__ = [
    "DEFAULT_STOP_TIMEOUT",
    "DEFAULT_WATCH_RESTART_DELAY",
    "MAX_WATCH_RESTART_DELAY",
    "TableFollower",
    "WatchHealth",
    "follow_l2_kept_tables",
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

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        """the change record of one epoch, read through the watcher this wraps when it reads them.

        :param table_name: the table
        :ptype table_name: str
        :param token: the generation that epoch's advance wrote
        :ptype token: str
        :return: the record; ``None`` when the wrapped watcher reads none, so the follower is behind
        :rtype: EpochChange | None
        """
        if isinstance(self._inner, EpochChangeReader):
            return await self._inner.change(table_name, token)
        return None

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        """the epochs of a table since a mark, read through the watcher this wraps when it reads them.

        :param table_name: the table
        :ptype table_name: str
        :param after: the mark
        :ptype after: str
        :param upto: the generation read
        :ptype upto: str
        :return: the walk; an incomplete empty one when the wrapped watcher reads none
        :rtype: EpochWalk
        """
        if isinstance(self._inner, EpochChangeReader):
            return await self._inner.changes_after(table_name, after, upto)
        return EpochWalk(changes=(), complete=False, first_missing=upto)

    async def continues(self, incarnation: str) -> tuple[str, int | None] | None:
        """the lost log a restored one continues, read through the watcher this wraps when it can say.

        :param incarnation: the log's incarnation
        :ptype incarnation: str
        :return: the lost incarnation and the sequence its notes cover it from; ``None`` when none or unknown
        :rtype: tuple[str, int | None] | None
        """
        if isinstance(self._inner, EpochLogExtent):
            return await self._inner.continues(incarnation)
        return None

    async def quiet_after(self, table_name: str, mark: str) -> EpochQuiet:
        """what the wrapped watcher's log vouches for after a mark; nothing when it cannot say.

        :param table_name: the table
        :ptype table_name: str
        :param mark: the follower's mark
        :ptype mark: str
        :return: the log's answer
        :rtype: EpochQuiet
        """
        if isinstance(self._inner, EpochLogExtent):
            return await self._inner.quiet_after(table_name, mark)
        return EpochQuiet(through=None)


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

    **A quiet table never ages out.** A watch is pushed only when its table is written, so a table
    nobody writes would keep its mark while the change log's retention moves past it, and then be judged
    behind (``aged_out``) for a stretch in which nothing happened. Where the reader can read the log's
    head (:class:`~threetears.epoch.generation_tick.QuietReader`, as
    :class:`~threetears.epoch.generation.EpochGenerationReader` does), the follower also runs one quiet
    pass for all its tables (:func:`~threetears.epoch.generation_tick.quiet_marks_tick`) every
    :func:`~threetears.epoch.generation_tick.quiet_pass_interval`, a fraction of the log's own retention:
    it reads only NATS, moves each quiet table's mark to the log's head, and catches up any table the log
    holds an unapplied epoch of through the ordinary walk.
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
        advance it is pushed, its first push included, which is the key's latest. A table the registry
        is behind on (an epoch whose record could not be read) stays served: behind is said through
        the registry's log, counter and gauge, never by withdrawing a cache.

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
        # one quiet pass for every table, where the reader can read the log's head
        quiet = (
            [
                asyncio.create_task(
                    _run_quiet_passes(self._registry, self._source, self._tables), name="generation-quiet-pass"
                )
            ]
            if isinstance(self._source, QuietReader)
            else []
        )
        self._tasks = [
            *(asyncio.create_task(self._watch(table), name=f"follow-generation:{table}") for table in self._tables),
            *quiet,
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


async def _run_quiet_passes(registry: CollectionRegistry, reader: QuietReader, tables: Sequence[str]) -> None:
    """run the quiet pass over a follower's tables every pass interval, until cancelled.

    The first interval is read from the log itself, so it follows the log's declared retention; a pass
    that fails is logged and the next one runs at the same interval. Nothing is buffered: each pass reads
    the log and judges every table there and then.

    :param registry: the follower's registry
    :ptype registry: CollectionRegistry
    :param reader: reads the change log
    :ptype reader: QuietReader
    :param tables: the follower's tables
    :ptype tables: Sequence[str]
    :return: nothing; it ends only when cancelled
    :rtype: None
    """
    try:
        state, _ = await reader.newest_epochs(())
        interval = quiet_pass_interval(state)
    except GenerationUnavailableError as exc:
        # NOSILENT: the log could not be read; the retention a source declares by default sets the first interval
        log.warning(
            "the quiet pass could not read the change log's retention", extra={"extra_data": {"error": str(exc)}}
        )
        interval = quiet_pass_interval(None)
    while True:
        await asyncio.sleep(interval.total_seconds())
        try:
            interval = (await quiet_marks_tick(registry, reader, tables)).interval
        except asyncio.CancelledError:
            raise
        # prawduct:allow prawduct/broad-except -- a failed pass is retried at the next interval; stopping would leave quiet tables to age out
        except Exception as exc:  # noqa: BLE001
            contain(exc, log=log, what="a quiet pass failed; the next one runs at the same interval")


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


class _L2KeptGenerations:
    """watches, and reads the epochs of, the tables a registry keeps in L2 alone, through their collections.

    A table kept in L2 (``L2_WRITE_GENERATION``) keeps its generation, and each epoch's change record,
    beside its rows in the owner's scope. This hands a :class:`TableFollower` that table's generation key
    watch and its records, so a replica catches the table up on each advance pushed to it -- never on a
    timed pass.

    :param registry: the registry holding the collections
    :ptype registry: CollectionRegistry
    """

    def __init__(self, registry: CollectionRegistry) -> None:
        """capture the registry; no I/O.

        :param registry: the registry holding the collections
        :ptype registry: CollectionRegistry
        :return: nothing
        :rtype: None
        """
        self._registry = registry

    def _collection(self, table_name: str) -> Any:
        """the collection the registry holds for ``table_name``, which keeps its generation in L2.

        :param table_name: the table
        :ptype table_name: str
        :return: the collection
        :rtype: Any
        :raises GenerationUnavailableError: when the registry holds none that keeps it in L2
        """
        collection = self._registry.get_collection(table_name)
        if collection is None or not self._registry.generation_kept_in_l2(table_name):
            raise GenerationUnavailableError(f"{table_name!r} is no table this registry keeps in L2")
        return collection

    async def watch(self, table_name: str) -> AsyncGenerator[str | None]:
        """the table's generation as it stands, then each value it takes.

        :param table_name: the table
        :ptype table_name: str
        :return: the generations
        :rtype: AsyncGenerator[str | None]
        :raises GenerationUnavailableError: when the generation key cannot be watched
        """
        generations = self._collection(table_name).watch_l2_write_generation()
        try:
            async for token in generations:
                yield token
        finally:
            await generations.aclose()

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        """the change record of one epoch of the table, from beside its generation in L2.

        :param table_name: the table
        :ptype table_name: str
        :param token: the epoch
        :ptype token: str
        :return: the record, or ``None`` when L2 does not hold it
        :rtype: EpochChange | None
        """
        record: EpochChange | None = await self._collection(table_name).l2_epoch_change(table_name, token)
        return record

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        """the table's epochs since a mark: an L2-kept table numbers them one apart.

        :param table_name: the table
        :ptype table_name: str
        :param after: the mark
        :ptype after: str
        :param upto: the generation read
        :ptype upto: str
        :return: the walk
        :rtype: EpochWalk
        """
        return await walk_by_count(self, table_name, after, upto)


def follow_l2_kept_tables(
    registry: CollectionRegistry,
    *,
    grace: timedelta = DEFAULT_BROADCAST_GRACE,
    restart_delay: timedelta = DEFAULT_WATCH_RESTART_DELAY,
    max_restart_delay: timedelta = MAX_WATCH_RESTART_DELAY,
) -> TableFollower:
    """follow every table ``registry`` keeps in L2 by watching its generation key, and return the follower, started.

    The event-driven catch-up for ``L2_WRITE_GENERATION`` tables: each advance is pushed and its missed
    epochs applied as it lands, so no timed pass is needed. Register the collections first, and stop the
    follower before the registry's invalidation listener.

    :param registry: the registry holding the collections, whose listener hears their rows
    :ptype registry: CollectionRegistry
    :param grace: how long a watch waits for an advance's rows before walking its epochs
    :ptype grace: timedelta
    :param restart_delay: a failing watch's first wait before it starts again
    :ptype restart_delay: timedelta
    :param max_restart_delay: the longest wait between a failing watch's attempts
    :ptype max_restart_delay: timedelta
    :return: the running follower
    :rtype: TableFollower
    :raises RuntimeError: when the registry's invalidation listener is not running
    """
    return follow_tables(
        registry,
        _L2KeptGenerations(registry),
        registry.l2_kept_tables(),
        grace=grace,
        restart_delay=restart_delay,
        max_restart_delay=max_restart_delay,
    )
