"""follow the access tables' write generations, one key watch per table, for as long as a process runs.

A registry that follows a table keeps a mark of the generations it has heard every row of, and
drops the table -- its rows, its scans, every cache derived from it -- when the epoch bucket shows
an advance it did not hear (epoch-task-06). The access tables are followed by watching their
generation keys (``threetears.epoch.follow_generation_key``), the owner's decision for these
tables: a missed broadcast is noticed when the generation moves, not at a scheduled pass.

**One call for a consumer: :func:`follow_access_tables`.** It binds an :class:`AclCache` to the
tables' row broadcasts and follows them, and returns one handle whose :meth:`AccessTableFollowing.stop`
undoes both. Each half is unsafe alone: binding without following never drops what a missed
broadcast left stale, and following without binding drops the registry's rows but never tells the
cache.

:class:`AccessTableFollower` runs one watch per table as a task, and starts a watch again when one
ends or fails, after a delay that doubles with each consecutive failure up to a cap and resets when
a watch is pushed a value. A restarted watch is first pushed the key's latest value, so an advance
missed while it was down is judged then. The delay is about the watch's liveness, not an age on any
cached value: nothing cached here expires. Each table's :class:`WatchHealth` is readable, so a
caller can report a follower whose watches keep failing.

Needs ``3tears-epoch``, which comes with the ``3tears-agent-acl[bus]`` extra.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Final

from threetears.epoch import DEFAULT_BROADCAST_GRACE, GenerationWatcher, follow_generation_key
from threetears.observe import get_logger

from threetears.agent.acl.access_tables import ACCESS_TABLES, DegradedEvictions, bind_acl_cache_to_access_tables

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from threetears.core.collections.registry import CollectionRegistry

    from threetears.agent.acl.cache import AclCache

__all__ = [
    "DEFAULT_WATCH_RESTART_DELAY",
    "MAX_WATCH_RESTART_DELAY",
    "AccessTableFollower",
    "AccessTableFollowing",
    "WatchHealth",
    "follow_access_tables",
]

log = get_logger(__name__)

#: how long a watch that ended or failed for the first time in a row waits before it starts again
DEFAULT_WATCH_RESTART_DELAY: Final = timedelta(seconds=1)

#: the longest a failing watch waits between attempts, however many times in a row it has failed
MAX_WATCH_RESTART_DELAY: Final = timedelta(seconds=60)


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


class AccessTableFollower:
    """one supervised generation-key watch per access table, on one registry.

    :param registry: the registry that follows the tables; its invalidation listener must be
        running, since it is what hears the rows a watch judges against
    :ptype registry: CollectionRegistry
    :param reader: reads and watches the epoch bucket
    :ptype reader: GenerationWatcher
    :param tables: the tables to follow; the access tables by default
    :ptype tables: Sequence[str]
    :param grace: how long a watch waits for an advance's rows before judging them missed
    :ptype grace: timedelta
    :param restart_delay: how long a watch waits after its first failure in a row; doubled after
        each further one, up to ``max_restart_delay``
    :ptype restart_delay: timedelta
    :param max_restart_delay: the longest wait between attempts
    :ptype max_restart_delay: timedelta
    """

    def __init__(
        self,
        registry: CollectionRegistry,
        reader: GenerationWatcher,
        *,
        tables: Sequence[str] = ACCESS_TABLES,
        grace: timedelta = DEFAULT_BROADCAST_GRACE,
        restart_delay: timedelta = DEFAULT_WATCH_RESTART_DELAY,
        max_restart_delay: timedelta = MAX_WATCH_RESTART_DELAY,
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
        self._health = {table: WatchHealth() for table in self._tables}
        self._reader = _ObservedWatcher(reader, self._health)
        self._tasks: list[asyncio.Task[None]] = []

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
        """
        if self._tasks:
            return
        for table in self._tables:
            self._registry.follow_generation(table)
            self._tasks.append(asyncio.create_task(self._watch(table), name=f"follow-generation:{table}"))
        log.info("following the access tables' write generations", extra={"extra_data": {"tables": self._tables}})

    async def stop(self) -> None:
        """cancel every watch and wait for it to end. Idempotent.

        :return: nothing
        :rtype: None
        """
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass  # NOSILENT: the cancellation this method just asked for, ending the watch

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
                health.last_error = f"{type(exc).__name__}: {exc}"
            health.consecutive_failures += 1
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


@dataclass(frozen=True)
class AccessTableFollowing:
    """an :class:`AclCache` bound to the access tables and following them; :meth:`stop` undoes both.

    :ivar follower: the watches, whose :attr:`AccessTableFollower.health` a caller can report
    :ivar degraded: the unknown-reach evictions counted for this cache
    """

    follower: AccessTableFollower
    degraded: DegradedEvictions
    _unbind: Callable[[], None]

    @property
    def healthy(self) -> bool:
        """whether every table's watch is running and pushed since it last failed.

        :return: the follower's health
        :rtype: bool
        """
        return self.follower.healthy

    async def stop(self) -> None:
        """stop the watches, then unbind the cache. Idempotent.

        :return: nothing
        :rtype: None
        """
        await self.follower.stop()
        self._unbind()


def follow_access_tables(
    registry: CollectionRegistry,
    cache: AclCache,
    reader: GenerationWatcher,
    *,
    grace: timedelta = DEFAULT_BROADCAST_GRACE,
    restart_delay: timedelta = DEFAULT_WATCH_RESTART_DELAY,
    max_restart_delay: timedelta = MAX_WATCH_RESTART_DELAY,
) -> AccessTableFollowing:
    """bind ``cache`` to the access tables' row broadcasts on ``registry`` and follow the tables.

    The one call a consumer makes. ``registry``'s invalidation listener must be running (or be
    started before the first write that matters): it is what hears the rows.

    :param registry: the registry whose listener hears the rows, and which follows the tables
    :ptype registry: CollectionRegistry
    :param cache: the cache to evict from
    :ptype cache: AclCache
    :param reader: reads and watches the epoch bucket
    :ptype reader: GenerationWatcher
    :param grace: how long a watch waits for an advance's rows before judging them missed
    :ptype grace: timedelta
    :param restart_delay: a failing watch's first wait before it starts again
    :ptype restart_delay: timedelta
    :param max_restart_delay: the longest wait between a failing watch's attempts
    :ptype max_restart_delay: timedelta
    :return: the handle that stops both
    :rtype: AccessTableFollowing
    """
    degraded = DegradedEvictions()
    unbind = bind_acl_cache_to_access_tables(registry, cache, degraded=degraded)
    follower = AccessTableFollower(
        registry, reader, grace=grace, restart_delay=restart_delay, max_restart_delay=max_restart_delay
    )
    follower.start()
    return AccessTableFollowing(follower=follower, degraded=degraded, _unbind=unbind)
