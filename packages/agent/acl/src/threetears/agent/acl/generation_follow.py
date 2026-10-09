"""follow the access tables' write generations, one key watch per table, for as long as a process runs.

A registry that follows a table keeps a mark of the generations it has heard every row of, and
drops the table -- its rows, its scans, every cache derived from it -- when the epoch bucket shows
an advance it did not hear (epoch-task-06). The four access tables are followed by watching their
generation keys (``threetears.epoch.follow_generation_key``), the owner's decision for these
tables: a missed broadcast is noticed when the generation moves, not at a scheduled pass.

:class:`AccessTableFollower` runs one watch per table as a task, and starts a watch again when one
ends or fails -- a closed connection ends every watch, and a reconnected client must be watched
again. A restarted watch is first pushed the key's latest value, so an advance missed while it was
down is judged then. The wait before a restart is a retry delay for the watch, not an age on any
cached value: nothing cached here expires.

Needs ``3tears-epoch``, which comes with the ``3tears-agent-acl[bus]`` extra.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import TYPE_CHECKING, Final

from threetears.epoch import DEFAULT_BROADCAST_GRACE, follow_generation_key
from threetears.observe import get_logger

from threetears.agent.acl.access_tables import ACCESS_TABLES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from threetears.core.collections.registry import CollectionRegistry
    from threetears.epoch import EpochGenerationReader

__all__ = ["DEFAULT_WATCH_RESTART_DELAY", "AccessTableFollower"]

log = get_logger(__name__)

#: how long a watch that ended or failed waits before it is started again
DEFAULT_WATCH_RESTART_DELAY: Final = timedelta(seconds=1)


class AccessTableFollower:
    """one supervised generation-key watch per access table, on one registry.

    :param registry: the registry that follows the tables; its invalidation listener must be
        running, since it is what hears the rows a watch judges against
    :ptype registry: CollectionRegistry
    :param reader: reads and watches the epoch bucket
    :ptype reader: EpochGenerationReader
    :param tables: the tables to follow; the four access tables by default
    :ptype tables: Sequence[str]
    :param grace: how long a watch waits for an advance's rows before judging them missed
    :ptype grace: timedelta
    :param restart_delay: how long a watch that ended or failed waits before it starts again
    :ptype restart_delay: timedelta
    """

    def __init__(
        self,
        registry: CollectionRegistry,
        reader: EpochGenerationReader,
        *,
        tables: Sequence[str] = ACCESS_TABLES,
        grace: timedelta = DEFAULT_BROADCAST_GRACE,
        restart_delay: timedelta = DEFAULT_WATCH_RESTART_DELAY,
    ) -> None:
        """capture what to follow; no I/O.

        :return: nothing
        :rtype: None
        """
        self._registry = registry
        self._reader = reader
        self._tables = tuple(tables)
        self._grace = grace
        self._restart_delay = restart_delay
        self._tasks: list[asyncio.Task[None]] = []

    @property
    def running(self) -> bool:
        """whether the watches are running.

        :return: ``True`` between :meth:`start` and :meth:`stop`
        :rtype: bool
        """
        return bool(self._tasks)

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

    async def _watch(self, table: str) -> None:
        """watch one table's generation key until cancelled, starting the watch again when it ends.

        :param table: the table
        :ptype table: str
        :return: nothing; it ends only when cancelled
        :rtype: None
        """
        while True:
            try:
                await follow_generation_key(self._registry, self._reader, table, grace=self._grace)
                log.warning("a write-generation watch ended; starting it again", extra={"extra_data": {"table": table}})
            except asyncio.CancelledError:
                raise
            # prawduct:allow prawduct/broad-except -- a watch that fails for any reason is started again;
            # stopping would leave the table followed with nothing to judge its mark
            except Exception:  # noqa: BLE001
                log.warning(
                    "a write-generation watch failed; starting it again",
                    exc_info=True,
                    extra={"extra_data": {"table": table}},
                )
            await asyncio.sleep(self._restart_delay.total_seconds())
