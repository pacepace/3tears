"""a cache derived from tables' rows, kept by key: each changed row is read again by its key, never the table whole.

The one wiring every such cache shares (a gateway's catalog and credit rates, a model-capability registry, a
publisher's identity): register on a registry for each table's row messages
(:meth:`~threetears.core.collections.registry.CollectionRegistry.register_derived_cache`), follow the tables' write
generations so a missed broadcast is caught up from the change record (:func:`~threetears.epoch.follow_tables`), and
run each row's keyed read as a task, so the synchronous row callback never blocks.

The cache's one whole read is its cold start, at the process's start, ledgered where the cache writes it. A table
dropped whole (an operator's reset) goes to ``on_table_dropped``, which never reads whole: the reset's refill reaches
the cache row by row. A load from files names its reach (``on_partition``), read by that prefix.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Mapping
from typing import Any

from threetears.core.collections.registry import CacheInvalidationMessage, CollectionRegistry
from threetears.observe import cancel_and_wait, get_logger

from threetears.epoch.table_follow import TableFollower, follow_tables

__all__ = ["RowFollower"]

log = get_logger(__name__)


class RowFollower:
    """follows tables' rows for a derived cache: each changed row's key goes to that table's keyed read.

    :param reads: per table, the keyed read of one changed row, given the row's primary key (``message.ids[0]``)
    :ptype reads: Mapping[str, Callable[[str], Coroutine[Any, Any, None]]]
    :param on_table_dropped: what a table dropped whole (an operator's reset) does; never a whole re-read
    :ptype on_table_dropped: Callable[[], None]
    :param name: the cache's name, for task names and logs
    :ptype name: str
    :param partition_reads: per table, the keyed read of a load's reach (its key prefix, column -> value), in place
        of dropping the cache; a table without one drops it (``on_table_dropped``)
    :ptype partition_reads: Mapping[str, Callable[[Mapping[str, str]], Coroutine[Any, Any, None]]] | None
    """

    def __init__(
        self,
        reads: Mapping[str, Callable[[str], Coroutine[Any, Any, None]]],
        on_table_dropped: Callable[[], None],
        *,
        name: str,
        partition_reads: Mapping[str, Callable[[Mapping[str, str]], Coroutine[Any, Any, None]]] | None = None,
    ) -> None:
        """bind the reads; no I/O.

        :return: nothing
        :rtype: None
        :raises ValueError: when no table is read, or a partition read names a table with no row read
        """
        if not reads:
            raise ValueError(f"{name}: a row follower reads at least one table's rows")
        unknown = sorted(set(partition_reads or {}) - set(reads))
        if unknown:
            raise ValueError(f"{name}: partition reads of {unknown} name tables it does not follow")
        self._reads_by_table = dict(reads)
        self._partition_reads = dict(partition_reads or {})
        self._on_table_dropped = on_table_dropped
        self._name = name
        self._tasks: set[asyncio.Task[None]] = set()
        self._follower: TableFollower | None = None
        self._unregister: list[Callable[[], None]] = []

    def on_row(self, table: str, message: CacheInvalidationMessage) -> None:
        """a row of ``table`` changed: start its keyed read as a task (synchronous).

        :param table: the table
        :ptype table: str
        :param message: the row's message; ``ids`` holds its primary key
        :ptype message: CacheInvalidationMessage
        :return: nothing
        :rtype: None
        """
        if message.ids:
            self.spawn(self._reads_by_table[table](message.ids[0]), f"{self._name}-{table}-read")

    def on_partition(self, table: str, reach: Mapping[str, str]) -> None:
        """a load from files wrote ``reach`` of ``table``: start that prefix's keyed read as a task (synchronous).

        :param table: the table
        :ptype table: str
        :param reach: the load's key prefix, column -> value
        :ptype reach: Mapping[str, str]
        :return: nothing
        :rtype: None
        """
        self.spawn(self._partition_reads[table](dict(reach)), f"{self._name}-{table}-partition")

    def _row_of(self, table: str) -> Callable[[CacheInvalidationMessage], None]:
        def on_row(message: CacheInvalidationMessage) -> None:
            self.on_row(table, message)

        return on_row

    def _partition_of(self, table: str) -> Callable[[Mapping[str, str]], None]:
        def on_partition(reach: Mapping[str, str]) -> None:
            self.on_partition(table, reach)

        return on_partition

    def spawn(self, work: Coroutine[Any, Any, None], name: str) -> None:
        """run ``work`` as a task kept until it ends.

        :param work: the coroutine
        :ptype work: Coroutine[Any, Any, None]
        :param name: the task's name
        :ptype name: str
        :return: nothing
        :rtype: None
        """
        task = asyncio.get_running_loop().create_task(work, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def settled(self) -> None:
        """wait for every read started so far, and any they started.

        :return: nothing
        :rtype: None
        """
        while self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    def follow(self, registry: CollectionRegistry, reader: Any) -> None:
        """register on ``registry`` for each table's rows and follow the tables' generations.

        Call once ``registry``'s invalidation listener runs; :meth:`stop` before it stops.

        :param registry: the registry holding the tables' collections, whose listener hears the rows
        :ptype registry: CollectionRegistry
        :param reader: reads and watches the epoch bucket
        :ptype reader: GenerationWatcher
        :return: nothing
        :rtype: None
        """
        self._unregister = [
            registry.register_derived_cache(
                table,
                on_row=self._row_of(table),
                on_table_dropped=self._on_table_dropped,
                on_partition=self._partition_of(table) if table in self._partition_reads else None,
            )
            for table in self._reads_by_table
        ]
        self._follower = follow_tables(registry, reader, tuple(self._reads_by_table))
        log.info("following tables by key: cache=%s tables=%s", self._name, sorted(self._reads_by_table))

    async def stop(self) -> None:
        """stop following, unregister, and cancel any read still running. Idempotent.

        :return: nothing
        :rtype: None
        """
        if self._follower is not None:
            await self._follower.stop()
            self._follower = None
        for unregister in self._unregister:
            unregister()
        self._unregister = []
        for task in tuple(self._tasks):
            await cancel_and_wait(task)
