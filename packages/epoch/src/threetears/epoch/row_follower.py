"""a cache derived from tables' rows, kept by key: each changed row is read again by its key, never the table whole.

The one wiring every such cache shares (a gateway's catalog and credit rates, a model-capability registry, a
publisher's identity): register on a registry for each table's row messages
(:meth:`~threetears.core.collections.registry.CollectionRegistry.register_derived_cache`), follow the tables' write
generations so a missed broadcast is caught up from the change record (:func:`~threetears.epoch.follow_tables`), and
run each row's keyed read as a task, so the synchronous row callback never blocks.

The cache's one whole read is its cold start, at the process's start, ledgered where the cache writes it. A table
dropped whole (an operator's reset) goes to ``on_table_dropped``, which never reads whole: the reset's refill reaches
the cache row by row. A load from files names its reach (``on_partition``), read by that prefix.

A cold read taken before the follow starts (a process's start reads its caches long before it follows them) is followed
from the generation it was read at: :func:`cold_read_generations` reads each table's generation just BEFORE the cold
read, and the follow (``RowFollower.follow(..., since=...)``, or :func:`follow_since` beside several followers) starts
each table's mark there, so every change written between the cold read and the follow is walked from the change log
and read by key. Without it the follow's first sight is taken as its mark, and such a change is missed until a
restart.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Mapping, Sequence
from typing import Any

from threetears.core.collections.generation import GenerationVerdict
from threetears.core.collections.registry import CacheInvalidationMessage, CollectionRegistry
from threetears.observe import cancel_and_wait, get_logger

from threetears.epoch.generation import EpochGenerationReader
from threetears.epoch.table_follow import TableFollower, follow_tables

__all__ = ["RowFollower", "cold_read_generations", "follow_since"]

log = get_logger(__name__)


async def cold_read_generations(reader: EpochGenerationReader, tables: Sequence[str]) -> dict[str, str | None]:
    """each table's write generation, read just BEFORE a cold read: the mark the follow starts from.

    A cache read cold holds every row as of a moment at or after these generations, so every change it can have missed
    between its cold read and its follow (:func:`follow_since`) is an epoch after them.

    :param reader: reads the epoch change log
    :ptype reader: EpochGenerationReader
    :param tables: the tables the cache follows
    :ptype tables: Sequence[str]
    :return: table -> its generation, ``None`` where the log holds none
    :rtype: dict[str, str | None]
    :raises GenerationUnavailableError: when the log cannot be read; a cache does not start blind to its gap
    """
    return {table: await reader.read(table) for table in tables}


def follow_since(registry: CollectionRegistry, generations: Mapping[str, str | None]) -> None:
    """start each table's mark at the generation read before a cold read, so no change after it is missed.

    Call right after the follow (``RowFollower.follow``, which registers and starts the tables' watches without
    yielding), before anything awaits: each watch's first push is the table's head, judged against this mark, so every
    epoch written after the cold read began is walked and its rows read by key -- by this cache and every other derived
    from the table, which re-read those keys harmlessly. Adopted after the follow, never before: a cache registers on
    the registry in its follow, and a write heard before it is registered must not count as heard.

    A mark the registry already holds at or before the cold read's (another follower of the table, started earlier) is
    kept: its walk covers this gap too, since this cache is registered by now. A mark past it is moved back to it. A mark
    under another incarnation is kept: the table's next judgement carries it over or falls behind, as for any follower.

    :param registry: the registry the tables are followed on
    :ptype registry: CollectionRegistry
    :param generations: :func:`cold_read_generations`' answer
    :ptype generations: Mapping[str, str | None]
    :return: nothing
    :rtype: None
    :raises KeyError: when a table is not followed on ``registry`` yet
    """
    marks = registry.generation_marks
    for table, generation in generations.items():
        if marks.judge(table, generation) in (GenerationVerdict.FIRST_SIGHT, GenerationVerdict.CURRENT):
            marks.adopt_generation(table, generation)


class RowFollower:
    """follows tables' rows for a derived cache: each changed row's key goes to that table's keyed read.

    :param reads: per table, the keyed read of one changed row, given its primary key's first column
        (``message.ids[0]``): a single-column key, or a table read by its key's lead
    :ptype reads: Mapping[str, Callable[[str], Coroutine[Any, Any, None]]]
    :param on_table_dropped: what a table dropped whole (an operator's reset) does; never a whole re-read
    :ptype on_table_dropped: Callable[[], None]
    :param name: the cache's name, for task names and logs
    :ptype name: str
    :param partition_reads: per table, the keyed read of a load's reach (its key prefix, column -> value), in place
        of dropping the cache; a table without one drops it (``on_table_dropped``)
    :ptype partition_reads: Mapping[str, Callable[[Mapping[str, str]], Coroutine[Any, Any, None]]] | None
    :param key_reads: per table, the keyed read of one changed row given its WHOLE primary key, every column in the
        key's declared order (``tuple(message.ids)``): a composite-key table whose cache applies a row by its full key
        (a tenant's row by ``(customer_id, id)``), never its tenant's rows. A table is in ``reads`` or here, not both
    :ptype key_reads: Mapping[str, Callable[[tuple[str, ...]], Coroutine[Any, Any, None]]] | None
    """

    def __init__(
        self,
        reads: Mapping[str, Callable[[str], Coroutine[Any, Any, None]]],
        on_table_dropped: Callable[[], None],
        *,
        name: str,
        partition_reads: Mapping[str, Callable[[Mapping[str, str]], Coroutine[Any, Any, None]]] | None = None,
        key_reads: Mapping[str, Callable[[tuple[str, ...]], Coroutine[Any, Any, None]]] | None = None,
    ) -> None:
        """bind the reads; no I/O.

        :return: nothing
        :rtype: None
        :raises ValueError: when no table is read, a table is given both a lead read and a whole-key read, or a
            partition read names a table with no row read
        """
        whole_keys = dict(key_reads or {})
        if not reads and not whole_keys:
            raise ValueError(f"{name}: a row follower reads at least one table's rows")
        both = sorted(set(reads) & set(whole_keys))
        if both:
            raise ValueError(f"{name}: {both} each take one row read, by its key's lead or by its whole key")
        unknown = sorted(set(partition_reads or {}) - set(reads) - set(whole_keys))
        if unknown:
            raise ValueError(f"{name}: partition reads of {unknown} name tables it does not follow")
        self._reads_by_table = dict(reads)
        self._key_reads_by_table = whole_keys
        self._tables = (*self._reads_by_table, *self._key_reads_by_table)
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
        if not message.ids:
            return
        whole_key = self._key_reads_by_table.get(table)
        if whole_key is not None:
            self.spawn(whole_key(tuple(message.ids)), f"{self._name}-{table}-read")
        else:
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

    def follow(
        self, registry: CollectionRegistry, reader: Any, *, since: Mapping[str, str | None] | None = None
    ) -> None:
        """register on ``registry`` for each table's rows and follow the tables' generations.

        Call once ``registry``'s invalidation listener runs; :meth:`stop` before it stops.

        :param registry: the registry holding the tables' collections, whose listener hears the rows
        :ptype registry: CollectionRegistry
        :param reader: reads and watches the epoch bucket
        :ptype reader: GenerationWatcher
        :param since: the generations read before the cache's cold read (:func:`cold_read_generations`), when it was
            read before this follow: each table is followed from there (:func:`follow_since`); None when the cache is
            read cold only after its follow starts, which hears every change from then on
        :ptype since: Mapping[str, str | None] | None
        :return: nothing
        :rtype: None
        :raises ValueError: when ``since`` names a table this follower does not follow
        """
        unknown = sorted(set(since or {}) - set(self._tables))
        if unknown:
            raise ValueError(f"{self._name}: since names {unknown}, which it does not follow")
        self._unregister = [
            registry.register_derived_cache(
                table,
                on_row=self._row_of(table),
                on_table_dropped=self._on_table_dropped,
                on_partition=self._partition_of(table) if table in self._partition_reads else None,
            )
            for table in self._tables
        ]
        self._follower = follow_tables(registry, reader, self._tables)
        if since is not None:
            follow_since(registry, since)
        log.info("following tables by key: cache=%s tables=%s", self._name, sorted(self._tables))

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
