"""see what the checkpoint tables hold, and shrink them, for any app that runs the saver.

LangGraph writes a full checkpoint after every superstep, and once a run ends only the newest
checkpoint of each namespace is read again: to resume a paused run, or to read its final state.
The rest is history nothing reads, and it is most of the table. This package measures that and
removes it, knowing nothing of the app beyond what the app hands it:

- :func:`checkpoint_usage` -- table sizes, row counts, live and dead tuples, what ``keep_latest``
  would remove, and, given the app's own query of the threads it still owns, what belongs to
  threads it no longer does. Read-only.
- :func:`prune_threads` -- runs :meth:`~threetears.langgraph.checkpoint.ThreeTierCheckpointSaver.aprune`
  with ``keep_latest`` over every thread holding history, a batch at a time.
- :func:`delete_unowned_threads` -- deletes every thread the app's query does not name.

``python -m threetears.langgraph.checkpoint_maintenance`` runs the three from a shell
(:mod:`threetears.langgraph.checkpoint_maintenance.__main__`).

**Stored thread ids.** Everything here works on ``thread_id`` as stored. A saver scoped to a
customer stores ``"<customer>/<thread>"`` (see
:meth:`~threetears.langgraph.checkpoint.ThreeTierCheckpointSaver.storage_thread_id`), so an
owned-threads query for such an app returns that form.

**Which saver.** :func:`prune_threads` and :func:`delete_unowned_threads` take an UNSCOPED saver
over the same tables, and refuse any other. They name threads by stored id, which an unscoped saver
addresses unchanged, so one pass reaches every customer's rows; a saver scoped to a customer would
prefix the id a second time and address nothing, and a config-resolved one has no customer to
address with. Customer scoping is a naming scheme inside the key, not an access control on the
table: an operator holding the database already reaches every row, and this states that instead of
pretending otherwise. An app that wants to prune its own threads under its own scope calls
``aprune`` directly with the ids it knows.

**Space.** Deleting rows frees nothing on disk. ``VACUUM`` makes the space reusable by the same
table; only ``VACUUM FULL`` or ``pg_repack`` returns it to the operating system, and ``VACUUM FULL``
locks the table for its whole run. :func:`checkpoint_usage` reports dead tuples so the report can
say how much is waiting on that.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from threetears.langgraph.checkpoint import (
    DELTA_COUNTERS_METADATA_KEY,
    CheckpointPruneResult,
    ThreeTierCheckpointSaver,
    keep_latest_survivors_sql,
)
from threetears.langgraph.protocols import AsyncQueryExecutor
from threetears.observe import get_logger

__all__ = [
    "DEFAULT_BATCH",
    "CheckpointSweep",
    "CheckpointUsage",
    "TableUsage",
    "UnownedUsage",
    "checkpoint_usage",
    "delete_unowned_threads",
    "prune_threads",
]

log = get_logger(__name__)

#: threads named per ``aprune`` call when the caller names no batch size.
DEFAULT_BATCH: Final = 200

_CHECKPOINT_BYTES = "pg_column_size({a}.checkpoint) + coalesce(pg_column_size({a}.metadata_), 0)"
_WRITE_BYTES = "pg_column_size({a}.blob)"


@dataclass(frozen=True, slots=True)
class TableUsage:
    """one checkpoint table's size and rows.

    :ivar table: the table's name
    :ivar total_bytes: ``pg_total_relation_size``: heap, indexes and TOAST
    :ivar heap_bytes: ``pg_relation_size``: the main heap alone
    :ivar rows: rows counted now
    :ivar live_tuples: ``pg_stat_user_tables.n_live_tup``, the statistics' estimate
    :ivar dead_tuples: ``pg_stat_user_tables.n_dead_tup``: deleted rows whose space ``VACUUM`` has
        not yet made reusable
    """

    table: str
    total_bytes: int
    heap_bytes: int
    rows: int
    live_tuples: int
    dead_tuples: int


@dataclass(frozen=True, slots=True)
class UnownedUsage:
    """what belongs to threads the app's owned-threads query does not name.

    :ivar owned_threads: rows the owned-threads query returned
    :ivar threads: stored thread ids in either table that the query does not name
    :ivar checkpoint_rows: their ``checkpoints`` rows
    :ivar write_rows: their ``checkpoint_writes`` rows
    :ivar bytes: their blob bytes as stored (``pg_column_size``, after compression)
    """

    owned_threads: int
    threads: int
    checkpoint_rows: int
    write_rows: int
    bytes: int


@dataclass(frozen=True, slots=True)
class CheckpointUsage:
    """what the checkpoint tables hold, and what pruning would remove.

    Byte counts of rows are blob bytes as stored -- ``pg_column_size`` of the checkpoint and
    metadata, or the write's blob, after compression -- without keys or tuple headers, so they
    are what a prune frees inside the table, not what it returns to the operating system.

    :ivar checkpoints: the ``checkpoints`` table
    :ivar writes: the ``checkpoint_writes`` table
    :ivar threads: distinct stored thread ids in either table
    :ivar threads_with_history: threads holding more than one checkpoint in some namespace
    :ivar prunable_checkpoint_rows: ``checkpoints`` rows ``keep_latest`` would remove
    :ivar prunable_write_rows: ``checkpoint_writes`` rows ``keep_latest`` would remove
    :ivar prunable_bytes: their blob bytes
    :ivar held_namespaces: namespaces ``keep_latest`` would leave whole, their newest checkpoint
        still needing its ancestors (a LangGraph ``DeltaChannel``)
    :ivar unowned: the threads outside the app's owned set, or None when no query was given
    """

    checkpoints: TableUsage
    writes: TableUsage
    threads: int
    threads_with_history: int
    prunable_checkpoint_rows: int
    prunable_write_rows: int
    prunable_bytes: int
    held_namespaces: int
    unowned: UnownedUsage | None


@dataclass(frozen=True, slots=True)
class CheckpointSweep:
    """what a :func:`prune_threads` or :func:`delete_unowned_threads` run removed, or would.

    :ivar strategy: ``"keep_latest"`` or ``"delete"``
    :ivar dry_run: True when nothing was deleted and the counts are what a real run would delete
    :ivar batches: batches of threads handled
    :ivar threads: threads handled
    :ivar checkpoint_rows: ``checkpoints`` rows deleted (or that would be)
    :ivar write_rows: ``checkpoint_writes`` rows deleted (or that would be)
    :ivar bytes: their blob bytes, read just before each batch was deleted
    :ivar namespaces_held: namespaces ``keep_latest`` left whole (a LangGraph ``DeltaChannel``)
    """

    strategy: str
    dry_run: bool
    batches: int
    threads: int
    checkpoint_rows: int
    write_rows: int
    bytes: int
    namespaces_held: int


def _owned_cte(owned_threads_sql: str) -> str:
    """the ``WITH owned AS (...)`` clause over the app's owned-threads query.

    The query's first column is read as the stored thread id, cast to text; a trailing semicolon is
    dropped so a query pasted from a shell works.

    :param owned_threads_sql: a SELECT returning the stored ids of the threads the app owns
    :ptype owned_threads_sql: str
    :return: the clause, ending in a space
    :rtype: str
    :raises ValueError: when the query is blank
    """
    body = owned_threads_sql.strip().rstrip(";").strip()
    if not body:
        raise ValueError("owned_threads_sql is blank; it must be a SELECT returning the stored thread ids still owned")
    return f"WITH owned AS (SELECT o.thread_id::text AS thread_id FROM ({body}) AS o(thread_id)) "


def _not_owned(alias: str) -> str:
    """the predicate true for a row of ``alias`` whose thread the owned set does not name.

    :param alias: the table alias in the enclosing query
    :ptype alias: str
    :return: SQL predicate
    :rtype: str
    """
    return f"NOT EXISTS (SELECT 1 FROM owned WHERE owned.thread_id = {alias}.thread_id)"


def _int(row: dict[str, Any] | None, key: str) -> int:
    """an integer column of a row that may be absent or NULL (an aggregate over no rows).

    :param row: the row, or None
    :ptype row: dict[str, Any] | None
    :param key: the column
    :ptype key: str
    :return: the value, 0 for None
    :rtype: int
    """
    value = None if row is None else row[key]
    return 0 if value is None else int(value)


async def _table_usage(executor: AsyncQueryExecutor, table: str) -> TableUsage:
    """size, rows and tuple statistics of one checkpoint table.

    :param executor: the database
    :ptype executor: AsyncQueryExecutor
    :param table: ``checkpoints`` or ``checkpoint_writes``, a fixed name, never caller input
    :ptype table: str
    :return: the table's usage
    :rtype: TableUsage
    """
    sizes = await executor.fetchrow(
        f"SELECT pg_total_relation_size('{table}'::regclass) AS total_bytes, "
        f"pg_relation_size('{table}'::regclass) AS heap_bytes, "
        f"(SELECT count(*) FROM {table}) AS row_count"
    )
    stats = await executor.fetchrow(
        f"SELECT n_live_tup, n_dead_tup FROM pg_stat_user_tables WHERE relid = '{table}'::regclass"
    )
    return TableUsage(
        table=table,
        total_bytes=_int(sizes, "total_bytes"),
        heap_bytes=_int(sizes, "heap_bytes"),
        rows=_int(sizes, "row_count"),
        live_tuples=_int(stats, "n_live_tup"),
        dead_tuples=_int(stats, "n_dead_tup"),
    )


async def _prunable(
    executor: AsyncQueryExecutor, storage_thread_ids: Sequence[str] | None
) -> tuple[int, int, int, int]:
    """what ``keep_latest`` would remove, by the predicate the saver's prune runs.

    :param executor: the database
    :ptype executor: AsyncQueryExecutor
    :param storage_thread_ids: the stored thread ids to measure, or None for every thread
    :ptype storage_thread_ids: Sequence[str] | None
    :return: checkpoint rows, write rows, blob bytes, and namespaces held whole
    :rtype: tuple[int, int, int, int]
    """
    marker = DELTA_COUNTERS_METADATA_KEY.encode()
    args: list[object] = [marker] if storage_thread_ids is None else [marker, list(storage_thread_ids)]
    threads_param = None if storage_thread_ids is None else 2
    survivors = keep_latest_survivors_sql(marker_param=1, threads_param=threads_param)
    match = "p.thread_id = k.thread_id AND p.checkpoint_ns = k.checkpoint_ns AND p.checkpoint_id < k.checkpoint_id"
    checkpoints = await executor.fetchrow(
        f"SELECT count(*) AS row_count, sum({_CHECKPOINT_BYTES.format(a='p')}) AS bytes "
        f"FROM checkpoints p JOIN ({survivors}) k ON {match}",
        *args,
    )
    writes = await executor.fetchrow(
        f"SELECT count(*) AS row_count, sum({_WRITE_BYTES.format(a='p')}) AS bytes "
        f"FROM checkpoint_writes p JOIN ({survivors}) k ON {match}",
        *args,
    )
    held = await executor.fetchrow(
        "SELECT count(*) AS held FROM ("
        + keep_latest_survivors_sql(marker_param=1, threads_param=threads_param, held=True)
        + ") h",
        *args,
    )
    return (
        _int(checkpoints, "row_count"),
        _int(writes, "row_count"),
        _int(checkpoints, "bytes") + _int(writes, "bytes"),
        _int(held, "held"),
    )


async def _thread_rows(executor: AsyncQueryExecutor, storage_thread_ids: Sequence[str]) -> tuple[int, int, int]:
    """every row the named threads hold, which ``delete`` removes.

    :param executor: the database
    :ptype executor: AsyncQueryExecutor
    :param storage_thread_ids: stored thread ids
    :ptype storage_thread_ids: Sequence[str]
    :return: checkpoint rows, write rows, blob bytes
    :rtype: tuple[int, int, int]
    """
    ids = list(storage_thread_ids)
    checkpoints = await executor.fetchrow(
        f"SELECT count(*) AS row_count, sum({_CHECKPOINT_BYTES.format(a='c')}) AS bytes "
        "FROM checkpoints c WHERE c.thread_id = ANY($1::text[])",
        ids,
    )
    writes = await executor.fetchrow(
        f"SELECT count(*) AS row_count, sum({_WRITE_BYTES.format(a='w')}) AS bytes "
        "FROM checkpoint_writes w WHERE w.thread_id = ANY($1::text[])",
        ids,
    )
    return (
        _int(checkpoints, "row_count"),
        _int(writes, "row_count"),
        _int(checkpoints, "bytes") + _int(writes, "bytes"),
    )


async def checkpoint_usage(executor: AsyncQueryExecutor, *, owned_threads_sql: str | None = None) -> CheckpointUsage:
    """measure the checkpoint tables. Read-only.

    Every count is exact (``count(*)``), so on a large table this reads all of it; run it when the
    database can afford a full scan. Tuple counts come from the statistics collector and are
    estimates.

    :param executor: the database holding ``checkpoints`` and ``checkpoint_writes`` (on its search path)
    :ptype executor: AsyncQueryExecutor
    :param owned_threads_sql: a SELECT whose first column is the stored ids of the threads the app
        still owns, compared with ``thread_id`` exactly as stored; None skips the unowned figures
    :ptype owned_threads_sql: str | None
    :return: the usage
    :rtype: CheckpointUsage
    :raises ValueError: when ``owned_threads_sql`` is blank
    """
    checkpoints = await _table_usage(executor, "checkpoints")
    writes = await _table_usage(executor, "checkpoint_writes")
    threads = await executor.fetchrow(
        "SELECT count(*) AS threads FROM (SELECT thread_id FROM checkpoints UNION SELECT thread_id FROM checkpoint_writes) t"
    )
    with_history = await executor.fetchrow(
        "SELECT count(DISTINCT thread_id) AS threads FROM "
        "(SELECT thread_id FROM checkpoints GROUP BY thread_id, checkpoint_ns HAVING count(*) > 1) multi"
    )
    prunable_checkpoints, prunable_writes, prunable_bytes, held = await _prunable(executor, None)
    unowned: UnownedUsage | None = None
    if owned_threads_sql is not None:
        owned = _owned_cte(owned_threads_sql)
        owned_count = await executor.fetchrow(f"{owned}SELECT count(*) AS owned FROM owned")
        unowned_threads = await executor.fetchrow(
            f"{owned}SELECT count(*) AS threads FROM (SELECT thread_id FROM checkpoints c WHERE {_not_owned('c')} "
            f"UNION SELECT thread_id FROM checkpoint_writes w WHERE {_not_owned('w')}) t"
        )
        unowned_checkpoints = await executor.fetchrow(
            f"{owned}SELECT count(*) AS row_count, sum({_CHECKPOINT_BYTES.format(a='c')}) AS bytes "
            f"FROM checkpoints c WHERE {_not_owned('c')}"
        )
        unowned_writes = await executor.fetchrow(
            f"{owned}SELECT count(*) AS row_count, sum({_WRITE_BYTES.format(a='w')}) AS bytes "
            f"FROM checkpoint_writes w WHERE {_not_owned('w')}"
        )
        unowned = UnownedUsage(
            owned_threads=_int(owned_count, "owned"),
            threads=_int(unowned_threads, "threads"),
            checkpoint_rows=_int(unowned_checkpoints, "row_count"),
            write_rows=_int(unowned_writes, "row_count"),
            bytes=_int(unowned_checkpoints, "bytes") + _int(unowned_writes, "bytes"),
        )
    return CheckpointUsage(
        checkpoints=checkpoints,
        writes=writes,
        threads=_int(threads, "threads"),
        threads_with_history=_int(with_history, "threads"),
        prunable_checkpoint_rows=prunable_checkpoints,
        prunable_write_rows=prunable_writes,
        prunable_bytes=prunable_bytes,
        held_namespaces=held,
        unowned=unowned,
    )


def _require_unscoped(saver: ThreeTierCheckpointSaver, operation: str) -> None:
    """refuse a saver that would not address stored thread ids unchanged.

    :param saver: the saver an operation was handed
    :ptype saver: ThreeTierCheckpointSaver
    :param operation: the function's name, for the message
    :ptype operation: str
    :return: nothing
    :rtype: None
    :raises ValueError: when the saver is scoped to a customer or resolves one per call
    """
    scope = saver.scope
    if scope.customer_id is not None or scope.config_key is not None:
        raise ValueError(
            f"{operation}() names threads by their stored id, which only an unscoped saver addresses unchanged; "
            f"this saver was built with {scope!r}. Build one with CheckpointScope.unscoped(reason=...) over the "
            "same tables.",
        )


def _check_batch(batch: int, limit: int | None) -> None:
    """refuse a batch size or limit that could not make progress.

    :param batch: threads per batch
    :ptype batch: int
    :param limit: most threads to handle, or None
    :ptype limit: int | None
    :return: nothing
    :rtype: None
    :raises ValueError: when ``batch`` is below 1 or ``limit`` is negative
    """
    if batch < 1:
        raise ValueError(f"batch must be at least 1; received {batch}")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must not be negative; received {limit}")


def _page_size(batch: int, limit: int | None, handled: int) -> int:
    """how many threads the next page may hold.

    :param batch: threads per batch
    :ptype batch: int
    :param limit: most threads to handle, or None
    :ptype limit: int | None
    :param handled: threads handled so far
    :ptype handled: int
    :return: the page size, 0 when the limit is reached
    :rtype: int
    """
    return batch if limit is None else min(batch, limit - handled)


async def prune_threads(
    saver: ThreeTierCheckpointSaver,
    executor: AsyncQueryExecutor,
    *,
    batch: int = DEFAULT_BATCH,
    limit: int | None = None,
    dry_run: bool = False,
) -> CheckpointSweep:
    """``keep_latest`` over every thread holding more than one checkpoint in some namespace.

    Threads are found ``batch`` at a time in stored-id order, each page starting after the last
    thread of the one before, so a run never revisits a thread and an interrupted run resumes by
    running again. Each batch is one :meth:`~ThreeTierCheckpointSaver.aprune` call. A dry run reads
    each batch's counts by the same predicate the prune runs and deletes nothing, so its totals are
    what a real run over an unchanged table deletes.

    A thread is found by its checkpoints. Writes recorded against no stored checkpoint (left by a
    checkpoint written while a prune ran) are removed when that thread is next found.

    ``saver`` must be unscoped and over the same tables as ``executor`` (see the package docstring
    for why). A saver with caches evicts each pruned thread from them; an L3-only saver leaves an
    app's caches as they are, which for ``keep_latest`` holds only the kept checkpoint anyway.

    :param saver: an unscoped saver over the tables
    :ptype saver: ThreeTierCheckpointSaver
    :param executor: the database the saver writes to
    :ptype executor: AsyncQueryExecutor
    :param batch: threads per ``aprune`` call
    :ptype batch: int
    :param limit: most threads to handle, or None for all
    :ptype limit: int | None
    :param dry_run: count without deleting
    :ptype dry_run: bool
    :return: the totals
    :rtype: CheckpointSweep
    :raises ValueError: when the saver is scoped, ``batch`` is below 1 or ``limit`` is negative
    """
    _require_unscoped(saver, "prune_threads")
    _check_batch(batch, limit)
    cursor = ""
    batches = threads = checkpoint_rows = write_rows = total_bytes = held = 0
    while (size := _page_size(batch, limit, threads)) > 0:
        rows = await executor.fetch(
            "SELECT DISTINCT thread_id FROM (SELECT thread_id FROM checkpoints WHERE thread_id > $1 "
            "GROUP BY thread_id, checkpoint_ns HAVING count(*) > 1 ORDER BY thread_id LIMIT $2) page "
            "ORDER BY thread_id",
            cursor,
            size,
        )
        if not rows:
            break
        page = [str(row["thread_id"]) for row in rows]
        cps, wrs, page_bytes, page_held = await _prunable(executor, page)
        if not dry_run:
            result: CheckpointPruneResult = await saver.aprune(page, strategy="keep_latest")
            cps, wrs, page_held = result.checkpoints_deleted, result.writes_deleted, result.namespaces_held
        batches += 1
        threads += len(page)
        checkpoint_rows += cps
        write_rows += wrs
        total_bytes += page_bytes
        held += page_held
        cursor = page[-1]
        log.info(
            "checkpoint prune batch",
            extra={"dry_run": dry_run, "threads": len(page), "checkpoint_rows": cps, "write_rows": wrs},
        )
    return CheckpointSweep(
        strategy="keep_latest",
        dry_run=dry_run,
        batches=batches,
        threads=threads,
        checkpoint_rows=checkpoint_rows,
        write_rows=write_rows,
        bytes=total_bytes,
        namespaces_held=held,
    )


async def delete_unowned_threads(
    saver: ThreeTierCheckpointSaver,
    executor: AsyncQueryExecutor,
    *,
    owned_threads_sql: str,
    batch: int = DEFAULT_BATCH,
    limit: int | None = None,
    dry_run: bool = False,
) -> CheckpointSweep:
    """delete every thread, in either table, that the app's owned-threads query does not name.

    Threads are found ``batch`` at a time in stored-id order and each batch is one
    :meth:`~ThreeTierCheckpointSaver.aprune` call with ``strategy="delete"``. The owned set is
    re-read for every page, so a thread the app creates during the run is not deleted.

    **An empty owned set is refused.** It would delete every row, which is what a query with a
    wrong schema or filter returns as readily as a genuinely empty app; an app that owns nothing
    and wants its tables emptied can say so with ``adelete_thread`` or ``TRUNCATE``.

    The owned-threads query is run as written, inside the statements here: it must be a single
    read-only SELECT, and it is trusted the way the database credential it runs under is.

    :param saver: an unscoped saver over the tables
    :ptype saver: ThreeTierCheckpointSaver
    :param executor: the database the saver writes to
    :ptype executor: AsyncQueryExecutor
    :param owned_threads_sql: a SELECT whose first column is the stored ids of the threads the app
        still owns
    :ptype owned_threads_sql: str
    :param batch: threads per ``aprune`` call
    :ptype batch: int
    :param limit: most threads to handle, or None for all
    :ptype limit: int | None
    :param dry_run: count without deleting
    :ptype dry_run: bool
    :return: the totals
    :rtype: CheckpointSweep
    :raises ValueError: when the saver is scoped, the query is blank or returns no rows, ``batch``
        is below 1 or ``limit`` is negative
    """
    _require_unscoped(saver, "delete_unowned_threads")
    _check_batch(batch, limit)
    owned = _owned_cte(owned_threads_sql)
    owned_count = await executor.fetchrow(f"{owned}SELECT count(*) AS owned FROM owned")
    if _int(owned_count, "owned") == 0:
        raise ValueError(
            "owned_threads_sql returned no rows, so every checkpoint would be deleted. Check the query "
            "(schema, filter); an app that owns no thread empties its tables some other way."
        )
    cursor = ""
    batches = threads = checkpoint_rows = write_rows = total_bytes = 0
    while (size := _page_size(batch, limit, threads)) > 0:
        rows = await executor.fetch(
            f"{owned}SELECT thread_id FROM ("
            f"(SELECT DISTINCT thread_id FROM checkpoints c WHERE thread_id > $1 AND {_not_owned('c')} "
            "ORDER BY thread_id LIMIT $2) UNION "
            f"(SELECT DISTINCT thread_id FROM checkpoint_writes w WHERE thread_id > $1 AND {_not_owned('w')} "
            "ORDER BY thread_id LIMIT $2)) page ORDER BY thread_id LIMIT $2",
            cursor,
            size,
        )
        if not rows:
            break
        page = [str(row["thread_id"]) for row in rows]
        cps, wrs, page_bytes = await _thread_rows(executor, page)
        if not dry_run:
            result: CheckpointPruneResult = await saver.aprune(page, strategy="delete")
            cps, wrs = result.checkpoints_deleted, result.writes_deleted
        batches += 1
        threads += len(page)
        checkpoint_rows += cps
        write_rows += wrs
        total_bytes += page_bytes
        cursor = page[-1]
        log.info(
            "checkpoint unowned-thread batch",
            extra={"dry_run": dry_run, "threads": len(page), "checkpoint_rows": cps, "write_rows": wrs},
        )
    return CheckpointSweep(
        strategy="delete",
        dry_run=dry_run,
        batches=batches,
        threads=threads,
        checkpoint_rows=checkpoint_rows,
        write_rows=write_rows,
        bytes=total_bytes,
        namespaces_held=0,
    )
