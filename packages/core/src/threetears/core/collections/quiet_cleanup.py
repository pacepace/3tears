"""quiet-time cleanup of versioned tables: older versions, "gone" markers and run markers below the restore window.

Append-only L3 (``design-append-only-l3.md``, decisions 2, 3 and 5; build plan chunk 6). A versioned table
(:func:`~threetears.core.ddl.row_epoch.row_versions`) keeps a version of a key a write, a delete appends a "gone"
marker and a whole run deleted is one run marker (:mod:`~threetears.core.ddl.run_markers`). Nothing a reader sees needs
the older versions: a read takes each key's newest. This module removes them, and is never relied on: the database is
sized for a burst night with no cleanup at all (Pace 2026-11-25).

**What it removes, and what it never touches.** Only versions stamped below the floor F, the lowest epoch a restore of a
lost change log could read (:func:`log_window_floor`, :func:`~threetears.core.ddl.row_epoch.cleanup_floor`): the window
is everything the NATS log still holds, and a restore reads every version at or above its own floor, at or above F. A
version below F is removed when a newer version of its key exists (it is superseded), and a "gone" marker below F is
removed together with every older version of its key. A key's newest live version is never removed. A run marker below
F is removed after every row of its run older than it; until then the marker stays, so no row of a gone run comes back.

WHY the window holds for superseded versions too: a restore folds each version it reads into the key and its declared
invalidation columns, and an older version's columns (a row that moved scope, a group it left) are what a follower
evicts by. Inside the window every version stays.

**In any order.** Each version cleanup visits removes what it supersedes: a live version, its key's versions older than
itself; a "gone" marker, its key's versions up to and including itself. Neither ever removes a version newer than the
one visited, so newest, oldest or middle first, no key comes back, and every removable version is eventually removed:
when F passes a key's newest version, cleanup visits it and clears all below. A run's rows go before its marker.

**Keyed, in small batches, a scope at a time.** Each versioned table carries a ``(scope, epoch, key...)`` index
(declared by ``TableSchema``). A batch is three keyed statements, Arrow end to end:

1. the scope's next page of versions in ``[cleaned_to, F)``, in (epoch, key) order past the last one visited, through
   that index (:func:`_candidates`);
2. every version below F of the keys the page names, by the primary key, as stored
   (``read_keyed_arrow(with_gone=True)``);
3. the versions to remove, worked out in Arrow on a thread (:func:`_removable`), deleted by their whole stored key
   (``delete_keyed``): point deletes, the cheapest a YugabyteDB delete gets (no read of the table to find the rows).

A run's rows go by the key prefix the marker names, a page a statement (``delete_before``). No statement reads a table
whole, scans it or truncates it (``TRUNCATE`` bumps YugabyteDB's catalog version cluster-wide), and nothing here is a
trigger. Each version is visited once: a scope's progress (:data:`~threetears.core.collections.cleanup_progress.
CLEANUP_PROGRESS_SCHEMA`) records the epoch below which every version was visited, so a round reads only what was
written since, and a round whose floor has not moved reads no data at all.

**Only while nothing loads, and never in a writer's way.** :class:`LoadGate` listens to the scope epochs table
(``ScopeEpochs.on_change``: this process's own writes and every peer's broadcast, no L3 read a change). A change heard
stops cleanup before its next batch, so it pauses within one batch of a load starting; only then does it read the
epochs (``ScopeEpochs.snapshot``, the rows that changed) to learn whether a write is in progress or a scope is pending.
It resumes on the next change heard once nothing is loading (a commit is one), from where it stopped. Nothing runs on a
timer. Cleanup deletes only rows stamped below F, which is at least ten minutes older than any transaction a writer may
still commit (``MAX_TRANSACTION_AGE_MS``), so it never contends for a row a writer writes, and it touches no scope
epoch, so it never holds up a ``begin``.

**Accounting** (metrics only; alerting is Grafana's): ``threetears_quiet_cleanup_removed_total`` by table and kind,
``threetears_quiet_cleanup_backlog_scopes`` by table (scopes not yet cleared to the floor), and
``threetears_quiet_cleanup_paused`` (1 while a load holds it) with ``threetears_quiet_cleanup_pauses_total``.

Needs the ``arrow`` extra (pyarrow at module top): the planner is Arrow's.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final, Protocol, runtime_checkable

import pyarrow as pa
import pyarrow.compute as pc

from threetears.core.backends.arrow_reply import l3_fetch_arrow
from threetears.core.backends.broker_generation import cleanup_deleting
from threetears.core.backends.dialect import known_on_yugabyte
from threetears.core.backends.schema_sql import key_array_type
from threetears.core.collections.cleanup_progress import CLEANUP_PROGRESS_TABLE
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    DATETIMETZ_TYPE,
    INT_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    TableSchema,
    rail_arrow_types,
)
from threetears.core.collections.scope_epochs import WHOLE, EpochSnapshot, ScopeEpochs
from threetears.core.ddl.row_epoch import EPOCH_COLUMN, GONE_COLUMN, cleanup_floor
from threetears.core.ddl.run_markers import RUN_MARKERS_TABLE
from threetears.core.keyed import Fragment, KeyBatch, Sort, delete_before, delete_keyed, keyset_rows, read_keyed_arrow
from threetears.core.sql_fragments import quote_identifier
from threetears.interfaces.keyed import next_past, page_statement
from threetears.interfaces.keyed.keys import index_hint, serving_index
from threetears.observe import get_logger
from threetears.observe.metrics import counter, gauge

__all__ = ["ChangeLogWindow", "CleanupPass", "LoadGate", "QuietCleanup", "log_window_floor"]

log = get_logger(__name__)

#: the versions a batch visits. WHY 500: measured on the local stack's YugabyteDB (2026-11-25,
#: test_quiet_cleanup_plans_live::test_what_a_cleanup_batch_costs, 100,000 superseded versions of 200,000 across 50
#: scopes): batch 100 cost 0.831 ms a version (16 ms a statement), 500 cost 0.296 ms (27 ms a statement, ~80 ms a
#: batch of three), 2000 cost 0.273 ms (54 ms a statement). Past 500 the saving is under 10 % while a load waits up to
#: four times longer for the batch in flight to end
_DEFAULT_BATCH: Final = 500

#: what a run marker's prefix joins its values with (:mod:`~threetears.core.ddl.run_markers`)
_PREFIX_SEPARATOR: Final = "\x1f"

_REMOVED = counter(
    "threetears_quiet_cleanup_removed_total",
    description="rows quiet-time cleanup removed from versioned tables: superseded versions and gone markers "
    "(version), a gone run's rows (run_row) and run markers (run_marker)",
    label_names=("table", "kind"),
)
_BACKLOG = gauge(
    "threetears_quiet_cleanup_backlog_scopes",
    description="scopes of a versioned table quiet-time cleanup has not yet cleared to the restore window's floor",
    label_names=("table",),
)
_PAUSED = gauge(
    "threetears_quiet_cleanup_paused",
    description="1 while quiet-time cleanup is held by a load (a write in progress or a scope pending), else 0",
)
_PAUSES = counter(
    "threetears_quiet_cleanup_pauses_total",
    description="times quiet-time cleanup stopped before a batch because a load started",
)


@runtime_checkable
class ChangeLogWindow(Protocol):
    """what :func:`log_window_floor` reads of the change log: an ``EpochGenerationSource`` satisfies it. The hub's
    broker reads the same window through it, to bound the floor a cleanup delete names (``CLEANUP_FLOOR_FIELD``)."""

    async def log_state(self) -> Any:
        """the log's state (``first_seq``, ``last_seq``), or None when absent."""

    async def epoch_at(self, seq: int) -> Any:
        """the record at ``seq`` (its ``s_min``), or None."""


async def log_window_floor(log_reader: ChangeLogWindow) -> int | None:
    """the restore window's floor F: :func:`~threetears.core.ddl.row_epoch.cleanup_floor` of the lowest stamp of the
    oldest record the change log holds. None when that cannot be known (no log, an empty one, or an oldest record that
    names no stamp): cleanup then removes nothing.

    :param log_reader: the change log (an ``EpochGenerationSource``)
    :ptype log_reader: ChangeLogWindow
    :return: the floor, or None
    :rtype: int | None
    """
    state = await log_reader.log_state()
    held = state is not None and state.last_seq >= state.first_seq
    oldest = await log_reader.epoch_at(state.first_seq) if held else None
    s_min = None if oldest is None else oldest.s_min
    return None if s_min is None else cleanup_floor(int(s_min))


class LoadGate:
    """whether anything is loading, from the scope epochs table: a write in progress, or a scope pending.

    Listens to ``ScopeEpochs.on_change`` (this process's own writes and every peer's broadcast): a change heard sets
    :attr:`stirred` at once, with no L3 read, and cleanup stops before its next batch. :meth:`look` then reads the
    epochs once (``ScopeEpochs.snapshot``, which reads only what changed) to learn whether the load is still on.

    :param epochs: the scope epochs of the tables cleaned
    :ptype epochs: ScopeEpochs
    """

    def __init__(self, epochs: ScopeEpochs) -> None:
        self._epochs = epochs
        self._heard = asyncio.Event()
        # the first look is owed: nothing is known yet
        self._heard.set()
        self._remove = epochs.on_change(self._heard.set)

    @property
    def stirred(self) -> bool:
        """whether an epoch changed since the last :meth:`look`: a load may have started.

        :return: whether one did
        :rtype: bool
        """
        return self._heard.is_set()

    async def heard(self) -> None:
        """wait until an epoch changes (returns at once when one already did since the last :meth:`look`).

        :return: nothing
        :rtype: None
        """
        await self._heard.wait()

    async def look(self) -> EpochSnapshot | None:
        """read the epochs once: the snapshot when nothing is loading, None while a write is in progress or a scope
        it touched is pending (one a commit withheld is not a load: it waits for its writer's next write).

        :return: the snapshot, or None while loading
        :rtype: EpochSnapshot | None
        """
        self._heard.clear()
        snapshot = await self._epochs.snapshot()
        loading = snapshot.writing is not None or bool(snapshot.pending - snapshot.withheld)
        return None if loading else snapshot

    def close(self) -> None:
        """stop listening.

        :return: nothing
        :rtype: None
        """
        self._remove()


@dataclass(frozen=True)
class CleanupPass:
    """what one :meth:`QuietCleanup.clean` did.

    :ivar removed: the rows it removed (versions, gone markers, a gone run's rows, run markers)
    :ivar paused: it stopped because a load started (or was on); it resumes on the next quiet look
    :ivar finished: every scope of every table is cleared to the floor
    :ivar floor: the restore window's floor it cleared to; None when unknown (it removed nothing)
    """

    removed: int
    paused: bool
    finished: bool
    floor: int | None


@dataclass(frozen=True)
class _Place:
    """where a paused scope resumes: its table, scope and floor, and the last version visited (its sort values)."""

    table: str
    scope: str
    floor: int
    past: tuple[Any, ...]


def _removable(candidates: pa.Table, versions: pa.Table, key: Sequence[str]) -> pa.Table:
    """the versions a page of visited versions removes, by their stored key (the key, then the epoch).

    Each visited version removes its key's versions older than itself, and a "gone" marker itself too: so a key's bound
    is the highest of its visited versions' ``epoch + gone``, and every version of the key below its bound goes. Never a
    version newer than one visited: whatever order the versions are visited in, no key's newest live version goes.

    Pure Arrow, on the caller's thread (cleanup runs it on a worker thread, never the loop).

    :param candidates: the visited versions: the key columns, ``epoch`` and ``gone``
    :ptype candidates: pyarrow.Table
    :param versions: the versions held of the keys ``candidates`` names: the key columns and ``epoch``
    :ptype versions: pyarrow.Table
    :param key: the key's columns
    :ptype key: Sequence[str]
    :return: the stored keys to delete: the key columns and ``epoch``
    :rtype: pyarrow.Table
    """
    names = list(key)
    gone = pc.cast(pc.fill_null(candidates.column(GONE_COLUMN), False), pa.int64())
    bounds = candidates.select(names).append_column("bound", pc.add(candidates.column(EPOCH_COLUMN), gone))
    highest = bounds.group_by(names).aggregate([("bound", "max")])
    joined = versions.select([*names, EPOCH_COLUMN]).join(highest, keys=names, join_type="inner")
    return joined.filter(pc.less(joined.column(EPOCH_COLUMN), joined.column("bound_max"))).select(
        [*names, EPOCH_COLUMN]
    )


def _decoded(kind: str, text: str) -> Any:
    """one value of a run marker's prefix, from the text the database wrote it as, typed as its column binds it.

    :param kind: the column's type tag
    :ptype kind: str
    :param text: its text
    :ptype text: str
    :return: the value
    :rtype: Any
    :raises ValueError: for a column type a run marker's prefix is not decoded from
    """
    if kind in (INT_TYPE, BIGINT_TYPE):
        return int(text)
    if kind == UUID_TYPE:
        return uuid.UUID(text)
    if kind == DATETIMETZ_TYPE:
        return datetime.fromisoformat(text)
    if kind == STRING_TYPE:
        return text
    raise ValueError(f"a run marker's prefix value of type {kind!r} is not decoded by cleanup")


class QuietCleanup:
    """removes versioned tables' older versions, gone markers and run markers below the restore window, only while
    nothing loads (the module's docstring).

    :param pool: the L3 the tables live in (a pool, or the rail's proxy)
    :ptype pool: Any
    :param tables: the versioned tables cleaned (``row_versions``), each registered with the store
    :ptype tables: Sequence[TableSchema]
    :param gate: the load signal
    :ptype gate: LoadGate
    :param floor: the restore window's floor F (:func:`log_window_floor` over the change log); None removes nothing
    :ptype floor: Callable[[], Awaitable[int | None]]
    :param scope_prefix: what the scope epochs table writes before a row's scope value (reports: ``state:``): the
        scopes cleaned are its scopes of that prefix
    :ptype scope_prefix: str
    :param batch: the versions a batch visits (and the most rows a run's delete removes a statement)
    :ptype batch: int
    :param progress: the cleanup progress table (:data:`~threetears.core.collections.cleanup_progress.
        CLEANUP_PROGRESS_SCHEMA`), in the tables' schema
    :ptype progress: str
    :param run_markers: the run-marker table in the tables' schema; None when they delete no whole runs
    :ptype run_markers: str | None
    :raises ValueError: for a table that is not versioned, or a batch under one
    """

    def __init__(
        self,
        pool: Any,
        tables: Sequence[TableSchema],
        gate: LoadGate,
        floor: Callable[[], Awaitable[int | None]],
        *,
        scope_prefix: str = "",
        batch: int = _DEFAULT_BATCH,
        progress: str = CLEANUP_PROGRESS_TABLE,
        run_markers: str | None = RUN_MARKERS_TABLE,
    ) -> None:
        refused = [schema.name for schema in tables if not schema.versioned]
        if refused:
            raise ValueError(f"quiet-time cleanup clears versioned tables (row_versions); not {refused}")
        if batch < 1:
            raise ValueError(f"a cleanup batch visits at least one version; got {batch}")
        self._pool = pool
        self._tables = tuple(tables)
        self._gate = gate
        self._floor = floor
        self._prefix = scope_prefix
        self._batch = batch
        self._progress = progress
        self._run_markers = run_markers
        #: where a paused scope resumes; replaced whole
        self._place: _Place | None = None

    async def run(self) -> None:
        """clean whenever an epoch change is heard and nothing is loading; forever, until cancelled.

        :return: never
        :rtype: None
        """
        while True:
            await self._gate.heard()
            await self.clean()

    async def clean(self) -> CleanupPass:
        """one pass: when nothing is loading, clear every scope of every table to the floor, stopping before the next
        batch the moment a load starts.

        :return: what it did
        :rtype: CleanupPass
        """
        snapshot = await self._gate.look()
        floor = None if snapshot is None else await self._floor()
        if snapshot is None:
            done = self._paused(0, None)
        elif floor is None:
            # NOSILENT: the window is unknown (no log, or its oldest record names no stamp): nothing is removed
            _PAUSED.set(0)
            done = CleanupPass(removed=0, paused=False, finished=False, floor=None)
        else:
            done = await self._clean_to(snapshot, floor)
        return done

    async def _clean_to(self, snapshot: EpochSnapshot, floor: int) -> CleanupPass:
        """clear every scope ``snapshot`` names of every table to ``floor``, a scope at a time, stopping when a load
        starts.

        :param snapshot: the scope epochs, read while nothing was loading
        :ptype snapshot: EpochSnapshot
        :param floor: F
        :ptype floor: int
        :return: what it did
        :rtype: CleanupPass
        """
        scopes = sorted(
            scope.removeprefix(self._prefix)
            for scope in snapshot.epochs
            if scope != WHOLE and scope.startswith(self._prefix)
        )
        removed = 0
        stopped = False
        for schema in self._tables:
            cleaned = await self._cleaned_to(schema.name)
            behind = [value for value in scopes if cleaned.get(value, 0) < floor]
            _BACKLOG.labels(table=schema.name).set(len(behind))
            for left, value in enumerate(behind):
                finished, count = await self._clean_scope(schema, value, cleaned.get(value, 0), floor)
                removed += count
                stopped = not finished
                if stopped:
                    break
                await self._record(schema.name, value, floor)
                _BACKLOG.labels(table=schema.name).set(len(behind) - left - 1)
            if stopped:
                break
        if not stopped:
            _PAUSED.set(0)
            log.info(
                "quiet-time cleanup cleared every scope to the restore window",
                extra={"extra_data": {"floor": floor, "removed": removed}},
            )
        return self._paused(removed, floor) if stopped else CleanupPass(removed, False, True, floor)

    def _paused(self, removed: int, floor: int | None) -> CleanupPass:
        """a pass that stopped for a load: counted, and the paused gauge raised until a pass finishes.

        :param removed: what it removed before stopping
        :ptype removed: int
        :param floor: the floor it cleared to, None when it looked no further than the load
        :ptype floor: int | None
        :return: the pass
        :rtype: CleanupPass
        """
        _PAUSED.set(1)
        _PAUSES.inc()
        return CleanupPass(removed=removed, paused=True, finished=False, floor=floor)

    async def _cleaned_to(self, table: str) -> dict[str, int]:
        """each scope's progress for ``table``: the epoch below which every version was visited, read by the
        progress table's key (``table_name``, one row a scope).

        :param table: the versioned table
        :ptype table: str
        :return: scope value -> its progress, function-local to the caller
        :rtype: dict[str, int]
        """
        rows = await keyset_rows(
            self._pool,
            self._progress,
            columns=("scope", "cleaned_to"),
            by={"table_name": table},
            order=(Sort("scope"),),
        )
        return {str(row["scope"]): int(row["cleaned_to"]) for row in rows}

    async def _record(self, table: str, scope: str, floor: int) -> None:
        """record that every version of ``scope`` below ``floor`` was visited: one upsert by the progress key, never
        moving it back.

        :param table: the versioned table
        :ptype table: str
        :param scope: the scope value
        :ptype scope: str
        :param floor: the floor reached
        :ptype floor: int
        :return: nothing
        :rtype: None
        """
        progress = quote_identifier(self._progress)
        await self._pool.execute(
            f"INSERT INTO {progress} (table_name, scope, cleaned_to) VALUES ($1, $2, $3) "  # noqa: S608 - trusted identifier
            f"ON CONFLICT (table_name, scope) DO UPDATE SET cleaned_to = GREATEST({progress}.cleaned_to, "
            "EXCLUDED.cleaned_to)",
            table,
            scope,
            floor,
        )

    async def _clean_scope(self, schema: TableSchema, value: str, cleaned_to: int, floor: int) -> tuple[bool, int]:
        """clear one scope of one table to ``floor``: its versions in ``[cleaned_to, floor)`` a batch at a time, then
        its run markers below ``floor``; stopping before a batch when a load started.

        :param schema: the table
        :ptype schema: TableSchema
        :param value: the scope's value of the table's scope column
        :ptype value: str
        :param cleaned_to: the scope's progress
        :ptype cleaned_to: int
        :param floor: F
        :ptype floor: int
        :return: whether it finished, and the rows it removed
        :rtype: tuple[bool, int]
        """
        held = self._place
        past = (
            held.past
            if held is not None and (held.table, held.scope, held.floor) == (schema.name, value, floor)
            else None
        )
        self._place = None
        removed = 0
        paged = False
        while not (paged or self._gate.stirred):
            candidates = await self._candidates(schema, value, cleaned_to, floor, past)
            removed += await self._clear(schema, candidates, floor)
            paged = candidates.num_rows < self._batch
            if not paged:
                last = {name: candidates.column(name)[-1].as_py() for name in candidates.column_names}
                past = tuple(next_past(self._order(schema), last))
        if not paged and past is not None:
            # a load started before the next batch: the scope resumes after the last version visited
            self._place = _Place(schema.name, value, floor, past)
        finished, runs = await self._clear_runs(schema, value, floor) if paged else (False, 0)
        return finished, removed + runs

    def _order(self, schema: TableSchema) -> tuple[Sort, ...]:
        """the order a scope's versions are visited in: the epoch, then the key (the scope column pinned, left out).

        :param schema: the table
        :ptype schema: TableSchema
        :return: the sort
        :rtype: tuple[Sort, ...]
        """
        scope = schema.row_epoch.scope if schema.row_epoch is not None else None
        keyed = [column for column in schema.pk_columns if column != scope]
        return (Sort(EPOCH_COLUMN, cast="bigint"), *(Sort(column, cast=self._cast(schema, column)) for column in keyed))

    @staticmethod
    def _cast(schema: TableSchema, column: str) -> str:
        """a key column's SQL type, its text cast back to it.

        :param schema: the table
        :ptype schema: TableSchema
        :param column: a key column
        :ptype column: str
        :return: the type
        :rtype: str
        :raises ValueError: for a key column with no type a key is bound as
        """
        kind = key_array_type(schema, column)
        if kind is None:
            raise ValueError(f"{schema.name}: key column {column!r} has no type cleanup can bind a key by")
        return kind.removesuffix("[]")

    async def _candidates(
        self, schema: TableSchema, value: str, cleaned_to: int, floor: int, past: tuple[Any, ...] | None
    ) -> pa.Table:
        """the scope's next page of versions stamped in ``[cleaned_to, floor)``, past ``past`` in (epoch, key) order:
        one range of the table's ``(scope, epoch, key...)`` index, as Arrow.

        :param schema: the table
        :ptype schema: TableSchema
        :param value: the scope value
        :ptype value: str
        :param cleaned_to: the scope's progress
        :ptype cleaned_to: int
        :param floor: F
        :ptype floor: int
        :param past: the last version visited, its sort values; the first page when None
        :ptype past: tuple[Any, ...] | None
        :return: the key columns, ``epoch`` and ``gone`` of at most a batch of versions
        :rtype: pyarrow.Table
        """
        assert schema.row_epoch is not None and schema.row_epoch.scope is not None  # noqa: S101 -- versioned
        by = {schema.row_epoch.scope: value}
        order = self._order(schema)
        yugabyte = await known_on_yugabyte(self._pool)
        hint = index_hint(serving_index(schema.name, by, order)) if yugabyte else ""
        columns = (*schema.pk_columns, EPOCH_COLUMN, GONE_COLUMN)
        sql, params = page_statement(
            schema.name,
            columns,
            by,
            order,
            self._batch,
            None if past is None else list(past),
            not yugabyte,
            (),
            (Fragment(f't."{EPOCH_COLUMN}" >= $1', (cleaned_to,)), Fragment(f't."{EPOCH_COLUMN}" < $1', (floor,))),
            None,
            seek_legs=yugabyte,
            hint=hint,
        )
        types = rail_arrow_types([schema.column(name) for name in columns])
        return await l3_fetch_arrow(self._pool, sql, *params, types=types)

    async def _clear(self, schema: TableSchema, candidates: pa.Table, floor: int) -> int:
        """remove what a page of visited versions supersedes: every version below ``floor`` of the keys it names is
        read by the primary key, the versions to remove worked out in Arrow on a thread, and deleted by stored key.

        :param schema: the table
        :ptype schema: TableSchema
        :param candidates: the visited versions
        :ptype candidates: pyarrow.Table
        :param floor: F
        :ptype floor: int
        :return: the versions removed
        :rtype: int
        """
        if not candidates.num_rows:
            return 0
        key = schema.pk_columns
        casts = tuple(self._cast(schema, column) for column in key)
        named = KeyBatch(columns=key, keys=[candidates.column(c).combine_chunks() for c in key], casts=casts)
        types = rail_arrow_types([schema.column(name) for name in (*key, EPOCH_COLUMN)])
        held = await read_keyed_arrow(
            self._pool,
            schema.name,
            keys=named,
            types=types,
            where=[Fragment(f't."{EPOCH_COLUMN}" < $1', (floor,))],
            # the versions themselves, as stored: a shared read would answer each key's newest live version alone
            with_gone=True,
        )
        versions = pa.concat_tables(held) if held else pa.schema(list(types.items())).empty_table()
        doomed = await asyncio.to_thread(_removable, candidates, versions, key)
        stored = (*key, EPOCH_COLUMN)
        count = 0
        if doomed.num_rows:
            # marked cleanup's with its floor (CLEANUP_DELETE_FIELD), and bounded by it in the statement itself: the
            # broker takes a delete of a versioned table only so, and only for a floor no higher than its own reading
            # of the log's window, so no version at or above F can go whatever keys a delete names
            with cleanup_deleting(floor):
                written = await delete_keyed(
                    self._pool,
                    schema.name,
                    keys=KeyBatch(
                        columns=stored,
                        keys=[doomed.column(c).combine_chunks() for c in stored],
                        casts=(*casts, "bigint"),
                    ),
                    where=[Fragment(f'"{EPOCH_COLUMN}" < $1', (floor,))],
                )
            count = written.count
            _REMOVED.labels(table=schema.name, kind="version").inc(count)
        return count

    async def _clear_runs(self, schema: TableSchema, value: str, floor: int) -> tuple[bool, int]:
        """remove the scope's run markers below ``floor``, each after every row of its run older than it: the rows by
        the key prefix the marker names, a page a statement, stopping before a page when a load started.

        :param schema: the table
        :ptype schema: TableSchema
        :param value: the scope value
        :ptype value: str
        :param floor: F
        :ptype floor: int
        :return: whether it finished, and the rows removed (run rows and markers)
        :rtype: tuple[bool, int]
        """
        if self._run_markers is None:
            return True, 0
        markers = await keyset_rows(
            self._pool,
            self._run_markers,
            columns=("prefix", EPOCH_COLUMN),
            by={"table_name": schema.name, "scope": value},
            order=(Sort("prefix"), Sort(EPOCH_COLUMN, cast="bigint")),
            where=[Fragment(f't."{EPOCH_COLUMN}" < $1', (floor,))],
        )
        removed = 0
        stopped = False
        for marker in markers:
            text, epoch = str(marker["prefix"]), int(marker[EPOCH_COLUMN])
            prefix = self._prefix_of(schema, text)
            swept_all = False
            while not (swept_all or self._gate.stirred):
                # marked cleanup's with its floor: the broker takes a run's rows only below a marker under F
                with cleanup_deleting(floor):
                    swept = await delete_before(
                        schema.name,
                        collection=None,
                        key=schema.stored_key,
                        column=EPOCH_COLUMN,
                        before=epoch,
                        page=self._batch,
                        by=prefix,
                        most_pages=1,
                        conn=self._pool,
                    )
                removed += swept.deleted
                _REMOVED.labels(table=schema.name, kind="run_row").inc(swept.deleted)
                swept_all = swept.deleted < self._batch
            stopped = not swept_all
            if stopped:
                # a load started: the marker stays until every row of its run is gone, so none comes back
                break
            gone = await delete_keyed(
                self._pool,
                self._run_markers,
                by={"table_name": schema.name, "scope": value, "prefix": text, EPOCH_COLUMN: epoch},
            )
            removed += gone.count
            _REMOVED.labels(table=schema.name, kind="run_marker").inc(gone.count)
        return not stopped, removed

    @staticmethod
    def _prefix_of(schema: TableSchema, text: str) -> Mapping[str, Any]:
        """a run marker's prefix as its key columns' values: its text split by the separator, each typed.

        :param schema: the table
        :ptype schema: TableSchema
        :param text: the marker's prefix
        :ptype text: str
        :return: key column -> value, the key's leading columns
        :rtype: Mapping[str, Any]
        :raises ValueError: when the prefix names more values than the key has columns
        """
        parts = text.split(_PREFIX_SEPARATOR)
        if len(parts) > len(schema.pk_columns):
            raise ValueError(f"{schema.name}: run marker prefix {text!r} names more values than its key's columns")
        return {
            name: _decoded(schema.column(name).column_type, part)
            for name, part in zip(schema.pk_columns, parts, strict=False)
        }
