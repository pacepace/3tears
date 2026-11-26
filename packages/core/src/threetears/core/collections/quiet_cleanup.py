"""quiet-time cleanup below the restore window: versioned tables' older versions, "gone" markers and run markers, and
stamped tables' gone rows.

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
A table's reset marker (``TABLE_RESET_SCOPE``: written before a ``TRUNCATE``, or alone in a transaction by
``SchemaBackedCollection.reset_table``) hides every version of the table older than itself; below F, those versions are
removed (a scope's walk then starts at the oldest version, each version below the marker removing every version of its
key below it), and the marker after them, by its key: no restore reads it once its rows are gone.

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

**Stamped tables too.** A stamped table (``ROW_STAMPED``: one row a key, updated in place) keeps no older versions;
its delete marks its row gone in place. Its gone rows below F go by
:func:`~threetears.core.collections.stamped_cleanup.clear_stamped_gone`, a bucket at a time through its
``(epoch_bucket, epoch)`` index, after the versioned tables, its progress one row under :data:`STAMPED_SCOPE`; a
stamped table a write moves between tenants has its moves below F deleted with them
(:func:`~threetears.core.collections.tenant_moves.clear_moves`, one range of the moves table's key a bucket).

**Only while nothing loads, and never in a writer's way.** :class:`LoadGate` is one gate with two feeds. Over the
scope epochs table (reports) it listens to ``ScopeEpochs.on_change`` (this process's own writes and every peer's
broadcast, no L3 read a change): a change heard stops cleanup before its next batch, so it pauses within one batch of a
load starting; only then does it read the epochs (``ScopeEpochs.snapshot``, the rows that changed) to learn whether a
write is in progress or a scope is pending, and it resumes on the next change heard once nothing is loading (a commit is
one). A database with no scope epochs (the hub's) feeds it the change log's own records as this process hears them
(:meth:`LoadGate.from_change_log`): a record heard pauses cleanup the same way, and since the last record of a load is
followed by silence, not by a change to hear, quiet is one timer armed on each record heard (cancelled and armed
again), never a periodic read; when it fires cleanup resumes. Either way it resumes from where it stopped. Cleanup
deletes only rows stamped below F, which is at least ten minutes older than any transaction a writer may still commit
(``MAX_TRANSACTION_AGE_MS``), so it never contends for a row a writer writes, and it touches no scope epoch, so it never
holds up a ``begin``.

**Accounting** (metrics only; alerting is Grafana's): ``threetears_quiet_cleanup_removed_total`` by table and kind,
``threetears_quiet_cleanup_referenced_total`` by table (a stamped table's gone rows left a round because another row
still references them, :mod:`~threetears.core.collections.stamped_cleanup`),
``threetears_quiet_cleanup_backlog_scopes`` by table (scopes not yet cleared to the floor), and
``threetears_quiet_cleanup_paused`` (1 while a load holds it) with ``threetears_quiet_cleanup_pauses_total``.

Needs the ``arrow`` extra (pyarrow at module top): the planner is Arrow's.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from threetears.core.arrow_keys import key_codes
from threetears.core.backends.arrow_reply import l3_fetch_arrow
from threetears.core.backends.broker_generation import cleanup_deleting
from threetears.core.backends.dialect import known_on_yugabyte
from threetears.core.backends.schema_sql import key_array_type
from threetears.core.collections.cleanup_progress import CLEANUP_PROGRESS_TABLE
from threetears.core.collections.epoch_change import EpochChange
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
from threetears.core.collections.stamped_cleanup import StampedSweep, clear_stamped_gone
from threetears.core.collections.tenant_moves import clear_moves
from threetears.core.ddl.row_epoch import EPOCH_COLUMN, GONE_COLUMN, cleanup_floor
from threetears.core.ddl.run_markers import RUN_MARKERS_TABLE, TABLE_RESET_SCOPE
from threetears.core.keyed import Fragment, KeyBatch, Sort, delete_before, delete_keyed, keyset_rows, read_keyed_arrow
from threetears.core.sql_fragments import quote_identifier
from threetears.interfaces.keyed import next_past, page_statement
from threetears.interfaces.keyed.keys import index_hint, serving_index
from threetears.observe import get_logger
from threetears.observe.metrics import counter, gauge

__all__ = [
    "ChangeLogWindow",
    "CleanupPass",
    "LoadGate",
    "QuietCleanup",
    "log_window_floor",
]

log = get_logger(__name__)

#: the versions a batch visits. WHY 500: measured on the local stack's YugabyteDB (2026-11-25,
#: test_quiet_cleanup_plans_live::test_what_a_cleanup_batch_costs, 100,000 superseded versions of 200,000 across 50
#: scopes): batch 100 cost 0.831 ms a version (16 ms a statement), 500 cost 0.296 ms (27 ms a statement, ~80 ms a
#: batch of three), 2000 cost 0.273 ms (54 ms a statement). Past 500 the saving is under 10 % while a load waits up to
#: four times longer for the batch in flight to end
_DEFAULT_BATCH: Final = 500

#: the progress scope a stamped table's cleanup is recorded under (``threetears_cleanup_progress``): its gone rows are
#: cleared a bucket at a time across the whole table, so it has one progress, never one a scope
STAMPED_SCOPE: Final = "*"

#: what a run marker's prefix joins its values with (:mod:`~threetears.core.ddl.run_markers`)
_PREFIX_SEPARATOR: Final = "\x1f"

_REMOVED = counter(
    "threetears_quiet_cleanup_removed_total",
    description="rows quiet-time cleanup removed: a versioned table's superseded versions and gone markers "
    "(version), a gone run's rows (run_row) and run markers (run_marker); a stamped table's gone rows (stamped_gone); "
    "a table's reset markers (reset_marker)",
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


#: how long the change log must stay silent before a gate fed by it (:meth:`LoadGate.from_change_log`) finds nothing
#: loading. A choice, not a measurement: cleanup deletes only rows stamped below F, never a row a writer writes, so a
#: resume between two commits of one load costs at most one batch beside it (~80 ms, measured at :data:`_DEFAULT_BATCH`)
#: before the next record pauses it again; 30 s keeps it from waking between the commits of a load that writes steadily
_QUIET_AFTER_SECONDS: Final = 30.0


class _EpochsFeed:
    """the gate's feed from the scope epochs table (reports): a change heard stirs it, and a look reads the epochs.

    :param epochs: the scope epochs of the tables cleaned
    :ptype epochs: ScopeEpochs
    """

    names_scopes: Final = True

    def __init__(self, epochs: ScopeEpochs) -> None:
        self._epochs = epochs

    def listen(self, stir: Callable[[], None]) -> Callable[[], None]:
        """call ``stir`` on every change heard (this process's own writes and every peer's broadcast).

        :param stir: the gate's stir
        :ptype stir: Callable[[], None]
        :return: a call that stops listening
        :rtype: Callable[[], None]
        """
        return self._epochs.on_change(stir)

    async def quiet(self) -> frozenset[str] | None:
        """read the epochs once (``ScopeEpochs.snapshot``, the rows that changed): their scopes when nothing is
        loading, None while a write is in progress or a scope it touched is pending (one a commit withheld is not a
        load: it waits for its writer's next write).

        :return: the scopes, or None while loading
        :rtype: frozenset[str] | None
        """
        snapshot: EpochSnapshot = await self._epochs.snapshot()
        loading = snapshot.writing is not None or bool(snapshot.pending - snapshot.withheld)
        return None if loading else frozenset(snapshot.epochs)


class _ChangeLogFeed:
    """the gate's feed from the change log's own records as this process hears them (a database with no scope epochs:
    the hub). A record heard stirs the gate and (re)arms ONE timer, cancelled and armed again on every record; the gate
    is quiet while no timer is armed, and the timer firing stirs it once more, so cleanup resumes then. Nothing is read
    from L3 to learn whether a load is on: the records are the load.

    Cleanup's own deletes are no load: their records say so (``EpochChange.cleanup_floor``, from the delete's
    ``CLEANUP_FLOOR_FIELD``) and are skipped, so cleanup does not pause itself after every batch it sends.

    :param on_record: subscribes a listener to the records this process hears (called on the loop with the record,
        an ``EpochChange``, which the gate reads only for whether it is cleanup's own), answering a call that removes it
    :ptype on_record: Callable[[Callable[..., None]], Callable[[], None]]
    :param quiet_after: seconds of silence after the last record heard before the gate is quiet
    :ptype quiet_after: float
    """

    names_scopes: Final = False

    def __init__(self, on_record: Callable[[Callable[..., None]], Callable[[], None]], quiet_after: float) -> None:
        if quiet_after <= 0:
            raise ValueError(f"a change-log gate waits some silence before it is quiet; got {quiet_after} s")
        self._on_record = on_record
        self._quiet_after = quiet_after
        #: the one timer, armed while the log is not yet quiet; replaced whole, owned by the loop
        self._timer: asyncio.TimerHandle | None = None
        self._stir: Callable[[], None] = lambda: None

    def listen(self, stir: Callable[[], None]) -> Callable[[], None]:
        """call ``stir`` on every record heard, and once more when the log has been quiet for ``quiet_after``.

        :param stir: the gate's stir
        :ptype stir: Callable[[], None]
        :return: a call that stops listening (and disarms the timer)
        :rtype: Callable[[], None]
        """
        self._stir = stir
        remove = self._on_record(self._heard)

        def stop() -> None:
            remove()
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

        return stop

    def _heard(self, *record: Any) -> None:
        """a record heard: stir the gate at once (cleanup stops before its next batch) and re-arm the quiet timer;
        nothing for a record of cleanup's own delete.

        :param record: the record
        :ptype record: Any
        :return: nothing
        :rtype: None
        """
        if record and all(isinstance(each, EpochChange) and each.cleanup_floor is not None for each in record):
            return
        self._stir()
        if self._timer is not None:
            self._timer.cancel()
        self._timer = asyncio.get_running_loop().call_later(self._quiet_after, self._fell_quiet)

    def _fell_quiet(self) -> None:
        """the timer fired: no record for ``quiet_after``. The gate is quiet, and is stirred so cleanup looks.

        :return: nothing
        :rtype: None
        """
        self._timer = None
        self._stir()

    async def quiet(self) -> frozenset[str] | None:
        """no scopes (the log names none) when no record was heard for ``quiet_after``, else None. No read.

        :return: an empty set, or None while loading
        :rtype: frozenset[str] | None
        """
        return None if self._timer is not None else frozenset()


class LoadGate:
    """whether anything is loading: one gate, fed by the scope epochs table (reports, :class:`LoadGate`'s own
    constructor) or by the change log's records as this process hears them (a database with no scope epochs, such as
    the hub's: :meth:`from_change_log`).

    Either feed stirs it at once, with no L3 read: :attr:`stirred` is set and cleanup stops before its next batch.
    :meth:`look` then asks the feed whether the load is still on: the scope epochs feed reads the epochs once
    (``ScopeEpochs.snapshot``, which reads only what changed) and is quiet on the next change heard once nothing is
    loading (a commit is one); the change-log feed reads nothing, and is quiet once no record was heard for its
    ``quiet_after`` (one timer, re-armed on each record, never a periodic read).

    :param epochs: the scope epochs of the tables cleaned (or the change log's feed :meth:`from_change_log` makes)
    :ptype epochs: ScopeEpochs
    """

    def __init__(self, epochs: ScopeEpochs | _ChangeLogFeed) -> None:
        self._feed = epochs if isinstance(epochs, _ChangeLogFeed) else _EpochsFeed(epochs)
        self._heard = asyncio.Event()
        # the first look is owed: nothing is known yet
        self._heard.set()
        self._remove = self._feed.listen(self._heard.set)

    @classmethod
    def from_change_log(
        cls,
        on_record: Callable[[Callable[..., None]], Callable[[], None]],
        *,
        quiet_after: float = _QUIET_AFTER_SECONDS,
    ) -> LoadGate:
        """a gate fed by the change log's records as this process hears them: every record heard pauses cleanup
        before its next batch, and cleanup resumes from where it stopped once the log was silent for
        ``quiet_after``. It names no scopes, so it cleans stamped tables (a versioned table is cleaned a scope at a
        time, from a scope epochs table: :class:`QuietCleanup` refuses one with this gate).

        :param on_record: subscribes a listener to the records this process hears (on the loop; the listener takes the
            record, unread), answering a call that removes it
        :ptype on_record: Callable[[Callable[..., None]], Callable[[], None]]
        :param quiet_after: seconds of silence before the gate is quiet (:data:`_QUIET_AFTER_SECONDS`)
        :ptype quiet_after: float
        :return: the gate
        :rtype: LoadGate
        :raises ValueError: for a silence of no time
        """
        return cls(_ChangeLogFeed(on_record, quiet_after))

    @property
    def names_scopes(self) -> bool:
        """whether a quiet look names the scopes to clean (the scope epochs feed), so versioned tables can be cleaned.

        :return: whether it does
        :rtype: bool
        """
        return self._feed.names_scopes

    @property
    def stirred(self) -> bool:
        """whether the feed stirred since the last :meth:`look`: a load may have started.

        :return: whether it did
        :rtype: bool
        """
        return self._heard.is_set()

    async def heard(self) -> None:
        """wait until the feed stirs (returns at once when it already did since the last :meth:`look`).

        :return: nothing
        :rtype: None
        """
        await self._heard.wait()

    async def look(self) -> frozenset[str] | None:
        """ask the feed once whether anything is loading: the scopes to clean when nothing is (the scope epochs table's,
        their prefix on; none from the change log), None while a load is on.

        :return: the scopes, or None while loading
        :rtype: frozenset[str] | None
        """
        self._heard.clear()
        return await self._feed.quiet()

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


def _removable(candidates: pa.Table, versions: pa.Table, key: Sequence[str], reset: int | None = None) -> pa.Table:
    """the versions a page of visited versions removes, by their stored key (the key, then the epoch).

    Each visited version removes its key's versions older than itself, and a "gone" marker itself too: so a key's bound
    is the highest of its visited versions' ``epoch + gone``, and every version of the key below its bound goes. Never a
    version newer than one visited: whatever order the versions are visited in, no key's newest live version goes.

    A table reset below the floor (``reset``, its newest reset marker's epoch) hides every version older than the
    marker, so each visited key's bound is at least the marker's epoch: every version of it below the marker goes, the
    visited one too, and none at or above it (a row its reset's transaction appended after the marker stays).

    Pure Arrow, on the caller's thread (cleanup runs it on a worker thread, never the loop).

    :param candidates: the visited versions: the key columns, ``epoch`` and ``gone``
    :ptype candidates: pyarrow.Table
    :param versions: the versions held of the keys ``candidates`` names: the key columns and ``epoch``
    :ptype versions: pyarrow.Table
    :param key: the key's columns
    :ptype key: Sequence[str]
    :param reset: the newest reset marker's epoch below the floor; None when the table holds none
    :ptype reset: int | None
    :return: the stored keys to delete: the key columns and ``epoch``
    :rtype: pyarrow.Table
    """
    names = list(key)
    stored = versions.select([*names, EPOCH_COLUMN])
    if not candidates.num_rows or not versions.num_rows:
        return stored.slice(0, 0)
    gone = pc.cast(pc.fill_null(candidates.column(GONE_COLUMN), False), pa.int64())
    bound = np.asarray(pc.add(candidates.column(EPOCH_COLUMN), gone).to_numpy(), dtype=np.int64)
    if reset is not None:
        # every version below the reset marker is hidden by it, the visited one too
        bound = np.maximum(bound, np.int64(reset))
    # each key coded by Arrow's hash kernels over both sides together, never an Acero plan (test_no_acero_tables)
    visited, held = key_codes([candidates.select(names), versions.select(names)], names)
    distinct, group = np.unique(visited, return_inverse=True)
    highest = np.full(len(distinct), np.iinfo(np.int64).min, dtype=np.int64)
    np.maximum.at(highest, group, bound)
    at = np.minimum(np.searchsorted(distinct, held), len(distinct) - 1)
    epochs = np.asarray(versions.column(EPOCH_COLUMN).to_numpy(), dtype=np.int64)
    removed = (distinct[at] == held) & (epochs < highest[at])
    return stored.take(pa.array(np.flatnonzero(removed), pa.int64()))


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
    """removes versioned tables' older versions, gone markers and run markers, and stamped tables' gone rows, below the
    restore window, only while nothing loads (the module's docstring).

    :param pool: the L3 the tables live in (a pool, or the rail's proxy)
    :ptype pool: Any
    :param tables: the tables cleaned, each registered with the store: versioned ones (``row_versions``) a scope at a
        time, stamped ones (``ROW_STAMPED``) a bucket at a time by
        :func:`~threetears.core.collections.stamped_cleanup.clear_stamped_gone`, their progress under the scope
        :data:`STAMPED_SCOPE`
    :ptype tables: Sequence[TableSchema]
    :param gate: the load signal: over the tables' scope epochs, or fed by the change log
        (:meth:`LoadGate.from_change_log`, stamped tables only)
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
    :raises ValueError: for a table that declares no row epoch, a versioned table with a gate that names no scopes,
        or a batch under one
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
        refused = [schema.name for schema in tables if schema.row_epoch is None]
        if refused:
            raise ValueError(
                f"quiet-time cleanup clears versioned (row_versions) and stamped (ROW_STAMPED) tables; not {refused}"
            )
        unscoped = [schema.name for schema in tables if schema.versioned and not gate.names_scopes]
        if unscoped:
            raise ValueError(
                f"quiet-time cleanup clears a versioned table a scope at a time, from the scope epochs table; a gate "
                f"fed by the change log (LoadGate.from_change_log) names no scopes, so it cannot clear {unscoped}. "
                f"Give it a LoadGate over the tables' ScopeEpochs"
            )
        if batch < 1:
            raise ValueError(f"a cleanup batch visits at least one version; got {batch}")
        self._pool = pool
        self._tables = tuple(schema for schema in tables if schema.versioned)
        self._stamped = tuple(schema for schema in tables if not schema.versioned)
        self._gate = gate
        self._floor = floor
        self._prefix = scope_prefix
        self._batch = batch
        self._progress = progress
        self._run_markers = run_markers
        #: where a paused scope resumes; replaced whole
        self._place: _Place | None = None

    async def run(self) -> None:
        """clean whenever the gate stirs (an epoch change or a record heard, or the log falling quiet) and nothing is
        loading; forever, until cancelled.

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
        quiet = await self._gate.look()
        floor = None if quiet is None else await self._floor()
        if quiet is None:
            done = self._paused(0, None)
        elif floor is None:
            # NOSILENT: the window is unknown (no log, or its oldest record names no stamp): nothing is removed
            _PAUSED.set(0)
            done = CleanupPass(removed=0, paused=False, finished=False, floor=None)
        else:
            done = await self._clean_to(quiet, floor)
        return done

    async def _clean_to(self, quiet: frozenset[str], floor: int) -> CleanupPass:
        """clear every scope ``quiet`` names of every table to ``floor``, a scope at a time, stopping when a load
        starts; then each table's reset markers below ``floor``.

        :param quiet: the scopes the gate named, looked at while nothing was loading (none from the change log)
        :ptype quiet: frozenset[str]
        :param floor: F
        :ptype floor: int
        :return: what it did
        :rtype: CleanupPass
        """
        scopes = sorted(
            scope.removeprefix(self._prefix) for scope in quiet if scope != WHOLE and scope.startswith(self._prefix)
        )
        removed = 0
        stopped = False
        for schema in self._tables:
            cleaned = await self._cleaned_to(schema.name)
            # the reset markers below F not yet cleared: the newest hides every version older than itself, which each
            # scope's walk then removes from its oldest version on, before the markers go
            resets = await self._resets_below(schema.name, cleaned.get(TABLE_RESET_SCOPE, 0), floor)
            reset = max((epoch for _, epoch in resets), default=None)
            behind = [value for value in scopes if cleaned.get(value, 0) < floor]
            _BACKLOG.labels(table=schema.name).set(len(behind))
            for left, value in enumerate(behind):
                since = 0 if reset is not None else cleaned.get(value, 0)
                finished, count = await self._clean_scope(schema, value, since, floor, reset)
                removed += count
                stopped = not finished
                if stopped:
                    break
                await self._record(schema.name, value, floor)
                _BACKLOG.labels(table=schema.name).set(len(behind) - left - 1)
            if stopped:
                break
            # every scope's versions below the newest marker are gone: the markers go after them
            removed += await self._clear_resets(schema.name, resets, floor)
        unfinished = False
        if not stopped:
            stopped, unfinished, count = await self._clean_stamped(floor)
            removed += count
        if not stopped:
            _PAUSED.set(0)
            log.info(
                "quiet-time cleanup cleared every scope to the restore window",
                extra={"extra_data": {"floor": floor, "removed": removed, "unfinished": unfinished}},
            )
        return self._paused(removed, floor) if stopped else CleanupPass(removed, False, not unfinished, floor)

    async def _clean_stamped(self, floor: int) -> tuple[bool, bool, int]:
        """clear every stamped table's gone rows to ``floor``, a bucket at a time
        (:func:`~threetears.core.collections.stamped_cleanup.clear_stamped_gone`), each table's progress one row of
        the progress table under :data:`STAMPED_SCOPE`; stopping when a load starts.

        :param floor: F
        :ptype floor: int
        :return: whether a load stopped it, whether a table was left unfinished for another reason (a full page none
            of which went, logged where it stopped), and the rows removed
        :rtype: tuple[bool, bool, int]
        """
        removed = 0
        unfinished = False
        for schema in self._stamped:
            cleaned = await self._cleaned_to(schema.name)
            resets = await self._resets_below(schema.name, cleaned.get(TABLE_RESET_SCOPE, 0), floor)
            removed += await self._clear_resets(schema.name, resets, floor)
            since = cleaned.get(STAMPED_SCOPE, 0)
            _BACKLOG.labels(table=schema.name).set(int(since < floor))
            if since >= floor:
                continue
            sweep: StampedSweep = await clear_stamped_gone(
                self._pool, schema, since=since, floor=floor, batch=self._batch, stirred=lambda: self._gate.stirred
            )
            removed += sweep.removed
            _REMOVED.labels(table=schema.name, kind="stamped_gone").inc(sweep.removed)
            if sweep.finished:
                # the table's moves below F too (tenant_moves): no restore reads them
                moves = await clear_moves(self._pool, schema, floor)
                removed += moves
                _REMOVED.labels(table=schema.name, kind="tenant_moves").inc(moves)
                await self._record(schema.name, STAMPED_SCOPE, floor)
                _BACKLOG.labels(table=schema.name).set(0)
            elif self._gate.stirred:
                return True, unfinished, removed
            else:
                unfinished = True
        return False, unfinished, removed

    async def _resets_below(self, table: str, cleaned_to: int, floor: int) -> tuple[tuple[str, int], ...]:
        """``table``'s reset markers (:data:`~threetears.core.ddl.run_markers.TABLE_RESET_SCOPE`, an empty prefix)
        stamped in ``[cleaned_to, floor)``, each its prefix and epoch: read by their key ``(table_name, scope)``. None
        for a round whose floor has not moved since they were cleared (:meth:`_clear_resets`).

        :param table: the table cleaned
        :ptype table: str
        :param cleaned_to: the epoch below which its reset markers were cleared
        :ptype cleaned_to: int
        :param floor: F
        :ptype floor: int
        :return: each marker's prefix and epoch, read-only
        :rtype: tuple[tuple[str, int], ...]
        """
        if self._run_markers is None or cleaned_to >= floor:
            return ()
        # the run-marker table's own columns, spelled as its declaration spells them (RUN_MARKERS_SCHEMA): its key
        # (table_name, scope, prefix, epoch), read a page at a time past the bound (table_name, scope)
        markers = await keyset_rows(
            self._pool,
            self._run_markers,
            columns=("prefix", "epoch"),
            by={"table_name": table, "scope": TABLE_RESET_SCOPE},
            order=(Sort("prefix"), Sort("epoch", cast="bigint")),
            where=[Fragment('t."epoch" >= $1', (cleaned_to,)), Fragment('t."epoch" < $1', (floor,))],
        )
        return tuple((str(marker["prefix"]), int(marker[EPOCH_COLUMN])) for marker in markers)

    async def _clear_resets(self, table: str, resets: Sequence[tuple[str, int]], floor: int) -> int:
        """remove ``table``'s reset markers below ``floor`` (:meth:`_resets_below`), each deleted by its whole key, then
        record the progress scope :data:`~threetears.core.ddl.run_markers.TABLE_RESET_SCOPE` at ``floor``, so a round
        whose floor has not moved reads none. Only once every version they hide is gone: a versioned table's scopes
        were each walked from their oldest version this round (:meth:`_clean_to`), and a ``TRUNCATE`` emptied the
        rest. A reset marker below F is read by no restore (each reads from its own floor, at or above F).

        :param table: the table cleaned
        :ptype table: str
        :param resets: its reset markers below ``floor`` not yet cleared, each its prefix and epoch
        :ptype resets: Sequence[tuple[str, int]]
        :param floor: F
        :ptype floor: int
        :return: the markers removed
        :rtype: int
        """
        if self._run_markers is None:
            return 0
        removed = 0
        for prefix, epoch in resets:
            gone = await delete_keyed(
                self._pool,
                self._run_markers,
                by={"table_name": table, "scope": TABLE_RESET_SCOPE, "prefix": prefix, EPOCH_COLUMN: epoch},
            )
            removed += gone.count
        _REMOVED.labels(table=table, kind="reset_marker").inc(removed)
        await self._record(table, TABLE_RESET_SCOPE, floor)
        return removed

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

    async def _clean_scope(
        self, schema: TableSchema, value: str, cleaned_to: int, floor: int, reset: int | None = None
    ) -> tuple[bool, int]:
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
        :param reset: the table's newest reset marker's epoch below ``floor`` not yet cleared: every version below it
            goes (:func:`_removable`); None for none
        :ptype reset: int | None
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
            removed += await self._clear(schema, candidates, floor, reset)
            paged = candidates.num_rows < self._batch
            if not paged:
                last = {name: candidates.column(name)[-1].as_py() for name in candidates.column_names}
                past = tuple(next_past(self._order(schema), last))
        if not paged and past is not None and self._place is None:
            # a load started before the next batch: the scope resumes after the last version visited. Only while the
            # place is still the one this call cleared: a place set during the awaits is newer and is kept
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

    async def _clear(self, schema: TableSchema, candidates: pa.Table, floor: int, reset: int | None = None) -> int:
        """remove what a page of visited versions supersedes: every version below ``floor`` of the keys it names is
        read by the primary key, the versions to remove worked out in Arrow on a thread, and deleted by stored key.

        :param schema: the table
        :ptype schema: TableSchema
        :param candidates: the visited versions
        :ptype candidates: pyarrow.Table
        :param floor: F
        :ptype floor: int
        :param reset: the table's newest reset marker's epoch below ``floor``, or None (:func:`_removable`)
        :ptype reset: int | None
        :return: the versions removed
        :rtype: int
        """
        if not candidates.num_rows:
            return 0
        key = schema.pk_columns
        casts = tuple(self._cast(schema, column) for column in key)
        types = rail_arrow_types([schema.column(name) for name in (*key, EPOCH_COLUMN)])
        held = await read_keyed_arrow(
            self._pool,
            schema.name,
            # the table's own key, written where the read is so the key it binds is the one read
            keys=KeyBatch(key, keys=[candidates.column(c).combine_chunks() for c in key], casts=casts),
            types=types,
            where=[Fragment(f't."{EPOCH_COLUMN}" < $1', (floor,))],
            # the versions themselves, as stored: a shared read would answer each key's newest live version alone
            with_gone=True,
        )
        versions = pa.concat_tables(held) if held else pa.schema(list(types.items())).empty_table()
        doomed = await asyncio.to_thread(_removable, candidates, versions, key, reset)
        # a versioned table's stored key: its key, then the epoch (a version a write)
        stored = schema.stored_key
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
                        stored,
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
        # the run-marker table's own columns, spelled as its declaration spells them (RUN_MARKERS_SCHEMA): its key
        # (table_name, scope, prefix, epoch), read a page at a time past the bound (table_name, scope)
        markers = await keyset_rows(
            self._pool,
            self._run_markers,
            columns=("prefix", "epoch"),
            by={"table_name": schema.name, "scope": value},
            order=(Sort("prefix"), Sort("epoch", cast="bigint")),
            where=[Fragment('t."epoch" < $1', (floor,))],
        )
        removed = 0
        stopped = False
        for marker in markers:
            text, epoch = str(marker["prefix"]), int(marker[EPOCH_COLUMN])
            values = self._prefix_of(schema, text)
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
                        # the run's rows by the key prefix its marker names: the key's leading columns, the first at
                        # least (a marker's text splits to one value at least; max says so where the rule reads it)
                        by=dict(zip(schema.pk_columns[: max(1, len(values))], values, strict=True)),
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
    def _prefix_of(schema: TableSchema, text: str) -> list[Any]:
        """a run marker's prefix as the values of its key's leading columns: its text split by the separator, each
        typed as its column, in key order.

        :param schema: the table
        :ptype schema: TableSchema
        :param text: the marker's prefix
        :ptype text: str
        :return: the values, one at least
        :rtype: list[Any]
        :raises ValueError: when the prefix names more values than the key has columns
        """
        parts = text.split(_PREFIX_SEPARATOR)
        if len(parts) > len(schema.pk_columns):
            raise ValueError(f"{schema.name}: run marker prefix {text!r} names more values than its key's columns")
        return [
            _decoded(schema.column(name).column_type, part)
            for name, part in zip(schema.pk_columns, parts, strict=False)
        ]
