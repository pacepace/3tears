"""the change log's keeper: it marks the log's position, puts a lost log back from the tables, and scrubs.

The epoch change log (``{ns}-epoch-changes``, memory storage in NATS, three replicas) is the only clock and the only
catch-up path. What it cannot survive alone is NATS losing it. Since the change-note ring retired (append-only L3,
``design-append-only-l3.md`` decision 3, Pace 2026-11-25) every table carries its own epoch, so the tables are their own
change record: a lost log is rebuilt from the versions and stamped rows written since a mark of it
(:mod:`threetears.core.collections.table_restore`). This module keeps what that needs, and nothing of the ring.

**Why a module of its own, needing the ``arrow`` extra** (Pace 2026-11-15: pyarrow stays optional in core). A restore
folds what it reads as Arrow in a worker process (:mod:`threetears.core.change_note_fold`), so this module imports
pyarrow and NumPy at module top; only the process that runs the keeper (the hub, identity-core) installs
``3tears[arrow]``.

**The marks.** Every 30 s (:data:`_MARK_EVERY`) the keeper writes a mark (P, T) into its database
(:mod:`threetears.core.ddl.keeper_marks`): the log's position P, read after the database's clock read T. Every record
appended after P carries stamps at or above X = (T - M) << 15 (:func:`~threetears.core.ddl.row_epoch.restore_floor`), so
a restore from that mark reads the tables from X and labels the new log as covering every record after P. Marks are kept
for the log's whole window (:meth:`LogKeeper.keep_window_marks`): the newest mark before the log's oldest record (the
anchor) and every one after it; never by age or a TTL.

**What the keeper knows of the log is its newest mark.** No row holds the keeper's state: the log it last knew is the
incarnation its newest mark names (:meth:`LogKeeper.newest_mark`, one row read backwards through the marks' time
index). A pass that finds another log than that one takes it at once by marking it, so the newest mark always names
the log the keeper last took. With no mark at all (a database's first start) the keeper knows no log: it declares one
where the broker holds none, and takes the one it finds otherwise.

**Putting a lost log back.** When the log is absent and the keeper's newest mark names one, that log was lost. The
keeper first declares a new log under a NEW incarnation, marked as being restored from the lost one and continuing
nothing yet, so a write waiting for the log appends at once. Then it reads the tables' changes from the oldest mark of
the lost log (the window's anchor: it vouches for every record the lost log held), appends them, and only then labels
the log as continuing the lost one from that mark's position (``covered_from``). Until the label a follower of the lost
log is behind (``replaced``), never carried over to a partial log. A restore interrupted before its label (a failed
append, a lost lease, a crash) is found by the next pass, whose newest mark still names the lost log, and done again in
full into the same log, under fresh append ids. Once labelled, a follower whose mark is at or past ``covered_from``
carries it to the new log's start and walks every record in it; an older one is behind (``record_lost``) until it
restarts. The keeper then marks the new log, and knows it.

**A restore too big for the log.** Before it appends anything, the keeper sizes the tables' changes from the anchor
against the restore budget (:data:`~threetears.core.collections.change_notes.RESTORE_BUDGET_RATIO` of the log's
``max_bytes``, what it already holds included), read in the restore's worker and summed, never held. When they do not
fit, it puts back the newest part that does: the changes since a later mark (a binary search over the lost log's marks,
each size one read), labelled from that mark's position. Followers older than it are behind; said at ERROR and counted
(``threetears_change_notes_restore_truncated_total``). Only when not even the part after the newest mark fits is the
restore refused (``threetears_change_notes_restores_refused_total``) and the new log taken as a fresh one. With no mark of
the lost log at all nothing vouches for the tables: the label is ``None`` (every follower behind), said at ERROR and
counted (``threetears_restore_unvouched_total``).

**A keeper beside a log another keeper owns** (Pace 2026-11-16). One change log is shared by every database whose
writers advance through it (the hub's platform database, identity-core's own), and one keeper, the hub's, owns its
lifecycle: it alone declares the log and declares and labels a lost one's restore. A second database's tables are read
only from that database, so its keeper is built ``keeps_log=False``: it marks the log in its own database, and once the
log is a restore of the one its newest mark names (being restored, or labelled) it puts its own tables' changes back
into it (``put_back``, labelling nothing), then marks the new log. Where its mark is after the owner's label's
``covered_from``, a follower whose mark lies between may have missed one of its writes: said at ERROR and counted
(``threetears_change_notes_put_back_short_total``, ``reason=floor_after_label``), as is a part too big to put back
(``reason=too_large``). While the log is absent its passes do nothing: the owner declares the restore.

**The scrub.** Every minute (:data:`_SCRUB_EVERY`) it checks up to ``scrub_batch`` records the log holds, from the
oldest it holds on. A record of a table a restore cannot read back (no versioned or stamped table of the name in this
database: one carrying no epoch) is logged at ERROR and counted (``threetears_change_notes_missing_total{table}``): a
restore after a NATS loss would not put it back. A record after a mark whose lowest stamp (``s_min``) is below that
mark's X is too (``threetears_restore_margin_violations_total``). Each keeper judges only its own database's records:
one naming its database, or (an older writer's, naming none) one of a table of its schema. The log's owner reads the
log by sequence; a keeper beside reads each of its schema's tables' records by the table's keys (``changes_after``),
never by sequence (2026-11-24: that read's subject rides in its request body, where no grant can narrow it). It only
checks and reports.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from threetears.nats.change_log import ChangeLogState
from threetears.observe import contain, get_logger
from threetears.observe.metrics import counter, gauge

from threetears.core.change_note_fold import RestoreFolder
from threetears.core.collections.change_notes import RESTORE_BUDGET_RATIO, NotedChange
from threetears.core.collections.epoch_change import EpochChange, EpochWalk
from threetears.core.collections.table_restore import RestoredTable, StampedTable, TableChanges, restored_tables
from threetears.core.coordination.lease import KVLease, LeaseUnavailable
from threetears.core.ddl.ddl_lock import DdlLockPolicy, database_ddl_lock
from threetears.core.ddl.epoch_catalog import refuse_moveable_without_moves, refuse_row_triggers
from threetears.core.ddl.index_health import ddl_body, rebuild_unusable_indexes
from threetears.core.ddl.keeper_marks import MARKS_TABLE, keeper_marks_ddl
from threetears.core.ddl.row_epoch import epoch_wall_ms, restore_floor
from threetears.core.ddl.session import ConnectionSession
from threetears.core.declared import whole_read
from threetears.core.keyed import Fragment, Sort, delete_before, keyset_page, keyset_rows
from threetears.core.sql_fragments import quote_identifier

if TYPE_CHECKING:
    from threetears.core.coordination.lease import HeldLease
    from threetears.nats.kv import KvCapable

__all__ = [
    "LOG_KEEPER_CLAIM_EVERY",
    "KeeperPass",
    "LogKeeper",
    "LogKeeping",
    "ScrubPass",
    "run_log_keeper",
]

log = get_logger(__name__)

#: how often the keeper marks the log's position: a restore's coverage starts at most this far before any record.
#: Twentieth of the ten-minute fence the ring marked by, kept
_MARK_EVERY: Final = timedelta(seconds=30)

#: how often the keeper scrubs: every minute, as the ring's scrub ran at its ten-minute fence
_SCRUB_EVERY: Final = timedelta(minutes=1)

#: the marks the window's sweep deletes a statement. Why a paged delete_before over the (at, id) index: a keeper writes
#: one mark every _MARK_EVERY and the sweep drops the few older than the window's anchor, so the sweep is one
#: statement in practice and 500 only bounds a sweep after a long outage. Measured (a local A/B on Postgres 17, best of
#: 5, 20 marks swept): 1.1 vs 1.5 ms at 60 marks, 1.4 vs 1.7 ms at 2,000, 2.8 vs 1.7 ms at 20,000 (a one-statement
#: delete vs this); its plan test proves the index read on Postgres and YugabyteDB
#: (test_keeper_marks_plans::test_threetears_change_note_marks_at_reads_the_windows_sweep).
_MARKS_SWEEP_PAGE: Final = 500

#: a lost log's marks, in position order (two marks may share a position: the id breaks the tie)
_MARKS_ORDER: Final = (Sort("seq"), Sort("id"))

#: a log's marks newest first: the window's anchor is the first before its oldest record (keep_window_marks)
_MARKS_NEWEST: Final = (Sort("seq", cast="bigint", descending=True), Sort("id", cast="uuid", descending=True))

#: every mark newest first by the database's clock: the first names the log the keeper last took (newest_mark)
_MARKS_LATEST: Final = (Sort("at", cast="timestamptz", descending=True), Sort("id", cast="uuid", descending=True))

#: the pause between keeper passes: a write waits for a lost log's restore up to one pass, well inside a source's
#: ``restore_wait``
_DEFAULT_KEEPER_EVERY: Final = timedelta(seconds=1)

#: records the scrub checks per pass
_SCRUB_BATCH: Final = 1000

#: how many log records (or, beside, tables) the scrub reads at once
_SCRUB_READS_AT_ONCE: Final = 64

#: a restored record's bytes in the log beyond its body (its key, the log's framing), as the sizing's estimate:
#: generous, so a part chosen to fit does fit
_RECORD_ESTIMATE: Final = 256

#: the base tables of the keeper's schema: what an older writer's record (naming no database) of a name is judged by,
#: and the tables a keeper beside reads the log by. A catalog read, its size the schema's tables
_SCHEMA_TABLES_SQL: Final = """
SELECT t.table_name::text AS table_name FROM information_schema.tables t
WHERE t.table_schema::text = $1 AND t.table_type = 'BASE TABLE'
ORDER BY 1
"""

_MISSING = counter(
    "threetears_change_notes_missing_total",
    description="records the change log holds of a table a restore cannot read back (no versioned or stamped table of "
    "the name: it carries no epoch); a restore after a NATS loss would not put them back. Alert on any",
    label_names=("table",),
)
_MARKS_DROPPED = counter(
    "threetears_change_log_marks_dropped_total",
    description="keeper marks of the change log dropped because no restore of the log's window could need them "
    "(older than the newest mark before the log's oldest record)",
)
_RESTORES = counter(
    "threetears_change_notes_restores_total",
    description="lost change logs the keeper put back from the tables' own epochs",
)
_RESTORES_REFUSED = counter(
    "threetears_change_notes_restores_refused_total",
    description="lost change logs NOT restored because not even the changes after the newest mark fit the log with "
    "headroom; alert on any",
)
_RESTORES_TRUNCATED = counter(
    "threetears_change_notes_restore_truncated_total",
    description="restores that put back only the changes after a later mark, the newest part that fit the log; "
    "followers older than its covered_from are behind",
)
_PUT_BACKS = counter(
    "threetears_change_notes_put_backs_total",
    description="restored change logs a keeper beside the log's owner put its own tables' changes back into",
)
_RESTORE_UNVOUCHED = counter(
    "threetears_restore_unvouched_total",
    description="lost change logs no keeper mark vouched for: every follower behind",
)
_MARGIN_VIOLATIONS = counter(
    "threetears_restore_margin_violations_total",
    description="log records after a keeper mark whose lowest stamp is below the mark's restore floor X; alert on any",
    label_names=("table",),
)
_PUT_BACK_SHORT = counter(
    "threetears_change_notes_put_back_short_total",
    description="a keeper beside the log's owner put back less than the restore's label vouches for: its mark is "
    "after the label's covered_from (floor_after_label), or its changes did not fit the log (too_large)",
    label_names=("reason",),
)
_KEEPER_LAST_PASS = gauge(
    "threetears_change_notes_keeper_last_pass_timestamp_seconds",
    description="unix time of this process's last whole log keeper pass; alert when no process has moved it for "
    "several minutes (no keeper holds the lease, or every pass fails)",
)
_SCRUB_BEHIND = gauge(
    "threetears_change_notes_scrub_behind",
    description="records the change log holds after the scrub's position: how far the scrub is behind",
)


@runtime_checkable
class _ChangeLogKeeping(Protocol):
    """the change log as the keeper reads it and puts it back: :class:`threetears.epoch.EpochGenerationSource`."""

    async def log_state(self) -> ChangeLogState | None:
        """what the log holds, or ``None`` when the broker holds none.

        :return: the state
        :rtype: ChangeLogState | None
        :raises GenerationUnavailableError: when the log cannot be read
        """
        ...

    async def epoch_at(self, seq: int) -> EpochChange | None:
        """the record the log holds at ``seq`` under its current incarnation, of whichever table.

        :param seq: the sequence
        :ptype seq: int
        :return: the record, or ``None`` when the log no longer holds it
        :rtype: EpochChange | None
        :raises GenerationUnavailableError: when the log cannot be read
        """
        ...

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        """every epoch of ``table_name`` after ``after`` and up to ``upto``, read by the table's keys.

        :param table_name: the table
        :ptype table_name: str
        :param after: the token to read past
        :ptype after: str
        :param upto: the last token wanted, under the same incarnation
        :ptype upto: str
        :return: the epochs the log holds in that range; another database's records of the name are walked past
        :rtype: EpochWalk
        :raises GenerationUnavailableError: when the log cannot be read
        """
        ...

    async def start_log(self) -> str:
        """declare a fresh log where the broker holds none.

        :return: the log's incarnation
        :rtype: str
        :raises GenerationUnavailableError: when the log can be neither read nor declared
        """
        ...

    async def begin_restore(self, lost: str) -> str | None:
        """declare the log a restore of ``lost`` fills, marked as being restored and continuing nothing yet.

        :param lost: the lost log's incarnation
        :ptype lost: str
        :return: the new log's incarnation; ``None`` when a log was already there
        :rtype: str | None
        :raises GenerationUnavailableError: when the log cannot be declared
        """
        ...

    async def finish_restore(
        self,
        changes: Sequence[NotedChange] | AsyncIterable[NotedChange],
        *,
        incarnation: str,
        lost: str,
        covered_from: int | None,
    ) -> int:
        """append one record per change into the restoring log, then label it as continuing ``lost``.

        :param changes: what the tables say changed, per table, held or streamed
        :ptype changes: Sequence[NotedChange] | AsyncIterable[NotedChange]
        :param incarnation: the restoring log
        :ptype incarnation: str
        :param lost: the lost log's incarnation
        :ptype lost: str
        :param covered_from: the lost log's sequence from which the changes cover every epoch; ``None`` when they
            vouch for none
        :ptype covered_from: int | None
        :return: the records appended
        :rtype: int
        :raises RestoreTooLargeError: when the restore would not fit with headroom
        :raises GenerationUnavailableError: when the log cannot be written
        """
        ...

    async def put_back(
        self,
        changes: Sequence[NotedChange] | AsyncIterable[NotedChange],
        *,
        incarnation: str,
        lost: str,
    ) -> int:
        """append one record per change into the log restored from ``lost``, labelling nothing (a keeper beside).

        :param changes: what this database's tables say changed, per table
        :ptype changes: Sequence[NotedChange] | AsyncIterable[NotedChange]
        :param incarnation: the restored log
        :ptype incarnation: str
        :param lost: the lost log's incarnation
        :ptype lost: str
        :return: the records appended
        :rtype: int
        :raises RestoreTooLargeError: when the changes would not fit with headroom
        :raises GenerationUnavailableError: when the log is not ``lost``'s restore, or cannot be written
        """
        ...


class _Pool(Protocol):
    """what the keeper needs of the L3 pool (an ``asyncpg.Pool`` is one)."""

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        """rows of ``query``."""
        ...

    async def execute(self, query: str, *args: Any) -> str:
        """run ``query``; its status."""
        ...

    def acquire(self) -> AbstractAsyncContextManager[Any]:
        """one connection, for a sequence that must share a session (the DDL lock's)."""
        ...


@dataclass(frozen=True, slots=True)
class _Mark:
    """one mark of the log: its position under its incarnation, at the database's time."""

    incarnation: str
    seq: int
    at: datetime


@dataclass(slots=True)
class ScrubPass:
    """what one scrub found.

    :ivar checked: records checked: those of this keeper's database
    :ivar missing: records of a table a restore cannot read back
    :ivar position: the newest record checked, ``incarnation:seq``; ``None`` when none was
    :ivar others: records skipped: another database's. The log's owner reads and skips them; a keeper beside never reads
        them, and counts the range's records not of its tables
    :ivar margin_violations: records after a keeper mark whose lowest stamp (``s_min``) is below the mark's restore
        floor X: a restore from that mark would miss part of them (:meth:`LogKeeper._check_margins`)
    """

    checked: int = 0
    missing: int = 0
    position: str | None = None
    others: int = 0
    margin_violations: int = 0


@dataclass(slots=True)
class KeeperPass:
    """what one keeper pass did.

    :ivar started: the incarnation of a log the keeper declared because none was there and it knew none
    :ivar restored: the incarnation of a lost log the keeper put back from the tables
    :ivar adopted: the incarnation of a log the keeper found that it did not know
    :ivar put_back: the incarnation of a restored log this keeper, beside the log's owner, put its own changes back into
    :ivar marked: the pass wrote a mark
    :ivar scrub: what the scrub found
    """

    started: str | None = None
    restored: str | None = None
    adopted: str | None = None
    put_back: str | None = None
    marked: bool = False
    scrub: ScrubPass = field(default_factory=ScrubPass)


def _token(incarnation: str, seq: int) -> str:
    """the token of ``seq`` under ``incarnation``.

    :param incarnation: the log's incarnation
    :ptype incarnation: str
    :param seq: the sequence
    :ptype seq: int
    :return: ``incarnation:seq``
    :rtype: str
    """
    return f"{incarnation}:{seq}"


def _seq(record: EpochChange) -> int:
    """a record's sequence in its log.

    :param record: the record
    :ptype record: EpochChange
    :return: the sequence its token names
    :rtype: int
    """
    return int(record.token.rpartition(":")[2])


def _x(mark: _Mark) -> int:
    """the restore floor X a mark vouches from (:func:`~threetears.core.ddl.row_epoch.restore_floor`).

    :param mark: the mark
    :ptype mark: _Mark
    :return: X
    :rtype: int
    """
    return restore_floor(int(mark.at.timestamp() * 1000))


def _bytes(change: NotedChange) -> int:
    """the bytes one change's records take in the log, by the sizing's generous estimate.

    :param change: the change
    :ptype change: NotedChange
    :return: the estimate
    :rtype: int
    """
    if change.bodies:
        return sum(len(body) for body in change.bodies) + _RECORD_ESTIMATE * len(change.bodies)
    named = sum(len(name) + len(value) for prefix in change.prefixes for name, value in prefix.items())
    return _RECORD_ESTIMATE + len(change.table) + named + 64 * len(change.rows)


class LogKeeper:
    """the one keeper of the change log in a database: marks it, puts a lost one back from the tables, scrubs.

    Run :meth:`run_pass` from one process at a time (a :class:`~threetears.core.coordination.lease.KVLease` held around
    it, :class:`LogKeeping`); two at once mark twice and scrub twice, and never restore twice (the log is declared once).

    Every process whose change log a restore may put back builds its generation source with ``restored_from_l3=True``,
    so a write after a loss waits for this keeper rather than starting a log no follower can continue from.
    """

    def __init__(
        self,
        pool: _Pool,
        *,
        schema: str,
        log: _ChangeLogKeeping,
        mark_every: timedelta = _MARK_EVERY,
        scrub_every: timedelta = _SCRUB_EVERY,
        scrub_batch: int = _SCRUB_BATCH,
        keeps_log: bool = True,
    ) -> None:
        """capture what the keeper works between; no I/O.

        :param pool: the L3 pool holding the marks, allowed DDL (installing takes the DDL lock)
        :ptype pool: _Pool
        :param schema: the schema holding the marks (the platform schema; a second database's own), a TRUSTED
            identifier; the database name every record of this database names
        :ptype schema: str
        :param log: the change log
        :ptype log: _ChangeLogKeeping
        :param mark_every: the least time between two marks
        :ptype mark_every: timedelta
        :param scrub_every: the least time between two scrubs
        :ptype scrub_every: timedelta
        :param scrub_batch: records the scrub checks per pass; at least one
        :ptype scrub_batch: int
        :param keeps_log: this keeper owns the change log: it declares it where it knows none and puts a lost one back.
            ``False`` for a keeper in a second database beside a log another keeper owns (module docstring)
        :ptype keeps_log: bool
        :return: nothing
        :rtype: None
        :raises ValueError: when an interval is not positive or the batch is under one; when the log names a database
            other than ``schema``; when a keeper beside the log names none
        """
        # the database the log's source names on every record (EpochGenerationSource.database); a log object without
        # one (a stand-in that only reads) names none
        named = getattr(log, "database", None)
        database = named if isinstance(named, str) else None
        # WHY (2026-11-16): a record names the database it was written in, and this keeper's scrub and restore judge
        # and name records by it. The name is the keeper's schema, never a second setting that could drift; and a
        # keeper beside another keeper's log exists only because two databases share it, so it may not name none
        if database is not None and database != schema:
            raise ValueError(
                f"a keeper's log names its database by the keeper's schema, {schema!r}; it names {database!r}"
            )
        if not keeps_log and database is None:
            raise ValueError(
                "a keeper beside another keeper's change log names its database on every record: build its "
                f"EpochGenerationSource with database={schema!r}"
            )
        if mark_every <= timedelta(0) or scrub_every < timedelta(0) or scrub_batch < 1:
            raise ValueError("a keeper marks at a positive interval, and its scrub checks at least one record a batch")
        self._pool = pool
        self._schema = schema
        self._log = log
        self._keeps_log = keeps_log
        self._database = database
        self._mark_every = mark_every
        self._scrub_every = scrub_every
        self._scrub_batch = scrub_batch
        self._marks = f"{quote_identifier(schema)}.{quote_identifier(MARKS_TABLE)}"
        #: the same, as the keyed helpers name a table (``schema.table``, quoted a segment at a time by them)
        self._marks_table = f"{schema}.{MARKS_TABLE}"
        #: when (monotonic) this process last marked; None before its first
        self._marked_at: float | None = None
        self._scrubbed_at: float | None = None
        #: the newest record the scrub has checked, and the log's head at the previous pass (what it checks up to)
        self._scrubbed: tuple[str, int] | None = None
        self._seen: tuple[str, int] | None = None
        #: a keeper beside: the restored log it put its changes back into before the owner labelled it, the lost log,
        #: and the mark it put back from, so the label's covered_from is checked against it once it is written
        self._unchecked: tuple[str, str, int | None] | None = None

    async def install(self, policy: DdlLockPolicy | None = None) -> None:
        """create the marks table where missing, under the database's DDL lock, every index of it usable: one a failed
        build left unusable is dropped and built again from its own statement. Then refuse a per-row trigger on any
        versioned table of the database (:func:`~threetears.core.ddl.epoch_catalog.refuse_row_triggers`), and a
        moveable stamped table whose schema holds no moves table
        (:func:`~threetears.core.ddl.epoch_catalog.refuse_moveable_without_moves`).

        :param policy: how long to wait for the DDL lock; the default waits until it is free
        :ptype policy: DdlLockPolicy | None
        :return: nothing
        :rtype: None
        :raises IndexStillUnusableError: when a marks index is still unusable after it was built again
        :raises PerRowTriggerError: when a versioned table carries a per-row trigger
        :raises MovesTableMissingError: when a moveable stamped table's schema holds no moves table
        """
        statements = keeper_marks_ddl(self._schema)
        # each index's name, as its own statement creates it (CREATE INDEX IF NOT EXISTS "<name>" ON ...), past the
        # ``-- purpose:`` lines that declare what reads it
        declared = {
            ddl_body(statement).split()[5].strip('"'): statement
            for statement in statements
            if ddl_body(statement).startswith("CREATE INDEX ")
        }
        async with self._pool.acquire() as conn:
            async with database_ddl_lock(ConnectionSession(conn), policy):
                for statement in statements:
                    await conn.execute(statement)
                await whole_read(
                    rebuild_unusable_indexes(
                        conn, [self._schema], names=list(declared), policy=policy, definitions=declared
                    ),
                    kind="schema_change",
                    reason="the marks' own indexes a failed build left unusable are built again; none is unique, so "
                    "no row is read",
                    ruling="Pace 2026-11-12: a schema change may read whole; coordinator 2026-11-12: schema_change "
                    "admits unknown-schema",
                    proof="packages/core/tests/unit/ddl/test_keeper_marks_ddl.py::test_the_marks_install_rebuilds_no_unique_index",
                    speed="packages/core/tests/integration/test_index_health_live.py::test_rebuild_speed_at_real_size",
                )
            # at every start: no versioned (page-written) table of the database carries a per-row trigger
            await refuse_row_triggers(conn)
            # at every start: every moveable stamped table's schema holds the moves table its moving writes append to
            await refuse_moveable_without_moves(conn)

    async def run_pass(self) -> KeeperPass:
        """keep the log for one pass: put back a lost log, take a new one, mark, scrub.

        :return: what the pass did
        :rtype: KeeperPass
        :raises GenerationUnavailableError: when the marks or the log cannot be read, or a restore failed
        """
        result = KeeperPass()
        # the database's clock read before the log's state: a mark's time must not be later than its position
        read_at = await self._now()
        newest = await self.newest_mark()
        known = None if newest is None else newest.incarnation
        state = await self._log.log_state()
        if not self._keeps_log:
            if state is None:
                # the log's owner declares it, or declares its restore; this keeper puts its changes back at a later pass
                _KEEPER_LAST_PASS.set(time.time())
                return result
            self._check_label(state)
            if known is not None and state.incarnation != known and known in (state.restoring, state.continues):
                # a restore of the log this keeper last took: the owner put back its own database's tables only
                async with RestoreFolder() as folder:
                    result.put_back = await self._put_back(state, known, folder)
                result.marked = await self._mark(await self._now(), force=True)
                _KEEPER_LAST_PASS.set(time.time())
                return result
        if state is None:
            if known is None:
                result.started = await self._log.start_log()
                result.marked = await self._mark(read_at, force=True)
            else:
                # the restore's worker exits when it ends (RestoreFolder): a restore is rare, the worker 193 MiB
                async with RestoreFolder() as folder:
                    result.restored = await self._restore(known, folder)
                if result.restored is not None:
                    result.marked = await self._mark(await self._now(), force=True)
            _KEEPER_LAST_PASS.set(time.time())
            return result
        if self._keeps_log and state.restoring is not None and state.restoring == known and state.continues is None:
            # a restore of the keeper's lost log was declared and never labelled (a failed append, a lost lease, a
            # crash): it is finished, or done again, before anything else
            async with RestoreFolder() as folder:
                result.restored = await self._restore(known, folder, incarnation=state.incarnation)
            if result.restored is not None:
                result.marked = await self._mark(await self._now(), force=True)
            _KEEPER_LAST_PASS.set(time.time())
            return result
        adopted = known != state.incarnation
        if adopted:
            self._adopt(known, state)
            result.adopted = state.incarnation
        result.marked = await self._mark(read_at, state, force=adopted)
        now = time.monotonic()
        if self._scrubbed_at is None or now - self._scrubbed_at >= self._scrub_every.total_seconds():
            self._scrubbed_at = now
            result.scrub = await self._scrub(state)
        _KEEPER_LAST_PASS.set(time.time())
        return result

    async def _now(self) -> datetime:
        """the database's clock.

        :return: its ``now()``
        :rtype: datetime
        """
        rows = await self._pool.fetch("SELECT now() AS now")
        at: datetime = rows[0]["now"]
        return at

    def _adopt(self, known: str | None, state: ChangeLogState) -> None:
        """say a log the keeper did not know is taken (its mark follows at once).

        :param known: the log the keeper's newest mark names; ``None`` before its first mark
        :ptype known: str | None
        :param state: the log found
        :ptype state: ChangeLogState
        :return: nothing
        :rtype: None
        """
        if known is None or (state.continues is not None and state.continues == known):
            return
        log.warning(
            "the change log is under an incarnation the log keeper did not put back; it is taken from its start",
            extra={"extra_data": {"was": known, "now": state.incarnation}},
        )

    async def newest_mark(self) -> _Mark | None:
        """the newest mark by the database's clock, of whichever log: the log the keeper last took. One row, read
        backwards through the marks' time index.

        :return: the mark; ``None`` when there is none
        :rtype: _Mark | None
        """
        page = await keyset_page(
            self._pool,
            self._marks_table,
            columns=("log_incarnation", "seq", "at", "id"),
            by={},
            order=_MARKS_LATEST,
            limit=1,
        )
        if not page.rows:
            return None
        row = page.rows[0]
        return _Mark(incarnation=str(row["log_incarnation"]), seq=int(row["seq"]), at=row["at"])

    async def _mark(self, read_at: datetime, state: ChangeLogState | None = None, *, force: bool = False) -> bool:
        """record the log's position every ``mark_every``, or at once when ``force``: where a restore may start, and
        (the newest mark) the log the keeper knows.

        The mark's time is the database's clock read before the log's state (``read_at``), so every record after the
        mark's position was appended after that time.

        :param read_at: the database's clock, read before the log's state
        :ptype read_at: datetime
        :param state: the log, read after it; ``None`` reads it here
        :ptype state: ChangeLogState | None
        :param force: mark now, whenever the last mark was (a log just taken: the newest mark must name it)
        :ptype force: bool
        :return: whether a mark was written
        :rtype: bool
        """
        now = time.monotonic()
        if not force and self._marked_at is not None and now - self._marked_at < self._mark_every.total_seconds():
            return False
        if state is None:
            state = await self._log.log_state()
            if state is None:
                return False
        self._marked_at = now
        await self._pool.execute(
            f"INSERT INTO {self._marks} (log_incarnation, seq, at) VALUES ($1, $2, $3)",  # noqa: S608 - trusted identifiers
            state.incarnation,
            state.last_seq,
            read_at,
        )
        await self.keep_window_marks(state)
        return True

    async def drop_marks_before(self, at: datetime) -> int:
        """delete the marks written before ``at``, a page at a time through the marks' time index (no cache serves
        the keeper's own table).

        :param at: the time: every mark older is deleted
        :ptype at: datetime
        :return: the marks deleted
        :rtype: int
        """
        swept = await delete_before(
            self._marks_table,
            collection=None,
            key=("id",),
            column="at",
            before=at,
            page=_MARKS_SWEEP_PAGE,
            conn=self._pool,
        )
        return swept.deleted

    async def marks_past(self, incarnation: str, floor: int | None) -> list[dict[str, Any]]:
        """the marks of the log ``incarnation`` names past the position ``floor``, in position order: a page at a time
        through the marks' position index.

        :param incarnation: the log's incarnation
        :ptype incarnation: str
        :param floor: the position the marks are past; every mark when None
        :ptype floor: int | None
        :return: the marks (``seq``, ``at``, ``id``)
        :rtype: list[dict[str, Any]]
        """
        return await keyset_rows(
            self._pool,
            self._marks_table,
            columns=("seq", "at", "id"),
            by={"log_incarnation": incarnation},
            order=_MARKS_ORDER,
            where=[Fragment('t."seq" > $1', (-1 if floor is None else floor,))],
        )

    async def keep_window_marks(self, state: ChangeLogState) -> int:
        """drop only the marks no restore of the log's window could need: those older than the window's anchor.

        The window is every record the log holds (``first_seq`` to ``last_seq``), the same window quiet-time cleanup
        keeps versions for (``quiet_cleanup.log_window_floor``). A mark at position P vouches for every record after P,
        so the window needs the newest mark before its first record (the anchor) and every mark after it; a restore
        then finds a mark at or before any record inside the window. Every older mark, of this log or a log before it,
        vouches only for records the log no longer holds, and goes: a page at a time through the marks' time index.
        With no anchor yet (the log holds every record since its first mark, or a fresh log after a restore) nothing
        goes: the lost log's marks stay until the new log's window passes them. Never by age or a TTL. The newest mark
        is never older than the anchor, so it always stays.

        :param state: the log, as this pass read it
        :ptype state: ChangeLogState
        :return: the marks dropped
        :rtype: int
        """
        anchor = await keyset_page(
            self._pool,
            self._marks_table,
            columns=("seq", "at", "id"),
            by={"log_incarnation": state.incarnation},
            order=_MARKS_NEWEST,
            limit=1,
            where=[Fragment('t."seq" < $1', (state.first_seq,))],
        )
        if not anchor.rows:
            return 0
        dropped = await self.drop_marks_before(anchor.rows[0]["at"])
        if dropped:
            _MARKS_DROPPED.inc(dropped)
        return dropped

    async def _marks_of(self, incarnation: str) -> list[_Mark]:
        """a log's marks, in position order.

        :param incarnation: the log's incarnation
        :ptype incarnation: str
        :return: the marks
        :rtype: list[_Mark]
        """
        return [
            _Mark(incarnation=incarnation, seq=int(row["seq"]), at=row["at"])
            for row in await self.marks_past(incarnation, None)
        ]

    async def _size(self, tables: Sequence[RestoredTable], mark: _Mark, folder: RestoreFolder) -> int:
        """the bytes the tables' changes since ``mark``'s X take in the log, by a generous estimate: read and folded in
        the restore's worker as the restore reads them, summed a change at a time, never held.

        :param tables: the versioned and stamped tables
        :ptype tables: Sequence[RestoredTable]
        :param mark: the mark
        :ptype mark: _Mark
        :param folder: the restore's worker
        :ptype folder: RestoreFolder
        :return: the estimate
        :rtype: int
        """
        total = 0
        async for change in TableChanges(self._pool, tables, _x(mark), folder, self._database):
            total += _bytes(change)
        return total

    async def _fitting(
        self, tables: Sequence[RestoredTable], marks: Sequence[_Mark], budget: float | None, folder: RestoreFolder
    ) -> _Mark | None:
        """the oldest mark whose changes since its X fit ``budget``: the oldest (the window's anchor, vouching for every
        record the lost log held) when they do, else a binary search over the later ones, each size one read (each is a
        suffix of the one before, so sizes fall along the list).

        :param tables: the versioned and stamped tables
        :ptype tables: Sequence[RestoredTable]
        :param marks: the lost log's marks, in position order; at least one
        :ptype marks: Sequence[_Mark]
        :param budget: the bytes the restore may take; ``None`` unbounded
        :ptype budget: float | None
        :param folder: the restore's worker
        :ptype folder: RestoreFolder
        :return: the mark; ``None`` when not even the newest one's part fits
        :rtype: _Mark | None
        """
        if budget is None or not tables or await self._size(tables, marks[0], folder) <= budget:
            return marks[0]
        low, high = 1, len(marks)
        while low < high:
            middle = (low + high) // 2
            if await self._size(tables, marks[middle], folder) <= budget:
                high = middle
            else:
                low = middle + 1
        return None if low == len(marks) else marks[low]

    async def _budget(self) -> float | None:
        """the bytes a restore may still append: :data:`RESTORE_BUDGET_RATIO` of the log's bound, what it holds less.

        :return: the budget; ``None`` when the log is unbounded or absent
        :rtype: float | None
        """
        state = await self._log.log_state()
        return None if state is None or not state.max_bytes else state.max_bytes * RESTORE_BUDGET_RATIO - state.bytes

    async def _restore(self, lost: str, folder: RestoreFolder, *, incarnation: str | None = None) -> str | None:
        """put the lost log back from the tables' changes since its oldest mark, or the newest part that fits.

        In order: declare the new log first (so writers waiting for it append at once), then size and read the tables,
        then append and label the log as continuing the lost one from the chosen mark's position. ``incarnation`` names
        a restore declared earlier and never labelled; it is done again, in full, into that log.

        :param lost: the lost log's incarnation (the keeper's newest mark names it)
        :ptype lost: str
        :param folder: the restore's worker, spawned here once the restore is this keeper's
        :ptype folder: RestoreFolder
        :param incarnation: the restoring log to finish; ``None`` declares one
        :ptype incarnation: str | None
        :return: the new log's incarnation; ``None`` when a log was there and it was not this restore's, or the restore
            was refused (the new log then marked, taken as a fresh one)
        :rtype: str | None
        :raises GenerationUnavailableError: when the log cannot be declared or written; the next pass redoes it
        """
        if incarnation is None:
            incarnation = await self._log.begin_restore(lost)
            if incarnation is None:
                return None
        folder.start()
        began = time.monotonic()
        tables = await restored_tables(self._pool)
        marks = await self._marks_of(lost)
        chosen: _Mark | None = None
        label: int | None = None
        appended = 0
        changes: TableChanges | None = None
        if not marks:
            _RESTORE_UNVOUCHED.inc()
            log.error(
                "a lost change log cannot be restored from the tables: the keeper holds no mark of it, so nothing says "
                "from which epoch to read them. Every follower of the lost log is behind until it restarts",
                extra={"extra_data": {"lost": lost, "tables": [table.relation for table in tables]}},
            )
            appended = await self._log.finish_restore((), incarnation=incarnation, lost=lost, covered_from=None)
        else:
            chosen = await self._fitting(tables, marks, await self._budget(), folder)
            if chosen is None:
                _RESTORES_REFUSED.inc()
                log.error(
                    "the lost epoch change log was NOT restored: not even the tables' changes after its newest mark "
                    "fit the log with headroom. Every follower of the lost log is behind (replaced) until it restarts; "
                    "the new log is taken as a fresh one",
                    extra={"extra_data": {"lost": lost, "restored": incarnation, "marks": len(marks)}},
                )
                self._scrubbed = None
                self._seen = None
                # the new log is taken as a fresh one: marked now, so the next pass does not restore it again
                await self._mark(await self._now(), force=True)
                return None
            label = chosen.seq
            changes = TableChanges(self._pool, tables, _x(chosen), folder, self._database)
            appended = await self._log.finish_restore(changes, incarnation=incarnation, lost=lost, covered_from=label)
            if chosen is not marks[0]:
                _RESTORES_TRUNCATED.inc()
                log.error(
                    "the lost epoch change log was put back from only the tables' changes after a later mark, the "
                    "newest part that fit it. A follower whose mark is older than covered_from is behind (record_lost) "
                    "until it restarts",
                    extra={
                        "extra_data": {
                            "lost": lost,
                            "restored": incarnation,
                            "covered_from": _token(lost, chosen.seq),
                            "oldest_mark": _token(lost, marks[0].seq),
                        }
                    },
                )
        self._scrubbed = None
        self._seen = None
        _RESTORES.inc()
        log.warning(
            "the epoch change log was put back from the tables under a new incarnation. A follower whose mark is at or "
            "past covered_from walks every restored record and catches up; an older one is behind (record_lost) until "
            "it restarts",
            extra={
                "extra_data": {
                    "lost": lost,
                    "restored": incarnation,
                    "covered_from": None if label is None else _token(lost, label),
                    "versioned_tables": sum(not isinstance(table, StampedTable) for table in tables),
                    "stamped_tables": sum(isinstance(table, StampedTable) for table in tables),
                    "versions_from_epoch": None if chosen is None else _x(chosen),
                    "tables": 0 if changes is None else changes.tables,
                    "scope_reads": 0 if changes is None else changes.scopes,
                    "bucket_reads": 0 if changes is None else changes.buckets,
                    "records": appended,
                    "seconds": round(time.monotonic() - began, 3),
                }
            },
        )
        return incarnation

    async def _put_back(self, state: ChangeLogState, lost: str, folder: RestoreFolder) -> str:
        """a keeper beside the log's owner: append its own tables' changes into the restore of ``lost``.

        The owner declared the restore and labels it from its own database's tables (module docstring); this keeper puts
        back what only its database's tables hold, from its own oldest mark of the lost log (or the newest part that
        fits), and labels nothing. Interrupted before its mark of the new log, the next pass puts back again (a record
        appended twice is evicted twice).

        :param state: the restored log
        :ptype state: ChangeLogState
        :param lost: the lost log's incarnation, the one this keeper's newest mark names
        :ptype lost: str
        :param folder: the restore's worker
        :ptype folder: RestoreFolder
        :return: the restored log's incarnation
        :rtype: str
        :raises GenerationUnavailableError: when the log cannot be written; the next pass puts back again
        """
        incarnation = state.incarnation
        folder.start()
        began = time.monotonic()
        tables = await restored_tables(self._pool)
        marks = await self._marks_of(lost)
        budget = None if not state.max_bytes else state.max_bytes * RESTORE_BUDGET_RATIO - state.bytes
        chosen = await self._fitting(tables, marks, budget, folder) if marks and tables else None
        appended = 0
        if tables and chosen is None:
            _PUT_BACK_SHORT.labels(reason="too_large").inc()
            log.error(
                "a keeper beside the change log's owner did NOT put its tables' changes back into the restored log: "
                "it holds no mark of the lost log, or not even the changes after its newest mark fit what the log has "
                "left. A follower of this database's tables misses the epochs they name",
                extra={"extra_data": {"lost": lost, "restored": incarnation, "marks": len(marks), "budget": budget}},
            )
        elif chosen is not None:
            changes = TableChanges(self._pool, tables, _x(chosen), folder, self._database)
            appended = await self._log.put_back(changes, incarnation=incarnation, lost=lost)
        _PUT_BACKS.inc()
        log.warning(
            "a keeper beside the change log's owner put its own tables' changes back into the restored log",
            extra={
                "extra_data": {
                    "lost": lost,
                    "restored": incarnation,
                    "from_mark": None if chosen is None else _token(lost, chosen.seq),
                    "records": appended,
                    "seconds": round(time.monotonic() - began, 3),
                }
            },
        )
        self._unchecked = (incarnation, lost, None if chosen is None else chosen.seq)
        self._scrubbed = None
        self._seen = None
        after = await self._log.log_state()
        if after is not None:
            self._check_label(after)
        return incarnation

    def _check_label(self, state: ChangeLogState) -> None:
        """once the owner labels the restore this keeper put its changes back into, check what the label vouches for.

        The label vouches for every record after ``covered_from``, the owner's mark. A record of this database's tables
        after it and at or before this keeper's own mark may be missing from the put-back, so a follower whose mark lies
        between may have missed it: said at ERROR and counted. Checked by the process that put the changes back; a
        restart between its put-back and the label skips the check.

        :param state: the log as this pass read it
        :ptype state: ChangeLogState
        :return: nothing
        :rtype: None
        """
        if self._unchecked is None:
            return
        incarnation, lost, floor = self._unchecked
        if state.incarnation != incarnation:
            self._unchecked = None  # lost again before its label: that restore's label never comes
            return
        if state.continues != lost:
            return  # not yet labelled
        self._unchecked = None
        label = state.covered_from
        if label is None or (floor is not None and floor <= label):
            return
        _PUT_BACK_SHORT.labels(reason="floor_after_label").inc()
        log.error(
            "a restored change log's label vouches for records this keeper's put-back may not cover: its mark is after "
            "the label's covered_from. A follower of this database's tables whose mark lies between may have missed a "
            "write",
            extra={
                "extra_data": {
                    "lost": lost,
                    "restored": incarnation,
                    "covered_from": _token(lost, label),
                    "put_back_from": None if floor is None else _token(lost, floor),
                }
            },
        )

    async def _scrub(self, state: ChangeLogState) -> ScrubPass:
        """check the records after the scrub's position, up to the log's head one pass ago: each of a table a restore
        can read back, and each stamped at or above its mark's X. Checks only: a miss is logged and counted.

        :param state: the log
        :ptype state: ChangeLogState
        :return: what it found
        :rtype: ScrubPass
        """
        incarnation = state.incarnation
        seen = self._seen
        self._seen = (incarnation, state.last_seq)
        if seen is None or seen[0] != incarnation:
            return ScrubPass()
        began = self._scrubbed
        start = state.first_seq - 1
        if began is not None and began[0] == incarnation:
            start = max(start, began[1])
        end = min(seen[1], start + self._scrub_batch)
        if end <= start:
            _SCRUB_BEHIND.set(max(0, state.last_seq - start))
            return ScrubPass(position=_token(incarnation, start))
        schema_tables = {str(row["table_name"]) for row in await self._pool.fetch(_SCHEMA_TABLES_SQL, self._schema)}
        if self._keeps_log:
            read = await self._epochs_between(start, end)
            # WHY only this database's records (2026-11-16): the log is shared with another database's writers
            # (identity-core's), whose tables are in that database; read here, each of theirs would be judged against
            # this database's tables. A record naming its database is this keeper's exactly when it names this one;
            # only an older writer's, naming none, is judged by its table being of this keeper's schema
            records = [
                record
                for record in read
                if (record.database == self._database if record.database is not None else record.table in schema_tables)
            ]
            others = len(read) - len(records)
        else:
            # WHY by its tables' keys, never by sequence (2026-11-24, Pace): a read by sequence
            # ($JS.API.STREAM.MSG.GET) carries its subject in the request body, where no grant can narrow it, so a
            # pod's grant never names it (threetears.nats.subject_permissions) and every such read was refused and
            # waited out its 5 s deadline. A keeper beside judges only its own tables, so it reads only theirs
            records = await self._own_epochs_between(sorted(schema_tables), incarnation, start, end)
            # the range's other records, counted, never read: every sequence after the log's first is held
            others = max(0, end - start - len(records))
        result = ScrubPass(checked=len(records), position=_token(incarnation, end), others=others)
        result.margin_violations = await self._check_margins(incarnation, records)
        restorable = {table.table for table in await restored_tables(self._pool)} if records else set()
        for record in records:
            if record.table in restorable:
                continue
            result.missing += 1
            _MISSING.labels(table=record.table).inc()
            log.error(
                "a record the change log holds is of a table a restore cannot read back (no versioned or stamped table "
                "of the name carries an epoch): a restore after a NATS loss would not put it back",
                extra={"extra_data": {"table": record.table, "epoch": record.token}},
            )
        if self._scrubbed is began:
            # nothing reset or moved the position during the awaits above (a restore sets it to None): advance it.
            # Otherwise the next scrub starts from where that left it, so a reset is never undone
            self._scrubbed = (incarnation, end)
        _SCRUB_BEHIND.set(max(0, state.last_seq - end))
        return result

    async def _check_margins(self, incarnation: str, records: Sequence[EpochChange]) -> int:
        """the records after a keeper mark whose lowest stamp is below that mark's restore floor X: each is said at
        ERROR with its table, scopes and epochs, and counted (``threetears_restore_margin_violations_total``).

        A restore from a mark (P, T) reads the tables with ``epoch >= X``, X = (T - M) << 15
        (:func:`~threetears.core.ddl.row_epoch.restore_floor`); it is complete only while every record appended after P
        carries ``s_min >= X``. Each record is judged against the newest mark before it (the highest X a restore could
        start from), so a record that held under it held under every older one. A record naming no ``s_min`` stamped
        nothing.

        :param incarnation: the log's incarnation
        :ptype incarnation: str
        :param records: the records the scrub read
        :ptype records: Sequence[EpochChange]
        :return: the violations found
        :rtype: int
        """
        stamped = [record for record in records if record.s_min is not None]
        if not stamped:
            return 0
        marks = await self._marks_of(incarnation)
        violations = 0
        for record in stamped:
            seq = _seq(record)
            before = [mark for mark in marks if mark.seq < seq]
            if not before:
                continue
            mark = before[-1]
            floor = _x(mark)
            s_min = record.s_min
            assert s_min is not None  # narrow: filtered above
            if s_min >= floor:
                continue
            violations += 1
            _MARGIN_VIOLATIONS.labels(table=record.table).inc()
            log.error(
                "a change-log record after a keeper mark carries a stamp below the mark's restore floor: a restore from "
                "that mark would read its tables from X and miss what it wrote below X. Its writer held a transaction "
                "or its append longer than the margin M covers",
                extra={
                    "extra_data": {
                        "table": record.table,
                        "scopes": record.scopes,
                        "keys": [row.ids for row in record.rows[:5]],
                        "epoch": record.token,
                        "s_min": s_min,
                        "s_min_wall_ms": epoch_wall_ms(s_min),
                        "floor": floor,
                        "floor_wall_ms": epoch_wall_ms(floor),
                        "mark": _token(incarnation, mark.seq),
                    }
                },
            )
        return violations

    async def _epochs_between(self, start: int, end: int) -> list[EpochChange]:
        """the records the log holds after ``start`` up to ``end``, read :data:`_SCRUB_READS_AT_ONCE` at a time.

        :param start: the sequence after which to read
        :ptype start: int
        :param end: the last sequence to read
        :ptype end: int
        :return: the records held, in sequence order
        :rtype: list[EpochChange]
        """
        records: list[EpochChange] = []
        for first in range(start + 1, end + 1, _SCRUB_READS_AT_ONCE):
            batch = range(first, min(end, first + _SCRUB_READS_AT_ONCE - 1) + 1)
            read = await asyncio.gather(*(self._log.epoch_at(seq) for seq in batch))
            records.extend(record for record in read if record is not None)
        return records

    async def _own_epochs_between(
        self, tables: Sequence[str], incarnation: str, start: int, end: int
    ) -> list[EpochChange]:
        """a keeper beside's scrub read: its own tables' records after ``start`` up to ``end``, by each table's keys.

        Each table is read through :meth:`_ChangeLogKeeping.changes_after` (a consumer whose filter names the table's
        keys in its create subject, which a pod's grant names), :data:`_SCRUB_READS_AT_ONCE` tables at a time; the log
        is never read by sequence. A record naming another database is walked past by the read and again here.

        :param tables: the tables of this keeper's schema
        :ptype tables: Sequence[str]
        :param incarnation: the log's incarnation
        :ptype incarnation: str
        :param start: the sequence after which to read
        :ptype start: int
        :param end: the last sequence to read
        :ptype end: int
        :return: the records of this database's tables, in sequence order
        :rtype: list[EpochChange]
        :raises GenerationUnavailableError: when the log cannot be read
        """
        after, upto = _token(incarnation, start), _token(incarnation, end)
        walks: list[EpochWalk] = []
        for first in range(0, len(tables), _SCRUB_READS_AT_ONCE):
            batch = tables[first : first + _SCRUB_READS_AT_ONCE]
            walks.extend(await asyncio.gather(*(self._log.changes_after(table, after, upto) for table in batch)))
        records = [
            change
            for walk in walks
            for change in walk.changes
            if change.database is None or change.database == self._database
        ]
        return sorted(records, key=_seq)


async def run_log_keeper(held: HeldLease, keeper: LogKeeper, *, every: timedelta = _DEFAULT_KEEPER_EVERY) -> None:
    """run the keeper's pass every ``every``, for as long as ``held`` is held.

    The one keeper: take the lease (``KVLease.hold``) and hand it here. A pass that fails is logged at ERROR and the
    next pass runs at the same interval. A restore waits for this pass, so ``every`` bounds how long writes wait for a
    lost log (``EpochGenerationSource(restore_wait=...)`` must be longer).

    :param held: the lease that makes this process the keeper
    :ptype held: HeldLease
    :param keeper: the keeper
    :ptype keeper: LogKeeper
    :param every: the pause between passes
    :ptype every: timedelta
    :return: nothing, once the lease is lost
    :rtype: None
    """
    while held.lost_reason is None:
        try:
            await keeper.run_pass()
        # prawduct:allow prawduct/broad-except -- the keeper outlives one failed pass; the next runs at the same interval
        except Exception as exc:
            contain(exc, log=log, what="a log keeper pass failed; the next pass runs", level="error")
        await asyncio.sleep(every.total_seconds())


#: how often a process that does not hold a keeper's lease tries to take it: a lost holder is replaced within this
#: plus the lease's TTL
LOG_KEEPER_CLAIM_EVERY: Final = timedelta(seconds=15)

#: the keeper lease's TTL and renewal: a renewal every 10 s leaves room for two missed renewals
_KEEPER_LEASE_TTL: Final = timedelta(seconds=30)
_KEEPER_LEASE_RENEW: Final = timedelta(seconds=10)


class LogKeeping:
    """a keeper and the lease it runs under, for a process's lifespan: what every database's keeping process runs.

    The hub keeps the change log over its platform database; identity-core keeps its own database's part beside it
    (``LogKeeper(keeps_log=False)``). Each runs :meth:`start` once before it writes, then :meth:`claim_and_keep` every
    :data:`LOG_KEEPER_CLAIM_EVERY`, so one replica at a time keeps the log.

    :param keeper: the keeper over the process's L3 pool
    :ptype keeper: LogKeeper
    :param nats_client: the process's NATS client, for the lease in the ``leases`` bucket
    :ptype nats_client: KvCapable
    :param lease: the lease's name: one per database, since every keeping process shares the ``leases`` bucket
    :ptype lease: str
    :param key_scope: the owner scope the lease's key leads with, for a process granted only its own keys of the
        shared bucket (:class:`~threetears.core.coordination.lease.KVLease`); ``None`` for one granted the bucket
    :ptype key_scope: str | None
    :param create_if_missing: declare the ``leases`` bucket where absent; ``False`` for a process that only binds it
    :ptype create_if_missing: bool
    """

    def __init__(
        self,
        keeper: LogKeeper,
        nats_client: KvCapable,
        *,
        lease: str,
        key_scope: str | None = None,
        create_if_missing: bool = True,
    ) -> None:
        """capture the keeper and the lease's terms; no I/O.

        :param keeper: the keeper
        :ptype keeper: LogKeeper
        :param nats_client: the process's NATS client
        :ptype nats_client: KvCapable
        :param lease: the lease's name
        :ptype lease: str
        :param key_scope: the lease key's owner scope, or ``None``
        :ptype key_scope: str | None
        :param create_if_missing: declare the ``leases`` bucket where absent
        :ptype create_if_missing: bool
        :return: nothing
        :rtype: None
        """
        self.keeper = keeper
        self._nats = nats_client
        self._lease = lease
        self._key_scope = key_scope
        self._create_if_missing = create_if_missing

    async def start(self) -> KeeperPass:
        """the start's one pass, before anything of this process writes an epoch: create the marks table, then one
        keeper pass (the log's owner declares the log where it knows none, or puts a lost one back; a keeper beside it
        puts its own changes back into such a restore).

        Two replicas starting at once are safe: the log is declared or restored once, and the other finds it there; a
        keeper beside may put its changes back twice, and a record appended twice is evicted twice.

        :return: what the pass did
        :rtype: KeeperPass
        :raises GenerationUnavailableError: when the marks or the log cannot be read, or a restore failed
        """
        await self.keeper.install()
        done = await self.keeper.run_pass()
        if done.started is not None or done.restored is not None:
            log.info(
                "the log keeper's start declared the epoch change log",
                extra={"extra_data": {"started": done.started, "restored": done.restored}},
            )
        return done

    async def claim_and_keep(self, holder: str) -> None:
        """take the keeper's lease and run its passes until the lease is lost; return at once when another holds it.

        :param holder: this replica's name, so the lease names its holder
        :ptype holder: str
        :return: nothing
        :rtype: None
        """
        lease = KVLease(self._nats, pod_id=holder, create_if_missing=self._create_if_missing, key_scope=self._key_scope)
        try:
            held = await lease.hold(
                self._lease, ttl=_KEEPER_LEASE_TTL, renew_every=_KEEPER_LEASE_RENEW, name="the log keeper"
            )
        except LeaseUnavailable:
            # NOSILENT: another replica keeps the log; this one claims again at the next tick
            return
        log.info("this replica keeps the change log", extra={"extra_data": {"lease": self._lease}})
        try:
            await run_log_keeper(held, self.keeper)
        finally:
            await held.release()
        log.warning(
            "this replica stopped keeping the change log: its lease was lost",
            extra={"extra_data": {"lease": self._lease, "reason": f"{held.lost_reason}"}},
        )
