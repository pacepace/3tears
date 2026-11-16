"""every committed write leaves a change note in L3, in its own transaction, so a lost change log can be put back.

The epoch change log (``{ns}-epoch-changes``, memory storage in NATS, three replicas) is the only clock and
the only catch-up path: a follower walks it and fetches only the keys that changed, and nothing here
changes that. What it cannot survive alone is NATS losing it. So every noted table carries a trigger
(:mod:`threetears.core.ddl.change_notes`) that writes one **change note** per changed row -- the table, the
row's key and its declared invalidation columns, never its data -- into L3 in the same transaction as the
row. A write and its note commit together or not at all: no crash can leave one without the other. The
notes are read for one thing only, to put a lost log back (:meth:`ChangeNoteKeeper.run_pass`); a follower
never reads them to catch up.

**No contention by construction.** A note is an insert of a fresh row with a random uuid key, hash-sharded;
no shared counter, no sequence, no index on a rising value, no constraint across writers. Notes are not
ordered, and a restore does not need them to be.

**The ring.** Three note tables: the current one writers fill, the previous one, and one emptied and
waiting. The current and the previous are the coverage. One keeper (this module's
:class:`ChangeNoteKeeper`, under a lease) switches the ring when the current table has taken half the
restore budget: :data:`RESTORE_BUDGET_RATIO` (75 %) of the log's ``max_bytes``, over two -- measured from the
log itself as the bytes per sequence it holds times the sequences taken since the switch, never by counting
in L3. Why half: a restore reads the current and the previous table, and puts them back into a log it may
fill to 75 %; two tables of half that each fit whole. A whole log's worth per table would make the coverage
up to twice the log, and a restore on a busy day would fit only in part. It also switches when the
current table holds that size over 64 bytes in notes (``switch_notes``, counted every twentieth of the fence),
so a bulk raw-SQL write with one small epoch cannot pile notes behind the log's bytes; and a restore streams
the notes a page at a time, sized in the database, so the keeper's memory follows one page, not the
coverage. When a switch is held up (its wait, an emptying) and the coverage outgrows the budget, the
restore puts back its newest part instead (below). It
empties the oldest with batched deletes (:meth:`ChangeNoteKeeper.empty_slot`), never ``TRUNCATE``, and on Postgres
vacuums it before it counts as emptied. The ring
row in L3 names the current table; only the keeper writes it, once per switch.

**No save reads the ring row.** Each writing process keeps a cached copy of the current slot
(:class:`threetears.core.utils.change_note_slot.ChangeNoteSlot`), read at most every quarter of the fence and
never used once half the fence old, and each connection carries it as a session setting with its expiry
on the database's clock. The trigger takes its slot from that setting at a statement's first noted row,
and reads the ring row only when the setting is missing or expired; such a note is flagged
(``ring_read``), and the keeper counts them per table at each switch
(``threetears_change_notes_ring_reads_total``).

**Why emptying is safe.** A note goes into the slot its statement took at its first noted row, from a copy
read at most half the fence earlier (:func:`~threetears.core.ddl.change_notes.slot_max_age`, enforced by the
database's clock) or from the ring row itself. The ring row names a table current at the read. The keeper
never switches again until its **wait** has passed since the last switch -- at least the fence the writers
were told (``fence_seconds``, ten minutes by default), and at least the one told before the last switch,
so a copy read under an older fence is covered too -- and it empties only the idle table, the one switched
away from two switches back, at least one wait ago. So a copy naming it expired at least half a fence
before the emptying began, and any statement that took it had the other half to end: no write can still
reach the table being emptied. The emptying is recorded in the ring row (``emptied``), and the ring never
switches into a table whose emptying has not finished: such a switch waits, logged at ERROR on every pass it is held
(the current table grows past its planned size, so the restore budget is at risk) and counted
(``threetears_change_notes_switch_waits_total``).

**What the coverage vouches for.** When the wait after a switch has passed, the keeper reads the log's
position (``fence_seq``): every epoch appended after it is of a statement whose slot was read after the
switch, so its note is in the current table or a later one. At the next switch that position becomes the coverage's
**floor**: every epoch after the floor has its note in the current or the previous table, the two a
restore reads. An epoch whose writer stalled between its commit and its append for longer than the fence
breaks that promise, and the scrub reports it.

**A restore too big for the log.** When the coverage's records would take the new log past the budget, the
restore puts back the newest part that fits rather than nothing. The keeper marks the log's position every
twentieth of the fence (``threetears_change_note_marks``, by the database's clock, pruned at each switch). A
mark's position ``P`` at time ``T`` vouches that every epoch after ``P`` has its note written after ``T`` less
half the wait (a writing transaction ends within it). So the notes written since then, labelled with ``P`` as
their floor (``covered_from``), cover every epoch after ``P``: a follower at or past ``P`` catches up, an
older one is behind (``record_lost``). The oldest mark whose part fits is taken; the cut is said at ERROR and
counted (``threetears_change_notes_restore_truncated_total``). Only when not even the part after the newest
mark fits is the restore refused (``threetears_change_notes_restores_refused_total``).

**Putting a lost log back.** When the keeper finds the log absent it first declares a new log under a NEW
incarnation, marked as being restored from the lost one and continuing nothing yet, so a write waiting for
the log appends at once. Then it reads the notes of the current and the previous tables, folds them into
one record per table (each distinct key once; a whole-table note as a whole-table record), sizes them
against the log's bound, appends them, and only then labels the log as continuing the lost one from the
coverage's floor (``continues``, ``covered_from``). The label is the restore's completion: until it is
written a follower of the lost log is behind (``replaced``), never carried over to a partial log. A restore
interrupted before its label (a refused append, a lost lease, a crash) is found by the next pass and done
again in full into the same log, under fresh append ids. One too big for the log puts back its newest
part (above). Once labelled, a follower whose mark under the lost incarnation is at or past
the floor missed nothing the notes do not cover: it carries its mark to the start of the new log and walks
every record in it, the writes made during the restore included, evicting each key once more than it
strictly needed. One whose mark is older is behind (``record_lost``), serving what it holds, until it
restarts. Nothing drops a table it did not change; a whole-table record is the truncate or load that made
it, re-applied.

**The scrub.** Every tenth of the fence (``scrub_every``), not every pass, it checks that the epochs the
log holds after the floor have their notes: one whose note is not where the coverage needs it is logged at
ERROR and counted (``threetears_change_notes_missing_total``), for an alert. It reads up to ``scrub_batch``
records from the log, a few dozen at a time, and looks their notes up through the note tables'
``(key_text, table_name)`` index, so its cost follows the epochs checked, not the coverage's size. It only
checks and reports; it never writes a note, reloads anything or repairs.

**Which columns can be noted.** A note carries its key and invalidation columns as text, and a restore
names rows by that text, as an epoch record does. Only uuid, text, varchar and integer columns spell a
value exactly one way; a noted table with any other key or invalidation column gets no trigger, and an
ERROR names it.
"""

from __future__ import annotations

import asyncio
import json
import time
from bisect import bisect_right
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    Sequence,
)
from contextlib import AbstractAsyncContextManager, aclosing, nullcontext
from dataclasses import dataclass, field
from itertools import accumulate, groupby
from operator import itemgetter, methodcaller
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

import asyncpg
from threetears.nats.change_log import ChangeLogState
from threetears.observe import contain, get_logger
from threetears.observe.background import cancel_and_wait
from threetears.observe.metrics import counter, gauge, histogram

from threetears.core.change_note_fold import RestoreFolder
from threetears.core.collections.epoch_change import (
    RESET_REASON,
    ChangedRow,
    EpochChange,
    KeyedRecordForm,
    check_whole_reach,
    keyed_record_form,
)
from threetears.core.collections.generation import NotedTable, split_generation_token
from threetears.core.collections.noted_keys import NotedKeysCursor, NotedKeysPage, NotedKeysReader
from threetears.core.ddl.change_notes import (
    DEFAULT_NOTE_FENCE,
    MARKS_TABLE,
    NOTE_RING_SIZE,
    NOTE_TRIGGER,
    NOTE_TRIGGERS,
    NOTE_TRUNCATE_TRIGGER,
    legacy_truncate_trigger_drop_sql,
    RING_TABLE,
    change_notes_ddl,
    inexact_note_columns,
    note_function_name,
    note_table,
    note_trigger_sql,
    ring_read_sql,
    ring_switch_sql,
    ring_update_sql,
    trigger_function_body,
    trigger_function_sql,
)
from threetears.core.ddl.ddl_lock import DdlLockPolicy, database_ddl_lock
from threetears.core.keyed import (
    Derived,
    Step,
    Fragment,
    Selected,
    Sort,
    delete_before,
    distinct,
    keyset_rows,
    read_row,
    read_row_beside,
    scan,
    union,
)
from threetears.core.ddl.index_health import ddl_body, rebuild_unusable_indexes
from threetears.core.declared import whole_read
from threetears.core.ddl.errors import DdlLockTimeoutError
from threetears.core.ddl.session import ConnectionSession
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.sql_fragments import quote_identifier

if TYPE_CHECKING:
    from threetears.core.coordination.lease import HeldLease

__all__ = [
    "RESTORE_BUDGET_RATIO",
    "NotedRows",
    "noted_rows_since",
    "noted_keys_page",
    "NotedKeysCursor",
    "NotedKeysPage",
    "NotedKeysReader",
    "ChangeNoteKeeper",
    "KeeperPass",
    "NoteRingState",
    "NotedChange",
    "ScrubPass",
    "run_change_note_keeper",
]

log = get_logger(__name__)

#: the most of a change log's ``max_bytes`` a restore may fill, what the log already holds included. The
#: ring switches at half of it (:data:`_SWITCHES_PER_BUDGET`), so on a day the switch is not held up the
#: current and the previous table -- the coverage a restore reads -- fit it whole
RESTORE_BUDGET_RATIO: Final = 0.75

#: the ring's switch size, by default: the restore budget over this, so the coverage's two tables fit it
_SWITCHES_PER_BUDGET: Final = 2

#: marks of the log's position per fence: a suffix restore can start at one every fence / this
_MARKS_PER_FENCE: Final = 20

#: the columns a restore reads of the coverage's notes
_COVERAGE_COLUMNS: Final = "table_name, key_text, keys, columns, tenant, noted_at"

#: the marks a switch's sweep deletes a statement. Why a paged delete_before over the (at, id) index: the marks
#: table had no index but its id, so the old one-statement ``DELETE ... WHERE at < $1`` read every mark; a keeper
#: writes one mark per fence/20 and a switch drops the ones older than the coverage, about 20-40, so the sweep is one
#: statement in practice and 500 only bounds a sweep after a long outage. Measured (a local A/B on Postgres 17,
#: best of 5, 20 marks swept): 1.1 vs 1.5 ms at 60 marks, 1.4 vs 1.7 ms at 2,000, 2.8 vs 1.7 ms at 20,000
#: (old vs new); its plan test proves the index read on Postgres and YugabyteDB
#: (test_change_note_ring_plans::test_threetears_change_note_marks_at_reads_a_switchs_sweep).
_MARKS_SWEEP_PAGE: Final = 500

#: a lost log's marks, in position order (two marks may share a position: the id breaks the tie)
_MARKS_ORDER: Final = (Sort("seq"), Sort("id"))

#: notes deleted per statement when the keeper empties the idle table; each batch is its own transaction
_EMPTY_BATCH: Final = 1000

#: YugabyteDB's hash codes of a key: ``yb_hash_code`` answers 0 to 65535
_HASH_SPACE: Final = 65536

#: the batches' worth of notes the count that sizes an emptying's first band reads at most: past it the table is
#: taken to hold that many, so the first band is small and the bands grow from the notes each deletes
_BAND_COUNT_BATCHES: Final = 64


async def _install_change_notes(
    conn: Any, schema: str, *, fence: timedelta = DEFAULT_NOTE_FENCE, policy: DdlLockPolicy | None = None
) -> None:
    """create the ring where missing (:func:`~threetears.core.ddl.change_notes.change_notes_ddl`), every index of it
    usable: one a failed build left unusable is dropped and built again from the ring's own statement
    (:func:`~threetears.core.ddl.index_health.rebuild_unusable_indexes`, by the ring's own index names and
    definitions).

    An index build that fails part way leaves its index behind not valid, which the planner never reads and
    ``CREATE INDEX IF NOT EXISTS`` takes as present. On YugabyteDB 2026.1 an online build racing other DDL on the
    database was seen to end that way with no error: a ring's every secondary index left invalid, so each page of a
    commit's noted keys read its whole note table. The rebuild is said at WARNING; an index still unusable after it
    raises, rather than leave the ring's reads scanning.

    Every statement runs under the database's DDL lock, taken here on ``conn``
    (:func:`~threetears.core.ddl.ddl_lock.database_ddl_lock`; re-entrant, so a caller already holding it loses
    nothing): two index builds in one YugabyteDB database leave both invalid.

    :param conn: ONE connection allowed DDL, outside a transaction (YugabyteDB builds an index on a populated table
        online only there), never a pool: the lock is held on its session
    :ptype conn: Any
    :param schema: the schema holding the ring, a TRUSTED identifier
    :ptype schema: str
    :param fence: the fence a new ring row tells the writers
    :ptype fence: timedelta
    :param policy: how long to wait for the DDL lock; the default waits until it is free
    :ptype policy: DdlLockPolicy | None
    :return: nothing
    :rtype: None
    :raises IndexStillUnusableError: when a ring index is still unusable after it was built again
    """
    statements = change_notes_ddl(schema, fence=fence)
    # each index's name, as its own statement creates it (CREATE INDEX IF NOT EXISTS "<name>" ON ...), past the
    # ``-- purpose:`` lines that declare what reads it
    bodies = {statement: ddl_body(statement) for statement in statements}
    declared = {
        body.split()[5].strip('"'): statement for statement, body in bodies.items() if body.startswith("CREATE INDEX ")
    }
    async with database_ddl_lock(ConnectionSession(conn), policy):
        for statement in statements:
            await conn.execute(statement)
        # built again from the ring's own statement: a failed build may have left another definition under its name
        await whole_read(
            rebuild_unusable_indexes(conn, [schema], names=list(declared), policy=policy, definitions=declared),
            kind="schema_change",
            reason="the ring's own indexes a failed build left unusable are built again; none is unique, so no row is read",
            ruling="Pace 2026-11-12: a schema change may read whole; coordinator 2026-11-12: schema_change admits unknown-schema",
            proof="packages/core/tests/unit/ddl/test_change_notes_ddl.py::test_install_change_notes_rebuilds_no_unique_index",
            speed="packages/core/tests/integration/test_index_health_live.py::test_rebuild_speed_at_real_size",
        )


def _status_count(status: str) -> int:
    """the row count a command tag names (``DELETE 812``), 0 when it names none.

    :param status: the tag
    :ptype status: str
    :return: the count
    :rtype: int
    """
    last = status.rpartition(" ")[2]
    return int(last) if last.isdigit() else 0


#: the pause between two delete batches, so the emptying never crowds the writers
_EMPTY_PAUSE: Final = timedelta(milliseconds=20)

#: how long one keeper pass keeps emptying before it hands over to the next pass (a restore must not wait)
_EMPTY_BUDGET: Final = timedelta(seconds=1)

#: the pause between keeper passes: a write waits for a lost log's restore up to one pass, well inside
#: a source's ``restore_wait``
_DEFAULT_KEEPER_EVERY: Final = timedelta(seconds=1)

#: how long a trigger install waits for the database's DDL lock before deferring to the next call
_LOCK_POLICY: Final = DdlLockPolicy(poll_interval=0.5, max_wait=5.0)

#: epochs the scrub checks per pass
_SCRUB_BATCH: Final = 1000

#: how many scrubs run per fence by default: every minute at the 10 min fence
_SCRUBS_PER_FENCE: Final = 10

#: how many log records the scrub reads at once
_SCRUB_READS_AT_ONCE: Final = 64

#: the separator a note's key columns are joined with to compare them in SQL: never in a key's text form
_KEY_SEPARATOR: Final = "\x1f"

#: the bytes of keyed notes (``COPY`` CSV lines) a restore hands its worker at once (:mod:`threetears.core.
#: change_note_fold`). WHY 1 MiB (L1, 2026-11-15): the loop copies each page once to hand it over (~0.1 ms a MiB), so
#: a page well under a MiB keeps every turn far inside the loop's 1 ms; and a page is a few thousand notes, so the
#: worker's fixed cost a page (the hand-off, the job, the claim) stays small beside the notes it folds
_RESTORE_PAGE_BYTES: Final = 1 << 20

#: the most keyed-note bytes the restore holds while its worker folds a page: the ``COPY`` waits past it
_RESTORE_BUFFER_BYTES: Final = 4 * _RESTORE_PAGE_BYTES

#: distinct rows of one table per change a restore streams: the most it holds at once (a chunk is cut only where the
#: key changes, so a key noted with several invalidation columns may carry it a few past)
_RESTORE_CHUNK_ROWS: Final = 5000

#: restored changes handed on between two turns of the loop when the reader takes them without waiting (a test's)
_RESTORE_TURN_CHANGES: Final = 32

#: a keyed record's form, for the restore's worker to assemble records in (:func:`keyed_record_form`)
_KEYED_FORM: Final[KeyedRecordForm] = keyed_record_form()

#: a restored row's bytes in its record beyond its key and columns, as the restore's estimate: generous,
#: so a restore chosen to fit does fit
_RESTORED_ROW_ESTIMATE: Final = 64

#: a restored table's record overhead, as the restore's estimate
_RESTORED_TABLE_ESTIMATE: Final = 256

#: the least bytes one note adds to a restore (its row's overhead): the ring also switches when the current
#: table holds the switch size over this many notes, so a bulk write with one small epoch (a raw-SQL
#: update, a migration) cannot pile notes behind the log's bytes
_NOTE_BYTES_FLOOR: Final = _RESTORED_ROW_ESTIMATE

_MISSING = counter(
    "threetears_change_notes_missing_total",
    description="epochs the change log holds after the coverage floor whose change note is not in L3; alert on any",
    label_names=("table",),
)
_SWITCHES = counter(
    "threetears_change_notes_switches_total",
    description="times the note ring switched to its next table",
)
_EMPTIED = counter(
    "threetears_change_notes_emptied_total",
    description="note tables the keeper emptied (batched deletes) once nothing could still write to them",
)
_EMPTIED_ROWS = counter(
    "threetears_change_notes_emptied_rows_total",
    description="notes the keeper deleted while emptying the ring's idle table",
)
_SWITCH_WAITS = counter(
    "threetears_change_notes_switch_waits_total",
    description="ring switches that were due but waited because the table to switch into was not yet empty",
)
# A held switch is loud on purpose: while the switch waits for the idle table's emptying, the current note table
# keeps taking every write's note past the size the ring planned for it (switch_at), and a restore reads the current
# and the previous tables whole -- so the longer a switch is held, the closer the coverage comes to not fitting the
# log's restore budget (RESTORE_BUDGET_RATIO), when a lost log could no longer be put back whole. Every held pass logs
# at ERROR, and these say for how long and how far past plan (metrics only; Grafana alerts on them).
_SWITCH_HELD_SINCE = gauge(
    "threetears_change_notes_switch_held_since_timestamp_seconds",
    description="when the due ring switch began waiting for the idle table's emptying (unix seconds); 0 when not held",
)
_CURRENT_BYTES = gauge(
    "threetears_change_notes_current_bytes",
    description="bytes of change records the log took since the ring's last switch: the current note table's size",
)
_SWITCH_AT_BYTES = gauge(
    "threetears_change_notes_switch_at_bytes",
    description="the bytes at which the ring plans to switch its current note table (its planned size)",
)
_RING_READS = counter(
    "threetears_change_notes_ring_reads_total",
    description="notes written by a statement that read the ring row (no valid cached slot), counted at each switch",
    label_names=("table",),
)
_RESTORES = counter(
    "threetears_change_notes_restores_total",
    description="lost change logs the keeper put back from the change notes",
)
_RESTORES_REFUSED = counter(
    "threetears_change_notes_restores_refused_total",
    description="lost change logs NOT restored because the restore would not fit the log with headroom; alert on any",
)
_RESTORES_TRUNCATED = counter(
    "threetears_change_notes_restore_truncated_total",
    description="restores that put back only the newest notes that fit the log; followers older than its floor are behind",
)
_KEEPER_LAST_PASS = gauge(
    "threetears_change_notes_keeper_last_pass_timestamp_seconds",
    description="unix time of this process's last whole change-note keeper pass; alert when no process has moved it "
    "for several fences (no keeper holds the lease, or every pass fails)",
)
_UNTRIGGERED = gauge(
    "threetears_change_notes_untriggered_tables",
    description="tables of another schema than the keeper's that share a noted table's name and carry no change-note "
    "trigger, at the keeper's last install check: the keeper gives none (its classes declare its own schema's "
    "tables); their schema's migration run does. Alert on any that stays",
)
# WHY (R1, 2026-11-15, Pace's question: is a slot's deleted-row debris cleared before the ring switches back into it?).
# No. The keeper empties a slot with batched deletes; on YugabyteDB each deleted note stays a tombstone until a
# compaction drops it, and a compaction may not drop one younger than timestamp_history_retention_interval_sec (900 s,
# the default, on the local b118) -- longer than the 10 min wait between switches -- nor runs at all on a table whose
# reads see under auto_compact_percent_obsolete (99 %) obsolete keys. Measured (test_reused_note_slot_live.py, YB
# 2026.1.0.0-b118 lane 1 under measure-lock; a slot filled to the switch size at the 64 MiB log, 393,216 notes, emptied
# by empty_slot, then holding 25,500 new notes, against two never-filled slots with the same notes): this count
# 62.3 ms against 6.4 ms (9.7x; the A/A gap 0.95x), a bare full scan 12.9x, and the same 5 min later (9.5x, 13.4x).
# Postgres 16 (integ095-pg, same test): 5.6 against 1.6 ms (3.4x) at once, 1.5x five minutes later as autovacuum
# reached it. The indexed reads are not slower on either (noted_rows_since 1.01x YB, 0.74x PG; noted_keys_page 1.01x,
# 1.00x; the scrub's look 0.55x/1.16x, inside its noise): their (table_name, noted_at) range starts past the
# tombstones' times. So the cost falls on the reads that walk the whole current slot: this count, every twentieth of
# the fence, and a restore's coverage read. This histogram is that cost, by how long the slot has been current, so a
# reused slot's debris shows as a first fence slower than the later ones (metrics only; no alert rule).
# Since V1 (Pace 2026-11-15) the keeper vacuums an emptied slot on Postgres before the ring may switch back into it, and
# the reused slot there reads as a fresh one (ChangeNoteKeeper._vacuum_emptied); YugabyteDB keeps its tombstones.
_CURRENT_COUNT_SECONDS = histogram(
    "threetears_change_notes_current_count_seconds",
    description="seconds the keeper's count of the current note table took (a walk of the whole slot, debris "
    "included), by how long the slot has been current: first_fence, fences_2_to_3, later",
    label_names=("since_switch",),
)
_CURRENT_COUNT_NOTES = gauge(
    "threetears_change_notes_current_count_notes",
    description="notes the keeper's last count of the current note table found (at most the switch's note count): "
    "what threetears_change_notes_current_count_seconds read besides any debris",
)
_SCRUB_BEHIND = gauge(
    "threetears_change_notes_scrub_behind",
    description="epochs the change log holds after the scrub's position: how far the scrub is behind",
)


@dataclass(frozen=True, slots=True)
class NotedChange:
    """what the notes of one table say changed, folded for a restore: its distinct rows, or the whole table.

    :ivar table: the table
    :ivar rows: each distinct row its notes name, with the invalidation columns they carry
    :ivar whole_table: a note of the whole table (a truncate, a file load) is among them; :attr:`rows`
        is then empty, since the whole-table record covers every row
    :ivar tenant: the tenant the rows belong to (:mod:`threetears.core.tenancy`), whose key the
        record is restored under; ``None`` for ``_all``
    :ivar prefixes: for a whole-table change, the key prefixes its notes named (a load's,
        :data:`~threetears.core.ddl.change_notes.NOTE_REACH_SETTING`): the restored record names them, so a follower
        reads each as a key range. Rows outside them come back as their own changes
    :ivar reason: for a whole-table change, the reason its notes named: only an operator's reset
        (:data:`~threetears.core.collections.epoch_change.RESET_REASON`), restored as a reset
    :ivar bodies: for a keyed change folded in a worker process (:mod:`threetears.core.change_note_fold`), its rows'
        records as the log stores them, each within the record bound; :attr:`rows` is then empty
    """

    table: str
    rows: tuple[ChangedRow, ...] = ()
    whole_table: bool = False
    tenant: str | None = None
    prefixes: tuple[Mapping[str, str], ...] = ()
    reason: str | None = None
    bodies: tuple[bytes, ...] = ()

    def __post_init__(self) -> None:
        check_whole_reach(self.prefixes or None, self.reason, whole_table=self.whole_table)
        if self.bodies and (self.rows or self.whole_table):
            raise ValueError("a change carrying its rows' records names no rows itself and is not whole-table")


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

        :param changes: what the notes say changed, per table, held or streamed
        :ptype changes: Sequence[NotedChange] | AsyncIterable[NotedChange]
        :param incarnation: the restoring log
        :ptype incarnation: str
        :param lost: the lost log's incarnation
        :ptype lost: str
        :param covered_from: the lost log's sequence from which the notes cover every epoch; ``None``
            when they vouch for none
        :ptype covered_from: int | None
        :return: the records appended
        :rtype: int
        :raises RestoreTooLargeError: when the restore would not fit with headroom; nothing was appended
        :raises GenerationUnavailableError: when the log cannot be written
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
class NoteRingState:
    """the ring row, as one read saw it.

    :ivar current: the note table writers fill, ``0`` to ``2``
    :ivar switch_no: how many times the ring has switched
    :ivar switched_at: when it last switched, by the database's clock
    :ivar switched_seq: the change log's position at that switch, ``incarnation:seq``
    :ivar floor: every epoch after it has its note in the current or the previous table; ``None`` while
        the ring has not yet been through a fence
    :ivar fence_seq: the log's position once the fence after the last switch passed: every epoch after
        it has its note in the current table or a later one, and it becomes the floor at the next switch;
        ``None`` while the fence is still running
    :ivar emptied: whether the table after the current one has been emptied since the switch
    :ivar log_incarnation: the change log's incarnation the keeper last saw; the one a restore continues
    :ivar now: the database's clock at the read
    :ivar fence_seconds: the fence the writers are told: a cached slot is used at most half of it after its read
    :ivar wait_seconds: the least time from the last switch to the next, covering the fence told before it too
    """

    current: int
    switch_no: int
    switched_at: datetime
    switched_seq: str | None
    floor: str | None
    fence_seq: str | None
    emptied: bool
    log_incarnation: str | None
    now: datetime
    fence_seconds: float = DEFAULT_NOTE_FENCE.total_seconds()
    wait_seconds: float = DEFAULT_NOTE_FENCE.total_seconds()

    @property
    def previous(self) -> int:
        """the table before the current one: the rest of the coverage.

        :return: its place in the ring
        :rtype: int
        """
        return (self.current - 1) % NOTE_RING_SIZE

    @property
    def oldest(self) -> int:
        """the table after the current one: emptied after each switch, and the next current.

        :return: its place in the ring
        :rtype: int
        """
        return (self.current + 1) % NOTE_RING_SIZE


@dataclass(slots=True)
class ScrubPass:
    """what one scrub found.

    :ivar checked: epochs checked
    :ivar missing: epochs whose note was not where the coverage needs it
    :ivar position: the newest epoch checked, ``incarnation:seq``; ``None`` when none was
    """

    checked: int = 0
    missing: int = 0
    position: str | None = None


@dataclass(slots=True)
class KeeperPass:
    """what one keeper pass did.

    :ivar started: the incarnation of a log the keeper declared because none was there and no ring had
        ever seen one
    :ivar restored: the incarnation of a lost log the keeper put back from the notes
    :ivar adopted: the incarnation of a log the keeper found that it had not seen before
    :ivar fenced: the fence after the last switch passed in this pass
    :ivar emptied: the note table this pass finished emptying
    :ivar emptied_rows: notes this pass deleted from the idle table
    :ivar switch_waited: a switch was due but the table to switch into was not yet empty
    :ivar switched_to: the note table this pass made current
    :ivar scrub: what the scrub found
    """

    started: str | None = None
    restored: str | None = None
    adopted: str | None = None
    fenced: bool = False
    emptied: int | None = None
    emptied_rows: int = 0
    switch_waited: bool = False
    switched_to: int | None = None
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


def _seq_under(token: str | None, incarnation: str) -> int | None:
    """the sequence ``token`` names under ``incarnation``, or ``None`` when it is under another or none.

    :param token: ``incarnation:seq``, or ``None``
    :ptype token: str | None
    :param incarnation: the incarnation it must be under
    :ptype incarnation: str
    :return: the sequence
    :rtype: int | None
    """
    parts = None if token is None else split_generation_token(token)
    return parts[1] if parts is not None and parts[0] == incarnation else None


def _taken_bytes(state: ChangeLogState, switched_seq: str | None) -> int:
    """how many bytes of records the log has taken since the switch, from the log's own measure.

    The bytes per sequence the log holds now, times the sequences appended since the switch. A switch
    under another incarnation counts every sequence of this one.

    :param state: the log's state
    :ptype state: ChangeLogState
    :param switched_seq: the log's position at the switch
    :ptype switched_seq: str | None
    :return: the bytes
    :rtype: int
    """
    held = state.last_seq - state.first_seq + 1
    if held <= 0 or state.bytes <= 0:
        return 0
    since = _seq_under(switched_seq, state.incarnation)
    taken = state.last_seq - (since if since is not None else 0)
    return int(state.bytes / held * max(0, taken))


def _since_switch(ring: NoteRingState) -> str:
    """how long the ring's current table has been current, in fences, as a bounded label: ``first_fence`` (a slot
    switched back into is read here while its last emptying's debris is youngest), ``fences_2_to_3``, ``later``.

    :param ring: the ring
    :ptype ring: NoteRingState
    :return: the label
    :rtype: str
    """
    fences = (ring.now - ring.switched_at).total_seconds() / max(ring.fence_seconds, 1e-9)
    if fences < 1:
        return "first_fence"
    if fences < 3:
        return "fences_2_to_3"
    return "later"


def _switch_threshold(max_bytes: int | None) -> int | None:
    """the ring's default switch size: half the restore budget of a log of ``max_bytes``.

    The coverage a restore reads is the current and the previous table, so two tables of this size fill the
    budget (:data:`RESTORE_BUDGET_RATIO` of the log) and no more: on a day the switch is not held up by its
    wait or an emptying, a lost log is put back whole.

    :param max_bytes: the log's bound; ``None`` when unbounded
    :ptype max_bytes: int | None
    :return: the bytes of records the current table takes before the ring switches; ``None`` never
    :rtype: int | None
    """
    if not max_bytes:
        return None
    return max(1, int(max_bytes * RESTORE_BUDGET_RATIO / _SWITCHES_PER_BUDGET))


def _columns_of(raw: Any) -> tuple[tuple[str, str | None], ...]:
    """a note's invalidation columns, as a hashable pairing; empty when it carries none.

    :param raw: the ``columns`` value read: a mapping, its JSON text (a driver with no jsonb codec), or ``None``
    :ptype raw: Any
    :return: the pairs, in name order
    :rtype: tuple[tuple[str, str | None], ...]
    :raises GenerationUnavailableError: when the value is not an object
    """
    if raw is None:
        return ()
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict):
        raise GenerationUnavailableError(f"a change note's columns are not an object: {raw!r}")
    return tuple(sorted((str(name), None if each is None else str(each)) for name, each in value.items()))


_NOTED_TABLES_SQL: Final = """
SELECT t.table_schema::text AS table_schema, t.table_name::text AS table_name FROM information_schema.tables t
WHERE t.table_type = 'BASE TABLE' AND t.table_name::text = ANY($1::text[])
ORDER BY 1, 2
"""

_TRIGGERS_SQL: Final = """
SELECT n.nspname::text AS table_schema, c.relname::text AS table_name, g.tgname::text AS trigger_name
FROM pg_trigger g JOIN pg_class c ON c.oid = g.tgrelid JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE g.tgname::text = ANY($1::text[])
"""

_COLUMN_TYPES_SQL: Final = """
SELECT c.table_schema::text AS table_schema, c.table_name::text AS table_name, c.column_name::text AS column_name,
       c.data_type::text AS data_type
FROM information_schema.columns c WHERE c.table_name::text = ANY($1::text[])
"""

_FUNCTIONS_SQL: Final = """
SELECT p.proname::text AS name, p.prosrc AS body FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname::text = $1 AND p.proname::text = ANY($2::text[])
"""


@dataclass(frozen=True, slots=True)
class _Install:
    """what one table still needs for its notes: its function, and which of its two triggers."""

    table_schema: str
    noted: NotedTable
    function: bool
    triggers: tuple[str, ...]
    #: each column's ``data_type``: how the function names a column's value (a ``timestamptz`` canonically)
    kinds: dict[str, str] = field(default_factory=dict)


class ChangeNoteKeeper:
    """the one keeper of the note ring: switches it, empties the oldest table, puts a lost log back, and scrubs.

    Run :meth:`run_pass` from one process at a time (a :class:`~threetears.core.coordination.lease.KVLease`
    held around it); two at once switch no twice, since every ring update names the switch it read, but
    would empty and scrub twice. Run :meth:`install_triggers` beside it, as the write marks' install
    backstop runs, so a table created since is noted.

    Every process whose change log a restore may put back builds its generation source with
    ``restored_from_l3=True``, so a write after a loss waits for this keeper rather than starting a log
    no follower can continue from.
    """

    def __init__(
        self,
        pool: _Pool,
        *,
        notes_schema: str,
        log: _ChangeLogKeeping,
        fence: timedelta = DEFAULT_NOTE_FENCE,
        scrub_batch: int = _SCRUB_BATCH,
        scrub_every: timedelta | None = None,
        switch_bytes: int | None = None,
        switch_notes: int | None = None,
        empty_batch: int = _EMPTY_BATCH,
        empty_pause: timedelta = _EMPTY_PAUSE,
        empty_budget: timedelta = _EMPTY_BUDGET,
    ) -> None:
        """capture what the keeper works between; no I/O.

        :param pool: the L3 pool holding the ring, allowed DDL (installing and emptying take the DDL lock)
        :ptype pool: _Pool
        :param notes_schema: the schema holding the ring (the platform schema), a TRUSTED identifier
        :ptype notes_schema: str
        :param log: the change log the notes cover
        :ptype log: _ChangeLogKeeping
        :param fence: how long after a switch the ring may switch again, and the fence the writers are told
            at each switch: a cached slot is used at most half of it, and a writing transaction must end
            within the other half
        :ptype fence: timedelta
        :param scrub_batch: epochs the scrub checks per pass; at least one
        :ptype scrub_batch: int
        :param scrub_every: the least time between two scrubs; ``None`` (the default) is a tenth of the fence.
            The scrub reads the log and the note tables, so it runs on its own interval, not every pass
        :ptype scrub_every: timedelta | None
        :param switch_bytes: how many bytes of change records the current table takes before the ring
            switches; ``None`` (the default) is half the restore budget, ``max_bytes`` × 75 % / 2, so the
            coverage's two tables fit a restore whole. Never set it below what the log takes between two
            keeper passes
        :ptype switch_bytes: int | None
        :param switch_notes: how many notes the current table holds before the ring switches, whatever the
            log's bytes; ``None`` (the default) is the switch size over :data:`_NOTE_BYTES_FLOOR`. Counted
            every twentieth of the fence, stopping at the threshold
        :ptype switch_notes: int | None
        :param empty_batch: notes deleted per statement while emptying the idle table; at least one
        :ptype empty_batch: int
        :param empty_pause: the pause between two delete batches
        :ptype empty_pause: timedelta
        :param empty_budget: how long one pass keeps emptying; it always deletes at least one batch
        :ptype empty_budget: timedelta
        :return: nothing
        :rtype: None
        :raises ValueError: when the fence is not positive, a batch is under one or the switch size is
        """
        if (
            fence <= timedelta(0)
            or scrub_batch < 1
            or empty_batch < 1
            or (switch_bytes is not None and switch_bytes < 1)
        ):
            raise ValueError(
                "a keeper's fence is positive, its scrub checks and its emptying deletes at least one a batch, "
                "and its switch size is positive"
            )
        self._pool = pool
        self._schema = notes_schema
        self._owner = quote_identifier(notes_schema)
        self._log = log
        self._fence = fence
        self._scrub_batch = scrub_batch
        #: the noted tables refused a trigger for a column type, so each is said once
        self._refused: set[str] = set()
        #: another schema's tables of a noted name found without a note trigger, so each is said once
        self._untriggered: set[str] = set()
        self._scrub_every = fence / _SCRUBS_PER_FENCE if scrub_every is None else scrub_every
        self._scrubbed_at: float | None = None
        self._marks = f"{self._owner}.{quote_identifier(MARKS_TABLE)}"
        #: the same, as the keyed helpers name a table (``schema.table``, quoted a segment at a time by them)
        self._marks_table = f"{notes_schema}.{MARKS_TABLE}"
        self._mark_every = fence / _MARKS_PER_FENCE
        self._marked_at: float | None = None
        self._switch_notes = switch_notes
        #: whether the current table held the switch's note count at the last count, and when that was
        self._notes_full = False
        self._counted_at: float | None = None
        self._switch_bytes = switch_bytes
        self._empty_batch = empty_batch
        self._empty_pause = empty_pause
        self._empty_budget = empty_budget
        #: the switch number whose due switch already waited for an emptying, so it is logged once
        self._waited_switch: int | None = None
        #: when (unix seconds) the due switch began waiting for the idle table's emptying; None when not held
        self._held_since: float | None = None
        #: the current table's notes at the last count (capped at the switch's note count: a count never reads more)
        self._notes_seen = 0
        #: the newest epoch the scrub has checked, and the log's head at the previous pass (what it checks up to)
        self._scrubbed: tuple[str, int] | None = None
        self._seen: tuple[str, int] | None = None
        #: whether the ring's database is YugabyteDB, read once: an emptying steps through the notes in its key's
        #: own order, which there is the key's hash
        self._yugabyte: bool | None = None
        #: the table being emptied (one at a time: the ring's idle table) and where its emptying has reached: the
        #: next hash band's start (YugabyteDB) or the last id deleted (Postgres), so the next batch, or the next
        #: pass, starts past the notes already gone; None before a walk starts
        self._emptying_at: tuple[int, Any] | None = None
        #: the hash codes a YugabyteDB emptying's next statement covers, sized to about a batch of notes
        self._band = _HASH_SPACE // 1024

    def _ring_table(self, index: int) -> str:
        """the qualified name of the ring's note table at ``index``.

        :param index: its place in the ring
        :ptype index: int
        :return: the name
        :rtype: str
        """
        return f"{self._owner}.{quote_identifier(note_table(index))}"

    async def install(self, policy: DdlLockPolicy | None = None) -> None:
        """create the ring where missing, under the database's DDL lock; a ring already there keeps its state.

        :param policy: how long to wait for the DDL lock; the default waits until it is free
        :ptype policy: DdlLockPolicy | None
        :return: nothing
        :rtype: None
        """
        async with self._pool.acquire() as conn:
            await _install_change_notes(conn, self._schema, fence=self._fence, policy=policy)

    async def install_triggers(self, tables: Iterable[NotedTable], policy: DdlLockPolicy = _LOCK_POLICY) -> list[str]:
        """give every noted table of the keeper's own schema its note function and triggers; say which it gave.

        Only the keeper's schema (the platform schema, where this process's classes declare their tables): a table
        of the same name in another schema is another declaration's (a pod's table in its agent schema), whose key
        and tenancy these classes do not know. Its triggers come from its own schema's migration run
        (:func:`~threetears.core.data.migrations.change_notes.give_noted_tables_their_note_trigger`, every run); one
        without them is said here, once, and counted (``threetears_change_notes_untriggered_tables``), never given.

        Reads the catalogue first and returns at once when every table has its function, as the table's
        declaration writes it, and both triggers. Otherwise takes the DDL lock with ``policy``'s bounded
        wait, reads again under it and creates what is still missing; a function whose key or columns
        changed is replaced. A lock that stays busy defers the install to the next call.

        :param tables: the noted tables (:func:`~threetears.core.collections.generation.noted_tables`)
        :ptype tables: Iterable[NotedTable]
        :param policy: how long to wait for the DDL lock
        :ptype policy: DdlLockPolicy
        :return: ``schema.table`` for each table this call gave anything
        :rtype: list[str]
        """
        noted = {each.table: each for each in tables}
        if not noted or not await self._installs(self._pool, noted):
            return []
        given: list[str] = []
        async with self._pool.acquire() as conn:
            try:
                async with database_ddl_lock(ConnectionSession(conn), policy):
                    for install in await self._installs(conn, noted):
                        name = f"{install.table_schema}.{install.noted.table}"
                        try:
                            await self._give(conn, install)
                        # prawduct:allow prawduct/broad-except -- a table dropped meanwhile; logged, and the next install retries
                        except Exception as exc:
                            contain(
                                exc,
                                log=log,
                                what="could not give a noted table its change-note trigger; its writes leave no note",
                                extra={"table": name, "error": f"{type(exc).__name__}: {exc}"},
                                level="error",
                            )
                            continue
                        given.append(name)
            except DdlLockTimeoutError:
                log.info("change-note trigger install deferred: the database's DDL lock is busy")
                return []
        if given:
            log.info("change-note triggers installed", extra={"extra_data": {"tables": given}})
        return given

    async def _installs(self, reader: Any, noted: dict[str, NotedTable]) -> list[_Install]:
        """what each noted table of the keeper's schema still needs; another schema's table of a noted name with no
        note trigger is said and counted (:meth:`install_triggers`).

        :param reader: the pool, or one connection
        :ptype reader: Any
        :param noted: the noted tables by name
        :ptype noted: dict[str, NotedTable]
        :return: the work, in schema and table order
        :rtype: list[_Install]
        """
        named = [
            (str(row["table_schema"]), str(row["table_name"]))
            for row in await reader.fetch(_NOTED_TABLES_SQL, sorted(noted))
        ]
        triggers = {
            (str(row["table_schema"]), str(row["table_name"]), str(row["trigger_name"]))
            for row in await reader.fetch(_TRIGGERS_SQL, list(NOTE_TRIGGERS))
        }
        present = [(schema, table) for schema, table in named if schema == self._schema]
        self._say_untriggered(
            [
                (schema, table)
                for schema, table in named
                if schema != self._schema and (schema, table, NOTE_TRIGGER) not in triggers
            ]
        )
        types: dict[tuple[str, str], dict[str, str]] = {}
        for row in await reader.fetch(_COLUMN_TYPES_SQL, sorted(noted)):
            types.setdefault((str(row["table_schema"]), str(row["table_name"])), {})[str(row["column_name"])] = str(
                row["data_type"]
            )
        names = [note_function_name(schema, table) for schema, table in present]
        bodies = {str(row["name"]): str(row["body"]) for row in await reader.fetch(_FUNCTIONS_SQL, self._schema, names)}
        work: list[_Install] = []
        for schema, table in present:
            each = noted[table]
            tenant_columns = (each.tenancy.column,) if each.tenancy.column else ()
            kinds = types.get((schema, table), {})
            refused = inexact_note_columns(kinds, (*each.keys, *each.columns, *tenant_columns))
            if refused:
                self._refuse(schema, table, refused)
                continue
            wanted = trigger_function_body(
                notes_schema=self._schema, keys=each.keys, columns=each.columns, tenancy=each.tenancy, kinds=kinds
            )
            stale = bodies.get(note_function_name(schema, table)) != wanted
            missing = tuple(name for name in NOTE_TRIGGERS if (schema, table, name) not in triggers)
            if stale or missing:
                work.append(_Install(table_schema=schema, noted=each, function=stale, triggers=missing, kinds=kinds))
        return work

    def _say_untriggered(self, tables: Sequence[tuple[str, str]]) -> None:
        """count another schema's tables of a noted name that carry no note trigger, and say each once.

        :param tables: each such table's schema and name
        :ptype tables: Sequence[tuple[str, str]]
        :return: nothing
        :rtype: None
        """
        _UNTRIGGERED.set(len(tables))
        for schema, table in tables:
            name = f"{schema}.{table}"
            if name in self._untriggered:
                continue
            self._untriggered.add(name)
            log.warning(
                "a table shares a noted table's name in another schema and carries no change-note trigger; the "
                "keeper gives triggers only in its own schema. Its schema's migration run gives them, when that "
                "schema's classes note it",
                extra={"extra_data": {"table": name, "keeper_schema": self._schema}},
            )

    def _refuse(self, schema: str, table: str, refused: Sequence[tuple[str, str]]) -> None:
        """say, once per table, that it gets no note trigger: a key or invalidation column is of a type whose
        text form does not round-trip exactly, so its note could name a row another way than its epoch.

        :param schema: the table's schema
        :ptype schema: str
        :param table: the table
        :ptype table: str
        :param refused: each refused column and its type
        :ptype refused: Sequence[tuple[str, str]]
        :return: nothing
        :rtype: None
        """
        name = f"{schema}.{table}"
        if name in self._refused:
            return
        self._refused.add(name)
        log.error(
            "a noted table gets no change-note trigger: a key or invalidation column is not uuid, text, varchar, "
            "an integer or timestamptz, so its text form does not name the value exactly. Change the column or what the class "
            "declares; the scrub reports its epochs as missing notes",
            extra={"extra_data": {"table": name, "columns": [f"{column} ({kind})" for column, kind in refused]}},
        )

    async def _give(self, conn: Any, install: _Install) -> None:
        """create one table's function, when stale, and its missing triggers.

        :param conn: the connection holding the DDL lock
        :ptype conn: Any
        :param install: what the table needs
        :ptype install: _Install
        :return: nothing
        :rtype: None
        """
        each = install.noted
        if install.function:
            await conn.execute(
                trigger_function_sql(
                    notes_schema=self._schema,
                    table_schema=install.table_schema,
                    table=each.table,
                    keys=each.keys,
                    columns=each.columns,
                    tenancy=each.tenancy,
                    kinds=install.kinds,
                )
            )
        statements = note_trigger_sql(notes_schema=self._schema, table_schema=install.table_schema, table=each.table)
        if NOTE_TRUNCATE_TRIGGER in install.triggers:
            # a table given its triggers before a truncate's note named its tenant ran the shared function
            await conn.execute(legacy_truncate_trigger_drop_sql(table_schema=install.table_schema, table=each.table))
        for name in install.triggers:
            await conn.execute(statements[name])

    async def ring(self) -> NoteRingState:
        """the ring row, and the database's clock.

        :return: the state
        :rtype: NoteRingState
        :raises GenerationUnavailableError: when the ring is not installed
        """
        rows = await self._pool.fetch(ring_read_sql(self._schema))
        if not rows:
            raise GenerationUnavailableError(f"the change-note ring of {self._schema!r} is not installed")
        row = rows[0]
        return NoteRingState(
            current=int(row["current_ring"]),
            switch_no=int(row["switch_no"]),
            switched_at=row["switched_at"],
            switched_seq=row["switched_seq"],
            floor=row["floor"],
            fence_seq=row["fence_seq"],
            emptied=bool(row["emptied"]),
            log_incarnation=row["log_incarnation"],
            now=row["now"],
            fence_seconds=float(row["fence_seconds"]),
            wait_seconds=float(row["wait_seconds"]),
        )

    def _wait(self, ring: NoteRingState) -> timedelta:
        """the least time from the ring's last switch to its next: this keeper's fence, and the one told before.

        :param ring: the ring
        :ptype ring: NoteRingState
        :return: the wait
        :rtype: timedelta
        """
        return max(self._fence, timedelta(seconds=ring.wait_seconds))

    async def _update(self, ring: NoteRingState, **fields: Any) -> NoteRingState:
        """write ``fields`` to the ring row, only while it is still at the switch ``ring`` read, and read it again.

        :param ring: the state the change is based on
        :ptype ring: NoteRingState
        :param fields: the columns to set, by name
        :ptype fields: Any
        :return: the state after
        :rtype: NoteRingState
        """
        names = sorted(fields)
        assignments = ", ".join(f"{name} = ${index}" for index, name in enumerate(names, start=2))
        await self._pool.execute(
            ring_update_sql(self._schema, assignments),
            ring.switch_no,
            *(fields[name] for name in names),
        )
        return await self.ring()

    async def run_pass(self) -> KeeperPass:
        """keep the ring for one pass: put back a lost log, take a new one, pass the fence, empty, switch, scrub.

        :return: what the pass did
        :rtype: KeeperPass
        :raises GenerationUnavailableError: when the ring or the log cannot be read, or a restore failed
        """
        result = KeeperPass()
        ring = await self.ring()
        # the database's clock read before the log's state: a mark's time must not be later than its position
        read_at = ring.now
        state = await self._log.log_state()
        if state is None:
            if ring.log_incarnation is None:
                result.started = await self._log.start_log()
                await self._update(ring, log_incarnation=result.started)
            else:
                # the restore's worker exits when it ends (RestoreFolder): a restore is rare, the worker 193 MiB
                async with RestoreFolder() as folder:
                    result.restored = await self._restore(ring, folder)
            _KEEPER_LAST_PASS.set(time.time())
            return result
        if state.restoring is not None and state.restoring == ring.log_incarnation and state.continues is None:
            # a restore of the ring's lost log was declared and never labelled whole (a failed append, a lost
            # lease, a crash): it is finished, or done again, before anything else
            async with RestoreFolder() as folder:
                result.restored = await self._restore(ring, folder, incarnation=state.incarnation)
            _KEEPER_LAST_PASS.set(time.time())
            return result
        if ring.log_incarnation != state.incarnation:
            ring = await self._adopt(ring, state)
            result.adopted = state.incarnation
        await self._mark(read_at, state)
        if ring.fence_seq is None and ring.now >= ring.switched_at + self._wait(ring):
            # every slot copy read before the last switch has expired and its statements have ended: an epoch
            # appended from here on has its note in the current table or a later one
            ring = await self._update(ring, fence_seq=_token(state.incarnation, state.last_seq))
            result.fenced = True
        if not ring.emptied:
            idle = ring.oldest
            done, result.emptied_rows = await self.empty_slot(idle)
            if done:
                ring = await self._update(ring, emptied=True)
                result.emptied = idle
        switch_at = self._switch_bytes or _switch_threshold(state.max_bytes)
        notes_full = await self._current_notes_full(ring, switch_at)
        taken = _taken_bytes(state, ring.switched_seq)
        _CURRENT_BYTES.set(taken)
        _SWITCH_AT_BYTES.set(switch_at or 0)
        held = False
        if ring.fence_seq is not None and ((switch_at and taken >= switch_at) or notes_full):
            if ring.emptied:
                switched = await self._switch(ring, state)
                if switched is not None:
                    ring = switched
                    result.switched_to = ring.current
            else:
                held = True
                result.switch_waited = True
                self._switch_waits(ring, taken, switch_at)
        if not held:
            self._held_since = None
            _SWITCH_HELD_SINCE.set(0)
        now = time.monotonic()
        if self._scrubbed_at is None or now - self._scrubbed_at >= self._scrub_every.total_seconds():
            self._scrubbed_at = now
            result.scrub = await self._scrub(ring, state)
        _KEEPER_LAST_PASS.set(time.time())
        return result

    async def _adopt(self, ring: NoteRingState, state: ChangeLogState) -> NoteRingState:
        """take a log the keeper has not seen as the one the notes cover.

        The first log a ring sees is simply recorded. A later one the keeper did not put back itself was
        declared afresh after a loss: every epoch of it is of a write since, whose note is in the coverage,
        so the floor is its start; and, where the fence has passed, so is the next floor.

        :param ring: the ring
        :ptype ring: NoteRingState
        :param state: the log found
        :ptype state: ChangeLogState
        :return: the ring after
        :rtype: NoteRingState
        """
        if ring.log_incarnation is None:
            return await self._update(ring, log_incarnation=state.incarnation)
        start = _token(state.incarnation, 0)
        if state.continues is not None and state.continues == ring.log_incarnation:
            # this keeper's own restore, labelled whole, whose ring update did not happen
            return await self._update(
                ring,
                log_incarnation=state.incarnation,
                floor=start,
                fence_seq=None if ring.fence_seq is None else start,
            )
        log.warning(
            "the change log is under an incarnation the note keeper did not put back; its notes cover it from its start",
            extra={"extra_data": {"was": ring.log_incarnation, "now": state.incarnation}},
        )
        return await self._update(
            ring,
            log_incarnation=state.incarnation,
            floor=start,
            fence_seq=None if ring.fence_seq is None else start,
        )

    # WHY BATCHED DELETES AND NEVER TRUNCATE (measured; docs/runbook-epoch-write-cost.md, "Recorded results": "The
    # cached slot and batched-delete emptying, 2026-11-08" and "The change note (declared+notes), 2026-11-08"; 3tears
    # 44216b89e, 63a8c70a3, c9cb7c317, 1c636600f). Do not change the batch, pause, budget or band walk without
    # measuring again with scripts/measure-epoch-write-cost.py.
    # - TRUNCATE invalidates every session's cached plan of the note function, so every save overlapping an emptying
    #   replans: on Postgres at 8 writers those saves rose to an 18.8 ms p99 (40 ms max) in run 2.
    # - TRUNCATE takes ACCESS EXCLUSIVE on the table: every writer still inserting a note waits behind it.
    # - On YugabyteDB TRUNCATE is a DDL-like operation (a catalog change), not a row write.
    # - The batches: _EMPTY_BATCH (1000) notes a statement, each its own short transaction, _EMPTY_PAUSE (20 ms)
    #   between batches, _EMPTY_BUDGET (1 s) a pass, the next pass carrying on. Measured with them: saves overlapping
    #   a batch kept the run's p99 at 1 and 8 writers on YugabyteDB (13.6 vs 15.2 ms, 11.1 vs 13.8 ms).
    # - The walk: a batch picking its notes from the table's start stepped over every tombstone the batches before
    #   it left, about 200 ms of tombstones a batch by 200,000 notes on YugabyteDB (N^2/2/batch reads). So the
    #   batches walk the key in its own order: _delete_band by bands of yb_hash_code on YugabyteDB, _delete_past by
    #   id elsewhere (test_note_emptying_live: within twice a one-statement keyed delete of as many rows).
    # - The local YugabyteDB tserver crashes seen on 2026-11-08 were the kernel (yugabyte-db #33458, tcmalloc per-CPU
    #   caches on Linux 6.19 to 7.0.14 arm64), NOT TRUNCATE and not b118 (runbook, "Correction (2026-11-08)"): that
    #   is not the reason, and TRUNCATE is not the fix for anything here.
    # The gate's ring_empty kind (Pace 2026-11-14) declares these deletes; it changes nothing of how they run.
    async def empty_slot(self, index: int) -> tuple[bool, int]:
        """empty the ring's idle note table at ``index`` with batched deletes; never ``TRUNCATE``.

        The one place the ring's emptying is done, so how it is done is one change. Called only for the
        idle table (the one after the current), which the ring switched away from two switches back, at
        least one wait ago: no write can still reach it (the module docstring's argument). Each batch of
        about ``empty_batch`` notes is its own short transaction, with ``empty_pause`` between batches; a pass
        stops after ``empty_budget`` and the next pass carries on. The table counts as empty only once a
        read finds no note left in it.

        **Each batch starts where the last stopped.** A deleted note leaves a tombstone its table's reads step
        over until it is compacted away; a batch that picked its notes from the table's start stepped over every
        tombstone the batches before it left, so emptying ``N`` notes read about ``N²/2/batch`` (YugabyteDB:
        about 200 ms of tombstones a batch by 200,000 notes). The batches walk the key in its own order instead:
        on YugabyteDB the key's hash, a band of hash codes a statement, each band sized to about a batch from
        the last; elsewhere the key itself, past the last id deleted.

        :param index: the idle table
        :ptype index: int
        :return: whether it is now empty, and how many notes this call deleted
        :rtype: tuple[bool, int]
        """
        table = self._ring_table(index)
        if self._yugabyte is None:
            self._yugabyte = "-YB-" in str((await self._pool.fetch("SELECT version() AS v"))[0]["v"])
        deleted = 0
        began = time.monotonic()
        while True:
            if self._yugabyte:
                count, done = await self._delete_band(table, index)
            else:
                count, done = await self._delete_past(table, index)
            deleted += count
            if done:
                break
            if time.monotonic() - began >= self._empty_budget.total_seconds():
                break
            await asyncio.sleep(self._empty_pause.total_seconds())
        if deleted:
            _EMPTIED_ROWS.inc(deleted)
        if not done:
            # the pass's budget ran out mid-walk: the next pass carries on from where this one stopped
            return False, deleted
        left = await self._pool.fetch(f"SELECT 1 FROM {table} LIMIT 1")  # noqa: S608 - trusted identifiers
        if left:
            # a note the walk passed (none can be written here, but a walk may restart after a keeper restarts):
            # the next pass walks the table again from its start
            self._emptying_at = None
            return False, deleted
        self._emptying_at = None
        if not self._yugabyte:
            # before the slot counts as emptied (the ring may switch back into it only then): its dead notes vacuumed
            await self._vacuum_emptied(table)
        _EMPTIED.inc()
        log.info(
            "a change-note table was emptied",
            extra={"extra_data": {"table": note_table(index), "deleted": deleted}},
        )
        return True, deleted

    # WHY A VACUUM OF THE EMPTIED SLOT, ON POSTGRES ONLY (Pace 2026-11-15: "sounds like we need vacuum if you are running
    # pg"). The batched deletes leave every note of the slot as a dead row (and dead index entries) until a vacuum. R1
    # measured the ring switching back into such a slot (test_reused_note_slot_live.py; the numbers are above
    # _CURRENT_COUNT_SECONDS): on Postgres 16 the keeper's count of the reused slot took 3.4x a fresh slot's at once
    # and 1.5x five minutes later as autovacuum reached it, a full scan 3.2x / 2.0x. So the slot is vacuumed here,
    # after its last delete and before empty_slot reports it empty: run_pass marks the ring row ``emptied`` only on
    # that report, and never switches into a slot not so marked, so a switch never lands on an unvacuumed slot.
    # - Plain VACUUM, never FULL: FULL rewrites the table under ACCESS EXCLUSIVE; the plain one takes SHARE UPDATE
    #   EXCLUSIVE, which no note insert waits behind (and none can reach the idle slot anyway: empty_slot's docstring).
    # - Through the pool, outside any transaction, as Postgres requires of VACUUM.
    # - Measured with it (V1 2026-11-15, the same test on integ095-pg under measure-lock, load 3.9, trusted; S/v1/m4-pg):
    #   at a full slot (393,216 notes + 2,000) the VACUUM took 104 ms of the emptying's 1.37 s. Right after it the
    #   reused slot reads as a fresh one: the keeper's count 0.92x a fresh slot's (A/A gap 0.90x; it was 3.44x), a
    #   full scan 1.01x (A/A 1.07x; it was 3.22x), the indexed reads 0.96x-1.03x. Three noisier runs said the same
    #   (0.87x-1.06x). It runs once a slot, on the pass that finishes the walk: that pass's emptying (empty_budget,
    #   1 s of deletes) takes about a tenth more, every other pass nothing. Its speed test holds it within twice a
    #   bare VACUUM of a table of the slot's shape holding as many deleted rows.
    # - Never on YugabyteDB: VACUUM is a no-op there (b118 answers it with the notice "VACUUM is a no-op statement since
    #   YugabyteDB performs garbage collection of dead tuples automatically"). Its deleted notes are
    #   tombstones only a compaction drops, and a compaction may not drop one younger than
    #   timestamp_history_retention_interval_sec (900 s, the default on the local b118), nor runs on a table whose
    #   reads see under auto_compact_percent_obsolete (99 %) obsolete keys; no SQL statement asks for a compaction.
    #   R1 measured the reused slot there at 9.7x a fresh one's count (12.9x a full scan), unchanged 5 min later.
    async def _vacuum_emptied(self, table: str) -> None:
        """vacuum the ring slot just emptied (Postgres only; its caller, empty_slot, never calls it on YugabyteDB).

        :param table: the qualified table, empty of notes
        :ptype table: str
        :return: nothing
        :rtype: None
        """
        await whole_read(
            self._pool.execute(f"VACUUM {table}"),  # noqa: S608 - trusted identifiers
            kind="ring_empty",
            reason="the ring slot just emptied is vacuumed (Postgres) before the ring may switch back into it",
            ruling="Pace 2026-11-14: emptying a used-up change-notes ring table goes on the whole-drop list; "
            "Pace 2026-11-15: and its VACUUM on Postgres",
            proof="packages/core/tests/unit/collections/test_change_note_restore_reads.py::test_a_ring_slot_is_emptied_whole_only_when_idle",
            speed="packages/core/tests/integration/test_note_emptying_live.py::test_an_emptied_full_slot_s_vacuum_costs_within_twice_a_bare_vacuum",
        )

    # why batches, this walk and never TRUNCATE: the measured reasons above empty_slot; keep them as they are
    async def _delete_band(self, table: str, index: int) -> tuple[int, bool]:
        """delete the notes of the next band of key hashes (YugabyteDB); how many, and whether the walk is done.

        A band's statement reads only its own hash codes, so no batch steps over another's tombstones. The band
        is resized toward ``empty_batch`` notes from the count the last one deleted.

        :param table: the qualified table
        :ptype table: str
        :param index: its place in the ring
        :ptype index: int
        :return: notes deleted, and whether the band reached the end of the hash space
        :rtype: tuple[int, bool]
        """
        if self._reached(index) is None:
            # a walk's start: the first band is sized from a count of the table's notes, capped so the count reads
            # at most _BAND_COUNT_BATCHES batches' worth however many the table holds (past the cap the band starts
            # small and grows from what each band deletes); a table holding none is done without a statement a band
            cap = self._empty_batch * _BAND_COUNT_BATCHES
            held = int(
                (
                    await self._pool.fetch(
                        f"SELECT count(*) AS notes FROM (SELECT 1 FROM {table} LIMIT $1) AS held",  # noqa: S608 - trusted identifiers
                        cap,
                    )
                )[0]["notes"]
            )
            if held == 0:
                return 0, True
            self._band = max(1, min(_HASH_SPACE, _HASH_SPACE * self._empty_batch // held))
        start = int(self._reached(index) or 0)
        end = min(start + self._band, _HASH_SPACE)
        status = await whole_read(
            self._pool.execute(
                f"DELETE FROM {table} WHERE yb_hash_code(id) >= $1 AND yb_hash_code(id) < $2",  # noqa: S608 - trusted identifiers
                start,
                end,
            ),
            kind="ring_empty",
            reason="the idle ring slot two switches back is emptied whole: no write can reach it any more",
            ruling="Pace 2026-11-14: emptying a used-up change-notes ring table goes on the whole-drop list, as TRUNCATE (11-11)",
            proof="packages/core/tests/unit/collections/test_change_note_restore_reads.py::test_a_ring_slot_is_emptied_whole_only_when_idle",
            speed="packages/core/tests/integration/test_note_emptying_live.py::test_an_idle_table_is_emptied_once_each_note_within_twice_a_keyed_delete",
        )
        count = _status_count(status)
        self._emptying_at = (index, end)
        if count:
            self._band = max(1, min(_HASH_SPACE, self._band * self._empty_batch // count))
        elif self._band < _HASH_SPACE:
            self._band = min(_HASH_SPACE, self._band * 2)
        return count, end >= _HASH_SPACE

    def _reached(self, index: int) -> Any:
        """where the emptying of the table at ``index`` has reached; None when no walk of it has started.

        :param index: the table's place in the ring
        :ptype index: int
        :return: the next band's start or the last id deleted, or None
        :rtype: Any
        """
        if self._emptying_at is None or self._emptying_at[0] != index:
            return None
        return self._emptying_at[1]

    # why batches, this walk and never TRUNCATE: the measured reasons above empty_slot; keep them as they are
    async def _delete_past(self, table: str, index: int) -> tuple[int, bool]:
        """delete the next ``empty_batch`` notes past the last id deleted, in id order; how many, and whether done.

        :param table: the qualified table
        :ptype table: str
        :param index: its place in the ring
        :ptype index: int
        :return: notes deleted, and whether the batch reached the table's end
        :rtype: tuple[int, bool]
        """
        rows = await whole_read(
            self._pool.fetch(
                f"DELETE FROM {table} WHERE id IN (SELECT id FROM {table} "  # noqa: S608 - trusted identifiers
                "WHERE $2::uuid IS NULL OR id > $2 ORDER BY id LIMIT $1) RETURNING id",
                self._empty_batch,
                self._reached(index),
            ),
            kind="ring_empty",
            reason="the idle ring slot two switches back is emptied whole: no write can reach it any more",
            ruling="Pace 2026-11-14: emptying a used-up change-notes ring table goes on the whole-drop list, as TRUNCATE (11-11)",
            proof="packages/core/tests/unit/collections/test_change_note_restore_reads.py::test_a_ring_slot_is_emptied_whole_only_when_idle",
            speed="packages/core/tests/integration/test_note_emptying_live.py::test_an_idle_table_is_emptied_once_each_note_within_twice_a_keyed_delete",
        )
        if rows:
            self._emptying_at = (index, max(row["id"] for row in rows))
        return len(rows), len(rows) < self._empty_batch

    def _switch_waits(self, ring: NoteRingState, taken: int, switch_at: int | None) -> None:
        """say, at ERROR on every keeper pass while it lasts, that a due switch waits for the idle table's emptying:
        for how long, and how far the current table has grown past the size planned for it (the why is above
        ``_SWITCH_HELD_SINCE``: a held switch puts the restore budget at risk). Counted once per switch.

        :param ring: the ring
        :ptype ring: NoteRingState
        :param taken: bytes of change records the log took since the last switch (the current table's size)
        :ptype taken: int
        :param switch_at: the planned switch size in bytes, None when the ring switches on its note count alone
        :ptype switch_at: int | None
        :return: nothing
        :rtype: None
        """
        now = time.time()
        if self._held_since is None:
            self._held_since = now
        _SWITCH_HELD_SINCE.set(self._held_since)
        if self._waited_switch != ring.switch_no:
            self._waited_switch = ring.switch_no
            _SWITCH_WAITS.inc()
        log.error(
            "a change-note ring switch is HELD: the table it would switch into is still being emptied, so the "
            "current note table keeps growing past its planned size and the restore budget is at risk",
            extra={
                "extra_data": {
                    "switch": ring.switch_no,
                    "held_seconds": round(now - self._held_since, 1),
                    "current": note_table(ring.current),
                    "current_notes_at_least": self._notes_seen,
                    "current_bytes": taken,
                    "switch_at_bytes": switch_at,
                    "waiting_on": note_table(ring.oldest),
                }
            },
        )

    async def _switch(self, ring: NoteRingState, state: ChangeLogState) -> NoteRingState | None:
        """make the emptied idle table current; the fence's position becomes the coverage's floor.

        The update names the switch it read and requires the idle table's emptying finished, so it never
        lands on a table still being emptied. It tells the writers this keeper's fence, and keeps the wait
        until the next switch at least the fence told before, which copies read until now were used under.

        :param ring: the ring
        :ptype ring: NoteRingState
        :param state: the log
        :ptype state: ChangeLogState
        :return: the ring after; ``None`` when the ring row was not as read (another keeper, or not emptied)
        :rtype: NoteRingState | None
        """
        fence = self._fence.total_seconds()
        status = await self._pool.execute(
            ring_switch_sql(self._schema),
            ring.switch_no,
            ring.oldest,
            _token(state.incarnation, state.last_seq),
            fence,
        )
        if status.rpartition(" ")[2] != "1":
            return None
        after = await self.ring()
        self._notes_full = False
        self._counted_at = None
        _SWITCHES.inc()
        # a mark older than the table the coverage now starts with can floor no restore
        await self.drop_marks_before(ring.switched_at - self._wait(ring))
        log.info(
            "the change-note ring switched",
            extra={"extra_data": {"current": after.current, "switch": after.switch_no, "floor": after.floor}},
        )
        await self.count_ring_reads(ring.current)
        return after

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

    async def count_ring_reads(self, index: int) -> None:
        """count, per table, the notes of the table just switched away from whose statement read the ring row.

        Once per switch: those writers had no valid cached slot (raw SQL outside 3tears, or a process that
        could not refresh its copy). They are correct, only slower; a steady count names the table whose
        writers bypass the cache.

        :param index: the table the ring just switched away from
        :ptype index: int
        :return: nothing
        :rtype: None
        """
        rows = await self._pool.fetch(
            f"SELECT table_name, count(*) AS notes FROM {self._ring_table(index)} WHERE ring_read GROUP BY 1"  # noqa: S608 - trusted identifiers
        )
        counted = {str(row["table_name"]): int(row["notes"]) for row in rows}
        for table, notes in counted.items():
            _RING_READS.labels(table=table).inc(notes)
        if counted:
            log.info(
                "change notes written without a cached slot (the statement read the ring row)",
                extra={"extra_data": {"note_table": note_table(index), "tables": counted}},
            )

    async def _current_notes_full(self, ring: NoteRingState, switch_at: int | None) -> bool:
        """whether the current table holds the switch's note count, counted every twentieth of the fence.

        The count stops at the threshold, so it reads at most that many notes.

        :param ring: the ring
        :ptype ring: NoteRingState
        :param switch_at: the switch size in bytes, the default count derives from
        :ptype switch_at: int | None
        :return: whether the ring should switch on its notes
        :rtype: bool
        """
        threshold = self._switch_notes or (switch_at // _NOTE_BYTES_FLOOR if switch_at else None)
        if not threshold:
            return False
        now = time.monotonic()
        if self._counted_at is not None and now - self._counted_at < self._mark_every.total_seconds():
            return self._notes_full
        self._counted_at = now
        began = time.perf_counter()
        rows = await self._pool.fetch(
            f"SELECT count(*) AS notes FROM (SELECT 1 FROM {self._ring_table(ring.current)} LIMIT $1) n",  # noqa: S608 - trusted identifiers
            threshold,
        )
        _CURRENT_COUNT_SECONDS.labels(since_switch=_since_switch(ring)).observe(time.perf_counter() - began)
        self._notes_seen = int(rows[0]["notes"])
        _CURRENT_COUNT_NOTES.set(self._notes_seen)
        self._notes_full = self._notes_seen >= threshold
        return self._notes_full

    async def _mark(self, read_at: datetime, state: ChangeLogState) -> None:
        """record the log's position every ``fence / 20``: where a suffix restore may start.

        The mark's time is the database's clock read before the log's state (``read_at``), so every epoch
        after the mark's position was appended after that time, and its note written at most half the wait
        before it (a writing transaction ends within that half, the same bound the ring's emptying relies on).

        :param read_at: the database's clock at this pass's first ring read, before the log's state
        :ptype read_at: datetime
        :param state: the log, read after it
        :ptype state: ChangeLogState
        :return: nothing
        :rtype: None
        """
        now = time.monotonic()
        if self._marked_at is not None and now - self._marked_at < self._mark_every.total_seconds():
            return
        self._marked_at = now
        await self._pool.execute(
            f"INSERT INTO {self._marks} (log_incarnation, seq, at) VALUES ($1, $2, $3)",  # noqa: S608 - trusted identifiers
            state.incarnation,
            state.last_seq,
            read_at,
        )

    async def _restore(
        self, ring: NoteRingState, folder: RestoreFolder, *, incarnation: str | None = None
    ) -> str | None:
        """put the lost log back from the notes of the current and the previous tables, or their newest part.

        In order: declare the new log first (so writers waiting for it append at once), then read the notes,
        then append them and label the log as continuing the lost one. The label is the restore's completion:
        until it is written a follower of the lost log is behind, never carried over. ``incarnation`` names a
        restore declared earlier and never labelled; it is done again, in full, into that log.

        The whole coverage is put back when its records fit :data:`RESTORE_BUDGET_RATIO` of the log. When
        they do not, the newest part that fits is: the notes written since a mark of the log's position
        (less half the wait), with that position as the floor (``covered_from``). A follower at or past it
        catches up; an older one is behind (``record_lost``) until it restarts. Said at ERROR and counted
        (``threetears_change_notes_restore_truncated_total``). Only when not even the part after the newest
        mark fits is the restore refused (``threetears_change_notes_restores_refused_total``) and the new log
        taken as a fresh one. Never a reload.

        :param ring: the ring
        :ptype ring: NoteRingState
        :param folder: the restore's worker, spawned here once the restore is this keeper's, beside its sizing
        :ptype folder: RestoreFolder
        :param incarnation: the restoring log to finish; ``None`` declares one
        :ptype incarnation: str | None
        :return: the new log's incarnation; ``None`` when a log was there and it was not this restore's, or
            the restore was refused
        :rtype: str | None
        :raises GenerationUnavailableError: when the log cannot be declared or written; the next pass redoes it
        """
        lost = ring.log_incarnation
        assert lost is not None  # narrow: the caller restores only a log the ring has seen
        if incarnation is None:
            incarnation = await self._log.begin_restore(lost)
            if incarnation is None:
                return None
        folder.start()
        covered = _seq_under(ring.floor, lost)
        if covered is None:
            covered = _seq_under(ring.fence_seq, lost)
        began = time.monotonic()
        state = await self._log.log_state()
        budget = None if state is None or not state.max_bytes else state.max_bytes * RESTORE_BUDGET_RATIO - state.bytes
        # WHY the sizing runs beside the stream (owner #101, 2026-11-14, test_change_note_restore_speed_live's
        # over-bound case: each row noted twice puts the count's bound past the budget while the distinct rows fit):
        # one after another the restore took the count 19 ms + the DISTINCT estimate 88 ms + the stream ~290 ms against
        # ~158 ms of bare reads, 2.4x on PG (2.34x in o97's split). The count, and the estimate when the count's bound
        # does not fit, now run on a second connection while the stream's own reads run, and the stream drops a
        # repeated note in the database (DISTINCT ON, see _StreamedChanges._stream): 1.50-1.67x on PG; YugabyteDB
        # 1.6-1.7x before the DISTINCT ON (the test's fixture, best of 5 interleaved). And
        # :class:`_Gated` lets no change reach ``finish_restore`` until they rule: it holds the whole-table changes
        # (one per table and tenant) and at most one keyed chunk, by when the stream's sorted read is done. A coverage
        # that does not fit is refused before its first append, so an overflow never leaves a record in the new log
        # (appending and undoing would leave the log's bytes spent, and the cut that follows would not fit). Only a
        # pool with room for a second connection overlaps; a pool of one sizes first, as before (the stream holds the
        # one connection, so a sizing waiting for it would wait for ever).
        verdict: asyncio.Task[tuple[int, bool]] | None = None
        if budget is not None and _room(self._pool) >= 2:
            verdict = asyncio.create_task(self._sized(ring, budget))
            chosen: tuple[int | None, datetime | None] | None = (covered, None)
            total = 0
        else:
            total, bound = await self._count_since(ring, None)
            chosen = await self._fitting(ring, lost, covered, budget, bound)
        appended: int | None = None
        changes: _StreamedChanges | None = None
        try:
            if verdict is not None:
                changes = self._changes_since(ring, None, folder)
                try:
                    appended = await self._log.finish_restore(
                        _Gated(changes, verdict), incarnation=incarnation, lost=lost, covered_from=covered
                    )
                    total = verdict.result()[0]
                except _CoverageOverflows:
                    total = verdict.result()[0]
                    # nothing was appended: the gate refused before the first change
                    assert budget is not None  # narrow: a verdict is asked only under a budget
                    chosen = await self._cut(ring, lost, covered, budget)
                    changes = None
        finally:
            if verdict is not None and not verdict.done():
                # the stream failed before the sizing ruled: stop it, so its connection goes back
                await cancel_and_wait(verdict)
        if chosen is None:
            _RESTORES_REFUSED.inc()
            log.error(
                "the lost epoch change log was NOT restored from the change notes: not even the notes after the "
                "newest mark fit the log with headroom. Every follower of the lost log is behind (replaced) until "
                "it restarts; the new log is taken as a fresh one",
                extra={"extra_data": {"lost": lost, "restored": incarnation, "notes": total, "budget": budget}},
            )
            await self._update(ring, log_incarnation=incarnation, floor=_token(incarnation, 0), fence_seq=None)
            self._scrubbed = None
            self._seen = None
            return None
        floor, since = chosen
        kept = total if since is None else (await self._count_since(ring, since))[0]
        if changes is None:
            # streamed a page at a time, so the keeper's memory follows one change, not the coverage; should the log
            # have grown past the estimate meanwhile, finish_restore stops unlabelled and the next pass sizes again
            changes = self._changes_since(ring, since, folder)
            appended = await self._log.finish_restore(changes, incarnation=incarnation, lost=lost, covered_from=floor)
        tables = changes.tables
        if kept < total:
            _RESTORES_TRUNCATED.inc()
            log.error(
                "the lost epoch change log was put back from only the newest change notes that fit it. A follower "
                "whose mark is older than covered_from is behind (record_lost) until it restarts",
                extra={
                    "extra_data": {
                        "lost": lost,
                        "restored": incarnation,
                        "covered_from": None if floor is None else _token(lost, floor),
                        "coverage_floor": None if covered is None else _token(lost, covered),
                        "notes_kept": kept,
                        "notes_left_out": total - kept,
                    }
                },
            )
        state = await self._log.log_state()
        head = state.last_seq if state is not None and state.incarnation == incarnation else 0
        await self._update(
            ring,
            log_incarnation=incarnation,
            floor=_token(incarnation, 0),
            fence_seq=None if ring.fence_seq is None else _token(incarnation, head),
        )
        self._scrubbed = None
        self._seen = None
        _RESTORES.inc()
        log.warning(
            "the epoch change log was put back from the change notes under a new incarnation. A follower whose mark "
            "is at or past the coverage's floor walks every restored record and catches up; an older one is behind "
            "(record_lost) until it restarts",
            extra={
                "extra_data": {
                    "lost": lost,
                    "restored": incarnation,
                    "covered_from": floor,
                    "notes": kept,
                    "tables": tables,
                    "records": appended,
                    "seconds": round(time.monotonic() - began, 3),
                }
            },
        )
        return incarnation

    async def _count_since(self, ring: NoteRingState, since: datetime | None) -> tuple[int, int]:
        """how many notes of the coverage were written at or after ``since`` (all of them for ``None``), and a bound
        no smaller than :meth:`_estimate` of them.

        The bound sizes every note as :meth:`_estimate` sizes a distinct row (a row noted twice counted twice), so it
        is never below the estimate: when it fits the budget the estimate does too, and the restore needs no read of
        the distinct rows to know it. Summed per table, so the table count needs no sort of the notes.

        WHY (o97, 2026-11-14, test_change_note_restore_speed_live at 200,000 notes): the count and the estimate were
        two reads of the coverage, ~8 ms + ~68 ms on PG 16 and ~56 ms + ~105 ms on YugabyteDB, the estimate's distinct
        rows a hash aggregate spilling to disk; this one read, grouped by table alone, answers both when the
        coverage fits, as a restore's coverage nearly always does (the ring switches at half the restore budget).

        :param ring: the ring
        :ptype ring: NoteRingState
        :param since: the cut
        :ptype since: datetime | None
        :return: ``(notes, bound)``
        :rtype: tuple[int, int]
        """
        rows = await whole_read(
            self._pool.fetch(
                "SELECT coalesce(sum(g.notes), 0) AS notes, count(*) AS tables, coalesce(sum(g.bytes), 0) AS bytes "  # noqa: S608 - trusted identifiers
                "FROM (SELECT table_name, count(*) AS notes, sum(length(table_name) + coalesce(length(key_text), 0) "
                "+ coalesce(length(columns::text), 0)) AS bytes "
                f"FROM {_coverage_sql(self._owner, ring.current, ring.previous)} n "
                "WHERE $1::timestamptz IS NULL OR n.noted_at >= $1 GROUP BY table_name) g",
                since,
            ),
            kind="restore",
            reason="a lost change log is put back from every note of the ring's coverage, its current and previous slots",
            ruling="Pace 2026-11-14: the restore's full read of the coverage is a cold-start read (whole_read), as TRUNCATE (11-11)",
            proof="packages/core/tests/unit/collections/test_change_note_restore_reads.py::test_the_coverage_is_read_whole_only_by_a_restore",
            speed="packages/core/tests/integration/test_change_note_restore_speed_live.py::test_a_restore_reads_its_coverage_within_twice_the_bare_reads",
        )
        row = rows[0]
        notes = int(row["notes"])
        return notes, (
            int(row["bytes"]) + notes * _RESTORED_ROW_ESTIMATE + int(row["tables"]) * _RESTORED_TABLE_ESTIMATE
        )

    async def _estimate(self, ring: NoteRingState, since: datetime | None) -> int:
        """the bytes the records of the coverage's notes since ``since`` would take, estimated in the database.

        Each distinct row is its table, key and columns as text plus :data:`_RESTORED_ROW_ESTIMATE`; each table
        one record's overhead. Computed where the notes are, so nothing of them is held here.

        :param ring: the ring
        :ptype ring: NoteRingState
        :param since: the cut; ``None`` the whole coverage
        :ptype since: datetime | None
        :return: the estimate
        :rtype: int
        """
        rows = await whole_read(
            self._pool.fetch(
                # summed per table first, so the table count needs no sort of every distinct row
                "SELECT coalesce(sum(g.rows), 0) AS rows, count(*) AS tables, coalesce(sum(g.bytes), 0) AS bytes "  # noqa: S608 - trusted identifiers
                "FROM (SELECT table_name, count(*) AS rows, sum(length(table_name) + coalesce(length(key_text), 0) "
                "+ coalesce(length(columns::text), 0)) AS bytes "
                f"FROM (SELECT DISTINCT table_name, key_text, columns FROM {_coverage_sql(self._owner, ring.current, ring.previous)} n "
                "WHERE $1::timestamptz IS NULL OR n.noted_at >= $1) d GROUP BY table_name) g",
                since,
            ),
            kind="restore",
            reason="a lost change log is put back from every note of the ring's coverage, its current and previous slots",
            ruling="Pace 2026-11-14: the restore's full read of the coverage is a cold-start read (whole_read), as TRUNCATE (11-11)",
            proof="packages/core/tests/unit/collections/test_change_note_restore_reads.py::test_the_coverage_is_read_whole_only_by_a_restore",
            speed="packages/core/tests/integration/test_change_note_restore_speed_live.py::test_a_restore_reads_its_coverage_within_twice_the_bare_reads",
        )
        row = rows[0]
        return (
            int(row["bytes"])
            + int(row["rows"]) * _RESTORED_ROW_ESTIMATE
            + int(row["tables"]) * _RESTORED_TABLE_ESTIMATE
        )

    async def _fitting(
        self, ring: NoteRingState, lost: str, covered: int | None, budget: float | None, bound: int
    ) -> tuple[int | None, datetime | None] | None:
        """the oldest-reaching part of the coverage whose records fit ``budget``: its floor and its cut.

        The candidates, oldest first: every note, with the coverage's floor; then, for each mark of the lost
        log after that floor, the notes written since the mark's time less half the wait, with the mark's
        position as the floor. Each is a suffix of the one before, so the first that fits is the largest; a
        binary search sizes only ``log2`` of them, each in the database. Memory holds the marks, not the notes.

        :param ring: the ring
        :ptype ring: NoteRingState
        :param lost: the lost log's incarnation
        :ptype lost: str
        :param covered: the coverage's floor under it; ``None`` when it vouches for none
        :ptype covered: int | None
        :param budget: the bytes the restore may take; ``None`` unbounded
        :ptype budget: float | None
        :param bound: a size no smaller than the whole coverage's estimate (:meth:`_count_since`): when it fits, the
            whole coverage does, unsized
        :ptype bound: int
        :return: ``(floor, cut)``, the cut ``None`` for the whole coverage; ``None`` when not even the newest
            part fits
        :rtype: tuple[int | None, datetime | None] | None
        """
        if budget is None or bound <= budget or await self._estimate(ring, None) <= budget:
            return covered, None
        return await self._cut(ring, lost, covered, budget)

    async def _sized(self, ring: NoteRingState, budget: float) -> tuple[int, bool]:
        """how many notes the coverage holds, and whether its records fit ``budget``: by the count's bound
        (:meth:`_count_since`) when it fits, else by :meth:`_estimate` of the distinct rows.

        :param ring: the ring
        :ptype ring: NoteRingState
        :param budget: the bytes the restore may take
        :ptype budget: float
        :return: ``(notes, fits)``
        :rtype: tuple[int, bool]
        """
        total, bound = await self._count_since(ring, None)
        return total, bound <= budget or await self._estimate(ring, None) <= budget

    async def _cut(
        self, ring: NoteRingState, lost: str, covered: int | None, budget: float
    ) -> tuple[int | None, datetime | None] | None:
        """the oldest-reaching part after a mark of the lost log whose records fit ``budget``: its floor and its cut
        (:meth:`_fitting`, once the whole coverage is known not to fit).

        :param ring: the ring
        :ptype ring: NoteRingState
        :param lost: the lost log's incarnation
        :ptype lost: str
        :param covered: the coverage's floor under it; ``None`` when it vouches for none
        :ptype covered: int | None
        :param budget: the bytes the restore may take
        :ptype budget: float
        :return: ``(floor, cut)``; ``None`` when not even the newest part fits
        :rtype: tuple[int | None, datetime | None] | None
        """
        lag = self._wait(ring) / 2
        marks = [(int(mark["seq"]), mark["at"] - lag) for mark in await self.marks_past(lost, covered)]
        low, high = 0, len(marks)
        while low < high:  # the first (oldest-reaching) mark whose part fits; sizes fall along the list
            middle = (low + high) // 2
            if await self._estimate(ring, marks[middle][1]) <= budget:
                high = middle
            else:
                low = middle + 1
        return None if low == len(marks) else marks[low]

    def _changes_since(self, ring: NoteRingState, since: datetime | None, folder: RestoreFolder) -> _StreamedChanges:
        """the coverage's notes since ``since``, folded per table and streamed a page at a time.

        :param ring: the ring
        :ptype ring: NoteRingState
        :param since: the cut; ``None`` the whole coverage
        :ptype since: datetime | None
        :param folder: the restore's worker
        :ptype folder: RestoreFolder
        :return: the changes, as an async iterable
        :rtype: _StreamedChanges
        """
        return _StreamedChanges(self._pool, self._owner, (ring.current, ring.previous), since, folder)

    async def _scrub(self, ring: NoteRingState, state: ChangeLogState) -> ScrubPass:
        """check that the epochs after the floor, up to the log's head one pass ago, have their notes.

        An epoch after the fence's position needs its note in the current table, the one that outlives the
        next emptying; any other after the floor, in the current or the previous table. A whole-table or
        keys-unknown epoch needs any note of its table there. Checks only: a miss is logged and counted.

        :param ring: the ring
        :ptype ring: NoteRingState
        :param state: the log
        :ptype state: ChangeLogState
        :return: what it found
        :rtype: ScrubPass
        """
        incarnation = state.incarnation
        seen = self._seen
        self._seen = (incarnation, state.last_seq)
        floor = _seq_under(ring.floor, incarnation)
        if floor is None:
            floor = _seq_under(ring.fence_seq, incarnation)
        if floor is None or seen is None or seen[0] != incarnation:
            return ScrubPass()
        start = max(floor, self._scrubbed[1] if self._scrubbed and self._scrubbed[0] == incarnation else floor)
        start = max(start, state.first_seq - 1)
        end = min(seen[1], start + self._scrub_batch)
        if end <= start:
            _SCRUB_BEHIND.set(max(0, state.last_seq - start))
            return ScrubPass(position=_token(incarnation, start))
        records = await self._epochs_between(start, end)
        fence = _seq_under(ring.fence_seq, incarnation)
        found = await self.notes_found(ring, records)
        result = ScrubPass(checked=len(records), position=_token(incarnation, end))
        for record in records:
            seq = int(record.token.rpartition(":")[2])
            where = (ring.current,) if fence is not None and seq > fence else (ring.current, ring.previous)
            missing = _first_unnoted(record, where, found)
            if missing is None:
                continue
            result.missing += 1
            _MISSING.labels(table=record.table).inc()
            log.error(
                "an epoch the change log holds has no change note where the coverage needs it: a restore after a "
                "NATS loss would not put it back",
                extra={
                    "extra_data": {
                        "table": record.table,
                        "epoch": record.token,
                        "missing": missing,
                        "looked_in": [note_table(index) for index in where],
                    }
                },
            )
        self._scrubbed = (incarnation, end)
        _SCRUB_BEHIND.set(max(0, state.last_seq - end))
        return result

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

    async def notes_found(self, ring: NoteRingState, records: Sequence[EpochChange]) -> dict[int, _Found]:
        """the notes the coverage tables hold for ``records``, by table.

        Each query is led by an index, so its cost follows the epochs checked, not the size of the coverage: the
        keyed notes by the ``(key_text, table_name)`` index, whether a table is noted whole by its whole-note
        index, and whether a table has any note (asked only for a record naming no rows: a whole-table or
        keys-unknown epoch) by the ``(table_name, noted_at)`` index. The last two are ``DISTINCT`` over their
        index, which YugabyteDB answers by skipping from one table's first note to the next's: a note a table.

        :param ring: the ring
        :ptype ring: NoteRingState
        :param records: the epochs checked
        :ptype records: Sequence[EpochChange]
        :return: per note table, what it holds of them
        :rtype: dict[int, _Found]
        """
        tables = sorted({record.table for record in records})
        keys = sorted({_KEY_SEPARATOR.join(row.ids) for record in records for row in record.rows})
        rowless = sorted({record.table for record in records if not record.rows})
        found: dict[int, _Found] = {}
        for index in (ring.current, ring.previous):
            table = self._ring_table(index)
            keyed = await self._pool.fetch(
                f"SELECT DISTINCT table_name, key_text FROM {table} "  # noqa: S608 - trusted identifiers
                "WHERE key_text = ANY($2::text[]) AND table_name = ANY($1::text[])",
                tables,
                keys,
            )
            # DISTINCT over the whole-note index: YugabyteDB reads one note a table noted whole
            whole = await self._pool.fetch(
                f"SELECT DISTINCT table_name FROM {table} WHERE key_text IS NULL AND table_name = ANY($1::text[])",  # noqa: S608 - trusted identifiers
                tables,
            )
            any_rows = (
                await self._pool.fetch(
                    # DISTINCT over the (table_name, noted_at) index: YugabyteDB skips from one table's first note
                    # to the next table's, reading one a table; an EXISTS a table read ahead a batch of its notes
                    # each (500, measured)
                    f"SELECT DISTINCT table_name FROM {table} WHERE table_name = ANY($1::text[])",  # noqa: S608 - trusted identifiers
                    rowless,
                )
                if rowless
                else []
            )
            found[index] = _Found(
                keyed={(str(row["table_name"]), str(row["key_text"])) for row in keyed},
                whole={str(row["table_name"]) for row in whole},
                any_note={str(row["table_name"]) for row in any_rows},
            )
        return found


def _coverage_sql(owner: str, current: int, previous: int) -> str:
    """the SQL of the coverage's notes: the ring's current and previous tables, one relation.

    :param owner: the ring's schema, quoted
    :ptype owner: str
    :param current: the current table's place in the ring
    :ptype current: int
    :param previous: the previous table's
    :ptype previous: int
    :return: a parenthesized ``UNION ALL``
    :rtype: str
    """
    return (
        # the slots spelled as the ring's DDL names them (note_table), so the gate reads the ring's template here
        f'(SELECT {_COVERAGE_COLUMNS} FROM {owner}."threetears_change_notes_{current:d}" '  # noqa: S608 - trusted identifiers
        f'UNION ALL SELECT {_COVERAGE_COLUMNS} FROM {owner}."threetears_change_notes_{previous:d}")'
    )


class _CoverageOverflows(Exception):
    """the whole coverage's records do not fit the restore's budget: raised by :class:`_Gated` before its first
    change, so ``finish_restore`` has appended nothing."""


def _room(pool: Any) -> int:
    """how many connections ``pool`` may hold at once (an ``asyncpg.Pool``'s ``max_size``); 1 when it does not say.

    :param pool: the pool
    :ptype pool: Any
    :return: its bound
    :rtype: int
    """
    size = getattr(pool, "get_max_size", None)
    return int(size()) if callable(size) else 1


class _Gated:
    """a stream of changes that lets none through until ``sizing`` rules the whole coverage fits the budget.

    While the ruling is pending it holds the whole-table changes (one per table and tenant, small) and at most one
    keyed chunk, so the stream's reads run beside the estimate and memory still holds one chunk. A coverage ruled
    too large raises :class:`_CoverageOverflows` before any change is yielded; the stream is closed either way.

    :ivar tables: the tables streamed so far
    """

    def __init__(self, changes: _StreamedChanges, sizing: Awaitable[tuple[int, bool]]) -> None:
        self._changes = changes
        self._sizing = sizing

    @property
    def tables(self) -> int:
        """the tables streamed so far.

        :return: the count
        :rtype: int
        """
        return self._changes.tables

    def __aiter__(self) -> AsyncIterator[NotedChange]:
        return self._gate()

    async def _gate(self) -> AsyncIterator[NotedChange]:
        async with aclosing(self._changes.stream()) as stream:
            held: list[NotedChange] = []
            async for change in stream:
                held.append(change)
                if not change.whole_table:
                    break
            _notes, fits = await self._sizing
            if not fits:
                raise _CoverageOverflows
            for change in held:
                yield change
            async for change in stream:
                yield change


class _StreamedChanges:
    """the coverage's notes folded into changes as a cursor reads them: whole tables first, then each table's
    distinct rows in chunks of :data:`_RESTORE_CHUNK_ROWS`, so memory holds one chunk.

    :ivar tables: the tables streamed so far
    """

    def __init__(
        self, pool: _Pool, owner: str, slots: tuple[int, int], since: datetime | None, folder: RestoreFolder
    ) -> None:
        self._pool = pool
        self._folder = folder
        self._owner = owner
        self._current, self._previous = slots
        self._since = since
        self.tables = 0
        #: the table the last keyed change streamed was of, so a table is counted once
        self._last_table: str | None = None

    def __aiter__(self) -> AsyncIterator[NotedChange]:
        return self._stream()

    def stream(self) -> AsyncGenerator[NotedChange]:
        """the changes, as a generator its reader can close (and so give the connection back) before its end.

        :return: the generator
        :rtype: AsyncGenerator[NotedChange]
        """
        return self._stream()

    async def _stream(self) -> AsyncGenerator[NotedChange]:
        async with self._pool.acquire() as conn, conn.transaction():
            reaches: dict[tuple[str, str | None], list[Any]] = {}
            # the whole notes that cover their tenant's rows (a load's names only its prefixes): read here once, so
            # the keyed read below leaves those rows out against this short list, not a second read of the coverage
            covering: dict[tuple[str, str | None], None] = {}
            for row in await whole_read(
                conn.fetch(
                    # a whole note's columns are its write's reach (NOTE_REACH_SETTING): a load's prefixes, a reset
                    f"SELECT DISTINCT table_name, tenant, columns::text AS reach, "  # noqa: S608 - trusted identifiers
                    f"(columns IS NULL OR NOT columns ? 'prefixes') AS covers FROM {_coverage_sql(self._owner, self._current, self._previous)} n "
                    "WHERE n.key_text IS NULL AND ($1::timestamptz IS NULL OR n.noted_at >= $1)",
                    self._since,
                ),
                kind="restore",
                reason="a lost change log is put back from every note of the ring's coverage, its current and previous slots",
                ruling="Pace 2026-11-14: the restore's full read of the coverage is a cold-start read (whole_read), as TRUNCATE (11-11)",
                proof="packages/core/tests/unit/collections/test_change_note_restore_reads.py::test_the_coverage_is_read_whole_only_by_a_restore",
                speed="packages/core/tests/integration/test_change_note_restore_speed_live.py::test_a_restore_reads_its_coverage_within_twice_the_bare_reads",
            ):
                pair = (str(row["table_name"]), None if row["tenant"] is None else str(row["tenant"]))
                reaches.setdefault(pair, []).append(row["reach"])
                if row["covers"]:
                    covering[pair] = None
            for noted, tenant in sorted(reaches, key=lambda pair: (pair[0], pair[1] or "")):
                self.tables += 1
                # a whole-table change carries no row: under its schema for a schema table, else on _all
                yield _whole_change(noted, tenant, reaches[(noted, tenant)])
            # a row its tenant's whole-table change (or the table's _all one) already covers is not named again; a
            # load's names only its prefixes, so a row beside them is named (one inside them twice, which is harmless).
            # WHY no DISTINCT and the byte ("C") order (o97, 2026-11-14, test_change_note_restore_speed_live at 200k
            # notes, PG 16 best of 5): the old DISTINCT + NOT EXISTS over a second read of the coverage, sorted in the
            # database's collation, took ~330 ms of the statement alone (a hash aggregate and a locale sort, both
            # spilling) against ~90 ms for the bare read of every note; this read sorted in byte order takes ~180 ms.
            # Grouping needs only equal values next to each other, which any order gives, and byte order skips the
            # locale's comparisons. A chunk is cut only where the key changes, so a restored change names each
            # distinct row once.
            # WHY DISTINCT ON over that same order (owner #101, 2026-11-14, the same file's over-bound case, each row
            # noted twice, PG 16 best of 5): the database drops a repeated note as it walks its own sort (a Unique over
            # the sorted rows, no hash), so half the rows cross the wire and reach the fold: 190 ms -> 144 ms for the
            # read alone; with every row noted once it costs nothing (196 vs 191 ms).
            # WHY COPY text folded in a worker process (L1, 2026-11-15, Pace: no CPU work in a serving process): read
            # as rows and folded here, a restore held the event loop up to 29 ms a turn (see
            # threetears.core.change_note_fold for the numbers); now the loop only moves the COPY's bytes and the
            # records the worker hands back. Each value is written as JSON text, so a raw line end is always a note's
            # end and a page is cut at its last one. The read itself is _folded's.
            # closed with the stream: a reader stopping early (a coverage that does not fit) must see the COPY ended
            # before the transaction ends (_folded's finally), never left to the collector
            async with aclosing(
                self._folded(conn, [noted for noted, _ in covering], [tenant for _, tenant in covering])
            ) as folded:
                async for change in folded:
                    yield change

    async def _folded(
        self, conn: Any, covered: list[str], covered_tenants: list[str | None]
    ) -> AsyncGenerator[NotedChange]:
        """the coverage's keyed notes, folded into changes by the restore's worker process a page at a time.

        The ``COPY`` streams into a buffer while the worker folds the page before: the loop moves bytes only. A
        page's last chunk may go on in the next page, so the worker answers the bytes it folded and the rest is
        handed again at the next page's front (:func:`~threetears.core.change_note_fold_job._fold_page`).

        :param conn: the connection, inside the restore's transaction
        :ptype conn: Any
        :param covered: the tables a whole-table note covers (beside ``covered_tenants``): their rows are not read
        :ptype covered: list[str]
        :param covered_tenants: each one's tenant, ``None`` for every tenant
        :ptype covered_tenants: list[str | None]
        :return: the changes, in the read's order
        :rtype: AsyncGenerator[NotedChange]
        :raises GenerationUnavailableError: when a note cannot be folded
        """
        buffer = bytearray()
        arrived = asyncio.Event()
        room = asyncio.Event()
        room.set()

        dropping = False

        async def sink(chunk: bytes) -> None:
            if dropping:
                return
            buffer.extend(chunk)
            arrived.set()
            if len(buffer) >= _RESTORE_BUFFER_BYTES:
                room.clear()
                await room.wait()

        copying = asyncio.ensure_future(
            whole_read(
                conn.copy_from_query(
                    'SELECT DISTINCT ON (n.table_name COLLATE "C", n.tenant COLLATE "C", n.key_text COLLATE "C", n.columns) '
                    "to_json(n.table_name)::text, to_json(n.tenant)::text, array_to_json(n.keys)::text, n.columns::text "
                    f"FROM {_coverage_sql(self._owner, self._current, self._previous)} n "  # noqa: S608 - trusted identifiers
                    "WHERE n.key_text IS NOT NULL AND ($1::timestamptz IS NULL OR n.noted_at >= $1) "
                    "AND NOT EXISTS (SELECT 1 FROM unnest($2::text[], $3::text[]) w(table_name, tenant) "
                    "WHERE w.table_name = n.table_name AND (w.tenant IS NULL OR w.tenant = n.tenant)) "
                    'ORDER BY n.table_name COLLATE "C", n.tenant COLLATE "C", n.key_text COLLATE "C", n.columns',
                    self._since,
                    covered,
                    covered_tenants,
                    output=sink,
                    format="csv",
                    force_quote=True,
                ),
                kind="restore",
                reason="a lost change log is put back from every note of the ring's coverage, its current and previous slots",
                ruling="Pace 2026-11-14: the restore's full read of the coverage is a cold-start read (whole_read), as TRUNCATE (11-11)",
                proof="packages/core/tests/unit/collections/test_change_note_restore_reads.py::test_the_coverage_is_read_whole_only_by_a_restore",
                speed="packages/core/tests/integration/test_change_note_restore_speed_live.py::test_a_restore_reads_its_coverage_within_twice_the_bare_reads",
            )
        )
        copying.add_done_callback(lambda _done: arrived.set())
        form = (
            _KEYED_FORM.head,
            _KEYED_FORM.middle,
            _KEYED_FORM.tail,
            _KEYED_FORM.row_head,
            _KEYED_FORM.row_columns,
            _KEYED_FORM.row_tail,
        )
        carried = b""
        want = _RESTORE_PAGE_BYTES
        try:
            while True:
                while not copying.done() and len(carried) + len(buffer) < want:
                    arrived.clear()
                    room.set()
                    await arrived.wait()
                done = copying.done()
                if done:
                    copying.result()
                # a page of about the page's bytes, at a note's end, however much arrived meanwhile: what the loop
                # copies at once stays small
                room_left = max(want - len(carried), 1)
                if done and len(buffer) <= room_left:
                    cut = len(buffer)
                else:
                    cut = buffer.rfind(b"\n", 0, room_left) + 1 or buffer.rfind(b"\n") + 1
                    if not cut:
                        # one note longer than a page: wait for its end
                        want = len(carried) + len(buffer) + _RESTORE_PAGE_BYTES
                        continue
                final = done and cut == len(buffer)
                with memoryview(buffer) as view:
                    page = b"".join((carried, view[:cut]))
                del buffer[:cut]
                room.set()
                if not page:
                    return
                folded, records = await self._folder.fold(
                    page, final=final, form=form, chunk_rows=_RESTORE_CHUNK_ROWS, max_bytes=_KEYED_FORM.max_bytes
                )
                carried = page[folded:]
                # a page holding no whole chunk yet is handed again with more: never the same page twice
                want = max(_RESTORE_PAGE_BYTES, 2 * len(carried))
                handed = 0
                for _number, chunk in groupby(records, key=itemgetter(0)):
                    held = list(chunk)
                    _, table, tenant, _ = held[0]
                    if table != self._last_table:
                        self.tables += 1
                        self._last_table = table
                    yield NotedChange(table=table, tenant=tenant, bodies=tuple(body for *_, body in held))
                    handed += 1
                    if handed % _RESTORE_TURN_CHANGES == 0:
                        # a reader that takes each change without waiting (none appends) still lets the loop turn
                        await asyncio.sleep(0)
                if final:
                    return
        finally:
            if not copying.done():
                # the reader stopped early (a coverage that does not fit): the COPY is read to its end and dropped,
                # never cancelled -- a cancelled COPY leaves the connection mid-operation, and the transaction's end
                # then fails ("another operation is in progress")
                dropping = True
                buffer.clear()
                room.set()
                try:
                    await asyncio.shield(copying)
                except asyncio.CancelledError:
                    await cancel_and_wait(copying)
                    raise
                except Exception as exc:  # prawduct:allow prawduct/broad-except -- the reader already stopped; the COPY's own failure is only said
                    contain(exc, log=log, what="a restore's dropped COPY of keyed notes failed", level="warning")


def _whole_change(table: str, tenant: str | None, reaches: Sequence[Any]) -> NotedChange:
    """one table's whole-table notes of one tenant folded into one change, keeping the reach they name.

    A reset among them makes the change a reset (it reads the table as on a first start, which covers any load). A
    note naming no reach (an older writer's, a bare truncate, a whole-marked copy) makes it the plain whole table,
    the one answer that covers it. Otherwise it names every prefix its loads named, each once.

    :param table: the table
    :ptype table: str
    :param tenant: the notes' tenant, ``None`` for ``_all``
    :ptype tenant: str | None
    :param reaches: each distinct whole note's reach as JSON text, or ``None`` for a note naming none
    :ptype reaches: Sequence[Any]
    :return: the change
    :rtype: NotedChange
    :raises GenerationUnavailableError: when a note's reach is not one this module writes
    """
    prefixes: dict[tuple[tuple[str, str], ...], None] = {}
    plain = False
    for raw in reaches:
        value = None if raw is None else json.loads(raw) if isinstance(raw, str) else raw
        if value is None:
            plain = True
        elif isinstance(value, dict) and value.get("reason") == RESET_REASON:
            return NotedChange(table=table, tenant=tenant, whole_table=True, reason=RESET_REASON)
        elif isinstance(value, dict) and isinstance(value.get("prefixes"), list) and value["prefixes"]:
            for prefix in value["prefixes"]:
                if not isinstance(prefix, dict) or not prefix:
                    raise GenerationUnavailableError(
                        f"a whole-table note of {table!r} names a malformed prefix: {raw!r}"
                    )
                prefixes[tuple(sorted((str(column), str(each)) for column, each in prefix.items()))] = None
        else:
            raise GenerationUnavailableError(f"a whole-table note of {table!r} names a reach not known: {raw!r}")
    if plain or not prefixes:
        return NotedChange(table=table, tenant=tenant, whole_table=True)
    return NotedChange(
        table=table, tenant=tenant, whole_table=True, prefixes=tuple(dict(prefix) for prefix in sorted(prefixes))
    )


@dataclass(frozen=True, slots=True)
class NotedRows:
    """what the change notes say a table's writes since a time changed.

    :ivar rows: each distinct row the notes name that no whole-table note covers, with its invalidation columns and
        its tenant
    :ivar whole_table: a note of the whole table (a truncate, a file load) is among them
    :ivar whole_tenants: each such note's tenant, ``None`` for ``_all``
    :ivar whole: each tenant's whole-table notes folded into one change, keeping the reach they name (a load's key
        prefixes, an operator's reset), as a restore folds them (:func:`_whole_change`). A plain or reset change covers
        its tenant's rows (``_all``'s, every tenant's), never another schema's; a load's names only its prefixes, so
        a row beside them is still named
    """

    rows: tuple[ChangedRow, ...] = ()
    whole_table: bool = False
    whole_tenants: tuple[str | None, ...] = ()
    whole: tuple[NotedChange, ...] = ()


#: a note's columns :func:`noted_rows_since` reads, each slot's alike
_NOTE_COLUMNS: Final = ("key_text", "keys", "columns", "tenant")

#: the ring row's columns :func:`noted_rows_since` reads (``r``): its slot, switch and wait, and the end of the time a
#: copy of the slot read before the last switch may still write the previous table (the switch, plus the larger of the
#: fence and the wait)
_ROWS_RING_COLUMNS: Final = (
    "current_ring",
    "switch_no",
    "previous_switched_at",
    "wait_seconds",
    Selected(
        "t.switched_at + make_interval(secs => GREATEST(t.fence_seconds, t.wait_seconds))", "writes_previous_until"
    ),
)

#: a slot's one-time condition on the ring row (``$1`` the slot, ``$2`` the ring's size, ``$3`` since): it is the
#: current table, or the previous one while the reach (since less half the wait) is before the last switch's writes
#: there could end
_SLOT_READ: Final = (
    "r.current_ring = $1 OR ((r.current_ring - 1 + $2) % $2 = $1 "
    "AND $3::timestamptz - make_interval(secs => r.wait_seconds / 2) < r.writes_previous_until)"
)


def _noted_rows_step(schema: str, table: str, since: datetime) -> Step:
    """the notes :func:`noted_rows_since` reads beside the ring row (``r``): each note table's notes of ``table`` from
    the reach on, the distinct (key, columns, tenant) of them.

    Each note table's leg is read only when it is the current or the previous table (:data:`_SLOT_READ`, a one-time
    filter on the ring row); the previous one only while the reach is before the last switch's fence has passed (a
    copy of the slot read before the switch writes there for at most the fence after it, so past that it holds no note
    from the reach on); from the reach (``since`` less half the ring's wait); and only when the coverage reaches back
    that far (before a first switch nothing was ever emptied). Each leg is answered from the ``(table_name,
    noted_at)`` index, which carries the columns read (:func:`~threetears.core.ddl.change_notes.change_notes_ddl`).

    :param schema: the schema holding the ring
    :ptype schema: str
    :param table: the table whose notes are read
    :ptype table: str
    :param since: the oldest write's time, by the database's clock
    :ptype since: datetime
    :return: the step
    :rtype: Step
    """
    coverage = Fragment(
        "r.switch_no = 0 OR $1::timestamptz - make_interval(secs => r.wait_seconds / 2) >= r.previous_switched_at",
        (since,),
    )
    from_reach = Fragment(
        "t.table_name = $1 AND t.noted_at >= $2::timestamptz - make_interval(secs => r.wait_seconds / 2)",
        (table, since),
    )
    # one leg a slot of the ring (NOTE_RING_SIZE, 3), each its own index read
    first = scan(
        f"{schema}.{note_table(0)}",
        _NOTE_COLUMNS,
        where=[Fragment(_SLOT_READ, (0, NOTE_RING_SIZE, since)), coverage, from_reach],
    )
    second = scan(
        f"{schema}.{note_table(1)}",
        _NOTE_COLUMNS,
        where=[Fragment(_SLOT_READ, (1, NOTE_RING_SIZE, since)), coverage, from_reach],
    )
    third = scan(
        f"{schema}.{note_table(2)}",
        _NOTE_COLUMNS,
        where=[Fragment(_SLOT_READ, (2, NOTE_RING_SIZE, since)), coverage, from_reach],
    )
    return distinct(union(first, second, third, keep_all=True), _NOTE_COLUMNS)


async def noted_rows_since(pool: Any, *, notes_schema: str, table: str, since: datetime, cap: int) -> NotedRows | str:
    """the rows the change notes say ``table``'s writes changed since ``since``, or why they cannot say.

    For a writer-side process that knows a table was written and not which rows (the write-mark reconciler):
    every committed write to a noted table left a note with its keys in its own transaction. The notes are
    read from the ring's current and previous tables, from ``since`` less half the ring's wait (a transaction
    writes its notes up to that long before its commit), and only when the coverage reaches back that far.

    :param pool: the L3 pool
    :ptype pool: Any
    :param notes_schema: the schema holding the ring
    :ptype notes_schema: str
    :param table: the table
    :ptype table: str
    :param since: the oldest write's time, by the database's clock
    :ptype since: datetime
    :param cap: the most distinct rows to name; more is answered ``"too_many_rows"``
    :ptype cap: int
    :return: the rows; or ``"notes_unavailable"`` (no ring), ``"not_covered"`` (the coverage starts after
        ``since``), ``"no_notes"`` (no note of the table since), ``"too_many_rows"``
    :rtype: NotedRows | str
    :raises GenerationUnavailableError: when a whole-table note names a reach this module never writes
    """
    try:
        # the ring row and its notes in one statement (read_row_beside): one round trip, the ring and its notes read in
        # one snapshot, as the reconciler's ask of a marked table was made one statement for (364941102: on YugabyteDB
        # 2.56 ms at 1M notes, 1.60x its bare read; the two-statement form it replaced answered the same in every case).
        # Through the helper (2026-11-15) the plan is the hand-written one's, node for node (each slot a one-time
        # filter over its (table_name, noted_at) index read); building the statement costs ~0.09 ms of Python a call.
        # Its budget: test_noted_rows_scale_live (YugabyteDB, within 2x the bare read)
        found = await read_row_beside(
            pool,
            f"{notes_schema}.{RING_TABLE}",  # the helper quotes it a segment at a time
            by={"id": 1},
            columns=_ROWS_RING_COLUMNS,
            beside=_noted_rows_step(notes_schema, table, since),
            limit=cap + 1,
            present="noted",
        )
    except asyncpg.UndefinedTableError, asyncpg.InvalidSchemaNameError:
        # NOSILENT: no ring installed here: the caller names the whole table instead, said and counted there
        return "notes_unavailable"
    if not found:
        return "notes_unavailable"
    ring = found[0]
    # the reach as the statement's legs compute it (since less half the wait, wait_seconds a DOUBLE PRECISION), as
    # _noted_keys_read computes it from the same ring row
    reach = since - timedelta(seconds=float(ring["wait_seconds"]) / 2)
    # before a first switch nothing was ever emptied: every note since the triggers were given is there
    if int(ring["switch_no"]) != 0 and reach < ring["previous_switched_at"]:
        return "not_covered"
    # each note's values by position (the select list: the ring's five _ROWS_RING_COLUMNS, then noted, key_text, keys,
    # columns, tenant): at 500 notes the rows' handling, not the statement, was the larger part of the call
    notes = [note for note in found if note[5]]
    if not notes:
        return "no_notes"
    if len(notes) > cap:
        return "too_many_rows"
    # the ring holds every schema's notes of the table: each tenant's whole-table notes fold into one change with the
    # reach they name (a whole-table note keeps it as its columns), as a restore folds them
    reaches: dict[str | None, list[Any]] = {}
    for note in notes:
        if note[6] is None:
            reaches.setdefault(None if note[9] is None else str(note[9]), []).append(note[8])
    whole = tuple(_whole_change(table, tenant, reached) for tenant, reached in reaches.items())
    # a plain or reset change covers its own tenant's rows (or every tenant's, when it is _all's), never another
    # schema's; a load's names only its prefixes, so a row beside them is still named
    covering = {change.tenant for change in whole if not change.prefixes}
    rows: list[ChangedRow] = []
    if None not in covering:
        # the statement answers each (key, columns, tenant) once (its DISTINCT), so each note is one row
        for note in notes:
            if note[6] is None:
                continue
            token = None if note[9] is None else str(note[9])
            if token in covering:
                continue
            columns = note[8]
            rows.append(
                ChangedRow(
                    ids=[str(key) for key in note[7]],
                    columns=dict(_columns_of(columns)) or None if columns is not None else None,
                    tenant=token,
                )
            )
    return NotedRows(rows=tuple(rows), whole_table=bool(whole), whole_tenants=tuple(reaches), whole=whole)


#: a note's ``noted_at`` as whole microseconds since the epoch: a page compares its notes' times (where an instant
#: starts and ends) and turns one into its cursor, and an integer decodes several times faster than a datetime (on
#: 20,000 notes about 3 ms of a page's read, on Postgres and YugabyteDB). Exact: date_part's epoch is the stored
#: microseconds over 1e6 in a double, off by at most a quarter microsecond before 2112, so times 1e6 and rounded it
#: is the stored value (checked on 322,100 notes; :func:`_noted_time` turns it back)
_NOTED_MICROS: Final = "round(date_part('epoch', noted_at) * 1000000)::bigint"

#: the epoch :data:`_NOTED_MICROS` counts from
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)


def _noted_time(micros: int) -> datetime:
    """a note's ``noted_at`` from its :data:`_NOTED_MICROS`.

    :param micros: microseconds since the epoch
    :ptype micros: int
    :return: the time, in UTC
    :rtype: datetime
    """
    return _EPOCH + timedelta(microseconds=micros)


def _noted_keys_sql(
    owner: str, slot: int, *, all_tenants: bool, every_tenant: bool = False, arrays: bool = False
) -> str:
    """one ring note table's notes of a table and tenant in a bracket, after a time, in ``noted_at`` order.

    Answered from the ``(table_name, noted_at)`` index, which carries ``keys``, ``key_text``, ``columns`` and
    ``tenant`` (:func:`~threetears.core.ddl.change_notes.change_notes_ddl`): an index range scan of the table's notes
    in the bracket, in the index's order, stopped at the limit; no sort, no other table's notes. A tenant's notes are
    picked by ``tenant = $2``, which YugabyteDB filters in its storage layer (``IS NOT DISTINCT FROM`` it filters
    only after the rows reach the query layer).

    Each note's keys' bytes (:func:`_key_bytes`) are counted by the database, so the page's bound costs the caller no
    per-key work: from ``key_text``, the keys joined by ``chr(31)``, whose length is theirs plus one less than their
    count. A whole-table note has no ``key_text``, so its bytes are NULL. The few notes with invalidation columns
    add theirs in the caller.

    :param owner: the quoted schema holding the ring
    :ptype owner: str
    :param slot: the note table's place in the ring
    :ptype slot: int
    :param all_tenants: the notes of ``_all`` (``tenant IS NULL``) rather than of the tenant ``$2`` names
    :ptype all_tenants: bool
    :param every_tenant: every tenant's notes of the table (``$2`` unused): an operator's check
    :ptype every_tenant: bool
    :param arrays: answer each note's keys as their array where its key text cannot be split back into them
        (:data:`_KEYS_UNLESS_SPLIT`); else its separators, ``cardinality(keys) - 1`` (:data:`_SEPARATORS`), which the
        page checks against its key text's own and asks again in this form only when they differ
    :ptype arrays: bool
    :return: the statement, answering ``noted`` (``noted_at`` in microseconds since the epoch, :data:`_NOTED_MICROS`),
        ``keys`` (or ``separators``), ``columns``, ``bytes`` by position: ``$1`` table,
        ``$2`` tenant (NULL for ``_all``), ``$3`` after (the last time read, or one microsecond before the bracket:
        both ends of the range are index conditions), ``$4`` until, ``$5`` the most notes (NULL for all), ``$6`` to
        ``$8`` the planner settings to put back (:data:`_RESTORED`)
    :rtype: str
    """
    if every_tenant:
        # $2 still names its type, so the statement takes the same parameters
        tenant = "($2::text IS NULL OR TRUE)"
    elif all_tenants:
        tenant = "tenant IS NULL AND $2::text IS NULL"
    else:
        tenant = "tenant = $2::text"
    return (  # each interpolated piece is a fixed name this module builds
        f"SELECT {_NOTED_MICROS} AS noted, {_KEYS_UNLESS_SPLIT if arrays else _SEPARATORS}, columns, "
        "char_length(key_text) + 7 * cardinality(keys) + 1 AS bytes, key_text "
        f"FROM {owner}.{quote_identifier(note_table(slot))} "  # noqa: S608
        f"WHERE table_name = $1 AND noted_at > $3 AND noted_at <= $4 AND {tenant} AND {_RESTORED} "
        "ORDER BY noted_at LIMIT $5"
    )


#: the database clock's grain: ``timestamptz`` holds microseconds
_CLOCK_GRAIN: Final = timedelta(microseconds=1)

#: a page's reads go through the ``(table_name, noted_at)`` index in its order, stopped at the limit. The planner
#: judges a bracket's notes from the column statistics as if a table's notes were spread over the night, so it counts
#: too few of a large commit's (a burst of one table's notes in a second or two) to stop early at the limit, and reads
#: the whole rest of the bracket by bitmap and sorts it: every page reads every later page's notes again. Told to keep
#: the index's order (on the ring row's read; each notes read puts the settings back as it starts, :data:`_RESTORED`,
#: so nothing is left on a pooled connection), each page reads only its own notes. Measured on a ring of 322,000 notes and a 20,000-row commit,
#: tables freshly written and analyzed: 212,062 rows scanned for 20,162 named (10.52x), told so 20,262 (1.005x), each
#: page an Index Only Scan with both ``noted_at`` bounds in its Index Cond, on Postgres and YugabyteDB.
_INDEX_ORDER: Final = (
    "set_config('enable_bitmapscan', 'off', false), set_config('enable_seqscan', 'off', false), "
    "set_config('enable_sort', 'off', false)"
)

#: the settings :data:`_INDEX_ORDER` changes, as they were, read before it in the same statement: the ring row's
#: read reads this table-less row beside it (``read_row(beside=)``, ``FROM (<this>) AS b CROSS JOIN <ring> AS t``),
#: whose values are taken before the select list's ``set_config`` runs (``OFFSET 0`` keeps it a row of its own)
_SETTINGS_WERE: Final = Derived(
    "SELECT current_setting('enable_bitmapscan') AS bitmapscan, current_setting('enable_seqscan') AS seqscan, "
    "current_setting('enable_sort') AS sort OFFSET 0"
)

#: the ring row's columns a noted keys page reads: the settings as they were (:data:`_SETTINGS_WERE`), the planner
#: told (:data:`_INDEX_ORDER`, one item a setting), and the ring's slot, switch and fence
_PAGE_RING_COLUMNS: Final = (
    Selected("b.bitmapscan", "bitmapscan"),
    Selected("b.seqscan", "seqscan"),
    Selected("b.sort", "sort"),
    Selected("set_config('enable_bitmapscan', 'off', false)", "told_bitmapscan"),
    Selected("set_config('enable_seqscan', 'off', false)", "told_seqscan"),
    Selected("set_config('enable_sort', 'off', false)", "told_sort"),
    "current_ring",
    "switch_no",
    "previous_switched_at",
    "wait_seconds",
    Selected(
        "t.switched_at + make_interval(secs => GREATEST(t.fence_seconds, t.wait_seconds))", "writes_previous_until"
    ),
)

#: a notes read's one-time condition putting the settings back (``$6`` to ``$8``, as they were): run once as the
#: statement starts, after it was planned under :data:`_INDEX_ORDER`, so the page leaves nothing on its connection
#: and needs no transaction of its own (a round trip each to open and end it, on every page)
_RESTORED: Final = (
    "(SELECT set_config('enable_bitmapscan', $6, false) || set_config('enable_seqscan', $7, false) "
    "|| set_config('enable_sort', $8, false)) IS NOT NULL"
)


#: the note's keys as an array only where its ``key_text`` (the keys joined by ``chr(31)``) cannot be split back into
#: them: a key value holding ``chr(31)`` adds a piece. Otherwise NULL, and the page splits ``key_text`` (one text a
#: note decodes in a fraction of an array's time)
_KEYS_UNLESS_SPLIT: Final = (
    "CASE WHEN cardinality(keys) = 1 + char_length(key_text) - char_length(replace(key_text, chr(31), '')) "
    "THEN NULL ELSE keys END AS keys"
)

#: a note's separators as its keys array says (``cardinality(keys) - 1``): the page counts its key text's own in C
#: (``str.count``) and reads the notes again with :data:`_KEYS_UNLESS_SPLIT` only when one differs (a key value holding
#: ``chr(31)``, rare). WHY: the database's own check, that CASE (its ``replace`` and the array it may answer), was a
#: quarter of a 5,001-note page's read on YugabyteDB: 4.0 ms a page against 3.0 ms with this column, lane 1 under
#: measure-lock (2026-11-15); the page's whole read had run at 1.61-1.93x of its 2x budget
#: (test_noted_keys_page_live.py::TestACommitsKeysAreReadBackFromItsNotes::
#: test_reading_every_page_keeps_within_twice_a_bare_read_of_the_same_notes)
_SEPARATORS: Final = "cardinality(keys) - 1 AS separators"

#: a note's key text's separators, counted in C
_COUNT_SEPARATORS: Final = methodcaller("count", _KEY_SEPARATOR)

#: a page's note's keys (NULL: split its key text), columns, bytes and key text, by their place in
#: :func:`_noted_keys_sql`'s select list
_KEYS_AT: Final = itemgetter(1)
_COLUMNS_AT: Final = itemgetter(2)
_BYTES_AT: Final = itemgetter(3)
_KEY_TEXT_AT: Final = itemgetter(4)

#: a note's key text split back into its keys
_SPLIT_KEYS: Final = methodcaller("split", _KEY_SEPARATOR)


def _split_whole(notes: Sequence[Any]) -> bool:
    """whether every note read with :data:`_SEPARATORS` splits its key text back into its keys: each holds a key text
    (no whole-table note) whose separators are as many as its keys array says. Passes of C, no Python step a note.

    :param notes: the notes, by :func:`_noted_keys_sql`'s positions
    :ptype notes: Sequence[Any]
    :return: whether they do
    :rtype: bool
    """
    texts = list(map(_KEY_TEXT_AT, notes))
    return None not in texts and list(map(_COUNT_SEPARATORS, texts)) == list(map(_KEYS_AT, notes))


def _key_bytes(keys: Sequence[str], columns: Sequence[tuple[str, str | None]] = ()) -> int:
    """the bytes a page counts for one note: its keys and its invalidation columns, each with 8 bytes of framing.

    The page's own count is the database's (:func:`_noted_keys_sql`'s ``bytes``); this is that count's definition.

    :param keys: the note's key values
    :ptype keys: Sequence[str]
    :param columns: its invalidation columns' names and values
    :ptype columns: Sequence[tuple[str, str | None]]
    :return: the bytes
    :rtype: int
    """
    return (
        sum(len(value) for value in keys)
        + 8 * len(keys)
        + sum(len(name) + len(value or "") + 8 for name, value in columns)
    )


async def noted_keys_page(
    conn: Any,
    *,
    notes_schema: str,
    table: str,
    tenant_token: str | None,
    noted_between: tuple[datetime, datetime],
    cursor: NotedKeysCursor | None,
    limit: int,
    max_bytes: int,
    every_tenant: bool = False,
) -> NotedKeysPage | str:
    """a page of the keys ``table``'s notes of ``tenant_token`` name in a commit's notes bracket, or why they cannot be read.

    For the broker's answer to a copy follower holding a keys-unknown record whose commit named too many rows
    (:attr:`~threetears.core.collections.epoch_change.EpochChange.noted_between`). The commit wrote a note per row in
    its own transaction, each stamped on its connection's clock inside the bracket, so the bracket's notes name every
    key it changed (and perhaps keys other commits changed then, which a follower reads again harmlessly). The notes
    are read from the ring's previous and current tables, one at a time, in ``noted_at`` order through the index, and
    continued by the last ``noted_at`` read: no offset, no sort. A page ends at ``limit`` notes or ``max_bytes`` of
    keys and columns, whichever first, over both tables together, at a whole ``noted_at`` (a page never splits the
    notes of one instant, so continuing past it skips none); one instant's notes past the page's bound are read whole,
    which only a burst at one microsecond could make large.

    :param conn: a connection to the L3 database (the broker's, on the platform schema), or a pool to take one
        from; the page is read on that one connection, in two round trips (the ring row, then the notes) with no
        transaction of its own: a page was four (its transaction's start and end besides), on Postgres a fifth of a
        20,000-note commit's paged read
    :ptype conn: Any
    :param notes_schema: the schema holding the ring
    :ptype notes_schema: str
    :param table: the table, as its notes name it
    :ptype table: str
    :param tenant_token: the tenant whose notes are read (the asker's schema's), ``None`` for ``_all``: never another's
    :ptype tenant_token: str | None
    :param noted_between: the commit's bracket, ``(from, until)`` on the database's clock
    :ptype noted_between: tuple[datetime, datetime]
    :param cursor: where the last page ended; ``None`` for the first
    :ptype cursor: NotedKeysCursor | None
    :param limit: the most notes a page reads
    :ptype limit: int
    :param max_bytes: the most bytes of keys a page holds
    :ptype max_bytes: int
    :param every_tenant: read every tenant's notes of the table, ``tenant_token`` unused: for an operator's check of a
        table whose rows carry their tenant in their key (a backup's live check), never a tenant's own reader
    :ptype every_tenant: bool
    :return: the page; or ``"notes_unavailable"`` (no ring) or ``"not_covered"`` (the ring no longer holds every
        note of the bracket, or moved past the cursor's table between pages)
    :rtype: NotedKeysPage | str
    """
    if limit < 1 or max_bytes < 1:
        raise ValueError("a page reads at least one note and one byte")
    owner = quote_identifier(notes_schema)
    since, until = noted_between
    # one connection: a pool's own, or the one given
    held = conn.acquire() if not hasattr(conn, "transaction") else nullcontext(conn)
    async with held as one:
        return await _noted_keys_page(
            one, notes_schema, owner, table, tenant_token, since, until, cursor, limit, max_bytes, every_tenant
        )


async def _noted_keys_page(
    conn: Any,
    schema: str,
    owner: str,
    table: str,
    tenant: str | None,
    since: datetime,
    until: datetime,
    cursor: NotedKeysCursor | None,
    limit: int,
    max_bytes: int,
    every: bool,
) -> NotedKeysPage | str:
    try:
        # the ring row by its key, and the planner told to keep the index's order for the notes read after it
        # (_INDEX_ORDER), the settings as they were read first, beside it (_SETTINGS_WERE): one round trip (f93ef9522:
        # a page in two round trips, not four). Told only where the ring row is
        ring = await read_row(
            conn,
            f"{schema}.{RING_TABLE}",  # the helper quotes it a segment at a time
            by={"id": 1},
            columns=_PAGE_RING_COLUMNS,
            beside=_SETTINGS_WERE,
            records=True,
        )
    except asyncpg.UndefinedTableError, asyncpg.InvalidSchemaNameError:
        # NOSILENT: no ring installed here: the asker is behind on the table, said and counted there
        return "notes_unavailable"
    if ring is None:
        return "notes_unavailable"
    was = (ring["bitmapscan"], ring["seqscan"], ring["sort"])
    # each notes read puts the settings back as it starts (_RESTORED); a read after another in the page tells the
    # planner again first. A page that ends with them still told (no notes read since, or a failure) puts them back
    told = [True]

    async def notes(sql: str, *args: Any) -> Any:
        if not told[0]:
            await conn.execute(f"SELECT {_INDEX_ORDER}")
            told[0] = True
        found = await conn.fetch(sql, *args, *was)
        # put back as the read started; a read that failed may not have, and is put back on the way out
        told[0] = False
        return found

    try:
        return await _noted_keys_read(
            notes, owner, table, tenant, since, until, cursor, limit, max_bytes, ring, every=every
        )
    finally:
        if told[0]:
            await _put_back(conn, was)


async def _put_back(conn: Any, was: tuple[str, str, str]) -> None:
    """put the planner settings a page told back as they were, saying so if it cannot (the page's own error stands).

    :param conn: the page's connection
    :ptype conn: Any
    :param was: ``enable_bitmapscan``, ``enable_seqscan``, ``enable_sort`` as they were
    :ptype was: tuple[str, str, str]
    :return: nothing
    :rtype: None
    """
    try:
        await conn.execute(
            "SELECT set_config('enable_bitmapscan', $1, false), set_config('enable_seqscan', $2, false), "
            "set_config('enable_sort', $3, false)",
            *was,
        )
    # prawduct:allow prawduct/broad-except -- a page's own failure is raised by its caller; this one is contained
    except Exception as exc:
        contain(
            exc,
            log=log,
            what="a noted keys page could not put its planner settings back on its connection",
            level="warning",
        )


async def _noted_keys_read(
    notes: Callable[..., Awaitable[Any]],
    owner: str,
    table: str,
    tenant: str | None,
    since: datetime,
    until: datetime,
    cursor: NotedKeysCursor | None,
    limit: int,
    max_bytes: int,
    ring: Any,
    *,
    every: bool = False,
) -> NotedKeysPage | str:
    current = int(ring["current_ring"])
    previous = (current - 1 + NOTE_RING_SIZE) % NOTE_RING_SIZE
    # a writer's cached slot may name the table it had up to half the wait after a switch (noted_rows_since's reach)
    reach = since - timedelta(seconds=float(ring["wait_seconds"]) / 2)
    if int(ring["switch_no"]) != 0 and reach < ring["previous_switched_at"]:
        return "not_covered"
    order = (previous, current)
    # a bracket from after the last writer could still name the previous table holds no note of it: the first page
    # starts at the current table, one read fewer (as noted_rows_since skips it)
    first = current if reach >= ring["writes_previous_until"] else previous
    at = cursor or NotedKeysCursor(slot=first)
    if at.slot not in order:
        return "not_covered"
    keys: list[tuple[str, ...]] = []
    columns: list[dict[str, str | None] | None] = []
    whole = False
    used = 0
    left = limit

    def page(next_at: NotedKeysCursor | None) -> NotedKeysPage:
        return NotedKeysPage(keys=tuple(keys), columns=tuple(columns), whole_table=whole, next=next_at)

    def take(notes: Sequence[Any], sizes: list[int | None], held: list[Any], arrays: bool) -> None:
        # each note by position (_noted_keys_sql's select list): noted_at, keys, columns, bytes (NULL: whole table).
        # Passes of C (map, itemgetter), never a Python step a note: at 20,000 notes those were the larger part of
        # the read. A whole-table note or one with columns (few) takes the slower way
        nonlocal whole
        if None in sizes:
            whole = True
            notes = [note for note, size in zip(notes, sizes, strict=True) if size is not None]
            held = [each for each, size in zip(held, sizes, strict=True) if size is not None]
        given = list(map(_KEYS_AT, notes)) if arrays else []
        if given.count(None) == len(given):
            # every note's key text splits back into its keys (the separators read checked it, _split_whole)
            keys.extend(map(tuple, map(_SPLIT_KEYS, map(_KEY_TEXT_AT, notes))))
        else:
            # a key value holding the separator (rare): that note's keys as their array
            keys.extend(
                tuple(_SPLIT_KEYS(note[4]) if array is None else array)
                for note, array in zip(notes, given, strict=True)
            )
        if held.count(None) == len(held):
            columns.extend(held)
        else:
            columns.extend([None if each is None else dict(_columns_of(each)) or None for each in held])

    for slot in order[order.index(at.slot) :]:
        # where this table's read starts: the cursor's time in its own table, else the bracket's start
        start = at.after if slot == at.slot else None
        # the time read up to, or a microsecond (the clock's grain) before the bracket: one range for the index
        after = start if start is not None else since - _CLOCK_GRAIN
        sql = _noted_keys_sql(owner, slot, all_tenants=tenant is None, every_tenant=every)
        arrays_sql = _noted_keys_sql(owner, slot, all_tenants=tenant is None, every_tenant=every, arrays=True)

        async def read(*args: Any) -> tuple[Any, bool]:
            # the notes with their separators; again with their keys' arrays only when a key text does not split
            # back into its keys (_SEPARATORS), or a whole-table note is among them
            found = await notes(sql, table, tenant, *args)
            if _split_whole(found):
                return found, False
            return await notes(arrays_sql, table, tenant, *args), True

        found, arrays = await read(after, until, left + 1)
        if not found:
            continue
        full = len(found) > left
        taken: Sequence[Any] = found[:left] if full else found
        if full:
            last = taken[-1][0]
            if taken[0][0] == last:
                # the notes are in noted_at order, so the first and the last at one time means every one is
                if left < limit:
                    # the page already holds the earlier table's notes: this table's first instant starts the next
                    return page(NotedKeysCursor(slot=slot, after=start))
                # one instant's notes fill the page: read that instant whole, so continuing past it skips none
                taken, arrays = await read(_noted_time(last) - _CLOCK_GRAIN, _noted_time(last), None)
            else:
                # end at a whole instant: the last instant's notes go to the next page
                end = len(taken) - 1
                while taken[end - 1][0] == last:
                    end -= 1
                taken = taken[:end]
        # past the page's bytes, the page ends before the first keyed note that starts an instant, once it holds a
        # key: the running count says where it passes the bound, and only the notes from there are looked at
        sizes: list[int | None] = list(map(_BYTES_AT, taken))
        held: list[Any] = list(map(_COLUMNS_AT, taken))
        if None in sizes or held.count(None) != len(held):
            counts = [
                0 if size is None else size if each is None else size + _key_bytes((), _columns_of(each))
                for size, each in zip(sizes, held, strict=True)
            ]
        else:
            counts = sizes  # type: ignore[assignment]  # no whole note: every size is an int
        counted = list(accumulate(counts, initial=used))
        # counted[i + 1] is the count with note i: the first note past the bound (the first, if the page already is)
        index = max(bisect_right(counted, max_bytes) - 1, 0)
        first_keyed = next((at_index for at_index, size in enumerate(sizes) if size is not None), len(sizes))
        while index < len(taken):
            note = taken[index]
            if note[3] is not None and (index == 0 or taken[index - 1][0] < note[0]) and (keys or first_keyed < index):
                take(taken[:index], sizes[:index], held[:index], arrays)
                return page(NotedKeysCursor(slot=slot, after=_noted_time(taken[index - 1][0]) if index else start))
            index += 1
        take(taken, sizes, held, arrays)
        used = counted[-1]
        if full:
            return page(NotedKeysCursor(slot=slot, after=_noted_time(taken[-1][0])))
        left -= len(taken)
        if left < 1:
            # the page is full at this table's last note; the next page reads on from it
            return page(NotedKeysCursor(slot=slot, after=_noted_time(taken[-1][0])))
    return page(None)


@dataclass(frozen=True, slots=True)
class _Found:
    """what one note table holds of the epochs a scrub checks.

    :ivar keyed: ``(table, joined key)`` of each keyed note found
    :ivar whole: the tables a whole-table note was found for
    :ivar any_note: the tables any note was found for
    """

    keyed: set[tuple[str, str]]
    whole: set[str]
    any_note: set[str]


def _first_unnoted(record: EpochChange, where: Sequence[int], found: dict[int, _Found]) -> str | None:
    """the first part of ``record`` no note in ``where`` covers, or ``None`` when every part is covered.

    :param record: the epoch
    :ptype record: EpochChange
    :param where: the note tables its notes must be in
    :ptype where: Sequence[int]
    :param found: what each note table holds
    :ptype found: dict[int, _Found]
    :return: the key not covered, ``"*"`` for a whole-table or keys-unknown epoch; ``None`` when covered
    :rtype: str | None
    """
    held = [found[index] for index in where]
    if not record.rows:
        return None if any(record.table in each.any_note for each in held) else "*"
    for row in record.rows:
        joined = _KEY_SEPARATOR.join(row.ids)
        if not any((record.table, joined) in each.keyed or record.table in each.whole for each in held):
            return "/".join(row.ids)
    return None


async def run_change_note_keeper(
    held: HeldLease, keeper: ChangeNoteKeeper, *, every: timedelta = _DEFAULT_KEEPER_EVERY
) -> None:
    """run the keeper's pass every ``every``, for as long as ``held`` is held.

    The one keeper: take the lease (``KVLease.hold``) and hand it here. A pass that fails is logged at
    ERROR and the next pass runs at the same interval. A restore waits for this pass, so ``every`` bounds
    how long writes wait for a lost log (``EpochGenerationSource(restore_wait=...)`` must be longer).

    :param held: the lease that makes this process the keeper
    :ptype held: HeldLease
    :param keeper: the keeper
    :ptype keeper: ChangeNoteKeeper
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
            contain(exc, log=log, what="a change-note keeper pass failed; the next pass runs", level="error")
        await asyncio.sleep(every.total_seconds())
