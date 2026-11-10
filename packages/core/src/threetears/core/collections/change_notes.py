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
empties the oldest with batched deletes (:meth:`ChangeNoteKeeper.empty_slot`), never ``TRUNCATE``. The ring
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
switches into a table whose emptying has not finished: such a switch waits, logged and counted
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
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

import asyncpg
from threetears.nats.change_log import ChangeLogState
from threetears.observe import contain, get_logger
from threetears.observe.metrics import counter, gauge

from threetears.core.collections.epoch_change import RESET_REASON, ChangedRow, EpochChange, check_whole_reach
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
    trigger_function_body,
    trigger_function_sql,
)
from threetears.core.ddl.ddl_lock import DdlLockPolicy, database_ddl_lock
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

#: notes deleted per statement when the keeper empties the idle table; each batch is its own transaction
_EMPTY_BATCH: Final = 1000

#: YugabyteDB's hash codes of a key: ``yb_hash_code`` answers 0 to 65535
_HASH_SPACE: Final = 65536

#: the batches' worth of notes the count that sizes an emptying's first band reads at most: past it the table is
#: taken to hold that many, so the first band is small and the bands grow from the notes each deletes
_BAND_COUNT_BATCHES: Final = 64


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

#: notes read per round trip when a restore streams the coverage
_RESTORE_PAGE: Final = 5000

#: distinct rows of one table per change a restore streams: the most it holds at once
_RESTORE_CHUNK_ROWS: Final = 5000

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
    """

    table: str
    rows: tuple[ChangedRow, ...] = ()
    whole_table: bool = False
    tenant: str | None = None
    prefixes: tuple[Mapping[str, str], ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        check_whole_reach(self.prefixes or None, self.reason, whole_table=self.whole_table)


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
        self._ring = f"{self._owner}.{quote_identifier(RING_TABLE)}"
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
        async with self._pool.acquire() as conn, database_ddl_lock(ConnectionSession(conn), policy):
            for statement in change_notes_ddl(self._schema, fence=self._fence):
                await conn.execute(statement)

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
        rows = await self._pool.fetch(
            "SELECT current_ring, switch_no, switched_at, switched_seq, floor, fence_seq, emptied, log_incarnation, "  # noqa: S608 - trusted identifiers
            f"fence_seconds, wait_seconds, now() AS now FROM {self._ring} WHERE id = 1"
        )
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
            f"UPDATE {self._ring} SET {assignments} WHERE id = 1 AND switch_no = $1",  # noqa: S608 - fixed column names
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
                result.restored = await self._restore(ring)
            _KEEPER_LAST_PASS.set(time.time())
            return result
        if state.restoring is not None and state.restoring == ring.log_incarnation and state.continues is None:
            # a restore of the ring's lost log was declared and never labelled whole (a failed append, a lost
            # lease, a crash): it is finished, or done again, before anything else
            result.restored = await self._restore(ring, incarnation=state.incarnation)
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
        if ring.fence_seq is not None and (
            (switch_at and _taken_bytes(state, ring.switched_seq) >= switch_at) or notes_full
        ):
            if ring.emptied:
                switched = await self._switch(ring, state)
                if switched is not None:
                    ring = switched
                    result.switched_to = ring.current
            else:
                result.switch_waited = True
                self._switch_waits(ring)
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
        _EMPTIED.inc()
        log.info(
            "a change-note table was emptied",
            extra={"extra_data": {"table": note_table(index), "deleted": deleted}},
        )
        return True, deleted

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
        status = await self._pool.execute(
            f"DELETE FROM {table} WHERE yb_hash_code(id) >= $1 AND yb_hash_code(id) < $2",  # noqa: S608 - trusted identifiers
            start,
            end,
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

    async def _delete_past(self, table: str, index: int) -> tuple[int, bool]:
        """delete the next ``empty_batch`` notes past the last id deleted, in id order; how many, and whether done.

        :param table: the qualified table
        :ptype table: str
        :param index: its place in the ring
        :ptype index: int
        :return: notes deleted, and whether the batch reached the table's end
        :rtype: tuple[int, bool]
        """
        rows = await self._pool.fetch(
            f"DELETE FROM {table} WHERE id IN (SELECT id FROM {table} "  # noqa: S608 - trusted identifiers
            "WHERE $2::uuid IS NULL OR id > $2 ORDER BY id LIMIT $1) RETURNING id",
            self._empty_batch,
            self._reached(index),
        )
        if rows:
            self._emptying_at = (index, max(row["id"] for row in rows))
        return len(rows), len(rows) < self._empty_batch

    def _switch_waits(self, ring: NoteRingState) -> None:
        """say once per switch that it is due but waits for the idle table's emptying.

        :param ring: the ring
        :ptype ring: NoteRingState
        :return: nothing
        :rtype: None
        """
        if self._waited_switch == ring.switch_no:
            return
        self._waited_switch = ring.switch_no
        _SWITCH_WAITS.inc()
        log.warning(
            "a change-note ring switch is due but waits: the table it would switch into is still being emptied",
            extra={"extra_data": {"switch": ring.switch_no, "waiting_on": note_table(ring.oldest)}},
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
            f"UPDATE {self._ring} SET current_ring = $2, switch_no = switch_no + 1, switched_at = now(), "  # noqa: S608 - trusted identifiers
            "switched_seq = $3, floor = fence_seq, fence_seq = NULL, emptied = FALSE, previous_switched_at = switched_at, "
            "fence_seconds = $4, wait_seconds = greatest($4, fence_seconds) "
            "WHERE id = 1 AND switch_no = $1 AND emptied",
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
        await self._pool.execute(
            f"DELETE FROM {self._marks} WHERE at < $1",  # noqa: S608 - trusted identifiers
            ring.switched_at - self._wait(ring),
        )
        log.info(
            "the change-note ring switched",
            extra={"extra_data": {"current": after.current, "switch": after.switch_no, "floor": after.floor}},
        )
        await self.count_ring_reads(ring.current)
        return after

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
        rows = await self._pool.fetch(
            f"SELECT count(*) AS notes FROM (SELECT 1 FROM {self._ring_table(ring.current)} LIMIT $1) n",  # noqa: S608 - trusted identifiers
            threshold,
        )
        self._notes_full = int(rows[0]["notes"]) >= threshold
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

    async def _restore(self, ring: NoteRingState, *, incarnation: str | None = None) -> str | None:
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
        covered = _seq_under(ring.floor, lost)
        if covered is None:
            covered = _seq_under(ring.fence_seq, lost)
        began = time.monotonic()
        state = await self._log.log_state()
        budget = None if state is None or not state.max_bytes else state.max_bytes * RESTORE_BUDGET_RATIO - state.bytes
        total = await self._count_since(ring, None)
        chosen = await self._fitting(ring, lost, covered, budget)
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
        kept = total if since is None else await self._count_since(ring, since)
        # streamed a page at a time, so the keeper's memory follows one change, not the coverage; should the log
        # have grown past the estimate meanwhile, finish_restore stops unlabelled and the next pass sizes again
        changes = self._changes_since(ring, since)
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

    def _coverage(self, ring: NoteRingState) -> str:
        """the SQL of the coverage's notes: the current and the previous table, one relation.

        :param ring: the ring
        :ptype ring: NoteRingState
        :return: a parenthesized ``UNION ALL``
        :rtype: str
        """
        columns = "table_name, key_text, keys, columns, tenant, noted_at"
        return (
            f"(SELECT {columns} FROM {self._ring_table(ring.current)} "  # noqa: S608 - trusted identifiers
            f"UNION ALL SELECT {columns} FROM {self._ring_table(ring.previous)})"
        )

    async def _count_since(self, ring: NoteRingState, since: datetime | None) -> int:
        """how many notes of the coverage were written at or after ``since`` (all of them for ``None``).

        :param ring: the ring
        :ptype ring: NoteRingState
        :param since: the cut
        :ptype since: datetime | None
        :return: the count
        :rtype: int
        """
        rows = await self._pool.fetch(
            f"SELECT count(*) AS notes FROM {self._coverage(ring)} n "  # noqa: S608 - trusted identifiers
            "WHERE $1::timestamptz IS NULL OR n.noted_at >= $1",
            since,
        )
        return int(rows[0]["notes"])

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
        rows = await self._pool.fetch(
            "SELECT count(*) AS rows, count(DISTINCT table_name) AS tables, coalesce(sum(length(table_name) + "  # noqa: S608 - trusted identifiers
            "coalesce(length(key_text), 0) + coalesce(length(columns::text), 0)), 0) AS bytes "
            f"FROM (SELECT DISTINCT table_name, key_text, columns FROM {self._coverage(ring)} n "
            "WHERE $1::timestamptz IS NULL OR n.noted_at >= $1) d",
            since,
        )
        row = rows[0]
        return (
            int(row["bytes"])
            + int(row["rows"]) * _RESTORED_ROW_ESTIMATE
            + int(row["tables"]) * _RESTORED_TABLE_ESTIMATE
        )

    async def _fitting(
        self, ring: NoteRingState, lost: str, covered: int | None, budget: float | None
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
        :return: ``(floor, cut)``, the cut ``None`` for the whole coverage; ``None`` when not even the newest
            part fits
        :rtype: tuple[int | None, datetime | None] | None
        """
        if budget is None or await self._estimate(ring, None) <= budget:
            return covered, None
        lag = self._wait(ring) / 2
        marks = [
            (int(mark["seq"]), mark["at"] - lag)
            for mark in await self._pool.fetch(
                f"SELECT seq, at FROM {self._marks} WHERE log_incarnation = $1 AND seq > $2 ORDER BY seq",  # noqa: S608 - trusted identifiers
                lost,
                -1 if covered is None else covered,
            )
        ]
        low, high = 0, len(marks)
        while low < high:  # the first (oldest-reaching) mark whose part fits; sizes fall along the list
            middle = (low + high) // 2
            if await self._estimate(ring, marks[middle][1]) <= budget:
                high = middle
            else:
                low = middle + 1
        return None if low == len(marks) else marks[low]

    def _changes_since(self, ring: NoteRingState, since: datetime | None) -> _StreamedChanges:
        """the coverage's notes since ``since``, folded per table and streamed a page at a time.

        :param ring: the ring
        :ptype ring: NoteRingState
        :param since: the cut; ``None`` the whole coverage
        :ptype since: datetime | None
        :return: the changes, as an async iterable
        :rtype: _StreamedChanges
        """
        return _StreamedChanges(self._pool, self._coverage(ring), since)

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


class _StreamedChanges:
    """the coverage's notes folded into changes as a cursor reads them: whole tables first, then each table's
    distinct rows in chunks of :data:`_RESTORE_CHUNK_ROWS`, so memory holds one chunk.

    :ivar tables: the tables streamed so far
    """

    def __init__(self, pool: _Pool, coverage: str, since: datetime | None) -> None:
        self._pool = pool
        self._coverage = coverage
        self._since = since
        self.tables = 0

    def __aiter__(self) -> AsyncIterator[NotedChange]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[NotedChange]:
        async with self._pool.acquire() as conn, conn.transaction():
            reaches: dict[tuple[str, str | None], list[Any]] = {}
            for row in await conn.fetch(
                # a whole note's columns are its write's reach (NOTE_REACH_SETTING): a load's prefixes, a reset
                f"SELECT DISTINCT table_name, tenant, columns::text AS reach FROM {self._coverage} n "  # noqa: S608 - trusted identifiers
                "WHERE n.key_text IS NULL AND ($1::timestamptz IS NULL OR n.noted_at >= $1)",
                self._since,
            ):
                pair = (str(row["table_name"]), None if row["tenant"] is None else str(row["tenant"]))
                reaches.setdefault(pair, []).append(row["reach"])
            for noted, tenant in sorted(reaches, key=lambda pair: (pair[0], pair[1] or "")):
                self.tables += 1
                # a whole-table change carries no row: under its schema for a schema table, else on _all
                yield _whole_change(noted, tenant, reaches[(noted, tenant)])
            # a row its tenant's whole-table change (or the table's _all one) already covers is not named again; a
            # load's names only its prefixes, so a row beside them is named (one inside them twice, which is harmless)
            cursor = conn.cursor(
                f"SELECT DISTINCT table_name, tenant, key_text, keys, columns FROM {self._coverage} n "  # noqa: S608 - trusted identifiers
                "WHERE n.key_text IS NOT NULL AND ($1::timestamptz IS NULL OR n.noted_at >= $1) "
                f"AND NOT EXISTS (SELECT 1 FROM {self._coverage} w WHERE w.key_text IS NULL "
                "AND w.table_name = n.table_name AND (w.tenant IS NULL OR w.tenant = n.tenant) "
                "AND (w.columns IS NULL OR NOT w.columns ? 'prefixes') "
                "AND ($1::timestamptz IS NULL OR w.noted_at >= $1)) ORDER BY table_name, tenant",
                self._since,
                prefetch=_RESTORE_PAGE,
            )
            group: tuple[str, str | None] | None = None
            rows: dict[tuple[tuple[str, ...], tuple[tuple[str, str | None], ...]], None] = {}
            async for note in cursor:
                name = str(note["table_name"])
                tenant = None if note["tenant"] is None else str(note["tenant"])
                if group is not None and (group != (name, tenant) or len(rows) >= _RESTORE_CHUNK_ROWS):
                    yield _change(group[0], rows, group[1])
                    rows = {}
                if group is None or group[0] != name:
                    self.tables += 1
                group = (name, tenant)
                rows[(tuple(str(key) for key in note["keys"]), _columns_of(note["columns"]))] = None
            if group is not None and rows:
                yield _change(group[0], rows, group[1])


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


def _change(
    table: str, rows: dict[tuple[tuple[str, ...], tuple[tuple[str, str | None], ...]], None], tenant: str | None
) -> NotedChange:
    """one table's folded rows of one tenant as a change.

    :param table: the table
    :ptype table: str
    :param rows: each distinct row: its keys and its invalidation columns
    :ptype rows: dict[tuple[tuple[str, ...], tuple[tuple[str, str | None], ...]], None]
    :param tenant: the rows' tenant, ``None`` for ``_all``
    :ptype tenant: str | None
    :return: the change
    :rtype: NotedChange
    """
    return NotedChange(
        table=table,
        tenant=tenant,
        rows=tuple(
            ChangedRow(ids=list(ids), columns=dict(columns) if columns else None, tenant=tenant)
            for ids, columns in rows
        ),
    )


@dataclass(frozen=True, slots=True)
class NotedRows:
    """what the change notes say a table's writes since a time changed.

    :ivar rows: each distinct row the notes name that no whole-table note covers, with its invalidation columns and
        its tenant
    :ivar whole_table: a note of the whole table (a truncate, a file load) is among them
    :ivar whole_tenants: each such note's tenant, ``None`` for ``_all``: it covers that tenant's rows (``_all``'s,
        every tenant's), never another schema's
    """

    rows: tuple[ChangedRow, ...] = ()
    whole_table: bool = False
    whole_tenants: tuple[str | None, ...] = ()


def _noted_rows_sql(owner: str) -> str:
    """the one statement :func:`noted_rows_since` asks: the ring row, and the table's notes in its coverage.

    One statement, not a read of the ring row then one of the notes, so the reconciler's ask a marked table
    is one round trip, and the ring and its notes are read in one snapshot. Each note table's branch is read
    only when it is the current or the previous table (a one-time filter on the ring row); the previous one only
    while the reach is before the last switch's fence has passed (a copy of the slot read before the switch writes
    there for at most the fence after it, so past that it holds no note from the reach on, and is not read), from the reach
    (``since`` less half the ring's wait), and only when the coverage reaches back that far. Each branch is
    answered from the ``(table_name, noted_at)`` index, which carries the columns read
    (:func:`~threetears.core.ddl.change_notes.change_notes_ddl`). A row comes back for the ring with no note
    (``noted`` NULL) when the coverage holds none, and none at all when the ring has no row.

    :param owner: the quoted schema holding the ring
    :ptype owner: str
    :return: the statement: ``$1`` the table, ``$2`` since, ``$3`` the most notes to answer
    :rtype: str
    """
    branches = " UNION ALL ".join(
        f"SELECT keys, columns, key_text, tenant FROM {owner}.{quote_identifier(note_table(index))} "  # noqa: S608 - trusted identifiers
        f"WHERE (r.current_ring = {index} OR ((r.current_ring - 1 + {NOTE_RING_SIZE}) % {NOTE_RING_SIZE} = {index} "
        "AND r.reach < r.writes_previous_until)) "
        "AND table_name = $1 AND noted_at >= r.reach"
        for index in range(NOTE_RING_SIZE)
    )
    return (  # one statement; each interpolated piece is a fixed name this module builds
        "SELECT r.switch_no, r.previous_switched_at, r.reach, n.noted, n.key_text, n.keys, n.columns, n.tenant "
        f"FROM (SELECT current_ring, switch_no, previous_switched_at, "
        "$2::timestamptz - make_interval(secs => wait_seconds / 2) AS reach, "
        "switched_at + make_interval(secs => GREATEST(fence_seconds, wait_seconds)) AS writes_previous_until "
        f"FROM {owner}.{quote_identifier(RING_TABLE)} WHERE id = 1) r "
        "LEFT JOIN LATERAL (SELECT TRUE AS noted, key_text, keys, columns, tenant "
        f"FROM (SELECT DISTINCT key_text, keys, columns, tenant FROM ({branches}) u) d LIMIT $3) n "
        "ON r.switch_no = 0 OR r.reach >= r.previous_switched_at"
    )


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
    """
    owner = quote_identifier(notes_schema)
    try:
        found = await pool.fetch(_noted_rows_sql(owner), table, since, cap + 1)
    except asyncpg.UndefinedTableError, asyncpg.InvalidSchemaNameError:
        # NOSILENT: no ring installed here: the caller names the whole table instead, said and counted there
        return "notes_unavailable"
    if not found:
        return "notes_unavailable"
    ring = found[0]
    # before a first switch nothing was ever emptied: every note since the triggers were given is there
    if int(ring["switch_no"]) != 0 and ring["reach"] < ring["previous_switched_at"]:
        return "not_covered"
    # each note's values by position (_noted_rows_sql's select list: noted, key_text, keys, columns, tenant from 3):
    # at 500 notes the rows' handling, not the statement, was the larger part of the call
    notes = [note for note in found if note[3]]
    if not notes:
        return "no_notes"
    if len(notes) > cap:
        return "too_many_rows"
    # the ring holds every schema's notes of the table: a whole-table note covers its own tenant's rows (or every
    # tenant's, when it is _all's), never another schema's, whose rows are still named
    whole = dict.fromkeys(None if note[7] is None else str(note[7]) for note in notes if note[4] is None)
    rows: list[ChangedRow] = []
    if None not in whole:
        # the statement answers each (key, columns, tenant) once (its DISTINCT), so each note is one row
        for note in notes:
            if note[4] is None:
                continue
            token = None if note[7] is None else str(note[7])
            if token in whole:
                continue
            columns = note[6]
            rows.append(
                ChangedRow(
                    ids=[str(key) for key in note[5]],
                    columns=dict(_columns_of(columns)) or None if columns is not None else None,
                    tenant=token,
                )
            )
    return NotedRows(rows=tuple(rows), whole_table=bool(whole), whole_tenants=tuple(whole))


def _noted_keys_sql(owner: str, slot: int) -> str:
    """one ring note table's notes of a table and tenant in a bracket, after a time, in ``noted_at`` order.

    Answered from the ``(table_name, noted_at)`` index, which carries ``keys``, ``key_text``, ``columns`` and
    ``tenant`` (:func:`~threetears.core.ddl.change_notes.change_notes_ddl`): an index range scan of the table's notes
    in the bracket, in the index's order, stopped at the limit; no sort, no other table's notes.

    :param owner: the quoted schema holding the ring
    :ptype owner: str
    :param slot: the note table's place in the ring
    :ptype slot: int
    :return: the statement: ``$1`` table, ``$2`` tenant (NULL for ``_all``), ``$3`` after (the last time read, or
        one microsecond before the bracket: both ends of the range are index conditions), ``$4`` until, ``$5`` the
        most notes (NULL for all)
    :rtype: str
    """
    return (  # each interpolated piece is a fixed name this module builds
        f"SELECT noted_at, keys, columns, key_text IS NULL AS whole FROM {owner}.{quote_identifier(note_table(slot))} "  # noqa: S608
        "WHERE table_name = $1 AND noted_at > $3 AND noted_at <= $4 AND tenant IS NOT DISTINCT FROM $2 "
        "ORDER BY noted_at LIMIT $5"
    )


#: the database clock's grain: ``timestamptz`` holds microseconds
_CLOCK_GRAIN: Final = timedelta(microseconds=1)

#: a page's reads go through the ``(table_name, noted_at)`` index in its order, stopped at the limit. The planner
#: judges a bracket's notes from the column statistics as if a table's notes were spread over the night, so it counts
#: too few of a large commit's (a burst of one table's notes in a second or two) to stop early at the limit, and reads
#: the whole rest of the bracket by bitmap and sorts it: every page reads every later page's notes again. Told to keep
#: the index's order (``LOCAL``: the page's read-only transaction only, gone at its end, never left on a pooled
#: connection), each page reads only its own notes. Measured on a ring of 322,000 notes and a 20,000-row commit,
#: tables freshly written and analyzed: 212,062 rows scanned for 20,162 named (10.52x), told so 20,262 (1.005x), each
#: page an Index Only Scan with both ``noted_at`` bounds in its Index Cond, on Postgres and YugabyteDB.
_INDEX_ORDER: Final = "SET LOCAL enable_bitmapscan = off; SET LOCAL enable_seqscan = off; SET LOCAL enable_sort = off"


def _key_bytes(keys: Sequence[str], columns: Sequence[tuple[str, str | None]] = ()) -> int:
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
        from; the page is read in one read-only transaction on it
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
    async with held as one, one.transaction(readonly=True):
        # the planner prices the bracket's notes too few to read in index order: it is told to (_INDEX_ORDER)
        await one.execute(_INDEX_ORDER)
        return await _noted_keys_page(one, owner, table, tenant_token, since, until, cursor, limit, max_bytes)


async def _noted_keys_page(
    conn: Any,
    owner: str,
    table: str,
    tenant: str | None,
    since: datetime,
    until: datetime,
    cursor: NotedKeysCursor | None,
    limit: int,
    max_bytes: int,
) -> NotedKeysPage | str:
    try:
        rings = await conn.fetch(
            f"SELECT current_ring, switch_no, previous_switched_at, wait_seconds FROM {owner}.{quote_identifier(RING_TABLE)} "  # noqa: S608
            "WHERE id = 1"
        )
    except asyncpg.UndefinedTableError, asyncpg.InvalidSchemaNameError:
        # NOSILENT: no ring installed here: the asker is behind on the table, said and counted there
        return "notes_unavailable"
    if not rings:
        return "notes_unavailable"
    ring = rings[0]
    current = int(ring["current_ring"])
    previous = (current - 1 + NOTE_RING_SIZE) % NOTE_RING_SIZE
    # a writer's cached slot may name the table it had up to half the wait after a switch (noted_rows_since's reach)
    reach = since - timedelta(seconds=float(ring["wait_seconds"]) / 2)
    if int(ring["switch_no"]) != 0 and reach < ring["previous_switched_at"]:
        return "not_covered"
    order = (previous, current)
    at = cursor or NotedKeysCursor(slot=previous)
    if at.slot not in order:
        return "not_covered"
    keys: list[tuple[str, ...]] = []
    columns: list[dict[str, str | None] | None] = []
    whole = False
    used = 0
    left = limit

    def page(next_at: NotedKeysCursor | None) -> NotedKeysPage:
        return NotedKeysPage(keys=tuple(keys), columns=tuple(columns), whole_table=whole, next=next_at)

    for slot in order[order.index(at.slot) :]:
        # where this table's read starts: the cursor's time in its own table, else the bracket's start
        start = at.after if slot == at.slot else None
        # the time read up to, or a microsecond (the clock's grain) before the bracket: one range for the index
        after = start if start is not None else since - _CLOCK_GRAIN
        found = await conn.fetch(_noted_keys_sql(owner, slot), table, tenant, after, until, left + 1)
        if not found:
            continue
        full = len(found) > left
        taken = list(found[:left])
        if full and all(note["noted_at"] == taken[-1]["noted_at"] for note in taken):
            if left < limit:
                # the page already holds the earlier table's notes: this table's first instant starts the next
                return page(NotedKeysCursor(slot=slot, after=start))
            # one instant's notes fill the page: read that instant whole, so continuing past it skips none
            instant = taken[-1]["noted_at"]
            taken = list(
                await conn.fetch(_noted_keys_sql(owner, slot), table, tenant, instant - _CLOCK_GRAIN, instant, None)
            )
        elif full:
            # end at a whole instant: the last instant's notes go to the next page
            taken = [note for note in taken if note["noted_at"] < taken[-1]["noted_at"]]
        for index, note in enumerate(taken):
            starts_instant = index == 0 or taken[index - 1]["noted_at"] < note["noted_at"]
            if note["whole"]:
                whole = True
                continue
            values = tuple(str(value) for value in note["keys"])
            held = _columns_of(note["columns"])
            used += _key_bytes(values, held)
            if used > max_bytes and starts_instant and keys:
                # past the page's bytes at an instant's first note: the page ends before it
                return page(NotedKeysCursor(slot=slot, after=taken[index - 1]["noted_at"] if index else start))
            keys.append(values)
            columns.append(dict(held) if held else None)
        if full:
            return page(NotedKeysCursor(slot=slot, after=taken[-1]["noted_at"]))
        left -= len(taken)
        if left < 1:
            # the page is full at this table's last note; the next page reads on from it
            return page(NotedKeysCursor(slot=slot, after=taken[-1]["noted_at"]))
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
