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
:class:`ChangeNoteKeeper`, under a lease) switches the ring when the current table has taken one change
log's worth -- the bytes the log is configured to hold (``max_bytes``), measured from the log itself as the
bytes per sequence it holds times the sequences taken since the switch, never by counting in L3 -- and
empties the oldest with ``TRUNCATE``, never with deletes. The ring row in L3 names the current table; only
the keeper writes it, once per switch, and a writer reads it once per statement.

**Why emptying is safe.** A statement reads the ring once, at its first noted row, and writes every note of
that statement into the table it read. So the only notes that can still be heading for a table the ring has
switched away from are those of statements that began before the switch. The keeper never switches again
until its **fence** has passed since the last switch -- a grace period well beyond any statement's run time
(ten minutes by default) -- and it empties the oldest table only right after a switch. The table it empties
was switched away from two switches back, at least one fence ago: no statement that read it as current is
still running.

**What the coverage vouches for.** When the fence after a switch has passed, the keeper reads the log's
position (``fence_seq``): every epoch appended after it is of a statement that began after the switch, so
its note is in the current table or a later one. At the next switch that position becomes the coverage's
**floor**: every epoch after the floor has its note in the current or the previous table, the two a
restore reads. An epoch whose writer stalled between its commit and its append for longer than the fence
breaks that promise, and the scrub reports it.

**Putting a lost log back.** When the keeper finds the log absent it reads the notes of the current and the
previous tables, folds them into one record per table (each distinct key once; a whole-table note as a
whole-table record), and declares a new log under a NEW incarnation that names the lost one and the
coverage's floor (``continues``, ``covered_from``). A follower whose mark under the lost incarnation is at or
past the floor missed nothing the notes do not cover: it carries its mark to the start of the new log and
walks every record in it, evicting each key once more than it strictly needed. One whose mark is older is
behind (``record_lost``), serving what it holds, until it restarts. Nothing drops a table it did not
change; a whole-table record is the truncate or load that made it, re-applied.

**The scrub.** Each pass checks that the epochs the log holds after the floor have their notes: one whose
note is not where the coverage needs it is logged at ERROR and counted
(``threetears_change_notes_missing_total``), for an alert. It only checks and reports; it never writes a
note, reloads anything or repairs.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from threetears.nats.change_log import ChangeLogState
from threetears.observe import contain, get_logger
from threetears.observe.metrics import counter, gauge

from threetears.core.collections.epoch_change import ChangedRow, EpochChange
from threetears.core.collections.generation import NotedTable, split_generation_token
from threetears.core.ddl.change_notes import (
    NOTE_RING_SIZE,
    NOTE_TRIGGER,
    NOTE_TRUNCATE_TRIGGER,
    RING_TABLE,
    change_notes_ddl,
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
    "ChangeNoteKeeper",
    "KeeperPass",
    "NoteRingState",
    "NotedChange",
    "ScrubPass",
    "run_change_note_keeper",
]

log = get_logger(__name__)

#: how long after a switch the keeper waits before it may switch again, and so before the table two
#: switches back is emptied: well beyond any statement's run time, since a statement writes every note
#: into the table it read as current when it began
_DEFAULT_NOTE_FENCE: Final = timedelta(minutes=10)

#: the pause between keeper passes: a write waits for a lost log's restore up to one pass, well inside
#: a source's ``restore_wait``
_DEFAULT_KEEPER_EVERY: Final = timedelta(seconds=1)

#: how long an install or an emptying waits for the database's DDL lock before deferring to the next pass
_LOCK_POLICY: Final = DdlLockPolicy(poll_interval=0.5, max_wait=5.0)

#: epochs the scrub checks per pass
_SCRUB_BATCH: Final = 1000

#: the separator a note's key columns are joined with to compare them in SQL: never in a key's text form
_KEY_SEPARATOR: Final = "\x1f"

#: notes read per statement when a restore reads the coverage
_RESTORE_PAGE: Final = 5000

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
    description="note tables the keeper emptied (TRUNCATE) once nothing could still write to them",
)
_RESTORES = counter(
    "threetears_change_notes_restores_total",
    description="lost change logs the keeper put back from the change notes",
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
    """

    table: str
    rows: tuple[ChangedRow, ...] = ()
    whole_table: bool = False


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

    async def restore_from_notes(
        self, changes: Sequence[NotedChange], *, lost: str, covered_from: int | None
    ) -> str | None:
        """declare a new log continuing ``lost`` and append one record per change, where the broker holds none.

        :param changes: what the notes say changed, per table
        :ptype changes: Sequence[NotedChange]
        :param lost: the lost log's incarnation
        :ptype lost: str
        :param covered_from: the lost log's sequence from which the notes cover every epoch; ``None``
            when they vouch for none
        :ptype covered_from: int | None
        :return: the new log's incarnation; ``None`` when a log was already there
        :rtype: str | None
        :raises GenerationUnavailableError: when the log cannot be declared or written
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
    :ivar emptied: the note table this pass emptied
    :ivar switched_to: the note table this pass made current
    :ivar scrub: what the scrub found
    """

    started: str | None = None
    restored: str | None = None
    adopted: str | None = None
    fenced: bool = False
    emptied: int | None = None
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


def _changes_from_notes(notes: Iterable[Any]) -> list[NotedChange]:
    """fold the notes read into one change per table: each distinct row once, or the whole table.

    :param notes: rows of ``table_name``, ``keys``, ``columns``
    :ptype notes: Iterable[Any]
    :return: the changes, in table order
    :rtype: list[NotedChange]
    """
    whole: set[str] = set()
    rows: dict[str, dict[tuple[tuple[str, ...], tuple[tuple[str, str | None], ...]], None]] = {}
    for note in notes:
        table = str(note["table_name"])
        keys = note["keys"]
        if keys is None:
            whole.add(table)
            continue
        columns = note["columns"]
        carried = _columns_of(columns)
        rows.setdefault(table, {})[(tuple(str(key) for key in keys), carried)] = None
    changes: list[NotedChange] = []
    for table in sorted(whole | set(rows)):
        if table in whole:
            changes.append(NotedChange(table=table, whole_table=True))
            continue
        changes.append(
            NotedChange(
                table=table,
                rows=tuple(
                    ChangedRow(ids=list(ids), columns=dict(columns) if columns else None)
                    for ids, columns in rows[table]
                ),
            )
        )
    return changes


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
        fence: timedelta = _DEFAULT_NOTE_FENCE,
        scrub_batch: int = _SCRUB_BATCH,
        switch_bytes: int | None = None,
    ) -> None:
        """capture what the keeper works between; no I/O.

        :param pool: the L3 pool holding the ring, allowed DDL (installing and emptying take the DDL lock)
        :ptype pool: _Pool
        :param notes_schema: the schema holding the ring (the platform schema), a TRUSTED identifier
        :ptype notes_schema: str
        :param log: the change log the notes cover
        :ptype log: _ChangeLogKeeping
        :param fence: how long after a switch the ring may switch again: well beyond any statement's run time
        :ptype fence: timedelta
        :param scrub_batch: epochs the scrub checks per pass; at least one
        :ptype scrub_batch: int
        :param switch_bytes: how many bytes of change records the current table takes before the ring
            switches; ``None`` (the default) is one change log's worth, the log's own ``max_bytes``. A
            smaller ring keeps less coverage: never less than the log can hold between two keeper passes
        :ptype switch_bytes: int | None
        :return: nothing
        :rtype: None
        :raises ValueError: when the fence is not positive, the batch is under one or the switch size is
        """
        if fence <= timedelta(0) or scrub_batch < 1 or (switch_bytes is not None and switch_bytes < 1):
            raise ValueError(
                "a keeper's fence is positive, its scrub checks at least one epoch a pass and its switch size is positive"
            )
        self._pool = pool
        self._schema = notes_schema
        self._owner = quote_identifier(notes_schema)
        self._ring = f"{self._owner}.{quote_identifier(RING_TABLE)}"
        self._log = log
        self._fence = fence
        self._scrub_batch = scrub_batch
        self._switch_bytes = switch_bytes
        #: the newest epoch the scrub has checked, and the log's head at the previous pass (what it checks up to)
        self._scrubbed: tuple[str, int] | None = None
        self._seen: tuple[str, int] | None = None

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
            for statement in change_notes_ddl(self._schema):
                await conn.execute(statement)

    async def install_triggers(self, tables: Iterable[NotedTable], policy: DdlLockPolicy = _LOCK_POLICY) -> list[str]:
        """give every noted table, in every schema, its note function and triggers; say which it gave.

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
        """what each noted table, in every schema, still needs.

        :param reader: the pool, or one connection
        :ptype reader: Any
        :param noted: the noted tables by name
        :ptype noted: dict[str, NotedTable]
        :return: the work, in schema and table order
        :rtype: list[_Install]
        """
        present = [
            (str(row["table_schema"]), str(row["table_name"]))
            for row in await reader.fetch(_NOTED_TABLES_SQL, sorted(noted))
        ]
        triggers = {
            (str(row["table_schema"]), str(row["table_name"]), str(row["trigger_name"]))
            for row in await reader.fetch(_TRIGGERS_SQL, [NOTE_TRIGGER, NOTE_TRUNCATE_TRIGGER])
        }
        names = [note_function_name(schema, table) for schema, table in present]
        bodies = {str(row["name"]): str(row["body"]) for row in await reader.fetch(_FUNCTIONS_SQL, self._schema, names)}
        work: list[_Install] = []
        for schema, table in present:
            each = noted[table]
            wanted = trigger_function_body(notes_schema=self._schema, keys=each.keys, columns=each.columns)
            stale = bodies.get(note_function_name(schema, table)) != wanted
            missing = tuple(
                name for name in (NOTE_TRIGGER, NOTE_TRUNCATE_TRIGGER) if (schema, table, name) not in triggers
            )
            if stale or missing:
                work.append(_Install(table_schema=schema, noted=each, function=stale, triggers=missing))
        return work

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
                )
            )
        row, truncate = note_trigger_sql(notes_schema=self._schema, table_schema=install.table_schema, table=each.table)
        for name, statement in ((NOTE_TRIGGER, row), (NOTE_TRUNCATE_TRIGGER, truncate)):
            if name in install.triggers:
                await conn.execute(statement)

    async def ring(self) -> NoteRingState:
        """the ring row, and the database's clock.

        :return: the state
        :rtype: NoteRingState
        :raises GenerationUnavailableError: when the ring is not installed
        """
        rows = await self._pool.fetch(
            "SELECT current_ring, switch_no, switched_at, switched_seq, floor, fence_seq, emptied, log_incarnation, "  # noqa: S608 - trusted identifiers
            f"now() AS now FROM {self._ring} WHERE id = 1"
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
        )

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
        state = await self._log.log_state()
        if state is None:
            if ring.log_incarnation is None:
                result.started = await self._log.start_log()
                await self._update(ring, log_incarnation=result.started)
            else:
                result.restored = await self._restore(ring)
            return result
        if ring.log_incarnation != state.incarnation:
            ring = await self._adopt(ring, state)
            result.adopted = state.incarnation
        if ring.fence_seq is None and ring.now >= ring.switched_at + self._fence:
            # every statement that read the ring before the last switch has ended: an epoch appended from here
            # on has its note in the current table or a later one
            ring = await self._update(ring, fence_seq=_token(state.incarnation, state.last_seq))
            result.fenced = True
        if not ring.emptied:
            emptied = ring.oldest
            if await self._empty(emptied):
                ring = await self._update(ring, emptied=True)
                result.emptied = emptied
        switch_at = self._switch_bytes or state.max_bytes
        if (
            ring.fence_seq is not None
            and ring.emptied
            and switch_at
            and _taken_bytes(state, ring.switched_seq) >= switch_at
        ):
            ring = await self._switch(ring, state)
            result.switched_to = ring.current
        result.scrub = await self._scrub(ring, state)
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

    async def _empty(self, index: int) -> bool:
        """``TRUNCATE`` the note table at ``index``, under the DDL lock; never row by row.

        Called only right after a switch, for the table after the new current one: the ring switched away
        from it two switches back, and never switches again within a fence of the last, so every statement
        that read it as current has ended (the module docstring's argument).

        :param index: the table
        :ptype index: int
        :return: whether it was emptied; ``False`` when the DDL lock stayed busy (the next pass retries)
        :rtype: bool
        """
        async with self._pool.acquire() as conn:
            try:
                async with database_ddl_lock(ConnectionSession(conn), _LOCK_POLICY):
                    await conn.execute(f"TRUNCATE TABLE {self._ring_table(index)}")
            except DdlLockTimeoutError:
                log.info("emptying a change-note table deferred: the database's DDL lock is busy")
                return False
        _EMPTIED.inc()
        return True

    async def _switch(self, ring: NoteRingState, state: ChangeLogState) -> NoteRingState:
        """make the emptied table current; the fence's position becomes the coverage's floor.

        :param ring: the ring
        :ptype ring: NoteRingState
        :param state: the log
        :ptype state: ChangeLogState
        :return: the ring after
        :rtype: NoteRingState
        """
        await self._pool.execute(
            f"UPDATE {self._ring} SET current_ring = $2, switch_no = switch_no + 1, switched_at = now(), "  # noqa: S608 - trusted identifiers
            "switched_seq = $3, floor = fence_seq, fence_seq = NULL, emptied = FALSE WHERE id = 1 AND switch_no = $1",
            ring.switch_no,
            ring.oldest,
            _token(state.incarnation, state.last_seq),
        )
        after = await self.ring()
        _SWITCHES.inc()
        log.info(
            "the change-note ring switched",
            extra={"extra_data": {"current": after.current, "switch": after.switch_no, "floor": after.floor}},
        )
        return after

    async def _read_notes(self, indexes: Sequence[int]) -> list[Any]:
        """every note of the ring's tables at ``indexes``.

        :param indexes: the tables
        :ptype indexes: Sequence[int]
        :return: the notes: ``table_name``, ``keys``, ``columns``
        :rtype: list[Any]
        """
        notes: list[Any] = []
        async with self._pool.acquire() as conn, conn.transaction():
            for index in indexes:
                cursor = conn.cursor(
                    f"SELECT table_name, keys, columns FROM {self._ring_table(index)}",  # noqa: S608 - trusted identifiers
                    prefetch=_RESTORE_PAGE,
                )
                async for note in cursor:
                    notes.append(note)
        return notes

    async def _restore(self, ring: NoteRingState) -> str | None:
        """put the lost log back from the notes of the current and the previous tables.

        :param ring: the ring
        :ptype ring: NoteRingState
        :return: the new log's incarnation; ``None`` when a log was there by the time it was declared
        :rtype: str | None
        """
        lost = ring.log_incarnation
        assert lost is not None  # narrow: the caller restores only a log the ring has seen
        covered = _seq_under(ring.floor, lost)
        if covered is None:
            covered = _seq_under(ring.fence_seq, lost)
        began = time.monotonic()
        notes = await self._read_notes((ring.current, ring.previous))
        changes = _changes_from_notes(notes)
        restored = await self._log.restore_from_notes(changes, lost=lost, covered_from=covered)
        if restored is None:
            return None
        state = await self._log.log_state()
        head = state.last_seq if state is not None and state.incarnation == restored else 0
        await self._update(
            ring,
            log_incarnation=restored,
            floor=_token(restored, 0),
            fence_seq=None if ring.fence_seq is None else _token(restored, head),
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
                    "restored": restored,
                    "covered_from": covered,
                    "notes": len(notes),
                    "tables": len(changes),
                    "seconds": round(time.monotonic() - began, 3),
                }
            },
        )
        return restored

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
        records = [record for seq in range(start + 1, end + 1) if (record := await self._log.epoch_at(seq)) is not None]
        fence = _seq_under(ring.fence_seq, incarnation)
        found = await self._found(ring, records)
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

    async def _found(self, ring: NoteRingState, records: Sequence[EpochChange]) -> dict[int, _Found]:
        """the notes the coverage tables hold for ``records``, by table.

        :param ring: the ring
        :ptype ring: NoteRingState
        :param records: the epochs checked
        :ptype records: Sequence[EpochChange]
        :return: per note table, what it holds of them
        :rtype: dict[int, _Found]
        """
        tables = sorted({record.table for record in records})
        keys = sorted({_KEY_SEPARATOR.join(row.ids) for record in records for row in record.rows})
        found: dict[int, _Found] = {}
        for index in (ring.current, ring.previous):
            rows = await self._pool.fetch(
                f"SELECT DISTINCT table_name, array_to_string(keys, $3) AS joined FROM {self._ring_table(index)} "  # noqa: S608 - trusted identifiers
                "WHERE table_name = ANY($1::text[]) AND (keys IS NULL OR array_to_string(keys, $3) = ANY($2::text[]))",
                tables,
                keys,
                _KEY_SEPARATOR,
            )
            any_rows = await self._pool.fetch(
                f"SELECT DISTINCT table_name FROM {self._ring_table(index)} WHERE table_name = ANY($1::text[])",  # noqa: S608 - trusted identifiers
                tables,
            )
            found[index] = _Found(
                keyed={(str(row["table_name"]), str(row["joined"])) for row in rows if row["joined"] is not None},
                whole={str(row["table_name"]) for row in rows if row["joined"] is None},
                any_note={str(row["table_name"]) for row in any_rows},
            )
        return found


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
