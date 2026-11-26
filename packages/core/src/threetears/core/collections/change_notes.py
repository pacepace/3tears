"""the change notes' readers: what the notes say changed, read by a follower's catch-up and by the keeper's restore.

The notes themselves, the ring that holds them and the keeper that switches, empties, scrubs and restores from them are
:mod:`threetears.core.collections.change_note_keeper` (its docstring is the design). This module holds what reads the
notes outside the keeper: :func:`noted_rows_since` (a table's noted rows since a time, a follower's bounded catch-up),
:func:`noted_keys_page` (a keys-unknown record's notes, a page at a time) and :class:`NotedChange`, the folded change
both these reads and a restore produce.

**Why apart from the keeper** (Pace 2026-11-15, pyarrow stays an optional extra of core): a restore folds its notes as
Arrow in a worker process (:mod:`threetears.core.change_note_fold`), so the keeper imports pyarrow and NumPy at module
top. Every process that follows the epoch log imports these readers (``threetears.epoch``, the write marks, the
migrations); with the keeper beside them each needed pyarrow (pyarrow became a core dependency, after the SDK's
unit suite failed 111 tests on ``ModuleNotFoundError: pyarrow``). Only the process that runs the keeper (the hub)
needs the ``arrow`` extra now; ``tests/enforcement/test_core_imports_without_arrow.py`` holds the line.
"""

from __future__ import annotations

import json
from collections.abc import (
    Sequence,
)
from dataclasses import dataclass
from operator import itemgetter, methodcaller
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import asyncpg
from threetears.observe import contain, get_logger

from threetears.core.collections.epoch_change import (
    RESET_REASON,
    ChangedRow,
)
from threetears.core.collections.epoch_reads import epoch_keys_page
from threetears.core.collections.noted_change import NotedChange
from threetears.core.collections.noted_keys import NotedKeysCursor, NotedKeysPage, NotedKeysReader
from threetears.core.ddl.change_notes import (
    NOTE_KEY_SEPARATOR,
    NOTE_RING_SIZE,
    RING_TABLE,
    note_table,
)
from threetears.core.keyed import (
    Derived,
    KeyBatch,
    Step,
    Fragment,
    Selected,
    distinct,
    read_keyed,
    read_row_beside,
    read_step,
    scan,
    union,
)
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.sql_fragments import quote_identifier

__all__ = [
    "RESTORE_BUDGET_RATIO",
    "NotedChange",
    "NotedKeysCursor",
    "NotedKeysPage",
    "NotedKeysReader",
    "NotedRows",
    "PoolNotedKeysReader",
    "noted_keys_page",
    "noted_rows_since",
    "whole_noted_change",
]

log = get_logger(__name__)

#: the most of a change log's ``max_bytes`` a restore may fill, what the log already holds included. The ring
#: switches at half of it (:mod:`threetears.core.collections.change_note_keeper`'s ``_SWITCHES_PER_BUDGET``), so on a
#: day the switch is not held up the current and the previous table -- the coverage a restore reads -- fit it whole
RESTORE_BUDGET_RATIO: Final = 0.75


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


def whole_noted_change(table: str, tenant: str | None, reaches: Sequence[Any]) -> NotedChange:
    """one table's whole-table notes of one tenant folded into one change, keeping the reach they name: the fold
    :func:`noted_rows_since` and the keeper's restore (:mod:`threetears.core.collections.change_note_keeper`) share.

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
    :ivar too_many_rows: the notes name more distinct rows than the read's cap: ``rows`` is empty and names none of
        them (a caller names them by the notes' bracket, a page at a time), while ``whole`` still holds every
        whole-table note's change, each a write of its own whose advance may have been lost too
    """

    rows: tuple[ChangedRow, ...] = ()
    whole_table: bool = False
    whole_tenants: tuple[str | None, ...] = ()
    whole: tuple[NotedChange, ...] = ()
    too_many_rows: bool = False


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


def _noted_rows_step(
    schema: str, table: str, since: datetime, *, whole_from: datetime, until: datetime | None = None
) -> Step:
    """the notes :func:`noted_rows_since` reads beside the ring row (``r``): each note table's notes of ``table`` from
    the reach on (to ``until``, when given), the distinct (key, columns, tenant) of them; a whole-table note only from
    ``whole_from`` on.

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
    :param whole_from: the earliest whole-table note read (:data:`_WHOLE_NOTE_SKEW` before ``since``)
    :ptype whole_from: datetime
    :param until: the latest note read; ``None`` for every note since
    :ptype until: datetime | None
    :return: the step
    :rtype: Step
    """
    coverage = Fragment(
        "r.switch_no = 0 OR $1::timestamptz - make_interval(secs => r.wait_seconds / 2) >= r.previous_switched_at",
        (since,),
    )
    # a whole-table note (no key_text) is the window's own only from whole_from on, filtered in the statement: the
    # (table_name, noted_at) index carries key_text, so nothing is read beyond the range. Before (2026-11-22, the
    # write marks' last change) the filter compared the answered row's ``noted`` column, which is TRUE, never a time:
    # every whole note on the named-rows path was dropped
    # the window's end is a plain bound, a far-future time when the ask names none (every ask but a lost advance's
    # recovery): COALESCE($until, 'infinity') in its place cost the reconciler's ask on YugabyteDB 2.2-2.5x its
    # calibrated read (2026-11-22, test_noted_rows_scale_live, 1M notes: 2.9 ms against 1.3 ms)
    from_reach = Fragment(
        "t.table_name = $1 AND t.noted_at >= $2::timestamptz - make_interval(secs => r.wait_seconds / 2) "
        "AND t.noted_at <= $3::timestamptz AND (t.key_text IS NOT NULL OR t.noted_at >= $4::timestamptz)",
        (table, since, _NO_END if until is None else until, whole_from),
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


#: how far before a window's start a whole-table note is still the window's own: the window starts at its writes'
#: start, already widened by the clocks' skew, and only the nodes' skew separates a note from it (YugabyteDB bounds it
#: at max_clock_skew_usec, 500 ms by default). A whole-table note older than this belongs to an earlier write, whose
#: advance was heard: re-recorded, it told every follower to drop the table and read it whole again (2026-11-22
#: replay: one operator reset re-announced three times by the write-mark reconciler, each a whole read on every
#: follower, one of which left five states out of the copy). Rows keep the ring's half-wait reach; a whole note never
#: does
#: the window's end when an ask names none: later than any note's time, so the bound holds every note
_NO_END: Final = datetime(9999, 12, 31, tzinfo=UTC)

_WHOLE_NOTE_SKEW: Final = timedelta(seconds=5)


async def noted_rows_since(
    pool: Any, *, notes_schema: str, table: str, since: datetime, cap: int, until: datetime | None = None
) -> NotedRows | str:
    """the rows the change notes say ``table``'s writes changed since ``since`` (to ``until``), or why they cannot say.

    For a writer-side process that knows a table may have been written in a window and not which rows (a lost
    advance's recovery, :mod:`threetears.core.collections.lost_writes`): every committed write to a noted table left a
    note with its keys in its own transaction. The notes are read from the ring's current and previous tables, from
    ``since`` less half the ring's wait (a transaction writes its notes up to that long before its commit), and only
    when the coverage reaches back that far. A whole-table note is read only from :data:`_WHOLE_NOTE_SKEW` before
    ``since``: it is re-recorded whole, so it must be the window's own.

    :param pool: the L3 pool
    :ptype pool: Any
    :param notes_schema: the schema holding the ring
    :ptype notes_schema: str
    :param table: the table
    :ptype table: str
    :param since: the oldest write's time, by the database's clock
    :ptype since: datetime
    :param cap: the most distinct rows to name; more is answered :attr:`NotedRows.too_many_rows`, naming no row
        and every whole-table change
    :ptype cap: int
    :param until: the window's end, by the database's clock; ``None`` for every note since
    :ptype until: datetime | None
    :return: the rows; or ``"notes_unavailable"`` (no ring), ``"not_covered"`` (the coverage starts after
        ``since``), ``"no_notes"`` (no note of the table since)
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
            beside=_noted_rows_step(notes_schema, table, since, whole_from=since - _WHOLE_NOTE_SKEW, until=until),
            limit=cap + 1,
            present="noted",
            # the ring row's columns on one answered row, not repeated on each of hundreds of notes (2026-11-16,
            # YugabyteDB, 500 notes: 2,500 values fewer to send and decode, the statement's median 2.08 -> 1.64 ms)
            first="ring_row",
        )
    except asyncpg.UndefinedTableError, asyncpg.InvalidSchemaNameError:
        # NOSILENT: no ring installed here: the caller names the whole table instead, said and counted there
        return "notes_unavailable"
    if not found:
        return "notes_unavailable"
    # the one row carrying the ring's columns (ring_row not FALSE: TRUE, or NULL on the ring row alone)
    ring = next(row for row in found if row[6] is not False)
    # the reach as the statement's legs compute it (since less half the wait, wait_seconds a DOUBLE PRECISION), as
    # _noted_keys_read computes it from the same ring row
    reach = since - timedelta(seconds=float(ring["wait_seconds"]) / 2)
    # the whole-table notes of the window's writes themselves: none stamped before the window (less the skew)
    whole_from = since - _WHOLE_NOTE_SKEW
    # before a first switch nothing was ever emptied: every note since the triggers were given is there
    if int(ring["switch_no"]) != 0 and reach < ring["previous_switched_at"]:
        return "not_covered"
    # each note's values by position (the select list: the ring's five _ROWS_RING_COLUMNS, then noted, ring_row,
    # key_text, keys, columns, tenant): at 500 notes the rows' handling, not the statement, was the larger part of the
    # call
    notes = [note for note in found if note[5]]
    if not notes:
        return "no_notes"
    if len(notes) > cap:
        # past the cap the statement stopped at an arbitrary cap + 1 notes, so the whole-table notes among the rest
        # were never read. Each is a write of its own (a truncate, a load) whose advance may have been lost with the
        # rest, and the caller's bracket of keys does not carry one: read them alone, from the whole-table notes'
        # own partial index, so a truncate in the window still reaches every follower whole (2026-11-16: before,
        # this answered "too_many_rows" and the truncate's removed rows stayed in followers).
        whole = await _noted_whole_since(pool, notes_schema, table, whole_from, ring, until=until)
        return NotedRows(
            whole_table=bool(whole),
            whole_tenants=tuple(change.tenant for change in whole),
            whole=whole,
            too_many_rows=True,
        )
    # the ring holds every schema's notes of the table: each tenant's whole-table notes fold into one change with the
    # reach they name (a whole-table note keeps it as its columns), as a restore folds them
    reaches: dict[str | None, list[Any]] = {}
    for note in notes:
        if note[7] is None:
            reaches.setdefault(None if note[10] is None else str(note[10]), []).append(note[9])
    whole = tuple(whole_noted_change(table, tenant, reached) for tenant, reached in reaches.items())
    # a plain or reset change covers its own tenant's rows (or every tenant's, when it is _all's), never another
    # schema's; a load's names only its prefixes, so a row beside them is still named
    covering = {change.tenant for change in whole if not change.prefixes}
    rows: list[ChangedRow] = []
    if None not in covering:
        # the statement answers each (key, columns, tenant) once (its DISTINCT), so each note is one row
        for note in notes:
            if note[7] is None:
                continue
            token = None if note[10] is None else str(note[10])
            if token in covering:
                continue
            columns = note[9]
            rows.append(
                ChangedRow(
                    ids=[str(key) for key in note[8]],
                    columns=dict(_columns_of(columns)) or None if columns is not None else None,
                    tenant=token,
                )
            )
    return NotedRows(rows=tuple(rows), whole_table=bool(whole), whole_tenants=tuple(reaches), whole=whole)


async def _noted_whole_since(
    pool: Any, schema: str, table: str, reach: datetime, ring: Any, *, until: datetime | None = None
) -> tuple[NotedChange, ...]:
    """``table``'s whole-table notes from ``reach`` on, each tenant's folded into one change: :func:`noted_rows_since`'s
    answer past its cap.

    The slots are the ones its statement read (:data:`_SLOT_READ`, decided here from the ring row it read): the current
    table, and the previous one while the reach is before the last switch's writes there could end. Each is looked at
    through the whole-table notes' partial index (``(table_name, noted_at) INCLUDE (tenant) WHERE key_text IS NULL``,
    :func:`~threetears.core.ddl.change_notes.change_notes_ddl`), which holds those notes alone, so the many keyed notes
    that took the read past its cap are never visited; the reach of each one from ``reach`` on is then read by its key.

    WHY two reads, and the time compared here (2026-11-16, ``test_change_note_ring_plans`` on YugabyteDB): read
    in one statement with the reach (``columns``, which the partial index does not carry), or with ``noted_at >= $2``
    beside the table, YugabyteDB answered it from the ``(table_name, noted_at)`` index that carries every column of
    every note, filtering ``key_text IS NULL``: all 101 notes of a table holding one whole note. The look reads only what
    the partial index holds (its key, its tenant, and the row's key every YugabyteDB index entry carries), as the scrub's
    whole-table look does, which both engines plan through it. A slot holds the few whole notes of one switch period,
    so those before the reach cost a row each, and a slot with none in reach costs no second read.

    :param pool: the L3 pool
    :ptype pool: Any
    :param schema: the schema holding the ring
    :ptype schema: str
    :param table: the table
    :ptype table: str
    :param reach: the earliest note time read (``since`` less half the ring's wait)
    :ptype reach: datetime
    :param ring: the ring row :func:`noted_rows_since` read (its ``_ROWS_RING_COLUMNS``)
    :ptype ring: Any
    :param until: the latest note time read; ``None`` for every note from ``reach`` on
    :ptype until: datetime | None
    :return: each tenant's change
    :rtype: tuple[NotedChange, ...]
    :raises GenerationUnavailableError: when a whole-table note names a reach this module never writes
    """
    current = int(ring["current_ring"])
    slots = [current]
    if reach < ring["writes_previous_until"]:
        slots.append((current - 1 + NOTE_RING_SIZE) % NOTE_RING_SIZE)
    reaches: dict[str | None, list[Any]] = {}
    for slot in slots:
        notes = f"{schema}.{note_table(slot)}"
        looked = await read_step(
            pool,
            scan(notes, ("id", "noted_at"), where=[Fragment("t.key_text IS NULL AND t.table_name = $1", (table,))]),
        )
        held = [
            str(note["id"])
            for note in looked
            if note["noted_at"] >= reach and (until is None or note["noted_at"] <= until)
        ]
        if not held:
            continue
        for note in await read_keyed(
            pool,
            notes,
            keys=KeyBatch(("id",), [held], casts=("uuid",)),
            columns=("tenant", Selected("t.columns::text", "reach")),
        ):
            reaches.setdefault(None if note["tenant"] is None else str(note["tenant"]), []).append(note["reach"])
    return tuple(whole_noted_change(table, tenant, reached) for tenant, reached in reaches.items())


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
        f"WHERE table_name = $1 AND noted_at > $3 AND noted_at <= $4 AND ({tenant}) AND ({_RESTORED}) "
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
_COUNT_SEPARATORS: Final = methodcaller("count", NOTE_KEY_SEPARATOR)

#: a page's note's keys (NULL: split its key text), columns, bytes and key text, by their place in
#: :func:`_noted_keys_sql`'s select list
_KEYS_AT: Final = itemgetter(1)
_COLUMNS_AT: Final = itemgetter(2)
_BYTES_AT: Final = itemgetter(3)
_KEY_TEXT_AT: Final = itemgetter(4)

#: a note's key text split back into its keys
_SPLIT_KEYS: Final = methodcaller("split", NOTE_KEY_SEPARATOR)


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
    """a page of the keys ``table`` changed in a commit's bracket, of ``tenant_token``'s rows, or why they cannot be
    read: the table's own versions (or stamped rows) in the bracket's epoch range
    (:func:`~threetears.core.collections.epoch_reads.epoch_keys_page`, the restore's read narrowed to the range).

    For the broker's answer to a copy follower holding a keys-unknown record whose commit named too many rows
    (:attr:`~threetears.core.collections.epoch_change.EpochChange.noted_between`), and for every
    :class:`~threetears.core.collections.noted_keys.NotedKeysReader` in this process. The read and the wire are the
    notes' until 0.95 (the request, the page, a cursor to go on from); only the cursor's place is the range's now.

    :param conn: a connection to the L3 database (the broker's, on the platform schema), or a pool
    :ptype conn: Any
    :param notes_schema: the platform schema: an undeclared table there is ``_all``'s, one in a pod's schema the pod's
    :ptype notes_schema: str
    :param table: the table's name
    :ptype table: str
    :param tenant_token: the tenant whose rows are read (the asker's schema's), ``None`` for ``_all``: never another's
    :ptype tenant_token: str | None
    :param noted_between: the commit's bracket, ``(from, until)``
    :ptype noted_between: tuple[datetime, datetime]
    :param cursor: where the last page ended; ``None`` for the first
    :ptype cursor: NotedKeysCursor | None
    :param limit: the most rows a page reads
    :ptype limit: int
    :param max_bytes: the most bytes of keys a page holds
    :ptype max_bytes: int
    :param every_tenant: read every tenant's rows of the table, ``tenant_token`` unused: for an operator's check of a
        table whose rows carry their tenant in their key (a backup's live check), never a tenant's own reader
    :ptype every_tenant: bool
    :return: the page; or ``"notes_unavailable"`` (the table carries no epoch) or ``"not_covered"`` (the cursor names
        a relation the database no longer holds)
    :rtype: NotedKeysPage | str
    """
    return await epoch_keys_page(
        conn,
        table=table,
        platform_schema=notes_schema,
        tenant_token=tenant_token,
        noted_between=noted_between,
        cursor=cursor,
        limit=limit,
        max_bytes=max_bytes,
        every_tenant=every_tenant,
    )


class PoolNotedKeysReader:
    """a :class:`NotedKeysReader` over a direct L3 pool: the notes read in this process, as the broker reads them.

    For a registry on a direct pool (a platform process: the hub's own registries, the gateway's), whose transport is
    no broker to ask (:meth:`~threetears.core.backends.nats_proxy.NatsProxyL3Backend.noted_keys`). Without one, such a
    registry stays behind on a table at any keys-unknown record carrying a notes bracket (a writer's commit too large
    to record, a reconciled lost write of more rows than one record names) until it restarts, the write's rows stale
    in it (2026-11-16). With one it pages the bracket's keys from the ring (:func:`noted_keys_page`, an index range
    read a page at a time) and evicts exactly them: nothing dropped, nothing read whole.

    Whose notes: a tenant's (``tenant_token``, ``None`` for ``_all``), or every tenant's (``every_tenant``). A platform
    process caches every tenant's rows of the tables it follows, so it reads every tenant's notes: a key of another
    tenant the bracket names is evicted too and read again on demand, which is never wrong.

    A database role that may not read the ring is answered ``"notes_unavailable"``, said at WARNING: the registry is
    then behind on the table, as it was with no reader, and its catch-up pass never fails on it.

    :param pool: the L3 pool (anything :func:`noted_keys_page` takes: a pool, or one connection)
    :ptype pool: Any
    :param notes_schema: the schema holding the ring (the platform schema)
    :ptype notes_schema: str
    :param tenant_token: whose notes are read, ``None`` for ``_all``; unused with ``every_tenant``
    :ptype tenant_token: str | None
    :param every_tenant: read every tenant's notes of the table
    :ptype every_tenant: bool
    """

    def __init__(
        self, pool: Any, *, notes_schema: str, tenant_token: str | None = None, every_tenant: bool = False
    ) -> None:
        self._pool = pool
        self._notes_schema = notes_schema
        self._tenant_token = tenant_token
        self._every_tenant = every_tenant

    async def noted_keys(
        self,
        table: str,
        *,
        noted_between: tuple[datetime, datetime],
        cursor: NotedKeysCursor | None,
        limit: int,
        max_bytes: int,
    ) -> NotedKeysPage | str:
        """a page of the keys ``table``'s notes name in ``noted_between`` (:meth:`NotedKeysReader.noted_keys`).

        :param table: the table, as its notes name it
        :ptype table: str
        :param noted_between: the commit's bracket, on the database's clock
        :ptype noted_between: tuple[datetime, datetime]
        :param cursor: where the last page ended; ``None`` for the first
        :ptype cursor: NotedKeysCursor | None
        :param limit: the most notes a page reads
        :ptype limit: int
        :param max_bytes: the most bytes of keys a page holds
        :ptype max_bytes: int
        :return: the page; or ``"notes_unavailable"`` (no ring, or no grant to read it) or ``"not_covered"``
        :rtype: NotedKeysPage | str
        """
        try:
            return await noted_keys_page(
                self._pool,
                notes_schema=self._notes_schema,
                table=table,
                tenant_token=self._tenant_token,
                noted_between=noted_between,
                cursor=cursor,
                limit=limit,
                max_bytes=max_bytes,
                every_tenant=self._every_tenant,
            )
        except asyncpg.InsufficientPrivilegeError as exc:
            # NOSILENT: this role may not read the ring: the registry is behind on the table, said here and there
            contain(exc, log=log, what="a direct pool's role may not read the change notes", extra={"table": table})
            return "notes_unavailable"
