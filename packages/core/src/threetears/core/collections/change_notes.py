"""the change notes' readers: what the notes say changed, read by a follower's catch-up and by the keeper's restore.

The notes themselves, the ring that holds them and the keeper that switches, empties, scrubs and restores from them are
:mod:`threetears.core.collections.change_note_keeper` (its docstring is the design). This module holds what reads the
notes outside the keeper: :func:`noted_rows_since` (a table's noted rows since a time, a follower's bounded catch-up),
:func:`noted_keys_page` (a keys-unknown record's notes, a page at a time) and :class:`NotedChange`, the folded change
both these reads and a restore produce.

**Why apart from the keeper** (Pace 2026-11-15, pyarrow stays an optional extra of core): a restore folds its notes as
Arrow in a worker process (:mod:`threetears.core.change_note_fold`), so the keeper imports pyarrow and NumPy at module
top. Every process that follows the epoch log imports these readers (``threetears.epoch``, the write marks, the
migrations); with the keeper beside them each needed pyarrow (26dabec04 made it a core dependency, after the SDK's
unit suite failed 111 tests on ``ModuleNotFoundError: pyarrow``). Only the process that runs the keeper (the hub)
needs the ``arrow`` extra now; ``tests/enforcement/test_core_imports_without_arrow.py`` holds the line.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections.abc import (
    Awaitable,
    Callable,
    Mapping,
    Sequence,
)
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import accumulate
from operator import itemgetter, methodcaller
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import asyncpg
from threetears.observe import contain, get_logger

from threetears.core.collections.epoch_change import (
    RESET_REASON,
    ChangedRow,
    check_whole_reach,
)
from threetears.core.collections.noted_keys import NotedKeysCursor, NotedKeysPage, NotedKeysReader
from threetears.core.ddl.change_notes import (
    NOTE_KEY_SEPARATOR,
    NOTE_RING_SIZE,
    RING_TABLE,
    note_table,
)
from threetears.core.keyed import (
    Derived,
    Step,
    Fragment,
    Selected,
    distinct,
    read_row,
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
    :param cap: the most distinct rows to name; more is answered :attr:`NotedRows.too_many_rows`, naming no row
        and every whole-table change
    :ptype cap: int
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
        # past the cap the statement stopped at an arbitrary cap + 1 notes, so the whole-table notes among the rest
        # were never read. Each is a write of its own (a truncate, a load) whose advance may have been lost with the
        # rest, and the caller's bracket of keys does not carry one: read them alone, from the whole-table notes'
        # own partial index, so a truncate in the window still reaches every follower whole (RC2, 2026-11-16: before,
        # this answered "too_many_rows" and the truncate's removed rows stayed in followers).
        whole = await _noted_whole_since(pool, notes_schema, table, reach, ring)
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
        if note[6] is None:
            reaches.setdefault(None if note[9] is None else str(note[9]), []).append(note[8])
    whole = tuple(whole_noted_change(table, tenant, reached) for tenant, reached in reaches.items())
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


async def _noted_whole_since(pool: Any, schema: str, table: str, reach: datetime, ring: Any) -> tuple[NotedChange, ...]:
    """``table``'s whole-table notes from ``reach`` on, each tenant's folded into one change: :func:`noted_rows_since`'s
    answer past its cap.

    The slots are the ones its statement read (:data:`_SLOT_READ`, decided here from the ring row it read): the current
    table, and the previous one while the reach is before the last switch's writes there could end. Each is read
    through the whole-table notes' partial index (``(table_name, noted_at) WHERE key_text IS NULL``,
    :func:`~threetears.core.ddl.change_notes.change_notes_ddl`), which holds those notes alone, so the many keyed notes
    that took the read past its cap are never visited.

    WHY the time is compared here, not in the statement (RC2, 2026-11-16, ``test_change_note_ring_plans``): with
    ``noted_at >= $2`` beside the table, YugabyteDB planned the read through the ``(table_name, noted_at)`` index of
    every note, filtering ``key_text IS NULL``, and scanned all 101 notes of a table holding one whole note; Postgres
    took the partial index either way. Pinned by the table alone, the read is the scrub's whole-table look, which both
    plan through the partial index. A slot holds the few whole notes of its one switch period, so the ones before the
    reach cost a row each.

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
    :return: each tenant's change
    :rtype: tuple[NotedChange, ...]
    :raises GenerationUnavailableError: when a whole-table note names a reach this module never writes
    """
    current = int(ring["current_ring"])
    slots = [current]
    if reach < ring["writes_previous_until"]:
        slots.append((current - 1 + NOTE_RING_SIZE) % NOTE_RING_SIZE)
    legs = [
        scan(
            f"{schema}.{note_table(slot)}",
            ("tenant", "noted_at", Selected("t.columns::text", "reach")),
            where=[Fragment("t.key_text IS NULL AND t.table_name = $1", (table,))],
        )
        for slot in slots
    ]
    # the current slot alone once the previous one can hold no note from the reach on: a union names two or more
    found = await read_step(pool, legs[0] if len(legs) == 1 else union(*legs, keep_all=True))
    reaches: dict[str | None, list[Any]] = {}
    for note in found:
        if note["noted_at"] >= reach:
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


class PoolNotedKeysReader:
    """a :class:`NotedKeysReader` over a direct L3 pool: the notes read in this process, as the broker reads them.

    For a registry on a direct pool (a platform process: the hub's own registries, the gateway's), whose transport is
    no broker to ask (:meth:`~threetears.core.backends.nats_proxy.NatsProxyL3Backend.noted_keys`). Without one, such a
    registry stays behind on a table at any keys-unknown record carrying a notes bracket (a writer's commit too large
    to record, a reconciled lost write of more rows than one record names) until it restarts, the write's rows stale
    in it (RC1, 2026-11-16). With one it pages the bracket's keys from the ring (:func:`noted_keys_page`, an index range
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
