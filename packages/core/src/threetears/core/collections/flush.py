"""Write-buffer and flush strategy for deferred collection persistence."""

from __future__ import annotations

import asyncio
import json
import logging
from array import array
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final, NamedTuple, TYPE_CHECKING

import asyncpg
from sqlalchemy import Column, Integer, MetaData, String, Table, Text

from threetears.core.backends.schema_sql import json_default
from threetears.core.declared import whole_read
from threetears.core import packed_keys
from threetears.core.collections.generation import WriteGeneration
from threetears.core.collections.l2_order import l2_order_of
from threetears.core.backends.protocol import RowcountUnreadableError
from threetears.core.exceptions import CorruptCacheEntry, GenerationUnavailableError
from threetears.observe import contain, get_logger
from threetears.core.eviction import Declared, Policy
from threetears.core.cache.l1_lru import L1_EVICTION_INFO

__all__ = [
    "FlushStrategy",
    "PendingWrite",
    "WriteBuffer",
    "flush_pending",
]

if TYPE_CHECKING:
    from threetears.core.cache.sqlite import SQLiteBackend
    from threetears.core.collections.registry import CollectionRegistry

log = get_logger(__name__)

_WRITE_BUFFER_METADATA = MetaData()

_write_buffer_table = Table(
    "write_buffer",
    _WRITE_BUFFER_METADATA,
    Column("key", String, primary_key=True),
    Column("table_name", Text, nullable=False),
    Column("entity_id", Text, nullable=False),
    Column("data", Text, nullable=False),
    Column("retries", Integer, nullable=False, default=0),
    Column("date_updated", String, nullable=True),
    info={L1_EVICTION_INFO: Declared(Policy.KEEP, "writes waiting for L3: dropping one loses the write")},
)

# Retry budget for general flush failures (transient DB errors,
# serialization, etc.). Once a write fails this many times it is
# dropped from the buffer and a permanent-failure event logged.
_MAX_FLUSH_RETRIES = 10

# Retry budget for foreign-key-violation failures specifically. FK
# violations almost always mean "my parent hasn't reached Postgres
# yet" -- the parent is either later in the toposort within this
# drain (already addressed) or pending in a separate drain batch.
# In the latter case the child just needs to wait for the parent
# to land before retrying. The pre-2026-05-13 behavior of capping
# FK retries at 10 (~5 minutes at the 30s default flush interval)
# was too tight: a single dropped parent message permanently
# orphaned every descendant in the conversation, producing the
# cascading "messages dropped, half conversation missing"
# fingerprint in production (conv ``019e2372-fcdd``,
# 2026-05-13 incident). 100 retries at the same interval = ~50min,
# enough headroom for any realistic transient. Beyond that the
# parent genuinely failed and the child is unreachable -- drop
# with a clear "orphan chain" log so operators can investigate.
_FK_RETRY_LIMIT = 100


def _is_fk_violation(exc: BaseException) -> bool:
    """Detect whether an exception is a Postgres foreign-key violation.

    Two signals are checked: ``isinstance`` against the asyncpg
    typed exception, AND a substring match in the exception message
    (covers cases where the violation was raised through a wrapper
    or re-raised as a different class). Either match counts.
    """
    if isinstance(exc, asyncpg.exceptions.ForeignKeyViolationError):
        return True
    return "violates foreign key constraint" in str(exc)


class FlushStrategy(StrEnum):
    ALWAYS = "ALWAYS"
    ON_CHECKPOINT = "ON_CHECKPOINT"
    ON_SCHEDULE = "ON_SCHEDULE"
    ON_SHUTDOWN = "ON_SHUTDOWN"


class PendingWrite(NamedTuple):
    table_name: str
    entity_id: Any
    data: dict[str, Any]
    retries: int = 0


#: an entry's state, one byte an entry of a run's :attr:`_Run.states`
_PENDING: Final = ord("p")  # a write waiting for a drain
_CLAIMED: Final = ord("c")  # a write a drain claimed: in flight until acked or re-enqueued
_RECOVERED: Final = ord("r")  # a durable row a drain claimed with no write of this process behind it (a restart's)
_GONE: Final = ord("g")  # the key's write left the buffer (acked or removed); covers older entries until compacted


def _buffer_text(key: tuple[str, str]) -> str:
    """``key``'s text in a run's packed keys (:func:`packed_keys.key_text`).

    :param key: the normalised ``(table_name, entity-id text)``
    :ptype key: tuple[str, str]
    :return: the text
    :rtype: str
    """
    return packed_keys.key_text(key)


@dataclass(frozen=True, slots=True)
class _Entry:
    """the newest entry a :class:`WriteBuffer` holds for one key, as one lookup found it (function-local).

    :ivar state: :data:`_PENDING`, :data:`_CLAIMED`, :data:`_RECOVERED` or :data:`_GONE`
    :ivar write: the write, or ``None`` for a gone or recovered key
    :ivar first: the key's place in drain order: where it entered the buffer, kept while it stays in it
    :ivar run: the run holding it, by place in the buffer's runs
    :ivar row: its row in that run
    """

    state: int
    write: PendingWrite | None
    first: int
    run: int = -1
    row: int = -1


@dataclass(frozen=True, slots=True)
class _Run:
    """an immutable run of a :class:`WriteBuffer`'s entries, sorted by key text, one a key.

    Columnar: the keys are one packed string (:mod:`~threetears.core.packed_keys`) with each text's offset
    in one ``array('q')``, the states one ``bytes``, the drain places one ``array('q')``; only the writes -- the rows
    callers handed in -- are a tuple of their objects. A key is found by bisection, a text sliced a step.

    :ivar keys: the entries' key texts, packed (``NUL text NUL ... NUL``), in sorted order
    :ivar offsets: where each text's leading NUL is in ``keys``, and last where the closing NUL is
    :ivar states: each entry's state byte
    :ivar firsts: each entry's drain place
    :ivar writes: each entry's write, ``None`` for a gone or recovered key
    """

    keys: str
    offsets: array[int]
    states: bytes
    firsts: array[int]
    writes: tuple[PendingWrite | None, ...]

    def text(self, row: int) -> str:
        """entry ``row``'s key text.

        :param row: the entry
        :ptype row: int
        :return: the text
        :rtype: str
        """
        return packed_keys.text_at(self.keys, self.offsets, row)

    def find(self, text: str) -> int:
        """the entry for the key whose text is ``text``, by bisection.

        :param text: the key's :func:`_buffer_text`
        :ptype text: str
        :return: the entry's row, or ``-1`` when the run holds none for the key
        :rtype: int
        """
        return packed_keys.find_sorted(self.keys, self.offsets, text)


def _run_of(texts: Sequence[str], states: bytes, firsts: array[int], writes: tuple[PendingWrite | None, ...]) -> _Run:
    """a run of ``texts`` (sorted, distinct) and their entries' columns.

    :param texts: the key texts, sorted, one an entry
    :ptype texts: Sequence[str]
    :param states: each entry's state byte
    :ptype states: bytes
    :param firsts: each entry's drain place
    :ptype firsts: array[int]
    :param writes: each entry's write
    :ptype writes: tuple[PendingWrite | None, ...]
    :return: the run
    :rtype: _Run
    """
    return _Run(
        keys=packed_keys.packed(texts),
        offsets=packed_keys.offsets_of(texts),
        states=states,
        firsts=firsts,
        writes=writes,
    )


#: the buffer's runs when it holds nothing
_NO_RUNS: Final[tuple[_Run, ...]] = ()


def _merged(older: _Run, newer: _Run, *, oldest: bool) -> _Run:
    """``older`` and ``newer`` as one run: a merge of two sorted runs, ``newer``'s entry kept where both hold a key.

    :param older: the older run
    :ptype older: _Run
    :param newer: the newer run
    :ptype newer: _Run
    :param oldest: whether the result is the buffer's oldest run, so nothing older is left for a gone key to cover and
        gone keys are dropped
    :ptype oldest: bool
    :return: the run
    :rtype: _Run
    """
    # each run's texts split once (C), so the merge compares strings, never slices them a step
    old_texts, new_texts = packed_keys.texts(older.keys), packed_keys.texts(newer.keys)
    texts: list[str] = []
    states = bytearray()
    firsts = array("q")
    writes: list[PendingWrite | None] = []
    old_count, new_count = len(old_texts), len(new_texts)
    i = j = 0
    while i < old_count and j < new_count:
        old_text, new_text = old_texts[i], new_texts[j]
        if old_text < new_text:
            run, row, text = older, i, old_text
            i += 1
        else:
            if old_text == new_text:
                i += 1
            run, row, text = newer, j, new_text
            j += 1
        state = run.states[row]
        if oldest and state == _GONE:
            continue
        texts.append(text)
        states.append(state)
        firsts.append(run.firsts[row])
        writes.append(run.writes[row])
    for run, rest, start in ((older, old_texts, i), (newer, new_texts, j)):
        for row in range(start, len(rest)):
            state = run.states[row]
            if oldest and state == _GONE:
                continue
            texts.append(rest[row])
            states.append(state)
            firsts.append(run.firsts[row])
            writes.append(run.writes[row])
    return _run_of(texts, bytes(states), firsts, tuple(writes))


class WriteBuffer:
    """Coalescing async write buffer keyed by (table_name, entity_id).

    when l1_backend is provided, pending writes are persisted to
    SQLite so they survive process crashes; drains then read them from
    there.

    the buffer follows a claim/ack lifecycle so the write-through
    guarantee holds across a crash: :meth:`drain` *claims* pending
    writes (marks them in-flight) but does NOT delete their durable
    rows; the row is reclaimed only once :meth:`ack` confirms the L3
    write landed, or re-armed for retry by :meth:`re_enqueue`. a crash
    between claim and ack therefore leaves the durable row intact, so
    the next process replays it instead of losing the write from both
    tiers. the in-flight claim doubles as a version guard: any newer
    write that coalesces in via :meth:`add` during the flush window
    clears the claim, so a stale :meth:`ack` / :meth:`re_enqueue` can
    never clobber that newer value (lost-update protection).

    **No state a key** (Pace, 2026-11-23). What the buffer holds is a tuple of immutable :class:`_Run` s, each sorted
    by key, its keys one packed string (:mod:`~threetears.core.packed_keys`) and its states one ``bytes``,
    never a dict or set of keys. An add or a retry appends a run of one entry (an ack or removal marks its entry gone),
    and the last two runs are merged (a sorted merge) while the older holds no more than twice the newer's entries, so
    the runs stay logarithmic in the keys held and an entry is copied a logarithmic number of times over its life,
    never every key at each change. A key's newest entry is the one in the newest run holding it, found by bisection
    of each run. A drain merges every run into one, newest first. A claim is the entry's state byte: a newer write's
    entry is pending, which is what releases the claim. Every change replaces the tuple whole under the buffer's lock,
    with no await between its read and its replacement.

    Measured (2026-11-23, M-series, one coroutine; the dict it replaced in brackets): at 10,000 keys an add costs
    9 us (1.2), an ack 3.5 us (0.5) and a drain 3.5 ms (1.1); at 100,000 keys an add 12 us (0.7), an ack 8 us (0.5), a
    drain 36 ms (14). An ack or removal marks its entry gone in a copy of its run's state bytes; nothing is appended.
    """

    #: an older run is merged into the newer one behind it while it holds at most this many times as many entries
    _MERGE_RATIO: Final = 2

    def __init__(self, l1_backend: SQLiteBackend | None = None) -> None:
        """initialize write buffer with optional L1 persistence.

        :param l1_backend: optional SQLiteBackend for crash-safe buffering
        :ptype l1_backend: SQLiteBackend | None
        """
        self._runs: tuple[_Run, ...] = _NO_RUNS
        # pending and claimed writes held (pending_count), and the next drain place a key entering the buffer takes
        self._held = 0
        self._next_first = 0
        self._lock = asyncio.Lock()
        self._l1 = l1_backend
        if self._l1 is not None and not self._l1.is_initialized():
            self._l1.initialize(_WRITE_BUFFER_METADATA)

    @staticmethod
    def _key(table_name: str, entity_id: Any) -> tuple[str, str]:
        """normalize a (table, entity) pair to a stable string-keyed tuple.

        the durable SQLite row stores the entity id in its string form while
        in-memory callers pass the original typed id (e.g. ``UUID``). normalizing
        both to text keeps the buffer's runs, their claims, and the durable row
        addressed by ONE key, so the version guard lines up across the L1 and
        non-L1 paths.

        :param table_name: destination table name
        :ptype table_name: str
        :param entity_id: entity primary-key value in any form
        :ptype entity_id: Any
        :return: normalized ``(table_name, <entity-id-as-text>)`` key
        :rtype: tuple[str, str]
        """
        entity_key = str(entity_id)  # convert at border: keyspace aligns with the persisted write_buffer String PK
        return (table_name, entity_key)

    def _newest(self, text: str) -> _Entry | None:
        """the newest entry for the key whose text is ``text``; caller MUST hold ``self._lock``.

        :param text: the key's :func:`_buffer_text`
        :ptype text: str
        :return: the entry, or ``None`` when no run holds one
        :rtype: _Entry | None
        """
        runs = self._runs
        for at in range(len(runs) - 1, -1, -1):
            run = runs[at]
            row = run.find(text)
            if row >= 0:
                return _Entry(state=run.states[row], write=run.writes[row], first=run.firsts[row], run=at, row=row)
        return None

    def _append(self, text: str, state: int, write: PendingWrite | None, first: int) -> None:
        """append one entry as a run of its own, merging the last runs while the older is not much the larger.

        :param text: the key's :func:`_buffer_text`
        :ptype text: str
        :param state: the entry's state byte
        :ptype state: int
        :param write: the entry's write
        :ptype write: PendingWrite | None
        :param first: the entry's drain place
        :ptype first: int
        :return: nothing
        :rtype: None
        """
        runs = (*self._runs, _run_of((text,), bytes((state,)), array("q", (first,)), (write,)))
        while len(runs) >= 2 and len(runs[-2].writes) <= self._MERGE_RATIO * len(runs[-1].writes):
            runs = (*runs[:-2], _merged(runs[-2], runs[-1], oldest=len(runs) == 2))
        self._runs = runs

    def _add_locked(self, table_name: str, entity_id: Any, data: dict[str, Any], retries: int) -> None:
        """insert-or-replace a pending write; caller MUST hold ``self._lock``.

        :param table_name: destination table name
        :ptype table_name: str
        :param entity_id: entity primary-key value
        :ptype entity_id: Any
        :param data: row payload keyed by column name
        :ptype data: dict[str, Any]
        :param retries: failed-flush attempts recorded so far
        :ptype retries: int
        :return: nothing
        :rtype: None
        """
        text = _buffer_text(self._key(table_name, entity_id))
        newest = self._newest(text)
        held = newest is not None and newest.write is not None and newest.state in (_PENDING, _CLAIMED)
        if held and newest is not None and newest.write is not None and _orders_after(newest.write.data, data):
            # two compare-and-swap winners for one row can reach this buffer in either order (a
            # coroutine suspended between its win and this add). The newer order is the one L3
            # must end on, so an older one arriving last is dropped rather than coalesced over it.
            log.debug(
                "buffered write superseded by a newer compare-and-swap order already pending",
                extra={"extra_data": {"table": table_name, "entity_id": str(entity_id)}},
            )
            return
        if held and newest is not None:
            first = newest.first
        else:
            first = self._next_first
            self._next_first += 1
            self._held += 1
        # a fresh (re)write supersedes any in-flight claim for this key: its entry is pending, so the
        # flush that claimed the old value must NOT evict or re-enqueue over this newer one when it completes.
        self._append(text, _PENDING, PendingWrite(table_name, entity_id, data, retries), first)
        if self._l1 is not None:
            l1_key = f"{table_name}:{entity_id}"
            self._l1.upsert(
                "write_buffer",
                {
                    "key": l1_key,
                    "table_name": table_name,
                    "entity_id": str(entity_id),
                    "data": json.dumps(data, default=json_default),
                    "retries": retries,
                    "date_updated": datetime.now(UTC).isoformat(),
                },
                primary_key="key",
            )

    def _release_locked(self, text: str, newest: _Entry) -> None:
        """mark the key whose text is ``text`` gone from the buffer; caller MUST hold ``self._lock``.

        :param text: the key's :func:`_buffer_text`
        :ptype text: str
        :param newest: the key's newest entry
        :ptype newest: _Entry
        :return: nothing
        :rtype: None
        """
        if newest.state in (_PENDING, _CLAIMED):
            self._held -= 1
        # the entry's state byte, in a copy of its run's states (one memcpy): no run appended, nothing merged. The run's
        # write stays referenced until a merge into the oldest run drops the gone entry
        runs = self._runs
        run = runs[newest.run]
        states = run.states[: newest.row] + bytes((_GONE,)) + run.states[newest.row + 1 :]
        changed = _Run(keys=run.keys, offsets=run.offsets, states=states, firsts=run.firsts, writes=run.writes)
        self._runs = (*runs[: newest.run], changed, *runs[newest.run + 1 :])

    async def add(self, table_name: str, entity_id: Any, data: dict[str, Any], retries: int = 0) -> None:
        """Add or replace a pending write for the given entity."""
        async with self._lock:
            self._add_locked(table_name, entity_id, data, retries)

    async def drain(self, decode: Callable[[str, str], dict[str, Any]] | None = None) -> list[PendingWrite]:
        """Claim all un-claimed pending writes for flushing.

        marks the returned writes in-flight so a concurrent drain cannot
        re-claim them, but does NOT delete their durable rows: the buffer entry
        is reclaimed only once :meth:`ack` confirms the L3 write landed (or
        :meth:`re_enqueue` re-arms it for retry). this is the write-through
        ordering — persist to L3 first, evict from L1 only after the durable
        write is acked — so a crash mid-flush replays the write instead of
        losing it from both tiers.

        with an L1 backend every claimed row is read back from its JSON text -- on the same
        process's next drain as well as after a crash -- so its UUIDs, instants, Decimals and
        bytes arrive as the strings the encoder wrote. ``decode`` turns that text back into the
        typed row the write was made with; :func:`flush_pending` passes one that rehydrates
        through the owning collection's ``decode_row``. a row ``decode`` cannot read is handed on as
        parsed JSON and logged, so the flush's retry budget decides its fate instead of one
        unreadable row stopping every drain.

        Without an L1 backend the writes come in the order their keys entered the buffer.

        :param decode: ``(table_name, json_text) -> row``, or ``None`` for plain ``json.loads``
        :ptype decode: Callable[[str, str], dict[str, Any]] | None
        :return: pending writes newly claimed by this call
        :rtype: list[PendingWrite]
        """
        async with self._lock:
            # newest first: the runs grow toward the oldest, so each merge copies about the entries merged so far and
            # the whole drain about twice the buffer's (oldest first copied the oldest run once a run: 25x, at 100,000)
            runs = self._runs
            run = runs[-1] if runs else _run_of((), b"", array("q"), ())
            for at in range(len(runs) - 2, -1, -1):
                run = _merged(runs[at], run, oldest=at == 0)
            if len(runs) == 1 and _GONE in run.states:
                # acks and removals mark their entries gone in place; a drain drops them
                run = _merged(_run_of((), b"", array("q"), ()), run, oldest=True)
            states = bytearray(run.states)
            claimed: list[PendingWrite] = []
            recovered: list[str] = []
            if self._l1 is not None:
                rows = whole_read(
                    self._l1.execute_query("SELECT * FROM write_buffer"),
                    kind="drain",
                    reason="the process's own write buffer (L1, never L3): every pending write is claimed to flush",
                    ruling="Pace 2026-11-12: a drain of a process-local buffer reads it whole (declared kind drain)",
                    proof="packages/core/tests/unit/collections/test_write_buffer_typed_recovery.py::test_drain_reads_only_the_process_local_write_buffer",
                )
                for row in rows:
                    text = _buffer_text(self._key(row["table_name"], row["entity_id"]))
                    at = run.find(text)
                    if at >= 0 and states[at] in (_CLAIMED, _RECOVERED):
                        continue
                    raw_data = row["data"]
                    parsed_data = (
                        _decoded_row(row["table_name"], row["entity_id"], raw_data, decode)
                        if isinstance(raw_data, str)
                        else raw_data
                    )
                    claimed.append(
                        PendingWrite(
                            table_name=row["table_name"],
                            entity_id=row["entity_id"],
                            data=parsed_data,
                            retries=row["retries"],
                        )
                    )
                    if at < 0 or states[at] == _GONE:
                        recovered.append(text)
                    else:
                        states[at] = _CLAIMED
            else:
                order = sorted(
                    (first, at) for at, (state, first) in enumerate(zip(run.states, run.firsts)) if state == _PENDING
                )
                for _, at in order:
                    write = run.writes[at]
                    assert write is not None  # narrow: a pending entry carries its write
                    claimed.append(write)
                    states[at] = _CLAIMED
            self._runs = (
                (_Run(keys=run.keys, offsets=run.offsets, states=bytes(states), firsts=run.firsts, writes=run.writes),)
                if run.writes
                else _NO_RUNS
            )
            for text in recovered:
                self._append(text, _RECOVERED, None, self._next_first)
                self._next_first += 1
            return claimed

    async def ack(self, table_name: str, entity_id: Any) -> None:
        """Evict a durably-persisted write once its L3 write is acked.

        version guard: the eviction is applied ONLY while the write is still the
        in-flight one this flush claimed. if a newer write for the same key
        coalesced in via :meth:`add` during the flush window (which clears the
        in-flight claim), the eviction is skipped so the newer value survives to
        be flushed on the next cycle.

        :param table_name: destination table name
        :ptype table_name: str
        :param entity_id: entity primary-key value
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        async with self._lock:
            text = _buffer_text(self._key(table_name, entity_id))
            newest = self._newest(text)
            if newest is None or newest.state not in (_CLAIMED, _RECOVERED):
                return
            self._release_locked(text, newest)
            if self._l1 is not None:
                l1_key = f"{table_name}:{entity_id}"
                self._l1.delete_by_id("write_buffer", l1_key, primary_key="key")

    async def re_enqueue(self, table_name: str, entity_id: Any, data: dict[str, Any], retries: int) -> bool:
        """Return a failed write to the buffer for a later retry, version-guarded.

        re-enqueues ONLY while the write is still the in-flight one this flush
        claimed. a stale failed write can therefore never overwrite a newer value
        that coalesced in during the flush window — the newer :meth:`add` cleared
        the in-flight claim, so this re-enqueue is dropped and the newer value is
        kept (lost-update protection).

        :param table_name: destination table name
        :ptype table_name: str
        :param entity_id: entity primary-key value
        :ptype entity_id: Any
        :param data: row payload keyed by column name
        :ptype data: dict[str, Any]
        :param retries: updated failed-flush attempt count
        :ptype retries: int
        :return: True when re-enqueued, False when dropped as superseded
        :rtype: bool
        """
        async with self._lock:
            newest = self._newest(_buffer_text(self._key(table_name, entity_id)))
            if newest is None or newest.state not in (_CLAIMED, _RECOVERED):
                return False
            self._add_locked(table_name, entity_id, data, retries)
            return True

    async def remove(self, table_name: str, entity_id: Any) -> bool:
        """Remove a pending write. Returns True if it existed."""
        async with self._lock:
            text = _buffer_text(self._key(table_name, entity_id))
            newest = self._newest(text)
            existed = newest is not None and newest.state in (_PENDING, _CLAIMED)
            if newest is not None and newest.state != _GONE:
                self._release_locked(text, newest)
            if self._l1 is not None:
                l1_key = f"{table_name}:{entity_id}"
                self._l1.delete_by_id("write_buffer", l1_key, primary_key="key")
            return existed

    def run_sizes(self) -> tuple[int, ...]:
        """how many entries each run holds, oldest first: what the buffer keeps, gone keys' entries included.

        :return: each run's count of entries
        :rtype: tuple[int, ...]
        """
        return tuple(len(run.writes) for run in self._runs)

    def held_runs(self) -> tuple[_Run, ...]:
        """the runs as the buffer holds them, oldest first: immutable values, each change a new tuple, so a caller
        holding one sees it unchanged after any later add, ack or drain.

        :return: the runs
        :rtype: tuple[_Run, ...]
        """
        return self._runs

    def pending_count(self) -> int:
        """Return the number of pending writes in the buffer."""
        return self._held


def _decoded_row(
    table_name: str,
    entity_id: str,
    text: str,
    decode: Callable[[str, str], dict[str, Any]] | None,
) -> dict[str, Any]:
    """one buffered row read back from its JSON text, typed when a decoder is given.

    :param table_name: the row's destination table
    :ptype table_name: str
    :param entity_id: the row's entity id, as the buffer stores it
    :ptype entity_id: str
    :param text: the stored JSON text
    :ptype text: str
    :param decode: the typed decoder, or ``None`` for plain ``json.loads``
    :ptype decode: Callable[[str, str], dict[str, Any]] | None
    :return: the row
    :rtype: dict[str, Any]
    """
    result: dict[str, Any] | None = None
    if decode is not None:
        try:
            result = decode(table_name, text)
        except (ValueError, TypeError, CorruptCacheEntry) as exc:
            # the flush retry budget governs it from here: its L3 write fails on the untyped
            # values, is retried, and is dropped with a permanent-failure log if it never lands
            log.error(
                "buffered write could not be rehydrated by its table's collection; flushing it as "
                "parsed JSON. inspect the write_buffer row for a value its column type cannot hold",
                extra={"extra_data": {"table": table_name, "entity_id": entity_id, "error": str(exc)}},
            )
    if result is None:
        result = json.loads(text)
    return result


def _collection_decoder(registry: CollectionRegistry) -> Callable[[str, str], dict[str, Any]]:
    """the decoder :func:`flush_pending` hands :meth:`WriteBuffer.drain`: the owning collection's own decode.

    Every registered collection decodes the text through
    :meth:`~threetears.core.collections.base.BaseCollection.decode_row`: its codec, then its
    declared instants rehydrated aware (a legacy naive or ``str(dt)`` spelling read as UTC). A
    :class:`~threetears.core.collections.schema_backed.SchemaBackedCollection`'s codec types every
    declared column besides; a durable-store, dynamic or hand-written collection gets at least its
    instants back typed. Both encoders the text could have come from write the one stored form
    (:func:`~threetears.core.backends.schema_sql.json_default`). A table with no registered
    collection gets the parsed JSON: without a collection there is nothing to type it by.

    :param registry: the registry resolving a table to its collection
    :ptype registry: CollectionRegistry
    :return: ``(table_name, json_text) -> row``
    :rtype: Callable[[str, str], dict[str, Any]]
    """

    def decode(table_name: str, text: str) -> dict[str, Any]:
        """rehydrate one buffered row through its table's collection.

        :param table_name: the row's destination table
        :ptype table_name: str
        :param text: the stored JSON text
        :ptype text: str
        :return: the typed row, or the parsed JSON when the table has no collection
        :rtype: dict[str, Any]
        :raises CorruptCacheEntry: when a declared instant will not parse
        """
        collection = registry.get_collection(table_name)
        result: dict[str, Any]
        if collection is not None:
            result = collection.decode_row(text.encode("utf-8"))
        else:
            result = json.loads(text)
        return result

    return decode


def _orders_after(pending: dict[str, Any], incoming: dict[str, Any]) -> bool:
    """whether a pending row carries a compare-and-swap order strictly after an incoming one.

    Rows without an order -- every write that is not a compare-and-swap -- compare as not after,
    so they keep the buffer's ordinary last-write-wins coalescing.

    :param pending: the row already buffered
    :ptype pending: dict[str, Any]
    :param incoming: the row being added
    :ptype incoming: dict[str, Any]
    :return: whether ``pending`` must be kept over ``incoming``
    :rtype: bool
    """
    pending_order = l2_order_of(pending)
    incoming_order = l2_order_of(incoming)
    return pending_order is not None and incoming_order is not None and pending_order > incoming_order


def _toposort_pending(
    pending: list[PendingWrite],
    parent_key_map: dict[str, str] | None = None,
) -> list[PendingWrite]:
    """Sort pending writes so parents are flushed before children.

    parent_key_map maps table_name -> FK column pointing to parent.
    Default: {"messages": "parent_message_id"}.
    """
    if parent_key_map is None:
        parent_key_map = {"messages": "parent_message_id"}

    # Separate into tables with FK deps vs without
    no_deps: list[PendingWrite] = []
    with_deps: list[PendingWrite] = []
    for pw in pending:
        if pw.table_name in parent_key_map:
            with_deps.append(pw)
        else:
            no_deps.append(pw)

    if not with_deps:
        return no_deps

    # Kahn's algorithm for each table group
    by_id: dict[Any, PendingWrite] = {pw.entity_id: pw for pw in with_deps}
    in_degree: dict[Any, int] = {pw.entity_id: 0 for pw in with_deps}
    children_of: dict[Any, list[Any]] = {}

    for pw in with_deps:
        parent_col = parent_key_map[pw.table_name]
        parent_id = pw.data.get(parent_col)
        if parent_id is not None and parent_id in by_id:
            in_degree[pw.entity_id] = in_degree.get(pw.entity_id, 0) + 1
            children_of.setdefault(parent_id, []).append(pw.entity_id)

    queue = [eid for eid, deg in in_degree.items() if deg == 0]
    ordered: list[PendingWrite] = []
    while queue:
        eid = queue.pop(0)
        ordered.append(by_id[eid])
        for child_id in children_of.get(eid, []):
            in_degree[child_id] -= 1
            if in_degree[child_id] == 0:
                queue.append(child_id)

    # Handle cycles — append remaining so nothing is silently dropped
    if len(ordered) < len(with_deps):
        ordered_ids = {pw.entity_id for pw in ordered}
        for pw in with_deps:
            if pw.entity_id not in ordered_ids:
                ordered.append(pw)

    return no_deps + ordered


def _resolve_batch_backend(
    sorted_pending: list[PendingWrite],
    registry: CollectionRegistry,
) -> Any:
    """Resolve the single shared backend for an atomic-batch flush, or ``None``.

    The atomic-batch path is only taken when **every** pending collection resolves to
    the **same** backend object AND that backend exposes a usable ``transaction()``.
    Any of: an unregistered table, a missing backend, divergent backends, or a backend
    without ``transaction()`` → ``None``, so the caller degrades to the per-entity loop.
    (The git store's ``transaction()`` is a change set: its batch is one commit.)

    :param sorted_pending: toposorted pending writes.
    :ptype sorted_pending: list[PendingWrite]
    :param registry: the collection registry.
    :ptype registry: CollectionRegistry
    :return: the shared backend exposing ``transaction()``, or ``None``.
    :rtype: Any
    """
    backend: Any = None
    for pw in sorted_pending:
        collection = registry.get_collection(pw.table_name)
        if collection is None:
            return None
        b = registry.get_l3_pool(pw.table_name)
        if b is None or not callable(getattr(b, "transaction", None)):
            return None
        if backend is None:
            backend = b
        elif b is not backend:
            return None
    return backend


def _absorbs_conflicts(collection: Any) -> bool:
    """Whether this collection's table declares that a conflicting row is absorbed.

    ``on_conflict="ignore"`` generates ``ON CONFLICT (pk) DO NOTHING``, whose 0 rowcount
    on a duplicate is the policy working rather than a write going missing. The policy is
    declared on the schema for a ``SchemaBackedCollection`` and as a class attribute on a
    ``DurableStoreCollection``, so both are consulted -- schema first, because a
    schema-backed collection carries both and the schema is the one its SQL is generated
    from.

    Compared by equality against the literal policy so a mock collection, whose
    auto-created attributes are truthy objects rather than strings, reads as declaring
    nothing.

    :param collection: the collection whose table policy is being read
    :ptype collection: Any
    :return: whether a 0 rowcount is this table's expected duplicate outcome
    :rtype: bool
    """
    schema = getattr(collection, "schema", None)
    policy = getattr(schema, "on_conflict", None) if schema is not None else None
    if policy is None:
        policy = getattr(collection, "on_conflict", None)
    return policy == "ignore"


def _write_landed(collection: Any, pending: PendingWrite, rows_affected: Any) -> bool:
    """Report whether a flushed write actually landed a row, saying so when it did not.

    ``save_entity`` treats a 0 rowcount as a hard failure, so the synchronous write path
    can never report a row it did not persist. The deferred path replays the same write
    through ``persist_to_store`` and its rowcount is the only statement the durable tier
    makes about it -- a write the tier declined is evicted from the buffer either way, so
    counting it as flushed and logging nothing turns a permanent loss into a success.

    The collection answers what its own 0 means, so the report is graded rather than
    uniform:

    - ``persists_l2_order`` with a row carrying its compare-and-swap order says the write
      was fenced on that order, so a 0 means L3 already holds a newer one -- a later swap
      that built on this row persisted first. Nothing was lost; reported at debug.
    - ``emits_cas_fence`` says every write it generates carries a fence, so a 0 is
      provably a lost race -- reported at error. Only reachable through a buffer that
      already held rows when the table was fenced, because
      ``SchemaBackedCollection._reject_deferred_flush_on_cas_null_safe`` refuses to
      construct a fenced collection whose table is configured for deferred flush.
    - ``on_conflict="ignore"`` says a conflicting row is absorbed on purpose, so a 0 is
      the ordinary duplicate outcome -- reported at debug. Escalating it would put a
      warning in the log for every duplicate the policy exists to absorb.
    - Otherwise 0 is ambiguous, so it is reported at warning rather than escalated on a
      guess.

    Both flags are read defensively because a bare mock collection's auto-created
    attributes are truthy objects: ``emits_cas_fence`` for identity, so a mock does not
    read as fenced, and ``on_conflict`` by equality against the literal policy.

    A count that is not ``0`` counts as landed, ``None`` included. A backend that
    answers ``None`` violates ``save_to_store``'s ``-> int`` contract, and reading it as
    a loss here would newly drop writes for a non-SQL ``DurableStore`` that has always
    answered that way.

    :param collection: the collection whose ``persist_to_store`` produced the count
    :ptype collection: Any
    :param pending: the buffered write, for the address the log has to carry
    :ptype pending: PendingWrite
    :param rows_affected: rows-affected count reported by the durable tier
    :ptype rows_affected: Any
    :return: whether the write may be counted as persisted
    :rtype: bool
    """
    if rows_affected != 0:
        return True
    address = {
        "table": pending.table_name,
        "entity_id": str(pending.entity_id),
        "rows_affected": rows_affected,
    }
    if getattr(collection, "persists_l2_order", False) is True and l2_order_of(pending.data) is not None:
        log.debug(
            "Deferred compare-and-swap write superseded by a newer order already in L3; the later "
            "swap built on this one and carries it",
            extra={"extra_data": address},
        )
    elif getattr(collection, "emits_cas_fence", False) is True:
        log.error(
            "Deferred L3 write lost its CAS race and was dropped; the buffered payload "
            "was decided under a fence value another writer has since moved",
            extra={"extra_data": address},
        )
    elif _absorbs_conflicts(collection):
        log.debug(
            "Deferred L3 write matched an existing row on a table that absorbs conflicts",
            extra={"extra_data": address},
        )
    else:
        log.warning(
            "Deferred L3 write affected no row and was dropped from the buffer; the durable tier took nothing for it",
            extra={"extra_data": address},
        )
    return False


async def _flush_batch_atomic(
    sorted_pending: list[PendingWrite],
    registry: CollectionRegistry,
    backend: Any,
    landed: list[PendingWrite],
) -> int:
    """Persist the whole toposorted batch inside ONE backend transaction.

    Raises on any failure so the caller can fall back to the per-entity loop (which
    keeps the ``_is_fk_violation`` classification + re-enqueue). The transaction is
    rolled back by the backend's ``transaction()`` context manager on exception.

    :param sorted_pending: toposorted pending writes.
    :ptype sorted_pending: list[PendingWrite]
    :param registry: the collection registry.
    :ptype registry: CollectionRegistry
    :param backend: the shared backend exposing ``transaction()``.
    :ptype backend: Any
    :param landed: extended, once the transaction has committed, with the writes the durable tier
        took a row for.
    :ptype landed: list[PendingWrite]
    :return: number of entities the durable tier took a row for. Less than the whole
        batch when a write reported 0 rows; the transaction still commits and the
        caller still acks the batch, so a declined write is reported, not replayed.
    :rtype: int
    """
    took: list[PendingWrite] = []
    async with backend.transaction() as conn:
        for pw in sorted_pending:
            collection = registry.get_collection(pw.table_name)
            # _resolve_batch_backend already proved every table resolves to a
            # collection (and to this same backend); assert for the type-checker.
            assert collection is not None
            rows_affected = await collection.persist_to_store(pw.data, conn=conn)
            if _write_landed(collection, pw, rows_affected):
                took.append(pw)
    # only now: a transaction that raised committed none of them
    landed.extend(took)
    return len(took)


async def _flush_per_entity(
    sorted_pending: list[PendingWrite],
    write_buffer: WriteBuffer,
    registry: CollectionRegistry,
    landed: list[PendingWrite],
) -> int:
    """Persist each pending write independently, re-enqueuing on failure.

    The original per-entity flush loop: an unregistered table is skipped, and a failed
    write is re-enqueued under the FK-aware retry policy (FK violations get the generous
    ``_FK_RETRY_LIMIT`` budget; all other failures use ``_MAX_FLUSH_RETRIES``).

    :param sorted_pending: toposorted pending writes.
    :ptype sorted_pending: list[PendingWrite]
    :param write_buffer: the write buffer (for re-enqueue).
    :ptype write_buffer: WriteBuffer
    :param registry: the collection registry.
    :ptype registry: CollectionRegistry
    :param landed: extended with each write the durable tier took a row for.
    :ptype landed: list[PendingWrite]
    :return: number of entities successfully persisted.
    :rtype: int
    """
    flushed = 0
    for pw in sorted_pending:
        collection = registry.get_collection(pw.table_name)
        if collection is None:
            log.warning(
                "No collection registered for table, skipping flush",
                extra={"extra_data": {"table": pw.table_name, "entity_id": str(pw.entity_id)}},
            )
            # unrecoverable (no collection can ever persist it): release the
            # in-flight claim so the poison write does not stay claimed forever.
            await write_buffer.ack(pw.table_name, pw.entity_id)
            continue
        try:
            rows_affected = await collection.persist_to_store(pw.data)
            if _write_landed(collection, pw, rows_affected):
                flushed += 1
                landed.append(pw)
            # the durable tier answered -> safe to evict from the buffer either way. A
            # declined write is not re-enqueued: the buffered payload is what the tier
            # already refused, so replaying it refuses again, and an ``ON CONFLICT DO
            # NOTHING`` no-op would burn the retry budget on every duplicate.
            await write_buffer.ack(pw.table_name, pw.entity_id)
        except RowcountUnreadableError as exc:
            # the statement ran -- the tag is its answer -- but how many rows it touched is unknown.
            # replaying it would run it again, and dropping it as "affected no row" would be a guess,
            # so it is acknowledged, counted as landed (an announce for a row that did not change
            # is harmless; a missed one leaves a derived cache stale), and said, distinctly.
            contain(
                exc,
                log=log,
                what="Deferred L3 write ran but its row count could not be read; acknowledged, not replayed",
                extra={"table": pw.table_name, "entity_id": str(pw.entity_id), "error": str(exc)},
                level="error",
            )
            flushed += 1
            landed.append(pw)
            await write_buffer.ack(pw.table_name, pw.entity_id)
        except Exception as exc:
            # FK violations are "my parent hasn't landed yet" -- treat
            # them as deferral, not failure: re-enqueue with the
            # generous _FK_RETRY_LIMIT budget so the parent has time
            # to land in a subsequent drain. All other errors use the
            # original _MAX_FLUSH_RETRIES budget.
            fk_violation = _is_fk_violation(exc)
            retry_limit = _FK_RETRY_LIMIT if fk_violation else _MAX_FLUSH_RETRIES
            next_retry = pw.retries + 1
            if next_retry >= retry_limit:
                # Permanent drop. For FK violations, this means the
                # parent will never land -- log as an "orphan chain"
                # event so operators can run the conversation repair
                # endpoint (or otherwise reset the cache).
                contain(
                    exc,
                    log=log,
                    what=(
                        "Orphan chain — FK violation exhausted retries, dropping"
                        if fk_violation
                        else "Flush write permanently failed after max retries, dropping"
                    ),
                    extra={
                        "table": pw.table_name,
                        "entity_id": str(pw.entity_id),
                        "retries": next_retry,
                        "retry_limit": retry_limit,
                        "fk_violation": fk_violation,
                        "error": str(exc),
                    },
                    level="error",
                )
                # permanent drop: release the in-flight claim and evict the
                # durable row (version-guarded — a newer coalesced write is kept).
                await write_buffer.ack(pw.table_name, pw.entity_id)
            else:
                # An FK deferral repeats once per drain until the parent lands or
                # the budget runs out, and a parent that was deleted never lands:
                # one row wrote this line up to _FK_RETRY_LIMIT times. The first
                # deferral is the event; the repeats are DEBUG, and the drop above
                # is the ERROR that says it never landed.
                level = logging.WARNING if (not fk_violation or next_retry == 1) else logging.DEBUG
                log.log(
                    level,
                    "Flush write deferred (FK parent pending), re-adding to buffer"
                    if fk_violation
                    else "Flush write failed, re-adding to buffer for retry",
                    extra={
                        "extra_data": {
                            "table": pw.table_name,
                            "entity_id": str(pw.entity_id),
                            "retry": next_retry,
                            "retry_limit": retry_limit,
                            "fk_violation": fk_violation,
                            "error": str(exc),
                        }
                    },
                )
                await write_buffer.re_enqueue(pw.table_name, pw.entity_id, pw.data, retries=next_retry)
    log.debug("Flush complete", extra={"extra_data": {"flushed": flushed, "total": len(sorted_pending)}})
    return flushed


async def flush_pending(
    write_buffer: WriteBuffer,
    registry: CollectionRegistry,
    parent_key_map: dict[str, str] | None = None,
) -> int:
    """Drain the write buffer and persist all pending writes to the durable tier.

    **Retry partition (orphan isolation).** After the toposort, pending writes are
    split by their ``retries`` count. Writes with ``retries == 0`` (never failed)
    form the *fresh* set and take the atomic-batch fast path; writes with
    ``retries > 0`` (already failed at least once — e.g. an FK orphan whose parent
    row was deleted and is never coming back) route STRAIGHT to the per-entity loop.
    This keeps a previously-failing write out of the atomic transaction entirely: a
    single un-satisfiable FK among the already-failed writes can never abort the
    batch, so a co-buffered fresh write still commits instead of being dragged into
    per-entity fallback every cycle for the whole ``_FK_RETRY_LIMIT`` budget. The
    already-failed writes keep the per-entity loop's ``_is_fk_violation``
    classification + FK-aware re-enqueue, so a genuinely-transient FK still drains
    once its parent lands.

    **Fresh-set atomic batch.** When every collection in the fresh set shares ONE
    backend that exposes a usable ``transaction()``, the toposorted fresh writes are
    persisted inside a SINGLE ``async with backend.transaction() as conn`` (one DB tx
    for a SQL backend; one commit for a git backend) — each write threading ``conn``
    through ``persist_to_store``. **Graceful degrade**: on ANY exception in the batch
    path the fresh set falls back to the per-entity loop, which keeps the
    ``_is_fk_violation`` classification + re-enqueue intact. A backend without
    ``transaction()`` degrades to the per-entity
    loop directly. The total returned is the sum of both paths' flushed counts.

    :param write_buffer: the coalescing write buffer to drain.
    :ptype write_buffer: WriteBuffer
    :param registry: the collection registry resolving table → collection + backend.
    :ptype registry: CollectionRegistry
    **Write generations.** Once everything above has run, each table whose collection is switched
    on to carry a write generation (``BaseCollection.write_generation``) advances it ONCE for this
    flush, however many of its rows landed, and announces each landed row again naming that
    advance (``BaseCollection.announce_flushed``). Not at save time: the row is not in L3 until
    here, and a cache derived from the table reads L3. One table's failed advance does not stop
    the next table's; the first failure is raised after every table has been attempted, and by
    then every write the durable tier took has been acknowledged, so nothing is replayed for it.

    :param parent_key_map: optional table → parent-FK-column map for toposort.
    :ptype parent_key_map: dict[str, str] | None
    :return: number of entities successfully persisted (both paths summed).
    :rtype: int
    :raises GenerationUnavailableError: when a switched-on table's rows landed and its write
        generation could not be advanced.
    """
    pending = await whole_read(
        write_buffer.drain(decode=_collection_decoder(registry)),
        kind="drain",
        reason="the process's own write buffer (L1, never L3): every pending write is claimed to flush",
        ruling="Pace 2026-11-12: a drain of a process-local buffer reads it whole (declared kind drain)",
        proof="packages/core/tests/unit/collections/test_write_buffer_typed_recovery.py::test_drain_reads_only_the_process_local_write_buffer",
    )
    if not pending:
        return 0

    sorted_pending = _toposort_pending(pending, parent_key_map)

    # Partition by retry count: fresh (retries == 0) writes are eligible for the
    # atomic batch; already-failed (retries > 0) writes route straight to the
    # per-entity loop so a poisoned orphan can never abort the fresh batch.
    fresh: list[PendingWrite] = [pw for pw in sorted_pending if pw.retries == 0]
    already_failed: list[PendingWrite] = [pw for pw in sorted_pending if pw.retries > 0]

    flushed = 0
    landed: list[PendingWrite] = []

    if fresh:
        backend = _resolve_batch_backend(fresh, registry)
        if backend is not None:
            try:
                batch_flushed = await _flush_batch_atomic(fresh, registry, backend, landed)
                log.debug(
                    "Flush complete (atomic batch)",
                    extra={"extra_data": {"flushed": batch_flushed, "total": len(fresh)}},
                )
                flushed += batch_flushed
                # transaction committed -> now safe to evict the whole batch.
                # ack is version-guarded, so any write that coalesced in during
                # the commit window is preserved for the next cycle.
                for pw in fresh:
                    await write_buffer.ack(pw.table_name, pw.entity_id)
            except Exception as exc:
                # Graceful degrade: the whole transaction rolled back, so NOTHING
                # in the fresh set was committed -- replay it through the per-entity
                # loop, which preserves the FK-aware re-enqueue policy per write.
                # The fallback is the safety net, never weakened.
                contain(
                    exc,
                    log=log,
                    what="Atomic batch flush failed, falling back to per-entity flush",
                    extra={"total": len(fresh), "error": str(exc)},
                )
                flushed += await _flush_per_entity(fresh, write_buffer, registry, landed)
        else:
            # No single shared transaction-capable backend: degrade the fresh set to the
            # per-entity loop directly.
            flushed += await _flush_per_entity(fresh, write_buffer, registry, landed)

    if already_failed:
        # Previously-failed writes are isolated in the per-entity loop so one
        # un-satisfiable FK orphan cannot abort the fresh batch above.
        flushed += await _flush_per_entity(already_failed, write_buffer, registry, landed)

    await _announce_landed(landed, registry)
    return flushed


async def _announce_landed(landed: list[PendingWrite], registry: CollectionRegistry) -> None:
    """advance each switched-on table's write generation once for this flush, and announce its rows.

    :param landed: the writes the durable tier took a row for, in the order they landed.
    :ptype landed: list[PendingWrite]
    :param registry: the collection registry.
    :ptype registry: CollectionRegistry
    :return: nothing.
    :rtype: None
    :raises GenerationUnavailableError: the first table's failure to advance, after every table
        has been attempted.
    """
    by_table: dict[str, list[dict[str, Any]]] = {}
    for pw in landed:
        by_table.setdefault(pw.table_name, []).append(pw.data)
    failure: GenerationUnavailableError | None = None
    for table_name, rows in by_table.items():
        collection = registry.get_collection(table_name)
        # read off the declaration rather than called on whatever is registered: a collection that
        # is not switched on has nothing to announce, and a stand-in for one has no such method.
        if collection is None or not isinstance(getattr(collection, "write_generation", None), WriteGeneration):
            continue
        try:
            await collection.announce_flushed(rows)
        except GenerationUnavailableError as exc:
            # logged where the advance failed; the rows were announced all the same
            if failure is None:
                failure = exc
    if failure is not None:
        raise failure
