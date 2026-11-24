"""The SQLite L1's bound: a byte budget per process, least-recently-used eviction of ``EVICTABLE`` tables only.

**Why** (Pace, 2026-11-15: "this L1 should have LRU as well"; "some tables can't be LRU'd"). The L1 is a file on
``/dev/shm`` (:mod:`~threetears.core.cache.l1_file`), and tmpfs pages are the container's memory: a size cannot be
right while any table grows without a cap. So every L1 table declares its policy where it is declared
(:mod:`threetears.core.eviction`: ``EVICTABLE`` or ``KEEP``, with why, no default), and the process holds its L1
under one byte budget:

- ``KEEP`` tables (complete copies, rows a write in flight or a settle needs, state a reader treats as complete) are
  never evicted. They count first; when they alone pass the budget it is a sizing failure (counted and logged),
  never an eviction. The L1 does not refuse a ``KEEP`` write: those rows are correctness, and the volume's size and
  autosize are what make room.
- ``EVICTABLE`` tables (rows cached on a read; a miss reads through to L2/L3 and refills) share what is left, least
  recently used first across every such table of the process. An eviction is a cache miss, never a wrong answer:
  a row evicted is a row the next read refills, and invalidation by epochs is unchanged (no TTLs).

**Recency, cheaply, and columnar** (Pace 2026-11-23: no per-row Python objects, no shared state a thread changes).
No write per read, and no lock on the hot path: a read hit, a write and a delete each hand the ledger's thread one
immutable hand-off (the keys as one packed batch, :mod:`threetears.core.packed_keys`, their bytes and the clock) on a
queue, and touch nothing else. The thread alone owns each ``EVICTABLE`` table's recency: its rows least recent first,
as one packed batch of key texts beside parallel ``array`` columns (estimated bytes, last use, order of use), and the
hand-offs not yet folded in as a tuple of such batches. It folds them (latest use of a key wins) when the tuple grows
past a bound, and before it picks evictions; the least recent rows are then the front of each table's batch, taken
across the tables in order of use. Every value is replaced whole, never changed in place. At 1M cached rows this holds
about 40 bytes a row where the ``OrderedDict`` it replaces held about 300 (``test_l1_lru.py`` measures both).
``KEEP`` tables keep nothing in memory.

The ledger decides on an upper bound of its bytes between folds (a rewrite counts its row's bytes again, a delete
frees none until the fold), so a bound under the allowance needs no fold; past it the ledger folds, and its exact
bytes decide, as before.

**Bytes, estimated then measured.** A row's bytes are estimated on write from its serialized values; a thread
measures each table's real pages (``dbstat``) every :data:`_MEASURE_SECONDS` and scales the estimate by what it
found, so the hot path never touches the file system and the budget is held in real bytes.

**The budget**: ``THREETEARS_L1_BUDGET_BYTES`` (the deploy config sets it from SIZING.md), else half the L1
directory's file system (a memory-backed ``emptyDir`` is mounted at its ``sizeLimit``), said once in the log.

**Metrics**: the cache family (:mod:`threetears.core.eviction`), ``tier="l1"``: ``name`` a table for per-table
values, ``""`` for the process. The ledger sets the budget, hits, misses, evictions, evicted bytes and sizing
failures; ``threetears_cache_held_bytes{tier="l1"}`` is SQ2's sampler's (``dbstat`` per table), which reads each
table's policy from the declarations here. The ledger's own byte accounting decides evictions and is held within a
stated tolerance of that measurement by a test (``test_l1_lru.py``).
"""

from __future__ import annotations

import ast
import enum
import os
import queue
import threading
import time
import weakref
from array import array
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Protocol

from threetears.core.config import DEFAULT_L1_LEDGER_DRAIN_TIMEOUT_SECONDS
from threetears.core.eviction import (
    BUDGET_BYTES,
    EVICTED_AGE_SECONDS,
    EVICTED_BYTES,
    EVICTIONS,
    HITS,
    MISSES,
    SIZING_FAILURES,
    Declared,
    Policy,
    Tier,
    policy_declare,
    policies_declared,
)
from threetears.core.packed_keys import NO_KEYS, key_text, packed, parts_of, texts
from threetears.observe import BuildOnce, FrozenWeakValues, contain, get_logger

__all__ = [
    "L1_EVICTION_INFO",
    "L1Ledger",
    "TableCounters",
    "declare_l1_table",
    "ledger",
    "row_bytes",
]

log = get_logger(__name__)

#: the key of a SQLAlchemy table's ``info`` that holds its declared L1 policy
L1_EVICTION_INFO: Final = "l1_eviction"
#: the process's L1 byte budget, as the deploy config sets it from SIZING.md
_L1_BUDGET_ENV: Final = "THREETEARS_L1_BUDGET_BYTES"
#: how often the thread measures the real pages of every table: a ``dbstat`` walk of the file, off the loop
_MEASURE_SECONDS: Final = 30.0
#: evict down to this share of what is left for ``EVICTABLE`` rows, so an eviction runs once per stretch of
#: writes, not once per write
_EVICT_TO: Final = 0.9
#: an estimate's fixed cost per row (SQLite's record header, cell pointer and the key's index entry, roughly) until
#: the first measure scales it
_ROW_OVERHEAD: Final = 48
_TIER: Final = Tier.L1.value
#: rows of a write checked against the budget at a time, so a big batch evicts as it crosses it, as one row would
_SLICE_ROWS: Final = 256
#: a table folds its hand-offs into its recency past this many rows of them, or past a quarter of its rows if more:
#: a fold copies the table's columns (C), so each row is copied a few times over its life, never once a hand-off
_FOLD_ROWS: Final = 4096
#: a table's hand-offs are joined into one batch past this many (a read's hand-off is one row)
_HANDOFF_BATCHES: Final = 32
#: a hand-off row's bytes when it is a read's touch, or a delete
_TOUCH: Final = -1
_REMOVED: Final = -2
#: tags a key part that is not text, in its key text: the part's type, then its spelling
_TAG: Final = "\x1d"
#: advanced by every L1 declaration, so a backend that read its tables' policies sees a table become KEEP; read through
#: :func:`policy_epoch` (an import of the name would keep the value it had then)
_policy_epoch: int = 0


def policy_epoch() -> int:
    """the L1 policy epoch now: advanced by every :func:`declare_l1_table`.

    :return: it
    :rtype: int
    """
    return _policy_epoch


def declare_l1_table(name: str, policy: Declared) -> None:
    """declare one L1 table's policy in the cache family's declarations (:func:`threetears.core.eviction.policy_declare`,
    ``Tier.L1``, the kind named by the table), where every reader of a table's policy finds it
    (:func:`~threetears.core.eviction.policy_lookup`): the ledger, and the held-bytes sampler's ``policy`` label.

    A table's name is its kind: one table, one policy, in every L1 of the process. ``TableSchema(l1_eviction=...)``
    reaches this where the table is declared, and ``SQLiteBackend.initialize`` for a table that carries its policy
    in ``Table.info`` (``info={L1_EVICTION_INFO: ...}``).

    Two declarations of one name that disagree (two tables of one name in one process, or one table two users
    read differently) resolve to ``KEEP``, said once in the log: one of its users needs it complete, and keeping a
    cache costs only bytes where evicting what a user needs complete answers wrongly.

    :param name: the table
    :ptype name: str
    :param policy: its policy and why
    :ptype policy: Declared
    :return: nothing
    :rtype: None
    """
    before = policies_declared(Tier.L1).get(name)
    if before is not None and (before.policy is policy.policy or before.policy is Policy.KEEP):
        if before.policy is not policy.policy:
            log.warning(
                "an L1 table declared EVICTABLE is also declared KEEP; it is kept",
                extra={"extra_data": {"table": name, "keep": before.why, "evictable": policy.why}},
            )
        return
    if before is not None:
        log.warning(
            "an L1 table declared EVICTABLE is also declared KEEP; it is kept",
            extra={"extra_data": {"table": name, "keep": policy.why, "evictable": before.why}},
        )
    kinds = enum.StrEnum("L1Table", {"TABLE": name})  # type: ignore[misc]
    policy_declare(Tier.L1, kinds, {kinds["TABLE"]: policy})
    global _policy_epoch
    _policy_epoch += 1


def row_bytes(values: Sequence[Any]) -> int:
    """a row's estimated bytes, from the values as written (serialized).

    :param values: the serialized values
    :ptype values: Sequence[Any]
    :return: the estimate
    :rtype: int
    """
    size = _ROW_OVERHEAD
    for value in values:
        if isinstance(value, str | bytes):
            size += len(value) + 2
        else:
            size += 9
    return size


def _budget_from_env(directory: Path) -> int:
    """the process's budget: ``THREETEARS_L1_BUDGET_BYTES``, else half of ``directory``'s file system.

    :param directory: where the L1 files are
    :ptype directory: Path
    :return: bytes
    :rtype: int
    :raises ValueError: when the variable is set and not a positive integer
    """
    named = os.environ.get(_L1_BUDGET_ENV)
    if named:
        budget = int(named)
        if budget < 1:
            raise ValueError(f"{_L1_BUDGET_ENV} must be a positive number of bytes, got {named!r}")
        return budget
    stat = os.statvfs(directory)
    budget = stat.f_blocks * stat.f_frsize // 2
    log.info(
        "L1 budget not set; half the L1 directory's file system",
        extra={"extra_data": {"env": _L1_BUDGET_ENV, "directory": str(directory), "budget_bytes": budget}},
    )
    return budget


class _Store(Protocol):
    """what the ledger needs of a backend: delete keys, and measure its tables."""

    store_name: str

    def evict_keys(self, table: str, keys: Sequence[tuple[Any, ...]]) -> None:
        """delete rows the ledger evicted, by their keys as already serialized; blocking."""
        ...

    def table_bytes(self) -> Mapping[str, int]:
        """each table's measured bytes."""
        ...

    def table_policies(self) -> Mapping[str, Declared]:
        """each table's declared policy."""
        ...

    def side_bytes(self) -> int:
        """the bytes the store holds beside its tables (the WAL)."""
        ...


@dataclass
class TableCounters:
    """a table's bound hit and miss counters, made once so a read pays no label lookup."""

    hits: Any
    misses: Any


def _part_text(part: Any) -> str:
    """a key part's text: text as itself (tagged when it starts with :data:`_TAG`), anything else tagged by type.

    :param part: a serialized key part (text, an integer, a float or None, as SQLite binds them)
    :ptype part: Any
    :return: its text
    :rtype: str
    """
    if type(part) is str:
        return _TAG + "s" + part if part.startswith(_TAG) else part
    if part is None:
        return _TAG + "n"
    if type(part) is int:
        return _TAG + "i" + str(part)
    if type(part) is float:
        return _TAG + "f" + repr(part)
    return _TAG + "r" + repr(part)


def _part_of(text: str) -> Any:
    """the key part whose text is ``text``: the inverse of :func:`_part_text`.

    :param text: a part's text
    :ptype text: str
    :return: the part
    :rtype: Any
    """
    if not text.startswith(_TAG):
        return text
    kind, spelled = text[1], text[2:]
    if kind == "s":
        return spelled
    if kind == "n":
        return None
    if kind == "i":
        return int(spelled)
    if kind == "f":
        return float(spelled)
    return ast.literal_eval(spelled)


def _key_text(key: Sequence[Any]) -> str:
    """a serialized key's text in a packed batch: exact, and :func:`_key_of` gives the key back.

    :param key: the serialized key
    :ptype key: Sequence[Any]
    :return: its text
    :rtype: str
    """
    for part in key:
        if type(part) is not str or part.startswith(_TAG):
            return key_text([_part_text(each) for each in key])
    return key_text(key)


def _key_of(text: str) -> tuple[Any, ...]:
    """the serialized key whose text is ``text``.

    :param text: a key's :func:`_key_text`
    :ptype text: str
    :return: the key
    :rtype: tuple[Any, ...]
    """
    return tuple(_part_of(part) for part in parts_of(text))


@dataclass(frozen=True)
class _Rows:
    """rows of a table's recency as columns: a packed batch of key texts and, row for row, the estimated bytes (in a
    hand-off :data:`_TOUCH` or :data:`_REMOVED` for a read or a delete), the clock at the use and the use's place in
    the ledger's order. Never changed: replaced whole."""

    texts: str
    sizes: array[int]
    ticks: array[float]
    seqs: array[int]

    def __len__(self) -> int:
        return len(self.sizes)


_NO_ROWS: Final = _Rows(NO_KEYS, array("q"), array("d"), array("q"))


def _joined(batches: Sequence[_Rows]) -> _Rows:
    """batches end to end, as one.

    :param batches: the batches
    :ptype batches: Sequence[_Rows]
    :return: one batch
    :rtype: _Rows
    """
    return _Rows(
        NO_KEYS + "".join(batch.texts[1:] for batch in batches),
        array("q", b"".join(batch.sizes.tobytes() for batch in batches)),
        array("d", b"".join(batch.ticks.tobytes() for batch in batches)),
        array("q", b"".join(batch.seqs.tobytes() for batch in batches)),
    )


@dataclass(frozen=True)
class _Table:
    """one ``EVICTABLE`` table's recency, owned by the ledger's thread and replaced whole.

    ``held``: its rows, distinct, least recently used first; ``estimated`` their bytes. ``pending``: the hand-offs not
    yet folded in, oldest first; ``unsettled`` the bytes their writes name (a rewrite counted again).
    """

    store: weakref.ReferenceType[Any]
    name: str
    held: _Rows = _NO_ROWS
    estimated: int = 0
    pending: tuple[_Rows, ...] = ()
    pending_rows: int = 0
    unsettled: int = 0


def _folded(table: _Table) -> _Table:
    """``table`` with its hand-offs folded into its rows: a key's latest use wins, a read's touch moves a held row (and
    is dropped for a row not held), a delete takes it out; the rows a hand-off names leave their place and follow the
    rest in order of use. Columns are copied in slices (C); the loops are over the hand-offs' rows and one pass over
    the held keys' texts.

    :param table: the table
    :ptype table: _Table
    :return: the table, folded
    :rtype: _Table
    """
    if not table.pending:
        return table
    # function-local: each named key's latest use, (bytes, tick, order); _TOUCH waits for the held row's bytes
    latest: dict[str, tuple[int, float, int]] = {}
    for batch in table.pending:
        for text, size, tick, seq in zip(texts(batch.texts), batch.sizes, batch.ticks, batch.seqs, strict=True):
            if size != _TOUCH:
                latest[text] = (size, tick, seq)
                continue
            before = latest.get(text)
            if before is None or before[0] == _TOUCH:
                latest[text] = (_TOUCH, tick, seq)
            elif before[0] != _REMOVED:
                latest[text] = (before[0], tick, seq)
    held = table.held
    names = texts(held.texts)
    cuts = [row for row, text in enumerate(names) if text in latest]
    kept: list[str] = []
    sizes, ticks, seqs = array("q"), array("d"), array("q")
    start = 0
    for cut in [*cuts, len(names)]:
        kept += names[start:cut]
        sizes += held.sizes[start:cut]
        ticks += held.ticks[start:cut]
        seqs += held.seqs[start:cut]
        start = cut + 1
    for cut in cuts:
        use = latest[names[cut]]
        if use[0] == _TOUCH:
            latest[names[cut]] = (held.sizes[cut], use[1], use[2])
    # the hand-offs' rows are in order of use already: each key follows the rest at its latest use (a walk, never a
    # sort, which would hold the GIL in C for the whole of it)
    for batch in table.pending:
        for text, seq in zip(texts(batch.texts), batch.seqs, strict=True):
            use = latest[text]
            if use[2] == seq and use[0] >= 0:
                kept.append(text)
                sizes.append(use[0])
                ticks.append(use[1])
                seqs.append(seq)
    return replace(
        table,
        held=_Rows(packed(kept), sizes, ticks, seqs),
        estimated=sum(sizes),
        pending=(),
        pending_rows=0,
        unsettled=0,
    )


#: a hand-off: ("w", store, table, packed texts, bytes, tick), ("t", store, table, text, tick),
#: ("r", store, table, packed texts), ("e", store, table), ("l", store id) or ("s", event, fold): set once those before
#: it are taken (and every table folded, when ``fold``)
_HandOff = tuple[Any, ...]


class L1Ledger:
    """the process's L1 byte budget and the recency of every ``EVICTABLE`` table in it (one per process).

    :param budget: bytes; ``None`` reads it from the environment when the first store joins
    :ptype budget: int | None
    :param thread: whether hand-offs, evictions and measures run on the ledger's own thread; ``False`` leaves the
        hand-offs queued for :meth:`drain`, :meth:`measure` and :meth:`status` to run on the caller (a test's ledger,
        run step by step)
    :ptype thread: bool
    """

    def __init__(self, budget: int | None = None, *, thread: bool = True) -> None:
        self._threaded = thread
        self._lock = threading.Lock()
        self._budget = budget
        #: per (store, table), replaced whole by the ledger's thread alone
        self._tables: Mapping[tuple[int, str], _Table] = MappingProxyType({})
        #: the joined stores by ``id``, held weakly (a dropped store leaves with it); replaced whole under the lock
        self._stores: FrozenWeakValues[int, Any] = FrozenWeakValues()
        #: the exact bytes of every table's held rows, and the bytes their hand-offs' writes name; the ledger's thread's
        self._estimated = 0
        self._unsettled = 0
        #: the ledger's order of use, given each hand-off row as the thread takes it
        self._seq = 0
        #: measured real bytes per estimated byte of the EVICTABLE tables
        self._scale = 1.0
        self._kept = 0
        self._overhead = 0
        self._sizing_failed = False
        self._handoffs: queue.SimpleQueue[_HandOff] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        #: per (store, table), replaced whole
        self._counters: Mapping[tuple[int, str], TableCounters] = MappingProxyType({})

    # --- joining ---

    def join(self, store: Any, directory: Path) -> None:
        """a backend joins the process's budget; the first starts the ledger's thread.

        :param store: the backend
        :ptype store: Any
        :param directory: where its file is, for a budget from the file system
        :ptype directory: Path
        :return: nothing
        :rtype: None
        """
        with self._lock:
            if self._budget is None:
                self._budget = _budget_from_env(directory)
            BUDGET_BYTES.labels(tier=_TIER, name="").set(self._budget)
            self._stores = self._stores.with_entry(id(store), store)
            if self._threaded and (self._thread is None or not self._thread.is_alive()):
                self._thread = threading.Thread(target=self._run, name="threetears-l1-ledger", daemon=True)
                self._thread.start()

    def leave(self, store: Any) -> None:
        """a backend resets: its recency goes.

        :param store: the backend
        :ptype store: Any
        :return: nothing
        :rtype: None
        """
        with self._lock:
            self._stores = self._stores.without_key(id(store))
        self._handoffs.put(("l", id(store)))

    @property
    def budget(self) -> int:
        """the process's budget, in bytes (0 before any store joined).

        :return: bytes
        :rtype: int
        """
        return self._budget or 0

    def set_budget(self, budget: int) -> None:
        """set the budget (a test's, or a resize).

        :param budget: bytes
        :ptype budget: int
        :return: nothing
        :rtype: None
        """
        with self._lock:
            self._budget = budget
        BUDGET_BYTES.labels(tier=_TIER, name="").set(budget)

    def counters(self, store: Any, table: str) -> TableCounters:
        """a table's hit and miss counters, bound once.

        :param store: the backend
        :ptype store: Any
        :param table: the table
        :ptype table: str
        :return: the counters
        :rtype: TableCounters
        """
        key = (id(store), table)
        bound = self._counters.get(key)
        if bound is None:
            # two threads binding one table at once each assign a snapshot; the one lost is bound again on its next
            # read, to the same labelled counters, so nothing is counted wrong
            bound = TableCounters(
                hits=HITS.labels(tier=_TIER, name=table), misses=MISSES.labels(tier=_TIER, name=table)
            )
            self._counters = MappingProxyType({**self._counters, key: bound})
        return bound

    # --- the hot path: each hands the ledger's thread one immutable hand-off, and touches nothing else ---

    def touched(self, store: Any, table: str, key: Hashable) -> None:
        """a read hit ``key``: it is now the most recent.

        :param store: the backend
        :ptype store: Any
        :param table: an ``EVICTABLE`` table
        :ptype table: str
        :param key: the serialized key
        :ptype key: Hashable
        :return: nothing
        :rtype: None
        """
        self._handoffs.put(("t", store, table, _key_text(key), time.monotonic()))  # type: ignore[arg-type]

    def written(self, store: Any, table: str, sized: Sequence[tuple[Hashable, int]]) -> None:
        """rows of an ``EVICTABLE`` table were written; the ledger's thread evicts past the budget.

        :param store: the backend
        :ptype store: Any
        :param table: the table
        :ptype table: str
        :param sized: each row's serialized key and estimated bytes
        :ptype sized: Sequence[tuple[Hashable, int]]
        :return: nothing
        :rtype: None
        """
        if not sized:
            return
        batch = packed([_key_text(key) for key, _size in sized])  # type: ignore[arg-type]
        sizes = array("q", [size for _key, size in sized])
        self._handoffs.put(("w", store, table, batch, sizes, time.monotonic()))

    def removed(self, store: Any, table: str, keys: Sequence[Hashable]) -> None:
        """rows left the table (a delete, an invalidation): they leave its recency.

        :param store: the backend
        :ptype store: Any
        :param table: the table
        :ptype table: str
        :param keys: the serialized keys
        :ptype keys: Sequence[Hashable]
        :return: nothing
        :rtype: None
        """
        if keys:
            self._handoffs.put(("r", store, table, packed([_key_text(key) for key in keys])))  # type: ignore[arg-type]

    def emptied(self, store: Any, table: str) -> None:
        """every row of the table left it.

        :param store: the backend
        :ptype store: Any
        :param table: the table
        :ptype table: str
        :return: nothing
        :rtype: None
        """
        self._handoffs.put(("e", store, table))

    # --- the ledger's thread: hand-offs in, recency folded, evictions out ---

    def _settle(self, first: _HandOff | None = None) -> None:
        """take the hand-offs queued now (``first`` already taken), in order, into the tables' recency; evict past the
        budget as a write crosses it. The ledger's thread (or a step-by-step ledger's caller) alone.

        :param first: a hand-off already taken from the queue
        :ptype first: _HandOff | None
        :return: nothing
        :rtype: None
        """
        # function-local: each table's rows handed off in this pass, as columns, until they join its hand-offs
        local: dict[tuple[int, str], tuple[list[str], array[int], array[float], array[int]]] = {}
        written: dict[tuple[int, str], int] = {}
        count = self._handoffs.qsize()  # those queued now: a busy reader never keeps a drain from ending
        taken = 0
        while True:
            if first is not None:
                handoff, first = first, None
            elif taken < count:
                try:
                    handoff = self._handoffs.get_nowait()
                except queue.Empty:  # NOSILENT: the queue ran dry before its count; the pass is done
                    break
                taken += 1
            else:
                break
            op = handoff[0]
            if op in ("e", "l", "s"):
                self._join_local(local, written)
                local, written = {}, {}
                if op == "s":
                    if handoff[2]:
                        self._fold_all()
                    handoff[1].set()
                else:
                    self._drop(handoff)
                continue
            key = (id(handoff[1]), handoff[2])
            if key not in self._tables:
                if op != "w":
                    continue  # a read or delete of a table the ledger holds nothing of
                self._tables = MappingProxyType(
                    {**self._tables, key: _Table(store=weakref.ref(handoff[1]), name=handoff[2])}
                )
            columns = local.get(key)
            if columns is None:
                columns = local[key] = ([], array("q"), array("d"), array("q"))
            if op == "t":
                columns[0].append(handoff[3])
                columns[1].append(_TOUCH)
                columns[2].append(handoff[4])
                columns[3].append(self._seq)
                self._seq += 1
            elif op == "r":
                names = texts(handoff[3])
                columns[0].extend(names)
                columns[1].extend([_REMOVED] * len(names))
                columns[2].extend([0.0] * len(names))
                columns[3].extend(range(self._seq, self._seq + len(names)))
                self._seq += len(names)
            else:
                names, sizes, tick = texts(handoff[3]), handoff[4], handoff[5]
                for start in range(0, len(names), _SLICE_ROWS):
                    part = sizes[start : start + _SLICE_ROWS]
                    columns[0].extend(names[start : start + _SLICE_ROWS])
                    columns[1].extend(part)
                    columns[2].extend([tick] * len(part))
                    columns[3].extend(range(self._seq, self._seq + len(part)))
                    self._seq += len(part)
                    added = sum(part)
                    written[key] = written.get(key, 0) + added
                    self._unsettled += added
                    if self._over_bound():
                        self._join_local(local, written)
                        self._evict_over()
                        local, written = {}, {}
                        columns = local[key] = ([], array("q"), array("d"), array("q"))
        self._join_local(local, written)

    def _join_local(
        self,
        local: Mapping[tuple[int, str], tuple[list[str], array[int], array[float], array[int]]],
        written: Mapping[tuple[int, str], int],
    ) -> None:
        """a pass's rows join their tables' hand-offs, one batch a table; a table past its bound folds.

        :param local: each table's rows of the pass, as columns
        :ptype local: Mapping
        :param written: each table's bytes its writes of the pass name
        :ptype written: Mapping[tuple[int, str], int]
        :return: nothing
        :rtype: None
        """
        if not local:
            return
        tables = dict(self._tables)  # function-local: the next snapshot, assigned whole below
        for key, (names, sizes, ticks, seqs) in local.items():
            table = tables.get(key)
            if table is None or not names:
                continue
            batch = _Rows(packed(names), sizes, ticks, seqs)
            pending = (*table.pending, batch)
            if len(pending) > _HANDOFF_BATCHES:
                pending = (_joined(pending),)
            table = replace(
                table,
                pending=pending,
                pending_rows=table.pending_rows + len(batch),
                unsettled=table.unsettled + written.get(key, 0),
            )
            if table.pending_rows > max(_FOLD_ROWS, len(table.held) // 4):
                table = self._fold(table)
            tables[key] = table
        self._tables = MappingProxyType(tables)

    def _fold(self, table: _Table) -> _Table:
        """``table`` folded (:func:`_folded`), with the ledger's tallies moved to match.

        :param table: the table
        :ptype table: _Table
        :return: it, folded
        :rtype: _Table
        """
        done = _folded(table)
        self._estimated += done.estimated - table.estimated
        self._unsettled -= table.unsettled
        return done

    def _fold_all(self) -> None:
        """fold every table: the ledger's bytes are then exact."""
        self._tables = MappingProxyType({key: self._fold(table) for key, table in self._tables.items()})

    def _drop(self, handoff: _HandOff) -> None:
        """a table emptied (``e``) or a store left (``l``): their recency goes.

        :param handoff: the hand-off
        :ptype handoff: _HandOff
        :return: nothing
        :rtype: None
        """
        if handoff[0] == "e":
            gone = {(id(handoff[1]), handoff[2])}
        else:
            gone = {key for key in self._tables if key[0] == handoff[1]}
        for key in gone:
            table = self._tables.get(key)
            if table is not None:
                self._estimated -= table.estimated
                self._unsettled -= table.unsettled
        self._tables = MappingProxyType({key: table for key, table in self._tables.items() if key not in gone})

    # --- the budget ---

    def _allowance(self) -> float:
        """what is left for ``EVICTABLE`` rows: the budget less what ``KEEP`` holds and the files' overhead."""
        return (self._budget or 0) - self._kept - self._overhead

    def _over_bound(self) -> bool:
        """whether the upper bound of the rows' bytes passes the allowance (then a fold says whether they do)."""
        return self._budget is not None and (self._estimated + self._unsettled) * self._scale > self._allowance()

    def _evict_over(self) -> None:
        """fold every table, and evict the least recent rows past the allowance, if its exact bytes pass it."""
        self._fold_all()
        victims = self._over_budget()
        if victims:
            self._evict(victims)

    def _over_budget(self) -> list[tuple[_Table, list[str], int, float]]:
        """pick the least recent rows past the allowance across the (folded) tables, in order of use, taking them out
        of the recency.

        :return: each table's victims: the table, their key texts, their estimated bytes, the last one's last use
        :rtype: list[tuple[_Table, list[str], int, float]]
        """
        if self._budget is None:
            return []
        allowance = self._allowance()
        if self._estimated * self._scale <= allowance:
            return []
        target = max(0.0, allowance * _EVICT_TO) / self._scale
        keys = list(self._tables)  # function-local
        rows = [self._tables[key].held for key in keys]
        fronts = [0] * len(keys)
        estimated = self._estimated
        last = (-1, 0.0)
        while estimated > target:
            oldest = -1
            for at, held in enumerate(rows):
                if fronts[at] < len(held) and (oldest < 0 or held.seqs[fronts[at]] < rows[oldest].seqs[fronts[oldest]]):
                    oldest = at
            if oldest < 0:
                break
            estimated -= rows[oldest].sizes[fronts[oldest]]
            last = (oldest, rows[oldest].ticks[fronts[oldest]])
            fronts[oldest] += 1
        victims: list[tuple[_Table, list[str], int, float]] = []
        tables = dict(self._tables)  # function-local: the next snapshot, assigned whole below
        for at, key in enumerate(keys):
            taken = fronts[at]
            if not taken:
                continue
            table, held = tables[key], rows[at]
            names = texts(held.texts)
            freed = sum(held.sizes[:taken])
            tables[key] = replace(
                table,
                held=_Rows(packed(names[taken:]), held.sizes[taken:], held.ticks[taken:], held.seqs[taken:]),
                estimated=table.estimated - freed,
            )
            victims.append((table, names[:taken], freed, last[1] if last[0] == at else held.ticks[taken - 1]))
        self._tables = MappingProxyType(tables)
        self._estimated = estimated
        if last[0] >= 0:
            # the newest victim's last use, as the age metric reads it: moved to the end
            victims.sort(key=lambda victim: victim[3])
        return victims

    def _evict(self, victims: list[tuple[_Table, list[str], int, float]]) -> None:
        """delete the victims' rows, a table at a time, on the ledger's thread; count them.

        :param victims: from :meth:`_over_budget`
        :ptype victims: list[tuple[_Table, list[str], int, float]]
        :return: nothing
        :rtype: None
        """
        for table, names, size, _tick in victims:
            store = table.store()
            if store is None:
                continue
            try:
                store.evict_keys(table.name, [_key_of(text) for text in names])
            except Exception as exc:  # noqa: BLE001  # NOSILENT: a failed eviction leaves rows cached, a miss later at worst; said
                contain(exc, log=log, what=f"L1 eviction of {table.name} failed; its rows stay cached")
                continue
            EVICTIONS.labels(tier=_TIER, name=table.name).inc(len(names))
            EVICTED_BYTES.labels(tier=_TIER, name=table.name).inc(round(size * self._scale))
        if victims:
            EVICTED_AGE_SECONDS.labels(tier=_TIER, name="").set(time.monotonic() - victims[-1][3])

    # --- the thread ---

    def _run(self) -> None:
        """the ledger's thread: take hand-offs as they come, and measure every :data:`_MEASURE_SECONDS`."""
        next_measure = time.monotonic() + _MEASURE_SECONDS
        while True:
            try:
                first: _HandOff | None = self._handoffs.get(timeout=max(0.0, next_measure - time.monotonic()))
            except queue.Empty:
                first = None
            try:
                if first is not None:
                    self._settle(first)
                if time.monotonic() >= next_measure:
                    next_measure = time.monotonic() + _MEASURE_SECONDS
                    self.measure()
            except Exception as exc:  # noqa: BLE001  # NOSILENT: the thread outlives one bad pass; said
                contain(exc, log=log, what="L1 ledger pass failed; the next pass tries again")

    def measure(self) -> dict[str, int]:
        """take the queued hand-offs, measure every store's tables, set the gauges, fold and rescale the estimate, and
        evict if over; blocking, on the ledger's thread (or a step-by-step ledger's caller).

        :return: the process's held bytes by policy (``keep``, ``evictable``) and its files' ``overhead``
        :rtype: dict[str, int]
        """
        self._settle()
        kept = evictable = overhead = 0
        for store in list(self._stores.values()):
            policies = store.table_policies()
            measured = store.table_bytes()
            for table, size in measured.items():
                declared = policies.get(table)
                if declared is None:
                    overhead += size  # SQLite's own schema pages, a raw table's shadow tables
                    continue
                if declared.policy is Policy.KEEP:
                    kept += size
                else:
                    evictable += size
            overhead += store.side_bytes()
        self._fold_all()
        self._kept, self._overhead = kept, overhead
        if self._estimated > 0 and evictable > 0:
            self._scale = evictable / self._estimated
        failed = self._budget is not None and kept > self._budget
        victims = self._over_budget()
        if failed and not self._sizing_failed:
            SIZING_FAILURES.labels(tier=_TIER, name="").inc()
            log.error(
                "L1 sizing failure: KEEP tables alone pass the budget; nothing KEEP is evicted, the budget is wrong",
                extra={"extra_data": {"kept_bytes": kept, "budget_bytes": self._budget, "env": _L1_BUDGET_ENV}},
            )
        self._sizing_failed = failed
        if victims:
            self._evict(victims)
        return {"keep": kept, "evictable": evictable, "overhead": overhead}

    def status(self) -> dict[str, int]:
        """the tallies, for autosize (:class:`threetears.core.shm_autosize.ShmAutosize`): never the file system.

        ``evictable_bytes`` is the ledger's upper bound: exact after a fold (every measure folds), above it by the
        bytes of rows rewritten or deleted since.

        :return: ``kept_bytes`` (``KEEP``, last measured), ``max_bytes`` (the budget) and ``evictable_bytes``
        :rtype: dict[str, int]
        """
        if not self._threaded:
            self._settle()
        return {
            "kept_bytes": self._kept,
            "max_bytes": self._budget or 0,
            "evictable_bytes": round((self._estimated + self._unsettled) * self._scale),
        }

    def held(self, store: Any, table: str) -> tuple[tuple[Any, ...], ...]:
        """the keys the ledger holds of a table, least recently used first, once every hand-off queued now is taken and
        every table folded (its bytes, :meth:`status`, are then exact); tests and diagnostics.

        :param store: the backend
        :ptype store: Any
        :param table: the table
        :ptype table: str
        :return: the keys, serialized
        :rtype: tuple[tuple[Any, ...], ...]
        """
        self.drain(fold=True)
        held = self._tables.get((id(store), table))
        return () if held is None else tuple(_key_of(text) for text in texts(held.held.texts))

    def drain(self, timeout: float = DEFAULT_L1_LEDGER_DRAIN_TIMEOUT_SECONDS, *, fold: bool = False) -> None:
        """every hand-off queued now taken, evicting as they pass the budget: on the calling thread for a step-by-step
        ledger, else waiting for the ledger's thread to reach them (tests, and a caller that must see them done).

        :param timeout: seconds to wait for the ledger's thread
        :ptype timeout: float
        :param fold: fold every table too, so the ledger's bytes are exact
        :ptype fold: bool
        :return: nothing
        :rtype: None
        :raises TimeoutError: when the ledger's thread did not reach them in ``timeout``
        """
        if not self._threaded:
            self._settle()
            if fold:
                self._fold_all()
            return
        reached = threading.Event()
        self._handoffs.put(("s", reached, fold))
        if not reached.wait(timeout):
            raise TimeoutError(f"the L1 ledger's thread took no hand-offs in {timeout}s")


#: the process's ledger, built once by the first :func:`ledger`, however many threads ask first
_LEDGER: BuildOnce[None, L1Ledger] = BuildOnce()


def ledger() -> L1Ledger:
    """the process's ledger.

    :return: it
    :rtype: L1Ledger
    """
    return _LEDGER.get(None, L1Ledger)
