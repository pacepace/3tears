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

**Recency, cheaply.** No write per read: each ``EVICTABLE`` table keeps an in-memory ``OrderedDict`` of its keys
(key -> [estimated bytes, tick]); a hit moves its key to the end under one lock. Eviction takes the least recent head
across the tables (a handful of tables, so a scan of heads). ``KEEP`` tables keep nothing in memory.

**Bytes, estimated then measured.** A row's bytes are estimated on write from its serialized values; a thread
measures each table's real pages (``dbstat``) every :data:`MEASURE_SECONDS` and scales the estimate by what it
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

import enum
import os
import queue
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

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
    declare,
    declared,
)
from threetears.observe import contain, get_logger

__all__ = [
    "L1_EVICTION_INFO",
    "L1_BUDGET_ENV",
    "MEASURE_SECONDS",
    "L1Ledger",
    "TableCounters",
    "budget_from_env",
    "declare_l1",
    "declare_l1_table",
    "ledger",
    "row_bytes",
]

log = get_logger(__name__)

#: the key of a SQLAlchemy table's ``info`` that holds its declared L1 policy (:func:`declare_l1`)
L1_EVICTION_INFO: Final = "l1_eviction"
#: the process's L1 byte budget, as the deploy config sets it from SIZING.md
L1_BUDGET_ENV: Final = "THREETEARS_L1_BUDGET_BYTES"
#: how often the thread measures the real pages of every table: a ``dbstat`` walk of the file, off the loop
MEASURE_SECONDS: Final = 30.0
#: evict down to this share of what is left for ``EVICTABLE`` rows, so an eviction runs once per stretch of
#: writes, not once per write
_EVICT_TO: Final = 0.9
#: an estimate's fixed cost per row (SQLite's record header, cell pointer and the key's index entry, roughly) until
#: the first measure scales it
_ROW_OVERHEAD: Final = 48
_TIER: Final = Tier.L1.value
#: rows a write records under one hold of the ledger's lock, so a big batch never holds it long
_LOCK_ROWS: Final = 256
#: advanced by every L1 declaration, so a backend that read its tables' policies sees a table become KEEP
POLICY_EPOCH: list[int] = [0]


def declare_l1[T](table: T, policy: Policy, why: str) -> T:
    """declare a SQLAlchemy table's L1 policy where the table is declared; the SQLite L1 refuses a table without one.

    :param table: the ``sqlalchemy.Table``
    :ptype table: T
    :param policy: ``EVICTABLE`` (a cache of reads; a miss refills from L2/L3) or ``KEEP`` (a reader treats it as
        complete, or it holds state a write or settle needs)
    :ptype policy: Policy
    :param why: the one-line reason
    :ptype why: str
    :return: the table
    :rtype: T
    """
    policy_and_why = Declared(policy, why)
    table.info[L1_EVICTION_INFO] = policy_and_why  # type: ignore[attr-defined]
    declare_l1_table(table.name, policy_and_why)  # type: ignore[attr-defined]
    return table


def declare_l1_table(name: str, policy: Declared) -> None:
    """declare one L1 table's policy in the cache family's declarations (:func:`threetears.core.eviction.declare`,
    ``Tier.L1``, the kind named by the table), where every reader of a table's policy finds it
    (:func:`~threetears.core.eviction.policy_of`): the ledger, and the held-bytes sampler's ``policy`` label.

    A table's name is its kind: one table, one policy, in every L1 of the process. ``TableSchema(l1_eviction=...)``
    and :func:`declare_l1` call this where the table is declared; ``SQLiteBackend.initialize`` for a table that
    carries its policy in ``Table.info``.

    :param name: the table
    :ptype name: str
    :param policy: its policy and why
    :ptype policy: Declared
    :return: nothing
    :rtype: None
    Two declarations of one name that disagree (two tables of one name in one process, or one table two users
    read differently) resolve to ``KEEP``, said once in the log: one of its users needs it complete, and keeping a
    cache costs only bytes where evicting what a user needs complete answers wrongly.

    :return: nothing
    :rtype: None
    """
    before = declared(Tier.L1).get(name)
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
    declare(Tier.L1, kinds, {kinds["TABLE"]: policy})
    POLICY_EPOCH[0] += 1


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


def budget_from_env(directory: Path) -> int:
    """the process's budget: ``THREETEARS_L1_BUDGET_BYTES``, else half of ``directory``'s file system.

    :param directory: where the L1 files are
    :ptype directory: Path
    :return: bytes
    :rtype: int
    :raises ValueError: when the variable is set and not a positive integer
    """
    named = os.environ.get(L1_BUDGET_ENV)
    if named:
        budget = int(named)
        if budget < 1:
            raise ValueError(f"{L1_BUDGET_ENV} must be a positive number of bytes, got {named!r}")
        return budget
    stat = os.statvfs(directory)
    budget = stat.f_blocks * stat.f_frsize // 2
    log.info(
        "L1 budget not set; half the L1 directory's file system",
        extra={"extra_data": {"env": L1_BUDGET_ENV, "directory": str(directory), "budget_bytes": budget}},
    )
    return budget


class _Store(Protocol):
    """what the ledger needs of a backend: delete keys, and measure its tables."""

    store_name: str

    def evict_keys(self, table: str, keys: Sequence[tuple[Any, ...]]) -> None: ...

    def table_bytes(self) -> Mapping[str, int]: ...

    def table_policies(self) -> Mapping[str, Declared]: ...

    def side_bytes(self) -> int: ...


@dataclass
class TableCounters:
    """a table's bound hit and miss counters, made once so a read pays no label lookup."""

    hits: Any
    misses: Any


@dataclass
class _Table:
    """one ``EVICTABLE`` table's recency: key -> [estimated bytes, tick], least recent first."""

    store: weakref.ReferenceType[Any]
    name: str
    order: OrderedDict[Hashable, list[int | float]] = field(default_factory=OrderedDict)
    estimated: int = 0


class L1Ledger:
    """the process's L1 byte budget and the recency of every ``EVICTABLE`` table in it (one per process).

    :param budget: bytes; ``None`` reads it from the environment when the first store joins
    :ptype budget: int | None
    :param thread: whether evictions and measures run on the ledger's own thread; ``False`` leaves them queued for
        :meth:`drain` and :meth:`measure` (a test's ledger, run step by step)
    :ptype thread: bool
    """

    def __init__(self, budget: int | None = None, *, thread: bool = True) -> None:
        self._threaded = thread
        self._lock = threading.Lock()
        self._budget = budget
        self._tables: dict[tuple[int, str], _Table] = {}
        self._stores: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()
        self._estimated = 0
        #: measured real bytes per estimated byte of the EVICTABLE tables
        self._scale = 1.0
        self._kept = 0
        self._overhead = 0
        self._sizing_failed = False
        self._jobs: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._counters: dict[tuple[int, str], TableCounters] = {}

    # --- joining ---

    def join(self, store: Any, directory: Path) -> None:
        """a backend joins the process's budget; the first starts the measuring thread.

        :param store: the backend
        :ptype store: Any
        :param directory: where its file is, for a budget from the file system
        :ptype directory: Path
        :return: nothing
        :rtype: None
        """
        with self._lock:
            if self._budget is None:
                self._budget = budget_from_env(directory)
            BUDGET_BYTES.labels(tier=_TIER, name="").set(self._budget)
            self._stores[id(store)] = store
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
            self._stores.pop(id(store), None)
            for key in [key for key in self._tables if key[0] == id(store)]:
                self._estimated -= self._tables.pop(key).estimated
            for key in [key for key in self._counters if key[0] == id(store)]:
                del self._counters[key]

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
            bound = self._counters[key] = TableCounters(
                hits=HITS.labels(tier=_TIER, name=table), misses=MISSES.labels(tier=_TIER, name=table)
            )
        return bound

    # --- the hot path ---

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
        held = self._tables.get((id(store), table))
        # a read never waits for the ledger: while a writer's batch or an eviction holds it, this touch is skipped
        # and the key keeps its older place (recency is a heuristic; an eviction is a miss, never a wrong answer)
        if held is None or not self._lock.acquire(blocking=False):
            return
        try:
            entry = held.order.get(key)
            if entry is not None:
                entry[1] = time.monotonic()
                held.order.move_to_end(key)
        finally:
            self._lock.release()

    def written(self, store: Any, table: str, sized: Sequence[tuple[Hashable, int]]) -> None:
        """rows of an ``EVICTABLE`` table were written; evict past the budget.

        :param store: the backend
        :ptype store: Any
        :param table: the table
        :ptype table: str
        :param sized: each row's serialized key and estimated bytes
        :ptype sized: Sequence[tuple[Hashable, int]]
        :return: nothing
        :rtype: None
        """
        now = time.monotonic()
        for start in range(0, len(sized), _LOCK_ROWS):
            self._written(store, table, sized[start : start + _LOCK_ROWS], now)

    def _written(self, store: Any, table: str, sized: Sequence[tuple[Hashable, int]], now: float) -> None:
        """:meth:`written` for one slice of rows, under the lock once.

        :param store: the backend
        :ptype store: Any
        :param table: the table
        :ptype table: str
        :param sized: the slice's keys and estimated bytes
        :ptype sized: Sequence[tuple[Hashable, int]]
        :param now: the monotonic clock at the write
        :ptype now: float
        :return: nothing
        :rtype: None
        """
        with self._lock:
            held = self._tables.get((id(store), table))
            if held is None:
                held = self._tables[(id(store), table)] = _Table(store=weakref.ref(store), name=table)
            for key, size in sized:
                entry = held.order.get(key)
                if entry is None:
                    held.order[key] = [size, now]
                    delta = size
                else:
                    delta = size - int(entry[0])
                    entry[0], entry[1] = size, now
                    held.order.move_to_end(key)
                held.estimated += delta
                self._estimated += delta
            victims = self._over_budget(now)
        if victims:
            self._jobs.put(lambda: self._evict(victims))

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
        held = self._tables.get((id(store), table))
        if held is None:
            return
        with self._lock:
            for key in keys:
                entry = held.order.pop(key, None)
                if entry is not None:
                    held.estimated -= int(entry[0])
                    self._estimated -= int(entry[0])

    def emptied(self, store: Any, table: str) -> None:
        """every row of the table left it.

        :param store: the backend
        :ptype store: Any
        :param table: the table
        :ptype table: str
        :return: nothing
        :rtype: None
        """
        with self._lock:
            held = self._tables.pop((id(store), table), None)
            if held is not None:
                self._estimated -= held.estimated

    # --- the budget ---

    def _allowance(self) -> float:
        """what is left for ``EVICTABLE`` rows: the budget less what ``KEEP`` holds and the files' overhead."""
        return (self._budget or 0) - self._kept - self._overhead

    def _over_budget(self, now: float) -> list[tuple[_Table, Hashable, int, float]]:
        """pick the least recent rows past the allowance, taking them out of the recency; under the lock.

        :param now: the monotonic clock now
        :ptype now: float
        :return: the victims: table, key, estimated bytes, last use
        :rtype: list[tuple[_Table, Hashable, int, float]]
        """
        if self._budget is None:
            return []
        allowance = self._allowance()
        if self._estimated * self._scale <= allowance:
            return []
        target = max(0.0, allowance * _EVICT_TO) / self._scale
        victims: list[tuple[_Table, Hashable, int, float]] = []
        while self._estimated > target:
            oldest: _Table | None = None
            oldest_tick = now + 1
            for held in self._tables.values():
                if held.order:
                    head = next(iter(held.order.values()))
                    if head[1] < oldest_tick:
                        oldest, oldest_tick = held, float(head[1])
            if oldest is None:
                break
            key, (size, tick) = oldest.order.popitem(last=False)
            oldest.estimated -= int(size)
            self._estimated -= int(size)
            victims.append((oldest, key, int(size), float(tick)))
        return victims

    def _evict(self, victims: list[tuple[_Table, Hashable, int, float]]) -> None:
        """delete the victims' rows, a table at a time, on the ledger's thread; count them.

        :param victims: from :meth:`_over_budget`
        :ptype victims: list[tuple[_Table, Hashable, int, float]]
        :return: nothing
        :rtype: None
        """
        by_table: dict[int, tuple[_Table, list[Any]]] = {}
        for held, key, _size, _tick in victims:
            by_table.setdefault(id(held), (held, []))[1].append(key)
        for held, keys in by_table.values():
            store = held.store()
            if store is None:
                continue
            try:
                store.evict_keys(held.name, keys)
            except Exception as exc:  # noqa: BLE001  # NOSILENT: a failed eviction leaves rows cached, a miss later at worst; said
                contain(exc, log=log, what=f"L1 eviction of {held.name} failed; its rows stay cached")
                continue
            freed = round(sum(size for victim, _k, size, _t in victims if victim is held) * self._scale)
            EVICTIONS.labels(tier=_TIER, name=held.name).inc(len(keys))
            EVICTED_BYTES.labels(tier=_TIER, name=held.name).inc(freed)
        if victims:
            EVICTED_AGE_SECONDS.labels(tier=_TIER, name="").set(time.monotonic() - victims[-1][3])

    # --- the thread ---

    def _run(self) -> None:
        """the ledger's thread: run evictions as they come, and measure every :data:`MEASURE_SECONDS`."""
        next_measure = time.monotonic() + MEASURE_SECONDS
        while True:
            try:
                job = self._jobs.get(timeout=max(0.0, next_measure - time.monotonic()))
            except queue.Empty:
                job = None
            try:
                if job is not None:
                    job()
                if time.monotonic() >= next_measure:
                    next_measure = time.monotonic() + MEASURE_SECONDS
                    self.measure()
            except Exception as exc:  # noqa: BLE001  # NOSILENT: the thread outlives one bad pass; said
                contain(exc, log=log, what="L1 ledger pass failed; the next pass tries again")

    def measure(self) -> dict[str, int]:
        """measure every store's tables, set the gauges, rescale the estimate, and evict if over; blocking.

        :return: the process's held bytes by policy (``keep``, ``evictable``) and its files' ``overhead``
        :rtype: dict[str, int]
        """
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
        with self._lock:
            self._kept, self._overhead = kept, overhead
            if self._estimated > 0 and evictable > 0:
                self._scale = evictable / self._estimated
            failed = self._budget is not None and kept > self._budget
            victims = self._over_budget(time.monotonic())
        if failed and not self._sizing_failed:
            SIZING_FAILURES.labels(tier=_TIER, name="").inc()
            log.error(
                "L1 sizing failure: KEEP tables alone pass the budget; nothing KEEP is evicted, the budget is wrong",
                extra={"extra_data": {"kept_bytes": kept, "budget_bytes": self._budget, "env": L1_BUDGET_ENV}},
            )
        self._sizing_failed = failed
        if victims:
            self._evict(victims)
        return {"keep": kept, "evictable": evictable, "overhead": overhead}

    def status(self) -> dict[str, int]:
        """the tallies, for autosize (:func:`threetears.core.shm_autosize.measure`): never the file system.

        :return: ``kept_bytes`` (``KEEP``, last measured), ``max_bytes`` (the budget) and ``evictable_bytes``
        :rtype: dict[str, int]
        """
        return {
            "kept_bytes": self._kept,
            "max_bytes": self._budget or 0,
            "evictable_bytes": round(self._estimated * self._scale),
        }

    def drain(self) -> None:
        """run every queued eviction now, on the calling thread (tests, and a measure that must see them done)."""
        while True:
            try:
                job = self._jobs.get_nowait()
            except queue.Empty:
                return
            job()


_LEDGER: list[L1Ledger] = []
_LEDGER_LOCK = threading.Lock()


def ledger() -> L1Ledger:
    """the process's ledger.

    :return: it
    :rtype: L1Ledger
    """
    if not _LEDGER:
        with _LEDGER_LOCK:
            if not _LEDGER:
                _LEDGER.append(L1Ledger())
    return _LEDGER[0]
