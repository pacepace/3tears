"""How big each L1 table is: rows and bytes per table, measured from SQLite's own pages, as gauges.

**Why.** An L1 table grows with the rows read through it (``collections/base.py``), and nothing said how big one gets
in a pod: the SQLite audit (SQ1, 2026-11-15) could not say how long a table-wide drop holds the event loop in prod
(22 ms at 10k rows, 276 ms at 100k) without the size. So every process measures its L1s and exports the one cache
family's gauges (:mod:`threetears.core.eviction`): ``threetears_cache_held_entries`` (rows) and
``threetears_cache_held_bytes`` (bytes), ``tier="l1"``, ``name`` the table, ``policy`` its declared eviction policy.
Metrics only: alerting, if any, is configured in Grafana from them.

**What it measures.** One walk of SQLite's ``dbstat`` table of pages per database: a table's bytes are the pages of
its own b-tree and of every index on it; its rows are the cells of its b-tree's leaf pages (exactly ``count(*)``,
held by ``tests/unit/cache/test_l1_sizes.py``), so one pass answers both for every table. This measurement owns the
two gauges for the L1 (coordinator 2026-11-15); the L1's own eviction keeps its own accounting for its budget and
never sets them.

**Off the loop, every minute.** A daemon thread (``l1-sizes``) samples every :data:`_INTERVAL_SECONDS`, on its own
thread's connection: never the event loop. One sample of a 38 MiB database (100k rows of a 300-byte table, an
index, a second table) took 12.8 ms median, 15.7 ms max; 10k rows 1.6 ms (macOS, CPython 3.14.3, SQLite 3.50.4,
SQ2 2026-11-15). While the in-memory (memdb) L1 holds its whole-database lock, a write on another thread waits out
that walk once a minute (the audit's item 5); a WAL L1 lets the write through.

**Which tables.** Every table of a watched L1 is measured; a table whose eviction policy is not declared
(:func:`threetears.core.eviction.declared` for :attr:`~threetears.core.eviction.Tier.L1`) is not exported, since
the family has no "no policy" label, and is logged once.
"""

from __future__ import annotations

import sqlite3
import threading
import weakref
from collections.abc import Mapping
from typing import Final

from threetears.core.cache.l1_pages import TableSize, measure
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.eviction import HELD_BYTES, HELD_ENTRIES, Tier, declared
from threetears.observe import contain, get_logger

__all__ = ["L1_SIZES", "measure"]

log = get_logger(__name__)

#: how often a process measures its L1s: a size moves with loads, minutes apart; a sample of a 100k-row L1 costs
#: about 13 ms of a thread (module docstring)
_INTERVAL_SECONDS: Final = 60.0


class _L1Sizes:
    """the process's L1 size gauges: the L1s it watches, measured every ``interval`` on a thread of its own.

    :param interval: seconds between samples
    :ptype interval: float
    """

    def __init__(self, interval: float = _INTERVAL_SECONDS) -> None:
        self._interval = interval
        self._backends: weakref.WeakSet[SQLiteBackend] = weakref.WeakSet()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        #: the label sets set by the last sample, so a table gone since reads 0 rather than its last size
        self._exported: set[tuple[str, str]] = set()
        #: the tables already logged as undeclared, each logged once
        self._undeclared: set[str] = set()

    def watch(self, backend: object) -> None:
        """measure an L1 from now on (held weakly: a dropped backend is no longer measured), starting the sampling
        thread on the first. Anything but a SQLite L1 is not measured here.

        :param backend: the L1 backend
        :ptype backend: object
        :return: nothing
        :rtype: None
        """
        if not isinstance(backend, SQLiteBackend):
            return
        with self._lock:
            self._backends.add(backend)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="l1-sizes", daemon=True)
                self._thread.start()

    def stop(self) -> None:
        """stop sampling (a test's, or a process's shutdown).

        :return: nothing
        :rtype: None
        """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self._interval)

    def sample_once(self) -> dict[str, TableSize]:
        """measure every watched L1 now and set the gauges: a table's sizes summed over the L1s holding one of its
        name.

        :return: table -> its size, every table measured (declared or not)
        :rtype: dict[str, TableSize]
        """
        with self._lock:
            backends = list(self._backends)
        sizes: dict[str, TableSize] = {}
        for backend in backends:
            for table, size in self._measured(backend).items():
                was = sizes.get(table, TableSize(rows=0, bytes=0))
                sizes[table] = TableSize(rows=was.rows + size.rows, bytes=was.bytes + size.bytes)
        self._export(sizes)
        return sizes

    def _measured(self, backend: SQLiteBackend) -> Mapping[str, TableSize]:
        """one L1's sizes; none for one not (or no longer) initialized, or one that fails mid-measure.

        :param backend: the L1
        :ptype backend: SQLiteBackend
        :return: table -> its size
        :rtype: Mapping[str, TableSize]
        """
        found: Mapping[str, TableSize] = {}
        if backend.is_initialized():
            try:
                found = measure(backend.get_connection())
            except (sqlite3.Error, RuntimeError) as exc:
                # NOSILENT: a backend reset while it is measured (closed connection, uninitialized) or a SQLite
                # without dbstat: this sample has no size for it; the next one tries again
                contain(exc, log=log, what="l1 sizes: an L1 could not be measured this sample")
        return found

    def _export(self, sizes: Mapping[str, TableSize]) -> None:
        """set each declared table's gauges, zero those of tables gone since the last sample, and log each
        undeclared table once.

        :param sizes: table -> its size
        :ptype sizes: Mapping[str, TableSize]
        :return: nothing
        :rtype: None
        """
        policies = declared(Tier.L1)
        exported: set[tuple[str, str]] = set()
        for table, size in sizes.items():
            policy = policies.get(table)
            if policy is None:
                self._log_undeclared(table)
                continue
            labels = (table, policy.policy.value)
            HELD_ENTRIES.labels(tier=Tier.L1.value, name=table, policy=labels[1]).set(size.rows)
            HELD_BYTES.labels(tier=Tier.L1.value, name=table, policy=labels[1]).set(size.bytes)
            exported.add(labels)
        for table, policy_name in self._exported - exported:
            HELD_ENTRIES.labels(tier=Tier.L1.value, name=table, policy=policy_name).set(0)
            HELD_BYTES.labels(tier=Tier.L1.value, name=table, policy=policy_name).set(0)
        self._exported = exported

    def _log_undeclared(self, table: str) -> None:
        if table not in self._undeclared:
            self._undeclared.add(table)
            log.info(
                "l1 sizes: a table with no declared eviction policy is measured but not exported",
                extra={"extra_data": {"table": table}},
            )

    def _run(self) -> None:
        """sample every ``interval`` until stopped, the first after one interval: the sampling thread.

        :return: nothing
        :rtype: None
        """
        while not self._stop.wait(self._interval):
            self.sample_once()


#: the process's sampler: every L1 a collection registry is configured with is watched (``CollectionRegistry.configure``)
L1_SIZES: Final = _L1Sizes()
