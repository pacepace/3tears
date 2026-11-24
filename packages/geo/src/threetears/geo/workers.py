"""geo's CPU in worker processes, its rows handed over through shared memory.

**Why.** A tile build (decoding, simplifying, projecting, encoding) and a chunk's bounds read are CPU, and CPU in a
serving process runs in a worker process, never on the event loop or a thread (the interpreter lock). The rows cross
through a :class:`~threetears.core.shared_arrow.SharedArrowStore` hand-off: written as Arrow IPC into a
memory-backed directory with the lock released, mapped read-only by the worker, never pickled; what the pool pickles
is the hand-off's name. A bounds read's answer comes back the same way, through a result slot the worker writes.

**Where the files live.** One store per process, in a directory of its own beside the process's L1 files
(:func:`~threetears.core.cache.l1_file.l1_directory`: ``/dev/shm``, the memory-backed volume a deployment already
mounts for its L1), named ``THREETEARS_GEO_HANDOFF_DIR`` to put it elsewhere; bounded by
``THREETEARS_GEO_HANDOFF_BYTES``. A store made over a directory clears what no live process holds, so a crashed
process's hand-offs go when the next one starts.

**Small work stays here.** A tile or chunk of at most :data:`INLINE_ROWS` rows is built on the loop: the hand-off's
round trip would cost the loop more than the work.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pyarrow as pa
import pyarrow.compute as pc

from threetears.core.cache.l1_file import l1_directory
from threetears.core.process_workers import ProcessWorkers
from threetears.core.shared_arrow import HandedTables, SegmentSlot, SharedArrowStore
from threetears.geo.bounds import BOUNDS_COLUMNS, BoundsOf
from threetears.geo.geometry import point_bounds, wkb_bounds
from threetears.observe import counter, get_logger

__all__ = [
    "INLINE_ROWS",
    "GeometryColumns",
    "mapped_rows",
    "ran_inline",
    "read_bounds",
    "rows_of",
    "run_over_rows",
]

log = get_logger(__name__)

#: worker processes for geo's CPU: a precompute pass's builds (its ``concurrency``) and a viewport's misses run side by
#: side across them, the loop holding none of their CPU
_WORKERS_COUNT: Final = 2

#: the pool every tile build and bounds read past :data:`INLINE_ROWS` runs in
_WORKERS: Final = ProcessWorkers(workers=_WORKERS_COUNT, name="geo tile builds", process_name="3t-geo-tile")

#: at most this many rows are worked on the loop. measured 2026-11-23 (M-series, CPython 3.14, z12 feature band of
#: random polygons): a build held the loop 0.15 ms at 1 row, 0.6 ms at 16, 1.1 ms at 32 and 13 ms at 3,000, while a
#: worker's build held no turn of the loop past 0.01 ms (0.4-0.9 ms of wall time at these sizes). so past 16 rows,
#: which would hold the loop past half a millisecond, the work goes to a worker
INLINE_ROWS: Final = 16

#: names the hand-off store's directory, in place of one beside the process's L1 files
_HANDOFF_DIR_ENV: Final = "THREETEARS_GEO_HANDOFF_DIR"

#: the most bytes the hand-off store may hold
_HANDOFF_BYTES_ENV: Final = "THREETEARS_GEO_HANDOFF_BYTES"

#: the store's bound when the environment names none: the rows of the builds and bounds reads running at once (two
#: workers), a low-zoom tile of a dense layer each
_DEFAULT_HANDOFF_BYTES: Final = 1 << 30

#: a bounds read's answer slot: four float64 columns and their validity, per row, and the file's own
_BOUNDS_ROW_BYTES: Final = 4 * 9
_BOUNDS_FILE_BYTES: Final = 1 << 20

#: the hand-off's table name
_ROWS: Final = "rows"

#: the bounds answer's table name
_BOUNDS: Final = "bounds"

_RUNS = counter(
    "geo_worker_runs_total",
    description="geo CPU runs (tile builds, bounds reads) by where they ran: worker (a worker process over a "
    "shared-memory hand-off), inline (on the loop, a few rows) or thread (a bounds callable that is not plain data)",
    label_names=("work", "where"),
)


@dataclass(frozen=True, slots=True)
class GeometryColumns:
    """a layer's geometry columns, as a :data:`~threetears.geo.bounds.BoundsOf`: every row's bounds from a whole
    table (:func:`~threetears.geo.geometry.wkb_bounds` for a WKB column, else
    :func:`~threetears.geo.geometry.point_bounds` for a lon/lat pair).

    plain data, so a :class:`~threetears.geo.features.FeatureCache` handed one reads bounds in a worker process
    (:func:`read_bounds`); any other callable is read on a thread.

    :param geometry_column: the WKB/EWKB column, for a polygon-shaped layer
    :ptype geometry_column: str | None
    :param longitude_column: degrees east, for a point-shaped layer
    :ptype longitude_column: str | None
    :param latitude_column: degrees north, for a point-shaped layer
    :ptype latitude_column: str | None
    """

    geometry_column: str | None = None
    longitude_column: str | None = None
    latitude_column: str | None = None

    def __call__(self, table: pa.Table) -> pa.Table:
        """every row's bounds.

        :param table: source rows
        :ptype table: pa.Table
        :return: the four float64 :data:`~threetears.geo.bounds.BOUNDS_COLUMNS`, one row per row
        :rtype: pa.Table
        """
        return _bounds_of_columns(table, self.geometry_column, self.longitude_column, self.latitude_column)


def _bounds_of_columns(table: pa.Table, geometry: str | None, longitude: str | None, latitude: str | None) -> pa.Table:
    """every row's bounds from the named columns: the one place :class:`GeometryColumns` and its worker job read them.

    :raises ValueError: when the columns name neither a geometry column nor a lon/lat pair
    """
    if geometry is not None:
        return wkb_bounds(table, geometry)
    if longitude is None or latitude is None:
        raise ValueError("geometry columns name neither a geometry column nor a longitude/latitude pair")
    return point_bounds(table, longitude, latitude)


class _Handoffs:
    """the hand-off store of the directory the environment names, made at first use (on a thread: a store's start
    sweeps its directory). One store, replaced whole when the directory named changes (a test's)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held: tuple[str, SharedArrowStore] | None = None

    def store(self) -> SharedArrowStore:
        """this process's store. Blocking.

        :return: the store
        :rtype: SharedArrowStore
        """
        named = os.environ.get(_HANDOFF_DIR_ENV, "").strip()
        root = named or str(l1_directory() / f"3tears-geo-handoffs-{os.getpid()}")
        with self._lock:
            if self._held is None or self._held[0] != root:
                max_bytes = int(os.environ.get(_HANDOFF_BYTES_ENV, "").strip() or _DEFAULT_HANDOFF_BYTES)
                Path(root).mkdir(parents=True, exist_ok=True)
                self._held = (root, SharedArrowStore(Path(root), max_bytes=max_bytes, name="geo hand-offs"))
            return self._held[1]


_HANDOFFS: Final = _Handoffs()


def _plain(handed: HandedTables) -> tuple[str, str, tuple[str, ...], int]:
    """a hand-off as plain data, for the worker (``HandedTables(*plain)`` there)."""
    return (handed.root, handed.handoff_id, tuple(handed.tables), int(handed.nbytes))


async def run_over_rows(job: Callable[..., Any], rows: pa.Table, *args: Any, work: str) -> Any:
    """``job`` over ``rows`` in a worker process, the rows handed over through shared memory; the loop only waits.

    the job is called ``job(handoff, *args)`` with the hand-off as plain data: it maps the rows with
    :func:`mapped_rows`. the hand-off is released once the job is done, whichever way it ended.

    :param job: a module function, importable by name in a fresh process
    :ptype job: Callable[..., Any]
    :param rows: the rows
    :ptype rows: pa.Table
    :param args: the job's other arguments: plain data
    :ptype args: Any
    :param work: what the work is, for the run counter
    :ptype work: str
    :return: the job's answer: plain data
    :rtype: Any
    :raises SharedArrowFullError: when the store has no room for the rows
    """
    store = await asyncio.to_thread(_HANDOFFS.store)
    handed = await asyncio.to_thread(store.hand, {_ROWS: rows})
    try:
        answer = await _WORKERS.run(job, _plain(handed), *args)
    finally:
        await asyncio.to_thread(store.release, handed)
    _RUNS.labels(work=work, where="worker").inc()
    return answer


def mapped_rows(handoff: tuple[str, str, tuple[str, ...], int]) -> Any:
    """the worker's mapping of a hand-off's rows: a context manager whose ``tables["rows"]`` is the table.

    :param handoff: the hand-off as :func:`run_over_rows` passed it
    :ptype handoff: tuple[str, str, tuple[str, ...], int]
    :return: the mapped tables (close it, or use it as a context manager, before the job returns)
    :rtype: MappedTables
    """
    return HandedTables(*handoff).mapped()


def rows_of(mapped: Any) -> pa.Table:
    """the rows of a :func:`mapped_rows` mapping.

    :param mapped: the mapping
    :ptype mapped: MappedTables
    :return: the rows, mapped read-only
    :rtype: pa.Table
    """
    table: pa.Table = mapped.tables[_ROWS]
    return table


async def read_bounds(bounds_of: BoundsOf, table: pa.Table) -> pa.Table:
    """every row's bounds, off the loop: a layer's own :class:`GeometryColumns` in a worker process over shared
    memory (the answer back through a result slot), any other callable on a thread, since it is not plain data and
    cannot cross to a worker. which path ran is logged and counted.

    :param bounds_of: the bounds read
    :ptype bounds_of: BoundsOf
    :param table: the rows
    :ptype table: pa.Table
    :return: the four :data:`~threetears.geo.bounds.BOUNDS_COLUMNS`, one row per row, in this process's memory
    :rtype: pa.Table
    """
    if not isinstance(bounds_of, GeometryColumns):
        log.debug(
            "bounds read on a thread: the bounds callable is not plain data",
            extra={"extra_data": {"rows": table.num_rows}},
        )
        _RUNS.labels(work="bounds", where="thread").inc()
        found: pa.Table = await asyncio.to_thread(bounds_of, table)
        return found
    if table.num_rows <= INLINE_ROWS:
        _RUNS.labels(work="bounds", where="inline").inc()
        return bounds_of(table)
    store = await asyncio.to_thread(_HANDOFFS.store)
    slot = await asyncio.to_thread(store.reserve, _BOUNDS_FILE_BYTES + _BOUNDS_ROW_BYTES * table.num_rows)
    try:
        await run_over_rows(
            _bounds_job,
            table,
            (bounds_of.geometry_column, bounds_of.longitude_column, bounds_of.latitude_column),
            (slot.root, slot.slot_id, slot.max_bytes),
            work="bounds",
        )
    except BaseException:
        await asyncio.to_thread(store.discard, slot)
        raise
    log.debug("bounds read in a worker process", extra={"extra_data": {"rows": table.num_rows}})
    return await asyncio.to_thread(_claimed_bounds, store, slot)


def _claimed_bounds(store: SharedArrowStore, slot: SegmentSlot) -> pa.Table:
    """a bounds job's answer, claimed and copied into this process's memory, its slot let go: the cache keeps bounds
    for as long as it holds a chunk, which must not pin the shared store's files. Blocking.

    :raises LookupError: when the worker wrote no answer
    """
    segment = store.claim(slot)
    if segment is None:
        raise LookupError("a bounds job wrote no answer")
    with segment:
        mapped: pa.Table = segment.tables[_BOUNDS]
        # take copies every buffer: nothing here keeps the segment's pages
        everything = pa.array(range(mapped.num_rows), type=pa.int64())
        return pa.table({name: pc.take(mapped.column(name), everything) for name in BOUNDS_COLUMNS})


def _bounds_job(
    handoff: tuple[str, str, tuple[str, ...], int],
    columns: tuple[str | None, str | None, str | None],
    slot: tuple[str, str, int],
) -> int:
    """a bounds read in a worker: the rows mapped, their bounds written into the answer slot.

    :return: the bytes written
    :rtype: int
    """
    with mapped_rows(handoff) as mapped:
        found = _bounds_of_columns(rows_of(mapped), *columns)
        return SegmentSlot(*slot).write({_BOUNDS: found})


def ran_inline(work: str) -> None:
    """count a run of ``work`` done on the loop (:data:`INLINE_ROWS` rows or fewer).

    :param work: what the work is
    :ptype work: str
    :return: nothing
    :rtype: None
    """
    _RUNS.labels(work=work, where="inline").inc()
