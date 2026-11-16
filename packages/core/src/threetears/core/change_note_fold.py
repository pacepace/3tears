"""a restore's keyed change notes folded into their epoch records in a worker process, never on the event loop.

**Why.** A lost change log is put back from the change notes (:class:`~threetears.core.collections.change_notes.
ChangeNoteKeeper`) inside a serving process (the hub). The restore read every keyed note as rows (asyncpg records),
folded them into each table's distinct rows and validated them as :class:`~threetears.core.collections.epoch_change.
ChangedRow` models on the event loop, and ``finish_restore`` then wrote each chunk's records there too. Measured (L1,
2026-11-15, ``test_change_note_restore_speed_live.py::test_a_restore_holds_the_loop_under_1_ms_a_turn``, the
200,000-note ring, each turn timed on the loop thread's CPU clock): the loop was held 29.1 ms at worst on PG 16 and
22.8 ms on YugabyteDB, p99 7.8 / 8.5 ms -- each page's fold and validation one turn of ~8.5 ms (39 such turns), and
the driver's decode of the rows, with the collector's pauses over the objects it made (18.9 and 22.5 ms), the worst.
The budget is 1 ms a turn. With the fold here (same test and ring, 3 restores each, e033c51d9/5d680cbe3): the
longest turn 0.33-0.45 ms on PG, 0.31-0.38 ms on YugabyteDB, p99 0.13-0.18 / 0.16-0.24 ms; the restore speed test
(2x the bare reads) stays green on both.

**What moved.** The serving process now reads the keyed notes as ``COPY ... TO STDOUT`` CSV text (bytes, nothing
decoded there), cuts it into pages at line ends, and hands each page to a worker through
:mod:`threetears.core.shared_arrow` (a one-value ``large_binary`` table, :data:`_PAGE_SCHEMA`, written off the loop
and mapped by the worker, never pickled). The worker (:func:`~threetears.core.change_note_fold_job._fold_page`) parses the CSV with Arrow, finds each table's
chunks by the restore's rule, and assembles each chunk's records as the log stores them, in the form
:func:`~threetears.core.collections.epoch_change.keyed_record_form` cuts out of ``epoch_change_body``; they come back
the same way (:data:`_RECORDS_SCHEMA`). The loop hands bytes over and appends the bytes it gets back.

**One line a note.** Every value is written so it holds no raw line end: the table, the tenant and the ids as JSON
text (``to_json``, ``array_to_json``), the columns as their ``jsonb`` text. A raw ``\\n`` in the text is then always a
note's end, so the serving process cuts a page at its last one with one ``rfind``, never parsing the CSV.

**The chunk rule** is the restore's own (``_RESTORE_CHUNK_ROWS``): a table's (and tenant's) distinct rows, in order,
cut into chunks only where the key changes once a chunk holds the chunk's rows or more. A page's last chunk may go on
in the next page, so it is not folded: the worker answers where it starts (the bytes it consumed), and the serving
process hands those bytes again at the front of the next page. The final page folds everything.

**What is checked.** A note's ids come from its ``TEXT[]`` keys and its columns from the trigger's
``jsonb_build_object`` of ``::text`` values (:mod:`threetears.core.ddl.change_notes`), so each row is a
:class:`ChangedRow` by construction; the worker refuses columns that are not a JSON object, as the fold always did,
and a row too large for any record. A follower reads each record through ``read_epoch_change_body``, which validates
it whole.

The worker's side (:func:`~threetears.core.change_note_fold_job._fold_page`) is a module of its own, which
imports only Arrow, NumPy and the shared-Arrow job types, so a spawned worker starts small.
"""

from __future__ import annotations

import asyncio
import atexit
import tempfile
import threading
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa

from threetears.core.change_note_fold_job import (
    _PAGE_SCHEMA,
    _PAGE_TABLE,
    _RECORDS_TABLE,
    FoldRefusedError,
    _fold_page,
    _ready,
)
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.shared_arrow import SharedArrowPool, SharedArrowStore
from threetears.observe import contain, get_logger, remove_tree

__all__ = ["RestoreFolder"]

log = get_logger(__name__)


def _page_table(data: bytes) -> pa.Table:
    """``data`` as a page to hand over (:data:`_PAGE_SCHEMA`), its bytes not copied.

    :param data: whole note lines
    :ptype data: bytes
    :return: the table
    :rtype: pa.Table
    """
    values = pa.py_buffer(data)
    offsets = pa.py_buffer(np.array([0, values.size], dtype=np.int64).tobytes())
    array = pa.Array.from_buffers(pa.large_binary(), 1, [None, offsets, values])
    return pa.Table.from_arrays([array], schema=_PAGE_SCHEMA)


#: one record a page folded into: its chunk's number in the page, its table, its tenant, its body
_FoldedRecord = tuple[int, str, str | None, bytes]

#: the most the folder's store holds at once (a page handed, its records, the next page): its pages are about
#: :data:`~threetears.core.collections.change_notes._RESTORE_PAGE_BYTES` and its records about twice that, so this is
#: far above what a restore takes, and below the 64 MiB a container's ``/dev/shm`` has by default
_STORE_BYTES: Final = 48 << 20

#: a worker holding more private memory than this after a page is replaced
_WORKER_PRIVATE_BYTES: Final = 512 << 20

#: each folder's shutdown still running (:meth:`RestoreFolder.close`): held here so none is collected mid-way
_CLOSING: set[asyncio.Task[None]] = set()


class RestoreFolder:
    """one restore's worker process and the shared-memory store its pages and records cross: made for the restore,
    gone with it.

    WHY one per restore, never kept (R4, 2026-11-15, Pace: memory only for a measured need): the first version kept
    one worker per process from its first restore to its exit. Measured after a 200,000-note fold
    (S/r4/rss.py, macOS arm64, Python 3.14, the keeper's modules imported as a hub imports them): the kept worker
    held 193 MiB resident and its store directory stayed, for a restore that comes only at a cold start or a lost
    log. Now the worker exits and the store is removed when the restore ends (:meth:`close`): nothing is left. The
    cost is a spawn per restore, started on a thread as the restore begins (:meth:`start`), beside its sizing reads,
    so it is off the loop. Measured in ``test_change_note_restore_speed_live``'s world (PG 16, 200,000 notes, quiet
    machine, R4 2026-11-15): spawn to ready 196-211 ms (237-242 ms before the job moved to
    :mod:`threetears.core.change_note_fold_job`); the first page waits for it from ~80 ms, the six pages after fold
    in ~15 ms each, and the worker's exit takes ~25-55 ms more: a restore 331-372 ms against 148-176 ms of bare reads,
    2.1-2.4x (the kept worker's restore was inside 2x). YugabyteDB, whose bare reads are slower, stays inside 2x.
    So (Pace 2026-11-15, R4 option 1) a restore's speed budget is twice its bare reads plus one worker start, and the
    start has its own budget (``test_change_note_fold.py::test_a_restore_worker_starts_within_its_budget``); the
    worker's exit is off the restore's path: :meth:`close` hands it to a task of its own and returns.

    The store's directory is a fresh one, memory-backed where the system has ``/dev/shm``. Used as ``async with``.
    """

    def __init__(self) -> None:
        #: held while a thread makes the store and the pool: never by the event loop
        self._lock = threading.Lock()
        self._pool: SharedArrowPool | None = None
        self._root: Path | None = None
        #: the worker's spawn, started by :meth:`start`
        self._warming: asyncio.Future[None] | None = None
        #: the worker's exit and the store's removal, started by :meth:`close`
        self._closing: asyncio.Task[None] | None = None

    async def __aenter__(self) -> RestoreFolder:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    def _opened(self) -> SharedArrowPool:
        """the pool, made on first use (on a thread: the store's start sweeps its directory).

        :return: the pool
        :rtype: SharedArrowPool
        """
        with self._lock:
            if self._pool is None:
                shm = Path("/dev/shm")
                parent = shm if shm.is_dir() else Path(tempfile.gettempdir())
                self._root = Path(tempfile.mkdtemp(prefix="change-note-restore-", dir=parent))
                # a process that dies mid-restore never reaches close(): its exit removes the store
                atexit.register(self._remove_root)
                self._pool = SharedArrowPool(
                    SharedArrowStore(self._root, max_bytes=_STORE_BYTES),
                    workers=1,
                    private_bytes=_WORKER_PRIVATE_BYTES,
                    name="change-note-restore",
                )
            return self._pool

    def _remove_root(self) -> None:
        """remove the store's directory, once (at :meth:`close`, or at the process's exit).

        :return: nothing
        :rtype: None
        """
        with self._lock:
            root, self._root = self._root, None
        if root is not None:
            atexit.unregister(self._remove_root)
            remove_tree(root, log=log, what="the change-note restore's store")

    async def _warm(self) -> None:
        """make the pool and spawn its worker (on a thread), running :func:`_ready`.

        :return: nothing
        :rtype: None
        """
        pool = await asyncio.to_thread(self._opened)
        await pool.run(_ready, read=False)

    def start(self) -> None:
        """spawn the worker now, off the loop, so the first page finds it ready; once.

        :return: nothing
        :rtype: None
        """
        if self._warming is None:
            self._warming = asyncio.ensure_future(self._warm())

    async def fold(
        self, page: bytes, *, final: bool, form: tuple[bytes, ...], chunk_rows: int, max_bytes: int
    ) -> tuple[int, list[_FoldedRecord]]:
        """fold ``page`` into its records in the worker (:func:`_fold_page`); the loop only hands bytes and waits.

        :param page: whole note lines, the last chunk's lines of the page before at its front
        :ptype page: bytes
        :param final: the page ends the notes
        :ptype final: bool
        :param form: the record's pieces, as :func:`_fold_page` takes them
        :ptype form: tuple[bytes, ...]
        :param chunk_rows: the rows a chunk holds before it may be cut
        :ptype chunk_rows: int
        :param max_bytes: the most one record may take
        :ptype max_bytes: int
        :return: the page's bytes folded, and the records, in order
        :rtype: tuple[int, list[_FoldedRecord]]
        :raises GenerationUnavailableError: as :func:`_fold_page`
        :raises SharedArrowFullError: when the store has no room for the page or its records
        """
        pool = await asyncio.to_thread(self._opened)
        try:
            result = await pool.run(
                _fold_page,
                final,
                form,
                chunk_rows,
                max_bytes,
                read=False,
                tables={_PAGE_TABLE: _page_table(page)},
                result_bytes=2 * len(page) + (1 << 20),
            )
        except FoldRefusedError as exc:
            raise GenerationUnavailableError(str(exc)) from exc
        segment = result.segment
        assert segment is not None  # narrow: the job always writes its records
        try:
            records = segment.tables[_RECORDS_TABLE]
            folded = list(
                zip(
                    records.column("chunk").to_pylist(),
                    records.column("table").to_pylist(),
                    records.column("tenant").to_pylist(),
                    records.column("body").to_pylist(),
                    strict=True,
                )
            )
        finally:
            # closing sweeps the store's directory: on a thread, never the loop
            await asyncio.to_thread(segment.close)
        return int(result.value), folded

    async def _shut(self) -> None:
        """wait for the spawn, shut the worker down (waiting for it to exit) and remove the store. Raises nothing: it
        runs on a task of its own, after the restore has its answer, so a failure is only said.

        :return: nothing
        :rtype: None
        """
        if self._warming is not None:
            try:
                await self._warming
            except Exception as exc:  # noqa: BLE001 -- the restore already has its answer; a failed spawn is only said
                contain(exc, log=log, what="the change-note restore's worker failed to start", level="warning")
        with self._lock:
            pool = self._pool
        try:
            if pool is not None:
                await pool.close()
        except Exception as exc:  # noqa: BLE001 -- the restore already has its answer; a failed exit is only said
            contain(exc, log=log, what="the change-note restore's worker failed to shut down", level="warning")
        finally:
            await asyncio.to_thread(self._remove_root)

    async def close(self) -> None:
        """start the worker's exit and the store's removal on a task of their own, and return: the restore never waits
        for them (the exit took ~25-55 ms of a restore's 331-372 ms, R4 2026-11-15). The task is held until it ends
        (``_CLOSING``) and never belongs to the caller, so a caller cancelled meanwhile leaves it to finish; a process
        ending before it does removes the store at its exit (``atexit``). :meth:`closed` waits for it. Once.

        :return: nothing
        :rtype: None
        """
        if self._closing is None:
            self._closing = asyncio.get_running_loop().create_task(self._shut())
            _CLOSING.add(self._closing)
            self._closing.add_done_callback(_CLOSING.discard)

    async def closed(self) -> None:
        """wait until the shutdown :meth:`close` started has ended: the worker exited and the store is gone. A caller
        cancelled here leaves the shutdown running.

        :return: nothing
        :rtype: None
        """
        if self._closing is not None:
            await asyncio.shield(self._closing)
