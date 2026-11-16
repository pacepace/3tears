"""change_note_fold: a restore's keyed notes, as the keeper's ``COPY`` writes them, folded into the records the old fold
on the event loop made (each distinct row of a table and tenant, in chunks cut where the key changes, each chunk's
rows split into records within the bound), off the loop: through the front door (``RestoreFolder.fold``), the page handed to
the spawned worker and its records claimed back over shared Arrow."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import tempfile
import time
from collections.abc import Iterator, Sequence
from itertools import groupby
from pathlib import Path
from typing import Any

import pytest

from threetears.core.change_note_fold import RestoreFolder
from threetears.core.collections.epoch_change import (
    ChangedRow,
    changed_row_runs,
    epoch_change_body,
    keyed_record_form,
    read_epoch_change_body,
)
from threetears.core.exceptions import GenerationUnavailableError

_FORM = keyed_record_form()
_PIECES = (_FORM.head, _FORM.middle, _FORM.tail, _FORM.row_head, _FORM.row_columns, _FORM.row_tail)

#: a note as the restore reads it: its table, tenant, ids and invalidation columns
_Note = tuple[str, str | None, list[str], dict[str, str | None] | None]


def _field(value: Any) -> str:
    """one CSV field as ``COPY ... CSV FORCE_QUOTE *`` writes a JSON text: quoted, quotes doubled; NULL bare."""
    if value is None:
        return ""
    text = json.dumps(value, ensure_ascii=False)
    return '"' + text.replace('"', '""') + '"'


def _csv(notes: Sequence[_Note]) -> bytes:
    """the notes as the keeper's COPY writes them (to_json, array_to_json, the columns' jsonb text), one line each."""
    return "".join(",".join(_field(value) for value in note) + "\n" for note in notes).encode()


_Record = tuple[int, str, str | None, bytes]

#: one folder for the module's fold tests, so its worker spawns once; closed at the module's end
_FOLDER: RestoreFolder | None = None


@pytest.fixture(scope="module", autouse=True)
def _module_folder() -> Iterator[None]:
    global _FOLDER  # noqa: PLW0603 -- the module's one folder, set and closed here
    _FOLDER = RestoreFolder()
    try:
        yield
    finally:
        asyncio.run(_ended(_FOLDER))
        _FOLDER = None


async def _ended(folder: RestoreFolder) -> None:
    """close ``folder`` and wait for its shutdown, which runs on a task of its own."""
    await folder.close()
    await folder.closed()


async def _fold(
    page: bytes, *, final: bool, chunk_rows: int, max_bytes: int = _FORM.max_bytes
) -> tuple[int, list[_Record]]:
    assert _FOLDER is not None
    return await _FOLDER.fold(page, final=final, form=_PIECES, chunk_rows=chunk_rows, max_bytes=max_bytes)


def _changes(records: list[_Record]) -> list[tuple[str, str | None, list[list[ChangedRow]]]]:
    """each change (a chunk's records) as its table, tenant and each record's rows, read as a follower reads them."""
    changes = []
    for _chunk, held in groupby(records, key=lambda record: record[0]):
        records_of = list(held)
        _, table, tenant, _ = records_of[0]
        runs = [read_epoch_change_body(body, table, "t:1").rows for *_, body in records_of]
        changes.append((table, tenant, runs))
    return changes


async def _paged(
    notes: Sequence[_Note], page_lines: int, chunk_rows: int
) -> list[tuple[str, str | None, list[list[ChangedRow]]]]:
    """the notes folded a few lines at a time, the unfolded rest handed again at the next page's front."""
    lines = _csv(notes).splitlines(keepends=True)
    changes = []
    carried = b""
    at = 0
    while True:
        new = b"".join(lines[at : at + page_lines])
        at += page_lines
        final = at >= len(lines)
        page = carried + new
        folded, records = await _fold(page, final=final, chunk_rows=chunk_rows)
        carried = page[folded:]
        changes.extend(_changes(records))
        if final:
            assert carried == b""
            return changes


def _reference(notes: Sequence[_Note], chunk_rows: int) -> list[tuple[str, str | None, list[list[ChangedRow]]]]:
    """the old fold on the loop (``_StreamedChanges._stream`` before L1): each table and tenant's rows in chunks cut
    where the key changes once a chunk holds ``chunk_rows``, each chunk validated and split by ``changed_row_runs``."""
    changes = []
    for (table, tenant), group in groupby(notes, key=lambda note: (note[0], note[1])):
        held: list[dict[str, Any]] = []
        last: list[str] | None = None
        chunks: list[list[dict[str, Any]]] = []
        for _, _, ids, columns in group:
            if ids != last and len(held) >= chunk_rows:
                chunks.append(held)
                held = []
            last = ids
            held.append({"ids": ids, "columns": columns, "tenant": tenant})
        chunks.append(held)
        for chunk in chunks:
            rows = [ChangedRow.model_validate(row) for row in chunk]
            runs = [
                read_epoch_change_body(epoch_change_body(table, run), table, "t:1").rows
                for run in changed_row_runs(table, rows)
            ]
            changes.append((table, tenant, runs))
    return changes


def _rows(changes: list[tuple[str, str | None, list[list[ChangedRow]]]]) -> list[tuple[str, str | None, list[Any]]]:
    """each change's rows, its records joined: what a follower applies of it (records split differently apply alike)."""
    return [
        (table, tenant, [(row.ids, row.columns, row.l3_pending) for run in runs for row in run])
        for table, tenant, runs in changes
    ]


def _notes() -> list[_Note]:
    """notes of three tables and two tenants, keys noted with several invalidation columns, awkward text."""
    notes: list[_Note] = []
    for table in ("a", 'b "quoted",\nline', "c\\slash"):
        for tenant in (None, "t,1", "ü"):
            for key in range(7):
                ids = [f"k{key}", 'x"\n\\y' if key % 3 == 0 else "plain"]
                notes.append((table, tenant, ids, None))
                if key % 2:
                    notes.append((table, tenant, ids, {"owner": f"o{key}", "gone": None}))
                    notes.append((table, tenant, ids, {"owner": f"p{key}\n", "gone": "x"}))
    return notes


def test_a_record_assembled_in_its_form_is_the_body_epoch_change_body_writes() -> None:
    """Flips if the form cut out of ``epoch_change_body`` drifts from it: a record of rows with and without columns
    assembled from the pieces is byte for byte the body the model writes."""
    rows = [ChangedRow(ids=["a", "b"]), ChangedRow(ids=["c"], columns={"x": "1", "y": None})]
    row_json = [
        _FORM.row_head + b'["a","b"]' + _FORM.row_tail,
        _FORM.row_head + b'["c"]' + _FORM.row_columns + b'{"x":"1","y":null}' + _FORM.row_tail,
    ]
    assembled = _FORM.head + b'"t"' + _FORM.middle + b",".join(row_json) + _FORM.tail
    assert assembled == epoch_change_body("t", rows)


@pytest.mark.parametrize("chunk_rows", [1, 2, 3, 5000])
async def test_a_page_folds_into_the_changes_the_old_fold_made(chunk_rows: int) -> None:
    """Flips if the worker's fold names other rows, other columns, another order, or cuts its chunks elsewhere than the
    old fold on the loop: one change per chunk, cut only where the key changes."""
    notes = _notes()
    folded, records = await _fold(_csv(notes), final=True, chunk_rows=chunk_rows)
    assert folded == len(_csv(notes))
    assert _rows(_changes(records)) == _rows(_reference(notes, chunk_rows))


@pytest.mark.parametrize(("page_lines", "chunk_rows"), [(1, 2), (4, 3), (7, 5000), (50, 2)])
async def test_pages_folded_one_after_another_make_the_changes_one_page_makes(page_lines: int, chunk_rows: int) -> None:
    """Flips if a page's last chunk, which may go on in the next page, is folded before it ends, or lost, or folded
    twice: page by page, the unfolded rest handed again, the changes are the old fold's."""
    notes = _notes()
    assert _rows(await _paged(notes, page_lines, chunk_rows)) == _rows(_reference(notes, chunk_rows))


async def test_a_chunk_too_large_for_one_record_is_split_into_records_that_each_fit() -> None:
    """Flips if a chunk's records pass the bound, lose or repeat a row, or reorder them: each record fits
    ``max_bytes``, and together they hold the chunk's rows in order."""
    notes: list[_Note] = [("t", None, [f"key-{n:04d}"], {"c": "v" * (n % 9)}) for n in range(300)]
    _folded, records = await _fold(_csv(notes), final=True, chunk_rows=5000, max_bytes=2048)
    bodies = [body for *_, body in records]
    assert len(bodies) > 5 and all(len(body) <= 2048 for body in bodies)
    assert {number for number, *_ in records} == {0}
    assert _rows(_changes(records)) == _rows(_reference(notes, 5000))


async def test_a_row_no_record_can_name_is_refused() -> None:
    with pytest.raises(GenerationUnavailableError, match="no epoch record can name it"):
        await _fold(_csv([("t", None, ["k" * 400], None)]), final=True, chunk_rows=5000, max_bytes=256)


async def test_columns_that_are_not_an_object_are_refused() -> None:
    page = b'"""t""",,"[""k""]","[1]"\n'
    with pytest.raises(GenerationUnavailableError, match="not an object"):
        await _fold(page, final=True, chunk_rows=5000)


async def test_a_page_that_is_not_whole_lines_is_refused() -> None:
    with pytest.raises(GenerationUnavailableError, match="whole note lines"):
        await _fold(_csv([("t", None, ["k"], None)]).rstrip(b"\n"), final=True, chunk_rows=5000)


async def test_an_empty_final_page_folds_into_nothing() -> None:
    folded, records = await _fold(b"", final=True, chunk_rows=5000)
    assert folded == 0 and records == []


def _stores() -> set[Path]:
    """the restore stores' directories there are now, where a folder makes them (``/dev/shm``, else the temp dir)."""
    shm = Path("/dev/shm")
    parent = shm if shm.is_dir() else Path(tempfile.gettempdir())
    return set(parent.glob("change-note-restore-*"))


async def test_a_closed_folder_leaves_no_worker_and_no_store() -> None:
    """Flips if a restore's worker outlives its restore, or its shared-memory store stays: after ``close`` no child
    process of the folder's is alive and its store's directory is gone (R4, 2026-11-15: kept, the worker held 193 MiB
    for the process's life)."""
    before = {child.pid for child in multiprocessing.active_children()}
    stores_before = _stores()
    folder = RestoreFolder()
    folder.start()
    folded, _records = await folder.fold(
        _csv(_notes()), final=True, form=_PIECES, chunk_rows=3, max_bytes=_FORM.max_bytes
    )
    assert folded == len(_csv(_notes()))
    stores = _stores() - stores_before
    spawned = {child.pid for child in multiprocessing.active_children()} - before
    assert len(stores) == 1 and spawned
    await _ended(folder)
    assert not any(store.exists() for store in stores)
    assert not ({child.pid for child in multiprocessing.active_children()} & spawned)


async def test_close_returns_before_the_worker_exits_and_the_exit_still_ends() -> None:
    """Flips if a restore waits for its worker's exit (off its path since Pace 2026-11-15: ~25-55 ms of a restore's
    331-372 ms, R4), or if the exit started then does not end with no worker and no store, a caller cancelled while
    it waits included."""
    before = {child.pid for child in multiprocessing.active_children()}
    stores_before = _stores()
    folder = RestoreFolder()
    folder.start()
    await folder.fold(b"", final=True, form=_PIECES, chunk_rows=5000, max_bytes=_FORM.max_bytes)
    stores = _stores() - stores_before
    spawned = {child.pid for child in multiprocessing.active_children()} - before
    await folder.close()
    # close only started the shutdown: the store and the worker are still there as it returns
    assert all(store.exists() for store in stores) and stores
    assert spawned <= {child.pid for child in multiprocessing.active_children()}
    waiting = asyncio.ensure_future(folder.closed())
    await asyncio.sleep(0)
    waiting.cancel()
    await folder.closed()
    assert not any(store.exists() for store in stores)
    assert not ({child.pid for child in multiprocessing.active_children()} & spawned)


#: one restore worker's start, spawn to its first answer, at most (R4 option 1, Pace 2026-11-15: a restore's speed is
#: budgeted as twice its bare reads plus one worker start, and the start budgeted here on its own). Measured spawn to
#: ready 196-211 ms (R4 2026-11-15, macOS arm64, Python 3.14, quiet machine) and 308-313 ms (LAND1 2026-11-15, the
#: same laptop under measure-lock at load 2.8, five starts: scratchpad/land1/start_probe.log); the job's module
#: imports only Arrow, NumPy and the shared-Arrow job types (threetears.core.change_note_fold_job). The budget is
#: twice the slowest. A start past it is the worker importing more, or the spawn path growing: see
#: change_note_fold_job's why first. The same probe: close() returns in 0.02 ms, its shutdown ends 63-77 ms later.
_WORKER_START_BUDGET = 0.63


async def test_a_restore_worker_starts_within_its_budget() -> None:
    """Flips if a restore's worker takes longer than its budget to start and give its first answer (the best of
    three): the start every restore pays beside its reads."""
    starts: list[float] = []
    for _ in range(3):
        folder = RestoreFolder()
        started = time.perf_counter()
        folder.start()
        await folder.fold(b"", final=True, form=_PIECES, chunk_rows=5000, max_bytes=_FORM.max_bytes)
        starts.append(time.perf_counter() - started)
        await _ended(folder)
    assert min(starts) <= _WORKER_START_BUDGET, f"the worker started in {min(starts) * 1000:.0f} ms at best: {starts}"


async def test_a_folder_never_started_spawns_nothing() -> None:
    """Flips if a folder spawns a worker or makes a store before a restore of this keeper's starts it: another
    keeper's restore (``begin_restore`` answered None) costs no process."""
    before = {child.pid for child in multiprocessing.active_children()}
    stores_before = _stores()
    async with RestoreFolder():
        pass
    assert _stores() == stores_before
    assert {child.pid for child in multiprocessing.active_children()} == before
