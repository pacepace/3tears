"""the worker side of a restore's fold (:mod:`threetears.core.change_note_fold`): what its spawned process imports.

WHY a module of its own (R4, 2026-11-15): a restore now spawns its worker per restore (see
:class:`~threetears.core.change_note_fold.RestoreFolder`), so the worker's start is on the restore's path. The worker
imports the job's module; with the job beside the serving side it imported ``threetears.core.exceptions`` and through
it ``threetears.nats`` (57-80 ms of the worker's ~205 ms of imports, measured with ``python -X importtime``, macOS
arm64, quiet machine). Here it imports only Arrow, NumPy and the shared-Arrow job types (which the worker loads
anyway), and refuses a page with :class:`FoldRefusedError`, which the serving side raises as
``GenerationUnavailableError``.
"""

from __future__ import annotations

import json
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from threetears.core.shared_arrow import JobContext

__all__: list[str] = []


class FoldRefusedError(ValueError):
    """a page the fold refuses: columns that are not an object, a row too large for any record, or a page that is not
    whole lines. The serving side raises it as ``GenerationUnavailableError``."""


#: the hand-off's table holding a page
_PAGE_TABLE: Final = "page"

#: a page as it is handed to the worker: the CSV text of whole note lines, one value, never copied
_PAGE_SCHEMA: Final = pa.schema([pa.field("csv", pa.large_binary(), nullable=False)])

#: the result's table of records
_RECORDS_TABLE: Final = "records"

#: the records a page folds into, in order: each chunk's number in the page (a chunk's records are one change), its
#: table and tenant (``None`` for ``_all``), and one record's body as the log stores it
_RECORDS_SCHEMA: Final = pa.schema(
    [
        pa.field("chunk", pa.int64(), nullable=False),
        pa.field("table", pa.string(), nullable=False),
        pa.field("tenant", pa.string()),
        pa.field("body", pa.large_binary(), nullable=False),
    ]
)

#: the columns of a page's CSV, as the restore's ``COPY`` writes them: each a JSON text, the tenant and the columns
#: NULL (an unquoted empty field) when the note names none
_CSV_COLUMNS: Final = ("table_json", "tenant_json", "keys_json", "columns_json")

_CSV_CONVERT: Final = pacsv.ConvertOptions(
    column_types=dict.fromkeys(_CSV_COLUMNS, pa.large_string()),
    strings_can_be_null=True,
    quoted_strings_can_be_null=False,
    null_values=[""],
)

_CSV_PARSE: Final = pacsv.ParseOptions(newlines_in_values=False)


def _changed_at(values: Any) -> np.ndarray:
    """where each value differs from the one before it (the first always), as a boolean array.

    :param values: a string array, NULL read as the empty text (never a JSON text)
    :ptype values: pa.Array
    :return: one flag per value
    :rtype: np.ndarray
    """
    count = len(values)
    flags = np.ones(count, dtype=bool)
    if count > 1:
        filled = pc.fill_null(values, "")
        flags[1:] = pc.not_equal(filled.slice(1), filled.slice(0, count - 1)).to_numpy(zero_copy_only=False)
    return flags


def _chunks(groups: np.ndarray, keys: np.ndarray, count: int, chunk_rows: int) -> list[tuple[int, int]]:
    """each chunk as ``(first row, past its last)``: a group's rows cut where the key changes once a chunk holds
    ``chunk_rows`` rows or more.

    :param groups: where each group (a table and tenant) starts, ascending, the first 0
    :ptype groups: np.ndarray
    :param keys: where each key starts (every group start among them), ascending
    :ptype keys: np.ndarray
    :param count: the rows
    :ptype count: int
    :param chunk_rows: the rows a chunk holds before it may be cut
    :ptype chunk_rows: int
    :return: the chunks, in order, covering every row
    :rtype: list[tuple[int, int]]
    """
    chunks: list[tuple[int, int]] = []
    ends = [*groups[1:].tolist(), count]
    for first, end in zip(groups.tolist(), ends, strict=True):
        start = first
        while True:
            at = int(np.searchsorted(keys, start + chunk_rows, side="left"))
            cut = int(keys[at]) if at < len(keys) else count
            if cut >= end:
                chunks.append((start, end))
                break
            chunks.append((start, cut))
            start = cut
    return chunks


def _runs(sizes: np.ndarray, start: int, end: int, room: int) -> list[tuple[int, int]]:
    """a chunk's rows cut into consecutive runs whose rows, comma-separated, fit ``room`` bytes: each as long as fits.

    :param sizes: each row's bytes plus its comma
    :ptype sizes: np.ndarray
    :param start: the chunk's first row
    :ptype start: int
    :param end: past its last
    :ptype end: int
    :param room: the bytes a record has for its rows
    :ptype room: int
    :return: the runs, in order
    :rtype: list[tuple[int, int]]
    :raises FoldRefusedError: when one row alone does not fit
    """
    total = np.cumsum(sizes[start:end])
    runs: list[tuple[int, int]] = []
    first, before = start, 0
    while first < end:
        # the last comma is not written: a run fits while its sizes sum to room + 1
        past = start + int(np.searchsorted(total, before + room + 1, side="right"))
        if past == first:
            raise FoldRefusedError(
                f"a restored row is {int(sizes[first]) - 1} bytes, past the {room}-byte room of a record: no epoch "
                "record can name it"
            )
        runs.append((first, past))
        before = int(total[past - start - 1])
        first = past
    return runs


def _fold_page(context: JobContext, final: bool, form: tuple[bytes, ...], chunk_rows: int, max_bytes: int) -> int:
    """fold a handed page of keyed notes into their records (in the worker): written to the job's result.

    :param context: the job's context: the page handed (:data:`_PAGE_TABLE`), the result's slot
    :ptype context: JobContext
    :param final: the page ends the notes: its last chunk is folded too
    :ptype final: bool
    :param form: the record's pieces (:class:`~threetears.core.collections.epoch_change.KeyedRecordForm`), in its
        field order: head, middle, tail, row head, row columns, row tail
    :ptype form: tuple[bytes, ...]
    :param chunk_rows: the rows a chunk holds before it may be cut
    :ptype chunk_rows: int
    :param max_bytes: the most one record may take
    :ptype max_bytes: int
    :return: the page's bytes folded: the rest (its last chunk, unless ``final``) is to be handed again before the
        next page
    :rtype: int
    :raises FoldRefusedError: when a note's columns are not an object, a row fits no record, or the page
        is not whole lines
    """
    assert context.tables is not None and context.result is not None  # narrow: the keeper hands both
    head, middle, tail, row_head, row_columns, row_tail = form
    data = context.tables.tables[_PAGE_TABLE].column("csv").chunk(0).buffers()[2]
    if data.size == 0:
        context.result.write({_RECORDS_TABLE: _RECORDS_SCHEMA.empty_table()})
        return 0
    raw = np.frombuffer(data, dtype=np.uint8)
    ends = np.flatnonzero(raw == 0x0A)
    notes = pacsv.read_csv(
        pa.BufferReader(data),
        read_options=pacsv.ReadOptions(column_names=list(_CSV_COLUMNS)),
        parse_options=_CSV_PARSE,
        convert_options=_CSV_CONVERT,
    ).combine_chunks()
    count = notes.num_rows
    if count != len(ends) or int(ends[-1]) != data.size - 1:
        raise FoldRefusedError(f"a restore's page is not {count} whole note lines ({len(ends)} line ends)")
    tables, tenants, keys, columns = (notes.column(name).chunk(0) for name in _CSV_COLUMNS)
    carried = pc.is_valid(columns)
    if pc.any(pc.invert(pc.starts_with(pc.fill_null(columns, "{"), "{"))).as_py():
        raise FoldRefusedError("a change note's columns are not an object")
    grouped = _changed_at(tables) | _changed_at(tenants)
    chunks = _chunks(np.flatnonzero(grouped), np.flatnonzero(grouped | _changed_at(keys)), count, chunk_rows)
    folded = int(data.size)
    if not final:
        # the last chunk may go on in the next page: handed again at its front
        last_start = chunks.pop()[0]
        folded = 0 if last_start == 0 else int(ends[last_start - 1]) + 1
    # each row's JSON and its comma, laid end to end in one buffer: a run is one slice of it
    opened, between, closed = (
        pa.scalar(piece.decode(), pa.large_string()) for piece in (row_head, row_columns, row_tail + b",")
    )
    nothing = pa.scalar("", pa.large_string())
    rows = pc.if_else(
        carried,
        pc.binary_join_element_wise(opened, keys, between, columns, closed, nothing),
        pc.binary_join_element_wise(opened, keys, closed, nothing),
    )
    offsets = np.frombuffer(rows.buffers()[1], dtype=np.int64)[rows.offset : rows.offset + count + 1]
    laid = rows.buffers()[2]
    sizes = np.diff(offsets)
    numbers: list[int] = []
    names: list[str] = []
    owners: list[str | None] = []
    bodies: list[bytes] = []
    for number, (start, end) in enumerate(chunks):
        table_json = tables[start].as_py()
        tenant_json = tenants[start].as_py()
        name = json.loads(table_json)
        owner = None if tenant_json is None else json.loads(tenant_json)
        before = head + table_json.encode() + middle
        for first, past in _runs(sizes, start, end, max_bytes - len(before) - len(tail)):
            numbers.append(number)
            names.append(name)
            owners.append(owner)
            # the run's rows less the last comma
            body = laid.slice(int(offsets[first]), int(offsets[past] - offsets[first]) - 1).to_pybytes()
            bodies.append(before + body + tail)
    context.result.write(
        {
            _RECORDS_TABLE: pa.Table.from_arrays(
                [
                    pa.array(numbers, pa.int64()),
                    pa.array(names, pa.string()),
                    pa.array(owners, pa.string()),
                    pa.array(bodies, pa.large_binary()),
                ],
                schema=_RECORDS_SCHEMA,
            )
        }
    )
    return folded


def _ready(_context: JobContext) -> None:
    """nothing: the job a restore's folder (:class:`~threetears.core.change_note_fold.RestoreFolder`) runs first, so its worker spawns and imports while the restore sizes.

    :param _context: the job's context (unused)
    :ptype _context: JobContext
    :return: nothing
    :rtype: None
    """
