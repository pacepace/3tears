"""a copy follower's key work, the CPU of a pass that scales with its keys, run in its worker process.

**Why.** CPU in a serving process belongs in a worker process, never on the event loop and never on a thread
(:mod:`~threetears.core.process_workers`): a thread's Arrow kernels release the interpreter lock, but they still take
the pod's cores from page serving. A pass's key work held the loop 12.2 s in one page (``distinct_keys`` under
``_stage_keyed``) and 24.3 s more hearing keys (``key_text_table`` under ``_drained``) in the local burst replay of
2026-11-25, and a waited-for file's 200,000 keys held one turn 11.1 ms on the pool host brooks-14th-mbp. Each piece
runs here instead, in the copy's own worker process (:class:`~threetears.core.collections.copy_follower.CopyFollower`'s
one :class:`~threetears.core.process_workers.ProcessWorkers`, the one its drift checks run in).

**What crosses.** Plain data (:mod:`~threetears.core.process_workers`): the key's columns, its timestamp columns, a
scope's prefix, the held scopes, epochs as text, every table as Arrow IPC stream bytes, and back the tables as IPC bytes
and the flags as numpy ``bool`` bytes. Each ``*_job`` is the worker's entry; its pure core (the same name without
``_job``) runs on the caller's thread for input under :data:`INLINE_KEYS`, where the round trip costs more than the work.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any, Final

from threetears.core.collections.delta_files import (
    DeltaFileUnreadableError,
    HeldDeltaRows,
    joined_key_texts,
    key_text_table,
    read_delta_table,
)
from threetears.core.collections.generation import split_generation_token
from threetears.core.utils.cpu_quota import bound_native_threads

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as pa_ipc

    # Arrow's own key kernels: a module of the Arrow extra, so imported with it
    from threetears.core.arrow_keys import distinct_keys, key_codes
except ImportError:  # NOSILENT: the snapshot extra is optional; a copy follower says so when built
    pa = None

__all__ = [
    "HEARD_INLINE_TEXT_KEYS",
    "INLINE_KEYS",
    "cleared_reach_job",
    "cleared_reach_of",
    "epoch_plan_answer",
    "epoch_plan_job",
    "epoch_plan_of",
    "from_ipc",
    "heard_job",
    "heard_of",
    "heard_schema",
    "heard_scopes_job",
    "heard_scopes_of",
    "held_by_full",
    "held_scopes_answer",
    "held_scopes_job",
    "held_scopes_of",
    "ipc_bytes",
    "joined",
    "joined_ipc",
    "leaving_keys_job",
    "leaving_keys_of",
    "merged_answer",
    "merged_job",
    "newest_by_scope_job",
    "newest_by_scope_of",
    "page_scopes_answer",
    "page_scopes_job",
    "page_scopes_of",
    "partition_mine_job",
    "partition_mine_of",
    "partition_page_job",
    "partition_page_of",
    "past_floor_job",
    "past_floor_of",
    "past_states_job",
    "stage_cut",
    "staged_of",
    "staged_scopes_job",
    "taken_answer",
    "taken_job",
    "taken_of",
    "touched_job",
    "touched_of",
    "whole_page_answer",
    "whole_page_job",
    "whole_page_of",
    "without_keys_answer",
    "without_keys_job",
    "without_keys_of",
]

#: a heard batch's epoch column
_EPOCH: Final = "__epoch"

#: a key's scope column, beside its key columns
_SCOPE: Final = "__scope"

#: a pair's key's position among a pass's keys
_AT: Final = "__at"

#: the scope of a table of no scope while scopes are coded (its one state is named None): no scope is this text
_NO_SCOPE: Final = "\x00no scope"

#: below this many keys the work runs on the caller's thread: the worker's round trip (the IPC written, pickled through
#: the pool's pipe and read back) costs more than the work. Measured 2026-11-25 (macOS arm64, Python 3.14, a loaded
#: host): a 200,000-key page's heard batch took 3-8 ms on a thread and 25-144 ms through the worker (12-30 ms of it the
#: serving process's CPU); its page scopes 196-290 ms on a thread (all of it the serving process's CPU) and 342-656 ms
#: through the worker (43-74 ms of it the serving process's CPU);
#: the jobs added since keep the bound, each measured against it the same day: at 2,000 keys the work here took
#: 0.43 ms (a no-scope table's touched keys made distinct), 0.33 ms (a cleared prefix's reach), 0.28 ms (a whole read's
#: page with a scope held), 0.05 ms (a partition's page) of the loop's CPU, while the worker's round trip cost the
#: serving process 0.6-0.8 ms of CPU however small the job; past it the work grows with the keys (15 ms, 2.3 ms, 8.5 ms
#: at 100,000) and the round trip barely does (1.7 ms, 1.7 ms, 0.9 ms)
INLINE_KEYS: Final = 2_000

#: a heard batch of text keys (no timestamp key column) is spelled inline on the loop up to this many keys, in the worker
#: past it. Why: its work is a cast that keeps text as it is and one column of the epoch, about 6 ns a key of the loop's
#: CPU (2026-11-25, macOS arm64, Python 3.14, 4 text key columns: 0.33 ms at 50,000 keys, 0.79 ms at 100,000, 1.24 ms
#: at 200,000), while the worker's round trip cost the serving process 12-30 ms of CPU at 200,000 (the IPC written and
#: the answer read back). Past 100,000 keys (under 1 ms) the turn would grow toward the tests' 10 ms bar on slower pool
#: hosts, so a larger batch still goes to the worker. A timestamp key column is spelled a distinct value at a time in
#: Python (key_text), so such a batch goes to the worker past :data:`INLINE_KEYS`
HEARD_INLINE_TEXT_KEYS: Final = 100_000


def _one_cpu() -> bool:
    """bound this worker's Arrow pools to one CPU (each job: setting a pool to its own size changes nothing): the copy's
    worker is one process doing one job at a time, and pyarrow sizes its pools from the machine's cores (64 compute threads a worker on a 64-core host, each
    counted against the pod's thread limit: "can't start new thread" on the pool host monad-1, 2026-11-25).

    :return: whether pyarrow was there to bound
    :rtype: bool
    """
    return bound_native_threads(1)


def ipc_bytes(table: Any) -> bytes:
    """a table as Arrow IPC stream bytes.

    :param table: the table
    :ptype table: pyarrow.Table
    :return: the bytes
    :rtype: bytes
    """
    sink = pa.BufferOutputStream()
    with pa_ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return bytes(sink.getvalue())


def from_ipc(data: bytes) -> Any:
    """a table from Arrow IPC stream bytes (its buffers the bytes', not copied).

    :param data: the bytes
    :ptype data: bytes
    :return: the table
    :rtype: pyarrow.Table
    """
    return pa_ipc.open_stream(pa.py_buffer(data)).read_all()


def _flags(flags: Any) -> bytes:
    return np.asarray(flags, dtype=bool).tobytes()


def _unflags(data: bytes) -> Any:
    return np.frombuffer(data, dtype=bool) if data else None


def _flags_in(values: Any, among: Any) -> Any:
    """whether each value is among ``among``, as numpy flags (a null is not).

    :param values: the values
    :ptype values: pyarrow.Array | pyarrow.ChunkedArray
    :param among: the values looked for
    :ptype among: pyarrow.Array
    :return: one flag a value
    :rtype: numpy.ndarray
    """
    found = pc.fill_null(pc.is_in(values, value_set=among), False)
    return np.asarray(found.to_numpy(zero_copy_only=False), dtype=bool)


def _texts_of(column: Any) -> Any:
    """a column's values as text, a null as ``None`` (``str(value)``'s spelling of each, as a scope is named).

    :param column: the column
    :ptype column: pyarrow.Array | pyarrow.ChunkedArray
    :return: the texts
    :rtype: pyarrow.Array
    """
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    texts = column if pa.types.is_string(column.type) else column.cast(pa.string())
    return pc.fill_null(texts, "None")


def _scope_texts(prefix: str, column: Any) -> Any:
    """each value of a scope column as the scope a row holding it belongs to: ``prefix`` and the value's text.

    :param prefix: the scope's prefix (``state:``)
    :ptype prefix: str
    :param column: the scope column's values
    :ptype column: pyarrow.Array | pyarrow.ChunkedArray
    :return: the scopes
    :rtype: pyarrow.Array
    """
    return pc.binary_join_element_wise(pa.scalar(prefix), _texts_of(column), "")


def heard_of(key: Sequence[str], instants: Collection[str], keys: Any, epoch: str | None) -> Any:
    """keys heard at one epoch, as one immutable batch: each key column as text (a timestamp key column's in its
    canonical text), and the epoch.

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param instants: its timestamp key columns
    :ptype instants: Collection[str]
    :param keys: one row a key, its key columns' values, in key order (by position)
    :ptype keys: pyarrow.Table
    :param epoch: the epoch they were heard at; None when heard with none
    :ptype epoch: str | None
    :return: the batch
    :rtype: pyarrow.Table
    """
    count = keys.num_rows
    named = pa.table({column: keys.column(at) for at, column in enumerate(key)})
    texts = key_text_table(named, key, instants)
    columns = [texts.column(column).combine_chunks() for column in key]
    epochs = pa.nulls(count, pa.string()) if epoch is None else pa.repeat(pa.scalar(epoch, pa.string()), count)
    return pa.Table.from_arrays([*columns, epochs], schema=heard_schema(key))


def heard_job(
    key: tuple[str, ...], instants: frozenset[str], keys: tuple[bytes, ...], epochs: tuple[str | None, ...]
) -> tuple[bytes, ...]:
    """:func:`heard_of` of each batch of keys at its epoch: the worker's job.

    :param key: the table's key columns
    :ptype key: tuple[str, ...]
    :param instants: its timestamp key columns
    :ptype instants: frozenset[str]
    :param keys: each batch's keys, IPC
    :ptype keys: tuple[bytes, ...]
    :param epochs: each batch's epoch
    :ptype epochs: tuple[str | None, ...]
    :return: each heard batch, IPC
    :rtype: tuple[bytes, ...]
    """
    _one_cpu()
    return tuple(
        ipc_bytes(heard_of(key, instants, from_ipc(batch), epoch)) for batch, epoch in zip(keys, epochs, strict=True)
    )


def _reached_of(key: Sequence[str], instants: Collection[str], held: Sequence[Any], known: Any, epoch: str) -> Any:
    """the keys a cleared prefix reaches, each once, as the batch heard at the clearing epoch: the keys of the copy's
    rows under it, and those heard or staged under it.

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param instants: its timestamp key columns
    :ptype instants: Collection[str]
    :param held: the copy's rows under the prefix (its key columns, as the copy holds them), in batches
    :ptype held: Sequence[pyarrow.Table]
    :param known: the keys heard or staged under it, key columns as text
    :ptype known: pyarrow.Table
    :param epoch: the clearing epoch
    :ptype epoch: str
    :return: the batch
    :rtype: pyarrow.Table
    """
    reached = [key_text_table(rows, key, instants) for rows in held] + [known.select(list(key))]
    keys = distinct_keys(pa.concat_tables(reached), list(key))
    return heard_of(key, instants, keys, epoch)


def held_scopes_of(
    key: Sequence[str], instants: Collection[str], scope: tuple[str, str], page: Any, held: Any, scopes: Any
) -> tuple[Any, Any]:
    """the rows the copy holds of a page's keys as each key and its scope, and which keys of the page are staged now (a
    key whose held row is of a scope a write holds is left out).

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param instants: its timestamp key columns
    :ptype instants: Collection[str]
    :param scope: the scope column and its prefix
    :ptype scope: tuple[str, str]
    :param page: the page's keys, key columns as text
    :ptype page: pyarrow.Table
    :param held: the copy's rows of them: key and scope columns as the copy holds them
    :ptype held: pyarrow.Table
    :param scopes: the scopes a write in progress holds; None for none
    :ptype scopes: pyarrow.Array | None
    :return: the held rows' keys as text and ``__scope``, and per key of the page whether it is staged now (None:
        every one is)
    :rtype: tuple[pyarrow.Table, numpy.ndarray | None]
    """
    mine = key_text_table(held, key, instants).append_column(_SCOPE, _scope_texts(scope[1], held.column(scope[0])))
    if scopes is None or not mine.num_rows:
        return mine, None
    late = mine.filter(pa.array(_flags_in(mine.column(_SCOPE), scopes)))
    if not late.num_rows:
        return mine, None
    return mine, ~_flags_in(joined_key_texts(page, list(key)), joined_key_texts(late, list(key)))


def held_scopes_job(
    key: tuple[str, ...],
    instants: frozenset[str],
    scope: tuple[str, str],
    page: bytes,
    held: bytes,
    scopes: tuple[str, ...] | None,
) -> tuple[bytes, bytes]:
    """:func:`held_scopes_of`: the worker's job.

    :param key: the table's key columns
    :ptype key: tuple[str, ...]
    :param instants: its timestamp key columns
    :ptype instants: frozenset[str]
    :param scope: the scope column and its prefix
    :ptype scope: tuple[str, str]
    :param page: the page's keys, IPC
    :ptype page: bytes
    :param held: the copy's rows of them, IPC
    :ptype held: bytes
    :param scopes: the scopes a write in progress holds; None for none
    :ptype scopes: tuple[str, ...] | None
    :return: the held rows' keys and scopes, IPC; the flags, or empty when every key is staged
    :rtype: tuple[bytes, bytes]
    """
    _one_cpu()
    among = None if scopes is None else pa.array(scopes, pa.string())
    mine, keep = held_scopes_of(key, instants, scope, from_ipc(page), from_ipc(held), among)
    return ipc_bytes(mine), b"" if keep is None else _flags(keep)


def held_scopes_answer(answer: tuple[bytes, bytes]) -> tuple[Any, Any]:
    """:func:`held_scopes_job`'s answer read back.

    :param answer: the answer
    :ptype answer: tuple[bytes, bytes]
    :return: as :func:`held_scopes_of`
    :rtype: tuple[pyarrow.Table, numpy.ndarray | None]
    """
    return from_ipc(answer[0]), _unflags(answer[1])


def page_scopes_of(
    key: Sequence[str],
    instants: Collection[str],
    scope: tuple[str, str],
    chunk: Any,
    found: Any,
    mine: Any,
    scopes: Any,
) -> tuple[Any, Any, Any]:
    """a page's keys' scopes, the row the copy holds and the row read now (a delete's, a move's) each, and which keys
    and rows stay staged: a key whose new row is of a scope a write holds waits.

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param instants: its timestamp key columns
    :ptype instants: Collection[str]
    :param scope: the scope column and its prefix
    :ptype scope: tuple[str, str]
    :param chunk: the page's keys staged now, key columns as text
    :ptype chunk: pyarrow.Table
    :param found: the rows read now: key and scope columns, as read
    :ptype found: pyarrow.Table
    :param mine: the rows the copy holds of the page (:func:`held_scopes_of`)
    :ptype mine: pyarrow.Table
    :param scopes: the scopes a write in progress holds; None for none
    :ptype scopes: pyarrow.Array | None
    :return: the keys with each row's scope (``__scope``), and per key of the chunk and per row read whether it stays
        (None: every one does)
    :rtype: tuple[pyarrow.Table, numpy.ndarray | None, numpy.ndarray | None]
    """
    columns = list(key)
    found_texts = key_text_table(found, columns, instants)
    found_scoped = found_texts.append_column(_SCOPE, _scope_texts(scope[1], found.column(scope[0])))
    both = distinct_keys(pa.concat_tables([mine, found_scoped]), [*columns, _SCOPE])
    chunk_texts = joined_key_texts(chunk, columns)
    both = both.filter(pa.array(_flags_in(joined_key_texts(both, columns), chunk_texts)))
    if scopes is None:
        return both, None, None
    left = both.filter(pa.array(_flags_in(both.column(_SCOPE), scopes)))
    if not left.num_rows:
        return both, None, None
    left_texts = joined_key_texts(left, columns)
    stays = ~_flags_in(chunk_texts, left_texts)
    rows_stay = ~_flags_in(joined_key_texts(found_texts, columns), left_texts)
    return both.filter(pa.array(~_flags_in(joined_key_texts(both, columns), left_texts))), stays, rows_stay


def page_scopes_job(
    key: tuple[str, ...],
    instants: frozenset[str],
    scope: tuple[str, str],
    chunk: bytes,
    found: bytes,
    mine: bytes,
    scopes: tuple[str, ...] | None,
) -> tuple[bytes, bytes, bytes]:
    """:func:`page_scopes_of`: the worker's job.

    :param key: the table's key columns
    :ptype key: tuple[str, ...]
    :param instants: its timestamp key columns
    :ptype instants: frozenset[str]
    :param scope: the scope column and its prefix
    :ptype scope: tuple[str, str]
    :param chunk: the page's keys staged now, IPC
    :ptype chunk: bytes
    :param found: the rows read now (key and scope columns), IPC
    :ptype found: bytes
    :param mine: the rows the copy holds of the page, IPC
    :ptype mine: bytes
    :param scopes: the scopes a write in progress holds; None for none
    :ptype scopes: tuple[str, ...] | None
    :return: the keys with their scopes, IPC; the chunk's and the rows' flags, each empty when every one stays
    :rtype: tuple[bytes, bytes, bytes]
    """
    _one_cpu()
    among = None if scopes is None else pa.array(scopes, pa.string())
    both, stays, rows_stay = page_scopes_of(
        key, instants, scope, from_ipc(chunk), from_ipc(found), from_ipc(mine), among
    )
    return ipc_bytes(both), b"" if stays is None else _flags(stays), b"" if rows_stay is None else _flags(rows_stay)


def page_scopes_answer(answer: tuple[bytes, bytes, bytes]) -> tuple[Any, Any, Any]:
    """:func:`page_scopes_job`'s answer read back.

    :param answer: the answer
    :ptype answer: tuple[bytes, bytes, bytes]
    :return: as :func:`page_scopes_of`
    :rtype: tuple[pyarrow.Table, numpy.ndarray | None, numpy.ndarray | None]
    """
    return from_ipc(answer[0]), _unflags(answer[1]), _unflags(answer[2])


def heard_schema(key: Sequence[str]) -> Any:
    """the schema of a batch of keys heard: each key column as text, named as the key, and the epoch.

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :return: the schema
    :rtype: pyarrow.Schema
    """
    return pa.schema([(column, pa.string()) for column in key] + [(_EPOCH, pa.string())])


def joined(batches: Sequence[Any], schema: Any) -> Any:
    """some batches as one table (their chunks, not copied); the schema's empty table when there are none.

    :param batches: the batches, each of ``schema``'s columns at least
    :ptype batches: Sequence[pyarrow.Table]
    :param schema: what each holds
    :ptype schema: pyarrow.Schema
    :return: the table
    :rtype: pyarrow.Table
    """
    if not batches:
        return schema.empty_table()
    return pa.concat_tables([batch.select(schema.names).cast(schema) for batch in batches])


def joined_ipc(batches: Sequence[Any], schema: Any) -> bytes:
    """some batches as one Arrow IPC stream, for the copy's worker process: their chunks joined (nothing copied) and
    written whole, one record batch a chunk. A copy of their buffers into the stream: the hand-off, so the caller runs
    it on a thread (IO, as :func:`ipc_bytes`).

    :param batches: the batches
    :ptype batches: Sequence[pyarrow.Table]
    :param schema: what each holds
    :ptype schema: pyarrow.Schema
    :return: the bytes
    :rtype: bytes
    """
    return ipc_bytes(joined(batches, schema))


def _later_than(rows: Any, key: Sequence[str], bar: str | None) -> Any:
    """per row of keys heard at their epochs, whether its key was heard at an epoch past ``bar`` (the table's generation
    as read before the pass's stamp): every row of such a key, so a key waits whole. One judgment a distinct epoch (a
    commit), never one a key; an epoch of another incarnation, or none, is never past it (nothing orders them).

    :param rows: the keys at their epochs (key columns as text, ``__epoch``)
    :ptype rows: pyarrow.Table
    :param key: the key's columns
    :ptype key: Sequence[str]
    :param bar: the generation read before the stamp; None for none
    :ptype bar: str | None
    :return: the flags, or None when no row is past it
    :rtype: numpy.ndarray | None
    """
    split = split_generation_token(bar) if bar is not None else None
    if split is None or not rows.num_rows:
        return None
    epochs = rows.column(_EPOCH)
    later = []
    for epoch in pc.unique(pc.drop_null(epochs)).to_pylist():  # one a distinct epoch (a commit), never one a key
        parsed = split_generation_token(epoch)
        if parsed is not None and parsed[0] == split[0] and parsed[1] > split[1]:
            later.append(epoch)
    if not later:
        return None
    newer = rows.filter(pa.array(_flags_in(epochs, pa.array(later, pa.string()))))
    return _flags_in(joined_key_texts(rows, list(key)), joined_key_texts(newer, list(key)))


def taken_of(
    key: Sequence[str],
    scope_at: tuple[str, int] | None,
    delta_files: bool,
    heard: Any,
    staged: Any,
    held: Any,
    bar: str | None,
) -> tuple[Any, Any, Any, int, bool]:
    """a pass's keys of one table from its batches heard (joined): each key at each epoch once, those of a scope a
    write holds left pending (when the key names its scope), and for a table that opts in to delta files, every epoch
    a key was staged from by an earlier pass beside the new ones, so its row is merged again from every change not yet
    applied.

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param scope_at: the scope's prefix and the key column naming it, when the key names it; None otherwise
    :ptype scope_at: tuple[str, int] | None
    :param delta_files: whether the table opts in to delta files
    :ptype delta_files: bool
    :param heard: the batches heard, joined (key columns as text, ``__epoch``)
    :ptype heard: pyarrow.Table
    :param staged: the keys at their epochs earlier passes staged and no apply has taken yet, joined; None for none
    :ptype staged: pyarrow.Table | None
    :param held: the scopes a write in progress holds; None for none
    :ptype held: pyarrow.Array | None
    :param bar: the table's generation read before the pass's stamp; None for none
    :ptype bar: str | None
    :return: the keys (each once, in key order), the pairs (each key at each epoch, ``__at`` its key's position, in
        that order), the pairs left pending (None for none), how many keys they hold, and whether some were heard past
        ``bar``
    :rtype: tuple[pyarrow.Table, pyarrow.Table, pyarrow.Table | None, int, bool]
    """
    key = list(key)
    rows = distinct_keys(heard, [*key, _EPOCH])
    left = None
    left_keys = 0
    if held is not None and scope_at is not None:
        hold = pa.array(_flags_in(_scope_texts(scope_at[0], rows.column(key[scope_at[1]])), held))
        left = rows.filter(hold)
        rows = rows.filter(pc.invert(hold))
        left_keys = distinct_keys(left, key).num_rows
        left = left if left.num_rows else None
    later = _later_than(rows, key, bar)
    held_later = later is not None and bool(later.any())
    if held_later:
        # a key heard at an epoch past the generation read before the pass's stamp: its commit may be newer than the
        # stamp, so every epoch of it waits for the next pass, whose stamp holds it (one read, never read and redone)
        late = rows.filter(pa.array(later))
        rows = rows.filter(pa.array(~later))
        left_keys += distinct_keys(late, key).num_rows
        left = late if left is None else pa.concat_tables([left, late])
    if staged is not None and staged.num_rows and rows.num_rows:
        # a key heard again at an epoch an earlier pass staged it at (its row broadcast and its epoch's record, one
        # before the pass and one after) is staged already: read again, it was read twice for one change (the live
        # refresh test, 2 runs in 6, 2026-11-24). Each record's keys are read once; a key heard with no epoch has none
        # to match and is read as before
        again = pc.and_(
            pc.is_valid(rows.column(_EPOCH)),
            pa.array(_flags_in(joined_key_texts(rows, [*key, _EPOCH]), joined_key_texts(staged, [*key, _EPOCH]))),
        )
        rows = rows.filter(pc.invert(pc.fill_null(again, False)))
        if delta_files and rows.num_rows:
            bring = staged.filter(pa.array(_flags_in(joined_key_texts(staged, key), joined_key_texts(rows, key))))
            if bring.num_rows:
                rows = distinct_keys(pa.concat_tables([rows, bring]), [*key, _EPOCH])
    keys = distinct_keys(rows, key).sort_by([(column, "ascending") for column in key])
    positions = pc.index_in(joined_key_texts(rows, key), value_set=joined_key_texts(keys, key)).cast(pa.int64())
    pairs = rows.append_column(_AT, positions).sort_by([(_AT, "ascending")])
    return keys, pairs, left, left_keys, held_later


def taken_job(
    key: tuple[str, ...],
    scope_at: tuple[str, int] | None,
    delta_files: bool,
    heard: bytes,
    staged: bytes,
    held: tuple[str, ...] | None,
    bar: str | None,
) -> tuple[bytes, bytes, bytes, int, bool]:
    """:func:`taken_of`: the worker's job.

    :param key: the table's key columns
    :ptype key: tuple[str, ...]
    :param scope_at: the scope's prefix and the key column naming it; None when the key names none
    :ptype scope_at: tuple[str, int] | None
    :param delta_files: whether the table opts in to delta files
    :ptype delta_files: bool
    :param heard: the batches heard, joined, IPC
    :ptype heard: bytes
    :param staged: the keys earlier passes staged, joined, IPC; empty for none
    :ptype staged: bytes
    :param held: the scopes a write in progress holds; None for none
    :ptype held: tuple[str, ...] | None
    :param bar: the table's generation read before the pass's stamp; None for none
    :ptype bar: str | None
    :return: the keys and the pairs, IPC; the pairs left pending, IPC (empty for none); their keys' count; whether
        some were heard past ``bar``
    :rtype: tuple[bytes, bytes, bytes, int, bool]
    """
    _one_cpu()
    among = None if held is None else pa.array(held, pa.string())
    keys, pairs, left, left_keys, later = taken_of(
        key, scope_at, delta_files, from_ipc(heard), from_ipc(staged) if staged else None, among, bar
    )
    return ipc_bytes(keys), ipc_bytes(pairs), b"" if left is None else ipc_bytes(left), left_keys, later


def taken_answer(answer: tuple[bytes, bytes, bytes, int, bool]) -> tuple[Any, Any, Any, int, bool]:
    """:func:`taken_job`'s answer read back.

    :param answer: the answer
    :ptype answer: tuple[bytes, bytes, bytes, int, bool]
    :return: as :func:`taken_of`
    :rtype: tuple[pyarrow.Table, pyarrow.Table, pyarrow.Table | None, int, bool]
    """
    keys, pairs, left, left_keys, later = answer
    return from_ipc(keys), from_ipc(pairs), from_ipc(left) if left else None, left_keys, later


def under_of(key: Sequence[str], rows: Any, prefixes: Sequence[Sequence[tuple[str, str]]]) -> Any:
    """the keys of ``rows`` (keys at epochs, as text) under one of ``prefixes``, each once (compared as text).

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param rows: the keys heard or staged, joined
    :ptype rows: pyarrow.Table
    :param prefixes: the prefixes, each its columns' values as ``(column, value)`` pairs
    :ptype prefixes: Sequence[Sequence[tuple[str, str]]]
    :return: the keys, key columns as text
    :rtype: pyarrow.Table
    """
    under = np.zeros(rows.num_rows, dtype=bool)
    for prefix in prefixes:
        within = np.ones(rows.num_rows, dtype=bool)
        for column, value in prefix:
            equal = pc.fill_null(pc.equal(rows.column(column), pa.scalar(value, pa.string())), False)
            within &= np.asarray(equal.to_numpy(zero_copy_only=False), dtype=bool)
        under |= within
    return distinct_keys(rows.filter(pa.array(under)), list(key))


def cleared_reach_of(
    key: Sequence[str],
    instants: Collection[str],
    held: Sequence[Any],
    known: Any,
    prefixes: Sequence[Sequence[tuple[str, str]]],
    epoch: str,
) -> Any:
    """the keys cleared prefixes reach, each once, as the batch heard at the clearing epoch: the keys of the copy's rows
    under them, and those heard or staged under them (:func:`under_of`, :func:`_reached_of`).

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param instants: its timestamp key columns
    :ptype instants: Collection[str]
    :param held: the copy's rows under the prefixes (its key columns, as the copy holds them), in batches
    :ptype held: Sequence[pyarrow.Table]
    :param known: the keys heard or staged, joined (key columns as text, ``__epoch``)
    :ptype known: pyarrow.Table
    :param prefixes: the prefixes, each its columns' values as ``(column, value)`` pairs
    :ptype prefixes: Sequence[Sequence[tuple[str, str]]]
    :param epoch: the clearing epoch
    :ptype epoch: str
    :return: the batch
    :rtype: pyarrow.Table
    """
    return _reached_of(key, instants, held, under_of(key, known, prefixes), epoch)


def cleared_reach_job(
    key: tuple[str, ...],
    instants: frozenset[str],
    held: tuple[bytes, ...],
    known: bytes,
    prefixes: tuple[tuple[tuple[str, str], ...], ...],
    epoch: str,
) -> bytes:
    """:func:`cleared_reach_of`: the worker's job.

    :param key: the table's key columns
    :ptype key: tuple[str, ...]
    :param instants: its timestamp key columns
    :ptype instants: frozenset[str]
    :param held: the copy's rows under the prefixes, IPC
    :ptype held: tuple[bytes, ...]
    :param known: the keys heard or staged, joined, IPC
    :ptype known: bytes
    :param prefixes: the prefixes
    :ptype prefixes: tuple[tuple[tuple[str, str], ...], ...]
    :param epoch: the clearing epoch
    :ptype epoch: str
    :return: the batch, IPC
    :rtype: bytes
    """
    _one_cpu()
    return ipc_bytes(
        cleared_reach_of(key, instants, [from_ipc(rows) for rows in held], from_ipc(known), prefixes, epoch)
    )


def touched_of(key: Sequence[str], keys: Any) -> Any:
    """the keys an apply replaced in a table of no scope, each once: what a checkpoint patches its one part by.

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param keys: the keys staged, joined (key columns as text)
    :ptype keys: pyarrow.Table
    :return: the keys
    :rtype: pyarrow.Table
    """
    return distinct_keys(keys, list(key))


def touched_job(key: tuple[str, ...], keys: bytes) -> bytes:
    """:func:`touched_of`: the worker's job.

    :param key: the table's key columns
    :ptype key: tuple[str, ...]
    :param keys: the keys staged, joined, IPC
    :ptype keys: bytes
    :return: the keys, IPC
    :rtype: bytes
    """
    _one_cpu()
    return ipc_bytes(touched_of(key, from_ipc(keys)))


def _scopes_of(prefix: str | None, column: Any, count: int) -> Any:
    """each row's scope: ``prefix`` and its scope column's text; nulls for a table of no scope.

    :param prefix: the scope's prefix; None for a table of no scope
    :ptype prefix: str | None
    :param column: the scope column's values; None for a table of no scope
    :ptype column: pyarrow.ChunkedArray | None
    :param count: the rows
    :ptype count: int
    :return: the scopes
    :rtype: pyarrow.Array
    """
    if prefix is None or column is None:
        return pa.nulls(count, pa.string())
    return _scope_texts(prefix, column)


def _distinct_scopes(scopes: Any) -> tuple[str | None, ...]:
    """the distinct scopes of a column of them, sorted (a null first): one a state, never one a row.

    :param scopes: the scopes
    :ptype scopes: pyarrow.Array
    :return: the scopes
    :rtype: tuple[str | None, ...]
    """
    found = pc.unique(scopes).to_pylist()  # one a distinct scope (a state), never one a row
    return tuple(sorted(found, key=lambda scope: (scope is not None, scope or "")))


def whole_page_of(
    scope: tuple[str, str] | None, page: Any, held: Any
) -> tuple[Any, tuple[str | None, ...], tuple[tuple[str, int], ...]]:
    """a page of a whole read judged by its scope column: which rows stay (a row of a scope a write holds is left out;
    None: every row stays), the distinct scopes of the rows that stay, and how many rows of each held scope were left
    out. The page itself is filtered by the caller: a copy of the rows kept, memory, not a pass of judgment.

    :param scope: the scope column and its prefix; None for a table of no scope
    :ptype scope: tuple[str, str] | None
    :param page: the page (its scope column at least)
    :ptype page: pyarrow.Table
    :param held: the scopes a write in progress holds; None for none
    :ptype held: pyarrow.Array | None
    :return: the flags or None, the scopes, the held scopes' counts
    :rtype: tuple[numpy.ndarray | None, tuple[str | None, ...], tuple[tuple[str, int], ...]]
    """
    if scope is None:
        return None, (None,) if page.num_rows else (), ()
    of = _scope_texts(scope[1], page.column(scope[0]))
    if held is None:
        return None, _distinct_scopes(of) if page.num_rows else (), ()
    hit = _flags_in(of, held)
    if not hit.any():
        return None, _distinct_scopes(of) if page.num_rows else (), ()
    # one count a held scope (a state), never one a row
    counts = tuple(
        (found["values"], found["counts"]) for found in pc.value_counts(of.filter(pa.array(hit))).to_pylist()
    )
    stays = ~hit
    kept = of.filter(pa.array(stays))
    return stays, _distinct_scopes(kept) if len(kept) else (), counts


def whole_page_job(
    scope: tuple[str, str], page: bytes, held: tuple[str, ...] | None
) -> tuple[bytes, tuple[str | None, ...], tuple[tuple[str, int], ...]]:
    """:func:`whole_page_of`: the worker's job.

    :param scope: the scope column and its prefix
    :ptype scope: tuple[str, str]
    :param page: the page's scope column, IPC
    :ptype page: bytes
    :param held: the scopes a write in progress holds; None for none
    :ptype held: tuple[str, ...] | None
    :return: the flags (empty: every row stays), the scopes, the held scopes' counts
    :rtype: tuple[bytes, tuple[str | None, ...], tuple[tuple[str, int], ...]]
    """
    _one_cpu()
    among = None if held is None else pa.array(held, pa.string())
    stays, scopes, counts = whole_page_of(scope, from_ipc(page), among)
    return b"" if stays is None else _flags(stays), scopes, counts


def whole_page_answer(
    answer: tuple[bytes, tuple[str | None, ...], tuple[tuple[str, int], ...]],
) -> tuple[Any, tuple[str | None, ...], tuple[tuple[str, int], ...]]:
    """:func:`whole_page_job`'s answer read back.

    :param answer: the answer
    :ptype answer: tuple[bytes, tuple[str | None, ...], tuple[tuple[str, int], ...]]
    :return: as :func:`whole_page_of`
    :rtype: tuple[numpy.ndarray | None, tuple[str | None, ...], tuple[tuple[str, int], ...]]
    """
    return _unflags(answer[0]), answer[1], answer[2]


def partition_page_of(
    key: Sequence[str], instants: Collection[str], scope: tuple[str, str] | None, page: Any, held: Any
) -> tuple[bool, tuple[str | None, ...], Any]:
    """a page of a partition's L3 rows judged: whether a row is of a scope a write holds (the partition waits), the
    distinct scopes of its rows, and for a table of no scope the keys it replaces, as text.

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param instants: its timestamp key columns
    :ptype instants: Collection[str]
    :param scope: the scope column and its prefix; None for a table of no scope
    :ptype scope: tuple[str, str] | None
    :param page: the page's key and scope columns
    :ptype page: pyarrow.Table
    :param held: the scopes a write in progress holds; None for none
    :ptype held: pyarrow.Array | None
    :return: whether it clashed, the scopes, the keys replaced (None for a scoped table or a clash)
    :rtype: tuple[bool, tuple[str | None, ...], pyarrow.Table | None]
    """
    of = _scopes_of(scope[1] if scope else None, page.column(scope[0]) if scope else None, page.num_rows)
    if held is not None and _flags_in(of, held).any():
        return True, (), None
    replaced = key_text_table(page, key, instants) if scope is None else None
    return False, _distinct_scopes(of) if page.num_rows else (), replaced


def partition_page_job(
    key: tuple[str, ...],
    instants: frozenset[str],
    scope: tuple[str, str] | None,
    page: bytes,
    held: tuple[str, ...] | None,
) -> tuple[bool, tuple[str | None, ...], bytes]:
    """:func:`partition_page_of`: the worker's job.

    :param key: the table's key columns
    :ptype key: tuple[str, ...]
    :param instants: its timestamp key columns
    :ptype instants: frozenset[str]
    :param scope: the scope column and its prefix; None for a table of no scope
    :ptype scope: tuple[str, str] | None
    :param page: the page's key and scope columns, IPC
    :ptype page: bytes
    :param held: the scopes a write in progress holds; None for none
    :ptype held: tuple[str, ...] | None
    :return: whether it clashed, the scopes, the keys replaced, IPC (empty for none)
    :rtype: tuple[bool, tuple[str | None, ...], bytes]
    """
    _one_cpu()
    among = None if held is None else pa.array(held, pa.string())
    clashed, scopes, replaced = partition_page_of(key, instants, scope, from_ipc(page), among)
    return clashed, scopes, b"" if replaced is None else ipc_bytes(replaced)


def partition_mine_of(
    key: Sequence[str],
    instants: Collection[str],
    scope: tuple[str, str] | None,
    last: str,
    mine: Any,
    held: Any,
) -> tuple[tuple[str, ...], Any, tuple[str | None, ...], Any]:
    """the copy's own rows of some partitions (one prefix shape) judged: the values of the prefix's last column a row
    of a scope a write holds sits under (those partitions wait for the write), the rows kept (their key columns, as
    the copy holds them), their distinct scopes, and for a table of no scope their keys as text.

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param instants: its timestamp key columns
    :ptype instants: Collection[str]
    :param scope: the scope column and its prefix; None for a table of no scope
    :ptype scope: tuple[str, str] | None
    :param last: the prefix's last column
    :ptype last: str
    :param mine: the copy's rows under the prefixes: key and scope columns, as the copy holds them
    :ptype mine: pyarrow.Table
    :param held: the scopes a write in progress holds; None for none
    :ptype held: pyarrow.Array | None
    :return: the late values (sorted), the rows kept (key columns), the scopes, the keys replaced (None for a scoped
        table)
    :rtype: tuple[tuple[str, ...], pyarrow.Table, tuple[str | None, ...], pyarrow.Table | None]
    """
    of = _scopes_of(scope[1] if scope else None, mine.column(scope[0]) if scope else None, mine.num_rows)
    late: tuple[str, ...] = ()
    if held is not None and mine.num_rows:
        hit = _flags_in(of, held)
        if hit.any():
            values = key_text_table(mine, [last], instants).column(last)
            # the prefix values a held scope's rows sit under: one a prefix of the batch, never one a row
            late = tuple(sorted(pc.unique(values.filter(pa.array(hit))).to_pylist()))
            stays = pa.array(~_flags_in(values, pa.array(late, pa.string())))
            mine = mine.filter(stays)
            of = of.filter(stays)
    replaced = key_text_table(mine, key, instants) if scope is None and mine.num_rows else None
    return late, mine.select(list(key)), _distinct_scopes(of) if mine.num_rows else (), replaced


def partition_mine_job(
    key: tuple[str, ...],
    instants: frozenset[str],
    scope: tuple[str, str] | None,
    last: str,
    mine: bytes,
    held: tuple[str, ...] | None,
) -> tuple[tuple[str, ...], bytes, tuple[str | None, ...], bytes]:
    """:func:`partition_mine_of`: the worker's job.

    :param key: the table's key columns
    :ptype key: tuple[str, ...]
    :param instants: its timestamp key columns
    :ptype instants: frozenset[str]
    :param scope: the scope column and its prefix; None for a table of no scope
    :ptype scope: tuple[str, str] | None
    :param last: the prefix's last column
    :ptype last: str
    :param mine: the copy's rows under the prefixes, IPC
    :ptype mine: bytes
    :param held: the scopes a write in progress holds; None for none
    :ptype held: tuple[str, ...] | None
    :return: the late values, the rows kept (IPC), the scopes, the keys replaced (IPC, empty for none)
    :rtype: tuple[tuple[str, ...], bytes, tuple[str | None, ...], bytes]
    """
    _one_cpu()
    among = None if held is None else pa.array(held, pa.string())
    late, kept, scopes, replaced = partition_mine_of(key, instants, scope, last, from_ipc(mine), among)
    return late, ipc_bytes(kept), scopes, b"" if replaced is None else ipc_bytes(replaced)


def _positions(data: bytes) -> Any:
    """positions read back from their bytes (int64).

    :param data: the bytes
    :ptype data: bytes
    :return: the positions
    :rtype: numpy.ndarray
    """
    return np.frombuffer(data, dtype=np.int64)


def _bools(data: bytes) -> Any:
    """flags read back from their bytes, an empty run for none (never None, as :func:`_unflags` reads them).

    :param data: the bytes
    :ptype data: bytes
    :return: the flags
    :rtype: numpy.ndarray
    """
    return np.frombuffer(data, dtype=bool)


def _ipc_or_empty(table: Any) -> bytes:
    """a table as IPC bytes; empty bytes for None.

    :param table: the table, or None
    :ptype table: pyarrow.Table | None
    :return: the bytes
    :rtype: bytes
    """
    return b"" if table is None else ipc_bytes(table)


def _table_or_none(data: bytes) -> Any:
    """a table from IPC bytes; None for empty bytes.

    :param data: the bytes
    :ptype data: bytes
    :return: the table, or None
    :rtype: pyarrow.Table | None
    """
    return from_ipc(data) if data else None


def epoch_plan_of(pairs: Any, count: int) -> tuple[tuple[str, ...], Any, Any, Any]:
    """which keys each epoch heard names, oldest epoch first, and which keys no epoch orders (none, or two
    incarnations'): Arrow and numpy over a pass's pairs, never a Python object a key.

    :param pairs: the pass's keys at their epochs: ``__epoch`` and ``__at`` (each key's position among the keys)
    :ptype pairs: pyarrow.Table
    :param count: how many keys the pass took
    :ptype count: int
    :return: the epochs in order; every epoch's keys' positions, one run an epoch, in that order; where each run
        starts (and the last ends); per key, whether no epoch orders it
    :rtype: tuple[tuple[str, ...], numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    at = np.asarray(pairs.column(_AT).to_numpy(), dtype=np.int64)
    epochs = pairs.column(_EPOCH).combine_chunks()
    distinct = pc.unique(pc.drop_null(epochs))
    names = distinct.to_pylist()  # one a distinct epoch heard this pass (a commit), never one a key
    parsed = [split_generation_token(name) for name in names]
    incarnations = {incarnation: n for n, incarnation in enumerate(sorted({p[0] for p in parsed if p is not None}))}
    inc_of = np.array([incarnations[p[0]] if p is not None else -1 for p in parsed] + [-1], dtype=np.int64)
    code = np.asarray(pc.fill_null(pc.index_in(epochs, value_set=distinct), -1).to_numpy(), dtype=np.int64)
    row_inc = inc_of[code]  # a null epoch's code is -1: the trailing -1
    no_epoch = np.zeros(count, dtype=bool)
    no_epoch[at[row_inc < 0]] = True
    low = np.full(count, np.iinfo(np.int64).max, dtype=np.int64)
    high = np.full(count, -1, dtype=np.int64)
    np.minimum.at(low, at, row_inc)
    np.maximum.at(high, at, row_inc)
    # heard under two incarnations of the log, whose counts do not order them
    no_epoch |= low != high
    keep = ~no_epoch[at]
    codes, positions = code[keep], at[keep]
    by_code = np.argsort(codes, kind="stable")
    codes, positions = codes[by_code], positions[by_code]
    # function-local: the epochs in order and each one's run of positions, one an epoch (a commit), never one a key
    order: list[str] = []
    runs: list[Any] = []
    for _parsed, n in sorted((p, n) for n, p in enumerate(parsed) if p is not None):
        low_at, high_at = (int(edge) for edge in np.searchsorted(codes, [n, n + 1]))
        if high_at > low_at:
            order.append(names[n])
            runs.append(positions[low_at:high_at])
    bounds = np.zeros(len(runs) + 1, dtype=np.int64)
    bounds[1:] = np.cumsum([len(run) for run in runs], dtype=np.int64)
    joined_runs = np.concatenate(runs).astype(np.int64) if runs else np.zeros(0, dtype=np.int64)
    return tuple(order), joined_runs, bounds, no_epoch


def epoch_plan_job(pairs: bytes, count: int) -> tuple[tuple[str, ...], bytes, bytes, bytes]:
    """:func:`epoch_plan_of`: the worker's job.

    :param pairs: the pass's pairs (``__epoch``, ``__at``), IPC
    :ptype pairs: bytes
    :param count: how many keys the pass took
    :ptype count: int
    :return: the epochs, and the positions, the runs' bounds and the flags as bytes
    :rtype: tuple[tuple[str, ...], bytes, bytes, bytes]
    """
    _one_cpu()
    order, positions, bounds, no_epoch = epoch_plan_of(from_ipc(pairs), count)
    return order, positions.tobytes(), bounds.tobytes(), _flags(no_epoch)


def epoch_plan_answer(answer: tuple[tuple[str, ...], bytes, bytes, bytes]) -> tuple[tuple[str, ...], Any, Any, Any]:
    """:func:`epoch_plan_job`'s answer read back.

    :param answer: the answer
    :ptype answer: tuple[tuple[str, ...], bytes, bytes, bytes]
    :return: as :func:`epoch_plan_of`
    :rtype: tuple[tuple[str, ...], numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    order, positions, bounds, no_epoch = answer
    return order, _positions(positions), _positions(bounds), _bools(no_epoch)


def _merged_of(
    table: str,
    key: Sequence[str],
    instants: Collection[str],
    keys: Any,
    held: Any,
    plan: tuple[Sequence[str], Any, Any, Any],
    named: Sequence[tuple[str, str]],
    whys: Sequence[tuple[str, str]],
    root: str | None,
) -> tuple[tuple[str, str, str, str] | None, Any, Any, Any, tuple[tuple[str, int, tuple[str, ...]], ...], int]:
    """a pass's rows of one table that opts in to delta files: the rows the copy holds, each epoch's file read off the
    volume in turn (the one before let go first) and applied to the keys heard at it
    (:class:`~threetears.core.collections.delta_files.HeldDeltaRows`), as L3 merged them. A key one of whose epochs
    names no file (an older writer's), or heard with no epoch that orders it, is read from L3 by key instead (a
    fallback, counted by why).

    :param table: the table
    :ptype table: str
    :param key: its key columns
    :ptype key: Sequence[str]
    :param instants: its timestamp key columns
    :ptype instants: Collection[str]
    :param keys: the pass's keys, key columns as text, in key order
    :ptype keys: pyarrow.Table
    :param held: the rows the copy holds of the keys, typed as the rail answers them: its schema is the rows'
    :ptype held: pyarrow.Table
    :param plan: each epoch's keys (:func:`epoch_plan_of`)
    :ptype plan: tuple[Sequence[str], numpy.ndarray, numpy.ndarray, numpy.ndarray]
    :param named: each epoch whose record names a file, and the file
    :ptype named: Sequence[tuple[str, str]]
    :param whys: each epoch whose file cannot be named, and why
    :ptype whys: Sequence[tuple[str, str]]
    :param root: the delta files' directory; None when no volume is mounted
    :ptype root: str | None
    :return: what the table waits for instead (the file, its epoch, why, and the error) or None; the rows that exist
        after the change and each one's key's position, ascending; per key, whether it is read from L3 instead; the
        keys so read by why (their count and the files named); how many changes were applied
    :rtype: tuple[tuple[str, str, str, str] | None, pyarrow.Table | None, numpy.ndarray | None,
        numpy.ndarray | None, tuple[tuple[str, int, tuple[str, ...]], ...], int]
    """
    order, positions, bounds, no_epoch = plan
    count = keys.num_rows
    # function-local and read-only: each epoch's file or why it has none, one an epoch (a commit), never one a key
    files_of = dict(named)
    why_of = dict(whys)
    from_l3 = np.zeros(count, dtype=bool)
    # function-local: by why, the keys read from L3 and the files named
    fallbacks: dict[str, tuple[int, frozenset[str]]] = {}

    def fell(why: str, mine: Any) -> None:
        new = mine[~from_l3[mine]]
        if len(new):
            from_l3[new] = True
            had, files = fallbacks.get(why, (0, frozenset()))
            fallbacks[why] = (had + int(len(new)), files)

    fell("no_epoch", np.flatnonzero(no_epoch))
    for at, epoch in enumerate(order):
        if epoch in why_of:
            fell(why_of[epoch], positions[int(bounds[at]) : int(bounds[at + 1])])
    rows = HeldDeltaRows(key, keys, dict(zip(held.schema.names, held.schema.types, strict=True)), instants=instants)
    rows.hold(held)
    # function-local: by file, the keys its changes were applied to (a commit of several records names one file: its
    # changes of a key are applied once, at its first record)
    applied: dict[str, Any] = {}
    file: tuple[str, Any] | None = None
    for at, epoch in enumerate(order):
        if epoch in why_of:
            continue
        name = files_of[epoch]
        done = applied.setdefault(name, np.zeros(count, dtype=bool))
        mine = positions[int(bounds[at]) : int(bounds[at + 1])]
        mine = mine[~from_l3[mine] & ~done[mine]]
        if not len(mine):
            continue
        if file is None or file[0] != name:
            file = None  # the file before is let go first: one file's Arrow held at a time
            if root is None:
                return (name, epoch, "file_missing", ""), None, None, None, (), 0
            try:
                file = (name, read_delta_table(Path(root) / name, table=table))
            except FileNotFoundError:
                # NOSILENT: answered as the wait it is; the serving process counts it and logs it at WARNING
                return (name, epoch, "file_missing", ""), None, None, None, (), 0
            except DeltaFileUnreadableError as exc:
                # NOSILENT: answered as the wait it is, with the error; the serving process logs it at WARNING
                return (name, epoch, "file_unreadable", str(exc)), None, None, None, (), 0
        # the record named the key at this epoch and its file holds no change of it, nor clears a prefix of it: never
        # guessed, and never read from L3 (decision 29): the writer is asked to recreate it from the record
        if len(rows.apply(file[1], mine)):
            return (name, epoch, "key_not_in_file", ""), None, None, None, (), 0
        done[mine] = True
    file = None
    merged, at_of = rows.rows()
    keep = ~from_l3[at_of]
    if not keep.all():
        merged, at_of = merged.filter(pa.array(keep)), at_of[keep]
    fell_back = tuple((why, keys_read, tuple(sorted(files))) for why, (keys_read, files) in sorted(fallbacks.items()))
    return None, merged, np.asarray(at_of, dtype=np.int64), from_l3, fell_back, rows.applied


def merged_job(
    table: str,
    key: tuple[str, ...],
    instants: frozenset[str],
    keys: bytes,
    held: bytes,
    plan: tuple[tuple[str, ...], bytes, bytes, bytes],
    named: tuple[tuple[str, str], ...],
    whys: tuple[tuple[str, str], ...],
    root: str | None,
) -> tuple[tuple[str, str, str, str] | None, bytes, bytes, bytes, tuple[tuple[str, int, tuple[str, ...]], ...], int]:
    """:func:`_merged_of`: the worker's job. The files are read here, off the volume, as the merge needs them.

    :param table: the table
    :ptype table: str
    :param key: its key columns
    :ptype key: tuple[str, ...]
    :param instants: its timestamp key columns
    :ptype instants: frozenset[str]
    :param keys: the pass's keys, IPC
    :ptype keys: bytes
    :param held: the rows the copy holds of them, IPC
    :ptype held: bytes
    :param plan: the epoch plan as :func:`epoch_plan_job` answered it
    :ptype plan: tuple[tuple[str, ...], bytes, bytes, bytes]
    :param named: each epoch naming a file, and the file
    :ptype named: tuple[tuple[str, str], ...]
    :param whys: each epoch naming none, and why
    :ptype whys: tuple[tuple[str, str], ...]
    :param root: the delta files' directory; None for none
    :ptype root: str | None
    :return: the wait or None; the rows, IPC; their positions, int64 bytes; the keys read from L3, bool bytes; the
        fallbacks; the changes applied (each empty on a wait)
    :rtype: tuple[tuple[str, str, str, str] | None, bytes, bytes, bytes,
        tuple[tuple[str, int, tuple[str, ...]], ...], int]
    """
    _one_cpu()
    wait, rows, at, from_l3, fell_back, applied = _merged_of(
        table, key, instants, from_ipc(keys), from_ipc(held), epoch_plan_answer(plan), named, whys, root
    )
    if wait is not None:
        return wait, b"", b"", b"", (), 0
    return None, ipc_bytes(rows), at.tobytes(), _flags(from_l3), fell_back, applied


def merged_answer(
    answer: tuple[
        tuple[str, str, str, str] | None, bytes, bytes, bytes, tuple[tuple[str, int, tuple[str, ...]], ...], int
    ],
) -> tuple[tuple[str, str, str, str] | None, Any, Any, Any, tuple[tuple[str, int, tuple[str, ...]], ...], int]:
    """:func:`merged_job`'s answer read back.

    :param answer: the answer
    :ptype answer: tuple
    :return: as :func:`_merged_of`
    :rtype: tuple
    """
    wait, rows, at, from_l3, fell_back, applied = answer
    if wait is not None:
        return wait, None, None, None, (), 0
    return None, from_ipc(rows), _positions(at), _bools(from_l3), fell_back, applied


def stage_cut(key: Sequence[str], keys: Any, pairs: Any, staged: Any, left_out: Any) -> tuple[Any, Any, Any]:
    """a pass's take cut by which keys were staged and which left out: filters, each a copy of the rows kept (memory,
    no judgment a value), each table whole (its chunks combined).

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param keys: the take's keys, key columns as text
    :ptype keys: pyarrow.Table
    :param pairs: the take's keys at their epochs (key columns, ``__epoch``, ``__at``)
    :ptype pairs: pyarrow.Table
    :param staged: per key of the take, whether it was staged
    :ptype staged: numpy.ndarray
    :param left_out: per key of the take, whether it was left pending for a held scope a row of it is of
    :ptype left_out: numpy.ndarray
    :return: the staged keys at their epochs, the keys at their epochs left pending (None for none), the staged keys
    :rtype: tuple[pyarrow.Table, pyarrow.Table | None, pyarrow.Table]
    """
    heard = [*key, _EPOCH]
    at = np.asarray(pairs.column(_AT).to_numpy(), dtype=np.int64)
    kept_pairs = pairs.filter(pa.array(staged[at])).select(heard).combine_chunks()
    late = pairs.filter(pa.array(left_out[at])).select(heard).combine_chunks() if left_out.any() else None
    return kept_pairs, late if late is not None and late.num_rows else None, keys.filter(pa.array(staged))


def _staged_scopes_of(prefix: str | None, column: Any) -> tuple[Any, tuple[str | None, ...]]:
    """the scopes of a stage's keys: for a table whose key names its scope, each key's from its scope column
    (``prefix`` and the value's text) and their distinct ones; for one whose scope is in its rows, the distinct ones of
    the scopes already read with its rows.

    :param prefix: the scope's prefix, when ``column`` is the keys' scope key column; None when it is the scopes
    :ptype prefix: str | None
    :param column: the keys' scope key column, or their scopes
    :ptype column: pyarrow.Array | pyarrow.ChunkedArray
    :return: each key's scope (None when ``column`` is the scopes already), and the distinct scopes
    :rtype: tuple[pyarrow.Array | None, tuple[str | None, ...]]
    """
    scopes = column if prefix is None else _scope_texts(prefix, column)
    return (None if prefix is None else scopes), (_distinct_scopes(scopes) if len(scopes) else ())


def staged_scopes_job(prefix: str | None, column: bytes) -> tuple[bytes, tuple[str | None, ...]]:
    """:func:`_staged_scopes_of`: the worker's job.

    :param prefix: the scope's prefix, or None
    :ptype prefix: str | None
    :param column: the column (a table of that one column), IPC
    :ptype column: bytes
    :return: each key's scope (a table of one column ``__scope``), IPC, empty when ``prefix`` is None; the distinct
        scopes
    :rtype: tuple[bytes, tuple[str | None, ...]]
    """
    _one_cpu()
    scopes, distinct = _staged_scopes_of(prefix, from_ipc(column).column(0))
    return (b"" if scopes is None else ipc_bytes(pa.table({_SCOPE: scopes}))), distinct


def staged_of(
    key: Sequence[str],
    scope_at: tuple[str, int] | None,
    row_scoped: bool,
    keys: Any,
    pairs: Any,
    staged: Any,
    left_out: Any,
    row_scopes: Any,
) -> tuple[Any, Any, tuple[str | None, ...], Any, int]:
    """one table's keyed stage of a pass, cut from its take by which keys were staged and which left out
    (:func:`stage_cut`), with each staged key's scope (:func:`_staged_scopes_of`).

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param scope_at: the scope's prefix and the key column naming it, when the key names it; None otherwise
    :ptype scope_at: tuple[str, int] | None
    :param row_scoped: whether the table's scope is in its rows only (``row_scopes`` holds each staged key's)
    :ptype row_scoped: bool
    :param keys: the take's keys, key columns as text
    :ptype keys: pyarrow.Table
    :param pairs: the take's keys at their epochs (key columns, ``__epoch``, ``__at``)
    :ptype pairs: pyarrow.Table
    :param staged: per key of the take, whether it was staged
    :ptype staged: numpy.ndarray
    :param left_out: per key of the take, whether it was left pending for a held scope a row of it is of
    :ptype left_out: numpy.ndarray
    :param row_scopes: for a table whose scope is in its rows, the staged keys with their rows' scopes (key columns,
        ``__scope``), joined; None otherwise
    :ptype row_scopes: pyarrow.Table | None
    :return: the staged keys at their epochs, the staged keys with their scopes, the distinct scopes, the keys at
        their epochs left pending (None for none), and how many keys they are
    :rtype: tuple[pyarrow.Table, pyarrow.Table, tuple[str | None, ...], pyarrow.Table | None, int]
    """
    key = list(key)
    kept_pairs, late, kept = stage_cut(key, keys, pairs, staged, left_out)
    if scope_at is not None:
        column, scope_set = _staged_scopes_of(scope_at[0], kept.column(key[scope_at[1]]))
        scopes = kept.append_column(_SCOPE, column)
    elif row_scoped:
        scopes = row_scopes
        _, scope_set = _staged_scopes_of(None, scopes.column(_SCOPE))
    else:
        scopes = kept.append_column(_SCOPE, pa.nulls(kept.num_rows, pa.string()))
        scope_set = (None,) if kept.num_rows else ()
    return kept_pairs, scopes.select([*key, _SCOPE]).combine_chunks(), scope_set, late, int(left_out.sum())


def leaving_keys_of(key: Sequence[str], scopes: Any, pairs: Any, hit: Any) -> tuple[Any, bool]:
    """the keys of a keyed stage with a row of a scope in ``hit`` (the stamp moved under it), and whether that is every
    key staged.

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param scopes: the stage's keys with their scopes, joined
    :ptype scopes: pyarrow.Table
    :param pairs: the stage's keys (key columns): every key staged; None to take ``scopes``' leaving keys as every one
    :ptype pairs: pyarrow.Table | None
    :param hit: the scopes left out
    :ptype hit: pyarrow.Array
    :return: the keys leaving (key columns as text) and whether they are every key staged
    :rtype: tuple[pyarrow.Table, bool]
    """
    key = list(key)
    leaving = distinct_keys(scopes.filter(pa.array(_flags_in(scopes.column(_SCOPE), hit))), key)
    staged = distinct_keys(pairs, key) if pairs is not None else leaving
    return leaving, leaving.num_rows >= staged.num_rows


def leaving_keys_job(key: tuple[str, ...], scopes: bytes, pairs: bytes, hit: tuple[str, ...]) -> tuple[bytes, bool]:
    """:func:`leaving_keys_of`: the worker's job.

    :param key: the table's key columns
    :ptype key: tuple[str, ...]
    :param scopes: the stage's keys with their scopes, IPC
    :ptype scopes: bytes
    :param pairs: the stage's keys, IPC; empty for none
    :ptype pairs: bytes
    :param hit: the scopes left out
    :ptype hit: tuple[str, ...]
    :return: the keys leaving, IPC, and whether they are every key staged
    :rtype: tuple[bytes, bool]
    """
    _one_cpu()
    leaving, every = leaving_keys_of(key, from_ipc(scopes), _table_or_none(pairs), pa.array(hit, pa.string()))
    return ipc_bytes(leaving), every


def without_keys_of(
    key: Sequence[str], heard: Any, scopes: Any, keys: Any
) -> tuple[Any, Any, Any, tuple[str | None, ...]]:
    """a keyed stage's bookkeeping less ``keys``.

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param heard: the stage's keys at their epochs, joined (key columns as text, ``__epoch``)
    :ptype heard: pyarrow.Table
    :param scopes: the stage's keys with their scopes, joined; None for none
    :ptype scopes: pyarrow.Table | None
    :param keys: the keys taken out (key columns as text)
    :ptype keys: pyarrow.Table
    :return: the keys taken out at the epochs they were staged from, the keys kept at theirs, the keys kept with
        their scopes (None when ``scopes`` is), and the distinct scopes kept
    :rtype: tuple[pyarrow.Table, pyarrow.Table, pyarrow.Table | None, tuple[str | None, ...]]
    """
    key = list(key)
    texts = joined_key_texts(keys, key)
    out = pa.array(_flags_in(joined_key_texts(heard, key), texts))
    if scopes is None:
        return heard.filter(out), heard.filter(pc.invert(out)), None, ()
    kept = scopes.filter(pc.invert(pa.array(_flags_in(joined_key_texts(scopes, key), texts))))
    return heard.filter(out), heard.filter(pc.invert(out)), kept, _distinct_scopes(kept.column(_SCOPE))


def without_keys_job(
    key: tuple[str, ...], heard: bytes, scopes: bytes, keys: bytes
) -> tuple[bytes, bytes, bytes, tuple[str | None, ...]]:
    """:func:`without_keys_of`: the worker's job.

    :param key: the table's key columns
    :ptype key: tuple[str, ...]
    :param heard: the stage's keys at their epochs, IPC
    :ptype heard: bytes
    :param scopes: the stage's keys with their scopes, IPC; empty for none
    :ptype scopes: bytes
    :param keys: the keys taken out, IPC
    :ptype keys: bytes
    :return: the keys going back, the pairs kept and the scopes kept (empty for none), IPC; the distinct scopes kept
    :rtype: tuple[bytes, bytes, bytes, tuple[str | None, ...]]
    """
    _one_cpu()
    back, kept, kept_scopes, scope_set = without_keys_of(key, from_ipc(heard), _table_or_none(scopes), from_ipc(keys))
    return ipc_bytes(back), ipc_bytes(kept), _ipc_or_empty(kept_scopes), scope_set


def without_keys_answer(
    answer: tuple[bytes, bytes, bytes, tuple[str | None, ...]],
) -> tuple[Any, Any, Any, tuple[str | None, ...]]:
    """:func:`without_keys_job`'s answer read back.

    :param answer: the answer
    :ptype answer: tuple[bytes, bytes, bytes, tuple[str | None, ...]]
    :return: as :func:`without_keys_of`
    :rtype: tuple[pyarrow.Table, pyarrow.Table, pyarrow.Table | None, tuple[str | None, ...]]
    """
    back, kept, scopes, scope_set = answer
    return from_ipc(back), from_ipc(kept), _table_or_none(scopes), scope_set


def newest_by_scope_of(key: Sequence[str], pairs: Any, scopes: Any, unknown: str) -> tuple[tuple[str | None, str], ...]:
    """each scope's newest epoch among the staged keys of it (decision 32's held version): ``unknown`` when one of them
    was heard with no epoch that orders it (none, or one of another incarnation than the rest). Keys matched by their
    codes (:func:`~threetears.core.arrow_keys.key_codes`, never an Acero join).

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param pairs: the staged keys at their epochs, joined (key columns as text, ``__epoch``)
    :ptype pairs: pyarrow.Table
    :param scopes: the staged keys with their scopes, joined (key columns as text, ``__scope``)
    :ptype scopes: pyarrow.Table
    :param unknown: what a scope's version is when no epoch orders it
    :ptype unknown: str
    :return: each scope and its version, one a scope (a state), never one a key
    :rtype: tuple[tuple[str | None, str], ...]
    """
    key = list(key)
    heard = distinct_keys(pairs, [*key, _EPOCH])
    of = distinct_keys(scopes, [*key, _SCOPE])
    if not heard.num_rows or not of.num_rows:
        return ()
    heard_codes, of_codes = key_codes([heard.select(key), of.select(key)], key)
    # each (scope, epoch) of a key: the key's epochs, once for each of its scopes
    order = np.argsort(heard_codes, kind="stable")
    sorted_codes = heard_codes[order]
    low = np.searchsorted(sorted_codes, of_codes, side="left")
    counts = np.searchsorted(sorted_codes, of_codes, side="right") - low
    total = int(counts.sum())
    if not total:
        return ()
    scope_rows = np.repeat(np.arange(of.num_rows), counts)
    # each scope row's run of its key's epochs: its start, then one on, a row at a time
    starts = np.repeat(low, counts)
    within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    heard_rows = order[starts + within]
    # a table of no scope names its one state None: a sentinel no scope is, while the scopes are coded
    named = pc.fill_null(of.column(_SCOPE).combine_chunks(), _NO_SCOPE)
    coded = pc.dictionary_encode(named)
    scope_of = np.asarray(coded.indices.to_numpy(zero_copy_only=False), dtype=np.int64)[scope_rows]
    epochs = heard.column(_EPOCH).combine_chunks()
    distinct = pc.unique(pc.drop_null(epochs))
    parsed = [split_generation_token(name) for name in distinct.to_pylist()]  # one a distinct epoch, never a key
    incarnations = {inc: n for n, inc in enumerate(sorted({p[0] for p in parsed if p is not None}))}
    inc_of = np.array([incarnations[p[0]] if p is not None else -1 for p in parsed] + [-1], dtype=np.int64)
    count_of = np.array([p[1] if p is not None else -1 for p in parsed] + [-1], dtype=np.int64)
    code = np.asarray(pc.fill_null(pc.index_in(epochs, value_set=distinct), -1).to_numpy(), dtype=np.int64)
    row_inc, row_count = inc_of[code[heard_rows]], count_of[code[heard_rows]]
    states = len(coded.dictionary)
    low_inc = np.full(states, np.iinfo(np.int64).max, dtype=np.int64)
    high_inc = np.full(states, -1, dtype=np.int64)
    newest = np.full(states, -1, dtype=np.int64)
    np.minimum.at(low_inc, scope_of, row_inc)
    np.maximum.at(high_inc, scope_of, row_inc)
    np.maximum.at(newest, scope_of, row_count)
    names = sorted(incarnations, key=incarnations.__getitem__)
    versions: list[tuple[str | None, str]] = []
    for at, scope in enumerate(coded.dictionary.to_pylist()):  # one a scope (a state), never one a key
        ordered = low_inc[at] >= 0 and low_inc[at] == high_inc[at]
        versions.append(
            (
                None if scope == _NO_SCOPE else scope,
                f"{names[int(high_inc[at])]}:{int(newest[at])}" if ordered else unknown,
            )
        )
    return tuple(versions)


def newest_by_scope_job(
    key: tuple[str, ...], pairs: bytes, scopes: bytes, unknown: str
) -> tuple[tuple[str | None, str], ...]:
    """:func:`newest_by_scope_of`: the worker's job.

    :param key: the table's key columns
    :ptype key: tuple[str, ...]
    :param pairs: the staged keys at their epochs, IPC
    :ptype pairs: bytes
    :param scopes: the staged keys with their scopes, IPC
    :ptype scopes: bytes
    :param unknown: what a scope's version is when no epoch orders it
    :ptype unknown: str
    :return: each scope and its version
    :rtype: tuple[tuple[str | None, str], ...]
    """
    _one_cpu()
    return newest_by_scope_of(key, from_ipc(pairs), from_ipc(scopes), unknown)


def held_by_full(epoch: str | None, floor: tuple[str, int] | None, incarnation: str | None) -> bool:
    """whether a full reloaded at ``floor`` already holds the change of ``epoch``.

    :param epoch: the epoch a change was heard at; None when it was heard with none (never judged held)
    :ptype epoch: str | None
    :param floor: the full's version of the table, split; None when it is not a generation token
    :ptype floor: tuple[str, int] | None
    :param incarnation: after a lost log, the incarnation the copy reads now: an epoch of the lost one is held too;
        None otherwise
    :ptype incarnation: str | None
    :return: whether it is held
    :rtype: bool
    """
    of = split_generation_token(epoch) if epoch is not None else None
    if of is None or floor is None:
        return False
    if of[0] != floor[0]:
        return incarnation is not None and of[0] != incarnation
    return of[1] <= floor[1]


def past_floor_of(key: Sequence[str], heard: Any, floor: str, incarnation: str | None) -> Any:
    """the keys heard of a table while it waited, at the epochs the full it reloaded from does not hold, each key at
    each epoch once; a key heard with no epoch is kept.

    :param key: the table's key columns
    :ptype key: Sequence[str]
    :param heard: the keys heard at their epochs, joined
    :ptype heard: pyarrow.Table
    :param floor: the full's version of the table (its oldest)
    :ptype floor: str
    :param incarnation: after a lost log, the incarnation the copy reads now; None otherwise
    :ptype incarnation: str | None
    :return: the keys at the epochs it does not hold
    :rtype: pyarrow.Table
    """
    rows = distinct_keys(heard, [*key, _EPOCH])
    epochs = pc.unique(pc.drop_null(rows.column(_EPOCH)))
    split = split_generation_token(floor)
    # one judgment a distinct epoch (a commit), never one a key
    held = [epoch for epoch in epochs.to_pylist() if held_by_full(epoch, split, incarnation)]
    if not held:
        return rows
    return rows.filter(pc.invert(pa.array(_flags_in(rows.column(_EPOCH), pa.array(held, pa.string())))))


def past_floor_job(key: tuple[str, ...], heard: bytes, floor: str, incarnation: str | None) -> bytes:
    """:func:`past_floor_of`: the worker's job.

    :param key: the table's key columns
    :ptype key: tuple[str, ...]
    :param heard: the keys heard at their epochs, IPC
    :ptype heard: bytes
    :param floor: the full's version of the table
    :ptype floor: str
    :param incarnation: after a lost log, the incarnation the copy reads now; None otherwise
    :ptype incarnation: str | None
    :return: the keys kept, IPC
    :rtype: bytes
    """
    _one_cpu()
    return ipc_bytes(past_floor_of(key, from_ipc(heard), floor, incarnation))


def _past_states_of(
    key: Sequence[str],
    scope: tuple[str, str],
    heard: Any,
    versions: Sequence[tuple[str | None, str]],
    incarnation: str | None,
) -> Any:
    """the keys heard of a table while it waited, each key at each epoch once, less those of a reloaded state at an
    epoch its part holds; a key of every other state is kept, as is one heard with no epoch.

    :param key: the table's key columns (naming its state column)
    :ptype key: Sequence[str]
    :param scope: the state column and its scopes' prefix
    :ptype scope: tuple[str, str]
    :param heard: the keys heard at their epochs, joined
    :ptype heard: pyarrow.Table
    :param versions: each reloaded state's scope and its part's version
    :ptype versions: Sequence[tuple[str | None, str]]
    :param incarnation: after a lost log, the incarnation the copy reads now; None otherwise
    :ptype incarnation: str | None
    :return: the keys kept
    :rtype: pyarrow.Table
    """
    column, prefix = scope
    rows = distinct_keys(heard, [*key, _EPOCH])
    if not rows.num_rows:
        return rows
    drop = np.zeros(rows.num_rows, dtype=bool)
    for state, version in versions:  # one a reloaded state, never one a key
        if state is None:
            continue
        of_state = np.asarray(
            pc.fill_null(pc.equal(rows.column(column), state[len(prefix) :]), False).to_numpy(zero_copy_only=False),
            dtype=bool,
        )
        if not of_state.any():
            continue
        split = split_generation_token(version)
        epochs = pc.unique(pc.drop_null(rows.filter(pa.array(of_state)).column(_EPOCH)))
        # one judgment a distinct epoch of the state (a commit), never one a key
        held = [epoch for epoch in epochs.to_pylist() if held_by_full(epoch, split, incarnation)]
        if held:
            drop |= of_state & _flags_in(rows.column(_EPOCH), pa.array(held, pa.string()))
    return rows.filter(pa.array(~drop)) if drop.any() else rows


def past_states_job(
    key: tuple[str, ...],
    scope: tuple[str, str],
    heard: bytes,
    versions: tuple[tuple[str | None, str], ...],
    incarnation: str | None,
) -> bytes:
    """:func:`_past_states_of`: the worker's job.

    :param key: the table's key columns
    :ptype key: tuple[str, ...]
    :param scope: the state column and its scopes' prefix
    :ptype scope: tuple[str, str]
    :param heard: the keys heard at their epochs, IPC
    :ptype heard: bytes
    :param versions: each reloaded state's scope and version
    :ptype versions: tuple[tuple[str | None, str], ...]
    :param incarnation: after a lost log, the incarnation the copy reads now; None otherwise
    :ptype incarnation: str | None
    :return: the keys kept, IPC
    :rtype: bytes
    """
    _one_cpu()
    return ipc_bytes(_past_states_of(key, scope, from_ipc(heard), versions, incarnation))


def heard_scopes_of(prefix: str, column: Any) -> tuple[str, ...]:
    """the distinct scopes the keys heard of a table name in their key (its state column's values), sorted.

    :param prefix: the scope's prefix
    :ptype prefix: str
    :param column: the keys' state column, as text
    :ptype column: pyarrow.Array | pyarrow.ChunkedArray
    :return: the scopes, one a state, never one a key
    :rtype: tuple[str, ...]
    """
    if not len(column):
        return ()
    return tuple(str(scope) for scope in _distinct_scopes(_scope_texts(prefix, column)))


def heard_scopes_job(prefix: str, column: bytes) -> tuple[str, ...]:
    """:func:`heard_scopes_of`: the worker's job.

    :param prefix: the scope's prefix
    :ptype prefix: str
    :param column: the keys' state column (a table of that one column), IPC
    :ptype column: bytes
    :return: the scopes
    :rtype: tuple[str, ...]
    """
    _one_cpu()
    return heard_scopes_of(prefix, from_ipc(column).column(0))
