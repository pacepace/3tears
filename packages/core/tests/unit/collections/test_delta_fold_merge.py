"""The writer's fold merges a state's part and its delta files in Arrow (``fold_delta_tables``), never a Python row a
changed key: exactly the rows the old merge (``delta_fold._applied``: the changed rows as dicts, ``apply_delta_row`` a
change, a cleared prefix taking out every row under it before its file's changes) folded.

The contract this pins (a property test against the old merge, kept here as the reference): for any held part, any
mix of upserts that set every column or some, deletes, a key's several changes in one file, cleared prefixes and
several files in order, the folded rows are the old merge's, one a key; and the rows no file changes or clears are
kept as they were read, first, in their order.
"""

from __future__ import annotations

import tracemalloc
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

pa = pytest.importorskip("pyarrow")

from threetears.core.collections.delta_files import (  # noqa: E402
    DeltaChange,
    TableDelta,
    fold_delta_tables,
    read_delta_table,
    under_prefix,
    write_delta_file,
)
from threetears.core.testing.delta_files import apply_delta_row, read_delta_file  # noqa: E402

_KEY = ("state", "geo_id")
_NAMES = ("state", "geo_id", "votes", "note", "kind")
_TYPES = {"state": pa.string(), "geo_id": pa.int64(), "votes": pa.int64(), "note": pa.string(), "kind": pa.string()}
_SCHEMA = pa.schema(list(_TYPES.items()))
_STATES = ("VA", "IL")
_SETS = (list(_NAMES), ["state", "geo_id", "votes"], ["state", "geo_id", "note", "kind"], ["state", "geo_id"])


def _old_merge(
    held: Sequence[Mapping[str, Any]], files: Sequence[tuple[Sequence[Mapping[str, str]], Sequence[DeltaChange]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """the fold's merge before it was Arrow (``delta_fold._applied``), as it was: each file's clear, then its changes,
    a row at a time over the held rows the files name or clear. The rows it kept untouched, in their order, and the
    rows it merged."""

    def key_of(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(row[name] for name in _KEY)

    def under(key: tuple[Any, ...], prefixes: Sequence[Mapping[str, str]]) -> bool:
        return under_prefix(dict(zip(_KEY, key, strict=True)), prefixes)

    touched = {key_of(change.row) for _, changes in files for change in changes}
    every_cleared = [prefix for prefixes, _ in files for prefix in prefixes]
    hit = [key_of(row) in touched or (bool(every_cleared) and under(key_of(row), every_cleared)) for row in held]
    merged = {key_of(row): dict(row) for row, h in zip(held, hit, strict=True) if h}
    for prefixes, changes in files:
        if prefixes:
            merged = {key: row for key, row in merged.items() if not under(key, prefixes)}
        for change in changes:
            key = key_of(change.row)
            after = apply_delta_row(merged.get(key), change)
            if after is None:
                merged.pop(key, None)
            else:
                merged[key] = {name: after.get(name) for name in _NAMES}
    untouched = [dict(row) for row, h in zip(held, hit, strict=True) if not h]
    return untouched, list(merged.values())


_change = st.tuples(
    st.sampled_from(_STATES),
    st.integers(0, 7),
    st.booleans(),
    st.sampled_from(range(len(_SETS))),
    st.integers(0, 99),
    st.sampled_from(["a", "b", None]),
)
_file = st.tuples(st.lists(st.sampled_from(_STATES), max_size=1), st.lists(_change, max_size=12))


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    held=st.dictionaries(
        st.tuples(st.sampled_from(_STATES), st.integers(0, 7)), st.tuples(st.integers(0, 9), st.text(max_size=3))
    ),
    files=st.lists(_file, max_size=4),
)
def test_the_arrow_fold_is_the_old_merge(
    tmp_path: Path, held: dict[tuple[str, int], tuple[int, str]], files: list[Any]
) -> None:
    """Flips if the Arrow fold differs from the old merge on any row, for any held part and any files in order: full
    and partial upserts, deletes, a key's several changes in one file, cleared prefixes; or if a row no file changes
    or clears is not kept first, in the order it was held."""
    held_rows = [{"state": s, "geo_id": g, "votes": v, "note": n, "kind": "held"} for (s, g), (v, n) in held.items()]
    deltas = []
    python = []
    for index, (cleared, changes) in enumerate(files):
        delta = TableDelta(name="results", types=_TYPES, key=_KEY)
        delta.clears([{"state": state} for state in cleared])
        for state, geo, deleted, written, votes, note in changes:
            if deleted:
                delta.deletes([(state, geo)])
            else:
                row = {"state": state, "geo_id": geo, "votes": votes, "note": note, "kind": f"f{index}"}
                delta.rows([row], _SETS[written])
        name = write_delta_file(tmp_path, delta, f"w{len(list(tmp_path.rglob('*.arrow')))}")
        read = read_delta_file(tmp_path / name)
        python.append((read.prefixes, read.changes))
        deltas.append(read_delta_table(tmp_path / name))

    folded = fold_delta_tables(_KEY, pa.Table.from_pylist(held_rows, schema=_SCHEMA), deltas, _TYPES)
    untouched, merged = _old_merge(held_rows, python)

    got = folded.select(list(_NAMES)).to_pylist()
    assert got[: len(untouched)] == untouched, "the rows no file changes or clears, first, as they were held"

    def by_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (row["state"], row["geo_id"])

    assert sorted(got[len(untouched) :], key=by_key) == sorted(merged, key=by_key), "the merged rows, one a key"


#: a real state's part: LSF's largest state part held 337k rows (the 0.95 fold audit)
_HELD = 337_000
#: the rows each of its two deltas changes: every column set in one, one column in the other, a delete each 1000th
_CHANGED = 60_000
_WIDE = ("state", "geo_id", *(f"c{index:02d}" for index in range(13)))
_WIDE_TYPES = {"state": pa.string(), "geo_id": pa.int64(), **{column: pa.string() for column in _WIDE[2:]}}

#: the most Python memory one fold of the real-size part may allocate at its peak, in bytes (tracemalloc, numpy's
#: arrays counted, Arrow's buffers not). Measured 2026-11-23 (macOS arm64, Python 3.14): the old merge (each changed
#: held row and each change a dict, apply_delta_row a change) peaked at 254 MB here; the Arrow fold at 7.1 MB (numpy
#: positions and flags, 8 bytes a key). A dict a changed row again (about 1 KB a row of 15 columns) trips it many times
#: over.
_PYTHON_PEAK_BYTES = 40_000_000


def _wide(count: int, start: int, tag: str) -> Any:
    geo = pa.array(range(start, start + count), pa.int64())
    texts = {column: pa.array([f"{column}-{tag}"] * count) for column in _WIDE[2:]}
    return pa.table({"state": pa.array(["VA"] * count), "geo_id": geo, **texts}, schema=pa.schema(_WIDE_TYPES.items()))


def test_a_real_size_part_is_folded_with_no_python_row(tmp_path: Path) -> None:
    """Flips if folding a 337,000-row, 15-column part with two 60,000-row deltas builds a Python object a changed row
    (the old merge: 254 MB at this size, measured), or its rows are not the merge's: the first delta sets every column
    of its rows (half held, half new), the second c00 alone of rows the first wrote, every thousandth of them deleted."""
    full = TableDelta(name="wide", types=_WIDE_TYPES, key=_KEY)
    full.arrow(_wide(_CHANGED, _HELD - _CHANGED // 2, "first"), list(_WIDE))
    partial = TableDelta(name="wide", types=_WIDE_TYPES, key=_KEY)
    partial.arrow(_wide(_CHANGED, _HELD - _CHANGED // 2, "second"), ["state", "geo_id", "c00"])
    partial.deletes(
        pa.table(
            {
                "state": ["VA"] * 60,
                "geo_id": pa.array(range(_HELD - _CHANGED // 2 + 500, _HELD + _CHANGED // 2, 1000), pa.int64()),
            }
        )
    )
    deltas = [
        read_delta_table(tmp_path / write_delta_file(tmp_path, delta, f"w{at}"))
        for at, delta in enumerate((full, partial))
    ]
    held = _wide(_HELD, 0, "held")
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        folded = fold_delta_tables(_KEY, held, deltas, _WIDE_TYPES)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < _PYTHON_PEAK_BYTES, f"the Arrow fold allocated {peak / 1e6:.1f} MB of Python at its peak"
    assert folded.num_rows == _HELD + _CHANGED // 2 - 60
    untouched = _HELD - _CHANGED // 2
    assert folded.slice(0, untouched).equals(held.slice(0, untouched)), "the rows no delta names, first, as held"
    merged = folded.slice(untouched).sort_by("geo_id")
    first, last = merged.slice(0, 1).to_pylist()[0], merged.slice(merged.num_rows - 1).to_pylist()[0]
    assert (first["geo_id"], first["c00"], first["c01"]) == (untouched, "c00-second", "c01-first")
    assert (last["geo_id"], last["c00"], last["c01"]) == (_HELD + _CHANGED // 2 - 1, "c00-second", "c01-first")
    assert _HELD - _CHANGED // 2 + 500 not in set(merged.column("geo_id").slice(0, 2000).to_pylist())
