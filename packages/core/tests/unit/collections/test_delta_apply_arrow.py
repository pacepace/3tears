"""A delta file is applied as Arrow, a column at a time (``HeldDeltaRows``), never as Python rows (0.95.0 copy audit,
P0 memory): exactly as ``apply_delta_row`` applies each change, and in bounded memory at real size.

The copy turned every file into Python rows (``to_pylist``, about ten times its Arrow: 291 MB for 200,000 rows of 15
text columns) and kept each epoch's through the pass. The contract this pins:

- the result equals ``apply_delta_row`` applied change by change, key by key, for any mix of held and new keys,
  upserts that set every column or some, deletes, several changes of a key in one file, cleared prefixes, and several
  files in order (a property test);
- a key the file neither changes nor clears is answered, never guessed;
- a pass of a 200,000-row, 15-column file builds no Python row: its Python allocations stay a small fraction of the
  rows' Arrow, and the result is the old apply's.
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
np = pytest.importorskip("numpy")

from threetears.core.collections.delta_files import (  # noqa: E402
    DeltaChange,
    HeldDeltaRows,
    TableDelta,
    apply_delta_row,
    read_delta_table,
    under_prefix,
    write_delta_file,
)
from threetears.core.testing.delta_files import read_delta_file

_KEY = ("state", "geo_id")
_NAMES = ("state", "geo_id", "votes", "note", "kind")
_TYPES = {"state": pa.string(), "geo_id": pa.int64(), "votes": pa.int64(), "note": pa.string(), "kind": pa.string()}
_STATES = ("VA", "IL")
_SETS = (list(_NAMES), ["state", "geo_id", "votes"], ["state", "geo_id", "note", "kind"], ["state", "geo_id"])


def _of_key(changes: Sequence[DeltaChange], prefixes: Sequence[Mapping[str, str]], key: tuple[str, ...]) -> Any:
    """a key's changes in a file as the copy took them before (``_FileChanges.of_key``): a delete first when a prefix
    the file clears holds it; None when the file neither changes nor clears it."""
    mine = [change for change in changes if (str(change.row["state"]), str(change.row["geo_id"])) == key]
    named = dict(zip(_KEY, key, strict=True))
    if prefixes and under_prefix(named, prefixes):
        return [DeltaChange(row=named, deleted=True, written=frozenset()), *mine]
    return mine or None


_change = st.tuples(
    st.sampled_from(_STATES),
    st.integers(0, 7),
    st.booleans(),
    st.sampled_from(range(len(_SETS))),
    st.integers(0, 99),
    st.sampled_from(["a", "b", None]),
)
_file = st.tuples(st.lists(st.sampled_from(_STATES), max_size=1), st.lists(_change, max_size=12))


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    held=st.dictionaries(
        st.tuples(st.sampled_from(_STATES), st.integers(0, 7)), st.tuples(st.integers(0, 9), st.text(max_size=3))
    ),
    files=st.lists(_file, min_size=1, max_size=4),
    reach=st.lists(st.booleans(), min_size=16, max_size=16),
)
def test_the_arrow_apply_is_apply_delta_row_key_by_key(
    tmp_path: Path, held: dict[tuple[str, int], tuple[int, str]], files: list[Any], reach: list[bool]
) -> None:
    """Flips if the Arrow apply differs from apply_delta_row on any key, for any mix of held and new keys, full and
    partial upserts, deletes, a key's several changes in one file, cleared prefixes and files in order; or if a key the
    file neither changes nor clears is not answered as such (the old of_key's None)."""
    keys = [(state, str(geo)) for state in _STATES for geo in range(8)]
    rows = HeldDeltaRows(_KEY, keys, _TYPES)
    held_rows = [
        {"state": s, "geo_id": g, "votes": v, "note": n, "kind": "held", "extra": 1} for (s, g), (v, n) in held.items()
    ]
    rows.hold(pa.Table.from_pylist(held_rows, schema=pa.schema([*_TYPES.items(), ("extra", pa.int64())])))
    old: dict[tuple[str, ...], dict[str, Any] | None] = {
        (s, str(g)): {"state": s, "geo_id": g, "votes": v, "note": n, "kind": "held"} for (s, g), (v, n) in held.items()
    }
    positions = [at for at, inside in enumerate(reach) if inside]
    for index, (cleared, changes) in enumerate(files):
        delta = TableDelta(name="results", types=_TYPES, key=_KEY)
        delta.clears([{"state": state} for state in cleared])
        for state, geo, deleted, written, votes, note in changes:
            if deleted:
                delta.deletes([(state, geo)])
            else:
                row = {"state": state, "geo_id": geo, "votes": votes, "note": note, "kind": f"f{index}"}
                delta.rows([row], _SETS[written])
        name = write_delta_file(tmp_path, delta, f"w{index}")
        python = read_delta_file(tmp_path / name)
        missing = rows.apply(read_delta_table(tmp_path / name), np.array(positions, dtype=np.int64))
        expected_missing = []
        for at in positions:
            key = keys[at]
            of = _of_key(python.changes, python.prefixes, key)
            if of is None:
                expected_missing.append(at)
                continue
            merged = old.get(key)
            for change in of:
                after = apply_delta_row(merged, change)
                merged = None if after is None else {column: after.get(column) for column in _NAMES}
            old[key] = merged
        assert sorted(missing.tolist()) == expected_missing
    table, at = rows.rows()
    got = {keys[position]: row for position, row in zip(at.tolist(), table.to_pylist(), strict=True)}
    assert got == {key: row for key, row in old.items() if row is not None}


#: the real size the audit measured: a 200,000-row file of 15 text columns
_ROWS = 200_000
_WIDE = ("state", "geo_id", *(f"c{index:02d}" for index in range(13)))
_WIDE_TYPES = {column: pa.string() for column in _WIDE}

#: the most Python memory one apply of the real-size file may allocate at its peak, in bytes (tracemalloc, numpy's
#: arrays counted, Arrow's buffers not). Measured 2026-11-23 (macOS arm64, Python 3.14): the old apply (read_delta_file's
#: to_pylist, a list of changes a key, apply_delta_row a change) peaked at 709 MB at 200k rows; the Arrow apply at
#: 23.6 MB, the keys' text lists and numpy positions (8 bytes a key each). Twice that: a dict a row again (about 1 KB a
#: row of 15 columns, 200 MB here) trips it many times over.
_PYTHON_PEAK_BYTES = 48_000_000


def _wide_file(tmp_path: Path) -> tuple[list[tuple[str, str]], Any, str]:
    keys = [("VA", f"{index:07d}") for index in range(_ROWS)]
    held = pa.table({column: [f"{column}-old-{index:07d}" for index in range(_ROWS)] for column in _WIDE[2:]})
    held = held.append_column("geo_id", pa.array([key[1] for key in keys])).append_column(
        "state", pa.array(["VA"] * _ROWS)
    )
    delta = TableDelta(name="wide", types=_WIDE_TYPES, key=("state", "geo_id"))
    changed = pa.table({column: [f"{column}-new-{index:07d}" for index in range(_ROWS)] for column in _WIDE[2:]})
    changed = changed.append_column("geo_id", held.column("geo_id")).append_column("state", held.column("state"))
    half = _ROWS // 2
    delta.rows(changed.slice(0, half).to_pylist(), list(_WIDE))
    delta.rows(changed.slice(half).to_pylist(), ["state", "geo_id", "c00"])
    deleted = [("VA", f"{index:07d}") for index in range(0, _ROWS, 1000)]
    delta.deletes(deleted)
    return keys, held, write_delta_file(tmp_path, delta, "wide")


def test_a_real_size_file_is_applied_with_no_python_row_and_as_apply_delta_row_would(tmp_path: Path) -> None:
    """Flips if applying a 200,000-row, 15-column file builds a Python object a row (the old to_pylist and changes a key:
    709 MB at this size, measured), or its result differs from apply_delta_row's: half the rows set every column, half
    set c00 alone (the rest kept from the held row), every thousandth key deleted after."""
    keys, held, name = _wide_file(tmp_path)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        rows = HeldDeltaRows(("state", "geo_id"), keys, _WIDE_TYPES)
        rows.hold(held)
        missing = rows.apply(read_delta_table(tmp_path / name), np.arange(_ROWS, dtype=np.int64))
        table, at = rows.rows()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < _PYTHON_PEAK_BYTES, f"the Arrow apply allocated {peak / 1e6:.1f} MB of Python at its peak"
    assert len(missing) == 0
    assert table.num_rows == _ROWS - _ROWS // 1000 == len(at)
    first, last = table.slice(0, 1).to_pylist()[0], table.slice(table.num_rows - 1).to_pylist()[0]
    assert (first["geo_id"], first["c00"], first["c01"]) == ("0000001", "c00-new-0000001", "c01-new-0000001")
    assert (last["geo_id"], last["c00"], last["c01"]) == ("0199999", "c00-new-0199999", "c01-old-0199999")
    assert "0001000" not in set(table.column("geo_id").slice(0, 2000).to_pylist())
