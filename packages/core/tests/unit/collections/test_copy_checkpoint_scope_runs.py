"""a checkpoint write's cut of a batch by scope (``copy_checkpoint._scope_runs``) gives what the cut before 2026-11-26
gave, cheaper when each scope's rows run together.

The contract this pins:

- each value of the scope column with the rows holding it, in the order the values first appear, the same values and
  the same rows as the reference (the cut as it was, kept below), for rows grouped by scope and not, nulls grouped and
  scattered, one scope, no rows and one row;
- when each value's rows run together, each value's rows are a slice of the batch (no copy);
- on rows grouped by scope (a copy's), the cut costs well under the reference's, which dictionary-encoded every batch.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pc = pytest.importorskip("pyarrow.compute")

from threetears.core.collections.copy_checkpoint import _scope_runs  # noqa: E402
from threetears.core.testing.calibrate import best_cpu_seconds, thread_cpu_seconds  # noqa: E402


def _reference_runs(batch: Any, column: str) -> Iterator[tuple[Any, Any]]:
    """the cut as it was before 2026-11-26 (3tears 981349c83): the column dictionary encoded, runs found on its codes,
    one stable sort by code when a value's rows are not together."""
    if batch.num_rows == 0:
        return
    encoded = batch.column(batch.schema.get_field_index(column)).dictionary_encode(null_encoding="encode")
    values = encoded.dictionary.to_pylist()
    codes = encoded.indices

    def starts_of(codes: Any) -> list[int]:
        changed = pc.not_equal(codes.slice(1), codes.slice(0, len(codes) - 1))
        return [0, *(int(index) + 1 for index in pc.indices_nonzero(changed).to_pylist())]

    starts = starts_of(codes)
    if len(starts) != len(values):
        order = pc.sort_indices(codes)
        batch, codes = batch.take(order), codes.take(order)
        starts = starts_of(codes)
    for start, end in zip(starts, [*starts[1:], batch.num_rows], strict=True):
        yield values[codes[start].as_py()], batch.slice(start, end - start)


def _batch(scopes: list[Any], scope_type: Any = None) -> Any:
    return pa.record_batch(
        {
            "state": pa.array(scopes, scope_type or pa.string()),
            "geo_id": pa.array(range(len(scopes)), pa.int64()),
        }
    )


def _assert_same(batch: Any) -> list[tuple[Any, Any]]:
    found = list(_scope_runs(batch, "state"))
    expected = list(_reference_runs(batch, "state"))
    assert [value for value, _ in found] == [value for value, _ in expected]
    for (_, rows), (_, want) in zip(found, expected, strict=True):
        assert rows.equals(want)
    return found


def _assert_slices(batch: Any, found: list[tuple[Any, Any]]) -> None:
    """each part shares the batch's buffers: a slice, no copy."""
    for _, rows in found:
        for index in range(batch.num_columns):
            assert rows.column(index).buffers()[-1].address == batch.column(index).buffers()[-1].address


@pytest.mark.parametrize(
    ("scopes", "scope_type", "values"),
    [
        (["S00", "S00", "S00", "S01", "S01", "S02", "S02", "S02", "S02"], None, ["S00", "S01", "S02"]),
        ([None, None, "a", "b", "b"], None, [None, "a", "b"]),
        (["a", "a", None, None, None, "b"], None, ["a", None, "b"]),
        (["a", "a", None], None, ["a", None]),
        ([7, 7, None, None, 3], pa.int64(), [7, None, 3]),
        (["only"] * 5, None, ["only"]),
        ([None] * 4, None, [None]),
        (["one"], None, ["one"]),
        ([None], None, [None]),
    ],
    ids=[
        "grouped",
        "nulls_grouped_first",
        "nulls_grouped_between",
        "nulls_grouped_last",
        "int_nulls_grouped",
        "single_scope",
        "all_null",
        "one_row",
        "one_null_row",
    ],
)
def test_rows_grouped_by_scope_are_cut_as_the_reference_cuts_them_as_slices(
    scopes: list[Any], scope_type: Any, values: list[Any]
) -> None:
    batch = _batch(scopes, scope_type)
    found = _assert_same(batch)
    assert [value for value, _ in found] == values
    assert sum(rows.num_rows for _, rows in found) == batch.num_rows
    _assert_slices(batch, found)


@pytest.mark.parametrize(
    ("scopes", "scope_type", "values"),
    [
        (["S01", "S00", "S01", "S02", "S00"], None, ["S01", "S00", "S02"]),
        ([None, "a", None, "b"], None, [None, "a", "b"]),
        (["a", None, "a", None, "a"], None, ["a", None]),
        ([3, None, 3, 1, None], pa.int64(), [3, None, 1]),
        (["a", "b", "a"], None, ["a", "b"]),
    ],
    ids=[
        "ungrouped",
        "nulls_scattered",
        "nulls_between_repeats",
        "int_nulls_scattered",
        "value_returns",
    ],
)
def test_rows_not_grouped_by_scope_are_cut_as_the_reference_cuts_them(
    scopes: list[Any], scope_type: Any, values: list[Any]
) -> None:
    found = _assert_same(_batch(scopes, scope_type))
    assert [value for value, _ in found] == values
    for value, rows in found:
        assert rows.column(0).to_pylist() == [value] * rows.num_rows
        ids = rows.column(1).to_pylist()
        assert ids == sorted(ids)  # stable: each value's rows keep the batch's order


def test_no_rows_is_no_runs() -> None:
    assert list(_scope_runs(_batch([]), "state")) == []
    assert list(_reference_runs(_batch([]), "state")) == []


async def test_rows_grouped_by_scope_cost_well_under_the_reference_cut() -> None:
    """the cost the cut lost (measured 2026-11-26: 12.3 -> ~5.8 ms a 1M-row write of 50 scopes, one thread): a batch of
    a copy's rows, grouped by scope, cut for at most 0.75x the reference's CPU on this thread (about 0.5x measured)."""
    per = 5_000
    batch = _batch([f"S{s:02d}" for s in range(50) for _ in range(per)])

    async def cut() -> None:
        for _ in _scope_runs(batch, "state"):
            pass

    async def reference() -> None:
        for _ in _reference_runs(batch, "state"):
            pass

    took = await best_cpu_seconds(cut, clock=thread_cpu_seconds)
    baseline = await best_cpu_seconds(reference, clock=thread_cpu_seconds)
    assert took <= 0.75 * baseline, f"the cut took {took * 1000:.2f} ms, the reference {baseline * 1000:.2f} ms"
