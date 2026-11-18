"""Rows matched by a key of several columns in Arrow's hash kernels (:mod:`threetears.core.arrow_keys`), never an
Acero plan: the same rows a semi join, an anti join and a ``group_by`` gave, held to a plain Python reference."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pyarrow as pa
import pytest

from threetears.core import arrow_keys
from threetears.core.arrow_keys import distinct_keys, key_codes, last_rows, rows_named, rows_not_named

KEY = ("state", "run_id", "geo", "at")
START = datetime(2026, 11, 3, 23, tzinfo=UTC)


def _rows(rng: np.random.Generator, count: int, chunks: int = 1) -> pa.Table:
    """rows of a four-column key (text, int, text, time) drawn from small domains, so keys repeat, in ``chunks``."""
    pieces = []
    for _ in range(chunks):
        n = count // chunks
        pieces.append(
            pa.table(
                {
                    "state": pa.array(rng.choice(["VA", "NC", "GA"], n)),
                    "run_id": pa.array(rng.integers(0, 4, n), pa.int64()),
                    "geo": pa.array([f"g{v}" for v in rng.integers(0, 6, n)]),
                    "at": pa.array(
                        [START + timedelta(minutes=int(v)) for v in rng.integers(0, 3, n)], pa.timestamp("us", tz="UTC")
                    ),
                    "value": pa.array(rng.random(n)),
                }
            )
        )
    return pa.concat_tables(pieces)


def _keys(table: pa.Table) -> list[tuple[object, ...]]:
    return list(zip(*(table[column].to_pylist() for column in KEY), strict=True))


@pytest.mark.parametrize("seed", range(20))
def test_rows_named_are_the_semi_joins_rows_and_the_rest_the_anti_joins(seed: int) -> None:
    rng = np.random.default_rng(seed)
    held = _rows(rng, 600, chunks=3)
    named = _rows(rng, 40).select(list(KEY))
    wanted = set(_keys(named))
    expected = [row for row, values in enumerate(_keys(held)) if values in wanted]
    assert rows_named(held, named, KEY).tolist() == expected
    assert rows_not_named(held, named, KEY).to_pylist() == [r for r in range(held.num_rows) if r not in set(expected)]


def test_nothing_named_names_no_row_and_leaves_every_row() -> None:
    held = _rows(np.random.default_rng(1), 30)
    empty = held.select(list(KEY)).slice(0, 0)
    assert rows_named(held, empty, KEY).tolist() == []
    assert rows_not_named(held, empty, KEY).to_pylist() == list(range(30))
    assert rows_named(held.slice(0, 0), held, KEY).tolist() == []


def test_a_null_is_a_value_of_its_own() -> None:
    held = pa.table({"a": ["x", None, "x", None], "b": [1, 1, None, None]})
    named = pa.table({"a": [None], "b": [1]})
    assert rows_named(held, named, ("a", "b")).tolist() == [1]


def test_codes_stay_exact_when_the_combined_count_would_pass_an_int64(monkeypatch: pytest.MonkeyPatch) -> None:
    """the codes so far are coded again before the next column would overflow them: held to a room of 7 distinct
    keys, every column folds through a recode, and equal codes are still exactly equal keys."""
    monkeypatch.setattr(arrow_keys, "_CODE_ROOM", 7)
    rng = np.random.default_rng(3)
    held = _rows(rng, 500, chunks=2)
    (codes,) = key_codes([held], KEY)
    keys = _keys(held)
    by_code: dict[int, tuple[object, ...]] = {}
    for code, values in zip(codes.tolist(), keys, strict=True):
        assert by_code.setdefault(code, values) == values
    assert len(by_code) == len(set(keys))
    named = _rows(rng, 30).select(list(KEY))
    wanted = set(_keys(named))
    assert rows_named(held, named, KEY).tolist() == [r for r, values in enumerate(keys) if values in wanted]


def test_distinct_keys_are_each_key_once_in_first_seen_order() -> None:
    held = _rows(np.random.default_rng(5), 400, chunks=4)
    seen: dict[tuple[object, ...], None] = {}
    for values in _keys(held):
        seen.setdefault(values, None)
    distinct = distinct_keys(held, KEY)
    assert distinct.column_names == list(KEY)
    assert _keys(distinct) == list(seen)
    assert distinct_keys(held.slice(0, 0), KEY).num_rows == 0


@pytest.mark.parametrize("seed", range(10))
def test_last_rows_are_the_last_row_of_each_key_in_row_order(seed: int) -> None:
    held = _rows(np.random.default_rng(seed), 300, chunks=3)
    last: dict[tuple[object, ...], int] = {}
    for row, values in enumerate(_keys(held)):
        last[values] = row
    assert last_rows(held, KEY).tolist() == sorted(last.values())
    assert last_rows(held.slice(0, 0), KEY).tolist() == []


def test_last_rows_of_unique_keys_are_every_row() -> None:
    held = pa.table({"a": ["x", "y", None], "b": [1, 1, 1]})
    assert last_rows(held, ("a", "b")).tolist() == [0, 1, 2]
