"""A copied table's parts (:mod:`threetears.core.collections.copy_parts`): cut by its declared scope column once it is
over the size floor, one part below it or with no scope column, a key without the scope column found through its
locator, and all of it in Arrow, never a Python object a row."""

from __future__ import annotations

import tracemalloc
from collections.abc import Callable
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from threetears.core.collections.copy_follower import FollowedTable
from threetears.core.collections.copy_parts import (
    SPLIT_FLOOR_BYTES,
    WHOLE_PART,
    Locator,
    PartRule,
    part_rules,
    parts_of_keys,
    split_parts,
)
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column

_STATES = pa.array(["VA", "NC", "GA", "PA", "AZ"], pa.string())


def _followed(name: str, key: tuple[str, ...], *, scope: str | None = "state") -> FollowedTable:
    columns = (
        Column(name="state", column_type=STRING_TYPE),
        Column(name="office_key", column_type=STRING_TYPE),
        Column(name="candidate_id", column_type=BIGINT_TYPE),
        Column(name="votes", column_type=BIGINT_TYPE),
    )
    return FollowedTable(name=name, key=key, columns=columns, scope=None if scope is None else (scope, f"{scope}:"))


def _rows(count: int, seed: int = 3) -> pa.Table:
    """rows over five states, unsorted, each race (``office_key``) in one state."""
    rng = np.random.default_rng(seed)
    race = rng.integers(0, max(count // 4, 1), count)
    return pa.table(
        {
            "state": _STATES.take(pa.array(race % len(_STATES))),
            "office_key": pc.cast(pa.array(race), pa.string()),
            "candidate_id": pa.array(np.arange(count, dtype=np.int64)),
            "votes": pa.array(rng.integers(0, 1_000, count)),
        }
    )


_SCOPED_KEY = ("state", "office_key", "candidate_id")
_RACE_KEY = ("office_key", "candidate_id")


class TestPartRules:
    def test_a_scoped_table_over_the_floor_is_cut_by_its_scope(self) -> None:
        rules = part_rules([_followed("results", _SCOPED_KEY)], {"results": SPLIT_FLOOR_BYTES})
        assert rules["results"] == PartRule("results", _SCOPED_KEY, scope_column="state")
        assert rules["results"].split

    def test_a_scoped_table_under_the_floor_is_one_part(self) -> None:
        rules = part_rules([_followed("results", _SCOPED_KEY)], {"results": SPLIT_FLOOR_BYTES - 1})
        assert rules["results"] == PartRule("results", _SCOPED_KEY)
        assert not rules["results"].split

    def test_a_table_with_no_scope_column_is_one_part_at_any_size(self) -> None:
        rules = part_rules([_followed("catalog", _SCOPED_KEY, scope=None)], {"catalog": 100 * SPLIT_FLOOR_BYTES})
        assert not rules["catalog"].split

    def test_a_split_table_stays_split_down_to_half_the_floor(self) -> None:
        table = [_followed("results", _SCOPED_KEY)]
        held = part_rules(table, {"results": SPLIT_FLOOR_BYTES})
        assert part_rules(table, {"results": SPLIT_FLOOR_BYTES // 2}, held=held)["results"].split
        assert not part_rules(table, {"results": SPLIT_FLOOR_BYTES // 2 - 1}, held=held)["results"].split
        # never split, the same size is one part
        assert not part_rules(table, {"results": SPLIT_FLOOR_BYTES // 2})["results"].split

    def test_a_key_without_its_scope_column_needs_a_locator_at_every_size(self) -> None:
        table = [_followed("results", _RACE_KEY)]
        with pytest.raises(ValueError, match="needs a locator"):
            part_rules(table, {"results": 1})
        rules = part_rules(table, {"results": SPLIT_FLOOR_BYTES}, locators={"results": "office_key"})
        assert rules["results"] == PartRule("results", _RACE_KEY, scope_column="state", locator="office_key")

    def test_a_locator_must_be_a_key_column_of_a_scoped_copied_table(self) -> None:
        with pytest.raises(ValueError, match="not a key column"):
            part_rules([_followed("results", _RACE_KEY)], {"results": 1}, locators={"results": "votes"})
        with pytest.raises(ValueError, match="declares none"):
            part_rules(
                [_followed("catalog", _RACE_KEY, scope=None)], {"catalog": 1}, locators={"catalog": "office_key"}
            )
        with pytest.raises(ValueError, match="not copied"):
            part_rules([_followed("results", _SCOPED_KEY)], {"results": 1}, locators={"other": "office_key"})

    def test_every_table_needs_its_size(self) -> None:
        with pytest.raises(ValueError, match="no size"):
            part_rules([_followed("results", _SCOPED_KEY)], {})

    def test_the_rules_are_read_only(self) -> None:
        rules = part_rules([_followed("results", _SCOPED_KEY)], {"results": 1})
        with pytest.raises(TypeError):
            rules["other"] = PartRule("other", ("k",))  # type: ignore[index]


class TestSplitParts:
    def test_a_split_table_is_one_part_a_scope_value_holding_exactly_its_rows(self) -> None:
        rows = _rows(1_000)
        parts = split_parts(PartRule("results", _SCOPED_KEY, scope_column="state"), rows)
        assert [part.name for part in parts] == sorted(_STATES.to_pylist())
        for part in parts:
            expected = rows.filter(pc.equal(rows["state"], part.name))
            # each part keeps its rows' order in the rows split
            assert part.rows.equals(expected.combine_chunks())
        assert sum(part.rows.num_rows for part in parts) == rows.num_rows

    def test_a_table_held_whole_is_one_part_of_every_row(self) -> None:
        rows = _rows(100)
        (part,) = split_parts(PartRule("results", _SCOPED_KEY), rows)
        assert part.name == WHOLE_PART
        assert part.rows.equals(rows)

    def test_no_rows_is_no_parts(self) -> None:
        rule = PartRule("results", _SCOPED_KEY, scope_column="state")
        assert split_parts(rule, _rows(10).slice(0, 0)) == ()

    def test_an_int_scope_is_named_by_its_text(self) -> None:
        rows = pa.table({"geo": pa.array([7, 12, 7], pa.int64()), "k": pa.array([1, 2, 3], pa.int64())})
        parts = split_parts(PartRule("geo_rows", ("geo", "k"), scope_column="geo"), rows)
        assert [(part.name, part.rows["k"].to_pylist()) for part in parts] == [("12", [2]), ("7", [1, 3])]

    def test_a_row_of_no_scope_is_refused(self) -> None:
        rows = pa.table({"state": pa.array(["VA", None]), "k": pa.array([1, 2])})
        with pytest.raises(ValueError, match="hold no state"):
            split_parts(PartRule("t", ("state", "k"), scope_column="state"), rows)


class TestLocator:
    _RULE = PartRule("results", _RACE_KEY, scope_column="state", locator="office_key")

    def test_keys_without_the_scope_find_their_rows_part(self) -> None:
        rows = _rows(1_000)
        locator = Locator().extended(self._RULE, rows)
        keys = rows.select(list(_RACE_KEY))
        assert parts_of_keys(self._RULE, keys, locator).equals(rows["state"].combine_chunks())

    def test_a_locator_value_never_held_has_no_part(self) -> None:
        locator = Locator().extended(self._RULE, _rows(100))
        keys = pa.table({"office_key": pa.array(["no such race"]), "candidate_id": pa.array([1], pa.int64())})
        assert parts_of_keys(self._RULE, keys, locator).to_pylist() == [None]

    def test_a_value_moved_to_another_part_is_found_in_its_newest(self) -> None:
        first = pa.table({"state": ["VA", "NC"], "office_key": ["r1", "r2"], "candidate_id": [1, 2], "votes": [0, 0]})
        moved = pa.table({"state": ["GA"], "office_key": ["r1"], "candidate_id": [1], "votes": [0]})
        locator = Locator().extended(self._RULE, first).extended(self._RULE, moved)
        keys = pa.table({"office_key": ["r1", "r2"], "candidate_id": pa.array([1, 2], pa.int64())})
        assert parts_of_keys(self._RULE, keys, locator).to_pylist() == ["GA", "NC"]
        assert locator.located.num_rows == 2

    def test_rows_naming_one_locator_value_in_two_parts_are_refused(self) -> None:
        rows = pa.table({"state": ["VA", "NC"], "office_key": ["r1", "r1"], "candidate_id": [1, 2], "votes": [0, 0]})
        with pytest.raises(ValueError, match="two state"):
            Locator().extended(self._RULE, rows)

    def test_a_key_holding_its_scope_needs_no_locator(self) -> None:
        rule = PartRule("results", _SCOPED_KEY, scope_column="state")
        rows = _rows(50)
        assert parts_of_keys(rule, rows.select(list(_SCOPED_KEY))).equals(rows["state"].combine_chunks())

    def test_every_key_of_a_whole_table_is_in_its_one_part(self) -> None:
        keys = _rows(5).select(list(_SCOPED_KEY))
        assert parts_of_keys(PartRule("results", _SCOPED_KEY), keys).to_pylist() == [WHOLE_PART] * 5

    def test_a_located_table_without_its_locator_is_refused(self) -> None:
        keys = _rows(5).select(list(_RACE_KEY))
        with pytest.raises(ValueError, match="through its locator"):
            parts_of_keys(self._RULE, keys)


class TestArrowEndToEnd:
    """Arrow's buffers live in its own memory pool, which ``tracemalloc`` does not see; every Python object does (and
    every NumPy buffer). A split, a locator and a lookup over 200k rows allocate a few KB of them (one a part), where
    one object a row would be megabytes."""

    _ROWS = 200_000
    #: well under a byte a row: one Python object a row (a 28-byte int, a 50-byte str) is 5-10 MB here
    _BOUND = 256 << 10

    @staticmethod
    def _python_bytes(run: Callable[[], Any]) -> int:
        tracemalloc.start()
        try:
            run()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return peak

    def test_a_split_makes_no_python_object_a_row_and_its_parts_are_slices(self) -> None:
        rows = _rows(self._ROWS)
        rule = PartRule("results", _SCOPED_KEY, scope_column="state")
        assert self._python_bytes(lambda: split_parts(rule, rows)) < self._BOUND
        parts = split_parts(rule, rows)
        # every part is a slice of one sorted copy: its columns share that copy's buffers
        first, last = parts[0].rows["candidate_id"].chunks[0], parts[-1].rows["candidate_id"].chunks[0]
        assert first.buffers()[1].address == last.buffers()[1].address
        assert last.offset == rows.num_rows - len(last)

    def test_a_locator_and_a_lookup_make_no_python_object_a_key(self) -> None:
        rows = _rows(self._ROWS)
        rule = PartRule("results", _RACE_KEY, scope_column="state", locator="office_key")
        keys = rows.select(list(_RACE_KEY))
        assert self._python_bytes(lambda: Locator().extended(rule, rows)) < self._BOUND
        locator = Locator().extended(rule, rows)
        assert isinstance(locator.located, pa.Table)
        assert self._python_bytes(lambda: parts_of_keys(rule, keys, locator)) < self._BOUND
        assert self._python_bytes(lambda: parts_of_keys(PartRule("results", _RACE_KEY), keys)) < self._BOUND
