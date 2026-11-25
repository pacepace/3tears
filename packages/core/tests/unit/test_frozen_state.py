"""Read-only snapshots: each change makes a new one and leaves the old one as it was; the module imports stdlib only."""

from __future__ import annotations

import random
import subprocess
import sys
from collections.abc import Mapping

import pytest

from threetears.core.frozen_state import (
    EMPTY,
    LoggedMap,
    frozen,
    logged,
    with_chunk,
    with_entries,
    with_entry,
    with_newest,
    without_entries,
    without_entry,
)

#: refuses pyarrow, asyncpg and the collections package at import, as shared_arrow's cycle and an install
#: without the arrow extra would
BLOCK_HEAVY = """
import importlib.abc, sys
class _Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("pyarrow", "asyncpg") or name.startswith("threetears.core.collections"):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None
sys.meta_path.insert(0, _Refuse())
"""


def test_a_change_makes_a_new_snapshot_and_leaves_the_one_a_reader_holds() -> None:
    held = with_entry(EMPTY, "a", 1)
    changed = with_entry(held, "b", 2)

    assert dict(held) == {"a": 1}
    assert dict(changed) == {"a": 1, "b": 2}
    assert dict(without_entries(changed, ["a"])) == {"b": 2}
    assert dict(without_entry(changed, "b")) == {"a": 1}
    assert dict(with_entries(held, {"c": 3})) == {"a": 1, "c": 3}
    assert dict(frozen([("x", 9)])) == {"x": 9}
    with pytest.raises(TypeError):
        changed["c"] = 3  # type: ignore[index]


def test_removing_nothing_hands_back_the_same_snapshot() -> None:
    held = with_entry(EMPTY, "a", 1)

    assert without_entry(held, "z") is held
    assert without_entries(held, ["z"]) is held


def test_the_newest_goes_last_and_the_oldest_go_past_the_limit() -> None:
    held = with_newest(with_newest(with_newest(EMPTY, "a", 1), "b", 2), "c", 3)

    assert list(with_newest(held, "a", 1)) == ["b", "c", "a"]
    assert list(with_newest(held, "d", 4, limit=2)) == ["c", "d"]
    assert list(held) == ["a", "b", "c"]


def test_a_chunked_log_keeps_every_entry_in_order_in_few_chunks() -> None:
    log: tuple[tuple[int, ...], ...] = ()
    seen = []
    for each in range(1_000):
        before = log
        log = with_chunk(log, (each,))
        seen.append(before)

    assert [entry for chunk in log for entry in chunk] == list(range(1_000))
    assert len(log) <= 11
    assert [entry for chunk in seen[500] for entry in chunk] == list(range(500))
    assert with_chunk(log, ()) is log
    assert [entry for chunk in with_chunk(log, (7, 8)) for entry in chunk][-2:] == [7, 8]


def test_it_imports_without_pyarrow_asyncpg_or_the_collections() -> None:
    code = BLOCK_HEAVY + "import threetears.core.frozen_state\n"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert done.returncode == 0, done.stderr


def test_a_logged_map_reads_as_a_dict_changed_the_same_way_order_included() -> None:
    """Flips if a :class:`LoggedMap` set and unset a key at a time reads otherwise than a dict changed alike: its
    keys, their order (a key set again keeps its place, one re-added after its removal takes the last), values, size."""
    for seed in range(100):
        draw = random.Random(seed)
        expected: dict[int, float] = {}
        held: Mapping[int, float] = logged()
        for _ in range(draw.randint(1, 300)):
            key = draw.randint(0, 30)
            if draw.random() < 0.3:
                expected.pop(key, None)
                held = without_entry(held, key)
            else:
                value = draw.random()
                expected[key] = value
                held = with_entry(held, key, value)
            assert isinstance(held, LoggedMap)
            assert list(held) == list(expected) and len(held) == len(expected) and held == expected
            assert all((key in held) == (key in expected) for key in range(32))


def test_a_logged_map_copies_a_key_about_log2_n_times_and_leaves_its_snapshots_as_they_were() -> None:
    """Flips if setting a key at a time copies the whole map a change (n copies a key, quadratic), or changes a snapshot
    already handed out."""
    held: Mapping[int, int] = logged()
    snapshots = []
    for key in range(4096):
        held = with_entry(held, key, key)
        assert isinstance(held, LoggedMap)
        # a chunk a set bit of the size, each at most half the one before: a key is merged once each time its chunk
        # doubles, about log2(n) times
        assert len(held.chunks) <= (key + 1).bit_length()
        if key % 1024 == 0:
            snapshots.append(held)
    assert len(held) == 4096
    assert [len(snapshot) for snapshot in snapshots] == [1, 1025, 2049, 3073]
    assert dict(snapshots[1]) == {key: key for key in range(1025)}
