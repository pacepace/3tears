"""Packed key batches: one string a batch of keys, exact texts, searched and bisected in C; the module imports stdlib
only."""

from __future__ import annotations

import subprocess
import sys

import pytest

from threetears.core import packed_keys

#: refuses pyarrow, numpy, asyncpg and the collections package at import, as an install without the arrow extra would
BLOCK_HEAVY = """
import importlib.abc, sys
class _Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("pyarrow", "numpy", "asyncpg") or name.startswith("threetears.core.collections"):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None
sys.meta_path.insert(0, _Refuse())
"""


def test_it_imports_without_pyarrow_numpy_asyncpg_or_the_collections() -> None:
    code = BLOCK_HEAVY + "import threetears.core.packed_keys\nimport pyarrow\n"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    # the module imported; the block then refused pyarrow, so the block was live
    assert "No module named 'pyarrow'" in done.stderr, done.stderr
    assert "threetears.core.packed_keys" not in done.stderr


def test_a_key_text_is_exact() -> None:
    """Flips if two different keys share a text: a part holding the separator, or a key split differently."""
    keys = [("a", "b"), ("a\x1fb",), ("a\x00b",), ("a", "\x1eb"), ("",), ("", ""), ("ab",), ()]
    texts = [packed_keys.key_text(key) for key in keys]
    assert len(set(texts)) == len(keys)
    assert not any(packed_keys.SEP in text for text in texts)


def test_a_batch_names_its_keys_and_no_other() -> None:
    """Flips if a batch names a key it does not hold: one holding another's text inside it."""
    batch = packed_keys.packed([packed_keys.key_text(("ab", "c")), packed_keys.key_text(("x",))])
    assert packed_keys.names(batch, packed_keys.key_text(("ab", "c")))
    assert packed_keys.names(batch, packed_keys.key_text(("x",)))
    for other in (("a",), ("b", "c"), ("ab",), ("c",), ("x", "y")):
        assert not packed_keys.names(batch, packed_keys.key_text(other))
    assert not packed_keys.names(packed_keys.NO_KEYS, "")


def test_a_batch_is_a_multiset_and_loses_one_entry_at_a_time() -> None:
    batch = packed_keys.with_text(packed_keys.with_text(packed_keys.with_text(packed_keys.NO_KEYS, "a"), "b"), "a")
    assert packed_keys.texts(batch) == ["a", "b", "a"]
    batch = packed_keys.without_text(batch, "a")
    assert packed_keys.texts(batch) == ["b", "a"] and packed_keys.names(batch, "a")
    batch = packed_keys.without_text(packed_keys.without_text(batch, "a"), "b")
    assert batch == packed_keys.NO_KEYS and packed_keys.texts(batch) == []
    with pytest.raises(ValueError):
        packed_keys.without_text(batch, "a")


def test_a_sorted_batch_finds_each_text_by_bisection() -> None:
    texts = sorted(packed_keys.key_text((f"s{i % 7}", f"{i:05d}")) for i in range(1_000))
    batch, offsets = packed_keys.packed(texts), packed_keys.offsets_of(texts)
    assert [packed_keys.text_at(batch, offsets, row) for row in range(len(texts))] == texts
    assert all(packed_keys.find_sorted(batch, offsets, text) == row for row, text in enumerate(texts))
    for missing in ("", "s0", "s9\x1f00001", "￿"):
        assert packed_keys.find_sorted(batch, offsets, missing) == -1
    empty = packed_keys.packed([])
    assert packed_keys.find_sorted(empty, packed_keys.offsets_of([]), "a") == -1
