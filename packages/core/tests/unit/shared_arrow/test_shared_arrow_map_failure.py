"""A version that cannot be mapped whole raises why, and leaves nothing it mapped open.

A reader maps one file a part, each holding a descriptor while mapped (2240 for the reports pod's version). Out of
descriptors part way, the constructor unmapped what it had mapped through :meth:`MappedTables.close`, which read
``tables`` before the constructor had set it: the runner saw ``AttributeError: 'MappedTables' object has no attribute
'tables'`` in place of the ``OSError`` (Too many open files) that stopped it (brooks-5950x, 2026-11-21).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from threetears.core import shared_arrow
from threetears.core.shared_arrow import MappedVersion, Parts, SharedArrowStore


def test_a_version_mapped_part_way_raises_its_own_error_and_unmaps_what_it_mapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SharedArrowStore(tmp_path / "shm", max_bytes=1 << 30)
    part = pa.table({"a": pa.array(range(4), pa.int64())})
    store.publish("v1", {"long": Parts(schema=part.schema, parts={f"p{index:03d}": part for index in range(12)})})
    mapped_before = 0
    opened: list[Any] = []
    real = pa.memory_map

    def runs_out(file: Any, mode: str = "r") -> Any:
        nonlocal mapped_before
        if mapped_before == 5:
            raise OSError(24, "Too many open files")
        mapped_before += 1
        opened.append(real(file, mode))
        return opened[-1]

    # shared_arrow maps through the one pyarrow module, so patching it there is patching it in shared_arrow
    monkeypatch.setattr(pa, "memory_map", runs_out)
    with pytest.raises(OSError, match="Too many open files"):
        MappedVersion(store.root, "v1")
    assert mapped_before == 5
    # the maps this version opened, each closed: counting the process's descriptors instead read another test's
    # file closing in the same worker as a leak (19 before, 18 after, on a shared runner)
    assert len(opened) == 5
    assert all(handle.closed for handle in opened), "a version mapped part way left a map open"


def test_a_let_go_frees_its_tables_handing_the_lock_over_between_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if the let-go frees a version's tables leaving the event loop to the switch interval between them: an idle
    reader letting 2236 lent files go held the loop 20-31 ms (the let-go loop-timing test, bound 20 ms)."""
    store = SharedArrowStore(tmp_path / "shm", max_bytes=1 << 30)
    part = pa.table({"a": pa.array(range(4), pa.int64())})
    store.publish("v1", {"long": Parts(schema=part.schema, parts={f"p{index:03d}": part for index in range(5)})})
    mapped = MappedVersion(store.root, "v1")
    freed = len(mapped.by_inode) + len(mapped.tables)
    turns: list[int] = []

    class _Turns:
        def __init__(self, loop: object = None) -> None:
            # the loop a let-go hands its turns to: none here, the mapping closed off any loop
            assert loop is None

        def take(self) -> None:
            turns.append(1)

    monkeypatch.setattr(shared_arrow, "LoopTurns", _Turns, raising=False)
    mapped.close()
    assert freed >= 6
    assert len(turns) == freed, "a table was freed without a turn after it"
