"""The store makes no reference cycle: reference counting frees it.

Why: a pod that runs no full collection (reports' ``pod.interpreter``, 2026-11-25: one walked the whole heap with the
interpreter lock held, up to 0.94 s of the serving loop) frees only what reference counting frees. A store let go held
itself through its held-files tally's callback (a bound method of the store's, held by an object the store holds), and
kept its tally (Arrow batches of every directory's files) until a full collection.
"""

from __future__ import annotations

import gc
import threading
import weakref
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pytest

from threetears.core.shared_arrow import SharedArrowStore


@pytest.fixture
def collector_off() -> Iterator[None]:
    """the cycle collector off for the test: only reference counting frees anything."""
    was = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was:
            gc.enable()


@pytest.mark.usefixtures("collector_off")
def test_a_store_let_go_is_freed_by_reference_counting(tmp_path: Path) -> None:
    store = SharedArrowStore(tmp_path / "shm", max_bytes=64 << 20)
    store.publish("v1", {"t": pa.table({"a": list(range(100))})})
    store.publish("v2", {"t": pa.table({"a": list(range(50))})})
    # the sweep a publish leaves runs on the store's thread, which holds the store while it runs: done first
    store.settle()
    for thread in threading.enumerate():
        if thread.name == "shared-arrow-settle":
            thread.join(timeout=30)
    gone = weakref.ref(store)
    del store
    assert gone() is None, "the store was left in a reference cycle"
