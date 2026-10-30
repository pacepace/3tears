"""A compact per-result reading holds at most a fifth of the bytes of the result it was built from (W4b P1).

A reader that folds a scope holds every result it read, so a result's bytes bound how many results one read can
answer. Measured at the production shape (the hot-path corpus at 1x: about 3,100 results of about 6.8 KB stored),
each row handed out as a SQL store hands it, freshly decoded from JSON so no two rows share a string, and read
through ``EvalStorage.iter_result_readings``, which holds a read's repeated values once.

Measured 2026-10-30 (tracemalloc, CPython 3.14): a hydrated ``EvalResult`` holds about 28.9 KB, and its reading
about 3.4 KB (0.12 of it). The reading's own bytes are mostly its record lists' slim items (usage rows and scores)
and its three open mappings (host measures, phase timings, covariates), which hold per-result values.
"""

from __future__ import annotations

import functools
import gc
import json
import tracemalloc
from collections.abc import Mapping, Sequence
from typing import Any

from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.schema.models import EvalResult
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.hot_path_support import HOT_PATH_SCOPE, hot_path_corpus

#: The most of an ``EvalResult``'s bytes its reading may hold.
MOST_OF_A_RESULT = 1 / 5

#: The most bytes a reading may hold at the production shape: the measured 3.4 KB with room for allocator noise.
READING_BYTES = 4096


class _DecodedRowsStore(InMemoryDocumentStore):
    """The reference store, handing each row out freshly decoded from JSON, as a SQL store's driver does."""

    def by_doc_type(
        self, doc_type: str, scope_id: str, *, field_in: Mapping[str, Sequence[Any]] | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        rows = super().by_doc_type(doc_type, scope_id, field_in=field_in, **kwargs)
        decoded: list[dict[str, Any]] = json.loads(json.dumps(rows))
        return decoded

    def get_many(self, doc_type: str, doc_ids: Sequence[str], scope_id: str) -> list[dict[str, Any]]:
        decoded: list[dict[str, Any]] = json.loads(json.dumps(super().get_many(doc_type, doc_ids, scope_id)))
        return decoded


@functools.cache
def _storage() -> EvalStorage:
    """The production-shaped campaign at 1x, read through a store that decodes each row."""
    store = _DecodedRowsStore()
    hot_path_corpus(scale=1, store=store)
    return EvalStorage(store)


def _held_bytes(build: Any) -> tuple[int, int]:
    """The bytes ``build()``'s answer holds once built, and how many items it holds."""
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        held = build()
        gc.collect()
        after = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    return after - before, len(held)


def test_a_reading_holds_at_most_a_fifth_of_its_result() -> None:
    storage = _storage()
    result_bytes, n_results = _held_bytes(lambda: storage.query_eval_results(HOT_PATH_SCOPE))
    reading_bytes, n_readings = _held_bytes(lambda: list(storage.iter_result_readings(None, HOT_PATH_SCOPE)))

    per_result, per_reading = result_bytes / n_results, reading_bytes / n_readings
    assert n_readings == n_results > 3000
    assert per_reading <= MOST_OF_A_RESULT * per_result, (
        f"a reading holds {per_reading:,.0f} bytes, {per_reading / per_result:.2f} of its result's {per_result:,.0f}"
    )
    assert per_reading <= READING_BYTES, f"a reading holds {per_reading:,.0f} bytes"


def test_a_read_holds_no_result_past_its_page() -> None:
    """Every result hydrated for a page is dropped once its readings are built: none outlives the read."""
    storage = _storage()
    before = _results_alive()

    readings = list(storage.iter_result_readings(None, HOT_PATH_SCOPE))

    alive = _results_alive() - before
    assert len(readings) > 3000 and alive == 0, f"{alive} hydrated results outlived the read"


def _results_alive() -> int:
    """How many hydrated results are alive now."""
    gc.collect()
    return sum(1 for held in gc.get_objects() if type(held) is EvalResult)
