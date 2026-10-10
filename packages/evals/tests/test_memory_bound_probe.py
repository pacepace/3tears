"""A run's peak memory is bounded by its matrix, not by its total trace volume (#603).

The run loop keeps one :class:`~threetears.evals.kernel.scoring.CellSummary` per cell and hands each
full result and trace to storage, so what a run holds at once is one cell's record plus a summary per
cell. If it accumulated the records too, its peak would grow with every byte of trace the run produced,
and a large sweep could exhaust its container before it finished.

The probe drives a fixed toy-host matrix through the runner at a launch's default cell width, with each
cell's stored output padded to a chosen size, into a SQLite store on disk (whose pages live outside the
Python heap that ``tracemalloc`` sees, as a production store's live outside the process). It measures the
Python heap's peak at one trace volume and at twice that, and holds the peak to two bounds:

- doubling the trace volume at a fixed matrix raises the peak by less than :data:`MAX_GROWTH_SHARE` of the
  trace it added. The cells in flight hold their own records, so the peak grows with the size of one cell's
  trace times the width, never with the matrix; a run accumulating its records grows it by at least all of it;
- the peak stays under :data:`MAX_PEAK_BYTES`, well under the trace the matrix stores at the higher volume
  (accumulating holds all of it).
"""

from __future__ import annotations

import gc
import tracemalloc
from pathlib import Path

import pytest

from threetears.evals.kernel import EvalStorage
from threetears.evals.schema import EvalTestCase
from threetears.evals.kernel.candidate_kind import CandidateOutput, CellSink
from threetears.evals.kernel.host import WorldRegistry
from threetears.evals.run.executor import DEFAULT_MAX_CONCURRENT_CELLS
from threetears.evals.run.runner import RunnerOptions, execute_run
from threetears.evals.storage.sqlite import SqliteDocumentStore
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.kind import (
    TOY_EXTRACTOR_KIND,
    TOY_SCRIPTS,
    ScriptedExtractionClient,
    ToyExtractorInstance,
    ToyExtractorKind,
)
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_run, toyhost_template, toyhost_test_cases

pytestmark = pytest.mark.memory_probe

#: Repeats of each of the toy template's three cases: the fixed matrix.
K_RUNS = 48
N_CELLS = 3 * K_RUNS
#: The trace each cell stores at the lower volume, in bytes; the higher volume is twice it.
TRACE_BYTES = 16 * 1024
#: Doubling the trace volume may raise the peak by at most this share of the trace it added across the matrix.
#: A run holding only its cells in flight raised it by 0.06 of it (at 4 cells wide); one accumulating every
#: record raises it by more than the whole of it.
MAX_GROWTH_SHARE = 0.25
#: The most the run's peak heap may be at the higher volume, in bytes. Its cells store 4.5 MiB of trace between
#: them there, which a run accumulating its records holds all of at its end.
MAX_PEAK_BYTES = 3 * 512 * 1024


class _PaddedExtractorKind(ToyExtractorKind):
    """The toy extractor, its stored output padded to a fixed size: the trace volume the probe turns."""

    def __init__(
        self,
        *,
        padding_bytes: int,
        client: ScriptedExtractionClient,
        world: WorldRegistry,
        goal_checks: tuple[str, ...],
    ) -> None:
        super().__init__(client=client, world=world, goal_checks=goal_checks)
        self.padding_bytes = padding_bytes

    async def invoke(self, instance: ToyExtractorInstance, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Extract as the toy does, then pad the stored record.

        Args:
            instance: As the toy kind takes it.
            test_case: As the toy kind takes it.
            sink: As the toy kind takes it.

        Returns:
            The toy's output, its one document carrying ``padding_bytes`` of padding.
        """
        produced = await super().invoke(instance, test_case, sink)
        # A fresh string per cell, so no two cells share one object the heap would count once.
        padding = "".join(("x", test_case.id, str(len(self.measures)))).ljust(self.padding_bytes, "x")
        return produced.model_copy(update={"output": [{**produced.output[0], "padding": padding}]})


async def _peak_bytes(tmp_path: Path, *, padding_bytes: int) -> int:
    """Run the fixed matrix at one trace volume and return the Python heap's peak while it ran.

    Args:
        tmp_path: Where the run's SQLite store is written.
        padding_bytes: Each cell's stored padding.

    Returns:
        The peak traced heap, in bytes, above what was allocated when the run started.
    """
    store = SqliteDocumentStore(tmp_path / f"probe-{padding_bytes}.sqlite3")
    host = toyhost_host(storage=EvalStorage(store))
    world = host.profile.world
    assert world is not None
    template = toyhost_template()
    cases = toyhost_test_cases(template)
    kind = _PaddedExtractorKind(
        padding_bytes=padding_bytes,
        client=ScriptedExtractionClient(tuple(type(s)(**{**vars(s), "latency_s": 0.0}) for s in TOY_SCRIPTS)),
        world=world,
        goal_checks=tuple(template.goal_state_checks),
    )
    run = toyhost_run(model=RUN_MODELS[0], template=template, kind=kind, world=world).model_copy(
        update={"k_runs": K_RUNS, "test_case_ids": [case.id for case in cases]}
    )
    host.storage.save_eval_run(run)
    for case in cases:
        host.storage.save_test_case(case)
    # At the launch's default width, so the probe holds the run as a launched run executes it: several cells at once.
    options = RunnerOptions(
        candidate_kinds={TOY_EXTRACTOR_KIND: lambda _cell: kind}, max_concurrent_cells=DEFAULT_MAX_CONCURRENT_CELLS
    )

    gc.collect()
    tracemalloc.start()
    try:
        baseline, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        summaries = await execute_run(
            host, run=run, template=template, test_cases=cases, judge_service=None, options=options
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(summaries) == N_CELLS and all(summary.persisted for summary in summaries)
    stored = host.storage.query_eval_results_by_run(run.id, run.scope_id)
    assert len(stored) == N_CELLS, "every cell's record reached the store, so its trace volume was produced"
    return peak - baseline


async def test_a_runs_peak_memory_is_bounded_by_its_matrix_not_its_trace_volume(tmp_path: Path) -> None:
    # A warm-up drive, so allocations made once per process (caches, lazily built validators) land outside
    # the measured drives rather than inflating whichever runs first.
    warm = tmp_path / "warm"
    warm.mkdir()
    await _peak_bytes(warm, padding_bytes=1024)

    lower = await _peak_bytes(tmp_path, padding_bytes=TRACE_BYTES)
    higher = await _peak_bytes(tmp_path, padding_bytes=2 * TRACE_BYTES)

    added = N_CELLS * TRACE_BYTES
    growth = higher - lower
    assert growth < MAX_GROWTH_SHARE * added, (
        f"doubling each cell's trace from {TRACE_BYTES} to {2 * TRACE_BYTES} bytes over a fixed {N_CELLS}-cell matrix "
        f"raised the run's peak heap by {growth} bytes ({lower} -> {higher}), {growth / added:.2f} of the {added} "
        f"bytes of trace it added; a run holding only its cells in flight stays under {MAX_GROWTH_SHARE}"
    )
    assert higher < MAX_PEAK_BYTES, (
        f"the run's peak heap was {higher} bytes, above {MAX_PEAK_BYTES} for a {N_CELLS}-cell matrix storing "
        f"{N_CELLS * 2 * TRACE_BYTES} bytes of trace: the run is holding records it has already stored"
    )
