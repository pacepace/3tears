"""A run executes its cells concurrently unless its launch declares latency under test (#701).

Serial execution protects a latency reading from contention, and when latency is not under test it is
pure cost. These drive the real launch and the real runner over the toy host, whose extractor sleeps for
every call, and read what was stored:

* a launch that does not declare latency runs its cells side by side and finishes far sooner, and every
  cell it records says it was read under concurrency;
* a launch that declares it runs one cell at a time, its arms one after another with nothing beside
  them, and records it;
* a host's own executor runs the cells when it supplies one;
* the shared state of a run stays exact under concurrency: each cell's slice of the metered-call ledger
  is its own calls, the ceiling holds across cells calling at once, and the cost cap is exceeded by no
  more than the work in flight when it was reached.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest

from threetears.evals.schema import EvalRun
from threetears.evals.kernel import EvalStorage
from threetears.evals.schema.external_spend import ExternalSpend
from threetears.evals.run import (
    CellExecutor,
    CellWork,
    InProcessCellExecutor,
    LaunchHost,
    MeteredCallLedger,
    RunnerOptions,
    execute_run,
    start_run,
)
from threetears.evals.run.budget import BudgetStoppedError, EvalRunCostCap
from threetears.evals.run.executor import DEFAULT_MAX_CONCURRENT_CELLS
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.kind import (
    TOY_DOCUMENTS,
    TOY_EXTRACTOR_KIND,
    TOY_SCRIPTS,
    ExtractionResult,
    ScriptedExtractionClient,
    ToyExtractorKind,
)
from packages.evals.tests.fixtures.toyhost.product import ExtractionRequest
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS, toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_run, toyhost_template, toyhost_test_cases

#: How long every extraction sleeps where a test reads only how many ran at once.
_SLEEP_S = 0.02
#: How long it sleeps where a test times the matrix: long enough that a launch's own overhead does not
#: decide which finished first, short enough that the file runs in a few seconds.
_TIMED_SLEEP_S = 0.15
#: Repeats per case: the toy template has three cases, so twelve cells.
_K = 4


class _SleepingClient(ScriptedExtractionClient):
    """The toy extractor's client, every call sleeping ``sleep_s`` and counting how many are in flight."""

    def __init__(self, sleep_s: float = _SLEEP_S) -> None:
        super().__init__(tuple(replace(script, latency_s=sleep_s) for script in TOY_SCRIPTS))
        self.running = 0
        self.peak = 0

    async def extract(self, request: ExtractionRequest) -> ExtractionResult:
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            return await super().extract(request)
        finally:
            self.running -= 1


class _RecordingExecutor(CellExecutor):
    """A host's own executor: it records what it was asked to run, and runs it in-process."""

    def __init__(self) -> None:
        self.asked: list[tuple[int, int]] = []

    async def execute(self, cells: Sequence[CellWork], *, width: int) -> None:
        self.asked.append((len(cells), width))
        await InProcessCellExecutor().execute(cells, width=width)


def _launching(
    client: ScriptedExtractionClient, executor: CellExecutor | None = None, **settings: Any
) -> tuple[LaunchHost, EvalStorage]:
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_template(toyhost_template())
    launch_settings = TOYHOST_LAUNCH_SETTINGS.model_copy(update=settings)
    host, _client = toyhost_launch_host(
        storage=storage, client=client, cell_executor=executor, settings=lambda: launch_settings
    )
    return host, storage


async def _launch(host: LaunchHost, *, models: Sequence[str] = RUN_MODELS[:1], **arguments: Any) -> list[EvalRun]:
    runs = await start_run(
        host,
        template_id=toyhost_template().id,
        scope_id=TOYHOST_SCOPE,
        subject_id=TOYHOST_SUBJECT.subject_id,
        models=list(models),
        k_runs=_K,
        **arguments,
    )
    async with asyncio.timeout(30):
        while any(host.job_manager.is_active(run.id) for run in runs):
            await asyncio.sleep(0.005)
    return runs


def _modes(storage: EvalStorage, run: EvalRun) -> set[Any]:
    return {
        result.covariates.get("execution_mode") for result in storage.query_eval_results_by_run(run.id, run.scope_id)
    }


# =============================================================================
# Latency not under test: the cells run side by side, faster, and say so
# =============================================================================


async def test_a_run_not_measuring_latency_runs_its_cells_concurrently_and_measurably_faster() -> None:
    concurrent_client, serial_client = _SleepingClient(_TIMED_SLEEP_S), _SleepingClient(_TIMED_SLEEP_S)
    concurrent_host, concurrent_store = _launching(concurrent_client)
    serial_host, serial_store = _launching(serial_client)

    began = time.perf_counter()
    (concurrent,) = await _launch(concurrent_host)
    concurrent_s = time.perf_counter() - began
    began = time.perf_counter()
    (serial,) = await _launch(serial_host, measure_latency=True)
    serial_s = time.perf_counter() - began

    cells = len(toyhost_test_cases(toyhost_template())) * _K
    assert concurrent_client.peak == DEFAULT_MAX_CONCURRENT_CELLS, "the cells ran up to the default width at once"
    assert serial_client.peak == 1, "a run measuring latency ran one cell at a time"
    assert len(concurrent_client.calls) == len(serial_client.calls) == cells, "both ran the whole matrix"
    # Twelve cells of 150 ms: about 1.8 s one at a time and 0.45 s four at a time. Half is a wide margin.
    assert concurrent_s < serial_s / 2, f"concurrent {concurrent_s:.3f}s is not measurably faster than {serial_s:.3f}s"

    stored_concurrent = concurrent_store.load_eval_run(concurrent.id, TOYHOST_SCOPE)
    stored_serial = serial_store.load_eval_run(serial.id, TOYHOST_SCOPE)
    assert stored_concurrent is not None and stored_serial is not None
    assert (stored_concurrent.status, stored_serial.status) == ("completed", "completed")
    assert (stored_concurrent.measure_latency, stored_concurrent.cell_concurrency) == (
        False,
        DEFAULT_MAX_CONCURRENT_CELLS,
    )
    assert (stored_serial.measure_latency, stored_serial.cell_concurrency) == (True, 1)
    # The mark: every cell of the concurrent run says its latency was read under concurrency, and the
    # serial run, executing alone in the manager, says it was not.
    assert _modes(concurrent_store, concurrent) == {"concurrent"}
    assert _modes(serial_store, serial) == {"serial"}


async def test_the_width_is_the_hosts_setting() -> None:
    client = _SleepingClient()
    host, storage = _launching(client, max_concurrent_cells=2)

    (run,) = await _launch(host)

    stored = storage.load_eval_run(run.id, TOYHOST_SCOPE)
    assert stored is not None and stored.cell_concurrency == 2
    assert client.peak == 2


# =============================================================================
# Latency under test: serial cells, serial arms, nothing beside them
# =============================================================================


async def test_a_launch_measuring_latency_runs_its_arms_one_after_another_with_nothing_beside_them() -> None:
    client = _SleepingClient()
    host, storage = _launching(client)

    runs = await _launch(host, models=RUN_MODELS[:2], measure_latency=True)

    assert client.peak == 1, "two arms of a latency launch ran side by side"
    for run in runs:
        stored = storage.load_eval_run(run.id, TOYHOST_SCOPE)
        assert stored is not None and stored.status == "completed"
        assert (stored.measure_latency, stored.cell_concurrency) == (True, 1)
        assert _modes(storage, run) == {"serial"}, "an arm of a latency launch read its latency beside another"


async def test_the_arms_of_a_launch_not_measuring_latency_still_run_side_by_side() -> None:
    client = _SleepingClient()
    host, storage = _launching(client)

    runs = await _launch(host, models=RUN_MODELS[:2])

    assert client.peak > DEFAULT_MAX_CONCURRENT_CELLS, "the two arms' cells should overlap, each run at its width"
    for run in runs:
        assert _modes(storage, run) == {"concurrent"}


# =============================================================================
# The executor seam
# =============================================================================


async def test_a_host_supplied_executor_runs_every_cell_at_the_width_the_launch_decided() -> None:
    executor = _RecordingExecutor()
    client = _SleepingClient()
    host, storage = _launching(client, executor)

    (concurrent,) = await _launch(host)
    (serial,) = await _launch(host, measure_latency=True)

    cells = len(toyhost_test_cases(toyhost_template())) * _K
    assert executor.asked == [(cells, DEFAULT_MAX_CONCURRENT_CELLS), (cells, 1)]
    for run in (concurrent, serial):
        assert len(storage.query_eval_results_by_run(run.id, TOYHOST_SCOPE)) == cells


async def test_the_default_executor_starts_cells_in_order_and_raises_the_first_failure() -> None:
    started: list[int] = []

    def cell(index: int) -> CellWork:
        async def work() -> None:
            started.append(index)
            await asyncio.sleep(0.01)
            if index == 2:
                raise RuntimeError("cell 2 broke")

        return work

    with pytest.raises(RuntimeError, match="cell 2 broke"):
        await InProcessCellExecutor().execute([cell(index) for index in range(10)], width=3)

    assert started[:3] == [0, 1, 2], "cells start in the order given"
    assert len(started) < 10, "no further cell starts once one has raised"


# =============================================================================
# Shared state under concurrency
# =============================================================================


async def test_each_cell_is_credited_its_own_metered_calls_and_the_ceiling_holds_across_cells() -> None:
    ledger = MeteredCallLedger("run-1", ceiling=7)
    spend = ExternalSpend(provider="search", calls=1, provider_units=2, provider_unit="credits")
    tallies: dict[int, Any] = {}

    async def cell(index: int, calls: int) -> None:
        with ledger.cell() as meter:
            for _ in range(calls):
                ledger.admit(tool="search", action="query", spend=spend)
                await asyncio.sleep(0)  # let the other cells call in between
            tallies[index] = meter.tally()

    await asyncio.gather(cell(0, 3), cell(1, 3), cell(2, 3))

    run = ledger.tally()
    assert (run.calls, run.refused) == (7, 2), "the ceiling bounds the run, however many cells call at once"
    assert sum(tally.calls for tally in tallies.values()) == run.calls
    assert sum(tally.refused for tally in tallies.values()) == run.refused
    assert all(tally.calls + tally.refused == 3 for tally in tallies.values()), "each cell is credited its own"
    assert ledger.open_cell() is None, "no cell's meter outlives its block"


async def test_the_cost_cap_is_exceeded_by_no_more_than_the_cells_in_flight_when_it_was_reached() -> None:
    host = toyhost_host()
    world = host.profile.world
    assert world is not None
    template = toyhost_template()
    client = _SleepingClient()
    kind = ToyExtractorKind(client=client, world=world, goal_checks=tuple(template.goal_state_checks))
    cases = toyhost_test_cases(template)
    run = toyhost_run(model=RUN_MODELS[0], template=template, kind=kind, world=world).model_copy(update={"k_runs": _K})
    host.storage.save_template(template)
    for case in cases:
        host.storage.save_test_case(case)
    host.storage.save_eval_run(run)
    script = next(script for script in TOY_SCRIPTS if script.model == RUN_MODELS[0])
    dearest = max(script.cost_of(document) for document in TOY_DOCUMENTS)
    cap = EvalRunCostCap(run.id, max_cost_usd=dearest * 1.5, enabled=True)
    width = 4

    with pytest.raises(BudgetStoppedError):
        await execute_run(
            host,
            run=run,
            template=template,
            test_cases=cases,
            judge_service=None,
            options=RunnerOptions(candidate_kinds={TOY_EXTRACTOR_KIND: lambda _cell: kind}, max_concurrent_cells=width),
            budget_gate=cap.check,
            on_cost=cap.record,
        )

    results = host.storage.query_eval_results_by_run(run.id, run.scope_id)
    spent = sum(result.cost_usd or 0.0 for result in results)
    assert spent > cap.max_cost_usd, "the cap was reached"
    assert spent <= cap.max_cost_usd + width * dearest, "the cap was overshot by more than the cells in flight"
    assert len(results) < len(cases) * _K, "no cell started once the cap was reached"
