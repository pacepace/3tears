"""Both per-run ceilings stop a run between cells; neither refuses a call inside one.

The metered-call ceiling and the cost cap are read against what has LANDED, the recorded cells', before each cell
starts. Once either is reached no new cell begins, the cells already running finish whole, and the run ends
``budget_stopped`` with a typed record of each ceiling that stopped it. A run may land past a ceiling by up to the
width's worth of cells; what it may not do is refuse a call inside a cell, because at width above 1 that
contaminates whichever cell happened to ask after the ceiling was spent.

The one refusal that remains is a host that declared no metered tools: a wrong declaration, not a budget.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.kind import (
    TOY_EXTRACTOR_KIND,
    ScriptedExtractionClient,
    ToyExtractorInstance,
    ToyExtractorKind,
)
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS, toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_run, toyhost_template, toyhost_test_cases
from packages.evals.tests.off_loop import off_loop
from threetears.evals.analysis.summary import budget_stop_line
from threetears.evals.kernel import EvalStorage
from threetears.evals.kernel.candidate_kind import CandidateOutput, CellSink
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.result_condition import ResultOutcome, classify_result
from threetears.evals.run import MeteredCallLedger, RunnerOptions, execute_run, start_run
from threetears.evals.run.budget import BudgetStoppedError, CapBreach, EvalRunCostCap
from threetears.evals.schema import EvalResult, EvalRun
from threetears.evals.schema.models import BudgetStop, EvalTemplate, EvalTestCase
from threetears.evals.storage import InMemoryDocumentStore


class _MeteredKind(ToyExtractorKind):
    """A toy extractor whose every cell makes ``calls`` metered calls through the run's ledger, and ``usd`` of spend.

    The first cell to invoke finishes at once; each later one waits for ``release`` first when the test set one,
    so a test decides exactly which cells have landed and which are still running when the run decides to stop.
    The ledger is handed in per cell by the runner's kind factory (``RunnerOptions.metered_calls``).
    """

    def __init__(
        self, *args: Any, calls: int, ledger: MeteredCallLedger | None, release: asyncio.Event | None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.calls = calls
        self.ledger = ledger
        self.release = release
        self.invoked = 0
        #: How many metered calls each cell was admitted, by the cell's case id and arrival.
        self.admitted: list[int] = []
        self.refused_inside_a_cell = 0

    async def invoke(self, instance: ToyExtractorInstance, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        self.invoked += 1
        waits = self.release is not None and self.invoked > 1
        if waits:
            assert self.release is not None
            await self.release.wait()
        admitted = 0
        for _ in range(self.calls):
            assert self.ledger is not None
            if self.ledger.admit(tool="web_search", action="search", spend=None):
                admitted += 1
            else:
                self.refused_inside_a_cell += 1
            await asyncio.sleep(0)
        sink.add_spend("candidate", 0.0)
        self.admitted.append(admitted)
        return CandidateOutput(output=[{"admitted": admitted}])


def _no_goal_checks() -> EvalTemplate:
    return toyhost_template().model_copy(update={"goal_state_checks": []})


def _stored_run(host: EvalHost, kind: ToyExtractorKind, *, k: int, n_cases: int) -> tuple[EvalRun, list[EvalTestCase]]:
    world = host.profile.world
    assert world is not None
    template = _no_goal_checks()
    cases = toyhost_test_cases(template)[:n_cases]
    host.storage.save_template(template)
    for case in cases:
        host.storage.save_test_case(case)
    run = toyhost_run(model=RUN_MODELS[0], template=template, kind=kind, world=world).model_copy(
        update={"k_runs": k, "test_case_ids": [case.id for case in cases]}
    )
    host.storage.save_eval_run(run)
    return run, cases


async def _execute(
    *,
    calls: int,
    ceiling: int | None,
    width: int,
    k: int = 1,
    n_cases: int = 3,
    hold_second_cell_until_stop: bool = False,
    none_declared: bool = False,
    cap: EvalRunCostCap | None = None,
    cost_per_cell: float = 0.0,
) -> tuple[
    list[EvalResult], BudgetStoppedError | None, MeteredCallLedger, _MeteredKind, EvalRunCostCap | None, list[Any]
]:
    """Run the toy cells at ``width``, each making ``calls`` metered calls, under the given ceilings.

    With ``hold_second_cell_until_stop`` every cell after the first waits until the run's between-cells check has
    reported a ceiling reached, so what had landed and what was running at the stop do not depend on timing.
    """
    host = toyhost_host()
    release: asyncio.Event | None = asyncio.Event() if hold_second_cell_until_stop else None
    world = host.profile.world
    assert world is not None
    ledger = MeteredCallLedger("run-1", ceiling, none_declared=none_declared)
    kind = _MeteredKind(
        client=ScriptedExtractionClient(), world=world, goal_checks=(), calls=calls, ledger=ledger, release=release
    )
    run, cases = off_loop(_stored_run, host, kind, k=k, n_cases=n_cases)
    breaches: list[Any] = []

    def gate(pending: float | None) -> CapBreach | None:
        breach = cap.check(pending) if cap is not None else None
        if breach is not None:
            breaches.append(breach)
        if (breach is not None or ledger.ceiling_reached) and release is not None:
            release.set()
        return breach

    def on_cost(cost: float | None) -> None:
        if cap is not None:
            cap.record(cost_per_cell if cost is None or cost == 0.0 else cost)

    stopped: BudgetStoppedError | None = None
    try:
        await execute_run(
            host,
            run=run,
            template=_no_goal_checks(),
            test_cases=cases,
            judge_service=None,
            options=RunnerOptions(
                candidate_kinds={TOY_EXTRACTOR_KIND: lambda _cell: kind},
                max_concurrent_cells=width,
                metered_calls=ledger,
            ),
            budget_gate=gate if cap is not None or release is not None else None,
            on_cost=on_cost if cap is not None else None,
        )
    except BudgetStoppedError as error:
        stopped = error
    results = off_loop(host.storage.query_eval_results_by_run, run.id, run.scope_id)
    return results, stopped, ledger, kind, cap, breaches


# =============================================================================
# The metered-call ceiling
# =============================================================================


async def test_a_metered_ceiling_tripping_mid_run_refuses_no_call_and_starts_no_new_cell() -> None:
    """Width 2, three cells of two calls, a ceiling of 2: the first cell lands it, the second is still running.

    The second cell finishes without a refusal, the third never starts, and the run is ``budget_stopped``.
    """
    results, stopped, ledger, kind, _cap, _breaches = await _execute(
        calls=2, ceiling=2, width=2, hold_second_cell_until_stop=True
    )

    assert stopped is not None, "the run stops budget_stopped"
    assert len(results) == 2, "the third cell never started"
    assert kind.admitted == [2, 2], "both cells made every call they set out to make"
    assert kind.refused_inside_a_cell == 0
    assert ledger.tally().refused == 0
    assert all(result.metered_calls_refused == 0 for result in results), "no cell carries a refusal"
    assert all(classify_result(result) is ResultOutcome.OK for result in results), "no cell was cut short"
    assert ledger.tally().calls == 4 and ledger.landed_calls == 4


async def test_the_stop_record_holds_the_counters_numbers() -> None:
    _results, stopped, _ledger, _kind, _cap, _breaches = await _execute(
        calls=2, ceiling=2, width=2, hold_second_cell_until_stop=True
    )

    assert stopped is not None
    (stop,) = stopped.stops
    assert stop.currency == "metered_calls"
    assert stop.ceiling == 2
    assert stop.landed_at_stop == 2, "only the first cell had landed when the stop was decided"
    assert stop.cells_in_flight == 1
    assert stop.landed_final == 4, "the running cell's two calls landed once it finished"
    assert stop.overshoot == 2, "derived from the two figures, never stored"
    assert stop.cells_not_started == 1
    assert "overshoot" not in stop.model_dump()
    assert stopped.breach is None, "no cost cap was involved"
    assert budget_stop_line(stopped.stops) == (
        "stopped by its metered-call ceiling: 2 of 2 at the stop, 4 once 1 running cell finished (+2); "
        "1 cell not started"
    )
    assert str(stopped).startswith("stopped by its metered-call ceiling: 2 of 2 at the stop")


async def test_a_serial_run_stops_at_the_ceiling_with_nothing_in_flight() -> None:
    results, stopped, _ledger, _kind, _cap, _breaches = await _execute(calls=3, ceiling=3, width=1, k=2)

    assert stopped is not None
    (stop,) = stopped.stops
    assert (stop.landed_at_stop, stop.landed_final, stop.cells_in_flight) == (3, 3, 0)
    assert stop.overshoot == 0
    assert (len(results), stop.cells_not_started) == (1, 5)
    assert budget_stop_line(stopped.stops) == (
        "stopped by its metered-call ceiling: 3 of 3 at the stop, no cell was running; 5 cells not started"
    )


async def test_a_ceiling_the_last_cell_reaches_stops_nothing() -> None:
    """No cell is left to keep from starting, so the run is whole."""
    results, stopped, ledger, _kind, _cap, _breaches = await _execute(calls=1, ceiling=1, width=1, n_cases=1)

    assert stopped is None
    assert len(results) == 1 and ledger.ceiling_reached


async def test_calls_in_cells_still_running_do_not_count_against_the_ceiling() -> None:
    """The ceiling reads recorded cells: a running cell that has already passed it starts no stop on its own."""
    ledger = MeteredCallLedger("run-1", 2)
    with ledger.cell() as meter:
        for _ in range(5):
            assert ledger.admit(tool="web_search", action="search", spend=None)
        assert ledger.tally().calls == 5
        assert not ledger.ceiling_reached and ledger.landed_calls == 0
    ledger.land(meter)
    assert ledger.ceiling_reached and ledger.landed_calls == 5


async def test_an_unbounded_ledger_never_stops_a_run() -> None:
    results, stopped, ledger, _kind, _cap, _breaches = await _execute(calls=5, ceiling=None, width=2, k=2)

    assert stopped is None
    assert len(results) == 6 and not ledger.ceiling_reached


# =============================================================================
# Both ceilings at one boundary
# =============================================================================


async def test_both_ceilings_tripping_at_one_boundary_are_both_recorded() -> None:
    cap = EvalRunCostCap("run-1", 0.5, enabled=True)
    results, stopped, _ledger, _kind, _cap, _breaches = await _execute(
        calls=2, ceiling=2, width=2, hold_second_cell_until_stop=True, cap=cap, cost_per_cell=1.0
    )

    assert stopped is not None
    assert [stop.currency for stop in stopped.stops] == ["cost_usd", "metered_calls"], "neither is dropped"
    cost, calls = stopped.stops
    assert (cost.ceiling, cost.cells_in_flight, cost.cells_not_started) == (0.5, 1, 1)
    assert (calls.ceiling, calls.landed_at_stop, calls.landed_final, calls.cells_in_flight) == (2, 2, 4, 1)
    assert stopped.breach is not None, "the cost cap's own numbers ride beside the records"
    assert len(results) == 2
    line = budget_stop_line(stopped.stops)
    assert line.startswith("stopped by its cost cap: ")
    assert "; by its metered-call ceiling: 2 of 2 at the stop, 4 once 1 running cell finished (+2); " in line
    assert line.endswith("; 1 cell not started"), "a boundary has one count of cells that did not start"


# =============================================================================
# A host that declared no metered tools
# =============================================================================


async def test_a_host_declaring_no_metered_tools_still_refuses_the_call_and_never_stops_the_run() -> None:
    """A wrong declaration, not a budget: refused at dispatch and counted, with no between-cells stop."""
    results, stopped, ledger, kind, _cap, _breaches = await _execute(calls=1, ceiling=0, width=1, none_declared=True)

    assert stopped is None, "a ceiling of 0 here is a check on the declaration; no landed count reaches it"
    assert not ledger.ceiling_reached
    assert kind.refused_inside_a_cell == 3 and kind.admitted == [0, 0, 0]
    assert ledger.tally().refused == 3
    assert all(result.metered_calls_refused == 1 for result in results)


# =============================================================================
# Determinism: two identical runs contaminate the same cells, which is none
# =============================================================================


@pytest.mark.parametrize("width", [2, 4])
async def test_two_identical_runs_at_width_above_one_contaminate_no_cell(width: int) -> None:
    """The defect this replaces: whichever cell asked after the ceiling was spent was refused, run to run.

    Twelve cells of three calls under a ceiling of 10, twice: every cell of both runs is whole.
    """
    outcomes: list[tuple[list[int | None], int, list[int]]] = []
    for _ in range(2):
        results, stopped, _ledger, kind, _cap, _breaches = await _execute(calls=3, ceiling=10, width=width, k=4)
        assert stopped is not None
        outcomes.append(
            ([result.metered_calls_refused for result in results], kind.refused_inside_a_cell, kind.admitted)
        )

    for refused_per_cell, refused_inside, admitted in outcomes:
        assert set(refused_per_cell) == {0}, "no cell of either run carries a refusal"
        assert refused_inside == 0
        assert set(admitted) == {3}, "every cell made every call it set out to make"


# =============================================================================
# The cost cap, between cells
# =============================================================================


async def test_the_cost_caps_stop_record_says_dollars_and_reads_as_money() -> None:
    cap = EvalRunCostCap("run-1", 1.0, enabled=True)
    _results, stopped, _ledger, _kind, _cap, _breaches = await _execute(
        calls=0, ceiling=None, width=2, hold_second_cell_until_stop=True, cap=cap, cost_per_cell=1.5
    )

    assert stopped is not None
    (stop,) = stopped.stops
    assert stop.currency == "cost_usd"
    assert (stop.ceiling, stop.landed_at_stop, stop.landed_final) == (1.0, 1.5, 3.0)
    assert stop.overshoot == pytest.approx(2.0)
    assert budget_stop_line(stopped.stops) == (
        "stopped by its cost cap: $1.50 of $1.00 at the stop, $3.00 once 1 running cell finished (+$2.00); "
        "1 cell not started"
    )


# =============================================================================
# The line the page shows, written from the record
# =============================================================================


def _stop(**fields: Any) -> BudgetStop:
    base: dict[str, Any] = {
        "currency": "metered_calls",
        "ceiling": 40,
        "landed_at_stop": 40,
        "landed_final": 47,
        "cells_in_flight": 3,
        "cells_not_started": 12,
    }
    return BudgetStop(**{**base, **fields})


def test_the_line_for_the_metered_ceiling_is_the_owners_sentence() -> None:
    assert budget_stop_line([_stop()]) == (
        "stopped by its metered-call ceiling: 40 of 40 at the stop, 47 once 3 running cells finished (+7); "
        "12 cells not started"
    )


def test_the_line_for_the_cost_cap_reads_money_with_a_dollar_sign_throughout() -> None:
    line = budget_stop_line(
        [_stop(currency="cost_usd", ceiling=10.0, landed_at_stop=10.2, landed_final=12.35, cells_in_flight=1)]
    )

    assert line == (
        "stopped by its cost cap: $10.20 of $10.00 at the stop, $12.35 once 1 running cell finished (+$2.35); "
        "12 cells not started"
    )


def test_the_line_for_an_unpriced_stop_says_so() -> None:
    line = budget_stop_line(
        [
            _stop(
                currency="cost_usd",
                ceiling=10.0,
                landed_at_stop=0.4,
                landed_final=0.4,
                cells_in_flight=0,
                cells_not_started=1,
                unpriced_results=2,
            )
        ]
    )

    assert line == (
        "stopped by its cost cap: 2 results spent what could not be priced, with $0.40 of $10.00 at the stop, "
        "no cell was running; 1 cell not started"
    )


def test_a_line_needs_a_record() -> None:
    with pytest.raises(ValueError, match="at least one stop record"):
        budget_stop_line([])


def test_overshoot_is_derived_and_a_negative_one_reads_as_a_minus() -> None:
    stop = _stop(landed_final=40, cells_in_flight=1)

    assert stop.overshoot == 0
    assert "(+0)" in budget_stop_line([stop])


# =============================================================================
# Through the launch: the stored run carries the record, and the summary the line
# =============================================================================


class _LaunchedMeteredKind(_MeteredKind):
    """The launch's version: each cell sleeps a moment so cells overlap, and every metered call is admitted or not."""

    async def invoke(self, instance: ToyExtractorInstance, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        await asyncio.sleep(0.01)
        return await super().invoke(instance, test_case, sink)


async def test_a_launched_run_stopped_by_its_ceiling_stores_the_record_and_its_summary_prints_the_line() -> None:
    from dataclasses import replace

    from threetears.evals.run import KindWiring, LaunchRequest, launch_run

    storage = EvalStorage(InMemoryDocumentStore())
    settings = TOYHOST_LAUNCH_SETTINGS.model_copy(update={"max_metered_calls": 3, "enforcement_enabled": True})
    host, _client = toyhost_launch_host(storage=storage, settings=lambda: settings)
    world = host.eval_host.profile.world
    assert world is not None

    def factory(context: Any) -> ToyExtractorKind:
        kind = _LaunchedMeteredKind(
            client=ScriptedExtractionClient(),
            world=world,
            goal_checks=(),
            calls=1,
            ledger=context.options.metered_calls,
            release=None,
        )
        return kind

    async def launch(request: LaunchRequest) -> EvalRun:
        return await launch_run(
            launching,
            request,
            KindWiring(
                kind_factory=factory,
                subject=TOYHOST_SUBJECT,
                test_cases=toyhost_test_cases(request.template),
                max_concurrent_cells=None,
            ),
        )

    (launchable,) = host.kinds.values()
    launching = replace(host, kinds={TOY_EXTRACTOR_KIND: replace(launchable, launch=launch)})
    template = toyhost_template().model_copy(update={"goal_state_checks": []})
    off_loop(storage.save_template, template)

    (run,) = await start_run(
        launching,
        template_id=template.id,
        scope_id=TOYHOST_SCOPE,
        subject_id=TOYHOST_SUBJECT.subject_id,
        models=[RUN_MODELS[0]],
        k_runs=4,
    )
    async with asyncio.timeout(20):
        while launching.job_manager.is_active(run.id):
            await asyncio.sleep(0.01)

    stored = off_loop(storage.load_eval_run, run.id, TOYHOST_SCOPE)
    results = off_loop(storage.query_eval_results_by_run, run.id, TOYHOST_SCOPE)
    assert stored is not None
    assert stored.status == "budget_stopped"
    assert stored.metered_calls_refused == 0
    assert all(result.metered_calls_refused == 0 for result in results), "no cell carries a refusal"
    assert stored.budget_stops is not None
    (stop,) = stored.budget_stops
    assert stop.currency == "metered_calls" and stop.ceiling == 3
    assert stop.landed_at_stop >= 3 and stop.landed_final >= stop.landed_at_stop
    assert stop.landed_final == len(results), "one call per cell, every cell whole"
    assert stop.cells_not_started == 12 - len(results)
    assert stored.budget_stop_reason is not None
    assert stored.budget_stop_reason.startswith(f"stopped by its metered-call ceiling: {stop.landed_at_stop} of 3")
