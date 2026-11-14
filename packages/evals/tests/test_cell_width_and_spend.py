"""How many cells run at once is the kind's to declare, and the cost cap is read between cells, never inside one.

Driven through the real launch and the real runner over the toy host, reading what was stored:

* **The width.** A run not measuring latency executes at the narrowest of what its kind declares it is
  safe at, the host's ceiling and its matrix. A launcher that says nothing runs its kind serially, and
  the runner refuses — before anything is spent — a kind declaring less than the run executes at.
* **Spend by source, never overwritten.** A cell's sink keeps one running total per source, each only
  ever added to.
* **No refusal of a candidate's call inside a cell.** A priced call and an unpriced one are made whatever the
  run has spent: the cap is read between cells, against what the recorded cells landed, so a candidate is measured
  whole or never begun.
* **Judging is admitted against the cap.** A cell's judge phase is paid for only when its price ceiling fits what
  the run has spent; one that does not fit is set aside unasked, left out of the judged readings and never scored,
  and stops the run, so judging stays within the cap wherever its ceiling bounds the bill. Spend landing exactly on
  the cap is within it, and leaves no room for another cell to start.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, get_args

import pytest

from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.judge import TOY_JUDGE_MODEL, ToyJudgeCompletion, toyhost_judged_template
from packages.evals.tests.fixtures.toyhost.kind import (
    TOY_EXTRACTOR_KIND,
    TOY_SCRIPTS,
    ExtractionResult,
    ScriptedExtractionClient,
    ToyExtractorInstance,
    ToyExtractorKind,
)
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS, toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.product import ExtractionRequest
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_run, toyhost_template, toyhost_test_cases
from packages.evals.tests.scripted_table import FakeCellSink, ScriptedResponse, ScriptedTable, actor
from threetears.evals.analysis.summary import (
    NOT_JUDGED_LINE,
    CaseOutcome,
    CaseResult,
    budget_stop_line,
    summarize_run,
)
from threetears.evals.kernel import CappedCompletion, call_ceiling
from threetears.evals.kernel.candidate_kind import CandidateOutput, CandidateTelemetry, CellSink
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.result_condition import (
    ResultOutcome,
    classify_result,
    resolve_result_condition,
    trial_exclusion,
)
from threetears.evals.kernel.usage_capture import RoleUsageLedger
from threetears.evals.run import (
    KindWiring,
    LaunchHost,
    RunnerOptions,
    drive_conversation,
    execute_run,
    rejudge_result,
    start_run,
)
from threetears.evals.run.budget import (
    AccountExhaustedError,
    BudgetStoppedError,
    EvalRunCostCap,
    ModelUnservedError,
)
from threetears.evals.run.judge import JUDGE_REQUEST_SETTINGS
from threetears.evals.run.launch import build_judge_service
from threetears.evals.run.simulator import CandidateTurn, SimulatorTurn, TurnDriver
from threetears.evals.schema import CompletionProtocolDefect, EvalResult, EvalRun
from threetears.evals.schema.models import (
    DEFAULT_JUDGE_TEMPERATURE,
    ConversationSpec,
    ConversationStopCause,
    EvalTemplate,
    EvalTestCase,
    JudgedArtifact,
    RubricDim,
)
from threetears.evals.kernel import EvalStorage
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.off_loop import off_loop

#: Repeats per case in the launch-driven width tests: the toy template has three cases, so twelve cells.
_K = 4


# =============================================================================
# The width: the narrowest of the kind's declaration, the host's ceiling and the matrix
# =============================================================================


class _SleepingClient(ScriptedExtractionClient):
    """The toy extractor's client, every call sleeping a little and counting how many are in flight."""

    def __init__(self) -> None:
        super().__init__(tuple(replace(script, latency_s=0.02) for script in TOY_SCRIPTS))
        self.running = 0
        self.peak = 0

    async def extract(self, request: ExtractionRequest) -> ExtractionResult:
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            return await super().extract(request)
        finally:
            self.running -= 1


def _launching(*, host_width: int, kind_width: int | None) -> tuple[LaunchHost, EvalStorage, _SleepingClient]:
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_template(toyhost_template())
    client = _SleepingClient()
    settings = TOYHOST_LAUNCH_SETTINGS.model_copy(update={"max_concurrent_cells": host_width})
    host, _ = toyhost_launch_host(
        storage=storage, client=client, settings=lambda: settings, kind_max_concurrent_cells=kind_width
    )
    return host, storage, client


async def _launch(host: LaunchHost) -> EvalRun:
    (run,) = await start_run(
        host,
        template_id=toyhost_template().id,
        scope_id=TOYHOST_SCOPE,
        subject_id=TOYHOST_SUBJECT.subject_id,
        models=[RUN_MODELS[0]],
        k_runs=_K,
    )
    async with asyncio.timeout(30):
        while host.job_manager.is_active(run.id):
            await asyncio.sleep(0.005)
    return run


@pytest.mark.parametrize(
    ("host_width", "kind_width", "executed"),
    [
        pytest.param(4, 2, 2, id="the-kind-narrower-than-the-host"),
        pytest.param(3, 8, 3, id="the-host-a-ceiling-on-a-wider-kind"),
        pytest.param(2, None, 2, id="a-kind-with-no-limit-of-its-own"),
        pytest.param(4, 1, 1, id="a-serial-kind"),
    ],
)
async def test_a_run_executes_at_the_narrower_of_the_hosts_ceiling_and_its_kinds_declaration(
    host_width: int, kind_width: int | None, executed: int
) -> None:
    host, storage, client = off_loop(_launching, host_width=host_width, kind_width=kind_width)

    run = await _launch(host)

    stored = off_loop(storage.load_eval_run, run.id, TOYHOST_SCOPE)
    assert stored is not None and stored.status == "completed"
    assert stored.cell_concurrency == executed, "the run records the width it executed at"
    assert client.peak == executed
    modes = {
        result.covariates.get("execution_mode")
        for result in off_loop(storage.query_eval_results_by_run, run.id, TOYHOST_SCOPE)
    }
    assert modes == ({"serial"} if executed == 1 else {"concurrent"})


def test_a_launcher_that_states_no_width_declares_its_kind_serial() -> None:
    wiring = KindWiring(kind_factory=lambda _cell: None, subject=TOYHOST_SUBJECT, test_cases=[])  # type: ignore[arg-type,return-value]

    assert wiring.max_concurrent_cells == 1


@pytest.mark.parametrize("width", [0, -1, True, 1.5])
def test_a_launcher_stating_a_width_no_run_can_execute_at_is_refused(width: Any) -> None:
    with pytest.raises(ValueError, match="an int of at least 1, or None"):
        KindWiring(
            kind_factory=lambda _cell: None,  # type: ignore[arg-type,return-value]
            subject=TOYHOST_SUBJECT,
            test_cases=[],
            max_concurrent_cells=width,
        )


def _toy_run(
    host: EvalHost, kind: ToyExtractorKind, *, template: EvalTemplate, k: int = 1
) -> tuple[EvalRun, list[EvalTestCase]]:
    world = host.profile.world
    assert world is not None
    cases = toyhost_test_cases(template)
    host.storage.save_template(template)
    for case in cases:
        host.storage.save_test_case(case)
    run = toyhost_run(model=RUN_MODELS[0], template=template, kind=kind, world=world).model_copy(
        update={"k_runs": k, "test_case_ids": [case.id for case in cases]}
    )
    host.storage.save_eval_run(run)
    return run, cases


class _NoDeclaration(ToyExtractorKind):
    """A kind written before widths were declared: it carries no ``max_concurrent_cells`` at all."""

    def __getattribute__(self, name: str) -> Any:
        if name == "max_concurrent_cells":
            raise AttributeError(name)
        return super().__getattribute__(name)


@pytest.mark.parametrize(
    "declared",
    [pytest.param(1, id="declares-one"), pytest.param(None, id="declares-nothing")],
)
async def test_the_runner_refuses_a_kind_declaring_less_than_the_run_executes_at_before_anything_is_spent(
    declared: int | None,
) -> None:
    host = toyhost_host()
    world = host.profile.world
    assert world is not None
    template = toyhost_template()
    client = ScriptedExtractionClient()
    if declared is None:
        kind: ToyExtractorKind = _NoDeclaration(
            client=client, world=world, goal_checks=tuple(template.goal_state_checks)
        )
    else:
        kind = ToyExtractorKind(
            client=client, world=world, goal_checks=tuple(template.goal_state_checks), max_concurrent_cells=declared
        )
    run, cases = off_loop(_toy_run, host, kind, template=template)

    with pytest.raises(ValueError, match="declares max_concurrent_cells=1, and this run executes 2 cells at once"):
        await execute_run(
            host,
            run=run,
            template=template,
            test_cases=cases,
            judge_service=None,
            options=RunnerOptions(candidate_kinds={TOY_EXTRACTOR_KIND: lambda _cell: kind}, max_concurrent_cells=2),
        )

    assert client.calls == [], "no cell's candidate was called"


async def test_a_declaration_no_run_can_execute_at_is_refused_naming_the_kind() -> None:
    host = toyhost_host()
    world = host.profile.world
    assert world is not None
    template = toyhost_template()
    kind = ToyExtractorKind(
        client=ScriptedExtractionClient(),
        world=world,
        goal_checks=tuple(template.goal_state_checks),
        max_concurrent_cells=0,
    )
    run, cases = off_loop(_toy_run, host, kind, template=template)

    with pytest.raises(TypeError, match="ToyExtractorKind declares max_concurrent_cells=0"):
        await execute_run(
            host,
            run=run,
            template=template,
            test_cases=cases,
            judge_service=None,
            options=RunnerOptions(candidate_kinds={TOY_EXTRACTOR_KIND: lambda _cell: kind}),
        )


# =============================================================================
# A kind that pays for its own calls in the cell, and the run's cap over several of them at once
# =============================================================================


@dataclass
class _PricedClient:
    """A completion client whose every call costs ``cost_usd``, and which can say beforehand it costs at most ``ceiling_usd``.

    ``ceiling_usd=None`` is a client that cannot price a call before making it; ``cost_usd=None`` is a call the
    provider priced nothing for.
    """

    cost_usd: float | None
    ceiling_usd: float | None
    calls: int = 0
    running: int = 0
    peak: int = 0

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        del system, user, response_format
        return self.ceiling_usd

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        del system, user, response_format
        self.calls += 1
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            await asyncio.sleep(0.01)
        finally:
            self.running -= 1
        return ScriptedResponse(content="paid", model=RUN_MODELS[0], cost_usd=self.cost_usd)


class _PaysInCellKind(ToyExtractorKind):
    """A kind whose candidate makes ``calls`` paid calls per cell through the cell's sink, and grades nothing.

    Each call goes through :class:`CappedCompletion`, so its spend lands on the cell's sink. The cap refuses
    none: every call a cell sets out to make is made.
    """

    def __init__(self, *args: Any, paid: _PricedClient, calls: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.paid = paid
        self.calls = calls

    async def invoke(self, instance: ToyExtractorInstance, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        client = CappedCompletion(self.paid, sink=sink, source="candidate")
        ledger = RoleUsageLedger(role="candidate")
        made = 0
        for _ in range(self.calls):
            ledger.add_llm_result(await client.generate(system="s", user="u"))
            made += 1
        return CandidateOutput(output=[{"paid_calls": made}], telemetry=CandidateTelemetry(usage=ledger.rows()))


def _output(host: EvalHost, run: EvalRun, result: EvalResult) -> list[dict[str, Any]]:
    """What the cell's kind returned as its output, from the trace stored beside the result."""
    trace = host.storage.load_eval_trace(result.id, run.scope_id)
    assert trace is not None
    return trace.trace


def _no_goal_checks() -> EvalTemplate:
    """The toy template with no goal checks: these kinds grade no extraction."""
    return toyhost_template().model_copy(update={"goal_state_checks": []})


async def _capped_run(
    kind_for: Any,
    *,
    cap_usd: float,
    width: int,
    k: int = 1,
    n_cases: int | None = None,
    on_breach: Callable[[], None] | None = None,
) -> tuple[EvalHost, EvalRun, list[EvalResult], BudgetStoppedError | None, EvalRunCostCap]:
    """Run ``kind_for(world)`` over the toy cases at ``width``, under an enforcing cap of ``cap_usd``.

    ``on_breach`` is called the moment the cap's gate reports a breach, which is when the run decides to start no
    new cell: a test holds a running cell until then, so what had landed at the stop does not depend on timing.
    """
    host = toyhost_host()
    world = host.profile.world
    assert world is not None
    template = _no_goal_checks()
    kind = kind_for(world)
    run, cases = off_loop(_toy_run, host, kind, template=template, k=k)
    if n_cases is not None:
        cases = cases[:n_cases]
        run = run.model_copy(update={"test_case_ids": [case.id for case in cases]})
    cap = EvalRunCostCap(run.id, cap_usd, enabled=True)
    stopped: BudgetStoppedError | None = None

    def gate(pending: float | None) -> Any:
        breach = cap.check(pending)
        if breach is not None and on_breach is not None:
            on_breach()
        return breach

    try:
        await execute_run(
            host,
            run=run,
            template=template,
            test_cases=cases,
            judge_service=None,
            options=RunnerOptions(candidate_kinds={TOY_EXTRACTOR_KIND: lambda _cell: kind}, max_concurrent_cells=width),
            budget_gate=gate,
            on_cost=cap.record,
        )
    except BudgetStoppedError as error:
        stopped = error
    return host, run, off_loop(host.storage.query_eval_results_by_run, run.id, run.scope_id), stopped, cap


async def test_a_tiny_cap_starts_no_third_cell_and_refuses_the_running_cell_nothing() -> None:
    """Two cells run together under a cap the first one's landing passes: the third never starts, the second finishes.

    The second cell is held until the run has decided to stop, so it is still running at the stop. It is not refused
    its paid call: it finishes whole, and the stop record says what had landed at the stop and once it had finished.
    """
    paid = _PricedClient(cost_usd=1.0, ceiling_usd=1.0)
    release = asyncio.Event()

    class _HoldsTheSecondCell(_PaysInCellKind):
        invoked = 0

        async def invoke(
            self, instance: ToyExtractorInstance, test_case: EvalTestCase, sink: CellSink
        ) -> CandidateOutput:
            type(self).invoked += 1
            if type(self).invoked == 2:
                await release.wait()
            return await super().invoke(instance, test_case, sink)

    def kind_for(world: Any) -> ToyExtractorKind:
        return _HoldsTheSecondCell(client=ScriptedExtractionClient(), world=world, goal_checks=(), paid=paid, calls=1)

    host, run, results, stopped, cap = await _capped_run(
        kind_for, cap_usd=0.5, width=2, n_cases=3, on_breach=release.set
    )

    assert paid.calls == 2, "two cells paid, whole; the third never started"
    assert len(results) == 2
    assert all(off_loop(_output, host, run, result) == [{"paid_calls": 1}] for result in results)
    assert all(classify_result(result) is ResultOutcome.OK for result in results), "no cell was cut short"
    assert all(not result.runner_error for result in results)
    assert stopped is not None, "the run stops budget_stopped"
    (stop,) = stopped.stops
    assert stop.currency == "cost_usd" and stop.ceiling == pytest.approx(0.5)
    assert stop.landed_at_stop == pytest.approx(1.0), "only the first cell had landed when the stop was decided"
    assert stop.cells_in_flight == 1
    assert stop.landed_final == pytest.approx(2.0) == pytest.approx(cap.accumulated_usd)
    assert stop.overshoot == pytest.approx(1.5)
    assert stop.cells_not_started == 1


async def test_a_concurrent_run_lands_past_its_cap_by_at_most_the_widths_worth_of_cells() -> None:
    """Twelve cells, four at a time, three priced calls each: no cell is cut short, and the overshoot is bounded.

    The cap is read between cells against what has landed, so the cell that crosses it and the cells running
    beside it finish whole; nothing starts after.
    """
    paid = _PricedClient(cost_usd=0.008, ceiling_usd=0.01)

    def kind_for(world: Any) -> ToyExtractorKind:
        return _PaysInCellKind(client=ScriptedExtractionClient(), world=world, goal_checks=(), paid=paid, calls=3)

    host, run, results, stopped, cap = await _capped_run(kind_for, cap_usd=0.1, width=4, k=4)

    assert stopped is not None
    assert paid.peak > 1, "the cells' calls overlapped"
    assert len(results) < 12, "no cell started once the cap had been reached by what landed"
    assert all(off_loop(_output, host, run, result) == [{"paid_calls": 3}] for result in results), (
        "every cell made all of its calls: none was refused inside a cell"
    )
    assert all(classify_result(result) is ResultOutcome.OK for result in results)
    assert sum(result.cost_usd or 0.0 for result in results) == pytest.approx(cap.accumulated_usd)
    assert cap.accumulated_usd > 0.1
    (stop,) = stopped.stops
    assert stop.landed_final == pytest.approx(cap.accumulated_usd)
    assert stop.cells_not_started == 12 - len(results)
    assert stop.cells_in_flight <= 3, "the cell that decided is the fourth running"
    assert stop.overshoot <= 0.024 * 4, "by no more than the cells in flight and the one that crossed it"


async def test_an_unpriced_call_is_made_like_any_other_so_every_cell_makes_all_of_its_calls() -> None:
    """A call nobody can price is never refused either: two cells running together make all six calls."""
    paid = _PricedClient(cost_usd=1.0, ceiling_usd=None)

    def kind_for(world: Any) -> ToyExtractorKind:
        return _PaysInCellKind(client=ScriptedExtractionClient(), world=world, goal_checks=(), paid=paid, calls=3)

    _host, _run, results, stopped, cap = await _capped_run(kind_for, cap_usd=1.5, width=2, n_cases=2)

    assert paid.calls == 6
    assert cap.accumulated_usd == pytest.approx(6.0)
    assert len(results) == 2
    assert stopped is None, (
        "both cells had started before anything landed: there was no cell left to keep from starting"
    )


# =============================================================================
# Spend by source: each total only grows, and the cap reads their sum and every sibling's
# =============================================================================


class _ReportsBySourceKind(ToyExtractorKind):
    """Two cells of one run, side by side: the first reports spend on two sources, the second adds its own past the cap.

    The first to invoke reports $0.6 of candidate spend and $0.3 of background work and waits; the second, once it
    has, adds a $0.2 call of its own, taking the cells past the run's $1 cap while both still run. Neither is
    refused: the cap is read between cells.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reported = asyncio.Event()
        self.added = asyncio.Event()
        self.order: list[str] = []
        self.negative_refused: list[str] = []
        self.finished: list[str] = []

    async def invoke(self, instance: ToyExtractorInstance, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        self.order.append(test_case.id)
        if len(self.order) == 1:
            sink.add_spend("candidate", 0.6)
            sink.add_spend("background_work", 0.3)
            try:
                sink.add_spend("candidate", -0.6)
            except ValueError as refused:
                self.negative_refused.append(str(refused))
            self.reported.set()
            await self.added.wait()
            self.finished.append(test_case.id)
            return CandidateOutput(output=[{"first": True}])
        await self.reported.wait()
        sink.add_spend("candidate", 0.2)
        self.added.set()
        self.finished.append(test_case.id)
        return CandidateOutput(output=[{"first": False}])


async def test_a_cells_spend_by_source_only_grows_and_no_sibling_is_refused_by_it() -> None:
    kinds: list[_ReportsBySourceKind] = []

    def kind_for(world: Any) -> ToyExtractorKind:
        kinds.append(_ReportsBySourceKind(client=ScriptedExtractionClient(), world=world, goal_checks=()))
        return kinds[0]

    _host, _run, results, stopped, _cap = await _capped_run(kind_for, cap_usd=1.0, width=2, n_cases=2)

    (kind,) = kinds
    assert kind.negative_refused and "0 or more" in kind.negative_refused[0], "a source's total never decreases"
    assert sorted(kind.finished) == sorted(kind.order) and len(kind.order) == 2, (
        "$0.9 beside a $0.2 call is asked of no cell: the cap is read between cells"
    )
    assert len(results) == 2
    assert stopped is None


# =============================================================================
# A conversing kind is not cut short by the run's spend
# =============================================================================


class _ConversingPaidKind(ToyExtractorKind):
    """A conversing kind whose candidate pays $0.01 for each answer, counted on the cell's sink."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.answered: list[str] = []

    async def invoke(self, instance: ToyExtractorInstance, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        driver = TurnDriver(conversation=ConversationSpec(actors=[actor("a")], max_turns=5), variation={})
        table = ScriptedTable(lines={"a": [f"a{n}" for n in range(1, 6)]}, ceiling_usd=0.001)
        answered: list[str] = []
        candidate = RoleUsageLedger(role="candidate")

        async def answer(round_turns: Sequence[SimulatorTurn]) -> CandidateTurn:
            response = ScriptedResponse(content="answer", model=RUN_MODELS[0], cost_usd=0.01)
            candidate.add_llm_result(response)
            sink.add_spend("candidate", response.cost_usd)
            answered.append(round_turns[-1].content)
            return CandidateTurn(content=f"answer {len(answered)}")

        async def post(_turn: SimulatorTurn) -> None:
            return None

        stop = await drive_conversation(driver, answer, post, llm=table, sink=sink)
        self.answered = answered
        simulator = RoleUsageLedger(role="simulator")
        driver.fold_usage(simulator)
        return CandidateOutput(
            output=[{"answered": answered}],
            telemetry=CandidateTelemetry(usage=[*simulator.rows(), *candidate.rows()]),
            stop_cause=stop,
        )


async def test_a_conversation_the_run_has_outspent_is_not_cut_short_mid_cell() -> None:
    kinds: list[_ConversingPaidKind] = []

    def kind_for(world: Any) -> ToyExtractorKind:
        kinds.append(_ConversingPaidKind(client=ScriptedExtractionClient(), world=world, goal_checks=()))
        return kinds[0]

    # A cap the cell passes in its second round: the cell still answers all five.
    host, run, results, stopped, cap = await _capped_run(kind_for, cap_usd=0.025, width=1, n_cases=1)

    (result,) = results
    assert result.stop_cause is ConversationStopCause.MAX_TURNS, "the conversation ran to its own end"
    assert result.candidate_error is None and result.runner_error is None
    assert classify_result(result) is ResultOutcome.OK
    assert off_loop(_output, host, run, result) == [{"answered": ["a1", "a2", "a3", "a4", "a5"]}]
    assert cap.accumulated_usd > 0.025, "the cell landed past the cap, whole"
    assert stopped is None, "no cell was left to keep from starting"


# =============================================================================
# The judge phase is admitted against the cap before any of its calls is paid for
# =============================================================================


@dataclass
class _PricedJudge:
    """A judge client that scores every dim a 4, each call costing ``cost_usd`` and priced beforehand at ``ceiling_usd``."""

    cost_usd: float
    ceiling_usd: float | None
    calls: list[str] = field(default_factory=list)
    #: Whether the first call never answers, so the phase asking it holds its room until the cell's deadline.
    first_hangs: bool = False
    #: A dim whose first call never answers, so the phase asking it holds its room, its other calls billed.
    hangs_once_on: str | None = None
    #: The ceiling each phase prices its calls at, in the order the phases price them, in place of ``ceiling_usd``:
    #: so two cells of one run can be priced apart. The last entry answers every phase after it.
    ceilings_in_turn: Sequence[float | None] = ()
    priced: int = 0
    #: How long each call takes, and the most calls that were in flight at once.
    call_s: float = 0.0
    in_flight: int = 0
    peak: int = 0
    #: Whether every call on the first dim asked answers with no JSON, so the judge fails that dim, each call billed.
    first_fails: bool = False

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        del system, user, response_format
        if not self.ceilings_in_turn:
            return self.ceiling_usd
        self.priced += 1
        return self.ceilings_in_turn[min(self.priced, len(self.ceilings_in_turn)) - 1]

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        del user, response_format
        match = re.search(r'the single key "(.+?)"', system)
        assert match is not None
        dim = match.group(1)
        self.calls.append(dim)
        if (self.first_hangs and len(self.calls) == 1) or (dim == self.hangs_once_on and self.calls.count(dim) == 1):
            await asyncio.Event().wait()
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self.call_s)
        finally:
            self.in_flight -= 1
        fails = self.first_fails and dim == self.calls[0]
        return ToyJudgeCompletion(
            content="no verdict" if fails else json.dumps({"reasoning": "fine", "criteria_scores": {dim: 4}}),
            input_tokens=100,
            output_tokens=40,
            cost_usd=self.cost_usd,
            model=TOY_JUDGE_MODEL,
            price_source="toyhost_script",
        )

    async def aclose(self) -> None:
        """Nothing to release."""


async def _judged_run(judge: _PricedJudge, *, cap_usd: float) -> tuple[list[EvalResult], BudgetStoppedError | None]:
    _host, _run, results, stopped = await _judged_run_kept(judge, cap_usd=cap_usd)
    return results, stopped


class _QuarterDollarClient(ScriptedExtractionClient):
    """The toy extractor's client, every extraction billed $0.25 exactly, so sums against a cap are exact; the first
    extraction takes ``first_delay_s`` longer, so the cell started first reaches its judging last."""

    def __init__(self, *, first_delay_s: float = 0.0) -> None:
        super().__init__()
        self.first_delay_s = first_delay_s
        self.extracted = 0

    async def extract(self, request: ExtractionRequest) -> ExtractionResult:
        self.extracted += 1
        if self.extracted == 1:
            await asyncio.sleep(self.first_delay_s)
        return replace(await super().extract(request), cost_usd=0.25)


async def _judged_run_kept(
    judge: _PricedJudge,
    *,
    cap_usd: float,
    k: int = 1,
    width: int = 1,
    client: ScriptedExtractionClient | None = None,
    cell_timeout_s: float | None = None,
    template: EvalTemplate | None = None,
) -> tuple[EvalHost, EvalRun, list[EvalResult], BudgetStoppedError | None]:
    """``k`` judged toy cells of one case under ``cap_usd``, ``width`` at once: the host, the run, its stored results
    and the stop, if one."""
    host, run, results, stopped = await _judged_run_ended(
        judge,
        cap_usd=cap_usd,
        k=k,
        width=width,
        client=client,
        cell_timeout_s=cell_timeout_s,
        template=template,
    )
    assert stopped is None or isinstance(stopped, BudgetStoppedError), stopped
    return host, run, results, stopped


async def _judged_run_ended(
    judge: _PricedJudge,
    *,
    cap_usd: float,
    k: int = 1,
    width: int = 1,
    client: ScriptedExtractionClient | None = None,
    cell_timeout_s: float | None = None,
    template: EvalTemplate | None = None,
    kind_type: type[ToyExtractorKind] = ToyExtractorKind,
) -> tuple[EvalHost, EvalRun, list[EvalResult], BudgetStoppedError | AccountExhaustedError | ModelUnservedError | None]:
    """As :func:`_judged_run_kept`, its cells run by a kind of ``kind_type``, and ended by whichever stop ended it."""
    host = toyhost_host(clients=lambda _role, _model, *, temperature=None: judge)
    world = host.profile.world
    assert world is not None
    template = template or toyhost_judged_template()
    kind = kind_type(
        client=client or ScriptedExtractionClient(),
        world=world,
        judged=True,
        goal_checks=tuple(template.goal_state_checks),
    )
    run, cases = off_loop(_toy_run, host, kind, template=template, k=k)
    cases = cases[:1]
    run_judge = off_loop(build_judge_service, host, template, TOY_JUDGE_MODEL, judged_artifact=JudgedArtifact.DOCUMENT)
    run = run.model_copy(
        update={
            "test_case_ids": [cases[0].id],
            "judge_model": TOY_JUDGE_MODEL,
            "effective_judges": run_judge.effective_judges,
            "judge_config_ids": {dim: config.id for dim, config in run_judge.configs.items()},
            "judge_asked_settings": JUDGE_REQUEST_SETTINGS,
            "judge_temperature": DEFAULT_JUDGE_TEMPERATURE,
        }
    )
    off_loop(host.storage.save_eval_run, run)
    cap = EvalRunCostCap(run.id, cap_usd, enabled=True)
    stopped: BudgetStoppedError | AccountExhaustedError | ModelUnservedError | None = None
    try:
        await execute_run(
            host,
            run=run,
            template=template,
            test_cases=cases,
            judge_service=run_judge.service,
            options=RunnerOptions(
                candidate_kinds={TOY_EXTRACTOR_KIND: lambda _cell: kind},
                max_concurrent_cells=width,
                **({"cell_timeout_s": cell_timeout_s} if cell_timeout_s is not None else {}),
            ),
            budget_gate=cap.check,
            on_cost=cap.record,
        )
    except (BudgetStoppedError, AccountExhaustedError, ModelUnservedError) as error:
        stopped = error
    return host, run, off_loop(host.storage.query_eval_results_by_run, run.id, run.scope_id), stopped


async def test_judging_the_cap_has_no_room_for_is_set_aside_unasked_and_stops_the_run() -> None:
    """The judge prices its one call at $1 an attempt, two attempts, under a $0.50 cap: nothing is asked, the cell is
    excluded rather than scored, and the run stops on its cost cap saying the cell was not judged."""
    judge = _PricedJudge(cost_usd=0.0004, ceiling_usd=1.0)

    (result,), stopped = await _judged_run(judge, cap_usd=0.5)

    assert judge.calls == [], "no judge call was paid for"
    assert result.rubric_scores == []
    assert result.judge_error is not None and "the run's cost cap had too little left" in result.judge_error
    # Set aside: its candidate's measures count, and only its judged readings leave it out, under the cap's own cause.
    assert classify_result(result) is ResultOutcome.OK, "never a fail, and no fault of the rig"
    assert result.judging_set_aside
    assert trial_exclusion(result) == "cost_cap", "its own cause, never a setup fault"
    assert result.cost_usd is not None and result.cost_usd < 0.5, "the candidate's spend alone landed"
    assert stopped is not None
    (stop,) = stopped.stops
    assert (stop.currency, stop.cells_not_judged, stop.cells_not_started) == ("cost_usd", 1, 0)
    assert stop.landed_final == pytest.approx(result.cost_usd)
    assert budget_stop_line(stopped.stops).endswith(
        "; 0 cells not started; 1 cell not judged, as the cap had too little left for their judging"
    )


async def test_the_run_summary_names_a_case_the_cap_set_aside_as_not_judged_with_its_remedy() -> None:
    """Even with no stop line to read, the summary says which case the cap left unjudged and what judges it: the case
    reads not judged, never scored, with the line naming the cap; the results line counts it apart; and its judging
    state is set aside, not a judge that errored."""
    judge = _PricedJudge(cost_usd=0.0004, ceiling_usd=1.0)
    host, run, (result,), _stopped = await _judged_run_kept(judge, cap_usd=0.5)

    summary = off_loop(summarize_run, host, run.id, run.scope_id)
    case = CaseResult.of(result, case="the case", given=None, expected=None, answer=None)

    assert (summary.n_scored, summary.n_not_judged) == (0, 1)
    assert "1 not judged, as the cost cap had too little left for their judging" in summary.results_text
    assert any(error.endswith(NOT_JUDGED_LINE) for error in summary.errors)
    assert case.outcome == "not_judged" and case.errors == [NOT_JUDGED_LINE]
    assert case.render().splitlines()[0].startswith("case the case (repeat 1, not judged): "), "in words"
    assert "re-judge it" in NOT_JUDGED_LINE
    assert resolve_result_condition(result).judging == "set_aside"


@pytest.mark.parametrize("outcome", get_args(CaseOutcome))
def test_a_case_line_prints_every_case_outcome_in_words(outcome: CaseOutcome) -> None:
    """Every outcome a case can have has its words on the case line, so an outcome added without them fails here
    rather than when a run's case is printed."""
    case = CaseResult(case="c", case_id="c", input=None, repeat=1, outcome=outcome, state="passed")

    assert case.render().startswith(f"case c (repeat 1, {outcome.replace('_', ' ')}): ")


async def test_a_re_judge_takes_up_a_cell_the_cap_set_aside_and_clears_its_mark() -> None:
    """The cell the cap set aside stored its evidence: a re-judge asks every dim it did not ask, scores the cell, and
    clears the mark, so it counts in every reading."""
    judge = _PricedJudge(cost_usd=0.0004, ceiling_usd=1.0)
    host, run, (result,), stopped = await _judged_run_kept(judge, cap_usd=0.5)
    assert stopped is not None and result.judging_set_aside and judge.calls == []
    off_loop(host.storage.save_eval_run, run.model_copy(update={"status": "budget_stopped"}))

    rejudged = await rejudge_result(host, result.id, run.scope_id)

    assert judge.calls, "the re-judge asked what the cap had set aside"
    assert rejudged.judge_error is None and rejudged.rubric_scores
    assert not rejudged.judging_set_aside
    assert trial_exclusion(rejudged) is None and classify_result(rejudged) is ResultOutcome.OK


async def test_a_cell_starts_only_while_the_cap_has_room_left_and_spend_exactly_on_it_leaves_none() -> None:
    """One cell at a time: the first lands the run exactly on its $0.75 cap ($0.25 of candidate, $0.50 of judging at
    its ceiling), so the second never starts. Started, it would pay for its candidate and have its judging set aside."""
    judge = _PricedJudge(cost_usd=0.5, ceiling_usd=0.25)

    _host, _run, results, stopped = await _judged_run_kept(judge, cap_usd=0.75, k=2, client=_QuarterDollarClient())

    assert len(results) == 1 and results[0].judge_error is None
    assert stopped is not None
    (stop,) = stopped.stops
    assert (stop.cells_not_started, stop.cells_not_judged) == (1, 0)
    assert stop.landed_at_stop == stop.landed_final == 0.75


async def test_two_cells_side_by_side_leave_the_third_unstarted_once_their_judging_fills_the_cap() -> None:
    """Two cells at once under a $1.50 cap, each $0.25 of candidate and $0.50 of judging at its ceiling. The second's
    judging waits while the first's holds its room, and both fit to the cent. When a slot frees, what has landed is
    under the cap, but with the running cell's spend and the judging it holds there is nothing left, so the third cell
    never starts, and the run ends on its cap with every cell it ran judged."""
    judge = _PricedJudge(cost_usd=0.5, ceiling_usd=0.25)

    _host, _run, results, stopped = await _judged_run_kept(
        judge, cap_usd=1.5, k=3, width=2, client=_QuarterDollarClient()
    )

    assert len(results) == 2 and all(result.judge_error is None for result in results)
    assert stopped is not None
    (stop,) = stopped.stops
    assert (stop.cells_not_started, stop.cells_not_judged) == (1, 0)
    assert stop.landed_final == 1.5


#: The second dim of the two-dim template, which the phase's hanging call asks.
_VENDOR_DIM = "extraction.vendor_named"


def _two_dim_template() -> EvalTemplate:
    """The toy judged template with a second dim, so a judge phase makes two calls and can hold its room part billed."""
    template = toyhost_judged_template()
    return template.model_copy(
        update={
            "rubric": [
                *template.rubric,
                RubricDim(
                    name=_VENDOR_DIM,
                    description="the vendor is named as the invoice names it",
                    scale="ordinal",
                    judge="builtin",
                ),
            ]
        }
    )


async def test_a_hold_counts_only_what_its_calls_have_not_billed_so_a_cell_starts_into_the_room_left() -> None:
    """Two cells at once under a $2.625 cap, a phase of two calls each priced at $0.50 (two attempts at $0.25) and
    billed $0.50, so a phase holds $1. The cell started second writes first: its judging is admitted, its first call
    bills and its second never answers, so it holds $0.50 of its $1 while it waits. The first cell's judging fits beside
    that ($0.75 + $0.25 + $0.50 + $1 = $2.50) and is judged whole. When its slot frees, $1.25 has landed and the running
    cell has spent $0.75 and holds $0.50 more: $2.50, under the cap, so the third cell starts. Counting the hold's whole
    $1 beside the call it already billed would read $3, past the cap: the first cell would never be admitted, and the
    third never start. The run also ends within its cap, but that check catches only a hold that counts too little
    to keep the run inside it. A hold that counts nothing also passes it here, since the call it holds for never bills;
    the deadline-during-wait test below catches that one."""
    judge = _PricedJudge(cost_usd=0.5, ceiling_usd=0.25, hangs_once_on=_VENDOR_DIM)

    _host, _run, results, stopped = await _judged_run_kept(
        judge,
        cap_usd=2.625,
        k=3,
        width=2,
        client=_QuarterDollarClient(first_delay_s=0.15),
        cell_timeout_s=0.6,
        template=_two_dim_template(),
    )

    assert len(results) == 3, "the third cell started into the room the part-billed hold left"
    assert sum(1 for result in results if result.judge_error is None and len(result.rubric_scores) == 2) == 1, (
        "the cell admitted beside the part-billed hold was judged whole"
    )
    assert stopped is not None and stopped.stops[0].landed_final <= 2.625, "the run ended within its cap"


async def test_a_deadline_while_judging_waits_for_the_cap_s_room_sets_the_cell_aside_under_the_cap() -> None:
    """Two cells at once under a $1.25 cap. The cell started second writes first: its judging is admitted and never
    answers, so it holds its room. The cell started first writes slowly, and its judging does not fit beside that hold,
    so it waits, and its deadline, the earlier, strikes while it waits. It never asked the judge, because the cap had
    no room, so it is set aside under the cap and its measures count; the other is cut off in its judge call, a fault of
    the rig. The run is not stopped as cost-capped: the phase waited on a hold that might have billed less than its
    ceiling, so the cap's shortage was never established."""
    judge = _PricedJudge(cost_usd=0.5, ceiling_usd=0.25, first_hangs=True)

    _host, _run, results, stopped = await _judged_run_kept(
        judge, cap_usd=1.25, k=2, width=2, client=_QuarterDollarClient(first_delay_s=0.15), cell_timeout_s=0.5
    )

    assert len(judge.calls) == 1, "the waiting cell asked nothing"
    assert sorted(str(trial_exclusion(result)) for result in results) == ["cost_cap", "infra"]
    (waited,) = [result for result in results if result.judging_set_aside]
    assert classify_result(waited) is ResultOutcome.OK, "its measures count; only its judged readings leave it out"
    assert stopped is None, "a wait the deadline cut short establishes no shortage, so the cap stopped nothing"


async def _candidate_spend() -> float:
    """What the one toy cell's candidate spends, read off an uncapped run's stored row."""
    (result,), _ = await _judged_run(_PricedJudge(cost_usd=0.0, ceiling_usd=0.0), cap_usd=1000.0)
    (spend,) = [row.cost_usd for row in result.usage if row.role == "candidate"]
    assert spend is not None and spend > 0
    return spend


async def test_judging_whose_spend_lands_exactly_on_the_cap_is_judged_and_the_run_is_not_stopped() -> None:
    """The judge's one call costs exactly its $0.50 ceiling (two attempts at $0.25), and the cap is the candidate's
    spend plus that: the phase fits to the cent, is judged, and the run ends on its cap, within it."""
    candidate = await _candidate_spend()
    judge = _PricedJudge(cost_usd=0.5, ceiling_usd=0.25)

    (result,), stopped = await _judged_run(judge, cap_usd=candidate + 0.5)

    assert judge.calls, "the phase was admitted"
    assert result.judge_error is None
    assert result.cost_usd == candidate + 0.5, "the run landed exactly on its cap"
    assert stopped is None, "spend on the cap is within it"


async def test_judging_that_would_land_the_run_a_hair_past_its_cap_is_set_aside() -> None:
    """The same judge under a cap the smallest step below the candidate's spend plus the phase's ceiling: nothing is
    asked, and the run stops with its spend under the cap."""
    candidate = await _candidate_spend()
    judge = _PricedJudge(cost_usd=0.5, ceiling_usd=0.25)
    cap = math.nextafter(candidate + 0.5, 0.0)

    (result,), stopped = await _judged_run(judge, cap_usd=cap)

    assert judge.calls == []
    assert result.cost_usd == candidate
    assert stopped is not None and stopped.stops[0].cells_not_judged == 1
    assert stopped.stops[0].landed_final <= cap


async def test_a_judge_that_cannot_price_its_calls_passes_the_cap_by_one_call_at_most() -> None:
    """A judge with no ceiling is asked one call at a time while the run is within its cap. Its one call is asked
    while the candidate's spend alone is under the cap, and lands the run one call past it; nothing more is asked."""
    candidate = await _candidate_spend()
    judge = _PricedJudge(cost_usd=0.5, ceiling_usd=None)

    (result,), stopped = await _judged_run(judge, cap_usd=candidate + 0.1)

    assert len(judge.calls) == 1
    assert result.cost_usd == pytest.approx(candidate + 0.5), "past the cap by the one call"
    assert stopped is None, "the cell was judged; there was no cell left to keep from starting"


@pytest.mark.parametrize(
    "ceilings",
    [
        pytest.param((0.25, None), id="the-unpriced-phase-waits-for-the-priced-one-to-release"),
        pytest.param((None, 0.25), id="the-priced-phase-waits-while-the-unpriced-one-holds-alone"),
    ],
)
async def test_an_unpriced_judge_phase_never_runs_beside_a_priced_one(ceilings: tuple[float | None, ...]) -> None:
    """Two cells at once under a roomy cap, the judge pricing one phase and not the other. The cell started second
    reaches its judging first and is admitted. The other's phase waits until the first releases, whichever of the two
    is the unpriced one: an unpriced phase holds the run's judging alone, so no call of it is ever in flight beside a
    call of another phase. Both cells are judged."""
    judge = _PricedJudge(cost_usd=0.0004, ceiling_usd=None, ceilings_in_turn=ceilings, call_s=0.2)

    _host, _run, results, stopped = await _judged_run_kept(
        judge, cap_usd=10.0, k=2, width=2, client=_QuarterDollarClient(first_delay_s=0.05)
    )

    assert judge.priced == 2, "each phase was priced once, one with a ceiling and one without"
    assert len(judge.calls) == 2 and all(result.judge_error is None for result in results)
    assert judge.peak == 1, "the unpriced phase's call was never in flight beside the priced phase's"
    assert stopped is None


@pytest.mark.parametrize(
    ("first_fails", "exclusion", "not_judged"),
    [
        pytest.param(True, "infra", 0, id="a-judge-fault-first-stays-the-judges"),
        pytest.param(False, "cost_cap", 1, id="a-score-first-is-set-aside"),
    ],
)
async def test_a_dim_the_judge_failed_before_the_cap_ran_out_leaves_the_cell_the_judges_fault(
    first_fails: bool, exclusion: str, not_judged: int
) -> None:
    """A judge with no ceiling, two dims. Its first call costs $0.50, past what the cap had left, so the cap keeps the
    second dim unasked and the run stops on its cap. When the judge had failed the first dim, the cell is the judge's
    fault, excluded as one, which a re-judge recovers, and the cap does not count it as set aside. When the first dim
    scored, the cell is set aside under the cap and counted, as a phase the cap ran out on is."""
    candidate = await _candidate_spend()
    judge = _PricedJudge(cost_usd=0.5, ceiling_usd=None, first_fails=first_fails)

    _host, _run, (result,), stopped = await _judged_run_kept(
        judge, cap_usd=candidate + 0.1, template=_two_dim_template()
    )

    assert len(set(judge.calls)) == 1, "the second dim was never asked"
    assert result.judge_error is not None and "the run's cost cap had too little left" in result.judge_error
    assert result.judging_set_aside is not first_fails
    assert trial_exclusion(result) == exclusion
    assert stopped is not None
    (stop,) = stopped.stops
    assert (stop.currency, stop.cells_not_judged) == ("cost_usd", not_judged)


class _StopsFirstKind(ToyExtractorKind):
    """The toy extractor, whose first ``stopping`` cells invoked meet what stops a run at once, with nothing to judge;
    every later cell extracts a beat later, so it reaches its judging after the stop was decided."""

    stopping = 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.invoked = 0

    def stopped(self) -> CandidateOutput:
        """What a cell that meets the stop returns."""
        return CandidateOutput(account_refused=True, infra_errors=["apparatus: refused for the calling account"])

    async def invoke(self, instance: ToyExtractorInstance, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        self.invoked += 1
        if self.invoked <= self.stopping:
            return self.stopped()
        await asyncio.sleep(0.1)
        return await super().invoke(instance, test_case, sink)


class _UnservedFirstKind(_StopsFirstKind):
    """The first cells, as many as the unserved-model circuit counts, find the run's model not served."""

    stopping = 3

    def stopped(self) -> CandidateOutput:
        return CandidateOutput(model_unserved=True, infra_errors=["apparatus: the model is no longer available"])


@pytest.mark.parametrize(
    ("kind_type", "error_type"),
    [
        pytest.param(_StopsFirstKind, AccountExhaustedError, id="the-accounts-refusal"),
        pytest.param(_UnservedFirstKind, ModelUnservedError, id="the-unserved-model-circuit"),
    ],
)
async def test_a_run_another_stop_ended_keeps_the_caps_count_of_the_cells_it_set_aside(
    kind_type: type[_StopsFirstKind], error_type: type[Exception]
) -> None:
    """Every cell starts at once. The first ones meet the account's refusal, or the circuit's count of an unserved
    model, which stops the run; the last reaches its judging after that, and the $0.60 cap has too little left for it
    ($0.25 of candidate and a $0.50 ceiling), so it is set aside. The run ends on what gave out, and the error carries
    the cap's record counting that cell, which the job manager writes on the run."""
    judge = _PricedJudge(cost_usd=0.5, ceiling_usd=0.25)

    _host, _run, results, stopped = await _judged_run_ended(
        judge,
        cap_usd=0.6,
        k=kind_type.stopping + 1,
        width=kind_type.stopping + 1,
        client=_QuarterDollarClient(),
        kind_type=kind_type,
    )

    assert isinstance(stopped, error_type)
    assert isinstance(stopped, AccountExhaustedError | ModelUnservedError)
    assert judge.calls == []
    assert sum(result.judging_set_aside for result in results) == 1
    (stop,) = stopped.stops
    assert (stop.currency, stop.cells_not_judged) == ("cost_usd", 1)
    assert "the cost cap had too little left to judge 1 cell(s)" in str(stopped)


async def test_a_judge_phase_the_cap_has_room_for_is_judged_and_its_spend_counted() -> None:
    judge = _PricedJudge(cost_usd=0.0004, ceiling_usd=0.001)

    (result,), stopped = await _judged_run(judge, cap_usd=10.0)

    assert judge.calls, "every dim was judged"
    assert result.judge_error is None
    assert stopped is None


# =============================================================================
# The client wrapper and its pricing
# =============================================================================


async def test_a_capped_client_makes_every_priced_call_and_counts_each_at_what_it_cost() -> None:
    paid = _PricedClient(cost_usd=0.004, ceiling_usd=0.01)
    sink = FakeCellSink()
    client = CappedCompletion(paid, sink=sink, source="candidate")

    for _ in range(5):
        await client.generate(system="s", user="u")

    assert paid.calls == 5
    assert sink.spent == {"candidate": pytest.approx(0.02)}, "each cost is added as it lands"


@pytest.mark.parametrize("cost_usd", [None, 0.004])
async def test_a_capped_client_makes_a_call_its_client_prices_at_nan_and_counts_what_it_cost(
    cost_usd: float | None,
) -> None:
    """A cell prices no call before making it: a ceiling no cap could read (NaN) neither refuses the call nor raises,
    and the call lands at what it cost, unpriced when the provider priced nothing. Launch pricing still refuses an
    unusable ceiling (``call_ceiling``, below)."""
    paid = _PricedClient(cost_usd=cost_usd, ceiling_usd=math.nan)
    sink = FakeCellSink()
    client = CappedCompletion(paid, sink=sink, source="candidate")

    await client.generate(system="s", user="u")

    assert paid.calls == 1, "the call was made"
    assert sink.spent == {"candidate": None if cost_usd is None else pytest.approx(cost_usd)}


async def test_a_capped_client_counts_nothing_when_the_call_raises() -> None:
    class _Broken(_PricedClient):
        async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
            raise RuntimeError("the provider fell over")

    sink = FakeCellSink()
    client = CappedCompletion(_Broken(cost_usd=0.1, ceiling_usd=0.5), sink=sink, source="candidate")

    with pytest.raises(RuntimeError, match="fell over"):
        await client.generate(system="s", user="u")

    assert sink.spent == {}, "a call that raised counts nothing"


@dataclass
class _MistypedSentResponse(ScriptedResponse):
    """A billed response whose ``sent_settings`` holds a word where request settings belong: the host's defect."""

    sent_settings: Any = "low"


@dataclass
class _MistypedSentClient(_PricedClient):
    """A priced client whose every completion fills ``sent_settings`` wrong."""

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        self.calls += 1
        return _MistypedSentResponse(content="paid", model=RUN_MODELS[0], cost_usd=self.cost_usd)


@pytest.mark.parametrize("source", ["candidate", "simulator"])
async def test_a_capped_client_counts_a_billed_call_whatever_its_sent_settings_hold(source: Any) -> None:
    """The cap reads the spend alone: a wrong-typed ``sent_settings`` neither leaves the call uncounted nor fails the
    call, and a candidate's call is never read for what it was sent with."""
    sink = FakeCellSink()
    client = CappedCompletion(_MistypedSentClient(cost_usd=0.004, ceiling_usd=0.01), sink=sink, source=source)

    returned = await client.generate(system="s", user="u")

    assert returned.sent_settings == "low", "the completion is handed back unchanged"
    assert sink.spent == {source: pytest.approx(0.004)}


async def test_a_simulator_call_whose_sent_settings_is_refused_is_counted_and_recorded_before_the_defect_leaves() -> (
    None
):
    """Through the conversation loop: the call's cost is counted, the driver's calls hold it, and the host's defect
    stops the driver ``simulator_error`` and propagates."""
    sink = FakeCellSink()
    driver = TurnDriver(conversation=ConversationSpec(actors=[actor("a")], max_turns=2), variation={})

    async def answer(_turns: Sequence[SimulatorTurn]) -> CandidateTurn:
        raise AssertionError("no round is answered: the first simulator call is refused")

    async def post(_turn: SimulatorTurn) -> None:
        return None

    with pytest.raises(CompletionProtocolDefect, match="sent_settings"):
        await drive_conversation(
            driver, answer, post, llm=_MistypedSentClient(cost_usd=0.004, ceiling_usd=0.01), sink=sink
        )

    assert sink.spent == {"simulator": pytest.approx(0.004)}
    assert driver.stop_cause is ConversationStopCause.SIMULATOR_ERROR
    ledger = RoleUsageLedger(role="simulator")
    driver.fold_usage(ledger)
    (row,) = ledger.rows()
    assert (row.call_count, row.cost_usd, row.sent_settings) == (1, pytest.approx(0.004), None)


def test_a_ceiling_is_the_clients_own_times_the_attempts_and_unpriced_without_one() -> None:
    assert call_ceiling(_PricedClient(cost_usd=0.0, ceiling_usd=0.25), system="s", user="u", attempts=3) == 0.75
    assert call_ceiling(_PricedClient(cost_usd=0.0, ceiling_usd=None), system="s", user="u") is None
    assert call_ceiling(object(), system="s", user="u") is None, "a client with no price_ceiling cannot say"
    with pytest.raises(ValueError, match="a ceiling is a finite amount"):
        call_ceiling(_PricedClient(cost_usd=0.0, ceiling_usd=-1.0), system="s", user="u")
    with pytest.raises(ValueError, match="a ceiling is a finite amount"):
        call_ceiling(_PricedClient(cost_usd=0.0, ceiling_usd=math.nan), system="s", user="u")
