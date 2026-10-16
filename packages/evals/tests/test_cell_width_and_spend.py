"""How many cells run at once is the kind's to declare, and the cost cap is read between cells, never inside one.

Driven through the real launch and the real runner over the toy host, reading what was stored:

* **The width.** A run not measuring latency executes at the narrowest of what its kind declares it is
  safe at, the host's ceiling and its matrix. A launcher that says nothing runs its kind serially, and
  the runner refuses — before anything is spent — a kind declaring less than the run executes at.
* **Spend by source, never overwritten.** A cell's sink keeps one running total per source, each only
  ever added to.
* **No refusal inside a cell.** A priced call, an unpriced one and a judge call are all made whatever the
  run has spent: the cap is read between cells, against what the recorded cells landed, so a cell is measured
  whole or never begun, and a run lands past its cap by up to the width's worth of cells.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

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
from threetears.evals.kernel import CappedCompletion, call_ceiling
from threetears.evals.kernel.candidate_kind import CandidateOutput, CandidateTelemetry, CellSink
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.result_condition import ResultOutcome, classify_result
from threetears.evals.kernel.usage_capture import RoleUsageLedger
from threetears.evals.run import KindWiring, LaunchHost, RunnerOptions, drive_conversation, execute_run, start_run
from threetears.evals.run.budget import BudgetStoppedError, EvalRunCostCap
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

    ``ceiling_usd=None`` is a client that cannot price a call before making it.
    """

    cost_usd: float
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
    """Two cells of one run, side by side: the first reports spend on two sources, the second asks the sink.

    The first to invoke reports $0.6 of candidate spend and $0.3 of background work and waits; the second, once it
    has, asks to hold a $0.2 call and asks whether the cap is reached. Neither is refused: the $1 cap is read
    between cells.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reported = asyncio.Event()
        self.asked = asyncio.Event()
        self.order: list[str] = []
        self.refused: list[str] = []
        self.negative_refused: list[str] = []
        self.cap_reached: list[bool] = []

    async def invoke(self, instance: ToyExtractorInstance, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        self.order.append(test_case.id)
        if len(self.order) == 1:
            sink.add_spend("candidate", 0.6)
            sink.add_spend("background_work", 0.3)
            try:
                sink.add_spend("candidate", -0.6)
            except ValueError as refused:
                self.negative_refused.append(str(refused))
            self.cap_reached.append(sink.cost_cap_reached())
            self.reported.set()
            await self.asked.wait()
            return CandidateOutput(output=[{"first": True}])
        await self.reported.wait()
        sink.reserve("candidate", 0.2).release()
        self.cap_reached.append(sink.cost_cap_reached())
        self.asked.set()
        return CandidateOutput(output=[{"first": False}])


async def test_a_cells_spend_by_source_only_grows_and_no_sibling_is_refused_by_it() -> None:
    kinds: list[_ReportsBySourceKind] = []

    def kind_for(world: Any) -> ToyExtractorKind:
        kinds.append(_ReportsBySourceKind(client=ScriptedExtractionClient(), world=world, goal_checks=()))
        return kinds[0]

    _host, _run, _results, stopped, _cap = await _capped_run(kind_for, cap_usd=1.0, width=2, n_cases=2)

    (kind,) = kinds
    assert kind.negative_refused and "0 or more" in kind.negative_refused[0], "a source's total never decreases"
    assert kind.cap_reached == [False, False], (
        "$0.9 beside a $0.2 call is asked of no cell: the cap is read between cells"
    )
    assert stopped is None


# =============================================================================
# A conversing kind is not cut short by the run's spend
# =============================================================================


class _ConversingPaidKind(ToyExtractorKind):
    """A conversing kind whose candidate pays $0.01 for each answer, reserved through the cell's sink."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.answered: list[str] = []

    async def invoke(self, instance: ToyExtractorInstance, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        driver = TurnDriver(conversation=ConversationSpec(actors=[actor("a")], max_turns=5), variation={})
        table = ScriptedTable(lines={"a": [f"a{n}" for n in range(1, 6)]}, ceiling_usd=0.001)
        answered: list[str] = []
        candidate = RoleUsageLedger(role="candidate")

        async def answer(round_turns: Sequence[SimulatorTurn]) -> CandidateTurn:
            held = sink.reserve("candidate", 0.01)
            response = ScriptedResponse(content="answer", model=RUN_MODELS[0], cost_usd=0.01)
            candidate.add_llm_result(response)
            held.settle(response.cost_usd)
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
# The judge phase holds each call against the cap too
# =============================================================================


@dataclass
class _PricedJudge:
    """A judge client that scores every dim a 4, each call costing ``cost_usd`` and priced beforehand at ``ceiling_usd``."""

    cost_usd: float
    ceiling_usd: float | None
    calls: list[str] = field(default_factory=list)

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        del system, user, response_format
        return self.ceiling_usd

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        del user, response_format
        match = re.search(r'the single key "(.+?)"', system)
        assert match is not None
        dim = match.group(1)
        self.calls.append(dim)
        return ToyJudgeCompletion(
            content=json.dumps({"reasoning": "fine", "criteria_scores": {dim: 4}}),
            input_tokens=100,
            output_tokens=40,
            cost_usd=self.cost_usd,
            model=TOY_JUDGE_MODEL,
            price_source="toyhost_script",
        )

    async def aclose(self) -> None:
        """Nothing to release."""


async def _judged_run(judge: _PricedJudge, *, cap_usd: float) -> tuple[list[EvalResult], BudgetStoppedError | None]:
    host = toyhost_host(clients=lambda _role, _model, *, temperature=None: judge)
    world = host.profile.world
    assert world is not None
    template = toyhost_judged_template()
    kind = ToyExtractorKind(
        client=ScriptedExtractionClient(), world=world, judged=True, goal_checks=tuple(template.goal_state_checks)
    )
    run, cases = off_loop(_toy_run, host, kind, template=template)
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
    stopped: BudgetStoppedError | None = None
    try:
        await execute_run(
            host,
            run=run,
            template=template,
            test_cases=cases,
            judge_service=run_judge.service,
            options=RunnerOptions(candidate_kinds={TOY_EXTRACTOR_KIND: lambda _cell: kind}),
            budget_gate=cap.check,
            on_cost=cap.record,
        )
    except BudgetStoppedError as error:
        stopped = error
    return off_loop(host.storage.query_eval_results_by_run, run.id, run.scope_id), stopped


async def test_a_judge_call_is_sent_whatever_the_cap_has_room_for() -> None:
    judge = _PricedJudge(cost_usd=0.0004, ceiling_usd=1.0)

    (result,), stopped = await _judged_run(judge, cap_usd=0.5)

    assert judge.calls, "every dim was judged: a judge call is not held against the cap inside the cell"
    assert result.judge_error is None
    assert stopped is None, "one cell, already started: there was none left to keep from starting"


async def test_a_judge_phase_the_cap_has_room_for_is_judged_and_its_spend_counted() -> None:
    judge = _PricedJudge(cost_usd=0.0004, ceiling_usd=0.001)

    (result,), stopped = await _judged_run(judge, cap_usd=10.0)

    assert judge.calls, "every dim was judged"
    assert result.judge_error is None
    assert stopped is None


# =============================================================================
# The client wrapper and its pricing
# =============================================================================


async def test_a_capped_client_makes_every_priced_call_and_settles_each_to_what_it_cost() -> None:
    paid = _PricedClient(cost_usd=0.004, ceiling_usd=0.01)
    sink = FakeCellSink()
    client = CappedCompletion(paid, sink=sink, source="candidate")

    for _ in range(5):
        await client.generate(system="s", user="u")

    assert paid.calls == 5
    assert sink.spent == {"candidate": pytest.approx(0.02)}
    assert sink.held == [], "every reservation was closed"


async def test_a_capped_client_releases_its_reservation_when_the_call_raises() -> None:
    class _Broken(_PricedClient):
        async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
            raise RuntimeError("the provider fell over")

    sink = FakeCellSink()
    client = CappedCompletion(_Broken(cost_usd=0.1, ceiling_usd=0.5), sink=sink, source="candidate")

    with pytest.raises(RuntimeError, match="fell over"):
        await client.generate(system="s", user="u")

    assert sink.held == [] and sink.spent == {}, "a call that raised holds nothing and counts nothing"


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
async def test_a_capped_client_settles_a_billed_call_whatever_its_sent_settings_hold(source: Any) -> None:
    """The cap reads the spend alone: a wrong-typed ``sent_settings`` neither leaves the ceiling held nor fails the
    call, and a candidate's call is never read for what it was sent with."""
    sink = FakeCellSink()
    client = CappedCompletion(_MistypedSentClient(cost_usd=0.004, ceiling_usd=0.01), sink=sink, source=source)

    returned = await client.generate(system="s", user="u")

    assert returned.sent_settings == "low", "the completion is handed back unchanged"
    assert sink.held == [] and sink.spent == {source: pytest.approx(0.004)}


async def test_a_simulator_call_whose_sent_settings_is_refused_is_settled_and_recorded_before_the_defect_leaves() -> (
    None
):
    """Through the conversation loop: the reservation is settled to the call's cost, the driver's calls hold it, and
    the host's defect stops the driver ``simulator_error`` and propagates."""
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

    assert sink.held == [] and sink.spent == {"simulator": pytest.approx(0.004)}
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
