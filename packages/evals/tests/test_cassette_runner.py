"""The runner hands each cell its cassettes, and holds the kind's report of background work to the record.

These drive :func:`~threetears.evals.run.runner.execute_run` end to end with one game-master kind
instance wired for every cell, the shape an adopter writes: its ``prepare`` wires whatever
``cassettes`` it is handed, and nothing about the run's corpus, template or case is bound into it.

The session rolls a d20 twice and sends two scouts — north (slow) and the chapel (fast) — so the
corpus has a repeated ask and two pieces of background work that finish in the opposite order to the
one they were sent in. Its live tools answer from the model it runs on, so a replay that went live,
or replayed another model's capture, would show.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from packages.evals.tests.factories import DEFAULT_KIND, make_eval_run, make_template, make_test_case
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from threetears.evals.kernel.candidate_kind import CandidateOutput, CandidateTelemetry, CellSink, VariantConfig
from threetears.evals.kernel.cassettes import (
    ActionSeam,
    CellCassettes,
    DeliveryRecorder,
    DeliveryReplay,
    DeliverySeam,
    ToolWrap,
)
from threetears.evals.kernel.host.apparatus import ApparatusError
from threetears.evals.kernel.host.eval_host import EvalHost
from threetears.evals.schema.models import (
    AsyncDelivery,
    AsyncExternalSpend,
    CassetteKey,
    EvalResult,
    EvalRun,
    JudgedArtifact,
    RoleUsage,
)
from threetears.evals.kernel.spend import ExternalRateTable
from threetears.evals.kernel.usage_capture import count_substituted_deliveries, production_replicating_cost
from threetears.evals.run.cassette_proxy import DELIVERY_ACTION, ReplayReportDefect, params_hash
from threetears.evals.run.runner import EveryCellApparatusFailedError, RunnerOptions, execute_run

_SCOUT_MODEL = "scout/model"


class _Roll(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total: int


class _Report(BaseModel):
    model_config = ConfigDict(extra="forbid")
    area: str
    findings: str


# parity-with: threetears.evals.kernel.cassettes.ToolLike
class _FakeDice:
    """Rolls answered from the model the session runs on, so two models' rolls never coincide."""

    def __init__(self, seed: int) -> None:
        self._next = seed
        self.live_rolls = 0

    @property
    def name(self) -> str:
        return "roll"

    def can_dispatch(self, action: str) -> bool:
        return action == "roll"

    async def act(self, action: str, parameters: dict[str, Any]) -> _Roll:
        self.live_rolls += 1
        self._next += 1
        await asyncio.sleep(0)
        return _Roll(total=self._next)


# parity-with: threetears.evals.kernel.cassettes.DeliverySeam
class _FakeScouts:
    """The asynchronous tool: each scout is started now and reports after its own latency."""

    _LATENCY = {"north": 5, "chapel": 1}

    def __init__(self, model: str) -> None:
        self._model = model
        self.recorder: DeliveryRecorder | None = None
        self.replay: DeliveryReplay[Any] | None = None
        self.live_scouts = 0
        self.reports: dict[str, str] = {}
        self.pending: list[asyncio.Task[None]] = []
        self.served = 0

    @property
    def payload_type(self) -> type[_Report]:
        return _Report

    def arm_capture(self, recorder: DeliveryRecorder) -> None:
        self.recorder = recorder

    def arm_replay(self, replay: DeliveryReplay[Any]) -> None:
        self.replay = replay

    def send(self, area: str) -> None:
        if self.replay is not None:
            delivery = self.replay.next({"area": area})
            self.served += 1
            if delivery.payload is not None:
                self.reports[area] = delivery.payload.findings
            return
        self.live_scouts += 1
        ticket = self.recorder.started({"area": area}) if self.recorder is not None else None
        self.pending.append(asyncio.create_task(self._scout(area, ticket)))

    async def _scout(self, area: str, ticket: Any) -> None:
        for _ in range(self._LATENCY[area]):
            await asyncio.sleep(0)
        report = _Report(area=area, findings=f"{area} as {self._model} saw it")
        self.reports[area] = report.findings
        if ticket is not None:
            ticket.delivered(report)


# parity-with: threetears.evals.kernel.cassettes.CassetteSeams
class _FakeSession:
    """One prepared game-master session: its dice, its scouts, and the seams it exposes."""

    def __init__(self, model: str) -> None:
        self.dice = _FakeDice(seed=100 if model == "gm/a" else 200)
        self.scouts = _FakeScouts(model)
        self.tools: dict[str, Any] = {"roll": self.dice}

    @property
    def action_seam(self) -> ActionSeam | None:
        return self  # type: ignore[return-value]

    @property
    def delivery_seams(self) -> Mapping[str, DeliverySeam]:
        return {"scout_ahead": self.scouts}

    @property
    def recorded_tools(self) -> Mapping[str, type[_Roll]]:
        return {"roll": _Roll}

    def arm_tools(self, wrap: ToolWrap) -> None:
        self.tools = wrap(self.tools)


# parity-with: threetears.evals.kernel.candidate_kind.CandidateKind
class _FakeGmKind:
    """One instance for every cell, as an adopter writes it: ``prepare`` wires what it is handed.

    ``report`` says how the kind reports a scout it was served from a replay: ``"honest"`` (as
    substituted), ``"as-live"`` (the defect a replay must refuse), or ``"omitted"`` (left out).
    """

    judged_artifact = JudgedArtifact.UNJUDGED

    def __init__(self, *, wire: bool = True, report: str = "honest") -> None:
        self._wire = wire
        self._report = report
        self.sessions: list[_FakeSession] = []

    async def prepare(
        self,
        *,
        subject_snapshot: Any,
        variant_config: VariantConfig,
        world_seed: Any,
        span_window: Any,
        cassettes: CellCassettes | None,
        world: Any,
    ) -> _FakeSession:
        session = _FakeSession(variant_config.candidate_model)
        if cassettes is not None and self._wire:
            cassettes.wire(session)
        self.sessions.append(session)
        return session

    async def invoke(self, instance: _FakeSession, test_case: Any, sink: CellSink) -> CandidateOutput:
        rolls = [(await instance.tools["roll"].act("roll", {"dice": "1d20"})).total for _ in range(2)]
        instance.scouts.send("north")
        instance.scouts.send("chapel")
        await asyncio.gather(*instance.scouts.pending)
        replayed = instance.scouts.replay is not None
        deliveries = [
            AsyncDelivery(
                tool="scout_ahead",
                status="delivered",
                substituted=replayed and self._report != "as-live",
                model=None if replayed else _SCOUT_MODEL,
                summary=instance.scouts.reports[area],
                cost_usd=None if replayed else 0.01,
            )
            for area in ("north", "chapel")
        ]
        if replayed and self._report == "omitted":
            deliveries = deliveries[:1]
        return CandidateOutput(
            output=[{"rolls": rolls, "reports": dict(sorted(instance.scouts.reports.items()))}],
            async_deliveries=deliveries,
            telemetry=CandidateTelemetry(usage=[RoleUsage(role="candidate", cost_usd=0.002)]),
        )


# No goal checks: the fake kinds here grade none, and the runner refuses a completed cell that left a
# template's checks ungraded.
_TEMPLATE = make_template(id="tpl-ambush", candidate_kind=DEFAULT_KIND, goal_state_checks=[])
_CASES = [make_test_case(id=f"case-{n}", template_id=_TEMPLATE.id) for n in (1, 2)]


def _run(*, model: str = "gm/a", **cassette: Any) -> EvalRun:
    return make_eval_run(
        template_id=_TEMPLATE.id,
        test_case_ids=[case.id for case in _CASES],
        candidate_kind=DEFAULT_KIND,
        candidate_model=model,
        **cassette,
    )


async def _execute(host: EvalHost, run: EvalRun, kind: _FakeGmKind) -> list[EvalResult]:
    await execute_run(
        host,
        run=run,
        template=_TEMPLATE,
        test_cases=_CASES,
        judge_service=None,
        options=RunnerOptions(candidate_kinds={DEFAULT_KIND: lambda _cell: kind}),
    )
    return sorted(host.storage.query_eval_results_by_run(run.id, run.scope_id), key=lambda r: r.test_case_id)


def _outputs(host: EvalHost, results: list[EvalResult]) -> list[Any]:
    return [host.storage.load_eval_trace(result.id, result.scope_id).trace for result in results]  # type: ignore[union-attr]


# =============================================================================
# One kind instance, every cell: the lane reaches the cell the runner is executing
# =============================================================================


async def test_one_kind_instance_captures_every_cell_and_a_replay_reproduces_it_without_going_live() -> None:
    host = toyhost_host()
    capture_kind = _FakeGmKind()
    capture = _run(cassette_mode="capture")
    captured = await _execute(host, capture, capture_kind)

    replay_kind = _FakeGmKind()
    replayed = await _execute(host, _run(cassette_mode="replay", cassette_corpus_id=capture.id), replay_kind)

    assert _outputs(host, replayed) == _outputs(host, captured)
    assert [s.dice.live_rolls for s in replay_kind.sessions] == [0, 0]
    assert [s.scouts.live_scouts for s in replay_kind.sessions] == [0, 0]
    # The slow north scout and the fast chapel scout each got their own report back.
    (first, *_) = _outputs(host, replayed)
    assert first[0]["reports"] == {"chapel": "chapel as gm/a saw it", "north": "north as gm/a saw it"}


async def test_two_capture_runs_executed_at_once_each_replay_as_their_own_model_captured() -> None:
    """A launch over two models runs one capture per model concurrently, over the same cases."""
    host = toyhost_host()
    run_a, run_b = _run(model="gm/a", cassette_mode="capture"), _run(model="gm/b", cassette_mode="capture")
    captured_a, captured_b = await asyncio.gather(
        _execute(host, run_a, _FakeGmKind()), _execute(host, run_b, _FakeGmKind())
    )

    for capture, captured in ((run_a, captured_a), (run_b, captured_b)):
        replayed = await _execute(
            host, _run(model="gm/c", cassette_mode="replay", cassette_corpus_id=capture.id), _FakeGmKind()
        )
        assert _outputs(host, replayed) == _outputs(host, captured)


async def test_a_kind_that_does_not_wire_its_cassettes_is_refused_before_its_candidate_runs() -> None:
    host = toyhost_host()
    kind = _FakeGmKind(wire=False)
    with pytest.raises(ValueError, match="did not wire them"):
        await _execute(host, _run(cassette_mode="replay", cassette_corpus_id="run-capture"), kind)
    assert [s.dice.live_rolls for s in kind.sessions] == [0]


async def test_a_run_with_cassettes_off_hands_its_cells_none() -> None:
    host = toyhost_host()
    kind = _FakeGmKind()
    (result, _) = await _execute(host, _run(), kind)
    assert [s.dice.live_rolls for s in kind.sessions] == [2, 2]
    assert result.async_deliveries is not None and not any(d.substituted for d in result.async_deliveries)


# =============================================================================
# A replayed delivery is substituted by construction
# =============================================================================


async def test_a_replayed_delivery_is_reported_substituted_and_withholds_production_cost() -> None:
    host = toyhost_host()
    capture = _run(cassette_mode="capture")
    await _execute(host, capture, _FakeGmKind())
    replayed = await _execute(host, _run(cassette_mode="replay", cassette_corpus_id=capture.id), _FakeGmKind())

    for result in replayed:
        assert count_substituted_deliveries(result) == 2
        assert production_replicating_cost(result.usage, substituted_deliveries=2) is None


@pytest.mark.parametrize(
    ("report", "message"), [("as-live", "as live"), ("omitted", "leaves out replayed work")], ids=["as-live", "omitted"]
)
async def test_a_kind_misreporting_what_it_was_replayed_is_refused(report: str, message: str) -> None:
    host = toyhost_host()
    capture = _run(cassette_mode="capture")
    await _execute(host, capture, _FakeGmKind())

    with pytest.raises(ReplayReportDefect, match=message):
        await _execute(host, _run(cassette_mode="replay", cassette_corpus_id=capture.id), _FakeGmKind(report=report))


# =============================================================================
# Background work's spend travels on its AsyncDelivery and the runner folds it
# =============================================================================


async def test_background_work_spend_reaches_the_result_and_its_production_cost() -> None:
    host = toyhost_host()
    (result, _) = await _execute(host, _run(), _FakeGmKind())

    inner = [row for row in result.usage if row.role == "inner_agent"]
    assert [(row.model, row.cost_usd) for row in inner] == [(_SCOUT_MODEL, 0.02)]
    # The kind's own 0.002 plus the two scouts' 0.01 each.
    assert result.cost_usd == pytest.approx(0.022)
    assert production_replicating_cost(result.usage, substituted_deliveries=0) == pytest.approx(0.022)


# parity-with: threetears.evals.kernel.candidate_kind.CandidateKind
class _FakeDoubleReportingKind(_FakeGmKind):
    """Reports its background work's spend twice: on the delivery and as its own telemetry row."""

    async def invoke(self, instance: _FakeSession, test_case: Any, sink: CellSink) -> CandidateOutput:
        output = await super().invoke(instance, test_case, sink)
        doubled = [*output.telemetry.usage, RoleUsage(role="inner_agent", model=_SCOUT_MODEL, cost_usd=0.02)]
        return output.model_copy(update={"telemetry": output.telemetry.model_copy(update={"usage": doubled})})


async def test_a_kind_reporting_background_spend_on_its_telemetry_too_is_refused() -> None:
    host = toyhost_host()
    with pytest.raises(ValueError, match="would count it twice"):
        await _execute(host, _run(), _FakeDoubleReportingKind())


# parity-with: threetears.evals.kernel.candidate_kind.CandidateKind
class _FakeScoutStillOutKind(_FakeGmKind):
    """Starts a scout, reports it in flight with what it has spent so far, and outlives the deadline."""

    async def invoke(self, instance: _FakeSession, test_case: Any, sink: CellSink) -> CandidateOutput:
        sink.report_progress(
            lambda: CandidateOutput(
                output=[{"waiting": "on the scout"}],
                async_deliveries=[
                    AsyncDelivery(
                        tool="scout_ahead",
                        status="undelivered",
                        model=_SCOUT_MODEL,
                        substituted=False,
                        cost_usd=0.03,
                        input_tokens=900,
                    )
                ],
            )
        )
        sink.waiting_on("background_work")
        await asyncio.Event().wait()
        raise AssertionError("unreachable: the cell deadline cancels this wait")


async def test_an_undelivered_scout_s_spend_survives_the_cell_deadline() -> None:
    host = toyhost_host()
    run = _run().model_copy(update={"test_case_ids": [_CASES[0].id]})
    await execute_run(
        host,
        run=run,
        template=_TEMPLATE,
        test_cases=_CASES[:1],
        judge_service=None,
        options=RunnerOptions(
            candidate_kinds={DEFAULT_KIND: lambda _cell: _FakeScoutStillOutKind()}, cell_timeout_s=0.05
        ),
    )
    (result,) = host.storage.query_eval_results_by_run(run.id, run.scope_id)

    assert result.termination == "cell_timeout"
    assert [(row.role, row.cost_usd, row.prompt_tokens) for row in result.usage] == [("inner_agent", 0.03, 900)]
    assert result.cost_usd == pytest.approx(0.03)


# =============================================================================
# Every cut-short exit closes the cell's cassettes and folds its background work's spend
# =============================================================================


# parity-with: threetears.evals.kernel.candidate_kind.CandidateKind
class _FakeRigBreaksMidSceneKind(_FakeGmKind):
    """Sends the slow north scout under a capture, reports it in flight, then meets a rig fault.

    Its own teardown cancels the scout, as a kind's unwind would, so the capture never settles it:
    only the cell's close can record it.
    """

    async def invoke(self, instance: _FakeSession, test_case: Any, sink: CellSink) -> CandidateOutput:
        instance.scouts.send("north")
        sink.report_progress(
            lambda: CandidateOutput(
                async_deliveries=[
                    AsyncDelivery(
                        tool="scout_ahead", status="undelivered", model=_SCOUT_MODEL, substituted=False, cost_usd=0.01
                    )
                ],
                telemetry=CandidateTelemetry(usage=[RoleUsage(role="candidate", cost_usd=0.002)]),
            )
        )
        for task in instance.scouts.pending:
            task.cancel()
        raise ApparatusError("the encounter table no longer loads")


async def test_an_apparatus_fault_mid_capture_closes_the_cells_cassettes_and_keeps_its_background_spend() -> None:
    host = toyhost_host()
    run = _run(cassette_mode="capture").model_copy(update={"test_case_ids": [_CASES[0].id]})
    with pytest.raises(EveryCellApparatusFailedError):
        await execute_run(
            host,
            run=run,
            template=_TEMPLATE,
            test_cases=_CASES[:1],
            judge_service=None,
            options=RunnerOptions(candidate_kinds={DEFAULT_KIND: lambda _cell: _FakeRigBreaksMidSceneKind()}),
        )
    (result,) = host.storage.query_eval_results_by_run(run.id, run.scope_id)

    assert result.termination == "apparatus_failed"
    assert [(row.role, row.cost_usd) for row in result.usage] == [("candidate", 0.002), ("inner_agent", 0.01)]
    assert result.cost_usd == pytest.approx(0.012)
    recorded = host.storage.get_cassette(
        CassetteKey(
            corpus_id=run.id,
            template_id=_TEMPLATE.id,
            test_case_id=_CASES[0].id,
            tool="scout_ahead",
            action=DELIVERY_ACTION,
            params_hash=params_hash({"area": "north"}),
            occurrence=0,
        ),
        run.scope_id,
    )
    assert recorded is not None and recorded.outcome == "undelivered"


# parity-with: threetears.evals.kernel.candidate_kind.CandidateKind
class _FakeBreaksWhilePreparingKind(_FakeGmKind):
    """Wires its cassettes, starts a scout while setting the scene, then meets a rig fault in ``prepare``.

    ``invoke`` never runs, so nothing but the cut-short record can close the cell's handle.
    """

    async def prepare(self, **kwargs: Any) -> _FakeSession:
        session = await super().prepare(**kwargs)
        session.scouts.send("north")
        for task in session.scouts.pending:
            task.cancel()
        raise ApparatusError("the encounter table no longer loads")


async def test_an_apparatus_fault_in_prepare_still_closes_the_cells_cassettes() -> None:
    host = toyhost_host()
    run = _run(cassette_mode="capture").model_copy(update={"test_case_ids": [_CASES[0].id]})
    with pytest.raises(EveryCellApparatusFailedError):
        await execute_run(
            host,
            run=run,
            template=_TEMPLATE,
            test_cases=_CASES[:1],
            judge_service=None,
            options=RunnerOptions(candidate_kinds={DEFAULT_KIND: lambda _cell: _FakeBreaksWhilePreparingKind()}),
        )

    recorded = host.storage.get_cassette(
        CassetteKey(
            corpus_id=run.id,
            template_id=_TEMPLATE.id,
            test_case_id=_CASES[0].id,
            tool="scout_ahead",
            action=DELIVERY_ACTION,
            params_hash=params_hash({"area": "north"}),
            occurrence=0,
        ),
        run.scope_id,
    )
    assert recorded is not None and recorded.outcome == "undelivered"


# =============================================================================
# Background work's spend reaches the cost, under the unpriced rule
# =============================================================================


# parity-with: threetears.evals.kernel.candidate_kind.CandidateKind
class _FakeReportingKind(_FakeGmKind):
    """Reports exactly the background work it was built with, beside its own priced call."""

    def __init__(self, deliveries: list[AsyncDelivery]) -> None:
        super().__init__()
        self._deliveries = deliveries

    async def invoke(self, instance: _FakeSession, test_case: Any, sink: CellSink) -> CandidateOutput:
        return CandidateOutput(
            output=[{"turn": 1}],
            async_deliveries=self._deliveries,
            telemetry=CandidateTelemetry(usage=[RoleUsage(role="candidate", cost_usd=0.002)]),
        )


def _scout(**spend: Any) -> AsyncDelivery:
    return AsyncDelivery(tool="scout_ahead", status="delivered", model=_SCOUT_MODEL, substituted=False, **spend)


async def _one_cell(kind: _FakeGmKind, *, rates: ExternalRateTable | None = None) -> EvalResult:
    host = toyhost_host()
    run = _run().model_copy(update={"test_case_ids": [_CASES[0].id]})
    await execute_run(
        host,
        run=run,
        template=_TEMPLATE,
        test_cases=_CASES[:1],
        judge_service=None,
        options=RunnerOptions(candidate_kinds={DEFAULT_KIND: lambda _cell: kind}, external_rates=rates),
    )
    (result,) = host.storage.query_eval_results_by_run(run.id, run.scope_id)
    return result


async def test_background_work_whose_model_went_unpriced_makes_the_cells_cost_unknown() -> None:
    result = await _one_cell(_FakeReportingKind([_scout(input_tokens=900, output_tokens=40, llm_calls=2)]))

    (inner,) = [row for row in result.usage if row.role == "inner_agent"]
    assert inner.cost_usd is None and inner.prompt_tokens == 900
    assert result.cost_usd is None


_RATES = ExternalRateTable(rates={("search", "credits"): 0.001})


async def test_background_works_paid_calls_are_in_the_cost_when_the_run_prices_them() -> None:
    spend = AsyncExternalSpend(provider="search", calls=2, provider_units=4, provider_unit="credits")
    result = await _one_cell(_FakeReportingKind([_scout(cost_usd=0.01, external_spend=[spend])]), rates=_RATES)

    assert "external" in result.cost_roles
    assert result.cost_usd == pytest.approx(0.002 + 0.01 + 0.004)


async def test_background_works_paid_calls_a_rated_run_cannot_price_make_the_cost_unknown() -> None:
    """The run's cost claims its external spend, so a provider it holds no rate for is unknown, not free."""
    spend = AsyncExternalSpend(provider="maps", calls=3)
    result = await _one_cell(_FakeReportingKind([_scout(cost_usd=0.01, external_spend=[spend])]), rates=_RATES)

    (external,) = [row for row in result.usage if row.role == "external"]
    assert external.cost_usd is None and external.call_count == 3
    assert result.cost_usd is None


async def test_paid_calls_in_a_run_that_declared_no_rates_are_volume_outside_its_cost() -> None:
    """With no rates declared the cost never claimed external dollars, so the calls are counted only."""
    spend = AsyncExternalSpend(provider="maps", calls=3)
    result = await _one_cell(_FakeReportingKind([_scout(cost_usd=0.01, external_spend=[spend])]))

    assert "external" not in result.cost_roles
    assert result.cost_usd == pytest.approx(0.012)


async def test_a_provider_reported_charge_reaches_the_cells_cost_where_the_run_holds_no_rate() -> None:
    """Background work billed per image: the provider's own charge prices calls no rate covers."""
    spend = AsyncExternalSpend(provider="images", calls=2, money=0.08)
    result = await _one_cell(_FakeReportingKind([_scout(cost_usd=0.01, external_spend=[spend])]), rates=_RATES)

    (external,) = [row for row in result.usage if row.role == "external"]
    assert (external.cost_usd, external.price_source, external.call_count) == (0.08, "images:reported", 2)
    assert result.cost_usd == pytest.approx(0.002 + 0.01 + 0.08)


async def test_a_provider_reported_charge_wins_over_the_runs_rate_for_the_same_unit() -> None:
    spend = AsyncExternalSpend(provider="search", calls=2, provider_units=4, provider_unit="credits", money=0.05)
    result = await _one_cell(_FakeReportingKind([_scout(cost_usd=0.01, external_spend=[spend])]), rates=_RATES)

    (external,) = [row for row in result.usage if row.role == "external"]
    assert (external.cost_usd, external.price_source) == (0.05, "search:reported")
    assert result.cost_usd == pytest.approx(0.002 + 0.01 + 0.05)


async def test_a_substituted_delivery_adds_nothing_to_the_cost() -> None:
    seeded = AsyncDelivery(tool="scout_ahead", status="delivered", substituted=True)
    result = await _one_cell(_FakeReportingKind([seeded]), rates=_RATES)

    assert [row.role for row in result.usage] == ["candidate"]
    assert result.cost_usd == pytest.approx(0.002)
