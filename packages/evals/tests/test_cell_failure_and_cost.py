"""What one cell's failure costs, and what its spend is recorded as — driven through the real runner.

Every case here is producer-built: the toy host's kind and the engine's own :func:`execute_run`
produce the result, and the assertion reads what storage holds. The defects these pin were each a
record the runner WROTE wrongly, so a hand-built result could not have shown any of them.

* A judge reply whose ``reasoning`` is not a string is a judge-reply defect for that dim, with its
  spend kept — never a run failure.
* An apparatus fault — a replay miss, any :class:`ApparatusError` — out of a kind's ``prepare`` or
  ``invoke`` excludes THAT cell with what it had spent, and the run goes on; a run whose every cell
  the rig excluded measured nothing and fails.
* A deadline in the judge phase keeps the cell's evidence and the scores that came back, names each
  dim that did not finish, and leaves the cell one a re-judge can recover.
* A cancel that lands mid-cell records the cell's spend before the cancel ends the run.
* Unpriced spend is unknown, never zero: a cell with an unpriced model call costs ``None``, a capped
  run stops on it, and every aggregate leaves it out of its dollars and counts it.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from threetears.evals.contracts import EvalResult, EvalStorage
from threetears.evals.contracts.candidate_kind import (
    CandidateOutput,
    CandidateTelemetry,
    CellSink,
    CellSpanWindow,
    VariantConfig,
)
from threetears.evals.contracts.host import ApparatusError, EvalHost, SubjectSnapshot, WorldRegistry
from threetears.evals.contracts.models import (
    CassetteKey,
    EvalTemplate,
    EvalTestCase,
    EvalTrace,
    JudgedArtifact,
    RoleUsage,
    RubricDim,
    WorldSeed,
)
from threetears.evals.contracts.result_condition import ResultOutcome, classify_result
from threetears.evals.contracts.cassettes import CassetteMiss, CellCassettes
from threetears.evals.run.judge import CANNOT_TELL, JUDGE_REQUEST_SETTINGS
from threetears.evals.run.launch import build_judge_service
from threetears.evals.run.lifecycle import rejudge_result
from threetears.evals.analysis.bundle import assemble_context_bundle
from threetears.evals.analysis.reporting import compute_estimate_cost, compute_orphaned_runs, compute_program_budget
from threetears.evals.contracts.campaign import EvalCampaign
from threetears.evals.contracts.usage_capture import RoleUsageLedger
from threetears.evals.run.budget import BudgetStoppedError, EvalRunCostCap
from threetears.evals.run.runner import EveryCellApparatusFailedError, RunnerOptions, execute_run
from packages.evals.tests.factories import make_eval_result, make_eval_run
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.judge import (
    FAITHFULNESS_DIM,
    TOY_JUDGE_MODEL,
    ToyJudgeCompletion,
    toyhost_judged_template,
)
from packages.evals.tests.fixtures.toyhost.kind import (
    DOCUMENT_PARAM,
    TOY_EXTRACTOR_KIND,
    ScriptedExtractionClient,
    ToyExtractorInstance,
    ToyExtractorKind,
)
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_run, toyhost_template, toyhost_test_cases

# =============================================================================
# The drive: one toy run through the real runner, as small as the case needs
# =============================================================================


@dataclass
class _Drive:
    """One toy run, and what it left in storage."""

    host: EvalHost
    run_id: str
    scope_id: str
    cases: list[EvalTestCase]

    @property
    def storage(self) -> EvalStorage:
        """The host's store."""
        return self.host.storage

    @property
    def results(self) -> list[EvalResult]:
        """Every result the run stored."""
        return self.storage.query_eval_results_by_run(self.run_id, self.scope_id)

    def result_for(self, case: EvalTestCase) -> EvalResult:
        """The one result stored for ``case``."""
        (result,) = [result for result in self.results if result.test_case_id == case.id]
        return result

    def trace(self, result: EvalResult) -> EvalTrace | None:
        """The trace stored beside one result."""
        return self.storage.load_eval_trace(result.id, self.scope_id)


KindFor = Callable[[WorldRegistry], ToyExtractorKind]


def _drive(
    kind_for: KindFor | None = None,
    *,
    judge: Any = None,
    n_cases: int = 1,
    cell_timeout_s: float | None = None,
    template: EvalTemplate | None = None,
    cap: EvalRunCostCap | None = None,
) -> tuple[_Drive, Callable[[], Any]]:
    """Set up a toy run over ``n_cases`` invoices at ``k=1``, judged by ``judge`` when one is given.

    Returned unstarted, with the coroutine that runs it, so a test can read the drive after a run
    that raised and can cancel one mid-cell. A judged drive is launch-shaped: the template and cases
    are stored and the run records the judge attribution the launch's own judge build makes, so a
    re-judge of what it stored is the one an operator would make.

    Args:
        kind_for: Builds the kind from the host's world; ``None`` is the shipped toy extractor,
            judged when ``judge`` is.
        judge: A judge client — the host's judge for this drive and for any re-judge after it — or
            ``None`` for no judging.
        n_cases: How many of the three invoices to run.
        cell_timeout_s: The cell deadline; ``None`` keeps the runner's default.
        template: The template; ``None`` is the toy template, with its rubric when judged.
        cap: The run's cost cap, gating each cell and recording each result's cost as a launch
            wires it; ``None`` for none.

    Returns:
        The drive, and the no-argument coroutine function that runs it — and, however it ends,
        marks the run ``completed`` as the job manager would.
    """
    host = toyhost_host(clients=(lambda _role, _model, *, temperature=None: judge) if judge is not None else None)
    world = host.profile.world
    assert world is not None
    kind = (
        kind_for(world)
        if kind_for is not None
        else ToyExtractorKind(client=ScriptedExtractionClient(), world=world, judged=judge is not None)
    )
    if template is None:
        template = toyhost_judged_template() if judge is not None else toyhost_template()
    cases = toyhost_test_cases(template)[:n_cases]
    host.storage.save_template(template)
    for case in cases:
        host.storage.save_test_case(case)
    run_fields: dict[str, Any] = {"k_runs": 1, "test_case_ids": [case.id for case in cases]}
    service = None
    if judge is not None:
        run_judge = build_judge_service(host, template, TOY_JUDGE_MODEL, judged_artifact=JudgedArtifact.DOCUMENT)
        service = run_judge.service
        run_fields |= {
            "judge_model": TOY_JUDGE_MODEL,
            "effective_judges": run_judge.effective_judges,
            "judge_config_ids": {dim: config.id for dim, config in run_judge.configs.items()},
            "judge_request_settings": JUDGE_REQUEST_SETTINGS,
        }
    run = toyhost_run(model=RUN_MODELS[0], template=template, kind=kind, world=world).model_copy(update=run_fields)
    host.storage.save_eval_run(run)
    options = RunnerOptions(candidate_kinds={TOY_EXTRACTOR_KIND: lambda _cell: kind})
    if cell_timeout_s is not None:
        options.cell_timeout_s = cell_timeout_s

    async def go() -> None:
        try:
            await execute_run(
                host,
                run=run,
                template=template,
                test_cases=cases,
                judge_service=service,
                options=options,
                budget_gate=cap.check if cap is not None else None,
                on_cost=cap.record if cap is not None else None,
            )
        finally:
            host.storage.save_eval_run(run.model_copy(update={"status": "completed"}))

    return _Drive(host=host, run_id=run.id, scope_id=run.scope_id, cases=cases), go


async def _driven(kind_for: KindFor | None = None, **kwargs: Any) -> _Drive:
    """:func:`_drive`, run to its end."""
    drive, go = _drive(kind_for, **kwargs)
    await go()
    return drive


def _dim_of(system: str) -> str:
    """The dim a judge prompt asks to be scored."""
    match = re.search(r'the single key "(.+?)"', system)
    assert match is not None
    return match.group(1)


@dataclass
class _ScriptedJudge:
    """A judge client answering each dim with its scripted reply, priced the way a provider prices it.

    A dim named in ``hangs`` never answers — the call waits until the cell's deadline cancels it.
    """

    replies: dict[str, dict[str, Any]] = field(default_factory=dict)
    default: dict[str, Any] | None = None
    hangs: set[str] = field(default_factory=set)
    cost_usd: float | None = 0.0004
    #: The dim of every call, in call order.
    calls: list[str] = field(default_factory=list)

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, str] | None = None
    ) -> ToyJudgeCompletion:
        """Answer the dim the prompt asks for.

        Args:
            system: The system prompt, which names the dim.
            user: Ignored.
            response_format: Ignored.

        Returns:
            The dim's reply.
        """
        dim = _dim_of(system)
        self.calls.append(dim)
        if dim in self.hangs:
            await asyncio.Event().wait()
        reply = self.replies.get(dim, self.default)
        assert reply is not None, f"no reply scripted for {dim}"
        return ToyJudgeCompletion(
            content=json.dumps(reply),
            input_tokens=100,
            output_tokens=40,
            cost_usd=self.cost_usd,
            model=TOY_JUDGE_MODEL,
            price_source="toyhost_script" if self.cost_usd is not None else None,
        )

    async def aclose(self) -> None:
        """Nothing to release."""


def _scored(dim: str, score: int = 4) -> dict[str, Any]:
    return {"reasoning": f"{dim} read as a {score}", "criteria_scores": {dim: score}}


# =============================================================================
# A malformed judge reply is that dim's defect, with its spend kept
# =============================================================================


async def test_a_cannot_tell_whose_reasoning_is_not_a_string_is_a_judge_defect_not_a_failed_run():
    judge = _ScriptedJudge(
        default={"reasoning": {"missing": "the totals"}, "criteria_scores": {FAITHFULNESS_DIM: CANNOT_TELL}}
    )

    drive = await _driven(judge=judge)

    (result,) = drive.results
    assert result.judge_error is not None and FAITHFULNESS_DIM in result.judge_error
    assert "Failed to parse" in result.judge_error
    assert result.judge_cannot_tell == {}
    # Both paid attempts are on the record.
    (judge_row,) = [row for row in result.usage if row.role == "judge"]
    assert judge_row.call_count == 2 == len(judge.calls)
    assert judge_row.cost_usd == pytest.approx(0.0008)


# =============================================================================
# An apparatus fault excludes its cell, never the run
# =============================================================================

#: What the kind below had billed before the rig broke under it.
_SPENT_BEFORE_THE_FAULT = 0.0031


class _RigBreaksUnderKind(ToyExtractorKind):
    """A toy extractor whose tool boundary re-raises a replay miss, as the cassette seam tells it to.

    It breaks on the documents named in ``breaks_on``, after a first call it has already paid for
    and reported through the sink; every other cell runs as the shipped kind does.
    """

    def __init__(self, *args: Any, breaks_on: frozenset[str], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.breaks_on = breaks_on

    async def invoke(self, instance: ToyExtractorInstance, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        if test_case.variation_params[DOCUMENT_PARAM] not in self.breaks_on:
            return await super().invoke(instance, test_case, sink)
        sink.report_progress(
            lambda: CandidateOutput(
                output=[{"turn": "asked the ledger for the vendor"}],
                telemetry=CandidateTelemetry(
                    usage=[
                        RoleUsage(
                            role="candidate",
                            model=instance.model,
                            prompt_tokens=300,
                            completion_tokens=20,
                            cost_usd=_SPENT_BEFORE_THE_FAULT,
                            price_source="toyhost_script",
                            call_count=1,
                        )
                    ],
                ),
            )
        )
        raise CassetteMiss(
            CassetteKey(
                corpus_id="capture-run",
                template_id=test_case.template_id,
                test_case_id=test_case.id,
                tool="ledger",
                action="lookup",
                params_hash="0" * 16,
                occurrence=0,
            )
        )


def _breaking_on(*documents: str) -> KindFor:
    return lambda world: _RigBreaksUnderKind(
        client=ScriptedExtractionClient(), world=world, breaks_on=frozenset(documents)
    )


async def test_an_apparatus_fault_mid_cell_excludes_that_cell_with_its_spend_and_the_run_goes_on():
    first = toyhost_test_cases(toyhost_template())[0]

    drive = await _driven(_breaking_on(first.variation_params[DOCUMENT_PARAM]), n_cases=3)

    assert len(drive.results) == 3, "the fault ended one cell, not the run"
    faulted = drive.result_for(first)
    assert faulted.termination == "apparatus_failed"
    assert faulted.infra_error is not None and faulted.infra_error.startswith("apparatus: CassetteMiss")
    assert faulted.candidate_error is None
    assert classify_result(faulted) is ResultOutcome.INFRA_EXCLUDE
    (row,) = faulted.usage
    assert row.cost_usd == _SPENT_BEFORE_THE_FAULT
    assert faulted.cost_usd == _SPENT_BEFORE_THE_FAULT
    trace = drive.trace(faulted)
    assert trace is not None and trace.trace == [{"turn": "asked the ledger for the vendor"}]
    others = [result for result in drive.results if result.id != faulted.id]
    assert [result.termination for result in others] == ["completed", "completed"]


class _RigFailsToPrepareKind(ToyExtractorKind):
    """A toy extractor whose preparation meets a corrupt recording on its first cell only."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.prepared = 0

    async def prepare(
        self,
        *,
        subject_snapshot: SubjectSnapshot | None,
        variant_config: VariantConfig,
        world_seed: WorldSeed,
        span_window: CellSpanWindow,
        cassettes: CellCassettes | None,
    ) -> ToyExtractorInstance:
        self.prepared += 1
        if self.prepared == 1:
            raise ApparatusError("the delivery corpus for this case does not rebuild")
        return await super().prepare(
            subject_snapshot=subject_snapshot,
            variant_config=variant_config,
            world_seed=world_seed,
            span_window=span_window,
            cassettes=cassettes,
        )


async def test_an_apparatus_fault_in_prepare_excludes_that_cell_and_the_run_goes_on():
    drive = await _driven(
        lambda world: _RigFailsToPrepareKind(client=ScriptedExtractionClient(), world=world), n_cases=2
    )

    assert sorted(result.termination for result in drive.results) == ["apparatus_failed", "completed"]
    (faulted,) = [result for result in drive.results if result.termination == "apparatus_failed"]
    assert faulted.infra_error == "apparatus: ApparatusError: the delivery corpus for this case does not rebuild"
    assert faulted.usage == []
    assert classify_result(faulted) is ResultOutcome.INFRA_EXCLUDE


async def test_a_run_whose_every_cell_the_rig_excluded_fails_after_recording_them_all():
    """Every cell is still recorded; then the run fails, because it measured nothing."""
    documents = [case.variation_params[DOCUMENT_PARAM] for case in toyhost_test_cases(toyhost_template())]
    drive, go = _drive(_breaking_on(*documents), n_cases=2)

    with pytest.raises(EveryCellApparatusFailedError, match="every one of the run's 2 cell") as raised:
        await go()

    assert raised.value.total == 2
    assert "CassetteMiss" in str(raised.value)
    assert [result.termination for result in drive.results] == ["apparatus_failed", "apparatus_failed"]


# =============================================================================
# A deadline in the judge phase keeps what the cell paid for, and a re-judge recovers it
# =============================================================================

_SECOND_DIM = "extraction.vendor_named"


def _two_dim_template() -> EvalTemplate:
    template = toyhost_judged_template()
    return template.model_copy(
        update={
            "rubric": [
                *template.rubric,
                RubricDim(name=_SECOND_DIM, description="the vendor is named as the invoice names it", scale="ordinal"),
            ]
        }
    )


async def test_a_deadline_in_the_judge_phase_keeps_the_evidence_and_the_scores_that_came_back():
    judge = _ScriptedJudge(replies={FAITHFULNESS_DIM: _scored(FAITHFULNESS_DIM, 5)}, hangs={_SECOND_DIM})

    drive = await _driven(judge=judge, template=_two_dim_template(), cell_timeout_s=0.3)

    (result,) = drive.results
    assert result.termination == "cell_timeout"
    # The score that came back is on the record, with its call's spend.
    assert [(score.dim, score.score) for score in result.rubric_scores] == [(FAITHFULNESS_DIM, 5)]
    (judge_row,) = [row for row in result.usage if row.role == "judge"]
    assert judge_row.call_count == 1
    # The dim the deadline cut off is the judge's failure, named; the candidate's finished work is not.
    assert result.judge_error is not None and result.judge_error.startswith(f"{_SECOND_DIM}: did not finish")
    assert "cell timeout while waiting on the judge" in result.judge_error
    assert result.infra_error is None and result.candidate_error is None
    assert classify_result(result) is ResultOutcome.INFRA_EXCLUDE
    # What the judge read is kept, so it can be sent again.
    trace = drive.trace(result)
    assert trace is not None and trace.judge_evidence is not None
    assert trace.judged_artifact is JudgedArtifact.DOCUMENT


async def test_a_cell_whose_judging_the_deadline_cut_is_recovered_by_a_rejudge_of_the_unfinished_dim():
    judge = _ScriptedJudge(replies={FAITHFULNESS_DIM: _scored(FAITHFULNESS_DIM, 5)}, hangs={_SECOND_DIM})
    drive = await _driven(judge=judge, template=_two_dim_template(), cell_timeout_s=0.3)
    (result,) = drive.results
    judge.hangs.clear()
    judge.replies[_SECOND_DIM] = _scored(_SECOND_DIM, 3)
    judge.calls.clear()

    rejudged = await rejudge_result(drive.host, result.id, drive.scope_id)

    assert judge.calls == [_SECOND_DIM], "only the dim the deadline cut off is asked again"
    assert [(score.dim, score.score) for score in rejudged.rubric_scores] == [(FAITHFULNESS_DIM, 5), (_SECOND_DIM, 3)]
    assert rejudged.judge_error is None
    assert classify_result(rejudged) is ResultOutcome.OK


# =============================================================================
# NOTE — a cancel mid-cell keeps the cell's spend
# =============================================================================


class _StillWorkingKind(ToyExtractorKind):
    """A toy extractor that has paid for a call, reports it, and is still working when the run is cancelled."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.working = asyncio.Event()

    async def invoke(self, instance: ToyExtractorInstance, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        sink.report_progress(
            lambda: CandidateOutput(
                telemetry=CandidateTelemetry(
                    usage=[
                        RoleUsage(
                            role="candidate",
                            model=instance.model,
                            cost_usd=_SPENT_BEFORE_THE_FAULT,
                            price_source="toyhost_script",
                            call_count=1,
                        )
                    ],
                )
            )
        )
        self.working.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable: the run is cancelled during this wait")


async def test_a_cancel_mid_cell_records_the_cells_spend_before_ending_the_run():
    kinds: list[_StillWorkingKind] = []

    def kind_for(world: WorldRegistry) -> ToyExtractorKind:
        kinds.append(_StillWorkingKind(client=ScriptedExtractionClient(), world=world))
        return kinds[-1]

    drive, go = _drive(kind_for, n_cases=2)
    running = asyncio.ensure_future(go())
    await kinds[0].working.wait()
    running.cancel()

    with pytest.raises(asyncio.CancelledError):
        await running

    (result,) = drive.results
    assert result.termination == "cancelled"
    assert result.cost_usd == _SPENT_BEFORE_THE_FAULT
    assert [row.cost_usd for row in result.usage] == [_SPENT_BEFORE_THE_FAULT]
    assert result.infra_error is not None and "cancelled with its run" in result.infra_error
    assert classify_result(result) is ResultOutcome.INFRA_EXCLUDE


# =============================================================================
# Unpriced spend is a state, never zero
# =============================================================================


class _PartlyUnpricedKind(ToyExtractorKind):
    """A toy extractor whose client priced its first call and reported no price for its second.

    A local model, or a client that prices nothing, reports ``cost_usd=None``; the provider port
    allows it. The kind folds its calls the way every kind does, through a usage ledger.
    """

    async def invoke(self, instance: ToyExtractorInstance, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        ledger = RoleUsageLedger(role="candidate")
        for cost in (0.003, None):
            ledger.add(
                model=instance.model, prompt_tokens=200, completion_tokens=30, reasoning_tokens=None, cost_usd=cost
            )
        return CandidateOutput(output=[{"fields": {}}], telemetry=CandidateTelemetry(usage=ledger.rows()))


def _partly_unpriced(world: WorldRegistry) -> ToyExtractorKind:
    return _PartlyUnpricedKind(client=ScriptedExtractionClient(), world=world)


async def test_a_cell_whose_model_call_went_unpriced_costs_unknown_not_its_priced_part():
    drive = await _driven(_partly_unpriced)

    (result,) = drive.results
    (row,) = result.usage
    assert row.call_count == 2
    assert row.cost_usd is None, "a row holding an unpriced call cannot claim the priced call's dollars as its own"
    assert result.cost_usd is None


async def test_an_unpriced_judge_makes_the_cells_cost_unknown_while_its_candidate_stays_priced():
    drive = await _driven(judge=_ScriptedJudge(default=_scored(FAITHFULNESS_DIM), cost_usd=None))

    (result,) = drive.results
    rows = {row.role: row for row in result.usage}
    assert rows["candidate"].cost_usd is not None
    assert rows["judge"].cost_usd is None
    assert result.cost_usd is None


async def test_a_priced_cell_still_sums_its_rows():
    """The accepting side: every call priced, the cell's cost is their sum."""
    drive = await _driven(judge=_ScriptedJudge(default=_scored(FAITHFULNESS_DIM)))

    (result,) = drive.results
    assert result.cost_usd is not None
    assert result.cost_usd == pytest.approx(sum(row.cost_usd or 0.0 for row in result.usage))
    assert all(row.cost_usd is not None for row in result.usage)


async def test_a_capped_run_stops_on_its_first_unpriced_cell_however_far_below_its_ceiling():
    cap = EvalRunCostCap("toy-run", 100.0, enabled=True)
    drive, go = _drive(_partly_unpriced, n_cases=3, cap=cap)

    with pytest.raises(BudgetStoppedError, match="could not be priced") as stopped:
        await go()

    assert stopped.value.completed == 1
    assert stopped.value.breach.unpriced_results == 1
    assert len(drive.results) == 1, "the cell that went unpriced is kept, and no further cell was launched"


async def test_an_uncapped_run_carries_on_through_unpriced_cells_and_counts_them():
    cap = EvalRunCostCap("toy-run", 100.0, enabled=False)
    drive = await _driven(_partly_unpriced, n_cases=3, cap=cap)

    assert len(drive.results) == 3
    assert cap.unpriced_results == 3


def test_the_program_budget_leaves_unpriced_spend_out_of_its_dollars_and_counts_it():
    run = make_eval_run(status="completed")
    results = [
        make_eval_result(eval_run_id=run.id, cost_usd=0.10),
        make_eval_result(eval_run_id=run.id, cost_usd=None),
    ]

    budget = compute_program_budget([run], results)

    assert budget.total_cost_usd == pytest.approx(0.10)
    assert budget.n_unpriced == 1
    (row,) = budget.runs
    assert (row.cost_usd, row.n_results, row.n_unpriced) == (pytest.approx(0.10), 2, 1)


def test_an_orphaned_runs_unpriced_spend_is_counted_beside_its_dollars():
    run = make_eval_run(status="completed")
    results = [make_eval_result(eval_run_id=run.id, cost_usd=0.10), make_eval_result(eval_run_id=run.id, cost_usd=None)]

    orphans = compute_orphaned_runs([run], results, [])

    assert orphans.orphaned_cost_usd == pytest.approx(0.10)
    assert orphans.n_unpriced == 1
    assert orphans.orphaned_runs[0].n_unpriced == 1


def test_a_cost_estimate_draws_on_priced_history_and_counts_what_it_left_out():
    run = make_eval_run(status="completed", candidate_model="sonnet")
    results = [
        make_eval_result(eval_run_id=run.id, scope_id=run.scope_id, model="sonnet", cost_usd=0.20),
        make_eval_result(eval_run_id=run.id, scope_id=run.scope_id, model="sonnet", cost_usd=None),
    ]

    (cell,) = compute_estimate_cost(
        [run], results, models=["sonnet"], k_runs=1, n_test_cases=1, profile=toyhost_profile()
    ).cells

    assert cell.n_historical == 1
    assert cell.mean_cost_per_observation == pytest.approx(0.20), "an unpriced result is not a zero in the mean"
    assert cell.n_unpriced_historical == 1


def test_the_analysis_bundle_counts_unpriced_results_and_never_reads_their_rows_as_their_cost():
    """The rows decompose a cost; with the cost unknown, their priced part must not stand in for it."""
    from threetears.evals.contracts.models import RoleUsage as Row

    run = make_eval_run(status="completed")
    results = [
        make_eval_result(
            eval_run_id=run.id,
            test_case_id=f"tc-{i}",
            cost_usd=None,
            usage=[Row(role="candidate", model="m"), Row(role="judge", model="j", cost_usd=0.01, price_source="p")],
        )
        for i in range(2)
    ]
    campaign = EvalCampaign(
        scope_id=run.scope_id,
        name="c",
        subject_id=run.subject_snapshot.subject_id,
        subject_kind="s",
        behavior="b",
        run_ids=[run.id],
        created_by="test:fixture",
    )

    bundle = assemble_context_bundle(
        campaign, storage=ToyhostStorage([run], {run.id: results}), profile=toyhost_profile()
    )

    (summary,) = bundle.run_summaries
    assert summary.n_cost_unpriced == 2
    assert summary.cost_usd == 0.0
    assert "cost_usd" not in {measure.name for measure in bundle.telemetry.measures.measures}
