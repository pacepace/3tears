"""An arm whose calls were refused reads as failed — never as the fastest, cheapest arm on any surface.

A classifier arm whose every candidate call was refused showed 53 ms and $0 in the decision surface, and
nothing said every result had failed. Refusals are candidate failures, and they stay in the cell — every
rate and bar counts them against the arm — but their round trip and their empty spend were averaged into
the cell's latency and cost as if each were a turn.

**One predicate decides what a turn's time and spend are averaged over**:
:func:`~threetears.evals.kernel.delivered_a_turn`. It leaves out the harness's faults and the failures
that took no turn — the candidate's model refused or errored straight away. A failure that DID take a turn
— the host's turn budget ended it, the output cap cut it, the cell's deadline struck a pending call —
stays in, because that time and spend are what failing cost the arm: leaving them out read a contestant
whose slowest, costliest turns all failed as faster and cheaper than the control.

Pinned here, through a real store, the bundle, both reports, the frontier and a run's summary:

- **The predicate**, case by case.
- **A cost or latency reading is over the turns the candidate took** — on the cells, the run summaries,
  the frontier and the comparisons alike — while the failures still count in the arm's pass rate.
- **A turn the budget ended stays in cost and latency**, so a contestant whose budget-ended half took a
  minute never reads "improved" on either against the control.
- **An arm where no result took a turn** carries no cost or latency reading, counts its failures, reads
  ``no successful results`` under the surface's cost, latency and latency-bar columns and in the strata
  table, is named in a disclosure on both reports and in the bundle, dominates nothing on the frontier,
  and is compared on latency as "every result failed", not "too few cases".
- **A classifier's refusal is a miss**: an all-refusing arm reads accuracy 0 and is tested against the
  control, and a half-refusing arm does not read more accurate than the control for refusing.
- **Unmeasured cost is read over the turns taken**, so a billed refusal does not make a cell whose turns
  reported no spend look measured.
- **A run's summary** reads a cost or latency measure the same way and names what it left out.
- **The stored shape** refuses counts that cannot be true, keeps an older analysis readable as "not
  counted", and refuses ``delivered`` declared on anything but a turn's time or spend.

Mutations that turn this file red: reading ``scored`` or ``all_observed`` for an undeclared cost or latency
measure in ``summary_population``; widening ``delivered_a_turn`` to every candidate failure; the frontier's
latency or cost not asking ``delivered_a_turn``; ``_accuracy_leaves`` reading a failure's ``match`` as a hit;
dropping ``_failures_as_misses``; dropping ``n_no_turn`` from ``_cell_measures``; leaving the blank under an
all-failed cell's merit or bar column; not emitting the table's all-failed disclosure in ``_surface_blocks``.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Any

import jsonschema
import pytest
from pydantic import ValidationError

from threetears.evals.analysis import (
    AnalysisContextBundle,
    DisclosureBlock,
    Report,
    TableBlock,
    assemble_context_bundle,
    build_code_only_report,
    build_report,
    inspect_campaign_bundle,
    published_report_schema,
)
from threetears.evals.analysis.arms import short_digest
from threetears.evals.analysis.bundle import CellCoordinate, bundle_decision_surface
from threetears.evals.analysis.surface_table import NO_SUCCESSFUL_RESULTS, surface_table_of
from threetears.evals.kernel import CampaignDesign, CellFacts, EvalStorage, StratumFacts, delivered_a_turn
from threetears.evals.schema import EvalResult, EvalTestCase, GoalStateOutcome, LatencyMetrics, RoleUsage
from threetears.evals.kernel.analysis_measures import MeasureSummary
from threetears.evals.kernel.covariates import TRUNCATED_ROUNDS_KEY, TURN_BUDGET_ENDED_KEY, count_delivered_turns
from threetears.evals.kernel.declaration import BarOverride
from threetears.evals.kernel.host import SHARED_CORE, HostProfile, MeasureRegistry
from threetears.evals.kernel.metrics import MetricDescriptor, goal_check_measure
from threetears.evals.quick import Comparison, callable_host, compare, summarize_run
from threetears.evals.quick.one_call import UNUSABLE_ANSWER
from threetears.evals.run import list_results
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import (
    fixture_variant_key,
    make_analysis,
    make_campaign,
    make_eval_result,
    make_eval_run,
    minimal_declaration,
)

_SCOPE = "uni-1"

#: A host's own spend measure on the cost axis, as a host computes it from what its calls reported — the
#: shape that read $0 for a refusing arm, since a refusal reports nothing and the host's sum of nothing is 0.
_TURN_COST = "turn_cost_usd"


def _host_cost(**update: Any) -> MetricDescriptor:
    fields: dict[str, Any] = dict(
        name=_TURN_COST,
        reader_name="Turn cost",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description="What the candidate's calls for one turn cost, as the host summed them.",
        higher_is_better=False,
        unit="usd",
        merit_axis="cost",
    )
    return MetricDescriptor(**(fields | update))


_PROFILE = HostProfile(host_id="all-failed", host_sweepables=SHARED_CORE, measures=MeasureRegistry([_host_cost()]))

_CHECK = "state.answer.delivered == true"
_CHECK_RATE = goal_check_measure(_CHECK)
_CASES = [f"case-{index}" for index in range(4)]

#: The arms: every result a turn, every call refused, and half of each.
_STEADY, _REFUSING, _FLAKY = "steady", "refusing", "flaky"


# =============================================================================
# Results, one kind of ending each
# =============================================================================


def _result(model: str, case_id: str, *, ms: float, cost: float | None, **update: Any) -> EvalResult:
    """One result at ``ms`` whose candidate reported ``cost`` dollars (None: reported nothing), its check passed."""
    fields: dict[str, Any] = dict(
        scope_id=_SCOPE,
        test_case_id=case_id,
        model=model,
        variant_key=fixture_variant_key(model),
        latency=LatencyMetrics(total_ms=ms),
        usage=[] if cost is None else [RoleUsage(role="candidate", model=model, cost_usd=cost)],
        cost_usd=0.0 if cost is None else cost,
        host_measures={_TURN_COST: 0.0 if cost is None else cost},
        goal_state_outcomes=[GoalStateOutcome(expression=_CHECK, passed=True)],
        rubric_scores=[],
    )
    return make_eval_result(**(fields | update))


def _delivered(model: str, case_id: str, *, ms: float = 400.0, cost: float | None = 0.002) -> EvalResult:
    """A turn the candidate took and answered."""
    return _result(model, case_id, ms=ms, cost=cost)


def _refused(model: str, case_id: str, *, billed: bool = False, ms: float = 53.0) -> EvalResult:
    """A call the provider refused: a round trip, and a check that reads True on a turn nobody took.

    ``billed`` puts a reported $0.0001 on the refusal — spend that WAS observed, so only the turn rule, not
    the unobserved-spend rule, keeps it out of a cost reading. Unbilled, it reports nothing, and the host's
    sum of nothing is the $0 the shakedown saw.
    """
    return _result(
        model, case_id, ms=ms, cost=0.0001 if billed else None, candidate_error="the provider refused the request"
    )


def _budget_ended(model: str, case_id: str, *, ms: float = 60_000.0, cost: float = 0.20) -> EvalResult:
    """A turn the host's budget ended after a minute — a failure that took a turn, and spent on it."""
    return _result(model, case_id, ms=ms, cost=cost, covariates={TURN_BUDGET_ENDED_KEY: 1})


def _store(
    arms: Mapping[str, list[EvalResult]],
    *,
    design: CampaignDesign | None = None,
    cases: list[EvalTestCase] | None = None,
) -> tuple[EvalStorage, Any]:
    """A real store holding one run per arm, the cases when given, and a campaign over the runs."""
    storage = EvalStorage(InMemoryDocumentStore())
    for case in cases or []:
        storage.save_test_case(case)
    run_ids = []
    for model, results in arms.items():
        run = make_eval_run(
            id=f"run-{model}",
            scope_id=_SCOPE,
            candidate_model=model,
            k_runs=1,
            test_case_ids=sorted({result.test_case_id for result in results}),
            status="completed",
        )
        storage.save_eval_run(run)
        for result in results:
            storage.save_eval_result(result.model_copy(update={"eval_run_id": run.id}))
        run_ids.append(run.id)
    campaign = make_campaign(scope_id=_SCOPE, run_ids=run_ids, declared_design=design)
    storage.save_campaign(campaign)
    return storage, campaign


def _three_arms() -> dict[str, list[EvalResult]]:
    return {
        _STEADY: [_delivered(_STEADY, case) for case in _CASES],
        _REFUSING: [_refused(_REFUSING, case) for case in _CASES],
        _FLAKY: [
            *(_delivered(_FLAKY, case) for case in _CASES[:2]),
            *(_refused(_FLAKY, case, billed=True) for case in _CASES[2:]),
        ],
    }


def _bundle(
    arms: Mapping[str, list[EvalResult]] | None = None,
    *,
    design: CampaignDesign | None = None,
    profile: HostProfile = _PROFILE,
    cases: list[EvalTestCase] | None = None,
) -> AnalysisContextBundle:
    storage, campaign = _store(_three_arms() if arms is None else arms, design=design, cases=cases)
    return assemble_context_bundle(campaign, storage=storage, profile=profile)


def _controlled(control: str, **bars: tuple[float, str]) -> CampaignDesign:
    """A declaration naming ``control``, holding each named measure to a ``(threshold, direction)`` bar."""
    return minimal_declaration(control=fixture_variant_key(control)).model_copy(
        update={
            "bars": [
                BarOverride(measure_id=name, threshold=threshold, direction=direction)  # type: ignore[arg-type]
                for name, (threshold, direction) in bars.items()
            ]
        }
    )


def _cell(bundle: AnalysisContextBundle, model: str) -> CellFacts:
    return next(cell for cell in bundle.cell_measures if cell.variant_key == fixture_variant_key(model))


def _summaries(cell: CellFacts | StratumFacts) -> dict[str, MeasureSummary]:
    return {summary.name: summary for summary in cell.measures.measures}


def _comparison(bundle: AnalysisContextBundle, reading: str, model: str) -> Any:
    (found,) = [
        comparison
        for family in bundle.multiple_comparisons.families
        for comparison in family.comparisons
        if comparison.name == reading and comparison.contrast.variant_key == fixture_variant_key(model)
    ]
    return found


def _frontier_point(bundle: AnalysisContextBundle, model: str) -> Any:
    (point,) = [point for subject in bundle.frontier.subjects for point in subject.points if point.model == model]
    return point


def _code_only(bundle: AnalysisContextBundle) -> Report:
    return build_code_only_report(bundle, measures=_PROFILE.measures, assembled_at="2026-10-09T00:00:00+00:00")


def _table(report: Report, name: str) -> TableBlock:
    (table,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == name]
    return table


def _row(table: TableBlock, model: str, **match: str) -> dict[str, Any]:
    """The row of one arm — named, in this fixture, by its variant key's digest — matching ``match``."""
    digest = short_digest(fixture_variant_key(model))
    return next(
        row
        for row in table.rows
        if digest in str(row["arm"]) and all(row.get(key) == value for key, value in match.items())
    )


def _column(table: TableBlock, measure: str) -> str:
    """The key of the column a measure is shown under — its header leads with what a reader calls the measure."""
    return next(column.key for column in table.columns if column.header.startswith(measure))


def _all_failed_disclosures(report: Report) -> list[str]:
    return [
        block.text
        for block in report.blocks
        if isinstance(block, DisclosureBlock) and block.text.startswith("Every result failed")
    ]


def _label(row: dict[str, Any]) -> str:
    """A surface row's arm label, without the control marker the report adds."""
    return str(row["arm"]).removesuffix(" (control)")


# =============================================================================
# The predicate
# =============================================================================


class TestDeliveredATurn:
    def test_a_turn_answered_is_one(self) -> None:
        assert delivered_a_turn(_delivered("m", "c"))

    def test_a_refused_or_errored_call_took_none(self) -> None:
        assert not delivered_a_turn(_refused("m", "c"))

    def test_a_turn_the_budget_ended_or_the_cap_cut_took_one(self) -> None:
        assert delivered_a_turn(_budget_ended("m", "c"))
        assert delivered_a_turn(_result("m", "c", ms=900.0, cost=0.01, covariates={TRUNCATED_ROUNDS_KEY: 1}))

    def test_a_model_call_the_deadline_struck_took_one(self) -> None:
        struck = _refused("m", "c", ms=30_000.0).model_copy(update={"termination": "cell_timeout"})
        assert delivered_a_turn(struck)

    def test_a_faulted_result_is_none(self) -> None:
        assert not delivered_a_turn(_result("m", "c", ms=10.0, cost=None, infra_error="cassette miss"))

    def test_a_model_failure_after_delivered_turns_took_them(self) -> None:
        late = _refused("m", "c", ms=40_000.0).model_copy(update={"turns_delivered": 5})
        assert delivered_a_turn(late)

    def test_a_single_call_refused_counts_no_turn(self) -> None:
        assert not delivered_a_turn(_refused("m", "c").model_copy(update={"turns_delivered": 0}))

    def test_a_result_that_counted_no_turns_falls_back_to_its_cause(self) -> None:
        # Stored before the count was kept, or of a kind that counts nothing: the cause alone decides, as before.
        assert _refused("m", "c").turns_delivered is None
        assert not delivered_a_turn(_refused("m", "c"))
        assert delivered_a_turn(_delivered("m", "c"))


class TestTheRunnerCountsTheTurnsDelivered:
    def test_a_kind_s_own_count_wins(self) -> None:
        assert count_delivered_turns([{"turn_record": {}}], reported=3) == 3
        assert count_delivered_turns([], reported=0) == 0

    def test_otherwise_the_turn_records_its_trace_stamps(self) -> None:
        trace = [{"role": "user"}, {"turn_record": {}}, {"role": "user"}, {"turn_record": {}}]
        assert count_delivered_turns(trace, reported=None) == 2

    def test_with_neither_nothing_counted(self) -> None:
        assert count_delivered_turns([{"role": "user"}], reported=None) is None

    async def test_a_quick_call_stores_one_turn_answered_and_none_refused(self) -> None:
        comparison = await compare(
            _QUICK_CASES,
            {"control": _guess, "candidate": _refuse},
            expected=lambda case: str(case["label"]),
            control="control",
            scope_id="all-failed-turns",
            k=1,
        )
        bundle = inspect_campaign_bundle(comparison.host, comparison.campaign_id, "all-failed-turns").bundle
        stored = [
            result
            for run_id in bundle.run_ids
            for result in list_results(comparison.host.storage, run_id, "all-failed-turns")
        ]
        assert sorted({(result.candidate_error is None, result.turns_delivered) for result in stored}) == [
            (False, 0),
            (True, 1),
        ]


# =============================================================================
# Cost and latency are read over the turns taken; failures still count against the arm
# =============================================================================


class TestCostAndLatencyAreReadOverTurnsTaken:
    def test_a_mixed_arm_states_the_latency_and_spend_of_its_turns(self) -> None:
        cell = _cell(_bundle(), _FLAKY)
        summaries = _summaries(cell)

        assert (cell.n_candidate_failed, cell.n_no_turn) == (2, 2)
        assert not cell.all_failed
        # The 53 ms refusals are not turns: the latency is the two answered turns', not (2×400 + 2×53) / 4.
        latency = summaries["total_ms"]
        assert (latency.mean, latency.n, latency.population) == (400.0, 2, "delivered")
        # The billed refusals observed spend, and are still no turn's cost on the cost axis.
        turn = summaries[_TURN_COST]
        assert (turn.mean, turn.n, turn.population) == (0.002, 2, "delivered")
        # `cost_usd` is measuring spend, so every dollar billed counts — the refusals' too, as the pivot, the
        # history series and a run summary's program total read it.
        spend = summaries["cost_usd"]
        assert (spend.mean, spend.n, spend.population) == (
            pytest.approx((2 * 0.002 + 2 * 0.0001) / 4),
            4,
            "all_observed",
        )

    def test_a_cell_s_measuring_spend_is_the_run_summary_s_program_mean(self) -> None:
        """One population for `cost_usd` on every surface: the cell, the run summary, the pivot and history."""
        from threetears.evals.kernel.scoring import compute_cost_summary

        arms = _three_arms()
        spend = _summaries(_cell(_bundle(arms), _FLAKY))["cost_usd"]
        (program,) = compute_cost_summary(arms[_FLAKY]).values()
        assert (spend.mean, spend.n) == (pytest.approx(program["mean_cost_usd"]), program["n_cost_usd"])

    def test_the_failures_still_count_against_the_arm_in_its_pass_rate(self) -> None:
        rate = _summaries(_cell(_bundle(), _FLAKY))[_CHECK_RATE]
        # Every result's check read True; the two the candidate failed count as failed all the same.
        assert (rate.mean, rate.n, rate.population) == (0.5, 4, "scored")

    def test_an_arm_whose_every_call_was_answered_reads_as_before(self) -> None:
        cell = _cell(_bundle(), _STEADY)
        assert (cell.n_candidate_failed, cell.n_no_turn, cell.all_failed) == (0, 0, False)
        assert _summaries(cell)["total_ms"].n == 4

    def test_a_run_summary_reads_the_same_turns(self) -> None:
        bundle = _bundle()
        summaries = {summary.candidate_model: summary for summary in bundle.run_summaries}
        flaky = {measure.name: measure for measure in summaries[_FLAKY].measures.measures}
        assert (flaky["total_ms"].mean, flaky["total_ms"].n, flaky["total_ms"].population) == (400.0, 2, "delivered")
        assert summaries[_FLAKY].n_prod_cost_usd == 2
        refusing = {measure.name for measure in summaries[_REFUSING].measures.measures}
        assert not {"total_ms", "cost_usd", _TURN_COST} & refusing


# =============================================================================
# A failure that took a turn stays in cost and latency
# =============================================================================


def _budget_campaign() -> AnalysisContextBundle:
    """A control at ~1 s and $0.01, against an arm whose odd cases the budget ended at 60 s and $0.20.

    The arm's answered half is twice as fast and half as cheap as the control. Read over its answered turns
    alone it would beat the control on both; its budget-ended half took a minute and spent twenty times as
    much, and that is what the arm costs.
    """
    cases = [f"case-{index}" for index in range(8)]
    arms = {
        "control": [_delivered("control", case, ms=1000.0 + 10 * index, cost=0.01) for index, case in enumerate(cases)],
        _FLAKY: [
            _budget_ended(_FLAKY, case) if index % 2 else _delivered(_FLAKY, case, ms=500.0 + index, cost=0.005)
            for index, case in enumerate(cases)
        ],
    }
    return _bundle(arms, design=_controlled("control"))


class TestATurnTheBudgetEndedStaysInCostAndLatency:
    def test_the_cell_reads_every_turn_it_took(self) -> None:
        cell = _cell(_budget_campaign(), _FLAKY)
        summaries = _summaries(cell)
        assert (cell.n_candidate_failed, cell.n_no_turn, cell.all_failed) == (4, 0, False)
        assert summaries["total_ms"].n == 8 and summaries["total_ms"].mean is not None
        assert summaries["total_ms"].mean > 30_000
        assert summaries["cost_usd"].mean == pytest.approx((4 * 0.20 + 4 * 0.005) / 8)

    # The candidate's spend is contrasted as production_replicating_cost: cost_usd, which would also sum a
    # judge's spend, is on no merit axis and enters no family.
    @pytest.mark.parametrize("reading", ["total_ms", "production_replicating_cost", _TURN_COST])
    def test_it_never_reads_faster_or_cheaper_than_the_control(self, reading: str) -> None:
        comparison = _comparison(_budget_campaign(), reading, _FLAKY)
        assert comparison.verdict != "improved"
        assert comparison.delta is not None and comparison.delta > 0

    def test_the_frontier_ranks_it_on_its_minute_long_turns(self) -> None:
        bundle = _budget_campaign()
        flaky, control = _frontier_point(bundle, _FLAKY), _frontier_point(bundle, "control")
        assert flaky.mean_total_ms > control.mean_total_ms
        assert flaky.n_latency == 8 and flaky.n_no_turn == 0
        assert not control.dominated_by


def _late_failure_campaign() -> AnalysisContextBundle:
    """A control at ~15 s and $0.15, against an arm whose odd conversations failed on their LAST turn.

    The provider failed the sixth turn after five delivered ones, 40 s and $0.40 in; the arm's other
    conversations took 10 s and $0.10. Read over its clean conversations alone it beats the control on time
    and spend; with the late failures' delivered turns, which the runner counted, it does not.
    """
    cases = [f"case-{index:02d}" for index in range(12)]
    control = [_delivered("control", case, ms=15_000.0 + 10 * index, cost=0.15) for index, case in enumerate(cases)]
    late = [
        _result(
            "late",
            case,
            ms=40_000.0 + 10 * index,
            cost=0.40,
            candidate_error="provider 503 on turn 6",
            turns_delivered=5,
            goal_state_outcomes=[GoalStateOutcome(expression=_CHECK, passed=False)],
        )
        if index % 2
        else _delivered("late", case, ms=10_000.0 + 10 * index, cost=0.10)
        for index, case in enumerate(cases)
    ]
    return _bundle({"control": control, "late": late}, design=_controlled("control"))


class TestAModelFailureAfterDeliveredTurnsStaysInCostAndLatency:
    def test_the_cell_reads_every_conversation(self) -> None:
        cell = _cell(_late_failure_campaign(), "late")
        assert (cell.n_candidate_failed, cell.n_no_turn, cell.all_failed) == (6, 0, False)
        latency = _summaries(cell)["total_ms"]
        assert latency.n == 12 and latency.mean is not None and latency.mean > 15_000

    @pytest.mark.parametrize("reading", ["total_ms", "production_replicating_cost"])
    def test_it_never_reads_faster_or_cheaper_than_the_control(self, reading: str) -> None:
        comparison = _comparison(_late_failure_campaign(), reading, "late")
        assert comparison.verdict != "improved"
        assert comparison.delta is not None and comparison.delta > 0

    def test_the_control_dominates_it_on_the_frontier(self) -> None:
        bundle = _late_failure_campaign()
        late = _frontier_point(bundle, "late")
        assert late.n_latency == 12 and late.n_no_turn == 0
        assert [dominator.model for dominator in late.dominated_by] == ["control"]

    def test_the_frozen_surface_carries_the_lens_standing_a_frontier_chart_draws(self) -> None:
        """Copied off the lens, so the chart's dominated mark and the frontier table cannot disagree."""
        bundle = _late_failure_campaign()
        standings = bundle_decision_surface(bundle).frontier_dominance
        points = [point for subject in bundle.frontier.subjects for point in subject.points]
        assert standings == {point.variant_key: point.dominance for point in points}
        assert standings[_frontier_point(bundle, "late").variant_key] == "dominated"


# =============================================================================
# An arm where no result took a turn
# =============================================================================


class TestAnArmWhereNoResultTookATurn:
    def test_it_counts_every_result_as_failed_and_carries_no_cost_or_latency(self) -> None:
        cell = _cell(_bundle(), _REFUSING)
        summaries = _summaries(cell)

        assert cell.n_candidate_failed == cell.n_no_turn == cell.n_observations == 4
        assert cell.all_failed
        assert not {"total_ms", "cost_usd", _TURN_COST} & set(summaries), "no 53 ms, no $0"
        assert (summaries[_CHECK_RATE].mean, summaries[_CHECK_RATE].n) == (0.0, 4)

    def test_the_bundle_names_it_with_the_sentence_to_quote(self) -> None:
        bundle = _bundle()
        refusing = _cell(bundle, _REFUSING)
        assert bundle.all_failed_cells == [
            CellCoordinate(variant_key=refusing.variant_key, apparatus_class_id=refusing.apparatus_class_id)
        ]
        assert bundle.all_failed is not None
        assert bundle.all_failed.startswith("Every result failed in 1 of 3 cells")
        assert "no cost or latency to read" in bundle.all_failed
        # Its cost is absent because every call failed, not because nobody reported spend.
        assert bundle.cost_unmeasured_cells == [] and bundle.cost_unmeasured is None

    def test_the_code_only_report_says_so_in_its_surface_and_beside_it(self) -> None:
        bundle = _bundle()
        report = _code_only(bundle)
        table = _table(report, "surface")
        refusing, steady = _row(table, _REFUSING), _row(table, _STEADY)

        for measure in ("Turn time", "Turn cost"):
            assert refusing[_column(table, measure)] == NO_SUCCESSFUL_RESULTS
            assert steady[_column(table, measure)] not in (None, NO_SUCCESSFUL_RESULTS)
        assert "4 failed by the candidate (4 with no turn taken" in str(refusing["replication"])
        assert "2 failed by the candidate (2 with no turn taken" in str(_row(table, _FLAKY)["replication"])
        assert "failed" not in str(steady["replication"])

        (said,) = _all_failed_disclosures(report)
        assert said.startswith(bundle.all_failed or "")
        assert said.endswith(f"Every result failed in: {_label(refusing)}.")
        jsonschema.Draft202012Validator(published_report_schema()).validate(json.loads(report.to_canonical_json()))

    def test_a_stored_analysis_says_so_from_its_frozen_surface_alone(self) -> None:
        bundle = _bundle()
        analysis = make_analysis(decision_surface=bundle_decision_surface(bundle), variant_index=bundle.variant_index)
        storage = EvalStorage(InMemoryDocumentStore())
        storage.save_analysis(analysis)
        loaded = storage.load_analysis(analysis.id, analysis.scope_id)
        assert loaded is not None

        report = build_report(loaded)
        table = _table(report, "surface")
        assert _row(table, _REFUSING)[_column(table, "Turn time")] == NO_SUCCESSFUL_RESULTS
        assert len(_all_failed_disclosures(report)) == 1
        jsonschema.Draft202012Validator(published_report_schema()).validate(json.loads(report.to_canonical_json()))

    def test_no_disclosure_when_every_arm_took_a_turn(self) -> None:
        arms = _three_arms()
        del arms[_REFUSING]
        bundle = _bundle(arms)
        assert bundle.all_failed_cells == [] and bundle.all_failed is None
        assert _all_failed_disclosures(_code_only(bundle)) == []

    def test_a_table_where_every_cell_failed_names_no_arm(self) -> None:
        arms = {model: [_refused(model, case) for case in _CASES] for model in (_REFUSING, "also-refusing")}
        bundle = _bundle(arms)
        table = surface_table_of(bundle_decision_surface(bundle), bundle.variant_index)
        assert all(row.all_failed for row in table.rows)
        assert table.all_failed_disclosure == bundle.all_failed
        assert (table.all_failed_disclosure or "").startswith("Every result failed: ")
        assert "Every result failed in:" not in (table.all_failed_disclosure or "")

    def test_a_latency_bar_reads_no_successful_results_not_a_blank(self) -> None:
        bundle = _bundle(design=_controlled(_STEADY, total_ms=(1000.0, "lower_is_better")))
        (bar,) = [bar for bar in bundle.bar_adjudications if bar.measure_id == "total_ms"]
        verdict = next(v for v in bar.verdicts if v.variant_key == fixture_variant_key(_REFUSING))
        assert (verdict.value, verdict.n, verdict.cleared) == (None, 0, None)

        table = surface_table_of(bundle_decision_surface(bundle), bundle.variant_index)
        bar_index = next(i for i, column in enumerate(table.columns) if column.kind == "bar")
        refusing = next(row for row in table.rows if row.variant_key == fixture_variant_key(_REFUSING))
        value = refusing.values[bar_index]
        assert value is not None
        assert (value.text, value.verdict_word) == (NO_SUCCESSFUL_RESULTS, "no data")

    def test_a_latency_bar_no_cell_could_read_says_every_call_failed(self) -> None:
        arms = {model: [_refused(model, case) for case in _CASES] for model in (_REFUSING, "also-refusing")}
        bundle = _bundle(arms, design=_controlled(_REFUSING, total_ms=(1000.0, "lower_is_better")))
        (bar,) = [bar for bar in bundle.bar_adjudications if bar.measure_id == "total_ms"]
        assert bar.state == "names_no_stored_measure"
        assert "no result took a turn" in (bar.reason or "")

    def test_its_latency_contrast_says_every_result_failed(self) -> None:
        comparison = _comparison(_bundle(design=_controlled(_STEADY)), "total_ms", _REFUSING)
        assert comparison.verdict == "untested"
        assert comparison.untested_reason == (
            "every result of the arm failed with no turn taken, so there is no turn's time or spend to compare"
        )

    def test_the_frontier_cost_leaves_out_a_billed_refusal(self) -> None:
        flaky = _frontier_point(_bundle(), _FLAKY)
        # The two refusals were billed $0.0001 each; they took no turn, so the mean is the answered turns'.
        assert (flaky.production_replicating_cost, flaky.n_cost) == (pytest.approx(0.002), 2)

    def test_it_dominates_nothing_on_the_frontier(self) -> None:
        bundle = _bundle()
        refusing = _frontier_point(bundle, _REFUSING)
        assert (refusing.mean_total_ms, refusing.n_latency, refusing.n_no_turn) == (None, 0, 4)
        assert refusing.production_replicating_cost is None
        refusing_key = fixture_variant_key(_REFUSING)
        for subject in bundle.frontier.subjects:
            for point in subject.points:
                assert all(dominator.variant_key != refusing_key for dominator in point.dominated_by)

    def test_its_stratum_reads_no_successful_results_beside_the_strata_that_answered(self) -> None:
        cases = [
            EvalTestCase(id=case, scope_id=_SCOPE, template_id="tpl-1", stratum="hard" if index < 2 else "easy")
            for index, case in enumerate(_CASES)
        ]
        # Answers its easy cases and is refused on every hard one.
        arms = {
            _FLAKY: [
                _refused(_FLAKY, case) if index < 2 else _delivered(_FLAKY, case) for index, case in enumerate(_CASES)
            ]
        }
        bundle = _bundle(arms, cases=cases)
        strata = {stratum.stratum: stratum for stratum in _cell(bundle, _FLAKY).strata}
        assert strata["hard"].all_failed and not strata["easy"].all_failed
        assert (strata["hard"].n_candidate_failed, strata["hard"].n_no_turn) == (2, 2)

        table = _table(_code_only(bundle), "strata")
        latency = _row(table, _FLAKY, reading="Turn time (ms)")
        assert latency[_column(table, "hard")] == NO_SUCCESSFUL_RESULTS
        assert latency[_column(table, "easy")] not in (None, NO_SUCCESSFUL_RESULTS)


# =============================================================================
# A classifier's refusal is a miss
# =============================================================================


def _classified(model: str, case_id: str, index: int, *, matched: bool) -> EvalResult:
    return _result(model, case_id, ms=800.0 + index, cost=None, host_measures={"match": matched})


def _classifier_campaign(*, half: bool) -> AnalysisContextBundle:
    """A control matching three cases in four, against an arm refused on every case or every other one.

    A host's kind that lands no ``match`` on a refusal: the refusals carry none. The half-refusing arm matches
    every case it answers, so read over its answers alone it is more accurate than the control.
    """
    cases = [f"case-{index:02d}" for index in range(24)]
    control = [_classified("control", case, index, matched=index % 4 != 0) for index, case in enumerate(cases)]
    refusing = [
        _classified(_REFUSING, case, index, matched=True) if half and index % 2 == 0 else _refused(_REFUSING, case)
        for index, case in enumerate(cases)
    ]
    return _bundle({"control": control, _REFUSING: refusing}, design=_controlled("control"))


class TestAClassifiersRefusalIsAMiss:
    def test_an_all_refusing_arm_reads_accuracy_zero_and_is_tested_against_the_control(self) -> None:
        bundle = _classifier_campaign(half=False)
        accuracy = _summaries(_cell(bundle, _REFUSING))["accuracy"]
        assert (accuracy.mean, accuracy.n) == (0.0, 24)
        comparison = _comparison(bundle, "accuracy", _REFUSING)
        assert comparison.p_adjusted is not None
        assert comparison.verdict == "regressed"

    def test_a_half_refusing_arm_does_not_read_more_accurate_for_refusing(self) -> None:
        bundle = _classifier_campaign(half=True)
        accuracy = _summaries(_cell(bundle, _REFUSING))["accuracy"]
        assert (accuracy.mean, accuracy.n) == (0.5, 24)
        comparison = _comparison(bundle, "accuracy", _REFUSING)
        assert comparison.verdict != "improved"
        assert comparison.delta is not None and comparison.delta < 0

    def test_a_kind_that_classifies_nothing_gains_no_accuracy(self) -> None:
        assert "accuracy" not in _summaries(_cell(_bundle(), _REFUSING))

    def test_a_run_grading_by_something_else_beside_a_classifier_gains_no_accuracy(self) -> None:
        # One callable kind, two runs over the same cases: one classifies, the other only scores — and two of its
        # calls raised. Its answers carry no verdict on cases the classifier classified, so its failures are no
        # misses of a classification it never made.
        cases = [f"case-{index}" for index in range(8)]
        arms = {
            "classifier": [
                _classified("classifier", case, index, matched=index % 4 != 0) for index, case in enumerate(cases)
            ],
            "scorer": [
                _refused("scorer", case) if index < 2 else _result("scorer", case, ms=800.0, cost=None)
                for index, case in enumerate(cases)
            ],
        }
        bundle = _bundle(arms, design=_controlled("classifier"))
        assert "accuracy" not in _summaries(_cell(bundle, "scorer"))
        assert _summaries(_cell(bundle, "classifier"))["accuracy"].n == 8

    def test_a_failure_on_a_case_nothing_classified_is_no_miss(self) -> None:
        # One arm: four cases classified, a refusal on one of them, two cases graded otherwise, and a refusal on
        # a case nothing classified. Only the refusal on a classified case is a miss.
        classified = [f"case-{index}" for index in range(4)]
        results = [
            *(_classified("one", case, index, matched=True) for index, case in enumerate(classified)),
            _refused("one", classified[1]).model_copy(update={"k_iteration": 2}),
            *(_result("one", case, ms=800.0, cost=None) for case in ("scored-0", "scored-1")),
            _refused("one", "unclassified"),
        ]
        accuracy = _summaries(_cell(_bundle({"one": results}), "one"))["accuracy"]
        assert (accuracy.mean, accuracy.n) == (0.8, 5)


_QUICK_SCOPE = "all-failed-quick"
_QUICK_CASES = [
    {"text": "a cat", "label": "animal"},
    {"text": "a dog", "label": "animal"},
    {"text": "a fir", "label": "plant"},
    {"text": "an oak", "label": "plant"},
]


async def _guess(case: Mapping[str, Any]) -> str:
    return "plant"


async def _refuse(case: Mapping[str, Any]) -> str:
    raise RuntimeError("refused")


class TestAQuickClassifierLandsItsRefusalAsAMiss:
    async def _comparison(self) -> Comparison:
        return await compare(
            _QUICK_CASES,
            {"control": _guess, "candidate": _refuse},
            expected=lambda case: str(case["label"]),
            control="control",
            scope_id=_QUICK_SCOPE,
            k=2,
        )

    async def test_the_refusing_arm_has_an_accuracy_and_its_refusals_are_unusable_answers(self) -> None:
        comparison = await self._comparison()
        bundle = inspect_campaign_bundle(comparison.host, comparison.campaign_id, _QUICK_SCOPE).bundle
        refusing = next(cell for cell in bundle.cell_measures if cell.n_candidate_failed)
        summaries = _summaries(refusing)
        assert (summaries["accuracy"].mean, summaries["accuracy"].n) == (0.0, 8)
        assert summaries["confusion_cell"].categories == {
            f"animal → {UNUSABLE_ANSWER}": 4,
            f"plant → {UNUSABLE_ANSWER}": 4,
        }

    async def test_its_accuracy_contrast_is_tested_not_untested(self) -> None:
        comparison = await self._comparison()
        (row,) = [row for row in comparison.contrasts() if row["reading"] == "Accuracy"]
        assert row["p_adjusted"] is not None and row["delta"] == -0.5
        assert not str(row["verdict"]).startswith("untested")


# =============================================================================
# Unmeasured cost is read over the turns taken
# =============================================================================


class TestUnmeasuredCostIsReadOverTurnsTaken:
    def test_a_billed_refusal_does_not_make_unpriced_turns_look_measured(self) -> None:
        arms = {
            _STEADY: [_delivered(_STEADY, case) for case in _CASES],
            # Its answered turns reported no spend; its refusals were billed.
            _FLAKY: [
                *(_delivered(_FLAKY, case, cost=None) for case in _CASES[:2]),
                *(_refused(_FLAKY, case, billed=True) for case in _CASES[2:]),
            ],
        }
        bundle = _bundle(arms)
        flaky = _cell(bundle, _FLAKY)
        assert [cell.variant_key for cell in bundle.cost_unmeasured_cells] == [flaky.variant_key]
        # Measuring spend states what was billed and over how many: the two refusals, never the unpriced turns.
        spend = _summaries(flaky)["cost_usd"]
        assert (spend.mean, spend.n) == (pytest.approx(0.0001), 2)


# =============================================================================
# A run's summary
# =============================================================================


class TestARunSummaryReadsCostAndLatencyTheSameWay:
    def _summary(self, model: str, arms: Mapping[str, list[EvalResult]] | None = None) -> Any:
        storage, _ = _store(_three_arms() if arms is None else arms)
        host = dataclasses.replace(callable_host(), profile=_PROFILE, storage=storage)
        return summarize_run(host, f"run-{model}", _SCOPE)

    def test_a_run_where_no_result_took_a_turn_has_no_successful_results_to_average(self) -> None:
        summary = self._summary(_REFUSING)
        (cost,) = [measure for measure in summary.measures if measure.name == _TURN_COST]
        assert (cost.n, cost.mean, cost.n_no_turn, cost.n_faulted) == (0, None, 4, 0)
        assert (
            f"  {_TURN_COST}: {NO_SUCCESSFUL_RESULTS}, left out: 4 refused or errored with no turn taken"
            in summary.render()
        )

    def test_a_mixed_run_averages_its_turns_and_says_what_it_left_out(self) -> None:
        summary = self._summary(_FLAKY)
        (cost,) = [measure for measure in summary.measures if measure.name == _TURN_COST]
        assert (cost.n, cost.mean, cost.n_no_turn) == (2, 0.002, 2)
        assert "left out: 2 refused or errored with no turn taken" in summary.render()

    def test_a_run_whose_every_result_was_faulted_says_so_not_that_it_failed(self) -> None:
        faulted = [_result("rigged", case, ms=10.0, cost=None, infra_error="cassette miss") for case in _CASES]
        summary = self._summary("rigged", {"rigged": faulted})
        (cost,) = [measure for measure in summary.measures if measure.name == _TURN_COST]
        assert (cost.n, cost.n_no_turn, cost.n_faulted) == (0, 0, 4)
        rendered = summary.render()
        assert f"{_TURN_COST}: every result carrying it was excluded, left out: 4 excluded as a fault" in rendered
        assert NO_SUCCESSFUL_RESULTS not in rendered


# =============================================================================
# The stored shape
# =============================================================================


def _facts(**update: Any) -> CellFacts:
    fields: dict[str, Any] = dict(variant_key="v", apparatus_class_id="r", run_ids=["run"], n_observations=4, n_cases=4)
    return CellFacts(**(fields | update))


class TestTheShapeRefusesWhatCannotBeTrue:
    def test_more_failed_and_faulted_than_observed_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="faulted, failed or neither"):
            _facts(n_infra_excluded=1, n_candidate_failed=4, n_no_turn=0)

    def test_more_no_turn_failures_than_failures_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="subset"):
            _facts(n_candidate_failed=1, n_no_turn=2)

    def test_one_count_without_the_other_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="kept together"):
            _facts(n_candidate_failed=1)

    def test_failures_the_strata_do_not_add_up_to_are_refused(self) -> None:
        strata = [
            StratumFacts(stratum="a", n_observations=2, n_cases=2, n_candidate_failed=1, n_no_turn=1),
            StratumFacts(stratum="b", n_observations=2, n_cases=2, n_candidate_failed=0, n_no_turn=0),
        ]
        assert _facts(strata=strata, n_candidate_failed=1, n_no_turn=1).n_no_turn == 1
        with pytest.raises(ValidationError, match="every one of its failed observations"):
            _facts(strata=strata, n_candidate_failed=2, n_no_turn=1)
        with pytest.raises(ValidationError, match="every one of its no-turn observations"):
            _facts(strata=strata, n_candidate_failed=1, n_no_turn=0)

    def test_all_failed_is_no_counted_result_taking_a_turn(self) -> None:
        assert _facts(n_candidate_failed=3, n_no_turn=3, n_infra_excluded=1).all_failed
        assert not _facts(n_candidate_failed=0, n_no_turn=0, n_infra_excluded=4).all_failed
        # Every result failed, but the budget ended turns that ran: they have a cost and latency to state.
        assert not _facts(n_candidate_failed=4, n_no_turn=0).all_failed

    def test_an_analysis_frozen_before_the_counts_were_kept_reads_as_not_counted(self) -> None:
        stored = _facts().model_dump(mode="json")
        del stored["n_candidate_failed"], stored["n_no_turn"]
        legacy = CellFacts.model_validate(stored)
        assert (legacy.n_candidate_failed, legacy.n_no_turn, legacy.all_failed) == (None, None, False)


class TestDeliveredIsDeclaredOnlyOnATurnsTimeOrSpend:
    def test_a_cost_or_latency_measure_may_declare_it(self) -> None:
        assert _host_cost(population="delivered").population == "delivered"
        assert _host_cost(merit_axis="latency", unit="ms", population="delivered").population == "delivered"

    def test_a_quality_measure_may_not(self) -> None:
        with pytest.raises(ValidationError, match="for a turn's time or spend"):
            _host_cost(merit_axis="quality", unit=None, higher_is_better=True, population="delivered")

    def test_a_measure_on_no_axis_may_not(self) -> None:
        with pytest.raises(ValidationError, match="for a turn's time or spend"):
            _host_cost(merit_axis=None, population="delivered")

    def test_an_explicit_all_observed_cost_keeps_every_raw_row(self) -> None:
        profile = dataclasses.replace(_PROFILE, measures=MeasureRegistry([_host_cost(population="all_observed")]))
        cost = _summaries(_cell(_bundle(profile=profile), _FLAKY))[_TURN_COST]
        assert (cost.n, cost.population) == (4, "all_observed")
