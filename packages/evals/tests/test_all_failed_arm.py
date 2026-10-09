"""An arm whose every result failed reads as failed — never as the fastest, cheapest arm on the surface.

A classifier arm whose every candidate call was refused showed 53 ms and $0 in the decision surface, and
nothing said every result had failed. Refusals are candidate failures, and they stay in the cell — every
rate and bar counts them against the arm — but their round trip and their empty spend were averaged into
the cell's latency and cost as if each were a delivered turn.

Pinned here, through a real store, the bundle, both reports and a run's summary:

- **A cost or latency measure is read over the results the candidate delivered** (population
  ``delivered``): an arm that delivered half its results states the latency and spend of that half, while
  its goal-check rate still counts the failed half as failures.
- **An arm that delivered nothing carries no cost or latency reading**, counts its failures
  (``n_candidate_failed``), reads ``no successful results`` under the surface's cost and latency columns,
  and is named in a disclosure — on the code-only report, on an analysis's report, and in the bundle the
  analysis is written from.
- **A run's summary reads a cost or latency measure the same way**, and says a run that delivered nothing
  has no successful results to average.
- **The stored shape refuses what cannot be true**, and an analysis frozen before the count was kept reads
  as not counted rather than as nothing failed.

Mutations that turn this file red: reading ``scored`` for a cost or latency measure in
``summary_population``; letting a candidate failure into a ``delivered`` population in the measure walk;
dropping ``n_candidate_failed`` from ``_cell_measures``; leaving the blank under an all-failed cell's merit
column; not emitting the table's all-failed disclosure in ``_surface_blocks``.
"""

from __future__ import annotations

import dataclasses
import json
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
    published_report_schema,
)
from threetears.evals.analysis.arms import short_digest
from threetears.evals.analysis.bundle import CellCoordinate, bundle_decision_surface
from threetears.evals.analysis.surface_table import NO_SUCCESSFUL_RESULTS
from threetears.evals.contracts import (
    CellFacts,
    EvalResult,
    EvalStorage,
    GoalStateOutcome,
    LatencyMetrics,
    RoleUsage,
    StratumFacts,
)
from threetears.evals.contracts.analysis_measures import MeasureSummary
from threetears.evals.contracts.host import SHARED_CORE, HostProfile, MeasureRegistry
from threetears.evals.contracts.metrics import MetricDescriptor, goal_check_measure
from threetears.evals.quick import callable_host, summarize_run
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import (
    fixture_variant_key,
    make_analysis,
    make_campaign,
    make_eval_result,
    make_eval_run,
)

_SCOPE = "uni-1"

#: A host's own spend measure on the cost axis, as a host computes it from what its calls reported — the
#: shape that read $0 for a refusing arm, since a refusal reports nothing and the host's sum of nothing is 0.
_TURN_COST = "turn_cost_usd"
_PROFILE = HostProfile(
    host_id="all-failed",
    host_sweepables=SHARED_CORE,
    measures=MeasureRegistry(
        [
            MetricDescriptor(
                name=_TURN_COST,
                data_type="numeric",
                family="mechanical",
                transferability_class="mechanical",
                attribution_scope="end_to_end",
                description="What the candidate's calls for one turn cost, as the host summed them.",
                higher_is_better=False,
                unit="usd",
                merit_axis="cost",
            )
        ]
    ),
)

_CHECK = "state.answer.delivered == true"
_CHECK_RATE = goal_check_measure(_CHECK)
_CASES = [f"case-{index}" for index in range(4)]

#: The three arms: every result delivered, every result refused, and half of each.
_STEADY, _REFUSING, _FLAKY = "steady", "refusing", "flaky"


def _delivered(model: str, case_id: str) -> EvalResult:
    """A turn the candidate delivered: 400 ms, $0.002 reported, its check passed."""
    return make_eval_result(
        scope_id=_SCOPE,
        test_case_id=case_id,
        model=model,
        variant_key=fixture_variant_key(model),
        latency=LatencyMetrics(total_ms=400.0),
        usage=[RoleUsage(role="candidate", model=model, cost_usd=0.002)],
        cost_usd=0.002,
        host_measures={_TURN_COST: 0.002},
        goal_state_outcomes=[GoalStateOutcome(expression=_CHECK, passed=True)],
        rubric_scores=[],
    )


def _refused(model: str, case_id: str, *, billed: bool) -> EvalResult:
    """A call the provider refused: a 53 ms round trip, and a check that reads True on a turn nobody took.

    ``billed`` puts a reported $0.0001 on the refusal — spend that WAS observed, so only the delivered
    rule, not the unobserved-spend rule, keeps it out of a cost reading. Unbilled, it reports nothing, and
    the host's sum of nothing is the $0 the shakedown saw.
    """
    usage = [RoleUsage(role="candidate", model=model, cost_usd=0.0001)] if billed else []
    return make_eval_result(
        scope_id=_SCOPE,
        test_case_id=case_id,
        model=model,
        variant_key=fixture_variant_key(model),
        candidate_error="the provider refused the request",
        latency=LatencyMetrics(total_ms=53.0),
        usage=usage,
        cost_usd=0.0001 if billed else 0.0,
        host_measures={_TURN_COST: 0.0001 if billed else 0.0},
        goal_state_outcomes=[GoalStateOutcome(expression=_CHECK, passed=True)],
        rubric_scores=[],
    )


def _store() -> tuple[EvalStorage, Any]:
    """A real store holding one run per arm, four cases each, and a campaign over the three runs."""
    arms = {
        _STEADY: [_delivered(_STEADY, case) for case in _CASES],
        _REFUSING: [_refused(_REFUSING, case, billed=False) for case in _CASES],
        _FLAKY: [
            *(_delivered(_FLAKY, case) for case in _CASES[:2]),
            *(_refused(_FLAKY, case, billed=True) for case in _CASES[2:]),
        ],
    }
    storage = EvalStorage(InMemoryDocumentStore())
    run_ids = []
    for model, results in arms.items():
        run = make_eval_run(
            id=f"run-{model}",
            scope_id=_SCOPE,
            candidate_model=model,
            k_runs=1,
            test_case_ids=list(_CASES),
            status="completed",
        )
        storage.save_eval_run(run)
        for result in results:
            storage.save_eval_result(result.model_copy(update={"eval_run_id": run.id}))
        run_ids.append(run.id)
    campaign = make_campaign(scope_id=_SCOPE, run_ids=run_ids)
    storage.save_campaign(campaign)
    return storage, campaign


def _bundle() -> AnalysisContextBundle:
    storage, campaign = _store()
    return assemble_context_bundle(campaign, storage=storage, profile=_PROFILE)


def _cell(bundle: AnalysisContextBundle, model: str) -> CellFacts:
    return next(cell for cell in bundle.cell_measures if cell.variant_key == fixture_variant_key(model))


def _summaries(cell: CellFacts) -> dict[str, MeasureSummary]:
    return {summary.name: summary for summary in cell.measures.measures}


def _surface_table(report: Report) -> TableBlock:
    (table,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == "surface"]
    return table


def _row(table: TableBlock, model: str) -> dict[str, Any]:
    """The surface row of one arm — named, in this fixture, by its variant key's digest."""
    digest = short_digest(fixture_variant_key(model))
    return next(row for row in table.rows if digest in str(row["arm"]))


def _column(table: TableBlock, measure: str) -> str:
    """The key of the merit column a measure is shown under — its header leads with the measure's name."""
    return next(column.key for column in table.columns if column.header.startswith(measure))


def _all_failed_disclosures(report: Report) -> list[str]:
    return [
        block.text
        for block in report.blocks
        if isinstance(block, DisclosureBlock) and block.text.startswith("Every result failed")
    ]


# =============================================================================
# The bundle: cost and latency over what was delivered
# =============================================================================


class TestCostAndLatencyAreReadOverDeliveredResults:
    def test_a_mixed_arm_states_the_latency_and_spend_of_what_it_delivered(self) -> None:
        cell = _cell(_bundle(), _FLAKY)
        summaries = _summaries(cell)

        assert cell.n_candidate_failed == 2
        assert not cell.all_failed
        # The 53 ms refusals are not turns: the latency is the two delivered turns', not (2×400 + 2×53) / 4.
        latency = summaries["total_ms"]
        assert (latency.mean, latency.n, latency.population) == (400.0, 2, "delivered")
        # The billed refusals observed spend, and are still no delivered result's cost.
        for spend in ("cost_usd", _TURN_COST):
            assert (summaries[spend].mean, summaries[spend].n, summaries[spend].population) == (
                0.002,
                2,
                "delivered",
            )

    def test_the_failures_still_count_against_the_arm_in_its_pass_rate(self) -> None:
        rate = _summaries(_cell(_bundle(), _FLAKY))[_CHECK_RATE]
        # Every result's check read True; the two the candidate failed count as failed all the same.
        assert (rate.mean, rate.n, rate.population) == (0.5, 4, "scored")

    def test_an_arm_that_delivered_everything_reads_as_before(self) -> None:
        cell = _cell(_bundle(), _STEADY)
        assert cell.n_candidate_failed == 0 and not cell.all_failed
        assert _summaries(cell)["total_ms"].n == 4


# =============================================================================
# An arm whose every result failed
# =============================================================================


class TestAnArmWhoseEveryResultFailed:
    def test_it_counts_every_result_as_failed_and_carries_no_cost_or_latency(self) -> None:
        cell = _cell(_bundle(), _REFUSING)
        summaries = _summaries(cell)

        assert cell.n_candidate_failed == cell.n_observations == 4
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
        # Its cost is absent because every result failed, not because nobody reported spend.
        assert bundle.cost_unmeasured_cells == [] and bundle.cost_unmeasured is None

    def test_the_code_only_report_says_so_in_its_surface_and_beside_it(self) -> None:
        bundle = _bundle()
        report = build_code_only_report(bundle, measures=_PROFILE.measures, assembled_at="2026-10-09T00:00:00+00:00")
        table = _surface_table(report)
        refusing, steady = _row(table, _REFUSING), _row(table, _STEADY)

        for measure in ("total_ms", _TURN_COST):
            assert refusing[_column(table, measure)] == NO_SUCCESSFUL_RESULTS
            assert steady[_column(table, measure)] not in (None, NO_SUCCESSFUL_RESULTS)
        assert "4 failed by the candidate" in str(refusing["replication"])
        assert "2 failed by the candidate" in str(_row(table, _FLAKY)["replication"])
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
        table = _surface_table(report)
        assert _row(table, _REFUSING)[_column(table, "total_ms")] == NO_SUCCESSFUL_RESULTS
        assert len(_all_failed_disclosures(report)) == 1
        jsonschema.Draft202012Validator(published_report_schema()).validate(json.loads(report.to_canonical_json()))

    def test_no_disclosure_when_every_arm_delivered_something(self) -> None:
        storage, campaign = _store()
        steady_and_flaky = campaign.model_copy(update={"run_ids": [f"run-{_STEADY}", f"run-{_FLAKY}"]})
        bundle = assemble_context_bundle(steady_and_flaky, storage=storage, profile=_PROFILE)
        report = build_code_only_report(bundle, measures=_PROFILE.measures, assembled_at="2026-10-09T00:00:00+00:00")
        assert bundle.all_failed_cells == [] and bundle.all_failed is None
        assert _all_failed_disclosures(report) == []


def _label(row: dict[str, Any]) -> str:
    """A surface row's arm label, without the control marker the report adds."""
    return str(row["arm"]).removesuffix(" (control)")


# =============================================================================
# A run's summary
# =============================================================================


class TestARunSummaryReadsCostAndLatencyTheSameWay:
    def _summary(self, model: str) -> Any:
        storage, _ = _store()
        host = dataclasses.replace(callable_host(), profile=_PROFILE, storage=storage)
        return summarize_run(host, f"run-{model}", _SCOPE)

    def test_a_run_that_delivered_nothing_has_no_successful_results_to_average(self) -> None:
        summary = self._summary(_REFUSING)
        (cost,) = [measure for measure in summary.measures if measure.name == _TURN_COST]
        assert (cost.n, cost.mean, cost.n_undelivered) == (0, None, 4)
        assert f"  {_TURN_COST}: {NO_SUCCESSFUL_RESULTS}, 4 undelivered result(s) left out" in summary.render()

    def test_a_mixed_run_averages_what_it_delivered_and_says_what_it_left_out(self) -> None:
        summary = self._summary(_FLAKY)
        (cost,) = [measure for measure in summary.measures if measure.name == _TURN_COST]
        assert (cost.n, cost.mean, cost.n_undelivered) == (2, 0.002, 2)
        assert "2 undelivered result(s) left out" in summary.render()


# =============================================================================
# The stored shape
# =============================================================================


def _facts(**update: Any) -> CellFacts:
    fields: dict[str, Any] = dict(variant_key="v", apparatus_class_id="r", run_ids=["run"], n_observations=4, n_cases=4)
    return CellFacts(**(fields | update))


class TestTheShapeRefusesWhatCannotBeTrue:
    def test_more_failed_and_faulted_than_observed_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="faulted, failed or delivered"):
            _facts(n_infra_excluded=1, n_candidate_failed=4)

    def test_failures_the_strata_do_not_add_up_to_are_refused(self) -> None:
        strata = [
            StratumFacts(stratum="a", n_observations=2, n_cases=2, n_candidate_failed=1),
            StratumFacts(stratum="b", n_observations=2, n_cases=2, n_candidate_failed=0),
        ]
        assert _facts(strata=strata, n_candidate_failed=1).n_candidate_failed == 1
        with pytest.raises(ValidationError, match="every one of its failed observations"):
            _facts(strata=strata, n_candidate_failed=2)

    def test_every_counted_result_failing_is_all_failed_and_none_counted_is_not(self) -> None:
        assert _facts(n_candidate_failed=3, n_infra_excluded=1).all_failed
        assert not _facts(n_candidate_failed=0, n_infra_excluded=4).all_failed

    def test_an_analysis_frozen_before_the_count_was_kept_reads_as_not_counted(self) -> None:
        stored = _facts().model_dump(mode="json")
        del stored["n_candidate_failed"]
        legacy = CellFacts.model_validate(stored)
        assert legacy.n_candidate_failed is None and not legacy.all_failed
