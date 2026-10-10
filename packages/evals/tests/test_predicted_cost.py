"""Predicted cost: the estimate writes a prediction per planned cell, and the pivot sets it beside the cost observed.

A prediction and an observation are two claims, so the estimate's cells carry a
:class:`~threetears.evals.analysis.PredictedValue` (``method_id="usage-history"``) and a cost pivot
handed the estimate made before the run puts each planned model's prediction in ``predicted``, beside
``value`` and never in it.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``compute_estimate_cost``: a method id other than ``usage-history``; ignoring ``computed_at``.
- ``_planned_cost_per_observation``: dividing the value but not the band's upper end; not dividing
  the value;
  removing either refusal (another metric, no model axis).
- ``compute_pivot``: not setting ``predicted`` on a not-run cell; an empty ``unplaced_predicted_models``; placing
  a plan by its model alone, ignoring its template; not counting the observations its runs did not make.
- ``reads.pivot``: not threading ``predicted_cost`` through; removing the malformed-estimate refusal.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.analysis import COST_PREDICTION_METHOD, PredictedValue, pivot
from threetears.evals.analysis.reporting import METRIC_COMPOSITE, METRIC_COST_USD, ScoreRecord
from threetears.evals.analysis.lenses.cost_estimate import CostEstimate, PlannedCost, compute_estimate_cost
from threetears.evals.analysis.lenses.pivot import PivotError, compute_pivot
from threetears.evals.analysis.stats import lognormal_sum_prediction_band
from threetears.evals.kernel import ValidationFailedError
from threetears.evals.schema.models import EvalResult, EvalRun
from threetears.evals.run.reads import list_runs
from packages.evals.tests.factories import make_eval_result, make_eval_run, memory_storage
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

PROFILE = toyhost_profile(every_seat=True)
STAMP = "2026-10-05T00:00:00+00:00"


def _history(model: str, costs: list[float]) -> tuple[EvalRun, list[EvalResult]]:
    run = make_eval_run(candidate_model=model, status="completed")
    return run, [
        make_eval_result(
            id=f"{model}-hist-{index}",
            eval_run_id=run.id,
            scope_id=run.scope_id,
            model=model,
            test_case_id=f"tc-{index}",
            cost_usd=cost,
        )
        for index, cost in enumerate(costs)
    ]


def _estimate(*, models: list[str], n_test_cases: int = 4) -> CostEstimate:
    sonnet_run, sonnet = _history("sonnet", [0.10, 0.20, 0.30])
    haiku_run, haiku = _history("haiku", [0.05])
    return compute_estimate_cost(
        [sonnet_run, haiku_run],
        [*sonnet, *haiku],
        models=models,
        k_runs=1,
        n_test_cases=n_test_cases,
        computed_at=STAMP,
        profile=PROFILE,
    )


def _cost_record(
    model: str, case: str, value: float, template: str = "tpl-1", run_id: str | None = None
) -> ScoreRecord:
    return ScoreRecord(
        subject_id="ent-maple",
        scope_id="uni-1",
        run_id=run_id or f"run-{model}",
        result_id=f"res-{model}-{template}-{case}",
        template_id=template,
        test_case_id=case,
        model=model,
        k_iteration=1,
        metric=METRIC_COST_USD,
        value=value,
        outcome="ok",
    )


class TestTheEstimateWritesAPredictionPerPlannedCell:
    def test_each_priced_cell_carries_a_usage_history_prediction(self) -> None:
        estimate = _estimate(models=["sonnet", "haiku", "opus"])
        by_model = {cell.model: cell for cell in estimate.cells}

        sonnet = by_model["sonnet"].predicted
        assert isinstance(sonnet, PredictedValue)
        assert sonnet.method_id == COST_PREDICTION_METHOD == "usage-history"
        assert sonnet.computed_at == STAMP
        assert sonnet.value == pytest.approx(0.20 * 4), "the planned cell's total: mean per observation x 4"
        assert sonnet.interval_low is not None and sonnet.interval_high is not None
        assert sonnet.interval_low < sonnet.value < sonnet.interval_high

        haiku = by_model["haiku"].predicted
        assert haiku is not None and haiku.value == pytest.approx(0.05 * 4)
        assert (haiku.interval_low, haiku.interval_high) == (None, None), "one observation: no band"

        assert by_model["opus"].predicted is None, "no history predicts nothing, never zero"
        assert estimate.total_estimated_cost == pytest.approx(sonnet.value + haiku.value)

    def test_an_unstamped_estimate_stamps_now(self) -> None:
        run, results = _history("sonnet", [0.10, 0.20, 0.30])
        estimate = compute_estimate_cost([run], results, models=["sonnet"], k_runs=1, n_test_cases=1, profile=PROFILE)
        predicted = estimate.cells[0].predicted
        assert predicted is not None and predicted.computed_at and predicted.computed_at != STAMP


class TestThePivotSetsThePredictionBesideTheObservedCost:
    def _table(self, estimate: CostEstimate, records: list[ScoreRecord] | None = None):  # type: ignore[no-untyped-def]
        rows = records or [
            _cost_record(model, f"tc-{case}", cost)
            for model, costs in (("sonnet", [0.25, 0.35]), ("haiku", [0.04, 0.06]))
            for case, cost in enumerate(costs)
        ]
        return compute_pivot(
            rows,
            row_factor="template_id",
            column_factor="model",
            metric=METRIC_COST_USD,
            predicted_cost=estimate,
            profile=PROFILE,
        )

    def test_each_cell_carries_its_models_per_observation_prediction_and_keeps_its_observed_value(self) -> None:
        estimate = _estimate(models=["sonnet", "haiku"])
        table = self._table(estimate)
        grid = {(cell.row, cell.column): cell for cell in table.cells}

        sonnet = grid[("tpl-1", "sonnet")]
        assert sonnet.value == pytest.approx(0.30), "the observed mean, untouched by the prediction"
        assert sonnet.predicted is not None
        assert sonnet.predicted.value == pytest.approx(0.20), "per observation: the planned total over its 4 draws"
        assert sonnet.predicted.method_id == COST_PREDICTION_METHOD

        haiku = grid[("tpl-1", "haiku")]
        assert haiku.value == pytest.approx(0.05)
        assert haiku.predicted is not None and haiku.predicted.value == pytest.approx(0.05)
        assert table.unplaced_predicted_models == []

    def test_the_divided_band_is_the_prediction_band_for_the_mean_of_the_planned_draws(self) -> None:
        estimate = _estimate(models=["sonnet"])
        (cell,) = [cell for cell in self._table(estimate).cells if cell.column == "sonnet"]
        assert cell.predicted is not None and cell.predicted.interval_high is not None

        # History 0.10/0.20/0.30, the planned cell n = 4 draws: the band on their mean is the band on their total
        # over 4.
        total_band = lognormal_sum_prediction_band([0.10, 0.20, 0.30], 4)
        assert total_band is not None
        assert cell.predicted.interval_high == pytest.approx(total_band[1] / 4)
        assert cell.predicted.interval_low == pytest.approx(total_band[0] / 4)

    def test_a_planned_cell_that_did_not_run_still_shows_its_prediction(self) -> None:
        estimate = _estimate(models=["sonnet", "haiku"])
        records = [
            _cost_record("sonnet", "tc-0", 0.25, template="tpl-1"),
            _cost_record("haiku", "tc-0", 0.05, template="tpl-2"),
        ]
        grid = {(cell.row, cell.column): cell for cell in self._table(estimate, records).cells}

        not_run = grid[("tpl-1", "haiku")]
        assert not_run.status == "not_run" and not_run.value is None
        assert not_run.predicted is not None and not_run.predicted.value == pytest.approx(0.05)

    def test_a_planned_model_with_no_level_is_named_not_dropped(self) -> None:
        estimate = _estimate(models=["sonnet", "haiku"])
        records = [_cost_record("sonnet", "tc-0", 0.25)]
        assert self._table(estimate, records).unplaced_predicted_models == ["haiku"]

    def test_without_an_estimate_nothing_is_predicted(self) -> None:
        table = compute_pivot(
            [_cost_record("sonnet", "tc-0", 0.25)],
            row_factor="template_id",
            column_factor="model",
            metric=METRIC_COST_USD,
            profile=PROFILE,
        )
        assert all(cell.predicted is None for cell in table.cells)

    def test_a_prediction_handed_to_a_pivot_of_another_metric_is_refused(self) -> None:
        with pytest.raises(PivotError, match="predicted cost sits beside an observed cost"):
            compute_pivot(
                [],
                row_factor="template_id",
                column_factor="model",
                metric=METRIC_COMPOSITE,
                predicted_cost=_estimate(models=["sonnet"]),
                profile=PROFILE,
            )

    def test_a_prediction_handed_to_a_pivot_with_no_model_axis_is_refused(self) -> None:
        with pytest.raises(PivotError, match="neither axis is 'model'"):
            compute_pivot(
                [_cost_record("sonnet", "tc-0", 0.25)],
                row_factor="template_id",
                column_factor="test_case_id",
                metric=METRIC_COST_USD,
                predicted_cost=_estimate(models=["sonnet"]),
                profile=PROFILE,
            )


def _plan(model: str, value: float, *, template: str | None = "tpl-A", run_ids: tuple[str, ...] = ()) -> PlannedCost:
    return PlannedCost(
        model=model,
        template_id=template,
        run_ids=list(run_ids),
        n_observations=4,
        predicted=PredictedValue(value=value, method_id="m", computed_at=STAMP),
    )


def _by_template_and_model(records: list[ScoreRecord], plans: list[PlannedCost]) -> dict[tuple[str, str], Any]:
    table = compute_pivot(
        records,
        row_factor="template_id",
        column_factor="model",
        metric=METRIC_COST_USD,
        predicted_cost=plans,
        profile=PROFILE,
    )
    return {(cell.row, cell.column): cell for cell in table.cells} | {("", "unplaced"): table.unplaced_predicted_models}


class TestAPredictionSitsOnlyWhereItsPlanRan:
    """A plan is one model on one template; a cell of another template at that model is not what it predicted."""

    RECORDS = [
        _cost_record("sonnet", "tc-0", 0.25, "tpl-A"),
        _cost_record("sonnet", "tc-1", 0.35, "tpl-A"),
        _cost_record("sonnet", "tc-0", 5.0, "tpl-OTHER"),
        _cost_record("sonnet", "tc-1", 6.0, "tpl-OTHER"),
    ]

    def test_another_templates_cell_at_the_planned_model_carries_no_prediction(self) -> None:
        grid = _by_template_and_model(self.RECORDS, [_plan("sonnet", 0.8)])

        assert grid[("tpl-A", "sonnet")].predicted is not None
        assert grid[("tpl-A", "sonnet")].predicted.value == pytest.approx(0.2)
        assert grid[("tpl-OTHER", "sonnet")].predicted is None, "its 5.5 is not what the plan predicted"

    def test_a_cell_pooling_templates_carries_none_and_the_plan_is_named_unplaced(self) -> None:
        table = compute_pivot(
            self.RECORDS,
            row_factor="test_case_id",
            column_factor="model",
            metric=METRIC_COST_USD,
            predicted_cost=[_plan("sonnet", 0.8)],
            profile=PROFILE,
        )

        assert all(cell.predicted is None for cell in table.cells)
        assert table.unplaced_predicted_models == ["sonnet (tpl-A)"]

    def test_one_model_planned_on_two_templates_sits_in_each_its_own(self) -> None:
        grid = _by_template_and_model(self.RECORDS, [_plan("sonnet", 0.8), _plan("sonnet", 20.0, template="tpl-OTHER")])

        assert grid[("tpl-A", "sonnet")].predicted.value == pytest.approx(0.2)
        assert grid[("tpl-OTHER", "sonnet")].predicted.value == pytest.approx(5.0)

    def test_one_model_planned_twice_on_one_template_is_refused(self) -> None:
        with pytest.raises(PivotError, match="plans sonnet on tpl-A more than once"):
            _by_template_and_model(self.RECORDS, [_plan("sonnet", 0.4), _plan("sonnet", 40.0)])

    def test_a_cell_says_how_many_of_its_observations_the_plan_did_not_make(self) -> None:
        """The history the prediction was drawn from pools into the same cell; the cell says how much of it."""
        records = [
            _cost_record("sonnet", "tc-0", 0.25, "tpl-A", run_id="history"),
            _cost_record("sonnet", "tc-1", 0.35, "tpl-A", run_id="history"),
            _cost_record("sonnet", "tc-0", 0.30, "tpl-A", run_id="launched"),
        ]

        launched = _by_template_and_model(records, [_plan("sonnet", 0.8, run_ids=("launched",))])
        before = _by_template_and_model(records, [_plan("sonnet", 0.8)])

        assert launched[("tpl-A", "sonnet")].n_unplanned == 2
        assert before[("tpl-A", "sonnet")].n_unplanned is None, "a plan naming no run made none of them"


class TestTheLensTakesTheEstimateTheCallerMadeBeforeTheRun:
    def _stored(self):  # type: ignore[no-untyped-def]
        storage, _ = memory_storage()
        run = make_eval_run(candidate_model="sonnet", status="completed", template_id="tpl-1")
        storage.save_eval_run(run)
        for case, cost in enumerate([0.25, 0.35]):
            storage.save_eval_result(
                make_eval_result(
                    id=f"obs-{case}",
                    eval_run_id=run.id,
                    scope_id=run.scope_id,
                    model="sonnet",
                    test_case_id=f"tc-{case}",
                    cost_usd=cost,
                )
            )
        return storage, run.scope_id

    def _pivot(self, storage, scope_id, predicted_cost):  # type: ignore[no-untyped-def]
        return pivot(
            storage,
            scope_id,
            list_runs=lambda scope, **kwargs: list_runs(toyhost_host(storage=storage), scope, **kwargs),
            row_factor="template_id",
            column_factor="model",
            metric=METRIC_COST_USD,
            predicted_cost=predicted_cost,
            profile=PROFILE,
        )

    def test_the_estimate_as_the_lens_returned_it_reaches_the_cells(self) -> None:
        storage, scope_id = self._stored()

        table = self._pivot(storage, scope_id, _estimate(models=["sonnet"]))

        (cell,) = table.cells
        assert cell.value == pytest.approx(0.30)
        assert cell.predicted is not None
        assert cell.predicted.value == pytest.approx(0.20)
        assert cell.predicted.method_id == "usage-history"

    def test_the_estimate_as_a_caller_across_a_wire_holds_it_reaches_the_cells(self) -> None:
        storage, scope_id = self._stored()

        table = self._pivot(storage, scope_id, _estimate(models=["sonnet"]).model_dump(mode="json"))

        (cell,) = table.cells
        assert cell.predicted is not None and cell.predicted.value == pytest.approx(0.20)

    def test_something_that_is_not_an_estimate_is_refused(self) -> None:
        storage, scope_id = self._stored()
        with pytest.raises(ValidationFailedError, match="not a cost estimate"):
            self._pivot(storage, scope_id, {"cells": "nope"})
