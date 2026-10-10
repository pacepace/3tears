"""A pooled composite names the dimensions it was meaned over, and a ragged pool is marked where it is shown (#638).

A composite is a mean across whatever capability dimensions a result carried, so two results scored on disjoint
dimension sets each give a number on 0-1, and their mean is arithmetic over two different questions. Each test
here pools two such results and fails if the pooled figure drops either basis or the ragged mark.
"""

from __future__ import annotations

import pytest

from threetears.evals.analysis.reads import compare_two_runs
from threetears.evals.analysis.reporting import (
    METRIC_COMPOSITE,
    compute_frontier,
    compute_history,
    compute_pivot,
    pooled_composite_basis,
    project_score_records,
)
from threetears.evals.contracts.errors import NotFoundError
from threetears.evals.contracts.models import EvalTemplate, GoalStateOutcome, RubricScore
from threetears.evals.contracts.scoring import composite_basis, compute_composite_summary, pool_composite_bases
from threetears.evals.ops.lenses import RunsCompared, history_text, pivot_text, runs_compared_text
from packages.evals.tests.factories import make_eval_result, make_eval_run, make_subject
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

_HOST = toyhost_profile(every_seat=True)

#: Two dimension sets with nothing in common.
_LEFT = ["extraction.accuracy", "extraction.format"]
_RIGHT = ["extraction.tone"]


def _result(run, case: str, dims: list[str], *, score: int = 4, **overrides):
    return make_eval_result(
        eval_run_id=run.id,
        scope_id=run.scope_id,
        model=run.candidate_model,
        test_case_id=case,
        goal_state_outcomes=[],
        rubric_scores=[RubricScore(dim=dim, score=score, scale="ordinal") for dim in dims],
        **overrides,
    )


def _ragged_run(**run_overrides):
    run = make_eval_run(subject_snapshot=make_subject("ent-maple", "Maple"), **run_overrides)
    return run, [_result(run, "tc1", _LEFT), _result(run, "tc2", _RIGHT, score=2)]


def _no_template(template_id: str) -> EvalTemplate:
    raise NotFoundError("template", template_id)


class TestThePooledBasis:
    def test_two_results_with_no_dimension_in_common_name_both_bases_and_are_ragged(self) -> None:
        """The issue's Done-when, on the summary every compare reads."""
        run, results = _ragged_run()

        summary = compute_composite_summary(results)[(run.candidate_model, run.id)]

        assert summary["mean_composite"] == pytest.approx((0.75 + 0.25) / 2)
        assert summary["composite_basis"] == {
            "dimensions": sorted([*_LEFT, *_RIGHT]),
            "bases": sorted([_LEFT, _RIGHT]),
            "ragged": True,
        }

    def test_one_shared_basis_is_not_ragged(self) -> None:
        run = make_eval_run()
        summary = compute_composite_summary([_result(run, "tc1", _LEFT), _result(run, "tc2", _LEFT)])
        assert summary[(run.candidate_model, run.id)]["composite_basis"] == {
            "dimensions": _LEFT,
            "bases": [_LEFT],
            "ragged": False,
        }

    def test_a_candidate_failure_scores_by_policy_and_does_not_make_a_pool_ragged(self) -> None:
        run = make_eval_run()
        failed = _result(run, "tc2", [], candidate_error="boom")
        assert composite_basis(failed) == []
        basis = pool_composite_bases([composite_basis(_result(run, "tc1", _LEFT)), composite_basis(failed)])
        assert basis is not None and basis.bases == [_LEFT] and not basis.ragged

    def test_no_composite_has_no_basis(self) -> None:
        run = make_eval_run()
        goal_only = make_eval_result(
            eval_run_id=run.id, goal_state_outcomes=[GoalStateOutcome(expression="x", passed=True)], rubric_scores=[]
        )
        assert composite_basis(goal_only) is None
        assert pool_composite_bases([None]) is None
        assert compute_composite_summary([goal_only])[(goal_only.model, run.id)]["composite_basis"] is None

    def test_the_disclosure_names_each_set(self) -> None:
        run, results = _ragged_run()
        basis = pooled_composite_basis(results)
        assert basis is not None
        sentence = basis.disclosure()
        assert sentence is not None and "ragged composite" in sentence
        assert all(dim in sentence for dim in [*_LEFT, *_RIGHT])


class TestWhereThePoolIsShown:
    def test_a_pivot_cell_and_its_table_are_marked(self) -> None:
        run, results = _ragged_run()
        records = project_score_records([run], results, profile=_HOST).records
        table = compute_pivot(
            records, row_factor="model", column_factor="template_id", metric=METRIC_COMPOSITE, profile=_HOST
        )

        (cell,) = table.cells
        assert cell.composite_basis is not None and cell.composite_basis.ragged
        assert cell.composite_basis.bases == sorted([_LEFT, _RIGHT])
        assert table.composite_bases_differ is True
        text = pivot_text(table)
        assert "ragged composite" in text and "extraction.tone" in text

    def test_a_frontier_point_is_marked(self) -> None:
        run, results = _ragged_run()
        (point,) = compute_frontier([run], results).subjects[0].points
        assert point.composite_basis is not None and point.composite_basis.ragged
        assert point.composite_basis.dimensions == sorted([*_LEFT, *_RIGHT])

    def test_a_history_point_is_marked(self) -> None:
        run, results = _ragged_run(test_case_ids=["tc1", "tc2"])
        out = compute_history([run], results, metric=METRIC_COMPOSITE, profile=_HOST)
        (point,) = out.series[0].points
        assert point.composite_basis is not None and point.composite_basis.ragged
        assert "ragged composite" in history_text(out)

    def test_a_run_comparison_carries_each_sides_basis_and_marks_the_ragged_one(self) -> None:
        run_a, results_a = _ragged_run(test_case_ids=["tc1", "tc2"])
        run_b = make_eval_run(
            subject_snapshot=run_a.subject_snapshot, template_id=run_a.template_id, test_case_ids=["tc1", "tc2"]
        )
        results_b = [_result(run_b, "tc1", _LEFT), _result(run_b, "tc2", _LEFT)]

        class _Store:
            def query_eval_results_by_run(self, run_id: str, scope_id: str):
                return results_a if run_id == run_a.id else results_b

            def load_eval_run(self, run_id: str, scope_id: str):
                return run_a if run_id == run_a.id else run_b

        compared = compare_two_runs(
            _Store(),  # type: ignore[arg-type]
            run_a.id,
            run_b.id,
            run_a.scope_id,
            load_template=_no_template,
            subject_detail=lambda run: {},
        )
        arm = compared["comparison"]["arm"]
        assert arm["composite_basis_a"]["ragged"] is True
        assert arm["composite_basis_b"] == {"dimensions": _LEFT, "bases": [_LEFT], "ragged": False}
        assert arm["composite_bases_differ"] is True
        text = runs_compared_text(
            RunsCompared(
                baseline_run_id=run_a.id,
                candidate_run_id=run_b.id,
                comparison=compared,
                completeness_disclosures={},
                measurement_window_disclosure=None,
                cassette_mode_disclosure=None,
            )
        )
        assert "composite bases differ" in text and "(ragged)" in text
        assert compared["rubric_threshold"] == 3
        assert "pass^k (k=1, criterion >= 3 of 5)" in text, "the printed pass^k states the threshold it used"


def test_a_lever_dispersion_built_from_a_ragged_pool_says_so() -> None:
    """The bundle's per-lever spread is a figure built from composites; its text carries the ragged mark."""
    from threetears.evals.analysis import assemble_context_bundle
    from threetears.evals.contracts import EvalCampaign

    from packages.evals.tests.factories import fixture_variant_key, minimal_declaration
    from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage

    runs, results = [], {}
    for model in ("control-model", "contrast-model"):
        run = make_eval_run(status="completed", candidate_model=model)
        runs.append(run)
        results[run.id] = [
            _result(run, f"tc-{case}", _LEFT if case % 2 else _RIGHT, score=2 + case % 3, id=f"{model}-{case}")
            for case in range(6)
        ]
    campaign = EvalCampaign(
        scope_id=runs[0].scope_id,
        name="ragged",
        subject_id=runs[0].subject_snapshot.subject_id,
        subject_kind="s",
        behavior="b",
        run_ids=[run.id for run in runs],
        declared_design=minimal_declaration(control=fixture_variant_key("control-model")),
        created_by="test:fixture",
    )
    bundle = assemble_context_bundle(campaign, storage=ToyhostStorage(runs, results), profile=toyhost_profile())

    (model_lever,) = [lever for lever in bundle.coverage if lever.name == "model"]
    assert model_lever.dispersion.startswith("±")
    assert "ragged composite" in model_lever.dispersion and "extraction.tone" in model_lever.dispersion
