"""#1187: the lenses that read runs outside a campaign's cells keep apart what the cells keep apart.

Owner ruling (2026-10-23, restated for #1187): runs that differ on a recorded coordinate of their identity (the model
that answered, the definition of a host measure), or where the coordinate went unrecorded, never pool; an unrecorded
coordinate matches nothing. Each lens is read through its front door, the read a host serves:

* a comparison set badges runs that asked for one model and were answered by two, and a silence nobody stated;
* a history marks a change of either coordinate between two runs as a boundary: no verdict across it, a fresh
  baseline after it, and the rest of the series still read;
* a pivot withholds a cell where one arm's observations are apart on either, and reads it where they are not;
* a frontier point's composite and cost read its headline pass^k cell, never every run of the variant.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.analysis import comparison_sets, frontier, history, pivot
from threetears.evals.analysis.reporting import METRIC_COMPOSITE
from threetears.evals.kernel import EvalStorage
from threetears.evals.kernel.identity import IDENTITY_VERSION
from threetears.evals.schema import EvalResult, EvalRun, RoleUsage
from threetears.evals.schema.models import RubricScore, ServedModelAttestation
from threetears.evals.schema.reading_states import StateOf
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_eval_result, make_eval_run
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

#: The alias every run asks for, and the two models a provider resolved it to months apart.
_ALIAS = "~vendor/model-latest"
_MARCH = "vendor/model-2026-03"
_JUNE = "vendor/model-2026-06"
_CASES = ("c1", "c2", "c3")
_PROFILE = toyhost_profile(every_seat=True)
_SCOPE = "uni-1"
#: Two definitions of one host measure, as two runs recorded them.
_BEFORE = {"field_accuracy": "digest-before"}
_AFTER = {"field_accuracy": "digest-after"}


def _run(run_id: str, month: int, **fields: Any) -> EvalRun:
    """A completed run of the alias, stamped with one context and one template revision, so only what a test varies
    can keep it apart."""
    return make_eval_run(
        id=run_id,
        status="completed",
        scope_id=_SCOPE,
        candidate_model=_ALIAS,
        test_case_ids=list(_CASES),
        created_at=f"2026-{month:02d}-01T00:00:00+00:00",
        context_key="ctx-1",
        identity_version=IDENTITY_VERSION,
        template_digest="digest-1",
        **fields,
    )


def _results(run: EvalRun, served: str | None, *, score: int = 4, cost: float = 0.01) -> list[EvalResult]:
    """One result per case, each candidate call's response naming ``served`` (``None``: it named none)."""
    return [
        make_eval_result(
            id=f"{run.id}-{case}",
            eval_run_id=run.id,
            scope_id=run.scope_id,
            model=_ALIAS,
            test_case_id=case,
            rubric_scores=[
                RubricScore(
                    dim="conversation.tone", score=score, scale="ordinal", axis="capability", judge_temperature=0.0
                )
            ],
            usage=[RoleUsage(role="candidate", model=_ALIAS, served_model=served, cost_usd=cost)],
        )
        for case in _CASES
    ]


class _Scope:
    """A scope holding the given runs and results, read through the lens reads a host serves."""

    def __init__(self, runs: list[EvalRun], results: list[EvalResult]) -> None:
        self.storage = EvalStorage(InMemoryDocumentStore())
        self.runs = runs
        for run in runs:
            self.storage.save_eval_run(run)
        for result in results:
            self.storage.save_eval_result(result)

    def list_runs(self, scope_id: str, **_filters: Any) -> list[EvalRun]:
        return [run for run in self.runs if run.scope_id == scope_id]


def _attested(run: EvalRun, model: str) -> EvalRun:
    """``run`` with its host's statement of the model that answered its silent calls."""
    statement = ServedModelAttestation(
        attested_by="ops", served_model=model, result_ids=sorted(f"{run.id}-{case}" for case in _CASES)
    )
    return run.model_copy(update={"served_model_attestation": statement})


# =============================================================================
# Comparison sets
# =============================================================================


def _comparison(scope: _Scope) -> dict[str, Any]:
    (grouped,) = comparison_sets(scope.storage, _SCOPE, list_runs=scope.list_runs, profile=_PROFILE)["comparison_sets"]
    return grouped


def test_a_comparison_set_badges_runs_of_one_requested_model_answered_by_two_and_names_them() -> None:
    march, june = _run("run-march", 3), _run("run-june", 6)
    grouped = _comparison(_Scope([march, june], _results(march, _MARCH) + _results(june, _JUNE)))

    assert "served_model_differs" in grouped["badges"]
    disclosure = grouped["served_model_disclosure"]
    assert "named no model" not in disclosure, "every response named its model"
    assert _ALIAS in disclosure and f"{_MARCH} (run-march)" in disclosure and f"{_JUNE} (run-june)" in disclosure


def test_a_comparison_set_of_one_answering_model_carries_no_served_model_badge_or_disclosure() -> None:
    march, june = _run("run-march", 3), _run("run-june", 6)
    grouped = _comparison(_Scope([march, june], _results(march, _MARCH) + _results(june, _MARCH)))

    assert "served_model_differs" not in grouped["badges"]
    assert "served_model_disclosure" not in grouped


def test_a_silence_nobody_stated_is_incomplete_and_a_stated_one_is_read_through_and_said() -> None:
    march, june = _run("run-march", 3), _run("run-june", 6)
    results = _results(march, _MARCH) + _results(june, None)

    silent = _comparison(_Scope([march, june], results))
    assert "context_incomplete" in silent["badges"], "nothing says what answered run-june"
    assert "served_model_differs" not in silent["badges"], "a silence evidences no second model"
    said = silent["served_model_disclosure"]
    assert "run-june" in said and "named no model" in said and "runs_attest" in said

    stated = _comparison(_Scope([march, _attested(june, _JUNE)], results))
    assert "served_model_differs" in stated["badges"], "read through the statement, two models answered"
    said = stated["served_model_disclosure"]
    assert "named no model, so nothing says" not in said, "the statement says what answered"
    assert "attested, not recorded" in said


# =============================================================================
# History
# =============================================================================


def _series(scope: _Scope) -> Any:
    (series,) = history(scope.storage, _SCOPE, list_runs=scope.list_runs, profile=_PROFILE).series
    return series


def test_a_history_marks_a_change_of_served_model_as_a_boundary_and_reads_the_steps_either_side() -> None:
    march, june, july = _run("run-march", 3), _run("run-june", 6), _run("run-july", 7)
    results = _results(march, _MARCH, score=5) + _results(june, _JUNE, score=3) + _results(july, _JUNE, score=2)

    first, changed, after = _series(_Scope([march, june, july], results)).points

    assert first.is_baseline and first.split_on == []
    assert changed.split_on == ["served_model"], "another model answered: no step is read across it"
    assert changed.state == StateOf(state="withheld", reason="split", coordinate="served_model")
    assert changed.regression is None
    assert changed.is_baseline and changed.delta_from_baseline == 0.0, "the stretch after a boundary starts afresh"
    assert after.split_on == [] and after.state is None
    assert after.regression is not None, "the step between two runs of one model is read"
    assert after.delta_from_baseline == after.value - changed.value


def test_two_runs_whose_responses_named_no_model_are_apart_in_a_history_unless_their_host_stated_it() -> None:
    june, july = _run("run-june", 6), _run("run-july", 7)
    results = _results(june, None) + _results(july, None)

    _, silent = _series(_Scope([june, july], results)).points
    assert silent.split_on == ["served_model"], "a silence matches nothing, not even another silence"

    _, stated = _series(_Scope([_attested(june, _JUNE), _attested(july, _JUNE)], results)).points
    assert stated.split_on == [] and stated.regression is not None, "both stated as one model: read through"


def test_a_history_marks_a_change_of_measure_definition_as_a_boundary() -> None:
    before, after = _run("run-before", 3, measure_definitions=_BEFORE), _run("run-after", 6, measure_definitions=_AFTER)
    results = _results(before, _MARCH) + _results(after, _MARCH)

    _, changed = _series(_Scope([before, after], results)).points

    assert changed.split_on == ["measure_definition"]
    assert changed.state == StateOf(state="withheld", reason="split", coordinate="measure_definition")
    assert changed.regression is None and changed.is_baseline


# =============================================================================
# Pivot
# =============================================================================


def _table(scope: _Scope, row_factor: str = "model") -> Any:
    return pivot(
        scope.storage,
        _SCOPE,
        list_runs=scope.list_runs,
        row_factor=row_factor,
        column_factor="template_id",
        metric=METRIC_COMPOSITE,
        profile=_PROFILE,
    )


def test_a_pivot_withholds_a_cell_where_one_arm_was_answered_by_two_models_and_an_axis_on_it_reads_each() -> None:
    march, june = _run("run-march", 3), _run("run-june", 6)
    scope = _Scope([march, june], _results(march, _MARCH, score=5) + _results(june, _JUNE, score=1))

    (cell,) = _table(scope).cells
    assert (cell.status, cell.value) == ("withheld", None)
    assert cell.split_on == ["served_model"]
    assert cell.state == StateOf(state="withheld", reason="split", coordinate="served_model")
    assert cell.n == 6, "what the cell held is still said"

    by_model = _table(scope, row_factor="served_model")
    assert [(cell.row, cell.status, cell.value) for cell in by_model.cells] == [
        (_MARCH, "measured", 1.0),
        (_JUNE, "measured", 0.0),
    ]


def test_a_pivot_withholds_a_cell_read_under_two_definitions_and_reads_one_read_under_one() -> None:
    before, after = _run("run-before", 3, measure_definitions=_BEFORE), _run("run-after", 6, measure_definitions=_AFTER)
    again = _run("run-again", 7, measure_definitions=_AFTER)

    (split,) = _table(_Scope([before, after], _results(before, _MARCH) + _results(after, _MARCH))).cells
    assert split.status == "withheld" and split.split_on == ["measure_definition"]

    (one,) = _table(_Scope([after, again], _results(after, _MARCH) + _results(again, _MARCH))).cells
    assert one.status == "measured" and one.split_on == [] and one.state is None


# =============================================================================
# The frontier's composite and cost
# =============================================================================


def _point(scope: _Scope) -> dict[str, Any]:
    (subject,) = frontier(scope.storage, _SCOPE, list_runs=scope.list_runs, profile=_PROFILE)["subjects"]
    (point,) = subject["points"]
    return point


def test_a_frontier_point_reads_its_composite_and_cost_off_its_headline_cell_not_another_served_model() -> None:
    a, b, c = _run("run-a", 3), _run("run-b", 4), _run("run-c", 6)
    results = (
        _results(a, _MARCH, score=5, cost=0.01)
        + _results(b, _MARCH, score=5, cost=0.01)
        + _results(c, _JUNE, score=1, cost=0.05)
    )

    point = _point(_Scope([a, b, c], results))

    headline, apart = point["pass_hat_k_cells"]
    assert headline["run_ids"] == ["run-a", "run-b"] and apart["run_ids"] == ["run-c"]
    assert apart["reason"] == "other_served_model"
    assert point["mean_composite"] == 1.0, "run-c's composite is kept out, not pooled in"
    assert point["production_replicating_cost"] == 0.01
    assert point["n_cost"] == 6


def test_a_frontier_point_reads_its_composite_off_one_definition_of_a_host_measure() -> None:
    a = _run("run-a", 3, measure_definitions=_AFTER)
    b = _run("run-b", 4, measure_definitions=_AFTER)
    c = _run("run-c", 6, measure_definitions=_BEFORE)
    results = _results(a, _MARCH, score=5) + _results(b, _MARCH, score=5) + _results(c, _MARCH, score=1)

    point = _point(_Scope([a, b, c], results))

    _headline, apart = point["pass_hat_k_cells"]
    assert apart["run_ids"] == ["run-c"] and apart["reason"] == "other_measure_definition"
    assert point["mean_composite"] == 1.0
