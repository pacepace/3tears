"""#684: every direct read surface that groups by contestant names the models that answered it.

A contestant is keyed by the model id its launch asked for, and a provider resolves a floating alias on its
side, so two runs of one alias can be answered by different models and still pool as one contestant. The
mixture is disclosed, not split, on each surface the bundle's ``arm_served_models`` does not reach:

* a frontier point (and the verdict that picks it), a history series and each of its points, a pivot cell
  grouped by ``variant_key`` or ``model``, and each side of a two-run comparison name the served models they
  pooled, as ``one`` / ``pooled`` / ``unrecorded``;
* a run whose responses named no model reads ``unrecorded``, never the alias;
* ``served_model`` is a pivotable coordinate and an export column, so a reader can split by it.
"""

from __future__ import annotations

import csv
import io

from threetears.evals.analysis import SERVED_MODEL_UNRECORDED, ServedModelReading, compare_two_runs
from threetears.evals.analysis.reporting import METRIC_COMPOSITE, compute_frontier, compute_pivot, project_score_records
from threetears.evals.analysis.lenses.history import compute_history
from threetears.evals.analysis.lenses.export import export_records_csv
from threetears.evals.schema import EvalResult, EvalRun, EvalTemplate, RoleUsage
from threetears.evals.kernel import EvalStorage
from threetears.evals.ops import pivot_text
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_eval_result, make_eval_run, make_template
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

#: The alias both runs ask for, and the two models a provider resolved it to months apart.
_ALIAS = "~vendor/model-latest"
_MARCH = "vendor/model-2026-03"
_JUNE = "vendor/model-2026-06"
_CASES = ("c1", "c2", "c3")
_PROFILE = toyhost_profile(every_seat=True)


def _run(run_id: str, created_at: str) -> EvalRun:
    return make_eval_run(
        id=run_id, status="completed", candidate_model=_ALIAS, test_case_ids=list(_CASES), created_at=created_at
    )


def _results(run: EvalRun, served: str | None) -> list[EvalResult]:
    """One result per case, each candidate call's response naming ``served`` (``None``: it named none)."""
    return [
        make_eval_result(
            id=f"{run.id}-{case}",
            eval_run_id=run.id,
            scope_id=run.scope_id,
            model=_ALIAS,
            test_case_id=case,
            usage=[RoleUsage(role="candidate", model=_ALIAS, served_model=served, cost_usd=0.01)],
        )
        for case in _CASES
    ]


def _corpus(second: str | None = _JUNE) -> tuple[list[EvalRun], list[EvalResult]]:
    """Two runs of one alias: March's answered by one model, the later one by ``second``."""
    march, june = _run("run-march", "2026-03-01T00:00:00+00:00"), _run("run-june", "2026-06-01T00:00:00+00:00")
    return [march, june], _results(march, _MARCH) + _results(june, second)


def _pooled(reading: ServedModelReading | None) -> None:
    assert reading is not None
    assert reading.state == "pooled"
    assert reading.served_models == [_MARCH, _JUNE]
    assert _ALIAS not in reading.served_models


class TestTheFrontier:
    def test_a_point_pooling_two_served_models_names_both(self) -> None:
        runs, results = _corpus()

        (subject,) = compute_frontier(runs, results, bar=0.0, archived_run_ids=None).subjects
        (point,) = subject.points

        _pooled(point.served_models)
        assert subject.verdict is not None
        assert subject.verdict.served_models == point.served_models

    def test_a_point_whose_responses_named_no_model_reads_unrecorded(self) -> None:
        runs, results = _corpus(second=None)

        (subject,) = compute_frontier(runs, results, archived_run_ids=None).subjects
        (point,) = subject.points

        assert point.served_models is not None
        assert point.served_models.state == "unrecorded"
        assert point.served_models.served_models == [_MARCH]
        assert point.served_models.n_unrecorded == 3


class TestTheHistory:
    def test_the_series_names_both_and_each_point_names_its_own(self) -> None:
        runs, results = _corpus()

        (series,) = compute_history(runs, results, archived_run_ids=None, profile=_PROFILE).series

        _pooled(series.served_models)
        march, june = series.points
        assert march.served_models is not None and march.served_models.served_models == [_MARCH]
        assert june.served_models is not None and june.served_models.served_models == [_JUNE]

    def test_a_point_whose_responses_named_no_model_reads_unrecorded_never_the_alias(self) -> None:
        runs, results = _corpus(second=None)

        (series,) = compute_history(runs, results, archived_run_ids=None, profile=_PROFILE).series

        later = series.points[1].served_models
        assert later is not None
        assert (later.state, later.served_models) == ("unrecorded", [])


class TestThePivot:
    def _records(self, second: str | None = _JUNE) -> list:
        runs, results = _corpus(second)
        return project_score_records(runs, results, archived_run_ids=None, profile=_PROFILE).records

    def test_a_cell_grouped_by_contestant_names_both_and_the_table_says_so(self) -> None:
        table = compute_pivot(
            self._records(), row_factor="model", column_factor="template_id", metric=METRIC_COMPOSITE, profile=_PROFILE
        )

        (cell,) = table.cells
        _pooled(cell.served_models)
        assert table.served_model_disclosure is not None
        assert _MARCH in table.served_model_disclosure and _JUNE in table.served_model_disclosure
        assert "pivot on served_model" in pivot_text(table)

    def test_served_model_is_an_axis_that_splits_them(self) -> None:
        table = compute_pivot(
            self._records(), row_factor="served_model", column_factor="model", metric=METRIC_COMPOSITE, profile=_PROFILE
        )

        assert table.rows == [_MARCH, _JUNE]
        assert all(cell.n_cases == 3 for cell in table.cells)

    def test_a_cell_whose_responses_named_no_model_reads_unrecorded(self) -> None:
        records = self._records(second=None)
        table = compute_pivot(
            records, row_factor="served_model", column_factor="variant_key", metric=METRIC_COMPOSITE, profile=_PROFILE
        )

        assert table.rows == sorted([_MARCH, SERVED_MODEL_UNRECORDED])
        assert _ALIAS not in table.rows
        (unrecorded,) = [cell for cell in table.cells if cell.row == SERVED_MODEL_UNRECORDED]
        assert unrecorded.served_models is not None and unrecorded.served_models.state == "unrecorded"

    def test_the_export_carries_it_as_a_column(self) -> None:
        rows = list(csv.DictReader(io.StringIO(export_records_csv(self._records()))))

        assert {row["served_model"] for row in rows} == {_MARCH, _JUNE}


class TestCompareRuns:
    def _compare(self, second: str | None) -> dict:
        storage = EvalStorage(InMemoryDocumentStore())
        runs, results = _corpus(second)
        for run in runs:
            storage.save_eval_run(run)
        for result in results:
            storage.save_eval_result(result)

        def template(template_id: str) -> EvalTemplate:
            return make_template(id=template_id)

        return compare_two_runs(
            storage,
            "run-march",
            "run-june",
            runs[0].scope_id,
            load_template=template,
            subject_detail=lambda run: {},
        )["comparison"]["arm"]

    def test_each_side_names_the_model_that_answered_it(self) -> None:
        arm = self._compare(_JUNE)

        assert arm["model_a"] == arm["model_b"] == _ALIAS
        assert arm["served_models_a"]["served_models"] == [_MARCH]
        assert arm["served_models_b"]["served_models"] == [_JUNE]
        assert arm["served_models_a"]["state"] == arm["served_models_b"]["state"] == "one"

    def test_a_side_whose_responses_named_no_model_reads_unrecorded(self) -> None:
        arm = self._compare(None)

        assert arm["served_models_b"]["state"] == "unrecorded"
        assert arm["served_models_b"]["served_models"] == []
