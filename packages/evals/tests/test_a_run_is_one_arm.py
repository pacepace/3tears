"""A run is one arm, and its document cannot say otherwise.

The arm is the unit: a launch naming several candidate models starts one run per
model. The run used to carry ``models: list[str]`` anyway, and an analysis-time refusal turned away
any run whose list (or whose results) named more than one. That refusal guarded a state the type
let every writer produce. The run now names one ``candidate_model``, so the state is not
representable and the refusal is gone; these tests are what make the type's refusal fire.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.evals.analysis.bundle import RunSummary
from threetears.evals.contracts.models import EvalRun
from threetears.evals.run.runner import cell_execution_order
from packages.evals.tests.factories import make_eval_run, make_test_case


def _run_fields() -> dict[str, object]:
    """A valid run's fields, as a stored document carries them.

    Returns:
        The fields of a run that validates.
    """
    return make_eval_run(candidate_model="m1").model_dump(mode="json")


def test_a_stored_run_validates_with_one_candidate_model() -> None:
    """The positive control: without it, every refusal below could be a broken fixture."""
    assert EvalRun.model_validate(_run_fields()).candidate_model == "m1"


def test_a_run_naming_no_candidate_model_is_refused() -> None:
    fields = _run_fields()
    fields["candidate_model"] = ""

    with pytest.raises(ValidationError, match="candidate_model"):
        EvalRun.model_validate(fields)


def test_a_run_naming_several_candidate_models_is_refused() -> None:
    fields = _run_fields()
    fields["candidate_model"] = ["m1", "m2"]

    with pytest.raises(ValidationError, match="candidate_model"):
        EvalRun.model_validate(fields)


def test_the_retired_model_list_is_refused_rather_than_read() -> None:
    """A document carrying the old list is from another schema, and a strict read refuses it."""
    fields = _run_fields()
    del fields["candidate_model"]
    fields["models"] = ["m1"]

    with pytest.raises(ValidationError, match="models"):
        EvalRun.model_validate(fields)


def test_a_run_summary_names_one_candidate_model() -> None:
    fields = {
        "run_id": "r1",
        "status": "completed",
        "created_at": "2026-10-04T00:00:00+00:00",
        "k_runs": 1,
        "n_results": 0,
        "n_errors": 0,
        "cost_usd": 0.0,
        "n_cost_unpriced": 0,
    }

    assert RunSummary(**fields, candidate_model="m1").candidate_model == "m1"
    with pytest.raises(ValidationError, match="candidate_model"):
        RunSummary(**fields, candidate_model="")


def test_the_matrix_is_cases_by_repeats_at_the_one_model() -> None:
    """What a run promises and what its loop walks are the same count, with no model factor."""
    cases = [make_test_case(id=f"tc-{i}") for i in range(3)]
    run = make_eval_run(candidate_model="m1", k_runs=2, test_case_ids=[case.id for case in cases])

    cells = cell_execution_order(run, cases)

    assert run.expected_cells == len(cells) == 6
    assert sorted((k, case.id) for k, case in cells) == sorted((k, case.id) for k in (1, 2) for case in cases)
