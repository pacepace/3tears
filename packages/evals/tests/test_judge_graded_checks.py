"""A judge-graded number never decides a trial's pass: a check lives in a code-graded family, or it is no check.

A check (a boolean, or a pass/fail scorer, with a better end) fails its trial on its value alone, so it moves the
ranked pass rate. Nothing ranks on a judge-graded measure (``is_code_graded``), yet ``failed_checks`` read every check
whatever its family, so a host could rank on a judge's opinion with no judge identity and no check behind it.
Contracts held here (``MeasureRegistry``, ``kernel.scoring.failed_checks``, ``compute_pass_hat_k``,
``measure_catalog_of``):

- a registry refuses a check in a judge-graded family, a host's or the engine's, naming the measure, its family and
  the rule, and every such measure at once;
- an undirected boolean and a directed numeric reading in a judge-graded family, and a check in a code-graded host
  family, are declared as before;
- a check whose family no code grades fails no trial, holds no arm's pass criterion, and leaves pass^k where it was;
- a check in a code-graded host family fails its trial in pass^k and in the census, as it did.

Every expectation is counted from the fixture or read off the census, never typed.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from packages.evals.tests.factories import make_eval_result, make_eval_run
from threetears.evals.analysis.bundle.trials import read_trial
from threetears.evals.kernel import MeasureFamily, MetricDescriptor
from threetears.evals.kernel.host import MeasureRegistrationError, MeasureRegistry
from threetears.evals.kernel.scoring import compute_pass_hat_k, failed_checks, measure_catalog_of, pass_hat_k_at
from threetears.evals.schema.models import EvalResult, EvalRun

_JUDGED = MeasureFamily(name="table_feel", graded_by="judge", description="How a table felt, as a judge read it.")
_TALLY = MeasureFamily(name="tally", graded_by="code", description="A count code took.")

CHECK = "felt_welcome"
K = 3
CASES = 4
THRESHOLD = 3

#: The (case, repeat) trials on which the check reads its failing value.
FAILING = {(0, 1), (1, 2), (1, 3), (3, 1)}


def _check(family: str | None, *, name: str = CHECK) -> MetricDescriptor:
    """A boolean with a better end, in ``family``: true passes."""
    return MetricDescriptor(
        name=name,
        reader_name=name.replace("_", " ").capitalize(),
        data_type="boolean",
        family=family,
        transferability_class="judge_mediated",
        attribution_scope="end_to_end",
        description="Whether the table felt welcome.",
        higher_is_better=True,
    )


def _pass_fail(family: str, *, name: str = "welcome_graded") -> MetricDescriptor:
    """A 0-or-1 scorer with a better end, in ``family``."""
    return MetricDescriptor(
        name=name,
        reader_name=name.replace("_", " ").capitalize(),
        data_type="numeric",
        family=family,
        transferability_class="judge_mediated",
        attribution_scope="end_to_end",
        description="Whether the table was graded welcome.",
        higher_is_better=True,
        value_range=(0.0, 1.0),
        pass_fail=True,
    )


def _results(run: EvalRun, *, failing: set[tuple[int, int]]) -> list[EvalResult]:
    """Every case run ``K`` times, passing its goal check and its judged dimension; the check reads False on
    ``failing``."""
    return [
        make_eval_result(
            id=f"r-{case}-{k}",
            eval_run_id=run.id,
            scope_id=run.scope_id,
            model=run.candidate_model,
            test_case_id=f"tc-{case}",
            k_iteration=k,
            host_measures={CHECK: (case, k) not in failing},
        )
        for case in range(CASES)
        for k in range(1, K + 1)
    ]


# --- refused where the host declares it ---------------------------------------------------------------


def test_a_boolean_check_in_a_judge_graded_host_family_is_refused_by_name() -> None:
    with pytest.raises(MeasureRegistrationError) as refused:
        MeasureRegistry([_check(_JUDGED.name)], families=(_JUDGED,))
    message = str(refused.value)
    assert f"{CHECK} is a check (a boolean with a better end) in family {_JUDGED.name}" in message
    assert "graded by a judge" in message and "may not decide a pass" in message
    assert "graded_by='code'" in message and "higher_is_better=None" in message


def test_a_pass_fail_check_in_a_judge_graded_host_family_is_refused_by_name() -> None:
    with pytest.raises(MeasureRegistrationError, match=r"welcome_graded is a check \(a pass/fail scorer with a better"):
        MeasureRegistry([_pass_fail(_JUDGED.name)], families=(_JUDGED,))


@pytest.mark.parametrize("family", ["rubric", "dual_axis", "composite"])
def test_a_check_a_host_declares_in_a_judge_graded_engine_family_is_refused(family: str) -> None:
    with pytest.raises(MeasureRegistrationError, match=f"{CHECK} is a check .* in family {family}, which is graded"):
        MeasureRegistry([_check(family)])


def test_every_judge_graded_check_is_named_at_once() -> None:
    with pytest.raises(MeasureRegistrationError) as refused:
        MeasureRegistry([_check(_JUDGED.name), _pass_fail(_JUDGED.name)], families=(_JUDGED,))
    assert f"{CHECK} is a check" in str(refused.value) and "welcome_graded is a check" in str(refused.value)


def test_what_is_no_check_in_a_judge_graded_family_and_a_code_graded_check_are_declared_as_before() -> None:
    undirected = _check(_JUDGED.name, name="felt_seen").model_copy(update={"higher_is_better": None})
    graded = _pass_fail(_JUDGED.name, name="welcome_share").model_copy(update={"pass_fail": False})
    coded = _check(_TALLY.name, name="seated_on_time")
    measures = MeasureRegistry([undirected, graded, coded], families=(_JUDGED, _TALLY))
    assert measures.names == ("felt_seen", "welcome_share", "seated_on_time")


# --- a check no code graded decides nothing -------------------------------------------------------------


@pytest.mark.parametrize("family", ["rubric", "dual_axis", "composite", None])
def test_a_check_no_code_graded_fails_no_trial_and_leaves_pass_k_where_it_was(family: str | None) -> None:
    run = make_eval_run(status="completed", candidate_model="m", k_runs=K)
    results = _results(run, failing=FAILING)
    judged = {CHECK: _check(family)}

    assert all(not failed_checks(result, judged) for result in results)
    assert all(read_trial(result, rubric_threshold=THRESHOLD, catalog=judged).state == "passed" for result in results)
    held = compute_pass_hat_k(results, catalog=judged, rubric_threshold=THRESHOLD)[("m", run.id)]
    unheld = compute_pass_hat_k(results, catalog={}, rubric_threshold=THRESHOLD)[("m", run.id)]
    assert held["pass_hat_k"] == unheld["pass_hat_k"] == 1.0


@pytest.mark.parametrize("family", ["rubric", None])
def test_a_check_no_code_graded_holds_no_arm_alone(family: str | None) -> None:
    run = make_eval_run(status="completed", candidate_model="m", k_runs=K)
    bare = [
        result.model_copy(update={"goal_state_outcomes": [], "rubric_scores": [], "judge_model": None})
        for result in _results(run, failing=FAILING)
    ]
    row = compute_pass_hat_k(bare, catalog={CHECK: _check(family)}, rubric_threshold=THRESHOLD)[("m", run.id)]
    assert row["pass_hat_k_unmeasured_reason"] is not None, "no criterion is left, so pass^k is unmeasured, not 1"
    assert row["n_no_criterion_excluded"] == len(bare)


# --- a code-graded host check fails as it did -----------------------------------------------------------


def test_a_check_in_a_code_graded_host_family_fails_its_trials_in_pass_k_and_the_census() -> None:
    run = make_eval_run(status="completed", candidate_model="m", k_runs=K)
    results = _results(run, failing=FAILING)
    catalog = measure_catalog_of(results, MeasureRegistry([_check(_TALLY.name)], families=(_TALLY,)))
    assert catalog[CHECK].family == _TALLY.name

    failed = {(r.test_case_id, r.k_iteration) for r in results if failed_checks(r, catalog)}
    assert failed == {(f"tc-{case}", k) for case, k in FAILING}
    states = [read_trial(r, rubric_threshold=THRESHOLD, catalog=catalog).state for r in results]
    census = Fraction(states.count("passed"), len(states))
    assert census == Fraction(CASES * K - len(FAILING), CASES * K)

    row = compute_pass_hat_k(results, catalog=catalog, rubric_threshold=THRESHOLD)[("m", run.id)]
    assert pass_hat_k_at(row["pass_hat_k_curve"], 1)["pass_hat_k"] == float(census)
    untouched = {case for case in range(CASES) if not any((case, k) in FAILING for k in range(1, K + 1))}
    assert row["pass_hat_k"] == len(untouched) / CASES
