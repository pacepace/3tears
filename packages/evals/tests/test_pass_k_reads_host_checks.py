"""pass^k fails a trial on every check the census fails it on: one list of what a trial failed, read by both.

A boolean measure a host declares with a better end is a check: on its other value it fails the trial, whatever its
goal checks and judged dimensions read. The report's census (``read_trial``) failed such a trial while pass^k passed
it, so an arm could pass 2 of 180 trials on the scoreboard and read a high pass^k beside it. Contracts held here
(``kernel.scoring.compute_pass_hat_k``, ``compute_frontier``, ``compare_two_runs``, the bundle's frontier and
scoreboard):

- a trial whose goal checks and judged dimensions all pass, but whose directed boolean reads its failing value, fails
  pass^k, and pass^1 is the census's pass rate over the same results;
- a judge that could not tell on a dimension does not lift such a trial out of pass^k: the check has decided it;
- an arm held by such a check alone, no goal check and no judge, has a measured pass^k: the census's pass rate;
- a boolean with no better end, and a numeric measure that is no pass/fail scorer, fail nothing in either;
- a trial the check failed says so on the page in the words its pass rule uses, at the value that failed it;
- a check the host also declares a guardrail is still a check: a trial it fails is failed in the census and in pass^k,
  and the guardrail is read beside them, ``breached`` here, in no comparison family.

Every expectation is read off the census or counted from the fixture, never typed, but the check's name.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from packages.evals.tests.factories import (
    fixture_variant_key,
    make_campaign,
    make_eval_result,
    make_eval_run,
    minimal_declaration,
)
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile
from threetears.evals.analysis import (
    AnalysisContextBundle,
    TableBlock,
    assemble_context_bundle,
    build_code_only_report,
)
from threetears.evals.analysis.bundle.trials import read_trial
from threetears.evals.analysis.lenses.frontier import compute_frontier
from threetears.evals.analysis.reads import compare_two_runs
from threetears.evals.kernel.errors import NotFoundError
from threetears.evals.kernel.host.measures import MeasureRegistry
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.metrics import MetricDescriptor
from threetears.evals.kernel.run_pages import ResultKey
from threetears.evals.kernel.scoring import compute_pass_hat_k, pass_hat_k_at
from threetears.evals.schema.models import EvalResult, EvalRun, EvalTemplate, RoleUsage

#: The host's check: true on a trial whose candidate never topped the customer up, which fails it.
CHECK = "top_up_never"

DIRECTED = MetricDescriptor(
    name=CHECK,
    data_type="boolean",
    family="mechanical",
    transferability_class="mechanical",
    attribution_scope="end_to_end",
    description="Whether the candidate never topped the customer up.",
    reader_name="Never topped up",
    higher_is_better=False,
)

#: The same boolean with no better end: a reading, not a check.
UNDIRECTED = DIRECTED.model_copy(update={"name": "top_up_seen", "reader_name": "Top-up seen", "higher_is_better": None})

#: A directed count that is no pass/fail scorer: worse above zero, but no check.
COUNT = MetricDescriptor(
    name="top_ups",
    data_type="numeric",
    family="mechanical",
    transferability_class="mechanical",
    attribution_scope="end_to_end",
    description="How many times the candidate topped the customer up.",
    reader_name="Top-ups",
    higher_is_better=False,
    nonnegative=True,
)

CATALOG = {descriptor.name: descriptor for descriptor in (DIRECTED, UNDIRECTED, COUNT)}

CASES = 6
K = 3
THRESHOLD = 3

#: The (case, repeat) trials the check fails: every repeat of case 0, one of case 1, two of case 2.
FAILING = {(0, 1), (0, 2), (0, 3), (1, 2), (2, 1), (2, 3)}


def _results(run: EvalRun, *, failing: set[tuple[int, int]], prefix: str, cases: int = CASES) -> list[EvalResult]:
    """Every one of ``cases`` cases run ``K`` times, each passing its goal check and its judged dimension; the check reads its failing
    value on ``failing``, and the two measures that are no check read their worse end on every trial."""
    return [
        make_eval_result(
            id=f"{prefix}-{case}-{k}",
            eval_run_id=run.id,
            scope_id=run.scope_id,
            model=run.candidate_model,
            test_case_id=f"tc-{case}",
            k_iteration=k,
            host_measures={CHECK: (case, k) in failing, UNDIRECTED.name: True, COUNT.name: 4.0},
            usage=[RoleUsage(role="candidate", cost_usd=0.01)],
        )
        for case in range(cases)
        for k in range(1, K + 1)
    ]


def _census_rate(results: list[EvalResult]) -> Fraction:
    """The census's pass rate over the trials it could use: passed over passed and failed."""
    states = [read_trial(result, rubric_threshold=THRESHOLD, catalog=CATALOG).state for result in results]
    assert "lost" not in states, "the fixture loses no trial, so the census and pass^k read every one"
    return Fraction(states.count("passed"), len(states))


def _profile(catalog: dict[str, MetricDescriptor] = CATALOG) -> HostProfile:
    """The toy host, declaring the check and the two measures that are no check."""
    profile = toyhost_profile()
    measures = MeasureRegistry((*TOYHOST_MEASURES, *catalog.values()), families=profile.measures.families)
    return replace(profile, measures=measures)


def test_a_check_on_its_failing_value_fails_pass_k_as_it_fails_the_census() -> None:
    run = make_eval_run(status="completed", candidate_model="m", k_runs=K)
    results = _results(run, failing=FAILING, prefix="r")
    census = _census_rate(results)
    assert census == Fraction(CASES * K - len(FAILING), CASES * K), "the census fails exactly the check's trials"

    row = compute_pass_hat_k(results, catalog=CATALOG, rubric_threshold=THRESHOLD)[("m", run.id)]
    assert pass_hat_k_at(row["pass_hat_k_curve"], 1)["pass_hat_k"] == float(census)
    untouched = {case for case in range(CASES) if not any((case, k) in FAILING for k in range(1, K + 1))}
    assert row["pass_hat_k"] == len(untouched) / CASES, "pass^k at K passes only the cases the check never failed"


def test_a_judge_that_could_not_tell_leaves_a_check_failed_trial_failed() -> None:
    run = make_eval_run(status="completed", candidate_model="m", k_runs=1)
    (result,) = (
        make_eval_result(
            id="undecided",
            eval_run_id=run.id,
            scope_id=run.scope_id,
            model="m",
            rubric_scores=[],
            judge_model="judge",
            judge_cannot_tell={"conversation.tone": "the transcript never shows the reply"},
            host_measures={CHECK: True},
        ),
    )
    assert read_trial(result, rubric_threshold=THRESHOLD, catalog=CATALOG).state == "failed"

    row = compute_pass_hat_k([result], catalog=CATALOG, rubric_threshold=THRESHOLD)[("m", run.id)]
    assert row["pass_hat_k"] == 0.0 and row["n_test_cases"] == 1
    assert row["n_cannot_tell_excluded"] == 0, "the check decided the trial, so the can't-tell cannot lift it out"


def test_an_arm_held_by_a_check_alone_has_a_measured_pass_k() -> None:
    run = make_eval_run(status="completed", candidate_model="m", k_runs=K)
    results = [
        r.model_copy(update={"goal_state_outcomes": [], "rubric_scores": [], "judge_model": None})
        for r in _results(run, failing=FAILING, prefix="r")
    ]
    census = _census_rate(results)

    row = compute_pass_hat_k(results, catalog=CATALOG, rubric_threshold=THRESHOLD)[("m", run.id)]
    assert row["pass_hat_k_unmeasured_reason"] is None and row["n_no_criterion_excluded"] == 0
    assert pass_hat_k_at(row["pass_hat_k_curve"], 1)["pass_hat_k"] == float(census)
    assert all(not read_trial(r, rubric_threshold=THRESHOLD, catalog=CATALOG).no_criterion for r in results)


def test_an_undirected_boolean_or_a_numeric_measure_fails_nothing() -> None:
    run = make_eval_run(status="completed", candidate_model="m", k_runs=K)
    results = _results(run, failing=set(), prefix="r")
    assert all(UNDIRECTED.name in r.host_measures and r.host_measures[COUNT.name] > 0 for r in results)

    assert _census_rate(results) == 1
    row = compute_pass_hat_k(results, catalog=CATALOG, rubric_threshold=THRESHOLD)[("m", run.id)]
    assert row["pass_hat_k"] == 1.0 and row["n_test_cases"] == CASES


def test_the_frontier_reads_the_check_the_host_declares() -> None:
    run = make_eval_run(status="completed", candidate_model="m", k_runs=K)
    results = _results(run, failing=FAILING, prefix="r")

    (point,) = (
        compute_frontier([run], results, archived_run_ids=None, rubric_threshold=THRESHOLD, profile=_profile())
        .subjects[0]
        .points
    )
    census = _census_rate(results)
    assert census < 1, "the census fails the check's trials"
    assert pass_hat_k_at(point.pass_hat_k_curve, 1)["pass_hat_k"] == float(census)


def test_two_runs_compared_read_the_check_and_name_it() -> None:
    held = make_eval_run(status="completed", candidate_model="held", k_runs=K)
    failing = make_eval_run(
        status="completed", candidate_model="failing", k_runs=K, subject_snapshot=held.subject_snapshot
    )
    results = {
        held.id: _results(held, failing=set(), prefix="h"),
        failing.id: _results(failing, failing=FAILING, prefix="f"),
    }

    class _Store:
        def query_eval_result_keys_by_run(self, run_id: str, scope_id: str) -> list[ResultKey]:
            return [ResultKey(r.id, r.test_case_id, r.k_iteration, r.model, r.scored_at) for r in results[run_id]]

        def load_eval_results(self, result_ids: list[str], scope_id: str) -> list[EvalResult]:
            return [r for rows in results.values() for r in rows if r.id in set(result_ids)]

        def load_eval_run(self, run_id: str, scope_id: str) -> EvalRun:
            return held if run_id == held.id else failing

    def no_template(template_id: str) -> EvalTemplate:
        raise NotFoundError("template", template_id, remedy="list them first")

    compared = compare_two_runs(
        _Store(),  # type: ignore[arg-type]
        held.id,
        failing.id,
        held.scope_id,
        load_template=no_template,
        subject_detail=lambda run: {},
        rubric_threshold=THRESHOLD,
        profile=_profile(),
    )
    arm = compared["comparison"]["arm"]
    untouched = {case for case in range(CASES) if not any((case, k) in FAILING for k in range(1, K + 1))}
    assert arm["pass_hat_k_a"] == 1.0
    assert arm["pass_hat_k_b"] == len(untouched) / CASES
    assert arm["pass_checks_b"] == ["Never topped up is false"]


def _bundle(
    *,
    catalog: dict[str, MetricDescriptor] = CATALOG,
    failing_trials: set[tuple[int, int]] = FAILING,
    cases: int = CASES,
) -> AnalysisContextBundle:
    """A control the check never fails beside an arm it fails on ``failing_trials``, assembled under the declaring host."""
    held = make_eval_run(status="completed", candidate_model="held", k_runs=K)
    failing = make_eval_run(
        status="completed", candidate_model="failing", k_runs=K, subject_snapshot=held.subject_snapshot
    )
    results = {
        held.id: _results(held, failing=set(), prefix="h", cases=cases),
        failing.id: _results(failing, failing=failing_trials, prefix="f", cases=cases),
    }
    campaign = make_campaign(
        scope_id=held.scope_id,
        name="an arm a host's check fails",
        subject_id=held.subject_snapshot.subject_id,
        subject_kind="s",
        behavior="b",
        run_ids=[held.id, failing.id],
        declared_design=minimal_declaration(control=fixture_variant_key("held")),
        created_by="test:fixture",
    )
    return assemble_context_bundle(
        campaign, storage=ToyhostStorage([held, failing], results), profile=_profile(catalog)
    )


def test_the_bundle_s_frontier_and_scoreboard_agree_on_the_check() -> None:
    bundle = _bundle()
    boards = {board.variant_key: board for board in bundle.scoreboard}
    points = {p.variant_key: p for subject in bundle.frontier.subjects for p in subject.points}
    for model in ("held", "failing"):
        board = boards[fixture_variant_key(model)]
        pass_hat_1 = pass_hat_k_at(points[fixture_variant_key(model)].pass_hat_k_curve, 1)["pass_hat_k"]
        assert board.usable == CASES * K
        assert pass_hat_1 == board.passed / board.usable, model
    assert boards[fixture_variant_key("failing")].failed == {"check": len(FAILING)}


def test_a_trial_the_check_failed_says_why_in_the_words_its_pass_rule_uses() -> None:
    bundle = _bundle()
    page = build_code_only_report(bundle, assembled_at="2026-11-10T00:00:00+00:00")
    (table,) = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "trials"]
    whys = {row["why"].text for row in table.rows}

    assert len(table.rows) == len(FAILING)
    assert whys == {"check: Never topped up is true"}, "the failing value, named as the rule names the passing one"


#: The same check declared a guardrail too, with a 5-point margin, as ``grader(fn, guard=0.05)`` declares one.
GUARDED = MetricDescriptor.model_validate({**DIRECTED.model_dump(), "guardrail": True, "materiality_threshold": 0.05})

#: Enough cases for the guardrail to be decided: the check fails two of three repeats on 10 of 12 cases.
GUARDED_CASES = 12
GUARDED_FAILING = {(case, k) for case in range(10) for k in (1, 2)}


def test_a_check_declared_a_guardrail_fails_its_trials_and_is_read_as_a_guardrail_too() -> None:
    catalog = {**CATALOG, CHECK: GUARDED}
    bundle = _bundle(catalog=catalog, failing_trials=GUARDED_FAILING, cases=GUARDED_CASES)
    failing = fixture_variant_key("failing")

    board = {board.variant_key: board for board in bundle.scoreboard}[failing]
    assert board.failed == {"check": len(GUARDED_FAILING)}, "the census fails every trial the guardrail check failed"
    assert board.passed == GUARDED_CASES * K - len(GUARDED_FAILING)

    (point,) = [p for subject in bundle.frontier.subjects for p in subject.points if p.variant_key == failing]
    assert pass_hat_k_at(point.pass_hat_k_curve, 1)["pass_hat_k"] == board.passed / board.usable
    untouched = {
        case for case in range(GUARDED_CASES) if not any((case, k) in GUARDED_FAILING for k in range(1, K + 1))
    }
    assert pass_hat_k_at(point.pass_hat_k_curve, K)["pass_hat_k"] == len(untouched) / GUARDED_CASES, (
        "pass^k holds a trial to the guardrail check as the pass rate does"
    )

    (check,) = bundle.guardrails.checks
    assert (check.name, check.contrast.variant_key, check.decision) == (CHECK, failing, "breached")
    assert not any(c.name == CHECK for f in bundle.multiple_comparisons.families for c in f.comparisons), (
        "a guardrail joins no comparison family"
    )
