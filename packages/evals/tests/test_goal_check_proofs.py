"""A goal check's pass rate reads as a measurement only where the check is shown to beat doing nothing.

Controls were checked only where a template was authored, and nothing that SHOWS a check's pass rate read
them: a template saved straight to the store, or the quick path's, printed "passed 8/8" for a check a
candidate that did nothing also passes 8/8, with no mark. Now a launch freezes, per check, whether it is
proven (``EvalRun.goal_check_proofs``), and the run summary, the analysis bundle and the code-only report
mark every check that is not.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.analysis.bundle import goal_check_proofs_of
from threetears.evals.analysis.report import DisclosureBlock, build_code_only_report
from threetears.evals.contracts import ControlEndState, EvalRun, EvalTestCase, GoalCheckControls, GoalStateOutcome
from threetears.evals.contracts.errors import ValidationFailedError
from threetears.evals.contracts.models import (
    GOAL_CHECK_PROOF_RULES,
    goal_check_proofs_as_read,
    stale_goal_check_proofs,
    stored_variation,
)
from threetears.evals.contracts.storage import EvalStorage
from threetears.evals.analysis.summary import GoalCheckSummary, summarize_run
from threetears.evals.run import start_run
from threetears.evals.run.check_controls import (
    check_discriminations,
    control_end_state,
    goal_check_proofs,
    refuse_non_discriminating_checks,
)
from threetears.evals.run.launch import LaunchHost
from threetears.evals.run.runner import grade_goal_checks
from threetears.evals.storage.memory import InMemoryDocumentStore
from packages.evals.tests.factories import make_eval_result, make_eval_run
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT, ToyhostStorage
from packages.evals.tests.fixtures.toyhost.launch import toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.run import EVERY_FIELD_EMITTED, RUN_MODELS, toyhost_template


def _uncontrolled() -> Any:
    """The toy template as one saved straight to the store would be: its checks, no controls."""
    return toyhost_template().model_copy(update={"goal_check_controls": None})


def _with_a_control_that_shows_nothing() -> Any:
    """The toy template with its one check's control replaced by one in which nothing was called."""
    template = toyhost_template()
    controls = template.goal_check_controls
    assert controls is not None
    (entry,) = controls.checks
    return template.model_copy(
        update={
            "goal_check_controls": GoalCheckControls(
                checks=[entry.model_copy(update={"control": "idle"})],
                end_states={"idle": ControlEndState(describes="The extractor emitted nothing.")},
            )
        }
    )


class TestALaunchFreezesWhetherEachCheckIsProven:
    def test_an_authored_template_s_checks_are_proven(self) -> None:
        template = toyhost_template()
        assert goal_check_proofs(template, profile=toyhost_profile()) == dict.fromkeys(
            template.goal_state_checks, "proven"
        )

    def test_a_template_saved_without_controls_has_every_check_unproven(self) -> None:
        template = _uncontrolled()
        assert goal_check_proofs(template, profile=toyhost_profile()) == dict.fromkeys(
            template.goal_state_checks, "unproven"
        )

    def test_a_control_the_check_does_not_beat_refutes_it(self) -> None:
        template = _with_a_control_that_shows_nothing()
        assert goal_check_proofs(template, profile=toyhost_profile()) == dict.fromkeys(
            template.goal_state_checks, "refuted"
        )

    async def test_the_launched_run_records_them_and_its_summary_marks_the_unproven(self) -> None:
        storage = EvalStorage(InMemoryDocumentStore())
        storage.save_template(_uncontrolled())
        host, _client = toyhost_launch_host(storage=storage)
        (run,) = await _launch(host)
        stored = storage.load_eval_run(run.id, TOYHOST_SCOPE)
        assert stored is not None
        assert stored.goal_check_proofs == dict.fromkeys(toyhost_template().goal_state_checks, "unproven")
        summary = summarize_run(host.eval_host, run.id, TOYHOST_SCOPE)
        lines = [line for line in summary.render().splitlines() if "goal check" in line]
        assert lines and all(
            "— unproven: no control shows it tells acting from doing nothing" in line for line in lines
        )


async def _launch(host: LaunchHost) -> list[EvalRun]:
    runs = await start_run(
        host,
        template_id=toyhost_template().id,
        scope_id=TOYHOST_SCOPE,
        subject_id=TOYHOST_SUBJECT.subject_id,
        models=[RUN_MODELS[0]],
    )
    async with asyncio.timeout(10):
        while any(host.job_manager.is_active(run.id) for run in runs):
            await asyncio.sleep(0.01)
    return runs


class TestTheSummaryNeverPrintsAnUnqualifiedPassForAnUnprovenCheck:
    def test_a_proven_check_reads_as_its_pass_count(self) -> None:
        assert GoalCheckSummary(check="c", passed=5, n=8, proof="proven").line() == "goal check c: passed 5/8"

    def test_an_unproven_check_says_so_beside_its_count(self) -> None:
        line = GoalCheckSummary(check="c", passed=8, n=8, proof="unproven").line()
        assert line.startswith("goal check c: passed 8/8 — unproven")

    def test_a_check_with_no_recorded_proof_reads_as_unproven(self) -> None:
        line = GoalCheckSummary(check="c", passed=8, n=8, proof=None).line()
        assert "— unproven: this run recorded no proof" in line

    def test_a_refuted_check_says_its_control_does_not_show_it(self) -> None:
        assert "— refuted" in GoalCheckSummary(check="c", passed=8, n=8, proof="refuted").line()

    def test_a_check_doing_nothing_passes_everywhere_is_marked_as_no_measurement(self) -> None:
        goal = GoalCheckSummary(check="c", passed=8, n=8, proof="unproven", did_nothing_passed=4, did_nothing_cases=4)
        assert "NOT A MEASUREMENT: a candidate that did nothing passes it in 4 of 4 case(s)" in goal.line()

    def test_a_partial_baseline_is_stated_beside_the_unproven_mark(self) -> None:
        goal = GoalCheckSummary(check="c", passed=6, n=8, proof="unproven", did_nothing_passed=2, did_nothing_cases=4)
        assert goal.line().endswith("; a candidate that did nothing passes it in 2 of 4 case(s)")


def _graded(run_id: str, check: str) -> Any:
    return make_eval_result(
        eval_run_id=run_id, goal_state_outcomes=[GoalStateOutcome(expression=check, passed=True, detail="ok")]
    )


class TestTheBundleFoldsEachCheckAcrossItsRuns:
    def test_proven_only_when_every_run_that_graded_it_proved_it(self) -> None:
        runs = [
            make_eval_run(id="a", goal_check_proofs={"c": "proven"}, goal_check_proof_rules=GOAL_CHECK_PROOF_RULES),
            make_eval_run(id="b", goal_check_proofs={"c": "proven"}, goal_check_proof_rules=GOAL_CHECK_PROOF_RULES),
        ]
        (reading,) = goal_check_proofs_of(runs, [_graded("a", "c"), _graded("b", "c")])
        assert (reading.check, reading.measure_id, reading.proof, reading.runs) == ("c", "goal_state:c", "proven", 2)

    def test_a_run_that_recorded_no_proof_leaves_it_unproven(self) -> None:
        runs = [
            make_eval_run(id="a", goal_check_proofs={"c": "proven"}, goal_check_proof_rules=GOAL_CHECK_PROOF_RULES),
            make_eval_run(id="b"),
        ]
        (reading,) = goal_check_proofs_of(runs, [_graded("a", "c"), _graded("b", "c")])
        assert (reading.proof, reading.unrecorded) == ("unproven", 1)

    def test_one_refuting_run_refutes_it(self) -> None:
        runs = [
            make_eval_run(id="a", goal_check_proofs={"c": "proven"}),
            make_eval_run(id="b", goal_check_proofs={"c": "refuted"}),
        ]
        (reading,) = goal_check_proofs_of(runs, [_graded("a", "c"), _graded("b", "c")])
        assert reading.proof == "refuted"

    def test_the_code_only_report_discloses_every_check_not_proven(self) -> None:
        campaign, toy = toyhost_campaign()
        runs = toy.load_eval_runs(campaign.run_ids, TOYHOST_SCOPE)
        results = {run.id: toy.query_eval_results_by_run(run.id, TOYHOST_SCOPE) for run in runs}
        checks = {
            outcome.expression for members in results.values() for r in members for outcome in r.goal_state_outcomes
        }
        assert checks, "the toy campaign's results grade goal checks"
        storage = ToyhostStorage([run.model_copy(update={"goal_check_proofs": None}) for run in runs], results)
        bundle = assemble_context_bundle(campaign, storage=storage, profile=toyhost_profile())
        assert {reading.check for reading in bundle.goal_check_proofs} == checks
        assert all(reading.proof == "unproven" for reading in bundle.goal_check_proofs)
        report = build_code_only_report(
            bundle, measures=toyhost_profile().measures, assembled_at="2026-10-09T00:00:00+00:00"
        )
        texts = [block.text for block in report.blocks if isinstance(block, DisclosureBlock)]
        disclosed = [text for text in texts if text.startswith("Goal check ")]
        assert len(disclosed) == len(checks)
        assert all("is not shown to measure the behaviour" in text for text in disclosed)


# --- a control reads its parameters as a case stores them ---------------------------------------------


def _reading_variation(tail: str, stated: Any) -> tuple[Any, str]:
    """The toy template with one act check that also reads a case parameter, its control stating ``stated``."""
    template = toyhost_template()
    controls = template.goal_check_controls
    assert controls is not None
    (entry,) = controls.checks
    check = f"{EVERY_FIELD_EMITTED} and {tail}"
    end_state = controls.end_states[entry.control].model_copy(update={"variation": {"p": stated}})
    return (
        template.model_copy(
            update={
                "goal_state_checks": [check],
                "goal_check_controls": GoalCheckControls(
                    checks=[entry.model_copy(update={"check": check})], end_states={entry.control: end_state}
                ),
            }
        ),
        check,
    )


def _on_a_case(template: Any, check: str, stated: Any) -> bool:
    """The check's verdict on a cell that did what the control states, under a case stored with ``stated``."""
    profile = toyhost_profile()
    controls = template.goal_check_controls
    case = EvalTestCase(
        scope_id=TOYHOST_SCOPE, template_id=template.id, variation_params=stored_variation({"p": stated})
    )
    acted = control_end_state(template, next(iter(controls.end_states.values())), world=profile.world)
    (outcome,) = grade_goal_checks(
        [check],
        ledger=acted.ledger,
        end_state=acted.end_state,
        fired=acted.fired,
        variation=case.variation_params,
        world=profile.world,
    )
    return outcome.passed


class TestAControlCannotProveACheckOnATypeNoCaseHolds:
    """A control may state any JSON value as a parameter, but a case stores each as one string (#665).

    A check reading a parameter as a number, a list or a bool passed its control and was stamped proven,
    then failed every real case: a report showing a proven check at a 0% pass rate.
    """

    @pytest.mark.parametrize(
        ("tail", "stated", "named"),
        [
            ("variation.p == 3", 3, "int"),
            ('variation.p == ["eu", "us"]', ["eu", "us"], "list"),
            ("variation.p == True", True, "bool"),
        ],
    )
    def test_it_is_refuted_at_launch_and_refused_at_authoring(self, tail: str, stated: Any, named: str) -> None:
        template, check = _reading_variation(tail, stated)
        profile = toyhost_profile()

        assert not _on_a_case(template, check, stated), "the premise: no stored case can pass this check"
        assert goal_check_proofs(template, profile=profile) == {check: "refuted"}
        (discrimination,) = check_discriminations(template, profile=profile)
        assert not discrimination.controlled.passed, "the control was evaluated on a type no case holds"
        with pytest.raises(ValidationFailedError, match=f"states variation.p as a {named}"):
            refuse_non_discriminating_checks(template, profile=profile)

    def test_a_string_parameter_is_proven_and_grades_the_same_on_a_case(self) -> None:
        template, check = _reading_variation('variation.p == "3"', "3")
        assert goal_check_proofs(template, profile=toyhost_profile()) == {check: "proven"}
        assert _on_a_case(template, check, "3")

    def test_a_check_reading_a_parameter_as_a_list_is_refused_before_it_is_proven(self) -> None:
        template, check = _reading_variation('intersects(["eu"], variation.p)', "eu")
        assert goal_check_proofs(template, profile=toyhost_profile()) == {check: "refuted"}
        with pytest.raises(ValidationFailedError, match=r"intersects\(\) over variation.p"):
            refuse_non_discriminating_checks(template, profile=toyhost_profile())


# --- a proof earned under an earlier rule is not read as one ------------------------------------------


class TestAProofRecordedUnderAnEarlierRuleNeedsReProof:
    """A run stamped before #665 may hold a ``proven`` its control earned on a type no case carries.

    The controls are editable and were not frozen with the run, so the proof cannot be re-derived for the
    template the run graded; it is read as unproven, counted, and every surface says it needs re-proof.
    """

    def test_an_old_proven_reads_unproven_and_an_old_refuted_stands(self) -> None:
        run = make_eval_run(goal_check_proofs={"c": "proven", "d": "refuted", "e": "unproven"})
        assert goal_check_proofs_as_read(run) == {"c": "unproven", "d": "refuted", "e": "unproven"}
        assert stale_goal_check_proofs(run) == ["c"]

    def test_a_proof_under_the_current_rules_reads_as_recorded(self) -> None:
        run = make_eval_run(goal_check_proofs={"c": "proven"}, goal_check_proof_rules=GOAL_CHECK_PROOF_RULES)
        assert goal_check_proofs_as_read(run) == {"c": "proven"}
        assert stale_goal_check_proofs(run) == []

    def test_the_bundle_counts_the_stale_run_and_reads_the_check_unproven(self) -> None:
        runs = [
            make_eval_run(id="a", goal_check_proofs={"c": "proven"}, goal_check_proof_rules=GOAL_CHECK_PROOF_RULES),
            make_eval_run(id="b", goal_check_proofs={"c": "proven"}),
        ]
        (reading,) = goal_check_proofs_of(runs, [_graded("a", "c"), _graded("b", "c")])
        assert (reading.proof, reading.stale, reading.unrecorded) == ("unproven", 1, 0)

    def test_the_summary_line_says_it_needs_re_proving(self) -> None:
        line = GoalCheckSummary(check="c", passed=8, n=8, proof="unproven", stale_proof=True).line()
        assert "recorded under an earlier proof rule and needs re-proving by a new launch" in line

    async def test_a_launch_stamps_the_current_rules(self) -> None:
        storage = EvalStorage(InMemoryDocumentStore())
        storage.save_template(toyhost_template())
        host, _client = toyhost_launch_host(storage=storage)
        (run,) = await _launch(host)
        stored = storage.load_eval_run(run.id, TOYHOST_SCOPE)
        assert stored is not None
        assert stored.goal_check_proof_rules == GOAL_CHECK_PROOF_RULES
        assert set((goal_check_proofs_as_read(stored) or {}).values()) == {"proven"}
        assert stored.refused_goal_checks == {}


# --- a check the grammar now refuses is a counted exclusion, not a rig fault per cell ------------------

_REFUSED = 'intersects(["eu"], variation.p)'


class TestACheckTheCurrentGrammarRefusesIsExcludedAndNamed:
    """A template stored before a grammar rule keeps the check the rule now refuses (here, #665's variation read).

    Grading it raised in every cell, so each cell was a rig fault with the reason buried in a generic apparatus
    error. The launch now freezes the refused check with its reason, the cells grade the rest, and the summary,
    the bundle and the report name it.
    """

    async def test_the_cells_grade_the_other_checks_and_every_surface_names_the_refused_one(self) -> None:
        template = _uncontrolled()
        template = template.model_copy(update={"goal_state_checks": [*template.goal_state_checks, _REFUSED]})
        storage = EvalStorage(InMemoryDocumentStore())
        storage.save_template(template)
        host, _client = toyhost_launch_host(storage=storage)

        (run,) = await _launch(host)

        stored = storage.load_eval_run(run.id, TOYHOST_SCOPE)
        assert stored is not None and stored.refused_goal_checks is not None
        assert list(stored.refused_goal_checks) == [_REFUSED]
        assert "intersects() over variation.p" in stored.refused_goal_checks[_REFUSED]
        results = storage.query_eval_results_by_run(run.id, TOYHOST_SCOPE)
        assert results and all(result.infra_error is None for result in results), "no cell is a rig fault"
        graded = {outcome.expression for result in results for outcome in result.goal_state_outcomes}
        assert _REFUSED not in graded and set(toyhost_template().goal_state_checks) <= graded

        summary = summarize_run(host.eval_host, run.id, TOYHOST_SCOPE)
        (refused,) = [goal for goal in summary.goal_checks if goal.refused is not None]
        assert (refused.check, refused.excluded) == (_REFUSED, len(results))
        assert "refused under the current grammar" in refused.line()

        (reading,) = [r for r in goal_check_proofs_of([stored], results) if r.refused is not None]
        assert (reading.check, reading.proof, reading.runs) == (_REFUSED, "refuted", 1)
