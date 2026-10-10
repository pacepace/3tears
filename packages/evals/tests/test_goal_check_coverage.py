"""A completed cell is held to grading every goal check its template declares.

The engine knows the set (``template.goal_state_checks``); a kind grades it. A kind that grades a
subset, or none, would otherwise complete cells carrying no outcome for the checks it skipped, and every
per-check rate would be computed over the cells that happened to grade them.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.kernel.candidate_kind import CandidateOutput, CellSink
from threetears.evals.schema.models import EvalTestCase, GoalStateOutcome
from threetears.evals.run.runner import RunnerOptions, execute_run, hold_to_goal_checks
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.kind import TOY_EXTRACTOR_KIND, ScriptedExtractionClient, ToyExtractorKind
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_run, toyhost_template, toyhost_test_cases

_CHECKS = ('state.a == "x"', 'call_count("t.act") == 1')
_KIND_FACT = GoalStateOutcome(expression="field_accuracy >= 0.92", passed=True)


def _graded(*expressions: str) -> list[GoalStateOutcome]:
    return [GoalStateOutcome(expression=expression, passed=True) for expression in expressions]


def test_a_cell_that_graded_every_check_is_stored_as_reported() -> None:
    facts = [_KIND_FACT, *_graded(*_CHECKS)]

    assert hold_to_goal_checks("k", _CHECKS, CandidateOutput(mechanical_facts=facts)) == facts


@pytest.mark.parametrize("graded", [(), (_CHECKS[0],), (_CHECKS[1],)])
def test_a_clean_cell_that_left_a_check_ungraded_is_refused_naming_it(graded: tuple[str, ...]) -> None:
    """A kind-worded fact does not cover a check, so a kind that reports only its own facts is refused too."""
    output = CandidateOutput(mechanical_facts=[_KIND_FACT, *_graded(*graded)])

    with pytest.raises(ValueError, match="without grading the template's goal check") as refused:
        hold_to_goal_checks("k", _CHECKS, output)

    for skipped in set(_CHECKS) - set(graded):
        assert repr(skipped) in str(refused.value)


def test_a_check_written_twice_must_be_graded_twice() -> None:
    with pytest.raises(ValueError, match="without grading"):
        hold_to_goal_checks("k", (_CHECKS[0], _CHECKS[0]), CandidateOutput(mechanical_facts=_graded(_CHECKS[0])))


def test_a_failed_candidate_s_ungraded_checks_are_carried_as_failed_and_not_evaluated() -> None:
    """A candidate failure counts every check failed — but only the checks a result carries."""
    output = CandidateOutput(candidate_errors=["the model refused"], mechanical_facts=_graded(_CHECKS[1]))

    stored = hold_to_goal_checks("k", _CHECKS, output)

    assert [(fact.expression, fact.passed) for fact in stored] == [(_CHECKS[1], True), (_CHECKS[0], False)]
    assert stored[1].detail.startswith("not evaluated: the candidate failed")


def test_an_excluded_cell_is_stored_as_reported_since_it_is_in_no_rate() -> None:
    output = CandidateOutput(infra_errors=["apparatus: the rig broke"])

    assert hold_to_goal_checks("k", _CHECKS, output) == []


def test_a_template_with_no_goal_checks_holds_a_kind_to_nothing() -> None:
    assert hold_to_goal_checks("k", (), CandidateOutput(mechanical_facts=[_KIND_FACT])) == [_KIND_FACT]


async def test_a_run_whose_kind_was_built_without_the_template_s_checks_is_refused() -> None:
    """End to end: the toy extractor built with no checks completes a cell, and the run refuses it."""
    host = toyhost_host()
    world = host.profile.world
    assert world is not None
    template = toyhost_template()
    assert template.goal_state_checks, "the toy template declares a goal check, which this test relies on"
    kind = ToyExtractorKind(client=ScriptedExtractionClient(), world=world, goal_checks=())
    (case, *_) = toyhost_test_cases(template)
    run = toyhost_run(model=RUN_MODELS[0], template=template, kind=kind, world=world).model_copy(
        update={"k_runs": 1, "test_case_ids": [case.id]}
    )
    host.storage.save_eval_run(run)

    with pytest.raises(ValueError, match="without grading the template's goal check"):
        await execute_run(
            host,
            run=run,
            template=template,
            test_cases=[case],
            judge_service=None,
            options=RunnerOptions(candidate_kinds={TOY_EXTRACTOR_KIND: lambda _cell: kind}),
        )


class _RefusingExtractor(ToyExtractorKind):
    """The toy extractor whose candidate fails before its kind grades anything."""

    async def invoke(self, instance: Any, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        return CandidateOutput(candidate_errors=["the model refused"])


async def test_a_failed_candidate_s_stored_result_carries_every_template_check_as_failed() -> None:
    """End to end, read back from the store: the failed cell counts against every check, so no rate is inflated.

    The helper's own return is asserted above; this reads what the run SAVED, which is what every
    per-check rate is computed over.
    """
    host = toyhost_host()
    world = host.profile.world
    assert world is not None
    template = toyhost_template()
    assert template.goal_state_checks, "the toy template declares a goal check, which this test relies on"
    kind = _RefusingExtractor(
        client=ScriptedExtractionClient(), world=world, goal_checks=tuple(template.goal_state_checks)
    )
    (case, *_) = toyhost_test_cases(template)
    run = toyhost_run(model=RUN_MODELS[0], template=template, kind=kind, world=world).model_copy(
        update={"k_runs": 1, "test_case_ids": [case.id]}
    )
    host.storage.save_eval_run(run)

    await execute_run(
        host,
        run=run,
        template=template,
        test_cases=[case],
        judge_service=None,
        options=RunnerOptions(candidate_kinds={TOY_EXTRACTOR_KIND: lambda _cell: kind}),
    )

    (stored,) = host.storage.query_eval_results_by_run(run.id, run.scope_id)
    assert stored.candidate_error is not None
    assert [(fact.expression, fact.passed) for fact in stored.goal_state_outcomes] == [
        (expression, False) for expression in template.goal_state_checks
    ]
    assert all(fact.detail.startswith("not evaluated: the candidate failed") for fact in stored.goal_state_outcomes)
