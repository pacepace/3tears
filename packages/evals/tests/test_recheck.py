"""Re-grading a stored run's goal checks from the call ledgers its cells stored.

The toy host's kind records its extractor's calls through the engine's ``CallLedger`` and grades
the template's goal check through ``grade_goal_checks``; the runner stores the ledger on each cell's
trace. A re-check reads that ledger back and re-grades through the same function, so it must
reproduce every stored verdict, and must move exactly the verdict whose stored ledger moved.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.contracts import (
    CallLedger,
    ConflictError,
    EvalResult,
    EvalStorage,
    GoalStateOutcome,
    NotFoundError,
    ValidationFailedError,
)
from threetears.evals.run import recheck_goal_states, recheck_result
from packages.evals.tests.factories import (
    make_eval_result,
    make_eval_run,
    make_eval_trace,
    make_test_case,
    memory_storage,
)
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.kind import FIELD_ACCURACY
from packages.evals.tests.fixtures.toyhost.run import EVERY_FIELD_EMITTED, ToyhostRunPath, execute_toyhost_run


def _every_field(passed: bool) -> GoalStateOutcome:
    return GoalStateOutcome(expression=EVERY_FIELD_EMITTED, passed=passed)


def _ledger(emitted: int) -> CallLedger:
    ledger = CallLedger()
    for index in range(emitted):
        ledger.record("extractor", "emit_field", {"field": f"f{index}"})
    return ledger


# =============================================================================
# The toy host, end to end: stored verdicts reproduce, and a moved ledger moves its verdict
# =============================================================================


async def _toy_run() -> ToyhostRunPath:
    return await execute_toyhost_run(host=toyhost_host())


def _drop_last_call(path: ToyhostRunPath, result: EvalResult) -> None:
    """Rewrite one stored trace with the last call of its ledger removed."""
    trace = path.trace(result)
    assert trace is not None and trace.call_ledger is not None
    mutated = trace.call_ledger.model_copy(update={"calls": trace.call_ledger.calls[:-1]})
    path.host.storage.save_eval_result(result, trace.model_copy(update={"call_ledger": mutated}))


async def test_the_toy_kind_stores_the_ledger_it_graded_its_goal_check_against() -> None:
    path = await _toy_run()

    for result in path.results:
        trace = path.trace(result)
        assert trace is not None and trace.call_ledger is not None
        assert [(call.tool, call.action) for call in trace.call_ledger.calls] == [("extractor", "emit_field")] * 4
        (graded,) = [outcome for outcome in result.goal_state_outcomes if outcome.expression == EVERY_FIELD_EMITTED]
        assert graded.passed is True


async def test_re_grading_every_stored_toy_result_reproduces_its_verdict() -> None:
    path = await _toy_run()
    run = path.runs[0]

    recheck = recheck_goal_states(path.host.storage, run.id, run.scope_id, apply=False)

    assert recheck.results, "the run stored no results, so nothing was re-checked"
    for report in recheck.results:
        assert report.not_rechecked is None
        assert report.flips == []
        assert report.regraded == 1
        # The kind's own fact names no root the goal language admits, so it is left as stored.
        assert [kept.expression.split(" ")[0] for kept in report.kept_as_stored] == [FIELD_ACCURACY]


async def test_mutating_a_stored_ledger_flips_exactly_that_result_and_an_apply_stores_it() -> None:
    path = await _toy_run()
    run = path.runs[0]
    target, *others = path.host.storage.query_eval_results_by_run(run.id, run.scope_id)
    _drop_last_call(path, target)

    dry = recheck_goal_states(path.host.storage, run.id, run.scope_id, apply=False)
    flipped = {report.result_id: report.flips for report in dry.results if report.flips}
    assert list(flipped) == [target.id]
    (flip,) = flipped[target.id]
    assert (flip.expression, flip.was, flip.now) == (EVERY_FIELD_EMITTED, True, False)
    assert path.host.storage.load_eval_result(target.id, run.scope_id) == target, "a dry run wrote"

    applied = recheck_goal_states(path.host.storage, run.id, run.scope_id, apply=True)
    assert applied.write_conflicts == []
    rewritten = path.host.storage.load_eval_result(target.id, run.scope_id)
    assert rewritten is not None
    (graded,) = [o for o in rewritten.goal_state_outcomes if o.expression == EVERY_FIELD_EMITTED]
    assert graded.passed is False
    assert [o for o in rewritten.goal_state_outcomes if o.expression != EVERY_FIELD_EMITTED] == [
        o for o in target.goal_state_outcomes if o.expression != EVERY_FIELD_EMITTED
    ]
    assert all(path.host.storage.load_eval_result(other.id, run.scope_id) == other for other in others), (
        "a result whose verdict did not move was rewritten"
    )

    again = recheck_goal_states(path.host.storage, run.id, run.scope_id, apply=True)
    assert not any(report.flips for report in again.results), "an applied re-check is not idempotent"


# =============================================================================
# One result: what is re-graded, what is kept, what is not re-checked
# =============================================================================


def test_a_result_whose_checks_were_not_graded_at_the_cell_s_end_is_not_rechecked() -> None:
    result = make_eval_result(goal_state_outcomes=[_every_field(False)], termination="cell_timeout")

    report, outcomes = recheck_result(result, _ledger(4), variation={})

    assert outcomes is None
    assert report.not_rechecked is not None and "termination=cell_timeout" in report.not_rechecked


def test_a_result_with_no_stored_ledger_is_not_rechecked() -> None:
    report, outcomes = recheck_result(make_eval_result(goal_state_outcomes=[_every_field(True)]), None, variation={})

    assert outcomes is None
    assert report.not_rechecked is not None and "no call ledger" in report.not_rechecked


def test_a_result_with_no_goal_outcomes_is_not_rechecked() -> None:
    report, outcomes = recheck_result(make_eval_result(goal_state_outcomes=[]), _ledger(4), variation={})

    assert outcomes is None
    assert report.not_rechecked == "it carries no goal-state outcomes"


def test_a_check_reading_world_state_is_kept_as_stored() -> None:
    world_check = GoalStateOutcome(expression="state.shop.cart.length >= 1", passed=True)
    result = make_eval_result(goal_state_outcomes=[world_check, _every_field(True)])

    report, outcomes = recheck_result(result, _ledger(3), variation={})

    assert [kept.expression for kept in report.kept_as_stored] == [world_check.expression]
    assert "world state" in report.kept_as_stored[0].reason
    assert outcomes is not None and outcomes[0] == world_check
    assert outcomes[1].passed is False


@pytest.mark.parametrize(("variation", "flips"), [({"wanted": 4}, False), ({"wanted": 3}, True)])
def test_a_check_reading_the_case_is_re_graded_under_the_case_s_variation(
    variation: dict[str, Any], flips: bool
) -> None:
    check = GoalStateOutcome(expression='call_count("extractor.emit_field") == variation.wanted', passed=True)

    report, _ = recheck_result(make_eval_result(goal_state_outcomes=[check]), _ledger(4), variation=variation)

    assert bool(report.flips) is flips


def test_a_check_reading_the_case_is_kept_when_its_test_case_no_longer_resolves() -> None:
    check = GoalStateOutcome(expression='call_count("extractor.emit_field") == variation.wanted', passed=True)

    report, outcomes = recheck_result(make_eval_result(goal_state_outcomes=[check]), _ledger(4), variation=None)

    assert outcomes is None
    assert report.regraded == 0
    assert "test case no longer resolves" in report.kept_as_stored[0].reason


def test_a_check_that_no_longer_evaluates_stops_the_result_s_recheck() -> None:
    broken = GoalStateOutcome(expression="call_count(1) == 4", passed=True)

    report, outcomes = recheck_result(make_eval_result(goal_state_outcomes=[broken]), _ledger(4), variation={})

    assert outcomes is None
    assert report.not_rechecked is not None and "no longer evaluates" in report.not_rechecked


# =============================================================================
# A stored run: refusals, and a rewrite something else raced
# =============================================================================


def _stored_run(storage: EvalStorage, *, status: str = "completed", emitted: int = 3) -> tuple[str, EvalResult]:
    run = make_eval_run(status=status)
    storage.save_eval_run(run)
    storage.save_test_case(make_test_case(id="tc-1"))
    result = make_eval_result(eval_run_id=run.id, goal_state_outcomes=[_every_field(True)])
    storage.save_eval_result(
        result, make_eval_trace(result_id=result.id, eval_run_id=run.id, call_ledger=_ledger(emitted))
    )
    return run.id, result


def test_an_unknown_run_is_refused() -> None:
    storage, _ = memory_storage()

    with pytest.raises(NotFoundError):
        recheck_goal_states(storage, "no-such-run", "uni-1", apply=False)


@pytest.mark.parametrize("status", ["pending", "running"])
def test_a_run_still_being_written_is_refused(status: str) -> None:
    storage, _ = memory_storage()
    run_id, _ = _stored_run(storage, status=status)

    with pytest.raises(ValidationFailedError, match=f"is {status}"):
        recheck_goal_states(storage, run_id, "uni-1", apply=True)


def test_a_rewrite_something_else_raced_is_named_and_not_stored() -> None:
    storage, _ = memory_storage()
    run_id, result = _stored_run(storage)

    class _RacedStorage:
        """The run's store, whose every conditional rewrite loses a race."""

        def __getattr__(self, name: str) -> Any:
            return getattr(storage, name)

        def replace_eval_result(self, result: EvalResult, /, *, if_match: str | None) -> None:
            raise ConflictError(f"eval_result {result.id} was written since it was read")

    recheck = recheck_goal_states(_RacedStorage(), run_id, "uni-1", apply=True)  # type: ignore[arg-type]

    assert recheck.write_conflicts == [result.id]
    stored = storage.load_eval_result(result.id, "uni-1")
    assert stored is not None and stored.goal_state_outcomes == [_every_field(True)]
