"""Re-grading a stored run's goal checks from the call ledgers its cells stored.

The toy host's kind records its extractor's calls through the engine's ``CallLedger`` and grades
the template's goal check through ``grade_goal_checks``; the runner stores the ledger on each cell's
trace. A re-check reads that ledger back and re-grades through the same function, so it must
reproduce every stored verdict, and must move exactly the verdict whose stored ledger moved.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.kernel import NOT_ESTABLISHED, ConflictError, EvalStorage, NotFoundError, ValidationFailedError
from threetears.evals.schema import CallLedger, EvalResult, GoalStateOutcome, WorldEvent
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
from packages.evals.tests.fixtures.toyhost.run import (
    EVERY_FIELD_EMITTED,
    HOLD_FIRED,
    HOLD_IN_FORCE,
    ToyhostRunPath,
    execute_toyhost_run,
    toyhost_arming_template,
)


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

    recheck = recheck_goal_states(path.host.storage, run.id, run.scope_id, world=path.host.profile.world, apply=False)

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

    dry = recheck_goal_states(path.host.storage, run.id, run.scope_id, world=path.host.profile.world, apply=False)
    flipped = {report.result_id: report.flips for report in dry.results if report.flips}
    assert list(flipped) == [target.id]
    (flip,) = flipped[target.id]
    assert (flip.expression, flip.was, flip.now) == (EVERY_FIELD_EMITTED, True, False)
    assert path.host.storage.load_eval_result(target.id, run.scope_id) == target, "a dry run wrote"

    applied = recheck_goal_states(path.host.storage, run.id, run.scope_id, world=path.host.profile.world, apply=True)
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

    again = recheck_goal_states(path.host.storage, run.id, run.scope_id, world=path.host.profile.world, apply=True)
    assert not any(report.flips for report in again.results), "an applied re-check is not idempotent"


# =============================================================================
# The world half: a stored end state re-grades state.*, stored world events re-grade fired()
# =============================================================================


async def _arming_run() -> ToyhostRunPath:
    return await execute_toyhost_run(host=toyhost_host(), template=toyhost_arming_template())


async def test_re_grading_a_world_reading_run_reproduces_every_verdict_from_what_it_stored() -> None:
    path = await _arming_run()
    run = path.runs[0]

    recheck = recheck_goal_states(path.host.storage, run.id, run.scope_id, world=path.host.profile.world, apply=False)

    assert recheck.results
    for report in recheck.results:
        assert report.not_rechecked is None
        assert report.flips == []
        # The ledger check, the fired check and the end-state check: every goal-language outcome is re-graded.
        assert report.regraded == 3
        assert [kept.expression.split(" ")[0] for kept in report.kept_as_stored] == [FIELD_ACCURACY]


async def test_mutating_a_stored_end_state_flips_exactly_the_check_that_reads_it() -> None:
    path = await _arming_run()
    run = path.runs[0]
    target, *others = path.host.storage.query_eval_results_by_run(run.id, run.scope_id)
    trace = path.trace(target)
    assert trace is not None and trace.end_state is not None
    released = {**trace.end_state, "payment_hold": "released"}
    path.host.storage.save_eval_result(target, trace.model_copy(update={"end_state": released}))

    recheck = recheck_goal_states(path.host.storage, run.id, run.scope_id, world=path.host.profile.world, apply=True)

    flipped = {report.result_id: report.flips for report in recheck.results if report.flips}
    assert list(flipped) == [target.id]
    assert [(flip.expression, flip.was, flip.now) for flip in flipped[target.id]] == [(HOLD_IN_FORCE, True, False)]
    rewritten = path.host.storage.load_eval_result(target.id, run.scope_id)
    assert rewritten is not None
    assert {o.expression: o.passed for o in rewritten.goal_state_outcomes}[HOLD_IN_FORCE] is False
    assert all(path.host.storage.load_eval_result(other.id, run.scope_id) == other for other in others)


async def test_mutating_stored_world_events_flips_exactly_the_check_that_reads_them() -> None:
    path = await _arming_run()
    run = path.runs[0]
    target = path.host.storage.query_eval_results_by_run(run.id, run.scope_id)[0]
    trace = path.trace(target)
    path.host.storage.save_eval_result(target.model_copy(update={"world_events": []}), trace)

    recheck = recheck_goal_states(path.host.storage, run.id, run.scope_id, world=path.host.profile.world, apply=False)

    flipped = {report.result_id: report.flips for report in recheck.results if report.flips}
    assert [(flip.expression, flip.was, flip.now) for flip in flipped[target.id]] == [(HOLD_FIRED, True, False)]


def test_world_reading_checks_are_kept_when_what_they_read_was_not_stored() -> None:
    state_check = GoalStateOutcome(expression=HOLD_IN_FORCE, passed=True)
    fired_check = GoalStateOutcome(expression=HOLD_FIRED, passed=True)
    result = make_eval_result(goal_state_outcomes=[state_check, fired_check], world_events=None)

    report, outcomes = recheck_result(
        result, ledger=None, end_state={"payment_hold": "released"}, world=None, variation={}, provenance="commissioned"
    )

    assert outcomes is None and report.regraded == 0
    reasons = {kept.expression: kept.reason for kept in report.kept_as_stored}
    assert "given no world registry" in reasons[HOLD_IN_FORCE]
    assert "recorded no world events" in reasons[HOLD_FIRED]


# =============================================================================
# One result: what is re-graded, what is kept, what is not re-checked
# =============================================================================


def test_a_result_whose_checks_were_not_graded_at_the_cell_s_end_is_not_rechecked() -> None:
    result = make_eval_result(goal_state_outcomes=[_every_field(False)], termination="cell_timeout")

    report, outcomes = recheck_result(
        result, ledger=_ledger(4), end_state=None, world=None, variation={}, provenance="commissioned"
    )

    assert outcomes is None
    assert report.not_rechecked is not None and "termination=cell_timeout" in report.not_rechecked


def test_a_ledger_check_on_a_result_with_no_stored_ledger_is_kept_as_stored() -> None:
    report, outcomes = recheck_result(
        make_eval_result(goal_state_outcomes=[_every_field(True)]),
        ledger=None,
        end_state=None,
        world=None,
        variation={},
        provenance="commissioned",
    )

    assert outcomes is None
    assert report.not_rechecked is None and report.regraded == 0
    (kept,) = report.kept_as_stored
    assert kept.expression == EVERY_FIELD_EMITTED and "no call ledger is stored" in kept.reason


def test_a_result_with_no_goal_outcomes_is_not_rechecked() -> None:
    report, outcomes = recheck_result(
        make_eval_result(goal_state_outcomes=[]),
        ledger=_ledger(4),
        end_state=None,
        world=None,
        variation={},
        provenance="commissioned",
    )

    assert outcomes is None
    assert report.not_rechecked == "it carries no goal-state outcomes"


def test_a_check_reading_world_state_is_kept_as_stored_when_its_end_state_was_not() -> None:
    world_check = GoalStateOutcome(expression="state.shop.cart.length >= 1", passed=True)
    result = make_eval_result(goal_state_outcomes=[world_check, _every_field(True)])

    report, outcomes = recheck_result(
        result, ledger=_ledger(3), end_state=None, world=None, variation={}, provenance="commissioned"
    )

    assert [kept.expression for kept in report.kept_as_stored] == [world_check.expression]
    assert "end state was not stored" in report.kept_as_stored[0].reason
    assert outcomes is not None and outcomes[0] == world_check
    assert outcomes[1].passed is False


@pytest.mark.parametrize(("variation", "flips"), [({"wanted": 4}, False), ({"wanted": 3}, True)])
def test_a_check_reading_the_case_is_re_graded_under_the_case_s_variation(
    variation: dict[str, Any], flips: bool
) -> None:
    check = GoalStateOutcome(expression='call_count("extractor.emit_field") == variation.wanted', passed=True)

    report, _ = recheck_result(
        make_eval_result(goal_state_outcomes=[check]),
        ledger=_ledger(4),
        end_state=None,
        world=None,
        variation=variation,
        provenance="commissioned",
    )

    assert bool(report.flips) is flips


def test_a_check_reading_the_case_is_kept_when_its_test_case_no_longer_resolves() -> None:
    check = GoalStateOutcome(expression='call_count("extractor.emit_field") == variation.wanted', passed=True)

    report, outcomes = recheck_result(
        make_eval_result(goal_state_outcomes=[check]),
        ledger=_ledger(4),
        end_state=None,
        world=None,
        variation=None,
        provenance="commissioned",
    )

    assert outcomes is None
    assert report.regraded == 0
    assert "test case no longer resolves" in report.kept_as_stored[0].reason


def test_a_check_that_no_longer_evaluates_stops_the_result_s_recheck() -> None:
    broken = GoalStateOutcome(expression='call_count("extractor.emit_field") < "four"', passed=True)

    report, outcomes = recheck_result(
        make_eval_result(goal_state_outcomes=[broken]),
        ledger=_ledger(4),
        end_state=None,
        world=None,
        variation={},
        provenance="commissioned",
    )

    assert outcomes is None
    assert report.not_rechecked is not None and "no longer evaluates" in report.not_rechecked


# =============================================================================
# What a cell can establish: a witnessed cell's fired_armed stays not established on re-check
# =============================================================================

_ARMED = 'fired_armed("payment_hold")'
_NOT_ARMED = 'not fired_armed("payment_hold")'
_ANY_FIRING = 'fired("payment_hold")'
_NOT_ESTABLISHED_DETAIL = f"{NOT_ESTABLISHED}: fired_armed('payment_hold') resolved to nothing"


def _stored_firing_run(storage: EvalStorage, provenance: str, outcomes: list[GoalStateOutcome]) -> tuple[str, str]:
    """A finished run of ``provenance`` whose one cell recorded the world's own firing of the payment hold.

    The event says ``armed=False``, as every event of a witnessed cell does — no seed could mark it armed.
    """
    run = make_eval_run(
        status="completed",
        apparatus_provenance=provenance,
        **({"template_id": None} if provenance == "witnessed" else {}),
    )
    storage.save_eval_run(run)
    storage.save_test_case(make_test_case(id="tc-1", template_id=None if provenance == "witnessed" else "tpl-1"))
    firing = WorldEvent(
        kind="event", dimension="payment_hold", condition="extraction_posted", caused_by="world", event="posting-hold"
    )
    result = make_eval_result(eval_run_id=run.id, goal_state_outcomes=outcomes, world_events=[firing])
    storage.save_eval_result(result, None)
    return run.id, result.id


def test_a_witnessed_cell_s_fired_armed_checks_stay_not_established_negated_or_not() -> None:
    """The defect: its events all say armed=False, and re-reading them flipped ``not fired_armed`` to a pass."""
    storage, _ = memory_storage()
    graded = [
        GoalStateOutcome(expression=_ARMED, passed=False, detail=_NOT_ESTABLISHED_DETAIL),
        GoalStateOutcome(expression=_NOT_ARMED, passed=False, detail=_NOT_ESTABLISHED_DETAIL),
        GoalStateOutcome(expression=_ANY_FIRING, passed=True, detail="True"),
    ]
    run_id, result_id = _stored_firing_run(storage, "witnessed", graded)

    recheck = recheck_goal_states(storage, run_id, "uni-1", world=None, apply=True)

    (report,) = recheck.results
    assert report.not_rechecked is None and report.regraded == 3
    assert report.flips == []
    stored = storage.load_eval_result(result_id, "uni-1")
    assert stored is not None and stored.goal_state_outcomes == graded


def test_a_witnessed_cell_s_stale_pass_on_not_fired_armed_is_re_graded_as_not_established() -> None:
    storage, _ = memory_storage()
    run_id, result_id = _stored_firing_run(
        storage, "witnessed", [GoalStateOutcome(expression=_NOT_ARMED, passed=True, detail="True")]
    )

    recheck = recheck_goal_states(storage, run_id, "uni-1", world=None, apply=True)

    assert [(flip.expression, flip.was, flip.now) for flip in recheck.results[0].flips] == [(_NOT_ARMED, True, False)]
    stored = storage.load_eval_result(result_id, "uni-1")
    assert stored is not None
    (outcome,) = stored.goal_state_outcomes
    assert outcome.detail == _NOT_ESTABLISHED_DETAIL


def test_a_launched_cell_s_fired_armed_checks_re_grade_from_its_events() -> None:
    """The same events under a commissioned run say the seed's event did not fire — a verdict, so it moves."""
    storage, _ = memory_storage()
    run_id, result_id = _stored_firing_run(
        storage,
        "commissioned",
        [
            GoalStateOutcome(expression=_ARMED, passed=True, detail="True"),
            GoalStateOutcome(expression=_NOT_ARMED, passed=False, detail="False"),
        ],
    )

    recheck = recheck_goal_states(storage, run_id, "uni-1", world=None, apply=True)

    assert [(flip.expression, flip.was, flip.now) for flip in recheck.results[0].flips] == [
        (_ARMED, True, False),
        (_NOT_ARMED, False, True),
    ]
    stored = storage.load_eval_result(result_id, "uni-1")
    assert stored is not None
    assert [(o.passed, o.detail) for o in stored.goal_state_outcomes] == [(False, "False"), (True, "True")]


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
        recheck_goal_states(storage, "no-such-run", "uni-1", world=None, apply=False)


@pytest.mark.parametrize("status", ["pending", "running"])
def test_a_run_still_being_written_is_refused(status: str) -> None:
    storage, _ = memory_storage()
    run_id, _ = _stored_run(storage, status=status)

    with pytest.raises(ValidationFailedError, match=f"is {status}"):
        recheck_goal_states(storage, run_id, "uni-1", world=None, apply=True)


def test_a_rewrite_something_else_raced_is_named_and_not_stored() -> None:
    storage, _ = memory_storage()
    run_id, result = _stored_run(storage)

    class _RacedStorage:
        """The run's store, whose every conditional rewrite loses a race."""

        def __getattr__(self, name: str) -> Any:
            return getattr(storage, name)

        def replace_eval_result(self, result: EvalResult, /, *, if_match: str | None) -> None:
            raise ConflictError(f"eval_result {result.id} was written since it was read")

    recheck = recheck_goal_states(_RacedStorage(), run_id, "uni-1", world=None, apply=True)  # type: ignore[arg-type]

    assert recheck.write_conflicts == [result.id]
    stored = storage.load_eval_result(result.id, "uni-1")
    assert stored is not None and stored.goal_state_outcomes == [_every_field(True)]


def test_a_goal_check_today_s_language_refuses_is_kept_and_named_as_refused() -> None:
    """A check stored under a looser rule is not mistaken for a kind's own fact — it is named as refused today."""
    stored = GoalStateOutcome(expression="call_count(variation.spec) == 4", passed=True)

    report, outcomes = recheck_result(
        make_eval_result(goal_state_outcomes=[stored]),
        ledger=_ledger(4),
        end_state=None,
        world=None,
        variation={},
        provenance="commissioned",
    )

    assert outcomes is None and report.not_rechecked is None
    (kept,) = report.kept_as_stored
    assert "today's language refuses" in kept.reason and "never a computed spec" in kept.reason


# =============================================================================
# The vocabulary moved: a renamed or removed dimension keeps its stored verdict
# =============================================================================


def _toy_registry() -> Any:
    from packages.evals.tests.fixtures.toyhost.world import toyhost_world

    registry, _state = toyhost_world()
    return registry


def test_a_check_over_a_dimension_today_s_world_no_longer_declares_keeps_its_stored_pass() -> None:
    """Renamed or removed since the cell was graded: re-grading would read Missing and flip a pass to a fail."""
    stored = GoalStateOutcome(expression='state.payment_hold_v1 == "held"', passed=True)

    report, outcomes = recheck_result(
        make_eval_result(goal_state_outcomes=[stored]),
        ledger=None,
        end_state={"payment_hold_v1": "held"},
        world=_toy_registry(),
        variation={},
        provenance="commissioned",
    )

    assert outcomes is None and report.flips == [] and report.regraded == 0
    (kept,) = report.kept_as_stored
    assert "names no dimension today's world declares" in kept.reason


def test_a_check_whose_path_today_s_world_resolves_to_another_dimension_keeps_its_stored_pass() -> None:
    """The cell stored ``inbox.messages``; today's world declares only ``inbox``, which the end state never held."""
    from threetears.evals.kernel.host import WorldDimension, WorldRegistry

    today = WorldRegistry(
        [
            WorldDimension(
                name="inbox",
                carrier="mail",
                schema={"type": "object"},
                matters="the inbox the subject triages",
                seed="mail.seed",
                read="mail.read",
            )
        ],
        bindings={"mail.seed": lambda value: None, "mail.read": dict},
    )
    stored = GoalStateOutcome(expression="state.inbox.messages.length >= 1", passed=True)

    report, outcomes = recheck_result(
        make_eval_result(goal_state_outcomes=[stored]),
        ledger=None,
        end_state={"inbox.messages": ["hello"]},
        world=today,
        variation={},
        provenance="commissioned",
    )

    assert outcomes is None and report.flips == []
    (kept,) = report.kept_as_stored
    assert "stored under inbox.messages" in kept.reason and "resolves to inbox" in kept.reason


def test_a_negated_check_over_a_dimension_its_cell_never_held_is_re_graded_as_not_established() -> None:
    """Not a vocabulary change: today's world resolves the dimension and the cell never held it under any name.

    A negated comparison over that absent value was once stored as a pass; the re-check is how the
    three-valued rule reaches it, and it must flip to a fail that says it was not established.
    """
    stored = GoalStateOutcome(expression='not state.payment_hold == "released"', passed=True)

    report, outcomes = recheck_result(
        make_eval_result(goal_state_outcomes=[stored]),
        ledger=None,
        end_state={"document_language": "de"},
        world=_toy_registry(),
        variation={},
        provenance="commissioned",
    )

    assert [(flip.was, flip.now) for flip in report.flips] == [(True, False)]
    assert outcomes is not None
    assert outcomes[0].detail.startswith("not established: state.payment_hold")
