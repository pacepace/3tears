"""Re-grading a stored result's check dims from what its cell stored, through the one door: ``regrade``.

The toy host's kind records its extractor's calls through the engine's ``CallLedger`` and grades the template's goal
check through ``grade_goal_checks``; the runner stores the ledger on each cell's trace. A re-grade reads that ledger
back and re-grades through the same function, so it must reproduce every stored verdict, must move exactly the
verdict whose stored input moved, and must write only when told to. Ported from the re-check's tests (``.26``), whose
behaviour ``regrade`` keeps; its judged half is held in ``test_rejudge_reads_stored_evidence.py``, beside the
re-judge it asks through.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from threetears.evals.kernel import NOT_ESTABLISHED, ConflictError, EvalStorage, NotFoundError, ValidationFailedError
from threetears.evals.kernel.host import EvalHost, WorldDimension, WorldRegistry
from threetears.evals.run import Regrade, regrade
from threetears.evals.schema import CallLedger, EvalResult, GoalStateOutcome, JobLease, WorldEvent
from threetears.evals.schema.models import TERMINAL_RUN_STATUSES
from packages.evals.tests.factories import (
    make_eval_result,
    make_eval_run,
    make_eval_trace,
    make_test_case,
    memory_storage,
)
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.kind import FIELD_ACCURACY
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.run import (
    EVERY_FIELD_EMITTED,
    HOLD_FIRED,
    HOLD_IN_FORCE,
    ToyhostRunPath,
    execute_toyhost_run,
    toyhost_arming_template,
)
from packages.evals.tests.off_loop import off_loop

SCOPE = "uni-1"


def _every_field(passed: bool) -> GoalStateOutcome:
    return GoalStateOutcome(expression=EVERY_FIELD_EMITTED, passed=passed)


def _ledger(emitted: int) -> CallLedger:
    ledger = CallLedger()
    for index in range(emitted):
        ledger.record("extractor", "emit_field", {"field": f"f{index}"})
    return ledger


def _flips(report: Regrade) -> list[tuple[str, Any, Any]]:
    return [(change.dim, change.was, change.now) for change in report.changes]


# =============================================================================
# The toy host, end to end: stored verdicts reproduce, a moved ledger moves its verdict, and only apply writes
# =============================================================================


async def _toy_run(template: Any = None) -> ToyhostRunPath:
    if template is None:
        return await execute_toyhost_run(host=toyhost_host())
    return await execute_toyhost_run(host=toyhost_host(), template=template)


def _results(path: ToyhostRunPath) -> list[EvalResult]:
    run = path.runs[0]
    return off_loop(path.host.storage.query_eval_results_by_run, run.id, run.scope_id)


async def test_re_grading_every_stored_toy_result_reproduces_its_verdict() -> None:
    path = await _toy_run()
    results = _results(path)
    assert results, "the run stored no results, so nothing was re-graded"

    for result in results:
        report = await regrade(path.host, result.id, result.scope_id)

        assert report.checks_not_regraded is None and report.changes == []
        assert report.checks_regraded == 1
        # The kind's own fact names no root the goal language admits, so it is left as stored.
        assert [kept.split(" ")[0] for kept in report.kept_as_stored] == [FIELD_ACCURACY]
        assert report.judges_not_regraded is not None and report.judge_dims == []


async def _drop_last_call(path: ToyhostRunPath, result: EvalResult) -> None:
    """Rewrite one stored trace with the last call of its ledger removed."""
    trace = await path.trace(result)
    assert trace is not None and trace.call_ledger is not None
    mutated = trace.call_ledger.model_copy(update={"calls": trace.call_ledger.calls[:-1]})
    off_loop(path.host.storage.save_eval_result, result, trace.model_copy(update={"call_ledger": mutated}))


async def test_a_moved_ledger_flips_exactly_that_result_and_only_apply_stores_it() -> None:
    path = await _toy_run()
    target, *others = _results(path)
    await _drop_last_call(path, target)
    scope = target.scope_id

    dry = await regrade(path.host, target.id, scope)

    assert _flips(dry) == [(EVERY_FIELD_EMITTED, True, False)] and dry.applied is False
    assert off_loop(path.host.storage.load_eval_result, target.id, scope) == target, "a report wrote"
    for other in others:
        assert (await regrade(path.host, other.id, scope)).changes == []

    applied = await regrade(path.host, target.id, scope, apply=True)

    assert _flips(applied) == [(EVERY_FIELD_EMITTED, True, False)] and applied.applied
    rewritten = off_loop(path.host.storage.load_eval_result, target.id, scope)
    assert rewritten is not None
    assert {o.expression: o.passed for o in rewritten.goal_state_outcomes}[EVERY_FIELD_EMITTED] is False
    assert [o for o in rewritten.goal_state_outcomes if o.expression != EVERY_FIELD_EMITTED] == [
        o for o in target.goal_state_outcomes if o.expression != EVERY_FIELD_EMITTED
    ]
    assert all(off_loop(path.host.storage.load_eval_result, o.id, scope) == o for o in others), "an unmoved write"
    assert (await regrade(path.host, target.id, scope, apply=True)).changes == [], "an applied re-grade is idempotent"


async def test_checks_false_re_grades_no_check() -> None:
    path = await _toy_run()
    target = _results(path)[0]
    await _drop_last_call(path, target)

    report = await regrade(path.host, target.id, target.scope_id, checks=False, apply=True)

    assert report.changes == [] and report.checks_not_regraded == "checks were not asked for (checks=False)"
    assert off_loop(path.host.storage.load_eval_result, target.id, target.scope_id) == target


# =============================================================================
# The world half: a stored end state re-grades state.*, stored world events re-grade fired()
# =============================================================================


async def test_re_grading_a_world_reading_result_reproduces_every_verdict_from_what_it_stored() -> None:
    path = await _toy_run(toyhost_arming_template())

    for result in _results(path):
        report = await regrade(path.host, result.id, result.scope_id)

        assert report.checks_not_regraded is None and report.changes == []
        # The ledger check, the fired check and the end-state check: every goal-language outcome is re-graded.
        assert report.checks_regraded == 3
        assert [kept.split(" ")[0] for kept in report.kept_as_stored] == [FIELD_ACCURACY]


async def test_a_moved_end_state_flips_exactly_the_check_that_reads_it() -> None:
    path = await _toy_run(toyhost_arming_template())
    target = _results(path)[0]
    trace = await path.trace(target)
    assert trace is not None and trace.end_state is not None
    released = {**trace.end_state, "payment_hold": "released"}
    off_loop(path.host.storage.save_eval_result, target, trace.model_copy(update={"end_state": released}))

    report = await regrade(path.host, target.id, target.scope_id, apply=True)

    assert _flips(report) == [(HOLD_IN_FORCE, True, False)]
    rewritten = off_loop(path.host.storage.load_eval_result, target.id, target.scope_id)
    assert rewritten is not None
    assert {o.expression: o.passed for o in rewritten.goal_state_outcomes}[HOLD_IN_FORCE] is False


async def test_moved_world_events_flip_exactly_the_check_that_reads_them() -> None:
    path = await _toy_run(toyhost_arming_template())
    target = _results(path)[0]
    trace = await path.trace(target)
    off_loop(path.host.storage.save_eval_result, target.model_copy(update={"world_events": []}), trace)

    report = await regrade(path.host, target.id, target.scope_id)

    assert _flips(report) == [(HOLD_FIRED, True, False)]


# =============================================================================
# One stored result: what is re-graded, what is kept, what is not re-graded
# =============================================================================


def _store(
    storage: EvalStorage,
    outcomes: list[GoalStateOutcome],
    *,
    ledger: CallLedger | None = None,
    end_state: dict[str, Any] | None = None,
    variation: dict[str, Any] | None = None,
    provenance: str = "commissioned",
    status: str = "completed",
    **result_fields: Any,
) -> EvalResult:
    """A finished run's one stored result, its trace holding ``ledger`` and ``end_state`` (none when both are None),
    and its test case holding ``variation`` (no test case when None)."""
    # A run not yet ended is still being written: its process holds a lease it renews.
    live = JobLease(holder="the-writer", expires_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat())
    witnessed = provenance == "witnessed"
    run = make_eval_run(
        status=status,
        apparatus_provenance=provenance,
        lease=None if status in TERMINAL_RUN_STATUSES else live,
        **({"template_id": None} if witnessed else {}),
    )
    off_loop(storage.save_eval_run, run)
    if variation is not None:
        case = make_test_case(id="tc-1", template_id=None if witnessed else "tpl-1", variation_params=variation)
        off_loop(storage.save_test_case, case)
    result = make_eval_result(eval_run_id=run.id, goal_state_outcomes=outcomes, **result_fields)
    stored = ledger is not None or end_state is not None
    trace = make_eval_trace(result_id=result.id, eval_run_id=run.id, call_ledger=ledger, end_state=end_state)
    off_loop(storage.save_eval_result, result, trace if stored else None)
    loaded = off_loop(storage.load_eval_result, result.id, SCOPE)
    assert loaded is not None
    return loaded


def _host(storage: EvalStorage, *, world: bool = True) -> EvalHost:
    profile = toyhost_profile()
    return toyhost_host(storage=storage, profile=profile if world else dataclasses.replace(profile, world=None))


async def test_a_result_whose_checks_were_not_graded_at_the_cells_end_is_not_re_graded() -> None:
    storage, _ = memory_storage()
    result = _store(storage, [_every_field(False)], ledger=_ledger(4), variation={}, termination="cell_timeout")

    report = await regrade(_host(storage), result.id, SCOPE, apply=True)

    assert report.changes == [] and "termination=cell_timeout" in (report.checks_not_regraded or "")
    assert off_loop(storage.load_eval_result, result.id, SCOPE) == result


async def test_a_result_with_no_goal_outcomes_is_not_re_graded() -> None:
    storage, _ = memory_storage()
    result = _store(storage, [], ledger=_ledger(4), variation={})

    report = await regrade(_host(storage), result.id, SCOPE)

    assert report.checks_not_regraded == "it carries no goal-state outcomes"


async def test_a_ledger_check_on_a_result_with_no_stored_ledger_is_kept_as_stored() -> None:
    storage, _ = memory_storage()
    result = _store(storage, [_every_field(True)], variation={})

    report = await regrade(_host(storage), result.id, SCOPE)

    assert report.changes == [] and report.checks_regraded == 0
    assert "no call ledger is stored" in report.kept_as_stored[EVERY_FIELD_EMITTED]


async def test_world_reading_checks_are_kept_when_what_they_read_was_not_stored() -> None:
    storage, _ = memory_storage()
    outcomes = [
        GoalStateOutcome(expression=HOLD_IN_FORCE, passed=True),
        GoalStateOutcome(expression=HOLD_FIRED, passed=True),
    ]
    result = _store(storage, outcomes, end_state={"payment_hold": "released"}, variation={}, world_events=None)

    report = await regrade(_host(storage, world=False), result.id, SCOPE)

    assert report.changes == [] and report.checks_regraded == 0
    assert "declares no world registry" in report.kept_as_stored[HOLD_IN_FORCE]
    assert "recorded no world events" in report.kept_as_stored[HOLD_FIRED]


async def test_a_check_reading_world_state_is_kept_as_stored_when_its_end_state_was_not() -> None:
    storage, _ = memory_storage()
    world_check = GoalStateOutcome(expression="state.shop.cart.length >= 1", passed=True)
    result = _store(storage, [world_check, _every_field(True)], ledger=_ledger(3), variation={})

    report = await regrade(_host(storage), result.id, SCOPE, apply=True)

    assert list(report.kept_as_stored) == [world_check.expression]
    assert "end state was not stored" in report.kept_as_stored[world_check.expression]
    assert _flips(report) == [(EVERY_FIELD_EMITTED, True, False)]
    stored = off_loop(storage.load_eval_result, result.id, SCOPE)
    assert stored is not None and stored.goal_state_outcomes[0] == world_check
    assert stored.goal_state_outcomes[1].passed is False


@pytest.mark.parametrize(("variation", "flips"), [({"language": "de"}, False), ({"language": "fr"}, True)])
async def test_a_check_reading_the_case_is_re_graded_under_the_cases_variation(
    variation: dict[str, Any], flips: bool
) -> None:
    storage, _ = memory_storage()
    check = GoalStateOutcome(expression="state.document_language == variation.language", passed=True)
    result = _store(storage, [check], end_state={"document_language": "de"}, variation=variation)

    report = await regrade(_host(storage), result.id, SCOPE)

    assert report.checks_regraded == 1 and bool(report.changes) is flips


async def test_a_check_reading_the_case_is_kept_when_its_test_case_no_longer_resolves() -> None:
    storage, _ = memory_storage()
    check = GoalStateOutcome(expression='call_count("extractor.emit_field") == variation.wanted', passed=True)
    result = _store(storage, [check], ledger=_ledger(4), variation=None)

    report = await regrade(_host(storage), result.id, SCOPE)

    assert report.checks_regraded == 0
    assert "test case no longer resolves" in report.kept_as_stored[check.expression]


async def test_a_check_that_no_longer_evaluates_stops_the_results_re_grade() -> None:
    storage, _ = memory_storage()
    broken = GoalStateOutcome(expression='call_count("extractor.emit_field") < "four"', passed=True)
    result = _store(storage, [broken], ledger=_ledger(4), variation={})

    report = await regrade(_host(storage), result.id, SCOPE, apply=True)

    assert report.changes == [] and "no longer evaluates" in (report.checks_not_regraded or "")
    assert off_loop(storage.load_eval_result, result.id, SCOPE) == result


async def test_a_goal_check_todays_language_refuses_is_kept_and_named_as_refused() -> None:
    """A check stored under a looser rule is not mistaken for a kind's own fact: it is named as refused today."""
    storage, _ = memory_storage()
    stored = GoalStateOutcome(expression="call_count(variation.spec) == 4", passed=True)
    result = _store(storage, [stored], ledger=_ledger(4), variation={})

    report = await regrade(_host(storage), result.id, SCOPE)

    assert report.changes == [] and report.checks_not_regraded is None
    reason = report.kept_as_stored[stored.expression]
    assert "today's language refuses" in reason and "never a computed spec" in reason


# =============================================================================
# What a cell can establish: a witnessed cell's fired_armed stays not established on a re-grade
# =============================================================================

_ARMED = 'fired_armed("payment_hold")'
_NOT_ARMED = 'not fired_armed("payment_hold")'
_ANY_FIRING = 'fired("payment_hold")'
_NOT_ESTABLISHED_DETAIL = f"{NOT_ESTABLISHED}: fired_armed('payment_hold') resolved to nothing"
#: The world's own firing of the payment hold, which says ``armed=False``, as every event of a witnessed cell does.
_FIRING = WorldEvent(
    kind="event", dimension="payment_hold", condition="extraction_posted", caused_by="world", event="posting-hold"
)


async def test_a_witnessed_cells_fired_armed_checks_stay_not_established_negated_or_not() -> None:
    """The defect: its events all say armed=False, and re-reading them flipped ``not fired_armed`` to a pass."""
    storage, _ = memory_storage()
    graded = [
        GoalStateOutcome(expression=_ARMED, passed=False, detail=_NOT_ESTABLISHED_DETAIL),
        GoalStateOutcome(expression=_NOT_ARMED, passed=False, detail=_NOT_ESTABLISHED_DETAIL),
        GoalStateOutcome(expression=_ANY_FIRING, passed=True, detail="True"),
    ]
    result = _store(storage, graded, variation={}, provenance="witnessed", world_events=[_FIRING])

    report = await regrade(_host(storage), result.id, SCOPE, apply=True)

    assert report.checks_not_regraded is None and report.checks_regraded == 3 and report.changes == []
    stored = off_loop(storage.load_eval_result, result.id, SCOPE)
    assert stored is not None and stored.goal_state_outcomes == graded


async def test_a_witnessed_cells_stale_pass_on_not_fired_armed_is_re_graded_as_not_established() -> None:
    storage, _ = memory_storage()
    stale = [GoalStateOutcome(expression=_NOT_ARMED, passed=True, detail="True")]
    result = _store(storage, stale, variation={}, provenance="witnessed", world_events=[_FIRING])

    report = await regrade(_host(storage), result.id, SCOPE, apply=True)

    assert _flips(report) == [(_NOT_ARMED, True, False)]
    stored = off_loop(storage.load_eval_result, result.id, SCOPE)
    assert stored is not None
    (outcome,) = stored.goal_state_outcomes
    assert outcome.detail == _NOT_ESTABLISHED_DETAIL


async def test_a_launched_cells_fired_armed_checks_re_grade_from_its_events() -> None:
    """The same events under a commissioned run say the seed's event did not fire: a verdict, so it moves."""
    storage, _ = memory_storage()
    graded = [
        GoalStateOutcome(expression=_ARMED, passed=True, detail="True"),
        GoalStateOutcome(expression=_NOT_ARMED, passed=False, detail="False"),
    ]
    result = _store(storage, graded, variation={}, world_events=[_FIRING])

    report = await regrade(_host(storage), result.id, SCOPE, apply=True)

    assert _flips(report) == [(_ARMED, True, False), (_NOT_ARMED, False, True)]
    stored = off_loop(storage.load_eval_result, result.id, SCOPE)
    assert stored is not None
    assert [(o.passed, o.detail) for o in stored.goal_state_outcomes] == [(False, "False"), (True, "True")]


# =============================================================================
# Refusals, and a rewrite something else raced
# =============================================================================


async def test_an_unknown_result_is_refused() -> None:
    storage, _ = memory_storage()

    with pytest.raises(NotFoundError):
        await regrade(_host(storage), "no-such-result", SCOPE)


@pytest.mark.parametrize("status", ["pending", "running"])
async def test_a_result_of_a_run_still_being_written_is_refused(status: str) -> None:
    storage, _ = memory_storage()
    result = _store(storage, [_every_field(True)], ledger=_ledger(3), variation={}, status=status)

    with pytest.raises(ValidationFailedError, match=f"is {status}"):
        await regrade(_host(storage), result.id, SCOPE, apply=True)


async def test_a_rewrite_something_else_raced_is_refused_and_nothing_is_stored() -> None:
    storage, _ = memory_storage()
    result = _store(storage, [_every_field(True)], ledger=_ledger(3), variation={})

    class _RacedStorage:
        """The result's store, whose every conditional rewrite loses a race."""

        def __getattr__(self, name: str) -> Any:
            return getattr(storage, name)

        def replace_eval_result(self, result: EvalResult, /, *, if_match: str | None) -> None:
            raise ConflictError(f"eval_result {result.id} was written since it was read", remedy="try again")

    host = dataclasses.replace(_host(storage), storage=_RacedStorage())  # type: ignore[arg-type]

    with pytest.raises(ConflictError, match="changed while it was being re-graded; nothing was stored"):
        await regrade(host, result.id, SCOPE, apply=True)
    stored = off_loop(storage.load_eval_result, result.id, SCOPE)
    assert stored is not None and stored.goal_state_outcomes == [_every_field(True)]


# =============================================================================
# The vocabulary moved: a renamed or removed dimension keeps its stored verdict
# =============================================================================


async def test_a_check_over_a_dimension_todays_world_no_longer_declares_keeps_its_stored_pass() -> None:
    """Renamed or removed since the cell was graded: re-grading would read Missing and flip a pass to a fail."""
    storage, _ = memory_storage()
    stored = GoalStateOutcome(expression='state.payment_hold_v1 == "held"', passed=True)
    result = _store(storage, [stored], end_state={"payment_hold_v1": "held"}, variation={})

    report = await regrade(_host(storage), result.id, SCOPE)

    assert report.changes == [] and report.checks_regraded == 0
    assert "names no dimension today's world declares" in report.kept_as_stored[stored.expression]


async def test_a_check_whose_path_todays_world_resolves_to_another_dimension_keeps_its_stored_pass() -> None:
    """The cell stored ``inbox.messages``; today's world declares only ``inbox``, which the end state never held."""
    storage, _ = memory_storage()
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
    result = _store(storage, [stored], end_state={"inbox.messages": ["hello"]}, variation={})
    host = toyhost_host(storage=storage, profile=dataclasses.replace(toyhost_profile(), world=today))

    report = await regrade(host, result.id, SCOPE)

    assert report.changes == []
    reason = report.kept_as_stored[stored.expression]
    assert "stored under inbox.messages" in reason and "resolves to inbox" in reason


async def test_a_negated_check_over_a_dimension_its_cell_never_held_is_re_graded_as_not_established() -> None:
    """Not a vocabulary change: today's world resolves the dimension and the cell never held it under any name.

    A negated comparison over that absent value was once stored as a pass; the re-grade is how the three-valued
    rule reaches it, and it must flip to a fail that says it was not established.
    """
    storage, _ = memory_storage()
    stored = GoalStateOutcome(expression='not state.payment_hold == "released"', passed=True)
    result = _store(storage, [stored], end_state={"document_language": "de"}, variation={})

    report = await regrade(_host(storage), result.id, SCOPE)

    assert _flips(report) == [(stored.expression, True, False)]
    assert report.changes[0].detail.startswith("not established: state.payment_hold")
