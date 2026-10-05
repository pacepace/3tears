"""Re-grade a stored run's goal checks from the call ledgers its cells stored.

A goal check's call predicates (``called_before``, ``call_count``, ``calls``, …) read the cell's
call ledger, and the runner stores that ledger on the cell's trace
(:attr:`~threetears.evals.contracts.models.EvalTrace.call_ledger`) exactly as the kind recorded it.
So a stored result is re-graded from what its candidate DID, through
:func:`~threetears.evals.run.runner.grade_goal_checks` — the function every kind grades a live cell
with — and a change to the goal language reaches results stored before it without re-running them.
Nothing here is kind-shaped: no kind replays its own trace, because the ledger was stored as the
engine's own type.

**What it re-grades, and what it leaves as stored.** It re-grades every stored outcome whose
expression is a goal-state expression reading only the ledger and the case
(``variation.*``, read from the stored test case). It leaves as stored, and names:

* an expression that reads world state (``state.*``) — a cell's end state is not stored on its
  result, so there is nothing to re-read it from;
* an outcome that is not a goal-state expression at all — a fact a kind computed itself and
  reported under its own wording (``field_accuracy >= 0.92`` names no root the language admits);
* an expression reading ``variation.*`` whose test case no longer resolves.

**What it does not re-check at all, and says so.** A result whose checks were not graded at the
cell's end (any termination but ``completed``): a cell cut off on a deadline carries unevaluated,
failed checks, and a replay must never turn those into passes. A result with no stored ledger: its
kind kept none, or its trace did not land. A result one of whose checks no longer evaluates.

Rewrites go through each result's ETag, so a result something else wrote meanwhile is named and
left alone, and an applied re-check is idempotent: a second pass finds nothing to change.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import Field

from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.contracts.dsl import DSLError, extract_paths
from threetears.evals.contracts.errors import ConflictError, NotFoundError, ValidationFailedError
from threetears.evals.contracts.models import NON_TERMINAL_RUN_STATUSES, GoalStateOutcome
from threetears.evals.run.runner import GoalCheckUnevaluable, grade_goal_checks
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.contracts.call_ledger import CallLedger
    from threetears.evals.contracts.models import EvalResult, EvalRun, EvalTestCase, EvalTrace

log = get_logger(__name__)

#: The one termination whose checks were graded against the cell's end. Any other left them
#: unevaluated (a deadline), never ran them (excluded before a turn), or predates the field.
GRADED_TERMINATION = "completed"


class RecheckStore(Protocol):
    """The reads and the one rewrite a re-check makes, and no more.

    Structural, so a host's own storage satisfies it by having the methods. Scope parameters are
    positional-only, so this port says ``scope_id`` while an implementation names its partition.
    """

    def load_eval_run(self, run_id: str, scope_id: str, /) -> EvalRun | None:
        """One run within a scope, or ``None`` when it does not resolve."""
        ...

    def query_eval_results_by_run(self, run_id: str, scope_id: str, /) -> list[EvalResult]:
        """Every result of one run."""
        ...

    def load_eval_result_with_etag(self, result_id: str, scope_id: str, /) -> tuple[EvalResult | None, str | None]:
        """One result and the token a conditional rewrite of it must present."""
        ...

    def load_eval_trace(self, result_id: str, scope_id: str, /) -> EvalTrace | None:
        """One result's stored trace, or ``None`` when none is stored."""
        ...

    def load_test_case(self, test_case_id: str, scope_id: str, /) -> EvalTestCase | None:
        """One test case within a scope, or ``None`` when it does not resolve."""
        ...

    def replace_eval_result(self, result: EvalResult, /, *, if_match: str | None) -> None:
        """Rewrite a stored result, only if nothing else wrote it since it was read."""
        ...


class CheckFlip(EvalBaseModel):
    """One goal check whose verdict a re-grade changes."""

    expression: str
    was: bool
    now: bool


class KeptOutcome(EvalBaseModel):
    """One stored outcome a re-check left as it was, and why."""

    expression: str
    reason: str


class ResultRecheck(EvalBaseModel):
    """What a re-check found for one stored result."""

    result_id: str
    flips: list[CheckFlip] = Field(default_factory=list, description="The checks whose verdict moved.")
    regraded: int = Field(default=0, description="How many stored outcomes were re-graded, moved or not.")
    kept_as_stored: list[KeptOutcome] = Field(
        default_factory=list, description="The outcomes left as stored, each with the reason."
    )
    not_rechecked: str | None = Field(
        default=None, description="Why no outcome of this result was re-graded; None when it was re-checked."
    )


class RunRecheck(EvalBaseModel):
    """What a re-check of one run found, and whether it was written."""

    run_id: str
    applied: bool
    results: list[ResultRecheck]
    write_conflicts: list[str] = Field(
        default_factory=list, description="Results whose rewrite was refused because something else wrote them."
    )


def recheck_result(
    result: EvalResult, ledger: CallLedger | None, *, variation: Mapping[str, Any] | None
) -> tuple[ResultRecheck, list[GoalStateOutcome] | None]:
    """Re-grade one stored result's goal checks against its stored call ledger.

    Args:
        result: The stored result.
        ledger: The call ledger its trace stored, or ``None`` when none was stored.
        variation: Its test case's variation parameters, or ``None`` when the test case no longer
            resolves — an outcome reading ``variation.*`` is then kept as stored.

    Returns:
        The report, and the outcomes to store — ``None`` when no verdict moved, so a caller
        rewrites only results whose verdicts did. An outcome that did not move is kept verbatim,
        detail included, so a rewrite touches only what flipped.
    """
    if not result.goal_state_outcomes:
        return ResultRecheck(result_id=result.id, not_rechecked="it carries no goal-state outcomes"), None
    if result.termination != GRADED_TERMINATION:
        return ResultRecheck(
            result_id=result.id,
            not_rechecked=f"its checks were not graded at the cell's end (termination={result.termination})",
        ), None
    if ledger is None:
        return ResultRecheck(
            result_id=result.id, not_rechecked="no call ledger is stored for it — its kind keeps none"
        ), None

    outcomes: list[GoalStateOutcome] = []
    flips: list[CheckFlip] = []
    kept: list[KeptOutcome] = []
    for stored in result.goal_state_outcomes:
        reason = _kept_reason(stored.expression, has_variation=variation is not None)
        if reason is not None:
            kept.append(KeptOutcome(expression=stored.expression, reason=reason))
            outcomes.append(stored)
            continue
        try:
            (regraded,) = grade_goal_checks(
                [stored.expression], ledger=ledger, end_state={}, variation=variation or {}, world=None
            )
        except GoalCheckUnevaluable as unevaluable:
            return ResultRecheck(result_id=result.id, not_rechecked=f"a check no longer evaluates: {unevaluable}"), None
        if regraded.passed == stored.passed:
            outcomes.append(stored)
            continue
        flips.append(CheckFlip(expression=stored.expression, was=stored.passed, now=regraded.passed))
        outcomes.append(regraded)
    report = ResultRecheck(
        result_id=result.id,
        flips=flips,
        regraded=len(result.goal_state_outcomes) - len(kept),
        kept_as_stored=kept,
    )
    return report, outcomes if flips else None


def _kept_reason(expression: str, *, has_variation: bool) -> str | None:
    """Why a stored outcome cannot be re-graded from a ledger, or None when it can.

    Args:
        expression: The stored outcome's expression.
        has_variation: Whether the result's test case resolved.

    Returns:
        The reason, or None.
    """
    try:
        paths = extract_paths(expression)
    except DSLError:
        return "not a goal-state expression — a fact its kind computed and reported in its own words"
    if paths.world:
        return "it reads world state, and a cell's end state is not stored on its result"
    if paths.variation and not has_variation:
        return "it reads the case's variation, and the test case no longer resolves"
    return None


def recheck_goal_states(store: RecheckStore, run_id: str, scope_id: str, *, apply: bool) -> RunRecheck:
    """Re-grade every result of one stored run from its stored call ledgers, and optionally store the new verdicts.

    Args:
        store: The run's store.
        run_id: The run to re-check.
        scope_id: The partition it lives in.
        apply: ``False`` reports what would change and writes nothing.

    Returns:
        The per-result report, and the results whose rewrite was refused.

    Raises:
        NotFoundError: No run with that id in that scope.
        ValidationFailedError: The run is still pending or running — its cells are still being
            written, so a rewrite would race them.
    """
    run = store.load_eval_run(run_id, scope_id)
    if run is None:
        raise NotFoundError("run", run_id)
    if run.status in NON_TERMINAL_RUN_STATUSES:
        raise ValidationFailedError(
            f"run '{run_id}' is {run.status}: its cells are still being written, so re-check it once it has finished"
        )
    variations: dict[str, Mapping[str, Any] | None] = {}
    reports: list[ResultRecheck] = []
    conflicts: list[str] = []
    for listed in store.query_eval_results_by_run(run_id, scope_id):
        result, etag = store.load_eval_result_with_etag(listed.id, scope_id)
        if result is None or etag is None:
            reports.append(ResultRecheck(result_id=listed.id, not_rechecked="it was deleted while the re-check ran"))
            continue
        trace = store.load_eval_trace(result.id, scope_id) if result.has_trace else None
        if result.test_case_id not in variations:
            case = store.load_test_case(result.test_case_id, scope_id)
            variations[result.test_case_id] = case.variation_params if case is not None else None
        report, outcomes = recheck_result(
            result, trace.call_ledger if trace is not None else None, variation=variations[result.test_case_id]
        )
        reports.append(report)
        if not apply or outcomes is None:
            continue
        try:
            store.replace_eval_result(result.model_copy(update={"goal_state_outcomes": outcomes}), if_match=etag)
        except ConflictError:
            conflicts.append(result.id)
    flipped = [report.result_id for report in reports if report.flips]
    log.info(
        "eval.recheck_goal_states run=%s applied=%s results=%d flipped=%d conflicts=%s",
        run_id,
        apply,
        len(reports),
        len(flipped),
        conflicts,
    )
    return RunRecheck(run_id=run_id, applied=apply, results=reports, write_conflicts=conflicts)


__all__ = [
    "GRADED_TERMINATION",
    "CheckFlip",
    "KeptOutcome",
    "RecheckStore",
    "ResultRecheck",
    "RunRecheck",
    "recheck_goal_states",
    "recheck_result",
]
