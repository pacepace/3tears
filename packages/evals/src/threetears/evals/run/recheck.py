"""Re-grade a stored run's goal checks from what its cells stored: their call ledgers, end states and world events.

A goal check reads three things about a finished cell, and the runner stores all three exactly as the
cell left them: the calls its candidate made (the trace's
:attr:`~threetears.evals.contracts.models.EvalTrace.call_ledger`, read by ``called_before``,
``calls`` …), the world it left behind (the trace's
:attr:`~threetears.evals.contracts.models.EvalTrace.end_state`, read by ``state.<dimension>``), and the
triggered dimensions that fired (the result's
:attr:`~threetears.evals.contracts.models.EvalResult.world_events`, read by ``fired()`` and ``fired_armed()``). So a stored
result is re-graded from what its candidate DID and what its world BECAME, through
:func:`~threetears.evals.run.runner.grade_goal_checks` — the function every kind grades a live cell
with — and a change to the goal language reaches results stored before it without re-running them.
Nothing here is kind-shaped: no kind replays its own trace, because each input was stored as the
engine's own type.

**It establishes no more than the original grading could.** What fired is read through
:meth:`~threetears.evals.contracts.world_events.Firings.of` with the provenance of the run the result
belongs to — the rule the cell was graded under. A witnessed cell had no seed, so its stored events say
``armed=False`` because nothing could mark them armed; ``fired_armed()`` on it stays *not established*
on re-check, negated or not, instead of reading those events as "nothing armed fired".

**What it re-grades, and what it leaves as stored.** It re-grades every stored outcome whose
expression is a goal-state expression whose every input is stored. It leaves as stored, and names:

* an expression that reads the call ledger, for a cell whose kind kept none (or whose trace did not
  land);
* an expression that reads world state, for a cell whose end state was not stored, or a re-check given
  no world registry to resolve ``state.<dimension>`` through;
* an expression reading a world path that today's registry no longer resolves, or resolves to a
  dimension the cell's stored end state does not hold — the dimension was renamed or removed since
  the cell was graded, so re-grading would score the vocabulary change as the candidate's failure;
* an expression today's language refuses although it was written in the language's words — a goal
  check stored under an older, looser rule, named as refused rather than mistaken for a kind's fact;
* an expression that reads ``fired()`` or ``fired_armed()``, for a cell that recorded no world events;
* an outcome that is not a goal-state expression at all — a fact a kind computed itself and
  reported under its own wording (``field_accuracy >= 0.92`` names no root the language admits);
* an expression reading ``variation.*`` whose test case no longer resolves.

**What it does not re-check at all, and says so.** A result whose checks were not graded at the
cell's end (any termination but ``completed``): a cell cut off on a deadline carries unevaluated,
failed checks, and a replay must never turn those into passes. A result one of whose checks no longer
evaluates.

Rewrites go through each result's ETag, so a result something else wrote meanwhile is named and
left alone, and an applied re-check is idempotent: a second pass finds nothing to change.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from threetears.evals.contracts.call_ledger import CallLedger

from pydantic import Field

from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.contracts.goal_grammar import (
    DSLError,
    extract_paths,
    reads_call_ledger,
    referenced_fires,
    speaks_the_goal_language,
)
from threetears.evals.contracts.errors import ConflictError, NotFoundError, ValidationFailedError
from threetears.evals.contracts.models import NON_TERMINAL_RUN_STATUSES, ApparatusProvenance, GoalStateOutcome
from threetears.evals.contracts.world_events import Firings
from threetears.evals.run.runner import GoalCheckUnevaluable, grade_goal_checks
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.contracts.host.world import WorldRegistry
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
    result: EvalResult,
    *,
    ledger: CallLedger | None,
    end_state: Mapping[str, Any] | None,
    variation: Mapping[str, Any] | None,
    world: WorldRegistry | None,
    provenance: ApparatusProvenance,
) -> tuple[ResultRecheck, list[GoalStateOutcome] | None]:
    """Re-grade one stored result's goal checks against what its cell stored.

    What fired is read off the result itself (``world_events``); the ledger and the end state come off
    its trace.

    Args:
        result: The stored result.
        ledger: The call ledger its trace stored, or ``None`` when none was stored — an outcome reading
            the ledger is then kept as stored.
        end_state: The end state its trace stored, or ``None`` when none was — an outcome reading world
            state is then kept as stored.
        variation: Its test case's variation parameters, or ``None`` when the test case no longer
            resolves — an outcome reading ``variation.*`` is then kept as stored.
        world: The host's world registry, which ``state.<dimension>`` resolves through; ``None`` keeps
            every outcome reading world state as stored.
        provenance: The apparatus provenance of the run the result belongs to, which decides what its
            world events can establish (:meth:`~threetears.evals.contracts.world_events.Firings.of`) — a
            witnessed cell's ``fired_armed()`` is not established, as it was when the cell was graded.

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
    stored_inputs = _StoredInputs(
        has_ledger=ledger is not None,
        has_end_state=end_state is not None,
        has_world=world is not None,
        has_events=result.world_events is not None,
        has_variation=variation is not None,
    )
    fired = Firings.of(result.world_events, provenance=provenance) if result.world_events is not None else None

    outcomes: list[GoalStateOutcome] = []
    flips: list[CheckFlip] = []
    kept: list[KeptOutcome] = []
    for stored in result.goal_state_outcomes:
        reason = _kept_reason(stored.expression, stored_inputs) or _vocabulary_moved(
            stored.expression, end_state=end_state, world=world
        )
        if reason is not None:
            kept.append(KeptOutcome(expression=stored.expression, reason=reason))
            outcomes.append(stored)
            continue
        try:
            # An input this expression does not read is passed empty: ``_kept_reason`` has already kept
            # every expression reading an input that was not stored.
            (regraded,) = grade_goal_checks(
                [stored.expression],
                ledger=ledger if ledger is not None else CallLedger(),
                end_state=end_state if end_state is not None else {},
                fired=fired,
                variation=variation or {},
                world=world,
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


class _StoredInputs(EvalBaseModel):
    """Which of a goal check's inputs one stored result can supply."""

    has_ledger: bool
    has_end_state: bool
    has_world: bool
    has_events: bool
    has_variation: bool


def _kept_reason(expression: str, inputs: _StoredInputs) -> str | None:
    """Why a stored outcome cannot be re-graded from what its cell stored, or None when it can.

    Args:
        expression: The stored outcome's expression.
        inputs: Which inputs the result can supply.

    Returns:
        The reason, or None.
    """
    try:
        paths = extract_paths(expression)
        reads_calls = reads_call_ledger(expression)
        reads_fired = bool(referenced_fires(expression))
    except DSLError as refused:
        if speaks_the_goal_language(expression):
            return f"a goal check today's language refuses, so it cannot be re-graded under today's rules: {refused}"
        return "not a goal-state expression — a fact its kind computed and reported in its own words"
    if reads_calls and not inputs.has_ledger:
        return "it reads the call ledger, and no call ledger is stored for its cell — its kind keeps none"
    if paths.world and not inputs.has_end_state:
        return "it reads world state, and its cell's end state was not stored"
    if paths.world and not inputs.has_world:
        return "it reads world state, and this re-check was given no world registry to resolve it through"
    if reads_fired and not inputs.has_events:
        return "it reads what fired, and its cell recorded no world events"
    if paths.variation and not inputs.has_variation:
        return "it reads the case's variation, and the test case no longer resolves"
    return None


def _vocabulary_moved(
    expression: str, *, end_state: Mapping[str, Any] | None, world: WorldRegistry | None
) -> str | None:
    """Why a world-reading check cannot be re-graded against today's vocabulary, or None when it can.

    Re-grading reads ``state.<path>`` through the registry the re-check was given, which is today's.
    When the dimension a path resolves to TODAY differs from the one the cell stored it under — the
    path names no declared dimension any more, or names one other than the stored end state's key
    for it — re-grading would read Missing and rewrite the stored verdict as a check the candidate did
    not establish, when what changed was the host's vocabulary. Such an outcome is kept as stored.

    A path whose dimension today's registry resolves, and which the stored end state never held
    under any name (its carrier was not attached), is NOT a vocabulary change: the cell never had the
    value, and re-grading it as not established is the language's rule reaching a stored result.

    Args:
        expression: A stored outcome's expression, which :func:`_kept_reason` has already admitted.
        end_state: Its cell's stored end state (present whenever the expression reads world state).
        world: Today's world registry (present whenever the expression reads world state).

    Returns:
        The reason, or None.
    """
    if end_state is None or world is None:
        return None
    for path in extract_paths(expression).world:
        today = world.resolve_path(path)
        stored = max((name for name in end_state if path == name or path.startswith(f"{name}.")), key=len, default=None)
        if today is None:
            return (
                f"it reads state.{path}, which names no dimension today's world declares — renamed or removed "
                "since the cell was graded, so its stored verdict stands"
            )
        if stored is not None and stored != today:
            return (
                f"it reads state.{path}, which its cell stored under {stored} and today's world resolves to "
                f"{today} — the vocabulary moved since the cell was graded, so its stored verdict stands"
            )
    return None


def recheck_goal_states(
    store: RecheckStore, run_id: str, scope_id: str, *, world: WorldRegistry | None, apply: bool
) -> RunRecheck:
    """Re-grade every result of one stored run from what its cells stored, and optionally store the new verdicts.

    Args:
        store: The run's store.
        run_id: The run to re-check.
        scope_id: The partition it lives in.
        world: The host's world registry (``profile.world``), through which a ``state.<dimension>`` check
            reads a stored end state. ``None`` for a host that declares no world, whose checks read none.
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
            result,
            ledger=trace.call_ledger if trace is not None else None,
            end_state=trace.end_state if trace is not None else None,
            variation=variations[result.test_case_id],
            world=world,
            provenance=run.apparatus_provenance,
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
