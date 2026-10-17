"""Re-grade one stored result from what its cell stored: its check dims from the trace, its failed judged dims by
asking its judge again.

The one door a stored verdict is graded again through. It reports first: with ``apply=False`` (the default) it
computes every new verdict it can without spending and writes nothing; ``apply=True`` writes them, and re-asks the
failed judged dims, which spends.

**Check dims re-grade from the trace.** A goal check reads three things about a finished cell, and the runner stores
all three exactly as the cell left them: the calls its candidate made (the trace's
:attr:`~threetears.evals.schema.models.EvalTrace.call_ledger`, read by ``called_before``, ``calls`` …), the world it
left behind (the trace's :attr:`~threetears.evals.schema.models.EvalTrace.end_state`, read by
``state.<dimension>``), and the triggered dimensions that fired (the result's
:attr:`~threetears.evals.schema.models.EvalResult.world_events`, read by ``fired()`` and ``fired_armed()``). So a
stored result is re-graded from what its candidate DID and what its world BECAME, through
:func:`~threetears.evals.run.runner.grade_goal_checks`, the function every kind grades a live cell with, and a change
to the goal language reaches results stored before it without re-running them. Nothing here is kind-shaped: no kind
replays its own trace, because each input was stored as the engine's own type. It establishes no more than the
original grading could: what fired is read through :meth:`~threetears.evals.schema.world_events.Firings.of` with the
provenance of the result's run, so a witnessed cell's ``fired_armed()`` stays *not established*.

It leaves as stored, and names, every outcome it cannot re-grade from what was stored: one reading the call ledger,
the calls a tool refused, world state, what fired or the case's variation, for a cell that stored none of it; one
reading a world path today's registry no longer resolves, or resolves to another dimension than the cell stored it
under (the vocabulary moved, which re-grading would score as the candidate's failure); one today's language refuses
although written in its words; and an outcome that is not a goal-state expression at all (a fact a kind computed and
reported in its own words). It re-grades nothing of a result whose checks were not graded at the cell's end (any
termination but ``completed``: a cell cut off carries unevaluated checks, and a replay must never turn those into
passes), or one of whose checks no longer evaluates.

**Judged dims are asked again, never re-scored from a guess.** A judge's reply is not stored, so a judged dim can be
graded again only by asking its judge. The dims re-asked are those the result's judging failed to answer
(:func:`~threetears.evals.run.rejudge.failed_judge_dims`: a can't-tell is an answer and is never re-asked), each held to
:func:`~threetears.evals.run.rejudge.reproducible_judge_inputs` (``JudgeInputStore``) before anything is written or
spent, so a dim whose apparatus the run did not record is refused rather than asked differently. Reporting names
them and calls nothing. Applying asks them through :func:`~threetears.evals.run.lifecycle.rejudge_result`, which lands
the outcomes where the judge phase would have put them (:func:`~threetears.evals.run.rejudge.apply_rejudge`) and
records a ``JudgeRescore``.

Every write is conditional on what was read, so a result something else wrote meanwhile is refused
(``ConflictError``), never overwritten, and an applied re-grade is idempotent: a second pass finds nothing to change.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from threetears.evals.kernel.errors import ConflictError, NotFoundError, ValidationFailedError
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.remedies import FIND_RESULT, FIND_RUN
from threetears.evals.run.lifecycle import rejudge_result
from threetears.evals.run.rejudge import failed_judge_dims, reproducible_judge_inputs
from threetears.evals.run.runner import GoalCheckUnevaluable, grade_goal_checks
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.call_ledger import CallLedger
from threetears.evals.schema.goal_grammar import (
    DSLError,
    extract_paths,
    reads_call_ledger,
    reads_refusals,
    referenced_fires,
    speaks_the_goal_language,
)
from threetears.evals.schema.logs import get_logger
from threetears.evals.schema.models import NON_TERMINAL_RUN_STATUSES, ApparatusProvenance, GoalStateOutcome
from threetears.evals.schema.world_events import Firings

if TYPE_CHECKING:
    from threetears.evals.kernel.host.eval_host import EvalHost
    from threetears.evals.kernel.host.world import WorldRegistry
    from threetears.evals.kernel.storage import EvalStorage
    from threetears.evals.schema.models import EvalResult, EvalRun

log = get_logger(__name__)

#: The one termination whose checks were graded against the cell's end. Any other left them unevaluated (a deadline)
#: or never ran them (excluded before a turn).
GRADED_TERMINATION = "completed"


class RegradedDim(EvalBaseModel):
    """One dimension whose verdict a re-grade moved: a check's pass, or a failed judged dim's new answer."""

    dim: str = Field(description="A goal check's expression, or a judged dimension's id.")
    graded_by: Literal["check", "judge"] = Field(description="Re-graded from the trace, or asked of the judge again.")
    was: bool | int | None = Field(description="The stored verdict: a check's pass, or a judged dim's score (None).")
    now: bool | int | None = Field(
        description="The new verdict: a check's pass, or a judged dim's score (None: its judge answered can't tell)."
    )
    detail: str = Field(description="What the new verdict says: the check's detail, or the judge's answer.")


class Regrade(EvalBaseModel):
    """What re-grading one stored result found, and whether it was written."""

    result_id: str
    run_id: str
    applied: bool = Field(description="Whether the new verdicts were stored; False reports and writes nothing.")
    changes: list[RegradedDim] = Field(
        default_factory=list, description="Every dimension whose verdict moved, checks first; empty when none did."
    )
    checks_regraded: int = Field(default=0, description="Stored check outcomes re-graded from the trace, moved or not.")
    kept_as_stored: dict[str, str] = Field(
        default_factory=dict, description="Each check outcome left as stored, by expression, with why."
    )
    checks_not_regraded: str | None = Field(
        default=None, description="Why no check outcome was re-graded; None when they were."
    )
    judge_dims: list[str] = Field(
        default_factory=list,
        description="The judged dims whose judging failed: asked again when applied, otherwise what applying asks.",
    )
    judge_errors: dict[str, str] = Field(
        default_factory=dict, description="Each re-asked judged dim whose judging failed again, with its error."
    )
    judges_not_regraded: str | None = Field(
        default=None, description="Why no judged dim is asked again; None when some is (or, unapplied, would be)."
    )
    judge_cost_usd: float | None = Field(
        default=None, description="What asking the judge again cost; None when nothing was asked or it is unpriced."
    )


async def regrade(
    host: EvalHost,
    result_id: str,
    scope_id: str,
    *,
    checks: bool = True,
    judges: bool = True,
    apply: bool = False,
) -> Regrade:
    """Grade one stored result again from what its cell stored, and report every verdict that moved.

    Check dims (goal checks) are re-graded from the result's trace through the function the cell was graded with;
    judged dims whose judging failed are named and, applied, asked of the run's judge again. Nothing is written
    unless ``apply``; then the moved check verdicts are stored, conditional on the result as read, and the judged
    dims are asked and stored through ``rejudge_result``, in that order.

    :param host: The host: the result's store, the world registry a ``state.*`` check resolves through
        (``profile.world``), and, to apply a judged re-ask, the judge clients.
    :ptype host: EvalHost
    :param result_id: The stored result (one trial) to re-grade.
    :ptype result_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param checks: Re-grade its check dims from its trace.
    :ptype checks: bool
    :param judges: Name its failed judged dims, and ask them again when applying.
    :ptype judges: bool
    :param apply: Store the new verdicts and ask the judge; ``False`` (the default) reports and spends nothing.
    :ptype apply: bool
    :return: What moved, what was left as stored and why, and whether it was written.
    :rtype: Regrade
    :raises NotFoundError: No such result, or its run, in the scope.
    :raises ValidationFailedError: Its run is still pending or running; or a failed judged dim's apparatus cannot be
        reproduced from what the run recorded (:func:`~threetears.evals.run.rejudge.reproducible_judge_inputs`
        names it), before anything is written or spent; ask with ``judges=False`` to re-grade its checks alone.
    :raises ValueError: Applying a judged re-ask on a host that supplies no completion clients, before anything is
        written.
    :raises ConflictError: Something else wrote the result since it was read: nothing of the checks was stored. Raised
        by the judged re-ask too, after the checks were stored.
    """
    read = await run_blocking(host.blocking_executor, _read, host.storage, result_id, scope_id, judges=judges)
    result, etag, run, failed = read.result, read.etag, read.run, read.failed_judge_dims
    report = Regrade(result_id=result.id, run_id=run.id, applied=apply)
    outcomes: list[GoalStateOutcome] | None = None
    if checks:
        outcomes = _regrade_checks(report, result, read, world=host.profile.world, provenance=run.apparatus_provenance)
    else:
        report.checks_not_regraded = "checks were not asked for (checks=False)"
    report.judge_dims = failed
    report.judges_not_regraded = read.judges_not_regraded
    if apply and failed:
        # Refused here, before the checks are written, rather than after.
        host.completion_clients("a re-grade's judged dims")
    if apply and outcomes is not None:
        updated = result.model_copy(update={"goal_state_outcomes": outcomes})
        try:
            await run_blocking(host.blocking_executor, host.storage.replace_eval_result, updated, if_match=etag)
        except ConflictError as raced:
            raise ConflictError(
                f"result '{result.id}' changed while it was being re-graded; nothing was stored",
                remedy="re-grade it again",
            ) from raced
    if apply and failed:
        rejudged = await rejudge_result(host, result.id, scope_id)
        rescore = rejudged.judge_rescores[-1]
        report.judge_cost_usd = rescore.cost_usd
        report.judge_errors = dict(rescore.errors)
        report.changes += [
            RegradedDim(dim=dim, graded_by="judge", was=None, now=rescore.scores[dim], detail="scored")
            for dim in rescore.dims
            if dim in rescore.scores
        ]
        report.changes += [
            RegradedDim(dim=dim, graded_by="judge", was=None, now=None, detail=f"can't tell: {reason}")
            for dim, reason in rescore.cannot_tell.items()
        ]
    log.info(
        "eval.regrade result=%s run=%s applied=%s changed=%d kept=%d judge_dims=%s",
        result.id,
        run.id,
        apply,
        len(report.changes),
        len(report.kept_as_stored),
        ",".join(failed) or "-",
    )
    return report


@dataclass(frozen=True, slots=True)
class _Read:
    """What a re-grade reads before it writes or spends anything."""

    result: EvalResult
    etag: str | None
    run: EvalRun
    ledger: CallLedger | None
    end_state: dict[str, Any] | None
    variation: dict[str, Any] | None
    failed_judge_dims: list[str]
    judges_not_regraded: str | None


def _read(storage: EvalStorage, result_id: str, scope_id: str, *, judges: bool) -> _Read:
    """The result, its etag, its run, what its cell stored, and its failed judged dims held to reproducibility.

    :raises NotFoundError: No such result or run.
    :raises ValidationFailedError: The run is still being written, or a failed judged dim cannot be reproduced.
    """
    result, etag = storage.load_eval_result_with_etag(result_id, scope_id)
    if result is None:
        raise NotFoundError("result", result_id, remedy=FIND_RESULT)
    run = storage.load_eval_run(result.eval_run_id, scope_id)
    if run is None:
        raise NotFoundError("run", result.eval_run_id, remedy=FIND_RUN)
    if run.status in NON_TERMINAL_RUN_STATUSES:
        raise ValidationFailedError(
            f"run '{run.id}' is {run.status}: its cells are still being written",
            remedy=f"re-grade once it has finished (job_poll run:{run.id})",
        )
    trace = storage.load_eval_trace(result.id, scope_id) if result.has_trace else None
    case = storage.load_test_case(result.test_case_id, scope_id)
    failed, why_not = (
        _failed_judge_dims(storage, result, run, scope_id)
        if judges
        else ([], "judged dims were not asked for (judges=False)")
    )
    return _Read(
        result=result,
        etag=etag,
        run=run,
        ledger=trace.call_ledger if trace is not None else None,
        end_state=dict(trace.end_state) if trace is not None and trace.end_state is not None else None,
        variation=dict(case.variation_params) if case is not None else None,
        failed_judge_dims=failed,
        judges_not_regraded=why_not,
    )


def _failed_judge_dims(
    storage: EvalStorage, result: EvalResult, run: EvalRun, scope_id: str
) -> tuple[list[str], str | None]:
    """The judged dims whose judging failed, each reproducible as the run recorded it; or why there are none.

    :raises ValidationFailedError: A failed dim's apparatus cannot be reproduced (the check a re-judge makes).
    """
    if result.judge_error is None:
        return [], "no judged dim failed: each holds a score or a can't-tell, or the result was not judged"
    failed = failed_judge_dims(result, list(run.effective_judges or {}))
    inputs = reproducible_judge_inputs(
        storage, result, run, scope_id, request_settings="today", config_dims=failed, same_judge_dims=failed
    )
    # Narrowed to the dims this result was judged on, as a re-judge narrows them: a document's run attributes the two
    # conversation axes, which its cell was never asked.
    failed = [dim for dim in failed if dim in inputs.dims]
    if not failed:
        return [], "it carries a judge_error, and holds a score or a can't-tell for every dim its run judged"
    return failed, None


def _regrade_checks(
    report: Regrade,
    result: EvalResult,
    read: _Read,
    *,
    world: WorldRegistry | None,
    provenance: ApparatusProvenance,
) -> list[GoalStateOutcome] | None:
    """Re-grade one result's goal checks against what its cell stored, recording each on ``report``.

    :return: The outcomes to store, or ``None`` when no verdict moved, so only a moved result is rewritten. An outcome
        that did not move is kept verbatim, detail included.
    :rtype: list[GoalStateOutcome] | None
    """
    if not result.goal_state_outcomes:
        report.checks_not_regraded = "it carries no goal-state outcomes"
        return None
    if result.termination != GRADED_TERMINATION:
        report.checks_not_regraded = f"its checks were not graded at the cell's end (termination={result.termination})"
        return None
    ledger, end_state, variation = read.ledger, read.end_state, read.variation
    stored_inputs = _StoredInputs(
        has_ledger=ledger is not None,
        has_refusals=ledger is not None and ledger.refused is not None,
        has_end_state=end_state is not None,
        has_world=world is not None,
        has_events=result.world_events is not None,
        has_variation=variation is not None,
    )
    fired = Firings.of(result.world_events, provenance=provenance) if result.world_events is not None else None
    outcomes: list[GoalStateOutcome] = []
    moved: list[RegradedDim] = []
    kept: dict[str, str] = {}
    for stored in result.goal_state_outcomes:
        reason = _kept_reason(stored.expression, stored_inputs) or _vocabulary_moved(
            stored.expression, end_state=end_state, world=world
        )
        if reason is not None:
            kept[stored.expression] = reason
            outcomes.append(stored)
            continue
        try:
            # An input this expression does not read is passed empty: ``_kept_reason`` has already kept every
            # expression reading an input that was not stored.
            (regraded,) = grade_goal_checks(
                [stored.expression],
                ledger=ledger if ledger is not None else CallLedger(),
                end_state=end_state if end_state is not None else {},
                fired=fired,
                variation=variation or {},
                world=world,
            )
        except GoalCheckUnevaluable as unevaluable:
            report.checks_not_regraded = f"a check no longer evaluates: {unevaluable}"
            return None
        if regraded.passed == stored.passed:
            outcomes.append(stored)
            continue
        moved.append(
            RegradedDim(
                dim=stored.expression, graded_by="check", was=stored.passed, now=regraded.passed, detail=regraded.detail
            )
        )
        outcomes.append(regraded)
    report.changes += moved
    report.checks_regraded = len(result.goal_state_outcomes) - len(kept)
    report.kept_as_stored = kept
    return outcomes if moved else None


class _StoredInputs(EvalBaseModel):
    """Which of a goal check's inputs one stored result can supply."""

    has_ledger: bool
    has_refusals: bool
    has_end_state: bool
    has_world: bool
    has_events: bool
    has_variation: bool


def _kept_reason(expression: str, inputs: _StoredInputs) -> str | None:
    """Why a stored outcome cannot be re-graded from what its cell stored, or None when it can."""
    try:
        paths = extract_paths(expression)
        reads_calls = reads_call_ledger(expression)
        reads_refused = reads_refusals(expression)
        reads_fired = bool(referenced_fires(expression))
    except DSLError as refused:
        if speaks_the_goal_language(expression):
            return f"a goal check today's language refuses, so it cannot be re-graded under today's rules: {refused}"
        return "not a goal-state expression — a fact its kind computed and reported in its own words"
    if reads_calls and not inputs.has_ledger:
        return "it reads the call ledger, and no call ledger is stored for its cell — its kind keeps none"
    if reads_refused and not inputs.has_refusals:
        return (
            "it reads the calls a tool refused, and its cell's ledger keeps no refusal record — its kind records "
            "none, or it was stored before refusals were recorded"
        )
    if paths.world and not inputs.has_end_state:
        return "it reads world state, and its cell's end state was not stored"
    if paths.world and not inputs.has_world:
        return "it reads world state, and the host declares no world registry to resolve it through"
    if reads_fired and not inputs.has_events:
        return "it reads what fired, and its cell recorded no world events"
    if paths.variation and not inputs.has_variation:
        return "it reads the case's variation, and the test case no longer resolves"
    return None


def _vocabulary_moved(
    expression: str, *, end_state: Mapping[str, Any] | None, world: WorldRegistry | None
) -> str | None:
    """Why a world-reading check cannot be re-graded against today's vocabulary, or None when it can.

    Re-grading reads ``state.<path>`` through today's registry. When the dimension a path resolves to today differs
    from the one the cell stored it under (the path names no declared dimension any more, or names one other than the
    stored end state's key for it), re-grading would read Missing and rewrite the stored verdict as a check the
    candidate did not establish, when what changed was the host's vocabulary: the outcome is kept as stored. A path
    today's registry resolves, whose dimension the stored end state never held under any name, is NOT a vocabulary
    change: the cell never had the value, and re-grading it as not established is the language's rule reaching it.
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


__all__ = ["GRADED_TERMINATION", "Regrade", "RegradedDim", "regrade"]
