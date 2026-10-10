"""The run lifecycle: reading a run, cancelling it, reclaiming abandoned ones, re-judging a result, and recording completeness.

What happens to a run after its launch, over the run's own storage. Each operation is a function
whose dependencies are parameters — the store, the job manager that can see live tasks, and, for a
re-judge, the judge client factory and the subject the judge reads — the shape the curation family
set (:mod:`threetears.evals.run.curation`), so a client of the package can manage its runs without
a host's service layer, and every surface that reaches these does so through one implementation.

Archive and delete are not here: they are :mod:`threetears.evals.run.curation`'s, which this module
sits beside rather than wraps.

The scope a run lives in is ``scope_id`` here — the engine's word for a partition it never
interprets. The host chooses what a scope is and passes it through.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.kernel.errors import ConflictError, NotFoundError, StorageError, ValidationFailedError
from threetears.evals.schema.models import NON_TERMINAL_RUN_STATUSES, EvalRun, utc_now_iso
from threetears.evals.kernel.result_condition import ResultOutcome, classify_result
from threetears.evals.kernel.scoring import reconstruct_completeness, summarize_completeness
from threetears.evals.run.judge_service import JudgeService, judge_clients_for_run
from threetears.evals.run.rejudge import apply_rejudge, failed_judge_dims, reproducible_judge_inputs
from threetears.evals.run.run_document import update_eval_run
from threetears.evals.run.runner import build_judge_context, judge_dims
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.schema.models import EvalResult, RunCompleteness
    from threetears.evals.kernel.host.eval_host import EvalHost
    from threetears.evals.kernel.scoring import CellSummary
    from threetears.evals.run.jobs import EvalJobManager
    from threetears.evals.run.metering import MeteredCallTally
    from threetears.evals.kernel.storage import JobStore, RunRecordStore, RunStore

log = get_logger(__name__)

#: Recorded on every run the startup reclaim settles, so the record says why the
#: run ended rather than implying an operator chose to stop it. That distinction
#: does not survive in ``status`` — a reboot cancels in-flight runs and there is
#: deliberately no separate status for it — so this line is the only place an
#: analysis can tell infrastructure loss from a deliberate stop.
_ABANDONED_RUN_REASON = "abandoned by a process restart while in flight"


class AbandonedRunSweepReport(EvalBaseModel):
    """Outcome of one startup reclaim of runs left non-terminal by a dead process."""

    scanned: int = 0
    """Non-terminal rows the scan MATCHED, summed over every scope it was told to sweep.

    Rows matched, not runs returned: a row the scan could not reconstruct is
    counted here too, via ``unreadable``. Under the older reading — runs the scan
    handed back — a boot facing a wholly unreadable corpus would report a scan
    that found nothing, which is the one case where it found the most. The four
    outcome figures (``cancelled_run_ids``, ``already_settled``, ``skipped_live``,
    ``unreadable``) sum to this."""

    cancelled_run_ids: list[str] = []
    """Runs this sweep settled to ``cancelled``.

    Named rather than counted: a boot that cancels a run an operator was
    watching has to be able to say which one."""

    already_settled: int = 0
    """Scanned runs that were terminal or gone by the time their repair ran.

    Not a failure — something else reached them first, which is the state this
    sweep wanted them in."""

    skipped_live: int = 0
    """Scanned runs this process is actually executing, and so left alone.

    Zero on the boot path the sweep is built for (the job manager is empty
    then); non-zero means it was invoked with work in flight."""

    unreadable: int = 0
    """Scanned rows the sweep could not settle because something would not read.

    Two origins, deliberately one figure. Either the scan matched a row it could
    not reconstruct at all, or the row read fine and its *repair* did not —
    settling a run reconstructs its completeness from stored results, and that
    read still raises for a whole list. An operator's question is the same in
    both cases ("how much is stranded, and why can nothing settle it"), and the
    per-row ERROR logs say which happened.

    These are the runs the sweep can do nothing about: it cannot settle a
    document it cannot read, so they stay stamped non-terminal until the corpus
    is dropped. Counted rather than hidden — a reclaim that silently skipped
    them would report a clean scan while leaving exactly the rows an operator
    needs to act on. They are inside ``scanned``, so the four outcome figures
    still sum to it."""


def get_run(storage: RunStore, run_id: str, scope_id: str) -> EvalRun:
    """Load an eval run by id within its scope, whole.

    Args:
        storage: The run's store.
        run_id: The run to load.
        scope_id: The partition the run lives in.

    Returns:
        The run.

    Raises:
        NotFoundError: No run with that id in the scope.
    """
    run = storage.load_eval_run(run_id, scope_id)
    if run is None:
        raise NotFoundError("run", run_id)
    return run


def cancel_run(
    storage: RunRecordStore,
    run_id: str,
    scope_id: str,
    *,
    job_manager: EvalJobManager | None,
    reason: str | None = None,
) -> EvalRun:
    """Cancel a pending/running eval run so it stops burning cost and quota.

    Two paths, selected by whether the run has a live background task:

    - **Live task** — cancellation is requested through the job manager;
      the job's own ``CancelledError`` boundary writes ``cancelled`` (plus
      ``reason`` into ``cancellation_reason``) as the task unwinds, typically
      at its next await point. Request-then-converge: the run returned here
      may still read ``running`` for a moment — confirm via :func:`get_run`.
    - **No live task** (abandoned, e.g. by a web-process restart while the
      run was in flight) — the run document is repaired directly to
      ``cancelled`` with an explanatory reason, and its completeness is
      reconstructed from stored results since the process that held the
      loop's own tally is gone. Deliberately not "orphaned": an orphaned run
      is a curation concept — a run no campaign holds — and a run can be
      either, neither, or both.

    Both paths write the reason to ``cancellation_reason`` and never to
    ``error_details``: a cancel is a distinct terminal outcome, so an
    operator's decision to stop is readable apart from the things that broke.

    Results already persisted by the run remain valid and readable.

    Args:
        storage: The run's store.
        run_id: Run to cancel.
        scope_id: The partition the run lives in.
        job_manager: The job manager that owns this process's live tasks, the only thing that can
            see whether the run has one; ``None`` when the caller was built without one.
        reason: Optional operator-facing reason recorded on the run.

    Returns:
        The run as persisted after the cancel request (see convergence
        note above).

    Raises:
        NotFoundError: No run with that id in the scope.
        ValidationFailedError: The run is already terminal
            (``completed`` / ``failed`` / ``cancelled`` / ``budget_stopped`` / ``exhausted``).
        ConflictError: The no-live-task repair's conditional write lost its
            race — the document changed between the read and the write, so
            nothing was repaired and the caller must re-read before retrying.
        RuntimeError: ``job_manager`` is ``None`` — cancellation must go through
            the canonical wiring, which is the only place that can see live tasks.
    """
    run = get_run(storage, run_id, scope_id)
    require_cancellable(run)
    if job_manager is None:
        raise RuntimeError("cancel_run requires job_manager — without one, whether a run is live is unanswerable")

    if job_manager.cancel_job(run_id, reason=reason):
        log.info(
            "eval.cancel_run requested run=%s scope=%s reason=%s",
            run_id,
            scope_id,
            reason or "(none)",
        )
        return get_run(storage, run_id, scope_id)
    return repair_abandoned_run(storage, run_id, scope_id, reason=reason)


def require_cancellable(run: EvalRun) -> None:
    """Refuse a cancel of a run that has already ended — :func:`cancel_run`'s first check, for a caller that splits it.

    Args:
        run: The run as read.

    Raises:
        ValidationFailedError: The run is already terminal.
    """
    if run.status not in NON_TERMINAL_RUN_STATUSES:
        raise ValidationFailedError(f"run '{run.id}' is {run.status} — only pending/running runs can be cancelled")


def repair_abandoned_run(storage: RunRecordStore, run_id: str, scope_id: str, *, reason: str | None) -> EvalRun:
    """Repair a non-terminal run no live task is running to ``cancelled`` — :func:`cancel_run`'s no-live-task half.

    Every store call :func:`cancel_run` makes beyond its first read is here, so a caller serving an event
    loop can ask the job manager on the loop and run this on its blocking executor
    (:func:`~threetears.evals.ops.job_cancel` does).

    Args:
        storage: The run's store.
        run_id: The run.
        scope_id: The partition it lives in.
        reason: The operator-facing reason, recorded on the run.

    Returns:
        The run as persisted after the repair.

    Raises:
        NotFoundError: No run with that id in the scope.
        ValidationFailedError: The run is terminal on the authoritative re-read.
        ConflictError: The conditional write lost its race; nothing was repaired.
    """
    # No live task for a non-terminal run: the eval process restarted (or
    # crashed) while it was in flight, leaving the document stuck. Repair
    # it directly — the truthful terminal state is "cancelled by operator".
    detail = (reason or "cancelled by operator") + " (no live job — abandoned run repaired directly)"
    current, etag = storage.load_eval_run_with_etag(run_id, scope_id)
    if current is None:
        raise NotFoundError("run", run_id)
    # Self-defense, not just the earlier gate: the ETag reload is the
    # authoritative read for this write, so the invariant is enforced on it
    # rather than assumed from whatever window separated the first read from this one.
    require_cancellable(current)
    data = current.to_dict()
    data["status"] = "cancelled"
    data["cancellation_reason"] = detail
    data["completed_at"] = utc_now_iso()
    # The process that ran this died with its cell tally, so nothing recorded
    # how much of the matrix it delivered — and its scores would otherwise sit
    # on a short denominator with nothing saying so, which is exactly what the
    # kill that opened this cycle did to a sweep at 14 of 15. Reconstructed
    # from what reached storage, in the same write as the status so the repair
    # stays one conditional write. Never overwritten: a run whose own loop
    # recorded a tally before the process died has the better observation.
    if current.completeness is None:
        # The counts need each result's outcome and nothing it recorded, and a plain
        # by-run read now costs exactly that: the stored ``eval_result`` holds no
        # turn-by-turn record and no OTel spans — both live in sibling ``eval_trace``
        # documents this never fetches — so the read scales with how many cells there
        # were, not with what they observed. The condition-only projection this used to
        # call was deleted with the payloads it existed to leave behind.
        stored = storage.query_eval_results_by_run(run_id, scope_id)
        data["completeness"] = reconstruct_completeness(current, stored).to_dict()
    try:
        storage.save_eval_run(EvalRun.from_dict(data), if_match=etag)
    except ConflictError as e:
        # The conditional write lost its race, so nothing was repaired. Returning
        # normally here would report a repair that did not happen — and the startup
        # sweep, this function's other caller, would name the run in its "cancelled at
        # startup" log while the document still reads ``running``.
        raise ConflictError(f"run '{run_id}' changed while being cancelled — re-read it and retry") from e
    log.info("eval.cancel_run repaired abandoned run=%s scope=%s", run_id, scope_id)
    return get_run(storage, run_id, scope_id)


def sweep_abandoned_runs(
    storage: RunRecordStore, scopes: Iterable[str], *, job_manager: EvalJobManager | None
) -> AbandonedRunSweepReport:
    """Cancel runs this process cannot own, left non-terminal by a previous one.

    Cell and job timeouts both live *in the process* executing the run, so a
    hard kill (OOM, SIGKILL, node eviction) writes no terminal status: the
    document reads ``running`` forever and its pass^k silently sits on a
    partial denominator. This reclaims those documents.

    Intended to run **once at process start, before any run can be
    launched**: the job manager is empty at boot, so every non-terminal run
    this can see is abandoned by construction. Runs holding a live job here
    are skipped anyway rather than trusting that timing — the check costs
    nothing and keeps a healthy run safe from a mistimed second call.

    What that check cannot do is see *another process's* runs, because the
    job manager is process-local. So this must not run on more than one
    process against shared storage: the second one booting would find the
    first's healthy runs non-terminal, see no live job for them, and cancel
    them. The caller owns that gating.

    The repair itself is :func:`cancel_run`'s no-live-job branch, invoked
    rather than reimplemented: it re-reads under an ETag, re-checks the
    non-terminal invariant on that authoritative read, records the reason on
    ``cancellation_reason``, reconstructs the run's completeness from stored
    results, and stamps ``completed_at``. Results already persisted by the
    run stay valid and readable — this settles the run's status, it does not
    discard its work.

    A run that turns terminal (or is deleted) between the scan and its
    repair is counted as ``already_settled``, not an error: something else
    finished the job, which is the outcome this wanted. A storage failure is
    deliberately *not* caught here — swallowing it would leave the operator
    with runs that read ``running`` and a log line claiming they were swept.
    **That does not make it fatal to the caller**, and the boot path
    deliberately does not treat it that way: the web process binds and
    serves its health check independently of the database, so a database that
    cannot be reached at boot must not take the admin surface down with it.
    A host's boot path therefore catches its storage-unavailable error from
    this call, logs at ERROR and continues. The division is the point —
    this function refuses to lie about what it swept; its caller decides
    whether an unswept database is worth refusing to start over.

    A row that cannot be *reconstructed* is a third case, and it is not a
    storage failure: the database answered, and the answer is a document
    this build's model cannot express (a bare rename taken on the decision
    that the corpus is dropped). Tolerated per row in two places rather than
    raising for the list — the scan itself, and this loop, because settling a
    run reads more than the run and the completeness read still raises for a
    whole list. Both come back as ``unreadable``: counted, logged with their
    ids, and inside ``scanned``. The sweep can do nothing about them, and
    catching the second one is what keeps the first sentence of this
    docstring true — uncaught, a repair that cannot read aborts the sweep
    after earlier runs were already settled, and the caller then reports a
    sweep that never happened.

    Args:
        storage: The runs' store.
        scopes: The scopes to sweep, which the host names — the engine cannot enumerate the
            scopes a store holds and does not assume the store can. Each is scanned in turn.
        job_manager: The job manager that owns this process's live tasks; ``None`` when the
            caller was built without one.

    Returns:
        Counts plus the ids cancelled. All five reach the boot log: the two
        ids-and-scanned figures say what was reclaimed, and
        ``already_settled`` / ``skipped_live`` / ``unreadable`` say why a
        scan reclaimed less than it saw — a non-zero ``skipped_live`` in
        particular means this was invoked with work in flight, which the
        boot path never is, and a non-zero ``unreadable`` means the corpus
        has outlived the model that wrote it.

    Raises:
        RuntimeError: ``job_manager`` is ``None``: without one the liveness check is unanswerable,
            so sweeping would be guessing with cancellations.
    """
    if job_manager is None:
        raise RuntimeError(
            "sweep_abandoned_runs requires job_manager — without one, whether a run is live is unanswerable"
        )

    runs: list[EvalRun] = []
    unreadable_rows = 0
    for scope_id in scopes:
        scan = storage.query_non_terminal_eval_runs(scope_id)
        runs.extend(scan.runs)
        unreadable_rows += scan.unreadable
    cancelled: list[str] = []
    already_settled = 0
    skipped_live = 0
    unreadable = unreadable_rows
    for run in runs:
        if job_manager.is_active(run.id):
            skipped_live += 1
            continue
        try:
            cancel_run(storage, run.id, run.scope_id, job_manager=job_manager, reason=_ABANDONED_RUN_REASON)
        except NotFoundError, ValidationFailedError, ConflictError:
            # All three mean the same thing to this sweep: the run is no longer
            # ours to settle. ConflictError is the narrowest — the conditional
            # write lost its race, so something else wrote the document between
            # the scan and the repair. Counting it here rather than reporting a
            # cancellation keeps the boot log from naming a run it did not settle.
            already_settled += 1
            continue
        except ValidationError:
            # The run's OWN document read fine — the scan proved that — but the
            # repair reads more than the run: ``cancel_run`` reconstructs the
            # run's completeness from its stored results, and that read still
            # raises for a whole list. Caught per row for the reason the scan
            # isolates a refused row, and it is the sharper case: uncaught, this
            # propagates out of the sweep AFTER earlier runs were already
            # cancelled, and the boot log then reports a sweep that was skipped
            # — naming as unswept the very runs it settled. This function's own
            # contract is that it refuses to lie about what it swept.
            unreadable += 1
            log.exception(
                "Eval run %s (scope %s) was readable but its repair was not: a model "
                "rejected a value while settling it, so it stays stamped %s. The traceback "
                "is not raised here because the runs already settled by this sweep must "
                "still be reported.",
                run.id,
                run.scope_id,
                run.status,
            )
            continue
        cancelled.append(run.id)
    return AbandonedRunSweepReport(
        # Rows the scan MATCHED, not rows it could read — the unreadable ones
        # are the whole reason this figure and ``len(runs)`` are not the same
        # number, and leaving them out would make a boot that reclaimed
        # nothing report a scan that found nothing.
        scanned=len(runs) + unreadable_rows,
        cancelled_run_ids=cancelled,
        already_settled=already_settled,
        skipped_live=skipped_live,
        unreadable=unreadable,
    )


async def rejudge_result(
    host: EvalHost,
    result_id: str,
    scope_id: str,
) -> EvalResult:
    """Re-score a finished result's failed judge dimensions from the evidence its judge read.

    The dims re-asked are the ones the result holds neither a score nor a recorded "can't tell"
    for (see :func:`~threetears.evals.run.rejudge.failed_judge_dims`) — a can't-tell is the
    judge's answer and is never re-asked — and the outcomes land where the
    judge phase would have put them, with a :class:`~threetears.evals.schema.models.JudgeRescore`
    recording the re-judge on the result (:func:`~threetears.evals.run.rejudge.apply_rejudge`).

    **It re-asks the same judge the same question, or refuses.** What the judge reads about the
    candidate is the evidence the cell's kind rendered for the first judge, read back off the
    cell's stored trace — so a document is re-scored as the document it was, on its rubric
    alone, and nothing is re-rendered through a kind that may have changed since. Every other
    input is taken from what the run recorded at launch, through
    :func:`~threetears.evals.run.rejudge.reproducible_judge_inputs` — the one check the judge
    freeze makes too — with the request settings pinned to today's (``request_settings=
    "today"``), since a re-judge sends a call. Falling back to today's value for any
    input would score the dim under a different apparatus and file it under the old one,
    which is worse than leaving it missing. All refusals are raised before any judge
    call is paid for.

    Args:
        host: The host: the result's store, the judge clients it builds — through the same
            run-pinned resolution the launch scored through — and how a judge call that raised
            becomes the text its outcome keeps.
        result_id: The result to re-judge.
        scope_id: The partition the result's run lives in.

    Returns:
        The result as stored after the re-judge. Dims that failed again stay missing
        and are named in its recomposed ``judge_error``.

    Raises:
        NotFoundError: No result, run, template, judge config or test case with the recorded id.
        ValidationFailedError: The result has no failed judge dimension, or an apparatus
            input cannot be reproduced (:func:`~threetears.evals.run.rejudge.reproducible_judge_inputs`
            names which input and why).
        ConflictError: The rewrite did not land — something else wrote the result
            between the read and the write, or the write failed (storage reports both
            as one ``False``). Nothing was stored, and the judge spend is on the log line.
        ValueError: The host supplies no completion clients, so nothing could re-judge.
        StorageError: The result was re-scored and stored but its run's excluded-cell count
            could not be updated.
    """
    storage = host.storage
    clients = host.completion_clients("a re-judge")
    # Every store hands a found document's etag back (the port's contract), so the rewrite below is
    # conditional: a re-judge racing another writer of this result is refused rather than clobbering it.
    result, etag = storage.load_eval_result_with_etag(result_id, scope_id)
    if result is None:
        raise NotFoundError("result", result_id)
    if result.judge_error is None:
        raise ValidationFailedError(f"result '{result_id}' has no judge_error — no judge dimension failed to re-judge")
    run = get_run(storage, result.eval_run_id, scope_id)
    failed = failed_judge_dims(result, list(run.effective_judges or {}))
    inputs = reproducible_judge_inputs(storage, result, run, scope_id, request_settings="today", config_dims=failed)
    # Narrowed to the dims this result was judged on: a document's run attributes the two
    # conversation axes, which its cell was never asked and so holds no score for.
    failed = [dim for dim in failed if dim in inputs.dims]
    if not failed:
        raise ValidationFailedError(
            f"result '{result_id}' carries a judge_error but holds a score or a can't-tell for every dim its run "
            "judged, so there is no dimension to re-judge"
        )
    template = inputs.template

    context = build_judge_context(
        template=template,
        test_case=inputs.test_case,
        goal_outcomes=result.goal_state_outcomes,
        judged_artifact=inputs.judged_artifact,
        judge_evidence=inputs.judge_evidence,
    )
    judge_service = JudgeService(
        client_factory=judge_clients_for_run(clients, inputs.judge_model),
        configs=inputs.configs,
        failure_describer=host.failure_describer,
    )
    async with judge_service:
        outcomes = await judge_dims(
            template=template, judge_service=judge_service, context=context, only=frozenset(failed)
        )

    updated = apply_rejudge(result, template, outcomes, judge_model=inputs.judge_model)
    rescore = updated.judge_rescores[-1]
    try:
        storage.replace_eval_result(updated, if_match=etag)
    except (ConflictError, StorageError) as e:
        log.error(
            "eval.rejudge_result lost its write result=%s run=%s — the re-judge cost %s and nothing was stored",
            result_id,
            run.id,
            _dollars(rescore.cost_usd),
        )
        if isinstance(e, ConflictError):
            raise ConflictError(
                f"result '{result_id}' changed while it was being re-judged; nothing was stored "
                "— re-read it and re-judge again if a dim is still missing"
            ) from e
        raise
    _recount_rejudged_exclusion(storage, run.id, scope_id, before=result, after=updated)
    log.info(
        "eval.rejudge_result result=%s run=%s dims=%s scored=%s failed_again=%s cost=%s",
        result_id,
        run.id,
        ",".join(rescore.dims),
        ",".join(f"{dim}={score}" for dim, score in rescore.scores.items()) or "-",
        ",".join(rescore.errors) or "-",
        _dollars(rescore.cost_usd),
    )
    return updated


def _dollars(cost_usd: float | None) -> str:
    """A spend for a log line: dollars, or ``unpriced`` — never a zero standing in for unknown."""
    return f"${cost_usd:.6f}" if cost_usd is not None else "unpriced"


def _recount_rejudged_exclusion(
    storage: JobStore, run_id: str, scope_id: str, *, before: EvalResult, after: EvalResult
) -> None:
    """Keep the run's completeness true when a re-judge moves a result out of exclusion.

    ``RunCompleteness.infra_excluded_cells`` counts persisted cells a harness failure
    (a judge failure among them) removed from the aggregates. A re-judge that scores a
    result's last failed dim returns that cell to the aggregates, and a count left as it
    was would go on reporting the run as short a cell it now has. Only that transition
    moves the count: a re-judge that fails again, or a result another error still
    excludes, leaves it as it was.

    The run write is conditional and retried by re-reading and re-applying, because the
    run document has other writers. It runs after the result write has landed, so a
    failure here is reported as the run's count being stale, never as the re-judge failing.

    Raises:
        StorageError: The result is stored and re-scored, and the run's count could not be
            updated after every attempt.
    """
    if (
        classify_result(before) is not ResultOutcome.INFRA_EXCLUDE
        or classify_result(after) is ResultOutcome.INFRA_EXCLUDE
    ):
        return
    for _ in range(3):
        run, etag = storage.load_eval_run_with_etag(run_id, scope_id)
        if run is None:
            raise NotFoundError("run", run_id)
        if run.completeness is None or run.completeness.infra_excluded_cells == 0:
            return
        recounted = run.completeness.model_copy(
            update={"infra_excluded_cells": run.completeness.infra_excluded_cells - 1}
        )
        try:
            storage.save_eval_run(run.model_copy(update={"completeness": recounted}), if_match=etag)
        except ConflictError:
            # NOSILENT: a lost conditional write is re-read and re-applied; exhausting the retries is logged and raised below
            continue
        return
    log.error("eval.rejudge_result run=%s completeness NOT recounted after a re-judge scored a result", run_id)
    raise StorageError(
        f"the re-judged result is stored, but run '{run_id}' still counts it as an excluded cell — its "
        "completeness write kept losing to another writer"
    )


def record_completeness(
    storage: JobStore,
    run_id: str,
    scope_id: str,
    cells: Sequence[CellSummary],
    *,
    metered: MeteredCallTally | None = None,
) -> None:
    """Stamp how much of its matrix a run's loop delivered, and what bounded it.

    Called once per run, from the work function's ``finally`` — so a run
    stopped mid-flight is recorded too, which is the case the record most
    needs to describe. The counts are taken from what the loop appended
    rather than from the rows in storage, because a cell whose write failed
    left no row to count and is exactly the shortfall this exists to
    disclose. The one stop this cannot reach is a hard kill, which leaves no
    frame to run a ``finally``; those runs are reclaimed in a later process,
    where the counts have to be reconstructed from storage instead
    (:func:`~threetears.evals.kernel.scoring.reconstruct_completeness`).

    Runs BEFORE the job manager stamps the terminal status (which does its own
    read-modify-write of the same document), so the record is in the document
    the status lands on — and that adjacency is why the write is retried
    rather than attempted once: the two writers contend by construction, and a
    conditional write refused by the other one used to end this function with
    nothing recorded and a log line nobody reads.

    A failure to record is logged, never raised — a storage exception
    included, which is why the whole body sits inside the guard rather than
    only the outcome it returns. This runs inside the job's work function, so
    anything escaping here reaches
    :meth:`~threetears.evals.run.jobs.EvalJobManager._run_job`'s top-level boundary
    and stamps ``failed`` on a run that executed and persisted its entire
    matrix. Losing a disclosure line is strictly better than mislabelling
    the run it describes.

    Args:
        storage: The run's store.
        run_id: The run whose loop just finished.
        scope_id: The partition the run lives in.
        cells: One summary per cell the loop ran.
        metered: The run ledger's final tally of metered third-party calls, or
            ``None`` for a caller with no ledger. Written on the SAME document
            write as the completeness record because it answers the same
            question from the other side: completeness says how much of the
            matrix was delivered, this says whether what was delivered was
            measured against a world the candidate could fully reach. Both are
            lost if the write is lost, so pairing them costs nothing and keeps
            the two from disagreeing about which attempt won the race. Unlike
            the completeness counts it is NOT recomputed per attempt — the
            tally belongs to the run's ledger, not to whichever document
            version this write lands on.
    """
    record: RunCompleteness | None = None

    def _stamp(current: EvalRun) -> dict[str, Any]:
        # Recomputed per attempt, against whichever document won the race:
        # the denominator is read from the run, so a re-apply that reused the
        # first read's counts would write a record describing a document that
        # is no longer there.
        nonlocal record
        record = summarize_completeness(current, cells)
        data = current.to_dict()
        data["completeness"] = record.to_dict()
        if metered is not None:
            # Zero is written, not skipped: "the ceiling was in force and never
            # bound this run" is a fact an operator needs, and is what separates
            # it from a run whose loop counted nothing.
            data["metered_calls_refused"] = metered.refused
        return data

    try:
        outcome = update_eval_run(storage, run_id, scope_id, _stamp)
    # prawduct:ok-broad-except — any storage fault here would mislabel a run that succeeded
    except Exception:
        log.exception(
            "eval.completeness run=%s scope=%s NOT recorded — the run keeps its real status "
            "and its completeness reads as unknown",
            run_id,
            scope_id,
        )
        return

    if outcome == "missing":
        log.error("eval.completeness run=%s scope=%s vanished before its record could be written", run_id, scope_id)
        return
    if outcome == "refused":
        log.error(
            "eval.completeness run=%s scope=%s NOT recorded (no attempt completed — a refused write "
            "or a failing read, whichever the eval.update_run lines above show) — "
            "its completeness will read as unknown",
            run_id,
            scope_id,
        )
        return

    if record is not None and record.degraded:
        log.warning(
            "eval.completeness run=%s DEGRADED: %d of %d cells measured (produced=%d persisted=%d infra_excluded=%d)",
            run_id,
            record.measured_cells,
            record.expected_cells,
            record.produced_cells,
            record.persisted_cells,
            record.infra_excluded_cells,
        )


__all__ = [
    "AbandonedRunSweepReport",
    "cancel_run",
    "get_run",
    "record_completeness",
    "rejudge_result",
    "repair_abandoned_run",
    "require_cancellable",
    "sweep_abandoned_runs",
]
