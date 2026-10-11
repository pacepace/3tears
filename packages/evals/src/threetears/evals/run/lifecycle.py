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
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import ValidationError

from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.kernel.errors import ConflictError, NotFoundError, StorageError, ValidationFailedError
from threetears.evals.schema.models import NON_TERMINAL_RUN_STATUSES, EvalRun
from threetears.evals.kernel.result_condition import ResultOutcome, classify_result
from threetears.evals.kernel.scoring import summarize_completeness
from threetears.evals.kernel.storage import LeaseStore, NonTerminalRunScan
from threetears.evals.run.judge_service import JudgeService, judge_clients_for_run
from threetears.evals.kernel.job_leases import RUN_JOB, request_job_cancel, settle_job_record
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
    from threetears.evals.kernel.storage import JobStore, RunStore

log = get_logger(__name__)


class ReclaimStore(LeaseStore, Protocol):
    """What :func:`sweep_abandoned_runs` reads and writes: the boot scan of runs not ended, and the job records."""

    def query_non_terminal_eval_runs(self, scope_id: str, /) -> NonTerminalRunScan:
        """See :meth:`~threetears.evals.kernel.EvalStorage.query_non_terminal_eval_runs`."""
        ...


class AbandonedRunSweepReport(EvalBaseModel):
    """Outcome of one reclaim of runs left non-terminal by a process that has gone (:func:`sweep_abandoned_runs`)."""

    scanned: int = 0
    """Non-terminal rows the scan MATCHED, summed over every scope it was told to sweep.

    Rows matched, not runs returned: a row the scan could not reconstruct is
    counted here too, via ``unreadable``. Under the older reading — runs the scan
    handed back — a boot facing a wholly unreadable corpus would report a scan
    that found nothing, which is the one case where it found the most. The four
    outcome figures (``cancelled_run_ids``, ``already_settled``, ``skipped_live``,
    ``unreadable``) sum to this."""

    cancelled_run_ids: list[str] = []
    """Runs this sweep wrote lost — status ``cancelled``, the loss recorded on ``lost``.

    Named rather than counted: a sweep that ends a run an operator was
    watching has to be able to say which one."""

    already_settled: int = 0
    """Scanned runs that were terminal or gone by the time their repair ran.

    Not a failure — something else reached them first, which is the state this
    sweep wanted them in."""

    skipped_live: int = 0
    """Scanned runs whose lease is live — some process, this one or another, is running them — and so left alone."""

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
    storage: LeaseStore,
    run_id: str,
    scope_id: str,
    *,
    job_manager: EvalJobManager | None,
    reason: str | None = None,
) -> EvalRun:
    """Ask a pending/running eval run to stop so it stops burning cost and quota — from any process.

    The run is read as every process reads it (:func:`~threetears.evals.kernel.job_leases.settle_job_record`): one whose
    lease has lapsed is written lost first, and has then ended. A run that has not ended gets the cancel written on
    its record (:func:`~threetears.evals.kernel.job_leases.request_job_cancel`), which the process holding it reads at its
    next lease renewal; when that process is this one, ``job_manager`` cancels it at once as well. Either way the
    job's own ``CancelledError`` boundary writes ``cancelled`` (plus ``reason`` into ``cancellation_reason``) as
    the task unwinds. Request-then-converge: the run returned here may still read ``running`` for up to one
    renewal interval — confirm via :func:`get_run`.

    The reason goes to ``cancellation_reason`` and never to ``error_details``: a cancel is a distinct terminal
    outcome, so an operator's decision to stop is readable apart from the things that broke. Results already
    persisted by the run remain valid and readable.

    Args:
        storage: The run's store.
        run_id: Run to cancel.
        scope_id: The partition the run lives in.
        job_manager: This process's job manager, to cancel a run it runs without waiting for a renewal; ``None``
            when the caller has none, and the cancel lands through the record alone.
        reason: Optional operator-facing reason recorded on the run.

    Returns:
        The run as persisted after the cancel request (see convergence note above).

    Raises:
        NotFoundError: No run with that id in the scope.
        ValidationFailedError: The run has already ended (``completed`` / ``failed`` / ``cancelled`` /
            ``budget_stopped`` / ``exhausted``) — lost included, which this read may have just written.
        ConflictError: The run kept changing while its lapsed lease was being written lost; nothing was decided.
        StorageError: A read or the request's write failed.
    """
    run, _ = settle_job_record(storage, RUN_JOB, run_id, scope_id)
    if run is None:
        raise NotFoundError("run", run_id)
    require_cancellable(run)
    request_job_cancel(storage, run_id, scope_id, reason=reason)
    if job_manager is not None and job_manager.cancel_job(run_id, reason=reason):
        log.info("eval.cancel_run requested run=%s scope=%s reason=%s", run_id, scope_id, reason or "(none)")
    else:
        log.info(
            "eval.cancel_run asked run=%s scope=%s reason=%s on its record — its holder reads it at its next renewal",
            run_id,
            scope_id,
            reason or "(none)",
        )
    current, _ = storage.load_job_record(EvalRun, run_id, scope_id)
    if current is None:
        raise NotFoundError("run", run_id)
    return current


def require_cancellable(run: EvalRun) -> None:
    """Refuse a cancel of a run that has already ended — :func:`cancel_run`'s first check, for a caller that splits it.

    Args:
        run: The run as read.

    Raises:
        ValidationFailedError: The run is already terminal; a lost run says so, with its cause.
    """
    if run.status not in NON_TERMINAL_RUN_STATUSES:
        lost = f"; it was lost: {run.lost.cause}" if run.lost is not None else ""
        raise ValidationFailedError(
            f"run '{run.id}' is {run.status} — only pending/running runs can be cancelled{lost}"
        )


def sweep_abandoned_runs(storage: ReclaimStore, scopes: Iterable[str]) -> AbandonedRunSweepReport:
    """Write lost every run left not ended by a process that has gone, in the scopes the host names.

    Cell and job timeouts both live *in the process* executing the run, so a hard kill (OOM, SIGKILL, node
    eviction) writes no terminal status: the document reads ``running`` and its pass^k sits on a partial
    denominator. Its lease is how anyone can tell: a run whose lease has lapsed (or that carries none, written by
    a build before leases) is lost, and this writes it so — the same read every process makes
    (:func:`~threetears.evals.kernel.job_leases.settle_job_record`), here over every run a scan finds not ended, rather
    than waiting for someone to poll each. A run whose lease is live is running in SOME process, this one or
    another, and is left alone; so this may run at any time, on every process sharing a store, at once.

    The write sets the run ``cancelled`` with the loss as its ``cancellation_reason``, records the loss
    (``lost``), stamps ``completed_at``, and reconstructs its completeness from stored results unless its loop
    recorded its own. Results already persisted by the run stay valid and readable — this settles the run's
    status, it does not discard its work.

    A run that ends (or is deleted) between the scan and its write is counted as ``already_settled``, not an
    error: something else finished the job, which is the outcome this wanted. A storage failure is deliberately
    *not* caught here — swallowing it would leave the operator with runs that read ``running`` and a log line
    claiming they were swept. **That does not make it fatal to the caller**: a host's boot path catches its
    storage-unavailable error from this call, logs at ERROR and continues, since a database that cannot be reached
    at boot must not take the admin surface down with it. This function refuses to lie about what it swept; its
    caller decides whether an unswept database is worth refusing to start over.

    A row that cannot be *reconstructed* is a third case, and it is not a storage failure: the database answered,
    and the answer is a document this build's model cannot express. Tolerated per row in two places rather than
    raising for the list — the scan itself, and this loop, because settling a run reads more than the run and the
    completeness read still raises for a whole list. Both come back as ``unreadable``: counted, logged with their
    ids, and inside ``scanned``.

    Args:
        storage: The runs' store.
        scopes: The scopes to sweep, which the host names — the engine cannot enumerate the scopes a store holds
            and does not assume the store can. Each is scanned in turn.

    Returns:
        Counts plus the ids written lost. ``already_settled`` / ``skipped_live`` / ``unreadable`` say why a scan
        wrote less than it saw — ``skipped_live`` counts runs some process holds a live lease on, and a non-zero
        ``unreadable`` means the corpus has outlived the model that wrote it.
    """
    runs: list[EvalRun] = []
    unreadable_rows = 0
    for scope_id in scopes:
        scan = storage.query_non_terminal_eval_runs(scope_id)
        runs.extend(scan.runs)
        unreadable_rows += scan.unreadable
    lost: list[str] = []
    already_settled = 0
    skipped_live = 0
    unreadable = unreadable_rows
    for run in runs:
        try:
            settled, wrote_lost = settle_job_record(storage, RUN_JOB, run.id, run.scope_id)
        except ConflictError:
            # NOSILENT: the run kept changing under the write, so something else is writing it — not ours to settle
            already_settled += 1
            continue
        except ValidationError:
            # The run's OWN document read fine — the scan proved that — but writing it lost reads more than the
            # run: its completeness is reconstructed from its stored results, and that read still raises for a
            # whole list. Caught per row, so the runs already settled by this sweep are still reported.
            unreadable += 1
            log.exception(
                "Eval run %s (scope %s) was readable but writing it lost was not: a model rejected a value while "
                "settling it, so it stays stamped %s.",
                run.id,
                run.scope_id,
                run.status,
            )
            continue
        if wrote_lost:
            lost.append(run.id)
        elif settled is not None and not RUN_JOB.ended(settled):
            skipped_live += 1
        else:
            already_settled += 1
    return AbandonedRunSweepReport(
        # Rows the scan MATCHED, not rows it could read — the unreadable ones are the whole reason this figure and
        # ``len(runs)`` are not the same number.
        scanned=len(runs) + unreadable_rows,
        cancelled_run_ids=lost,
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

    def _stamp(current: EvalRun) -> dict[str, Any] | None:
        if current.lost is not None:
            # The run was written lost — its lease lapsed and a reader in some process recorded it, or this
            # process shut down — so this loop no longer holds it and writes nothing more; the loss's write
            # reconstructed the record from what reached storage.
            return None
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
    if outcome == "declined":
        log.warning(
            "eval.completeness run=%s scope=%s not recorded by its loop: the run was written lost, and its "
            "completeness is the one reconstructed then",
            run_id,
            scope_id,
        )
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
    "ReclaimStore",
    "cancel_run",
    "get_run",
    "record_completeness",
    "rejudge_result",
    "require_cancellable",
    "sweep_abandoned_runs",
]
