"""The run lifecycle: reading a run, cancelling it, reclaiming abandoned ones, re-judging a result, and recording completeness.

A run reclaimed here is settled, never continued: it reads lost, and its lease is never revived. The one way back is
an explicit resume under its lease's next epoch (:func:`~threetears.evals.run.resume.take_up_run`); what a resume would take
up is listed by :func:`~threetears.evals.run.resumable_jobs`, which a host offers to an operator at boot.

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

import math
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

from pydantic import ValidationError

from threetears.evals.kernel.numbers import round_half_even
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.kernel.errors import ConflictError, NotFoundError, StorageError, ValidationFailedError
from threetears.evals.schema.models import CLAIM_PENDING, NON_TERMINAL_RUN_STATUSES, EvalRun
from threetears.evals.kernel.result_condition import ResultOutcome, classify_result
from threetears.evals.kernel.scoring import summarize_completeness
from threetears.evals.kernel.storage import LeaseStore, NonTerminalRunScan
from threetears.evals.kernel.claims.derived import ClaimDerivation, claim_derivations
from threetears.evals.kernel.host.measures import ClaimReaderDefect
from threetears.evals.kernel.claims.grading import GradedCriterion, finish_web_step, grade_or_loss
from threetears.evals.kernel.claims.web import SearchBill, query_price, search_bill, send_search
from threetears.evals.kernel.usage_capture import ExternalRateTable
from threetears.evals.schema.web_search import WebSearch, WebSearchResult
from threetears.evals.kernel.host.apparatus import ApparatusError
from threetears.evals.kernel.usage_capture import CallUsage
from threetears.evals.run.judge_service import JudgeFold, JudgeService, fold_judge_outcomes, judge_clients_for_run
from threetears.evals.schema.completion import CompletionProtocolDefect, CompletionResult
from threetears.evals.kernel.job_leases import RUN_JOB, JobLeases, settle_job_record
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.run.rejudge import (
    ReproducibleClaimInputs,
    settled_pending_claims,
    apply_claim_rejudge,
    apply_rejudge,
    claims_due_rejudge,
    failed_judge_dims,
    recorded_judge_pins,
    refuse_claims_over_the_trace,
    reproducible_claim_inputs,
    reproducible_judge_inputs,
)
from threetears.evals.run.run_document import fault_entry, record_run_faults, update_eval_run
from threetears.evals.run.runner import build_judge_context, judge_dims
from threetears.evals.schema.logs import contain, get_logger
from threetears.evals.kernel.remedies import FIND_RESULT, FIND_RUN, FIND_TEMPLATE

if TYPE_CHECKING:
    from threetears.evals.schema.models import EvalResult, RunCompleteness
    from threetears.evals.kernel.host.eval_host import EvalHost
    from threetears.evals.kernel.scoring import CellSummary
    from threetears.evals.run.metering import MeteredCallTally
    from threetears.evals.kernel.storage import EvalStorage, JobStore, RunStore
    from threetears.evals.run.rejudge import ReproducibleJudgeInputs

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

    :param storage: The run's store.
    :param run_id: The run to load.
    :param scope_id: The partition the run lives in.
    :return: The run.
    :raises NotFoundError: No run with that id in the scope.
    """
    run = storage.load_eval_run(run_id, scope_id)
    if run is None:
        raise NotFoundError("run", run_id, remedy=FIND_RUN)
    return run


def require_cancellable(run: EvalRun) -> None:
    """Refuse a cancel of a run that has already ended — a cancel's first check, for a caller that splits it.

    :param run: The run as read.
    :raises ValidationFailedError: The run is already terminal; a lost run says so, with its cause.
    """
    if run.status not in NON_TERMINAL_RUN_STATUSES:
        lost = f"; it was lost: {run.lost.cause}" if run.lost is not None else ""
        raise ValidationFailedError(
            f"run '{run.id}' is {run.status} — only pending/running runs can be cancelled{lost}",
            remedy="read how it ended with run_get",
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

    **It settles a run; it never continues one.** A host offers that at boot, after this sweep: it lists what a
    resume would take up (:func:`~threetears.evals.run.resumable_jobs`, the ``jobs_resumable`` action, over the
    same scopes) and offers each to an operator, and a run is resumed only when someone asks
    (:func:`~threetears.evals.run.resume.take_up_run`, the ``job_resume`` action). Nothing resumes on its own.

    A row that cannot be *reconstructed* is a third case, and it is not a storage failure: the database answered,
    and the answer is a document this build's model cannot express. Tolerated per row in two places rather than
    raising for the list — the scan itself, and this loop, where writing a run lost may be refused by a model. Both
    come back as ``unreadable``: counted, logged with their ids, and inside ``scanned``. A run whose RESULTS cannot
    all be read is not one of them: it is written lost with its completeness unknown, since settling reads nothing
    past the run that can fail it (:func:`~threetears.evals.kernel.job_leases.settle_tolerantly`).

    :param storage: The runs' store.
    :param scopes: The scopes to sweep, which the host names — the engine cannot enumerate the scopes a store holds
        and does not assume the store can. Each is scanned in turn.
    :return: Counts plus the ids written lost. ``already_settled`` / ``skipped_live`` / ``unreadable`` say why a scan
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
            # The run's OWN document read fine — the scan proved that — but a model refused a value while it was
            # written lost. Caught per row, so the runs already settled by this sweep are still reported.
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
    *,
    web_search: WebSearch | None = None,
    external_rates: ExternalRateTable | None = None,
) -> EvalResult:
    """Re-score a finished result's failed judge dimensions and lost claim criteria from the evidence its judge read.

    The dims re-asked are the ones the result holds neither a score nor a recorded "can't tell"
    for (see :func:`~threetears.evals.run.rejudge.failed_judge_dims`) — a can't-tell is the
    judge's answer and is never re-asked — and the outcomes land where the
    judge phase would have put them, with a :class:`~threetears.evals.schema.models.JudgeRescore`
    recording the re-judge on the result (:func:`~threetears.evals.run.rejudge.apply_rejudge`).

    **Claim criteria are re-judged beside them.** Each criterion the result lost (``EvalResult.claim_failures``) is
    graded again whole from the claim evidence its trace stores, in the template's order
    (:func:`~threetears.evals.run.rejudge.reproducible_claim_inputs`). A completed grade is never asked again, but for
    one thing: a grade whose web step was lost alone (its tally's ``web_loss``) has that step finished, and only that
    step (:func:`~threetears.evals.kernel.claims.grading.finish_web_step`), over the claims it left lost, live through
    ``web_search``: its truth and ground steps stand as stored, and no claim it checked is searched again. A criterion
    or web step lost to the trace's cap (``evidence_over_cap``) is never graded again, since its evidence is the same:
    a re-judge with nothing else to do refuses it by name, before any call, and one that grades others names it in the
    rescore's ``skipped`` and on its log line. The new grades are written to the trace
    before the result, each write conditional on the document as read, and a
    :class:`~threetears.evals.schema.models.ClaimRescore` records it on the result
    (:func:`~threetears.evals.run.rejudge.apply_claim_rejudge`). A criterion lost again is recorded with its new loss,
    and a web step lost again with its new ``web_loss``.

    **A pending criterion is settled first, with no call and under no apparatus check**
    (:func:`~threetears.evals.run.rejudge.settled_pending_claims`): its tally recomputed from the grade its trace
    stores, or ``trace_lost`` where its trace was never stored. That write lands even when the rest of the re-judge
    is then refused, since it asks nothing an edited template or moved request settings could change.

    **It re-asks the same judge the same question, or refuses.** What the judge reads about the
    candidate is the evidence the cell's kind rendered for the first judge, read back off the
    cell's stored trace — so a document is re-scored as the document it was, on its rubric
    alone, and nothing is re-rendered through a kind that may have changed since. Every other
    input is taken from what the run recorded at launch, through
    :func:`~threetears.evals.run.rejudge.reproducible_judge_inputs` — the one check the judge
    freeze makes too — with the request settings pinned to today's (``request_settings=
    "today"``), since a re-judge sends a call. A claim criterion is graded again only under the definition the cell
    recorded for it (``EvalResult.claim_fingerprints``). Falling back to today's value for any
    input would score the dim under a different apparatus and file it under the old one,
    which is worse than leaving it missing. All refusals are raised before any judge
    call is paid for.

    :param host: The host: the result's store, the judge clients it builds — through the same
        run-pinned resolution the launch scored through — and how a judge call that raised
        becomes the text its outcome keeps.
    :param result_id: The result to re-judge.
    :param scope_id: The partition the result's run lives in.
    :param web_search: The web search a lost criterion with a web budget is graded again on, its web step included,
        and a lost web step is finished on, live: the run's own port, or a recording of it for a replay. ``None``
        refuses either by name before any call. A completed check is never searched again.
    :param external_rates: The run's rate table, which prices a search its port reports no cost for, as the run did:
        a search is billed at what it reports, else at the rate declared for ``(identity, "query")``, else at the
        port's ``price_ceiling``. ``None`` declares none.
    :return: The result as stored after the re-judge. Dims that failed again stay missing
        and are named in its recomposed ``judge_error``; criteria lost again stay lost, with their new loss.
    :raises NotFoundError: No result, run, template, judge config or test case with the recorded id.
    :raises ValidationFailedError: The result has neither a failed judge dimension nor a lost claim criterion or web
        step, or only ones lost to the trace's cap, or an apparatus input cannot be reproduced
        (:func:`~threetears.evals.run.rejudge.reproducible_judge_inputs` and
        :func:`~threetears.evals.run.rejudge.reproducible_claim_inputs` name which input and why).
    :raises ConflictError: A rewrite did not land — something else wrote the result, or its trace, between the read
        and the write. Its message says what was stored: nothing, when the trace's write lost; the new grades on the
        trace and not the result, when the result's lost after the trace's landed. The judge spend is on the log line.
    :raises ValueError: The host supplies no completion clients, so nothing could re-judge.
    :raises CompletionProtocolDefect: The judge client's completion does not meet the protocol — the host's defect.
        Nothing is stored; the spend is on the log line.
    :raises ApparatusError: The rig could not serve a claims call (a replay's miss). Nothing is stored.
    :raises ClaimReaderDefect: The host's reader of a claim criterion raised, or answered what its measures cannot
        hold — the host's defect. Nothing of the re-judge is stored.
    :raises StorageError: The result was re-scored and stored but its run's excluded-cell count
        could not be updated, or a claims re-judge's trace did not land (then nothing was stored).
    """
    storage = host.storage
    # Every read and write below is a store call, so each runs on the host's blocking executor, off the loop.
    settled, were_pending = await run_blocking(
        host.blocking_executor, _settle_before_rejudge, storage, result_id, scope_id, claim_derivations(host.profile)
    )
    # Settled, and nothing else failed: a criterion settled lost here had no stored trace, which nothing can grade.
    if (
        settled is not None
        and settled.judge_error is None
        and set(settled.claim_failures) <= were_pending
        and not claims_due_rejudge(settled)[1]
    ):
        return settled
    clients = host.completion_clients("a re-judge")
    read = await run_blocking(host.blocking_executor, _rejudge_inputs, storage, result_id, scope_id)
    result, etag, run, inputs, failed = read.result, read.etag, read.run, read.inputs, read.failed
    judge_model = (
        inputs.judge_model
        if inputs is not None
        else recorded_judge_pins(run, request_settings="today", claims_only=True)
    )
    judge_service = JudgeService(
        client_factory=judge_clients_for_run(clients, judge_model),
        configs=inputs.configs if inputs is not None else {},
        failure_describer=host.failure_describer,
    )
    updated = result
    trace_etag: str | None = None
    claims_due = any(claims_due_rejudge(result))
    async with judge_service:
        if claims_due:
            # Read before the inputs read the trace, so a trace another writer moved since is refused at its rewrite.
            _, trace_etag = await run_blocking(
                host.blocking_executor, storage.load_eval_trace_with_etag, result.id, scope_id
            )
        # Inside the service, so a refusal still releases the client its fingerprints were read through.
        claim_inputs = (
            await run_blocking(
                host.blocking_executor,
                partial(reproducible_claim_inputs, client=judge_service.default_client(), web=web_search),
                storage,
                result,
                run,
                scope_id,
            )
            if claims_due
            else None
        )
        if inputs is not None:
            context = build_judge_context(
                template=inputs.template,
                test_case=inputs.test_case,
                goal_outcomes=result.goal_state_outcomes,
                judged_artifact=inputs.judged_artifact,
                judge_evidence=inputs.judge_evidence,
            )
            outcomes = await judge_dims(
                template=inputs.template, judge_service=judge_service, context=context, only=frozenset(failed)
            )
            updated = apply_rejudge(result, inputs.template, outcomes, judge_model=inputs.judge_model)
        trace = None
        if claim_inputs is not None:
            billing = _BilledClaimsClient(judge_service.default_client(), web=web_search, rates=external_rates)
            try:
                regraded = await _regrade_claims(
                    claim_inputs, billing, result_id=result_id, derivations=claim_derivations(host.profile)
                )
            except CompletionProtocolDefect, ApparatusError, ClaimReaderDefect:
                log.error(
                    "eval.rejudge_result result=%s run=%s — a claims call could not be read or served, so nothing was "
                    "stored; the claims calls cost %s",
                    result_id,
                    run.id,
                    _dollars(_claims_fold(billing).cost_usd),
                )
                raise
            fold = _claims_fold(billing)
            updated, trace = apply_claim_rejudge(
                updated, claim_inputs, regraded, usage=fold.usage, cost_usd=fold.cost_usd
            )
    spent = _rejudge_spend(updated, before=result)
    if trace is not None:
        try:
            # The grades first, conditionally on the trace as read: no tally the rewrite lands stands before the grade
            # it is read from, and a trace another re-judge rewrote meanwhile is never written over.
            await run_blocking(host.blocking_executor, partial(storage.replace_eval_trace, trace, if_match=trace_etag))
        except (ConflictError, StorageError) as e:
            log.error(
                "eval.rejudge_result lost its trace write result=%s run=%s — the re-judge cost %s and nothing was stored",
                result_id,
                run.id,
                _dollars(spent),
            )
            if isinstance(e, ConflictError):
                raise ConflictError(
                    f"result '{result_id}''s trace changed while it was being re-judged; nothing was stored",
                    remedy="re-read it and re-judge again if a claim criterion is still lost",
                ) from e
            raise
    try:
        await run_blocking(host.blocking_executor, storage.replace_eval_result, updated, if_match=etag)
    except (ConflictError, StorageError) as e:
        stored = (
            "its trace holds the new grades and its result was not rewritten"
            if trace is not None
            else "nothing was stored"
        )
        log.error(
            "eval.rejudge_result lost its result write result=%s run=%s — the re-judge cost %s, and %s",
            result_id,
            run.id,
            _dollars(spent),
            stored,
        )
        if isinstance(e, ConflictError):
            raise ConflictError(
                f"result '{result_id}' changed while it was being re-judged; {stored}",
                remedy="re-read it and re-judge again if a dim is still missing or a claim criterion still lost",
            ) from e
        raise
    await run_blocking(
        host.blocking_executor, _recount_rejudged_exclusion, storage, run.id, scope_id, before=result, after=updated
    )
    if inputs is not None:
        rescore = updated.judge_rescores[-1]
        log.info(
            "eval.rejudge_result result=%s run=%s dims=%s scored=%s failed_again=%s cost=%s",
            result_id,
            run.id,
            ",".join(rescore.dims),
            ",".join(f"{dim}={score}" for dim, score in rescore.scores.items()) or "-",
            ",".join(rescore.errors) or "-",
            _dollars(rescore.cost_usd),
        )
    if claim_inputs is not None:
        claims = updated.claim_rescores[-1]
        log.info(
            "eval.rejudge_result result=%s run=%s criteria=%s lost_again=%s skipped=%s cost=%s",
            result_id,
            run.id,
            ",".join(claims.criteria),
            ",".join(f"{name}={loss}" for name, loss in claims.lost.items()) or "-",
            ",".join(f"{name}={loss}" for name, loss in claims.skipped.items()) or "-",
            _dollars(claims.cost_usd),
        )
    return updated


class _BilledClaimsClient:
    """The judge's client as a claims re-judge's calls see it: each completion billed once, as it is read.

    A completion whose ``sent_settings`` is the wrong type is billed for its spend alone before the defect is raised to
    the step that asked, so what the re-judge paid is still on its log line.
    """

    def __init__(self, inner: Any, *, web: WebSearch | None = None, rates: ExternalRateTable | None = None) -> None:
        """Bind the judge's client, and the web search a criterion with a web budget searches on.

        :param inner: The client, which names its model.
        :param web: The web search, or ``None``.
        :param rates: The run's rate table, which prices a search its port reports no cost for, or ``None``.
        """
        self._inner = inner
        self._web = web
        self._query_price = query_price(web, rates) if web is not None else (None, None)
        self.model_name: str = getattr(inner, "model_name", "")
        #: Every call's usage, in order.
        self.billed: list[CallUsage] = []
        #: Every search sent, in order.
        self.searched: list[SearchBill] = []

    async def search(self, query: str) -> WebSearchResult:
        """Send one search to the web port once and bill it, answered or not
        (:func:`~threetears.evals.kernel.claims.web.send_search`).

        :raises ClaimGradingFailed: The port failed it (``port_error``), the web step's to read.
        :raises ApparatusError: The port could not serve it from its recording (a replay's miss).
        """
        web = self._web
        if web is None:
            raise RuntimeError("a claims re-judge was asked to search with no web port bound")
        return await send_search(
            web,
            query,
            criterion="a re-judged criterion",
            bill=lambda reported: self.searched.append(search_bill(web, reported, self._query_price)),
        )

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> CompletionResult:
        """Send one call and bill what it returns.

        :raises CompletionProtocolDefect: The completion's ``sent_settings`` is the wrong type; its spend is billed.
        """
        completion: CompletionResult = await self._inner.generate(
            system=system, user=user, response_format=response_format
        )
        try:
            usage = CallUsage.of(completion)
        except CompletionProtocolDefect:
            self.billed.append(CallUsage.billed(completion))
            raise
        self.billed.append(usage)
        return completion

    async def aclose(self) -> None:
        """Release nothing: the judge's client is the judge service's, which releases it."""

    async def __aenter__(self) -> _BilledClaimsClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


async def _regrade_claims(
    inputs: ReproducibleClaimInputs,
    client: _BilledClaimsClient,
    *,
    result_id: str,
    derivations: Mapping[str, ClaimDerivation],
) -> dict[str, GradedCriterion | str]:
    """Grade each criterion a claims re-judge grades again, whole, and finish each lost web step, in the template's
    order, beside the grades the trace keeps.

    :return: criterion -> its grade, or its loss.
    :raises CompletionProtocolDefect: A completion in an extraction, a truth call or a web call breaks the protocol.
    :raises ApparatusError: The rig could not serve a call or a search.
    :raises ClaimReaderDefect: The host's reader of a criterion raised, or answered what its measures cannot hold.
    """
    evidence = inputs.claim_evidence
    criteria = {criterion.name: criterion for criterion in inputs.template.claim_criteria}
    graded = {name: grade for name, grade in inputs.trace.claim_grades.items() if name not in inputs.regrade}
    regraded: dict[str, GradedCriterion | str] = {}
    for name in (name for name in criteria if name in {*inputs.regrade, *inputs.web_steps}):
        outcome: GradedCriterion | str
        if name in inputs.web_steps:
            # A completed grade: its truth and ground steps stand, and only the web step it lost is finished.
            outcome = await finish_web_step(
                criteria[name],
                evidence,
                inputs.trace.claim_grades[name],
                client=client,
                graded={other: grade for other, grade in graded.items() if other != name},
                search=client.search,
                derivation=derivations.get(name),
            )
        else:
            outcome = await grade_or_loss(
                criteria[name],
                evidence,
                key=inputs.test_case.reference,
                client=client,
                graded=graded,
                what=f"Re-judge of result {result_id}: claim criterion {name}",
                derivation=derivations.get(name),
                search=client.search if criteria[name].web is not None else None,
            )
        regraded[name] = outcome
        if not isinstance(outcome, str):
            graded[name] = outcome.grade
            if outcome.ground_failure is not None:
                # Recorded on the grade and its tally; logged so an operator sees which step was lost and why.
                log.warning(
                    "Re-judge of result %s: claim criterion %s lost its ground step (%s): %s",
                    result_id,
                    name,
                    outcome.tally.ground_loss,
                    outcome.ground_failure,
                )
            if outcome.web_failure is not None:
                log.warning(
                    "Re-judge of result %s: claim criterion %s lost its web step (%s); truth stands: %s",
                    result_id,
                    name,
                    outcome.tally.web_loss,
                    outcome.web_failure,
                )
    return regraded


def _claims_fold(client: _BilledClaimsClient) -> JudgeFold:
    """What a claims re-judge's calls cost, folded as the cell's judge rows fold them."""
    return fold_judge_outcomes([], also_billed=client.billed, also_searched=client.searched)


def _rejudge_spend(updated: EvalResult, *, before: EvalResult) -> float | None:
    """What a re-judge's rescores, of its dims and of its claim criteria, cost together; ``None`` when any is
    unpriced."""
    added = [
        *(rescore.cost_usd for rescore in updated.judge_rescores[len(before.judge_rescores) :]),
        *(rescore.cost_usd for rescore in updated.claim_rescores[len(before.claim_rescores) :]),
    ]
    if any(cost is None for cost in added):
        return None
    return math.fsum(cost for cost in added if cost is not None)


def _settle_before_rejudge(
    storage: EvalStorage, result_id: str, scope_id: str, derivations: Mapping[str, ClaimDerivation]
) -> tuple[EvalResult | None, frozenset[str]]:
    """Settle a result's pending claim criteria before anything of a re-judge is checked or asked.

    Asks no call, so it passes none of the re-judge's apparatus checks: the run need only have ended, so the cell's
    writes are done. Its template is read for each criterion's ``grounded_in`` alone, which the stored grade is held to.

    :return: The result as stored after settling (``None`` when nothing was pending, or nothing could be settled), and
        the criteria that were pending.
    :raises NotFoundError: The result, or its run or template, does not load.
    :raises ValidationFailedError: The run is still running, or a stored grade gives no tally.
    :raises ConflictError: Something else wrote the result since it was read; nothing was stored.
    """
    result, etag = storage.load_eval_result_with_etag(result_id, scope_id)
    if result is None:
        raise NotFoundError("result", result_id, remedy=FIND_RESULT)
    pending = frozenset(name for name, loss in result.claim_failures.items() if loss == CLAIM_PENDING)
    if not pending:
        return None, pending
    run = get_run(storage, result.eval_run_id, scope_id)
    if run.status in NON_TERMINAL_RUN_STATUSES:
        raise ValidationFailedError(
            f"run '{run.id}' is {run.status} — its results are still being written",
            remedy=f"wait for it to end (job_poll run:{run.id}), then ask again",
        )
    if run.template_id is None:
        return None, pending
    template = storage.load_template(run.template_id, scope_id)
    if template is None:
        raise NotFoundError("template", run.template_id, remedy=FIND_TEMPLATE)
    try:
        settled = settled_pending_claims(
            result, storage.load_eval_trace(result_id, scope_id), template, derivations=derivations
        )
    except ValueError as unreadable:
        raise ValidationFailedError(
            f"result '{result_id}''s stored claim grade gives no tally: {unreadable}",
            remedy="re-judge it once its template declares the criteria it was graded under",
        ) from unreadable
    if settled is None:
        return None, pending
    try:
        storage.replace_eval_result(settled, if_match=etag)
    except ConflictError as e:
        raise ConflictError(
            f"result '{result_id}' changed while its pending claim criteria were being settled; nothing was stored",
            remedy="re-read it and re-judge again",
        ) from e
    log.info("eval.rejudge_result result=%s settled its claim criteria pending their trace", result_id)
    return settled, pending


class _RejudgeRead(NamedTuple):
    """Everything a re-judge reads before it pays for a call."""

    result: EvalResult
    etag: str | None
    run: EvalRun
    #: The run's reproduced judge inputs, when a judge dimension failed; ``None`` when only claim criteria were lost.
    inputs: ReproducibleJudgeInputs | None
    #: The dims to re-ask.
    failed: list[str]


def _rejudge_inputs(storage: EvalStorage, result_id: str, scope_id: str) -> _RejudgeRead:
    """Everything a re-judge reads before it pays for a call, or the refusal — the store reads of
    :func:`rejudge_result`, its claim criteria's apart (:func:`~threetears.evals.run.rejudge.reproducible_claim_inputs`).

    :return: The result, the etag its rewrite must present, its run, the run's reproduced judge inputs when a dim
        failed, and the dims to re-ask.
    :raises NotFoundError: No result, run, template, judge config or test case with the recorded id.
    :raises ValidationFailedError: Nothing to re-judge, or an apparatus input cannot be reproduced.
    """
    # Every store hands a found document's etag back (the port's contract), so the rewrite is
    # conditional: a re-judge racing another writer of this result is refused rather than clobbering it.
    result, etag = storage.load_eval_result_with_etag(result_id, scope_id)
    if result is None:
        raise NotFoundError("result", result_id, remedy=FIND_RESULT)
    claims_due = any(claims_due_rejudge(result))
    if result.judge_error is None and not claims_due:
        refuse_claims_over_the_trace(result)
        raise ValidationFailedError(
            f"result '{result_id}' has no judge_error and lost no claim criterion or web step — nothing failed to "
            "re-judge",
            remedy="re-judge only a result whose judging failed; to measure its judge again, use a judge repeat",
        )
    run = get_run(storage, result.eval_run_id, scope_id)
    if result.judge_error is None:
        return _RejudgeRead(result, etag, run, None, [])
    failed = failed_judge_dims(result, list(run.effective_judges or {}))
    inputs = reproducible_judge_inputs(
        storage, result, run, scope_id, request_settings="today", config_dims=failed, same_judge_dims=failed
    )
    # Narrowed to the dims this result was judged on: a document's run attributes the two
    # conversation axes, which its cell was never asked and so holds no score for.
    failed = [dim for dim in failed if dim in inputs.dims]
    if not failed and claims_due:
        # Every dim is answered; the lost claim criteria, or lost web steps, are what is left to re-judge.
        return _RejudgeRead(result, etag, run, None, [])
    if not failed:
        raise ValidationFailedError(
            f"result '{result_id}' carries a judge_error but holds a score or a can't-tell for every dim its run "
            "judged, so there is no dimension to re-judge",
            remedy="read its scores as they are; to measure its judge again, use a judge repeat",
        )
    return _RejudgeRead(result, etag, run, inputs, failed)


def _dollars(cost_usd: float | None) -> str:
    """A spend for a log line: dollars, or ``unpriced`` — never a zero standing in for unknown."""
    return f"${round_half_even(cost_usd, 6)}" if cost_usd is not None else "unpriced"


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

    :raises StorageError: The result is stored and re-scored, and the run's count could not be
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
            raise NotFoundError("run", run_id, remedy=FIND_RUN)
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


#: Where an ``error_details`` entry :func:`record_completeness` writes says its error happened.
_COMPLETENESS_NOT_RECORDED = "completeness record not written, so the run's completeness reads unknown"


def _record_unrecorded_completeness(storage: JobStore, run_id: str, scope_id: str, entry: str) -> None:
    """Write why a run's completeness record was not written onto the run; never raise.

    The second write of the run :func:`record_completeness` makes, and as guarded as the first: it runs inside the
    run's work function, where anything raised would reach the job's boundary and stamp ``failed`` on a run that
    executed its matrix. A failure here is logged naming both errors.
    """
    try:
        outcome = record_run_faults(storage, run_id, scope_id, [entry])
    # prawduct:ok-broad-except — any storage fault here would mislabel a run that succeeded; logged with both errors
    except Exception as unwritten:
        contain(
            unwritten,
            log=log,
            what=(
                f"eval.completeness run={run_id} scope={scope_id}: why its completeness was not recorded could not "
                f"be written on the run either; the first error: {entry}"
            ),
            level="error",
        )
        return
    if outcome in ("missing", "refused"):
        log.error(
            "eval.completeness run=%s scope=%s: why its completeness was not recorded could not be written on the "
            "run either (%s); the first error: %s",
            run_id,
            scope_id,
            outcome,
            entry,
        )


def record_completeness(
    storage: JobStore,
    run_id: str,
    scope_id: str,
    cells: Sequence[CellSummary],
    *,
    metered: MeteredCallTally | None = None,
    leases: JobLeases | None = None,
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
    the run it describes. **What did not get written is still written down on
    the run**: a record that raised, or whose write no attempt completed, adds
    an ``error_details`` entry saying so (:func:`record_run_faults`), so the
    run's unknown completeness reads with its cause beside it rather than only
    in the log; that write failing too is logged with both errors.

    :param storage: The run's store.
    :param run_id: The run whose loop just finished.
    :param scope_id: The partition the run lives in.
    :param cells: One summary per cell the loop ran.
    :param metered: The run ledger's final tally of metered third-party calls, or
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
    :param leases: The leases of the process running the run, so the record is the holder's own write
        (:func:`update_eval_run`): declined, and nothing written, once the run's lease has lapsed or passed to another
        holding of the run — another process, or a later epoch, which counts the cells itself. ``None`` for a run held
        under no lease.
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
        outcome = update_eval_run(storage, run_id, scope_id, _stamp, leases=leases)
    # prawduct:ok-broad-except — any storage fault here would mislabel a run that succeeded
    except Exception as broke:
        contain(
            broke,
            log=log,
            what=(
                f"eval.completeness run={run_id} scope={scope_id} NOT recorded — the run keeps its real status "
                "and its completeness reads as unknown"
            ),
            level="error",
        )
        _record_unrecorded_completeness(storage, run_id, scope_id, fault_entry(_COMPLETENESS_NOT_RECORDED, broke))
        return

    if outcome == "missing":
        log.error("eval.completeness run=%s scope=%s vanished before its record could be written", run_id, scope_id)
        return
    if outcome == "declined":
        log.warning(
            "eval.completeness run=%s scope=%s not recorded by its loop: the run was written lost, or passed to "
            "another holding of it, and its completeness is the one recorded there",
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
        _record_unrecorded_completeness(
            storage,
            run_id,
            scope_id,
            f"{_COMPLETENESS_NOT_RECORDED}: no attempt of its conditional write completed — a write another writer "
            "refused each time, or a read that failed each time",
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
    "get_run",
    "record_completeness",
    "rejudge_result",
    "require_cancellable",
    "sweep_abandoned_runs",
]
