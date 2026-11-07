"""The analysis service — generating, reading and curating a campaign's analyses and insights.

Every operation here is one a client of the packages needs to drive the analysis lens: generate
a campaign's analysis (the report writer's job, :func:`~threetears.evals.ops.analysis_generate`, ends it here) and
record how the attempt ended, read analyses and their attempts back,
re-assemble the bundle a generation read, read a stored analysis as its report, and read
the insight ledger. They are module functions whose dependencies are parameters, so a host calls them
with its own store and its own bindings; a host's service delegates to them.

**What arrives as an argument.** An operation that reads the host's vocabulary — assembling a
bundle or generating over one — takes the
:class:`~threetears.evals.kernel.host.EvalHost`: its storage, its profile, the client factory the
writer's clients are built from and the describer that says what a provider failure may keep. An operation
that reads documents alone takes storage, through :class:`AnalysisStore`, which a host's store
satisfies by having the methods; campaigns and runs are loaded through it, a missing one refused
with ``NotFoundError``.

**What stays with the host.** Admitting a generation against a job manager — one generation per
campaign at a time, the background task, the poll — is the run package's job manager, which
this package may not import, so that composition is the host's.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol

from threetears.evals.analysis.bundle.assemble import CampaignReadStore, assemble_context_bundle
from threetears.evals.analysis.bundle.insights import superseding_insights
from threetears.evals.analysis.campaign_bundle import (
    CampaignBundle,
    build_campaign_bundle,
)
from threetears.evals.analysis.campaign_readings import campaign_readings
from threetears.evals.analysis.campaigns import campaign_arms
from threetears.evals.analysis.generation_tally import GenerationTally
from threetears.evals.analysis.bundle.schema import FINGERPRINT_PREFIX, BundleInspection
from threetears.evals.analysis.report import CampaignReport, Verdict, build_campaign_report, build_code_only_report
from threetears.evals.analysis.report.model import ReportBlock, TrialDetail
from threetears.evals.analysis.report.build import analysis_campaign_bundle, analysis_writer
from threetears.evals.analysis.report.words import campaign_arm_tags, declared_arm_names, printed_arm_labels
from threetears.evals.analysis.report.campaign.answer_blocks import NO_LONGER_READS, retired_generation
from threetears.evals.analysis.report_doc_store import StoredReportPage, read_report_doc
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.kernel.campaign import CampaignDeclaration, EvalAnalysisAttempt, EvalCampaign
from threetears.evals.kernel.campaign_writes import serialized_campaign_write
from threetears.evals.kernel.declaration import CampaignCall, OverriddenBar, resolve_control_arm
from threetears.evals.kernel.errors import (
    ConflictError,
    EvalServiceError,
    JobFenced,
    NotFoundError,
    RefusedInput,
    StorageError,
    ValidationFailedError,
)
from threetears.evals.kernel.refusals import job_refusal
from threetears.evals.schema.refusal import JobRefusal
from threetears.evals.kernel.job_leases import ANALYSIS_JOB, JobLeases, write_held_record
from threetears.evals.kernel.storage import JobRecordStore
from threetears.evals.schema.models import utc_now_iso
from threetears.evals.kernel.offload import run_blocking, wait_through_cancellation
from threetears.evals.kernel.phase_log import phase_logged, took_in
from threetears.evals.schema.logs import get_logger
from threetears.evals.kernel.remedies import (
    FIND_ANALYSIS,
    FIND_CAMPAIGN,
    FIND_RUN,
)

if TYPE_CHECKING:
    from threetears.evals.kernel.storage import EvalStorage
    from threetears.evals.kernel.campaign import (
        AspectAnalysis,
        AttemptOutcome,
        EvalAnalysis,
        EvalInsight,
    )
    from threetears.evals.kernel.host.profile import HostProfile
    from threetears.evals.schema.models import EvalRun
    from threetears.evals.kernel.host.eval_host import EvalHost
    from concurrent.futures import Executor


log = get_logger(__name__)

#: How a generation attempt's ending write went: ``recorded``; ``fenced`` — refused, the stored attempt had already
#: ended (written lost by another process); ``unrecorded`` — the write failed, and was logged.
_AttemptRecorded = Literal["recorded", "fenced", "unrecorded"]

#: How many distinct scopes a refusal lists before summarising the rest.
#:
#: An insight's ``scope`` is free prose — a scope might read like "candidate
#: comparison, small case set, default judge" — so the
#: refusal is naming examples an operator can copy a substring out of, not
#: reproducing a vocabulary. Five is enough to show the shape.
_SCOPE_SAMPLE = 5


class AnalysisStore(CampaignReadStore, JobRecordStore, Protocol):
    """The storage calls the analysis service makes — its own, plus the bundle assembly's it hands the store to.

    It extends :class:`~threetears.evals.analysis.bundle.assemble.CampaignReadStore` rather than repeating it,
    because every function here that assembles a bundle passes this same store to
    :func:`~threetears.evals.analysis.bundle.assemble.assemble_context_bundle`, so a store satisfying this port
    has to satisfy that one. ``query_insights`` is declared again, wider: the ledger listing
    filters on the minting campaign as well as the subject, and the bundle's narrower call is one
    this signature accepts. A generation's attempt is written through the job record's read-modify-write
    (:class:`~threetears.evals.kernel.storage.JobRecordStore`), because a background generation's attempt is
    its job's record, carrying a lease another process may have found lapsed.

    Structural, so a host's own storage satisfies it by having the methods —
    :class:`~threetears.evals.kernel.storage.EvalStorage` does. Positional parameters are
    positional-only, which lets the port say ``scope_id`` while an implementation names the thing
    it partitions by.
    """

    def load_campaign(self, campaign_id: str, scope_id: str, /) -> EvalCampaign | None:
        """Load one campaign within a scope, or ``None`` when it does not resolve there."""
        ...

    def load_eval_run(self, run_id: str, scope_id: str, /) -> EvalRun | None:
        """Load one run within a scope, or ``None`` when it does not resolve there."""
        ...

    def save_analysis(self, analysis: EvalAnalysis, /) -> None:
        """Write a generated analysis; raises ``StorageError`` rather than returning a flag."""
        ...

    def load_analysis(self, analysis_id: str, scope_id: str, /) -> EvalAnalysis | None:
        """Load one analysis within a scope, or ``None`` when it does not resolve there."""
        ...

    def delete_analysis(self, analysis_id: str, scope_id: str, /) -> bool:
        """Delete one analysis — a generation withdrawing what it stored for an attempt written lost."""
        ...

    def delete_report_plan(self, plan_id: str, scope_id: str, /) -> bool:
        """Delete one report plan, the retired report planner's: an analysis's delete sweeping what it stored."""
        ...

    def list_analyses_by_campaign(self, campaign_id: str, scope_id: str, /) -> list[EvalAnalysis]:
        """Every analysis of one campaign, newest first."""
        ...

    def load_reported_analysis(self, campaign_id: str, scope_id: str, /) -> EvalAnalysis | None:
        """The campaign's newest analysis that is not archived — the one its page reports — read alone, never by
        loading the campaign's other analyses; ``None`` when it has none."""
        ...

    def list_analysis_attempts_by_campaign(self, campaign_id: str, scope_id: str, /) -> list[EvalAnalysisAttempt]:
        """Every generation attempt recorded against one campaign, newest first."""
        ...

    def save_insight(self, insight: EvalInsight, /) -> None:
        """Write an insight a generation minted; raises ``StorageError`` on failure."""
        ...

    def delete_insight(self, insight_id: str, scope_id: str, /) -> bool:
        """Delete one insight — a generation withdrawing one it minted for an attempt written lost."""
        ...

    def query_insights(
        self, scope_id: str, /, *, subject_id: str | None = None, source_campaign_id: str | None = None
    ) -> list[EvalInsight]:
        """Every insight in a scope, optionally narrowed by subject and by minting campaign, newest observation first."""
        ...

    def analysis_archived(self, analysis_id: str, scope_id: str, /) -> bool | None:
        """Whether one stored analysis is archived, or ``None`` when it no longer resolves — what a generation's
        insight write asks of each ledger insight's minting analysis."""
        ...


def describe_insight_id_filters(subject_id: str | None, source_campaign_id: str | None) -> str:
    """Name the id filters an insight read was narrowed by, for a caller-facing sentence.

    Two surfaces say this, for two different reasons — :func:`list_insights`' refusal says
    what had already narrowed the set before ``scope`` emptied it, and the MCP
    handler's empty read says what emptied it. Both need the identical phrase, and
    both spelled it themselves until a review pointed out that this was the one
    part of the ``insight_list`` contract NOT held together by the shared service
    method: adding a third id filter, or renaming either of these, would have
    named it on one surface and silently dropped it from the other, with no test
    comparing the two strings.

    :param subject_id: The subject filter, if the caller supplied one.
    :ptype subject_id: str | None
    :param source_campaign_id: The campaign filter, if the caller supplied one.
    :ptype source_campaign_id: str | None
    :return: A comma-joined ``name=value`` phrase, or ``""`` when neither was supplied.
    :rtype: str
    """
    return ", ".join(
        part
        for part in (
            f"subject_id={subject_id!r}" if subject_id else "",
            f"source_campaign_id={source_campaign_id!r}" if source_campaign_id else "",
        )
        if part
    )


def _load_campaign(storage: AnalysisStore, campaign_id: str, scope_id: str) -> EvalCampaign:
    """Load a campaign within a scope, refusing an unknown id as the campaign family does.

    :raises NotFoundError: No campaign with that id in the scope.
    """
    campaign = storage.load_campaign(campaign_id, scope_id)
    if campaign is None:
        raise NotFoundError("campaign", campaign_id, remedy=FIND_CAMPAIGN)
    return campaign


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class AnalysisGenerationEstimate(EvalBaseModel):
    """What a generation would be held to before it starts: its bound, and whether it would start.

    ``analysis_estimate`` (:func:`~threetears.evals.ops.analysis_estimate`) answers it for the report writer, whose
    bound is the host's report budget.

    :ivar campaign_id: The campaign.
    :ivar generator_model: The model the generation would call, as the host's client resolved it.
    :ivar first_call_ceiling_usd: Always ``None`` for the report writer, whose every turn is priced against the budget
        when it exists (a turn's price depends on the conversation before it). The field stays for the shape a host
        reads; it leaves in a later minor.
    :ivar max_calls: The most turns the writer can send (``writer_turns`` of the host's limits, or the report budget's
        turn cap where that is fewer: the count ``Ceilings().writer_ceiling_s`` multiplies).
    :ivar cap_usd: The report budget's dollar cap. ``None`` when the report budget caps tokens and turns (``bound``
        says which), never read as $0.
    :ivar would_start: Whether the start's checks pass: the host's checks and whether the writer's turns can be
        bounded. It is no promise that a report is written: what it does not check is in ``unchecked``.
    :ivar refusal: Why it would not, when it would not.
    :ivar bound: The bound in words (the report budget, in dollars or in tokens and turns).
    :ivar unchecked: What the estimate does not check and the job checks before any call: the campaign's evidence,
        read when the job freezes it.
    """

    campaign_id: str
    generator_model: str
    first_call_ceiling_usd: float | None
    max_calls: int
    cap_usd: float | None
    would_start: bool
    refusal: str | None = None
    bound: str | None = None
    unchecked: str | None = None


def require_evidence(bundle: CampaignBundle, campaign_id: str, scope_id: str) -> None:
    """Refuse a campaign bundle that resolves no runs: there is nothing to analyse, report on or freeze.

    The zero-evidence case, distinct from the thin-but-real data a generation or a report degrades gracefully over:
    every attached run failed to resolve (destroyed outside the delete cascade), was archived, or none is attached
    yet. Refused rather than spent on, or frozen as a snapshot no figure could come from. The two empty causes are
    named separately: "archived them all" is a curation the operator can undo in one call, while "unresolved" is
    data that is gone, and one message for both would send them looking in the wrong place.

    :param bundle: The campaign's assembled bundle.
    :ptype bundle: CampaignBundle
    :param campaign_id: The campaign, as the refusal names it.
    :ptype campaign_id: str
    :param scope_id: The scope it was assembled in.
    :ptype scope_id: str
    :raises ValidationFailedError: The bundle holds no run: every one is archived, or none resolves.
    """
    if bundle.run_ids:
        return
    archived_run_ids = [run for aspect in bundle.aspects for run in aspect.archived_run_ids]
    unresolved_run_ids = [run for aspect in bundle.aspects for run in aspect.unresolved_run_ids]
    if archived_run_ids and not unresolved_run_ids:
        raise ValidationFailedError(
            f"campaign '{campaign_id}' has {len(archived_run_ids)} run(s) in scope "
            f"'{scope_id}' and every one is archived — nothing to analyse",
            remedy="un-archive a run (run_archive with archived=false) to include it",
        )
    raise ValidationFailedError(
        f"campaign '{campaign_id}' has no runs resolvable in scope '{scope_id}' — nothing to "
        f"analyse (unresolved run ids: {unresolved_run_ids or 'none attached'}"
        + (f"; archived and excluded: {len(archived_run_ids)}" if archived_run_ids else "")
        + ")",
        remedy=f"create a campaign over runs this scope holds ({FIND_RUN})",
    )


#: What a running attempt's ``generator_model`` reads while the host's default writer is not yet resolved: the attempt
#: is recorded before its job builds the client that names it (:func:`running_analysis_attempt`).
DEFAULT_WRITER_UNRESOLVED = "(the host's default writer, not yet resolved)"


def running_analysis_attempt(
    *,
    attempt_id: str,
    campaign_id: str,
    scope_id: str,
    model: str | None,
    prompt_id: str,
    started_at: str | None = None,
) -> EvalAnalysisAttempt:
    """The attempt a background generation writes as it starts: ``running``, nothing prepared or sent yet.

    Written before the generation's job starts — carrying the job's lease, which the caller stamps — so the job is
    answered from the store in every process from its first poll. It is written before the bundle is assembled or
    the client built, since assembling reads every member run's results and the start returns before that, so what
    only they can say is not on it yet: ``generator_model`` is the writer asked for, or
    :data:`DEFAULT_WRITER_UNRESOLVED` when none was named, and ``bundle_fingerprint`` is empty. The job writes the
    resolved writer over it once its client is built, and the writer and the bundle's fingerprint again as it ends
    (:func:`record_written_generation`); an attempt refused while it prepared keeps what was known.

    :param attempt_id: The attempt's id: the generation's, and its job's.
    :ptype attempt_id: str
    :param campaign_id: The campaign to analyse.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param model: The writer asked for, or ``None`` for the host's default.
    :ptype model: str | None
    :param prompt_id: The registry key its prompt is resolved under.
    :ptype prompt_id: str
    :param started_at: When it was started; now when omitted.
    :ptype started_at: str | None
    :return: The attempt, ``running``.
    :rtype: EvalAnalysisAttempt
    """
    return EvalAnalysisAttempt(
        id=attempt_id,
        scope_id=scope_id,
        campaign_id=campaign_id,
        outcome="running",
        analysis_id=None,
        error=None,
        generator_model=model or DEFAULT_WRITER_UNRESOLVED,
        prompt_id=prompt_id,
        bundle_fingerprint="",
        started_at=started_at or utc_now_iso(),
    )


def _still_held(storage: AnalysisStore, attempt_id: str, scope_id: str, leases: JobLeases | None) -> None:
    """Refuse to go on with a generation whose attempt record has ended under it, or whose lease this process let go.

    A generation run inline has no stored attempt until it ends, and is never refused here.

    :raises JobFenced: The stored attempt has ended, or ``leases`` let go of it.
    """
    if leases is not None and leases.fenced(attempt_id):
        raise JobFenced(
            f"analysis attempt '{attempt_id}': this process let go of its lease; the generation stores nothing more"
        )
    stored, _ = storage.load_job_record(EvalAnalysisAttempt, attempt_id, scope_id)
    if stored is not None and ANALYSIS_JOB.ended(stored):
        raise JobFenced(
            f"analysis attempt '{attempt_id}' reads {stored.outcome}"
            f"{' (lost)' if stored.lost is not None else ''}; the generation stores nothing more"
        )


async def _offloaded[**P](
    executor: Executor | None, fn: Callable[P, object], /, *args: P.args, **kwargs: P.kwargs
) -> bool:
    """Run a store write on ``executor``, waiting for it through a cancellation of the waiter.

    :param executor: The host's blocking executor.
    :ptype executor: Executor | None
    :param fn: The blocking write.
    :ptype fn: Callable[P, object]
    :param *args: Its positional arguments.
    :ptype args: P.args
    :param **kwargs: Its keyword arguments.
    :ptype kwargs: P.kwargs
    :return: Whether a cancellation arrived while it ran — the caller delivers it once it has finished.
    :rtype: bool
    :raises Exception: Whatever the write raised.
    """
    return (await _offloaded_result(executor, fn, *args, **kwargs))[1]


async def _offloaded_result[**P, T](
    executor: Executor | None, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
) -> tuple[T, bool]:
    """:func:`_offloaded`, keeping what the write returned.

    :return: What ``fn`` returned, and whether a cancellation arrived while it ran.
    :rtype: tuple[T, bool]
    :raises Exception: Whatever the write raised.
    """
    work = asyncio.ensure_future(run_blocking(executor, fn, *args, **kwargs))
    cancelled = await wait_through_cancellation(work)
    return work.result(), cancelled


class WrittenAttempt(NamedTuple):
    """Who a report writer's generation is, as its attempt's ending records it: everything but how it ended.

    :ivar attempt_id: The attempt, its job's record.
    :ivar campaign_id: The campaign written about.
    :ivar scope_id: The scope it lives in.
    :ivar generator_model: The model the ``analyst`` client resolved.
    :ivar prompt_id: The analyst skill's digest, the generation's provenance.
    :ivar started_at: When the attempt was started.
    :ivar leases: The leases a background generation's attempt is held under.
    """

    attempt_id: str
    campaign_id: str
    scope_id: str
    generator_model: str
    prompt_id: str
    started_at: str
    leases: JobLeases | None


async def record_written_generation(
    host: EvalHost,
    attempt: WrittenAttempt,
    *,
    outcome: AttemptOutcome,
    tally: GenerationTally,
    bundle_fingerprint: str,
    error: str | None = None,
    refused_by: EvalServiceError | None = None,
    stored: tuple[EvalAnalysis, Sequence[EvalInsight]] | None = None,
) -> list[EvalInsight]:
    """End a report writer's generation (``analysis_generate``'s job): store the analysis it minted, then its ending.

    The commit for an analysis the job built from a stored report document. With ``stored``, the attempt must still be held: the analysis is written, then each insight (a
    restated claim written over the live insight that states it), then the attempt's ending ``stored``, last and
    conditionally, so a poll reading ``completed`` finds the ledger written; when that write finds the attempt already
    lost, what was stored is withdrawn. Without ``stored``, the attempt's ending is written alone, its tally kept.
    Every write runs on the host's blocking executor and is waited for through a cancellation, which is delivered once
    the ending is written.

    :param host: The host whose store holds the attempt, the analysis and the insights.
    :ptype host: EvalHost
    :param attempt: Who the generation is.
    :ptype attempt: WrittenAttempt
    :param outcome: How it ended: ``stored`` with ``stored``, any other ending without.
    :ptype outcome: AttemptOutcome
    :param tally: What it sent and spent, as the attempt keeps it (its turns as calls, its unpriced turns as
        unpriced calls).
    :ptype tally: GenerationTally
    :param bundle_fingerprint: The fingerprint of the bundle it was written over; empty when nothing was frozen.
    :ptype bundle_fingerprint: str
    :param error: Why it ended, for any ending but ``stored``.
    :ptype error: str | None
    :param refused_by: The engine refusal it ended on, kept on the attempt as data.
    :ptype refused_by: EvalServiceError | None
    :param stored: The analysis and the insights it minted, for a ``stored`` ending.
    :ptype stored: tuple[EvalAnalysis, Sequence[EvalInsight]] | None
    :return: The insights written (after the ledger's restatements); empty without ``stored``.
    :rtype: list[EvalInsight]
    :raises JobFenced: The attempt was written lost, or this process let go of its lease, before the analysis was
        stored, or before its ending was written (what it stored is then withdrawn).
    :raises StorageError: The analysis or an insight could not be stored; the attempt is recorded ``failed`` for the
        first, and ``stored`` with its ledger incomplete for the second.
    """
    storage, executor = host.storage, host.blocking_executor

    async def recorded(
        ending: AttemptOutcome,
        *,
        why: str | None = error,
        refusal: EvalServiceError | None = refused_by,
        analysis_id: str | None = None,
    ) -> tuple[_AttemptRecorded, bool]:
        return await _offloaded_result(
            executor,
            _record_analysis_attempt,
            storage,
            campaign_id=attempt.campaign_id,
            scope_id=attempt.scope_id,
            outcome=ending,
            tally=tally,
            generator_model=attempt.generator_model,
            prompt_id=attempt.prompt_id,
            bundle_fingerprint=bundle_fingerprint,
            started_at=attempt.started_at,
            error=why,
            analysis_id=analysis_id,
            attempt_id=attempt.attempt_id,
            leases=attempt.leases,
            refusal=None if refusal is None else job_refusal(refusal),
        )

    if stored is None:
        _, cancelled = await recorded(outcome)
        if cancelled:
            raise asyncio.CancelledError
        return []
    analysis, minted = stored
    await _offloaded(executor, _still_held, storage, attempt.attempt_id, attempt.scope_id, attempt.leases)
    try:
        cancelled = await _offloaded(executor, storage.save_analysis, analysis)
    except StorageError as e:
        unstored = StorageError(
            f"failed to persist the analysis '{analysis.id}' linking report document '{analysis.report_doc_id}' for "
            f"campaign '{attempt.campaign_id}' (the report is stored, and its turns were already paid for)"
        )
        await recorded("failed", why=f"the analysis could not be stored: {e}", refusal=unstored)
        raise unstored from e
    insights, cancelled = await _commit_insights_then_ending(
        storage,
        executor,
        analysis=analysis,
        minted=minted,
        attempt_id=attempt.attempt_id,
        scope_id=attempt.scope_id,
        record_stored=lambda: recorded("stored", why=None, refusal=None, analysis_id=analysis.id),
        left_behind=f"analysis '{analysis.id}'",
        cancelled=cancelled,
    )
    if cancelled:
        raise asyncio.CancelledError
    return insights


async def _commit_insights_then_ending(
    storage: AnalysisStore,
    executor: Executor | None,
    *,
    analysis: EvalAnalysis,
    minted: Sequence[EvalInsight],
    attempt_id: str,
    scope_id: str,
    record_stored: Callable[[], Awaitable[tuple[_AttemptRecorded, bool]]],
    left_behind: str,
    cancelled: bool,
) -> tuple[list[EvalInsight], bool]:
    """Write a stored analysis's insights, then its attempt's ending ``stored``, withdrawing both when it is fenced.

    The commit a generation makes once its analysis is stored (:func:`record_written_generation`). A restated claim
    replaces the live insight that states it rather than adding a row beside it, so regenerating over the same
    evidence leaves the ledger its size; the ledger is read now, not at assembly, since the ledger the bundle read is
    capped and another generation may have written since. The attempt is recorded ``stored`` only once the insights
    are written (or have failed to be), so a poll reading ``completed`` finds the ledger written. **The attempt's
    ending is the commit**: when its conditional write finds the attempt already ended (written lost by another
    process), what was stored is withdrawn (:func:`_withdraw_generation`), so a lost attempt never keeps an analysis.

    :param storage: Where the insights and the attempt are written.
    :ptype storage: AnalysisStore
    :param executor: The host's blocking executor, which every write runs on.
    :ptype executor: Executor | None
    :param analysis: The analysis, already stored.
    :ptype analysis: EvalAnalysis
    :param minted: The insights it minted, before the ledger's restatements.
    :ptype minted: Sequence[EvalInsight]
    :param attempt_id: Its attempt.
    :ptype attempt_id: str
    :param scope_id: The scope they live in.
    :ptype scope_id: str
    :param record_stored: Writes the attempt's ending ``stored``: whether it landed, and whether a cancellation arrived.
    :ptype record_stored: Callable[[], Awaitable[tuple[_AttemptRecorded, bool]]]
    :param left_behind: What a withdrawal that fails part-way may leave stored, as its fence names it.
    :ptype left_behind: str
    :param cancelled: Whether a cancellation arrived while the analysis was stored, delivered by the caller.
    :ptype cancelled: bool
    :return: The insights written (the ledger's restatements applied), and whether a cancellation arrived.
    :rtype: tuple[list[EvalInsight], bool]
    :raises JobFenced: The ending found the attempt written lost; what was stored is withdrawn, or the fence says what
        is left to remove by hand.
    :raises StorageError: An insight, or the ledger's read, failed; the attempt is recorded ``stored`` all the same,
        since its analysis is.
    """
    written: list[list[EvalInsight]] = []
    # The ledger rows the new insights are written over, as they stood — what a withdrawal puts back.
    overwritten: dict[str, EvalInsight] = {}
    stored_insights: list[EvalInsight] = []

    def supersede() -> None:
        ledger = storage.query_insights(scope_id, subject_id=analysis.subject_id)
        superseding = superseding_insights(
            list(minted), ledger, lambda source: storage.analysis_archived(source, scope_id)
        )
        prior = {insight.id: insight for insight in ledger}
        overwritten.update({insight.id: prior[insight.id] for insight in superseding if insight.id in prior})
        written.append(superseding)

    unfinished: StorageError | None = None
    try:
        try:
            cancelled = await _offloaded(executor, supersede) or cancelled
        except StorageError as e:
            raise StorageError(
                f"failed to read the insight ledger before writing analysis '{analysis.id}''s insights — the "
                "analysis itself was stored, but none of its insights were"
            ) from e
        for insight in written[0]:
            try:
                cancelled = await _offloaded(executor, storage.save_insight, insight) or cancelled
            except StorageError as e:
                raise StorageError(
                    f"failed to persist insight '{insight.id}' from analysis '{analysis.id}' — the analysis "
                    "itself was stored, but its insight ledger is incomplete"
                ) from e
            stored_insights.append(insight)
    except StorageError as e:
        unfinished = e
    ending, cancelled_now = await record_stored()
    cancelled = cancelled_now or cancelled
    if ending == "fenced":
        lost = f"analysis attempt '{attempt_id}' was written lost by another process before its ending was recorded"
        try:
            await _offloaded(
                executor, _withdraw_generation, storage, analysis, stored_insights, overwritten, scope_id=scope_id
            )
        # prawduct:ok-broad-except — a store's write raises its own types; re-raised as the fence, naming it
        except Exception as unwithdrawn:
            log.exception(
                "eval.generate_analysis analysis %s was stored for an attempt written lost, and withdrawing it failed "
                "part-way — delete it and restore its insights by hand: insights=%s",
                analysis.id,
                [insight.id for insight in stored_insights],
            )
            raise JobFenced(
                f"{lost}, and withdrawing what it stored failed part-way ({type(unwithdrawn).__name__}: "
                f"{unwithdrawn}): {left_behind} may still be stored — delete whatever of it is, and put back or delete insights "
                f"{[insight.id for insight in stored_insights]} by hand"
            ) from unwithdrawn
        raise JobFenced(f"{lost}; analysis '{analysis.id}' and its insights were withdrawn")
    if unfinished is not None:
        raise unfinished
    return written[0] if written else [], cancelled


def _record_analysis_attempt(
    storage: AnalysisStore,
    *,
    campaign_id: str,
    scope_id: str,
    outcome: AttemptOutcome,
    tally: GenerationTally,
    generator_model: str,
    prompt_id: str,
    bundle_fingerprint: str,
    started_at: str,
    error: str | None,
    analysis_id: str | None,
    attempt_id: str,
    leases: JobLeases | None,
    refusal: JobRefusal | None = None,
) -> _AttemptRecorded:
    """Store how one generation attempt ended, from the tally the writer kept.

    A write that fails is logged at ERROR with the whole record and NOT raised: this runs on the
    way out of every ending, including a failure the caller must still be shown, and a second
    exception here would replace the one that says what went wrong. Written over the attempt's stored
    ``running`` record when there is one, keeping its lease fields, and never over one that has ended: an
    attempt another process wrote lost keeps that ending, and this one is logged and dropped.

    :param storage: Where the attempt record is written.
    :ptype storage: AnalysisStore
    :param campaign_id: The campaign the generation was over.
    :ptype campaign_id: str
    :param scope_id: The scope the campaign lives in, where the record is written.
    :ptype scope_id: str
    :param outcome: How it ended.
    :ptype outcome: AttemptOutcome
    :param tally: What the writer sent, spent and was refused.
    :ptype tally: GenerationTally
    :param generator_model: The model the writer's client was built for.
    :ptype generator_model: str
    :param prompt_id: The registry key the prompt was resolved under.
    :ptype prompt_id: str
    :param bundle_fingerprint: The fingerprint of the bundle it ran over.
    :ptype bundle_fingerprint: str
    :param started_at: When the generation was started.
    :ptype started_at: str
    :param error: What the caller was told ended it; None when it was stored.
    :ptype error: str | None
    :param analysis_id: The analysis it stored; None unless it was.
    :ptype analysis_id: str | None
    :param attempt_id: The attempt's id, minted when the generation was prepared — a background
        generation's job id, so its poller can find this record once the job is gone.
    :ptype attempt_id: str
    :param leases: The leases the attempt is held under; once they let go of it, this ending is refused.
    :ptype leases: JobLeases | None
    :param refusal: The engine refusal the attempt ended on, as data; None for any other ending.
    :ptype refusal: JobRefusal | None
    :return: ``recorded``; ``fenced`` when the stored attempt had already ended — written lost — so this ending was
        refused; ``unrecorded`` when the write failed (logged).
    :rtype: _AttemptRecorded
    """
    attempt = EvalAnalysisAttempt(
        id=attempt_id,
        scope_id=scope_id,
        campaign_id=campaign_id,
        outcome=outcome,
        analysis_id=analysis_id,
        generator_model=generator_model,
        reported_models=tally.reported_models,
        prompt_id=prompt_id,
        prompt_version=tally.prompt_version,
        bundle_fingerprint=bundle_fingerprint,
        calls=tally.calls,
        unpriced_calls=tally.unpriced_calls,
        token_cost=tally.token_cost,
        refusals=tally.refusals,
        error=error,
        started_at=started_at,
        refusal=refusal,
    )
    try:
        write_held_record(storage, ANALYSIS_JOB, attempt, leases=leases)
    except JobFenced as fenced:
        log.warning(
            "eval.generate_analysis attempt %s campaign=%s outcome=%s NOT recorded: %s",
            attempt_id,
            campaign_id,
            outcome,
            fenced,
        )
        return "fenced"
    except StorageError, ConflictError:
        # Log only: the failed write IS the attempt record's own write, already retried by write_held_record, so no
        # record is left to carry the error — and writing the attempt lost instead would contradict an analysis it
        # may have stored. The whole record is in this line, so what the generation spent can be recovered from it.
        log.exception(
            "eval.generate_analysis attempt record NOT stored campaign=%s outcome=%s calls=%d cost=$%.4f model=%s — %s",
            campaign_id,
            outcome,
            attempt.calls,
            attempt.token_cost,
            generator_model,
            attempt.model_dump_json(),
        )
        return "unrecorded"
    return "recorded"


def _withdraw_generation(
    storage: AnalysisStore,
    analysis: EvalAnalysis,
    insights: list[EvalInsight],
    overwritten: dict[str, EvalInsight],
    *,
    scope_id: str,
) -> None:
    """Undo what a generation stored when its attempt turned out lost before its ending was recorded.

    Each insight it wrote is put back as the ledger held it (a restatement written over a live insight) or
    deleted (a new one); then the analysis is deleted, and its plan, so the lost attempt is left with nothing it did
    not record.
    Blocking. A withdrawal that fails part-way raises what the store raised; the caller raises the fence that made
    it necessary, saying the withdrawal failed and what is left to remove by hand, rather than that it happened.

    :param storage: Where the analysis and the insights were written.
    :ptype storage: AnalysisStore
    :param analysis: The analysis stored.
    :ptype analysis: EvalAnalysis
    :param insights: The insights written, in order.
    :ptype insights: list[EvalInsight]
    :param overwritten: The ledger rows those insights were written over, by id, as they stood.
    :ptype overwritten: dict[str, EvalInsight]
    :param scope_id: The scope they live in.
    :ptype scope_id: str
    :raises Exception: Whatever the store raised, part-way.
    """
    for insight in insights:
        if (prior := overwritten.get(insight.id)) is not None:
            storage.save_insight(prior)
        else:
            storage.delete_insight(insight.id, scope_id)
    storage.delete_analysis(analysis.id, scope_id)
    if analysis.report_plan_id is not None:
        storage.delete_report_plan(analysis.report_plan_id, scope_id)
    log.warning(
        "eval.generate_analysis analysis %s withdrawn: its attempt was written lost before its ending was recorded "
        "(%d insight(s) put back or deleted)",
        analysis.id,
        len(insights),
    )


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def get_analysis(storage: AnalysisStore, analysis_id: str, scope_id: str) -> EvalAnalysis:
    """Load a stored analysis by id.

    :param storage: Where analyses are read.
    :ptype storage: AnalysisStore
    :param analysis_id: The analysis to load.
    :ptype analysis_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :return: The analysis.
    :rtype: EvalAnalysis
    :raises NotFoundError: No analysis with that id in the scope.
    """
    analysis = storage.load_analysis(analysis_id, scope_id)
    if analysis is None:
        raise NotFoundError("analysis", analysis_id, remedy=FIND_ANALYSIS)
    return analysis


# ---------------------------------------------------------------------------
# Bundle inspection — the free read of the input a paid generation reads.
#
# The bundle is where the deterministic judgements live: which subtraction is
# unsound, which swing the containment default withholds, which divergences the
# cap dropped, what the campaign's design actually derived to. All of it used to
# reach a reader only if the writer chose to write it into prose, which made a
# rule the writer could ignore out of a rule that was moved out of the prompt
# precisely so it could not be. Both entry points below RE-ASSEMBLE rather than
# read a stored copy: the bundle's fingerprint is the invariant the prompt-tuning
# A/B loop rests on, and a second persisted copy is a second thing to keep true.
# ---------------------------------------------------------------------------


def _assembled_campaign_bundle(
    campaign: EvalCampaign, *, storage: EvalStorage, profile: HostProfile, insights_as_of: str | None = None
) -> CampaignBundle:
    """Assemble every aspect's bundle, the campaign's arms and code's readings into the campaign bundle. Blocking.

    Logs one ``eval.read_tier lens=assemble_context_bundle`` line per assembly
    (:func:`~threetears.evals.kernel.phase_log.phase_logged`). The results' read as readings
    (:meth:`~threetears.evals.kernel.storage.EvalStorage.iter_result_readings`) charges its store calls to
    ``store_ms`` and its rebuilding of results to ``hydrate_ms``, and fills ``path`` and ``store_calls``; ``fold_ms``
    is the rest, the assembly's other store reads (runs, the kept trials, insights) included, and ``render_ms`` is
    empty, since nothing renders here. ``runs_in`` is the runs the aspects' bundles summarise and ``results_in`` the
    results those runs produced.

    :param campaign: The campaign.
    :ptype campaign: EvalCampaign
    :param storage: Where its runs, results and insights are read, and its arms addressed.
    :ptype storage: EvalStorage
    :param profile: The host's vocabulary.
    :ptype profile: HostProfile
    :param insights_as_of: The insight ledger's cutoff, or None for now.
    :ptype insights_as_of: str | None
    :return: The campaign bundle.
    :rtype: CampaignBundle
    """
    with phase_logged("assemble_context_bundle", scope_id=campaign.scope_id, campaign_id=campaign.id):
        bundles = [
            assemble_context_bundle(
                campaign, aspect=aspect.label, storage=storage, profile=profile, insights_as_of=insights_as_of
            )
            for aspect in campaign.aspects
        ]
        arms = campaign_arms(storage, campaign, profile=profile)
        readings = campaign_readings(campaign, {bundle.aspect: bundle for bundle in bundles}, arms=arms)
        # A run belongs to one aspect at most (the campaign refuses one held twice), so the aspects' counts add.
        summaries = [summary for bundle in bundles for summary in bundle.run_summaries]
        took_in(runs=len(summaries), results=sum(summary.n_results for summary in summaries))
        return build_campaign_bundle(campaign, bundles, arms=arms, readings=readings)


def assemble_campaign_bundle(host: EvalHost, campaign_id: str, scope_id: str) -> CampaignBundle:
    """Assemble the campaign bundle a generation launched right now would read, without generating.

    Every aspect's evidence, the campaign's arms and floors, and code's readings across them — what the report
    writer's snapshot freezes. Costs nothing: storage reads and no model call. Not refused when it resolves no runs: an empty bundle is the
    answer, as :func:`inspect_campaign_bundle`'s is.

    :param host: The host: where the campaign's runs, results and insights are read, and its vocabulary.
    :ptype host: EvalHost
    :param campaign_id: The campaign.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :return: The campaign bundle.
    :rtype: CampaignBundle
    :raises NotFoundError: No campaign with that id in the scope.
    """
    campaign = _load_campaign(host.storage, campaign_id, scope_id)
    return _assembled_campaign_bundle(campaign, storage=host.storage, profile=host.profile)


def inspect_campaign_bundle(
    host: EvalHost,
    campaign_id: str,
    scope_id: str,
    *,
    aspect: str | None = None,
) -> BundleInspection:
    """Assemble one aspect's context bundle of a campaign and return it, without generating.

    The dry run: what a generation launched right now would read. This is the
    surface an operator checks a design against — ``set_campaign_control``
    designates a control, but whether the derivation produced a
    one-factor-at-a-time shape, and which lever each cell moved, was previously
    observable only by paying for a generation.

    Costs nothing: the assembler performs storage reads and no model call.

    :param host: The host: where the campaign's runs, results and insights are read, and the
        vocabulary the bundle is assembled in.
    :ptype host: EvalHost
    :param campaign_id: The campaign to assemble for.
    :ptype campaign_id: str
    :param scope_id: The scope the campaign and its member runs live in.
    :ptype scope_id: str
    :param aspect: The aspect whose bundle to assemble; None for a campaign of one aspect.
    :ptype aspect: str | None
    :return: The assembled bundle with its freshly computed fingerprint. No
        generation is being spoken about, so nothing is compared against.
    :rtype: BundleInspection
    :raises NotFoundError: No campaign with that id in the scope.
    :raises RefusedInput: ``aspect`` is None and the campaign has several aspects, or it names none of them.
    """
    campaign = _load_campaign(host.storage, campaign_id, scope_id)
    # Deliberately NOT refused when the bundle resolves no runs, unlike
    # a generation. There the refusal protects a paid call from an empty
    # bundle; here the empty bundle IS the answer — it names, in
    # unresolved_run_ids and archived_run_ids, which of the two causes emptied
    # it, and that is the diagnosis an operator whose generation just refused
    # came here to read.
    bundle = assemble_context_bundle(campaign, aspect=aspect, storage=host.storage, profile=host.profile)
    return BundleInspection.over(bundle)


def inspect_analysis_bundle(
    host: EvalHost,
    analysis_id: str,
    scope_id: str,
    *,
    aspect: str | None = None,
) -> BundleInspection:
    """Re-assemble one aspect's context bundle a stored analysis was generated over.

    The post-hoc read: what that generation saw, so a defect in its output can
    be diagnosed against its input rather than inferred from its prose. Because
    the bundle is re-assembled rather than stored, the answer states whether it
    still reproduces the generation — the recorded
    ``generation.bundle_fingerprint`` against the one computed now. A mismatch
    is a fact, not a failure of the read, and is reported as such rather than
    being silently presented as the original. Prior insights are read as of the
    instant the generation's bundle was assembled (``generation.bundle_assembled_at``),
    so an insight minted afterwards — including the ones this analysis minted itself,
    and one another analysis minted while this generation's provider call ran — does
    not move it. **Three causes reach that mismatch, and the inspection's
    ``mismatch_cause`` names which:** the package's bundle SHAPE moved (the recorded
    ``bundle_schema_version`` differs), the host's declarations moved (the recorded
    ``host_declarations_digest`` differs), or neither did and the evidence moved (a member
    archived, results deleted, an insight the generation read since deleted or superseded, or
    the analysis that minted one archived — which retracts it from every re-assembly). The scope cannot be a fourth
    cause: an analysis lives in its campaign's scope, which is the only scope its member runs
    can live in, so the re-assembly reads exactly the partition the generation read.

    Costs nothing: storage reads and no model call.

    :param host: The host: where the analysis, its campaign's runs, results and insights are read,
        and the vocabulary the bundle is re-assembled in.
    :ptype host: EvalHost
    :param analysis_id: The stored analysis to re-assemble the input of.
    :ptype analysis_id: str
    :param scope_id: The scope the analysis, its campaign and the campaign's runs live in.
    :ptype scope_id: str
    :param aspect: The aspect whose bundle to re-assemble; None for an analysis of one aspect. The recorded
        fingerprint compared is that aspect's (``AspectAnalysis.bundle_fingerprint``).
    :ptype aspect: str | None
    :return: The re-assembled bundle, the recorded fingerprint, and whether the two
        agree.
    :rtype: BundleInspection
    :raises NotFoundError: No analysis with that id, or the campaign it names no
        longer exists in the scope.
    :raises RefusedInput: ``aspect`` is None and the analysis covers several aspects, or it names none of them.
    """
    storage = host.storage
    analysis = get_analysis(storage, analysis_id, scope_id)  # NotFoundError if absent
    campaign = _load_campaign(storage, analysis.campaign_id, scope_id)  # NotFoundError if the campaign was deleted
    # The ledger as it stood when the generation read it. Generation fingerprints its bundle
    # BEFORE saving the insights it mints, so reading today's ledger put this analysis's own
    # output into its re-assembled input and no analysis that minted an insight could ever
    # reproduce. An insight minted later — by this analysis or any other — is equally not
    # something that generation read, and that includes one minted during the provider call:
    # the cutoff is the instant stamped before assembly, never `generated_at`, which is
    # stamped after the call returns.
    held = _analysed_aspect(analysis, aspect)
    bundle = assemble_context_bundle(
        campaign,
        aspect=held.label,
        storage=storage,
        profile=host.profile,
        insights_as_of=analysis.generation.bundle_assembled_at,
    )
    return BundleInspection.over(
        bundle,
        analysis_id=analysis.id,
        recorded_fingerprint=held.bundle_fingerprint,
        recorded_schema_version=analysis.generation.bundle_schema_version,
        recorded_host_declarations_digest=analysis.generation.host_declarations_digest,
    )


def _analysed_aspect(analysis: EvalAnalysis, label: str | None) -> AspectAnalysis:
    """The aspect of ``analysis`` named ``label``, or its only one when ``label`` is None.

    :raises RefusedInput: ``label`` is None and the analysis covers several aspects, or it names none of them.
    """
    labels = [aspect.label for aspect in analysis.aspects]
    if label is None:
        if len(analysis.aspects) == 1:
            return analysis.aspects[0]
        raise RefusedInput(
            f"analysis {analysis.id} covers {len(labels)} aspects, so which aspect's bundle to read is not implied",
            remedy=f"name the aspect: one of {labels}",
        )
    try:
        return analysis.aspect(label)
    except KeyError:
        raise RefusedInput(
            f"analysis {analysis.id} covers no aspect {label!r}", remedy=f"name one of its aspects: {labels}"
        ) from None


# ---------------------------------------------------------------------------
# The report — derived on every read, so a stored analysis stays pivotable
# ---------------------------------------------------------------------------


def _campaign_page(
    campaign: EvalCampaign,
    bundle: CampaignBundle,
    *,
    assembled_at: str,
    analysis: EvalAnalysis | None = None,
    fingerprints: Mapping[str, str] | None = None,
    trials: TrialDetail = "table",
    unread_analysis_id: str | None = None,
    unread_report_doc_id: str | None = None,
    unread_line: str = NO_LONGER_READS,
    folds_alone: bool = False,
    lead: Sequence[ReportBlock] | None = None,
) -> CampaignReport:
    """Lay a campaign bundle out as code's page for the campaign: code's readings, where its report is, the call.

    Each aspect's fold is that aspect's code-only page over its bundle, its arms named by the campaign's labels, its
    methods naming the analysis's writer where one is given.

    :param campaign: The campaign, declaring what the bundle was assembled against.
    :ptype campaign: EvalCampaign
    :param bundle: Its evidence: every aspect's bundle and code's readings across them.
    :ptype bundle: CampaignBundle
    :param assembled_at: When the bundle was assembled (ISO-8601), which dates each aspect's page.
    :ptype assembled_at: str
    :param analysis: The stored analysis the page is laid out for, whose writer the methods name and whose report
        document the first screen points at; None for a page no analysis is named on.
    :ptype analysis: EvalAnalysis | None
    :param fingerprints: Each aspect's bundle fingerprint in the current version, by label, where it was just checked
        against the bundle; an aspect not named is hashed.
    :ptype fingerprints: Mapping[str, str] | None
    :param trials: How much of each failed or lost trial each fold lays out (:data:`TrialDetail`).
    :ptype trials: TrialDetail
    :param unread_analysis_id: On a code-only page, the analysis this version no longer reads that it stands in for.
    :ptype unread_analysis_id: str | None
    :param unread_report_doc_id: On a code-only page, the report document read by its id that it stands in for.
    :ptype unread_report_doc_id: str | None
    :param unread_line: What ``unread_analysis_id`` was, as the page's first screen says it: ``NO_LONGER_READS``, or
        the retired writer's line (:func:`~threetears.evals.analysis.report.campaign.answer_blocks.retired_generation`).
    :ptype unread_line: str
    :param folds_alone: Whether the folds will be laid out without the page's own blocks, as a report document's
        evidence record is, so each keeps what only those blocks would state (its judge's calibration line).
    :ptype folds_alone: bool
    :param lead: The page's answer where its caller states its own, in place of code's lead
        (:func:`~threetears.evals.analysis.report.campaign.page.build_campaign_report`, ``lead``); None for code's.
    :ptype lead: Sequence[ReportBlock] | None
    :return: The page.
    :rtype: CampaignReport
    """
    # Every arm by the name every page prints it by (a computed lever name capitalised, a given label as written):
    # the folds and the campaign's own readings alike, the readings the bundle froze left as read.
    printed = printed_arm_labels(bundle)
    readings = bundle.readings.relabelled(printed)
    labels = readings.arm_labels()
    tags = campaign_arm_tags(bundle.aspects, readings.control)
    declared = declared_arm_names(bundle.arms)
    writer = analysis_writer(analysis) if analysis is not None else None
    checked = fingerprints or {}
    reports = {
        aspect.aspect: build_code_only_report(
            aspect,
            assembled_at=assembled_at,
            campaign_name=campaign.name,
            arm_labels=labels,
            arm_tags=tags,
            arm_short_names=declared,
            writer=writer,
            bundle_fingerprint=checked.get(aspect.aspect),
            trials=trials,
        )
        for aspect in bundle.aspects
    }
    page = build_campaign_report(
        campaign,
        readings,
        reports,
        call=campaign.call,
        report_doc_id=analysis.report_doc_id if analysis is not None else None,
        unread_analysis_id=unread_analysis_id,
        unread_report_doc_id=unread_report_doc_id,
        unread_line=unread_line,
        arms=bundle.arms,
        lead=lead,
        folds_alone=folds_alone,
    )
    if analysis is None:
        return page
    return page.model_copy(update={"analysis_id": analysis.id})


def _declared(declared: CampaignDeclaration) -> dict[str, Any]:  # Any: the campaign's fields, by name
    """A declaration as the campaign's fields it sets."""
    return {
        "question": declared.question,
        "declared_at": declared.declared_at,
        "declared_by": declared.declared_by or "",
        "arms": list(declared.arms),
        "aspects": list(declared.aspects),
        "floors": list(declared.floors),
    }


def _with_declaration(campaign: EvalCampaign, declared: CampaignDeclaration) -> EvalCampaign:
    """The campaign as it was declared when a bundle was assembled, and the call recorded since."""
    return campaign.model_copy(update=_declared(declared))


def _as_analysed(analysis: EvalAnalysis, campaign: EvalCampaign | None) -> EvalCampaign:
    """The campaign as the analysis's generation read it: what it declared then, and the call recorded since.

    An analysis outliving its campaign is still laid out, named by the campaign's id.
    """
    declared = analysis.design_snapshot
    snapshot = _declared(declared)
    if campaign is not None:
        return _with_declaration(campaign, declared)
    return EvalCampaign.model_validate(
        {
            "id": analysis.campaign_id,
            "scope_id": analysis.scope_id,
            "name": analysis.campaign_id,
            "subject_id": analysis.subject_id,
            "subject_kind": analysis.subject_kind,
            "behavior": analysis.behavior,
            "created_by": declared.declared_by or "unknown",
            **snapshot,
        }
    )


def _analysis_page(
    analysis: EvalAnalysis,
    campaign: EvalCampaign | None,
    *,
    trials: TrialDetail,
) -> CampaignReport:
    """A stored analysis laid out as its campaign's page, over everything frozen with it. One a retired writer wrote is
    code's page over that bundle, standing in for it with the retired writer's line (:func:`~threetears.evals.analysis.report.campaign.answer_blocks.retired_generation`)."""
    bundle = analysis_campaign_bundle(analysis)
    # Putting the bundle back together refused it unless each aspect's entry in the frame is that aspect's bundle
    # fingerprint in the entry's version, so an entry in the current version is the one each fold's source names.
    entries = analysis.campaign_bundle.get("aspects")
    fingerprints = {
        str(entry["aspect"]): str(entry["fingerprint"])
        for entry in (entries if isinstance(entries, list) else [])
        if isinstance(entry, dict) and str(entry.get("fingerprint", "")).startswith(FINGERPRINT_PREFIX)
    }
    retired = retired_generation(analysis.basis)
    if retired is not None:
        log.info("analysis %s reads code-only: %s", analysis.id, retired)
        page = _campaign_page(
            _as_analysed(analysis, campaign),
            bundle,
            assembled_at=analysis.generation.bundle_assembled_at,
            fingerprints=fingerprints,
            trials=trials,
            unread_analysis_id=analysis.id,
            unread_line=retired,
        )
        return CampaignReport.model_validate({**dict(page), "needs_regeneration": retired})
    return _campaign_page(
        _as_analysed(analysis, campaign),
        bundle,
        assembled_at=analysis.generation.bundle_assembled_at,
        analysis=analysis,
        fingerprints=fingerprints,
        trials=trials,
    )


def analysis_report(
    storage: AnalysisStore, analysis_id: str, scope_id: str, *, trials: TrialDetail = "table"
) -> CampaignReport:
    """Read one stored analysis as its campaign's page — the one page every door renders.

    Laid out over the bundles and code's readings frozen with the analysis at generation, so it reads as it was
    written whatever has been archived or re-run since: the campaign as it was declared then, every aspect's fold over
    its frozen bundle. The person's call is the campaign's as recorded now. Derived
    on every read and never written back, so a stored analysis stays pivotable.

    An analysis a retired report writer wrote (the memo writer's, ``basis="memo"``, or the report planner's,
    ``basis="plan"``) is laid out as code's page over the same frozen bundles (``basis="code_only"``), its first screen and ``needs_regeneration`` saying in one line
    which writer wrote it and to regenerate
    (:func:`~threetears.evals.analysis.report.campaign.answer_blocks.retired_generation`); its stored memo or plan is
    never read.

    :param storage: Where the analysis is read.
    :ptype storage: AnalysisStore
    :param analysis_id: The analysis to report.
    :ptype analysis_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param trials: How much of each failed or lost trial the page lays out (:data:`TrialDetail`): by default their
        table alone, each row naming its stored result; ``all`` adds each one's full record.
    :ptype trials: TrialDetail
    :return: The page, ``basis="code_only"`` over the analysis's frozen bundles, carrying its ``analysis_id``; :meth:`~threetears.evals.analysis.report.CampaignReport.write` puts it
        in a directory, and :func:`~threetears.evals.analysis.report.serialize_report` serializes it.
    :rtype: CampaignReport
    :raises NotFoundError: No analysis with that id in the scope.
    :raises RefusedInput: The analysis was written over a bundle of another shape; the remedy is to generate it again.
    """
    analysis = get_analysis(storage, analysis_id, scope_id)
    return _analysis_page(analysis, storage.load_campaign(analysis.campaign_id, scope_id), trials=trials)


def campaign_report(
    host: EvalHost,
    campaign_id: str,
    scope_id: str,
    *,
    trials: TrialDetail = "table",
    lead: Sequence[ReportBlock] | None = None,
) -> CampaignReport:
    """The campaign's page — THE answer to "what is this campaign's report", for every caller and every campaign.

    **The rule.** The campaign's newest analysis that is not archived. When it has none — no generation has run, or
    every analysis it had was archived (an archive is the claim that an analysis was wrong, so it is not the campaign's
    report) — the campaign's evidence assembled now (``basis="code_only"``), which says that no analysis was written.
    A particular analysis, archived ones included, is read by its id through :func:`analysis_report`, and a particular
    report document, any version, through :func:`report_doc_report`.

    **What the analysis is laid out as**, by what wrote it:

    - a report document's (``doc``, the report writer's): the page its stored document draws on its own snapshot
      (``basis="analysis"``, :attr:`~threetears.evals.analysis.report.CampaignReport.doc_page`), read as a stored
      document is, so it degrades rather than refuses
      (:func:`~threetears.evals.analysis.report_doc_store.read_report_doc`), its made line at its foot; the campaign's
      readings and folds are read off that snapshot's bundle, as the JSON form carries them;
    - a retired writer's (``memo``, the memo writer's, or ``plan``, the report planner's): the code-only page with the
      retired writer's line on its first screen
      (:func:`~threetears.evals.analysis.report.campaign.answer_blocks.retired_generation`,
      :attr:`~threetears.evals.analysis.report.CampaignReport.unread_analysis_id`). Its stored memo or plan is never
      read.

    A report document this version no longer reads — its record does not validate, its snapshot is gone, holds no
    bundle or no longer reads, or the campaign no longer lays out over it — is laid out as a retired writer's is: the
    code-only page with the one regenerate line. Neither is ever refused. Either way the page is the campaign's
    (:class:`~threetears.evals.analysis.report.CampaignReport`).

    Costs no model call: every page is storage reads and arithmetic. It reads one analysis however many the campaign
    holds: the store picks the newest not archived (:meth:`AnalysisStore.load_reported_analysis`), so a read's memory
    does not grow with the analyses archived.

    :param host: The host: where the campaign and its analyses are read, and the vocabulary its evidence is
        assembled in when there is no analysis.
    :ptype host: EvalHost
    :param campaign_id: The campaign.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param trials: How much of each failed or lost trial the page lays out (:data:`TrialDetail`): by default their
        table alone, each row naming its stored result (``result_get`` reads it); ``all`` adds each one's full
        record, which is most of a large campaign's page. A report document's page lays the folds, each such record
        in them, after the document as its evidence record with ``all``, and none with ``table``.
    :ptype trials: TrialDetail
    :param lead: The page's answer where its caller states its own (a harness whose eval reads a figure of its own),
        in place of the lead code writes on a code-only page
        (:func:`~threetears.evals.analysis.report.campaign.page.build_campaign_report`, ``lead``). A report document's
        page draws its own document, so it does not carry one. None for code's lead.
    :ptype lead: Sequence[ReportBlock] | None
    :return: The page; its ``basis`` says which it is.
    :rtype: CampaignReport
    :raises NotFoundError: No campaign with that id in the scope.
    """
    campaign = _load_campaign(host.storage, campaign_id, scope_id)
    reported = host.storage.load_reported_analysis(campaign_id, scope_id)
    if reported is None:
        return _code_only_page(host, campaign, trials=trials, lead=lead)
    if reported.report_doc_id is not None:
        try:
            read = read_report_doc(host.storage, reported.report_doc_id, scope_id)
        except NotFoundError as missing:
            return _stands_in(host, campaign, missing.message, trials=trials, unread_analysis_id=reported.id, lead=lead)
        except RefusedInput as unread:
            return _stands_in(host, campaign, unread.fault, trials=trials, unread_analysis_id=reported.id, lead=lead)
        return _doc_page(host, campaign, read, analysis_id=reported.id, trials=trials, lead=lead)
    # An analysis that links no report document was written by a retired writer (EvalAnalysis holds exactly one of a
    # memo, a plan and a document), so nothing of what it stored is read: code's page stands in for it.
    line = retired_generation(reported.basis) or NO_LONGER_READS
    return _stands_in(
        host,
        campaign,
        f"analysis {reported.id}",
        trials=trials,
        unread_analysis_id=reported.id,
        line=line,
        lead=lead,
    )


def report_doc_report(
    host: EvalHost, campaign_id: str, scope_id: str, report_doc_id: str, *, trials: TrialDetail = "table"
) -> CampaignReport:
    """One of the campaign's stored report documents, any version, laid out as :func:`campaign_report` lays out a
    report document's analysis: that document's page on its own snapshot, or, where this version no longer reads it,
    the code-only page with the one regenerate line, never refused.

    Read by its id, the page names no analysis (``analysis_id`` is None): what the campaign's report is stays
    :func:`campaign_report`'s answer. Its readings and folds are laid out over the campaign as the document's snapshot
    declared it, as every document's page is, so a version written before the campaign gained or lost an aspect or an
    arm reads as it was written.

    :param host: The host whose store holds the campaign and the document.
    :ptype host: EvalHost
    :param campaign_id: The campaign the document was written for.
    :ptype campaign_id: str
    :param scope_id: The scope they live in.
    :ptype scope_id: str
    :param report_doc_id: The document's id (``EvalAnalysis.report_doc_id``; a later version stores a new one).
    :ptype report_doc_id: str
    :param trials: How much of each failed or lost trial the page lays out (:func:`campaign_report`).
    :ptype trials: TrialDetail
    :return: The page.
    :rtype: CampaignReport
    :raises NotFoundError: No campaign with that id in the scope, or no report document with that id in the scope, or
        one written for another campaign, refused as soon as its record loads, before its snapshot is read. A record
        that does not validate cannot be attributed to a campaign: it reads as the code-only page of this one.
    """
    campaign = _load_campaign(host.storage, campaign_id, scope_id)
    try:
        read = read_report_doc(host.storage, report_doc_id, scope_id, campaign_id=campaign_id)
    except RefusedInput as unread:
        return _stands_in(host, campaign, unread.fault, trials=trials, unread_report_doc_id=report_doc_id)
    return _doc_page(host, campaign, read, analysis_id=None, trials=trials)


def _doc_page(
    host: EvalHost,
    campaign: EvalCampaign,
    read: StoredReportPage,
    *,
    analysis_id: str | None,
    trials: TrialDetail,
    lead: Sequence[ReportBlock] | None = None,
) -> CampaignReport:
    """A stored report document's page, its readings and folds off its snapshot's bundle laid out over the campaign as
    that bundle declared it (what the document was written against, which its analysis's ``design_snapshot`` copies);
    the code-only page of ``campaign`` with the one regenerate line where the snapshot holds no bundle or its bundle
    no longer lays out. With ``trials="all"`` the folds follow the document, each failed or lost trial's record in
    them, as the code-only page draws them. A caller's ``lead`` shows only on the code-only page standing in: the
    document's page holds no blocks of its own."""
    snapshot = read.snapshot
    try:
        if snapshot.bundle is None or snapshot.bundle_assembled_at is None:
            raise RefusedInput(
                f"report document {read.record.id}'s snapshot {snapshot.id} holds no bundle", remedy=NO_LONGER_READS
            )
        bundle = CampaignBundle.from_dict(snapshot.bundle, what=f"report snapshot {snapshot.id}")
        declared = _with_declaration(campaign, bundle.declaration)
        # The document's page drops the campaign's own blocks, so each fold keeps what only those would state.
        page = _campaign_page(
            declared, bundle, assembled_at=snapshot.bundle_assembled_at, trials=trials, folds_alone=True
        )
    except RefusedInput as unread:
        if analysis_id is None:
            return _stands_in(
                host, campaign, unread.fault, trials=trials, unread_report_doc_id=read.record.id, lead=lead
            )
        return _stands_in(host, campaign, unread.fault, trials=trials, unread_analysis_id=analysis_id, lead=lead)
    return CampaignReport.model_validate(
        {
            **dict(page),
            "basis": "analysis",
            "analysis_id": analysis_id,
            "blocks": [],
            "doc_page": read.page,
            "report_doc": read.summary(),
            "doc_trials": trials == "all",
        }
    )


def _stands_in(
    host: EvalHost,
    campaign: EvalCampaign,
    why: str,
    *,
    trials: TrialDetail,
    unread_analysis_id: str | None = None,
    unread_report_doc_id: str | None = None,
    line: str = NO_LONGER_READS,
    lead: Sequence[ReportBlock] | None = None,
) -> CampaignReport:
    """The code-only page standing in for an analysis or a report document this version no longer reads: its first
    screen says so in the one line (``NO_LONGER_READS``, or the retired writer's), and its ``needs_regeneration`` is
    that line with why."""
    named = unread_analysis_id or unread_report_doc_id
    log.info("campaign %s's report %s reads code-only: %s", campaign.id, named, why)
    page = _code_only_page(
        host,
        campaign,
        trials=trials,
        unread_analysis_id=unread_analysis_id,
        unread_report_doc_id=unread_report_doc_id,
        unread_line=line,
        lead=lead,
    )
    return CampaignReport.model_validate({**dict(page), "needs_regeneration": f"{line} ({why})"})


def _code_only_page(
    host: EvalHost,
    campaign: EvalCampaign,
    *,
    trials: TrialDetail = "table",
    unread_analysis_id: str | None = None,
    unread_report_doc_id: str | None = None,
    unread_line: str = NO_LONGER_READS,
    lead: Sequence[ReportBlock] | None = None,
) -> CampaignReport:
    """The campaign's code-only page over its evidence assembled now, whatever analyses it holds.

    :param host: The host: where the campaign's runs, results and insights are read, and its vocabulary.
    :ptype host: EvalHost
    :param campaign: The campaign.
    :ptype campaign: EvalCampaign
    :param trials: How much of each failed or lost trial each fold lays out.
    :ptype trials: TrialDetail
    :param unread_analysis_id: The campaign's analysis this version no longer reads, which the page stands in for,
        saying so in the one regenerate line; None for none.
    :ptype unread_analysis_id: str | None
    :param unread_report_doc_id: The report document, read by its id, that this version no longer reads and the page
        stands in for; None for none.
    :ptype unread_report_doc_id: str | None
    :param unread_line: What ``unread_analysis_id`` was, as the line says it.
    :ptype unread_line: str
    :param lead: The page's answer where its caller states its own; None for code's lead.
    :ptype lead: Sequence[ReportBlock] | None
    :return: The page, ``basis="code_only"``.
    :rtype: CampaignReport
    """
    assembled_at = utc_now_iso()
    bundle = _assembled_campaign_bundle(campaign, storage=host.storage, profile=host.profile)
    return _campaign_page(
        campaign,
        bundle,
        assembled_at=assembled_at,
        trials=trials,
        unread_analysis_id=unread_analysis_id,
        unread_report_doc_id=unread_report_doc_id,
        unread_line=unread_line,
        lead=lead,
    )


def campaign_verdicts(host: EvalHost, campaign_id: str, scope_id: str) -> list[Verdict]:
    """Every verdict code reaches on the campaign's evidence as it stands now, typed — what a CI gate reads.

    The code-only page's :attr:`~threetears.evals.analysis.report.CampaignReport.verdicts` over the evidence assembled
    now — on every aspect, each guardrail, each bar on each cell, each contrast against the control, each naming its
    aspect — whether or not an analysis was generated: a gate rests on code's verdicts, never on an analyst's words,
    and an analysis's frozen surface holds no contrast. Costs no model call.

    :param host: The host whose store holds the campaign, and whose vocabulary its evidence is assembled in.
    :ptype host: EvalHost
    :param campaign_id: The campaign.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :return: The verdicts, aspect by aspect in the campaign's order, each in its page's order.
    :rtype: list[Verdict]
    :raises NotFoundError: No campaign with that id in the scope.
    """
    return _code_only_page(host, _load_campaign(host.storage, campaign_id, scope_id)).verdicts


# ---------------------------------------------------------------------------
# The person's call — recorded beside code's readings, never in place of them
# ---------------------------------------------------------------------------


def record_campaign_call(
    host: EvalHost,
    campaign_id: str,
    scope_id: str,
    *,
    arm: str,
    called_by: str,
    reason: str = "",
    floors_overridden: Sequence[str] = (),
    bars_overridden: Sequence[OverriddenBar] = (),
    analysis_id: str | None = None,
) -> EvalCampaign:
    """Record a person's call on a campaign: the arm decided, why, and every broken floor and missed bar it overrides.

    Code reports, the writer recommends, a person decides. The call is read against the readings of the page it will
    appear on — never against a memo's words: with ``analysis_id``, the readings frozen with that analysis, which must
    be the campaign's report (its newest analysis not archived); without, code's readings of the evidence as it stands
    now. It is refused where it would leave a warning unanswered: an arm with a broken floor is recorded only when ``floors_overridden`` names each such floor by its
    key, and an arm that misses an absolute bar only when ``bars_overridden`` names each such bar by its aspect and
    measure; naming one the arm did not break or miss is refused too, so an override always answers a warning the
    page shows. The arm is matched by its variant key, which is what is recorded: a derived arm's label can shift when
    a variant joins. A measurement (one arm) has nothing to decide between, so a call on one is refused. In a choice,
    an arm shown behind another on some aspect is never refused: the person decides, and the page sets a warning line
    beside the call for each such pair. A new call replaces the campaign's earlier one; the page dates whichever it
    holds, and says how many runs have joined since it was made. **Known limit:** only runs added since are counted;
    a run curated away, or a result scored late on a run the call already read, is not told apart.

    :param host: The host: where the campaign's runs, results and analyses are read, and its vocabulary.
    :ptype host: EvalHost
    :param campaign_id: The campaign.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param arm: The decided arm, by the label the page shows it under or by its variant key; the control's keeps it.
    :ptype arm: str
    :param called_by: Who made the call, as the recording surface knows them.
    :ptype called_by: str
    :param reason: Why, in the decider's words.
    :ptype reason: str
    :param floors_overridden: The key (``Floor.key``) of each broken floor the call overrides.
    :ptype floors_overridden: Sequence[str]
    :param bars_overridden: Each missed bar the call overrides, by aspect and measure.
    :ptype bars_overridden: Sequence[OverriddenBar]
    :param analysis_id: The analysis whose memo the call was made over — the campaign's report; None for code's
        readings alone.
    :ptype analysis_id: str | None
    :return: The campaign as persisted, holding the call.
    :rtype: EvalCampaign
    :raises NotFoundError: No campaign with that id in the scope, or no analysis ``analysis_id`` in it.
    :raises ValidationFailedError: The campaign is a measurement; the arm is none of the campaign's; the analysis is
        another campaign's, archived, not the campaign's report, or frozen over a bundle of a shape this build no
        longer reads (generate it again); a broken floor or missed bar of the arm is not overridden, or an override names one the
        arm did not break or miss; or a floor or bar is named twice.
    :raises ConflictError: The campaign changed while its readings were read.
    """
    campaign = _load_campaign(host.storage, campaign_id, scope_id)
    if analysis_id is None:
        bundle = _assembled_campaign_bundle(campaign, storage=host.storage, profile=host.profile)
    else:
        try:
            bundle = analysis_campaign_bundle(_reported_analysis(host.storage, campaign_id, scope_id, analysis_id))
        except RefusedInput as unread:
            # Frozen over a bundle of another shape: no call can be read against it until it is generated again.
            raise ValidationFailedError.refusing(unread) from unread
    readings = bundle.readings
    if readings.kind == "measurement":
        raise ValidationFailedError(
            f"campaign {campaign_id} measures one arm, so there is nothing to decide between and no call to record",
            remedy="read its measures on its page (report_read); gather another arm's runs to make it a choice",
        )
    decided = next((each for each in bundle.arms if each.variant_key == arm), None) or next(
        (each for each in bundle.arms if each.label == arm), None
    )
    if decided is None:
        raise ValidationFailedError(
            f"campaign {campaign_id} has no arm {arm!r}",
            remedy=f"name one of its arms by label or variant key: {[each.label for each in bundle.arms]}",
        )
    keys = {decided.variant_key, *resolve_control_arm(decided).variant_keys}
    broken = {status.floor for status in readings.broken_floors() if status.arm in keys}
    missed = {(status.aspect, status.measure) for status in readings.missed_bars() if status.arm in keys}
    named_bars = {(bar.aspect, bar.measure) for bar in bars_overridden}
    unanswered = [
        *(f"floor {key}" for key in sorted(broken - set(floors_overridden))),
        *(f"the bar on {measure} on {aspect}" for aspect, measure in sorted(missed - named_bars)),
    ]
    if unanswered:
        raise ValidationFailedError(
            f"{decided.label} breaks or misses {', '.join(unanswered)}, and the call does not override it",
            remedy=(
                "override each by name — floors_overridden takes a floor's key, bars_overridden a bar's aspect and "
                "measure — with the reason in `reason`, or decide another arm"
            ),
        )
    stray = [
        *(f"floor {key}" for key in sorted(set(floors_overridden) - broken)),
        *(f"the bar on {measure} on {aspect}" for aspect, measure in sorted(named_bars - missed)),
    ]
    if stray:
        raise ValidationFailedError(
            f"the call overrides {', '.join(stray)}, which {decided.label} does not break or miss",
            remedy="override only the broken floors and missed bars the page warns of for the decided arm",
        )
    try:
        call = CampaignCall(
            arm=decided.label,
            variant_key=decided.variant_key,
            reason=reason,
            floors_overridden=list(floors_overridden),
            bars_overridden=list(bars_overridden),
            analysis_id=analysis_id,
            run_ids=list(campaign.run_ids),
            called_by=called_by,
        )
    except ValueError as invalid:
        raise ValidationFailedError(str(invalid), remedy="name each floor and each bar once") from invalid
    return _save_call(host.storage, campaign, call)


def _reported_analysis(storage: AnalysisStore, campaign_id: str, scope_id: str, analysis_id: str) -> EvalAnalysis:
    """The analysis a call names, refused unless it is the campaign's report: its newest analysis not archived.

    :raises NotFoundError: No analysis with that id in the scope.
    :raises ValidationFailedError: It is another campaign's, archived, or not the newest of the campaign's live ones.
    """
    named = get_analysis(storage, analysis_id, scope_id)
    remedy = "name the analysis the campaign's page reports (report_read gives its analysis_id), or none"
    if named.campaign_id != campaign_id:
        raise ValidationFailedError(
            f"analysis {analysis_id} is not an analysis of campaign {campaign_id}", remedy=remedy
        )
    if named.archived:
        raise ValidationFailedError(f"analysis {analysis_id} is archived, so no page reports it", remedy=remedy)
    reported = storage.load_reported_analysis(campaign_id, scope_id)
    if reported is None or reported.id != analysis_id:
        raise ValidationFailedError(
            f"analysis {analysis_id} is not campaign {campaign_id}'s report: its page reports "
            f"{reported.id if reported is not None else 'no analysis'}",
            remedy=remedy,
        )
    return named


@serialized_campaign_write
def _save_call(storage: EvalStorage, read: EvalCampaign, call: CampaignCall) -> EvalCampaign:
    """Write ``call`` onto the campaign ``read`` was, refusing when it has changed since: the call was read against
    its readings then.

    :raises ConflictError: The stored campaign's declaration or membership is no longer ``read``'s.
    :raises NotFoundError: The campaign was deleted meanwhile.
    """
    current = _load_campaign(storage, read.id, read.scope_id)
    unchanged = ("question", "arms", "aspects", "floors")
    if any(getattr(current, field) != getattr(read, field) for field in unchanged):
        raise ConflictError(
            f"campaign {read.id} changed while its readings were read, so the call was read against another campaign",
            remedy="record the call again; it is read against the campaign as it stands",
        )
    storage.save_campaign(current.model_copy(update={"call": call}))
    log.info("eval.record_campaign_call campaign=%s scope=%s arm=%s", read.id, read.scope_id, call.variant_key)
    return _load_campaign(storage, read.id, read.scope_id)


# ---------------------------------------------------------------------------
# The insight ledger
# ---------------------------------------------------------------------------


def list_insights(
    storage: AnalysisStore,
    scope_id: str,
    *,
    subject_id: str | None = None,
    scope: str | None = None,
    source_campaign_id: str | None = None,
) -> list[EvalInsight]:
    """List insights in a storage scope, newest observation first.

    ``scope_id`` is the storage partition; ``scope`` below is the insight's own free-prose
    field describing what it holds for, and the two are unrelated.

    The ledger read that makes an insight's id reachable at all: generation
    reports how many it minted, and the bundle feeds prior insights back to
    the generator, but neither hands an operator the id of one that turned out
    to be wrong.

    **``scope`` matches by substring, case-insensitively, and the two id
    filters do not.** That asymmetry is the point rather than an oversight.
    ``scope`` is declared free prose on the model itself and a real store holds
    long sentences in it, so an exact match is not a filter anyone can
    type: **every** hand-formed ``scope`` argument returned an empty list, and
    an empty list is indistinguishable from a truthful "no insights here" —
    the silent-empty-filter defect on a second surface. ``subject_id`` and
    ``source_campaign_id`` are ids, taken from a listing rather than composed,
    and a substring match over an id would silently widen a filter that reads
    as exact.

    **An unmatched ``scope`` is refused, an unmatched id is not.** A refusal
    says "this cannot be what you meant", and for free prose with no
    discoverable vocabulary that is true — the caller has no way to learn what
    exists except by being told. For an id it is not: an id that matches
    nothing is a real answer ("that campaign minted no insights"), and the
    caller can re-check the id where they got it. The MCP surface names what it
    searched on an empty id-filtered read instead; REST returns the empty list
    its ``response_model`` promises.

    The substring filter runs here rather than in storage because
    ``query_insights`` builds ``field_eq`` for ``by_doc_type``, which has no
    substring channel — and because ``query_insights`` passes no limit, so an
    unfiltered read already materialises the scope's whole ledger and
    filtering it in memory is strictly less work than the call every bare
    listing makes. (``by_doc_type`` itself does take ``limit``;
    what has no limit is this caller.)

    **That is an invariant now, not an observation.** A substring filter
    applied after the read is only correct while the read returns the whole
    set: page ``query_insights`` and this silently becomes "matches within the
    first page", which reports a filtered miss as an absence. The constraint is
    recorded on ``query_insights`` too, and a growing ledger will
    eventually force the question — an unbounded ledger cannot be read whole
    forever, and whoever bounds it owes this filter a substring channel.

    :param storage: Where the insight ledger is read.
    :ptype storage: AnalysisStore
    :param scope_id: The storage scope whose ledger to read.
    :ptype scope_id: str
    :param subject_id: Optional equality filter on the subject an insight is about.
    :ptype subject_id: str | None
    :param scope: Optional case-insensitive SUBSTRING filter over the insight's
        free-text scope.
    :ptype scope: str | None
    :param source_campaign_id: Optional equality filter on the minting campaign.
    :ptype source_campaign_id: str | None
    :return: The matching insights, newest observation first.
    :rtype: list[EvalInsight]
    :raises ValidationFailedError: ``scope`` was given and matched no insight.
    """
    insights = storage.query_insights(
        scope_id,
        subject_id=subject_id,
        source_campaign_id=source_campaign_id,
    )
    if scope is None:
        return insights

    needle = scope.strip().casefold()
    # prose-canary: allow — an operator's own search term filtering insights, not code judging what a model wrote
    matched = [insight for insight in insights if needle in insight.scope.casefold()]
    if matched:
        return matched

    present = sorted({insight.scope for insight in insights if insight.scope})
    shown = ", ".join(f"'{value[:60]}'" for value in present[:_SCOPE_SAMPLE])
    more = f" (+{len(present) - _SCOPE_SAMPLE} more)" if len(present) > _SCOPE_SAMPLE else ""
    vocabulary = f" Scopes present: {shown}{more}." if present else " No insight in this set records a scope at all."
    named = describe_insight_id_filters(subject_id, source_campaign_id)
    narrowed = f" (already narrowed by {named})" if named else ""
    raise ValidationFailedError(
        f"No insight's scope contains {scope!r}. Searched {len(insights)} insight(s){narrowed}.{vocabulary}",
        remedy="name a scope one of them records; `scope` matches by substring, case-insensitively",
    )


if TYPE_CHECKING:

    def _eval_storage_satisfies_the_port(storage: EvalStorage) -> None:
        """Hold the engine's own store to this consumer's port, so a drifted signature fails typecheck."""
        stores: tuple[AnalysisStore, CampaignReadStore] = (storage, storage)
        del stores


__all__ = [
    "AnalysisGenerationEstimate",
    "AnalysisStore",
    "DEFAULT_WRITER_UNRESOLVED",
    "analysis_report",
    "report_doc_report",
    "describe_insight_id_filters",
    "get_analysis",
    "assemble_campaign_bundle",
    "require_evidence",
    "inspect_analysis_bundle",
    "campaign_report",
    "campaign_verdicts",
    "record_campaign_call",
    "inspect_campaign_bundle",
    "running_analysis_attempt",
    "list_insights",
    "record_written_generation",
    "WrittenAttempt",
]
