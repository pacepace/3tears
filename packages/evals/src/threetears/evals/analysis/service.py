"""The analysis service — generating, reading and curating a campaign's analyses, insights and reporter cases.

Every operation here is one a client of the packages needs to drive the analysis lens: generate
a campaign's analysis (the report planner writes it as a report plan, stored beside it) and record how the attempt
ended, read analyses and their attempts back,
re-assemble the bundle a generation read, read a stored analysis as its report, read
the insight ledger, and freeze, bank and calibrate the reporter cases that measure the generator
itself. They are module functions whose dependencies are parameters, so a host calls them
with its own store and its own bindings; a host's service delegates to them.

**What arrives as an argument.** An operation that reads the host's vocabulary — assembling a
bundle, generating over one, freezing one — takes the
:class:`~threetears.evals.kernel.host.EvalHost`: its storage, its profile, the client factory the
generator is built from and the describer that says what a provider failure may keep. An operation
that reads documents alone takes storage, through :class:`AnalysisStore`, which a host's store
satisfies by having the methods; campaigns and runs are loaded through it, a missing one refused
with ``NotFoundError``. A template the reporter-case freeze judges against is loaded through the
caller's own loader (``load_template``), because a host may refuse a template for more than its
absence and that refusal is the host's to decide. The generator prompt arrives resolved
(``resolve_prompt``) — which prompt registry it comes from is the host's — and so do the registry
key the prompt was resolved under and the output cap the log line discloses.

**What stays with the host.** Admitting a generation against a job manager — one generation per
campaign at a time, the background task, the poll — is the run package's job manager, which
this package may not import, so that composition is the host's.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol

from threetears.evals.analysis.bundle.assemble import CampaignReadStore, assemble_context_bundle
from threetears.evals.analysis.bundle.insights import superseding_insights
from threetears.evals.analysis.errors import GenerationError, SoundnessRefusal
from threetears.evals.analysis.campaign_bundle import (
    CampaignBundle,
    build_campaign_bundle,
    build_campaign_user_message,
)
from threetears.evals.analysis.campaign_readings import campaign_readings
from threetears.evals.analysis.campaigns import campaign_arms
from threetears.evals.analysis.generator import GenerationTally
from threetears.evals.kernel.inner_ceilings import MAX_GENERATION_CALLS
from threetears.evals.analysis.planned_analysis import generate_planned_analysis, report_input_digest, report_plan_of
from threetears.evals.analysis.planner import first_plan_request, plan_user_message
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_input_build import report_input as build_report_input
from threetears.evals.kernel.numbers import format_number
from threetears.evals.analysis.reporter_bank import case_limits, decidable_reporter_case_bank, read_calibration
from threetears.evals.analysis.reporter_kind import (
    MEMO_CASE_VERSION,
    PLAN_CASE_VERSION,
    REPORTER_KIND,
    LabelCriterion,
    ReporterCase,
    ReporterLabel,
    render_memo_as_written,
    render_plan_for_judge,
    reporter_case_of,
    reporter_case_payload,
)
from threetears.evals.analysis.bundle.schema import FINGERPRINT_PREFIX, BundleInspection
from threetears.evals.analysis.report import CampaignReport, Verdict, build_campaign_report, build_code_only_report
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.model import TrialDetail
from threetears.evals.analysis.report.plan_page.build import plan_page
from threetears.evals.analysis.report.plan_page.words import MADE_TEMPLATE
from threetears.evals.analysis.report.build import analysis_campaign_bundle, analysis_writer
from threetears.evals.analysis.report.words import campaign_arm_tags, declared_arm_names, printed_arm_labels
from threetears.evals.analysis.report.campaign.memo_blocks import NO_LONGER_READS
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
    ProviderRefusedError,
    RefusedInput,
    StorageError,
    ValidationFailedError,
)
from threetears.evals.kernel.refusals import job_refusal
from threetears.evals.kernel.run_pages import ResultKey, results_without_payloads
from threetears.evals.schema.refusal import JobRefusal
from threetears.evals.kernel.job_leases import ANALYSIS_JOB, JobLeases, write_held_record
from threetears.evals.kernel.job_timeout import EvalJobTimeout, timed_out_detail
from threetears.evals.kernel.storage import JobRecordStore
from threetears.evals.schema.models import EvalTestCase, utc_now_iso
from threetears.evals.kernel.offload import run_blocking, wait_through_cancellation
from threetears.evals.kernel.capped_completion import refresh_prices
from threetears.evals.kernel.out_of_run import AdmittedCall, OutOfRunBudget, PlannedCall
from threetears.evals.kernel.provider import describe_failure, log_provider_failure, traceback_is_safe
from threetears.evals.schema.logs import get_logger
from threetears.evals.kernel.remedies import (
    FIND_ANALYSIS,
    FIND_CAMPAIGN,
    FIND_RUN,
    FIND_TEMPLATE,
    WRITER_REMEDY,
)
from pydantic import ValidationError

if TYPE_CHECKING:
    from threetears.evals.kernel.storage import EvalStorage
    from threetears.evals.analysis.reporter_bank import ReporterCalibration, ReporterCaseBank
    from threetears.evals.kernel.campaign import (
        AspectAnalysis,
        AttemptOutcome,
        EvalAnalysis,
        EvalInsight,
        EvalReportPlan,
    )
    from threetears.evals.kernel.host.profile import HostProfile
    from threetears.evals.schema.models import CalibrationRating, EvalResult, EvalRun, EvalTemplate
    from threetears.evals.kernel.host.eval_host import EvalHost
    from concurrent.futures import Executor

    from threetears.evals.schema.completion import BoundCompletionClient, CompletionResult

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

    def save_report_plan(self, plan: EvalReportPlan, /) -> None:
        """Write the report plan a planned analysis was written as; raises ``StorageError`` rather than returning a
        flag."""
        ...

    def load_report_plan(self, plan_id: str, scope_id: str, /) -> EvalReportPlan | None:
        """Load one report plan by id (``EvalAnalysis.report_plan_id``), or ``None`` when it does not resolve."""
        ...

    def delete_report_plan(self, plan_id: str, scope_id: str, /) -> bool:
        """Delete one report plan — a generation withdrawing what it stored."""
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

    def load_template(self, template_id: str, scope_id: str, /) -> EvalTemplate | None:
        """Load one template as stored in a scope, or ``None`` when it does not resolve — the calibration read's raw read."""
        ...

    def query_eval_result_keys_by_run(self, run_id: str, scope_id: str, /) -> list[ResultKey]:
        """The key of every result of one run within a scope, in one read that ships no other field — the calibration
        read's pass over the run, before it reads the run's results by id a page at a time."""
        ...

    def load_eval_results(self, result_ids: Sequence[str], scope_id: str, /) -> list[EvalResult]:
        """The named results within a scope, in one read by id — a page of the calibration read's."""
        ...

    def query_calibration_ratings(
        self, scope_id: str, /, *, run_id: str | None = None, run_ids: Sequence[str] | None = None
    ) -> list[CalibrationRating]:
        """Calibration ratings in a scope, oldest first, of one run's results (the calibration read) or of several
        runs' (the bundle's)."""
        ...

    def analysis_archived(self, analysis_id: str, scope_id: str, /) -> bool | None:
        """Whether one stored analysis is archived, or ``None`` when it no longer resolves — what a generation's
        insight write asks of each ledger insight's minting analysis."""
        ...

    def save_test_case(self, test_case: EvalTestCase, /) -> None:
        """Write a frozen reporter case; raises ``StorageError`` rather than returning a flag."""
        ...

    def query_test_cases(self, scope_id: str, /, *, template_id: str | None = None) -> list[EvalTestCase]:
        """Every test case in a scope, optionally narrowed to one template — the reporter case bank's read."""
        ...

    def load_test_cases_by_ids(self, test_case_ids: list[str], scope_id: str, /) -> list[EvalTestCase]:
        """The named test cases within a scope; an id that does not resolve is absent from the answer."""
        ...


class PreparedGeneration(NamedTuple):
    """Everything an analysis generation needs before it spends anything, checked and built.

    Split from the spend so every refusal (an unknown campaign, an empty bundle, an unknown preset, a writer the host
    does not allow, a first call over the cap) is raised before any call is made: to the caller of an estimate, and,
    in a background generation, inside its job, which ends its recorded attempt refused with it.
    """

    campaign_id: str
    scope_id: str
    bundle: CampaignBundle
    #: The report input built from ``bundle``, which the planner reads (:func:`~threetears.evals.analysis.report_input_build.report_input`).
    report_input: ReportInput
    assembled_at: str
    prompt: str
    #: Built from the host's client factory for the analysis role. The model is read back off it
    #: rather than re-derived, so the pre-spend disclosure and the stored provenance name what the
    #: client will actually call.
    client: BoundCompletionClient
    resolved_model: str
    attempt_id: str
    #: Every call the generation makes is priced and admitted against this before it is sent, and
    #: ledgered (purpose ``analysis``) after; its first call is already admitted.
    calls: BudgetedGenerator
    #: When the attempt this generation runs as was started, when a background start recorded it before its job
    #: prepared anything; ``None`` for a generation no recorded attempt precedes. The ended attempt keeps it, so
    #: it never reads as started when its preparation finished.
    started_at: str | None = None


#: Characters per token the logged input size assumes. A ballpark, not a measurement, and not a bound: a JSON-heavy
#: bundle (ids, numbers, punctuation) ran about 1.7 in production (#827), where 2 reads about 15% under the bill, and
#: prose runs about 4, where it reads about twice over. The cap is priced on a separate one-per-byte bound, never on
#: this.
CHARS_PER_TOKEN = 2

#: What the generation's ``approx_input_tokens`` log figure is, said on the line that logs it.
APPROX_INPUT_TOKENS_BASIS = f"characters of the exact request sent / {CHARS_PER_TOKEN}"


def _approx_input_tokens(system: str, user: str, response_format: dict[str, Any] | None) -> int:
    """A ballpark of the input tokens a call sends, for the pre-spend log line: its characters over
    :data:`CHARS_PER_TOKEN`, counting the system prompt, user message and response schema it actually sends."""
    schema = json.dumps(response_format) if response_format is not None else ""
    return (len(system) + len(user) + len(schema)) // CHARS_PER_TOKEN


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


class BudgetedGenerator:
    """A generation's calls, each priced and admitted against its out-of-run budget before it is sent, and ledgered.

    The generator is handed this as its client and its :meth:`admit` as its admission hook
    (:func:`~threetears.evals.analysis.generator.generate_analysis`), so the generator never learns what a
    cap is while every call it makes — the repair round-trip, whose prompt exists only once the first output
    is refused, included — goes through :class:`~threetears.evals.kernel.out_of_run.OutOfRunBudget`'s one
    rule: priced on the client before it is made, refused when the price would pass what is left of the cap,
    and written to the out-of-run ledger however it ends.
    """

    def __init__(self, budget: OutOfRunBudget, client: BoundCompletionClient) -> None:
        """Bind the budget to the client every call is priced on and made through.

        :param budget: The generation's own budget.
        :ptype budget: OutOfRunBudget
        :param client: The generator client; this does not own it.
        :ptype client: BoundCompletionClient
        """
        self._budget = budget
        self._client = client
        self._pending: AdmittedCall | None = None

    @property
    def budget(self) -> OutOfRunBudget:
        """The budget the calls are held to — its cap, what is committed, the rows it wrote."""
        return self._budget

    def admit(self, system: str, user: str, response_format: dict[str, Any] | None) -> None:
        """Price and admit the next call, or refuse it before anything is sent.

        A call already admitted with exactly this prompt (the first call, admitted when the generation was
        prepared) is not priced twice.

        :raises ValidationFailedError: The call cannot be priced under an enforced cap, or its price would pass
            what is left of the cap.
        """
        call = PlannedCall(system=system, user=user, response_format=response_format)
        if self._pending is not None and self._pending.call == call:
            return
        (self._pending,) = self._budget.admit(self._client, "analysis", [call])

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> CompletionResult:
        """Make the admitted call through the budget, which ledgers it whether it returns or raises.

        :raises ValueError: No call with exactly this prompt was admitted — a call that would go unpriced.
        """
        pending, self._pending = self._pending, None
        if pending is None or pending.call != PlannedCall(system=system, user=user, response_format=response_format):
            raise ValueError("an analysis generator call was sent without being admitted against its budget")
        result: CompletionResult = (await self._budget.generate(self._client, pending)).result
        return result


class AnalysisGenerationEstimate(EvalBaseModel):
    """What a generation would be held to before it starts: its bound, and whether it would start.

    ``analysis_estimate`` (:func:`~threetears.evals.ops.analysis_estimate`) answers it for the report writer, whose
    bound is the host's report budget; :func:`estimate_analysis_generation` for the report planner, whose bound is the
    host's out-of-run cap.

    :ivar campaign_id: The campaign.
    :ivar generator_model: The model the generation would call, as the host's client resolved it.
    :ivar first_call_ceiling_usd: The most the planner's first call can cost, as the client prices it; ``None`` when
        it cannot say, and always for the report writer, whose every turn is priced against the budget when it exists
        (a turn's price depends on the conversation before it).
    :ivar max_calls: How many calls a generation can make: for the report writer, the most turns it can send
        (``writer_turns`` of the host's limits, or the report budget's turn cap where that is fewer: the count
        ``Ceilings().writer_ceiling_s`` multiplies); for the planner, its
        structure call and its prose call, and the one repair round-trip a refused reply of either buys.
    :ivar cap_usd: The dollar bound the generation is held to: the report writer's report budget, or the planner's
        out-of-run cap. ``None`` when there is no dollar cap: the planner's host enforces none, or the report budget
        caps tokens and turns (``bound`` says which), never read as $0.
    :ivar would_start: Whether the start's checks pass: for the report writer, the host's checks and whether its turns
        can be bounded; for the planner, whether its first call would be admitted. For the report writer it is no
        promise that a report is written: what it does not check is in ``unchecked``.
    :ivar refusal: Why it would not, when it would not.
    :ivar bound: The bound in words (the report budget, in dollars or in tokens and turns); ``None`` for the planner,
        whose bound is ``cap_usd``.
    :ivar unchecked: What the estimate does not check and the job checks before any call (the report writer's: the
        campaign's evidence, read when the job freezes it); ``None`` for the planner, which reads it here.
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


class _Assembled(NamedTuple):
    """What a generation reads, checked and built: the bundle and its report input, the prompt, the client and its first
    call."""

    campaign: EvalCampaign
    bundle: CampaignBundle
    report_input: ReportInput
    assembled_at: str
    prompt: str
    client: BoundCompletionClient
    first_call: PlannedCall


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


async def _assemble(
    host: EvalHost,
    campaign_id: str,
    scope_id: str,
    *,
    model: str | None,
    resolve_prompt: Callable[[], Awaitable[str]],
) -> _Assembled:
    """Load the campaign, assemble its campaign bundle and its report input off the event loop, refuse an empty one,
    resolve the prompt, build the planner's first request and the client.

    Every aspect is assembled, and one plan is generated over all of them; a campaign of one aspect is the same call.

    :raises NotFoundError: No campaign with that id.
    :raises ValidationFailedError: The bundle has no resolvable runs, the preset does not exist, or the writer
        model is not one the host allows (``HostProfile.analysis_writer_models``).
    :raises ValidationFailedError: The report input's planner view is over its budget once every trim is made.
    :raises ValueError: The host supplies no completion clients.
    """
    clients = host.completion_clients("an analysis generation")
    storage, executor = host.storage, host.blocking_executor
    campaign = await run_blocking(executor, _load_campaign, storage, campaign_id, scope_id)

    # Stamped BEFORE assembly, because assembly is where the insight ledger is read: an
    # insight observed at or after this instant cannot have been in this bundle, and
    # `inspect_analysis_bundle` re-reads the ledger as of exactly this cutoff.
    assembled_at = utc_now_iso()
    # Assembly reads every member run, its results and the insight ledger — the heaviest read the engine
    # makes — so it runs on the host's blocking executor, never on the loop a transport serves calls on.
    bundle = await run_blocking(executor, _assembled_campaign_bundle, campaign, storage=storage, profile=host.profile)
    require_evidence(bundle, campaign_id, scope_id)

    # A requested writer the host does not allow is refused before the prompt is resolved or a client
    # built; the host's default is checked once the client says which model it resolved to.
    if model is not None and (ineligible := host.profile.analysis_writer_refusal(model)) is not None:
        raise ValidationFailedError(ineligible, remedy=WRITER_REMEDY)
    prompt = await resolve_prompt()
    # The report input is read off the bundle's typed objects, every reading and contrast of it: arithmetic over the
    # whole bundle, so it runs where the assembly did, never on the loop.
    report = await run_blocking(executor, build_report_input, bundle)
    try:
        system, user, contract = await run_blocking(executor, first_plan_request, report, prompt, host.profile)
    except RefusedInput as refused:
        # Over the planner's budget once every trim is made: nothing is sent, and a caller is told why as it is told
        # every other refusal a generation meets before it spends.
        raise ValidationFailedError(refused.fault, remedy=refused.remedy) from refused

    # The effective model is read back off the built client (its bound ``model_name``) for the
    # pre-spend disclosure log and stored provenance, rather than re-implementing the host's
    # resolution cascade here.
    client = clients("analysis", model)
    if model is None and (ineligible := host.profile.analysis_writer_refusal(client.model_name)) is not None:
        await _release(client)
        raise ValidationFailedError(f"the host's default writer: {ineligible}", remedy=WRITER_REMEDY)
    return _Assembled(
        campaign=campaign,
        bundle=bundle,
        report_input=report,
        assembled_at=assembled_at,
        prompt=prompt,
        client=client,
        first_call=PlannedCall(system=system, user=user, response_format=contract),
    )


def _generation_budget(host: EvalHost, assembled: _Assembled, out_of_run_cap_usd: float | None) -> OutOfRunBudget:
    """The one budget a generation's calls are held to: the host's out-of-run cap, ledgered in the campaign's scope."""
    return OutOfRunBudget(
        store=host.storage,
        scope_id=assembled.campaign.scope_id,
        cap_usd=out_of_run_cap_usd,
        subject_id=assembled.campaign.subject_id,
        campaign_id=assembled.campaign.id,
        blocking_executor=host.blocking_executor,
    )


async def _release(client: BoundCompletionClient) -> None:
    """Release a client the generation built and will not run — only the run would otherwise enter it."""
    async with client:
        pass


async def estimate_analysis_generation(
    host: EvalHost,
    campaign_id: str,
    scope_id: str,
    *,
    model: str | None,
    resolve_prompt: Callable[[], Awaitable[str]],
    out_of_run_cap_usd: float | None,
) -> AnalysisGenerationEstimate:
    """Price a generation's first call against the cap it would be held to, and make no call.

    The same assembly and the same pricing rule :func:`prepare_analysis_generation` refuses by, so an
    estimate reading ``would_start`` is the start's own answer.

    :param host: The host: where the campaign is read and the client is built.
    :ptype host: EvalHost
    :param campaign_id: The campaign.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param model: The generator model override, or ``None`` for the host's default.
    :ptype model: str | None
    :param resolve_prompt: Resolves the generator prompt.
    :ptype resolve_prompt: Callable[[], Awaitable[str]]
    :param out_of_run_cap_usd: The out-of-run cap the generation would be held to; ``None`` when the host
        enforces none.
    :ptype out_of_run_cap_usd: float | None
    :return: The estimate.
    :rtype: AnalysisGenerationEstimate
    :raises NotFoundError: No campaign with that id.
    :raises ValidationFailedError: The bundle has no resolvable runs, the preset does not exist, or the writer
        model is not one the host allows.
    :raises ValueError: The host supplies no completion clients.
    """
    assembled = await _assemble(host, campaign_id, scope_id, model=model, resolve_prompt=resolve_prompt)
    try:
        await refresh_prices(assembled.client)
        call = assembled.first_call
        ceiling = assembled.client.price_ceiling(
            system=call.system, user=call.user, response_format=call.response_format
        )
        refusal: str | None = None
        try:
            _generation_budget(host, assembled, out_of_run_cap_usd).quote(assembled.client, "analysis", [call])
        except ValidationFailedError as refused:
            refusal = str(refused)
        return AnalysisGenerationEstimate(
            campaign_id=campaign_id,
            generator_model=assembled.client.model_name,
            first_call_ceiling_usd=ceiling,
            max_calls=MAX_GENERATION_CALLS,
            cap_usd=out_of_run_cap_usd,
            would_start=refusal is None,
            refusal=refusal,
        )
    finally:
        await _release(assembled.client)


async def prepare_analysis_generation(
    host: EvalHost,
    campaign_id: str,
    scope_id: str,
    *,
    model: str | None,
    resolve_prompt: Callable[[], Awaitable[str]],
    out_of_run_cap_usd: float | None,
    attempt_id: str | None = None,
    on_assembled: Callable[[CampaignBundle, str], Awaitable[None]] | None = None,
    started_at: str | None = None,
) -> PreparedGeneration:
    """Check and build everything a generation needs before it spends anything, its first call priced and admitted.

    The refusals a generation documents for a missing campaign, an empty bundle, an unknown
    preset, a writer the host does not allow and a first call over the cap (or unpriceable under it) are all raised
    here, before any call. Assembling the bundle reads every member run's results, so a background start
    (:func:`~threetears.evals.ops.analysis_generate`) calls this inside its job, after recording the attempt, and
    ends the attempt refused with what this raised; :func:`estimate_analysis_generation` makes the same checks in
    its caller's request. A host that admits one generation per campaign at a time checks that BEFORE calling this,
    because preparing builds a client and resolves the prompt, work a refused second request need not pay for. Every
    store read runs on the host's blocking executor.

    :param host: The host: where the campaign's runs, results and insights are read, the vocabulary
        the bundle is assembled in, and the client factory the generator is built from.
    :ptype host: EvalHost
    :param campaign_id: The campaign to analyse.
    :ptype campaign_id: str
    :param scope_id: The scope the campaign, its member runs and everything generated from it live in.
    :ptype scope_id: str
    :param model: The generator model override, or None for the host's default.
    :ptype model: str | None
    :param resolve_prompt: Resolves the generator prompt this generation runs, refusing an unknown
        preset with ``ValidationFailedError``. Awaited only once the bundle has evidence.
    :ptype resolve_prompt: Callable[[], Awaitable[str]]
    :param out_of_run_cap_usd: The most the generation's calls may together be priced at — the host's
        out-of-run cap — or ``None`` when the host enforces none (each call is still priced where the
        client can say, and ledgered). Required, so every caller decides what bounds the spend.
    :ptype out_of_run_cap_usd: float | None
    :param attempt_id: The id of the attempt this generation runs as: the one a background start recorded before
        its job started. ``None`` mints one.
    :ptype attempt_id: str | None
    :param on_assembled: Awaited once the bundle is assembled and the client built and held to the host's allowed
        writers, before the first call is priced, with the bundle and the writer the client resolved — so a recorded
        attempt can name both, even when the price then refuses it. A raise releases the client and propagates.
    :ptype on_assembled: Callable[[CampaignBundle, str], Awaitable[None]] | None
    :param started_at: When the attempt was started, as its recorded start says, carried to the ended attempt;
        ``None`` stamps the start of the call.
    :ptype started_at: str | None
    :return: The prepared generation. It holds a built generator client, which the run enters and
        releases; one prepared and never run keeps its transport open until it is collected.
    :rtype: PreparedGeneration
    :raises NotFoundError: No campaign with that id.
    :raises ValidationFailedError: The bundle has no resolvable runs, the preset does not exist, the writer model is
        not one the host allows, the resolved prompt is the memo writer's seed (``EVAL_ANALYSIS_GEN_DEFAULT``: the
        remedy names the ``eval_report_plan`` prompt to register), the report input's view is over the planner's
        budget once every trim is made, or the first call cannot be priced under an enforced cap or is priced above
        it — the client it built released.
    :raises ValueError: The host supplies no completion clients.
    """
    assembled = await _assemble(host, campaign_id, scope_id, model=model, resolve_prompt=resolve_prompt)
    first = assembled.first_call
    try:
        # Built inside the guard: a raise here (a budget that cannot be derived) still releases the client.
        calls = BudgetedGenerator(_generation_budget(host, assembled, out_of_run_cap_usd), assembled.client)
        if on_assembled is not None:
            await on_assembled(assembled.bundle, assembled.client.model_name)
        await refresh_prices(assembled.client)
        calls.admit(first.system, first.user, first.response_format)
    except BaseException:
        # A refusal, a failed hook, or a cancellation while prices were refreshed: nothing will run this client.
        await _release(assembled.client)
        raise
    return PreparedGeneration(
        campaign_id=campaign_id,
        scope_id=scope_id,
        bundle=assembled.bundle,
        report_input=assembled.report_input,
        assembled_at=assembled.assembled_at,
        prompt=assembled.prompt,
        client=assembled.client,
        resolved_model=assembled.client.model_name,
        attempt_id=attempt_id or str(uuid.uuid7()),
        calls=calls,
        started_at=started_at,
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
    resolved writer and the bundle's fingerprint over it once it has assembled the bundle and built the client
    (:func:`assembled_analysis_attempt`), before its first call is priced, and :func:`run_analysis_generation` writes both again as it ends; an attempt refused while
    it prepared keeps what was known.

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


def assembled_analysis_attempt(
    started: EvalAnalysisAttempt, bundle: CampaignBundle, resolved_model: str
) -> EvalAnalysisAttempt:
    """``started`` with what its job's preparation resolved: the writer its client names and the bundle's fingerprint.

    What :func:`prepare_analysis_generation`'s ``on_assembled`` is handed, written over the running attempt before the
    first call is priced.

    :param started: The running attempt :func:`running_analysis_attempt` recorded.
    :ptype started: EvalAnalysisAttempt
    :param bundle: The campaign bundle the generation assembled.
    :ptype bundle: CampaignBundle
    :param resolved_model: The writer its client resolved.
    :ptype resolved_model: str
    :return: The attempt, still ``running``, to write over the stored one.
    :rtype: EvalAnalysisAttempt
    """
    return started.model_copy(
        update={
            "generator_model": resolved_model,
            "bundle_fingerprint": bundle.fingerprint(),
            "created_at": utc_now_iso(),
        }
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


async def run_analysis_generation(
    host: EvalHost,
    prepared: PreparedGeneration,
    *,
    prompt_id: str,
    max_output_tokens: int,
    deadline: AbstractAsyncContextManager[Any] | None = None,
    leases: JobLeases | None = None,
) -> tuple[EvalAnalysis, list[EvalInsight]]:
    """Make the paid planner calls for a prepared generation, then store and record the result.

    **The report planner writes the analysis.** The planner reads the generation's report input and writes the
    campaign's report plan in two calls, a structure and its prose, each repaired at most once
    (:func:`~threetears.evals.analysis.planned_analysis.generate_planned_analysis`). What is stored is a planned
    analysis (no memo; ``report_plan_id`` names its plan), its plan (:class:`~threetears.evals.kernel.campaign.EvalReportPlan`)
    and an insight per durable claim the plan states. The memo writer writes no new analysis.

    **Every attempt is recorded.** Once the generator is called, the attempt is stored as an
    :class:`~threetears.evals.kernel.campaign.EvalAnalysisAttempt` however it ends — stored,
    refused, failed or cancelled — with what it sent, what it cost and why it ended, read from a
    tally the generator writes as it goes.

    **A background generation's attempt is its job's record**, written ``running`` before it started
    (:func:`running_analysis_attempt`) and carrying the job's lease. Its ending is written over that record and
    never over one that has already ended: once another process has written it lost (its lease lapsed), the
    generation stores nothing more — not its ending, and not its analysis — and
    :class:`~threetears.evals.kernel.JobFenced` is raised. The attempt's ending is the commit: it is written last,
    conditionally, and when it finds the attempt already lost the analysis and the insights stored just before it
    are withdrawn, so a lost attempt never keeps an analysis.

    :param host: The host: where the analysis, its insights and the attempt record are written, the
        vocabulary the memo is checked in, and how a provider call that raised becomes the
        text the attempt keeps.
    :ptype host: EvalHost
    :param prepared: What :func:`prepare_analysis_generation` checked and built.
    :ptype prepared: PreparedGeneration
    :param prompt_id: The registry key the prompt was resolved under — provenance the generator
        cannot derive, because reading the registry is the host's.
    :ptype prompt_id: str
    :param max_output_tokens: The output cap the client was built with, for the pre-spend log line.
    :ptype max_output_tokens: int
    :param deadline: A background generation's wall-clock budget, not yet entered — the host's job timeout built for
        it (:class:`~threetears.evals.kernel.JobTimeoutFactory`). Entered here around the generator calls, so
        when it runs out the attempt is recorded ``timed_out`` (:func:`~threetears.evals.kernel.timed_out_detail`)
        and :class:`~threetears.evals.kernel.EvalJobTimeout` raised, as every job kind's budget reads; storing what
        the calls paid for is not cut short. ``None`` for a generation its caller bounds.
    :ptype deadline: AbstractAsyncContextManager[Any] | None
    :param leases: The leases a background generation's attempt is held under: once they have let go of it, the
        generation stores and records nothing more (:meth:`~threetears.evals.kernel.JobLeases.let_go`), even
        when the store would still take the write. ``None`` for a generation no process holds a lease for.
    :ptype leases: JobLeases | None
    :return: The stored analysis and the insights it minted; its plan is read with :func:`get_report_plan`.
    :rtype: tuple[EvalAnalysis, list[EvalInsight]]
    :raises ValidationFailedError: The planner's output failed the generation contract, or a call after the first (the
        prose call, or a repair round-trip) was refused by the generation's budget before it was sent.
    :raises JobFenced: The generation's attempt was written lost by another process, or this process let go of its
        lease, before its analysis was stored; nothing was stored.
    :raises EvalJobTimeout: ``deadline`` ran out while the generator was being called; the attempt reads ``timed_out``.
    :raises StorageError: The analysis, its plan or one of its insights could not be stored. An insight that fails
        leaves the attempt recorded ``stored`` all the same — the analysis is — so a background
        generation's poll reports the analysis stored while its insight ledger is incomplete; the
        background task catches that failure and logs it, and nothing else reports it.
    :raises ProviderRefusedError: The generator call raised and the host describer withheld its payload
        (a provider status error in its chain, or no describer able to say otherwise). The attempt is recorded
        ``failed`` and the failure logged, both through the host describer, before this is raised in its place. Any
        other failure is recorded ``failed`` and propagates as itself, so its status names no provider.
    """
    storage, failure_describer, executor = host.storage, host.failure_describer, host.blocking_executor
    campaign_id, scope_id, bundle = prepared.campaign_id, prepared.scope_id, prepared.bundle
    prompt, client, resolved_model = prepared.prompt, prepared.client, prepared.resolved_model

    # Visible Costs: disclose the SCALE of the call BEFORE making it — model,
    # input size, output cap. No dollar figure: what the call was priced at is on
    # its out-of-run ledger row, and what it cost is logged after, from the provenance.
    #
    # The input size is a ballpark over the exact first request the generator sends
    # (`first_request`, its one derivation). It was `chars/4` over a differently
    # rendered bundle, which logged 224,571 for a call billed 525,840 prompt tokens:
    # ids, numbers and punctuation tokenize far denser than prose, and it left the
    # system prompt's additions and the response schema out.
    #
    # The prose says "the generator model", not "opus". It said "opus call starting"
    # while interpolating `resolved_model` two lines down, so a run with a `model=`
    # override — a documented, supported way to A/B a cheaper generator — printed
    # `model=openai/gpt-5-mini (opus call starting)`. A log key naming the value it
    # carries is the observability norm; a key whose PROSE contradicts its own
    # interpolation is the same defect with an extra step.
    log.info(
        "eval.generate_analysis campaign=%s scope=%s model=%s runs=%d "
        "approx_input_tokens=%d (%s) max_output_tokens=%d (generator call starting — $ billed on completion)",
        campaign_id,
        scope_id,
        resolved_model,
        len(bundle.run_ids),
        _approx_input_tokens(*first_plan_request(prepared.report_input, prompt, host.profile)),
        APPROX_INPUT_TOKENS_BASIS,
        max_output_tokens,
    )

    # Written by the generator as each call is sent and returns, so the record below is exact
    # however the generation ends — including a cancellation, which returns nothing and raises
    # nothing the generator could have annotated.
    tally = GenerationTally()
    started_at = prepared.started_at or utc_now_iso()

    async def record(
        outcome: AttemptOutcome,
        *,
        error: str | None = None,
        analysis_id: str | None = None,
        refused_by: EvalServiceError | None = None,
    ) -> bool:
        # On the host's blocking executor, and waited for through a cancellation: the record is how the
        # attempt ended, so a cancel arriving while it is written must not leave it half-known. Returns
        # whether a cancellation arrived meanwhile, which the caller delivers once it has finished. An engine
        # error the ending raises to its caller is kept on the record as data (its kind and status).
        return (await recorded(outcome, error=error, analysis_id=analysis_id, refused_by=refused_by))[1]

    async def recorded(
        outcome: AttemptOutcome,
        *,
        error: str | None = None,
        analysis_id: str | None = None,
        refused_by: EvalServiceError | None = None,
    ) -> tuple[_AttemptRecorded, bool]:
        # :func:`record`, saying too whether the ending landed, or was refused because the attempt had ended.
        return await _offloaded_result(
            executor,
            _record_analysis_attempt,
            storage,
            campaign_id=campaign_id,
            scope_id=scope_id,
            outcome=outcome,
            tally=tally,
            generator_model=resolved_model,
            prompt_id=prompt_id,
            bundle_fingerprint=bundle.fingerprint(),
            started_at=started_at,
            error=error,
            analysis_id=analysis_id,
            attempt_id=prepared.attempt_id,
            leases=leases,
            refusal=None if refused_by is None else job_refusal(refused_by),
        )

    calls = prepared.calls
    try:
        # The client owns an httpx pool and this generation is its whole
        # lifetime; `async with` releases it on the raising path too, which
        # is a path this call reaches often enough to have its own handler.
        async with client, deadline if deadline is not None else contextlib.nullcontext():
            analysis, plan, insights = await generate_planned_analysis(
                bundle,
                prepared.report_input,
                prompt=prompt,
                model=resolved_model,
                client=calls,
                prompt_id=prompt_id,
                bundle_assembled_at=prepared.assembled_at,
                tally=tally,
                admit=calls.admit,
                profile=host.profile,
            )
    except EvalJobTimeout as timed_out:
        # Its own budget ran out mid-flight: not someone deciding to stop it, so not `cancelled`.
        await record("timed_out", error=timed_out_detail(timed_out))
        raise
    except GenerationError as e:
        # A call's reply refused twice is `refused`; every other generation-contract failure (a
        # call cut short, a precondition refused before any call) is `failed`.
        # A generation-contract violation is caller-restatable (retry, or edit
        # the prompt), not a server fault — surface it as a 422, like the
        # proposer surfaces its own draft-validation failures.
        contract = ValidationFailedError(
            f"analysis generation failed: {e}", remedy="start the generation again, or edit the prompt it reads"
        )
        await record("refused" if isinstance(e, SoundnessRefusal) else "failed", error=str(e), refused_by=contract)
        raise contract from e
    except asyncio.CancelledError:
        # A caller's budget or disconnect, or a shutdown cancelling a background generation.
        # Recorded synchronously, before the cancellation propagates, because this is the
        # ending no exception message ever carries to a caller.
        repairing = ", so a call after the first was running" if tally.calls > 1 else ""
        await record(
            "cancelled",
            error=(
                "cancelled mid-flight by the caller's budget or disconnect, or a shutdown, "
                f"after {tally.calls} provider call(s){repairing}"
            ),
        )
        raise
    except ValidationFailedError as e:
        # The budget refused a call before it was sent — only a call after the first can reach this (the prose
        # call, or a repair round-trip), since the first was admitted when the generation was prepared. Nothing was
        # sent for it, so the tally counts only what was; what the earlier calls cost is on the record.
        unrepaired = ValidationFailedError(
            f"analysis generation failed: call {tally.calls + 1} of the plan was refused before it was sent ({e}); "
            f"${format_number(tally.token_cost)} was billed on the {tally.calls} call(s) before it and nothing was "
            "stored",
            remedy="start the generation again, under a cap with room for every call a plan can make",
        )
        await record(
            "failed",
            error=f"call {tally.calls + 1} of the plan was refused before it was sent: {e}",
            refused_by=unrepaired,
        )
        raise unrepaired from e
    # prawduct:ok-broad-except — eval names no provider exception type; the host describer classifies what the call raised
    except Exception as e:
        # The exception's own text is not stored on the record: a provider status error stringifies
        # as its response envelope, account id included, and so does the last line of any traceback
        # that chains it. The host's describer says what is safe to keep. When it withheld the
        # payload, the failure is logged here once and raised on as an eval error rather than
        # propagated, because an exception escaping a REST route reaches the web server, which logs
        # its whole traceback; otherwise it propagates as itself and whoever catches it logs it.
        failure = describe_failure(failure_describer, e, logger=log, where="eval.generate_analysis")
        # The provider's refusal, when its payload was withheld: what the caller is raised, and what the record keeps.
        provider = (
            None
            if traceback_is_safe(failure)
            else ProviderRefusedError(f"analysis generation failed: {failure.description}")
        )
        await record("failed", error=failure.description, refused_by=provider)
        if provider is None:
            # Nothing a traceback would print is withheld, so no provider status error is in the
            # chain: most often a defect in the generation itself. It propagates as itself, so its
            # status and code name no provider, and whatever logs it keeps the diagnosing traceback.
            raise
        log_provider_failure(
            log,
            failure,
            e,
            "eval.generate_analysis generation %s of campaign %s: the generator call failed",
            prepared.attempt_id,
            campaign_id,
        )
        raise provider from None

    # A failed persist of the just-paid artifact is re-raised with what it cost, so
    # the operator reading the 503 knows the spend happened and nothing is stored.
    # Every write runs on the host's blocking executor and is waited for through a cancellation, so a
    # cancel arriving while the paid-for analysis is stored cannot leave it stored with no record saying
    # so; the cancellation is delivered once the analysis, its record and its insights are written.
    #
    # **The attempt's ending is the commit.** Checking the attempt is still held and then storing is
    # check-then-act: another process can write the attempt lost in between (its lease lapsed while this one
    # stalled). So the check below only refuses early; what decides is the attempt's own ending, written last
    # and conditionally (``write_held_record``, ``if_match``). When that write finds the attempt already
    # ended — lost — what this generation stored is withdrawn (:func:`_withdraw_generation`): the analysis
    # deleted, each insight it wrote deleted or put back as it stood. A lost attempt never keeps an analysis.
    # The ending stays last because a poll reading ``completed`` must find the ledger already written.
    await _offloaded(executor, _still_held, storage, prepared.attempt_id, prepared.scope_id, leases)
    try:
        # The plan first: an analysis is never stored naming a plan that is not. A plan whose analysis then fails to
        # store is withdrawn below, so neither is left alone.
        cancelled = await _offloaded(executor, storage.save_report_plan, plan)
        cancelled = await _offloaded(executor, storage.save_analysis, analysis) or cancelled
    except StorageError as e:
        unstored = StorageError(
            f"failed to persist generated analysis '{analysis.id}' and its plan for campaign '{campaign_id}' "
            f"(the generation cost ${format_number(analysis.generation.token_cost)} was already incurred)"
        )
        try:
            await _offloaded(executor, storage.delete_report_plan, plan.id, scope_id)
        except StorageError:
            # NOSILENT: logged for the operator; the failure to store is what the caller is told, below, and a plan
            # left behind names an analysis that does not resolve and is swept with the derived documents
            log.exception(
                "eval.generate_analysis plan %s of an analysis that failed to store could not be withdrawn",
                plan.id,
            )
        await record("failed", error=f"the generated analysis could not be stored: {e}", refused_by=unstored)
        raise unstored from e
    insights, cancelled = await _commit_insights_then_ending(
        storage,
        executor,
        analysis=analysis,
        minted=insights,
        attempt_id=prepared.attempt_id,
        scope_id=scope_id,
        record_stored=lambda: recorded("stored", analysis_id=analysis.id),
        left_behind=f"analysis '{analysis.id}', with its plan '{plan.id}',",
        cancelled=cancelled,
    )

    log.info(
        "eval.generate_analysis stored campaign=%s analysis=%s plan=%s insights=%d cost=$%.4f model=%s",
        campaign_id,
        analysis.id,
        plan.id,
        len(insights),
        analysis.generation.token_cost,
        analysis.generation.generator_model,
    )
    if cancelled:
        raise asyncio.CancelledError
    return analysis, insights


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

    The commit a planned generation makes (:func:`run_analysis_generation`), for an analysis the job built from a stored
    report document. With ``stored``, the attempt must still be held: the analysis is written, then each insight (a
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

    The commit both generation paths share once their analysis is stored (a planned one's, beside its plan, by
    :func:`run_analysis_generation`; a report document's by :func:`record_written_generation`). A restated claim
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
    """Store how one generation attempt ended, from the tally the generator kept.

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
    :param tally: What the generator sent, spent and was refused.
    :ptype tally: GenerationTally
    :param generator_model: The model the generator client was built for.
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


def list_analyses(storage: AnalysisStore, campaign_id: str, scope_id: str) -> list[EvalAnalysis]:
    """List every analysis attached to a campaign in a scope, newest first."""
    return storage.list_analyses_by_campaign(campaign_id, scope_id)


def list_analysis_attempts(storage: AnalysisStore, campaign_id: str, scope_id: str) -> list[EvalAnalysisAttempt]:
    """List every generation attempt recorded against a campaign in a scope, newest first — stored and failed alike."""
    return storage.list_analysis_attempts_by_campaign(campaign_id, scope_id)


# ---------------------------------------------------------------------------
# Bundle inspection — the free read of the input a paid generation reads.
#
# The bundle is where the deterministic judgements live: which subtraction is
# unsound, which swing the containment default withholds, which divergences the
# cap dropped, what the campaign's design actually derived to. All of it used to
# reach a reader only if the generator chose to write it into prose, which made a
# rule the generator could ignore out of a rule that was moved out of the prompt
# precisely so it could not be. Both entry points below RE-ASSEMBLE rather than
# read a stored copy: the bundle's fingerprint is the invariant the prompt-tuning
# A/B loop rests on, and a second persisted copy is a second thing to keep true.
# ---------------------------------------------------------------------------


def _assembled_campaign_bundle(
    campaign: EvalCampaign, *, storage: EvalStorage, profile: HostProfile, insights_as_of: str | None = None
) -> CampaignBundle:
    """Assemble every aspect's bundle, the campaign's arms and code's readings into the campaign bundle. Blocking.

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
    bundles = [
        assemble_context_bundle(
            campaign, aspect=aspect.label, storage=storage, profile=profile, insights_as_of=insights_as_of
        )
        for aspect in campaign.aspects
    ]
    arms = campaign_arms(storage, campaign, profile=profile)
    readings = campaign_readings(campaign, {bundle.aspect: bundle for bundle in bundles}, arms=arms)
    return build_campaign_bundle(campaign, bundles, arms=arms, readings=readings)


def assemble_campaign_bundle(host: EvalHost, campaign_id: str, scope_id: str) -> CampaignBundle:
    """Assemble the campaign bundle a generation launched right now would read, without generating.

    Every aspect's evidence, the campaign's arms and floors, and code's readings across them — the writer's whole
    input (:func:`~threetears.evals.analysis.campaign_bundle.build_campaign_user_message` renders what it is sent).
    Costs nothing: storage reads and no model call. Not refused when it resolves no runs: an empty bundle is the
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
    plan: EvalReportPlan | None = None,
    fingerprints: Mapping[str, str] | None = None,
    trials: TrialDetail = "table",
    unread_analysis_id: str | None = None,
    unread_report_doc_id: str | None = None,
) -> CampaignReport:
    """Lay a campaign bundle out as the campaign's page: code's readings, the memo or the plan where there is one, the
    call.

    Each aspect's fold is that aspect's code-only page over its bundle, its arms named by the campaign's labels; the
    memo, when there is one, is laid out once, on the campaign's page, never again inside a fold. A planned analysis is
    laid out as the page its plan draws (:func:`~threetears.evals.analysis.report.plan_page.build.plan_page`), the
    code-only page's folds its appendix.

    :param campaign: The campaign, declaring what the bundle was assembled against.
    :ptype campaign: EvalCampaign
    :param bundle: Its evidence: every aspect's bundle and code's readings across them.
    :ptype bundle: CampaignBundle
    :param assembled_at: When the bundle was assembled (ISO-8601), which dates each aspect's page.
    :ptype assembled_at: str
    :param analysis: The stored analysis whose memo or plan the page lays out; None for a code-only page.
    :ptype analysis: EvalAnalysis | None
    :param plan: The plan a planned ``analysis`` was written as; None for a memo or a code-only page.
    :ptype plan: EvalReportPlan | None
    :param fingerprints: Each aspect's bundle fingerprint in the current version, by label, where it was just checked
        against the bundle; an aspect not named is hashed.
    :ptype fingerprints: Mapping[str, str] | None
    :param trials: How much of each failed or lost trial each fold lays out (:data:`TrialDetail`).
    :ptype trials: TrialDetail
    :param unread_analysis_id: On a code-only page, the analysis this version no longer reads that it stands in for.
    :ptype unread_analysis_id: str | None
    :param unread_report_doc_id: On a code-only page, the report document read by its id that it stands in for.
    :ptype unread_report_doc_id: str | None
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
    memo = analysis.document if analysis is not None else None
    page = build_campaign_report(
        campaign,
        readings,
        reports,
        memo,
        call=campaign.call,
        recommended_variant_key=analysis.recommended_variant_key if analysis is not None else None,
        resolutions=analysis.resolutions if analysis is not None else (),
        trial_ids=bundle.trial_ids(),
        report_doc_id=analysis.report_doc_id if analysis is not None else None,
        unread_analysis_id=unread_analysis_id,
        unread_report_doc_id=unread_report_doc_id,
        arms=bundle.arms,
    )
    if analysis is None:
        return page
    if plan is None:
        return page.model_copy(update={"analysis_id": analysis.id})
    return _planned_page(page, bundle, plan, analysis_id=analysis.id)


def _planned_page(
    page: CampaignReport, bundle: CampaignBundle, plan: EvalReportPlan, *, analysis_id: str
) -> CampaignReport:
    """The page a planned analysis's report plan draws, over the report input built from its frozen bundle, in place
    of the code-only page's blocks; the code-only page's folds stay, as the planned page's appendix.

    :raises RefusedInput: A figure key or a chart in the plan no longer resolves against the report input built from
        the frozen bundle.
    """
    made = MADE_TEMPLATE.format(
        model=plan.generation.generator_model, date=plan.generation.generated_at[:10], analysis=analysis_id
    )
    try:
        planned = plan_page(report_plan_of(plan), build_report_input(bundle), made=made)
    except UnresolvableReference as unresolved:
        raise RefusedInput(
            f"analysis {analysis_id}'s plan names a figure or a chart its frozen evidence no longer holds: {unresolved}",
            remedy="generate the analysis again (analysis_generate), which plans over the campaign's evidence as it stands",
        ) from unresolved
    return CampaignReport.model_validate(
        {**dict(page), "basis": "analysis", "analysis_id": analysis_id, "blocks": [], "plan_page": planned}
    )


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
    plan: EvalReportPlan | None = None,
    bundle: CampaignBundle | None = None,
) -> CampaignReport:
    """A stored analysis laid out as its campaign's page, over everything frozen with it (``bundle``, where the caller
    has put it back together already): the page its plan draws, for a planned one."""
    bundle = bundle if bundle is not None else analysis_campaign_bundle(analysis)
    # Putting the bundle back together refused it unless each aspect's entry in the frame is that aspect's bundle
    # fingerprint in the entry's version, so an entry in the current version is the one each fold's source names.
    entries = analysis.campaign_bundle.get("aspects")
    fingerprints = {
        str(entry["aspect"]): str(entry["fingerprint"])
        for entry in (entries if isinstance(entries, list) else [])
        if isinstance(entry, dict) and str(entry.get("fingerprint", "")).startswith(FINGERPRINT_PREFIX)
    }
    return _campaign_page(
        _as_analysed(analysis, campaign),
        bundle,
        assembled_at=analysis.generation.bundle_assembled_at,
        analysis=analysis,
        plan=plan,
        fingerprints=fingerprints,
        trials=trials,
    )


def get_report_plan(storage: AnalysisStore, analysis: EvalAnalysis) -> EvalReportPlan | None:
    """The report plan a stored analysis was written as; None for one the memo writer wrote.

    :param storage: Where the plan is read.
    :ptype storage: AnalysisStore
    :param analysis: The analysis.
    :ptype analysis: EvalAnalysis
    :return: The plan, or None for a memo.
    :rtype: EvalReportPlan | None
    :raises NotFoundError: The analysis is planned and its plan does not resolve (deleted outside the analysis's
        delete).
    :raises RefusedInput: The stored plan no longer validates as the plan this build reads (its shape moved inside
        the regenerable version): the analysis needs generating again, which the remedy says.
    """
    if analysis.report_plan_id is None:
        return None
    try:
        plan = storage.load_report_plan(analysis.report_plan_id, analysis.scope_id)
    except ValidationError as unreadable:
        first = unreadable.errors()[0]
        where = ".".join(str(step) for step in first["loc"]) or "the document"
        raise RefusedInput(
            f"analysis {analysis.id!r} was written as a report plan this build cannot read "
            f"({unreadable.error_count()} field error(s); first at {where}: {first['msg']})",
            remedy="generate the analysis again (analysis_generate); a report plan is regenerated, never migrated",
        ) from unreadable
    if plan is None:
        raise NotFoundError(
            "report plan",
            analysis.report_plan_id,
            remedy="generate the analysis again (analysis_generate); its plan was deleted apart from it",
        )
    return plan


def analysis_report(
    storage: AnalysisStore, analysis_id: str, scope_id: str, *, trials: TrialDetail = "table"
) -> CampaignReport:
    """Read one stored analysis as its campaign's page — the one page every door renders.

    Laid out over the bundles and code's readings frozen with the analysis at generation, so it reads as it was
    written whatever has been archived or re-run since: the campaign as it was declared then, every aspect's fold over
    its frozen bundle, the memo once on the first screen. The person's call is the campaign's as recorded now. Derived
    on every read and never written back, so a stored analysis stays pivotable.

    :param storage: Where the analysis is read.
    :ptype storage: AnalysisStore
    :param analysis_id: The analysis to report.
    :ptype analysis_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param trials: How much of each failed or lost trial the page lays out (:data:`TrialDetail`): by default their
        table alone, each row naming its stored result; ``all`` adds each one's full record.
    :ptype trials: TrialDetail
    :return: The page, ``basis="analysis"``; :meth:`~threetears.evals.analysis.report.CampaignReport.write` puts it
        in a directory, and :func:`~threetears.evals.analysis.report.serialize_report` serializes it.
    :rtype: CampaignReport
    :raises NotFoundError: No analysis with that id in the scope.
    :raises RefusedInput: The analysis was written over a bundle of another shape, or as a report plan this build no
        longer reads (:func:`get_report_plan`); either way the remedy is to generate it again.
    """
    analysis = get_analysis(storage, analysis_id, scope_id)
    return _analysis_page(
        analysis,
        storage.load_campaign(analysis.campaign_id, scope_id),
        trials=trials,
        plan=get_report_plan(storage, analysis),
    )


def campaign_report(
    host: EvalHost, campaign_id: str, scope_id: str, *, trials: TrialDetail = "table"
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
    - a report plan's (``plan``, the report planner's): every one is written in a shape this version no longer reads,
      so its page is the code-only page with the one regenerate line on its first screen
      (:data:`~threetears.evals.analysis.report.campaign.memo_blocks.NO_LONGER_READS`,
      :attr:`~threetears.evals.analysis.report.CampaignReport.unread_analysis_id`). No planner page is reached;
    - a memo's: laid out over what was frozen with it, as :func:`analysis_report` lays it out; where it was frozen
      over a bundle of another shape this version no longer reads, as a plan's is (the code-only page, the regenerate
      line saying the bundle's shape).

    A report document this version no longer reads — its record does not validate, its snapshot is gone, holds no
    bundle or no longer reads, or the campaign no longer lays out over it — is laid out as a plan is: the code-only page
    with the one regenerate line. Neither is ever refused. Either way the page is the campaign's
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
    :return: The page; its ``basis`` says which it is.
    :rtype: CampaignReport
    :raises NotFoundError: No campaign with that id in the scope.
    :raises RefusedInput: A memo's page refuses what it is laid out over: an aspect with no page, a page keyed by a label
        the campaign does not hold, readings of another campaign. A bundle of another shape is never refused.
    """
    campaign = _load_campaign(host.storage, campaign_id, scope_id)
    reported = host.storage.load_reported_analysis(campaign_id, scope_id)
    if reported is None:
        return _code_only_page(host, campaign, trials=trials)
    if reported.report_doc_id is not None:
        try:
            read = read_report_doc(host.storage, reported.report_doc_id, scope_id)
        except NotFoundError as missing:
            return _stands_in(host, campaign, missing.message, trials=trials, unread_analysis_id=reported.id)
        except RefusedInput as unread:
            return _stands_in(host, campaign, unread.fault, trials=trials, unread_analysis_id=reported.id)
        return _doc_page(host, campaign, read, analysis_id=reported.id, trials=trials)
    if reported.report_plan_id is not None:
        why = f"analysis {reported.id} was written by the report planner, as report-plan/1"
        return _stands_in(host, campaign, why, trials=trials, unread_analysis_id=reported.id)
    try:
        frozen = analysis_campaign_bundle(reported)
    except RefusedInput as unread:
        # A memo frozen over a bundle of another shape is read as a plan is: the code-only page, led by the one
        # regenerate line, and why (the shape the bundle was frozen at) after it. Only putting the bundle back
        # together is caught: a page that refuses what it is laid out over (an aspect with no page, readings of
        # another campaign) is a fault, and reaches the caller.
        return _stands_in(host, campaign, unread.fault, trials=trials, unread_analysis_id=reported.id)
    return _analysis_page(reported, campaign, trials=trials, bundle=frozen)


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
    host: EvalHost, campaign: EvalCampaign, read: StoredReportPage, *, analysis_id: str | None, trials: TrialDetail
) -> CampaignReport:
    """A stored report document's page, its readings and folds off its snapshot's bundle laid out over the campaign as
    that bundle declared it (what the document was written against, which its analysis's ``design_snapshot`` copies);
    the code-only page of ``campaign`` with the one regenerate line where the snapshot holds no bundle or its bundle
    no longer lays out. With ``trials="all"`` the folds follow the document, each failed or lost trial's record in
    them, as the code-only page draws them."""
    snapshot = read.snapshot
    try:
        if snapshot.bundle is None or snapshot.bundle_assembled_at is None:
            raise RefusedInput(
                f"report document {read.record.id}'s snapshot {snapshot.id} holds no bundle", remedy=NO_LONGER_READS
            )
        bundle = CampaignBundle.from_dict(snapshot.bundle, what=f"report snapshot {snapshot.id}")
        declared = _with_declaration(campaign, bundle.declaration)
        page = _campaign_page(declared, bundle, assembled_at=snapshot.bundle_assembled_at, trials=trials)
    except RefusedInput as unread:
        if analysis_id is None:
            return _stands_in(host, campaign, unread.fault, trials=trials, unread_report_doc_id=read.record.id)
        return _stands_in(host, campaign, unread.fault, trials=trials, unread_analysis_id=analysis_id)
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
) -> CampaignReport:
    """The code-only page standing in for an analysis or a report document this version no longer reads: its first
    screen says so in the one regenerate line, and its ``needs_regeneration`` is that line with why."""
    named = unread_analysis_id or unread_report_doc_id
    log.info("campaign %s's report %s reads code-only: %s", campaign.id, named, why)
    page = _code_only_page(
        host,
        campaign,
        trials=trials,
        unread_analysis_id=unread_analysis_id,
        unread_report_doc_id=unread_report_doc_id,
    )
    return CampaignReport.model_validate({**dict(page), "needs_regeneration": f"{NO_LONGER_READS} ({why})"})


def _code_only_page(
    host: EvalHost,
    campaign: EvalCampaign,
    *,
    trials: TrialDetail = "table",
    unread_analysis_id: str | None = None,
    unread_report_doc_id: str | None = None,
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
        another campaign's, archived, or not the campaign's report; a broken floor or missed bar of the arm is not overridden, or an override names one the
        arm did not break or miss; or a floor or bar is named twice.
    :raises ConflictError: The campaign changed while its readings were read.
    """
    campaign = _load_campaign(host.storage, campaign_id, scope_id)
    if analysis_id is None:
        bundle = _assembled_campaign_bundle(campaign, storage=host.storage, profile=host.profile)
    else:
        bundle = analysis_campaign_bundle(_reported_analysis(host.storage, campaign_id, scope_id, analysis_id))
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


# ---------------------------------------------------------------------------
# Reporter cases — the cases that measure the generator itself
# ---------------------------------------------------------------------------


def reporter_case_bank(storage: AnalysisStore, template_id: str, scope_id: str) -> ReporterCaseBank:
    """Derive which of a reporter template's stored cases is live, refusing when that is undecidable.

    How every consumer of the case bank reads it — a reporter launch and its price, the freeze
    (:func:`freeze_reporter_case`) and the calibration read (:func:`reporter_calibration`) — over
    :func:`~threetears.evals.analysis.reporter_bank.decidable_reporter_case_bank`, which a
    retirement's restore (:func:`threetears.evals.analysis.reporter_curation.set_reporter_case_archived`) reads too.

    :param storage: Where the template's cases are read.
    :ptype storage: AnalysisStore
    :param template_id: The reporter template.
    :ptype template_id: str
    :param scope_id: The partition its cases live in.
    :ptype scope_id: str
    :return: The bank.
    :rtype: ReporterCaseBank
    :raises ValidationFailedError: A stored case of the template carries a reporter case this build
        cannot read.
    """
    return decidable_reporter_case_bank(
        storage.query_test_cases(scope_id, template_id=template_id), template_id=template_id
    )


def _label_set(labels: Sequence[ReporterLabel], *, criterion: bool, excerpt: bool = False) -> list[str]:
    """Labels as what makes two freezes the same case compares them: their verdicts, with or without the criterion
    each was written against, and never who gave or relayed them (``reader``, ``collected_by``) — provenance, so a
    re-freeze of the same verdicts names the stored case, whose own record of them stands.

    The memo passage each points at (``memo_excerpt``) is left out too unless ``excerpt`` asks for it: it is no part
    of the verdict, but a freeze whose excerpts differ from the stored case's would otherwise report a case that does
    not hold them, so the freeze compares them apart (:func:`freeze_reporter_case`)."""
    dropped: dict[str, Any] = {"reader": None, "collected_by": None}
    if not excerpt:
        dropped["memo_excerpt"] = None
    if not criterion:
        dropped["criterion"] = None
    return sorted(label.model_copy(update=dropped).model_dump_json() for label in labels)


def _whitespace_collapsed(text: str) -> str:
    """``text`` with every run of whitespace one space: how an excerpt and the memo it quotes are compared."""
    return re.sub(r"\s+", " ", text)


def _excerpts(labels: Sequence[ReporterLabel]) -> str:
    """The memo passages a case's labels quote, by dimension, as a refusal names them."""
    quoted = [f"{label.dimension}: {label.memo_excerpt!r}" for label in labels if label.memo_excerpt is not None]
    return "; ".join(quoted) or "none"


def _refuse_a_label_of_another_form(labels: Sequence[ReporterLabel], *, template_id: str) -> None:
    """Refuse a label whose form is not its dimension's scale, as the criterion the freeze stamped states it.

    A pass/fail dimension takes a ``verdict``, an ordinal one a ``direction``: a label in the other form could never
    be compared with the judge's score (:func:`~threetears.evals.analysis.reporter_bank.label_agrees`), and no
    mapping between the two is ratified. Applied at the freeze only, so a label stored before verdicts existed keeps
    reading.

    :raises ValidationFailedError: A label's form is not its stamped criterion's scale.
    """
    wrong = [
        f"{label.dimension} ({label.criterion.scale}) carries a {'verdict' if label.direction is None else 'direction'}"
        for label in labels
        if label.criterion is not None and (label.verdict is None) == (label.criterion.scale == "pass_fail")
    ]
    if wrong:
        raise ValidationFailedError(
            f"a label's form follows its dimension's scale on template {template_id!r}: a pass_fail dimension takes a "
            f"verdict and an ordinal one a direction, and {'; '.join(wrong)}",
            remedy="send a verdict (pass|fail) on each pass_fail dimension and a direction on each ordinal one",
        )


def _refuse_an_excerpt_not_in_the_memo(labels: Sequence[ReporterLabel], memo: str) -> None:
    """Refuse a label whose ``memo_excerpt`` is not a passage of the recorded memo the case pins.

    Compared after collapsing each run of whitespace to one space, in the excerpt and the memo alike, and trimming
    the excerpt's ends, and nothing else: a surface prints the memo's text whitespace-collapsed, so a passage copied
    from one matches, a line break carried at either end included, and any other difference is a passage the memo
    does not hold.

    :raises ValidationFailedError: An excerpt is not found in the memo.
    """
    collapsed = _whitespace_collapsed(memo)
    if missing := [
        label.dimension
        for label in labels
        if label.memo_excerpt is not None and _whitespace_collapsed(label.memo_excerpt).strip() not in collapsed
    ]:
        raise ValidationFailedError(
            f"the memo_excerpt of the label(s) on {', '.join(missing)} is not a passage of the recorded memo, compared "
            "with each run of whitespace collapsed to one space",
            remedy="copy each excerpt verbatim from the recorded memo as the judge reads it, or omit it",
        )


class _PlanCase(NamedTuple):
    """What a plan reporter case freezes of a planned analysis, read once.

    :ivar stored: The analysis's stored plan.
    :ivar bundle: The campaign bundle frozen with the analysis: what its planner planned over.
    :ivar report_input: The report input built from it, which the plan's keys resolve against.
    :ivar digest: That input's digest.
    :ivar text: The plan as the judge reads it (:func:`render_plan_for_judge`).
    :ivar message: What the planner read of that input (:func:`plan_user_message`).
    :ivar limits: Where the frozen input is not the one the plan was written against, as sentences.
    """

    stored: EvalReportPlan
    bundle: CampaignBundle
    report_input: ReportInput
    digest: str
    text: str
    message: str
    limits: list[str]


def _plan_case(storage: AnalysisStore, analysis: EvalAnalysis) -> _PlanCase:
    """Read what a plan reporter case freezes of a planned analysis.

    **The bundle is the one frozen with the analysis, never a re-assembly.** A plan's prose holds figure keys, not
    numbers, so what a reader of it sees is the keys written out against the report input built from that bundle, as
    the campaign's page writes them (:func:`_planned_page`). A re-assembly could move every number under unchanged
    prose, so the judged text would not be the one the planner wrote or a reader read.

    :raises NotFoundError: The analysis's plan does not resolve.
    :raises ValidationFailedError: A figure key of the plan does not resolve against the input built from its frozen
        bundle, or that input's view is over the planner's budget: the plan cannot be judged as written.
    """
    stored = get_report_plan(storage, analysis)
    if stored is None:
        raise ValueError(f"analysis {analysis.id} was read as planned and names no report plan")
    bundle = analysis_campaign_bundle(analysis)
    built = build_report_input(bundle)
    remedy = "generate the analysis again (analysis_generate), and freeze the new one"
    try:
        text = render_plan_for_judge(report_plan_of(stored), built)
        message = plan_user_message(built)
    except UnresolvableReference as unresolved:
        raise ValidationFailedError(
            f"analysis {analysis.id}'s plan names a figure its frozen evidence no longer holds, so it cannot be judged "
            f"as written: {unresolved}",
            remedy=remedy,
        ) from unresolved
    except RefusedInput as refused:
        raise ValidationFailedError(
            f"what analysis {analysis.id}'s planner read cannot be rendered again: {refused.fault}", remedy=remedy
        ) from refused
    digest = report_input_digest(built)
    limits: list[str] = []
    if digest != stored.report_input_digest:
        limits.append(
            f"The report input this build makes from the analysis's frozen bundle (digest {digest}) is not the one "
            f"the plan was written against (recorded {stored.report_input_digest}): how a report input is built "
            "changed since. The plan's figures are written out, and the judge's evidence rendered, from this build's."
        )
    return _PlanCase(stored, bundle, built, digest, text, message, limits)


def freeze_reporter_case(
    host: EvalHost,
    *,
    template_id: str,
    campaign_id: str,
    scope_id: str,
    analysis_id: str | None = None,
    labels: Sequence[Any] = (),
    supersedes: Sequence[str] = (),
    load_template: Callable[[str], EvalTemplate],
) -> EvalTestCase:
    """Freeze a campaign's bundle into a reporter case of ``template_id``.

    The case freezes the campaign bundle whole — what the writer of its one memo reads. A memo case (or one pinning
    no analysis) freezes a campaign of one aspect only: a freeze of a campaign of several is refused, naming the
    remedy.

    **An analysis the report planner wrote freezes a plan case** (``kind_version`` 2), of a campaign of any number of
    aspects: the bundle frozen with that analysis (never a re-assembly, since the plan's prose holds keys whose
    numbers a re-assembly could move), the report input built from it and its digest, the plan in key form and the
    planner's prompt version. Its ``recorded_memo`` is the plan as the judge reads it
    (:func:`~threetears.evals.analysis.reporter_kind.render_plan_for_judge`), and its ``writer_message`` what the
    planner read of that input (:func:`~threetears.evals.analysis.planner.plan_user_message`), checked against the
    digest the plan's generation recorded. Every other rule below holds of it as of a memo case. A report input built
    today that is not the one the plan was written against is frozen anyway and stated in ``limits``.

    A case is reproducible only if its evidence stops moving, and a bundle is otherwise
    re-assembled on demand from a store and a schema that both move. So the case carries the
    bundle itself, its fingerprint, the instant it was assembled, and — when ``analysis_id``
    names one — the memo that campaign actually got, which the as-recorded candidate replays.

    **What the labels were written against is pinned too.** With a memo, the case freezes the
    message that memo is judged against as TEXT (``ReporterCase.writer_message``), so a later
    change to how a bundle is rendered cannot move the evidence under its labels. It is rendered
    here from the bundle this freeze pins, since a stored analysis keeps no copy of the message
    its writer was sent. It does keep a digest of it (``generation.user_message_digest``), which
    the case copies and the freeze checks the rendered text against: a message that differs is
    frozen anyway and stated in ``limits``, as a bundle that does not reproduce is
    (:func:`~threetears.evals.analysis.reporter_kind.writer_message_check`). Each label is stamped with the criterion
    the template states for its dimension (``ReporterLabel.criterion``), so a reworded rubric shows
    on a calibration read instead of silently re-meaning the label; a label that arrives already
    carrying one is refused.

    **A label's form is its dimension's scale.** The criterion stamped on it says whether the dimension is
    ``pass_fail``, which takes a ``verdict``, or ``ordinal``, which takes a ``direction``; a label in the other form is
    refused, since it could never be compared with the judge's score. A label's ``memo_excerpt``, when given, must be
    a passage of the recorded memo, compared with each run of whitespace collapsed to one space on both sides and
    nothing else, the excerpt's ends trimmed. Both refusals are structural, and neither is applied when a stored case
    is read. The excerpt is not part of the verdict, but a freeze quoting other passages than the pair's live case is
    refused unless ``supersedes`` names it, as a label change is, so a correction is stored rather than dropped.

    **Who gave a verdict is taken as the caller names it.** This is the host's own door, trusted as its own
    ``rate_result`` is: a label's ``reader`` and ``collected_by`` are stored as given, or not at all. An action reaches
    it only through the ``reporter_case_freeze`` operation, which names each reader the host attests
    (``EvalHost.attests_person``) and the caller that relayed them. Neither is part of what makes two freezes the
    same case (below): a re-freeze of the same verdicts returns the stored case, whose record of them stands.

    **With an analysis, the bundle is assembled AS OF that analysis**: the insight ledger is
    read at its ``generation.bundle_assembled_at``, exactly as ``bundle_inspect`` re-assembles
    it, so the memo's own insights (and any minted since) are not in the evidence it is judged
    against. Without one, it is assembled now. Either way it is TODAY's re-assembly of the
    runs — a memo misled by evidence that has since been repaired is judged against the
    repaired evidence — and when that differs from what the memo read, the case's ``limits``
    say so.

    **A case is a (campaign, recorded memo) pair, and the bundle is not its identity.** One memo
    judged against its campaign's evidence is ONE case: the rule is stated beside
    :func:`~threetears.evals.analysis.reporter_bank.case_pair`, which every consumer of the case
    bank keys on. Because a freeze fingerprints TODAY's re-assembly, a re-freeze after the
    evidence moved yields a different fingerprint for the same memo, and it is treated as what
    it is — a revision of that memo's case — rather than as a second case the launch would
    judge alongside the first.

    **Idempotent, and immutable.** A freeze whose bundle fingerprint, labels (criteria and excerpts included)
    and writer message match the pair's live case RETURNS it rather than minting again, so two
    freezes of one campaign pool onto one case. Stored cases never change: a calibration read
    against labels (or evidence) a case did not carry when it was run would be a comparison
    against the wrong reader. Only their curation state does — a retired case
    (:func:`~threetears.evals.analysis.reporter_curation.set_reporter_case_archived`) is not live,
    so a freeze of its pair mints afresh.

    **A case is revised by supersession — its labels, its evidence, or both.** Among the stored
    cases of one pair, the LIVE case is the one no other stored case supersedes and no operator retired — decided by
    :func:`reporter_case_bank`, the derivation the launch and the calibration read also use.
    A freeze that differs from the live case — other labels, or a re-assembly whose fingerprint
    moved — is refused unless ``supersedes`` names the live case, and then a NEW case is minted
    carrying this freeze's bundle and labels and pointing at the one it replaces, which is left
    exactly as it was — so every run already measured against it still resolves it, and a
    launch never runs it again. Naming the replacement rather than
    superseding silently is deliberate: a re-run of a freeze meant only to read its receipt
    would otherwise re-point every later launch at evidence nobody chose to judge against.
    Only live cases can be replaced, and ``supersedes`` can only name cases of this pair.

    **Two live cases for one pair are RECOVERABLE, not unrepresentable — a stated choice.**
    This read-then-write has no conditional insert, so two supersessions of one live case racing
    each other can each mint a replacement. Making that unrepresentable needs a create-only
    write keyed on the superseded case, and the eval store offers only ``if_match`` against an
    EXISTING document's etag, which a stored case (immutable by design) never presents. So the
    state is detected instead of prevented: the launch and its price refuse a pair holding more
    than one live case, naming this remedy — a freeze whose ``supersedes`` names EVERY live case
    of the pair, which mints one case replacing them all. Naming all of them is required, so a
    freeze cannot retire a revision its author never saw; a launch never runs both.

    :param host: The host: where the campaign's evidence is read and the case is written, and the
        vocabulary the frozen bundle is assembled in.
    :ptype host: EvalHost
    :param template_id: The reporter template the case belongs to.
    :ptype template_id: str
    :param campaign_id: The campaign whose bundle is frozen.
    :ptype campaign_id: str
    :param scope_id: The scope its member runs (and the case) live in.
    :ptype scope_id: str
    :param analysis_id: A stored analysis of that campaign to pin as the recorded memo, or
        ``None`` to freeze the bundle alone.
    :ptype analysis_id: str | None
    :param labels: Reader verdicts on the recorded memo, as
        :class:`~threetears.evals.analysis.reporter_kind.ReporterLabel` objects or their
        JSON. Each must name a dimension the template scores.
    :ptype labels: Sequence[Any]
    :param supersedes: The ids of the live cases of this (campaign, recorded analysis) pair this
        freeze replaces — the one live case to revise its labels or its evidence, every live
        case to resolve a pair holding several — or empty when it replaces none. A bare
        string is refused: it would be read one character per id.
    :ptype supersedes: Sequence[str]
    :param load_template: The host's template load; raises ``NotFoundError`` for an unknown id, and
        whatever else the host refuses a template for.
    :ptype load_template: Callable[[str], EvalTemplate]
    :return: The stored case — the existing live one when this freeze's labels match it.
    :rtype: EvalTestCase
    :raises NotFoundError: The template, campaign or analysis does not exist.
    :raises ValidationFailedError: A template that is not of the reporter kind
        (:data:`~threetears.evals.analysis.reporter_kind.REPORTER_KIND`), a label that does not
        validate or names an unscored dimension, a label whose form (direction or verdict) is not its dimension's
        scale, a ``memo_excerpt`` that is not a passage of the recorded memo,
        labels with no recorded memo to be about, an analysis of another campaign,
        a campaign of several aspects frozen with a memo or none, a plan whose figure keys its frozen evidence no
        longer resolves, a bundle resolving no runs, a label change or a moved re-assembly that
        does not name the live case in ``supersedes``, a ``supersedes`` naming a case outside this pair or one
        already superseded or retired, ``supersedes`` on a pair with no live case, a label
        arriving with a criterion, a pair with more than one live case that ``supersedes`` does
        not name in full, or a stored case of the template this build cannot read.
    :raises StorageError: The new case could not be persisted.
    :raises TypeError: ``supersedes`` is a bare string rather than a sequence of ids.
    """
    if isinstance(supersedes, str):
        raise TypeError("supersedes is a sequence of case ids; a bare string would be read one character per id")
    replaced = [case_id for case_id in supersedes if case_id]
    template = load_template(template_id)  # NotFoundError if absent
    if template.candidate_kind != REPORTER_KIND:
        # A reporter case in any other kind's template is a case nothing reads: only the reporter
        # kind's launch runs one, and every other kind would meet a case it cannot interpret.
        raise ValidationFailedError(
            f"template {template_id!r} is of kind {template.candidate_kind!r}, not {REPORTER_KIND!r} — a reporter "
            f"case is frozen into a template of the {REPORTER_KIND!r} kind, whose launch is what runs it",
            remedy=f"freeze into a template of the {REPORTER_KIND!r} kind ({FIND_TEMPLATE})",
        )
    try:
        parsed = [
            label if isinstance(label, ReporterLabel) else ReporterLabel.model_validate(label) for label in labels
        ]
    except ValidationError as e:
        raise ValidationFailedError(
            f"invalid labels: {e}",
            remedy="send each label as a dimension, a direction or a verdict (one), and the reader's words",
        ) from e
    criteria = {dim.name: LabelCriterion.of(dim) for dim in template.rubric}
    if unscored := sorted({label.dimension for label in parsed} - set(criteria)):
        raise ValidationFailedError(
            f"labels name {', '.join(unscored)}, which template {template_id!r} does not score — a label on an "
            "unjudged dimension could never be compared with anything",
            remedy=f"label only the dimensions it scores: {', '.join(sorted(criteria)) or 'it scores none'}",
        )
    if supplied := sorted({label.dimension for label in parsed if label.criterion is not None}):
        raise ValidationFailedError(
            f"labels on {', '.join(supplied)} arrive carrying a criterion — the freeze stamps each label with the "
            f"criterion template {template_id!r} states for its dimension, so a label records what it was written "
            "against rather than what a caller said it was",
            remedy="send the labels without `criterion`",
        )
    # The criterion each verdict was written against, as the template words it now — the words a
    # reader of the memo had in front of them when the label is frozen with it.
    parsed = [label.model_copy(update={"criterion": criteria[label.dimension]}) for label in parsed]
    _refuse_a_label_of_another_form(parsed, template_id=template_id)
    if parsed and not analysis_id:
        raise ValidationFailedError(
            "labels are a reader's verdict on a recorded memo, and this freeze pins none",
            remedy=f"pass the analysis_id whose memo they were written about ({FIND_ANALYSIS})",
        )

    storage = host.storage
    campaign = _load_campaign(storage, campaign_id, scope_id)
    analysis = get_analysis(storage, analysis_id, scope_id) if analysis_id else None  # NotFoundError if absent
    if analysis is not None and analysis.campaign_id != campaign_id:
        raise ValidationFailedError(
            f"analysis {analysis.id!r} belongs to campaign {analysis.campaign_id!r}, not {campaign_id!r} — "
            "its memo would be judged against another campaign's evidence",
            remedy=f"pass an analysis of campaign {campaign_id!r} ({FIND_ANALYSIS})",
        )
    planned = analysis is not None and analysis.report_plan_id is not None
    if len(campaign.aspects) > 1 and not planned:
        # A memo case judges a memo over one aspect's evidence, and the memo reporter eval has not been measured over
        # several. A plan case judges the planner, which plans every aspect of its campaign.
        raise ValidationFailedError(
            f"campaign {campaign_id!r} has {len(campaign.aspects)} aspects "
            f"({', '.join(aspect.label for aspect in campaign.aspects)}), and a reporter case freezes a campaign of "
            "one aspect only when it pins a memo or none",
            remedy=(
                "freeze a campaign of one aspect: create one over the aspect's runs (campaign_create) and freeze it; "
                f"or freeze an analysis the report planner wrote of this campaign, as a plan case ({FIND_ANALYSIS})"
            ),
        )
    plan_case: _PlanCase | None = None
    if analysis is not None and planned:
        plan_case = _plan_case(storage, analysis)
        _refuse_an_excerpt_not_in_the_memo(parsed, plan_case.text)
        recorded_memo: str | None = plan_case.text
        assembled_at = analysis.generation.bundle_assembled_at
        bundle = plan_case.bundle
    elif analysis is not None:
        memo = render_memo_as_written(analysis)
        _refuse_an_excerpt_not_in_the_memo(parsed, memo)
        recorded_memo = memo
        # The ledger as the generation read it, on `inspect_analysis_bundle`'s terms.
        assembled_at = analysis.generation.bundle_assembled_at
        bundle = _assembled_campaign_bundle(
            campaign, storage=storage, profile=host.profile, insights_as_of=assembled_at
        )
    else:
        recorded_memo = None
        # Stamped BEFORE assembly, as a generation stamps it: the instant is the
        # insight ledger's cutoff.
        assembled_at = utc_now_iso()
        bundle = _assembled_campaign_bundle(campaign, storage=storage, profile=host.profile)
    if not bundle.run_ids:
        unresolved = [run_id for aspect in bundle.aspects for run_id in aspect.unresolved_run_ids]
        archived = [run_id for aspect in bundle.aspects for run_id in aspect.archived_run_ids]
        raise ValidationFailedError(
            f"campaign {campaign_id!r} resolves no runs in scope {scope_id!r} (unresolved: "
            f"{unresolved or 'none attached'}; archived: {archived or 'none'}) — "
            "a case with no evidence can ground no judgement",
            remedy="freeze a campaign whose runs this scope holds, un-archived",
        )

    fingerprint = bundle.fingerprint()
    wanted_analysis = analysis.id if analysis is not None else None
    # What the recorded memo is judged against from now on, frozen as text (see
    # `ReporterCase.writer_message`): rendered here, from the bundle this freeze pins, because the
    # stored analysis keeps no copy of the message its writer was sent — only its digest, which
    # the case carries so the text stays checkable against it.
    if plan_case is not None:
        # What the planner read, rendered from the report input the case freezes; its digest is the plan's own record.
        writer_message: str | None = plan_case.message
        recorded_message_digest: str | None = plan_case.stored.generation.user_message_digest
    else:
        writer_message = build_campaign_user_message(bundle) if analysis is not None else None
        recorded_message_digest = analysis.generation.user_message_digest if analysis is not None else None
    wanted_labels = _label_set(parsed, criterion=True)
    memo_text = f"analysis {wanted_analysis!r}" if wanted_analysis else "no recorded memo"
    bank = reporter_case_bank(storage, template.id, scope_id)
    # The pair is the campaign the bundle names and the memo — never the fingerprint, which
    # moves with every re-assembly of the same evidence (see `case_pair`).
    pair = bank.pair(bundle.campaign_id, wanted_analysis)
    live = [(stored, existing) for stored, existing in pair if bank.is_live(stored.id)]
    live_ids = sorted(stored.id for stored, _ in live)
    named = sorted(set(replaced))

    if outside := [case_id for case_id in named if case_id not in {stored.id for stored, _ in pair}]:
        raise ValidationFailedError(
            f"supersedes names {', '.join(repr(case_id) for case_id in outside)}, which "
            + ("is not a stored case" if len(outside) == 1 else "are not stored cases")
            + f" freezing campaign {bundle.campaign_id!r} with {memo_text} — a case can only replace a case of "
            "the same campaign and memo",
            remedy=(
                f"name that pair's live case(s) in supersedes: {', '.join(live_ids)}"
                if live
                else "no case of that pair is stored, so freeze without supersedes"
            ),
        )
    if named and not live:
        raise ValidationFailedError(
            f"supersedes names {', '.join(repr(case_id) for case_id in named)}, and campaign "
            f"{bundle.campaign_id!r} with {memo_text} has no live case — every stored case of it is superseded or "
            "retired, so there is nothing for this freeze to replace",
            remedy=(
                "freeze without supersedes: the new case is the pair's live case, and the retired ones stay as they "
                "are; supersedes takes only a live case, so to retire a retired case for good, restore it first "
                "(reporter_case_archive with archived=false), then freeze with supersedes naming it"
            ),
        )
    same_labels = len(live) == 1 and _label_set(live[0][1].labels, criterion=True) == wanted_labels
    same_verdicts = len(live) == 1 and _label_set(live[0][1].labels, criterion=False) == _label_set(
        parsed, criterion=False
    )
    # Today's bundle in the live case's fingerprint version, so a case frozen before fingerprints rounded their
    # floats still matches the evidence it froze; the case this freeze stores carries the current version.
    as_frozen = bundle.fingerprint(like=live[0][1].bundle_fingerprint) if len(live) == 1 else fingerprint
    same_evidence = len(live) == 1 and live[0][1].bundle_fingerprint == as_frozen
    # The digest rides with the message: a live case that froze this text without the digest its
    # analysis records is not the case this freeze would store.
    same_message = len(live) == 1 and (live[0][1].writer_message, live[0][1].recorded_writer_message_digest) == (
        writer_message,
        recorded_message_digest,
    )
    # The passages the labels quote are no part of the verdicts, but a freeze returning a case that quotes others
    # would report as stored what it is not.
    same_excerpts = len(live) == 1 and _label_set(live[0][1].labels, criterion=True, excerpt=True) == _label_set(
        parsed, criterion=True, excerpt=True
    )
    if same_labels and same_evidence and same_message and same_excerpts:
        # Idempotent on the LIVE case, whether or not a retried revision repeats its supersedes.
        return live[0][0]
    if live and named != live_ids:
        # Supersession names the WHOLE live set of the pair or nothing. With one live case that is
        # the ordinary revision; with several — two freezes racing a supersession can each mint a
        # replacement — it is the remedy, and requiring every one of them named keeps a freeze from
        # retiring a revision its author never saw.
        pointer = "[" + ", ".join(repr(case_id) for case_id in live_ids) + "]"
        if len(live) > 1:
            raise ValidationFailedError(
                f"campaign {bundle.campaign_id!r} with {memo_text} has {len(live)} live cases ({', '.join(live_ids)}) "
                "— each is launched, so no launch of this template runs until one case replaces them all",
                remedy=(
                    f"freeze again with the labels this pair should carry and supersedes={pointer}: that stores one "
                    "case replacing every one of them, and leaves them, and the runs measured against them, as they are"
                ),
            )
        if not named:
            differs = []
            if not same_verdicts:
                differs.append("under a DIFFERENT label set")
            elif not same_labels:
                differs.append(
                    "with the same verdicts written against criterion text that differs from the live case's (the "
                    "template's rubric was reworded since)"
                )
            elif not same_excerpts:
                differs.append(
                    f"with the same verdicts quoting other memo passages (stored: {_excerpts(live[0][1].labels)}; "
                    f"this freeze: {_excerpts(parsed)})"
                )
            if not same_message and same_evidence:
                differs.append(
                    "pinning a different writer message over the same bundle (the bundle's rendering changed since)"
                )
            if not same_evidence:
                differs.append(
                    f"from a bundle that no longer re-assembles the same (frozen {live[0][1].bundle_fingerprint}, "
                    f"now {fingerprint} — the evidence or the bundle's shape moved)"
                )
            raise ValidationFailedError(
                f"case {live_ids[0]!r} already freezes campaign {bundle.campaign_id!r} with {memo_text}, "
                + " and ".join(differs)
                + ". One memo is one case, and stored cases are immutable — a run already measured against it "
                "would otherwise be read against labels or evidence it never carried",
                remedy=(
                    f"to revise it, freeze again with supersedes={pointer}: that stores a new case carrying this "
                    f"freeze's bundle and labels, which every later launch runs instead, and leaves {live_ids[0]!r} — "
                    "and the runs measured against it — as they are"
                ),
            )
        raise ValidationFailedError(
            f"supersedes names {', '.join(repr(case_id) for case_id in named)}, and only the live case of this "
            f"campaign and memo can be replaced — that is {live_ids[0]!r}; the others are already superseded or "
            "retired",
            remedy=f"pass supersedes={pointer}",
        )

    limits = case_limits(
        bundle,
        recorded_fingerprint=analysis.generation.bundle_fingerprint if analysis is not None else None,
        writer_message=writer_message,
        recorded_message_digest=recorded_message_digest,
        kind_version=PLAN_CASE_VERSION if plan_case is not None else MEMO_CASE_VERSION,
    )
    planned_fields: dict[str, Any] = {}
    if plan_case is not None:
        limits += plan_case.limits
        planned_fields = {
            "kind_version": PLAN_CASE_VERSION,
            "report_input": plan_case.report_input.model_dump(mode="json", by_alias=True),
            "report_input_digest": plan_case.digest,
            "recorded_report_input_digest": plan_case.stored.report_input_digest,
            "recorded_plan": plan_case.stored.plan,
            "planner_prompt_version": plan_case.stored.generation.prompt_version,
        }
    case = ReporterCase(
        bundle=bundle.to_dict(),
        bundle_fingerprint=fingerprint,
        bundle_assembled_at=assembled_at,
        recorded_analysis_id=analysis.id if analysis is not None else None,
        recorded_memo=recorded_memo,
        writer_message=writer_message,
        recorded_writer_message_digest=recorded_message_digest,
        labels=parsed,
        limits=limits,
        supersedes=named,
        **planned_fields,
    )
    test_case = EvalTestCase(template_id=template.id, scope_id=scope_id, host_payload=reporter_case_payload(case))
    try:
        storage.save_test_case(test_case)
    except StorageError as e:
        raise StorageError(
            f"reporter case for campaign {campaign_id!r} could not be stored — nothing was frozen; retry the freeze"
        ) from e
    log.info(
        "eval.freeze_reporter_case template=%s campaign=%s analysis=%s scope=%s case=%s fingerprint=%s limits=%d "
        "supersedes=%s",
        template.id,
        campaign_id,
        wanted_analysis,
        scope_id,
        test_case.id,
        fingerprint,
        len(case.limits),
        ",".join(named) or None,
    )
    return test_case


def reporter_calibration(
    storage: AnalysisStore,
    run_id: str,
    scope_id: str,
) -> ReporterCalibration:
    """Read a reporter run against its cases' labels — the rubric's calibration.

    Pure read, no spend: every score comes from the run's stored results. Per case, each label
    is set beside the judge's score and reasoning on its dimension, with whether the score
    agrees with the label: an ordinal score inside its direction's band, a pass/fail score equal to its verdict
    (:func:`~threetears.evals.analysis.reporter_bank.label_agrees`); every dimension no label speaks to is listed with
    its score.
    A case whose labels were revised since the run says which stored case superseded it, so a
    reader of an earlier run sees that the labels it is read against were later replaced; a case
    retired since says so too. Each label reading says whether the criterion the label was
    written against still reads as the template states it (``criterion_drift``), and each case
    whether its recorded memo was judged against a frozen writer message.

    :param storage: Where the run's cases, results and template are read.
    :ptype storage: AnalysisStore
    :param run_id: The reporter run.
    :ptype run_id: str
    :param scope_id: The scope it ran in.
    :ptype scope_id: str
    :return: The calibration read.
    :rtype: ReporterCalibration
    :raises NotFoundError: No such run in that scope.
    :raises ValidationFailedError: The run froze no reporter case, so it is not a reporter run; or
        a stored case of its template carries a reporter case this build cannot read.
    """
    run = storage.load_eval_run(run_id, scope_id)
    if run is None:
        raise NotFoundError("run", run_id, remedy=FIND_RUN)
    loaded = {case.id: case for case in storage.load_test_cases_by_ids(list(run.test_case_ids), scope_id)}
    pairs = []
    missing: list[str] = []
    for case_id in run.test_case_ids:
        case = loaded.get(case_id)
        if case is None:
            missing.append(case_id)
            continue
        try:
            reporter_case = reporter_case_of(case)
        except ValueError as e:
            raise ValidationFailedError(
                f"case {case_id!r} of run {run_id!r} carries a reporter case this build cannot read: {e}",
                remedy="read it with the build that wrote it",
            ) from e
        if reporter_case is not None:
            pairs.append((case_id, reporter_case))
    if not pairs:
        raise ValidationFailedError(
            f"run {run_id!r} froze no reporter case"
            + (f" (and {len(missing)} of its cases no longer load)" if missing else "")
            + ", so it has no labels to calibrate against",
            remedy=f"calibrate a run of an analysis_reporter template ({FIND_RUN})",
        )
    # Who replaced whose labels, read off every stored case of the template rather than the
    # run's own: a superseding case is minted after the run it revises, so it is never one of
    # that run's cases. Read through the same bank the launch and the freeze read, so a case
    # this build cannot read is refused here as it is there — it may be the one that replaced a
    # case this read would otherwise call live.
    template_id = run.template_id
    if template_id is None:
        # An ad-hoc run names no template, so there is no bank to read supersession from — and
        # querying the scope's cases without one would read every template's.
        raise ValidationFailedError(
            f"run {run_id!r} names no template, so its cases' supersession cannot be read",
            remedy=f"calibrate a run of an analysis_reporter template ({FIND_RUN})",
        )
    bank = reporter_case_bank(storage, template_id, scope_id)
    superseded_by = {case_id: list(ids) for case_id, ids in bank.superseded_by.items()}
    # Read a page at a time, the results kept without their verbatim fields, which no calibration reads.
    results = results_without_payloads(storage, run_id, scope_id)
    template = storage.load_template(template_id, scope_id)
    # The template names the dims in order; a template deleted since the run still leaves its
    # results, so the dims they were scored on stand in for it rather than dropping the read.
    dimensions = (
        [dim.name for dim in template.rubric]
        if template is not None
        else sorted({score.dim for result in results for score in result.rubric_scores})
    )
    return read_calibration(
        run_id=run.id,
        template_id=template_id,
        status=run.status,
        dimensions=dimensions,
        cases=pairs,
        results=results,
        missing_case_ids=missing,
        superseded_by=superseded_by,
        archived=bank.archived,
        live_criteria=None if template is None else {dim.name: LabelCriterion.of(dim) for dim in template.rubric},
        judge_model=run.judge_model,
        effective_judges=run.effective_judges,
        ratings=storage.query_calibration_ratings(scope_id, run_id=run.id),
        template_digest=run.template_digest,
    )


if TYPE_CHECKING:

    def _eval_storage_satisfies_the_port(storage: EvalStorage) -> None:
        """Hold the engine's own store to this consumer's port, so a drifted signature fails typecheck."""
        stores: tuple[AnalysisStore, CampaignReadStore] = (storage, storage)
        del stores


__all__ = [
    "AnalysisGenerationEstimate",
    "AnalysisStore",
    "BudgetedGenerator",
    "DEFAULT_WRITER_UNRESOLVED",
    "PreparedGeneration",
    "analysis_report",
    "report_doc_report",
    "describe_insight_id_filters",
    "estimate_analysis_generation",
    "freeze_reporter_case",
    "get_analysis",
    "assemble_campaign_bundle",
    "require_evidence",
    "inspect_analysis_bundle",
    "campaign_report",
    "campaign_verdicts",
    "record_campaign_call",
    "inspect_campaign_bundle",
    "list_analyses",
    "list_analysis_attempts",
    "running_analysis_attempt",
    "list_insights",
    "prepare_analysis_generation",
    "assembled_analysis_attempt",
    "reporter_calibration",
    "reporter_case_bank",
    "record_written_generation",
    "run_analysis_generation",
    "WrittenAttempt",
]
