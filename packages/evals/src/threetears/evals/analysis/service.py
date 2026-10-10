"""The analysis service — generating, reading and curating a campaign's analyses, insights and reporter cases.

Every operation here is one a client of the packages needs to drive the analysis lens: generate
a campaign's memo and record how the attempt ended, read analyses and their attempts back,
re-assemble the bundle a generation read, read a stored analysis as its report, read
the insight ledger, and freeze, bank and calibrate the reporter cases that measure the generator
itself. They are module functions whose dependencies are parameters, so a host calls them
with its own store and its own bindings; a host's service delegates to them.

**What arrives as an argument.** An operation that reads the host's vocabulary — assembling a
bundle, generating over one, freezing one — takes the
:class:`~threetears.evals.contracts.host.EvalHost`: its storage, its profile, the client factory the
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
import json
import uuid
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

from threetears.evals.analysis.bundle import (
    BundleInspection,
    CampaignReadStore,
    assemble_context_bundle,
    superseding_insights,
)
from threetears.evals.analysis.errors import GenerationError, SoundnessRefusal
from threetears.evals.analysis.generator import MAX_GENERATION_CALLS, GenerationTally, build_user_message, first_request
from threetears.evals.analysis.generator import generate_analysis as _generate_analysis
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.reporter_bank import case_limits, decidable_reporter_case_bank, read_calibration
from threetears.evals.analysis.reporter_kind import (
    REPORTER_KIND,
    LabelCriterion,
    ReporterCase,
    ReporterLabel,
    render_memo_as_written,
    reporter_case_of,
    reporter_case_payload,
)
from threetears.evals.analysis.report import Report, build_code_only_report, build_report
from threetears.evals.analysis.viz.intent import chart_intent
from threetears.evals.analysis.viz.payloads import PayloadError
from threetears.evals.analysis.viz.policy import IntentPolicyError
from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.contracts.campaign import EvalAnalysisAttempt
from threetears.evals.contracts.errors import NotFoundError, ProviderRefusedError, StorageError, ValidationFailedError
from threetears.evals.contracts.models import EvalTestCase, utc_now_iso
from threetears.evals.contracts.offload import run_blocking, wait_through_cancellation
from threetears.evals.contracts.out_of_run import AdmittedCall, OutOfRunBudget, PlannedCall
from threetears.evals.contracts.provider import (
    describe_failure,
    log_provider_failure,
    traceback_is_safe,
)
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.contracts.storage import EvalStorage
    from threetears.evals.analysis.bundle import AnalysisContextBundle
    from threetears.evals.analysis.reporter_bank import ReporterCalibration, ReporterCaseBank
    from threetears.evals.analysis.viz.intent import ChartIntent
    from threetears.evals.contracts.campaign import AttemptOutcome, EvalAnalysis, EvalCampaign, EvalInsight
    from threetears.evals.contracts.models import EvalRun, EvalTemplate
    from threetears.evals.contracts.host.eval_host import EvalHost
    from concurrent.futures import Executor

    from threetears.evals.contracts.provider import BoundCompletionClient, CompletionResult

log = get_logger(__name__)

#: How many distinct scopes a refusal lists before summarising the rest.
#:
#: An insight's ``scope`` is free prose — a scope might read like "candidate
#: comparison, small case set, default judge" — so the
#: refusal is naming examples an operator can copy a substring out of, not
#: reproducing a vocabulary. Five is enough to show the shape.
_SCOPE_SAMPLE = 5


class AnalysisStore(CampaignReadStore, Protocol):
    """The storage calls the analysis service makes — its own, plus the bundle assembly's it hands the store to.

    It extends :class:`~threetears.evals.analysis.bundle.CampaignReadStore` rather than repeating it,
    because every function here that assembles a bundle passes this same store to
    :func:`~threetears.evals.analysis.bundle.assemble_context_bundle`, so a store satisfying this port
    has to satisfy that one. ``query_insights`` is declared again, wider: the ledger listing
    filters on the minting campaign as well as the subject, and the bundle's narrower call is one
    this signature accepts.

    Structural, so a host's own storage satisfies it by having the methods —
    :class:`~threetears.evals.contracts.storage.EvalStorage` does. Positional parameters are
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

    def list_analyses_by_campaign(self, campaign_id: str, scope_id: str, /) -> list[EvalAnalysis]:
        """Every analysis of one campaign, newest first."""
        ...

    def save_analysis_attempt(self, attempt: EvalAnalysisAttempt, /) -> None:
        """Write the record of how one generation attempt ended; raises ``StorageError`` on failure."""
        ...

    def list_analysis_attempts_by_campaign(self, campaign_id: str, scope_id: str, /) -> list[EvalAnalysisAttempt]:
        """Every generation attempt recorded against one campaign, newest first."""
        ...

    def save_insight(self, insight: EvalInsight, /) -> None:
        """Write an insight a generation minted; raises ``StorageError`` on failure."""
        ...

    def query_insights(
        self, scope_id: str, /, *, subject_id: str | None = None, source_campaign_id: str | None = None
    ) -> list[EvalInsight]:
        """Every insight in a scope, optionally narrowed by subject and by minting campaign, newest observation first."""
        ...

    def load_template(self, template_id: str, scope_id: str, /) -> EvalTemplate | None:
        """Load one template as stored in a scope, or ``None`` when it does not resolve — the calibration read's raw read."""
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

    Split from the spend so a background start can refuse an unknown campaign, an empty bundle or
    an unknown preset to its caller BEFORE returning, rather than in a task nobody is waiting on.
    """

    campaign_id: str
    scope_id: str
    bundle: AnalysisContextBundle
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


def _approx_analysis_input_tokens(bundle: AnalysisContextBundle, prompt: str) -> int:
    """Approximate the analysis generator call's input tokens, for pre-spend disclosure.

    A deliberately coarse ``chars/4`` heuristic over the serialized bundle (the
    generator's user message) plus the prompt (its system message) — the two
    strings the generator actually sends. It is a *scale* signal for the
    Visible-Costs log line, never a billing figure: the codebase has no per-token
    pricing table (cost is only ever observed, from the provider's reported usage),
    so the dollar cost is knowable only after the call. No tokenizer dependency is
    pulled in for a number that needs only an order of magnitude.

    Args:
        bundle: The assembled context bundle the generator will read.
        prompt: The resolved ``eval_analysis_gen`` system prompt.

    Returns:
        An approximate input-token count.
    """
    serialized = json.dumps(bundle.to_dict(), sort_keys=True)
    return (len(serialized) + len(prompt)) // 4


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

    Args:
        subject_id: The subject filter, if the caller supplied one.
        source_campaign_id: The campaign filter, if the caller supplied one.

    Returns:
        A comma-joined ``name=value`` phrase, or ``""`` when neither was supplied.
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

    Raises:
        NotFoundError: No campaign with that id in the scope.
    """
    campaign = storage.load_campaign(campaign_id, scope_id)
    if campaign is None:
        raise NotFoundError("campaign", campaign_id)
    return campaign


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class BudgetedGenerator:
    """A generation's calls, each priced and admitted against its out-of-run budget before it is sent, and ledgered.

    The generator is handed this as its client and its :meth:`admit` as its admission hook
    (:func:`~threetears.evals.analysis.generator.generate_analysis`), so the generator never learns what a
    cap is while every call it makes — the repair round-trip, whose prompt exists only once the first output
    is refused, included — goes through :class:`~threetears.evals.contracts.out_of_run.OutOfRunBudget`'s one
    rule: priced on the client before it is made, refused when the price would pass what is left of the cap,
    and written to the out-of-run ledger however it ends.
    """

    def __init__(self, budget: OutOfRunBudget, client: BoundCompletionClient) -> None:
        """Bind the budget to the client every call is priced on and made through.

        Args:
            budget: The generation's own budget.
            client: The generator client; this does not own it.
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

        Raises:
            ValidationFailedError: The call cannot be priced under an enforced cap, or its price would pass
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

        Raises:
            ValueError: No call with exactly this prompt was admitted — a call that would go unpriced.
        """
        pending, self._pending = self._pending, None
        if pending is None or pending.call != PlannedCall(system=system, user=user, response_format=response_format):
            raise ValueError("an analysis generator call was sent without being admitted against its budget")
        result: CompletionResult = (await self._budget.generate(self._client, pending)).result
        return result


class AnalysisGenerationEstimate(EvalBaseModel):
    """What a generation would be priced at before it starts, against the cap it would be held to.

    Attributes:
        campaign_id: The campaign.
        generator_model: The model the generation would call, as the host's client resolved it.
        first_call_ceiling_usd: The most the first call can cost, as the client prices it; ``None`` when it
            cannot say.
        max_calls: How many calls a generation can make: the first, and the one repair round-trip a refused
            output buys. The repair's prompt carries the refused output, so it is priced only when it exists —
            against what is left of the same cap, before it is sent.
        cap_usd: The out-of-run cap the generation is held to; ``None`` when the host enforces none.
        would_start: Whether the first call would be admitted.
        refusal: Why it would not, when it would not.
    """

    campaign_id: str
    generator_model: str
    first_call_ceiling_usd: float | None
    max_calls: int
    cap_usd: float | None
    would_start: bool
    refusal: str | None = None


class _Assembled(NamedTuple):
    """What a generation reads, checked and built: the bundle, the prompt, the client and its first call."""

    campaign: EvalCampaign
    bundle: AnalysisContextBundle
    assembled_at: str
    prompt: str
    client: BoundCompletionClient
    first_call: PlannedCall


async def _assemble(
    host: EvalHost,
    campaign_id: str,
    scope_id: str,
    *,
    model: str | None,
    resolve_prompt: Callable[[], Awaitable[str]],
) -> _Assembled:
    """Load the campaign, assemble its bundle off the event loop, refuse an empty one, resolve the prompt, build the client.

    Raises:
        NotFoundError: No campaign with that id.
        ValidationFailedError: The bundle has no resolvable runs, or the preset does not exist.
        ValueError: The host supplies no completion clients.
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
    bundle = await run_blocking(executor, assemble_context_bundle, campaign, storage=storage, profile=host.profile)
    if not bundle.run_ids:
        # No evidence to analyse: every attached run failed to resolve (destroyed
        # outside the delete cascade), was archived, or none is attached yet. Refuse rather than burn a paid generator call on an empty
        # bundle (Visible Costs — the very principle this action centres on).
        # This is the zero-evidence case, distinct from the thin-but-real data
        # generation is meant to degrade gracefully over.
        #
        # The two empty causes are named separately: "archived them all" is a
        # curation the operator can undo in one call, while "unresolved" is
        # data that is gone. One message for both would send them looking in
        # the wrong place.
        if bundle.archived_run_ids and not bundle.unresolved_run_ids:
            raise ValidationFailedError(
                f"campaign '{campaign_id}' has {len(bundle.archived_run_ids)} run(s) in scope "
                f"'{scope_id}' and every one is archived — nothing to analyse. Un-archive a run to "
                f"include it."
            )
        raise ValidationFailedError(
            f"campaign '{campaign_id}' has no runs resolvable in scope '{scope_id}' — nothing to "
            f"analyse (unresolved run ids: {bundle.unresolved_run_ids or 'none attached'}"
            + (f"; archived and excluded: {len(bundle.archived_run_ids)}" if bundle.archived_run_ids else "")
            + ")"
        )

    prompt = await resolve_prompt()
    system, user, contract = first_request(bundle, prompt, host.profile)

    # The effective model is read back off the built client (its bound ``model_name``) for the
    # pre-spend disclosure log and stored provenance, rather than re-implementing the host's
    # resolution cascade here.
    client = clients("analysis", model)
    return _Assembled(
        campaign=campaign,
        bundle=bundle,
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

    Args:
        host: The host: where the campaign is read and the client is built.
        campaign_id: The campaign.
        scope_id: The scope it lives in.
        model: The generator model override, or ``None`` for the host's default.
        resolve_prompt: Resolves the generator prompt.
        out_of_run_cap_usd: The out-of-run cap the generation would be held to; ``None`` when the host
            enforces none.

    Returns:
        The estimate.

    Raises:
        NotFoundError: No campaign with that id.
        ValidationFailedError: The bundle has no resolvable runs, or the preset does not exist.
        ValueError: The host supplies no completion clients.
    """
    assembled = await _assemble(host, campaign_id, scope_id, model=model, resolve_prompt=resolve_prompt)
    try:
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
) -> PreparedGeneration:
    """Check and build everything a generation needs before it spends anything, its first call priced and admitted.

    The refusals a generation documents for a missing campaign, an empty bundle, an unknown
    preset and a first call over the cap (or unpriceable under it) are all raised here, so a background
    start raises them to its caller. A host that admits one generation per campaign at a time checks that
    BEFORE calling this, because preparing builds a client and resolves the prompt, work a refused second
    request need not pay for. Every store read runs on the host's blocking executor.

    Args:
        host: The host: where the campaign's runs, results and insights are read, the vocabulary
            the bundle is assembled in, and the client factory the generator is built from.
        campaign_id: The campaign to analyse.
        scope_id: The scope the campaign, its member runs and everything generated from it live in.
        model: The generator model override, or None for the host's default.
        resolve_prompt: Resolves the generator prompt this generation runs, refusing an unknown
            preset with ``ValidationFailedError``. Awaited only once the bundle has evidence.
        out_of_run_cap_usd: The most the generation's calls may together be priced at — the host's
            out-of-run cap — or ``None`` when the host enforces none (each call is still priced where the
            client can say, and ledgered). Required, so every caller decides what bounds the spend.

    Returns:
        The prepared generation. It holds a built generator client, which the run enters and
        releases; one prepared and never run keeps its transport open until it is collected.

    Raises:
        NotFoundError: No campaign with that id.
        ValidationFailedError: The bundle has no resolvable runs, the preset does not exist, or the first call
            cannot be priced under an enforced cap or is priced above it — the client it built released.
        ValueError: The host supplies no completion clients.
    """
    assembled = await _assemble(host, campaign_id, scope_id, model=model, resolve_prompt=resolve_prompt)
    calls = BudgetedGenerator(_generation_budget(host, assembled, out_of_run_cap_usd), assembled.client)
    first = assembled.first_call
    try:
        calls.admit(first.system, first.user, first.response_format)
    except ValidationFailedError:
        await _release(assembled.client)
        raise
    return PreparedGeneration(
        campaign_id=campaign_id,
        scope_id=scope_id,
        bundle=assembled.bundle,
        assembled_at=assembled.assembled_at,
        prompt=assembled.prompt,
        client=assembled.client,
        resolved_model=assembled.client.model_name,
        attempt_id=str(uuid.uuid7()),
        calls=calls,
    )


async def _offloaded[**P](
    executor: Executor | None, fn: Callable[P, object], /, *args: P.args, **kwargs: P.kwargs
) -> bool:
    """Run a store write on ``executor``, waiting for it through a cancellation of the waiter.

    Args:
        executor: The host's blocking executor.
        fn: The blocking write.
        *args: Its positional arguments.
        **kwargs: Its keyword arguments.

    Returns:
        Whether a cancellation arrived while it ran — the caller delivers it once it has finished.

    Raises:
        Exception: Whatever the write raised.
    """
    work = asyncio.ensure_future(run_blocking(executor, fn, *args, **kwargs))
    cancelled = await wait_through_cancellation(work)
    work.result()
    return cancelled


async def run_analysis_generation(
    host: EvalHost,
    prepared: PreparedGeneration,
    *,
    prompt_id: str,
    max_output_tokens: int,
) -> tuple[EvalAnalysis, list[EvalInsight]]:
    """Make the paid generator call(s) for a prepared generation, then store and record the result.

    **Every attempt is recorded.** Once the generator is called, the attempt is stored as an
    :class:`~threetears.evals.contracts.campaign.EvalAnalysisAttempt` however it ends — stored,
    refused, failed or cancelled — with what it sent, what it cost and why it ended, read from a
    tally the generator writes as it goes.

    Args:
        host: The host: where the analysis, its insights and the attempt record are written, the
            vocabulary the memo is checked in, and how a provider call that raised becomes the
            text the attempt keeps.
        prepared: What :func:`prepare_analysis_generation` checked and built.
        prompt_id: The registry key the prompt was resolved under — provenance the generator
            cannot derive, because reading the registry is the host's.
        max_output_tokens: The output cap the client was built with, for the pre-spend log line.

    Returns:
        The stored analysis and the insights it minted.

    Raises:
        ValidationFailedError: The generator's output failed the generation contract, or the repair
            round-trip it bought was refused by the generation's budget before it was sent.
        StorageError: The analysis or one of its insights could not be stored. An insight fails
            AFTER the attempt is recorded ``stored``, so a background generation's poll reports
            the memo stored while its insight ledger is incomplete; the background task catches
            that failure and logs it, and nothing else reports it.
        ProviderRefusedError: The generator call raised and the host describer withheld its payload
            (a provider status error in its chain, or no describer able to say otherwise). The attempt is recorded ``failed`` and the failure
            logged, both through the host describer, before this is raised in its place. Any other
            failure is recorded ``failed`` and propagates as itself, so its status names no provider.
    """
    storage, failure_describer, executor = host.storage, host.failure_describer, host.blocking_executor
    campaign_id, scope_id, bundle = prepared.campaign_id, prepared.scope_id, prepared.bundle
    prompt, client, resolved_model = prepared.prompt, prepared.client, prepared.resolved_model

    # Visible Costs: disclose the SCALE of the call BEFORE making it — model,
    # approximate input size, output cap. No dollar figure (the codebase has no
    # pricing table; the real cost is logged after, from the provenance).
    #
    # The prose says "the generator model", not "opus". It said "opus call starting"
    # while interpolating `resolved_model` two lines down, so a run with a `model=`
    # override — a documented, supported way to A/B a cheaper generator — printed
    # `model=openai/gpt-5-mini (opus call starting)`. A log key naming the value it
    # carries is the observability norm; a key whose PROSE contradicts its own
    # interpolation is the same defect with an extra step.
    log.info(
        "eval.generate_analysis campaign=%s scope=%s model=%s runs=%d "
        "approx_input_tokens=%d max_output_tokens=%d (generator call starting — $ billed on completion)",
        campaign_id,
        scope_id,
        resolved_model,
        len(bundle.run_ids),
        _approx_analysis_input_tokens(bundle, prompt),
        max_output_tokens,
    )

    # Written by the generator as each call is sent and returns, so the record below is exact
    # however the generation ends — including a cancellation, which returns nothing and raises
    # nothing the generator could have annotated.
    tally = GenerationTally()
    started_at = utc_now_iso()

    async def record(outcome: AttemptOutcome, *, error: str | None = None, analysis_id: str | None = None) -> bool:
        # On the host's blocking executor, and waited for through a cancellation: the record is how the
        # attempt ended, so a cancel arriving while it is written must not leave it half-known. Returns
        # whether a cancellation arrived meanwhile, which the caller delivers once it has finished.
        return await _offloaded(
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
        )

    calls = prepared.calls
    try:
        # The client owns an httpx pool and this generation is its whole
        # lifetime; `async with` releases it on the raising path too, which
        # is a path this call reaches often enough to have its own handler.
        async with client:
            analysis, insights = await _generate_analysis(
                bundle,
                prompt=prompt,
                model=resolved_model,
                client=calls,
                prompt_id=prompt_id,
                bundle_assembled_at=prepared.assembled_at,
                tally=tally,
                admit=calls.admit,
                profile=host.profile,
            )
    except GenerationError as e:
        # Both calls' output refused is `refused`; every other generation-contract failure (a
        # call cut short, a precondition refused before any call) is `failed`.
        await record("refused" if isinstance(e, SoundnessRefusal) else "failed", error=str(e))
        # A generation-contract violation is caller-restatable (retry, or edit
        # the prompt), not a server fault — surface it as a 422, like the
        # proposer surfaces its own draft-validation failures.
        raise ValidationFailedError(f"analysis generation failed: {e}") from e
    except asyncio.CancelledError:
        # A caller's budget or disconnect, or a shutdown cancelling a background generation.
        # Recorded synchronously, before the cancellation propagates, because this is the
        # ending no exception message ever carries to a caller.
        repairing = ", so the repair round-trip was running" if tally.calls > 1 else ""
        await record(
            "cancelled",
            error=(
                "cancelled mid-flight by the caller's budget or disconnect, or a shutdown, "
                f"after {tally.calls} provider call(s){repairing}"
            ),
        )
        raise
    except ValidationFailedError as e:
        # The budget refused a call before it was sent — only the repair round-trip can reach this, since
        # the first call was admitted when the generation was prepared. Nothing was sent for it, so the
        # tally counts only what was; what the first call cost is on the record.
        await record("failed", error=f"the repair round-trip was refused before it was sent: {e}")
        raise ValidationFailedError(
            f"analysis generation failed: its output was refused and the one repair round-trip was refused before "
            f"it was sent ({e}); ${format_number(tally.token_cost)} was billed on the first call and nothing was stored"
        ) from e
    # prawduct:ok-broad-except — eval names no provider exception type; the host describer classifies what the call raised
    except Exception as e:
        # The exception's own text is not stored on the record: a provider status error stringifies
        # as its response envelope, account id included, and so does the last line of any traceback
        # that chains it. The host's describer says what is safe to keep. When it withheld the
        # payload, the failure is logged here once and raised on as an eval error rather than
        # propagated, because an exception escaping a REST route reaches the web server, which logs
        # its whole traceback; otherwise it propagates as itself and whoever catches it logs it.
        failure = describe_failure(failure_describer, e, logger=log, where="eval.generate_analysis")
        await record("failed", error=failure.description)
        if traceback_is_safe(failure):
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
        raise ProviderRefusedError(f"analysis generation failed: {failure.description}") from None

    # A failed persist of the just-paid artifact is re-raised with what it cost, so
    # the operator reading the 503 knows the spend happened and nothing is stored.
    # Every write runs on the host's blocking executor and is waited for through a cancellation, so a
    # cancel arriving while the paid-for analysis is stored cannot leave it stored with no record saying
    # so; the cancellation is delivered once the analysis, its record and its insights are written.
    try:
        cancelled = await _offloaded(executor, storage.save_analysis, analysis)
    except StorageError as e:
        await record("failed", error=f"the generated analysis could not be stored: {e}")
        raise StorageError(
            f"failed to persist generated analysis '{analysis.id}' for campaign '{campaign_id}' "
            f"(the generation cost ${format_number(analysis.generation.token_cost)} was already incurred)"
        ) from e
    cancelled = await record("stored", analysis_id=analysis.id) or cancelled
    # A restated claim replaces the live insight that states it rather than adding a row beside it, so
    # regenerating over the same evidence leaves the ledger its size. Read now, not at assembly: the ledger
    # the bundle read is capped, and another generation may have written since.
    written: list[list[EvalInsight]] = []

    def supersede() -> None:
        ledger = storage.query_insights(scope_id, subject_id=analysis.subject_id)
        written.append(
            superseding_insights(insights, ledger, lambda source: storage.analysis_archived(source, scope_id))
        )

    try:
        cancelled = await _offloaded(executor, supersede) or cancelled
    except StorageError as e:
        raise StorageError(
            f"failed to read the insight ledger before writing analysis '{analysis.id}''s insights — the analysis "
            f"itself was stored, but none of its insights were"
        ) from e
    insights = written[0]
    for insight in insights:
        try:
            cancelled = await _offloaded(executor, storage.save_insight, insight) or cancelled
        except StorageError as e:
            raise StorageError(
                f"failed to persist insight '{insight.id}' from analysis '{analysis.id}' — the analysis "
                f"itself was stored, but its insight ledger is incomplete"
            ) from e

    log.info(
        "eval.generate_analysis stored campaign=%s analysis=%s insights=%d cost=$%.4f model=%s",
        campaign_id,
        analysis.id,
        len(insights),
        analysis.generation.token_cost,
        analysis.generation.generator_model,
    )
    if cancelled:
        raise asyncio.CancelledError
    return analysis, insights


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
) -> None:
    """Store how one generation attempt ended, from the tally the generator kept.

    A write that fails is logged at ERROR with the whole record and NOT raised: this runs on the
    way out of every ending, including a failure the caller must still be shown, and a second
    exception here would replace the one that says what went wrong.

    Args:
        storage: Where the attempt record is written.
        campaign_id: The campaign the generation was over.
        scope_id: The scope the campaign lives in, where the record is written.
        outcome: How it ended.
        tally: What the generator sent, spent and was refused.
        generator_model: The model the generator client was built for.
        prompt_id: The registry key the prompt was resolved under.
        bundle_fingerprint: The fingerprint of the bundle it ran over.
        started_at: When the generation was started.
        error: What the caller was told ended it; None when it was stored.
        analysis_id: The analysis it stored; None unless it was.
        attempt_id: The attempt's id, minted when the generation was prepared — a background
            generation's job id, so its poller can find this record once the job is gone.
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
    )
    try:
        storage.save_analysis_attempt(attempt)
    except StorageError:
        log.exception(
            "eval.generate_analysis attempt record NOT stored campaign=%s outcome=%s calls=%d cost=$%.4f model=%s — %s",
            campaign_id,
            outcome,
            attempt.calls,
            attempt.token_cost,
            generator_model,
            attempt.model_dump_json(),
        )


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def get_analysis(storage: AnalysisStore, analysis_id: str, scope_id: str) -> EvalAnalysis:
    """Load a stored analysis by id.

    Args:
        storage: Where analyses are read.
        analysis_id: The analysis to load.
        scope_id: The scope it lives in.

    Returns:
        The analysis.

    Raises:
        NotFoundError: No analysis with that id in the scope.
    """
    analysis = storage.load_analysis(analysis_id, scope_id)
    if analysis is None:
        raise NotFoundError("analysis", analysis_id)
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


def inspect_campaign_bundle(
    host: EvalHost,
    campaign_id: str,
    scope_id: str,
) -> BundleInspection:
    """Assemble a campaign's context bundle and return it, without generating.

    The dry run: what a generation launched right now would read. This is the
    surface an operator checks a design against — ``set_campaign_control``
    designates a control, but whether the derivation produced a
    one-factor-at-a-time shape, and which lever each cell moved, was previously
    observable only by paying for a generation.

    Costs nothing: the assembler performs storage reads and no model call.

    Args:
        host: The host: where the campaign's runs, results and insights are read, and the
            vocabulary the bundle is assembled in.
        campaign_id: The campaign to assemble for.
        scope_id: The scope the campaign and its member runs live in.

    Returns:
        The assembled bundle with its freshly computed fingerprint. No
        generation is being spoken about, so nothing is compared against.

    Raises:
        NotFoundError: No campaign with that id in the scope.
    """
    campaign = _load_campaign(host.storage, campaign_id, scope_id)
    # Deliberately NOT refused when the bundle resolves no runs, unlike
    # a generation. There the refusal protects a paid call from an empty
    # bundle; here the empty bundle IS the answer — it names, in
    # unresolved_run_ids and archived_run_ids, which of the two causes emptied
    # it, and that is the diagnosis an operator whose generation just refused
    # came here to read.
    bundle = assemble_context_bundle(campaign, storage=host.storage, profile=host.profile)
    return BundleInspection.over(bundle)


def inspect_analysis_bundle(
    host: EvalHost,
    analysis_id: str,
    scope_id: str,
) -> BundleInspection:
    """Re-assemble the context bundle a stored analysis was generated over.

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
    the analysis that minted one archived — which retracts it from every re-assembly). An
    analysis stored before both were recorded reads ``cannot_say``. The scope cannot be a fourth
    cause: an analysis lives in its campaign's scope, which is the only scope its member runs
    can live in, so the re-assembly reads exactly the partition the generation read.

    Costs nothing: storage reads and no model call.

    Args:
        host: The host: where the analysis, its campaign's runs, results and insights are read,
            and the vocabulary the bundle is re-assembled in.
        analysis_id: The stored analysis to re-assemble the input of.
        scope_id: The scope the analysis, its campaign and the campaign's runs live in.

    Returns:
        The re-assembled bundle, the recorded fingerprint, and whether the two
        agree.

    Raises:
        NotFoundError: No analysis with that id, or the campaign it names no
            longer exists in the scope.
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
    bundle = assemble_context_bundle(
        campaign, storage=storage, profile=host.profile, insights_as_of=analysis.generation.bundle_assembled_at
    )
    return BundleInspection.over(
        bundle,
        analysis_id=analysis.id,
        recorded_fingerprint=analysis.generation.bundle_fingerprint,
        recorded_schema_version=analysis.generation.bundle_schema_version,
        recorded_host_declarations_digest=analysis.generation.host_declarations_digest,
    )


# ---------------------------------------------------------------------------
# The report and its charts — derived on every read, so a stored analysis stays pivotable
# ---------------------------------------------------------------------------


def analysis_report(storage: AnalysisStore, analysis_id: str, scope_id: str) -> Report:
    """Read one stored analysis as its report — the one document every surface renders.

    In place of the four reads a surface used to assemble for itself (the analysis, its arm table, its
    decision-surface table, its compiled charts): one document holds all of it, so a browser and an
    agent cannot disagree about which arm won, and a chart that cannot be drawn is served as such rather
    than omitted. Derived on every read and never written back, so a stored analysis stays pivotable.

    Args:
        storage: Where the analysis is read.
        analysis_id: The analysis to report.
        scope_id: The scope it lives in.

    Returns:
        The report; serialize it with :func:`~threetears.evals.analysis.report.report_markdown`,
        :func:`~threetears.evals.analysis.report.report_html` or its own JSON dump.

    Raises:
        NotFoundError: No analysis with that id in the scope.
    """
    return build_report(get_analysis(storage, analysis_id, scope_id))


def campaign_report(host: EvalHost, campaign_id: str, scope_id: str) -> Report:
    """The campaign's report — THE answer to "what is this campaign's report", for every caller.

    **The rule.** The campaign's newest analysis that is not archived, laid out by
    :func:`~threetears.evals.analysis.report.build_report` (``basis="analysis"``). When it has none —
    no generation has run, or every analysis it had was archived (an archive is the claim that an
    analysis was wrong, so it is not the campaign's report) — the campaign's evidence assembled now and
    laid out by :func:`~threetears.evals.analysis.report.build_code_only_report` (``basis="code_only"``),
    which says in its first block that no analysis was generated. A particular analysis, archived ones
    included, is read by its id through :func:`analysis_report`.

    Costs no model call either way: the code-only report is storage reads and arithmetic.

    Args:
        host: The host: where the campaign and its analyses are read, and the vocabulary its evidence is
            assembled in when there is no analysis.
        campaign_id: The campaign.
        scope_id: The scope it lives in.

    Returns:
        The report; its ``basis`` says which it is.

    Raises:
        NotFoundError: No campaign with that id in the scope.
    """
    campaign = _load_campaign(host.storage, campaign_id, scope_id)
    live = [analysis for analysis in list_analyses(host.storage, campaign_id, scope_id) if not analysis.archived]
    if live:
        return build_report(live[0])
    assembled_at = utc_now_iso()
    bundle = assemble_context_bundle(campaign, storage=host.storage, profile=host.profile)
    return build_code_only_report(
        bundle, measures=host.profile.measures, assembled_at=assembled_at, campaign_name=campaign.name
    )


def finding_chart_intent(storage: AnalysisStore, analysis_id: str, scope_id: str, finding_id: str) -> ChartIntent:
    """Decide one finding's chart, for a surface that draws a single chart at a time with its own renderer.

    Args:
        storage: Where the analysis is read.
        analysis_id: Analysis holding the finding.
        scope_id: The scope it lives in.
        finding_id: The finding's position in the authored document, as a string (``"0"`` is the first).

    Returns:
        The chart's intent — what it draws and what it must say — for a
        :class:`~threetears.evals.analysis.viz.ChartRenderer` to draw.

    Raises:
        NotFoundError: No such analysis, or no such finding, or the finding
            carries no chart this build can decide.
    """
    analysis = get_analysis(storage, analysis_id, scope_id)
    # A finding is identified by its position in the authored document, as a string.
    position = int(finding_id) if finding_id.isdigit() else -1
    if not 0 <= position < len(analysis.resolutions):
        raise NotFoundError("finding", finding_id)
    viz = analysis.resolutions[position].chart
    if viz is None:
        raise NotFoundError("chart for finding", finding_id)
    try:
        return chart_intent(viz.type, viz.payload)
    except (PayloadError, IntentPolicyError) as exc:
        # Surfaced rather than swallowed: this caller asked for THIS chart by id,
        # so an empty answer would read as "no chart here" when the truth is that
        # the stored payload cannot be drawn.
        raise NotFoundError("drawable chart for finding", f"{finding_id} ({exc})") from exc


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

    Args:
        storage: Where the insight ledger is read.
        scope_id: The storage scope whose ledger to read.
        subject_id: Optional equality filter on the subject an insight is about.
        scope: Optional case-insensitive SUBSTRING filter over the insight's
            free-text scope.
        source_campaign_id: Optional equality filter on the minting campaign.

    Returns:
        The matching insights, newest observation first.

    Raises:
        ValidationFailedError: ``scope`` was given and matched no insight.
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
        f"No insight's scope contains {scope!r}. Searched {len(insights)} insight(s){narrowed}; "
        f"`scope` matches by substring, case-insensitively.{vocabulary}"
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

    Args:
        storage: Where the template's cases are read.
        template_id: The reporter template.
        scope_id: The partition its cases live in.

    Returns:
        The bank.

    Raises:
        ValidationFailedError: A stored case of the template carries a reporter case this build
            cannot read.
    """
    return decidable_reporter_case_bank(
        storage.query_test_cases(scope_id, template_id=template_id), template_id=template_id
    )


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
    """Freeze a campaign's analysis bundle into a reporter case of ``template_id``.

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
    (:func:`~threetears.evals.analysis.reporter_kind.writer_message_check`). Each label is stamped with the criterion the
    template states for its dimension (``ReporterLabel.criterion``), so a reworded rubric shows
    on a calibration read instead of silently re-meaning the label; a label that arrives already
    carrying one is refused.

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

    **Idempotent, and immutable.** A freeze whose bundle fingerprint, labels (criteria included)
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

    Args:
        host: The host: where the campaign's evidence is read and the case is written, and the
            vocabulary the frozen bundle is assembled in.
        template_id: The reporter template the case belongs to.
        campaign_id: The campaign whose bundle is frozen.
        scope_id: The scope its member runs (and the case) live in.
        analysis_id: A stored analysis of that campaign to pin as the recorded memo, or
            ``None`` to freeze the bundle alone.
        labels: Reader verdicts on the recorded memo, as
            :class:`~threetears.evals.analysis.reporter_kind.ReporterLabel` objects or their
            JSON. Each must name a dimension the template scores.
        supersedes: The ids of the live cases of this (campaign, recorded analysis) pair this
            freeze replaces — the one live case to revise its labels or its evidence, every live
            case to resolve a pair holding several — or empty when it replaces none. A bare
            string is refused: it would be read one character per id.
        load_template: The host's template load; raises ``NotFoundError`` for an unknown id, and
            whatever else the host refuses a template for.

    Returns:
        The stored case — the existing live one when this freeze's labels match it.

    Raises:
        NotFoundError: The template, campaign or analysis does not exist.
        ValidationFailedError: A template that is not of the reporter kind
            (:data:`~threetears.evals.analysis.reporter_kind.REPORTER_KIND`), a label that does not
            validate or names an unscored dimension,
            labels with no recorded memo to be about, an analysis of another campaign,
            a bundle resolving no runs, a label change or a moved re-assembly that
            does not name the live case in ``supersedes``, a ``supersedes`` naming a case outside this pair or one
            already superseded or retired, ``supersedes`` on a pair with no live case, a label
            arriving with a criterion, a pair with more than one live case that ``supersedes`` does
            not name in full, or a stored case of the template this build cannot read.
        StorageError: The new case could not be persisted.
        TypeError: ``supersedes`` is a bare string rather than a sequence of ids.
    """
    from pydantic import ValidationError

    if isinstance(supersedes, str):
        raise TypeError("supersedes is a sequence of case ids; a bare string would be read one character per id")
    replaced = [case_id for case_id in supersedes if case_id]
    template = load_template(template_id)  # NotFoundError if absent
    if template.candidate_kind != REPORTER_KIND:
        # A reporter case in any other kind's template is a case nothing reads: only the reporter
        # kind's launch runs one, and every other kind would meet a case it cannot interpret.
        raise ValidationFailedError(
            f"template {template_id!r} is of kind {template.candidate_kind!r}, not {REPORTER_KIND!r} — a reporter "
            f"case is frozen into a template of the {REPORTER_KIND!r} kind, whose launch is what runs it."
        )
    try:
        parsed = [
            label if isinstance(label, ReporterLabel) else ReporterLabel.model_validate(label) for label in labels
        ]
    except ValidationError as e:
        raise ValidationFailedError(f"invalid labels: {e}") from e
    criteria = {dim.name: LabelCriterion.of(dim) for dim in template.rubric}
    if unscored := sorted({label.dimension for label in parsed} - set(criteria)):
        raise ValidationFailedError(
            f"labels name {', '.join(unscored)}, which template {template_id!r} does not score "
            f"({', '.join(sorted(criteria)) or 'no rubric dims'}) — a label on an unjudged dimension could never "
            "be compared with anything."
        )
    if supplied := sorted({label.dimension for label in parsed if label.criterion is not None}):
        raise ValidationFailedError(
            f"labels on {', '.join(supplied)} arrive carrying a criterion — the freeze stamps each label with the "
            f"criterion template {template_id!r} states for its dimension, so a label records what it was written "
            "against rather than what a caller said it was. Send the labels without `criterion`."
        )
    # The criterion each verdict was written against, as the template words it now — the words a
    # reader of the memo had in front of them when the label is frozen with it.
    parsed = [label.model_copy(update={"criterion": criteria[label.dimension]}) for label in parsed]
    if parsed and not analysis_id:
        raise ValidationFailedError(
            "labels are a reader's verdict on a recorded memo, and this freeze pins none — pass the analysis_id "
            "whose memo they were written about."
        )

    storage = host.storage
    campaign = _load_campaign(storage, campaign_id, scope_id)
    analysis = get_analysis(storage, analysis_id, scope_id) if analysis_id else None  # NotFoundError if absent
    if analysis is not None:
        if analysis.campaign_id != campaign_id:
            raise ValidationFailedError(
                f"analysis {analysis.id!r} belongs to campaign {analysis.campaign_id!r}, not {campaign_id!r} — "
                "its memo would be judged against another campaign's evidence."
            )
        # The ledger as the generation read it, on `inspect_analysis_bundle`'s terms.
        assembled_at = analysis.generation.bundle_assembled_at
        bundle = assemble_context_bundle(campaign, storage=storage, profile=host.profile, insights_as_of=assembled_at)
    else:
        # Stamped BEFORE assembly, as a generation stamps it: the instant is the
        # insight ledger's cutoff.
        assembled_at = utc_now_iso()
        bundle = assemble_context_bundle(campaign, storage=storage, profile=host.profile)
    if not bundle.run_ids:
        raise ValidationFailedError(
            f"campaign {campaign_id!r} resolves no runs in scope {scope_id!r} (unresolved: "
            f"{bundle.unresolved_run_ids or 'none attached'}; archived: {bundle.archived_run_ids or 'none'}) — "
            "a case with no evidence can ground no judgement."
        )

    fingerprint = bundle.fingerprint()
    wanted_analysis = analysis.id if analysis is not None else None
    # What the recorded memo is judged against from now on, frozen as text (see
    # `ReporterCase.writer_message`): rendered here, from the bundle this freeze pins, because the
    # stored analysis keeps no copy of the message its writer was sent — only its digest, which
    # the case carries so the text stays checkable against it.
    writer_message = build_user_message(bundle) if analysis is not None else None
    recorded_message_digest = analysis.generation.user_message_digest if analysis is not None else None
    wanted_labels = sorted(label.model_dump_json() for label in parsed)
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
            "the same campaign and memo. "
            + (
                f"That pair's live case(s): {', '.join(live_ids)}."
                if live
                else "No case of that pair is stored, so freeze without supersedes."
            )
        )
    if named and not live:
        raise ValidationFailedError(
            f"supersedes names {', '.join(repr(case_id) for case_id in named)}, and campaign "
            f"{bundle.campaign_id!r} with {memo_text} has no live case — every stored case of it is superseded or "
            "retired, so there is nothing for this freeze to replace. Freeze without supersedes: the new case is "
            "the pair's live case, and the retired ones stay as they are."
        )
    same_labels = len(live) == 1 and sorted(label.model_dump_json() for label in live[0][1].labels) == wanted_labels
    same_verdicts = len(live) == 1 and sorted(
        label.model_copy(update={"criterion": None}).model_dump_json() for label in live[0][1].labels
    ) == sorted(label.model_copy(update={"criterion": None}).model_dump_json() for label in parsed)
    same_evidence = len(live) == 1 and live[0][1].bundle_fingerprint == fingerprint
    # The digest rides with the message: a live case that froze this text without the digest its
    # analysis records is not the case this freeze would store.
    same_message = len(live) == 1 and (live[0][1].writer_message, live[0][1].recorded_writer_message_digest) == (
        writer_message,
        recorded_message_digest,
    )
    if same_labels and same_evidence and same_message:
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
                "— each is launched, so no launch of this template runs until one case replaces them all. Freeze "
                f"again with the labels this pair should carry and supersedes={pointer}: that stores one case "
                "replacing every one of them, and leaves them, and the runs measured against them, as they are."
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
                "would otherwise be read against labels or evidence it never carried. To revise it, freeze again "
                f"with supersedes={pointer}: that stores a new case carrying this freeze's bundle and labels, which "
                f"every later launch runs instead, and leaves {live_ids[0]!r} — and the runs measured against it — "
                "as they are."
            )
        raise ValidationFailedError(
            f"supersedes names {', '.join(repr(case_id) for case_id in named)}, and only the live case of this "
            f"campaign and memo can be replaced — that is {live_ids[0]!r}; the others are already superseded or "
            "retired. "
            f"Pass supersedes={pointer}."
        )

    case = ReporterCase(
        bundle=bundle.to_dict(),
        bundle_fingerprint=fingerprint,
        bundle_assembled_at=assembled_at,
        recorded_analysis_id=analysis.id if analysis is not None else None,
        recorded_memo=render_memo_as_written(analysis) if analysis is not None else None,
        writer_message=writer_message,
        recorded_writer_message_digest=recorded_message_digest,
        labels=parsed,
        limits=case_limits(
            bundle,
            recorded_fingerprint=analysis.generation.bundle_fingerprint if analysis is not None else None,
            writer_message=writer_message,
            recorded_message_digest=recorded_message_digest,
        ),
        supersedes=named,
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
    falls in the label's band; every dimension no label speaks to is listed with its score.
    A case whose labels were revised since the run says which stored case superseded it, so a
    reader of an earlier run sees that the labels it is read against were later replaced; a case
    retired since says so too. Each label reading says whether the criterion the label was
    written against still reads as the template states it (``criterion_drift``), and each case
    whether its recorded memo was judged against a frozen writer message.

    Args:
        storage: Where the run's cases, results and template are read.
        run_id: The reporter run.
        scope_id: The scope it ran in.

    Returns:
        The calibration read.

    Raises:
        NotFoundError: No such run in that scope.
        ValidationFailedError: The run froze no reporter case, so it is not a reporter run; or
            a stored case of its template carries a reporter case this build cannot read.
    """
    run = storage.load_eval_run(run_id, scope_id)
    if run is None:
        raise NotFoundError("run", run_id)
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
                f"case {case_id!r} of run {run_id!r} carries a reporter case this build cannot read: {e}"
            ) from e
        if reporter_case is not None:
            pairs.append((case_id, reporter_case))
    if not pairs:
        raise ValidationFailedError(
            f"run {run_id!r} froze no reporter case"
            + (f" (and {len(missing)} of its cases no longer load)" if missing else "")
            + ", so it has no labels to calibrate against — calibration reads a run of an analysis_reporter template."
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
            f"run {run_id!r} names no template, so its cases' supersession cannot be read — calibration reads a "
            "run of an analysis_reporter template."
        )
    bank = reporter_case_bank(storage, template_id, scope_id)
    superseded_by = {case_id: list(ids) for case_id, ids in bank.superseded_by.items()}
    results = storage.query_eval_results_by_run(run_id, scope_id)
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
    "PreparedGeneration",
    "analysis_report",
    "finding_chart_intent",
    "describe_insight_id_filters",
    "estimate_analysis_generation",
    "freeze_reporter_case",
    "get_analysis",
    "inspect_analysis_bundle",
    "campaign_report",
    "inspect_campaign_bundle",
    "list_analyses",
    "list_analysis_attempts",
    "list_insights",
    "prepare_analysis_generation",
    "reporter_calibration",
    "reporter_case_bank",
    "run_analysis_generation",
]
