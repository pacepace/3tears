"""The pairwise pass: a writer campaign's candidate and control pages, side by side, judged in both orders.

An absolute judge reads one page, and the writing loop revises every page toward the rubric that judge scores, so
absolute scores bunch at the top; and a page's omission is invisible to a judge reading it alone. Read beside the other
arm's page of the same case, it shows. This pass asks that question
(:mod:`~threetears.evals.ops.report_writer.preference`) of a finished writer campaign, with no engine change:

- **Pairs by case and repeat.** Each result of the candidate arm is paired with the control's result of the same case
  and repeat. A result with no partner is named and left out.
- **Both orders, from the stored pages.** Each pair's evidence is the two results' stored judge evidence
  (``EvalStorage.load_judge_evidence``), nothing re-rendered, built once with the control as Report A and once with
  the candidate as Report A (:func:`~threetears.evals.ops.report_writer.preference.pair_evidence`), and scored on
  :func:`~threetears.evals.ops.report_writer.preference.preference_rubric` through the judge service.
- **A side with no page loses.** A result whose cell stored no evidence wrote no page a reader could use, so the other
  side wins the pair without a call; both missing is a tie. Each is recorded on its pair, and weighs as one pair asked
  in both orders.
- **The campaign's scoring judge.** The arms must have been scored by one judge on one rubric
  (:func:`~threetears.evals.ops.report_writer.writer_contract.refuse_unfixed_scoring_judge`), and that judge is asked.
  It is held to another family than every writer it reads, by the launcher's rule
  (:func:`~threetears.evals.ops.report_writer.writer_launch.judge_family_check`), on the host's own reading of a
  model's family.
- **Priced before it is paid for.** Every call, parse retries included, is admitted against the host's out-of-run cap
  under purpose ``judge`` before the first is sent; a pass the cap cannot cover is refused with nothing spent.
- **The verdict.** Each case's candidate share (:func:`~threetears.evals.ops.report_writer.preference.case_preference`)
  is read against the control's, one minus it, paired on the range 0 to 1, by the bounded test and interval every rate
  contrast reads (:func:`~threetears.evals.analysis.stats.bounded_separation_p`,
  :func:`~threetears.evals.analysis.stats.bounded_difference_interval`). Its family is this one reading, so its
  adjusted p is its p. The same reading over the pairs whose two pages were both stored finished is reported beside
  it, since a draft's page carries its open editor findings and a judge may simply prefer the page without them.
- **How far the judge can be trusted on it.** Order consistency (the two orders pick the same side), the lean to the
  first slot, and the can't-tell count, over the pages it was actually asked about. A case scored in one order only
  is counted apart and left out of the consistency rate, since one order cannot agree or disagree with itself.

It runs as a job (:func:`writer_pairwise`): every refusal and every admission is made in the call, and the job's record
(:class:`~threetears.evals.run.EvalActionJob`) holds the :class:`WriterPairwiseReport`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Final, Literal

from pydantic import Field

from threetears.evals.analysis.report.blocks import ContrastOutcome, Verdict, VerdictReason
from threetears.evals.analysis.report.verdict import MIN_CASES
from threetears.evals.analysis.report.words import comparison_words
from threetears.evals.analysis.service import assemble_campaign_bundle
from threetears.evals.analysis.stats import (
    INTERVAL_LEVEL,
    SIGNIFICANCE_ALPHA,
    bounded_difference_interval,
    bounded_separation_p,
)
from threetears.evals.kernel.capped_completion import refresh_prices
from threetears.evals.kernel.declaration import CampaignArm, resolve_control_arm
from threetears.evals.kernel.errors import NotFoundError, ValidationFailedError
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.out_of_run import OutOfRunBudget, PlannedCall
from threetears.evals.kernel.remedies import FIND_CAMPAIGN, FIND_TEMPLATE
from threetears.evals.kernel.run_pages import results_without_payloads
from threetears.evals.ops.jobs import AdmittedJob, JobsStarted, start_job
from threetears.evals.ops.report_writer.preference import (
    CasePreference,
    PairJudgment,
    Slot,
    case_preference,
    pair_evidence,
    preference_rubric,
)
from threetears.evals.ops.report_writer.writer_contract import FINISHED_MEASURE, refuse_unfixed_scoring_judge
from threetears.evals.ops.report_writer.writer_launch import ModelFamily, judge_family_check
from threetears.evals.run.job_work import EvalActionJob, JobWork, action_job_work
from threetears.evals.run.judge import JUDGE_CALL_ATTEMPTS
from threetears.evals.run.judge_repeat import BudgetedJudgeClient, planned_judge_call
from threetears.evals.run.judge_service import JudgeContext, JudgeService, judge_clients_for_run
from threetears.evals.run.runner import build_judge_context
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.logs import get_logger
from threetears.evals.schema.models import (
    EvalResult,
    EvalTemplate,
    EvalTestCase,
    JudgedArtifact,
    JudgeEvidence,
    fingerprint_criterion,
)

if TYPE_CHECKING:
    from threetears.evals.kernel.host.eval_host import EvalHost
    from threetears.evals.ops.host import OpsHost
    from threetears.evals.run.judge_service import JudgeOutcome

log = get_logger(__name__)

#: The name the pass's job record (:class:`~threetears.evals.run.EvalActionJob`) carries as its ``action``, and the
#: engine action that starts it.
WRITER_PAIRWISE_ACTION: Final = "writer_pairwise"

#: The pass's wall-clock ceiling as a job: every call is bounded by its client and the pass stops on an account
#: refusal, so this only ends one whose provider hangs.
WRITER_PAIRWISE_BUDGET_S: Final = 6 * 3600.0

#: The heading the pass's verdict on the preference carries, which a reader reads it by.
PREFERRED_HEADING: Final = "Preferred by the judge, side by side"

#: The candidate's slot in each of a pair's two orders: the control is read first, then the candidate.
_SLOTS: Final[tuple[Slot, Slot]] = ("B", "A")

#: The range a share lies on, which the bounded test and interval read.
_SHARE_RANGE: Final = (0.0, 1.0)

#: Which side of a pair stored no page: the side that loses the pair, or both, a tie.
MissingPage = Literal["candidate", "control", "both"]

#: How the judge answered one order of a pair.
PairAnswerKind = Literal["scored", "cannot_tell", "failed"]

#: What a missing page stands for, in each order: the score that gives the side with a page the whole pair.
_FORCED: Final[dict[MissingPage, tuple[PairJudgment, PairJudgment]]] = {
    "candidate": (PairJudgment(candidate="A", score=5), PairJudgment(candidate="B", score=1)),
    "control": (PairJudgment(candidate="A", score=1), PairJudgment(candidate="B", score=5)),
    "both": (PairJudgment(candidate="A", score=3), PairJudgment(candidate="B", score=3)),
}


#: Which side is missing, by whether the candidate's page and the control's are absent.
_MISSING: Final[dict[tuple[bool, bool], MissingPage]] = {
    (True, True): "both",
    (True, False): "candidate",
    (False, True): "control",
}


class PairAnswer(EvalBaseModel):
    """The judge's answer on one order of one pair."""

    candidate: Slot = Field(description="The slot the candidate's page sat in: A, read first, or B.")
    answer: PairAnswerKind = Field(description="scored, cannot_tell, or failed (the call raised or did not parse).")
    score: int | None = Field(description="The 1-5 score when scored: 1 is Report A clearly better, 5 Report B.")


class PairwisePair(EvalBaseModel):
    """One case and repeat: the two results paired, and what was read of them."""

    case_id: str = Field(description="The case.")
    repeat: int = Field(description="The repeat of the case, from 1.")
    candidate_result_id: str = Field(description="The candidate arm's result.")
    control_result_id: str = Field(description="The control arm's result.")
    missing: MissingPage | None = Field(
        description="The side that stored no page and so lost the pair, or both, a tie; None when both had a page."
    )
    both_finished: bool = Field(description="Whether both pages were stored finished, so in the finished-only reading.")
    answers: list[PairAnswer] = Field(description="The judge's answers, control first as Report A; none when missing.")


class PairwiseCase(EvalBaseModel):
    """One case's preference: the share the candidate won, and how the judge read its pages."""

    case_id: str = Field(description="The case.")
    pairs: int = Field(description="Its pairs, one per repeat both arms ran.")
    share: float | None = Field(
        description="The candidate's share, 0 to 1, over every pair, a missing page counted; None when nothing scored."
    )
    share_as_a: float | None = Field(description="The candidate's mean share where the judge read it first.")
    share_as_b: float | None = Field(description="The same where the judge read it second.")
    scored: int = Field(description="The judge's answers that gave a score.")
    cannot_tell: int = Field(description="The judge's answers that it cannot tell, in no share.")
    order_consistent: bool | None = Field(
        description="Whether the two orders picked the same side; None unless the judge scored both orders."
    )
    first_slot_bias: float | None = Field(
        description="The judge's lean to Report A, -0.5 to 0.5; None unless it scored both orders."
    )


class PairwiseReading(EvalBaseModel):
    """The verdict on one set of pairs: every pair, or only those whose pages were both stored finished."""

    pairs: int = Field(description="The pairs read.")
    cases: int = Field(description="The cases with a share, each one observation of the test.")
    candidate_share: float | None = Field(description="The candidate's mean share over those cases; None with none.")
    p: float | None = Field(description="The bounded test's two-sided p; None when it cannot run.")
    verdict: Verdict = Field(description="The candidate against the control on report_writer.preferred.")


class PairwiseReliability(EvalBaseModel):
    """How far the judge's preference can be trusted, over the pairs it was asked about."""

    both_orders: int = Field(description="Cases the judge scored in both orders: the consistency rate's denominator.")
    order_consistent: int = Field(description="Of those, the cases whose two orders picked the same side.")
    order_consistency: float | None = Field(description="The share of them; None when no case was scored both ways.")
    one_order: int = Field(description="Cases scored in one order only, left out of the rate.")
    first_slot_bias: float | None = Field(
        description="The mean lean to Report A over the cases scored both ways, -0.5 to 0.5; None with none."
    )
    cannot_tell: int = Field(description="Answers the judge gave that it cannot tell.")
    failed: int = Field(description="Calls that raised or did not parse, in no share.")


class WriterPairwiseReport(EvalBaseModel):
    """What the pairwise pass read of a writer campaign's candidate against its control, and what it spent."""

    campaign_id: str = Field(description="The campaign.")
    candidate: str = Field(description="The candidate arm, by its label.")
    candidate_key: str = Field(description="The candidate arm's variant key.")
    control: str = Field(description="The control arm, by its label.")
    control_key: str = Field(description="The control arm's variant key.")
    judge_model: str = Field(description="The judge asked: the campaign's scoring judge.")
    criterion: str = Field(description="The fingerprint of the preference criterion the judge was asked.")
    judge_family: list[str] = Field(description="The family check for each writer model against the judge.")
    pairs: list[PairwisePair] = Field(description="Every pair, by case then repeat.")
    unpaired: list[str] = Field(description="Results of either arm whose case and repeat the other arm did not run.")
    cases: list[PairwiseCase] = Field(description="Each case's preference, by case.")
    whole: PairwiseReading = Field(description="The reading over every pair.")
    finished_only: PairwiseReading = Field(description="The reading over the pairs whose pages were both finished.")
    reliability: PairwiseReliability = Field(description="Order consistency, first-slot lean and can't-tell.")
    stopped: str | None = Field(
        description="Why the pass stopped before its last pair, when it did; its verdicts are then untested."
    )
    calls_made: int = Field(description="Judge calls made, each ledgered under purpose judge.")
    cost_usd: float | None = Field(description="What they cost together; None when any call went unpriced.")
    cap_usd: float | None = Field(description="The out-of-run cap the calls were admitted under; None when none.")


# --- what the pass reads, before anything is built --------------------------------------------------------------


@dataclass(frozen=True)
class _Side:
    """One arm's result of a pair, and its stored page."""

    result: EvalResult
    evidence: JudgeEvidence | None

    @property
    def finished(self) -> bool:
        """Whether the cell stored its page finished."""
        return self.result.host_measures.get(FINISHED_MEASURE) is True


@dataclass(frozen=True)
class _Pair:
    """A case and repeat both arms ran."""

    case: EvalTestCase
    repeat: int
    candidate: _Side
    control: _Side

    @property
    def missing(self) -> MissingPage | None:
        """The side with no stored page, or both; None when both stored one."""
        gone = (self.candidate.evidence is None, self.control.evidence is None)
        return _MISSING.get(gone)


@dataclass(frozen=True)
class _Collected:
    """What the pass reads of the campaign."""

    campaign_id: str
    candidate: CampaignArm
    control: CampaignArm
    apparatus_class_id: str
    judge_model: str
    template: EvalTemplate
    writer_models: list[str]
    pairs: list[_Pair]
    unpaired: list[str]


def _arm(arms: Sequence[CampaignArm], named: str, campaign_id: str) -> CampaignArm:
    """The arm a caller named, by variant key or label."""
    arm = next((each for each in arms if each.variant_key == named), None) or next(
        (each for each in arms if each.label == named), None
    )
    if arm is None:
        raise ValidationFailedError(
            f"campaign {campaign_id} has no arm {named!r}",
            remedy=f"name one of its arms by label or variant key: {[each.label for each in arms]}",
        )
    return arm


def _keys(arm: CampaignArm) -> frozenset[str]:
    """Every variant key an arm's results may carry under this build."""
    return frozenset({arm.variant_key, *resolve_control_arm(arm).variant_keys})


def _by_case_and_repeat(results: Sequence[EvalResult], arm: CampaignArm) -> dict[tuple[str, int], EvalResult]:
    """An arm's results by case and repeat, refusing two results of one arm on one case and repeat."""
    indexed: dict[tuple[str, int], EvalResult] = {}
    for result in results:
        key = (result.test_case_id, result.k_iteration)
        if key in indexed:
            raise ValidationFailedError(
                f"arm {arm.label} holds two results of case {key[0]} repeat {key[1]} ({indexed[key].id}, {result.id}), "
                "so a pair of it has no one page",
                remedy="take the extra run off the campaign (campaign_runs_remove), then ask again",
            )
        indexed[key] = result
    return indexed


def _collect(host: EvalHost, campaign_id: str, scope_id: str, candidate: str) -> _Collected:
    """Read the campaign, its arms, its results paired by case and repeat, and their pages: blocking, so off the loop.

    :raises NotFoundError: No campaign, a run it names is not stored, or the runs' template is not stored.
    :raises ValidationFailedError: The arms were not scored by one judge on one rubric; the campaign has no control;
        the candidate is none of its arms or is the control; an arm sits under no rig or more than one; an arm ran a
        case and repeat twice; no case and repeat was run by both arms; or a stored page carries a subject.
    """
    storage = host.storage
    campaign = storage.load_campaign(campaign_id, scope_id)
    if campaign is None:
        raise NotFoundError("campaign", campaign_id, remedy=FIND_CAMPAIGN)
    loaded = {run_id: storage.load_eval_run(run_id, scope_id) for run_id in campaign.run_ids}
    if missing := [run_id for run_id, run in loaded.items() if run is None]:
        raise NotFoundError(
            f"campaign {campaign_id}'s run",
            ", ".join(missing),
            remedy="restore the run, or take it off the campaign (campaign_runs_remove), then ask again",
        )
    runs = [run for run in loaded.values() if run is not None]
    refuse_unfixed_scoring_judge(runs)
    judge_model, template_id = runs[0].judge_model, runs[0].template_id
    assert judge_model is not None  # refuse_unfixed_scoring_judge refuses a run that recorded none
    template = storage.load_template(template_id, scope_id) if template_id else None
    if template is None:
        raise NotFoundError("template", template_id or "(none)", remedy=FIND_TEMPLATE)
    bundle = assemble_campaign_bundle(host, campaign_id, scope_id)
    control = next((arm for arm in bundle.arms if arm.role == "control"), None)
    if control is None:
        raise ValidationFailedError(
            f"campaign {campaign_id} names no control, so no page is read against one",
            remedy="name the control arm (campaign_update, control_from_run_id), then ask again",
        )
    arm = _arm(bundle.arms, candidate, campaign_id)
    if arm.variant_key == control.variant_key:
        raise ValidationFailedError(
            f"{arm.label} is the campaign's control, and the pass reads a candidate's pages against it",
            remedy="name a candidate arm",
        )
    candidate_keys, control_keys = _keys(arm), _keys(control)
    rigs = {
        side.label: sorted(
            {cell.apparatus_class_id for aspect in bundle.aspects for cell in aspect.cells if cell.variant_key in keys}
        )
        for side, keys in ((arm, candidate_keys), (control, control_keys))
    }
    if any(len(each) != 1 for each in rigs.values()) or rigs[arm.label] != rigs[control.label]:
        counts = ", ".join(f"{label} under {len(each)}" for label, each in rigs.items())
        raise ValidationFailedError(
            f"a pair compares two pages written under one rig, and the arms were not each measured under the same "
            f"one ({counts})",
            remedy="take the runs of the other rigs off the campaign (campaign_runs_remove), then ask again",
        )
    results = [result for run in runs for result in results_without_payloads(storage, run.id, scope_id)]
    mine = _by_case_and_repeat([result for result in results if result.variant_key in candidate_keys], arm)
    theirs = _by_case_and_repeat([result for result in results if result.variant_key in control_keys], control)
    shared = sorted(mine.keys() & theirs.keys())
    if not shared:
        raise ValidationFailedError(
            f"arms {arm.label} and {control.label} ran no case and repeat in common, so no page has a partner",
            remedy="launch both arms over the same cases and repeats (run_launch), then ask again",
        )
    unpaired = sorted(result.id for key, result in (*mine.items(), *theirs.items()) if key not in shared)
    paired = [result for key in shared for result in (mine[key], theirs[key])]
    pages = storage.load_judge_evidence([result.id for result in paired], scope_id)
    if carrying := sorted(result_id for result_id, page in pages.items() if page.subject is not None):
        raise ValidationFailedError(
            f"the stored evidence of result(s) {', '.join(carrying)} names a subject, so it is not a page that speaks "
            "for itself, and a pair has no place for one",
            remedy="read a report writer campaign, whose pages carry no subject",
        )
    cases = {case.id: case for case in storage.load_test_cases_by_ids(sorted({key[0] for key in shared}), scope_id)}
    if absent := sorted({key[0] for key in shared} - cases.keys()):
        raise NotFoundError("test case", ", ".join(absent), remedy="restore the case, then ask again")
    pairs = [
        _Pair(
            case=cases[case_id],
            repeat=repeat,
            candidate=_Side(mine[case_id, repeat], pages.get(mine[case_id, repeat].id)),
            control=_Side(theirs[case_id, repeat], pages.get(theirs[case_id, repeat].id)),
        )
        for case_id, repeat in shared
    ]
    return _Collected(
        campaign_id=campaign_id,
        candidate=arm,
        control=control,
        apparatus_class_id=rigs[arm.label][0],
        judge_model=judge_model,
        template=template,
        writer_models=sorted({result.model for result in paired}),
        pairs=pairs,
        unpaired=unpaired,
    )


# --- what the judge is asked, priced before any of it is sent --------------------------------------------------


@dataclass
class _Prepared:
    """A pass built and admitted: the judge over budgeted clients, and each pair's two contexts."""

    collected: _Collected
    judge_family: list[str]
    budget: OutOfRunBudget
    service: JudgeService
    #: Per pair, its two orders' contexts (control as A, then candidate as A); None for a pair with a page missing.
    contexts: list[tuple[JudgeContext, JudgeContext] | None]


def _orders(collected: _Collected, pair: _Pair) -> tuple[JudgeContext, JudgeContext] | None:
    """The pair's two orders, control first as Report A, then the candidate; None when a page is missing."""
    mine, theirs = pair.candidate.evidence, pair.control.evidence
    if mine is None or theirs is None:
        return None

    def context(evidence: JudgeEvidence) -> JudgeContext:
        return build_judge_context(
            template=collected.template,
            test_case=pair.case,
            goal_outcomes=[],
            judged_artifact=JudgedArtifact.DOCUMENT,
            judge_evidence=evidence,
        )

    return context(pair_evidence(theirs, mine)), context(pair_evidence(mine, theirs))


async def _prepare(
    host: EvalHost,
    campaign_id: str,
    scope_id: str,
    *,
    candidate: str,
    model_family: ModelFamily,
    out_of_run_cap_usd: float | None,
) -> _Prepared:
    """Collect the campaign, check the judge's family, build the judge over budgeted clients and admit every call.

    :raises ValidationFailedError: Anything :func:`_collect` refuses; the judge is of a writer's family; or the calls
        are priced above the cap or cannot be priced under it. All before any call.
    """
    collected = await run_blocking(host.blocking_executor, _collect, host, campaign_id, scope_id, candidate)
    try:
        judge_family = [
            judge_family_check(model, collected.judge_model, model_family) for model in collected.writer_models
        ]
    except ValidationFailedError as same:
        # The launcher's refusal, whose remedy names a judge to launch with; the pass asks the campaign's own judge.
        raise ValidationFailedError(
            same.message,
            remedy="launch the arms again with a scoring judge of another family (run_launch, judge_model), "
            "and read that campaign",
        ) from same
    clients = host.completion_clients("a writer pairwise pass")
    budget = OutOfRunBudget(
        store=host.storage,
        scope_id=scope_id,
        cap_usd=out_of_run_cap_usd,
        template_id=collected.template.id,
        campaign_id=campaign_id,
        blocking_executor=host.blocking_executor,
    )
    inner = judge_clients_for_run(clients, collected.judge_model)

    def budgeted(model: str | None, temperature: float | None) -> BudgetedJudgeClient:
        return BudgetedJudgeClient(inner(model, temperature), budget, "judge")

    service = JudgeService(client_factory=budgeted, configs={}, failure_describer=host.failure_describer)
    try:
        dim = preference_rubric()
        contexts = [_orders(collected, pair) for pair in collected.pairs]
        by_client: dict[int, tuple[BudgetedJudgeClient, list[PlannedCall]]] = {}
        for orders in contexts:
            for context in orders or ():
                request = service.dimension_request(dim, context)
                client: BudgetedJudgeClient = service.client_for(request)
                by_client.setdefault(id(client), (client, []))[1].extend(
                    [planned_judge_call(request)] * JUDGE_CALL_ATTEMPTS
                )
        for client, calls in by_client.values():
            await refresh_prices(client.inner)
            client.admit(calls)
    except BaseException:
        await service.aclose()
        raise
    return _Prepared(collected=collected, judge_family=judge_family, budget=budget, service=service, contexts=contexts)


# --- the reading ---------------------------------------------------------------------------------------------------


def _answer(candidate: Slot, outcome: JudgeOutcome) -> PairAnswer:
    """One order's answer, from the candidate's side."""
    if outcome.score is not None:
        return PairAnswer(candidate=candidate, answer="scored", score=int(outcome.score.score))
    return PairAnswer(
        candidate=candidate, answer="cannot_tell" if outcome.cannot_tell is not None else "failed", score=None
    )


def _judgments(pair: PairwisePair) -> list[PairJudgment]:
    """What a pair says of the candidate: the judge's answers, or what its missing page stands for."""
    if pair.missing is not None:
        return list(_FORCED[pair.missing])
    return [
        PairJudgment(candidate=answer.candidate, score=answer.score)
        for answer in pair.answers
        if answer.answer != "failed"
    ]


def _verdict(shares: Sequence[float], *, collected: _Collected, stopped: str | None) -> tuple[Verdict, float | None]:
    """The candidate's per-case shares against the control's, as the contrast verdict on the preference, and its p."""
    control = [1.0 - share for share in shares]
    # A pass stopped part-way read some pairs and not others, so no test is run on what it read.
    p = None if stopped is not None else bounded_separation_p(control, shares, paired=True, value_range=_SHARE_RANGE)
    interval = (
        None
        if stopped is not None
        else bounded_difference_interval(
            control, shares, paired=True, value_range=_SHARE_RANGE, confidence=INTERVAL_LEVEL
        )
    )
    delta = (math.fsum(shares) - math.fsum(control)) / len(shares) if shares else None
    arm, against = collected.candidate.label, collected.control.label
    outcome: ContrastOutcome
    reason: VerdictReason
    detail: str | None = None
    if stopped is not None:
        outcome, reason, detail = "untested", "untestable", stopped
    elif p is None or interval is None:
        outcome, reason = "untested", "untestable"
        detail = f"{len(shares)} case{'' if len(shares) == 1 else 's'} with a share, and the test needs two"
    elif p < SIGNIFICANCE_ALPHA and (interval[0] > 0 or interval[1] < 0):
        outcome, reason = ("improved" if interval[0] > 0 else "regressed"), "separated"
    else:
        outcome, reason = "not_separated", "no_margin"
    words = comparison_words(outcome, arm, against)
    if outcome != "untested" and len(shares) < MIN_CASES:
        reason, words = "too_few_cases", f"not tested (n={len(shares)})"
        detail = f"{len(shares)} case{'' if len(shares) == 1 else 's'}, under the {MIN_CASES} a rate needs"
    preferred = preference_rubric()
    verdict = Verdict(
        kind="contrast",
        outcome=outcome,
        reason=reason,
        reason_detail=detail,
        reading="judged",
        name=preferred.name,
        heading=PREFERRED_HEADING,
        arm=arm,
        variant_key=collected.candidate.variant_key,
        apparatus_class_id=collected.apparatus_class_id,
        control=against,
        delta=delta,
        interval=interval,
        interval_level=INTERVAL_LEVEL if interval is not None else None,
        # A family of one: its Holm adjustment is the p itself.
        p_adjusted=p,
        words=words,
    )
    return verdict, p


def _reading(
    pairs: Sequence[PairwisePair], *, collected: _Collected, stopped: str | None
) -> tuple[PairwiseReading, list[PairwiseCase]]:
    """The verdict over ``pairs``, and each case's preference, a missing page counted as the side it leaves."""
    by_case: dict[str, list[PairwisePair]] = {}
    for pair in pairs:
        by_case.setdefault(pair.case_id, []).append(pair)
    cases: list[PairwiseCase] = []
    for case_id, its in sorted(by_case.items()):
        every = [judgment for pair in its for judgment in _judgments(pair)]
        judged = [judgment for pair in its if pair.missing is None for judgment in _judgments(pair)]
        # A case whose every call failed has no judgment at all: no share, and nothing the judge said of it.
        whole: CasePreference | None = case_preference(every) if every else None
        asked: CasePreference | None = case_preference(judged) if judged else None
        cases.append(
            PairwiseCase(
                case_id=case_id,
                pairs=len(its),
                share=None if whole is None else whole.share,
                share_as_a=None if asked is None else asked.share_as_a,
                share_as_b=None if asked is None else asked.share_as_b,
                scored=0 if asked is None else asked.scored,
                cannot_tell=0 if asked is None else asked.cannot_tell,
                order_consistent=None if asked is None else asked.order_consistent,
                first_slot_bias=None if asked is None else asked.first_slot_bias,
            )
        )
    shares = [case.share for case in cases if case.share is not None]
    verdict, p = _verdict(shares, collected=collected, stopped=stopped)
    reading = PairwiseReading(
        pairs=len(pairs),
        cases=len(shares),
        candidate_share=math.fsum(shares) / len(shares) if shares else None,
        p=p,
        verdict=verdict,
    )
    return reading, cases


def _reliability(cases: Sequence[PairwiseCase], pairs: Sequence[PairwisePair]) -> PairwiseReliability:
    """Order consistency over the cases scored both ways, the first-slot lean, and the answers that scored nothing."""
    both = [case for case in cases if case.order_consistent is not None]
    consistent = sum(1 for case in both if case.order_consistent)
    biases = [case.first_slot_bias for case in both if case.first_slot_bias is not None]
    answers = [answer for pair in pairs for answer in pair.answers]
    return PairwiseReliability(
        both_orders=len(both),
        order_consistent=consistent,
        order_consistency=float(Fraction(consistent, len(both))) if both else None,
        one_order=sum(1 for case in cases if case.scored and case.order_consistent is None),
        first_slot_bias=math.fsum(biases) / len(biases) if biases else None,
        cannot_tell=sum(1 for answer in answers if answer.answer == "cannot_tell"),
        failed=sum(1 for answer in answers if answer.answer == "failed"),
    )


async def _run(prepared: _Prepared, *, out_of_run_cap_usd: float | None) -> WriterPairwiseReport:
    """Ask every admitted pair in both orders, then read the shares; the report."""
    collected = prepared.collected
    dim = preference_rubric()
    pairs: list[PairwisePair] = []
    stopped: str | None = None
    for pair, orders in zip(collected.pairs, prepared.contexts, strict=True):
        answers: list[PairAnswer] = []
        if orders is not None and stopped is None:
            for candidate, context in zip(_SLOTS, orders, strict=True):
                outcome = await prepared.service.score_dimension(dim, context)
                answers.append(_answer(candidate, outcome))
                if outcome.owner == "account":
                    stopped = (
                        f"the account behind the judge refused a call on case {pair.case.id} repeat {pair.repeat}, "
                        "so every later call would be refused too"
                    )
                    break
        pairs.append(
            PairwisePair(
                case_id=pair.case.id,
                repeat=pair.repeat,
                candidate_result_id=pair.candidate.result.id,
                control_result_id=pair.control.result.id,
                missing=pair.missing,
                both_finished=pair.candidate.finished and pair.control.finished,
                answers=answers,
            )
        )
    whole, cases = _reading(pairs, collected=collected, stopped=stopped)
    finished_only, _ = _reading([pair for pair in pairs if pair.both_finished], collected=collected, stopped=stopped)
    spends = [*prepared.budget.recorded, *prepared.budget.unrecorded]
    costs = [spend.cost_usd for spend in spends]
    report = WriterPairwiseReport(
        campaign_id=collected.campaign_id,
        candidate=collected.candidate.label,
        candidate_key=collected.candidate.variant_key,
        control=collected.control.label,
        control_key=collected.control.variant_key,
        judge_model=collected.judge_model,
        criterion=fingerprint_criterion(dim),
        judge_family=prepared.judge_family,
        pairs=pairs,
        unpaired=collected.unpaired,
        cases=cases,
        whole=whole,
        finished_only=finished_only,
        reliability=_reliability(cases, pairs),
        stopped=stopped,
        calls_made=len(spends),
        cost_usd=None if None in costs else math.fsum(cost for cost in costs if cost is not None),
        cap_usd=out_of_run_cap_usd,
    )
    log.info(
        "eval.writer_pairwise campaign=%s candidate=%s pairs=%d cases=%d outcome=%s calls=%d stopped=%s",
        report.campaign_id,
        report.candidate,
        len(report.pairs),
        report.whole.cases,
        report.whole.verdict.outcome,
        report.calls_made,
        report.stopped is not None,
    )
    return report


# --- the work, and its start ---------------------------------------------------------------------------------------


async def _writer_pairwise_work(
    host: EvalHost,
    campaign_id: str,
    scope_id: str,
    *,
    candidate: str,
    model_family: ModelFamily,
    out_of_run_cap_usd: float | None,
) -> JobWork[EvalActionJob]:
    """Admit the pairwise pass over a writer campaign's candidate and control: every refusal made, every call priced.

    The call half (the module docstring): the campaign is read and refused here, the judge's family checked, and every
    call the pass can make admitted against ``out_of_run_cap_usd``, so a pass refused has spent nothing. The returned
    work asks each pair in both orders and returns the :class:`WriterPairwiseReport`.

    :param host: The host: the campaign's store, the judge clients, and how a judge call that raised reads.
    :ptype host: EvalHost
    :param campaign_id: The writer campaign.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param candidate: The candidate arm, by its label or its variant key; never the control.
    :ptype candidate: str
    :param model_family: The host's reading of a model's family, which holds the judge to another than each writer's.
    :ptype model_family: ModelFamily
    :param out_of_run_cap_usd: The most the pass's calls may be priced at together; ``None`` when the host enforces no
        out-of-run cap.
    :ptype out_of_run_cap_usd: float | None
    :return: The admitted work; its record is an :class:`~threetears.evals.run.EvalActionJob` under
        :data:`WRITER_PAIRWISE_ACTION`.
    :rtype: JobWork[EvalActionJob]
    :raises NotFoundError: No campaign with that id, or a run, case or template it names is not stored.
    :raises ValidationFailedError: The arms were not scored by one judge on one rubric; the campaign has no control;
        the candidate is none of its arms or is the control; an arm sits under more than one rig or ran a case and
        repeat twice; the arms share no case and repeat; a stored page names a subject; the judge is of a writer's
        family; or the calls are priced above the cap or cannot be priced under it. All before any call.
    :raises ValueError: The host supplies no completion clients.
    """
    prepared = await _prepare(
        host,
        campaign_id,
        scope_id,
        candidate=candidate,
        model_family=model_family,
        out_of_run_cap_usd=out_of_run_cap_usd,
    )

    async def run() -> WriterPairwiseReport:
        async with prepared.service:
            return await _run(prepared, out_of_run_cap_usd=out_of_run_cap_usd)

    return action_job_work(
        scope_id,
        action=WRITER_PAIRWISE_ACTION,
        run=run,
        label=_label(prepared.collected),
        budget_s=WRITER_PAIRWISE_BUDGET_S,
        discard=prepared.service.aclose,
    )


def _label(collected: _Collected) -> str:
    """The job's line: the campaign and the arms read."""
    return f"campaign {collected.campaign_id}: {collected.candidate.label} against {collected.control.label}"


def _places_none(_model: str) -> str | None:
    """A host that places no model's family: every check is recorded as not made."""
    return None


async def writer_pairwise(host: OpsHost, campaign_id: str, scope_id: str, *, candidate: str) -> JobsStarted:
    """Admit the pairwise pass over a writer campaign, then start it as a job, under the host's out-of-run cap.

    Every refusal is made and every call admitted here, with nothing spent; the job makes the calls. The judge's family is
    the host's to tell (:attr:`~threetears.evals.ops.OpsHost.model_family`); a host that tells none has each check
    recorded as not made, as a launch records it.

    :param host: The host: its store, its judge clients, its out-of-run cap and its reading of a model's family.
    :ptype host: OpsHost
    :param campaign_id: The writer campaign.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param candidate: The candidate arm, by its label or its variant key.
    :ptype candidate: str
    :return: The pass's one job (``action:<id>``); polled once it completes, its result is the
        :class:`WriterPairwiseReport`.
    :rtype: JobsStarted
    :raises NotFoundError: No campaign with that id, or a run, case or template it names is not stored.
    :raises ValidationFailedError: The arms were not scored by one judge on one rubric; the campaign has no control;
        the candidate is none of its arms or is the control; an arm sits under more than one rig or ran a case and
        repeat twice; the arms share no case and repeat; a stored page names a subject; the judge is of a writer's
        family; or the calls are priced above the cap or cannot be priced under it.
    """
    family: Callable[[str], str | None] = host.model_family or _places_none
    work = await _writer_pairwise_work(
        host.eval_host,
        campaign_id,
        scope_id,
        candidate=candidate,
        model_family=family,
        out_of_run_cap_usd=host.out_of_run_cap(),
    )
    return await start_job(
        host, AdmittedJob(work=work, kind="action", target_id=WRITER_PAIRWISE_ACTION, label=work.record.label)
    )


#: The engine's ``writer_pairwise`` action starts the pass for a host, and the promotion read
#: (:func:`~threetears.evals.ops.report_writer.writer_promotion.writer_promotion`) reads its report back off the job.
__all__ = ["WriterPairwiseReport", "writer_pairwise"]
