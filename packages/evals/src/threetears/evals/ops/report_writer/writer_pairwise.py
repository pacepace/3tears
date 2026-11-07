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
- **A stopping rule, declared before the first look** (:class:`StoppingRule`): ``stop_when="separated"`` judges the
  cases the rule planned in the bet's fixed order (:func:`~threetears.evals.analysis.stats.bet_order` of their count,
  over their sorted ids), reads the preference after each case as a prefix of that one bet, and stops when it
  separates, either way: valid at any stop (Ville's inequality). ``never``, like no rule, reads every case and its
  verdict is the fixed-n one over them, exactly as with no rule. A rule declared after the arms' first result keeps its
  verdict and says so; cases found that the rule did not plan, or planned and not found, are said, never refused. Each
  look is recorded (:class:`PairwiseStopping`).
- **Batches judged once.** A writer eval that launches its cases in batches runs the pass after each batch, handing
  it the last pass's report (``judged``): a pair it already answered, the same two results, keeps its answers and is
  not asked again, so each pass asks only the new pairs and is admitted under what the last ones left of the cap. Its
  looks are recomputed from every answer, the same prefix of the same bet. ``past_the_stop`` reads every planned case
  past the first separation, for a run that went on to its last planned case to read its guardrails again.
- **How far the judge can be trusted on it.** Order consistency (the two orders pick the same side), the lean to the
  first slot, and the can't-tell count, over the pages it was actually asked about. A case scored in one order only
  is counted apart and left out of the consistency rate, since one order cannot agree or disagree with itself.

It runs as a job (:func:`writer_pairwise`): every refusal and every admission is made in the call, and the job's record
(:class:`~threetears.evals.run.EvalActionJob`) holds the :class:`WriterPairwiseReport`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import Field, field_validator, model_validator

from threetears.evals.analysis.report.blocks import ContrastOutcome, Verdict, VerdictReason
from threetears.evals.analysis.report.verdict import MIN_CASES
from threetears.evals.analysis.report.words import comparison_words
from threetears.evals.analysis.service import assemble_campaign_bundle
from threetears.evals.analysis.stats import (
    INTERVAL_LEVEL,
    PROJECTION_CASE_CAP,
    SIGNIFICANCE_ALPHA,
    bet_order,
    bounded_difference_interval,
    bounded_separation_p,
    smallest_separable_move,
)
from threetears.evals.kernel.capped_completion import refresh_prices
from threetears.evals.kernel.declaration import CampaignArm, aware_instant, resolve_control_arm
from threetears.evals.kernel.numbers import format_number, spell_count
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
    EvalRun,
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


#: What a stopping rule stops on: ``separated``, the first look at which the preference separates, either way; or
#: ``never``, every case read.
StopWhen = Literal["separated", "never"]


class StoppingRule(EvalBaseModel):
    """When the pass stops judging, declared before its first look: what it stops on, and the cases the bet is fixed
    for.

    ``case_ids`` are the held-out cases as declared, and fix the bet under ``separated``: its order
    (:func:`~threetears.evals.analysis.stats.bet_order` of their count, over them sorted) and its stakes (tuned to
    ``max_cases``, their count), so each look is a prefix of one bet whichever of them have run. A set that grows later
    leaves this rule's cases as declared; the next run declares anew. ``never`` fixes no bet: it reads every case the
    pass finds, the fixed-n reading over them.
    """

    stop_when: StopWhen = Field(
        description="separated: stop at the first look the preference separates, either way; never: read every case."
    )
    case_ids: list[str] = Field(min_length=1, description="The cases planned, sorted: the held-out set as declared.")
    max_cases: int = Field(ge=1, description="Their count, N: the cases the bet is fixed for.")
    declared_at: str | None = Field(
        default=None,
        description="When the rule was declared, ISO-8601 with an offset; None for no time, which reads as late.",
    )
    order: list[str] | None = Field(
        default=None,
        description="The planned cases in the order the bet reads them, recorded as declared, so a later read replays "
        "that order whatever derives it then; None for a rule recorded before orders were, whose order is "
        "bet_order(max_cases) over case_ids.",
    )
    batch_size: int | None = Field(
        default=None,
        ge=1,
        description="The cases launched at a time, in the bet's order, before each look; None when every case ran "
        "before the first.",
    )

    @field_validator("case_ids")
    @classmethod
    def _sorted_once(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("a stopping rule lists its planned cases sorted, each once")
        return value

    @model_validator(mode="after")
    def _counted(self) -> StoppingRule:
        if self.max_cases != len(self.case_ids):
            raise ValueError(f"max_cases is the {len(self.case_ids)} cases planned, not {self.max_cases}")
        if self.order is not None and sorted(self.order) != self.case_ids:
            raise ValueError("a stopping rule's order lists its planned cases, each once")
        return self

    @property
    def bet(self) -> list[str]:
        """The planned cases in the order the bet reads them: the order recorded, or, for a rule that recorded none,
        :func:`~threetears.evals.analysis.stats.bet_order` of their count over them."""
        if self.order is not None:
            return list(self.order)
        return [self.case_ids[index] for index in bet_order(self.max_cases)]

    @field_validator("declared_at")
    @classmethod
    def _orderable(cls, value: str | None) -> str | None:
        if value is not None:
            aware_instant(value, "a stopping rule's declaration")
        return value


class PairwiseLook(EvalBaseModel):
    """The preference read after one more case, as a prefix of the one bet: what a running chart draws."""

    cases: int = Field(description="The cases with a share read so far, this one included.")
    case_id: str = Field(description="The case this look added.")
    share: float = Field(description="Its candidate share, 0 to 1.")
    p: float | None = Field(description="The bounded test's two-sided p over the cases so far; None under two.")
    interval: tuple[float, float] | None = Field(
        description="The interval on the candidate's share less the control's, at the pass's level; None under two."
    )
    separated: bool = Field(description="Whether the reading separated here, on the cases a rate needs.")


class PairwiseStopping(EvalBaseModel):
    """How the pass ran against its stopping rule: the rule, when it stopped, each look, and what it must disclose."""

    rule: StoppingRule | None = Field(
        description="The rule as declared; None when none was, and every case was read as a fixed-n reading."
    )
    first_result_at: str | None = Field(
        description="The earliest stored result of either arm: what a rule declared after it is timed against."
    )
    late: bool = Field(
        description="Whether the rule was declared after the first result, or with no time: disclosed, the verdict "
        "standing as read."
    )
    cases_found: int = Field(description="The cases both arms ran, which the pass could judge.")
    absent: list[str] = Field(description="Cases the rule planned that the pass did not find, so never read.")
    unplanned: list[str] = Field(
        description="Cases found that the rule did not plan: read under never, as every case is; under separated the "
        "bet holds no place for them and they are not read."
    )
    stopped_at: int | None = Field(
        description="The cases judged when the rule stopped the pass; None when it read on to its last case."
    )
    first_separated_at: int | None = Field(
        description="The cases read at the first look that separated; None when none did."
    )
    looks: list[PairwiseLook] = Field(description="The preference after each case, in the order the bet read them.")
    stopped_on: Literal["separated", "all_cases", "spend_cap"] | None = Field(
        default=None,
        description="Why the pass read no further: separated, all_cases (every planned case read) or spend_cap (the run's next "
        "batch could pass what the spend cap had left); None for a pass with no rule or one read short otherwise.",
    )
    past_the_stop: bool = Field(
        default=False,
        description="Whether the pass read on past its first separation to the last planned case, because the readings read there "
        "did not decide: a second look at the gated readings and guardrails.",
    )
    line: str = Field(description="What happened, in one sentence, as the page says it.")


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
    reached: bool = Field(
        default=True,
        description="Whether the pass reached the pair's case; False past a stop, where nothing was asked and a "
        "missing page decides nothing.",
    )


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
    stopping: PairwiseStopping | None = Field(
        default=None,
        description="The stopping rule and how the pass ran against it, with each look; None in a report stored "
        "before rules were recorded.",
    )
    outside_run_ids: list[str] = Field(
        default_factory=list,
        description="Runs launched for the campaign and left out of its readings (a batch that stored no page on either "
        "arm), whose pairs this pass read as any pair.",
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
    #: The earliest orderable ``scored_at`` among the paired results; None with none.
    first_result_at: str | None
    #: The runs read beside the campaign's, left out of its readings.
    outside: list[str]


def _first_result_at(results: Sequence[EvalResult]) -> str | None:
    """The earliest of the results' ``scored_at`` that can be ordered, as stored; None with none: a stamp with no
    offset says nothing of when the others became visible."""
    ordered = []
    for result in results:
        try:
            ordered.append((aware_instant(result.scored_at, "a result"), result.scored_at))
        except ValueError:
            # NOSILENT: an unorderable stamp times nothing; the earliest orderable one is what a rule is timed against
            continue
    return min(ordered)[1] if ordered else None


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


def _rig_of(host: EvalHost, runs: Sequence[EvalRun], results: Sequence[EvalResult]) -> dict[str, Any]:
    """Each run's rig, by run id: its provenance and its value on every dimension of the rig the host declares, read as
    the campaign's apparatus classes read them, so two runs compare equal exactly when they share a rig."""
    sweepables = host.profile.sweepables
    names = [declared.name for declared in sweepables.rig_dimensions]
    by_run: dict[str, list[EvalResult]] = {}
    for result in results:
        by_run.setdefault(result.eval_run_id, []).append(result)
    rigs: dict[str, Any] = {}
    for run in runs:
        values = sweepables.read_all(run, by_run.get(run.id, []))
        rigs[run.id] = (run.apparatus_provenance, [values.get(name) for name in names])
    return rigs


def _collect(
    host: EvalHost, campaign_id: str, scope_id: str, candidate: str, outside: Sequence[str] = ()
) -> _Collected:
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
    # Runs the writer eval launched and left out of the campaign's readings (a batch that stored no page on either
    # arm): their pairs are read here as any pair is, and they sit under no rig of the campaign's.
    aside = {run_id: storage.load_eval_run(run_id, scope_id) for run_id in outside if run_id not in loaded}
    if missing := [run_id for run_id, run in aside.items() if run is None]:
        raise NotFoundError(
            "run left out of the campaign", ", ".join(missing), remedy="restore the run, then ask again"
        )
    campaign_runs = [run for run in loaded.values() if run is not None]
    runs = [*campaign_runs, *(run for run in aside.values() if run is not None)]
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
    counts = ", ".join(f"{label} under {len(each)}" for label, each in rigs.items())
    unlike = ValidationFailedError(
        f"a pair compares two pages written under one rig, and the arms were not each measured under the same one "
        f"({counts})",
        remedy="take the runs of the other rigs off the campaign (campaign_runs_remove), then ask again",
    )
    if not rigs[arm.label] or rigs[arm.label] != rigs[control.label]:
        raise unlike
    results = [result for run in runs for result in results_without_payloads(storage, run.id, scope_id)]
    mine = _by_case_and_repeat([result for result in results if result.variant_key in candidate_keys], arm)
    theirs = _by_case_and_repeat([result for result in results if result.variant_key in control_keys], control)
    shared = sorted(mine.keys() & theirs.keys())
    if len(rigs[arm.label]) > 1:
        # A launch made in batches is several runs of each arm, and a batch whose every page went unjudged records no
        # judge, so it reads as a rig of its own. Its pairs are still two pages written under one rig when each pair's
        # two runs read the same rig, as the two arms' runs of one batch do.
        rig = _rig_of(host, runs, results)
        if any(rig.get(mine[key].eval_run_id) != rig.get(theirs[key].eval_run_id) for key in shared):
            raise unlike
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
        first_result_at=_first_result_at(paired),
        outside=sorted(aside),
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
    #: The rule declared before the first look; None when none was.
    rule: StoppingRule | None
    #: The answers an earlier pass gave, by pair index, for the pairs it asked of these same two results.
    carried: dict[int, list[PairAnswer]]
    #: The earlier pass whose answers are carried; None for none.
    judged: WriterPairwiseReport | None
    #: Whether to read every planned case past the first separation.
    past_the_stop: bool
    #: The cases launched so far, for a run launched in batches; None when every case launched at once.
    launched: frozenset[str] | None
    #: Why the run stopped launching at its spend cap, in one sentence; None when it did not.
    out_of_budget: str | None


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
    rule: StoppingRule | None,
    judged: WriterPairwiseReport | None = None,
    past_the_stop: bool = False,
    launched: Collection[str] | None = None,
    out_of_budget: str | None = None,
    outside: Sequence[str] = (),
) -> _Prepared:
    """Collect the campaign, check the judge's family, build the judge over budgeted clients and admit every call
    no earlier pass (``judged``) answered.

    :raises ValidationFailedError: Anything :func:`_collect` refuses; the judge is of a writer's family; or the calls
        are priced above the cap or cannot be priced under it. All before any call.
    """
    collected = await run_blocking(host.blocking_executor, _collect, host, campaign_id, scope_id, candidate, outside)
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
    carried = _carried(collected, judged)
    try:
        dim = preference_rubric()
        contexts = [_orders(collected, pair) for pair in collected.pairs]
        by_client: dict[int, tuple[BudgetedJudgeClient, list[PlannedCall]]] = {}
        for index, orders in enumerate(contexts):
            if index in carried:
                continue
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
    return _Prepared(
        collected=collected,
        judge_family=judge_family,
        budget=budget,
        service=service,
        contexts=contexts,
        rule=rule,
        carried=carried,
        judged=judged,
        past_the_stop=past_the_stop,
        launched=None if launched is None else frozenset(launched),
        out_of_budget=out_of_budget,
    )


def _carried(collected: _Collected, judged: WriterPairwiseReport | None) -> dict[int, list[PairAnswer]]:
    """The answers an earlier pass gave, by the index of each pair of ``collected`` it asked of the same two
    results: a pair it reached with both pages, so its answers stand and it is not asked again."""
    if judged is None:
        return {}
    asked = {
        (pair.case_id, pair.repeat, pair.candidate_result_id, pair.control_result_id): pair.answers
        for pair in judged.pairs
        if pair.reached and pair.missing is None and pair.answers
    }
    carried: dict[int, list[PairAnswer]] = {}
    for index, pair in enumerate(collected.pairs):
        key = (pair.case.id, pair.repeat, pair.candidate.result.id, pair.control.result.id)
        if key in asked:
            carried[index] = list(asked[key])
    return carried


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


def _separation(shares: Sequence[float], planned_n: int | None) -> tuple[float | None, tuple[float, float] | None]:
    """The bounded test's p and interval on the candidate's shares against the control's, one minus each: over the
    shares in case order with no ``planned_n``, and as a prefix of the bet fixed for ``planned_n`` cases, in the order
    bet, with one."""
    control = [1.0 - share for share in shares]
    p = bounded_separation_p(control, shares, paired=True, value_range=_SHARE_RANGE, planned_n=planned_n)
    interval = bounded_difference_interval(
        control, shares, paired=True, value_range=_SHARE_RANGE, confidence=INTERVAL_LEVEL, planned_n=planned_n
    )
    return p, interval


def _separated(p: float | None, interval: tuple[float, float] | None, cases: int) -> bool:
    """Whether a reading separated: its p under α, its interval clear of zero, on the cases a rate needs."""
    return (
        p is not None
        and interval is not None
        and p < SIGNIFICANCE_ALPHA
        and (interval[0] > 0 or interval[1] < 0)
        and cases >= MIN_CASES
    )


def _verdict(
    shares: Sequence[float], *, collected: _Collected, stopped: str | None, planned_n: int | None = None
) -> tuple[Verdict, float | None]:
    """The candidate's per-case shares against the control's, as the contrast verdict on the preference, and its p."""
    control = [1.0 - share for share in shares]
    # A pass stopped part-way read some pairs and not others, so no test is run on what it read.
    p, interval = (None, None) if stopped is not None else _separation(shares, planned_n)
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
    pairs: Sequence[PairwisePair],
    *,
    collected: _Collected,
    stopped: str | None,
    bet: Sequence[str] | None = None,
    planned_n: int | None = None,
) -> tuple[PairwiseReading, list[PairwiseCase]]:
    """The verdict over ``pairs`` the pass reached, and each case's preference, a missing page counted as the side it
    leaves: over the shares in case order, or with a rule, in the order the bet read its cases (``bet``) as a prefix
    of the bet fixed for ``planned_n``."""
    by_case: dict[str, list[PairwisePair]] = {}
    for pair in pairs:
        if pair.reached:
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
    if bet is not None:
        held = {case.case_id: case.share for case in cases if case.share is not None}
        shares = [held[case_id] for case_id in bet if case_id in held]
    verdict, p = _verdict(shares, collected=collected, stopped=stopped, planned_n=planned_n)
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


@dataclass(frozen=True)
class _Plan:
    """The order the pass judges its cases in, and the bet its preference is read as."""

    #: The cases in the order judged.
    judged: list[str]
    #: The cases in the order bet on, under a rule to stop; None otherwise, when the fixed-n reading over the cases
    #: found bets in their own order.
    bet: list[str] | None
    #: The cases the bet is fixed for; None when no bet is fixed.
    planned_n: int | None
    #: Cases the rule planned that the pass did not find.
    absent: list[str]
    #: Cases found that the rule did not plan.
    unplanned: list[str]


def _plan(rule: StoppingRule | None, case_ids: Sequence[str]) -> _Plan:
    """The judging order. Under a rule to stop on separation, the bet's: :func:`~threetears.evals.analysis.stats.bet_order`
    of the planned count over the planned ids, those found, so each look is a prefix of the one bet declared whatever
    order or batch the cases ran in. Otherwise (no rule, or ``never``) the sorted ids found, every one read and no look
    acted on: the fixed-n reading, today's."""
    found = sorted(case_ids)
    planned = [] if rule is None else rule.case_ids
    absent = sorted(set(planned) - set(found))
    unplanned = [] if rule is None else sorted(set(found) - set(planned))
    if rule is None or rule.stop_when == "never":
        return _Plan(judged=found, bet=None, planned_n=None, absent=absent, unplanned=unplanned)
    present = set(found)
    bet = [case_id for case_id in rule.bet if case_id in present]
    return _Plan(judged=bet, bet=bet, planned_n=rule.max_cases, absent=absent, unplanned=unplanned)


def _looks(cases: Sequence[PairwiseCase], plan: _Plan) -> list[PairwiseLook]:
    """The preference after each case with a share, in the order the bet reads them: the declared bet under a rule to
    stop, or else the bet the fixed-n reading over the shares makes, which says where it would have stopped."""
    held = {case.case_id: case.share for case in cases if case.share is not None}
    if plan.bet is not None:
        sequence = [(case_id, held[case_id]) for case_id in plan.bet if case_id in held]
        planned_n = plan.planned_n
    else:
        found = sorted(held.items())
        sequence = [found[index] for index in bet_order(len(found))]
        planned_n = len(found)
    looks: list[PairwiseLook] = []
    for k in range(1, len(sequence) + 1):
        shares = [share for _, share in sequence[:k]]
        p, interval = _separation(shares, planned_n) if k >= 2 else (None, None)
        looks.append(
            PairwiseLook(
                cases=k,
                case_id=sequence[k - 1][0],
                share=sequence[k - 1][1],
                p=p,
                interval=interval,
                separated=_separated(p, interval, k),
            )
        )
    return looks


async def _run(prepared: _Prepared, *, out_of_run_cap_usd: float | None) -> WriterPairwiseReport:
    """Ask the admitted pairs in both orders, case by case in the plan's order and up to the rule's stop, then read
    the shares; the report.

    The stop reads the primary alone. Every arm has run every case launched before the pass starts, so the gated
    readings and guardrails are read once, on every case launched, by the promotion read: one fixed-n read at the stop.
    A pair an earlier pass answered keeps its answers and is not asked again.
    """
    collected = prepared.collected
    rule = prepared.rule
    dim = preference_rubric()
    by_case: dict[str, list[int]] = {}
    for index, pair in enumerate(collected.pairs):
        by_case.setdefault(pair.case.id, []).append(index)
    plan = _plan(rule, list(by_case))
    answers: dict[int, list[PairAnswer]] = {}
    reached: list[str] = []
    stopped: str | None = None
    stop_at: int | None = None
    for case_id in plan.judged:
        if stopped is not None or stop_at is not None:
            break
        reached.append(case_id)
        for index in by_case[case_id]:
            pair, orders = collected.pairs[index], prepared.contexts[index]
            if index in prepared.carried:
                answers[index] = list(prepared.carried[index])
                continue
            asked = answers.setdefault(index, [])
            if orders is None or stopped is not None:
                continue
            for candidate, context in zip(_SLOTS, orders, strict=True):
                outcome = await prepared.service.score_dimension(dim, context)
                asked.append(_answer(candidate, outcome))
                if outcome.owner == "account":
                    stopped = (
                        f"the account behind the judge refused a call on case {pair.case.id} repeat {pair.repeat}, "
                        "so every later call would be refused too"
                    )
                    break
        if plan.bet is None or stopped is not None:
            continue
        _, cases = _reading(
            _pairs(collected, answers, reached),
            collected=collected,
            stopped=None,
            bet=plan.bet,
            planned_n=plan.planned_n,
        )
        looks = _looks(cases, plan)
        if looks and looks[-1].separated and not prepared.past_the_stop:
            stop_at = len(reached)
    pairs = _pairs(collected, answers, reached)
    whole, cases = _reading(pairs, collected=collected, stopped=stopped, bet=plan.bet, planned_n=plan.planned_n)
    finished_only, _ = _reading([pair for pair in pairs if pair.both_finished], collected=collected, stopped=stopped)
    looks = _looks(cases, plan)
    stopping = _stopping(
        collected,
        rule,
        plan,
        looks,
        cases_found=len(by_case),
        reached=len(reached),
        stop_at=stop_at,
        past_the_stop=prepared.past_the_stop,
        launched=prepared.launched,
        out_of_budget=prepared.out_of_budget,
    )
    spends = [*prepared.budget.recorded, *prepared.budget.unrecorded]
    costs: list[float | None] = [spend.cost_usd for spend in spends]
    earlier = prepared.judged
    if earlier is not None:
        # The earlier passes' calls are this report's too: their answers are in it.
        costs.append(earlier.cost_usd)
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
        reliability=_reliability(cases, [pair for pair in pairs if pair.reached]),
        stopped=stopped,
        stopping=stopping,
        outside_run_ids=collected.outside,
        calls_made=len(spends) + (0 if earlier is None else earlier.calls_made),
        cost_usd=None if None in costs else math.fsum(cost for cost in costs if cost is not None),
        cap_usd=out_of_run_cap_usd,
    )
    log.info(
        "eval.writer_pairwise campaign=%s candidate=%s pairs=%d cases=%d outcome=%s calls=%d stopped=%s stop_at=%s",
        report.campaign_id,
        report.candidate,
        len(report.pairs),
        report.whole.cases,
        report.whole.verdict.outcome,
        report.calls_made,
        report.stopped is not None,
        stop_at,
    )
    return report


def _pairs(collected: _Collected, answers: dict[int, list[PairAnswer]], reached: Collection[str]) -> list[PairwisePair]:
    """Every pair, by case then repeat, with what the judge answered of it and whether the pass reached its case."""
    return [
        PairwisePair(
            case_id=pair.case.id,
            repeat=pair.repeat,
            candidate_result_id=pair.candidate.result.id,
            control_result_id=pair.control.result.id,
            missing=pair.missing,
            both_finished=pair.candidate.finished and pair.control.finished,
            answers=answers.get(index, []),
            reached=pair.case.id in reached,
        )
        for index, pair in enumerate(collected.pairs)
    ]


def _late(rule: StoppingRule | None, first_result_at: str | None) -> bool:
    """Whether a declared rule came after the arms' first result, or carries no time: disclosed, never a demotion."""
    if rule is None or first_result_at is None:
        return False
    if rule.declared_at is None:
        return True
    return aware_instant(rule.declared_at, "a stopping rule's declaration") > aware_instant(first_result_at, "a result")


def _stopping(
    collected: _Collected,
    rule: StoppingRule | None,
    plan: _Plan,
    looks: Sequence[PairwiseLook],
    *,
    cases_found: int,
    reached: int,
    stop_at: int | None,
    past_the_stop: bool = False,
    launched: frozenset[str] | None = None,
    out_of_budget: str | None = None,
) -> PairwiseStopping:
    """The record of how the pass ran against its rule, with the sentence the page prints."""
    late = _late(rule, collected.first_result_at)
    first = next((look.cases for look in looks if look.separated), None)
    line = _line(collected, rule, looks, late=late, reached=reached, stop_at=stop_at, past_the_stop=past_the_stop)
    # Launched in batches, the planned cases past the last batch were never launched, and the launch says how many
    # ran; a launched case the pass did not find is said, as any planned case not found is.
    absent = plan.absent if launched is None else [case_id for case_id in plan.absent if case_id in launched]
    if absent or plan.unplanned:
        line += f" {_differs(plan, absent=absent, read=rule is not None and rule.stop_when == 'never')}"
    stopped_on: Literal["separated", "all_cases", "spend_cap"] | None = None
    if out_of_budget is not None:
        stopped_on, line = "spend_cap", f"{line} {out_of_budget}"
    elif stop_at is not None:
        stopped_on = "separated"
    elif rule is not None and reached >= rule.max_cases:
        stopped_on = "all_cases"
    return PairwiseStopping(
        rule=rule,
        first_result_at=collected.first_result_at,
        late=late,
        cases_found=cases_found,
        absent=plan.absent,
        unplanned=plan.unplanned,
        stopped_at=stop_at,
        first_separated_at=first,
        looks=list(looks),
        stopped_on=stopped_on,
        past_the_stop=past_the_stop,
        line=line,
    )


def _differs(plan: _Plan, *, absent: Sequence[str], read: bool) -> str:
    """How the cases found differ from those the rule planned, in one sentence: disclosed, never refused."""
    parts = []
    if absent:
        parts.append(
            f"{len(absent)} case{'' if len(absent) == 1 else 's'} the rule planned {'was' if len(absent) == 1 else 'were'} not found"
        )
    if plan.unplanned:
        count = len(plan.unplanned)
        parts.append(
            f"{count} found {'was' if count == 1 else 'were'} not planned, "
            + (
                "read as every case is"
                if read
                else "and the bet holds no place for it, so it was not read"
                if count == 1
                else "and the bet holds no place for them, so they were not read"
            )
        )
    return "; ".join(parts).capitalize() + "."


def _line(
    collected: _Collected,
    rule: StoppingRule | None,
    looks: Sequence[PairwiseLook],
    *,
    late: bool,
    reached: int,
    stop_at: int | None,
    past_the_stop: bool = False,
) -> str:
    """What happened against the rule, in one plain sentence."""
    first = next((look for look in looks if look.separated), None)
    if rule is None:
        where = "no look separated" if first is None else f"the preference first separated at case {first.cases}"
        return f"Read all {reached} cases with no stopping rule declared: {where}."
    timed = (
        "the rule declared before the first result"
        if not late
        else "a rule declared after the first result, its verdict standing as read"
        if rule.declared_at is not None
        else "a rule declared with no time, its verdict standing as read"
    )
    if stop_at is not None and first is not None:
        candidate = first.interval is not None and first.interval[0] > 0
        side = collected.candidate.label if candidate else collected.control.label
        won = sum(1 for look in looks[: first.cases] if (look.share > 0.5 if candidate else look.share < 0.5))
        return (
            f"Stopped at {reached} of {rule.max_cases} cases under {timed}: the judge preferred {side} on {won} of "
            f"{first.cases}, separated."
        )
    # Under never no rule fixed the bet, so a count off the plan prints alone, never as more read than planned.
    of = str(reached) if rule.stop_when == "never" and reached != rule.max_cases else f"{reached} of {rule.max_cases}"
    if first is None:
        return f"Read {of} cases under {timed}: every look up to {len(looks)} was consistent with no preference."
    if past_the_stop:
        return (
            f"Read {of} cases under {timed}: the preference first separated at case {first.cases}, and the run went "
            "on to the last case because its guardrails were not shown held there, so its gated readings and guardrails had "
            "a second look."
        )
    return f"Read {of} cases under {timed}: the preference first separated at case {first.cases}."


@dataclass(frozen=True)
class WriterPrior:
    """An earlier run of a writer eval on the same cases, as far as the next run's resolution line reads it: the cases it ran and what
    its promotion read projected would settle its primary.

    :ivar case_names: The case files it ran on its side, by name.
    :ivar cases_to_settle: The held-out cases like these its read projected would settle a primary that did not
        separate (``WriterPromotion.cases_to_settle``); ``None`` when it projected none.
    """

    case_names: frozenset[str]
    cases_to_settle: int | None


def resolution_floor(cases: int) -> float | None:
    """The smallest candidate share, held on every case, that the pass's own test separates at ``cases`` cases: its
    resolution, said before a launch and on the page so a gate asked a question its cases cannot answer says so.

    The gate's bounded test inverted (:func:`~threetears.evals.analysis.stats.smallest_separable_move`), not a normal
    approximation: 0.7232 at 11 cases and 0.5903 at 25. A preference with any spread across cases needs more.

    :param cases: The held-out cases the pass will read.
    :ptype cases: int
    :return: The share, above 0.5 and at most 1; None when even a candidate preferred outright on every case would
        not separate at this count.
    :rtype: float | None
    """
    move = smallest_separable_move(cases, value_range=_SHARE_RANGE)
    # The test reads the candidate's share less the control's, one minus it: a share s moves it by 2s - 1.
    return None if move is None else (1.0 + move) / 2.0


def resolution_line(cases: int, *, projected: int | None = None, fewest: int | None = None) -> str:
    """The resolution floor (:func:`resolution_floor`) as one line, a disclosure and never a refusal; with the
    projection the last run on these same cases made, when the caller has one.

    :param cases: The held-out cases the pass will read.
    :ptype cases: int
    :param projected: The held-out cases like these the last run on these cases projected would settle its primary
        (its ``cases_to_settle``); None for none.
    :ptype projected: int | None
    :param fewest: The fewest cases the caller asks a run to read; a run under it says so in the same line (``At 7
        cases, under the 10 this eval asks for, this run can ...``), and still runs: a floor is a disclosure.
    :ptype fewest: int | None
    :return: ``At 11 cases this run can separate only a preference of 73% or more, held on every case; the last run
        on these cases projected 43 to settle.``
    :rtype: str
    """
    floor = resolution_floor(cases)
    at = f"At {cases} case{'' if cases == 1 else 's'}"
    if fewest is not None and cases < fewest:
        at = f"{at}, under the {fewest} this eval asks for,"
    if floor is None:
        head = f"{at} this run cannot separate any preference, even one held on every case"
    else:
        # Rounded up, so the line never claims a share the test would not separate.
        percent = math.ceil(floor * 100 - 1e-9)
        head = f"{at} this run can separate only a preference of {format_number(percent)}% or more, held on every case"
    if projected is None:
        return f"{head}."
    settle = (
        f"more than {spell_count(PROJECTION_CASE_CAP)}" if projected > PROJECTION_CASE_CAP else spell_count(projected)
    )
    return f"{head}; the last run on these cases projected {settle} to settle."


# --- the work, and its start ---------------------------------------------------------------------------------------


async def _writer_pairwise_work(
    host: EvalHost,
    campaign_id: str,
    scope_id: str,
    *,
    candidate: str,
    model_family: ModelFamily,
    out_of_run_cap_usd: float | None,
    stopping_rule: StoppingRule | None = None,
    judged: WriterPairwiseReport | None = None,
    past_the_stop: bool = False,
    launched: Collection[str] | None = None,
    out_of_budget: str | None = None,
    outside: Sequence[str] = (),
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
    :param stopping_rule: The rule declared before the first look (:class:`StoppingRule`); ``None`` reads every case,
        the fixed-n reading over them.
    :ptype stopping_rule: StoppingRule | None
    :param judged: The last pass's report over this campaign's earlier batches: its answers on the same two results
        are kept and not asked again, and its spend comes off ``out_of_run_cap_usd``; ``None`` asks every pair.
    :ptype judged: WriterPairwiseReport | None
    :param past_the_stop: Read every planned case past the first separation (the module docstring).
    :ptype past_the_stop: bool
    :param launched: The cases launched so far, for a run launched in batches: a planned case not launched yet is not
        said as not found, and a launched one is; ``None`` when every case launched at once.
    :ptype launched: Collection[str] | None
    :param out_of_budget: Why the run stopped launching at its spend cap, in one sentence, which the record carries;
        ``None`` when it did not.
    :ptype out_of_budget: str | None
    :param outside: Runs launched for this campaign and left out of its readings (a batch that stored no page on
        either arm), whose pairs are read as any pair is; the report names them.
    :ptype outside: Sequence[str]
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
    left = out_of_run_cap_usd
    if left is not None and judged is not None and judged.cost_usd is not None:
        # One cap over every batch's pass: what the earlier passes spent is gone from it.
        left = max(0.0, left - judged.cost_usd)
    prepared = await _prepare(
        host,
        campaign_id,
        scope_id,
        candidate=candidate,
        model_family=model_family,
        out_of_run_cap_usd=left,
        rule=stopping_rule,
        judged=judged,
        past_the_stop=past_the_stop,
        launched=launched,
        out_of_budget=out_of_budget,
        outside=outside,
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


async def writer_pairwise(
    host: OpsHost,
    campaign_id: str,
    scope_id: str,
    *,
    candidate: str,
    stopping_rule: StoppingRule | None = None,
    judged: WriterPairwiseReport | None = None,
    past_the_stop: bool = False,
    launched: Collection[str] | None = None,
    out_of_budget: str | None = None,
    outside: Sequence[str] = (),
) -> JobsStarted:
    """Admit the pairwise pass over a writer campaign, then start it as a job, under the host's out-of-run cap.

    Every refusal is made and every call admitted here, with nothing spent; the job makes the calls. The judge's family is
    the host's to tell (:attr:`~threetears.evals.ops.OpsHost.model_family`); a host that tells none has each check
    recorded as not made, as a launch records it. Every call is admitted whatever the rule, since where a pass stops
    is not known until it does.

    :param host: The host: its store, its judge clients, its out-of-run cap and its reading of a model's family.
    :ptype host: OpsHost
    :param campaign_id: The writer campaign.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param candidate: The candidate arm, by its label or its variant key.
    :ptype candidate: str
    :param stopping_rule: The rule declared before the first look, which the report records; ``None`` reads every
        case. A rule declared after the arms' first result is read as it stands and disclosed, never refused.
    :ptype stopping_rule: StoppingRule | None
    :param judged: The last pass's report over this campaign's earlier batches, whose answers on the same two results
        stand and whose spend comes off the cap; ``None`` asks every pair.
    :ptype judged: WriterPairwiseReport | None
    :param past_the_stop: Read every planned case past the first separation, for a run gone on to its last case.
    :ptype past_the_stop: bool
    :param launched: The cases launched so far, for a run launched in batches; ``None`` when every case launched at
        once.
    :ptype launched: Collection[str] | None
    :param out_of_budget: Why the run stopped launching at its spend cap, which the record carries; ``None`` when not.
    :ptype out_of_budget: str | None
    :param outside: Runs launched for this campaign and left out of its readings, whose pairs are read as any pair is.
    :ptype outside: Sequence[str]
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
        stopping_rule=stopping_rule,
        judged=judged,
        past_the_stop=past_the_stop,
        launched=launched,
        out_of_budget=out_of_budget,
        outside=outside,
    )
    return await start_job(
        host, AdmittedJob(work=work, kind="action", target_id=WRITER_PAIRWISE_ACTION, label=work.record.label)
    )


#: The engine's ``writer_pairwise`` action starts the pass for a host, and the promotion read
#: (:func:`~threetears.evals.ops.report_writer.writer_promotion.writer_promotion`) reads its report back off the job.
__all__ = [
    "PairwiseStopping",
    "StopWhen",
    "StoppingRule",
    "WriterPairwiseReport",
    "WriterPrior",
    "resolution_line",
    "writer_pairwise",
]
