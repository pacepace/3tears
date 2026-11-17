"""A host's door onto the blind read: draw one over a campaign's judged outputs, serve its sheet, take answers, have
a declared model read it, and close it.

The second door onto the one core (:mod:`threetears.evals.analysis.blind_read`); the quick path's files are the
first. Nothing here draws, blinds, samples or reads a preference itself: it loads what the core needs from the host's
store, calls the core, and stores what the core returns (a :class:`~threetears.evals.schema.BlindRead`, durable).

**The draw takes no seed.** A caller cannot draw again until an order suits it: the seed is derived from the
population (its digest), the arms, and how many reads were drawn over that population before, and the line says how
many were.

**Who may answer.** A person's answer is recorded as theirs only when the host attests the caller relaying it for
that person (``EvalHost.attests_person``, the check every person's record passes, :func:`~threetears.evals.ops.runs.attested_person`);
otherwise it is refused, 403, and nothing is written. A model's answers enter a read only from the read's own review
job (:func:`keep_review`, called inside the job :func:`blind_read_review` starts, whose id the read records before
the job runs), for the one model the read declared when it was drawn. No action takes a model's answer.

**What a rater is shown gives nothing away.** The sheet (:func:`blind_read_sheet`) carries content, criteria and
slots: no arm, run, result or case id, no judge score, no label key, and no count of who preferred what. A read's line
carries no check, preference or rating until it is closed.

**Close** (:func:`blind_read_close`) stores what the core read, conditionally on the read as loaded, and refuses
every later answer; only then does it write each answer as a rating through :func:`~threetears.evals.run.rate_result`
(a person's as the person's, naming the read and the caller that relayed it; a model's as an agent's, counted nowhere
as a person's), so no rating exists for a read still open. A rating write that fails reaches the caller, and closing
again writes what is missing. A read closed before any rater answered every planned item is closed early, and its
preference is read by the anytime test.

These are the catalogue's operations (``blind_read_create`` and its siblings), called by name through the action
catalogue: none is exported from :mod:`threetears.evals.ops`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Final, Literal

from pydantic import Field

from threetears.evals.analysis.blind_read import (
    BLIND_READ_SEED,
    agreed_pairs,
    blind_sheet,
    check_answer,
    close_blind_read,
    draw_blind_read,
    read_preference,
    read_sample,
    shares_family,
)
from threetears.evals.analysis.report.blind_rates import StatedRate, blind_rates
from threetears.evals.analysis.service import assemble_campaign_bundle
from threetears.evals.kernel.declaration import CampaignArm, resolve_control_arm
from threetears.evals.kernel.errors import (
    ConflictError,
    EvalServiceError,
    NotFoundError,
    StorageError,
    ValidationFailedError,
)
from threetears.evals.kernel.host.eval_host import EvalHost
from threetears.evals.kernel.capped_completion import refresh_prices
from threetears.evals.kernel.job_context import running_job
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.out_of_run import OutOfRunBudget
from threetears.evals.kernel.remedies import FIND_CAMPAIGN
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.jobs import ACTION_JOB_PREFIX, AdmittedJob, JobsStarted, job_poll, start_job
from threetears.evals.ops.runs import attested_person
from threetears.evals.run.blind_review import REVIEWER_PURPOSE, answer_score, answer_sheet, plan_review
from threetears.evals.run.job_work import DEFAULT_ACTION_JOB_BUDGET_S, action_job_work
from threetears.evals.run.ratings import rate_result
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.blind_read import (
    BlindAgreement,
    BlindAnswer,
    BlindArm,
    BlindAsk,
    BlindPreferenceAnswer,
    BlindRead,
    BlindReview,
    BlindReviewerCheck,
    BlindSheet,
    BlindVerdict,
    BlindVerdictTable,
)
from threetears.evals.schema.logs import get_logger
from threetears.evals.schema.models import (
    RESERVED_DIM_IDS,
    CalibrationRating,
    EvalResult,
    EvalRun,
    RubricDim,
    RubricScale,
)
from threetears.evals.schema.reading_states import Maker, SampleKind, StateOf

log = get_logger(__name__)

#: Where a caller finds a read's id.
FIND_BLIND_READ: Final = "blind_reads_list names this scope's blind reads"

#: What the action that keeps a model's answers is called, as its job records it.
REVIEW_ACTION: Final = "blind_read_review"

#: A change to a read, applied to it as stored.
_Apply = Callable[[BlindRead], BlindRead]

#: How often a write that lost its race to another answer is read again and applied again before it is refused.
_WRITE_ATTEMPTS: Final = 3

#: What a caller relaying a person's answer does when the host vouches for no person at all.
_UNCHECKED_REMEDY: Final = (
    "a host that relays people's answers is built with attests_person=<its check>; until then a blind read is "
    "answered on the quick path's page"
)

#: What a caller relaying a person's answer does when the host does not attest it for that person.
_REFUSED_REMEDY: Final = "relay the answer as a caller the host attests for that person"


# --- what the door hands back ----------------------------------------------------------------------------------


class BlindReadRater(EvalBaseModel):
    """One rater's progress through a read: who, a person or a model, and whether what they answered is a random
    sample."""

    rater: str = Field(description="The person as the host names them, or the reviewer model's id.")
    made_by: Maker = Field(description="A person, or a model.")
    sample: SampleKind = Field(
        description="random: the first items of the drawn order. selected: any other set. none: nothing answered."
    )
    answered: int = Field(ge=0, description="The items answered, skips and part-answers included.")
    skipped: int = Field(ge=0, description="How many of them were skipped.")
    partial: int = Field(ge=0, description="How many were answered in part.")


class BlindReadRate(EvalBaseModel):
    """A rate read off a check's counts: of ``of`` answers, ``count``, with its exact 95% interval.

    Exact (Clopper-Pearson): it covers the true rate at least 95% of the time at any count, the few answers a reader
    fails included. The answers are counted as independent; two results of one pair item share a case, which
    ``cases`` on the check states.
    """

    count: int = Field(ge=0)
    of: int = Field(ge=1)
    interval: tuple[float, float] = Field(description="The exact (Clopper-Pearson) 95% interval on count / of.")
    said: str = Field(description="The rate in words, with its count and interval.")


class BlindReadCheck(EvalBaseModel):
    """How the judge's scores stood, on one criterion, against the people who answered a closed read, or against its
    model as one served model: counts, and the rates read off them.

    No kappa, bar, tier or estimate of results still needed (#1208): where one side gave a single verdict a kappa is
    0 or undefined at any agreement, and a read's population is fixed. ``said`` states, in words, where one side gave
    a single verdict and where no rate exists.
    """

    dim: str = Field(description="The criterion.")
    scale: RubricScale = Field(description="pass_fail, or ordinal (1-5).")
    judge_model: str | None = Field(description="The model that served the judge's scores; None where unrecorded.")
    made_by: Maker = Field(description="Whose answers the judge was read against: people's, or a model's.")
    raters: list[str] = Field(description="The people, sorted; or the one model, by id.")
    served_model: str | None = Field(
        description="A model's check: the model that served every answer counted here. None for people."
    )
    results: int = Field(ge=0, description="The distinct results read.")
    cases: int = Field(ge=0, description="The distinct cases those results answer.")
    random_results: int = Field(ge=0, description="How many of the pairs read came from a random sample.")
    agreed: int = Field(ge=0, description="The pairs where the reader and the judge gave the same score.")
    pairs: int = Field(ge=1, description="The (answer, judge score) pairs read.")
    table: BlindVerdictTable = Field(
        description="Every pair, counted by the reader's score (rows) and the judge's (columns): the confusion counts "
        "on pass/fail, the full cross-table on 1-5."
    )
    catch: BlindReadRate | None = Field(
        description="Pass/fail: of the answers the reader failed, how many the judge failed too. None where the reader "
        "failed none, or on 1-5."
    )
    false_fail: BlindReadRate | None = Field(
        description="Pass/fail: of the answers the reader passed, how many the judge failed (the miss the other way). "
        "None where the reader passed none, or on 1-5."
    )
    shares_family: bool = Field(description="A model's check only: the model and the judge name one provider.")
    said: list[str] = Field(
        default_factory=list,
        description="Plain statements: a side that gave a single verdict, and a rate that does not exist, with why.",
    )


class BlindReadReviewerCheck(EvalBaseModel):
    """How the read's reviewer model stood against people on the outputs both answered, as counts: the people are the
    reader, the reviewer sits where the judge would."""

    dim: str = Field(description="The criterion.")
    reviewer: str = Field(description="The reviewer model's id.")
    served_model: str | None = Field(description="The model that served the answers counted; None where unrecorded.")
    agreed: int = Field(ge=0, description="The outputs where the person and the reviewer gave the same score.")
    outputs: int = Field(ge=1, description="The outputs both answered on this criterion.")
    table: BlindVerdictTable = Field(description="Every pair, by the person's score (rows) and the reviewer's.")
    catch: BlindReadRate | None = Field(
        description="Pass/fail: of the answers people failed, how many the reviewer failed too; None where none."
    )
    false_fail: BlindReadRate | None = Field(
        description="Pass/fail: of the answers people passed, how many the reviewer failed; None where none."
    )
    said: list[str] = Field(default_factory=list, description="Plain statements, as a check's.")


class BlindReadPreference(EvalBaseModel):
    """Which of a pair the people (together) or a model preferred over a closed read, and whether that separates."""

    made_by: Maker
    raters: list[str]
    served_model: str | None = Field(
        description="A model's preference: the model that served every answer it is read over; one preference per "
        "served model, never pooled. None for people, and where the served model was not recorded."
    )
    control: str = Field(description="The control arm, by its key.")
    candidate: str = Field(description="The candidate arm, by its key.")
    planned: int = Field(ge=0, description="The items the preference was planned over, fixed at the draw.")
    answered: int = Field(ge=0, description="How many of those the rater furthest along answered.")
    items: int = Field(ge=0, description="The pair items carrying a preference.")
    prefer_candidate: int = Field(ge=0)
    prefer_control: int = Field(ge=0)
    ties: int = Field(ge=0)
    missing: int = Field(ge=0, description="Planned pair items answered with no preference: skipped or left blank.")
    random: bool = Field(description="Whether every rater behind it answered a random sample.")
    test: str = Field(
        description="exact_sign (one rater), bounded_share (several), or anytime (a read closed early: valid at any "
        "stop)."
    )
    p: float | None
    verdict: Literal["improved", "regressed", "not_separated"] | None = Field(
        description="The candidate shown better or worse, or not told apart; None where not read, and state says why."
    )
    state: StateOf | None = Field(description="Why it was not read, where it was not.")


class BlindReadLine(EvalBaseModel):
    """A blind read as its door reads it: what was drawn, who has answered how far, and once closed what it found.

    It never names a result, and while the read is open it carries no check, preference or rating.
    """

    read_id: str
    name: str
    campaign_id: str | None
    arms: list[BlindArm] = Field(description="The control (and candidate) drawn over, each with its runs.")
    criteria: list[str] = Field(description="The criteria asked, by dimension.")
    ask: list[BlindAsk]
    reviewer: str | None = Field(description="The model declared to read it, by id; None for none.")
    review_job_id: str | None = Field(description="The review's job, to poll with job_poll; None until started.")
    cases: int = Field(ge=0, description="The cases in the drawn order: one item each.")
    planned_items: int = Field(ge=0, description="The items the preference is planned over.")
    excluded: dict[str, int] = Field(description="Cases left out of the order or its pairs, by reason.")
    earlier_reads: int = Field(
        ge=0, description="How many blind reads over the same outputs and arms were drawn before this one."
    )
    raters: list[BlindReadRater] = Field(description="Each rater so far, with their sample.")
    closed_at: str | None
    closed_by: str | None
    closed_early: bool = Field(description="Closed before any rater answered every planned item.")
    checks: list[BlindReadCheck] = Field(default_factory=list, description="Once closed: the judge's checks.")
    reviewer_checks: list[BlindReadReviewerCheck] = Field(
        default_factory=list, description="Once closed: the reviewer model against people, on what both answered."
    )
    not_checked: dict[str, list[str]] = Field(
        default_factory=dict, description="Once closed: each model whose answers check nothing, with why."
    )
    preferences: list[BlindReadPreference] = Field(default_factory=list, description="Once closed: the preferences.")
    reviews: list[BlindReview] = Field(default_factory=list, description="Each model reading: its calls and cost.")
    ratings_written: int = Field(default=0, ge=0, description="On the close's own line: the ratings it wrote.")
    ratings_not_written: list[str] = Field(
        default_factory=list, description="On the close's own line: each rating it did not write, with why."
    )


class BlindReadListing(EvalBaseModel):
    """The blind reads in a scope, oldest first."""

    reads: list[BlindReadLine]


# --- reading the store -----------------------------------------------------------------------------------------


def _loaded(host: EvalHost, read_id: str, scope_id: str) -> tuple[BlindRead, str]:
    """The read and its etag.

    :raises NotFoundError: No blind read with that id in the scope.
    """
    read, etag = host.storage.load_blind_read_with_etag(read_id, scope_id)
    if read is None or etag is None:
        raise NotFoundError("blind read", read_id, remedy=FIND_BLIND_READ)
    return read, etag


def _refuse_closed(read: BlindRead, what: str) -> None:
    """:raises ConflictError: The read is closed."""
    if read.closed_at is not None:
        raise ConflictError(
            f"blind read {read.id} was closed at {read.closed_at}, so it takes no {what}",
            remedy="draw another read over the same outputs (blind_read_create)",
        )


def _same_population(read: BlindRead, other: BlindRead) -> bool:
    return other.draw.population_digest == read.draw.population_digest and sorted(
        arm.key for arm in other.arms
    ) == sorted(arm.key for arm in read.arms)


def _earlier(reads: Sequence[BlindRead], read: BlindRead) -> int:
    """How many of ``reads`` were drawn over ``read``'s population and arms before it."""
    return sum(
        1
        for other in reads
        if other.id != read.id and _same_population(read, other) and other.created_at <= read.created_at
    )


def _line(read: BlindRead, earlier: int, **close: object) -> BlindReadLine:
    keys = list(dict.fromkeys((answer.rater, answer.made_by) for answer in read.answers))
    raters = []
    for rater, kind in keys:
        sample = read_sample(read, rater, kind)
        raters.append(
            BlindReadRater(
                rater=rater,
                made_by=kind,
                sample=sample.sample,
                answered=sample.answered,
                skipped=sample.skipped,
                partial=sample.partial,
            )
        )
    report = read.report if read.closed_at is not None else None
    checks = [_check(entry) for entry in (report.agreement if report is not None else [])]
    preferences = [
        BlindReadPreference(
            made_by=entry.anchor,
            raters=list(entry.anchor_names),
            served_model=entry.served_model,
            control=entry.control,
            candidate=entry.candidate,
            planned=entry.planned,
            answered=entry.answered,
            items=entry.items,
            prefer_candidate=entry.prefer_candidate,
            prefer_control=entry.prefer_control,
            ties=entry.ties,
            missing=entry.missing,
            random=entry.random,
            test=entry.test,
            p=entry.p,
            verdict=entry.verdict,
            state=entry.state,
        )
        # Its p and verdict are read off its stored counts here, by the computation a close reads them by (#1236).
        for entry in (read_preference(stored) for stored in (report.preferences if report is not None else []))
    ]
    return BlindReadLine(
        read_id=read.id,
        name=read.name,
        campaign_id=read.campaign_id,
        arms=list(read.arms),
        criteria=[dim.name for dim in read.criteria],
        ask=list(read.ask),
        reviewer=read.reviewer,
        review_job_id=f"{ACTION_JOB_PREFIX}{read.review_job_id}" if read.review_job_id is not None else None,
        cases=len(read.items),
        planned_items=read.draw.planned_items,
        excluded=dict(read.draw.excluded),
        earlier_reads=earlier,
        raters=raters,
        closed_at=read.closed_at,
        closed_by=read.closed_by,
        closed_early=read.closed_early,
        checks=checks,
        reviewer_checks=[_reviewer_check(check) for check in (report.reviewer_checks if report is not None else [])],
        not_checked=dict(report.not_checked) if report is not None else {},
        preferences=preferences,
        reviews=list(read.reviews),
        **close,  # type: ignore[arg-type]
    )


def _rate(rate: StatedRate | None) -> BlindReadRate | None:
    """A rate as the door serves it, in the words every door says it in (:func:`blind_rates`)."""
    if rate is None:
        return None
    return BlindReadRate(count=rate.count, of=rate.of, interval=rate.interval, said=rate.said)


def _rates(
    table: BlindVerdictTable, reader: str, judge: str
) -> tuple[BlindReadRate | None, BlindReadRate | None, list[str]]:
    """The catch rate and the false-fail rate read off a stored table, and the plain statements beside them.

    ``reader`` and ``judge`` are how the statements name the two sides.
    """
    rates = blind_rates(table, reader=reader, judge=judge)
    return _rate(rates.catch), _rate(rates.false_fail), rates.said


def _check(entry: BlindAgreement) -> BlindReadCheck:
    """A stored agreement as the door serves it: its counts, and the rates read off them on load."""
    catch, false_fail, said = _rates(entry.table, "the reader", "the judge")
    return BlindReadCheck(
        dim=entry.dim,
        scale=entry.scale,
        judge_model=entry.judge_model,
        made_by=entry.anchor,
        raters=list(entry.anchor_names),
        served_model=entry.served_model,
        results=entry.results,
        cases=entry.cases,
        random_results=entry.random_pairs,
        agreed=agreed_pairs(entry.table),
        pairs=entry.pairs,
        table=entry.table,
        catch=catch,
        false_fail=false_fail,
        shares_family=entry.anchor == "model" and shares_family(entry.anchor_names[0], entry.judge_model),
        said=said,
    )


def _reviewer_check(check: BlindReviewerCheck) -> BlindReadReviewerCheck:
    """A stored reviewer check as the door serves it: counts, and the rates read off them on load."""
    catch, false_fail, said = _rates(check.table, "people", "the reviewer")
    return BlindReadReviewerCheck(
        dim=check.dim,
        reviewer=check.reviewer,
        served_model=check.served_model,
        agreed=agreed_pairs(check.table),
        outputs=check.outputs,
        table=check.table,
        catch=catch,
        false_fail=false_fail,
        said=said,
    )


def _line_of(host: EvalHost, read: BlindRead, **close: object) -> BlindReadLine:
    return _line(read, _earlier(host.storage.query_blind_reads(read.scope_id), read), **close)


# --- the draw --------------------------------------------------------------------------------------------------


def _arm_keys(arm: CampaignArm) -> frozenset[str]:
    """Every variant key an arm's results may carry under this build."""
    return frozenset({arm.variant_key, *resolve_control_arm(arm).variant_keys})


def _named_arm(arms: Sequence[CampaignArm], named: str, campaign_id: str) -> CampaignArm:
    """The arm a caller named, by variant key or label.

    :raises ValidationFailedError: None of the campaign's arms.
    """
    arm = next((each for each in arms if each.variant_key == named), None) or next(
        (each for each in arms if each.label == named), None
    )
    if arm is None:
        raise ValidationFailedError(
            f"campaign {campaign_id} has no arm {named!r}",
            remedy=f"name one of its arms by label or variant key: {[each.label for each in arms]}",
        )
    return arm


def _runs(host: EvalHost, run_ids: Sequence[str], scope_id: str) -> list[EvalRun]:
    """:raises NotFoundError: A run named is not stored."""
    runs: list[EvalRun] = []
    for run_id in dict.fromkeys(run_ids):
        run = host.storage.load_eval_run(run_id, scope_id)
        if run is None:
            raise NotFoundError("run", run_id, remedy="runs_list names this scope's runs")
        runs.append(run)
    return runs


def _campaign_arms(
    host: EvalHost, campaign_id: str, scope_id: str, candidate: str | None
) -> tuple[list[BlindArm], list[EvalResult]]:
    """The campaign's control (and the named candidate) as arms, each with the runs holding its results.

    :raises NotFoundError: No such campaign, or a run it names is gone.
    :raises ValidationFailedError: It names no control; the candidate is none of its arms, or is the control; a run
        holds results of both arms.
    """
    campaign = host.storage.load_campaign(campaign_id, scope_id)
    if campaign is None:
        raise NotFoundError("campaign", campaign_id, remedy=FIND_CAMPAIGN)
    bundle = assemble_campaign_bundle(host, campaign_id, scope_id)
    control = next((arm for arm in bundle.arms if arm.role == "control"), None)
    if control is None:
        raise ValidationFailedError(
            f"campaign {campaign_id} names no control, so a blind read has no arm to draw over",
            remedy="name the control arm (campaign_update, control_from_run_id), or draw over runs (run_ids)",
        )
    chosen: list[tuple[CampaignArm, Literal["control", "candidate"]]] = [(control, "control")]
    if candidate is not None:
        arm = _named_arm(bundle.arms, candidate, campaign_id)
        if arm.variant_key == control.variant_key:
            raise ValidationFailedError(
                f"{arm.label} is the campaign's control, and a pair read is a candidate against it",
                remedy="name a candidate arm, or none for a read of the control's outputs alone",
            )
        chosen.append((arm, "candidate"))
    results: list[EvalResult] = []
    runs_of: dict[str, list[str]] = {arm.variant_key: [] for arm, _ in chosen}
    for run in _runs(host, campaign.run_ids, scope_id):
        held = host.storage.query_eval_results_by_run(run.id, scope_id)
        holders = {arm.variant_key for arm, _ in chosen if any(r.variant_key in _arm_keys(arm) for r in held)}
        if len(holders) > 1:
            raise ValidationFailedError(
                f"run {run.id} holds results of both arms, so its outputs cannot be told apart by arm",
                remedy="draw over runs that each hold one arm (run_ids), or split the run's arms into runs of their own",
            )
        if holders:
            (key,) = holders
            runs_of[key].append(run.id)
            results.extend(held)
    arms = []
    for arm, role in chosen:
        if not runs_of[arm.variant_key]:
            raise ValidationFailedError(
                f"arm {arm.label} of campaign {campaign_id} has no run holding its results",
                remedy="add the arm's runs to the campaign (campaign_runs_add), then draw",
            )
        arms.append(BlindArm(key=arm.variant_key, role=role, run_ids=runs_of[arm.variant_key]))
    return arms, results


def _run_arm(host: EvalHost, run_ids: Sequence[str], scope_id: str) -> tuple[list[BlindArm], list[EvalResult]]:
    """Runs of one arm as the read's one arm, keyed by the variant key their results share.

    :raises NotFoundError: A run is not stored.
    :raises ValidationFailedError: Their results carry more than one variant key, or none.
    """
    results: list[EvalResult] = []
    for run in _runs(host, run_ids, scope_id):
        results.extend(host.storage.query_eval_results_by_run(run.id, scope_id))
    keys = sorted({result.variant_key for result in results})
    if len(keys) != 1:
        raise ValidationFailedError(
            f"the runs hold results of {len(keys)} arms, and a read over runs is over one",
            remedy="name runs of one arm, or the campaign and its candidate (campaign_id, candidate) for a pair read",
        )
    return [BlindArm(key=keys[0], role="control", run_ids=list(dict.fromkeys(run_ids)))], results


def _criteria(
    host: EvalHost,
    runs: Sequence[EvalRun],
    results: Sequence[EvalResult],
    scope_id: str,
    named: Sequence[str] | None,
) -> list[RubricDim]:
    """The criteria asked, as the runs' one template words them, or else the scope's criteria catalogue.

    Named, those; else every criterion the results were judged on.

    :raises ValidationFailedError: The runs ran more than one template; a criterion is worded by neither; or nothing
        was judged.
    :raises NotFoundError: The template is gone.
    """
    judged = {score.dim for result in results for score in result.rubric_scores if score.dim not in RESERVED_DIM_IDS}
    templates = sorted({run.template_id for run in runs if run.template_id})
    if len(templates) > 1:
        raise ValidationFailedError(
            f"the read's runs ran {len(templates)} templates, and a read asks one rubric's criteria",
            remedy="draw over runs of one template",
        )
    rubric: dict[str, RubricDim] = {}
    if templates:
        template = host.storage.load_template(templates[0], scope_id)
        if template is None:
            raise NotFoundError("template", templates[0], remedy="templates_list names this scope's templates")
        rubric = {dim.name: dim for dim in template.rubric}
    wanted = list(dict.fromkeys(named)) if named is not None else sorted(judged)
    found: list[RubricDim] = []
    for name in wanted:
        dim = rubric.get(name)
        if dim is None and (catalogued := host.storage.load_active_rubric_dim(name, scope_id)) is not None:
            dim = catalogued.dim
        if dim is None:
            raise ValidationFailedError(
                f"neither the runs' template nor the scope's criteria word {name!r}, so a rater has no question",
                remedy=f"name criteria the runs were judged on and the template words: {sorted(rubric) or sorted(judged)}",
            )
        found.append(dim)
    if not found:
        raise ValidationFailedError(
            "the read's runs were judged on no criterion, so a rater has nothing to answer",
            remedy="judge the runs, then draw",
        )
    return found


def _seed(digest: str, arms: Sequence[BlindArm], earlier: int) -> int:
    """The seed of the next read over a population: its digest, its arms and how many were drawn over it before."""
    keys = "|".join(sorted(f"{arm.role}:{arm.key}" for arm in arms))
    return int(hashlib.sha256(f"blind-read-seed|{digest}|{keys}|{earlier}".encode()).hexdigest()[:15], 16)


def blind_read_create(
    host: EvalHost,
    scope_id: str,
    *,
    created_by: str,
    campaign_id: str | None = None,
    run_ids: Sequence[str] | None = None,
    candidate: str | None = None,
    criteria: Sequence[str] | None = None,
    planned_items: int | None = None,
    name: str | None = None,
    reviewer: str | None = None,
) -> BlindReadLine:
    """Draw a blind read over a campaign's control (and a candidate), or over runs of one arm, and store it.

    No seed is taken: it is derived from the population, the arms and how many reads were drawn over them before,
    which the line states (``earlier_reads``). A reviewer model is declared here or never, and is refused when it is
    the judge's own model, whose answers would repeat the judge.

    :param host: The host whose store holds the runs.
    :param scope_id: The caller's scope.
    :param created_by: Who drew it.
    :param campaign_id: The campaign whose control (and candidate) the read is over; or None with ``run_ids``.
    :param run_ids: Runs of one arm, for a read of their outputs alone; or None with ``campaign_id``.
    :param candidate: The candidate arm of the campaign, by label or variant key, for a pair read; None for singles.
    :param criteria: The criteria asked, by dimension; None for every criterion of the runs' rubric.
    :param planned_items: The items the preference is read over; None for the population, up to 30.
    :param name: What the read is called; None names it after the campaign or the runs.
    :param reviewer: A model to read the sheet as a reference reviewer, by id; None for none.
    :return: The read as stored.
    :raises ValidationFailedError: Neither or both of ``campaign_id`` and ``run_ids``; a candidate without a
        campaign; the arms or criteria cannot be read (see the helpers); no case where every arm has a judged answer
        with stored evidence; the reviewer is the judge's own model.
    :raises NotFoundError: A campaign, run or template named is not stored.
    :raises ConflictError: Another read over the same population was drawn at the same moment; ask again.
    """
    if (campaign_id is None) == (run_ids is None):
        raise ValidationFailedError(
            "a blind read is drawn over a campaign (campaign_id) or over runs (run_ids), one of the two",
            remedy="name the campaign, and a candidate for a pair read; or the runs of one arm",
        )
    if campaign_id is None and candidate is not None:
        raise ValidationFailedError(
            "a candidate is an arm of a campaign, and the read names none",
            remedy="name the campaign (campaign_id) with the candidate",
        )
    if campaign_id is not None:
        arms, results = _campaign_arms(host, campaign_id, scope_id, candidate)
    else:
        assert run_ids is not None
        arms, results = _run_arm(host, run_ids, scope_id)
    runs = _runs(host, [run_id for arm in arms for run_id in arm.run_ids], scope_id)
    asked = _criteria(host, runs, results, scope_id, criteria)
    judges = {run.judge_model for run in runs} | {
        score.served_model
        for result in results
        for score in result.rubric_scores
        if score.dim in {d.name for d in asked}
    }
    if reviewer is not None and reviewer in judges:
        raise ValidationFailedError(
            f"{reviewer} is the judge's own model, so its answers would repeat the judge and check nothing",
            remedy="declare a reviewer of another model, or none",
        )
    mine = [result for result in results if any(result.eval_run_id in arm.run_ids for arm in arms)]
    evidence = host.storage.load_judge_evidence([result.id for result in mine], scope_id)
    label = name or (f"blind read of campaign {campaign_id}" if campaign_id else "blind read")

    def draw(seed: int) -> BlindRead:
        return draw_blind_read(
            mine,
            evidence,
            asked,
            arms,
            scope_id=scope_id,
            created_by=created_by,
            name=label,
            seed=seed,
            campaign_id=campaign_id,
            planned_items=planned_items,
        )

    probe = draw(BLIND_READ_SEED)
    if not probe.items:
        raise ValidationFailedError(
            "no case has a judged answer with stored evidence from every arm on every criterion asked, so there is "
            f"nothing to read (left out: {probe.draw.excluded or 'no case at all'})",
            remedy="judge the runs on the criteria asked, or ask fewer criteria",
        )
    stored = host.storage.query_blind_reads(scope_id)
    earlier = sum(1 for other in stored if _same_population(probe, other))
    read = draw(_seed(probe.draw.population_digest, arms, earlier))
    if reviewer is not None:
        read = read.model_copy(update={"reviewer": reviewer})
    host.storage.save_blind_read(read, create_only=True)
    log.info("eval.blind_read_create read=%s scope=%s by=%s cases=%d", read.id, scope_id, created_by, len(read.items))
    return _line(read, earlier)


# --- the sheet and a person's answer ---------------------------------------------------------------------------


def blind_read_sheet(host: EvalHost, read_id: str, scope_id: str, *, rater: str | None = None) -> BlindSheet:
    """The read's sheet: what a rater is shown, and nothing else. No arm, run, result or case id, no judge score, no
    label key, and nothing of what anyone answered but the named rater's own progress.

    :param host: The host whose store holds the read.
    :param read_id: The read.
    :param scope_id: The caller's scope.
    :param rater: Whose progress to give (``answered``, ``next``), as their answers name them; None for none.
    :return: The sheet.
    :raises NotFoundError: No such read.
    """
    read, _ = _loaded(host, read_id, scope_id)
    return blind_sheet(read, rater=rater, label_keys=False)


def blind_read_answer(
    host: EvalHost,
    read_id: str,
    scope_id: str,
    *,
    collected_by: str,
    person: str,
    item: str,
    verdicts: Mapping[str, str] | None = None,
    prefer: BlindPreferenceAnswer | None = None,
    skipped: bool = False,
    reason: str = "",
) -> BlindReadLine:
    """Record a person's answer to one item, relayed by a caller the host attests for them.

    The host is asked first (``EvalHost.attests_person``); without its word nothing is read or written. The person's
    earlier answer to the same item is replaced. A model's answer never comes through here (:func:`keep_review`).

    :param host: The host whose check attests the person.
    :param read_id: The read.
    :param scope_id: The caller's scope.
    :param collected_by: The calling identity relaying the answer, as the host resolved it.
    :param person: Whose answer it is, in the caller's spelling.
    :param item: The item answered, as the sheet names it.
    :param verdicts: A verdict per criterion per output shown, keyed ``<slot>/<criterion>`` (``A/answer.grounded``),
        each the answer as the sheet's levels word it: pass or fail, or 1 to 5. None for a skip.
    :param prefer: Of a pair: the better slot, or tie.
    :param skipped: The person passed on the item; then no verdict or preference.
    :param reason: The person's own words for the answer, where given.
    :return: The read as stored, with the person's progress; no check, preference or tally while it is open.
    :raises EvalServiceError: Status 403: the host supplies no ``attests_person``, or does not attest this caller for
        this person; nothing is written.
    :raises NotFoundError: No such read.
    :raises ConflictError: The read is closed; or other answers kept landing as this one was written.
    :raises ValidationFailedError: The answer is not one the item asks.
    """
    canonical = attested_person(
        host,
        scope_id=scope_id,
        collected_by=collected_by,
        person=person,
        operation="blind_read_answer",
        what="blind-read answer",
        unchecked_remedy=_UNCHECKED_REMEDY,
        refused_remedy=_REFUSED_REMEDY,
    )

    def apply(read: BlindRead) -> BlindRead:
        _refuse_closed(read, "answer")
        try:
            answer = BlindAnswer(
                item=item,
                rater=canonical,
                made_by="person",
                collected_by=collected_by,
                verdicts=_verdicts(read, verdicts or {}, reason),
                prefer=prefer,
                skipped=skipped,
            )
            check_answer(read, answer)
        except ValueError as unasked:
            # A pydantic ValidationError is a ValueError too: a skip carrying an answer is refused the same way.
            raise ValidationFailedError(
                str(unasked), remedy="answer what the item's sheet asks, or skip it"
            ) from unasked
        kept = [
            each
            for each in read.answers
            if not (each.item == item and each.rater == canonical and each.made_by == "person")
        ]
        return read.model_copy(update={"answers": [*kept, answer]})

    return _line_of(host, _rewrite(host, read_id, scope_id, apply))


def _verdicts(read: BlindRead, labels: Mapping[str, str], reason: str) -> list[BlindVerdict]:
    """Each ``<slot>/<criterion>`` answer as a verdict, read on the criterion's scale as a labels file is.

    :raises ValueError: A key that is not ``<slot>/<criterion>`` of the read, or an answer its scale does not hold.
    """
    scales = {dim.name: dim.scale for dim in read.criteria}
    verdicts: list[BlindVerdict] = []
    for key, label in labels.items():
        slot, _, dim = key.partition("/")
        scale = scales.get(dim)
        if slot not in ("A", "B") or scale is None:
            raise ValueError(f"verdict {key!r} is not <slot>/<criterion> of this read's criteria ({sorted(scales)})")
        score = answer_score(str(label).strip().lower(), scale)
        if score is None:
            raise ValueError(f"verdict {key!r} answers {label!r}, which the {scale} scale of {dim!r} does not hold")
        verdicts.append(BlindVerdict(slot="A" if slot == "A" else "B", dim=dim, score=score, reason=reason))
    return verdicts


def _rewrite(host: EvalHost, read_id: str, scope_id: str, apply: _Apply) -> BlindRead:
    """Read the read, apply ``apply`` and write it back conditionally, again on a lost race, a few times.

    :raises ConflictError: Every attempt lost its race.
    """
    for attempt in range(_WRITE_ATTEMPTS):
        read, etag = _loaded(host, read_id, scope_id)
        changed = apply(read)
        try:
            host.storage.save_blind_read(changed, if_match=etag)
        except ConflictError:
            if attempt + 1 == _WRITE_ATTEMPTS:
                raise
            continue
        return changed
    raise AssertionError("unreachable: the last attempt returns or raises")


# --- a model's reading -----------------------------------------------------------------------------------------


def keep_review(
    host: EvalHost, read_id: str, scope_id: str, answers: Sequence[BlindAnswer], review: BlindReview
) -> BlindRead:
    """Keep a model's answers on the read: only inside the read's own review job, and only the declared reviewer's.

    Called by the job :func:`blind_read_review` starts. The job it runs in (:func:`~threetears.evals.kernel.running_job`)
    must be the one the read recorded when the review was started; anything else is refused and nothing is written,
    so no caller can put a model's answer on a read by any other door.

    :param host: The host whose store holds the read.
    :param read_id: The read.
    :param scope_id: Its scope.
    :param answers: The model's answers, each named by the reviewer.
    :param review: The reading: its calls, cost and what went unanswered.
    :return: The read as stored.
    :raises EvalServiceError: Status 403: not inside the read's review job, or an answer is not the declared
        reviewer's; nothing is written.
    :raises NotFoundError: No such read.
    :raises ConflictError: The read was closed while the model read it.
    """
    job = running_job()

    def apply(read: BlindRead) -> BlindRead:
        if job is None or read.review_job_id is None or job != read.review_job_id:
            log.warning(
                "eval.blind_read_review refused job=%s read=%s scope=%s: not the read's review job",
                job,
                read_id,
                scope_id,
            )
            raise EvalServiceError(
                f"blind read {read_id} takes a model's answers only from its own review job, and this is "
                f"{'no job' if job is None else f'job {job}'}",
                status_code=403,
                details={"read_id": read_id},
                remedy="have the read's declared reviewer read it (blind_read_review)",
            )
        if stray := sorted(
            {answer.rater for answer in answers if answer.made_by != "model" or answer.rater != read.reviewer}
        ):
            raise EvalServiceError(
                f"blind read {read_id} declared {read.reviewer} as its reviewer, and answers name {', '.join(stray)}",
                status_code=403,
                details={"read_id": read_id},
                remedy="have the read's declared reviewer read it (blind_read_review)",
            )
        _refuse_closed(read, "model's answers")
        for answer in answers:
            check_answer(read, answer)
        given = {answer.item for answer in answers}
        kept = [
            each
            for each in read.answers
            if not (each.item in given and each.rater == read.reviewer and each.made_by == "model")
        ]
        return read.model_copy(update={"answers": [*kept, *answers], "reviews": [*read.reviews, review]})

    return _rewrite(host, read_id, scope_id, apply)


async def blind_read_review(host: OpsHost, read_id: str, scope_id: str) -> JobsStarted:
    """Have the read's declared reviewer model answer what it has not, as a job: priced and admitted first.

    Each unanswered item is one call (:func:`~threetears.evals.run.blind_review.plan_review`), admitted together
    against the host's out-of-run cap under the purpose ``reviewer``; a cap that cannot pay refuses the call and
    nothing is spent. The read records the job before it starts, and the job's answers are kept only from inside it
    (:func:`keep_review`); a job that fails to start gives the read back its earlier job, and the failure reaches the
    caller, so a later review can start. ``job_poll`` on the job returns the read's line once it ends.

    :param host: The host whose clients and job manager serve the review.
    :param read_id: The read.
    :param scope_id: The caller's scope.
    :return: The started job.
    :raises NotFoundError: No such read.
    :raises ConflictError: The read is closed, or its review is already running.
    :raises ValidationFailedError: The read declared no reviewer, the reviewer has answered every item, or the cap
        cannot pay for the calls.
    :raises StorageError: The job's record could not be written; nothing started, and the read is as it was.
    """
    eval_host = host.eval_host
    read, _ = await run_blocking(eval_host.blocking_executor, _loaded, eval_host, read_id, scope_id)
    _refuse_closed(read, "review")
    if read.reviewer is None:
        raise ValidationFailedError(
            f"blind read {read_id} declared no reviewer, and one is declared when a read is drawn, never after",
            remedy="draw a read with a reviewer (blind_read_create, reviewer)",
        )
    if (
        read.review_job_id is not None
        and not (await job_poll(host, f"{ACTION_JOB_PREFIX}{read.review_job_id}", scope_id)).done
    ):
        raise ConflictError(
            f"blind read {read_id}'s review is running as job {ACTION_JOB_PREFIX}{read.review_job_id}",
            remedy="poll it (job_poll) and ask again once it ends",
        )
    answered = {answer.item for answer in read.answers if answer.made_by == "model"}
    todo = {item.item for item in read.items} - answered
    if not todo:
        raise ValidationFailedError(
            f"{read.reviewer} has answered every item of blind read {read_id}",
            remedy="close the read (blind_read_close)",
        )
    sheet = blind_sheet(read, label_keys=False)
    planned = plan_review(sheet, todo)
    budget = OutOfRunBudget(
        store=eval_host.storage,
        scope_id=scope_id,
        cap_usd=host.out_of_run_cap(),
        campaign_id=read.campaign_id,
        blocking_executor=eval_host.blocking_executor,
    )
    client = eval_host.completion_clients("a blind read's reviewer")("judge", read.reviewer)
    try:
        await refresh_prices(client)
        admitted = await run_blocking(
            eval_host.blocking_executor, budget.admit, client, REVIEWER_PURPOSE, list(planned.values())
        )
    # prawduct:ok-broad-except — not handled: a client that admitted nothing is released, and what refused is re-raised
    except BaseException:
        await client.aclose()
        raise
    calls = dict(zip(planned, admitted, strict=True))
    reviewer = read.reviewer

    async def made() -> BlindReadLine:
        try:
            answers, review = await answer_sheet(sheet, client, budget, calls, reused=len(answered))
        finally:
            await client.aclose()
        # Named by the model the read declared: the client may name an alias of it, and its served model is kept.
        named = [answer.model_copy(update={"rater": reviewer}) for answer in answers]
        kept = await run_blocking(
            eval_host.blocking_executor,
            keep_review,
            eval_host,
            read_id,
            scope_id,
            named,
            review.model_copy(update={"reviewer": reviewer}),
        )
        return await run_blocking(eval_host.blocking_executor, _line_of, eval_host, kept)

    label = f"{reviewer} reads blind read {read.name}"
    work = action_job_work(
        scope_id,
        action=REVIEW_ACTION,
        run=made,
        label=label,
        budget_s=DEFAULT_ACTION_JOB_BUDGET_S,
        discard=client.aclose,
    )

    def claim(current: BlindRead) -> BlindRead:
        _refuse_closed(current, "review")
        if current.review_job_id != read.review_job_id:
            raise ConflictError(
                f"blind read {read_id}'s review was started by another call as this one was",
                remedy="poll its job (blind_read_get names it) and ask again once it ends",
            )
        return current.model_copy(update={"review_job_id": work.record_id})

    try:
        await run_blocking(eval_host.blocking_executor, _rewrite, eval_host, read_id, scope_id, claim)
    # prawduct:ok-broad-except — not handled: the admitted client is released, and what refused is re-raised
    except BaseException:
        await client.aclose()
        raise

    def release(current: BlindRead) -> BlindRead:
        # The claim this call made, and no later one, goes back to the job before it.
        if current.review_job_id != work.record_id:
            return current
        return current.model_copy(update={"review_job_id": read.review_job_id})

    try:
        return await start_job(host, AdmittedJob(work=work, kind="action", target_id=REVIEW_ACTION, label=label))
    # prawduct:ok-broad-except — not handled: a job that did not start releases its claim, and what refused is re-raised
    except BaseException:
        # A claim on a job with no record would refuse every later review of the read (job_poll finds no such job).
        await run_blocking(eval_host.blocking_executor, _rewrite, eval_host, read_id, scope_id, release)
        raise


# --- close, and reading back -----------------------------------------------------------------------------------


def _unwritten(host: EvalHost, ratings: Sequence[CalibrationRating], scope_id: str) -> list[CalibrationRating]:
    """Those of a closed read's ``ratings`` the store does not hold: none is stored under the rating's id given at the
    same time or later (a later one is the rater's own, which stands)."""
    stored: dict[str, dict[str, CalibrationRating]] = {}
    pending: list[CalibrationRating] = []
    for rating in ratings:
        if rating.result_id not in stored:
            stored[rating.result_id] = {
                each.id: each for each in host.storage.query_calibration_ratings(scope_id, result_id=rating.result_id)
            }
        held = stored[rating.result_id].get(rating.id)
        if held is None:
            pending.append(rating)
            continue
        mine, theirs = _instant(rating.rated_at), _instant(held.rated_at)
        if mine is not None and theirs is not None:
            behind = theirs < mine
        else:
            behind = held.rated_at != rating.rated_at
        if behind:
            pending.append(rating)
    return pending


def _instant(stamp: str) -> datetime | None:
    """``stamp`` as an aware instant; None where it is not ISO-8601 with an offset (then it is compared as written)."""
    try:
        instant = datetime.fromisoformat(stamp)
    except ValueError:
        # NOSILENT: a stamp that is not ISO-8601 is an expected input, not a fault: the caller compares it as written
        return None
    return instant if instant.tzinfo is not None else None


class _RatingsUnwritten(StorageError):
    """The store refused a closed read's rating: a store failure whose remedy names the call that finishes the close."""

    def __init__(self, message: str):
        EvalServiceError.__init__(
            self,
            message=message,
            status_code=503,
            remedy="close the read again (blind_read_close) once the store answers: it writes the ratings still missing",
        )


def blind_read_close(host: EvalHost, read_id: str, scope_id: str, *, closed_by: str) -> BlindReadLine:
    """Close the read: store what the core read, refuse every later answer, then write its answers as ratings.

    The conditional save of the closed read decides the close: an answer that lands after the read was loaded
    refuses it (ask again), and no rating is written for a read still open. Only then is each verdict written through
    :func:`~threetears.evals.run.rate_result`, which reads the result and binds the rating to what was rated: a
    person's as the person's (naming the read and the caller that relayed it), a model's as an agent's. A rating not
    written (its result gone, or the person's own later rating stands) is listed with why, never dropped.

    A rating write that fails (the store) reaches the caller with the read closed. Closing it again writes the ratings
    the first close did not; a closed read whose ratings are all stored, or none of whose missing ones can be written,
    refuses a second close. A read closed before any rater answered every planned item is closed early, and its
    preference is the anytime test's.

    :param host: The host whose store holds the read and its results.
    :param read_id: The read.
    :param scope_id: The caller's scope.
    :param closed_by: Who closed it.
    :return: The closed read, with what it found and the ratings written.
    :raises NotFoundError: No such read.
    :raises ConflictError: It is already closed with every rating stored; or closed, and each rating still missing
        is refused (listed with why, in the message and ``details["not_written"]``); or an answer landed while it was
        being closed (ask again).
    :raises StorageError: The store refused a rating; the read is closed, the store's failure is the cause, and the
        remedy is to close it again, which writes the ratings still missing.
    """
    read, etag = _loaded(host, read_id, scope_id)
    result_ids = list(dict.fromkeys(output.result_id for item in read.items for output in item.outputs))
    results = [
        result for result_id in result_ids if (result := host.storage.load_eval_result(result_id, scope_id)) is not None
    ]
    run_ids = sorted({run_id for arm in read.arms for run_id in arm.run_ids})
    judges = {
        run.judge_model for run_id in run_ids if (run := host.storage.load_eval_run(run_id, scope_id)) is not None
    }
    judge_models = {judge for judge in judges if judge is not None}
    if read.closed_at is None:
        closed = close_blind_read(read, results, judge_models=judge_models, closed_by=closed_by)
        host.storage.save_blind_read(closed.read, if_match=etag)
        stored, ratings = closed.read, closed.ratings
    else:
        # A close whose rating writes failed after the read was closed: what it left unwritten, from the read as stored.
        stored = read
        ratings = _unwritten(host, close_blind_read(read, results, judge_models=judge_models).ratings, scope_id)
    written = 0
    not_written: list[str] = []
    for rating in ratings:
        try:
            rate_result(
                host.storage,
                result_id=rating.result_id,
                scope_id=scope_id,
                rubric_dim=rating.rubric_dim,
                rater=rating.rater,
                rater_kind=rating.rater_kind,
                score=rating.score,
                reason=rating.reason,
                source=rating.source,
                collected_by=rating.collected_by,
                rated_at=rating.rated_at,
            )
        except (NotFoundError, ConflictError, ValidationFailedError) as refused:
            not_written.append(f"{rating.rater} on {rating.rubric_dim} of result {rating.result_id}: {refused}")
            continue
        except StorageError as failed:
            log.warning(
                "eval.blind_read_close ratings_unwritten read=%s scope=%s written=%d of=%d",
                read_id,
                scope_id,
                written,
                len(ratings),
                exc_info=True,
            )
            raise _RatingsUnwritten(
                f"blind read {read_id} is closed, and the store refused a rating after {written} of {len(ratings)} "
                f"were written: {failed.message}"
            ) from failed
        written += 1
    if read.closed_at is not None and not written:
        if not not_written:
            # Every rating is stored: nothing is left to close.
            _refuse_closed(read, "second close")
        refused_all = ConflictError(
            f"blind read {read_id} is closed, and none of the {len(not_written)} rating(s) it has not written can be: "
            + "; ".join(not_written),
            remedy="nothing is left to close: each rating listed stays unwritten for the reason given",
        )
        refused_all.details["not_written"] = not_written
        raise refused_all
    log.info(
        "eval.blind_read_close read=%s scope=%s by=%s early=%s ratings=%d not_written=%d resumed=%s",
        read_id,
        scope_id,
        closed_by,
        stored.closed_early,
        written,
        len(not_written),
        read.closed_at is not None,
    )
    return _line_of(host, stored, ratings_written=written, ratings_not_written=not_written)


def blind_read_get(host: EvalHost, read_id: str, scope_id: str) -> BlindReadLine:
    """One read's line: what was drawn, who has answered how far, and once closed what it found.

    :param host: The host whose store holds the read.
    :param read_id: The read.
    :param scope_id: The caller's scope.
    :return: Its line.
    :raises NotFoundError: No such read.
    """
    read, _ = _loaded(host, read_id, scope_id)
    return _line_of(host, read)


def blind_reads_list(host: EvalHost, scope_id: str, *, campaign_id: str | None = None) -> BlindReadListing:
    """The scope's blind reads, oldest first, optionally those drawn over one campaign.

    :param host: The host whose store holds them.
    :param scope_id: The caller's scope.
    :param campaign_id: Only the reads over this campaign; None for every read.
    :return: Each read's line.
    """
    every = host.storage.query_blind_reads(scope_id)
    shown = [read for read in every if campaign_id is None or read.campaign_id == campaign_id]
    return BlindReadListing(reads=[_line(read, _earlier(every, read)) for read in shown])


__all__ = [
    "BlindReadLine",
    "BlindReadListing",
    "blind_read_answer",
    "blind_read_close",
    "blind_read_create",
    "blind_read_get",
    "blind_read_review",
    "blind_read_sheet",
    "blind_reads_list",
]
