"""A host's door onto the claim read: draw one over stored runs or a campaign's, serve its sheet, take people's
answers, close it, read its figures and check, and have a judge grade its claims again.

The door onto the one core (:mod:`threetears.evals.analysis.claim_read`), as the blind read's door
(:mod:`threetears.evals.ops.blind_reads`) is onto its own. Nothing here draws, blinds or reads a figure itself: it loads
what the core needs from the host's store, calls the core, and stores what the core returns (a
:class:`~threetears.evals.schema.ClaimRead`, durable).

**The draw takes no seed.** A caller cannot draw again until an order suits it: the seed is derived from the
population (its digest), the stratifier, the questions and how many reads were drawn over that population before, and
the line says how many were and which seed it took. What the caller chooses is fixed before anything is drawn: the
questions, the stratifier, and the items planned in all or per stratum.

**Who may answer.** A person's answer is recorded as theirs only when the host attests the caller relaying it for
that person (``EvalHost.attests_person``, :func:`~threetears.evals.ops.runs.attested_person`); otherwise it is refused,
403, and nothing is written. No action takes a model's answer.

**What a rater is shown gives nothing away.** The sheet (:func:`claim_read_sheet`) carries each claim as the
instrument read it, its excerpt, its segment and the numbered evidence it was shown: no verdict, citation, stratum or
arm, no run, result or case id, and no count of who answered what. A read's line carries no figure until it is closed.

**One look.** Its figures and check are read once it is closed (:func:`claim_read_close`), and a closed read takes no
answer, so a check is read at the look its plan fixed, never at whichever look first met the bar. A read closed with
items no person answered is closed early, and says so.

**Repeats** (:func:`claim_read_repeat`) grade the drawn claims' cells again on a judge: the run's own judge (the
instrument against itself) or another (a second judge), over the evidence and the web windows each grade stored,
with the extraction fixed and nothing searched (:func:`~threetears.evals.kernel.claims.repeat.repeat_grade`). A
repeat's prompts change as its repairs ask again, so it is priced at its worst case under the criterion's budgets
(:func:`~threetears.evals.kernel.claims.repeat.repeat_ceiling`) and refused before any call where those ceilings
together pass the out-of-run cap; each call is then admitted at its own prompt against the same cap, and a cell
never spends past its own worst case (``hold_exceeded``). Each cell's repeats are written on the read as its calls
return, only from inside the read's own repeat job, so no caller can put a model's reading on a read by any other
door.

These are the catalogue's operations (``claim_read_create`` and its siblings), called by name through the action
catalogue: none is exported from :mod:`threetears.evals.ops`.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Self

from pydantic import Field

from threetears.evals.analysis.checks import Check
from threetears.evals.analysis.claim_read import (
    CLAIM_READ_SEED,
    ClaimCalibration,
    Estimate,
    GradedCell,
    draw_claim_read,
    read_claim_calibration,
    read_claim_repeats,
)
from threetears.evals.kernel.capped_completion import refresh_prices
from threetears.evals.kernel.claims.failures import ClaimGradingFailed
from threetears.evals.kernel.claims.grading import claim_evidence_digest, criterion_fingerprint, criterion_steps
from threetears.evals.kernel.claims.repeat import refuse_repeat_past_cap, repeat_ceiling, repeat_grade
from threetears.evals.kernel.errors import ConflictError, EvalServiceError, NotFoundError, ValidationFailedError
from threetears.evals.kernel.host.eval_host import EvalHost
from threetears.evals.kernel.job_context import running_job
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.out_of_run import OutOfRunBudget, PlannedCall
from threetears.evals.kernel.remedies import FIND_CAMPAIGN
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.jobs import ACTION_JOB_PREFIX, AdmittedJob, JobsStarted, job_poll, start_job
from threetears.evals.ops.runs import attested_person
from threetears.evals.run.job_work import DEFAULT_ACTION_JOB_BUDGET_S, action_job_work
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.claim_read import (
    ClaimAnswer,
    ClaimItem,
    ClaimItemRepeat,
    ClaimQuestion,
    ClaimRead,
    ClaimReadArm,
    ClaimRepeatAgainst,
    ClaimStratifier,
    ClaimStratum,
    ClaimStratumDraw,
    ClaimVerdict,
    ShownEvidence,
)
from threetears.evals.schema.completion import CompletionResult
from threetears.evals.schema.logs import contain, get_logger
from threetears.evals.schema.models import ClaimCriterion, ClaimEvidence, ClaimGrade, EvalRun, utc_now_iso
from threetears.evals.schema.out_of_run_spend import KnownOutOfRunPurpose
from threetears.evals.schema.reading_states import Maker
from threetears.evals.schema.reference import ReferenceKey

log = get_logger(__name__)

#: Where a caller finds a read's id.
FIND_CLAIM_READ: Final = "claim_reads_list names this scope's claim reads"

#: What the action that grades a read's claims again is called, as its job records it.
REPEAT_ACTION: Final = "claim_read_repeat"

#: What each question asks a rater, in the words the sheet shows.
QUESTION_WORDS: Final[Mapping[ClaimQuestion, str]] = {
    "true": "Is the claim true? Pass it unless it is false. You may search the web to decide.",
    "grounded": (
        "Does a passage shown, from a document the writer was given (marked run, or case where the writer was given "
        "the case's documents), support the claim? Pass it if one does; fail it if none does."
    ),
    "is_claim": (
        "Is this a checkable claim the text makes? Pass it if the text states it as fact and it can be checked; fail "
        "it if it is an opinion, a question, or not what the text says."
    ),
}

#: What every sheet tells its rater first.
SHEET_INSTRUCTIONS: Final = (
    "Each item is one claim taken from a piece of writing, with the words it stands in, the passage they were said "
    "in, and the evidence the grader was shown, numbered. Answer each question asked pass or fail, from the claim and "
    "the evidence (and, where a question says so, your own search). Take the items in order, starting at next; a "
    "skip is an answer."
)

#: A change to a read, applied to it as stored.
_Apply = Callable[[ClaimRead], ClaimRead]

#: How often a write that lost its race to another answer is read again and applied again before it is refused.
_WRITE_ATTEMPTS: Final = 3

#: Every question a read asks by default where its grades read groundedness, in the order a read lists them.
_DEFAULT_QUESTIONS: Final[tuple[ClaimQuestion, ...]] = ("true", "grounded", "is_claim")

#: What a caller relaying a person's answer does when the host vouches for no person at all.
_UNCHECKED_REMEDY: Final = "a host that relays people's answers is built with attests_person=<its check>"

#: What a caller relaying a person's answer does when the host does not attest it for that person.
_REFUSED_REMEDY: Final = "relay the answer as a caller the host attests for that person"


# --- what the door hands back ----------------------------------------------------------------------------------


class ClaimFigure(EvalBaseModel):
    """A figure with its interval: the lower end one-sided 95%, the upper one-sided 97.5%; None where undefined."""

    value: float | None
    lower: float | None = None
    upper: float | None = None


class ClaimReadRater(EvalBaseModel):
    """One rater's progress through a read."""

    rater: str = Field(description="The person as the host names them, or a model's id.")
    made_by: Maker = Field(description="A person, or a model. Only a person's answers are read in a figure.")
    answered: int = Field(ge=0, description="The items answered, skips included.")
    skipped: int = Field(ge=0, description="How many of them were skipped.")


class ClaimStratumAgreement(EvalBaseModel):
    """How people stood against the instrument in one stratum of the draw, unweighted."""

    stratum: ClaimStratum
    population: int = Field(ge=0, description="The stratum's claims.")
    items: int = Field(ge=0, description="Its items people answered the question on.")
    agreed: float | None = Field(description="The share answered as the instrument did; None with none answered.")


class ClaimArmFigures(EvalBaseModel):
    """One arm's figures on one question, read through the draw's weights."""

    arm: str = Field(description="The arm, by its key (ClaimReadArm.key).")
    items: int = Field(ge=0)
    cells: int = Field(ge=0)
    recall: ClaimFigure = Field(description="Of the arm's claims people fail, the share the instrument fails.")
    false_fail: ClaimFigure = Field(description="Of the arm's claims people pass, the share the instrument fails.")
    precision: ClaimFigure = Field(description="Of the arm's claims the instrument fails, the share people fail.")
    bias: ClaimFigure = Field(description="The instrument's fail rate over people's, on the arm's claims.")
    failed: ClaimFigure = Field(description="The share of the arm's claims people fail.")


class ClaimQuestionFigures(EvalBaseModel):
    """What a closed read says of the instrument on one question, weighted to the claims it was drawn from."""

    question: ClaimQuestion
    items: int = Field(ge=0, description="The items people answered it on.")
    cells: int = Field(ge=0, description="The distinct cells they come from: what the calibration floor counts.")
    cases: int = Field(ge=0)
    skipped: int = Field(ge=0, description="Items skipped or left unanswered on this question.")
    table: tuple[float, float, float, float] | None = Field(
        description="The population's shares: both fail, only the instrument fails, only people fail, both pass. "
        "None with no answered item."
    )
    kappa: ClaimFigure
    recall: ClaimFigure = Field(description="Of the claims people fail, the share the instrument fails.")
    false_fail: ClaimFigure = Field(description="Of the claims people pass, the share the instrument fails.")
    precision: ClaimFigure = Field(description="Of the claims the instrument fails, the share people fail.")
    bias: ClaimFigure = Field(description="The instrument's fail rate over people's.")
    failed: ClaimFigure = Field(description="The share of claims people fail.")
    strata: list[ClaimStratumAgreement]
    unread_strata: list[ClaimStratum] = Field(description="Strata holding claims and no answered item.")
    arms: list[ClaimArmFigures]
    lean_p: float | None = Field(
        description="The test across arms of a lean on recall and the false-fail rate; None where no lean was read."
    )
    lean_untested: list[str] = Field(
        description="The arms, by key, the lean test read no rate on: too few of their answered claims on a side."
    )
    check: Check | None = Field(
        description="The check of the instrument on this question against people; None on is_claim, which only "
        "people answer."
    )


class ClaimRepeatStratum(EvalBaseModel):
    """How a repeat stood to the first grade in one stratum of the draw."""

    stratum: ClaimStratum
    items: int = Field(ge=0, description="The repeats both readings answered the question on.")
    agreed: int = Field(ge=0)


class ClaimRepeatQuestion(EvalBaseModel):
    """How a judge's repeats stood to the first grade on one question, as counts over the repeated items.

    The counts are of the drawn items, not weighted to the campaign: the draw took more of the rarer strata, so the
    per-stratum counts are what each says alone.
    """

    question: ClaimQuestion = Field(description="true or grounded.")
    items: int = Field(ge=0)
    cells: int = Field(ge=0, description="The distinct cells they come from.")
    both_fail: int = Field(ge=0)
    first_only: int = Field(ge=0, description="The first grade fails the claim, the repeat passes it.")
    repeat_only: int = Field(ge=0, description="The repeat fails the claim, the first grade passes it.")
    both_pass: int = Field(ge=0)
    agreed: int = Field(ge=0)
    strata: list[ClaimRepeatStratum]


class ClaimRepeatWeb(EvalBaseModel):
    """Claims both readings decided on the web, by what each found."""

    first: str = Field(description="What the first grade's web check found.")
    repeat: str = Field(description="What the repeat found reading the same windows again.")
    claims: int = Field(ge=1)


class ClaimRepeatFigures(EvalBaseModel):
    """What one judge's repeats of a read's items say of the instrument."""

    against: ClaimRepeatAgainst = Field(
        description="itself: the run's judge asked again. model: another judge asked the same claims."
    )
    judge_model: str
    passes: int = Field(ge=1, description="The repeat passes behind it.")
    repeats: int = Field(ge=1, description="Its repeats in all, lost ones included.")
    lost: dict[str, int] = Field(description="The repeats that gave no verdict, by the loss that cost it.")
    questions: list[ClaimRepeatQuestion]
    web: list[ClaimRepeatWeb]


class ClaimReadLine(EvalBaseModel):
    """A claim read as its door reads it: what was drawn, who has answered how far, and once closed what it found.

    It never names an item's stratum, arm or result, and while the read is open it carries no figure and no check.
    """

    read_id: str
    name: str
    campaign_id: str | None
    criterion: str = Field(description="The claim criterion whose grades were drawn.")
    instrument: str = Field(description="The instrument, as its checks name it: claims:<criterion>@<fingerprint, 12>.")
    questions: list[ClaimQuestion]
    stratify_by: ClaimStratifier
    seed: int = Field(description="The seed it was drawn by, derived, never chosen.")
    arms: list[ClaimReadArm] = Field(description="The arms drawn over, each with its runs.")
    strata: list[ClaimStratumDraw] = Field(description="Each stratum: its claims, planned draws and draws.")
    excluded: dict[str, int] = Field(description="Claims left out of the population, by why.")
    items: int = Field(ge=0)
    earlier_reads: int = Field(ge=0, description="How many claim reads over the same claims were drawn before it.")
    raters: list[ClaimReadRater]
    closed_at: str | None
    closed_by: str | None
    closed_early: bool = Field(description="Closed with items no person answered.")
    figures: list[ClaimQuestionFigures] = Field(
        default_factory=list, description="Once closed: one per question it asks."
    )
    model_answers: int = Field(default=0, ge=0, description="Once closed: answers a model gave, read in no figure.")
    borrowed: int = Field(
        default=0, ge=0, description="Once closed: people's answers taken from earlier reads of byte-identical views."
    )
    repeat_job_id: str | None = Field(description="The latest repeat's job, to poll with job_poll; None until started.")
    repeats: list[ClaimRepeatFigures] = Field(
        default_factory=list, description="Each judge's repeats of the drawn claims against the first grade."
    )


class ClaimReadListing(EvalBaseModel):
    """The claim reads in a scope, oldest first."""

    reads: list[ClaimReadLine]


class ClaimSheetQuestion(EvalBaseModel):
    """One question as the rater reads it."""

    question: ClaimQuestion
    asks: str = Field(min_length=1)


class ClaimSheetItem(EvalBaseModel):
    """One claim as the rater reads it: what was said, and what the grader was shown."""

    item: str = Field(min_length=1)
    claim: str = Field(description="The proposition, stated on its own.")
    excerpt: str = Field(description="The words it stands in.")
    start: int = Field(ge=0, description="Where the excerpt starts in the segment.")
    end: int = Field(ge=1, description="Where it ends.")
    segment: str = Field(description="The passage it was said in, whole.")
    evidence: list[ShownEvidence] = Field(description="The evidence the grader was shown, numbered, in order.")


class ClaimSheet(EvalBaseModel):
    """The blinded view of a claim read: all a renderer may show.

    Show the items in order and offer ``next`` first; show nothing but what the sheet carries; a skip is an answer.
    """

    read_id: str
    name: str
    instructions: str
    questions: list[ClaimSheetQuestion]
    items: list[ClaimSheetItem]
    answered: list[str] = Field(default_factory=list, description="The items the rater named has answered.")
    next: str | None = Field(default=None, description="The first unanswered item in order; None when all are.")


# --- reading the store -----------------------------------------------------------------------------------------


def _loaded(host: EvalHost, read_id: str, scope_id: str) -> tuple[ClaimRead, str]:
    """The read and its etag.

    :raises NotFoundError: No claim read with that id in the scope.
    """
    read, etag = host.storage.load_claim_read_with_etag(read_id, scope_id)
    if read is None or etag is None:
        raise NotFoundError("claim read", read_id, remedy=FIND_CLAIM_READ)
    return read, etag


def _refuse_closed(read: ClaimRead, what: str) -> None:
    """:raises ConflictError: The read is closed."""
    if read.closed_at is not None:
        raise ConflictError(
            f"claim read {read.id} was closed at {read.closed_at}, so it takes no {what}",
            remedy="draw another read over the same claims (claim_read_create)",
        )


def _same_population(read: ClaimRead, other: ClaimRead) -> bool:
    return other.draw.population_digest == read.draw.population_digest


def _earlier(reads: Sequence[ClaimRead], read: ClaimRead) -> int:
    """How many of ``reads`` were drawn over ``read``'s population before it."""
    return sum(
        1
        for other in reads
        if other.id != read.id and _same_population(read, other) and other.created_at <= read.created_at
    )


def _figure(estimate: Estimate) -> ClaimFigure:
    return ClaimFigure(value=estimate.value, lower=estimate.lower, upper=estimate.upper)


def _figures(calibration: ClaimCalibration) -> list[ClaimQuestionFigures]:
    return [
        ClaimQuestionFigures(
            question=reading.question,
            items=reading.items,
            cells=reading.cells,
            cases=reading.cases,
            skipped=reading.skipped,
            table=reading.table,
            kappa=_figure(reading.kappa),
            recall=_figure(reading.recall),
            false_fail=_figure(reading.false_fail),
            precision=_figure(reading.precision),
            bias=_figure(reading.bias),
            failed=_figure(reading.failed),
            strata=[
                ClaimStratumAgreement(
                    stratum=stratum.stratum, population=stratum.population, items=stratum.items, agreed=stratum.agreed
                )
                for stratum in reading.strata
            ],
            unread_strata=list(reading.unread_strata),
            arms=[
                ClaimArmFigures(
                    arm=arm.arm,
                    items=arm.items,
                    cells=arm.cells,
                    recall=_figure(arm.recall),
                    false_fail=_figure(arm.false_fail),
                    precision=_figure(arm.precision),
                    bias=_figure(arm.bias),
                    failed=_figure(arm.failed),
                )
                for arm in reading.arms
            ],
            lean_p=reading.lean_p,
            lean_untested=list(reading.lean_untested),
            check=reading.check,
        )
        for reading in calibration.questions
    ]


def _repeat_figures(read: ClaimRead) -> list[ClaimRepeatFigures]:
    return [
        ClaimRepeatFigures(
            against=reading.against,
            judge_model=reading.judge_model,
            passes=reading.passes,
            repeats=reading.repeats,
            lost=dict(reading.lost),
            questions=[
                ClaimRepeatQuestion(
                    question=question.question,
                    items=question.items,
                    cells=question.cells,
                    both_fail=question.table[0],
                    first_only=question.table[1],
                    repeat_only=question.table[2],
                    both_pass=question.table[3],
                    agreed=question.agreed,
                    strata=[
                        ClaimRepeatStratum(stratum=stratum.stratum, items=stratum.items, agreed=stratum.agreed)
                        for stratum in question.strata
                    ],
                )
                for question in reading.questions
            ],
            web=[ClaimRepeatWeb(first=first, repeat=again, claims=n) for (first, again), n in reading.web.items()],
        )
        for reading in read_claim_repeats(read)
    ]


def _person_answered(read: ClaimRead) -> set[str]:
    """The items some person answered or skipped."""
    return {answer.item for answer in read.answers if answer.made_by == "person"}


def _line(read: ClaimRead, every: Sequence[ClaimRead]) -> ClaimReadLine:
    """The read as its door serves it: no figure while it is open; every figure and check once it is closed, its
    answers read with those of the scope's other reads of the same definition (:func:`read_claim_calibration`)."""
    keys = list(dict.fromkeys((answer.rater, answer.made_by) for answer in read.answers))
    raters = [
        ClaimReadRater(
            rater=rater,
            made_by=kind,
            answered=sum(1 for a in read.answers if a.rater == rater and a.made_by == kind),
            skipped=sum(1 for a in read.answers if a.rater == rater and a.made_by == kind and a.skipped),
        )
        for rater, kind in keys
    ]
    closed = read.closed_at is not None
    calibration = read_claim_calibration(read, earlier=[r for r in every if r.id != read.id]) if closed else None
    return ClaimReadLine(
        read_id=read.id,
        name=read.name,
        campaign_id=read.campaign_id,
        criterion=read.criterion,
        instrument=f"claims:{read.criterion}@{read.criterion_fingerprint[:12]}",
        questions=list(read.questions),
        stratify_by=read.draw.stratify_by,
        seed=read.draw.seed,
        arms=list(read.arms),
        strata=list(read.draw.strata),
        excluded={str(why): count for why, count in read.draw.excluded.items()},
        items=len(read.items),
        earlier_reads=_earlier(every, read),
        raters=raters,
        closed_at=read.closed_at,
        closed_by=read.closed_by,
        closed_early=closed and bool({item.item for item in read.items} - _person_answered(read)),
        figures=_figures(calibration) if calibration is not None else [],
        model_answers=calibration.model_answers if calibration is not None else 0,
        borrowed=calibration.borrowed if calibration is not None else 0,
        repeat_job_id=f"{ACTION_JOB_PREFIX}{read.repeat_job_id}" if read.repeat_job_id is not None else None,
        repeats=_repeat_figures(read),
    )


def _line_of(host: EvalHost, read: ClaimRead) -> ClaimReadLine:
    return _line(read, host.storage.query_claim_reads(read.scope_id))


# --- the draw --------------------------------------------------------------------------------------------------


def _runs(host: EvalHost, run_ids: Sequence[str], scope_id: str) -> list[EvalRun]:
    """:raises NotFoundError: A run named is not stored."""
    runs: list[EvalRun] = []
    for run_id in dict.fromkeys(run_ids):
        run = host.storage.load_eval_run(run_id, scope_id)
        if run is None:
            raise NotFoundError("run", run_id, remedy="runs_list names this scope's runs")
        runs.append(run)
    return runs


def _graded(
    host: EvalHost, runs: Sequence[EvalRun], scope_id: str, criterion: str
) -> tuple[list[ClaimReadArm], list[GradedCell]]:
    """Every cell of ``runs`` holding a stored grade under ``criterion``, and the arms they fall in: one per variant
    key the graded cells carry, each with the runs that hold its cells.

    A cell whose result tallies the criterion and whose trace holds no grade of it is a store that lost a write; it is
    left out and logged, never drawn from.

    :raises ValidationFailedError: A run holds graded cells of more than one arm.
    """
    runs_of: dict[str, list[str]] = {}
    cells: list[GradedCell] = []
    untraced: list[str] = []
    for run in runs:
        held: set[str] = set()
        for result in host.storage.query_eval_results_by_run(run.id, scope_id):
            if criterion not in result.claim_tallies:
                continue
            trace = host.storage.load_eval_trace(result.id, scope_id)
            grade = trace.claim_grades.get(criterion) if trace is not None else None
            if trace is None or trace.claim_evidence is None or grade is None:
                untraced.append(result.id)
                continue
            held.add(result.variant_key)
            cells.append(
                GradedCell(
                    run_id=run.id,
                    result_id=result.id,
                    case_id=result.test_case_id,
                    grade=grade,
                    evidence=trace.claim_evidence,
                )
            )
        if len(held) > 1:
            raise ValidationFailedError(
                f"run {run.id} holds graded cells of {len(held)} arms, so its claims cannot be told apart by arm",
                remedy="draw over runs that each hold one arm (run_ids)",
            )
        for key in held:
            runs_of.setdefault(key, []).append(run.id)
    if untraced:
        log.warning(
            "eval.claim_read_create criterion=%s scope=%s left out %d cell(s) tallied with no stored grade: %s",
            criterion,
            scope_id,
            len(untraced),
            ", ".join(untraced[:10]),
        )
    arms = [ClaimReadArm(key=key, run_ids=run_ids) for key, run_ids in sorted(runs_of.items())]
    return arms, cells


def _fingerprint(cells: Sequence[GradedCell], criterion: str) -> str:
    """The definition the read is drawn under: the one most graded cells were graded under.

    Cells graded under another are left out of the population, counted (``redefined``).

    :raises ValidationFailedError: Two definitions grade equally many cells.
    """
    counts = Counter(cell.grade.criterion_fingerprint for cell in cells).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        tied = ", ".join(f"{fingerprint[:12]} ({n} cells)" for fingerprint, n in counts if n == counts[0][1])
        raise ValidationFailedError(
            f"claim criterion {criterion!r} grades the runs' cells under definitions {tied} equally often, so no one "
            "definition is the read's",
            remedy="draw over runs graded under one definition (run_ids)",
        )
    return counts[0][0]


def _seed(digest: str, stratify_by: ClaimStratifier, questions: Sequence[ClaimQuestion], earlier: int) -> int:
    """The seed of the next read over a population: its digest, its stratifier and questions, and how many reads were
    drawn over it before."""
    asked = "|".join(questions)
    return int(
        hashlib.sha256(f"claim-read-seed|{digest}|{stratify_by}|{asked}|{earlier}".encode()).hexdigest()[:15], 16
    )


def claim_read_create(
    host: EvalHost,
    scope_id: str,
    *,
    created_by: str,
    criterion: str,
    campaign_id: str | None = None,
    run_ids: Sequence[str] | None = None,
    questions: Sequence[ClaimQuestion] | None = None,
    stratify_by: ClaimStratifier = "truth",
    planned_items: int | None = None,
    planned: Mapping[ClaimStratum, int] | None = None,
    name: str | None = None,
) -> ClaimReadLine:
    """Draw a claim read over a campaign's runs, or over named runs, and store it.

    No seed is taken: it is derived from the population, the stratifier, the questions and how many reads were drawn
    over the same claims before, which the line states (``seed``, ``earlier_reads``). Each arm is one variant key the
    graded cells carry.

    :param host: The host whose store holds the runs.
    :param scope_id: The caller's scope.
    :param created_by: Who drew it.
    :param criterion: The claim criterion whose grades are drawn, by name.
    :param campaign_id: The campaign whose runs the read is over; or None with ``run_ids``.
    :param run_ids: The runs it is over; or None with ``campaign_id``.
    :param questions: What each item asks; None for true, grounded (where a ground step read the claims) and is_claim.
    :param stratify_by: The instrument's answer the draw is stratified by: its truth verdict, or its groundedness.
    :param planned_items: The items planned in all, spread over the strata by the misses each is expected to hold;
        None for the core's default. Not with ``planned``.
    :param planned: The items planned per stratum, fixed now; a stratum left out plans none. Not with
        ``planned_items``.
    :param name: What the read is called; None names it after the campaign or the runs.
    :return: The read as stored.
    :raises ValidationFailedError: Neither or both of ``campaign_id`` and ``run_ids``; both plans; a run holding two
        arms; no stored grade of the criterion; two definitions grading equally many cells; a plan past what a
        stratum holds; or nothing to draw.
    :raises NotFoundError: A campaign or run named is not stored.
    :raises ConflictError: Another read over the same claims was drawn at the same moment; ask again.
    """
    if (campaign_id is None) == (run_ids is None):
        raise ValidationFailedError(
            "a claim read is drawn over a campaign (campaign_id) or over runs (run_ids), one of the two",
            remedy="name the campaign, or the runs",
        )
    if planned_items is not None and planned is not None:
        raise ValidationFailedError(
            "a claim read plans its items in all (planned_items) or per stratum (per_stratum), one of the two",
            remedy="name one plan",
        )
    if campaign_id is not None:
        campaign = host.storage.load_campaign(campaign_id, scope_id)
        if campaign is None:
            raise NotFoundError("campaign", campaign_id, remedy=FIND_CAMPAIGN)
        run_ids = campaign.run_ids
    assert run_ids is not None
    arms, cells = _graded(host, _runs(host, run_ids, scope_id), scope_id, criterion)
    if not cells:
        raise ValidationFailedError(
            f"no cell of the runs holds a stored grade under claim criterion {criterion!r}, so there is nothing to draw",
            remedy="name a claim criterion the runs' template declares, and runs that graded it",
        )
    fingerprint = _fingerprint(cells, criterion)
    if questions is None:
        grounded = any(claim.grounded is not None for cell in cells for claim in cell.grade.claims)
        questions = [question for question in _DEFAULT_QUESTIONS if grounded or question != "grounded"]
    label = name or (f"claim read of campaign {campaign_id}" if campaign_id else "claim read")

    def draw(seed: int) -> ClaimRead:
        try:
            return draw_claim_read(
                cells,
                arms,
                criterion=criterion,
                criterion_fingerprint=fingerprint,
                questions=questions,
                scope_id=scope_id,
                created_by=created_by,
                name=label,
                seed=seed,
                stratify_by=stratify_by,
                planned_items=planned_items,
                planned=planned,
                campaign_id=campaign_id,
            )
        except ValueError as refused:
            raise ValidationFailedError(
                str(refused), remedy="plan each stratum at most the claims it holds"
            ) from refused

    probe = draw(CLAIM_READ_SEED)
    if not probe.items:
        raise ValidationFailedError(
            f"the draw over claim criterion {criterion!r}'s grades gave no item (left out: "
            f"{probe.draw.excluded or 'none'}; planned: {[(s.stratum, s.planned) for s in probe.draw.strata]})",
            remedy="plan more items, ask fewer questions, or draw over runs graded under one definition",
        )
    stored = host.storage.query_claim_reads(scope_id)
    earlier = sum(1 for other in stored if _same_population(probe, other))
    read = draw(_seed(probe.draw.population_digest, stratify_by, questions, earlier))
    host.storage.save_claim_read(read, create_only=True)
    log.info(
        "eval.claim_read_create read=%s scope=%s by=%s criterion=%s items=%d",
        read.id,
        scope_id,
        created_by,
        criterion,
        len(read.items),
    )
    return _line(read, [*stored, read])


# --- the sheet and a person's answer ---------------------------------------------------------------------------


def claim_read_sheet(host: EvalHost, read_id: str, scope_id: str, *, rater: str | None = None) -> ClaimSheet:
    """The read's sheet: what a rater is shown, and nothing else. No verdict, citation, stratum or arm, no run,
    result or case id, and nothing of what anyone answered but the named rater's own progress.

    :param host: The host whose store holds the read.
    :param read_id: The read.
    :param scope_id: The caller's scope.
    :param rater: Whose progress to give (``answered``, ``next``), as their answers name them; None for none.
    :return: The sheet.
    :raises NotFoundError: No such read.
    """
    read, _ = _loaded(host, read_id, scope_id)
    answered = {answer.item for answer in read.answers if rater is not None and answer.rater == rater}
    return ClaimSheet(
        read_id=read.id,
        name=read.name,
        instructions=SHEET_INSTRUCTIONS,
        questions=[ClaimSheetQuestion(question=q, asks=QUESTION_WORDS[q]) for q in read.questions],
        items=[
            ClaimSheetItem(
                item=item.item,
                claim=item.view.claim,
                excerpt=item.view.excerpt,
                start=item.view.start,
                end=item.view.end,
                segment=item.view.segment,
                evidence=list(item.view.evidence),
            )
            for item in read.items
        ],
        answered=[item.item for item in read.items if item.item in answered],
        next=next((item.item for item in read.items if item.item not in answered), None) if rater is not None else None,
    )


def _verdicts(read: ClaimRead, answers: Mapping[str, str], reason: str) -> list[ClaimVerdict]:
    """Each question's answer as a verdict: pass (1) or fail (0).

    :raises ValueError: A question the read does not ask, or an answer that is not pass or fail.
    """
    verdicts: list[ClaimVerdict] = []
    for question, label in answers.items():
        asked = next((q for q in read.questions if q == question), None)
        if asked is None:
            raise ValueError(f"question {question!r} is not one this read asks ({list(read.questions)})")
        word = str(label).strip().lower()
        if word not in ("pass", "fail"):
            raise ValueError(f"question {question!r} is answered {label!r}; a claim read's answers are pass or fail")
        verdicts.append(ClaimVerdict(question=asked, score=int(word == "pass"), reason=reason))
    return verdicts


def _rewrite(host: EvalHost, read_id: str, scope_id: str, apply: _Apply) -> ClaimRead:
    """Read the read, apply ``apply`` and write it back conditionally, again on a lost race, a few times.

    :raises ConflictError: Every attempt lost its race.
    """
    for attempt in range(_WRITE_ATTEMPTS):
        read, etag = _loaded(host, read_id, scope_id)
        changed = apply(read)
        try:
            host.storage.save_claim_read(changed, if_match=etag)
        except ConflictError:
            if attempt + 1 == _WRITE_ATTEMPTS:
                raise
            continue
        return changed
    raise AssertionError("unreachable: the last attempt returns or raises")


def claim_read_answer(
    host: EvalHost,
    read_id: str,
    scope_id: str,
    *,
    collected_by: str,
    person: str,
    item: str,
    answers: Mapping[str, str] | None = None,
    skipped: bool = False,
    reason: str = "",
) -> ClaimReadLine:
    """Record a person's answer to one item, relayed by a caller the host attests for them.

    The host is asked first (``EvalHost.attests_person``); without its word nothing is read or written. The person's
    earlier answer to the same item is replaced. A model's answer never comes through here.

    :param host: The host whose check attests the person.
    :param read_id: The read.
    :param scope_id: The caller's scope.
    :param collected_by: The calling identity relaying the answer, as the host resolved it.
    :param person: Whose answer it is, in the caller's spelling.
    :param item: The item answered, as the sheet names it.
    :param answers: Each question answered, pass or fail; the questions left out are unanswered. None for a skip.
    :param skipped: The person passed on the item; then no answer.
    :param reason: The person's own words for the answer, where given.
    :return: The read as stored, with the person's progress; no figure while it is open.
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
        operation="claim_read_answer",
        what="claim-read answer",
        unchecked_remedy=_UNCHECKED_REMEDY,
        refused_remedy=_REFUSED_REMEDY,
    )

    def apply(read: ClaimRead) -> ClaimRead:
        _refuse_closed(read, "answer")
        kept = [
            each
            for each in read.answers
            if not (each.item == item and each.rater == canonical and each.made_by == "person")
        ]
        try:
            if item not in {each.item for each in read.items}:
                raise ValueError(f"item {item!r} is no item of claim read {read.id}")
            answer = ClaimAnswer(
                item=item,
                rater=canonical,
                made_by="person",
                collected_by=collected_by,
                verdicts=_verdicts(read, answers or {}, reason),
                skipped=skipped,
            )
        except ValueError as unasked:
            # A pydantic ValidationError is a ValueError too: a skip carrying an answer is refused the same way.
            raise ValidationFailedError(
                str(unasked), remedy="answer what the sheet asks of one of its items, or skip it"
            ) from unasked
        return read.model_copy(update={"answers": [*kept, answer]})

    return _line_of(host, _rewrite(host, read_id, scope_id, apply))


# --- close, and reading back -----------------------------------------------------------------------------------


def claim_read_close(host: EvalHost, read_id: str, scope_id: str, *, closed_by: str) -> ClaimReadLine:
    """Close the read: its figures and checks are read from now on, and every later answer is refused.

    :param host: The host whose store holds the read.
    :param read_id: The read.
    :param scope_id: The caller's scope.
    :param closed_by: Who closed it.
    :return: The closed read, with what it found.
    :raises NotFoundError: No such read.
    :raises ConflictError: It is already closed, or answers kept landing as it was closed (ask again).
    """

    def apply(read: ClaimRead) -> ClaimRead:
        _refuse_closed(read, "second close")
        return read.model_copy(update={"closed_at": utc_now_iso(), "closed_by": closed_by})

    closed = _rewrite(host, read_id, scope_id, apply)
    log.info(
        "eval.claim_read_close read=%s scope=%s by=%s unanswered=%d",
        read_id,
        scope_id,
        closed_by,
        len({item.item for item in closed.items} - _person_answered(closed)),
    )
    return _line_of(host, closed)


def claim_read_get(host: EvalHost, read_id: str, scope_id: str) -> ClaimReadLine:
    """One read's line: what was drawn, who has answered how far, its repeats, and once closed what it found.

    :param host: The host whose store holds the read.
    :param read_id: The read.
    :param scope_id: The caller's scope.
    :return: Its line.
    :raises NotFoundError: No such read.
    """
    read, _ = _loaded(host, read_id, scope_id)
    return _line_of(host, read)


def claim_reads_list(host: EvalHost, scope_id: str, *, campaign_id: str | None = None) -> ClaimReadListing:
    """The scope's claim reads, oldest first, optionally those drawn over one campaign.

    :param host: The host whose store holds them.
    :param scope_id: The caller's scope.
    :param campaign_id: Only the reads over this campaign; None for every read.
    :return: Each read's line.
    """
    every = host.storage.query_claim_reads(scope_id)
    shown = [read for read in every if campaign_id is None or read.campaign_id == campaign_id]
    return ClaimReadListing(reads=[_line(read, every) for read in shown])


# --- repeats ---------------------------------------------------------------------------------------------------


class _HeldClient:
    """A judge's client as one cell's repeat sees it: each call admitted at its own prompt against the out-of-run
    budget, and never past the cell's worst case.

    A repeat's prompts change as its repairs ask again, so the job was admitted on the cells' worst cases
    (:func:`~threetears.evals.kernel.claims.repeat.repeat_ceiling`); each call is then priced as it is about to be
    sent. One that would take the cell past its worst case is refused (``hold_exceeded``: the price assumed less than
    the call came to), one the cap has no room for too (``cost_cap``): either is the step's own failure, which the
    repeat reads as the grade would (a ground step lost, a web check skipped, the cell's truth lost).
    """

    def __init__(
        self, inner: Any, budget: OutOfRunBudget, purpose: KnownOutOfRunPurpose, hold_usd: float | None
    ) -> None:
        self._inner = inner
        self._budget = budget
        self._purpose: KnownOutOfRunPurpose = purpose
        self._hold = hold_usd
        self._admitted = 0.0
        self.model_name: str = inner.model_name

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        """What the host's client prices the call at, at most."""
        ceiling: float | None = self._inner.price_ceiling(system=system, user=user, response_format=response_format)
        return ceiling

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> CompletionResult:
        """Admit one call at its own prompt and make it through the budget, which ledgers it.

        :raises ClaimGradingFailed: The call would take the cell past its worst case (``hold_exceeded``), or the cap
            has no room for it (``cost_cap``).
        """
        call = PlannedCall(system=system, user=user, response_format=response_format)
        ceiling = self.price_ceiling(system=system, user=user, response_format=response_format)
        if self._hold is not None and ceiling is not None and self._admitted + ceiling > self._hold:
            raise ClaimGradingFailed(
                f"a repeat call priced at up to {ceiling} would take its cell past the {self._hold} it was priced at",
                loss="hold_exceeded",
            )
        try:
            (admitted,) = self._budget.admit(self._inner, self._purpose, [call])
        except ValidationFailedError as refused:
            raise ClaimGradingFailed(str(refused), loss="cost_cap") from refused
        self._admitted += admitted.ceiling_usd or 0.0
        result: CompletionResult = (await self._budget.generate(self._inner, admitted)).result
        return result

    async def aclose(self) -> None:
        """Release nothing: the host's client is the repeat's, which releases it when the job ends."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


@dataclass
class _Cell:
    """One cell to repeat: its stored grade and evidence, its criterion, the case's key, the client it is asked on,
    and its items."""

    result_id: str
    criterion: ClaimCriterion
    evidence: ClaimEvidence
    grade: ClaimGrade
    key: ReferenceKey | None
    model: str
    against: ClaimRepeatAgainst
    items: list[ClaimItem]
    ceiling: float | None = None


@dataclass
class _Plan:
    """A repeat's cells, the clients they are asked on, and the cells left out, with why."""

    cells: list[_Cell] = field(default_factory=list)
    clients: dict[str, Any] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)


def _plan(
    host: EvalHost, read: ClaimRead, scope_id: str, judge_model: str | None, client_for: Callable[[str], Any]
) -> _Plan:
    """Load every cell of the read's items, and leave out with why each that cannot be repeated as it was graded.

    Blocking, so off the loop. ``client_for`` builds (once per model) the client a model's calls go to.
    """
    plan = _Plan()
    by_cell: dict[str, list[ClaimItem]] = {}
    for item in read.items:
        by_cell.setdefault(item.result_id, []).append(item)
    for result_id, items in by_cell.items():
        run = host.storage.load_eval_run(items[0].run_id, scope_id)
        if run is None:
            plan.skipped.append(f"{result_id}: its run {items[0].run_id} is gone")
            continue
        model = judge_model or run.judge_model
        if model is None:
            plan.skipped.append(f"{result_id}: its run recorded no judge, and none was named")
            continue
        done = {repeat.item for repeat in read.repeats if repeat.judge_model == model}
        items = [item for item in items if item.item not in done]
        if not items:
            continue
        trace = host.storage.load_eval_trace(result_id, scope_id)
        grade = trace.claim_grades.get(read.criterion) if trace is not None else None
        if trace is None or trace.claim_evidence is None or grade is None:
            plan.skipped.append(f"{result_id}: its trace no longer holds the grade")
            continue
        if grade.claim_evidence_digest != claim_evidence_digest(trace.claim_evidence):
            plan.skipped.append(f"{result_id}: its stored evidence is not what was graded")
            continue
        template = host.storage.load_template(run.template_id, scope_id) if run.template_id else None
        criterion = next((c for c in (template.claim_criteria if template else ()) if c.name == read.criterion), None)
        if criterion is None:
            plan.skipped.append(f"{result_id}: its run's template no longer declares {read.criterion!r}")
            continue
        if model not in plan.clients:
            plan.clients[model] = client_for(model)
        client = plan.clients[model]
        if criterion_fingerprint(criterion, *criterion_steps(criterion, client)) != read.criterion_fingerprint:
            plan.skipped.append(f"{result_id}: its run's template now defines {read.criterion!r} otherwise")
            continue
        case = host.storage.load_test_case(items[0].case_id, scope_id)
        plan.cells.append(
            _Cell(
                result_id=result_id,
                criterion=criterion,
                evidence=trace.claim_evidence,
                grade=grade,
                key=case.reference if case is not None else None,
                model=model,
                against="itself" if model == run.judge_model else "model",
                items=items,
            )
        )
    return plan


def keep_repeats(host: EvalHost, read_id: str, scope_id: str, repeats: Sequence[ClaimItemRepeat]) -> ClaimRead:
    """Keep a judge's repeats on the read: only inside the read's own repeat job.

    :param host: The host whose store holds the read.
    :param read_id: The read.
    :param scope_id: Its scope.
    :param repeats: The repeats, each of an item of the read.
    :return: The read as stored.
    :raises EvalServiceError: Status 403: not inside the read's repeat job; nothing is written.
    :raises NotFoundError: No such read.
    """
    job = running_job()

    def apply(read: ClaimRead) -> ClaimRead:
        if job is None or read.repeat_job_id is None or job != read.repeat_job_id:
            log.warning(
                "eval.claim_read_repeat refused job=%s read=%s scope=%s: not the read's repeat job",
                job,
                read_id,
                scope_id,
            )
            raise EvalServiceError(
                f"claim read {read_id} takes a judge's repeats only from its own repeat job, and this is "
                f"{'no job' if job is None else f'job {job}'}",
                status_code=403,
                details={"read_id": read_id},
                remedy="repeat the read's claims through its repeat job (claim_read_repeat)",
            )
        return read.model_copy(update={"repeats": [*read.repeats, *repeats]})

    return _rewrite(host, read_id, scope_id, apply)


def _repeats_of(cell: _Cell, repeat_id: str, regraded: ClaimGrade | None, loss: str | None) -> list[ClaimItemRepeat]:
    """Each of the cell's items as the repeat read it, beside the first grade's web outcome."""
    first_web = {record.claim: record.outcome for record in cell.grade.web}
    again_web = {record.claim: record.outcome for record in regraded.web} if regraded is not None else {}
    made = []
    for item in cell.items:
        claim = regraded.claims[item.claim] if regraded is not None else None
        made.append(
            ClaimItemRepeat(
                item=item.item,
                repeat_id=repeat_id,
                judge_model=cell.model,
                against=cell.against,
                verdict=claim.verdict if claim is not None else None,
                grounded=claim.grounded if claim is not None else None,
                first_web=first_web.get(item.claim),
                web=again_web.get(item.claim),
                loss=None if claim is not None and claim.verdict is not None else (loss or "provider_error"),
            )
        )
    return made


async def claim_read_repeat(
    host: OpsHost, read_id: str, scope_id: str, *, judge_model: str | None = None
) -> JobsStarted:
    """Grade the read's claims' cells again on a judge, as a job: priced at their worst case and admitted first.

    With no judge named, each cell's own run's judge is asked again (the instrument against itself); a named judge
    other than a cell's own is a second judge on it. Each cell's stored claims are verified again over the evidence
    its grade stored, and its web verdicts read again over the windows its grade stored; nothing is listed or searched
    (:func:`~threetears.evals.kernel.claims.repeat.repeat_grade`). Cells whose every item that judge already repeated
    are not asked again. Every call is ledgered under ``judge`` (the same judge) or ``second_judge``.

    :param host: The host whose clients and job manager serve the repeat.
    :param read_id: The read.
    :param scope_id: The caller's scope.
    :param judge_model: The judge to ask, by model id; None for each cell's own run's judge.
    :return: The started job; ``job_poll`` on it returns the read's line once it ends.
    :raises NotFoundError: No such read.
    :raises ConflictError: Its repeat is already running.
    :raises ValidationFailedError: No cell can be repeated, or the cells' worst cases cannot be priced under the
        enforced cap or pass it; nothing is spent.
    :raises StorageError: The job's record could not be written; nothing started.
    """
    eval_host = host.eval_host
    read, _ = await run_blocking(eval_host.blocking_executor, _loaded, eval_host, read_id, scope_id)
    if (
        read.repeat_job_id is not None
        and not (await job_poll(host, f"{ACTION_JOB_PREFIX}{read.repeat_job_id}", scope_id)).done
    ):
        raise ConflictError(
            f"claim read {read_id}'s repeat is running as job {ACTION_JOB_PREFIX}{read.repeat_job_id}",
            remedy="poll it (job_poll) and ask again once it ends",
        )
    factory = eval_host.completion_clients("a claim read's repeat")
    plan = await run_blocking(
        eval_host.blocking_executor, _plan, eval_host, read, scope_id, judge_model, lambda m: factory("judge", m)
    )

    async def release() -> None:
        for client in plan.clients.values():
            await client.aclose()

    cap = host.out_of_run_cap()
    try:
        if not plan.cells:
            why = "; ".join(plan.skipped) or "every item was repeated by that judge already"
            raise ValidationFailedError(
                f"no cell of claim read {read_id} can be repeated: {why}",
                remedy="name another judge (judge_model), or draw a read over runs whose grades are stored",
            )
        for client in plan.clients.values():
            await refresh_prices(client)
        for cell in plan.cells:
            cell.ceiling = repeat_ceiling(cell.criterion, cell.evidence, key=cell.key, client=plan.clients[cell.model])
        refuse_repeat_past_cap(
            [cell.ceiling for cell in plan.cells],
            cap,
            what=f"the repeat of claim read {read_id}'s {len(plan.cells)} cell(s)",
        )
    # prawduct:ok-broad-except — not handled: the clients built are released, and what refused is re-raised
    except BaseException:
        await release()
        raise
    budget = OutOfRunBudget(
        store=eval_host.storage,
        scope_id=scope_id,
        cap_usd=cap,
        campaign_id=read.campaign_id,
        blocking_executor=eval_host.blocking_executor,
    )
    skipped = list(plan.skipped)

    async def made() -> ClaimReadLine:
        try:
            for cell in plan.cells:
                purpose: KnownOutOfRunPurpose = "judge" if cell.against == "itself" else "second_judge"
                held = _HeldClient(plan.clients[cell.model], budget, purpose, cell.ceiling)
                loss: str | None = None
                regraded: ClaimGrade | None = None
                try:
                    repeat = await repeat_grade(cell.criterion, cell.evidence, cell.grade, key=cell.key, client=held)
                    regraded = repeat.grade
                except ClaimGradingFailed as failed:
                    loss = failed.loss
                except Exception as failed:  # prawduct:ok-broad-except — a provider's failure costs this cell alone
                    contain(failed, log=log, what=f"eval.claim_read_repeat read={read_id} cell={cell.result_id}")
                    loss = "provider_error"
                await run_blocking(
                    eval_host.blocking_executor,
                    keep_repeats,
                    eval_host,
                    read_id,
                    scope_id,
                    _repeats_of(cell, work.record_id, regraded, loss),
                )
        finally:
            await release()
        spends = [*budget.recorded, *budget.unrecorded]
        log.info(
            "eval.claim_read_repeat read=%s scope=%s cells=%d skipped=%d calls=%d",
            read_id,
            scope_id,
            len(plan.cells),
            len(skipped),
            len(spends),
        )
        line: ClaimReadLine = await run_blocking(
            eval_host.blocking_executor, claim_read_get, eval_host, read_id, scope_id
        )
        return line

    label = f"{judge_model or 'the runs judge'} repeats claim read {read.name}"
    work = action_job_work(
        scope_id,
        action=REPEAT_ACTION,
        run=made,
        label=label,
        budget_s=DEFAULT_ACTION_JOB_BUDGET_S,
        discard=release,
    )

    def claim(current: ClaimRead) -> ClaimRead:
        if current.repeat_job_id != read.repeat_job_id:
            raise ConflictError(
                f"claim read {read_id}'s repeat was started by another call as this one was",
                remedy="poll its job (claim_read_get names it) and ask again once it ends",
            )
        return current.model_copy(update={"repeat_job_id": work.record_id})

    try:
        await run_blocking(eval_host.blocking_executor, _rewrite, eval_host, read_id, scope_id, claim)
    # prawduct:ok-broad-except — not handled: the admitted clients are released, and what refused is re-raised
    except BaseException:
        await release()
        raise

    def unclaim(current: ClaimRead) -> ClaimRead:
        # The claim this call made, and no later one, goes back to the job before it.
        if current.repeat_job_id != work.record_id:
            return current
        return current.model_copy(update={"repeat_job_id": read.repeat_job_id})

    try:
        return await start_job(host, AdmittedJob(work=work, kind="action", target_id=REPEAT_ACTION, label=label))
    # prawduct:ok-broad-except — not handled: a job that did not start releases its claim, and what refused is re-raised
    except BaseException:
        await run_blocking(eval_host.blocking_executor, _rewrite, eval_host, read_id, scope_id, unclaim)
        raise


__all__ = [
    "ClaimReadLine",
    "ClaimReadListing",
    "ClaimSheet",
    "claim_read_answer",
    "claim_read_close",
    "claim_read_create",
    "claim_read_get",
    "claim_read_repeat",
    "claim_read_sheet",
    "claim_reads_list",
]
