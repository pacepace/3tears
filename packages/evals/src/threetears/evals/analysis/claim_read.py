"""Drawing a claim read and reading a claim instrument's calibration off it.

The shapes are :mod:`threetears.evals.schema.claim_read`. Three steps:

- :func:`allocate_draws` plans each stratum's draws **by the misses it is expected to hold**. Where the instrument
  said a claim is false (``contradicted``) people mostly agree, so few labels settle it; the instrument's misses, false
  claims it let through, hide in ``unverifiable`` and ``supported``, which are large and sparse. Every stratum first
  gets a floor (:data:`FLOOR_SHARE` of the plan, or all it holds), then the rest goes by Neyman allocation over
  :data:`MISS_PRIOR`: a stratum's draws grow with its size times the spread of people's answers expected in it.
- :func:`draw_claim_read` draws: within each stratum a seeded order of its claims, taken from the top while a cell
  has given fewer than :data:`~threetears.evals.schema.claim_read.CLAIMS_PER_CELL` items, the strata taken in turn so
  none spends a cell's share first. A grade made under another definition, whose stored evidence is not what it
  graded, or whose web results a recording served (``replayed``: no live search of this run stands behind it, so no
  check of the instrument reads it) is left out and counted, as is a restatement, which takes its original's verdict.
- :func:`read_claim_calibration` reads people's answers against the instrument's, per question. Every population
  figure is inverse-probability weighted: an answered item stands for its stratum's population over the stratum's
  answered items. Per question: the population's 2x2 (the instrument fails a claim, people fail it), kappa on it with
  a stratified bootstrap interval (resampling answered items within each stratum), the per-stratum agreement, and
  **detection recall** (of claims people fail, the share the instrument fails), **precision** (of those the
  instrument fails, the share people fail) and **rate bias** (the instrument's fail rate over people's), overall and
  per arm. Whether the instrument leans by arm is **one homogeneity test across every arm** on recall and precision
  together (:func:`lean_test`), never pairwise intervals: with several arms, pairwise tests withhold far more often
  than their level. A lean is the check's ``not_met`` with reason ``instrument_leans``.

**What the figures do not correct.** The per-cell cap makes a claim of a claim-heavy cell less likely to be drawn
than its stratum's weight assumes; ``tests/stats_certificate.py`` certifies the interval's coverage and the lean
test's level and power at two claims per cell, and the unit suite keeps a fast seeded case of each
(``tests/test_simulated_claim_read.py``). An answer left unanswered is taken as missing at random within its stratum.
A model's answer is never a person's: it is counted and read in no figure.

Engine-internal: nothing here is exported. A host draws and reads a claim read through the host door once it exists.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, NamedTuple

from threetears.evals.analysis.checks import Check
from threetears.evals.analysis.stats import SIGNIFICANCE_ALPHA
from threetears.evals.kernel.claims.repeat import replayed_state
from threetears.evals.kernel.claims.grading import claim_evidence_digest
from threetears.evals.kernel.evidence_tiers import calibration_criterion
from threetears.evals.schema.claim_read import (
    CLAIMS_PER_CELL,
    STRATA_OF,
    ClaimDraw,
    ClaimItem,
    ClaimQuestion,
    ClaimRead,
    ClaimReadArm,
    ClaimStratifier,
    ClaimStratum,
    ClaimAnswer,
    ClaimItemView,
    ClaimStratumDraw,
    ClaimTruthVerdict,
    ClaimVerdict,
    ShownEvidence,
    item_view_digest,
)
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.models import ClaimEvidence, ClaimGrade, utc_now_iso
from threetears.evals.schema.reading_states import Reason, StateOf

#: The seed a claim read is drawn by when none is given.
CLAIM_READ_SEED: Final = 20261122

#: The items a draw plans when none is said: about the fewest per question the sizing table finds enough to decide
#: kappa against its bar at the agreement people reach with each other.
PLANNED_ITEMS_DEFAULT: Final = 200

#: The share of the plan every stratum is given first, or all it holds where it holds fewer: so no stratum the
#: instrument used goes unread, which would leave its population's share of every figure unestimated.
FLOOR_SHARE: Final = 0.1

#: How likely people are to fail a claim in each stratum, as a planning guess: what Neyman allocation spreads draws
#: by, never a figure read. From a host's 954 hand-labelled claims with a second label on 180 drawn by the
#: first label: of claims the first label called false the second called 81% false, of those it could not settle 37%,
#: of those it supported none. ``true_ungrounded`` and the groundedness strata have no such source yet and are guesses.
MISS_PRIOR: Final[Mapping[ClaimStratum, float]] = MappingProxyType(
    {
        "contradicted": 0.8,
        "true_ungrounded": 0.1,
        "unverifiable": 0.35,
        "supported": 0.05,
        "grounded": 0.1,
        "ungrounded": 0.8,
    }
)

#: The bootstrap draws a reading takes: each a resample of the answered items within every stratum.
BOOTSTRAP_DRAWS: Final = 1000

#: The fewest bootstrap draws that must give a figure for it to carry an interval: below half, the figure is too often
#: undefined (no claim people fail, say) for its spread to be read.
_DEFINED_SHARE: Final = 0.5

#: The one-sided level each end of an interval is read at, as the tier criteria read it: the lower end one-sided 95%,
#: the upper one-sided 97.5%.
_LOWER: Final = 0.05
_UPPER: Final = 0.975

EXCLUDED_REDEFINED: Final = "graded under another definition of the criterion"
EXCLUDED_OTHER_EVIDENCE: Final = "stored evidence differs from what was graded"
#: The word the reading states use: no live search of this run stands behind the grade.
EXCLUDED_REPLAYED: Final = "replayed"
EXCLUDED_RESTATEMENT: Final = "a restatement, which takes its original's verdict"
EXCLUDED_NO_GROUND: Final = "no groundedness reading"


class GradedCell(NamedTuple):
    """One graded cell a claim read may draw from: where it is, its grade under the criterion, and its evidence."""

    run_id: str
    result_id: str
    case_id: str
    grade: ClaimGrade
    evidence: ClaimEvidence


def _sha(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def allocate_draws(
    population: Mapping[ClaimStratum, int],
    total: int,
    *,
    prior: Mapping[ClaimStratum, float] = MISS_PRIOR,
    floor_share: float = FLOOR_SHARE,
) -> dict[ClaimStratum, int]:
    """Plan each stratum's draws: a floor each, then Neyman allocation by the misses each is expected to hold.

    Each stratum first gets ``ceil(total * floor_share)``, or all it holds where it holds fewer. The rest goes to the
    strata with claims left in proportion to ``N_h * sqrt(q_h * (1 - q_h))`` (``q_h`` the prior's chance that people
    fail a claim of the stratum), by largest remainder, never past what a stratum holds; where every weight is zero,
    in proportion to what each has left.

    :param population: Each stratum's claims.
    :param total: The draws planned in all; capped at the population.
    :param prior: People's expected fail share in each stratum.
    :param floor_share: The share of ``total`` each stratum is given first.
    :return: The draws planned of each stratum, keyed as ``population``.
    :raises ValueError: A negative total or count, or a floor share outside [0, 1].
    """
    if total < 0 or any(count < 0 for count in population.values()):
        raise ValueError(f"a draw plans no negative count; got total {total} over {dict(population)}")
    if not 0.0 <= floor_share <= 1.0:
        raise ValueError(f"a floor share is a share of the plan, in [0, 1]; got {floor_share}")
    total = min(total, sum(population.values()))
    floor = math.ceil(total * floor_share)
    planned = {stratum: min(count, floor) for stratum, count in population.items()}
    while (left := total - sum(planned.values())) > 0:
        room = {stratum: population[stratum] - planned[stratum] for stratum in population}
        open_strata = [stratum for stratum in population if room[stratum] > 0]
        weights = {
            stratum: population[stratum] * math.sqrt(prior[stratum] * (1.0 - prior[stratum])) for stratum in open_strata
        }
        if sum(weights.values()) <= 0.0:
            weights = {stratum: float(room[stratum]) for stratum in open_strata}
        scale = sum(weights.values())
        shares = {stratum: left * weights[stratum] / scale for stratum in open_strata}
        given = {stratum: min(room[stratum], math.floor(shares[stratum])) for stratum in open_strata}
        remainder = left - sum(given.values())
        for stratum in sorted(open_strata, key=lambda s: (-(shares[s] - math.floor(shares[s])), s)):
            if remainder <= 0:
                break
            if given[stratum] < room[stratum]:
                given[stratum] += 1
                remainder -= 1
        if not any(given.values()):
            # A share below one draw everywhere: the largest weight takes one, so the loop always moves.
            given[max(open_strata, key=lambda s: (weights[s], s))] = 1
        for stratum, count in given.items():
            planned[stratum] += count
    return planned


def item_view(grade: ClaimGrade, evidence: ClaimEvidence, place: int) -> ClaimItemView:
    """One graded claim as a rater reads it: the claim, its excerpt and segment, every piece of evidence the grade
    numbered for the cell and every window its web check of the claim read, and nothing the grade concluded.

    :param grade: The cell's grade.
    :param evidence: The cell's claim evidence, which the grade graded.
    :param place: The claim's place in the grade.
    :return: The view.
    :raises ValueError: The grade numbers a passage of a document the evidence does not hold.
    """
    claim = grade.claims[place]
    documents = {document.id: document for document in evidence.documents or ()}
    shown: list[ShownEvidence] = []
    for entry in grade.evidence:
        if entry.fact is not None and entry.entity is not None:
            shown.append(
                ShownEvidence(ref=entry.ref, provenance=entry.provenance, title=entry.entity, text=entry.fact.claim)
            )
            continue
        assert entry.document is not None and entry.span is not None  # an EvidenceRef is a passage or a key fact
        document = documents.get(entry.document)
        if document is None:
            raise ValueError(
                f"the grade numbers {entry.ref!r} of document {entry.document!r}, which the evidence lacks"
            )
        shown.append(
            ShownEvidence(
                ref=entry.ref,
                provenance=entry.provenance,
                title=document.title,
                text=document.text[entry.span.start : entry.span.end],
            )
        )
    record = next((check for check in grade.web if check.claim == place), None)
    for at, window in enumerate(record.windows if record is not None else (), start=1):
        shown.append(ShownEvidence(ref=f"W{at}", provenance="web", title=window.title or window.url, text=window.text))
    return ClaimItemView(
        claim=claim.claim,
        excerpt=claim.excerpt,
        start=claim.start,
        end=claim.end,
        segment=evidence.segments[claim.segment].text,
        evidence=shown,
    )


@dataclass(frozen=True)
class _Candidate:
    arm: str
    cell: GradedCell
    place: int
    stratum: ClaimStratum


def _stratum(stratify_by: ClaimStratifier, verdict: ClaimTruthVerdict, grounded: bool | None) -> ClaimStratum | None:
    if stratify_by == "truth":
        return verdict
    if grounded is None:
        return None
    return "grounded" if grounded else "ungrounded"


def draw_claim_read(
    cells: Iterable[GradedCell],
    arms: Sequence[ClaimReadArm],
    *,
    criterion: str,
    criterion_fingerprint: str,
    questions: Sequence[ClaimQuestion],
    scope_id: str,
    created_by: str,
    name: str = "claim read",
    seed: int = CLAIM_READ_SEED,
    stratify_by: ClaimStratifier = "truth",
    planned_items: int | None = None,
    planned: Mapping[ClaimStratum, int] | None = None,
    prior: Mapping[ClaimStratum, float] = MISS_PRIOR,
    per_cell: int = CLAIMS_PER_CELL,
    campaign_id: str | None = None,
    created_at: str | None = None,
) -> ClaimRead:
    """Draw a claim read over the arms' graded cells: the population by stratum, the plan, and the items.

    The population is every world claim that restates nothing, in a cell of one of the arms' runs whose grade was made
    under ``criterion_fingerprint`` over the evidence stored beside it and read no recorded search; with groundedness
    asked or stratified by, only claims a ground step read. Each stratum's order is a seeded digest per claim, sorted,
    so nothing drawn depends on the order cells are listed in. The strata are taken in turn, one claim each, each from
    the top of its order skipping claims of a cell that has given ``per_cell`` items, until each has its plan or runs
    out. The items are listed in another seeded order, so where an item sits says nothing of its stratum.

    :param cells: The graded cells, each under ``criterion``.
    :param arms: The arms, each naming its runs.
    :param criterion: The claim criterion.
    :param criterion_fingerprint: The definition every drawn grade must have been made under.
    :param questions: What each item asks.
    :param scope_id: The scope the read belongs to.
    :param created_by: Who drew it.
    :param name: What the read is called.
    :param seed: The seed.
    :param stratify_by: The instrument answer the strata are.
    :param planned_items: The draws planned in all, allocated by :func:`allocate_draws`; None for
        :data:`PLANNED_ITEMS_DEFAULT`. Ignored where ``planned`` is given.
    :param planned: Each stratum's draws, as planned by the caller; a stratum left out plans none.
    :param prior: People's expected fail share per stratum, which the allocation reads.
    :param per_cell: The most items one cell gives.
    :param campaign_id: The campaign drawn over, where there is one.
    :param created_at: When it was drawn; None for now.
    :return: The read, with no answers.
    :raises ValueError: No question, a question twice, a run named by two arms, or a planned count past its stratum.
    """
    if not questions or len(set(questions)) != len(questions):
        raise ValueError(f"a claim read asks at least one question, each once; got {list(questions)}")
    arm_of_run: dict[str, str] = {}
    for arm in arms:
        for run_id in arm.run_ids:
            if arm_of_run.setdefault(run_id, arm.key) != arm.key:
                raise ValueError(f"run {run_id!r} is named by two arms of the claim read")
    needs_ground = stratify_by == "grounded" or "grounded" in questions
    excluded: Counter[str] = Counter()
    candidates: list[_Candidate] = []
    for cell in cells:
        arm_key = arm_of_run.get(cell.run_id)
        if arm_key is None:
            continue
        # prose-canary: allow — `scope` is the closed Literal world/session the extractor checked, not free text
        world = [place for place, claim in enumerate(cell.grade.claims) if claim.scope == "world"]
        if cell.grade.criterion_fingerprint != criterion_fingerprint:
            excluded[EXCLUDED_REDEFINED] += len(world)
            continue
        if cell.grade.claim_evidence_digest != claim_evidence_digest(cell.evidence):
            excluded[EXCLUDED_OTHER_EVIDENCE] += len(world)
            continue
        if replayed_state(cell.grade) is not None:
            excluded[EXCLUDED_REPLAYED] += len(world)
            continue
        for place in world:
            claim = cell.grade.claims[place]
            if claim.restates is not None:
                excluded[EXCLUDED_RESTATEMENT] += 1
                continue
            if needs_ground and claim.grounded is None:
                excluded[EXCLUDED_NO_GROUND] += 1
                continue
            assert claim.verdict is not None  # a world claim has a verdict (GradedClaim's own rule)
            stratum = _stratum(stratify_by, claim.verdict, claim.grounded)
            assert stratum is not None  # a claim without a groundedness reading was left out above
            candidates.append(_Candidate(arm_key, cell, place, stratum))
    strata = STRATA_OF[stratify_by]
    digest = canonical_digest(
        {
            "claim_read_population": sorted(
                (c.arm, c.cell.result_id, c.cell.grade.claim_evidence_digest, c.place, c.stratum) for c in candidates
            ),
            "criterion": criterion,
            "fingerprint": criterion_fingerprint,
        }
    )
    by_stratum: dict[ClaimStratum, list[_Candidate]] = {stratum: [] for stratum in strata}
    for candidate in candidates:
        by_stratum[candidate.stratum].append(candidate)
    for stratum, members in by_stratum.items():
        members.sort(key=lambda c: _sha("claim-order", seed, digest, stratum, c.cell.result_id, c.place))
    population = {stratum: len(members) for stratum, members in by_stratum.items()}
    if planned is None:
        total = PLANNED_ITEMS_DEFAULT if planned_items is None else max(planned_items, 0)
        plan = allocate_draws(population, total, prior=prior)
    else:
        plan = {stratum: planned.get(stratum, 0) for stratum in strata}
        if past := sorted(s for s in strata if not 0 <= plan[s] <= population[s]):
            raise ValueError(f"strata {past} are planned past the claims they hold ({population})")
    taken: dict[ClaimStratum, list[_Candidate]] = {stratum: [] for stratum in strata}
    cursor = dict.fromkeys(strata, 0)
    per_result: Counter[str] = Counter()
    moving = True
    while moving:
        moving = False
        for stratum in strata:
            members = by_stratum[stratum]
            while len(taken[stratum]) < plan[stratum] and cursor[stratum] < len(members):
                candidate = members[cursor[stratum]]
                cursor[stratum] += 1
                if per_result[candidate.cell.result_id] >= per_cell:
                    continue
                per_result[candidate.cell.result_id] += 1
                taken[stratum].append(candidate)
                moving = True
                break
    items = []
    for stratum in strata:
        for candidate in taken[stratum]:
            claim = candidate.cell.grade.claims[candidate.place]
            view = item_view(candidate.cell.grade, candidate.cell.evidence, candidate.place)
            assert claim.verdict is not None
            items.append(
                ClaimItem(
                    item=_sha("claim-item", seed, digest, candidate.cell.result_id, candidate.place)[:12],
                    stratum=stratum,
                    arm=candidate.arm,
                    run_id=candidate.cell.run_id,
                    result_id=candidate.cell.result_id,
                    case_id=candidate.cell.case_id,
                    claim=candidate.place,
                    verdict=claim.verdict,
                    grounded=claim.grounded,
                    view_digest=item_view_digest(view),
                    view=view,
                )
            )
    items.sort(key=lambda item: _sha("claim-sheet", seed, digest, item.item))
    return ClaimRead(
        id=f"claimread-{_sha('claim-read', seed, digest, stratify_by, *questions)[:12]}",
        scope_id=scope_id,
        name=name,
        created_at=created_at if created_at is not None else utc_now_iso(),
        created_by=created_by,
        campaign_id=campaign_id,
        criterion=criterion,
        criterion_fingerprint=criterion_fingerprint,
        questions=list(questions),
        arms=list(arms),
        draw=ClaimDraw(
            seed=seed,
            stratify_by=stratify_by,
            per_cell=per_cell,
            population_digest=digest,
            strata=[
                ClaimStratumDraw(
                    stratum=stratum,
                    population=population[stratum],
                    cells=len({c.cell.result_id for c in by_stratum[stratum]}),
                    planned=plan[stratum],
                    drawn=len(taken[stratum]),
                )
                for stratum in strata
            ],
            excluded=dict(sorted(excluded.items())),
        ),
        items=items,
    )


# --- reading ---------------------------------------------------------------------------------------------------------


class Estimate(NamedTuple):
    """A figure with its interval: the lower end one-sided 95%, the upper one-sided 97.5%; None where undefined."""

    value: float | None
    lower: float | None = None
    upper: float | None = None


_UNDEFINED: Final = Estimate(None)


@dataclass(frozen=True)
class StratumAgreement:
    """How people stood against the instrument in one stratum, unweighted.

    :ivar stratum: The stratum.
    :ivar population: Its claims.
    :ivar items: Its items people answered the question on.
    :ivar agreed: The share of them people answered as the instrument did (a split between people counting by its
        share); None with no item.
    """

    stratum: ClaimStratum
    population: int
    items: int
    agreed: float | None


@dataclass(frozen=True)
class ArmCalibration:
    """One arm's figures on one question, read through the strata's weights.

    :ivar arm: The arm.
    :ivar items: Its answered items.
    :ivar cells: The distinct cells they come from.
    :ivar recall: Of the arm's claims people fail, the share the instrument fails.
    :ivar precision: Of the arm's claims the instrument fails, the share people fail.
    :ivar bias: The instrument's fail rate over people's, on the arm's claims.
    :ivar failed: The share of the arm's claims people fail.
    """

    arm: str
    items: int
    cells: int
    recall: Estimate
    precision: Estimate
    bias: Estimate
    failed: Estimate


@dataclass(frozen=True)
class QuestionCalibration:
    """One question's calibration.

    :ivar question: The question.
    :ivar items: The items people answered it on.
    :ivar cells: The distinct cells they come from: what the calibration floor counts.
    :ivar cases: The distinct cases.
    :ivar skipped: Items people skipped or left this question unanswered on.
    :ivar table: The population's shares, weighted: (both fail, only the instrument fails, only people fail, both
        pass); None with no answered item.
    :ivar kappa: Kappa on the weighted table, with its stratified bootstrap interval.
    :ivar recall: Detection recall, of the claims people fail.
    :ivar precision: Precision, of the claims the instrument fails.
    :ivar bias: The instrument's fail rate over people's.
    :ivar failed: The share of claims people fail; on ``is_claim``, one less the extractor's precision.
    :ivar strata: Each stratum's unweighted agreement.
    :ivar unread_strata: Strata holding claims and no answered item, whose share no figure can stand for.
    :ivar arms: Each arm's figures.
    :ivar lean_p: The homogeneity test's p across arms on recall and precision; None where fewer than two arms carry
        both (or the question has no instrument answer to lean).
    :ivar check: The check of the instrument on this question; None on ``is_claim``, where the instrument passes
        every claim it listed and only people's share is read.
    """

    question: ClaimQuestion
    items: int
    cells: int
    cases: int
    skipped: int
    table: tuple[float, float, float, float] | None
    kappa: Estimate
    recall: Estimate
    precision: Estimate
    bias: Estimate
    failed: Estimate
    strata: tuple[StratumAgreement, ...]
    unread_strata: tuple[ClaimStratum, ...]
    arms: tuple[ArmCalibration, ...]
    lean_p: float | None
    check: Check | None


@dataclass(frozen=True)
class ClaimCalibration:
    """What a claim read says of its instrument: one reading per question, and the answers no figure reads.

    :ivar read_id: The read.
    :ivar instrument: The instrument, as a check names it: ``claims:<criterion>@<fingerprint, 12>``.
    :ivar questions: One reading per question the read asks, in its order.
    :ivar model_answers: Answers a model gave, read in no figure.
    :ivar borrowed: Person answers taken from earlier reads whose item showed a byte-identical view.
    """

    read_id: str
    instrument: str
    questions: tuple[QuestionCalibration, ...]
    model_answers: int
    borrowed: int


def instrument_fails(item: ClaimItem, question: ClaimQuestion) -> int:
    """Whether the instrument failed the item's claim on ``question``: 1 when it read the claim false or not grounded;
    0 otherwise, and always on ``is_claim``, since it listed the claim.

    :param item: The item.
    :param question: The question.
    :return: 1 or 0.
    """
    if question == "true":
        return int(item.verdict == "contradicted")
    if question == "grounded":
        return int(item.grounded is False)
    return 0


@dataclass(frozen=True)
class _Point:
    stratum: int
    arm: int
    x: int
    y: float


class _Sums:
    """Weighted sums over points: weight, the instrument's fails, people's fails, and both."""

    __slots__ = ("both", "instrument", "people", "weight")

    def __init__(self) -> None:
        self.weight = 0.0
        self.instrument = 0.0
        self.people = 0.0
        self.both = 0.0

    def add(self, weight: float, x: int, y: float) -> None:
        self.weight += weight
        self.instrument += weight * x
        self.people += weight * y
        self.both += weight * x * y


class _Figures(NamedTuple):
    kappa: float | None
    recall: float | None
    precision: float | None
    bias: float | None
    failed: float | None


def _figures(sums: _Sums) -> _Figures:
    """Kappa, recall, precision, bias and people's fail share off one set of weighted sums."""
    if sums.weight <= 0.0:
        return _Figures(None, None, None, None, None)
    p_i = sums.instrument / sums.weight
    p_h = sums.people / sums.weight
    both = sums.both / sums.weight
    neither = 1.0 - p_i - p_h + both
    expected = p_i * p_h + (1.0 - p_i) * (1.0 - p_h)
    kappa = None if expected >= 1.0 - 1e-12 else ((both + neither) - expected) / (1.0 - expected)
    recall = both / p_h if p_h > 0.0 else None
    precision = both / p_i if p_i > 0.0 else None
    bias = p_i / p_h if p_h > 0.0 else None
    return _Figures(kappa, recall, precision, bias, p_h)


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    at = q * (len(ordered) - 1)
    low = math.floor(at)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (at - low)


def _estimate(value: float | None, replicates: Sequence[float | None], draws: int) -> Estimate:
    """A figure with its interval: the bootstrap draws' spread about their median, set about the figure.

    The smoothed draws (:func:`_smoothed_draw`) sit a little toward even odds, which is what gives a stratum of
    unanimous answers its spread; set about their own median, only that spread is read, never the pull. No interval
    when fewer than half the draws give the figure.
    """
    if value is None:
        return _UNDEFINED
    defined = [r for r in replicates if r is not None]
    if draws == 0 or len(defined) < _DEFINED_SHARE * draws:
        return Estimate(value)
    middle = _percentile(defined, 0.5)
    return Estimate(
        value,
        min(value, value + _percentile(defined, _LOWER) - middle),
        max(value, value + _percentile(defined, _UPPER) - middle),
    )


def _weighted(points: Sequence[_Point], weights: Sequence[float], arms: int) -> tuple[_Sums, list[_Sums]]:
    overall = _Sums()
    per_arm = [_Sums() for _ in range(arms)]
    for point in points:
        weight = weights[point.stratum]
        overall.add(weight, point.x, point.y)
        per_arm[point.arm].add(weight, point.x, point.y)
    return overall, per_arm


#: The pseudo-count each answer, fail and pass, gets in each bootstrap draw: Jeffreys' half. Without it a stratum whose items people all passed draws no spread at all, though
#: its weight carries the most uncertainty of any: the instrument's misses among the claims it supported.
_PSEUDO_COUNT: Final = 0.5


def _groups(by_stratum: Sequence[Sequence[_Point]]) -> list[dict[tuple[int, int], list[float]]]:
    """Per stratum, per (arm, instrument fail), people's answers counted as (passes, fails), a split counted by its
    share."""
    grouped: list[dict[tuple[int, int], list[float]]] = []
    for members in by_stratum:
        groups: dict[tuple[int, int], list[float]] = {}
        for point in members:
            counts = groups.setdefault((point.arm, point.x), [0.0, 0.0])
            counts[0] += 1.0 - point.y
            counts[1] += point.y
        grouped.append(groups)
    return grouped


def _smoothed_draw(
    groups: Sequence[Mapping[tuple[int, int], Sequence[float]]],
    population: Sequence[int],
    arms: int,
    rng: random.Random,
    *,
    per_arm: bool,
) -> tuple[_Sums, list[_Sums]]:
    """One draw of a stratified Bayesian bootstrap: each stratum's population spread over its answered groups by a
    Dirichlet draw on their counts, each raised by :data:`_PSEUDO_COUNT`.

    ``per_arm`` False spreads each instrument answer's pseudo-count over the arms by their share of its items, so the
    population figures are smoothed once, as one stratum; True gives each arm its own, so an arm whose items people all
    passed still draws its own spread, which the per-arm figures and the lean test read.
    """
    overall = _Sums()
    by_arm = [_Sums() for _ in range(arms)]
    for stratum, held in enumerate(groups):
        if not held:
            continue
        answered = {x: sum(sum(c) for (_, at_x), c in held.items() if at_x == x) for _, x in held}
        masses = [
            (
                arm,
                x,
                y,
                rng.gammavariate(counts[y] + _PSEUDO_COUNT * (1.0 if per_arm else sum(counts) / answered[x]), 1.0),
            )
            for (arm, x), counts in held.items()
            for y in (0, 1)
        ]
        total = sum(mass for *_, mass in masses)
        for arm, x, y, mass in masses:
            share = population[stratum] * mass / total
            overall.add(share, x, y)
            by_arm[arm].add(share, x, y)
    return overall, by_arm


def _variance(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)


#: The permutations the lean test draws: its p is read to within about a hundredth near the 5% level.
LEAN_PERMUTATIONS: Final = 999


def lean_test(
    points: Sequence[tuple[int, int, int]],
    *,
    permutations: int = LEAN_PERMUTATIONS,
    rng: random.Random,
) -> float | None:
    """One test across every arm of whether the instrument fails claims at a rate that depends on the arm, among
    claims people answered alike: a conditional permutation test of the instrument's answer and the arm, given
    people's answer.

    A fair instrument fails the claims people fail at one rate (its recall) and the claims people pass at one rate (its
    false fails), whatever arm said them, though arms differ in how often they say false things. So the null is that
    the instrument's answer and the arm are independent given people's answer, and the test permutes the arm labels
    among the items people failed, and apart from them among those people passed. The draw is stratified by the
    instrument's answer alone, so it keeps that independence in the sample. Precision and rate bias move with an arm's
    own share of false claims, so they are read per arm and never tested here.

    The statistic sums, over people's two answers, each arm's items times its squared distance from the pooled fail
    share, over the pooled share's binomial variance; the p is ``(1 + permutations at least as large) / (1 +
    permutations)``.

    :param points: Each answered item's (people fail it, 1 or 0; the instrument fails it, 1 or 0; its arm's index).
        An item people split on evenly is left out.
    :param permutations: The permutations drawn.
    :param rng: The generator the permutations are drawn by.
    :return: The p, or None where neither of people's answers holds items of two arms with the instrument answering
        both ways among them.
    """
    groups: dict[int, tuple[list[int], list[int]]] = {}
    for people, instrument, arm in points:
        xs, arms = groups.setdefault(people, ([], []))
        xs.append(instrument)
        arms.append(arm)
    informative = [(xs, arms) for xs, arms in groups.values() if len(set(arms)) >= 2 and 0 < sum(xs) < len(xs)]
    if not informative:
        return None

    def statistic(assignments: Sequence[Sequence[int]]) -> float:
        total = 0.0
        for (xs, _), arms in zip(informative, assignments, strict=True):
            pooled = sum(xs) / len(xs)
            fails: Counter[int] = Counter()
            counts: Counter[int] = Counter(arms)
            for x, arm in zip(xs, arms, strict=True):
                fails[arm] += x
            total += sum(counts[arm] * (fails[arm] / counts[arm] - pooled) ** 2 for arm in counts) / (
                pooled * (1.0 - pooled)
            )
        return total

    observed = statistic([arms for _, arms in informative])
    larger = 0
    shuffled = [list(arms) for _, arms in informative]
    for _ in range(permutations):
        for arms in shuffled:
            rng.shuffle(arms)
        larger += statistic(shuffled) >= observed - 1e-12
    return (1 + larger) / (1 + permutations)


def _persons_verdicts(answer: ClaimAnswer) -> list[ClaimVerdict]:
    """The verdicts an answer gives as a person's: none of a model's answer, or of a skip."""
    if answer.made_by != "person" or answer.skipped:
        return []
    return answer.verdicts


def _person_scores(
    read: ClaimRead, earlier: Sequence[ClaimRead]
) -> tuple[dict[str, dict[ClaimQuestion, list[int]]], dict[str, int], int, int]:
    """Per item, per question, each person's score; per item, the skips; the model answers; the answers borrowed."""
    by_view: dict[str, list[str]] = {}
    for item in read.items:
        by_view.setdefault(item.view_digest, []).append(item.item)
    scores: dict[str, dict[ClaimQuestion, dict[str, int]]] = {item.item: {} for item in read.items}
    skips: Counter[str] = Counter()
    models = borrowed = 0
    for answer in read.answers:
        if answer.made_by != "person":
            models += 1
            continue
        if answer.skipped:
            skips[answer.item] += 1
            continue
        for verdict in answer.verdicts:
            scores[answer.item].setdefault(verdict.question, {})[answer.rater] = verdict.score
    for other in earlier:
        if other.id == read.id or other.criterion_fingerprint != read.criterion_fingerprint:
            continue
        views = {item.item: item.view_digest for item in other.items}
        for answer in other.answers:
            for item_id in by_view.get(views.get(answer.item, ""), ()):
                for verdict in _persons_verdicts(answer):
                    if verdict.question in read.questions:
                        held = scores[item_id].setdefault(verdict.question, {})
                        if answer.rater not in held:
                            held[answer.rater] = verdict.score
                            borrowed += 1
    return (
        {item: {q: list(by.values()) for q, by in asked.items()} for item, asked in scores.items()},
        dict(skips),
        models,
        borrowed,
    )


def read_claim_calibration(
    read: ClaimRead,
    *,
    earlier: Sequence[ClaimRead] = (),
    draws: int = BOOTSTRAP_DRAWS,
    seed: int | None = None,
) -> ClaimCalibration:
    """Read a claim read's answers against its instrument, per question: weighted, with intervals and per arm.

    People's answer on an item is their mean fail (0 passes, 1 fails; two people split count a half each). An item
    stands for its stratum's population over the stratum's items answered on the question. ``earlier`` lends person
    answers given on a byte-identical view of another read of the same criterion definition, a person's own answer
    on this read standing over a borrowed one.

    :param read: The read.
    :param earlier: Other claim reads whose answers may stand for this read's items.
    :param draws: The bootstrap draws.
    :param seed: The bootstrap's seed; None reads it off the read's id, so a read reads the same every time.
    :return: The calibration.
    """
    scores, skips, models, borrowed = _person_scores(read, earlier)
    strata = [stratum.stratum for stratum in read.draw.strata]
    stratum_index = {stratum: at for at, stratum in enumerate(strata)}
    population = {stratum.stratum: stratum.population for stratum in read.draw.strata}
    arm_keys = [arm.key for arm in read.arms]
    arm_index = {key: at for at, key in enumerate(arm_keys)}
    instrument = f"claims:{read.criterion}@{read.criterion_fingerprint[:12]}"
    replayed = read.draw.excluded.get(EXCLUDED_REPLAYED, 0) > 0 and not any(population.values())
    readings = []
    for question in read.questions:
        points: list[_Point] = []
        answered: list[ClaimItem] = []
        unanswered = 0
        for item in read.items:
            given = scores[item.item].get(question, [])
            if not given:
                unanswered += 1
                continue
            answered.append(item)
            fails = sum(1 - score for score in given) / len(given)
            points.append(
                _Point(stratum_index[item.stratum], arm_index[item.arm], instrument_fails(item, question), fails)
            )
        counts = Counter(point.stratum for point in points)
        weights = [population[s] / counts[at] if counts[at] else 0.0 for at, s in enumerate(strata)]
        unread = tuple(s for at, s in enumerate(strata) if population[s] > 0 and counts[at] == 0)
        overall, per_arm = _weighted(points, weights, len(arm_keys))
        figures = _figures(overall)
        arm_figures = [_figures(sums) for sums in per_arm]
        rng = random.Random(seed if seed is not None else int(_sha("claim-calibration", read.id, question)[:16], 16))
        by_stratum: list[list[_Point]] = [[] for _ in strata]
        for point in points:
            by_stratum[point.stratum].append(point)
        groups = _groups(by_stratum)
        sizes = [population[s] for s in strata]
        boot: list[_Figures] = []
        boot_arms: list[list[_Figures]] = []
        for _ in range(draws if points else 0):
            sums, _ = _smoothed_draw(groups, sizes, len(arm_keys), rng, per_arm=False)
            _, sums_by_arm = _smoothed_draw(groups, sizes, len(arm_keys), rng, per_arm=True)
            boot.append(_figures(sums))
            boot_arms.append([_figures(s) for s in sums_by_arm])
        taken = len(boot)

        def est(value: float | None, pick: int, replicates: Sequence[_Figures]) -> Estimate:
            return _estimate(value, [r[pick] for r in replicates], taken)

        arms = tuple(
            ArmCalibration(
                arm=key,
                items=sum(1 for p in points if p.arm == at),
                cells=len({item.result_id for item in answered if item.arm == key}),
                recall=est(arm_figures[at].recall, 1, [b[at] for b in boot_arms]),
                precision=est(arm_figures[at].precision, 2, [b[at] for b in boot_arms]),
                bias=est(arm_figures[at].bias, 3, [b[at] for b in boot_arms]),
                failed=est(arm_figures[at].failed, 4, [b[at] for b in boot_arms]),
            )
            for at, key in enumerate(arm_keys)
        )
        leans = question != "is_claim" and len(arm_keys) > 1
        lean_p = (
            lean_test(
                [(int(point.y > 0.5), point.x, point.arm) for point in points if point.y != 0.5],
                rng=random.Random(rng.random()),
            )
            if leans
            else None
        )
        kappa = est(figures.kappa, 0, boot)
        cells = len({item.result_id for item in answered})
        check = None
        if question != "is_claim":
            check = _check(
                instrument=instrument,
                criterion=f"{read.criterion}.{question}",
                read_id=read.id,
                items=len(points),
                cells=cells,
                cases=len({item.case_id for item in answered}),
                skipped=sum(skips.values()),
                kappa=kappa,
                unread=bool(unread),
                lean_p=lean_p,
                replayed=replayed,
            )
        readings.append(
            QuestionCalibration(
                question=question,
                items=len(points),
                cells=cells,
                cases=len({item.case_id for item in answered}),
                skipped=unanswered,
                table=None
                if overall.weight <= 0.0
                else (
                    overall.both / overall.weight,
                    (overall.instrument - overall.both) / overall.weight,
                    (overall.people - overall.both) / overall.weight,
                    1.0 - (overall.instrument + overall.people - overall.both) / overall.weight,
                ),
                kappa=kappa,
                recall=est(figures.recall, 1, boot),
                precision=est(figures.precision, 2, boot),
                bias=est(figures.bias, 3, boot),
                failed=est(figures.failed, 4, boot),
                strata=tuple(
                    StratumAgreement(
                        stratum=s,
                        population=population[s],
                        items=counts[at],
                        agreed=None
                        if not counts[at]
                        else sum(1.0 - abs(p.x - p.y) for p in by_stratum[at]) / counts[at],
                    )
                    for at, s in enumerate(strata)
                ),
                unread_strata=unread,
                arms=arms,
                lean_p=lean_p,
                check=check,
            )
        )
    return ClaimCalibration(
        read_id=read.id, instrument=instrument, questions=tuple(readings), model_answers=models, borrowed=borrowed
    )


def _check(
    *,
    instrument: str,
    criterion: str,
    read_id: str,
    items: int,
    cells: int,
    cases: int,
    skipped: int,
    kappa: Estimate,
    unread: bool,
    lean_p: float | None,
    replayed: bool,
) -> Check:
    """The check of the instrument on one question: kappa held to the calibration bar over cells, a lean across arms
    reading ``not_met`` (``instrument_leans``), and a read whose every claim was replayed reading not available."""
    if replayed:
        return Check(
            instrument=instrument,
            criterion=criterion,
            against="people",
            sample="none",
            read_id=read_id,
            result="insufficient",
            state=StateOf(state="not_available", reason="replayed"),
        )
    interval = None if kappa.lower is None or kappa.upper is None else (kappa.lower, kappa.upper)
    tier = calibration_criterion(
        items, min(cells, items), None if unread else kappa.value, None if unread else interval
    )
    result = tier.state
    reason: Reason | None = "instrument_below_bar" if result == "not_met" else None
    if lean_p is not None and lean_p <= SIGNIFICANCE_ALPHA:
        result, reason = "not_met", "instrument_leans"
    return Check(
        instrument=instrument,
        criterion=criterion,
        against="people",
        sample="random" if items else "none",
        sample_results=cells,
        sample_cases=cases,
        skipped=skipped,
        read_id=read_id,
        result=result,
        reason=reason,
        kappa=kappa.value,
        kappa_interval=interval,
        reads=items,
    )


__all__: list[str] = []
