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
  figure is inverse-probability weighted: an answered item stands for one over its chance of being drawn (its
  stratum's rate times its chance of surviving its cell's cap), raised for its stratum's unanswered draws. Per question: the population's 2x2 (the instrument fails a claim, people fail it), kappa on it with
  a smoothed stratified bootstrap interval (:func:`_campaign_draw`), the per-stratum agreement, and **detection
  recall** (of claims people fail, the share the instrument fails), the **false-fail rate** (of claims people pass,
  the share the instrument fails), **precision** (of those the instrument fails, the share people fail) and **rate
  bias** (the instrument's fail rate over people's), overall and per arm. Whether the instrument leans by arm is
  **one homogeneity test across every arm** on recall and the false-fail rate together (:func:`lean_test`), never
  pairwise intervals: with several arms, pairwise tests withhold far more often than their level. Those two are the
  rates a fair instrument holds equal across arms whatever each arm's share of false claims; precision and bias move
  with that share, so they are shown per arm and never tested. A lean is the check's ``not_met`` with reason
  ``instrument_leans``. What the draw left out, and why, is carried on the reading beside the figures.

**Two readings of a figure.** The population's figures stand for the campaign's own claims, so their bootstrap holds
each stratum to its population. An arm's figures, and the lean test across arms, stand for how the instrument behaves
on the arm, of which the campaign's claims are one draw: their bootstrap gives every cell one weight for all its items
(two claims of one cell are never read as two independent ones) and holds no stratum to its population.
``tests/stats_certificate.py`` certifies kappa's interval, one arm's recall's interval and the lean test's level, on
arms alike and on arms whose verdict mixes, base rates and cell sizes differ sharply, and the test's power; the unit
suite keeps a fast seeded case of each (``tests/test_simulated_claim_read.py``). An answer left unanswered is taken as
missing at random within its stratum.
A model's answer is never a person's: it is counted and read in no figure.

Engine-internal: nothing here is exported, and no host can draw or read a claim read yet; the host door is its own
change.
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
    ClaimExclusion,
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

#: How likely people are to fail a claim in each stratum: a planning guess Neyman allocation spreads draws by, never a
#: figure read. A wrong guess widens a figure's interval, never moves its weighting, since every figure is weighted by the
#: draws each stratum actually gave.
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

#: Why a claim was left out of a draw (``ClaimDraw.excluded``): its grade was made under another definition; its
#: stored evidence is not what was graded; a recording served its grade's web results (no live search of this run
#: stands behind it); it restates an earlier claim, whose verdict it takes; no ground step read it.
EXCLUDED_REDEFINED: Final[ClaimExclusion] = "redefined"
EXCLUDED_OTHER_EVIDENCE: Final[ClaimExclusion] = "other_evidence"
EXCLUDED_REPLAYED: Final[ClaimExclusion] = "replayed"
EXCLUDED_RESTATEMENT: Final[ClaimExclusion] = "restatement"
EXCLUDED_NO_GROUND: Final[ClaimExclusion] = "no_ground_reading"


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


def _unit(*parts: object) -> float:
    """A seeded uniform in [0, 1) read off a digest of ``parts``."""
    return int(_sha(*parts)[:13], 16) / 16**13


def _kept_given_selected(others: Sequence[float], per_cell: int) -> float:
    """The chance a selected claim is kept under the cap, when each other claim of its cell is selected at its own
    rate: ``E[min(1, per_cell / (1 + K))]``, K the others selected (a Poisson-binomial count)."""
    counts = [1.0]
    for p in others:
        grown = [0.0] * (len(counts) + 1)
        for k, mass in enumerate(counts):
            grown[k] += mass * (1.0 - p)
            grown[k + 1] += mass * p
        counts = grown
    return sum(mass * min(1.0, per_cell / (k + 1)) for k, mass in enumerate(counts))


def _select(
    candidates: Sequence[_Candidate],
    rate: Mapping[ClaimStratum, float],
    per_cell: int,
    *,
    seed: int,
    digest: str,
) -> tuple[dict[ClaimStratum, list[_Candidate]], dict[tuple[str, int], float]]:
    """Select each claim at its stratum's rate by a seeded coin, then keep at most ``per_cell`` of a cell's selected
    claims, chosen in a seeded order; and each claim's chance of being drawn so, exactly.

    A cap on a cell makes a claim of a claim-heavy cell less likely to be drawn than its stratum's rate. Drawn so, the
    chance is known for every claim (its stratum's rate times its chance of surviving the cap among the other claims of
    its cell), so every figure weights a drawn claim by one over it and no cell's size or mix leans a figure.
    """
    by_cell: dict[str, list[_Candidate]] = {}
    for candidate in candidates:
        by_cell.setdefault(candidate.cell.result_id, []).append(candidate)
    taken: dict[ClaimStratum, list[_Candidate]] = {stratum: [] for stratum in rate}
    inclusion: dict[tuple[str, int], float] = {}
    for result_id in sorted(by_cell):
        members = sorted(by_cell[result_id], key=lambda c: c.place)
        selected = [c for c in members if _unit("claim-select", seed, digest, result_id, c.place) < rate[c.stratum]]
        selected.sort(key=lambda c: _sha("claim-keep", seed, digest, result_id, c.place))
        for candidate in selected[:per_cell]:
            others = [rate[other.stratum] for other in members if other is not candidate]
            inclusion[(result_id, candidate.place)] = rate[candidate.stratum] * _kept_given_selected(others, per_cell)
            taken[candidate.stratum].append(candidate)
    return taken, inclusion


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
    asked or stratified by, only claims a ground step read. Each claim is selected by a seeded coin at its stratum's
    rate (the stratum's plan over its claims), and a cell that selected more than ``per_cell`` keeps that many, in a
    seeded order (:func:`_select`); every item records its exact chance of being drawn so. A stratum's draws are its
    plan in expectation, less what the cap took. Nothing drawn depends on the order cells are listed in. The items are
    listed in another seeded order, so where an item sits says nothing of its stratum.

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
    excluded: Counter[ClaimExclusion] = Counter()
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
    population = {stratum: len(members) for stratum, members in by_stratum.items()}
    if planned is None:
        total = PLANNED_ITEMS_DEFAULT if planned_items is None else max(planned_items, 0)
        plan = allocate_draws(population, total, prior=prior)
    else:
        plan = {stratum: planned.get(stratum, 0) for stratum in strata}
        if past := sorted(s for s in strata if not 0 <= plan[s] <= population[s]):
            raise ValueError(f"strata {past} are planned past the claims they hold ({population})")
    rate = {stratum: plan[stratum] / population[stratum] if population[stratum] else 0.0 for stratum in strata}
    taken, inclusion = _select(candidates, rate, per_cell, seed=seed, digest=digest)
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
                    inclusion=inclusion[(candidate.cell.result_id, candidate.place)],
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
                    by_arm=dict(sorted(Counter(c.arm for c in by_stratum[stratum]).items())),
                )
                for stratum in strata
            ],
            excluded=dict(sorted(excluded.items())),
        ),
        items=items,
    )


# --- reading ---------------------------------------------------------------------------------------------------------


class Estimate(NamedTuple):
    """A figure with its interval: the lower end one-sided 95%, the upper one-sided 97.5%, each held to the figure's
    range; None where undefined. An arm's recall or false-fail rate carries no interval where its answered claims expect
    fewer than :data:`LEAN_MIN_EXPECTED` on either side (an arm with three false claims answered, say), since no
    bootstrap of so few can say how far the rate might be. Kappa's ends and an arm's recall's are certified
    (``tests/stats_certificate.py``, ``claim-read.*``); the others' are read the same way and are not."""

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
    :ivar false_fail: Of the arm's claims people pass, the share the instrument fails.
    :ivar precision: Of the arm's claims the instrument fails, the share people fail.
    :ivar bias: The instrument's fail rate over people's, on the arm's claims.
    :ivar failed: The share of the arm's claims people fail.
    """

    arm: str
    items: int
    cells: int
    recall: Estimate
    false_fail: Estimate
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
    :ivar false_fail: The false-fail rate, of the claims people pass.
    :ivar precision: Precision, of the claims the instrument fails.
    :ivar bias: The instrument's fail rate over people's.
    :ivar failed: The share of claims people fail; on ``is_claim``, one less the extractor's precision.
    :ivar strata: Each stratum's unweighted agreement.
    :ivar unread_strata: Strata holding claims and no answered item, whose share no figure can stand for.
    :ivar arms: Each arm's figures.
    :ivar lean_p: The homogeneity test's p across arms on recall and the false-fail rate; None where fewer than two
        arms carry either with enough answered claims to read (:data:`LEAN_MIN_EXPECTED`), or the question has no
        instrument answer to lean: no lean is read, never a lean ruled out.
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
    false_fail: Estimate
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
    :ivar excluded: The claims the draw left out, by why (``ClaimDraw.excluded``): every figure is read over the rest,
        so a read with any left out stands for less than the campaign it was drawn over.
    """

    read_id: str
    instrument: str
    questions: tuple[QuestionCalibration, ...]
    model_answers: int
    borrowed: int
    excluded: Mapping[ClaimExclusion, int]


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
    """One answered item as the figures read it: its stratum, arm and cell, the instrument's fail and people's, and the
    claims it stands for (one over its chance of being drawn, raised for its stratum's unanswered draws)."""

    stratum: int
    arm: int
    cell: int
    x: int
    y: float
    weight: float


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
    false_fail: float | None


#: Each figure's range, which an interval's ends are held to, in :class:`_Figures` order.
_RANGES: Final = ((-1.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, math.inf), (0.0, 1.0), (0.0, 1.0))


def _figures(sums: _Sums) -> _Figures:
    """Kappa, recall, precision, bias, people's fail share and the false-fail rate off one set of weighted sums."""
    if sums.weight <= 0.0:
        return _Figures(None, None, None, None, None, None)
    p_i = sums.instrument / sums.weight
    p_h = sums.people / sums.weight
    both = sums.both / sums.weight
    neither = 1.0 - p_i - p_h + both
    expected = p_i * p_h + (1.0 - p_i) * (1.0 - p_h)
    kappa = None if expected >= 1.0 - 1e-12 else ((both + neither) - expected) / (1.0 - expected)
    recall = both / p_h if p_h > 0.0 else None
    precision = both / p_i if p_i > 0.0 else None
    bias = p_i / p_h if p_h > 0.0 else None
    false_fail = (p_i - both) / (1.0 - p_h) if p_h < 1.0 else None
    return _Figures(kappa, recall, precision, bias, p_h, false_fail)


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    at = q * (len(ordered) - 1)
    low = math.floor(at)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (at - low)


def _estimate(
    value: float | None, replicates: Sequence[float | None], draws: int, bounds: tuple[float, float]
) -> Estimate:
    """A figure with its interval: the bootstrap draws' spread about their median, set about the figure and held to
    the figure's range.

    The smoothed draws (:func:`_campaign_draw`, :func:`_instrument_draw`) sit a little toward even odds, which is what gives a stratum of
    unanimous answers its spread; set about their own median, only that spread is read, never the pull. No interval
    when fewer than half the draws give the figure.
    """
    if value is None:
        return _UNDEFINED
    defined = [r for r in replicates if r is not None]
    if draws == 0 or len(defined) < _DEFINED_SHARE * draws:
        return Estimate(value)
    middle = _percentile(defined, 0.5)
    low, high = bounds
    return Estimate(
        value,
        max(low, min(value, value + _percentile(defined, _LOWER) - middle)),
        min(high, max(value, value + _percentile(defined, _UPPER) - middle)),
    )


def _weighted(points: Sequence[_Point], arms: int) -> tuple[_Sums, list[_Sums]]:
    overall = _Sums()
    per_arm = [_Sums() for _ in range(arms)]
    for point in points:
        overall.add(point.weight, point.x, point.y)
        per_arm[point.arm].add(point.weight, point.x, point.y)
    return overall, per_arm


#: The pseudo-count each answer, fail and pass, gets in each bootstrap draw: Jeffreys' half. Without it a stratum
#: whose items people all passed draws no spread at all, though its weight carries the most uncertainty of any: the
#: instrument's misses among the claims it supported.
_PSEUDO_COUNT: Final = 0.5


@dataclass(frozen=True)
class _Group:
    """The answered items of one (stratum, arm, instrument fail): each people's answer's count and claims stood for."""

    stratum: int
    arm: int
    x: int
    counts: tuple[float, float]
    weights: tuple[float, float]

    @property
    def mean_weight(self) -> float:
        return sum(self.weights) / sum(self.counts)


def _groups(points: Sequence[_Point]) -> list[_Group]:
    """The points' groups, a split answer counted by its share."""
    held: dict[tuple[int, int, int], list[float]] = {}
    for point in points:
        sums = held.setdefault((point.stratum, point.arm, point.x), [0.0, 0.0, 0.0, 0.0])
        sums[0] += 1.0 - point.y
        sums[1] += point.y
        sums[2] += point.weight * (1.0 - point.y)
        sums[3] += point.weight * point.y
    return [
        _Group(stratum, arm, x, (sums[0], sums[1]), (sums[2], sums[3]))
        for (stratum, arm, x), sums in sorted(held.items())
    ]


def _campaign_draw(groups: Sequence[_Group], population: Sequence[int], rng: random.Random) -> _Sums:
    """One draw of the campaign's own figures: in each stratum, its population spread over its groups' answers by a
    Dirichlet draw on their counts, each instrument answer's pseudo-count spread over the arms by their share of its
    items, each group's answers standing for its mean claims per item."""
    overall = _Sums()
    by_stratum: dict[int, list[_Group]] = {}
    for group in groups:
        by_stratum.setdefault(group.stratum, []).append(group)
    for stratum, held in by_stratum.items():
        answered: dict[int, float] = {}
        for group in held:
            answered[group.x] = answered.get(group.x, 0.0) + sum(group.counts)
        masses = [
            (
                group.x,
                y,
                group.mean_weight
                * rng.gammavariate(group.counts[y] + _PSEUDO_COUNT * sum(group.counts) / answered[group.x], 1.0),
            )
            for group in held
            for y in (0, 1)
        ]
        total = sum(mass for *_, mass in masses)
        for x, y, mass in masses:
            overall.add(population[stratum] * mass / total, x, y)
    return overall


def _instrument_draw(
    points: Sequence[_Point], groups: Sequence[_Group], cells: int, arms: int, rng: random.Random
) -> list[_Sums]:
    """One draw of each arm's figures read as the instrument's behaviour on it, of which the campaign's claims are one
    draw: every cell's items share one Gamma(1) weight, so two claims of one cell are never read as two independent
    ones; and every group adds a pseudo-item per answer, a Gamma draw on :data:`_PSEUDO_COUNT` standing for its mean
    claims per item, so an arm whose items people all passed still draws its spread. Nothing is held to a stratum's
    population: a stratum drawn whole still carries the spread of how the instrument might have answered on it."""
    by_arm = [_Sums() for _ in range(arms)]
    multiplier = [rng.gammavariate(1.0, 1.0) for _ in range(cells)]
    for point in points:
        by_arm[point.arm].add(point.weight * multiplier[point.cell], point.x, point.y)
    for group in groups:
        for y in (0, 1):
            by_arm[group.arm].add(group.mean_weight * rng.gammavariate(_PSEUDO_COUNT, 1.0), group.x, y)
    return by_arm


#: The fewest answered claims an arm's rate must expect on each side, at the pooled rate, for the lean test to read it:
#: its answered fails times the pooled recall, and times one less it, for recall; its answered passes times the pooled
#: false-fail rate, and times one less it, for false fails. Below it the rate turns on a claim or two (an arm that
#: researched may have three false claims answered, and a false-fail rate near 2% expects none in thirty passes), which
#: no null drawn from the read can stand for, so it is shown and not tested: the rule a chi-square statistic is read by.
LEAN_MIN_EXPECTED: Final = 5.0

#: The draws the lean test's null is read from: its p is read to within about a hundredth near the 5% level.
LEAN_DRAWS: Final = 999


@dataclass(frozen=True)
class _LeanCell:
    """One (arm, stratum, instrument fail) of a read: its answered items, people's fails among them, and the claims
    they stand for."""

    arm: int
    x: int
    answered: int
    fails: float
    claims: float


class _LeanRates(NamedTuple):
    """Per arm, then pooled last: recall and the false-fail rate (None where undefined), and the lies and passes the
    answers stand for; per arm only, the lies and passes as answered."""

    recall: list[float | None]
    false_fail: list[float | None]
    answered_lies: list[float]
    answered_passes: list[float]
    lies: list[float]
    passes: list[float]


def _lean_rates(cells: Sequence[_LeanCell], fails: Sequence[float], arms: int) -> _LeanRates:
    """Each arm's recall and false-fail rate, its lies and passes as answered, and the lies and passes its answers stand
    for, with ``fails`` standing for each cell's people's fails; the pooled figures last in the lists that hold them."""
    caught = [0.0] * (arms + 1)
    lies = [0.0] * (arms + 1)
    false = [0.0] * (arms + 1)
    passes = [0.0] * (arms + 1)
    answered_lies = [0.0] * arms
    answered_passes = [0.0] * arms
    for cell, k in zip(cells, fails, strict=True):
        share = k / cell.answered
        for at in (cell.arm, arms):
            lies[at] += cell.claims * share
            passes[at] += cell.claims * (1.0 - share)
            if cell.x:
                caught[at] += cell.claims * share
                false[at] += cell.claims * (1.0 - share)
        answered_lies[cell.arm] += k
        answered_passes[cell.arm] += cell.answered - k
    recall = [caught[at] / lies[at] if lies[at] > 0.0 else None for at in range(arms + 1)]
    false_fail = [false[at] / passes[at] if passes[at] > 0.0 else None for at in range(arms + 1)]
    return _LeanRates(recall, false_fail, answered_lies, answered_passes, lies, passes)


def _lean_statistic(cells: Sequence[_LeanCell], fails: Sequence[float], arms: int) -> float:
    """How far the arms' recall and false-fail rates sit from the pooled ones: each arm's squared distance times its
    answered lies (for recall) or passes (for false fails), over the pooled rate's binomial variance, for every arm
    whose rate expects :data:`LEAN_MIN_EXPECTED` answered claims on each side."""
    read = _lean_rates(cells, fails, arms)
    total = 0.0
    for rates, weights, stand in (
        (read.recall, read.answered_lies, read.lies),
        (read.false_fail, read.answered_passes, read.passes),
    ):
        pooled = rates[arms]
        if pooled is None or not 0.0 < pooled < 1.0:
            continue
        # Only arms that expect enough answered claims each side enter, and they are held to their own pooled rate.
        entering = {
            at: rate
            for at, rate in enumerate(rates[:arms])
            if rate is not None and min(pooled, 1.0 - pooled) * weights[at] >= LEAN_MIN_EXPECTED
        }
        if len(entering) < 2:
            continue
        mean = sum(rate * stand[at] for at, rate in entering.items()) / sum(stand[at] for at in entering)
        if not 0.0 < mean < 1.0:
            continue
        total += sum(weights[at] * (rate - mean) ** 2 for at, rate in entering.items()) / (mean * (1.0 - mean))
    return total


def lean_test(points: Sequence[_Point], arms: int, *, draws: int = LEAN_DRAWS, rng: random.Random) -> float | None:
    """One test across every arm of whether the instrument fails claims at a rate that depends on the arm, among
    claims people answered alike: a parametric bootstrap under a fair instrument, on the draw's own weights.

    A fair instrument fails the claims people fail at one rate (its recall) and the claims people pass at one rate
    (its false fails), whatever arm said them, though arms differ in how often they say false things, in how their
    claims spread over the strata, and in how hard each stratum was drawn. The statistic reads each arm's two rates
    through the draw's weights and sums their squared distances from the pooled rates (:func:`_lean_statistic`).

    Its law under a fair instrument is drawn, not assumed. The pooled rates are imposed on every arm
    (:func:`_null_rates`), keeping each arm's own claims per (stratum, instrument answer) and its own spread of lies
    over them, which gives each answered item its fail rate under the null; the spread is drawn afresh from its
    Jeffreys posterior in each draw, since where an arm's missed lies hide is what its sample says least about. Each
    draw lends every cell a frailty drawn from its arm's (:func:`_frailties`), draws every answered item's fail at its
    rate times its cell's frailty, and reads the statistic again. The p is ``(1 + draws at least as large) / (1 + draws)``.

    :param points: The answered items, each with its arm, stratum, cell, instrument answer, people's answer and weight.
    :param arms: How many arms.
    :param draws: The null draws.
    :param rng: The generator they are drawn by.
    :return: The p, or None where no two arms carry a rate the pooled rates can be imposed on.
    """
    if _answer_varies_within_a_stratum(points):
        return _permuted_lean(points, arms, draws=draws, rng=rng)
    held: dict[tuple[int, int, int], list[float]] = {}
    for point in points:
        sums = held.setdefault((point.arm, point.stratum, point.x), [0.0, 0.0, 0.0])
        sums[0] += 1.0
        sums[1] += point.y
        sums[2] += point.weight
    keys = sorted(held)
    cells = [_LeanCell(arm, x, int(held[key][0]), held[key][1], held[key][2]) for key in keys for arm, _, x in (key,)]
    observed = [cell.fails for cell in cells]
    rates = _lean_rates(cells, observed, arms)
    pooled_recall, pooled_false = rates.recall[arms], rates.false_fail[arms]
    if pooled_recall is None or pooled_false is None or len({cell.arm for cell in cells}) < 2:
        return None
    rate = dict(zip(keys, _null_rates(cells, arms, pooled_recall, pooled_false), strict=True))
    place = {key: at for at, key in enumerate(keys)}
    items = [(place[(p.arm, p.stratum, p.x)], p.cell, rate[(p.arm, p.stratum, p.x)], p.y) for p in points]
    clusters = max((p.cell for p in points), default=-1) + 1
    arm_of_cell = {p.cell: p.arm for p in points}
    frailty = _frailties(items, arm_of_cell, arms)
    statistic = _lean_statistic(cells, observed, arms)
    larger = 0
    for _ in range(draws):
        null = _null_rates(cells, arms, pooled_recall, pooled_false, rng)
        lift = [rng.choice(frailty[arm_of_cell[cell]]) if cell in arm_of_cell else 1.0 for cell in range(clusters)]
        drawn = [0.0] * len(cells)
        for at, cell, _, _ in items:
            drawn[at] += 1.0 if rng.random() < min(1.0, null[at] * lift[cell]) else 0.0
        larger += _lean_statistic(cells, drawn, arms) >= statistic - 1e-12
    return (1 + larger) / (1 + draws)


def _answer_varies_within_a_stratum(points: Sequence[_Point]) -> bool:
    """Whether the instrument's answer on the question varies within a stratum: so for groundedness under a draw by
    truth verdict, and never for the question the draw is stratified by, whose answer the stratum is."""
    seen: dict[int, int] = {}
    return any(seen.setdefault(point.stratum, point.x) != point.x for point in points)


def _permuted_lean(points: Sequence[_Point], arms: int, *, draws: int, rng: random.Random) -> float | None:
    """The lean test where the instrument's answer varies within a stratum: a permutation of the arms' labels among the
    items of each (stratum, people's answer), every item keeping its own weight and answers.

    The draw was stratified on the stratum alone, and people's answer is held, so where the instrument answers alike on
    every arm, an arm's label is exchangeable among the items of one (stratum, people's answer), and the permutation
    gives the statistic's law exactly, whatever the arms' mixes, base rates or cell sizes. A lean is the instrument
    answering differently by arm within one stratum and people's answer. The statistic is :func:`_lean_statistic`'s,
    read off the items. The p is ``(1 + permutations at least as large) / (1 + permutations)``.
    """
    groups: dict[tuple[int, int], list[int]] = {}
    for at, point in enumerate(points):
        groups.setdefault((point.stratum, int(point.y >= 0.5)), []).append(at)
    labels = [point.arm for point in points]

    def statistic(assigned: Sequence[int]) -> float:
        cells: dict[tuple[int, int, int], list[float]] = {}
        for point, arm in zip(points, assigned, strict=True):
            sums = cells.setdefault((arm, point.stratum, point.x), [0.0, 0.0, 0.0])
            sums[0] += 1.0
            sums[1] += point.y
            sums[2] += point.weight
        held = [_LeanCell(arm, x, int(sums[0]), sums[1], sums[2]) for (arm, _, x), sums in sorted(cells.items())]
        return _lean_statistic(held, [cell.fails for cell in held], arms)

    observed = statistic(labels)
    if observed <= 0.0 and len(set(labels)) < 2:
        return None
    larger = 0
    shuffled = list(labels)
    for _ in range(draws):
        for members in groups.values():
            drawn = [labels[at] for at in members]
            rng.shuffle(drawn)
            for at, arm in zip(members, drawn, strict=True):
                shuffled[at] = arm
        larger += statistic(shuffled) >= observed - 1e-12
    return (1 + larger) / (1 + draws)


def _frailties(
    items: Sequence[tuple[int, int, float, float]], arm_of_cell: Mapping[int, int], arms: int
) -> list[list[float]]:
    """Each arm's cells' frailties: how much more often than its null rates a cell's claims were failed, each read as
    (fails + 1) / (expected fails + 1), so a cell of few items reads near 1.

    A null draw lends every cell a frailty drawn from its own arm's, shared by all its claims: two claims of one cell
    then move together as a campaign's do (a cell whose claims are mostly false is drawn that way whole), and a cell of
    one claim keeps its rate on average, since the frailties average about 1.
    """
    fails: dict[int, float] = {}
    expected: dict[int, float] = {}
    for _, cell, null, y in items:
        fails[cell] = fails.get(cell, 0.0) + y
        expected[cell] = expected.get(cell, 0.0) + null
    pools: list[list[float]] = [[] for _ in range(arms)]
    for cell, arm in arm_of_cell.items():
        pools[arm].append((fails[cell] + 1.0) / (expected[cell] + 1.0))
    return [pool or [1.0] for pool in pools]


def _null_rates(
    cells: Sequence[_LeanCell], arms: int, recall: float, false_fail: float, rng: random.Random | None = None
) -> list[float]:
    """Each (arm, stratum, instrument answer)'s people's fail rate under a fair instrument with the pooled rates.

    An arm's claims the instrument failed number ``F`` of its ``N``; a fair instrument at recall ``r`` and false-fail
    rate ``f`` gives it ``L = (F - f N) / (r - f)`` lies (held between none and all), ``r L`` of them failed and the
    rest passed. Each part is spread over the arm's cells of that instrument answer as its answers spread its lies,
    each cell's own share raised by Jeffreys' half, so a lie can sit in a stratum where none was answered."""
    rates = [0.0] * len(cells)
    for arm in range(arms):
        mine = [at for at, cell in enumerate(cells) if cell.arm == arm]
        claims = sum(cells[at].claims for at in mine)
        flagged = sum(cells[at].claims for at in mine if cells[at].x)
        if claims <= 0.0:
            continue
        lies = claims * recall if recall == false_fail else (flagged - false_fail * claims) / (recall - false_fail)
        lies = min(claims, max(0.0, lies))
        for x, share in ((1, recall), (0, 1.0 - recall)):
            part = [at for at in mine if cells[at].x == x]
            spread = {
                at: cells[at].claims
                * (
                    (cells[at].fails + 0.5) / (cells[at].answered + 1.0)
                    if rng is None
                    else rng.betavariate(cells[at].fails + 0.5, cells[at].answered - cells[at].fails + 0.5)
                )
                for at in part
            }
            total = sum(spread.values())
            for at in part:
                lies_here = lies * share * spread[at] / total if total > 0.0 else 0.0
                rates[at] = min(1.0, lies_here / cells[at].claims)
    return rates


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


def _calibrated_weights(answered: Sequence[ClaimItem], read: ClaimRead) -> dict[str, float]:
    """The claims each answered item stands for: one over its chance of being drawn, held so each arm's answered items
    of a stratum stand for exactly that arm's claims of it (``ClaimStratumDraw.by_arm``).

    Held so, an arm's figures never move with how many of its claims a stratum happened to draw, which unanswered
    draws (taken as missing at random within the arm's stratum) would move too. Claims of an arm whose items of a
    stratum went unanswered are stood for by the stratum's other answered items, by their own weights, so the
    population's figures still count them.
    """
    by_cell: dict[tuple[str, str], list[ClaimItem]] = {}
    for item in answered:
        by_cell.setdefault((item.stratum, item.arm), []).append(item)
    weights: dict[str, float] = {}
    for stratum in read.draw.strata:
        held = {arm: by_cell.get((stratum.stratum, arm), []) for arm in stratum.by_arm}
        unread = sum(claims for arm, claims in stratum.by_arm.items() if not held[arm])
        read_claims = sum(claims for arm, claims in stratum.by_arm.items() if held[arm])
        for arm, items in held.items():
            if not items:
                continue
            scale = stratum.by_arm[arm] / sum(1.0 / item.inclusion for item in items)
            for item in items:
                weights[item.item] = scale / item.inclusion * (1.0 + unread / read_claims)
    return weights


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
        answered: list[ClaimItem] = []
        fails_of: list[float] = []
        unanswered = 0
        for item in read.items:
            given = scores[item.item].get(question, [])
            if not given:
                unanswered += 1
                continue
            answered.append(item)
            fails_of.append(sum(1 - score for score in given) / len(given))
        counts = Counter(stratum_index[item.stratum] for item in answered)
        weight_of = _calibrated_weights(answered, read)
        cell_index: dict[str, int] = {}
        points = [
            _Point(
                stratum=stratum_index[item.stratum],
                arm=arm_index[item.arm],
                cell=cell_index.setdefault(item.result_id, len(cell_index)),
                x=instrument_fails(item, question),
                y=fails,
                weight=weight_of[item.item],
            )
            for item, fails in zip(answered, fails_of, strict=True)
        ]
        by_stratum: list[list[_Point]] = [[] for _ in strata]
        for point in points:
            by_stratum[point.stratum].append(point)
        unread = tuple(s for at, s in enumerate(strata) if population[s] > 0 and counts[at] == 0)
        overall, per_arm = _weighted(points, len(arm_keys))
        figures = _figures(overall)
        arm_figures = [_figures(sums) for sums in per_arm]
        rng = random.Random(seed if seed is not None else int(_sha("claim-calibration", read.id, question)[:16], 16))
        groups = _groups(points)
        sizes = [population[s] for s in strata]
        boot: list[_Figures] = []
        boot_arms: list[list[_Figures]] = []
        for _ in range(draws if points else 0):
            boot.append(_figures(_campaign_draw(groups, sizes, rng)))
            boot_arms.append(
                [_figures(sums) for sums in _instrument_draw(points, groups, len(cell_index), len(arm_keys), rng)]
            )
        taken = len(boot)

        def est(value: float | None, pick: int, replicates: Sequence[_Figures]) -> Estimate:
            return _estimate(value, [r[pick] for r in replicates], taken, _RANGES[pick])

        def thin(at: int, value: float | None, fails: bool) -> bool:
            """Whether an arm's rate expects too few answered claims on either side to carry an interval."""
            answered_side = sum(p.y if fails else 1.0 - p.y for p in points if p.arm == at)
            return value is None or min(value, 1.0 - value) * answered_side < LEAN_MIN_EXPECTED

        def arm_est(at: int, pick: int, fails: bool) -> Estimate:
            value = arm_figures[at][pick]
            if thin(at, value, fails):
                return Estimate(value)
            return est(value, pick, [b[at] for b in boot_arms])

        arms = tuple(
            ArmCalibration(
                arm=key,
                items=sum(1 for p in points if p.arm == at),
                cells=len({item.result_id for item in answered if item.arm == key}),
                recall=arm_est(at, 1, fails=True),
                false_fail=arm_est(at, 5, fails=False),
                precision=est(arm_figures[at].precision, 2, [b[at] for b in boot_arms]),
                bias=est(arm_figures[at].bias, 3, [b[at] for b in boot_arms]),
                failed=est(arm_figures[at].failed, 4, [b[at] for b in boot_arms]),
            )
            for at, key in enumerate(arm_keys)
        )
        leans = question != "is_claim" and len(arm_keys) > 1
        lean_p = lean_test(points, len(arm_keys), rng=random.Random(rng.random())) if leans else None
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
                false_fail=est(figures.false_fail, 5, boot),
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
        read_id=read.id,
        instrument=instrument,
        questions=tuple(readings),
        model_answers=models,
        borrowed=borrowed,
        excluded=MappingProxyType(dict(read.draw.excluded)),
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
