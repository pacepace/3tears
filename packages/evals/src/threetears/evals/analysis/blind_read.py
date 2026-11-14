"""The blind read's mechanism, as pure functions over stored shapes: draw, sheet, standing, close.

The shapes and the model are :mod:`threetears.evals.schema.blind_read`. Every door (the quick path's files, a host's
actions) calls these four and re-implements none of them:

- :func:`draw_blind_read` builds the population, the seeded order and the items. Deterministic in its inputs: the
  same answers under the same criteria and seed draw the same order, whichever order the store lists them in, so a
  door that keeps no document recomputes the draw.
- :func:`blind_sheet` is the blinded view: content and criteria, no arm, run or result, no judge score.
- :func:`sample_standing` says whether a rater's answered items are the first ``m`` of the order (a random sample)
  or any other set (``selected``).
- :func:`close_blind_read` turns the answers into ratings and reads them: agreement per criterion with its tier, a
  pass/fail criterion's two-by-two, the preference with its test, and each rater's standing.

**Who answered is never mixed.** A person's answers are read against the judge as agreement with people. A reference
reviewer's (a model's) are read apart, as agreement with that model, and decide a criterion's tier only where no
person has answered on it. A reference reviewer that is the judge's own model is refused as calibration: its
answers repeat the judge and check nothing. Where a person and a reference reviewer answered the same outputs, their
agreement with each other is reported, since the anchor is itself a measurement.

**The preference** is over cases. Each pair item gives the candidate's share: 1, 0, or a half for a tie. With one
answer per item the test is the exact two-sided sign test over the non-tie items (:func:`exact_sign_test_p`), valid
at every n; with several raters the shares are fractions and the bounded test on shares against a half is used,
which treats the raters as fixed.

**The preference is read once, at a length fixed when the read was drawn** (``BlindDraw.planned_items``): over the
first that-many items of the order and no others, and only from a rater who has answered all of them. Before that it
is ``withheld``: no direction, count or p. A sign test re-read after every item and stopped when it separates is
falsely separated about 18% of the time at a 5% level, and a running count tells the rater which arm is ahead.
Answers past the planned length never enter it, so reading on is not a second look.
``tests/test_simulated_blind_read.py`` holds the sign test's level, a rater who stops when it looks good, the
prefix's lack of bias and the slot balance.

This module computes numbers and prints nothing. It imports the stored shapes, the tier rules and the agreement and
statistics code, and no page, form or quick-path module.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Final, NamedTuple

from threetears.evals.analysis.agreement import (
    DimensionAgreement,
    JudgeKey,
    judge_agreement,
    judge_evidence_tiers,
    judge_key,
    judge_key_of,
    judge_self_agreement,
    person_scores_by_result,
)
from threetears.evals.analysis.stats import SIGNIFICANCE_ALPHA, bounded_separation_p, cohen_kappa, wilson_interval
from threetears.evals.kernel.evidence_tiers import JudgeEvidenceTier, agreement_statistic, calibration_criterion
from threetears.evals.schema.blind_read import (
    REFERENCE_RATER_PREFIX,
    BlindAgreement,
    BlindAnchorCheck,
    BlindAnswer,
    BlindArm,
    BlindAsk,
    BlindDraw,
    BlindItem,
    BlindOutput,
    BlindPreference,
    BlindPreferenceTest,
    BlindPreferenceVerdict,
    BlindRaterKind,
    BlindRead,
    BlindReadReport,
    BlindRefusedAnchor,
    BlindSampleStanding,
    BlindSheet,
    BlindSlot,
    BlindStanding,
    BlindVerdictTable,
    SheetAsk,
    SheetCriterion,
    SheetItem,
    SheetLevel,
    SheetOutput,
)
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.models import (
    SCALES,
    CalibrationRating,
    EvalResult,
    JudgeEvidence,
    RubricDim,
    utc_now_iso,
)

#: The seed a read is drawn by unless its caller names one: fixed, so a door that stores no draw recomputes it.
BLIND_READ_SEED: Final = 20261114

#: The fewest non-tie items a preference is tested over; below it the reading is ``too_few``.
#: Five cannot separate at the two-sided 5% level: five of five has p 0.0625.
PREFERENCE_MIN_ITEMS: Final = 6

#: The most items a read's preference is planned over unless its caller fixes another length.
PLANNED_ITEMS_DEFAULT: Final = 30

#: Why a case is left out of the order: some chosen arm has no judged, usable answer on it.
EXCLUDED_NO_ANSWER: Final = "an arm gave no judged answer"

#: Why a case is left out of the order: an arm's answers on it were judged, and none has stored evidence to show.
EXCLUDED_NO_EVIDENCE: Final = "no stored evidence"

#: Why a case is shown once, as a single, with no preference asked: both arms gave byte-identical outputs.
EXCLUDED_IDENTICAL: Final = "identical outputs"

#: What a sheet tells its rater, in one paragraph.
SHEET_INSTRUCTIONS: Final = (
    "Read each item and answer it from what is on the page. Answer the items in the order shown, starting at the "
    "top; you may stop anywhere, and what you have answered from the top is a random sample. Skipping ahead is "
    "allowed, and then the sample is no longer random. Where two outputs are shown they are in a random order: "
    "judge each on its own, then say which is better, or that neither is."
)

#: What a rating written from a blind read says in place of the rated content: the read carries it.
_RATED_ON_THE_READ: Final = "given on a blind read, which carries the output and its case as the rater read them"


class BlindClose(NamedTuple):
    """What closing a blind read gives: the read with its report, and the ratings its answers are.

    ``ratings`` are the read's own answers as ratings, a person's as a person's (``source`` naming the read) and a
    reference reviewer's as an agent's, under the name ``reference:<model>``: never a person's. ``random`` holds the
    ids of those given by a rater whose answered items are a random sample, which is what a reading that needs a
    random sample admits.
    """

    read: BlindRead
    ratings: list[CalibrationRating]
    random: frozenset[str]


def _sha(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _sha12(*parts: object) -> str:
    return _sha(*parts)[:12]


def _usable(result: EvalResult, criteria: Sequence[RubricDim]) -> dict[str, str] | None:
    """Each criterion's fingerprint on ``result``'s judge scores, or None where a criterion has no stamped score."""
    fingerprints: dict[str, str] = {}
    output: set[str] = set()
    for dim in criteria:
        score = result.judge_score(dim.name)
        if score is None or score.output_fingerprint is None or score.criterion_fingerprint is None:
            return None
        if score.scale != dim.scale:
            return None
        fingerprints[dim.name] = score.criterion_fingerprint
        output.add(score.output_fingerprint)
    return fingerprints if len(output) == 1 else None


def draw_blind_read(
    results: Iterable[EvalResult],
    evidence: Mapping[str, JudgeEvidence],
    criteria: Sequence[RubricDim],
    arms: Sequence[BlindArm],
    *,
    scope_id: str,
    created_by: str,
    name: str = "blind read",
    seed: int = BLIND_READ_SEED,
    ask: Sequence[BlindAsk] = ("verdicts", "preference"),
    campaign_id: str | None = None,
    created_at: str | None = None,
    planned_items: int | None = None,
) -> BlindRead:
    """Draw a blind read: the population, one seeded order over it, and the items.

    The population is every case on which each arm has a result judged on every criterion with stored evidence. One
    result per arm per case is drawn among the case's repeats. The order is one seeded permutation of the cases.
    With two arms, each item's slots are a seeded coin in blocks of two pair items, so any prefix of ``2m`` pairs
    holds ``m`` with the control in slot A. Two byte-identical outputs cannot be told apart, so such a case is shown
    once, as a single with no preference asked, and counted.

    Nothing drawn depends on the order results are listed in or on a result's or run's id: the pick among repeats is
    by output, and the order by case.

    :param results: The arms' results.
    :ptype results: Iterable[EvalResult]
    :param evidence: What the judge read of each result, by result id; a result with none here is not shown.
    :ptype evidence: Mapping[str, JudgeEvidence]
    :param criteria: The criteria asked, as worded.
    :ptype criteria: Sequence[RubricDim]
    :param arms: One arm (singles), or a control and a candidate (pairs), each naming its runs.
    :ptype arms: Sequence[BlindArm]
    :param scope_id: The scope the read belongs to.
    :ptype scope_id: str
    :param created_by: Who drew it.
    :ptype created_by: str
    :param name: What the read is called.
    :ptype name: str
    :param seed: The seed.
    :ptype seed: int
    :param ask: What each item asks; a preference is asked only of pairs.
    :ptype ask: Sequence[BlindAsk]
    :param campaign_id: The campaign drawn over, where there is one.
    :ptype campaign_id: str | None
    :param created_at: When it was drawn; None for now.
    :ptype created_at: str | None
    :param planned_items: The length the preference is read at; None for the whole order up to
        :data:`PLANNED_ITEMS_DEFAULT`. Capped at the order's length.
    :ptype planned_items: int | None
    :return: The read, with no answers.
    :rtype: BlindRead
    :raises ValueError: No criterion; no arm or more than two; two arms that are not one control and one candidate;
        or a run named by two arms.
    """
    if not criteria:
        raise ValueError("a blind read asks at least one criterion")
    if len(arms) not in (1, 2):
        raise ValueError(f"a blind read is over one arm, or a control and one candidate; {len(arms)} were given")
    if len(arms) == 2 and sorted(arm.role for arm in arms) != ["candidate", "control"]:
        raise ValueError("a blind read over two arms is a control against one candidate")
    arm_of_run: dict[str, str] = {}
    for arm in arms:
        for run_id in arm.run_ids:
            if arm_of_run.setdefault(run_id, arm.key) != arm.key:
                raise ValueError(f"run {run_id!r} is named by two arms of the blind read")
    # Each arm's usable results per case, and the cases an arm answered with no usable result.
    usable: dict[str, dict[str, list[tuple[EvalResult, dict[str, str]]]]] = {arm.key: {} for arm in arms}
    unshown: dict[str, set[str]] = {arm.key: set() for arm in arms}
    cases: set[str] = set()
    for result in results:
        arm_key = arm_of_run.get(result.eval_run_id)
        if arm_key is None:
            continue
        cases.add(result.test_case_id)
        fingerprints = _usable(result, criteria)
        if fingerprints is None:
            continue
        if result.id not in evidence:
            unshown[arm_key].add(result.test_case_id)
            continue
        usable[arm_key].setdefault(result.test_case_id, []).append((result, fingerprints))
    excluded: dict[str, int] = {}
    population: list[str] = []
    for case in sorted(cases):
        if all(case in usable[arm.key] for arm in arms):
            population.append(case)
        elif any(case in unshown[arm.key] and case not in usable[arm.key] for arm in arms):
            excluded[EXCLUDED_NO_EVIDENCE] = excluded.get(EXCLUDED_NO_EVIDENCE, 0) + 1
        else:
            excluded[EXCLUDED_NO_ANSWER] = excluded.get(EXCLUDED_NO_ANSWER, 0) + 1
    digest = canonical_digest(
        {
            "blind_read_population": sorted(
                {
                    (case, arm.key, _output_fingerprint(result, criteria), criterion)
                    for arm in arms
                    for case in population
                    for result, fingerprints in usable[arm.key][case]
                    for criterion in fingerprints.values()
                }
            )
        }
    )
    # A seeded digest per case, sorted: a uniform order that is a pure function of the seed and the population.
    order = sorted(population, key=lambda case: _sha("blind-order", seed, digest, case))
    control = next((arm for arm in arms if arm.role == "control"), arms[0])
    others = [arm for arm in arms if arm is not control]
    items: list[BlindItem] = []
    pairs_drawn = 0
    control_first = True
    for position, case in enumerate(order):
        picked = {arm.key: _pick(usable[arm.key][case], criteria, seed, case, arm.key) for arm in arms}
        shown = [control]
        if others:
            (candidate,) = others
            same = _output_fingerprint(picked[control.key][0], criteria) == _output_fingerprint(
                picked[candidate.key][0], criteria
            )
            if same:
                excluded[EXCLUDED_IDENTICAL] = excluded.get(EXCLUDED_IDENTICAL, 0) + 1
            else:
                # A coin per block of two pair items: the first takes it, the second its opposite.
                if pairs_drawn % 2 == 0:
                    control_first = int(_sha("blind-slot", seed, digest, pairs_drawn // 2), 16) % 2 == 0
                else:
                    control_first = not control_first
                pairs_drawn += 1
                shown = [control, candidate] if control_first else [candidate, control]
        slots: tuple[BlindSlot, ...] = ("A", "B")
        items.append(
            BlindItem(
                item=_sha12("blind-item", seed, digest, position),
                case_id=case,
                outputs=[
                    BlindOutput(
                        slot=slot,
                        arm=arm.key,
                        run_id=picked[arm.key][0].eval_run_id,
                        result_id=picked[arm.key][0].id,
                        output_fingerprint=_output_fingerprint(picked[arm.key][0], criteria),
                        criterion_fingerprints=picked[arm.key][1],
                        evidence=evidence[picked[arm.key][0].id],
                    )
                    for slot, arm in zip(slots, shown, strict=False)
                ],
            )
        )
    asked = [entry for entry in dict.fromkeys(ask) if entry != "preference" or len(arms) == 2]
    return BlindRead(
        id=f"blind-{_sha12('blind-read', seed, digest)}",
        scope_id=scope_id,
        name=name,
        created_at=created_at if created_at is not None else utc_now_iso(),
        created_by=created_by,
        campaign_id=campaign_id,
        arms=list(arms),
        criteria=list(criteria),
        ask=asked or ["verdicts"],
        draw=BlindDraw(
            seed=seed,
            population=len(population),
            population_digest=digest,
            planned_items=min(len(items), PLANNED_ITEMS_DEFAULT if planned_items is None else max(planned_items, 0)),
            excluded=excluded,
        ),
        items=items,
    )


def _output_fingerprint(result: EvalResult, criteria: Sequence[RubricDim]) -> str:
    score = result.judge_score(criteria[0].name)
    assert score is not None and score.output_fingerprint is not None  # only usable results reach here
    return score.output_fingerprint


def _pick(
    candidates: Sequence[tuple[EvalResult, dict[str, str]]],
    criteria: Sequence[RubricDim],
    seed: int,
    case: str,
    arm: str,
) -> tuple[EvalResult, dict[str, str]]:
    """One of a case's repeats, by a seeded digest of its output, so the pick is the same whichever repeat holds
    which output and whatever the results' ids are."""
    return min(
        candidates,
        key=lambda entry: (
            hashlib.sha256(f"{seed}|{case}|{arm}|{_output_fingerprint(entry[0], criteria)}".encode()).hexdigest(),
            entry[0].id,
        ),
    )


def blind_sheet(read: BlindRead, *, rater: str | None = None) -> BlindSheet:
    """The blinded view of ``read``: what a rater is shown, and nothing else.

    :param read: The read.
    :ptype read: BlindRead
    :param rater: Whose progress to report (``answered`` and ``next``); None reports none.
    :ptype rater: str | None
    :return: The sheet: no arm, run, result or case id, and no judge score.
    :rtype: BlindSheet
    """
    answered = {answer.item for answer in read.answers if rater is not None and answer.rater == rater}
    verdicts = [dim.name for dim in read.criteria] if "verdicts" in read.ask else []
    return BlindSheet(
        read_id=read.id,
        name=read.name,
        instructions=SHEET_INSTRUCTIONS,
        criteria=[
            SheetCriterion(
                dim=dim.name,
                question=dim.description,
                scale=dim.scale,
                levels=[
                    SheetLevel(value=level, meaning=dim.scoring_guide.get(level, ""))
                    for level in SCALES[dim.scale].levels
                ],
            )
            for dim in read.criteria
        ],
        items=[
            SheetItem(
                item=item.item,
                case_material=item.outputs[0].evidence.case_material,
                outputs=[SheetOutput(slot=output.slot, text=output.evidence.artifact) for output in item.outputs],
                ask=SheetAsk(verdicts=verdicts, preference="preference" in read.ask and len(item.outputs) == 2),
            )
            for item in read.items
        ],
        answered=[item.item for item in read.items if item.item in answered],
        next=next((item.item for item in read.items if item.item not in answered), None),
    )


def sample_standing(read: BlindRead, rater: str) -> BlindStanding:
    """Whether ``rater``'s answered items are a random sample: the first ``m`` of the order, or not.

    A skip is an answer, so a skipped item keeps a prefix a prefix.

    :param read: The read.
    :ptype read: BlindRead
    :param rater: The rater, as their answers name them.
    :ptype rater: str
    :return: ``random`` with the count, ``selected``, or ``none``.
    :rtype: BlindStanding
    """
    own = [answer for answer in read.answers if answer.rater == rater]
    answered = {answer.item for answer in own}
    order = [item.item for item in read.items]
    kind: BlindRaterKind = own[0].rater_kind if own else "person"
    standing: BlindSampleStanding = (
        "none" if not answered else "random" if answered == set(order[: len(answered)]) else "selected"
    )
    return BlindStanding(
        rater=rater,
        rater_kind=kind,
        standing=standing,
        answered=len(answered),
        skipped=sum(1 for answer in own if answer.skipped),
        items=len(order),
    )


def exact_sign_test_p(successes: int, trials: int) -> float | None:
    """The exact two-sided sign test: the chance, under no preference, of a split at least as lopsided.

    Binomial(``trials``, one half) over the non-tie items: twice the smaller tail, capped at 1. Valid at every n and
    for any distribution of cases, since it assumes only that the items are independent. Being exact on a discrete
    count it rejects at most as often as its level, and usually less.

    :param successes: The items one side took.
    :ptype successes: int
    :param trials: The non-tie items.
    :ptype trials: int
    :return: The p; None with no trials.
    :rtype: float | None
    :raises ValueError: ``successes`` outside 0 to ``trials``.
    """
    if not 0 <= successes <= trials:
        raise ValueError(f"a sign test needs 0 <= successes <= trials; got {successes} of {trials}")
    if trials == 0:
        return None
    tail = sum(math.comb(trials, count) for count in range(min(successes, trials - successes) + 1))
    return min(1.0, 2 * tail / 2**trials)


def reference_rater(model: str) -> str:
    """The rater name a reference reviewer's answers and ratings carry: ``reference:<model>``."""
    return f"{REFERENCE_RATER_PREFIX}{model}"


def _ratings_of(read: BlindRead) -> list[CalibrationRating]:
    """The read's answers as ratings, each bound to the output and criterion it was given on.

    :raises ValueError: An answer to no item of the read, a verdict on a slot the item does not show or on no
        criterion of the read, or a score its criterion's scale cannot hold.
    """
    items = {item.item: item for item in read.items}
    scales = {dim.name: dim.scale for dim in read.criteria}
    ratings: dict[str, CalibrationRating] = {}
    for answer in read.answers:
        item = items.get(answer.item)
        if item is None:
            raise ValueError(f"an answer names item {answer.item!r}, which is no item of blind read {read.id!r}")
        outputs = {output.slot: output for output in item.outputs}
        person = answer.rater_kind == "person"
        for verdict in answer.verdicts:
            output = outputs.get(verdict.slot)
            scale = scales.get(verdict.dim)
            if output is None or scale is None:
                raise ValueError(
                    f"the answer to item {answer.item!r} gives a verdict on slot {verdict.slot} and {verdict.dim!r}, "
                    "which the item does not ask"
                )
            low, high = SCALES[scale].scores
            if not low <= verdict.score <= high:
                raise ValueError(f"{verdict.score!r} is not on the {scale} scale of {verdict.dim!r}")
            rating = CalibrationRating(
                scope_id=read.scope_id,
                run_id=output.run_id,
                result_id=output.result_id,
                rubric_dim=verdict.dim,
                rater=answer.rater if person else reference_rater(answer.rater),
                # The durable rating knows two kinds; a reference reviewer is a model, so it is never the person's.
                rater_kind="person" if person else "agent",
                source=f"blind_read:{read.id}" if person else None,
                collected_by=(answer.collected_by or answer.rater) if person else None,
                scale=scale,
                score=verdict.score,
                reason=verdict.reason or f"given on blind read {read.name}",
                rated_at=answer.answered_at,
                output_fingerprint=output.output_fingerprint,
                criterion_fingerprint=output.criterion_fingerprints[verdict.dim],
                rated_withheld=_RATED_ON_THE_READ,
            )
            ratings[rating.id] = rating
    return list(ratings.values())


def _same_family(reviewer: str, judge_model: str | None) -> bool:
    """Whether two model ids name one provider: the part before the first ``/``, where both have one."""
    if judge_model is None or "/" not in reviewer or "/" not in judge_model:
        return False
    return reviewer.split("/", 1)[0] == judge_model.split("/", 1)[0]


def _table(
    scores: Mapping[tuple[str, str], Sequence[int]], by_id: Mapping[str, EvalResult], dim: str, judge: JudgeKey
) -> BlindVerdictTable:
    """A pass/fail criterion's two-by-two over the pairs one judge's agreement read."""
    cells = {(1, 1): 0, (1, 0): 0, (0, 1): 0, (0, 0): 0}
    for (result_id, rated_dim), given in scores.items():
        result = by_id.get(result_id)
        if rated_dim != dim or result is None or judge_key(result, dim) != judge:
            continue
        score = result.judge_score(dim)
        assert score is not None  # its judge key was read off it
        for rater_score in given:
            cells[(rater_score, score.score)] += 1
    failed = cells[(0, 0)] + cells[(0, 1)]
    return BlindVerdictTable(
        both_pass=cells[(1, 1)],
        rater_pass_judge_fail=cells[(1, 0)],
        rater_fail_judge_pass=cells[(0, 1)],
        both_fail=cells[(0, 0)],
        detection=cells[(0, 0)] / failed if failed else None,
        detection_interval=wilson_interval(cells[(0, 0)], failed) if failed else None,
    )


def _agreements(
    ratings: Sequence[CalibrationRating],
    results: Sequence[EvalResult],
    *,
    anchor: BlindRaterKind,
    names: list[str] | None,
    before: Mapping[JudgeKey, str],
    own: frozenset[str],
    random_ids: frozenset[str],
) -> list[tuple[DimensionAgreement, BlindAgreement]]:
    """One anchor's agreement with each judge on each criterion, with the tier it bears on that anchor."""
    if not ratings:
        return []
    rated_by = "person" if anchor == "person" else "agent"
    agreement = judge_agreement(ratings, results, rated_by=rated_by)
    judged = {key for result in results for score in result.judge_scores() if (key := judge_key(result, score.dim))}
    tiers = {
        judge_key_of(tier): tier for tier in judge_evidence_tiers(agreement, judge_self_agreement(results), judged)
    }
    scores = person_scores_by_result(ratings, results, rated_by=rated_by)
    by_id = {result.id: result for result in results}
    read: list[tuple[DimensionAgreement, BlindAgreement]] = []
    for entry in agreement.dimensions:
        key = judge_key_of(entry)
        statistic = agreement_statistic(entry.scale, entry.kappa, entry.weighted_kappa)
        criterion = calibration_criterion(entry.n, entry.results, statistic, entry.agreement_interval)
        on_dim = [rating for rating in ratings if rating.rubric_dim == entry.rubric_dim]
        read.append(
            (
                entry,
                BlindAgreement(
                    dim=entry.rubric_dim,
                    scale=entry.scale,
                    judge_model=entry.judge_model,
                    anchor=anchor,
                    anchor_names=names if names is not None else entry.raters,
                    decides=anchor == "person",
                    shares_family=anchor == "reference"
                    and names is not None
                    and _same_family(names[0], entry.judge_model),
                    pairs=entry.n,
                    results=entry.results,
                    cases=len(
                        {
                            by_id[result_id].test_case_id
                            for (result_id, rated_dim) in scores
                            if rated_dim == entry.rubric_dim
                            and result_id in by_id
                            and judge_key(by_id[result_id], rated_dim) == key
                        }
                    ),
                    blind_pairs=sum(1 for rating in on_dim if rating.id in own),
                    random_pairs=sum(1 for rating in on_dim if rating.id in random_ids),
                    exact_agreement=entry.exact_agreement,
                    kappa=statistic,
                    lower=None if entry.agreement_interval is None else entry.agreement_interval[0],
                    upper=None if entry.agreement_interval is None else entry.agreement_interval[1],
                    calibration=criterion.state,
                    results_needed=criterion.results_needed,
                    tier_before=before.get(key, "undetermined"),
                    tier=tiers[key].tier,
                    table=_table(scores, by_id, entry.rubric_dim, key) if entry.scale == "pass_fail" else None,
                ),
            )
        )
    return read


def _preference(
    read: BlindRead, answers: Sequence[BlindAnswer], standings: Mapping[str, BlindStanding]
) -> BlindPreference:
    """One kind of rater's preference over the planned items, per case, with its test; withheld until a rater has
    answered every planned item."""
    control = next(arm.key for arm in read.arms if arm.role == "control")
    candidate = next(arm.key for arm in read.arms if arm.role == "candidate")
    planned = read.items[: read.draw.planned_items]
    planned_ids = {item.item for item in planned}
    items = {item.item: item for item in planned}
    by_rater: dict[str, list[BlindAnswer]] = {}
    for answer in answers:
        if answer.item in planned_ids:
            by_rater.setdefault(answer.rater, []).append(answer)
    done = sorted(rater for rater, own in by_rater.items() if {answer.item for answer in own} == planned_ids)
    kind = answers[0].rater_kind
    if not done or not planned:
        return BlindPreference(
            anchor=kind,
            anchor_names=sorted({answer.rater for answer in answers}),
            random=False,
            control=control,
            candidate=candidate,
            planned=len(planned),
            answered=max((len({answer.item for answer in own}) for own in by_rater.values()), default=0),
            items=0,
            prefer_candidate=0,
            prefer_control=0,
            ties=0,
            test="exact_sign",
            p=None,
            verdict="withheld",
            slot_a_share=None,
            slot_choices=0,
        )
    shares: dict[str, list[float]] = {}
    slot_a = slot_choices = skipped = 0
    for rater in done:
        for answer in by_rater[rater]:
            item = items[answer.item]
            skipped += answer.skipped
            if answer.prefer is None or len(item.outputs) != 2:
                continue
            if answer.prefer == "tie":
                share = 0.5
            else:
                preferred = next(output for output in item.outputs if output.slot == answer.prefer)
                share = 1.0 if preferred.arm == candidate else 0.0
                slot_choices += 1
                slot_a += answer.prefer == "A"
            shares.setdefault(answer.item, []).append(share)
    means = [sum(given) / len(given) for given in shares.values()]
    for_candidate = sum(1 for mean in means if mean > 0.5)
    for_control = sum(1 for mean in means if mean < 0.5)
    decided = for_candidate + for_control
    one_each = all(len(given) == 1 for given in shares.values())
    test: BlindPreferenceTest = "exact_sign" if one_each else "bounded_share"
    verdict: BlindPreferenceVerdict
    p: float | None = None
    if decided < PREFERENCE_MIN_ITEMS:
        verdict = "too_few"
    else:
        p = (
            exact_sign_test_p(for_candidate, decided)
            if one_each
            else bounded_separation_p([0.5] * len(means), means, paired=True, value_range=(0.0, 1.0))
        )
        if p is None or p >= SIGNIFICANCE_ALPHA or for_candidate == for_control:
            verdict = "not_separated"
        else:
            verdict = "improved" if for_candidate > for_control else "regressed"
    return BlindPreference(
        anchor=kind,
        anchor_names=done,
        random=all(standings[rater].standing == "random" for rater in done),
        control=control,
        candidate=candidate,
        planned=len(planned),
        answered=len(planned),
        skipped=skipped,
        items=len(means),
        prefer_candidate=for_candidate,
        prefer_control=for_control,
        ties=len(means) - decided,
        test=test,
        p=p,
        verdict=verdict,
        slot_a_share=slot_a / slot_choices if slot_choices else None,
        slot_a=slot_a,
        slot_choices=slot_choices,
    )


def _anchor_checks(ratings: Sequence[CalibrationRating], criteria: Sequence[RubricDim]) -> list[BlindAnchorCheck]:
    """Each reference reviewer's agreement with people on the outputs both answered, per criterion."""
    people: dict[tuple[str, str, str], list[int]] = {}
    reviewers: dict[str, dict[tuple[str, str, str], int]] = {}
    for rating in ratings:
        if rating.label_key is None:
            continue
        key = (rating.rubric_dim, *rating.label_key)
        if rating.rater_kind == "person":
            people.setdefault(key, []).append(rating.score)
        elif rating.rater.startswith(REFERENCE_RATER_PREFIX):
            reviewers.setdefault(rating.rater.removeprefix(REFERENCE_RATER_PREFIX), {})[key] = rating.score
    checks: list[BlindAnchorCheck] = []
    for reviewer in sorted(reviewers):
        for dim in criteria:
            pairs = [
                (score, given)
                for key, given in reviewers[reviewer].items()
                if key[0] == dim.name
                for score in people.get(key, ())
            ]
            if not pairs:
                continue
            low, high = SCALES[dim.scale].scores
            checks.append(
                BlindAnchorCheck(
                    dim=dim.name,
                    reviewer=reviewer,
                    outputs=len(pairs),
                    exact_agreement=sum(1 for person, model in pairs if person == model) / len(pairs),
                    kappa=cohen_kappa(
                        pairs,
                        list(range(low, high + 1)),
                        weights="quadratic" if dim.scale == "ordinal" else "none",
                    ),
                )
            )
    return checks


def close_blind_read(
    read: BlindRead,
    results: Iterable[EvalResult],
    tiers_before: Sequence[JudgeEvidenceTier] | None = None,
    *,
    other_ratings: Sequence[CalibrationRating] = (),
    unread: Sequence[str] = (),
    closed_by: str | None = None,
    closed_at: str | None = None,
) -> BlindClose:
    """Close ``read``: its answers as ratings, and what they say of the judge and of the arms.

    A person's answers and a reference reviewer's are read apart. On each criterion the person reading decides the
    tier where there is one; a reference reading decides only in its absence, and is refused outright where the
    reviewer is the judge's own model.

    :param read: The read, with its answers.
    :ptype read: BlindRead
    :param results: The results the answers may rate: every result the read was drawn over.
    :ptype results: Iterable[EvalResult]
    :param tiers_before: The judge's tiers before these answers; None reads them from ``results`` with no ratings.
    :ptype tiers_before: Sequence[JudgeEvidenceTier] | None
    :param other_ratings: People's ratings from outside the read (an older labels sheet), which count toward
        agreement as ``selected``: never as a random sample.
    :ptype other_ratings: Sequence[CalibrationRating]
    :param unread: Answers offered that could not be read against the read, each with why; carried on the report.
    :ptype unread: Sequence[str]
    :param closed_by: Who closed it; None leaves the read open, its report computed.
    :ptype closed_by: str | None
    :param closed_at: When; None for now, where ``closed_by`` is given.
    :ptype closed_at: str | None
    :return: The read carrying its report, the ratings, and which of them came from a random sample.
    :rtype: BlindClose
    :raises ValueError: An answer the read does not ask (see :func:`_ratings_of`).
    """
    results = list(results)
    ratings = _ratings_of(read)
    raters = list(dict.fromkeys(answer.rater for answer in read.answers))
    standings = {rater: sample_standing(read, rater) for rater in raters}
    kind_of = {answer.rater: answer.rater_kind for answer in read.answers}
    random_raters = {
        (rater if kind_of[rater] == "person" else reference_rater(rater))
        for rater, standing in standings.items()
        if standing.standing == "random"
    }
    own = frozenset(rating.id for rating in ratings)
    random_ids = frozenset(rating.id for rating in ratings if rating.rater in random_raters)
    if tiers_before is None:
        judged = {key for result in results for score in result.judge_scores() if (key := judge_key(result, score.dim))}
        tiers_before = judge_evidence_tiers(judge_agreement([], results), judge_self_agreement(results), judged)
    before = {judge_key_of(tier): tier.tier for tier in tiers_before}
    people = [rating for rating in (*ratings, *other_ratings) if rating.rater_kind == "person"]
    agreement = [
        reading
        for _, reading in _agreements(
            people, results, anchor="person", names=None, before=before, own=own, random_ids=random_ids
        )
    ]
    refused: list[BlindRefusedAnchor] = []
    served = {
        answer.rater: {answer.rater, *([answer.reviewer_served_model] if answer.reviewer_served_model else [])}
        for answer in read.answers
        if answer.rater_kind == "reference"
    }
    for reviewer in sorted(served):
        own_ratings = [rating for rating in ratings if rating.rater == reference_rater(reviewer)]
        for entry, reading in _agreements(
            own_ratings,
            results,
            anchor="reference",
            names=[reviewer],
            before=before,
            own=own,
            random_ids=random_ids,
        ):
            if entry.judge_model is not None and entry.judge_model in served[reviewer]:
                refused.append(
                    BlindRefusedAnchor(
                        reviewer=reviewer,
                        dim=entry.rubric_dim,
                        reason=(
                            "the reviewer is the judge's own model, so its answers repeat the judge and are not read "
                            "as calibration"
                        ),
                    )
                )
                continue
            agreement.append(reading)
    agreement = _with_the_deciding_anchor(agreement)
    preferences: list[BlindPreference] = []
    if "preference" in read.ask:
        for kind in ("person", "reference"):
            for group in _preference_groups(read, kind):
                if group:
                    preferences.append(_preference(read, group, standings))
    report = BlindReadReport(
        standings=list(standings.values()),
        agreement=agreement,
        anchor_checks=_anchor_checks([*ratings, *other_ratings], read.criteria),
        refused_anchors=refused,
        preferences=preferences,
        other_ratings=len(other_ratings),
        unread=list(unread),
    )
    closed = read.model_copy(
        update={
            "report": report,
            "closed_by": closed_by,
            "closed_at": (closed_at or utc_now_iso()) if closed_by is not None else None,
        }
    )
    return BlindClose(read=closed, ratings=ratings, random=random_ids)


def _with_the_deciding_anchor(agreement: Sequence[BlindAgreement]) -> list[BlindAgreement]:
    """Mark which reading decides each criterion's tier for each judge.

    A person reading decides once it is itself decided (met or not met), or where no reference reading stands beside
    it. Until then the first reference reading decides, so a few person labels never remove a standing a reference
    reviewer earned; the person reading is shown beside it.
    """
    marked: list[BlindAgreement] = []
    for reading in agreement:
        beside = [other for other in agreement if (other.dim, other.judge_model) == (reading.dim, reading.judge_model)]
        person = next((other for other in beside if other.anchor == "person"), None)
        references = [other for other in beside if other.anchor == "reference"]
        person_decides = person is not None and (person.calibration in ("met", "not_met") or not references)
        if reading.anchor == "person":
            decides = person_decides
        else:
            decides = not person_decides and reading is references[0]
        marked.append(reading.model_copy(update={"decides": decides}))
    return marked


def _preference_groups(read: BlindRead, kind: str) -> list[list[BlindAnswer]]:
    """The answers a preference is read over: every person's together, and each reference reviewer's alone."""
    answers = [answer for answer in read.answers if answer.rater_kind == kind]
    if kind == "person":
        return [answers] if answers else []
    return [[answer for answer in answers if answer.rater == rater] for rater in sorted({a.rater for a in answers})]


#: What another module calls today. :func:`sample_standing` is the core's fourth function and joins these when the
#: host door calls it; until then it is read through a closed read's report.
__all__ = [
    "PREFERENCE_MIN_ITEMS",
    "BlindClose",
    "blind_sheet",
    "close_blind_read",
    "draw_blind_read",
    "exact_sign_test_p",
]
