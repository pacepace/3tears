"""The blind read's mechanism, as pure functions over stored shapes: draw, sheet, standing, close.

The shapes and the model are :mod:`threetears.evals.schema.blind_read`. Every door (the quick path's files, a host's
actions) calls these four and re-implements none of them:

- :func:`draw_blind_read` builds the population, the seeded order and the items. Deterministic in its inputs: the
  same answers under the same criteria and seed draw the same order, whichever order the store lists them in, so a
  door that keeps no document recomputes the draw.
- :func:`blind_sheet` is the blinded view: content and criteria, no arm, run or result, no judge score.
- :func:`read_sample` says whether a rater's answered items are the first ``m`` of the order (a random sample)
  or any other set (``selected``).
- :func:`close_blind_read` turns the answers into ratings and reads them: the judge's scores against each reader
  configuration per criterion, stored as counts (on pass/fail the confusion counts, on 1-5 the full cross-table), the
  preference with its test, and each rater's standing. Nothing derived from those counts is stored (#1208): a host's
  door reads its catch rates off them, and the kappa and tier the quick path's print still states are computed at the
  close beside them (:class:`BlindAgreementReading`).

**Who answered is never mixed.** A person's answers are read against the judge as agreement with people. A reference
reviewer's (a model's) are read apart, as agreement with that model, once per model that served its answers (two
served models under one reviewer id are two readings, never pooled), and decide a criterion's tier only where no
person has answered on it. A reference reviewer that is the judge's own model is refused as calibration: its
answers repeat the judge and check nothing. Where a person and a reference reviewer answered the same outputs, their
agreement with each other is reported, since the anchor is itself a measurement.

**The preference** is over cases. Each pair item gives the candidate's share: 1, 0, or a half for a tie. With one
answer per item the test is the exact two-sided sign test over the non-tie items (:func:`exact_sign_test_p`), valid
at every n; with several raters the shares are fractions and the bounded test on shares against a half is used,
which treats the raters as fixed. A preference stores those shares and never its p or verdict, which
:func:`read_preference` computes off them wherever a read is closed or loaded (#1236): one computation for every reader.

**The preference is read once, at a length fixed when the read was drawn** (``BlindDraw.planned_items``): over the
first that-many items of the order and no others, and only from a rater who has answered all of them. Before that it
is ``withheld``: no direction, count or p. A sign test re-read after every item and stopped when it separates is
falsely separated about 18% of the time at a 5% level, and a running count tells the rater which arm is ahead.
Answers past the planned length never enter it, so reading on is not a second look. **A read closed early** (before
any rater answered every planned item) is the one exception, and it is read once, at the close: by the anytime test,
the bounded test on shares in the drawn order with its stakes fixed for the planned pair items, whose p holds at any
stop. ``tests/test_simulated_blind_read.py`` holds the sign test's level, a rater who stops when it looks good (and one
who closes early when it looks good), the prefix's lack of bias and the slot balance.

This module computes numbers and prints nothing. It imports the stored shapes, the tier rules and the agreement and
statistics code, and no page, form or quick-path module.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final, Literal, NamedTuple

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
from threetears.evals.analysis.stats import SIGNIFICANCE_ALPHA, bounded_separation_p, cohen_kappa
from threetears.evals.kernel.evidence_tiers import JudgeEvidenceTier, agreement_statistic, calibration_criterion
from threetears.evals.schema.blind_read import (
    BlindAgreement,
    BlindReviewerCheck,
    BlindAnswer,
    BlindArm,
    BlindAsk,
    BlindDraw,
    BlindItem,
    BlindOutput,
    BlindPreference,
    BlindPreferenceTest,
    BlindRead,
    BlindReadReport,
    BlindSheet,
    BlindSlot,
    BlindSample,
    BlindVerdictTable,
    SheetAsk,
    SheetCriterion,
    SheetItem,
    SheetLevel,
    SheetOutput,
)
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.reading_states import Maker, SampleKind, StateOf
from threetears.evals.schema.models import (
    SCALES,
    CalibrationRating,
    EvalResult,
    JudgeEvidence,
    RaterKind,
    RubricDim,
    RubricScale,
    fingerprint_criterion,
    fingerprint_judged_output,
    utc_now_iso,
)

#: The seed a read is drawn by unless its caller names one: fixed, so a door that stores no draw recomputes it.
BLIND_READ_SEED: Final = 20261114

#: The fewest non-tie items a preference is tested over; below it the reading is not available, as too few.
#: Five cannot separate at the two-sided 5% level: five of five has p 0.0625.
PREFERENCE_MIN_ITEMS: Final = 6

#: The most items a read's preference is planned over unless its caller fixes another length.
PLANNED_ITEMS_DEFAULT: Final = 30

#: Why a case is left out of the order: some chosen arm has no judged, usable answer on it.
EXCLUDED_NO_ANSWER: Final = "an arm gave no judged answer"

#: Why a case is left out of the order: an arm's answers on it were judged, and none has stored evidence to show.
EXCLUDED_NO_EVIDENCE: Final = "no stored evidence"

#: Why a case is left out of the order: an arm's answers on it were judged on other wording of a criterion than the
#: read asks (the template was edited since), so a rating given on the read would be bound to a question its rater
#: was not shown.
EXCLUDED_REWORDED: Final = "judged on other wording of a criterion"

#: Why a case is left out of the order: an arm's stored evidence is not the output its judge score was given on, so a
#: rater would be shown other text than the judge read.
EXCLUDED_OTHER_TEXT: Final = "stored evidence differs from what was judged"

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
    model's as an agent's: never a person's. ``random`` holds the ids of those given by a rater whose answered items
    are a random sample, which is what a reading that needs a random sample admits. ``reviewed`` says which ratings
    are a model reviewer's, by rating id, with the reviewer's model id: read off the answer each came from. ``agreement``
    is each stored agreement (``read.report.agreement``, counts only) beside the kappa and tier the quick path's print
    still reads, computed here and never stored. No code
    reads that out of a rater's name. It stands in for the typed ``made_by`` a rating is to carry: when a rating
    records who made it (a person, or a model with its id, served model and the read it answered), this map and the
    ``agent`` kind it qualifies are what that field replaces.
    """

    read: BlindRead
    ratings: list[CalibrationRating]
    random: frozenset[str]
    reviewed: Mapping[str, str]
    agreement: tuple[BlindAgreementReading, ...] = ()


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


def _judged_as_asked(fingerprints: Mapping[str, str], criteria: Sequence[RubricDim]) -> bool:
    """Whether the judge read every criterion in the words the read asks it in.

    A rating is bound to the criterion the judge read (its fingerprint, which the rating carries), and a rater is shown
    the criterion as the read words it. Templates are edited in place, so the two can differ: a result judged before
    an edit would have a person answer the new question and the answer be stored, and read, as a label of the old one.
    Such a result is left out of the draw, and counted, so what a rater is shown is what each rating is bound to.
    """
    return all(fingerprints[dim.name] == fingerprint_criterion(dim) for dim in criteria)


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

    The population is every case on which each arm has a result judged on every criterion, in the words ``criteria``
    gives them, with stored evidence that is the output judged. A result judged on other wording (a template edited
    since) or whose stored evidence is other text is left out and counted, so every rating a rater gives on the read
    is bound to the criterion and the output the rater was shown. One result per arm per case is drawn among the
    case's repeats. The order is one seeded permutation of the cases.
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
    # The cases an arm's judged answer was set aside on, with why: judged on other wording, or other text stored.
    set_aside: dict[str, dict[str, str]] = {arm.key: {} for arm in arms}
    cases: set[str] = set()
    for result in results:
        arm_key = arm_of_run.get(result.eval_run_id)
        if arm_key is None:
            continue
        cases.add(result.test_case_id)
        fingerprints = _usable(result, criteria)
        if fingerprints is None:
            continue
        if not _judged_as_asked(fingerprints, criteria):
            set_aside[arm_key].setdefault(result.test_case_id, EXCLUDED_REWORDED)
            continue
        if result.id not in evidence:
            unshown[arm_key].add(result.test_case_id)
            continue
        if fingerprint_judged_output(evidence[result.id]) != _output_fingerprint(result, criteria):
            set_aside[arm_key].setdefault(result.test_case_id, EXCLUDED_OTHER_TEXT)
            continue
        usable[arm_key].setdefault(result.test_case_id, []).append((result, fingerprints))
    excluded: dict[str, int] = {}
    population: list[str] = []
    for case in sorted(cases):
        if all(case in usable[arm.key] for arm in arms):
            population.append(case)
            continue
        missing = [arm.key for arm in arms if case not in usable[arm.key]]
        reason = next(
            (set_aside[key][case] for key in missing if case in set_aside[key]),
            EXCLUDED_NO_EVIDENCE if any(case in unshown[key] for key in missing) else EXCLUDED_NO_ANSWER,
        )
        excluded[reason] = excluded.get(reason, 0) + 1
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


def blind_sheet(read: BlindRead, *, rater: str | None = None, label_keys: bool = True) -> BlindSheet:
    """The blinded view of ``read``: what a rater is shown, and nothing else.

    :param read: The read.
    :ptype read: BlindRead
    :param rater: Whose progress to report (``answered`` and ``next``); None reports none.
    :ptype rater: str | None
    :param label_keys: Whether each output carries its label key (its fingerprint and each criterion's), which a
        labels file needs to pair a label with its output away from the read. A host's door answers by item and slot
        and leaves them out: a fingerprint is also what a judge score carries, so a rater who can read the scores
        could pair it with its arm.
    :ptype label_keys: bool
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
                outputs=[
                    SheetOutput(
                        slot=output.slot,
                        text=output.evidence.artifact,
                        output_fingerprint=output.output_fingerprint if label_keys else None,
                        criterion_fingerprints=output.criterion_fingerprints if label_keys else {},
                    )
                    for output in item.outputs
                ],
                ask=SheetAsk(verdicts=verdicts, preference="preference" in read.ask and len(item.outputs) == 2),
            )
            for item in read.items
        ],
        answered=[item.item for item in read.items if item.item in answered],
        next=next((item.item for item in read.items if item.item not in answered), None),
    )


def _answered_in_full(read: BlindRead, item: BlindItem, answer: BlindAnswer) -> bool:
    """Whether ``answer`` gives everything ``item`` asks: a verdict per criterion per output, and the preference of a
    pair where one is asked. A skip gives nothing and is its own count."""
    if "verdicts" in read.ask:
        given = {(verdict.slot, verdict.dim) for verdict in answer.verdicts}
        if any((output.slot, dim.name) not in given for output in item.outputs for dim in read.criteria):
            return False
    return not ("preference" in read.ask and len(item.outputs) == 2 and answer.prefer is None)


def read_sample(read: BlindRead, rater: str, kind: Maker = "person") -> BlindSample:
    """Whether ``rater``'s answered items are a random sample: the first ``m`` of the order, or not.

    A skip is an answer, so a skipped item keeps a prefix a prefix; so is an item answered in part. Both follow
    content (a hard item is the one skipped or left half done), so both are counted, for every reading of the sample
    to say.

    :param read: The read.
    :ptype read: BlindRead
    :param rater: The rater, as their answers name them.
    :ptype rater: str
    :param kind: Whether the rater is a person or a model: the two never share a sample, whatever they are named.
    :ptype kind: Maker
    :return: The sample: ``random`` with its counts, ``selected``, or ``none``.
    :rtype: BlindSample
    """
    own = [answer for answer in read.answers if (answer.rater, answer.made_by) == (rater, kind)]
    answered = {answer.item for answer in own}
    order = [item.item for item in read.items]
    items = {item.item: item for item in read.items}
    sample: SampleKind = "none" if not answered else "random" if answered == set(order[: len(answered)]) else "selected"
    return BlindSample(
        rater=rater,
        made_by=kind,
        sample=sample,
        answered=len(answered),
        skipped=len({answer.item for answer in own if answer.skipped}),
        partial=len(
            {
                answer.item
                for answer in own
                if not answer.skipped
                and answer.item in items
                and not _answered_in_full(read, items[answer.item], answer)
            }
        ),
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
    return min(1.0, float(2 * tail / 2**trials))


def check_answer(read: BlindRead, answer: BlindAnswer) -> None:
    """Refuse an answer ``read`` does not ask: to no item of it, a verdict on a slot the item does not show or on no
    criterion of the read, a score its criterion's scale cannot hold, or a preference of an item that asks none.

    Every door checks an answer here before it keeps one, so a close never meets an answer it cannot read.

    :param read: The read.
    :ptype read: BlindRead
    :param answer: The answer offered.
    :ptype answer: BlindAnswer
    :raises ValueError: Naming what the read does not ask.
    """
    item = next((each for each in read.items if each.item == answer.item), None)
    if item is None:
        raise ValueError(f"an answer names item {answer.item!r}, which is no item of blind read {read.id!r}")
    _refuse_unasked_verdicts(read, item, answer)
    if answer.prefer is not None and not ("preference" in read.ask and len(item.outputs) == 2):
        raise ValueError(f"item {answer.item!r} asks no preference, and the answer gives one")


def _refuse_unasked_verdicts(read: BlindRead, item: BlindItem, answer: BlindAnswer) -> None:
    """Refuse a verdict on a slot ``item`` does not show or on no criterion of ``read``, or off its scale.

    :raises ValueError: Naming the verdict.
    """
    slots = {output.slot for output in item.outputs}
    scales = {dim.name: dim.scale for dim in read.criteria}
    for verdict in answer.verdicts:
        scale = scales.get(verdict.dim)
        if verdict.slot not in slots or scale is None:
            raise ValueError(
                f"the answer to item {answer.item!r} gives a verdict on slot {verdict.slot} and {verdict.dim!r}, "
                "which the item does not ask"
            )
        low, high = SCALES[scale].scores
        if not low <= verdict.score <= high:
            raise ValueError(f"{verdict.score!r} is not on the {scale} scale of {verdict.dim!r}")


def _ratings_of(read: BlindRead) -> list[tuple[CalibrationRating, BlindAnswer]]:
    """The read's answers as ratings, each bound to the output and criterion it was given on, beside the answer it
    came from: who gave a rating is read off that answer, never off the rating's rater name.

    :raises ValueError: An answer to no item of the read, a verdict on a slot the item does not show or on no
        criterion of the read, or a score its criterion's scale cannot hold.
    """
    items = {item.item: item for item in read.items}
    scales = {dim.name: dim.scale for dim in read.criteria}
    ratings: dict[str, tuple[CalibrationRating, BlindAnswer]] = {}
    for answer in read.answers:
        item = items.get(answer.item)
        if item is None:
            raise ValueError(f"an answer names item {answer.item!r}, which is no item of blind read {read.id!r}")
        outputs = {output.slot: output for output in item.outputs}
        person = answer.made_by == "person"
        _refuse_unasked_verdicts(read, item, answer)
        for verdict in answer.verdicts:
            output = outputs[verdict.slot]
            scale = scales[verdict.dim]
            rating = CalibrationRating(
                scope_id=read.scope_id,
                run_id=output.run_id,
                result_id=output.result_id,
                rubric_dim=verdict.dim,
                rater=answer.rater,
                # The durable rating knows two kinds, and a model's answer is never the person's. That this agent's
                # rating is a blind read's model reviewer's is held beside it (BlindClose.reviewed), not in its name.
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
            ratings[rating.id] = (rating, answer)
    return list(ratings.values())


def shares_family(reviewer: str, judge_model: str | None) -> bool:
    """Whether two model ids name one provider: the part before the first ``/``, where both have one.

    :param reviewer: A reviewer model's id.
    :ptype reviewer: str
    :param judge_model: The judge's served model, or None where unrecorded.
    :ptype judge_model: str | None
    :return: Whether the two may share a lean their agreement cannot show.
    :rtype: bool
    """
    if judge_model is None or "/" not in reviewer or "/" not in judge_model:
        return False
    return reviewer.split("/", 1)[0] == judge_model.split("/", 1)[0]


# --- what is read off a stored table of counts --------------------------------------------------------------------


class PassFailCounts(NamedTuple):
    """A pass/fail table's four counts, named: the reader down, the judge across."""

    both_fail: int
    reader_fail_judge_pass: int
    reader_pass_judge_fail: int
    both_pass: int


def pass_fail_counts(table: BlindVerdictTable) -> PassFailCounts | None:
    """The four confusion counts of a pass/fail table; None for a 1-5 one, which has no fixed pass.

    :param table: A stored table.
    :ptype table: BlindVerdictTable
    :return: The counts, or None.
    :rtype: PassFailCounts | None
    """
    if table.scores != list(SCALES["pass_fail"].scores):
        return None
    (both_fail, fail_pass), (pass_fail, both_pass) = table.counts
    return PassFailCounts(both_fail, fail_pass, pass_fail, both_pass)


def agreed_pairs(table: BlindVerdictTable) -> int:
    """How many pairs the reader and the judge gave the same score.

    :param table: A stored table.
    :ptype table: BlindVerdictTable
    :return: The count on the table's diagonal.
    :rtype: int
    """
    return sum(table.counts[index][index] for index in range(len(table.scores)))


def single_verdicts(table: BlindVerdictTable) -> tuple[int | None, int | None]:
    """The one score each side gave every pair, where it gave only one: the reader's, then the judge's.

    Where one side gave a single verdict, chance alone accounts for every agreement, so kappa is 0 or undefined at
    any agreement; the counts say what was seen.

    :param table: A stored table.
    :ptype table: BlindVerdictTable
    :return: ``(reader's one score or None, judge's one score or None)``; both None on an empty table.
    :rtype: tuple[int | None, int | None]
    """
    rows = [score for score, row in zip(table.scores, table.counts, strict=True) if sum(row)]
    columns = [score for index, score in enumerate(table.scores) if sum(row[index] for row in table.counts)]
    return (rows[0] if len(rows) == 1 else None, columns[0] if len(columns) == 1 else None)


def table_kappa(table: BlindVerdictTable, scale: RubricScale) -> float | None:
    """Cohen's kappa over a table's pairs, quadratic-weighted on 1-5, read at load from the stored counts. Only the
    quick path's print still states it; no host surface serves it (#1208).

    :param table: A stored table.
    :ptype table: BlindVerdictTable
    :param scale: The criterion's scale.
    :ptype scale: RubricScale
    :return: The kappa; None where undefined.
    :rtype: float | None
    """
    pairs = [
        (reader, judge)
        for reader, row in zip(table.scores, table.counts, strict=True)
        for judge, count in zip(table.scores, row, strict=True)
        for _ in range(count)
    ]
    return cohen_kappa(pairs, list(table.scores), weights="quadratic" if scale == "ordinal" else "none")


def _table(
    scores: Mapping[tuple[str, str], Sequence[int]],
    by_id: Mapping[str, EvalResult],
    dim: str,
    judge: JudgeKey,
    scale: RubricScale,
) -> BlindVerdictTable:
    """Every pair one judge's agreement read on ``dim``, counted by the reader's score and the judge's."""
    low, high = SCALES[scale].scores
    values = list(range(low, high + 1))
    at = {value: index for index, value in enumerate(values)}
    counts = [[0] * len(values) for _ in values]
    for (result_id, rated_dim), given in scores.items():
        result = by_id.get(result_id)
        if rated_dim != dim or result is None or judge_key(result, dim) != judge:
            continue
        score = result.judge_score(dim)
        assert score is not None  # its judge key was read off it
        for rater_score in given:
            counts[at[rater_score]][at[score.score]] += 1
    return BlindVerdictTable(scores=values, counts=counts)


@dataclass(frozen=True)
class BlindAgreementReading:
    """One stored agreement (its counts) beside what the quick path's print still reads at close: kappa with its
    bounds against the bar, and the tier.

    Computed when a read is closed, from its answers and the results, and never stored or served by a host's door
    (#1208: a judge's agreement is stated as counts). The figures are kept here only for the quick path's console
    lines, which a later change converts; the stored document holds :attr:`counts` alone.

    :ivar counts: The stored agreement: who read, how many pairs, and the table.
    :ivar exact_agreement: The share of pairs where both sides gave the same score.
    :ivar kappa: Kappa on pass/fail, quadratic-weighted on 1-5, per person and pooled by result; None where undefined.
    :ivar lower: Its one-sided 95% lower bound.
    :ivar upper: Its one-sided 97.5% upper bound.
    :ivar calibration: How the calibration criterion reads: met, not_met, undecided or insufficient.
    :ivar results_needed: About how many more results would decide it; None once decided.
    :ivar tier_before: The tier the judge's readings bore before these answers.
    :ivar tier: The tier they bear with them, on this anchor.
    :ivar decides: Whether this reading decides the criterion's tier.
    :ivar shares_family: A model's reading only: the reviewer and the judge name one provider.
    """

    counts: BlindAgreement
    exact_agreement: float
    kappa: float | None
    lower: float | None
    upper: float | None
    calibration: str
    results_needed: int | None
    tier_before: str
    tier: str
    decides: bool
    shares_family: bool

    @property
    def dim(self) -> str:
        """The criterion."""
        return self.counts.dim

    @property
    def anchor(self) -> Maker:
        """Whose answers the judge is read against."""
        return self.counts.anchor

    @property
    def anchor_names(self) -> list[str]:
        """The people, or the one reviewer."""
        return self.counts.anchor_names

    @property
    def judge_model(self) -> str | None:
        """The model that served the judge's scores."""
        return self.counts.judge_model

    @property
    def pairs(self) -> int:
        """The pairs read."""
        return self.counts.pairs

    @property
    def results(self) -> int:
        """The distinct results the pairs are of."""
        return self.counts.results

    @property
    def blind_pairs(self) -> int:
        """The pairs given on this blind read."""
        return self.counts.blind_pairs

    @property
    def random_pairs(self) -> int:
        """The pairs from a rater whose sample is random."""
        return self.counts.random_pairs

    @property
    def cases(self) -> int:
        """The distinct cases the pairs' results answer."""
        return self.counts.cases

    @property
    def table(self) -> BlindVerdictTable:
        """Every pair, counted."""
        return self.counts.table


def _agreements(
    ratings: Sequence[CalibrationRating],
    results: Sequence[EvalResult],
    *,
    anchor: Maker,
    names: list[str] | None,
    served_model: str | None,
    before: Mapping[JudgeKey, str],
    own: frozenset[str],
    random_ids: frozenset[str],
) -> list[tuple[DimensionAgreement, BlindAgreementReading]]:
    """One reader configuration's agreement with each judge on each criterion: the counts stored, and beside them
    the tier it bears on that anchor, read at close and kept off the document."""
    if not ratings:
        return []
    rated_by: RaterKind = "person" if anchor == "person" else "agent"
    agreement = judge_agreement(ratings, results, rated_by=rated_by)
    judged = {key for result in results for score in result.judge_scores() if (key := judge_key(result, score.dim))}
    tiers = {
        judge_key_of(tier): tier for tier in judge_evidence_tiers(agreement, judge_self_agreement(results), judged)
    }
    scores = person_scores_by_result(ratings, results, rated_by=rated_by)
    by_id = {result.id: result for result in results}
    read: list[tuple[DimensionAgreement, BlindAgreementReading]] = []
    for entry in agreement.dimensions:
        key = judge_key_of(entry)
        statistic = agreement_statistic(entry.scale, entry.kappa, entry.weighted_kappa)
        criterion = calibration_criterion(entry.n, entry.results, statistic, entry.agreement_interval)
        on_dim = [rating for rating in ratings if rating.rubric_dim == entry.rubric_dim]
        anchor_names = names if names is not None else entry.raters
        counts = BlindAgreement(
            dim=entry.rubric_dim,
            scale=entry.scale,
            judge_model=entry.judge_model,
            anchor=anchor,
            anchor_names=anchor_names,
            served_model=served_model,
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
            table=_table(scores, by_id, entry.rubric_dim, key, entry.scale),
        )
        read.append(
            (
                entry,
                BlindAgreementReading(
                    counts=counts,
                    exact_agreement=entry.exact_agreement,
                    kappa=statistic,
                    lower=None if entry.agreement_interval is None else entry.agreement_interval[0],
                    upper=None if entry.agreement_interval is None else entry.agreement_interval[1],
                    calibration=criterion.state,
                    results_needed=criterion.results_needed,
                    tier_before=before.get(key, "undetermined"),
                    tier=tiers[key].tier,
                    decides=anchor == "person",
                    shares_family=anchor == "model" and shares_family(anchor_names[0], entry.judge_model),
                ),
            )
        )
    return read


def _preference(
    read: BlindRead,
    answers: Sequence[BlindAnswer],
    samples: Mapping[tuple[str, Maker], BlindSample],
    *,
    closing: bool = False,
    served_model: str | None = None,
) -> BlindPreference:
    """One kind of rater's preference over the planned items, per case, with its test; withheld until a rater has
    answered every planned item, unless the read is being closed (``closing``), when an unfinished one is read by the
    anytime test (:func:`_anytime_preference`). ``served_model`` is the model that served a model's ``answers``,
    stated on the preference; None for people."""
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
    kind = answers[0].made_by
    if not done and planned and closing:
        return _anytime_preference(read, answers, samples, by_rater, served_model=served_model)
    if not done or not planned:
        return BlindPreference(
            anchor=kind,
            served_model=served_model,
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
            shares=[],
            state=StateOf(state="withheld", reason="unfinished"),
            slot_a_share=None,
            slot_choices=0,
        )
    shares: dict[str, list[float]] = {}
    slot_a = slot_choices = missing = 0
    for rater in done:
        for answer in by_rater[rater]:
            item = items[answer.item]
            if len(item.outputs) != 2:
                continue
            if answer.prefer is None:
                # Skipped, or answered with the preference left blank: counted, never read as a tie.
                missing += 1
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
    state = StateOf(state="not_available", reason="too_few") if decided < PREFERENCE_MIN_ITEMS else None
    stored = BlindPreference(
        anchor=kind,
        served_model=served_model,
        anchor_names=done,
        random=all(samples[(rater, kind)].sample == "random" for rater in done),
        control=control,
        candidate=candidate,
        planned=len(planned),
        answered=len(planned),
        missing=missing,
        items=len(means),
        prefer_candidate=for_candidate,
        prefer_control=for_control,
        ties=len(means) - decided,
        test=test,
        shares=means,
        state=state,
        slot_a_share=slot_a / slot_choices if slot_choices else None,
        slot_a=slot_a,
        slot_choices=slot_choices,
    )
    return read_preference(stored)


def _anytime_preference(
    read: BlindRead,
    answers: Sequence[BlindAnswer],
    samples: Mapping[tuple[str, Maker], BlindSample],
    by_rater: Mapping[str, Sequence[BlindAnswer]],
    *,
    served_model: str | None = None,
) -> BlindPreference:
    """The preference of a read closed before any rater answered every planned item: read once, at the close, by the
    bounded test on shares in the drawn order with its stakes fixed for the planned pair items before the first
    answer (:func:`~threetears.evals.analysis.stats.bounded_separation_p` with ``planned_n``). Its p holds its level
    at whatever item the read was stopped, so a rater who closes the read when it looks good gains nothing
    (``tests/test_simulated_blind_read.py``). Items with no preference given are left out and counted."""
    control = next(arm.key for arm in read.arms if arm.role == "control")
    candidate = next(arm.key for arm in read.arms if arm.role == "candidate")
    kind = answers[0].made_by
    planned = read.items[: read.draw.planned_items]
    pairs = [item for item in planned if len(item.outputs) == 2]
    shares: dict[str, list[float]] = {}
    slot_a = slot_choices = missing = 0
    for item in pairs:
        for answer in (answer for own in by_rater.values() for answer in own if answer.item == item.item):
            if answer.prefer is None:
                missing += 1
                continue
            if answer.prefer == "tie":
                share = 0.5
            else:
                preferred = next(output for output in item.outputs if output.slot == answer.prefer)
                share = 1.0 if preferred.arm == candidate else 0.0
                slot_choices += 1
                slot_a += answer.prefer == "A"
            shares.setdefault(item.item, []).append(share)
    # In the drawn order: the order the stakes were fixed for.
    means = [sum(shares[item.item]) / len(shares[item.item]) for item in pairs if item.item in shares]
    for_candidate = sum(1 for mean in means if mean > 0.5)
    for_control = sum(1 for mean in means if mean < 0.5)
    decided = for_candidate + for_control
    state = StateOf(state="not_available", reason="too_few") if decided < PREFERENCE_MIN_ITEMS else None
    raters = sorted(by_rater)
    stored = BlindPreference(
        anchor=kind,
        served_model=served_model,
        anchor_names=raters or sorted({answer.rater for answer in answers}),
        random=bool(raters) and all(samples[(rater, kind)].sample == "random" for rater in raters),
        control=control,
        candidate=candidate,
        planned=len(planned),
        answered=max((len({answer.item for answer in own}) for own in by_rater.values()), default=0),
        missing=missing,
        items=len(means),
        prefer_candidate=for_candidate,
        prefer_control=for_control,
        ties=len(means) - decided,
        test="anytime",
        shares=means,
        stake_items=len(pairs),
        state=state,
        slot_a_share=slot_a / slot_choices if slot_choices else None,
        slot_a=slot_a,
        slot_choices=slot_choices,
    )
    return read_preference(stored)


def read_preference(preference: BlindPreference) -> BlindPreference:
    """``preference`` as it reads: its p and verdict computed off its stored counts. The one computation of either:
    a close reads its preferences through it, and a door reads a loaded read's through it (#1236).

    A preference stored since durable 17.5 holds the shares its test reads, in the order it reads them, and on the
    anytime test the planned pair items its stakes were fixed for; never a p or verdict. The test is the one it names:
    the exact sign test over the non-tie items (one answer per item), the bounded test on shares against a half
    (several raters), or the anytime test, the bounded test on shares in the drawn order with its stakes fixed for
    ``stake_items``. A preference not read (``state``: withheld, or too few) has neither, and one stored before 17.5
    (no ``shares``) carries the p and verdict its close computed: both are returned as they are.

    :param preference: A preference as stored, or as a close built it.
    :ptype preference: BlindPreference
    :return: The preference with its p and verdict; ``preference`` itself where it has none to compute.
    :rtype: BlindPreference
    """
    shares = preference.shares
    if shares is None or preference.state is not None:
        return preference
    for_candidate, for_control = preference.prefer_candidate, preference.prefer_control
    p = (
        exact_sign_test_p(for_candidate, for_candidate + for_control)
        if preference.test == "exact_sign"
        else bounded_separation_p(
            [0.5] * len(shares),
            shares,
            paired=True,
            value_range=(0.0, 1.0),
            planned_n=preference.stake_items if preference.test == "anytime" else None,
        )
    )
    verdict: Literal["improved", "regressed", "not_separated"]
    if p is None or p >= SIGNIFICANCE_ALPHA or for_candidate == for_control:
        verdict = "not_separated"
    else:
        verdict = "improved" if for_candidate > for_control else "regressed"
    return preference.model_copy(update={"p": p, "verdict": verdict})


def _reviewer_checks(
    people: Sequence[CalibrationRating],
    reviewed: Mapping[tuple[str, str | None], Sequence[CalibrationRating]],
    criteria: Sequence[RubricDim],
) -> list[BlindReviewerCheck]:
    """Each model reviewer's answers against people's on the outputs both answered, per criterion, as counts.

    ``reviewed`` is each reviewer's ratings as the read's own answers gave them, by its model id and the model that
    served them: answers two models served are counted apart, never pooled under the reviewer's name.
    """
    by_people: dict[tuple[str, str, str], list[int]] = {}
    for rating in people:
        if rating.label_key is not None:
            by_people.setdefault((rating.rubric_dim, *rating.label_key), []).append(rating.score)
    checks: list[BlindReviewerCheck] = []
    for reviewer, served in sorted(reviewed, key=lambda key: (key[0], key[1] or "")):
        for dim in criteria:
            pairs = [
                (score, rating.score)
                for rating in reviewed[(reviewer, served)]
                if rating.rubric_dim == dim.name and rating.label_key is not None
                for score in by_people.get((rating.rubric_dim, *rating.label_key), ())
            ]
            if not pairs:
                continue
            low, high = SCALES[dim.scale].scores
            values = list(range(low, high + 1))
            checks.append(
                BlindReviewerCheck(
                    dim=dim.name,
                    reviewer=reviewer,
                    served_model=served,
                    outputs=len(pairs),
                    table=BlindVerdictTable(
                        scores=values,
                        counts=[
                            [sum(1 for pair in pairs if pair == (row, column)) for column in values] for row in values
                        ],
                    ),
                )
            )
    return checks


#: Why a model reviewer's answers are not read as a check of the judge: it is the judge's own model.
REFUSED_SAME_MODEL: Final = (
    "the reviewer is the judge's own model, so its answers repeat the judge and are not read as a check of it"
)

#: Why they are not: which model served the judge's scores, or the reviewer's answers, was not recorded.
REFUSED_UNRECORDED: Final = (
    "which model served {side} was not recorded, so the reviewer cannot be shown to be another model than the judge, "
    "and its answers are not read as a check of it"
)


def close_blind_read(
    read: BlindRead,
    results: Iterable[EvalResult],
    tiers_before: Sequence[JudgeEvidenceTier] | None = None,
    *,
    other_ratings: Sequence[CalibrationRating] = (),
    unread: Sequence[str] = (),
    judge_models: Collection[str] = (),
    closed_by: str | None = None,
    closed_at: str | None = None,
) -> BlindClose:
    """Close ``read``: its answers as ratings, and what they say of the judge and of the arms.

    A person's answers and a model's are read apart. On each criterion the person reading decides the tier once it is
    itself decided; until then a model's reading stands. A model's reading is refused where the model is the judge's
    own (its id, or any model that served one of its answers, is the judge's asked or served model), and where which
    model served either side was not recorded: an unrecorded model matches nothing, so the guard cannot be passed.

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
    :param judge_models: The ids the judge was asked as, beside the served model each score records: a reviewer
        named or served as any of them is the judge's own model.
    :ptype judge_models: Collection[str]
    :param closed_by: Who closed it; None leaves the read open, its report computed. A read closed before any rater
        answered every planned item is closed early (``BlindRead.closed_early``): an unfinished preference is then
        read once, by the anytime test, where an open read withholds it.
    :ptype closed_by: str | None
    :param closed_at: When; None for now, where ``closed_by`` is given.
    :ptype closed_at: str | None
    :return: The read carrying its report, the ratings, and which of them came from a random sample.
    :rtype: BlindClose
    :raises ValueError: An answer the read does not ask (see :func:`_ratings_of`).
    """
    results = list(results)
    given = _ratings_of(read)
    ratings = [rating for rating, _ in given]
    keys = list(dict.fromkeys((answer.rater, answer.made_by) for answer in read.answers))
    samples = {key: read_sample(read, *key) for key in keys}
    own = frozenset(rating.id for rating in ratings)
    random_ids = frozenset(
        rating.id for rating, answer in given if samples[(answer.rater, answer.made_by)].sample == "random"
    )
    if tiers_before is None:
        judged = {key for result in results for score in result.judge_scores() if (key := judge_key(result, score.dim))}
        tiers_before = judge_evidence_tiers(judge_agreement([], results), judge_self_agreement(results), judged)
    before = {judge_key_of(tier): tier.tier for tier in tiers_before}
    people = [
        *(rating for rating, answer in given if answer.made_by == "person"),
        *(rating for rating in other_ratings if rating.rater_kind == "person"),
    ]
    agreement = [
        reading
        for _, reading in _agreements(
            people,
            results,
            anchor="person",
            names=None,
            served_model=None,
            before=before,
            own=own,
            random_ids=random_ids,
        )
    ]
    # A model's answers, by the model and by the model that served each: read off the answers themselves, every one
    # of them. A reviewer is one model id; the answers two served models gave under it are read apart, never pooled.
    reviewed: dict[tuple[str, str | None], list[CalibrationRating]] = {}
    served: dict[str, set[str]] = {}
    unrecorded: set[str] = set()
    for answer in read.answers:
        if answer.made_by != "model" or answer.skipped:
            continue
        if answer.reviewer_served_model is None:
            unrecorded.add(answer.rater)
        else:
            served.setdefault(answer.rater, set()).add(answer.reviewer_served_model)
    for rating, answer in given:
        if answer.made_by == "model":
            reviewed.setdefault((answer.rater, answer.reviewer_served_model), []).append(rating)
    asked = set(judge_models)
    refused: dict[str, list[str]] = {}
    for reviewer, by in sorted(reviewed, key=lambda key: (key[0], key[1] or "")):
        for entry, reading in _agreements(
            reviewed[(reviewer, by)],
            results,
            anchor="model",
            names=[reviewer],
            served_model=by,
            before=before,
            own=own,
            random_ids=random_ids,
        ):
            judge_names = asked | ({entry.judge_model} if entry.judge_model is not None else set())
            reason: str | None = None
            if ({reviewer} | served.get(reviewer, set())) & judge_names:
                reason = REFUSED_SAME_MODEL
            elif entry.judge_model is None:
                reason = REFUSED_UNRECORDED.format(side="the judge's scores")
            elif reviewer in unrecorded:
                reason = REFUSED_UNRECORDED.format(side="the reviewer's answers")
            if reason is not None:
                if reason not in refused.setdefault(reviewer, []):
                    refused[reviewer].append(reason)
                continue
            agreement.append(reading)
    agreement = _with_the_deciding_anchor(agreement)
    closing = closed_by is not None
    planned_ids = {item.item for item in read.items[: read.draw.planned_items]}
    finished = any(
        planned_ids <= {answer.item for answer in read.answers if (answer.rater, answer.made_by) == key} for key in keys
    )
    preferences: list[BlindPreference] = []
    if "preference" in read.ask:
        makers: tuple[Maker, ...] = ("person", "model")
        for kind in makers:
            for served_model, group in _preference_groups(read, kind):
                if group:
                    preferences.append(_preference(read, group, samples, closing=closing, served_model=served_model))
    report = BlindReadReport(
        cases=len(read.items),
        reviews=list(read.reviews),
        samples=list(samples.values()),
        agreement=[reading.counts for reading in agreement],
        reviewer_checks=_reviewer_checks(people, reviewed, read.criteria),
        not_checked=refused,
        preferences=preferences,
        other_ratings=len(other_ratings),
        unread=list(unread),
    )
    closed = read.model_copy(
        update={
            "report": report,
            "closed_by": closed_by,
            "closed_at": (closed_at or utc_now_iso()) if closed_by is not None else None,
            "closed_early": closing and bool(planned_ids) and not finished,
        }
    )
    return BlindClose(
        read=closed,
        ratings=ratings,
        random=random_ids,
        reviewed={rating.id: answer.rater for rating, answer in given if answer.made_by == "model"},
        agreement=tuple(agreement),
    )


def _with_the_deciding_anchor(agreement: Sequence[BlindAgreementReading]) -> list[BlindAgreementReading]:
    """Mark which reading decides each criterion's tier for each judge.

    A person reading decides once it is itself decided (met or not met), or where no reference reading stands beside
    it. Until then the first reference reading decides, so a few person labels never remove a standing a reference
    reviewer earned; the person reading is shown beside it.
    """
    marked: list[BlindAgreementReading] = []
    for reading in agreement:
        beside = [other for other in agreement if (other.dim, other.judge_model) == (reading.dim, reading.judge_model)]
        person = next((other for other in beside if other.anchor == "person"), None)
        references = [other for other in beside if other.anchor == "model"]
        person_decides = person is not None and (person.calibration in ("met", "not_met") or not references)
        if reading.anchor == "person":
            decides = person_decides
        else:
            decides = not person_decides and reading is references[0]
        marked.append(replace(reading, decides=decides))
    return marked


def _preference_groups(read: BlindRead, kind: Maker) -> list[tuple[str | None, list[BlindAnswer]]]:
    """The answers a preference is read over, each beside the model that served them: every person's together (no
    served model), and each reference reviewer's alone, once per model that served its answers, never pooled.

    A skipped answer names no served model, since no reply was used: it goes with every served model the reviewer's
    answers name, where it counts toward how far the read is and as a planned item with no preference given.
    """
    answers = [answer for answer in read.answers if answer.made_by == kind]
    if kind == "person":
        return [(None, answers)] if answers else []
    groups: list[tuple[str | None, list[BlindAnswer]]] = []
    for rater in sorted({answer.rater for answer in answers}):
        own = [answer for answer in answers if answer.rater == rater]
        served = sorted(
            {answer.reviewer_served_model for answer in own if not answer.skipped}, key=lambda name: name or ""
        )
        if not served:
            groups.append((None, own))
            continue
        for model in served:
            groups.append(
                (model, [answer for answer in own if answer.skipped or answer.reviewer_served_model == model])
            )
    return groups


#: What another module calls: the quick path's files and the host's door (:mod:`threetears.evals.ops.blind_reads`).
__all__ = [
    "PREFERENCE_MIN_ITEMS",
    "BlindAgreementReading",
    "BlindClose",
    "agreed_pairs",
    "blind_sheet",
    "check_answer",
    "close_blind_read",
    "draw_blind_read",
    "exact_sign_test_p",
    "pass_fail_counts",
    "read_preference",
    "read_sample",
    "shares_family",
    "single_verdicts",
    "table_kappa",
]
