"""A judged contrast read again with a blind read's labels: the ``corrected`` check of its verdict.

A judge's difference between two arms carries the judge's noise in its interval already, and a bias equal in both
arms cancels. A bias that differs by arm (a lean) does not, and nothing in the judge's own figures can see it. A blind
read answered from the top is a random sample of the read's cases, and on each pair item it gives what the reader
saw between the two outputs beside what the judge saw between the same two. That measures the lean, and removes it.

**The estimate** (:func:`prediction_powered_difference`) is prediction-powered inference on paired differences. For
each case ``i`` of the read's population ``G``, ``F_i`` is the judge's difference of the two arms' case means. On the
labelled cases ``L`` (a random sample of ``n``), ``Y_i`` is the reader's difference on the two drawn outputs and
``F^L_i`` the judge's on the same two, and ``R_i = Y_i - F^L_i``::

    estimate = mean over G of F_i  +  mean over L of R_i

Its standard error is the case-clustered linearisation of
:func:`~threetears.evals.analysis.stats.prediction_powered_mean`, which this calls rather than restates, read on t with
``n - 1`` degrees of freedom. A case whose outputs are byte-identical across every repeat of both arms is a known
stratum: the reader's difference there is zero, so it is weighted in at its share of ``G`` and estimates nothing. A
case whose outputs vary across repeats is never in it, even where its two drawn outputs happen to coincide: that draw
is an ordinary labelled pair, the reader's difference zero and the judge's its own on the two results, since the
case's difference in expectation is not zero.

**The check** (:func:`corrected_reading`) is one field on the contrast, ``corrected``: a result of the four
:data:`~threetears.evals.kernel.evidence_tiers.CriterionState` values on the contrast's one verdict, never a second
verdict. ``met`` confirms it; ``not_met`` with ``people_disagree`` contradicts it; ``not_met`` with
``instrument_leans`` finds the judge leaning toward the arm the verdict favours, unconfirmed; ``undecided`` confirms
nothing at the decision floor or above; ``insufficient`` with ``too_few`` holds figures below it. Where no figure can
be stated the field's state is ``not_available`` with one reason of the shared set
(:mod:`~threetears.evals.schema.reading_states`). Whether a check may move a decision is read off its labelled cases
(:func:`decides`), never stored, and only a person's check does: a model's check is the field only where no person
read the contrast, and otherwise rides beside the person's (``model_check``) so a lean it found is never lost.

``tests/test_simulated_people_corrected.py`` holds the estimator's coverage and the decision floor
(:data:`PPI_DECISION_MIN_LABELLED_CASES`, the smallest number of labelled cases whose rows met every bar with two Monte
Carlo standard errors to spare, and at least 40: 80).

This module computes numbers and prints nothing.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Final, Literal, NamedTuple

from threetears.evals.analysis.blind_read import sample_standing
from threetears.evals.analysis.checks import Corrected
from threetears.evals.analysis.stats import prediction_powered_mean, t_critical_two_sided
from threetears.evals.kernel.evidence_tiers import CriterionState
from threetears.evals.kernel.numbers import format_number
from threetears.evals.kernel.result_condition import ResultOutcome, classify_result
from threetears.evals.kernel.result_reading import ResultView
from threetears.evals.schema.blind_read import BlindAnswer, BlindItem, BlindRaterKind, BlindRead
from threetears.evals.schema.models import SCALES
from threetears.evals.schema.reading_states import NotAvailableReason, Reason, SampleKind, StateOf

#: What production calls: the bundle's judged-contrast filler and the cell reads.
__all__ = ["corrected_reading", "counted_label", "pair_labels"]

#: The fewest labelled cases a ``corrected`` check states figures from. Below it the field is ``not_available``,
#: ``too_few``: the rectifier's spread is too uncertain to put an interval on.
PPI_REPORT_MIN_LABELLED_CASES: Final = 20

#: The fewest labelled cases a ``corrected`` check decides from. From :data:`PPI_REPORT_MIN_LABELLED_CASES` up to it
#: the result is ``insufficient``, ``too_few``, its figures shown and moving nothing. One constant, not a rule computed
#: from the data: the failure it guards, a lean concentrated in a few cases or answers that the sample never met (its
#: rectifier near zero with little spread, the corrected interval collapsing toward the judge's own), cannot be seen in
#: the sample. Its basis is ``tests/test_simulated_people_corrected.py``'s deciding rows, 16,000 replicates each: the
#: smallest n at which every rare-lean row covers at least 93.5% and confirms a difference people do not see at most
#: 3.5%, each with two Monte Carlo standard errors to spare. At 60 a judge giving a 5 to one candidate answer in twenty
#: on 1 to 5 confirms 3.61% (SE 0.15). At 80 it confirms 2.93% (SE 0.13), about three standard errors above the nominal
#: 2.5%: the interval still over-confirms under that lean, inside the bar only by the slack the bar allows. Read at 95%;
#: production reads at the Holm step's level, 97.5% and up, where it confirms less.
PPI_DECISION_MIN_LABELLED_CASES: Final = 80

#: The largest share of a sample's answered items a reader may skip, on the criterion read, before the sample stops
#: standing for the cases: a skip follows content (the hard item is the one passed over), so past it the field is
#: ``not_available``, ``skipped_past_limit``.
SKIPPED_LIMIT: Final = 0.10


class PeopleCorrectedDifference(NamedTuple):
    """A judged difference between two arms corrected by a reader's labels on some of its cases.

    Every figure is on the contrast's own scale, as candidate minus control.

    :ivar estimate: The judge's mean difference plus the rectifier, weighted with the identical stratum's zero.
    :ivar judge_difference: The judge's mean difference over the population, identical cases included.
    :ivar rectifier: The mean of reader minus judge over the labelled pair items.
    :ivar sem: The estimate's standard error; None below two labelled cases.
    :ivar interval: The interval at the asked confidence on ``labelled_cases - 1`` degrees of freedom; None where
        ``sem`` is.
    :ivar lean: The judge's lean, judge minus reader over the labelled pair items (minus ``rectifier``): positive
        where the judge reads the candidate higher against the control than the reader does.
    :ivar lean_interval: The interval on the lean at the same confidence; None below two labelled cases.
    :ivar cases: The population's cases.
    :ivar labelled_cases: The labelled pair items.
    :ivar identical_cases: The population's cases whose outputs were byte-identical across every repeat of both arms.
    """

    estimate: float
    judge_difference: float
    rectifier: float
    sem: float | None
    interval: tuple[float, float] | None
    lean: float
    lean_interval: tuple[float, float] | None
    cases: int
    labelled_cases: int
    identical_cases: int


def prediction_powered_difference(
    judge_differences: Mapping[str, float],
    labelled: Mapping[str, tuple[float, float]],
    *,
    identical: Collection[str] = (),
    confidence: float,
) -> PeopleCorrectedDifference | None:
    """The prediction-powered estimate of the difference between two arms, with the judge's lean.

    The non-identical cases are one prediction-powered mean (:func:`prediction_powered_mean`), whose judge scores are
    the cases' ``F_i`` and whose labels are ``F_i + R_i`` on the labelled cases, so its rectifier is the mean of
    ``R_i``; its variance is that function's linearisation over cases, on the labelled cases' degrees of freedom. The
    identical cases add zero at their share of the population, so the estimate and its standard error are scaled by
    the other cases' share.

    :param judge_differences: ``F_i``, the judge's candidate-minus-control difference of case means, for every case of
        the population, identical cases included.
    :ptype judge_differences: Mapping[str, float]
    :param labelled: For each labelled case, ``(Y_i, F^L_i)``: the reader's and the judge's candidate-minus-control
        difference on the two drawn outputs.
    :ptype labelled: Mapping[str, tuple[float, float]]
    :param identical: The population's cases whose outputs were byte-identical across every repeat of both arms.
    :ptype identical: Collection[str]
    :param confidence: The intervals' two-sided level.
    :ptype confidence: float
    :return: The estimate; None with no labelled case.
    :rtype: PeopleCorrectedDifference | None
    :raises ValueError: A labelled or identical case outside the population, a labelled case that is identical, or a
        confidence outside (0, 1).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"a confidence is between 0 and 1; got {confidence!r}")
    population = set(judge_differences)
    stray = (set(labelled) | set(identical)) - population
    if stray:
        raise ValueError(f"cases outside the population: {sorted(map(str, stray))}")
    if set(labelled) & set(identical):
        raise ValueError("an identical case has no pair to label: its reader difference is zero by construction")
    if not labelled:
        return None
    same = set(identical)
    rest = [case for case in judge_differences if case not in same]
    share = len(rest) / len(population)
    gaps = {case: reader - judge for case, (reader, judge) in labelled.items()}
    powered = prediction_powered_mean(
        [judge_differences[case] for case in rest],
        rest,
        [judge_differences[case] + gaps[case] if case in gaps else None for case in rest],
    )
    assert powered is not None  # some case is labelled
    n = len(gaps)
    estimate = share * powered.mean
    sem = None if powered.sem is None else share * powered.sem
    rectifier = math.fsum(gaps.values()) / n
    lean_interval = None
    interval = None
    if sem is not None:
        critical = t_critical_two_sided(confidence, n - 1)
        interval = (estimate - critical * sem, estimate + critical * sem)
        spread = math.fsum((gap - rectifier) ** 2 for gap in gaps.values()) / (n - 1)
        half = critical * math.sqrt(spread / n)
        lean_interval = (-rectifier - half, -rectifier + half)
    return PeopleCorrectedDifference(
        estimate=estimate,
        judge_difference=math.fsum(judge_differences.values()) / len(population),
        rectifier=rectifier,
        sem=sem,
        interval=interval,
        lean=-rectifier,
        lean_interval=lean_interval,
        cases=len(population),
        labelled_cases=n,
        identical_cases=len(same),
    )


def counted_label(result: ResultView, dimension: str, score: float) -> float:
    """A reader's score of one result, counted as the judge's score of it is.

    On a candidate failure the judge's score counts the scale's floor
    (:func:`~threetears.evals.kernel.result_condition.counted_score`), so the reader's does too: the correction
    measures the judge where it was read, never a disagreement about a turn the measure already floors.

    :param result: The result read.
    :ptype result: ResultView
    :param dimension: The judged dimension.
    :ptype dimension: str
    :param score: The reader's score, or several readers' mean.
    :ptype score: float
    :return: The score as counted.
    :rtype: float
    """
    judged = result.judge_score(dimension)
    if judged is not None and classify_result(result) is ResultOutcome.CANDIDATE_FAIL:
        return float(SCALES[judged.scale].scores[0])
    return score


class PairLabels(NamedTuple):
    """One reader's labels of one criterion on one blind read over a control and a candidate.

    :ivar against: Who read: ``people`` for a person, ``model`` for a reference reviewer.
    :ivar against_id: The model's id; None for a person.
    :ivar read_id: The blind read.
    :ivar sample: Whether the reader's answered items are the first of the drawn order.
    :ivar answered: The items the reader answered, skipped ones included.
    :ivar skipped: Of those, the items that gave no verdict on both outputs on this criterion.
    :ivar population: The read's cases on which both arms carry the judge's case mean.
    :ivar identical: Of those, the cases whose outputs were byte-identical across every repeat of both arms.
    :ivar pairs: Each labelled case's ``(Y_i, F^L_i)``, candidate minus control.
    """

    against: Literal["people", "model"]
    against_id: str | None
    read_id: str
    sample: SampleKind
    answered: int
    skipped: int
    population: frozenset[str]
    identical: frozenset[str]
    pairs: Mapping[str, tuple[float, float]]


def pair_labels(
    read: BlindRead,
    dimension: str,
    *,
    control: tuple[str, Mapping[str, ResultView]],
    candidate: tuple[str, Mapping[str, ResultView]],
    judge_values: Mapping[str, float],
    case_differences: Mapping[str, float],
) -> list[PairLabels]:
    """Every reader's labels of ``dimension`` on ``read``, where the read was drawn over these two cells.

    A read applies when its control and candidate are the two arms and every output it shows is a result of the
    matching cell: one drawn over other runs labels another population. A model's answers and a person's are read
    apart and never pooled.

    :param read: The blind read.
    :ptype read: BlindRead
    :param dimension: The criterion.
    :ptype dimension: str
    :param control: The control's variant key and its cell's results by id.
    :ptype control: tuple[str, Mapping[str, ResultView]]
    :param candidate: The candidate's variant key and its cell's results by id.
    :ptype candidate: tuple[str, Mapping[str, ResultView]]
    :param judge_values: The judge's counted score of each result on ``dimension``, by result id; a result with none
        (faulted, or the judge could not tell) labels nothing.
    :ptype judge_values: Mapping[str, float]
    :param case_differences: ``F_i`` per case, from the cells' judged case means.
    :ptype case_differences: Mapping[str, float]
    :return: One entry per reader who answered the read; empty when the read does not apply.
    :rtype: list[PairLabels]
    """
    roles = {arm.role: arm.key for arm in read.arms}
    if roles != {"control": control[0], "candidate": candidate[0]} or dimension not in {c.name for c in read.criteria}:
        return []
    cells = {control[0]: control[1], candidate[0]: candidate[1]}
    if any(output.result_id not in cells[output.arm] for item in read.items for output in item.outputs):
        return []
    population = frozenset(item.case_id for item in read.items if item.case_id in case_differences)
    identical = _identical_cases(cells.values(), dimension, population, judge_values)
    readers = sorted({(answer.rater, answer.rater_kind) for answer in read.answers})
    found: list[PairLabels] = []
    for rater, kind in readers:
        own: dict[str, BlindAnswer] = {}
        for given_answer in read.answers:
            if (given_answer.rater, given_answer.rater_kind) == (rater, kind):
                own[given_answer.item] = given_answer
        pairs: dict[str, tuple[float, float]] = {}
        skipped = 0
        for item in read.items:
            answer = own.get(item.item)
            if answer is None or item.case_id in identical:
                continue
            given = {verdict.slot: verdict.score for verdict in answer.verdicts if verdict.dim == dimension}
            if answer.skipped or set(given) != {output.slot for output in item.outputs}:
                skipped += 1
                continue
            if item.case_id not in population:
                continue
            if len(item.outputs) == 1:
                # The two drawn outputs coincided on a case whose outputs vary across repeats: a pair the reader sees
                # no difference in, read against the judge's own on the two results.
                coinciding = _coinciding(item, cells[candidate[0]], dimension, judge_values)
                if coinciding is not None:
                    pairs[item.case_id] = (0.0, coinciding)
                continue
            by_arm = {output.arm: output for output in item.outputs}
            low, high = by_arm[control[0]], by_arm[candidate[0]]
            if low.result_id not in judge_values or high.result_id not in judge_values:
                continue
            reader = counted_label(cells[candidate[0]][high.result_id], dimension, float(given[high.slot])) - (
                counted_label(cells[control[0]][low.result_id], dimension, float(given[low.slot]))
            )
            pairs[item.case_id] = (reader, judge_values[high.result_id] - judge_values[low.result_id])
        standing = sample_standing(read, rater, kind)
        found.append(
            PairLabels(
                against=_against(kind),
                against_id=None if kind == "person" else rater,
                read_id=read.id,
                sample=standing.sample,
                answered=standing.answered,
                skipped=skipped,
                population=population,
                identical=identical,
                pairs=pairs,
            )
        )
    return found


def _identical_cases(
    cells: Iterable[Mapping[str, ResultView]],
    dimension: str,
    population: Collection[str],
    judge_values: Mapping[str, float],
) -> frozenset[str]:
    """The cases of ``population`` whose every counted result, in both cells and on every repeat, judged one output.

    A result whose score carries no output fingerprint cannot say, so its case is not in the stratum.
    """
    seen: dict[str, set[str | None]] = {}
    for cell in cells:
        for result in cell.values():
            if result.test_case_id in population and result.id in judge_values:
                score = result.judge_score(dimension)
                seen.setdefault(result.test_case_id, set()).add(None if score is None else score.output_fingerprint)
    return frozenset(case for case, outputs in seen.items() if len(outputs) == 1 and None not in outputs)


def _coinciding(
    item: BlindItem, candidate: Mapping[str, ResultView], dimension: str, judge_values: Mapping[str, float]
) -> float | None:
    """The judge's candidate-minus-control difference on an item whose two drawn outputs coincided.

    The item shows the control's result; the candidate's drawn one is a result of the case judging the same output,
    so the candidate's side is the judge's mean over those. None where either side has no counted score.
    """
    (shown,) = item.outputs
    if shown.result_id not in judge_values:
        return None
    same = [
        judge_values[result.id]
        for result in candidate.values()
        if result.test_case_id == item.case_id
        and result.id in judge_values
        and (score := result.judge_score(dimension)) is not None
        and score.output_fingerprint == shown.output_fingerprint
    ]
    if not same:
        return None
    return math.fsum(same) / len(same) - judge_values[shown.result_id]


def _against(kind: BlindRaterKind) -> Literal["people", "model"]:
    """What a reader of this kind checks a judge against."""
    return "people" if kind == "person" else "model"


def _unavailable(labels: PairLabels) -> NotAvailableReason | None:
    """Why a reader's labels state no figure, in the order a remedy is reached; None where they state one."""
    if labels.sample != "random":
        return "not_drawn_at_random"
    if labels.answered and labels.skipped / labels.answered > SKIPPED_LIMIT:
        return "skipped_past_limit"
    if len(labels.pairs) < PPI_REPORT_MIN_LABELLED_CASES:
        return "too_few"
    return None


def corrected_reading(
    labels: Sequence[PairLabels],
    case_differences: Mapping[str, float],
    *,
    direction: Literal[-1, 1],
    confidence: float,
    contrast_state: StateOf | None = None,
) -> Corrected | None:
    """The ``corrected`` check of one judged contrast's verdict, from the readers who labelled its criterion.

    A person's check is the field wherever a person labelled the criterion, whatever it reads: a model's read never
    lifts a state a person's read is required for, so it never stands in for a person's check that cannot decide.
    Where a model also labelled it, the model's check rides beside it (``Corrected.model_check``), so a lean the model
    found is never hidden by a person's check that is undecided or short of labels. With no person, the model's check
    is the field. Among readers of one kind, one whose labels state a figure goes before one whose do not, and more
    labelled cases before fewer; the rest are ties, broken by the read and the reader. A person and a model are never
    pooled, nor two readers.

    The result, where a figure is stated:

    - below :data:`PPI_DECISION_MIN_LABELLED_CASES`: ``insufficient``, ``too_few``;
    - else ``met`` where the corrected interval excludes zero in the verdict's direction;
    - ``not_met``, ``people_disagree``, where it excludes zero the other way (raised for a model's labels too, whose
      field says ``against``);
    - ``not_met``, ``instrument_leans``, where the lean's interval excludes zero toward the arm the verdict favours;
    - ``undecided`` otherwise.

    The field's state is ``not_available`` where the contrast's runs left a coordinate unrecorded
    (``identity_unrecorded``), the reader's sample was not drawn at random, more than :data:`SKIPPED_LIMIT` of it was
    skipped, or fewer than :data:`PPI_REPORT_MIN_LABELLED_CASES` cases were labelled.

    :param labels: Every reader's labels of the contrast's criterion on the reads drawn over its two cells.
    :ptype labels: Sequence[PairLabels]
    :param case_differences: ``F_i`` per case over the cells' shared cases.
    :ptype case_differences: Mapping[str, float]
    :param direction: The sign of the verdict's delta: 1 where the candidate's figure rose, -1 where it fell.
    :ptype direction: Literal[-1, 1]
    :param confidence: The level the contrast's Holm step was read at.
    :ptype confidence: float
    :param contrast_state: The contrast's own state, where it is not counted.
    :ptype contrast_state: StateOf | None
    :return: The check; None where no reader labelled the criterion on a read over these two cells.
    :rtype: Corrected | None
    """
    if not labels:
        return None
    people = [one for one in labels if one.against == "people"]
    models = [one for one in labels if one.against == "model"]
    checks = {
        kind: _check(
            _first(readers), case_differences, direction=direction, confidence=confidence, state=contrast_state
        )
        for kind, readers in (("people", people), ("model", models))
        if readers
    }
    if "people" not in checks:
        return checks["model"]
    return checks["people"].model_copy(update={"model_check": checks.get("model")})


def _first(readers: Sequence[PairLabels]) -> PairLabels:
    """The reader of one kind whose labels speak: one stating a figure, then the most labelled cases, then by id."""
    return min(
        readers,
        key=lambda one: (_unavailable(one) is not None, -len(one.pairs), one.read_id, one.against_id or ""),
    )


def _check(
    reader: PairLabels,
    case_differences: Mapping[str, float],
    *,
    direction: Literal[-1, 1],
    confidence: float,
    state: StateOf | None,
) -> Corrected:
    """One reader's check of the verdict: its figures and result, or the state that keeps them from being stated."""
    level = f"{format_number(100 * confidence)}%"
    named = Corrected(
        against=reader.against,
        against_id=reader.against_id,
        labelled_cases=len(reader.pairs),
        identical_cases=len(reader.identical),
        read_id=reader.read_id,
    )
    if state is not None and state.reason == "identity_unrecorded":
        unrecorded = StateOf(state="not_available", reason="identity_unrecorded", coordinate=state.coordinate)
        return named.model_copy(update={"state": unrecorded})
    if (reason := _unavailable(reader)) is not None:
        return named.model_copy(update={"state": StateOf(state="not_available", reason=reason)})
    figures = prediction_powered_difference(
        {case: case_differences[case] for case in reader.population},
        reader.pairs,
        identical=reader.identical,
        confidence=confidence,
    )
    assert figures is not None and figures.interval is not None and figures.lean_interval is not None  # n >= 20
    result, why = _result(figures, direction)
    return named.model_copy(
        update={
            "result": result,
            "reason": why,
            "estimate": figures.estimate,
            "interval": figures.interval,
            "level": level,
            "lean": figures.lean,
            "lean_interval": figures.lean_interval,
        }
    )


def _result(figures: PeopleCorrectedDifference, direction: Literal[-1, 1]) -> tuple[CriterionState, Reason | None]:
    """The check's result and reason from its figures, on a sample that states them."""
    if figures.labelled_cases < PPI_DECISION_MIN_LABELLED_CASES:
        return "insufficient", "too_few"
    assert figures.interval is not None and figures.lean_interval is not None
    if _excludes_zero(figures.interval, direction):
        return "met", None
    if _excludes_zero(figures.interval, -direction):
        return "not_met", "people_disagree"
    if _excludes_zero(figures.lean_interval, direction):
        return "not_met", "instrument_leans"
    return "undecided", None


def _excludes_zero(interval: tuple[float, float], direction: int) -> bool:
    """Whether an interval lies wholly on the ``direction`` side of zero."""
    low, high = interval
    return low > 0.0 if direction > 0 else high < 0.0


def decides(corrected: Corrected) -> bool:
    """Whether a ``corrected`` check may move a decision: a person's check stating figures from at least
    :data:`PPI_DECISION_MIN_LABELLED_CASES` labelled cases. Read off the field, never stored on it.

    A model's check never decides, the field's or the one carried beside a person's (``model_check``): its coverage is
    for the model's opinion, not people's. A lean it found (``not_met``, ``instrument_leans``) is a warning the reader
    is shown, never a decision this reads.

    :param corrected: The check.
    :ptype corrected: Corrected
    :return: Whether it decides.
    :rtype: bool
    """
    return (
        corrected.against == "people"
        and corrected.state is None
        and corrected.result is not None
        and (corrected.labelled_cases or 0) >= PPI_DECISION_MIN_LABELLED_CASES
    )
