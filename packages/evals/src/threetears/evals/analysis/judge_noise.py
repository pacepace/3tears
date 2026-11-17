"""Re-score noise: a judge checked against itself, read one step down in the judge's own table.

A judge asked again about a reply it already scored can score it differently. That is re-score noise, and only that:
a judge that repeats itself exactly (a deterministic or cached one) reads no noise and may still be wrong, so the
figures here never say how good a judge is.

Three figures, each filled where the bundle assembles the judge's table (:func:`with_rescore_noise`):

- **Re-score variance**, per arm: half the mean squared difference between a counted score and its repeat, which is
  the variance between two scorings of one reply. Repeats are read as they count (a candidate failure counts the
  scale's floor on both readings, so it moves nothing), only a repeat by the very judge that scored the arm, and only
  a repeat of a random draw of the arm's results (``JudgeRepeat.sample``): every figure here, the share at k = 1 most
  of all, holds for the arm only if the results repeated stand for it. Results named by id, or repeats stored before
  the draw was recorded, read ``not available`` (``not_drawn_at_random``, ``not_recorded``).
- **The judge's share of a contrast's width**, per paired judged contrast:
  ``(σ²_control · mean(1/k_control) + σ²_candidate · mean(1/k_candidate)) / variance of the per-case differences``,
  where ``k`` is a case's counted scores on that side. It exists at k = 1, since the repeats, not the spread within a
  case, measure the judge. Each arm's own variance is read for it: two arms' answers, or their judges, can differ, and
  one arm's repeats are not taken to speak for the other's.
- **The judge's ceiling on pass^k**, per arm of a pass/fail dimension that counts toward a trial's pass: ``e`` is the
  share of repeats that fail among results whose first reading passed, and the ceiling is ``(1 − e)^(k − 1)``.

Each is a range at :data:`~threetears.evals.analysis.stats.INTERVAL_LEVEL`. The variance's and the share's are one
interval, Fay and Feuer's gamma interval on a weighted sum of counts (1997), since a mean of half squared differences
on an integer scale is one: the low end is a scaled chi-square quantile on the Satterthwaite degrees of freedom of the
estimate's case-clustered standard error, at most one per result read (a normal judge's half squared difference is a
chi-square on one), and the high end the same with one more of the largest step the estimate can take (one flip of
the scale's smallest step, in one result's share of the mean). On a rare flip that is Garwood's interval on the count
of flips, and with no disagreement at all its high end is one flip's. The share's is that interval on the sum of the
two arms' terms, their errors added in quadrature; it is not the share at each end of the arms' ranges, which covers
more often than its level and is about one and a half times as wide. ``e``'s is the Wilson interval over cases
(:func:`~threetears.evals.analysis.stats.proportion_interval`). Each range's coverage is measured in
``tests/test_simulated_judge_noise.py`` and held from both sides: at least near its level, and neither more often nor
wider than a stated bar.

Below :data:`RESCORE_MIN_RESULTS` repeated results from :data:`RESCORE_MIN_CASES` cases an arm's figures are not
available (``too_few``) and no share is read over it; where the ceiling is read for a dimension and cannot be (no
first reading passed, or one case is behind every one that did), it alone is ``not available``, ``too_few``
(``Rescore.ceiling_state``). Tolerance by scale, at that floor: about ±30% on 1 to 5, and about ±70% on pass/fail
with ``e`` near 0.05, where twenty repeats hold about two flips (``tests/test_simulated_judge_noise.py``).

Nothing here changes an interval, a verdict or a state, and nothing here is read by a decision: the figures sit in the
judge's table (``JudgedArm.rescore``, ``FamilyComparison.rescore_share``) and nowhere else.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Final

from threetears.evals.analysis.agreement import judge_key, sent_settings_key
from threetears.evals.analysis.bundle.cell_reads import CellKey, judged_values, non_faulted
from threetears.evals.analysis.bundle.schema import FamilyComparison, JudgedArm, JudgedMeasure, MultipleComparisons
from threetears.evals.analysis.checks import Rescore
from threetears.evals.analysis.reporting import ScoreRecord
from threetears.evals.analysis.stats import (
    INTERVAL_LEVEL,
    clustered_standard_error,
    contrast_cases,
    proportion_interval,
)
from threetears.evals.kernel.result_condition import ResultOutcome, classify_result
from threetears.evals.kernel.result_reading import ResultView
from threetears.evals.schema.models import SCALES, PASS_FAIL_SCORES
from threetears.evals.schema.reading_states import Reason, SampleKind, StateOf

__all__ = ["with_rescore_noise"]

#: The fewest repeated results an arm's re-score figures are read from (review C-11). Below it, or below
#: :data:`RESCORE_MIN_CASES` cases, the figures are not available.
RESCORE_MIN_RESULTS: Final = 20

#: The fewest distinct cases those repeated results must come from.
RESCORE_MIN_CASES: Final = 10

#: Why an arm's figures are absent below the floor: too few repeats to read them from.
_TOO_FEW: Final = StateOf(state="not_available", reason="too_few")


@dataclass(frozen=True)
class Repeated:
    """One repeated result, as its readings count: its case, its first reading and each repeat's.

    :ivar case: The result's test case.
    :ivar first: The first reading, as it counts.
    :ivar again: Each repeat's reading, as it counts, oldest first.
    """

    case: str
    first: float
    again: tuple[float, ...]


@dataclass(frozen=True)
class NoiseRange:
    """A figure and its range.

    :ivar point: The estimate.
    :ivar low: The range's low end.
    :ivar high: Its high end.
    """

    point: float
    low: float
    high: float


@dataclass(frozen=True)
class ArmVariance:
    """An arm's re-score variance, its range, and what the range is built from, so a share can sum two of them.

    :ivar point: The estimate.
    :ivar low: The range's low end.
    :ivar high: Its high end.
    :ivar error: The estimate's standard error, over cases, at least that of one degree of freedom per result read.
    :ivar unit: The largest step the estimate can take: one flip of the scale's smallest step, in one result's share.
    :ivar most: The most a score on the scale can vary.
    """

    point: float
    low: float
    high: float
    error: float
    unit: float
    most: float


#: The normal deviate of a two-sided range at the package's interval level.
_Z: Final = NormalDist().inv_cdf(1.0 - (1.0 - INTERVAL_LEVEL) / 2.0)


def _chi_square_quantile(dof: float, z: float) -> float:
    """The chi-square quantile on ``dof`` degrees of freedom at the normal deviate ``z`` (Wilson and Hilferty).

    Within a few percent from two degrees of freedom up; below that its lower quantiles fall short, which lowers a
    range's low end and never raises it. Zero where the cube root's base is not positive.
    """
    spread = 2.0 / (9.0 * dof)
    base = 1.0 - spread + z * math.sqrt(spread)
    return dof * base**3 if base > 0.0 else 0.0


def _gamma_range(estimate: float, error: float, unit: float, most: float) -> tuple[float, float]:
    """Fay and Feuer's interval on a non-negative weighted sum of counts, at the package's interval level.

    The low end is the estimate's lower chi-square quantile on its Satterthwaite degrees of freedom
    (``2 estimate² / error²``) over those degrees of freedom; the high end is the upper one with ``unit`` added to the
    estimate and ``unit²`` to its variance. Zero below a zero estimate; never past ``most``, and never inside the
    estimate.
    """
    low = 0.0
    if estimate > 0.0 and error > 0.0:
        dof = 2.0 * estimate * estimate / (error * error)
        low = estimate * _chi_square_quantile(dof, -_Z) / dof
    shifted = estimate + unit
    dof = 2.0 * shifted * shifted / (error * error + unit * unit)
    high = shifted * _chi_square_quantile(dof, _Z) / dof
    return min(low, estimate), max(min(high, most), estimate)


def rescore_variance(repeated: Sequence[Repeated], *, value_range: tuple[float, float]) -> ArmVariance | None:
    """The variance between two scorings of one reply, from repeats, with its range.

    Each repeated result gives the mean, over its repeats, of half the squared difference between its first reading
    and the repeat: an unbiased reading of that reply's variance. The estimate is their mean, and the range is
    :func:`_gamma_range` on it, over its case-clustered standard error, taken at least as large as one degree of
    freedom per result would make it. With no disagreement at all the range runs from zero to one flip's bound.

    :param repeated: The repeated results.
    :param value_range: The scale's inclusive bounds.
    :return: The variance and its range at :data:`~threetears.evals.analysis.stats.INTERVAL_LEVEL`, or None with no
        repeated result.
    """
    if not repeated:
        return None
    halves = [sum((entry.first - again) ** 2 for again in entry.again) / (2.0 * len(entry.again)) for entry in repeated]
    mean = math.fsum(halves) / len(halves)
    bottom, top = value_range
    most = (top - bottom) ** 2 / 4.0
    clustered = clustered_standard_error(halves, [entry.case for entry in repeated]) or 0.0
    error = max(clustered, mean * math.sqrt(2.0 / len(halves)))
    # Half a squared step of one, in the share of the mean of the result with the fewest repeats.
    unit = 0.5 / min(len(entry.again) for entry in repeated) / len(halves)
    low, high = _gamma_range(mean, error, unit, most)
    return ArmVariance(mean, low, high, error, unit, most)


def fail_given_pass(repeated: Sequence[Repeated], *, passed: float) -> NoiseRange | None:
    """``e``: how often a repeat fails a result whose first reading passed, with the Wilson interval over cases.

    :param repeated: The repeated results on a pass/fail dimension, as they count.
    :param passed: The reading that is a pass.
    :return: ``e`` and its range, or None where no first reading passed.
    """
    outcomes = [again != passed for entry in repeated if entry.first == passed for again in entry.again]
    cases = [entry.case for entry in repeated if entry.first == passed for _ in entry.again]
    if not outcomes:
        return None
    rate = sum(outcomes) / len(outcomes)
    interval = proportion_interval(outcomes, cases)
    if interval is None:
        # One case behind every repeat: no spread between cases to bound it with.
        return None
    return NoiseRange(rate, interval[0], interval[1])


def pass_k_ceiling(e: float, k: int) -> float:
    """The ceiling a judge that fails a passing reply on ``e`` of its readings puts on pass^k: ``(1 − e)^(k − 1)``.

    :param e: The judge's fail rate on a reply whose first reading passed.
    :param k: The depth pass^k is read at.
    :return: The ceiling.
    """
    return (1.0 - e) ** (k - 1)


def rescore_of(
    repeated: Sequence[Repeated],
    *,
    value_range: tuple[float, float],
    k: int,
    passed: float | None,
) -> tuple[Rescore, ArmVariance | None]:
    """An arm's re-score figures: its variance, and on a pass/fail dimension counting toward pass, ``e`` and the
    ceiling on pass^k at ``k``.

    :param repeated: The arm's repeated results on the dimension, as they count.
    :param value_range: The scale's inclusive bounds.
    :param k: The depth the ceiling is read at.
    :param passed: The reading that is a pass, on a pass/fail dimension that counts toward a trial's pass; None
        elsewhere, where no ceiling is read.
    :return: The figures, ``not available`` with ``too_few`` below the floor (and the ceiling alone ``not available``,
        ``too_few``, where it is read and no first reading passed, or one case is behind every repeat that did); and
        the variance with its range, for the share, or None below the floor.
    """
    results, cases = len(repeated), len({entry.case for entry in repeated})
    if results < RESCORE_MIN_RESULTS or cases < RESCORE_MIN_CASES:
        return Rescore(state=_TOO_FEW, repeated_results=results, repeated_cases=cases), None
    variance = rescore_variance(repeated, value_range=value_range)
    assert variance is not None  # at least the floor's results were read
    e = fail_given_pass(repeated, passed=passed) if passed is not None else None
    return (
        Rescore(
            variance=variance.point,
            variance_interval=(variance.low, variance.high),
            repeated_results=results,
            repeated_cases=cases,
            fail_given_pass=None if e is None else e.point,
            fail_given_pass_interval=None if e is None else (e.low, e.high),
            ceiling_low=None if e is None else pass_k_ceiling(e.high, k),
            ceiling_high=None if e is None else pass_k_ceiling(e.low, k),
            k=None if e is None else k,
            # Read for this dimension and not readable: no first reading passed, or one case is behind every repeat.
            ceiling_state=_TOO_FEW if passed is not None and e is None else None,
        ),
        variance,
    )


@dataclass(frozen=True)
class SideNoise:
    """One side of a contrast, as its share reads it: the arm's re-score variance and its counted scores per case.

    :ivar variance: The arm's re-score variance and its range.
    :ivar depth: Counted scores per case on the dimension.
    """

    variance: ArmVariance
    depth: Mapping[str, int]


def judge_share(
    control: SideNoise, candidate: SideNoise, cases: Sequence[str], differences: Sequence[float]
) -> NoiseRange | None:
    """The share of a paired contrast's variance that is its judge's re-score noise, with its range.

    :param control: The control's side.
    :param candidate: The candidate's side.
    :param cases: The cases the contrast paired.
    :param differences: Each paired case's candidate mean minus its control mean, aligned with ``cases``.
    :return: The share and its range, :func:`_gamma_range` on the sum of the two arms' terms (each arm's variance over
        its mean ``1/k``), their standard errors added in quadrature; None below two cases, or where the differences
        have no spread.
    """
    if len(differences) < 2:
        return None
    mean = math.fsum(differences) / len(differences)
    spread = math.fsum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
    if spread <= 0.0:
        return None
    terms = [
        (side.variance, math.fsum(1.0 / side.depth[case] for case in cases) / len(cases))
        for side in (control, candidate)
    ]
    estimate = math.fsum(variance.point * weight for variance, weight in terms)
    error = math.sqrt(math.fsum((variance.error * weight) ** 2 for variance, weight in terms))
    unit = max(variance.unit * weight for variance, weight in terms)
    low, high = _gamma_range(estimate, error, unit, math.fsum(variance.most * weight for variance, weight in terms))
    return NoiseRange(estimate / spread, low / spread, high / spread)


# --- read off the bundle's results ----------------------------------------------------------------


def _repeated_of(
    members: Sequence[ResultView], dimension: str
) -> tuple[list[Repeated], list[tuple[str, SampleKind | None]]] | None:
    """Each of a cell's results repeated on ``dimension`` by the judge that scored it, as its readings count.

    A repeat is read only where its judge (served model, config, temperature and request settings) is the one the
    first reading recorded and the one the result holds the dimension's score from: a repeat by another judge, or of
    a score since re-judged by another, measures a different judge. A failed repeat, and one that could not tell, has
    no reading to compare. A harness-faulted result is in no measure; a candidate failure counts the scale's floor on
    both readings. Only a repeat of a random draw is read (``sample``): results named by id, or a repeat stored before
    its draw was recorded, may not stand for the arm's results, so they are passed over and said to be.

    :return: The results repeated in a random draw; each other result whose only repeats by its judge were not, by
        case and how it was chosen (None where unrecorded). None where no result in the cell recorded a repeat of the
        dimension at all.
    """
    found = False
    repeated: list[Repeated] = []
    passed_over: list[tuple[str, SampleKind | None]] = []
    for result in members:
        outcome = classify_result(result)
        current = judge_key(result, dimension)
        first: float | None = None
        again: list[float] = []
        chosen: set[SampleKind | None] = set()
        for repeat in result.judge_repeats:
            for entry in repeat.scores:
                if entry.dim != dimension:
                    continue
                found = True
                score = entry.repeat
                if score is None or outcome is ResultOutcome.INFRA_EXCLUDE or current is None:
                    continue
                key = (
                    entry.first_served_model,
                    entry.first_judge_config_id,
                    entry.first_judge_temperature,
                    sent_settings_key(entry.first_judge_sent_settings),
                )
                if (
                    key
                    != (
                        score.served_model,
                        repeat.judge_config_ids.get(dimension),
                        score.judge_temperature,
                        sent_settings_key(score.judge_sent_settings),
                    )
                    or key
                    != (
                        current.judge_model,
                        current.judge_config_id,
                        current.judge_temperature,
                        current.judge_sent_settings,
                    )
                    or entry.scale != current.scale
                ):
                    continue
                if repeat.sample != "random":
                    chosen.add(repeat.sample)
                    continue
                floor = float(SCALES[entry.scale].scores[0])
                failed = outcome is ResultOutcome.CANDIDATE_FAIL
                first = floor if failed else float(entry.first_score)
                again.append(floor if failed else float(score.score))
        if first is not None:
            repeated.append(Repeated(result.test_case_id, first, tuple(again)))
        elif chosen:
            passed_over.append((result.test_case_id, None if chosen == {None} else "selected"))
    return (repeated, passed_over) if found else None


def _depths(members: Sequence[ResultView], rows: Mapping[str, list[ScoreRecord]]) -> dict[str, dict[str, int]]:
    """Counted scores per case, per dimension, in one cell: the ``k`` each case's mean in a contrast is over.

    Read as a contrast reads its per-case values: the results the harness did not fault, their valued judged rows.
    """
    depth: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for result in non_faulted(list(members)):
        for dimension, record in judged_values(rows.get(result.id, [])):
            if record.value is not None:
                depth[dimension][result.test_case_id] += 1
    return depth


def _case_means(
    members: Sequence[ResultView], rows: Mapping[str, list[ScoreRecord]], dimension: str
) -> dict[str, float]:
    """Each case's mean counted score on ``dimension`` in one cell, as a contrast reads it."""
    scores: dict[str, list[float]] = defaultdict(list)
    for result in non_faulted(list(members)):
        for name, record in judged_values(rows.get(result.id, [])):
            if name == dimension and record.value is not None:
                scores[result.test_case_id].append(float(record.value))
    return {case: sum(values) / len(values) for case, values in scores.items()}


def with_rescore_noise(
    measures: list[JudgedMeasure],
    comparisons: MultipleComparisons,
    results_by_cell: Mapping[CellKey, list[ResultView]],
    records: Sequence[ScoreRecord],
) -> tuple[list[JudgedMeasure], MultipleComparisons]:
    """The judge's table with its re-score noise filled: each judged arm's ``rescore``, each paired judged contrast's
    ``rescore_share``.

    An arm none of whose results recorded a repeat of the dimension carries no ``rescore``, as one nobody rated
    carries no prediction-powered estimate. One whose repeats by its judge were none of them a random draw carries
    ``not available``, ``not_drawn_at_random`` (named by id) or ``not_recorded`` (stored before the draw was). A share is read only over a paired contrast whose two arms both have their
    figures; elsewhere it stays unset.

    :param measures: The bundle's judged measures.
    :param comparisons: The aspect's corrected families.
    :param results_by_cell: Each cell's results, the grouping both were read over.
    :param records: The score projection rows the judged values were read off.
    :return: The judged measures and the families, each with its figures filled.
    :raises ValueError: A contrast's paired cases here are not the ones its test read.
    """
    rows: dict[str, list[ScoreRecord]] = defaultdict(list)
    for record in records:
        rows[record.result_id].append(record)
    depths = {key: _depths(members, rows) for key, members in results_by_cell.items()}
    sides: dict[tuple[str, CellKey], SideNoise] = {}
    filled = []
    for measure in measures:
        if measure.value_range is None:
            filled.append(measure)
            continue
        arms: list[JudgedArm] = []
        for arm in measure.arms:
            key = (arm.variant_key, arm.apparatus_class_id)
            found = _repeated_of(results_by_cell.get(key, []), measure.name)
            if found is None:
                arms.append(arm)
                continue
            repeated, passed_over = found
            if not repeated and passed_over:
                arms.append(arm.model_copy(update={"rescore": _not_drawn(passed_over)}))
                continue
            depth = depths.get(key, {}).get(measure.name, {})
            # A pass rate (a pass/fail dimension's ratio scale) that counts toward a trial's pass carries the ceiling.
            counts_toward_pass = measure.scale == "ratio" and measure.axis == "capability"
            passed = float(PASS_FAIL_SCORES["pass"]) if counts_toward_pass else None
            rescore, variance = rescore_of(
                repeated, value_range=measure.value_range, k=max(depth.values(), default=1), passed=passed
            )
            if variance is not None:
                sides[(measure.name, key)] = SideNoise(variance, depth)
            arms.append(arm.model_copy(update={"rescore": rescore}))
        filled.append(measure.model_copy(update={"arms": arms}))
    families = []
    for family in comparisons.families:
        compared: list[FamilyComparison] = []
        for comparison in family.comparisons:
            share = _share_of(comparison, sides, results_by_cell, rows)
            compared.append(
                comparison
                if share is None
                else comparison.model_copy(update={"rescore_share": (share.low, share.high)})
            )
        families.append(family.model_copy(update={"comparisons": compared}))
    return filled, comparisons.model_copy(update={"families": families})


def _not_drawn(passed_over: Sequence[tuple[str, SampleKind | None]]) -> Rescore:
    """An arm whose repeats by its judge are none of them a random draw: ``not available``, and why.

    ``not_drawn_at_random`` where any was chosen by id, else ``not_recorded``: a draw nobody recorded supports no claim.
    """
    reason: Reason = "not_recorded" if all(how is None for _, how in passed_over) else "not_drawn_at_random"
    return Rescore(
        state=StateOf(state="not_available", reason=reason),
        repeated_results=len(passed_over),
        repeated_cases=len({case for case, _ in passed_over}),
    )


def _share_of(
    comparison: FamilyComparison,
    sides: Mapping[tuple[str, CellKey], SideNoise],
    results_by_cell: Mapping[CellKey, list[ResultView]],
    rows: Mapping[str, list[ScoreRecord]],
) -> NoiseRange | None:
    """A judged contrast's share, or None where it is not a paired judged contrast with both arms' figures."""
    if comparison.reading != "judged" or comparison.test != "paired":
        return None
    control_key = (comparison.control.variant_key, comparison.control.apparatus_class_id)
    candidate_key = (comparison.contrast.variant_key, comparison.contrast.apparatus_class_id)
    control, candidate = sides.get((comparison.name, control_key)), sides.get((comparison.name, candidate_key))
    if control is None or candidate is None:
        return None
    control_means = _case_means(results_by_cell[control_key], rows, comparison.name)
    candidate_means = _case_means(results_by_cell[candidate_key], rows, comparison.name)
    cases, _, paired = contrast_cases(control_means, candidate_means)
    if not paired or len(cases) != comparison.control.n_cases:
        raise ValueError(
            f"the contrast of {candidate_key} against {control_key} on {comparison.name!r} paired "
            f"{comparison.control.n_cases} cases, and its re-score share read {len(cases)}"
        )
    differences = [candidate_means[case] - control_means[case] for case in cases]
    return judge_share(control, candidate, cases, differences)
