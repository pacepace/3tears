"""Re-score noise: a judge checked against itself, read one step down in the judge's own table.

A judge asked again about a reply it already scored can score it differently. That is re-score noise, and only that:
a judge that repeats itself exactly (a deterministic or cached one) reads no noise and may still be wrong, so the
figures here never say how good a judge is.

Three figures, each filled where the bundle assembles the judge's table (:func:`with_rescore_noise`):

- **Re-score variance**, per arm: half the mean squared difference between a counted score and its repeat, which is
  the variance between two scorings of one reply. Repeats are read as they count (a candidate failure counts the
  scale's floor on both readings, so it moves nothing), and only a repeat by the very judge that scored the arm.
- **The judge's share of a contrast's width**, per paired judged contrast:
  ``(σ²_control · mean(1/k_control) + σ²_candidate · mean(1/k_candidate)) / variance of the per-case differences``,
  where ``k`` is a case's counted scores on that side. It exists at k = 1, since the repeats, not the spread within a
  case, measure the judge. Each arm's own variance is read for it: two arms' answers, or their judges, can differ, and
  one arm's repeats are not taken to speak for the other's.
- **The judge's ceiling on pass^k**, per arm of a pass/fail dimension that counts toward a trial's pass: ``e`` is the
  share of repeats that fail among results whose first reading passed, and the ceiling is ``(1 − e)^(k − 1)``.

Each is a range. The variance's is a scaled chi-square interval (Satterthwaite) on the case-clustered mean of the
half squared differences, which is Garwood's interval on a count of flips where disagreements are rare; with no
disagreement at all it is bounded above as if the next one were the smallest the scale holds. ``e``'s is the Wilson
interval over cases (:func:`~threetears.evals.analysis.stats.proportion_interval`). The share's is the share at each
end of the variances' ranges: the contrast's own width is what it is, and the uncertain part is the judge's.

Below :data:`RESCORE_MIN_RESULTS` repeated results from :data:`RESCORE_MIN_CASES` cases an arm's figures are not
available (``too_few``) and no share is read over it. Tolerance by scale, at that floor: about ±30% on 1 to 5, and
about ±70% on pass/fail with ``e`` near 0.05, where twenty repeats hold about two flips
(``tests/test_simulated_judge_noise.py``).

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
from threetears.evals.schema.reading_states import StateOf

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


def _chi_square_quantile(dof: float, z: float) -> float:
    """The chi-square quantile on ``dof`` degrees of freedom at the normal deviate ``z`` (Wilson and Hilferty).

    Within a few percent from two degrees of freedom up; below that its lower quantiles fall short, which widens an
    interval's upper end and never narrows it. Zero where the cube root's base is not positive.
    """
    spread = 2.0 / (9.0 * dof)
    base = 1.0 - spread + z * math.sqrt(spread)
    return dof * base**3 if base > 0.0 else 0.0


def rescore_variance(repeated: Sequence[Repeated], *, value_range: tuple[float, float]) -> NoiseRange | None:
    """The variance between two scorings of one reply, from repeats, with its range.

    Each repeated result gives the mean, over its repeats, of half the squared difference between its first reading
    and the repeat: an unbiased reading of that reply's variance. The estimate is their mean, and the range is a
    scaled chi-square interval on it, its degrees of freedom matched to the case-clustered standard error
    (Satterthwaite): ``2 m² / se²``, between one and the results read. With no disagreement at all the range runs
    from zero to the bound one flip of the scale's smallest step would give, read as a count (Garwood at zero). The
    high end never passes the most a score on the scale can vary.

    :param repeated: The repeated results.
    :ptype repeated: Sequence[Repeated]
    :param value_range: The scale's inclusive bounds.
    :ptype value_range: tuple[float, float]
    :return: The variance and its range at :data:`~threetears.evals.analysis.stats.INTERVAL_LEVEL`, or None with no
        repeated result.
    :rtype: NoiseRange | None
    """
    if not repeated:
        return None
    halves = [sum((entry.first - again) ** 2 for again in entry.again) / (2.0 * len(entry.again)) for entry in repeated]
    cases = [entry.case for entry in repeated]
    mean = math.fsum(halves) / len(halves)
    bottom, top = value_range
    most = (top - bottom) ** 2 / 4.0
    z = NormalDist().inv_cdf(1.0 - (1.0 - INTERVAL_LEVEL) / 2.0)
    if mean == 0.0:
        # Nothing moved: read as a count of zero flips, each worth half a squared step of one, over the results.
        return NoiseRange(0.0, 0.0, min(most, 0.5 * _chi_square_quantile(2.0, z) / 2.0 / len(halves)))
    error = clustered_standard_error(halves, cases)
    dof = float(len(halves)) if not error else min(max(2.0 * mean * mean / (error * error), 1.0), float(len(halves)))
    upper_quantile, lower_quantile = _chi_square_quantile(dof, z), _chi_square_quantile(dof, -z)
    low = mean * dof / upper_quantile
    high = most if lower_quantile <= 0.0 else min(most, mean * dof / lower_quantile)
    return NoiseRange(mean, min(low, mean), max(high, mean))


def fail_given_pass(repeated: Sequence[Repeated], *, passed: float) -> NoiseRange | None:
    """``e``: how often a repeat fails a result whose first reading passed, with the Wilson interval over cases.

    :param repeated: The repeated results on a pass/fail dimension, as they count.
    :ptype repeated: Sequence[Repeated]
    :param passed: The reading that is a pass.
    :ptype passed: float
    :return: ``e`` and its range, or None where no first reading passed.
    :rtype: NoiseRange | None
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
    :ptype e: float
    :param k: The depth pass^k is read at.
    :ptype k: int
    :return: The ceiling.
    :rtype: float
    """
    return (1.0 - e) ** (k - 1)


def rescore_of(
    repeated: Sequence[Repeated],
    *,
    value_range: tuple[float, float],
    k: int,
    passed: float | None,
) -> tuple[Rescore, NoiseRange | None]:
    """An arm's re-score figures: its variance, and on a pass/fail dimension counting toward pass, ``e`` and the
    ceiling on pass^k at ``k``.

    :param repeated: The arm's repeated results on the dimension, as they count.
    :ptype repeated: Sequence[Repeated]
    :param value_range: The scale's inclusive bounds.
    :ptype value_range: tuple[float, float]
    :param k: The depth the ceiling is read at.
    :ptype k: int
    :param passed: The reading that is a pass, on a pass/fail dimension that counts toward a trial's pass; None
        elsewhere, where no ceiling is read.
    :ptype passed: float | None
    :return: The figures, ``not available`` with ``too_few`` below the floor; and the variance with its range, for
        the share, or None below the floor.
    :rtype: tuple[Rescore, NoiseRange | None]
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
            repeated_results=results,
            repeated_cases=cases,
            fail_given_pass=None if e is None else e.point,
            fail_given_pass_interval=None if e is None else (e.low, e.high),
            ceiling_low=None if e is None else pass_k_ceiling(e.high, k),
            ceiling_high=None if e is None else pass_k_ceiling(e.low, k),
            k=None if e is None else k,
        ),
        variance,
    )


@dataclass(frozen=True)
class SideNoise:
    """One side of a contrast, as its share reads it: the arm's re-score variance and its counted scores per case.

    :ivar variance: The arm's re-score variance and its range.
    :ivar depth: Counted scores per case on the dimension.
    """

    variance: NoiseRange
    depth: Mapping[str, int]


def judge_share(
    control: SideNoise, candidate: SideNoise, cases: Sequence[str], differences: Sequence[float]
) -> NoiseRange | None:
    """The share of a paired contrast's variance that is its judge's re-score noise, with its range.

    :param control: The control's side.
    :ptype control: SideNoise
    :param candidate: The candidate's side.
    :ptype candidate: SideNoise
    :param cases: The cases the contrast paired.
    :ptype cases: Sequence[str]
    :param differences: Each paired case's candidate mean minus its control mean, aligned with ``cases``.
    :ptype differences: Sequence[float]
    :return: The share and its range; None below two cases, or where the differences have no spread.
    :rtype: NoiseRange | None
    """
    if len(differences) < 2:
        return None
    mean = math.fsum(differences) / len(differences)
    spread = math.fsum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
    if spread <= 0.0:
        return None
    control_weight = math.fsum(1.0 / control.depth[case] for case in cases) / len(cases)
    candidate_weight = math.fsum(1.0 / candidate.depth[case] for case in cases) / len(cases)

    def share(control_variance: float, candidate_variance: float) -> float:
        return (control_variance * control_weight + candidate_variance * candidate_weight) / spread

    return NoiseRange(
        share(control.variance.point, candidate.variance.point),
        share(control.variance.low, candidate.variance.low),
        share(control.variance.high, candidate.variance.high),
    )


# --- read off the bundle's results ----------------------------------------------------------------


def _repeated_of(members: Sequence[ResultView], dimension: str) -> list[Repeated] | None:
    """Each of a cell's results repeated on ``dimension`` by the judge that scored it, as its readings count.

    A repeat is read only where its judge (served model, config, temperature and request settings) is the one the
    first reading recorded and the one the result holds the dimension's score from: a repeat by another judge, or of
    a score since re-judged by another, measures a different judge. A failed repeat, and one that could not tell, has
    no reading to compare. A harness-faulted result is in no measure; a candidate failure counts the scale's floor on
    both readings.

    :return: The repeated results, or None where no result in the cell recorded a repeat of the dimension at all.
    """
    found = False
    repeated: list[Repeated] = []
    for result in members:
        outcome = classify_result(result)
        current = judge_key(result, dimension)
        first: float | None = None
        again: list[float] = []
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
                floor = float(SCALES[entry.scale].scores[0])
                failed = outcome is ResultOutcome.CANDIDATE_FAIL
                first = floor if failed else float(entry.first_score)
                again.append(floor if failed else float(score.score))
        if first is not None:
            repeated.append(Repeated(result.test_case_id, first, tuple(again)))
    return repeated if found else None


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
    carries no prediction-powered estimate. A share is read only over a paired contrast whose two arms both have their
    figures; elsewhere it stays unset.

    :param measures: The bundle's judged measures.
    :ptype measures: list[JudgedMeasure]
    :param comparisons: The aspect's corrected families.
    :ptype comparisons: MultipleComparisons
    :param results_by_cell: Each cell's results, the grouping both were read over.
    :ptype results_by_cell: Mapping[CellKey, list[ResultView]]
    :param records: The score projection rows the judged values were read off.
    :ptype records: Sequence[ScoreRecord]
    :return: The judged measures and the families, each with its figures filled.
    :rtype: tuple[list[JudgedMeasure], MultipleComparisons]
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
            repeated = _repeated_of(results_by_cell.get(key, []), measure.name)
            if repeated is None:
                arms.append(arm)
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
