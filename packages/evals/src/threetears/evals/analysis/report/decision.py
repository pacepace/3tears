"""The floors on an aspect, each read on one candidate: broken, held or unclear, and why.

A floor (:class:`~threetears.evals.kernel.declaration.Floor`) is a typed clause on one aspect:
``RateDrop(max_points=10)`` (no pass rate may drop by more than 10 points against the control) or ``CostLower()`` (the candidate must cost less per
reply). **A floor only warns.** It never decides: code reports, the writer recommends and a person decides, so a broken
floor is the first line of the page and the call that adopts the candidate anyway records it as overridden.
:func:`evaluate_floor` reads one floor on one candidate (:func:`read_floor` with the per-rate detail the page prints),
and :func:`aspect_pass_rate` reads the aspect's overall pass rate on a candidate, which the campaign's readings compare.

**Each floor is read at its nominal level, with no correction.** A floor is read intersection-union: it is broken where
any rate it reads is shown past it, so each rate is read on its own two-sided bounded interval at
:data:`~threetears.evals.analysis.stats.INTERVAL_LEVEL` (each end a 2.5% one-sided claim), computed again from per-case
values, never at the family's Bonferroni level: a warning that fires less often for every floor added is no warning.
The overall pass rate's and the cost per reply's values are the bundle's ``floor_contrasts``, carried whatever the
family compares; a check's are its family comparison's (``case_values``). Each rate a rate floor reads (the overall pass
rate and each check's, or the one reading it names) is in one of three states, and so is cost, on its own interval:

- **Broken.** Every arm has at least the case floor of usable cases
  (:data:`~threetears.evals.analysis.report.verdict.MIN_CASES`), and the bounded interval on the candidate minus the
  control lies wholly past the margin: the drop nearest zero it allows is still more than the margin, as is the drop
  with every lost trial given its best outcome. Only this reads as a regression.
- **Held.** Every arm is at the case floor, and the bounded interval bounds the drop at or below the margin, as does the
  bound over lost trials. A non-inferiority claim never reads the t interval the page's contrasts print: on a rate's
  coarse per-case shares it holds too often (``docs/measuring-soundly.md``).
- **Unclear**, with the reason in plain words. That includes a drop past the margin on the point estimate alone, whose
  interval reaches back within it: its reason says the point estimate is past and gives the interval, so it never reads
  as a measured regression. A rate both arms pass on every trial, or on none, and that no interval reads either way,
  cannot separate them, and its reason says so with the counts, before any other reason.
- **A rate floor** is broken when one rate it reads is broken, held when every one is held, and otherwise unclear.
- **Cost.** Read as a rate is, on the t interval on the per-case cost difference at the nominal level (cost declares no
  range, so no bounded interval exists): broken where every arm is at the case floor and the interval lies wholly above
  zero, held where it lies wholly below, unclear otherwise. Cost compared on no case, or alike on every case, is
  unclear and says which.

**Each reason says the counts and the margin in words, naming the arms, and only the rate it is about.** A broken rate
reads ``v2 passed 14 of 60 to v1's 29, a drop that exceeds the margin of 10 points on its whole interval (at least
12.3)``; one past the margin on its point estimate alone ``v2's drop is past 10 points on the point estimate alone, not
on its interval (from a gain of 35.5 to a drop of 92.7)``; one whose interval reaches past the margin ``v2's drop is not
shown to stay within 10 points (it could be as much as 53.6)``; one both arms pass on no trial ``the pass rate cannot
separate the arms (both 0 of 18)``. A rate past the floor's first is named in its reason.

Every sentence here names the arms by the names the page gives them, never by their role in the comparison.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import reduce
from typing import TYPE_CHECKING, Literal

from threetears.evals.analysis.bundle.schema import (
    FLOOR_COST_MEASURE,
    TRIAL_PASS_NAME,
    AnalysisContextBundle,
    FamilyComparison,
    FloorContrast,
)
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.stats import (
    INTERVAL_LEVEL,
    PROJECTION_CASE_CAP,
    UNPAIRED_PROJECTION_CAP,
    UNPAIRED_PROJECTION_RESOLUTION,
    SMALL_N_BAND_FLOOR,
    bounded_cases_to_clear,
    bounded_difference_interval,
    cases_to_clear,
    difference_interval,
)
from threetears.evals.analysis.style import code, humanize, measure_name
from threetears.evals.kernel.analysis_measures import MeasureSummary
from threetears.evals.kernel.declaration import CostLower, DecisionClause, Floor, RateDrop
from threetears.evals.kernel.metrics import ACCURACY_MEASURE, MATCH_MEASURE, goal_check_of, rule_rates

if TYPE_CHECKING:
    from threetears.evals.analysis.campaign_readings import FloorStatus

#: Where one floor stands on one candidate, and where each rate it reads does: the values of
#: :data:`~threetears.evals.analysis.campaign_readings.FloorState`, spelled here because that module reads this one.
FloorOutcome = Literal["broken", "held", "unclear"]


#: The fewest usable cases on every arm before a rate floor can be read broken or held: the page's case floor.
_FLOOR = SMALL_N_BAND_FLOOR

#: The share range every rate a floor reads lies in, which its bounded interval is read on.
_RATE_RANGE = (0.0, 1.0)


@dataclass(frozen=True)
class RateReading:
    """One pass rate on both arms: passed of measured, how many trials each could still add, and its interval.

    ``interval`` is the bounded interval on the arm minus the control, as a share (0 to 1) with higher better, at
    ``confidence`` (the nominal :data:`~threetears.evals.analysis.stats.INTERVAL_LEVEL`), computed from ``samples``;
    None where the bundle carries no per-case values for the rate. ``measure`` is the measure's key, None for the
    trials' own pass rate. ``samples`` are the control's and the arm's per-case values, higher better, which a
    projection reads again at more cases (:func:`project_rate`); ``paired`` whether they are one-to-one; ``rig`` the
    test setup (apparatus class) they were read under, None with no per-case values. ``trials`` are each side's usable
    trials per case, aligned with ``samples``, and ``passes`` each side's trials passed per case: carried on the overall
    pass rate, None elsewhere. Each case's share
    weighs the same in the interval whatever its trials, while the passed-of-measured counts pool every trial.
    """

    name: str
    control_passed: int
    control_n: int
    candidate_passed: int
    candidate_n: int
    control_remaining: int = 0
    candidate_remaining: int = 0
    interval: tuple[float, float] | None = None
    measure: str | None = None
    samples: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    paired: bool = True
    confidence: float | None = None
    rig: str | None = None
    trials: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    passes: tuple[tuple[int, ...], tuple[int, ...]] | None = None

    def drop_bounds(self) -> tuple[float, float]:
        """The arm's drop from the control in points, at best and at worst for the arm."""

        def bounds(passed: int, n: int, remaining: int) -> tuple[float, float]:
            total = n + remaining
            return (passed / total, (passed + remaining) / total) if total else (0.0, 1.0)

        control_low, control_high = bounds(self.control_passed, self.control_n, self.control_remaining)
        candidate_low, candidate_high = bounds(self.candidate_passed, self.candidate_n, self.candidate_remaining)
        return 100 * (control_low - candidate_high), 100 * (control_high - candidate_low)

    def worst_drop(self) -> float | None:
        """The largest drop in points the interval allows; None without an interval."""
        return None if self.interval is None else -100 * self.interval[0]

    def least_drop(self) -> float | None:
        """The smallest drop in points the interval allows, negative where it allows a gain; None without an
        interval."""
        return None if self.interval is None else -100 * self.interval[1]

    def point_drop(self) -> float:
        """The arm's drop from the control in points on what each measured: the point estimate, 0 where an arm
        measured nothing."""
        if not self.control_n or not self.candidate_n:
            return 0.0
        return 100 * (self.control_passed / self.control_n - self.candidate_passed / self.candidate_n)

    def degenerate(self) -> bool:
        """Whether both arms passed none of what they measured, or both passed all of it: a rate that cannot separate
        them."""
        if not self.control_n or not self.candidate_n:
            return False
        return (self.control_passed, self.candidate_passed) in ((0, 0), (self.control_n, self.candidate_n))

    def answers(self, reading: str) -> bool:
        """Whether this is the rate a floor naming ``reading`` reads: a check's by its verbatim text, any other
        measure's by its key."""
        return self.measure is not None and reading in (self.measure, goal_check_of(self.measure))

    def reading(self) -> str | None:
        """The rate as a floor's clause names it: a check by its verbatim text, any other measure by its key; None for
        the trials' own pass rate, which no clause names."""
        if self.measure is None:
            return None
        return goal_check_of(self.measure) or self.measure


@dataclass(frozen=True)
class FloorReadings:
    """What a floor is read against for one candidate.

    ``control_cases`` and ``candidate_cases`` are each arm's usable cases, None when not counted; ``cost_interval``
    is the t interval on the arm's cost per reply minus the control's at ``confidence``, computed from the per-case
    costs the bundle carries (``floor_contrasts``), uncorrected; None when none exists. ``cost_compared`` says whether
    any per-case costs were carried for the pair at all, so an absent interval reads as not compared or as no spread.
    """

    rates: tuple[RateReading, ...]
    control_cost: float | None = None
    candidate_cost: float | None = None
    control_cases: int | None = None
    candidate_cases: int | None = None
    cost_interval: tuple[float, float] | None = None
    cost_compared: bool = False
    confidence: float = INTERVAL_LEVEL

    def fewest_cases(self) -> int | None:
        """The fewer usable cases of the two arms; None when either was not counted."""
        if self.control_cases is None or self.candidate_cases is None:
            return None
        return min(self.control_cases, self.candidate_cases)

    def read_by(self, clause: RateDrop) -> tuple[RateReading, ...]:
        """The rates ``clause`` reads: every one, or the one it names."""
        return self.rates if clause.reading is None else tuple(r for r in self.rates if r.answers(clause.reading))


@dataclass(frozen=True)
class RateVerdict:
    """One rate a rate floor reads, read on its own: ``broken`` where it is shown past the floor, ``held`` where it is
    shown within it, ``unclear`` otherwise, with the reason in plain words about this rate alone."""

    rate: RateReading
    outcome: FloorOutcome
    reason: str


@dataclass(frozen=True)
class FloorVerdict:
    """One floor read on one candidate, with the reason in plain words and the floor as a sentence naming the arm.

    ``rate`` is the rate that decided a rate floor: the first broken, else the unclear one nearest a regression, else
    the one whose drop the interval allows most; None for a cost floor. ``rates`` is each rate it read, with its own
    outcome.
    """

    floor: Floor
    outcome: FloorOutcome
    reason: str
    statement: str
    rate: RateReading | None = None
    rates: tuple[RateVerdict, ...] = ()
    readings: FloorReadings | None = None

    def interval(self) -> tuple[float, float] | None:
        """The interval the outcome was read on, in the floor's units, the candidate minus the control: points of
        pass rate for a rate floor, dollars per reply for a cost floor. None when none was read."""
        if isinstance(self.floor.clause, CostLower):
            return self.readings.cost_interval if self.readings is not None else None
        if self.rate is None or self.rate.interval is None:
            return None
        low, high = self.rate.interval
        return 100 * low, 100 * high


def _reading_words(reading: str, readings: FloorReadings | None) -> str:
    """A reading a floor names, as the page names it: the rate's own name where the arms carry it, else a key in
    words, else a check's text as code."""
    rates = readings.rates if readings is not None else ()
    if (rate := next((rate for rate in rates if rate.answers(reading)), None)) is not None:
        return rate.name
    return humanize("dimension", reading) if reading.isidentifier() else code(reading)


def floor_statement(clause: DecisionClause, *, candidate: str, readings: FloorReadings | None = None) -> str:
    """One floor as the page says it, naming the arm: ``no pass rate drops by more than 2 points``,
    ``no tone rate drops by more than 2 points``, ``v2 costs less per reply``.

    :param clause: The floor's clause.
    :ptype clause: DecisionClause
    :param candidate: The arm it is read on, by name.
    :ptype candidate: str
    :param readings: What the floor is read against, whose rate names a named reading is said by; None to say it
        from its key.
    :ptype readings: FloorReadings | None
    :return: The floor, lower case, with no closing stop.
    :rtype: str
    """
    if isinstance(clause, CostLower):
        return f"{candidate} costs less per reply"
    which = "pass" if clause.reading is None else _reading_words(clause.reading, readings)
    return f"no {which} rate drops by more than {format_number(clause.max_points)} points"


def read_floor(floor: Floor, readings: FloorReadings, *, control: str, candidate: str) -> FloorVerdict:
    """Read ``floor`` on one candidate, rate by rate.

    :param floor: The floor.
    :ptype floor: Floor
    :param readings: The control's and the candidate's figures (:func:`floor_readings`).
    :ptype readings: FloorReadings
    :param control: The control, by the name the page gives it, which a reason says.
    :ptype control: str
    :param candidate: The candidate, likewise.
    :ptype candidate: str
    :return: The floor's outcome, with each rate's and reasons naming the arms.
    :rtype: FloorVerdict
    """
    clause = floor.clause
    statement = floor_statement(clause, candidate=candidate, readings=readings)
    if isinstance(clause, CostLower):
        outcome, reason = _cost_floor(readings, control, candidate)
        return FloorVerdict(floor, outcome, reason, statement, readings=readings)
    outcome, reason, rate, rates = _rate_floor(clause, readings, control, candidate)
    return FloorVerdict(floor, outcome, reason, statement, rate=rate, rates=rates, readings=readings)


def evaluate_floor(
    floor: Floor, readings: FloorReadings, *, arm: str, arm_label: str, control_label: str
) -> FloorStatus:
    """Whether one candidate breaks, holds or leaves unclear one floor: broken exactly where the floor's clause, read
    as a rule, would settle keep.

    Read at the nominal level with no correction (the module's rules): each rate's bounded interval is computed again
    from the family comparison's per-case values at :data:`~threetears.evals.analysis.stats.INTERVAL_LEVEL`.

    :param floor: The floor.
    :ptype floor: Floor
    :param readings: The control's and the candidate's figures on the floor's aspect (:func:`floor_readings`).
    :ptype readings: FloorReadings
    :param arm: The candidate's variant key.
    :ptype arm: str
    :param arm_label: What a reader calls the candidate, which the reason names it by.
    :ptype arm_label: str
    :param control_label: What a reader calls the control, likewise.
    :ptype control_label: str
    :return: The floor's status on the candidate, its interval in the floor's units (points of pass rate, or dollars
        per reply), the candidate minus the control.
    :rtype: FloorStatus
    """
    from threetears.evals.analysis.campaign_readings import FloorStatus  # this module is read by that one

    verdict = read_floor(floor, readings, control=control_label, candidate=arm_label)
    margin = floor.clause.max_points if isinstance(floor.clause, RateDrop) else None
    worst = verdict.rate.worst_drop() if margin is not None and verdict.rate is not None else None
    return FloorStatus(
        margin=margin,
        worst_drop=worst,
        short_by=(
            worst - margin
            if verdict.outcome == "unclear" and worst is not None and margin is not None and worst > margin
            else None
        ),
        floor=floor.key,
        aspect=floor.aspect,
        arm=arm,
        arm_label=arm_label,
        status=verdict.outcome,
        reading=verdict.rate.reading() if verdict.rate is not None else None,
        interval=verdict.interval(),
        reason=verdict.reason,
    )


def floor_warning(verdict: FloorVerdict, *, candidate: str) -> str:
    """The line a broken floor puts at the top of the page, the first thing a reader sees: the floor, its aspect, the
    candidate and the interval it was broken on.

    ``Floor broken on conversation: v2's pass rate falls past the 10-point margin on its whole interval, a drop of
    12.3 to 35.2 points.`` A cost floor: ``Floor broken on conversation: v2 costs more per reply than v1 on the whole
    interval.``

    :param verdict: A broken floor, read on ``candidate``.
    :ptype verdict: FloorVerdict
    :param candidate: The candidate, by name.
    :ptype candidate: str
    :return: One sentence.
    :rtype: str
    """
    lead = f"Floor broken on {verdict.floor.aspect}: "
    clause = verdict.floor.clause
    if isinstance(clause, CostLower) or verdict.rate is None:
        return f"{lead}{verdict.reason}."
    rate = verdict.rate
    near, worst = rate.least_drop(), rate.worst_drop()
    which = "pass rate" if rate.name == "the pass rate" else rate.name
    span = "" if near is None or worst is None else f", a drop of {_points(near)} to {_points(worst)} points"
    return (
        f"{lead}{candidate}'s {which} falls past the {format_number(clause.max_points)}-point margin on its whole "
        f"interval{span}."
    )


def _points(value: float) -> str:
    """A change in points as the page states one: to one decimal, a whole number whole."""
    return format_number(round(value, 1))


def _counts(rate: RateReading, control: str, candidate: str) -> str:
    """A rate's counts in words: ``v2 passed 14 of 60 to v1's 29``. The control's total is said only where it
    differs."""
    theirs = f"{rate.candidate_passed} of {rate.candidate_n}"
    ours = (
        str(rate.control_passed) if rate.control_n == rate.candidate_n else f"{rate.control_passed} of {rate.control_n}"
    )
    passed = f"passed {theirs}" if rate.name == "the pass rate" else f"passed {rate.name} on {theirs}"
    return f"{candidate} {passed} to {control}'s {ours}"


def _both(rate: RateReading, control: str, candidate: str) -> str:
    """What two arms that cannot be told apart each passed: ``both 0 of 18``, or each arm's where their totals
    differ."""
    if rate.control_n == rate.candidate_n:
        return f"both {rate.control_passed} of {rate.control_n}"
    return (
        f"{candidate} {rate.candidate_passed} of {rate.candidate_n}, {control} {rate.control_passed} of "
        f"{rate.control_n}"
    )


def _change(drop: float) -> str:
    """A drop in points as a change: ``a drop of 92.7``, ``a gain of 35.5``, ``no change``."""
    if round(drop, 1) == 0:
        return "no change"
    return f"a drop of {_points(drop)}" if drop > 0 else f"a gain of {_points(-drop)}"


def _rate_verdict(
    rate: RateReading, fewest: int, margin: float, control: str, candidate: str, *, named: bool
) -> RateVerdict:
    """One rate a rate floor reads, in one of three states, read in this order: broken (its whole interval past the
    margin, over enough cases), then held (its interval within the margin, over enough cases), then unclear, whose
    reason is the first that applies: both arms passing every trial or none (it cannot separate them), too few cases
    for any interval, a drop past the margin on the point estimate alone, then a drop not shown within the margin.

    ``fewest`` is the fewer usable cases of the two arms; ``named`` names the rate in the reason, as every rate past
    the floor's first is.
    """
    points = format_number(margin)
    drop = f"{candidate}'s drop on {rate.name}" if named else f"{candidate}'s drop"
    least, most = rate.drop_bounds()
    near, worst = rate.least_drop(), rate.worst_drop()
    floored = fewest >= _FLOOR
    if floored and near is not None and near > margin and least > margin:
        return RateVerdict(
            rate,
            "broken",
            f"{_counts(rate, control, candidate)}, a drop that exceeds the margin of {points} points on its whole "
            f"interval (at least {_points(min(near, least))})",
        )
    if floored and worst is not None and worst <= margin and most <= margin:
        at_most = f" (it is at most {_points(worst)})" if worst > 0 else ""
        return RateVerdict(rate, "held", f"{drop} is shown to stay within {points} points{at_most}")
    floor = (
        f"{fewest} cases are too few for the interval that would show it, which is estimated only from {_FLOOR} "
        "independent cases"
    )
    past = f"{drop} is past {points} points on the point estimate alone"
    unshown = f"{drop} is not shown to stay within {points} points"
    if rate.degenerate():
        return RateVerdict(rate, "unclear", f"{rate.name} cannot separate the arms ({_both(rate, control, candidate)})")
    if not floored:
        return RateVerdict(rate, "unclear", f"{past if rate.point_drop() > margin else unshown} ({floor})")
    missing = f"no interval was computed on {rate.name}"
    if rate.point_drop() > margin:
        if near is None or worst is None:
            return RateVerdict(rate, "unclear", f"{past} ({missing})")
        if near > margin:
            return RateVerdict(rate, "unclear", f"{past} (the lost trials could still bring it within)")
        return RateVerdict(rate, "unclear", f"{past}, not on its interval (from {_change(near)} to {_change(worst)})")
    if worst is None:
        return RateVerdict(rate, "unclear", f"{unshown} ({missing})")
    if worst > margin:
        return RateVerdict(rate, "unclear", f"{unshown} (it could be as much as {_points(worst)})")
    return RateVerdict(rate, "unclear", f"{unshown} (the lost trials could still take it past)")


def _rate_floor(
    clause: RateDrop, readings: FloorReadings, control: str, candidate: str
) -> tuple[FloorOutcome, str, RateReading | None, tuple[RateVerdict, ...]]:
    """A rate floor: broken where a rate it reads is broken, held where every one is held, otherwise unclear.

    The floor's reason and deciding rate are the first broken; else the first left unclear whose point estimate is past
    the margin, the one nearest a regression; else the first left unclear; else, held, the one whose interval allows
    the largest drop.
    """
    margin = clause.max_points
    rates = readings.read_by(clause)
    if not rates:
        if clause.reading is None:
            return "unclear", "no pass rate was measured on both arms", None, ()
        return "unclear", f"{_reading_words(clause.reading, readings)} was not measured on both arms", None, ()
    if (fewest := readings.fewest_cases()) is None:
        reason = "the usable cases were not counted on both arms"
        return "unclear", reason, None, tuple(RateVerdict(rate, "unclear", reason) for rate in rates)
    verdicts = tuple(
        _rate_verdict(rate, fewest, margin, control, candidate, named=rate is not rates[0]) for rate in rates
    )
    if broken := [verdict for verdict in verdicts if verdict.outcome == "broken"]:
        return "broken", broken[0].reason, broken[0].rate, verdicts
    if unclear := [verdict for verdict in verdicts if verdict.outcome == "unclear"]:
        lead = next((verdict for verdict in unclear if verdict.rate.point_drop() > margin), unclear[0])
        return "unclear", lead.reason, lead.rate, verdicts
    widest = max(rates, key=lambda rate: rate.worst_drop() or 0.0)
    most = widest.worst_drop() or 0.0
    at_most = f" (it is at most {_points(most)})" if most > 0 else ""
    return (
        "held",
        f"{candidate}'s drop is shown to stay within {format_number(margin)} points{at_most}",
        widest,
        verdicts,
    )


def _cost_floor(readings: FloorReadings, control: str, candidate: str) -> tuple[FloorOutcome, str]:
    """A cost floor, read on its own interval as a rate floor is: broken where the interval on the cost difference lies
    wholly above zero, held where it lies wholly below, each over enough cases; otherwise unclear, saying why."""
    if readings.control_cost is None or readings.candidate_cost is None:
        return "unclear", "cost per reply was not measured on both arms"
    if not readings.cost_compared:
        return "unclear", "cost per reply was not compared: no case carries it on both arms"
    fewest = readings.fewest_cases()
    if fewest is None:
        return "unclear", "the usable cases were not counted on both arms"
    if fewest < _FLOOR:
        return (
            "unclear",
            f"{fewest} cases are too few for the interval that would show the cost either way, which is estimated only "
            f"from {_FLOOR} independent cases",
        )
    interval = readings.cost_interval
    if interval is None:
        return "unclear", "every case cost alike on both arms, so no interval on the cost difference exists"
    if interval[0] > 0:
        return "broken", f"{candidate} costs more per reply than {control} on the whole interval"
    if interval[1] < 0:
        return "held", f"{candidate} costs less per reply than {control}"
    return "unclear", "the cost difference did not separate"


def _floor_contrast(
    bundle: AnalysisContextBundle, reading: str, name: str, control: str, candidate: str
) -> FloorContrast | None:
    """The per-case values a floor reads for ``candidate`` against ``control`` on one reading, whatever the family
    compares: the first rig both ran under, in the bundle's order. None where the bundle carries none."""
    return next(
        (
            contrast
            for contrast in bundle.floor_contrasts
            if contrast.reading == reading
            and contrast.name == name
            and contrast.control.variant_key == control
            and contrast.contrast.variant_key == candidate
        ),
        None,
    )


def _contrast(
    bundle: AnalysisContextBundle, reading: str, name: str, control: str, candidate: str
) -> FamilyComparison | None:
    """The contrast of ``candidate`` against ``control`` on one reading in the aspect's family, carrying the per-case
    values a floor's interval is computed again from. None when the family tested none, or none carrying them."""
    for family in bundle.multiple_comparisons.families:
        for comparison in family.comparisons:
            if (
                comparison.reading == reading
                and comparison.name == name
                and comparison.control.variant_key == control
                and comparison.contrast.variant_key == candidate
                and comparison.case_values is not None
            ):
                return comparison
    return None


def _held(summary: MeasureSummary, passes_high: bool | None) -> int | None:
    """How many of a pass/fail reading's observations passed, counted on its better end.

    ``passes_high`` is the measure's orientation (:func:`~threetears.evals.kernel.metrics.rule_rates`), None for a
    measure a floor does not read. A boolean counts its true observations, a pass/fail scorer and a goal check (which
    reaches the bundle as a mean of 0s and 1s, higher passing) the mean times the observations, read only where that
    is whole; a lower-is-better measure passes on the rest. None for any other reading.
    """
    if goal_check_of(summary.name) is not None:
        passes_high = True
    if passes_high is None:
        return None
    if summary.rate is not None and summary.n_true is not None:
        ones: int | None = summary.n_true
    elif summary.mean is None or summary.n <= 0:
        return None
    else:
        count = summary.mean * summary.n
        ones = round(count) if abs(count - round(count)) <= 1e-6 else None
    if ones is None:
        return None
    return ones if passes_high else summary.n - ones


def _compared_as(name: str) -> str:
    """The measure a check's pass rate is compared on: a label's ``match`` as ``accuracy``, its mean, else itself.

    ``match`` has no merit axis, so no family compares it: ``accuracy``, which the engine derives from it one for one,
    carries the axis (:data:`~threetears.evals.kernel.metrics.ACCURACY_MEASURE`). A label check's rate therefore reads
    the ``accuracy`` contrast, over the same per-case values.
    """
    return ACCURACY_MEASURE if name == MATCH_MEASURE else name


def _read_on(
    rate: RateReading, comparison: FamilyComparison | FloorContrast | None, *, confidence: float
) -> RateReading:
    """``rate`` with what its contrast carries: its per-case values with higher better (each ``1 - v`` where lower is
    better, on the rate's 0-1 range), whether they are paired, and the bounded interval on them computed again at
    ``confidence``, uncorrected."""
    if comparison is None or comparison.case_values is None:
        return rate
    control, contrast = comparison.case_values
    if isinstance(comparison, FamilyComparison) and not comparison.higher_is_better:
        control, contrast = [1.0 - v for v in control], [1.0 - v for v in contrast]
    paired = comparison.test == "paired"
    interval = bounded_difference_interval(
        control, contrast, paired=paired, value_range=_RATE_RANGE, confidence=confidence
    )
    trials = comparison.case_trials if isinstance(comparison, FloorContrast) else None
    passes = comparison.case_passed if isinstance(comparison, FloorContrast) else None
    # Trials that do not line up with the shares they sit beside are none: no depth is read off them.
    if trials is not None and (len(trials[0]), len(trials[1])) != (len(control), len(contrast)):
        trials = passes = None
    return replace(
        rate,
        interval=interval,
        samples=(tuple(control), tuple(contrast)),
        paired=paired,
        confidence=confidence,
        rig=comparison.control.apparatus_class_id,
        trials=None if trials is None else (tuple(trials[0]), tuple(trials[1])),
        passes=None if passes is None or trials is None else (tuple(passes[0]), tuple(passes[1])),
    )


def floor_readings(
    bundle: AnalysisContextBundle, control: str, candidate: str, *, confidence: float = INTERVAL_LEVEL
) -> FloorReadings:
    """The figures a floor is read against for one candidate on the bundle's aspect, from the bundle alone.

    The pass rates are every trial's, then each check's pass rate the arms' cells carry, counted on its better end (a
    goal check's and a pass/fail scorer's as its mean times its observations, a lower-is-better boolean's as its false
    count: :func:`_held` over :func:`~threetears.evals.kernel.metrics.rule_rates`, the list
    :func:`~threetears.evals.kernel.declaration.decision_readings` offers), then each pass/fail judged dimension on
    the capability axis, read the same way from its scores; each arm's lost trials are trials it could still add.
    Each rate carries the bounded interval computed again at ``confidence`` from its per-case values: the overall pass
    rate's from the bundle's ``floor_contrasts``, carried whatever the family compares, each check's from its family
    comparison, where the family compared it. Cost is each arm's cost per reply, with the t interval on the per-case
    costs ``floor_contrasts`` carries, at ``confidence``. Each arm carries its usable cases.

    :param bundle: The aspect's evidence.
    :ptype bundle: AnalysisContextBundle
    :param control: The control's variant key.
    :ptype control: str
    :param candidate: The candidate's variant key.
    :ptype candidate: str
    :param confidence: The two-sided level every interval is read at: the nominal one for a floor; a campaign's
        readings pass the level their Holm step tests at.
    :ptype confidence: float
    :return: The readings; no rate where either arm is off the scoreboard.
    :rtype: FloorReadings
    """
    boards = {board.variant_key: board for board in bundle.scoreboard}
    if control not in boards or candidate not in boards:
        return FloorReadings(rates=())
    ours, theirs = boards[control], boards[candidate]
    lost_control, lost_candidate = sum(ours.lost.values()), sum(theirs.lost.values())
    rates = [
        _read_on(
            RateReading(
                "the pass rate", ours.passed, ours.usable, theirs.passed, theirs.usable, lost_control, lost_candidate
            ),
            _floor_contrast(bundle, "trial", TRIAL_PASS_NAME, control, candidate),
            confidence=confidence,
        )
    ]
    oriented = rule_rates(bundle.measure_catalog.values())
    measures: dict[str, dict[str, tuple[int, int]]] = {}
    for cell in bundle.cell_measures:
        for summary in cell.measures.measures:
            if (held := _held(summary, oriented.get(summary.name))) is None:
                continue
            # An arm over several rigs is counted over all of them, as its trials are.
            passed, n = measures.setdefault(summary.name, {}).get(cell.variant_key, (0, 0))
            measures[summary.name][cell.variant_key] = (passed + held, n + summary.n)
    for name, by_arm in sorted(measures.items()):
        if control in by_arm and candidate in by_arm:
            descriptor = bundle.measure_catalog.get(name)
            heading = measure_name(name, descriptor.reader_name if descriptor is not None else None)
            rates.append(
                _read_on(
                    RateReading(
                        heading, *by_arm[control], *by_arm[candidate], lost_control, lost_candidate, measure=name
                    ),
                    _contrast(bundle, "measure", _compared_as(name), control, candidate),
                    confidence=confidence,
                )
            )
    # A judged dimension on the pass/fail scale is a pass rate too: each score a 1 or a 0, its count the mean times
    # its scores. A boundary dimension is a guardrail, held on its own, never a floor's reading.
    for judged in sorted(bundle.judged_measures, key=lambda measure: measure.name):
        read = any(rate.measure == judged.name for rate in rates)
        if read or judged.value_range != (0.0, 1.0) or judged.axis != "capability":
            continue
        counts: dict[str, tuple[int, int]] = {}
        for scored in judged.arms:
            count = scored.mean * scored.n if scored.mean is not None else None
            if count is None or abs(count - round(count)) > 1e-6:
                continue
            passed, n = counts.get(scored.variant_key, (0, 0))
            counts[scored.variant_key] = (passed + round(count), n + scored.n)
        if control in counts and candidate in counts:
            rates.append(
                _read_on(
                    RateReading(
                        humanize("dimension", judged.name),
                        *counts[control],
                        *counts[candidate],
                        lost_control,
                        lost_candidate,
                        measure=judged.name,
                    ),
                    _contrast(bundle, "judged", judged.name, control, candidate),
                    confidence=confidence,
                )
            )
    cost = _floor_contrast(bundle, "measure", FLOOR_COST_MEASURE, control, candidate)
    cost_interval = (
        None
        if cost is None
        else difference_interval(*cost.case_values, paired=cost.test == "paired", confidence=confidence)
    )
    return FloorReadings(
        rates=tuple(rates),
        control_cost=ours.reply_cost_usd,
        candidate_cost=theirs.reply_cost_usd,
        control_cases=ours.usable_cases,
        candidate_cases=theirs.usable_cases,
        cost_interval=cost_interval,
        cost_compared=cost is not None,
        confidence=confidence,
    )


def aspect_pass_rate(
    bundle: AnalysisContextBundle, control: str, candidate: str, *, confidence: float = INTERVAL_LEVEL
) -> RateReading | None:
    """The aspect's overall pass rate on one candidate against the control: every usable trial's, with the per-case
    shares its family compared, which a campaign's readings test across aspects.

    :param bundle: The aspect's evidence.
    :ptype bundle: AnalysisContextBundle
    :param control: The control's variant key.
    :ptype control: str
    :param candidate: The candidate's variant key.
    :ptype candidate: str
    :param confidence: The two-sided level its interval is read at; the nominal one by default.
    :ptype confidence: float
    :return: The rate, its bounded interval at ``confidence`` and the per-case shares it read (:class:`RateReading`),
        from the bundle's ``floor_contrasts`` whatever the family compares; its ``samples`` and ``interval`` None only
        where a side has fewer than two cases with a usable trial, or no control resolved. None where either arm is
        off the scoreboard.
    :rtype: RateReading | None
    """
    rates = floor_readings(bundle, control, candidate, confidence=confidence).rates
    return next((rate for rate in rates if rate.measure is None), None)


# =============================================================================
# The cases that would show a floor holds: an unclear floor's interval, computed again at more independent cases
# =============================================================================

#: How a projection says what showing a floor holds needs on one side of the question: ``more`` cases than the run
#: had (``cases`` of them), ``enough`` (the run's own ``cases`` already clear), ``lost`` (the interval already clears and only the
#: lost trials keep the drop from being shown within the margin), ``never`` (no number of cases would), or ``beyond``
#: (more than ``cases``, the most searched).
NeedKind = Literal["more", "enough", "lost", "never", "beyond"]

#: Each kind's rank when two floors are needed together: the one needing more decides.
_NEED_RANK: dict[str, int] = {"enough": 0, "lost": 1, "more": 2, "beyond": 3, "never": 4}


@dataclass(frozen=True)
class CasesNeeded:
    """What showing a floor holds needs on one side of the question (:data:`NeedKind`), and the cases that says."""

    kind: NeedKind
    cases: int = 0

    def most(self, other: CasesNeeded) -> CasesNeeded:
        """The more demanding of the two, as showing both reads them."""
        return max(self, other, key=lambda need: (_NEED_RANK[need.kind], need.cases))


@dataclass(frozen=True)
class SampleProjection:
    """The independent cases that would show an unclear floor holds, from the interval that left it unclear
    (:func:`floor_projection`), or that would settle an aspect's pass rate against its margin (:func:`project_rate`).

    **Only holding is projected.** A floor shown broken has already warned; more cases are what would show an unclear
    one holds, so that is the one reading they can change.

    ``cases`` is the usable cases the interval was read over, the fewer of the two arms'. ``if_none`` is what holding
    needs if the arms truly do not differ, the optimistic bound; ``if_observed`` the same if they differ as the run
    observed. A count is always more than ``cases`` and at least the case floor: one the run already reaches is
    ``enough``, or ``lost`` where the interval clears and only the lost trials stand in the way. ``per_case_usd`` is
    what one case costs over every arm (:func:`case_cost_usd`), None where unpriced. A count is of cases: repeats of a
    case narrow an interval far less than new cases, and none is counted.
    """

    cases: int
    if_none: CasesNeeded
    if_observed: CasesNeeded
    per_case_usd: float | None = None

    def cost(self, need: CasesNeeded) -> float | None:
        """What running the cases ``need`` counts over every arm costs; None where unpriced or it counts none."""
        if need.kind != "more" or self.per_case_usd is None:
            return None
        return need.cases * self.per_case_usd

    def counts(self) -> tuple[int, ...]:
        """The case counts it states, fewest first: each ``more`` count and each ``beyond`` cap."""
        return tuple(
            sorted({need.cases for need in (self.if_none, self.if_observed) if need.kind in ("more", "beyond")})
        )

    def joined(self, other: SampleProjection) -> SampleProjection:
        """This and ``other`` both shown, as showing two floors hold reads them: the more demanding of each."""
        return SampleProjection(
            cases=min(self.cases, other.cases),
            if_none=self.if_none.most(other.if_none),
            if_observed=self.if_observed.most(other.if_observed),
            per_case_usd=self.per_case_usd,
        )


def _needed(found: int | None, cases: int, *, cap: int = PROJECTION_CASE_CAP) -> CasesNeeded:
    """A count found by a search as what holding needs: more than this run's ``cases``, at least the floor."""
    if found is None:
        return CasesNeeded("never")
    if found > cap:
        return CasesNeeded("beyond", cap)
    if found <= cases and cases >= _FLOOR:
        return CasesNeeded("enough", cases)
    return CasesNeeded("more", max(found, _FLOOR, cases + 1))


def project_sample(
    *, half_width: float, cases: int, margin: float, observed: float, per_case_usd: float | None = None
) -> SampleProjection:
    """The cases a t interval needs to show a floor holds, if the arms do not differ and if they differ as observed,
    scaled as ``1/√n`` (:func:`~threetears.evals.analysis.stats.cases_to_clear`): a cost floor's.

    Every figure is on the floor's worse side and in one unit: ``observed`` is the observed difference toward worse
    (the cost an arm adds per reply), ``margin`` how far the true difference may go that way with the floor still
    holding (zero for cost), and ``half_width`` how far the interval reaches past ``observed`` on that side. A rate
    floor reads the bounded interval, which this scaling overstates: :func:`project_rate` computes it again instead.

    :param half_width: How far the interval reaches past the observed difference on the worse side, at least 0.
    :ptype half_width: float
    :param cases: The independent cases the interval was read over, at least 1.
    :ptype cases: int
    :param margin: The worst true difference with which the floor still holds.
    :ptype margin: float
    :param observed: The observed difference toward worse.
    :ptype observed: float
    :param per_case_usd: What one case costs over every arm; None where unpriced.
    :ptype per_case_usd: float | None
    :return: The projection.
    :rtype: SampleProjection
    :raises ValueError: As :func:`~threetears.evals.analysis.stats.cases_to_clear`.
    """
    return SampleProjection(
        cases=cases,
        if_none=_needed(cases_to_clear(half_width, cases, margin), cases),
        if_observed=_needed(cases_to_clear(half_width, cases, margin - observed), cases),
        per_case_usd=per_case_usd,
    )


def project_rate(
    rate: RateReading, *, cases: int, margin: float, per_case_usd: float | None = None
) -> SampleProjection | None:
    """The cases that would show one rate within a margin (a floor's, or an aspect's), if the arms do not differ and if
    they differ as observed, from the bounded interval computed again at each count.

    The rate's own per-case values (``RateReading.samples``, the ones its bounded interval read) are repeated to each
    count and the bounded interval read again at the rate's own level (``RateReading.confidence``)
    (:func:`~threetears.evals.analysis.stats.bounded_cases_to_clear`), so the count keeps the run's spread and the
    interval's small-sample excess. With no true difference the interval may reach the whole margin below its
    estimate; as observed, the margin less the drop the per-case values show. Where the interval already keeps the
    drop within the margin as observed and the floor is met, only the lost trials kept it open (``lost``).

    :param rate: One rate read and left unclear.
    :ptype rate: RateReading
    :param cases: The fewer usable cases of the two arms.
    :ptype cases: int
    :param margin: The margin, in points.
    :ptype margin: float
    :param per_case_usd: What one case costs over every arm; None where unpriced.
    :ptype per_case_usd: float | None
    :return: The projection; None where the rate carries no per-case values or no level to read them at.
    :rtype: SampleProjection | None
    """
    if rate.samples is None or rate.confidence is None:
        return None
    control, contrast = rate.samples
    if len(control) < 2 or len(contrast) < 2:
        return None
    paired = rate.paired
    # An unpaired rate's interval is read over each arm's cases apart, at a cost in proportion to them: its search
    # stops sooner and coarser (UNPAIRED_PROJECTION_CAP, or four times the cases observed where that is more, so it
    # always reads past what the run held), so a page of unclear rows is laid out in seconds.
    observed_cases = max(len(control), len(contrast))
    cap = PROJECTION_CASE_CAP if paired else max(UNPAIRED_PROJECTION_CAP, 4 * observed_cases)
    resolution = 0.0 if paired else UNPAIRED_PROJECTION_RESOLUTION

    def search(room: float) -> int | None:
        return bounded_cases_to_clear(
            control,
            contrast,
            paired=paired,
            value_range=(0.0, 1.0),
            confidence=rate.confidence or 0.0,
            room=room,
            cap=cap,
            resolution=resolution,
        )

    drop = (sum(control) / len(control) - sum(contrast) / len(contrast)) * 100
    observed = _needed(search((margin - drop) / 100), cases, cap=cap)
    if observed.kind == "enough":
        observed = CasesNeeded("lost")
    return SampleProjection(
        cases=cases,
        if_none=_needed(search(margin / 100), cases, cap=cap),
        if_observed=observed,
        per_case_usd=per_case_usd,
    )


def case_cost_usd(bundle: AnalysisContextBundle) -> float | None:
    """What one more case costs over every arm, from what the run spent per case: every arm's test cost over the most
    cases an arm ran.

    :param bundle: The campaign's evidence.
    :ptype bundle: AnalysisContextBundle
    :return: The cost in US dollars; None where an arm went unpriced or no case ran.
    :rtype: float | None
    """
    spent = [board.test_cost_usd for board in bundle.scoreboard]
    cases = max((board.cases for board in bundle.scoreboard), default=0)
    if not cases or any(cost is None for cost in spent):
        return None
    return sum(cost or 0.0 for cost in spent) / cases


def floor_projection(verdict: FloorVerdict, *, per_case_usd: float | None = None) -> SampleProjection | None:
    """The cases that would show one unclear floor holds, from the interval the floor was read on.

    A rate floor reads each rate it left unclear through :func:`project_rate`, over the fewer usable cases of the two
    arms, and needs the most any of them needs. A cost floor reads the cost contrast's t interval, scaled
    (:func:`project_sample`), its midpoint the observed difference and its margin zero, so arms that do not differ are
    never shown cheaper.

    :param verdict: The floor as read on one candidate.
    :ptype verdict: FloorVerdict
    :param per_case_usd: What one case costs over every arm (:func:`case_cost_usd`); None where unpriced.
    :ptype per_case_usd: float | None
    :return: The projection; None for a floor broken or held, and where the cases were not counted or a rate it left
        unclear carries nothing to project from.
    :rtype: SampleProjection | None
    """
    readings = verdict.readings
    fewest = readings.fewest_cases() if readings is not None else None
    if verdict.outcome != "unclear" or readings is None or fewest is None or fewest < 1:
        return None
    clause = verdict.floor.clause
    if isinstance(clause, CostLower):
        if readings.cost_interval is None:
            return None
        low, high = readings.cost_interval
        return project_sample(
            half_width=(high - low) / 2, cases=fewest, margin=0.0, observed=(low + high) / 2, per_case_usd=per_case_usd
        )
    found = rate_projections(verdict, per_case_usd=per_case_usd)
    return None if found is None else reduce(SampleProjection.joined, (projection for _, projection in found))


def rate_projections(
    verdict: FloorVerdict, *, per_case_usd: float | None = None
) -> tuple[tuple[RateVerdict, SampleProjection], ...] | None:
    """Each rate an unclear rate floor left unclear, with the cases that would show it holds: what
    :func:`floor_projection` joins into the floor's count, so a reader naming the rate that decides the count reads the
    very projections the count is made of.

    :param verdict: The floor as read on one candidate.
    :ptype verdict: FloorVerdict
    :param per_case_usd: What one case costs over every arm (:func:`case_cost_usd`); None where unpriced.
    :ptype per_case_usd: float | None
    :return: Each unclear rate and its projection, in the floor's order; None for a floor broken or held, a cost
        floor, where the cases were not counted, where no rate is unclear, and where any unclear rate carries nothing
        to project from (so the floor has no count).
    :rtype: tuple[tuple[RateVerdict, SampleProjection], ...] | None
    """
    readings = verdict.readings
    fewest = readings.fewest_cases() if readings is not None else None
    clause = verdict.floor.clause
    if verdict.outcome != "unclear" or fewest is None or fewest < 1 or not isinstance(clause, RateDrop):
        return None
    found = [
        (read, project_rate(read.rate, cases=fewest, margin=clause.max_points, per_case_usd=per_case_usd))
        for read in verdict.rates
        if read.outcome == "unclear"
    ]
    projected = tuple((read, projection) for read, projection in found if projection is not None)
    if not projected or len(projected) < len(found):
        return None
    return projected


def floors_projection(
    verdicts: Sequence[FloorVerdict], *, per_case_usd: float | None = None
) -> SampleProjection | None:
    """The cases that would show every unclear floor on one candidate holds: the most any of them needs
    (:func:`floor_projection`).

    :param verdicts: The floors as read on one candidate.
    :ptype verdicts: Sequence[FloorVerdict]
    :param per_case_usd: What one case costs over every arm; None where unpriced.
    :ptype per_case_usd: float | None
    :return: The projection; None where no floor is unclear, or one left unclear cannot be projected.
    :rtype: SampleProjection | None
    """
    found = [
        floor_projection(verdict, per_case_usd=per_case_usd) for verdict in verdicts if verdict.outcome == "unclear"
    ]
    projections = [projection for projection in found if projection is not None]
    if not projections or len(projections) < len(found):
        return None
    return reduce(SampleProjection.joined, projections)


def projected_usd(value: float) -> str:
    """A projected cost to two figures, since the cases it prices are about: ``$2.7``, ``$120``."""
    if value <= 0:
        return format_number(0.0, prefix="$")
    places = 1 - math.floor(math.log10(value))
    return format_number(round(value, places), prefix="$")


def projected_cell(projection: SampleProjection, need: CasesNeeded) -> str:
    """One side of a projection as a table cell says it, with its cost: ``107 cases, $2.7``, ``this run's 12 cases``,
    ``re-run the lost trials``, ``more than 10000 cases`` or ``none would``."""
    if need.kind == "never":
        return "none would"
    if need.kind == "lost":
        return "re-run the lost trials"
    if need.kind == "enough":
        return f"this run's {projection.cases} cases"
    if need.kind == "beyond":
        return f"more than {format_number(need.cases)} cases"
    cost = projection.cost(need)
    return f"{need.cases} cases" + ("" if cost is None else f", {projected_usd(cost)}")


def projection_note(projections: Sequence[SampleProjection], *, run_cases: int, cost_clause: bool = False) -> str:
    """The sentences under the projected cases: how each count was found and that it is about, what repeats do, what a
    cost floor's blank means, and, where holding needs more cases than the run used on every side, that it does.

    :param projections: Each arm's projection over its floors (:func:`floors_projection`), not each floor's: what
        showing that arm's floors hold needs.
    :ptype projections: Sequence[SampleProjection]
    :param run_cases: The cases the run used: the most any arm ran.
    :ptype run_cases: int
    :param cost_clause: Whether the table shows a cost floor, whose no-difference cell is blank.
    :ptype cost_clause: bool
    :return: The sentences.
    :rtype: str
    """
    said = [
        "Each count is about: it reads the interval again over this run's cases repeated to that many, so it assumes "
        "new cases vary as these did.",
        "Repeats of a case narrow an interval far less than new cases.",
    ]
    if cost_clause:
        said.append(
            "A cost floor has no margin, so arms that do not differ are never shown cheaper; its count scales its "
            "interval by the square root of the cases."
        )
    needs = [need for projection in projections for need in (projection.if_none, projection.if_observed)]
    counted = sorted(need.cases for need in needs if need.kind == "more")
    if counted and all(need.kind in ("more", "beyond", "never") for need in needs):
        beyond = any(need.kind == "beyond" for need in needs)
        caps = sorted({need.cases for need in needs if need.kind == "beyond"})
        if not beyond:
            span = f"about {counted[0]}" if counted[0] == counted[-1] else f"about {counted[0]} to {counted[-1]}"
        elif caps[0] >= counted[-1]:
            span = f"about {counted[0]} to more than {format_number(caps[-1])}"
        else:
            # A search stopped at a cap below a count another row reached: that row needs more than its cap, which
            # may be fewer or more than the counts, so the range is the counts' and the cap is said apart.
            low, high = counted[0], counted[-1]
            counts = f"about {low}" if low == high else f"about {low} to {high}"
            span = f"{counts}, or more than {format_number(caps[0])} where a search stopped there"
        said.append(f"This run used {run_cases} cases; showing the floors hold needs {span}, more than it ran.")
    return " ".join(said)


__all__ = [
    "CasesNeeded",
    "FloorOutcome",
    "FloorReadings",
    "FloorVerdict",
    "NeedKind",
    "RateReading",
    "RateVerdict",
    "SampleProjection",
    "aspect_pass_rate",
    "case_cost_usd",
    "evaluate_floor",
    "floor_projection",
    "floor_readings",
    "floor_statement",
    "floor_warning",
    "floors_projection",
    "project_rate",
    "project_sample",
    "projected_cell",
    "projected_usd",
    "projection_note",
    "rate_projections",
    "read_floor",
]
