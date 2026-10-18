"""A declared decision rule, read clause by clause: which clauses the evidence settles, and which way.

A question declares its rule as typed clauses (:class:`~threetears.evals.kernel.declaration.DecisionRule` on
``Question.decision``): ``RateDrop(max_points=10)`` and ``CostLower()``. No rule is read from a question's words.
:func:`evaluate_rule` settles each clause or says why it cannot, and :func:`rule_statement` says the rule in one
sentence with the arms' names, the one the page opens its Decision section with.

**A rate clause settles either way only on sampling evidence.** Showing the arm breaks a clause and showing it holds
one are both claims about cases not yet run, so each needs an interval. Each rate the clause reads (the overall pass
rate and each check's, or the one reading it names) is read on its own, in one of three states; cost is read as
observed:

- **Shown to break: ``keep``.** Every arm has at least the case floor of usable cases
  (:data:`~threetears.evals.analysis.report.verdict.MIN_CASES`), and the contrast's bounded interval (the arm minus
  the control, ``bounded_interval``: the bounded test's on the rate's 0-1 range at its family's level) lies wholly
  past the margin: the drop nearest zero it allows is still more than the margin, as is the drop with every lost
  trial given its best outcome. Only this reads as a regression.
- **Shown to hold: ``switch``.** Every arm is at the case floor, and the bounded interval bounds the drop at or below
  the margin, as does the bound over lost trials.

  **Switch never reads the t interval the page's contrasts print.** "No worse by more than the margin" is a
  non-inferiority claim, and on a rate's coarse per-case shares the t interval makes it too often: simulated at the
  margin, it let switch through up to 12% of the time against the 2.5% it states, where the bounded interval holds its
  rate at every n (``docs/measuring-soundly.md``). A guardrail's ``held`` reads the same interval for the same reason.
  Its reason names it "the interval a switch reads", since the page's contrasts print the narrower t interval.
  A bundle before shape 55 carries no bounded interval, so its rate clause cannot settle either way; it never falls
  back to the t interval.
- **Neither: ``unsettled``**, with the reason in plain words. That includes a drop past the margin on the point
  estimate alone, whose interval reaches back within it: its reason says the point estimate is past and gives the
  interval, so it never reads as a measured regression. A rate both arms pass on every trial, or on none, cannot
  separate them, and its reason says so with the counts, on any number of cases.
- **A rate clause.** ``keep`` when one rate it reads is shown to break it, ``switch`` when every one is shown to hold
  it, otherwise ``unsettled``. Through ``0.70`` an observed drop past the margin settled ``keep`` on any number of
  cases.
- **Cost, keep.** An arm observed to cost no less per reply settles ``keep``: cost is the arm's own price, not a
  sample of cases.
- **Cost, switch.** Every arm is at the case floor and the cost contrast's interval lies wholly below zero.
  Otherwise ``unsettled``: the cost difference did not separate.
- **The rule's own logic decides.** "Switch only if A and B" keeps the control the moment one clause settles keep,
  switches only when every clause settles switch, and is otherwise unsettled.

**Each reason says the counts and the margin in words, naming the arms, and only the rate it is about.** A rate
shown to break reads ``v2 passed 14 of 60 to v1's 29, a drop that exceeds the margin of 10 points on its whole
interval (at least 12.3)``; one past the margin on its point estimate alone reads ``v2's drop is past 10 points on the
point estimate alone, not on its interval (from a gain of 35.5 to a drop of 92.7)``; one whose interval reaches past
the margin reads ``v2's drop is not shown to stay within 10 points (it could be as much as 53.6)``; one both arms pass on
no trial reads ``the pass rate cannot separate the arms (both 0 of 18)``. A rate past the clause's first is named in
its reason. The page's Decision section gives each rate its own row and its own outcome, so a row's outcome is
always about the reading on that row, and says once that keep is the default, so an unsettled rule keeps the control
for now, and that a clause reads the interval a switch reads (:func:`keep_is_the_default`), so a clause unsettled
beside a contrast that is not separated reads as the two rulers it is.

Every sentence here names the arms by the names the page gives them, never by their role in the comparison.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import reduce
from typing import Literal

from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME, AnalysisContextBundle, FamilyComparison
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.stats import (
    PROJECTION_CASE_CAP,
    SMALL_N_BAND_FLOOR,
    bounded_cases_to_clear,
    cases_to_clear,
)
from threetears.evals.analysis.style import code, humanize, measure_name
from threetears.evals.kernel.analysis_measures import MeasureSummary
from threetears.evals.kernel.declaration import (
    CostLower,
    DecisionClause,
    DecisionRule,
    Question,
    RateDrop,
    states_a_prose_rule,
)
from threetears.evals.kernel.metrics import ACCURACY_MEASURE, MATCH_MEASURE, goal_check_of, rule_rates

#: Where a clause, or the rule, came out.
RuleOutcome = Literal["keep", "switch", "unsettled"]

#: The line a page's Decision section opens with when a question wrote its rule as prose and declared none typed.
PROSE_RULE_LINE = "This question's rule was written as prose, which is no longer read; declare it as a rule."


def keep_is_the_default(control: str) -> str:
    """What the page's Decision section says once, under the clauses: what an unsettled rule means for the reader,
    and why a clause can read unsettled beside a contrast the contrasts table calls not separated, or, on cost, keep.

    :param control: The control, by the name the page gives it.
    :ptype control: str
    :return: Two sentences.
    :rtype: str
    """
    return (
        f"Keep is the default: a switch needs every clause shown, so an unsettled rule keeps {control} for now. A "
        "clause reads the interval a switch reads, which can be wider than the contrasts table's."
    )


#: The fewest usable cases on every arm before a clause can settle ``switch``: the page's case floor.
_FLOOR = SMALL_N_BAND_FLOOR

#: The measure the scoreboard's cost per reply is the mean of, whose contrast decides a cost clause.
_REPLY_COST = "production_replicating_cost"


@dataclass(frozen=True)
class RateReading:
    """One pass rate on both arms: passed of measured, how many trials each could still add, and its interval.

    ``interval`` is the contrast's bounded interval on the arm minus the control (``bounded_interval``), as a share
    (0 to 1) with higher better: the interval a switch reads. None when the bundle computed none for this rate.
    ``measure`` is the measure's key, None for the trials' own pass rate. ``samples`` are the control's and the arm's
    per-case values that interval read (``case_values``), higher better, which a projection reads again at more
    cases (:func:`project_rate`); ``paired`` whether they are one-to-one; ``confidence`` the family's level. None
    before bundle shape 58.
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
        """Whether this is the rate a clause naming ``reading`` reads: a check's by its verbatim text, any other
        measure's by its key."""
        return self.measure is not None and reading in (self.measure, goal_check_of(self.measure))


@dataclass(frozen=True)
class RuleReadings:
    """What a rule is read against for one arm.

    ``control_cases`` and ``candidate_cases`` are each arm's usable cases, None when not counted; ``cost_interval``
    is the bundle's contrast interval on the arm's cost per reply minus the control's, None when none was
    computed. ``predates_bounded`` is True for a bundle shaped before any contrast carried a bounded interval (55), so a
    rate clause says that, not that no interval was computed.
    """

    rates: tuple[RateReading, ...]
    control_cost: float | None = None
    candidate_cost: float | None = None
    control_cases: int | None = None
    candidate_cases: int | None = None
    cost_interval: tuple[float, float] | None = None
    predates_bounded: bool = False

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
    """One rate a rate clause reads, settled on its own: ``keep`` where it is shown to break the clause, ``switch``
    where it is shown to hold it, ``unsettled`` otherwise, with the reason in plain words about this rate alone."""

    rate: RateReading
    outcome: RuleOutcome
    reason: str


@dataclass(frozen=True)
class ClauseVerdict:
    """One clause settled, or not, with the reason in plain words, the clause as a sentence naming the arm, the rate
    that settled it ``keep``, and, for a rate clause, each rate it reads with its own outcome."""

    clause: DecisionClause
    outcome: RuleOutcome
    reason: str
    statement: str
    rate: RateReading | None = None
    rates: tuple[RateVerdict, ...] = ()


@dataclass(frozen=True)
class RuleVerdict:
    """The rule's outcome and each clause's, with the readings they were settled on."""

    outcome: RuleOutcome
    clauses: tuple[ClauseVerdict, ...]
    readings: RuleReadings | None = None

    def settles(self, rate: RateReading) -> bool:
        """Whether a rate clause settled ``rate`` either way: a rate it read, by its measure, shown to break the clause
        or shown to hold it. A rate it read and left unsettled, or one it did not read, is not settled, however far
        its counts fell."""
        return any(
            read.outcome != "unsettled" and read.rate.measure == rate.measure
            for verdict in self.clauses
            if isinstance(verdict.clause, RateDrop)
            for read in verdict.rates
        )

    def settles_cost(self) -> bool:
        """Whether a cost clause settled, either way."""
        return any(isinstance(verdict.clause, CostLower) and verdict.outcome != "unsettled" for verdict in self.clauses)


def ruled_question(questions: Sequence[Question]) -> Question | None:
    """The first of ``questions`` that declares a decision rule; None when none does."""
    return next((question for question in questions if question.decision is not None), None)


def prose_ruled(questions: Sequence[Question]) -> bool:
    """Whether ``questions`` declare no typed rule and one of them wrote its rule as prose, which nothing reads.

    A question stored before rules were typed loads as it was written; its page says the rule went unread, and its
    verdict reads exploratory rather than as an answer to a rule nobody applied.
    """
    return ruled_question(questions) is None and any(states_a_prose_rule(question) for question in questions)


def _reading_words(reading: str, readings: RuleReadings | None) -> str:
    """A reading a clause names, as the page names it: the rate's own name where the arms carry it, else a key in
    words, else a check's text as code."""
    rates = readings.rates if readings is not None else ()
    if (rate := next((rate for rate in rates if rate.answers(reading)), None)) is not None:
        return rate.name
    return humanize("dimension", reading) if reading.isidentifier() else code(reading)


def clause_statement(clause: DecisionClause, *, candidate: str, readings: RuleReadings | None = None) -> str:
    """One clause as the page says it, naming the arm: ``no pass rate drops by more than 2 points``,
    ``no tone rate drops by more than 2 points``, ``v2 costs less per reply``.

    :param clause: The clause.
    :ptype clause: DecisionClause
    :param candidate: The arm it is read on, by name; ``it`` where the sentence names several.
    :ptype candidate: str
    :param readings: What the clause is read against, whose rate names a named reading is said by; None to say it
        from its key.
    :ptype readings: RuleReadings | None
    :return: The clause, lower case, with no closing stop.
    :rtype: str
    """
    if isinstance(clause, CostLower):
        return f"{candidate} costs less per reply"
    which = "pass" if clause.reading is None else _reading_words(clause.reading, readings)
    return f"no {which} rate drops by more than {format_number(clause.max_points)} points"


def rule_statement(
    rule: DecisionRule, *, control: str, candidates: Sequence[str], readings: RuleReadings | None = None
) -> str:
    """The rule in the sentence the first screen's subhead states it in, the arms by name.

    ``Switch to v2 only if no pass rate drops by more than 2 points and v2 costs less per reply. Otherwise stay on
    v1.`` With several arms the switch names each, and a cost clause says ``it``.

    :param rule: The declared rule.
    :ptype rule: DecisionRule
    :param control: The control, by name.
    :ptype control: str
    :param candidates: Every other arm, by name.
    :ptype candidates: Sequence[str]
    :param readings: The first arm's readings, whose rate names a named reading is said by.
    :ptype readings: RuleReadings | None
    :return: Two sentences: when to switch, and otherwise.
    :rtype: str
    """
    subject = candidates[0] if len(candidates) == 1 else "it"
    target = " or ".join(candidates) if candidates else "another arm"
    said = [clause_statement(clause, candidate=subject, readings=readings) for clause in rule.clauses]
    conditions = said[0] if len(said) == 1 else f"{', '.join(said[:-1])} and {said[-1]}"
    return f"Switch to {target} only if {conditions}. Otherwise stay on {control}."


def evaluate_rule(rule: DecisionRule, readings: RuleReadings, *, control: str, candidate: str) -> RuleVerdict:
    """Settle each clause of ``rule`` on ``readings``, then the rule.

    :param rule: The declared rule.
    :ptype rule: DecisionRule
    :param readings: The control's and the arm's figures.
    :ptype readings: RuleReadings
    :param control: The control, by the name the page gives it, which a reason says.
    :ptype control: str
    :param candidate: The arm, likewise.
    :ptype candidate: str
    :return: The rule's outcome and each clause's, with reasons naming the arms.
    :rtype: RuleVerdict
    """
    verdicts = tuple(_clause(clause, readings, control, candidate) for clause in rule.clauses)
    outcomes = {verdict.outcome for verdict in verdicts}
    outcome: RuleOutcome = (
        "keep" if "keep" in outcomes else "switch" if outcomes == {"switch"} and verdicts else "unsettled"
    )
    return RuleVerdict(outcome=outcome, clauses=verdicts, readings=readings)


def _clause(clause: DecisionClause, readings: RuleReadings, control: str, candidate: str) -> ClauseVerdict:
    """One clause's verdict: a rate clause settles either way only on an interval over enough cases, a cost clause
    keeps on a cost observed no lower and switches only on an interval over enough cases."""
    statement = clause_statement(clause, candidate=candidate, readings=readings)
    if isinstance(clause, CostLower):
        outcome, reason = _cost_clause(readings, control, candidate)
        return ClauseVerdict(clause, outcome, reason, statement)
    outcome, reason, rate, rates = _rate_clause(clause, readings, control, candidate)
    return ClauseVerdict(clause, outcome, reason, statement, rate=rate, rates=rates)


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
    rate: RateReading, readings: RuleReadings, fewest: int, margin: float, control: str, candidate: str, *, named: bool
) -> RateVerdict:
    """One rate a rate clause reads, in one of three states: shown to break the clause (its whole interval past the
    margin, over enough cases), shown to hold it (its interval within the margin, over enough cases), or neither.

    ``fewest`` is the fewer usable cases of the two arms; ``named`` names the rate in the reason, as every rate past
    the clause's first is.
    """
    points = format_number(margin)
    drop = f"{candidate}'s drop on {rate.name}" if named else f"{candidate}'s drop"
    least, most = rate.drop_bounds()
    near, worst = rate.least_drop(), rate.worst_drop()
    floored = fewest >= _FLOOR
    if floored and near is not None and near > margin and least > margin:
        return RateVerdict(
            rate,
            "keep",
            f"{_counts(rate, control, candidate)}, a drop that exceeds the margin of {points} points on its whole "
            f"interval (at least {_points(min(near, least))})",
        )
    if floored and worst is not None and worst <= margin and most <= margin:
        at_most = f" (it is at most {_points(worst)})" if worst > 0 else ""
        return RateVerdict(rate, "switch", f"{drop} is shown to stay within {points} points{at_most}")
    floor = (
        f"{fewest} cases are too few for the interval that would show it, which is estimated only from {_FLOOR} "
        "independent cases"
    )
    past = f"{drop} is past {points} points on the point estimate alone"
    unshown = f"{drop} is not shown to stay within {points} points"
    if rate.degenerate():
        return RateVerdict(
            rate, "unsettled", f"{rate.name} cannot separate the arms ({_both(rate, control, candidate)})"
        )
    if not floored:
        return RateVerdict(rate, "unsettled", f"{past if rate.point_drop() > margin else unshown} ({floor})")
    missing = (
        "this bundle predates the interval a switch needs"
        if readings.predates_bounded
        else f"no interval was computed on {rate.name}"
    )
    if rate.point_drop() > margin:
        if near is None or worst is None:
            return RateVerdict(rate, "unsettled", f"{past} ({missing})")
        if near > margin:
            return RateVerdict(rate, "unsettled", f"{past} (the lost trials could still bring it within)")
        return RateVerdict(rate, "unsettled", f"{past}, not on its interval (from {_change(near)} to {_change(worst)})")
    if worst is None:
        return RateVerdict(rate, "unsettled", f"{unshown} ({missing})")
    if worst > margin:
        return RateVerdict(rate, "unsettled", f"{unshown} (it could be as much as {_points(worst)})")
    return RateVerdict(rate, "unsettled", f"{unshown} (the lost trials could still take it past)")


def _rate_clause(
    clause: RateDrop, readings: RuleReadings, control: str, candidate: str
) -> tuple[RuleOutcome, str, RateReading | None, tuple[RateVerdict, ...]]:
    """A rate clause: keep where a rate it reads is shown to break it, switch where every one is shown to hold it,
    otherwise unsettled.

    The clause's reason is the deciding rate's: the first shown to break; else the first left unsettled whose point
    estimate is past the margin, the one nearest a regression; else the first left unsettled.
    """
    margin = clause.max_points
    rates = readings.read_by(clause)
    if not rates:
        if clause.reading is None:
            return "unsettled", "no pass rate was measured on both arms", None, ()
        return "unsettled", f"{_reading_words(clause.reading, readings)} was not measured on both arms", None, ()
    if (fewest := readings.fewest_cases()) is None:
        reason = "the usable cases were not counted on both arms"
        return "unsettled", reason, None, tuple(RateVerdict(rate, "unsettled", reason) for rate in rates)
    verdicts = tuple(
        _rate_verdict(rate, readings, fewest, margin, control, candidate, named=rate is not rates[0]) for rate in rates
    )
    if broken := [verdict for verdict in verdicts if verdict.outcome == "keep"]:
        return "keep", broken[0].reason, broken[0].rate, verdicts
    if unsettled := [verdict for verdict in verdicts if verdict.outcome == "unsettled"]:
        lead = next((verdict for verdict in unsettled if verdict.rate.point_drop() > margin), unsettled[0])
        return "unsettled", lead.reason, None, verdicts
    widest = max(rate.worst_drop() or 0.0 for rate in rates)
    at_most = f" (it is at most {_points(widest)})" if widest > 0 else ""
    return (
        "switch",
        f"{candidate}'s drop is shown to stay within {format_number(margin)} points{at_most}",
        None,
        verdicts,
    )


def _cost_clause(readings: RuleReadings, control: str, candidate: str) -> tuple[RuleOutcome, str]:
    """A cost clause: keep on a cost observed no lower, switch on an interval wholly below zero over enough cases."""
    ours, theirs = readings.control_cost, readings.candidate_cost
    if ours is None or theirs is None:
        return "unsettled", "cost per reply was not measured on both arms"
    if theirs > ours:
        return "keep", f"{candidate} costs more per reply than {control}"
    if theirs == ours:
        return "keep", f"{candidate} costs no less per reply than {control}"
    fewest = readings.fewest_cases()
    if fewest is None:
        return "unsettled", "the usable cases were not counted on both arms"
    if fewest < _FLOOR:
        return (
            "unsettled",
            f"{fewest} cases cannot show the cost is lower, which takes an interval estimated only from {_FLOOR} "
            "independent cases",
        )
    if readings.cost_interval is None or readings.cost_interval[1] >= 0:
        return "unsettled", "the cost difference did not separate"
    return "switch", f"{candidate} costs less per reply than {control}"


#: The bundle shape from which a family comparison carries ``bounded_interval``.
_BOUNDED_SHAPE = 55


def _contrast(
    bundle: AnalysisContextBundle,
    reading: str,
    name: str,
    control: str,
    candidate: str,
    *,
    bounded: bool,
    question_id: str | None,
) -> FamilyComparison | None:
    """The contrast of ``candidate`` against ``control`` on one reading, in ``question_id``'s family, with the interval
    asked for.

    ``bounded`` asks for the bounded interval a non-inferiority claim reads, else the family's interval. Only the
    family of the question the rule was stated in is read: another family's interval is at another level. None when
    that family tested none, or none carrying that interval.
    """
    for family in bundle.multiple_comparisons.families:
        if family.question_id != question_id:
            continue
        for comparison in family.comparisons:
            if (
                comparison.reading == reading
                and comparison.name == name
                and comparison.control.variant_key == control
                and comparison.contrast.variant_key == candidate
                and (comparison.bounded_interval if bounded else comparison.interval) is not None
            ):
                return comparison
    return None


def _held(summary: MeasureSummary, passes_high: bool | None) -> int | None:
    """How many of a pass/fail reading's observations passed, counted on its better end.

    ``passes_high`` is the measure's orientation (:func:`~threetears.evals.kernel.metrics.rule_rates`), None for a
    measure a rule does not read. A boolean counts its true observations, a pass/fail scorer and a goal check (which
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
    the ``accuracy`` contrast, over the same per-case values. Until ``.30`` it looked for a ``match`` contrast, found
    none, and could never settle switch.
    """
    return ACCURACY_MEASURE if name == MATCH_MEASURE else name


def _bounded_better_way(comparison: FamilyComparison | None) -> tuple[float, float] | None:
    """A contrast's bounded interval with higher better: as stored, or negated where lower is better."""
    if comparison is None or comparison.bounded_interval is None:
        return None
    low, high = comparison.bounded_interval
    return (low, high) if comparison.higher_is_better else (-high, -low)


def _read_on(
    rate: RateReading, comparison: FamilyComparison | None, bundle: AnalysisContextBundle, question_id: str | None
) -> RateReading:
    """``rate`` with what its contrast carries: the bounded interval with higher better, and what a projection reads
    again (its per-case values with higher better, each ``1 - v`` where lower is better on the rate's 0-1 range,
    whether they are paired, and its family's level)."""
    rate = replace(rate, interval=_bounded_better_way(comparison))
    if comparison is None or comparison.case_values is None:
        return rate
    level = next(
        (
            family.interval_level
            for family in bundle.multiple_comparisons.families
            if family.question_id == question_id and any(one is comparison for one in family.comparisons)
        ),
        None,
    )
    control, contrast = comparison.case_values
    if not comparison.higher_is_better:
        control, contrast = [1.0 - v for v in control], [1.0 - v for v in contrast]
    return replace(
        rate, samples=(tuple(control), tuple(contrast)), paired=comparison.test == "paired", confidence=level
    )


def rule_readings(
    bundle: AnalysisContextBundle, control: str, candidate: str, *, question_id: str | None
) -> RuleReadings:
    """The figures a rule is read against for one candidate arm, from the bundle alone.

    The pass rates are every trial's, then each check's pass rate the arms' cells carry, counted on its better end (a
    goal check's and a pass/fail scorer's as its mean times its observations, a lower-is-better boolean's as its false
    count: :func:`_held` over :func:`~threetears.evals.kernel.metrics.rule_rates`, the list
    :func:`~threetears.evals.kernel.declaration.decision_readings` offers), then each pass/fail judged dimension on
    the capability axis, read the same way from its scores; each arm's lost trials are trials it could still add.
    Each rate carries its contrast's bounded interval, the one a switch reads, where the bundle computed one (the
    overall pass rate's is its ``trial`` contrast). Cost is each arm's cost per reply, with
    the cost contrast's interval. Each arm carries its usable cases.

    :param bundle: The campaign's evidence.
    :ptype bundle: AnalysisContextBundle
    :param control: The control's variant key.
    :ptype control: str
    :param candidate: The candidate's variant key.
    :ptype candidate: str
    :param question_id: The question the rule is stated in, whose family alone the intervals are read from; None for
        the campaign-wide family.
    :ptype question_id: str | None
    :return: The readings.
    :rtype: RuleReadings
    """
    boards = {board.variant_key: board for board in bundle.scoreboard}
    if control not in boards or candidate not in boards:
        return RuleReadings(rates=())
    ours, theirs = boards[control], boards[candidate]
    lost_control, lost_candidate = sum(ours.lost.values()), sum(theirs.lost.values())
    rates = [
        _read_on(
            RateReading(
                "the pass rate", ours.passed, ours.usable, theirs.passed, theirs.usable, lost_control, lost_candidate
            ),
            _contrast(bundle, "trial", TRIAL_PASS_NAME, control, candidate, bounded=True, question_id=question_id),
            bundle,
            question_id,
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
                    _contrast(
                        bundle, "measure", _compared_as(name), control, candidate, bounded=True, question_id=question_id
                    ),
                    bundle,
                    question_id,
                )
            )
    # A judged dimension on the pass/fail scale is a pass rate too: each score a 1 or a 0, its count the mean times
    # its scores. A boundary dimension is a guardrail, held on its own, never a rule's reading.
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
                    _contrast(bundle, "judged", judged.name, control, candidate, bounded=True, question_id=question_id),
                    bundle,
                    question_id,
                )
            )
    cost = _contrast(bundle, "measure", _REPLY_COST, control, candidate, bounded=False, question_id=question_id)
    return RuleReadings(
        rates=tuple(rates),
        control_cost=ours.reply_cost_usd,
        candidate_cost=theirs.reply_cost_usd,
        control_cases=ours.usable_cases,
        candidate_cases=theirs.usable_cases,
        cost_interval=None if cost is None else cost.interval,
        predates_bounded=bundle.schema_version < _BOUNDED_SHAPE,
    )


# =============================================================================
# The cases a switch needs: an unsettled clause's interval, computed again at more independent cases
# =============================================================================

#: How a projection says what a switch needs on one side of the question: ``more`` cases than the run had (``cases``
#: of them), ``enough`` (the run's own ``cases`` already clear), ``lost`` (the interval already clears and only the
#: lost trials keep the drop from being shown within the margin), ``never`` (no number of cases would), or ``beyond``
#: (more than ``cases``, the most searched).
NeedKind = Literal["more", "enough", "lost", "never", "beyond"]

#: Each kind's rank when two clauses are needed together: the one needing more decides.
_NEED_RANK: dict[str, int] = {"enough": 0, "lost": 1, "more": 2, "beyond": 3, "never": 4}


@dataclass(frozen=True)
class CasesNeeded:
    """What a switch needs on one side of the question (:data:`NeedKind`), and the cases that says."""

    kind: NeedKind
    cases: int = 0

    def most(self, other: CasesNeeded) -> CasesNeeded:
        """The more demanding of the two, as a switch needing both reads them."""
        return max(self, other, key=lambda need: (_NEED_RANK[need.kind], need.cases))


@dataclass(frozen=True)
class SampleProjection:
    """The independent cases that would show an unsettled clause holds, so that a switch could follow, from the
    interval that left it unsettled (:func:`clause_projection`).

    **Only a switch is projected.** Keep is the default (:func:`keep_is_the_default`): an unsettled rule keeps the
    control already, so a switch is the one decision more cases can change.

    ``cases`` is the usable cases the interval was read over, the fewer of the two arms'. ``if_none`` is what a switch
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
        """This and ``other`` both shown, as a switch needing both clauses reads them: the more demanding of each."""
        return SampleProjection(
            cases=min(self.cases, other.cases),
            if_none=self.if_none.most(other.if_none),
            if_observed=self.if_observed.most(other.if_observed),
            per_case_usd=self.per_case_usd,
        )


def _needed(found: int | None, cases: int, *, cap: int = PROJECTION_CASE_CAP) -> CasesNeeded:
    """A count found by a search as what a switch needs: more than this run's ``cases``, at least the floor."""
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
    """The cases a t interval needs to show a clause holds, if the arms do not differ and if they differ as observed,
    scaled as ``1/√n`` (:func:`~threetears.evals.analysis.stats.cases_to_clear`): a cost clause's.

    Every figure is on the clause's worse side and in one unit: ``observed`` is the observed difference toward worse
    (the cost an arm adds per reply), ``margin`` how far the true difference may go that way with the clause still
    holding (zero for cost), and ``half_width`` how far the interval reaches past ``observed`` on that side. A rate
    clause reads the bounded interval, which this scaling overstates: :func:`project_rate` computes it again instead.

    :param half_width: How far the interval reaches past the observed difference on the worse side, at least 0.
    :ptype half_width: float
    :param cases: The independent cases the interval was read over, at least 1.
    :ptype cases: int
    :param margin: The worst true difference with which the clause still holds.
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
    """The cases that would show one rate within a clause's margin, if the arms do not differ and if they differ as
    observed, from the bounded interval computed again at each count.

    The rate's own per-case values (``RateReading.samples``, the ones its bounded interval read) are repeated to each
    count and the bounded interval read again at the family's level
    (:func:`~threetears.evals.analysis.stats.bounded_cases_to_clear`), so the count keeps the run's spread and the
    interval's small-sample excess. With no true difference the interval may reach the whole margin below its
    estimate; as observed, the margin less the drop the per-case values show. Where the interval already keeps the
    drop within the margin as observed and the floor is met, only the lost trials kept the clause open (``lost``).

    :param rate: One rate the clause read and left unsettled.
    :ptype rate: RateReading
    :param cases: The fewer usable cases of the two arms.
    :ptype cases: int
    :param margin: The clause's margin, in points.
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

    def search(room: float) -> int | None:
        return bounded_cases_to_clear(
            control, contrast, paired=paired, value_range=(0.0, 1.0), confidence=rate.confidence or 0.0, room=room
        )

    drop = (sum(control) / len(control) - sum(contrast) / len(contrast)) * 100
    observed = _needed(search((margin - drop) / 100), cases)
    if observed.kind == "enough":
        observed = CasesNeeded("lost")
    return SampleProjection(
        cases=cases, if_none=_needed(search(margin / 100), cases), if_observed=observed, per_case_usd=per_case_usd
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


def clause_projection(
    verdict: ClauseVerdict, readings: RuleReadings, *, per_case_usd: float | None = None
) -> SampleProjection | None:
    """The cases that would show one unsettled clause holds, from the interval the clause was read on.

    A rate clause reads each rate it left unsettled through :func:`project_rate`, over the fewer usable cases of the
    two arms, and needs the most any of them needs. A cost clause reads the cost contrast's t interval, scaled
    (:func:`project_sample`), its midpoint the observed difference and its margin zero, so arms that do not differ are
    never shown cheaper.

    :param verdict: The clause as the rule read it.
    :ptype verdict: ClauseVerdict
    :param readings: What it was read against.
    :ptype readings: RuleReadings
    :param per_case_usd: What one case costs over every arm (:func:`case_cost_usd`); None where unpriced.
    :ptype per_case_usd: float | None
    :return: The projection; None for a settled clause, and where the cases were not counted or a rate it left
        unsettled carries nothing to project from.
    :rtype: SampleProjection | None
    """
    fewest = readings.fewest_cases()
    if verdict.outcome != "unsettled" or fewest is None or fewest < 1:
        return None
    clause = verdict.clause
    if isinstance(clause, CostLower):
        if readings.cost_interval is None:
            return None
        low, high = readings.cost_interval
        return project_sample(
            half_width=(high - low) / 2, cases=fewest, margin=0.0, observed=(low + high) / 2, per_case_usd=per_case_usd
        )
    found = [
        project_rate(read.rate, cases=fewest, margin=clause.max_points, per_case_usd=per_case_usd)
        for read in verdict.rates
        if read.outcome == "unsettled"
    ]
    projections = [projection for projection in found if projection is not None]
    if not projections or len(projections) < len(found):
        return None
    return reduce(SampleProjection.joined, projections)


def rule_projection(verdict: RuleVerdict, *, per_case_usd: float | None = None) -> SampleProjection | None:
    """The cases that would let an unsettled rule switch: every clause it left unsettled shown to hold, so the most
    any of them needs (:func:`clause_projection`).

    :param verdict: The rule as read on one arm.
    :ptype verdict: RuleVerdict
    :param per_case_usd: What one case costs over every arm; None where unpriced.
    :ptype per_case_usd: float | None
    :return: The projection; None for a settled rule, and where a clause it left unsettled cannot be projected.
    :rtype: SampleProjection | None
    """
    if verdict.outcome != "unsettled" or verdict.readings is None:
        return None
    readings = verdict.readings
    found = [
        clause_projection(clause, readings, per_case_usd=per_case_usd)
        for clause in verdict.clauses
        if clause.outcome == "unsettled"
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


def _count_words(need: CasesNeeded, *, cases: int, short: bool) -> str:
    """One count as the projection sentence says it: ``about 107 independent cases`` (``about 107`` when ``short``),
    ``this run's 12 cases``, ``more than 10,000 independent cases``."""
    if need.kind == "enough":
        return f"this run's {cases} cases"
    said = f"more than {need.cases:,}" if need.kind == "beyond" else f"about {need.cases}"
    return said if short else f"{said} independent cases"


def projection_sentence(projection: SampleProjection) -> str:
    """The cases a switch needs in one sentence, the verdict line's and a next step's: ``A switch needs about 107
    independent cases if the arms do not differ, about 205 if they differ as observed; it counts cases, not repeats.``

    Where the interval already keeps the drop within the margin as observed, the lost trials are what stand in the
    way, and the sentence says so instead.

    :param projection: The rule's projection (:func:`rule_projection`).
    :ptype projection: SampleProjection
    :return: The sentence, held to the page's numeral and length limits.
    :rtype: str
    """
    none, observed, cases = projection.if_none, projection.if_observed, projection.cases
    if observed.kind == "lost":
        return (
            "This run's interval already keeps the drop within the margin as observed: re-running its lost trials, "
            "not more cases, would settle it."
        )
    tail = "; it counts cases, not repeats."
    if none.kind != "never" and observed.kind != "never":
        return (
            f"A switch needs {_count_words(none, cases=cases, short=False)} if the arms do not differ, "
            f"{_count_words(observed, cases=cases, short=True)} if they differ as observed{tail}"
        )
    if none.kind != "never":
        return (
            f"A switch needs {_count_words(none, cases=cases, short=False)} if the arms do not differ, and none would "
            f"if they differ as observed{tail}"
        )
    if observed.kind != "never":
        return (
            f"A switch needs {_count_words(observed, cases=cases, short=False)} if the arms differ as observed, and "
            f"none would if they do not differ{tail}"
        )
    return "No number of new cases would show a switch, whether the arms do not differ or differ as observed."


def projected_cell(projection: SampleProjection, need: CasesNeeded) -> str:
    """One side of a projection as a table cell says it, with its cost: ``107 cases, $2.7``, ``this run's 12 cases``,
    ``re-run the lost trials``, ``more than 10,000 cases`` or ``none would``."""
    if need.kind == "never":
        return "none would"
    if need.kind == "lost":
        return "re-run the lost trials"
    if need.kind == "enough":
        return f"this run's {projection.cases} cases"
    if need.kind == "beyond":
        return f"more than {need.cases:,} cases"
    cost = projection.cost(need)
    return f"{need.cases} cases" + ("" if cost is None else f", {projected_usd(cost)}")


def projection_note(projections: Sequence[SampleProjection], *, run_cases: int, cost_clause: bool = False) -> str:
    """The sentences under the projected cases: how each count was found and that it is about, what repeats do, what a
    cost clause's blank means, and, where a switch needs more cases than the run used on every side, that it does.

    :param projections: Each arm's rule projection (:func:`rule_projection`), not each clause's: what a switch on that
        arm needs.
    :ptype projections: Sequence[SampleProjection]
    :param run_cases: The cases the run used: the most any arm ran.
    :ptype run_cases: int
    :param cost_clause: Whether the table shows a cost clause, whose no-difference cell is blank.
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
            "A cost clause has no margin, so arms that do not differ are never shown cheaper; its count scales its "
            "interval by the square root of the cases."
        )
    needs = [need for projection in projections for need in (projection.if_none, projection.if_observed)]
    counted = sorted(need.cases for need in needs if need.kind == "more")
    if counted and all(need.kind in ("more", "beyond", "never") for need in needs):
        beyond = any(need.kind == "beyond" for need in needs)
        top = f"more than {PROJECTION_CASE_CAP:,}" if beyond else str(counted[-1])
        span = f"about {counted[0]}" if counted[0] == counted[-1] and not beyond else f"about {counted[0]} to {top}"
        said.append(f"This run used {run_cases} cases; a switch needs {span}, more than it ran.")
    return " ".join(said)


__all__ = [
    "PROSE_RULE_LINE",
    "ClauseVerdict",
    "RateReading",
    "RateVerdict",
    "RuleOutcome",
    "RuleReadings",
    "RuleVerdict",
    "CasesNeeded",
    "NeedKind",
    "SampleProjection",
    "case_cost_usd",
    "clause_projection",
    "clause_statement",
    "evaluate_rule",
    "keep_is_the_default",
    "project_rate",
    "project_sample",
    "projected_cell",
    "projected_usd",
    "projection_note",
    "projection_sentence",
    "prose_ruled",
    "rule_projection",
    "rule_readings",
    "rule_statement",
    "ruled_question",
]
