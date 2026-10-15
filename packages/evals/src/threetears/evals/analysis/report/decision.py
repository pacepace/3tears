"""A declared decision rule, read clause by clause: which clauses the evidence settles, and which way.

A question declares its rule as typed clauses (:class:`~threetears.evals.kernel.declaration.DecisionRule` on
``Question.decision``): ``RateDrop(max_points=10)`` and ``CostLower()``. No rule is read from a question's words.
:func:`evaluate_rule` settles each clause or says why it cannot, and :func:`rule_statement` says the rule in one
sentence with the arms' names, the one the page opens its Decision section with.

**Keep may settle on what the run observed; switch needs sampling evidence.** The rule's default is the control, so
showing the arm breaks a clause is enough not to switch. Showing it holds a clause is a claim about cases not
yet run, so it needs an interval:

- **A pass rate, keep.** The arm's drop from the control is bounded by giving every lost trial its best and its
  worst outcome. A drop past the margin even at best settles ``keep``.
- **A pass rate, switch.** Every arm has at least the case floor of usable cases
  (:data:`~threetears.evals.analysis.report.verdict.MIN_CASES`), and for every rate the clause reads (the overall pass
  rate and each check's, or the one reading it names) the contrast's bounded interval (the arm minus the control,
  ``bounded_interval``: the bounded test's on the rate's 0-1 range at its family's level) bounds the drop at or below
  the margin, as does the bound over lost trials. Anything else is ``unsettled``, with the reason in plain words.

  **Switch never reads the t interval the page's contrasts print.** "No worse by more than the margin" is a
  non-inferiority claim, and on a rate's coarse per-case shares the t interval makes it too often: simulated at the
  margin, it let switch through up to 12% of the time against the 2.5% it states, where the bounded interval holds its
  rate at every n (``docs/measuring-soundly.md``). A guardrail's ``held`` reads the same interval for the same reason.
  Its reason names it "the interval a switch reads", since the page's contrasts print the narrower t interval.
  A bundle before shape 55 carries no bounded interval, so its rate clause cannot settle switch; it never falls back
  to the t interval.
- **Cost, keep.** An arm observed to cost no less per reply settles ``keep``.
- **Cost, switch.** Every arm is at the case floor and the cost contrast's interval lies wholly below zero.
  Otherwise ``unsettled``: the cost difference did not separate.
- **The rule's own logic decides.** "Switch only if A and B" keeps the control the moment one clause settles keep,
  switches only when every clause settles switch, and is otherwise unsettled.

**Each reason says the counts and the margin in words, naming the arms.** A rate kept on the counts reads
``v2 passed 14 of 60 to v1's 29, a drop of 25 points against a margin of 10``; one whose interval reaches past the
margin reads ``v2's drop is not shown to stay within 10 points (it could be as much as 53.6)``. The page's Decision
section says once that keep is the default, so an unsettled rule keeps the control for now, and that a clause reads
the interval a switch reads (:func:`keep_is_the_default`), so a clause unsettled or kept beside a contrast that is not
separated reads as the two rulers it is.

Nothing here estimates the size of a difference: under the case floor a settled ``keep`` is read on the counts alone.
Every sentence here names the arms by the names the page gives them, never by their role in the comparison.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME, AnalysisContextBundle, FamilyComparison
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.stats import SMALL_N_BAND_FLOOR
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
    and why a clause can read keep or unsettled beside a contrast the contrasts table calls not separated.

    Args:
        control: The control, by the name the page gives it.

    Returns:
        Two sentences.
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
    ``measure`` is the measure's key, None for the trials' own pass rate.
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
class ClauseVerdict:
    """One clause settled, or not, with the reason in plain words, the clause as a sentence naming the arm, and the
    rate that settled it ``keep``."""

    clause: DecisionClause
    outcome: RuleOutcome
    reason: str
    statement: str
    rate: RateReading | None = None


@dataclass(frozen=True)
class RuleVerdict:
    """The rule's outcome and each clause's, with the readings they were settled on."""

    outcome: RuleOutcome
    clauses: tuple[ClauseVerdict, ...]
    readings: RuleReadings | None = None

    def settles(self, rate: RateReading) -> bool:
        """Whether a settled rate clause decided ``rate``: under ``keep`` any rate it reads observed past its margin
        (the run shows it breaks the clause), under ``switch`` one of the rates it read."""
        for verdict in self.clauses:
            clause = verdict.clause
            if not isinstance(clause, RateDrop) or (clause.reading is not None and not rate.answers(clause.reading)):
                continue
            if verdict.outcome == "keep" and rate.drop_bounds()[0] > clause.max_points:
                return True
            read = {r.measure for r in self.readings.read_by(clause)} if self.readings is not None else set()
            if verdict.outcome == "switch" and rate.measure is not None and rate.measure in read:
                return True
        return False

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

    Args:
        clause: The clause.
        candidate: The arm it is read on, by name; ``it`` where the sentence names several.
        readings: What the clause is read against, whose rate names a named reading is said by; None to say it
            from its key.

    Returns:
        The clause, lower case, with no closing stop.
    """
    if isinstance(clause, CostLower):
        return f"{candidate} costs less per reply"
    which = "pass" if clause.reading is None else _reading_words(clause.reading, readings)
    return f"no {which} rate drops by more than {format_number(clause.max_points)} points"


def rule_statement(
    rule: DecisionRule, *, control: str, candidates: Sequence[str], readings: RuleReadings | None = None
) -> str:
    """The rule in the sentence the page opens its Decision section with, the arms by name.

    ``Switch to v2 only if no pass rate drops by more than 2 points and v2 costs less per reply. Otherwise stay on
    v1.`` With several arms the switch names each, and a cost clause says ``it``.

    Args:
        rule: The declared rule.
        control: The control, by name.
        candidates: Every other arm, by name.
        readings: The first arm's readings, whose rate names a named reading is said by.

    Returns:
        Two sentences: when to switch, and otherwise.
    """
    subject = candidates[0] if len(candidates) == 1 else "it"
    target = " or ".join(candidates) if candidates else "another arm"
    said = [clause_statement(clause, candidate=subject, readings=readings) for clause in rule.clauses]
    conditions = said[0] if len(said) == 1 else f"{', '.join(said[:-1])} and {said[-1]}"
    return f"Switch to {target} only if {conditions}. Otherwise stay on {control}."


def evaluate_rule(rule: DecisionRule, readings: RuleReadings, *, control: str, candidate: str) -> RuleVerdict:
    """Settle each clause of ``rule`` on ``readings``, then the rule.

    Args:
        rule: The declared rule.
        readings: The control's and the arm's figures.
        control: The control, by the name the page gives it, which a reason says.
        candidate: The arm, likewise.

    Returns:
        The rule's outcome and each clause's, with reasons naming the arms.
    """
    verdicts = tuple(_clause(clause, readings, control, candidate) for clause in rule.clauses)
    outcomes = {verdict.outcome for verdict in verdicts}
    outcome: RuleOutcome = (
        "keep" if "keep" in outcomes else "switch" if outcomes == {"switch"} and verdicts else "unsettled"
    )
    return RuleVerdict(outcome=outcome, clauses=verdicts, readings=readings)


def _clause(clause: DecisionClause, readings: RuleReadings, control: str, candidate: str) -> ClauseVerdict:
    """One clause's verdict: keep on what the run observed, switch only on an interval over enough cases."""
    statement = clause_statement(clause, candidate=candidate, readings=readings)
    if isinstance(clause, CostLower):
        outcome, reason = _cost_clause(readings, control, candidate)
        return ClauseVerdict(clause, outcome, reason, statement)
    outcome, reason, rate = _rate_clause(clause, readings, control, candidate)
    return ClauseVerdict(clause, outcome, reason, statement, rate=rate)


def _points(value: float) -> str:
    """A change in points as the page states one: to one decimal, a whole number whole."""
    return format_number(round(value, 1))


def _kept_on_counts(rate: RateReading, least: float, margin: float, control: str, candidate: str) -> str:
    """A rate kept on what the run observed, as counts and the margin in words: ``v2 passed 14 of 60 to v1's 29, a
    drop of 25 points against a margin of 10``. The control's total is said only where it differs."""
    theirs = f"{rate.candidate_passed} of {rate.candidate_n}"
    ours = (
        str(rate.control_passed) if rate.control_n == rate.candidate_n else f"{rate.control_passed} of {rate.control_n}"
    )
    passed = f"passed {theirs}" if rate.name == "the pass rate" else f"passed {rate.name} on {theirs}"
    bound = "at least " if rate.control_remaining or rate.candidate_remaining else ""
    return (
        f"{candidate} {passed} to {control}'s {ours}, a drop of {bound}{_points(least)} points against a margin of "
        f"{format_number(margin)}"
    )


def _rate_clause(
    clause: RateDrop, readings: RuleReadings, control: str, candidate: str
) -> tuple[RuleOutcome, str, RateReading | None]:
    """A rate clause: keep on an observed drop past the margin, switch on intervals within it over enough cases."""
    margin = clause.max_points
    rates = readings.read_by(clause)
    if not rates:
        if clause.reading is None:
            return "unsettled", "no pass rate was measured on both arms", None
        return "unsettled", f"{_reading_words(clause.reading, readings)} was not measured on both arms", None
    points = format_number(margin)
    for rate in rates:
        least, _ = rate.drop_bounds()
        if least > margin:
            return "keep", _kept_on_counts(rate, least, margin, control, candidate), rate
    fewest = readings.fewest_cases()
    if fewest is None:
        return "unsettled", "the usable cases were not counted on both arms", None
    if fewest < _FLOOR:
        return (
            "unsettled",
            f"{candidate}'s drop is not shown to stay within {points} points ({fewest} cases are too few for the "
            f"interval that would show it, which is estimated only from {_FLOOR} independent cases)",
            None,
        )
    for rate in rates:
        # The first rate is the row the reason sits on; any other is named.
        drop = f"{candidate}'s drop" if rate is rates[0] else f"{candidate}'s drop on {rate.name}"
        unshown = f"{drop} is not shown to stay within {points} points"
        worst = rate.worst_drop()
        if worst is None and readings.predates_bounded:
            return "unsettled", f"{unshown} (this bundle predates the interval a switch needs)", None
        if worst is None:
            return "unsettled", f"{unshown} (no interval was computed on {rate.name})", None
        if worst > margin:
            return "unsettled", f"{unshown} (it could be as much as {_points(worst)})", None
        if rate.drop_bounds()[1] > margin:
            return "unsettled", f"{unshown} (the lost trials could still take it past)", None
    widest = max(rate.worst_drop() or 0.0 for rate in rates)
    at_most = f" (it is at most {_points(widest)})" if widest > 0 else ""
    return "switch", f"{candidate}'s drop is shown to stay within {points} points{at_most}", None


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

    Args:
        bundle: The campaign's evidence.
        control: The control's variant key.
        candidate: The candidate's variant key.
        question_id: The question the rule is stated in, whose family alone the intervals are read from; None for
            the campaign-wide family.

    Returns:
        The readings.
    """
    boards = {board.variant_key: board for board in bundle.scoreboard}
    if control not in boards or candidate not in boards:
        return RuleReadings(rates=())
    ours, theirs = boards[control], boards[candidate]
    lost_control, lost_candidate = sum(ours.lost.values()), sum(theirs.lost.values())
    rates = [
        RateReading(
            "the pass rate",
            ours.passed,
            ours.usable,
            theirs.passed,
            theirs.usable,
            lost_control,
            lost_candidate,
            interval=_bounded_better_way(
                _contrast(bundle, "trial", TRIAL_PASS_NAME, control, candidate, bounded=True, question_id=question_id)
            ),
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
                RateReading(
                    heading,
                    *by_arm[control],
                    *by_arm[candidate],
                    lost_control,
                    lost_candidate,
                    interval=_bounded_better_way(
                        _contrast(
                            bundle,
                            "measure",
                            _compared_as(name),
                            control,
                            candidate,
                            bounded=True,
                            question_id=question_id,
                        )
                    ),
                    measure=name,
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
                RateReading(
                    humanize("dimension", judged.name),
                    *counts[control],
                    *counts[candidate],
                    lost_control,
                    lost_candidate,
                    interval=_bounded_better_way(
                        _contrast(
                            bundle, "judged", judged.name, control, candidate, bounded=True, question_id=question_id
                        )
                    ),
                    measure=judged.name,
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


__all__ = [
    "PROSE_RULE_LINE",
    "ClauseVerdict",
    "RateReading",
    "RuleOutcome",
    "RuleReadings",
    "RuleVerdict",
    "clause_statement",
    "evaluate_rule",
    "keep_is_the_default",
    "prose_ruled",
    "rule_readings",
    "rule_statement",
    "ruled_question",
]
