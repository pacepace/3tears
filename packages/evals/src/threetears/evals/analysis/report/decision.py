"""A declared decision rule, read clause by clause: which clauses the evidence settles, and which way.

A campaign states its rule in its question's words (``Decision rule: switch … only if no dimension drops by more than
10 points AND the cost per turn is lower; otherwise keep …``). :func:`parse_decision_rule` reads the clauses it knows;
:func:`evaluate_rule` settles each one or says why it cannot.

**Keep may settle on what the run observed; switch needs sampling evidence.** The rule's default is the control, so
showing the candidate breaks a clause is enough not to switch. Showing it holds a clause is a claim about cases not
yet run, so it needs an interval:

- **A pass rate, keep.** The candidate's drop from the control is bounded by giving every lost trial its best and its
  worst outcome. A drop past the margin even at best settles ``keep``.
- **A pass rate, switch.** Every arm has at least the case floor of usable cases
  (:data:`~threetears.evals.analysis.report.verdict.MIN_CASES`), and for every rate (the overall pass rate and each
  check's) the contrast's bounded interval (the candidate minus the control, ``bounded_interval``: the bounded test's
  on the rate's 0-1 range at its family's level) bounds the drop at or below the margin, as does the bound over lost
  trials. Anything else is ``unsettled``, with the reason in plain words.

  **Switch never reads the t interval the page's contrasts print.** "No worse by more than the margin" is a
  non-inferiority claim, and on a rate's coarse per-case shares the t interval makes it too often: simulated at the
  margin, it let switch through up to 12% of the time against the 2.5% it states, where the bounded interval holds its
  rate at every n (``docs/measuring-soundly.md``). A guardrail's ``held`` reads the same interval for the same reason.
  Its reason names it "the interval a switch reads", since the page's contrasts print the narrower t interval.
  A bundle before shape 55 carries no bounded interval, so its rate clause cannot settle switch; it never falls back
  to the t interval.
- **Cost, keep.** A candidate observed to cost no less per reply settles ``keep``.
- **Cost, switch.** Every arm is at the case floor and the cost contrast's interval lies wholly below zero.
  Otherwise ``unsettled``: the cost difference did not separate.
- **The rule's own logic decides.** "Switch only if A and B" keeps the control the moment one clause settles keep,
  switches only when every clause settles switch, and is otherwise unsettled.

Nothing here estimates the size of a difference: under the case floor a settled ``keep`` is read on the counts alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME, AnalysisContextBundle, FamilyComparison
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.stats import SMALL_N_BAND_FLOOR
from threetears.evals.analysis.style import measure_name
from threetears.evals.kernel.analysis_measures import MeasureSummary
from threetears.evals.kernel.metrics import ACCURACY_MEASURE, MATCH_MEASURE, goal_check_of

#: What a clause reads: a pass rate that must not drop by more than a margin, or a cost that must be lower.
ClauseKind = Literal["rate_drop", "cost_lower", "unread"]

#: Where a clause, or the rule, came out.
RuleOutcome = Literal["keep", "switch", "unsettled"]

_RULE = re.compile(r"decision rule:\s*(?P<body>.+)", re.I | re.S)
_ONLY_IF = re.compile(r"only if (?P<conditions>.+?)(?:;|\.\s|\.$|, otherwise|$)", re.I | re.S)
_RATE = re.compile(
    r"no [\w/ -]*?(?:dimension|check|rate)s? drops? (?:by )?(?:more than|over|>)\s*(?P<points>\d+(?:\.\d+)?)", re.I
)
_COST = re.compile(r"cost(?: per (?:turn|reply|trial))? is lower", re.I)

#: The fewest usable cases on every arm before a clause can settle ``switch``: the page's case floor.
_FLOOR = SMALL_N_BAND_FLOOR

#: The measure the scoreboard's cost per reply is the mean of, whose contrast decides a cost clause.
_REPLY_COST = "production_replicating_cost"


@dataclass(frozen=True)
class RuleClause:
    """One clause of a decision rule, as written and as read."""

    text: str
    kind: ClauseKind
    margin_points: float | None = None


@dataclass(frozen=True)
class DecisionRule:
    """A rule that switches from the control only if every clause holds."""

    text: str
    clauses: tuple[RuleClause, ...]


@dataclass(frozen=True)
class RateReading:
    """One pass rate on both arms: passed of measured, how many trials each could still add, and its interval.

    ``interval`` is the contrast's bounded interval on the candidate minus the control (``bounded_interval``), as a
    share (0 to 1) with higher better: the interval a switch reads. None when the bundle computed none for this rate.
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
        """The candidate's drop from the control in points, at best and at worst for the candidate."""

        def bounds(passed: int, n: int, remaining: int) -> tuple[float, float]:
            total = n + remaining
            return (passed / total, (passed + remaining) / total) if total else (0.0, 1.0)

        control_low, control_high = bounds(self.control_passed, self.control_n, self.control_remaining)
        candidate_low, candidate_high = bounds(self.candidate_passed, self.candidate_n, self.candidate_remaining)
        return 100 * (control_low - candidate_high), 100 * (control_high - candidate_low)

    def worst_drop(self) -> float | None:
        """The largest drop in points the interval allows; None without an interval."""
        return None if self.interval is None else -100 * self.interval[0]


@dataclass(frozen=True)
class RuleReadings:
    """What a rule is read against for one candidate arm.

    ``control_cases`` and ``candidate_cases`` are each arm's usable cases, None when not counted; ``cost_interval``
    is the bundle's contrast interval on the candidate's cost per reply minus the control's, None when none was
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


@dataclass(frozen=True)
class ClauseVerdict:
    """One clause settled, or not, with the reason in plain words, and the rate that settled it ``keep``."""

    clause: RuleClause
    outcome: RuleOutcome
    reason: str
    rate: RateReading | None = None


@dataclass(frozen=True)
class RuleVerdict:
    """The rule's outcome and each clause's, with the readings they were settled on."""

    outcome: RuleOutcome
    clauses: tuple[ClauseVerdict, ...]
    readings: RuleReadings | None = None

    def settles(self, rate: RateReading) -> bool:
        """Whether a settled rate clause decided ``rate``: under ``keep`` any rate observed past the clause's margin
        (the run shows it breaks the clause, whether or not the rule read it), under ``switch`` one of the rates it
        read."""
        read = {reading.measure for reading in self.readings.rates} if self.readings is not None else set()
        for verdict in self.clauses:
            margin = verdict.clause.margin_points
            if verdict.clause.kind != "rate_drop" or margin is None:
                continue
            if verdict.outcome == "keep" and rate.drop_bounds()[0] > margin:
                return True
            if verdict.outcome == "switch" and rate.measure is not None and rate.measure in read:
                return True
        return False

    def settles_cost(self) -> bool:
        """Whether a cost clause settled, either way."""
        return any(verdict.clause.kind == "cost_lower" and verdict.outcome != "unsettled" for verdict in self.clauses)


def parse_decision_rule(text: str) -> DecisionRule | None:
    """The decision rule a question states, clause by clause; None when it states none.

    Args:
        text: The declared question.

    Returns:
        The rule; a clause this reader does not know is kept as ``unread``, never dropped.
    """
    return split_decision_rule(text)[1]


def split_decision_rule(text: str) -> tuple[str, DecisionRule | None]:
    """A declared question split where its decision rule begins: the question's own words, and the rule.

    The page lays each part out where it is read — the question on the first screen, the rule at the head of the
    Decision section — so the split is the parser's own boundary (the ``Decision rule:`` statement), never a
    boundary of the page's choosing.

    Args:
        text: The declared question.

    Returns:
        ``(asked, rule)``: the words before the rule statement, and the rule as :func:`parse_decision_rule` reads
        it. The whole text and None where no rule parses.
    """
    if (stated := _RULE.search(text)) is None:
        return text, None
    body = stated["body"].strip()
    conditions = _ONLY_IF.search(body)
    if conditions is None:
        return text, None
    parts = re.split(r"\s+AND\s+", conditions["conditions"].strip())
    if len(parts) == 1:
        parts = re.split(r"\s+and\s+(?=the |no |every )", parts[0])
    clauses = []
    for part in parts:
        if rate := _RATE.search(part):
            clauses.append(RuleClause(text=part, kind="rate_drop", margin_points=float(rate["points"])))
        elif _COST.search(part):
            clauses.append(RuleClause(text=part, kind="cost_lower"))
        else:
            clauses.append(RuleClause(text=part, kind="unread"))
    return text[: stated.start()].strip(), DecisionRule(text=body, clauses=tuple(clauses))


def evaluate_rule(rule: DecisionRule, readings: RuleReadings) -> RuleVerdict:
    """Settle each clause of ``rule`` on ``readings``, then the rule.

    Args:
        rule: The parsed rule.
        readings: The control's and the candidate's figures.

    Returns:
        The rule's outcome and each clause's, with reasons.
    """
    verdicts = tuple(_clause(clause, readings) for clause in rule.clauses)
    outcomes = {verdict.outcome for verdict in verdicts}
    outcome: RuleOutcome = (
        "keep" if "keep" in outcomes else "switch" if outcomes == {"switch"} and verdicts else "unsettled"
    )
    return RuleVerdict(outcome=outcome, clauses=verdicts, readings=readings)


def _clause(clause: RuleClause, readings: RuleReadings) -> ClauseVerdict:
    """One clause's verdict: keep on what the run observed, switch only on an interval over enough cases."""
    if clause.kind == "cost_lower":
        return _cost_clause(clause, readings)
    if clause.kind == "rate_drop" and clause.margin_points is not None:
        return _rate_clause(clause, clause.margin_points, readings)
    return ClauseVerdict(clause, "unsettled", "the page cannot read this clause")


def _rate_clause(clause: RuleClause, margin: float, readings: RuleReadings) -> ClauseVerdict:
    """A rate clause: keep on an observed drop past the margin, switch on intervals within it over enough cases."""
    if not readings.rates:
        return ClauseVerdict(clause, "unsettled", "no pass rate was measured on both arms")
    points = format_number(margin)
    for rate in readings.rates:
        least, _ = rate.drop_bounds()
        if least > margin:
            bound = "at least " if rate.control_remaining or rate.candidate_remaining else ""
            return ClauseVerdict(
                clause, "keep", f"{rate.name} drops {bound}{format_number(round(least))} points", rate=rate
            )
    fewest = readings.fewest_cases()
    if fewest is None:
        return ClauseVerdict(clause, "unsettled", "the usable cases were not counted on both arms")
    if fewest < _FLOOR:
        return ClauseVerdict(
            clause,
            "unsettled",
            f"{fewest} cases cannot show the drop stays within {points} points, which takes an interval estimated "
            f"only from {_FLOOR} independent cases",
        )
    for rate in readings.rates:
        worst = rate.worst_drop()
        if worst is None and readings.predates_bounded:
            return ClauseVerdict(clause, "unsettled", "this bundle predates the interval a switch needs")
        if worst is None:
            return ClauseVerdict(clause, "unsettled", f"no interval was computed on {rate.name}")
        if worst > margin:
            return ClauseVerdict(
                clause,
                "unsettled",
                f"the interval a switch reads allows {rate.name} to drop up to {format_number(round(worst))} points",
            )
        if rate.drop_bounds()[1] > margin:
            return ClauseVerdict(clause, "unsettled", f"the lost trials could still take {rate.name} past the margin")
    return ClauseVerdict(clause, "switch", f"the interval keeps every pass rate within {points} points")


def _cost_clause(clause: RuleClause, readings: RuleReadings) -> ClauseVerdict:
    """A cost clause: keep on a cost observed no lower, switch on an interval wholly below zero over enough cases."""
    control, candidate = readings.control_cost, readings.candidate_cost
    if control is None or candidate is None:
        return ClauseVerdict(clause, "unsettled", "cost per reply was not measured on both arms")
    if candidate > control:
        return ClauseVerdict(clause, "keep", "the candidate costs more per reply")
    if candidate == control:
        return ClauseVerdict(clause, "keep", "the candidate costs no less per reply")
    fewest = readings.fewest_cases()
    if fewest is None:
        return ClauseVerdict(clause, "unsettled", "the usable cases were not counted on both arms")
    if fewest < _FLOOR:
        return ClauseVerdict(
            clause,
            "unsettled",
            f"{fewest} cases cannot show the cost is lower, which takes an interval estimated only from {_FLOOR} "
            "independent cases",
        )
    if readings.cost_interval is None or readings.cost_interval[1] >= 0:
        return ClauseVerdict(clause, "unsettled", "the cost difference did not separate")
    return ClauseVerdict(clause, "switch", "the candidate costs less per reply")


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


def _held(summary: MeasureSummary) -> int | None:
    """How many of a pass/fail reading's observations passed: a rate's count, or a goal check's mean times its n.

    A goal check reaches the bundle as a mean of 0s and 1s, not a rate, and is the same proportion: its count is the
    mean times the observations, read only where that is whole. None for any other reading.
    """
    if summary.rate is not None and summary.n_true is not None:
        return summary.n_true
    if goal_check_of(summary.name) is None or summary.mean is None or summary.n <= 0:
        return None
    count = summary.mean * summary.n
    return round(count) if abs(count - round(count)) <= 1e-6 else None


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

    The pass rates are every trial's, then each check's pass rate the arms' cells carry (a goal check's as its mean
    times its observations, the count it is: :func:`_held`); each arm's lost trials are
    trials it could still add. Each rate carries its contrast's bounded interval, the one a switch reads, where the
    bundle computed one (the overall pass rate's is its ``trial`` contrast). Cost is each arm's cost per reply, with
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
    measures: dict[str, dict[str, tuple[int, int]]] = {}
    for cell in bundle.cell_measures:
        for summary in cell.measures.measures:
            if (held := _held(summary)) is None or summary.name.startswith("classifier:"):
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
    "ClauseKind",
    "ClauseVerdict",
    "DecisionRule",
    "RateReading",
    "RuleClause",
    "RuleOutcome",
    "RuleReadings",
    "RuleVerdict",
    "evaluate_rule",
    "parse_decision_rule",
    "rule_readings",
    "split_decision_rule",
]
