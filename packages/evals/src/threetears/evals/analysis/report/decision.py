"""A declared decision rule, read clause by clause: which clauses the evidence settles, and which way.

A campaign states its rule in its question's words (``Decision rule: switch … only if no dimension drops by more than
10 points AND the cost per turn is lower; otherwise keep …``). :func:`parse_decision_rule` reads the clauses it knows;
:func:`evaluate_rule` settles each one or says why it cannot:

- **A measured quantity needs no case floor.** Cost per reply is what the trials spent: lower or not is read as
  measured.
- **A pass rate is settled by a bound.** The candidate's drop from the control is bounded by giving every trial
  still to run (lost ones included) its best and its worst outcome. A drop over the margin even at best settles
  ``keep``; one within the margin even at worst settles ``switch``; anything between is unsettled.
- **The rule's own logic decides.** "Switch only if A and B" keeps the control the moment one clause settles keep,
  switches only when every clause settles switch, and is otherwise unsettled.

The case floor (:data:`~threetears.evals.analysis.report.verdict.MIN_CASES`) still governs estimating a magnitude:
nothing here estimates one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.numbers import format_number

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
    """One pass rate on both arms: passed of measured, and how many trials each could still add."""

    name: str
    control_passed: int
    control_n: int
    candidate_passed: int
    candidate_n: int
    control_remaining: int = 0
    candidate_remaining: int = 0

    def drop_bounds(self) -> tuple[float, float]:
        """The candidate's drop from the control in points, at best and at worst for the candidate."""

        def bounds(passed: int, n: int, remaining: int) -> tuple[float, float]:
            total = n + remaining
            return (passed / total, (passed + remaining) / total) if total else (0.0, 1.0)

        control_low, control_high = bounds(self.control_passed, self.control_n, self.control_remaining)
        candidate_low, candidate_high = bounds(self.candidate_passed, self.candidate_n, self.candidate_remaining)
        return 100 * (control_low - candidate_high), 100 * (control_high - candidate_low)


@dataclass(frozen=True)
class RuleReadings:
    """What a rule is read against for one candidate arm."""

    rates: tuple[RateReading, ...]
    control_cost: float | None = None
    candidate_cost: float | None = None


@dataclass(frozen=True)
class ClauseVerdict:
    """One clause settled, or not, with the reason in plain words."""

    clause: RuleClause
    outcome: RuleOutcome
    reason: str


@dataclass(frozen=True)
class RuleVerdict:
    """The rule's outcome and each clause's."""

    outcome: RuleOutcome
    clauses: tuple[ClauseVerdict, ...]


def parse_decision_rule(text: str) -> DecisionRule | None:
    """The decision rule a question states, clause by clause; None when it states none.

    Args:
        text: The declared question.

    Returns:
        The rule; a clause this reader does not know is kept as ``unread``, never dropped.
    """
    if (stated := _RULE.search(text)) is None:
        return None
    body = stated["body"].strip()
    conditions = _ONLY_IF.search(body)
    if conditions is None:
        return None
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
    return DecisionRule(text=body, clauses=tuple(clauses))


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
    return RuleVerdict(outcome=outcome, clauses=verdicts)


def _clause(clause: RuleClause, readings: RuleReadings) -> ClauseVerdict:
    """One clause's verdict."""
    if clause.kind == "cost_lower":
        control, candidate = readings.control_cost, readings.candidate_cost
        if control is None or candidate is None:
            return ClauseVerdict(clause, "unsettled", "cost per reply was not measured on both arms")
        if candidate < control:
            return ClauseVerdict(clause, "switch", "the candidate costs less per reply")
        return ClauseVerdict(clause, "keep", "the candidate costs more per reply")
    if clause.kind == "rate_drop" and clause.margin_points is not None:
        margin = clause.margin_points
        if not readings.rates:
            return ClauseVerdict(clause, "unsettled", "no pass rate was measured on both arms")
        for rate in readings.rates:
            least, _ = rate.drop_bounds()
            if least > margin:
                return ClauseVerdict(clause, "keep", f"{rate.name} drops at least {format_number(round(least))} points")
        if all(rate.drop_bounds()[1] <= margin for rate in readings.rates):
            return ClauseVerdict(clause, "switch", f"no pass rate can drop more than {format_number(margin)} points")
        open_rates = [rate.name for rate in readings.rates if rate.drop_bounds()[1] > margin]
        return ClauseVerdict(clause, "unsettled", f"more cases could still decide {open_rates[0]}")
    return ClauseVerdict(clause, "unsettled", "the page cannot read this clause")


def rule_readings(bundle: AnalysisContextBundle, control: str, candidate: str) -> RuleReadings:
    """The figures a rule is read against for one candidate arm, from the bundle alone.

    The pass rates are every trial's, then each check's pass rate the arms' cells carry; each arm's lost trials are
    trials it could still add. Cost is each arm's cost per reply.

    Args:
        bundle: The campaign's evidence.
        control: The control's variant key.
        candidate: The candidate's variant key.

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
        )
    ]
    measures: dict[str, dict[str, tuple[int, int]]] = {}
    for cell in bundle.cell_measures:
        for summary in cell.measures.measures:
            if summary.rate is None or summary.n_true is None or summary.name.startswith("classifier:"):
                continue
            measures.setdefault(summary.name, {})[cell.variant_key] = (summary.n_true, summary.n)
    for name, by_arm in sorted(measures.items()):
        if control in by_arm and candidate in by_arm:
            descriptor = bundle.measure_catalog.get(name)
            heading = (descriptor.reader_name if descriptor is not None else None) or name
            rates.append(RateReading(heading, *by_arm[control], *by_arm[candidate], lost_control, lost_candidate))
    return RuleReadings(rates=tuple(rates), control_cost=ours.reply_cost_usd, candidate_cost=theirs.reply_cost_usd)


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
]
