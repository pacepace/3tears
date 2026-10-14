"""A declared decision rule, read clause by clause, on the G2 campaign's own numbers.

G2 (2026-10) declared: "switch Kairo's model only if no dimension drops by more than 10 points AND the cost per turn
is lower; otherwise keep deepseek-v4.1-flash", and ran 4 cases × 3 repeats: production passed 10 of 12, the cheaper
candidate 2 of 12, at a cost per reply of 0.003092 against 0.003301. Its memo called the question unanswerable,
because four cases are under the floor for estimating a rate. Neither clause needs that floor to settle ``keep``: a
cost observed higher, and a pass rate whose observed drop is past the margin, are enough not to switch. Settling
``switch`` is another matter: it needs every arm at the case floor and an interval that bounds the clause.
"""

from __future__ import annotations

from threetears.evals import Result
from threetears.evals.analysis import Report
from threetears.evals.analysis.report import DisclosureBlock, TableBlock
from threetears.evals.analysis.report.decision import RateReading, RuleReadings, evaluate_rule, parse_decision_rule
import pytest

from packages.evals.tests.g2_support import G2_QUESTION, g2_ruled_report


@pytest.fixture
async def g2_ruled() -> tuple[Result, Report]:
    """The G2 numbers through the quick path, with the G2 decision rule declared before the run."""
    return await g2_ruled_report()


def test_the_g2_rule_keeps_production_on_both_clauses() -> None:
    rule = parse_decision_rule(G2_QUESTION)
    assert rule is not None and [clause.kind for clause in rule.clauses] == ["rate_drop", "cost_lower"]
    readings = RuleReadings(
        rates=(RateReading("the pass rate", 10, 12, 2, 12),), control_cost=0.003092, candidate_cost=0.003301
    )
    verdict = evaluate_rule(rule, readings)
    assert [clause.outcome for clause in verdict.clauses] == ["keep", "keep"]
    assert verdict.outcome == "keep"
    # Four cases is under the floor for estimating a magnitude, never for keeping on what the run observed. Nothing
    # remains to run, so the drop is stated as observed, without "at least".
    assert verdict.clauses[0].reason == "the pass rate drops 67 points"
    assert verdict.clauses[0].rate is readings.rates[0]


def test_a_rate_clause_the_remaining_trials_could_still_decide_is_unsettled() -> None:
    rule = parse_decision_rule(G2_QUESTION)
    assert rule is not None
    close = RuleReadings(
        rates=(RateReading("the pass rate", 10, 12, 8, 10, 0, 2, interval=(-0.05, 0.02)),),
        control_cost=2.0,
        candidate_cost=1.0,
        control_cases=6,
        candidate_cases=6,
        cost_interval=(-1.5, -0.5),
    )
    verdict = evaluate_rule(rule, close)
    assert [clause.outcome for clause in verdict.clauses] == ["unsettled", "switch"]
    assert verdict.outcome == "unsettled"
    assert verdict.clauses[0].reason == "the lost trials could still take the pass rate past the margin"
    # Equal rates switch only on an interval over enough cases. The bare counts (no case count, no interval) once
    # settled switch here: that was the defect.
    settled = RuleReadings(
        rates=(RateReading("the pass rate", 12, 12, 12, 12, interval=(-0.04, 0.04)),),
        control_cost=2.0,
        candidate_cost=1.0,
        control_cases=6,
        candidate_cases=6,
        cost_interval=(-1.5, -0.5),
    )
    assert evaluate_rule(rule, settled).outcome == "switch"
    bare = RuleReadings(rates=(RateReading("the pass rate", 12, 12, 12, 12),), control_cost=2.0, candidate_cost=1.0)
    assert evaluate_rule(rule, bare).outcome == "unsettled"


def test_equal_rates_on_too_few_cases_are_unsettled_not_switch() -> None:
    rule = parse_decision_rule(G2_QUESTION)
    assert rule is not None
    equal = RuleReadings(
        rates=(RateReading("the pass rate", 10, 12, 10, 12, interval=(-0.02, 0.02)),),
        control_cost=2.0,
        candidate_cost=1.0,
        control_cases=4,
        candidate_cases=4,
        cost_interval=(-1.5, -0.5),
    )
    verdict = evaluate_rule(rule, equal)
    assert [clause.outcome for clause in verdict.clauses] == ["unsettled", "unsettled"]
    assert verdict.outcome == "unsettled"
    assert verdict.clauses[0].reason == "4 cases cannot show the drop stays within 10 points (a rate needs 5)"
    assert verdict.clauses[1].reason == "4 cases cannot show the cost is lower (a difference needs 5)"


def test_a_floor_met_run_with_a_tight_interval_switches() -> None:
    rule = parse_decision_rule(G2_QUESTION)
    assert rule is not None
    tight = RuleReadings(
        rates=(RateReading("the pass rate", 50, 60, 49, 60, interval=(-0.08, 0.05)),),
        control_cost=0.0031,
        candidate_cost=0.0021,
        control_cases=20,
        candidate_cases=20,
        cost_interval=(-0.0014, -0.0006),
    )
    verdict = evaluate_rule(rule, tight)
    assert [clause.outcome for clause in verdict.clauses] == ["switch", "switch"]
    assert verdict.outcome == "switch"
    # The same run with an interval reaching past the margin is unsettled, and says how far it reaches.
    wide = RuleReadings(
        rates=(RateReading("the pass rate", 50, 60, 49, 60, interval=(-0.23, 0.2)),),
        control_cost=0.0031,
        candidate_cost=0.0021,
        control_cases=20,
        candidate_cases=20,
        cost_interval=(-0.0014, -0.0006),
    )
    wide_verdict = evaluate_rule(rule, wide)
    assert wide_verdict.outcome == "unsettled"
    assert wide_verdict.clauses[0].reason == "the interval allows the pass rate to drop up to 23 points"


def test_a_cost_lower_by_a_point_estimate_only_is_unsettled() -> None:
    rule = parse_decision_rule(G2_QUESTION)
    assert rule is not None
    readings = RuleReadings(
        rates=(RateReading("the pass rate", 50, 60, 50, 60, interval=(-0.05, 0.05)),),
        control_cost=0.0031,
        candidate_cost=0.0029,
        control_cases=20,
        candidate_cases=20,
        cost_interval=(-0.0006, 0.0002),
    )
    verdict = evaluate_rule(rule, readings)
    assert [clause.outcome for clause in verdict.clauses] == ["switch", "unsettled"]
    assert verdict.clauses[1].reason == "the cost difference did not separate"
    assert verdict.outcome == "unsettled"
    # A cost observed higher still settles keep, on any number of cases.
    dearer = RuleReadings(rates=readings.rates, control_cost=0.0029, candidate_cost=0.0031)
    assert evaluate_rule(rule, dearer).clauses[1].outcome == "keep"


async def test_the_verdict_line_reads_the_rules_outcome_in_plain_words(g2_ruled: tuple[Result, Report]) -> None:
    _, report = g2_ruled
    (verdict,) = [b for b in report.blocks if isinstance(b, DisclosureBlock) and b.source == "verdict"]
    # Plain counts and both costs, each arm by its model: "cheaper candidate cost more" would read as a contradiction.
    assert verdict.text == (
        "Keep production: deepseek-v4-flash passed 2 of 12 trials to deepseek-v4.1-flash's 10 of 12, "
        "and cost more per reply ($0.003301 vs $0.003092)."
    )
    (table,) = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "decision"]
    # One candidate: each arm heads its own column by name.
    assert [column.header for column in table.columns] == [
        "Clause",
        "Reading",
        "production",
        "cheaper candidate",
        "Outcome",
    ]
    outcomes = [row["outcome"] for row in table.values() if row["outcome"] is not None]
    assert outcomes == ["settled: keep", "settled: keep"]
    assert {"reading": "cost per reply", "control": "$0.003092", "candidate": "$0.003301"}.items() <= next(
        row for row in table.values() if row["reading"] == "cost per reply"
    ).items()
