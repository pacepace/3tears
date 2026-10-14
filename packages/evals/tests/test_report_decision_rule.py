"""A declared decision rule, read clause by clause, on the G2 campaign's own numbers.

G2 (2026-10) declared: "switch Kairo's model only if no dimension drops by more than 10 points AND the cost per turn
is lower; otherwise keep deepseek-v4.1-flash", and ran 4 cases × 3 repeats: production passed 10 of 12, the cheaper
candidate 2 of 12, at a cost per reply of 0.003092 against 0.003301. Its memo called the question unanswerable,
because four cases are under the floor for estimating a rate. Neither clause needs that floor: a measured cost is read
as measured, and a pass rate is settled by a bound once the losing arm cannot close the gap with the trials left.
Both clauses settle ``keep``.
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
    # Four cases is under the floor for estimating a magnitude, never for a measured cost or a bounded rate.
    assert verdict.clauses[0].reason == "the pass rate drops at least 67 points"


def test_a_rate_clause_the_remaining_trials_could_still_decide_is_unsettled() -> None:
    rule = parse_decision_rule(G2_QUESTION)
    assert rule is not None
    close = RuleReadings(
        rates=(RateReading("the pass rate", 10, 12, 8, 10, 0, 2),), control_cost=2.0, candidate_cost=1.0
    )
    verdict = evaluate_rule(rule, close)
    assert [clause.outcome for clause in verdict.clauses] == ["unsettled", "switch"]
    assert verdict.outcome == "unsettled"
    settled = RuleReadings(rates=(RateReading("the pass rate", 12, 12, 12, 12),), control_cost=2.0, candidate_cost=1.0)
    assert evaluate_rule(rule, settled).outcome == "switch"


async def test_the_verdict_line_reads_the_rules_outcome_in_plain_words(g2_ruled: tuple[Result, Report]) -> None:
    _, report = g2_ruled
    (verdict,) = [b for b in report.blocks if isinstance(b, DisclosureBlock) and b.source == "verdict"]
    assert verdict.text == (
        "Keep production: the pass rate drops at least 67 points, and cheaper candidate costs more per reply."
    )
    (table,) = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "decision"]
    outcomes = [row["outcome"] for row in table.values() if row["outcome"] is not None]
    assert outcomes == ["settled: keep", "settled: keep"]
    assert {"reading": "cost per reply", "control": "$0.003092", "candidate": "$0.003301"}.items() <= next(
        row for row in table.values() if row["reading"] == "cost per reply"
    ).items()
