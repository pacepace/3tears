"""A declared decision rule, read clause by clause, on the G2 campaign's own numbers.

G2 (2026-10) declared: "switch Kairo's model only if no dimension drops by more than 10 points AND the cost per turn
is lower; otherwise keep deepseek-v4.1-flash", and ran 4 cases × 3 repeats: production passed 10 of 12, the cheaper
candidate 2 of 12, at a cost per reply of 0.003092 against 0.003301. Its memo called the question unanswerable,
because four cases are under the floor for estimating a rate. Neither clause needs that floor to settle ``keep``: a
cost observed higher, and a pass rate whose observed drop is past the margin, are enough not to switch. Settling
``switch`` is another matter: it needs every arm at the case floor and an interval that bounds the clause.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from threetears.evals import Answer, Eval, Result
from threetears.evals.analysis import AnalysisContextBundle, Report, assemble_context_bundle, get_campaign
from threetears.evals.analysis.report import DisclosureBlock, TableBlock
from threetears.evals.analysis.report.decision import (
    RateReading,
    RuleReadings,
    evaluate_rule,
    parse_decision_rule,
    rule_readings,
)
from threetears.evals.analysis.style import lint_prose
from threetears.evals.kernel.declaration import Question
import pytest

from packages.evals.tests.g2_support import G2_QUESTION, g2_ruled_report
from packages.evals.tests.off_loop import off_loop

#: A bundle the engine stored before shape 55: a contrast for each check, none for the overall pass rate, and no
#: bounded interval on any contrast.
_STORED_BEFORE_55 = Path(__file__).parent / "fixtures" / "generate_analysis" / "recordings" / "llm_analysis.bundle.json"


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
    # Plain counts, each arm by its model ("cheaper candidate cost more" would read as a contradiction), within the
    # page's three figures a sentence: the costs are the decision table's, directly below.
    assert verdict.text == (
        "Keep production: deepseek-v4-flash passed 2 of 12 trials to deepseek-v4.1-flash's 10, and cost more per reply."
    )
    assert lint_prose(verdict.text) == []
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


# --- switch reads the bounded interval, on every rate the rule reads -----------------------------------------------


def _passing(passes: Sequence[int], cost: float) -> Callable[[Mapping[str, Any]], Coroutine[Any, Any, Answer]]:
    """A runner that answers case ``i`` right on its first ``passes[i]`` repeats and wrong after, each at ``cost``."""
    calls: Counter[str] = Counter()

    async def run(case: Mapping[str, Any]) -> Answer:
        calls[case["text"]] += 1
        right = calls[case["text"]] <= passes[int(case["text"].rsplit(" ", 1)[1])]
        return Answer(
            case["label"] if right else "skipped", model="m", input_tokens=100, output_tokens=10, cost_usd=cost
        )

    return run


async def _ruled(
    control_passes: Sequence[int], candidate_passes: Sequence[int]
) -> tuple[AnalysisContextBundle, str, str]:
    """Two arms over ``len(control_passes)`` cases × 3 repeats, each passing its cases' counts, with G2's rule asked first.

    The question is on the campaign before its bundle is assembled, so the family of comparisons is the one the rule
    reads: the label check's pass rate (compared as ``accuracy``) and the overall pass rate.
    """
    cases = [{"text": f"queue request {index}", "label": "queued"} for index in range(len(control_passes))]
    # Each arm at one cost per reply: a gap with no spread on a measure with no declared range, which no test reads, so
    # the family is the two rates alone.
    arms = {"production": _passing(control_passes, 0.0031), "candidate": _passing(candidate_passes, 0.0021)}
    comparison = await Eval(cases, arms, ["label"], k=3, name="ruled").run(store=":memory:")
    storage = comparison.host.storage
    campaign = off_loop(get_campaign, storage, comparison.campaign_id, comparison.scope_id)
    assert campaign.declared_design is not None
    asked = Question(id="q-g2", text=G2_QUESTION, asked_at="2000-01-01T00:00:00+00:00")
    campaign = campaign.model_copy(
        update={"declared_design": campaign.declared_design.model_copy(update={"questions": [asked]})}
    )
    bundle = off_loop(assemble_context_bundle, campaign, storage=storage, profile=comparison.host.profile)
    assert bundle.design.control_arm is not None
    control = bundle.design.control_arm.variant_key
    (candidate,) = [board.variant_key for board in bundle.scoreboard if board.variant_key != control]
    assert [board.passed for board in bundle.scoreboard] == [sum(control_passes), sum(candidate_passes)]
    return bundle, control, candidate


def _rate_clause(bundle: AnalysisContextBundle, control: str, candidate: str) -> tuple[str, str]:
    """The rate clause of G2's rule on the bundle: its outcome and its reason."""
    rule = parse_decision_rule(G2_QUESTION)
    assert rule is not None
    clause = evaluate_rule(rule, rule_readings(bundle, control, candidate)).clauses[0]
    assert clause.clause.kind == "rate_drop"
    return clause.outcome, clause.reason


async def test_the_overall_pass_rate_joins_the_family_of_the_question_whose_rule_reads_it() -> None:
    bundle, control, candidate = await _ruled([0] * 7, [3] * 7)
    (family,) = bundle.multiple_comparisons.families
    (passed,) = [comparison for comparison in family.comparisons if comparison.reading == "trial"]
    assert passed.name == "passed"
    # The case is the unit: seven cases a side, each the share of its three trials that passed.
    assert (passed.control.n_cases, passed.contrast.n_cases) == (7, 7)
    assert (passed.control.mean, passed.contrast.mean) == (0.0, 1.0)
    # It is one of the family's corrected comparisons, beside the label check's (as accuracy).
    tested = {
        (comparison.reading, comparison.name) for comparison in family.comparisons if comparison.p_raw is not None
    }
    assert tested == {("trial", "passed"), ("measure", "accuracy")}
    assert family.family_size == 2
    assert passed.bounded_interval is not None


async def test_seven_cases_with_the_bounded_interval_inside_the_margin_switch_on_the_rate_clause() -> None:
    # Only a sweep settles a 10-point clause on seven cases: the candidate passes every trial, the control none.
    bundle, control, candidate = await _ruled([0] * 7, [3] * 7)
    readings = rule_readings(bundle, control, candidate)
    assert [rate.name for rate in readings.rates] == ["the pass rate", "Label matched"]
    assert all(rate.interval is not None and rate.interval[0] > -0.10 for rate in readings.rates)
    assert _rate_clause(bundle, control, candidate) == ("switch", "the interval keeps every pass rate within 10 points")


async def test_seven_cases_with_the_bounded_interval_too_wide_are_unsettled_and_say_how_far_it_reaches() -> None:
    # One trial in 21 lost on the control: the arms all but agree, and seven cases still bound nothing near 10 points.
    bundle, control, candidate = await _ruled([3, 3, 3, 3, 3, 3, 2], [3] * 7)
    assert _rate_clause(bundle, control, candidate) == (
        "unsettled",
        "the interval allows the pass rate to drop up to 49 points",
    )


async def test_four_cases_are_unsettled_on_the_case_floor() -> None:
    bundle, control, candidate = await _ruled([3] * 4, [3] * 4)
    assert _rate_clause(bundle, control, candidate) == (
        "unsettled",
        "4 cases cannot show the drop stays within 10 points (a rate needs 5)",
    )


def test_a_bundle_stored_before_the_bounded_interval_never_settles_switch() -> None:
    bundle = AnalysisContextBundle.from_json(_STORED_BEFORE_55.read_text(encoding="utf-8"))
    assert bundle.schema_version == 54
    control = bundle.design.control_arm.variant_key if bundle.design.control_arm is not None else ""
    (candidate,) = [board.variant_key for board in bundle.scoreboard if board.variant_key != control]
    # Twelve cases a side and a check contrast with a t interval: what a switch needs is still missing.
    assert min(board.usable_cases for board in bundle.scoreboard) >= 5
    assert _rate_clause(bundle, control, candidate) == ("unsettled", "this bundle predates the interval a switch needs")


async def test_a_check_clause_no_longer_switches_on_a_t_interval_the_bounded_one_does_not_back() -> None:
    """The .29 defect: a check's rate read switch off the family's t interval, which overstates its confidence.

    Simulated at the margin, that t interval let switch through up to 12% of the time against the 2.5% it states
    (``docs/measuring-soundly.md``). Here it clears the margin while the bounded interval reaches far past it.
    """
    bundle, control, candidate = await _ruled([3, 3, 3, 3, 3, 3, 2], [3] * 7)
    (check,) = [
        comparison
        for family in bundle.multiple_comparisons.families
        for comparison in family.comparisons
        if comparison.name == "accuracy"
    ]
    assert check.basis == "t" and check.interval is not None and check.bounded_interval is not None
    assert check.interval[0] > -0.10, "the t interval clears the 10-point margin"
    assert check.bounded_interval[0] < -0.10, "the bounded interval does not"
    rule = parse_decision_rule(G2_QUESTION)
    assert rule is not None
    readings = rule_readings(bundle, control, candidate)
    (label,) = [rate for rate in readings.rates if rate.name == "Label matched"]
    assert label.interval == check.bounded_interval
    alone = replace(readings, rates=(label,))
    assert evaluate_rule(rule, alone).clauses[0].outcome == "unsettled"
    # Read on the t interval, as .29 read it, the same clause switched.
    on_t = replace(readings, rates=(replace(label, interval=check.interval),))
    assert evaluate_rule(rule, on_t).clauses[0].outcome == "switch"
