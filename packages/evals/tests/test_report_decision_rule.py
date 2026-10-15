"""A declared decision rule, read clause by clause, on the G2 campaign's own numbers.

G2 (2026-10) declared: switch only if no pass rate drops by more than 10 points and the cheaper model costs less per
reply, otherwise keep production (``RateDrop(max_points=10)`` and ``CostLower()``), and ran 4 cases × 3 repeats:
production passed 10 of 12, the cheaper candidate 2 of 12, at a cost per reply of 0.003092 against 0.003301. Its memo called the question unanswerable,
because four cases are under the floor for estimating a rate. Neither clause needs that floor to settle ``keep``: a
cost observed higher, and a pass rate whose observed drop is past the margin, are enough not to switch. Settling
``switch`` is another matter: it needs every arm at the case floor and an interval that bounds the clause.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from threetears.evals import Answer, Eval, Result
from threetears.evals.analysis import (
    AnalysisContextBundle,
    Report,
    assemble_context_bundle,
    build_code_only_report,
    get_campaign,
)
from threetears.evals.analysis.bundle.schema import JudgedArm, JudgedMeasure
from threetears.evals.analysis.report import DisclosureBlock, TableBlock
from threetears.evals.analysis import report_html, report_markdown
from threetears.evals.analysis.report.decision import (
    PROSE_RULE_LINE,
    RateReading,
    RuleReadings,
    evaluate_rule,
    keep_is_the_default,
    rule_readings,
    rule_statement,
)
from threetears.evals.analysis.style import lint_prose
from threetears.evals.kernel.declaration import (
    CostLower,
    DecisionRule,
    DesignRefused,
    Question,
    RateDrop,
    decision_readings,
    reconcile_question_edits,
    refuse_an_undeclarable_design,
)
from threetears.evals.kernel.metrics import ACCURACY_MEASURE, MetricDescriptor, goal_check_measure
import pytest
from pydantic import ValidationError

from packages.evals.tests.factories import minimal_declaration
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from packages.evals.tests.g2_support import G2_QUESTION, G2_RULE, declaring, g2_numbers, g2_ruled_report
from packages.evals.tests.off_loop import off_loop

#: A bundle the engine stored before shape 55: a contrast for each check, none for the overall pass rate, and no
#: bounded interval on any contrast.
_STORED_BEFORE_55 = Path(__file__).parent / "fixtures" / "generate_analysis" / "recordings" / "llm_analysis.bundle.json"


#: The arms by the names a page gives them, which every reason says.
_ARMS = {"control": "production", "candidate": "cheaper candidate"}


@pytest.fixture
async def g2_ruled() -> tuple[Result, Report]:
    """The G2 numbers through the quick path, with the G2 decision rule declared before the run."""
    return await g2_ruled_report()


def test_the_g2_rule_keeps_production_on_both_clauses() -> None:
    rule = G2_RULE
    readings = RuleReadings(
        rates=(RateReading("the pass rate", 10, 12, 2, 12),), control_cost=0.003092, candidate_cost=0.003301
    )
    verdict = evaluate_rule(rule, readings, **_ARMS)
    assert [clause.outcome for clause in verdict.clauses] == ["keep", "keep"]
    assert verdict.outcome == "keep"
    # Four cases is under the floor for estimating a magnitude, never for keeping on what the run observed. Nothing
    # remains to run, so the drop is stated as observed, without "at least": the counts and the margin in words, the
    # arms by name, the change in points to one decimal.
    assert verdict.clauses[0].reason == (
        "cheaper candidate passed 2 of 12 to production's 10, a drop of 66.7 points against a margin of 10"
    )
    assert verdict.clauses[0].rate is readings.rates[0]


def test_a_rate_clause_the_remaining_trials_could_still_decide_is_unsettled() -> None:
    rule = G2_RULE
    close = RuleReadings(
        rates=(RateReading("the pass rate", 10, 12, 8, 10, 0, 2, interval=(-0.05, 0.02)),),
        control_cost=2.0,
        candidate_cost=1.0,
        control_cases=6,
        candidate_cases=6,
        cost_interval=(-1.5, -0.5),
    )
    verdict = evaluate_rule(rule, close, **_ARMS)
    assert [clause.outcome for clause in verdict.clauses] == ["unsettled", "switch"]
    assert verdict.outcome == "unsettled"
    assert verdict.clauses[0].reason == (
        "cheaper candidate's drop is not shown to stay within 10 points (the lost trials could still take it past)"
    )
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
    assert evaluate_rule(rule, settled, **_ARMS).outcome == "switch"
    bare = RuleReadings(rates=(RateReading("the pass rate", 12, 12, 12, 12),), control_cost=2.0, candidate_cost=1.0)
    assert evaluate_rule(rule, bare, **_ARMS).outcome == "unsettled"


def test_equal_rates_on_too_few_cases_are_unsettled_not_switch() -> None:
    rule = G2_RULE
    equal = RuleReadings(
        rates=(RateReading("the pass rate", 10, 12, 10, 12, interval=(-0.02, 0.02)),),
        control_cost=2.0,
        candidate_cost=1.0,
        control_cases=4,
        candidate_cases=4,
        cost_interval=(-1.5, -0.5),
    )
    verdict = evaluate_rule(rule, equal, **_ARMS)
    assert [clause.outcome for clause in verdict.clauses] == ["unsettled", "unsettled"]
    assert verdict.outcome == "unsettled"
    # What is withheld and why, in the reader's terms: never a bare "needs 5".
    assert verdict.clauses[0].reason == (
        "cheaper candidate's drop is not shown to stay within 10 points (4 cases are too few for the interval that "
        "would show it, which is estimated only from 5 independent cases)"
    )
    assert verdict.clauses[1].reason == (
        "4 cases cannot show the cost is lower, which takes an interval estimated only from 5 independent cases"
    )


def test_a_floor_met_run_with_a_tight_interval_switches() -> None:
    rule = G2_RULE
    tight = RuleReadings(
        rates=(RateReading("the pass rate", 50, 60, 49, 60, interval=(-0.08, 0.05)),),
        control_cost=0.0031,
        candidate_cost=0.0021,
        control_cases=20,
        candidate_cases=20,
        cost_interval=(-0.0014, -0.0006),
    )
    verdict = evaluate_rule(rule, tight, **_ARMS)
    assert [clause.outcome for clause in verdict.clauses] == ["switch", "switch"]
    assert verdict.outcome == "switch"
    assert verdict.clauses[0].reason == "cheaper candidate's drop is shown to stay within 10 points (it is at most 8)"
    # The same run with an interval reaching past the margin is unsettled, and says how far it reaches.
    wide = RuleReadings(
        rates=(RateReading("the pass rate", 50, 60, 49, 60, interval=(-0.23, 0.2)),),
        control_cost=0.0031,
        candidate_cost=0.0021,
        control_cases=20,
        candidate_cases=20,
        cost_interval=(-0.0014, -0.0006),
    )
    wide_verdict = evaluate_rule(rule, wide, **_ARMS)
    assert wide_verdict.outcome == "unsettled"
    assert wide_verdict.clauses[0].reason == (
        "cheaper candidate's drop is not shown to stay within 10 points (it could be as much as 23)"
    )


def test_a_cost_lower_by_a_point_estimate_only_is_unsettled() -> None:
    rule = G2_RULE
    readings = RuleReadings(
        rates=(RateReading("the pass rate", 50, 60, 50, 60, interval=(-0.05, 0.05)),),
        control_cost=0.0031,
        candidate_cost=0.0029,
        control_cases=20,
        candidate_cases=20,
        cost_interval=(-0.0006, 0.0002),
    )
    verdict = evaluate_rule(rule, readings, **_ARMS)
    assert [clause.outcome for clause in verdict.clauses] == ["switch", "unsettled"]
    assert verdict.clauses[1].reason == "the cost difference did not separate"
    assert verdict.outcome == "unsettled"
    # A cost observed higher still settles keep, on any number of cases.
    dearer = RuleReadings(rates=readings.rates, control_cost=0.0029, candidate_cost=0.0031)
    assert evaluate_rule(rule, dearer, **_ARMS).clauses[1].outcome == "keep"


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
    assert outcomes == [
        "keep: cheaper candidate passed 2 of 12 to production's 10, a drop of 66.7 points against a margin of 10",
        "keep: cheaper candidate costs more per reply than production",
    ]
    # Under a cent per reply on every arm, the cost row reads per 1,000 replies, as the scoreboard does.
    assert {"reading": "cost per 1,000 replies", "control": "$3.092", "candidate": "$3.301"}.items() <= next(
        row for row in table.values() if row["reading"] == "cost per 1,000 replies"
    ).items()


async def test_a_kept_clause_beside_a_contrast_not_separated_says_why_once_under_the_clauses(
    g2_ruled: tuple[Result, Report],
) -> None:
    """Prod, .30: a 10-point clause read keep while the contrasts read the same pass rate as not separated.

    Both are right: keep is the default, so an unsettled rule keeps the control for now, and a clause reads the
    bounded interval a switch reads, which can be wider than the t interval the contrasts table prints. The page says
    so once, under the clauses, and each clause's row carries its own outcome, its reason saying the counts and the
    margin in words, naming the arms.
    """
    _, report = g2_ruled
    # What unsettled means for the reader, the control by its label.
    sentence = (
        "Keep is the default: a switch needs every clause shown, so an unsettled rule keeps production for now. A "
        "clause reads the interval a switch reads, which can be wider than the contrasts table's."
    )
    assert keep_is_the_default("production") == sentence
    (said,) = [b for b in report.blocks if isinstance(b, DisclosureBlock) and b.section == "decision"]
    assert (said.source, said.text) == ("rule", sentence)
    markdown = report_markdown(report)
    assert markdown.count(sentence) == 1
    decision = markdown.split("\n## Decision\n", 1)[1].split("\n## ", 1)[0]
    assert decision.index("**Clauses**") < decision.index(sentence), "under the clauses table"
    assert (
        "keep: cheaper candidate passed 2 of 12 to production's 10, a drop of 66.7 points against a margin of 10"
        in (decision)
    )
    # The prod numbers: v2 passed 14 of 60 trials to v1's 29, and on a small sample the interval reaches 53.6 points.
    on = {"control": "v1", "candidate": "v2"}
    rule = DecisionRule(clauses=(RateDrop(max_points=10),))
    dropped = RuleReadings(rates=(RateReading("the pass rate", 29, 60, 14, 60),), control_cases=20, candidate_cases=20)
    assert evaluate_rule(rule, dropped, **on).clauses[0].reason == (
        "v2 passed 14 of 60 to v1's 29, a drop of 25 points against a margin of 10"
    )
    wide = RuleReadings(
        rates=(RateReading("the pass rate", 29, 60, 28, 60, interval=(-0.536, 0.036)),),
        control_cases=20,
        candidate_cases=20,
    )
    (unshown,) = evaluate_rule(rule, wide, **on).clauses
    assert (unshown.outcome, unshown.reason) == (
        "unsettled",
        "v2's drop is not shown to stay within 10 points (it could be as much as 53.6)",
    )


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


#: G2's question, as the one live question a ruled campaign asks unless told otherwise.
_G2_ASKED = Question(id="q-g2", text=G2_QUESTION, decision=G2_RULE, asked_at="2000-01-01T00:00:00+00:00")


async def _ruled(
    control_passes: Sequence[int], candidate_passes: Sequence[int], questions: Sequence[Question] = (_G2_ASKED,)
) -> tuple[AnalysisContextBundle, str, str]:
    """Two arms over ``len(control_passes)`` cases × 3 repeats, each passing its cases' counts, with G2's rule asked first.

    The questions are on the campaign before its bundle is assembled, so each family of comparisons is one a rule can
    read: G2's is the label check's pass rate (compared as ``accuracy``) and the overall pass rate.
    """
    cases = [{"text": f"queue request {index}", "label": "queued"} for index in range(len(control_passes))]
    # Each arm at one cost per reply: a gap with no spread on a measure with no declared range, which no test reads, so
    # the family is the two rates alone.
    arms = {"production": _passing(control_passes, 0.0031), "candidate": _passing(candidate_passes, 0.0021)}
    comparison = await Eval(cases, arms, ["label"], k=3, name="ruled").run(store=":memory:")
    storage = comparison.host.storage
    campaign = off_loop(get_campaign, storage, comparison.campaign_id, comparison.scope_id)
    assert campaign.declared_design is not None
    campaign = campaign.model_copy(
        update={"declared_design": campaign.declared_design.model_copy(update={"questions": list(questions)})}
    )
    bundle = off_loop(assemble_context_bundle, campaign, storage=storage, profile=comparison.host.profile)
    assert bundle.design.control_arm is not None
    control = bundle.design.control_arm.variant_key
    (candidate,) = [board.variant_key for board in bundle.scoreboard if board.variant_key != control]
    assert [board.passed for board in bundle.scoreboard] == [sum(control_passes), sum(candidate_passes)]
    return bundle, control, candidate


def _rate_clause(
    bundle: AnalysisContextBundle, control: str, candidate: str, question_id: str | None = "q-g2"
) -> tuple[str, str]:
    """The rate clause of G2's rule on the bundle, read in ``question_id``'s family: its outcome and its reason."""
    rule = G2_RULE
    readings = rule_readings(bundle, control, candidate, question_id=question_id)
    clause = evaluate_rule(rule, readings, **_ARMS).clauses[0]
    assert isinstance(clause.clause, RateDrop)
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
    readings = rule_readings(bundle, control, candidate, question_id="q-g2")
    assert [rate.name for rate in readings.rates] == ["the pass rate", "Label matched"]
    assert all(rate.interval is not None and rate.interval[0] > -0.10 for rate in readings.rates)
    assert _rate_clause(bundle, control, candidate) == (
        "switch",
        "cheaper candidate's drop is shown to stay within 10 points",
    )


async def test_seven_cases_with_the_bounded_interval_too_wide_are_unsettled_and_say_how_far_it_reaches() -> None:
    # One trial in 21 lost on the control: the arms all but agree, and seven cases still bound nothing near 10 points.
    bundle, control, candidate = await _ruled([3, 3, 3, 3, 3, 3, 2], [3] * 7)
    assert _rate_clause(bundle, control, candidate) == (
        "unsettled",
        "cheaper candidate's drop is not shown to stay within 10 points (it could be as much as 49)",
    )


async def test_four_cases_are_unsettled_on_the_case_floor() -> None:
    bundle, control, candidate = await _ruled([3] * 4, [3] * 4)
    assert _rate_clause(bundle, control, candidate) == (
        "unsettled",
        "cheaper candidate's drop is not shown to stay within 10 points (4 cases are too few for the interval that "
        "would show it, which is estimated only from 5 independent cases)",
    )


def test_a_bundle_stored_before_the_bounded_interval_never_settles_switch() -> None:
    bundle = AnalysisContextBundle.from_json(_STORED_BEFORE_55.read_text(encoding="utf-8"))
    assert bundle.schema_version == 54
    control = bundle.design.control_arm.variant_key if bundle.design.control_arm is not None else ""
    (candidate,) = [board.variant_key for board in bundle.scoreboard if board.variant_key != control]
    # Twelve cases a side and a check contrast with a t interval: what a switch needs is still missing.
    assert min(board.usable_cases for board in bundle.scoreboard) >= 5
    # It asked no question: its one family is campaign-wide.
    assert [family.question_id for family in bundle.multiple_comparisons.families] == [None]
    assert _rate_clause(bundle, control, candidate, question_id=None) == (
        "unsettled",
        "cheaper candidate's drop is not shown to stay within 10 points (this bundle predates the interval a switch "
        "needs)",
    )


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
    rule = G2_RULE
    readings = rule_readings(bundle, control, candidate, question_id="q-g2")
    (label,) = [rate for rate in readings.rates if rate.name == "Label matched"]
    assert label.interval == check.bounded_interval
    alone = replace(readings, rates=(label,))
    assert evaluate_rule(rule, alone, **_ARMS).clauses[0].outcome == "unsettled"
    # Read on the t interval, as .29 read it, the same clause switched.
    on_t = replace(readings, rates=(replace(label, interval=check.interval),))
    assert evaluate_rule(rule, on_t, **_ARMS).clauses[0].outcome == "switch"


async def test_a_goal_check_carried_as_a_mean_is_read_as_its_count_with_its_bounded_interval() -> None:
    """A goal check reaches the bundle as a mean of 0s and 1s, not a rate: the rule reads it as the count it is.

    The ruled campaign's label check stands in for one: its per-cell summary and its contrast, renamed to a goal
    check's measure and carried as a mean, as a goal check's are.
    """
    bundle, control, candidate = await _ruled([3, 3, 3, 3, 3, 3, 2], [3] * 7)
    check = goal_check_measure('call_count("music.queue_track") >= 2')
    cells = [
        cell.model_copy(
            update={
                "measures": cell.measures.model_copy(
                    update={
                        "measures": [
                            *cell.measures.measures,
                            *(
                                summary.model_copy(update={"name": check, "rate": None, "n_true": None})
                                for summary in cell.measures.measures
                                if summary.name == "accuracy"
                            ),
                        ]
                    }
                )
            }
        )
        for cell in bundle.cell_measures
    ]
    families = [
        family.model_copy(
            update={
                "comparisons": [
                    *family.comparisons,
                    *(c.model_copy(update={"name": check}) for c in family.comparisons if c.name == "accuracy"),
                ]
            }
        )
        for family in bundle.multiple_comparisons.families
    ]
    as_goal_check = bundle.model_copy(
        update={
            "cell_measures": cells,
            "multiple_comparisons": bundle.multiple_comparisons.model_copy(update={"families": families}),
        }
    )
    (contrast,) = [c for family in families for c in family.comparisons if c.name == check]
    (rate,) = [
        rate
        for rate in rule_readings(as_goal_check, control, candidate, question_id="q-g2").rates
        if rate.measure == check
    ]
    assert (rate.control_passed, rate.control_n, rate.candidate_passed, rate.candidate_n) == (20, 21, 21, 21)
    assert rate.interval == contrast.bounded_interval


async def test_a_rule_reads_its_own_questions_family_and_never_another_at_another_level() -> None:
    """Two questions, two families of different sizes, so each carries the label check's interval at its own level.

    The first asks about quality with no rule: its family is the label check alone, at 95%. G2's, which states the
    rule, adds the overall pass rate, so its intervals are at 97.5%. A rule read the first matching contrast across
    families until ``.30``, which here was the other question's.
    """
    quality = Question(
        id="q-quality", text="Is the candidate better?", merit_axes=["quality"], asked_at=_G2_ASKED.asked_at
    )
    bundle, control, candidate = await _ruled([3, 3, 3, 3, 3, 3, 2], [3] * 7, questions=(quality, _G2_ASKED))
    families = {family.question_id: family for family in bundle.multiple_comparisons.families}
    assert list(families) == ["q-quality", "q-g2"]
    assert (families["q-quality"].family_size, families["q-g2"].family_size) == (1, 2)
    label = {
        question: next(c.bounded_interval for c in family.comparisons if c.name == "accuracy")
        for question, family in families.items()
    }
    assert label["q-quality"] != label["q-g2"]
    readings = rule_readings(bundle, control, candidate, question_id="q-g2")
    assert [rate.name for rate in readings.rates] == ["the pass rate", "Label matched"]
    assert readings.rates[1].interval == label["q-g2"]
    # The question with no rule has no pass-rate contrast: read there, the overall pass rate has no interval.
    elsewhere = rule_readings(bundle, control, candidate, question_id="q-quality")
    assert elsewhere.rates[0].interval is None and elsewhere.rates[1].interval == label["q-quality"]
    # The page reads the rule in the question that states it.
    report = build_code_only_report(bundle, assembled_at="2026-10-14T12:00:00+00:00", campaign_name="two questions")
    (table,) = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "decision"]
    (outcome,) = [
        row["outcome"] for row in table.values() if row["outcome"] is not None and row["reading"] == "the pass rate"
    ]
    # The clause's own outcome: one shown neither way is unsettled, as the verdict line says, with why.
    assert outcome == "unsettled: candidate's drop is not shown to stay within 10 points (it could be as much as 49)"


# --- the rule is typed: built, stored and refused as data, never parsed from words ---------------------------------


def test_a_rule_needs_a_clause_and_a_margin_inside_the_scale() -> None:
    with pytest.raises(ValidationError, match="clauses"):
        DecisionRule(clauses=())
    for margin in (0.0, -1.0, 100.0, float("nan")):
        with pytest.raises(ValidationError, match="max_points"):
            RateDrop(max_points=margin)
    with pytest.raises(ValidationError, match="reading"):
        RateDrop(max_points=2, reading="")


def test_a_rule_reads_back_from_its_stored_form_clause_by_clause() -> None:
    rule = DecisionRule(clauses=(RateDrop(max_points=2, reading="tone"), CostLower(), RateDrop(max_points=5)))
    stored = rule.model_dump(mode="json")
    assert DecisionRule.model_validate(stored) == rule
    assert [type(clause) for clause in DecisionRule.model_validate(stored).clauses] == [RateDrop, CostLower, RateDrop]


def test_the_rule_is_said_in_one_sentence_naming_the_arms() -> None:
    rule = DecisionRule(clauses=(RateDrop(max_points=2), CostLower()))
    assert rule_statement(rule, control="v1", candidates=["v2"]) == (
        "Switch to v2 only if no pass rate drops by more than 2 points and v2 costs less per reply. Otherwise stay "
        "on v1."
    )
    scoped = DecisionRule(clauses=(RateDrop(max_points=2.5, reading="no_invented_total"),))
    assert rule_statement(scoped, control="v1", candidates=["v2"]) == (
        "Switch to v2 only if no no invented total rate drops by more than 2.5 points. Otherwise stay on v1."
    )
    three = DecisionRule(clauses=(RateDrop(max_points=2), RateDrop(max_points=1, reading="tone"), CostLower()))
    assert rule_statement(three, control="v1", candidates=["v2", "v3"]) == (
        "Switch to v2 or v3 only if no pass rate drops by more than 2 points, no tone rate drops by more than 1 "
        "points and it costs less per reply. Otherwise stay on v1."
    )


def test_every_reason_names_the_arms_by_their_names() -> None:
    cheaper = RuleReadings(
        rates=(RateReading("the pass rate", 50, 60, 50, 60, interval=(-0.05, 0.05)),),
        control_cost=0.0031,
        candidate_cost=0.0021,
        control_cases=20,
        candidate_cases=20,
        cost_interval=(-0.0014, -0.0006),
    )
    dearer = replace(cheaper, candidate_cost=0.0041)
    level = replace(cheaper, candidate_cost=0.0031)
    on = {"control": "v1", "candidate": "v2"}
    assert evaluate_rule(G2_RULE, cheaper, **on).clauses[1].reason == "v2 costs less per reply than v1"
    assert evaluate_rule(G2_RULE, dearer, **on).clauses[1].reason == "v2 costs more per reply than v1"
    assert evaluate_rule(G2_RULE, level, **on).clauses[1].reason == "v2 costs no less per reply than v1"
    statements = [clause.statement for clause in evaluate_rule(G2_RULE, cheaper, **on).clauses]
    assert statements == ["no pass rate drops by more than 10 points", "v2 costs less per reply"]


async def test_a_clause_naming_one_reading_reads_that_rate_alone() -> None:
    # The overall pass rate drops 33 points; the label check, which the clause names, does too: keep on it alone.
    bundle, control, candidate = await _ruled([3] * 7, [2] * 7)
    scoped = DecisionRule(clauses=(RateDrop(max_points=10, reading="match"),))
    readings = rule_readings(bundle, control, candidate, question_id="q-g2")
    (clause,) = evaluate_rule(scoped, readings, **_ARMS).clauses
    assert clause.outcome == "keep"
    assert clause.rate is not None and clause.rate.measure == "match"
    assert clause.statement == "no Label matched rate drops by more than 10 points"
    # A name the arms do not carry is said, and left unsettled, never read as every rate.
    (absent,) = evaluate_rule(
        DecisionRule(clauses=(RateDrop(max_points=10, reading="tone"),)), readings, **_ARMS
    ).clauses
    assert (absent.outcome, absent.reason) == ("unsettled", "tone was not measured on both arms")


async def test_a_rule_whose_rate_clauses_each_name_a_reading_does_not_compare_the_overall_pass_rate() -> None:
    scoped = _G2_ASKED.model_copy(
        update={"decision": DecisionRule(clauses=(RateDrop(max_points=10, reading="match"),))}
    )
    bundle, _, _ = await _ruled([0] * 7, [3] * 7, questions=(scoped,))
    (family,) = bundle.multiple_comparisons.families
    assert not [comparison for comparison in family.comparisons if comparison.reading == "trial"]


_PROSE = (
    "Does v2 hold the pass rate at lower cost? Decision rule: switch only if no dimension drops by more than 10 points "
    "AND the cost per turn is lower; otherwise keep v1."
)


def test_a_question_declaring_its_rule_as_prose_is_refused_naming_the_field() -> None:
    prose = Question(id="q-prose", text=_PROSE)
    design = minimal_declaration().model_copy(update={"questions": [prose]})
    with pytest.raises(DesignRefused, match=r"Question\.decision") as refused:
        refuse_an_undeclarable_design(design, behavior="b", template=toyhost_template(), profile=toyhost_profile())
    assert "q-prose" in refused.value.fault and "Question.decision" in refused.value.remedy
    # Typed beside the same words, or retired, it is declared.
    typed = prose.model_copy(update={"decision": G2_RULE})
    retired = prose.model_copy(update={"retired_at": "2026-10-14T00:00:00+00:00"})
    for question in (typed, retired):
        accepted = design.model_copy(update={"questions": [question]})
        refuse_an_undeclarable_design(accepted, behavior="b", template=toyhost_template(), profile=toyhost_profile())
    # A stored question loads as it was written: the refusal is the declaration's, never a read's.
    stored = design.model_dump(mode="json")
    assert type(design).model_validate(stored).questions[0].decision is None


def test_a_changed_rule_mints_a_new_question_as_changed_words_do() -> None:
    asked = Question(id="q-1", text="Does v2 hold?", decision=G2_RULE)
    looser = asked.model_copy(update={"decision": DecisionRule(clauses=(RateDrop(max_points=20),))})
    kept, minted = reconcile_question_edits([asked], [looser])
    assert kept.decision == G2_RULE and minted.supersedes == "q-1" and minted.decision == looser.decision
    assert reconcile_question_edits([asked], [asked.model_copy()]) == [asked]


async def test_a_stored_prose_rule_is_said_unread_and_the_verdict_is_exploratory() -> None:
    comparison = await g2_numbers()
    bundle = declaring(comparison.report.bundle, _PROSE, decision=None)
    report = build_code_only_report(bundle, assembled_at="2026-10-14T00:00:00+00:00", campaign_name="prose")
    decision = [block for block in report.blocks if block.section == "decision"]
    assert [block.text for block in decision if isinstance(block, DisclosureBlock)] == [PROSE_RULE_LINE]
    assert not [block for block in decision if isinstance(block, TableBlock)]
    (verdict,) = [b for b in report.blocks if isinstance(b, DisclosureBlock) and b.source == "verdict"]
    assert verdict.text.startswith("Exploratory: ")
    assert f"## Decision\n\n{PROSE_RULE_LINE}\n" in report_markdown(report)


# --- the Decision section speaks the reader's words, never the engine's ---------------------------------------------

#: The engine's names for a rule's parts, which the Decision section never prints.
_SYSTEM_NAMES = re.compile(r"rate_drop|cost_lower|max_points|\bclauses\b|\bkind\b")

#: The roles of a comparison's arms, which the Decision section never names an arm by.
_ROLE_WORDS = re.compile(r"\b(candidate|control)\b", re.I)


def _decision_section(page: str, *, html_page: bool) -> str:
    """The Decision section's visible text, from its heading to the next section's."""
    if html_page:
        body = page[page.index('<section id="decision">') :]
        return html.unescape(re.sub(r"<[^>]+>", " ", body[: body.index("</section>")]))
    body = page.split("\n## Decision\n", 1)[1]
    return body.split("\n## ", 1)[0]


async def test_the_decision_section_names_no_system_word_and_no_arm_by_its_role() -> None:
    _, one = await g2_ruled_report()
    bundle, _, _ = await _ruled([3] * 7, [2] * 7)
    two = build_code_only_report(bundle, assembled_at="2026-10-14T12:00:00+00:00", campaign_name="ruled")
    for report, labels in ((one, ("cheaper candidate", "production")), (two, ("candidate", "production"))):
        for door, page in (("markdown", report_markdown(report)), ("html", report_html(report))):
            section = _decision_section(page, html_page=door == "html")
            # The rule's sentence is the first screen's subhead, said once: the section links up to it.
            first = page.split("## Decision", 1)[0] if door == "markdown" else page.split("<h2>", 1)[0]
            assert "Switch to " not in section and "Switch to " in first, (door, section[:200])
            assert door == "markdown" or '<h3><a href="#rule">Clauses</a></h3>' in page
            assert not _SYSTEM_NAMES.search(section), (door, _SYSTEM_NAMES.findall(section))
            # An arm's own name may hold a role word (an arm a host called "cheaper candidate"); nothing else may.
            for label in labels:
                section = re.sub(rf"(?<![\w-]){re.escape(label)}(?![\w-])", "ARM", section)
            assert not _ROLE_WORDS.search(section), (door, _ROLE_WORDS.findall(section), section[:400])


# --- a rule's reading is checked where the design is declared, against what the campaign measures -------------------

#: The toy host's template's one goal-state check, verbatim.
_TOY_CHECK = 'call_count("extractor.emit_field") == 4'


def _declare(*readings: str | None) -> None:
    """Declare one question whose rule has a rate clause per reading, through the declaration gate."""
    rule = DecisionRule(clauses=tuple(RateDrop(max_points=10, reading=reading) for reading in readings))
    design = minimal_declaration().model_copy(
        update={"questions": [Question(id="q-scoped", text="Does v2 hold?", decision=rule)]}
    )
    refuse_an_undeclarable_design(design, behavior="b", template=toyhost_template(), profile=toyhost_profile())


def test_a_reading_the_campaign_does_not_measure_is_refused_naming_what_it_does() -> None:
    with pytest.raises(DesignRefused, match="question q-scoped names 'tone'") as refused:
        _declare("tone")
    # The remedy lists what the template and the host offer: the check verbatim and the host's pass/fail measures.
    assert repr(_TOY_CHECK) in refused.value.remedy and "'match'" in refused.value.remedy
    assert decision_readings(toyhost_template(), toyhost_profile()) == [_TOY_CHECK, "goal_state", "match"]


def test_a_reading_the_campaign_measures_or_none_is_declared() -> None:
    _declare(_TOY_CHECK, "match", None)


async def test_a_pass_fail_judged_dimension_is_read_as_its_count() -> None:
    """A judged dimension on the pass/fail scale is a pass rate a rule names: its mean times its scores, per arm."""
    bundle, control, candidate = await _ruled([3] * 7, [3] * 7)

    def arm(variant_key: str, passed: int) -> JudgedArm:
        return JudgedArm(
            variant_key=variant_key,
            apparatus_class_id="rig",
            run_ids=["run"],
            n=21,
            n_independent=7,
            mean=passed / 21,
            evidence_tier="undetermined",
        )

    def judged(name: str, axis: Literal["capability", "boundary"]) -> JudgedMeasure:
        arms = [arm(control, 20), arm(candidate, 12)]
        return JudgedMeasure(name=name, family="rubric", value_range=(0.0, 1.0), scale="ratio", axis=axis, arms=arms)

    scored = bundle.model_copy(update={"judged_measures": [judged("tone", "capability"), judged("safe", "boundary")]})
    readings = rule_readings(scored, control, candidate, question_id="q-g2")
    (tone,) = [rate for rate in readings.rates if rate.measure == "tone"]
    assert (tone.control_passed, tone.control_n, tone.candidate_passed, tone.candidate_n) == (20, 21, 12, 21)
    assert not [rate for rate in readings.rates if rate.measure == "safe"], "a guardrail is never a rule's reading"
    (clause,) = evaluate_rule(
        DecisionRule(clauses=(RateDrop(max_points=10, reading="tone"),)), readings, **_ARMS
    ).clauses
    assert (clause.outcome, clause.statement) == ("keep", "no tone rate drops by more than 10 points")


# --- a rule reads every check, each on its better end, as the declaration offers it ------------------------------


def _hallucinated(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
    """``bundle`` with a lower-is-better boolean ``hallucinated``, true on every trial whose label missed.

    Its per-cell summaries are the label check's, counted the other way up, and its contrast is ``accuracy``'s on
    the measure's own scale (the arm minus the control, so a fall in hallucinations is a negative difference).
    """
    name = "hallucinated"
    descriptor = MetricDescriptor(
        name=name,
        data_type="boolean",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description="The answer asserted something the case does not hold.",
        higher_is_better=False,
        merit_axis="quality",
    )

    def flipped(summary: Any) -> Any:
        n_true = summary.n - round(summary.mean * summary.n)
        return summary.model_copy(update={"name": name, "rate": n_true / summary.n, "n_true": n_true, "mean": None})

    cells = [
        cell.model_copy(
            update={
                "measures": cell.measures.model_copy(
                    update={
                        "measures": [
                            *cell.measures.measures,
                            *(flipped(s) for s in cell.measures.measures if s.name == ACCURACY_MEASURE),
                        ]
                    }
                )
            }
        )
        for cell in bundle.cell_measures
    ]

    def negated(interval: tuple[float, float] | None) -> tuple[float, float] | None:
        return None if interval is None else (-interval[1], -interval[0])

    families = [
        family.model_copy(
            update={
                "comparisons": [
                    *family.comparisons,
                    *(
                        c.model_copy(
                            update={
                                "name": name,
                                "higher_is_better": False,
                                "interval": negated(c.interval),
                                "bounded_interval": negated(c.bounded_interval),
                            }
                        )
                        for c in family.comparisons
                        if c.name == ACCURACY_MEASURE
                    ),
                ]
            }
        )
        for family in bundle.multiple_comparisons.families
    ]
    return bundle.model_copy(
        update={
            "cell_measures": cells,
            "measure_catalog": {**bundle.measure_catalog, name: descriptor},
            "multiple_comparisons": bundle.multiple_comparisons.model_copy(update={"families": families}),
        }
    )


async def test_a_lower_is_better_check_reads_its_fall_as_a_gain_and_settles_switch() -> None:
    """``hallucinated`` true on 20 of 60 trials for the control and 2 of 60 for v2 is a 30-point gain, not a drop."""
    bundle, control, candidate = await _ruled([2] * 20, [3] * 18 + [2] * 2)
    readings = rule_readings(_hallucinated(bundle), control, candidate, question_id="q-g2")
    (fewer,) = [rate for rate in readings.rates if rate.measure == "hallucinated"]
    assert (fewer.control_passed, fewer.control_n, fewer.candidate_passed, fewer.candidate_n) == (40, 60, 58, 60)
    (clause,) = evaluate_rule(DecisionRule(clauses=(RateDrop(max_points=10),)), readings, **_ARMS).clauses
    assert clause.outcome == "switch", clause.reason


def exact_match(case: Mapping[str, Any], answer: Any) -> bool:
    """The answer is the case's label."""
    return bool(answer == case["label"])


async def test_a_pass_fail_scorer_is_a_reading_the_rule_declares_and_reads() -> None:
    cases = [{"text": f"queue request {index}", "label": "queued"} for index in range(7)]
    arms = {"production": _passing([3] * 7, 0.0031), "candidate": _passing([2] * 7, 0.0021)}
    comparison = await Eval(cases, arms, [exact_match], k=3, name="scored").run(store=":memory:")
    storage, profile = comparison.host.storage, comparison.host.profile
    named = _G2_ASKED.model_copy(
        update={"decision": DecisionRule(clauses=(RateDrop(max_points=10, reading="exact_match"),))}
    )
    design = minimal_declaration().model_copy(update={"questions": [named]})
    refuse_an_undeclarable_design(design, behavior="b", template=None, profile=profile)

    campaign = off_loop(get_campaign, storage, comparison.campaign_id, comparison.scope_id)
    assert campaign.declared_design is not None
    campaign = campaign.model_copy(
        update={"declared_design": campaign.declared_design.model_copy(update={"questions": [_G2_ASKED]})}
    )
    bundle = off_loop(assemble_context_bundle, campaign, storage=storage, profile=profile)
    assert bundle.design.control_arm is not None
    control = bundle.design.control_arm.variant_key
    (candidate,) = [board.variant_key for board in bundle.scoreboard if board.variant_key != control]
    readings = rule_readings(bundle, control, candidate, question_id="q-g2")
    (scored,) = [rate for rate in readings.rates if rate.measure == "exact_match"]
    assert (scored.control_passed, scored.control_n, scored.candidate_passed, scored.candidate_n) == (21, 21, 14, 21)
    assert named.decision is not None
    (alone,) = evaluate_rule(named.decision, readings, **_ARMS).clauses
    assert alone.outcome == "keep" and alone.rate == scored
    every = evaluate_rule(DecisionRule(clauses=(RateDrop(max_points=10),)), readings, **_ARMS).clauses[0]
    assert every.outcome == "keep" and isinstance(every.clause, RateDrop) and scored in readings.read_by(every.clause)
    # The declaration offers exactly the checks the page reads, wherever the run measured them.
    offered = [name for name in decision_readings(None, profile) if name in bundle.measure_catalog]
    assert offered == [rate.measure for rate in readings.rates if rate.measure is not None] == ["exact_match"]
