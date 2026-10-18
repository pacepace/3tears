"""An unsettled clause states the cases that would settle it, computed, and a memo's next step never guesses at them.

Discodon's character campaign (2026-10-17, 12 cases × 3) left its 10-point no-worse clause unsettled, and its memo's
next step proposed "Expanding the case suite to 30 cases … would formally settle question q1 for roughly $0.40". From
the memo's own numbers, settling needed about 107 cases even with no true difference: the writer understated it by
4-15x (#846). Code now projects the independent cases a switch needs from the interval the clause was read on (the
bounded interval a switch reads), with no true difference and with the difference as observed, each with its cost;
a settled clause projects nothing; repeats of a case never enter it; and the style gate replaces a next step's own
sample size with code's.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import replace
from typing import Any

from threetears.evals import Answer, Eval, Result
from threetears.evals.analysis import (
    AnalysisContextBundle,
    Report,
    assemble_context_bundle,
    build_code_only_report,
    build_report,
    generate_analysis,
    get_campaign,
    report_markdown,
)
from threetears.evals.analysis.bundle.schema import ArmScoreboard
from threetears.evals.analysis.report import DisclosureBlock, TableBlock, TextBlock
from threetears.evals.analysis.report.decision import (
    RateReading,
    RuleReadings,
    SampleProjection,
    clause_projection,
    evaluate_rule,
    project_sample,
    projected_cell,
    projection_note,
    projection_sentence,
    rule_projection,
    rule_readings,
)
from threetears.evals.analysis.report.memo_style import MemoStyle, decision_disclosures
from threetears.evals.analysis.report.verdict import rule_sentence
from threetears.evals.analysis.stats import cases_to_clear
from threetears.evals.analysis.style import lint_page, lint_prose
from threetears.evals.kernel.declaration import CostLower, DecisionRule, Question, RateDrop
import pytest

from packages.evals.tests.g2_support import G2_QUESTION
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient

#: The character campaign's rule: one 10-point no-worse clause on the pass rate.
RULE = DecisionRule(clauses=(RateDrop(max_points=10),))

#: The arms by the names a page gives them.
ARMS = {"control": "production", "candidate": "candidate"}

#: What one more case cost over every arm in the character campaign, as its headline said.
PER_CASE = 0.02557

#: The character campaign's shape: 12 paired cases × 3 repeats, production 30 of 36 to the candidate's 29, a drop of
#: 1/36 = 2.78 points, and an interval on the drop reaching 29.8 points past it (to 32.58): the half-width the issue
#: read off its contrasts table. The rule reads whatever interval the rate carries, so this one is that half-width.
CHARACTER = RuleReadings(
    rates=(RateReading("the pass rate", 30, 36, 29, 36, interval=(-(29.8 + 100 / 36) / 100, 0.27)),),
    control_cases=12,
    candidate_cases=12,
)


def _verdict_line(report: Report) -> str:
    """The page's verdict line."""
    (verdict,) = [b for b in report.blocks if isinstance(b, DisclosureBlock) and b.source == "verdict"]
    return verdict.text


def _projection_table(report: Report) -> TableBlock | None:
    """The Decision section's ``To switch`` table, or None where the page projects nothing."""
    tables = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "projection"]
    return tables[0] if tables else None


def _priced(bundle_cases: int, per_case: float) -> AnalysisContextBundle:
    """A bundle carrying only a scoreboard, two arms over ``bundle_cases`` cases costing ``per_case`` over both."""
    boards = [
        ArmScoreboard(
            variant_key=key,
            ran=3 * bundle_cases,
            usable=3 * bundle_cases,
            passed=3 * bundle_cases,
            cases=bundle_cases,
            usable_cases=bundle_cases,
            test_cost_usd=per_case * bundle_cases / 2,
        )
        for key in ("production", "candidate")
    ]
    return AnalysisContextBundle.model_construct(scoreboard=boards, cell_measures=[])


# =============================================================================
# 1. The projection itself: scaled from the interval, over cases, never trials
# =============================================================================


def test_the_scaling_is_one_over_root_n_over_independent_cases() -> None:
    """12 × (29.8 / 10)² = 106.56: 107 cases bring a 29.8-point half-width within 10. None where nothing can."""
    assert cases_to_clear(29.8, 12, 10.0) == 107
    assert cases_to_clear(10.0, 12, 10.0) == 12, "an interval already at the margin needs the cases it has"
    assert cases_to_clear(29.8, 12, 0.0) is None
    assert cases_to_clear(29.8, 12, -2.0) is None
    with pytest.raises(ValueError):
        cases_to_clear(29.8, 0, 10.0)


def test_a_character_shaped_clause_projects_107_cases_with_no_true_difference_and_its_cost() -> None:
    """The acceptance fixture: 12 paired cases, an interval reaching 29.8 points past the point estimate, margin 10.

    With no true difference the interval must reach no further than 10: 12 × (29.8 / 10)² = 106.56, so 107 cases,
    at $0.02557 a case $2.74, said to two figures as $2.7. At the observed drop of 2.78 points it may reach only
    10 − 2.78 = 7.22: 12 × (29.8 / 7.22)² = 204.3, so 205 cases, $5.24, said $5.2.
    """
    verdict = evaluate_rule(RULE, CHARACTER, **ARMS)
    assert verdict.outcome == "unsettled"
    projection = clause_projection(verdict.clauses[0], CHARACTER, per_case_usd=PER_CASE)
    assert projection == SampleProjection(cases=12, if_none=107, if_observed=205, per_case_usd=PER_CASE)
    assert rule_projection(verdict, per_case_usd=PER_CASE) == projection
    assert projected_cell(projection, projection.if_none) == "107 cases, $2.7"
    assert projected_cell(projection, projection.if_observed) == "205 cases, $5.2"
    sentence = projection_sentence(projection)
    assert sentence == (
        "A switch needs about 107 independent cases if the arms do not differ, 205 if they differ as observed; only "
        "new cases count, not repeats."
    )
    assert lint_prose(sentence) == []


def test_the_verdict_line_says_the_cases_a_switch_needs_then_what_each_case_costs() -> None:
    verdict = evaluate_rule(RULE, CHARACTER, **ARMS)
    line = rule_sentence(verdict, bundle=_priced(12, PER_CASE), **ARMS)
    assert line == (
        "Unsettled between production and candidate: candidate's drop is not shown to stay within 10 points (it could "
        "be as much as 32.6). A switch needs about 107 independent cases if the arms do not differ, 205 if they differ "
        "as observed; only new cases count, not repeats. Each more case costs about $0.02557 over every arm."
    )
    assert lint_prose(line) == []


def test_a_template_too_small_to_settle_says_new_cases_must_be_written() -> None:
    projection = project_sample(half_width=29.8, cases=12, margin=10.0, observed=100 / 36)
    note = projection_note([projection], template_cases=12)
    assert note.endswith("This template has 12 cases; a switch needs about 107 to 205, so new cases must be written.")
    assert "about" in note and "Repeats of a case do not count; new cases do." in note
    assert lint_prose(note) == []
    # A template holding the cases a switch needs says nothing of writing more.
    assert "must be written" not in projection_note([projection], template_cases=300)


def test_a_drop_past_the_margin_says_no_number_of_cases_shows_it_within() -> None:
    """At an observed 28.6-point drop no sample shows the drop within 10: the projection says so, not a number."""
    projection = project_sample(half_width=64.1, cases=7, margin=10.0, observed=28.6)
    assert (projection.if_none, projection.if_observed) == (288, None)
    assert projected_cell(projection, None) == "none would"
    assert projection_sentence(projection) == (
        "A switch needs about 288 independent cases if the arms do not differ, and none would if they differ as "
        "observed; only new cases count, not repeats."
    )
    assert lint_prose(projection_sentence(projection)) == []


def test_tripling_the_repeats_at_a_fixed_case_count_leaves_the_projection_unchanged() -> None:
    """Three times the trials (90 of 108 to 87), the same twelve cases and the same interval: the same cases."""
    tripled = replace(
        CHARACTER,
        rates=(replace(CHARACTER.rates[0], control_passed=90, control_n=108, candidate_passed=87, candidate_n=108),),
    )
    once = evaluate_rule(RULE, CHARACTER, **ARMS)
    thrice = evaluate_rule(RULE, tripled, **ARMS)
    assert rule_projection(thrice, per_case_usd=PER_CASE) == rule_projection(once, per_case_usd=PER_CASE)


# =============================================================================
# 2. A settled clause, or a settled rule, projects nothing
# =============================================================================


def test_a_settled_clause_renders_no_projection() -> None:
    held = replace(CHARACTER, rates=(replace(CHARACTER.rates[0], interval=(-0.06, 0.05)),))
    verdict = evaluate_rule(RULE, held, **ARMS)
    assert verdict.outcome == "switch"
    assert clause_projection(verdict.clauses[0], held, per_case_usd=PER_CASE) is None
    assert rule_projection(verdict, per_case_usd=PER_CASE) is None
    assert "A switch needs" not in rule_sentence(verdict, bundle=_priced(12, PER_CASE), **ARMS)


def test_a_rule_kept_on_cost_projects_nothing_for_its_open_rate_clause() -> None:
    """Keep is the default, and a cost observed higher has settled it: no number of cases changes that decision."""
    rule = DecisionRule(clauses=(RateDrop(max_points=10), CostLower()))
    costly = replace(CHARACTER, control_cost=0.003, candidate_cost=0.004)
    verdict = evaluate_rule(rule, costly, **ARMS)
    assert [clause.outcome for clause in verdict.clauses] == ["unsettled", "keep"]
    assert clause_projection(verdict.clauses[1], costly, per_case_usd=PER_CASE) is None
    assert rule_projection(verdict, per_case_usd=PER_CASE) is None


def test_an_open_cost_clause_beside_an_open_rate_clause_needs_the_most_either_needs() -> None:
    """Cost has no margin: arms that do not differ are never shown cheaper, so the rule's no-difference count is none.

    The cost interval is −1.5 to 0.5 cents around −0.5: half-width 1 over 12 cases needs 12 × (1 / 0.5)² = 48 at the
    observed difference, fewer than the rate's 205, so the rule needs 205 as observed and none with no difference.
    """
    rule = DecisionRule(clauses=(RateDrop(max_points=10), CostLower()))
    cheaper = replace(CHARACTER, control_cost=0.02, candidate_cost=0.015, cost_interval=(-0.015, 0.005))
    verdict = evaluate_rule(rule, cheaper, **ARMS)
    assert [clause.outcome for clause in verdict.clauses] == ["unsettled", "unsettled"]
    cost = clause_projection(verdict.clauses[1], cheaper)
    assert cost is not None and (cost.if_none, cost.if_observed) == (None, 48)
    projection = rule_projection(verdict)
    assert projection is not None and (projection.if_none, projection.if_observed) == (None, 205)
    assert projection_sentence(projection) == (
        "A switch needs about 205 independent cases if the arms differ as observed, and none would if they do not "
        "differ; only new cases count, not repeats."
    )


def test_a_rate_with_no_interval_projects_nothing() -> None:
    bare = replace(CHARACTER, rates=(replace(CHARACTER.rates[0], interval=None),))
    verdict = evaluate_rule(RULE, bare, **ARMS)
    assert verdict.outcome == "unsettled"
    assert rule_projection(verdict, per_case_usd=PER_CASE) is None


# =============================================================================
# 3. On the page, from the interval the bundle carries
# =============================================================================


def _by_case(passes: Sequence[int], cost: float) -> Callable[[Mapping[str, Any]], Coroutine[Any, Any, Answer]]:
    """A runner answering case ``i`` right on its first ``passes[i]`` repeats and wrong after, each at ``cost``."""
    calls: Counter[str] = Counter()

    async def run(case: Mapping[str, Any]) -> Answer:
        calls[case["text"]] += 1
        right = calls[case["text"]] <= passes[int(case["text"].rsplit(" ", 1)[1])]
        return Answer(
            case["label"] if right else "skipped", model="m", input_tokens=100, output_tokens=10, cost_usd=cost
        )

    return run


#: The character campaign's question, with its one rate clause, declared before the run.
ASKED = Question(id="q-g2", text=G2_QUESTION, decision=RULE, asked_at="2000-01-01T00:00:00+00:00")

#: Twelve cases a side: production passing every trial, the candidate missing one trial of the last case.
CONTROL_PASSES = [3] * 12
CANDIDATE_PASSES = [3] * 11 + [2]


async def _ruled(k: int = 3) -> tuple[Result, AnalysisContextBundle]:
    """The twelve cases × ``k`` repeats, each arm passing the same share of each case's repeats, with the rule asked
    before the bundle is assembled, so its family carries the interval the clause reads."""
    cases = [{"text": f"queue request {index}", "label": "queued"} for index in range(12)]
    scale = k // 3
    arms = {
        "production": _by_case([passes * scale for passes in CONTROL_PASSES], 0.0031),
        "candidate": _by_case([passes * scale for passes in CANDIDATE_PASSES], 0.0021),
    }
    comparison = await Eval(cases, arms, ["label"], k=k, name="character").run(store=":memory:")
    storage = comparison.host.storage
    campaign = off_loop(get_campaign, storage, comparison.campaign_id, comparison.scope_id)
    assert campaign.declared_design is not None
    campaign = campaign.model_copy(
        update={"declared_design": campaign.declared_design.model_copy(update={"questions": [ASKED]})}
    )
    bundle = off_loop(assemble_context_bundle, campaign, storage=storage, profile=comparison.host.profile)
    return comparison, bundle


def _expected(bundle: AnalysisContextBundle) -> tuple[int, int]:
    """The cases the page should project, derived by hand from the bounded interval the bundle carries: its reach past
    the point estimate ``h`` over 12 cases needs ``⌈12 · (h / 10)²⌉`` with no difference and ``⌈12 · (h / (10 − d))²⌉``
    at the observed drop ``d``."""
    assert bundle.design.control_arm is not None
    control = bundle.design.control_arm.variant_key
    (candidate,) = [board.variant_key for board in bundle.scoreboard if board.variant_key != control]
    (rate,) = [
        rate for rate in rule_readings(bundle, control, candidate, question_id="q-g2").rates if rate.measure is None
    ]
    worst = rate.worst_drop()
    assert worst is not None
    reach, drop = worst - rate.point_drop(), rate.point_drop()
    return math.ceil(12 * (reach / 10) ** 2), math.ceil(12 * (reach / (10 - drop)) ** 2)


async def test_an_unsettled_page_tables_the_cases_a_switch_needs_with_their_cost() -> None:
    _, bundle = await _ruled()
    none, observed = _expected(bundle)
    assert 12 < none < observed, "twelve cases settle nothing here"
    report = build_code_only_report(bundle, assembled_at="2026-10-17T12:00:00+00:00", campaign_name="character")
    line = _verdict_line(report)
    assert f"A switch needs about {none} independent cases if the arms do not differ, {observed} if they" in line
    assert "Each more case costs about $" in line
    table = _projection_table(report)
    assert table is not None and table.title == "To switch"
    (row,) = table.values()
    assert row["clause"] == "no pass rate drops by more than 10 points"
    assert str(row["none"]).startswith(f"{none} cases, $") and str(row["observed"]).startswith(f"{observed} cases, $")
    (note,) = [
        b.text
        for b in report.blocks
        if isinstance(b, DisclosureBlock) and b.section == "decision" and "count is about" in b.text
    ]
    assert note.endswith(
        f"This template has 12 cases; a switch needs about {none} to {observed}, so new cases must be written."
    )
    assert lint_page(report_markdown(report), question=G2_QUESTION) == []


async def test_tripling_k_on_the_page_leaves_the_projection_unchanged() -> None:
    """The same twelve cases at k = 9, each passing the same share of its repeats: the same cases, the cost aside."""
    _, thrice = await _ruled(k=9)
    _, once = await _ruled()
    assert _expected(thrice) == _expected(once)
    tables = [
        _projection_table(build_code_only_report(b, assembled_at="2026-10-17T12:00:00+00:00", campaign_name="c"))
        for b in (once, thrice)
    ]
    counts = [
        [str(row[key]).split(" cases")[0] for row in table.values() for key in ("none", "observed")]
        for table in tables
        if table is not None
    ]
    assert len(counts) == 2 and counts[0] == counts[1]


async def test_a_settled_page_tables_no_projection() -> None:
    """Twenty cases a side, the candidate failing half outright: the drop's whole interval is past the margin."""
    cases = [{"text": f"queue request {index}", "label": "queued"} for index in range(20)]
    arms = {"production": _by_case([3] * 20, 0.0031), "candidate": _by_case([3] * 10 + [0] * 10, 0.0021)}
    comparison = await Eval(cases, arms, ["label"], k=3, name="settled").run(store=":memory:")
    storage = comparison.host.storage
    campaign = off_loop(get_campaign, storage, comparison.campaign_id, comparison.scope_id)
    assert campaign.declared_design is not None
    campaign = campaign.model_copy(
        update={"declared_design": campaign.declared_design.model_copy(update={"questions": [ASKED]})}
    )
    bundle = off_loop(assemble_context_bundle, campaign, storage=storage, profile=comparison.host.profile)
    report = build_code_only_report(bundle, assembled_at="2026-10-17T12:00:00+00:00", campaign_name="settled")
    assert _verdict_line(report).startswith("Keep production")
    assert _projection_table(report) is None
    assert "A switch needs" not in report_markdown(report)


# =============================================================================
# 4. A next step states no sample size of its own
# =============================================================================


#: The character memo's next step, as Discodon's writer put it.
THIRTY = "Expand the case suite to 30 cases, which would formally settle question q1 for roughly $0.40."


def _style(sample: SampleProjection | None) -> MemoStyle:
    return MemoStyle(bundle=_priced(12, PER_CASE), names={}, sample=sample, samples=() if sample is None else (sample,))


def test_a_next_step_proposing_its_own_sample_size_is_code_s_projection_instead() -> None:
    projection = project_sample(half_width=29.8, cases=12, margin=10.0, observed=100 / 36, per_case_usd=PER_CASE)
    style = _style(projection)
    assert style.sentence(THIRTY, decides=False, step=True) == projection_sentence(projection)
    assert style.proposed == [THIRTY]
    assert any("a sample size no code computed" in line and "30 cases" in line for line in decision_disclosures(style))
    # A step's why loses the sentence proposing a size, and keeps the rest.
    why = "Run 30 more cases for under $1. It would show whether the drop is real."
    assert style.prose(why, step=True) == "It would show whether the drop is real."
    # Every way a size is said is one.
    for said in (
        "Add at least 30 independent test cases.",
        "Run 60 cases for approximately 0.20 USD.",
        "Grow the suite to 40.",
        "Run 30-60 more cases.",
    ):
        assert style.proposes_sample(said), said


def test_a_next_step_citing_the_projection_or_the_run_s_own_cases_stands() -> None:
    projection = project_sample(half_width=29.8, cases=12, margin=10.0, observed=100 / 36, per_case_usd=PER_CASE)
    style = _style(projection)
    for said in (
        "Write about 107 new cases before the next run.",
        "Write 107 to 205 new cases, about $2.7 to $5.2.",
        "Beyond this template's 12 cases, write new ones.",
        "Rerun case 4 with a longer turn limit.",
        "Repeat the run at a cost of $0.40.",
    ):
        assert not style.proposes_sample(said), said
        assert style.sentence(said, decides=False, step=True) == said
    assert style.proposed == []
    # Off a next step the gate does not read: a finding's title is held to its own rules.
    assert style.sentence(THIRTY, decides=False) == THIRTY


def test_with_no_projection_a_next_step_s_own_sample_size_is_not_printed() -> None:
    style = _style(None)
    assert style.sentence(THIRTY, decides=False, step=True) == ""
    assert style.proposed == [THIRTY]


async def test_a_thirty_case_next_step_does_not_survive_the_style_gate_on_the_page() -> None:
    comparison, bundle = await _ruled()
    none, observed = _expected(bundle)
    memo = {
        "findings": [],
        "decisions": [],
        "questions": [
            {"question_id": "q-g2", "resolution": "unanswerable", "answer": "Too few cases to tell.", "rests_on": []}
        ],
        "next": [
            {
                "title": THIRTY,
                "why": "At least 30 independent test cases would settle it. It also tells the arms apart on tone.",
                "leverage": "high",
                "lever": "",
            }
        ],
    }
    analysis, _ = await generate_analysis(
        bundle,
        prompt=PROMPT,
        model=MODEL,
        client=FixturedClient(json.dumps(memo)),
        prompt_id=PROMPT_ID,
        bundle_assembled_at="2026-10-17T12:00:00+00:00",
        profile=comparison.host.profile,
    )
    report = build_report(analysis, campaign_name="character")
    (step,) = [b for b in report.blocks if isinstance(b, TextBlock) and b.role == "next_step"]
    assert step.body.startswith(f"A switch needs about {none} independent cases if the arms do not differ, {observed}")
    (why,) = [b for b in report.blocks if isinstance(b, TextBlock) and b.role == "next_step_why"]
    assert why.body == "It also tells the arms apart on tone."
    markdown = report_markdown(report)
    assert "30 cases" not in markdown.split("## Methods")[0] and "$0.40" not in markdown.split("## Methods")[0]
    assert "a sample size no code computed" in markdown
