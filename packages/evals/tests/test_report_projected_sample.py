"""An unsettled clause states the cases that would settle it, computed, and a memo never guesses at them.

Discodon's character campaign (2026-10-17, 12 cases × 3) left its 10-point no-worse clause unsettled, and its memo's
next step proposed "Expanding the case suite to 30 cases … would formally settle question q1 for roughly $0.40" (#846).
Code now projects the independent cases a switch needs, with no true difference and with the difference as observed,
each with its cost. It computes the bounded interval the clause was read on again at each count, over the run's own
cases repeated, rather than scaling it as ``1/√n``: from a dozen cases that scaling overstates the bounded interval's
need by up to about twice. The issue's "about 107" scaled the contrasts table's t interval, which never settles a
clause, and is deliberately not what code computes. On the character-shaped fixture below the bounded interval needs
121 cases with no true difference and 223 as observed, where scaling it would have said 158 and 302. A settled clause
projects nothing, a count is always more than the run had, and the style gate replaces a sample size or a cost the
writer proposes with code's.
"""

from __future__ import annotations

import json
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
from threetears.evals.analysis.bundle.schema import ArmScoreboard, MultipleComparisons
from threetears.evals.analysis.generator import build_user_message
from threetears.evals.analysis.references import cell_aliases
from threetears.evals.analysis.report import DisclosureBlock, TableBlock, TextBlock
from threetears.evals.analysis.report.decision import (
    CasesNeeded,
    RateReading,
    RuleReadings,
    SampleProjection,
    clause_projection,
    evaluate_rule,
    project_rate,
    project_sample,
    projected_cell,
    projection_note,
    projection_sentence,
    rule_projection,
)
from threetears.evals.analysis.report.memo_style import MemoStyle, decision_disclosures
from threetears.evals.analysis.report.verdict import rule_sentence
from threetears.evals.analysis.stats import bounded_cases_to_clear, bounded_difference_interval, cases_to_clear
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

#: The character campaign's shape: 12 paired cases × 3 repeats, production passing 30 of 36 and the candidate 29, each
#: case's passes out of three. The paired differences spread as the campaign's did: its t interval at the family's
#: 97.5% reaches 31 points past the drop of 2.78, where the issue read 29.8 off the contrasts table.
CONTROL_PASSES = [2, 3, 2, 3, 1, 3, 2, 3, 2, 3, 3, 3]
CANDIDATE_PASSES = [3, 2, 3, 2, 3, 1, 3, 2, 3, 2, 3, 2]
CONTROL = tuple(passes / 3 for passes in CONTROL_PASSES)
CANDIDATE = tuple(passes / 3 for passes in CANDIDATE_PASSES)

#: A family of two rates (the pass rate and the label check), so each interval is at 97.5%.
LEVEL = 0.975

#: The bounded interval the clause reads: -39.0 to 38.9 points.
BOUNDED = bounded_difference_interval(CONTROL, CANDIDATE, paired=True, value_range=(0.0, 1.0), confidence=LEVEL)

CHARACTER = RuleReadings(
    rates=(
        RateReading("the pass rate", 30, 36, 29, 36, interval=BOUNDED, samples=(CONTROL, CANDIDATE), confidence=LEVEL),
    ),
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
    """A bundle carrying only a scoreboard, two arms over ``bundle_cases`` cases × 3 costing ``per_case`` over both."""
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
    return AnalysisContextBundle.model_construct(
        scoreboard=boards, cell_measures=[], run_summaries=[], multiple_comparisons=MultipleComparisons()
    )


# =============================================================================
# 1. The projection: the bounded interval computed again at each count, over cases, never trials
# =============================================================================


def test_a_t_interval_scales_as_one_over_root_n_over_independent_cases() -> None:
    """12 × (29.8 / 10)² = 106.56: 107 cases bring a 29.8-point t half-width within 10, the issue's figure."""
    assert cases_to_clear(29.8, 12, 10.0) == 107
    assert cases_to_clear(29.8, 12, 0.0) is None
    with pytest.raises(ValueError):
        cases_to_clear(29.8, 0, 10.0)


def test_the_bounded_projection_is_the_fewest_cases_whose_interval_clears() -> None:
    """With no true difference the interval may reach the whole 10-point margin below its estimate; as observed, the
    margin less the 2.78-point drop. The count is the fewest repeated cases at which it does: searched no further than
    one case fewer, it is not reached. Scaling the 12-case bounded interval instead would have said 158 and 302, and
    the order the cases are listed in changes nothing."""
    margins = {"none": 0.10, "observed": 0.10 - 1 / 36}
    found = {
        side: bounded_cases_to_clear(
            CONTROL, CANDIDATE, paired=True, value_range=(0.0, 1.0), confidence=LEVEL, room=room
        )
        for side, room in margins.items()
    }
    assert found == {"none": 121, "observed": 223}
    for side, room in margins.items():
        cases = found[side]
        assert cases is not None
        short = bounded_cases_to_clear(
            CONTROL, CANDIDATE, paired=True, value_range=(0.0, 1.0), confidence=LEVEL, room=room, cap=cases - 1
        )
        assert short == cases, f"{side}: {cases - 1} cases do not clear"
    shuffled = (CONTROL[::-1], CANDIDATE[::-1])
    assert bounded_cases_to_clear(*shuffled, paired=True, value_range=(0.0, 1.0), confidence=LEVEL, room=0.10) == 121
    assert BOUNDED is not None
    reach = 100 * (-1 / 36 - BOUNDED[0])
    assert (cases_to_clear(reach, 12, 10.0), cases_to_clear(reach, 12, 10.0 - 100 / 36)) == (158, 302)


def test_a_character_shaped_clause_projects_its_cases_with_their_cost() -> None:
    """121 cases at $0.02557 a case are $3.09, said to two figures as $3.1; 223 are $5.70, $5.7."""
    verdict = evaluate_rule(RULE, CHARACTER, **ARMS)
    assert verdict.outcome == "unsettled"
    projection = clause_projection(verdict.clauses[0], CHARACTER, per_case_usd=PER_CASE)
    assert projection == SampleProjection(
        cases=12, if_none=CasesNeeded("more", 121), if_observed=CasesNeeded("more", 223), per_case_usd=PER_CASE
    )
    assert rule_projection(verdict, per_case_usd=PER_CASE) == projection
    assert projected_cell(projection, projection.if_none) == "121 cases, $3.1"
    assert projected_cell(projection, projection.if_observed) == "223 cases, $5.7"
    sentence = projection_sentence(projection)
    assert sentence == (
        "A switch needs about 121 independent cases if the arms do not differ, about 223 if they differ as observed; it "
        "counts cases, not repeats."
    )
    assert lint_prose(sentence) == []


def test_the_verdict_line_says_the_cases_a_switch_needs_then_what_each_case_costs() -> None:
    verdict = evaluate_rule(RULE, CHARACTER, **ARMS)
    line = rule_sentence(verdict, bundle=_priced(12, PER_CASE), **ARMS)
    assert line == (
        "Unsettled between production and candidate: candidate's drop is not shown to stay within 10 points (it could "
        "be as much as 39). A switch needs about 121 independent cases if the arms do not differ, about 223 if they "
        "differ as observed; it counts cases, not repeats. Each more case costs about $0.02557 over every arm."
    )
    assert lint_prose(line) == []


def test_the_note_says_each_count_is_about_and_more_than_the_run_used() -> None:
    projection = rule_projection(evaluate_rule(RULE, CHARACTER, **ARMS), per_case_usd=PER_CASE)
    assert projection is not None
    note = projection_note([projection], run_cases=12)
    assert note == (
        "Each count is about: it reads the interval again over this run's cases repeated to that many, so it assumes "
        "new cases vary as these did. Repeats of a case narrow an interval far less than new cases. This run used 12 "
        "cases; a switch needs about 121 to 223, more than it ran."
    )
    assert lint_prose(note) == []


def test_a_drop_past_the_margin_says_no_number_of_cases_shows_it_within() -> None:
    """The candidate passing one trial in three where production passes all: a 66.7-point drop, past 10 on any count."""
    rate = RateReading(
        "the pass rate", 36, 36, 12, 36, samples=((1.0,) * 12, (1 / 3,) * 6 + (1 / 3,) * 6), confidence=LEVEL
    )
    projection = project_rate(rate, cases=12, margin=10.0)
    assert projection is not None and projection.if_observed == CasesNeeded("never")
    assert projection.if_none.kind == "more" and projection.if_none.cases > 12
    assert projected_cell(projection, projection.if_observed) == "none would"
    assert projection_sentence(projection) == (
        f"A switch needs about {projection.if_none.cases} independent cases if the arms do not differ, and none would "
        "if they differ as observed; it counts cases, not repeats."
    )


def test_a_switch_kept_open_by_lost_trials_says_so_never_fewer_cases_than_the_run() -> None:
    """Sixty cases a side, every usable trial passing, the candidate losing 30 of its 180 trials: the interval keeps the
    drop within 8 points, but had every lost trial failed the drop would be 16.7. More cases are not what is missing,
    and no count says fewer cases than the run had (the defect: "about 5 … 7 if as observed" after 12 had run)."""
    readings = RuleReadings(
        rates=(
            RateReading(
                "the pass rate",
                180,
                180,
                150,
                150,
                0,
                30,
                interval=bounded_difference_interval(
                    [1.0] * 60, [1.0] * 60, paired=True, value_range=(0.0, 1.0), confidence=LEVEL
                ),
                samples=((1.0,) * 60, (1.0,) * 60),
                confidence=LEVEL,
            ),
        ),
        control_cases=60,
        candidate_cases=60,
    )
    verdict = evaluate_rule(RULE, readings, **ARMS)
    assert verdict.outcome == "unsettled"
    projection = rule_projection(verdict)
    assert projection is not None
    assert (projection.if_none, projection.if_observed) == (CasesNeeded("enough", 60), CasesNeeded("lost"))
    assert projected_cell(projection, projection.if_none) == "this run's 60 cases"
    assert projected_cell(projection, projection.if_observed) == "re-run the lost trials"
    assert projection_sentence(projection) == (
        "This run's interval already keeps the drop within the margin as observed: re-running its lost trials, not "
        "more cases, would settle it."
    )
    # A count is always past the run's own: a search that clears below it reads as more than the run, at the floor.
    assert project_sample(half_width=4.0, cases=12, margin=10.0, observed=8.0).if_observed == CasesNeeded("more", 48)
    assert project_sample(half_width=1.0, cases=3, margin=10.0, observed=0.0).if_none == CasesNeeded("more", 5)


def test_the_projection_counts_cases_and_reads_their_shares_not_their_trial_counts() -> None:
    """What this proves, and no more: the projection reads each case's share and how many cases there are. Three times
    the trials at the same shares (90 of 108 to 87) is the same projection, since no trial count enters it. Repeats do
    narrow an interval, by making each case's share less noisy: a sample whose shares sit nearer each other, as more
    repeats of the same cases would leave them, needs fewer cases. New cases narrow it far more."""
    tripled = replace(
        CHARACTER,
        rates=(replace(CHARACTER.rates[0], control_passed=90, control_n=108, candidate_passed=87, candidate_n=108),),
    )
    once = rule_projection(evaluate_rule(RULE, CHARACTER, **ARMS))
    thrice = rule_projection(evaluate_rule(RULE, tripled, **ARMS))
    assert once is not None and thrice == once
    # Each case's share pulled a third of the way toward the arm's mean: the noise more repeats take out.
    steadier = tuple(tuple(share + (sum(arm) / 12 - share) / 3 for share in arm) for arm in (CONTROL, CANDIDATE))
    calmer = project_rate(replace(CHARACTER.rates[0], samples=(steadier[0], steadier[1])), cases=12, margin=10.0)
    assert calmer is not None and calmer.if_none.cases < once.if_none.cases


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

    The cost interval is −1.5 to 0.5 cents around −0.5: a t half-width of 1 over 12 cases needs 12 × (1 / 0.5)² = 48 at
    the observed difference, fewer than the rate's 223, so the rule needs 223 as observed and none with no difference.
    The note reads the rule's projection, not each clause's, so it never mixes the cost's 48 into its range.
    """
    rule = DecisionRule(clauses=(RateDrop(max_points=10), CostLower()))
    cheaper = replace(CHARACTER, control_cost=0.02, candidate_cost=0.015, cost_interval=(-0.015, 0.005))
    verdict = evaluate_rule(rule, cheaper, **ARMS)
    assert [clause.outcome for clause in verdict.clauses] == ["unsettled", "unsettled"]
    cost = clause_projection(verdict.clauses[1], cheaper)
    assert cost is not None and (cost.if_none, cost.if_observed) == (CasesNeeded("never"), CasesNeeded("more", 48))
    projection = rule_projection(verdict)
    assert projection is not None
    assert (projection.if_none, projection.if_observed) == (CasesNeeded("never"), CasesNeeded("more", 223))
    assert projection_sentence(projection) == (
        "A switch needs about 223 independent cases if the arms differ as observed, and none would if they do not "
        "differ; it counts cases, not repeats."
    )
    note = projection_note([projection], run_cases=12, cost_clause=True)
    assert "A cost clause has no margin, so arms that do not differ are never shown cheaper" in note
    assert note.endswith("This run used 12 cases; a switch needs about 223, more than it ran.")


def test_a_rate_with_nothing_to_project_from_projects_nothing() -> None:
    bare = replace(CHARACTER, rates=(replace(CHARACTER.rates[0], samples=None),))
    verdict = evaluate_rule(RULE, bare, **ARMS)
    assert verdict.outcome == "unsettled"
    assert rule_projection(verdict, per_case_usd=PER_CASE) is None


# =============================================================================
# 3. On the page, from the per-case values the bundle carries
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


async def _ruled(
    k: int = 3,
    *,
    control: Sequence[int] = CONTROL_PASSES,
    candidate: Sequence[int] = CANDIDATE_PASSES,
    third: bool = False,
) -> tuple[Result, AnalysisContextBundle]:
    """The cases × ``k`` repeats, each arm passing the same share of each case's repeats at any ``k``, with the rule
    asked before the bundle is assembled, so its family carries what the clause reads; ``third`` adds a second
    candidate passing as the first does."""
    cases = [{"text": f"queue request {index}", "label": "queued"} for index in range(len(control))]
    scale = k // 3
    arms = {
        "production": _by_case([passes * scale for passes in control], 0.0031),
        "candidate": _by_case([passes * scale for passes in candidate], 0.0021),
    }
    if third:
        arms["other candidate"] = _by_case([passes * scale for passes in candidate], 0.0022)
    comparison = await Eval(cases, arms, ["label"], k=k, name="character").run(store=":memory:")
    storage = comparison.host.storage
    campaign = off_loop(get_campaign, storage, comparison.campaign_id, comparison.scope_id)
    assert campaign.declared_design is not None
    campaign = campaign.model_copy(
        update={"declared_design": campaign.declared_design.model_copy(update={"questions": [ASKED]})}
    )
    bundle = off_loop(assemble_context_bundle, campaign, storage=storage, profile=comparison.host.profile)
    return comparison, bundle


def _page(bundle: AnalysisContextBundle) -> Report:
    return build_code_only_report(bundle, assembled_at="2026-10-17T12:00:00+00:00", campaign_name="character")


async def test_the_bundle_carries_what_the_bounded_interval_read_and_the_writer_is_not_shown_it() -> None:
    _, bundle = await _ruled()
    (family,) = bundle.multiple_comparisons.families
    (passed,) = [comparison for comparison in family.comparisons if comparison.reading == "trial"]
    assert passed.case_values is not None and family.interval_level == LEVEL
    control, contrast = passed.case_values
    assert sorted(control) == sorted(CONTROL) and sorted(contrast) == sorted(CANDIDATE)
    again = bounded_difference_interval(control, contrast, paired=True, value_range=(0.0, 1.0), confidence=LEVEL)
    assert again == passed.bounded_interval
    assert '"case_values"' not in build_user_message(bundle)


async def test_an_unsettled_page_tables_the_cases_a_switch_needs_with_their_cost() -> None:
    _, bundle = await _ruled()
    report = _page(bundle)
    line = _verdict_line(report)
    assert "A switch needs about 121 independent cases if the arms do not differ, about 223 if they" in line
    assert "Each more case costs about $" in line
    table = _projection_table(report)
    assert table is not None and table.title == "To switch"
    (row,) = table.values()
    assert row["clause"] == "no pass rate drops by more than 10 points"
    assert str(row["none"]).startswith("121 cases, $") and str(row["observed"]).startswith("223 cases, $")
    (note,) = [
        b.text
        for b in report.blocks
        if isinstance(b, DisclosureBlock) and b.section == "decision" and "count is about" in b.text
    ]
    assert note.endswith("This run used 12 cases; a switch needs about 121 to 223, more than it ran.")
    assert lint_page(report_markdown(report), question=G2_QUESTION) == []


async def test_tripling_k_at_the_same_shares_leaves_the_projection_unchanged_on_the_page() -> None:
    """The same twelve cases at k = 9, each passing the same share of its repeats. What this proves: the page reads
    each case's share and the number of cases, never the trial count. Repeats that changed the shares would move it."""
    tables = []
    for k in (3, 9):
        _, bundle = await _ruled(k=k)
        table = _projection_table(_page(bundle))
        assert table is not None
        tables.append([str(row[key]).split(" cases")[0] for row in table.values() for key in ("none", "observed")])
    assert tables[0] == tables[1] == ["121", "223"]


async def test_a_settled_page_tables_no_projection() -> None:
    """Twenty cases a side, the candidate failing half outright: the drop's whole interval is past the margin."""
    _, bundle = await _ruled(control=[3] * 20, candidate=[3] * 10 + [0] * 10)
    report = _page(bundle)
    assert _verdict_line(report).startswith("Keep production")
    assert _projection_table(report) is None
    assert "A switch needs" not in report_markdown(report)


# =============================================================================
# 4. What the memo says to run next states no sample size or cost of its own
# =============================================================================


#: The character memo's next step, as Discodon's writer put it.
THIRTY = "Expand the case suite to 30 cases, which would formally settle question q1 for roughly $0.40."

#: The character clause's projection, 121 and 223 cases at $0.02557 a case.
PROJECTED = SampleProjection(
    cases=12, if_none=CasesNeeded("more", 121), if_observed=CasesNeeded("more", 223), per_case_usd=PER_CASE
)


def _style(sample: SampleProjection | None) -> MemoStyle:
    return MemoStyle(bundle=_priced(12, PER_CASE), names={}, sample=sample, samples=() if sample is None else (sample,))


def test_a_next_step_proposing_its_own_sample_size_is_code_s_projection_instead() -> None:
    style = _style(PROJECTED)
    assert style.sentence(THIRTY, decides=False, plans=True) == projection_sentence(PROJECTED)
    assert style.proposed == [THIRTY]
    assert any(
        "a sample size or cost no code computed" in line and "30 cases" in line for line in decision_disclosures(style)
    )
    # A step's why loses the sentence proposing a size, and keeps the rest.
    why = "Run 30 more cases for under $1. It would show whether the drop is real."
    assert style.prose(why, plans=True) == "It would show whether the drop is real."
    # Every way a size or a cost of its own is said is one.
    for said in (
        "Add at least 30 independent test cases.",
        "Run 60 cases for approximately 0.20 USD.",
        "Grow the suite to 40.",
        "Run 30-60 more cases.",
        "Write thirty more cases.",
        "Write thirty-five more cases.",
        "Double the cases.",
        "Run 3x more cases.",
        "Add 30+ cases.",
        "Build a 50-case suite.",
        "Draw a sample of 30.",
        "Run another 50 trials.",
        "Add dozens of new scenarios.",
        "Raise k to 10.",
        "Expand the suite for roughly $0.40.",
        # A cost of its own with no count is still the writer's guess at what settling costs (#846's "or cost"); until
        # this the gate let it stand.
        "Repeat the run at a cost of $0.40.",
        "Run the new prompt for $0.",
    ):
        assert style.proposes_sample(said), said


def test_a_sentence_citing_the_projection_or_the_run_s_own_cases_stands() -> None:
    style = _style(PROJECTED)
    for said in (
        "Write about 121 new cases before the next run.",
        "Write 121 to 223 new cases, about $3.1 to $5.7.",
        "Each new case costs about $0.02557.",
        "Read the 3 cases the candidate missed.",
        "Inspect the 10 paired cases.",
        "Every arm needs at least 5 cases.",
        "Rerun case 4 with a longer turn limit.",
        "Check the suite of 3 checks.",
        "Rerun the 12 cases with the new prompt.",
    ):
        assert not style.proposes_sample(said), said
        assert style.sentence(said, decides=False, plans=True) == said
    assert style.proposed == []
    # Off what to run next the gate does not read: a finding's title is held to its own rules.
    assert style.sentence(THIRTY, decides=False) == THIRTY


def test_with_no_projection_a_sample_size_of_its_own_is_not_printed() -> None:
    style = _style(None)
    assert style.sentence(THIRTY, decides=False, plans=True) == ""
    assert style.proposed == [THIRTY]


async def _memo(comparison: Result, bundle: AnalysisContextBundle, memo: dict[str, Any]) -> Report:
    analysis, _ = await generate_analysis(
        bundle,
        prompt=PROMPT,
        model=MODEL,
        client=FixturedClient(json.dumps(memo)),
        prompt_id=PROMPT_ID,
        bundle_assembled_at="2026-10-17T12:00:00+00:00",
        profile=comparison.host.profile,
    )
    return build_report(analysis, campaign_name="character")


def _next(title: str, why: str = "", lever: str = "") -> dict[str, Any]:
    return {"title": title, "why": why, "leverage": "high", "lever": lever}


async def test_a_thirty_case_next_step_and_answer_do_not_survive_the_style_gate_on_the_page() -> None:
    comparison, bundle = await _ruled()
    finding = {
        "title": "The candidate skipped the queue on some cases",
        "body": "",
        "confidence": "low",
        "axes": [],
        "evidence": [
            {"cell": alias, "measure_id": "match", "reading": "measure"} for alias in cell_aliases(bundle.cell_measures)
        ],
        "trials": [],
        "caveats": [],
        "invalidates": [],
        "durable": "",
    }
    memo = {
        "findings": [finding],
        "decisions": [
            {
                "proposal": "Switch to candidate.",
                "disposition": "deferred",
                "confidence": "low",
                "cells": [],
                "rests_on": [0],
                "revisit_when": "After 30 more cases. When the drop is shown either way.",
            }
        ],
        "questions": [
            {
                "question_id": "q-g2",
                "resolution": "unanswerable",
                "answer": "Too few cases to tell. Thirty more cases would settle it for $0.40.",
                "rests_on": [],
            }
        ],
        "next": [_next(THIRTY, "At least 30 independent test cases would settle it. It also tells the arms apart.")],
    }
    report = await _memo(comparison, bundle, memo)
    said = projection_sentence(
        SampleProjection(cases=12, if_none=CasesNeeded("more", 121), if_observed=CasesNeeded("more", 223))
    )
    (answer,) = [b for b in report.blocks if isinstance(b, TextBlock) and b.role == "answer"]
    assert answer.body == f"Too few cases to tell. {said}"
    (step,) = [b for b in report.blocks if isinstance(b, TextBlock) and b.role == "next_step"]
    assert step.body == said and step.facts == []
    # The writer's why and facts were about the step it replaced: they go with its title.
    assert not [b for b in report.blocks if isinstance(b, TextBlock) and b.role == "next_step_why"]
    assert not [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "next_step_facts"]
    (revisit,) = [b for b in report.blocks if isinstance(b, TextBlock) and b.role == "revisit_when"]
    assert revisit.body == "When the drop is shown either way."
    page = report_markdown(report).split("## Methods")
    assert "30 cases" not in page[0] and "$0.40" not in page[0] and "Thirty" not in page[0]
    assert "a sample size or cost no code computed" in page[1]


async def test_with_two_candidates_a_next_step_may_cite_either_arm_s_projected_cases() -> None:
    """Each arm has its own To switch row, and a step citing one of them is citing code, not guessing."""
    comparison, bundle = await _ruled(third=True)
    table = _projection_table(_page(bundle))
    assert table is not None and len(table.values()) == 2
    cited = str(next(iter(table.values()))["none"]).split(" cases")[0]
    title = f"Write about {cited} new cases before the next run."
    report = await _memo(
        comparison,
        bundle,
        {
            "findings": [],
            "decisions": [],
            "questions": [
                {"question_id": "q-g2", "resolution": "unanswerable", "answer": "Too few cases.", "rests_on": []}
            ],
            "next": [_next(title)],
        },
    )
    (step,) = [b for b in report.blocks if isinstance(b, TextBlock) and b.role == "next_step"]
    assert step.body == title
