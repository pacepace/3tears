"""A memo page where the declared decision rule settled: no answer line repeating it, no finding disputing it, and no
arm label claiming what its own row contradicts.

The owner's read of the G2 report (`.29`): "**Answered.** See the decision rule above." repeated what the verdict
line and the decision table had said; a finding headlined "four cases cannot statistically confirm the drop" sat under
a ``keep`` the rule had settled on that drop, and read as disputing it; and an arm labelled "cheaper candidate" cost
more per reply, so the verdict and the tables read as a contradiction. The G2 numbers are replayed through the quick
path, with the G2 decision rule declared before the run, as the presentation tests replay them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from threetears.evals import Eval, Result
from threetears.evals.analysis import (
    AnalysisContextBundle,
    Report,
    build_code_only_report,
    build_report,
    generate_analysis,
    report_markdown,
)
from threetears.evals.analysis.references import cell_aliases
from threetears.evals.analysis.report import DisclosureBlock, TextBlock
from threetears.evals.analysis.report.decision import (
    RateReading,
    RuleReadings,
    evaluate_rule,
    parse_decision_rule,
)
from threetears.evals.analysis.report.memo_style import MemoStyle, decision_disclosures
from threetears.evals.analysis.bundle.schema import CellCoordinate
from threetears.evals.analysis.style import LABEL_CLAIMS, lint_page, lint_prose
from packages.evals.tests.g2_support import CANDIDATE, CASES, G2_QUESTION, PRODUCTION, answering, declaring
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient

#: The G2 campaign's title.
TITLE = "G2: can the cheaper model replace production?"

#: The G2 finding headline the owner read as disputing the settled keep.
DISPUTING = "Candidate model frequently underfilled the queue, but four cases cannot statistically confirm the drop"


async def _compare(candidate_cost: float, *, candidate: str = CANDIDATE) -> tuple[Result, AnalysisContextBundle]:
    """G2's pass counts (production 10 of 12, the candidate 2 of 12), the candidate at ``candidate_cost`` per reply,
    with the G2 decision rule declared before the run."""
    arms = {PRODUCTION: answering(10, 0.003092), candidate: answering(2, candidate_cost)}
    comparison = await Eval(CASES, arms, ["label"], k=3, name="Kairo queue discipline").run(store=":memory:")
    return comparison, declaring(comparison.report.bundle, G2_QUESTION)


def _finding(title: str, measure: str, aliases: dict[str, str], body: str = "") -> dict[str, Any]:
    """A finding on one reading of both arms."""
    return {
        "title": title,
        "body": body,
        "confidence": "low",
        "axes": [],
        "evidence": [{"cell": alias, "measure_id": measure, "reading": "measure"} for alias in aliases.values()],
        "trials": [],
        "caveats": [],
        "invalidates": [],
        "durable": "",
    }


async def _memo_page(
    comparison: Result, bundle: AnalysisContextBundle, findings: list[dict[str, Any]], answer: str
) -> Report:
    """The page a stored analysis of ``findings`` and one answer to the G2 question renders."""
    memo = {
        "findings": findings,
        "decisions": [],
        "questions": [{"question_id": "q-g2", "resolution": "answered", "answer": answer, "rests_on": [0]}],
        "next": [],
    }
    analysis, _ = await generate_analysis(
        bundle,
        prompt=PROMPT,
        model=MODEL,
        client=FixturedClient(json.dumps(memo)),
        prompt_id=PROMPT_ID,
        bundle_assembled_at="2026-10-14T19:23:41+00:00",
        profile=comparison.host.profile,
    )
    return build_report(analysis, campaign_name=TITLE)


def _aliases(comparison: Result, bundle: AnalysisContextBundle, candidate: str = CANDIDATE) -> dict[str, str]:
    """Each arm's cell alias, production first."""
    by_variant = {ref.split(":")[0]: alias for alias, ref in cell_aliases(bundle.cell_measures).items()}
    return {name: by_variant[comparison.arm_variants[name]] for name in (PRODUCTION, candidate)}


def _answers(report: Report) -> list[TextBlock]:
    return [block for block in report.blocks if isinstance(block, TextBlock) and block.role == "answer"]


# =============================================================================
# 1. The answer line: none where the rule settled every clause, the open clauses where it did not
# =============================================================================


async def test_a_rule_settled_on_every_clause_leaves_its_question_no_answer_line() -> None:
    comparison, bundle = await _compare(0.003301)
    aliases = _aliases(comparison, bundle)
    report = await _memo_page(
        comparison, bundle, [_finding("The candidate skipped the queue", "match", aliases)], "Keep production."
    )
    markdown = report_markdown(report)
    assert _answers(report) == []
    assert "Answered." not in markdown and "See the decision rule above." not in markdown
    assert markdown.count("Keep production") == 1, "the verdict line, once"
    assert lint_page(markdown, question=G2_QUESTION) == []


async def test_a_clause_left_open_is_the_one_answer_line() -> None:
    # The candidate costs less per reply, which four cases cannot show; its pass rate settles keep.
    comparison, bundle = await _compare(0.0029)
    aliases = _aliases(comparison, bundle)
    report = await _memo_page(
        comparison, bundle, [_finding("The candidate skipped the queue", "match", aliases)], "Keep production."
    )
    (answer,) = _answers(report)
    assert answer.body == "Still open: whether the cost per turn is lower."
    markdown = report_markdown(report)
    assert "**Partly answered.** Still open: whether the cost per turn is lower." in markdown
    assert "Keep production." not in markdown, "the memo's own answer is not printed"
    assert lint_page(markdown, question=G2_QUESTION) == []


# =============================================================================
# 2. A finding never disputes a difference a settled clause decided
# =============================================================================


async def test_a_title_disputing_a_settled_drop_is_code_s_counts_and_its_body_hedge_goes() -> None:
    comparison, bundle = await _compare(0.003301)
    aliases = _aliases(comparison, bundle)
    body = "The candidate skipped the queue. The drop is not statistically significant."
    cost = "Cheaper candidate cost more per reply, but four cases cannot establish the difference"
    report = await _memo_page(
        comparison,
        bundle,
        [_finding(DISPUTING, "match", aliases, body), _finding(cost, "production_replicating_cost", aliases)],
        "Keep production.",
    )
    markdown = report_markdown(report)
    titles = [b.body for b in report.blocks if isinstance(b, TextBlock) and b.role == "finding_title"]
    # Four numbers do not fit one sentence, so the size is said in the case floor line's own words; the cost per
    # 1,000 replies counts as one of them.
    assert titles == [
        "Cheaper candidate passed 2 of 12 on Label matched to production's 10; too few cases to estimate how large "
        "the difference is.",
        "Cheaper candidate cost $3.301 per 1,000 replies to production's $3.092; too few cases to estimate how large "
        "the difference is.",
    ]
    assert "cannot statistically confirm" not in markdown.split("## Methods", 1)[0]
    # The body keeps what it adds and drops its hedge: the title already states code's counts.
    (said,) = [b.body for b in report.blocks if isinstance(b, TextBlock) and b.role == "finding_body"]
    assert said == "The candidate skipped the queue."
    disclosed = [b.text for b in report.blocks if isinstance(b, DisclosureBlock) and "disputes a difference" in b.text]
    assert len(disclosed) == 3 and DISPUTING in disclosed[0]
    assert lint_page(markdown, question=G2_QUESTION) == []


async def test_a_hedge_on_a_reading_no_settled_clause_reads_is_left_to_the_writer() -> None:
    comparison, bundle = await _compare(0.003301)
    aliases = _aliases(comparison, bundle)
    # Accuracy is a mean, not a pass rate the rule reads, so no settled clause decided its difference.
    hedge = "Accuracy fell, but four cases cannot confirm by how much"
    report = await _memo_page(comparison, bundle, [_finding(hedge, "accuracy", aliases)], "Keep production.")
    titles = [b.body for b in report.blocks if isinstance(b, TextBlock) and b.role == "finding_title"]
    assert titles == [hedge]


@pytest.mark.parametrize(
    ("said", "disputes"),
    [
        (DISPUTING, True),
        ("Too few cases to establish the difference", True),
        ("The data could not show a drop", True),
        ("Four cases are not enough to reliably demonstrate the gap", True),
        ("The drop is not statistically significant", True),
        ("The cost difference did not statistically separate", True),
        ("Candidate model frequently underfilled the music queue", False),
        ("The judge cannot see the tool calls", False),
        ("The trials show the candidate dropping tracks", False),
        ("Four cases are enough to apply the rule", False),
    ],
)
def test_a_settled_difference_is_disputed_only_in_a_narrow_set_of_words(said: str, disputes: bool) -> None:
    style = MemoStyle(bundle=AnalysisContextBundle.model_construct(scoreboard=[], cell_measures=[]), names={})
    assert (style.sentence(said, settled="Code's counts.") == "Code's counts.") is disputes
    assert style.hedged == ([said] if disputes else [])
    # Without a settled clause on the reading, nothing is replaced.
    assert MemoStyle(bundle=style.bundle, names={}).sentence(said) == said
    if disputes:
        (line,) = decision_disclosures(style)
        assert said in line and "code's sentence on that reading replaced it" in line


def test_a_keep_settles_every_rate_observed_past_the_margin_and_no_other() -> None:
    rule = parse_decision_rule(G2_QUESTION)
    assert rule is not None
    verdict = evaluate_rule(
        rule,
        RuleReadings(
            rates=(RateReading("the pass rate", 10, 12, 2, 12),), control_cost=0.003092, candidate_cost=0.003301
        ),
    )
    check = 'goal_state:call_count("music.queue_track") >= 2'
    # A check the rule did not read, observed 42 points down: the clause it breaks has settled it.
    assert verdict.settles(RateReading("check", 11, 12, 6, 12, measure=check))
    # Within the margin, or past it only if every lost trial went the candidate's way, it is not settled.
    assert not verdict.settles(RateReading("check", 11, 12, 10, 12, measure=check))
    assert verdict.settles(RateReading("check", 11, 12, 9, 12, measure=check))
    assert not verdict.settles(RateReading("check", 11, 12, 9, 12, candidate_remaining=6, measure=check))
    assert verdict.settles_cost()
    unsettled = evaluate_rule(
        rule, RuleReadings(rates=(RateReading("the pass rate", 10, 12, 10, 12),), control_cases=4, candidate_cases=4)
    )
    assert unsettled.outcome == "unsettled"
    assert not unsettled.settles(RateReading("check", 11, 12, 6, 12, measure=check))
    assert not unsettled.settles_cost()


# =============================================================================
# 3. A label never claims what its own row contradicts
# =============================================================================


def _label_lines(report: Report) -> list[str]:
    return [
        block.text
        for block in report.blocks
        if isinstance(block, DisclosureBlock) and block.section == "contrasts" and block.text.startswith("The label")
    ]


async def test_a_cheaper_label_on_the_dearer_arm_gets_one_line_in_the_contrasts() -> None:
    _, bundle = await _compare(0.003301)
    report = build_code_only_report(bundle, assembled_at="2026-10-14T19:23:41+00:00", campaign_name=TITLE)
    line = "The label 'cheaper candidate' says cheaper; it cost more per reply than production."
    assert _label_lines(report) == [line]
    assert lint_prose(line) == []
    markdown = report_markdown(report)
    first = markdown.split("\n## ", 1)[0]
    contrasts = markdown.split("\n## Contrasts\n", 1)[1].split("\n## ", 1)[0]
    # A step down from the first screen, beside the cost contrast it bears on.
    assert line not in first and line in contrasts
    assert lint_page(markdown, question=G2_QUESTION) == []


async def test_a_label_its_row_agrees_with_gets_no_line() -> None:
    _, bundle = await _compare(0.0029)
    report = build_code_only_report(bundle, assembled_at="2026-10-14T19:23:41+00:00", campaign_name=TITLE)
    assert _label_lines(report) == []


async def test_a_dearer_word_on_the_cheaper_arm_is_contradicted_too() -> None:
    _, bundle = await _compare(0.0029, candidate="deepseek-v4-flash (pricier candidate)")
    report = build_code_only_report(bundle, assembled_at="2026-10-14T19:23:41+00:00", campaign_name=TITLE)
    assert _label_lines(report) == [
        "The label 'pricier candidate' says pricier; it cost less per reply than production."
    ]


def _with_turn_times(
    bundle: AnalysisContextBundle, times: dict[str, float], contended: Sequence[str] = ()
) -> AnalysisContextBundle:
    """``bundle`` with each arm's turn time read (in ms), the arms in ``contended`` read under concurrency."""
    cells = []
    for cell in bundle.cell_measures:
        template = next(m for m in cell.measures.measures if m.name == "cost_usd")
        timed = template.model_copy(update={"name": "total_ms", "mean": times[cell.variant_key]})
        measures = cell.measures.model_copy(update={"measures": [*cell.measures.measures, timed]})
        cells.append(cell.model_copy(update={"measures": measures}))
    coordinates = [
        CellCoordinate(variant_key=cell.variant_key, apparatus_class_id=cell.apparatus_class_id)
        for cell in bundle.cell_measures
        if cell.variant_key in contended
    ]
    turn_time = bundle.measure_catalog["cost_usd"].model_copy(
        update={"name": "total_ms", "reader_name": "Turn time", "unit": "ms", "merit_axis": "latency"}
    )
    catalog = {**bundle.measure_catalog, "total_ms": turn_time}
    return bundle.model_copy(
        update={"cell_measures": cells, "latency_contended_cells": coordinates, "measure_catalog": catalog}
    )


async def test_a_faster_label_is_read_against_turn_time_only_when_read_serially() -> None:
    comparison, bundle = await _compare(0.0029, candidate="deepseek-v4-flash (faster candidate)")
    production = comparison.arm_variants[PRODUCTION]
    candidate = comparison.arm_variants["deepseek-v4-flash (faster candidate)"]
    slower = _with_turn_times(bundle, {production: 4000.0, candidate: 6000.0})
    report = build_code_only_report(slower, assembled_at="2026-10-14T19:23:41+00:00", campaign_name=TITLE)
    assert _label_lines(report) == [
        "The label 'faster candidate' says faster; its replies took longer than production's."
    ]
    # The page compares no latency read under concurrency, so neither does the label line.
    contended = _with_turn_times(bundle, {production: 4000.0, candidate: 6000.0}, contended=[candidate])
    report = build_code_only_report(contended, assembled_at="2026-10-14T19:23:41+00:00", campaign_name=TITLE)
    assert _label_lines(report) == []
    # A faster arm that was faster gets none.
    faster = _with_turn_times(bundle, {production: 6000.0, candidate: 4000.0})
    report = build_code_only_report(faster, assembled_at="2026-10-14T19:23:41+00:00", campaign_name=TITLE)
    assert _label_lines(report) == []


def test_the_label_words_are_the_cost_and_speed_comparatives_only() -> None:
    assert {word for word, _, _ in LABEL_CLAIMS} == {
        "cheaper",
        "cheap",
        "lower-cost",
        "costlier",
        "pricier",
        "more expensive",
        "faster",
        "quicker",
        "slower",
    }
