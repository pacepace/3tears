"""A memo page where the declared decision rule settled: no answer line repeating it, no finding disputing it, and no
arm label claiming what its own row contradicts.

The owner's read of the G2 report (`.29`): "**Answered.** See the decision rule above." repeated what the verdict
line and the decision table had said; a finding headlined "four cases cannot statistically confirm the drop" sat under
a ``keep`` the rule had settled on that drop, and read as disputing it; and an arm labelled "cheaper candidate" cost
more per reply, so the verdict and the tables read as a contradiction. The G2 numbers are replayed through the quick
path, with the G2 decision rule declared before the run, as the presentation tests replay them.

Since #845 a rate clause settles only on an interval over enough cases, so G2's four cases leave its rate clause
unsettled. Where a test needs the rate settled, the same arms run twenty cases, the candidate failing half of them
outright: a drop whose whole interval is past the margin.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

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
from threetears.evals.analysis.references import cell_aliases
from threetears.evals.analysis.report import DisclosureBlock, TextBlock
from threetears.evals.analysis.report.decision import RateReading, RuleReadings, evaluate_rule
from threetears.evals.analysis.report.memo_style import MemoStyle, decision_disclosures
from threetears.evals.analysis.bundle.schema import CellCoordinate
from threetears.evals.analysis.style import LABEL_CLAIMS, lint_page, lint_prose
from packages.evals.tests.g2_support import CANDIDATE, CASES, G2_QUESTION, G2_RULE, PRODUCTION, answering, declaring
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient

#: The G2 campaign's title.
TITLE = "G2: can the cheaper model replace production?"

#: The G2 finding headline the owner read as disputing the settled keep.
DISPUTING = "Candidate model frequently underfilled the queue, but four cases cannot statistically confirm the drop"


async def _compare(
    candidate_cost: float, *, candidate: str = CANDIDATE, shown: bool = False
) -> tuple[Result, AnalysisContextBundle]:
    """G2's pass counts (production 10 of 12, the candidate 2 of 12), the candidate at ``candidate_cost`` per reply,
    with the G2 decision rule declared before the run.

    ``shown`` runs twenty cases instead, production passing every trial and the candidate the trials of the first ten
    cases only (30 of 60): a drop shown past the margin on its whole interval, which settles the rate clause keep.
    """
    if shown:
        cases = [{"text": f"queue request {index}", "label": "queued"} for index in range(20)]
        arms = {PRODUCTION: answering(60, 0.003092), candidate: _first_cases(10, candidate_cost)}
    else:
        cases = CASES
        arms = {PRODUCTION: answering(10, 0.003092), candidate: answering(2, candidate_cost)}
    comparison = await Eval(cases, arms, ["label"], k=3, name="Kairo queue discipline").run(store=":memory:")
    if not shown:
        return comparison, declaring(comparison.report.bundle, G2_QUESTION)
    # The question is on the campaign before its bundle is assembled, so its family carries the intervals a rate
    # clause reads.
    storage = comparison.host.storage
    campaign = off_loop(get_campaign, storage, comparison.campaign_id, comparison.scope_id)
    assert campaign.declared_design is not None
    asked = declaring(comparison.report.bundle, G2_QUESTION)
    assert asked.declared_design is not None
    campaign = campaign.model_copy(update={"declared_design": asked.declared_design})
    bundle = off_loop(assemble_context_bundle, campaign, storage=storage, profile=comparison.host.profile)
    return comparison, declaring(bundle, G2_QUESTION)


def _first_cases(right: int, cost: float) -> Any:
    """A runner answering right on every trial of the first ``right`` cases and wrong on the rest, at ``cost``."""

    async def run(case: Mapping[str, Any]) -> Answer:
        passes = int(case["text"].rsplit(" ", 1)[1]) < right
        return Answer(
            case["label"] if passes else "skipped", model="deepseek", input_tokens=100, output_tokens=10, cost_usd=cost
        )

    return run


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
    comparison, bundle = await _compare(0.003301, shown=True)
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
    # The candidate costs less per reply, which a cost with no spread cannot show; its pass rate settles keep.
    comparison, bundle = await _compare(0.0029, shown=True)
    aliases = _aliases(comparison, bundle)
    report = await _memo_page(
        comparison, bundle, [_finding("The candidate skipped the queue", "match", aliases)], "Keep production."
    )
    (answer,) = _answers(report)
    assert answer.body == "Still open: whether cheaper candidate costs less per reply."
    markdown = report_markdown(report)
    assert "**Partly answered.** Still open: whether cheaper candidate costs less per reply." in markdown
    assert "Keep production." not in markdown, "the memo's own answer is not printed"
    assert lint_page(markdown, question=G2_QUESTION) == []


# =============================================================================
# 2. A finding never disputes a difference a settled clause decided
# =============================================================================


async def test_a_title_disputing_a_settled_drop_is_code_s_counts_and_its_body_hedge_goes() -> None:
    comparison, bundle = await _compare(0.003301, shown=True)
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
    # The two values plainly (the gap is measured and stated), then one sentence of what is withheld and why: the
    # gap's interval, never the gap, and never a bare "needs 5".
    assert titles == [
        "Cheaper candidate passed 30 of 60 on Label matched to production's 60.",
        "Cheaper candidate cost $3.301 per 1,000 replies to production's $3.092.",
    ]
    assert "cannot statistically confirm" not in markdown.split("## Methods", 1)[0]
    # The body keeps what it adds and drops its hedge: the title already states code's counts.
    (said,) = [b.body for b in report.blocks if isinstance(b, TextBlock) and b.role == "finding_body"]
    assert said == "The candidate skipped the queue."
    disclosed = [b.text for b in report.blocks if isinstance(b, DisclosureBlock) and "disputes a difference" in b.text]
    assert len(disclosed) == 3 and DISPUTING in disclosed[0]
    assert lint_page(markdown, question=G2_QUESTION) == []


async def test_a_hedge_on_a_drop_past_the_margin_on_its_point_estimate_alone_stands() -> None:
    """G2's own four cases: the drop is past the margin on its point estimate only, so the rate clause is unsettled
    (#845), and a finding saying four cases cannot confirm it says what the rule says. Only the cost is replaced."""
    comparison, bundle = await _compare(0.003301)
    aliases = _aliases(comparison, bundle)
    cost = "Cheaper candidate cost more per reply, but four cases cannot establish the difference"
    report = await _memo_page(
        comparison,
        bundle,
        [_finding(DISPUTING, "match", aliases), _finding(cost, "production_replicating_cost", aliases)],
        "Keep production.",
    )
    titles = [b.body for b in report.blocks if isinstance(b, TextBlock) and b.role == "finding_title"]
    assert titles == [
        DISPUTING,
        "Cheaper candidate cost $3.301 per 1,000 replies to production's $3.092. Whether that gap holds on more "
        "cases is not estimated below 5 independent cases.",
    ]


async def test_a_finding_naming_no_reading_is_noted_with_what_was_not_tested_and_leaves_the_memo() -> None:
    comparison, bundle = await _compare(0.003301)
    aliases = _aliases(comparison, bundle)
    unread = {
        **_finding("Turn latency comparison was withheld due to concurrent execution.", "match", {}),
        "evidence": [],
    }
    report = await _memo_page(
        comparison, bundle, [_finding("The candidate skipped the queue", "match", aliases), unread], "Keep production."
    )
    markdown = report_markdown(report)
    # Findings are evidence: one citing no reading is not one, so What it means holds the other alone.
    titles = [b.body for b in report.blocks if isinstance(b, TextBlock) and b.role == "finding_title"]
    assert titles == ["The candidate skipped the queue"]
    assert report.finding_count == 2 and not [b for b in report.blocks if b.finding == 1]
    noted = "The analysis also noted: Turn latency comparison was withheld due to concurrent execution."
    (said,) = [
        b for b in report.blocks if isinstance(b, DisclosureBlock) and b.section == "not_tested" and b.source == "memo"
    ]
    assert said.text == noted
    # After code's own Not-tested lines, before the folded disclosures.
    section = [b for b in report.blocks if b.section == "not_tested"]
    assert [b.fold for b in section[: section.index(said)]] == [None] * section.index(said)
    assert all(b.fold is not None for b in section[section.index(said) + 1 :])
    not_tested = markdown.split("\n## Not tested\n", 1)[1].split("\n## ", 1)[0]
    assert noted in not_tested and markdown.count("Turn latency comparison") == 1
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


def test_a_rule_settles_a_rate_it_read_and_showed_either_way_and_no_other() -> None:
    """#845 changed this contract: under a keep, any rate observed past the margin was settled, even one the rule
    never read. Now a rate is settled only where a rate clause read it and showed it break or hold the clause."""
    rule = G2_RULE
    check = 'goal_state:call_count("music.queue_track") >= 2'
    broken = RateReading("check", 60, 60, 30, 60, interval=(-0.81, -0.124), measure=check)
    held = RateReading("held", 60, 60, 60, 60, interval=(-0.05, 0.05), measure="held")
    past = RateReading("past", 60, 60, 48, 60, interval=(-0.4, 0.05), measure="past")
    verdict = evaluate_rule(
        rule,
        RuleReadings(
            rates=(RateReading("the pass rate", 60, 60, 60, 60, interval=(-0.05, 0.05)), broken, held, past),
            control_cost=0.003092,
            candidate_cost=0.003301,
            control_cases=20,
            candidate_cases=20,
        ),
        control="production",
        candidate="cheaper candidate",
    )
    assert verdict.outcome == "keep"
    assert [read.outcome for read in verdict.clauses[0].rates] == ["switch", "keep", "switch", "unsettled"]
    # Shown to break, and shown to hold: both settled.
    assert verdict.settles(broken) and verdict.settles(held)
    # Past the margin on its point estimate alone: 20 points down, and not settled.
    assert not verdict.settles(past)
    # A check the rule did not read, observed 42 points down: no clause settled it.
    assert not verdict.settles(RateReading("other", 11, 12, 6, 12, measure="goal_state:other"))
    assert verdict.settles_cost()
    unsettled = evaluate_rule(
        rule,
        RuleReadings(rates=(RateReading("the pass rate", 10, 12, 10, 12),), control_cases=4, candidate_cases=4),
        control="production",
        candidate="cheaper candidate",
    )
    assert unsettled.outcome == "unsettled"
    assert not unsettled.settles(RateReading("check", 11, 12, 6, 12, measure=check))
    assert not unsettled.settles_cost()


# =============================================================================
# 3. A label never claims what its own row contradicts
# =============================================================================


def _label_lines(report: Report) -> list[str]:
    """The first screen's caveat lines after the case floor's: an arm's label its own row contradicts."""
    return [
        block.text
        for block in report.blocks
        if isinstance(block, DisclosureBlock)
        and block.section == "scoreboard"
        and block.source == "caveat"
        and block.term is None
    ]


async def test_a_cheaper_label_on_the_dearer_arm_gets_one_line_on_the_first_screen() -> None:
    _, bundle = await _compare(0.003301)
    report = build_code_only_report(bundle, assembled_at="2026-10-14T19:23:41+00:00", campaign_name=TITLE)
    line = "'cheaper candidate' is not cheaper: it cost more per reply than production."
    assert _label_lines(report) == [line]
    assert lint_prose(line) == []
    markdown = report_markdown(report)
    first = markdown.split("\n## ", 1)[0]
    # A fact that changes how the verdict reads: on the first screen, directly after the case floor's caveat, and
    # nowhere else.
    floor = [b for b in report.blocks if isinstance(b, DisclosureBlock) and b.term == "case floor"]
    assert floor and first.index(floor[0].text) < first.index(line) and markdown.count(line) == 1
    assert lint_page(markdown, question=G2_QUESTION) == []


async def test_a_label_its_row_agrees_with_gets_no_line() -> None:
    _, bundle = await _compare(0.0029)
    report = build_code_only_report(bundle, assembled_at="2026-10-14T19:23:41+00:00", campaign_name=TITLE)
    assert _label_lines(report) == []


async def test_a_dearer_word_on_the_cheaper_arm_is_contradicted_too() -> None:
    _, bundle = await _compare(0.0029, candidate="deepseek-v4-flash (pricier candidate)")
    report = build_code_only_report(bundle, assembled_at="2026-10-14T19:23:41+00:00", campaign_name=TITLE)
    assert _label_lines(report) == ["'pricier candidate' is not pricier: it cost less per reply than production."]


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
    assert _label_lines(report) == ["'faster candidate' is not faster: its replies took longer than production's."]
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
