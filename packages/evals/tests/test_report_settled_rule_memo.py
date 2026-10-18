"""A memo page over floors read either way: the memo's own answer, no finding disputing a floor's reading, and no arm
label claiming what its own row contradicts.

A floor warns and never decides, so the memo's answer to the question is printed as written, under the page's warning
line for each floor broken. The owner's read of the G2 report (`.29`) still holds for a reading a floor settled: a
finding headlined "four cases cannot statistically confirm the drop" under a drop a floor read as broken reads as
disputing it; and an arm labelled "cheaper candidate" cost more per reply, so the verdict and the tables read as a
contradiction. The G2 numbers are replayed through the quick path, with the G2 floors declared before the run, as the
presentation tests replay them.

Since #845 a rate floor is read either way only on an interval over enough cases, so G2's four cases leave its rate
floor unclear. Where a test needs the rate settled, the same arms run twenty cases, the candidate failing half of them
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
from threetears.evals.analysis.report.decision import FloorReadings, RateReading, read_floor, settles, settles_cost
from threetears.evals.analysis.report.memo_style import MemoStyle, decision_disclosures
from threetears.evals.analysis.bundle.schema import CellCoordinate
from threetears.evals.analysis.style import LABEL_CLAIMS, lint_page, lint_prose
from threetears.evals.kernel.declaration import Floor
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.aspect_support import redeclared
from packages.evals.tests.g2_support import CANDIDATE, CASES, G2_FLOORS, G2_QUESTION, PRODUCTION, answering, declaring
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient

#: The G2 campaign's title.
TITLE = "G2: can the cheaper model replace production?"

#: The G2 finding headline the owner read as disputing the drop a floor read as broken.
DISPUTING = "Candidate model frequently underfilled the queue, but four cases cannot statistically confirm the drop"


async def _compare(
    candidate_cost: float, *, candidate: str = CANDIDATE, shown: bool = False
) -> tuple[Result, AnalysisContextBundle]:
    """G2's pass counts (production 10 of 12, the candidate 2 of 12), the candidate at ``candidate_cost`` per reply,
    with the G2 floors declared before the run.

    ``shown`` runs twenty cases instead, production passing every trial and the candidate the trials of the first ten
    cases only (30 of 60): a drop shown past the margin on its whole interval, which breaks the rate floor.
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
    # The question and the floors are on the campaign before its bundle is assembled, so its family carries the per-case
    # values a rate floor reads.
    storage = comparison.host.storage
    campaign = off_loop(get_campaign, storage, comparison.campaign_id, comparison.scope_id)
    campaign = redeclared(
        campaign,
        question=G2_QUESTION,
        asked_at="2026-10-14T00:00:00+00:00",
        design={"merit_axes": []},
        margin=10.0,
        floors=G2_FLOORS,
    )
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
        "questions": [{"question_id": bundle.aspect, "resolution": "answered", "answer": answer, "rests_on": [0]}],
        "next": [],
    }
    analysis, _ = await generate_analysis(
        campaign_bundle(bundle),
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
# 1. The answer line: the memo's own, under the warning a broken floor opens the page with
# =============================================================================


def _warnings(report: Report) -> list[str]:
    return [
        b.text
        for b in report.blocks
        if isinstance(b, DisclosureBlock) and b.section == "header" and b.source == "floor"
    ]


async def test_a_broken_floor_never_replaces_the_memo_s_answer() -> None:
    """Until floors, a settled rule's question got no answer line ("See the decision rule above."). A floor decides
    nothing: the memo's answer is printed as written, and each broken floor is the page's first line."""
    comparison, bundle = await _compare(0.003301, shown=True)
    aliases = _aliases(comparison, bundle)
    report = await _memo_page(
        comparison, bundle, [_finding("The candidate skipped the queue", "match", aliases)], "Keep production."
    )
    (answer,) = _answers(report)
    assert answer.body == "Keep production."
    markdown = report_markdown(report)
    assert "**Answered.** Keep production." in markdown
    assert "See the decision rule above." not in markdown and "Still open" not in markdown
    (warning,) = _warnings(report)
    assert warning.startswith(f"Floor broken on {bundle.aspect}: ")
    assert markdown.startswith(f"**{warning}**\n\n# ")
    assert lint_page(markdown, question=G2_QUESTION) == []


async def test_a_memo_recommending_the_arm_a_floor_warns_against_is_printed_as_written() -> None:
    comparison, bundle = await _compare(0.003301, shown=True)
    aliases = _aliases(comparison, bundle)
    report = await _memo_page(
        comparison, bundle, [_finding("The candidate skipped the queue", "match", aliases)], "Switch to the candidate."
    )
    (answer,) = _answers(report)
    assert answer.body == "Switch to the candidate."
    assert not [b for b in report.blocks if isinstance(b, DisclosureBlock) and "contradicts" in b.text]


async def test_the_broken_rate_floor_is_one_warning_and_the_verdict_names_no_keep() -> None:
    """The candidate costs more on every case by one amount: no interval reads the cost floor, so it is unclear and
    warns of nothing, though the candidate is dearer (until the cost floor read its own interval, it broke)."""
    _, bundle = await _compare(0.003301, shown=True)
    report = build_code_only_report(bundle, assembled_at="2026-10-14T19:23:41+00:00", campaign_name=TITLE)
    (rate,) = _warnings(report)
    assert rate.startswith(
        f"Floor broken on {bundle.aspect}: cheaper candidate's pass rate falls past the 10-point margin on its whole "
        "interval, a drop of "
    ), rate
    assert lint_prose(rate) == []
    (verdict,) = [b.text for b in report.blocks if isinstance(b, DisclosureBlock) and b.source == "verdict"]
    assert not verdict.startswith("Keep")


# =============================================================================
# 2. A finding never disputes a difference a floor read either way decided
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
    # gap's interval, never the gap, and never a bare "needs 5". The cost floor read nothing either way (every case
    # cost alike), so the cost finding is the writer's own, its hedge left to it.
    assert titles == ["Cheaper candidate passed 30 of 60 on Label matched to production's 60.", cost]
    assert "cannot statistically confirm" not in markdown.split("## Methods", 1)[0]
    # The body keeps what it adds and drops its hedge: the title already states code's counts.
    (said,) = [b.body for b in report.blocks if isinstance(b, TextBlock) and b.role == "finding_body"]
    assert said == "The candidate skipped the queue."
    disclosed = [b.text for b in report.blocks if isinstance(b, DisclosureBlock) and "disputes a difference" in b.text]
    assert len(disclosed) == 2 and DISPUTING in disclosed[0]
    assert lint_page(markdown, question=G2_QUESTION) == []


async def test_a_hedge_on_a_drop_past_the_margin_on_its_point_estimate_alone_stands() -> None:
    """G2's own four cases: the drop is past the margin on its point estimate only, so the rate floor is unclear (#845),
    and a finding saying four cases cannot confirm it says what the floor says. The cost floor is unclear on four cases
    too, so its finding stands as well (until the cost floor read its own interval, it was replaced)."""
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
    assert titles == [DISPUTING, cost]


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
    # Without a floor read either way on the reading, nothing is replaced.
    assert MemoStyle(bundle=style.bundle, names={}).sentence(said) == said
    if disputes:
        (line,) = decision_disclosures(style)
        assert said in line and "code's sentence on that reading replaced it" in line


def test_a_floor_settles_a_rate_it_read_and_showed_either_way_and_no_other() -> None:
    """#845 changed this contract: under a keep, any rate observed past the margin was settled, even one the rule
    never read. Now a rate is settled only where a rate floor read it and showed it broken or held."""
    check = 'goal_state:call_count("music.queue_track") >= 2'
    broken = RateReading("check", 60, 60, 30, 60, interval=(-0.81, -0.124), measure=check)
    held = RateReading("held", 60, 60, 60, 60, interval=(-0.05, 0.05), measure="held")
    past = RateReading("past", 60, 60, 48, 60, interval=(-0.4, 0.05), measure="past")
    readings = FloorReadings(
        rates=(RateReading("the pass rate", 60, 60, 60, 60, interval=(-0.05, 0.05)), broken, held, past),
        control_cost=0.003092,
        candidate_cost=0.003301,
        control_cases=20,
        candidate_cases=20,
        cost_interval=(0.0001, 0.0003),
        cost_compared=True,
    )
    on = {"control": "production", "candidate": "cheaper candidate"}
    verdicts = [read_floor(Floor(aspect="queue", clause=clause), readings, **on) for clause in G2_FLOORS]
    assert [verdict.outcome for verdict in verdicts] == ["broken", "broken"]
    assert [read.outcome for read in verdicts[0].rates] == ["held", "broken", "held", "unclear"]
    # Shown broken, and shown held: both settled.
    assert settles(verdicts, broken) and settles(verdicts, held)
    # Past the margin on its point estimate alone: 20 points down, and not settled.
    assert not settles(verdicts, past)
    # A check no floor read, observed 42 points down: no floor settled it.
    assert not settles(verdicts, RateReading("other", 11, 12, 6, 12, measure="goal_state:other"))
    assert settles_cost(verdicts)
    thin = FloorReadings(rates=(RateReading("the pass rate", 10, 12, 10, 12),), control_cases=4, candidate_cases=4)
    unclear = [read_floor(Floor(aspect="queue", clause=clause), thin, **on) for clause in G2_FLOORS]
    assert [verdict.outcome for verdict in unclear] == ["unclear", "unclear"]
    assert not settles(unclear, RateReading("check", 11, 12, 6, 12, measure=check))
    assert not settles_cost(unclear)


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
