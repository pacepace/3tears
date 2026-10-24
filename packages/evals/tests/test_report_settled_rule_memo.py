"""A memo page over floors read either way: the memo's own answer, printed as written.

A floor warns and never decides, so the memo's answer to the question is printed as written, under the page's warning
line for each floor broken. A finding disputing a floor's reading is the writer's prompt to prevent, never the page's to
rewrite (owner, 2026-10-24). The G2 numbers are replayed through the quick path, with the G2 floors declared before the run, as the
presentation tests replay them.

Since #845 a rate floor is read either way only on an interval over enough cases, so G2's four cases leave its rate
floor unclear. Where a test needs the rate settled, the same arms run twenty cases, the candidate failing half of them
outright: a drop whose whole interval is past the margin.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
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
from threetears.evals.analysis.references import cell_aliases
from threetears.evals.analysis.report import DisclosureBlock, TextBlock
from threetears.evals.analysis.style import lint_prose
from packages.evals.tests.style_support import lint_page
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.campaign_bundle_support import memo as authored_memo
from packages.evals.tests.aspect_support import redeclared
from packages.evals.tests.g2_support import CANDIDATE, CASES, G2_FLOORS, G2_QUESTION, PRODUCTION, answering, declaring
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient

#: The G2 campaign's title.
TITLE = "G2: can the cheaper model replace production?"


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
    memo = authored_memo(bundle, findings, means=answer)
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
    assert "Keep production." in markdown
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
# 2. A finding naming no reading is noted with what was not tested
# =============================================================================


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
