"""A page over floors read either way: a broken floor warns, on the first screen, and never decides.

A floor warns and never decides, so each floor broken is one warning line under the page's title, and the verdict names
no keep. The G2 numbers are replayed through the quick path, with the G2 floors declared before the run, as the
presentation tests replay them.

Since #845 a rate floor is read either way only on an interval over enough cases, so G2's four cases leave its rate
floor unclear. Where a test needs the rate settled, the same arms run twenty cases, the candidate failing half of them
outright: a drop whose whole interval is past the margin.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


from threetears.evals import Answer, Eval, Result
from threetears.evals.analysis import (
    AnalysisContextBundle,
    Report,
    assemble_context_bundle,
    build_code_only_report,
    get_campaign,
    report_markdown,
)
from threetears.evals.analysis.report import DisclosureBlock
from threetears.evals.analysis.style import lint_prose
from packages.evals.tests.style_support import lint_page
from packages.evals.tests.aspect_support import redeclared
from packages.evals.tests.g2_support import CANDIDATE, CASES, G2_FLOORS, G2_QUESTION, PRODUCTION, answering, declaring
from packages.evals.tests.off_loop import off_loop

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


# =============================================================================
# A broken floor is a warning on the first screen
# =============================================================================


def _warnings(report: Report) -> list[str]:
    return [
        b.text
        for b in report.blocks
        if isinstance(b, DisclosureBlock) and b.section == "header" and b.source == "floor"
    ]


async def test_a_broken_floor_is_the_page_s_first_warning_line() -> None:
    """Until floors, a settled rule's question got no answer line ("See the decision rule above."). A floor decides
    nothing: each broken floor is a warning line on the page's first screen, bolded under its title."""
    _, bundle = await _compare(0.003301, shown=True)
    report = build_code_only_report(bundle, assembled_at="2026-10-14T19:23:41+00:00", campaign_name=TITLE)
    markdown = report_markdown(report)
    assert "See the decision rule above." not in markdown and "Still open" not in markdown
    (warning,) = _warnings(report)
    assert warning == f"Adopting cheaper candidate overrides the 10-point floor it breaks on {bundle.aspect}."
    assert markdown.startswith("# ") and f"\n**{warning}**\n" in markdown
    assert lint_page(markdown, question=G2_QUESTION) == []


async def test_the_broken_rate_floor_is_one_warning_and_the_verdict_names_no_keep() -> None:
    """The candidate costs more on every case by one amount: no interval reads the cost floor, so it is unclear and
    warns of nothing, though the candidate is dearer (until the cost floor read its own interval, it broke)."""
    _, bundle = await _compare(0.003301, shown=True)
    report = build_code_only_report(bundle, assembled_at="2026-10-14T19:23:41+00:00", campaign_name=TITLE)
    (rate,) = _warnings(report)
    assert rate == f"Adopting cheaper candidate overrides the 10-point floor it breaks on {bundle.aspect}."
    assert lint_prose(rate) == []
    (verdict,) = [b.text for b in report.blocks if isinstance(b, DisclosureBlock) and b.source == "verdict"]
    assert not verdict.startswith("Keep")
