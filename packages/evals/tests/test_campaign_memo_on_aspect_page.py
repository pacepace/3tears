"""An aspect's page lays out the writer's summary of it in full, after its summary across the campaign, prints the
writer's whys as written, reads the writer's trial ids as its own bundle's, and leads a finding's evidence with
its aspect where the campaign holds several. The reporter reads the memo as written, its summary first.
"""

from __future__ import annotations

from threetears.evals.analysis import build_report, render_memo_as_written, report_markdown
from threetears.evals.analysis.report.words import EVIDENCE_COLUMNS, evidence_columns
from threetears.evals.kernel.authored import AspectSummary, FloorOverride
from threetears.evals.kernel.campaign import EvalAnalysis
from threetears.evals.kernel.text_block import one_line

from packages.evals.tests.report_support import toy_report


def _with_document(analysis: EvalAnalysis, **changes: object) -> EvalAnalysis:
    return analysis.model_copy(update={"document": analysis.document.model_copy(update=changes)})


async def test_the_writer_s_summary_of_the_aspect_is_laid_out_in_full() -> None:
    _, analysis, _ = await toy_report()
    (aspect,) = analysis.aspects
    summary = AspectSummary(
        aspect=aspect.label,
        what_changed="The candidate answers more requests in the house tone.",
        how_sure="Not shown either way.",
        means="Worth a second look.",
    )
    markdown = report_markdown(build_report(_with_document(analysis, aspect_summaries=[summary])))
    changed = markdown.index("**What changed:** The candidate answers more requests in the house tone.")
    assert changed < markdown.index("**How sure:** Not shown either way.") < markdown.index("Worth a second look.")


async def test_a_why_for_an_overridden_floor_is_printed_as_written() -> None:
    _, analysis, _ = await toy_report()
    over = FloorOverride(floor="toy/rate_drop/*", why="The memo rests on tone. Refunds are rare this quarter.")
    recommended = analysis.document.recommended()
    assert recommended is not None
    recommendation = recommended.model_copy(update={"floors_overridden": [over]})
    markdown = report_markdown(build_report(_with_document(analysis, recommendation=[recommendation])))
    # The page reads none of the writer's words: a phrase the page's own lint bans is the writer's to avoid.
    assert "toy/rate_drop/*: The memo rests on tone. Refunds are rare this quarter." in markdown


async def test_the_writer_s_trial_ids_are_read_as_the_page_s_own() -> None:
    _, analysis, _ = await toy_report()
    (aspect,) = analysis.aspects
    listed = aspect.bundle["trials"]
    assert isinstance(listed, list) and listed
    own = listed[0]["trial_id"]
    finding = analysis.document.findings[0].model_copy(update={"trials": ["T1"]})
    page = build_report(_with_document(analysis, findings=[finding, *analysis.document.findings[1:]]))
    facts = [fact for block in page.blocks for fact in getattr(block, "facts", []) if fact.name == "Trials"]
    assert [fact.value for fact in facts] == [own]


async def test_a_finding_s_evidence_is_led_by_its_aspect_where_the_campaign_holds_several() -> None:
    _, analysis, _ = await toy_report()
    (aspect,) = analysis.aspects
    assert evidence_columns(analysis) == EVIDENCE_COLUMNS
    two = analysis.model_copy(update={"aspects": [aspect, aspect.model_copy(update={"label": "tone"})]})
    assert evidence_columns(two) == (("aspect", "Aspect"), *EVIDENCE_COLUMNS)


async def test_the_writer_s_summary_across_the_campaign_leads_the_memo() -> None:
    _, analysis, _ = await toy_report()
    (aspect,) = analysis.aspects
    summary = AspectSummary(
        aspect=aspect.label,
        what_changed="The candidate answers more requests in the house tone.",
        how_sure="",
        means="",
    )
    across = "The candidate is the better arm on tone, and nothing else moved."
    markdown = report_markdown(build_report(_with_document(analysis, aspect_summaries=[summary], summary=across)))
    memo = markdown[markdown.index("## What it means") :]
    assert memo.index(across) < memo.index("**What changed:** The candidate answers more requests in the house tone.")


async def test_the_reporter_reads_the_summary_first_under_the_question_it_reads_across() -> None:
    """The reporter's rubrics judge the memo as written, so the summary is in it, where the campaign's page puts it."""
    _, analysis, _ = await toy_report()
    across = "The candidate is the better arm on tone, and nothing else moved."
    memo = render_memo_as_written(_with_document(analysis, summary=across))
    question = analysis.design_snapshot.question
    assert question
    assert memo.startswith(f"## Summary\n\nQuestion: {one_line(question)}\n\n{across}\n\n## Recommendation\n")
