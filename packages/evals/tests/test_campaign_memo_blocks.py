"""The writer's and the person's words on a campaign's page: the summary, the recommendation, each aspect's
summary, the findings, and the call, each as written."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from threetears.evals.analysis import (
    CampaignReport,
    build_code_only_report,
    build_campaign_report,
    campaign_readings,
    campaign_report_html,
    campaign_report_markdown,
)
from threetears.evals.analysis.report.campaign_page import NO_CALL, NO_RECOMMENDATION
from threetears.evals.analysis.report.model import (
    DisclosureBlock,
    TextBlock,
)
from threetears.evals.analysis.report.words import NO_RECOMMENDATION_WORDS
from threetears.evals.kernel.authored import (
    AuthoredCampaignAnalysis,
    BehindOn,
)
from threetears.evals.kernel.declaration import CampaignCall, OverriddenBar

from packages.evals.tests.campaign_support import (
    ASSEMBLED_AT,
    CANDIDATE_LABEL,
    CONTROL_LABEL,
    joined,
)

from packages.evals.tests.campaign_page_support import (
    aspect_fixture,
    choice_memo,
    first_screen_markdown,
    header_html,
    kind_fixture,
    kind_page_of,
    opening,
    page_of,
    recommending_nothing,
    two_aspect_memo,
)


async def test_with_no_memo_the_page_says_no_recommendation_was_written_and_holds_no_writer_s_words() -> None:
    page = await page_of("billing", "tone")
    assert page.basis == "code_only"
    assert NO_RECOMMENDATION in first_screen_markdown(page)
    assert not [block for block in page.blocks if isinstance(block, TextBlock)]
    with pytest.raises(ValueError, match="holds no writer's words"):
        CampaignReport(
            basis=page.basis,
            title=page.title,
            campaign_id=page.campaign_id,
            scope_id=page.scope_id,
            finding_count=0,
            blocks=[*page.blocks, TextBlock(section="header", role="reason", body="a reason")],
            aspects=page.aspects,
            readings=page.readings,
        )


async def test_the_writer_s_summary_of_each_aspect_opens_its_fold() -> None:
    page = await page_of("billing", "tone", memo=two_aspect_memo())
    markdown = campaign_report_markdown(page)
    billing = markdown[markdown.index("## billing") : markdown.index("## tone")]
    assert billing.index("**What changed:** v2 fails most billing") < billing.index("**Scoreboard**")
    assert "**How sure:** Shown worse past the margin." in billing
    assert "**What it means:** Adopting v2 would lose refunds." in billing
    assert "v2 shown worse." in billing
    findings = markdown[markdown.index("## Findings") : markdown.index("## billing")]
    assert "v2 resolves tone and breaks billing." in findings and "## What it means" not in markdown


@pytest.mark.parametrize("memo_of", [two_aspect_memo, lambda: recommending_nothing("Nothing is ahead.")])
async def test_the_memo_s_findings_and_next_steps_are_their_own_sections_before_the_folds(
    memo_of: Callable[[], AuthoredCampaignAnalysis],
) -> None:
    page = await page_of("billing", "tone", memo=memo_of())
    by_role = {block.role: block.section for block in page.blocks if isinstance(block, TextBlock)}
    assert (by_role["finding_title"], by_role["finding_body"], by_role["next_step"]) == ("findings", "findings", "next")
    assert by_role["summary"] == "answer"
    trace = [block for block in page.blocks if block.finding is not None and block.fold == "Evidence"]
    assert trace and {block.section for block in trace} == {"findings"}
    markdown = campaign_report_markdown(page)
    order = [markdown.index(mark) for mark in ("## Findings", "**v2 resolves tone", "## Next steps", "**Next:** Fix")]
    assert order == sorted(order) and order[-1] < markdown.index("## billing")
    html = campaign_report_html(page)
    assert html.index('<section id="findings">') < html.index('<section id="next">') < html.index('id="aspect-')
    assert '<a href="#findings">Findings</a>' in html and '<a href="#next">Next steps</a>' in html


async def test_a_code_only_page_has_no_findings_and_no_next_steps() -> None:
    page = await page_of("billing", "tone")
    assert not [block for block in page.blocks if block.section in ("findings", "next")]
    assert "## Findings" not in campaign_report_markdown(page)


async def test_a_recorded_call_says_who_decided_and_the_broken_floor_it_overrides() -> None:
    billing = await aspect_fixture("billing")
    call = CampaignCall(
        arm=CANDIDATE_LABEL,
        variant_key=billing.candidate,
        reason="Tone matters more this quarter.",
        floors_overridden=["billing/rate_drop/*"],
        called_at="2026-10-18T09:00:00+00:00",
        called_by="brooks",
    )
    page = await page_of("billing", "tone", memo=two_aspect_memo(), call=call)
    first = first_screen_markdown(page)
    assert "Called: adopt v2, by brooks on 18 Oct 2026." in first
    assert 'Reason: "Tone matters more this quarter."' in first
    assert "The call overrides a broken floor: no pass rate drops by more than 10 points on billing." in first
    assert NO_CALL not in first


async def test_a_call_that_keeps_the_control_overrides_nothing() -> None:
    billing = await aspect_fixture("billing")
    control = billing.campaign.control
    assert control is not None
    call = CampaignCall(arm=CONTROL_LABEL, variant_key=control.variant_key, called_by="brooks")
    page = await page_of("billing", "tone", call=call)
    called = [block.text for block in page.blocks if isinstance(block, DisclosureBlock) and block.source == "call"]
    # The floor's warning says adopting v2 overrides it; no line of the call says it overrides anything.
    assert "Called: keep production" in first_screen_markdown(page) and not any("overrides" in line for line in called)


async def test_a_choice_with_no_clear_winner_opens_on_its_summary_then_says_so_then_the_pairs() -> None:
    summary = "Neither arm pulls ahead on billing or refunds, so cost is the only thing left to choose on."
    page = await kind_page_of("billing", "refunds", memo=recommending_nothing(summary))
    assert page.readings.kind == "choice" and page.recommendation is None
    assert opening(page) == [("text", "summary"), ("memo", NO_RECOMMENDATION_WORDS["choice"])]
    first = first_screen_markdown(page)
    assert first.index(summary) < first.index("No clear winner")
    # The pairs are code's readings, laid out after the answer wherever the scorecard puts them.
    markdown, html = campaign_report_markdown(page), campaign_report_html(page)
    assert markdown.index("No clear winner") < markdown.index("By aspect, pair by pair")
    header = header_html(page)
    assert header.index("<h1>") < header.index('data-role="summary"')
    assert html.index('data-role="summary"') < html.index('data-table="pairs"')


async def test_the_memo_s_words_are_printed_as_written_in_any_language() -> None:
    """Code checks code, and a model's prose is a model's to check (owner, 2026-10-24): the page reads none of the memo's
    words. A sentence an English word list once dropped (a restated count, a discount by data age, a proposed sample)
    and a sentence in French reach the page as written, and the methods list nothing the memo said."""
    memo = two_aspect_memo()
    summary = "La production reste : la facturation perd des remboursements. Run 200 more billing cases for $40."
    body = "Production passed 54 of 60 billing trials. The runs predate the campaign, so read the tone gain as provisional."
    finding = memo.findings[0].model_copy(update={"title": "Le ton progresse sur v2", "body": body})
    page = await page_of("billing", "tone", memo=memo.model_copy(update={"summary": summary, "findings": [finding]}))
    said = {block.role: block.body for block in page.blocks if isinstance(block, TextBlock)}
    assert (said["summary"], said["finding_title"], said["finding_body"]) == (summary, "Le ton progresse sur v2", body)
    assert not [
        block
        for block in page.blocks
        if isinstance(block, DisclosureBlock) and block.section == "methods" and block.source == "memo"
    ]


async def test_a_choice_s_why_for_an_arm_shown_behind_answers_under_the_recommendation() -> None:
    """The why is the writer's answer to a reading that is not a warning: it belongs to the answer, under the
    recommendation it explains, on the first screen, wherever the pairs it answers are laid out."""
    memo = choice_memo("beta")
    recommended = memo.recommended()
    assert recommended is not None
    behind = BehindOn(aspect="tone", ahead_arm="alpha", why="Tone matters less than cost here.")
    recommendation = recommended.model_copy(update={"behind_on": [behind]})
    page = await kind_page_of("tone", "billing", memo=memo.model_copy(update={"recommendation": [recommendation]}))
    first = first_screen_markdown(page)
    why = first.index("**The memo recommends beta over alpha, ahead on tone:** Tone matters less than cost here.")
    assert first.index("**Recommended:** beta") < why
    (block,) = [b for b in page.blocks if isinstance(b, TextBlock) and b.role == "override"]
    assert block.section == "answer"
    markdown = campaign_report_markdown(page)
    assert markdown.index("**The memo recommends beta over alpha") < markdown.index("By aspect, pair by pair")


async def test_a_recorded_call_names_each_missed_bar_it_overrides_and_each_it_does_not() -> None:
    """Alpha misses the cost bar on tone and on billing; the call overrides tone's and is silent on billing's, and the
    page says both, beside the call."""
    measured = [await kind_fixture(label) for label in ("tone", "billing")]
    campaign = joined(measured)
    bundles = {one.bundle.aspect: one.bundle for one in measured}
    read = campaign_readings(campaign, bundles, arms=measured[0].arms())
    missed = {status.aspect: status for status in read.missed_bars() if status.arm_label == "alpha"}
    assert set(missed) == {"tone", "billing"}
    call = CampaignCall(
        arm="alpha",
        variant_key=missed["tone"].arm,
        bars_overridden=[OverriddenBar(aspect="tone", measure=missed["tone"].measure)],
        called_at="2026-10-18T09:00:00+00:00",
        called_by="brooks",
    )
    reports = {
        label: build_code_only_report(
            evidence, assembled_at=ASSEMBLED_AT, campaign_name="support", arm_labels=read.arm_labels()
        )
        for label, evidence in bundles.items()
    }
    first = first_screen_markdown(build_campaign_report(campaign, read, reports, None, call=call))
    called = first.index("Called: adopt alpha, by brooks on 18 Oct 2026.")
    overrides = first.index(f"The call overrides a missed bar on tone: {missed['tone'].reason}.")
    silent = first.index(f"The call does not name the missed bar on billing as overridden: {missed['billing'].reason}.")
    assert called < overrides < silent
