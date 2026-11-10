"""What a campaign's page says of its report and the person's call: with no report, its lead and no line on what it
lacks, no writer's words, no findings; and the call, once recorded, who made it and what it overrides."""

from __future__ import annotations


import pytest

from threetears.evals.analysis import (
    CampaignReport,
    build_code_only_report,
    build_campaign_report,
    campaign_readings,
    campaign_report_html,
    campaign_report_markdown,
)
from threetears.evals.analysis.report.model import DisclosureBlock
from threetears.evals.kernel.declaration import CampaignCall, OverriddenBar

from packages.evals.tests.style_support import unlinked
from packages.evals.tests.campaign_support import (
    ASSEMBLED_AT,
    CANDIDATE_LABEL,
    CONTROL_LABEL,
    joined,
)

from packages.evals.tests.campaign_page_support import (
    aspect_fixture,
    first_screen_markdown,
    header_html,
    kind_fixture,
    kind_page_of,
    opening,
    page_of,
)


#: The lead on billing and tone: each aspect's decision, what Accuracy was on each arm, what the interval allows, at
#: the whole points its intervals of about 37 points set (number policy, rule 2).
_LEAD = (
    "Accuracy on billing: v2 worse than production. Of 60 trials each, Accuracy was right on 54 for production and "
    "18 for v2. Over 20 paired cases, the interval puts v2 between 17 and 92 points worse. Accuracy on tone: v2 "
    "better than production. Of 60 trials each, Accuracy was right on 18 for production and 57 for v2. Over 20 "
    "paired cases, the interval puts v2 between 21 and 96 points better."
)


async def test_with_no_report_the_page_leads_on_code_s_reading_and_holds_no_writer_s_words() -> None:
    page = await page_of("billing", "tone")
    assert page.basis == "code_only"
    # The page is code's: it says what code read, and nothing of a report it does not hold (H5).
    first = first_screen_markdown(page)
    assert f"**{_LEAD}**" in first
    assert not [block for block in page.blocks if isinstance(block, DisclosureBlock) and block.source == "memo"]
    with pytest.raises(ValueError, match="'text' found using 'kind' does not match any of the expected tags"):
        CampaignReport.model_validate(
            {
                "basis": page.basis,
                "title": page.title,
                "campaign_id": page.campaign_id,
                "scope_id": page.scope_id,
                "blocks": [*page.blocks, {"kind": "text", "section": "header", "role": "reason", "body": "a reason"}],
                "aspects": page.aspects,
                "readings": page.readings,
            }
        )


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
    page = await page_of("billing", "tone", call=call)
    first = first_screen_markdown(page)
    assert "Called: adopt v2, by brooks on 18 Oct 2026." in first
    assert 'Reason: "Tone matters more this quarter."' in first
    assert "The call overrides a broken floor: no pass rate drops by more than 10 points on billing." in first


async def test_a_call_that_keeps_the_control_overrides_nothing() -> None:
    billing = await aspect_fixture("billing")
    control = billing.campaign.control
    assert control is not None
    call = CampaignCall(arm=CONTROL_LABEL, variant_key=control.variant_key, called_by="brooks")
    page = await page_of("billing", "tone", call=call)
    called = [block.text for block in page.blocks if isinstance(block, DisclosureBlock) and block.source == "call"]
    # The floor's warning says adopting v2 overrides it; no line of the call says it overrides anything.
    assert "Called: keep production" in first_screen_markdown(page) and not any("overrides" in line for line in called)


def _unbolded(markup: str) -> str:
    """``markup`` without its bold tags, so a sentence reads as one string whatever of it the page sets in bold."""
    return markup.replace("<strong>", "").replace("</strong>", "")


async def test_a_choice_with_no_report_leads_on_its_pairs_then_lays_the_pairs_out() -> None:
    page = await kind_page_of("billing", "refunds")
    assert page.readings.kind == "choice" and "recommendation" not in page.json_document()
    # The lead is each aspect's pairs; the cost's routing is a Design & provenance row, not a first-screen line.
    lead = (
        "On billing, the evidence cannot tell alpha and beta apart. On refunds, the evidence cannot tell alpha and "
        "beta apart."
    )
    assert opening(page) == [("verdict", lead)]
    # The pairs are code's readings, laid out after the answer wherever the scorecard puts them.
    # The lead's words as printed: the undecided reading it first says is linked inside it, each decision in bold.
    markdown, html = campaign_report_markdown(page), _unbolded(unlinked(campaign_report_html(page)))
    assert markdown.index(lead) < markdown.index("By aspect, pair by pair")
    header = _unbolded(unlinked(header_html(page)))
    assert header.index("<h1>") < header.index(lead)
    assert html.index(lead) < html.index('data-table="pairs"')


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
    first = first_screen_markdown(build_campaign_report(campaign, read, reports, call=call))
    called = first.index("Called: adopt alpha, by brooks on 18 Oct 2026.")
    overrides = first.index(f"The call overrides a missed bar on tone: {missed['tone'].reason}.")
    silent = first.index(f"The call does not name the missed bar on billing as overridden: {missed['billing'].reason}.")
    assert called < overrides < silent
