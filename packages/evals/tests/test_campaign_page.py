"""A campaign's page: code's readings across its aspects on the first screen, then one fold per aspect.

The fixture is two real runs read as one campaign of two aspects: on billing the candidate regresses (it passes six of
twenty cases to production's eighteen) and breaks the 10-point floor declared there; on tone it improves (nineteen to
six). Both readings are on the first screen, the broken floor first of all; the writer's recommendation, when there is
a memo, comes next, then code's table, then the person's call. Each aspect's own page sits in its fold.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser

import pytest

from threetears.evals.analysis import (
    CampaignReport,
    build_campaign_report,
    campaign_readings,
    campaign_report_html,
    campaign_report_markdown,
    report_html,
    serialize_report,
)
from threetears.evals.analysis.report.campaign_page import NO_CALL, NO_RECOMMENDATION
from threetears.evals.analysis.report.model import DisclosureBlock, TableBlock, TextBlock
from threetears.evals.analysis.report.words import FIRST_SCREEN_BANNED
from threetears.evals.analysis.style import lint_page
from threetears.evals.kernel.authored import (
    AspectSummary,
    AuthoredCampaignAnalysis,
    Finding,
    NextStep,
    Recommendation,
)
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import CampaignCall, RateDrop

from packages.evals.tests.campaign_support import (
    CANDIDATE_LABEL,
    CONTROL_LABEL,
    QUESTION,
    MeasuredAspect,
    arms_of,
    measured_aspect,
)

_MEASURED: dict[str, MeasuredAspect] = {}


async def _aspect(label: str) -> MeasuredAspect:
    """Each fixture aspect, measured once for the module: billing regresses past its floor, tone improves, and late
    improves on a question asked only after its runs."""
    if label not in _MEASURED:
        if label == "billing":
            _MEASURED[label] = await measured_aspect(
                "billing", production=18, candidate=6, floors=(RateDrop(max_points=10),)
            )
        elif label == "tone":
            _MEASURED[label] = await measured_aspect("tone", production=6, candidate=19)
        else:
            _MEASURED[label] = await measured_aspect(label, production=6, candidate=19, asked_at=None)
    return _MEASURED[label]


async def _campaign(
    *labels: str, margins: dict[str, float | None] | None = None
) -> tuple[EvalCampaign, list[MeasuredAspect]]:
    """One campaign holding the fixture aspects named, billing's campaign carrying every one and every floor."""
    measured = [await _aspect(label) for label in labels]
    aspects = [
        aspect.model_copy(update={"margin": (margins or {}).get(aspect.label, aspect.margin)})
        for one in measured
        for aspect in one.campaign.aspects
    ]
    floors = [floor for one in measured for floor in one.campaign.floors]
    first = measured[0].campaign
    campaign = first.model_copy(update={"aspects": aspects, "floors": floors, "name": "support queue"})
    return campaign, measured


async def _page(
    *labels: str,
    memo: AuthoredCampaignAnalysis | None = None,
    call: CampaignCall | None = None,
    margins: dict[str, float | None] | None = None,
) -> CampaignReport:
    campaign, measured = await _campaign(*labels, margins=margins)
    bundles = {one.bundle.aspect: one.bundle for one in measured}
    readings = campaign_readings(campaign, bundles, arms=arms_of(measured[0]))
    reports = {one.bundle.aspect: one.report for one in measured}
    return build_campaign_report(campaign, readings, reports, memo, call=call)


def _memo() -> AuthoredCampaignAnalysis:
    finding = Finding(
        title="v2 resolves tone and breaks billing.",
        body="It passes far more tone requests and far fewer billing ones.",
        confidence="high",
        axes=[],
        evidence=[],
        trials=[],
        caveats=[],
        invalidates=[],
        durable="",
    )
    return AuthoredCampaignAnalysis(
        findings=[finding],
        recommendation=Recommendation(
            arm=CONTROL_LABEL,
            reasons=["The billing floor is broken, and billing is where refunds are lost."],
            confidence="high",
            rests_on=[0],
        ),
        aspect_summaries=[
            AspectSummary(
                aspect="billing",
                what_changed="v2 fails most billing requests production passes.",
                how_sure="Shown worse past the margin.",
                means="Adopting v2 would lose refunds.",
            ),
            AspectSummary(
                aspect="tone",
                what_changed="v2 answers in the house tone far more often.",
                how_sure="Shown better.",
                means="Worth keeping for a later candidate.",
            ),
        ],
        next=[
            NextStep(
                title="Fix the billing prompt and run again.", why="It is the only loss.", leverage="high", lever=""
            )
        ],
    )


def _first_screen_markdown(page: CampaignReport) -> str:
    """The Markdown above the first section heading: the first screen."""
    return campaign_report_markdown(page).split("\n## ", 1)[0]


def _header_html(page: CampaignReport) -> str:
    html = campaign_report_html(page)
    return html[html.index("<header>") : html.index("</header>")]


async def test_both_aspects_readings_are_on_the_first_screen_and_the_broken_floor_is_first() -> None:
    page = await _page("billing", "tone")
    table = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "aspects")
    assert [(row["aspect"].text, row["arm"].text, row["reading"].text) for row in table.rows] == [
        ("billing", CANDIDATE_LABEL, "shown worse"),
        ("tone", CANDIDATE_LABEL, "shown better"),
    ]
    first = _first_screen_markdown(page)
    assert first.startswith("**Floor broken on billing: v2's pass rate falls past the 10-point margin")
    assert "shown worse" in first and "shown better" in first
    header = _header_html(page)
    assert header.index('role="alert"') < header.index("<h1>") < header.index('data-table="aspects"')
    assert "shown worse" in header and "shown better" in header
    # The broken floor is code's: the readings carry it, and the page warns once.
    assert [status.floor for status in page.readings.broken_floors()] == ["billing/rate_drop/*"]
    assert header.count('role="alert"') == 1


async def test_the_first_screen_runs_in_order_floor_question_recommendation_table_call() -> None:
    page = await _page("billing", "tone", memo=_memo())
    first = _first_screen_markdown(page)
    order = [
        first.index("**Floor broken on billing"),
        first.index(f'"{QUESTION}"'),
        first.index(f"**Recommended:** {CONTROL_LABEL} (high confidence)"),
        first.index("- The billing floor is broken"),
        first.index("**By aspect**"),
        first.index(NO_CALL),
    ]
    assert order == sorted(order)
    assert page.basis == "analysis" and page.finding_count == 1


async def test_with_no_memo_the_page_says_no_recommendation_was_written_and_holds_no_writer_s_words() -> None:
    page = await _page("billing", "tone")
    assert page.basis == "code_only"
    assert NO_RECOMMENDATION in _first_screen_markdown(page)
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
    page = await _page("billing", "tone", memo=_memo())
    markdown = campaign_report_markdown(page)
    billing = markdown[markdown.index("## billing") : markdown.index("## tone")]
    assert billing.index("**What changed:** v2 fails most billing") < billing.index("**Scoreboard**")
    assert "**How sure:** Shown worse past the margin." in billing
    assert "**What it means:** Adopting v2 would lose refunds." in billing
    assert "v2 shown worse." in billing
    assert "## What it means" in markdown and "v2 resolves tone and breaks billing." in markdown


async def test_a_recorded_call_says_who_decided_and_the_broken_floor_it_overrides() -> None:
    billing = await _aspect("billing")
    call = CampaignCall(
        arm=CANDIDATE_LABEL,
        variant_key=billing.candidate,
        reason="Tone matters more this quarter.",
        floors_overridden=["billing/rate_drop/*"],
        called_at="2026-10-18T09:00:00+00:00",
        called_by="brooks",
    )
    page = await _page("billing", "tone", memo=_memo(), call=call)
    first = _first_screen_markdown(page)
    assert "Called: adopt v2, by brooks on 18 Oct 2026." in first
    assert 'Reason: "Tone matters more this quarter."' in first
    assert "The call overrides a broken floor: no pass rate drops by more than 10 points on billing." in first
    assert NO_CALL not in first


async def test_a_call_that_keeps_the_control_overrides_nothing() -> None:
    billing = await _aspect("billing")
    control = billing.campaign.control
    assert control is not None
    call = CampaignCall(arm=CONTROL_LABEL, variant_key=control.variant_key, called_by="brooks")
    first = _first_screen_markdown(await _page("billing", "tone", call=call))
    assert "Called: keep production" in first and "overrides" not in first


async def test_a_campaign_of_one_aspect_has_one_row_and_its_fold_holds_the_aspect_s_page() -> None:
    page = await _page("billing")
    table = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "aspects")
    assert len(table.rows) == 1
    assert "The interval is read at 95%, as a floor's is." in _first_screen_markdown(page)
    (fold,) = page.aspects
    html = campaign_report_html(page)
    own = report_html(fold.report)
    # Every section the aspect's own page holds is in the fold, under the fold's prefix.
    sections = re.findall(r'<section id="([a-z_]+)">', own)
    assert sections and all(
        f'<section id="a-billing-{section}">' in html for section in sections if section != "glossary"
    )
    assert html.count("<h1>") == 1


async def test_an_aspect_with_no_margin_is_said_never_shown_worse_and_its_row_is_to_settle() -> None:
    page = await _page("billing", "tone", margins={"billing": None})
    first = _first_screen_markdown(page)
    assert "No margin was declared on billing, so no arm is ever shown worse there." in first
    settle = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "to_settle")
    (row,) = settle.rows
    assert row["aspect"].text == "billing" and "no margin was declared" in row["why"].text


async def test_an_aspect_asked_after_its_runs_reads_as_a_lead() -> None:
    page = await _page("billing", "late")
    first = _first_screen_markdown(page)
    assert "The question was asked after the first run on late, so its readings are leads." in first
    assert "shown better, a lead" in first
    (late,) = page.readings.on_aspect("late")
    assert not late.confirmatory


class _Tags(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))


async def test_the_page_keeps_the_embed_s_invariants() -> None:
    page = await _page("billing", "tone", memo=_memo())
    html = campaign_report_html(page)
    parser = _Tags()
    parser.feed(html)
    tags = parser.tags
    assert not [tag for tag, _ in tags if tag in {"script", "iframe", "object", "embed", "link", "base", "img"}]
    assert not [name for _, attrs in tags for name in attrs if name.startswith("on")]
    ids = [attrs["id"] for _, attrs in tags if attrs.get("id")]
    assert len(ids) == len(set(ids)), "an id is repeated"
    hrefs = [attrs["href"] for _, attrs in tags if "href" in attrs]
    assert all(href is not None and href.startswith("#") for href in hrefs)
    assert {href[1:] for href in hrefs if href} <= set(ids), "a link lands nowhere"
    assert "<header>" in html and '<section id="aspect-billing">' in html and '<section id="glossary">' in html
    assert ("article", {"data-basis": "analysis", "data-page": "campaign"}) in tags


async def test_the_page_passes_the_style_rules_and_its_first_screen_speaks_a_reader_s_words() -> None:
    page = await _page("billing", "tone", memo=_memo())
    markdown = campaign_report_markdown(page)
    assert lint_page(markdown, question=QUESTION, names=[CONTROL_LABEL, f"▸ {CONTROL_LABEL}", CANDIDATE_LABEL]) == []
    first = _first_screen_markdown(page)
    assert [
        word for word in FIRST_SCREEN_BANNED if re.search(rf"(?<![\w-]){re.escape(word)}s?(?![\w-])", first, re.I)
    ] == []
    # Code's own lines name the arms by their labels, never by their roles.
    lines = [block.text for block in page.blocks if isinstance(block, DisclosureBlock) and block.source != "basis"]
    assert not [line for line in lines if re.search(r"\b(candidate|control)\b", line)]


async def test_json_is_the_readings_and_each_aspect_s_bundle_by_its_fingerprint() -> None:
    page = await _page("billing", "tone")
    document = json.loads(serialize_report(page, "json"))
    assert document["readings"]["aspects"] == ["billing", "tone"]
    assert [aspect["bundle_fingerprint"] for aspect in document["aspects"]] == [
        fold.report.source.bundle_fingerprint for fold in page.aspects
    ]
    assert serialize_report(page, "markdown") == campaign_report_markdown(page)
    assert serialize_report(page, "html") == campaign_report_html(page)
