"""A campaign's page: code's readings across its aspects on the first screen, then one fold per aspect.

The fixture (:mod:`~packages.evals.tests.campaign_page_support`) is two real runs read as one campaign of two
aspects: billing regresses past its floor, tone improves. The broken floor is first of all; the writer's summary
across the aspects and its recommendation, when there is a memo, come next, then code's table, then the person's
call. Each aspect's own page sits in its fold. These are the page's order, its folds, its JSON and its invariants.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser

import pytest

from threetears.evals.analysis import (
    AnalysisContextBundle,
    CampaignReport,
    build_code_only_report,
    build_campaign_report,
    campaign_readings,
    campaign_report_html,
    campaign_report_markdown,
    report_html,
    serialize_report,
)
from threetears.evals.analysis.report.model import (
    CAMPAIGN_JSON_SHAPE,
    CampaignAspectFold,
    DisclosureBlock,
    TableBlock,
)
from threetears.evals.analysis.report.words import FIRST_SCREEN_BANNED
from packages.evals.tests.style_support import lint_page
from threetears.evals.kernel.errors import RefusedInput

from packages.evals.tests.campaign_support import (
    ASSEMBLED_AT,
    CANDIDATE_LABEL,
    CONTROL_LABEL,
    QUESTION,
    arms_of,
)

from packages.evals.tests.campaign_page_support import (
    aspect_fixture,
    campaign_fixture,
    first_screen_markdown,
    kind_fixture,
    page_of,
    two_aspect_memo,
)


async def test_a_campaign_of_one_aspect_has_one_row_and_its_fold_holds_the_aspect_s_page() -> None:
    page = await page_of("billing")
    table = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "aspects")
    assert len(table.rows) == 1
    assert "The interval is read at 95%, as a floor's is." in first_screen_markdown(page)
    (fold,) = page.aspects
    html = campaign_report_html(page)
    own = report_html(fold.report)
    # Every section the aspect's own page holds is in the fold, under the fold's prefix.
    sections = re.findall(r'<section id="([a-z_]+)">', own)
    assert sections and all(
        f'<section id="a-billing-{section}">' in html for section in sections if section != "glossary"
    )
    assert html.count("<h1>") == 1


class _Tags(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))


async def test_the_page_keeps_the_embed_s_invariants() -> None:
    page = await page_of("billing", "tone", memo=two_aspect_memo())
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
    page = await page_of("billing", "tone", memo=two_aspect_memo())
    markdown = campaign_report_markdown(page)
    assert lint_page(markdown, question=QUESTION, names=[CONTROL_LABEL, f"▸ {CONTROL_LABEL}", CANDIDATE_LABEL]) == []
    first = first_screen_markdown(page)
    assert [
        word for word in FIRST_SCREEN_BANNED if re.search(rf"(?<![\w-]){re.escape(word)}s?(?![\w-])", first, re.I)
    ] == []
    # Code's own lines name the arms by their labels, never by their roles.
    lines = [block.text for block in page.blocks if isinstance(block, DisclosureBlock) and block.source != "basis"]
    assert not [line for line in lines if re.search(r"\b(candidate|control)\b", line)]


async def test_json_is_the_readings_and_each_aspect_s_bundle_by_its_fingerprint() -> None:
    page = await page_of("billing", "tone")
    document = json.loads(serialize_report(page, "json"))
    assert (document["kind"], document["shape"]) == ("campaign", CAMPAIGN_JSON_SHAPE)
    assert document["recommendation"] is None
    assert document["readings"]["aspects"] == ["billing", "tone"]
    assert [aspect["bundle_fingerprint"] for aspect in document["aspects"]] == [
        fold.report.source.bundle_fingerprint for fold in page.aspects
    ]
    assert serialize_report(page, "markdown") == campaign_report_markdown(page)
    assert serialize_report(page, "html") == campaign_report_html(page)


async def test_with_a_memo_the_json_carries_the_recommendation_its_arm_named_by_label_and_by_key() -> None:
    page = await page_of("billing", "tone", memo=two_aspect_memo())
    document = json.loads(serialize_report(page, "json"))
    control = page.readings.control
    assert control is not None and page.recommended_variant_key == control
    assert document["recommendation"] == {"arm": CONTROL_LABEL, "variant_key": control, "confidence": "high"}


async def test_a_page_whose_memo_recommends_an_arm_is_refused_without_that_arm_s_key() -> None:
    campaign, measured = await campaign_fixture("billing", "tone")
    bundles = {one.bundle.aspect: one.bundle for one in measured}
    read = campaign_readings(campaign, bundles, arms=arms_of(measured[0]))
    reports = {
        label: build_code_only_report(
            evidence, assembled_at=ASSEMBLED_AT, campaign_name="support", arm_labels=read.arm_labels()
        )
        for label, evidence in bundles.items()
    }
    with pytest.raises(RefusedInput, match="no variant key for that arm") as refused:
        build_campaign_report(campaign, read, reports, two_aspect_memo())
    assert "EvalAnalysis.recommended_variant_key" in refused.value.remedy
    # A key is the recommendation's: a page with none holds none.
    page = build_campaign_report(campaign, read, reports, None, recommended_variant_key="a" * 64)
    assert (page.recommendation, page.recommended_variant_key) == (None, None)


async def test_a_fold_names_the_arms_as_the_campaign_s_table_does() -> None:
    page = await page_of("billing", "tone")
    markdown = campaign_report_markdown(page)
    billing = markdown[markdown.index("## billing") : markdown.index("## tone")]
    # The run named its arm `candidate`; the campaign calls it v2, and so does every table in the fold.
    assert f"| {CANDIDATE_LABEL} |" in billing and "| candidate |" not in billing
    assert f"| ▸ {CONTROL_LABEL} |" in billing


async def test_a_fold_leaves_out_the_aspect_s_own_verdict_line() -> None:
    page = await page_of("billing", "tone")
    for fold in page.aspects:
        verdict = next(
            block for block in fold.report.blocks if isinstance(block, DisclosureBlock) and block.source == "verdict"
        )
        assert verdict not in fold.head
        assert verdict.text not in campaign_report_markdown(page)
        assert 'class="verdict"' not in campaign_report_html(page)


async def test_a_code_only_page_refuses_a_fold_that_renders_an_analysis() -> None:
    page = await page_of("billing")
    (fold,) = page.aspects
    analysed = CampaignAspectFold(
        aspect=fold.aspect,
        shows=fold.shows,
        report=fold.report.model_copy(
            update={
                "basis": "analysis",
                "source": fold.report.source.model_copy(update={"analysis_id": "analysis-1", "generator_model": "m"}),
            }
        ),
    )
    with pytest.raises(ValueError, match="renders an analysis"):
        CampaignReport(
            basis="code_only",
            title=page.title,
            campaign_id=page.campaign_id,
            scope_id=page.scope_id,
            finding_count=0,
            blocks=page.blocks,
            aspects=[analysed],
            readings=page.readings,
        )


async def test_a_fold_s_markdown_heading_is_inert() -> None:
    def relabel(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
        return bundle.model_copy(update={"aspect": "# billing"})

    billing = await aspect_fixture("billing")
    (aspect,) = billing.campaign.aspects
    page = await page_of(
        "billing",
        bundle=relabel,
        aspects=[aspect.model_copy(update={"label": "# billing"})],
        floors=[],
    )
    assert "\n## \\# billing\n" in campaign_report_markdown(page)


async def test_a_fold_of_a_choice_names_the_arms_by_the_campaign_s_labels() -> None:
    tone = await kind_fixture("tone")
    arms = [arm.model_copy(update={"label": arm.label.upper()}) for arm in tone.arms()]
    read = campaign_readings(tone.campaign, {"tone": tone.bundle}, arms=arms)
    assert read.arm_labels() == {tone.keys["alpha"]: "ALPHA", tone.keys["beta"]: "BETA"}
    report = build_code_only_report(
        tone.bundle, assembled_at=ASSEMBLED_AT, campaign_name="support", arm_labels=read.arm_labels()
    )
    page = build_campaign_report(tone.campaign, read, {"tone": report}, None)
    fold = campaign_report_markdown(page).split("\n## tone", 1)[1]
    assert "| ALPHA |" in fold and "| alpha |" not in fold
