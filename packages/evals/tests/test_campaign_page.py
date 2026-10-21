"""A campaign's page: code's readings across its aspects on the first screen, then one fold per aspect.

The fixture is two real runs read as one campaign of two aspects: on billing the candidate regresses (it passes six of
twenty cases to production's eighteen) and breaks the 10-point floor declared there; on tone it improves (nineteen to
six). Both readings are on the first screen, the broken floor first of all; the writer's recommendation, when there is
a memo, comes next, then code's table, then the person's call. Each aspect's own page sits in its fold.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from html.parser import HTMLParser
from typing import Any

import pytest

from threetears.evals.analysis import (
    AnalysisContextBundle,
    CampaignReadings,
    CampaignReport,
    build_code_only_report,
    build_campaign_report,
    campaign_readings,
    campaign_report_html,
    campaign_report_markdown,
    report_html,
    serialize_report,
)
from threetears.evals.analysis.lenses.frontier import FrontierCostDecision, FrontierCostTie, FrontierVerdict
from threetears.evals.analysis.report.campaign_page import CHOICE_NOTE, NO_CALL, NO_RECOMMENDATION
from threetears.evals.analysis.report.model import (
    CAMPAIGN_JSON_SHAPE,
    CampaignAspectFold,
    DisclosureBlock,
    TableBlock,
    TextBlock,
)
from threetears.evals.analysis.report.words import FIRST_SCREEN_BANNED
from threetears.evals.analysis.style import lint_page
from threetears.evals.kernel.authored import (
    AspectSummary,
    AuthoredCampaignAnalysis,
    BarMissed,
    BehindOn,
    Finding,
    FloorOverride,
    NextStep,
    Recommendation,
)
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import BarOverride, CampaignCall, OverriddenBar, RateDrop
from threetears.evals.kernel.errors import RefusedInput

from packages.evals.tests.campaign_support import (
    ASSEMBLED_AT,
    CANDIDATE_LABEL,
    CONTROL_LABEL,
    QUESTION,
    MeasuredAspect,
    MeasuredKind,
    arms_of,
    joined,
    measured_aspect,
    measured_kind,
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
    *labels: str, margins: dict[str, float | None] | None = None, **changes: Any
) -> tuple[EvalCampaign, list[MeasuredAspect]]:
    """One campaign holding the fixture aspects named, billing's campaign carrying every one and every floor, with
    ``changes`` to its fields."""
    measured = [await _aspect(label) for label in labels]
    aspects = [
        aspect.model_copy(update={"margin": (margins or {}).get(aspect.label, aspect.margin)})
        for one in measured
        for aspect in one.campaign.aspects
    ]
    floors = [floor for one in measured for floor in one.campaign.floors]
    first = measured[0].campaign
    campaign = first.model_copy(update={"aspects": aspects, "floors": floors, "name": "support queue", **changes})
    return campaign, measured


async def _page(
    *labels: str,
    memo: AuthoredCampaignAnalysis | None = None,
    call: CampaignCall | None = None,
    margins: dict[str, float | None] | None = None,
    bundle: Callable[[AnalysisContextBundle], AnalysisContextBundle] = lambda bundle: bundle,
    readings: Callable[[CampaignReadings], CampaignReadings] = lambda readings: readings,
    **changes: Any,
) -> CampaignReport:
    """The page over the fixture aspects named, each aspect's page built as a host builds it, with its arms named by
    the campaign's labels; ``bundle`` and ``readings`` change what is read, ``changes`` the campaign."""
    campaign, measured = await _campaign(*labels, margins=margins, **changes)
    bundles = {changed.aspect: changed for changed in (bundle(one.bundle) for one in measured)}
    read = readings(campaign_readings(campaign, bundles, arms=arms_of(measured[0])))
    reports = {
        label: build_code_only_report(
            evidence, assembled_at=ASSEMBLED_AT, campaign_name="support", arm_labels=read.arm_labels()
        )
        for label, evidence in bundles.items()
    }
    return build_campaign_report(
        campaign, read, reports, memo, call=call, recommended_variant_key=_recorded_key(read, memo)
    )


def _recorded_key(read: CampaignReadings, memo: AuthoredCampaignAnalysis | None) -> str | None:
    """The key an analysis records for the arm its memo recommends: the arm the readings label so."""
    recommended = memo.recommended() if memo is not None else None
    if recommended is None:
        return None
    return {label: key for key, label in read.arm_labels().items()}.get(recommended.arm)


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
        recommendation=[
            Recommendation(
                arm=CONTROL_LABEL,
                reasons=["The billing floor is broken, and billing is where refunds are lost."],
                confidence="high",
                rests_on=[0],
                floors_overridden=[],
                behind_on=[],
                bars_missed=[],
            )
        ],
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


async def test_an_aspect_with_no_margin_is_said_never_shown_worse_and_has_no_row_to_settle_against_one() -> None:
    page = await _page("billing", "tone", margins={"billing": None})
    first = _first_screen_markdown(page)
    assert "No margin was declared on billing, so no arm is ever shown worse there." in first
    (billing,) = page.readings.on_aspect("billing")
    assert billing.reading == "unclear" and "no margin was declared" in billing.reason
    # Its row would be a row of dashes under "To show no worse than the margin": it has none to show it within.
    assert not [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "to_settle"]
    assert "To show no worse than the margin" not in first


async def test_an_unclear_row_with_a_margin_is_still_to_settle_beside_one_without() -> None:
    def tone_unclear(readings: CampaignReadings) -> CampaignReadings:
        rows = [
            row.model_copy(update={"reading": "unclear"}) if row.aspect == "tone" else row for row in readings.readings
        ]
        return readings.model_copy(update={"readings": rows})

    page = await _page("billing", "tone", margins={"billing": 70.0, "tone": None}, readings=tone_unclear)
    assert {row.aspect: (row.reading, row.margin) for row in page.readings.readings} == {
        "billing": ("unclear", 70.0),
        "tone": ("unclear", None),
    }
    settle = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "to_settle")
    assert [row["aspect"].text for row in settle.rows] == ["billing"]


async def test_a_margin_late_on_some_aspects_is_one_line_naming_them() -> None:
    late = "2099-01-01T00:00:00+00:00"

    def tone_late(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
        return _late(late)(bundle) if bundle.aspect == "tone" else bundle

    page = await _page("billing", "tone", bundle=tone_late)
    first = _first_screen_markdown(page)
    assert "The margin on tone was declared after its first result." in first
    assert "The question was worded after the first result on tone; its readings keep the time it was asked." in first
    # Billing's margin and wording came before its first result: nothing is said of it.
    assert first.count("after its first result") == 1 and first.count("after the first result") == 1


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
    assert (document["kind"], document["shape"]) == ("campaign", CAMPAIGN_JSON_SHAPE)
    assert document["recommendation"] is None
    assert document["readings"]["aspects"] == ["billing", "tone"]
    assert [aspect["bundle_fingerprint"] for aspect in document["aspects"]] == [
        fold.report.source.bundle_fingerprint for fold in page.aspects
    ]
    assert serialize_report(page, "markdown") == campaign_report_markdown(page)
    assert serialize_report(page, "html") == campaign_report_html(page)


async def test_with_a_memo_the_json_carries_the_recommendation_its_arm_named_by_label_and_by_key() -> None:
    page = await _page("billing", "tone", memo=_memo())
    document = json.loads(serialize_report(page, "json"))
    control = page.readings.control
    assert control is not None and page.recommended_variant_key == control
    assert document["recommendation"] == {"arm": CONTROL_LABEL, "variant_key": control, "confidence": "high"}


async def test_a_page_whose_memo_recommends_an_arm_is_refused_without_that_arm_s_key() -> None:
    campaign, measured = await _campaign("billing", "tone")
    bundles = {one.bundle.aspect: one.bundle for one in measured}
    read = campaign_readings(campaign, bundles, arms=arms_of(measured[0]))
    reports = {
        label: build_code_only_report(
            evidence, assembled_at=ASSEMBLED_AT, campaign_name="support", arm_labels=read.arm_labels()
        )
        for label, evidence in bundles.items()
    }
    with pytest.raises(RefusedInput, match="no variant key for that arm") as refused:
        build_campaign_report(campaign, read, reports, _memo())
    assert "EvalAnalysis.recommended_variant_key" in refused.value.remedy
    # A key is the recommendation's: a page with none holds none.
    page = build_campaign_report(campaign, read, reports, None, recommended_variant_key="a" * 64)
    assert (page.recommendation, page.recommended_variant_key) == (None, None)


async def test_a_fold_names_the_arms_as_the_campaign_s_table_does() -> None:
    page = await _page("billing", "tone")
    markdown = campaign_report_markdown(page)
    billing = markdown[markdown.index("## billing") : markdown.index("## tone")]
    # The run named its arm `candidate`; the campaign calls it v2, and so does every table in the fold.
    assert f"| {CANDIDATE_LABEL} |" in billing and "| candidate |" not in billing
    assert f"| ▸ {CONTROL_LABEL} |" in billing


async def test_a_fold_leaves_out_the_aspect_s_own_verdict_line() -> None:
    page = await _page("billing", "tone")
    for fold in page.aspects:
        verdict = next(
            block for block in fold.report.blocks if isinstance(block, DisclosureBlock) and block.source == "verdict"
        )
        assert verdict not in fold.head
        assert verdict.text not in campaign_report_markdown(page)
        assert 'class="verdict"' not in campaign_report_html(page)


async def test_a_floor_broken_beside_a_row_not_shown_worse_says_the_two_read_different_intervals() -> None:
    # A 70-point margin on billing: its 60-point drop is not past it, while the 10-point floor is broken.
    page = await _page("billing", "tone", margins={"billing": 70.0})
    (row,) = page.readings.on_aspect("billing")
    assert row.reading == "unclear"
    assert (
        "The floor broken on billing for v2 reads its pass rate at 95% on its own, while its row reads only the pass "
        "rate, corrected across v2's aspects." in _first_screen_markdown(page)
    )


def _late(stamp: str) -> Callable[[AnalysisContextBundle], AnalysisContextBundle]:
    """A bundle whose aspect's question was worded, and its margin declared, at ``stamp``."""

    def change(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
        assert bundle.declared is not None
        declared = bundle.declared.model_copy(update={"question_declared_at": stamp, "margin_declared_at": stamp})
        return bundle.model_copy(update={"declared": declared})

    return change


async def test_what_was_declared_after_an_aspect_s_first_result_is_said_so() -> None:
    billing = await _aspect("billing")
    assert billing.bundle.first_result_at is not None
    (floor,) = billing.campaign.floors
    late = "2099-01-01T00:00:00+00:00"
    page = await _page("billing", "tone", bundle=_late(late), floors=[floor.model_copy(update={"declared_at": late})])
    first = _first_screen_markdown(page)
    # Late on both aspects, each kind of lateness is one line naming every aspect, never a line per aspect.
    assert (
        "The question was worded after the first result on every aspect; their readings keep the time it was asked."
        in first
    )
    assert "The margins on every aspect were declared after their first results." in first
    assert first.count("after the first result") == 1 and first.count("declared after") == 2
    assert (
        "The floor that no pass rate drops by more than 10 points on billing was declared after its first result."
        in first
    )
    # Declared before the first result, nothing is said.
    early = await _page("billing", "tone", bundle=_late("2000-01-01T00:00:00+00:00"))
    assert "after its first result" not in _first_screen_markdown(early)
    assert "worded after" not in _first_screen_markdown(early)


async def test_an_unclear_row_s_cases_show_it_no_worse_than_the_margin_and_not_better() -> None:
    page = await _page("billing", "tone", margins={"billing": 70.0})
    settle = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "to_settle")
    assert settle.title == "To show no worse than the margin"
    note = page.blocks[page.blocks.index(settle) + 1]
    assert isinstance(note, DisclosureBlock)
    assert "would not show the arm better, so its row stays unclear unless its gain clears too" in note.text


async def test_a_code_only_page_refuses_a_fold_that_renders_an_analysis() -> None:
    page = await _page("billing")
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


async def test_with_no_question_every_reading_is_a_lead() -> None:
    first = _first_screen_markdown(await _page("billing", "tone", question=""))
    assert "No question was declared, so every reading is a lead." in first


async def test_many_broken_floors_are_warned_of_one_line_per_arm() -> None:
    def four(readings: CampaignReadings) -> CampaignReadings:
        (status,) = readings.floors
        statuses = [
            status.model_copy(update={"aspect": aspect, "floor": f"{aspect}/rate_drop/*"})
            for aspect in ("billing", "tone", "refunds", "returns")
        ]
        return readings.model_copy(update={"floors": statuses})

    page = await _page("billing", "tone", readings=four)
    alerts = [block for block in page.blocks if isinstance(block, DisclosureBlock) and block.source == "floor"]
    assert [alert.text for alert in alerts] == [
        "Floors broken for v2: its pass rate on billing; its pass rate on tone; its pass rate on refunds; its pass "
        "rate on returns."
    ]


async def test_a_fold_s_markdown_heading_is_inert() -> None:
    def relabel(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
        return bundle.model_copy(update={"aspect": "# billing"})

    billing = await _aspect("billing")
    (aspect,) = billing.campaign.aspects
    page = await _page(
        "billing",
        bundle=relabel,
        aspects=[aspect.model_copy(update={"label": "# billing"})],
        floors=[],
    )
    assert "\n## \\# billing\n" in campaign_report_markdown(page)


# =============================================================================
# A choice and a measurement
# =============================================================================

_KINDS: dict[str, MeasuredKind] = {}

#: No reply may cost more than a tenth of a cent; alpha costs two tenths a reply, so it misses this bar everywhere.
_COST_BAR = BarOverride(measure_id="cost_usd", threshold=0.001, direction="lower_is_better")


async def _kind(label: str) -> MeasuredKind:
    """Each fixture aspect of the other two kinds, measured once: on tone alpha passes nineteen cases to beta's six,
    on billing ten to nine, on refunds ten to ten; solo is alpha alone."""
    if label not in _KINDS:
        shapes: dict[str, tuple[dict[str, int], bool]] = {
            "tone": ({"alpha": 19, "beta": 6}, False),
            "billing": ({"alpha": 10, "beta": 9}, False),
            "refunds": ({"alpha": 10, "beta": 10}, False),
            "solo": ({"alpha": 15}, True),
        }
        arms, control = shapes[label]
        _KINDS[label] = await measured_kind(label, arms, control=control, bars=[_COST_BAR], costs={"beta": 0.0005})
    return _KINDS[label]


async def _kind_page(
    *labels: str,
    memo: AuthoredCampaignAnalysis | None = None,
    bundle: Callable[[AnalysisContextBundle], AnalysisContextBundle] = lambda bundle: bundle,
) -> CampaignReport:
    measured = [await _kind(label) for label in labels]
    campaign = joined(measured)
    bundles = {one.bundle.aspect: bundle(one.bundle) for one in measured}
    read = campaign_readings(campaign, bundles, arms=measured[0].arms())
    reports = {
        label: build_code_only_report(
            evidence, assembled_at=ASSEMBLED_AT, campaign_name="support", arm_labels=read.arm_labels()
        )
        for label, evidence in bundles.items()
    }
    return build_campaign_report(campaign, read, reports, memo, recommended_variant_key=_recorded_key(read, memo))


def _choice_memo(arm: str = "beta") -> AuthoredCampaignAnalysis:
    memo = _memo()
    recommended = memo.recommended()
    assert recommended is not None
    recommendation = recommended.model_copy(update={"arm": arm, "reasons": ["beta costs a quarter as much."]})
    return memo.model_copy(update={"recommendation": [recommendation], "aspect_summaries": []})


async def test_a_choice_shows_an_arm_ahead_on_one_aspect_and_no_separation_on_another() -> None:
    page = await _kind_page("tone", "billing", memo=_choice_memo())
    assert page.readings.kind == "choice"
    table = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "pairs")
    assert [(row["aspect"].text, row["ahead"].text) for row in table.rows] == [
        ("tone", "alpha"),
        ("billing", "no separation"),
    ]
    first = _first_screen_markdown(page)
    assert CHOICE_NOTE in first
    # The pairs, then the writer's recommendation; no control-based table.
    assert first.index("**By aspect, pair by pair**") < first.index("**Recommended:** beta (high confidence)")
    assert "**By aspect**\n" not in first
    markdown = campaign_report_markdown(page)
    assert "alpha ahead of beta." in markdown and "alpha and beta not separated." in markdown


async def test_a_choice_with_nothing_separated_says_so_on_every_pair() -> None:
    page = await _kind_page("billing", "refunds")
    table = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "pairs")
    assert {row["ahead"].text for row in table.rows} == {"no separation"}
    assert all(pair.ahead is None for pair in page.readings.pairs)


async def test_a_measurement_s_missed_bar_is_first_on_the_page_and_no_arm_is_recommended() -> None:
    page = await _kind_page("solo", memo=_choice_memo("alpha"))
    assert page.readings.kind == "measurement"
    first = _first_screen_markdown(page)
    assert first.startswith("**Bar missed on solo: alpha misses the bar on ")
    header = _header_html(page)
    assert header.index('role="alert"') < header.index("<h1>")
    assert "Recommended" not in first and page.recommendation is None
    assert json.loads(serialize_report(page, "json"))["recommendation"] is None
    measures = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "measures")
    assert measures.rows[0]["measure"].text == "pass rate"
    bars = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "bars")
    assert [(row["arm"].text, row["status"].text) for row in bars.rows] == [("alpha", "missed")]
    assert NO_CALL not in first


async def test_a_missed_bar_warns_first_on_a_choice_too() -> None:
    first = _first_screen_markdown(await _kind_page("tone"))
    assert first.startswith("**Bar missed on tone: alpha misses the bar on ")


async def test_the_writer_s_why_for_an_overridden_floor_sits_beside_its_warning() -> None:
    memo = _memo()
    over = FloorOverride(floor="billing/rate_drop/*", why="Billing refunds are rare this quarter.")
    recommended = memo.recommended()
    assert recommended is not None
    recommendation = recommended.model_copy(update={"arm": CANDIDATE_LABEL, "floors_overridden": [over]})
    page = await _page("billing", "tone", memo=memo.model_copy(update={"recommendation": [recommendation]}))
    first = _first_screen_markdown(page)
    warning = first.index("**Floor broken on billing")
    why = first.index("**The memo recommends v2 over it:** Billing refunds are rare this quarter.")
    assert warning < why < first.index("# support queue")
    header = _header_html(page)
    assert header.index('role="alert"') < header.index('data-role="override"') < header.index("<h1>")


async def test_the_writer_s_words_go_through_the_style_pass_and_what_it_drops_is_counted_in_methods() -> None:
    memo = _memo()
    # 54 of 60 is billing's measured count for production: a sentence restating it is dropped, the tables say it.
    finding = memo.findings[0].model_copy(update={"body": "Production passed 54 of 60 billing trials. Tone gains."})
    page = await _page("billing", "tone", memo=memo.model_copy(update={"findings": [finding]}))
    body = next(block for block in page.blocks if isinstance(block, TextBlock) and block.role == "finding_body")
    assert body.body == "Tone gains."
    methods = [block.text for block in page.blocks if block.section == "methods"]
    assert any("restated a measured number" in text for text in methods)
    assert "## Methods" in campaign_report_markdown(page)


_UNREAD = (
    "The campaign names no control, so a margin of 5 points and floor tone/rate_drop/* declared on tone are read "
    "against nothing and not read; its arms are held to its bars."
)


def _unread(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
    assert bundle.declared is not None
    return bundle.model_copy(
        update={"declared": bundle.declared.model_copy(update={"unread_without_control": _UNREAD})}
    )


async def test_a_margin_and_floors_stored_without_a_control_are_said_unread_on_both_pages() -> None:
    page = await _kind_page("tone", bundle=_unread)
    assert _UNREAD in _first_screen_markdown(page)
    (fold,) = page.aspects
    decision = [block for block in fold.report.blocks if block.section == "decision"]
    assert [block.text for block in decision if isinstance(block, DisclosureBlock)] == [_UNREAD]


async def test_a_choice_s_why_for_an_arm_shown_behind_sits_beside_the_pairs() -> None:
    memo = _choice_memo("beta")
    recommended = memo.recommended()
    assert recommended is not None
    behind = BehindOn(aspect="tone", ahead_arm="alpha", why="Tone matters less than cost here.")
    recommendation = recommended.model_copy(update={"behind_on": [behind]})
    page = await _kind_page("tone", "billing", memo=memo.model_copy(update={"recommendation": [recommendation]}))
    first = _first_screen_markdown(page)
    pairs = first.index("**By aspect, pair by pair**")
    why = first.index("**The memo recommends beta over alpha, ahead on tone:** Tone matters less than cost here.")
    assert pairs < why < first.index("**Recommended:** beta")


async def test_a_why_for_a_missed_bar_sits_beside_its_warning() -> None:
    memo = _choice_memo("alpha")
    recommended = memo.recommended()
    assert recommended is not None
    missed = BarMissed(aspect="tone", measure="cost_usd", why="A cent a reply is affordable at this volume.")
    recommendation = recommended.model_copy(update={"bars_missed": [missed]})
    page = await _kind_page("tone", memo=memo.model_copy(update={"recommendation": [recommendation]}))
    first = _first_screen_markdown(page)
    warning = first.index("**Bar missed on tone: alpha")
    why = first.index("**The memo recommends alpha over it:** A cent a reply is affordable at this volume.")
    assert warning < why < first.index("# support queue")


async def test_a_recorded_call_names_each_missed_bar_it_overrides_and_each_it_does_not() -> None:
    """Alpha misses the cost bar on tone and on billing; the call overrides tone's and is silent on billing's, and the
    page says both, beside the call."""
    measured = [await _kind(label) for label in ("tone", "billing")]
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
    first = _first_screen_markdown(build_campaign_report(campaign, read, reports, None, call=call))
    called = first.index("Called: adopt alpha, by brooks on 18 Oct 2026.")
    overrides = first.index(f"The call overrides a missed bar on tone: {missed['tone'].reason}.")
    silent = first.index(f"The call does not name the missed bar on billing as overridden: {missed['billing'].reason}.")
    assert called < overrides < silent


async def test_a_why_for_an_overridden_floor_answers_only_the_arm_the_memo_recommends() -> None:
    def both(readings: CampaignReadings) -> CampaignReadings:
        (status,) = readings.floors
        other = status.model_copy(update={"arm": "c" * 64, "arm_label": "v3"})
        return readings.model_copy(update={"floors": [status, other]})

    memo = _memo()
    over = FloorOverride(floor="billing/rate_drop/*", why="Billing refunds are rare this quarter.")
    recommended = memo.recommended()
    assert recommended is not None
    recommendation = recommended.model_copy(update={"arm": CANDIDATE_LABEL, "floors_overridden": [over]})
    page = await _page(
        "billing", "tone", memo=memo.model_copy(update={"recommendation": [recommendation]}), readings=both
    )
    header = [block for block in page.blocks if block.section == "header"]
    kinds = [
        "why" if isinstance(block, TextBlock) and block.role == "override" else block.text
        for block in header
        if isinstance(block, TextBlock)
        and block.role == "override"
        or isinstance(block, DisclosureBlock)
        and block.source == "floor"
    ]
    # v2's warning, its why, then v3's warning with none: the why answers v2 breaking the floor, not v3.
    assert len(kinds) == 3 and kinds[1] == "why"
    assert kinds[0].startswith(f"Floor broken on billing: {CANDIDATE_LABEL}'s") and kinds[2].startswith(
        "Floor broken on billing: v3's"
    )


async def test_a_fold_of_a_choice_names_the_arms_by_the_campaign_s_labels() -> None:
    tone = await _kind("tone")
    arms = [arm.model_copy(update={"label": arm.label.upper()}) for arm in tone.arms()]
    read = campaign_readings(tone.campaign, {"tone": tone.bundle}, arms=arms)
    assert read.arm_labels() == {tone.keys["alpha"]: "ALPHA", tone.keys["beta"]: "BETA"}
    report = build_code_only_report(
        tone.bundle, assembled_at=ASSEMBLED_AT, campaign_name="support", arm_labels=read.arm_labels()
    )
    page = build_campaign_report(tone.campaign, read, {"tone": report}, None)
    fold = campaign_report_markdown(page).split("\n## tone", 1)[1]
    assert "| ALPHA |" in fold and "| alpha |" not in fold


def _frontier(
    tone: MeasuredKind, decision: FrontierCostDecision, tied: bool
) -> Callable[[AnalysisContextBundle], AnalysisContextBundle]:
    """The tone bundle with its frontier reading alpha as clearing the bar, by ``decision``, beta beside it when tied."""

    def change(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
        ties = (
            [
                FrontierCostTie(
                    variant_key=tone.keys["beta"],
                    model="beta",
                    variant_identity_version=1,
                    production_replicating_cost=0.0005,
                )
            ]
            if tied
            else []
        )
        verdict = FrontierVerdict(
            variant_key=tone.keys["alpha"],
            model="alpha",
            pass_hat_k=0.95,
            variant_identity_version=1,
            cost_decision=decision,
            tied_with=ties,
        )
        (subject,) = bundle.frontier.subjects
        frontier = bundle.frontier.model_copy(update={"subjects": [subject.model_copy(update={"verdict": verdict})]})
        return bundle.model_copy(update={"frontier": frontier})

    return change


@pytest.mark.parametrize(
    ("decision", "tied", "said"),
    [
        (
            "not_separated",
            True,
            "Arms shown to clear the bar on tone: alpha and beta; none is shown cheaper than the rest.",
        ),
        (
            "untested",
            True,
            "Arms shown to clear the bar on tone: alpha and beta; too few priced cases to show any of them cheaper.",
        ),
        (
            "shown_cheapest",
            False,
            "Cheapest arm shown to clear the bar on tone: alpha, shown cheaper than every other that cleared.",
        ),
        ("only_cleared", False, "Only arm shown to clear the bar on tone at a known cost: alpha."),
    ],
)
async def test_the_frontier_line_says_what_the_frontier_read_and_names_every_tied_arm(
    decision: FrontierCostDecision, tied: bool, said: str
) -> None:
    tone = await _kind("tone")
    page = await _kind_page("tone", bundle=_frontier(tone, decision, tied))
    lines = [
        block.text for block in page.blocks if isinstance(block, DisclosureBlock) and "clear the bar" in block.text
    ]
    assert lines == [said]


async def test_a_change_whose_control_alone_ran_says_its_margin_and_floors_wait_for_a_candidate() -> None:
    campaign, (billing,) = await _campaign("billing")
    control = campaign.control
    assert control is not None
    read = campaign_readings(campaign, {"billing": billing.bundle}, arms=[control])
    assert read.kind == "measurement"
    report = build_code_only_report(billing.bundle, assembled_at=ASSEMBLED_AT, campaign_name="support")
    page = build_campaign_report(campaign, read, {"billing": report}, None)
    said = f"Only the control, {CONTROL_LABEL}, has run, so the margin on billing and 1 floor are read once a candidate runs."
    assert said in _first_screen_markdown(page)


async def test_a_measurement_states_its_pass_rate_as_a_count_and_each_rig_apart() -> None:
    def two_rigs(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
        moved = [cell.model_copy(update={"apparatus_class_id": "f" * 64}) for cell in bundle.cell_measures]
        return bundle.model_copy(update={"cell_measures": [*bundle.cell_measures, *moved]})

    one = await _kind_page("solo")
    table = next(block for block in one.blocks if isinstance(block, TableBlock) and block.name == "measures")
    assert "setup" not in {column.key for column in table.columns}
    assert table.rows[0]["value"].text == "45 of 60" and table.rows[0]["interval"].text.endswith("%]")
    page = await _kind_page("solo", bundle=two_rigs)
    table = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "measures")
    setups = [row["setup"].text for row in table.rows]
    assert setups[0] == "every setup" and len(set(setups[1:])) == 2
